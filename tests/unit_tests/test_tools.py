import json
from collections.abc import Callable

import httpx
import pytest
from langchain_core.messages import ToolCall
from pydantic import SecretStr, ValidationError

from langchain_string import StringFetch, StringSearch, StringWebAccessToolkit

API_KEY = "test-key"

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch):
    monkeypatch.delenv("STRING_API_KEY", raising=False)


class Recorder:
    """Answers every request with one handler and keeps what was sent."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    @property
    def payload(self) -> dict:
        return json.loads(self.requests[-1].content)


def _clients(recorder: Recorder) -> dict:
    return {
        "http_client": httpx.Client(transport=httpx.MockTransport(recorder)),
        "http_async_client": httpx.AsyncClient(transport=httpx.MockTransport(recorder)),
    }


def search_tool(handler: Handler, **kwargs) -> tuple[StringSearch, Recorder]:
    recorder = Recorder(handler)
    tool = StringSearch(api_key=SecretStr(API_KEY), **_clients(recorder), **kwargs)
    return tool, recorder


def fetch_tool(handler: Handler, **kwargs) -> tuple[StringFetch, Recorder]:
    recorder = Recorder(handler)
    tool = StringFetch(api_key=SecretStr(API_KEY), **_clients(recorder), **kwargs)
    return tool, recorder


def results_body(*positions: int, **surfaces) -> Handler:
    body = {"results": [{"position": n} for n in positions], "zeroResults": False}
    return lambda _request: httpx.Response(200, json={**body, **surfaces})


def markdown(text: str, status: int = 200) -> Handler:
    return lambda _request: httpx.Response(
        200,
        text=text,
        headers={"content-type": "text/markdown", "x-status-code": str(status)},
    )


class TestStringSearch:
    def test_posts_the_query_with_the_key(self):
        tool, recorder = search_tool(results_body(1, 2))

        out = tool.invoke({"query": "running shoes"})

        request = recorder.requests[-1]
        assert str(request.url) == "https://request.usestring.ai/v1/search"
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        assert recorder.payload == {"query": "running shoes"}
        assert json.loads(out)["results"] == [{"position": 1}, {"position": 2}]

    def test_sends_engine_country_language_and_search_count(self):
        tool, recorder = search_tool(results_body(1, 2, 3), max_results=2)

        out = tool.invoke(
            {"query": "q", "engine": "brave", "country": "GB", "language": "en"}
        )

        assert recorder.payload == {
            "query": "q",
            "engine": "brave",
            "country": "GB",
            "language": "en",
            "searchCount": 2,
        }
        assert len(json.loads(out)["results"]) == 2

    def test_does_not_offer_bing(self):
        tool, recorder = search_tool(results_body(1))

        with pytest.raises(ValidationError):
            tool.invoke({"query": "q", "engine": "bing"})

        assert recorder.requests == []

    @pytest.mark.parametrize("max_results", [0, -1, 51])
    def test_rejects_a_result_count_the_api_cannot_serve(self, max_results):
        with pytest.raises(ValidationError):
            StringSearch(api_key=SecretStr(API_KEY), max_results=max_results)

    def test_keeps_the_surfaces_that_answer_a_query_without_results(self):
        entity = {"title": "Example Dental", "rating": 4.9}
        tool, _ = search_tool(results_body(entity=entity, places=[{"name": "Dr X"}]))

        out = json.loads(tool.invoke({"query": "dentist springfield"}))

        assert out["results"] == []
        assert out["entity"] == entity
        assert out["places"] == [{"name": "Dr X"}]

    def test_reports_no_results_when_nothing_came_back(self):
        tool, _ = search_tool(
            lambda _r: httpx.Response(200, json={"results": [], "zeroResults": True})
        )

        assert "No results found" in tool.invoke({"query": "nothing at all"})

    def test_keeps_non_ascii_text_readable(self):
        tool, _ = search_tool(
            lambda _r: httpx.Response(200, json={"results": [{"title": "東京 café"}]})
        )

        assert "東京 café" in tool.invoke({"query": "q"})

    async def test_async_path(self):
        tool, recorder = search_tool(results_body(1))

        out = await tool.ainvoke({"query": "q"})

        assert json.loads(out)["results"] == [{"position": 1}]
        assert recorder.payload == {"query": "q"}


class TestStringFetch:
    def test_requests_markdown(self):
        tool, recorder = fetch_tool(markdown("# Example Domain"))

        out = tool.invoke({"url": "https://example.com"})

        assert out == "# Example Domain"
        assert str(recorder.requests[-1].url) == "https://request.usestring.ai/v1/fetch"
        assert recorder.payload == {
            "url": "https://example.com",
            "format": "markdown",
            "markdownMode": "full",
            "mainContentOnly": False,
        }

    def test_passes_rendering_and_proxy_options_only_when_set(self):
        tool, recorder = fetch_tool(markdown("page"), main_content_only=True)

        tool.invoke(
            {"url": "https://example.com", "execute_js": True, "country_code": "GB"}
        )

        assert recorder.payload["executeJS"] is True
        assert recorder.payload["countryCode"] == "GB"
        assert recorder.payload["mainContentOnly"] is True

    def test_says_when_the_site_answered_with_an_error_status(self):
        tool, _ = fetch_tool(markdown("# Not Found", status=404))

        out = tool.invoke({"url": "https://example.com/gone"})

        assert out.startswith("The site answered HTTP 404")
        assert out.endswith("# Not Found")

    def test_truncates_and_says_so(self):
        tool, _ = fetch_tool(markdown("abcdefgh"), max_content_length=4)

        out = tool.invoke({"url": "https://example.com"})

        assert out.startswith("abcd\n\n")
        assert "4 of 8 characters" in out

    @pytest.mark.parametrize("length", [0, -3])
    def test_rejects_a_length_that_is_not_a_cap(self, length):
        with pytest.raises(ValidationError):
            StringFetch(api_key=SecretStr(API_KEY), max_content_length=length)

    async def test_async_path(self):
        tool, _ = fetch_tool(markdown("# Async"))

        assert await tool.ainvoke({"url": "https://example.com"}) == "# Async"


class TestErrors:
    def test_returns_the_api_explanation_to_the_model(self):
        body = {
            "error": "Validation error",
            "message": "Invalid enum value",
            "issues": [{"field": "country", "message": "Invalid country code"}],
        }
        tool, _ = search_tool(lambda _r: httpx.Response(400, json=body))

        message = tool.invoke(
            ToolCall(
                name="string_search", args={"query": "q"}, id="1", type="tool_call"
            )
        )

        assert message.status == "error"
        assert "HTTP 400" in message.content
        assert "country: Invalid country code" in message.content

    def test_names_the_destination_status_on_an_upstream_failure(self):
        body = {"error": "Request failed", "statusCode": 403, "message": "Upstream"}
        tool, _ = fetch_tool(lambda _r: httpx.Response(502, json=body))

        out = tool.invoke({"url": "https://example.com"})

        assert "HTTP 502" in out
        assert "destination answered HTTP 403" in out

    def test_reports_a_timeout_instead_of_raising(self):
        def timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        tool, _ = fetch_tool(timeout, timeout=5)

        assert "timed out after 5 seconds" in tool.invoke(
            {"url": "https://example.com"}
        )

    async def test_async_errors_are_reported_too(self):
        tool, _ = search_tool(
            lambda _r: httpx.Response(402, json={"error": "Insufficient balance"})
        )

        out = await tool.ainvoke({"query": "q"})

        assert "HTTP 402" in out
        assert "Insufficient balance" in out


class TestApiKey:
    def test_reads_the_environment(self, monkeypatch):
        monkeypatch.setenv("STRING_API_KEY", "from-env")

        assert StringSearch().api_key.get_secret_value() == "from-env"

    def test_explicit_key_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("STRING_API_KEY", "from-env")

        assert (
            StringSearch(api_key=SecretStr("explicit")).api_key.get_secret_value()
            == "explicit"
        )

    def test_missing_key_raises_before_any_request(self):
        recorder = Recorder(results_body(1))
        tool = StringSearch(**_clients(recorder))

        with pytest.raises(ValueError, match="STRING_API_KEY"):
            tool.invoke({"query": "q"})

        assert recorder.requests == []

    def test_the_key_is_not_in_the_repr(self):
        assert "super-secret" not in repr(
            StringSearch(api_key=SecretStr("super-secret"))
        )


class TestToolkit:
    def test_returns_both_tools(self):
        tools = StringWebAccessToolkit(api_key=SecretStr(API_KEY)).get_tools()

        assert [tool.name for tool in tools] == ["string_search", "string_fetch"]

    def test_passes_its_configuration_down(self):
        toolkit = StringWebAccessToolkit(
            api_key=SecretStr(API_KEY),
            base_url="http://localhost:8080/v1",
            timeout=5.0,
            max_results=3,
            markdown_mode="readable",
            main_content_only=True,
            max_content_length=100,
        )

        search, fetch = toolkit.get_tools()

        assert search.max_results == 3
        assert search.base_url == "http://localhost:8080/v1"
        assert search.timeout == 5.0
        assert fetch.markdown_mode == "readable"
        assert fetch.main_content_only is True
        assert fetch.max_content_length == 100

    def test_leaves_tools_to_read_the_environment_when_it_holds_no_key(
        self, monkeypatch
    ):
        monkeypatch.setenv("STRING_API_KEY", "from-env")

        search, _fetch = StringWebAccessToolkit().get_tools()

        assert search.api_key.get_secret_value() == "from-env"


class TestToolContract:
    @pytest.mark.parametrize("tool_class", [StringSearch, StringFetch])
    def test_names_descriptions_and_schemas_are_populated(self, tool_class):
        tool = tool_class()

        assert tool.name
        assert len(tool.description) > 40
        assert tool.args_schema is not None
        assert tool.get_input_schema().model_json_schema()["properties"]
