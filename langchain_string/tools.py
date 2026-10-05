"""LangChain tools over the String Web Access API."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, Field, PrivateAttr, SecretStr

from langchain_string._client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT, apost, post

Engine = Literal["google", "duckduckgo", "brave", "mojeek"]

SEARCH_COUNT_MAX = 50


class _StringToolBase(BaseTool):
    """Configuration every String tool shares."""

    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("STRING_API_KEY", default=None)
    )
    """String API key. Reads ``STRING_API_KEY`` when unset.

    Get one at https://usestring.ai.
    """

    base_url: str = DEFAULT_BASE_URL
    """API base URL. Change it only to point at a different String deployment."""

    timeout: float = DEFAULT_TIMEOUT
    """HTTP timeout in seconds.

    A browser-rendered fetch of a slow or CAPTCHA-gated page can take minutes, and it
    is billed even when the client gives up first.
    """

    http_client: httpx.Client | None = Field(default=None, exclude=True)
    """Client for sync calls, for proxies or a custom transport.

    One is created on first use when unset.
    """

    http_async_client: httpx.AsyncClient | None = Field(default=None, exclude=True)
    """Client for async calls. A new one is opened per call when unset."""

    handle_tool_error: bool | str | Callable[[ToolException], str] | None = True
    """Return API and network failures to the model as the tool's output.

    The message carries the API's own explanation, such as a rejected field or the
    status the destination answered with, so the model can change its call.
    """

    _default_client: httpx.Client | None = PrivateAttr(default=None)

    def _key(self) -> str:
        if self.api_key is None:
            msg = (
                "No String API key. Set STRING_API_KEY, or pass api_key=. "
                "Get a key at https://usestring.ai."
            )
            raise ValueError(msg)
        return self.api_key.get_secret_value()

    def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        key = self._key()
        client = self.http_client
        if client is None:
            if self._default_client is None:
                self._default_client = httpx.Client()
            client = self._default_client
        return post(client, self.base_url, path, payload, key, self.timeout)

    async def _apost(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        key = self._key()
        if self.http_async_client is not None:
            return await apost(
                self.http_async_client, self.base_url, path, payload, key, self.timeout
            )
        async with httpx.AsyncClient() as client:
            return await apost(client, self.base_url, path, payload, key, self.timeout)


class StringSearchInput(BaseModel):
    """Input for :class:`StringSearch`."""

    query: str = Field(
        description=(
            "The search query to run. Quoted phrases and site: pass through to the "
            "engine."
        )
    )
    engine: Engine | None = Field(
        default=None,
        description=(
            "Search engine to query. Google when unset, and only Google returns "
            "the panels, local listings and overviews beside the organic results."
        ),
    )
    country: str | None = Field(
        default=None,
        description=(
            "ISO 3166-1 alpha-2 country code used to localize the results. "
            "US when unset."
        ),
    )
    language: str | None = Field(
        default=None,
        description="Language tag for the results, such as 'en' or 'pt-br'.",
    )


class StringSearch(_StringToolBase):
    """Search the web through String Web Access.

    Setup:
        Install `langchain-string` and set the `STRING_API_KEY` environment variable.

        ```bash
        pip install -U langchain-string
        export STRING_API_KEY="your-api-key"
        ```

    Instantiation:
        ```python
        from langchain_string import StringSearch

        tool = StringSearch(max_results=5)
        ```

    Invocation with args:
        ```python
        tool.invoke({"query": "python httpx library"})
        ```

        ```python
        '{"results": [{"position": 1, "title": "...", "url": "...", ...}, ...], ...}'
        ```

    Invocation with ToolCall:
        ```python
        tool.invoke(
            {
                "args": {"query": "python httpx library"},
                "id": "1",
                "name": tool.name,
                "type": "tool_call",
            }
        )
        ```

        ```python
        ToolMessage(
            content='{"results": [{"position": 1, ...}, ...], ...}',
            name="string_search",
            tool_call_id="1",
        )
        ```
    """

    name: str = "string_search"
    description: str = (
        "Search the web and get the results back as JSON. `results` holds the ranked "
        "pages, each with position, title, url, displayUrl and snippet. A Google "
        "search can also return what the results page showed around them: a "
        "knowledge panel (`entity`), local businesses (`places`), AI overviews "
        "(`overviews`), People also ask and answer widgets. Read those before "
        "treating an empty `results` as no answer. A snippet is not the page: use "
        "string_fetch on a result's url to read it."
    )
    args_schema: type[BaseModel] = StringSearchInput

    max_results: int | None = Field(default=None, ge=1, le=SEARCH_COUNT_MAX)
    """Organic results to return, 1 to 50.

    Unset returns one results page, about ten results. Google is paged until it has
    this many, and each page is billed as one search. Other engines return their one
    page, capped to this many.
    """

    def _payload(
        self,
        query: str,
        engine: Engine | None,
        country: str | None,
        language: str | None,
    ) -> dict[str, Any]:
        fields = {
            "engine": engine,
            "country": country,
            "language": language,
            "searchCount": self.max_results,
        }
        return {"query": query, **{k: v for k, v in fields.items() if v is not None}}

    def _format(self, body: dict[str, Any], query: str) -> str:
        results = body.get("results") or []
        if self.max_results is not None:
            results = results[: self.max_results]
        surfaces = {
            key: value
            for key, value in body.items()
            if key not in ("results", "zeroResults") and value
        }
        if not results and not surfaces:
            return (
                f"No results found for '{query}'. The engine reported no match, so "
                "broaden the query rather than retrying it."
            )
        return json.dumps(
            {"results": results, "zeroResults": body.get("zeroResults"), **surfaces},
            ensure_ascii=False,
        )

    def _run(
        self,
        query: str,
        engine: Engine | None = None,
        country: str | None = None,
        language: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        payload = self._payload(query, engine, country, language)
        return self._format(self._post("/search", payload).json(), query)

    async def _arun(
        self,
        query: str,
        engine: Engine | None = None,
        country: str | None = None,
        language: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        payload = self._payload(query, engine, country, language)
        response = await self._apost("/search", payload)
        return self._format(response.json(), query)


class StringFetchInput(BaseModel):
    """Input for :class:`StringFetch`."""

    url: str = Field(description="The http/https URL to fetch.")
    execute_js: bool = Field(
        default=False,
        description=(
            "Render the page in a browser before capturing it. Use when a fetch of a "
            "JavaScript-rendered site comes back empty."
        ),
    )
    country_code: str | None = Field(
        default=None,
        description=(
            "ISO 3166-1 alpha-2 country to route the request through, e.g. 'GB'."
        ),
    )


class StringFetch(_StringToolBase):
    r"""Fetch a URL through String Web Access and return it as clean Markdown.

    Setup:
        Install `langchain-string` and set the `STRING_API_KEY` environment variable.

        ```bash
        pip install -U langchain-string
        export STRING_API_KEY="your-api-key"
        ```

    Instantiation:
        ```python
        from langchain_string import StringFetch

        tool = StringFetch(main_content_only=True)
        ```

    Invocation with args:
        ```python
        tool.invoke({"url": "https://example.com"})
        ```

        ```python
        "# Example Domain\n\n..."
        ```

    Invocation with ToolCall:
        ```python
        tool.invoke(
            {
                "args": {"url": "https://example.com"},
                "id": "1",
                "name": tool.name,
                "type": "tool_call",
            }
        )
        ```

        ```python
        ToolMessage(
            content="# Example Domain\n\n...",
            name="string_fetch",
            tool_call_id="1",
        )
        ```
    """

    name: str = "string_fetch"
    description: str = (
        "Fetch any URL and get the page back as clean, LLM-ready Markdown. Set "
        "execute_js when a JavaScript-rendered page comes back empty, and "
        "country_code to fetch it from a specific country. When the site answers "
        "with an error status, the output says so above the page it returned."
    )
    args_schema: type[BaseModel] = StringFetchInput

    markdown_mode: Literal["full", "readable"] = "full"
    """How much of the page structure to preserve in the Markdown."""

    main_content_only: bool = False
    """Strip navigation, footers and other page chrome."""

    max_content_length: int | None = Field(default=None, ge=1)
    """Truncate the returned page to this many characters.

    ``None`` returns all of it. A truncated page ends with a note giving its full
    length.
    """

    def _payload(
        self, url: str, execute_js: bool, country_code: str | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "url": url,
            "format": "markdown",
            "markdownMode": self.markdown_mode,
            "mainContentOnly": self.main_content_only,
        }
        if execute_js:
            payload["executeJS"] = True
        if country_code is not None:
            payload["countryCode"] = country_code
        return payload

    def _truncate(self, content: str) -> str:
        limit = self.max_content_length
        if limit is None or len(content) <= limit:
            return content
        note = f"[Truncated to {limit} of {len(content)} characters.]"
        return f"{content[:limit]}\n\n{note}"

    def _format(self, response: httpx.Response) -> str:
        content = self._truncate(response.text)
        status = response.headers.get("x-status-code", "")
        if status.isdigit() and int(status) >= 400:
            return (
                f"The site answered HTTP {status}, so this is likely its error page "
                f"rather than the content asked for.\n\n{content}"
            )
        return content

    def _run(
        self,
        url: str,
        execute_js: bool = False,
        country_code: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        payload = self._payload(url, execute_js, country_code)
        return self._format(self._post("/fetch", payload))

    async def _arun(
        self,
        url: str,
        execute_js: bool = False,
        country_code: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        payload = self._payload(url, execute_js, country_code)
        return self._format(await self._apost("/fetch", payload))
