# langchain-string

[![PyPI version](https://img.shields.io/pypi/v/langchain-string)](https://pypi.org/project/langchain-string/)
[![Python versions](https://img.shields.io/pypi/pyversions/langchain-string)](https://pypi.org/project/langchain-string/)
[![Downloads](https://static.pepy.tech/badge/langchain-string/month)](https://pepy.tech/project/langchain-string)
[![License: MIT](https://img.shields.io/pypi/l/langchain-string)](LICENSE)

String Web Access tools for [LangChain](https://docs.langchain.com).

Search the web and fetch any URL as clean, LLM-ready Markdown.

## Installation

```bash
pip install -U langchain-string
# or
uv add langchain-string
```

## Credentials

Get an API key at [usestring.ai](https://usestring.ai) and set it:

```bash
export STRING_API_KEY=...
```

Or pass it per tool: `StringSearch(api_key=SecretStr("..."))`.

## Tools

| Tool | What it does |
| --- | --- |
| `StringSearch` | Search Google, DuckDuckGo, Brave or Mojeek and return the results as JSON, with Google's knowledge panel, local listings and AI overviews when the page carried them. |
| `StringFetch` | Fetch any URL and return the page as Markdown. |

Both implement `_run` and `_arun`, so they work in sync and async agents. API and network failures
come back to the model as the tool's output, with the API's reason attached, rather than ending the
run; pass `handle_tool_error=False` to raise them instead.

### StringSearch

```python
from langchain_string import StringSearch

search = StringSearch(max_results=5)
print(search.invoke({"query": "python httpx library"}))
```

### StringFetch

```python
from langchain_string import StringFetch

fetch = StringFetch(main_content_only=True)
print(fetch.invoke({"url": "https://example.com"}))
```

Pass `execute_js=True` when a JavaScript-rendered site comes back empty, and `country_code="GB"` to
route the request through a specific country. When the site answers with an error status, such as a
404, the output says so above the page it returned.

### Invoke with a ToolCall

Invoked with a model-generated `ToolCall`, either tool returns a `ToolMessage`:

```python
fetch.invoke(
    {
        "args": {"url": "https://example.com"},
        "id": "1",
        "name": fetch.name,
        "type": "tool_call",
    }
)
```

## Use within an agent

`StringWebAccessToolkit` returns both tools and passes its configuration to each.

```python
# pip install -U "langchain[anthropic]" to run the agent and call the model
from langchain.agents import create_agent
from langchain_string import StringWebAccessToolkit

agent = create_agent(
    model="anthropic:claude-sonnet-5",
    tools=StringWebAccessToolkit(max_results=5).get_tools(),
    system_prompt=(
        "You research questions using the live web. Search first, then read the pages "
        "worth reading, and cite every URL you used."
    ),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Find the latest Python release and summarize what changed.",
            }
        ]
    }
)
print(result["messages"][-1].content)
```

## Configuration

Shared by both tools:

| Field | Default | Description |
| --- | --- | --- |
| `api_key` | `STRING_API_KEY` | Your String API key, held as a `SecretStr`. |
| `base_url` | `https://request.usestring.ai/v1` | Point this elsewhere only for a different String deployment. |
| `timeout` | `600.0` | HTTP timeout in seconds. A browser-rendered fetch can take minutes. |
| `http_client` / `http_async_client` | — | Your own `httpx` clients, for proxies or custom transports. |

`StringSearch` adds `max_results`, 1 to 50. Unset, a search returns one results page, about ten
results. Google is paged until it has `max_results`, and each page is billed as one search.

`StringFetch` adds `markdown_mode` (`full` or `readable`), `main_content_only` and
`max_content_length`.

## Development

```bash
uv sync
make test                              # offline; network sockets are blocked
STRING_API_KEY=... make integration_test   # live, billed to that key
make lint
```

### Releasing

Bump `version` in `pyproject.toml` on `main`. `.github/workflows/publish.yml` builds the wheel and
sdist and publishes them to PyPI through trusted publishing, which needs a pending publisher
configured on PyPI first.

## Docs

- LangChain guide: https://portal.usestring.ai/docs/guides/langchain
- REST API: https://portal.usestring.ai/docs/api-reference/overview
- MCP server: https://portal.usestring.ai/docs/mcp/overview

## License

MIT
