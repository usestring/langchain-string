"""A toolkit bundling every String Web Access tool."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import Field, SecretStr

from langchain_string._client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT
from langchain_string.tools import SEARCH_COUNT_MAX, StringFetch, StringSearch


class StringWebAccessToolkit(BaseToolkit):
    """Search the web and read any page, as one toolkit.

    Setup:
        Install `langchain-string` and set the `STRING_API_KEY` environment variable.

        ```bash
        pip install -U langchain-string
        export STRING_API_KEY="your-api-key"
        ```

    Key init args:
        max_results: int | None
            Organic results per search, passed to `StringSearch`.
        markdown_mode: Literal["full", "readable"]
            Markdown detail, passed to `StringFetch`.
        main_content_only: bool
            Strip page chrome, passed to `StringFetch`.
        max_content_length: int | None
            Truncate each page, passed to `StringFetch`.

    Instantiation:
        ```python
        from langchain_string import StringWebAccessToolkit

        toolkit = StringWebAccessToolkit(max_results=5)
        ```

    Tools:
        ```python
        toolkit.get_tools()
        ```

        ```python
        [StringSearch(...), StringFetch(...)]
        ```

    Use within an agent:
        ```python
        from langchain.agents import create_agent

        agent = create_agent(
            model="anthropic:claude-sonnet-5",
            tools=toolkit.get_tools(),
        )
        agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Find the latest Python release and summarize what "
                            "changed."
                        ),
                    }
                ]
            }
        )
        ```
    """

    api_key: SecretStr | None = None
    """String API key. Each tool falls back to ``STRING_API_KEY`` when this is unset."""

    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT

    max_results: int | None = Field(default=None, ge=1, le=SEARCH_COUNT_MAX)
    """Passed to :class:`~langchain_string.tools.StringSearch`."""

    markdown_mode: Literal["full", "readable"] = "full"
    """Passed to :class:`~langchain_string.tools.StringFetch`."""

    main_content_only: bool = False
    """Passed to :class:`~langchain_string.tools.StringFetch`."""

    max_content_length: int | None = Field(default=None, ge=1)
    """Passed to :class:`~langchain_string.tools.StringFetch`."""

    def get_tools(self) -> list[BaseTool]:
        shared: dict[str, Any] = {"base_url": self.base_url, "timeout": self.timeout}
        if self.api_key is not None:
            shared["api_key"] = self.api_key

        return [
            StringSearch(max_results=self.max_results, **shared),
            StringFetch(
                markdown_mode=self.markdown_mode,
                main_content_only=self.main_content_only,
                max_content_length=self.max_content_length,
                **shared,
            ),
        ]
