"""LangChain's standard integration suite, run against the live String API.

Every test is skipped unless ``STRING_API_KEY`` is set. Each run is billed to that
key: one search and one fetch per test.
"""

import os
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_string import StringFetch, StringSearch

pytestmark = pytest.mark.skipif(
    not os.environ.get("STRING_API_KEY"), reason="STRING_API_KEY is not set"
)

# A live failure should fail the test rather than come back as the tool's output.
RAISE_ERRORS = {"handle_tool_error": False}


class TestStringSearchIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return StringSearch

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {**RAISE_ERRORS, "max_results": 3}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "python httpx library"}


class TestStringFetchIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return StringFetch

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return RAISE_ERRORS

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"url": "https://example.com"}


def test_search_returns_results():
    out = StringSearch(**RAISE_ERRORS, max_results=3).invoke(
        {"query": "python httpx library"}
    )

    assert "https://" in out


def test_fetch_returns_the_page_as_markdown():
    out = StringFetch(**RAISE_ERRORS).invoke({"url": "https://example.com"})

    assert "Example Domain" in out
