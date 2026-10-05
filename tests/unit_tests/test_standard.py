"""LangChain's own standard test suite, as the contributing guide asks for.

https://docs.langchain.com/oss/python/contributing/standard-tests-langchain
"""

from typing import Any

import pytest
from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_string import StringFetch, StringSearch


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("STRING_API_KEY", "test-key")


class TestStringSearchUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return StringSearch

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "python httpx library"}


class TestStringFetchUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return StringFetch

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"url": "https://example.com"}
