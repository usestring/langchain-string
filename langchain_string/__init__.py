"""String Web Access tools for LangChain."""

from importlib import metadata

from langchain_string.toolkits import StringWebAccessToolkit
from langchain_string.tools import StringFetch, StringSearch

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Running from a source checkout that was never installed.
    __version__ = ""
del metadata

__all__ = [
    "StringFetch",
    "StringSearch",
    "StringWebAccessToolkit",
    "__version__",
]
