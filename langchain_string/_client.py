"""Shared HTTP plumbing for the String Web Access API."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.tools import ToolException

DEFAULT_BASE_URL = "https://request.usestring.ai/v1"
# The API holds a connection open for about ten minutes while it renders a page or
# solves a CAPTCHA, and bills the fetch whether or not the caller is still waiting.
DEFAULT_TIMEOUT = 600.0

_BODY_EXCERPT_LIMIT = 2000


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def error_message(response: httpx.Response) -> str:
    """Turn a String API error response into text a model can act on."""
    parts = [f"String API returned HTTP {response.status_code}"]
    try:
        body = response.json()
    except ValueError:
        body = None

    if isinstance(body, dict):
        detail = body.get("message") or body.get("reason") or body.get("error")
        if detail:
            parts.append(str(detail))
        for issue in body.get("issues") or []:
            if isinstance(issue, dict):
                parts.append(f"{issue.get('field')}: {issue.get('message')}")
        if body.get("statusCode") is not None:
            parts.append(f"the destination answered HTTP {body['statusCode']}")
        if body.get("errorType"):
            parts.append(f"{body['errorType']}, retrying will not help")
        if isinstance(body.get("data"), str):
            excerpt = body["data"][:_BODY_EXCERPT_LIMIT]
            parts.append(f"the destination's response: {excerpt}")
    elif response.text:
        parts.append(response.text[:_BODY_EXCERPT_LIMIT])

    return ". ".join(parts) + "."


def _checked(response: httpx.Response) -> httpx.Response:
    if response.is_error:
        raise ToolException(error_message(response))
    return response


def _transport_error(exc: httpx.RequestError, timeout: float) -> ToolException:
    if isinstance(exc, httpx.TimeoutException):
        return ToolException(f"String API request timed out after {timeout:g} seconds.")
    return ToolException(f"Could not reach the String API: {exc}")


def post(
    client: httpx.Client,
    base_url: str,
    path: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> httpx.Response:
    try:
        response = client.post(
            _url(base_url, path),
            headers=_headers(api_key),
            json=payload,
            timeout=timeout,
        )
    except httpx.RequestError as exc:
        raise _transport_error(exc, timeout) from exc
    return _checked(response)


async def apost(
    client: httpx.AsyncClient,
    base_url: str,
    path: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> httpx.Response:
    try:
        response = await client.post(
            _url(base_url, path),
            headers=_headers(api_key),
            json=payload,
            timeout=timeout,
        )
    except httpx.RequestError as exc:
        raise _transport_error(exc, timeout) from exc
    return _checked(response)
