"""Minimal ClinicalTrials.gov proxy used for PHI-sensitive retrieval queries."""

from __future__ import annotations

import hmac
import os
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

app = FastAPI(title="ClinicalTrials.gov Proxy")

_ALLOWED_HOST = "clinicaltrials.gov"
_ALLOWED_PATH_PREFIX = "/api/v2"


class CtgovProxyRequest(BaseModel):
    endpoint: str
    params: dict[str, Any] = Field(default_factory=dict)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


def _validate_token(x_proxy_token: str | None) -> None:
    expected = os.getenv("CTGOV_PROXY_TOKEN", "").strip()
    if expected and not hmac.compare_digest(expected, (x_proxy_token or "").strip()):
        raise HTTPException(status_code=401, detail="Invalid proxy token")


def _validate_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "https"
        or parsed.hostname != _ALLOWED_HOST
        or not parsed.path.startswith(_ALLOWED_PATH_PREFIX)
    ):
        raise HTTPException(status_code=400, detail="Unsupported ClinicalTrials.gov endpoint")
    return endpoint


def _upstream_headers() -> dict[str, str]:
    return {
        "User-Agent": os.getenv("CTGOV_USER_AGENT", "clinical-trial-agent-proxy"),
        "Accept": os.getenv("CTGOV_ACCEPT", "application/json"),
    }


@app.post("/ctgov/search")
async def proxy_search(
    request: CtgovProxyRequest,
    x_proxy_token: str | None = Header(default=None),
) -> Response:
    _validate_token(x_proxy_token)
    endpoint = _validate_endpoint(request.endpoint)
    try:
        async with httpx.AsyncClient(timeout=30.0, headers=_upstream_headers()) as client:
            upstream = await client.get(endpoint, params=request.params)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="ClinicalTrials.gov upstream unavailable") from exc

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )
