"""Minimal PHI-safe proxy for ClinicalTrials.gov retrieval."""

from __future__ import annotations

import asyncio
import json
import os
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

APP_NAME = "ctgov-proxy"
CTGOV_HOST = "clinicaltrials.gov"
CTGOV_PATH_PREFIX = "/api/v2/"

app = FastAPI(title=APP_NAME)


class ProxyRequest(BaseModel):
    endpoint: str = Field(description="ClinicalTrials.gov endpoint URL")
    params: dict[str, object] = Field(default_factory=dict)


def _expected_token() -> str | None:
    token = os.getenv("CTGOV_PROXY_TOKEN", "").strip()
    return token or None


def _validate_endpoint(endpoint: str) -> None:
    parsed = urlparse(endpoint)
    if parsed.scheme != "https":
        raise HTTPException(status_code=400, detail="endpoint must use https")
    if parsed.netloc != CTGOV_HOST:
        raise HTTPException(status_code=400, detail="endpoint host is not allowed")
    if not parsed.path.startswith(CTGOV_PATH_PREFIX):
        raise HTTPException(status_code=400, detail="endpoint path is not allowed")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ctgov/search")
async def ctgov_search(
    body: ProxyRequest,
    x_proxy_token: str = Header(default="", alias="X-Proxy-Token"),
) -> dict[str, object]:
    expected = _expected_token()
    if expected is not None and x_proxy_token != expected:
        raise HTTPException(status_code=401, detail="unauthorized")

    _validate_endpoint(body.endpoint)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }

    def _fetch() -> dict[str, object]:
        query = urlencode(body.params, doseq=True)
        full_url = f"{body.endpoint}?{query}" if query else body.endpoint
        request = Request(full_url, headers=headers)
        with urlopen(request, timeout=30.0) as response:  # nosec B310
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}

    try:
        payload = await asyncio.to_thread(_fetch)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="upstream payload is not valid JSON") from exc
    except Exception as exc:
        status = getattr(exc, "code", None)
        if isinstance(status, int):
            raise HTTPException(status_code=status, detail=f"upstream status={status}") from exc
        raise HTTPException(status_code=502, detail=f"upstream request error: {exc}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="upstream payload is not an object")
    return payload
