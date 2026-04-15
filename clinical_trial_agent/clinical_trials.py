"""ClinicalTrials.gov API v2 tools."""

import asyncio
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import httpx
from loguru import logger
from pydantic import BaseModel, Field
from tools.ctgov_parsing import (
    extract_single_study as _extract_single_study,
)
from tools.ctgov_parsing import (
    extract_studies as _extract_studies,
)
from tools.ctgov_parsing import (
    parse_trial_from_response,
)
from tools.errors import ClinicalTrialsClientError

from clinical_trial_agent.config import get_settings

_CTGOV_HEADERS: dict[str, str] = {
    "User-Agent": get_settings().ctgov_user_agent,
    "Accept": get_settings().ctgov_accept,
}


class ToolError(BaseModel):
    type: str
    message: str
    status_code: int | None = None
    retryable: bool = False


class SearchTrialsInput(BaseModel):
    condition: str | None = None
    intervention: str | None = None
    status: list[str] | None = None
    page_size: int = Field(default=20, ge=1, le=1000)


class SearchTrialsOutput(BaseModel):
    studies: list[dict[str, Any]] = Field(default_factory=list)
    error: ToolError | None = None


class FetchTrialDetailOutput(BaseModel):
    trial: dict[str, Any] | None = None
    error: ToolError | None = None


def _urllib_get_json(url: str, params: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    query = urlencode(params, doseq=True)
    full_url = f"{url}?{query}" if query else url
    req = Request(full_url, headers=_CTGOV_HEADERS)
    with urlopen(req, timeout=timeout) as response:  # nosec B310
        payload = response.read()
    parsed = httpx.Response(200, content=payload).json()
    return parsed if isinstance(parsed, dict) else {}


def _studies_endpoint_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/studies") else f"{normalized}/studies"


def _contains_phi_params(params: dict[str, Any]) -> bool:
    return bool(params.get("query.cond") or params.get("query.intr"))


async def _request_with_transport(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any],
    request_timeout: httpx.Timeout,
) -> httpx.Response:
    settings = get_settings()
    has_phi = _contains_phi_params(params)
    proxy_url = settings.ctgov_proxy_url

    if has_phi and proxy_url:
        payload = {"endpoint": url, "params": params}
        return await client.post(proxy_url, json=payload, timeout=request_timeout)

    if has_phi and not proxy_url:
        # Keep PHI out of URL query parameters when no tokenizing proxy is configured.
        return await client.post(url, json=params, timeout=request_timeout)

    if settings.ctgov_transport_mode == "post":
        return await client.post(url, json=params, timeout=request_timeout)
    return await client.get(url, params=params, timeout=request_timeout)


async def _request_json_with_retry(url: str, params: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    retries = max(1, settings.ctgov_retry_attempts)
    timeout = httpx.Timeout(30.0)
    backoff = max(1.0, settings.ctgov_retry_backoff_base)

    async with httpx.AsyncClient(timeout=timeout, headers=_CTGOV_HEADERS) as client:
        for attempt in range(retries):
            try:
                response = await _request_with_transport(client, url, params, timeout)
            except httpx.RequestError as exc:
                logger.warning("CT.gov request error on attempt {}: {}", attempt + 1, exc)
                if attempt < retries - 1:
                    await asyncio.sleep(backoff**attempt)
                    continue
                raise ClinicalTrialsClientError(
                    "ClinicalTrials.gov request failed due to connection or DNS error",
                    retryable=True,
                ) from exc

            status = response.status_code
            if status == 429 or 500 <= status <= 599:
                logger.warning("CT.gov transient status {} on attempt {}", status, attempt + 1)
                if attempt < retries - 1:
                    await asyncio.sleep(backoff**attempt)
                    continue
                raise ClinicalTrialsClientError(
                    f"ClinicalTrials.gov transient failure after retries (status={status})",
                    status_code=status,
                    retryable=True,
                )

            if status == 403:
                allow_urllib_fallback = not _contains_phi_params(params)
                if allow_urllib_fallback:
                    try:
                        fallback_payload = await asyncio.to_thread(
                            _urllib_get_json, url, params, 30.0
                        )
                        if isinstance(fallback_payload, dict):
                            return fallback_payload
                    except Exception as exc:
                        logger.warning("CT.gov urllib fallback failed after 403: {}", exc)
                raise ClinicalTrialsClientError(
                    "ClinicalTrials.gov request rejected (status=403)",
                    status_code=403,
                    retryable=False,
                )

            if 400 <= status <= 499:
                raise ClinicalTrialsClientError(
                    f"ClinicalTrials.gov request rejected (status={status})",
                    status_code=status,
                    retryable=False,
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise ClinicalTrialsClientError(
                    "ClinicalTrials.gov returned invalid JSON payload",
                    status_code=status,
                    retryable=False,
                ) from exc

            if not isinstance(payload, dict):
                raise ClinicalTrialsClientError(
                    "ClinicalTrials.gov payload is not an object",
                    status_code=status,
                    retryable=False,
                )
            return payload

    raise ClinicalTrialsClientError(
        "ClinicalTrials.gov request failed unexpectedly", retryable=True
    )


async def search_trials(
    condition: str | None = None,
    intervention: str | None = None,
    status: list[str] | None = None,
    page_size: int = 20,
) -> dict[str, Any]:
    """Search ClinicalTrials.gov API v2."""
    validated = SearchTrialsInput(
        condition=condition,
        intervention=intervention,
        status=status,
        page_size=page_size,
    )
    params: dict[str, Any] = {"format": "json", "pageSize": validated.page_size}
    if validated.condition:
        params["query.cond"] = validated.condition
    if validated.intervention:
        params["query.intr"] = validated.intervention
    if validated.status:
        params["filter.overallStatus"] = "|".join(validated.status)

    try:
        data = await _request_json_with_retry(
            _studies_endpoint_url(get_settings().ctgov_base_url),
            params,
        )
        studies = [parse_trial_from_response(s) for s in _extract_studies(data)]
        return SearchTrialsOutput(studies=studies).model_dump()
    except ClinicalTrialsClientError as exc:
        return SearchTrialsOutput(
            error=ToolError(
                type="request_error",
                message=str(exc),
                status_code=exc.status_code,
                retryable=exc.retryable,
            )
        ).model_dump()


async def fetch_trial_detail(nct_id: str) -> dict[str, Any]:
    """Fetch a single trial by NCT ID."""
    nct = nct_id.strip()
    if not nct:
        return FetchTrialDetailOutput(
            error=ToolError(
                type="validation_error",
                message="NCT ID must be non-empty",
                retryable=False,
            )
        ).model_dump()

    url = f"{_studies_endpoint_url(get_settings().ctgov_base_url)}/{nct}"
    try:
        data = await _request_json_with_retry(url, {"format": "json"})
        study = _extract_single_study(data)
        if not study:
            raise ClinicalTrialsClientError(
                f"ClinicalTrials.gov returned no study for NCT ID {nct}", retryable=False
            )
        trial = parse_trial_from_response(study)
        if not str(trial.get("nct_id", "")).strip():
            raise ClinicalTrialsClientError(
                "ClinicalTrials.gov study payload missing non-empty nct_id", retryable=False
            )
        return FetchTrialDetailOutput(trial=trial).model_dump()
    except ClinicalTrialsClientError as exc:
        logger.warning("Error fetching {}: {}", nct, exc)
        return FetchTrialDetailOutput(
            error=ToolError(
                type="request_error",
                message=str(exc),
                status_code=exc.status_code,
                retryable=exc.retryable,
            )
        ).model_dump()
