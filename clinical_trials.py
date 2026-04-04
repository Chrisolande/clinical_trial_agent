"""ClinicalTrials.gov API v2 tools."""

import asyncio
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import httpx
from config import get_settings
from loguru import logger
from pydantic import BaseModel, Field
from tools.errors import ClinicalTrialsClientError

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


async def _request_json_with_retry(url: str, params: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    retries = max(1, settings.ctgov_retry_attempts)
    timeout = httpx.Timeout(30.0)
    backoff = max(1.0, settings.ctgov_retry_backoff_base)

    async with httpx.AsyncClient(timeout=timeout, headers=_CTGOV_HEADERS) as client:
        for attempt in range(retries):
            try:
                response = await client.get(url, params=params, timeout=timeout)
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
                try:
                    fallback_payload = await asyncio.to_thread(_urllib_get_json, url, params, 30.0)
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


def _get(obj: dict[str, Any], *keys: str) -> Any:
    current: Any = obj
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _split_eligibility_criteria(raw: str) -> tuple[str, str]:
    """Split CT.gov eligibilityCriteria blob into inclusion and exclusion text."""
    if not raw:
        return "", ""
    if "Exclusion Criteria:" in raw:
        inclusion_part, exclusion_part = raw.split("Exclusion Criteria:", 1)
        return (
            inclusion_part.replace("Inclusion Criteria:", "").strip(),
            exclusion_part.strip(),
        )
    return raw.replace("Inclusion Criteria:", "").strip(), ""


def parse_trial_from_response(study: dict[str, Any]) -> dict[str, Any]:
    """Extract a flat trial dict from a ClinicalTrials.gov v2 study object."""
    proto = study.get("protocolSection", {})

    def sec(name: str) -> dict[str, Any]:
        section = proto.get(name, {})
        return section if isinstance(section, dict) else {}

    id_mod = sec("identificationModule")
    status_mod = sec("statusModule")
    elig_mod = sec("eligibilityModule")
    design_mod = sec("designModule")
    outcomes_mod = sec("outcomesModule")
    contacts_mod = sec("contactsLocationsModule")
    arms_mod = sec("armsInterventionsModule")

    phases = design_mod.get("phases", [])

    locations = [
        {
            "facility": loc.get("facility"),
            "city": loc.get("city"),
            "state": loc.get("state"),
            "country": loc.get("country"),
            "status": loc.get("status"),
        }
        for loc in contacts_mod.get("locations", [])
    ]

    primary_outcomes = [
        {
            "measure": o.get("measure", ""),
            "time_frame": o.get("timeFrame"),
            "description": o.get("description"),
        }
        for o in outcomes_mod.get("primaryOutcomes", [])
    ]

    interventions = [name for arm in arms_mod.get("interventions", []) if (name := arm.get("name"))]

    completion_date_obj = status_mod.get("primaryCompletionDateStruct", {})
    completion_date = (
        completion_date_obj.get("date") if isinstance(completion_date_obj, dict) else None
    )

    raw_criteria = elig_mod.get("eligibilityCriteria")
    inclusion_text, exclusion_text = _split_eligibility_criteria(str(raw_criteria or ""))

    parsed = {
        "nct_id": id_mod.get("nctId", ""),
        "brief_title": id_mod.get("briefTitle", ""),
        "official_title": id_mod.get("officialTitle"),
        "overall_status": status_mod.get("overallStatus", ""),
        "phase": ", ".join(phases) if phases else None,
        "lead_sponsor": _get(sec("sponsorCollaboratorsModule"), "leadSponsor", "name"),
        "eligibility_criteria_raw": raw_criteria,
        "locations": locations,
        "primary_outcomes": primary_outcomes,
        "primary_completion_date": completion_date,
        "minimum_age": elig_mod.get("minimumAge"),
        "maximum_age": elig_mod.get("maximumAge"),
        "sex_eligibility": elig_mod.get("sex"),
        "healthy_volunteers": elig_mod.get("healthyVolunteers"),
        "brief_summary": _get(sec("descriptionModule"), "briefSummary"),
        "conditions": sec("conditionsModule").get("conditions", []),
        "interventions": interventions,
    }
    if isinstance(raw_criteria, str) and raw_criteria.strip():
        parsed["inclusion_criteria_parsed"] = inclusion_text
        parsed["exclusion_criteria_parsed"] = exclusion_text
    return parsed


def _extract_studies(data: dict[str, Any]) -> list[dict[str, Any]]:
    studies = data.get("studies", [])
    if isinstance(studies, list):
        return [s for s in studies if isinstance(s, dict)]
    return []


def _extract_single_study(data: dict[str, Any]) -> dict[str, Any]:
    if "protocolSection" in data and isinstance(data.get("protocolSection"), dict):
        return data
    studies = _extract_studies(data)
    return studies[0] if studies else {}


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
            _studies_endpoint_url(get_settings().ctgov_base_url), params
        )
        studies_raw = _extract_studies(data)
        studies = [parse_trial_from_response(study) for study in studies_raw]
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
