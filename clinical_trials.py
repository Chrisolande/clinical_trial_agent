"""ClinicalTrials.gov API v2 tools."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import get_settings
from loguru import logger
from pydantic import BaseModel, Field

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


def _studies_endpoint_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/studies") else f"{normalized}/studies"


def _urllib_get_json(url: str, params: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    query = urlencode(params, doseq=True)
    full_url = f"{url}?{query}" if query else url
    req = Request(full_url, headers=_CTGOV_HEADERS)
    with urlopen(req, timeout=timeout) as resp:  # nosec B310 — URL is a hardcoded CT.gov endpoint, not user input
        raw = json.loads(resp.read())
    return raw if isinstance(raw, dict) else {}


async def _ctgov_request_with_retry(url: str, params: dict[str, Any]) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(get_settings().ctgov_retry_attempts):
        try:
            return await asyncio.to_thread(_urllib_get_json, url, params, 30.0)
        except Exception as exc:
            last_exc = exc
            logger.warning("CT.gov request attempt {} failed: {}", attempt + 1, exc)
            if attempt < get_settings().ctgov_retry_attempts - 1:
                await asyncio.sleep(get_settings().ctgov_retry_backoff_base ** attempt)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("ClinicalTrials.gov request failed without a captured exception")


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
        data = await _ctgov_request_with_retry(
            _studies_endpoint_url(get_settings().ctgov_base_url), params
        )
        studies_raw = _extract_studies(data)
        studies = [parse_trial_from_response(study) for study in studies_raw]
        return cast("dict[str, Any]", SearchTrialsOutput(studies=studies).model_dump())
    except Exception as exc:
        return cast(
            "dict[str, Any]",
            SearchTrialsOutput(
                error=ToolError(
                    type="request_error",
                    message=f"ClinicalTrials.gov request error: {exc}",
                    retryable=True,
                ),
            ).model_dump(),
        )


async def fetch_trial_detail(nct_id: str) -> dict[str, Any]:
    """Fetch a single trial by NCT ID."""
    url = f"{_studies_endpoint_url(get_settings().ctgov_base_url)}/{nct_id}"
    try:
        data = await _ctgov_request_with_retry(url, {"format": "json"})
        study = _extract_single_study(data)
        return cast(
            "dict[str, Any]",
            FetchTrialDetailOutput(trial=parse_trial_from_response(study)).model_dump(),
        )
    except Exception as exc:
        logger.warning("Error fetching {}: {}", nct_id, exc)
        return cast(
            "dict[str, Any]",
            FetchTrialDetailOutput(
                error=ToolError(
                    type="request_error",
                    message=f"ClinicalTrials.gov detail request error: {exc}",
                    retryable=True,
                ),
            ).model_dump(),
        )
