"""ClinicalTrials.gov API v2 tools."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import settings
from loguru import logger
from pydantic import BaseModel, Field

_CTGOV_HEADERS: dict[str, str] = {
    "User-Agent": settings.ctgov_user_agent,
    "Accept": settings.ctgov_accept,
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
    with urlopen(req, timeout=timeout) as resp:  # nosec B310
        raw = json.loads(resp.read())
    return raw if isinstance(raw, dict) else {}


async def _ctgov_request_with_retry(url: str, params: dict[str, Any]) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(settings.ctgov_retry_attempts):
        try:
            return await asyncio.to_thread(_urllib_get_json, url, params, 30.0)
        except Exception as exc:
            last_exc = exc
            logger.warning("CT.gov request attempt {} failed: {}", attempt + 1, exc)
            if attempt < settings.ctgov_retry_attempts - 1:
                await asyncio.sleep(settings.ctgov_retry_backoff_base**attempt)
    raise last_exc  # type: ignore[misc]


def _get(obj: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)  # type: ignore[assignment]
    return obj


def parse_trial_from_response(study: dict[str, Any]) -> dict[str, Any]:
    """Extract a flat trial dict from a ClinicalTrials.gov v2 study object."""
    proto = study.get("protocolSection", {})

    def sec(name: str) -> dict[str, Any]:
        return proto.get(name, {})

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

    return {
        "nct_id": id_mod.get("nctId", ""),
        "brief_title": id_mod.get("briefTitle", ""),
        "official_title": id_mod.get("officialTitle"),
        "overall_status": status_mod.get("overallStatus", ""),
        "phase": ", ".join(phases) if phases else None,
        "lead_sponsor": _get(sec("sponsorCollaboratorsModule"), "leadSponsor", "name"),
        "eligibility_criteria_raw": elig_mod.get("eligibilityCriteria"),
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
        data = await _ctgov_request_with_retry(settings.ctgov_base_url, params)
        return SearchTrialsOutput(studies=data.get("studies", [])).model_dump()
    except Exception as exc:
        return SearchTrialsOutput(
            error=ToolError(
                type="request_error",
                message=f"ClinicalTrials.gov request error: {exc}",
                retryable=True,
            ),
        ).model_dump()


async def fetch_trial_detail(nct_id: str) -> dict[str, Any]:
    """Fetch a single trial by NCT ID."""
    url = f"{settings.ctgov_base_url}/{nct_id}"
    try:
        data = await _ctgov_request_with_retry(url, {"format": "json"})
        return FetchTrialDetailOutput(trial=parse_trial_from_response(data)).model_dump()
    except Exception as exc:
        logger.warning("Error fetching {}: {}", nct_id, exc)
        return FetchTrialDetailOutput(
            error=ToolError(
                type="request_error",
                message=f"ClinicalTrials.gov detail request error: {exc}",
                retryable=True,
            ),
        ).model_dump()
