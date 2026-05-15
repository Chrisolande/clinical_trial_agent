"""ClinicalTrials.gov payload parsing helpers."""

from datetime import UTC, datetime
from typing import Any


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
    raw_criteria = elig_mod.get("eligibilityCriteria")
    inclusion_text, exclusion_text = _split_eligibility_criteria(str(raw_criteria or ""))
    has_criteria = isinstance(raw_criteria, str) and bool(raw_criteria.strip())

    parsed = {
        "nct_id": id_mod.get("nctId", ""),
        "brief_title": id_mod.get("briefTitle", ""),
        "official_title": id_mod.get("officialTitle"),
        "overall_status": status_mod.get("overallStatus", ""),
        "phase": ", ".join(phases) if phases else None,
        "lead_sponsor": _get(sec("sponsorCollaboratorsModule"), "leadSponsor", "name"),
        "eligibility_criteria_raw": raw_criteria,
        "criteria_source": "ctgov_api" if has_criteria else "missing",
        "criteria_source_verified": has_criteria,
        "criteria_retrieved_at": datetime.now(UTC).isoformat(),
        "criteria_completeness": "full" if has_criteria else "missing",
        "locations": _parse_locations(contacts_mod),
        "primary_outcomes": _parse_primary_outcomes(outcomes_mod),
        "primary_completion_date": _primary_completion_date(status_mod),
        "minimum_age": elig_mod.get("minimumAge"),
        "maximum_age": elig_mod.get("maximumAge"),
        "sex_eligibility": elig_mod.get("sex"),
        "healthy_volunteers": elig_mod.get("healthyVolunteers"),
        "brief_summary": _get(sec("descriptionModule"), "briefSummary"),
        "conditions": sec("conditionsModule").get("conditions", []),
        "interventions": _parse_interventions(arms_mod),
    }
    if has_criteria:
        parsed["inclusion_criteria_parsed"] = inclusion_text
        parsed["exclusion_criteria_parsed"] = exclusion_text
    return parsed


def _parse_locations(contacts_mod: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "facility": loc.get("facility"),
            "city": loc.get("city"),
            "state": loc.get("state"),
            "country": loc.get("country"),
            "status": loc.get("status"),
        }
        for loc in contacts_mod.get("locations", [])
    ]


def _parse_primary_outcomes(outcomes_mod: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "measure": outcome.get("measure", ""),
            "time_frame": outcome.get("timeFrame"),
            "description": outcome.get("description"),
        }
        for outcome in outcomes_mod.get("primaryOutcomes", [])
    ]


def _parse_interventions(arms_mod: dict[str, Any]) -> list[str]:
    return [name for arm in arms_mod.get("interventions", []) if (name := arm.get("name"))]


def _primary_completion_date(status_mod: dict[str, Any]) -> Any:
    completion_date_obj = status_mod.get("primaryCompletionDateStruct", {})
    return completion_date_obj.get("date") if isinstance(completion_date_obj, dict) else None


def extract_studies(data: dict[str, Any]) -> list[dict[str, Any]]:
    studies = data.get("studies", [])
    if isinstance(studies, list):
        return [s for s in studies if isinstance(s, dict)]
    return []


def extract_single_study(data: dict[str, Any]) -> dict[str, Any]:
    if "protocolSection" in data and isinstance(data.get("protocolSection"), dict):
        return data
    studies = extract_studies(data)
    return studies[0] if studies else {}
