import re
from typing import Any

_DOSE_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*(mg|mcg|g|ml)\s*$", re.IGNORECASE)

# Minimal deterministic interaction matrix for high-risk combinations.
_INTERACTION_MATRIX: set[tuple[str, str]] = {
    ("warfarin", "aspirin"),
    ("warfarin", "ibuprofen"),
    ("methotrexate", "trimethoprim"),
    ("linezolid", "sertraline"),
}


def _normalize_medications(patient_profile: dict[str, Any]) -> list[str]:
    meds: list[str] = []
    for med in patient_profile.get("medications", []) or []:
        if isinstance(med, dict):
            name = str(med.get("name", "")).strip().lower()
            if not name:
                raise ValueError("Medication safety validation failed: medication.name is required")
            dose = med.get("dose")
            if dose is not None and str(dose).strip() and not _DOSE_RE.match(str(dose)):
                raise ValueError(
                    f"Medication safety validation failed: malformed dose '{dose}' for {name}"
                )
            meds.append(name)
        else:
            name = str(med).strip().lower()
            if not name:
                raise ValueError("Medication safety validation failed: empty medication entry")
            meds.append(name)
    return list(dict.fromkeys(meds))


def _normalize_interventions(trial: dict[str, Any]) -> list[str]:
    values = [str(v).strip().lower() for v in trial.get("interventions", []) if str(v).strip()]
    return list(dict.fromkeys(values))


def evaluate_medication_safety(
    patient_profile: dict[str, Any], trial: dict[str, Any]
) -> dict[str, Any]:
    medications = _normalize_medications(patient_profile)
    interventions = _normalize_interventions(trial)

    if not medications or not interventions:
        return {"safe": True, "disqualify": False, "severity": "low", "issues": []}

    issues: list[str] = []
    for med in medications:
        for intr in interventions:
            if (med, intr) in _INTERACTION_MATRIX or (intr, med) in _INTERACTION_MATRIX:
                issues.append(
                    f"Potentially unsafe interaction: medication={med} intervention={intr}"
                )

    if issues:
        return {
            "safe": False,
            "disqualify": True,
            "severity": "critical",
            "issues": issues,
        }

    return {"safe": True, "disqualify": False, "severity": "low", "issues": []}
