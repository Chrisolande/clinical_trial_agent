from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_CONDITION_SIGNALS = frozenset(
    {
        "cancer",
        "tumor",
        "tumour",
        "carcinoma",
        "sarcoma",
        "lymphoma",
        "leukemia",
        "leukaemia",
        "melanoma",
        "adenocarcinoma",
        "disease",
        "syndrome",
        "disorder",
        "diabetes",
        "hypertension",
        "fibrosis",
        "sclerosis",
        "arthritis",
        "infection",
        "hepatitis",
        "nsclc",
        "sclc",
        "glioblastoma",
        "myeloma",
        "diagnosis",
        "diagnosed",
        "condition",
        "illness",
        "pathology",
    }
)

_CONTEXT_SIGNALS = frozenset(
    {
        "year",
        "age",
        "old",
        "patient",
        "female",
        "male",
        "woman",
        "man",
        "ecog",
        "stage",
        "metastatic",
        "advanced",
        "recurrent",
        "relapsed",
        "treatment",
        "therapy",
        "chemo",
        "radiation",
        "surgery",
        "biopsy",
        "mutation",
        "biomarker",
        "egfr",
        "her2",
        "brca",
        "pdl1",
        "kras",
        "prior",
        "previous",
        "history",
        "lab",
        "creatinine",
        "hemoglobin",
    }
)

_AGE_RE = re.compile(r"\b(age|\d{2}\s*(year|yr|y\.o))\b", re.IGNORECASE)
_STAGE_RE = re.compile(r"\b(stage|metastatic|advanced|localized|early)\b", re.IGNORECASE)


@dataclass
class ValidationResult:
    is_valid: bool
    missing_required: list[str] = field(default_factory=list)
    missing_helpful: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "missing_required": self.missing_required,
            "missing_helpful": self.missing_helpful,
            "message": self.message,
        }


def _has_signal(text: str, signals: frozenset[str]) -> bool:
    # Extracts all standalone words from the text and checks for intersections
    words = set(re.findall(r"\b\w+\b", text))
    return not signals.isdisjoint(words)


def validate_patient_profile(profile_raw: str) -> ValidationResult:
    if not profile_raw or not profile_raw.strip():
        return ValidationResult(
            is_valid=False,
            missing_required=["medical_condition", "patient_context"],
            message=(
                "Patient profile is empty. Please provide at minimum: "
                "medical condition and basic patient context."
            ),
        )

    text = profile_raw.strip().lower()

    missing_required = [
        key
        for key, failed in (
            ("medical_condition", not _has_signal(text, _CONDITION_SIGNALS)),
            ("patient_context", not _has_signal(text, _CONTEXT_SIGNALS)),
            ("sufficient_detail", len(text.split()) < 10),
        )
        if failed
    ]

    missing_helpful = [
        key
        for key, failed in (
            ("patient_age", not _AGE_RE.search(text)),
            ("disease_stage", not _STAGE_RE.search(text)),
        )
        if failed
    ]

    if missing_required:
        items = ", ".join(missing_required)
        return ValidationResult(
            is_valid=False,
            missing_required=missing_required,
            missing_helpful=missing_helpful,
            message=(
                f"Profile lacks required information: {items}. "
                "Running the full pipeline would produce only UNCERTAIN verdicts."
            ),
        )

    return ValidationResult(
        is_valid=True,
        missing_helpful=missing_helpful,
        message="Profile contains sufficient information to proceed.",
    )
