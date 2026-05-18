from __future__ import annotations

import pytest
from pydantic import ValidationError
from tools import medication_safety, profile_validator


def test_validate_structured_profile_success() -> None:
    profile = profile_validator.validate_structured_profile(
        {
            "age": 56,
            "sex": "female",
            "conditions": ["nsclc"],
            "biomarkers": ["egfr"],
            "medications": ["aspirin"],
            "prior_treatments": ["chemo"],
        }
    )
    assert profile.age == 56
    assert profile.conditions == ["nsclc"]


def test_validate_structured_profile_invalid() -> None:
    with pytest.raises(ValidationError):
        profile_validator.validate_structured_profile(
            {
                "age": 160,
                "sex": "female",
                "conditions": [],
            }
        )


def test_validation_result_to_dict() -> None:
    result = profile_validator.ValidationResult(
        is_valid=False,
        missing_required=["medical_condition"],
        missing_helpful=["patient_age"],
        message="missing",
    )
    assert result.to_dict() == {
        "is_valid": False,
        "missing_required": ["medical_condition"],
        "missing_helpful": ["patient_age"],
        "message": "missing",
    }


def test_has_signal_detects_word_intersection() -> None:
    assert profile_validator._has_signal("patient with nsclc", frozenset({"nsclc"}))
    assert not profile_validator._has_signal("plain text", frozenset({"nsclc"}))


def test_validate_patient_profile_empty() -> None:
    result = profile_validator.validate_patient_profile("   ")
    assert result.is_valid is False
    assert result.missing_required == ["medical_condition", "patient_context"]


def test_validate_patient_profile_missing_required_and_helpful() -> None:
    result = profile_validator.validate_patient_profile("patient old with history")
    assert result.is_valid is False
    assert "medical_condition" in result.missing_required
    assert "sufficient_detail" in result.missing_required
    assert "patient_age" in result.missing_helpful
    assert "disease_stage" in result.missing_helpful


def test_validate_patient_profile_valid_with_helpful_gaps() -> None:
    text = (
        "female patient with nsclc diagnosis history prior treatment ecog mutation biomarker data"
    )
    result = profile_validator.validate_patient_profile(text)
    assert result.is_valid is True
    assert "patient_age" in result.missing_helpful
    assert "disease_stage" in result.missing_helpful


def test_validate_patient_profile_fully_valid() -> None:
    text = (
        "62 year old female patient diagnosed with metastatic nsclc "
        "with egfr mutation and prior treatment history"
    )
    result = profile_validator.validate_patient_profile(text)
    assert result.is_valid is True
    assert result.missing_helpful == []


def test_normalize_medication_dict_entry_and_errors() -> None:
    assert (
        medication_safety._normalize_medication_dict_entry({"name": " Warfarin ", "dose": "5 mg"})
        == "warfarin"
    )

    with pytest.raises(ValueError, match=r"medication\.name is required"):
        medication_safety._normalize_medication_dict_entry({"dose": "5 mg"})

    with pytest.raises(ValueError, match="malformed dose"):
        medication_safety._normalize_medication_dict_entry({"name": "warfarin", "dose": "five"})


def test_normalize_medication_entry_variants() -> None:
    assert medication_safety._normalize_medication_entry(" Aspirin ") == "aspirin"
    assert medication_safety._normalize_medication_entry({"name": "Ibuprofen"}) == "ibuprofen"
    with pytest.raises(ValueError, match="empty medication entry"):
        medication_safety._normalize_medication_entry("")


def test_normalize_medications_and_interventions_deduplicate() -> None:
    meds = medication_safety._normalize_medications(
        {
            "medications": [
                "Aspirin",
                "aspirin",
                {"name": "Warfarin"},
                {"name": "warfarin"},
            ]
        }
    )
    interventions = medication_safety._normalize_interventions(
        {"interventions": ["  ibuprofen ", "", 1, "ibuprofen", "linezolid"]}
    )
    assert meds == ["aspirin", "warfarin"]
    assert interventions == ["ibuprofen", "linezolid"]


def test_evaluate_medication_safety_empty_inputs() -> None:
    result = medication_safety.evaluate_medication_safety(
        {"medications": []}, {"interventions": []}
    )
    assert result == {
        "safe": True,
        "disqualify": False,
        "severity": "low",
        "issues": [],
    }


def test_evaluate_medication_safety_detects_interaction() -> None:
    result = medication_safety.evaluate_medication_safety(
        {"medications": ["Warfarin"]}, {"interventions": ["aspirin"]}
    )
    assert result["safe"] is False
    assert result["disqualify"] is True
    assert result["severity"] == "critical"
    assert "warfarin" in result["issues"][0]


def test_evaluate_medication_safety_no_interaction() -> None:
    result = medication_safety.evaluate_medication_safety(
        {"medications": ["metformin"]}, {"interventions": ["linezolid"]}
    )
    assert result == {
        "safe": True,
        "disqualify": False,
        "severity": "low",
        "issues": [],
    }
