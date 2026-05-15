import pytest
from pydantic import ValidationError
from tools.profile_validator import (
    PatientProfile,
    validate_patient_profile,
    validate_structured_profile,
)


def test_validate_structured_profile_success():
    profile = {
        "age": 45,
        "sex": "female",
        "conditions": ["NSCLC"],
        "biomarkers": ["EGFR+"],
        "medications": ["pembrolizumab"],
        "prior_treatments": ["chemotherapy"],
    }
    result = validate_structured_profile(profile)
    assert isinstance(result, PatientProfile)
    assert result.age == 45


def test_validate_structured_profile_invalid():
    profile = {"age": -1}  # Invalid age
    with pytest.raises(ValidationError):  # Pydantic ValidationError
        validate_structured_profile(profile)


def test_validate_patient_profile_valid():
    text = "45 year old female patient with advanced NSCLC and EGFR mutation. Prior chemotherapy."
    result = validate_patient_profile(text)
    assert result.is_valid is True
    assert result.missing_required == []


def test_validate_patient_profile_empty():
    result = validate_patient_profile("")
    assert result.is_valid is False
    assert "empty" in result.message


def test_validate_patient_profile_missing_condition():
    text = "45 year old female patient. Prior chemotherapy."  # No condition signal
    result = validate_patient_profile(text)
    assert result.is_valid is False
    assert "medical_condition" in result.missing_required


def test_validate_patient_profile_missing_context():
    text = "NSCLC diagnosis."  # Too short, no context
    result = validate_patient_profile(text)
    assert result.is_valid is False
    assert "patient_context" in result.missing_required


def test_validate_patient_profile_missing_helpful():
    text = "Female patient with cancer diagnosis. This is more than ten words to pass detail check."
    result = validate_patient_profile(text)
    assert result.is_valid is True
    assert "patient_age" in result.missing_helpful
    assert "disease_stage" in result.missing_helpful


def test_validation_result_to_dict():
    text = "NSCLC diagnosis."
    result = validate_patient_profile(text)
    d = result.to_dict()
    assert d["is_valid"] is False
    assert "missing_required" in d
