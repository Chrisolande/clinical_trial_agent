import pytest
from tools.medication_safety import evaluate_medication_safety


def test_evaluate_medication_safety_safe_no_meds():
    patient = {"medications": []}
    trial = {"interventions": ["aspirin"]}
    result = evaluate_medication_safety(patient, trial)
    assert result["safe"] is True
    assert result["disqualify"] is False
    assert result["severity"] == "low"


def test_evaluate_medication_safety_safe_no_interventions():
    patient = {"medications": ["warfarin"]}
    trial = {"interventions": []}
    result = evaluate_medication_safety(patient, trial)
    assert result["safe"] is True
    assert result["disqualify"] is False


def test_evaluate_medication_safety_unsafe_interaction():
    patient = {"medications": ["warfarin"]}
    trial = {"interventions": ["aspirin"]}
    result = evaluate_medication_safety(patient, trial)
    assert result["safe"] is False
    assert result["disqualify"] is True
    assert result["severity"] == "critical"
    assert "Potentially unsafe interaction" in result["issues"][0]


def test_evaluate_medication_safety_unsafe_interaction_reversed():
    patient = {"medications": ["aspirin"]}
    trial = {"interventions": ["warfarin"]}
    result = evaluate_medication_safety(patient, trial)
    assert result["safe"] is False
    assert result["disqualify"] is True


def test_evaluate_medication_safety_normalized_names():
    patient = {"medications": ["  Warfarin  "]}
    trial = {"interventions": ["ASPIRIN"]}
    result = evaluate_medication_safety(patient, trial)
    assert result["safe"] is False


def test_evaluate_medication_safety_dict_entries():
    patient = {"medications": [{"name": "warfarin", "dose": "5mg"}]}
    trial = {"interventions": ["aspirin"]}
    result = evaluate_medication_safety(patient, trial)
    assert result["safe"] is False


def test_evaluate_medication_safety_malformed_dose():
    patient = {"medications": [{"name": "warfarin", "dose": "bad dose"}]}
    trial = {"interventions": ["aspirin"]}
    with pytest.raises(ValueError, match="malformed dose"):
        evaluate_medication_safety(patient, trial)


def test_evaluate_medication_safety_missing_name():
    patient = {"medications": [{"dose": "5mg"}]}
    trial = {"interventions": ["aspirin"]}
    import re

    with pytest.raises(ValueError, match=re.escape("medication.name is required")):
        evaluate_medication_safety(patient, trial)


def test_evaluate_medication_safety_empty_entry():
    patient = {"medications": [""]}
    trial = {"interventions": ["aspirin"]}
    with pytest.raises(ValueError, match="empty medication entry"):
        evaluate_medication_safety(patient, trial)


def test_evaluate_medication_safety_deduplication():
    # Deduplication is internal but we can verify it doesn't cause issues
    patient = {"medications": ["warfarin", "warfarin"]}
    trial = {"interventions": ["aspirin", "aspirin"]}
    result = evaluate_medication_safety(patient, trial)
    assert len(result["issues"]) == 1
