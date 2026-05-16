import pytest
from agents import missing_info
from agents.missing_info_catalog import fallback_missing_info_recommendations


def test_build_uncertain_summary_dedupes_fields_and_limits_examples() -> None:
    verdicts = {
        "T1": {"verdicts": [{"verdict": "UNCERTAIN", "criterion_text": "EGFR mutation status"}]},
        "T2": {"verdicts": [{"verdict": "UNCERTAIN", "criterion_text": "EGFR mutation status"}]},
    }
    by_field, summary = missing_info._build_uncertain_summary(verdicts)
    assert by_field
    assert "egfr" in next(iter(by_field.keys()))
    assert "T1" in summary and "T2" in summary


def test_format_profile_summary_redacts_selected_fields() -> None:
    text = missing_info._format_profile_summary(
        {
            "age": 60,
            "sex": "female",
            "primary_condition": "NSCLC",
            "conditions": ["NSCLC"],
            "medications": ["drug"],
            "free_text": "hello world",
        }
    )
    assert "age: [redacted]" in text
    assert "free_text: hello world" in text


def test_enrich_with_trial_context_fills_defaults_and_priority() -> None:
    uncertain_by_field = {
        "egfr_mutation_status": {
            "field_id": "egfr_mutation_status",
            "display_name": "EGFR mutation status",
            "category": "biomarker",
            "why_needed": "EGFR needed",
            "affected_trial_ids": ["T1", "T2", "T3"],
        }
    }
    enriched = missing_info._enrich_with_trial_context(
        items=[{"field_id": "egfr_mutation_status", "affected_trial_ids": ["T1"]}],
        uncertain_by_field=uncertain_by_field,
    )
    assert enriched
    assert enriched[0]["field_id"] == "egfr_mutation_status"
    assert enriched[0]["display_name"]
    assert enriched[0]["priority"] in {"high", "medium", "low"}


@pytest.mark.asyncio
async def test_identify_missing_info_falls_back_when_llm_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLINICAL_DATA_EXTERNAL_LLM_CONSENT", "true")
    monkeypatch.setenv("LLM_PRIVACY_MODE", "full_consent")

    async def fake_invoke(_profile: dict, _summary: str):
        # Simulate an empty structured result.
        return missing_info.CompletenessAssessmentList(results=[])

    monkeypatch.setattr(missing_info, "_invoke_missing_info_llm", fake_invoke)
    verdicts = {
        "T1": {"verdicts": [{"verdict": "UNCERTAIN", "criterion_text": "EGFR mutation status"}]}
    }
    result = await missing_info.identify_missing_info({"age": 60}, verdicts)
    assert result == fallback_missing_info_recommendations(
        missing_info._build_uncertain_summary(verdicts)[0]
    )
