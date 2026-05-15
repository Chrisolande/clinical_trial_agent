from typing import Any

import pytest
from agents.eligibility_prompt_builder import build_judge_messages, format_patient_summary

from clinical_trial_agent.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _profile() -> dict[str, Any]:
    return {
        "name": "Synthetic Person",
        "age": 54,
        "primary_condition": "synthetic NSCLC",
        "biomarkers": ["EGFR exon 19 deletion"],
        "email": "synthetic@example.invalid",
        "ecog_performance_status": 1,
    }


def _trial() -> dict[str, str]:
    return {"nct_id": "NCT00000001", "brief_title": "Synthetic Trial"}


def _criteria() -> list[dict[str, str]]:
    return [{"criteria_type": "inclusion", "text": "Age >= 18 and EGFR mutation"}]


def _set_privacy_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    provider: str,
    mode: str,
    consent: bool = False,
) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_URI", "postgresql://x")
    monkeypatch.setenv("MEMORY_DB_DSN", "postgresql://x")
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_PRIVACY_MODE", mode)
    monkeypatch.setenv("CLINICAL_DATA_EXTERNAL_LLM_CONSENT", str(consent).lower())


def _message_text() -> str:
    return "\n".join(
        str(message.content) for message in build_judge_messages(_profile(), _trial(), _criteria())
    )


def test_external_provider_blocked_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_privacy_env(monkeypatch, provider="openai", mode="blocked")
    with pytest.raises(RuntimeError, match="blocked forbids external"):
        build_judge_messages(_profile(), _trial(), _criteria())


def test_external_provider_deidentified_uses_age_range_no_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_privacy_env(monkeypatch, provider="openai", mode="deidentified")
    text = _message_text()
    assert "age_range: 50-59" in text
    assert "Synthetic Person" not in text
    assert "synthetic@example.invalid" not in text
    assert "EGFR exon 19 deletion" in text


def test_external_provider_full_consent_false_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_privacy_env(monkeypatch, provider="openai", mode="full_consent", consent=False)
    with pytest.raises(RuntimeError, match="CLINICAL_DATA_EXTERNAL_LLM_CONSENT=true"):
        build_judge_messages(_profile(), _trial(), _criteria())


def test_external_provider_full_consent_true_sends_clinical_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_privacy_env(monkeypatch, provider="openai", mode="full_consent", consent=True)
    text = _message_text()
    assert "synthetic NSCLC" in text
    assert "EGFR exon 19 deletion" in text
    assert "Synthetic Person" not in text


def test_local_provider_local_only_sends_clinical_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_privacy_env(monkeypatch, provider="ollama", mode="local_only", consent=False)
    text = _message_text()
    assert "synthetic NSCLC" in text
    assert "EGFR exon 19 deletion" in text


def test_local_provider_not_redacted_by_external_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_privacy_env(monkeypatch, provider="ollama", mode="full_consent", consent=False)
    summary = format_patient_summary(_profile())
    assert "synthetic NSCLC" in summary
    assert "EGFR exon 19 deletion" in summary
