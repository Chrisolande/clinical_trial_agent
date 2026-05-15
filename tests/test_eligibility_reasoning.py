from typing import Any

import pytest
from agents import eligibility_reasoner
from agents.eligibility_reasoner import evaluate_criteria_batch
from models.judge_verdict import JudgeVerdict
from subagents.eligibility.nodes import _is_plausibly_relevant

from clinical_trial_agent.config import get_settings
from clinical_trial_agent.constants import ELIGIBILITY_TIMEOUT_FALLBACK_MESSAGE


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class DummyResp:
    def __init__(self, content: str) -> None:
        self.content = content


class DummyLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, *_: Any, **__: Any) -> DummyResp:
        return DummyResp(self._content)


@pytest.mark.asyncio
async def test_malformed_judge_response_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLINICAL_DATA_EXTERNAL_LLM_CONSENT", "true")
    monkeypatch.setenv("LLM_PRIVACY_MODE", "full_consent")
    get_settings.cache_clear()
    monkeypatch.setattr("agents.eligibility_reasoner.get_llm", lambda **_: DummyLLM("no tags here"))
    result = await evaluate_criteria_batch(
        patient_profile={"age": 50},
        trial={"nct_id": "NCTX", "brief_title": "Trial"},
        all_criteria=[{"criteria_type": "inclusion", "text": "Age >= 18"}],
    )
    assert result["match_tier"] == "weak"
    assert result["match_score"] == 0.1
    assert result["key_concern"] == "LLM response parsing failed"


@pytest.mark.asyncio
async def test_eligibility_timeout_returns_weak_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_judge(*_: object, **__: object) -> object:
        raise TimeoutError()

    monkeypatch.setattr(eligibility_reasoner, "_judge_trial", fake_judge)
    result = await eligibility_reasoner.evaluate_criteria_batch(
        patient_profile={"age": 62},
        trial={"nct_id": "NCT05983432", "brief_title": "Demo Trial"},
        all_criteria=[{"criteria_type": "inclusion", "text": "Age >= 18"}],
    )
    assert result["trial_id"] == "NCT05983432"
    assert result["match_tier"] == "weak"
    assert result["key_concern"] == ELIGIBILITY_TIMEOUT_FALLBACK_MESSAGE


def test_relevance_matches_abbreviation_to_full_term() -> None:
    trial = {
        "brief_title": "A study in colorectal cancer",
        "conditions": ["metastatic colorectal adenocarcinoma"],
    }
    assert _is_plausibly_relevant(trial, "CRC")


def test_relevance_matches_full_term_to_abbreviation() -> None:
    trial = {
        "brief_title": "NSCLC targeted therapy",
        "conditions": ["NSCLC"],
    }
    assert _is_plausibly_relevant(trial, "non small cell lung cancer")


def test_relevance_rejects_unrelated_conditions() -> None:
    trial = {"brief_title": "AML trial", "conditions": ["acute myeloid leukemia"]}
    assert not _is_plausibly_relevant(trial, "colorectal cancer")


@pytest.mark.asyncio
async def test_unverified_partial_criteria_cannot_produce_strong_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_judge(*_: object, **__: object) -> JudgeVerdict:
        return JudgeVerdict(
            match_score=0.91,
            match_tier="strong",
            major_criteria_assessable=True,
            inclusion_met=[
                "histology confirmed synthetic NSCLC",
                "EGFR biomarker present",
                "ECOG performance status 0-1",
                "measurable disease by RECIST",
            ],
            inclusion_failed=[],
            inclusion_uncertain=[],
            exclusion_triggered=[],
            exclusion_uncertain=[],
            critical_missing_info=[],
            key_concern="",
            rationale="Synthetic strong response from model",
        )

    monkeypatch.setattr(eligibility_reasoner, "_judge_trial", fake_judge)
    result = await eligibility_reasoner.evaluate_criteria_batch(
        patient_profile={"age": 62},
        trial={
            "nct_id": "NCTSNIPPET",
            "brief_title": "Demo Trial",
            "criteria_source": "tavily_snippet",
            "criteria_source_verified": False,
            "criteria_completeness": "partial",
        },
        all_criteria=[
            {"criteria_type": "inclusion", "text": "histology confirmed synthetic NSCLC"},
            {"criteria_type": "inclusion", "text": "EGFR biomarker present"},
            {"criteria_type": "inclusion", "text": "ECOG performance status 0-1"},
            {"criteria_type": "inclusion", "text": "measurable disease by RECIST"},
        ],
    )

    assert result["match_tier"] == "moderate"
    assert result["match_score"] <= 0.65
    assert result["criteria_source"] == "tavily_snippet"
