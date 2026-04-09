from typing import Any

import pytest
from agents import eligibility_reasoner
from agents.eligibility_reasoner import evaluate_criteria_batch
from subagents.eligibility.nodes import _is_plausibly_relevant

from constants import ELIGIBILITY_TIMEOUT_FALLBACK_MESSAGE


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
    monkeypatch.setattr("agents.eligibility_reasoner.get_llm", lambda: DummyLLM("no tags here"))
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
