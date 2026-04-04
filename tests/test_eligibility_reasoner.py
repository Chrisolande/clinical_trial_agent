from __future__ import annotations

from typing import Any

import pytest
from agents.eligibility_reasoner import evaluate_criteria_batch


class DummyResp:
    def __init__(self, content: str) -> None:
        self.content = content


class DummyLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, *_: Any, **__: Any) -> DummyResp:
        return DummyResp(self._content)


@pytest.mark.asyncio
async def test_malformed_judge_response_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
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
