from __future__ import annotations

import pytest
from agents import eligibility_reasoner


@pytest.mark.asyncio
async def test_eligibility_timeout_returns_weak_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert result["key_concern"] == "Eligibility judge timeout"
