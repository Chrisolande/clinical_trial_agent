from __future__ import annotations

import pytest
from agents import missing_info


@pytest.mark.asyncio
async def test_missing_info_uses_fallback_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_invoke(patient_profile: dict[str, object], uncertain_summary: str) -> object:
        _ = (patient_profile, uncertain_summary)
        raise TimeoutError()

    monkeypatch.setattr(missing_info, "_invoke_missing_info_llm", fake_invoke)
    eligibility_verdicts = {
        "NCT1": {
            "verdicts": [
                {"verdict": "UNCERTAIN", "criterion_text": "EGFR mutation status required"}
            ]
        },
        "NCT2": {
            "verdicts": [
                {"verdict": "UNCERTAIN", "criterion_text": "EGFR mutation status required"}
            ]
        },
    }
    result = await missing_info.identify_missing_info({"age": 62}, eligibility_verdicts)
    assert len(result) == 1
    assert result[0]["field"] == "EGFR mutation status required"
    assert result[0]["priority"] == "medium"
    assert result[0]["affected_trial_ids"] == ["NCT1", "NCT2"]
