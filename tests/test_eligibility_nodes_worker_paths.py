from types import SimpleNamespace

import pytest
from subagents.eligibility import nodes


@pytest.mark.asyncio
async def test_identify_missing_info_errors_and_worker_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def missing_bad(_profile, _verdicts):
        raise RuntimeError("x")

    monkeypatch.setattr(nodes.missing_info, "identify_missing_info", missing_bad)
    out_bad = await nodes.identify_missing_info({"patient_profile": {}, "eligibility_verdicts": {}})
    assert out_bad["missing_info_recommendations"] == []

    monkeypatch.setattr(nodes, "get_settings", lambda: SimpleNamespace(min_match_tier="moderate"))
    sig = await nodes.assess_viability_signal({"viable_trial_count": 0})
    assert sig["retrieval_needs_broadening"] is True

    monkeypatch.setattr(
        nodes,
        "_cached_worker_result",
        lambda *_: {
            "processed_verdicts": [{"trial_id": "C"}],
            "processed_trials_with_criteria": [],
        },
    )
    cached = await nodes.evaluate_trial_worker({"trial": {"nct_id": "C"}, "patient_profile": {}})
    assert cached["processed_verdicts"][0]["trial_id"] == "C"

    monkeypatch.setattr(nodes, "_cached_worker_result", lambda *_: None)
    monkeypatch.setattr(
        nodes,
        "_irrelevant_worker_result",
        lambda *_: {
            "processed_verdicts": [{"trial_id": "D"}],
            "processed_trials_with_criteria": [],
        },
    )
    irr = await nodes.evaluate_trial_worker({"trial": {"nct_id": "D"}, "patient_profile": {}})
    assert irr["processed_verdicts"][0]["trial_id"] == "D"

    monkeypatch.setattr(nodes, "_irrelevant_worker_result", lambda *_: None)

    async def parse_empty(_trial, _nct):
        return {
            "trial": {"nct_id": "E"},
            "inclusion_criteria": [],
            "exclusion_criteria": [],
        }

    monkeypatch.setattr(nodes, "_parse_trial_criteria", parse_empty)
    empty = await nodes.evaluate_trial_worker({"trial": {"nct_id": "E"}, "patient_profile": {}})
    assert empty["processed_verdicts"][0]["trial_id"] == "E"

    async def parse_nonempty(_trial, _nct):
        return {
            "trial": {"nct_id": "F"},
            "inclusion_criteria": [{"text": "A"}],
            "exclusion_criteria": [],
        }

    async def eval_ok(**_kwargs):
        return {"trial_id": "F", "match_tier": "moderate", "match_score": 0.6}

    called = {"cache": 0}

    def set_cache(**_kwargs):
        called["cache"] += 1

    monkeypatch.setattr(nodes, "_parse_trial_criteria", parse_nonempty)
    monkeypatch.setattr(nodes, "_evaluate_with_optional_semaphore", eval_ok)
    monkeypatch.setattr(nodes, "set_cached_eligibility_verdict", set_cache)

    full = await nodes.evaluate_trial_worker({"trial": {"nct_id": "F"}, "patient_profile": {}})
    assert full["processed_verdicts"][0]["trial_id"] == "F"
    assert called["cache"] == 1
