from __future__ import annotations

from types import SimpleNamespace

import pytest
from subagents.retrieval import nodes


def test_get_new_unique_trials_deduplicates_and_ignores_missing_id() -> None:
    fetched = [{"nct_id": "A"}, {"nct_id": "A"}, {"x": 1}, {"nct_id": "B"}]
    result = nodes._get_new_unique_trials(fetched, {"B"})
    assert result == [{"nct_id": "A"}]


@pytest.mark.asyncio
async def test_initialize_retrieval_initial_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nodes.trial_search,
        "build_search_queries",
        lambda terms, profile, include_not_yet_recruiting=False: [
            f"q:{terms.get('primary_search_terms', [])}:{include_not_yet_recruiting}"
        ],
    )
    out = await nodes.initialize_retrieval(
        {
            "retry_count": 0,
            "normalized_terms": {"primary_search_terms": ["x"]},
            "patient_profile": {},
            "trials_raw": [{"nct_id": "N1"}],
        }
    )
    assert out["existing_nct_ids"] == ["N1"]
    assert out["include_not_yet_recruiting"] is False


@pytest.mark.asyncio
async def test_initialize_retrieval_retry_uses_refiner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nodes.search_refiner,
        "refine_search_strategy",
        lambda **_: {
            "refined_terms": {"primary_search_terms": ["refined"]},
            "include_not_yet_recruiting": True,
            "decision_note": "refined",
        },
    )
    monkeypatch.setattr(
        nodes.trial_search,
        "build_search_queries",
        lambda terms, profile, include_not_yet_recruiting=False: [
            f"{terms['primary_search_terms'][0]}:{include_not_yet_recruiting}"
        ],
    )
    out = await nodes.initialize_retrieval(
        {
            "retry_count": 1,
            "normalized_terms": {},
            "patient_profile": {},
            "trials_raw": [],
        }
    )
    assert out["include_not_yet_recruiting"] is True
    assert "refined:True" in out["current_queries"][0]


@pytest.mark.asyncio
async def test_execute_searches_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_settings", lambda: SimpleNamespace(max_trials_per_query=3))

    async def fake_run(queries, max_trials):
        _ = max_trials
        return [{"nct_id": "X", "q": queries[0]}]

    monkeypatch.setattr(nodes.trial_search, "_run_search_queries", fake_run)
    ok = await nodes.execute_searches({"current_queries": ["q1"]})
    assert ok["fetched_trials"][0]["nct_id"] == "X"

    async def bad_run(_queries, _max_trials):
        raise RuntimeError("oops")

    monkeypatch.setattr(nodes.trial_search, "_run_search_queries", bad_run)
    bad = await nodes.execute_searches({"current_queries": ["q1"]})
    assert bad["fetched_trials"] == []
    assert bad["internal_errors"]


@pytest.mark.asyncio
async def test_assess_and_finalize_and_retry_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    out = await nodes.assess_and_finalize(
        {
            "fetched_trials": [{"nct_id": "A"}, {"nct_id": "A"}],
            "existing_nct_ids": [],
            "internal_decisions": ["d1"],
            "executed_query_strings": ["q"],
            "internal_errors": [],
        }
    )
    assert len(out["trials_raw"]) == 1

    monkeypatch.setattr(
        nodes,
        "get_settings",
        lambda: SimpleNamespace(one_pass_mode=True, retrieval_internal_max_retries=1),
    )
    assert nodes.should_retry_search({"fetched_trials": [], "existing_nct_ids": []}) == "finalize"

    monkeypatch.setattr(
        nodes,
        "get_settings",
        lambda: SimpleNamespace(one_pass_mode=False, retrieval_internal_max_retries=2),
    )
    route = nodes.should_retry_search(
        {"fetched_trials": [{"nct_id": "A"}], "existing_nct_ids": [], "internal_retry_count": 0}
    )
    assert route == "retry_search"


@pytest.mark.asyncio
async def test_broaden_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nodes.search_refiner,
        "refine_search_strategy",
        lambda **_: {
            "refined_terms": {"primary_search_terms": ["broad"]},
            "include_not_yet_recruiting": False,
            "decision_note": "broadened",
        },
    )
    monkeypatch.setattr(
        nodes.trial_search,
        "build_search_queries",
        lambda terms, profile, include_not_yet_recruiting=False: [terms["primary_search_terms"][0]],
    )
    out = await nodes.broaden_and_retry(
        {
            "internal_retry_count": 0,
            "normalized_terms": {},
            "patient_profile": {},
            "fetched_trials": [],
        }
    )
    assert out["internal_retry_count"] == 1
    assert out["current_queries"] == ["broad"]
