import pytest
from agents import trial_search


def test_extract_studies_from_results_parses_raw_protocol_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trial_search, "parse_trial_from_response", lambda payload: {"nct_id": "NCT1"}
    )
    results = [
        {"studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT1"}}}]},
        {"studies": [{"nct_id": "NCT2"}]},
    ]
    trials = trial_search._extract_studies_from_results(results)
    assert {"nct_id": "NCT1"} in trials
    assert {"nct_id": "NCT2"} in trials


@pytest.mark.asyncio
async def test_run_search_queries_passes_term_condition_intervention_and_filters_missing_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = []

    async def fake_search_trials(**kwargs):
        called.append(kwargs)
        # include one bad item without nct_id to ensure filtering
        return {"studies": [{"nct_id": "NCTX"}, {"brief_title": "bad"}], "error": None}

    monkeypatch.setattr(trial_search, "search_trials", fake_search_trials)
    queries = [
        {"condition": "cond", "intervention": "drug", "status": ["RECRUITING"]},
        {"term": "cond EGFR", "status": ["RECRUITING"]},
    ]
    trials = await trial_search._run_search_queries(queries, page_size=10)
    assert len(called) == 2
    assert called[0]["condition"] == "cond"
    assert called[0]["intervention"] == "drug"
    assert called[1]["term"] == "cond EGFR"
    assert all(t.get("nct_id") for t in trials)
