from __future__ import annotations

from types import SimpleNamespace

import pytest
from agents import trial_search


class DummyTavily:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    def invoke(self, _: dict[str, str]) -> list[dict[str, str]]:
        return [
            {"content": "Inclusion criteria: Age >= 18; ECOG 0-1; histologically confirmed NSCLC."}
        ]


@pytest.mark.asyncio
async def test_tavily_supplements_missing_eligibility_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agents.trial_search.TavilySearch", DummyTavily)
    monkeypatch.setattr(
        trial_search,
        "settings",
        SimpleNamespace(
            tavily_enable_ctgov_supplement=True,
            tavily_api_key="test-key",
            tavily_max_results=3,
            tavily_max_trials_to_enrich=8,
            criteria_text_max_chars=8000,
        ),
    )
    trials = [
        {"nct_id": "NCT1", "brief_title": "Trial 1", "eligibility_criteria_raw": None},
        {"nct_id": "NCT2", "brief_title": "Trial 2", "eligibility_criteria_raw": "Already present"},
    ]

    updated, count = await trial_search._supplement_trials_from_tavily(trials)
    assert count == 1
    assert updated[0]["eligibility_criteria_raw"] is not None
    assert updated[1]["eligibility_criteria_raw"] == "Already present"


def test_build_search_queries_biases_condition_intervention_and_biomarker() -> None:
    normalized_terms = {
        "primary_search_terms": ["colorectal adenocarcinoma"],
        "intervention_search_terms": ["FOLFOX", "chemotherapy"],
    }
    patient_profile = {
        "primary_condition": "colorectal adenocarcinoma",
        "biomarkers": ["MSI-H"],
        "medications": ["FOLFOX"],
    }
    queries = trial_search.build_search_queries(normalized_terms, patient_profile)
    assert queries
    assert queries[0].get("condition") == "colorectal adenocarcinoma"
    assert queries[0].get("intervention") == "FOLFOX"
    assert any("MSI-H" in str(q.get("condition", "")) for q in queries)
