from __future__ import annotations

import pytest
from clinical_trials import _studies_endpoint_url, search_trials


def test_studies_endpoint_url_appends_studies_suffix() -> None:
    assert _studies_endpoint_url("https://clinicaltrials.gov/api/v2") == (
        "https://clinicaltrials.gov/api/v2/studies"
    )


def test_studies_endpoint_url_preserves_existing_studies_suffix() -> None:
    assert _studies_endpoint_url("https://clinicaltrials.gov/api/v2/studies/") == (
        "https://clinicaltrials.gov/api/v2/studies"
    )


@pytest.mark.asyncio
async def test_search_trials_parses_study_records(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request(_: str, __: dict[str, object]) -> dict[str, object]:
        return {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT123",
                            "briefTitle": "Trial Title",
                        },
                        "statusModule": {"overallStatus": "RECRUITING"},
                        "designModule": {"phases": ["PHASE2"]},
                    }
                }
            ]
        }

    monkeypatch.setattr("clinical_trials._ctgov_request_with_retry", fake_request)
    result = await search_trials(condition="x")
    assert result["error"] is None
    assert result["studies"] == [
        {
            "nct_id": "NCT123",
            "brief_title": "Trial Title",
            "official_title": None,
            "overall_status": "RECRUITING",
            "phase": "PHASE2",
            "lead_sponsor": None,
            "eligibility_criteria_raw": None,
            "locations": [],
            "primary_outcomes": [],
            "primary_completion_date": None,
            "minimum_age": None,
            "maximum_age": None,
            "sex_eligibility": None,
            "healthy_volunteers": None,
            "brief_summary": None,
            "conditions": [],
            "interventions": [],
        }
    ]
