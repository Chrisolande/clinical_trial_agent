import pytest
from pydantic import SecretStr
from tools.errors import ClinicalTrialsClientError

import clinical_trial_agent.clinical_trials as clinical_trials


@pytest.mark.parametrize(
    "params",
    [
        {"query.cond": "lung cancer"},
        {"query.intr": "FOLFOX"},
        {"query.term": "NSCLC EGFR positive"},
        {"query.cond": "NSCLC", "query.intr": "osimertinib", "query.term": "EGFR exon 19"},
    ],
)
def test_phi_params_include_condition_intervention_term_and_mixed(
    params: dict[str, str],
) -> None:
    assert clinical_trials._contains_phi_params({"format": "json", **params}) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {"query.cond": "lung cancer"},
        {"query.intr": "FOLFOX"},
        {"query.term": "NSCLC EGFR positive"},
        {"query.cond": "NSCLC", "query.intr": "osimertinib", "query.term": "EGFR exon 19"},
    ],
)
async def test_phi_params_without_proxy_are_not_sent_directly(
    monkeypatch: pytest.MonkeyPatch,
    params: dict[str, str],
) -> None:
    calls: list[str] = []

    class FakeClient:
        async def get(self, *_args: object, **_kwargs: object) -> object:
            calls.append("get")
            raise AssertionError("PHI params must not use direct GET transport")

        async def post(self, *_args: object, **_kwargs: object) -> object:
            calls.append("post")
            raise AssertionError("PHI params must not use direct POST transport")

    monkeypatch.setattr(
        clinical_trials,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "ctgov_proxy_url": "",
                "ctgov_proxy_token": SecretStr(""),
                "ctgov_transport_mode": "get",
            },
        )(),
    )

    with pytest.raises(ClinicalTrialsClientError, match="requires CTGOV_PROXY_URL"):
        await clinical_trials._request_with_transport(
            FakeClient(),
            "https://clinicaltrials.gov/api/v2/studies",
            {"format": "json", **params},
            clinical_trials.httpx.Timeout(30.0),
        )

    assert calls == []


@pytest.mark.asyncio
async def test_search_trials_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request(url: str, params: dict[str, object]) -> dict[str, object]:
        assert "studies" in url
        assert params["format"] == "json"
        return {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT123", "briefTitle": "Trial A"},
                        "statusModule": {"overallStatus": "RECRUITING"},
                        "eligibilityModule": {},
                        "designModule": {},
                        "outcomesModule": {},
                        "contactsLocationsModule": {},
                        "armsInterventionsModule": {},
                        "conditionsModule": {},
                        "descriptionModule": {},
                        "sponsorCollaboratorsModule": {},
                    }
                }
            ]
        }

    monkeypatch.setattr(clinical_trials, "_request_json_with_retry", fake_request)

    result = await clinical_trials.search_trials(condition="lung")
    assert result["error"] is None
    assert len(result["studies"]) == 1
    assert result["studies"][0]["nct_id"] == "NCT123"


@pytest.mark.asyncio
async def test_search_trials_maps_domain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request(url: str, params: dict[str, object]) -> dict[str, object]:
        raise ClinicalTrialsClientError("boom", status_code=429, retryable=True)

    monkeypatch.setattr(clinical_trials, "_request_json_with_retry", fake_request)

    result = await clinical_trials.search_trials(condition="lung")
    assert result["studies"] == []
    assert result["error"]["status_code"] == 429
    assert result["error"]["retryable"] is True


@pytest.mark.asyncio
async def test_fetch_trial_detail_rejects_empty_nct() -> None:
    result = await clinical_trials.fetch_trial_detail("   ")
    assert result["trial"] is None
    assert result["error"]["type"] == "validation_error"


@pytest.mark.asyncio
async def test_fetch_trial_detail_missing_nct_in_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_request(url: str, params: dict[str, object]) -> dict[str, object]:
        return {
            "protocolSection": {
                "identificationModule": {"nctId": "", "briefTitle": "Broken"},
                "statusModule": {},
                "eligibilityModule": {},
                "designModule": {},
                "outcomesModule": {},
                "contactsLocationsModule": {},
                "armsInterventionsModule": {},
                "conditionsModule": {},
                "descriptionModule": {},
                "sponsorCollaboratorsModule": {},
            }
        }

    monkeypatch.setattr(clinical_trials, "_request_json_with_retry", fake_request)

    result = await clinical_trials.fetch_trial_detail("NCTBAD")
    assert result["trial"] is None
    assert result["error"]["type"] == "request_error"
    assert result["error"]["retryable"] is False


@pytest.mark.asyncio
async def test_request_helper_uses_urllib_fallback_on_403(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 403

        @staticmethod
        def json() -> dict[str, object]:
            return {"blocked": True}

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            _ = (exc_type, exc, tb)

        async def get(self, *args: object, **kwargs: object) -> FakeResponse:
            _ = (args, kwargs)
            return FakeResponse()

    monkeypatch.setattr(clinical_trials.httpx, "AsyncClient", lambda **_: FakeClient())
    monkeypatch.setattr(
        clinical_trials,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "ctgov_retry_attempts": 1,
                "ctgov_retry_backoff_base": 2.0,
                "ctgov_user_agent": "test-agent",
                "ctgov_accept": "application/json",
                "ctgov_proxy_url": "",
                "ctgov_proxy_token": SecretStr(""),
                "ctgov_transport_mode": "get",
            },
        )(),
    )
    monkeypatch.setattr(
        clinical_trials,
        "_urllib_get_json",
        lambda url, params, timeout=30.0: {"studies": [{"protocolSection": {}}]},
    )

    result = await clinical_trials._request_json_with_retry(
        "https://clinicaltrials.gov/api/v2/studies", {"format": "json"}
    )
    assert isinstance(result, dict)
    assert "studies" in result


@pytest.mark.asyncio
async def test_request_helper_returns_proxy_unavailable_error_for_phi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = clinical_trials.httpx.Request("POST", "http://localhost:8000/ctgov/search")
    req_error = clinical_trials.httpx.ConnectError("boom", request=request)

    class FakeClient:
        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            _ = (exc_type, exc, tb)

        async def post(self, *args: object, **kwargs: object) -> object:
            _ = (args, kwargs)
            raise req_error

    monkeypatch.setattr(clinical_trials.httpx, "AsyncClient", lambda **_: FakeClient())
    monkeypatch.setattr(
        clinical_trials,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "ctgov_retry_attempts": 1,
                "ctgov_retry_backoff_base": 2.0,
                "ctgov_user_agent": "test-agent",
                "ctgov_accept": "application/json",
                "ctgov_proxy_url": "http://localhost:8000/ctgov/search",
                "ctgov_proxy_token": SecretStr(""),
                "ctgov_transport_mode": "get",
            },
        )(),
    )

    with pytest.raises(ClinicalTrialsClientError, match="ClinicalTrials proxy unavailable"):
        await clinical_trials._request_json_with_retry(
            "https://clinicaltrials.gov/api/v2/studies",
            {"format": "json", "query.cond": "lung cancer"},
        )
