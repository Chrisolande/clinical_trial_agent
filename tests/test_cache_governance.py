from types import SimpleNamespace

from tools import cache


def _settings(**overrides):
    values = {
        "use_cache": True,
        "cache_ttl_seconds": 3600,
        "tenant_id": "tenant-a",
        "facility_id": "facility-a",
        "llm_privacy_mode": "deidentified",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _verdict() -> dict:
    return {
        "trial_id": "NCT1",
        "match_score": 0.6,
        "match_tier": "moderate",
        "verdicts": [
            {
                "criterion_id": "NCT1_inc_0",
                "criterion_text": "Age >= 18 years",
                "criterion_type": "inclusion",
                "source_type": "parsed_inclusion",
                "verdict": "MEETS",
                "evidence_refs": [
                    {
                        "source_type": "parsed_inclusion",
                        "trial_id": "NCT1",
                        "criterion_id": "NCT1_inc_0",
                    }
                ],
            }
        ],
    }


class _DiskCacheShouldNotBeUsed:
    def get(self, _key):
        raise AssertionError("disk cache get should not be used by default")

    def set(self, *_args, **_kwargs):
        raise AssertionError("disk cache set should not be used by default")


def test_eligibility_verdict_cache_uses_memory_not_disk_by_default(
    monkeypatch,
) -> None:
    cache._eligibility_memory_cache.clear()
    monkeypatch.delenv("ELIGIBILITY_VERDICT_ALLOW_DISK_CACHE", raising=False)
    monkeypatch.setattr(cache, "get_settings", lambda: _settings())
    monkeypatch.setattr(cache, "_cache", _DiskCacheShouldNotBeUsed())

    cache.set_cached_eligibility_verdict("NCT1", "profile-hash", _verdict())

    cached = cache.get_cached_eligibility_verdict("NCT1", "profile-hash")
    assert cached is not None
    assert cached["trial_id"] == "NCT1"
    assert cached["evidence_contract_version"] == cache.ELIGIBILITY_EVIDENCE_CONTRACT_VERSION


def test_eligibility_cache_scope_includes_privacy_and_tenant_context(monkeypatch) -> None:
    settings = _settings()
    monkeypatch.setattr(cache, "get_settings", lambda: settings)

    params = cache._eligibility_cache_params("NCT1", "profile-hash")
    assert params["tenant_id"] == "tenant-a"
    assert params["facility_id"] == "facility-a"
    assert params["privacy_mode"] == "deidentified"
    assert params["cache_schema_version"] == cache.ELIGIBILITY_CACHE_SCHEMA_VERSION
    assert params["evidence_contract_version"] == cache.ELIGIBILITY_EVIDENCE_CONTRACT_VERSION

    settings.llm_privacy_mode = "full_consent"
    assert cache._eligibility_cache_params("NCT1", "profile-hash") != params


def test_legacy_eligibility_disk_cache_payload_is_ignored(monkeypatch) -> None:
    class LegacyDiskCache:
        def get(self, _key):
            return {
                "trial_id": "NCT1",
                "match_score": 0.9,
                "match_tier": "strong",
                "verdicts": [{"criterion_text": "Age >= 18 years", "verdict": "MEETS"}],
            }

    cache._eligibility_memory_cache.clear()
    monkeypatch.setenv("ELIGIBILITY_VERDICT_ALLOW_DISK_CACHE", "true")
    monkeypatch.setattr(cache, "get_settings", lambda: _settings())
    monkeypatch.setattr(cache, "_cache", LegacyDiskCache())

    assert cache.get_cached_eligibility_verdict("NCT1", "profile-hash") is None
