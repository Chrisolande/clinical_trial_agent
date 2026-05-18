import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from subagents.eligibility import nodes


@dataclass
class _HasDump:
    value: dict

    def model_dump(self):
        return self.value


def test_patient_profile_to_dict_and_trial_id_and_tokenize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert nodes._patient_profile_to_dict({"a": 1}) == {"a": 1}
    assert nodes._patient_profile_to_dict(_HasDump({"x": 2})) == {"x": 2}
    assert nodes._patient_profile_to_dict("x") == {}

    assert nodes._trial_id({"nct_id": "N1"}) == "N1"
    assert nodes._trial_id({"trial_id": "T1"}) == "T1"
    assert nodes._trial_id({}) == "unknown"

    toks = nodes._tokenize("The AGE and ECOG in trial")
    assert "the" not in toks
    assert "age" in toks

    monkeypatch.setenv("PROFILE_HASH_SALT", "salt")
    settings = SimpleNamespace(
        tenant_id="tenant-a",
        facility_id="facility-a",
        llm_privacy_mode="deidentified",
    )
    monkeypatch.setattr(nodes._nodes_helpers, "get_settings", lambda: settings)
    baseline_hash = nodes._profile_hash_for_cache({"a": 1})
    assert baseline_hash == nodes._profile_hash_for_cache({"a": 1})
    settings.tenant_id = "tenant-b"
    assert nodes._profile_hash_for_cache({"a": 1}) != baseline_hash


@pytest.mark.asyncio
async def test_fanout_dispatch_and_helpers() -> None:
    fan = await nodes.fan_out_trials(
        {
            "trials_deduplicated": [{"nct_id": "A"}, {"nct_id": "B"}],
            "eligibility_verdicts": {"A": {"trial_id": "A"}},
        }
    )
    assert len(fan["trials_to_evaluate"]) == 1

    sends = nodes.dispatch_trial_workers(
        {"trials_to_evaluate": [{"nct_id": "A"}], "patient_profile": {"a": 1}}
    )
    assert len(sends) == 1

    merged = nodes._merge_eligibility_verdicts({"A": {"trial_id": "A"}}, [{"trial_id": "B"}])
    assert set(merged.keys()) == {"A", "B"}


def test_cached_and_irrelevant_worker_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodes, "get_cached_eligibility_verdict", lambda *_: {"trial_id": "N1"})
    cached = nodes._cached_worker_result({"nct_id": "N1"}, "N1", "h")
    assert cached is not None

    monkeypatch.setattr(nodes, "get_cached_eligibility_verdict", lambda *_: None)
    assert nodes._cached_worker_result({"nct_id": "N2"}, "N2", "h") is None

    irr = nodes._irrelevant_worker_result(
        {"nct_id": "N3", "brief_title": "AML", "conditions": ["AML"]},
        "N3",
        "colorectal cancer",
    )
    assert irr is not None
    assert irr["processed_verdicts"][0]["match_tier"] == "disqualified"

    assert nodes._irrelevant_worker_result({"nct_id": "N4"}, "N4", "") is None


def test_normalize_and_collect_criteria_helpers() -> None:
    normalized = nodes._normalize_criteria([{"text": "A"}, "B"], "inclusion")
    assert normalized[1]["criteria_type"] == "inclusion"

    all_criteria = nodes._collect_all_criteria(
        {"inclusion_criteria": [{"text": "A"}], "exclusion_criteria": ["B"]}
    )
    assert len(all_criteria) == 2

    empty = nodes._empty_criteria_worker_result({"trial": {"nct_id": "N5"}}, "N5")
    assert empty["processed_verdicts"][0]["match_tier"] == "weak"


def test_collect_verdicts_and_build_scored_trial() -> None:
    lookup = nodes._build_trial_lookup(
        [{"trial": {"nct_id": "N1", "brief_title": "Trial"}}, {"trial": {}}]
    )
    assert "N1" in lookup

    verdicts = [
        {"verdict": "MEETS", "criterion_type": "inclusion", "criterion_text": "a"},
        {"verdict": "FAILS", "criterion_type": "exclusion", "criterion_text": "b"},
        {"verdict": "UNCERTAIN", "criterion_type": "inclusion", "criterion_text": "c"},
    ]
    assert nodes._collect_verdict_texts(
        verdicts, verdict_name="MEETS", criteria_type="inclusion"
    ) == ["a"]

    scored_trial = nodes._build_scored_trial(
        "N1",
        {
            "match_score": 0.7,
            "match_tier": "moderate",
            "verdicts": verdicts,
            "meets_count": 1,
            "fails_count": 1,
            "uncertain_count": 1,
            "hard_exclusion_failures": 0,
            "major_criteria_assessable": True,
            "key_concern": "x",
            "critical_missing_info": [],
            "rationale": "r",
        },
        {"N1": {"nct_id": "N1", "brief_title": "Trial", "locations": []}},
    )
    assert scored_trial["trial_id"] == "N1"


def test_rank_trials_and_count_viable() -> None:
    ranked = nodes._rank_trials(
        [
            {"trial_id": "A", "tier": "weak", "score": 0.9},
            {"trial_id": "B", "tier": "moderate", "score": 0.1},
        ]
    )
    assert ranked[0]["trial_id"] == "B"
    assert nodes._count_viable_trials(ranked) == 1


@pytest.mark.asyncio
async def test_parse_and_semaphore_and_aggregation_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def parse_ok(_trials):
        return [
            {
                "trial": {"nct_id": "N1"},
                "inclusion_criteria": [],
                "exclusion_criteria": [],
            }
        ]

    monkeypatch.setattr(nodes.criteria_parser, "parse_criteria_for_trials", parse_ok)
    parsed = await nodes._parse_trial_criteria({"nct_id": "N1"}, "N1")
    assert parsed["trial"]["nct_id"] == "N1"

    async def parse_bad(_trials):
        raise RuntimeError("bad")

    monkeypatch.setattr(nodes.criteria_parser, "parse_criteria_for_trials", parse_bad)
    parsed_bad = await nodes._parse_trial_criteria({"nct_id": "N2"}, "N2")
    assert parsed_bad["parse_error"] == "bad"

    async def eval_batch(**_kwargs):
        return {"trial_id": "N3", "match_score": 0.5, "match_tier": "weak"}

    monkeypatch.setattr(nodes.eligibility_reasoner, "evaluate_criteria_batch", eval_batch)
    nodes.set_llm_semaphore(None)
    out = await nodes._evaluate_with_optional_semaphore(
        patient_profile={}, trial={"nct_id": "N3"}, all_criteria=[{"text": "x"}]
    )
    assert out["trial_id"] == "N3"

    nodes.set_llm_semaphore(asyncio.Semaphore(1))
    out2 = await nodes._evaluate_with_optional_semaphore(
        patient_profile={}, trial={"nct_id": "N4"}, all_criteria=[{"text": "x"}]
    )
    assert out2["trial_id"] == "N3"

    monkeypatch.setattr(nodes, "get_settings", lambda: SimpleNamespace(min_match_tier="moderate"))
    agg = await nodes.aggregate_results(
        {
            "eligibility_verdicts": {
                "A": {"trial_id": "A", "match_tier": "weak", "match_score": 0.2}
            },
            "processed_trials_with_criteria": [{"trial": {"nct_id": "A", "brief_title": "A"}}],
            "processed_verdicts": [{"trial_id": "B", "match_tier": "moderate", "match_score": 0.6}],
        }
    )
    assert agg["viable_trial_count"] == 1


@pytest.mark.asyncio
async def test_identify_missing_info_and_viability_and_worker_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def missing_ok(_profile, _verdicts):
        return [{"field": "biomarker"}]

    monkeypatch.setattr(nodes.missing_info, "identify_missing_info", missing_ok)
    out = await nodes.identify_missing_info({"patient_profile": {}, "eligibility_verdicts": {}})
    assert out["missing_info_recommendations"] == [{"field": "biomarker"}]
