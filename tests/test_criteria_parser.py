from types import SimpleNamespace

import pytest
from agents import criteria_parser
from models.criteria import ParsedEligibilityCriterion


def test_assign_ids_sets_defaults() -> None:
    parsed = {
        "inclusion_criteria": [{"text": "A"}],
        "exclusion_criteria": [{"text": "B"}],
    }
    assigned = criteria_parser._assign_ids(parsed, "NCT1")
    assert assigned["inclusion_criteria"][0]["criterion_id"] == "NCT1_inc_0"
    assert assigned["exclusion_criteria"][0]["criterion_id"] == "NCT1_exc_0"
    assert assigned["inclusion_criteria"][0]["criterion_type"] == "inclusion"
    assert assigned["exclusion_criteria"][0]["criterion_type"] == "exclusion"


@pytest.mark.asyncio
async def test_parse_eligibility_criteria_short_text_returns_empty() -> None:
    result = await criteria_parser.parse_eligibility_criteria("short", "NCTX")
    assert result == {"inclusion_criteria": [], "exclusion_criteria": []}


@pytest.mark.asyncio
async def test_parse_eligibility_criteria_uses_cache_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        criteria_parser,
        "get_settings",
        lambda: SimpleNamespace(use_cache=True, criteria_text_max_chars=8000),
    )

    async def fake_to_thread(func, *args, **kwargs):
        _ = (args, kwargs)
        if func is criteria_parser.cache.get_cached:
            return {"inclusion_criteria": [{"text": "cached"}], "exclusion_criteria": []}
        return None

    monkeypatch.setattr(criteria_parser.asyncio, "to_thread", fake_to_thread)
    result = await criteria_parser.parse_eligibility_criteria("Age >= 18 years and more text", "N1")
    assert result["inclusion_criteria"][0]["text"] == "cached"


def test_strip_json_fences_removes_code_block() -> None:
    text = """```json
    {"inclusion_criteria": [], "exclusion_criteria": []}
    ```"""
    cleaned = criteria_parser._strip_json_fences(text)
    assert cleaned.startswith("{")
    assert cleaned.endswith("}")


def test_extract_json_object_parses_wrapped() -> None:
    text = 'LLM output: {"inclusion_criteria": [], "exclusion_criteria": []}'
    parsed = criteria_parser._extract_json_object(text)
    assert parsed == {"inclusion_criteria": [], "exclusion_criteria": []}


@pytest.mark.asyncio
async def test_parse_eligibility_criteria_empty_llm_result_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        criteria_parser,
        "get_settings",
        lambda: SimpleNamespace(use_cache=False, criteria_text_max_chars=8000),
    )
    monkeypatch.setattr(criteria_parser, "_get_structured_chain", lambda: object())
    monkeypatch.setattr(criteria_parser, "_get_json_chain", lambda: object())

    async def fake_structured(_chain, _inputs):
        return ParsedEligibilityCriterion(inclusion_criteria=[], exclusion_criteria=[])

    async def fake_json(_chain, _inputs):
        return ParsedEligibilityCriterion(
            inclusion_criteria=[
                {"text": "Age >= 18 years", "is_hard_exclusion": False, "category": "age"}
            ],
            exclusion_criteria=[],
        )

    monkeypatch.setattr(criteria_parser, "_invoke_structured_criteria_llm", fake_structured)
    monkeypatch.setattr(criteria_parser, "_invoke_json_criteria_llm", fake_json)

    result = await criteria_parser.parse_eligibility_criteria(
        "Inclusion Criteria:\nAge >= 18 years\nExclusion Criteria:\nprior therapy",
        "N2",
    )
    assert result["inclusion_criteria"]
    assert result["inclusion_criteria"][0]["criterion_id"] == "N2_inc_0"


@pytest.mark.asyncio
async def test_parse_eligibility_criteria_exception_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        criteria_parser,
        "get_settings",
        lambda: SimpleNamespace(use_cache=False, criteria_text_max_chars=8000),
    )
    monkeypatch.setattr(criteria_parser, "_get_structured_chain", lambda: object())
    monkeypatch.setattr(criteria_parser, "_get_json_chain", lambda: object())

    async def raise_structured(_chain, _inputs):
        raise RuntimeError("boom")

    async def fake_json(_chain, _inputs):
        return ParsedEligibilityCriterion(
            inclusion_criteria=[
                {"text": "Age >= 18 years", "is_hard_exclusion": False, "category": "age"}
            ],
            exclusion_criteria=[],
        )

    monkeypatch.setattr(criteria_parser, "_invoke_structured_criteria_llm", raise_structured)
    monkeypatch.setattr(criteria_parser, "_invoke_json_criteria_llm", fake_json)

    result = await criteria_parser.parse_eligibility_criteria(
        "Inclusion Criteria:\nAge >= 18 years\nExclusion Criteria:\nprior therapy",
        "N3",
    )
    assert result["inclusion_criteria"]


@pytest.mark.asyncio
async def test_parse_criteria_for_trials_raises_on_any_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_parse(text: str, nct_id: str):
        if nct_id == "BAD":
            raise RuntimeError("x")
        return {"inclusion_criteria": [{"text": text}], "exclusion_criteria": []}

    monkeypatch.setattr(criteria_parser, "parse_eligibility_criteria", fake_parse)
    trials = [
        {"nct_id": "OK", "eligibility_criteria_raw": "Age >= 18"},
        {"nct_id": "BAD", "eligibility_criteria_raw": "Age >= 18"},
    ]
    with pytest.raises(RuntimeError, match="Criteria parsing failed"):
        await criteria_parser.parse_criteria_for_trials(trials)
