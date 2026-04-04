from types import SimpleNamespace

from agents import search_refiner


def test_refine_search_strategy_first_retry_uses_broader_terms() -> None:
    normalized = {
        "conditions": {
            "c1": {
                "broader_terms": ["solid tumor", "neoplasm"],
                "synonyms": ["crc"],
                "search_terms": ["colorectal cancer"],
            }
        },
        "primary_search_terms": ["colorectal cancer"],
    }
    strategy = search_refiner.refine_search_strategy(
        normalized_terms=normalized,
        patient_profile={"conditions": ["colorectal cancer"]},
        retry_count=0,
        current_trial_count=1,
    )
    assert strategy["use_broader_terms"] is True
    assert strategy["include_not_yet_recruiting"] is False
    assert strategy["refined_terms"]["primary_search_terms"]


def test_refine_search_strategy_second_retry_includes_not_yet_recruiting() -> None:
    strategy = search_refiner.refine_search_strategy(
        normalized_terms={"primary_search_terms": ["x"]},
        patient_profile={},
        retry_count=1,
        current_trial_count=0,
    )
    assert strategy["include_not_yet_recruiting"] is True
    assert "including trials not yet recruiting" in strategy["decision_note"].lower()


def test_refine_search_strategy_third_retry_adds_related_conditions() -> None:
    normalized = {
        "conditions": {"c1": {"narrower_terms": ["rectal cancer"]}},
        "primary_search_terms": ["colorectal cancer"],
    }
    strategy = search_refiner.refine_search_strategy(
        normalized_terms=normalized,
        patient_profile={"medical_history": ["adenoma"]},
        retry_count=2,
        current_trial_count=0,
    )
    terms = strategy["refined_terms"]["primary_search_terms"]
    assert "colorectal cancer" in terms
    assert "rectal cancer" in terms


def test_get_broader_terms_falls_back_to_profile_conditions() -> None:
    result = search_refiner._get_broader_terms({}, {"conditions": ["A", "B", "C"]})
    assert result == ["A", "B", "C"]


def test_get_related_conditions_deduplicates_and_caps() -> None:
    normalized = {
        "conditions": {
            "a": {"narrower_terms": ["x", "x", "y"]},
            "b": {"narrower_terms": ["z"]},
        }
    }
    result = search_refiner._get_related_conditions(
        normalized,
        {"medical_history": ["y", "w"]},
    )
    assert len(result) <= 3
    assert len(result) == len(set(result))


def test_should_continue_refining_uses_settings_limit(monkeypatch) -> None:
    monkeypatch.setattr(
        search_refiner,
        "get_settings",
        lambda: SimpleNamespace(max_retry_attempts=2),
    )
    assert search_refiner.should_continue_refining(0)
    assert search_refiner.should_continue_refining(1)
    assert not search_refiner.should_continue_refining(2)
