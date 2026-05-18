from pathlib import Path

import pytest
from agents import report_synthesizer
from agents.eligibility_prompt_builder import build_judge_messages
from langchain_core.messages import BaseMessage
from models.report import ReportPlan
from pydantic import ValidationError

from clinical_trial_agent.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_eligibility_prompt_builder_returns_base_messages() -> None:
    messages = build_judge_messages(
        profile={"age": 54, "primary_condition": "NSCLC"},
        trial={"nct_id": "NCT1", "brief_title": "Example Trial"},
        criteria=[{"criteria_type": "inclusion", "text": "Age >= 18"}],
    )
    assert messages
    assert all(isinstance(message, BaseMessage) for message in messages)


def test_eligibility_prompt_builder_rejects_empty_patient_summary() -> None:
    with pytest.raises(ValidationError):
        build_judge_messages(
            profile={},
            trial={"nct_id": "NCT1", "brief_title": "Example Trial"},
            criteria=[{"criteria_type": "inclusion", "text": "Age >= 18"}],
        )


def test_privacy_mode_blocks_external_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_PRIVACY_MODE", "blocked")
    with pytest.raises(RuntimeError, match="LLM_PRIVACY_MODE=blocked"):
        build_judge_messages(
            profile={"age": 54, "primary_condition": "NSCLC"},
            trial={"nct_id": "NCT1", "brief_title": "Example Trial"},
            criteria=[{"criteria_type": "inclusion", "text": "Age >= 18"}],
        )


@pytest.mark.asyncio
async def test_synthesis_prompt_uses_chat_prompt_template(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"from_template": False, "with_structured_output": False, "model": None}

    class DummyPrompt:
        def __or__(self, other):
            return other

    class DummyChain:
        async def ainvoke(self, _context, config=None):
            return ReportPlan.model_validate(
                {
                    "patient_summary": "summary",
                    "executive_summary": "comparative summary",
                    "bottom_line": "summary bottom line",
                    "strong_matches": [],
                    "moderate_matches": [],
                    "information_gaps": [],
                    "recommended_actions": [],
                    "excluded_summary": "none",
                    "limitations": [],
                }
            )

    class LLMStub:
        def with_structured_output(self, model):
            called["with_structured_output"] = True
            called["model"] = model
            return DummyChain()

    def _from_template(_template: str) -> DummyPrompt:
        called["from_template"] = True
        return DummyPrompt()

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(report_synthesizer, "get_llm", lambda **_kwargs: LLMStub())
    monkeypatch.setattr(report_synthesizer.ChatPromptTemplate, "from_template", _from_template)

    await report_synthesizer.generate_report_plan(
        patient_profile={"age": 60, "sex": "male", "primary_condition": "NSCLC"},
        scored_trials=[],
        eligibility_verdicts={},
        missing_info=[],
        qa_issues=[],
    )

    assert called["from_template"] is True
    assert called["with_structured_output"] is True
    assert called["model"] is ReportPlan


@pytest.mark.asyncio
async def test_eligibility_reasoner_prefers_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agents import eligibility_reasoner

    called = {"with_structured_output": False, "model": None}

    class DummyStructured:
        async def ainvoke(self, *_args, **_kwargs):
            # Return a dict to drive validate_verdict path.
            return {
                "match_score": 0.2,
                "match_tier": "weak",
                "major_criteria_assessable": False,
                "inclusion_met": [],
                "inclusion_failed": [],
                "inclusion_uncertain": ["Age >= 18"],
                "exclusion_triggered": [],
                "exclusion_uncertain": [],
                "critical_missing_info": [],
                "key_concern": "test",
                "rationale": "test",
            }

    class LLMStub:
        def with_structured_output(self, model):
            called["with_structured_output"] = True
            called["model"] = model
            return DummyStructured()

    monkeypatch.setattr(eligibility_reasoner, "get_llm", lambda **_: LLMStub())
    # Ensure privacy guard doesn't block prompt build.
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    result = await eligibility_reasoner.evaluate_criteria_batch(
        patient_profile={"age": 54, "primary_condition": "NSCLC"},
        trial={"nct_id": "NCT1", "brief_title": "Example Trial"},
        all_criteria=[{"criteria_type": "inclusion", "text": "Age >= 18"}],
    )

    assert called["with_structured_output"] is True
    assert called["model"] is eligibility_reasoner.JudgeVerdict
    assert result["trial_id"] == "NCT1"


def test_architecture_guard_disallows_raw_llm_prompt_antipatterns() -> None:
    root = Path(__file__).resolve().parents[1]
    search_roots = ("agents", "subagents", "clinical_trial_agent", "tools")
    py_files: list[Path] = []
    for folder in search_roots:
        py_files.extend((root / folder).rglob("*.py"))
    violations: list[str] = []
    for path in py_files:
        text = path.read_text(encoding="utf-8")
        if (
            '.ainvoke("' in text
            or ".ainvoke('" in text
            or '.invoke("' in text
            or ".invoke('" in text
        ):
            violations.append(str(path.relative_to(root)))
        # Avoid manual role/content dict messages except in the explicit supervisor ReAct adapter.
        if (
            '"role": "user"' in text or '"role": "system"' in text or '"role": "assistant"' in text
        ) and path.relative_to(root).as_posix() != "agents/supervisor.py":
            violations.append(str(path.relative_to(root)))
    assert not violations, f"Raw-string invoke anti-patterns found: {violations}"
