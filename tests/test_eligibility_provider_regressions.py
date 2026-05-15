from typing import Any

import pytest
from agents.eligibility_reasoner import evaluate_criteria_batch
from langchain_core.messages import AIMessage
from models.judge_verdict import JudgeVerdict

from clinical_trial_agent.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class StructuredProvider:
    def with_structured_output(self, _: Any) -> "StructuredProvider":
        return self

    async def ainvoke(self, *_: Any, **__: Any) -> JudgeVerdict:
        return JudgeVerdict(
            match_score=0.92,
            match_tier="strong",
            major_criteria_assessable=True,
            inclusion_met=[
                "histology confirmed synthetic NSCLC",
                "EGFR biomarker present",
                "ECOG performance status 0-1",
                "measurable disease by RECIST",
            ],
            key_concern="",
            rationale="Synthetic structured output",
        )


class JsonStringProvider:
    async def ainvoke(self, *_: Any, **__: Any) -> AIMessage:
        return AIMessage(
            content=(
                '{"match_score":0.6,"match_tier":"moderate",'
                '"major_criteria_assessable":true,"inclusion_met":["Age >= 18"],'
                '"key_concern":"","rationale":"Synthetic JSON"}'
            )
        )


class MalformedProvider:
    async def ainvoke(self, *_: Any, **__: Any) -> AIMessage:
        return AIMessage(content="not-json")


class TimeoutProvider:
    async def ainvoke(self, *_: Any, **__: Any) -> AIMessage:
        raise TimeoutError()


class ExceptionProvider:
    async def ainvoke(self, *_: Any, **__: Any) -> AIMessage:
        raise RuntimeError("synthetic provider failure")


def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_URI", "postgresql://x")
    monkeypatch.setenv("MEMORY_DB_DSN", "postgresql://x")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_PRIVACY_MODE", "full_consent")
    monkeypatch.setenv("CLINICAL_DATA_EXTERNAL_LLM_CONSENT", "true")


def _trial() -> dict[str, Any]:
    return {
        "nct_id": "NCTPROVIDER",
        "criteria_source": "ctgov_api",
        "criteria_source_verified": True,
        "criteria_completeness": "full",
    }


def _criteria() -> list[dict[str, str]]:
    return [
        {"criteria_type": "inclusion", "text": "histology confirmed synthetic NSCLC"},
        {"criteria_type": "inclusion", "text": "EGFR biomarker present"},
        {"criteria_type": "inclusion", "text": "ECOG performance status 0-1"},
        {"criteria_type": "inclusion", "text": "measurable disease by RECIST"},
    ]


@pytest.mark.parametrize(
    ("provider", "expected_tier"),
    [
        (StructuredProvider(), "strong"),
        (JsonStringProvider(), "weak"),
        (MalformedProvider(), "weak"),
        (TimeoutProvider(), "weak"),
        (ExceptionProvider(), "weak"),
    ],
)
@pytest.mark.asyncio
async def test_fake_provider_outputs_do_not_create_unsafe_high_tier(
    monkeypatch: pytest.MonkeyPatch,
    provider: Any,
    expected_tier: str,
) -> None:
    _env(monkeypatch)
    monkeypatch.setattr("agents.eligibility_reasoner.get_llm", lambda **_: provider)

    result = await evaluate_criteria_batch(
        {"age": 60, "primary_condition": "synthetic NSCLC"},
        _trial(),
        _criteria(),
    )

    assert result["match_tier"] == expected_tier
    if expected_tier == "weak":
        assert result["match_score"] <= 0.45
