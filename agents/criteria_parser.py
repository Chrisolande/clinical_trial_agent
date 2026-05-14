import asyncio
import json
import re
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from models.criteria import ParsedEligibilityCriterion
from prompts.criteria_parser import build_criteria_parser_prompt
from tools import cache
from tools.retry import llm_retry

from clinical_trial_agent.config import get_llm, get_settings

CriterionCategoryLiteral = Literal[
    "age", "lab", "biomarker", "diagnosis", "medication", "performance", "other"
]

_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_template(build_criteria_parser_prompt())


_JSON_REPAIR_PROMPT = ChatPromptTemplate.from_template(
    """
You are parsing clinical trial eligibility criteria.

Return ONLY valid JSON. Do not return markdown. Do not explain.

The JSON must match this schema exactly:

{{
  "inclusion_criteria": [
    {{
      "text": "criterion text",
      "criterion_type": "inclusion",
      "is_hard_exclusion": false,
      "category": "age | lab | biomarker | diagnosis | medication | performance | other"
    }}
  ],
  "exclusion_criteria": [
    {{
      "text": "criterion text",
      "criterion_type": "exclusion",
      "is_hard_exclusion": true,
      "category": "age | lab | biomarker | diagnosis | medication | performance | other"
    }}
  ]
}}

Rules:
- Preserve the original medical meaning.
- Do not invent criteria.
- Split compound criteria only when clearly separable.
- Inclusion criteria must go under inclusion_criteria.
- Exclusion criteria must go under exclusion_criteria.
- If a section is absent, return an empty list for that section.
- Categories must be one of: age, lab, biomarker, diagnosis, medication, performance, other.

NCT ID:
{nct_id}

Eligibility criteria:
{eligibility_criteria_raw}
""".strip()
)


def _get_structured_chain() -> Any:
    """Build a fresh parser chain per call to avoid stale loop-bound transports."""
    return _PROMPT | get_llm().with_structured_output(ParsedEligibilityCriterion)


def _get_json_chain() -> Any:
    """Fallback LLM parser that asks for raw JSON instead of tool/structured output."""
    return _JSON_REPAIR_PROMPT | get_llm()


def _extract_text_from_message(result: Any) -> str:
    if isinstance(result, str):
        return result

    if isinstance(result, BaseMessage):
        content = result.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts)

    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content

    return str(result)


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    return cleaned.strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_json_fences(text)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                f"No JSON object found in LLM output: {cleaned[:500]}"
            ) from None

        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise TypeError(f"Expected JSON object, got {type(parsed)!r}")

    return parsed


def _ensure_non_empty_parse(parsed: ParsedEligibilityCriterion, nct_id: str) -> None:
    if parsed.inclusion_criteria or parsed.exclusion_criteria:
        return

    raise ValueError(
        f"LLM produced empty criteria parse for {nct_id}. "
        "Refusing deterministic fallback because criteria parsing must be LLM-backed."
    )


async def parse_eligibility_criteria(eligibility_text: str, nct_id: str) -> dict[str, Any]:
    if not eligibility_text or len(eligibility_text.strip()) < 20:
        return {"inclusion_criteria": [], "exclusion_criteria": []}

    cache_params = {
        "nct_id": nct_id,
        "eligibility_criteria_raw": eligibility_text[: get_settings().criteria_text_max_chars],
    }

    if get_settings().use_cache:
        cached = await asyncio.to_thread(cache.get_cached, "criteria_parser", cache_params)
        if isinstance(cached, dict):
            return cached

    parsed_obj = await _parse_with_llm_or_raise(cache_params)

    result = parsed_obj.model_dump()
    parsed = _assign_ids(result, nct_id)

    if get_settings().use_cache:
        await asyncio.to_thread(cache.set_cached, "criteria_parser", cache_params, parsed)

    return parsed


async def _parse_with_llm_or_raise(inputs: dict[str, Any]) -> ParsedEligibilityCriterion:
    nct_id = str(inputs.get("nct_id", "unknown"))

    structured_error: Exception | None = None

    try:
        parsed = await _invoke_structured_criteria_llm(_get_structured_chain(), inputs)
        _ensure_non_empty_parse(parsed, nct_id)
        return parsed
    except Exception as exc:
        structured_error = exc
        logger.warning(
            "Structured criteria parse failed for {}. Retrying with raw JSON LLM parse. Error: {}",
            nct_id,
            exc,
        )

    try:
        parsed = await _invoke_json_criteria_llm(_get_json_chain(), inputs)
        _ensure_non_empty_parse(parsed, nct_id)
        return parsed
    except Exception as json_exc:
        raise RuntimeError(
            f"Criteria parsing failed for {nct_id}. "
            f"Structured parse error: {structured_error}. "
            f"JSON parse error: {json_exc}. "
            "No deterministic fallback was used."
        ) from json_exc


@llm_retry
async def _invoke_structured_criteria_llm(
    chain: Any,
    inputs: dict[str, Any],
) -> ParsedEligibilityCriterion:
    result = await chain.ainvoke(
        inputs,
        config={"run_name": "criteria_parse_structured", "tags": ["eligibility", "parse"]},
    )

    if result is None:
        raise ValueError("LLM returned None for structured criteria parse")

    if isinstance(result, ParsedEligibilityCriterion):
        return result

    if isinstance(result, dict):
        return ParsedEligibilityCriterion.model_validate(result)

    raise TypeError(f"Unexpected structured criteria parser output type: {type(result)!r}")


@llm_retry
async def _invoke_json_criteria_llm(
    chain: Any,
    inputs: dict[str, Any],
) -> ParsedEligibilityCriterion:
    result = await chain.ainvoke(
        inputs,
        config={"run_name": "criteria_parse_json", "tags": ["eligibility", "parse", "json"]},
    )

    text = _extract_text_from_message(result)
    parsed_dict = _extract_json_object(text)
    return ParsedEligibilityCriterion.model_validate(parsed_dict)


def _assign_ids(
    parsed: dict[str, list[dict[str, Any]]],
    nct_id: str,
) -> dict[str, list[dict[str, Any]]]:
    for i, crit in enumerate(parsed.get("inclusion_criteria", [])):
        crit["criterion_id"] = f"{nct_id}_inc_{i}"
        crit.setdefault("criterion_type", "inclusion")
        crit.setdefault("is_hard_exclusion", False)
        crit.setdefault("category", "other")

    for i, crit in enumerate(parsed.get("exclusion_criteria", [])):
        crit["criterion_id"] = f"{nct_id}_exc_{i}"
        crit.setdefault("criterion_type", "exclusion")
        crit.setdefault("is_hard_exclusion", True)
        crit.setdefault("category", "other")

    return parsed


async def parse_criteria_for_trials(
    trials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tasks = [
        parse_eligibility_criteria(
            (
                f"Inclusion Criteria:\n{trial.get('inclusion_criteria_parsed', '')}\n\n"
                f"Exclusion Criteria:\n{trial.get('exclusion_criteria_parsed', '')}"
            ).strip()
            if trial.get("inclusion_criteria_parsed") or trial.get("exclusion_criteria_parsed")
            else (trial.get("eligibility_criteria_raw", "") or ""),
            trial.get("nct_id", "unknown"),
        )
        for trial in trials
    ]

    results_parsed = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    failures: list[str] = []

    for trial, parsed in zip(trials, results_parsed, strict=False):
        nct_id = trial.get("nct_id", "unknown")

        if isinstance(parsed, BaseException):
            logger.exception("Criteria parsing failed for {}: {}", nct_id, parsed)
            failures.append(str(nct_id))
            continue

        results.append(
            {
                "trial": trial,
                "inclusion_criteria": parsed.get("inclusion_criteria", []),
                "exclusion_criteria": parsed.get("exclusion_criteria", []),
            }
        )

    if failures:
        raise RuntimeError(
            "Criteria parsing failed for trial(s): "
            + ", ".join(failures)
            + ". No deterministic fallback was used."
        )

    return results
