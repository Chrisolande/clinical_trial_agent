from __future__ import annotations

import asyncio
from typing import Any, cast

from config import get_llm, settings
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from models.criteria import ParsedEligibilityCriterion
from prompts.criteria_parser import build_criteria_parser_prompt
from tools import cache
from tools.retry import llm_retry


async def parse_eligibility_criteria(eligibility_text: str, nct_id: str) -> dict[str, Any]:
    if not eligibility_text or len(eligibility_text) < 20:
        return {"inclusion_criteria": [], "exclusion_criteria": []}

    prompt = ChatPromptTemplate.from_template(build_criteria_parser_prompt())
    structured_llm = get_llm().with_structured_output(ParsedEligibilityCriterion)

    chain = prompt | structured_llm

    cache_params = {
        "nct_id": nct_id,
        "eligibility_criteria_raw": eligibility_text[: settings.criteria_text_max_chars],
    }
    if settings.use_cache:
        cached = cache.get_cached("criteria_parser", cache_params)
        if isinstance(cached, dict):
            return cached
    try:
        parsed_obj = await _invoke_criteria_llm(chain, cache_params)
        result = parsed_obj.model_dump()
        parsed = _assign_ids(result, nct_id)

        if settings.use_cache:
            cache.set_cached("criteria_parser", cache_params, parsed)

        return parsed
    except Exception as exc:
        logger.error("Criteria parsing failed for %s: %s", nct_id, exc)
        return {"inclusion_criteria": [], "exclusion_criteria": []}


@llm_retry
async def _invoke_criteria_llm(chain: Any, inputs: dict[str, Any]) -> ParsedEligibilityCriterion:
    result = await asyncio.wait_for(
        chain.ainvoke(
            inputs,
            config={"run_name": "criteria_parse", "tags": ["eligibility", "parse"]},
        ),
        timeout=settings.llm_call_timeout_seconds,
    )
    if not result:
        raise ValueError(f"Unexpected criteria parser result type: {type(result)}")
    return cast("ParsedEligibilityCriterion", result)


def _assign_ids(
    parsed: dict[str, list[dict[str, Any]]], nct_id: str
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
    results = []
    for trial in trials:
        nct_id = trial.get("nct_id", "unknown")
        raw_criteria = trial.get("eligibility_criteria_raw", "")

        parsed = await parse_eligibility_criteria(raw_criteria or "", nct_id)

        results.append(
            {
                "trial": trial,
                "inclusion_criteria": parsed.get("inclusion_criteria", []),
                "exclusion_criteria": parsed.get("exclusion_criteria", []),
            }
        )
    return results
