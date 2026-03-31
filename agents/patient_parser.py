from __future__ import annotations

from typing import Any

from config import get_llm
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from models.patient import ExtractedPatientProfile
from prompts.patient_extraction import build_patient_extraction_prompt
from tools.retry import llm_retry

chain = ChatPromptTemplate.from_template(
    build_patient_extraction_prompt()
) | get_llm().with_structured_output(ExtractedPatientProfile)


@llm_retry
async def _invoke_patient_parser_llm(raw_text: str) -> ExtractedPatientProfile:
    """Invoke the LLM chain to extract structured patient data."""
    result = await chain.ainvoke(
        {"clinical_text": raw_text},
        config={"run_name": "patient_parse", "tags": ["supervisor", "parse"]},
    )
    return result


async def parse_patient_profile(raw_text: str) -> dict[str, Any]:
    """Parse raw clinical text into a structured dictionary."""
    try:
        parsed = await _invoke_patient_parser_llm(raw_text)
        return parsed.model_dump()
    except Exception as exc:
        logger.error("Patient parsing failed after retries: %s", exc)

        fallback_profile = ExtractedPatientProfile(
            additional_notes=f"[PARSING FAILED - RAW TEXT INCLUDED]\n\n{raw_text}"
        )
        return fallback_profile.model_dump()
