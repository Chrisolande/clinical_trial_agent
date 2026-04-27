from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from models.patient import ExtractedPatientProfile
from prompts.patient_extraction import build_patient_extraction_prompt
from tools.retry import llm_retry
from tools.sanitizer import sanitize_patient_profile

from agents.consent import assert_external_llm_consent
from clinical_trial_agent.config import get_llm


def _get_chain() -> Any:
    prompt = ChatPromptTemplate.from_template(build_patient_extraction_prompt())
    return prompt | get_llm().with_structured_output(ExtractedPatientProfile)


@llm_retry
async def _invoke_patient_parser_llm(raw_text: str) -> ExtractedPatientProfile:
    assert_external_llm_consent()
    result = await _get_chain().ainvoke(
        {"clinical_text": raw_text},
        config={"run_name": "patient_parse", "tags": ["supervisor", "parse"]},
    )
    if isinstance(result, ExtractedPatientProfile):
        return result
    if isinstance(result, dict):
        return ExtractedPatientProfile.model_validate(result)
    raise TypeError(f"Unexpected patient parser output type: {type(result)!r}")


async def parse_patient_profile(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("Clinical note input is empty.")

    sanitized = sanitize_patient_profile(text).text
    try:
        parsed = await _invoke_patient_parser_llm(sanitized)
        return parsed.model_dump()
    except Exception as exc:
        logger.error("Patient parsing failed after retries: {}", exc)
        fallback_profile = ExtractedPatientProfile(
            additional_notes=f"[PARSING FAILED - SANITIZED RAW TEXT INCLUDED]\n\n{sanitized}"
        )
        return fallback_profile.model_dump()
