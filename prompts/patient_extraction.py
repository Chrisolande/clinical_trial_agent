from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_patient_extraction_prompt() -> str:
    return """
ROLE:
You are a clinical data extraction specialist.

TASK:
Extract structured patient data from free text.

CONSTRAINTS:
- Extract only explicit facts or directly inferable facts.
- Omit unknown fields instead of guessing.
- Capture primary condition and comorbidities.
- Normalize medication names to generic when known.
- Preserve clinical meaning for treatments and surgeries.
- Keep lab units as provided unless normalization is unambiguous.
- Set lab abnormal=true only when evidence indicates abnormality.
- Map ECOG narrative descriptions to integer 0 to 5 when possible.
- Keep smoking_status concise and standardized when possible.

OUTPUT FORMAT:
- Return only data matching the structured schema.
- Deterministic extraction. No markdown. No commentary.

INPUT:
{clinical_text}
""".strip()


CLINICAL_EXTRACTION_GUIDELINES = build_patient_extraction_prompt()
PATIENT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(CLINICAL_EXTRACTION_GUIDELINES)
