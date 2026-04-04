from langchain_core.prompts import ChatPromptTemplate


def build_patient_extraction_prompt() -> str:
    return """
ROLE
You are a clinical data extraction specialist.

TASK
Extract structured patient data from clinical free text.

RULES
- Use only explicit facts in the text and deterministic normalization.
- If data is absent or uncertain, omit it; never guess.
- Capture primary condition and comorbidities when explicitly stated.
- Normalize medication names to generic when explicitly known.
- Preserve clinical meaning for treatments and surgeries.
- Keep lab units as provided unless normalization is unambiguous.
- Set lab abnormal=true only when abnormality is explicitly supported.
- Map ECOG narrative to integer 0-5 only when clearly supported.
- Keep smoking_status concise and standardized when supported.
- Be deterministic and conservative.

OUTPUT
- Return schema-compliant structured data only.
- No markdown. No commentary. No extra keys.

INPUT
{clinical_text}
""".strip()


CLINICAL_EXTRACTION_GUIDELINES = build_patient_extraction_prompt()
PATIENT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(CLINICAL_EXTRACTION_GUIDELINES)
