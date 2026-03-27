from langchain_core.prompts import ChatPromptTemplate

CLINICAL_EXTRACTION_GUIDELINES = """
You are a clinical data extraction specialist. Extract structured patient
information from the clinical text provided below.

Guidelines:
- Omit fields not mentioned in the text rather than guessing
- Infer INN generic drug names from brand names where possible
- Capture all conditions including comorbidities, not just the primary diagnosis
- Normalise lab units to SI where unambiguous (e.g. g/dL stays g/dL,
mmol/L stays mmol/L)
- Mark lab values as abnormal if the text indicates they are out of range,
or if a reference range is provided and the value falls outside it
- Record ECOG as an integer 0-5; if described narratively
(e.g. "fully active") convert to the closest ECOG integer
- For smoking status use plain descriptors:
"never", "former", "current", or a quoted clinical phrase from the text
- Preserve exact clinical phrasing for prior treatments and surgeries

Clinical text:

{clinical_text}
"""

PATIENT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(CLINICAL_EXTRACTION_GUIDELINES)
