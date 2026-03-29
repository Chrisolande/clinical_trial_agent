from langchain_core.prompts import ChatPromptTemplate

CLINICAL_EXTRACTION_GUIDELINES = """
You WILL act as a clinical data extraction specialist.

MANDATORY INSTRUCTIONS:
1. You MUST extract structured patient information from the clinical text provided below.
2. You MUST omit fields not mentioned in the text rather than guessing.
3. You MUST infer INN generic drug names from brand names where possible.
4. You MUST capture all conditions including comorbidities, not just the primary diagnosis.
5. You MUST normalise lab units to SI where unambiguous (e.g. g/dL stays g/dL, mmol/L stays mmol/L).
6. You MUST mark lab values as abnormal if the text indicates they are out of range, or if a reference range is provided and the value falls outside it.
7. You MUST record ECOG as an integer 0-5; if described narratively (e.g. "fully active") you MUST convert to the closest ECOG integer.
8. For smoking status, you MUST use plain descriptors: "never", "former", "current", or a quoted clinical phrase from the text.
9. You MUST preserve exact clinical phrasing for prior treatments and surgeries.

SUCCESS CRITERIA:
- Output is a structured patient profile with only fields present in the text.
- All rules above are followed exactly.

EXAMPLE:
Clinical text:
"Jane Smith, 55-year-old female with breast cancer. Hemoglobin 11.5 g/dL (low). Prior treatment: Tamoxifen. ECOG: fully active. Never smoked."

Expected Output:
Name: Jane Smith
Age: 55
Diagnosis: Breast cancer
Hemoglobin: 11.5 g/dL (abnormal)
Prior treatments: Tamoxifen (INN: tamoxifen)
ECOG: 0
Smoking status: never

Clinical text:

{clinical_text}
"""

PATIENT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(CLINICAL_EXTRACTION_GUIDELINES)
