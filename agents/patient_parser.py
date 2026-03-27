from __future__ import annotations

from config import get_llm
from models.patient import ExtractedPatientProfile
from prompts.patient_extraction import PATIENT_EXTRACTION_PROMPT

chain = PATIENT_EXTRACTION_PROMPT | get_llm().with_structured_output(ExtractedPatientProfile)
