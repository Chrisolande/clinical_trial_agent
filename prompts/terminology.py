from __future__ import annotations


def build_terminology_prompt() -> str:
    return """
ROLE:
You are a biomedical terminology normalization specialist.

TASK:
Normalize condition and medication terms for clinical trial search.

CONSTRAINTS:
- For each condition provide: canonical, icd10, mesh_id, synonyms, broader_terms, narrower_terms, search_terms.
- For each medication provide: generic_name, drug_classes, synonyms, search_terms.
- Expand abbreviations and shorthand when unambiguous.
- Preserve clinically meaningful subtype qualifiers.
- Omit uncertain codes/IDs rather than guessing.
- Select 3 to 5 primary_search_terms.
- Select 3 to 5 intervention_search_terms.
- Prefer terms commonly indexed by ClinicalTrials.gov.

OUTPUT FORMAT:
- Return only data matching the structured schema.
- No markdown. No commentary.

INPUT TERMS:
{terms}
""".strip()


TERMINOLOGY_NORMALISATION_PROMPT = build_terminology_prompt()
