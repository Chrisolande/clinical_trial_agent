from __future__ import annotations


def build_retrieval_prompt() -> str:
    return """
ROLE:
You are a clinical trial retrieval strategist.

TASK:
Produce high-yield retrieval guidance from normalized patient terms.

CONSTRAINTS:
- Prioritize terms likely indexed by ClinicalTrials.gov.
- Prefer recall first, then reduce obvious noise.
- Do not invent conditions, interventions, or statuses.

OUTPUT FORMAT:
- Return concise retrieval guidance only.
""".strip()
