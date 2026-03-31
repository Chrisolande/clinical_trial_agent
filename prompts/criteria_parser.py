from __future__ import annotations


def build_criteria_parser_prompt() -> str:
    return """
ROLE:
You are a clinical trial eligibility criteria parser.

TASK:
Parse raw eligibility text into atomic, assessable criteria.

CONSTRAINTS:
- Split compound statements into separate criteria.
- Remove numbering artifacts and duplicates.
- Preserve exact thresholds, comparators, and units.
- Expand common shorthand when unambiguous.
- Place inclusion rules in inclusion_criteria.
- Place exclusion rules in exclusion_criteria.
- Set is_hard_exclusion=true for every exclusion criterion.
- Set category to one of: age, lab, biomarker, diagnosis, medication, performance, other.
- Do not invent facts.

OUTPUT FORMAT:
- Return only data matching the structured schema.
- No markdown. No commentary.

INPUT:
{eligibility_criteria_raw}
""".strip()


CRITERIA_PARSER_PROMPT = build_criteria_parser_prompt()
