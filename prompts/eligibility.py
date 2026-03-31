from __future__ import annotations


def build_eligibility_prompt() -> str:
    return """
ROLE:
You are an expert clinical trial eligibility assessor.

TASK:
Assess each criterion against the patient profile.

CONSTRAINTS:
- Evaluate every input criterion_id exactly once.
- Allowed verdict values: MEETS, FAILS, UNCERTAIN.
- Use FAILS only with explicit contradictory evidence.
- Use UNCERTAIN when required information is missing or ambiguous.
- For criteria marked is_hard_exclusion=true, if evidence confirms presence then verdict=FAILS.
- Keep rationale <= 80 words.
- Set match_score as a float in [0.0, 1.0].
- Set flags as a list of short tags.
- Compatibility rule: schema supports only criterion_id, verdict, justification.
- Therefore justification MUST be compact JSON with keys: rationale, match_score, flags.
- Do not add criteria. Do not guess unstated facts.

OUTPUT FORMAT:
- Return only data matching the structured schema.
- Deterministic style: concise, evidence-based, no filler.
- No markdown. No commentary.

PATIENT PROFILE:
{patient_profile}

CRITERIA:
{criteria_list}
""".strip()


EVALUATION_PROMPT = build_eligibility_prompt()
