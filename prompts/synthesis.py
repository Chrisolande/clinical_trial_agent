from __future__ import annotations


def build_synthesis_prompt() -> str:
    return """
ROLE:
You are a clinical trial matching report writer.

TASK:
Generate a concise, clinician-facing executive summary.

CONSTRAINTS:
- Keep the summary between 150 and 250 words.
- Preserve the trial tier counts and evidence-grounded rationale.
- Highlight the top 1-2 strongest trials and why they rank highly.
- Mention critical missing information that impacts confidence.
- Use physician-facing clinical language.

OUTPUT FORMAT:
- Return structured output with fields:
  - executive_summary
  - patient_summary

INPUT:
Patient: {patient_summary}
Results: {strong_count} strong, {possible_count} possible, {unlikely_count} unlikely out of {total}
Top trials:
{top_trials}
Key missing information: {missing_info}
""".strip()
