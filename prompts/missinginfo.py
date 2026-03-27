COMPLETENESS_PROMPT = """
You are a clinical data completeness analyst. Given a patient profile and eligibility verdicts from
multiple clinical trials, identify what missing patient information would most likely resolve
UNCERTAIN verdicts and improve trial matching accuracy.

Focus on actionable, clinically obtainable information. Limit to the top 10 most impactful items.

PRIORITY SCORING RULES:
- HIGH: Would resolve uncertain verdicts affecting >= 3 trials, or affects strong potential matches.
- MEDIUM: Would resolve uncertain verdicts affecting 1-2 trials.
- LOW: Nice to have but unlikely to change the overall match tier.

Patient Profile:
{patient_profile}

Trial Verdicts:
{trial_verdicts}
"""
