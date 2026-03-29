COMPLETENESS_PROMPT = """
You WILL act as a clinical data completeness analyst.

MANDATORY INSTRUCTIONS:
1. You MUST review the patient profile and eligibility verdicts from multiple clinical trials.
2. You MUST identify missing patient information that would most likely resolve UNCERTAIN verdicts and improve trial matching accuracy.
3. You MUST focus only on actionable, clinically obtainable information.
4. You MUST limit your output to the top 10 most impactful missing items.
5. You MUST assign a priority score to each item:
	- HIGH: Would resolve uncertain verdicts affecting 3 or more trials, or affects strong potential matches.
	- MEDIUM: Would resolve uncertain verdicts affecting 1-2 trials.
	- LOW: Nice to have but unlikely to change the overall match tier.

SUCCESS CRITERIA:
- Output is a ranked list of up to 10 missing information items, each with a priority score (HIGH, MEDIUM, LOW).
- All items are actionable and clinically obtainable.
- All rules above are followed exactly.

EXAMPLE:
Patient Profile:
Name: Jane Smith
Age: 55
Diagnosis: Breast cancer
Lab: Hemoglobin 11.5 g/dL
Prior treatments: Tamoxifen

Trial Verdicts:
Trial 1: UNCERTAIN (missing HER2 status)
Trial 2: MEETS
Trial 3: UNCERTAIN (missing ECOG score)

Expected Output:
1. HER2 status (HIGH)
2. ECOG performance status (MEDIUM)

Patient Profile:
{patient_profile}

Trial Verdicts:
{trial_verdicts}
"""
