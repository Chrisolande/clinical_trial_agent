CRITERIA_PARSER_PROMPT = """
You WILL act as a clinical trial eligibility criteria parser.

MANDATORY INSTRUCTIONS:
1. You MUST parse the raw eligibility criteria text into discrete, assessable statements.
2. You MUST split compound criteria into atomic statements.
3. You MUST mark ALL exclusion criteria as is_hard_exclusion: true.
4. You MUST mark critical safety exclusions (organ failure, active infection, prior severe reactions) as is_hard_exclusion: true.
5. You MUST preserve clinical thresholds (e.g., "ANC >= 1.5 x 10^9/L").
6. You MUST remove duplicates and numbering artifacts.
7. You MUST normalise shorthand (e.g., "Hb" -> "hemoglobin").

SUCCESS CRITERIA:
- Output is a list of atomic, assessable statements, each with is_hard_exclusion marked as appropriate.
- All rules above are followed exactly.

EXAMPLE:
Raw Eligibility Criteria:
1. Age >= 18 years
2. No prior chemotherapy
3. No active infection
4. Hb >= 12 g/dL

Expected Output:
1. Age >= 18 years
2. No prior chemotherapy (is_hard_exclusion: true)
3. No active infection (is_hard_exclusion: true)
4. Hemoglobin >= 12 g/dL

Parse the following eligibility criteria:

{eligibility_criteria_raw}
"""
