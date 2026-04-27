import os

from clinical_trial_agent.config import external_llm_requires_consent


def assert_external_llm_consent() -> None:
    if not external_llm_requires_consent():
        return
    consent = os.environ.get("CLINICAL_DATA_EXTERNAL_LLM_CONSENT", "false").strip().lower()
    if consent != "true":
        raise RuntimeError(
            "CLINICAL_DATA_EXTERNAL_LLM_CONSENT=true is required before sending patient data to external LLMs."
        )
