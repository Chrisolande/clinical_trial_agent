import os
from unittest.mock import patch

import pytest
from agents.consent import assert_external_llm_consent


def test_assert_external_llm_consent_no_requirement():
    with patch("agents.consent.external_llm_requires_consent", return_value=False):
        # Should not raise
        assert_external_llm_consent()


def test_assert_external_llm_consent_granted():
    with (
        patch("agents.consent.external_llm_requires_consent", return_value=True),
        patch.dict(os.environ, {"CLINICAL_DATA_EXTERNAL_LLM_CONSENT": "true"}),
    ):
        # Should not raise
        assert_external_llm_consent()


def test_assert_external_llm_consent_denied():
    with (
        patch("agents.consent.external_llm_requires_consent", return_value=True),
        patch.dict(os.environ, {"CLINICAL_DATA_EXTERNAL_LLM_CONSENT": "false"}),
        pytest.raises(RuntimeError, match="CLINICAL_DATA_EXTERNAL_LLM_CONSENT=true is required"),
    ):
        assert_external_llm_consent()


def test_assert_external_llm_consent_missing():
    with (
        patch("agents.consent.external_llm_requires_consent", return_value=True),
        patch.dict(os.environ, {}),
    ):
        if "CLINICAL_DATA_EXTERNAL_LLM_CONSENT" in os.environ:
            del os.environ["CLINICAL_DATA_EXTERNAL_LLM_CONSENT"]
        with pytest.raises(
            RuntimeError, match="CLINICAL_DATA_EXTERNAL_LLM_CONSENT=true is required"
        ):
            assert_external_llm_consent()
