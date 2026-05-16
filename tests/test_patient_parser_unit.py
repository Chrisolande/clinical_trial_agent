import pytest
from agents import patient_parser


@pytest.mark.asyncio
async def test_parse_patient_profile_rejects_empty() -> None:
    with pytest.raises(ValueError, match="Clinical note input is empty"):
        await patient_parser.parse_patient_profile("   ")


@pytest.mark.asyncio
async def test_parse_patient_profile_returns_fallback_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(_raw: str):
        raise RuntimeError("x")

    monkeypatch.setattr(patient_parser, "_invoke_patient_parser_llm", boom)
    result = await patient_parser.parse_patient_profile("Patient has NSCLC.")
    assert "additional_notes" in result
    assert "PARSING FAILED" in str(result.get("additional_notes", ""))


@pytest.mark.asyncio
async def test_parse_patient_profile_happy_path_uses_model_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Dummy:
        def model_dump(self):
            return {"age": 60, "primary_condition": "NSCLC"}

    async def ok(_raw: str):
        return Dummy()

    monkeypatch.setattr(patient_parser, "_invoke_patient_parser_llm", ok)
    result = await patient_parser.parse_patient_profile("Patient has NSCLC.")
    assert result["age"] == 60
    assert result["primary_condition"] == "NSCLC"
