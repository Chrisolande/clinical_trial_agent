import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def _canonical_field_id(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "additional_clinical_detail"


class CompletenessAssessment(BaseModel):
    field_id: str = Field(
        description="Canonical stable identifier for the missing field (snake_case)."
    )
    display_name: str = Field(description="Human-readable name for the missing field.")
    category: str = Field(
        default="clinical",
        description="Category of missing data (e.g., labs, imaging, pathology, clinical_assessment).",
    )
    field: str = Field(
        description="The specific clinical field missing (e.g., 'EGFR mutation status', 'creatinine clearance')"
    )
    why_needed: str = Field(
        description="Clinical rationale for why this information is needed to reduce uncertainty."
    )
    evidence_text: str = Field(
        description="Evidence-style rationale text, same intent as why_needed for compatibility."
    )
    description: str = Field(
        description="Clinical rationale for why this information is needed and what specific verdicts it would clarify."
    )
    affected_trial_ids: list[str] = Field(
        description="List of NCT IDs for the trials where this missing data caused an UNCERTAIN verdict."
    )
    priority: Literal["high", "medium", "low"] = Field(
        description="Priority level based on how many trials or strong matches are affected."
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        display_name = str(data.get("display_name") or data.get("field") or "").strip()
        field_id = str(data.get("field_id") or "").strip() or _canonical_field_id(display_name)
        why_needed = str(
            data.get("why_needed") or data.get("evidence_text") or data.get("description") or ""
        ).strip()
        evidence_text = str(
            data.get("evidence_text") or data.get("why_needed") or data.get("description") or ""
        ).strip()
        description = str(
            data.get("description") or data.get("why_needed") or data.get("evidence_text") or ""
        ).strip()
        raw_ids = data.get("affected_trial_ids") or []
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        affected_trial_ids = list(dict.fromkeys(str(tid) for tid in raw_ids))

        data["field_id"] = field_id
        data["display_name"] = display_name or field_id.replace("_", " ").title()
        data["field"] = str(data.get("field") or data["display_name"])
        data["category"] = str(data.get("category") or "clinical")
        data["why_needed"] = why_needed
        data["evidence_text"] = evidence_text
        data["description"] = description
        data["affected_trial_ids"] = affected_trial_ids
        return data

    @model_validator(mode="after")
    def _sync_legacy_fields(self) -> "CompletenessAssessment":
        self.field = self.display_name
        if not self.description:
            self.description = self.why_needed or self.evidence_text
        if not self.why_needed:
            self.why_needed = self.evidence_text or self.description
        if not self.evidence_text:
            self.evidence_text = self.why_needed or self.description
        self.affected_trial_ids = list(dict.fromkeys(self.affected_trial_ids))
        return self


class CompletenessAssessmentList(BaseModel):
    results: list[CompletenessAssessment] = Field(
        description="List of the top missing data items, limited to the 10 most impactful."
    )
