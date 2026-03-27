from typing import Literal

from pydantic import BaseModel, Field


class MedicationEntry(BaseModel):
    name: str
    generic_name: str | None = None
    dose: str | None = None
    frequency: str | None = None
    drug_class: str | None = None


class LabValue(BaseModel):
    name: str
    value: float
    unit: str
    reference_range: str | None = None
    abnormal: bool | None = None


class Biomarker(BaseModel):
    name: str
    result: str
    method: str | None = None


class ExtractedPatientProfile(BaseModel):
    age: int | None = None
    sex: Literal["male", "female", "other"] | None = None
    weight_kg: float | None = None
    height_cm: float | None = None
    bmi: float | None = None
    primary_condition: str | None = None
    conditions: list[str] = Field(default_factory=list)
    medical_history: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)
    medications: list[MedicationEntry] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    lab_values: list[LabValue] = Field(default_factory=list)
    biomarkers: list[Biomarker] = Field(default_factory=list)
    ecog_performance_status: int | None = Field(
        default=None, ge=0, le=5, description="ECOG score 0-5."
    )
    smoking_status: str | None = None
    alcohol_use: str | None = None
    prior_treatments: list[str] = Field(default_factory=list)
    prior_surgeries: list[str] = Field(default_factory=list)
    additional_notes: str | None = None
