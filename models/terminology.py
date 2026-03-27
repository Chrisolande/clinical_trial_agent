from pydantic import BaseModel, Field


class NormalisedCondition(BaseModel):
    canonical: str = Field(description="MeSH or ICD preferred name.")
    icd10: str | None = Field(default=None, description="ICD-10 code if known.")
    mesh_id: str | None = Field(default=None, description="MeSH ID if known.")
    synonyms: list[str] = Field(default_factory=list)
    broader_terms: list[str] = Field(default_factory=list, description="Parent concepts.")
    narrower_terms: list[str] = Field(default_factory=list, description="Child concepts.")
    search_terms: list[str] = Field(default_factory=list)


class NormalisedMedication(BaseModel):
    generic_name: str = Field(description="INN generic name.")
    drug_classes: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)


class NormalisedTerminology(BaseModel):
    conditions: dict[str, NormalisedCondition] = Field(
        default_factory=dict, description="Keyed by original term as supplied in the input."
    )
    medications: dict[str, NormalisedMedication] = Field(
        default_factory=dict, description="Keyed by original name as supplied in the input."
    )
    primary_search_terms: list[str] = Field(
        default_factory=list, description="Top 3-5 terms for ClinicalTrials.gov condition search."
    )
    intervention_search_terms: list[str] = Field(
        default_factory=list, description="Top 3-5 drug or intervention search terms."
    )
