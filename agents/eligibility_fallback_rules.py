import re
from typing import Any


def _to_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def profile_blob(patient_profile: dict[str, Any]) -> str:
    keys = [
        "primary_condition",
        "conditions",
        "stage",
        "medical_history",
        "medications",
        "biomarkers",
        "ecog_performance_status",
    ]
    return " ".join(_to_text(patient_profile.get(k, "")) for k in keys).lower()


def biomarker_blob(patient_profile: dict[str, Any]) -> str:
    return _to_text(patient_profile.get("biomarkers", "")).lower()


def _is_reproductive_criterion(lowered: str) -> bool:
    return any(keyword in lowered for keyword in ("pregnan", "nursing", "lactat", "breastfeed"))


def _sex_specific_not_applicable(
    lowered: str, patient_profile: dict[str, Any]
) -> tuple[str, str] | None:
    if not _is_reproductive_criterion(lowered):
        return None
    sex = str(patient_profile.get("sex", "")).strip().lower()
    if sex == "male":
        return "NOT_APPLICABLE", "Reproductive criterion not applicable to male patient"
    return None


def _extract_age_bound(text: str) -> tuple[int | None, int | None]:
    lowered = text.lower()
    min_age: int | None = None
    max_age: int | None = None
    ge_match = re.search(r"(?:>=|≥|at least)\s*(\d{1,3})", lowered)
    if ge_match:
        min_age = int(ge_match.group(1))
    le_match = re.search(r"(?:<=|≤|at most|up to)\s*(\d{1,3})", lowered)
    if le_match:
        max_age = int(le_match.group(1))
    between_match = re.search(r"between\s+(\d{1,3})\s+(?:and|to)\s+(\d{1,3})", lowered)
    if between_match:
        min_age = int(between_match.group(1))
        max_age = int(between_match.group(2))
    return min_age, max_age


def _keyword_reason_mappings() -> tuple[tuple[tuple[str, ...], str], ...]:
    return (
        (("ecog", "karnofsky", "performance status"), "Missing ECOG/performance status documentation"),
        (("creatinine", "egfr", "renal"), "Missing renal function labs (creatinine/eGFR)"),
        (("ast", "alt", "bilirubin", "liver"), "Missing liver function labs (AST/ALT/bilirubin)"),
        (("anc", "hemoglobin", "platelet", "cbc", "hematolog"), "Missing CBC/hematology labs (ANC/Hb/platelets)"),
        (("ldh",), "Missing LDH value"),
        (("braf", "pd-l1", "biomarker", "mutation"), "Missing molecular biomarker result"),
        (("recist", "measurable lesion", "imaging", "measurable disease"), "Missing baseline imaging/measurable disease assessment"),
        (("histology", "pathology", "diagnosis"), "Missing pathology/diagnosis confirmation"),
        (("bleeding", "hemorrhage"), "Missing recent bleeding history"),
        (("cardiac", "ecg", "echo"), "Missing cardiac history/assessment"),
        (("retinal", "ophthalm"), "Missing ophthalmologic/retinal assessment"),
    )


def missing_reason_from_criterion(text: str, *, exclusion: bool = False) -> str:
    lowered = text.lower()
    for keywords, message in _keyword_reason_mappings():
        if any(keyword in lowered for keyword in keywords):
            return message
    return "Missing trial-specific clinical detail" if not exclusion else "Missing exclusion-history detail"


def _assess_inclusion_age(lowered: str, age: Any) -> tuple[str, str] | None:
    if "age" not in lowered or not isinstance(age, int | float):
        return None
    min_age, max_age = _extract_age_bound(lowered)
    if min_age is not None and age < min_age:
        return "FAILS", "Age below required threshold"
    if max_age is not None and age > max_age:
        return "FAILS", "Age above allowed threshold"
    return "MEETS", "Age appears within stated bounds"


def _assess_inclusion_melanoma(lowered: str, profile_blob_text: str) -> tuple[str, str] | None:
    if "melanoma" not in lowered:
        return None
    if "melanoma" in profile_blob_text:
        return "MEETS", "Diagnosis mentions melanoma and patient condition includes melanoma"
    return "UNCERTAIN", "Melanoma diagnosis not explicit in profile"


def _extract_biomarker_targets(lowered: str) -> set[str]:
    markers = (
        "alk", "braf", "brca1", "brca2", "egfr", "erbb2", "fgfr", "her2", "idh1", "idh2",
        "kras", "met", "msi", "ntrk", "pd-l1", "pik3ca", "ret", "ros1", "tmb",
    )
    return {marker for marker in markers if re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", lowered)}


def _has_explicit_molecular_evidence(biomarker_blob_text: str) -> bool:
    if _extract_biomarker_targets(biomarker_blob_text):
        return True
    return bool(
        re.search(
            r"\b[a-z0-9-]{3,12}\s*(?:mutation|mutated|fusion|rearrangement|amplification|deletion)\b",
            biomarker_blob_text,
        )
    )


def _any_biomarker_target_present(criterion_targets: set[str], biomarker_blob_text: str) -> bool:
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(target)}(?![a-z0-9])", biomarker_blob_text)
        for target in criterion_targets
    )


def _is_generic_mutation_criterion(lowered: str) -> bool:
    return "mutation" in lowered or "molecular" in lowered or "genomic" in lowered


def _assess_targeted_biomarker(criterion_targets: set[str], biomarker_blob_text: str) -> tuple[str, str]:
    if _any_biomarker_target_present(criterion_targets, biomarker_blob_text):
        return "MEETS", "Criterion-specific biomarker evidence exists in profile"
    return "UNCERTAIN", "Criterion-specific biomarker result missing"


def _assess_inclusion_biomarker(lowered: str, patient_profile: dict[str, Any]) -> tuple[str, str] | None:
    biomarker_keywords = ("braf", "pd-l1", "biomarker", "mutation", "molecular", "genomic")
    if not any(keyword in lowered for keyword in biomarker_keywords):
        return None
    criterion_targets = _extract_biomarker_targets(lowered)
    biomarker_blob_text = biomarker_blob(patient_profile)
    if criterion_targets:
        return _assess_targeted_biomarker(criterion_targets, biomarker_blob_text)
    if _is_generic_mutation_criterion(lowered):
        return "UNCERTAIN", "Criterion requires specific biomarker target; generic mutation evidence is insufficient"
    if "biomarker" in lowered and _has_explicit_molecular_evidence(biomarker_blob_text):
        return "MEETS", "Biomarker evidence exists in profile"
    return "UNCERTAIN", "Required biomarker information missing"


def _assess_inclusion_performance(lowered: str, patient_profile: dict[str, Any]) -> tuple[str, str] | None:
    if not any(keyword in lowered for keyword in ("ecog", "performance status", "karnofsky")):
        return None
    if patient_profile.get("ecog_performance_status") is not None:
        return "MEETS", "Performance status present in profile"
    return "UNCERTAIN", "Performance status missing"


def assess_inclusion(text: str, patient_profile: dict[str, Any], profile_blob_text: str) -> tuple[str, str]:
    lowered = text.lower()
    not_applicable = _sex_specific_not_applicable(lowered, patient_profile)
    if not_applicable is not None:
        return not_applicable
    age_result = _assess_inclusion_age(lowered, patient_profile.get("age"))
    if age_result is not None:
        return age_result
    melanoma_result = _assess_inclusion_melanoma(lowered, profile_blob_text)
    if melanoma_result is not None:
        return melanoma_result
    biomarker_result = _assess_inclusion_biomarker(lowered, patient_profile)
    if biomarker_result is not None:
        return biomarker_result
    performance_result = _assess_inclusion_performance(lowered, patient_profile)
    if performance_result is not None:
        return performance_result
    return "UNCERTAIN", missing_reason_from_criterion(text, exclusion=False)


def assess_exclusion(text: str, patient_profile: dict[str, Any], profile_blob_text: str) -> tuple[str, str]:
    lowered = text.lower()
    not_applicable = _sex_specific_not_applicable(lowered, patient_profile)
    if not_applicable is not None:
        return not_applicable
    if "uveal melanoma" in lowered:
        if "uveal" in profile_blob_text:
            return "FAILS", "Profile indicates uveal melanoma exclusion"
        return "MEETS", "No explicit uveal melanoma evidence"
    if any(k in lowered for k in ["autoimmune", "organ transplant", "active infection"]):
        if any(k in profile_blob_text for k in ["autoimmune", "transplant", "infection"]):
            return "FAILS", "Potential exclusion condition appears in profile"
        return "UNCERTAIN", missing_reason_from_criterion(text, exclusion=True)
    return "UNCERTAIN", missing_reason_from_criterion(text, exclusion=True)


def evaluate_single_criterion(
    text: str,
    ctype: str,
    patient_profile: dict[str, Any],
    profile_blob_text: str,
) -> tuple[str, str]:
    if ctype == "exclusion":
        return assess_exclusion(text, patient_profile, profile_blob_text)
    return assess_inclusion(text, patient_profile, profile_blob_text)


def append_outcome(
    verdict: str,
    reason: str,
    text: str,
    *,
    is_exclusion: bool,
    inclusion_met: list[str],
    inclusion_failed: list[str],
    inclusion_uncertain: list[str],
    exclusion_triggered: list[str],
    exclusion_uncertain: list[str],
    missing: list[str],
) -> None:
    if is_exclusion:
        if verdict == "FAILS":
            exclusion_triggered.append(text)
        elif verdict == "UNCERTAIN":
            exclusion_uncertain.append(text)
            missing.append(reason)
        return
    if verdict == "MEETS":
        inclusion_met.append(text)
    elif verdict == "FAILS":
        inclusion_failed.append(text)
    elif verdict != "NOT_APPLICABLE":
        inclusion_uncertain.append(text)
        missing.append(reason)


def select_timeout_tier_and_score(
    exclusion_triggered: list[str], inclusion_failed: list[str], total: int
) -> tuple[str, float]:
    if exclusion_triggered:
        return "disqualified", 0.0
    if inclusion_failed:
        return "weak", 0.3
    return "weak", 0.25 if total == 0 else 0.42
