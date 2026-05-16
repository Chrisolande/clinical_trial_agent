from agents import query_helpers


def test_build_search_queries_includes_broad_condition_only_queries() -> None:
    queries = query_helpers.build_search_queries(
        normalized_terms={"primary_search_terms": ["colorectal adenocarcinoma"]},
        patient_profile={"primary_condition": "colorectal adenocarcinoma"},
    )
    assert any(q.get("condition") == "colorectal adenocarcinoma" for q in queries)


def test_build_search_queries_biases_intervention_from_normalized_terms() -> None:
    queries = query_helpers.build_search_queries(
        normalized_terms={
            "primary_search_terms": ["colorectal adenocarcinoma"],
            "intervention_search_terms": ["FOLFOX"],
        },
        patient_profile={
            "primary_condition": "colorectal adenocarcinoma",
            "biomarkers": ["MSI-H"],
            "medications": ["FOLFOX"],
        },
    )
    assert queries
    assert queries[0].get("condition") == "colorectal adenocarcinoma"
    assert queries[0].get("intervention") == "FOLFOX"


def test_build_search_queries_generates_biomarker_term_queries_and_ignores_missing_markers() -> (
    None
):
    queries = query_helpers.build_search_queries(
        normalized_terms={"primary_search_terms": ["NSCLC"]},
        patient_profile={
            "primary_condition": "NSCLC",
            "biomarkers": [
                {"name": "EGFR", "result": "positive"},
                {"name": "ALK", "result": "unknown"},
                {"name": "", "result": "positive"},
            ],
        },
    )
    term_queries = [q for q in queries if q.get("term")]
    assert any("EGFR" in str(q.get("term")) for q in term_queries)
    assert not any("ALK unknown" in str(q.get("term")) for q in term_queries)


def test_build_search_queries_non_oncology_profile_still_works_and_caps_at_10() -> None:
    queries = query_helpers.build_search_queries(
        normalized_terms={"primary_search_terms": ["rheumatoid arthritis"]},
        patient_profile={
            "primary_condition": "rheumatoid arthritis",
            "conditions": ["rheumatoid arthritis"],
            "biomarkers": [],
        },
    )
    assert queries
    assert len(queries) <= 10
