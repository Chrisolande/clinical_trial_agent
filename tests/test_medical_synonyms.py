import tools.medical_synonyms as medical_synonyms


def test_expand_condition_tokens_fallback_expands_nsclc(monkeypatch) -> None:
    medical_synonyms._get_term_parser.cache_clear()
    medical_synonyms._get_stopwords.cache_clear()

    monkeypatch.setattr(medical_synonyms, "_get_term_parser", lambda: None)
    monkeypatch.setattr(medical_synonyms, "_get_stopwords", lambda: {"the", "and"})

    expanded = medical_synonyms.expand_condition_tokens({"NSCLC", "the"})
    assert "nsclc" in expanded
    assert "nonsmallcelllungcancer" in expanded
    assert "the" not in expanded


def test_expand_condition_tokens_handles_missing_stopwords(monkeypatch) -> None:
    medical_synonyms._get_term_parser.cache_clear()
    medical_synonyms._get_stopwords.cache_clear()

    monkeypatch.setattr(medical_synonyms, "_get_term_parser", lambda: None)
    monkeypatch.setattr(medical_synonyms, "_get_stopwords", lambda: set())

    expanded = medical_synonyms.expand_condition_tokens({"HER2"})
    assert "her2" in expanded
    assert "erbb2" in expanded
