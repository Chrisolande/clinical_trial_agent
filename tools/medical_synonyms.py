CONDITION_SYNONYMS: dict[str, list[str]] = {
    "crc": ["colorectal", "colorectal cancer", "colon cancer", "rectal cancer"],
    "nsclc": ["non small cell", "non-small-cell", "non-small-cell lung cancer"],
    "sclc": ["small cell lung", "small-cell lung cancer"],
    "her2": ["erbb2", "her-2"],
    "tnbc": ["triple negative breast cancer", "triple-negative breast cancer"],
    "aml": ["acute myeloid leukemia", "acute myelogenous leukemia"],
    "cll": ["chronic lymphocytic leukemia"],
    "nhl": ["non hodgkin lymphoma", "non-hodgkin lymphoma"],
    "mm": ["multiple myeloma"],
    "rcc": ["renal cell carcinoma"],
}


def expand_condition_tokens(tokens: set[str]) -> set[str]:
    expanded = {token.lower() for token in tokens}
    for key, values in CONDITION_SYNONYMS.items():
        normalized_key = key.lower()
        synonym_tokens = set(normalized_key.replace("-", " ").split())
        for phrase in values:
            synonym_tokens.update(phrase.lower().replace("-", " ").split())
        if normalized_key in expanded or expanded.intersection(synonym_tokens):
            expanded.add(normalized_key)
            expanded.update(synonym_tokens)
    return expanded
