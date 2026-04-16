from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

from loguru import logger

_MAX_EXPANDED_TOKENS = 64

_CONDITION_SYNONYMS_FALLBACK: dict[str, list[str]] = {
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


def _normalize_text_to_tokens(text: str) -> set[str]:
    stopwords = _get_stopwords()
    return {
        token
        for token in re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
        if len(token) > 2 and token not in stopwords
    }


@lru_cache(maxsize=1)
def _get_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords
    except ImportError as exc:
        logger.warning(
            "NLTK not available for stopword filtering; continuing without stopwords. {}", exc
        )
        return set()

    try:
        words = stopwords.words("english")
    except LookupError as exc:
        logger.warning(
            "NLTK stopwords corpus missing; run `python -m nltk.downloader stopwords`. {}",
            exc,
        )
        return set()

    return {str(word).lower().strip() for word in words if str(word).strip()}


@lru_cache(maxsize=1)
def _load_term_parser() -> tuple[Any | None, str | None]:
    try:
        import spacy
    except ImportError as exc:
        return None, f"spaCy/scispaCy import failed: {exc}"

    requested_model = os.getenv("SCISPACY_MODEL", "").strip()
    model_candidates = [
        *([requested_model] if requested_model else []),
        "en_core_sci_sm",
        "en_core_sci_md",
    ]

    errors: list[str] = []
    seen: set[str] = set()
    for model_name in model_candidates:
        if model_name in seen:
            continue
        seen.add(model_name)
        try:
            return spacy.load(model_name), None
        except OSError as exc:
            errors.append(f"{model_name}: {exc}")

    return None, "No scispaCy model available. Tried " + "; ".join(errors)


@lru_cache(maxsize=1)
def _get_term_parser() -> Any | None:
    parser, reason = _load_term_parser()
    if parser is None:
        logger.warning("Medical terminology parser unavailable; fallback enabled. {}", reason)
    return parser


def _legacy_expand(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for key, values in _CONDITION_SYNONYMS_FALLBACK.items():
        normalized_key = key.lower()
        synonym_tokens = set(normalized_key.replace("-", " ").split())
        for phrase in values:
            synonym_tokens.update(phrase.lower().replace("-", " ").split())
        if normalized_key in expanded or expanded.intersection(synonym_tokens):
            expanded.add(normalized_key)
            expanded.update(synonym_tokens)
    return expanded


def _extract_with_term_parser(tokens: set[str], parser: Any) -> set[str]:
    doc = parser(" ".join(sorted(tokens)))
    extracted: set[str] = set()

    for token in doc:
        for candidate in (
            str(getattr(token, "lemma_", "") or ""),
            str(getattr(token, "norm_", "") or ""),
            str(getattr(token, "text", "") or ""),
        ):
            extracted.update(_normalize_text_to_tokens(candidate))

    for ent in getattr(doc, "ents", []):
        extracted.update(_normalize_text_to_tokens(str(getattr(ent, "text", ""))))

    noun_chunks = getattr(doc, "noun_chunks", None)
    if noun_chunks is not None:
        try:
            for chunk in noun_chunks:
                extracted.update(_normalize_text_to_tokens(str(getattr(chunk, "text", ""))))
        except (ValueError, TypeError) as exc:
            logger.debug("Skipping noun_chunks extraction due to parser constraints: {}", exc)

    return extracted


def _bounded(tokens: set[str]) -> set[str]:
    if len(tokens) <= _MAX_EXPANDED_TOKENS:
        return tokens
    return set(sorted(tokens)[:_MAX_EXPANDED_TOKENS])


def expand_condition_tokens(tokens: set[str]) -> set[str]:
    stopwords = _get_stopwords()
    normalized = {
        token.lower().strip() for token in tokens if isinstance(token, str) and token.strip()
    }
    normalized = {token for token in normalized if token not in stopwords}
    if not normalized:
        return set()

    parser = _get_term_parser()
    if parser is None:
        return _bounded(_legacy_expand(normalized))

    expanded = set(normalized)
    try:
        expanded.update(_extract_with_term_parser(normalized, parser))
    except Exception as exc:
        logger.warning("Medical terminology parser failed at runtime; fallback enabled. {}", exc)
        return _bounded(_legacy_expand(normalized))

    expanded = _legacy_expand(expanded)
    return _bounded(expanded)
