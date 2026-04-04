import re
from dataclasses import dataclass, field

from loguru import logger

_MAX_PROFILE_LENGTH = 8_000

_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"system\s*:?\s*you\s+are",
    r"new\s+instructions?\s*:",
    r"###\s*instruction",
    r"<\s*system\s*>",
    r"disregard\s+(the\s+)?(above|previous|prior)",
    r"act\s+as\s+(if\s+you\s+are|a\s+different)",
    r"jailbreak",
    r"DAN\s+mode",
    r"you\s+are\s+now\s+a",
]

_COMBINED = re.compile(
    "|".join(f"(?:{p})" for p in _INJECTION_PATTERNS),
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass
class SanitisationResult:
    text: str
    warnings: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.warnings


def sanitize_patient_profile(text: str) -> SanitisationResult:
    warnings: list[str] = []

    def _redact(m: re.Match) -> str:
        warnings.append(f"Potential injection pattern removed: '{m.group(0)[:40]}'")
        return "[REDACTED]"

    text = _COMBINED.sub(_redact, text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) > _MAX_PROFILE_LENGTH:
        text = text[:_MAX_PROFILE_LENGTH]
        warnings.append(f"Profile truncated to {_MAX_PROFILE_LENGTH} characters.")

    if warnings:
        logger.warning("Sanitiser: {} warning(s) for patient profile", len(warnings))

    return SanitisationResult(text=text, warnings=warnings)
