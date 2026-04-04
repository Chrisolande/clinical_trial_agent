"""Project exception hierarchy."""


class ClinicalTrialAgentError(Exception):
    """Base exception for all domain-level failures in the clinical trial agent."""


class ClinicalTrialsClientError(ClinicalTrialAgentError):
    """Raised when ClinicalTrials.gov request/response handling fails."""

    def __init__(
        self, message: str, *, status_code: int | None = None, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
