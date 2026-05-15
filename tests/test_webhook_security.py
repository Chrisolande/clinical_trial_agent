import pytest
import typer
from tools.cli_support import validate_webhook_url


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/hook",
        "https://user:pass@example.com/hook",  # pragma: allowlist secret
        "ftp://example.com/hook",
        "https://localhost/hook",
        "https://127.0.0.1/hook",
        "https://[::1]/hook",
        "https://10.0.0.2/hook",
        "https://192.168.1.10/hook",
        "https://172.16.0.5/hook",
        "https://169.254.169.254/latest/meta-data",
    ],
)
def test_validate_webhook_url_blocks_ssrf_targets(url: str) -> None:
    with pytest.raises(typer.BadParameter):
        validate_webhook_url(url)


def test_validate_webhook_url_allows_public_https() -> None:
    validate_webhook_url("https://hooks.example.com/clinical")


def test_validate_webhook_url_allows_local_only_when_explicit() -> None:
    validate_webhook_url("http://127.0.0.1:8080/hook", allow_local=True)


def test_validate_webhook_url_enforces_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBHOOK_ALLOWED_HOSTS", "hooks.example.com")
    validate_webhook_url("https://hooks.example.com/clinical")
    with pytest.raises(typer.BadParameter):
        validate_webhook_url("https://other.example.com/clinical")
