import json

import pytest
from cryptography.fernet import Fernet

from clinical_trial_agent.memory import (
    _deserialize_encrypted_json,
    _get_fernet_key,
    _get_profile_hash_salt,
    _patient_hash,
    _serialize_encrypted_json,
)


@pytest.fixture(autouse=True)
def scoped_memory_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from clinical_trial_agent.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("TENANT_ID", "test-tenant")
    monkeypatch.setenv("FACILITY_ID", "test-facility")
    yield
    get_settings.cache_clear()


def test_get_profile_hash_salt_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROFILE_HASH_SALT", raising=False)
    with pytest.raises(RuntimeError, match="PROFILE_HASH_SALT must be set"):
        _get_profile_hash_salt()


def test_get_profile_hash_salt_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_HASH_SALT", "")
    with pytest.raises(RuntimeError, match="PROFILE_HASH_SALT must be set"):
        _get_profile_hash_salt()


def test_get_profile_hash_salt_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_HASH_SALT", "test-salt-123")
    assert _get_profile_hash_salt() == "test-salt-123"


def test_get_fernet_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DB_ENCRYPTION_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DB_ENCRYPTION_KEY must be set"):
        _get_fernet_key()


def test_get_fernet_key_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_ENCRYPTION_KEY", "")
    with pytest.raises(RuntimeError, match="DB_ENCRYPTION_KEY must be set"):
        _get_fernet_key()


def test_get_fernet_key_success(monkeypatch: pytest.MonkeyPatch) -> None:
    key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setenv("DB_ENCRYPTION_KEY", key)
    result = _get_fernet_key()
    assert isinstance(result, bytes)
    assert result == key.encode("utf-8")


def test_patient_hash_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_HASH_SALT", "fixed-salt")
    profile = {"age": 50, "conditions": ["cancer"]}
    hash1 = _patient_hash(profile)
    hash2 = _patient_hash(profile)
    assert hash1 == hash2
    assert len(hash1) == 64


def test_patient_hash_different_for_different_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROFILE_HASH_SALT", "fixed-salt")
    profile1 = {"age": 50, "conditions": ["cancer"]}
    profile2 = {"age": 51, "conditions": ["cancer"]}
    hash1 = _patient_hash(profile1)
    hash2 = _patient_hash(profile2)
    assert hash1 != hash2


def test_patient_hash_uses_salt(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = {"age": 50, "conditions": ["cancer"]}
    monkeypatch.setenv("PROFILE_HASH_SALT", "salt1")
    hash1 = _patient_hash(profile)
    monkeypatch.setenv("PROFILE_HASH_SALT", "salt2")
    hash2 = _patient_hash(profile)
    assert hash1 != hash2


def test_serialize_encrypted_json() -> None:
    fernet = Fernet(Fernet.generate_key())
    payload = {"key": "value", "num": 123}
    encrypted = _serialize_encrypted_json(payload, fernet)
    assert isinstance(encrypted, str)
    decrypted_bytes = fernet.decrypt(encrypted.encode("utf-8"))
    decrypted_dict = json.loads(decrypted_bytes)
    assert decrypted_dict == payload


def test_deserialize_encrypted_json_success() -> None:
    fernet = Fernet(Fernet.generate_key())
    payload = {"key": "value", "num": 123}
    encrypted = fernet.encrypt(json.dumps(payload).encode("utf-8")).decode("utf-8")
    result = _deserialize_encrypted_json(encrypted, fernet)
    assert result == payload


def test_deserialize_encrypted_json_invalid_returns_none() -> None:
    fernet = Fernet(Fernet.generate_key())
    result = _deserialize_encrypted_json("not-valid-encrypted-data", fernet)
    assert result is None


def test_deserialize_encrypted_json_non_dict_returns_none() -> None:
    fernet = Fernet(Fernet.generate_key())
    encrypted = fernet.encrypt(json.dumps(["not", "a", "dict"]).encode("utf-8")).decode("utf-8")
    result = _deserialize_encrypted_json(encrypted, fernet)
    assert result is None
