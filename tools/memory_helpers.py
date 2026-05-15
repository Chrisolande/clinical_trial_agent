"""Shared helper utilities for PostgreSQL-backed episodic memory."""

import base64
import json
import os
from contextlib import AbstractAsyncContextManager
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from loguru import logger

from clinical_trial_agent.config import get_settings

DDL = """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY
    );

    CREATE TABLE IF NOT EXISTS patient_runs (
        profile_hash  TEXT PRIMARY KEY,
        tenant_id     TEXT NOT NULL CHECK (tenant_id <> ''),
        facility_id   TEXT NOT NULL CHECK (facility_id <> ''),
        result_json   JSONB NOT NULL,
        created_at    TIMESTAMPTZ NOT NULL,
        expires_at    TIMESTAMPTZ NOT NULL,
        UNIQUE (tenant_id, facility_id, profile_hash)
    );

    CREATE INDEX IF NOT EXISTS idx_patient_runs_expires_at
        ON patient_runs (expires_at);

    CREATE TABLE IF NOT EXISTS llm_cache (
        cache_key    TEXT PRIMARY KEY,
        prefix       TEXT NOT NULL,
        value_json   JSONB NOT NULL,
        created_at   TIMESTAMPTZ NOT NULL,
        expires_at   TIMESTAMPTZ NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_llm_cache_expires_at
        ON llm_cache (expires_at);

    CREATE INDEX IF NOT EXISTS idx_llm_cache_prefix
        ON llm_cache (prefix);

    CREATE TABLE IF NOT EXISTS pipeline_audit_log (
        id SERIAL PRIMARY KEY,
        tenant_id TEXT NOT NULL CHECK (tenant_id <> ''),
        facility_id TEXT NOT NULL CHECK (facility_id <> ''),
        profile_hash TEXT NOT NULL,
        run_id TEXT NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        outcome_tier_counts JSONB NOT NULL,
        model_version TEXT NOT NULL,
        consent_flag BOOLEAN NOT NULL
    );

    CREATE TABLE IF NOT EXISTS physician_feedback (
        id SERIAL PRIMARY KEY,
        tenant_id TEXT NOT NULL CHECK (tenant_id <> ''),
        facility_id TEXT NOT NULL CHECK (facility_id <> ''),
        profile_hash TEXT NOT NULL,
        run_id TEXT NOT NULL,
        nct_id TEXT NOT NULL,
        verdict TEXT NOT NULL CHECK (verdict IN ('confirmed', 'rejected')),
        note TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL
    );
"""

SCHEMA_VERSION = 8


def get_profile_hash_salt() -> str:
    salt = os.getenv("PROFILE_HASH_SALT", "").strip()
    if not salt:
        raise RuntimeError("PROFILE_HASH_SALT must be set for patient profile hashing")
    return salt


def get_fernet_key() -> bytes:
    key = os.getenv("DB_ENCRYPTION_KEY", "").strip()
    if not key:
        raise RuntimeError("DB_ENCRYPTION_KEY must be set for encrypted memory storage")

    try:
        base64.urlsafe_b64decode(key.encode("utf-8"))
    except Exception as exc:
        raise RuntimeError("DB_ENCRYPTION_KEY must be a valid Fernet key") from exc

    return key.encode("utf-8")


def serialize_encrypted_json(payload: dict[str, Any], fernet: Fernet) -> str:
    encrypted = fernet.encrypt(
        json.dumps(payload, default=str).encode("utf-8")
    ).decode("utf-8")
    return str(encrypted)


def deserialize_encrypted_json(payload: str, fernet: Fernet) -> dict[str, Any] | None:
    try:
        decrypted = fernet.decrypt(payload.encode("utf-8")).decode("utf-8")
        parsed = json.loads(decrypted)
    except (InvalidToken, json.JSONDecodeError, ValueError, TypeError):
        return None

    return parsed if isinstance(parsed, dict) else None


def get_checkpointer(
    dsn: str | None = None,
) -> AbstractAsyncContextManager[Any] | None:
    """Return an async context manager for LangGraph PostgreSQL checkpointing.

    Important:
        This function returns a context manager, not the raw checkpointer.

        Correct usage:

            checkpointer_cm = get_checkpointer()

            if checkpointer_cm is not None:
                async with checkpointer_cm as checkpointer:
                    await checkpointer.setup()
                    graph = builder.compile(checkpointer=checkpointer)
            else:
                graph = builder.compile()
    """
    active_dsn = dsn or get_settings().database_uri

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        logger.exception(
            "Failed to import LangGraph PostgreSQL checkpointer. "
            "Package may be missing or one of its internal imports failed: {}",
            exc,
        )
        return None

    try:
        return AsyncPostgresSaver.from_conn_string(active_dsn)
    except Exception as exc:
        logger.exception(
            "Failed to create PostgreSQL checkpointer context manager: {}",
            exc,
        )
        return None
