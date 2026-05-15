import asyncpg


async def setup_memory_schema(conn: asyncpg.Connection, *, ddl: str, schema_version: int) -> None:
    async with conn.transaction():
        await conn.execute(ddl)
        await conn.execute("ALTER TABLE patient_runs ADD COLUMN IF NOT EXISTS tenant_id TEXT")
        await conn.execute("ALTER TABLE patient_runs ADD COLUMN IF NOT EXISTS facility_id TEXT")
        await conn.execute("ALTER TABLE pipeline_audit_log ADD COLUMN IF NOT EXISTS tenant_id TEXT")
        await conn.execute("ALTER TABLE pipeline_audit_log ADD COLUMN IF NOT EXISTS facility_id TEXT")
        await conn.execute("ALTER TABLE physician_feedback ADD COLUMN IF NOT EXISTS tenant_id TEXT")
        await conn.execute("ALTER TABLE physician_feedback ADD COLUMN IF NOT EXISTS facility_id TEXT")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_patient_runs_tenant_facility ON patient_runs (tenant_id, facility_id, created_at DESC)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_audit_log_tenant_facility ON pipeline_audit_log (tenant_id, facility_id, timestamp DESC)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_physician_feedback_tenant_facility ON physician_feedback (tenant_id, facility_id, created_at DESC)")
        await conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_patient_runs_tenant_facility_profile
            ON patient_runs (tenant_id, facility_id, profile_hash)
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_physician_feedback_lookup
            ON physician_feedback (tenant_id, facility_id, profile_hash, created_at DESC)
            """
        )
        row = await conn.fetchrow("SELECT version FROM schema_version LIMIT 1")
        current_version = int(row["version"]) if row and "version" in row else 0

        if current_version < 6:
            await conn.execute("ALTER TABLE llm_cache ADD COLUMN IF NOT EXISTS prefix TEXT")
            needs_backfill = await conn.fetchval("SELECT EXISTS (SELECT 1 FROM llm_cache WHERE prefix IS NULL LIMIT 1)")
            if bool(needs_backfill):
                await conn.execute("UPDATE llm_cache SET prefix = '' WHERE prefix IS NULL")
            await conn.execute("ALTER TABLE llm_cache ALTER COLUMN prefix SET NOT NULL")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_cache_prefix ON llm_cache (prefix)")

        if current_version < 7:
            await conn.execute(
                """
                ALTER TABLE pipeline_audit_log
                ALTER COLUMN outcome_tier_counts
                TYPE JSONB
                USING
                    CASE
                        WHEN outcome_tier_counts IS NULL THEN '{}'::jsonb
                        WHEN pg_typeof(outcome_tier_counts)::text = 'jsonb' THEN outcome_tier_counts
                        ELSE outcome_tier_counts::text::jsonb
                    END
                """
            )

        if current_version < 8:
            await conn.execute(
                """
                DO $$
                BEGIN
                    ALTER TABLE patient_runs
                    ADD CONSTRAINT patient_runs_tenant_nonempty
                    CHECK (tenant_id IS NOT NULL AND tenant_id <> '') NOT VALID;
                EXCEPTION WHEN duplicate_object THEN NULL;
                END $$;
                """
            )
            await conn.execute(
                """
                DO $$
                BEGIN
                    ALTER TABLE patient_runs
                    ADD CONSTRAINT patient_runs_facility_nonempty
                    CHECK (facility_id IS NOT NULL AND facility_id <> '') NOT VALID;
                EXCEPTION WHEN duplicate_object THEN NULL;
                END $$;
                """
            )
            await conn.execute(
                """
                DO $$
                BEGIN
                    ALTER TABLE pipeline_audit_log
                    ADD CONSTRAINT pipeline_audit_tenant_nonempty
                    CHECK (tenant_id IS NOT NULL AND tenant_id <> '') NOT VALID;
                EXCEPTION WHEN duplicate_object THEN NULL;
                END $$;
                """
            )
            await conn.execute(
                """
                DO $$
                BEGIN
                    ALTER TABLE pipeline_audit_log
                    ADD CONSTRAINT pipeline_audit_facility_nonempty
                    CHECK (facility_id IS NOT NULL AND facility_id <> '') NOT VALID;
                EXCEPTION WHEN duplicate_object THEN NULL;
                END $$;
                """
            )
            await conn.execute(
                """
                DO $$
                BEGIN
                    ALTER TABLE physician_feedback
                    ADD CONSTRAINT physician_feedback_tenant_nonempty
                    CHECK (tenant_id IS NOT NULL AND tenant_id <> '') NOT VALID;
                EXCEPTION WHEN duplicate_object THEN NULL;
                END $$;
                """
            )
            await conn.execute(
                """
                DO $$
                BEGIN
                    ALTER TABLE physician_feedback
                    ADD CONSTRAINT physician_feedback_facility_nonempty
                    CHECK (facility_id IS NOT NULL AND facility_id <> '') NOT VALID;
                EXCEPTION WHEN duplicate_object THEN NULL;
                END $$;
                """
            )
            await conn.execute(
                """
                DO $$
                BEGIN
                    ALTER TABLE physician_feedback
                    ADD CONSTRAINT physician_feedback_verdict_check
                    CHECK (verdict IN ('confirmed', 'rejected')) NOT VALID;
                EXCEPTION WHEN duplicate_object THEN NULL;
                END $$;
                """
            )

        if row is None:
            await conn.execute("INSERT INTO schema_version (version) VALUES ($1)", schema_version)
        elif current_version < schema_version:
            await conn.execute("UPDATE schema_version SET version = $1", schema_version)
