"""
Wearable Data Processing and Modeling project
Google Fitness DataLoader managed-state identity and participant-status access.
"""


from __future__ import annotations
from contextlib import closing
from pathlib import Path
import sqlite3
from wearable_project.DataLoaders._contracts import (
    RootIdentity, StateManifestContract, participant_component_issue,
)
from wearable_project.DataLoaders._sources import WearableSource


def validate_root_identity(identity: RootIdentity, phase: str) -> None:
    from wearable_project.DataLoaders._base import DataLoaderStateError

    if not identity.managed:
        return
    if identity.source is not WearableSource.GOOGLE_FITNESS or not identity.verified:
        raise DataLoaderStateError(
            f"managed Google Fitness {phase} state could not establish the requested source contract: "
            f"{identity.evidence}"
        )

    state_version = identity.metadata.get("state_schema_version")
    if phase == "native":
        from wearable_project.processing.google_fitness.registry import REGISTRY_VERSION
        from wearable_project.processing.google_fitness.writer import STATE_SCHEMA_VERSION

        registry_version = identity.metadata.get("registry_version")
        if state_version != STATE_SCHEMA_VERSION:
            raise DataLoaderStateError(
                "incompatible Google Fitness native state schema: "
                f"root records {state_version!r}, installed loader requires {STATE_SCHEMA_VERSION!r}"
            )
        if registry_version != REGISTRY_VERSION:
            raise DataLoaderStateError(
                "incompatible Google Fitness native registry: "
                f"root records {registry_version!r}, installed loader requires {REGISTRY_VERSION!r}"
            )
        return

    from wearable_project.curation.google_fitness.contract import CURATED_SCHEMA_VERSION, contract_fingerprint
    from wearable_project.curation.google_fitness.state import CURATION_STATE_SCHEMA_VERSION

    expected = {
        "product_kind": "curated",
        "state_schema_version": CURATION_STATE_SCHEMA_VERSION,
        "curated_schema_version": CURATED_SCHEMA_VERSION,
        "contract_fingerprint": contract_fingerprint(),
    }
    problems = [
        f"{key}: root {identity.metadata.get(key)!r} vs installed {value!r}"
        for key, value in expected.items()
        if identity.metadata.get(key) != value
    ]
    if problems:
        raise DataLoaderStateError(
            "incompatible Google Fitness curated-root state metadata: " + "; ".join(problems)
        )


def participant_statuses(root: Path) -> dict[str, str]:
    database = root / ".wearable_state.sqlite"
    if not database.is_file():
        return {}
    uri = f"{database.resolve().as_uri()}?immutable=1"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            rows = connection.execute("SELECT participant_id, status FROM participants").fetchall()
    except sqlite3.Error as exc:
        from wearable_project.DataLoaders._base import DataLoaderStateError

        raise DataLoaderStateError(f"could not read participant statuses from {database}: {exc}") from exc

    from wearable_project.DataLoaders._base import DataLoaderStateError

    statuses: dict[str, str] = {}
    for raw_participant, raw_status in rows:
        raw_text = "" if raw_participant is None else str(raw_participant)
        participant_id = raw_text.strip()
        if raw_text != participant_id:
            raise DataLoaderStateError(
                f"{database} participants contains participant_id {raw_participant!r} with leading or trailing "
                "whitespace"
            )
        issue = participant_component_issue(participant_id)
        if issue is not None:
            raise DataLoaderStateError(
                f"{database} participants contains participant_id {raw_participant!r} that {issue}"
            )
        if participant_id in statuses:
            raise DataLoaderStateError(
                f"{database} participants contains duplicate rows for participant {participant_id!r}"
            )
        status = "" if raw_status is None else str(raw_status).strip()
        if not status:
            raise DataLoaderStateError(
                f"{database} participants contains a blank status for participant {participant_id!r}"
            )
        statuses[participant_id] = status
    return statuses


def manifest_contract(phase: str) -> StateManifestContract:
    """Google state-table contract for native lineage-rich and curated schema-v2 products."""

    if phase == "native":
        return StateManifestContract(
            table="feature_outputs",
            query="SELECT * FROM feature_outputs WHERE feature = ?",
            row_count_column="row_count",
            required_columns=(
                "participant_id", "feature", "filename", "row_count", "size_bytes", "sha256",
            ),
            nonnegative_integer_columns=("row_count", "size_bytes"),
            sha256_columns=("sha256",),
        )
    return StateManifestContract(
        table="curation_outputs",
        query="SELECT * FROM curation_outputs WHERE feature = ?",
        row_count_column="curated_rows",
        required_columns=(
            "participant_id",
            "feature",
            "filename",
            "native_rows",
            "curated_rows",
            "size_bytes",
            "sha256",
            "policy_fingerprint",
            "active_materialized_columns_json",
            "pass_rows",
            "review_rows",
            "exclude_default_rows",
            "included_by_default_rows",
            "excluded_by_default_rows",
            "canonical_value_rows",
            "flag_counts_json",
            "acquisition_counts_json",
            "derivation_counts_json",
            "tracking_counts_json",
            "step_count_role_counts_json",
            "unit_status_counts_json",
            "unit_evidence_counts_json",
        ),
        nonnegative_integer_columns=(
            "native_rows",
            "curated_rows",
            "size_bytes",
            "pass_rows",
            "review_rows",
            "exclude_default_rows",
            "included_by_default_rows",
            "excluded_by_default_rows",
            "canonical_value_rows",
        ),
        sha256_columns=("sha256",),
    )
