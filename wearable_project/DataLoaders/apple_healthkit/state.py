"""
Wearable Data Processing and Modeling project
Apple HealthKit managed-root identity checks used independently of optional file-state verification.
"""


from __future__ import annotations
from wearable_project.DataLoaders._contracts import RootIdentity, StateManifestContract
from wearable_project.DataLoaders._sources import WearableSource


def validate_root_identity(identity: RootIdentity, phase: str) -> None:
    """
    Reject a managed state database whose source cannot be established.
    Legacy Apple products predate ``source_key`` metadata, so their recognized table signatures are accepted.
    An absent state database remains an unmanaged trimmed tree and is allowed under the existing ``auto``/``off``
    behavior; a present but unreadable, active, or unknown state database is not silently downgraded to unmanaged.
    """

    if not identity.managed:
        return
    if not identity.verified or identity.source is not WearableSource.APPLE_HEALTHKIT:
        from wearable_project.DataLoaders._base import DataLoaderStateError

        raise DataLoaderStateError(
            f"managed Apple HealthKit {phase} state could not establish the requested source contract: "
            f"{identity.evidence}"
        )


def manifest_contract(phase: str) -> StateManifestContract:
    """Legacy Apple state-table contract, preserving the exact accepted query surface."""

    if phase == "native":
        return StateManifestContract(
            table="feature_outputs",
            query=(
                "SELECT participant_id, row_count, size_bytes, sha256 "
                "FROM feature_outputs WHERE feature = ?"
            ),
            row_count_column="row_count",
            required_columns=("participant_id", "feature", "row_count", "size_bytes", "sha256"),
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
            "curated_rows",
            "size_bytes",
            "sha256",
            "policy_fingerprint",
            "pass_rows",
            "review_rows",
            "exclude_default_rows",
            "included_by_default_rows",
            "excluded_by_default_rows",
            "canonical_value_rows",
            "ambiguous_unit_rows",
            "flag_counts_json",
            "acquisition_counts_json",
            "unit_status_counts_json",
        ),
        nonnegative_integer_columns=(
            "curated_rows",
            "size_bytes",
            "pass_rows",
            "review_rows",
            "exclude_default_rows",
            "included_by_default_rows",
            "excluded_by_default_rows",
            "canonical_value_rows",
            "ambiguous_unit_rows",
        ),
        sha256_columns=("sha256",),
    )
