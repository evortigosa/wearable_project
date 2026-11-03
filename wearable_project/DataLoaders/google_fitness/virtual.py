"""
Wearable Data Processing and Modeling project
Deterministic Google Fitness curated-schema virtual projections. The curated CSV stores the unchanged native prefix
plus a small set of reviewed materialized decisions. This module validates those materialized provenance/group
decisions against native evidence and reconstructs the contract-defined virtual columns. It performs no file or
state I/O; root-contract and sparse-storage handling in :mod:`wearable_project.DataLoaders.google_fitness.curated`.
"""


from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from wearable_project.curation.google_fitness.context import build_step_count_context
from wearable_project.curation.google_fitness.models import StepCountRole
from wearable_project.curation.google_fitness.registry import get_policy
from wearable_project.exceptions import DataLoaderReadError
from wearable_project.processing.google_fitness.registry import get_feature_spec


_STEP_ROLE_VALUES = frozenset(item.value for item in StepCountRole)


def _text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def _is_google_identifier(values: pd.Series) -> pd.Series:
    text = _text(values).str.casefold()
    return (
        text.str.startswith("com.google.")
        | text.str.startswith("derived:com.google.")
        | text.str.contains("google fitness", regex=False)
        | text.str.contains("google fit", regex=False)
    ) & text.ne("")


def _expected_acquisition(frame: pd.DataFrame, feature: str) -> pd.Series:
    """Vectorized equivalent of the reviewed provenance classifier's acquisition dimension."""

    count = len(frame)
    expected = pd.Series(np.full(count, "unknown", dtype=object), index=frame.index, dtype=object)
    spec = get_feature_spec(feature)
    family = spec.family.value
    if family in {"activity_interval", "sleep_state_interval"}:
        # Three accepted activity schemas intentionally omit source_id. The curation provenance classifier
        # treats an absent mapping key exactly like an empty value, so preserve that behavior rather than
        # requiring a synthetic column.
        source_values = (
            frame["source_id"]
            if "source_id" in frame.columns
            else pd.Series("", index=frame.index, dtype="string")
        )
        source_id = _text(source_values)
        google = _is_google_identifier(source_values)
        third_party = source_id.ne("") & ~google
        device_evidence = _text(frame["device"]).ne("") | _text(frame["source_name"]).ne("")
        expected.loc[third_party] = "third_party_import"
        expected.loc[~third_party & device_evidence] = "device_estimate"
        expected.loc[~third_party & ~device_evidence & google] = "health_import_unknown"
        return expected
    if feature == "StepCount":
        versions = _text(frame["collecting_method_version"])
        legacy = versions.eq("1.0")
        expected.loc[legacy] = "health_import_unknown"
        modern = versions.eq("2.0")
        package = _text(frame["app_package_name"])
        package_google = _is_google_identifier(frame["app_package_name"])
        third_party = modern & package.ne("") & ~package_google
        device_evidence = pd.Series(False, index=frame.index)
        for name in ("device_manufacturer", "device_model", "device_type", "device_uid"):
            device_evidence |= _text(frame[name]).ne("")
        expected.loc[modern & ~third_party] = "health_import_unknown"
        expected.loc[modern & ~third_party & device_evidence] = "device_estimate"
        expected.loc[third_party] = "third_party_import"
        return expected
    if feature in {"Weight", "Height"}:
        added = _text(frame["added_by"])
        google = _is_google_identifier(frame["added_by"])
        expected.loc[added.ne("") & ~google] = "third_party_import"
        expected.loc[added.ne("") & google] = "health_import_unknown"
        return expected
    # Calories and Distance are application-level summaries with unknown acquisition.
    return expected


def validate_acquisition(frame: pd.DataFrame, path: Path, feature: str) -> None:
    """Verify the stored acquisition decision against the unchanged native prefix."""

    actual = _text(frame["acquisition_method"])
    expected = _expected_acquisition(frame, feature).astype("string")
    bad = actual.ne(expected)
    if bool(bad.any()):
        row = int(bad[bad].index[0])
        raise DataLoaderReadError(
            f"{path}: acquisition_method at data row {row + 1} is {actual.loc[row]!r}, but the validated "
            f"schema-v2 contract reconstructs {expected.loc[row]!r} from the unchanged native prefix"
        )


def _records_as_text(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for values in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        records.append({
            column: "" if pd.isna(value) else str(value)
            for column, value in zip(columns, values)
        })
    return records


def validate_step_count_roles(frame: pd.DataFrame, path: Path) -> None:
    """Verify every stored StepCount role against the complete alternative group."""

    actual = _text(frame["step_count_role"])
    invalid = ~actual.isin(_STEP_ROLE_VALUES)
    if bool(invalid.any()):
        row = int(invalid[invalid].index[0])
        raise DataLoaderReadError(
            f"{path}: unrecognized step_count_role {actual.loc[row]!r} at data row {row + 1}"
        )
    native_columns = get_feature_spec("StepCount").column_names
    contexts = build_step_count_context(_records_as_text(frame, native_columns))
    expected = pd.Series(
        [contexts[index].role.value for index in range(len(frame))],
        index=frame.index,
        dtype="string",
    )
    bad = actual.ne(expected)
    if bool(bad.any()):
        row = int(bad[bad].index[0])
        raise DataLoaderReadError(
            f"{path}: step_count_role at data row {row + 1} is {actual.loc[row]!r}, but the validated "
            f"group contract reconstructs {expected.loc[row]!r}"
        )


def materialize_virtual(frame: pd.DataFrame, feature: str) -> dict[str, pd.Series]:
    """Vectorized implementation of the accepted contract's deterministic virtual row values."""

    policy = get_policy(feature, allow_fallback=False)
    rows = len(frame)
    values: dict[str, pd.Series] = {}
    spec = get_feature_spec(feature)
    family = spec.family.value

    if family in {"activity_interval", "sleep_state_interval"}:
        values["derivation_method"] = pd.Series(
            np.full(rows, "activity_classification", dtype=object), index=frame.index,
        )
        tracked = _text(frame["tracked"]).str.casefold()
        status = pd.Series(np.full(rows, "unknown", dtype=object), index=frame.index)
        status.loc[tracked.isin({"1", "1.0", "true", "t", "yes", "y"})] = "tracked"
        status.loc[tracked.isin({"0", "0.0", "false", "f", "no", "n"})] = "untracked"
        values["tracking_status"] = status
    elif feature in {"Calories", "Distance", "Weight", "Height"}:
        proposed = {"Calories": "kcal", "Distance": "m", "Weight": "kg", "Height": "m"}[feature]
        values["canonical_value"] = pd.Series(
            pd.array([pd.NA] * rows, dtype="Float64"), index=frame.index,
        )
        values["canonical_unit"] = pd.Series([pd.NA] * rows, index=frame.index, dtype=object)
        values["curation_unit_status"] = pd.Series(
            np.full(rows, "withheld_exporter_mapping_pending", dtype=object), index=frame.index,
        )
        values["unit_evidence"] = pd.Series(
            np.full(rows, f"exporter_mapping_pending:proposed_{proposed}", dtype=object), index=frame.index,
        )
        derivation = "raw" if feature in {"Weight", "Height"} else "application_summary"
        values["derivation_method"] = pd.Series(
            np.full(rows, derivation, dtype=object), index=frame.index,
        )
    elif feature == "StepCount":
        versions = _text(frame["collecting_method_version"])
        numeric = pd.to_numeric(frame["value"], errors="coerce")
        finite_nonnegative = np.isfinite(numeric.to_numpy(dtype=float)) & numeric.ge(0).to_numpy(dtype=bool)
        # Keep a NumPy float representation until participant frames are concatenated. The shared engine then
        # applies the declared nullable-Int64 dtype once, avoiding pandas extension-dtype union ambiguity while
        # preserving every whole-number count and missing value.
        canonical = pd.Series(np.full(rows, np.nan, dtype=float), index=frame.index)
        if bool(finite_nonnegative.any()):
            canonical.loc[finite_nonnegative] = numeric.loc[finite_nonnegative].astype("int64")
        values["canonical_value"] = canonical
        values["canonical_unit"] = pd.Series(np.full(rows, "count", dtype=object), index=frame.index)
        unit_status = pd.Series(
            np.full(rows, "unresolved_collecting_method_version", dtype=object), index=frame.index,
        )
        unit_evidence = pd.Series(
            np.full(rows, "unsupported_collecting_method_version", dtype=object), index=frame.index,
        )
        v1 = versions.eq("1.0")
        v2 = versions.eq("2.0")
        unit_status.loc[v1] = "resolved_source_convention"
        unit_evidence.loc[v1] = "source_convention:google_fitness_step_count_v1"
        unit_status.loc[v2] = "resolved_payload_explicit"
        unit_evidence.loc[v2] = "payload_explicit:com.google.step_count.delta"
        values["curation_unit_status"] = unit_status
        values["unit_evidence"] = unit_evidence

        derivation = pd.Series(np.full(rows, "unknown", dtype=object), index=frame.index)
        derivation.loc[v1] = "legacy_summary"
        aggregation = _text(frame["aggregation_source"])
        folded = aggregation.str.casefold()
        estimated = v2 & folded.str.contains("estimated_steps", regex=False)
        merged = v2 & folded.str.contains("merge_step_deltas", regex=False)
        other = v2 & aggregation.ne("") & ~estimated & ~merged
        derivation.loc[estimated] = "estimated"
        derivation.loc[merged] = "merged"
        derivation.loc[other] = "third_party_aggregate"
        values["derivation_method"] = derivation

    result: dict[str, pd.Series] = {}
    for column in policy.output.virtual_columns:
        if column in values:
            result[column] = values[column]
        elif column == "canonical_value":
            result[column] = pd.Series(pd.array([pd.NA] * rows, dtype="Float64"), index=frame.index)
        else:
            result[column] = pd.Series([pd.NA] * rows, index=frame.index, dtype=object)
    return result


__all__ = ["materialize_virtual", "validate_acquisition", "validate_step_count_roles"]
