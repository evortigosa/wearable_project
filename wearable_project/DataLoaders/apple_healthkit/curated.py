"""
Wearable Data Processing and Modeling project
Apple HealthKit sparse-curated-frame reconstruction and state reconciliation.
"""


from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
import pandas as pd
from wearable_project.DataLoaders._contracts import CuratedFramePreparation
from wearable_project.DataLoaders.apple_healthkit.schema import (
    CURATION_FLAG_SEPARATOR, DENSE_CURATED_DEFAULTS, DENSE_FILL_BLANKS,
)


def _as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    lowered = series.astype("string").str.strip().str.lower()
    truthy = lowered.isin({"1", "true", "t", "yes", "y"}).fillna(False).to_numpy(dtype=bool)
    nonzero = numeric.fillna(0).to_numpy(dtype=float) != 0
    return pd.Series(truthy | nonzero, index=series.index)


def prepare_curated_frame(frame: pd.DataFrame, path: Path, feature: str) -> CuratedFramePreparation:
    """Restore the exact dense Apple curation schema historically exposed by ``DataLoaders._base``."""

    del path, feature  # The legacy schema is source-wide; arguments keep the adapter contract uniform.
    physical_columns = tuple(str(column) for column in frame.columns)
    frame = frame.copy()
    reconstructed_rows: dict[str, int] = {}
    reconstructed_columns: list[str] = []
    rows = int(len(frame))
    inclusion_mask = pd.Series(False, index=frame.index, dtype=bool)

    for column, default in DENSE_CURATED_DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default
            reconstructed_rows[column] = rows
            reconstructed_columns.append(column)
            if column == "include_by_default":
                inclusion_mask[:] = True
        elif column in DENSE_FILL_BLANKS:
            values = frame[column]
            blank = (
                values.isna().to_numpy(dtype=bool)
                | values.astype("string").str.strip().eq("").fillna(False).to_numpy(dtype=bool)
            )
            count = int(blank.sum())
            if count:
                frame[column] = values.astype(object).where(~blank, default)
                reconstructed_rows[column] = count
                reconstructed_columns.append(column)

    frame["include_by_default"] = _as_boolean(frame["include_by_default"]).astype(bool)
    active = tuple(column for column in DENSE_CURATED_DEFAULTS if column in physical_columns)
    return CuratedFramePreparation(
        frame=frame,
        physical_columns=physical_columns,
        active_materialized_columns=active,
        reconstructed_rows=reconstructed_rows,
        reconstructed_columns=tuple(reconstructed_columns),
        virtual_columns=(),
        inclusion_reconstructed_mask=inclusion_mask,
    )


def _integrity_problems(path: Path, entry: Mapping[str, Any]) -> list[str]:
    size = int(path.stat().st_size)
    recorded = int(entry["size_bytes"])
    if size != recorded:
        return [f"size_bytes: file {size!r} vs state {recorded!r}"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != str(entry["sha256"]):
        return [f"sha256: file {digest.hexdigest()!r} vs state {str(entry['sha256'])!r}"]
    return []


def reconcile_curated_frame(
    prepared: CuratedFramePreparation, path: Path, feature: str, entry: Mapping[str, Any],
) -> bool:
    """Check a dense Apple curated frame against the complete ``curation_outputs`` state row."""

    del feature
    from wearable_project.DataLoaders._base import DataLoaderStateError

    frame = prepared.frame
    problems: list[str] = []

    def compare(label: str, observed: Any, recorded: Any) -> None:
        if observed != recorded:
            problems.append(f"{label}: file {observed!r} vs state {recorded!r}")

    recorded_rows = int(entry["curated_rows"])
    problems.extend(_integrity_problems(path, entry))
    compare("rows", int(len(frame)), recorded_rows)
    status = frame["curation_status"]
    compare("pass rows", int((status == "pass").sum()), int(entry["pass_rows"]))
    compare("review rows", int((status == "review").sum()), int(entry["review_rows"]))
    compare(
        "exclude_default rows",
        int((status == "exclude_default").sum()),
        int(entry["exclude_default_rows"]),
    )

    included_recorded = int(entry["included_by_default_rows"])
    excluded_recorded = int(entry["excluded_by_default_rows"])
    inclusion_checked = included_recorded + excluded_recorded == recorded_rows
    if inclusion_checked:
        include = frame["include_by_default"]
        compare("included_by_default rows", int(include.sum()), included_recorded)
        compare("excluded_by_default rows", int((~include).sum()), excluded_recorded)

    acquisition = Counter(str(value) for value in frame["acquisition_method"])
    compare(
        "acquisition counts",
        dict(sorted(acquisition.items())),
        dict(sorted(_json_counter(entry, "acquisition_counts_json").items())),
    )
    flags: Counter[str] = Counter()
    for value in frame["curation_flags"]:
        flags.update(part for part in str(value).split(CURATION_FLAG_SEPARATOR) if part)
    compare(
        "flag counts",
        dict(sorted(flags.items())),
        dict(sorted(_json_counter(entry, "flag_counts_json").items())),
    )

    if problems:
        raise DataLoaderStateError(
            f"{path} disagrees with its curation_outputs row in the curation state database: "
            + "; ".join(problems)
            + ". The file may be damaged, the state database stale, or the writer version "
            "incompatible with this loader; its rows should not be used until this is resolved."
        )
    return inclusion_checked


def dense_default(feature: str, column: str, phase: str) -> Any:
    del feature
    return DENSE_CURATED_DEFAULTS.get(column) if phase == "curated" else None


def _state_int(entry: Mapping[str, Any], name: str) -> int:
    from wearable_project.DataLoaders._base import DataLoaderStateError

    try:
        return int(entry[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataLoaderStateError(
            f"Apple HealthKit curation state column {name!r} must contain an integer; "
            f"found {entry.get(name)!r}"
        ) from exc


def _json_counter(entry: Mapping[str, Any], name: str) -> Counter[str]:
    from wearable_project.DataLoaders._base import DataLoaderStateError

    raw = entry.get(name)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw or "{}", object_pairs_hook=pairs)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataLoaderStateError(
            f"Apple HealthKit curation state column {name!r} is invalid JSON counter data"
        ) from exc
    if not isinstance(payload, dict):
        raise DataLoaderStateError(
            f"Apple HealthKit curation state column {name!r} must contain a JSON object"
        )
    counts: Counter[str] = Counter()
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise DataLoaderStateError(
                f"Apple HealthKit curation state column {name!r} contains a blank counter label"
            )
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DataLoaderStateError(
                f"Apple HealthKit curation state column {name!r} must map labels to nonnegative integer counts; "
                f"found {value!r} for {key!r}"
            )
        counts[key] = value
    return counts


def _classified_fraction(counts: Counter[str]) -> float | None:
    total = sum(counts.values())
    return (total - counts.get("unclassified", 0)) / total if total else None


def curated_profile_record(entry: Mapping[str, Any], installed_fingerprint: str | None,) -> dict[str, Any]:
    """Return the exact legacy Apple per-participant curated profile fields."""

    stored = _state_int(entry, "curated_rows")
    included = _state_int(entry, "included_by_default_rows")
    excluded = _state_int(entry, "excluded_by_default_rows")
    inclusion_valid = included + excluded == stored
    acquisition = _json_counter(entry, "acquisition_counts_json")
    return {
        "pass_rows": _state_int(entry, "pass_rows"),
        "review_rows": _state_int(entry, "review_rows"),
        "exclude_default_rows": _state_int(entry, "exclude_default_rows"),
        "included_by_default_rows": included if inclusion_valid else None,
        "excluded_by_default_rows": excluded if inclusion_valid else None,
        "acquisition_classified_fraction": _classified_fraction(acquisition),
        "canonical_value_rows": _state_int(entry, "canonical_value_rows"),
        "ambiguous_unit_rows": _state_int(entry, "ambiguous_unit_rows"),
        "policy_current": (
            None if installed_fingerprint is None
            else str(entry.get("policy_fingerprint")) == installed_fingerprint
        ),
    }


def curated_profile_summary(
    entries: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...], installed_fingerprint: str | None,
) -> dict[str, Any]:
    """Aggregate the accepted Apple curated profile metrics without reading feature CSVs."""

    status: Counter[str] = Counter()
    acquisition: Counter[str] = Counter()
    unit_status: Counter[str] = Counter()
    included = excluded = unverifiable = canonical = ambiguous = diverged = 0

    for entry in entries:
        stored = _state_int(entry, "curated_rows")
        status.update({
            "pass": _state_int(entry, "pass_rows"),
            "review": _state_int(entry, "review_rows"),
            "exclude_default": _state_int(entry, "exclude_default_rows"),
        })
        row_included = _state_int(entry, "included_by_default_rows")
        row_excluded = _state_int(entry, "excluded_by_default_rows")
        if row_included + row_excluded == stored:
            included += row_included
            excluded += row_excluded
        else:
            unverifiable += 1
        acquisition.update(_json_counter(entry, "acquisition_counts_json"))
        unit_status.update(_json_counter(entry, "unit_status_counts_json"))
        canonical += _state_int(entry, "canonical_value_rows")
        ambiguous += _state_int(entry, "ambiguous_unit_rows")
        if installed_fingerprint is not None and str(entry.get("policy_fingerprint")) != installed_fingerprint:
            diverged += 1

    denominator = canonical + ambiguous
    return {
        "status_counts": dict(sorted(status.items())),
        "default_inclusion": {
            "included": included,
            "excluded": excluded,
            "unverifiable_files": unverifiable,
        },
        "acquisition": {
            "counts": dict(sorted(acquisition.items())),
            "classified_fraction": _classified_fraction(acquisition),
        },
        "unit_resolution": {
            "status_counts": dict(sorted(unit_status.items())),
            "canonical_value_rows": canonical,
            "ambiguous_unit_rows": ambiguous,
            "resolved_fraction": canonical / denominator if denominator else None,
        },
        "policy": {
            "installed_fingerprint": installed_fingerprint,
            "files_diverged": diverged,
        },
    }


def curated_profile_columns() -> dict[str, str]:
    """Nullable dtypes for the accepted Apple curated participant profile table."""

    return {
        "pass_rows": "Int64",
        "review_rows": "Int64",
        "exclude_default_rows": "Int64",
        "included_by_default_rows": "Int64",
        "excluded_by_default_rows": "Int64",
        "acquisition_classified_fraction": "Float64",
        "canonical_value_rows": "Int64",
        "ambiguous_unit_rows": "Int64",
        "policy_current": "boolean",
    }
