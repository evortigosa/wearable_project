"""
Wearable Data Processing and Modeling project
Google Fitness curated schema-v2 validation, sparse defaults, and virtual projections.
The curated CSV is deliberately not a denormalized analytical table. It stores the complete native prefix,
a small ordered subset of materialized curation columns, and leaves deterministic feature/version projections in
the root ``.curation_contract.json``. This module validates that root contract, restores sparse defaults without
changing native evidence, reconstructs the virtual columns, and reconciles the resulting logical file with the
curation state database.
"""


from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from wearable_project.DataLoaders._contracts import CuratedFramePreparation
from wearable_project.curation.google_fitness.contract import (
    CONTRACT_SCHEMA_VERSION, CURATED_SCHEMA_VERSION, CURATION_CONTRACT_FILENAME, MATERIALIZED_COLUMN_ORDER,
    ROW_DEFAULTS, VIRTUAL_COLUMN_ORDER, contract_definition, contract_fingerprint,
)
from wearable_project.curation.google_fitness.engine import feature_policy_fingerprint
from wearable_project.curation.google_fitness.models import CurationStatus
from wearable_project.curation.google_fitness.registry import get_policy
from wearable_project.curation.google_fitness.state import CURATION_STATE_SCHEMA_VERSION
from wearable_project.exceptions import DataLoaderReadError
from wearable_project.processing.google_fitness.registry import get_feature_spec


_CURATED_BOOLEAN_TEXT: dict[str, bool] = {
    "true": True, "t": True, "yes": True, "y": True, "1": True, "1.0": True,
    "false": False, "f": False, "no": False, "n": False, "0": False, "0.0": False,
}
_STATUS_VALUES = frozenset(item.value for item in CurationStatus)


@dataclass(frozen=True, slots=True)
class RootContract:
    """Validated root-level scientific contract and its file identity."""

    path: Path
    sha256: str
    payload: Mapping[str, Any]

    @property
    def metadata(self) -> dict[str, Any]:
        implementation = self.payload.get("implementation")
        implementation = implementation if isinstance(implementation, Mapping) else {}
        return {
            "contract_file": str(self.path),
            "contract_file_sha256": self.sha256,
            "contract_schema_version": self.payload.get("contract_schema_version"),
            "curated_schema_version": self.payload.get("curated_schema_version"),
            "curation_registry_version": self.payload.get("curation_registry_version"),
            "policy_contract_version": self.payload.get("policy_contract_version"),
            "contract_fingerprint": self.payload.get("contract_fingerprint"),
            "curation_engine_version": implementation.get("engine_version"),
            "curation_pipeline_version": implementation.get("pipeline_version"),
            "curation_state_schema_version": implementation.get("state_schema_version"),
        }


def _json_object_no_duplicates(text: str, path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise DataLoaderReadError(f"{path}: curation contract repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(text, object_pairs_hook=pairs)
    except DataLoaderReadError:
        raise
    except json.JSONDecodeError as exc:
        raise DataLoaderReadError(f"invalid Google Fitness curation contract {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataLoaderReadError(f"Google Fitness curation contract must be a JSON object: {path}")
    return payload


def validate_root_contract(root: Path) -> RootContract:
    """Require the exact installed scientific schema-v2 contract before any curated CSV is interpreted."""

    path = Path(root) / CURATION_CONTRACT_FILENAME
    if not path.is_file():
        from wearable_project.DataLoaders._base import DataLoaderStateError

        raise DataLoaderStateError(
            f"Google Fitness curated loading requires {CURATION_CONTRACT_FILENAME} at the root, but it is absent: "
            f"{path}. Virtual projections and sparse defaults cannot be interpreted safely without it."
        )
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        from wearable_project.DataLoaders._base import DataLoaderStateError

        raise DataLoaderStateError(f"could not read Google Fitness curation contract {path}: {exc}") from exc
    try:
        payload = _json_object_no_duplicates(text, path)
    except DataLoaderReadError as exc:
        from wearable_project.DataLoaders._base import DataLoaderStateError

        raise DataLoaderStateError(str(exc)) from exc

    expected = contract_definition()
    expected_keys = set(expected) | {"contract_fingerprint", "implementation"}
    observed_keys = set(payload)
    issues: list[str] = []
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        if missing:
            issues.append("missing field(s): " + ", ".join(missing))
        if extra:
            issues.append("unexpected field(s): " + ", ".join(extra))
    for key, value in expected.items():
        if payload.get(key) != value:
            issues.append(f"contract field differs: {key}")
    installed_fingerprint = contract_fingerprint()
    if payload.get("contract_fingerprint") != installed_fingerprint:
        issues.append(
            "contract fingerprint differs: "
            f"root {payload.get('contract_fingerprint')!r} vs installed {installed_fingerprint!r}"
        )
    implementation = payload.get("implementation")
    if not isinstance(implementation, Mapping):
        issues.append("implementation must be an object")
    else:
        if implementation.get("state_schema_version") != CURATION_STATE_SCHEMA_VERSION:
            issues.append(
                "implementation state schema differs: "
                f"root {implementation.get('state_schema_version')!r} vs installed "
                f"{CURATION_STATE_SCHEMA_VERSION!r}"
            )
        for name in ("engine_version", "pipeline_version"):
            if not isinstance(implementation.get(name), str) or not str(implementation.get(name)).strip():
                issues.append(f"implementation {name} is missing or blank")
    if issues:
        from wearable_project.DataLoaders._base import DataLoaderStateError

        raise DataLoaderStateError(
            f"{path} is incompatible with the installed Google Fitness curated schema-v2 loader: "
            + "; ".join(issues)
            + ". Preserve that root and use the package version that wrote it, or rebuild into a new curated root."
        )
    return RootContract(path=path, sha256=sha256(raw).hexdigest(), payload=payload)


def _blank(values: pd.Series) -> pd.Series:
    return values.isna() | values.astype("string").str.strip().eq("").fillna(False)


def _text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip()


def _as_boolean(values: pd.Series, path: Path, column: str) -> pd.Series:
    result: list[bool] = []
    for index, value in values.items():
        if isinstance(value, (bool, np.bool_)):
            result.append(bool(value))
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
            result.append(bool(value))
            continue
        token = str(value).strip().lower()
        if token not in _CURATED_BOOLEAN_TEXT:
            raise DataLoaderReadError(
                f"{path}: column {column!r} must hold true/false flags; data row {int(index) + 1} "
                f"holds {value!r}"
            )
        result.append(_CURATED_BOOLEAN_TEXT[token])
    return pd.Series(result, index=values.index, dtype=bool, name=values.name)


def prepare_curated_frame(frame: pd.DataFrame, path: Path, feature: str) -> CuratedFramePreparation:
    """Validate the physical schema, restore sparse defaults, and append contract-defined virtual columns."""

    policy = get_policy(feature, allow_fallback=False)
    spec = get_feature_spec(feature)
    physical_columns = tuple(str(column) for column in frame.columns)
    native_columns = tuple(spec.column_names)
    actual_prefix = physical_columns[: len(native_columns)]
    if actual_prefix != native_columns:
        raise DataLoaderReadError(
            f"{path} does not preserve the registered Google Fitness native prefix for {feature}: "
            f"expected {list(native_columns)!r}, found {list(actual_prefix)!r}"
        )
    active = physical_columns[len(native_columns) :]
    allowed_materialized = tuple(policy.output.materialized_columns)
    unknown = [column for column in active if column not in allowed_materialized]
    if unknown:
        raise DataLoaderReadError(
            f"{path} contains column(s) outside the curated schema-v2 materialized contract: "
            + ", ".join(unknown)
        )
    expected_active_order = tuple(column for column in allowed_materialized if column in active)
    if active != expected_active_order:
        raise DataLoaderReadError(
            f"{path} does not preserve curated materialized-column order: expected "
            f"{list(expected_active_order)!r}, found {list(active)!r}"
        )
    missing_fixed = [column for column in policy.output.fixed_columns if column not in active]
    if missing_fixed:
        raise DataLoaderReadError(
            f"{path} is missing fixed curated schema-v2 column(s): {', '.join(missing_fixed)}"
        )

    frame = frame.copy()
    reconstructed_rows: dict[str, int] = {}
    reconstructed_columns: list[str] = []
    inclusion_mask = pd.Series(False, index=frame.index, dtype=bool)
    for column in policy.output.sparse_columns:
        default = ROW_DEFAULTS[column]
        if column not in frame.columns:
            frame[column] = "" if column == "curation_flags" else default
            reconstructed_rows[column] = int(len(frame))
            reconstructed_columns.append(column)
            if column == "include_by_default":
                inclusion_mask[:] = True
            continue
        blanks = _blank(frame[column])
        count = int(blanks.sum())
        if count:
            fill = "" if column == "curation_flags" else default
            frame[column] = frame[column].astype(object).where(~blanks, fill)
            reconstructed_rows[column] = count
            reconstructed_columns.append(column)
            if column == "include_by_default":
                inclusion_mask |= blanks

    frame["include_by_default"] = _as_boolean(frame["include_by_default"], path, "include_by_default")
    statuses = _text(frame["curation_status"])
    invalid_status = ~statuses.isin(_STATUS_VALUES)
    if bool(invalid_status.any()):
        row = int(invalid_status[invalid_status].index[0])
        raise DataLoaderReadError(
            f"{path}: unrecognized curation_status {statuses.loc[row]!r} at data row {row + 1}"
        )
    frame["curation_status"] = statuses
    frame["curation_flags"] = frame["curation_flags"].fillna("").astype(str)

    from wearable_project.DataLoaders.google_fitness.virtual import (
        materialize_virtual, validate_acquisition, validate_step_count_roles,
    )

    validate_acquisition(frame, path, feature)
    if feature == "StepCount":
        validate_step_count_roles(frame, path)

    virtual_values = materialize_virtual(frame, feature)
    for column in policy.output.virtual_columns:
        frame[column] = virtual_values[column]

    logical_order = [
        *native_columns,
        *policy.output.materialized_columns,
        *policy.output.virtual_columns,
    ]
    frame = frame.loc[:, logical_order]
    return CuratedFramePreparation(
        frame=frame,
        physical_columns=physical_columns,
        active_materialized_columns=active,
        reconstructed_rows=reconstructed_rows,
        reconstructed_columns=tuple(reconstructed_columns),
        virtual_columns=tuple(policy.output.virtual_columns),
        inclusion_reconstructed_mask=inclusion_mask,
    )


def _counter(values: pd.Series, *, default: str | None = None) -> dict[str, int]:
    if default is not None and values.empty:
        return {}
    text = values.astype("string").fillna("").str.strip()
    if default is not None:
        text = text.where(text.ne(""), default)
    else:
        text = text[text.ne("")]
    return dict(sorted(Counter(str(value) for value in text).items()))


def _flag_counts(values: pd.Series) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values.fillna(""):
        counts.update(item for item in str(value).split(";") if item)
    return dict(sorted(counts.items()))


def _json_counter(entry: Mapping[str, Any], name: str) -> dict[str, int]:
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
        from wearable_project.DataLoaders._base import DataLoaderStateError

        raise DataLoaderStateError(f"curation state column {name!r} is invalid JSON counter data: {raw!r}") from exc
    if not isinstance(payload, dict):
        from wearable_project.DataLoaders._base import DataLoaderStateError

        raise DataLoaderStateError(f"curation state column {name!r} must hold a JSON object")
    result: dict[str, int] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            from wearable_project.DataLoaders._base import DataLoaderStateError

            raise DataLoaderStateError(f"curation state column {name!r} contains a blank counter label")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            from wearable_project.DataLoaders._base import DataLoaderStateError

            raise DataLoaderStateError(
                f"curation state column {name!r} must map labels to nonnegative integer counts; "
                f"found {value!r} for {key!r}"
            )
        result[key] = value
    return dict(sorted(result.items()))


def _file_integrity(path: Path, entry: Mapping[str, Any]) -> list[str]:
    size = int(path.stat().st_size)
    recorded_size = int(entry["size_bytes"])
    if size != recorded_size:
        return [f"size_bytes: file {size!r} vs state {recorded_size!r}"]
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    observed = digest.hexdigest()
    recorded = str(entry["sha256"])
    return [] if observed == recorded else [f"sha256: file {observed!r} vs state {recorded!r}"]


def reconcile_curated_frame(
    prepared: CuratedFramePreparation, path: Path, feature: str, entry: Mapping[str, Any],
) -> bool:
    """Reconcile physical bytes and every persisted logical counter with curation state."""

    from wearable_project.DataLoaders._base import DataLoaderStateError

    frame = prepared.frame
    policy = get_policy(feature, allow_fallback=False)
    problems = _file_integrity(path, entry)

    def compare(label: str, observed: Any, recorded: Any) -> None:
        if observed != recorded:
            problems.append(f"{label}: file {observed!r} vs state {recorded!r}")

    compare("filename", path.name, str(entry["filename"]))
    compare("native rows", int(len(frame)), int(entry["native_rows"]))
    compare("curated rows", int(len(frame)), int(entry["curated_rows"]))
    compare(
        "policy fingerprint",
        feature_policy_fingerprint(policy),
        str(entry["policy_fingerprint"]),
    )
    try:
        active_recorded = tuple(json.loads(entry.get("active_materialized_columns_json") or "[]"))
    except (TypeError, json.JSONDecodeError) as exc:
        raise DataLoaderStateError(
            f"{path}: active_materialized_columns_json in curation state is invalid"
        ) from exc
    compare("active materialized columns", prepared.active_materialized_columns, active_recorded)

    status = frame["curation_status"].astype("string")
    compare("pass rows", int(status.eq("pass").sum()), int(entry["pass_rows"]))
    compare("review rows", int(status.eq("review").sum()), int(entry["review_rows"]))
    compare(
        "exclude_default rows",
        int(status.eq("exclude_default").sum()),
        int(entry["exclude_default_rows"]),
    )
    include = frame["include_by_default"].astype(bool)
    included = int(include.sum())
    excluded = int((~include).sum())
    compare("included_by_default rows", included, int(entry["included_by_default_rows"]))
    compare("excluded_by_default rows", excluded, int(entry["excluded_by_default_rows"]))
    compare("default inclusion accounting", included + excluded, int(len(frame)))

    compare("flag counts", _flag_counts(frame["curation_flags"]), _json_counter(entry, "flag_counts_json"))
    compare(
        "acquisition counts",
        _counter(frame["acquisition_method"], default="unclassified"),
        _json_counter(entry, "acquisition_counts_json"),
    )
    derivation = (
        _counter(frame["derivation_method"], default="unclassified")
        if "derivation_method" in frame else {"unclassified": int(len(frame))}
    )
    tracking = (
        _counter(frame["tracking_status"], default="not_applicable")
        if "tracking_status" in frame else {"not_applicable": int(len(frame))}
    )
    roles = (
        _counter(frame["step_count_role"], default="not_applicable")
        if "step_count_role" in frame else {"not_applicable": int(len(frame))}
    )
    unit_status = (
        _counter(frame["curation_unit_status"], default="not_applicable")
        if "curation_unit_status" in frame else {"not_applicable": int(len(frame))}
    )
    unit_evidence = (
        _counter(frame["unit_evidence"], default="none")
        if "unit_evidence" in frame else {"none": int(len(frame))}
    )
    compare("derivation counts", derivation, _json_counter(entry, "derivation_counts_json"))
    compare("tracking counts", tracking, _json_counter(entry, "tracking_counts_json"))
    compare("step-count role counts", roles, _json_counter(entry, "step_count_role_counts_json"))
    compare("unit-status counts", unit_status, _json_counter(entry, "unit_status_counts_json"))
    compare("unit-evidence counts", unit_evidence, _json_counter(entry, "unit_evidence_counts_json"))
    canonical_rows = int(frame["canonical_value"].notna().sum()) if "canonical_value" in frame else 0
    compare("canonical-value rows", canonical_rows, int(entry["canonical_value_rows"]))

    if problems:
        raise DataLoaderStateError(
            f"{path} disagrees with its Google Fitness curation_outputs row: "
            + "; ".join(problems)
            + ". The file may be damaged, state may be stale, or the installed schema-v2 contract may be "
            "incompatible; its rows should not be used until this is resolved."
        )
    return True


def _state_int(entry: Mapping[str, Any], name: str) -> int:
    from wearable_project.DataLoaders._base import DataLoaderStateError

    try:
        return int(entry[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataLoaderStateError(
            f"Google Fitness curation state column {name!r} must contain an integer; "
            f"found {entry.get(name)!r}"
        ) from exc


def _active_materialized_columns(entry: Mapping[str, Any]) -> tuple[str, ...]:
    from wearable_project.DataLoaders._base import DataLoaderStateError

    raw = entry.get("active_materialized_columns_json")
    try:
        payload = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError) as exc:
        raise DataLoaderStateError(
            "Google Fitness curation state column 'active_materialized_columns_json' is invalid JSON"
        ) from exc
    if not isinstance(payload, list) or any(not isinstance(value, str) for value in payload):
        raise DataLoaderStateError(
            "Google Fitness curation state column 'active_materialized_columns_json' must contain a JSON string list"
        )
    observed = tuple(payload)
    expected = tuple(column for column in MATERIALIZED_COLUMN_ORDER if column in observed)
    if observed != expected or len(observed) != len(set(observed)):
        raise DataLoaderStateError(
            "Google Fitness curation state records invalid materialized-column order: "
            f"{list(observed)!r}"
        )
    return observed


def curated_profile_record(entry: Mapping[str, Any], installed_fingerprint: str | None,) -> dict[str, Any]:
    """Stable per-participant fields for a zero-CSV curated profile."""

    stored = _state_int(entry, "curated_rows")
    included = _state_int(entry, "included_by_default_rows")
    excluded = _state_int(entry, "excluded_by_default_rows")
    inclusion_valid = included >= 0 and excluded >= 0 and included + excluded == stored
    acquisition = _json_counter(entry, "acquisition_counts_json")
    acquisition_total = sum(acquisition.values())
    classified_fraction = (
        (acquisition_total - acquisition.get("unclassified", 0)) / acquisition_total
        if acquisition_total else None
    )
    _active_materialized_columns(entry)  # Validate even though the compact participant table omits the list.
    return {
        "pass_rows": _state_int(entry, "pass_rows"),
        "review_rows": _state_int(entry, "review_rows"),
        "exclude_default_rows": _state_int(entry, "exclude_default_rows"),
        "included_by_default_rows": included if inclusion_valid else None,
        "excluded_by_default_rows": excluded if inclusion_valid else None,
        "acquisition_classified_fraction": classified_fraction,
        "canonical_value_rows": _state_int(entry, "canonical_value_rows"),
        "policy_current": (
            None if installed_fingerprint is None
            else str(entry.get("policy_fingerprint")) == installed_fingerprint
        ),
    }


def curated_profile_summary(entries: Sequence[Mapping[str, Any]], installed_fingerprint: str | None,) -> dict[str, Any]:
    """Aggregate every curated-schema-v2 state metric without parsing feature CSVs."""

    status: Counter[str] = Counter()
    acquisition: Counter[str] = Counter()
    derivation: Counter[str] = Counter()
    tracking: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    unit_status: Counter[str] = Counter()
    unit_evidence: Counter[str] = Counter()
    materialized_file_counts: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    included = excluded = unverifiable = canonical = diverged = 0

    for entry in entries:
        stored = _state_int(entry, "curated_rows")
        pass_rows = _state_int(entry, "pass_rows")
        review_rows = _state_int(entry, "review_rows")
        exclude_rows = _state_int(entry, "exclude_default_rows")
        status.update({"pass": pass_rows, "review": review_rows, "exclude_default": exclude_rows})

        row_included = _state_int(entry, "included_by_default_rows")
        row_excluded = _state_int(entry, "excluded_by_default_rows")
        if row_included >= 0 and row_excluded >= 0 and row_included + row_excluded == stored:
            included += row_included
            excluded += row_excluded
        else:
            unverifiable += 1

        acquisition.update(_json_counter(entry, "acquisition_counts_json"))
        derivation.update(_json_counter(entry, "derivation_counts_json"))
        tracking.update(_json_counter(entry, "tracking_counts_json"))
        roles.update(_json_counter(entry, "step_count_role_counts_json"))
        unit_status.update(_json_counter(entry, "unit_status_counts_json"))
        unit_evidence.update(_json_counter(entry, "unit_evidence_counts_json"))
        flags.update(_json_counter(entry, "flag_counts_json"))
        materialized_file_counts.update(_active_materialized_columns(entry))
        canonical += _state_int(entry, "canonical_value_rows")
        if installed_fingerprint is not None and str(entry.get("policy_fingerprint")) != installed_fingerprint:
            diverged += 1

    acquisition_rows = sum(acquisition.values())
    acquisition_fraction = (
        (acquisition_rows - acquisition.get("unclassified", 0)) / acquisition_rows
        if acquisition_rows else None
    )
    resolved = sum(count for name, count in unit_status.items() if name.startswith("resolved"))
    withheld = sum(count for name, count in unit_status.items() if name.startswith("withheld"))
    unresolved = sum(count for name, count in unit_status.items() if name.startswith("unresolved"))
    applicable = resolved + withheld + unresolved
    return {
        "status_counts": dict(sorted(status.items())),
        "default_inclusion": {
            "included": included,
            "excluded": excluded,
            "unverifiable_files": unverifiable,
        },
        "acquisition": {
            "counts": dict(sorted(acquisition.items())),
            "classified_fraction": acquisition_fraction,
        },
        "unit_resolution": {
            "status_counts": dict(sorted(unit_status.items())),
            "evidence_counts": dict(sorted(unit_evidence.items())),
            "canonical_value_rows": canonical,
            "applicable_rows": applicable,
            "resolved_rows": resolved,
            "withheld_rows": withheld,
            "unresolved_rows": unresolved,
            "resolved_fraction": resolved / applicable if applicable else None,
        },
        "provenance": {
            "derivation_method_counts": dict(sorted(derivation.items())),
            "tracking_status_counts": dict(sorted(tracking.items())),
        },
        "step_count_roles": dict(sorted(roles.items())),
        "curation_flags": dict(sorted(flags.items())),
        "materialized_columns": {
            "file_counts": dict(sorted(materialized_file_counts.items())),
            "files": len(entries),
        },
        "policy": {
            "installed_fingerprint": installed_fingerprint,
            "files_diverged": diverged,
        },
    }


def curated_profile_columns() -> dict[str, str]:
    """Nullable dtypes appended to the shared participant profile table."""

    return {
        "pass_rows": "Int64",
        "review_rows": "Int64",
        "exclude_default_rows": "Int64",
        "included_by_default_rows": "Int64",
        "excluded_by_default_rows": "Int64",
        "acquisition_classified_fraction": "Float64",
        "canonical_value_rows": "Int64",
        "policy_current": "boolean",
    }

def installed_policy_fingerprint(feature: str) -> str:
    return feature_policy_fingerprint(get_policy(feature, allow_fallback=False))


def dense_default(feature: str, column: str) -> Any:
    policy = get_policy(feature, allow_fallback=False)
    if column not in policy.output.sparse_columns:
        return None
    default = ROW_DEFAULTS[column]
    return "" if column == "curation_flags" else bool(default) if column == "include_by_default" else default


def logical_counts(frame: pd.DataFrame) -> dict[str, Any]:
    """Counts over returned curated rows for load reports, including not-applicable dimensions."""

    rows = int(len(frame))
    return {
        "curation_status_counts": _counter(frame["curation_status"], default="pass"),
        "step_count_role_counts": (
            _counter(frame["step_count_role"], default="not_applicable")
            if "step_count_role" in frame else {"not_applicable": rows}
        ),
        "derivation_method_counts": (
            _counter(frame["derivation_method"], default="unclassified")
            if "derivation_method" in frame else {"unclassified": rows}
        ),
        "tracking_status_counts": (
            _counter(frame["tracking_status"], default="not_applicable")
            if "tracking_status" in frame else {"not_applicable": rows}
        ),
        "curation_unit_status_counts": (
            _counter(frame["curation_unit_status"], default="not_applicable")
            if "curation_unit_status" in frame else {"not_applicable": rows}
        ),
        "unit_evidence_counts": (
            _counter(frame["unit_evidence"], default="none")
            if "unit_evidence" in frame else {"none": rows}
        ),
        "canonical_value_rows": (
            int(frame["canonical_value"].notna().sum()) if "canonical_value" in frame else 0
        ),
    }


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "CURATED_SCHEMA_VERSION",
    "CURATION_CONTRACT_FILENAME",
    "CuratedFramePreparation",
    "RootContract",
    "curated_profile_columns",
    "curated_profile_record",
    "curated_profile_summary",
    "dense_default",
    "installed_policy_fingerprint",
    "logical_counts",
    "prepare_curated_frame",
    "reconcile_curated_frame",
    "validate_root_contract",
]
