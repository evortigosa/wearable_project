"""
Wearable Data Processing and Modeling project
Formal native-to-curated Google Fitness acceptance audit. The audit is read-only with respect to both products.
It validates the managed native and curated roots, compares every participant-feature CSV row by row, recomputes
all sparse-default curation counters, checks persistent state and current policy signatures, and compares the result.
"""


from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import argparse
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
from tqdm import tqdm
from wearable_project.curation.google_fitness.decisions import decisions_fingerprint
from wearable_project.curation.google_fitness.engine import (
    DERIVED_COLUMN_ORDER, ENGINE_VERSION, feature_policy_fingerprint, sha256_file,
)
from wearable_project.curation.google_fitness.evidence import evidence_fingerprint
from wearable_project.curation.google_fitness.guidance import guidance_fingerprint
from wearable_project.curation.google_fitness.pipeline import PIPELINE_VERSION, NativeFeatureInput, _signature
from wearable_project.curation.google_fitness.registry import get_policy, registry_fingerprint
from wearable_project.curation.google_fitness.rules import rules_fingerprint
from wearable_project.curation.google_fitness.state import CURATION_STATE_FILENAME, CURATION_STATE_SCHEMA_VERSION
from wearable_project.exceptions import InputLayoutError
from wearable_project.processing.google_fitness.registry import SOURCE_KEY, get_feature_spec
from wearable_project.processing.google_fitness.writer import STATE_FILENAME as NATIVE_STATE_FILENAME


ACCEPTANCE_VERSION = "google-fitness-curation-sample-acceptance-cwp5-0.1.0"
ACCEPTANCE_REPORT_SCHEMA_VERSION = "google-fitness-curation-acceptance-report-0.1.0"
REFERENCE_SCHEMA_VERSION = "google-fitness-curation-reference-sample-0.1.0"
TEMP_DIRNAME = ".wearable_curation_acceptance_tmp"

_STATUS_VALUES = frozenset({"pass", "review", "exclude_default"})
_INCLUDE_VALUES = frozenset({"0", "1"})


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_field_limit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _state_quiet(path: Path, label: str) -> None:
    wal = path.with_name(path.name + "-wal")
    if wal.is_file() and wal.stat().st_size:
        raise InputLayoutError(f"{label} state has an active WAL: {wal}")


def _connect_immutable(path: Path, label: str) -> sqlite3.Connection:
    if not path.is_file():
        raise InputLayoutError(f"{label} state does not exist: {path}")
    _state_quiet(path, label)
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        return {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key,value FROM metadata")
        }
    except sqlite3.Error as exc:
        raise InputLayoutError("state database lacks a readable metadata table") from exc


def _guard_roots(native_root: Path, curated_root: Path, output: Path) -> tuple[Path, Path, Path]:
    native = Path(native_root).expanduser().resolve()
    curated = Path(curated_root).expanduser().resolve()
    report = Path(output).expanduser().resolve()
    if not native.is_dir():
        raise InputLayoutError(f"native root is not a directory: {native}")
    if not curated.is_dir():
        raise InputLayoutError(f"curated root is not a directory: {curated}")
    if native == curated or native in curated.parents or curated in native.parents:
        raise InputLayoutError(f"native and curated roots must be distinct and non-nested: {native}, {curated}")
    for protected in (native, curated):
        if report == protected or protected in report.parents or report in protected.parents:
            raise InputLayoutError(f"acceptance output must be outside both data roots: {report}")
    return native, curated, report


@dataclass(frozen=True, slots=True)
class NativeStateFile:
    participant_id: str
    feature: str
    filename: str
    row_count: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CuratedStateFile:
    participant_id: str
    feature: str
    filename: str
    native_rows: int
    curated_rows: int
    size_bytes: int
    sha256: str
    native_sha256: str
    policy_fingerprint: str
    active_derived_columns: tuple[str, ...]
    pass_rows: int
    review_rows: int
    exclude_default_rows: int
    included_by_default_rows: int
    excluded_by_default_rows: int
    canonical_value_rows: int
    flag_counts: dict[str, int]
    acquisition_counts: dict[str, int]
    derivation_counts: dict[str, int]
    tracking_counts: dict[str, int]
    step_count_role_counts: dict[str, int]
    unit_status_counts: dict[str, int]
    unit_evidence_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class CuratedParticipantState:
    participant_id: str
    status: str
    native_signature: str | None
    policy_signature: str | None
    engine_version: str | None
    last_error: str | None


@dataclass(frozen=True, slots=True)
class FileTask:
    participant_id: str
    feature: str
    native_path: Path
    curated_path: Path
    native_state: NativeStateFile
    curated_state: CuratedStateFile
    verify_hashes: bool


@dataclass(frozen=True, slots=True)
class ParticipantTask:
    participant_id: str
    native_dir: Path
    curated_dir: Path
    files: tuple[FileTask, ...]


@dataclass(slots=True)
class FileAudit:
    participant_id: str
    feature: str
    filename: str
    native_rows: int = 0
    curated_rows: int = 0
    native_size_bytes: int = 0
    curated_size_bytes: int = 0
    native_sha256: str | None = None
    curated_sha256: str | None = None
    native_header: tuple[str, ...] = ()
    derived_columns: tuple[str, ...] = ()
    pass_rows: int = 0
    review_rows: int = 0
    exclude_default_rows: int = 0
    included_by_default_rows: int = 0
    excluded_by_default_rows: int = 0
    canonical_value_rows: int = 0
    flag_counts: dict[str, int] = field(default_factory=dict)
    acquisition_counts: dict[str, int] = field(default_factory=dict)
    derivation_counts: dict[str, int] = field(default_factory=dict)
    tracking_counts: dict[str, int] = field(default_factory=dict)
    step_count_role_counts: dict[str, int] = field(default_factory=dict)
    unit_status_counts: dict[str, int] = field(default_factory=dict)
    unit_evidence_counts: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues


@dataclass(slots=True)
class ParticipantAudit:
    participant_id: str
    files: list[FileAudit] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    runtime_seconds: float | None = None

    @property
    def valid(self) -> bool:
        return not self.issues and all(item.valid for item in self.files)


def _load_native_state(path: Path) -> tuple[dict[str, str], dict[tuple[str, str], NativeStateFile], int]:
    with _connect_immutable(path, "native") as connection:
        metadata = _metadata(connection)
        if metadata.get("source_key") != SOURCE_KEY:
            raise InputLayoutError(
                f"native state source is {metadata.get('source_key')!r}, expected {SOURCE_KEY!r}"
            )
        rows = list(connection.execute(
            "SELECT participant_id,feature,filename,row_count,size_bytes,sha256 "
            "FROM feature_outputs ORDER BY participant_id,feature"
        ))
        try:
            active_runs = int(connection.execute(
                "SELECT COUNT(*) FROM runs WHERE status='in_progress'"
            ).fetchone()[0])
        except sqlite3.Error:
            active_runs = 0
    manifest = {
        (str(row["participant_id"]), str(row["feature"])): NativeStateFile(
            participant_id=str(row["participant_id"]),
            feature=str(row["feature"]),
            filename=str(row["filename"]),
            row_count=int(row["row_count"]),
            size_bytes=int(row["size_bytes"]),
            sha256=str(row["sha256"]),
        )
        for row in rows
    }
    return metadata, manifest, active_runs


def _output_from_row(row: sqlite3.Row) -> CuratedStateFile:
    return CuratedStateFile(
        participant_id=str(row["participant_id"]),
        feature=str(row["feature"]),
        filename=str(row["filename"]),
        native_rows=int(row["native_rows"]),
        curated_rows=int(row["curated_rows"]),
        size_bytes=int(row["size_bytes"]),
        sha256=str(row["sha256"]),
        native_sha256=str(row["native_sha256"]),
        policy_fingerprint=str(row["policy_fingerprint"]),
        active_derived_columns=tuple(_json_load(row["active_derived_columns_json"], [])),
        pass_rows=int(row["pass_rows"]),
        review_rows=int(row["review_rows"]),
        exclude_default_rows=int(row["exclude_default_rows"]),
        included_by_default_rows=int(row["included_by_default_rows"]),
        excluded_by_default_rows=int(row["excluded_by_default_rows"]),
        canonical_value_rows=int(row["canonical_value_rows"]),
        flag_counts=dict(_json_load(row["flag_counts_json"], {})),
        acquisition_counts=dict(_json_load(row["acquisition_counts_json"], {})),
        derivation_counts=dict(_json_load(row["derivation_counts_json"], {})),
        tracking_counts=dict(_json_load(row["tracking_counts_json"], {})),
        step_count_role_counts=dict(_json_load(row["step_count_role_counts_json"], {})),
        unit_status_counts=dict(_json_load(row["unit_status_counts_json"], {})),
        unit_evidence_counts=dict(_json_load(row["unit_evidence_counts_json"], {})),
    )


def _load_curated_state(
    path: Path,
) -> tuple[
    dict[str, str],
    dict[str, CuratedParticipantState],
    dict[tuple[str, str], NativeStateFile],
    dict[tuple[str, str], CuratedStateFile],
    int,
]:
    with _connect_immutable(path, "curated") as connection:
        metadata = _metadata(connection)
        if metadata.get("source_key") != SOURCE_KEY:
            raise InputLayoutError(
                f"curated state source is {metadata.get('source_key')!r}, expected {SOURCE_KEY!r}"
            )
        if metadata.get("product_kind") != "curated":
            raise InputLayoutError("curated state product_kind is not 'curated'")
        if metadata.get("state_schema_version") != CURATION_STATE_SCHEMA_VERSION:
            raise InputLayoutError(
                "curated state schema differs from the current implementation: "
                f"{metadata.get('state_schema_version')!r}"
            )
        participant_rows = list(connection.execute(
            "SELECT participant_id,status,native_signature,policy_signature,engine_version,last_error "
            "FROM curation_participants ORDER BY participant_id"
        ))
        native_rows = list(connection.execute(
            "SELECT participant_id,feature,filename,row_count,size_bytes,sha256,policy_fingerprint "
            "FROM curation_native_files ORDER BY participant_id,feature"
        ))
        output_rows = list(connection.execute(
            "SELECT * FROM curation_outputs ORDER BY participant_id,feature"
        ))
        active_runs = int(connection.execute(
            "SELECT COUNT(*) FROM curation_runs WHERE status='in_progress'"
        ).fetchone()[0])
    participants = {
        str(row["participant_id"]): CuratedParticipantState(
            participant_id=str(row["participant_id"]),
            status=str(row["status"]),
            native_signature=row["native_signature"],
            policy_signature=row["policy_signature"],
            engine_version=row["engine_version"],
            last_error=row["last_error"],
        )
        for row in participant_rows
    }
    native_manifest = {
        (str(row["participant_id"]), str(row["feature"])): NativeStateFile(
            participant_id=str(row["participant_id"]),
            feature=str(row["feature"]),
            filename=str(row["filename"]),
            row_count=int(row["row_count"]),
            size_bytes=int(row["size_bytes"]),
            sha256=str(row["sha256"]),
        )
        for row in native_rows
    }
    outputs = {
        (str(row["participant_id"]), str(row["feature"])): _output_from_row(row)
        for row in output_rows
    }
    return metadata, participants, native_manifest, outputs, active_runs


def _disk_csv_keys(root: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for participant_dir in sorted(root.iterdir(), key=lambda path: path.name):
        if not participant_dir.is_dir() or participant_dir.name.startswith("."):
            continue
        for path in sorted(participant_dir.glob("*.csv"), key=lambda item: item.name):
            keys.add((participant_dir.name, path.stem))
    return keys


def _split_flags(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(";") if item)


def _counter_value(row: Mapping[str, str], column: str, default: str) -> str:
    return (row.get(column) or "").strip() or default


def _compare_counter(name: str, observed: Counter[str], expected: Mapping[str, int], issues: list[str]) -> None:
    actual = dict(sorted((key, int(value)) for key, value in observed.items() if value))
    wanted = dict(sorted((str(key), int(value)) for key, value in expected.items() if value))
    if actual != wanted:
        issues.append(f"{name} differs from curated state: observed={actual}, expected={wanted}")


def _audit_file(task: FileTask) -> FileAudit:
    result = FileAudit(task.participant_id, task.feature, task.native_state.filename)
    native = task.native_path
    curated = task.curated_path
    try:
        if not native.is_file():
            result.issues.append(f"native file is missing: {native}")
            return result
        if not curated.is_file():
            result.issues.append(f"curated file is missing: {curated}")
            return result
        result.native_size_bytes = native.stat().st_size
        result.curated_size_bytes = curated.stat().st_size
        if result.native_size_bytes != task.native_state.size_bytes:
            result.issues.append("native file size differs from native state")
        if result.curated_size_bytes != task.curated_state.size_bytes:
            result.issues.append("curated file size differs from curated state")
        if task.verify_hashes:
            result.native_sha256 = sha256_file(native)
            result.curated_sha256 = sha256_file(curated)
            if result.native_sha256 != task.native_state.sha256:
                result.issues.append("native file SHA-256 differs from native state")
            if result.curated_sha256 != task.curated_state.sha256:
                result.issues.append("curated file SHA-256 differs from curated state")
        else:
            result.native_sha256 = task.native_state.sha256
            result.curated_sha256 = task.curated_state.sha256

        policy = get_policy(task.feature)
        current_policy = feature_policy_fingerprint(policy)
        if task.curated_state.policy_fingerprint != current_policy:
            result.issues.append("curated state policy fingerprint differs from current policy")
        if task.curated_state.native_sha256 != task.native_state.sha256:
            result.issues.append("curated state native SHA-256 differs from native state")
        if task.curated_state.native_rows != task.native_state.row_count:
            result.issues.append("curated state native row count differs from native state")

        with native.open("r", encoding="utf-8-sig", newline="") as native_handle, curated.open(
            "r", encoding="utf-8-sig", newline=""
        ) as curated_handle:
            native_reader = csv.reader(native_handle)
            curated_reader = csv.reader(curated_handle)
            native_header = tuple(next(native_reader, ()))
            curated_header = tuple(next(curated_reader, ()))
            result.native_header = native_header
            result.derived_columns = curated_header[len(native_header):]
            expected_native = tuple(get_feature_spec(task.feature).column_names)
            if native_header != expected_native:
                result.issues.append(
                    f"native header differs from registry: observed={native_header}, expected={expected_native}"
                )
            if curated_header[: len(native_header)] != native_header:
                result.issues.append("curated native-column prefix differs from native header")
            expected_derived = task.curated_state.active_derived_columns
            if result.derived_columns != expected_derived:
                result.issues.append(
                    f"derived columns differ from state: observed={result.derived_columns}, expected={expected_derived}"
                )
            if not set(result.derived_columns) <= set(policy.derived_columns):
                result.issues.append("curated file contains undeclared derived columns")
            ordered = tuple(column for column in DERIVED_COLUMN_ORDER if column in set(result.derived_columns))
            if result.derived_columns != ordered:
                result.issues.append("derived columns are not in the canonical engine order")

            indices = {name: index for index, name in enumerate(curated_header)}
            status_counts: Counter[str] = Counter()
            include_counts: Counter[str] = Counter()
            flags: Counter[str] = Counter()
            acquisition: Counter[str] = Counter()
            derivation: Counter[str] = Counter()
            tracking: Counter[str] = Counter()
            roles: Counter[str] = Counter()
            unit_status: Counter[str] = Counter()
            unit_evidence: Counter[str] = Counter()
            canonical_rows = 0

            row_index = 0
            while True:
                native_row = next(native_reader, None)
                curated_row = next(curated_reader, None)
                if native_row is None and curated_row is None:
                    break
                if native_row is None or curated_row is None:
                    result.issues.append(f"native/curated row-count mismatch at row {row_index}")
                    break
                if tuple(curated_row[: len(native_header)]) != tuple(native_row):
                    result.issues.append(f"native value or row order changed at row {row_index}")
                    break
                if len(curated_row) != len(curated_header):
                    result.issues.append(f"curated row width differs from header at row {row_index}")
                    break
                values = {name: curated_row[index] for name, index in indices.items()}
                status = _counter_value(values, "curation_status", "pass")
                include = _counter_value(values, "include_by_default", "1")
                if status not in _STATUS_VALUES:
                    result.issues.append(f"invalid curation_status {status!r} at row {row_index}")
                if include not in _INCLUDE_VALUES:
                    result.issues.append(f"invalid include_by_default {include!r} at row {row_index}")
                status_counts[status] += 1
                include_counts[include] += 1
                if (values.get("canonical_value") or "").strip():
                    canonical_rows += 1
                flags.update(_split_flags(values.get("curation_flags", "")))
                acquisition[_counter_value(values, "acquisition_method", "unknown")] += 1
                derivation[_counter_value(values, "derivation_method", "unknown")] += 1
                tracking[_counter_value(values, "tracking_status", "not_applicable")] += 1
                roles[_counter_value(values, "step_count_role", "not_applicable")] += 1
                unit_status[_counter_value(values, "curation_unit_status", "not_applicable")] += 1
                unit_evidence[_counter_value(values, "unit_evidence", "none")] += 1
                row_index += 1

            result.native_rows = row_index
            result.curated_rows = row_index
            result.pass_rows = status_counts["pass"]
            result.review_rows = status_counts["review"]
            result.exclude_default_rows = status_counts["exclude_default"]
            result.included_by_default_rows = include_counts["1"]
            result.excluded_by_default_rows = include_counts["0"]
            result.canonical_value_rows = canonical_rows
            result.flag_counts = dict(sorted(flags.items()))
            result.acquisition_counts = dict(sorted(acquisition.items()))
            result.derivation_counts = dict(sorted(derivation.items()))
            result.tracking_counts = dict(sorted(tracking.items()))
            result.step_count_role_counts = dict(sorted(roles.items()))
            result.unit_status_counts = dict(sorted(unit_status.items()))
            result.unit_evidence_counts = dict(sorted(unit_evidence.items()))

        if result.native_rows != task.native_state.row_count:
            result.issues.append("physical native row count differs from native state")
        if result.curated_rows != task.curated_state.curated_rows:
            result.issues.append("physical curated row count differs from curated state")
        if task.curated_state.native_rows != task.curated_state.curated_rows:
            result.issues.append("curated state records native/curated row-count inequality")
        if result.pass_rows != task.curated_state.pass_rows:
            result.issues.append("pass-row count differs from curated state")
        if result.review_rows != task.curated_state.review_rows:
            result.issues.append("review-row count differs from curated state")
        if result.exclude_default_rows != task.curated_state.exclude_default_rows:
            result.issues.append("exclude-default row count differs from curated state")
        if result.included_by_default_rows != task.curated_state.included_by_default_rows:
            result.issues.append("included-by-default count differs from curated state")
        if result.excluded_by_default_rows != task.curated_state.excluded_by_default_rows:
            result.issues.append("excluded-by-default count differs from curated state")
        if result.canonical_value_rows != task.curated_state.canonical_value_rows:
            result.issues.append("canonical-value count differs from curated state")
        _compare_counter("flag_counts", Counter(result.flag_counts), task.curated_state.flag_counts, result.issues)
        _compare_counter("acquisition_counts", Counter(result.acquisition_counts), task.curated_state.acquisition_counts, result.issues)
        _compare_counter("derivation_counts", Counter(result.derivation_counts), task.curated_state.derivation_counts, result.issues)
        _compare_counter("tracking_counts", Counter(result.tracking_counts), task.curated_state.tracking_counts, result.issues)
        _compare_counter("step_count_role_counts", Counter(result.step_count_role_counts), task.curated_state.step_count_role_counts, result.issues)
        _compare_counter("unit_status_counts", Counter(result.unit_status_counts), task.curated_state.unit_status_counts, result.issues)
        _compare_counter("unit_evidence_counts", Counter(result.unit_evidence_counts), task.curated_state.unit_evidence_counts, result.issues)
    except Exception as exc:  # participant isolation: return the failure to the parent
        result.issues.append(f"{type(exc).__name__}: {exc}")
    return result


def _audit_participant(task: ParticipantTask) -> ParticipantAudit:
    started = time.perf_counter()
    result = ParticipantAudit(task.participant_id)
    try:
        expected = {item.filename for item in (file.native_state for file in task.files)}
        actual_native = {path.name for path in task.native_dir.glob("*.csv")}
        actual_curated = {path.name for path in task.curated_dir.glob("*.csv")}
        if actual_native != expected:
            result.issues.append(
                f"native file set differs: expected={sorted(expected)}, actual={sorted(actual_native)}"
            )
        if actual_curated != expected:
            result.issues.append(
                f"curated file set differs: expected={sorted(expected)}, actual={sorted(actual_curated)}"
            )
        result.files = [_audit_file(item) for item in task.files]
    except Exception as exc:
        result.issues.append(f"{type(exc).__name__}: {exc}")
    result.runtime_seconds = time.perf_counter() - started
    return result


def _merge_dict_counter(results: Iterable[FileAudit], attribute: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in results:
        counter.update(getattr(item, attribute))
    return dict(sorted(counter.items()))


def _tree_fingerprint(files: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(files.items())), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _semantic_fingerprint(payload: Any) -> str:
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _aggregate_curated_state(outputs: Mapping[tuple[str, str], CuratedStateFile]) -> dict[str, Any]:
    rows = list(outputs.values())
    flags: Counter[str] = Counter()
    acquisition: Counter[str] = Counter()
    derivation: Counter[str] = Counter()
    tracking: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    unit_status: Counter[str] = Counter()
    unit_evidence: Counter[str] = Counter()
    active_columns: Counter[str] = Counter()
    for row in rows:
        flags.update(row.flag_counts)
        acquisition.update(row.acquisition_counts)
        derivation.update(row.derivation_counts)
        tracking.update(row.tracking_counts)
        roles.update(row.step_count_role_counts)
        unit_status.update(row.unit_status_counts)
        unit_evidence.update(row.unit_evidence_counts)
        active_columns.update(row.active_derived_columns)
    snapshot = {
        "participants": len({row.participant_id for row in rows}),
        "feature_files": len(rows),
        "native_rows": sum(row.native_rows for row in rows),
        "curated_rows": sum(row.curated_rows for row in rows),
        "curated_size_bytes": sum(row.size_bytes for row in rows),
        "pass_rows": sum(row.pass_rows for row in rows),
        "review_rows": sum(row.review_rows for row in rows),
        "exclude_default_status_rows": sum(row.exclude_default_rows for row in rows),
        "included_by_default_rows": sum(row.included_by_default_rows for row in rows),
        "excluded_by_default_rows": sum(row.excluded_by_default_rows for row in rows),
        "canonical_value_rows": sum(row.canonical_value_rows for row in rows),
        "flag_counts": dict(sorted(flags.items())),
        "acquisition_counts": dict(sorted(acquisition.items())),
        "derivation_counts": dict(sorted(derivation.items())),
        "tracking_counts": dict(sorted(tracking.items())),
        "step_count_role_counts": dict(sorted(roles.items())),
        "unit_status_counts": dict(sorted(unit_status.items())),
        "unit_evidence_counts": dict(sorted(unit_evidence.items())),
        "active_derived_column_file_counts": dict(sorted(active_columns.items())),
    }
    snapshot["status_accounting_complete"] = (
        snapshot["pass_rows"] + snapshot["review_rows"] + snapshot["exclude_default_status_rows"]
        == snapshot["curated_rows"]
    )
    snapshot["default_inclusion_accounting_complete"] = (
        snapshot["included_by_default_rows"] + snapshot["excluded_by_default_rows"]
        == snapshot["curated_rows"]
    )
    return snapshot


def _flatten_snapshot(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_snapshot(child, value[key]))
        return flattened
    return {prefix: value}


def _reference_view(
    *, participant_ids: Sequence[str], file_results: Sequence[FileAudit], snapshot: Mapping[str, Any],
    planner_counts: Mapping[str, int], native_state_semantic_sha256: str, curated_state_semantic_sha256: str,
) -> dict[str, Any]:
    native_hashes = {
        f"{item.participant_id}/{item.filename}": str(item.native_sha256) for item in file_results
    }
    curated_hashes = {
        f"{item.participant_id}/{item.filename}": str(item.curated_sha256) for item in file_results
    }
    return {
        "source_key": SOURCE_KEY,
        "participants": list(participant_ids),
        "snapshot": dict(snapshot),
        "planner_action_counts": dict(sorted(planner_counts.items())),
        "fingerprints": {
            "registry": registry_fingerprint(),
            "evidence": evidence_fingerprint(),
            "rules": rules_fingerprint(),
            "decisions": decisions_fingerprint(),
            "guidance": guidance_fingerprint(),
            "engine": ENGINE_VERSION,
            "pipeline": PIPELINE_VERSION,
            "state_schema": CURATION_STATE_SCHEMA_VERSION,
        },
        "native_csv_tree_sha256": _tree_fingerprint(native_hashes),
        "curated_csv_tree_sha256": _tree_fingerprint(curated_hashes),
        "native_state_semantic_sha256": native_state_semantic_sha256,
        "curated_state_semantic_sha256": curated_state_semantic_sha256,
        "native_file_sha256": dict(sorted(native_hashes.items())),
        "curated_file_sha256": dict(sorted(curated_hashes.items())),
    }


def build_reference_manifest(reference_id: str, reference_view: Mapping[str, Any]) -> dict[str, Any]:
    """Create a frozen reference manifest from a passing acceptance result."""

    return {
        "reference_schema_version": REFERENCE_SCHEMA_VERSION,
        "reference_id": reference_id,
        "expected": dict(reference_view),
    }


def _compare_reference(reference: Mapping[str, Any], actual: Mapping[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if reference.get("reference_schema_version") != REFERENCE_SCHEMA_VERSION:
        issues.append({
            "path": "reference_schema_version",
            "expected": REFERENCE_SCHEMA_VERSION,
            "observed": reference.get("reference_schema_version"),
        })
        return issues
    expected = reference.get("expected")
    if not isinstance(expected, Mapping):
        return [{"path": "expected", "expected": "mapping", "observed": type(expected).__name__}]

    def compare(path: str, wanted: Any, observed: Any) -> None:
        if isinstance(wanted, Mapping):
            if not isinstance(observed, Mapping):
                issues.append({"path": path, "expected": wanted, "observed": observed})
                return
            for key in sorted(wanted):
                compare(f"{path}.{key}" if path else str(key), wanted[key], observed.get(key))
        elif isinstance(wanted, list):
            if wanted != observed:
                issues.append({"path": path, "expected": wanted, "observed": observed})
        elif wanted != observed:
            issues.append({"path": path, "expected": wanted, "observed": observed})

    compare("", expected, actual)
    return issues


def _check(check_id: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"check_id": check_id, "status": "pass" if passed else "fail", "detail": detail}


def _render_report(payload: Mapping[str, Any]) -> str:
    snapshot = payload["snapshot"]
    lines = [
        "# Google Fitness CWP5 curation acceptance",
        "",
        f"- Status: **{str(payload['status']).upper()}**",
        f"- Acceptance version: `{payload['acceptance_version']}`",
        f"- Participants: {snapshot['participants']:,}",
        f"- Participant-feature files: {snapshot['feature_files']:,}",
        f"- Native rows: {snapshot['native_rows']:,}",
        f"- Curated rows: {snapshot['curated_rows']:,}",
        f"- Failed checks: {payload['failed_check_count']}",
        "",
        "## Checks",
        "",
    ]
    for item in payload["checks"]:
        marker = "PASS" if item["status"] == "pass" else "FAIL"
        lines.append(f"- **{marker}** `{item['check_id']}` — {item['detail']}")
    lines.extend(["", "## Snapshot", "", "```json", json.dumps(snapshot, indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


def _publish_reports(stage: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise InputLayoutError(f"acceptance output already exists; use --overwrite: {output}")
        backup = output.with_name(f".{output.name}.backup")
        shutil.rmtree(backup, ignore_errors=True)
        os.replace(output, backup)
        try:
            os.replace(stage, output)
        except Exception:
            os.replace(backup, output)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, output)


def run_acceptance(
    native_root: Path, curated_root: Path, output: Path, *, workers: int = 4, max_in_flight: int | None = None,
    reference_manifest: Path | None = None, verify_hashes: bool = True, overwrite: bool = False,
    no_progress: bool = False,
) -> dict[str, Any]:
    """Run the read-only CWP5 acceptance audit and publish deterministic reports."""

    native, curated, report_root = _guard_roots(native_root, curated_root, output)
    native_state_path = native / NATIVE_STATE_FILENAME
    curated_state_path = curated / CURATION_STATE_FILENAME
    started_at = _now()
    started = time.perf_counter()
    native_state_sha_before = sha256_file(native_state_path)
    curated_state_sha_before = sha256_file(curated_state_path)

    native_metadata, native_manifest, active_native_runs = _load_native_state(native_state_path)
    curated_metadata, participant_states, curated_native_manifest, curated_outputs, active_runs = _load_curated_state(curated_state_path)

    native_keys = set(native_manifest)
    curated_native_keys = set(curated_native_manifest)
    curated_output_keys = set(curated_outputs)
    native_disk_keys = _disk_csv_keys(native)
    curated_disk_keys = _disk_csv_keys(curated)
    participants_native = sorted({participant for participant, _ in native_keys})
    participants_curated = sorted({participant for participant, _ in curated_output_keys})

    tasks: list[ParticipantTask] = []
    task_build_issues: list[str] = []
    for participant_id in sorted(set(participants_native) | set(participants_curated)):
        keys = sorted(key for key in native_keys | curated_output_keys if key[0] == participant_id)
        files: list[FileTask] = []
        for key in keys:
            native_item = native_manifest.get(key)
            curation_native = curated_native_manifest.get(key)
            output_item = curated_outputs.get(key)
            if native_item is None or curation_native is None or output_item is None:
                task_build_issues.append(
                    f"incomplete state linkage for {participant_id}/{key[1]}: "
                    f"native={native_item is not None}, curated_native={curation_native is not None}, output={output_item is not None}"
                )
                continue
            if curation_native.sha256 != native_item.sha256 or curation_native.row_count != native_item.row_count:
                task_build_issues.append(f"curation native-state record differs from native state for {participant_id}/{key[1]}")
            files.append(FileTask(
                participant_id=participant_id,
                feature=key[1],
                native_path=native / participant_id / native_item.filename,
                curated_path=curated / participant_id / output_item.filename,
                native_state=native_item,
                curated_state=output_item,
                verify_hashes=verify_hashes,
            ))
        tasks.append(ParticipantTask(
            participant_id=participant_id,
            native_dir=native / participant_id,
            curated_dir=curated / participant_id,
            files=tuple(files),
        ))

    effective_workers = max(1, min(int(workers), os.cpu_count() or 1, len(tasks) or 1))
    inflight_limit = max(1, int(max_in_flight or effective_workers))
    results: list[ParticipantAudit] = []
    if effective_workers == 1:
        iterator = tasks
        progress = tqdm(
            total=len(tasks), desc="Accepting curated participants", unit="participant", disable=no_progress,
        )
        try:
            for task in iterator:
                results.append(_audit_participant(task))
                progress.update(1)
        finally:
            progress.close()
    else:
        context = mp.get_context("spawn")
        progress = tqdm(
            total=len(tasks), desc="Accepting curated participants", unit="participant", disable=no_progress,
        )
        pending: dict[Future[ParticipantAudit], ParticipantTask] = {}
        index = 0
        try:
            with ProcessPoolExecutor(max_workers=effective_workers, mp_context=context) as executor:
                while index < len(tasks) or pending:
                    while index < len(tasks) and len(pending) < inflight_limit:
                        task = tasks[index]
                        pending[executor.submit(_audit_participant, task)] = task
                        index += 1
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        task = pending.pop(future)
                        try:
                            results.append(future.result())
                        except Exception as exc:
                            results.append(ParticipantAudit(task.participant_id, issues=[f"{type(exc).__name__}: {exc}"]))
                        progress.update(1)
        finally:
            progress.close()
    results.sort(key=lambda item: item.participant_id)
    file_results = sorted(
        [file for result in results for file in result.files],
        key=lambda item: (item.participant_id, item.feature),
    )

    participant_issues = [
        {"participant_id": result.participant_id, "issues": [*result.issues, *[issue for file in result.files for issue in file.issues]]}
        for result in results if not result.valid
    ]
    all_file_issues = [issue for result in participant_issues for issue in result["issues"]]

    snapshot = {
        "participants": len(results),
        "feature_files": len(file_results),
        "native_rows": sum(item.native_rows for item in file_results),
        "curated_rows": sum(item.curated_rows for item in file_results),
        "native_size_bytes": sum(item.native_size_bytes for item in file_results),
        "curated_size_bytes": sum(item.curated_size_bytes for item in file_results),
        "pass_rows": sum(item.pass_rows for item in file_results),
        "review_rows": sum(item.review_rows for item in file_results),
        "exclude_default_status_rows": sum(item.exclude_default_rows for item in file_results),
        "included_by_default_rows": sum(item.included_by_default_rows for item in file_results),
        "excluded_by_default_rows": sum(item.excluded_by_default_rows for item in file_results),
        "canonical_value_rows": sum(item.canonical_value_rows for item in file_results),
        "flag_counts": _merge_dict_counter(file_results, "flag_counts"),
        "acquisition_counts": _merge_dict_counter(file_results, "acquisition_counts"),
        "derivation_counts": _merge_dict_counter(file_results, "derivation_counts"),
        "tracking_counts": _merge_dict_counter(file_results, "tracking_counts"),
        "step_count_role_counts": _merge_dict_counter(file_results, "step_count_role_counts"),
        "unit_status_counts": _merge_dict_counter(file_results, "unit_status_counts"),
        "unit_evidence_counts": _merge_dict_counter(file_results, "unit_evidence_counts"),
        "active_derived_column_file_counts": dict(sorted(Counter(
            column for item in file_results for column in item.derived_columns
        ).items())),
    }
    snapshot["status_accounting_complete"] = (
        snapshot["pass_rows"] + snapshot["review_rows"] + snapshot["exclude_default_status_rows"]
        == snapshot["curated_rows"]
    )
    snapshot["default_inclusion_accounting_complete"] = (
        snapshot["included_by_default_rows"] + snapshot["excluded_by_default_rows"]
        == snapshot["curated_rows"]
    )
    curated_state_snapshot = _aggregate_curated_state(curated_outputs)
    physical_state_comparable = {
        key: value for key, value in snapshot.items() if key != "native_size_bytes"
    }
    state_snapshot_matches = physical_state_comparable == curated_state_snapshot

    planner_reasons: dict[str, str] = {}
    planner_actions: dict[str, str] = {}
    planner_counts: Counter[str] = Counter()
    for participant_id in participants_curated:
        participant = participant_states.get(participant_id)
        files = [item for item in file_results if item.participant_id == participant_id]
        native_inputs = tuple(NativeFeatureInput(
            feature=item.feature,
            filename=item.filename,
            path=native / participant_id / item.filename,
            row_count=item.native_rows,
            size_bytes=item.native_size_bytes,
            sha256=str(item.native_sha256),
            policy_fingerprint=feature_policy_fingerprint(get_policy(item.feature)),
        ) for item in files)
        action = "skip"
        reason = "native input, policies, and curated outputs are current"
        if participant is None:
            action, reason = "rebuild", "participant has no curation state"
        elif participant.status != "complete":
            action, reason = "rebuild", f"participant state is {participant.status!r}"
        elif participant.engine_version != ENGINE_VERSION:
            action, reason = "rebuild", "engine version differs"
        elif participant.native_signature != _signature(native_inputs, include_policy=False):
            action, reason = "rebuild", "native signature differs"
        elif participant.policy_signature != _signature(native_inputs, include_policy=True):
            action, reason = "rebuild", "policy signature differs"
        elif any(not item.valid for item in files):
            action, reason = "rebuild", "one or more physical files failed acceptance"
        planner_counts[action] += 1
        planner_actions[participant_id] = action
        planner_reasons[participant_id] = reason

    native_state_semantic_sha256 = _semantic_fingerprint({
        "metadata": native_metadata,
        "files": [asdict(native_manifest[key]) for key in sorted(native_manifest)],
    })
    curated_state_semantic_sha256 = _semantic_fingerprint({
        "metadata": curated_metadata,
        "participants": [asdict(participant_states[key]) for key in sorted(participant_states)],
        "native_files": [asdict(curated_native_manifest[key]) for key in sorted(curated_native_manifest)],
        "outputs": [asdict(curated_outputs[key]) for key in sorted(curated_outputs)],
    })

    reference_view = _reference_view(
        participant_ids=participants_curated,
        file_results=file_results,
        snapshot=snapshot,
        planner_counts=planner_counts,
        native_state_semantic_sha256=native_state_semantic_sha256,
        curated_state_semantic_sha256=curated_state_semantic_sha256,
    )
    reference_payload: dict[str, Any] | None = None
    reference_issues: list[dict[str, Any]] = []
    if reference_manifest is not None:
        reference_payload = json.loads(Path(reference_manifest).read_text(encoding="utf-8"))
        reference_issues = _compare_reference(reference_payload, reference_view)

    state_snapshot = {
        "native_metadata": native_metadata,
        "curated_metadata": curated_metadata,
        "native_state_feature_files": len(native_manifest),
        "active_native_runs": active_native_runs,
        "curated_state_native_files": len(curated_native_manifest),
        "curated_state_output_files": len(curated_outputs),
        "curated_state_participants": len(participant_states),
        "active_curation_runs": active_runs,
        "curated_output_snapshot": curated_state_snapshot,
    }

    checks = [
        _check("source_qualified_states", native_metadata.get("source_key") == SOURCE_KEY and curated_metadata.get("source_key") == SOURCE_KEY, "native and curated state source keys are google_fitness"),
        _check("quiescent_native_state", active_native_runs == 0, f"active native runs: {active_native_runs}"),
        _check("quiescent_curation_state", active_runs == 0, f"active curation runs: {active_runs}"),
        _check("participant_sets_equal", participants_native == participants_curated, f"native={len(participants_native)}, curated={len(participants_curated)}"),
        _check("state_file_sets_equal", native_keys == curated_native_keys == curated_output_keys, f"native={len(native_keys)}, curated_native={len(curated_native_keys)}, outputs={len(curated_output_keys)}"),
        _check("disk_file_sets_match_state", native_disk_keys == native_keys and curated_disk_keys == curated_output_keys, f"native_disk={len(native_disk_keys)}, native_state={len(native_keys)}, curated_disk={len(curated_disk_keys)}, curated_state={len(curated_output_keys)}"),
        _check("state_linkage_complete", not task_build_issues, f"state-linkage issues: {len(task_build_issues)}"),
        _check("physical_file_validation", not all_file_issues, f"physical validation issues: {len(all_file_issues)}"),
        _check("physical_state_accounting_match", state_snapshot_matches, "physical curated counters match the persistent curation state"),
        _check("native_rows_equal_curated_rows", snapshot["native_rows"] == snapshot["curated_rows"], f"native={snapshot['native_rows']}, curated={snapshot['curated_rows']}"),
        _check("status_accounting_complete", bool(snapshot["status_accounting_complete"]), "pass + review + exclude_default equals curated rows"),
        _check("default_inclusion_accounting_complete", bool(snapshot["default_inclusion_accounting_complete"]), "included + excluded equals curated rows"),
        _check("participant_state_complete", all(item.status == "complete" for item in participant_states.values()), "all curated participants have complete state"),
        _check("current_engine_and_policies", all(count == 0 for action, count in planner_counts.items() if action != "skip"), f"planner actions: {dict(sorted(planner_counts.items()))}"),
        _check("hash_verification", bool(verify_hashes), "native and curated SHA-256 verification was enabled" if verify_hashes else "hash verification was disabled"),
        _check("reference_manifest_match", reference_manifest is None or not reference_issues, "reference not requested" if reference_manifest is None else f"reference mismatches: {len(reference_issues)}"),
    ]
    failed = [item for item in checks if item["status"] == "fail"]
    status = "pass" if not failed else "fail"

    native_state_sha_after = sha256_file(native_state_path)
    curated_state_sha_after = sha256_file(curated_state_path)
    root_unchanged = (
        native_state_sha_before == native_state_sha_after and curated_state_sha_before == curated_state_sha_after
    )
    checks.append(_check("state_databases_unchanged", root_unchanged, "native and curated state database hashes are unchanged"))
    if not root_unchanged:
        status = "fail"
        failed = [item for item in checks if item["status"] == "fail"]

    acceptance_payload = {
        "tool": "google_fitness_curation_acceptance",
        "source_key": SOURCE_KEY,
        "acceptance_version": ACCEPTANCE_VERSION,
        "report_schema_version": ACCEPTANCE_REPORT_SCHEMA_VERSION,
        "status": status,
        "check_count": len(checks),
        "passed_check_count": sum(item["status"] == "pass" for item in checks),
        "failed_check_count": sum(item["status"] == "fail" for item in checks),
        "checks": checks,
        "snapshot": snapshot,
        "curated_state_snapshot": curated_state_snapshot,
        "state_snapshot": state_snapshot,
        "planner_action_counts": dict(sorted(planner_counts.items())),
        "participant_issue_count": len(participant_issues),
        "file_issue_count": len(all_file_issues),
        "reference": {
            "requested": reference_manifest is not None,
            "reference_id": reference_payload.get("reference_id") if reference_payload else None,
            "mismatch_count": len(reference_issues),
            "mismatches": reference_issues,
        },
        "fingerprints": reference_view["fingerprints"],
        "native_csv_tree_sha256": reference_view["native_csv_tree_sha256"],
        "curated_csv_tree_sha256": reference_view["curated_csv_tree_sha256"],
        "native_state_semantic_sha256": native_state_semantic_sha256,
        "curated_state_semantic_sha256": curated_state_semantic_sha256,
    }

    file_rows: list[dict[str, Any]] = []
    for item in file_results:
        file_rows.append({
            "participant_id": item.participant_id,
            "feature": item.feature,
            "filename": item.filename,
            "native_rows": item.native_rows,
            "curated_rows": item.curated_rows,
            "native_size_bytes": item.native_size_bytes,
            "curated_size_bytes": item.curated_size_bytes,
            "native_sha256": item.native_sha256,
            "curated_sha256": item.curated_sha256,
            "derived_columns": ";".join(item.derived_columns),
            "pass_rows": item.pass_rows,
            "review_rows": item.review_rows,
            "exclude_default_rows": item.exclude_default_rows,
            "included_by_default_rows": item.included_by_default_rows,
            "excluded_by_default_rows": item.excluded_by_default_rows,
            "canonical_value_rows": item.canonical_value_rows,
            "issue_count": len(item.issues),
            "issues": " | ".join(item.issues),
        })

    participant_rows: list[dict[str, Any]] = []
    for item in results:
        files = item.files
        participant_rows.append({
            "participant_id": item.participant_id,
            "feature_files": len(files),
            "native_rows": sum(file.native_rows for file in files),
            "curated_rows": sum(file.curated_rows for file in files),
            "planner_action": planner_actions.get(item.participant_id, "rebuild"),
            "planner_reason": planner_reasons.get(item.participant_id, ""),
            "valid": int(item.valid),
            "issue_count": len(item.issues) + sum(len(file.issues) for file in files),
        })

    physical_flat = _flatten_snapshot("", physical_state_comparable)
    state_flat = _flatten_snapshot("", curated_state_snapshot)
    counter_comparison_rows = [
        {
            "metric": metric,
            "physical_value": physical_flat.get(metric, ""),
            "state_value": state_flat.get(metric, ""),
            "matches": int(physical_flat.get(metric) == state_flat.get(metric)),
        }
        for metric in sorted(set(physical_flat) | set(state_flat))
    ]

    feature_aggregate: dict[str, dict[str, Any]] = {}
    for feature in sorted({item.feature for item in file_results}):
        items = [item for item in file_results if item.feature == feature]
        feature_aggregate[feature] = {
            "feature": feature,
            "participant_files": len(items),
            "native_rows": sum(item.native_rows for item in items),
            "curated_rows": sum(item.curated_rows for item in items),
            "pass_rows": sum(item.pass_rows for item in items),
            "review_rows": sum(item.review_rows for item in items),
            "exclude_default_rows": sum(item.exclude_default_rows for item in items),
            "included_by_default_rows": sum(item.included_by_default_rows for item in items),
            "excluded_by_default_rows": sum(item.excluded_by_default_rows for item in items),
            "canonical_value_rows": sum(item.canonical_value_rows for item in items),
            "issue_count": sum(len(item.issues) for item in items),
        }

    stage_parent = report_root.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{report_root.name}.stage-", dir=stage_parent))
    try:
        _json_write(stage / "cwp5_acceptance.json", acceptance_payload)
        _json_write(stage / "reference_view.json", reference_view)
        _json_write(stage / "participant_issues.json", participant_issues)
        _json_write(stage / "reference_comparison.json", {
            "requested": reference_manifest is not None,
            "reference_id": reference_payload.get("reference_id") if reference_payload else None,
            "mismatches": reference_issues,
        })
        _csv_write(stage / "file_manifest.csv", file_rows, (
            "participant_id", "feature", "filename", "native_rows", "curated_rows",
            "native_size_bytes", "curated_size_bytes", "native_sha256", "curated_sha256",
            "derived_columns", "pass_rows", "review_rows", "exclude_default_rows",
            "included_by_default_rows", "excluded_by_default_rows", "canonical_value_rows",
            "issue_count", "issues",
        ))
        _csv_write(stage / "participant_summary.csv", participant_rows, (
            "participant_id", "feature_files", "native_rows", "curated_rows", "planner_action",
            "planner_reason", "valid", "issue_count",
        ))
        _csv_write(stage / "counter_comparison.csv", counter_comparison_rows, (
            "metric", "physical_value", "state_value", "matches",
        ))
        _csv_write(stage / "feature_summary.csv", list(feature_aggregate.values()), (
            "feature", "participant_files", "native_rows", "curated_rows", "pass_rows",
            "review_rows", "exclude_default_rows", "included_by_default_rows",
            "excluded_by_default_rows", "canonical_value_rows", "issue_count",
        ))
        (stage / "REPORT.md").write_text(_render_report(acceptance_payload), encoding="utf-8")
        _json_write(stage / "run_manifest.json", {
            "tool": "google_fitness_curation_acceptance",
            "acceptance_version": ACCEPTANCE_VERSION,
            "started_at": started_at,
            "finished_at": _now(),
            "native_root": str(native),
            "curated_root": str(curated),
            "output": str(report_root),
            "workers_requested": int(workers),
            "workers_effective": effective_workers,
            "max_in_flight": inflight_limit,
            "verify_hashes": bool(verify_hashes),
            "reference_manifest": str(Path(reference_manifest).resolve()) if reference_manifest else None,
            "native_state_sha256_before": native_state_sha_before,
            "native_state_sha256_after": native_state_sha_after,
            "curated_state_sha256_before": curated_state_sha_before,
            "curated_state_sha256_after": curated_state_sha_after,
            "wall_clock_seconds": round(time.perf_counter() - started, 6),
        })
        _publish_reports(stage, report_root, overwrite)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return acceptance_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run formal Google Fitness native-to-curated acceptance checks.")
    parser.add_argument("--input-native", type=Path, required=True)
    parser.add_argument("--input-curated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--no-verify-hashes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--json-summary", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_acceptance(
            args.input_native,
            args.input_curated,
            args.output,
            workers=args.workers,
            max_in_flight=args.max_in_flight,
            reference_manifest=args.reference_manifest,
            verify_hashes=not args.no_verify_hashes,
            overwrite=args.overwrite,
            no_progress=args.no_progress,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json_summary:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"Google Fitness CWP5 acceptance {result['status'].upper()}: "
            f"{result['passed_check_count']}/{result['check_count']} checks passed; "
            f"{result['snapshot']['feature_files']} files; {result['snapshot']['curated_rows']} rows"
        )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
