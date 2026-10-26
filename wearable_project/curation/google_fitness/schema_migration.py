"""
Wearable Data Processing and Modeling project
Read-only semantic audit from accepted Google curation schema v1 to schema v2. The audit compares one accepted
native root, the frozen schema-v1 curated root, and a new schema-v2 curated root. It proves that schema v2 removes
only reconstructable or policy-level repetitions while preserving every row-level curation decision.
"""


from __future__ import annotations
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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
from typing import Any, Iterable, Mapping
from tqdm import tqdm
from wearable_project.exceptions import InputLayoutError
from wearable_project.processing.google_fitness.registry import SOURCE_KEY
from wearable_project.processing.google_fitness.writer import STATE_FILENAME as NATIVE_STATE_FILENAME
from .contract import (
    CURATED_SCHEMA_VERSION, CURATION_CONTRACT_FILENAME, effective_row_decisions, validate_contract_file,
    virtual_default, virtual_row_values,
)
from .engine import ENGINE_VERSION
from .pipeline import PIPELINE_VERSION
from .registry import get_policy
from .state import CURATION_STATE_FILENAME, CURATION_STATE_SCHEMA_VERSION


SCHEMA_MIGRATION_AUDIT_VERSION = "google-fitness-curation-schema-migration-audit-v2-0.1.0"
SCHEMA_MIGRATION_REPORT_VERSION = "google-fitness-curation-schema-migration-report-v2-0.1.0"
LEGACY_STATE_SCHEMA_VERSION = "google-fitness-curation-state-cwp4-0.1.0"
TEMP_DIRNAME = ".wearable_schema_migration_tmp"
POLICY_LEVEL_V1_FLAGS = frozenset({
    "activity_quantity_semantics_unresolved",
    "canonical_unit_withheld_pending_exporter_mapping",
})


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_quiet(path: Path, label: str) -> None:
    wal = path.with_name(path.name + "-wal")
    if wal.is_file() and wal.stat().st_size:
        raise InputLayoutError(f"{label} state has an active WAL: {wal}")


def _metadata(path: Path, label: str) -> dict[str, str]:
    if not path.is_file():
        raise InputLayoutError(f"{label} state does not exist: {path}")
    _state_quiet(path, label)
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key,value FROM metadata")
        }
    except sqlite3.Error as exc:
        raise InputLayoutError(f"{label} state lacks readable metadata") from exc
    finally:
        connection.close()


def _guard_roots(native: Path, legacy: Path, current: Path, output: Path) -> tuple[Path, Path, Path, Path]:
    roots = tuple(Path(item).expanduser().resolve() for item in (native, legacy, current))
    report = Path(output).expanduser().resolve()
    labels = ("native", "schema-v1 curated", "schema-v2 curated")
    for label, root in zip(labels, roots):
        if not root.is_dir():
            raise InputLayoutError(f"{label} root is not a directory: {root}")
    for index, left in enumerate(roots):
        for right in roots[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise InputLayoutError(f"data roots must be distinct and non-nested: {left}, {right}")
    for root in roots:
        if report == root or report in root.parents or root in report.parents:
            raise InputLayoutError(f"audit output must be outside every data root: {report}")
    return roots[0], roots[1], roots[2], report


def _participant_files(root: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for directory in sorted(root.iterdir(), key=lambda item: item.name):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        files = tuple(sorted(path.name for path in directory.glob("*.csv") if path.is_file()))
        if files:
            result[directory.name] = files
    return result


def _flags(value: str) -> tuple[str, ...]:
    return tuple(sorted(item for item in value.split(";") if item))


def _legacy_effective(row: Mapping[str, str]) -> dict[str, object]:
    return {
        "curation_status": (row.get("curation_status") or "").strip() or "pass",
        "curation_flags": _flags(row.get("curation_flags", "")),
        "include_by_default": (row.get("include_by_default") or "").strip() != "0",
    }


@dataclass(frozen=True, slots=True)
class ParticipantTask:
    participant_id: str
    native_root: Path
    legacy_root: Path
    current_root: Path
    files: tuple[str, ...]
    max_examples: int


@dataclass(slots=True)
class ParticipantResult:
    participant_id: str
    files: int = 0
    rows: int = 0
    virtual_values_checked: int = 0
    policy_flags_removed: Counter[str] = field(default_factory=Counter)
    issues: list[dict[str, object]] = field(default_factory=list)
    issue_count: int = 0
    feature_rows: Counter[str] = field(default_factory=Counter)

    def add_issue(self, feature: str, row_index: int | None, check: str, detail: str, limit: int) -> None:
        self.issue_count += 1
        if len(self.issues) < limit:
            self.issues.append({
                "participant_id": self.participant_id,
                "feature": feature,
                "row_index": row_index,
                "check": check,
                "detail": detail,
            })

    def as_dict(self) -> dict[str, object]:
        return {
            "participant_id": self.participant_id,
            "files": self.files,
            "rows": self.rows,
            "virtual_values_checked": self.virtual_values_checked,
            "policy_flags_removed": dict(sorted(self.policy_flags_removed.items())),
            "issue_count": self.issue_count,
            "issues": self.issues,
            "feature_rows": dict(sorted(self.feature_rows.items())),
        }


def _compare_file(task: ParticipantTask, filename: str, result: ParticipantResult) -> None:
    feature = Path(filename).stem
    policy = get_policy(feature)
    native_path = task.native_root / task.participant_id / filename
    legacy_path = task.legacy_root / task.participant_id / filename
    current_path = task.current_root / task.participant_id / filename
    for label, path in (("native", native_path), ("schema-v1", legacy_path), ("schema-v2", current_path)):
        if not path.is_file():
            result.add_issue(feature, None, "missing_file", f"missing {label} file: {path}", task.max_examples)
            return

    with native_path.open("r", encoding="utf-8-sig", newline="") as nh, \
         legacy_path.open("r", encoding="utf-8-sig", newline="") as lh, \
         current_path.open("r", encoding="utf-8-sig", newline="") as ch:
        native_reader = csv.DictReader(nh)
        legacy_reader = csv.DictReader(lh)
        current_reader = csv.DictReader(ch)
        native_fields = tuple(native_reader.fieldnames or ())
        legacy_fields = tuple(legacy_reader.fieldnames or ())
        current_fields = tuple(current_reader.fieldnames or ())
        if legacy_fields[: len(native_fields)] != native_fields:
            result.add_issue(feature, None, "legacy_native_prefix", "schema-v1 native prefix differs", task.max_examples)
        if current_fields[: len(native_fields)] != native_fields:
            result.add_issue(feature, None, "current_native_prefix", "schema-v2 native prefix differs", task.max_examples)
        if set(current_fields[len(native_fields):]) & set(policy.virtual_columns):
            result.add_issue(feature, None, "virtual_materialized", "schema-v2 physically contains virtual columns", task.max_examples)

        index = 0
        while True:
            native_row = next(native_reader, None)
            legacy_row = next(legacy_reader, None)
            current_row = next(current_reader, None)
            if native_row is None and legacy_row is None and current_row is None:
                break
            if native_row is None or legacy_row is None or current_row is None:
                result.add_issue(feature, index, "row_count", "native/schema-v1/schema-v2 row counts differ", task.max_examples)
                break
            for column in native_fields:
                value = native_row.get(column, "")
                if legacy_row.get(column, "") != value:
                    result.add_issue(feature, index, "legacy_native_value", f"native value differs in {column}", task.max_examples)
                    break
                if current_row.get(column, "") != value:
                    result.add_issue(feature, index, "current_native_value", f"native value differs in {column}", task.max_examples)
                    break

            legacy_acquisition = legacy_row.get("acquisition_method", "")
            current_acquisition = current_row.get("acquisition_method", "")
            if legacy_acquisition != current_acquisition:
                result.add_issue(feature, index, "acquisition_method", f"{legacy_acquisition!r} != {current_acquisition!r}", task.max_examples)

            legacy_role = legacy_row.get("step_count_role", "")
            current_role = current_row.get("step_count_role", "")
            if legacy_role != current_role:
                result.add_issue(feature, index, "step_count_role", f"{legacy_role!r} != {current_role!r}", task.max_examples)

            legacy_decisions = _legacy_effective(legacy_row)
            current_decisions = effective_row_decisions(current_row)
            for key in ("curation_status", "include_by_default"):
                if legacy_decisions[key] != current_decisions[key]:
                    result.add_issue(feature, index, key, f"{legacy_decisions[key]!r} != {current_decisions[key]!r}", task.max_examples)

            legacy_flags = set(legacy_decisions["curation_flags"])
            current_flags = set(current_decisions["curation_flags"])
            removed = legacy_flags & POLICY_LEVEL_V1_FLAGS
            expected_current = legacy_flags - POLICY_LEVEL_V1_FLAGS
            result.policy_flags_removed.update(removed)
            if current_flags != expected_current:
                result.add_issue(
                    feature, index, "curation_flags",
                    f"expected {sorted(expected_current)}, observed {sorted(current_flags)}",
                    task.max_examples,
                )

            virtual = virtual_row_values(feature, native_row, policy)
            for column in policy.virtual_columns:
                expected = virtual.get(column, virtual_default(column))
                observed = legacy_row.get(column, "") or virtual_default(column)
                result.virtual_values_checked += 1
                if observed != expected:
                    result.add_issue(
                        feature, index, f"virtual:{column}",
                        f"schema-v1 {observed!r} != reconstructed {expected!r}",
                        task.max_examples,
                    )
            index += 1

    result.files += 1
    result.rows += index
    result.feature_rows[feature] += index


def _worker(task: ParticipantTask) -> ParticipantResult:
    result = ParticipantResult(participant_id=task.participant_id)
    for filename in task.files:
        _compare_file(task, filename, result)
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _publish(staging: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise InputLayoutError(f"audit output already exists; use --overwrite: {output}")
        shutil.rmtree(output)
    os.replace(staging, output)


def run_schema_migration_audit(
    native_root: Path,
    legacy_curated_root: Path,
    current_curated_root: Path,
    output: Path,
    *,
    workers: int = 8,
    max_in_flight: int | None = None,
    max_examples: int = 100,
    overwrite: bool = False,
    no_progress: bool = False,
) -> dict[str, object]:
    native, legacy, current, report = _guard_roots(native_root, legacy_curated_root, current_curated_root, output)
    native_state = native / NATIVE_STATE_FILENAME
    legacy_state = legacy / CURATION_STATE_FILENAME
    current_state = current / CURATION_STATE_FILENAME
    native_metadata = _metadata(native_state, "native")
    legacy_metadata = _metadata(legacy_state, "schema-v1 curated")
    current_metadata = _metadata(current_state, "schema-v2 curated")
    if native_metadata.get("source_key") != SOURCE_KEY:
        raise InputLayoutError(f"native source is not {SOURCE_KEY}: {native_metadata.get('source_key')!r}")
    if legacy_metadata.get("source_key") != SOURCE_KEY or legacy_metadata.get("product_kind") != "curated":
        raise InputLayoutError("schema-v1 root is not a managed Google Fitness curated root")
    if legacy_metadata.get("state_schema_version") != LEGACY_STATE_SCHEMA_VERSION:
        raise InputLayoutError(f"unexpected schema-v1 state version: {legacy_metadata.get('state_schema_version')!r}")
    if current_metadata.get("source_key") != SOURCE_KEY or current_metadata.get("product_kind") != "curated":
        raise InputLayoutError("schema-v2 root is not a managed Google Fitness curated root")
    if current_metadata.get("state_schema_version") != CURATION_STATE_SCHEMA_VERSION:
        raise InputLayoutError(f"unexpected schema-v2 state version: {current_metadata.get('state_schema_version')!r}")
    if current_metadata.get("curated_schema_version") != CURATED_SCHEMA_VERSION:
        raise InputLayoutError(f"unexpected curated schema version: {current_metadata.get('curated_schema_version')!r}")
    contract_path = current / CURATION_CONTRACT_FILENAME
    if not contract_path.is_file():
        raise InputLayoutError("schema-v2 root lacks .curation_contract.json")
    contract_issues = validate_contract_file(
        contract_path,
        engine_version=ENGINE_VERSION,
        pipeline_version=PIPELINE_VERSION,
        state_schema_version=CURATION_STATE_SCHEMA_VERSION,
    )
    if contract_issues:
        raise InputLayoutError("schema-v2 root contract is invalid: " + "; ".join(contract_issues))

    native_files = _participant_files(native)
    legacy_files = _participant_files(legacy)
    current_files = _participant_files(current)
    participants = sorted(set(native_files) | set(legacy_files) | set(current_files))
    setup_issues: list[dict[str, object]] = []
    tasks: list[ParticipantTask] = []
    for participant_id in participants:
        expected = native_files.get(participant_id, ())
        if legacy_files.get(participant_id, ()) != expected or current_files.get(participant_id, ()) != expected:
            setup_issues.append({
                "participant_id": participant_id,
                "feature": "",
                "row_index": None,
                "check": "file_set",
                "detail": (
                    f"native={list(expected)}, schema-v1={list(legacy_files.get(participant_id, ()))}, "
                    f"schema-v2={list(current_files.get(participant_id, ()))}"
                ),
            })
            continue
        tasks.append(ParticipantTask(participant_id, native, legacy, current, expected, max_examples))

    protected_files = (native_state, legacy_state, current_state, contract_path)
    state_hashes_before = {str(path): _sha256(path) for path in protected_files}
    started = time.perf_counter()
    worker_count = max(1, min(int(workers), os.cpu_count() or 1, len(tasks) or 1))
    in_flight = max(1, int(max_in_flight or worker_count))
    results: dict[str, ParticipantResult] = {}
    if worker_count == 1:
        iterator = tasks
        progress = tqdm(iterator, total=len(tasks), desc="Auditing curation schema migration", disable=no_progress, unit="participant")
        for task in progress:
            results[task.participant_id] = _worker(task)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            pending: dict[object, str] = {}
            task_iter = iter(tasks)
            progress = tqdm(total=len(tasks), desc="Auditing curation schema migration", disable=no_progress, unit="participant")
            try:
                while True:
                    while len(pending) < in_flight:
                        try:
                            task = next(task_iter)
                        except StopIteration:
                            break
                        pending[executor.submit(_worker, task)] = task.participant_id
                    if not pending:
                        break
                    done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for future in done:
                        participant_id = pending.pop(future)
                        results[participant_id] = future.result()
                        progress.update(1)
            finally:
                progress.close()

    all_issues = list(setup_issues)
    feature_rows: Counter[str] = Counter()
    removed: Counter[str] = Counter()
    files = rows = virtual_values = 0
    total_issue_count = len(setup_issues)
    participant_rows: list[dict[str, object]] = []
    for participant_id in sorted(results):
        item = results[participant_id]
        files += item.files
        rows += item.rows
        virtual_values += item.virtual_values_checked
        feature_rows.update(item.feature_rows)
        removed.update(item.policy_flags_removed)
        total_issue_count += item.issue_count
        if len(all_issues) < max_examples:
            all_issues.extend(item.issues[: max(0, max_examples - len(all_issues))])
        participant_rows.append({
            "participant_id": participant_id,
            "files": item.files,
            "rows": item.rows,
            "issues": item.issue_count,
        })

    state_hashes_after = {str(path): _sha256(path) for path in protected_files}
    state_unchanged = state_hashes_before == state_hashes_after
    checks = [
        {"check_id": "participant_file_sets", "status": "pass" if not setup_issues else "fail", "detail": f"setup issues: {len(setup_issues)}"},
        {"check_id": "row_level_semantic_equivalence", "status": "pass" if total_issue_count == 0 else "fail", "detail": f"issues: {total_issue_count}"},
        {"check_id": "managed_states_unchanged", "status": "pass" if state_unchanged else "fail", "detail": f"unchanged: {state_unchanged}"},
    ]
    status = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    payload: dict[str, object] = {
        "tool": "google_fitness_curation_schema_migration_audit",
        "audit_version": SCHEMA_MIGRATION_AUDIT_VERSION,
        "report_schema_version": SCHEMA_MIGRATION_REPORT_VERSION,
        "source_key": SOURCE_KEY,
        "status": status,
        "checks": checks,
        "participants_discovered": len(participants),
        "participants_compared": len(tasks),
        "files": files,
        "rows": rows,
        "virtual_values_checked": virtual_values,
        "policy_flags_removed": dict(sorted(removed.items())),
        "issue_count": total_issue_count,
        "issues": all_issues[:max_examples],
        "feature_rows": dict(sorted(feature_rows.items())),
        "state_schema_versions": {
            "schema_v1": legacy_metadata.get("state_schema_version"),
            "schema_v2": current_metadata.get("state_schema_version"),
        },
        "curated_schema_version": current_metadata.get("curated_schema_version"),
        "state_databases_unchanged": state_unchanged,
    }
    run_manifest = {
        "tool": "google_fitness_curation_schema_migration_audit",
        "audit_version": SCHEMA_MIGRATION_AUDIT_VERSION,
        "generated_at": _now(),
        "roots": {"native": str(native), "schema_v1": str(legacy), "schema_v2": str(current)},
        "output": str(report),
        "workers_requested": int(workers),
        "workers_effective": worker_count,
        "max_in_flight": in_flight,
        "wall_clock_seconds": time.perf_counter() - started,
    }

    report.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{report.name}.", dir=report.parent))
    try:
        _write_json(staging / "schema_v2_migration_audit.json", payload)
        _write_json(staging / "run_manifest.json", run_manifest)
        _write_json(staging / "issues.json", all_issues)
        with (staging / "participant_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("participant_id", "files", "rows", "issues"), lineterminator="\n")
            writer.writeheader()
            writer.writerows(participant_rows)
        report_lines = [
            "# Google Fitness curation schema-v2 migration audit",
            "",
            f"Status: **{status}**",
            f"Participants discovered: {len(participants):,}",
            f"Participants compared: {len(tasks):,}",
            f"Files: {files:,}",
            f"Rows: {rows:,}",
            f"Virtual values reconstructed: {virtual_values:,}",
            f"Issues: {total_issue_count:,}",
            "",
            "Policy-level schema-v1 flags removed:",
            *[f"- `{key}`: {value:,}" for key, value in sorted(removed.items())],
            "",
        ]
        (staging / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
        _publish(staging, report, overwrite)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-native", type=Path, required=True)
    parser.add_argument("--input-curated-v1", type=Path, required=True)
    parser.add_argument("--input-curated-v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--json-summary", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_schema_migration_audit(
            args.input_native,
            args.input_curated_v1,
            args.input_curated_v2,
            args.output,
            workers=args.workers,
            max_in_flight=args.max_in_flight,
            max_examples=args.max_examples,
            overwrite=args.overwrite,
            no_progress=args.no_progress,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json_summary:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"Schema migration audit: {result['status']} ({result['rows']:,} rows, {result['issue_count']} issue(s))")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
