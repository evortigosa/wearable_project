"""
Wearable Data Processing and Modeling project
Validated native CSV writing, source-qualified SQLite state, and atomic
participant-directory publication for Google Fitness WP4.
"""


from __future__ import annotations
import csv
import gzip
import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from wearable_project.exceptions import OutputValidationError
from wearable_project.processing.google_fitness.models import SourceFile
from wearable_project.processing.google_fitness.parser import sha256_file
from wearable_project.processing.google_fitness.registry import (
    REGISTRY_VERSION, SOURCE_KEY, UNREGISTERED_ACTIVITY_POLICY, get_feature_spec,
)


STATE_SCHEMA_VERSION = "google-fitness-native-state-wp4-0.1.0"
STATE_FILENAME = ".wearable_state.sqlite"


@dataclass(frozen=True, slots=True)
class FeatureOutputInfo:
    feature: str
    filename: str
    row_count: int
    size_bytes: int
    sha256: str
    represented_occurrences: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "filename": self.filename,
            "row_count": self.row_count,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "represented_occurrences": self.represented_occurrences,
        }


@dataclass(slots=True)
class StoredParticipantState:
    participant_id: str
    status: str
    parser_version: str | None
    registry_version: str | None
    cleaner_version: str | None
    pipeline_version: str | None
    source_signature: str | None
    last_error: str | None
    sources: dict[str, dict[str, Any]]
    outputs: dict[str, FeatureOutputInfo]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _new_temporary_file(directory: Path, prefix: str, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        path = directory / f"{prefix}{secrets.token_hex(8)}{suffix}"
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
        except FileExistsError:
            continue
        return path
    raise FileExistsError(f"could not create a temporary file in {directory}")


def read_only_mode(database: Path) -> str:
    """
    SQLite URI parameters for reading a state database without ever writing beside it.
    A plain ``mode=ro`` connection to a WAL database must create the ``-shm`` index next to it: it fails for a reader
    without write access to the root, and otherwise leaves ``-shm`` and ``-wal`` files in the data tree. With no write
    pending, nothing can change under the reader, so the database is opened immutable, which never writes. A non-empty
    write-ahead log or rollback journal, the evidence of a writer at work that the DataLoaders also use, may still hold
    committed pages, so the database is then read normally to see them.
    """

    for suffix in ("-wal", "-journal"):
        side = database.with_name(database.name + suffix)
        try:
            if side.is_file() and side.stat().st_size > 0:
                return "mode=ro"
        except OSError:
            return "mode=ro"
    return "mode=ro&immutable=1"


def _schema_for_feature(feature: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if feature == UNREGISTERED_ACTIVITY_POLICY.native_feature:
        return (
            f"{feature}.csv",
            UNREGISTERED_ACTIVITY_POLICY.column_names,
            (
                "raw_activity_name",
                "start_date",
                "end_date",
                "collecting_method_version",
                "outer_id",
                "payload_index",
            ),
        )
    spec = get_feature_spec(feature)
    return spec.filename, spec.column_names, spec.required_columns


def safe_feature_filename(feature: str) -> str:
    filename, _, _ = _schema_for_feature(feature)
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        raise OutputValidationError(f"unsafe registered feature filename: {filename!r}")
    if filename == "Activities.csv":
        raise OutputValidationError("Activities.csv is prohibited by the Google Fitness registry")
    return filename


def _serialized_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return value


def write_feature_csv(
    feature: str, rows: Sequence[Mapping[str, Any]], destination: Path, *, represented_occurrences: int,
) -> FeatureOutputInfo:
    """Write one already validated, deterministically ordered feature file."""

    filename, columns, required = _schema_for_feature(feature)
    if destination.name != filename:
        raise OutputValidationError(
            f"destination {destination.name!r} does not match registered filename {filename!r}"
        )
    if not rows:
        raise OutputValidationError(f"refusing to write an empty native feature file: {feature}")
    required_set = set(required)
    allowed = set(columns)
    temporary = _new_temporary_file(destination.parent, prefix=f".{destination.name}.", suffix=".tmp",)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(columns),
                extrasaction="raise",
                lineterminator="\n",
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            for index, row in enumerate(rows):
                extra = sorted(set(row) - allowed)
                if extra:
                    raise OutputValidationError(f"{feature} row {index} contains unregistered columns: {extra}")
                missing = sorted(name for name in required_set if row.get(name) is None)
                if missing:
                    raise OutputValidationError(f"{feature} row {index} lacks required values: {missing}")
                writer.writerow({name: _serialized_cell(row.get(name)) for name in columns})
            handle.flush()
            os.fsync(handle.fileno())
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return FeatureOutputInfo(
        feature=feature,
        filename=filename,
        row_count=len(rows),
        size_bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
        represented_occurrences=int(represented_occurrences),
    )


def validate_feature_file(
    path: Path, *, feature: str, expected: FeatureOutputInfo | None = None,
) -> FeatureOutputInfo:
    _csv_field_limit()
    filename, columns, required = _schema_for_feature(feature)
    if path.name != filename:
        raise OutputValidationError(f"{path} does not use the registered filename {filename!r}")
    if path.name == "Activities.csv":
        raise OutputValidationError("Activities.csv is prohibited")
    count = 0
    required_set = set(required)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            actual_header = tuple(reader.fieldnames or ())
            if actual_header != columns:
                raise OutputValidationError(
                    f"{path} header differs from registry: expected={columns}, actual={actual_header}"
                )
            for row_number, row in enumerate(reader, start=2):
                missing = sorted(name for name in required_set if row.get(name) in (None, ""))
                if missing:
                    raise OutputValidationError(f"{path}:{row_number} lacks required values: {missing}")
                count += 1
    except csv.Error as exc:
        raise OutputValidationError(f"could not parse {path}: {exc}") from exc
    if count == 0:
        raise OutputValidationError(f"{path} contains no data rows")
    info = FeatureOutputInfo(
        feature=feature,
        filename=path.name,
        row_count=count,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        represented_occurrences=(expected.represented_occurrences if expected is not None else count),
    )
    if expected is not None:
        for attribute in ("filename", "row_count", "size_bytes", "sha256"):
            if getattr(info, attribute) != getattr(expected, attribute):
                raise OutputValidationError(f"{path} {attribute} differs from the worker manifest")
    return info


def validate_participant_stage(
    stage_dir: Path, participant_id: str, expected_outputs: Iterable[FeatureOutputInfo],
) -> list[FeatureOutputInfo]:
    outputs = list(expected_outputs)
    expected_names = {item.filename for item in outputs}
    actual_names = {path.name for path in stage_dir.glob("*.csv")}
    if "Activities.csv" in actual_names:
        raise OutputValidationError(f"staged participant {participant_id} contains prohibited Activities.csv")
    if expected_names != actual_names:
        raise OutputValidationError(
            f"staged files differ from the worker manifest for {participant_id}: "
            f"expected={sorted(expected_names)}, actual={sorted(actual_names)}"
        )
    seen: set[str] = set()
    validated: list[FeatureOutputInfo] = []
    for expected in outputs:
        if expected.feature in seen:
            raise OutputValidationError(f"worker manifest repeats feature {expected.feature!r}")
        seen.add(expected.feature)
        path = stage_dir / expected.filename
        if not path.is_file():
            raise OutputValidationError(f"missing staged output: {path}")
        validated.append(validate_feature_file(path, feature=expected.feature, expected=expected))
    return validated


def source_signature(sources: Iterable[SourceFile]) -> str:
    payload = [
        {
            "month": item.canonical_month,
            "filename": item.path.name,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in sorted(sources, key=lambda value: value.canonical_month)
    ]
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def existing_output_is_usable(
    participant_dir: Path, state: StoredParticipantState, *, verify_hashes: bool = False,
) -> bool:
    if state.status == "complete_empty":
        return not participant_dir.exists() or not any(participant_dir.glob("*.csv"))
    if state.status != "complete" or not participant_dir.is_dir() or not state.outputs:
        return False
    if {path.name for path in participant_dir.glob("*.csv")} != {
        item.filename for item in state.outputs.values()
    }:
        return False
    for output in state.outputs.values():
        path = participant_dir / output.filename
        if not path.is_file() or path.stat().st_size != output.size_bytes:
            return False
        if verify_hashes and sha256_file(path) != output.sha256:
            return False
    return True


def write_ledger(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _new_temporary_file(path.parent, f".{path.name}.", ".tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_ledger(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise OutputValidationError(f"invalid worker ledger: {path}")
    return payload


class StateDatabase:
    """Source-qualified Google Fitness native state; parent process is sole writer."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        try:
            self._initialize()
        except Exception:
            self.connection.close()
            raise

    def __enter__(self) -> "StateDatabase":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        metadata_exists = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
        ).fetchone()
        if metadata_exists is not None:
            existing = {
                row["key"]: row["value"]
                for row in self.connection.execute("SELECT key,value FROM metadata")
            }
            if existing.get("source_key") not in (None, SOURCE_KEY):
                raise OutputValidationError(
                    f"state database is managed by source {existing['source_key']!r}, not {SOURCE_KEY!r}"
                )
            if existing.get("state_schema_version") not in (None, STATE_SCHEMA_VERSION):
                raise OutputValidationError(
                    f"incompatible Google Fitness state schema: {existing['state_schema_version']!r}"
                )

        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs(
              run_id TEXT PRIMARY KEY,
              started_at TEXT NOT NULL,
              finished_at TEXT,
              input_root TEXT NOT NULL,
              output_root TEXT NOT NULL,
              status TEXT NOT NULL,
              summary_json TEXT
            );
            CREATE TABLE IF NOT EXISTS participants(
              participant_id TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              parser_version TEXT,
              registry_version TEXT,
              cleaner_version TEXT,
              pipeline_version TEXT,
              source_signature TEXT,
              last_error TEXT,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS source_files(
              participant_id TEXT NOT NULL,
              canonical_month TEXT NOT NULL,
              filename TEXT NOT NULL,
              sha256 TEXT NOT NULL,
              size_bytes INTEGER NOT NULL,
              PRIMARY KEY(participant_id, canonical_month)
            );
            CREATE TABLE IF NOT EXISTS feature_outputs(
              participant_id TEXT NOT NULL,
              feature TEXT NOT NULL,
              filename TEXT NOT NULL,
              row_count INTEGER NOT NULL,
              size_bytes INTEGER NOT NULL,
              sha256 TEXT NOT NULL,
              represented_occurrences INTEGER NOT NULL,
              PRIMARY KEY(participant_id, feature)
            );
            CREATE TABLE IF NOT EXISTS source_occurrences(
              participant_id TEXT NOT NULL,
              occurrence_key TEXT NOT NULL,
              outer_participant_id TEXT,
              raw_source_label TEXT,
              source_feature TEXT,
              collecting_method_version TEXT,
              outer_id TEXT,
              outer_datetime_raw TEXT,
              created_at_raw TEXT,
              updated_at_raw TEXT,
              source_file TEXT,
              canonical_month TEXT,
              source_file_sha256 TEXT,
              outer_row_number INTEGER,
              raw_payload_sha256 TEXT,
              decoded_payload_sha256 TEXT,
              payload_item_count INTEGER,
              PRIMARY KEY(participant_id, occurrence_key)
            );
            CREATE TABLE IF NOT EXISTS malformed_candidates(
              participant_id TEXT NOT NULL,
              source_file TEXT,
              canonical_month TEXT,
              row_number INTEGER,
              outer_id TEXT,
              source_feature TEXT,
              collecting_method_version TEXT,
              payload_index INTEGER,
              reason_codes_json TEXT NOT NULL,
              item_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS parser_diagnostics(
              participant_id TEXT NOT NULL,
              source_file TEXT,
              canonical_month TEXT,
              row_number INTEGER,
              outer_id TEXT,
              feature TEXT,
              collecting_method_version TEXT,
              payload_index INTEGER,
              stage TEXT,
              code TEXT,
              message TEXT
            );
            CREATE TABLE IF NOT EXISTS reconciliation_decisions(
              participant_id TEXT NOT NULL,
              feature TEXT NOT NULL,
              stage TEXT NOT NULL,
              action TEXT NOT NULL,
              group_id TEXT NOT NULL,
              identity_json TEXT NOT NULL,
              representative_candidate_keys_json TEXT NOT NULL,
              removed_candidate_keys_json TEXT NOT NULL,
              all_candidate_keys_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS exceptional_row_audits(
              participant_id TEXT NOT NULL,
              feature TEXT NOT NULL,
              row_id TEXT NOT NULL,
              audit_json TEXT NOT NULL,
              PRIMARY KEY(participant_id, feature, row_id)
            );
            CREATE TABLE IF NOT EXISTS run_participants(
              run_id TEXT NOT NULL,
              participant_id TEXT NOT NULL,
              planned_action TEXT NOT NULL,
              plan_reason TEXT NOT NULL,
              final_status TEXT NOT NULL,
              result_json TEXT NOT NULL,
              error_message TEXT,
              PRIMARY KEY(run_id, participant_id)
            );
            CREATE TABLE IF NOT EXISTS run_features(
              run_id TEXT NOT NULL,
              participant_id TEXT NOT NULL,
              feature TEXT NOT NULL,
              result_json TEXT NOT NULL,
              PRIMARY KEY(run_id, participant_id, feature)
            );
            """
        )
        self.connection.executemany(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            (
                ("source_key", SOURCE_KEY),
                ("state_schema_version", STATE_SCHEMA_VERSION),
                ("registry_version", REGISTRY_VERSION),
            ),
        )
        self.connection.commit()

    def begin_run(self, run_id: str, input_root: Path, output_root: Path) -> None:
        self.connection.execute(
            "INSERT INTO runs(run_id,started_at,input_root,output_root,status) VALUES(?,?,?,?,?)",
            (run_id, now_utc(), str(input_root), str(output_root), "in_progress"),
        )
        self.connection.commit()

    def finish_run(self, run_id: str, status: str, summary: Mapping[str, Any]) -> None:
        self.connection.execute(
            "UPDATE runs SET finished_at=?,status=?,summary_json=? WHERE run_id=?",
            (
                now_utc(), status, json.dumps(summary, ensure_ascii=False, sort_keys=True), run_id,
            ),
        )
        self.connection.commit()

    def get_participant(self, participant_id: str) -> StoredParticipantState | None:
        row = self.connection.execute(
            "SELECT * FROM participants WHERE participant_id=?", (participant_id,)
        ).fetchone()
        if row is None:
            return None
        sources = {
            item["canonical_month"]: {
                "filename": item["filename"],
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            }
            for item in self.connection.execute(
                "SELECT * FROM source_files WHERE participant_id=?", (participant_id,)
            )
        }
        outputs = {
            item["feature"]: FeatureOutputInfo(
                feature=item["feature"],
                filename=item["filename"],
                row_count=item["row_count"],
                size_bytes=item["size_bytes"],
                sha256=item["sha256"],
                represented_occurrences=item["represented_occurrences"],
            )
            for item in self.connection.execute(
                "SELECT * FROM feature_outputs WHERE participant_id=?", (participant_id,)
            )
        }
        return StoredParticipantState(
            participant_id=row["participant_id"],
            status=row["status"],
            parser_version=row["parser_version"],
            registry_version=row["registry_version"],
            cleaner_version=row["cleaner_version"],
            pipeline_version=row["pipeline_version"],
            source_signature=row["source_signature"],
            last_error=row["last_error"],
            sources=sources,
            outputs=outputs,
        )

    def mark_in_progress(
        self, participant_id: str, *, parser_version: str, registry_version: str, cleaner_version: str,
        pipeline_version: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO participants(
              participant_id,status,parser_version,registry_version,
              cleaner_version,pipeline_version,updated_at
            ) VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(participant_id) DO UPDATE SET
              status='in_progress',parser_version=excluded.parser_version,
              registry_version=excluded.registry_version,
              cleaner_version=excluded.cleaner_version,
              pipeline_version=excluded.pipeline_version,
              last_error=NULL,updated_at=excluded.updated_at
            """,
            (
                participant_id,
                "in_progress",
                parser_version,
                registry_version,
                cleaner_version,
                pipeline_version,
                now_utc(),
            ),
        )
        self.connection.commit()

    def mark_failed(self, participant_id: str, message: str) -> None:
        self.connection.execute(
            """
            INSERT INTO participants(participant_id,status,last_error,updated_at)
            VALUES(?,?,?,?)
            ON CONFLICT(participant_id) DO UPDATE SET
              status='failed',last_error=excluded.last_error,updated_at=excluded.updated_at
            """,
            (participant_id, "failed", message, now_utc()),
        )
        self.connection.commit()

    def record_run_participant(
        self, run_id: str, participant_id: str, *, planned_action: str, plan_reason: str, final_status: str,
        result: Mapping[str, Any], feature_stats: Mapping[str, Mapping[str, Any]] = (),
        error_message: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO run_participants(
              run_id,participant_id,planned_action,plan_reason,
              final_status,result_json,error_message
            ) VALUES(?,?,?,?,?,?,?)
            """,
            (
                run_id,
                participant_id,
                planned_action,
                plan_reason,
                final_status,
                json.dumps(result, ensure_ascii=False, sort_keys=True),
                error_message,
            ),
        )
        self.connection.execute(
            "DELETE FROM run_features WHERE run_id=? AND participant_id=?",
            (run_id, participant_id),
        )
        self.connection.executemany(
            "INSERT INTO run_features(run_id,participant_id,feature,result_json) VALUES(?,?,?,?)",
            [
                (
                    run_id, participant_id, feature, json.dumps(payload, ensure_ascii=False, sort_keys=True),
                )
                for feature, payload in sorted(dict(feature_stats).items())
            ],
        )
        self.connection.commit()

    def commit_participant(
        self, *, participant_id: str, parser_version: str, registry_version: str, cleaner_version: str,
        pipeline_version: str, signature: str, sources: Iterable[SourceFile], outputs: Iterable[FeatureOutputInfo],
        ledger: Mapping[str, Any], status: str,
    ) -> None:
        source_items = list(sources)
        output_items = list(outputs)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO participants(
                  participant_id,status,parser_version,registry_version,
                  cleaner_version,pipeline_version,source_signature,last_error,updated_at
                ) VALUES(?,?,?,?,?,?,?,NULL,?)
                ON CONFLICT(participant_id) DO UPDATE SET
                  status=excluded.status,parser_version=excluded.parser_version,
                  registry_version=excluded.registry_version,
                  cleaner_version=excluded.cleaner_version,
                  pipeline_version=excluded.pipeline_version,
                  source_signature=excluded.source_signature,
                  last_error=NULL,updated_at=excluded.updated_at
                """,
                (
                    participant_id,
                    status,
                    parser_version,
                    registry_version,
                    cleaner_version,
                    pipeline_version,
                    signature,
                    now_utc(),
                ),
            )
            self.connection.execute(
                "DELETE FROM source_files WHERE participant_id=?", (participant_id,)
            )
            self.connection.executemany(
                "INSERT INTO source_files VALUES(?,?,?,?,?)",
                [
                    (
                        participant_id,
                        item.canonical_month,
                        item.path.name,
                        item.sha256,
                        item.size_bytes,
                    )
                    for item in source_items
                ],
            )
            self.connection.execute(
                "DELETE FROM feature_outputs WHERE participant_id=?", (participant_id,)
            )
            self.connection.executemany(
                """
                INSERT INTO feature_outputs(
                  participant_id,feature,filename,row_count,size_bytes,sha256,
                  represented_occurrences
                ) VALUES(?,?,?,?,?,?,?)
                """,
                [
                    (
                        participant_id,
                        item.feature,
                        item.filename,
                        item.row_count,
                        item.size_bytes,
                        item.sha256,
                        item.represented_occurrences,
                    )
                    for item in output_items
                ],
            )
            for table in (
                "source_occurrences",
                "malformed_candidates",
                "parser_diagnostics",
                "reconciliation_decisions",
                "exceptional_row_audits",
            ):
                self.connection.execute(f"DELETE FROM {table} WHERE participant_id=?", (participant_id,))
            self.connection.executemany(
                """
                INSERT INTO source_occurrences VALUES(
                  ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                [
                    (
                        participant_id,
                        item["occurrence_key"],
                        item.get("outer_participant_id"),
                        item.get("raw_source_label"),
                        item.get("source_feature"),
                        item.get("collecting_method_version"),
                        item.get("outer_id"),
                        item.get("outer_datetime_raw"),
                        item.get("created_at_raw"),
                        item.get("updated_at_raw"),
                        item.get("source_file"),
                        item.get("canonical_month"),
                        item.get("source_file_sha256"),
                        item.get("outer_row_number"),
                        item.get("raw_payload_sha256"),
                        item.get("decoded_payload_sha256"),
                        item.get("payload_item_count"),
                    )
                    for item in ledger.get("source_occurrences", [])
                ],
            )
            self.connection.executemany(
                "INSERT INTO malformed_candidates VALUES(?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        participant_id,
                        item.get("source_file"),
                        item.get("canonical_month"),
                        item.get("row_number"),
                        item.get("outer_id"),
                        item.get("source_feature"),
                        item.get("collecting_method_version"),
                        item.get("payload_index"),
                        json.dumps(item.get("reason_codes", []), sort_keys=True),
                        item.get("item_json", ""),
                    )
                    for item in ledger.get("malformed_candidates", [])
                ],
            )
            self.connection.executemany(
                "INSERT INTO parser_diagnostics VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        participant_id,
                        item.get("source_file"),
                        item.get("canonical_month"),
                        item.get("row_number"),
                        item.get("outer_id"),
                        item.get("feature"),
                        item.get("collecting_method_version"),
                        item.get("payload_index"),
                        item.get("stage"),
                        item.get("code"),
                        item.get("message"),
                    )
                    for item in ledger.get("parser_diagnostics", [])
                ],
            )
            self.connection.executemany(
                "INSERT INTO reconciliation_decisions VALUES(?,?,?,?,?,?,?,?,?)",
                [
                    (
                        participant_id,
                        item["feature"],
                        item["stage"],
                        item["action"],
                        item["group_id"],
                        item["identity_json"],
                        json.dumps(item.get("representative_candidate_keys", []), sort_keys=True),
                        json.dumps(item.get("removed_candidate_keys", []), sort_keys=True),
                        json.dumps(item.get("all_candidate_keys", []), sort_keys=True),
                    )
                    for item in ledger.get("reconciliation_decisions", [])
                ],
            )
            self.connection.executemany(
                "INSERT INTO exceptional_row_audits VALUES(?,?,?,?)",
                [
                    (
                        participant_id,
                        item["feature"],
                        item["row_id"],
                        json.dumps(item, ensure_ascii=False, sort_keys=True),
                    )
                    for item in ledger.get("exceptional_row_audits", [])
                ],
            )


@dataclass(slots=True)
class DirectorySwap:
    final_dir: Path
    stage_dir: Path
    backup_dir: Path
    had_previous: bool = False
    applied: bool = False

    def apply(self) -> None:
        self.backup_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.backup_dir, ignore_errors=True)
        self.had_previous = self.final_dir.exists()
        if self.had_previous:
            os.replace(self.final_dir, self.backup_dir)
        try:
            os.replace(self.stage_dir, self.final_dir)
            self.applied = True
        except Exception:
            if self.had_previous and self.backup_dir.exists() and not self.final_dir.exists():
                os.replace(self.backup_dir, self.final_dir)
            raise

    def rollback(self) -> None:
        if self.applied and self.final_dir.exists():
            os.replace(self.final_dir, self.stage_dir)
        if self.had_previous and self.backup_dir.exists():
            os.replace(self.backup_dir, self.final_dir)
        self.applied = False

    def finalize(self) -> None:
        shutil.rmtree(self.backup_dir, ignore_errors=True)
        shutil.rmtree(self.stage_dir, ignore_errors=True)
