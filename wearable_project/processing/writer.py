"""
Wearable Data Processing and Modeling project
Atomic participant-folder writing and compact SQLite processing state.
"""


from __future__ import annotations
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from wearable_project.exceptions import OutputValidationError
from wearable_project.processing.parser import SourceFile, sha256_file
from wearable_project.processing.registry import FeatureFamily, get_feature_spec


@dataclass(frozen=True, slots=True)
class FeatureOutputInfo:
    feature: str
    filename: str
    row_count: int
    size_bytes: int
    sha256: str


@dataclass(slots=True)
class StoredParticipantState:
    participant_id: str
    status: str
    parser_version: str | None
    registry_version: str | None
    source_signature: str | None
    last_error: str | None
    sources: dict[str, dict[str, Any]]
    outputs: dict[str, FeatureOutputInfo]



def _csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_feature_filename(feature: str) -> str:
    stem = "".join(char if char.isalnum() or char in "_-" else "_" for char in feature).strip("_")
    return f"{stem or 'feature'}.csv"


def atomic_write_csv(dataframe: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        dataframe.to_csv(temporary, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def drop_file_cache(path: Path) -> None:
    """
    Best-effort release of file-backed page cache on Linux. Large participant CSVs can otherwise keep hundreds
    of megabytes charged to a memory-constrained single node after a worker exits. This does not alter file
    contents and is a no-op where posix_fadvise is unavailable.
    """
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    try:
        with path.open("rb") as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass


def load_csv_records(path: Path) -> list[dict[str, Any]]:
    _csv_field_limit()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_stage(stage_dir: Path, existing_dir: Path | None, incremental: bool) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    if incremental and existing_dir and existing_dir.is_dir():
        for path in existing_dir.glob("*.csv"):
            link_or_copy(path, stage_dir / path.name)


def validate_feature_file(path: Path, participant_id: str, expected_feature: str) -> FeatureOutputInfo:
    _csv_field_limit()
    row_count = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            spec = get_feature_spec(expected_feature, fields)
            required = {"datetime", "created_at", "updated_at", "data_source", "collecting_method_version"}
            if spec.family != FeatureFamily.DAILY_SUMMARY:
                required.update({"start_date", "end_date"})
            required.update(spec.measurement_columns)
            missing_columns = required.difference(fields)
            if missing_columns:
                raise OutputValidationError(f"{path} lacks required columns: {sorted(missing_columns)}")
            for row in reader:
                if str(row.get("data_source", "")).strip().lower() != "applehealthkit":
                    raise OutputValidationError(f"{path} contains a non-AppleHealthkit row")
                row_count += 1
    except csv.Error as exc:
        raise OutputValidationError(f"Could not parse {path}: {exc}") from exc
    if row_count == 0:
        raise OutputValidationError(f"{path} has no data rows")
    return FeatureOutputInfo(expected_feature, path.name, row_count, path.stat().st_size, sha256_file(path))


def validate_participant_stage(
    stage_dir: Path, participant_id: str, expected_outputs: Iterable[FeatureOutputInfo]
) -> list[FeatureOutputInfo]:
    """
    Validate a worker-produced manifest without scanning every data row. Participant and feature identity are
    intentionally encoded by the parent directory and CSV filename rather than repeated in every row. The worker
    validates internal event identity before projection; the parent verifies the committed file set, checksums,
    native feature schema, and first-row Apple source marker.
    """
    _csv_field_limit()
    outputs = list(expected_outputs)
    expected_files = {item.filename for item in outputs}
    actual_files = {path.name for path in stage_dir.glob("*.csv")}
    if expected_files != actual_files:
        raise OutputValidationError(
            f"Staged files differ from worker manifest for {participant_id}: "
            f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
        )
    if not outputs:
        raise OutputValidationError(f"No feature CSV was generated for participant {participant_id}")
    seen_features: set[str] = set()
    for info in outputs:
        if info.feature in seen_features:
            raise OutputValidationError(f"Worker manifest repeats feature {info.feature!r}")
        seen_features.add(info.feature)
        path = stage_dir / info.filename
        if not path.is_file() or path.stat().st_size != info.size_bytes:
            raise OutputValidationError(f"Staged size mismatch for {path}")
        if sha256_file(path) != info.sha256:
            raise OutputValidationError(f"Staged checksum mismatch for {path}")
        drop_file_cache(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            spec = get_feature_spec(info.feature, fields)
            required = {"datetime", "created_at", "updated_at", "data_source", "collecting_method_version"}
            if spec.family != FeatureFamily.DAILY_SUMMARY:
                required.update({"start_date", "end_date"})
            required.update(spec.measurement_columns)
            missing_columns = required.difference(fields)
            if missing_columns:
                raise OutputValidationError(f"{path} lacks required columns: {sorted(missing_columns)}")
            try:
                first = next(reader)
            except StopIteration as exc:
                raise OutputValidationError(f"{path} contains no data rows") from exc
        if str(first.get("data_source", "")).strip().lower() != "applehealthkit":
            raise OutputValidationError(f"First row in {path} is not AppleHealthkit data")
    return outputs


def source_signature(sources: Iterable[SourceFile]) -> str:
    payload = [
        {"month": item.canonical_month, "filename": item.path.name, "sha256": item.sha256, "size": item.size_bytes}
        for item in sorted(sources, key=lambda value: value.canonical_month)
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class StateDatabase:
    """The parent process is the only writer to this hidden SQLite file."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs(
              run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
              input_root TEXT NOT NULL, output_root TEXT NOT NULL, status TEXT NOT NULL,
              summary_json TEXT
            );
            CREATE TABLE IF NOT EXISTS participants(
              participant_id TEXT PRIMARY KEY, status TEXT NOT NULL,
              parser_version TEXT, registry_version TEXT, source_signature TEXT,
              last_error TEXT, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS source_files(
              participant_id TEXT NOT NULL, canonical_month TEXT NOT NULL,
              filename TEXT NOT NULL, sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
              PRIMARY KEY(participant_id, canonical_month)
            );
            CREATE TABLE IF NOT EXISTS feature_outputs(
              participant_id TEXT NOT NULL, feature TEXT NOT NULL, filename TEXT NOT NULL,
              row_count INTEGER NOT NULL, size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
              PRIMARY KEY(participant_id, feature)
            );
            """
        )
        self.connection.commit()


    def __enter__(self) -> "StateDatabase":
        return self


    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.connection.close()


    def begin_run(self, run_id: str, input_root: Path, output_root: Path) -> None:
        self.connection.execute(
            "INSERT INTO runs(run_id,started_at,input_root,output_root,status) VALUES(?,?,?,?,?)",
            (run_id, now_utc(), str(input_root), str(output_root), "in_progress"),
        )
        self.connection.commit()


    def finish_run(self, run_id: str, status: str, summary: dict[str, Any]) -> None:
        self.connection.execute(
            "UPDATE runs SET finished_at=?,status=?,summary_json=? WHERE run_id=?",
            (now_utc(), status, json.dumps(summary, sort_keys=True), run_id),
        )
        self.connection.commit()


    def participant_ids(self) -> set[str]:
        return {row[0] for row in self.connection.execute("SELECT participant_id FROM participants")}


    def get_participant(self, participant_id: str) -> StoredParticipantState | None:
        row = self.connection.execute("SELECT * FROM participants WHERE participant_id=?", (participant_id,)).fetchone()
        if row is None:
            return None
        sources = {
            item["canonical_month"]: {"filename": item["filename"], "sha256": item["sha256"], "size_bytes": item["size_bytes"]}
            for item in self.connection.execute("SELECT * FROM source_files WHERE participant_id=?", (participant_id,))
        }
        outputs = {
            item["feature"]: FeatureOutputInfo(item["feature"], item["filename"], item["row_count"], item["size_bytes"], item["sha256"])
            for item in self.connection.execute("SELECT * FROM feature_outputs WHERE participant_id=?", (participant_id,))
        }
        return StoredParticipantState(row["participant_id"], row["status"], row["parser_version"], row["registry_version"], row["source_signature"], row["last_error"], sources, outputs)


    def mark_in_progress(self, participant_id: str, parser_version: str, registry_version: str) -> None:
        self.connection.execute(
            """INSERT INTO participants(participant_id,status,parser_version,registry_version,updated_at)
               VALUES(?,?,?,?,?) ON CONFLICT(participant_id) DO UPDATE SET
               status='in_progress',parser_version=excluded.parser_version,
               registry_version=excluded.registry_version,last_error=NULL,updated_at=excluded.updated_at""",
            (participant_id, "in_progress", parser_version, registry_version, now_utc()),
        )
        self.connection.commit()


    def mark_failed(self, participant_id: str, message: str) -> None:
        self.connection.execute(
            """INSERT INTO participants(participant_id,status,last_error,updated_at) VALUES(?,?,?,?)
               ON CONFLICT(participant_id) DO UPDATE SET status='failed',last_error=excluded.last_error,updated_at=excluded.updated_at""",
            (participant_id, "failed", message, now_utc()),
        )
        self.connection.commit()


    def commit_participant(
        self, participant_id: str, parser_version: str, registry_version: str,
        signature: str, sources: Iterable[SourceFile], outputs: Iterable[FeatureOutputInfo],
        *, status: str = "complete",
    ) -> None:
        sources, outputs = list(sources), list(outputs)
        with self.connection:
            self.connection.execute(
                """INSERT INTO participants(participant_id,status,parser_version,registry_version,source_signature,last_error,updated_at)
                   VALUES(?,?,?,?,?,NULL,?) ON CONFLICT(participant_id) DO UPDATE SET
                   status=excluded.status,parser_version=excluded.parser_version,
                   registry_version=excluded.registry_version,source_signature=excluded.source_signature,
                   last_error=NULL,updated_at=excluded.updated_at""",
                (participant_id, status, parser_version, registry_version, signature, now_utc()),
            )
            self.connection.execute("DELETE FROM source_files WHERE participant_id=?", (participant_id,))
            self.connection.executemany(
                "INSERT INTO source_files VALUES(?,?,?,?,?)",
                [(participant_id, item.canonical_month, item.path.name, item.sha256, item.size_bytes) for item in sources],
            )
            self.connection.execute("DELETE FROM feature_outputs WHERE participant_id=?", (participant_id,))
            self.connection.executemany(
                "INSERT INTO feature_outputs VALUES(?,?,?,?,?,?)",
                [(participant_id, item.feature, item.filename, item.row_count, item.size_bytes, item.sha256) for item in outputs],
            )


def existing_output_is_usable(participant_dir: Path, state: StoredParticipantState, verify_hashes: bool = False) -> bool:
    if not participant_dir.is_dir() or not state.outputs:
        return False
    for output in state.outputs.values():
        path = participant_dir / output.filename
        if not path.is_file() or path.stat().st_size != output.size_bytes:
            return False
        if verify_hashes and sha256_file(path) != output.sha256:
            return False
    return True


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
