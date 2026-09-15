"""
Wearable Data Processing and Modeling project
Transactional SQLite state and reporting for curated participant outputs.
"""


from __future__ import annotations
import json
import sqlite3
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from wearable_project.curation.decisions import decisions_fingerprint
from wearable_project.curation.registry import registry_fingerprint
from wearable_project.processing.parser import sha256_file


CURATION_ENGINE_VERSION = "curation-engine-0.2.0a2"
CURATION_STATE_SCHEMA_VERSION = 1


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class NativeFileState:
    feature: str
    filename: str
    row_count: int
    size_bytes: int
    sha256: str
    policy_fingerprint: str


@dataclass(frozen=True, slots=True)
class CuratedOutputState:
    feature: str
    filename: str
    native_rows: int
    curated_rows: int
    size_bytes: int
    sha256: str
    native_sha256: str
    policy_fingerprint: str
    pass_rows: int
    review_rows: int
    exclude_default_rows: int
    canonical_value_rows: int
    ambiguous_unit_rows: int
    scale_transition_rows: int
    flag_counts: dict[str, int]
    acquisition_counts: dict[str, int]


@dataclass(slots=True)
class StoredCurationParticipant:
    participant_id: str
    status: str
    native_signature: str | None
    policy_signature: str | None
    engine_version: str | None
    last_error: str | None
    native_files: dict[str, NativeFileState]
    outputs: dict[str, CuratedOutputState]


@dataclass(slots=True)
class CurationRunSummary:
    run_id: str
    discovered: int = 0
    skipped: int = 0
    rebuilt: int = 0
    committed: int = 0
    failed: int = 0
    failures: dict[str, str] = field(default_factory=dict)
    report: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.report or {
            "run_id": self.run_id,
            "participants": {
                "discovered": self.discovered,
                "skipped": self.skipped,
                "rebuilt": self.rebuilt,
                "committed": self.committed,
                "failed": self.failed,
            },
            "failures": self.failures,
        }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class CurationStateDatabase:
    """Parent-process-only writer for the hidden curation-state database."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS curation_runs(
              run_id TEXT PRIMARY KEY,
              started_at TEXT NOT NULL,
              finished_at TEXT,
              input_native_root TEXT NOT NULL,
              output_root TEXT NOT NULL,
              status TEXT NOT NULL,
              configuration_json TEXT NOT NULL,
              summary_json TEXT
            );
            CREATE TABLE IF NOT EXISTS curation_participants(
              participant_id TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              native_signature TEXT,
              policy_signature TEXT,
              engine_version TEXT,
              last_error TEXT,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS curation_native_files(
              participant_id TEXT NOT NULL,
              feature TEXT NOT NULL,
              filename TEXT NOT NULL,
              row_count INTEGER NOT NULL,
              size_bytes INTEGER NOT NULL,
              sha256 TEXT NOT NULL,
              policy_fingerprint TEXT NOT NULL,
              PRIMARY KEY(participant_id, feature)
            );
            CREATE TABLE IF NOT EXISTS curation_outputs(
              participant_id TEXT NOT NULL,
              feature TEXT NOT NULL,
              filename TEXT NOT NULL,
              native_rows INTEGER NOT NULL,
              curated_rows INTEGER NOT NULL,
              size_bytes INTEGER NOT NULL,
              sha256 TEXT NOT NULL,
              native_sha256 TEXT NOT NULL,
              policy_fingerprint TEXT NOT NULL,
              pass_rows INTEGER NOT NULL,
              review_rows INTEGER NOT NULL,
              exclude_default_rows INTEGER NOT NULL,
              canonical_value_rows INTEGER NOT NULL,
              ambiguous_unit_rows INTEGER NOT NULL,
              scale_transition_rows INTEGER NOT NULL,
              flag_counts_json TEXT NOT NULL,
              acquisition_counts_json TEXT NOT NULL,
              PRIMARY KEY(participant_id, feature)
            );
            CREATE TABLE IF NOT EXISTS curation_unit_epochs(
              participant_id TEXT NOT NULL,
              feature TEXT NOT NULL,
              epoch_id TEXT NOT NULL,
              context_id TEXT NOT NULL,
              start_index INTEGER NOT NULL,
              end_index INTEGER NOT NULL,
              first_time TEXT,
              last_time TEXT,
              row_count INTEGER NOT NULL,
              raw_unit TEXT,
              canonical_unit TEXT,
              status TEXT NOT NULL,
              evidence TEXT NOT NULL,
              scale_transition INTEGER NOT NULL,
              median_value REAL,
              PRIMARY KEY(participant_id, feature, epoch_id)
            );
            CREATE TABLE IF NOT EXISTS curation_run_participants(
              run_id TEXT NOT NULL,
              participant_id TEXT NOT NULL,
              planned_action TEXT NOT NULL,
              plan_reason TEXT NOT NULL,
              final_status TEXT NOT NULL,
              native_files INTEGER NOT NULL,
              native_rows INTEGER NOT NULL,
              curated_rows INTEGER NOT NULL,
              output_size_bytes INTEGER NOT NULL,
              runtime_seconds REAL,
              peak_rss_bytes INTEGER,
              error_message TEXT,
              traceback_text TEXT,
              PRIMARY KEY(run_id, participant_id)
            );
            CREATE TABLE IF NOT EXISTS curation_run_features(
              run_id TEXT NOT NULL,
              participant_id TEXT NOT NULL,
              feature TEXT NOT NULL,
              native_rows INTEGER NOT NULL,
              curated_rows INTEGER NOT NULL,
              output_size_bytes INTEGER NOT NULL,
              pass_rows INTEGER NOT NULL,
              review_rows INTEGER NOT NULL,
              exclude_default_rows INTEGER NOT NULL,
              canonical_value_rows INTEGER NOT NULL,
              ambiguous_unit_rows INTEGER NOT NULL,
              scale_transition_rows INTEGER NOT NULL,
              flag_counts_json TEXT NOT NULL,
              acquisition_counts_json TEXT NOT NULL,
              policy_fingerprint TEXT NOT NULL,
              PRIMARY KEY(run_id, participant_id, feature)
            );
            """
        )
        self.connection.commit()

    def __enter__(self) -> "CurationStateDatabase":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.connection.close()

    def begin_run(self, input_root: Path, output_root: Path, configuration: Mapping[str, Any]) -> str:
        run_id = uuid.uuid4().hex
        self.connection.execute(
            "INSERT INTO curation_runs VALUES(?,?,?,?,?,?,?,NULL)",
            (
                run_id, now_utc(), None, str(input_root), str(output_root),
                "in_progress", _json(dict(configuration)),
            ),
        )
        self.connection.commit()
        return run_id

    def mark_run_finished(self, run_id: str, status: str) -> None:
        """
        Persist completion before assembling the final report. Reporting reads the run row, so completion
        must be recorded first for ``finished_at`` and wall-clock duration to be present in the report.
        """

        self.connection.execute(
            "UPDATE curation_runs SET finished_at=?,status=? WHERE run_id=?",
            (now_utc(), status, run_id),
        )
        self.connection.commit()

    def finish_run(self, run_id: str, status: str, summary: Mapping[str, Any]) -> None:
        self.connection.execute(
            "UPDATE curation_runs SET "
            "finished_at=COALESCE(finished_at,?),status=?,summary_json=? WHERE run_id=?",
            (now_utc(), status, _json(summary), run_id),
        )
        self.connection.commit()

    def get_participant(self, participant_id: str) -> StoredCurationParticipant | None:
        row = self.connection.execute(
            "SELECT * FROM curation_participants WHERE participant_id=?", (participant_id,),
        ).fetchone()
        if row is None:
            return None
        native_files = {
            item["feature"]: NativeFileState(
                item["feature"], item["filename"], item["row_count"], item["size_bytes"],
                item["sha256"], item["policy_fingerprint"],
            )
            for item in self.connection.execute(
                "SELECT * FROM curation_native_files WHERE participant_id=?", (participant_id,),
            )
        }
        outputs = {
            item["feature"]: CuratedOutputState(
                feature=item["feature"], filename=item["filename"],
                native_rows=item["native_rows"], curated_rows=item["curated_rows"],
                size_bytes=item["size_bytes"], sha256=item["sha256"],
                native_sha256=item["native_sha256"], policy_fingerprint=item["policy_fingerprint"],
                pass_rows=item["pass_rows"], review_rows=item["review_rows"],
                exclude_default_rows=item["exclude_default_rows"],
                canonical_value_rows=item["canonical_value_rows"],
                ambiguous_unit_rows=item["ambiguous_unit_rows"],
                scale_transition_rows=item["scale_transition_rows"],
                flag_counts=json.loads(item["flag_counts_json"]),
                acquisition_counts=json.loads(item["acquisition_counts_json"]),
            )
            for item in self.connection.execute(
                "SELECT * FROM curation_outputs WHERE participant_id=?", (participant_id,),
            )
        }
        return StoredCurationParticipant(
            participant_id=row["participant_id"], status=row["status"],
            native_signature=row["native_signature"], policy_signature=row["policy_signature"],
            engine_version=row["engine_version"], last_error=row["last_error"],
            native_files=native_files, outputs=outputs,
        )

    def mark_in_progress(self, participant_id: str) -> None:
        self.connection.execute(
            """INSERT INTO curation_participants(participant_id,status,updated_at)
               VALUES(?,?,?) ON CONFLICT(participant_id) DO UPDATE SET
               status='in_progress',last_error=NULL,updated_at=excluded.updated_at""",
            (participant_id, "in_progress", now_utc()),
        )
        self.connection.commit()

    def mark_failed(self, participant_id: str, message: str) -> None:
        self.connection.execute(
            """INSERT INTO curation_participants(participant_id,status,last_error,updated_at)
               VALUES(?,?,?,?) ON CONFLICT(participant_id) DO UPDATE SET
               status='failed',last_error=excluded.last_error,updated_at=excluded.updated_at""",
            (participant_id, "failed", message, now_utc()),
        )
        self.connection.commit()

    def record_run_participant(
        self, run_id: str, participant_id: str, *, planned_action: str, plan_reason: str,
        final_status: str, native_files: int = 0, native_rows: int = 0,
        curated_rows: int = 0, output_size_bytes: int = 0,
        runtime_seconds: float | None = None, peak_rss_bytes: int | None = None,
        error_message: str | None = None, traceback_text: str | None = None,
    ) -> None:
        self.connection.execute(
            """INSERT OR REPLACE INTO curation_run_participants VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, participant_id, planned_action, plan_reason, final_status,
                native_files, native_rows, curated_rows, output_size_bytes,
                runtime_seconds, peak_rss_bytes, error_message, traceback_text,
            ),
        )
        self.connection.commit()

    def commit_participant(
        self, participant_id: str, native_signature: str, policy_signature: str,
        native_files: Iterable[NativeFileState], outputs: Iterable[CuratedOutputState],
        unit_epochs: Iterable[Mapping[str, Any]], *, run_id: str,
    ) -> None:
        native_files = list(native_files)
        outputs = list(outputs)
        epochs = list(unit_epochs)
        with self.connection:
            self.connection.execute(
                """INSERT INTO curation_participants(
                     participant_id,status,native_signature,policy_signature,engine_version,last_error,updated_at
                   ) VALUES(?,?,?,?,?,NULL,?)
                   ON CONFLICT(participant_id) DO UPDATE SET
                     status='complete',native_signature=excluded.native_signature,
                     policy_signature=excluded.policy_signature,engine_version=excluded.engine_version,
                     last_error=NULL,updated_at=excluded.updated_at""",
                (
                    participant_id, "complete", native_signature, policy_signature,
                    CURATION_ENGINE_VERSION, now_utc(),
                ),
            )
            for table in ("curation_native_files", "curation_outputs", "curation_unit_epochs"):
                self.connection.execute(f"DELETE FROM {table} WHERE participant_id=?", (participant_id,))
            self.connection.executemany(
                "INSERT INTO curation_native_files VALUES(?,?,?,?,?,?,?)",
                [
                    (
                        participant_id, item.feature, item.filename, item.row_count,
                        item.size_bytes, item.sha256, item.policy_fingerprint,
                    )
                    for item in native_files
                ],
            )
            self.connection.executemany(
                """INSERT INTO curation_outputs VALUES(
                     ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                   )""",
                [
                    (
                        participant_id, item.feature, item.filename, item.native_rows,
                        item.curated_rows, item.size_bytes, item.sha256, item.native_sha256,
                        item.policy_fingerprint, item.pass_rows, item.review_rows,
                        item.exclude_default_rows, item.canonical_value_rows,
                        item.ambiguous_unit_rows, item.scale_transition_rows,
                        _json(item.flag_counts), _json(item.acquisition_counts),
                    )
                    for item in outputs
                ],
            )
            self.connection.executemany(
                """INSERT INTO curation_unit_epochs VALUES(
                     ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                   )""",
                [
                    (
                        participant_id, epoch["feature"], epoch["epoch_id"], epoch["context_id"],
                        epoch["start_index"], epoch["end_index"], epoch.get("first_time"),
                        epoch.get("last_time"), epoch["row_count"], epoch.get("raw_unit"),
                        epoch.get("canonical_unit"), epoch["status"], epoch["evidence"],
                        int(bool(epoch.get("scale_transition"))), epoch.get("median_value"),
                    )
                    for epoch in epochs
                ],
            )
            self.connection.executemany(
                """INSERT OR REPLACE INTO curation_run_features VALUES(
                     ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                   )""",
                [
                    (
                        run_id, participant_id, item.feature, item.native_rows,
                        item.curated_rows, item.size_bytes, item.pass_rows,
                        item.review_rows, item.exclude_default_rows,
                        item.canonical_value_rows, item.ambiguous_unit_rows,
                        item.scale_transition_rows, _json(item.flag_counts),
                        _json(item.acquisition_counts), item.policy_fingerprint,
                    )
                    for item in outputs
                ],
            )

    def current_snapshot(self) -> dict[str, Any]:
        participants = self.connection.execute(
            "SELECT COUNT(*) FROM curation_participants WHERE status='complete'"
        ).fetchone()[0]
        row = self.connection.execute(
            """SELECT COUNT(*) feature_files, COALESCE(SUM(curated_rows),0) curated_rows,
                      COALESCE(SUM(native_rows),0) native_rows,
                      COALESCE(SUM(size_bytes),0) output_size_bytes,
                      COALESCE(SUM(pass_rows),0) pass_rows,
                      COALESCE(SUM(review_rows),0) review_rows,
                      COALESCE(SUM(exclude_default_rows),0) exclude_default_rows,
                      COALESCE(SUM(canonical_value_rows),0) canonical_value_rows,
                      COALESCE(SUM(ambiguous_unit_rows),0) ambiguous_unit_rows,
                      COALESCE(SUM(scale_transition_rows),0) scale_transition_rows
               FROM curation_outputs"""
        ).fetchone()
        largest = self.connection.execute(
            """SELECT participant_id, SUM(size_bytes) output_size_bytes,
                      SUM(curated_rows) curated_rows
               FROM curation_outputs GROUP BY participant_id
               ORDER BY output_size_bytes DESC LIMIT 1"""
        ).fetchone()
        return {
            "committed_participants": participants,
            "feature_files": row["feature_files"],
            "native_rows": row["native_rows"],
            "curated_rows": row["curated_rows"],
            "output_size_bytes": row["output_size_bytes"],
            "pass_rows": row["pass_rows"],
            "review_rows": row["review_rows"],
            "exclude_default_rows": row["exclude_default_rows"],
            "canonical_value_rows": row["canonical_value_rows"],
            "ambiguous_unit_rows": row["ambiguous_unit_rows"],
            "scale_transition_rows": row["scale_transition_rows"],
            "largest_participant": dict(largest) if largest else None,
        }

    def build_report(self, run_id: str, *, include_details: bool = False) -> dict[str, Any]:
        run = self.connection.execute(
            "SELECT * FROM curation_runs WHERE run_id=?", (run_id,),
        ).fetchone()
        if run is None:
            raise KeyError(f"Unknown curation run ID: {run_id}")
        participants = list(self.connection.execute(
            "SELECT * FROM curation_run_participants WHERE run_id=? ORDER BY participant_id", (run_id,),
        ))
        features = list(self.connection.execute(
            "SELECT * FROM curation_run_features WHERE run_id=? ORDER BY participant_id,feature", (run_id,),
        ))
        outcomes = Counter(row["final_status"] for row in participants)
        flag_counts: Counter[str] = Counter()
        acquisition_counts: Counter[str] = Counter()
        for row in features:
            flag_counts.update(json.loads(row["flag_counts_json"]))
            acquisition_counts.update(json.loads(row["acquisition_counts_json"]))
        runtime_values = [row["runtime_seconds"] for row in participants if row["runtime_seconds"] is not None]
        peak_values = [row["peak_rss_bytes"] for row in participants if row["peak_rss_bytes"] is not None]
        started = datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))
        finished = datetime.fromisoformat(run["finished_at"].replace("Z", "+00:00")) if run["finished_at"] else None
        payload: dict[str, Any] = {
            "run_id": run_id,
            "status": run["status"],
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "configuration": json.loads(run["configuration_json"]),
            "participants": {
                "discovered": len(participants),
                "committed": outcomes["committed"],
                "skipped": outcomes["skipped"],
                "failed": outcomes["failed"],
            },
            "run_activity": {
                "native_files_read": sum(row["native_files"] for row in participants if row["final_status"] == "committed"),
                "native_rows_read": sum(row["native_rows"] for row in participants if row["final_status"] == "committed"),
                "curated_rows_written": sum(row["curated_rows"] for row in participants if row["final_status"] == "committed"),
                "output_size_bytes": sum(row["output_size_bytes"] for row in participants if row["final_status"] == "committed"),
                "pass_rows": sum(row["pass_rows"] for row in features),
                "review_rows": sum(row["review_rows"] for row in features),
                "exclude_default_rows": sum(row["exclude_default_rows"] for row in features),
                "canonical_value_rows": sum(row["canonical_value_rows"] for row in features),
                "ambiguous_unit_rows": sum(row["ambiguous_unit_rows"] for row in features),
                "scale_transition_rows": sum(row["scale_transition_rows"] for row in features),
                "flag_counts": dict(sorted(flag_counts.items())),
                "acquisition_counts": dict(sorted(acquisition_counts.items())),
            },
            "snapshot": self.current_snapshot(),
            "performance": {
                "wall_clock_seconds": (finished - started).total_seconds() if finished else None,
                "participant_runtime_max_seconds": max(runtime_values) if runtime_values else None,
                "peak_worker_rss_bytes": max(peak_values) if peak_values else None,
            },
            "failures": {
                row["participant_id"]: row["error_message"]
                for row in participants if row["final_status"] == "failed"
            },
            "fingerprints": {
                "registry": registry_fingerprint(),
                "decisions": decisions_fingerprint(),
                "engine": CURATION_ENGINE_VERSION,
            },
        }
        if include_details:
            payload["participant_details"] = [dict(row) for row in participants]
            payload["feature_details"] = [
                dict(row) | {
                    "flag_counts": json.loads(row["flag_counts_json"]),
                    "acquisition_counts": json.loads(row["acquisition_counts_json"]),
                }
                for row in features
            ]
        return payload

    def latest_run_id(self) -> str:
        row = self.connection.execute(
            "SELECT run_id FROM curation_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise KeyError("No curation run is recorded")
        return str(row[0])


def format_curation_report(report: Mapping[str, Any]) -> str:
    participants = report["participants"]
    activity = report["run_activity"]
    snapshot = report["snapshot"]
    performance = report["performance"]
    return "\n".join([
        f"Curation run: {report['status']}",
        f"Run ID: {report['run_id']}",
        f"Participants: {participants['committed']} committed; {participants['skipped']} skipped; {participants['failed']} failed",
        f"Native files read: {activity['native_files_read']}",
        f"Native rows read: {activity['native_rows_read']}",
        f"Curated rows written: {activity['curated_rows_written']}",
        f"Status rows: {activity['pass_rows']} pass; {activity['review_rows']} review; {activity['exclude_default_rows']} exclude-default",
        f"Canonical-value rows: {activity['canonical_value_rows']}",
        f"Ambiguous-unit rows: {activity['ambiguous_unit_rows']}",
        f"Scale-transition rows: {activity['scale_transition_rows']}",
        f"Snapshot: {snapshot['committed_participants']} participants; {snapshot['feature_files']} feature files; {snapshot['curated_rows']} rows",
        f"Output size: {snapshot['output_size_bytes']} bytes",
        f"Wall time: {performance['wall_clock_seconds']} seconds",
    ]) + "\n"


def load_curation_report(
    state_file: Path, run_id: str | None = None, *, include_details: bool = False,
) -> dict[str, Any]:
    if not state_file.is_file():
        raise FileNotFoundError(state_file)
    with CurationStateDatabase(state_file) as state:
        return state.build_report(run_id or state.latest_run_id(), include_details=include_details)
