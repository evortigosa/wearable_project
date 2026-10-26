"""
Wearable Data Processing and Modeling project
Transactional Google Fitness curation state and persisted reporting.
"""


from __future__ import annotations
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping
from uuid import uuid4
from wearable_project.curation.google_fitness.contract import CURATED_SCHEMA_VERSION, contract_fingerprint
from wearable_project.curation.google_fitness.decisions import decisions_fingerprint
from wearable_project.curation.google_fitness.engine import ENGINE_VERSION
from wearable_project.curation.google_fitness.evidence import evidence_fingerprint
from wearable_project.curation.google_fitness.guidance import guidance_fingerprint
from wearable_project.curation.google_fitness.registry import registry_fingerprint
from wearable_project.curation.google_fitness.rules import rules_fingerprint
from wearable_project.exceptions import OutputValidationError
from wearable_project.processing.google_fitness.registry import SOURCE_KEY


CURATION_STATE_FILENAME = ".wearable_curation_state.sqlite"
CURATION_STATE_SCHEMA_VERSION = "google-fitness-curation-state-schema-v2-0.2.0"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads(value: str | None) -> Any:
    return json.loads(value) if value else {}


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
    active_materialized_columns: tuple[str, ...]
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
            "failures": dict(self.failures),
        }


class CurationStateDatabase:
    """Google Fitness curated-root state. The parent process is the sole writer."""

    def __init__(self, path: Path, *, read_only: bool = False):
        self.path = Path(path)
        self.read_only = read_only
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        if read_only:
            self._guard_source()
        else:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self._initialize()

    def __enter__(self) -> "CurationStateDatabase":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.connection.close()

    def _metadata(self) -> dict[str, str]:
        try:
            return {
                str(row["key"]): str(row["value"])
                for row in self.connection.execute("SELECT key,value FROM metadata")
            }
        except sqlite3.Error:
            return {}

    def _guard_source(self) -> None:
        metadata = self._metadata()
        if metadata.get("source_key") != SOURCE_KEY:
            raise OutputValidationError(
                f"curation state source is {metadata.get('source_key')!r}, expected {SOURCE_KEY!r}"
            )
        if metadata.get("product_kind") != "curated":
            raise OutputValidationError("state database is not a Google Fitness curated-root state")
        if metadata.get("state_schema_version") != CURATION_STATE_SCHEMA_VERSION:
            raise OutputValidationError(
                "incompatible Google Fitness curation state schema: "
                f"{metadata.get('state_schema_version')!r}"
            )
        if metadata.get("curated_schema_version") != CURATED_SCHEMA_VERSION:
            raise OutputValidationError(
                "incompatible Google Fitness curated schema: "
                f"{metadata.get('curated_schema_version')!r}; use a new output root for schema v2"
            )
        if metadata.get("contract_fingerprint") != contract_fingerprint():
            raise OutputValidationError("curation state contract fingerprint differs from the current schema contract")

    def _initialize(self) -> None:
        metadata_exists = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
        ).fetchone()
        if metadata_exists is not None:
            metadata = self._metadata()
            if metadata.get("source_key") not in (None, SOURCE_KEY):
                raise OutputValidationError(
                    f"curation state is managed by source {metadata.get('source_key')!r}, not {SOURCE_KEY!r}"
                )
            if metadata.get("product_kind") not in (None, "curated"):
                raise OutputValidationError("state database belongs to a different product kind")
            if metadata.get("state_schema_version") not in (None, CURATION_STATE_SCHEMA_VERSION):
                raise OutputValidationError(
                    "incompatible Google Fitness curation state schema: "
                    f"{metadata.get('state_schema_version')!r}"
                )
            if metadata.get("curated_schema_version") not in (None, CURATED_SCHEMA_VERSION):
                raise OutputValidationError(
                    "existing curated root uses a different schema version; preserve it and choose a new output root"
                )
            if metadata.get("contract_fingerprint") not in (None, contract_fingerprint()):
                raise OutputValidationError("existing curated root uses a different curation contract")

        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
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
              active_materialized_columns_json TEXT NOT NULL,
              pass_rows INTEGER NOT NULL,
              review_rows INTEGER NOT NULL,
              exclude_default_rows INTEGER NOT NULL,
              included_by_default_rows INTEGER NOT NULL,
              excluded_by_default_rows INTEGER NOT NULL,
              canonical_value_rows INTEGER NOT NULL,
              flag_counts_json TEXT NOT NULL,
              acquisition_counts_json TEXT NOT NULL,
              derivation_counts_json TEXT NOT NULL,
              tracking_counts_json TEXT NOT NULL,
              step_count_role_counts_json TEXT NOT NULL,
              unit_status_counts_json TEXT NOT NULL,
              unit_evidence_counts_json TEXT NOT NULL,
              PRIMARY KEY(participant_id, feature)
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
              included_by_default_rows INTEGER NOT NULL,
              excluded_by_default_rows INTEGER NOT NULL,
              canonical_value_rows INTEGER NOT NULL,
              active_materialized_columns_json TEXT NOT NULL,
              flag_counts_json TEXT NOT NULL,
              acquisition_counts_json TEXT NOT NULL,
              derivation_counts_json TEXT NOT NULL,
              tracking_counts_json TEXT NOT NULL,
              step_count_role_counts_json TEXT NOT NULL,
              unit_status_counts_json TEXT NOT NULL,
              unit_evidence_counts_json TEXT NOT NULL,
              policy_fingerprint TEXT NOT NULL,
              PRIMARY KEY(run_id, participant_id, feature)
            );
            """
        )
        self.connection.executemany(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            (
                ("source_key", SOURCE_KEY),
                ("product_kind", "curated"),
                ("state_schema_version", CURATION_STATE_SCHEMA_VERSION),
                ("curated_schema_version", CURATED_SCHEMA_VERSION),
                ("contract_fingerprint", contract_fingerprint()),
            ),
        )
        self.connection.commit()

    def begin_run(
        self, input_native_root: Path, output_root: Path, configuration: Mapping[str, Any],
    ) -> str:
        run_id = uuid4().hex
        self.connection.execute(
            "INSERT INTO curation_runs VALUES(?,?,?,?,?,?,?,NULL)",
            (
                run_id,
                now_utc(),
                None,
                str(input_native_root),
                str(output_root),
                "in_progress",
                _json(dict(configuration)),
            ),
        )
        self.connection.commit()
        return run_id

    def finish_run(self, run_id: str, status: str, summary: Mapping[str, Any]) -> None:
        self.connection.execute(
            "UPDATE curation_runs SET finished_at=?,status=?,summary_json=? WHERE run_id=?",
            (now_utc(), status, _json(dict(summary)), run_id),
        )
        self.connection.commit()

    def mark_interrupted_runs(self) -> tuple[str, ...]:
        """Close run records left ``in_progress`` by a terminated parent process."""

        run_ids = tuple(
            str(row[0])
            for row in self.connection.execute(
                "SELECT run_id FROM curation_runs WHERE status='in_progress' ORDER BY started_at,run_id"
            )
        )
        if not run_ids:
            return ()
        finished_at = now_utc()
        with self.connection:
            self.connection.executemany(
                "UPDATE curation_runs SET finished_at=?,status='interrupted' WHERE run_id=?",
                [(finished_at, run_id) for run_id in run_ids],
            )
        return run_ids

    def mark_in_progress(self, participant_id: str) -> None:
        self.connection.execute(
            """
            INSERT INTO curation_participants(participant_id,status,engine_version,updated_at)
            VALUES(?,?,?,?)
            ON CONFLICT(participant_id) DO UPDATE SET
              status='in_progress',engine_version=excluded.engine_version,
              last_error=NULL,updated_at=excluded.updated_at
            """,
            (participant_id, "in_progress", ENGINE_VERSION, now_utc()),
        )
        self.connection.commit()

    def mark_failed(self, participant_id: str, message: str) -> None:
        # A failed rebuild must not invalidate a previously committed, still-published participant product.
        # Preserve its complete state when output records exist; an initial build with no prior commit is marked failed.
        has_committed_output = self.connection.execute(
            "SELECT 1 FROM curation_outputs WHERE participant_id=? LIMIT 1",
            (participant_id,),
        ).fetchone() is not None
        status = "complete" if has_committed_output else "failed"
        self.connection.execute(
            """
            INSERT INTO curation_participants(participant_id,status,engine_version,last_error,updated_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(participant_id) DO UPDATE SET
              status=excluded.status,last_error=excluded.last_error,
              updated_at=excluded.updated_at
            """,
            (participant_id, status, ENGINE_VERSION, message, now_utc()),
        )
        self.connection.commit()

    @staticmethod
    def _output_from_row(row: sqlite3.Row) -> CuratedOutputState:
        return CuratedOutputState(
            feature=row["feature"],
            filename=row["filename"],
            native_rows=int(row["native_rows"]),
            curated_rows=int(row["curated_rows"]),
            size_bytes=int(row["size_bytes"]),
            sha256=row["sha256"],
            native_sha256=row["native_sha256"],
            policy_fingerprint=row["policy_fingerprint"],
            active_materialized_columns=tuple(_loads(row["active_materialized_columns_json"])),
            pass_rows=int(row["pass_rows"]),
            review_rows=int(row["review_rows"]),
            exclude_default_rows=int(row["exclude_default_rows"]),
            included_by_default_rows=int(row["included_by_default_rows"]),
            excluded_by_default_rows=int(row["excluded_by_default_rows"]),
            canonical_value_rows=int(row["canonical_value_rows"]),
            flag_counts=dict(_loads(row["flag_counts_json"])),
            acquisition_counts=dict(_loads(row["acquisition_counts_json"])),
            derivation_counts=dict(_loads(row["derivation_counts_json"])),
            tracking_counts=dict(_loads(row["tracking_counts_json"])),
            step_count_role_counts=dict(_loads(row["step_count_role_counts_json"])),
            unit_status_counts=dict(_loads(row["unit_status_counts_json"])),
            unit_evidence_counts=dict(_loads(row["unit_evidence_counts_json"])),
        )

    def get_participant(self, participant_id: str) -> StoredCurationParticipant | None:
        row = self.connection.execute(
            "SELECT * FROM curation_participants WHERE participant_id=?", (participant_id,)
        ).fetchone()
        if row is None:
            return None
        native_files = {
            item["feature"]: NativeFileState(
                feature=item["feature"], filename=item["filename"],
                row_count=int(item["row_count"]), size_bytes=int(item["size_bytes"]),
                sha256=item["sha256"], policy_fingerprint=item["policy_fingerprint"],
            )
            for item in self.connection.execute(
                "SELECT * FROM curation_native_files WHERE participant_id=?", (participant_id,)
            )
        }
        outputs = {
            item["feature"]: self._output_from_row(item)
            for item in self.connection.execute(
                "SELECT * FROM curation_outputs WHERE participant_id=?", (participant_id,)
            )
        }
        return StoredCurationParticipant(
            participant_id=participant_id,
            status=row["status"],
            native_signature=row["native_signature"],
            policy_signature=row["policy_signature"],
            engine_version=row["engine_version"],
            last_error=row["last_error"],
            native_files=native_files,
            outputs=outputs,
        )

    def commit_participant(
        self, participant_id: str, native_signature: str, policy_signature: str, native_files: Iterable[NativeFileState],
        outputs: Iterable[CuratedOutputState],
    ) -> None:
        native_files = tuple(native_files)
        outputs = tuple(outputs)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO curation_participants(
                  participant_id,status,native_signature,policy_signature,
                  engine_version,last_error,updated_at
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(participant_id) DO UPDATE SET
                  status='complete',native_signature=excluded.native_signature,
                  policy_signature=excluded.policy_signature,
                  engine_version=excluded.engine_version,last_error=NULL,
                  updated_at=excluded.updated_at
                """,
                (
                    participant_id, "complete", native_signature, policy_signature,
                    ENGINE_VERSION, None, now_utc(),
                ),
            )
            self.connection.execute(
                "DELETE FROM curation_native_files WHERE participant_id=?", (participant_id,)
            )
            self.connection.execute(
                "DELETE FROM curation_outputs WHERE participant_id=?", (participant_id,)
            )
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
                """
                INSERT INTO curation_outputs VALUES(
                  ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                [
                    (
                        participant_id, item.feature, item.filename,
                        item.native_rows, item.curated_rows, item.size_bytes,
                        item.sha256, item.native_sha256, item.policy_fingerprint,
                        _json(item.active_materialized_columns),
                        item.pass_rows, item.review_rows, item.exclude_default_rows,
                        item.included_by_default_rows, item.excluded_by_default_rows,
                        item.canonical_value_rows, _json(item.flag_counts),
                        _json(item.acquisition_counts), _json(item.derivation_counts),
                        _json(item.tracking_counts), _json(item.step_count_role_counts),
                        _json(item.unit_status_counts), _json(item.unit_evidence_counts),
                    )
                    for item in outputs
                ],
            )

    def record_run_participant(
        self, run_id: str, participant_id: str, *, planned_action: str, plan_reason: str, final_status: str,
        native_files: int, native_rows: int, curated_rows: int = 0, output_size_bytes: int = 0,
        runtime_seconds: float | None = None, peak_rss_bytes: int | None = None,
        error_message: str | None = None, traceback_text: str | None = None, outputs: Iterable[CuratedOutputState] = (),
    ) -> None:
        outputs = tuple(outputs)
        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO curation_run_participants VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id, participant_id, planned_action, plan_reason, final_status,
                    native_files, native_rows, curated_rows, output_size_bytes,
                    runtime_seconds, peak_rss_bytes, error_message, traceback_text,
                ),
            )
            self.connection.execute(
                "DELETE FROM curation_run_features WHERE run_id=? AND participant_id=?",
                (run_id, participant_id),
            )
            self.connection.executemany(
                """
                INSERT INTO curation_run_features VALUES(
                  ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                [
                    (
                        run_id, participant_id, item.feature, item.native_rows,
                        item.curated_rows, item.size_bytes, item.pass_rows,
                        item.review_rows, item.exclude_default_rows,
                        item.included_by_default_rows, item.excluded_by_default_rows,
                        item.canonical_value_rows, _json(item.active_materialized_columns),
                        _json(item.flag_counts), _json(item.acquisition_counts),
                        _json(item.derivation_counts), _json(item.tracking_counts),
                        _json(item.step_count_role_counts), _json(item.unit_status_counts),
                        _json(item.unit_evidence_counts), item.policy_fingerprint,
                    )
                    for item in outputs
                ],
            )

    @staticmethod
    def _merge_counter(rows: Iterable[sqlite3.Row], column: str) -> Counter[str]:
        result: Counter[str] = Counter()
        for row in rows:
            result.update(_loads(row[column]))
        return result

    def current_snapshot(self) -> dict[str, Any]:
        rows = list(self.connection.execute("SELECT * FROM curation_outputs"))
        participants = self.connection.execute(
            "SELECT COUNT(DISTINCT participant_id) FROM curation_outputs"
        ).fetchone()[0]
        curated_rows = sum(int(row["curated_rows"]) for row in rows)
        included = sum(int(row["included_by_default_rows"]) for row in rows)
        excluded = sum(int(row["excluded_by_default_rows"]) for row in rows)
        status_total = sum(
            int(row["pass_rows"]) + int(row["review_rows"]) + int(row["exclude_default_rows"])
            for row in rows
        )
        column_counts: Counter[str] = Counter()
        for row in rows:
            column_counts.update(_loads(row["active_materialized_columns_json"]))
        return {
            "curated_schema_version": CURATED_SCHEMA_VERSION,
            "contract_fingerprint": contract_fingerprint(),
            "committed_participants": int(participants),
            "feature_files": len(rows),
            "native_rows": sum(int(row["native_rows"]) for row in rows),
            "curated_rows": curated_rows,
            "output_size_bytes": sum(int(row["size_bytes"]) for row in rows),
            "pass_rows": sum(int(row["pass_rows"]) for row in rows),
            "review_rows": sum(int(row["review_rows"]) for row in rows),
            "exclude_default_status_rows": sum(int(row["exclude_default_rows"]) for row in rows),
            "status_accounting_complete": status_total == curated_rows,
            "included_by_default_rows": included,
            "excluded_by_default_rows": excluded,
            "default_inclusion_accounting_complete": included + excluded == curated_rows,
            "canonical_value_rows": sum(int(row["canonical_value_rows"]) for row in rows),
            "flag_counts": dict(sorted(self._merge_counter(rows, "flag_counts_json").items())),
            "acquisition_counts": dict(sorted(self._merge_counter(rows, "acquisition_counts_json").items())),
            "derivation_counts": dict(sorted(self._merge_counter(rows, "derivation_counts_json").items())),
            "tracking_counts": dict(sorted(self._merge_counter(rows, "tracking_counts_json").items())),
            "step_count_role_counts": dict(sorted(self._merge_counter(rows, "step_count_role_counts_json").items())),
            "unit_status_counts": dict(sorted(self._merge_counter(rows, "unit_status_counts_json").items())),
            "unit_evidence_counts": dict(sorted(self._merge_counter(rows, "unit_evidence_counts_json").items())),
            "active_materialized_column_file_counts": dict(sorted(column_counts.items())),
        }

    def build_report(self, run_id: str, *, include_details: bool = False) -> dict[str, Any]:
        run = self.connection.execute(
            "SELECT * FROM curation_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if run is None:
            raise KeyError(f"unknown curation run ID: {run_id}")
        participant_rows = list(self.connection.execute(
            "SELECT * FROM curation_run_participants WHERE run_id=? ORDER BY participant_id",
            (run_id,),
        ))
        feature_rows = list(self.connection.execute(
            "SELECT * FROM curation_run_features WHERE run_id=? ORDER BY participant_id,feature",
            (run_id,),
        ))
        outcomes = Counter(row["final_status"] for row in participant_rows)
        curated_rows = sum(int(row["curated_rows"]) for row in feature_rows)
        included = sum(int(row["included_by_default_rows"]) for row in feature_rows)
        excluded = sum(int(row["excluded_by_default_rows"]) for row in feature_rows)
        status_total = sum(
            int(row["pass_rows"]) + int(row["review_rows"]) + int(row["exclude_default_rows"])
            for row in feature_rows
        )
        started = datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))
        finished = (
            datetime.fromisoformat(run["finished_at"].replace("Z", "+00:00"))
            if run["finished_at"] else None
        )
        runtime_values = [row["runtime_seconds"] for row in participant_rows if row["runtime_seconds"] is not None]
        peak_values = [row["peak_rss_bytes"] for row in participant_rows if row["peak_rss_bytes"] is not None]
        payload: dict[str, Any] = {
            "tool": "google_fitness_curation",
            "source_key": SOURCE_KEY,
            "run_id": run_id,
            "status": run["status"],
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "configuration": _loads(run["configuration_json"]),
            "participants": {
                "discovered": len(participant_rows),
                "committed": outcomes["committed"],
                "skipped": outcomes["skipped"],
                "failed": outcomes["failed"],
            },
            "run_activity": {
                "native_files_read": sum(int(row["native_files"]) for row in participant_rows if row["final_status"] == "committed"),
                "native_rows_read": sum(int(row["native_rows"]) for row in participant_rows if row["final_status"] == "committed"),
                "curated_rows_written": sum(int(row["curated_rows"]) for row in participant_rows if row["final_status"] == "committed"),
                "output_size_bytes": sum(int(row["output_size_bytes"]) for row in participant_rows if row["final_status"] == "committed"),
                "pass_rows": sum(int(row["pass_rows"]) for row in feature_rows),
                "review_rows": sum(int(row["review_rows"]) for row in feature_rows),
                "exclude_default_status_rows": sum(int(row["exclude_default_rows"]) for row in feature_rows),
                "status_accounting_complete": status_total == curated_rows,
                "included_by_default_rows": included,
                "excluded_by_default_rows": excluded,
                "default_inclusion_accounting_complete": included + excluded == curated_rows,
                "canonical_value_rows": sum(int(row["canonical_value_rows"]) for row in feature_rows),
                "flag_counts": dict(sorted(self._merge_counter(feature_rows, "flag_counts_json").items())),
                "acquisition_counts": dict(sorted(self._merge_counter(feature_rows, "acquisition_counts_json").items())),
                "derivation_counts": dict(sorted(self._merge_counter(feature_rows, "derivation_counts_json").items())),
                "tracking_counts": dict(sorted(self._merge_counter(feature_rows, "tracking_counts_json").items())),
                "step_count_role_counts": dict(sorted(self._merge_counter(feature_rows, "step_count_role_counts_json").items())),
                "unit_status_counts": dict(sorted(self._merge_counter(feature_rows, "unit_status_counts_json").items())),
                "unit_evidence_counts": dict(sorted(self._merge_counter(feature_rows, "unit_evidence_counts_json").items())),
            },
            "snapshot": self.current_snapshot(),
            "performance": {
                "wall_clock_seconds": (finished - started).total_seconds() if finished else None,
                "participant_runtime_max_seconds": max(runtime_values) if runtime_values else None,
                "worker_process_high_water_rss_bytes": max(peak_values) if peak_values else None,
            },
            "failures": {
                row["participant_id"]: row["error_message"]
                for row in participant_rows if row["final_status"] == "failed"
            },
            "fingerprints": {
                "registry": registry_fingerprint(),
                "evidence": evidence_fingerprint(),
                "rules": rules_fingerprint(),
                "decisions": decisions_fingerprint(),
                "guidance": guidance_fingerprint(),
                "engine": ENGINE_VERSION,
                "state_schema": CURATION_STATE_SCHEMA_VERSION,
                "curated_schema": CURATED_SCHEMA_VERSION,
                "contract": contract_fingerprint(),
            },
        }
        if include_details:
            payload["participant_details"] = [dict(row) for row in participant_rows]
            details: list[dict[str, Any]] = []
            json_columns = (
                "active_materialized_columns_json", "flag_counts_json", "acquisition_counts_json",
                "derivation_counts_json", "tracking_counts_json", "step_count_role_counts_json",
                "unit_status_counts_json", "unit_evidence_counts_json",
            )
            for row in feature_rows:
                item = dict(row)
                for column in json_columns:
                    item[column[:-5] if column.endswith("_json") else column] = _loads(item.pop(column))
                details.append(item)
            payload["feature_details"] = details
        return payload

    def latest_run_id(self) -> str:
        row = self.connection.execute(
            "SELECT run_id FROM curation_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise KeyError("no curation run is recorded")
        return str(row[0])


def format_curation_report(report: Mapping[str, Any]) -> str:
    participants = report["participants"]
    activity = report["run_activity"]
    snapshot = report["snapshot"]
    performance = report["performance"]
    return "\n".join(
        [
            f"Google Fitness curation run: {report['status']}",
            f"Run ID: {report['run_id']}",
            (
                "Participants: "
                f"{participants['committed']} committed; {participants['skipped']} skipped; "
                f"{participants['failed']} failed"
            ),
            f"Native files read: {activity['native_files_read']}",
            f"Native rows read: {activity['native_rows_read']}",
            f"Curated rows written: {activity['curated_rows_written']}",
            (
                "Status rows: "
                f"{activity['pass_rows']} pass; {activity['review_rows']} review; "
                f"{activity['exclude_default_status_rows']} exclude-default"
            ),
            (
                "Default inclusion: "
                f"{activity['included_by_default_rows']} included; "
                f"{activity['excluded_by_default_rows']} excluded"
            ),
            f"Canonical projection rows: {activity['canonical_value_rows']}",
            (
                "Snapshot: "
                f"{snapshot['committed_participants']} participants; "
                f"{snapshot['feature_files']} feature files; {snapshot['curated_rows']} rows"
            ),
            f"Output size: {snapshot['output_size_bytes']} bytes",
            f"Wall time: {performance['wall_clock_seconds']} seconds",
        ]
    ) + "\n"


def load_curation_report(
    state_file: Path, run_id: str | None = None, *, include_details: bool = False,
) -> dict[str, Any]:
    with CurationStateDatabase(state_file, read_only=True) as state:
        return state.build_report(run_id or state.latest_run_id(), include_details=include_details)
