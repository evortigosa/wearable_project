"""
Wearable Data Processing and Modeling project
Run-scoped processing telemetry and reports.This module is deliberately operational: it tracks what the native
parser did, how much work it performed, which exceptional schemas it encountered, and how the single-node worker
configuration behaved. It does not compute cohort statistics from physiological values and does not modify
participant feature CSVs.
"""

from __future__ import annotations
import json
import math
import os
import sqlite3
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from wearable_project.processing.registry import is_registered_feature


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def total_system_memory_bytes() -> int | None:
    """
    Return the effective memory ceiling using standard Linux interfaces. On a bare-metal machine this is physical
    RAM. Inside a container or systemd/cgroup slice it is the smaller of physical RAM and the active cgroup memory
    limit. The worker-count recommendation must not use host RAM when the process is operating under a tighter
    memory ceiling.
    """
    candidates: list[int] = []
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        value = pages * page_size
        if value > 0:
            candidates.append(value)
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    if not candidates:
        try:
            with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        candidates.append(int(line.split()[1]) * 1024)
                        break
        except (OSError, ValueError, IndexError):
            pass

    # cgroup v2 and legacy cgroup v1, respectively. "max" means unlimited.
    for limit_path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            raw = limit_path.read_text(encoding="utf-8").strip()
            if raw and raw.lower() != "max":
                limit = int(raw)
                # Ignore conventional effectively-unlimited sentinel values.
                if 0 < limit < (1 << 60):
                    candidates.append(limit)
        except (OSError, ValueError):
            continue
    return min(candidates) if candidates else None


def peak_rss_bytes() -> int | None:
    """
    Return the current process high-water RSS where supported. Linux and most BSD systems report ``ru_maxrss``
    in KiB; macOS reports bytes. A worker handles exactly one participant and then exits, so this process
    high-water mark is also the participant worker peak.
    """
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if value <= 0:
            return None
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, AttributeError, OSError, ValueError):
        return None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _loads_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return [str(value)]
    if isinstance(loaded, list):
        return [str(item) for item in loaded]
    return [str(loaded)]


def _dict_rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(row) for row in cursor.fetchall()]


class ProcessingTracker:
    """
    Persist run activity without changing participant feature outputs. The parent process is the only writer.
    Worker processes return compact counters through their existing result pipe; this class stores those
    counters in the same hidden SQLite state file already used for incremental correctness.
    """

    SCHEMA_WARNING_STAGES = {"unknown-feature", "schema-warning"}

    def __init__(self, connection: sqlite3.Connection, *, initialize: bool = True):
        self.connection = connection
        if initialize:
            self.initialize_schema()

    def initialize_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_tracking(
              run_id TEXT PRIMARY KEY,
              mode TEXT NOT NULL,
              snapshot_policy TEXT NOT NULL,
              row_error_policy TEXT NOT NULL,
              workers_requested INTEGER NOT NULL,
              workers_effective INTEGER NOT NULL,
              max_in_flight INTEGER NOT NULL,
              automatic_retries INTEGER NOT NULL DEFAULT 0,
              cpu_count INTEGER,
              total_memory_bytes INTEGER,
              started_at TEXT NOT NULL,
              finished_at TEXT,
              wall_clock_seconds REAL,
              status TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS run_participants(
              run_id TEXT NOT NULL,
              participant_id TEXT NOT NULL,
              planned_action TEXT NOT NULL,
              plan_reason TEXT,
              final_status TEXT NOT NULL,
              source_files_discovered INTEGER NOT NULL DEFAULT 0,
              source_files_planned INTEGER NOT NULL DEFAULT 0,
              source_files_read INTEGER NOT NULL DEFAULT 0,
              source_files_completed INTEGER NOT NULL DEFAULT 0,
              source_bytes_discovered INTEGER NOT NULL DEFAULT 0,
              source_bytes_read INTEGER NOT NULL DEFAULT 0,
              outer_rows_read INTEGER NOT NULL DEFAULT 0,
              apple_rows_read INTEGER NOT NULL DEFAULT 0,
              payloads_decoded INTEGER NOT NULL DEFAULT 0,
              payload_decode_failures INTEGER NOT NULL DEFAULT 0,
              raw_occurrences_parsed INTEGER NOT NULL DEFAULT 0,
              features_touched INTEGER NOT NULL DEFAULT 0,
              canonical_rows_written INTEGER NOT NULL DEFAULT 0,
              represented_occurrences_written INTEGER NOT NULL DEFAULT 0,
              duplicates_reconciled INTEGER NOT NULL DEFAULT 0,
              revisions_reconciled INTEGER NOT NULL DEFAULT 0,
              unresolved_conflicts INTEGER NOT NULL DEFAULT 0,
              invalid_timestamp_rows INTEGER NOT NULL DEFAULT 0,
              unknown_features_json TEXT,
              schema_warning_count INTEGER NOT NULL DEFAULT 0,
              runtime_seconds REAL,
              peak_rss_bytes INTEGER,
              output_size_bytes INTEGER NOT NULL DEFAULT 0,
              worker_exit_code INTEGER,
              attempt_number INTEGER NOT NULL DEFAULT 1,
              retry_count INTEGER NOT NULL DEFAULT 0,
              error_message TEXT,
              traceback_text TEXT,
              started_at TEXT,
              finished_at TEXT,
              PRIMARY KEY(run_id, participant_id),
              FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS run_features(
              run_id TEXT NOT NULL,
              participant_id TEXT NOT NULL,
              feature TEXT NOT NULL,
              registered_policy INTEGER NOT NULL,
              feature_family TEXT,
              schema_status TEXT NOT NULL,
              input_rows INTEGER NOT NULL DEFAULT 0,
              output_rows INTEGER NOT NULL DEFAULT 0,
              represented_occurrences INTEGER NOT NULL DEFAULT 0,
              duplicates_reconciled INTEGER NOT NULL DEFAULT 0,
              revisions_reconciled INTEGER NOT NULL DEFAULT 0,
              unresolved_conflicts INTEGER NOT NULL DEFAULT 0,
              invalid_timestamp_rows INTEGER NOT NULL DEFAULT 0,
              output_filename TEXT,
              output_size_bytes INTEGER,
              output_sha256 TEXT,
              PRIMARY KEY(run_id, participant_id, feature),
              FOREIGN KEY(run_id, participant_id)
                REFERENCES run_participants(run_id, participant_id)
            );

            CREATE TABLE IF NOT EXISTS run_diagnostics(
              diagnostic_id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              participant_id TEXT NOT NULL,
              source_file TEXT,
              row_number INTEGER,
              feature TEXT,
              stage TEXT NOT NULL,
              message TEXT NOT NULL,
              FOREIGN KEY(run_id, participant_id)
                REFERENCES run_participants(run_id, participant_id)
            );

            CREATE INDEX IF NOT EXISTS idx_run_participants_status
              ON run_participants(run_id, final_status);
            CREATE INDEX IF NOT EXISTS idx_run_features_feature
              ON run_features(run_id, feature);
            CREATE INDEX IF NOT EXISTS idx_run_diagnostics_stage
              ON run_diagnostics(run_id, stage);
            """
        )
        self.connection.commit()


    def begin_run(
        self, run_id: str, *, mode: str, snapshot_policy: str, row_error_policy: str, workers_requested: int,
        workers_effective: int, max_in_flight: int,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO run_tracking(
              run_id,mode,snapshot_policy,row_error_policy,
              workers_requested,workers_effective,max_in_flight,
              automatic_retries,cpu_count,total_memory_bytes,started_at,status
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id, mode, snapshot_policy, row_error_policy,
                workers_requested, workers_effective, max_in_flight,
                0, os.cpu_count(), total_system_memory_bytes(), now_utc(), "in_progress",
            ),
        )
        self.connection.commit()


    def record_plan(
        self, run_id: str, participant_id: str, *, action: str, reason: str, final_status: str,
        source_files_discovered: int, source_files_planned: int, source_bytes_discovered: int,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO run_participants(
              run_id,participant_id,planned_action,plan_reason,final_status,
              source_files_discovered,source_files_planned,source_bytes_discovered
            ) VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id,participant_id) DO UPDATE SET
              planned_action=excluded.planned_action,
              plan_reason=excluded.plan_reason,
              final_status=excluded.final_status,
              source_files_discovered=excluded.source_files_discovered,
              source_files_planned=excluded.source_files_planned,
              source_bytes_discovered=excluded.source_bytes_discovered
            """,
            (
                run_id, participant_id, action, reason, final_status,
                source_files_discovered, source_files_planned, source_bytes_discovered,
            ),
        )
        self.connection.commit()


    def mark_participant_started(self, run_id: str, participant_id: str) -> None:
        self.connection.execute(
            """
            UPDATE run_participants
            SET final_status='in_progress',started_at=?
            WHERE run_id=? AND participant_id=?
            """,
            (now_utc(), run_id, participant_id),
        )
        self.connection.commit()


    def record_participant_result(
        self, run_id: str, participant_id: str, *, final_status: str, result: Any, worker_exit_code: int | None,
        committed_output_size_bytes: int, error_message: str | None = None, traceback_text: str | None = None,
    ) -> None:
        diagnostics = list(getattr(result, "diagnostics", []) or [])
        unknown_features = sorted(set(getattr(result, "unknown_features", []) or []))
        schema_warning_count = sum(
            1 for item in diagnostics
            if str(item.get("stage", "")) in self.SCHEMA_WARNING_STAGES
        )
        feature_stats: Mapping[str, Mapping[str, Any]] = getattr(result, "feature_stats", {}) or {}
        duplicates = sum(int(value.get("exact_duplicates_removed", 0)) for value in feature_stats.values())
        revisions = sum(int(value.get("revisions_resolved", 0)) for value in feature_stats.values())
        unresolved = sum(int(value.get("unresolved_conflicts", 0)) for value in feature_stats.values())
        invalid = sum(int(value.get("invalid_timestamp_rows", 0)) for value in feature_stats.values())
        canonical_rows = sum(int(value.get("output_rows", 0)) for value in feature_stats.values())
        represented = sum(int(value.get("represented_occurrences", 0)) for value in feature_stats.values())

        self.connection.execute(
            """
            UPDATE run_participants SET
              final_status=?,source_files_read=?,source_files_completed=?,source_bytes_read=?,
              outer_rows_read=?,apple_rows_read=?,payloads_decoded=?,payload_decode_failures=?,
              raw_occurrences_parsed=?,features_touched=?,canonical_rows_written=?,
              represented_occurrences_written=?,duplicates_reconciled=?,revisions_reconciled=?,
              unresolved_conflicts=?,invalid_timestamp_rows=?,unknown_features_json=?,
              schema_warning_count=?,runtime_seconds=?,peak_rss_bytes=?,output_size_bytes=?,
              worker_exit_code=?,error_message=?,traceback_text=?,finished_at=?
            WHERE run_id=? AND participant_id=?
            """,
            (
                final_status,
                int(getattr(result, "source_files_read", 0) or 0),
                int(getattr(result, "source_files_completed", 0) or 0),
                int(getattr(result, "source_bytes_read", 0) or 0),
                int(getattr(result, "outer_rows", 0) or 0),
                int(getattr(result, "apple_rows", 0) or 0),
                int(getattr(result, "payloads_decoded", 0) or 0),
                int(getattr(result, "payload_decode_failures", 0) or 0),
                int(getattr(result, "payload_items", 0) or 0),
                len(feature_stats), canonical_rows, represented,
                duplicates, revisions, unresolved, invalid,
                json.dumps(unknown_features, ensure_ascii=False), schema_warning_count,
                getattr(result, "runtime_seconds", None), getattr(result, "peak_rss_bytes", None),
                int(committed_output_size_bytes), worker_exit_code,
                error_message if error_message is not None else getattr(result, "error", None),
                traceback_text if traceback_text is not None else getattr(result, "traceback_text", None),
                now_utc(), run_id, participant_id,
            ),
        )

        output_by_feature = {
            item.feature: item for item in (getattr(result, "output_infos", []) or [])
        }
        self.connection.execute(
            "DELETE FROM run_features WHERE run_id=? AND participant_id=?",
            (run_id, participant_id),
        )
        feature_rows = []
        for feature, values in sorted(feature_stats.items()):
            output = output_by_feature.get(feature)
            feature_rows.append((
                run_id, participant_id, feature,
                1 if values.get("registered_policy") else 0,
                values.get("feature_family"), values.get("schema_status", "known"),
                int(values.get("input_rows", 0)), int(values.get("output_rows", 0)),
                int(values.get("represented_occurrences", 0)),
                int(values.get("exact_duplicates_removed", 0)),
                int(values.get("revisions_resolved", 0)),
                int(values.get("unresolved_conflicts", 0)),
                int(values.get("invalid_timestamp_rows", 0)),
                output.filename if output else None,
                output.size_bytes if output else None,
                output.sha256 if output else None,
            ))
        if feature_rows:
            self.connection.executemany(
                """
                INSERT INTO run_features(
                  run_id,participant_id,feature,registered_policy,feature_family,schema_status,
                  input_rows,output_rows,represented_occurrences,duplicates_reconciled,
                  revisions_reconciled,unresolved_conflicts,invalid_timestamp_rows,
                  output_filename,output_size_bytes,output_sha256
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                feature_rows,
            )

        self.connection.execute(
            "DELETE FROM run_diagnostics WHERE run_id=? AND participant_id=?",
            (run_id, participant_id),
        )
        diagnostic_rows = [
            (
                run_id, participant_id, item.get("source_file"), item.get("row_number"),
                item.get("feature"), str(item.get("stage", "unknown")), str(item.get("message", "")),
            )
            for item in diagnostics
        ]
        if diagnostic_rows:
            self.connection.executemany(
                """
                INSERT INTO run_diagnostics(
                  run_id,participant_id,source_file,row_number,feature,stage,message
                ) VALUES(?,?,?,?,?,?,?)
                """,
                diagnostic_rows,
            )
        self.connection.commit()


    def finish_run(self, run_id: str, *, status: str, wall_clock_seconds: float) -> None:
        finished = now_utc()
        self.connection.execute(
            """
            UPDATE run_tracking
            SET finished_at=?,wall_clock_seconds=?,status=?
            WHERE run_id=?
            """,
            (finished, float(wall_clock_seconds), status, run_id),
        )
        self.connection.execute(
            """
            UPDATE run_participants
            SET final_status='failed',
                error_message=COALESCE(error_message,'Run ended before participant completed'),
                finished_at=COALESCE(finished_at,?)
            WHERE run_id=? AND final_status IN ('planned','in_progress')
            """,
            (finished, run_id),
        )
        self.connection.commit()


    def latest_run_id(self) -> str | None:
        row = self.connection.execute(
            "SELECT run_id FROM run_tracking ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return None if row is None else str(row[0])


    def build_report(
        self, run_id: str, *, include_details: bool = False
    ) -> dict[str, Any]:
        tracking = self.connection.execute(
            "SELECT * FROM run_tracking WHERE run_id=?", (run_id,)
        ).fetchone()
        if tracking is None:
            raise KeyError(f"Unknown processing run: {run_id}")
        tracking = dict(tracking)

        status_counts = {
            str(row["final_status"]): int(row["count"])
            for row in self.connection.execute(
                """
                SELECT final_status,COUNT(*) AS count
                FROM run_participants WHERE run_id=? GROUP BY final_status
                """,
                (run_id,),
            )
        }
        participants = {
            "discovered": int(self.connection.execute(
                "SELECT COUNT(*) FROM run_participants WHERE run_id=? AND planned_action!='missing_participant'",
                (run_id,),
            ).fetchone()[0]),
            "committed": status_counts.get("committed", 0),
            "empty": status_counts.get("complete_empty", 0),
            "skipped": status_counts.get("skipped", 0),
            "blocked": status_counts.get("blocked", 0),
            "failed": status_counts.get("failed", 0),
            "incremental": int(self.connection.execute(
                """
                SELECT COUNT(*) FROM run_participants
                WHERE run_id=? AND planned_action='incremental'
                  AND final_status IN ('committed','complete_empty')
                """,
                (run_id,),
            ).fetchone()[0]),
            "rebuilt": int(self.connection.execute(
                """
                SELECT COUNT(*) FROM run_participants
                WHERE run_id=? AND planned_action='rebuild'
                  AND final_status IN ('committed','complete_empty')
                """,
                (run_id,),
            ).fetchone()[0]),
        }

        aggregate = dict(self.connection.execute(
            """
            SELECT
              COALESCE(SUM(source_files_discovered),0) AS monthly_files_discovered,
              COALESCE(SUM(source_files_planned),0) AS monthly_files_planned,
              COALESCE(SUM(source_files_read),0) AS monthly_files_read,
              COALESCE(SUM(source_files_completed),0) AS monthly_files_completed,
              COALESCE(SUM(source_bytes_read),0) AS source_bytes_read,
              COALESCE(SUM(outer_rows_read),0) AS outer_rows_read,
              COALESCE(SUM(apple_rows_read),0) AS apple_rows_read,
              COALESCE(SUM(payloads_decoded),0) AS payloads_decoded,
              COALESCE(SUM(payload_decode_failures),0) AS payload_parse_failures,
              COALESCE(SUM(raw_occurrences_parsed),0) AS raw_occurrences_parsed,
              COALESCE(SUM(features_touched),0) AS features_touched,
              COALESCE(SUM(canonical_rows_written),0) AS canonical_rows_written,
              COALESCE(SUM(represented_occurrences_written),0) AS represented_occurrences_written,
              COALESCE(SUM(duplicates_reconciled),0) AS duplicates_reconciled,
              COALESCE(SUM(revisions_reconciled),0) AS revisions_reconciled,
              COALESCE(SUM(unresolved_conflicts),0) AS unresolved_conflicts,
              COALESCE(SUM(invalid_timestamp_rows),0) AS invalid_timestamp_rows,
              COALESCE(SUM(schema_warning_count),0) AS schema_warning_count
            FROM run_participants WHERE run_id=?
            """,
            (run_id,),
        ).fetchone())
        run_activity = {key: int(value or 0) for key, value in aggregate.items()}

        current_counts = dict(self.connection.execute(
            """
            SELECT
              (SELECT COUNT(DISTINCT participant_id) FROM feature_outputs) AS committed_participants,
              (SELECT COUNT(DISTINCT source_files.participant_id)
                 FROM source_files
                 WHERE NOT EXISTS (
                   SELECT 1 FROM feature_outputs
                   WHERE feature_outputs.participant_id=source_files.participant_id
                 )) AS empty_participants,
              (SELECT COUNT(*) FROM source_files) AS committed_monthly_files,
              (SELECT COUNT(*) FROM feature_outputs) AS feature_files,
              (SELECT COALESCE(SUM(row_count),0) FROM feature_outputs) AS canonical_rows,
              (SELECT COALESCE(SUM(COALESCE(represented_occurrences,row_count)),0)
                 FROM feature_outputs) AS represented_source_occurrences,
              (SELECT COUNT(*) FROM feature_outputs WHERE represented_occurrences IS NULL)
                 AS represented_occurrences_unknown_files,
              (SELECT COALESCE(SUM(size_bytes),0) FROM feature_outputs) AS output_size_bytes
            """
        ).fetchone())
        snapshot = {key: int(value or 0) for key, value in current_counts.items()}
        snapshot["represented_occurrences_complete"] = snapshot.pop("represented_occurrences_unknown_files") == 0

        largest_output = self.connection.execute(
            """
            SELECT participant_id,SUM(size_bytes) AS output_size_bytes,
                   SUM(row_count) AS canonical_rows,
                   SUM(COALESCE(represented_occurrences,row_count)) AS represented_source_occurrences
            FROM feature_outputs GROUP BY participant_id
            ORDER BY output_size_bytes DESC,participant_id LIMIT 1
            """
        ).fetchone()
        snapshot["largest_participant"] = dict(largest_output) if largest_output else None

        participant_rows = _dict_rows(self.connection.execute(
            """
            SELECT participant_id,final_status,planned_action,runtime_seconds,
                   peak_rss_bytes,worker_exit_code,retry_count,error_message,traceback_text
            FROM run_participants WHERE run_id=?
            """,
            (run_id,),
        ))
        runtimes = [float(row["runtime_seconds"]) for row in participant_rows if row["runtime_seconds"] is not None]
        memories = [float(row["peak_rss_bytes"]) for row in participant_rows if row["peak_rss_bytes"] is not None]
        longest = max(
            (row for row in participant_rows if row["runtime_seconds"] is not None),
            key=lambda row: float(row["runtime_seconds"]), default=None,
        )
        highest_memory = max(
            (row for row in participant_rows if row["peak_rss_bytes"] is not None),
            key=lambda row: int(row["peak_rss_bytes"]), default=None,
        )
        peak_memory = int(max(memories)) if memories else None
        total_memory = tracking.get("total_memory_bytes")
        cpu_count = tracking.get("cpu_count")
        safe_worker_count = None
        if peak_memory and total_memory:
            memory_ceiling = max(1, int((0.70 * int(total_memory)) // peak_memory))
            cpu_ceiling = max(1, int(cpu_count or 1) - 1)
            safe_worker_count = max(1, min(memory_ceiling, cpu_ceiling))
        performance = {
            "wall_clock_seconds": tracking.get("wall_clock_seconds"),
            "participant_runtime_median_seconds": statistics.median(runtimes) if runtimes else None,
            "participant_runtime_p95_seconds": _percentile(runtimes, 0.95),
            "largest_participant_runtime_seconds": (
                float(longest["runtime_seconds"]) if longest else None
            ),
            "largest_participant_runtime_id": longest["participant_id"] if longest else None,
            "peak_worker_rss_bytes": peak_memory,
            "median_worker_peak_rss_bytes": statistics.median(memories) if memories else None,
            "p95_worker_peak_rss_bytes": _percentile(memories, 0.95),
            "peak_worker_rss_participant_id": highest_memory["participant_id"] if highest_memory else None,
            "safe_worker_count_advisory": safe_worker_count,
            "safe_worker_count_basis": (
                "min(cpu_count-1, floor(70% of effective memory limit / maximum observed worker RSS))"
                if safe_worker_count is not None else None
            ),
        }

        unknown_features: set[str] = set()
        for row in self.connection.execute(
            "SELECT unknown_features_json FROM run_participants WHERE run_id=?",
            (run_id,),
        ):
            unknown_features.update(_loads_list(row[0]))
        unknown_features.update(
            str(row[0]) for row in self.connection.execute(
                "SELECT DISTINCT feature FROM run_features WHERE run_id=? AND registered_policy=0",
                (run_id,),
            )
        )
        snapshot_unknown_features = sorted(
            str(row[0]) for row in self.connection.execute(
                "SELECT DISTINCT feature FROM feature_outputs ORDER BY feature"
            ) if not is_registered_feature(str(row[0]))
        )
        unknown_features.update(snapshot_unknown_features)
        schema_warnings = _dict_rows(self.connection.execute(
            """
            SELECT feature,stage,message,COUNT(*) AS occurrences
            FROM run_diagnostics
            WHERE run_id=? AND stage IN ('unknown-feature','schema-warning')
            GROUP BY feature,stage,message
            ORDER BY feature,stage,message
            """,
            (run_id,),
        ))
        diagnostic_counts = {
            str(row["stage"]): int(row["count"])
            for row in self.connection.execute(
                """
                SELECT stage,COUNT(*) AS count FROM run_diagnostics
                WHERE run_id=? GROUP BY stage ORDER BY stage
                """,
                (run_id,),
            )
        }

        blocked = {
            str(row["participant_id"]): str(row["plan_reason"])
            for row in self.connection.execute(
                """
                SELECT participant_id,plan_reason FROM run_participants
                WHERE run_id=? AND final_status='blocked' ORDER BY participant_id
                """,
                (run_id,),
            )
        }
        failures = {
            str(row["participant_id"]): str(row["error_message"])
            for row in self.connection.execute(
                """
                SELECT participant_id,error_message FROM run_participants
                WHERE run_id=? AND final_status='failed' ORDER BY participant_id
                """,
                (run_id,),
            )
        }
        worker_exit_codes = {
            str(row["participant_id"]): row["worker_exit_code"]
            for row in participant_rows if row["worker_exit_code"] not in (None, 0)
        }
        failure_behavior = {
            "automatic_retries": int(tracking.get("automatic_retries") or 0),
            "retry_count": sum(int(row["retry_count"] or 0) for row in participant_rows),
            "nonzero_worker_exit_codes": worker_exit_codes,
            "failed_participants": failures,
            "blocked_participants": blocked,
            "tracebacks_persisted": sum(1 for row in participant_rows if row["traceback_text"]),
        }

        report: dict[str, Any] = {
            "run_id": run_id,
            "status": tracking["status"],
            "started_at": tracking["started_at"],
            "finished_at": tracking["finished_at"],
            "configuration": {
                "mode": tracking["mode"],
                "snapshot_policy": tracking["snapshot_policy"],
                "row_error_policy": tracking["row_error_policy"],
                "workers_requested": int(tracking["workers_requested"]),
                "workers_effective": int(tracking["workers_effective"]),
                "max_in_flight": int(tracking["max_in_flight"]),
            },
            "system": {
                "cpu_count": tracking.get("cpu_count"),
                "effective_memory_limit_bytes": tracking.get("total_memory_bytes"),
            },
            "participants": participants,
            "run_activity": run_activity,
            "snapshot": snapshot,
            "unknowns": {
                "feature_names": sorted(unknown_features),
                "snapshot_feature_names": snapshot_unknown_features,
                "schema_warning_count": sum(int(item["occurrences"]) for item in schema_warnings),
                "unique_schema_warning_count": len(schema_warnings),
                "schema_warnings": schema_warnings,
                "diagnostic_counts": diagnostic_counts,
            },
            "performance": performance,
            "failure_behavior": failure_behavior,
        }
        if include_details:
            participant_details = _dict_rows(self.connection.execute(
                """
                SELECT * FROM run_participants
                WHERE run_id=? ORDER BY participant_id
                """,
                (run_id,),
            ))
            for item in participant_details:
                item["unknown_features"] = _loads_list(item.pop("unknown_features_json", None))
            report["participant_details"] = participant_details
            report["feature_details"] = _dict_rows(self.connection.execute(
                """
                SELECT * FROM run_features
                WHERE run_id=? ORDER BY participant_id,feature
                """,
                (run_id,),
            ))
            report["diagnostics"] = _dict_rows(self.connection.execute(
                """
                SELECT diagnostic_id,run_id,participant_id,source_file,row_number,
                       feature,stage,message
                FROM run_diagnostics
                WHERE run_id=? ORDER BY participant_id,source_file,row_number,diagnostic_id
                """,
                (run_id,),
            ))
        return report


def load_processing_report(
    state_file: Path, run_id: str | None = None, *, include_details: bool = False
) -> dict[str, Any]:
    path = state_file.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Processing state database does not exist: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='run_tracking'"
        ).fetchone()
        if table is None:
            raise KeyError(
                "This state database predates processing telemetry. Run wearable-project "
                "process with v0.1.2 or later before requesting a tracked report."
            )
        tracker = ProcessingTracker(connection, initialize=False)
        selected = run_id or tracker.latest_run_id()
        if selected is None:
            raise KeyError(f"No tracked processing runs exist in {path}")
        return tracker.build_report(selected, include_details=include_details)
    finally:
        connection.close()


def _format_bytes(value: Any) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    index = 0
    while abs(number) >= 1024 and index < len(units) - 1:
        number /= 1024
        index += 1
    return f"{number:.2f} {units[index]}"


def _format_seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    seconds = float(value)
    if seconds < 60:
        return f"{seconds:.2f} s"
    if seconds < 3600:
        return f"{seconds / 60:.2f} min"
    return f"{seconds / 3600:.2f} h"


def format_processing_report(report: Mapping[str, Any]) -> str:
    participants = report["participants"]
    activity = report["run_activity"]
    snapshot = report["snapshot"]
    performance = report["performance"]
    unknowns = report["unknowns"]
    failure = report["failure_behavior"]
    configuration = report["configuration"]
    largest = snapshot.get("largest_participant") or {}

    lines = [
        f"Processing run {report['run_id']} [{report['status']}]",
        (
            "Configuration: "
            f"mode={configuration['mode']} snapshot={configuration['snapshot_policy']} "
            f"workers={configuration['workers_effective']} max_in_flight={configuration['max_in_flight']}"
        ),
        "Participants: " + " ".join(
            f"{name}={participants[name]}" for name in
            ("discovered", "committed", "empty", "incremental", "rebuilt", "skipped", "blocked", "failed")
        ),
        "Run activity:",
        (
            f"  monthly_files_read={activity['monthly_files_read']} "
            f"completed={activity['monthly_files_completed']} "
            f"outer_rows={activity['outer_rows_read']} apple_rows={activity['apple_rows_read']}"
        ),
        (
            f"  payloads_decoded={activity['payloads_decoded']} "
            f"payload_failures={activity['payload_parse_failures']} "
            f"raw_occurrences={activity['raw_occurrences_parsed']}"
        ),
        (
            f"  features_touched={activity['features_touched']} "
            f"canonical_rows_written={activity['canonical_rows_written']} "
            f"duplicates={activity['duplicates_reconciled']} "
            f"revisions={activity['revisions_reconciled']} "
            f"unresolved_conflicts={activity['unresolved_conflicts']}"
        ),
        "Current snapshot:",
        (
            f"  committed_participants={snapshot['committed_participants']} "
            f"empty_participants={snapshot['empty_participants']} "
            f"monthly_files={snapshot['committed_monthly_files']} "
            f"feature_files={snapshot['feature_files']}"
        ),
        (
            f"  canonical_rows={snapshot['canonical_rows']} "
            f"represented_occurrences={snapshot['represented_source_occurrences']} "
            f"output_size={_format_bytes(snapshot['output_size_bytes'])}"
        ),
        (
            f"  largest_participant={largest.get('participant_id', 'n/a')} "
            f"size={_format_bytes(largest.get('output_size_bytes'))}"
        ),
        "Performance:",
        (
            f"  wall_clock={_format_seconds(performance.get('wall_clock_seconds'))} "
            f"largest_participant_runtime={_format_seconds(performance.get('largest_participant_runtime_seconds'))} "
            f"participant={performance.get('largest_participant_runtime_id') or 'n/a'}"
        ),
        (
            f"  peak_worker_rss={_format_bytes(performance.get('peak_worker_rss_bytes'))} "
            f"participant={performance.get('peak_worker_rss_participant_id') or 'n/a'} "
            f"safe_worker_advisory={performance.get('safe_worker_count_advisory') or 'n/a'}"
        ),
        (
            f"Unknowns: features={len(unknowns['feature_names'])} "
            f"schema_warnings={unknowns['schema_warning_count']}"
        ),
        (
            f"Failures: failed={participants['failed']} blocked={participants['blocked']} "
            f"automatic_retries={failure['automatic_retries']} retry_count={failure['retry_count']}"
        ),
    ]
    if unknowns["feature_names"]:
        lines.append("  Unknown feature names: " + ", ".join(unknowns["feature_names"]))
    return "\n".join(lines)
