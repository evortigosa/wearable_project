"""
Wearable Data Processing and Modeling project
Google Fitness WP4 run reporting backed by the source-qualified native state.
"""


from __future__ import annotations
import json
import resource
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping
from wearable_project.processing.google_fitness.registry import SOURCE_KEY
from wearable_project.processing.google_fitness.writer import STATE_FILENAME


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return value if value > 10**10 else value * 1024


def _json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


def _source_guard(connection: sqlite3.Connection) -> None:
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='source_key'"
        ).fetchone()
    except sqlite3.Error as exc:
        raise ValueError("not a Google Fitness WP4 state database") from exc
    if row is None or row["value"] != SOURCE_KEY:
        raise ValueError(
            f"state source is {None if row is None else row['value']!r}, expected {SOURCE_KEY!r}"
        )


def build_processing_report(
    connection: sqlite3.Connection, run_id: str, *, include_details: bool = False,
) -> dict[str, Any]:
    _source_guard(connection)
    run = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if run is None:
        raise KeyError(f"unknown processing run: {run_id}")
    participant_rows = list(
        connection.execute(
            "SELECT * FROM run_participants WHERE run_id=? ORDER BY participant_id", (run_id,),
        )
    )
    status_counts = Counter(row["final_status"] for row in participant_rows)
    action_counts = Counter(row["planned_action"] for row in participant_rows)
    totals: Counter[str] = Counter()
    failed: dict[str, str] = {}
    blocked: dict[str, str] = {}
    participant_details: list[dict[str, Any]] = []
    run_diagnostics_by_code: Counter[str] = Counter()
    for row in participant_rows:
        payload = _json(row["result_json"])
        for key, value in payload.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                if key == "peak_rss_bytes":
                    totals[key] = max(totals.get(key, 0), value)
                else:
                    totals[key] += value
        breakdown = payload.get("diagnostics_by_code")
        if isinstance(breakdown, dict):
            for code, count in breakdown.items():
                if (
                    isinstance(code, str)
                    and isinstance(count, int)
                    and not isinstance(count, bool)
                ):
                    run_diagnostics_by_code[code] += count
        if row["final_status"] == "failed":
            failed[row["participant_id"]] = row["error_message"] or "failed"
        if row["final_status"] == "blocked":
            blocked[row["participant_id"]] = row["plan_reason"]
        if include_details:
            participant_details.append(
                {
                    "participant_id": row["participant_id"],
                    "planned_action": row["planned_action"],
                    "plan_reason": row["plan_reason"],
                    "final_status": row["final_status"],
                    "error_message": row["error_message"],
                    "result": payload,
                }
            )

    feature_rows = list(
        connection.execute(
            "SELECT * FROM run_features WHERE run_id=? ORDER BY feature,participant_id", (run_id,),
        )
    )
    feature_totals: dict[str, Counter[str]] = defaultdict(Counter)
    feature_details: list[dict[str, Any]] = []
    for row in feature_rows:
        payload = _json(row["result_json"])
        for key, value in payload.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                feature_totals[row["feature"]][key] += value
        if include_details:
            feature_details.append(
                {
                    "participant_id": row["participant_id"],
                    "feature": row["feature"],
                    **payload,
                }
            )

    current_diagnostics_by_code = {
        row["code"]: int(row["count"])
        for row in connection.execute(
            """
            SELECT code, COUNT(*) AS count
            FROM parser_diagnostics
            GROUP BY code
            ORDER BY code
            """
        )
    }
    current_malformed_breakdown = []
    for row in connection.execute(
        """
        SELECT source_feature, collecting_method_version, reason_codes_json,
               COUNT(*) AS count
        FROM malformed_candidates
        GROUP BY source_feature, collecting_method_version, reason_codes_json
        ORDER BY source_feature, collecting_method_version, reason_codes_json
        """
    ):
        try:
            reasons = json.loads(row["reason_codes_json"])
        except (TypeError, json.JSONDecodeError):
            reasons = row["reason_codes_json"]
        current_malformed_breakdown.append(
            {
                "source_feature": row["source_feature"],
                "collecting_method_version": row["collecting_method_version"],
                "reason_codes": reasons,
                "count": int(row["count"]),
            }
        )

    summary = _json(run["summary_json"])
    summary_payload = dict(summary)
    summary_payload.setdefault("planned_rebuilds", action_counts.get("rebuild", 0))
    summary_payload.setdefault("completed_rebuilds", int(summary.get("rebuilt", 0)))
    summary_payload.setdefault(
        "completed_incremental", int(summary.get("incremental", 0))
    )
    planned_counts = dict(sorted(action_counts.items()))
    completed_counts = {
        "skip": status_counts.get("skipped", 0),
        "incremental": int(summary.get("incremental", 0)),
        "rebuild": int(summary.get("rebuilt", 0)),
        "block": status_counts.get("blocked", 0),
    }
    run_diagnostics_total = int(totals.get("diagnostics", 0))
    run_diagnostics_breakdown_total = sum(run_diagnostics_by_code.values())
    report: dict[str, Any] = {
        "tool": "google_fitness_native_processing",
        "source_key": SOURCE_KEY,
        "run": {
            "run_id": run["run_id"],
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "input_root": run["input_root"],
            "output_root": run["output_root"],
            "status": run["status"],
        },
        "configuration": dict(summary.get("configuration", {})),
        "actions": {
            "planned_counts": planned_counts,
            "completed_counts": completed_counts,
        },
        "participants": {
            "discovered": len(participant_rows),
            "status_counts": dict(sorted(status_counts.items())),
            "action_counts": planned_counts,
            "planned_action_counts": planned_counts,
            "failed": status_counts.get("failed", 0),
            "blocked": status_counts.get("blocked", 0),
            "committed": status_counts.get("committed", 0),
            "complete_empty": status_counts.get("complete_empty", 0),
            "skipped": status_counts.get("skipped", 0),
        },
        "accounting": dict(sorted(totals.items())),
        "features": {
            feature: dict(sorted(values.items()))
            for feature, values in sorted(feature_totals.items())
        },
        "diagnostics": {
            "run_total": run_diagnostics_total,
            "run_by_code": dict(sorted(run_diagnostics_by_code.items())),
            "run_breakdown_available": (
                run_diagnostics_total == run_diagnostics_breakdown_total
            ),
            "current_state_total": sum(current_diagnostics_by_code.values()),
            "current_state_by_code": current_diagnostics_by_code,
        },
        "malformed_candidates": {
            "run_total": int(totals.get("malformed_candidates", 0)),
            "current_state_total": sum(
                item["count"] for item in current_malformed_breakdown
            ),
            "current_state_by_contract_and_reason": current_malformed_breakdown,
        },
        "failure_behavior": {
            "failed_participants": failed,
            "blocked_participants": blocked,
        },
        "summary": summary_payload,
    }
    if include_details:
        report["participant_details"] = participant_details
        report["feature_details"] = feature_details
    return report


def load_processing_report(
    state_file: Path, run_id: str | None = None, *, include_details: bool = False,
) -> dict[str, Any]:
    path = Path(state_file)
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        _source_guard(connection)
        if run_id is None:
            row = connection.execute(
                "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise KeyError("state contains no processing runs")
            run_id = row["run_id"]
        return build_processing_report(
            connection, run_id, include_details=include_details
        )
    finally:
        connection.close()


def format_processing_report(report: Mapping[str, Any]) -> str:
    participants = report.get("participants", {})
    accounting = report.get("accounting", {})
    run = report.get("run", {})
    configuration = report.get("configuration", {})
    actions = report.get("actions", {})
    diagnostics = report.get("diagnostics", {})
    malformed = report.get("malformed_candidates", {})
    lines = [
        "Google Fitness native processing",
        f"Run: {run.get('run_id', report.get('run_id', 'unknown'))}",
        f"Status: {run.get('status', report.get('status', 'unknown'))}",
        "Configuration: "
        f"snapshot_policy={configuration.get('snapshot_policy')}, "
        f"verify_existing_hashes={configuration.get('verify_existing_hashes')}, "
        f"existing_output_validation={configuration.get('existing_output_validation')}",
        "Actions planned: " + ", ".join(
            f"{name}={count}"
            for name, count in sorted(actions.get("planned_counts", {}).items())
        ),
        "Actions completed: " + ", ".join(
            f"{name}={count}"
            for name, count in sorted(actions.get("completed_counts", {}).items())
        ),
        "Participants: "
        f"discovered={participants.get('discovered', report.get('discovered', 0))}, "
        f"committed={participants.get('committed', report.get('committed', 0))}, "
        f"empty={participants.get('complete_empty', report.get('empty', 0))}, "
        f"skipped={participants.get('skipped', report.get('skipped', 0))}, "
        f"failed={participants.get('failed', report.get('failed', 0))}, "
        f"blocked={participants.get('blocked', report.get('blocked', 0))}",
        "Accounting: "
        f"outer_rows={accounting.get('outer_rows', 0)}, "
        f"google_rows={accounting.get('google_rows', 0)}, "
        f"candidate_items={accounting.get('candidate_items', 0)}, "
        f"malformed={accounting.get('malformed_candidates', 0)}, "
        f"native_rows={accounting.get('output_rows', 0)}, "
        f"duplicates={accounting.get('exact_duplicates_removed', 0)}, "
        f"revisions={accounting.get('revisions_resolved', 0)}, "
        f"conflicts={accounting.get('unresolved_conflict_groups', 0)}",
        "Diagnostics: "
        f"run={diagnostics.get('run_total', 0)}, "
        f"current_state={diagnostics.get('current_state_total', 0)}, "
        f"by_code={diagnostics.get('current_state_by_code', {})}",
        "Malformed candidates: "
        f"run={malformed.get('run_total', 0)}, "
        f"current_state={malformed.get('current_state_total', 0)}",
    ]
    return "\n".join(lines) + "\n"


def default_state_file(output_root: Path) -> Path:
    return Path(output_root) / STATE_FILENAME
