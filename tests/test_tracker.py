"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import csv
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def write_rows(path: Path, participant: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "participant_id", "data_source", "name", "datetime", "data",
        "collecting_method_version", "created_at", "updated_at",
    ]
    step_items = [
        {
            "id": "step-a", "value": 100,
            "start_date": "2024-12-01T10:00:00+0200",
            "end_date": "2024-12-01T11:00:00+0200",
        },
        {
            "id": "step-b", "value": 100,
            "start_date": "2024-12-01T10:00:00+0200",
            "end_date": "2024-12-01T11:00:00+0200",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        common = {
            "participant_id": participant,
            "data_source": "AppleHealthkit",
            "datetime": "2024-12-01 00:00:00",
            "collecting_method_version": "2.0",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
        }
        writer.writerow({
            **common, "id": "outer-1", "name": "StepCount",
            "data": json.dumps(repr(step_items)),
        })
        writer.writerow({
            **common, "id": "outer-2", "name": "NovelMetric",
            "data": json.dumps(repr([{
                "id": "novel-1", "value": 3,
                "start_date": "2024-12-01T12:00:00+0200",
                "end_date": "2024-12-01T12:00:00+0200",
            }])),
        })
        writer.writerow({
            **common, "id": "outer-3", "name": "HeartRate", "data": "not-decodable",
        })


def run_process(input_root: Path, output_root: Path) -> dict:
    command = [
        sys.executable, "-m", "wearable_project", "process", "--input", str(input_root), "--output",
        str(output_root), "--workers", "1", "--max-in-flight", "1", "--row-error-policy", "skip-row",
        "--json-summary",
    ]
    result = subprocess.run(
        command, cwd=Path(__file__).parents[1], text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_processing_tracker_reports_run_and_snapshot_metrics(tmp_path: Path) -> None:
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    write_rows(input_root / "P1" / "2024-12.csv", "P1")

    report = run_process(input_root, output_root)
    assert report["participants"] == {
        "blocked": 0, "committed": 1, "discovered": 1, "empty": 0,
        "failed": 0, "incremental": 0, "rebuilt": 1, "skipped": 0,
    }
    activity = report["run_activity"]
    assert activity["monthly_files_read"] == 1
    assert activity["monthly_files_completed"] == 1
    assert activity["outer_rows_read"] == 3
    assert activity["apple_rows_read"] == 3
    assert activity["payloads_decoded"] == 2
    assert activity["payload_parse_failures"] == 1
    assert activity["raw_occurrences_parsed"] == 3
    assert activity["canonical_rows_written"] == 2
    assert activity["duplicates_reconciled"] == 1
    assert activity["revisions_reconciled"] == 0
    assert report["unknowns"]["feature_names"] == ["NovelMetric"]
    assert report["unknowns"]["diagnostic_counts"]["payload-decode"] == 1
    assert report["snapshot"]["canonical_rows"] == 2
    assert report["snapshot"]["represented_source_occurrences"] == 3
    assert report["performance"]["wall_clock_seconds"] > 0
    assert report["performance"]["largest_participant_runtime_id"] == "P1"

    state_file = output_root / ".wearable_state.sqlite"
    with sqlite3.connect(state_file) as connection:
        participant = connection.execute(
            "SELECT runtime_seconds,peak_rss_bytes FROM run_participants WHERE run_id=? AND participant_id='P1'",
            (report["run_id"],),
        ).fetchone()
        assert participant[0] > 0
        assert participant[1] is None or participant[1] > 0
        features = connection.execute(
            "SELECT COUNT(*) FROM run_features WHERE run_id=?", (report["run_id"],)
        ).fetchone()[0]
        assert features == 2

    noop = run_process(input_root, output_root)
    assert noop["participants"]["skipped"] == 1
    assert noop["run_activity"]["monthly_files_read"] == 0
    assert noop["snapshot"]["canonical_rows"] == 2

    command = [
        sys.executable, "-m", "wearable_project", "report",
        "--output", str(output_root), "--json", "--include-details",
    ]
    persisted = subprocess.run(
        command, cwd=Path(__file__).parents[1], text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    assert persisted.returncode == 0, persisted.stdout + persisted.stderr
    persisted_report = json.loads(persisted.stdout)
    assert persisted_report["run_id"] == noop["run_id"]
    assert persisted_report["snapshot"]["represented_source_occurrences"] == 3
    assert len(persisted_report["participant_details"]) == 1
    assert persisted_report["participant_details"][0]["participant_id"] == "P1"
    assert persisted_report["feature_details"] == []  # no-op run touched no features
    assert persisted_report["diagnostics"] == []


def test_tracker_migrates_v011_state_database_in_place(tmp_path: Path) -> None:
    from wearable_project.processing.writer import StateDatabase

    state_file = tmp_path / ".wearable_state.sqlite"
    with sqlite3.connect(state_file) as connection:
        connection.executescript(
            """
            CREATE TABLE runs(
              run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
              input_root TEXT NOT NULL, output_root TEXT NOT NULL, status TEXT NOT NULL,
              summary_json TEXT
            );
            CREATE TABLE participants(
              participant_id TEXT PRIMARY KEY, status TEXT NOT NULL,
              parser_version TEXT, registry_version TEXT, source_signature TEXT,
              last_error TEXT, updated_at TEXT NOT NULL
            );
            CREATE TABLE source_files(
              participant_id TEXT NOT NULL, canonical_month TEXT NOT NULL,
              filename TEXT NOT NULL, sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
              PRIMARY KEY(participant_id, canonical_month)
            );
            CREATE TABLE feature_outputs(
              participant_id TEXT NOT NULL, feature TEXT NOT NULL, filename TEXT NOT NULL,
              row_count INTEGER NOT NULL, size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
              PRIMARY KEY(participant_id, feature)
            );
            """
        )

    with StateDatabase(state_file) as state:
        columns = {
            row["name"] for row in state.connection.execute("PRAGMA table_info(feature_outputs)")
        }
        assert "represented_occurrences" in columns
        tables = {
            row[0] for row in state.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"run_tracking", "run_participants", "run_features", "run_diagnostics"} <= tables


def test_planning_failure_is_persisted_without_aborting_other_participants(tmp_path: Path) -> None:
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    write_rows(input_root / "GOOD" / "2024-12.csv", "GOOD")
    write_rows(input_root / "BAD" / "2024-1.csv", "BAD")
    write_rows(input_root / "BAD" / "2024-01.csv", "BAD")

    command = [
        sys.executable, "-m", "wearable_project", "process", "--input", str(input_root), "--output",
        str(output_root), "--workers", "1", "--max-in-flight", "1", "--row-error-policy", "skip-row",
        "--json-summary",
    ]
    result = subprocess.run(
        command, cwd=Path(__file__).parents[1], text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["participants"]["discovered"] == 2
    assert report["participants"]["committed"] == 1
    assert report["participants"]["failed"] == 1
    assert "BAD" in report["failure_behavior"]["failed_participants"]
    assert (output_root / "GOOD" / "StepCount.csv").is_file()
    assert not (output_root / "BAD").exists()


def test_report_explains_pre_telemetry_state_database(tmp_path: Path) -> None:
    state_file = tmp_path / ".wearable_state.sqlite"
    with sqlite3.connect(state_file) as connection:
        connection.execute(
            """CREATE TABLE runs(
                 run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
                 input_root TEXT NOT NULL, output_root TEXT NOT NULL, status TEXT NOT NULL,
                 summary_json TEXT
               )"""
        )
    command = [
        sys.executable, "-m", "wearable_project", "report", "--state-file", str(state_file), "--json",
    ]
    result = subprocess.run(
        command, cwd=Path(__file__).parents[1], text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    assert result.returncode == 2
    assert "predates processing telemetry" in result.stderr
