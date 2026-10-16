"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any
import pytest
from wearable_project.exceptions import OutputValidationError
from wearable_project.processing.google_fitness.pipeline import process_dataset
from wearable_project.processing.google_fitness.registry import (
    OUTER_SCHEMA_COLUMNS, STEP_COUNT_DATA_SOURCE_ID, STEP_COUNT_DATA_TYPE_NAME,
)
from wearable_project.processing.google_fitness.scan import scan_processing_plan
from wearable_project.processing.google_fitness.tracker import load_processing_report
from wearable_project.processing.google_fitness.writer import StateDatabase


def _row(
    outer_id: str, participant: str, source: str, feature: str, version: str, payload: Any,
    dt: str = "2024-01-01 00:00:00",
) -> dict[str, str]:
    return {
        "id": outer_id,
        "participant_id": participant,
        "data_source": source,
        "name": feature,
        "datetime": dt,
        "data": repr(payload),
        "collecting_method_version": version,
        "created_at": "2024-01-02 00:00:00",
        "updated_at": "2024-01-03 00:00:00",
    }


def _write_month(root: Path, participant: str, filename: str, rows: list[dict[str, str]]) -> Path:
    folder = root / participant
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTER_SCHEMA_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _step_alternative(source: str, date: str, value: int) -> dict[str, Any]:
    return {
        "source": source,
        "steps": [{"date": date, "value": value}],
        "raw_steps": [
            {
                "steps": value,
                "start_date": 1704067200000,
                "end_date": 1704153599999,
                "data_source_id": STEP_COUNT_DATA_SOURCE_ID,
                "data_type_name": STEP_COUNT_DATA_TYPE_NAME,
                "origin_data_source_id": "raw:com.google.step_count.delta:test",
                "app_name": "Example",
                "app_package_name": "org.example",
                "device_manufacturer": "Example",
                "device_model": "Model",
                "device_type": "watch",
                "device_uid": "uid",
            }
        ],
    }


def _fixture(root: Path) -> Path:
    raw = root / "raw"
    google_rows = [
        _row(
            "A1",
            "P1",
            "AppleHealthkit",
            "Electrocardiogram",
            "1.0",
            "invalid and intentionally ignored",
        ),
        _row(
            "G1",
            "P1",
            "GoogleFitness",
            "Activities",
            "1.0",
            [
                {
                    "start": 1704067200000,
                    "end": 1704070800000,
                    "activity_name": "walking",
                    "tracked": True,
                    "device": "Android",
                    "calories": 12.5,
                    "distance": 1000.0,
                    "quantity": 1200,
                    "source_name": "Android",
                    "source_id": "com.google.android.gms",
                },
                {
                    "start": 1704070800000,
                    "end": 1704074400000,
                    "activity_name": "sleep.deep",
                    "tracked": True,
                    "device": "Android Wear",
                    "calories": 1.0,
                    "quantity": 1,
                    "source_name": "Android Wear",
                    "source_id": "com.google.android.gms",
                },
            ],
        ),
        _row(
            "G2",
            "P1",
            "GoogleFitness",
            "Calories",
            "1.0",
            [{
                "start_date": "2024-01-01T00:00:00.000Z",
                "end_date": "2024-01-01T23:59:59.999Z",
                "calorie": -0.5,
                "day": "Mon",
            }],
        ),
        _row(
            "G3",
            "P1",
            "GoogleFitness",
            "StepCount",
            "2.0",
            [
                _step_alternative("estimated_steps", "2024-01-01", 100),
                _step_alternative("merge_step_deltas", "2024-01-01", 110),
            ],
        ),
    ]
    _write_month(raw, "P1", "2024-1.csv", google_rows)
    _write_month(
        raw,
        "P2",
        "2024-1.csv",
        [_row("A2", "P2", "AppleHealthkit", "StepCount", "1.0", "bad")],
    )
    return raw


def _csv_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*.csv"))
    }


def test_end_to_end_transactional_processing_and_noop(tmp_path: Path) -> None:
    raw = _fixture(tmp_path)
    output = tmp_path / "cleaned"
    summary = process_dataset(raw, output, workers=2, max_in_flight=2)
    assert summary.failed == 0
    assert summary.committed == 1
    assert summary.empty == 1
    assert (output / "P1" / "Walking.csv").is_file()
    assert (output / "P1" / "Sleep.csv").is_file()
    assert (output / "P1" / "StepCount.csv").is_file()
    assert not list(output.rglob("Activities.csv"))
    assert not (output / "P2").exists()

    with (output / "P1" / "Sleep.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["value"] for row in rows] == ["DEEP"]
    with (output / "P1" / "Calories.csv").open(newline="", encoding="utf-8") as handle:
        assert float(next(csv.DictReader(handle))["value"]) == -0.5
    with (output / "P1" / "StepCount.csv").open(newline="", encoding="utf-8") as handle:
        step_rows = list(csv.DictReader(handle))
    assert {row["aggregation_source"] for row in step_rows} == {
        "estimated_steps",
        "merge_step_deltas",
    }

    second = process_dataset(raw, output, workers=2, max_in_flight=2)
    assert second.skipped == 2
    assert second.committed == 0
    assert second.as_dict()["accounting"].get("outer_rows", 0) == 0

    scan = scan_processing_plan(raw, output, workers=2, show_progress=False)
    assert scan.safe_to_process
    assert scan.no_op
    assert scan.counts == {"skip": 2}

    report = load_processing_report(output / ".wearable_state.sqlite")
    assert report["participants"]["skipped"] == 2


def test_worker_count_does_not_change_native_csvs(tmp_path: Path) -> None:
    raw = _fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    process_dataset(raw, first, workers=2, max_in_flight=2)
    process_dataset(raw, second, workers=2, max_in_flight=1)
    assert _csv_bytes(first) == _csv_bytes(second)


def test_incremental_and_strict_missing_month_plans(tmp_path: Path) -> None:
    raw = _fixture(tmp_path)
    output = tmp_path / "cleaned"
    process_dataset(raw, output, workers=2, max_in_flight=2)
    _write_month(
        raw,
        "P1",
        "2024-2.csv",
        [_row(
            "G4",
            "P1",
            "GoogleFitness",
            "Calories",
            "1.0",
            [{
                "start_date": "2024-02-01T00:00:00.000Z",
                "end_date": "2024-02-01T23:59:59.999Z",
                "calorie": 10.0,
                "day": "Thu",
            }],
            dt="2024-02-01 00:00:00",
        )],
    )
    scan = scan_processing_plan(raw, output, workers=2, show_progress=False)
    actions = {item.participant_id: item.action for item in scan.participants}
    assert actions["P1"] == "incremental"
    (raw / "P1" / "2024-1.csv").unlink()
    blocked = scan_processing_plan(raw, output, workers=2, show_progress=False)
    actions = {item.participant_id: item.action for item in blocked.participants}
    assert actions["P1"] == "block"


def test_state_source_guard_rejects_other_source(tmp_path: Path) -> None:
    path = tmp_path / ".wearable_state.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    connection.execute("INSERT INTO metadata VALUES('source_key','apple_healthkit')")
    connection.commit()
    connection.close()
    with pytest.raises(OutputValidationError):
        StateDatabase(path)
