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


def test_scan_and_process_reports_record_hash_validation_mode(tmp_path: Path) -> None:
    raw = _fixture(tmp_path)
    output = tmp_path / "cleaned"
    first = process_dataset(raw, output, workers=2, max_in_flight=2)
    assert first.failed == 0

    scan = scan_processing_plan(
        raw,
        output,
        workers=2,
        verify_existing_hashes=True,
        show_progress=False,
    )
    scan_payload = scan.as_dict()
    assert scan_payload["configuration"]["verify_existing_hashes"] is True
    assert scan_payload["validation"] == {
        "source_files": "sha256",
        "existing_outputs": "file_set_size_and_sha256",
    }

    noop = process_dataset(
        raw,
        output,
        workers=2,
        max_in_flight=2,
        verify_existing_hashes=True,
    ).as_dict()
    assert noop["configuration"]["verify_existing_hashes"] is True
    assert (
        noop["configuration"]["existing_output_validation"]
        == "file_set_size_and_sha256"
    )
    assert noop["actions"]["planned_counts"] == {"skip": 2}
    assert noop["actions"]["completed_counts"] == {
        "skip": 2,
        "incremental": 0,
        "rebuild": 0,
        "block": 0,
    }
    assert noop["summary"]["planned_rebuilds"] == 0
    assert noop["summary"]["completed_rebuilds"] == 0


def test_report_distinguishes_run_and_current_state_diagnostics(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_month(
        raw,
        "P1",
        "2024-1.csv",
        [
            _row(
                "W2",
                "P1",
                "GoogleFitness",
                "Weight",
                "2.0",
                [None],
            )
        ],
    )
    output = tmp_path / "cleaned"
    first = process_dataset(raw, output, workers=1, max_in_flight=1).as_dict()

    assert first["accounting"]["malformed_candidates"] == 1
    assert first["diagnostics"] == {
        "run_total": 1,
        "run_by_code": {"malformed_candidate": 1},
        "run_breakdown_available": True,
        "current_state_total": 1,
        "current_state_by_code": {"malformed_candidate": 1},
    }
    assert first["malformed_candidates"] == {
        "run_total": 1,
        "current_state_total": 1,
        "current_state_by_contract_and_reason": [
            {
                "source_feature": "Weight",
                "collecting_method_version": "2.0",
                "reason_codes": [
                    "missing_or_nonnumeric_value",
                    "non_dict_list_item",
                ],
                "count": 1,
            }
        ],
    }

    noop = process_dataset(raw, output, workers=1, max_in_flight=1).as_dict()
    assert noop["diagnostics"]["run_total"] == 0
    assert noop["diagnostics"]["run_by_code"] == {}
    assert noop["diagnostics"]["current_state_by_code"] == {"malformed_candidate": 1}
    assert noop["malformed_candidates"]["run_total"] == 0
    assert noop["malformed_candidates"]["current_state_total"] == 1


def test_central_release_manifest_tracks_google_processing_modules() -> None:
    from wearable_project.utils.regenerate_manifest import compute
    from wearable_project.utils.release_manifest import EXPECTED_GFIT_MODULE_SHA256

    computed = compute()
    assert computed["GFIT_EXPECTED_MODULE_SHA256"] == dict(sorted(EXPECTED_GFIT_MODULE_SHA256.items()))
