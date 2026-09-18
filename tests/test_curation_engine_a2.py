"""
Wearable Data Processing and Modeling project
"""


import csv
import json
from pathlib import Path
from wearable_project.curation.engine import curate_feature_file
from wearable_project.curation.pipeline import curate_dataset
from wearable_project.curation.unit_resolution import build_participant_unit_context


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_curation_never_duplicates_native_column_names(tmp_path: Path) -> None:
    participant = tmp_path / "native" / "p1"
    native = participant / "BloodAlcoholContent.csv"
    fields = [
        "start_date", "end_date", "value", "datetime", "created_at", "updated_at",
        "data_source", "collecting_method_version", "record_id", "source_id",
        "source_name", "utc_offset_minutes", "acquisition_method", "canonical_value",
        "canonical_unit", "unit_status",
    ]
    row = {
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-01-01T00:00:00Z",
        "value": "0.001",
        "datetime": "2024-01-01T00:00:00Z",
        "created_at": "2024-01-02T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
        "data_source": "AppleHealthkit",
        "collecting_method_version": "2.0",
        "record_id": "r1",
        "source_id": "com.rwichmann.intellidrink-lite",
        "source_name": "IntelliDrink",
        "utc_offset_minutes": "0",
        "acquisition_method": "calculator_estimate",
        "canonical_value": "0.1",
        "canonical_unit": "%",
        "unit_status": "inferred_high_confidence",
    }
    _write_csv(native, fields, [row])
    context = build_participant_unit_context(participant)
    output = tmp_path / "curated.csv"
    curate_feature_file(native, output, "BloodAlcoholContent", context)

    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        curated = next(reader)

    assert len(header) == len(set(header))
    assert curated["canonical_value"] == "0.1"
    assert curated["canonical_unit"] == "%"
    assert curated["unit_status"] == "inferred_high_confidence"
    assert curated["acquisition_method"] == "calculator_estimate"
    assert curated["curation_unit_status"] == "resolved_reviewed"


def test_activity_summary_does_not_receive_scalar_unit_failure(tmp_path: Path) -> None:
    participant = tmp_path / "native" / "p1"
    native = participant / "ActivitySummary.csv"
    fields = [
        "datetime", "apple_stand_hours", "apple_exercise_time", "active_energy_burned",
        "apple_stand_hours_goal", "apple_exercise_time_goal", "active_energy_burned_goal",
        "created_at", "updated_at", "data_source", "collecting_method_version",
        "payload_index", "quality_flags",
    ]
    _write_csv(native, fields, [{
        "datetime": "2024-01-01T00:00:00Z",
        "apple_stand_hours": "10",
        "apple_exercise_time": "20",
        "active_energy_burned": "400",
        "apple_stand_hours_goal": "12",
        "apple_exercise_time_goal": "30",
        "active_energy_burned_goal": "500",
        "created_at": "2024-01-02T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
        "data_source": "AppleHealthkit",
        "collecting_method_version": "2.0",
        "payload_index": "0",
        "quality_flags": "summary_date_assignment_ambiguous",
    }])
    context = build_participant_unit_context(participant)
    output = tmp_path / "curated.csv"
    curate_feature_file(native, output, "ActivitySummary", context)
    with output.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    flags = set(filter(None, row.get("curation_flags", "").split(";")))
    assert "unit_unresolved" not in flags
    assert "derivation_blocked_by_policy" in flags
    assert row.get("curation_unit_status", "") == ""


def test_final_report_contains_finish_time_and_wall_clock(tmp_path: Path) -> None:
    native_root = tmp_path / "native"
    participant = native_root / "p1"
    _write_csv(participant / "HeartRate.csv", [
        "start_date", "end_date", "value", "datetime", "created_at", "updated_at",
        "data_source", "collecting_method_version", "record_id", "source_id", "source_name",
    ], [{
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-01-01T00:00:00Z",
        "value": "70",
        "datetime": "2024-01-01T00:00:00Z",
        "created_at": "2024-01-02T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
        "data_source": "AppleHealthkit",
        "collecting_method_version": "2.0",
        "record_id": "r1",
        "source_id": "watch",
        "source_name": "Watch",
    }])
    output = tmp_path / "curated"
    summary = curate_dataset(native_root, output, workers=1, max_in_flight=1, allow_unmanaged_native_root=True)
    report = summary.as_dict()
    assert report["finished_at"] is not None
    assert report["performance"]["wall_clock_seconds"] is not None
    assert report["run_activity"]["native_rows_read"] == 1
    assert report["run_activity"]["curated_rows_written"] == 1
