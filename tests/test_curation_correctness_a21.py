"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import csv
import math
import sqlite3
from pathlib import Path
from wearable_project.curation.engine import curate_feature_file, validate_engine_rule_coverage
from wearable_project.curation.pipeline import curate_dataset
from wearable_project.curation.state import CurationStateDatabase
from wearable_project.curation.unit_resolution import build_participant_unit_context


BASE_FIELDS = [
    "start_date", "end_date", "value", "datetime", "created_at", "updated_at",
    "data_source", "collecting_method_version", "record_id", "source_id", "source_name",
]


def _write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _base_row(*, when: str, value: str, record: str, source: str = "source-a") -> dict[str, str]:
    return {
        "start_date": when,
        "end_date": when,
        "value": value,
        "datetime": when[:10] + "T00:00:00Z",
        "created_at": "2024-04-01T00:00:00Z",
        "updated_at": "2024-04-01T00:00:00Z",
        "data_source": "AppleHealthkit",
        "collecting_method_version": "2.0",
        "record_id": record,
        "source_id": source,
        "source_name": source,
    }


def _curate_one(participant: Path, feature: str, tmp_path: Path) -> tuple[list[dict[str, str]], object]:
    context = build_participant_unit_context(participant)
    output = tmp_path / f"curated-{feature}.csv"
    result = curate_feature_file(participant / f"{feature}.csv", output, feature, context)
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows, result


def test_report_separates_status_from_default_inclusion(tmp_path: Path) -> None:
    native = tmp_path / "native" / "p1" / "ActivitySummary.csv"
    fields = [
        "datetime", "apple_stand_hours", "apple_exercise_time", "active_energy_burned",
        "apple_stand_hours_goal", "apple_exercise_time_goal", "active_energy_burned_goal",
        "created_at", "updated_at", "data_source", "collecting_method_version", "payload_index",
    ]
    _write(native, fields, [{
        "datetime": "2024-01-01T00:00:00Z",
        "apple_stand_hours": "10", "apple_exercise_time": "20",
        "active_energy_burned": "400", "apple_stand_hours_goal": "12",
        "apple_exercise_time_goal": "30", "active_energy_burned_goal": "500",
        "created_at": "2024-01-02T00:00:00Z", "updated_at": "2024-01-02T00:00:00Z",
        "data_source": "AppleHealthkit", "collecting_method_version": "2.0",
        "payload_index": "0",
    }])
    summary = curate_dataset(tmp_path / "native", tmp_path / "curated", workers=1, max_in_flight=1, allow_unmanaged_native_root=True)
    activity = summary.as_dict()["run_activity"]
    assert activity["review_rows"] == 1
    assert activity["exclude_default_status_rows"] == 0
    assert activity["included_by_default_rows"] == 0
    assert activity["excluded_by_default_rows"] == 1
    assert activity["default_inclusion_accounting_complete"] is True


def test_resolved_to_unresolved_scale_creates_ambiguous_epoch_and_review(tmp_path: Path) -> None:
    participant = tmp_path / "native" / "p1"
    _write(participant / "Weight.csv", BASE_FIELDS, [
        # 110.231131... lb is exactly 50.0 kg and carries a reviewed metric-to-imperial conversion fingerprint.
        _base_row(when="2024-01-01T12:00:00Z", value="110.23113109243879", record="w1"),
        _base_row(when="2024-02-01T12:00:00Z", value="120", record="w2"),
    ])
    rows, result = _curate_one(participant, "Weight", tmp_path)
    assert math.isclose(float(rows[0]["canonical_value"]), 50.0, rel_tol=1e-10)
    assert rows[0]["canonical_unit"] == "kg"
    assert rows[1].get("canonical_value", "") == ""
    assert rows[1]["curation_unit_status"] == "ambiguous"
    assert rows[1]["curation_status"] == "review"
    assert rows[1]["include_by_default"] == "0"
    flags = set(rows[1]["curation_flags"].split(";"))
    assert {"possible_unit_transition", "unit_unresolved"}.issubset(flags)
    assert result.info.scale_transition_rows == 1
    assert result.info.review_rows == 1
    assert any(
        epoch.transition_reason == "scale_became_unresolved"
        for epoch in build_participant_unit_context(participant).epochs
    )


def test_bounded_bmi_evidence_overrides_weight_magnitude_and_propagates(tmp_path: Path) -> None:
    participant = tmp_path / "native" / "p1"
    when = "2024-01-01T12:00:00Z"
    _write(participant / "Height.csv", BASE_FIELDS, [
        _base_row(when=when, value="60", record="h1"),
    ])
    _write(participant / "Weight.csv", BASE_FIELDS, [
        _base_row(when=when, value="100", record="w1"),
        _base_row(when="2024-02-01T12:00:00Z", value="101", record="w2"),
    ])
    _write(participant / "BMI.csv", BASE_FIELDS, [
        _base_row(when=when, value=str(45.359237 / (1.524 ** 2)), record="b1"),
    ])
    rows, _ = _curate_one(participant, "Weight", tmp_path)
    assert math.isclose(float(rows[0]["canonical_value"]), 45.359237, rel_tol=1e-8)
    assert math.isclose(float(rows[1]["canonical_value"]), 101 * 0.45359237, rel_tol=1e-8)
    assert rows[0]["canonical_unit"] == "kg"
    assert rows[1]["canonical_unit"] == "kg"
    assert rows[0]["unit_evidence"].startswith("bounded_bmi_consistency")
    # Epoch-level reporting uses the strongest evidence observed inside the stable scale run.
    assert rows[1]["unit_evidence"].startswith("bounded_bmi_consistency")


def test_zero_duration_resting_and_walking_summaries_are_valid(tmp_path: Path) -> None:
    for feature in ("RestingHeartRate", "WalkingHeartRate"):
        participant = tmp_path / feature / "p1"
        _write(participant / f"{feature}.csv", BASE_FIELDS, [
            _base_row(when="2024-01-01T12:00:00Z", value="65", record="r1"),
        ])
        rows, result = _curate_one(participant, feature, tmp_path / feature)
        assert rows[0].get("curation_flags", "") == ""
        assert rows[0].get("curation_status", "") == ""
        assert result.info.pass_rows == 1


def test_sleep_overlap_annotations_are_source_aware(tmp_path: Path) -> None:
    participant = tmp_path / "native" / "p1"
    fields = BASE_FIELDS + ["device", "metadata_device_name"]
    def sleep(start: str, end: str, state: str, record: str, source: str) -> dict[str, str]:
        row = _base_row(when=start, value=state, record=record, source=source)
        row["end_date"] = end
        row["device"] = source + "-watch"
        row["metadata_device_name"] = source + "-device"
        return row
    _write(participant / "Sleep.csv", fields, [
        sleep("2024-01-01T00:00:00Z", "2024-01-01T08:00:00Z", "INBED", "a0", "A"),
        sleep("2024-01-01T01:00:00Z", "2024-01-01T03:00:00Z", "CORE", "a1", "A"),
        sleep("2024-01-01T02:00:00Z", "2024-01-01T04:00:00Z", "REM", "a2", "A"),
        sleep("2024-01-01T02:00:00Z", "2024-01-01T04:00:00Z", "REM", "b1", "B"),
    ])
    rows, _ = _curate_one(participant, "Sleep", tmp_path)
    flags = [set(filter(None, row.get("curation_flags", "").split(";"))) for row in rows]
    assert "same_source_detailed_sleep_stage_conflict" in flags[1]
    assert "same_source_detailed_sleep_stage_conflict" in flags[2]
    assert "cross_source_sleep_overlap" in flags[2]
    assert "cross_source_sleep_overlap" in flags[3]
    assert "source_has_no_inbed_state" not in flags[1]
    assert "source_has_no_inbed_state" not in flags[2]
    assert "source_has_no_inbed_state" in flags[3]
    # Cross-source overlap and a source that omits INBED are informational; only the same-source detailed-stage
    # conflict escalates status.
    assert rows[3].get("curation_status", "") == "pass"


def test_negative_nutrition_and_unknown_motion_context_are_executable_rules(tmp_path: Path) -> None:
    carb = tmp_path / "carb" / "p1"
    _write(carb / "Carbohydrates.csv", BASE_FIELDS, [
        _base_row(when="2024-01-01T12:00:00Z", value="-2", record="c1"),
    ])
    rows, _ = _curate_one(carb, "Carbohydrates", tmp_path / "carb")
    assert "negative_nutrition_amount" in rows[0]["curation_flags"]
    assert rows[0]["curation_status"] == "review"

    heart = tmp_path / "heart" / "p1"
    fields = BASE_FIELDS + ["heart_rate_motion_context"]
    row = _base_row(when="2024-01-01T12:00:00Z", value="70", record="h1")
    row["heart_rate_motion_context"] = "walking_fast"
    _write(heart / "HeartRate.csv", fields, [row])
    _, result = _curate_one(heart, "HeartRate", tmp_path / "heart")
    assert result.info.unknown_category_counts == {
        "HeartRate:heart_rate_motion_context=walking_fast": 1
    }
    assert result.info.flag_counts["unknown_heart_rate_motion_context"] == 1


def test_blankish_cgm_metadata_is_not_reported_as_unknown_category(tmp_path: Path) -> None:
    participant = tmp_path / "native" / "p1"
    fields = BASE_FIELDS + ["status", "trend_arrow", "trend_rate"]
    row = _base_row(when="2024-01-01T12:00:00Z", value="6.2", record="g1")
    row.update({"status": "IN_RANGE", "trend_arrow": "None", "trend_rate": ""})
    _write(participant / "BloodGlucose.csv", fields, [row])
    rows, result = _curate_one(participant, "BloodGlucose", tmp_path)
    assert result.info.unknown_category_counts == {}
    assert "unknown_cgm_trend" not in result.info.flag_counts
    assert rows[0].get("curation_status", "") == ""


def test_every_referenced_runtime_rule_has_engine_implementation() -> None:
    assert validate_engine_rule_coverage() == ()


def test_state_database_migrates_a2_schema_additively(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
        CREATE TABLE curation_outputs(
          participant_id TEXT, feature TEXT, filename TEXT, native_rows INTEGER,
          curated_rows INTEGER, size_bytes INTEGER, sha256 TEXT, native_sha256 TEXT,
          policy_fingerprint TEXT, pass_rows INTEGER, review_rows INTEGER,
          exclude_default_rows INTEGER, canonical_value_rows INTEGER,
          ambiguous_unit_rows INTEGER, scale_transition_rows INTEGER,
          flag_counts_json TEXT, acquisition_counts_json TEXT,
          PRIMARY KEY(participant_id,feature)
        );
        CREATE TABLE curation_run_features(
          run_id TEXT, participant_id TEXT, feature TEXT, native_rows INTEGER,
          curated_rows INTEGER, output_size_bytes INTEGER, pass_rows INTEGER,
          review_rows INTEGER, exclude_default_rows INTEGER, canonical_value_rows INTEGER,
          ambiguous_unit_rows INTEGER, scale_transition_rows INTEGER,
          flag_counts_json TEXT, acquisition_counts_json TEXT, policy_fingerprint TEXT,
          PRIMARY KEY(run_id,participant_id,feature)
        );
        CREATE TABLE curation_unit_epochs(
          participant_id TEXT, feature TEXT, epoch_id TEXT, context_id TEXT,
          start_index INTEGER, end_index INTEGER, first_time TEXT, last_time TEXT,
          row_count INTEGER, raw_unit TEXT, canonical_unit TEXT, status TEXT,
          evidence TEXT, scale_transition INTEGER, median_value REAL,
          PRIMARY KEY(participant_id,feature,epoch_id)
        );
        """)
    with CurationStateDatabase(path) as state:
        assert state.connection.execute("PRAGMA user_version").fetchone()[0] == 3
        output_columns = {row[1] for row in state.connection.execute("PRAGMA table_info(curation_outputs)")}
        epoch_columns = {row[1] for row in state.connection.execute("PRAGMA table_info(curation_unit_epochs)")}
    assert {"included_by_default_rows", "excluded_by_default_rows", "unit_status_counts_json"}.issubset(output_columns)
    assert {
        "transition_from_unit", "transition_reason", "collecting_method_version",
        "source_id", "source_name", "device", "metadata_device_name",
        "time_zone", "was_user_entered",
    }.issubset(epoch_columns)
