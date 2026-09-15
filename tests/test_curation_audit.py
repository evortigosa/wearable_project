"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from wearable_project.curation.audit import run_curation_audit


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _native_fixture(root: Path) -> Path:
    participant = root / "p1"
    common = {
        "datetime": "2024-01-01T00:00:00Z",
        "created_at": "2024-02-01T00:00:00Z",
        "updated_at": "2024-02-01T00:00:00Z",
        "data_source": "AppleHealthkit",
        "collecting_method_version": "2.0",
        "record_id": "r1",
        "source_id": "source.app",
        "source_name": "Source",
        "utc_offset_minutes": "120",
    }
    _write(participant / "Weight.csv", [
        {"start_date": "2024-01-01T10:00:00Z", "end_date": "2024-01-01T10:00:00Z", "value": "176.3698", **common},
    ])
    _write(participant / "Height.csv", [
        {"start_date": "2024-01-01T10:00:00Z", "end_date": "2024-01-01T10:00:00Z", "value": "70", **common},
    ])
    _write(participant / "BMI.csv", [
        {"start_date": "2024-01-01T10:00:00Z", "end_date": "2024-01-01T10:00:00Z", "value": "25.3", **common},
    ])
    activity_fields = {
        "apple_stand_hours": "10", "apple_exercise_time": "30", "active_energy_burned": "500",
        "apple_stand_hours_goal": "12", "apple_exercise_time_goal": "30", "active_energy_burned_goal": "600",
        "created_at": common["created_at"], "updated_at": common["updated_at"],
        "data_source": "AppleHealthkit", "collecting_method_version": "2.0",
    }
    a = {"datetime": "2024-01-01T00:00:00Z", "payload_index": "0", **activity_fields}
    b = {"datetime": "2024-01-01T00:00:00Z", "payload_index": "1", **{**activity_fields, "active_energy_burned": "550"}}
    b2 = {"datetime": "2024-01-02T00:00:00Z", "payload_index": "0", **{**activity_fields, "active_energy_burned": "550"}}
    c = {"datetime": "2024-01-02T00:00:00Z", "payload_index": "1", **{**activity_fields, "active_energy_burned": "575"}}
    _write(participant / "ActivitySummary.csv", [a, b, b2, c])
    return root


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def test_audit_is_read_only_and_emits_calibration_tables(tmp_path: Path) -> None:
    native = _native_fixture(tmp_path / "native")
    before = _tree_hashes(native)
    output = tmp_path / "audit"
    summary = run_curation_audit(native, output, workers=1, policy_scope="provisional")
    assert summary.status == "complete"
    assert summary.participants_completed == 1
    assert summary.rows_read == 6
    assert summary.participants_failed == 0
    assert _tree_hashes(native) == before
    assert (output / "source_context_epochs.csv").exists()
    assert (output / "unit_candidate_distributions.csv").exists()
    assert (output / "cross_feature_consistency.csv").exists()
    assert (output / "activity_summary_alignment.csv").exists()
    assert (output / "temporal_scale_windows.csv").exists()
    assert (output / "audit_coverage.csv").exists()
    payload = json.loads((output / "policy_audit_summary.json").read_text())
    assert payload["registry_fingerprint"]

    activity = list(csv.DictReader((output / "activity_summary_alignment.csv").open()))
    assert activity[0]["second_item_equals_next_first"] == "1"
    coverage = list(csv.DictReader((output / "audit_coverage.csv").open()))
    assert any(row["audit_id"] == "activity_summary_sliding_pair_alignment" and row["status"] == "evidence_generated" for row in coverage)
    cross = list(csv.DictReader((output / "cross_feature_consistency.csv").open()))
    best = min(
        (row for row in cross if row["check"] == "bmi_height_weight_consistency"),
        key=lambda row: float(row["median_absolute_error"]),
    )
    assert best["candidate_height_unit"] == "in"
    assert best["candidate_weight_unit"] == "lb"
    assert best["candidate_selected"] == "False"


def test_audit_refuses_nested_output(tmp_path: Path) -> None:
    from wearable_project.curation.audit import AuditError
    native = _native_fixture(tmp_path / "native")
    try:
        run_curation_audit(native, native / "audit", workers=1)
    except AuditError as exc:
        assert "separate and non-nested" in str(exc)
    else:
        raise AssertionError("Expected AuditError")


def test_audit_emits_hardened_outputs_and_bounded_cross_feature_metrics(tmp_path: Path) -> None:
    native = _native_fixture(tmp_path / "native")
    output = tmp_path / "audit"
    summary = run_curation_audit(native, output, workers=1, policy_scope="calibration")
    assert summary.source_context_epochs >= summary.source_context_groups
    assert summary.invalid_timestamp_rows == 0
    for name in (
        "source_context_groups.csv",
        "source_context_epochs.csv",
        "measurement_input_quality.csv",
        "nutrition_energy_consistency.csv",
        "policy_calibration_decisions.json",
        "policy_payload_manifest.json",
        "audit_environment.json",
    ):
        assert (output / name).exists()
    cross = list(csv.DictReader((output / "cross_feature_consistency.csv").open()))
    bmi_rows = [row for row in cross if row["check"] == "bmi_height_weight_consistency"]
    assert bmi_rows
    assert "exact_match_count" in bmi_rows[0]
    assert "within_5_minutes_median_absolute_error" in bmi_rows[0]
    manifest = json.loads((output / "policy_payload_manifest.json").read_text())
    assert "DailyDistanceCycling" in manifest["policies"]
    assert manifest["policies"]["DailyDistanceCycling"]["policy"]


def test_source_context_groups_and_true_epochs_are_distinct(tmp_path: Path) -> None:
    root = tmp_path / "native"
    participant = root / "p1"
    common = {
        "end_date": "2024-01-01T00:00:00Z",
        "value": "70",
        "datetime": "2024-01-01T00:00:00Z",
        "created_at": "2024-02-01T00:00:00Z",
        "updated_at": "2024-02-01T00:00:00Z",
        "data_source": "AppleHealthkit",
        "collecting_method_version": "2.0",
        "source_name": "Source",
    }
    rows = [
        {"start_date": "2024-01-01T00:00:00Z", "source_id": "A", **common},
        {"start_date": "2024-01-02T00:00:00.000000Z", "source_id": "B", **common},
        {"start_date": "2024-01-03T00:00:00Z", "source_id": "A", **common},
    ]
    _write(participant / "Weight.csv", rows)
    output = tmp_path / "audit"
    run_curation_audit(root, output, workers=1, features=("Weight",))
    groups = list(csv.DictReader((output / "source_context_groups.csv").open()))
    epochs = list(csv.DictReader((output / "source_context_epochs.csv").open()))
    assert len(groups) == 2
    assert len(epochs) == 3
    group_a = next(row for row in groups if row["source_id"] == "A")
    assert group_a["context_segments"] == "2"


def test_mixed_iso_timestamps_are_parsed_without_loss(tmp_path: Path) -> None:
    root = tmp_path / "native"
    participant = root / "p1"
    common = {
        "value": "70",
        "datetime": "2024-01-01T00:00:00Z",
        "created_at": "2024-02-01T00:00:00Z",
        "updated_at": "2024-02-01T00:00:00Z",
        "data_source": "AppleHealthkit",
        "collecting_method_version": "2.0",
        "source_id": "A",
        "source_name": "Source",
    }
    _write(participant / "Weight.csv", [
        {"start_date": "2024-01-01T00:00:00Z", "end_date": "2024-01-01T00:00:00Z", **common},
        {"start_date": "2024-01-02T00:00:00.123456Z", "end_date": "2024-01-02T00:00:00.123456Z", **common},
    ])
    output = tmp_path / "audit"
    summary = run_curation_audit(root, output, workers=1, features=("Weight",))
    assert summary.invalid_timestamp_rows == 0
    quality = list(csv.DictReader((output / "measurement_input_quality.csv").open()))
    native_quality = [row for row in quality if row["audit_stage"] == "source-context-audit"]
    assert native_quality[0]["valid_timestamp_rows"] == "2"
    assert native_quality[0]["invalid_timestamp_rows"] == "0"


def test_nutrition_energy_audit_emits_kcal_and_kj_candidates(tmp_path: Path) -> None:
    root = tmp_path / "native"
    participant = root / "p1"
    common = {
        "start_date": "2024-01-01T12:00:00Z",
        "end_date": "2024-01-01T12:00:00Z",
        "datetime": "2024-01-01T00:00:00Z",
        "created_at": "2024-02-01T00:00:00Z",
        "updated_at": "2024-02-01T00:00:00Z",
        "data_source": "AppleHealthkit",
        "collecting_method_version": "2.0",
        "source_id": "nutrition.app",
        "source_name": "Nutrition",
    }
    _write(participant / "EnergyConsumed.csv", [{"value": "170", **common}])
    _write(participant / "Carbohydrates.csv", [{"value": "20", **common}])
    _write(participant / "Protein.csv", [{"value": "10", **common}])
    _write(participant / "TotalFat.csv", [{"value": "5.5555555556", **common}])
    output = tmp_path / "audit"
    run_curation_audit(root, output, workers=1, policy_scope="calibration")
    rows = list(csv.DictReader((output / "nutrition_energy_consistency.csv").open()))
    exact = [row for row in rows if row["match_scope"] == "exact_timestamp_same_source"]
    assert {row["candidate_energy_raw_unit"] for row in exact} == {"kcal", "kJ"}
    kcal = next(row for row in exact if row["candidate_energy_raw_unit"] == "kcal")
    assert float(kcal["median_absolute_error_kcal"]) < 1e-6
