"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import csv
from hashlib import sha256
import json
from pathlib import Path
import pytest
from wearable_project.curation.google_fitness import audit as audit_module
from wearable_project.curation.google_fitness.audit import run_audit
from wearable_project.curation.google_fitness.audit_models import (
    AUDIT_VERSION, REPORT_SCHEMA_VERSION, ParticipantAuditResult,
)
from wearable_project.curation.google_fitness.registry import get_policy
from wearable_project.curation.google_fitness.evidence import EVIDENCE
from wearable_project.processing.google_fitness.writer import StateDatabase, write_feature_csv


def _rows() -> dict[str, dict[str, list[dict[str, object]]]]:
    return {
        "P1": {
            "Walking": [
                {
                    "start_date": "2024-01-01T10:00:00.000000Z",
                    "end_date": "2024-01-01T10:10:00.000000Z",
                    "calories": 25.0,
                    "distance": 600.0,
                    "quantity": 800,
                    "tracked": 1,
                    "device": "Android Wear",
                    "source_name": "Android Wear",
                    "source_id": "com.example.watch",
                    "collecting_method_version": "1.0",
                    "outer_id": "a1",
                    "payload_index": 0,
                }
            ],
            "Sleep": [
                {
                    "start_date": "2024-01-01T22:00:00.000000Z",
                    "end_date": "2024-01-02T06:00:00.000000Z",
                    "value": "DEEP",
                    "calories": 300.0,
                    "distance": 0.0,
                    "quantity": 100,
                    "tracked": 0,
                    "device": "Android Wear",
                    "source_name": "Android Wear",
                    "source_id": "com.example.watch",
                    "collecting_method_version": "1.0",
                    "outer_id": "s1",
                    "payload_index": 0,
                },
                {
                    "start_date": "2024-01-02T05:50:00.000000Z",
                    "end_date": "2024-01-02T06:10:00.000000Z",
                    "value": "AWAKE",
                    "calories": 5.0,
                    "distance": 0.0,
                    "quantity": 1,
                    "tracked": 0,
                    "device": "Android Wear",
                    "source_name": "Android Wear",
                    "source_id": "com.example.watch",
                    "collecting_method_version": "1.0",
                    "outer_id": "s2",
                    "payload_index": 0,
                },
            ],
            "Calories": [
                {
                    "start_date": "2024-01-01T00:00:00.000000Z",
                    "end_date": "2024-01-01T23:59:59.000000Z",
                    "value": -0.0005,
                    "source_day_label": "Mon",
                    "collecting_method_version": "1.0",
                    "outer_id": "c1",
                },
                {
                    "start_date": "2024-01-02T00:00:00.000000Z",
                    "end_date": "2024-01-02T23:59:59.000000Z",
                    "value": 1700.0,
                    "source_day_label": "Tue",
                    "collecting_method_version": "2.0",
                    "outer_id": "c2",
                },
            ],
            "Distance": [
                {
                    "start_date": "2024-01-01T00:00:00.000000Z",
                    "end_date": "2024-01-01T23:59:59.000000Z",
                    "value": 600.0,
                    "source_day_label": "Mon",
                    "collecting_method_version": "1.0",
                    "outer_id": "d1",
                }
            ],
            "Weight": [
                {
                    "start_date": "2024-01-01T12:00:00.000000Z",
                    "end_date": "2024-01-01T12:05:00.000000Z",
                    "value": 75.0,
                    "source_day_label": "Mon",
                    "added_by": "com.example.scale",
                    "collecting_method_version": "2.0",
                    "outer_id": "w1",
                }
            ],
            "Height": [
                {
                    "start_date": "2024-01-01T12:00:00.000000Z",
                    "end_date": "2024-01-01T12:00:00.000000Z",
                    "value": 1.75,
                    "source_day_label": "Mon",
                    "added_by": "com.google.android.apps.fitness",
                    "collecting_method_version": "2.0",
                    "outer_id": "h1",
                }
            ],
            "StepCount": [
                {
                    "outer_datetime": "2023-12-31T00:00:00.000000Z",
                    "summary_date": None,
                    "start_date": None,
                    "end_date": None,
                    "value": 5000,
                    "raw_value": None,
                    "aggregation_source": None,
                    "origin_data_source_id": None,
                    "app_name": None,
                    "app_package_name": None,
                    "device_manufacturer": None,
                    "device_model": None,
                    "device_type": None,
                    "device_uid": None,
                    "collecting_method_version": "1.0",
                    "outer_id": "v1",
                    "payload_index": 0,
                },
                {
                    "outer_datetime": "2024-01-01T00:00:00.000000Z",
                    "summary_date": "2024-01-01",
                    "start_date": "2024-01-01T08:00:00.000000Z",
                    "end_date": "2024-01-01T20:00:00.000000Z",
                    "value": 5200,
                    "raw_value": 5200,
                    "aggregation_source": "com.google.android.gms:estimated_steps",
                    "origin_data_source_id": "raw:steps:phone",
                    "app_name": "Fit",
                    "app_package_name": "com.google.android.apps.fitness",
                    "device_manufacturer": "Example",
                    "device_model": "Phone",
                    "device_type": "Phone",
                    "device_uid": "dev1",
                    "collecting_method_version": "2.0",
                    "outer_id": "v2",
                    "payload_index": 0,
                },
                {
                    "outer_datetime": "2024-01-01T00:00:00.000000Z",
                    "summary_date": "2024-01-01",
                    "start_date": "2024-01-01T08:00:00.000000Z",
                    "end_date": "2024-01-01T20:00:00.000000Z",
                    "value": 5400,
                    "raw_value": 5400,
                    "aggregation_source": "com.google.android.gms:merge_step_deltas",
                    "origin_data_source_id": "raw:steps:phone",
                    "app_name": "Fit",
                    "app_package_name": "com.google.android.apps.fitness",
                    "device_manufacturer": "Example",
                    "device_model": "Phone",
                    "device_type": "Phone",
                    "device_uid": "dev1",
                    "collecting_method_version": "2.0",
                    "outer_id": "v2",
                    "payload_index": 1,
                },
            ],
        },
        "P2": {
            "Walking": [
                {
                    "start_date": "2024-02-01T10:00:00.000000Z",
                    "end_date": "2024-02-01T10:01:00.000000Z",
                    "calories": 2.0,
                    "distance": 50.0,
                    "quantity": 60,
                    "tracked": 1,
                    "device": "Android",
                    "source_name": "Android",
                    "source_id": "com.google.android.gms",
                    "collecting_method_version": "1.0",
                    "outer_id": "a2",
                    "payload_index": 0,
                }
            ]
        },
    }


def _build_native_root(
    root: Path, rows_by_participant: dict[str, dict[str, list[dict[str, object]]]] | None = None,
) -> Path:
    root.mkdir()
    state = StateDatabase(root / ".wearable_state.sqlite")
    try:
        for participant, by_feature in (rows_by_participant or _rows()).items():
            folder = root / participant
            folder.mkdir()
            outputs = []
            for feature, rows in by_feature.items():
                outputs.append(
                    write_feature_csv(
                        feature,
                        rows,
                        folder / f"{feature}.csv",
                        represented_occurrences=len(rows),
                    )
                )
            state.commit_participant(
                participant_id=participant,
                parser_version="test-parser",
                registry_version="test-registry",
                cleaner_version="test-cleaner",
                pipeline_version="test-pipeline",
                signature=f"sig-{participant}",
                sources=(),
                outputs=outputs,
                ledger={
                    "source_occurrences": [],
                    "malformed_candidates": [],
                    "parser_diagnostics": [],
                    "reconciliation_decisions": [],
                    "exceptional_row_audits": [],
                },
                status="complete",
            )
    finally:
        state.connection.close()
    return root


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_native_acceptance_evidence_is_snapshot_scoped() -> None:
    evidence = EVIDENCE["project_phase1_native_acceptance"]
    assert "13,476,849" not in evidence.summary
    assert "32,707" not in evidence.summary
    assert any("snapshot-specific" in item for item in evidence.limitations)
    assert "native_snapshot.json" in (evidence.artifact or "")
    malformed = EVIDENCE["project_native_malformed_weight"]
    assert "Four malformed" not in malformed.summary
    assert any("snapshot-specific" in item for item in malformed.limitations)
    assert "native_snapshot.json" in (malformed.artifact or "")


def test_cwp2_audit_is_read_only_complete_and_source_qualified(tmp_path: Path) -> None:
    native = _build_native_root(tmp_path / "native")
    before = _tree_hashes(native)
    output = tmp_path / "audit"
    result = run_audit(
        native,
        output,
        workers=2,
        max_in_flight=2,
        verify_native_hashes=True,
        progress=False,
    )
    assert result["acceptance"]["status"] == "pass"
    assert result["native_rows"] == 12
    assert result["audited_rows"] == 12
    assert _tree_hashes(native) == before
    snapshot = json.loads((output / "native_snapshot.json").read_text())
    assert snapshot["source_key"] == "google_fitness"
    assert snapshot["native_rows"] == 12
    assert snapshot["file_manifest_digest"]
    assert len(_read_csv(output / "policy_coverage.csv")) == 67


def test_stepcount_preference_and_rule_reports_use_qualified_sources(tmp_path: Path) -> None:
    native = _build_native_root(tmp_path / "native")
    output = tmp_path / "audit"
    run_audit(native, output, workers=1, progress=False)
    preference = _read_csv(output / "stepcount_preference_audit.csv")
    composition = next(row for row in preference if row["composition"].startswith("{"))
    assert json.loads(composition["composition"]) == {
        "estimated_steps": 1,
        "merge_step_deltas": 1,
    }
    assert composition["preference_outcome"] == "estimated_preferred"
    roles = {
        row["preference_outcome"]: int(row["group_count"])
        for row in preference
        if row["composition"] == "__role_total__"
    }
    assert roles["fit_app_preferred"] == 1
    assert roles["parallel_nonpreferred"] == 1
    rules = _read_csv(output / "quality_rule_trigger_audit.csv")
    triggered = {
        (row["rule_id"], row["feature"], row["collecting_method_version"]): int(row["trigger_count"])
        for row in rules
    }
    assert triggered[("step_count_estimated_preferred", "StepCount", "2.0")] == 1
    assert triggered[("step_count_parallel_nonpreferred", "StepCount", "2.0")] == 1
    assert triggered[("calories_negative_tiny", "Calories", "1.0")] == 1
    assert triggered[("weight_nonpoint_interval", "Weight", "2.0")] == 1
    assert ("native_interval_valid", "StepCount", "1.0") not in triggered

    for row in rules:
        if int(row["trigger_count"]) <= 0:
            continue
        assert row["rule_id"] in get_policy(row["feature"]).curation.rule_ids

    acceptance = json.loads((output / "audit_acceptance.json").read_text())
    alignment = next(
        item for item in acceptance["checks"]
        if item["check"] == "quality_rule_policy_alignment"
    )
    assert alignment["passed"] is True
    assert acceptance["quality_rule_policy_alignment_issues"] == []
    assert acceptance["check_count"] == 14
    assert AUDIT_VERSION == "google-fitness-curation-audit-1.0.0"
    assert REPORT_SCHEMA_VERSION == "google-fitness-curation-audit-report-1.0.0"


def test_invalid_interval_feature_still_triggers_registered_rule(tmp_path: Path) -> None:
    rows = _rows()
    rows["P1"]["Walking"][0]["end_date"] = "2024-01-01T09:59:00.000000Z"
    native = _build_native_root(tmp_path / "native", rows)
    output = tmp_path / "audit"
    run_audit(native, output, workers=1, progress=False)
    triggered = {
        (row["rule_id"], row["feature"], row["collecting_method_version"]): int(row["trigger_count"])
        for row in _read_csv(output / "quality_rule_trigger_audit.csv")
    }
    assert triggered[("native_interval_valid", "Walking", "1.0")] == 1
    assert triggered[("activity_interval_valid", "Walking", "1.0")] == 1


def test_quality_rule_alignment_helper_rejects_unregistered_rule() -> None:
    aggregate = ParticipantAuditResult("P1")
    aggregate.quality_rule_counts[("native_interval_valid", "StepCount", "1.0")] = 1
    assert audit_module._quality_rule_policy_alignment_issues(aggregate) == (
        "StepCount|1.0: observed 1 trigger(s) for unregistered rule 'native_interval_valid'",
    )


def test_acceptance_fails_when_quality_rule_policy_alignment_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = _build_native_root(tmp_path / "native")
    output = tmp_path / "audit"
    monkeypatch.setattr(
        audit_module,
        "_quality_rule_policy_alignment_issues",
        lambda aggregate: ("forced alignment failure",),
    )
    result = run_audit(native, output, workers=1, progress=False)
    assert result["acceptance"]["status"] == "fail"
    alignment = next(
        item for item in result["acceptance"]["checks"]
        if item["check"] == "quality_rule_policy_alignment"
    )
    assert alignment["passed"] is False
    assert result["acceptance"]["quality_rule_policy_alignment_issues"] == [
        "forced alignment failure"
    ]


def test_reports_are_deterministic_across_worker_counts(tmp_path: Path) -> None:
    native = _build_native_root(tmp_path / "native")
    one = tmp_path / "one"
    two = tmp_path / "two"
    run_audit(native, one, workers=1, max_in_flight=1, progress=False)
    run_audit(native, two, workers=2, max_in_flight=2, progress=False)
    excluded = {"audit_summary.json"}
    one_files = {path.name for path in one.iterdir() if path.is_file()} - excluded
    two_files = {path.name for path in two.iterdir() if path.is_file()} - excluded
    assert one_files == two_files
    for name in sorted(one_files):
        assert (one / name).read_bytes() == (two / name).read_bytes(), name


def test_empty_sleep_overlap_report_has_a_stable_header(tmp_path: Path) -> None:
    native = _build_native_root(tmp_path / "native")
    # Remove the overlapping Sleep row and update the state by rebuilding a root
    # with no Sleep file at all; the report must still remain a valid CSV.
    output = tmp_path / "audit"
    run_audit(native, output, workers=1, progress=False)
    header = (output / "sleep_overlap_audit.csv").read_text().splitlines()[0]
    assert header.startswith("overlap_scope,overlap_category,pair_count")


def test_hash_mismatch_is_isolated_and_fails_acceptance(tmp_path: Path) -> None:
    native = _build_native_root(tmp_path / "native")
    path = native / "P2" / "Walking.csv"
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    output = tmp_path / "audit"
    result = run_audit(
        native,
        output,
        workers=2,
        verify_native_hashes=True,
        progress=False,
    )
    assert result["acceptance"]["status"] == "fail"
    failures = json.loads((output / "participant_failures.json").read_text())
    assert failures and failures[0]["participant_id"] == "P2"


def test_output_must_be_outside_native_root(tmp_path: Path) -> None:
    native = _build_native_root(tmp_path / "native")
    with pytest.raises(ValueError, match="must be separate"):
        run_audit(native, native / "audit", workers=1, progress=False)


def test_phase0_localized_day_vocabulary_is_covered() -> None:
    from wearable_project.curation.google_fitness.audit import _WEEKDAY_LABELS

    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "google_fitness_phase0_day_labels.json").read_text(encoding="utf-8")
    )
    unmatched = [
        label for label in fixture["labels"]
        if label.strip().casefold() not in _WEEKDAY_LABELS
    ]
    assert unmatched == []


def test_extended_calibration_reports_cover_step_values_quantity_and_anthropometrics(tmp_path: Path) -> None:
    native = _build_native_root(tmp_path / "native")
    output = tmp_path / "audit"
    run_audit(native, output, workers=1, progress=False)

    value_raw = _read_csv(output / "stepcount_value_raw_audit.csv")
    by_source = {row["aggregation_source"]: row for row in value_raw}
    assert int(by_source["estimated_steps"]["comparison_pairs"]) == 1
    assert int(by_source["estimated_steps"]["comparison_exact_equal"]) == 1
    assert int(by_source["merge_step_deltas"]["comparison_exact_equal"]) == 1

    quantity_steps = _read_csv(output / "activity_quantity_stepcount_concordance.csv")
    walking_outer = next(
        row for row in quantity_steps
        if row["feature"] == "Walking" and row["step_count_date_basis"] == "outer_date"
    )
    assert int(walking_outer["shared_days"]) == 1
    assert int(walking_outer["comparison_pairs"]) == 1

    anthropometrics = _read_csv(output / "anthropometric_cross_feature_audit.csv")
    assert len(anthropometrics) == 1
    assert int(anthropometrics[0]["participants_with_both"]) == 1
    assert int(anthropometrics[0]["bmi_assumption_rows"]) == 1
    assert anthropometrics[0]["interpretation"].startswith("exploratory_")


def test_participant_filter_is_explicitly_partial_not_full_acceptance(tmp_path: Path) -> None:
    native = _build_native_root(tmp_path / "native")
    output = tmp_path / "audit"
    result = run_audit(
        native,
        output,
        workers=1,
        participants={"P1"},
        progress=False,
    )
    acceptance = result["acceptance"]
    assert acceptance["status"] == "partial_pass"
    assert acceptance["scope"] == "participant_subset"
    assert acceptance["full_native_coverage"] is False
    assert result["audited_rows"] == 11
    decisions = json.loads((output / "calibration_decisions.json").read_text())
    provisional = [item for item in decisions["decisions"] if item["current_status"] == "provisional"]
    assert provisional
    assert all(item["recommendation"] == "subset_evidence_only" for item in provisional)


def test_existing_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    native = _build_native_root(tmp_path / "native")
    output = tmp_path / "audit"
    run_audit(native, output, workers=1, progress=False)
    with pytest.raises(FileExistsError, match="already exists"):
        run_audit(native, output, workers=1, progress=False)
    result = run_audit(native, output, workers=1, overwrite=True, progress=False)
    assert result["acceptance"]["status"] == "pass"


def test_wrong_source_state_is_rejected_before_reading_csvs(tmp_path: Path) -> None:
    native = _build_native_root(tmp_path / "native")
    connection = __import__("sqlite3").connect(native / ".wearable_state.sqlite")
    try:
        connection.execute("UPDATE metadata SET value='apple_healthkit' WHERE key='source_key'")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ValueError, match="expected 'google_fitness'"):
        run_audit(native, tmp_path / "audit", workers=1, progress=False)


def test_audit_module_cli_writes_json_summary_and_returns_zero(tmp_path: Path, capsys) -> None:
    from wearable_project.curation.google_fitness.audit import main

    native = _build_native_root(tmp_path / "native")
    output = tmp_path / "audit"
    code = main([
        "--input-native", str(native),
        "--output", str(output),
        "--workers", "1",
        "--max-in-flight", "1",
        "--verify-native-hashes",
        "--no-progress",
        "--json-summary",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["acceptance"]["status"] == "pass"
    assert payload["audit_scope"] == "full_native_root"
    assert payload["operational"]["verify_native_hashes"] is True
