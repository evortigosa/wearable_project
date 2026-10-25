"""
Wearable Data Processing and Modeling project
Google Fitness full-cohort curation acceptance tests. These tests deliberately avoid the Apple/DataLoader
cohort-acceptance stack. The CWP6 tool validates the managed Google Fitness native and curated products from
state and physical CSV evidence. Source-aware DataLoader acceptance remains a later milestone.
"""


from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import pytest
from wearable_project.cli import main
from wearable_project.curation.google_fitness.cohort_acceptance import (
    COHORT_ACCEPTANCE_VERSION, build_expected_snapshot_from_native_root, run_cohort_acceptance,
)
from wearable_project.curation.google_fitness.engine import sha256_file
from wearable_project.curation.google_fitness.pipeline import curate_dataset
from wearable_project.exceptions import InputLayoutError
from wearable_project.processing.google_fitness.registry import get_feature_spec


def _write_feature(root: Path, participant: str, feature: str, rows: list[dict[str, object]]) -> Path:
    path = root / participant / f"{feature}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(get_feature_spec(feature).column_names)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def _managed_native_root(tmp_path: Path, participants: int = 2, empty_participants: int = 1) -> Path:
    root = tmp_path / "native"
    records: list[tuple[object, ...]] = []
    participant_rows: list[tuple[object, ...]] = []
    for index in range(participants):
        participant = f"P{index + 1}"
        calories = _write_feature(
            root,
            participant,
            "Calories",
            [{
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-02T00:00:00Z",
                "value": "1200" if index == 0 else "-0.0005",
                "source_day_label": "Monday",
                "collecting_method_version": "2.0",
                "outer_id": f"cal-{index}",
            }],
        )
        step = _write_feature(
            root,
            participant,
            "StepCount",
            [
                {
                    "outer_datetime": "2024-01-02T00:00:00Z",
                    "summary_date": "2024-01-01",
                    "start_date": "2024-01-01T00:00:00Z",
                    "end_date": "2024-01-02T00:00:00Z",
                    "value": "1000",
                    "raw_value": "1000",
                    "aggregation_source": "estimated_steps",
                    "origin_data_source_id": "origin",
                    "app_name": "Google Fit",
                    "app_package_name": "com.google.android.apps.fitness",
                    "device_manufacturer": "Google",
                    "device_model": "Pixel",
                    "device_type": "phone",
                    "device_uid": f"uid-{index}",
                    "collecting_method_version": "2.0",
                    "outer_id": f"step-{index}",
                    "payload_index": "0",
                },
                {
                    "outer_datetime": "2024-01-02T00:00:00Z",
                    "summary_date": "2024-01-01",
                    "start_date": "2024-01-01T00:00:00Z",
                    "end_date": "2024-01-02T00:00:00Z",
                    "value": "900",
                    "raw_value": "900",
                    "aggregation_source": "merge_step_deltas",
                    "origin_data_source_id": "origin",
                    "app_name": "Google Fit",
                    "app_package_name": "com.google.android.apps.fitness",
                    "device_manufacturer": "Google",
                    "device_model": "Pixel",
                    "device_type": "phone",
                    "device_uid": f"uid-{index}",
                    "collecting_method_version": "2.0",
                    "outer_id": f"step-{index}",
                    "payload_index": "1",
                },
            ],
        )
        for feature, path, count in (("Calories", calories, 1), ("StepCount", step, 2)):
            records.append((participant, feature, path.name, count, path.stat().st_size, sha256_file(path), count))
        participant_rows.append((participant, "complete"))
    for index in range(empty_participants):
        participant_rows.append((f"E{index + 1}", "complete_empty"))

    state = root / ".wearable_state.sqlite"
    connection = sqlite3.connect(state)
    connection.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE participants(participant_id TEXT PRIMARY KEY,status TEXT NOT NULL);
        CREATE TABLE feature_outputs(
          participant_id TEXT NOT NULL,
          feature TEXT NOT NULL,
          filename TEXT NOT NULL,
          row_count INTEGER NOT NULL,
          size_bytes INTEGER NOT NULL,
          sha256 TEXT NOT NULL,
          represented_occurrences INTEGER NOT NULL,
          PRIMARY KEY(participant_id,feature)
        );
        CREATE TABLE runs(run_id TEXT,status TEXT);
        CREATE TABLE source_occurrences(id INTEGER);
        CREATE TABLE malformed_candidates(
          source_feature TEXT,
          collecting_method_version TEXT,
          reason_codes_json TEXT
        );
        CREATE TABLE parser_diagnostics(code TEXT);
        CREATE TABLE reconciliation_decisions(id INTEGER);
        """
    )
    connection.executemany(
        "INSERT INTO metadata VALUES(?,?)",
        (
            ("source_key", "google_fitness"),
            ("state_schema_version", "google-fitness-native-state-wp4-0.1.0"),
            ("registry_version", "2026-10-google-fitness-native-wp1"),
        ),
    )
    connection.executemany("INSERT INTO participants VALUES(?,?)", participant_rows)
    connection.executemany("INSERT INTO feature_outputs VALUES(?,?,?,?,?,?,?)", records)
    connection.commit()
    connection.close()
    return root


def _expected_manifest(native: Path, tmp_path: Path) -> Path:
    payload = build_expected_snapshot_from_native_root("synthetic-cwp6", native)
    path = tmp_path / "expected_snapshot.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _prepare(
    tmp_path: Path, *, no_op: bool = True, no_op_verify_existing: bool = True,
) -> tuple[Path, Path, Path]:
    native = _managed_native_root(tmp_path)
    curated = tmp_path / "curated"
    curate_dataset(native, curated, workers=1, verify_native_hashes=True)
    if no_op:
        curate_dataset(
            native,
            curated,
            workers=1,
            verify_native_hashes=True,
            verify_existing_hashes=no_op_verify_existing,
        )
    return native, curated, _expected_manifest(native, tmp_path)


def _scientific_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }


def test_cohort_acceptance_passes_with_build_and_hash_verified_noop(tmp_path: Path) -> None:
    native, curated, expected = _prepare(tmp_path)
    report = tmp_path / "cohort_acceptance"
    result = run_cohort_acceptance(
        native,
        curated,
        report,
        expected_snapshot=expected,
        workers=1,
        no_progress=True,
    )
    assert result["status"] == "pass"
    assert result["cohort_acceptance_version"] == COHORT_ACCEPTANCE_VERSION
    assert result["failed_check_count"] == 0
    assert result["native_snapshot"]["participant_status_counts"] == {
        "complete": 2,
        "complete_empty": 1,
    }
    assert result["product_acceptance"]["snapshot"]["participants"] == 2
    assert result["run_evidence"]["latest_run"]["participants"] == {
        "discovered": 2,
        "skipped": 2,
        "committed": 0,
        "failed": 0,
    }
    assert (report / "product_acceptance" / "cwp5_acceptance.json").is_file()
    assert (report / "cohort_release_manifest.json").is_file()
    assert (report / "curation_run_history.csv").is_file()


def test_missing_noop_run_fails_formal_acceptance(tmp_path: Path) -> None:
    native, curated, expected = _prepare(tmp_path, no_op=False)
    result = run_cohort_acceptance(
        native, curated, tmp_path / "report", expected_snapshot=expected,
        workers=1, no_progress=True,
    )
    assert result["status"] == "fail"
    failed = {item["check_id"] for item in result["checks"] if item["status"] == "fail"}
    assert "latest_run_hash_verified_noop" in failed
    assert "noop_follows_build" in failed


def test_noop_without_existing_hash_verification_fails(tmp_path: Path) -> None:
    native, curated, expected = _prepare(tmp_path, no_op_verify_existing=False)
    result = run_cohort_acceptance(
        native, curated, tmp_path / "report", expected_snapshot=expected, workers=1, no_progress=True,
    )
    assert result["status"] == "fail"
    check = next(item for item in result["checks"] if item["check_id"] == "latest_run_hash_verified_noop")
    assert check["status"] == "fail"


def test_expected_snapshot_mismatch_fails(tmp_path: Path) -> None:
    native, curated, expected = _prepare(tmp_path)
    payload = json.loads(expected.read_text())
    payload["expected"]["native_snapshot"]["native_rows"] += 1
    expected.write_text(json.dumps(payload))
    result = run_cohort_acceptance(
        native, curated, tmp_path / "report", expected_snapshot=expected, workers=1, no_progress=True,
    )
    assert result["status"] == "fail"
    assert result["expected_snapshot"]["mismatch_count"] == 1


def test_noop_participant_history_must_cover_every_current_participant(tmp_path: Path) -> None:
    native, curated, expected = _prepare(tmp_path)
    state = curated / ".wearable_curation_state.sqlite"
    connection = sqlite3.connect(state)
    latest = connection.execute(
        "SELECT run_id FROM curation_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()[0]
    connection.execute(
        "DELETE FROM curation_run_participants WHERE run_id=? AND participant_id='P2'",
        (latest,),
    )
    connection.commit()
    connection.close()
    result = run_cohort_acceptance(
        native, curated, tmp_path / "report", expected_snapshot=expected, workers=1, no_progress=True,
    )
    assert result["status"] == "fail"
    check = next(item for item in result["checks"] if item["check_id"] == "latest_run_hash_verified_noop")
    assert check["status"] == "fail"


def test_scientific_reports_are_deterministic_across_worker_counts(tmp_path: Path) -> None:
    native, curated, expected = _prepare(tmp_path)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    run_cohort_acceptance(
        native, curated, serial, expected_snapshot=expected, workers=1, no_progress=True,
    )
    run_cohort_acceptance(
        native, curated, parallel, expected_snapshot=expected, workers=2, max_in_flight=2, no_progress=True,
    )
    assert _scientific_hashes(serial) == _scientific_hashes(parallel)


def test_cohort_acceptance_is_read_only_for_both_roots(tmp_path: Path) -> None:
    native, curated, expected = _prepare(tmp_path)

    def hashes(root: Path) -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file() and not path.name.endswith(("-wal", "-shm"))
        }

    native_before = hashes(native)
    curated_before = hashes(curated)
    run_cohort_acceptance(
        native, curated, tmp_path / "report", expected_snapshot=expected, workers=1, no_progress=True,
    )
    assert hashes(native) == native_before
    assert hashes(curated) == curated_before


def test_output_must_be_outside_both_products(tmp_path: Path) -> None:
    native, curated, expected = _prepare(tmp_path)
    with pytest.raises(InputLayoutError, match="outside both data roots"):
        run_cohort_acceptance(
            native, curated, curated / "reports", expected_snapshot=expected, workers=1, no_progress=True,
        )


def test_cli_cohort_acceptance(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    native, curated, expected = _prepare(tmp_path)
    report = tmp_path / "report"
    code = main([
        "curation-cohort-acceptance", "google-fitness",
        "--input-native", str(native),
        "--input-curated", str(curated),
        "--output", str(report),
        "--expected-snapshot", str(expected),
        "--workers", "1",
        "--no-progress",
        "--json-summary",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass"


def test_checked_in_full_cohort_expected_snapshot_is_snapshot_specific() -> None:
    fixture = Path(__file__).parent / "fixtures" / "google_fitness_cwp6_expected_native_snapshot_2024_12.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    assert payload["expected_snapshot_schema_version"] == (
        "google-fitness-curation-expected-native-snapshot-0.1.0"
    )
    assert payload["reference_id"] == "exported-at-2024-12-google-fitness-native-cwp6"
    expected = payload["expected"]["native_snapshot"]
    assert expected["participant_records"] == 9_798
    assert expected["complete_participants"] == 5_983
    assert expected["complete_empty_participants"] == 3_815
    assert expected["feature_files"] == 32_707
    assert expected["native_rows"] == 13_476_849
    assert expected["source_occurrences"] == 7_315_025
    assert expected["malformed_candidates"] == 4
    assert expected["feature_types"] == 67
    assert expected["activities_feature_present"] is False


def test_apple_source_is_not_advertised_for_google_cohort_acceptance() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["curation-cohort-acceptance", "apple-healthkit"])
    assert caught.value.code == 2


def test_archived_or_relocated_products_remain_acceptable(tmp_path: Path) -> None:
    native, curated, expected = _prepare(tmp_path)
    moved_native = tmp_path / "moved_native"
    moved_curated = tmp_path / "moved_curated"
    native.rename(moved_native)
    curated.rename(moved_curated)
    result = run_cohort_acceptance(
        moved_native,
        moved_curated,
        tmp_path / "report",
        expected_snapshot=expected,
        workers=1,
        no_progress=True,
    )
    assert result["status"] == "pass"


def test_expected_snapshot_requires_reference_id(tmp_path: Path) -> None:
    native, curated, expected = _prepare(tmp_path)
    payload = json.loads(expected.read_text())
    payload.pop("reference_id")
    expected.write_text(json.dumps(payload))
    result = run_cohort_acceptance(
        native,
        curated,
        tmp_path / "report",
        expected_snapshot=expected,
        workers=1,
        no_progress=True,
    )
    assert result["status"] == "fail"
    check = next(item for item in result["checks"] if item["check_id"] == "expected_snapshot_reference_id")
    assert check["status"] == "fail"


def test_existing_report_requires_explicit_overwrite(tmp_path: Path) -> None:
    native, curated, expected = _prepare(tmp_path)
    report = tmp_path / "report"
    run_cohort_acceptance(
        native, curated, report, expected_snapshot=expected, workers=1, no_progress=True,
    )
    with pytest.raises(InputLayoutError, match="--overwrite"):
        run_cohort_acceptance(
            native, curated, report, expected_snapshot=expected, workers=1, no_progress=True,
        )
    result = run_cohort_acceptance(
        native, curated, report, expected_snapshot=expected, workers=1, overwrite=True, no_progress=True,
    )
    assert result["status"] == "pass"


def test_invalid_expected_snapshot_schema_fails(tmp_path: Path) -> None:
    native, curated, expected = _prepare(tmp_path)
    payload = json.loads(expected.read_text())
    payload["expected_snapshot_schema_version"] = "wrong"
    expected.write_text(json.dumps(payload))
    result = run_cohort_acceptance(
        native, curated, tmp_path / "report", expected_snapshot=expected, workers=1, no_progress=True,
    )
    assert result["status"] == "fail"
    assert result["expected_snapshot"]["mismatches"][0]["path"] == (
        "expected_snapshot_schema_version"
    )


def test_missing_expected_snapshot_is_rejected(tmp_path: Path) -> None:
    native, curated, _ = _prepare(tmp_path)
    with pytest.raises(InputLayoutError, match="does not exist"):
        run_cohort_acceptance(
            native,
            curated,
            tmp_path / "report",
            expected_snapshot=tmp_path / "missing.json",
            workers=1,
            no_progress=True,
        )


def test_non_google_native_state_is_rejected(tmp_path: Path) -> None:
    native, curated, expected = _prepare(tmp_path)
    state = native / ".wearable_state.sqlite"
    connection = sqlite3.connect(state)
    connection.execute("UPDATE metadata SET value='apple_healthkit' WHERE key='source_key'")
    connection.commit()
    connection.close()
    with pytest.raises(InputLayoutError, match="native state source"):
        run_cohort_acceptance(
            native, curated, tmp_path / "report", expected_snapshot=expected, workers=1, no_progress=True,
        )


def test_active_native_wal_is_rejected(tmp_path: Path) -> None:
    native, curated, expected = _prepare(tmp_path)
    wal = native / ".wearable_state.sqlite-wal"
    wal.write_bytes(b"active")
    with pytest.raises(InputLayoutError, match="active WAL"):
        run_cohort_acceptance(
            native, curated, tmp_path / "report", expected_snapshot=expected, workers=1, no_progress=True,
        )


def test_module_main_prints_json_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from wearable_project.curation.google_fitness.cohort_acceptance import main as cohort_main

    native, curated, expected = _prepare(tmp_path)
    code = cohort_main([
        "--input-native", str(native),
        "--input-curated", str(curated),
        "--output", str(tmp_path / "report"),
        "--expected-snapshot", str(expected),
        "--workers", "1",
        "--no-progress",
        "--json-summary",
    ])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
