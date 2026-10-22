"""
Wearable Data Processing and Modeling project
Formal Google Fitness native-to-curated acceptance tests.
"""


from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import pytest
from wearable_project.cli import main
from wearable_project.curation.google_fitness.acceptance import (
    ACCEPTANCE_VERSION, build_reference_manifest, run_acceptance,
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


def _managed_native_root(tmp_path: Path, participants: int = 2) -> Path:
    root = tmp_path / "native"
    records: list[tuple[object, ...]] = []
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

    state = root / ".wearable_state.sqlite"
    connection = sqlite3.connect(state)
    connection.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
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
        CREATE TABLE malformed_candidates(id INTEGER);
        CREATE TABLE parser_diagnostics(id INTEGER);
        CREATE TABLE reconciliation_decisions(id INTEGER);
        """
    )
    connection.execute("INSERT INTO metadata VALUES('source_key','google_fitness')")
    connection.executemany("INSERT INTO feature_outputs VALUES(?,?,?,?,?,?,?)", records)
    connection.commit()
    connection.close()
    return root


def _prepare(tmp_path: Path, participants: int = 2) -> tuple[Path, Path]:
    native = _managed_native_root(tmp_path, participants)
    curated = tmp_path / "curated"
    curate_dataset(native, curated, workers=1, verify_native_hashes=True)
    return native, curated


def _scientific_hashes(root: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.name != "run_manifest.json"
    }


def test_acceptance_passes_and_recomputes_physical_accounting(tmp_path: Path) -> None:
    native, curated = _prepare(tmp_path)
    report = tmp_path / "acceptance"
    result = run_acceptance(native, curated, report, workers=1, no_progress=True)
    assert result["status"] == "pass"
    assert result["acceptance_version"] == ACCEPTANCE_VERSION
    assert result["failed_check_count"] == 0
    assert result["snapshot"]["feature_files"] == 4
    assert result["snapshot"]["native_rows"] == 6
    assert result["snapshot"]["curated_rows"] == 6
    assert result["planner_action_counts"] == {"skip": 2}
    assert result["snapshot"]["status_accounting_complete"] is True
    assert result["snapshot"]["default_inclusion_accounting_complete"] is True
    assert (report / "file_manifest.csv").is_file()
    assert (report / "REPORT.md").read_text().startswith("# Google Fitness CWP5")


def test_reference_manifest_freezes_exact_sample_outputs(tmp_path: Path) -> None:
    native, curated = _prepare(tmp_path)
    first = tmp_path / "first"
    run_acceptance(native, curated, first, workers=1, no_progress=True)
    reference_view = json.loads((first / "reference_view.json").read_text())
    reference = build_reference_manifest("synthetic-cwp5", reference_view)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n")
    second = tmp_path / "second"
    result = run_acceptance(
        native, curated, second, workers=2, max_in_flight=2,
        reference_manifest=reference_path, no_progress=True,
    )
    assert result["status"] == "pass"
    assert result["reference"] == {
        "requested": True,
        "reference_id": "synthetic-cwp5",
        "mismatch_count": 0,
        "mismatches": [],
    }


def test_worker_count_does_not_change_scientific_reports(tmp_path: Path) -> None:
    native, curated = _prepare(tmp_path)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    run_acceptance(native, curated, serial, workers=1, no_progress=True)
    run_acceptance(native, curated, parallel, workers=2, max_in_flight=2, no_progress=True)
    assert _scientific_hashes(serial) == _scientific_hashes(parallel)


def test_same_size_curated_mutation_fails_hash_and_prefix_checks(tmp_path: Path) -> None:
    native, curated = _prepare(tmp_path, participants=1)
    target = curated / "P1" / "Calories.csv"
    data = bytearray(target.read_bytes())
    position = data.index(b"1200")
    data[position:position + 4] = b"1300"
    target.write_bytes(data)
    result = run_acceptance(native, curated, tmp_path / "acceptance", workers=1, no_progress=True)
    assert result["status"] == "fail"
    failed = {item["check_id"] for item in result["checks"] if item["status"] == "fail"}
    assert "physical_file_validation" in failed
    assert "current_engine_and_policies" in failed


def test_reference_mismatch_fails_acceptance(tmp_path: Path) -> None:
    native, curated = _prepare(tmp_path, participants=1)
    first = tmp_path / "first"
    run_acceptance(native, curated, first, workers=1, no_progress=True)
    reference_view = json.loads((first / "reference_view.json").read_text())
    reference = build_reference_manifest("wrong", reference_view)
    reference["expected"]["snapshot"]["curated_rows"] += 1
    reference_path = tmp_path / "wrong.json"
    reference_path.write_text(json.dumps(reference))
    result = run_acceptance(
        native, curated, tmp_path / "second", workers=1,
        reference_manifest=reference_path, no_progress=True,
    )
    assert result["status"] == "fail"
    assert result["reference"]["mismatch_count"] == 1


def test_hash_verification_is_required_for_formal_pass(tmp_path: Path) -> None:
    native, curated = _prepare(tmp_path, participants=1)
    result = run_acceptance(
        native, curated, tmp_path / "acceptance", workers=1,
        verify_hashes=False, no_progress=True,
    )
    assert result["status"] == "fail"
    check = next(item for item in result["checks"] if item["check_id"] == "hash_verification")
    assert check["status"] == "fail"


def test_output_must_be_outside_both_data_roots(tmp_path: Path) -> None:
    native, curated = _prepare(tmp_path, participants=1)
    with pytest.raises(InputLayoutError, match="outside both data roots"):
        run_acceptance(native, curated, curated / "reports", workers=1, no_progress=True)


def test_existing_report_requires_overwrite(tmp_path: Path) -> None:
    native, curated = _prepare(tmp_path, participants=1)
    report = tmp_path / "acceptance"
    run_acceptance(native, curated, report, workers=1, no_progress=True)
    with pytest.raises(InputLayoutError, match="--overwrite"):
        run_acceptance(native, curated, report, workers=1, no_progress=True)
    result = run_acceptance(native, curated, report, workers=1, overwrite=True, no_progress=True)
    assert result["status"] == "pass"


def test_cli_acceptance_command(tmp_path: Path, capsys) -> None:
    native, curated = _prepare(tmp_path, participants=1)
    report = tmp_path / "acceptance"
    code = main([
        "curation-acceptance", "google-fitness",
        "--input-native", str(native),
        "--input-curated", str(curated),
        "--output", str(report),
        "--workers", "1",
        "--no-progress",
        "--json-summary",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass"


def test_apple_source_is_not_advertised_for_google_acceptance() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["curation-acceptance", "apple-healthkit"])
    assert caught.value.code == 2


def test_state_counter_mismatch_is_detected(tmp_path: Path) -> None:
    native, curated = _prepare(tmp_path, participants=1)
    state = curated / ".wearable_curation_state.sqlite"
    connection = sqlite3.connect(state)
    connection.execute(
        "UPDATE curation_outputs SET pass_rows=pass_rows+1 WHERE participant_id='P1' AND feature='Calories'"
    )
    connection.commit()
    connection.close()
    result = run_acceptance(native, curated, tmp_path / "acceptance", workers=1, no_progress=True)
    assert result["status"] == "fail"
    failed = {item["check_id"] for item in result["checks"] if item["status"] == "fail"}
    assert "physical_file_validation" in failed
    assert "physical_state_accounting_match" in failed


def test_unrecorded_curated_file_is_detected(tmp_path: Path) -> None:
    native, curated = _prepare(tmp_path, participants=1)
    extra = curated / "P1" / "Extra.csv"
    extra.write_text("x\n1\n", encoding="utf-8")
    result = run_acceptance(native, curated, tmp_path / "acceptance", workers=1, no_progress=True)
    assert result["status"] == "fail"
    check = next(item for item in result["checks"] if item["check_id"] == "disk_file_sets_match_state")
    assert check["status"] == "fail"


def test_acceptance_is_read_only_for_both_roots(tmp_path: Path) -> None:
    native, curated = _prepare(tmp_path, participants=1)

    def hashes(root: Path) -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and not path.name.endswith(("-wal", "-shm"))
        }

    native_before = hashes(native)
    curated_before = hashes(curated)
    run_acceptance(native, curated, tmp_path / "acceptance", workers=1, no_progress=True)
    assert hashes(native) == native_before
    assert hashes(curated) == curated_before


def test_checked_in_reference_sample_manifest_matches_current_contract() -> None:
    fixture = Path(__file__).parent / "fixtures" / "google_fitness_cwp5_reference_sample_manifest.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    expected = payload["expected"]
    assert payload["reference_schema_version"] == "google-fitness-curation-reference-sample-0.1.0"
    assert payload["reference_id"] == "nonapple-exported-at-2024-12-three-participant-cwp5"
    assert expected["source_key"] == "google_fitness"
    assert expected["participants"] == ["1814073142", "3919706846", "8790614327"]
    assert expected["snapshot"]["participants"] == 3
    assert expected["snapshot"]["feature_files"] == 37
    assert expected["snapshot"]["native_rows"] == 145_493
    assert expected["snapshot"]["curated_rows"] == 145_493
    assert len(expected["native_file_sha256"]) == 37
    assert len(expected["curated_file_sha256"]) == 37
    assert expected["fingerprints"]["engine"] == "google-fitness-curation-engine-cwp3-0.1.0"
    assert expected["fingerprints"]["pipeline"] == "google-fitness-curation-pipeline-cwp4-0.1.0"
