"""
Wearable Data Processing and Modeling project
Transactional Google Fitness curation pipeline and state tests (CWP4).
"""


from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import shutil
import pytest
from wearable_project.curation.google_fitness.engine import sha256_file
from wearable_project.curation.google_fitness.pipeline import curate_dataset
from wearable_project.curation.google_fitness.state import (
    CURATION_STATE_FILENAME, CurationStateDatabase, load_curation_report,
)
from wearable_project.exceptions import InputLayoutError, OutputValidationError
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
        participant = f"P{index+1}"
        calories = _write_feature(
            root,
            participant,
            "Calories",
            [
                {
                    "start_date": "2024-01-01T00:00:00Z",
                    "end_date": "2024-01-02T00:00:00Z",
                    "value": "1200" if index == 0 else "-0.0005",
                    "source_day_label": "Monday",
                    "collecting_method_version": "2.0",
                    "outer_id": f"cal-{index}",
                }
            ],
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
            records.append(
                (
                    participant,
                    feature,
                    path.name,
                    count,
                    path.stat().st_size,
                    sha256_file(path),
                    count,
                )
            )

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


def _csv_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.csv"))
    }


def test_transactional_pipeline_curates_and_noops(tmp_path: Path) -> None:
    native = _managed_native_root(tmp_path)
    output = tmp_path / "curated"
    summary = curate_dataset(
        native,
        output,
        workers=2,
        max_in_flight=2,
        verify_native_hashes=True,
    )
    report = summary.as_dict()
    assert report["participants"] == {
        "discovered": 2,
        "committed": 2,
        "skipped": 0,
        "failed": 0,
    }
    assert report["snapshot"]["feature_files"] == 4
    assert report["snapshot"]["native_rows"] == 6
    assert report["snapshot"]["curated_rows"] == 6
    assert report["snapshot"]["status_accounting_complete"] is True
    assert report["snapshot"]["default_inclusion_accounting_complete"] is True
    assert (output / CURATION_STATE_FILENAME).is_file()

    first_hashes = _csv_hashes(output)
    noop = curate_dataset(
        native,
        output,
        workers=1,
        verify_native_hashes=True,
        verify_existing_hashes=True,
    ).as_dict()
    assert noop["participants"] == {
        "discovered": 2,
        "committed": 0,
        "skipped": 2,
        "failed": 0,
    }
    assert noop["run_activity"]["native_files_read"] == 0
    assert _csv_hashes(output) == first_hashes


def test_worker_count_does_not_change_curated_bytes(tmp_path: Path) -> None:
    native = _managed_native_root(tmp_path)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    curate_dataset(native, serial, workers=1, verify_native_hashes=True)
    curate_dataset(native, parallel, workers=2, max_in_flight=2, verify_native_hashes=True)
    assert _csv_hashes(serial) == _csv_hashes(parallel)


def test_corrupt_output_rebuilds_with_hash_verification(tmp_path: Path) -> None:
    native = _managed_native_root(tmp_path, participants=1)
    output = tmp_path / "curated"
    curate_dataset(native, output, workers=1)
    target = output / "P1" / "Calories.csv"
    data = bytearray(target.read_bytes())
    data[-2] = ord("9") if data[-2] != ord("9") else ord("8")
    target.write_bytes(data)
    rebuilt = curate_dataset(
        native,
        output,
        workers=1,
        verify_existing_hashes=True,
    ).as_dict()
    assert rebuilt["participants"]["committed"] == 1
    assert rebuilt["participants"]["skipped"] == 0


def test_native_source_guard_rejects_wrong_source(tmp_path: Path) -> None:
    native = _managed_native_root(tmp_path, participants=1)
    state = native / ".wearable_state.sqlite"
    connection = sqlite3.connect(state)
    connection.execute("UPDATE metadata SET value='apple_healthkit' WHERE key='source_key'")
    connection.commit()
    connection.close()
    with pytest.raises(InputLayoutError, match="native state source"):
        curate_dataset(native, tmp_path / "curated", workers=1)


def test_native_hash_verification_detects_same_size_mutation(tmp_path: Path) -> None:
    native = _managed_native_root(tmp_path, participants=1)
    target = native / "P1" / "Calories.csv"
    data = bytearray(target.read_bytes())
    position = data.index(b"1200")
    data[position : position + 4] = b"1300"
    target.write_bytes(data)
    with pytest.raises(OutputValidationError, match="hash differs"):
        curate_dataset(
            native,
            tmp_path / "curated",
            workers=1,
            verify_native_hashes=True,
        )


def test_state_source_guard_rejects_foreign_curated_state(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO metadata VALUES(?,?)",
        (("source_key", "apple_healthkit"), ("product_kind", "curated"), ("state_schema_version", "x")),
    )
    connection.commit()
    connection.close()
    with pytest.raises(OutputValidationError, match="managed by source"):
        CurationStateDatabase(path)


def test_persisted_report_round_trip(tmp_path: Path) -> None:
    native = _managed_native_root(tmp_path, participants=1)
    output = tmp_path / "curated"
    summary = curate_dataset(native, output, workers=1)
    report = load_curation_report(output / CURATION_STATE_FILENAME, summary.run_id)
    assert report["run_id"] == summary.run_id
    assert report["snapshot"]["curated_rows"] == 3
    assert report["fingerprints"]["engine"]


def test_one_participant_failure_does_not_prevent_other_commit(tmp_path: Path) -> None:
    native = _managed_native_root(tmp_path, participants=2)
    broken = native / "P2" / "Calories.csv"
    lines = broken.read_text(encoding="utf-8").splitlines()
    lines[0] = "wrong,header"
    broken.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Keep state size/hash aligned so failure occurs in the worker's schema validation.
    connection = sqlite3.connect(native / ".wearable_state.sqlite")
    connection.execute(
        "UPDATE feature_outputs SET size_bytes=?,sha256=? WHERE participant_id='P2' AND feature='Calories'",
        (broken.stat().st_size, sha256_file(broken)),
    )
    connection.commit()
    connection.close()

    report = curate_dataset(native, tmp_path / "curated", workers=2).as_dict()
    assert report["participants"]["committed"] == 1
    assert report["participants"]["failed"] == 1
    assert (tmp_path / "curated" / "P1").is_dir()
    assert not (tmp_path / "curated" / "P2").exists()


def test_state_commit_failure_rolls_back_published_directory(tmp_path: Path, monkeypatch) -> None:
    native = _managed_native_root(tmp_path, participants=1)
    output = tmp_path / "curated"

    def fail_commit(*args, **kwargs):
        raise RuntimeError("injected state commit failure")

    monkeypatch.setattr(CurationStateDatabase, "commit_participant", fail_commit)
    report = curate_dataset(native, output, workers=1).as_dict()
    assert report["participants"]["failed"] == 1
    assert not (output / "P1").exists()


def test_pipeline_does_not_leave_empty_temporary_root(tmp_path: Path) -> None:
    native = _managed_native_root(tmp_path, participants=1)
    output = tmp_path / "curated"
    curate_dataset(native, output, workers=1)
    temporary = output / ".wearable_curation_tmp"
    assert not temporary.exists() or not any(temporary.iterdir())


def test_failed_rebuild_preserves_previous_committed_product(tmp_path: Path, monkeypatch) -> None:
    native = _managed_native_root(tmp_path, participants=1)
    output = tmp_path / "curated"
    curate_dataset(native, output, workers=1)
    before = _csv_hashes(output)

    def fail_commit(*args, **kwargs):
        raise RuntimeError("injected rebuild commit failure")

    monkeypatch.setattr(CurationStateDatabase, "commit_participant", fail_commit)
    report = curate_dataset(native, output, workers=1, mode="rebuild").as_dict()
    assert report["participants"]["failed"] == 1
    assert _csv_hashes(output) == before
    with CurationStateDatabase(output / CURATION_STATE_FILENAME, read_only=True) as state:
        stored = state.get_participant("P1")
        assert stored is not None
        assert stored.status == "complete"
        assert stored.last_error == "RuntimeError: injected rebuild commit failure"


def test_stale_stage_directory_is_removed_before_planning(tmp_path: Path) -> None:
    native = _managed_native_root(tmp_path, participants=1)
    output = tmp_path / "curated"
    curate_dataset(native, output, workers=1)
    stale = output / ".wearable_curation_tmp" / "dead-run" / "P1"
    stale.mkdir(parents=True)
    (stale / "partial.csv").write_text("partial", encoding="utf-8")

    report = curate_dataset(
        native,
        output,
        workers=1,
        verify_existing_hashes=True,
    ).as_dict()
    assert report["participants"]["skipped"] == 1
    assert report["participants"]["committed"] == 0
    assert not (output / ".wearable_curation_tmp").exists()


def test_stale_backup_is_restored_before_noop_planning(tmp_path: Path) -> None:
    native = _managed_native_root(tmp_path, participants=1)
    output = tmp_path / "curated"
    curate_dataset(native, output, workers=1)
    before = _csv_hashes(output)

    run_dir = output / ".wearable_curation_tmp" / "dead-run"
    backup = run_dir / ".P1.backup"
    run_dir.mkdir(parents=True)
    shutil.copytree(output / "P1", backup)
    target = output / "P1" / "Calories.csv"
    data = bytearray(target.read_bytes())
    data[-2] = ord("9") if data[-2] != ord("9") else ord("8")
    target.write_bytes(data)

    report = curate_dataset(
        native,
        output,
        workers=1,
        verify_existing_hashes=True,
    ).as_dict()
    assert report["participants"]["skipped"] == 1
    assert report["participants"]["committed"] == 0
    assert report["configuration"]["stale_transaction_recovery"] == ["restored_backup:P1"]
    assert _csv_hashes(output) == before
    assert not (output / ".wearable_curation_tmp").exists()


def test_stale_in_progress_run_is_marked_interrupted(tmp_path: Path) -> None:
    native = _managed_native_root(tmp_path, participants=1)
    output = tmp_path / "curated"
    curate_dataset(native, output, workers=1)
    state_path = output / CURATION_STATE_FILENAME
    with CurationStateDatabase(state_path) as state:
        stale_run_id = state.begin_run(native, output, {"injected": True})

    report = curate_dataset(
        native,
        output,
        workers=1,
        verify_existing_hashes=True,
    ).as_dict()
    assert report["configuration"]["interrupted_runs_recovered"] == [stale_run_id]
    with sqlite3.connect(state_path) as connection:
        status, finished_at = connection.execute(
            "SELECT status,finished_at FROM curation_runs WHERE run_id=?",
            (stale_run_id,),
        ).fetchone()
    assert status == "interrupted"
    assert finished_at
