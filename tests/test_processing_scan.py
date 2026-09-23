"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sqlite3
from pathlib import Path
from wearable_project.processing.scan import scan_processing_plan


def write_month(root: Path, participant: str, filename: str, *, value: float = 100,) -> None:
    directory = root / participant
    directory.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "participant_id", "data_source", "name", "datetime", "data",
        "collecting_method_version", "created_at", "updated_at",
    ]
    month = filename.rsplit(".", 1)[0]
    year, raw_month = month.split("-")
    month_number = int(raw_month)
    start = f"{int(year):04d}-{month_number:02d}-01T10:00:00+0200"
    end = f"{int(year):04d}-{month_number:02d}-01T11:00:00+0200"
    item = {
        "id": f"event-{filename}", "value": value,
        "start_date": start, "end_date": end,
    }
    with (directory / filename).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "id": f"outer-{filename}", "participant_id": participant,
            "data_source": "AppleHealthkit", "name": "StepCount",
            "datetime": f"{int(year):04d}-{month_number:02d}-01 00:00:00",
            "data": json.dumps(repr([item])),
            "collecting_method_version": "2.0",
            "created_at": "2025-02-01T00:00:00Z",
            "updated_at": "2025-02-01T00:00:00Z",
        })


def run_process(input_root: Path, output_root: Path) -> None:
    command = [
        sys.executable, "-m", "wearable_project", "process",
        "--input", str(input_root), "--output", str(output_root),
        "--workers", "1", "--max-in-flight", "1", "--json-summary",
    ]
    result = subprocess.run(
        command, cwd=Path(__file__).parents[1], text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fresh_scan_reports_rebuild_without_creating_output(tmp_path: Path) -> None:
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    write_month(input_root, "P1", "2024-12.csv")
    report = scan_processing_plan(input_root, output_root, workers=1, show_progress=False,)
    assert report.participant_counts["rebuild"] == 1
    assert report.participant_counts["skip"] == 0
    assert report.source_file_counts["to_parse"] == 1
    assert report.processing_needed is True
    assert report.safe_to_process is True
    assert report.no_op is False
    assert not output_root.exists()


def test_scan_noop_is_read_only(tmp_path: Path) -> None:
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    write_month(input_root, "P1", "2024-12.csv")
    run_process(input_root, output_root)
    state = output_root / ".wearable_state.sqlite"
    feature = output_root / "P1" / "StepCount.csv"
    before_state = digest(state)
    before_feature = digest(feature)
    before_state_mtime = state.stat().st_mtime_ns
    report = scan_processing_plan(input_root, output_root, workers=2, show_progress=False,)

    assert report.participant_counts["skip"] == 1
    assert report.participant_counts["incremental"] == 0
    assert report.participant_counts["rebuild"] == 0
    assert report.source_file_counts["to_parse"] == 0
    assert report.source_byte_counts["to_parse"] == 0
    assert report.processing_needed is False
    assert report.no_op is True
    assert digest(state) == before_state
    assert digest(feature) == before_feature
    assert state.stat().st_mtime_ns == before_state_mtime


def test_scan_reports_later_month_incremental(tmp_path: Path) -> None:
    first, second, output = tmp_path / "first", tmp_path / "second", tmp_path / "output"
    write_month(first, "P1", "2024-12.csv")
    shutil.copytree(first, second)
    write_month(second, "P1", "2025-1.csv", value=200)
    run_process(first, output)

    report = scan_processing_plan(second, output, workers=1, show_progress=False)
    detail = report.participant_details[0]
    assert report.participant_counts["incremental"] == 1
    assert report.source_file_counts["to_parse"] == 1
    assert detail.later_added_months == ("2025-01",)
    assert detail.source_files_to_parse == 1
    assert detail.reason_code == "later_months_added"


def test_scan_reports_changed_historical_month_rebuild(tmp_path: Path) -> None:
    first, changed, output = tmp_path / "first", tmp_path / "changed", tmp_path / "output"
    write_month(first, "P1", "2024-12.csv", value=100)
    shutil.copytree(first, changed)
    write_month(changed, "P1", "2024-12.csv", value=120)
    run_process(first, output)

    report = scan_processing_plan(changed, output, workers=1, show_progress=False)
    detail = report.participant_details[0]
    assert report.participant_counts["rebuild"] == 1
    assert detail.changed_months == ("2024-12",)
    assert detail.reason_code == "historical_content_changed"
    assert detail.source_files_to_parse == 1


def test_scan_reports_recovered_historical_month_rebuild(tmp_path: Path) -> None:
    first, recovered, output = tmp_path / "first", tmp_path / "recovered", tmp_path / "output"
    write_month(first, "P1", "2024-12.csv")
    shutil.copytree(first, recovered)
    write_month(recovered, "P1", "2024-11.csv", value=80)
    run_process(first, output)

    report = scan_processing_plan(recovered, output, workers=1, show_progress=False)
    detail = report.participant_details[0]
    assert report.participant_counts["rebuild"] == 1
    assert detail.recovered_historical_months == ("2024-11",)
    assert detail.reason_code == "recovered_historical_months"


def test_scan_reports_strict_missing_month_block(tmp_path: Path) -> None:
    complete, incomplete, output = tmp_path / "complete", tmp_path / "incomplete", tmp_path / "output"
    write_month(complete, "P1", "2024-12.csv")
    write_month(complete, "P1", "2025-1.csv")
    write_month(incomplete, "P1", "2025-1.csv")
    run_process(complete, output)

    report = scan_processing_plan(incomplete, output, workers=1, show_progress=False)
    detail = report.participant_details[0]
    assert report.participant_counts["block"] == 1
    assert report.safe_to_process is False
    assert detail.missing_months == ("2024-12",)
    assert detail.reason_code == "missing_committed_months"


def test_scan_rejects_curated_output_root(tmp_path: Path) -> None:
    input_root, output_root = tmp_path / "input", tmp_path / "curated"
    write_month(input_root, "P1", "2024-12.csv")
    output_root.mkdir()
    (output_root / ".wearable_curation_state.sqlite").touch()
    try:
        scan_processing_plan(input_root, output_root, workers=1, show_progress=False)
    except Exception as exc:
        assert "curated root" in str(exc)
    else:
        raise AssertionError("Expected curated output root to be rejected")


def test_process_scan_cli_json_and_exit_modes(tmp_path: Path) -> None:
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    write_month(input_root, "P1", "2024-12.csv")
    command = [
        sys.executable, "-m", "wearable_project", "process-scan",
        "--input", str(input_root), "--output", str(output_root),
        "--workers", "1", "--json", "--include-details", "--no-progress",
    ]
    result = subprocess.run(
        command, cwd=Path(__file__).parents[1], text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["processing_needed"] is True
    assert payload["participants"]["rebuild"] == 1
    assert payload["participant_details"][0]["participant_id"] == "P1"
    assert not output_root.exists()

    result = subprocess.run(
        [*command, "--fail-if-work-needed"],
        cwd=Path(__file__).parents[1], text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    assert result.returncode == 3


def test_process_scan_report_file(tmp_path: Path) -> None:
    input_root, output_root, report_path = tmp_path / "input", tmp_path / "output", tmp_path / "scan.json"
    write_month(input_root, "P1", "2024-12.csv")
    command = [
        sys.executable, "-m", "wearable_project", "process-scan",
        "--input", str(input_root), "--output", str(output_root),
        "--workers", "1", "--json", "--include-details", "--no-progress",
        "--report-output", str(report_path),
    ]
    result = subprocess.run(
        command, cwd=Path(__file__).parents[1], text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    assert result.returncode == 0
    assert result.stdout == ""
    payload = json.loads(report_path.read_text())
    assert payload["participants"]["rebuild"] == 1
    assert not output_root.exists()


def test_verify_existing_hashes_detects_same_size_output_corruption(tmp_path: Path) -> None:
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    write_month(input_root, "P1", "2024-12.csv")
    run_process(input_root, output_root)
    feature = output_root / "P1" / "StepCount.csv"
    original = feature.read_bytes()
    # Replace one ASCII digit with another so the file size is unchanged.
    corrupted = original.replace(b"100", b"101", 1)
    assert len(corrupted) == len(original)
    feature.write_bytes(corrupted)

    default_scan = scan_processing_plan(input_root, output_root, workers=1, show_progress=False,)
    assert default_scan.participant_counts["skip"] == 1

    verified_scan = scan_processing_plan(
        input_root, output_root, workers=1, show_progress=False,verify_existing_hashes=True,
    )
    assert verified_scan.participant_counts["rebuild"] == 1
    assert verified_scan.participant_details[0].reason_code == "output_missing_or_invalid"


def test_scan_blocks_participant_missing_from_strict_cumulative_snapshot(tmp_path: Path) -> None:
    complete, reduced, output = tmp_path / "complete", tmp_path / "reduced", tmp_path / "output"
    write_month(complete, "P1", "2024-12.csv")
    write_month(complete, "P2", "2024-12.csv")
    shutil.copytree(complete / "P1", reduced / "P1")
    run_process(complete, output)

    report = scan_processing_plan(reduced, output, workers=1, show_progress=False)
    missing = [item for item in report.participant_details if item.participant_id == "P2"]
    assert len(missing) == 1
    assert missing[0].planned_action == "block"
    assert missing[0].reason_code == "participant_missing_from_snapshot"
    assert report.participant_counts["block"] == 1
    assert report.safe_to_process is False


def test_scan_exposes_stored_versions_and_mismatch_only_warning(tmp_path: Path) -> None:
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    write_month(input_root, "P1", "2024-12.csv")
    run_process(input_root, output_root)
    state = output_root / ".wearable_state.sqlite"
    with sqlite3.connect(state) as connection:
        connection.execute(
            "UPDATE participants SET parser_version=?, registry_version=?",
            ("native-parser-local-test", "local-registry-test"),
        )
        connection.commit()

    report = scan_processing_plan(input_root, output_root, workers=1, show_progress=False,)
    detail = report.participant_details[0]
    assert detail.planned_action == "rebuild"
    assert detail.reason_code == "parser_or_registry_changed"
    assert detail.stored_parser_version == "native-parser-local-test"
    assert detail.stored_registry_version == "local-registry-test"
    assert detail.current_source_signature == detail.committed_source_signature
    assert detail.existing_output_usable is True
    assert report.state_versions["stored_parser_version_counts"] == {"native-parser-local-test": 1}
    assert report.state_versions["stored_registry_version_counts"] == {"local-registry-test": 1}
    assert report.state_versions["version_mismatch_only_participants"] == 1
    # A mismatch-only rebuild is flagged but deliberately not recommended until the processing version is verified.
    assert any("Do not rebuild solely from this result" in item for item in report.warnings)
    rendered = report.as_dict(include_details=True)
    assert rendered["participant_details"][0]["stored_parser_version"] == "native-parser-local-test"
    assert "processing_environment" in rendered


def test_processing_environment_cli_json() -> None:
    command = [
        sys.executable, "-m", "wearable_project", "processing-environment", "--json",
    ]
    result = subprocess.run(
        command, cwd=Path(__file__).parents[1], text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    # A source checkout is expected to return 1 because it intentionally emits a warning.
    assert result.returncode in {0, 1}, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["parser_version"]
    assert payload["registry_version"]
    assert set(payload["core_module_integrity"]) == {
        "__init__.py", "cleaners.py", "parser.py", "pipeline.py",
        "registry.py", "resampling.py", "tracker.py", "writer.py",
    }
    assert all(payload["core_module_integrity"].values())
