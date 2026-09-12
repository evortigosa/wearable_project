"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd


def write_month(root: Path, participant: str, filename: str, rows: list[tuple[str, str, list[dict]]]) -> None:
    directory = root / participant
    directory.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "participant_id", "data_source", "name", "datetime", "data",
        "collecting_method_version", "created_at", "updated_at",
    ]
    with (directory / filename).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (feature, outer_date, items) in enumerate(rows):
            writer.writerow({
                "id": f"outer-{filename}-{index}", "participant_id": participant,
                "data_source": "AppleHealthkit", "name": feature,
                "datetime": outer_date, "data": json.dumps(repr(items)),
                "collecting_method_version": "2.0",
                "created_at": "2025-02-01T00:00:00Z",
                "updated_at": "2025-02-01T00:00:00Z",
            })


def run_process(input_root: Path, output_root: Path, *, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable, "-m", "wearable_project", "process",
        "--input", str(input_root), "--output", str(output_root),
        "--workers", "1", "--max-in-flight", "1", "--json-summary",
    ]
    result = subprocess.run(
        command, cwd=Path(__file__).parents[1], text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    if expect_success:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_incremental_update_equals_from_scratch_and_historical_change_rebuilds(tmp_path: Path) -> None:
    first, second, changed = tmp_path / "first", tmp_path / "second", tmp_path / "changed"
    output, scratch = tmp_path / "output", tmp_path / "scratch"
    december = {
        "value": 100,
        "start_date": "2024-12-01T10:00:00+0200",
        "end_date": "2024-12-01T11:00:00+0200",
    }
    january = {
        "value": 200,
        "start_date": "2025-01-01T10:00:00+0200",
        "end_date": "2025-01-01T11:00:00+0200",
    }
    write_month(first, "P1", "2024-12.csv", [("StepCount", "2024-12-01 00:00:00", [december])])
    shutil.copytree(first, second)
    write_month(second, "P1", "2025-1.csv", [("StepCount", "2025-01-01 00:00:00", [january])])

    run_process(first, output)
    run_process(second, output)
    run_process(second, scratch)
    assert (output / "P1" / "StepCount.csv").read_bytes() == (scratch / "P1" / "StepCount.csv").read_bytes()

    shutil.copytree(second, changed)
    corrected = {**december, "value": 120}
    write_month(changed, "P1", "2024-12.csv", [("StepCount", "2024-12-01 00:00:00", [corrected])])
    run_process(changed, output)
    dataframe = pd.read_csv(output / "P1" / "StepCount.csv")
    assert sorted(dataframe["value"].tolist()) == [120, 200]
    assert 110 not in dataframe["value"].tolist()  # no median synthesis


def test_strict_snapshot_missing_month_keeps_committed_output(tmp_path: Path) -> None:
    complete, incomplete, output = tmp_path / "complete", tmp_path / "incomplete", tmp_path / "output"
    event = {
        "value": 100,
        "start_date": "2024-12-01T10:00:00+0200",
        "end_date": "2024-12-01T11:00:00+0200",
    }
    write_month(complete, "P1", "2024-12.csv", [("StepCount", "2024-12-01 00:00:00", [event])])
    write_month(incomplete, "P1", "2025-1.csv", [("StepCount", "2025-01-01 00:00:00", [event])])
    run_process(complete, output)
    before = (output / "P1" / "StepCount.csv").read_bytes()
    result = run_process(incomplete, output, expect_success=False)
    assert result.returncode == 1
    assert "snapshot omits committed months" in result.stdout
    assert (output / "P1" / "StepCount.csv").read_bytes() == before
