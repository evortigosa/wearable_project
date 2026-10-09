"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import csv
import json
import zipfile
from pathlib import Path
from wearable_project.inspection.cli import main


COLUMNS = [
    "id", "participant_id", "data_source", "name", "datetime", "data", "collecting_method_version",
    "created_at", "updated_at",
]


def write_month(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def row(pid: str, source: str, feature: str, payload, *, version: str = "1.0", day: str = "2024-01-01") -> dict[str, str]:
    return {
        "id": f"{pid}-{source}-{feature}-{version}-{day}",
        "participant_id": pid,
        "data_source": source,
        "name": feature,
        "datetime": f"{day} 00:00:00",
        "data": json.dumps(payload),
        "collecting_method_version": version,
        "created_at": f"{day} 00:00:00",
        "updated_at": f"{day} 00:00:00",
    }


def build_fixture(root: Path) -> None:
    pid = "100"
    large_apple_payload = [{"value": "x" * 200_000}]
    google_steps = [
        {
            "source": "com.google.android.gms:merge_step_deltas",
            "steps": [{"date": "2024-01-01", "value": 1200}],
            "raw_steps": [{
                "steps": 1200,
                "data_source_id": "derived:com.google.step_count.delta:com.google.android.gms:aggregated",
                "origin_data_source_id": "raw:com.google.step_count.cumulative:samsung:SM-X:uid:counter",
            }],
        },
        {
            "source": "com.google.android.gms:estimated_steps",
            "steps": [{"date": "2024-01-01", "value": 1000}],
            "raw_steps": [{
                "steps": 1000,
                "data_source_id": "derived:com.google.step_count.delta:com.google.android.gms:aggregated",
                "origin_data_source_id": "raw:com.google.step_count.cumulative:samsung:SM-X:uid:counter",
            }],
        },
    ]
    activities = [
        {
            "activity_name": "walking",
            "start": 1704067200000,
            "end": 1704067260000,
            "calories": 2.5,
            "distance": 80.0,
            "device": "Android",
            "source_id": "com.google.android.gms",
            "source_name": "Android",
            "tracked": True,
        },
        {
            "activity_name": "sleep.deep",
            "start": 1704067260000,
            "end": 1704070860000,
            "calories": 20.0,
            "device": "Android Wear",
            "source_id": "com.example.watch",
            "source_name": "Wear",
            "tracked": False,
        },
    ]
    rows = [
        row(pid, "AppleHealthkit", "StepCount", large_apple_payload, version="2.0"),
        row(pid, "GoogleFitness", "StepCount", google_steps, version="2.0"),
        row(pid, "GoogleFitness", "Activities", activities, version="1.0"),
    ]
    write_month(root / pid / "2024-1.csv", rows)
    (root / pid / "export-ended.txt").write_text("done\n", encoding="utf-8")

    pid2 = "200"
    write_month(
        root / pid2 / "2024-01.csv",
        [row(pid2, "GoogleFitness", "Weight", [{"value": 70.0, "start_date": "2024-01-01T00:00:00Z", "end_date": "2024-01-01T00:00:00Z"}], version="2.0")],
    )
    (root / pid2 / "export-ended.txt").write_text("done\n", encoding="utf-8")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_mixed_root_is_source_qualified_and_large_apple_field_is_safe(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    build_fixture(raw)
    output = tmp_path / "report"
    assert main([
        "--input", str(raw),
        "--output", str(output),
        "--decode-non-apple",
        "--checkpoint-every", "0",
        "--progress-every", "0",
    ]) == 0

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["participants"] == 2
    assert summary["outer_rows"] == 4
    assert summary["mixed_source_participants"] == 1
    assert summary["payload_decode_failures"] == 0
    assert summary["source_membership"] == {
        "applehealthkit + googlefitness": 1, "googlefitness": 1,
    }

    features = load_csv(output / "source_feature_versions.csv")
    apple_steps = [r for r in features if r["source_key"] == "applehealthkit" and r["feature"] == "StepCount"]
    google_steps = [r for r in features if r["source_key"] == "googlefitness" and r["feature"] == "StepCount"]
    assert apple_steps[0]["decoded_outer_rows"] == "0"
    assert google_steps[0]["decoded_outer_rows"] == "1"
    assert google_steps[0]["payload_items"] == "2"

    activities = {r["activity_name"]: r for r in load_csv(output / "activity_label_inventory.csv")}
    assert activities["walking"]["candidate_feature_file"] == "Walking.csv"
    assert activities["sleep.deep"]["candidate_feature_file"] == "Sleep.csv"

    steps = load_csv(output / "stepcount_alternative_summary.csv")
    v2 = next(r for r in steps if r["collecting_method_version"] == "2.0")
    assert v2["rows_with_both"] == "1"
    assert v2["rows_different"] == "1"


def test_zip_input_matches_directory_inventory(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    build_fixture(raw)
    archive = tmp_path / "raw.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(raw.rglob("*")):
            if path.is_file():
                handle.write(path, Path("wrapped") / path.relative_to(raw))

    directory_output = tmp_path / "directory_report"
    zip_output = tmp_path / "zip_report"
    common = ["--decode-non-apple", "--checkpoint-every", "0", "--progress-every", "0"]
    assert main(["--input", str(raw), "--output", str(directory_output), *common]) == 0
    assert main(["--input", str(archive), "--output", str(zip_output), *common]) == 0

    directory_summary = json.loads((directory_output / "summary.json").read_text(encoding="utf-8"))
    zip_summary = json.loads((zip_output / "summary.json").read_text(encoding="utf-8"))
    for key in (
        "participants",
        "monthly_files",
        "outer_rows",
        "decoded_outer_rows",
        "payload_items",
        "mixed_source_participants",
        "source_membership",
        "source_labels",
        "payload_decode_failures",
    ):
        assert directory_summary[key] == zip_summary[key]
