"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import csv
import math
from pathlib import Path
from wearable_project.curation.engine import curate_feature_file
from wearable_project.curation.pipeline import curate_dataset
from wearable_project.curation.state import load_curation_report
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


def _row(
    *, when: str, value: str, record: str, source: str = "source-a", end: str | None = None,
) -> dict[str, str]:
    return {
        "start_date": when,
        "end_date": end or when,
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


def _curate(participant: Path, feature: str, tmp_path: Path):
    context = build_participant_unit_context(participant)
    output = tmp_path / f"curated-{feature}.csv"
    result = curate_feature_file(participant / f"{feature}.csv", output, feature, context)
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows, result, context


def test_mass_magnitude_alone_never_resolves_overlapping_weight_range(tmp_path: Path) -> None:
    participant = tmp_path / "native" / "p1"
    _write(participant / "Weight.csv", BASE_FIELDS, [
        _row(when="2024-01-01T12:00:00Z", value="80", record="w1"),
        _row(when="2024-01-02T12:00:00Z", value="100", record="w2"),
    ])
    rows, result, context = _curate(participant, "Weight", tmp_path)
    assert all(row.get("canonical_value", "") == "" for row in rows)
    assert all(row["curation_unit_status"] == "ambiguous" for row in rows)
    assert all(row["curation_status"] == "review" for row in rows)
    assert all(row["include_by_default"] == "0" for row in rows)
    assert result.info.ambiguous_unit_rows == 2
    assert context.epochs[0].raw_unit is None
    assert context.epochs[0].evidence == "insufficient_epoch_evidence"


def test_round_metric_to_imperial_fingerprint_resolves_pounds(tmp_path: Path) -> None:
    participant = tmp_path / "native" / "p1"
    _write(participant / "Weight.csv", BASE_FIELDS, [
        _row(when="2024-01-01T12:00:00Z", value="110.23113109243879", record="w1"),
    ])
    rows, _, context = _curate(participant, "Weight", tmp_path)
    assert math.isclose(float(rows[0]["canonical_value"]), 50.0, rel_tol=1e-10)
    assert rows[0]["canonical_unit"] == "kg"
    assert rows[0]["unit_evidence"] == "round_metric_to_imperial_conversion_fingerprint"
    assert context.epochs[0].raw_unit == "lb"


def test_lean_mass_does_not_inherit_ambiguous_weight_unit(tmp_path: Path) -> None:
    participant = tmp_path / "native" / "p1"
    when = "2024-01-01T12:00:00Z"
    _write(participant / "Weight.csv", BASE_FIELDS, [
        _row(when=when, value="80", record="w1"),
    ])
    _write(participant / "BodyFatPercentage.csv", BASE_FIELDS, [
        _row(when=when, value="0.25", record="f1"),
    ])
    _write(participant / "LeanBodyMass.csv", BASE_FIELDS, [
        _row(when=when, value="60", record="l1"),
    ])
    rows, _, context = _curate(participant, "LeanBodyMass", tmp_path)
    assert rows[0].get("canonical_value", "") == ""
    assert rows[0]["curation_unit_status"] == "ambiguous"
    lean_epoch = next(epoch for epoch in context.epochs if epoch.feature == "LeanBodyMass")
    assert lean_epoch.raw_unit is None


def test_lean_mass_inherits_only_trusted_bmi_resolved_weight(tmp_path: Path) -> None:
    participant = tmp_path / "native" / "p1"
    when = "2024-01-01T12:00:00Z"
    _write(participant / "Height.csv", BASE_FIELDS, [
        _row(when=when, value="1.7", record="h1"),
    ])
    _write(participant / "Weight.csv", BASE_FIELDS, [
        _row(when=when, value="80", record="w1"),
    ])
    _write(participant / "BMI.csv", BASE_FIELDS, [
        _row(when=when, value=str(80 / (1.7**2)), record="b1"),
    ])
    _write(participant / "BodyFatPercentage.csv", BASE_FIELDS, [
        _row(when=when, value="0.25", record="f1"),
    ])
    _write(participant / "LeanBodyMass.csv", BASE_FIELDS, [
        _row(when=when, value="60", record="l1"),
    ])
    rows, _, context = _curate(participant, "LeanBodyMass", tmp_path)
    assert math.isclose(float(rows[0]["canonical_value"]), 60.0, rel_tol=1e-12)
    assert rows[0]["canonical_unit"] == "kg"
    assert rows[0]["unit_evidence"] == "bounded_body_composition_consistency_from_trusted_weight"
    lean_epoch = next(epoch for epoch in context.epochs if epoch.feature == "LeanBodyMass")
    assert lean_epoch.raw_unit == "kg"


def test_sleep_cross_source_and_missing_inbed_are_informational(tmp_path: Path) -> None:
    participant = tmp_path / "native" / "p1"
    fields = BASE_FIELDS + ["device", "metadata_device_name"]

    def sleep(start: str, end: str, state: str, record: str, source: str) -> dict[str, str]:
        row = _row(when=start, end=end, value=state, record=record, source=source)
        row["device"] = source + "-watch"
        row["metadata_device_name"] = source + "-device"
        return row

    _write(participant / "Sleep.csv", fields, [
        sleep("2024-01-01T00:00:00Z", "2024-01-01T08:00:00Z", "INBED", "a0", "A"),
        sleep("2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z", "CORE", "a1", "A"),
        sleep("2024-01-01T01:30:00Z", "2024-01-01T02:30:00Z", "REM", "b1", "B"),
    ])
    rows, result, _ = _curate(participant, "Sleep", tmp_path)
    flags = [set(filter(None, row.get("curation_flags", "").split(";"))) for row in rows]
    assert "cross_source_sleep_overlap" in flags[1]
    assert "cross_source_sleep_overlap" in flags[2]
    assert "source_has_no_inbed_state" in flags[2]
    assert rows[1].get("curation_status", "") in {"", "pass"}
    assert rows[2].get("curation_status", "") in {"", "pass"}
    assert result.info.review_rows == 0


def test_sleep_detailed_state_outside_available_inbed_is_review(tmp_path: Path) -> None:
    participant = tmp_path / "native" / "p1"
    _write(participant / "Sleep.csv", BASE_FIELDS, [
        _row(
            when="2024-01-01T00:00:00Z", end="2024-01-01T01:00:00Z", value="INBED", record="s0",
        ),
        _row(
            when="2024-01-01T02:00:00Z", end="2024-01-01T03:00:00Z", value="CORE", record="s1",
        ),
    ])
    rows, result, _ = _curate(participant, "Sleep", tmp_path)
    assert "detailed_state_outside_available_inbed" in rows[1]["curation_flags"]
    assert rows[1]["curation_status"] == "review"
    assert result.info.review_rows == 1


def test_acquisition_counts_include_unclassified_and_close_to_rows(tmp_path: Path) -> None:
    native = tmp_path / "native" / "p1" / "StepCount.csv"
    _write(native, BASE_FIELDS, [
        _row(
            when="2024-01-01T00:00:00Z", end="2024-01-01T01:00:00Z",
            value="100", record="s1", source="com.apple.Health",
        ),
    ])
    summary = curate_dataset(tmp_path / "native", tmp_path / "curated", workers=1, max_in_flight=1, allow_unmanaged_native_root=True)
    report = summary.as_dict()
    assert report["run_activity"]["acquisition_counts"] == {"unclassified": 1}
    assert report["run_activity"]["acquisition_classification"] == {
        "classified_rows": 0,
        "unclassified_rows": 1,
        "accounting_complete": True,
    }


def test_unit_epoch_report_exposes_source_context(tmp_path: Path) -> None:
    fields = BASE_FIELDS + ["raw_unit", "device", "metadata_device_name", "time_zone", "was_user_entered"]
    row = _row(when="2024-01-01T12:00:00Z", value="110.23113109243879", record="w1", source="scale.app")
    row.update({
        "raw_unit": "lb",
        "device": "scale-1",
        "metadata_device_name": "Smart Scale",
        "time_zone": "Asia/Jerusalem",
        "was_user_entered": "0",
    })
    _write(tmp_path / "native" / "p1" / "Weight.csv", fields, [row])
    curate_dataset(tmp_path / "native", tmp_path / "curated", workers=1, max_in_flight=1, allow_unmanaged_native_root=True)
    report = load_curation_report(
        tmp_path / "curated" / ".wearable_curation_state.sqlite", include_details=True,
    )
    epoch = report["snapshot"]["unit_resolution"]["epoch_details"][0]
    assert epoch["source_id"] == "scale.app"
    assert epoch["source_name"] == "scale.app"
    assert epoch["device"] == "scale-1"
    assert epoch["metadata_device_name"] == "Smart Scale"
    assert epoch["time_zone"] == "Asia/Jerusalem"
    assert epoch["was_user_entered"] == "0"
