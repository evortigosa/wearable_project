"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import csv
import math
from pathlib import Path
from wearable_project.curation.engine import curate_feature_file
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


def _energy_row(*, raw_unit: str, value: str = "400", **extra: str) -> dict[str, str]:
    row = {
        "start_date": "2024-01-01T12:00:00Z",
        "end_date": "2024-01-01T12:00:00Z",
        "value": value,
        "datetime": "2024-01-01T00:00:00Z",
        "created_at": "2024-01-02T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
        "data_source": "AppleHealthkit",
        "collecting_method_version": "2.0",
        "record_id": "energy-1",
        "source_id": "nutrition.app",
        "source_name": "Nutrition App",
        "raw_unit": raw_unit,
    }
    row.update(extra)
    return row


def _curate_energy(tmp_path: Path, fields: list[str], row: dict[str, str]):
    participant = tmp_path / "native" / "p1"
    native = participant / "EnergyConsumed.csv"
    _write(native, fields, [row])
    context = build_participant_unit_context(participant)
    output = tmp_path / "curated.csv"
    result = curate_feature_file(native, output, "EnergyConsumed", context)
    with output.open(encoding="utf-8", newline="") as handle:
        curated = next(csv.DictReader(handle))
    return curated, result, context


def test_registry_raw_unit_without_explicit_provenance_does_not_authorize_conversion(tmp_path: Path,) -> None:
    fields = BASE_FIELDS + ["raw_unit"]
    row, result, context = _curate_energy(tmp_path, fields, _energy_row(raw_unit="kcal"),)

    assert row.get("canonical_value", "") == ""
    assert row.get("canonical_unit", "") == ""
    assert row["curation_unit_status"] == "ambiguous"
    assert row["curation_status"] == "review"
    assert "canonical_unit_withheld" in row["curation_flags"]
    assert "unit_unresolved" in row["curation_flags"]
    assert row["unit_evidence"] == "native_unit_provenance_not_explicit"
    assert result.info.canonical_value_rows == 0
    assert result.info.ambiguous_unit_rows == 1
    assert context.epochs[0].raw_unit is None
    assert context.epochs[0].evidence == "native_unit_provenance_not_explicit"


def test_payload_explicit_identity_unit_can_authorize_energy_canonical_value(tmp_path: Path,) -> None:
    fields = BASE_FIELDS + ["raw_unit", "unit_status"]
    row, result, context = _curate_energy(
        tmp_path, fields, _energy_row(raw_unit="kcal", unit_status="explicit"),
    )
    assert math.isclose(float(row["canonical_value"]), 400.0, rel_tol=0, abs_tol=1e-12)
    assert row["canonical_unit"] == "kcal"
    assert row["curation_unit_status"] == "resolved_source_epoch"
    assert row["unit_evidence"] == "native_payload_explicit_unit"
    assert row.get("curation_status", "") in {"", "pass"}
    assert result.info.canonical_value_rows == 1
    assert result.info.ambiguous_unit_rows == 0
    assert context.epochs[0].raw_unit == "kcal"


def test_payload_unit_evidence_marker_can_authorize_energy_unit(tmp_path: Path) -> None:
    fields = BASE_FIELDS + ["raw_unit", "unit_evidence"]
    row, _, context = _curate_energy(
        tmp_path, fields, _energy_row(raw_unit="kJ", value="418.4", unit_evidence="payload_unit"),
    )
    assert math.isclose(float(row["canonical_value"]), 100.0, rel_tol=1e-12)
    assert row["canonical_unit"] == "kcal"
    # The native ``unit_evidence`` column is immutable; curation evidence is recorded on the unit epoch when
    # the native schema already uses that name.
    assert row["unit_evidence"] == "payload_unit"
    assert context.epochs[0].evidence == "native_payload_explicit_unit"
    assert context.epochs[0].raw_unit == "kJ"


def test_explicit_unresolved_marker_preserves_explicit_supported_unit(tmp_path: Path) -> None:
    fields = BASE_FIELDS + ["raw_unit", "unit_status"]
    row, _, context = _curate_energy(
        tmp_path, fields, _energy_row(raw_unit="kJ", value="836.8", unit_status="explicit_unresolved"),
    )
    assert math.isclose(float(row["canonical_value"]), 200.0, rel_tol=1e-12)
    assert row["canonical_unit"] == "kcal"
    assert row["unit_evidence"] == "native_payload_explicit_unit"
    assert context.epochs[0].raw_unit == "kJ"


def test_unsupported_explicit_unit_remains_unresolved(tmp_path: Path) -> None:
    fields = BASE_FIELDS + ["raw_unit", "unit_status"]
    row, result, context = _curate_energy(
        tmp_path, fields, _energy_row(raw_unit="cal", unit_status="explicit_unresolved"),
    )
    assert row.get("canonical_value", "") == ""
    assert row["curation_unit_status"] == "ambiguous"
    assert row["curation_status"] == "review"
    assert result.info.ambiguous_unit_rows == 1
    assert context.epochs[0].raw_unit is None
