"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import csv
from pathlib import Path
import pytest
from wearable_project.curation.google_fitness.contract import (
    expected_acquisition_method, virtual_default, virtual_row_values,
)
from wearable_project.curation.google_fitness.registry import get_policy
from wearable_project.curation.google_fitness.schema_migration import (
    LEGACY_STATE_SCHEMA_VERSION, SCHEMA_MIGRATION_AUDIT_VERSION, SCHEMA_MIGRATION_REPORT_VERSION,
    ParticipantTask, _guard_roots, _worker,
)
from wearable_project.processing.google_fitness.registry import get_feature_spec


def _write(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _native_row(feature: str) -> dict[str, str]:
    fields = get_feature_spec(feature).column_names
    row = {column: "" for column in fields}
    if feature == "Walking":
        row.update({
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-01T00:10:00Z",
            "calories": "2.5",
            "distance": "500",
            "quantity": "620",
            "tracked": "true",
            "device": "Android",
            "source_name": "com.google.android.gms",
            "source_id": "derived:walking",
            "collecting_method_version": "1.0",
            "outer_id": "A1",
            "payload_index": "0",
        })
    elif feature == "StepCount":
        row.update({
            "outer_datetime": "2024-01-01T00:00:00Z",
            "summary_date": "2024-01-01",
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-02T00:00:00Z",
            "value": "1000",
            "raw_value": "1000",
            "aggregation_source": "estimated_steps",
            "origin_data_source_id": "raw:steps",
            "app_name": "Google Play services",
            "app_package_name": "com.google.android.gms",
            "device_manufacturer": "Google",
            "device_model": "Pixel",
            "device_type": "phone",
            "device_uid": "device-1",
            "collecting_method_version": "2.0",
            "outer_id": "S1",
            "payload_index": "0",
        })
    else:  # pragma: no cover - helper is intentionally narrow.
        raise AssertionError(feature)
    return row


def _prepare(tmp_path: Path) -> ParticipantTask:
    participant = "P1"
    native = tmp_path / "native"
    legacy = tmp_path / "legacy"
    current = tmp_path / "current"
    filenames: list[str] = []
    for feature in ("Walking", "StepCount"):
        filename = f"{feature}.csv"
        filenames.append(filename)
        row = _native_row(feature)
        native_fields = tuple(get_feature_spec(feature).column_names)
        _write(native / participant / filename, native_fields, [row])

        policy = get_policy(feature)
        acquisition = expected_acquisition_method(feature, row, policy)
        virtual = virtual_row_values(feature, row, policy)
        legacy_values = dict(row)
        legacy_values["acquisition_method"] = acquisition
        for column in policy.virtual_columns:
            legacy_values[column] = virtual.get(column, virtual_default(column))
        if feature == "Walking":
            legacy_values.update({
                "curation_status": "pass",
                "curation_flags": "activity_quantity_semantics_unresolved",
                "include_by_default": "1",
            })
            legacy_fields = native_fields + (
                "acquisition_method", "derivation_method", "tracking_status",
                "curation_status", "curation_flags", "include_by_default",
            )
            current_values = {**row, "acquisition_method": acquisition}
            current_fields = native_fields + ("acquisition_method",)
        else:
            legacy_values.update({
                "step_count_role": "fit_app_preferred",
                "curation_status": "pass",
                "curation_flags": "",
                "include_by_default": "1",
            })
            legacy_fields = native_fields + (
                "acquisition_method", "canonical_value", "canonical_unit",
                "curation_unit_status", "unit_evidence", "derivation_method",
                "step_count_role", "curation_status", "curation_flags",
                "include_by_default",
            )
            current_values = {
                **row,
                "acquisition_method": acquisition,
                "step_count_role": "fit_app_preferred",
            }
            current_fields = native_fields + ("acquisition_method", "step_count_role")

        _write(legacy / participant / filename, legacy_fields, [legacy_values])
        _write(current / participant / filename, current_fields, [current_values])

    return ParticipantTask(participant, native, legacy, current, tuple(filenames), 20)


def test_schema_v2_semantics_reconstruct_schema_v1(tmp_path: Path) -> None:
    result = _worker(_prepare(tmp_path))
    assert result.issues == []
    assert result.files == 2
    assert result.rows == 2
    assert result.virtual_values_checked > 0
    assert result.policy_flags_removed == {"activity_quantity_semantics_unresolved": 1}


def test_schema_migration_detects_materialized_decision_change(tmp_path: Path) -> None:
    task = _prepare(tmp_path)
    path = task.current_root / task.participant_id / "Walking.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    rows[0]["acquisition_method"] = "unknown"
    _write(path, fields, rows)
    result = _worker(task)
    assert any(issue["check"] == "acquisition_method" for issue in result.issues)


def test_schema_migration_versions_are_explicit() -> None:
    assert SCHEMA_MIGRATION_AUDIT_VERSION == "google-fitness-curation-schema-migration-audit-v2-0.1.0"
    assert SCHEMA_MIGRATION_REPORT_VERSION == "google-fitness-curation-schema-migration-report-v2-0.1.0"
    assert LEGACY_STATE_SCHEMA_VERSION == "google-fitness-curation-state-cwp4-0.1.0"


def test_schema_migration_output_cannot_overlap_products(tmp_path: Path) -> None:
    native = tmp_path / "native"
    legacy = tmp_path / "legacy"
    current = tmp_path / "current"
    for root in (native, legacy, current):
        root.mkdir()
    with pytest.raises(Exception):
        _guard_roots(native, legacy, current, current / "reports")
