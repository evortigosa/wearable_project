"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import csv
import json
from pathlib import Path
import sqlite3
import pytest
from wearable_project.curation.google_fitness.contract import (
    CURATED_SCHEMA_VERSION, CURATION_CONTRACT_FILENAME, MATERIALIZED_COLUMN_ORDER, POLICY_LEVEL_FACTS,
    ROW_DEFAULTS, VIRTUAL_COLUMN_ORDER, contract_document, contract_fingerprint, effective_row_decisions,
    expected_acquisition_method, validate_contract_file, virtual_row_values,
)
from wearable_project.curation.google_fitness.engine import ENGINE_VERSION, curate_feature_file
from wearable_project.curation.google_fitness.pipeline import PIPELINE_VERSION, curate_dataset
from wearable_project.curation.google_fitness.registry import CURATION_POLICIES, get_policy
from wearable_project.curation.google_fitness.state import (
    CURATION_STATE_FILENAME, CURATION_STATE_SCHEMA_VERSION,
)
from wearable_project.exceptions import OutputValidationError
from wearable_project.processing.google_fitness.registry import get_feature_spec


def _write_native(path: Path, feature: str, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(get_feature_spec(feature).column_names)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _managed_native_root(tmp_path: Path) -> Path:
    root = tmp_path / "native"
    calories = _write_and_return(
        root / "P1" / "Calories.csv",
        "Calories",
        [{
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-02T00:00:00Z",
            "value": "-0.5",
            "source_day_label": "Mon",
            "collecting_method_version": "2.0",
            "outer_id": "c1",
        }],
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
    import hashlib
    connection.execute("INSERT INTO metadata VALUES('source_key','google_fitness')")
    connection.execute(
        "INSERT INTO feature_outputs VALUES(?,?,?,?,?,?,?)",
        ("P1", "Calories", calories.name, 1, calories.stat().st_size,
         hashlib.sha256(calories.read_bytes()).hexdigest(), 1),
    )
    connection.commit()
    connection.close()
    return root


def _write_and_return(path: Path, feature: str, rows: list[dict[str, object]]) -> Path:
    _write_native(path, feature, rows)
    return path


def test_schema_v2_policy_contract_is_explicit_and_complete() -> None:
    assert CURATED_SCHEMA_VERSION == "google-fitness-curated-schema-v2-0.1.0"
    assert ROW_DEFAULTS == {
        "curation_status": "pass",
        "curation_flags": [],
        "include_by_default": 1,
    }
    assert MATERIALIZED_COLUMN_ORDER == (
        "acquisition_method",
        "step_count_role",
        "curation_status",
        "curation_flags",
        "include_by_default",
    )
    assert set(VIRTUAL_COLUMN_ORDER) == {
        "canonical_value", "canonical_unit", "curation_unit_status",
        "unit_evidence", "derivation_method", "tracking_status",
    }
    assert {item["former_row_flag"] for item in POLICY_LEVEL_FACTS} == {
        "activity_quantity_semantics_unresolved",
        "canonical_unit_withheld_pending_exporter_mapping",
    }
    assert len(contract_fingerprint()) == 64
    assert len(CURATION_POLICIES) == 67


def test_activity_schema_materializes_only_acquisition_and_row_decisions() -> None:
    policy = get_policy("Walking")
    assert policy.output.materialized_columns == (
        "acquisition_method", "curation_status", "curation_flags", "include_by_default"
    )
    assert policy.output.virtual_columns == ("derivation_method", "tracking_status")
    assert set(policy.output.fixed_columns) == {"acquisition_method"}
    assert set(policy.output.sparse_columns) == {
        "curation_status", "curation_flags", "include_by_default"
    }


def test_step_count_schema_keeps_role_and_virtualizes_identity_units() -> None:
    policy = get_policy("StepCount")
    assert policy.output.materialized_columns == (
        "acquisition_method", "step_count_role", "curation_status",
        "curation_flags", "include_by_default",
    )
    assert set(policy.output.virtual_columns) == {
        "canonical_value", "canonical_unit", "curation_unit_status",
        "unit_evidence", "derivation_method",
    }


def test_virtual_projections_reconstruct_removed_columns(tmp_path: Path) -> None:
    native = tmp_path / "Walking.csv"
    curated = tmp_path / "curated" / "Walking.csv"
    _write_native(native, "Walking", [{
        "start_date": "2024-01-01T10:00:00Z",
        "end_date": "2024-01-01T10:10:00Z",
        "calories": "10",
        "distance": "500",
        "quantity": "700",
        "tracked": "1",
        "device": "Android Wear",
        "source_name": "Android Wear",
        "source_id": "com.example.watch",
        "collecting_method_version": "1.0",
        "outer_id": "a1",
        "payload_index": "0",
    }])
    curate_feature_file(native, curated, "Walking")
    fields, rows = _read(curated)
    row = rows[0]
    assert fields[-1] == "acquisition_method"
    assert not set(VIRTUAL_COLUMN_ORDER) & set(fields)
    assert row["acquisition_method"] == expected_acquisition_method("Walking", row)
    assert virtual_row_values("Walking", row) == {
        "derivation_method": "activity_classification",
        "tracking_status": "tracked",
    }
    assert effective_row_decisions(row) == {
        "curation_status": "pass",
        "curation_flags": (),
        "include_by_default": True,
    }


def test_policy_level_facts_are_not_repeated_as_flags(tmp_path: Path) -> None:
    native = tmp_path / "Calories.csv"
    curated = tmp_path / "curated" / "Calories.csv"
    _write_native(native, "Calories", [{
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-01-02T00:00:00Z",
        "value": "10",
        "source_day_label": "Mon",
        "collecting_method_version": "1.0",
        "outer_id": "c1",
    }])
    curate_feature_file(native, curated, "Calories")
    fields, rows = _read(curated)
    assert fields[-1] == "acquisition_method"
    assert "curation_flags" not in fields
    assert virtual_row_values("Calories", rows[0]) == {
        "curation_unit_status": "withheld_exporter_mapping_pending",
        "unit_evidence": "exporter_mapping_pending:proposed_kcal",
        "derivation_method": "application_summary",
    }


def test_root_contract_is_deterministic_and_valid(tmp_path: Path) -> None:
    native = _managed_native_root(tmp_path)
    curated = tmp_path / "curated"
    curate_dataset(native, curated, workers=1, verify_native_hashes=True)
    contract = curated / CURATION_CONTRACT_FILENAME
    assert contract.is_file()
    assert validate_contract_file(
        contract,
        engine_version=ENGINE_VERSION,
        pipeline_version=PIPELINE_VERSION,
        state_schema_version=CURATION_STATE_SCHEMA_VERSION,
    ) == ()
    payload = json.loads(contract.read_text())
    assert payload == contract_document(
        engine_version=ENGINE_VERSION,
        pipeline_version=PIPELINE_VERSION,
        state_schema_version=CURATION_STATE_SCHEMA_VERSION,
    )
    assert payload["contract_fingerprint"] == contract_fingerprint()


def test_schema_v1_state_is_rejected_and_requires_new_root(tmp_path: Path) -> None:
    native = _managed_native_root(tmp_path)
    curated = tmp_path / "legacy_curated"
    curated.mkdir()
    state = curated / CURATION_STATE_FILENAME
    connection = sqlite3.connect(state)
    connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO metadata VALUES(?,?)",
        (
            ("source_key", "google_fitness"),
            ("product_kind", "curated"),
            ("state_schema_version", "google-fitness-curation-state-cwp4-0.1.0"),
        ),
    )
    connection.commit()
    connection.close()
    with pytest.raises(OutputValidationError, match="incompatible Google Fitness curation state schema"):
        curate_dataset(native, curated, workers=1)
