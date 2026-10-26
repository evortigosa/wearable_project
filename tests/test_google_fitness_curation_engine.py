"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import csv
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import multiprocessing as mp
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import pytest
from wearable_project.curation.google_fitness import (
    ENGINE_VERSION, curate_feature_file, validate_curated_feature_file, validate_engine_contract,
)
from wearable_project.curation.google_fitness.context import build_sleep_context, build_step_count_context
from wearable_project.curation.google_fitness.contract import (
    CURATED_SCHEMA_VERSION, effective_row_decisions, virtual_row_values,
)
from wearable_project.curation.google_fitness.decisions import DECISIONS
from wearable_project.curation.google_fitness.engine import feature_policy_fingerprint
from wearable_project.curation.google_fitness.models import DecisionStatus
from wearable_project.curation.google_fitness.provenance import classify_provenance
from wearable_project.curation.google_fitness.registry import get_policy
from wearable_project.curation.google_fitness.rules import RULES
from wearable_project.curation.google_fitness.strategies import (
    ENGINE_RULE_IMPLEMENTATION, annotate_row, validate_engine_rule_coverage,
)
from wearable_project.processing.google_fitness.registry import NATIVE_FEATURE_SPECS


_SAMPLE_NATIVE_ROOT_TEXT = os.environ.get("WEARABLE_GOOGLE_NATIVE_SAMPLE")
SAMPLE_NATIVE_ROOT = (
    Path(_SAMPLE_NATIVE_ROOT_TEXT).expanduser().resolve()
    if _SAMPLE_NATIVE_ROOT_TEXT
    else None
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _spawn_curate(args: tuple[str, str, str]) -> str:
    native, destination, feature = args
    return curate_feature_file(Path(native), Path(destination), feature).info.curated_sha256


def _write_native(path: Path, feature: str, rows: list[dict[str, object]]) -> None:
    fields = list(NATIVE_FEATURE_SPECS[feature].column_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in fields})


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _step_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "outer_datetime": "2024-01-01T00:00:00.000000Z",
        "summary_date": "2024-01-01",
        "start_date": "2024-01-01T00:00:00.000000Z",
        "end_date": "2024-01-01T23:00:00.000000Z",
        "value": 100,
        "raw_value": 100,
        "aggregation_source": "com.google.android.gms:estimated_steps",
        "origin_data_source_id": "raw:steps:phone",
        "app_name": "Fit",
        "app_package_name": "com.google.android.apps.fitness",
        "device_manufacturer": "Example",
        "device_model": "Phone",
        "device_type": "Phone",
        "device_uid": "dev",
        "collecting_method_version": "2.0",
        "outer_id": "g1",
        "payload_index": 0,
    }
    row.update(updates)
    return row


def test_cwp3_versions_and_reviewed_decisions() -> None:
    assert ENGINE_VERSION == "google-fitness-curation-engine-schema-v2-0.2.0"
    assert CURATED_SCHEMA_VERSION == "google-fitness-curated-schema-v2-0.1.0"
    assert DECISIONS["step_count_preference"].status is DecisionStatus.ACCEPTED
    assert DECISIONS["negative_calories_buckets"].status is DecisionStatus.ACCEPTED
    assert validate_engine_contract() == ()


def test_engine_rule_coverage_is_explicit_and_complete() -> None:
    assert set(ENGINE_RULE_IMPLEMENTATION) == set(RULES)
    assert validate_engine_rule_coverage() == ()


def test_feature_policy_fingerprint_is_deterministic() -> None:
    policy = get_policy("StepCount")
    assert feature_policy_fingerprint(policy) == feature_policy_fingerprint(policy)
    assert len(feature_policy_fingerprint(policy)) == 64


def test_activity_row_preserves_prefix_and_adds_context(tmp_path: Path) -> None:
    native = tmp_path / "Walking.csv"
    curated = tmp_path / "curated" / "Walking.csv"
    _write_native(native, "Walking", [{
        "start_date": "2024-01-01T10:00:00.000000Z",
        "end_date": "2024-01-01T10:10:00.000000Z",
        "calories": 10,
        "distance": 500,
        "quantity": 700,
        "tracked": 1,
        "device": "Android Wear",
        "source_name": "Android Wear",
        "source_id": "com.example.watch",
        "collecting_method_version": "1.0",
        "outer_id": "a1",
        "payload_index": 0,
    }])
    before = _hash(native)
    result = curate_feature_file(native, curated, "Walking")
    assert _hash(native) == before
    fields, rows = _read(curated)
    assert fields[: len(NATIVE_FEATURE_SPECS["Walking"].column_names)] == list(NATIVE_FEATURE_SPECS["Walking"].column_names)
    assert rows[0]["acquisition_method"] == "third_party_import"
    virtual = virtual_row_values("Walking", rows[0])
    assert virtual["derivation_method"] == "activity_classification"
    assert virtual["tracking_status"] == "tracked"
    assert rows[0].get("curation_flags", "") == ""
    assert "canonical_value" not in fields
    assert "derivation_method" not in fields
    assert "tracking_status" not in fields
    assert result.info.native_rows == result.info.curated_rows == 1
    assert validate_curated_feature_file(native, curated, "Walking") == 1


@pytest.mark.parametrize(
    ("value", "status", "include", "flag"),
    [
        (-0.0005, "review", "0", "negative_calories_tiny"),
        (-0.5, "review", "0", "negative_calories_small"),
        (-50, "exclude_default", "0", "negative_calories_material"),
        (-500, "exclude_default", "0", "negative_calories_extreme"),
        (100, "pass", "1", ""),
    ],
)
def test_calories_reviewed_magnitude_policy(
    tmp_path: Path, value: float, status: str, include: str, flag: str
) -> None:
    native = tmp_path / f"Calories_{str(value).replace('.', '_')}.csv"
    curated = tmp_path / "out" / native.name
    _write_native(native, "Calories", [{
        "start_date": "2024-01-01T00:00:00.000000Z",
        "end_date": "2024-01-01T23:59:59.000000Z",
        "value": value,
        "source_day_label": "Mon",
        "collecting_method_version": "1.0",
        "outer_id": "c1",
    }])
    curate_feature_file(native, curated, "Calories")
    _, rows = _read(curated)
    row = rows[0]
    decisions = effective_row_decisions(row)
    assert decisions["curation_status"] == status
    assert ("1" if decisions["include_by_default"] else "0") == include
    flags = set(decisions["curation_flags"])
    assert "canonical_unit_withheld_pending_exporter_mapping" not in flags
    if flag:
        assert flag in flags
    else:
        assert not any(item.startswith("negative_calories_") for item in flags)
    virtual = virtual_row_values("Calories", row)
    assert virtual.get("canonical_value", "") == ""
    assert virtual["curation_unit_status"] == "withheld_exporter_mapping_pending"


def test_distance_quality_rules(tmp_path: Path) -> None:
    native = tmp_path / "Distance.csv"
    curated = tmp_path / "out" / "Distance.csv"
    _write_native(native, "Distance", [
        {
            "start_date": "2024-01-01T00:00:00.000000Z",
            "end_date": "2024-01-01T00:00:00.000000Z",
            "value": 10,
            "source_day_label": "Mon",
            "collecting_method_version": "1.0",
            "outer_id": "d1",
        },
        {
            "start_date": "2024-01-02T00:00:00.000000Z",
            "end_date": "2024-01-02T00:00:01.000000Z",
            "value": 101,
            "source_day_label": "Tue",
            "collecting_method_version": "1.0",
            "outer_id": "d2",
        },
        {
            "start_date": "2024-01-03T00:00:00.000000Z",
            "end_date": "2024-01-03T00:01:00.000000Z",
            "value": -1,
            "source_day_label": "Wed",
            "collecting_method_version": "1.0",
            "outer_id": "d3",
        },
    ])
    curate_feature_file(native, curated, "Distance")
    _, rows = _read(curated)
    assert "positive_distance_zero_duration" in rows[0]["curation_flags"]
    assert effective_row_decisions(rows[0]) == {
        "curation_status": "review",
        "curation_flags": ("positive_distance_zero_duration",),
        "include_by_default": True,
    }
    assert "distance_rate_above_source_limit" in rows[1]["curation_flags"]
    assert "negative_distance" in rows[2]["curation_flags"]
    assert rows[2]["include_by_default"] == "0"


def test_weight_nonpoint_intervals_are_preserved(tmp_path: Path) -> None:
    native = tmp_path / "Weight.csv"
    curated = tmp_path / "out" / "Weight.csv"
    _write_native(native, "Weight", [
        {
            "start_date": "2024-01-01T12:00:00.000000Z",
            "end_date": "2024-01-01T12:05:00.000000Z",
            "value": 75,
            "source_day_label": "Mon",
            "added_by": "com.example.scale",
            "collecting_method_version": "2.0",
            "outer_id": "w1",
        },
        {
            "start_date": "2024-01-02T12:00:00.000000Z",
            "end_date": "2024-01-04T12:00:00.000000Z",
            "value": 75,
            "source_day_label": "Tue",
            "added_by": "com.google.android.apps.fitness",
            "collecting_method_version": "2.0",
            "outer_id": "w2",
        },
    ])
    curate_feature_file(native, curated, "Weight")
    _, rows = _read(curated)
    assert rows[0]["start_date"] != rows[0]["end_date"]
    assert "weight_nonpoint_interval" in rows[0]["curation_flags"]
    assert rows[0]["curation_status"] == "review"
    assert effective_row_decisions(rows[0])["include_by_default"] is True
    assert "weight_long_interval" in rows[1]["curation_flags"]
    assert rows[1]["include_by_default"] == "0"
    assert rows[0]["acquisition_method"] == "third_party_import"
    assert rows[1]["acquisition_method"] == "health_import_unknown"


def test_height_range_and_unit_withholding(tmp_path: Path) -> None:
    native = tmp_path / "Height.csv"
    curated = tmp_path / "out" / "Height.csv"
    _write_native(native, "Height", [{
        "start_date": "2024-01-01T12:00:00.000000Z",
        "end_date": "2024-01-01T12:00:00.000000Z",
        "value": 4,
        "source_day_label": "Mon",
        "added_by": "com.google.android.apps.fitness",
        "collecting_method_version": "2.0",
        "outer_id": "h1",
    }])
    curate_feature_file(native, curated, "Height")
    _, rows = _read(curated)
    assert "height_outside_source_range" in rows[0]["curation_flags"]
    assert rows[0]["include_by_default"] == "0"
    assert virtual_row_values("Height", rows[0])["curation_unit_status"] == "withheld_exporter_mapping_pending"


def test_step_count_group_roles_and_disagreement(tmp_path: Path) -> None:
    native = tmp_path / "StepCount.csv"
    curated = tmp_path / "out" / "StepCount.csv"
    rows = [
        _step_row(
            outer_datetime="2023-12-31T00:00:00.000000Z",
            summary_date=None,
            start_date=None,
            end_date=None,
            value=5000,
            raw_value=None,
            aggregation_source=None,
            collecting_method_version="1.0",
            outer_id="v1",
            payload_index=0,
        ),
        _step_row(value=5200, raw_value=5200, outer_id="g1", payload_index=0),
        _step_row(
            value=5400,
            raw_value=5400,
            outer_id="g1",
            payload_index=1,
            aggregation_source="com.google.android.gms:merge_step_deltas",
        ),
        _step_row(
            value=100,
            raw_value=100,
            outer_id="g2",
            payload_index=0,
            aggregation_source="com.google.android.gms:merge_step_deltas",
        ),
        _step_row(
            value=80,
            raw_value=80,
            outer_id="g2",
            payload_index=1,
            aggregation_source="com.example:third_party",
            app_package_name="com.example.app",
        ),
        _step_row(
            value=70,
            raw_value=70,
            outer_id="g3",
            payload_index=0,
            aggregation_source="com.example:third_party",
            app_package_name="com.example.app",
        ),
        _step_row(
            value=50,
            raw_value=50,
            outer_id="g4",
            payload_index=0,
            aggregation_source="com.example:a",
        ),
        _step_row(
            value=60,
            raw_value=60,
            outer_id="g4",
            payload_index=1,
            aggregation_source="com.example:b",
        ),
    ]
    _write_native(native, "StepCount", rows)
    result = curate_feature_file(native, curated, "StepCount")
    _, curated_rows = _read(curated)
    by_group: dict[str, list[dict[str, str]]] = {}
    for row in curated_rows:
        by_group.setdefault(row["outer_id"], []).append(row)

    assert by_group["v1"][0]["step_count_role"] == "legacy_v1"
    assert "invalid_native_interval" not in by_group["v1"][0].get("curation_flags", "")
    v1_virtual = virtual_row_values("StepCount", by_group["v1"][0])
    assert v1_virtual["canonical_unit"] == "count"
    assert v1_virtual["curation_unit_status"] == "resolved_source_convention"

    preferred, nonpreferred = by_group["g1"]
    assert preferred["step_count_role"] == "fit_app_preferred"
    assert effective_row_decisions(preferred)["include_by_default"] is True
    assert nonpreferred["step_count_role"] == "parallel_nonpreferred"
    assert nonpreferred["include_by_default"] == "0"
    assert preferred["curation_status"] == nonpreferred["curation_status"] == "review"
    assert "step_count_alternatives_disagree" in preferred["curation_flags"]

    assert by_group["g2"][0]["step_count_role"] == "merge_fallback"
    assert by_group["g2"][0]["curation_status"] == "review"
    assert effective_row_decisions(by_group["g2"][0])["include_by_default"] is True
    assert by_group["g3"][0]["step_count_role"] == "third_party_fallback"
    assert all(row["step_count_role"] == "ambiguous_parallel" for row in by_group["g4"])
    assert all(row["include_by_default"] == "0" for row in by_group["g4"])
    assert result.info.native_rows == result.info.curated_rows == len(rows)
    assert result.info.step_count_role_counts == {
        "ambiguous_parallel": 2,
        "fit_app_preferred": 1,
        "legacy_v1": 1,
        "merge_fallback": 1,
        "parallel_nonpreferred": 2,
        "third_party_fallback": 1,
    }


def test_step_count_rate_and_negative_rules(tmp_path: Path) -> None:
    native = tmp_path / "StepCount.csv"
    curated = tmp_path / "out" / "StepCount.csv"
    _write_native(native, "StepCount", [
        _step_row(value=20, raw_value=20, end_date="2024-01-01T00:00:01.000000Z"),
        _step_row(value=-1, raw_value=-1, outer_id="g2", payload_index=0),
    ])
    curate_feature_file(native, curated, "StepCount")
    _, rows = _read(curated)
    assert "step_count_rate_above_source_limit" in rows[0]["curation_flags"]
    assert rows[0]["include_by_default"] == "0"
    assert "negative_step_count" in rows[1]["curation_flags"]
    assert virtual_row_values("StepCount", rows[1]).get("canonical_value", "") == ""


def test_sleep_overlap_context_preserves_rows(tmp_path: Path) -> None:
    native = tmp_path / "Sleep.csv"
    curated = tmp_path / "out" / "Sleep.csv"
    base = {
        "calories": "",
        "distance": "",
        "quantity": "",
        "tracked": 0,
        "device": "Android Wear",
        "source_name": "Android Wear",
        "source_id": "com.example.watch",
        "collecting_method_version": "1.0",
        "payload_index": 0,
    }
    rows = [
        {**base, "start_date": "2024-01-01T22:00:00.000000Z", "end_date": "2024-01-02T02:00:00.000000Z", "value": "DEEP", "outer_id": "s1"},
        {**base, "start_date": "2024-01-02T01:00:00.000000Z", "end_date": "2024-01-02T03:00:00.000000Z", "value": "AWAKE", "outer_id": "s2"},
        {**base, "start_date": "2024-01-02T03:00:00.000000Z", "end_date": "2024-01-02T04:00:00.000000Z", "value": "SLEEP", "outer_id": "s3"},
        {**base, "start_date": "2024-01-02T03:30:00.000000Z", "end_date": "2024-01-02T05:00:00.000000Z", "value": "LIGHT", "outer_id": "s4"},
    ]
    _write_native(native, "Sleep", rows)
    curate_feature_file(native, curated, "Sleep")
    _, out = _read(curated)
    assert len(out) == 4
    assert "awake_overlaps_sleep_state" in out[0]["curation_flags"]
    assert "awake_overlaps_sleep_state" in out[1]["curation_flags"]
    assert out[0]["curation_status"] == out[1]["curation_status"] == "review"
    assert "generic_sleep_overlaps_detailed_state" in out[2]["curation_flags"]
    assert "generic_sleep_overlaps_detailed_state" in out[3]["curation_flags"]
    assert effective_row_decisions(out[2])["curation_status"] == "pass"
    assert effective_row_decisions(out[3])["curation_status"] == "pass"
    assert all(row.get("include_by_default", "1") == "1" for row in out)


def test_source_day_label_rules(tmp_path: Path) -> None:
    native = tmp_path / "Calories.csv"
    curated = tmp_path / "out" / "Calories.csv"
    _write_native(native, "Calories", [
        {
            "start_date": "2024-01-01T00:00:00.000000Z",  # Monday
            "end_date": "2024-01-01T23:59:59.000000Z",
            "value": 10,
            "source_day_label": "Thu",
            "collecting_method_version": "1.0",
            "outer_id": "c1",
        },
        {
            "start_date": "2024-01-02T00:00:00.000000Z",
            "end_date": "2024-01-02T23:59:59.000000Z",
            "value": 10,
            "source_day_label": "not-a-weekday",
            "collecting_method_version": "1.0",
            "outer_id": "c2",
        },
    ])
    curate_feature_file(native, curated, "Calories")
    _, rows = _read(curated)
    assert "source_day_label_nonadjacent_disagreement" in rows[0]["curation_flags"]
    assert rows[0]["curation_status"] == "review"
    assert "unrecognized_source_day_label" in rows[1]["curation_flags"]
    assert effective_row_decisions(rows[1])["curation_status"] == "pass"


def test_unknown_feature_fallback_is_visible_and_excluded(tmp_path: Path) -> None:
    native = tmp_path / "FutureFeature.csv"
    curated = tmp_path / "out" / "FutureFeature.csv"
    with native.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y"], lineterminator="\n")
        writer.writeheader()
        writer.writerow({"x": "1", "y": "2"})
    curate_feature_file(native, curated, "FutureFeature")
    fields, rows = _read(curated)
    assert fields == [
        "x", "y", "acquisition_method", "curation_status",
        "curation_flags", "include_by_default",
    ]
    assert rows[0]["acquisition_method"] == "unknown"
    assert rows[0]["curation_status"] == "review"
    assert rows[0]["curation_flags"] == "unknown_feature_policy"
    assert rows[0]["include_by_default"] == "0"


def test_native_schema_mismatch_fails_before_output(tmp_path: Path) -> None:
    native = tmp_path / "Calories.csv"
    native.write_text("start_date,value\n2024-01-01,1\n", encoding="utf-8")
    curated = tmp_path / "out" / "Calories.csv"
    with pytest.raises(Exception, match="Native schema mismatch"):
        curate_feature_file(native, curated, "Calories")
    assert not curated.exists()


def test_engine_is_deterministic(tmp_path: Path) -> None:
    native = tmp_path / "StepCount.csv"
    _write_native(native, "StepCount", [
        _step_row(value=100, raw_value=100, outer_id="g", payload_index=0),
        _step_row(value=120, raw_value=120, outer_id="g", payload_index=1, aggregation_source="merge_step_deltas"),
    ])
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    curate_feature_file(native, first, "StepCount")
    curate_feature_file(native, second, "StepCount")
    assert first.read_bytes() == second.read_bytes()


def test_context_builders_do_not_reorder_or_drop_rows() -> None:
    rows = [_step_row(outer_id="g", payload_index=0), _step_row(outer_id="g", payload_index=1, aggregation_source="merge_step_deltas")]
    context = build_step_count_context([{key: str(value) if value is not None else "" for key, value in row.items()} for row in rows])
    assert set(context) == {0, 1}
    sleep_rows = [
        {"start_date": "2024-01-01T00:00:00Z", "end_date": "2024-01-01T01:00:00Z", "value": "DEEP", "collecting_method_version": "1.0", "source_id": "x", "source_name": "x", "device": "x"},
        {"start_date": "2024-01-01T00:30:00Z", "end_date": "2024-01-01T02:00:00Z", "value": "LIGHT", "collecting_method_version": "1.0", "source_id": "x", "source_name": "x", "device": "x"},
    ]
    sleep = build_sleep_context(sleep_rows)
    assert set(sleep) == {0, 1}


def test_provenance_classifier_matches_calibration_contract() -> None:
    policy = get_policy("Walking")
    ann = classify_provenance("Walking", policy.event_kind, {
        "source_id": "com.example.watch",
        "source_name": "Watch",
        "device": "Android Wear",
        "tracked": "1",
    })
    assert ann.acquisition_method.value == "third_party_import"
    assert ann.derivation_method.value == "activity_classification"
    assert ann.tracking_status.value == "tracked"


def test_annotation_uses_only_policy_materialized_columns() -> None:
    policy = get_policy("Calories")
    row = {
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-01-01T01:00:00Z",
        "value": "10",
        "source_day_label": "Mon",
        "collecting_method_version": "1.0",
        "outer_id": "x",
    }
    annotation = annotate_row("Calories", row, policy)
    assert set(annotation.as_sparse_dict(policy)) <= set(policy.materialized_columns)


@pytest.mark.skipif(
    SAMPLE_NATIVE_ROOT is None or not SAMPLE_NATIVE_ROOT.is_dir(),
    reason="set WEARABLE_GOOGLE_NATIVE_SAMPLE to the accepted native sample root",
)
def test_accepted_native_sample_curates_non_destructively(tmp_path: Path) -> None:
    assert SAMPLE_NATIVE_ROOT is not None
    before = {
        path.relative_to(SAMPLE_NATIVE_ROOT).as_posix(): _hash(path)
        for path in SAMPLE_NATIVE_ROOT.rglob("*.csv")
    }
    output = tmp_path / "curated"
    total_rows = 0
    total_files = 0
    infos = []
    for native in sorted(SAMPLE_NATIVE_ROOT.glob("*/*.csv")):
        relative = native.relative_to(SAMPLE_NATIVE_ROOT)
        destination = output / relative
        result = curate_feature_file(native, destination, native.stem)
        infos.append(result.info)
        total_rows += result.info.curated_rows
        total_files += 1
        assert validate_curated_feature_file(native, destination, native.stem) == result.info.native_rows

    assert total_files == 37
    assert total_rows == 145_493
    assert sum(info.native_rows for info in infos) == sum(info.curated_rows for info in infos)
    after = {
        path.relative_to(SAMPLE_NATIVE_ROOT).as_posix(): _hash(path)
        for path in SAMPLE_NATIVE_ROOT.rglob("*.csv")
    }
    assert after == before


def test_google_engine_has_no_apple_scientific_imports() -> None:
    modules = [
        "wearable_project.curation.google_fitness.context",
        "wearable_project.curation.google_fitness.provenance",
        "wearable_project.curation.google_fitness.unit_resolution",
        "wearable_project.curation.google_fitness.strategies",
        "wearable_project.curation.google_fitness.engine",
    ]
    for name in modules:
        source = inspect.getsource(importlib.import_module(name))
        assert "curation.apple_healthkit" not in source
        assert "processing.apple_healthkit" not in source


def test_result_is_json_serializable(tmp_path: Path) -> None:
    native = tmp_path / "Calories.csv"
    curated = tmp_path / "out" / "Calories.csv"
    _write_native(native, "Calories", [{
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-01-01T01:00:00Z",
        "value": 1,
        "source_day_label": "Mon",
        "collecting_method_version": "1.0",
        "outer_id": "x",
    }])
    result = curate_feature_file(native, curated, "Calories")
    json.dumps(result.as_dict(), sort_keys=True)


def test_engine_is_spawn_process_safe(tmp_path: Path) -> None:
    natives = []
    for index, value in enumerate((10, -0.5)):
        native = tmp_path / "native" / f"Calories{index}.csv"
        _write_native(native, "Calories", [{
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-01T01:00:00Z",
            "value": value,
            "source_day_label": "Mon",
            "collecting_method_version": "1.0",
            "outer_id": f"c{index}",
        }])
        natives.append(native)
    tasks = [
        (str(native), str(tmp_path / "out" / f"{index}.csv"), "Calories")
        for index, native in enumerate(natives)
    ]
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as executor:
        hashes = list(executor.map(_spawn_curate, tasks))
    assert len(hashes) == 2
    assert all(len(item) == 64 for item in hashes)


def test_curated_validator_detects_native_prefix_mutation(tmp_path: Path) -> None:
    native = tmp_path / "Calories.csv"
    curated = tmp_path / "out" / "Calories.csv"
    _write_native(native, "Calories", [{
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-01-01T01:00:00Z",
        "value": 10,
        "source_day_label": "Mon",
        "collecting_method_version": "1.0",
        "outer_id": "c",
    }])
    curate_feature_file(native, curated, "Calories")
    text = curated.read_text(encoding="utf-8").replace(",10,", ",11,", 1)
    curated.write_text(text, encoding="utf-8")
    with pytest.raises(Exception, match="Native value changed"):
        validate_curated_feature_file(native, curated, "Calories")
