"""
Wearable Data Processing and Modeling project
WP1 acceptance tests for the Google Fitness native registry and schemas.
"""


from __future__ import annotations
import csv
from dataclasses import FrozenInstanceError
import json
import operator
from pathlib import Path
import subprocess
import sys
import pytest
from wearable_project.processing.google_fitness.registry import (
    ACTIVITY_MEASUREMENT_ORDER,
    ACTIVITY_ROUTE_BY_LABEL,
    ACTIVITY_ROUTES,
    EXPECTED_REGISTRY_FINGERPRINT,
    NATIVE_FEATURE_SPECS,
    NORMALIZED_RAW_SOURCE,
    REGISTRY_VERSION,
    SLEEP_STATE_MAPPING,
    SOURCE_FEATURE_VERSION_BY_KEY,
    SOURCE_FEATURE_VERSION_SPECS,
    STEP_COUNT_DATA_SOURCE_ID,
    STEP_COUNT_DATA_TYPE_NAME,
    UNREGISTERED_ACTIVITY_POLICY,
    IdentityMaturity,
    NativeFeatureFamily,
    RouteKind,
    activity_feature_names,
    feature_names,
    get_activity_route,
    get_feature_spec,
    get_source_feature_version_spec,
    normalize_source_label,
    registry_fingerprint,
    registry_payload,
    validate_registry,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURE_DIR / "google_fitness_wp1_activity_inventory.csv"
FIXTURE_METADATA = (FIXTURE_DIR / "google_fitness_wp1_activity_inventory.metadata.json")


def _activity_fixture() -> list[dict[str, str]]:
    with FIXTURE.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_activity_fixture_is_traceable_to_accepted_phase0_census() -> None:
    metadata = json.loads(FIXTURE_METADATA.read_text(encoding="utf-8"))
    assert metadata["phase0_acceptance_status"] == "pass"
    assert metadata["profiler_version"] == "mixed-source-profiler-1.2.1"
    assert metadata["source_archive_sha256"] == "a0926f7865263d6977a8535bc675dd975d4ae6881cdaccf458d715c3019288e0"
    assert metadata["activity_route_count"] == 66
    assert metadata["activity_destination_count"] == 62
    assert metadata["native_feature_count"] == 67
    assert metadata["source_feature_version_count"] == 10
    assert metadata["accepted_registry_version"] == REGISTRY_VERSION
    assert metadata["accepted_registry_fingerprint"] == EXPECTED_REGISTRY_FINGERPRINT


def test_registry_is_internally_valid_and_has_expected_counts() -> None:
    validation = validate_registry()
    assert validation.valid, validation.issues
    assert len(ACTIVITY_ROUTES) == 66
    assert len(activity_feature_names()) == 62
    assert len(NATIVE_FEATURE_SPECS) == 67
    assert len(SOURCE_FEATURE_VERSION_SPECS) == 10
    assert "Activities" not in NATIVE_FEATURE_SPECS
    assert "Activities.csv" not in {spec.filename for spec in NATIVE_FEATURE_SPECS.values()}


def test_activity_routes_exactly_match_phase0_fixture() -> None:
    fixture = _activity_fixture()
    assert len(fixture) == 66
    assert {row["raw_label"] for row in fixture} == set(ACTIVITY_ROUTE_BY_LABEL)

    for row in fixture:
        route = get_activity_route(row["raw_label"])
        expected_measurements = tuple(
            name for name in ACTIVITY_MEASUREMENT_ORDER if row[f"has_{name}"] == "1"
        )
        assert route.native_feature == row["native_feature"]
        assert route.native_filename == row["native_filename"]
        assert route.kind.value == row["route_kind"]
        assert (route.native_state or "") == row["native_state"]
        assert route.measurements == expected_measurements
        assert route.include_source_id is (row["has_source_id"] == "1")


def test_only_sleep_has_many_to_one_activity_routing() -> None:
    destinations: dict[str, list[str]] = {}
    for route in ACTIVITY_ROUTES:
        destinations.setdefault(route.native_feature, []).append(route.raw_label)

    collisions = {
        feature: tuple(sorted(labels)) for feature, labels in destinations.items() if len(labels) > 1
    }
    assert collisions == {"Sleep": tuple(sorted(SLEEP_STATE_MAPPING))}


def test_sleep_state_mapping_is_lossless_and_apple_adjacent_not_apple_equivalent() -> None:
    assert dict(SLEEP_STATE_MAPPING) == {
        "sleep": "SLEEP",
        "sleep.awake": "AWAKE",
        "sleep.deep": "DEEP",
        "sleep.light": "LIGHT",
        "sleep.rem": "REM",
    }
    assert len(set(SLEEP_STATE_MAPPING.values())) == len(SLEEP_STATE_MAPPING)
    assert "ASLEEP" not in SLEEP_STATE_MAPPING.values()
    assert "CORE" not in SLEEP_STATE_MAPPING.values()

    sleep = get_feature_spec("Sleep")
    assert sleep.family is NativeFeatureFamily.SLEEP_STATE_INTERVAL
    assert sleep.state_columns == ("value",)
    assert sleep.activity_labels == tuple(sorted(SLEEP_STATE_MAPPING))
    assert "raw_activity_name" not in sleep.column_names
    assert "value" in sleep.column_names


def test_activity_schemas_are_fixed_per_feature_and_evidence_minimal() -> None:
    fixture = _activity_fixture()
    by_feature: dict[str, list[dict[str, str]]] = {}
    for row in fixture:
        by_feature.setdefault(row["native_feature"], []).append(row)

    for feature, rows in by_feature.items():
        spec = get_feature_spec(feature)
        expected_measurements = tuple(
            name for name in ACTIVITY_MEASUREMENT_ORDER if any(row[f"has_{name}"] == "1" for row in rows)
        )
        expected_source_id = any(row["has_source_id"] == "1" for row in rows)

        assert spec.source_feature == "Activities"
        assert spec.supported_versions == ("1.0",)
        assert spec.measurement_columns == expected_measurements
        assert ("source_id" in spec.column_names) is expected_source_id
        assert spec.occurrence_identity_columns == ("outer_id", "payload_index")
        assert spec.deduplicate_across_versions is False

        expected_prefix = ["start_date", "end_date"]
        if feature == "Sleep":
            expected_prefix.append("value")
        expected_prefix.extend(expected_measurements)
        assert list(spec.column_names[: len(expected_prefix)]) == expected_prefix


def test_activity_schema_does_not_repeat_reconstructable_or_path_identity_fields() -> None:
    forbidden = {
        "participant_id",
        "feature",
        "data_source",
        "raw_data_source",
        "raw_outer_feature",
        "raw_activity_name",
        "duration",
        "outer_datetime",
        "created_at",
        "updated_at",
        "source_month",
        "source_filename",
        "outer_row_number",
    }
    for feature in activity_feature_names():
        spec = get_feature_spec(feature)
        assert not (set(spec.column_names) & forbidden)
        omissions = {item.name: item for item in spec.omitted_fields}
        assert omissions["duration"].reconstructable_from == ("start_date", "end_date",)
        assert omissions["raw_activity_name"].retained_in in {"ACTIVITY_ROUTE_BY_LABEL", "SLEEP_STATE_MAPPING",}


def test_rare_activity_schemas_do_not_gain_unobserved_columns() -> None:
    assert "distance" not in get_feature_spec("RockClimbing").column_names
    assert "distance" not in get_feature_spec("GuidedBreathing").column_names
    assert "distance" not in get_feature_spec("Handball").column_names
    assert "distance" not in get_feature_spec("KickScooter").column_names
    assert "distance" not in get_feature_spec("RowingMachine").column_names
    assert "distance" not in get_feature_spec("VolleyballBeach").column_names

    assert "source_id" not in get_feature_spec("JumpRope").column_names
    assert "source_id" not in get_feature_spec("Kickboxing").column_names
    assert "source_id" not in get_feature_spec("VolleyballIndoor").column_names


def test_non_activity_schema_contracts_are_compact_and_version_aware() -> None:
    assert get_feature_spec("Calories").column_names == (
        "start_date",
        "end_date",
        "value",
        "source_day_label",
        "collecting_method_version",
        "outer_id",
    )
    assert get_feature_spec("Distance").column_names == (
        "start_date",
        "end_date",
        "value",
        "source_day_label",
        "collecting_method_version",
        "outer_id",
    )
    assert get_feature_spec("Weight").column_names == (
        "start_date",
        "end_date",
        "value",
        "source_day_label",
        "added_by",
        "collecting_method_version",
        "outer_id",
    )
    assert get_feature_spec("Height").column_names == (
        "start_date",
        "end_date",
        "value",
        "source_day_label",
        "added_by",
        "collecting_method_version",
        "outer_id",
    )
    assert get_feature_spec("Calories").supported_versions == ("1.0", "2.0")
    assert get_feature_spec("Distance").supported_versions == ("1.0", "2.0")
    assert get_feature_spec("Weight").supported_versions == ("1.0", "2.0")
    assert get_feature_spec("Height").supported_versions == ("2.0",)

    for name in ("Calories", "Distance", "Weight", "Height"):
        spec = get_feature_spec(name)
        assert "duration" not in spec.column_names
        assert "outer_datetime" not in spec.column_names
        assert "source_day_label" in spec.column_names
        assert spec.deduplicate_across_versions is False


def test_source_feature_version_contracts_cover_exactly_the_phase0_inventory() -> None:
    expected = {
        ("Activities", "1.0"),
        ("Calories", "1.0"),
        ("Calories", "2.0"),
        ("Distance", "1.0"),
        ("Distance", "2.0"),
        ("Height", "2.0"),
        ("StepCount", "1.0"),
        ("StepCount", "2.0"),
        ("Weight", "1.0"),
        ("Weight", "2.0"),
    }
    assert set(SOURCE_FEATURE_VERSION_BY_KEY) == expected
    assert get_source_feature_version_spec("StepCount", "1.0").envelope.value == "scalar_integer"
    assert get_source_feature_version_spec("Weight", "1.0").envelope.value == "dict"
    assert get_source_feature_version_spec("Weight", "2.0").envelope.value == "list"
    assert get_source_feature_version_spec("StepCount", "2.0").maximum_items == 3


def test_step_count_union_schema_preserves_all_distinct_source_concepts() -> None:
    step = get_feature_spec("StepCount")
    assert step.column_names == (
        "outer_datetime",
        "summary_date",
        "start_date",
        "end_date",
        "value",
        "raw_value",
        "aggregation_source",
        "origin_data_source_id",
        "app_name",
        "app_package_name",
        "device_manufacturer",
        "device_model",
        "device_type",
        "device_uid",
        "collecting_method_version",
        "outer_id",
        "payload_index",
    )
    assert step.measurement_columns == ("value", "raw_value")
    assert step.identity_for_version("1.0").maturity is (IdentityMaturity.SOURCE_OCCURRENCE_ONLY)
    assert step.identity_for_version("1.0").semantic_columns == ()
    assert step.identity_for_version("2.0").semantic_columns == (
        "collecting_method_version",
        "aggregation_source",
        "start_date",
        "end_date",
        "origin_data_source_id",
    )
    assert STEP_COUNT_DATA_TYPE_NAME == "com.google.step_count.delta"
    assert STEP_COUNT_DATA_SOURCE_ID == "derived:com.google.step_count.delta:com.google.android.gms:aggregated"
    assert "data_type_name" not in step.column_names
    assert "data_source_id" not in step.column_names


def test_registry_never_declares_cross_version_deduplication() -> None:
    assert all(
        spec.deduplicate_across_versions is False for spec in NATIVE_FEATURE_SPECS.values()
    )
    for spec in NATIVE_FEATURE_SPECS.values():
        assert set(spec.supported_versions) == {
            identity.version for identity in spec.semantic_identity_by_version
        }


def test_registry_does_not_assign_unsupported_units() -> None:
    payload = registry_payload()
    rendered = json.dumps(payload, sort_keys=True)
    assert '"raw_unit"' not in rendered
    assert '"canonical_unit"' not in rendered
    assert '"unit_policy"' not in rendered


def test_unregistered_activity_policy_is_controlled_and_lossless() -> None:
    policy = UNREGISTERED_ACTIVITY_POLICY
    assert policy.native_filename != "Activities.csv"
    assert policy.native_filename == "UnregisteredActivity.csv"
    assert policy.raw_label_column == "raw_activity_name"
    assert policy.emit_registry_warning is True
    assert policy.native_feature not in NATIVE_FEATURE_SPECS
    assert policy.column_names == (
        "start_date",
        "end_date",
        "raw_activity_name",
        "calories",
        "distance",
        "quantity",
        "tracked",
        "device",
        "source_name",
        "source_id",
        "collecting_method_version",
        "outer_id",
        "payload_index",
    )
    assert policy.validation_rules

    with pytest.raises(KeyError, match="Unregistered Google Fitness activity"):
        get_activity_route("sleep.restless")


def test_registry_fingerprint_and_payload_are_deterministic() -> None:
    first = registry_fingerprint()
    second = registry_fingerprint()
    assert first == second
    assert len(first) == 64
    assert first == EXPECTED_REGISTRY_FINGERPRINT

    payload = registry_payload()
    assert payload["registry_version"] == REGISTRY_VERSION
    assert payload["registry_fingerprint"] == first
    assert payload["validation"] == {"valid": True, "issues": []}
    assert payload["counts"] == {
        "activity_routes": 66,
        "activity_destinations": 62,
        "native_features": 67,
        "source_feature_versions": 10,
    }
    assert json.dumps(payload, sort_keys=True) == json.dumps(registry_payload(), sort_keys=True)


def test_registry_objects_and_mappings_are_immutable() -> None:
    route = get_activity_route("walking")
    with pytest.raises(FrozenInstanceError):
        setattr(route, "native_feature", "Other")
    with pytest.raises(TypeError):
        operator.setitem(ACTIVITY_ROUTE_BY_LABEL, "new", route)
    with pytest.raises(TypeError):
        operator.setitem(
            NATIVE_FEATURE_SPECS, "New", get_feature_spec("Walking"),
        )


def test_source_normalization_is_explicit_and_non_google_labels_do_not_match() -> None:
    assert normalize_source_label("GoogleFitness") == NORMALIZED_RAW_SOURCE
    assert normalize_source_label(" Google Fitness ") == NORMALIZED_RAW_SOURCE
    assert normalize_source_label("GOOGLE-FITNESS") == NORMALIZED_RAW_SOURCE
    assert normalize_source_label("AppleHealthkit") != NORMALIZED_RAW_SOURCE


def test_google_registry_import_does_not_import_apple_scientific_modules() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import wearable_project.processing.google_fitness.registry; "
            "assert not any(name.startswith('wearable_project.processing.apple_healthkit') "
            "for name in sys.modules)"
        ),
    ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_feature_lookup_surface_is_complete_and_stable() -> None:
    assert feature_names() == tuple(sorted(feature_names()))
    assert set(feature_names()) == set(NATIVE_FEATURE_SPECS)
    assert set(activity_feature_names()).issubset(feature_names())
    assert set(feature_names()) - set(activity_feature_names()) == {
        "Calories", "Distance", "Height", "StepCount", "Weight",
    }


def test_every_feature_declares_validation_and_omission_contracts() -> None:
    for spec in NATIVE_FEATURE_SPECS.values():
        assert spec.validation_rules
        omissions = {item.name: item for item in spec.omitted_fields}
        assert "participant_id" in omissions
        assert "feature" in omissions
        assert "data_source" in omissions
        assert "raw_data_source" in omissions
        assert "raw_outer_feature" in omissions
        assert "created_at" in omissions
        assert "updated_at" in omissions
        assert "source_month" in omissions
        assert "source_filename" in omissions
        assert "outer_row_number" in omissions


def test_singleton_payload_features_omit_redundant_payload_index() -> None:
    for feature in ("Calories", "Distance", "Weight", "Height"):
        spec = get_feature_spec(feature)
        assert "payload_index" not in spec.column_names
        omissions = {item.name: item for item in spec.omitted_fields}
        assert omissions["payload_index"].retained_in == "SourceFeatureVersionSpec cardinality"

    assert "payload_index" in get_feature_spec("StepCount").column_names
    assert "payload_index" in get_feature_spec("Walking").column_names
