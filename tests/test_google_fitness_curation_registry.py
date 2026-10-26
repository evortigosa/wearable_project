"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
from dataclasses import FrozenInstanceError
import inspect
import json
import operator
from pathlib import Path
import pytest
from wearable_project.curation.google_fitness.decisions import (
    DECISIONS,
    EXPECTED_DECISIONS_FINGERPRINT,
    decisions_fingerprint,
    decisions_payload,
    get_decision,
    known_decisions,
    validate_decisions,
)
from wearable_project.curation.google_fitness.evidence import (
    EVIDENCE,
    EXPECTED_EVIDENCE_FINGERPRINT,
    evidence_fingerprint,
    evidence_payload,
    get_evidence,
    known_evidence,
    validate_evidence,
)
from wearable_project.curation.google_fitness.guidance import (
    EXPECTED_GUIDANCE_FINGERPRINT,
    FEATURE_GUIDES,
    get_feature_guide,
    guidance_fingerprint,
    guidance_payload,
    known_guides,
    validate_guidance,
)
from wearable_project.curation.google_fitness.models import (
    DecisionStatus,
    EventKind,
    GroupPreferenceStrategy,
    InclusionPolicy,
    PolicyExecutionMode,
    PolicyMaturity,
    RawUnitStatus,
    SourceDayLabelTreatment,
)
from wearable_project.curation.google_fitness.registry import (
    CURATION_POLICIES,
    EXPECTED_REGISTRY_FINGERPRINT,
    UNKNOWN_POLICY,
    get_policy,
    known_curation_features,
    matrix_csv,
    matrix_rows,
    registry_fingerprint,
    registry_payload,
    validate_registry,
)
from wearable_project.curation.google_fitness.rules import (
    EXPECTED_RULES_FINGERPRINT,
    RULES,
    get_rule,
    known_rules,
    rules_fingerprint,
    rules_payload,
    validate_rules,
)
from wearable_project.processing.google_fitness.registry import NATIVE_FEATURE_SPECS, SLEEP_STATE_MAPPING


def test_cwp1_catalog_counts_and_fingerprints_are_frozen() -> None:
    assert len(CURATION_POLICIES) == 67
    assert len(FEATURE_GUIDES) == 67
    assert len(EVIDENCE) == 24
    assert len(RULES) == 45
    assert len(DECISIONS) == 24

    assert registry_fingerprint() == EXPECTED_REGISTRY_FINGERPRINT
    assert evidence_fingerprint() == EXPECTED_EVIDENCE_FINGERPRINT
    assert rules_fingerprint() == EXPECTED_RULES_FINGERPRINT
    assert decisions_fingerprint() == EXPECTED_DECISIONS_FINGERPRINT
    assert guidance_fingerprint() == EXPECTED_GUIDANCE_FINGERPRINT


def test_all_catalogs_validate() -> None:
    assert validate_registry().valid, validate_registry().issues
    assert validate_evidence() == ()
    assert validate_rules() == ()
    assert validate_decisions() == ()
    assert validate_guidance() == ()


def test_policy_feature_set_exactly_matches_native_registry() -> None:
    assert tuple(CURATION_POLICIES) == tuple(NATIVE_FEATURE_SPECS)
    assert "Activities" not in CURATION_POLICIES
    assert set(FEATURE_GUIDES) == set(CURATION_POLICIES)


def test_all_native_schema_contracts_are_exact() -> None:
    for feature, policy in CURATION_POLICIES.items():
        native = NATIVE_FEATURE_SPECS[feature]
        schema = policy.native_schema
        assert schema.native_columns == native.column_names
        assert schema.required_columns == native.required_columns
        assert schema.measurement_columns == native.measurement_columns
        assert schema.state_columns == native.state_columns
        assert schema.supported_versions == native.supported_versions
        assert policy.measurement_columns == native.measurement_columns


def test_all_policies_enforce_non_destructive_output() -> None:
    for feature, policy in CURATION_POLICIES.items():
        assert policy.output.preserve_native_prefix, feature
        assert policy.output.preserve_row_order, feature
        assert policy.output.preserve_row_count, feature
        assert not set(policy.materialized_columns) & set(policy.native_schema.native_columns)
        assert not set(policy.virtual_columns) & set(policy.native_schema.native_columns)
        assert set(policy.materialized_columns).isdisjoint(policy.virtual_columns)
        assert set(policy.output.fixed_columns).isdisjoint(policy.output.sparse_columns)
        assert set(policy.output.fixed_columns) | set(policy.output.sparse_columns) == set(policy.materialized_columns)


def test_policy_family_counts_are_exact() -> None:
    assert sum(policy.event_kind is EventKind.ACTIVITY_INTERVAL for policy in CURATION_POLICIES.values()) == 61
    assert sum(policy.event_kind is EventKind.STATE_INTERVAL for policy in CURATION_POLICIES.values()) == 1
    assert sum(policy.event_kind is EventKind.INTERVAL_TOTAL for policy in CURATION_POLICIES.values()) == 2
    assert sum(policy.event_kind is EventKind.POINT_OR_INTERVAL_SCALAR for policy in CURATION_POLICIES.values()) == 2
    assert sum(policy.event_kind is EventKind.MULTI_REPRESENTATION_COUNT for policy in CURATION_POLICIES.values()) == 1


def test_activity_policies_do_not_force_scalar_canonical_value() -> None:
    for feature, policy in CURATION_POLICIES.items():
        if policy.event_kind not in {EventKind.ACTIVITY_INTERVAL, EventKind.STATE_INTERVAL}:
            continue
        assert "canonical_value" not in policy.materialized_columns
        assert "tracking_status" not in policy.materialized_columns
        assert "tracking_status" in policy.virtual_columns
        assert "derivation_method" in policy.virtual_columns
        assert policy.materialized_columns == (
            "acquisition_method",
            "curation_status",
            "curation_flags",
            "include_by_default",
        )
        for measurement in policy.measurements:
            assert measurement.column in {"calories", "distance", "quantity"}
            assert not measurement.canonical_projection_enabled
            for unit in measurement.unit_by_version:
                assert unit.execution_mode is PolicyExecutionMode.BLOCKED_DERIVATION


def test_activity_quantity_is_explicitly_unresolved() -> None:
    policies_with_quantity = [
        policy for policy in CURATION_POLICIES.values() if "quantity" in policy.measurement_columns
    ]
    assert policies_with_quantity
    for policy in policies_with_quantity:
        quantity = next(item for item in policy.measurements if item.column == "quantity")
        assert not quantity.raw_projection_enabled
        assert not quantity.canonical_projection_enabled
        assert quantity.unit_by_version[0].raw_unit_status is RawUnitStatus.UNRESOLVED
        assert quantity.unit_by_version[0].canonical_unit is None
        assert "activity_quantity_unresolved" in policy.curation.rule_ids


def test_sleep_policy_preserves_native_source_vocabulary() -> None:
    policy = get_policy("Sleep")
    assert policy.native_schema.state_columns == ("value",)
    assert policy.native_schema.state_vocabulary == tuple(sorted(set(SLEEP_STATE_MAPPING.values())))
    assert policy.native_schema.state_vocabulary == ("AWAKE", "DEEP", "LIGHT", "REM", "SLEEP")
    assert policy.grouping.strategy is GroupPreferenceStrategy.SLEEP_OVERLAP_CONTEXT
    assert "sleep_no_apple_equivalence" in policy.curation.rule_ids
    assert "CORE" not in policy.native_schema.state_vocabulary
    assert "ASLEEP" not in policy.native_schema.state_vocabulary


def test_outer_scalar_units_remain_exporter_mapping_pending() -> None:
    expected = {
        "Calories": "kcal",
        "Distance": "m",
        "Weight": "kg",
        "Height": "m",
    }
    for feature, canonical_unit in expected.items():
        policy = get_policy(feature)
        measurement = policy.measurements[0]
        assert not measurement.canonical_projection_enabled
        for unit in measurement.unit_by_version:
            assert unit.raw_unit_status is RawUnitStatus.EXPORTER_MAPPING_PENDING
            assert unit.canonical_unit == canonical_unit
            assert unit.execution_mode is PolicyExecutionMode.CONSERVATIVE_ANNOTATION
        assert "unit_mapping_pending" in policy.curation.rule_ids


def test_step_count_unit_evidence_is_version_specific() -> None:
    policy = get_policy("StepCount")
    value = next(item for item in policy.measurements if item.column == "value")
    v1 = value.policy_for_version("1.0")
    v2 = value.policy_for_version("2.0")
    assert v1.raw_unit_status is RawUnitStatus.SOURCE_CONVENTION
    assert v1.execution_mode is PolicyExecutionMode.SOURCE_SPECIFIC_EXECUTION
    assert v1.canonical_unit == "count"
    assert v2.raw_unit_status is RawUnitStatus.PAYLOAD_EXPLICIT
    assert v2.execution_mode is PolicyExecutionMode.REVIEWED_EXECUTION
    assert v2.canonical_unit == "count"

    raw_value = next(item for item in policy.measurements if item.column == "raw_value")
    assert raw_value.policy_for_version("1.0").raw_unit_status is RawUnitStatus.NOT_APPLICABLE
    assert raw_value.policy_for_version("2.0").raw_unit_status is RawUnitStatus.PAYLOAD_EXPLICIT


def test_step_count_group_policy_preserves_all_rows_and_separates_inclusion() -> None:
    policy = get_policy("StepCount")
    assert policy.grouping.group_columns == ("collecting_method_version", "outer_id")
    assert policy.grouping.applies_to_versions == ("2.0",)
    assert policy.grouping.strategy is GroupPreferenceStrategy.STEP_COUNT_V2_SOURCE_PREFERENCE
    assert policy.grouping.role_column == "step_count_role"
    assert policy.curation.default_inclusion is InclusionPolicy.GROUP_PREFERENCE
    assert "step_count_role" in policy.materialized_columns
    assert "canonical_value" in policy.virtual_columns
    assert "canonical_unit" in policy.virtual_columns
    assert "derivation_method" in policy.virtual_columns
    assert "include_by_default" in policy.materialized_columns
    assert "curation_status" in policy.materialized_columns
    assert policy.temporal.canonical_date_derivation_allowed is False
    assert policy.temporal.source_day_label_treatment is SourceDayLabelTreatment.NOT_PRESENT


def test_step_count_preference_is_accepted_and_non_destructive() -> None:
    decision = DECISIONS["step_count_preference"]
    assert decision.status is DecisionStatus.ACCEPTED
    assert "stepcount_preference_audit" in decision.required_audits
    assert "drop" not in decision.selected_option.casefold()
    policy = get_policy("StepCount")
    assert policy.output.preserve_row_count
    assert policy.calibration.maturity is PolicyMaturity.PROVISIONAL


def test_calories_negative_buckets_are_explicit_and_non_overlapping() -> None:
    tiny = dict(RULES["calories_negative_tiny"].parameters)
    small = dict(RULES["calories_negative_small"].parameters)
    material = dict(RULES["calories_negative_material"].parameters)
    extreme = dict(RULES["calories_negative_extreme"].parameters)
    assert tiny == {"absolute_max": 0.001}
    assert small == {"absolute_max": 1.0, "absolute_min_exclusive": 0.001}
    assert material == {"absolute_max": 100.0, "absolute_min_exclusive": 1.0}
    assert extreme == {"absolute_min_exclusive": 100.0}
    assert DECISIONS["negative_calories_buckets"].status is DecisionStatus.ACCEPTED


def test_weight_policy_preserves_nonpoint_intervals() -> None:
    policy = get_policy("Weight")
    assert policy.temporal.start_column == "start_date"
    assert policy.temporal.end_column == "end_date"
    assert policy.temporal.preserve_all_boundaries
    assert "weight_nonpoint_interval" in policy.curation.rule_ids
    assert "weight_long_interval" in policy.curation.rule_ids
    decision = DECISIONS["weight_nonpoint_preserved"]
    assert decision.status is DecisionStatus.ACCEPTED
    assert "collapse" in " ".join(decision.rejected_options).casefold() or "overwrite" in " ".join(decision.rejected_options).casefold()


def test_source_day_label_is_context_only() -> None:
    for feature in ("Calories", "Distance", "Weight", "Height"):
        policy = get_policy(feature)
        assert policy.temporal.source_day_label_column == "source_day_label"
        assert policy.temporal.source_day_label_treatment is SourceDayLabelTreatment.CONTEXT_ONLY
        assert not policy.temporal.canonical_date_derivation_allowed
        assert "source_day_label_context" in policy.curation.rule_ids


def test_provenance_is_multidimensional() -> None:
    activity = get_policy("Walking")
    assert activity.provenance.output_dimensions == (
        "acquisition_method",
        "derivation_method",
        "tracking_status",
    )
    step = get_policy("StepCount")
    assert step.provenance.output_dimensions == (
        "acquisition_method",
        "derivation_method",
        "step_count_role",
    )
    assert DECISIONS["provenance_dimensions"].status is DecisionStatus.ACCEPTED
    assert DECISIONS["provenance_classifier"].status is DecisionStatus.PROVISIONAL


def test_unknown_feature_fallback_is_safe() -> None:
    assert get_policy("FutureFeature") is UNKNOWN_POLICY
    with pytest.raises(KeyError):
        get_policy("FutureFeature", allow_fallback=False)
    assert UNKNOWN_POLICY.unknown_fallback
    assert UNKNOWN_POLICY.curation.default_status.value == "review"
    assert UNKNOWN_POLICY.curation.default_inclusion is InclusionPolicy.EXCLUDE
    assert "unknown_feature_policy" in UNKNOWN_POLICY.curation.rule_ids
    assert not UNKNOWN_POLICY.measurements


def test_all_policy_references_resolve() -> None:
    for feature, policy in CURATION_POLICIES.items():
        assert set(policy.curation.rule_ids) <= set(RULES), feature
        assert set(policy.calibration.decision_ids) <= set(DECISIONS), feature
        evidence_refs = set(policy.evidence_refs) | set(policy.provenance.evidence_refs)
        for measurement in policy.measurements:
            for unit in measurement.unit_by_version:
                evidence_refs.update(unit.evidence_refs)
        assert evidence_refs <= set(EVIDENCE), feature


def test_unit_policies_cover_each_supported_version_once() -> None:
    for feature, policy in CURATION_POLICIES.items():
        versions = set(policy.native_schema.supported_versions)
        for measurement in policy.measurements:
            observed: list[str] = []
            for unit in measurement.unit_by_version:
                observed.extend(unit.versions)
            assert set(observed) == versions, (feature, measurement.column)
            assert len(observed) == len(set(observed)), (feature, measurement.column)


def test_all_current_policies_are_provisional_pending_cwp2() -> None:
    assert {
        policy.calibration.maturity for policy in CURATION_POLICIES.values()
    } == {PolicyMaturity.PROVISIONAL}
    for policy in CURATION_POLICIES.values():
        assert policy.calibration.required_audits


def test_evidence_catalog_distinguishes_contract_from_exporter_mapping() -> None:
    for evidence_id in (
        "google_fit_calories_expended",
        "google_fit_distance_delta",
        "google_fit_weight",
        "google_fit_height",
    ):
        item = EVIDENCE[evidence_id]
        assert item.url and item.url.startswith("https://developers.google.com/fit/")
        assert any("exporter" in limitation.casefold() for limitation in item.limitations)
    assert EVIDENCE["google_fit_step_count_delta"].url
    assert EVIDENCE["project_phase1_native_acceptance"].artifact


def test_decision_scaffold_separates_accepted_and_provisional_work() -> None:
    accepted = {name for name, item in DECISIONS.items() if item.status is DecisionStatus.ACCEPTED}
    provisional = {name for name, item in DECISIONS.items() if item.status is DecisionStatus.PROVISIONAL}
    assert {
        "non_destructive_curation",
        "status_inclusion_separate",
        "sleep_native_states_preserved",
        "step_count_preserve_all_rows",
        "step_count_no_canonical_day",
        "weight_nonpoint_preserved",
    } <= accepted
    assert {
        "step_count_preference",
        "negative_calories_buckets",
    } <= accepted
    assert {
        "calories_unit_mapping",
        "distance_unit_mapping",
        "weight_unit_mapping",
        "height_unit_mapping",
        "provenance_classifier",
    } <= provisional
    assert all(DECISIONS[name].required_audits for name in provisional)


def test_guidance_covers_all_features_and_states_limitations() -> None:
    assert set(FEATURE_GUIDES) == set(CURATION_POLICIES)
    for feature, guide in FEATURE_GUIDES.items():
        assert guide.feature == feature
        assert guide.row_meaning
        assert guide.recommended_use
        assert guide.default_projection
        assert guide.limitations
    sleep = FEATURE_GUIDES["Sleep"]
    rendered = " ".join(sleep.limitations)
    assert "Apple" in rendered
    assert "hypnogram" in rendered


def test_registry_payload_and_matrix_are_deterministic_and_json_serializable() -> None:
    first = registry_payload()
    second = registry_payload()
    assert first == second
    json.dumps(first, sort_keys=True)
    rows = matrix_rows()
    assert len(rows) == 67
    assert [row["feature"] for row in rows] == list(CURATION_POLICIES)
    assert {row["execution_mode"] for row in rows} == {
        "conservative_annotation",
        "source_specific_execution",
    }


def test_registry_objects_and_mappings_are_immutable() -> None:
    policy = CURATION_POLICIES["Walking"]
    with pytest.raises(FrozenInstanceError):
        setattr(policy, "feature", "Other")
    with pytest.raises(TypeError):
        operator.setitem(CURATION_POLICIES, "Other", policy)
    with pytest.raises(TypeError):
        operator.setitem(EVIDENCE, "other", next(iter(EVIDENCE.values())))
    with pytest.raises(TypeError):
        operator.setitem(RULES, "other", next(iter(RULES.values())))
    with pytest.raises(TypeError):
        operator.setitem(DECISIONS, "other", next(iter(DECISIONS.values())))


def test_google_curation_modules_do_not_import_apple_scientific_policy() -> None:
    package = Path(inspect.getfile(get_policy)).parent
    for path in sorted(package.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "wearable_project.curation.apple_healthkit" not in text, path.name
        assert "from ..apple_healthkit" not in text, path.name


def test_cwp4_registers_google_curation_cli_but_not_combined_run() -> None:
    from wearable_project.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "curate", "google-fitness", "--input-native", "x", "--output", "y"
    ])
    assert args.source == "google-fitness"
    assert args._handler.__module__ == "wearable_project.curation.cli"
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "google-fitness"])


def test_catalog_payloads_and_lookup_helpers_are_complete() -> None:
    assert known_curation_features() == tuple(CURATION_POLICIES)
    assert known_evidence() == tuple(EVIDENCE)
    assert known_rules() == tuple(RULES)
    assert known_decisions() == tuple(DECISIONS)
    assert known_guides() == tuple(FEATURE_GUIDES)

    assert get_evidence("google_fit_weight") is EVIDENCE["google_fit_weight"]
    assert get_rule("weight_nonpoint_interval") is RULES["weight_nonpoint_interval"]
    assert get_decision("weight_nonpoint_preserved") is DECISIONS["weight_nonpoint_preserved"]
    assert get_feature_guide("Weight") is FEATURE_GUIDES["Weight"]

    payloads = (evidence_payload(), rules_payload(), decisions_payload(), guidance_payload())
    for payload in payloads:
        assert payload["count"] > 0
        assert payload["validation_issues"] == []
        json.dumps(payload, sort_keys=True)

    matrix = matrix_csv()
    assert matrix.startswith("feature,event_kind,versions,")
    assert matrix.count("\n") == 68


def test_lookup_helpers_reject_unknown_identifiers() -> None:
    with pytest.raises(KeyError):
        get_evidence("missing")
    with pytest.raises(KeyError):
        get_rule("missing")
    with pytest.raises(KeyError):
        get_decision("missing")
    with pytest.raises(KeyError):
        get_feature_guide("missing")
    with pytest.raises(KeyError):
        get_policy("missing", allow_fallback=False)


def test_measurement_policy_rejects_unsupported_version() -> None:
    measurement = get_policy("StepCount").measurements[0]
    with pytest.raises(KeyError):
        measurement.policy_for_version("3.0")
