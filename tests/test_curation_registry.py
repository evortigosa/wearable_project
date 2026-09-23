"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import csv
import hashlib
import io
import json
from pathlib import Path
import pytest
from wearable_project.curation.models import CurationStatus, InclusionPolicy, PolicyMaturity, ResamplingSupport
from wearable_project.curation.registry import (
    COHORT_OBSERVED_FEATURES, CURATION_POLICIES, CURATION_REGISTRY_VERSION, UNKNOWN_POLICY,
    get_policy, known_curation_features, matrix_csv, registry_fingerprint, registry_payload, validate_registry,
)
from wearable_project.curation.models import EvidenceGrade, PolicyExecutionMode
from wearable_project.curation.environment import environment_manifest
from wearable_project.utils.release_manifest import EXPECTED_CORE_PROCESSING_MODULE_SHA256


def test_registry_covers_exactly_the_33_full_cohort_features() -> None:
    assert len(COHORT_OBSERVED_FEATURES) == 33
    assert len(CURATION_POLICIES) == 33
    assert set(known_curation_features()) == set(COHORT_OBSERVED_FEATURES)


def test_registry_validation_passes() -> None:
    result = validate_registry()
    assert result.valid, result.errors
    assert not result.errors
    assert result.warnings  # provisional evidence is intentionally visible


def test_all_cohort_policies_have_complete_contracts() -> None:
    for name, policy in CURATION_POLICIES.items():
        assert policy.name == name
        assert policy.identity.cohort_observed
        assert policy.identity.maturity is not PolicyMaturity.UNKNOWN
        assert policy.schema.feature_family
        assert policy.schema.accepted_payload_shapes
        assert policy.semantics.native_resolution
        assert policy.provenance.acquisition_method_strategy
        assert policy.reconciliation.occurrence_identity_strategy
        assert policy.curation.rule_ids
        assert policy.output.column_order_prefix
        assert policy.resampling.strategy
        assert policy.resampling.default_enabled is False
        assert policy.tests.fixture_ids
        assert policy.tests.invariants
        assert len(policy.fingerprint()) == 64


def test_unknown_feature_fallback_is_safe_and_non_destructive() -> None:
    policy = get_policy("FutureHealthKitFeature")
    assert policy.name == "FutureHealthKitFeature"
    assert policy.identity.maturity is PolicyMaturity.UNKNOWN
    assert policy.curation.default_status is CurationStatus.REVIEW
    assert policy.curation.default_inclusion is InclusionPolicy.EXCLUDE
    assert policy.resampling.support is ResamplingSupport.UNSUPPORTED
    assert policy.units.measurements[0].canonical_unit is None
    assert policy.units.measurements[0].conversion_rule == "not_applicable"


def test_sleep_policy_has_no_numeric_or_lexical_state_reducer() -> None:
    policy = get_policy("Sleep", allow_fallback=False)
    assert "finite_numeric" not in policy.curation.rule_ids
    assert policy.resampling.strategy == "state_interval_union"
    assert policy.resampling.state_handling == "multi_state_interval_union_and_coverage"
    assert policy.resampling.aggregation == "none"
    assert any("no_lexical_state_selection" == item for item in policy.tests.invariants)


def test_blood_pressure_components_remain_coupled() -> None:
    policy = get_policy("BloodPressure", allow_fallback=False)
    roles = {field.role for field in policy.schema.measurements}
    assert roles == {"systolic_pressure", "diastolic_pressure"}
    assert policy.semantics.measurement_kind.value == "multivariate_point"
    assert "no_componentwise_synthetic_pair_is_created" in policy.tests.invariants


def test_ecg_policy_is_waveform_native_only() -> None:
    policy = get_policy("Electrocardiogram", allow_fallback=False)
    assert policy.schema.waveform_columns == ("voltage_measurements",)
    assert policy.resampling.support is ResamplingSupport.WAVEFORM_SPECIFIC_ONLY
    assert policy.resampling.broadcasting == "prohibited"


def test_cgm_policy_retains_status_trend_and_timezone() -> None:
    policy = get_policy("BloodGlucose", allow_fallback=False)
    expected = {"status", "trend_arrow", "trend_rate", "time_zone", "metadata_device_name"}
    assert expected.issubset(set(policy.schema.context_columns))
    assert expected.issubset(set(policy.output.retained_context_columns))


def test_unit_policies_do_not_silently_guess_unresolved_epochs() -> None:
    unresolved = {
        "Weight": "resolved_mass_to_kilograms",
        "LeanBodyMass": "resolved_mass_to_kilograms",
        "Height": "resolved_length_to_metres",
        "BloodGlucose": "resolved_glucose_to_mmol_l",
        "EnergyConsumed": "resolved_energy_to_kcal",
    }
    for feature, strategy in unresolved.items():
        unit = get_policy(feature, allow_fallback=False).units.measurements[0]
        assert unit.conversion_rule == strategy
        assert unit.raw_unit is None

    reviewed = {
        "WaistCircumference": "inches_to_metres",
        "BodyTemperature": "identity",
        "DailyDistanceCycling": "identity",
        "HeartRateVariability": "seconds_to_milliseconds",
        "BloodAlcoholContent": "fraction_to_percent",
    }
    for feature, strategy in reviewed.items():
        unit = get_policy(feature, allow_fallback=False).units.measurements[0]
        assert unit.conversion_rule == strategy
        assert unit.raw_unit is not None


def test_activity_summary_remains_ambiguous_and_excluded_by_default() -> None:
    policy = get_policy("ActivitySummary", allow_fallback=False)
    assert policy.identity.maturity is PolicyMaturity.PROVISIONAL
    assert "activity_summary_date_ambiguity" in policy.curation.rule_ids
    assert policy.curation.default_status is CurationStatus.REVIEW
    assert policy.curation.default_inclusion is InclusionPolicy.EXCLUDE
    assert policy.resampling.strategy == "daily_context_only"
    assert policy.resampling.broadcasting == "prohibited_by_default"


def test_bac_has_source_specific_calculator_policy() -> None:
    policy = get_policy("BloodAlcoholContent", allow_fallback=False)
    calculator = [rule for rule in policy.provenance.source_rules if rule.source_id == "com.rwichmann.intellidrink-lite"]
    assert len(calculator) == 1
    assert calculator[0].acquisition_method.value == "calculator_estimate"
    assert calculator[0].inclusion_effect is InclusionPolicy.EXCLUDE


def test_registry_fingerprint_and_serialization_are_deterministic() -> None:
    first = registry_fingerprint()
    second = registry_fingerprint()
    assert first == second
    payload = registry_payload(include_evidence=True, include_rules=True)
    rendered = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    assert CURATION_REGISTRY_VERSION in rendered
    assert len(payload["policies"]) == 33
    assert payload["validation"]["valid"] is True


def test_matrix_csv_contains_one_row_per_feature() -> None:
    rows = list(csv.DictReader(io.StringIO(matrix_csv())))
    assert len(rows) == 33
    assert {row["feature"] for row in rows} == set(COHORT_OBSERVED_FEATURES)
    assert all(row["resampling_default_enabled"] == "False" for row in rows)


def test_milestone_one_processing_modules_are_byte_frozen() -> None:
    processing = Path(__file__).parents[1] / "wearable_project" / "processing"
    actual = {
        name: hashlib.sha256((processing / name).read_bytes()).hexdigest()
        for name in EXPECTED_CORE_PROCESSING_MODULE_SHA256
    }
    assert actual == EXPECTED_CORE_PROCESSING_MODULE_SHA256
    # RC2 adds one read-only operational utility
    assert (processing / "scan.py").is_file()


def test_discrete_heart_features_allow_point_or_short_interval_support() -> None:
    for feature in ("HeartRate", "HeartRateVariability"):
        policy = get_policy(feature, allow_fallback=False)
        assert policy.semantics.event_kind.value == "point_or_interval"
        assert policy.semantics.duration_model.value == "zero_or_explicit_interval"
        assert "interval_duration_positive" not in policy.curation.rule_ids


def test_ratio_features_are_declared_as_ratios() -> None:
    for feature in ("BodyFatPercentage", "OxygenSaturation", "BloodAlcoholContent"):
        assert get_policy(feature, allow_fallback=False).semantics.measurement_kind.value == "ratio"


def test_provisional_policies_have_explicit_execution_gates() -> None:
    for policy in CURATION_POLICIES.values():
        calibration = policy.calibration
        assert calibration.rationale
        assert calibration.safe_fallback
        if policy.identity.maturity is PolicyMaturity.PROVISIONAL:
            assert calibration.execution_mode is not PolicyExecutionMode.REVIEWED_EXECUTION
            assert calibration.required_audits
        if calibration.evidence_grade is EvidenceGrade.D_UNRESOLVED:
            assert calibration.execution_mode is PolicyExecutionMode.BLOCK_DERIVATION


def test_unknown_policy_blocks_derivation() -> None:
    assert UNKNOWN_POLICY.calibration.evidence_grade is EvidenceGrade.D_UNRESOLVED
    assert UNKNOWN_POLICY.calibration.execution_mode is PolicyExecutionMode.BLOCK_DERIVATION


def test_release_manifest_matches_curation_sources_and_fingerprints() -> None:
    manifest = environment_manifest()
    assert manifest.registry_fingerprint == registry_fingerprint()
    assert all(manifest.module_integrity.values())
    assert not [warning for warning in manifest.warnings if "release manifest" in warning]
