"""
Wearable Data Processing and Modeling project
Complete policy registry for Google Fitness curation. The registry covers every accepted native feature while
remaining independent of Apple HealthKit scientific policy. It is policy-only: no native or curated file is
read or written here.
"""


from __future__ import annotations
import csv
from dataclasses import replace
from hashlib import sha256
import io
import json
from types import MappingProxyType
from typing import Iterable, Mapping
from wearable_project.processing.google_fitness.registry import (
    NATIVE_FEATURE_SPECS, SLEEP_STATE_MAPPING, SOURCE_KEY, NativeFeatureFamily, feature_names as native_feature_names,
)
from .decisions import DECISIONS
from .evidence import EVIDENCE
from .models import (
    CanonicalizationStrategy,
    CurationPolicy,
    CurationStatus,
    DateAnchor,
    EventKind,
    EvidenceGrade,
    FeaturePolicy,
    GroupPolicy,
    GroupPreferenceStrategy,
    InclusionPolicy,
    MeasurementKind,
    MeasurementPolicy,
    NativeSchemaPolicy,
    OutputPolicy,
    PolicyCalibration,
    PolicyExecutionMode,
    PolicyMaturity,
    ProvenancePolicy,
    RawUnitStatus,
    RegistryValidationResult,
    ResamplingPolicy,
    ResamplingSupport,
    SourceDayLabelTreatment,
    TemporalPolicy,
    VersionUnitPolicy,
    to_primitive,
)
from .rules import RULES


CURATION_REGISTRY_VERSION = "2026-10-google-fitness-curation-schema-v2"
POLICY_CONTRACT_VERSION = "2.0.0"
EXPECTED_REGISTRY_FINGERPRINT = "899a973ae17795b3d519d6fb581d18bcc3b0a3773265845ad43bb84d3ca75d7c"

MATERIALIZED_STATUS_COLUMNS = (
    "curation_status",
    "curation_flags",
    "include_by_default",
)
MATERIALIZED_PROVENANCE_COLUMNS = (
    "acquisition_method",
)
VIRTUAL_PROVENANCE_COLUMNS = (
    "derivation_method",
)
VIRTUAL_ACTIVITY_COLUMNS = (
    "derivation_method",
    "tracking_status",
)
VIRTUAL_UNIT_COLUMNS = (
    "canonical_value",
    "canonical_unit",
    "curation_unit_status",
    "unit_evidence",
)
STEP_COUNT_MATERIALIZED_COLUMNS = (
    "acquisition_method",
    "step_count_role",
    "curation_status",
    "curation_flags",
    "include_by_default",
)
STEP_COUNT_VIRTUAL_COLUMNS = (
    "canonical_value",
    "canonical_unit",
    "curation_unit_status",
    "unit_evidence",
    "derivation_method",
)


def _native_schema(feature: str, row_meaning: str) -> NativeSchemaPolicy:
    spec = NATIVE_FEATURE_SPECS[feature]
    return NativeSchemaPolicy(
        feature=feature,
        native_columns=spec.column_names,
        required_columns=spec.required_columns,
        measurement_columns=spec.measurement_columns,
        state_columns=spec.state_columns,
        state_vocabulary=tuple(sorted(set(SLEEP_STATE_MAPPING.values()))) if feature == "Sleep" else (),
        supported_versions=spec.supported_versions,
        row_meaning=row_meaning,
    )


def _unit(
    versions: tuple[str, ...], *, status: RawUnitStatus, raw_unit: str | None, canonical_unit: str | None,
    grade: EvidenceGrade, mode: PolicyExecutionMode, strategy: CanonicalizationStrategy, evidence: tuple[str, ...],
    limitations: tuple[str, ...] = (),
) -> VersionUnitPolicy:
    return VersionUnitPolicy(
        versions=versions,
        raw_unit_status=status,
        raw_unit=raw_unit,
        canonical_unit=canonical_unit,
        evidence_grade=grade,
        execution_mode=mode,
        canonicalization=strategy,
        evidence_refs=evidence,
        limitations=limitations,
    )


def _measurement(
    column: str, kind: MeasurementKind, required: bool, *, raw_projection: bool, canonical_projection: bool,
    units: tuple[VersionUnitPolicy, ...], description: str,
) -> MeasurementPolicy:
    return MeasurementPolicy(
        column=column,
        kind=kind,
        required=required,
        raw_projection_enabled=raw_projection,
        canonical_projection_enabled=canonical_projection,
        unit_by_version=units,
        description=description,
    )


def _output(
    materialized: tuple[str, ...], *, virtual: tuple[str, ...] = (), fixed: tuple[str, ...] = (),
    sparse: tuple[str, ...] | None = None,
) -> OutputPolicy:
    return OutputPolicy(
        preserve_native_prefix=True,
        preserve_row_order=True,
        preserve_row_count=True,
        materialized_columns=materialized,
        virtual_columns=virtual,
        sparse_columns=materialized if sparse is None else sparse,
        fixed_columns=fixed,
    )


def _curation(
    rules: tuple[str, ...], *, status: CurationStatus = CurationStatus.PASS,
    inclusion: InclusionPolicy = InclusionPolicy.RULE_DRIVEN,
) -> CurationPolicy:
    return CurationPolicy(
        default_status=status,
        default_inclusion=inclusion,
        rule_ids=rules,
        flags_column="curation_flags",
        status_column="curation_status",
        inclusion_column="include_by_default",
    )


def _activity_measurement_policy(column: str) -> MeasurementPolicy:
    if column == "calories":
        return _measurement(
            column,
            MeasurementKind.EXTENSIVE_TOTAL,
            False,
            raw_projection=True,
            canonical_projection=False,
            units=(
                _unit(
                    ("1.0",),
                    status=RawUnitStatus.COHORT_SUGGESTIVE,
                    raw_unit=None,
                    canonical_unit="kcal",
                    grade=EvidenceGrade.C_SUGGESTIVE,
                    mode=PolicyExecutionMode.BLOCKED_DERIVATION,
                    strategy=CanonicalizationStrategy.WITHHOLD,
                    evidence=("project_phase0_activity_submeasurements", "google_fit_calories_expended"),
                    limitations=("Exporter-enriched activity field; mapping to the Google calories type is unverified.",),
                ),
            ),
            description="Exporter-enriched calories coupled to the activity interval.",
        )
    if column == "distance":
        return _measurement(
            column,
            MeasurementKind.EXTENSIVE_TOTAL,
            False,
            raw_projection=True,
            canonical_projection=False,
            units=(
                _unit(
                    ("1.0",),
                    status=RawUnitStatus.COHORT_SUGGESTIVE,
                    raw_unit=None,
                    canonical_unit="m",
                    grade=EvidenceGrade.C_SUGGESTIVE,
                    mode=PolicyExecutionMode.BLOCKED_DERIVATION,
                    strategy=CanonicalizationStrategy.WITHHOLD,
                    evidence=("project_phase0_activity_submeasurements", "google_fit_distance_delta"),
                    limitations=("Exporter-enriched activity field; mapping to a standard distance type is unverified.",),
                ),
            ),
            description="Exporter-enriched distance coupled to the activity interval.",
        )
    if column == "quantity":
        return _measurement(
            column,
            MeasurementKind.CONTEXTUAL_SOURCE_VALUE,
            False,
            raw_projection=False,
            canonical_projection=False,
            units=(
                _unit(
                    ("1.0",),
                    status=RawUnitStatus.UNRESOLVED,
                    raw_unit=None,
                    canonical_unit=None,
                    grade=EvidenceGrade.D_UNRESOLVED,
                    mode=PolicyExecutionMode.BLOCKED_DERIVATION,
                    strategy=CanonicalizationStrategy.WITHHOLD,
                    evidence=("project_phase0_activity_submeasurements",),
                    limitations=("Meaning and unit may vary by activity; no common canonical interpretation is supported.",),
                ),
            ),
            description="Activity-specific source quantity with unresolved meaning.",
        )
    raise KeyError(column)


def _activity_policy(feature: str) -> FeaturePolicy:
    spec = NATIVE_FEATURE_SPECS[feature]
    measurements = tuple(_activity_measurement_policy(column) for column in spec.measurement_columns)
    provenance_raw = tuple(
        column
        for column in ("tracked", "device", "source_name", "source_id")
        if column in spec.column_names
    )
    return FeaturePolicy(
        feature=feature,
        source_key=SOURCE_KEY,
        event_kind=EventKind.ACTIVITY_INTERVAL,
        native_schema=_native_schema(
            feature,
            f"One source-classified {feature} interval with all associated exporter submeasurements retained together.",
        ),
        measurements=measurements,
        temporal=TemporalPolicy(
            start_column="start_date",
            end_column="end_date",
            outer_datetime_column=None,
            summary_date_column=None,
            date_anchor=DateAnchor.START,
            source_day_label_column=None,
            source_day_label_treatment=SourceDayLabelTreatment.NOT_PRESENT,
            preserve_all_boundaries=True,
            canonical_date_derivation_allowed=False,
            notes=("Duration is derived when needed and is not appended to curated CSVs.",),
        ),
        provenance=ProvenancePolicy(
            raw_columns=provenance_raw,
            output_dimensions=MATERIALIZED_PROVENANCE_COLUMNS + VIRTUAL_ACTIVITY_COLUMNS,
            classifier_id="activity_provenance_v1",
            evidence_refs=("project_phase0_provenance", "google_fit_activity_segments"),
            limitations=("Raw identifiers do not uniquely establish a physical device or user-entry method.",),
        ),
        grouping=GroupPolicy((), GroupPreferenceStrategy.NONE, None, None),
        curation=_curation(
            (
                "native_interval_valid",
                "finite_measurement",
                "activity_interval_valid",
                "activity_measurements_nonnegative",
                "activity_quantity_unresolved",
                "activity_tracking_status",
                "activity_provenance_classification",
                "activity_provenance_unknown",
            ),
            inclusion=InclusionPolicy.RULE_DRIVEN,
        ),
        output=_output(
            MATERIALIZED_PROVENANCE_COLUMNS + MATERIALIZED_STATUS_COLUMNS,
            virtual=VIRTUAL_ACTIVITY_COLUMNS,
            fixed=MATERIALIZED_PROVENANCE_COLUMNS,
            sparse=MATERIALIZED_STATUS_COLUMNS,
        ),
        resampling=ResamplingPolicy(
            ResamplingSupport.SUPPORTED_LATER,
            "interval_overlap_duration_and_bouts",
            ("Associated measurements remain coupled and require feature-aware aggregation.",),
        ),
        calibration=PolicyCalibration(
            PolicyMaturity.PROVISIONAL,
            EvidenceGrade.C_SUGGESTIVE,
            PolicyExecutionMode.CONSERVATIVE_ANNOTATION,
            ("activity_measurement_concordance", "provenance_classification_coverage"),
            ("activity_multimeasurement_rows", "activity_quantity_unresolved", "provenance_dimensions", "provenance_classifier"),
            ("Activity submeasurement canonicalization remains blocked; CWP2 0.1.1 did not establish exporter-unit mappings.",),
        ),
        evidence_refs=(
            "google_fit_activity_segments",
            "project_phase0_activity_submeasurements",
            "project_phase0_provenance",
            "project_native_registry",
        ),
    )


def _sleep_policy() -> FeaturePolicy:
    feature = "Sleep"
    spec = NATIVE_FEATURE_SPECS[feature]
    measurements = tuple(_activity_measurement_policy(column) for column in spec.measurement_columns)
    provenance_raw = tuple(
        column
        for column in ("tracked", "device", "source_name", "source_id")
        if column in spec.column_names
    )
    return FeaturePolicy(
        feature=feature,
        source_key=SOURCE_KEY,
        event_kind=EventKind.STATE_INTERVAL,
        native_schema=_native_schema(
            feature,
            "One Google Fitness sleep-state interval using the native SLEEP, AWAKE, DEEP, LIGHT or REM vocabulary.",
        ),
        measurements=measurements,
        temporal=TemporalPolicy(
            start_column="start_date",
            end_column="end_date",
            outer_datetime_column=None,
            summary_date_column=None,
            date_anchor=DateAnchor.START,
            source_day_label_column=None,
            source_day_label_treatment=SourceDayLabelTreatment.NOT_PRESENT,
            preserve_all_boundaries=True,
            canonical_date_derivation_allowed=False,
            notes=("No single-state hypnogram is manufactured by curation.",),
        ),
        provenance=ProvenancePolicy(
            raw_columns=provenance_raw,
            output_dimensions=MATERIALIZED_PROVENANCE_COLUMNS + VIRTUAL_ACTIVITY_COLUMNS,
            classifier_id="sleep_activity_provenance_v1",
            evidence_refs=("project_phase0_provenance", "google_fit_sleep_states"),
            limitations=("Cross-source overlap is context rather than automatic conflict.",),
        ),
        grouping=GroupPolicy(
            ("collecting_method_version", "source_id", "source_name", "device"),
            GroupPreferenceStrategy.SLEEP_OVERLAP_CONTEXT,
            None,
            "sleep_detailed_state_overlap",
            ("1.0",),
            ("Generic SLEEP overlapping detailed states is contextual; no preferred row is selected.",),
        ),
        curation=_curation(
            (
                "native_interval_valid",
                "finite_measurement",
                "activity_measurements_nonnegative",
                "activity_quantity_unresolved",
                "activity_tracking_status",
                "activity_provenance_classification",
                "activity_provenance_unknown",
                "sleep_state_vocabulary",
                "sleep_same_state_overlap",
                "sleep_detailed_state_overlap",
                "sleep_awake_overlap",
                "sleep_generic_detail_overlap_context",
                "sleep_no_apple_equivalence",
            )
        ),
        output=_output(
            MATERIALIZED_PROVENANCE_COLUMNS + MATERIALIZED_STATUS_COLUMNS,
            virtual=VIRTUAL_ACTIVITY_COLUMNS,
            fixed=MATERIALIZED_PROVENANCE_COLUMNS,
            sparse=MATERIALIZED_STATUS_COLUMNS,
        ),
        resampling=ResamplingPolicy(
            ResamplingSupport.SUPPORTED_LATER,
            "state_interval_overlap_with_source_context",
            ("Cross-source harmonization is explicitly outside Google curation.",),
        ),
        calibration=PolicyCalibration(
            PolicyMaturity.PROVISIONAL,
            EvidenceGrade.B_CONVERGENT,
            PolicyExecutionMode.CONSERVATIVE_ANNOTATION,
            ("sleep_overlap_audit", "provenance_classification_coverage"),
            ("sleep_native_states_preserved", "provenance_dimensions", "provenance_classifier"),
            ("Overlap status effects require full-native calibration.",),
        ),
        evidence_refs=(
            "google_fit_sleep_states",
            "project_phase0_sleep_inventory",
            "project_phase0_provenance",
            "project_native_registry",
        ),
    )


def _outer_scalar_measurement(
    feature: str, canonical_unit: str, kind: MeasurementKind, evidence: tuple[str, ...], decision_id: str,
) -> MeasurementPolicy:
    spec = NATIVE_FEATURE_SPECS[feature]
    return _measurement(
        "value",
        kind,
        "value" in spec.required_columns,
        raw_projection=True,
        canonical_projection=False,
        units=(
            _unit(
                spec.supported_versions,
                status=RawUnitStatus.EXPORTER_MAPPING_PENDING,
                raw_unit=None,
                canonical_unit=canonical_unit,
                grade=EvidenceGrade.B_CONVERGENT,
                mode=PolicyExecutionMode.CONSERVATIVE_ANNOTATION,
                strategy=CanonicalizationStrategy.WITHHOLD,
                evidence=evidence,
                limitations=(f"CWP2 0.1.1 left decision {decision_id} unresolved; canonical values remain withheld.",),
            ),
        ),
        description=f"Native {feature} measurement retained unchanged pending exporter-contract calibration.",
    )


def _scalar_output() -> OutputPolicy:
    materialized = MATERIALIZED_PROVENANCE_COLUMNS + MATERIALIZED_STATUS_COLUMNS
    virtual = VIRTUAL_UNIT_COLUMNS + VIRTUAL_PROVENANCE_COLUMNS
    return _output(
        materialized,
        virtual=virtual,
        fixed=MATERIALIZED_PROVENANCE_COLUMNS,
        sparse=MATERIALIZED_STATUS_COLUMNS,
    )


def _calories_policy() -> FeaturePolicy:
    feature = "Calories"
    return FeaturePolicy(
        feature=feature,
        source_key=SOURCE_KEY,
        event_kind=EventKind.INTERVAL_TOTAL,
        native_schema=_native_schema(feature, "One exporter Calories total over its native start/end interval."),
        measurements=(
            _outer_scalar_measurement(
                feature,
                "kcal",
                MeasurementKind.EXTENSIVE_TOTAL,
                ("google_fit_calories_expended", "project_phase0_negative_calories"),
                "calories_unit_mapping",
            ),
        ),
        temporal=TemporalPolicy(
            "start_date",
            "end_date",
            None,
            None,
            DateAnchor.START,
            "source_day_label",
            SourceDayLabelTreatment.CONTEXT_ONLY,
            True,
            False,
            ("The source label never replaces the interval boundaries.",),
        ),
        provenance=ProvenancePolicy(
            (),
            MATERIALIZED_PROVENANCE_COLUMNS + VIRTUAL_PROVENANCE_COLUMNS,
            "outer_calories_provenance_v1",
            ("project_phase0_provenance",),
            ("The native row has no package or device columns; classification may remain unknown.",),
        ),
        grouping=GroupPolicy((), GroupPreferenceStrategy.NONE, None, None),
        curation=_curation(
            (
                "native_interval_valid",
                "finite_measurement",
                "source_day_label_context",
                "source_day_label_unrecognized",
                "source_day_label_nonadjacent_disagreement",
                "calories_negative_tiny",
                "calories_negative_small",
                "calories_negative_material",
                "calories_negative_extreme",
                "unit_mapping_pending",
            )
        ),
        output=_scalar_output(),
        resampling=ResamplingPolicy(
            ResamplingSupport.SUPPORTED_LATER,
            "interval_total_overlap_distribution",
            ("No distribution is executed during curation.",),
        ),
        calibration=PolicyCalibration(
            PolicyMaturity.PROVISIONAL,
            EvidenceGrade.B_CONVERGENT,
            PolicyExecutionMode.CONSERVATIVE_ANNOTATION,
            ("unit_contract_matrix", "negative_calories_audit", "source_day_label_audit"),
            ("calories_unit_mapping", "negative_calories_buckets", "source_day_label_context_only", "status_inclusion_separate"),
            ("Calories must not be relabeled as Apple active energy.",),
        ),
        evidence_refs=(
            "google_fit_calories_expended",
            "project_phase0_negative_calories",
            "project_phase0_day_labels",
            "project_native_registry",
        ),
    )


def _distance_policy() -> FeaturePolicy:
    feature = "Distance"
    return FeaturePolicy(
        feature=feature,
        source_key=SOURCE_KEY,
        event_kind=EventKind.INTERVAL_TOTAL,
        native_schema=_native_schema(feature, "One exporter Distance total over its native start/end interval."),
        measurements=(
            _outer_scalar_measurement(
                feature,
                "m",
                MeasurementKind.EXTENSIVE_TOTAL,
                ("google_fit_distance_delta", "project_phase0_census"),
                "distance_unit_mapping",
            ),
        ),
        temporal=TemporalPolicy(
            "start_date",
            "end_date",
            None,
            None,
            DateAnchor.START,
            "source_day_label",
            SourceDayLabelTreatment.CONTEXT_ONLY,
            True,
            False,
        ),
        provenance=ProvenancePolicy(
            (),
            MATERIALIZED_PROVENANCE_COLUMNS + VIRTUAL_PROVENANCE_COLUMNS,
            "outer_distance_provenance_v1",
            ("project_phase0_provenance",),
            ("The native row has no package or device columns; classification may remain unknown.",),
        ),
        grouping=GroupPolicy((), GroupPreferenceStrategy.NONE, None, None),
        curation=_curation(
            (
                "native_interval_valid",
                "finite_measurement",
                "source_day_label_context",
                "source_day_label_unrecognized",
                "source_day_label_nonadjacent_disagreement",
                "distance_nonnegative",
                "distance_positive_zero_duration",
                "distance_source_rate_limit",
                "unit_mapping_pending",
            )
        ),
        output=_scalar_output(),
        resampling=ResamplingPolicy(
            ResamplingSupport.SUPPORTED_LATER,
            "interval_total_overlap_distribution",
            ("No distribution is executed during curation.",),
        ),
        calibration=PolicyCalibration(
            PolicyMaturity.PROVISIONAL,
            EvidenceGrade.B_CONVERGENT,
            PolicyExecutionMode.CONSERVATIVE_ANNOTATION,
            ("unit_contract_matrix", "activity_measurement_concordance", "source_day_label_audit"),
            ("distance_unit_mapping", "source_day_label_context_only", "status_inclusion_separate"),
            (),
        ),
        evidence_refs=("google_fit_distance_delta", "project_phase0_census", "project_phase0_day_labels", "project_native_registry"),
    )


def _anthropometric_policy(feature: str) -> FeaturePolicy:
    spec = NATIVE_FEATURE_SPECS[feature]
    is_weight = feature == "Weight"
    canonical_unit = "kg" if is_weight else "m"
    official = "google_fit_weight" if is_weight else "google_fit_height"
    decision = "weight_unit_mapping" if is_weight else "height_unit_mapping"
    source_rule = "weight_source_range" if is_weight else "height_source_range"
    rules = [
        "native_interval_valid",
        "finite_measurement",
        "source_day_label_context",
        "source_day_label_unrecognized",
        "source_day_label_nonadjacent_disagreement",
        source_rule,
        "anthropometric_added_by_context",
        "unit_mapping_pending",
    ]
    if is_weight:
        rules.extend(("weight_nonpoint_interval", "weight_long_interval"))
    return FeaturePolicy(
        feature=feature,
        source_key=SOURCE_KEY,
        event_kind=EventKind.POINT_OR_INTERVAL_SCALAR,
        native_schema=_native_schema(
            feature,
            f"One native {feature} scalar with both source boundaries and added_by provenance retained.",
        ),
        measurements=(
            _outer_scalar_measurement(
                feature,
                canonical_unit,
                MeasurementKind.POINT_SCALAR,
                (official, "project_phase0_census"),
                decision,
            ),
        ),
        temporal=TemporalPolicy(
            "start_date",
            "end_date",
            None,
            None,
            DateAnchor.END,
            "source_day_label",
            SourceDayLabelTreatment.CONTEXT_ONLY,
            True,
            False,
            ("The source contract suggests end time as reading time, but curation preserves both native boundaries.",),
        ),
        provenance=ProvenancePolicy(
            tuple(column for column in ("added_by",) if column in spec.column_names),
            MATERIALIZED_PROVENANCE_COLUMNS + VIRTUAL_PROVENANCE_COLUMNS,
            "anthropometric_added_by_v1",
            ("project_phase0_provenance",),
            ("Application identity does not prove device measurement or user entry.",),
        ),
        grouping=GroupPolicy((), GroupPreferenceStrategy.NONE, None, None),
        curation=_curation(tuple(rules)),
        output=_scalar_output(),
        resampling=ResamplingPolicy(
            ResamplingSupport.SUPPORTED_LATER,
            "point_selection_using_end_time_after_policy_review",
            ("Curation does not collapse positive-duration Weight rows.",),
        ),
        calibration=PolicyCalibration(
            PolicyMaturity.PROVISIONAL,
            EvidenceGrade.B_CONVERGENT if is_weight else EvidenceGrade.C_SUGGESTIVE,
            PolicyExecutionMode.CONSERVATIVE_ANNOTATION,
            ("unit_contract_matrix", "weight_interval_audit" if is_weight else "policy_coverage", "source_day_label_audit", "provenance_classification_coverage"),
            (
                decision,
                "weight_nonpoint_preserved" if is_weight else "source_day_label_context_only",
                "provenance_dimensions",
                "provenance_classifier",
                "status_inclusion_separate",
            ),
            (("Only two native Height rows limit cohort-level calibration.",) if not is_weight else ()),
        ),
        evidence_refs=(official, "project_phase0_census", "project_phase0_provenance", "project_native_registry")
        + (("project_phase0_weight_intervals",) if is_weight else ()),
    )


def _step_count_policy() -> FeaturePolicy:
    spec = NATIVE_FEATURE_SPECS["StepCount"]
    value = _measurement(
        "value",
        MeasurementKind.COUNT,
        "value" in spec.required_columns,
        raw_projection=True,
        canonical_projection=True,
        units=(
            _unit(
                ("1.0",),
                status=RawUnitStatus.SOURCE_CONVENTION,
                raw_unit="count",
                canonical_unit="count",
                grade=EvidenceGrade.B_CONVERGENT,
                mode=PolicyExecutionMode.SOURCE_SPECIFIC_EXECUTION,
                strategy=CanonicalizationStrategy.WITHHOLD_IF_INVALID,
                evidence=("project_phase0_stepcount_alternatives", "google_fit_step_count_delta"),
                limitations=("Version 1 lacks payload-explicit data-type metadata.",),
            ),
            _unit(
                ("2.0",),
                status=RawUnitStatus.PAYLOAD_EXPLICIT,
                raw_unit="count",
                canonical_unit="count",
                grade=EvidenceGrade.A_DIRECT,
                mode=PolicyExecutionMode.REVIEWED_EXECUTION,
                strategy=CanonicalizationStrategy.WITHHOLD_IF_INVALID,
                evidence=("google_fit_step_count_delta", "project_phase0_stepcount_alternatives"),
            ),
        ),
        description="Exported StepCount summary value for the source representation.",
    )
    raw_value = _measurement(
        "raw_value",
        MeasurementKind.COUNT,
        False,
        raw_projection=True,
        canonical_projection=False,
        units=(
            _unit(
                ("1.0",),
                status=RawUnitStatus.NOT_APPLICABLE,
                raw_unit=None,
                canonical_unit=None,
                grade=EvidenceGrade.D_UNRESOLVED,
                mode=PolicyExecutionMode.BLOCKED_DERIVATION,
                strategy=CanonicalizationStrategy.NOT_APPLICABLE,
                evidence=("project_native_registry",),
                limitations=("raw_value is absent from the legacy version-1 representation.",),
            ),
            _unit(
                ("2.0",),
                status=RawUnitStatus.PAYLOAD_EXPLICIT,
                raw_unit="count",
                canonical_unit="count",
                grade=EvidenceGrade.A_DIRECT,
                mode=PolicyExecutionMode.REVIEWED_EXECUTION,
                strategy=CanonicalizationStrategy.IDENTITY,
                evidence=("google_fit_step_count_delta",),
            ),
        ),
        description="Nested raw StepCount value retained separately from the exported summary value.",
    )
    return FeaturePolicy(
        feature="StepCount",
        source_key=SOURCE_KEY,
        event_kind=EventKind.MULTI_REPRESENTATION_COUNT,
        native_schema=_native_schema(
            "StepCount",
            "One legacy version-1 scalar or one named version-2 aggregation alternative; all representations remain separate rows.",
        ),
        measurements=(value, raw_value),
        temporal=TemporalPolicy(
            "start_date",
            "end_date",
            "outer_datetime",
            "summary_date",
            DateAnchor.MULTIPLE_CONTEXTS,
            None,
            SourceDayLabelTreatment.NOT_PRESENT,
            True,
            False,
            ("No canonical day is derived from the available temporal contexts.",),
        ),
        provenance=ProvenancePolicy(
            tuple(
                column
                for column in (
                    "aggregation_source",
                    "origin_data_source_id",
                    "app_name",
                    "app_package_name",
                    "device_manufacturer",
                    "device_model",
                    "device_type",
                    "device_uid",
                )
                if column in spec.column_names
            ),
            ("acquisition_method", "derivation_method", "step_count_role"),
            "step_count_provenance_v1",
            ("project_phase0_provenance", "google_fit_daily_step_total"),
            ("A derived aggregate can retain an underlying device-origin stream; acquisition and derivation remain separate.",),
        ),
        grouping=GroupPolicy(
            ("collecting_method_version", "outer_id"),
            GroupPreferenceStrategy.STEP_COUNT_V2_SOURCE_PREFERENCE,
            "step_count_role",
            "step_count_alternatives_disagree",
            ("2.0",),
            ("Version 1 remains legacy_v1 and is not grouped with version 2.",),
        ),
        curation=_curation(
            (
                "finite_measurement",
                "step_count_nonnegative",
                "step_count_source_rate_limit",
                "step_count_v1_legacy_context",
                "step_count_v2_group_structure",
                "step_count_estimated_preferred",
                "step_count_merge_fallback",
                "step_count_sole_fallback",
                "step_count_parallel_nonpreferred",
                "step_count_ambiguous_parallel",
                "step_count_alternatives_disagree",
                "step_count_temporal_context",
                "step_count_cross_version_separate",
                "unit_payload_explicit",
            ),
            inclusion=InclusionPolicy.GROUP_PREFERENCE,
        ),
        output=_output(
            STEP_COUNT_MATERIALIZED_COLUMNS,
            virtual=STEP_COUNT_VIRTUAL_COLUMNS,
            fixed=("acquisition_method", "step_count_role"),
            sparse=MATERIALIZED_STATUS_COLUMNS,
        ),
        resampling=ResamplingPolicy(
            ResamplingSupport.BLOCKED_PENDING_EVIDENCE,
            "explicit_series_selection_then_interval_or_daily_handling",
            ("A DataLoader or derived layer must choose a StepCount representation explicitly.",),
        ),
        calibration=PolicyCalibration(
            PolicyMaturity.PROVISIONAL,
            EvidenceGrade.A_DIRECT,
            PolicyExecutionMode.SOURCE_SPECIFIC_EXECUTION,
            ("stepcount_preference_audit", "stepcount_cross_version_audit", "stepcount_temporal_audit", "unit_contract_matrix"),
            ("step_count_preserve_all_rows", "step_count_preference", "step_count_no_canonical_day", "step_count_unit_mapping", "status_inclusion_separate", "provenance_dimensions"),
            ("CWP2 0.1.1 approved the group-preference roles for execution while preserving every native row.",),
        ),
        evidence_refs=(
            "google_fit_step_count_delta",
            "google_fit_daily_step_total",
            "project_phase0_stepcount_alternatives",
            "project_phase0_stepcount_temporal",
            "project_phase0_provenance",
            "project_native_registry",
        ),
    )


_SCHEMA_V2_DECISION_IDS = (
    "curated_schema_v2_normalization",
    "acquisition_method_materialized",
    "virtual_policy_projections",
    "row_flags_findings_only",
    "sparse_default_encoding",
)


def _with_schema_v2_decisions(policy: FeaturePolicy) -> FeaturePolicy:
    decision_ids = tuple(dict.fromkeys((*policy.calibration.decision_ids, *_SCHEMA_V2_DECISION_IDS)))
    return replace(
        policy,
        calibration=replace(policy.calibration, decision_ids=decision_ids),
    )


def _build_policies() -> dict[str, FeaturePolicy]:
    policies: dict[str, FeaturePolicy] = {}
    for feature, spec in NATIVE_FEATURE_SPECS.items():
        if spec.family is NativeFeatureFamily.ACTIVITY_INTERVAL:
            policies[feature] = _activity_policy(feature)
        elif spec.family is NativeFeatureFamily.SLEEP_STATE_INTERVAL:
            policies[feature] = _sleep_policy()
        elif feature == "Calories":
            policies[feature] = _calories_policy()
        elif feature == "Distance":
            policies[feature] = _distance_policy()
        elif feature in {"Weight", "Height"}:
            policies[feature] = _anthropometric_policy(feature)
        elif feature == "StepCount":
            policies[feature] = _step_count_policy()
        else:  # pragma: no cover - validation protects this branch
            raise AssertionError(f"No Google curation family for {feature}")
    return {
        feature: _with_schema_v2_decisions(policy)
        for feature, policy in sorted(policies.items())
    }


_POLICIES = _build_policies()
CURATION_POLICIES: Mapping[str, FeaturePolicy] = MappingProxyType(_POLICIES)

UNKNOWN_POLICY = FeaturePolicy(
    feature="__unknown__",
    source_key=SOURCE_KEY,
    event_kind=EventKind.UNKNOWN,
    native_schema=NativeSchemaPolicy("__unknown__", (), (), (), (), (), (), "Unknown future native feature copied unchanged."),
    measurements=(),
    temporal=TemporalPolicy(None, None, None, None, DateAnchor.NONE, None, SourceDayLabelTreatment.NOT_PRESENT, True, False),
    provenance=ProvenancePolicy(
        (),
        MATERIALIZED_PROVENANCE_COLUMNS + VIRTUAL_PROVENANCE_COLUMNS,
        "unknown_feature",
        ("project_curation_invariant",),
        ("No semantics are inferred.",),
    ),
    grouping=GroupPolicy((), GroupPreferenceStrategy.NONE, None, None),
    curation=_curation(("unknown_feature_policy",), status=CurationStatus.REVIEW, inclusion=InclusionPolicy.EXCLUDE),
    output=_output(
        MATERIALIZED_PROVENANCE_COLUMNS + MATERIALIZED_STATUS_COLUMNS,
        virtual=VIRTUAL_PROVENANCE_COLUMNS,
        fixed=MATERIALIZED_PROVENANCE_COLUMNS,
        sparse=MATERIALIZED_STATUS_COLUMNS,
    ),
    resampling=ResamplingPolicy(ResamplingSupport.BLOCKED_PENDING_EVIDENCE, "blocked_unknown_feature", ()),
    calibration=PolicyCalibration(
        PolicyMaturity.UNKNOWN,
        EvidenceGrade.D_UNRESOLVED,
        PolicyExecutionMode.BLOCKED_DERIVATION,
        ("policy_coverage",),
        ("unknown_feature_fallback",),
        ("Unknown features receive no canonical conversion.",),
    ),
    evidence_refs=("project_curation_invariant",),
    unknown_fallback=True,
)


UNKNOWN_POLICY = _with_schema_v2_decisions(UNKNOWN_POLICY)


def known_curation_features() -> tuple[str, ...]:
    return tuple(CURATION_POLICIES)


def get_policy(feature: str, *, allow_fallback: bool = True) -> FeaturePolicy:
    try:
        return CURATION_POLICIES[feature]
    except KeyError:
        if allow_fallback:
            return UNKNOWN_POLICY
        raise KeyError(f"Unknown Google Fitness curation feature: {feature!r}")


def _all_evidence_refs(policy: FeaturePolicy) -> set[str]:
    refs = set(policy.evidence_refs)
    refs.update(policy.provenance.evidence_refs)
    for measurement in policy.measurements:
        for unit_policy in measurement.unit_by_version:
            refs.update(unit_policy.evidence_refs)
    return refs


def validate_registry() -> RegistryValidationResult:
    issues: list[str] = []
    native_names = set(native_feature_names())
    policy_names = set(CURATION_POLICIES)
    if policy_names != native_names:
        issues.append(f"policy/native feature mismatch: missing={sorted(native_names-policy_names)}, extra={sorted(policy_names-native_names)}")
    if len(CURATION_POLICIES) != 67:
        issues.append(f"expected 67 policies, found {len(CURATION_POLICIES)}")
    if "Activities" in CURATION_POLICIES:
        issues.append("Activities must not be a curation feature")

    if registry_fingerprint() != EXPECTED_REGISTRY_FINGERPRINT:
        issues.append(
            "registry fingerprint mismatch: "
            f"expected {EXPECTED_REGISTRY_FINGERPRINT}, observed {registry_fingerprint()}"
        )

    for feature, policy in CURATION_POLICIES.items():
        native = NATIVE_FEATURE_SPECS[feature]
        if policy.feature != feature:
            issues.append(f"policy key mismatch for {feature}")
        if policy.source_key != SOURCE_KEY:
            issues.append(f"source key mismatch for {feature}")
        if policy.native_schema.native_columns != native.column_names:
            issues.append(f"native columns differ for {feature}")
        if policy.native_schema.required_columns != native.required_columns:
            issues.append(f"required native columns differ for {feature}")
        if policy.native_schema.measurement_columns != native.measurement_columns:
            issues.append(f"measurement columns differ for {feature}")
        if policy.native_schema.state_columns != native.state_columns:
            issues.append(f"state columns differ for {feature}")
        expected_states = tuple(sorted(set(SLEEP_STATE_MAPPING.values()))) if feature == "Sleep" else ()
        if policy.native_schema.state_vocabulary != expected_states:
            issues.append(f"state vocabulary differs for {feature}")
        if policy.native_schema.supported_versions != native.supported_versions:
            issues.append(f"supported versions differ for {feature}")
        if policy.measurement_columns != native.measurement_columns:
            issues.append(f"measurement policies do not cover native measurements for {feature}")
        if set(policy.materialized_columns) & set(native.column_names):
            issues.append(f"materialized/native column collision for {feature}")
        if set(policy.virtual_columns) & set(native.column_names):
            issues.append(f"virtual/native column collision for {feature}")
        if set(policy.materialized_columns) & set(policy.virtual_columns):
            issues.append(f"materialized/virtual column overlap for {feature}")
        if set(policy.output.fixed_columns) | set(policy.output.sparse_columns) != set(policy.output.materialized_columns):
            issues.append(f"output fixed/sparse columns do not cover materialized columns for {feature}")
        if set(policy.output.fixed_columns) & set(policy.output.sparse_columns):
            issues.append(f"output fixed/sparse overlap for {feature}")
        if not (policy.output.preserve_native_prefix and policy.output.preserve_row_order and policy.output.preserve_row_count):
            issues.append(f"non-destructive invariant disabled for {feature}")

        for measurement in policy.measurements:
            covered: set[str] = set()
            for unit_policy in measurement.unit_by_version:
                overlap = covered & set(unit_policy.versions)
                if overlap:
                    issues.append(f"overlapping unit versions for {feature}/{measurement.column}: {sorted(overlap)}")
                covered.update(unit_policy.versions)
                unknown_versions = set(unit_policy.versions) - set(native.supported_versions)
                if unknown_versions:
                    issues.append(f"unknown unit versions for {feature}/{measurement.column}: {sorted(unknown_versions)}")
            if covered != set(native.supported_versions):
                issues.append(f"unit-version coverage incomplete for {feature}/{measurement.column}: {sorted(covered)}")

        unknown_rules = sorted(set(policy.curation.rule_ids) - set(RULES))
        if unknown_rules:
            issues.append(f"unknown rules for {feature}: {unknown_rules}")
        unknown_evidence = sorted(_all_evidence_refs(policy) - set(EVIDENCE))
        if unknown_evidence:
            issues.append(f"unknown evidence for {feature}: {unknown_evidence}")
        unknown_decisions = sorted(set(policy.calibration.decision_ids) - set(DECISIONS))
        if unknown_decisions:
            issues.append(f"unknown decisions for {feature}: {unknown_decisions}")
        if policy.calibration.maturity is PolicyMaturity.PROVISIONAL and not policy.calibration.required_audits:
            issues.append(f"provisional policy lacks audits for {feature}")

    sleep = CURATION_POLICIES.get("Sleep")
    if sleep is not None:
        if sleep.native_schema.state_columns != ("value",):
            issues.append("Sleep must preserve the native value state column")
        if sleep.grouping.strategy is not GroupPreferenceStrategy.SLEEP_OVERLAP_CONTEXT:
            issues.append("Sleep overlap context is not declared")

    step = CURATION_POLICIES.get("StepCount")
    if step is not None:
        if step.grouping.strategy is not GroupPreferenceStrategy.STEP_COUNT_V2_SOURCE_PREFERENCE:
            issues.append("StepCount preference strategy is not declared")
        if step.grouping.group_columns != ("collecting_method_version", "outer_id"):
            issues.append("StepCount grouping columns differ from accepted contract")
        if "step_count_role" not in step.derived_columns:
            issues.append("StepCount role column is missing")
        if step.temporal.canonical_date_derivation_allowed:
            issues.append("StepCount canonical date derivation must remain blocked")
        try:
            v1 = step.measurements[0].policy_for_version("1.0")
            v2 = step.measurements[0].policy_for_version("2.0")
            if v1.raw_unit_status is not RawUnitStatus.SOURCE_CONVENTION:
                issues.append("StepCount v1 unit evidence must remain source-convention")
            if v2.raw_unit_status is not RawUnitStatus.PAYLOAD_EXPLICIT:
                issues.append("StepCount v2 unit evidence must be payload-explicit")
        except (IndexError, KeyError):
            issues.append("StepCount value unit policies are incomplete")

    for feature in ("Calories", "Distance", "Weight", "Height"):
        policy = CURATION_POLICIES[feature]
        unit = policy.measurements[0].unit_by_version[0]
        if unit.raw_unit_status is not RawUnitStatus.EXPORTER_MAPPING_PENDING:
            issues.append(f"{feature} must remain exporter-mapping-pending")
        if unit.execution_mode is not PolicyExecutionMode.CONSERVATIVE_ANNOTATION:
            issues.append(f"{feature} must remain conservative annotation")

    return RegistryValidationResult(tuple(sorted(set(issues))))


def _base_payload() -> dict[str, object]:
    return {
        "source_key": SOURCE_KEY,
        "curation_registry_version": CURATION_REGISTRY_VERSION,
        "policy_contract_version": POLICY_CONTRACT_VERSION,
        "native_registry_feature_count": len(NATIVE_FEATURE_SPECS),
        "policies": [to_primitive(CURATION_POLICIES[name]) for name in CURATION_POLICIES],
        "unknown_policy": to_primitive(UNKNOWN_POLICY),
    }


def registry_fingerprint() -> str:
    rendered = json.dumps(
        _base_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(rendered).hexdigest()


def registry_payload() -> dict[str, object]:
    validation = validate_registry()
    payload = _base_payload()
    payload.update(
        {
            "registry_fingerprint": registry_fingerprint(),
            "validation": validation.as_dict(),
            "counts": {
                "policies": len(CURATION_POLICIES),
                "activity_policies": sum(policy.event_kind is EventKind.ACTIVITY_INTERVAL for policy in CURATION_POLICIES.values()),
                "sleep_policies": sum(policy.event_kind is EventKind.STATE_INTERVAL for policy in CURATION_POLICIES.values()),
                "scalar_policies": sum(policy.event_kind in {EventKind.INTERVAL_TOTAL, EventKind.POINT_OR_INTERVAL_SCALAR} for policy in CURATION_POLICIES.values()),
                "step_count_policies": sum(policy.event_kind is EventKind.MULTI_REPRESENTATION_COUNT for policy in CURATION_POLICIES.values()),
            },
        }
    )
    return payload


def matrix_rows(features: Iterable[str] | None = None) -> list[dict[str, object]]:
    names = tuple(features) if features is not None else known_curation_features()
    rows: list[dict[str, object]] = []
    for feature in names:
        policy = get_policy(feature, allow_fallback=False)
        rows.append(
            {
                "feature": feature,
                "event_kind": policy.event_kind.value,
                "versions": ";".join(policy.native_schema.supported_versions),
                "native_columns": ";".join(policy.native_schema.native_columns),
                "measurements": ";".join(policy.measurement_columns),
                "materialized_columns": ";".join(policy.materialized_columns),
                "virtual_columns": ";".join(policy.virtual_columns),
                "rules": ";".join(policy.curation.rule_ids),
                "default_status": policy.curation.default_status.value,
                "default_inclusion": policy.curation.default_inclusion.value,
                "maturity": policy.calibration.maturity.value,
                "evidence_grade": policy.calibration.evidence_grade.value,
                "execution_mode": policy.calibration.execution_mode.value,
                "required_audits": ";".join(policy.calibration.required_audits),
            }
        )
    return rows


def matrix_csv(features: Iterable[str] | None = None) -> str:
    rows = matrix_rows(features)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else [])
    if rows:
        writer.writeheader()
        writer.writerows(rows)
    return output.getvalue()


_VALIDATION = validate_registry()
if not _VALIDATION.valid:
    raise RuntimeError("Invalid Google Fitness curation registry: " + "; ".join(_VALIDATION.issues))
