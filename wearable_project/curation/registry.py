"""
Wearable Data Processing and Modeling project
Authoritative milestone-two feature-policy registry. Policies are fully typed, validated, versioned, and
exportable. The 0.2.0a2.1 curation engine executes only reviewed or explicitly scoped decisions while
the native parser and outputs remain unchanged.
"""


from __future__ import annotations
from dataclasses import replace
from hashlib import sha256
import csv
import io
import json
from typing import Iterable, Mapping
from wearable_project.curation.evidence import EVIDENCE, EvidenceSource
from wearable_project.curation.decisions import POLICY_CALIBRATION_DECISIONS, validate_decisions
from wearable_project.curation.models import (
    AcquisitionMethod,
    ConflictBehavior,
    ConversionConfidence,
    CrossFeaturePolicy,
    CurationPolicy,
    CurationStatus,
    DateAnchor,
    DurationModel,
    EventKind,
    EvidenceGrade,
    FeaturePolicy,
    IdentityPolicy,
    InclusionPolicy,
    IntervalClosure,
    MeasurementField,
    MeasurementKind,
    MeasurementUnitPolicy,
    MissingFieldAction,
    OutputPolicy,
    PolicyCalibration,
    PolicyExecutionMode,
    PolicyMaturity,
    PolicyTestContract,
    ProvenancePolicy,
    RawUnitStatus,
    ReconciliationPolicy,
    RegistryValidationResult,
    ResamplingPolicy,
    ResamplingSupport,
    SchemaPolicy,
    SemanticPolicy,
    SourceRule,
    UnitPolicy,
    UnresolvedUnitAction,
    UserEnteredHandling,
    to_primitive,
)
from wearable_project.curation.rules import RULES, rule_execution_kind
from wearable_project.curation.strategies import ALL_STRATEGY_CATALOGS


CURATION_REGISTRY_VERSION = "0.2.0a2.1-policy-4"
POLICY_CONTRACT_VERSION = "1.2.0-alpha1"

# Exactly the 33 Apple HealthKit feature names observed in the accepted full cohort run. New names are handled
# through UNKNOWN_POLICY and must not be silently converted.
COHORT_OBSERVED_FEATURES: tuple[str, ...] = (
    "ActiveEnergyBurned",
    "ActivitySummary",
    "BMI",
    "BasalEnergyBurned",
    "BloodAlcoholContent",
    "BloodGlucose",
    "BloodPressure",
    "BodyFatPercentage",
    "BodyTemperature",
    "Carbohydrates",
    "DailyDistanceCycling",
    "DailyDistanceSwimming",
    "DistanceWalkingRunning",
    "Electrocardiogram",
    "EnergyConsumed",
    "FlightsClimbed",
    "HeartRate",
    "HeartRateVariability",
    "Height",
    "LeanBodyMass",
    "Mindful",
    "OxygenSaturation",
    "PeakFlow",
    "Protein",
    "RespiratoryRate",
    "RestingHeartRate",
    "Sleep",
    "StepCount",
    "TotalFat",
    "Vo2Max",
    "WaistCircumference",
    "WalkingHeartRate",
    "Weight",
)

_COMMON_REQUIRED = (
    "start_date",
    "end_date",
    "datetime",
    "created_at",
    "updated_at",
    "data_source",
    "collecting_method_version",
)
_COMMON_OPTIONAL = (
    "record_id",
    "source_id",
    "source_name",
    "device",
    "was_user_entered",
    "utc_offset_minutes",
    "time_zone",
    "quality_flags",
    "occurrence_count",
    "duplicate_count",
    "revision_count",
    "conflict_group_id",
)
_COMMON_PREFIX = (
    "start_date",
    "end_date",
    "value",
    "datetime",
    "created_at",
    "updated_at",
    "data_source",
    "collecting_method_version",
)
_COMMON_PROVENANCE = (
    "record_id",
    "source_id",
    "source_name",
    "device",
    "was_user_entered",
)
_COMMON_DERIVED = (
    "canonical_value",
    "canonical_unit",
    "unit_status",
    "acquisition_method",
    "curation_status",
    "curation_flags",
    "include_by_default",
)
_COMMON_SPARSE = (
    "canonical_value",
    "canonical_unit",
    "unit_status",
    "acquisition_method",
    "curation_status",
    "curation_flags",
    "include_by_default",
)


def _identity(
    name: str, *, maturity: PolicyMaturity = PolicyMaturity.REVIEWED, aliases: tuple[str, ...] = ()
) -> IdentityPolicy:
    return IdentityPolicy(
        canonical_name=name,
        aliases=aliases,
        policy_version=POLICY_CONTRACT_VERSION,
        maturity=maturity,
        cohort_observed=True,
    )


def _measurement(
    column: str = "value", role: str = "value", *, value_kind: str = "numeric", required: bool = True
) -> MeasurementField:
    return MeasurementField(column=column, role=role, value_kind=value_kind, required=required)


def _scalar_schema(
    family: str, *, context: tuple[str, ...] = (), optional: tuple[str, ...] = (),
    accepted: tuple[str, ...] = ("list[dict]", "dict"),
) -> SchemaPolicy:
    return SchemaPolicy(
        feature_family=family,
        accepted_payload_shapes=accepted,
        required_columns=_COMMON_REQUIRED + ("value",),
        optional_columns=tuple(dict.fromkeys(_COMMON_OPTIONAL + optional + context)),
        measurements=(_measurement(),),
        context_columns=context,
        missing_measurement_action=MissingFieldAction.EXCLUDE_DEFAULT,
    )


def _unit(
    *, measurement: str = "value", status: RawUnitStatus, raw_unit: str | None, candidates: tuple[str, ...] = (),
    canonical: str | None, resolution: str, conversion: str, confidence: ConversionConfidence,
    unresolved: UnresolvedUnitAction, evidence: tuple[str, ...], precision: int | None = None,
) -> UnitPolicy:
    return UnitPolicy((MeasurementUnitPolicy(
        measurement=measurement,
        raw_unit_status=status,
        raw_unit=raw_unit,
        raw_unit_candidates=candidates,
        canonical_unit=canonical,
        unit_resolution_strategy=resolution,
        conversion_rule=conversion,
        conversion_confidence=confidence,
        unresolved_action=unresolved,
        evidence_refs=evidence,
        output_precision=precision,
    ),))


def _no_unit(measurement: str = "value") -> UnitPolicy:
    return _unit(
        measurement=measurement,
        status=RawUnitStatus.NOT_APPLICABLE,
        raw_unit=None,
        canonical=None,
        resolution="not_applicable",
        conversion="not_applicable",
        confidence=ConversionConfidence.NONE,
        unresolved=UnresolvedUnitAction.NOT_APPLICABLE,
        evidence=(),
    )


def _fixed_unit(
    raw: str, canonical: str, *, evidence: tuple[str, ...], conversion: str = "identity", precision: int | None = None
) -> UnitPolicy:
    return _unit(
        status=RawUnitStatus.SOURCE_CONVENTION,
        raw_unit=raw,
        canonical=canonical,
        resolution="fixed_source_convention",
        conversion=conversion,
        confidence=ConversionConfidence.EXACT,
        unresolved=UnresolvedUnitAction.RETAIN_RAW_ONLY,
        evidence=evidence,
        precision=precision,
    )


def _percent_fraction(*, evidence: tuple[str, ...], precision: int | None = 3) -> UnitPolicy:
    return _unit(
        status=RawUnitStatus.SOURCE_CONVENTION,
        raw_unit="fraction",
        canonical="%",
        resolution="healthkit_percent_fraction",
        conversion="fraction_to_percent",
        confidence=ConversionConfidence.EXACT,
        unresolved=UnresolvedUnitAction.RETAIN_RAW_ONLY,
        evidence=evidence,
        precision=precision,
    )


def _semantics(
    event: EventKind, measurement: MeasurementKind, duration: DurationModel, closure: IntervalClosure,
    *, native_resolution: str, date_anchor: DateAnchor = DateAnchor.EVENT_START_UTC,
    start_meaning: str = "native event start", end_meaning: str = "native event end",
) -> SemanticPolicy:
    return SemanticPolicy(
        event_kind=event,
        measurement_kind=measurement,
        duration_model=duration,
        interval_closure=closure,
        start_time_meaning=start_meaning,
        end_time_meaning=end_meaning,
        native_resolution=native_resolution,
        date_anchor=date_anchor,
    )


def _provenance(
    acquisition: str = "preserve_source_context", *,
    user_entered: UserEnteredHandling = UserEnteredHandling.ACCEPT_AND_FLAG,
    source_epoch: str = "source_id_device_version_time_zone", source_priority: str = "preserve_all_mark_conflicts",
    context: tuple[str, ...] = (), source_rules: tuple[SourceRule, ...] = (),
) -> ProvenancePolicy:
    return ProvenancePolicy(
        acquisition_method_strategy=acquisition,
        user_entered_handling=user_entered,
        source_epoch_strategy=source_epoch,
        source_priority_strategy=source_priority,
        retain_context_columns=tuple(dict.fromkeys(("source_id", "source_name", "device", "was_user_entered") + context)),
        source_rules=source_rules,
    )


def _reconcile(
    *, duplicate: str = "inherit_native_record_id_then_content", revision: str = "inherit_native_record_revision",
    conflict: ConflictBehavior = ConflictBehavior.PRESERVE_ALL_REVIEW, identity: str = "inherit_native_event_identity",
    note: str | None = None,
) -> ReconciliationPolicy:
    return ReconciliationPolicy(
        occurrence_identity_strategy=identity,
        exact_duplicate_strategy=duplicate,
        revision_strategy=revision,
        conflict_behavior=conflict,
        merge_safe=True,
        note=note,
    )


def _curation(
    *rules: str, status: CurationStatus = CurationStatus.PASS, inclusion: InclusionPolicy = InclusionPolicy.INCLUDE
) -> CurationPolicy:
    base = ("required_measurements_present", "finite_numeric", "timestamp_order")
    return CurationPolicy(rule_ids=tuple(dict.fromkeys(base + rules)), default_status=status, default_inclusion=inclusion)


def _output(
    primary: tuple[str, ...] = ("value",), *, context: tuple[str, ...] = (), derived: tuple[str, ...] = _COMMON_DERIVED,
    prefix: tuple[str, ...] | None = None, numeric: tuple[tuple[str, str], ...] = (), categorical: tuple[tuple[str, str], ...] = (),
) -> OutputPolicy:
    actual_prefix = prefix or tuple(col for col in _COMMON_PREFIX if col != "value" or "value" in primary)
    return OutputPolicy(
        primary_measurement_columns=primary,
        retained_provenance_columns=_COMMON_PROVENANCE,
        retained_context_columns=context,
        derived_columns=derived,
        sparse_columns=tuple(col for col in _COMMON_SPARSE if col in derived),
        column_order_prefix=actual_prefix,
        numeric_dtypes=numeric,
        categorical_dtypes=categorical,
    )


def _resampling(
    support: ResamplingSupport, strategy: str, *, aggregation: str = "none", point_assignment: str = "none",
    state_handling: str = "not_applicable", allocation: str = "none", broadcasting: str = "prohibited",
    supports: tuple[str, ...] = (), invariant: str | None = None,
) -> ResamplingPolicy:
    return ResamplingPolicy(
        support=support,
        default_enabled=False,
        strategy=strategy,
        aggregation=aggregation,
        point_assignment=point_assignment,
        state_handling=state_handling,
        interval_allocation=allocation,
        broadcasting=broadcasting,
        required_support_outputs=supports,
        conservation_invariant=invariant,
    )


def _tests(name: str, *invariants: str) -> PolicyTestContract:
    return PolicyTestContract(
        fixture_ids=(f"fixture_{name}",),
        invariants=(
            "native_row_count_is_preserved",
            "native_measurement_values_are_not_mutated",
            "all_nonpass_rows_have_explanatory_flags",
        ) + invariants,
    )


def _calibration(
    grade: EvidenceGrade, mode: PolicyExecutionMode, rationale: str, *, audits: tuple[str, ...] = (),
    fallback: str = "Preserve native values and annotate uncertainty without canonical conversion.",
    scope: str = "all_registered_sources",
) -> PolicyCalibration:
    return PolicyCalibration(
        evidence_grade=grade,
        execution_mode=mode,
        rationale=rationale,
        required_audits=audits,
        safe_fallback=fallback,
        source_scope=scope,
        decision_version="calibration-1.1",
    )


def _interval_total_policy(
    name: str, *, raw_unit: str | None, canonical_unit: str | None, evidence: tuple[str, ...],
    maturity: PolicyMaturity = PolicyMaturity.REVIEWED, unit_policy: UnitPolicy | None = None,
    provenance_strategy: str = "device_or_application_estimate", notes: tuple[str, ...] = (),
) -> FeaturePolicy:
    units = unit_policy or _fixed_unit(raw_unit or "unknown", canonical_unit or raw_unit or "unknown", evidence=evidence)
    return FeaturePolicy(
        identity=_identity(name, maturity=maturity),
        schema=_scalar_schema("interval_total"),
        semantics=_semantics(
            EventKind.INTERVAL, MeasurementKind.EXTENSIVE_TOTAL, DurationModel.EXPLICIT_INTERVAL, IntervalClosure.HALF_OPEN,
            native_resolution="source-defined interval total",
        ),
        units=units,
        provenance=_provenance(provenance_strategy),
        reconciliation=_reconcile(
            revision="inherit_native_interval_revision",
            note="Milestone 1 boundary-revision resolution is inherited; unresolved candidates remain explicit.",
        ),
        curation=_curation(
            "interval_duration_positive", "nonnegative_measurement", "unresolved_same_interval_conflict",
            "unit_resolved_for_canonical_value", "source_epoch_transition"
        ),
        cross_feature=CrossFeaturePolicy(),
        output=_output(),
        resampling=_resampling(
            ResamplingSupport.SUPPORTED,
            "extensive_overlap_sum",
            aggregation="sum",
            allocation="proportional_to_exact_overlap_seconds",
            supports=("n_contributing_events", "observed_seconds", "coverage_fraction", "is_disaggregated"),
            invariant="sum_of_allocated_values_equals_native_interval_total",
        ),
        evidence_refs=tuple(dict.fromkeys(evidence + ("project_interval_boundary_revisions",))),
        tests=_tests(name, "future_overlap_allocation_conserves_native_total"),
        notes=notes,
    )


def _point_policy(
    name: str, *, units: UnitPolicy, evidence: tuple[str, ...], maturity: PolicyMaturity = PolicyMaturity.REVIEWED,
    rules: tuple[str, ...] = (), provenance_strategy: str = "user_entered_or_imported",
    provenance_context: tuple[str, ...] = (), source_rules: tuple[SourceRule, ...] = (),
    cross_rules: tuple[str, ...] = (), measurement_kind: MeasurementKind = MeasurementKind.INTENSIVE_VALUE,
    notes: tuple[str, ...] = (),
) -> FeaturePolicy:
    return FeaturePolicy(
        identity=_identity(name, maturity=maturity),
        schema=_scalar_schema("point_scalar", context=provenance_context),
        semantics=_semantics(
            EventKind.POINT, measurement_kind, DurationModel.ZERO_DURATION_POINT, IntervalClosure.POINT,
            native_resolution="instantaneous point observation", end_meaning="same instant as native start",
        ),
        units=units,
        provenance=_provenance(provenance_strategy, context=provenance_context, source_rules=source_rules,),
        reconciliation=_reconcile(),
        curation=_curation("point_duration_zero", "unit_resolved_for_canonical_value", "manual_entry_context", "source_epoch_transition", *rules),
        cross_feature=CrossFeaturePolicy(cross_rules),
        output=_output(context=provenance_context),
        resampling=_resampling(
            ResamplingSupport.OPTIONAL,
            "point_retain",
            point_assignment="retain_native_points_by_default",
            supports=("point_count", "native_duration_seconds", "staleness_seconds"),
        ),
        evidence_refs=evidence,
        tests=_tests(name, "point_observation_never_acquires_artificial_duration"),
        notes=notes,
    )


_POLICIES: dict[str, FeaturePolicy] = {}


def _register(*policies: FeaturePolicy) -> None:
    for policy in policies:
        if policy.name in _POLICIES:
            raise ValueError(f"Duplicate curation feature policy: {policy.name}")
        _POLICIES[policy.name] = policy


# Physical activity, distance and energy interval totals.
_register(
    _interval_total_policy("StepCount", raw_unit="count", canonical_unit="count", evidence=("apple_healthkit_quantity_types",)),
    _interval_total_policy("DistanceWalkingRunning", raw_unit="m", canonical_unit="m", evidence=("apple_distance_walking_running", "project_unit_fingerprints")),
    _interval_total_policy(
        "DailyDistanceCycling", raw_unit="m", canonical_unit="m",
        evidence=("apple_healthkit_quantity_types", "project_unit_fingerprints"),
        maturity=PolicyMaturity.REVIEWED,
        unit_policy=_unit(
            status=RawUnitStatus.INFERRED_HIGH_CONFIDENCE, raw_unit="m",
            canonical="m",
            resolution="cohort_exporter_metres", conversion="identity",
            confidence=ConversionConfidence.HIGH,
            unresolved=UnresolvedUnitAction.RETAIN_RAW_ONLY,
            evidence=("apple_healthkit_quantity_types", "project_unit_fingerprints"),
        ),
        notes=(
            "Inner records are interval totals despite the Daily name in this exporter.",
            "The full-cohort audit supports metres, including exact upstream mile-to-metre conversion fingerprints.",
        ),
    ),
    _interval_total_policy(
        "DailyDistanceSwimming", raw_unit="m", canonical_unit="m",
        evidence=("apple_healthkit_quantity_types", "project_unit_fingerprints"),
        maturity=PolicyMaturity.REVIEWED,
        notes=("Metre convention is supported by observed round pool-session totals.",),
    ),
    _interval_total_policy("FlightsClimbed", raw_unit="count", canonical_unit="count", evidence=("apple_healthkit_quantity_types",)),
    _interval_total_policy("BasalEnergyBurned", raw_unit="kcal", canonical_unit="kcal", evidence=("apple_healthkit_quantity_types",)),
    _interval_total_policy("ActiveEnergyBurned", raw_unit="kcal", canonical_unit="kcal", evidence=("apple_healthkit_quantity_types",)),
)

# Daily activity summary: preserve unresolved payload item identity.
_activity_measurements = tuple(
    _measurement(column, column) for column in (
        "apple_stand_hours",
        "apple_exercise_time",
        "active_energy_burned",
        "apple_stand_hours_goal",
        "apple_exercise_time_goal",
        "active_energy_burned_goal",
    )
)
_register(FeaturePolicy(
    identity=_identity("ActivitySummary", maturity=PolicyMaturity.PROVISIONAL),
    schema=SchemaPolicy(
        feature_family="daily_summary",
        accepted_payload_shapes=("list[dict]", "dict"),
        required_columns=("datetime", "data_source", "collecting_method_version", "payload_index"),
        optional_columns=("created_at", "updated_at", "quality_flags"),
        measurements=_activity_measurements,
        missing_measurement_action=MissingFieldAction.REVIEW,
    ),
    semantics=_semantics(
        EventKind.DAILY_SUMMARY, MeasurementKind.MULTIVARIATE_SUMMARY,
        DurationModel.DATE_COMPONENT_SUMMARY, IntervalClosure.NOT_APPLICABLE,
        native_resolution="HealthKit daily activity summary object",
        date_anchor=DateAnchor.OUTER_UTC_BUCKET,
        start_meaning="not encoded by the exporter",
        end_meaning="not encoded by the exporter",
    ),
    units=UnitPolicy((
        MeasurementUnitPolicy("active_energy_burned", RawUnitStatus.SOURCE_CONVENTION, "kcal", (), "kcal", "fixed_source_convention", "identity", ConversionConfidence.EXACT, UnresolvedUnitAction.RETAIN_RAW_ONLY, ("apple_activity_summary",), 3),
        MeasurementUnitPolicy("active_energy_burned_goal", RawUnitStatus.SOURCE_CONVENTION, "kcal", (), "kcal", "fixed_source_convention", "identity", ConversionConfidence.EXACT, UnresolvedUnitAction.RETAIN_RAW_ONLY, ("apple_activity_summary",), 3),
        MeasurementUnitPolicy("apple_exercise_time", RawUnitStatus.SOURCE_CONVENTION, "min", (), "min", "fixed_source_convention", "identity", ConversionConfidence.EXACT, UnresolvedUnitAction.RETAIN_RAW_ONLY, ("apple_activity_summary",), 3),
        MeasurementUnitPolicy("apple_exercise_time_goal", RawUnitStatus.SOURCE_CONVENTION, "min", (), "min", "fixed_source_convention", "identity", ConversionConfidence.EXACT, UnresolvedUnitAction.RETAIN_RAW_ONLY, ("apple_activity_summary",), 3),
        MeasurementUnitPolicy("apple_stand_hours", RawUnitStatus.SOURCE_CONVENTION, "h", (), "h", "fixed_source_convention", "identity", ConversionConfidence.EXACT, UnresolvedUnitAction.RETAIN_RAW_ONLY, ("apple_activity_summary",), 3),
        MeasurementUnitPolicy("apple_stand_hours_goal", RawUnitStatus.SOURCE_CONVENTION, "h", (), "h", "fixed_source_convention", "identity", ConversionConfidence.EXACT, UnresolvedUnitAction.RETAIN_RAW_ONLY, ("apple_activity_summary",), 3),
    )),
    provenance=_provenance("activity_summary_system_object", user_entered=UserEnteredHandling.NOT_APPLICABLE, source_epoch="none"),
    reconciliation=_reconcile(
        identity="inherit_native_summary_identity",
        duplicate="inherit_native_summary_occurrence",
        revision="activity_summary_unresolved_date_assignment",
        conflict=ConflictBehavior.PRESERVE_ALL_EXCLUDE_DEFAULT,
        note="Sliding payload pairs remain separate pending exporter date-assignment validation.",
    ),
    curation=_curation("activity_summary_date_ambiguity", "activity_summary_nonnegative", status=CurationStatus.REVIEW, inclusion=InclusionPolicy.EXCLUDE),
    cross_feature=CrossFeaturePolicy(),
    output=_output(
        primary=tuple(item.column for item in _activity_measurements),
        context=("payload_index",),
        prefix=("datetime",) + tuple(item.column for item in _activity_measurements) + ("created_at", "updated_at", "data_source", "collecting_method_version"),
    ),
    resampling=_resampling(ResamplingSupport.OPTIONAL, "daily_context_only", aggregation="none", broadcasting="prohibited_by_default", supports=("summary_date_assignment_status",)),
    evidence_refs=("apple_activity_summary", "project_activity_summary_sliding_pair"),
    tests=_tests("ActivitySummary", "payload_items_are_never_averaged", "ambiguous_date_assignment_is_not_silently_resolved"),
    notes=("Daily context may be joined later, but is not a five-minute observation.",),
))

# Heart and cardiovascular features.
_register(FeaturePolicy(
    identity=_identity("HeartRate"),
    schema=_scalar_schema("intensive_interval", context=("heart_rate_motion_context",)),
    semantics=_semantics(EventKind.POINT_OR_INTERVAL, MeasurementKind.INTENSIVE_VALUE, DurationModel.ZERO_OR_EXPLICIT_INTERVAL, IntervalClosure.HALF_OPEN, native_resolution="source-defined discrete heart-rate observation with zero or short interval support"),
    units=_fixed_unit("beats/min", "beats/min", evidence=("apple_heart_rate",), precision=2),
    provenance=_provenance("heart_rate_source_and_motion_context", context=("heart_rate_motion_context",)),
    reconciliation=_reconcile(),
    curation=_curation("positive_measurement", "heart_rate_motion_context", "source_epoch_transition"),
    cross_feature=CrossFeaturePolicy(),
    output=_output(context=("heart_rate_motion_context",)),
    resampling=_resampling(ResamplingSupport.SUPPORTED, "intensive_window_distribution", aggregation="robust_statistic_configurable", supports=("n_observations", "observed_seconds", "source_count")),
    evidence_refs=("apple_heart_rate",),
    tests=_tests("HeartRate", "native_observations_are_not_preaggregated"),
))

for _name, _event, _maturity, _revision, _extra_rules, _evidence, _notes in (
    ("RestingHeartRate", EventKind.POINT_OR_INTERVAL, PolicyMaturity.REVIEWED, "inherit_native_record_revision", ("heart_rate_summary_interval",), ("apple_heart_rate",), ("A summary estimate may be represented as a point or interval and must not be interpreted as dense sampling.",)),
    ("WalkingHeartRate", EventKind.POINT_OR_INTERVAL, PolicyMaturity.REVIEWED, "walking_heart_rate_replaceable_estimate", ("heart_rate_summary_interval",), ("apple_walking_heart_rate",), ("Apple may replace estimates; preserve point-or-interval summary semantics and revision lineage.",)),
):
    _register(FeaturePolicy(
        identity=_identity(_name, maturity=_maturity),
        schema=_scalar_schema("long_summary_interval"),
        semantics=_semantics(_event, MeasurementKind.SUMMARY_STATISTIC, DurationModel.ZERO_OR_EXPLICIT_INTERVAL, IntervalClosure.HALF_OPEN, native_resolution="source-defined summary estimate represented as a point or explicit interval"),
        units=_fixed_unit("beats/min", "beats/min", evidence=_evidence, precision=2),
        provenance=_provenance("device_or_application_estimate"),
        reconciliation=_reconcile(revision=_revision),
        curation=_curation("positive_measurement", "source_epoch_transition", *_extra_rules),
        cross_feature=CrossFeaturePolicy(),
        output=_output(),
        resampling=_resampling(ResamplingSupport.OPTIONAL, "long_summary_reference", aggregation="none", broadcasting="prohibited_by_default", supports=("summary_event_reference", "native_support_seconds")),
        evidence_refs=_evidence,
        tests=_tests(_name, "summary_value_is_not_broadcast_by_default"),
        notes=_notes,
    ))

_register(FeaturePolicy(
    identity=_identity("HeartRateVariability", maturity=PolicyMaturity.REVIEWED),
    schema=_scalar_schema("intensive_interval", context=("metadata_sync_version",)),
    semantics=_semantics(EventKind.POINT_OR_INTERVAL, MeasurementKind.INTENSIVE_VALUE, DurationModel.ZERO_OR_EXPLICIT_INTERVAL, IntervalClosure.HALF_OPEN, native_resolution="discrete SDNN observation with zero or short interval support"),
    units=_unit(
        status=RawUnitStatus.INFERRED_HIGH_CONFIDENCE,
        raw_unit="s",
        canonical="ms",
        resolution="cohort_exporter_seconds",
        conversion="seconds_to_milliseconds",
        confidence=ConversionConfidence.HIGH,
        unresolved=UnresolvedUnitAction.RETAIN_RAW_ONLY,
        evidence=("apple_hrv_sdnn", "hrv_task_force_1996", "project_unit_fingerprints"),
        precision=3,
    ),
    provenance=_provenance("device_or_application_estimate", context=("metadata_sync_version",)),
    reconciliation=_reconcile(),
    curation=_curation("positive_measurement", "hrv_sdnn_context", "unit_resolved_for_canonical_value", "source_epoch_transition"),
    cross_feature=CrossFeaturePolicy(),
    output=_output(context=("metadata_sync_version",)),
    resampling=_resampling(ResamplingSupport.SUPPORTED, "intensive_window_distribution", aggregation="robust_statistic_configurable", supports=("n_observations", "observed_seconds", "source_count")),
    evidence_refs=("apple_hrv_sdnn", "hrv_task_force_1996", "project_unit_fingerprints"),
    tests=_tests("HeartRateVariability", "seconds_to_milliseconds_conversion_is_explicit"),
    notes=("The full-cohort audit found seconds-like encoding across every observed source/version/algorithm context; canonical SDNN is milliseconds.",),
))

# Blood pressure remains one multivariate point event.
_bp_fields = (
    _measurement("blood_pressure_systolic_value", "systolic_pressure"),
    _measurement("blood_pressure_diastolic_value", "diastolic_pressure"),
)
_register(FeaturePolicy(
    identity=_identity("BloodPressure"),
    schema=SchemaPolicy(
        feature_family="point_vector",
        accepted_payload_shapes=("list[dict]", "dict"),
        required_columns=_COMMON_REQUIRED + tuple(field.column for field in _bp_fields),
        optional_columns=_COMMON_OPTIONAL,
        measurements=_bp_fields,
        missing_measurement_action=MissingFieldAction.EXCLUDE_DEFAULT,
    ),
    semantics=_semantics(EventKind.POINT, MeasurementKind.MULTIVARIATE_POINT, DurationModel.ZERO_DURATION_POINT, IntervalClosure.POINT, native_resolution="instantaneous paired blood-pressure reading", end_meaning="same instant as native start"),
    units=UnitPolicy(tuple(
        MeasurementUnitPolicy(field.role, RawUnitStatus.SOURCE_CONVENTION, "mmHg", (), "mmHg", "fixed_source_convention", "identity", ConversionConfidence.EXACT, UnresolvedUnitAction.RETAIN_RAW_ONLY, ("apple_blood_pressure",), 1)
        for field in _bp_fields
    )),
    provenance=_provenance("user_entered_or_imported"),
    reconciliation=_reconcile(duplicate="inherit_native_vector_occurrence"),
    curation=_curation("point_duration_zero", "blood_pressure_pair_complete", "blood_pressure_pair_order", "manual_entry_context", "source_epoch_transition"),
    cross_feature=CrossFeaturePolicy(),
    output=_output(
        primary=tuple(field.column for field in _bp_fields),
        prefix=("start_date", "end_date") + tuple(field.column for field in _bp_fields) + ("datetime", "created_at", "updated_at", "data_source", "collecting_method_version"),
    ),
    resampling=_resampling(ResamplingSupport.OPTIONAL, "point_retain", point_assignment="retain_paired_points", supports=("point_count", "pair_complete")),
    evidence_refs=("apple_blood_pressure", "aha_blood_pressure_measurement"),
    tests=_tests("BloodPressure", "systolic_and_diastolic_remain_coupled", "no_componentwise_synthetic_pair_is_created"),
))

# Anthropometrics and body composition.
_register(
    _point_policy(
        "BMI",
        units=_fixed_unit("kg/m2", "kg/m2", evidence=("apple_healthkit_quantity_types", "who_bmi"), precision=3),
        evidence=("apple_healthkit_quantity_types", "who_bmi"),
        rules=("positive_measurement", "bmi_screening_context"),
        cross_rules=("bmi_height_weight_consistency",),
        notes=("BMI is retained as a screening measurement and not interpreted as a diagnosis.",),
    ),
    _point_policy(
        "BodyFatPercentage",
        measurement_kind=MeasurementKind.RATIO,
        units=_percent_fraction(evidence=("apple_body_fat_percentage", "project_body_composition_consistency")),
        evidence=("apple_body_fat_percentage", "project_body_composition_consistency"),
        rules=("nonnegative_measurement",),
        cross_rules=("lean_mass_weight_body_fat_consistency",),
    ),
    _point_policy(
        "Height",
        maturity=PolicyMaturity.PROVISIONAL,
        units=_unit(
            status=RawUnitStatus.PARTICIPANT_SOURCE_INFERENCE, raw_unit=None,
            candidates=("m", "cm", "in"), canonical="m",
            resolution="participant_source_epoch_length", conversion="resolved_length_to_metres",
            confidence=ConversionConfidence.PROVISIONAL,
            unresolved=UnresolvedUnitAction.EXCLUDE_FROM_DEFAULT_CANONICAL_ANALYSIS,
            evidence=("apple_healthkit_quantity_types", "project_unit_fingerprints"), precision=4,
        ),
        evidence=("apple_healthkit_quantity_types", "project_unit_fingerprints"),
        rules=("positive_measurement", "unit_epoch_discontinuity"),
        cross_rules=("bmi_height_weight_consistency",),
    ),
    _point_policy(
        "Weight",
        maturity=PolicyMaturity.PROVISIONAL,
        units=_unit(
            status=RawUnitStatus.PARTICIPANT_SOURCE_INFERENCE, raw_unit=None,
            candidates=("kg", "lb"), canonical="kg",
            resolution="participant_source_epoch_mass", conversion="resolved_mass_to_kilograms",
            confidence=ConversionConfidence.PROVISIONAL,
            unresolved=UnresolvedUnitAction.EXCLUDE_FROM_DEFAULT_CANONICAL_ANALYSIS,
            evidence=("apple_body_mass", "project_unit_fingerprints", "project_body_composition_consistency"), precision=3,
        ),
        evidence=("apple_body_mass", "project_unit_fingerprints", "project_body_composition_consistency"),
        rules=("positive_measurement", "unit_epoch_discontinuity"),
        cross_rules=("bmi_height_weight_consistency", "lean_mass_weight_body_fat_consistency"),
    ),
    _point_policy(
        "LeanBodyMass",
        maturity=PolicyMaturity.PROVISIONAL,
        units=_unit(
            status=RawUnitStatus.PARTICIPANT_SOURCE_INFERENCE, raw_unit=None,
            candidates=("kg", "lb"), canonical="kg",
            resolution="participant_source_epoch_mass", conversion="resolved_mass_to_kilograms",
            confidence=ConversionConfidence.PROVISIONAL,
            unresolved=UnresolvedUnitAction.EXCLUDE_FROM_DEFAULT_CANONICAL_ANALYSIS,
            evidence=("apple_body_mass", "project_body_composition_consistency"), precision=3,
        ),
        evidence=("apple_body_mass", "project_body_composition_consistency"),
        rules=("positive_measurement", "unit_epoch_discontinuity"),
        cross_rules=("lean_mass_weight_body_fat_consistency",),
    ),
    _point_policy(
        "WaistCircumference",
        maturity=PolicyMaturity.REVIEWED,
        units=_unit(
            status=RawUnitStatus.INFERRED_HIGH_CONFIDENCE, raw_unit="in",
            canonical="m",
            resolution="cohort_exporter_inches", conversion="inches_to_metres",
            confidence=ConversionConfidence.HIGH,
            unresolved=UnresolvedUnitAction.RETAIN_RAW_ONLY,
            evidence=("apple_waist_circumference", "project_unit_fingerprints"), precision=4,
        ),
        evidence=("apple_waist_circumference", "who_waist", "project_unit_fingerprints"),
        rules=("positive_measurement", "waist_protocol_context"),
        notes=("Unit conversion is reviewed; anatomical landmark and measurement protocol remain unknown.",),
    ),
)

# Physiological/metabolic point features.
_register(
    _point_policy(
        "OxygenSaturation",
        measurement_kind=MeasurementKind.RATIO,
        units=_percent_fraction(evidence=("apple_healthkit_units", "fda_pulse_oximetry")),
        evidence=("apple_healthkit_units", "fda_pulse_oximetry"),
        rules=("nonnegative_measurement", "pulse_ox_source_limitations"),
        provenance_strategy="device_or_application_estimate",
    ),
    _point_policy(
        "RespiratoryRate",
        units=_fixed_unit("breaths/min", "breaths/min", evidence=("apple_healthkit_quantity_types",), precision=2),
        evidence=("apple_healthkit_quantity_types",),
        rules=("positive_measurement",),
        provenance_strategy="device_or_application_estimate",
    ),
    _point_policy(
        "Vo2Max",
        units=_fixed_unit("mL/(kg*min)", "mL/(kg*min)", evidence=("apple_vo2max",), precision=2),
        evidence=("apple_vo2max",),
        rules=("positive_measurement", "vo2max_test_type"),
        provenance_strategy="vo2max_test_type",
        provenance_context=("vo2_max_test_type",),
    ),
    _point_policy(
        "BodyTemperature",
        maturity=PolicyMaturity.REVIEWED,
        units=_unit(
            status=RawUnitStatus.INFERRED_HIGH_CONFIDENCE, raw_unit="Cel",
            canonical="Cel",
            resolution="cohort_exporter_celsius", conversion="identity",
            confidence=ConversionConfidence.HIGH,
            unresolved=UnresolvedUnitAction.RETAIN_RAW_ONLY,
            evidence=("apple_body_temperature", "project_unit_fingerprints"), precision=3,
        ),
        evidence=("apple_body_temperature", "project_unit_fingerprints"),
        rules=("temperature_sensor_location",),
        source_rules=(SourceRule(
            "fevertracker_import",
            source_id="com.links.fevertracker",
            acquisition_method=AcquisitionMethod.THIRD_PARTY_IMPORT,
            evidence_refs=("project_unit_fingerprints",),
        ),),
    ),
    _point_policy(
        "PeakFlow",
        maturity=PolicyMaturity.PROVISIONAL,
        units=_unit(
            status=RawUnitStatus.INFERRED_PROVISIONAL, raw_unit="L/min",
            candidates=("L/min",), canonical="L/min",
            resolution="source_specific_peak_flow", conversion="identity",
            confidence=ConversionConfidence.PROVISIONAL,
            unresolved=UnresolvedUnitAction.RETAIN_RAW_AND_CANDIDATES,
            evidence=("apple_peak_flow", "ats_ers_spirometry_2019"), precision=1,
        ),
        evidence=("apple_peak_flow", "ats_ers_spirometry_2019"),
        rules=("positive_measurement", "peak_flow_session_context"),
    ),
)

# CGM has richer source/context policy and source-epoch unit inference.
_register(FeaturePolicy(
    identity=_identity("BloodGlucose", maturity=PolicyMaturity.PROVISIONAL),
    schema=_scalar_schema("point_scalar", context=("status", "trend_arrow", "trend_rate", "time_zone", "metadata_device_name")),
    semantics=_semantics(EventKind.POINT, MeasurementKind.INTENSIVE_VALUE, DurationModel.ZERO_DURATION_POINT, IntervalClosure.POINT, native_resolution="instantaneous CGM or glucose point", end_meaning="same instant as native start"),
    units=_unit(
        status=RawUnitStatus.PARTICIPANT_SOURCE_INFERENCE, raw_unit=None,
        candidates=("mg/dL", "mmol/L"), canonical="mmol/L",
        resolution="participant_source_epoch_glucose", conversion="resolved_glucose_to_mmol_l",
        confidence=ConversionConfidence.PROVISIONAL,
        unresolved=UnresolvedUnitAction.EXCLUDE_FROM_DEFAULT_CANONICAL_ANALYSIS,
        evidence=("apple_blood_glucose", "project_cgm_metadata", "ada_cgm_2026"), precision=3,
    ),
    provenance=_provenance("cgm_source_status_trend", context=("status", "trend_arrow", "trend_rate", "time_zone", "metadata_device_name")),
    reconciliation=_reconcile(),
    curation=_curation("point_duration_zero", "positive_measurement", "unit_resolved_for_canonical_value", "unit_epoch_discontinuity", "cgm_status_vocabulary", "cgm_trend_vocabulary", "cgm_time_zone_context", "source_epoch_transition"),
    cross_feature=CrossFeaturePolicy(),
    output=_output(context=("status", "trend_arrow", "trend_rate", "time_zone", "metadata_device_name")),
    resampling=_resampling(ResamplingSupport.OPTIONAL, "point_retain", point_assignment="retain_native_points_by_default", supports=("point_count", "native_cadence_seconds", "source_count")),
    evidence_refs=("apple_blood_glucose", "project_cgm_metadata", "ada_cgm_2026"),
    tests=_tests("BloodGlucose", "cgm_status_trend_and_timezone_are_preserved", "unit_is_not_guessed_globally"),
    notes=("Native CGM cadence does not convert points into measurement intervals.",),
))

# BAC requires source-specific acquisition-method handling.
_register(_point_policy(
    "BloodAlcoholContent",
    maturity=PolicyMaturity.REVIEWED,
    measurement_kind=MeasurementKind.RATIO,
    units=_percent_fraction(evidence=("apple_healthkit_units", "project_bac_sources"), precision=4),
    evidence=("apple_healthkit_units", "project_bac_sources"),
    rules=("nonnegative_measurement", "bac_calculator_estimate"),
    provenance_strategy="bac_source_regime",
    source_rules=(
        SourceRule(
            "intellidrink_calculator",
            source_id="com.rwichmann.intellidrink-lite",
            acquisition_method=AcquisitionMethod.CALCULATOR_ESTIMATE,
            status_effect=CurationStatus.REVIEW,
            inclusion_effect=InclusionPolicy.EXCLUDE,
            evidence_refs=("project_bac_sources",),
            note="Modeled calculator trajectory, not a direct wearable measurement.",
        ),
        SourceRule(
            "apple_health_manual_bac",
            source_id="com.apple.Health",
            metadata_equals=(("was_user_entered", "true"),),
            acquisition_method=AcquisitionMethod.USER_ENTERED,
            status_effect=CurationStatus.REVIEW,
            inclusion_effect=InclusionPolicy.INCLUDE,
            evidence_refs=("project_bac_sources",),
        ),
    ),
    notes=("Calculator and manual regimes must remain stratified.",),
))

# Nutrition events preserve distinct UUIDs.
def _nutrition_policy(name: str, raw: str, canonical: str, *, energy: bool = False) -> FeaturePolicy:
    units = (
        _unit(
            status=RawUnitStatus.PARTICIPANT_SOURCE_INFERENCE, raw_unit=None,
            candidates=("kcal", "kJ"), canonical="kcal",
            resolution="participant_source_epoch_energy", conversion="resolved_energy_to_kcal",
            confidence=ConversionConfidence.PROVISIONAL,
            unresolved=UnresolvedUnitAction.RETAIN_RAW_AND_CANDIDATES,
            evidence=("apple_nutrition",), precision=3,
        ) if energy else _fixed_unit(raw, canonical, evidence=("apple_nutrition",), precision=3)
    )
    return FeaturePolicy(
        identity=_identity(name, maturity=PolicyMaturity.PROVISIONAL if energy else PolicyMaturity.REVIEWED),
        schema=_scalar_schema("nutrition_event"),
        semantics=_semantics(EventKind.NUTRITION_EVENT, MeasurementKind.EVENT_AMOUNT, DurationModel.ZERO_DURATION_POINT, IntervalClosure.POINT, native_resolution="discrete nutrition entry", end_meaning="same instant as native start"),
        units=units,
        provenance=_provenance("nutrition_source_and_user_entry"),
        reconciliation=_reconcile(duplicate="inherit_native_record_id_only", note="Different UUIDs remain distinct even with equal timestamp and value."),
        curation=_curation("point_duration_zero", "nutrition_preserve_distinct_uuid", "nutrition_nonnegative", "unit_resolved_for_canonical_value", "manual_entry_context", "source_epoch_transition"),
        cross_feature=CrossFeaturePolicy(),
        output=_output(),
        resampling=_resampling(ResamplingSupport.OPTIONAL, "nutrition_event_retain", point_assignment="retain_native_events", supports=("event_count",)),
        evidence_refs=("apple_nutrition",),
        tests=_tests(name, "distinct_record_ids_are_not_collapsed"),
    )

_register(
    _nutrition_policy("EnergyConsumed", "kcal", "kcal", energy=True),
    _nutrition_policy("Carbohydrates", "g", "g"),
    _nutrition_policy("Protein", "g", "g"),
    _nutrition_policy("TotalFat", "g", "g"),
)

# Sleep state intervals.
_register(FeaturePolicy(
    identity=_identity("Sleep"),
    schema=SchemaPolicy(
        feature_family="state_interval",
        accepted_payload_shapes=("list[dict]", "dict"),
        required_columns=_COMMON_REQUIRED + ("value",),
        optional_columns=_COMMON_OPTIONAL,
        measurements=(_measurement("value", "sleep_state", value_kind="categorical"),),
        categorical_columns=("value",),
        missing_measurement_action=MissingFieldAction.EXCLUDE_DEFAULT,
    ),
    semantics=_semantics(EventKind.STATE_INTERVAL, MeasurementKind.CATEGORICAL_STATE, DurationModel.EXPLICIT_INTERVAL, IntervalClosure.HALF_OPEN, native_resolution="native categorical sleep-state interval", date_anchor=DateAnchor.EVENT_LOCAL_DATE),
    units=_no_unit("sleep_state"),
    provenance=_provenance("sleep_source_state"),
    reconciliation=_reconcile(duplicate="inherit_native_sleep_occurrence"),
    curation=CurationPolicy(
        ("required_measurements_present", "timestamp_order", "interval_duration_positive",
         "sleep_state_vocabulary", "sleep_same_source_same_state_overlap",
         "sleep_same_source_detailed_stage_conflict", "sleep_cross_source_overlap",
         "sleep_compatible_inbed_support", "source_epoch_transition"),
        CurationStatus.PASS, InclusionPolicy.INCLUDE,
    ),
    cross_feature=CrossFeaturePolicy(),
    output=_output(primary=("value",), categorical=(("value", "string"),)),
    resampling=_resampling(ResamplingSupport.SUPPORTED, "state_interval_union", aggregation="none", state_handling="multi_state_interval_union_and_coverage", supports=("state_seconds", "coverage_fraction", "conflict_seconds")),
    evidence_refs=("apple_sleep_analysis", "project_sleep_intervals"),
    tests=_tests("Sleep", "no_lexical_state_selection", "all_native_state_intervals_are_preserved"),
    notes=("INBED may overlap detailed stages; detailed stage conflicts are reported rather than resolved lexically.",),
))

# ECG waveform events.
_ecg_measurements = (
    _measurement("average_heart_rate", "average_heart_rate"),
    _measurement("sampling_frequency", "sampling_frequency"),
    _measurement("voltage_measurements", "waveform", value_kind="nested_numeric"),
)
_register(FeaturePolicy(
    identity=_identity("Electrocardiogram"),
    schema=SchemaPolicy(
        feature_family="waveform",
        accepted_payload_shapes=("list[dict]", "dict"),
        required_columns=_COMMON_REQUIRED + tuple(field.column for field in _ecg_measurements),
        optional_columns=_COMMON_OPTIONAL + ("classification", "algorithm_version", "waveform_sample_count"),
        measurements=_ecg_measurements,
        categorical_columns=("classification", "algorithm_version"),
        context_columns=("classification", "algorithm_version", "waveform_sample_count"),
        waveform_columns=("voltage_measurements",),
        missing_measurement_action=MissingFieldAction.EXCLUDE_DEFAULT,
    ),
    semantics=_semantics(EventKind.WAVEFORM, MeasurementKind.SIGNAL, DurationModel.INTERNAL_SIGNAL_CLOCK, IntervalClosure.HALF_OPEN, native_resolution="native waveform sampling clock", date_anchor=DateAnchor.INTERNAL_SIGNAL_START),
    units=UnitPolicy((
        MeasurementUnitPolicy("average_heart_rate", RawUnitStatus.SOURCE_CONVENTION, "beats/min", (), "beats/min", "fixed_source_convention", "identity", ConversionConfidence.EXACT, UnresolvedUnitAction.RETAIN_RAW_ONLY, ("apple_ecg",), 2),
        MeasurementUnitPolicy("sampling_frequency", RawUnitStatus.SOURCE_CONVENTION, "Hz", (), "Hz", "fixed_source_convention", "identity", ConversionConfidence.EXACT, UnresolvedUnitAction.RETAIN_RAW_ONLY, ("apple_ecg_sampling_frequency",), 3),
        MeasurementUnitPolicy("waveform", RawUnitStatus.UNKNOWN, None, (), None, "waveform_unit_unresolved", "not_applicable", ConversionConfidence.NONE, UnresolvedUnitAction.RETAIN_RAW_ONLY, ("apple_ecg",), None),
    )),
    provenance=_provenance("ecg_source_device_algorithm", context=("classification", "algorithm_version")),
    reconciliation=_reconcile(identity="inherit_native_waveform_identity", duplicate="inherit_native_waveform_occurrence"),
    curation=_curation("interval_duration_positive", "ecg_waveform_shape", "ecg_sample_count_consistency", "ecg_relative_time_monotonic", "ecg_finite_amplitude", "source_epoch_transition"),
    cross_feature=CrossFeaturePolicy(),
    output=_output(
        primary=tuple(field.column for field in _ecg_measurements),
        context=("classification", "algorithm_version", "waveform_sample_count"),
        derived=("expected_waveform_sample_count", "waveform_integrity_status", "acquisition_method", "curation_status", "curation_flags", "include_by_default"),
        prefix=("start_date", "end_date") + tuple(field.column for field in _ecg_measurements) + ("classification", "algorithm_version", "datetime", "created_at", "updated_at", "data_source", "collecting_method_version"),
    ),
    resampling=_resampling(ResamplingSupport.WAVEFORM_SPECIFIC_ONLY, "waveform_native_only", aggregation="none", broadcasting="prohibited", supports=("waveform_event_reference",)),
    evidence_refs=("apple_ecg", "apple_ecg_sampling_frequency"),
    tests=_tests("Electrocardiogram", "one_native_ecg_remains_one_waveform_event", "waveform_is_never_broadcast_into_ordinary_windows"),
))

# Mindful session duration events.
_register(FeaturePolicy(
    identity=_identity("Mindful"),
    schema=SchemaPolicy(
        feature_family="duration_event",
        accepted_payload_shapes=("list[dict]", "dict"),
        required_columns=_COMMON_REQUIRED,
        optional_columns=_COMMON_OPTIONAL + ("value",),
        measurements=(),
        missing_measurement_action=MissingFieldAction.COPY_UNCHANGED,
    ),
    semantics=_semantics(EventKind.DURATION_EVENT, MeasurementKind.DURATION, DurationModel.EXPLICIT_INTERVAL, IntervalClosure.HALF_OPEN, native_resolution="native mindful-session duration"),
    units=_no_unit("duration"),
    provenance=_provenance("mindful_session_source"),
    reconciliation=_reconcile(),
    curation=CurationPolicy(("timestamp_order", "mindful_interval_valid", "source_epoch_transition"), CurationStatus.PASS, InclusionPolicy.INCLUDE),
    cross_feature=CrossFeaturePolicy(),
    output=_output(primary=(), derived=("acquisition_method", "curation_status", "curation_flags", "include_by_default"), prefix=("start_date", "end_date", "datetime", "created_at", "updated_at", "data_source", "collecting_method_version")),
    resampling=_resampling(ResamplingSupport.SUPPORTED, "duration_interval_coverage", aggregation="duration_union", allocation="exact_overlap_seconds", supports=("overlap_seconds", "coverage_fraction")),
    evidence_refs=("apple_mindful_session",),
    tests=_tests("Mindful", "native_session_duration_is_preserved"),
))

# Policy-calibration gates. Reviewed policies default to convergent evidence and reviewed execution. Provisional
# policies receive explicit source/audit gates.
_CALIBRATION_AUDITS: dict[str, tuple[str, ...]] = {
    "ActivitySummary": ("activity_summary_sliding_pair_alignment", "exporter_date_component_review"),
    "BloodAlcoholContent": ("bac_source_regime_inventory", "bac_fraction_encoding_confirmation"),
    "BloodGlucose": ("glucose_source_epoch_unit_audit", "glucose_scale_transition_audit"),
    "BodyTemperature": ("temperature_source_epoch_unit_audit", "temperature_sensor_location_coverage"),
    "DailyDistanceCycling": ("cycling_distance_unit_audit", "cycling_implied_speed_audit"),
    "EnergyConsumed": ("dietary_energy_unit_audit", "nutrition_source_regime_audit", "nutrition_energy_macronutrient_consistency"),
    "HeartRateVariability": ("hrv_source_version_unit_audit", "hrv_algorithm_epoch_audit"),
    "Height": ("anthropometric_unit_epoch_audit", "bmi_height_weight_consistency"),
    "LeanBodyMass": ("anthropometric_unit_epoch_audit", "lean_mass_weight_body_fat_consistency"),
    "PeakFlow": ("peak_flow_unit_inventory", "peak_flow_source_context_audit"),
    "WaistCircumference": ("waist_unit_epoch_audit", "waist_protocol_context_audit"),
    "Weight": ("anthropometric_unit_epoch_audit", "bmi_height_weight_consistency", "lean_mass_weight_body_fat_consistency"),
}

_DECISION_CALIBRATIONS: dict[str, PolicyCalibration] = {
    name: PolicyCalibration(
        evidence_grade=decision.evidence_grade,
        execution_mode=decision.execution_mode,
        rationale=decision.decision,
        required_audits=_CALIBRATION_AUDITS.get(name, ()),
        safe_fallback=decision.safe_fallback,
        source_scope=decision.source_scope,
        decision_version="full-cohort-audit-1",
    )
    for name, decision in POLICY_CALIBRATION_DECISIONS.items()
}

for _feature_name, _policy in tuple(_POLICIES.items()):
    _decision_value = POLICY_CALIBRATION_DECISIONS.get(_feature_name)
    if _decision_value is not None and _policy.identity.maturity is not _decision_value.maturity:
        _policy = replace(_policy, identity=replace(_policy.identity, maturity=_decision_value.maturity))
    _calibration_value = _DECISION_CALIBRATIONS.get(_feature_name)
    if _calibration_value is None:
        _calibration_value = _calibration(
            EvidenceGrade.B_CONVERGENT,
            PolicyExecutionMode.REVIEWED_EXECUTION,
            "Reviewed feature semantics with convergent official technical, clinical-context, and project evidence.",
        )
    _POLICIES[_feature_name] = replace(_policy, calibration=_calibration_value)


# Unknown fallback: explicit, safe, and never silently converted.
UNKNOWN_POLICY = FeaturePolicy(
    identity=IdentityPolicy(
        canonical_name="__UNKNOWN__",
        aliases=(),
        data_sources=("AppleHealthkit",),
        collecting_method_versions=("*",),
        policy_version=POLICY_CONTRACT_VERSION,
        maturity=PolicyMaturity.UNKNOWN,
        cohort_observed=False,
    ),
    schema=SchemaPolicy(
        feature_family="unknown",
        accepted_payload_shapes=("list[dict]", "dict", "scalar", "unknown"),
        required_columns=("data_source", "collecting_method_version"),
        optional_columns=_COMMON_OPTIONAL + _COMMON_REQUIRED + ("value",),
        measurements=(),
        missing_measurement_action=MissingFieldAction.COPY_UNCHANGED,
    ),
    semantics=_semantics(EventKind.UNKNOWN, MeasurementKind.UNKNOWN, DurationModel.UNKNOWN, IntervalClosure.UNKNOWN, native_resolution="unknown", date_anchor=DateAnchor.UNKNOWN),
    units=_no_unit(),
    provenance=_provenance("preserve_source_context"),
    reconciliation=_reconcile(),
    curation=CurationPolicy(("unknown_feature_policy",), CurationStatus.REVIEW, InclusionPolicy.EXCLUDE),
    cross_feature=CrossFeaturePolicy(),
    output=_output(),
    resampling=_resampling(ResamplingSupport.UNSUPPORTED, "unknown_unsupported"),
    evidence_refs=(),
    tests=_tests("unknown", "unknown_feature_is_copied_without_canonical_conversion", "unknown_feature_is_excluded_by_default"),
    calibration=_calibration(
        EvidenceGrade.D_UNRESOLVED,
        PolicyExecutionMode.BLOCK_DERIVATION,
        "No reviewed policy exists for this feature.",
        audits=("unknown_feature_schema_and_semantics_review",),
        fallback="Copy native rows unchanged, mark review, exclude from default curated analyses, and prohibit automatic resampling.",
        scope="unknown_feature",
    ),
    notes=("Fallback is visible and non-destructive; it never guesses a unit or resampling rule.",),
)

CURATION_POLICIES: Mapping[str, FeaturePolicy] = dict(sorted(_POLICIES.items()))


def known_curation_features() -> tuple[str, ...]:
    return tuple(CURATION_POLICIES)


def get_policy(feature: str, *, allow_fallback: bool = True) -> FeaturePolicy:
    policy = CURATION_POLICIES.get(feature)
    if policy is not None:
        return policy
    if allow_fallback:
        return replace(UNKNOWN_POLICY, identity=replace(UNKNOWN_POLICY.identity, canonical_name=feature))
    raise KeyError(feature)


def _all_evidence_refs(policy: FeaturePolicy) -> set[str]:
    refs = set(policy.evidence_refs)
    for unit in policy.units.measurements:
        refs.update(unit.evidence_refs)
    for source_rule in policy.provenance.source_rules:
        refs.update(source_rule.evidence_refs)
    for rule_id in policy.curation.rule_ids:
        rule = RULES.get(rule_id)
        if rule is not None:
            refs.update(rule.evidence_refs)
    return refs


def validate_registry() -> RegistryValidationResult:
    errors: list[str] = list(validate_decisions())
    warnings: list[str] = []

    observed = set(COHORT_OBSERVED_FEATURES)
    actual = set(CURATION_POLICIES)
    missing = sorted(observed - actual)
    extra = sorted(actual - observed)
    if missing:
        errors.append(f"Missing cohort-observed policies: {missing}")
    if extra:
        errors.append(f"Unexpected policies marked as cohort-observed: {extra}")

    fingerprints: dict[str, str] = {}
    for name, policy in CURATION_POLICIES.items():
        prefix = f"{name}: "
        if policy.name != name:
            errors.append(prefix + f"identity canonical_name is {policy.name!r}")
        if not policy.identity.cohort_observed:
            errors.append(prefix + "cohort policy is not marked cohort_observed")
        if policy.identity.maturity is PolicyMaturity.UNKNOWN:
            errors.append(prefix + "cohort policy maturity cannot be unknown")
        if not policy.schema.feature_family:
            errors.append(prefix + "feature_family is empty")
        if not policy.schema.accepted_payload_shapes:
            errors.append(prefix + "accepted_payload_shapes is empty")
        if not policy.semantics.native_resolution:
            errors.append(prefix + "native_resolution is empty")
        if policy.resampling.default_enabled:
            errors.append(prefix + "resampling must not be enabled in milestone 2")
        if policy.resampling.strategy not in ALL_STRATEGY_CATALOGS["resampling"]:
            errors.append(prefix + f"unknown resampling strategy {policy.resampling.strategy!r}")
        if policy.provenance.acquisition_method_strategy not in ALL_STRATEGY_CATALOGS["acquisition"]:
            errors.append(prefix + f"unknown acquisition strategy {policy.provenance.acquisition_method_strategy!r}")
        if policy.provenance.source_epoch_strategy not in ALL_STRATEGY_CATALOGS["source_epoch"]:
            errors.append(prefix + f"unknown source-epoch strategy {policy.provenance.source_epoch_strategy!r}")
        if policy.provenance.source_priority_strategy not in ALL_STRATEGY_CATALOGS["source_priority"]:
            errors.append(prefix + f"unknown source-priority strategy {policy.provenance.source_priority_strategy!r}")
        if policy.reconciliation.occurrence_identity_strategy not in ALL_STRATEGY_CATALOGS["occurrence_identity"]:
            errors.append(prefix + f"unknown occurrence identity strategy {policy.reconciliation.occurrence_identity_strategy!r}")
        if policy.reconciliation.exact_duplicate_strategy not in ALL_STRATEGY_CATALOGS["duplicate"]:
            errors.append(prefix + f"unknown duplicate strategy {policy.reconciliation.exact_duplicate_strategy!r}")
        if policy.reconciliation.revision_strategy not in ALL_STRATEGY_CATALOGS["revision"]:
            errors.append(prefix + f"unknown revision strategy {policy.reconciliation.revision_strategy!r}")

        measurement_roles = {item.role for item in policy.schema.measurements}
        measurement_columns = {item.column for item in policy.schema.measurements}
        for unit in policy.units.measurements:
            if unit.measurement not in measurement_roles and unit.measurement not in measurement_columns and unit.measurement not in {"duration", "sleep_state"}:
                errors.append(prefix + f"unit policy references unknown measurement {unit.measurement!r}")
            if unit.unit_resolution_strategy not in ALL_STRATEGY_CATALOGS["unit_resolution"]:
                errors.append(prefix + f"unknown unit resolution strategy {unit.unit_resolution_strategy!r}")
            if unit.conversion_rule not in ALL_STRATEGY_CATALOGS["conversion"]:
                errors.append(prefix + f"unknown conversion rule {unit.conversion_rule!r}")
            if unit.canonical_unit is not None and unit.conversion_rule == "not_applicable":
                errors.append(prefix + f"canonical unit {unit.canonical_unit!r} has no conversion rule")

        for rule_id in policy.curation.rule_ids:
            if rule_id not in RULES:
                errors.append(prefix + f"unknown curation rule {rule_id!r}")
                continue
            try:
                rule_execution_kind(rule_id)
            except KeyError:
                errors.append(prefix + f"curation rule {rule_id!r} has no implementation classification")
        for cross_id in policy.cross_feature.rule_ids:
            if cross_id not in ALL_STRATEGY_CATALOGS["cross_feature"]:
                errors.append(prefix + f"unknown cross-feature strategy {cross_id!r}")
        for ref in sorted(_all_evidence_refs(policy)):
            if ref not in EVIDENCE:
                errors.append(prefix + f"unknown evidence reference {ref!r}")

        if not policy.tests.fixture_ids:
            errors.append(prefix + "has no policy fixture IDs")
        if not policy.tests.invariants:
            errors.append(prefix + "has no policy invariants")
        calibration = policy.calibration
        if not calibration.rationale.strip():
            errors.append(prefix + "calibration rationale is empty")
        if not calibration.safe_fallback.strip():
            errors.append(prefix + "calibration safe_fallback is empty")
        if policy.identity.maturity is PolicyMaturity.PROVISIONAL:
            warnings.append(prefix + "policy is provisional and must remain visible in curated output/reporting")
            if calibration.execution_mode is PolicyExecutionMode.REVIEWED_EXECUTION:
                errors.append(prefix + "provisional policy cannot use reviewed_execution")
        if calibration.evidence_grade is EvidenceGrade.D_UNRESOLVED and calibration.execution_mode is not PolicyExecutionMode.BLOCK_DERIVATION:
            errors.append(prefix + "D_unresolved evidence must block derivation")
        if calibration.execution_mode is PolicyExecutionMode.SOURCE_SPECIFIC_EXECUTION:
            if not calibration.required_audits:
                errors.append(prefix + "source-specific execution requires at least one calibration audit")
            if calibration.source_scope == "all_registered_sources":
                errors.append(prefix + "source-specific execution requires a non-global source_scope")
        if calibration.execution_mode in {
            PolicyExecutionMode.CONSERVATIVE_ANNOTATION,
            PolicyExecutionMode.BLOCK_DERIVATION,
        } and not calibration.required_audits and policy.identity.maturity is PolicyMaturity.PROVISIONAL:
            warnings.append(prefix + "conservative/block policy has no required audit declaration")

        fingerprint = policy.fingerprint()
        if fingerprint in fingerprints:
            errors.append(prefix + f"policy fingerprint collides with {fingerprints[fingerprint]!r}")
        fingerprints[fingerprint] = name

    for feature, decision in POLICY_CALIBRATION_DECISIONS.items():
        policy = CURATION_POLICIES.get(feature)
        if policy is None:
            errors.append(f"{feature}: calibration decision has no policy")
            continue
        if policy.identity.maturity is not decision.maturity:
            errors.append(f"{feature}: policy maturity does not match calibration decision")
        if policy.calibration.evidence_grade is not decision.evidence_grade:
            errors.append(f"{feature}: evidence grade does not match calibration decision")
        if policy.calibration.execution_mode is not decision.execution_mode:
            errors.append(f"{feature}: execution mode does not match calibration decision")

    if UNKNOWN_POLICY.identity.cohort_observed:
        errors.append("Unknown fallback must not be cohort_observed")
    if UNKNOWN_POLICY.identity.maturity is not PolicyMaturity.UNKNOWN:
        errors.append("Unknown fallback maturity must be unknown")
    if UNKNOWN_POLICY.curation.default_status is not CurationStatus.REVIEW:
        errors.append("Unknown fallback must default to review")
    if UNKNOWN_POLICY.curation.default_inclusion is not InclusionPolicy.EXCLUDE:
        errors.append("Unknown fallback must be excluded by default")
    if UNKNOWN_POLICY.resampling.support is not ResamplingSupport.UNSUPPORTED:
        errors.append("Unknown fallback resampling must be unsupported")
    if UNKNOWN_POLICY.calibration.evidence_grade is not EvidenceGrade.D_UNRESOLVED:
        errors.append("Unknown fallback evidence grade must be D_unresolved")
    if UNKNOWN_POLICY.calibration.execution_mode is not PolicyExecutionMode.BLOCK_DERIVATION:
        errors.append("Unknown fallback must block derivation")

    return RegistryValidationResult(tuple(errors), tuple(warnings))


def registry_fingerprint() -> str:
    payload = {
        "registry_version": CURATION_REGISTRY_VERSION,
        "policies": {name: policy.as_dict() for name, policy in CURATION_POLICIES.items()},
        "unknown_policy": UNKNOWN_POLICY.as_dict(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def registry_payload(
    *, features: Iterable[str] | None = None, include_evidence: bool = False, include_rules: bool = False,
    include_unknown: bool = True,
) -> dict[str, object]:
    selected_names = tuple(features) if features is not None else known_curation_features()
    selected = {name: get_policy(name, allow_fallback=False).as_dict() for name in selected_names}
    validation = validate_registry()
    payload: dict[str, object] = {
        "registry_version": CURATION_REGISTRY_VERSION,
        "policy_contract_version": POLICY_CONTRACT_VERSION,
        "registry_fingerprint": registry_fingerprint(),
        "policy_count": len(CURATION_POLICIES),
        "cohort_observed_feature_count": len(COHORT_OBSERVED_FEATURES),
        "cohort_observed_features": list(COHORT_OBSERVED_FEATURES),
        "validation": {"valid": validation.valid, "errors": list(validation.errors), "warnings": list(validation.warnings)},
        "policies": selected,
    }
    if include_unknown:
        payload["unknown_policy"] = UNKNOWN_POLICY.as_dict()
    if include_evidence:
        payload["evidence"] = {key: to_primitive(value) for key, value in sorted(EVIDENCE.items())}
    if include_rules:
        payload["rules"] = {key: to_primitive(value) for key, value in sorted(RULES.items())}
    return payload


def _unit_summary(policy: FeaturePolicy) -> str:
    parts: list[str] = []
    for item in policy.units.measurements:
        target = item.canonical_unit or "raw-only"
        source = item.raw_unit or "/".join(item.raw_unit_candidates) or "n/a"
        parts.append(f"{item.measurement}:{source}->{target} [{item.raw_unit_status.value}]")
    return "; ".join(parts)


def _measurement_summary(policy: FeaturePolicy) -> str:
    return "; ".join(f"{item.role}={item.column}" for item in policy.schema.measurements) or "none"


def matrix_rows(features: Iterable[str] | None = None) -> list[dict[str, object]]:
    names = tuple(features) if features is not None else known_curation_features()
    rows: list[dict[str, object]] = []
    for name in names:
        policy = get_policy(name, allow_fallback=False)
        rows.append({
            "feature": name,
            "family": policy.schema.feature_family,
            "maturity": policy.identity.maturity.value,
            "policy_version": policy.identity.policy_version,
            "evidence_grade": policy.calibration.evidence_grade.value,
            "execution_mode": policy.calibration.execution_mode.value,
            "required_audits": ";".join(policy.calibration.required_audits),
            "calibration_scope": policy.calibration.source_scope,
            "event_kind": policy.semantics.event_kind.value,
            "measurement_kind": policy.semantics.measurement_kind.value,
            "duration_model": policy.semantics.duration_model.value,
            "measurements": _measurement_summary(policy),
            "unit_policy": _unit_summary(policy),
            "acquisition_strategy": policy.provenance.acquisition_method_strategy,
            "user_entered_handling": policy.provenance.user_entered_handling.value,
            "duplicate_strategy": policy.reconciliation.exact_duplicate_strategy,
            "revision_strategy": policy.reconciliation.revision_strategy,
            "conflict_behavior": policy.reconciliation.conflict_behavior.value,
            "curation_rules": ";".join(policy.curation.rule_ids),
            "default_status": policy.curation.default_status.value,
            "default_inclusion": policy.curation.default_inclusion.value,
            "cross_feature_rules": ";".join(policy.cross_feature.rule_ids),
            "output_primary_columns": ";".join(policy.output.primary_measurement_columns),
            "resampling_support": policy.resampling.support.value,
            "resampling_strategy": policy.resampling.strategy,
            "resampling_default_enabled": policy.resampling.default_enabled,
            "evidence_refs": ";".join(policy.evidence_refs),
            "policy_fingerprint": policy.fingerprint(),
        })
    return rows


def matrix_csv(features: Iterable[str] | None = None) -> str:
    rows = matrix_rows(features)
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def matrix_table(features: Iterable[str] | None = None) -> str:
    rows = matrix_rows(features)
    headers = ("feature", "family", "maturity", "evidence_grade", "execution_mode", "event_kind", "default_status")
    widths = {header: max(len(header), *(len(str(row[header])) for row in rows)) for header in headers}
    def render(row: Mapping[str, object]) -> str:
        return "  ".join(str(row[header]).ljust(widths[header]) for header in headers)
    lines = [render({header: header for header in headers}), render({header: "-" * widths[header] for header in headers})]
    lines.extend(render(row) for row in rows)
    validation = validate_registry()
    lines.extend(("", f"Registry: {CURATION_REGISTRY_VERSION}", f"Policies: {len(CURATION_POLICIES)}", f"Valid: {validation.valid}", f"Provisional policies: {sum(policy.identity.maturity is PolicyMaturity.PROVISIONAL for policy in CURATION_POLICIES.values())}"))
    return "\n".join(lines)
