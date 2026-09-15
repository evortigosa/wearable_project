"""
Wearable Data Processing and Modeling project
Reviewed rule declarations referenced by feature policies. Rules are declarations in 0.2.0a1. Their strategy
implementations are added by later sub-releases. Keeping flags and effects here prevents each feature from
redefining the meaning of a warning.
"""


from __future__ import annotations
from wearable_project.curation.models import InclusionEffect, RuleClass, RuleDefinition, Severity, StatusEffect


RULES: dict[str, RuleDefinition] = {}


def _add(*rules: RuleDefinition) -> None:
    for rule in rules:
        if rule.rule_id in RULES:
            raise ValueError(f"Duplicate curation rule ID: {rule.rule_id}")
        RULES[rule.rule_id] = rule


_add(
    RuleDefinition(
        "required_measurements_present", RuleClass.STRUCTURAL, "required_measurements_present",
        Severity.ERROR, "missing_required_measurement", StatusEffect.EXCLUDE_DEFAULT,
        InclusionEffect.EXCLUDE_DEFAULT, True,
        description="A required feature measurement is absent or null.",
    ),
    RuleDefinition(
        "finite_numeric", RuleClass.STRUCTURAL, "finite_numeric", Severity.ERROR,
        "nonfinite_measurement", StatusEffect.EXCLUDE_DEFAULT,
        InclusionEffect.EXCLUDE_DEFAULT, True,
        description="A numeric measurement is NaN or infinite.",
    ),
    RuleDefinition(
        "timestamp_order", RuleClass.TEMPORAL, "timestamp_order", Severity.ERROR,
        "negative_native_interval", StatusEffect.EXCLUDE_DEFAULT,
        InclusionEffect.EXCLUDE_DEFAULT, True,
        description="Native end time precedes start time.",
    ),
    RuleDefinition(
        "point_duration_zero", RuleClass.TEMPORAL, "point_duration_zero", Severity.WARNING,
        "point_has_nonzero_duration", StatusEffect.REVIEW, InclusionEffect.KEEP, True,
        description="A feature declared as a point has nonzero native duration.",
    ),
    RuleDefinition(
        "interval_duration_positive", RuleClass.TEMPORAL, "interval_duration_positive",
        Severity.WARNING, "interval_has_zero_duration", StatusEffect.REVIEW,
        InclusionEffect.KEEP, True,
        description="An interval feature has zero duration.",
    ),
    RuleDefinition(
        "nonnegative_measurement", RuleClass.PHYSIOLOGICAL_SCREENING,
        "nonnegative_measurement", Severity.WARNING, "negative_measurement",
        StatusEffect.REVIEW, InclusionEffect.KEEP, True,
        description="A measurement expected to be nonnegative is negative.",
    ),
    RuleDefinition(
        "positive_measurement", RuleClass.PHYSIOLOGICAL_SCREENING, "positive_measurement",
        Severity.WARNING, "nonpositive_measurement", StatusEffect.REVIEW,
        InclusionEffect.KEEP, True,
        description="A measurement normally expected to be positive is non-positive.",
    ),
    RuleDefinition(
        "unresolved_same_interval_conflict", RuleClass.SOURCE,
        "unresolved_same_interval_conflict", Severity.WARNING,
        "unresolved_same_interval_conflict", StatusEffect.REVIEW,
        InclusionEffect.EXCLUDE_DEFAULT, False,
        ("project_interval_boundary_revisions",),
        "Multiple conflicting values remain for the same native interval.",
    ),
    RuleDefinition(
        "unit_resolved_for_canonical_value", RuleClass.UNIT,
        "unit_resolved_for_canonical_value", Severity.ERROR,
        "canonical_value_without_resolved_unit", StatusEffect.EXCLUDE_DEFAULT,
        InclusionEffect.EXCLUDE_DEFAULT, True,
        description="A canonical value must never be emitted without a resolved unit.",
    ),
    RuleDefinition(
        "unit_epoch_discontinuity", RuleClass.UNIT, "unit_epoch_discontinuity",
        Severity.WARNING, "possible_unit_transition", StatusEffect.REVIEW,
        InclusionEffect.KEEP, False,
        description="A participant/source epoch shows a scale change compatible with a unit transition.",
    ),
    RuleDefinition(
        "manual_entry_context", RuleClass.PROVENANCE, "manual_entry_context",
        Severity.INFO, "user_entered_observation", StatusEffect.NONE,
        InclusionEffect.KEEP, True,
        description="The observation was explicitly user-entered.",
    ),
    RuleDefinition(
        "source_epoch_transition", RuleClass.SOURCE, "source_epoch_transition",
        Severity.INFO, "source_epoch_transition", StatusEffect.NONE,
        InclusionEffect.KEEP, False,
        description="The participant changed source, device, version or time-zone epoch.",
    ),
)

_add(
    RuleDefinition(
        "heart_rate_motion_context", RuleClass.PROVENANCE,
        "heart_rate_motion_context_vocabulary", Severity.INFO,
        "unknown_heart_rate_motion_context", StatusEffect.REVIEW,
        InclusionEffect.KEEP, True, ("apple_heart_rate",),
        "Validate known motion-context values while preserving unknown values.",
    ),
    RuleDefinition(
        "heart_rate_summary_interval", RuleClass.TEMPORAL,
        "heart_rate_summary_interval_context", Severity.INFO,
        "long_summary_interval", StatusEffect.NONE, InclusionEffect.KEEP, True,
        ("apple_walking_heart_rate",),
        "Long heart-rate summaries must not be interpreted as dense sampling.",
    ),
    RuleDefinition(
        "hrv_sdnn_context", RuleClass.PROVENANCE, "hrv_sdnn_context",
        Severity.INFO, "hrv_sdnn_algorithm_context", StatusEffect.NONE,
        InclusionEffect.KEEP, True, ("apple_hrv_sdnn", "hrv_task_force_1996"),
        "Retain SDNN and algorithm-version context.",
    ),
    RuleDefinition(
        "sleep_state_vocabulary", RuleClass.STRUCTURAL, "sleep_state_vocabulary",
        Severity.WARNING, "unknown_sleep_state", StatusEffect.REVIEW,
        InclusionEffect.KEEP, True, ("apple_sleep_analysis",),
        "Preserve unknown states but mark them for review.",
    ),
    RuleDefinition(
        "sleep_same_state_overlap", RuleClass.TEMPORAL, "sleep_same_state_overlap",
        Severity.INFO, "overlapping_same_sleep_state", StatusEffect.REVIEW,
        InclusionEffect.KEEP, False, ("apple_sleep_analysis", "project_sleep_intervals"),
        "Detect overlapping intervals with the same sleep state.",
    ),
    RuleDefinition(
        "sleep_detailed_state_overlap", RuleClass.TEMPORAL,
        "sleep_detailed_state_overlap", Severity.WARNING,
        "overlapping_detailed_sleep_states", StatusEffect.REVIEW,
        InclusionEffect.KEEP, False, ("apple_sleep_analysis", "project_sleep_intervals"),
        "Detailed awake/core/deep/REM samples are expected not to overlap.",
    ),
    RuleDefinition(
        "sleep_inbed_nesting", RuleClass.TEMPORAL, "sleep_inbed_nesting",
        Severity.INFO, "sleep_state_outside_inbed", StatusEffect.REVIEW,
        InclusionEffect.KEEP, False, ("apple_sleep_analysis", "project_sleep_intervals"),
        "Assess detailed states against in-bed intervals without requiring complete edge coverage.",
    ),
)

_add(
    RuleDefinition(
        "blood_pressure_pair_complete", RuleClass.STRUCTURAL,
        "blood_pressure_pair_complete", Severity.ERROR,
        "incomplete_blood_pressure_pair", StatusEffect.EXCLUDE_DEFAULT,
        InclusionEffect.EXCLUDE_DEFAULT, True, ("apple_blood_pressure",),
        "Systolic and diastolic values must remain coupled in one event.",
    ),
    RuleDefinition(
        "blood_pressure_pair_order", RuleClass.PHYSIOLOGICAL_SCREENING,
        "blood_pressure_pair_order", Severity.WARNING,
        "blood_pressure_pair_order_warning", StatusEffect.REVIEW,
        InclusionEffect.KEEP, True,
        ("apple_blood_pressure", "aha_blood_pressure_measurement"),
        "Flag systolic values that are not greater than diastolic values; do not swap them.",
    ),
    RuleDefinition(
        "activity_summary_date_ambiguity", RuleClass.TEMPORAL,
        "activity_summary_date_ambiguity", Severity.WARNING,
        "summary_date_assignment_ambiguous", StatusEffect.REVIEW,
        InclusionEffect.EXCLUDE_DEFAULT, False,
        ("apple_activity_summary", "project_activity_summary_sliding_pair"),
        "The exporter payload position cannot yet be assigned to a calendar date unambiguously.",
    ),
    RuleDefinition(
        "activity_summary_nonnegative", RuleClass.PHYSIOLOGICAL_SCREENING,
        "activity_summary_goal_nonnegative", Severity.WARNING,
        "negative_activity_summary_value", StatusEffect.REVIEW,
        InclusionEffect.KEEP, True, ("apple_activity_summary",),
        "Activity summary and goal fields should be nonnegative.",
    ),
)

_add(
    RuleDefinition(
        "ecg_waveform_shape", RuleClass.SIGNAL_INTEGRITY, "ecg_waveform_shape",
        Severity.ERROR, "invalid_ecg_waveform_shape", StatusEffect.EXCLUDE_DEFAULT,
        InclusionEffect.EXCLUDE_DEFAULT, True, ("apple_ecg",),
        "Waveform rows must retain paired relative-time and amplitude samples.",
    ),
    RuleDefinition(
        "ecg_sample_count_consistency", RuleClass.SIGNAL_INTEGRITY,
        "ecg_sample_count_consistency", Severity.WARNING,
        "ecg_sample_count_mismatch", StatusEffect.REVIEW,
        InclusionEffect.KEEP, True, ("apple_ecg", "apple_ecg_sampling_frequency"),
        "Observed waveform length should be consistent with duration and sampling frequency.",
    ),
    RuleDefinition(
        "ecg_relative_time_monotonic", RuleClass.SIGNAL_INTEGRITY,
        "ecg_relative_time_monotonic", Severity.ERROR,
        "ecg_relative_time_not_monotonic", StatusEffect.EXCLUDE_DEFAULT,
        InclusionEffect.EXCLUDE_DEFAULT, True, ("apple_ecg",),
        "Relative waveform time must increase monotonically.",
    ),
    RuleDefinition(
        "ecg_finite_amplitude", RuleClass.SIGNAL_INTEGRITY, "ecg_finite_amplitude",
        Severity.ERROR, "nonfinite_ecg_amplitude", StatusEffect.EXCLUDE_DEFAULT,
        InclusionEffect.EXCLUDE_DEFAULT, True, ("apple_ecg",),
        "ECG amplitude values must be finite.",
    ),
)

_add(
    RuleDefinition(
        "cgm_status_vocabulary", RuleClass.PROVENANCE, "cgm_status_vocabulary",
        Severity.INFO, "unknown_cgm_status", StatusEffect.REVIEW,
        InclusionEffect.KEEP, True, ("project_cgm_metadata", "ada_cgm_2026"),
        "Validate but preserve source status values.",
    ),
    RuleDefinition(
        "cgm_trend_vocabulary", RuleClass.PROVENANCE, "cgm_trend_vocabulary",
        Severity.INFO, "unknown_cgm_trend", StatusEffect.REVIEW,
        InclusionEffect.KEEP, True, ("project_cgm_metadata",),
        "Validate but preserve trend arrows and rates.",
    ),
    RuleDefinition(
        "cgm_time_zone_context", RuleClass.PROVENANCE, "cgm_time_zone_context",
        Severity.WARNING, "invalid_cgm_time_zone", StatusEffect.REVIEW,
        InclusionEffect.KEEP, True, ("project_cgm_metadata",),
        "Validate IANA time-zone metadata while preserving travel epochs.",
    ),
    RuleDefinition(
        "pulse_ox_source_limitations", RuleClass.PROVENANCE,
        "pulse_ox_source_limitations", Severity.INFO,
        "pulse_ox_source_context_limited", StatusEffect.REVIEW,
        InclusionEffect.KEEP, False, ("fda_pulse_oximetry",),
        "Pulse oximetry should retain device/source context because accuracy can vary.",
    ),
    RuleDefinition(
        "temperature_sensor_location", RuleClass.PROVENANCE,
        "temperature_sensor_location_context", Severity.INFO,
        "temperature_sensor_location_unknown", StatusEffect.REVIEW,
        InclusionEffect.KEEP, True, ("apple_body_temperature",),
        "Temperature interpretation depends on measurement location and method.",
    ),
    RuleDefinition(
        "peak_flow_session_context", RuleClass.PROVENANCE,
        "peak_flow_session_context", Severity.INFO,
        "peak_flow_session_context_unknown", StatusEffect.REVIEW,
        InclusionEffect.KEEP, False, ("apple_peak_flow", "ats_ers_spirometry_2019"),
        "Peak flow lacks maneuver/session quality context in this export.",
    ),
    RuleDefinition(
        "vo2max_test_type", RuleClass.PROVENANCE, "vo2max_test_type_context",
        Severity.INFO, "vo2max_test_type_unknown", StatusEffect.REVIEW,
        InclusionEffect.KEEP, True, ("apple_vo2max",),
        "VO2Max interpretation depends on whether it is an Apple estimate or another test type.",
    ),
    RuleDefinition(
        "bac_calculator_estimate", RuleClass.PROVENANCE, "bac_calculator_estimate",
        Severity.INFO, "calculator_estimate", StatusEffect.REVIEW,
        InclusionEffect.EXCLUDE_DEFAULT, True, ("project_bac_sources",),
        "Calculator-generated BAC trajectories are estimates rather than direct measurements.",
    ),
)

_add(
    RuleDefinition(
        "nutrition_preserve_distinct_uuid", RuleClass.SOURCE,
        "nutrition_preserve_distinct_uuid", Severity.INFO,
        "distinct_nutrition_entry", StatusEffect.NONE,
        InclusionEffect.KEEP, True, ("apple_nutrition",),
        "Identical values with different record IDs may be separate food entries.",
    ),
    RuleDefinition(
        "nutrition_nonnegative", RuleClass.PHYSIOLOGICAL_SCREENING,
        "nutrition_nonnegative", Severity.WARNING, "negative_nutrition_amount",
        StatusEffect.REVIEW, InclusionEffect.KEEP, True, ("apple_nutrition",),
        "Nutrient and consumed-energy amounts are ordinarily nonnegative.",
    ),
    RuleDefinition(
        "mindful_interval_valid", RuleClass.TEMPORAL, "mindful_interval_valid",
        Severity.ERROR, "invalid_mindful_interval", StatusEffect.EXCLUDE_DEFAULT,
        InclusionEffect.EXCLUDE_DEFAULT, True, ("apple_mindful_session",),
        "Mindful-session duration is represented by start and end times.",
    ),
    RuleDefinition(
        "bmi_screening_context", RuleClass.PHYSIOLOGICAL_SCREENING,
        "bmi_screening_context", Severity.INFO, "bmi_is_screening_measure",
        StatusEffect.NONE, InclusionEffect.KEEP, True, ("who_bmi",),
        "BMI is a screening measure and must not be treated as a diagnosis.",
    ),
    RuleDefinition(
        "waist_protocol_context", RuleClass.PROVENANCE, "waist_protocol_context",
        Severity.INFO, "waist_measurement_protocol_unknown", StatusEffect.REVIEW,
        InclusionEffect.KEEP, True, ("who_waist",),
        "Waist circumference depends on anatomical landmark and tape protocol.",
    ),
    RuleDefinition(
        "unknown_feature_policy", RuleClass.STRUCTURAL, "unknown_feature_policy",
        Severity.WARNING, "unknown_feature_policy", StatusEffect.REVIEW,
        InclusionEffect.EXCLUDE_DEFAULT, False,
        description="Unknown features are copied unchanged without canonical conversion.",
    ),
)


def get_rule(rule_id: str) -> RuleDefinition:
    return RULES[rule_id]


def known_rule_ids() -> tuple[str, ...]:
    return tuple(sorted(RULES))
