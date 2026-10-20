"""
Wearable Data Processing and Modeling project
Stable rule catalog for Google Fitness curation. CWP1 declares rule identity, evidence, flags and
status/inclusion effects. The future curation engine must execute these definitions without changing native
values or silently deleting rows.
"""


from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping
from .evidence import EVIDENCE
from .models import InclusionEffect, RuleClass, Severity, StatusEffect, to_primitive


EXPECTED_RULES_FINGERPRINT = "9f819581273d7cb066731440cda6c6467980452b73f0710385fb5a6a3ec0ac03"


@dataclass(frozen=True, slots=True)
class RuleDefinition:
    rule_id: str
    rule_class: RuleClass
    severity: Severity
    flag: str
    status_effect: StatusEffect
    inclusion_effect: InclusionEffect
    row_local: bool
    evidence_refs: tuple[str, ...]
    description: str
    parameters: tuple[tuple[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return to_primitive(self)


def _r(
    rule_id: str, rule_class: RuleClass, severity: Severity, flag: str, status_effect: StatusEffect,
    inclusion_effect: InclusionEffect, row_local: bool, evidence_refs: tuple[str, ...], description: str,
    **parameters: object,
) -> RuleDefinition:
    return RuleDefinition(
        rule_id=rule_id,
        rule_class=rule_class,
        severity=severity,
        flag=flag,
        status_effect=status_effect,
        inclusion_effect=inclusion_effect,
        row_local=row_local,
        evidence_refs=evidence_refs,
        description=description,
        parameters=tuple(sorted(parameters.items())),
    )


_RULES = (
    _r(
        "native_interval_valid",
        RuleClass.TEMPORAL,
        Severity.ERROR,
        "invalid_native_interval",
        StatusEffect.EXCLUDE_DEFAULT,
        InclusionEffect.EXCLUDE_DEFAULT,
        True,
        ("project_native_registry",),
        "Start and end timestamps must remain parseable and end must not precede start.",
    ),
    _r(
        "finite_measurement",
        RuleClass.STRUCTURAL,
        Severity.ERROR,
        "nonfinite_measurement",
        StatusEffect.EXCLUDE_DEFAULT,
        InclusionEffect.EXCLUDE_DEFAULT,
        True,
        ("project_native_registry",),
        "Present numeric measurements must be finite; raw values are never clipped or replaced.",
    ),
    _r(
        "source_day_label_context",
        RuleClass.TEMPORAL,
        Severity.INFO,
        "source_day_label_context",
        StatusEffect.NONE,
        InclusionEffect.KEEP,
        True,
        ("project_phase0_day_labels",),
        "Localized source day labels are context only and never define a canonical event date.",
    ),
    _r(
        "source_day_label_unrecognized",
        RuleClass.TEMPORAL,
        Severity.INFO,
        "unrecognized_source_day_label",
        StatusEffect.NONE,
        InclusionEffect.KEEP,
        True,
        ("project_phase0_day_labels",),
        "Preserve unrecognized localized day labels without inventing a calendar date.",
    ),
    _r(
        "source_day_label_nonadjacent_disagreement",
        RuleClass.TEMPORAL,
        Severity.WARNING,
        "source_day_label_nonadjacent_disagreement",
        StatusEffect.REVIEW,
        InclusionEffect.KEEP,
        True,
        ("project_phase0_day_labels",),
        "A nonadjacent weekday discrepancy requires review; one-day offsets remain contextual.",
    ),
    _r(
        "unknown_feature_policy",
        RuleClass.STRUCTURAL,
        Severity.WARNING,
        "unknown_feature_policy",
        StatusEffect.REVIEW,
        InclusionEffect.EXCLUDE_DEFAULT,
        False,
        ("project_curation_invariant",),
        "Unknown future native features are copied unchanged and excluded from the default analytical subset.",
    ),
    _r(
        "activity_interval_valid",
        RuleClass.TEMPORAL,
        Severity.ERROR,
        "invalid_activity_interval",
        StatusEffect.EXCLUDE_DEFAULT,
        InclusionEffect.EXCLUDE_DEFAULT,
        True,
        ("google_fit_activity_segments", "project_native_registry"),
        "Activity intervals must preserve strictly positive source boundaries without reconstruction.",
    ),
    _r(
        "activity_measurements_nonnegative",
        RuleClass.QUALITY,
        Severity.WARNING,
        "negative_activity_submeasurement",
        StatusEffect.REVIEW,
        InclusionEffect.KEEP,
        True,
        ("project_phase0_activity_submeasurements",),
        "Present activity calories, distance and quantity are expected to be nonnegative; retain values and flag anomalies.",
    ),
    _r(
        "activity_quantity_unresolved",
        RuleClass.UNIT,
        Severity.INFO,
        "activity_quantity_semantics_unresolved",
        StatusEffect.NONE,
        InclusionEffect.KEEP,
        True,
        ("project_phase0_activity_submeasurements",),
        "Activity quantity lacks a defensible common unit and is excluded from canonical projection without invalidating the interval.",
    ),
    _r(
        "activity_tracking_status",
        RuleClass.PROVENANCE,
        Severity.INFO,
        "activity_tracking_status",
        StatusEffect.NONE,
        InclusionEffect.KEEP,
        True,
        ("project_phase0_provenance",),
        "Map the raw tracked field to tracked, untracked or unknown without inferring validity or entry method.",
    ),
    _r(
        "activity_provenance_classification",
        RuleClass.PROVENANCE,
        Severity.INFO,
        "activity_provenance_classified",
        StatusEffect.NONE,
        InclusionEffect.KEEP,
        True,
        ("project_phase0_provenance",),
        "Classify acquisition and derivation dimensions conservatively while retaining raw identifiers.",
    ),
    _r(
        "activity_provenance_unknown",
        RuleClass.PROVENANCE,
        Severity.INFO,
        "activity_provenance_unknown",
        StatusEffect.NONE,
        InclusionEffect.KEEP,
        True,
        ("project_phase0_provenance",),
        "Insufficient application or stream evidence remains explicitly unclassified.",
    ),
    _r(
        "unknown_activity_policy",
        RuleClass.STRUCTURAL,
        Severity.WARNING,
        "unknown_activity_policy",
        StatusEffect.REVIEW,
        InclusionEffect.EXCLUDE_DEFAULT,
        False,
        ("project_native_registry",),
        "An unregistered activity fallback remains visible but is excluded from default analysis until reviewed.",
    ),
    _r(
        "sleep_state_vocabulary",
        RuleClass.STRUCTURAL,
        Severity.ERROR,
        "unknown_sleep_state",
        StatusEffect.EXCLUDE_DEFAULT,
        InclusionEffect.EXCLUDE_DEFAULT,
        True,
        ("google_fit_sleep_states", "project_phase0_sleep_inventory"),
        "Native Sleep values must remain one of SLEEP, AWAKE, DEEP, LIGHT or REM.",
    ),
    _r(
        "sleep_same_state_overlap",
        RuleClass.TEMPORAL,
        Severity.INFO,
        "overlapping_same_sleep_state",
        StatusEffect.REVIEW,
        InclusionEffect.KEEP,
        False,
        ("project_phase0_sleep_inventory",),
        "Retain same-state overlaps within the same provenance and flag them for review.",
    ),
    _r(
        "sleep_detailed_state_overlap",
        RuleClass.TEMPORAL,
        Severity.WARNING,
        "overlapping_detailed_sleep_states",
        StatusEffect.REVIEW,
        InclusionEffect.KEEP,
        False,
        ("google_fit_sleep_states",),
        "Retain incompatible detailed sleep-state overlaps within the same provenance.",
    ),
    _r(
        "sleep_awake_overlap",
        RuleClass.TEMPORAL,
        Severity.WARNING,
        "awake_overlaps_sleep_state",
        StatusEffect.REVIEW,
        InclusionEffect.KEEP,
        False,
        ("google_fit_sleep_states",),
        "Retain AWAKE intervals that overlap sleep intervals and mark the same-provenance disagreement.",
    ),
    _r(
        "sleep_generic_detail_overlap_context",
        RuleClass.TEMPORAL,
        Severity.INFO,
        "generic_sleep_overlaps_detailed_state",
        StatusEffect.NONE,
        InclusionEffect.KEEP,
        False,
        ("google_fit_sleep_states",),
        "Generic SLEEP overlapping a detailed state is contextual rather than automatically contradictory.",
    ),
    _r(
        "sleep_no_apple_equivalence",
        RuleClass.SOURCE,
        Severity.INFO,
        "google_sleep_state_not_harmonized",
        StatusEffect.NONE,
        InclusionEffect.KEEP,
        True,
        ("google_fit_sleep_states", "project_phase0_sleep_inventory"),
        "Google sleep states remain source-specific and are not remapped to Apple terminology in curation.",
    ),
    _r(
        "calories_negative_tiny",
        RuleClass.QUALITY,
        Severity.WARNING,
        "negative_calories_tiny",
        StatusEffect.REVIEW,
        InclusionEffect.EXCLUDE_DEFAULT,
        True,
        ("project_phase0_negative_calories",),
        "Retain tiny negative Calories values, withhold canonical value and exclude from the default subset.",
        absolute_max=0.001,
    ),
    _r(
        "calories_negative_small",
        RuleClass.QUALITY,
        Severity.WARNING,
        "negative_calories_small",
        StatusEffect.REVIEW,
        InclusionEffect.EXCLUDE_DEFAULT,
        True,
        ("project_phase0_negative_calories",),
        "Retain small negative Calories values, withhold canonical value and exclude from the default subset.",
        absolute_min_exclusive=0.001,
        absolute_max=1.0,
    ),
    _r(
        "calories_negative_material",
        RuleClass.QUALITY,
        Severity.ERROR,
        "negative_calories_material",
        StatusEffect.EXCLUDE_DEFAULT,
        InclusionEffect.EXCLUDE_DEFAULT,
        True,
        ("project_phase0_negative_calories",),
        "Retain material negative Calories values but exclude them from default analysis.",
        absolute_min_exclusive=1.0,
        absolute_max=100.0,
    ),
    _r(
        "calories_negative_extreme",
        RuleClass.QUALITY,
        Severity.ERROR,
        "negative_calories_extreme",
        StatusEffect.EXCLUDE_DEFAULT,
        InclusionEffect.EXCLUDE_DEFAULT,
        True,
        ("project_phase0_negative_calories",),
        "Retain extreme negative Calories values but exclude them from default analysis.",
        absolute_min_exclusive=100.0,
    ),
    _r(
        "distance_nonnegative",
        RuleClass.QUALITY,
        Severity.WARNING,
        "negative_distance",
        StatusEffect.REVIEW,
        InclusionEffect.EXCLUDE_DEFAULT,
        True,
        ("project_phase0_census",),
        "Negative Distance values are retained but excluded from the default subset.",
    ),
    _r(
        "distance_positive_zero_duration",
        RuleClass.TEMPORAL,
        Severity.WARNING,
        "positive_distance_zero_duration",
        StatusEffect.REVIEW,
        InclusionEffect.KEEP,
        True,
        ("google_fit_data_model",),
        "Positive interval distance with zero duration requires review; no interval is invented.",
    ),
    _r(
        "distance_source_rate_limit",
        RuleClass.QUALITY,
        Severity.WARNING,
        "distance_rate_above_source_limit",
        StatusEffect.REVIEW,
        InclusionEffect.KEEP,
        True,
        ("google_fit_distance_delta",),
        "Flag source-implied distance rates above the documented Google source contract without clipping the value.",
        meters_per_second_max=100.0,
    ),
    _r(
        "weight_source_range",
        RuleClass.QUALITY,
        Severity.WARNING,
        "weight_outside_source_range",
        StatusEffect.REVIEW,
        InclusionEffect.EXCLUDE_DEFAULT,
        True,
        ("google_fit_weight",),
        "Flag values outside the official Google source range while preserving the raw Weight value.",
        minimum=0.0,
        maximum=1000.0,
    ),
    _r(
        "height_source_range",
        RuleClass.QUALITY,
        Severity.WARNING,
        "height_outside_source_range",
        StatusEffect.REVIEW,
        InclusionEffect.EXCLUDE_DEFAULT,
        True,
        ("google_fit_height",),
        "Flag values outside the official Google source range while preserving the raw Height value.",
        minimum=0.0,
        maximum=3.0,
    ),
    _r(
        "weight_nonpoint_interval",
        RuleClass.TEMPORAL,
        Severity.WARNING,
        "weight_nonpoint_interval",
        StatusEffect.REVIEW,
        InclusionEffect.KEEP,
        True,
        ("google_fit_weight", "project_phase0_weight_intervals"),
        "Preserve Weight observations with positive duration and mark their mismatch with the instantaneous source contract.",
        duration_seconds_min_exclusive=0,
        duration_seconds_max=86400,
    ),
    _r(
        "weight_long_interval",
        RuleClass.TEMPORAL,
        Severity.WARNING,
        "weight_long_interval",
        StatusEffect.REVIEW,
        InclusionEffect.EXCLUDE_DEFAULT,
        True,
        ("project_phase0_weight_intervals",),
        "Preserve unusually long Weight intervals but exclude them from the default subset.",
        duration_seconds_min_exclusive=86400,
    ),
    _r(
        "anthropometric_added_by_context",
        RuleClass.PROVENANCE,
        Severity.INFO,
        "anthropometric_application_context",
        StatusEffect.NONE,
        InclusionEffect.KEEP,
        True,
        ("project_phase0_provenance",),
        "The added_by application is retained as provenance and does not by itself prove user entry or device measurement.",
    ),
    _r(
        "step_count_nonnegative",
        RuleClass.QUALITY,
        Severity.ERROR,
        "negative_step_count",
        StatusEffect.EXCLUDE_DEFAULT,
        InclusionEffect.EXCLUDE_DEFAULT,
        True,
        ("google_fit_step_count_delta",),
        "Negative step counts are retained but excluded from default analysis.",
    ),
    _r(
        "step_count_source_rate_limit",
        RuleClass.QUALITY,
        Severity.WARNING,
        "step_count_rate_above_source_limit",
        StatusEffect.REVIEW,
        InclusionEffect.EXCLUDE_DEFAULT,
        True,
        ("google_fit_step_count_delta",),
        "Flag rates above 10 steps per second for version-2 rows with a valid raw interval.",
        steps_per_second_max=10.0,
    ),
    _r(
        "step_count_v1_legacy_context",
        RuleClass.SOURCE,
        Severity.INFO,
        "step_count_legacy_v1",
        StatusEffect.NONE,
        InclusionEffect.KEEP,
        True,
        ("project_phase0_stepcount_alternatives",),
        "Version 1 remains a separate legacy scalar representation anchored by outer datetime.",
    ),
    _r(
        "step_count_v2_group_structure",
        RuleClass.STRUCTURAL,
        Severity.ERROR,
        "invalid_step_count_group_structure",
        StatusEffect.REVIEW,
        InclusionEffect.EXCLUDE_DEFAULT,
        False,
        ("project_phase0_stepcount_alternatives",),
        "Version-2 preference is evaluated within collecting-method version and outer ID without deleting any alternative.",
    ),
    _r(
        "step_count_estimated_preferred",
        RuleClass.GROUP_SELECTION,
        Severity.INFO,
        "step_count_fit_app_preferred",
        StatusEffect.NONE,
        InclusionEffect.INCLUDE_DEFAULT,
        False,
        ("google_fit_daily_step_total", "project_phase0_stepcount_alternatives"),
        "Exactly one estimated_steps row is the default analytical representative for its version-2 outer group.",
    ),
    _r(
        "step_count_merge_fallback",
        RuleClass.GROUP_SELECTION,
        Severity.WARNING,
        "step_count_merge_fallback",
        StatusEffect.REVIEW,
        InclusionEffect.INCLUDE_DEFAULT,
        False,
        ("project_phase0_stepcount_alternatives",),
        "Use exactly one merge_step_deltas row as fallback only when estimated_steps is absent.",
    ),
    _r(
        "step_count_sole_fallback",
        RuleClass.GROUP_SELECTION,
        Severity.WARNING,
        "step_count_sole_fallback",
        StatusEffect.REVIEW,
        InclusionEffect.INCLUDE_DEFAULT,
        False,
        ("project_phase0_stepcount_alternatives",),
        "A sole version-2 alternative is retained as a reviewable fallback when no preferred source exists.",
    ),
    _r(
        "step_count_parallel_nonpreferred",
        RuleClass.GROUP_SELECTION,
        Severity.INFO,
        "step_count_parallel_nonpreferred",
        StatusEffect.NONE,
        InclusionEffect.EXCLUDE_DEFAULT,
        False,
        ("google_fit_daily_step_total", "project_phase0_stepcount_alternatives"),
        "Valid alternatives parallel to the selected default remain source evidence but are excluded from the default subset.",
    ),
    _r(
        "step_count_ambiguous_parallel",
        RuleClass.GROUP_SELECTION,
        Severity.WARNING,
        "step_count_ambiguous_parallel",
        StatusEffect.REVIEW,
        InclusionEffect.EXCLUDE_DEFAULT,
        False,
        ("project_phase0_stepcount_alternatives",),
        "When no unique preferred or fallback row exists, retain all alternatives and select none by default.",
    ),
    _r(
        "step_count_alternatives_disagree",
        RuleClass.GROUP_SELECTION,
        Severity.WARNING,
        "step_count_alternatives_disagree",
        StatusEffect.REVIEW,
        InclusionEffect.GROUP_DECISION,
        False,
        ("project_phase0_stepcount_alternatives",),
        "Flag disagreement among alternatives in the same version-2 outer group without arithmetic resolution.",
    ),
    _r(
        "step_count_temporal_context",
        RuleClass.TEMPORAL,
        Severity.INFO,
        "step_count_multiple_time_contexts",
        StatusEffect.NONE,
        InclusionEffect.KEEP,
        True,
        ("google_fit_daily_step_total", "project_phase0_stepcount_temporal"),
        "Preserve outer datetime, summary date and raw interval separately and do not manufacture a canonical day.",
    ),
    _r(
        "step_count_cross_version_separate",
        RuleClass.SOURCE,
        Severity.INFO,
        "step_count_cross_version_separate",
        StatusEffect.NONE,
        InclusionEffect.KEEP,
        False,
        ("project_phase0_stepcount_alternatives", "project_phase0_stepcount_temporal"),
        "Version 1 and version 2 remain separate analytical series and are not deduplicated by date.",
    ),
    _r(
        "unit_mapping_pending",
        RuleClass.UNIT,
        Severity.INFO,
        "canonical_unit_withheld_pending_exporter_mapping",
        StatusEffect.NONE,
        InclusionEffect.KEEP,
        True,
        ("google_fit_data_model",),
        "Official Google units are not executed until the exporter-to-data-type mapping is frozen.",
    ),
    _r(
        "unit_payload_explicit",
        RuleClass.UNIT,
        Severity.INFO,
        "canonical_unit_resolved_payload_explicit",
        StatusEffect.NONE,
        InclusionEffect.KEEP,
        True,
        ("google_fit_step_count_delta",),
        "A payload-explicit Google data type permits reviewed identity canonicalization.",
    ),
)

RULES: Mapping[str, RuleDefinition] = MappingProxyType(
    {rule.rule_id: rule for rule in sorted(_RULES, key=lambda rule: rule.rule_id)}
)


def known_rules() -> tuple[str, ...]:
    return tuple(RULES)


def get_rule(rule_id: str) -> RuleDefinition:
    try:
        return RULES[rule_id]
    except KeyError as exc:
        raise KeyError(f"Unknown Google Fitness curation rule: {rule_id!r}") from exc


def validate_rules() -> tuple[str, ...]:
    issues: list[str] = []
    if len(RULES) != len(_RULES):
        issues.append("duplicate rule IDs")
    flags: set[str] = set()
    if rules_fingerprint() != EXPECTED_RULES_FINGERPRINT:
        issues.append(
            "rules fingerprint mismatch: "
            f"expected {EXPECTED_RULES_FINGERPRINT}, observed {rules_fingerprint()}"
        )
    for rule_id, rule in RULES.items():
        if rule_id != rule.rule_id:
            issues.append(f"rule key mismatch for {rule_id}")
        if not rule.evidence_refs:
            issues.append(f"rule lacks evidence references: {rule_id}")
        missing = sorted(set(rule.evidence_refs) - set(EVIDENCE))
        if missing:
            issues.append(f"rule {rule_id} references unknown evidence: {missing}")
        if rule.flag in flags and rule.flag not in {"activity_provenance_unknown",}:
            issues.append(f"duplicate rule flag: {rule.flag}")
        flags.add(rule.flag)
    return tuple(sorted(set(issues)))


def rules_payload() -> dict[str, object]:
    return {
        "rules": [RULES[name].as_dict() for name in RULES],
        "count": len(RULES),
        "fingerprint": rules_fingerprint(),
        "validation_issues": list(validate_rules()),
    }


def rules_fingerprint() -> str:
    rendered = json.dumps(
        [RULES[name].as_dict() for name in RULES],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(rendered).hexdigest()


_RULE_ISSUES = validate_rules()
if _RULE_ISSUES:
    raise RuntimeError("Invalid Google Fitness curation rules: " + "; ".join(_RULE_ISSUES))
