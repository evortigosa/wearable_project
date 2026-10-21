"""
Wearable Data Processing and Modeling project

Executable CWP3 row and group strategies for Google Fitness curation.

The strategies annotate native rows; they never delete, reorder, clip, repair,
or resample an observation.  Rules are executed only when registered for the
feature policy.  Policy-wide context is represented through derived columns
rather than repeated flags where a flag would duplicate the same fact.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Mapping

from .context import SleepRowContext, StepCountRowContext, finite_float, parse_datetime
from .models import (
    CurationStatus,
    FeaturePolicy,
    InclusionEffect,
    InclusionPolicy,
    StatusEffect,
)
from .provenance import classify_provenance
from .rules import RULES
from .unit_resolution import resolve_units


_STATUS_RANK = {
    CurationStatus.PASS: 0,
    CurationStatus.REVIEW: 1,
    CurationStatus.EXCLUDE_DEFAULT: 2,
}

# The audit and engine intentionally share the observed localized vocabulary.
_WEEKDAY_LABELS: dict[str, int] = {
    "mon": 0, "mon.": 0, "monday": 0,
    "tue": 1, "tue.": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "wed.": 2, "wednesday": 2,
    "thu": 3, "thu.": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "fri.": 4, "friday": 4,
    "sat": 5, "sat.": 5, "saturday": 5,
    "sun": 6, "sun.": 6, "sunday": 6,
    "יום ב׳": 0, "יום ג׳": 1, "יום ד׳": 2, "יום ה׳": 3,
    "יום ו׳": 4, "שבת": 5, "יום א׳": 6,
    "пн": 0, "вт": 1, "ср": 2, "чт": 3, "пт": 4, "сб": 5, "вс": 6,
    "seg": 0, "seg.": 0, "ter": 1, "ter.": 1, "qua": 2, "qua.": 2,
    "qui": 3, "qui.": 3, "sex": 4, "sex.": 4, "sáb": 5, "sáb.": 5,
    "dom": 6, "dom.": 6,
    "lun": 0, "lun.": 0, "mar": 1, "mar.": 1, "mer": 2, "mer.": 2,
    "jeu": 3, "jeu.": 3, "ven": 4, "ven.": 4, "sam": 5, "sam.": 5,
    "dim": 6, "dim.": 6,
    "mié": 2, "mié.": 2, "jue": 3, "jue.": 3, "vie": 4, "vie.": 4,
    "mo.": 0, "di.": 1, "mi.": 2, "do.": 3, "fr.": 4, "sa.": 5, "so.": 6,
    "mån": 0, "tis": 1, "ons": 2, "tors": 3, "fre": 4, "lör": 5, "sön": 6,
    "нд": 6,
}


# Explicit execution channel for every registered rule.  This is a safety gate:
# adding a policy rule without an engine implementation fails tests and import
# validation rather than being silently ignored.
ENGINE_RULE_IMPLEMENTATION: dict[str, str] = {
    "native_interval_valid": "row_predicate",
    "finite_measurement": "row_predicate",
    "source_day_label_context": "policy_context",
    "source_day_label_unrecognized": "row_predicate",
    "source_day_label_nonadjacent_disagreement": "row_predicate",
    "unknown_feature_policy": "fallback",
    "activity_interval_valid": "row_predicate",
    "activity_measurements_nonnegative": "row_predicate",
    "activity_quantity_unresolved": "row_predicate",
    "activity_tracking_status": "derived_dimension",
    "activity_provenance_classification": "derived_dimension",
    "activity_provenance_unknown": "row_predicate",
    "unknown_activity_policy": "fallback",
    "sleep_state_vocabulary": "row_predicate",
    "sleep_same_state_overlap": "group_context",
    "sleep_detailed_state_overlap": "group_context",
    "sleep_awake_overlap": "group_context",
    "sleep_generic_detail_overlap_context": "group_context",
    "sleep_no_apple_equivalence": "policy_context",
    "calories_negative_tiny": "row_predicate",
    "calories_negative_small": "row_predicate",
    "calories_negative_material": "row_predicate",
    "calories_negative_extreme": "row_predicate",
    "distance_nonnegative": "row_predicate",
    "distance_positive_zero_duration": "row_predicate",
    "distance_source_rate_limit": "row_predicate",
    "weight_source_range": "row_predicate",
    "height_source_range": "row_predicate",
    "weight_nonpoint_interval": "row_predicate",
    "weight_long_interval": "row_predicate",
    "anthropometric_added_by_context": "derived_dimension",
    "step_count_nonnegative": "row_predicate",
    "step_count_source_rate_limit": "row_predicate",
    "step_count_v1_legacy_context": "group_context",
    "step_count_v2_group_structure": "group_context",
    "step_count_estimated_preferred": "group_context",
    "step_count_merge_fallback": "group_context",
    "step_count_sole_fallback": "group_context",
    "step_count_parallel_nonpreferred": "group_context",
    "step_count_ambiguous_parallel": "group_context",
    "step_count_alternatives_disagree": "group_context",
    "step_count_temporal_context": "policy_context",
    "step_count_cross_version_separate": "policy_context",
    "unit_mapping_pending": "unit_annotation",
    "unit_payload_explicit": "unit_annotation",
}




def _rule_parameters(rule_id: str) -> dict[str, object]:
    """Return the immutable registry parameters as a normal lookup mapping."""

    return dict(RULES[rule_id].parameters)


def validate_engine_rule_coverage() -> tuple[str, ...]:
    referenced = set().union(*(set(policy.curation.rule_ids) for policy in _all_policies()))
    missing = referenced - set(ENGINE_RULE_IMPLEMENTATION)
    unknown = set(ENGINE_RULE_IMPLEMENTATION) - set(RULES)
    return tuple(sorted({*(f"missing:{item}" for item in missing), *(f"unknown:{item}" for item in unknown)}))


def _all_policies():
    # Local import avoids a registry/strategies import cycle during module load.
    from .registry import CURATION_POLICIES, UNKNOWN_POLICY

    return (*CURATION_POLICIES.values(), UNKNOWN_POLICY)


@dataclass(slots=True)
class RowAnnotation:
    status: CurationStatus
    include_by_default: bool
    values: dict[str, str] = field(default_factory=dict)
    flags: set[str] = field(default_factory=set)

    def escalate(self, status: CurationStatus) -> None:
        if _STATUS_RANK[status] > _STATUS_RANK[self.status]:
            self.status = status

    def apply_rule(self, rule_id: str, active_rule_ids: frozenset[str], *, emit_flag: bool = True) -> None:
        if rule_id not in active_rule_ids:
            return
        rule = RULES[rule_id]
        if emit_flag:
            self.flags.add(rule.flag)
        if rule.status_effect is StatusEffect.EXCLUDE_DEFAULT:
            self.escalate(CurationStatus.EXCLUDE_DEFAULT)
        elif rule.status_effect is StatusEffect.REVIEW:
            self.escalate(CurationStatus.REVIEW)
        if rule.inclusion_effect is InclusionEffect.EXCLUDE_DEFAULT:
            self.include_by_default = False
        elif rule.inclusion_effect is InclusionEffect.INCLUDE_DEFAULT:
            self.include_by_default = True

    def apply_flag(self, flag: str, active_rule_ids: frozenset[str]) -> None:
        for rule_id in active_rule_ids:
            if RULES[rule_id].flag == flag:
                self.apply_rule(rule_id, active_rule_ids)
                return
        # Context builders only emit registered flags.  Failing loudly protects
        # against a future context/registry drift.
        raise KeyError(f"No active policy rule emits flag {flag!r}")

    def as_sparse_dict(self, policy: FeaturePolicy) -> dict[str, str]:
        result = dict(self.values)
        if self.status is not CurationStatus.PASS:
            result[policy.curation.status_column] = self.status.value
        if not self.include_by_default:
            result[policy.curation.inclusion_column] = "0"
        if self.flags:
            result[policy.curation.flags_column] = ";".join(sorted(self.flags))
        return result


def _default_include(policy: FeaturePolicy) -> bool:
    return policy.curation.default_inclusion is not InclusionPolicy.EXCLUDE


def _finite(value: Any) -> bool:
    if value in (None, ""):
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _duration_seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def _weekday_relation(label: str, start: datetime | None, end: datetime | None) -> str:
    expected = _WEEKDAY_LABELS.get(label.strip().casefold())
    if expected is None:
        return "unrecognized"
    weekdays = [value.weekday() for value in (start, end) if value is not None]
    if not weekdays:
        return "recognized_no_timestamp"
    if expected in weekdays:
        return "matches"
    adjacent = {(value - 1) % 7 for value in weekdays} | {(value + 1) % 7 for value in weekdays}
    return "adjacent" if expected in adjacent else "nonadjacent"


def _source_day_rules(annotation: RowAnnotation, row: Mapping[str, str], active: frozenset[str], start: datetime | None, end: datetime | None) -> None:
    label = (row.get("source_day_label") or "").strip()
    if not label:
        return
    relation = _weekday_relation(label, start, end)
    if relation == "unrecognized":
        annotation.apply_rule("source_day_label_unrecognized", active)
    elif relation == "nonadjacent":
        annotation.apply_rule("source_day_label_nonadjacent_disagreement", active)


def _measurement_rules(annotation: RowAnnotation, policy: FeaturePolicy, row: Mapping[str, str], active: frozenset[str]) -> None:
    for measurement in policy.measurements:
        value = row.get(measurement.column)
        if value not in (None, "") and not _finite(value):
            annotation.apply_rule("finite_measurement", active)
            return


def _activity_rules(annotation: RowAnnotation, policy: FeaturePolicy, row: Mapping[str, str], active: frozenset[str], start: datetime | None, end: datetime | None) -> None:
    if start is None or end is None or end <= start:
        annotation.apply_rule("activity_interval_valid", active)
    for column in policy.measurement_columns:
        value = finite_float(row.get(column))
        if value is not None and value < 0:
            annotation.apply_rule("activity_measurements_nonnegative", active)
            break
    if "quantity" in policy.measurement_columns and (row.get("quantity") or "").strip():
        annotation.apply_rule("activity_quantity_unresolved", active)
    if annotation.values.get("acquisition_method") == "unknown":
        annotation.apply_rule("activity_provenance_unknown", active)


def _calories_rules(annotation: RowAnnotation, row: Mapping[str, str], active: frozenset[str]) -> None:
    value = finite_float(row.get("value"))
    if value is None or value >= 0:
        return
    magnitude = abs(value)
    tiny_max = float(_rule_parameters("calories_negative_tiny")["absolute_max"])
    small_max = float(_rule_parameters("calories_negative_small")["absolute_max"])
    material_max = float(_rule_parameters("calories_negative_material")["absolute_max"])
    if magnitude <= tiny_max:
        annotation.apply_rule("calories_negative_tiny", active)
    elif magnitude <= small_max:
        annotation.apply_rule("calories_negative_small", active)
    elif magnitude <= material_max:
        annotation.apply_rule("calories_negative_material", active)
    else:
        annotation.apply_rule("calories_negative_extreme", active)


def _distance_rules(annotation: RowAnnotation, row: Mapping[str, str], active: frozenset[str], start: datetime | None, end: datetime | None) -> None:
    value = finite_float(row.get("value"))
    if value is None:
        return
    duration = _duration_seconds(start, end)
    if value < 0:
        annotation.apply_rule("distance_nonnegative", active)
    if value > 0 and duration == 0:
        annotation.apply_rule("distance_positive_zero_duration", active)
    rate_max = float(_rule_parameters("distance_source_rate_limit")["meters_per_second_max"])
    if value >= 0 and duration is not None and duration > 0 and value / duration > rate_max:
        annotation.apply_rule("distance_source_rate_limit", active)


def _anthropometric_rules(annotation: RowAnnotation, feature: str, row: Mapping[str, str], active: frozenset[str], start: datetime | None, end: datetime | None) -> None:
    value = finite_float(row.get("value"))
    if feature == "Weight" and value is not None:
        params = _rule_parameters("weight_source_range")
        if not float(params["minimum"]) <= value <= float(params["maximum"]):
            annotation.apply_rule("weight_source_range", active)
    if feature == "Height" and value is not None:
        params = _rule_parameters("height_source_range")
        if not float(params["minimum"]) <= value <= float(params["maximum"]):
            annotation.apply_rule("height_source_range", active)
    if feature == "Weight":
        duration = _duration_seconds(start, end)
        nonpoint_min = float(_rule_parameters("weight_nonpoint_interval")["duration_seconds_min_exclusive"])
        long_min = float(_rule_parameters("weight_long_interval")["duration_seconds_min_exclusive"])
        if duration is not None and duration > nonpoint_min:
            annotation.apply_rule("weight_nonpoint_interval", active)
        if duration is not None and duration > long_min:
            annotation.apply_rule("weight_long_interval", active)


def _step_count_rules(annotation: RowAnnotation, row: Mapping[str, str], active: frozenset[str], context: StepCountRowContext | None, start: datetime | None, end: datetime | None) -> None:
    if context is None:
        annotation.escalate(CurationStatus.REVIEW)
        annotation.include_by_default = False
        annotation.apply_rule("step_count_v2_group_structure", active)
    else:
        annotation.values["step_count_role"] = context.role.value
        annotation.escalate(context.status)
        annotation.include_by_default = context.include_by_default
        for flag in context.flags:
            annotation.apply_flag(flag, active)

    value = finite_float(row.get("value"))
    if value is not None and value < 0:
        annotation.apply_rule("step_count_nonnegative", active)
    version = (row.get("collecting_method_version") or "").strip()
    duration = _duration_seconds(start, end)
    if version == "2.0" and value is not None and value >= 0 and duration is not None and duration > 0:
        rate_max = float(_rule_parameters("step_count_source_rate_limit")["steps_per_second_max"])
        if value / duration > rate_max:
            annotation.apply_rule("step_count_source_rate_limit", active)


def annotate_row(
    feature: str,
    row: Mapping[str, str],
    policy: FeaturePolicy,
    *,
    step_context: StepCountRowContext | None = None,
    sleep_context: SleepRowContext | None = None,
) -> RowAnnotation:
    """Create one non-destructive row annotation from the reviewed CWP3 policy."""

    annotation = RowAnnotation(
        status=policy.curation.default_status,
        include_by_default=_default_include(policy),
    )
    active = frozenset(policy.curation.rule_ids)

    provenance = classify_provenance(feature, policy.event_kind, row)
    annotation.values.update(provenance.as_values(policy.provenance.output_dimensions))

    unit = resolve_units(feature, row, policy)
    if unit is not None:
        annotation.values.update(unit.as_values())
        if feature in {"Calories", "Distance", "Weight", "Height"}:
            annotation.apply_rule("unit_mapping_pending", active)

    _measurement_rules(annotation, policy, row, active)

    start = parse_datetime(row.get(policy.temporal.start_column)) if policy.temporal.start_column else None
    end = parse_datetime(row.get(policy.temporal.end_column)) if policy.temporal.end_column else None
    if "native_interval_valid" in active:
        if start is None or end is None or end < start:
            annotation.apply_rule("native_interval_valid", active)

    _source_day_rules(annotation, row, active, start, end)

    if policy.event_kind.value in {"activity_interval", "state_interval"}:
        _activity_rules(annotation, policy, row, active, start, end)

    if feature == "Sleep":
        state = (row.get("value") or "").strip()
        if state not in set(policy.native_schema.state_vocabulary):
            annotation.apply_rule("sleep_state_vocabulary", active)
        if sleep_context is not None:
            for flag in sleep_context.flags:
                annotation.apply_flag(flag, active)
    elif feature == "Calories":
        _calories_rules(annotation, row, active)
    elif feature == "Distance":
        _distance_rules(annotation, row, active, start, end)
    elif feature in {"Weight", "Height"}:
        _anthropometric_rules(annotation, feature, row, active, start, end)
    elif feature == "StepCount":
        _step_count_rules(annotation, row, active, step_context, start, end)
    elif policy.unknown_fallback:
        annotation.apply_rule("unknown_feature_policy", active)

    return annotation


_ENGINE_COVERAGE_ISSUES = validate_engine_rule_coverage()
if _ENGINE_COVERAGE_ISSUES:
    raise RuntimeError("Google Fitness curation engine rule coverage is incomplete: " + "; ".join(_ENGINE_COVERAGE_ISSUES))
