"""
Wearable Data Processing and Modeling project
Reviewed and provisional calibration decisions for Google Fitness curation. CWP1 records decisions before an
executable engine exists. Accepted decisions may constrain later implementation; provisional decisions must be
confirmed by CWP2 full-native calibration reports before they become executable policy.
"""


from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping
from .evidence import EVIDENCE
from .models import DecisionStatus, to_primitive
from .rules import RULES


EXPECTED_DECISIONS_FINGERPRINT = "f72b056a228565476ec03221942f000278cec977b321c21f628a6d56e6844108"


@dataclass(frozen=True, slots=True)
class CalibrationDecision:
    decision_id: str
    title: str
    status: DecisionStatus
    selected_option: str
    rejected_options: tuple[str, ...]
    rationale: str
    affected_features: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    rule_ids: tuple[str, ...]
    required_audits: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return to_primitive(self)


_DECISIONS = (
    CalibrationDecision(
        "non_destructive_curation",
        "Preserve native rows and native prefix",
        DecisionStatus.ACCEPTED,
        "Curated rows equal native rows; native columns and row order remain unchanged; only policy-derived columns are appended.",
        ("rewrite native values", "drop excluded rows", "reorder rows for convenience"),
        "The native root is the stable evidence product and curation must remain independently reversible.",
        ("*",),
        ("project_curation_invariant", "project_phase1_native_acceptance"),
        (),
        ("native_prefix_audit", "row_parity_audit"),
    ),
    CalibrationDecision(
        "status_inclusion_separate",
        "Keep curation status and default inclusion independent",
        DecisionStatus.ACCEPTED,
        "Store curation_status and include_by_default as separate decisions.",
        ("derive inclusion from pass/review status",),
        "Valid source rows can be nonpreferred for analysis, while some review rows may remain useful by default.",
        ("*",),
        ("project_curation_invariant",),
        (
            "step_count_merge_fallback",
            "step_count_parallel_nonpreferred",
            "step_count_ambiguous_parallel",
        ),
        ("status_inclusion_accounting_audit",),
    ),
    CalibrationDecision(
        "activity_multimeasurement_rows",
        "Treat activity rows as multimeasurement intervals",
        DecisionStatus.ACCEPTED,
        "Keep calories, distance and quantity coupled to the classified interval; do not force one scalar canonical_value.",
        ("split activity measurements into independent rows", "select one activity measurement as the row value"),
        "The exporter supplied these values together and the native schema intentionally preserves that coupling.",
        tuple(),
        ("project_phase0_activity_submeasurements", "google_fit_activity_segments"),
        ("activity_measurements_nonnegative",),
        ("activity_measurement_concordance",),
    ),
    CalibrationDecision(
        "activity_quantity_unresolved",
        "Block canonical activity quantity semantics",
        DecisionStatus.ACCEPTED,
        "Retain raw quantity but exclude it from canonical/default measurement projection until activity-specific meaning is proven.",
        ("assign one unit to all activity quantity values", "discard quantity"),
        "No defensible common unit or meaning is supported across the activity vocabulary.",
        tuple(),
        ("project_phase0_activity_submeasurements",),
        ("activity_quantity_unresolved",),
        ("activity_quantity_by_label_audit",),
    ),
    CalibrationDecision(
        "sleep_native_states_preserved",
        "Preserve native Google sleep states",
        DecisionStatus.ACCEPTED,
        "Keep SLEEP, AWAKE, DEEP, LIGHT and REM unchanged in curation.",
        ("renormalize states during curation", "map LIGHT to Apple CORE", "map SLEEP to Apple ASLEEP"),
        "The state mapping was already completed losslessly in native processing and cross-source equivalence is not established.",
        ("Sleep",),
        ("google_fit_sleep_states", "project_phase0_sleep_inventory"),
        ("sleep_state_vocabulary", "sleep_no_apple_equivalence"),
        ("sleep_overlap_audit",),
    ),
    CalibrationDecision(
        "source_day_label_context_only",
        "Treat localized day label as context only",
        DecisionStatus.ACCEPTED,
        "Never use source_day_label to create or replace an event date.",
        ("derive canonical date from weekday label", "treat one-day mismatch as corruption"),
        "Phase 0 found localized labels and common adjacent-day differences consistent with timezone boundaries.",
        ("Calories", "Distance", "Weight", "Height"),
        ("project_phase0_day_labels",),
        (
            "source_day_label_context",
            "source_day_label_unrecognized",
            "source_day_label_nonadjacent_disagreement",
        ),
        ("source_day_label_audit",),
    ),
    CalibrationDecision(
        "step_count_preserve_all_rows",
        "Preserve all StepCount versions and alternatives",
        DecisionStatus.ACCEPTED,
        "Retain version 1 and every version-2 alternative; curation annotates analytical role but does not delete or merge rows.",
        ("take maximum", "sum alternatives", "average alternatives", "drop version 1", "deduplicate by date"),
        "The alternatives frequently disagree and cross-version overlap does not prove duplicate identity.",
        ("StepCount",),
        ("project_phase0_stepcount_alternatives", "project_phase0_stepcount_temporal"),
        (
            "step_count_v1_legacy_context",
            "step_count_v2_group_structure",
            "step_count_cross_version_separate",
        ),
        ("stepcount_preference_audit", "stepcount_cross_version_audit"),
    ),
    CalibrationDecision(
        "step_count_preference",
        "Annotate one default StepCount alternative when evidence permits",
        DecisionStatus.PROVISIONAL,
        "Prefer a unique estimated_steps row; otherwise use a unique merge_step_deltas row or sole alternative as reviewable fallback; select none for ambiguous groups.",
        ("always choose the largest value", "always choose merge_step_deltas", "select multiple defaults"),
        "Official guidance supports estimated_steps for Fit-app daily totals, but fallback behavior must be calibrated against the native cohort.",
        ("StepCount",),
        ("google_fit_daily_step_total", "project_phase0_stepcount_alternatives"),
        (
            "step_count_estimated_preferred",
            "step_count_merge_fallback",
            "step_count_sole_fallback",
            "step_count_parallel_nonpreferred",
            "step_count_ambiguous_parallel",
            "step_count_alternatives_disagree",
        ),
        ("stepcount_preference_audit",),
        ("Preference is an analytical default, not a declaration that nonpreferred rows are invalid.",),
    ),
    CalibrationDecision(
        "step_count_no_canonical_day",
        "Do not derive one canonical StepCount day",
        DecisionStatus.ACCEPTED,
        "Preserve outer datetime, summary date and raw interval as independent evidence.",
        ("replace raw interval with summary date", "derive date from UTC outer timestamp alone"),
        "The export lacks sufficient timezone context and summary dates can differ from raw intervals.",
        ("StepCount",),
        ("google_fit_daily_step_total", "project_phase0_stepcount_temporal"),
        ("step_count_temporal_context",),
        ("stepcount_temporal_audit",),
    ),
    CalibrationDecision(
        "negative_calories_buckets",
        "Retain and classify negative Calories by magnitude",
        DecisionStatus.PROVISIONAL,
        "Use tiny, small, material and extreme magnitude buckets; never clip values to zero.",
        ("replace with zero", "drop all negative rows", "treat every negative value identically"),
        "Phase 0 demonstrated distinct numerical regimes; exact status effects must be reproduced by the curation audit before execution.",
        ("Calories",),
        ("project_phase0_negative_calories",),
        (
            "calories_negative_tiny",
            "calories_negative_small",
            "calories_negative_material",
            "calories_negative_extreme",
        ),
        ("negative_calories_audit",),
    ),
    CalibrationDecision(
        "weight_nonpoint_preserved",
        "Preserve nonpoint Weight intervals",
        DecisionStatus.ACCEPTED,
        "Keep start and end boundaries; flag positive duration without collapsing to a point.",
        ("overwrite start with end", "drop interval rows", "invent point timestamp"),
        "A substantial accepted-native subset has nonzero duration despite the official instantaneous data-type contract.",
        ("Weight",),
        ("google_fit_weight", "project_phase0_weight_intervals"),
        ("weight_nonpoint_interval", "weight_long_interval"),
        ("weight_interval_audit",),
    ),
    CalibrationDecision(
        "provenance_dimensions",
        "Separate acquisition, derivation and tracking dimensions",
        DecisionStatus.ACCEPTED,
        "Append acquisition_method, derivation_method and tracking_status independently where applicable.",
        ("one generic provenance label", "infer physical device from package alone"),
        "The native source exposes application, stream, origin, device and tracking evidence that answer different questions.",
        ("*",),
        ("project_phase0_provenance",),
        (
            "activity_tracking_status",
            "activity_provenance_classification",
            "activity_provenance_unknown",
            "anthropometric_added_by_context",
        ),
        ("provenance_classification_coverage",),
    ),
    CalibrationDecision(
        "provenance_classifier",
        "Use conservative provenance classification",
        DecisionStatus.PROVISIONAL,
        "Classify only when package/stream evidence is explicit; otherwise retain health_import_unknown or unknown.",
        ("infer user_entered from any app package", "infer device_measurement from a scale-brand package"),
        "Application identity does not uniquely establish acquisition method.",
        ("*",),
        ("project_phase0_provenance",),
        ("activity_provenance_classification", "activity_provenance_unknown", "anthropometric_added_by_context"),
        ("provenance_classification_coverage",),
    ),
    CalibrationDecision(
        "calories_unit_mapping",
        "Map outer Calories to the Google calories-expended contract only after exporter verification",
        DecisionStatus.PROVISIONAL,
        "Proposed canonical unit kcal; withhold canonical values until exporter mapping is frozen.",
        ("assume Apple active energy", "infer unit from magnitude"),
        "The official type contract is clear, but the outer exporter name alone does not prove the queried type.",
        ("Calories",),
        ("google_fit_calories_expended", "project_phase0_census"),
        ("unit_mapping_pending",),
        ("unit_contract_matrix", "activity_measurement_concordance"),
    ),
    CalibrationDecision(
        "distance_unit_mapping",
        "Map outer Distance to meters only after exporter verification",
        DecisionStatus.PROVISIONAL,
        "Proposed canonical unit m; withhold canonical values until exporter mapping is frozen.",
        ("infer miles or kilometers by magnitude",),
        "The official distance contract is clear, but exporter mapping remains separate evidence.",
        ("Distance",),
        ("google_fit_distance_delta", "project_phase0_census"),
        ("unit_mapping_pending",),
        ("unit_contract_matrix", "activity_measurement_concordance"),
    ),
    CalibrationDecision(
        "weight_unit_mapping",
        "Map outer Weight to kilograms only after exporter verification",
        DecisionStatus.PROVISIONAL,
        "Proposed canonical unit kg; withhold canonical values until exporter mapping is frozen.",
        ("infer pounds or kilograms from magnitude",),
        "The official weight type is kilograms, but exporter mapping must be named evidence.",
        ("Weight",),
        ("google_fit_weight", "project_phase0_census"),
        ("unit_mapping_pending", "weight_source_range"),
        ("unit_contract_matrix", "weight_interval_audit"),
    ),
    CalibrationDecision(
        "height_unit_mapping",
        "Map outer Height to meters only after exporter verification",
        DecisionStatus.PROVISIONAL,
        "Proposed canonical unit m; withhold canonical values until exporter mapping is frozen.",
        ("infer inches or centimeters from magnitude",),
        "The official height type is meters, but only two native rows exist and exporter mapping must be explicit.",
        ("Height",),
        ("google_fit_height", "project_phase0_census"),
        ("unit_mapping_pending", "height_source_range"),
        ("unit_contract_matrix",),
    ),
    CalibrationDecision(
        "step_count_unit_mapping",
        "Resolve StepCount version 2 from payload-explicit type evidence",
        DecisionStatus.ACCEPTED,
        "Use count as the canonical unit for version 2; treat version 1 as a source-convention count pending audit.",
        ("infer unit from value distribution",),
        "Version 2 carries com.google.step_count.delta evidence; version 1 lacks equivalent payload-explicit metadata.",
        ("StepCount",),
        ("google_fit_step_count_delta", "project_phase0_stepcount_alternatives"),
        ("unit_payload_explicit", "step_count_v1_legacy_context"),
        ("unit_contract_matrix",),
    ),
    CalibrationDecision(
        "unknown_feature_fallback",
        "Use a safe fallback for unknown future native features",
        DecisionStatus.ACCEPTED,
        "Copy native rows unchanged, append review/exclusion metadata, and perform no canonical conversion.",
        ("guess policy from column names", "silently skip unknown files"),
        "Future source drift must remain visible without inventing semantics.",
        ("__unknown__",),
        ("project_curation_invariant",),
        ("unknown_feature_policy",),
        ("policy_coverage",),
    ),
)

DECISIONS: Mapping[str, CalibrationDecision] = MappingProxyType(
    {item.decision_id: item for item in sorted(_DECISIONS, key=lambda item: item.decision_id)}
)


def known_decisions() -> tuple[str, ...]:
    return tuple(DECISIONS)


def get_decision(decision_id: str) -> CalibrationDecision:
    try:
        return DECISIONS[decision_id]
    except KeyError as exc:
        raise KeyError(f"Unknown Google Fitness curation decision: {decision_id!r}") from exc


def validate_decisions() -> tuple[str, ...]:
    issues: list[str] = []
    if len(DECISIONS) != len(_DECISIONS):
        issues.append("duplicate decision IDs")
    if decisions_fingerprint() != EXPECTED_DECISIONS_FINGERPRINT:
        issues.append(
            "decisions fingerprint mismatch: "
            f"expected {EXPECTED_DECISIONS_FINGERPRINT}, observed {decisions_fingerprint()}"
        )
    for decision_id, item in DECISIONS.items():
        if decision_id != item.decision_id:
            issues.append(f"decision key mismatch for {decision_id}")
        missing_evidence = sorted(set(item.evidence_refs) - set(EVIDENCE))
        if missing_evidence:
            issues.append(f"decision {decision_id} references unknown evidence: {missing_evidence}")
        missing_rules = sorted(set(item.rule_ids) - set(RULES))
        if missing_rules:
            issues.append(f"decision {decision_id} references unknown rules: {missing_rules}")
        if item.status is DecisionStatus.PROVISIONAL and not item.required_audits:
            issues.append(f"provisional decision lacks required audit: {decision_id}")
    return tuple(sorted(set(issues)))


def decisions_payload() -> dict[str, object]:
    return {
        "decisions": [DECISIONS[name].as_dict() for name in DECISIONS],
        "count": len(DECISIONS),
        "fingerprint": decisions_fingerprint(),
        "validation_issues": list(validate_decisions()),
    }


def decisions_fingerprint() -> str:
    rendered = json.dumps(
        [DECISIONS[name].as_dict() for name in DECISIONS],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(rendered).hexdigest()


_DECISION_ISSUES = validate_decisions()
if _DECISION_ISSUES:
    raise RuntimeError("Invalid Google Fitness curation decisions: " + "; ".join(_DECISION_ISSUES))
