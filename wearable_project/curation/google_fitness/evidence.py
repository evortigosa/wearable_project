"""
Wearable Data Processing and Modeling project
Evidence catalog for Google Fitness curation policy design. The catalog separates official Google data-type
contracts from project-specific exporter observations. An official data type documents the unit and semantics
of that Google type; it does not, by itself, prove that every cohort export field was produced from that type.
That exporter-mapping distinction is a first-class limitation in CWP1.
"""


from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping
from .models import EvidenceGrade, to_primitive


EXPECTED_EVIDENCE_FINGERPRINT = "820f23c43c7cade8a287595393ca6ec61cdb469af60b392f59915c990c507e44"


class EvidenceKind(str, Enum):
    OFFICIAL_SOURCE_CONTRACT = "official_source_contract"
    OFFICIAL_USAGE_GUIDANCE = "official_usage_guidance"
    PROJECT_CENSUS = "project_census"
    PROJECT_NATIVE_ACCEPTANCE = "project_native_acceptance"
    PROJECT_ARCHITECTURE = "project_architecture"
    PROJECT_LIMITATION = "project_limitation"


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    evidence_id: str
    title: str
    kind: EvidenceKind
    authority: str
    grade: EvidenceGrade
    summary: str
    supports: tuple[str, ...]
    limitations: tuple[str, ...]
    url: str | None = None
    artifact: str | None = None
    retrieved_date: str | None = None

    def as_dict(self) -> dict[str, object]:
        return to_primitive(self)


_SOURCES = (
    EvidenceSource(
        "google_fit_data_model",
        "Google Fit data types",
        EvidenceKind.OFFICIAL_SOURCE_CONTRACT,
        "Google for Developers",
        EvidenceGrade.A_DIRECT,
        "Google Fit data points are typed observations or aggregates with field and timestamp contracts.",
        ("typed_source_contracts", "instantaneous_vs_interval_semantics"),
        ("Does not identify which Google type the cohort exporter queried for every outer feature.",),
        url="https://developers.google.com/fit/datatypes",
        retrieved_date="2026-10-19",
    ),
    EvidenceSource(
        "google_fit_activity_segments",
        "Google Fit activity data types",
        EvidenceKind.OFFICIAL_SOURCE_CONTRACT,
        "Google for Developers",
        EvidenceGrade.A_DIRECT,
        "The activity data type represents classified activity segments over time intervals.",
        ("activity_interval_semantics", "activity_classification_context"),
        (
            "The cohort exporter adds calories, distance, quantity, tracking and provenance fields that are not proven to be "
            "fields of the standard activity-segment type.",
        ),
        url="https://developers.google.com/fit/datatypes/activity",
        retrieved_date="2026-10-19",
    ),
    EvidenceSource(
        "google_fit_calories_expended",
        "Google Fit calories expended contract",
        EvidenceKind.OFFICIAL_SOURCE_CONTRACT,
        "Google for Developers",
        EvidenceGrade.A_DIRECT,
        "com.google.calories.expended is an interval total in kilocalories and includes resting energy.",
        ("calories_unit_kcal", "calories_interval_total", "calories_include_resting_energy"),
        ("The cohort outer feature name Calories does not by itself prove exporter mapping to com.google.calories.expended.",),
        url="https://developers.google.com/fit/datatypes/activity#calories_burned",
        retrieved_date="2026-10-19",
    ),
    EvidenceSource(
        "google_fit_distance_delta",
        "Google Fit distance delta contract",
        EvidenceKind.OFFICIAL_SOURCE_CONTRACT,
        "Google for Developers",
        EvidenceGrade.A_DIRECT,
        "com.google.distance.delta is an interval total in meters with a documented source-valid rate of 0 to 100 meters per second.",
        ("distance_unit_m", "distance_interval_total", "distance_source_rate_limit"),
        ("The cohort outer feature name Distance does not by itself prove exporter mapping to com.google.distance.delta.",),
        url="https://developers.google.com/fit/datatypes/location#distance_delta",
        retrieved_date="2026-10-19",
    ),
    EvidenceSource(
        "google_fit_step_count_delta",
        "Google Fit step count delta contract",
        EvidenceKind.OFFICIAL_SOURCE_CONTRACT,
        "Google for Developers",
        EvidenceGrade.A_DIRECT,
        "com.google.step_count.delta is an interval count with a documented source-valid rate of 0 to 10 steps per second.",
        ("step_count_unit_count", "step_count_interval_semantics", "step_count_source_rate_limit"),
        (),
        url="https://developers.google.com/fit/datatypes/activity#step_count_delta",
        retrieved_date="2026-10-19",
    ),
    EvidenceSource(
        "google_fit_daily_step_total",
        "Google Fit daily step total guidance",
        EvidenceKind.OFFICIAL_USAGE_GUIDANCE,
        "Google for Developers",
        EvidenceGrade.A_DIRECT,
        "Google documents estimated_steps as the derived source used to reproduce the Fit app daily total; daily boundaries use the device current timezone.",
        ("estimated_steps_preference_evidence", "local_day_context"),
        ("The cohort export does not retain enough timezone context to reconstruct one defensible canonical calendar day for every row.",),
        url="https://developers.google.com/fit/scenarios/read-daily-step-total",
        retrieved_date="2026-10-19",
    ),
    EvidenceSource(
        "google_fit_weight",
        "Google Fit weight contract",
        EvidenceKind.OFFICIAL_SOURCE_CONTRACT,
        "Google for Developers",
        EvidenceGrade.A_DIRECT,
        "com.google.weight is an instantaneous body-weight reading in kilograms with a documented 0 to 1000 kg range.",
        ("weight_unit_kg", "weight_point_semantics", "weight_source_range"),
        ("The cohort outer feature name Weight does not by itself prove exporter mapping to com.google.weight.",),
        url="https://developers.google.com/fit/datatypes/health#weight",
        retrieved_date="2026-10-19",
    ),
    EvidenceSource(
        "google_fit_height",
        "Google Fit height contract",
        EvidenceKind.OFFICIAL_SOURCE_CONTRACT,
        "Google for Developers",
        EvidenceGrade.A_DIRECT,
        "com.google.height is an instantaneous height reading in meters with a documented 0 to 3 m range.",
        ("height_unit_m", "height_point_semantics", "height_source_range"),
        ("The cohort outer feature name Height does not by itself prove exporter mapping to com.google.height.",),
        url="https://developers.google.com/fit/datatypes/health#height",
        retrieved_date="2026-10-19",
    ),
    EvidenceSource(
        "google_fit_sleep_states",
        "Google Fit sleep-state vocabulary",
        EvidenceKind.OFFICIAL_SOURCE_CONTRACT,
        "Google for Developers",
        EvidenceGrade.A_DIRECT,
        "Google Fit documents awake, sleeping, light, deep and REM sleep segment categories.",
        ("google_sleep_state_vocabulary",),
        ("This does not establish one-to-one equivalence with Apple HealthKit sleep states.",),
        url="https://developers.google.com/fit/datatypes/health#sleep",
        retrieved_date="2026-10-19",
    ),
    EvidenceSource(
        "project_native_registry",
        "Google Fitness native registry WP1",
        EvidenceKind.PROJECT_ARCHITECTURE,
        "HPP wearable project",
        EvidenceGrade.A_DIRECT,
        "The accepted native registry defines 67 fixed feature schemas, 66 activity routes and source/version contracts.",
        ("native_schema_contract", "activity_routing", "version_contracts"),
        (),
        artifact="wearable_project.processing.google_fitness.registry",
    ),
    EvidenceSource(
        "project_phase0_census",
        "Phase 0 full-cohort Google Fitness census",
        EvidenceKind.PROJECT_CENSUS,
        "HPP wearable project",
        EvidenceGrade.A_DIRECT,
        "The full-cohort inspection established source, feature, version, schema, activity-label and provenance inventories.",
        ("cohort_feature_inventory", "version_coexistence", "schema_coverage"),
        (),
        artifact="wearable_census_mp phase0 closure reports",
    ),
    EvidenceSource(
        "project_phase0_activity_submeasurements",
        "Phase 0 activity submeasurement profiles",
        EvidenceKind.PROJECT_CENSUS,
        "HPP wearable project",
        EvidenceGrade.B_CONVERGENT,
        "Activity rows commonly carry calories, distance and quantity coupled to the classified interval.",
        ("activity_measurement_presence", "activity_measurements_stay_coupled"),
        ("The exporter-specific units and meaning of quantity remain unresolved and may vary by activity.",),
        artifact="activity_submeasurement_profile.csv",
    ),
    EvidenceSource(
        "project_phase0_stepcount_alternatives",
        "Phase 0 StepCount alternative audit",
        EvidenceKind.PROJECT_CENSUS,
        "HPP wearable project",
        EvidenceGrade.A_DIRECT,
        "Version 2 contains named alternatives that frequently disagree and must remain separate source observations.",
        ("step_count_alternative_preservation", "step_count_disagreement"),
        (),
        artifact="stepcount_alternative_summary.csv",
    ),
    EvidenceSource(
        "project_phase0_stepcount_temporal",
        "Phase 0 StepCount temporal audit",
        EvidenceKind.PROJECT_CENSUS,
        "HPP wearable project",
        EvidenceGrade.A_DIRECT,
        "Outer datetime, summary date and raw interval are distinct temporal evidence; summary dates sometimes differ from raw interval dates.",
        ("step_count_multiple_time_contexts", "block_canonical_day"),
        (),
        artifact="stepcount_temporal_alignment_counts.csv",
    ),
    EvidenceSource(
        "project_phase0_negative_calories",
        "Phase 0 negative Calories distribution",
        EvidenceKind.PROJECT_CENSUS,
        "HPP wearable project",
        EvidenceGrade.A_DIRECT,
        "Negative Calories values occur in tiny, small, material and extreme magnitude regimes.",
        ("negative_calories_buckets", "do_not_clip_calories"),
        (),
        artifact="calories_negative_distribution.csv",
    ),
    EvidenceSource(
        "project_phase0_weight_intervals",
        "Phase 0 Weight interval audit",
        EvidenceKind.PROJECT_CENSUS,
        "HPP wearable project",
        EvidenceGrade.A_DIRECT,
        "A material subset of valid Weight rows has nonzero duration and therefore must retain both native boundaries.",
        ("weight_nonpoint_intervals", "preserve_weight_boundaries"),
        (),
        artifact="weight_interval_duration_distribution.csv",
    ),
    EvidenceSource(
        "project_phase0_provenance",
        "Phase 0 Google provenance inventory",
        EvidenceKind.PROJECT_CENSUS,
        "HPP wearable project",
        EvidenceGrade.B_CONVERGENT,
        "Payloads expose separate application, source-stream, origin-stream and device dimensions.",
        ("multidimensional_provenance", "avoid_single_provenance_label"),
        ("Raw package and stream identifiers do not prove a unique physical device or user-entry method.",),
        artifact="participant_provenance_summary.csv and provenance_values.csv",
    ),
    EvidenceSource(
        "project_phase0_day_labels",
        "Phase 0 localized day-label audit",
        EvidenceKind.PROJECT_CENSUS,
        "HPP wearable project",
        EvidenceGrade.A_DIRECT,
        "Localized weekday labels sometimes differ from UTC-derived dates by one day and are not canonical event dates.",
        ("source_day_label_context_only",),
        (),
        artifact="temporal_alignment_counts.csv",
    ),
    EvidenceSource(
        "project_phase0_sleep_inventory",
        "Phase 0 Google sleep inventory",
        EvidenceKind.PROJECT_CENSUS,
        "HPP wearable project",
        EvidenceGrade.A_DIRECT,
        "Five Google sleep labels were observed and losslessly normalized into the native Sleep state vocabulary.",
        ("native_sleep_state_mapping",),
        ("The census alone does not establish Apple/Google sleep-state equivalence.",),
        artifact="activity_label_inventory.csv",
    ),
    EvidenceSource(
        "project_phase1_native_acceptance",
        "Phase 1 Google native full-cohort acceptance",
        EvidenceKind.PROJECT_NATIVE_ACCEPTANCE,
        "HPP wearable project",
        EvidenceGrade.A_DIRECT,
        "The accepted native root contains 13,476,849 rows in 32,707 feature files with exact raw-to-native accounting and no silent candidate loss.",
        ("native_product_stability", "curation_input_contract"),
        (),
        artifact="google_fitness_native_first_acceptance.json",
    ),
    EvidenceSource(
        "project_native_malformed_weight",
        "Native malformed Weight ledger",
        EvidenceKind.PROJECT_LIMITATION,
        "HPP wearable project",
        EvidenceGrade.A_DIRECT,
        "Four malformed Weight 2.0 source items have no numeric native row and remain in the native state audit ledger.",
        ("malformed_weight_not_curated",),
        (),
        artifact=".wearable_state.sqlite malformed_candidates",
    ),
    EvidenceSource(
        "project_curation_invariant",
        "Project non-destructive curation invariant",
        EvidenceKind.PROJECT_ARCHITECTURE,
        "HPP wearable project",
        EvidenceGrade.A_DIRECT,
        "Curation preserves native columns and row order and appends only policy-derived information.",
        ("native_prefix_preservation", "row_parity", "no_silent_deletion"),
        (),
        artifact="accepted Apple curation architecture",
    ),
)

EVIDENCE: Mapping[str, EvidenceSource] = MappingProxyType(
    {item.evidence_id: item for item in sorted(_SOURCES, key=lambda item: item.evidence_id)}
)


def known_evidence() -> tuple[str, ...]:
    return tuple(EVIDENCE)


def get_evidence(evidence_id: str) -> EvidenceSource:
    try:
        return EVIDENCE[evidence_id]
    except KeyError as exc:
        raise KeyError(f"Unknown Google Fitness curation evidence: {evidence_id!r}") from exc


def validate_evidence() -> tuple[str, ...]:
    issues: list[str] = []
    if len(EVIDENCE) != len(_SOURCES):
        issues.append("duplicate evidence IDs")
    if evidence_fingerprint() != EXPECTED_EVIDENCE_FINGERPRINT:
        issues.append(
            "evidence fingerprint mismatch: "
            f"expected {EXPECTED_EVIDENCE_FINGERPRINT}, observed {evidence_fingerprint()}"
        )
    for evidence_id, item in EVIDENCE.items():
        if evidence_id != item.evidence_id:
            issues.append(f"evidence key mismatch for {evidence_id}")
        if item.kind in {EvidenceKind.OFFICIAL_SOURCE_CONTRACT, EvidenceKind.OFFICIAL_USAGE_GUIDANCE} and not item.url:
            issues.append(f"official evidence lacks URL: {evidence_id}")
        if item.kind not in {EvidenceKind.OFFICIAL_SOURCE_CONTRACT, EvidenceKind.OFFICIAL_USAGE_GUIDANCE} and not item.artifact:
            issues.append(f"project evidence lacks artifact: {evidence_id}")
        if not item.supports:
            issues.append(f"evidence has no declared support: {evidence_id}")
    return tuple(sorted(set(issues)))


def evidence_payload() -> dict[str, object]:
    return {
        "sources": [EVIDENCE[name].as_dict() for name in EVIDENCE],
        "count": len(EVIDENCE),
        "fingerprint": evidence_fingerprint(),
        "validation_issues": list(validate_evidence()),
    }


def evidence_fingerprint() -> str:
    rendered = json.dumps(
        [EVIDENCE[name].as_dict() for name in EVIDENCE],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(rendered).hexdigest()


_EVIDENCE_ISSUES = validate_evidence()
if _EVIDENCE_ISSUES:
    raise RuntimeError("Invalid Google Fitness evidence catalog: " + "; ".join(_EVIDENCE_ISSUES))
