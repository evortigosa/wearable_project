"""
Wearable Data Processing and Modeling project
User-facing, evidence-linked guidance for every cohort feature. Guidance is explanatory metadata. It does not
execute curation rules and is fingerprinted separately from the executable policy registry so that wording or
citation updates do not invalidate curated participant outputs.
"""


from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Iterable
from wearable_project.curation.evidence import EVIDENCE
from wearable_project.curation.registry import COHORT_OBSERVED_FEATURES, get_policy


GUIDANCE_VERSION = "0.2.0a2.2-guidance-4"


@dataclass(frozen=True, slots=True)
class FeatureGuide:
    feature: str
    category: str
    short_description: str
    one_row_means: str
    native_time_semantics: str
    unit_summary: str
    acquisition_summary: str
    curation_summary: str
    important_caveats: tuple[str, ...]
    not_equivalent_to: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _GuideSpec:
    category: str
    short_description: str
    one_row_means: str
    acquisition_summary: str
    curation_summary: str
    caveats: tuple[str, ...] = ()
    not_equivalent_to: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()


_SPECS: dict[str, _GuideSpec] = {
    "StepCount": _GuideSpec(
        "Physical activity and energy",
        "Number of steps accumulated over a source-defined interval.",
        "One interval total for the original start and end timestamps.",
        "Usually estimated by motion sensors and may be written or merged by several devices or applications.",
        "Preserve the native interval total; flag invalid duration, negative values, unresolved interval conflicts, and source-epoch changes.",
        ("A longer interval does not reveal when within that interval the steps occurred.",),
        ("An observed five-minute step count unless the native interval is actually five minutes.",),
        ("apple_step_count", "apple_healthkit_units", "project_interval_boundary_revisions"),
    ),
    "DistanceWalkingRunning": _GuideSpec(
        "Physical activity and energy",
        "Distance accumulated while walking or running over a source-defined interval.",
        "One extensive distance total attached to its native interval.",
        "Estimated from motion, location, device, and application inputs; source provenance should be retained.",
        "Retain metres and the original interval; screen duration, nonnegativity, implied rate, source transitions, and unresolved interval conflicts.",
        ("The total does not encode the within-interval trajectory or activity type mix.",),
        ("A continuously sampled speed trace.",),
        ("apple_distance_walking_running", "project_unit_fingerprints", "project_interval_boundary_revisions"),
    ),
    "DailyDistanceCycling": _GuideSpec(
        "Physical activity and energy",
        "Cycling-distance total; this exporter may store interval records despite the word Daily.",
        "One native interval distance total, not necessarily one calendar-day total.",
        "May be estimated or imported from cycling devices and applications; the full-cohort exporter output is represented in metres.",
        "Retain metre interval totals, source provenance, and non-destructive duration/implied-speed quality flags.",
        ("The name does not prove daily granularity.", "Metres are the exported unit, but extreme values can still be erroneous."),
        ("A daily summary unless the timestamps demonstrate that interpretation.",),
        ("apple_cycling_distance", "project_unit_fingerprints"),
    ),
    "DailyDistanceSwimming": _GuideSpec(
        "Physical activity and energy",
        "Swimming-distance total over a native source interval.",
        "One interval total with original timestamps; observed pool-session fingerprints support metres in this cohort.",
        "Typically produced by a watch or swimming application when a swim is recorded.",
        "Retain native intervals, metre values, source provenance, and conflict flags.",
        ("The feature name does not guarantee one row per day.",),
        ("A lap-by-lap trajectory.",),
        ("apple_swimming_distance", "project_unit_fingerprints"),
    ),
    "FlightsClimbed": _GuideSpec(
        "Physical activity and energy",
        "Count of flights climbed over a native interval.",
        "One cumulative count attached to its source interval.",
        "Usually estimated from barometric and motion signals, but source/application provenance can vary.",
        "Preserve counts and intervals; screen nonnegative values, duration, revisions, and source epochs.",
        ("A flight is a HealthKit quantity, not a direct floor-by-floor elevation trace.",),
        ("Exact vertical metres climbed.",),
        ("apple_flights_climbed", "apple_healthkit_units"),
    ),
    "ActiveEnergyBurned": _GuideSpec(
        "Physical activity and energy",
        "Energy attributed to physical activity over a native interval.",
        "One extensive energy total for the recorded interval.",
        "Estimated by a device or application using movement, heart rate, profile data, and proprietary algorithms.",
        "Preserve native kcal totals; screen interval duration, nonnegativity, revisions, conflicts, and source changes.",
        ("It is an estimate and may vary by device, algorithm, and user profile.",),
        ("Direct calorimetry or a time-resolved energy-rate signal.",),
        ("apple_active_energy", "apple_healthkit_quantity_types", "project_interval_boundary_revisions"),
    ),
    "BasalEnergyBurned": _GuideSpec(
        "Physical activity and energy",
        "Estimated resting or basal energy expenditure over a native interval.",
        "One extensive energy total for the source interval.",
        "Usually calculated from profile attributes and device/application models rather than directly measured.",
        "Preserve native kcal totals; screen duration, nonnegativity, revisions, source epochs, and unresolved conflicts.",
        ("Model changes or profile updates can change values without a physiological event.",),
        ("A laboratory basal metabolic rate test.",),
        ("apple_basal_energy", "apple_healthkit_quantity_types", "project_interval_boundary_revisions"),
    ),
    "ActivitySummary": _GuideSpec(
        "Physical activity and energy",
        "Daily move, exercise, and stand summary object.",
        "One exported payload item from an HKActivitySummary object; the exporter did not preserve HealthKit date components.",
        "Created by the Apple activity-summary system rather than an ordinary timestamped sample.",
        "Preserve every payload item and payload index; block unique daily-date derivation until the sliding-pair mapping is established.",
        ("The outer UTC bucket is not proven to be the represented local calendar day.", "Observed adjacent rows form a sliding-pair pattern."),
        ("An ordinary HealthKit sample with start_date and end_date.", "A five-minute activity value."),
        ("apple_activity_summary", "project_activity_summary_sliding_pair"),
    ),
    "HeartRate": _GuideSpec(
        "Heart and cardiovascular",
        "Heart-rate observation expressed as beats per minute.",
        "One native point or short interval observation with its original source and motion context when present.",
        "May come from PPG, another sensor, manual entry, or an imported application; cadence varies by source.",
        "Preserve all native observations, source identity, motion context, duration, and provenance; do not aggregate in Milestone 2.",
        ("Sampling density is not uniform and record counts should not be treated as equal-duration support.",),
        ("An ECG rhythm diagnosis.", "A universal five-minute median heart rate."),
        ("apple_heart_rate",),
    ),
    "RestingHeartRate": _GuideSpec(
        "Heart and cardiovascular",
        "Device- or application-estimated resting heart-rate summary.",
        "One source-defined summary represented either as a point or as an explicit interval.",
        "Usually estimated from periods of low activity and source-specific algorithms.",
        "Preserve point-or-interval summary semantics and source epochs; distinguish estimates from dense heart-rate observations.",
        ("A zero-duration summary is valid; an interval summary still does not imply continuous measurement at the summary value.",),
        ("A resting clinical vital-sign measurement taken under a standardized protocol.",),
        ("apple_resting_heart_rate", "apple_heart_rate"),
    ),
    "WalkingHeartRate": _GuideSpec(
        "Heart and cardiovascular",
        "Estimated average heart rate associated with walking.",
        "One source-defined walking-heart-rate summary represented either as a point or as an explicit interval.",
        "Produced by HealthKit/device estimation and may be revised as more data become available.",
        "Preserve one point-or-interval summary event; never broadcast it into dense windows by default.",
        ("A zero-duration summary is valid; long interval coverage does not imply high information density.",),
        ("Continuous heart-rate observations throughout the interval.",),
        ("apple_walking_heart_rate",),
    ),
    "HeartRateVariability": _GuideSpec(
        "Heart and cardiovascular",
        "Heart-rate variability represented as SDNN over a short source interval.",
        "One native HRV estimate with source, interval, and algorithm context.",
        "Typically derived from beat-to-beat timing by a device or application algorithm.",
        "Convert the reviewed exporter seconds representation to millisecond SDNN while retaining raw values, source, and algorithm version.",
        ("SDNN depends on recording duration, signal quality, and algorithm; values from unlike contexts are not automatically comparable.",),
        ("A generic stress score or a diagnosis of autonomic dysfunction.",),
        ("apple_hrv_sdnn", "hrv_task_force_1996", "project_unit_fingerprints"),
    ),
    "Vo2Max": _GuideSpec(
        "Heart and cardiovascular",
        "Maximum oxygen-consumption estimate or test result in mL/kg/min.",
        "One point estimate with source and test-type metadata when available.",
        "May be a device estimate, nonexercise prediction, submaximal prediction, or maximal test result.",
        "Retain test type and acquisition provenance; compare values only within compatible methods and sources.",
        ("Apple Watch values are estimates and method metadata materially affects interpretation.",),
        ("A directly measured laboratory cardiopulmonary exercise test unless provenance supports that claim.",),
        ("apple_vo2max",),
    ),
    "BloodPressure": _GuideSpec(
        "Heart and cardiovascular",
        "Paired systolic and diastolic blood-pressure measurement.",
        "One coupled point-vector event containing both pressure components.",
        "May be measured by a cuff, imported, or manually entered; technique, device, cuff, posture, and repeated-reading context matter.",
        "Keep the pair coupled; flag missing components, reversed ordering, source context, and repeated-reading sessions without synthesizing a new pair.",
        ("A single reading is sensitive to measurement protocol and context.",),
        ("Two independent scalar series that can be median-combined separately.",),
        ("apple_blood_pressure", "aha_blood_pressure_measurement"),
    ),
    "Electrocardiogram": _GuideSpec(
        "Heart and cardiovascular",
        "One ECG waveform event with summary metadata and an internal sample clock.",
        "One recording containing voltage measurements, sampling frequency, classification, and average heart rate when available.",
        "HealthKit provides read-only ECGs recorded by supported devices such as Apple Watch.",
        "Validate sample count, frequency, relative-time monotonicity, finite amplitudes, classification, and algorithm version while preserving the waveform as one event.",
        ("Classification is an algorithm output and the waveform unit may remain unresolved in this exporter.",),
        ("A scalar feature suitable for ordinary fixed-window broadcasting.", "A definitive clinical diagnosis."),
        ("apple_ecg", "apple_ecg_sampling_frequency"),
    ),
    "Sleep": _GuideSpec(
        "Sleep and recovery",
        "Categorical intervals for time in bed and sleep states.",
        "One state interval from one source, such as INBED, AWAKE, CORE, DEEP, REM, or unspecified ASLEEP.",
        "Created by watches, sleep applications, or manual/imported sources; sources may overlap.",
        "Preserve every interval; allow INBED to overlap detailed stages, flag same-source stage conflicts, retain cross-source overlap as informational provenance, distinguish sources that never emit INBED from detailed stages outside available INBED, and never use lexical mode.",
        (
            "Detailed samples may not cover the beginning or end of an INBED interval.",
            "A source may emit detailed stages without exporting INBED; this is recorded as context rather than treated as a failed sleep-state measurement.",
            "Different sources can provide overlapping or incompatible state models, so cross-source overlap is not interpreted as an internal stage conflict.",
        ),
        ("One exclusive state per time bin without an explicit hierarchy.",),
        ("apple_sleep_analysis", "project_sleep_intervals"),
    ),
    "Mindful": _GuideSpec(
        "Sleep and recovery",
        "Mindfulness or mindful-session duration event.",
        "One interval marking the start and end of a recorded mindful session.",
        "May be created by Apple or third-party mindfulness applications.",
        "Validate interval order and duration while preserving source and record identity.",
        ("The interval indicates a recorded session, not verified attention or physiological state.",),
        ("A continuous mental-state sensor.",),
        ("apple_mindful_session",),
    ),
    "BloodGlucose": _GuideSpec(
        "Glucose and metabolism",
        "Discrete blood-glucose measurement or CGM sample.",
        "One point observation with value plus status, trend, device, and time-zone context when exported.",
        "May come from a CGM, meter, manual entry, or application import; sources may use mg/dL or mmol/L.",
        "Resolve units by participant/source epoch, preserve CGM status and trend metadata, and withhold canonical conversion when the epoch is ambiguous.",
        ("Regional and device unit conventions differ.", "CGM values, finger-stick values, and manual values are not automatically interchangeable."),
        ("A five-minute interval total or an automatically diagnosed glycemic event.",),
        ("apple_blood_glucose", "project_cgm_metadata", "ada_cgm_2026"),
    ),
    "OxygenSaturation": _GuideSpec(
        "Respiratory and vital signs",
        "Discrete oxygen-saturation estimate stored as a HealthKit percent quantity.",
        "One point estimate, represented on the native 0-1 fraction scale in this export before optional conversion to percent.",
        "Usually produced by an optical sensor or imported application; device and acquisition context affect interpretation.",
        "Convert reviewed fraction values to percent, retain device/source context, and flag limited provenance rather than diagnose from a threshold.",
        ("Pulse oximeters have accuracy limitations under some conditions and performance can differ across skin pigmentation.",),
        ("An arterial blood-gas measurement or a diagnosis from one reading.",),
        ("apple_oxygen_saturation", "apple_healthkit_units", "fda_pulse_oximetry"),
    ),
    "RespiratoryRate": _GuideSpec(
        "Respiratory and vital signs",
        "Discrete respiratory-rate estimate in breaths per minute.",
        "One point observation; in this cohort many observations are nocturnal but that is not universal.",
        "May be estimated by a wearable or imported from another source.",
        "Preserve points, source, and time context; do not require overlap with exported Sleep intervals.",
        ("Nocturnal timing does not prove that every sample occurred during a recorded sleep state.",),
        ("A continuous respiratory waveform or a clinical respiratory assessment.",),
        ("apple_respiratory_rate",),
    ),
    "BodyTemperature": _GuideSpec(
        "Respiratory and vital signs",
        "Discrete body-temperature observation.",
        "One point measurement with source, user-entry status, and sensor-location metadata when present.",
        "May be manually entered or imported from a thermometer/application; the current exporter output is represented in degrees Celsius.",
        "Retain Celsius values, source and measurement-site context, and avoid universal screening when site or method is missing.",
        ("Temperature thresholds depend on measurement site and method.",),
        ("A core-temperature measurement unless the source and sensor location establish it.",),
        ("apple_body_temperature", "apple_body_temperature_location", "project_unit_fingerprints"),
    ),
    "PeakFlow": _GuideSpec(
        "Respiratory and vital signs",
        "Maximum flow achieved during a forceful exhalation.",
        "One discrete point measurement; the exporter does not encode maneuver quality or session identity.",
        "May be manually entered or imported from a peak-flow meter or spirometry application.",
        "Preserve every point, source, and user-entry context; do not infer best-of-session or convert units until the exporter/source convention is reviewed.",
        ("Effort, technique, repeatability, and device quality are not encoded in the sampled payload.",),
        ("A complete standardized spirometry test or automatically selected best maneuver.",),
        ("apple_peak_flow", "ats_ers_spirometry_2019"),
    ),
    "Height": _GuideSpec(
        "Anthropometrics and body composition",
        "Discrete height measurement.",
        "One point observation whose raw length unit may depend on participant and source epoch.",
        "Often user entered or imported from an application; repeated values may be sparse.",
        "Resolve metres/centimetres/inches with source, continuity, and BMI evidence; retain raw values when ambiguous.",
        ("User-entered status does not directly identify the unit.",),
        ("A time-varying dense signal.",),
        ("apple_height", "who_bmi", "project_unit_fingerprints"),
    ),
    "Weight": _GuideSpec(
        "Anthropometrics and body composition",
        "Discrete body-mass measurement.",
        "One point observation whose raw mass unit may vary by participant/source epoch.",
        "May come from a scale, application import, or manual entry.",
        "Resolve kilograms/pounds using source epochs, continuity, BMI, and body-composition evidence; retain raw values when ambiguous.",
        ("Identical values with different source records can be repeated synchronization or legitimate repeated measurements.",),
        ("A diagnosis or a unit-resolved value when the source epoch is unresolved.",),
        ("apple_body_mass", "who_bmi", "project_body_composition_consistency", "project_unit_fingerprints"),
    ),
    "BMI": _GuideSpec(
        "Anthropometrics and body composition",
        "Body mass index recorded as a discrete scalar.",
        "One point BMI observation, which may be imported or calculated by a source application.",
        "May be user entered, source calculated, or imported; provenance should be retained.",
        "Preserve the raw BMI and compare with temporally matched canonical height and weight when available; use BMI as a screening measure, not a diagnosis.",
        ("BMI may not reflect the same adiposity or risk across individuals and populations.",),
        ("A direct measurement of body fat or an individual diagnosis.",),
        ("apple_bmi", "who_bmi"),
    ),
    "WaistCircumference": _GuideSpec(
        "Anthropometrics and body composition",
        "Discrete waist-circumference measurement.",
        "One point length observation represented in inches by the current exporter; measurement protocol remains unknown.",
        "Usually manually entered or imported from an application.",
        "Convert inches to metres while retaining a protocol-context warning because anatomical landmark and technique are absent.",
        ("Waist measurements depend on landmark, tape placement, tension, and respiratory phase.",),
        ("A protocol-standardized clinical waist measurement when protocol metadata is absent.",),
        ("apple_waist_circumference", "who_waist", "project_unit_fingerprints"),
    ),
    "BodyFatPercentage": _GuideSpec(
        "Anthropometrics and body composition",
        "Discrete body-fat percentage estimate.",
        "One point ratio observation stored as a HealthKit fraction before conversion to percent.",
        "Usually produced by a scale, body-composition device, or application estimate.",
        "Convert fraction to percent, retain source/device context, and compare with Weight and LeanBodyMass when synchronized.",
        ("Estimation methods differ across devices and may not be directly comparable.",),
        ("A reference-method body-composition assessment.",),
        ("apple_body_fat_percentage", "apple_healthkit_units", "project_body_composition_consistency"),
    ),
    "LeanBodyMass": _GuideSpec(
        "Anthropometrics and body composition",
        "Discrete lean-body-mass estimate.",
        "One point mass observation whose raw unit may depend on source epoch.",
        "Usually produced with Weight and BodyFatPercentage by a body-composition device or application.",
        "Resolve kilograms/pounds by source epoch and use synchronized body-composition consistency as evidence without overwriting values.",
        ("Lean body mass is an estimate and may depend on the device algorithm.",),
        ("A directly measured tissue mass or a unit-resolved value when the epoch is ambiguous.",),
        ("apple_lean_body_mass", "project_body_composition_consistency"),
    ),
    "EnergyConsumed": _GuideSpec(
        "Nutrition",
        "Dietary energy amount recorded for one nutrition event.",
        "One discrete event amount, potentially in kcal or kJ depending on source convention.",
        "Often manually entered or imported from a nutrition application.",
        "Preserve distinct record IDs, source, and user-entry context; resolve kcal/kJ by source epoch before canonical conversion.",
        ("Nutrition entries may be incomplete, duplicated by synchronization, or represent foods with the same timestamp and value.",),
        ("A continuous energy-consumption rate.",),
        ("apple_nutrition",),
    ),
    "Carbohydrates": _GuideSpec(
        "Nutrition",
        "Carbohydrate amount recorded for one nutrition event.",
        "One discrete mass amount, normally represented in grams by this policy.",
        "Usually manually entered or imported from a food-tracking application.",
        "Preserve distinct UUIDs and source context; flag negative amounts without collapsing equal food entries.",
        ("Equal values at the same timestamp can represent distinct foods or synchronization duplicates.",),
        ("A continuous nutrient-intake rate.",),
        ("apple_nutrition",),
    ),
    "Protein": _GuideSpec(
        "Nutrition",
        "Protein amount recorded for one nutrition event.",
        "One discrete mass amount, normally represented in grams by this policy.",
        "Usually manually entered or imported from a food-tracking application.",
        "Preserve distinct UUIDs and source context; flag negative amounts without collapsing equal food entries.",
        ("The data may not represent complete daily intake.",),
        ("A continuous nutrient-intake rate.",),
        ("apple_nutrition",),
    ),
    "TotalFat": _GuideSpec(
        "Nutrition",
        "Total-fat amount recorded for one nutrition event.",
        "One discrete mass amount, normally represented in grams by this policy.",
        "Usually manually entered or imported from a food-tracking application.",
        "Preserve distinct UUIDs and source context; flag negative amounts without collapsing equal food entries.",
        ("The data may not represent complete daily intake.",),
        ("A continuous nutrient-intake rate.",),
        ("apple_nutrition",),
    ),
    "BloodAlcoholContent": _GuideSpec(
        "Other health data",
        "Blood-alcohol-content quantity recorded as a HealthKit percent ratio.",
        "One point value on the native fraction scale, with acquisition meaning determined by source.",
        "May be manually entered, imported, or generated by a BAC calculator application.",
        "Convert reviewed fraction encodings to percent, classify calculator estimates separately, and exclude calculator trajectories from default direct-measurement analyses.",
        ("A calculator trajectory is a model output, not a direct sensor measurement.",),
        ("A verified laboratory or breath-analyzer measurement unless provenance establishes it.",),
        ("apple_blood_alcohol_content", "apple_healthkit_units", "project_bac_sources"),
    ),
}


def _unit_summary(feature: str) -> str:
    policy = get_policy(feature, allow_fallback=False)
    parts: list[str] = []
    for unit in policy.units.measurements:
        raw = unit.raw_unit or "/".join(unit.raw_unit_candidates) or "unresolved"
        target = unit.canonical_unit or "raw-only"
        parts.append(f"{unit.measurement}: {raw} -> {target} ({unit.raw_unit_status.value})")
    return "; ".join(parts) or "No scalar unit conversion applies."


def _time_semantics(feature: str) -> str:
    policy = get_policy(feature, allow_fallback=False)
    return (
        f"{policy.semantics.event_kind.value}; {policy.semantics.measurement_kind.value}; "
        f"duration model={policy.semantics.duration_model.value}; "
        f"native resolution={policy.semantics.native_resolution}."
    )


def _build_guides() -> dict[str, FeatureGuide]:
    missing = set(COHORT_OBSERVED_FEATURES) - set(_SPECS)
    extra = set(_SPECS) - set(COHORT_OBSERVED_FEATURES)
    if missing or extra:
        raise ValueError(f"Feature-guide coverage mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    guides: dict[str, FeatureGuide] = {}
    for feature in COHORT_OBSERVED_FEATURES:
        spec = _SPECS[feature]
        policy = get_policy(feature, allow_fallback=False)
        refs = tuple(dict.fromkeys(policy.evidence_refs + spec.evidence_refs))
        unknown_refs = [ref for ref in refs if ref not in EVIDENCE]
        if unknown_refs:
            raise ValueError(f"{feature}: unknown guidance evidence refs {unknown_refs}")
        guides[feature] = FeatureGuide(
            feature=feature,
            category=spec.category,
            short_description=spec.short_description,
            one_row_means=spec.one_row_means,
            native_time_semantics=_time_semantics(feature),
            unit_summary=_unit_summary(feature),
            acquisition_summary=spec.acquisition_summary,
            curation_summary=spec.curation_summary,
            important_caveats=spec.caveats,
            not_equivalent_to=spec.not_equivalent_to,
            evidence_refs=refs,
        )
    return guides


FEATURE_GUIDES = _build_guides()


def known_guides() -> tuple[str, ...]:
    return tuple(sorted(FEATURE_GUIDES))


def get_feature_guide(feature: str) -> FeatureGuide:
    try:
        return FEATURE_GUIDES[feature]
    except KeyError as exc:
        raise KeyError(f"No feature guide for {feature!r}") from exc


def guide_payload(features: Iterable[str] | None = None) -> dict[str, object]:
    names = tuple(features) if features is not None else known_guides()
    return {
        "guidance_version": GUIDANCE_VERSION,
        "guidance_fingerprint": guidance_fingerprint(),
        "feature_count": len(names),
        "features": {name: get_feature_guide(name).as_dict() for name in names},
    }


def guidance_fingerprint() -> str:
    payload = {
        "guidance_version": GUIDANCE_VERSION,
        "guides": {name: guide.as_dict() for name, guide in sorted(FEATURE_GUIDES.items())},
        "evidence": {
            evidence_id: {
                "brief_summary": source.brief_summary,
                "limitations": source.limitations,
                "last_verified": source.last_verified,
                "citation_text": source.citation_text,
                "locator": source.locator,
            }
            for evidence_id, source in sorted(EVIDENCE.items())
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()
