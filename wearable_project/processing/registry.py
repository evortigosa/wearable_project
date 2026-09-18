"""
Wearable Data Processing and Modeling project
Explicit feature-family and unit policies for native Apple HealthKit rows.
"""


from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


REGISTRY_VERSION = "2026-09-native-processing"
SPECS: dict[str, FeatureSpec] = {}


class FeatureFamily(str, Enum):
    INTERVAL_TOTAL = "interval_total"
    INTENSIVE_INTERVAL = "intensive_interval"
    POINT_SCALAR = "point_scalar"
    POINT_VECTOR = "point_vector"
    STATE_INTERVAL = "state_interval"
    LONG_SUMMARY_INTERVAL = "long_summary_interval"
    DAILY_SUMMARY = "daily_summary"
    WAVEFORM = "waveform"
    DURATION_EVENT = "duration_event"
    GENERIC = "generic"


class DedupStrategy(str, Enum):
    RECORD_ID_THEN_CONTENT = "record_id_then_content"
    RECORD_ID_ONLY = "record_id_only"
    INTERVAL_REVISION = "interval_revision"
    SUMMARY_OCCURRENCE = "summary_occurrence"


@dataclass(frozen=True, slots=True)
class UnitPolicy:
    raw_unit: str
    canonical_unit: str
    scale: float = 1.0
    offset: float = 0.0
    status: str = "source_convention"
    evidence: str | None = None

    def convert(self, value: float) -> float:
        return value * self.scale + self.offset


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    family: FeatureFamily
    measurement_columns: tuple[str, ...]
    dedup_strategy: DedupStrategy
    unit_policy: UnitPolicy | None = None
    preserve_content_duplicates: bool = False
    resolve_boundary_revisions: bool = False
    provisional: bool = False


def U(raw: str, canonical: str, scale: float = 1.0, *, status: str = "source_convention", evidence: str | None = None) -> UnitPolicy:
    return UnitPolicy(raw, canonical, scale, 0.0, status, evidence)


def add(*specs: FeatureSpec) -> None:
    for spec in specs:
        SPECS[spec.name] = spec


# Additive interval aggregates. These retain their original intervals; no fixed-window allocation occurs in native processing.
add(
    FeatureSpec("StepCount", FeatureFamily.INTERVAL_TOTAL, ("value",), DedupStrategy.INTERVAL_REVISION, U("count", "count"), resolve_boundary_revisions=True),
    FeatureSpec("DistanceWalkingRunning", FeatureFamily.INTERVAL_TOTAL, ("value",), DedupStrategy.INTERVAL_REVISION, U("m", "m", status="validated_source_convention"), resolve_boundary_revisions=True),
    FeatureSpec("FlightsClimbed", FeatureFamily.INTERVAL_TOTAL, ("value",), DedupStrategy.INTERVAL_REVISION, U("count", "count"), resolve_boundary_revisions=True),
    FeatureSpec("BasalEnergyBurned", FeatureFamily.INTERVAL_TOTAL, ("value",), DedupStrategy.INTERVAL_REVISION, U("kcal", "kcal", status="feature_convention"), resolve_boundary_revisions=True),
    FeatureSpec("ActiveEnergyBurned", FeatureFamily.INTERVAL_TOTAL, ("value",), DedupStrategy.INTERVAL_REVISION, U("kcal", "kcal", status="feature_convention"), resolve_boundary_revisions=True),
    FeatureSpec("DailyDistanceSwimming", FeatureFamily.INTERVAL_TOTAL, ("value",), DedupStrategy.INTERVAL_REVISION, U("m", "m", status="inferred_high_confidence", evidence="round pool-session totals and paired daily records"), resolve_boundary_revisions=True),
    FeatureSpec("DailyDistanceCycling", FeatureFamily.INTERVAL_TOTAL, ("value",), DedupStrategy.INTERVAL_REVISION, None, resolve_boundary_revisions=True, provisional=True),
)

# Intensive and point observations.
add(
    FeatureSpec("HeartRate", FeatureFamily.INTENSIVE_INTERVAL, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("beats/min", "beats/min")),
    FeatureSpec("HeartRateVariability", FeatureFamily.INTENSIVE_INTERVAL, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, None, provisional=True),
    FeatureSpec("RestingHeartRate", FeatureFamily.LONG_SUMMARY_INTERVAL, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("beats/min", "beats/min")),
    FeatureSpec("WalkingHeartRate", FeatureFamily.LONG_SUMMARY_INTERVAL, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("beats/min", "beats/min")),
    FeatureSpec("Vo2Max", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("mL/(kg*min)", "mL/(kg*min)", status="feature_convention")),
    FeatureSpec("OxygenSaturation", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("fraction", "%", 100.0, status="inferred_high_confidence", evidence="observed 0.82-1.00 encoding")),
    FeatureSpec("RespiratoryRate", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("breaths/min", "breaths/min", status="feature_convention")),
    FeatureSpec("BloodGlucose", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, None, provisional=True),
    FeatureSpec("Weight", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, None, provisional=True),
    FeatureSpec("Height", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, None, provisional=True),
    FeatureSpec("BMI", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("kg/m2", "kg/m2", status="feature_convention")),
    FeatureSpec("BodyFatPercentage", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("fraction", "%", 100.0, status="validated_cross_feature", evidence="Weight/LeanBodyMass consistency")),
    FeatureSpec("LeanBodyMass", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, None, provisional=True),
    FeatureSpec("BodyTemperature", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("Cel", "Cel", status="inferred_high_confidence", evidence="observed values and Fahrenheit-conversion fingerprints")),
    FeatureSpec("BasalBodyTemperature", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("Cel", "Cel", status="inferred_high_confidence", evidence="observed basal-temperature values and manual entry")),
    FeatureSpec("PeakFlow", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("L/min", "L/min", status="inferred_provisional", evidence="observed 400-500 manual entries"), provisional=True),
    FeatureSpec("PeakExpiratoryFlow", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("L/min", "L/min", status="inferred_high_confidence", evidence="observed 400-500 manual peak-flow entries")),
    FeatureSpec("WaistCircumference", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("in", "m", 0.0254, status="inferred_high_confidence", evidence="exact centimetre-to-inch conversion fingerprints")),
    FeatureSpec("BloodAlcoholContent", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("fraction", "%", 100.0, status="inferred_high_confidence", evidence="calculator/manual source-scale evidence")),
)

# Additional Apple feature families observed in the supplied cohort samples.
add(
    FeatureSpec("BodyWaterMass", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, None, provisional=True),
    FeatureSpec("ElectrodermalActivity", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("uS", "uS", status="inferred_provisional", evidence="feature semantics and observed Garmin range"), provisional=True),
    FeatureSpec("EnvironmentalAudioExposure", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("dBASPL", "dBASPL", status="feature_convention")),
    FeatureSpec("HeadphoneAudioExposure", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("dBASPL", "dBASPL", status="feature_convention")),
    FeatureSpec("HeartRateRecoveryOneMinute", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("beats/min", "beats/min", status="feature_convention")),
    FeatureSpec("RunningGroundContactTime", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("s", "s", status="inferred_high_confidence", evidence="observed 0.28-0.30 values")),
    FeatureSpec("RunningPower", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("W", "W", status="feature_convention")),
    FeatureSpec("RunningSpeed", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("m/s", "m/s", status="feature_convention")),
    FeatureSpec("RunningStrideLength", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("m", "m", status="feature_convention")),
    FeatureSpec("RunningVerticalOscillation", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("m", "m", status="feature_convention")),
    FeatureSpec("SixMinuteWalkTestDistance", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("m", "m", status="feature_convention")),
    FeatureSpec("StairAscentSpeed", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("m/s", "m/s", status="feature_convention")),
    FeatureSpec("StairDescentSpeed", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("m/s", "m/s", status="feature_convention")),
    FeatureSpec("TimeInDaylight", FeatureFamily.INTERVAL_TOTAL, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("min", "min", status="validated_sample_relationship", evidence="value equals native interval duration in minutes")),
    FeatureSpec("UVExposure", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("UV index", "UV index", status="feature_convention")),
    FeatureSpec("WalkingAsymmetryPercentage", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("fraction", "%", 100.0, status="feature_convention")),
    FeatureSpec("WalkingDoubleSupportPercentage", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("fraction", "%", 100.0, status="feature_convention")),
    FeatureSpec("WalkingSpeed", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("m/s", "m/s", status="feature_convention")),
    FeatureSpec("WalkingStepLength", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, U("m", "m", status="feature_convention")),
    FeatureSpec("WeightLossGoal", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT, None, provisional=True),
)

add(
    FeatureSpec("Sleep", FeatureFamily.STATE_INTERVAL, ("value",), DedupStrategy.RECORD_ID_THEN_CONTENT),
    FeatureSpec("BloodPressure", FeatureFamily.POINT_VECTOR, ("blood_pressure_systolic_value", "blood_pressure_diastolic_value"), DedupStrategy.RECORD_ID_THEN_CONTENT, U("mmHg", "mmHg", status="feature_convention")),
    FeatureSpec("Electrocardiogram", FeatureFamily.WAVEFORM, ("average_heart_rate", "sampling_frequency", "voltage_measurements"), DedupStrategy.RECORD_ID_THEN_CONTENT),
    FeatureSpec("ActivitySummary", FeatureFamily.DAILY_SUMMARY, ("apple_stand_hours", "apple_exercise_time", "active_energy_burned", "apple_stand_hours_goal", "apple_exercise_time_goal", "active_energy_burned_goal"), DedupStrategy.SUMMARY_OCCURRENCE),
    FeatureSpec("Mindful", FeatureFamily.DURATION_EVENT, tuple(), DedupStrategy.RECORD_ID_THEN_CONTENT),
)

# Nutrition entries may contain two legitimate identical food items at the same timestamp. Do not content-deduplicate
# different record IDs.
add(
    FeatureSpec("EnergyConsumed", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_ONLY, U("kcal", "kcal", status="feature_convention"), preserve_content_duplicates=True),
    FeatureSpec("Carbohydrates", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_ONLY, U("g", "g", status="feature_convention"), preserve_content_duplicates=True),
    FeatureSpec("Protein", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_ONLY, U("g", "g", status="feature_convention"), preserve_content_duplicates=True),
    FeatureSpec("TotalFat", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_ONLY, U("g", "g", status="feature_convention"), preserve_content_duplicates=True),
    FeatureSpec("DietaryCaffeine", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_ONLY, None, preserve_content_duplicates=True, provisional=True),
    FeatureSpec("DietaryWater", FeatureFamily.POINT_SCALAR, ("value",), DedupStrategy.RECORD_ID_ONLY, None, preserve_content_duplicates=True, provisional=True),
)


def get_feature_spec(name: str, columns: Iterable[str] = ()) -> FeatureSpec:
    if name in SPECS:
        return SPECS[name]
    columns = set(columns)
    if {"blood_pressure_systolic_value", "blood_pressure_diastolic_value"}.issubset(columns):
        family, measures = FeatureFamily.POINT_VECTOR, ("blood_pressure_systolic_value", "blood_pressure_diastolic_value")
    elif {"average_heart_rate", "sampling_frequency", "voltage_measurements"}.issubset(columns):
        family, measures = FeatureFamily.WAVEFORM, ("average_heart_rate", "sampling_frequency", "voltage_measurements")
    elif "value" in columns:
        family, measures = FeatureFamily.GENERIC, ("value",)
    elif {"start_date_raw", "end_date_raw"}.issubset(columns):
        family, measures = FeatureFamily.DURATION_EVENT, tuple()
    else:
        family, measures = FeatureFamily.GENERIC, tuple()
    return FeatureSpec(name, family, measures, DedupStrategy.RECORD_ID_THEN_CONTENT, provisional=True)


def is_registered_feature(name: str) -> bool:
    """Return whether a feature has an explicit reviewed registry policy."""
    return name in SPECS


def known_features() -> tuple[str, ...]:
    return tuple(sorted(SPECS))
