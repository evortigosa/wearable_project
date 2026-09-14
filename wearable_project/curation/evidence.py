"""
Wearable Data Processing and Modeling project
Evidence catalog for the curation policy registry. The catalog deliberately separates authoritative external
references from project-specific empirical evidence.  The alpha release uses the references to justify
policy declarations; no clinical screening threshold is executed yet.
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    evidence_id: str
    title: str
    organization: str
    year: int | None
    kind: str
    locator: str
    supports: tuple[str, ...]
    note: str = ""


EVIDENCE: dict[str, EvidenceSource] = {}


def _add(*sources: EvidenceSource) -> None:
    for source in sources:
        if source.evidence_id in EVIDENCE:
            raise ValueError(f"Duplicate evidence ID: {source.evidence_id}")
        EVIDENCE[source.evidence_id] = source


_add(
    EvidenceSource(
        "apple_healthkit_quantity_types",
        "HKQuantityTypeIdentifier",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier",
        ("HealthKit feature definitions", "quantity type semantics", "aggregation families"),
    ),
    EvidenceSource(
        "apple_healthkit_units",
        "HKUnit and unit-string semantics",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkunit/init(from:)-9qont",
        (
            "SI and non-SI HealthKit units",
            "percent values stored on a 0.0-1.0 scale",
            "count values stored as doubles",
        ),
    ),
    EvidenceSource(
        "apple_distance_walking_running",
        "distanceWalkingRunning",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/distancewalkingrunning",
        ("cumulative distance semantics", "length units", "possible sample coalescing"),
    ),
    EvidenceSource(
        "apple_heart_rate",
        "heartRate",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/heartrate",
        ("discrete heart-rate samples", "count/time units", "motion-context metadata"),
    ),
    EvidenceSource(
        "apple_hrv_sdnn",
        "heartRateVariabilitySDNN",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/heartratevariabilitysdnn",
        ("SDNN definition", "time units", "algorithm-version metadata"),
    ),
    EvidenceSource(
        "apple_walking_heart_rate",
        "walkingHeartRateAverage",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/walkingheartrateaverage",
        (
            "walking heart-rate estimate semantics",
            "HealthKit replacement of improving current/previous-day estimates",
        ),
    ),
    EvidenceSource(
        "apple_vo2max",
        "vo2Max",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/vo2max",
        (
            "mL/kg/min units",
            "estimated versus clinical-test provenance",
            "VO2Max test-type metadata",
        ),
    ),
    EvidenceSource(
        "apple_sleep_analysis",
        "HKCategoryValueSleepAnalysis",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkcategoryvaluesleepanalysis",
        (
            "valid sleep-state vocabulary",
            "overlap of in-bed with detailed sleep states",
            "non-overlap expectation among detailed stages",
        ),
    ),
    EvidenceSource(
        "apple_ecg",
        "HKElectrocardiogram",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkelectrocardiogram",
        (
            "ECG event semantics",
            "classification and average-heart-rate metadata",
            "voltage measurement series",
        ),
    ),
    EvidenceSource(
        "apple_ecg_sampling_frequency",
        "HKElectrocardiogram.samplingFrequency",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkelectrocardiogram/samplingfrequency",
        ("sampling frequency in hertz",),
    ),
    EvidenceSource(
        "apple_blood_glucose",
        "bloodGlucose",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/bloodglucose",
        (
            "mg/dL and mmol/L alternatives",
            "discrete measurement semantics",
            "meal-time metadata",
        ),
    ),
    EvidenceSource(
        "apple_blood_pressure",
        "bloodPressure",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkcorrelationtypeidentifier/bloodpressure",
        ("systolic and diastolic values form one correlation/reading",),
    ),
    EvidenceSource(
        "apple_body_mass",
        "bodyMass",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/bodymass",
        ("discrete mass measurement semantics", "mass-unit compatibility"),
    ),
    EvidenceSource(
        "apple_body_fat_percentage",
        "bodyFatPercentage",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/bodyfatpercentage",
        ("percentage-unit semantics", "discrete measurement semantics"),
    ),
    EvidenceSource(
        "apple_body_temperature",
        "bodyTemperature",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/bodytemperature",
        ("temperature-unit semantics", "sensor-location metadata"),
    ),
    EvidenceSource(
        "apple_waist_circumference",
        "waistCircumference",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/waistcircumference",
        ("length-unit semantics", "discrete measurement semantics"),
    ),
    EvidenceSource(
        "apple_peak_flow",
        "peakExpiratoryFlowRate",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/peakexpiratoryflowrate",
        ("maximum forceful-expiration flow", "volume/time units"),
    ),
    EvidenceSource(
        "apple_activity_summary",
        "HKActivitySummary",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkactivitysummary",
        ("daily move, exercise and stand summary", "date-component semantics"),
    ),
    EvidenceSource(
        "apple_mindful_session",
        "mindfulSession",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkcategorytypeidentifier/mindfulsession",
        ("mindful-session category sample", "duration represented by start/end"),
    ),
    EvidenceSource(
        "apple_nutrition",
        "Nutrition Type Identifiers",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/nutrition-type-identifiers",
        (
            "macronutrient mass semantics",
            "energy in calories or kilojoules",
            "food correlations and hierarchy",
        ),
    ),
    EvidenceSource(
        "who_bmi",
        "BMI definition and adult screening categories",
        "World Health Organization",
        1995,
        "clinical_public_health_guidance",
        "https://apps.who.int/nutrition/landscape/help.aspx?helpid=420",
        ("BMI formula", "BMI is a population/clinical screening measure"),
    ),
    EvidenceSource(
        "who_waist",
        "Waist Circumference and Waist-Hip Ratio: Report of a WHO Expert Consultation",
        "World Health Organization",
        2011,
        "clinical_measurement_guidance",
        "https://apps.who.int/iris/handle/10665/44583",
        ("waist measurement protocol dependence", "anatomical landmark context"),
    ),
    EvidenceSource(
        "aha_blood_pressure_measurement",
        "Measurement of Blood Pressure in Humans: A Scientific Statement",
        "American Heart Association",
        2019,
        "clinical_measurement_guidance",
        "https://professional.heart.org/en/guidelines-statements/measurement-of-blood-pressure-in-humans-a-scientific-statement-from-thehyp0000000000000087",
        ("paired blood-pressure measurement", "measurement technique and repeated-reading context"),
    ),
    EvidenceSource(
        "fda_pulse_oximetry",
        "Pulse Oximeters",
        "U.S. Food and Drug Administration",
        2025,
        "regulatory_safety_guidance",
        "https://www.fda.gov/medical-devices/products-and-medical-procedures/pulse-oximeters",
        (
            "pulse-oximetry limitations",
            "source/device context",
            "possible accuracy differences by skin pigmentation and other factors",
        ),
    ),
    EvidenceSource(
        "ada_cgm_2026",
        "Standards of Care in Diabetes—2026: Glycemic Goals and CGM Metrics",
        "American Diabetes Association",
        2026,
        "clinical_guideline",
        "https://diabetesjournals.org/care/article/49/Supplement_1/S132/163927/6-Glycemic-Goals-Hypoglycemia-and-Hyperglycemic",
        ("CGM context", "time-in-range metrics", "unit-equivalent glucose thresholds"),
    ),
    EvidenceSource(
        "hrv_task_force_1996",
        "Heart rate variability: standards of measurement, physiological interpretation and clinical use",
        "ESC/NASPE Task Force",
        1996,
        "clinical_measurement_standard",
        "https://pubmed.ncbi.nlm.nih.gov/8737210/",
        ("HRV measurement standards", "SDNN interpretation context"),
    ),
    EvidenceSource(
        "ats_ers_spirometry_2019",
        "Standardization of Spirometry 2019 Update",
        "American Thoracic Society and European Respiratory Society",
        2019,
        "technical_standard",
        "https://pubmed.ncbi.nlm.nih.gov/31613151/",
        ("forced-expiratory measurement quality", "repeatability and effort context"),
    ),
)

# Project-specific evidence is intentionally marked as empirical rather than external clinical authority.
_add(
    EvidenceSource(
        "project_full_cohort_m1",
        "Native processing full-cohort acceptance report",
        "wearable-project",
        2026,
        "project_empirical_evidence",
        "project:///net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/third-party/EV_report_files/full-cohort-acceptance",
        (
            "33 observed Apple HealthKit feature names",
            "feature prevalence and row counts",
            "duplicate/revision/conflict patterns",
        ),
        "The report describes the processed 2024-12 cumulative cohort snapshot.",
    ),
    EvidenceSource(
        "project_interval_boundary_revisions",
        "Observed Apple interval-boundary revisions",
        "wearable-project",
        2026,
        "project_empirical_evidence",
        "project://analysis/apple-boundary-revisions",
        ("partial versus completed interval copies", "same-interval conflict preservation"),
    ),
    EvidenceSource(
        "project_body_composition_consistency",
        "Observed Weight, BodyFatPercentage and LeanBodyMass consistency",
        "wearable-project",
        2026,
        "project_empirical_evidence",
        "project://analysis/body-composition-consistency",
        ("body-fat fraction encoding", "cross-feature unit evidence"),
    ),
    EvidenceSource(
        "project_activity_summary_sliding_pair",
        "Observed ActivitySummary sliding-pair payload structure",
        "wearable-project",
        2026,
        "project_empirical_evidence",
        "project://analysis/activity-summary-sliding-pair",
        ("date-assignment ambiguity", "preservation of payload items"),
    ),
    EvidenceSource(
        "project_bac_sources",
        "Observed BloodAlcoholContent source regimes",
        "wearable-project",
        2026,
        "project_empirical_evidence",
        "project://analysis/bac-source-regimes",
        ("calculator trajectories", "manual Apple Health entries", "fraction encoding"),
    ),
    EvidenceSource(
        "project_cgm_metadata",
        "Observed CGM status, trend, device and IANA time-zone metadata",
        "wearable-project",
        2026,
        "project_empirical_evidence",
        "project://analysis/cgm-metadata",
        ("CGM acquisition context", "travel/time-zone epochs", "trend metadata"),
    ),
    EvidenceSource(
        "project_unit_fingerprints",
        "Observed unit-conversion fingerprints in Apple exports",
        "wearable-project",
        2026,
        "project_empirical_evidence",
        "project://analysis/unit-fingerprints",
        (
            "waist centimetre-to-inch fingerprints",
            "temperature Fahrenheit-to-Celsius fingerprints",
            "distance source conventions",
            "HRV seconds-to-milliseconds fingerprint",
        ),
    ),
    EvidenceSource(
        "project_sleep_intervals",
        "Observed sleep state intervals and overlaps",
        "wearable-project",
        2026,
        "project_empirical_evidence",
        "project://analysis/sleep-intervals",
        ("in-bed nesting", "cross-state overlap", "avoidance of lexical tie-breaking"),
    ),
)


def get_evidence(evidence_id: str) -> EvidenceSource:
    return EVIDENCE[evidence_id]


def known_evidence_ids() -> tuple[str, ...]:
    return tuple(sorted(EVIDENCE))


def evidence_subset(ids: Iterable[str]) -> tuple[EvidenceSource, ...]:
    return tuple(EVIDENCE[evidence_id] for evidence_id in ids)
