"""
Wearable Data Processing and Modeling project
Evidence catalog for the curation policy registry. The catalog deliberately separates authoritative external
references from project-specific empirical evidence. The alpha release uses the references to justify policy
declarations; no clinical screening threshold is executed yet.
"""


from __future__ import annotations
from dataclasses import dataclass, replace
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
    brief_summary: str = ""
    limitations: str = ""
    last_verified: str | None = None
    citation_text: str = ""

    @property
    def is_external(self) -> bool:
        return self.locator.startswith(("https://", "http://"))


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

# Additional feature-specific HealthKit references used by the user-facing guidance layer. They refine
# explanations without changing executable policy.
_add(
    EvidenceSource(
        "apple_step_count",
        "stepCount",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/stepcount",
        ("step-count quantity semantics", "count units", "cumulative aggregation style"),
    ),
    EvidenceSource(
        "apple_active_energy",
        "activeEnergyBurned",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/activeenergyburned",
        ("active-energy quantity semantics", "energy units", "cumulative aggregation style"),
    ),
    EvidenceSource(
        "apple_basal_energy",
        "basalEnergyBurned",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/basalenergyburned",
        ("basal-energy quantity semantics", "energy units", "cumulative aggregation style"),
    ),
    EvidenceSource(
        "apple_flights_climbed",
        "flightsClimbed",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/flightsclimbed",
        ("flights-climbed quantity semantics", "count units", "cumulative aggregation style"),
    ),
    EvidenceSource(
        "apple_cycling_distance",
        "distanceCycling",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/distancecycling",
        ("cycling-distance quantity semantics", "length units", "cumulative aggregation style"),
    ),
    EvidenceSource(
        "apple_swimming_distance",
        "distanceSwimming",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/distanceswimming",
        ("swimming-distance quantity semantics", "length units", "cumulative aggregation style"),
    ),
    EvidenceSource(
        "apple_resting_heart_rate",
        "restingHeartRate",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/restingheartrate",
        ("resting-heart-rate estimate semantics", "count/time units"),
    ),
    EvidenceSource(
        "apple_respiratory_rate",
        "respiratoryRate",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/respiratoryrate",
        ("respiratory-rate quantity semantics", "count/time units", "discrete aggregation style"),
    ),
    EvidenceSource(
        "apple_oxygen_saturation",
        "oxygenSaturation",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/oxygensaturation",
        ("oxygen-saturation quantity semantics", "percent units", "discrete aggregation style"),
    ),
    EvidenceSource(
        "apple_height",
        "height",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/height",
        ("height measurement semantics", "length-unit compatibility", "discrete aggregation style"),
    ),
    EvidenceSource(
        "apple_lean_body_mass",
        "leanBodyMass",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/leanbodymass",
        ("lean-body-mass semantics", "mass-unit compatibility", "discrete aggregation style"),
    ),
    EvidenceSource(
        "apple_bmi",
        "bodyMassIndex",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/bodymassindex",
        ("BMI quantity semantics", "count-compatible scalar representation", "discrete aggregation style"),
    ),
    EvidenceSource(
        "apple_blood_alcohol_content",
        "bloodAlcoholContent",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/bloodalcoholcontent",
        ("blood-alcohol-content quantity semantics", "percent-unit compatibility"),
    ),
    EvidenceSource(
        "apple_body_temperature_location",
        "HKBodyTemperatureSensorLocation",
        "Apple Developer Documentation",
        2026,
        "official_technical_documentation",
        "https://developer.apple.com/documentation/healthkit/hkbodytemperaturesensorlocation",
        ("body-temperature measurement-site vocabulary", "measurement context"),
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
            "mass metric-to-imperial conversion fingerprints",
        ),
    ),
    EvidenceSource(
        "project_sleep_intervals",
        "Observed sleep state intervals and overlaps",
        "wearable-project",
        2026,
        "project_empirical_evidence",
        "project://analysis/sleep-intervals",
        (
            "in-bed nesting",
            "same-source detailed-stage conflicts",
            "cross-source overlap as provenance context",
            "source epochs that omit INBED",
            "avoidance of lexical tie-breaking",
        ),
    ),
)


_LAST_VERIFIED = "2026-09-14"


_EXPLICIT_SUMMARIES: dict[str, str] = {
    "apple_healthkit_units": (
        "Defines HealthKit unit construction and conversion semantics, including the 0-1 representation used by percent quantities."
    ),
    "apple_sleep_analysis": (
        "Defines INBED and detailed AWAKE/CORE/DEEP/REM sleep categories. Detailed stages may overlap INBED but are not expected to overlap one another."
    ),
    "apple_ecg": (
        "Defines an ECG as a waveform collection with voltage measurements plus high-level context such as classification, average heart rate, and sampling frequency."
    ),
    "apple_blood_glucose": (
        "Defines discrete blood-glucose measurements and explicitly permits mg/dL or mmol/L depending on region and user preference."
    ),
    "apple_activity_summary": (
        "Defines move, exercise, and stand values for one calendar day, identified by date components rather than ordinary sample start and end times."
    ),
    "apple_blood_pressure": (
        "Defines blood pressure as a correlation that keeps systolic and diastolic quantities together as one reading."
    ),
    "apple_body_temperature": (
        "Defines discrete body-temperature quantities and associated measurement-location context."
    ),
    "apple_peak_flow": (
        "Defines peak expiratory flow as a discrete maximum flow rate from a forceful exhalation using volume-per-time units."
    ),
    "aha_blood_pressure_measurement": (
        "Describes standardized blood-pressure technique and the importance of device, cuff, posture, repeated readings, and measurement setting."
    ),
    "fda_pulse_oximetry": (
        "Explains limitations of pulse-oximeter readings and the need to retain device and measurement context rather than treating one value as a diagnosis."
    ),
    "who_bmi": (
        "Defines BMI as weight in kilograms divided by squared height in metres and frames it as a screening measure rather than a direct body-composition measurement."
    ),
    "who_waist": (
        "Describes standardized waist-circumference measurement and its dependence on anatomical landmark and measurement procedure."
    ),
    "project_activity_summary_sliding_pair": (
        "The cohort shows adjacent two-item payloads in which the second item of one outer day often equals the first item of the next, leaving calendar assignment ambiguous."
    ),
    "project_body_composition_consistency": (
        "Synchronized Weight, BodyFatPercentage, and LeanBodyMass records satisfy the expected body-composition equation in the observed sample and can support unit inference."
    ),
    "project_cgm_metadata": (
        "The cohort contains CGM status, trend arrow, trend rate, device, and IANA time-zone metadata, including travel-related time-zone changes."
    ),
    "project_unit_fingerprints": (
        "Observed decimal patterns reveal deterministic conversions for selected sources, but source-epoch review remains necessary before applying them globally."
    ),
}


def _default_limitations(source: EvidenceSource) -> str:
    if source.kind == "project_empirical_evidence":
        return (
            "Project-specific empirical evidence from the supplied samples or processed cohort; "
            "it may not generalize to other exporters, HealthKit versions, or populations."
        )
    if source.kind in {"clinical_guidance", "clinical_measurement_standard", "technical_standard"}:
        return (
            "Supports measurement or interpretation context; it is not a row-level diagnosis, "
            "a universal exclusion threshold, or evidence that a wearable estimate equals a clinical test."
        )
    return (
        "Defines HealthKit or platform semantics but does not, by itself, prove which unit or "
        "serialization convention this upstream exporter selected."
    )


def _citation_text(source: EvidenceSource) -> str:
    # Apple and other living web documentation is continuously updated; the separately recorded last_verified
    # date is more precise than presenting the current year as a publication year.
    year = "" if source.kind == "official_technical_documentation" else (
        f" ({source.year})" if source.year is not None else ""
    )
    return f"{source.organization}. {source.title}{year}."


def _enrich_catalog() -> None:
    for evidence_id, source in tuple(EVIDENCE.items()):
        summary = source.brief_summary or _EXPLICIT_SUMMARIES.get(
            evidence_id, "; ".join(source.supports).rstrip(".") + "."
        )
        EVIDENCE[evidence_id] = replace(
            source,
            brief_summary=summary,
            limitations=source.limitations or _default_limitations(source),
            last_verified=source.last_verified or _LAST_VERIFIED,
            citation_text=source.citation_text or _citation_text(source),
        )


_enrich_catalog()


def get_evidence(evidence_id: str) -> EvidenceSource:
    return EVIDENCE[evidence_id]


def known_evidence_ids() -> tuple[str, ...]:
    return tuple(sorted(EVIDENCE))


def evidence_subset(ids: Iterable[str]) -> tuple[EvidenceSource, ...]:
    return tuple(EVIDENCE[evidence_id] for evidence_id in ids)


def evidence_by_kind(kind: str) -> tuple[EvidenceSource, ...]:
    return tuple(source for _, source in sorted(EVIDENCE.items()) if source.kind == kind)
