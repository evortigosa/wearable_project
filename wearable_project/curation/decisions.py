"""
Wearable Data Processing and Modeling project
Human-reviewed calibration decisions derived from the full-cohort audit. This module records decisions; it does
not curate participant data. Decisions are deliberately separated from the evidence-generating audit so that an
audit can never silently mutate executable feature policy.
"""


from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Mapping
from wearable_project.curation.models import EvidenceGrade, PolicyExecutionMode, PolicyMaturity


DECISION_SET_VERSION = "0.2.0a1.2-decisions-1"
DECISION_BASIS = "full-cohort-audit-2026-09-14"


@dataclass(frozen=True, slots=True)
class CalibrationDecision:
    feature: str
    maturity: PolicyMaturity
    evidence_grade: EvidenceGrade
    execution_mode: PolicyExecutionMode
    decision: str
    unit_decision: str
    source_scope: str
    safe_fallback: str
    required_before_engine: tuple[str, ...] = ()
    permanent_caveats: tuple[str, ...] = ()
    evidence_artifacts: tuple[str, ...] = ()
    approved_for_policy_registry: bool = True

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["maturity"] = self.maturity.value
        payload["evidence_grade"] = self.evidence_grade.value
        payload["execution_mode"] = self.execution_mode.value
        return payload

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


def _decision(
    feature: str, maturity: PolicyMaturity, grade: EvidenceGrade, mode: PolicyExecutionMode, decision: str,
    unit_decision: str, source_scope: str, fallback: str, *, required: tuple[str, ...] = (), caveats: tuple[str, ...] = (),
) -> CalibrationDecision:
    return CalibrationDecision(
        feature=feature,
        maturity=maturity,
        evidence_grade=grade,
        execution_mode=mode,
        decision=decision,
        unit_decision=unit_decision,
        source_scope=source_scope,
        safe_fallback=fallback,
        required_before_engine=required,
        permanent_caveats=caveats,
        evidence_artifacts=(
            "policy_audit_summary.json",
            "source_context_groups.csv",
            "source_context_epochs.csv",
            "unit_candidate_distributions.csv",
            "temporal_scale_windows.csv",
            "cross_feature_consistency.csv",
            "nutrition_energy_consistency.csv",
            "activity_summary_alignment.csv",
        ),
    )


POLICY_CALIBRATION_DECISIONS: Mapping[str, CalibrationDecision] = {
    "ActivitySummary": _decision(
        "ActivitySummary",
        PolicyMaturity.PROVISIONAL,
        EvidenceGrade.D_UNRESOLVED,
        PolicyExecutionMode.BLOCK_DERIVATION,
        "The cohort-wide sliding-pair pattern is confirmed, but payload position cannot be assigned to one calendar date from native CSV evidence alone.",
        "No daily-date derivation is approved.",
        "all_registered_sources",
        "Preserve every payload item and payload_index; keep summary-date ambiguity explicit and exclude from default daily-summary analysis.",
        required=("Review exporter HKActivitySummary date-component/query code.",),
    ),
    "BloodAlcoholContent": _decision(
        "BloodAlcoholContent",
        PolicyMaturity.REVIEWED,
        EvidenceGrade.B_CONVERGENT,
        PolicyExecutionMode.SOURCE_SPECIFIC_EXECUTION,
        "HealthKit fraction encoding is consistent across the observed calculator and manual-entry regimes; acquisition meaning remains source-specific.",
        "Convert fraction to percent for all registered BAC rows.",
        "source_id_and_user_entry_specific",
        "Unknown sources retain the converted percentage but remain review-status with unknown acquisition method.",
        caveats=("Calculator trajectories are modeled estimates and are excluded from default direct-measurement analyses.",),
    ),
    "BloodGlucose": _decision(
        "BloodGlucose",
        PolicyMaturity.PROVISIONAL,
        EvidenceGrade.B_CONVERGENT,
        PolicyExecutionMode.SOURCE_SPECIFIC_EXECUTION,
        "The current cohort is strongly mmol/L-like, but exporter unit-selection code or an equivalent direct convention should be confirmed before global execution.",
        "Resolve participant/source epochs; current cohort evidence favours mmol/L.",
        "participant_feature_source_epoch",
        "Retain raw glucose and omit canonical values for unresolved epochs.",
        required=("Confirm exporter HKUnit selection for blood glucose.",),
    ),
    "BodyTemperature": _decision(
        "BodyTemperature",
        PolicyMaturity.REVIEWED,
        EvidenceGrade.B_CONVERGENT,
        PolicyExecutionMode.REVIEWED_EXECUTION,
        "All observed source contexts use a Celsius-like scale; Fahrenheit interpretation is incompatible with the cohort distribution.",
        "Treat exported raw values as degrees Celsius.",
        "all_registered_sources",
        "Preserve raw values and source/sensor context; do not apply one universal clinical threshold.",
        caveats=("Measurement site is often absent and peripheral/skin temperature is not equivalent to core temperature.",),
    ),
    "DailyDistanceCycling": _decision(
        "DailyDistanceCycling",
        PolicyMaturity.REVIEWED,
        EvidenceGrade.B_CONVERGENT,
        PolicyExecutionMode.REVIEWED_EXECUTION,
        "Metres are overwhelmingly supported; kilometre and mile interpretations produce incompatible magnitudes, while version-1 values contain exact mile-to-metre conversion fingerprints.",
        "Treat exported raw values as metres.",
        "all_registered_sources",
        "Preserve raw interval totals and flag nonpositive duration or extreme implied speed without clipping.",
    ),
    "EnergyConsumed": _decision(
        "EnergyConsumed",
        PolicyMaturity.PROVISIONAL,
        EvidenceGrade.C_SUGGESTIVE,
        PolicyExecutionMode.CONSERVATIVE_ANNOTATION,
        "The cohort contains heterogeneous source regimes and extreme values; kcal versus kJ cannot be selected safely from marginal distributions alone.",
        "No automatic canonical energy conversion is approved.",
        "participant_feature_source_epoch",
        "Retain raw energy, emit unit candidates, and omit canonical kcal when unresolved.",
        required=("Review exporter energy-unit selection.", "Review nutrition energy-versus-macronutrient audit."),
    ),
    "HeartRateVariability": _decision(
        "HeartRateVariability",
        PolicyMaturity.REVIEWED,
        EvidenceGrade.B_CONVERGENT,
        PolicyExecutionMode.REVIEWED_EXECUTION,
        "All observed source/version/algorithm contexts are seconds-like and convert coherently to millisecond SDNN values.",
        "Convert exported seconds to milliseconds.",
        "all_registered_sources",
        "Preserve raw values and algorithm/source context; flag zeros and extreme values non-destructively.",
    ),
    "Height": _decision(
        "Height",
        PolicyMaturity.PROVISIONAL,
        EvidenceGrade.C_SUGGESTIVE,
        PolicyExecutionMode.SOURCE_SPECIFIC_EXECUTION,
        "The dominant regime is inches, but metre-, centimetre-, and malformed regimes exist.",
        "Resolve metres/centimetres/inches per bounded participant/source epoch.",
        "participant_feature_source_epoch",
        "Retain raw height and omit canonical metres for ambiguous epochs.",
        required=("Use temporally bounded BMI consistency and continuity evidence.",),
    ),
    "LeanBodyMass": _decision(
        "LeanBodyMass",
        PolicyMaturity.PROVISIONAL,
        EvidenceGrade.C_SUGGESTIVE,
        PolicyExecutionMode.SOURCE_SPECIFIC_EXECUTION,
        "Most values are pound-like; body-composition consistency verifies same-unit structure but cannot by itself distinguish kilograms from pounds.",
        "Resolve kilograms/pounds jointly with Weight within bounded participant/source epochs.",
        "participant_feature_source_epoch",
        "Retain raw lean mass and omit canonical kilograms for ambiguous epochs.",
        required=("Use bounded source epochs and temporally matched Weight/body-fat evidence.",),
    ),
    "PeakFlow": _decision(
        "PeakFlow",
        PolicyMaturity.PROVISIONAL,
        EvidenceGrade.C_SUGGESTIVE,
        PolicyExecutionMode.CONSERVATIVE_ANNOTATION,
        "Only three manual records are available; L/min is plausible but not established from cohort data alone.",
        "No automatic unit conversion is approved without exporter confirmation.",
        "all_registered_sources",
        "Preserve points and candidate L/min annotation; do not sessionize or choose best attempts.",
        required=("Confirm exporter HKUnit selection for peak expiratory flow.",),
    ),
    "WaistCircumference": _decision(
        "WaistCircumference",
        PolicyMaturity.REVIEWED,
        EvidenceGrade.B_CONVERGENT,
        PolicyExecutionMode.REVIEWED_EXECUTION,
        "All observed values and conversion fingerprints support inches as the exported unit.",
        "Convert inches to metres.",
        "all_registered_sources",
        "Preserve raw values and retain a measurement-protocol warning.",
        caveats=("Anatomical landmark, tape placement, respiratory phase, and measurement protocol are absent.",),
    ),
    "Weight": _decision(
        "Weight",
        PolicyMaturity.PROVISIONAL,
        EvidenceGrade.C_SUGGESTIVE,
        PolicyExecutionMode.SOURCE_SPECIFIC_EXECUTION,
        "The dominant regime is pounds, but valid kilogram and malformed regimes exist.",
        "Resolve kilograms/pounds per bounded participant/source epoch.",
        "participant_feature_source_epoch",
        "Retain raw weight and omit canonical kilograms for ambiguous epochs.",
        required=("Use temporally bounded BMI and body-composition evidence plus continuity/source context.",),
    ),
}


def known_decisions() -> tuple[str, ...]:
    return tuple(sorted(POLICY_CALIBRATION_DECISIONS))


def get_decision(feature: str) -> CalibrationDecision:
    try:
        return POLICY_CALIBRATION_DECISIONS[feature]
    except KeyError as exc:
        raise KeyError(f"No calibration decision for {feature!r}") from exc


def decisions_payload() -> dict[str, Any]:
    return {
        "decision_set_version": DECISION_SET_VERSION,
        "decision_basis": DECISION_BASIS,
        "decision_fingerprint": decisions_fingerprint(),
        "features": {
            name: decision.as_dict() for name, decision in sorted(POLICY_CALIBRATION_DECISIONS.items())
        },
    }


def decisions_fingerprint() -> str:
    payload = {
        "decision_set_version": DECISION_SET_VERSION,
        "decision_basis": DECISION_BASIS,
        "features": {
            name: decision.as_dict() for name, decision in sorted(POLICY_CALIBRATION_DECISIONS.items())
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def validate_decisions() -> tuple[str, ...]:
    errors: list[str] = []
    expected = {
        "ActivitySummary", "BloodAlcoholContent", "BloodGlucose", "BodyTemperature",
        "DailyDistanceCycling", "EnergyConsumed", "HeartRateVariability", "Height",
        "LeanBodyMass", "PeakFlow", "WaistCircumference", "Weight",
    }
    if set(POLICY_CALIBRATION_DECISIONS) != expected:
        errors.append("Calibration decisions must cover exactly the twelve audited policy features")
    for feature, decision in POLICY_CALIBRATION_DECISIONS.items():
        if feature != decision.feature:
            errors.append(f"{feature}: decision feature mismatch")
        if decision.evidence_grade is EvidenceGrade.D_UNRESOLVED and decision.execution_mode is not PolicyExecutionMode.BLOCK_DERIVATION:
            errors.append(f"{feature}: D_unresolved must block derivation")
        if decision.execution_mode is PolicyExecutionMode.SOURCE_SPECIFIC_EXECUTION and not decision.source_scope:
            errors.append(f"{feature}: source-specific execution requires source scope")
        if not decision.safe_fallback:
            errors.append(f"{feature}: safe fallback is required")
    return tuple(errors)
