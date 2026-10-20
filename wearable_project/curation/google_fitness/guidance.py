"""
Wearable Data Processing and Modeling project
Human-readable guidance for every Google Fitness curation policy. Guidance is generated from the source-specific
policy registry but retains feature-family explanations so downstream users can understand what a native row
means, what curation may add, and which interpretations remain blocked.
"""


from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping
from .evidence import EVIDENCE
from .models import EventKind, PolicyExecutionMode, to_primitive
from .registry import CURATION_POLICIES


EXPECTED_GUIDANCE_FINGERPRINT = "32657a51986d45223b884864c289797ada4dda850e25a17baf833ddb3ee37a57"


@dataclass(frozen=True, slots=True)
class FeatureGuide:
    feature: str
    title: str
    row_meaning: str
    recommended_use: str
    default_projection: str
    unit_interpretation: str
    temporal_interpretation: str
    provenance_interpretation: str
    limitations: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return to_primitive(self)


def _unit_summary(feature: str) -> str:
    policy = CURATION_POLICIES[feature]
    if not policy.measurements:
        return "No canonical scalar measurement is defined by this policy."
    parts: list[str] = []
    for measurement in policy.measurements:
        version_parts: list[str] = []
        for unit in measurement.unit_by_version:
            canonical = unit.canonical_unit or "unresolved"
            version_parts.append(
                f"versions {','.join(unit.versions)}: {canonical} ({unit.raw_unit_status.value}; {unit.execution_mode.value})"
            )
        parts.append(f"{measurement.column}: " + "; ".join(version_parts))
    return " | ".join(parts)


def _activity_guide(feature: str) -> FeatureGuide:
    policy = CURATION_POLICIES[feature]
    measurements = ", ".join(policy.measurement_columns) or "no exporter submeasurements"
    return FeatureGuide(
        feature=feature,
        title=f"Google Fitness {feature} intervals",
        row_meaning=policy.native_schema.row_meaning,
        recommended_use=(
            "Use start and end boundaries for interval-based activity exposure, bouts, and later overlap-aware resampling. "
            "Keep exporter submeasurements coupled to the interval."
        ),
        default_projection=(
            f"The interval and raw {measurements} remain available. Canonical activity submeasurement projection is "
            f"blocked until exporter semantics are verified."
        ),
        unit_interpretation=_unit_summary(feature),
        temporal_interpretation=(
            "The native interval is authoritative. Duration may be derived exactly from start_date and end_date; "
            "curation does not append or replace it."
        ),
        provenance_interpretation=(
            "Acquisition, derivation and tracking are separate dimensions. Application or stream identifiers do not by "
            "themselves prove a physical device or user-entry method."
        ),
        limitations=(
            "Activity quantity has no defensible common unit across activity types.",
            "Activity calories and distance are exporter-enriched fields whose standard Google data-type mapping is not yet frozen.",
            "This feature is source-specific and is not automatically equivalent to an Apple measurement with a similar name.",
        ),
        evidence_refs=policy.evidence_refs,
    )


def _sleep_guide() -> FeatureGuide:
    policy = CURATION_POLICIES["Sleep"]
    return FeatureGuide(
        feature="Sleep",
        title="Google Fitness sleep-state intervals",
        row_meaning=policy.native_schema.row_meaning,
        recommended_use=(
            "Use the native SLEEP, AWAKE, DEEP, LIGHT and REM states as source-specific interval evidence. "
            "Retain all overlaps and provenance rather than forcing one hypnogram."
        ),
        default_projection=(
            "Native state and boundaries remain unchanged. Curation may append provenance, overlap flags, status and "
            "inclusion metadata."
        ),
        unit_interpretation=_unit_summary("Sleep"),
        temporal_interpretation=(
            "All intervals remain native. Same-provenance overlaps may be annotated; cross-provenance overlap is "
            "contextual and not automatically resolved."
        ),
        provenance_interpretation=(
            "Tracking and source identifiers are retained. Google sleep states are not remapped to Apple sleep terminology."
        ),
        limitations=(
            "SLEEP is not automatically equivalent to Apple ASLEEP.",
            "LIGHT is not automatically equivalent to Apple CORE.",
            "Curation does not create a single-state hypnogram or choose one source over another.",
        ),
        evidence_refs=policy.evidence_refs,
    )


def _scalar_guide(feature: str) -> FeatureGuide:
    policy = CURATION_POLICIES[feature]
    descriptions = {
        "Calories": (
            "Google Fitness interval energy totals",
            "Use as source interval totals. Do not interpret as Apple active energy because the proposed Google contract "
            "includes resting energy.",
            "Canonical kcal values remain withheld in CWP1 until the exporter mapping is frozen.",
        ),
        "Distance": (
            "Google Fitness interval distance totals",
            "Use as source interval totals without imposing calendar-day boundaries.",
            "Canonical meter values remain withheld in CWP1 until the exporter mapping is frozen.",
        ),
        "Weight": (
            "Google Fitness body-weight observations",
            "Preserve both source boundaries and added_by provenance. Positive-duration rows remain intervals and are "
            "not collapsed to points.",
            "Canonical kilogram values remain withheld in CWP1 until the exporter mapping is frozen.",
        ),
        "Height": (
            "Google Fitness height observations",
            "Preserve both source boundaries and added_by provenance. The cohort contains only two native rows, so "
            "calibration evidence is limited.",
            "Canonical meter values remain withheld in CWP1 until the exporter mapping is frozen.",
        ),
    }
    title, use, projection = descriptions[feature]
    limitations = list(policy.calibration.limitations)
    if not limitations:
        limitations.append("Canonical execution remains provisional until the full-native calibration audit freezes the exporter mapping.")
    if feature == "Calories":
        limitations.append("Negative source values are retained and categorized by magnitude; they are never clipped to zero.")
    if feature == "Weight":
        limitations.append("added_by identifies an application, not necessarily a measurement method.")
    return FeatureGuide(
        feature=feature,
        title=title,
        row_meaning=policy.native_schema.row_meaning,
        recommended_use=use,
        default_projection=projection,
        unit_interpretation=_unit_summary(feature),
        temporal_interpretation=(
            "start_date and end_date remain the native temporal evidence. source_day_label is contextual and never "
            "creates a canonical date."
        ),
        provenance_interpretation=(
            "Where provenance is absent, acquisition and derivation may remain unknown. added_by is preserved without "
            "inferring user entry or device measurement."
        ),
        limitations=tuple(limitations),
        evidence_refs=policy.evidence_refs,
    )


def _step_count_guide() -> FeatureGuide:
    policy = CURATION_POLICIES["StepCount"]
    return FeatureGuide(
        feature="StepCount",
        title="Google Fitness StepCount representations",
        row_meaning=policy.native_schema.row_meaning,
        recommended_use=(
            "Choose a source series explicitly. Version 1 is a separate legacy scalar representation. Version 2 retains e"
            "very named alternative."
        ),
        default_projection=(
            "Curation may annotate one preferred version-2 row per outer group when evidence permits, but every row "
            "remains physically present."
        ),
        unit_interpretation=_unit_summary("StepCount"),
        temporal_interpretation=(
            "outer_datetime, summary_date, start_date and end_date are separate evidence. Curation does not manufacture "
            "one canonical day."
        ),
        provenance_interpretation=(
            "aggregation source, application, origin stream and device fields are retained. Acquisition and derivation "
            "are classified separately."
        ),
        limitations=(
            "estimated_steps preference is an analytical default, not a declaration that other alternatives are invalid.",
            "Version 1 and version 2 are not deduplicated or merged by date.",
            "Fallback preference and disagreement effects remain provisional until CWP2 calibration.",
        ),
        evidence_refs=policy.evidence_refs,
    )


def _build_guides() -> dict[str, FeatureGuide]:
    guides: dict[str, FeatureGuide] = {}
    for feature, policy in CURATION_POLICIES.items():
        if feature == "Sleep":
            guides[feature] = _sleep_guide()
        elif feature == "StepCount":
            guides[feature] = _step_count_guide()
        elif policy.event_kind is EventKind.ACTIVITY_INTERVAL:
            guides[feature] = _activity_guide(feature)
        else:
            guides[feature] = _scalar_guide(feature)
    return dict(sorted(guides.items()))


_GUIDES = _build_guides()
FEATURE_GUIDES: Mapping[str, FeatureGuide] = MappingProxyType(_GUIDES)


def known_guides() -> tuple[str, ...]:
    return tuple(FEATURE_GUIDES)


def get_feature_guide(feature: str) -> FeatureGuide:
    try:
        return FEATURE_GUIDES[feature]
    except KeyError as exc:
        raise KeyError(f"Unknown Google Fitness curation guide: {feature!r}") from exc


def validate_guidance() -> tuple[str, ...]:
    issues: list[str] = []
    if set(FEATURE_GUIDES) != set(CURATION_POLICIES):
        issues.append("guidance feature set differs from curation registry")
    if guidance_fingerprint() != EXPECTED_GUIDANCE_FINGERPRINT:
        issues.append(
            "guidance fingerprint mismatch: "
            f"expected {EXPECTED_GUIDANCE_FINGERPRINT}, observed {guidance_fingerprint()}"
        )
    for feature, guide in FEATURE_GUIDES.items():
        if guide.feature != feature:
            issues.append(f"guide key mismatch for {feature}")
        missing = sorted(set(guide.evidence_refs) - set(EVIDENCE))
        if missing:
            issues.append(f"guide {feature} references unknown evidence: {missing}")
        if not guide.limitations:
            issues.append(f"guide lacks explicit limitations: {feature}")
        if CURATION_POLICIES[feature].calibration.execution_mode is PolicyExecutionMode.BLOCKED_DERIVATION and "blocked" not in guide.default_projection.casefold():
            issues.append(f"blocked derivation is not visible in guide: {feature}")
    return tuple(sorted(set(issues)))


def guidance_fingerprint() -> str:
    rendered = json.dumps(
        [FEATURE_GUIDES[name].as_dict() for name in FEATURE_GUIDES],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(rendered).hexdigest()


def guidance_payload() -> dict[str, object]:
    return {
        "guides": [FEATURE_GUIDES[name].as_dict() for name in FEATURE_GUIDES],
        "count": len(FEATURE_GUIDES),
        "fingerprint": guidance_fingerprint(),
        "validation_issues": list(validate_guidance()),
    }


_GUIDANCE_ISSUES = validate_guidance()
if _GUIDANCE_ISSUES:
    raise RuntimeError("Invalid Google Fitness curation guidance: " + "; ".join(_GUIDANCE_ISSUES))
