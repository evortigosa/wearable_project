"""
Wearable Data Processing and Modeling project

Conservative provenance classification for Google Fitness curation.

The classifier keeps acquisition, derivation, and tracking as separate
questions.  Raw application, stream, and device identifiers remain unchanged
in the native prefix; these annotations never claim a unique physical device
or user-entry method without direct evidence.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .context import aggregation_kind
from .models import (
    AcquisitionMethod,
    DerivationMethod,
    EventKind,
    TrackingStatus,
)


@dataclass(frozen=True, slots=True)
class ProvenanceAnnotation:
    acquisition_method: AcquisitionMethod
    derivation_method: DerivationMethod
    tracking_status: TrackingStatus

    def as_values(self, output_dimensions: tuple[str, ...]) -> dict[str, str]:
        values: dict[str, str] = {}
        if "acquisition_method" in output_dimensions:
            values["acquisition_method"] = self.acquisition_method.value
        if "derivation_method" in output_dimensions:
            values["derivation_method"] = self.derivation_method.value
        if "tracking_status" in output_dimensions:
            values["tracking_status"] = self.tracking_status.value
        return values


def is_google_identifier(value: str | None) -> bool:
    text = (value or "").strip().casefold()
    return bool(text) and (
        text.startswith("com.google.")
        or text.startswith("derived:com.google.")
        or "google fitness" in text
        or "google fit" in text
    )


def tracking_status(value: str | None) -> TrackingStatus:
    text = (value or "").strip().casefold()
    if text in {"1", "true", "yes"}:
        return TrackingStatus.TRACKED
    if text in {"0", "false", "no"}:
        return TrackingStatus.UNTRACKED
    return TrackingStatus.UNKNOWN


def _activity(row: Mapping[str, str]) -> ProvenanceAnnotation:
    source_id = (row.get("source_id") or "").strip()
    source_name = (row.get("source_name") or "").strip()
    device = (row.get("device") or "").strip()
    if source_id and not is_google_identifier(source_id):
        acquisition = AcquisitionMethod.THIRD_PARTY_IMPORT
    elif device or source_name:
        acquisition = AcquisitionMethod.DEVICE_ESTIMATE
    elif source_id and is_google_identifier(source_id):
        acquisition = AcquisitionMethod.HEALTH_IMPORT_UNKNOWN
    else:
        acquisition = AcquisitionMethod.UNKNOWN
    return ProvenanceAnnotation(
        acquisition,
        DerivationMethod.ACTIVITY_CLASSIFICATION,
        tracking_status(row.get("tracked")),
    )


def _step_count(row: Mapping[str, str]) -> ProvenanceAnnotation:
    version = (row.get("collecting_method_version") or "").strip()
    if version == "1.0":
        return ProvenanceAnnotation(
            AcquisitionMethod.HEALTH_IMPORT_UNKNOWN,
            DerivationMethod.LEGACY_SUMMARY,
            TrackingStatus.NOT_APPLICABLE,
        )

    package = (row.get("app_package_name") or "").strip()
    device_present = any((row.get(name) or "").strip() for name in (
        "device_manufacturer", "device_model", "device_type", "device_uid"
    ))
    if package and not is_google_identifier(package):
        acquisition = AcquisitionMethod.THIRD_PARTY_IMPORT
    elif device_present:
        acquisition = AcquisitionMethod.DEVICE_ESTIMATE
    else:
        acquisition = AcquisitionMethod.HEALTH_IMPORT_UNKNOWN

    source = aggregation_kind(row.get("aggregation_source"))
    if source == "estimated_steps":
        derivation = DerivationMethod.ESTIMATED
    elif source == "merge_step_deltas":
        derivation = DerivationMethod.MERGED
    elif source not in {"", "<missing>"}:
        derivation = DerivationMethod.THIRD_PARTY_AGGREGATE
    else:
        derivation = DerivationMethod.UNKNOWN
    return ProvenanceAnnotation(acquisition, derivation, TrackingStatus.NOT_APPLICABLE)


def _anthropometric(row: Mapping[str, str]) -> ProvenanceAnnotation:
    added_by = (row.get("added_by") or "").strip()
    if added_by and not is_google_identifier(added_by):
        acquisition = AcquisitionMethod.THIRD_PARTY_IMPORT
    elif added_by:
        acquisition = AcquisitionMethod.HEALTH_IMPORT_UNKNOWN
    else:
        acquisition = AcquisitionMethod.UNKNOWN
    return ProvenanceAnnotation(acquisition, DerivationMethod.RAW, TrackingStatus.NOT_APPLICABLE)


def _outer_summary() -> ProvenanceAnnotation:
    return ProvenanceAnnotation(
        AcquisitionMethod.UNKNOWN,
        DerivationMethod.APPLICATION_SUMMARY,
        TrackingStatus.NOT_APPLICABLE,
    )


def classify_provenance(feature: str, event_kind: EventKind, row: Mapping[str, str]) -> ProvenanceAnnotation:
    if event_kind in {EventKind.ACTIVITY_INTERVAL, EventKind.STATE_INTERVAL}:
        return _activity(row)
    if feature == "StepCount":
        return _step_count(row)
    if feature in {"Weight", "Height"}:
        return _anthropometric(row)
    return _outer_summary()
