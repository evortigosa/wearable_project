"""
Wearable Data Processing and Modeling project
Column roles and named DataLoader projections for native and curated feature files.
This module declares what each stored column is *for*, which is a different question from where it came
from. ``processing/cleaners.py`` already partitions the native column universe by origin
(``PROVENANCE_COLUMNS``, ``CONTEXT_COLUMNS``, ``UNIT_COLUMNS``, ``AUDIT_COLUMNS``) and uses those groups
to fix the stored column order. A projection needs a different partition of the same names: origin
groups ``utc_offset_minutes`` and ``acquisition_method`` together with ``record_id`` and ``source_name``,
yet an ordinary analysis wants the first two and not the last two.

Scope and guarantees
--------------------
Nothing here reads, writes, or transforms stored data. A projection only decides which stored columns a
loader returns; ``projection="full"`` always returns every column present in the file, so no information
is reachable only through this module.
Dependencies are deliberately limited to the standard library and ``wearable_project.processing.registry``,
which is itself a pure data module. The curation policy stack is not imported here.

Unknown columns
---------------
``processing/cleaners.output_dataframe`` passes through any column it does not recognize. A column with
no declared role is therefore treated as unknown: it is withheld from ``default`` and ``analysis``, and
returned by ``full``. Withholding is the safe direction, because an undeclared column may carry
identifying or outsized content that has not been reviewed. ``undeclared_columns`` reports them so a
test or a release check can catch the omission.
"""


from __future__ import annotations
from enum import Enum
from typing import Iterable, Mapping
from wearable_project.processing.registry import SPECS


SCHEMA_VERSION = "2026-09-dataloader-projection-1"


class ColumnRole(str, Enum):
    """What a stored column is for, independent of which layer wrote it."""

    TEMPORAL = "temporal"
    MEASUREMENT = "measurement"
    UNIT = "unit"
    CURATION_VERDICT = "curation_verdict"
    CURATION_AUDIT = "curation_audit"
    ACQUISITION = "acquisition"
    LOCAL_TIME = "local_time"
    TIME_ZONE = "time_zone"
    QUALITY = "quality"
    FEATURE_CONTEXT = "feature_context"
    DEVICE = "device"
    RECONCILIATION = "reconciliation"
    IDENTIFIER = "identifier"
    PAYLOAD = "payload"
    INGEST = "ingest"


COLUMN_ROLES: Mapping[str, ColumnRole] = {
    # Event timing. ``start_date`` (or ``datetime`` for ActivitySummary) also becomes the HPP ``Date``
    # index level; it is retained as a column so the interval stays visible alongside the anchor.
    "start_date": ColumnRole.TEMPORAL,
    "end_date": ColumnRole.TEMPORAL,
    "datetime": ColumnRole.TEMPORAL,

    # Measured quantities. These mirror ``FeatureSpec.measurement_columns`` for every registered
    # feature, with one deliberate exception recorded under PAYLOAD below.
    "value": ColumnRole.MEASUREMENT,
    "blood_pressure_systolic_value": ColumnRole.MEASUREMENT,
    "blood_pressure_diastolic_value": ColumnRole.MEASUREMENT,
    "average_heart_rate": ColumnRole.MEASUREMENT,
    "sampling_frequency": ColumnRole.MEASUREMENT,
    "apple_stand_hours": ColumnRole.MEASUREMENT,
    "apple_exercise_time": ColumnRole.MEASUREMENT,
    "active_energy_burned": ColumnRole.MEASUREMENT,
    "apple_stand_hours_goal": ColumnRole.MEASUREMENT,
    "apple_exercise_time_goal": ColumnRole.MEASUREMENT,
    "active_energy_burned_goal": ColumnRole.MEASUREMENT,

    # Units as stored and as resolved. Canonical fields stay NA where resolution was withheld; that
    # absence is meaningful and is never filled.
    "raw_unit": ColumnRole.UNIT,
    "canonical_value": ColumnRole.UNIT,
    "canonical_unit": ColumnRole.UNIT,
    "unit_status": ColumnRole.UNIT,

    # Curation verdicts: what the policy concluded about a row.
    "curation_status": ColumnRole.CURATION_VERDICT,
    "curation_flags": ColumnRole.CURATION_VERDICT,
    "include_by_default": ColumnRole.CURATION_VERDICT,
    "curation_unit_status": ColumnRole.CURATION_VERDICT,

    # Curation audit trail: why the policy concluded it.
    "unit_evidence": ColumnRole.CURATION_AUDIT,
    "unit_epoch_id": ColumnRole.CURATION_AUDIT,

    # How the observation was obtained. Both are needed to separate measured from entered values.
    "acquisition_method": ColumnRole.ACQUISITION,
    "was_user_entered": ColumnRole.ACQUISITION,

    # Offset needed to recover local wall-clock time from the stored UTC instant.
    "utc_offset_minutes": ColumnRole.LOCAL_TIME,

    # The IANA zone name. Analytically useful and also quasi-identifying, because a dated sequence of
    # zones is a travel trace, so it is separated from the numeric offset and withheld from ``default``.
    "time_zone": ColumnRole.TIME_ZONE,

    # Milestone 1 row-level quality observations.
    "quality_flags": ColumnRole.QUALITY,

    # Feature-specific interpretive context that changes how a value should be read. ``status`` marks
    # CGM readings pinned at the sensor reporting limits, so it belongs with the value it qualifies.
    "heart_rate_motion_context": ColumnRole.FEATURE_CONTEXT,
    "status": ColumnRole.FEATURE_CONTEXT,
    "trend_arrow": ColumnRole.FEATURE_CONTEXT,
    "trend_rate": ColumnRole.FEATURE_CONTEXT,
    "vo2_max_test_type": ColumnRole.FEATURE_CONTEXT,
    "classification": ColumnRole.FEATURE_CONTEXT,
    "waveform_sample_count": ColumnRole.FEATURE_CONTEXT,
    "payload_index": ColumnRole.FEATURE_CONTEXT,

    # Device and algorithm descriptors: ordinary analytical provenance, not personal labels.
    "device": ColumnRole.DEVICE,
    "algorithm_version": ColumnRole.DEVICE,

    # Milestone 1 deduplication and revision bookkeeping.
    "occurrence_count": ColumnRole.RECONCILIATION,
    "duplicate_count": ColumnRole.RECONCILIATION,
    "revision_count": ColumnRole.RECONCILIATION,
    "duplicate_details": ColumnRole.RECONCILIATION,
    "revision_details": ColumnRole.RECONCILIATION,
    "conflict_group_id": ColumnRole.RECONCILIATION,

    # Persistent identifiers and free-text labels. ``source_name`` and ``metadata_device_name`` carry
    # user-chosen device names; ``metadata`` is the unparsed payload the other context columns were
    # extracted from. Available through ``full`` for authorized provenance work.
    "record_id": ColumnRole.IDENTIFIER,
    "source_id": ColumnRole.IDENTIFIER,
    "source_name": ColumnRole.IDENTIFIER,
    "metadata_device_name": ColumnRole.IDENTIFIER,
    "metadata": ColumnRole.IDENTIFIER,
    "duplicate_record_ids": ColumnRole.IDENTIFIER,

    # Embedded high-volume payload. ``voltage_measurements`` is a declared ECG measurement column, but a
    # single row holds roughly 15,360 sample pairs, so returning it by default would dominate any ECG
    # load. It is reachable through ``full`` or an explicit ``columns=`` request.
    "voltage_measurements": ColumnRole.PAYLOAD,

    # Pipeline bookkeeping with no analytical meaning at the row level.
    "created_at": ColumnRole.INGEST,
    "updated_at": ColumnRole.INGEST,
    "data_source": ColumnRole.INGEST,
    "collecting_method_version": ColumnRole.INGEST,
    "metadata_sync_version": ColumnRole.INGEST,
}


DEFAULT_ROLES: frozenset[ColumnRole] = frozenset({
    ColumnRole.TEMPORAL,
    ColumnRole.MEASUREMENT,
    ColumnRole.UNIT,
    ColumnRole.CURATION_VERDICT,
    ColumnRole.ACQUISITION,
    ColumnRole.LOCAL_TIME,
    ColumnRole.QUALITY,
    ColumnRole.FEATURE_CONTEXT,
})

ANALYSIS_ROLES: frozenset[ColumnRole] = DEFAULT_ROLES | frozenset({
    ColumnRole.CURATION_AUDIT,
    ColumnRole.DEVICE,
    ColumnRole.RECONCILIATION,
    ColumnRole.TIME_ZONE,
})

FULL_ROLES: frozenset[ColumnRole] = frozenset(ColumnRole)


PROJECTIONS: Mapping[str, frozenset[ColumnRole]] = {
    "default": DEFAULT_ROLES,
    "analysis": ANALYSIS_ROLES,
    "full": FULL_ROLES,
}

PROJECTION_SUMMARIES: Mapping[str, str] = {
    "default": (
        "Timing, measurements, units, curation verdicts, acquisition method, UTC offset, quality flags "
        "and feature-specific context. Excludes identifiers, free-text device labels, the ECG waveform "
        "payload, reconciliation bookkeeping and ingest metadata."
    ),
    "analysis": (
        "Everything in 'default' plus the unit-resolution audit trail, device and algorithm descriptors, "
        "deduplication bookkeeping and the IANA time zone."
    ),
    "full": (
        "Every column present in the stored file, including persistent identifiers, user-chosen device "
        "names, the raw metadata payload and the ECG waveform. Appropriate for authorised provenance "
        "work inside the protected environment; avoid when producing derived exports."
    ),
}

DEFAULT_PROJECTION = "default"


def available_projections() -> tuple[str, ...]:
    """Return the declared projection names, widest last."""

    return ("default", "analysis", "full")


def is_known_projection(name: str) -> bool:
    return name in PROJECTIONS


def projection_roles(name: str) -> frozenset[ColumnRole]:
    """Return the roles a projection admits. Raises KeyError for an unknown name."""

    return PROJECTIONS[name]


def describe_projection(name: str) -> str:
    return PROJECTION_SUMMARIES[name]


def role_of(column: str) -> ColumnRole | None:
    """Return a column's declared role, or None when the column has no declaration."""

    return COLUMN_ROLES.get(column)


def columns_for_projection(name: str, available: Iterable[str]) -> list[str]:
    """
    Return the subset of ``available`` admitted by the named projection, preserving input order.
    ``available`` is the column list of one stored file, so the result is naturally feature-specific and
    phase-specific: curation-appended columns simply are not present in a native file. Columns with no
    declared role are returned only by ``full``.
    """

    roles = PROJECTIONS[name]
    if roles == FULL_ROLES:
        return list(available)
    return [column for column in available
            if (role := COLUMN_ROLES.get(column)) is not None and role in roles]


def undeclared_columns(available: Iterable[str]) -> tuple[str, ...]:
    """Return the columns in ``available`` that carry no declared role, sorted."""

    return tuple(sorted({column for column in available if column not in COLUMN_ROLES}))


def declared_measurement_columns() -> frozenset[str]:
    """Every column any registered feature declares as a measurement, from the processing registry."""

    return frozenset(column for spec in SPECS.values() for column in spec.measurement_columns)


def columns_by_role() -> dict[ColumnRole, tuple[str, ...]]:
    """Group the declared column names by role, for documentation and tests."""

    grouped: dict[ColumnRole, list[str]] = {role: [] for role in ColumnRole}
    for column, role in COLUMN_ROLES.items():
        grouped[role].append(column)
    return {role: tuple(sorted(names)) for role, names in grouped.items()}
