"""
Wearable Data Processing and Modeling project
Google Fitness DataLoader column roles, projections, and stable dtype declarations.
The native registry owns physical source columns.  The curated schema-v2 contract appends sparse physical
review decisions and defines deterministic virtual projections. This module presents both phases through one
feature-bound logical schema while preserving the established native projections and keeping persistent source,
package, and device identifiers restricted to ``full``.
"""


from __future__ import annotations
from enum import Enum
from functools import lru_cache
from typing import Iterable, Mapping
from wearable_project.processing.google_fitness.registry import (
    ColumnRole as NativeColumnRole, ColumnType, NativeFeatureSpec, get_feature_spec,
)


SCHEMA_VERSION = "google-fitness-dataloader-native-schema-1"
CURATED_DATALOADER_SCHEMA_VERSION = "google-fitness-dataloader-curated-schema-v2-1"
DEFAULT_PROJECTION = "default"


class ColumnRole(str, Enum):
    """Analytical role of a Google Fitness DataLoader column."""

    TEMPORAL = "temporal"
    MEASUREMENT = "measurement"
    STATE = "state"
    SOURCE_VERSION = "source_version"
    ACQUISITION = "acquisition"
    DERIVATION = "derivation"
    FEATURE_CONTEXT = "feature_context"
    CANONICAL = "canonical"
    CURATION_DECISION = "curation_decision"
    CURATION_AUDIT = "curation_audit"
    DEVICE = "device"
    IDENTIFIER = "identifier"
    LINEAGE = "lineage"


ROLE_DESCRIPTIONS: Mapping[ColumnRole, str] = {
    ColumnRole.TEMPORAL: (
        "Source timing. UTC instants remain timezone-aware; summary_date is a calendar label rather than an instant."
    ),
    ColumnRole.MEASUREMENT: "A source measurement retained in its native exporter representation.",
    ColumnRole.STATE: "A source categorical state, such as a Google Fitness sleep state.",
    ColumnRole.SOURCE_VERSION: "Exporter collecting-method version controlling the source payload contract.",
    ColumnRole.ACQUISITION: "Source evidence or reviewed interpretation of how an observation entered the product.",
    ColumnRole.DERIVATION: "Reviewed derivation or tracking interpretation reconstructed from source evidence.",
    ColumnRole.FEATURE_CONTEXT: "Context required to interpret a value, source day label, or aggregation alternative.",
    ColumnRole.CANONICAL: (
        "Contract-defined canonical value/unit projection or explicit unit-resolution verdict; missing values may "
        "record deliberate withholding rather than absent source data."
    ),
    ColumnRole.CURATION_DECISION: (
        "Row- or group-level curation status, findings, default-inclusion decision, or StepCount representation role."
    ),
    ColumnRole.CURATION_AUDIT: "Detailed evidence supporting a curation or unit-resolution interpretation.",
    ColumnRole.DEVICE: "Application or coarse device descriptors useful for source-stratified analysis.",
    ColumnRole.IDENTIFIER: (
        "Persistent identifiers, package identifiers, and free-text source/application labels; restricted to full."
    ),
    ColumnRole.LINEAGE: "Outer-row and payload lineage used to reproduce source grouping and alternative order.",
}

DEFAULT_ROLES: frozenset[ColumnRole] = frozenset({
    ColumnRole.TEMPORAL,
    ColumnRole.MEASUREMENT,
    ColumnRole.STATE,
    ColumnRole.SOURCE_VERSION,
    ColumnRole.ACQUISITION,
    ColumnRole.DERIVATION,
    ColumnRole.FEATURE_CONTEXT,
    ColumnRole.CANONICAL,
    ColumnRole.CURATION_DECISION,
})
ANALYSIS_ROLES: frozenset[ColumnRole] = DEFAULT_ROLES | frozenset({
    ColumnRole.CURATION_AUDIT,
    ColumnRole.DEVICE,
    ColumnRole.LINEAGE,
})
FULL_ROLES: frozenset[ColumnRole] = frozenset(ColumnRole)
PROJECTIONS: Mapping[str, frozenset[ColumnRole]] = {
    "default": DEFAULT_ROLES,
    "analysis": ANALYSIS_ROLES,
    "full": FULL_ROLES,
}
NATIVE_PROJECTION_SUMMARIES: Mapping[str, str] = {
    "default": (
        "Timing, measurements or states, collecting-method version, tracking evidence, and feature context. "
        "Excludes persistent identifiers, free-text source/application labels, detailed device descriptors, and lineage."
    ),
    "analysis": (
        "Everything in 'default' plus non-unique application/device descriptors and outer-row/payload lineage. "
        "Persistent source, package, and device identifiers remain withheld."
    ),
    "full": (
        "Every stored column, including persistent identifiers, package names, free-text source labels, and device UID. "
        "Use only for authorised provenance work inside the protected environment."
    ),
}
CURATED_PROJECTION_SUMMARIES: Mapping[str, str] = {
    "default": (
        "Native timing, measurements or states, source-version/context, reviewed provenance, canonical/unit verdicts, "
        "and row/group curation decisions. Persistent identifiers, detailed device descriptors, lineage, and detailed "
        "unit evidence remain withheld."
    ),
    "analysis": (
        "Everything in 'default' plus detailed unit evidence, non-unique application/device descriptors, and "
        "outer-row/payload lineage. Persistent source, package, and device identifiers remain withheld."
    ),
    "full": (
        "Every logical curated column: the complete stored native prefix, sparse physical decisions reconstructed to "
        "their defaults, contract-defined virtual projections, and restricted identifiers."
    ),
}
# Backward-compatible module-level name used by Increment 2 callers/tests.
PROJECTION_SUMMARIES = NATIVE_PROJECTION_SUMMARIES

_IDENTIFIER_COLUMNS = frozenset({
    "source_name",
    "source_id",
    "origin_data_source_id",
    "app_package_name",
    "device_uid",
    "added_by",
})
_DEVICE_COLUMNS = frozenset({
    "device",
    "app_name",
    "device_manufacturer",
    "device_model",
    "device_type",
})
_ACQUISITION_COLUMNS = frozenset({"tracked", "acquisition_method"})
_DERIVATION_COLUMNS = frozenset({"derivation_method", "tracking_status"})
_FEATURE_CONTEXT_COLUMNS = frozenset({"source_day_label", "aggregation_source"})
_CANONICAL_COLUMNS = frozenset({
    "canonical_value", "canonical_unit", "curation_unit_status",
})
_CURATION_DECISION_COLUMNS = frozenset({
    "step_count_role", "curation_status", "curation_flags", "include_by_default",
})
_CURATION_AUDIT_COLUMNS = frozenset({"unit_evidence"})
_LINEAGE_COLUMNS = frozenset({"outer_id", "payload_index"})


class GoogleFitnessFeatureSchema:
    """Feature- and phase-bound projection/dtype schema used by the shared loader engine."""

    ColumnRole = ColumnRole
    ROLE_DESCRIPTIONS = ROLE_DESCRIPTIONS
    DEFAULT_PROJECTION = DEFAULT_PROJECTION

    def __init__(self, feature: str, phase: str = "native") -> None:
        normalized = str(phase).strip().lower()
        if normalized not in {"native", "curated"}:
            raise ValueError(f"unknown Google Fitness DataLoader phase: {phase!r}")
        self.feature = feature
        self.phase = normalized
        self.spec: NativeFeatureSpec = get_feature_spec(feature)
        self.policy = None
        if normalized == "curated":
            from wearable_project.curation.google_fitness.registry import get_policy

            self.policy = get_policy(feature, allow_fallback=False)
        self.SCHEMA_VERSION = (
            SCHEMA_VERSION if normalized == "native" else CURATED_DATALOADER_SCHEMA_VERSION
        )
        self._roles = self._build_roles()

        categorical: set[str] = set()
        strings: set[str] = set()
        integers: set[str] = set()
        booleans: set[str] = set()
        numeric: set[str] = set()
        timestamps: set[str] = set()
        dates: set[str] = set()
        text: set[str] = set()
        for column in self.spec.columns:
            if column.data_type is ColumnType.TIMESTAMP_UTC:
                timestamps.add(column.name)
            elif column.data_type is ColumnType.DATE:
                dates.add(column.name)
                text.add(column.name)
            elif column.data_type is ColumnType.NUMBER:
                numeric.add(column.name)
            elif column.data_type is ColumnType.INTEGER:
                numeric.add(column.name)
                integers.add(column.name)
            elif column.data_type is ColumnType.BOOLEAN:
                booleans.add(column.name)
            elif column.data_type is ColumnType.STRING:
                text.add(column.name)
                role = self._roles[column.name]
                if role in {
                    ColumnRole.STATE,
                    ColumnRole.SOURCE_VERSION,
                    ColumnRole.ACQUISITION,
                    ColumnRole.FEATURE_CONTEXT,
                    ColumnRole.DEVICE,
                }:
                    categorical.add(column.name)
                else:
                    strings.add(column.name)

        if self.policy is not None:
            for column in self.policy.output.materialized_columns:
                if column == "include_by_default":
                    booleans.add(column)
                else:
                    text.add(column)
                    categorical.add(column)
            for column in self.policy.output.virtual_columns:
                if column == "canonical_value":
                    numeric.add(column)
                    if feature == "StepCount":
                        integers.add(column)
                else:
                    text.add(column)
                    categorical.add(column)

        self.CATEGORICAL_COLUMNS = frozenset(categorical)
        self.STRING_COLUMNS = frozenset(strings)
        self.INTEGER_COLUMNS = frozenset(integers)
        self.NULLABLE_BOOLEAN_COLUMNS = frozenset(booleans)
        self.NUMERIC_COLUMNS = frozenset(numeric)
        self.TIMESTAMP_COLUMNS = tuple(
            column.name for column in self.spec.columns if column.name in timestamps
        )
        self.CALENDAR_DATE_COLUMNS = tuple(
            column.name for column in self.spec.columns if column.name in dates
        )
        self.TEXT_COLUMNS = frozenset(text)

    def _build_roles(self) -> dict[str, ColumnRole]:
        roles: dict[str, ColumnRole] = {}
        state_columns = set(self.spec.state_columns)
        measurement_columns = set(self.spec.measurement_columns)
        for column in self.spec.columns:
            name = column.name
            if name in state_columns or column.role is NativeColumnRole.STATE:
                role = ColumnRole.STATE
            elif name in measurement_columns or column.role is NativeColumnRole.MEASUREMENT:
                role = ColumnRole.MEASUREMENT
            elif column.role in {
                NativeColumnRole.TIME_START,
                NativeColumnRole.TIME_END,
                NativeColumnRole.TIME_OUTER,
                NativeColumnRole.SUMMARY_DATE,
            }:
                role = ColumnRole.TEMPORAL
            elif column.role is NativeColumnRole.SOURCE_VERSION:
                role = ColumnRole.SOURCE_VERSION
            elif name in _ACQUISITION_COLUMNS:
                role = ColumnRole.ACQUISITION
            elif name in _FEATURE_CONTEXT_COLUMNS:
                role = ColumnRole.FEATURE_CONTEXT
            elif name in _DEVICE_COLUMNS:
                role = ColumnRole.DEVICE
            elif name in _IDENTIFIER_COLUMNS:
                role = ColumnRole.IDENTIFIER
            elif name in _LINEAGE_COLUMNS or column.role is NativeColumnRole.LINEAGE:
                role = ColumnRole.LINEAGE
            else:
                # A new provenance column is safer in the restricted full-only role until reviewed.
                role = ColumnRole.IDENTIFIER
            roles[name] = role

        if self.policy is not None:
            for name in self.policy.output.materialized_columns + self.policy.output.virtual_columns:
                if name in _ACQUISITION_COLUMNS:
                    roles[name] = ColumnRole.ACQUISITION
                elif name in _DERIVATION_COLUMNS:
                    roles[name] = ColumnRole.DERIVATION
                elif name in _CANONICAL_COLUMNS:
                    roles[name] = ColumnRole.CANONICAL
                elif name in _CURATION_DECISION_COLUMNS:
                    roles[name] = ColumnRole.CURATION_DECISION
                elif name in _CURATION_AUDIT_COLUMNS:
                    roles[name] = ColumnRole.CURATION_AUDIT
                else:
                    roles[name] = ColumnRole.CURATION_AUDIT
        return roles

    def categorical_columns(self, feature: str | None = None) -> frozenset[str]:
        return self.CATEGORICAL_COLUMNS

    @staticmethod
    def available_projections() -> tuple[str, ...]:
        return ("default", "analysis", "full")

    @staticmethod
    def is_known_projection(name: str) -> bool:
        return name in PROJECTIONS

    def describe_projection(self, name: str) -> str:
        summaries = NATIVE_PROJECTION_SUMMARIES if self.phase == "native" else CURATED_PROJECTION_SUMMARIES
        return summaries[name]

    def role_of(self, column: str) -> ColumnRole | None:
        return self._roles.get(column)

    def columns_for_projection(self, name: str, available: Iterable[str]) -> list[str]:
        roles = PROJECTIONS[name]
        if roles == FULL_ROLES:
            return list(available)
        return [
            column
            for column in available
            if (role := self._roles.get(column)) is not None and role in roles
        ]

    def undeclared_columns(self, available: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted({column for column in available if column not in self._roles}))


@lru_cache(maxsize=None)
def get_schema(feature: str, phase: str = "native") -> GoogleFitnessFeatureSchema:
    return GoogleFitnessFeatureSchema(feature, phase)


__all__ = [
    "CURATED_DATALOADER_SCHEMA_VERSION",
    "ColumnRole",
    "DEFAULT_PROJECTION",
    "GoogleFitnessFeatureSchema",
    "NATIVE_PROJECTION_SUMMARIES",
    "CURATED_PROJECTION_SUMMARIES",
    "PROJECTION_SUMMARIES",
    "ROLE_DESCRIPTIONS",
    "SCHEMA_VERSION",
    "get_schema",
]
