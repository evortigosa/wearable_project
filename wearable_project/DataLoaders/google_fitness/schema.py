"""
Wearable Data Processing and Modeling project
Google Fitness native DataLoader column roles, named projections, and stable dtype declarations.
The Google native registry describes where columns came from. This module describes how a DataLoader should
expose them. Unknown future columns are withheld from ``default`` and ``analysis`` and remain reachable through
``full`` or an explicit ``columns=`` request. Persistent source labels, package identifiers and device UIDs are
withheld from ordinary projections; the normal analytical view retains timing, measurements, state, collecting
version, acquisition evidence and source context needed to interpret alternative representations.
"""


from __future__ import annotations
from enum import Enum
from functools import lru_cache
from typing import Iterable, Mapping
from wearable_project.processing.google_fitness.registry import (
    ColumnRole as NativeColumnRole, ColumnType, NativeFeatureSpec, get_feature_spec,
)


SCHEMA_VERSION = "google-fitness-dataloader-native-schema-1"
DEFAULT_PROJECTION = "default"


class ColumnRole(str, Enum):
    """Analytical role of a Google Fitness DataLoader column."""

    TEMPORAL = "temporal"
    MEASUREMENT = "measurement"
    STATE = "state"
    SOURCE_VERSION = "source_version"
    ACQUISITION = "acquisition"
    FEATURE_CONTEXT = "feature_context"
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
    ColumnRole.ACQUISITION: "Source evidence about whether and how an interval was tracked.",
    ColumnRole.FEATURE_CONTEXT: "Context required to interpret a value, source day label, or aggregation alternative.",
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
    ColumnRole.FEATURE_CONTEXT,
})
ANALYSIS_ROLES: frozenset[ColumnRole] = DEFAULT_ROLES | frozenset({
    ColumnRole.DEVICE,
    ColumnRole.LINEAGE,
})
FULL_ROLES: frozenset[ColumnRole] = frozenset(ColumnRole)
PROJECTIONS: Mapping[str, frozenset[ColumnRole]] = {
    "default": DEFAULT_ROLES,
    "analysis": ANALYSIS_ROLES,
    "full": FULL_ROLES,
}
PROJECTION_SUMMARIES: Mapping[str, str] = {
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
_ACQUISITION_COLUMNS = frozenset({"tracked"})
_FEATURE_CONTEXT_COLUMNS = frozenset({"source_day_label", "aggregation_source"})
_LINEAGE_COLUMNS = frozenset({"outer_id", "payload_index"})


class GoogleFitnessFeatureSchema:
    """Feature-bound projection and dtype schema implementing the shared loader schema interface."""

    ColumnRole = ColumnRole
    ROLE_DESCRIPTIONS = ROLE_DESCRIPTIONS
    DEFAULT_PROJECTION = DEFAULT_PROJECTION
    SCHEMA_VERSION = SCHEMA_VERSION

    def __init__(self, feature: str) -> None:
        self.feature = feature
        self.spec: NativeFeatureSpec = get_feature_spec(feature)
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
        return roles

    def categorical_columns(self, feature: str | None = None) -> frozenset[str]:
        return self.CATEGORICAL_COLUMNS

    @staticmethod
    def available_projections() -> tuple[str, ...]:
        return ("default", "analysis", "full")

    @staticmethod
    def is_known_projection(name: str) -> bool:
        return name in PROJECTIONS

    @staticmethod
    def describe_projection(name: str) -> str:
        return PROJECTION_SUMMARIES[name]

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
def get_schema(feature: str) -> GoogleFitnessFeatureSchema:
    return GoogleFitnessFeatureSchema(feature)


__all__ = [
    "ColumnRole",
    "DEFAULT_PROJECTION",
    "GoogleFitnessFeatureSchema",
    "PROJECTION_SUMMARIES",
    "ROLE_DESCRIPTIONS",
    "SCHEMA_VERSION",
    "get_schema",
]
