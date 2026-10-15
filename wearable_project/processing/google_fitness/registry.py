"""
Wearable Data Processing and Modeling project
Versioned native-schema registry for Google Fitness processing. This module is deliberately declarative.
It defines source/version envelopes, lossless activity routing, compact native feature schemas, event-identity
contracts, and explicit reasons for fields that are not repeated in feature CSVs. Parsing, reconciliation, and
I/O are implemented in later Phase 1 work packages and must consume these contracts rather than reinterpret them.
"""


from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Iterable, Mapping


SOURCE_KEY = "google_fitness"
NORMALIZED_RAW_SOURCE = "googlefitness"
RAW_SOURCE_LABELS = frozenset({"GoogleFitness"})
REGISTRY_VERSION = "2026-10-google-fitness-native-wp1"
REGISTRY_SCHEMA_VERSION = 1
EXPECTED_REGISTRY_FINGERPRINT = "3b10ffc152f3b1ad3b966b367e1c40b1eeb8ba670bfdabecb46c9ca8bab47d68"

OUTER_SCHEMA_COLUMNS = (
    "id",
    "participant_id",
    "data_source",
    "name",
    "datetime",
    "data",
    "collecting_method_version",
    "created_at",
    "updated_at",
)

ACTIVITY_OUTER_FEATURE = "Activities"
ACTIVITY_SOURCE_VERSION = "1.0"
ACTIVITY_MEASUREMENT_ORDER = ("calories", "distance", "quantity")
STEP_COUNT_DATA_TYPE_NAME = "com.google.step_count.delta"
STEP_COUNT_DATA_SOURCE_ID = "derived:com.google.step_count.delta:com.google.android.gms:aggregated"


class NativeFeatureFamily(str, Enum):
    """Scientific row family represented by a native feature file."""

    ACTIVITY_INTERVAL = "activity_interval"
    SLEEP_STATE_INTERVAL = "sleep_state_interval"
    INTERVAL_TOTAL = "interval_total"
    POINT_OR_INTERVAL_SCALAR = "point_or_interval_scalar"
    STEP_COUNT = "step_count"


class ColumnType(str, Enum):
    """Logical types expected before CSV serialization."""

    TIMESTAMP_UTC = "timestamp_utc"
    DATE = "date"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    STRING = "string"


class ColumnRole(str, Enum):
    """Semantic role of a native output column."""

    TIME_START = "time_start"
    TIME_END = "time_end"
    TIME_OUTER = "time_outer"
    SUMMARY_DATE = "summary_date"
    MEASUREMENT = "measurement"
    STATE = "state"
    PROVENANCE = "provenance"
    SOURCE_VERSION = "source_version"
    LINEAGE = "lineage"


class OmissionKind(str, Enum):
    """Why a source or convenience field is not repeated in native CSVs."""

    EXACT_DERIVATION = "exact_derivation"
    REGISTRY_CONSTANT = "registry_constant"
    PATH_IDENTITY = "path_identity"
    STATE_LINEAGE = "state_lineage"
    ROUTING_IDENTITY = "routing_identity"
    NOT_NATIVE_SEMANTICS = "not_native_semantics"


class PayloadEnvelope(str, Enum):
    """Observed top-level decoded payload shape."""

    LIST = "list"
    DICT = "dict"
    SCALAR_INTEGER = "scalar_integer"


class DecoderKind(str, Enum):
    """Version-aware parser strategy selected by the registry."""

    ACTIVITY_ITEMS = "activity_items"
    INTERVAL_TOTAL = "interval_total"
    ANTHROPOMETRIC = "anthropometric"
    STEP_COUNT_LEGACY = "step_count_legacy"
    STEP_COUNT_ALTERNATIVES = "step_count_alternatives"


class RouteKind(str, Enum):
    """How an Activities item becomes a native feature row."""

    ONE_LABEL_PER_FEATURE = "one_label_per_feature"
    SLEEP_STATE = "sleep_state"


class IdentityMaturity(str, Enum):
    """Whether a semantic identity is supported beyond source occurrence."""

    SUPPORTED = "supported"
    SOURCE_OCCURRENCE_ONLY = "source_occurrence_only"


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    name: str
    data_type: ColumnType
    nullable: bool
    role: ColumnRole
    source_paths: tuple[str, ...]
    description: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "data_type": self.data_type.value,
            "nullable": self.nullable,
            "role": self.role.value,
            "source_paths": list(self.source_paths),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class OmittedFieldSpec:
    name: str
    kind: OmissionKind
    reason: str
    reconstructable_from: tuple[str, ...] = ()
    retained_in: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "reason": self.reason,
            "reconstructable_from": list(self.reconstructable_from),
            "retained_in": self.retained_in,
        }


@dataclass(frozen=True, slots=True)
class VersionIdentitySpec:
    version: str
    maturity: IdentityMaturity
    semantic_columns: tuple[str, ...]
    rationale: str

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "maturity": self.maturity.value,
            "semantic_columns": list(self.semantic_columns),
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class NativeFeatureSpec:
    name: str
    filename: str
    source_feature: str
    family: NativeFeatureFamily
    supported_versions: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]
    measurement_columns: tuple[str, ...]
    state_columns: tuple[str, ...]
    occurrence_identity_columns: tuple[str, ...]
    semantic_identity_by_version: tuple[VersionIdentitySpec, ...]
    sort_columns: tuple[str, ...]
    validation_rules: tuple[str, ...]
    omitted_fields: tuple[OmittedFieldSpec, ...]
    activity_labels: tuple[str, ...] = ()
    deduplicate_across_versions: bool = False
    provisional: bool = False

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if not column.nullable)

    @property
    def provenance_columns(self) -> tuple[str, ...]:
        return tuple(
            column.name
            for column in self.columns
            if column.role is ColumnRole.PROVENANCE
        )

    def identity_for_version(self, version: str) -> VersionIdentitySpec:
        for identity in self.semantic_identity_by_version:
            if identity.version == version:
                return identity
        raise KeyError(f"{self.name!r} does not support collecting version {version!r}")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "filename": self.filename,
            "source_feature": self.source_feature,
            "family": self.family.value,
            "supported_versions": list(self.supported_versions),
            "columns": [column.as_dict() for column in self.columns],
            "measurement_columns": list(self.measurement_columns),
            "state_columns": list(self.state_columns),
            "occurrence_identity_columns": list(self.occurrence_identity_columns),
            "semantic_identity_by_version": [
                identity.as_dict() for identity in self.semantic_identity_by_version
            ],
            "sort_columns": list(self.sort_columns),
            "validation_rules": list(self.validation_rules),
            "omitted_fields": [item.as_dict() for item in self.omitted_fields],
            "activity_labels": list(self.activity_labels),
            "deduplicate_across_versions": self.deduplicate_across_versions,
            "provisional": self.provisional,
        }


@dataclass(frozen=True, slots=True)
class ActivityRoute:
    raw_label: str
    native_feature: str
    measurements: tuple[str, ...]
    include_source_id: bool
    native_state: str | None = None

    @property
    def native_filename(self) -> str:
        return f"{self.native_feature}.csv"

    @property
    def kind(self) -> RouteKind:
        return (
            RouteKind.SLEEP_STATE
            if self.native_state is not None
            else RouteKind.ONE_LABEL_PER_FEATURE
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "raw_label": self.raw_label,
            "native_feature": self.native_feature,
            "native_filename": self.native_filename,
            "kind": self.kind.value,
            "native_state": self.native_state,
            "measurements": list(self.measurements),
            "include_source_id": self.include_source_id,
        }


@dataclass(frozen=True, slots=True)
class SourceFeatureVersionSpec:
    feature: str
    version: str
    envelope: PayloadEnvelope
    decoder: DecoderKind
    minimum_items: int
    maximum_items: int | None
    accepted_item_types: tuple[str, ...]
    expected_source_fields: tuple[str, ...]
    malformed_items_are_diagnostics: bool
    notes: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return (self.feature, self.version)

    def as_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "version": self.version,
            "envelope": self.envelope.value,
            "decoder": self.decoder.value,
            "minimum_items": self.minimum_items,
            "maximum_items": self.maximum_items,
            "accepted_item_types": list(self.accepted_item_types),
            "expected_source_fields": list(self.expected_source_fields),
            "malformed_items_are_diagnostics": self.malformed_items_are_diagnostics,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class UnregisteredActivityPolicy:
    action: str
    native_feature: str
    native_filename: str
    raw_label_column: str
    columns: tuple[ColumnSpec, ...]
    occurrence_identity_columns: tuple[str, ...]
    sort_columns: tuple[str, ...]
    validation_rules: tuple[str, ...]
    emit_registry_warning: bool
    rationale: str

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "native_feature": self.native_feature,
            "native_filename": self.native_filename,
            "raw_label_column": self.raw_label_column,
            "columns": [column.as_dict() for column in self.columns],
            "occurrence_identity_columns": list(self.occurrence_identity_columns),
            "sort_columns": list(self.sort_columns),
            "validation_rules": list(self.validation_rules),
            "emit_registry_warning": self.emit_registry_warning,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class RegistryValidation:
    issues: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, object]:
        return {"valid": self.valid, "issues": list(self.issues)}


def _column(
    name: str, data_type: ColumnType, nullable: bool, role: ColumnRole, *source_paths: str, description: str,
) -> ColumnSpec:
    return ColumnSpec(
        name=name,
        data_type=data_type,
        nullable=nullable,
        role=role,
        source_paths=tuple(source_paths),
        description=description,
    )


def _omitted(
    name: str, kind: OmissionKind, reason: str, *, reconstructable_from: tuple[str, ...] = (),
    retained_in: str | None = None,
) -> OmittedFieldSpec:
    return OmittedFieldSpec(
        name=name,
        kind=kind,
        reason=reason,
        reconstructable_from=reconstructable_from,
        retained_in=retained_in,
    )


_COMMON_PATH_AND_LINEAGE_OMISSIONS = (
    _omitted(
        "participant_id",
        OmissionKind.PATH_IDENTITY,
        "Participant identity is encoded once by the participant directory.",
        retained_in="participant directory name",
    ),
    _omitted(
        "feature",
        OmissionKind.PATH_IDENTITY,
        "Native feature identity is encoded once by the CSV filename.",
        retained_in="feature filename",
    ),
    _omitted(
        "data_source",
        OmissionKind.REGISTRY_CONSTANT,
        "The product root is source-specific and the registry fixes the source family.",
        retained_in="registry and output-root source guard",
    ),
    _omitted(
        "raw_data_source",
        OmissionKind.STATE_LINEAGE,
        "The original outer source label is lineage evidence, not a repeated measurement.",
        retained_in="source occurrence ledger keyed by outer_id",
    ),
    _omitted(
        "raw_outer_feature",
        OmissionKind.REGISTRY_CONSTANT,
        "The source outer feature is fixed by each NativeFeatureSpec.",
        retained_in="NativeFeatureSpec.source_feature",
    ),
    _omitted(
        "created_at",
        OmissionKind.STATE_LINEAGE,
        "Outer lifecycle metadata is retained once for revision analysis.",
        retained_in="source occurrence ledger keyed by outer_id",
    ),
    _omitted(
        "updated_at",
        OmissionKind.STATE_LINEAGE,
        "Outer lifecycle metadata is retained once for revision analysis.",
        retained_in="source occurrence ledger keyed by outer_id",
    ),
    _omitted(
        "source_month",
        OmissionKind.STATE_LINEAGE,
        "The canonical source month is file-level lineage rather than row semantics.",
        retained_in="source-files state table",
    ),
    _omitted(
        "source_filename",
        OmissionKind.STATE_LINEAGE,
        "The physical monthly file is tracked once with its content hash.",
        retained_in="source-files state table",
    ),
    _omitted(
        "outer_row_number",
        OmissionKind.STATE_LINEAGE,
        "Physical row position is diagnostic lineage and is not a native measurement.",
        retained_in="source occurrence ledger keyed by outer_id",
    ),
)

_INTERVAL_OMISSIONS = (
    _omitted(
        "duration",
        OmissionKind.EXACT_DERIVATION,
        "Duration is exactly reconstructable from normalized interval boundaries.",
        reconstructable_from=("start_date", "end_date"),
    ),
)

_ACTIVITY_OMISSIONS = (
    *_COMMON_PATH_AND_LINEAGE_OMISSIONS,
    *_INTERVAL_OMISSIONS,
    _omitted(
        "outer_datetime",
        OmissionKind.NOT_NATIVE_SEMANTICS,
        "Activity event semantics are carried by the item start/end interval.",
        retained_in="source occurrence ledger keyed by outer_id",
    ),
)

_NON_ACTIVITY_INTERVAL_OMISSIONS = (
    *_COMMON_PATH_AND_LINEAGE_OMISSIONS,
    *_INTERVAL_OMISSIONS,
    _omitted(
        "outer_datetime",
        OmissionKind.NOT_NATIVE_SEMANTICS,
        "The payload interval carries the native event boundaries.",
        retained_in="source occurrence ledger keyed by outer_id",
    ),
    _omitted(
        "payload_index",
        OmissionKind.REGISTRY_CONSTANT,
        "These source/version contracts emit exactly one valid item per outer row.",
        retained_in="SourceFeatureVersionSpec cardinality",
    ),
)


SLEEP_STATE_MAPPING: Mapping[str, str] = MappingProxyType(
    {
        "sleep": "SLEEP",
        "sleep.awake": "AWAKE",
        "sleep.deep": "DEEP",
        "sleep.light": "LIGHT",
        "sleep.rem": "REM",
    }
)


_ACTIVITY_ROUTES = (
    ActivityRoute(raw_label='aerobics', native_feature='Aerobics', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='basketball', native_feature='Basketball', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='biking', native_feature='Biking', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='biking.hand', native_feature='BikingHand', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='biking.mountain', native_feature='BikingMountain', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='biking.road', native_feature='BikingRoad', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='biking.spinning', native_feature='BikingSpinning', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='biking.stationary', native_feature='BikingStationary', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='biking.utility', native_feature='BikingUtility', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='boxing', native_feature='Boxing', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='calisthenics', native_feature='Calisthenics', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='circuit_training', native_feature='CircuitTraining', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='crossfit', native_feature='Crossfit', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='dancing', native_feature='Dancing', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='elliptical', native_feature='Elliptical', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='ergometer', native_feature='Ergometer', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='gardening', native_feature='Gardening', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='guided_breathing', native_feature='GuidedBreathing', measurements=('calories', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='gymnastics', native_feature='Gymnastics', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='handball', native_feature='Handball', measurements=('calories', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='hiking', native_feature='Hiking', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='horseback_riding', native_feature='HorsebackRiding', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='in_vehicle', native_feature='InVehicle', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='interval_training', native_feature='IntervalTraining', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='interval_training.high_intensity', native_feature='IntervalTrainingHighIntensity', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='jump_rope', native_feature='JumpRope', measurements=('calories', 'distance', 'quantity'), include_source_id=False, native_state=None),
    ActivityRoute(raw_label='kayaking', native_feature='Kayaking', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='kick_scooter', native_feature='KickScooter', measurements=('calories', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='kickboxing', native_feature='Kickboxing', measurements=('calories', 'distance', 'quantity'), include_source_id=False, native_state=None),
    ActivityRoute(raw_label='martial_arts', native_feature='MartialArts', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='meditation', native_feature='Meditation', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='other', native_feature='OtherActivity', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='pilates', native_feature='Pilates', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='rock_climbing', native_feature='RockClimbing', measurements=('calories', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='rowing', native_feature='Rowing', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='rowing.machine', native_feature='RowingMachine', measurements=('calories', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='running', native_feature='Running', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='running.jogging', native_feature='RunningJogging', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='running.sand', native_feature='RunningSand', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='running.treadmill', native_feature='RunningTreadmill', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='skating.cross', native_feature='SkatingCross', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='skiing.cross_country', native_feature='SkiingCrossCountry', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='sleep', native_feature='Sleep', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state='SLEEP'),
    ActivityRoute(raw_label='sleep.awake', native_feature='Sleep', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state='AWAKE'),
    ActivityRoute(raw_label='sleep.deep', native_feature='Sleep', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state='DEEP'),
    ActivityRoute(raw_label='sleep.light', native_feature='Sleep', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state='LIGHT'),
    ActivityRoute(raw_label='sleep.rem', native_feature='Sleep', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state='REM'),
    ActivityRoute(raw_label='stair_climbing', native_feature='StairClimbing', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='stair_climbing.machine', native_feature='StairClimbingMachine', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='still', native_feature='Still', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='strength_training', native_feature='StrengthTraining', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='swimming', native_feature='Swimming', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='swimming.open_water', native_feature='SwimmingOpenWater', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='swimming.pool', native_feature='SwimmingPool', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='tennis', native_feature='Tennis', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='unknown', native_feature='UnknownActivity', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='volleyball', native_feature='Volleyball', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='volleyball.beach', native_feature='VolleyballBeach', measurements=('calories', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='volleyball.indoor', native_feature='VolleyballIndoor', measurements=('calories', 'distance', 'quantity'), include_source_id=False, native_state=None),
    ActivityRoute(raw_label='walking', native_feature='Walking', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='walking.fitness', native_feature='WalkingFitness', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='walking.nordic', native_feature='WalkingNordic', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='walking.treadmill', native_feature='WalkingTreadmill', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='weightlifting', native_feature='Weightlifting', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='yoga', native_feature='Yoga', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
    ActivityRoute(raw_label='zumba', native_feature='Zumba', measurements=('calories', 'distance', 'quantity'), include_source_id=True, native_state=None),
)

ACTIVITY_ROUTES: tuple[ActivityRoute, ...] = tuple(
    sorted(_ACTIVITY_ROUTES, key=lambda route: route.raw_label)
)
ACTIVITY_ROUTE_BY_LABEL: Mapping[str, ActivityRoute] = MappingProxyType(
    {route.raw_label: route for route in ACTIVITY_ROUTES}
)


def _activity_columns(
    *, measurements: tuple[str, ...], include_source_id: bool, sleep: bool,
) -> tuple[ColumnSpec, ...]:
    columns: list[ColumnSpec] = [
        _column(
            "start_date",
            ColumnType.TIMESTAMP_UTC,
            False,
            ColumnRole.TIME_START,
            "Activities[].start",
            description="Normalized UTC activity interval start.",
        ),
        _column(
            "end_date",
            ColumnType.TIMESTAMP_UTC,
            False,
            ColumnRole.TIME_END,
            "Activities[].end",
            description="Normalized UTC activity interval end.",
        ),
    ]
    if sleep:
        columns.append(
            _column(
                "value",
                ColumnType.STRING,
                False,
                ColumnRole.STATE,
                "Activities[].activity_name",
                description="Lossless normalized Google Fitness sleep state.",
            )
        )
    measurement_sources = {
        "calories": "Activities[].calories",
        "distance": "Activities[].distance",
        "quantity": "Activities[].quantity",
    }
    measurement_descriptions = {
        "calories": "Source activity calories value; unit is intentionally unresolved.",
        "distance": "Source activity distance value; unit is intentionally unresolved.",
        "quantity": "Source activity quantity value; source semantics remain unresolved.",
    }
    for name in ACTIVITY_MEASUREMENT_ORDER:
        if name not in measurements:
            continue
        columns.append(
            _column(
                name,
                ColumnType.NUMBER,
                True,
                ColumnRole.MEASUREMENT,
                measurement_sources[name],
                description=measurement_descriptions[name],
            )
        )
    columns.extend(
        [
            _column(
                "tracked",
                ColumnType.BOOLEAN,
                False,
                ColumnRole.PROVENANCE,
                "Activities[].tracked",
                description="Source tracking indicator retained without reinterpretation.",
            ),
            _column(
                "device",
                ColumnType.STRING,
                False,
                ColumnRole.PROVENANCE,
                "Activities[].device",
                description="Coarse source device label.",
            ),
            _column(
                "source_name",
                ColumnType.STRING,
                False,
                ColumnRole.PROVENANCE,
                "Activities[].source_name",
                description="Source-provided activity stream name.",
            ),
        ]
    )
    if include_source_id:
        columns.append(
            _column(
                "source_id",
                ColumnType.STRING,
                True,
                ColumnRole.PROVENANCE,
                "Activities[].source_id",
                description="Optional source application or stream identifier.",
            )
        )
    columns.extend(
        [
            _column(
                "collecting_method_version",
                ColumnType.STRING,
                False,
                ColumnRole.SOURCE_VERSION,
                "collecting_method_version",
                description="Exporter collecting-method version.",
            ),
            _column(
                "outer_id",
                ColumnType.STRING,
                False,
                ColumnRole.LINEAGE,
                "id",
                description="Outer export row identifier.",
            ),
            _column(
                "payload_index",
                ColumnType.INTEGER,
                False,
                ColumnRole.LINEAGE,
                "Activities[] index",
                description="Zero-based item index within the decoded Activities payload.",
            ),
        ]
    )
    return tuple(columns)


def _unregistered_activity_columns() -> tuple[ColumnSpec, ...]:
    registered = list(
        _activity_columns(
            measurements=ACTIVITY_MEASUREMENT_ORDER,
            include_source_id=True,
            sleep=False,
        )
    )
    registered.insert(
        2,
        _column(
            "raw_activity_name",
            ColumnType.STRING,
            False,
            ColumnRole.STATE,
            "Activities[].activity_name",
            description=(
                "Unregistered source label retained verbatim until a deliberate "
                "registry update assigns native semantics."
            ),
        ),
    )
    return tuple(registered)


UNREGISTERED_ACTIVITY_POLICY = UnregisteredActivityPolicy(
    action="route_to_fallback_and_emit_registry_warning",
    native_feature="UnregisteredActivity",
    native_filename="UnregisteredActivity.csv",
    raw_label_column="raw_activity_name",
    columns=_unregistered_activity_columns(),
    occurrence_identity_columns=("outer_id", "payload_index"),
    sort_columns=(
        "start_date",
        "end_date",
        "raw_activity_name",
        "outer_id",
        "payload_index",
    ),
    validation_rules=(
        "start_date_and_end_date_are_parseable_utc",
        "end_date_is_strictly_after_start_date",
        "raw_activity_name_is_nonempty_and_unregistered",
        "present_measurements_are_finite_numbers",
        "registry_warning_is_emitted",
    ),
    emit_registry_warning=True,
    rationale=(
        "A future activity label must remain visible without being guessed into an "
        "existing semantic feature. The fallback is not part of the 67 accepted "
        "Phase 0 feature specifications and is activated only by an explicit "
        "unregistered-label diagnostic."
    ),
)


def _activity_omissions(*, sleep: bool) -> tuple[OmittedFieldSpec, ...]:
    if sleep:
        raw_label = _omitted(
            "raw_activity_name",
            OmissionKind.ROUTING_IDENTITY,
            "The raw sleep label is bijectively reconstructable from value and the versioned sleep mapping.",
            reconstructable_from=("value",),
            retained_in="SLEEP_STATE_MAPPING",
        )
    else:
        raw_label = _omitted(
            "raw_activity_name",
            OmissionKind.ROUTING_IDENTITY,
            "A one-to-one registered route makes the raw label reconstructable from the feature filename.",
            reconstructable_from=("feature filename",),
            retained_in="ACTIVITY_ROUTE_BY_LABEL",
        )
    return (*_ACTIVITY_OMISSIONS, raw_label)


def _build_activity_specs() -> dict[str, NativeFeatureSpec]:
    grouped: dict[str, list[ActivityRoute]] = {}
    for route in ACTIVITY_ROUTES:
        grouped.setdefault(route.native_feature, []).append(route)

    result: dict[str, NativeFeatureSpec] = {}
    for feature_name, routes in sorted(grouped.items()):
        sleep = feature_name == "Sleep"
        measurements = tuple(
            name
            for name in ACTIVITY_MEASUREMENT_ORDER
            if any(name in route.measurements for route in routes)
        )
        include_source_id = any(route.include_source_id for route in routes)
        columns = _activity_columns(
            measurements=measurements,
            include_source_id=include_source_id,
            sleep=sleep,
        )
        semantic_columns = [
            "collecting_method_version",
            "start_date",
            "end_date",
            "device",
            "source_name",
        ]
        if include_source_id:
            semantic_columns.append("source_id")
        if sleep:
            semantic_columns.append("value")
        result[feature_name] = NativeFeatureSpec(
            name=feature_name,
            filename=f"{feature_name}.csv",
            source_feature=ACTIVITY_OUTER_FEATURE,
            family=(
                NativeFeatureFamily.SLEEP_STATE_INTERVAL
                if sleep
                else NativeFeatureFamily.ACTIVITY_INTERVAL
            ),
            supported_versions=(ACTIVITY_SOURCE_VERSION,),
            columns=columns,
            measurement_columns=measurements,
            state_columns=("value",) if sleep else (),
            occurrence_identity_columns=("outer_id", "payload_index"),
            semantic_identity_by_version=(
                VersionIdentitySpec(
                    version=ACTIVITY_SOURCE_VERSION,
                    maturity=IdentityMaturity.SUPPORTED,
                    semantic_columns=tuple(semantic_columns),
                    rationale=(
                        "Same-version activity identity requires interval and provenance; "
                        "measurement equality alone is never sufficient."
                    ),
                ),
            ),
            sort_columns=("start_date", "end_date", "outer_id", "payload_index"),
            validation_rules=(
                "start_date_and_end_date_are_parseable_utc",
                "end_date_is_strictly_after_start_date",
                "present_measurements_are_finite_numbers",
                "tracked_is_boolean",
                "each_source_item_routes_exactly_once",
                *(
                    ("value_is_registered_sleep_state",)
                    if sleep
                    else ("feature_has_one_registered_raw_activity_label",)
                ),
            ),
            omitted_fields=_activity_omissions(sleep=sleep),
            activity_labels=tuple(sorted(route.raw_label for route in routes)),
        )
    return result


def _source_day_column(source_feature: str) -> ColumnSpec:
    return _column(
        "source_day_label",
        ColumnType.STRING,
        False,
        ColumnRole.PROVENANCE,
        f"{source_feature}[].day",
        description=(
            "Localized source weekday label retained because it is not exactly "
            "reconstructable from normalized timestamps."
        ),
    )


def _interval_total_spec(name: str, raw_value_field: str) -> NativeFeatureSpec:
    versions = ("1.0", "2.0")
    columns = (
        _column(
            "start_date",
            ColumnType.TIMESTAMP_UTC,
            False,
            ColumnRole.TIME_START,
            f"{name}[].start_date",
            description="Normalized UTC interval start.",
        ),
        _column(
            "end_date",
            ColumnType.TIMESTAMP_UTC,
            False,
            ColumnRole.TIME_END,
            f"{name}[].end_date",
            description="Normalized UTC interval end.",
        ),
        _column(
            "value",
            ColumnType.NUMBER,
            False,
            ColumnRole.MEASUREMENT,
            f"{name}[].{raw_value_field}",
            description=(
                f"Source {name} total retained unchanged; unit is intentionally unresolved."
            ),
        ),
        _source_day_column(name),
        _column(
            "collecting_method_version",
            ColumnType.STRING,
            False,
            ColumnRole.SOURCE_VERSION,
            "collecting_method_version",
            description="Exporter collecting-method version.",
        ),
        _column(
            "outer_id",
            ColumnType.STRING,
            False,
            ColumnRole.LINEAGE,
            "id",
            description="Outer export row identifier; payload cardinality is one.",
        ),
    )
    identities = tuple(
        VersionIdentitySpec(
            version=version,
            maturity=IdentityMaturity.SUPPORTED,
            semantic_columns=(
                "collecting_method_version",
                "start_date",
                "end_date",
            ),
            rationale=(
                "Same-version interval identity is temporal; equal totals alone do not "
                "identify duplicate observations."
            ),
        )
        for version in versions
    )
    return NativeFeatureSpec(
        name=name,
        filename=f"{name}.csv",
        source_feature=name,
        family=NativeFeatureFamily.INTERVAL_TOTAL,
        supported_versions=versions,
        columns=columns,
        measurement_columns=("value",),
        state_columns=(),
        occurrence_identity_columns=("outer_id",),
        semantic_identity_by_version=identities,
        sort_columns=("start_date", "end_date", "outer_id"),
        validation_rules=(
            "start_date_and_end_date_are_parseable_utc",
            "end_date_is_not_before_start_date",
            "value_is_finite_number",
            "negative_values_are_preserved_not_clipped",
            "source_day_label_is_retained_verbatim",
            "one_valid_payload_item_per_outer_row",
        ),
        omitted_fields=_NON_ACTIVITY_INTERVAL_OMISSIONS,
    )


def _anthropometric_spec(name: str, versions: tuple[str, ...]) -> NativeFeatureSpec:
    columns = (
        _column(
            "start_date",
            ColumnType.TIMESTAMP_UTC,
            False,
            ColumnRole.TIME_START,
            f"{name}[].start_date",
            description="Normalized UTC measurement start.",
        ),
        _column(
            "end_date",
            ColumnType.TIMESTAMP_UTC,
            False,
            ColumnRole.TIME_END,
            f"{name}[].end_date",
            description=(
                "Normalized UTC measurement end; retained because Weight is not "
                "universally point-like."
            ),
        ),
        _column(
            "value",
            ColumnType.NUMBER,
            False,
            ColumnRole.MEASUREMENT,
            f"{name}[].value",
            description=(
                f"Source {name} value retained unchanged; unit is intentionally unresolved."
            ),
        ),
        _source_day_column(name),
        _column(
            "added_by",
            ColumnType.STRING,
            False,
            ColumnRole.PROVENANCE,
            f"{name}[].added_by",
            description="Source application/package identifier retained verbatim.",
        ),
        _column(
            "collecting_method_version",
            ColumnType.STRING,
            False,
            ColumnRole.SOURCE_VERSION,
            "collecting_method_version",
            description="Exporter collecting-method version.",
        ),
        _column(
            "outer_id",
            ColumnType.STRING,
            False,
            ColumnRole.LINEAGE,
            "id",
            description="Outer export row identifier; valid payload cardinality is one.",
        ),
    )
    identities = tuple(
        VersionIdentitySpec(
            version=version,
            maturity=IdentityMaturity.SUPPORTED,
            semantic_columns=(
                "collecting_method_version",
                "start_date",
                "end_date",
                "added_by",
            ),
            rationale=(
                "Same-version anthropometric identity requires timing and source "
                "application; equal values alone are not duplicates."
            ),
        )
        for version in versions
    )
    return NativeFeatureSpec(
        name=name,
        filename=f"{name}.csv",
        source_feature=name,
        family=NativeFeatureFamily.POINT_OR_INTERVAL_SCALAR,
        supported_versions=versions,
        columns=columns,
        measurement_columns=("value",),
        state_columns=(),
        occurrence_identity_columns=("outer_id",),
        semantic_identity_by_version=identities,
        sort_columns=("start_date", "end_date", "outer_id"),
        validation_rules=(
            "start_date_and_end_date_are_parseable_utc",
            "end_date_is_not_before_start_date",
            "value_is_finite_number",
            "nonpoint_intervals_are_preserved",
            "added_by_is_retained_verbatim",
            "malformed_items_are_diagnostics_not_numeric_rows",
            "one_valid_payload_item_per_outer_row",
        ),
        omitted_fields=_NON_ACTIVITY_INTERVAL_OMISSIONS,
    )


def _step_count_spec() -> NativeFeatureSpec:
    columns = (
        _column(
            "outer_datetime",
            ColumnType.TIMESTAMP_UTC,
            False,
            ColumnRole.TIME_OUTER,
            "datetime",
            description=(
                "Normalized outer timestamp; the only temporal anchor available to "
                "version 1.0 and independent evidence for version 2.0."
            ),
        ),
        _column(
            "summary_date",
            ColumnType.DATE,
            True,
            ColumnRole.SUMMARY_DATE,
            "StepCount[].steps[].date",
            description="Version 2.0 exported summary date; unavailable in version 1.0.",
        ),
        _column(
            "start_date",
            ColumnType.TIMESTAMP_UTC,
            True,
            ColumnRole.TIME_START,
            "StepCount[].raw_steps[].start_date",
            description="Version 2.0 raw aggregation interval start.",
        ),
        _column(
            "end_date",
            ColumnType.TIMESTAMP_UTC,
            True,
            ColumnRole.TIME_END,
            "StepCount[].raw_steps[].end_date",
            description="Version 2.0 raw aggregation interval end.",
        ),
        _column(
            "value",
            ColumnType.INTEGER,
            False,
            ColumnRole.MEASUREMENT,
            "StepCount scalar",
            "StepCount[].steps[].value",
            description="Legacy scalar count or version 2.0 summary count.",
        ),
        _column(
            "raw_value",
            ColumnType.INTEGER,
            True,
            ColumnRole.MEASUREMENT,
            "StepCount[].raw_steps[].steps",
            description=(
                "Version 2.0 raw interval count retained separately from summary value; "
                "unavailable in version 1.0."
            ),
        ),
        _column(
            "aggregation_source",
            ColumnType.STRING,
            True,
            ColumnRole.PROVENANCE,
            "StepCount[].source",
            description="Version 2.0 named aggregation alternative.",
        ),
        _column(
            "origin_data_source_id",
            ColumnType.STRING,
            True,
            ColumnRole.PROVENANCE,
            "StepCount[].raw_steps[].origin_data_source_id",
            description="Version 2.0 origin stream identifier.",
        ),
        _column(
            "app_name",
            ColumnType.STRING,
            True,
            ColumnRole.PROVENANCE,
            "StepCount[].raw_steps[].app_name",
            description="Optional source application name.",
        ),
        _column(
            "app_package_name",
            ColumnType.STRING,
            True,
            ColumnRole.PROVENANCE,
            "StepCount[].raw_steps[].app_package_name",
            description="Optional source application package identifier.",
        ),
        _column(
            "device_manufacturer",
            ColumnType.STRING,
            True,
            ColumnRole.PROVENANCE,
            "StepCount[].raw_steps[].device_manufacturer",
            description="Optional source device manufacturer.",
        ),
        _column(
            "device_model",
            ColumnType.STRING,
            True,
            ColumnRole.PROVENANCE,
            "StepCount[].raw_steps[].device_model",
            description="Optional source device model.",
        ),
        _column(
            "device_type",
            ColumnType.STRING,
            True,
            ColumnRole.PROVENANCE,
            "StepCount[].raw_steps[].device_type",
            description="Optional source device type.",
        ),
        _column(
            "device_uid",
            ColumnType.STRING,
            True,
            ColumnRole.PROVENANCE,
            "StepCount[].raw_steps[].device_uid",
            description="Optional source device identifier.",
        ),
        _column(
            "collecting_method_version",
            ColumnType.STRING,
            False,
            ColumnRole.SOURCE_VERSION,
            "collecting_method_version",
            description="Exporter collecting-method version.",
        ),
        _column(
            "outer_id",
            ColumnType.STRING,
            False,
            ColumnRole.LINEAGE,
            "id",
            description="Outer export row identifier.",
        ),
        _column(
            "payload_index",
            ColumnType.INTEGER,
            False,
            ColumnRole.LINEAGE,
            "StepCount alternative index",
            description=(
                "Zero-based aggregation-alternative index; fixed to zero for the legacy scalar."
            ),
        ),
    )
    omissions = (
        *_COMMON_PATH_AND_LINEAGE_OMISSIONS,
        _omitted(
            "duration",
            OmissionKind.EXACT_DERIVATION,
            "Version 2.0 duration is exactly reconstructable from raw interval boundaries.",
            reconstructable_from=("start_date", "end_date"),
        ),
        _omitted(
            "data_type_name",
            OmissionKind.REGISTRY_CONSTANT,
            "All observed version 2.0 alternatives use the registered Google Fit data type.",
            retained_in="STEP_COUNT_DATA_TYPE_NAME",
        ),
        _omitted(
            "data_source_id",
            OmissionKind.REGISTRY_CONSTANT,
            "All observed version 2.0 alternatives use the registered derived source ID.",
            retained_in="STEP_COUNT_DATA_SOURCE_ID",
        ),
    )
    return NativeFeatureSpec(
        name="StepCount",
        filename="StepCount.csv",
        source_feature="StepCount",
        family=NativeFeatureFamily.STEP_COUNT,
        supported_versions=("1.0", "2.0"),
        columns=columns,
        measurement_columns=("value", "raw_value"),
        state_columns=(),
        occurrence_identity_columns=("outer_id", "payload_index"),
        semantic_identity_by_version=(
            VersionIdentitySpec(
                version="1.0",
                maturity=IdentityMaturity.SOURCE_OCCURRENCE_ONLY,
                semantic_columns=(),
                rationale=(
                    "Version 1.0 exposes only a scalar and outer timestamp; repeated "
                    "counts are not semantic duplicates."
                ),
            ),
            VersionIdentitySpec(
                version="2.0",
                maturity=IdentityMaturity.SUPPORTED,
                semantic_columns=(
                    "collecting_method_version",
                    "aggregation_source",
                    "start_date",
                    "end_date",
                    "origin_data_source_id",
                ),
                rationale=(
                    "Version 2.0 identity requires the named alternative, raw interval, "
                    "and origin stream. Summary-value equality is insufficient."
                ),
            ),
        ),
        sort_columns=(
            "outer_datetime",
            "summary_date",
            "start_date",
            "aggregation_source",
            "outer_id",
            "payload_index",
        ),
        validation_rules=(
            "outer_datetime_is_parseable_utc",
            "version_1_has_scalar_value_and_no_invented_interval",
            "version_2_has_summary_date_raw_interval_and_named_alternative",
            "version_2_data_type_name_matches_registry_constant",
            "version_2_data_source_id_matches_registry_constant",
            "one_native_row_is_emitted_per_aggregation_alternative",
            "value_and_raw_value_are_preserved_as_distinct_source_concepts",
            "no_preferred_alternative_is_selected",
        ),
        omitted_fields=omissions,
    )


_SOURCE_FEATURE_VERSION_SPECS = (
    SourceFeatureVersionSpec(
        feature="Activities",
        version="1.0",
        envelope=PayloadEnvelope.LIST,
        decoder=DecoderKind.ACTIVITY_ITEMS,
        minimum_items=1,
        maximum_items=None,
        accepted_item_types=("dict",),
        expected_source_fields=(
            "activity_name",
            "start",
            "end",
            "calories",
            "distance",
            "quantity",
            "tracked",
            "device",
            "source_name",
            "source_id",
        ),
        malformed_items_are_diagnostics=True,
        notes=("Each item is routed exactly once; no Activities.csv is written.",),
    ),
    SourceFeatureVersionSpec(
        feature="Calories",
        version="1.0",
        envelope=PayloadEnvelope.LIST,
        decoder=DecoderKind.INTERVAL_TOTAL,
        minimum_items=1,
        maximum_items=1,
        accepted_item_types=("dict",),
        expected_source_fields=("start_date", "end_date", "calorie", "day"),
        malformed_items_are_diagnostics=True,
    ),
    SourceFeatureVersionSpec(
        feature="Calories",
        version="2.0",
        envelope=PayloadEnvelope.LIST,
        decoder=DecoderKind.INTERVAL_TOTAL,
        minimum_items=1,
        maximum_items=1,
        accepted_item_types=("dict",),
        expected_source_fields=("start_date", "end_date", "calorie", "day"),
        malformed_items_are_diagnostics=True,
    ),
    SourceFeatureVersionSpec(
        feature="Distance",
        version="1.0",
        envelope=PayloadEnvelope.LIST,
        decoder=DecoderKind.INTERVAL_TOTAL,
        minimum_items=1,
        maximum_items=1,
        accepted_item_types=("dict",),
        expected_source_fields=("start_date", "end_date", "distance", "day"),
        malformed_items_are_diagnostics=True,
    ),
    SourceFeatureVersionSpec(
        feature="Distance",
        version="2.0",
        envelope=PayloadEnvelope.LIST,
        decoder=DecoderKind.INTERVAL_TOTAL,
        minimum_items=1,
        maximum_items=1,
        accepted_item_types=("dict",),
        expected_source_fields=("start_date", "end_date", "distance", "day"),
        malformed_items_are_diagnostics=True,
    ),
    SourceFeatureVersionSpec(
        feature="Height",
        version="2.0",
        envelope=PayloadEnvelope.LIST,
        decoder=DecoderKind.ANTHROPOMETRIC,
        minimum_items=1,
        maximum_items=1,
        accepted_item_types=("dict",),
        expected_source_fields=("start_date", "end_date", "value", "day", "added_by"),
        malformed_items_are_diagnostics=True,
    ),
    SourceFeatureVersionSpec(
        feature="StepCount",
        version="1.0",
        envelope=PayloadEnvelope.SCALAR_INTEGER,
        decoder=DecoderKind.STEP_COUNT_LEGACY,
        minimum_items=1,
        maximum_items=1,
        accepted_item_types=("integer",),
        expected_source_fields=(),
        malformed_items_are_diagnostics=True,
        notes=("Scalar values are expected and are not payload-shape warnings.",),
    ),
    SourceFeatureVersionSpec(
        feature="StepCount",
        version="2.0",
        envelope=PayloadEnvelope.LIST,
        decoder=DecoderKind.STEP_COUNT_ALTERNATIVES,
        minimum_items=1,
        maximum_items=3,
        accepted_item_types=("dict",),
        expected_source_fields=("source", "steps", "raw_steps"),
        malformed_items_are_diagnostics=True,
        notes=("Emit one native row per named aggregation alternative.",),
    ),
    SourceFeatureVersionSpec(
        feature="Weight",
        version="1.0",
        envelope=PayloadEnvelope.DICT,
        decoder=DecoderKind.ANTHROPOMETRIC,
        minimum_items=1,
        maximum_items=1,
        accepted_item_types=("dict",),
        expected_source_fields=("start_date", "end_date", "value", "day", "added_by"),
        malformed_items_are_diagnostics=True,
    ),
    SourceFeatureVersionSpec(
        feature="Weight",
        version="2.0",
        envelope=PayloadEnvelope.LIST,
        decoder=DecoderKind.ANTHROPOMETRIC,
        minimum_items=1,
        maximum_items=1,
        accepted_item_types=("dict",),
        expected_source_fields=("start_date", "end_date", "value", "day", "added_by"),
        malformed_items_are_diagnostics=True,
        notes=("Non-dict/null items are malformed diagnostics and do not become numeric rows.",),
    ),
)

SOURCE_FEATURE_VERSION_SPECS: tuple[SourceFeatureVersionSpec, ...] = tuple(
    sorted(_SOURCE_FEATURE_VERSION_SPECS, key=lambda spec: spec.key)
)
SOURCE_FEATURE_VERSION_BY_KEY: Mapping[
    tuple[str, str], SourceFeatureVersionSpec
] = MappingProxyType({spec.key: spec for spec in SOURCE_FEATURE_VERSION_SPECS})


def _build_feature_specs() -> dict[str, NativeFeatureSpec]:
    result = _build_activity_specs()
    result["Calories"] = _interval_total_spec("Calories", "calorie")
    result["Distance"] = _interval_total_spec("Distance", "distance")
    result["Height"] = _anthropometric_spec("Height", ("2.0",))
    result["Weight"] = _anthropometric_spec("Weight", ("1.0", "2.0"))
    result["StepCount"] = _step_count_spec()
    return dict(sorted(result.items()))


_NATIVE_FEATURE_SPECS = _build_feature_specs()
NATIVE_FEATURE_SPECS: Mapping[str, NativeFeatureSpec] = MappingProxyType(
    _NATIVE_FEATURE_SPECS
)


def normalize_source_label(value: object) -> str:
    """Normalize an outer source label without conflating distinct alphanumerics."""

    return "".join(character for character in str(value).casefold() if character.isalnum())


def get_activity_route(raw_label: str) -> ActivityRoute:
    try:
        return ACTIVITY_ROUTE_BY_LABEL[raw_label]
    except KeyError as exc:
        raise KeyError(f"Unregistered Google Fitness activity label: {raw_label!r}") from exc


def get_feature_spec(name: str) -> NativeFeatureSpec:
    try:
        return NATIVE_FEATURE_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"Unregistered Google Fitness native feature: {name!r}") from exc


def get_source_feature_version_spec(feature: str, version: str) -> SourceFeatureVersionSpec:
    try:
        return SOURCE_FEATURE_VERSION_BY_KEY[(feature, version)]
    except KeyError as exc:
        raise KeyError(
            f"Unsupported Google Fitness source feature/version: {feature!r}/{version!r}"
        ) from exc


def activity_feature_names() -> tuple[str, ...]:
    return tuple(
        sorted({route.native_feature for route in ACTIVITY_ROUTES})
    )


def feature_names() -> tuple[str, ...]:
    return tuple(NATIVE_FEATURE_SPECS)


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return tuple(sorted(duplicate))


def validate_registry() -> RegistryValidation:
    issues: list[str] = []

    if normalize_source_label("GoogleFitness") != NORMALIZED_RAW_SOURCE:
        issues.append("normalized raw source key does not match GoogleFitness")

    if len(ACTIVITY_ROUTES) != 66:
        issues.append(f"expected 66 activity routes, found {len(ACTIVITY_ROUTES)}")
    duplicate_labels = _duplicates(route.raw_label for route in ACTIVITY_ROUTES)
    if duplicate_labels:
        issues.append(f"duplicate activity labels: {duplicate_labels}")

    destinations: dict[str, list[ActivityRoute]] = {}
    for route in ACTIVITY_ROUTES:
        destinations.setdefault(route.native_feature, []).append(route)
        if tuple(name for name in ACTIVITY_MEASUREMENT_ORDER if name in route.measurements) != route.measurements:
            issues.append(f"activity measurement order is invalid for {route.raw_label}")
        if set(route.measurements) - set(ACTIVITY_MEASUREMENT_ORDER):
            issues.append(f"unknown activity measurement for {route.raw_label}")
        expected_state = SLEEP_STATE_MAPPING.get(route.raw_label)
        if route.native_state != expected_state:
            issues.append(f"sleep-state mapping mismatch for {route.raw_label}")
        if expected_state is not None and route.native_feature != "Sleep":
            issues.append(f"sleep route does not target Sleep: {route.raw_label}")

    if len(destinations) != 62:
        issues.append(f"expected 62 activity destinations, found {len(destinations)}")
    collisions = {
        name: tuple(sorted(route.raw_label for route in routes))
        for name, routes in destinations.items()
        if len(routes) > 1
    }
    expected_sleep_collision = {"Sleep": tuple(sorted(SLEEP_STATE_MAPPING))}
    if collisions != expected_sleep_collision:
        issues.append(f"unexpected activity destination collisions: {collisions}")

    fallback_columns = set(UNREGISTERED_ACTIVITY_POLICY.column_names)
    required_fallback_columns = {
        "start_date",
        "end_date",
        "raw_activity_name",
        "tracked",
        "device",
        "source_name",
        "collecting_method_version",
        "outer_id",
        "payload_index",
    }
    if not required_fallback_columns.issubset(fallback_columns):
        issues.append("unregistered activity fallback schema is incomplete")
    if UNREGISTERED_ACTIVITY_POLICY.native_feature in NATIVE_FEATURE_SPECS:
        issues.append("unregistered activity fallback must not be a registered feature")

    if len(SOURCE_FEATURE_VERSION_SPECS) != 10:
        issues.append(
            "expected 10 source feature/version contracts, found "
            f"{len(SOURCE_FEATURE_VERSION_SPECS)}"
        )
    duplicate_contracts = _duplicates(
        f"{spec.feature}\0{spec.version}" for spec in SOURCE_FEATURE_VERSION_SPECS
    )
    if duplicate_contracts:
        issues.append(f"duplicate source feature/version contracts: {duplicate_contracts}")

    if len(NATIVE_FEATURE_SPECS) != 67:
        issues.append(f"expected 67 native feature specs, found {len(NATIVE_FEATURE_SPECS)}")
    if "Activities" in NATIVE_FEATURE_SPECS:
        issues.append("Activities must not be a native output feature")

    duplicate_filenames = _duplicates(spec.filename for spec in NATIVE_FEATURE_SPECS.values())
    if duplicate_filenames:
        issues.append(f"duplicate native filenames: {duplicate_filenames}")

    forbidden_repeated_columns = {
        "participant_id",
        "feature",
        "data_source",
        "raw_data_source",
        "raw_outer_feature",
        "raw_activity_name",
        "duration",
        "created_at",
        "updated_at",
        "source_month",
        "source_filename",
        "outer_row_number",
    }

    for name, spec in NATIVE_FEATURE_SPECS.items():
        columns = spec.column_names
        column_set = set(columns)
        if spec.filename != f"{name}.csv":
            issues.append(f"filename/name mismatch for {name}")
        duplicate_columns = _duplicates(columns)
        if duplicate_columns:
            issues.append(f"duplicate columns for {name}: {duplicate_columns}")
        leaked = sorted(column_set & forbidden_repeated_columns)
        if leaked:
            issues.append(f"redundant columns in {name}: {leaked}")
        referenced = (
            set(spec.measurement_columns)
            | set(spec.state_columns)
            | set(spec.occurrence_identity_columns)
            | set(spec.sort_columns)
        )
        for identity in spec.semantic_identity_by_version:
            referenced.update(identity.semantic_columns)
            if (
                identity.maturity is IdentityMaturity.SOURCE_OCCURRENCE_ONLY
                and identity.semantic_columns
            ):
                issues.append(
                    f"source-occurrence-only identity exposes semantic columns for {name}/{identity.version}"
                )
        missing = sorted(referenced - column_set)
        if missing:
            issues.append(f"schema references missing columns for {name}: {missing}")
        if set(spec.supported_versions) != {
            identity.version for identity in spec.semantic_identity_by_version
        }:
            issues.append(f"identity versions do not match supported versions for {name}")
        if not spec.validation_rules:
            issues.append(f"missing validation rules for {name}")
        if spec.deduplicate_across_versions:
            issues.append(f"cross-version deduplication is prohibited for {name}")

    for destination, routes in destinations.items():
        spec = NATIVE_FEATURE_SPECS.get(destination)
        if spec is None:
            issues.append(f"missing native spec for activity destination {destination}")
            continue
        if spec.source_feature != ACTIVITY_OUTER_FEATURE:
            issues.append(f"activity spec has wrong source feature: {destination}")
        expected_measurements = tuple(
            name
            for name in ACTIVITY_MEASUREMENT_ORDER
            if any(name in route.measurements for route in routes)
        )
        if spec.measurement_columns != expected_measurements:
            issues.append(f"activity measurement schema mismatch for {destination}")
        expected_source_id = any(route.include_source_id for route in routes)
        if ("source_id" in spec.column_names) != expected_source_id:
            issues.append(f"activity source_id schema mismatch for {destination}")
        if destination == "Sleep":
            if spec.state_columns != ("value",):
                issues.append("Sleep must have exactly one state column named value")
        elif spec.state_columns:
            issues.append(f"non-sleep activity has state columns: {destination}")

    step = NATIVE_FEATURE_SPECS.get("StepCount")
    if step is not None:
        expected_step_columns = (
            "outer_datetime",
            "summary_date",
            "start_date",
            "end_date",
            "value",
            "raw_value",
            "aggregation_source",
            "origin_data_source_id",
            "app_name",
            "app_package_name",
            "device_manufacturer",
            "device_model",
            "device_type",
            "device_uid",
            "collecting_method_version",
            "outer_id",
            "payload_index",
        )
        if step.column_names != expected_step_columns:
            issues.append("StepCount union schema does not match the accepted contract")
        if step.identity_for_version("1.0").maturity is not IdentityMaturity.SOURCE_OCCURRENCE_ONLY:
            issues.append("StepCount 1.0 must remain source-occurrence-only")

    actual_fingerprint = registry_fingerprint()
    if actual_fingerprint != EXPECTED_REGISTRY_FINGERPRINT:
        issues.append(
            "registry fingerprint mismatch: "
            f"expected {EXPECTED_REGISTRY_FINGERPRINT}, observed {actual_fingerprint}"
        )

    return RegistryValidation(tuple(sorted(set(issues))))


def _base_registry_payload() -> dict[str, object]:
    return {
        "source_key": SOURCE_KEY,
        "normalized_raw_source": NORMALIZED_RAW_SOURCE,
        "raw_source_labels": sorted(RAW_SOURCE_LABELS),
        "registry_version": REGISTRY_VERSION,
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "outer_schema_columns": list(OUTER_SCHEMA_COLUMNS),
        "step_count_constants": {
            "data_type_name": STEP_COUNT_DATA_TYPE_NAME,
            "data_source_id": STEP_COUNT_DATA_SOURCE_ID,
        },
        "sleep_state_mapping": dict(sorted(SLEEP_STATE_MAPPING.items())),
        "activity_routes": [route.as_dict() for route in ACTIVITY_ROUTES],
        "unregistered_activity_policy": UNREGISTERED_ACTIVITY_POLICY.as_dict(),
        "source_feature_versions": [
            spec.as_dict() for spec in SOURCE_FEATURE_VERSION_SPECS
        ],
        "native_features": [
            NATIVE_FEATURE_SPECS[name].as_dict() for name in NATIVE_FEATURE_SPECS
        ],
    }


def registry_fingerprint() -> str:
    payload = _base_registry_payload()
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(rendered).hexdigest()


def registry_payload() -> dict[str, object]:
    payload = _base_registry_payload()
    payload["registry_fingerprint"] = registry_fingerprint()
    payload["validation"] = validate_registry().as_dict()
    payload["counts"] = {
        "activity_routes": len(ACTIVITY_ROUTES),
        "activity_destinations": len(activity_feature_names()),
        "native_features": len(NATIVE_FEATURE_SPECS),
        "source_feature_versions": len(SOURCE_FEATURE_VERSION_SPECS),
    }
    return payload


_VALIDATION = validate_registry()
if not _VALIDATION.valid:
    raise RuntimeError(
        "Invalid Google Fitness native registry: " + "; ".join(_VALIDATION.issues)
    )
