"""
Wearable Data Processing and Modeling project
Typed, source-specific policy contracts for Google Fitness curation. CWP1 is declaration-only: these objects
describe what a later curation audit and engine are allowed to do. They deliberately contain no CSV I/O, no row
mutation, and no dependency on Apple HealthKit scientific policies.
"""


from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any


class StringEnum(str, Enum):
    """String-valued enum with stable JSON serialization."""


class PolicyMaturity(StringEnum):
    REVIEWED = "reviewed"
    PROVISIONAL = "provisional"
    EXPERIMENTAL = "experimental"
    UNKNOWN = "unknown"


class EvidenceGrade(StringEnum):
    """Strength of evidence for the complete executable interpretation."""

    A_DIRECT = "A_direct"
    B_CONVERGENT = "B_convergent"
    C_SUGGESTIVE = "C_suggestive"
    D_UNRESOLVED = "D_unresolved"


class PolicyExecutionMode(StringEnum):
    """How a future engine may use a policy decision."""

    REVIEWED_EXECUTION = "reviewed_execution"
    SOURCE_SPECIFIC_EXECUTION = "source_specific_execution"
    CONSERVATIVE_ANNOTATION = "conservative_annotation"
    BLOCKED_DERIVATION = "blocked_derivation"


class EventKind(StringEnum):
    ACTIVITY_INTERVAL = "activity_interval"
    STATE_INTERVAL = "state_interval"
    INTERVAL_TOTAL = "interval_total"
    POINT_OR_INTERVAL_SCALAR = "point_or_interval_scalar"
    MULTI_REPRESENTATION_COUNT = "multi_representation_count"
    UNKNOWN = "unknown"


class MeasurementKind(StringEnum):
    EXTENSIVE_TOTAL = "extensive_total"
    POINT_SCALAR = "point_scalar"
    COUNT = "count"
    CONTEXTUAL_SOURCE_VALUE = "contextual_source_value"


class RawUnitStatus(StringEnum):
    PAYLOAD_EXPLICIT = "payload_explicit"
    VERIFIED_EXPORTER_MAPPING = "verified_exporter_mapping"
    SOURCE_CONVENTION = "source_convention"
    EXPORTER_MAPPING_PENDING = "exporter_mapping_pending"
    COHORT_SUGGESTIVE = "cohort_suggestive"
    UNRESOLVED = "unresolved"
    NOT_APPLICABLE = "not_applicable"


class CanonicalizationStrategy(StringEnum):
    IDENTITY = "identity"
    WITHHOLD_IF_INVALID = "withhold_if_invalid"
    WITHHOLD = "withhold"
    NOT_APPLICABLE = "not_applicable"


class DateAnchor(StringEnum):
    START = "start"
    END = "end"
    OUTER = "outer"
    MULTIPLE_CONTEXTS = "multiple_contexts"
    NONE = "none"


class SourceDayLabelTreatment(StringEnum):
    CONTEXT_ONLY = "context_only"
    NOT_PRESENT = "not_present"


class AcquisitionMethod(StringEnum):
    DEVICE_MEASUREMENT = "device_measurement"
    DEVICE_ESTIMATE = "device_estimate"
    USER_ENTERED = "user_entered"
    THIRD_PARTY_IMPORT = "third_party_import"
    APPLICATION_SUMMARY = "application_summary"
    HEALTH_IMPORT_UNKNOWN = "health_import_unknown"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class DerivationMethod(StringEnum):
    RAW = "raw"
    ACTIVITY_CLASSIFICATION = "activity_classification"
    ESTIMATED = "estimated"
    MERGED = "merged"
    LEGACY_SUMMARY = "legacy_summary"
    APPLICATION_SUMMARY = "application_summary"
    THIRD_PARTY_AGGREGATE = "third_party_aggregate"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class TrackingStatus(StringEnum):
    TRACKED = "tracked"
    UNTRACKED = "untracked"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class StepCountRole(StringEnum):
    LEGACY_V1 = "legacy_v1"
    FIT_APP_PREFERRED = "fit_app_preferred"
    MERGE_FALLBACK = "merge_fallback"
    THIRD_PARTY_FALLBACK = "third_party_fallback"
    PARALLEL_NONPREFERRED = "parallel_nonpreferred"
    AMBIGUOUS_PARALLEL = "ambiguous_parallel"
    NOT_APPLICABLE = "not_applicable"


class CurationStatus(StringEnum):
    PASS = "pass"
    REVIEW = "review"
    EXCLUDE_DEFAULT = "exclude_default"


class InclusionPolicy(StringEnum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    RULE_DRIVEN = "rule_driven"
    GROUP_PREFERENCE = "group_preference"


class RuleClass(StringEnum):
    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    UNIT = "unit"
    QUALITY = "quality"
    PROVENANCE = "provenance"
    GROUP_SELECTION = "group_selection"
    SOURCE = "source"


class Severity(StringEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class StatusEffect(StringEnum):
    NONE = "none"
    REVIEW = "review"
    EXCLUDE_DEFAULT = "exclude_default"


class InclusionEffect(StringEnum):
    KEEP = "keep"
    EXCLUDE_DEFAULT = "exclude_default"
    INCLUDE_DEFAULT = "include_default"
    GROUP_DECISION = "group_decision"


class GroupPreferenceStrategy(StringEnum):
    NONE = "none"
    STEP_COUNT_V2_SOURCE_PREFERENCE = "step_count_v2_source_preference"
    SLEEP_OVERLAP_CONTEXT = "sleep_overlap_context"


class DecisionStatus(StringEnum):
    ACCEPTED = "accepted"
    PROVISIONAL = "provisional"
    DEFERRED = "deferred"
    REJECTED = "rejected"


class ResamplingSupport(StringEnum):
    SUPPORTED_LATER = "supported_later"
    BLOCKED_PENDING_EVIDENCE = "blocked_pending_evidence"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class NativeSchemaPolicy:
    feature: str
    native_columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    measurement_columns: tuple[str, ...]
    state_columns: tuple[str, ...]
    state_vocabulary: tuple[str, ...]
    supported_versions: tuple[str, ...]
    row_meaning: str


@dataclass(frozen=True, slots=True)
class VersionUnitPolicy:
    versions: tuple[str, ...]
    raw_unit_status: RawUnitStatus
    raw_unit: str | None
    canonical_unit: str | None
    evidence_grade: EvidenceGrade
    execution_mode: PolicyExecutionMode
    canonicalization: CanonicalizationStrategy
    evidence_refs: tuple[str, ...]
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MeasurementPolicy:
    column: str
    kind: MeasurementKind
    required: bool
    raw_projection_enabled: bool
    canonical_projection_enabled: bool
    unit_by_version: tuple[VersionUnitPolicy, ...]
    description: str

    def policy_for_version(self, version: str) -> VersionUnitPolicy:
        for policy in self.unit_by_version:
            if version in policy.versions:
                return policy
        raise KeyError(f"No unit policy for {self.column!r} at version {version!r}")


@dataclass(frozen=True, slots=True)
class TemporalPolicy:
    start_column: str | None
    end_column: str | None
    outer_datetime_column: str | None
    summary_date_column: str | None
    date_anchor: DateAnchor
    source_day_label_column: str | None
    source_day_label_treatment: SourceDayLabelTreatment
    preserve_all_boundaries: bool
    canonical_date_derivation_allowed: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProvenancePolicy:
    raw_columns: tuple[str, ...]
    output_dimensions: tuple[str, ...]
    classifier_id: str
    evidence_refs: tuple[str, ...]
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GroupPolicy:
    group_columns: tuple[str, ...]
    strategy: GroupPreferenceStrategy
    role_column: str | None
    disagreement_rule_id: str | None
    applies_to_versions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CurationPolicy:
    default_status: CurationStatus
    default_inclusion: InclusionPolicy
    rule_ids: tuple[str, ...]
    flags_column: str
    status_column: str
    inclusion_column: str


@dataclass(frozen=True, slots=True)
class OutputPolicy:
    """
    Physical and virtual curation-output contract. ``materialized_columns`` are appended to the participant feature
    CSV. ``virtual_columns`` remain reproducible policy projections exposed through the root curation contract and
    future DataLoaders, but are not repeated on every row.  ``sparse_columns`` use documented blank/absent defaults;
    ``fixed_columns`` are physically present for every row of the feature.
    """

    preserve_native_prefix: bool
    preserve_row_order: bool
    preserve_row_count: bool
    materialized_columns: tuple[str, ...]
    virtual_columns: tuple[str, ...]
    sparse_columns: tuple[str, ...]
    fixed_columns: tuple[str, ...]

    @property
    def derived_columns(self) -> tuple[str, ...]:
        """Compatibility alias for physically materialized columns."""

        return self.materialized_columns


@dataclass(frozen=True, slots=True)
class ResamplingPolicy:
    support: ResamplingSupport
    strategy: str
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PolicyCalibration:
    maturity: PolicyMaturity
    evidence_grade: EvidenceGrade
    execution_mode: PolicyExecutionMode
    required_audits: tuple[str, ...]
    decision_ids: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FeaturePolicy:
    feature: str
    source_key: str
    event_kind: EventKind
    native_schema: NativeSchemaPolicy
    measurements: tuple[MeasurementPolicy, ...]
    temporal: TemporalPolicy
    provenance: ProvenancePolicy
    grouping: GroupPolicy
    curation: CurationPolicy
    output: OutputPolicy
    resampling: ResamplingPolicy
    calibration: PolicyCalibration
    evidence_refs: tuple[str, ...]
    unknown_fallback: bool = False

    @property
    def measurement_columns(self) -> tuple[str, ...]:
        return tuple(item.column for item in self.measurements)

    @property
    def materialized_columns(self) -> tuple[str, ...]:
        return self.output.materialized_columns

    @property
    def derived_columns(self) -> tuple[str, ...]:
        """Compatibility alias for ``materialized_columns``."""

        return self.materialized_columns

    @property
    def virtual_columns(self) -> tuple[str, ...]:
        return self.output.virtual_columns


@dataclass(frozen=True, slots=True)
class RegistryValidationResult:
    issues: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, object]:
        return {"valid": self.valid, "issues": list(self.issues)}


def to_primitive(value: Any) -> Any:
    """Convert immutable policy objects to deterministic JSON-compatible values."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: to_primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [to_primitive(item) for item in value]
    return value
