"""
Wearable Data Processing and Modeling project
Typed contracts for feature curation policies. This module contains declarations only. The 0.2.0a1 release
does not curate participant files; it establishes the complete, validated policy contract that later
curation and resampling implementations must execute.
"""


from __future__ import annotations
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Any


class StringEnum(str, Enum):
    """String-valued enum with stable JSON serialization."""


class PolicyMaturity(StringEnum):
    REVIEWED = "reviewed"
    PROVISIONAL = "provisional"
    EXPERIMENTAL = "experimental"
    UNKNOWN = "unknown"


class EventKind(StringEnum):
    POINT = "point"
    POINT_OR_INTERVAL = "point_or_interval"
    INTERVAL = "interval"
    STATE_INTERVAL = "state_interval"
    DAILY_SUMMARY = "daily_summary"
    LONG_SUMMARY_INTERVAL = "long_summary_interval"
    WAVEFORM = "waveform"
    DURATION_EVENT = "duration_event"
    NUTRITION_EVENT = "nutrition_event"
    UNKNOWN = "unknown"


class MeasurementKind(StringEnum):
    EXTENSIVE_TOTAL = "extensive_total"
    INTENSIVE_VALUE = "intensive_value"
    RATIO = "ratio"
    CATEGORICAL_STATE = "categorical_state"
    MULTIVARIATE_POINT = "multivariate_point"
    MULTIVARIATE_SUMMARY = "multivariate_summary"
    SUMMARY_STATISTIC = "summary_statistic"
    EVENT_AMOUNT = "event_amount"
    SIGNAL = "signal"
    DURATION = "duration"
    UNKNOWN = "unknown"


class DurationModel(StringEnum):
    ZERO_DURATION_POINT = "zero_duration_point"
    ZERO_OR_EXPLICIT_INTERVAL = "zero_or_explicit_interval"
    EXPLICIT_INTERVAL = "explicit_interval"
    OUTER_BUCKET_ONLY = "outer_bucket_only"
    INTERNAL_SIGNAL_CLOCK = "internal_signal_clock"
    DATE_COMPONENT_SUMMARY = "date_component_summary"
    UNKNOWN = "unknown"


class IntervalClosure(StringEnum):
    HALF_OPEN = "[start,end)"
    POINT = "point"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class DateAnchor(StringEnum):
    EVENT_START_UTC = "event_start_utc"
    EVENT_LOCAL_DATE = "event_local_date"
    OUTER_UTC_BUCKET = "outer_utc_bucket"
    HEALTHKIT_DATE_COMPONENTS = "healthkit_date_components"
    INTERNAL_SIGNAL_START = "internal_signal_start"
    UNKNOWN = "unknown"


class MissingFieldAction(StringEnum):
    REVIEW = "review"
    EXCLUDE_DEFAULT = "exclude_default"
    FAIL_CURATION = "fail_curation"
    COPY_UNCHANGED = "copy_unchanged"


class RawUnitStatus(StringEnum):
    EXPLICIT = "explicit"
    SOURCE_CONVENTION = "source_convention"
    INFERRED_HIGH_CONFIDENCE = "inferred_high_confidence"
    INFERRED_PROVISIONAL = "inferred_provisional"
    PARTICIPANT_SOURCE_INFERENCE = "participant_source_inference"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class ConversionConfidence(StringEnum):
    EXACT = "exact"
    HIGH = "high"
    PROVISIONAL = "provisional"
    NONE = "none"


class UnresolvedUnitAction(StringEnum):
    RETAIN_RAW_ONLY = "retain_raw_only"
    RETAIN_RAW_AND_CANDIDATES = "retain_raw_and_candidates"
    EXCLUDE_FROM_DEFAULT_CANONICAL_ANALYSIS = "exclude_from_default_canonical_analysis"
    FAIL_POLICY = "fail_policy"
    NOT_APPLICABLE = "not_applicable"


class AcquisitionMethod(StringEnum):
    DEVICE_MEASUREMENT = "device_measurement"
    DEVICE_ESTIMATE = "device_estimate"
    USER_ENTERED = "user_entered"
    THIRD_PARTY_IMPORT = "third_party_import"
    CALCULATOR_ESTIMATE = "calculator_estimate"
    APPLICATION_SUMMARY = "application_summary"
    HEALTH_IMPORT_UNKNOWN = "health_import_unknown"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class UserEnteredHandling(StringEnum):
    ACCEPT = "accept"
    ACCEPT_AND_FLAG = "accept_and_flag"
    USE_AS_UNIT_EVIDENCE = "use_as_unit_evidence"
    EXCLUDE_FROM_DEVICE_ONLY_VIEW = "exclude_from_device_only_view"
    EXCLUDE_DEFAULT = "exclude_default"
    NOT_APPLICABLE = "not_applicable"


class ConflictBehavior(StringEnum):
    PRESERVE_ALL_REVIEW = "preserve_all_review"
    PRESERVE_ALL_EXCLUDE_DEFAULT = "preserve_all_exclude_default"
    SELECT_REVIEWED_CANONICAL = "select_reviewed_canonical"
    NOT_APPLICABLE = "not_applicable"


class CurationStatus(StringEnum):
    PASS = "pass"
    REVIEW = "review"
    EXCLUDE_DEFAULT = "exclude_default"


class InclusionPolicy(StringEnum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


class RuleClass(StringEnum):
    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    UNIT = "unit"
    SOURCE = "source"
    PROVENANCE = "provenance"
    PHYSIOLOGICAL_SCREENING = "physiological_screening"
    CROSS_FEATURE = "cross_feature"
    SIGNAL_INTEGRITY = "signal_integrity"


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
    NONE = "none"


class ResamplingSupport(StringEnum):
    SUPPORTED = "supported"
    OPTIONAL = "optional"
    UNSUPPORTED = "unsupported"
    WAVEFORM_SPECIFIC_ONLY = "waveform_specific_only"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class IdentityPolicy:
    canonical_name: str
    aliases: tuple[str, ...] = ()
    data_sources: tuple[str, ...] = ("AppleHealthkit",)
    collecting_method_versions: tuple[str, ...] = ("1.0", "2.0")
    policy_version: str = "1.0.0-alpha1"
    maturity: PolicyMaturity = PolicyMaturity.REVIEWED
    cohort_observed: bool = True


@dataclass(frozen=True, slots=True)
class MeasurementField:
    column: str
    role: str
    value_kind: str = "numeric"
    required: bool = True


@dataclass(frozen=True, slots=True)
class SchemaPolicy:
    feature_family: str
    accepted_payload_shapes: tuple[str, ...]
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    measurements: tuple[MeasurementField, ...]
    categorical_columns: tuple[str, ...] = ()
    context_columns: tuple[str, ...] = ()
    waveform_columns: tuple[str, ...] = ()
    missing_measurement_action: MissingFieldAction = MissingFieldAction.REVIEW


@dataclass(frozen=True, slots=True)
class SemanticPolicy:
    event_kind: EventKind
    measurement_kind: MeasurementKind
    duration_model: DurationModel
    interval_closure: IntervalClosure
    start_time_meaning: str
    end_time_meaning: str
    native_resolution: str
    date_anchor: DateAnchor
    local_time_strategy: str = "preserve_utc_offset_and_iana_when_available"


@dataclass(frozen=True, slots=True)
class MeasurementUnitPolicy:
    measurement: str
    raw_unit_status: RawUnitStatus
    raw_unit: str | None
    raw_unit_candidates: tuple[str, ...]
    canonical_unit: str | None
    unit_resolution_strategy: str
    conversion_rule: str
    conversion_confidence: ConversionConfidence
    unresolved_action: UnresolvedUnitAction
    evidence_refs: tuple[str, ...]
    output_precision: int | None = None


@dataclass(frozen=True, slots=True)
class UnitPolicy:
    measurements: tuple[MeasurementUnitPolicy, ...]


@dataclass(frozen=True, slots=True)
class SourceRule:
    rule_id: str
    source_id: str | None = None
    source_name_contains: str | None = None
    metadata_equals: tuple[tuple[str, str], ...] = ()
    acquisition_method: AcquisitionMethod | None = None
    status_effect: CurationStatus | None = None
    inclusion_effect: InclusionPolicy | None = None
    evidence_refs: tuple[str, ...] = ()
    note: str | None = None


@dataclass(frozen=True, slots=True)
class ProvenancePolicy:
    acquisition_method_strategy: str
    user_entered_handling: UserEnteredHandling
    source_epoch_strategy: str
    source_priority_strategy: str
    retain_context_columns: tuple[str, ...]
    source_rules: tuple[SourceRule, ...] = ()


@dataclass(frozen=True, slots=True)
class ReconciliationPolicy:
    occurrence_identity_strategy: str
    exact_duplicate_strategy: str
    revision_strategy: str
    conflict_behavior: ConflictBehavior
    merge_safe: bool
    note: str | None = None


@dataclass(frozen=True, slots=True)
class RuleDefinition:
    rule_id: str
    rule_class: RuleClass
    strategy: str
    severity: Severity
    flag: str
    status_effect: StatusEffect
    inclusion_effect: InclusionEffect
    recomputable: bool
    evidence_refs: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True, slots=True)
class CurationPolicy:
    rule_ids: tuple[str, ...]
    default_status: CurationStatus
    default_inclusion: InclusionPolicy


@dataclass(frozen=True, slots=True)
class CrossFeaturePolicy:
    rule_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OutputPolicy:
    primary_measurement_columns: tuple[str, ...]
    retained_provenance_columns: tuple[str, ...]
    retained_context_columns: tuple[str, ...]
    derived_columns: tuple[str, ...]
    sparse_columns: tuple[str, ...]
    column_order_prefix: tuple[str, ...]
    numeric_dtypes: tuple[tuple[str, str], ...] = ()
    categorical_dtypes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ResamplingPolicy:
    support: ResamplingSupport
    default_enabled: bool
    strategy: str
    aggregation: str
    point_assignment: str
    state_handling: str
    interval_allocation: str
    broadcasting: str
    required_support_outputs: tuple[str, ...]
    conservation_invariant: str | None = None


@dataclass(frozen=True, slots=True)
class PolicyTestContract:
    fixture_ids: tuple[str, ...]
    invariants: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FeaturePolicy:
    identity: IdentityPolicy
    schema: SchemaPolicy
    semantics: SemanticPolicy
    units: UnitPolicy
    provenance: ProvenancePolicy
    reconciliation: ReconciliationPolicy
    curation: CurationPolicy
    cross_feature: CrossFeaturePolicy
    output: OutputPolicy
    resampling: ResamplingPolicy
    evidence_refs: tuple[str, ...]
    tests: PolicyTestContract
    notes: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.identity.canonical_name

    def as_dict(self) -> dict[str, Any]:
        return to_primitive(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RegistryValidationResult:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def to_primitive(value: Any) -> Any:
    """Convert nested policy dataclasses and enums to JSON-safe primitives."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: to_primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [to_primitive(item) for item in value]
    if isinstance(value, list):
        return [to_primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in value.items()}
    return value
