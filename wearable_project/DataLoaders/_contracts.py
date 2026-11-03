"""
Wearable Data Processing and Modeling project
Source-neutral contracts used by the DataLoader orchestration layer.
These objects describe source identity, source-qualified feature timing, and the logical schema information the
shared loader engine needs.  Scientific details remain in the source adapters: the engine never imports one
source's processing or curation registry to interpret another source's files.
"""


from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Protocol, Sequence, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from wearable_project.DataLoaders._sources import WearableSource


def participant_component_issue(value: str) -> str | None:
    """Why ``value`` is unsafe as one participant-directory component, or ``None`` when it is safe."""

    if not value:
        return "is empty"
    if "\x00" in value:
        return "contains a NUL character"
    if value in {".", ".."}:
        return "is a relative path marker"
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        return "is an absolute or drive-qualified path"
    if len(posix.parts) != 1 or len(windows.parts) != 1:
        return "contains a path separator"
    return None


@dataclass(frozen=True, slots=True)
class DateContract:
    """The timestamp column used mechanically for the HPP ``Date`` index and its scientific meaning."""

    column: str
    semantics: str


@dataclass(frozen=True, slots=True)
class LoaderFeatureContract:
    """
    Source-qualified physical/logical schema facts consumed by the shared loader engine.
    ``timestamp_columns`` are absolute instants parsed as timezone-aware UTC. ``calendar_date_columns`` are
    source calendar labels parsed as timezone-naive dates; they must never be converted into synthetic UTC
    midnights. ``text_columns`` are forced to text during CSV parsing so identifiers and source versions retain
    their lexical representation. The remaining dtype groups are applied after participant concatenation so a
    column's dtype does not depend on which participants were loaded.
    """

    feature: str
    date: DateContract
    timestamp_columns: tuple[str, ...]
    calendar_date_columns: tuple[str, ...]
    text_columns: frozenset[str]
    categorical_columns: frozenset[str]
    string_columns: frozenset[str]
    numeric_columns: frozenset[str]
    integer_columns: frozenset[str]
    boolean_columns: frozenset[str]
    measurement_columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    physical_columns: tuple[str, ...]
    unknown_columns_as_text: bool = False
    index_sort_kind: str | None = None


@dataclass(frozen=True, slots=True)
class StateManifestContract:
    """Source/phase-specific state query and validation fields consumed by the shared engine."""

    table: str
    query: str
    row_count_column: str
    required_columns: tuple[str, ...]
    nonnegative_integer_columns: tuple[str, ...] = ()
    sha256_columns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ColumnMetadataContract:
    """Source-qualified metadata for one returned analytical column."""

    role: str
    description: str
    registry_unit: str | None = None
    dense_default: Any = None


@dataclass(slots=True)
class CuratedFramePreparation:
    """One physical curated CSV expanded to the source's complete logical row schema."""

    frame: Any
    physical_columns: tuple[str, ...]
    active_materialized_columns: tuple[str, ...]
    reconstructed_rows: dict[str, int]
    reconstructed_columns: tuple[str, ...]
    virtual_columns: tuple[str, ...]
    inclusion_reconstructed_mask: Any


@dataclass(frozen=True, slots=True)
class InfoDocument:
    """Source-specific static documentation before the public ``InfoReport`` compatibility wrapper."""

    kind: str
    title: str
    payload: dict[str, Any]
    text: str


@dataclass(frozen=True, slots=True)
class LoadReportContext:
    """Source-neutral facts available when an adapter finalizes load-report extensions."""

    requested_empty_for_source: tuple[str, ...]
    physical_columns_by_participant: Mapping[str, Sequence[str]]
    reconstructed_columns: tuple[str, ...]
    virtual_columns: tuple[str, ...]
    root_contract_metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RootIdentity:
    """
    Source and phase evidence obtained from a managed root without reading any feature CSV.
    ``source`` is ``None`` for an unmanaged/trimmed tree or for a present state database whose source cannot be
    established. Only the former is loadable as source-unverified: a present but unreadable, active, corrupt, or
    unrecognized managed state is refused independently of optional file-hash verification. ``verified`` means the
    state metadata or a source-specific state-schema signature establishes the source unambiguously.
    """

    source: "WearableSource | None"
    phase: str | None
    managed: bool
    verified: bool
    evidence: str
    state_database: Path | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    tables: tuple[str, ...] = ()


@runtime_checkable
class SourceAdapter(Protocol):
    """Source contract consumed by the compatibility-preserving shared DataLoader engine."""

    source: "WearableSource"
    implemented_phases: frozenset[str]

    @property
    def native_root(self) -> Path: ...

    @property
    def curated_root(self) -> Path: ...

    def supports_feature(self, feature: str, phase: str) -> bool: ...

    def features(self, phase: str) -> tuple[str, ...]: ...

    def date_contract(self, feature: str) -> DateContract: ...

    def feature_contract(self, feature: str, phase: str) -> LoaderFeatureContract: ...

    def projection_schema(self, feature: str, phase: str) -> Any: ...

    def validate_root_identity(self, identity: RootIdentity, phase: str) -> None: ...

    def validate_root_contract(self, root: Path, phase: str) -> Mapping[str, Any]: ...

    def state_manifest_contract(self, phase: str) -> StateManifestContract: ...

    def prepare_curated_frame(self, frame: Any, path: Path, feature: str,) -> CuratedFramePreparation: ...

    def reconcile_curated_frame(
        self, prepared: CuratedFramePreparation, path: Path, feature: str, entry: Mapping[str, Any],
    ) -> bool: ...

    def installed_policy_fingerprint(self, feature: str, phase: str) -> str | None: ...

    def requires_installed_policy_without_state(self, phase: str) -> bool: ...

    def dense_default(self, feature: str, column: str, phase: str) -> Any: ...

    def curated_logical_counts(self, frame: Any, feature: str) -> Mapping[str, Any]: ...

    def frame_load_metrics(self, frame: Any, feature: str, phase: str) -> Mapping[str, Any]: ...

    def load_report_extension(self, metrics: Mapping[str, Any], context: LoadReportContext,) -> Mapping[str, Any]: ...

    def curated_profile_record(
        self, entry: Mapping[str, Any], installed_fingerprint: str | None,
    ) -> Mapping[str, Any]: ...

    def curated_profile_summary(
        self, entries: Sequence[Mapping[str, Any]], installed_fingerprint: str | None,
    ) -> Mapping[str, Any]: ...

    def curated_profile_columns(self) -> Mapping[str, str]: ...

    def validate_frame(self, frame: Any, path: Path, feature: str, phase: str) -> None: ...

    def participant_statuses(self, root: Path) -> dict[str, str]: ...

    def contract_metadata(self, feature: str, phase: str) -> Mapping[str, Any]: ...

    def registry_unit(self, feature: str, column: str) -> str | None: ...

    def column_metadata(
        self, feature: str, column: str, phase: str, schema: Any, *, dense: bool,
    ) -> ColumnMetadataContract: ...

    def undeclared_columns_warning(self, feature: str, columns: Sequence[str]) -> str: ...

    def overview_info(self, *, native_root: Path, curated_root: Path) -> InfoDocument: ...

    def feature_info(self, loader: Any, *, include_evidence: bool) -> InfoDocument: ...

    def with_harmonized_values(self, data: Any, feature: str, phase: str) -> Any: ...
