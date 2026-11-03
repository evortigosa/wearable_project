"""
Wearable Data Processing and Modeling project
Source-neutral contracts used by the DataLoader orchestration layer.
These objects describe source identity and source-qualified feature timing without importing either source's
processing or curation implementation. They are deliberately small: the mature Apple loader remains the
behavioral reference while source-specific machinery is extracted behind this boundary.
"""


from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from wearable_project.DataLoaders._sources import WearableSource


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


@dataclass(frozen=True, slots=True)
class RootIdentity:
    """
    Source and phase evidence obtained from a managed root without reading any feature CSV.
    ``source`` is ``None`` for an unmanaged/trimmed tree or an unrecognized state schema. Such a tree may still
    be loaded under ``state_validation='auto'`` or ``'off'``, but its source cannot be proven. ``verified`` means
    the state metadata or a source-specific state-schema signature establishes the source unambiguously.
    """

    source: "WearableSource | None"
    phase: str | None
    managed: bool
    verified: bool
    evidence: str
    state_database: Path | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    tables: tuple[str, ...] = ()


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

    def prepare_curated_frame(self, frame: Any, path: Path, feature: str) -> Any: ...

    def reconcile_curated_frame(
        self, prepared: Any, path: Path, feature: str, entry: Mapping[str, Any],
    ) -> bool: ...

    def installed_policy_fingerprint(self, feature: str, phase: str) -> str | None: ...

    def dense_default(self, feature: str, column: str, phase: str) -> Any: ...

    def curated_logical_counts(self, frame: Any, feature: str) -> Mapping[str, Any]: ...

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
