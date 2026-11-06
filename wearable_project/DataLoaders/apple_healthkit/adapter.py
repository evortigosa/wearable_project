"""
Wearable Data Processing and Modeling project
Apple HealthKit adapter for the source-neutral DataLoader seam.
"""


from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
from wearable_project.DataLoaders._contracts import (
    ColumnMetadataContract, CuratedFramePreparation, DateContract, InfoDocument, LoadReportContext,
    LoaderFeatureContract, NonPublicOutputContract, RootIdentity, StateManifestContract,
)
from wearable_project.DataLoaders._sources import (
    APPLE_HEALTHKIT_PUBLIC_FEATURES, DEFAULT_APPLE_HEALTHKIT_CURATED_ROOT, DEFAULT_APPLE_HEALTHKIT_NATIVE_ROOT,
    WearableSource, supported_features_for_source,
)


class AppleHealthKitAdapter:
    """
    Compatibility adapter for the accepted Apple loader implementation.
    Public class paths and the production orchestration class remain in ``DataLoaders._base``. Source-specific
    schemas, units, root identity, curated sparse reconstruction, and policy lookups live behind this adapter so
    Google paths do not import or reinterpret Apple contracts.
    """

    source = WearableSource.APPLE_HEALTHKIT
    implemented_phases = frozenset({"native", "curated"})

    @property
    def native_root(self) -> Path:
        return DEFAULT_APPLE_HEALTHKIT_NATIVE_ROOT

    @property
    def curated_root(self) -> Path:
        return DEFAULT_APPLE_HEALTHKIT_CURATED_ROOT

    def features(self, phase: str) -> tuple[str, ...]:
        return supported_features_for_source(self.source, phase)

    def supports_feature(self, feature: str, phase: str) -> bool:
        # All production loaders can be validated without importing the Apple processing package. Native-only
        # extension classes still consult the complete native registry on demand.
        if feature in APPLE_HEALTHKIT_PUBLIC_FEATURES:
            return True
        return phase == "native" and feature in self.features(phase)

    def date_contract(self, feature: str) -> DateContract:
        from wearable_project.DataLoaders.apple_healthkit.contracts import date_contract

        return date_contract(feature)

    def projection_schema(self, feature: str, phase: str) -> Any:
        del feature, phase
        from wearable_project.DataLoaders.apple_healthkit.schema import projection_schema

        return projection_schema()

    def feature_contract(self, feature: str, phase: str) -> LoaderFeatureContract:
        from wearable_project.DataLoaders.apple_healthkit.contracts import feature_contract

        return feature_contract(feature, phase)

    def validate_root_identity(self, identity: RootIdentity, phase: str) -> None:
        from wearable_project.DataLoaders.apple_healthkit.state import validate_root_identity

        validate_root_identity(identity, phase)

    def validate_root_contract(self, root: Path, phase: str) -> dict[str, Any]:
        del root, phase
        # Apple curated semantics are physically stored and need no root-level virtual-projection contract.
        return {}

    def state_manifest_contract(self, phase: str) -> StateManifestContract:
        from wearable_project.DataLoaders.apple_healthkit.state import manifest_contract

        return manifest_contract(phase)

    def non_public_output_contracts(self, phase: str) -> tuple[NonPublicOutputContract, ...]:
        """Apple native types supported by processing but intentionally absent from the public loader catalog."""

        if phase != "native":
            return ()
        public = set(APPLE_HEALTHKIT_PUBLIC_FEATURES)
        return tuple(
            NonPublicOutputContract(
                feature=feature,
                classification="supported_non_public",
                detail=(
                    "Apple native processing supports this feature, but wearable-project 0.4.0 deliberately "
                    "does not expose it through the established 33-feature public DataLoader catalog"
                ),
            )
            for feature in self.features("native")
            if feature not in public
        )

    def required_root_files(self, phase: str) -> tuple[str, ...]:
        del phase
        # Apple interpretations needed by the loader are physically stored in the CSV and state database.
        return ()

    def prepare_curated_frame(self, frame: Any, path: Path, feature: str,) -> CuratedFramePreparation:
        from wearable_project.DataLoaders.apple_healthkit.curated import prepare_curated_frame

        return prepare_curated_frame(frame, path, feature)

    def reconcile_curated_frame(
        self, prepared: CuratedFramePreparation, path: Path, feature: str, entry: Mapping[str, Any],
    ) -> bool:
        from wearable_project.DataLoaders.apple_healthkit.curated import reconcile_curated_frame

        return reconcile_curated_frame(prepared, path, feature, entry)

    def installed_policy_fingerprint(self, feature: str, phase: str) -> str | None:
        if phase != "curated":
            return None
        from wearable_project.curation.apple_healthkit.registry import get_policy

        try:
            return str(get_policy(feature, allow_fallback=False).fingerprint())
        except KeyError:
            return None

    def requires_installed_policy_without_state(self, phase: str) -> bool:
        del phase
        # Legacy Apple curated products physically store every interpretation used by the loader. Preserve the
        # accepted behavior of consulting the installed policy only when a curation-state row is available.
        return False

    def dense_default(self, feature: str, column: str, phase: str) -> Any:
        from wearable_project.DataLoaders.apple_healthkit.curated import dense_default

        return dense_default(feature, column, phase)

    def curated_logical_counts(self, frame: Any, feature: str) -> Mapping[str, Any]:
        del frame, feature
        # Apple load reports retain their accepted counters computed by the compatibility engine.
        return {}

    def frame_load_metrics(self, frame: Any, feature: str, phase: str) -> Mapping[str, Any]:
        del frame, feature, phase
        return {}

    def load_report_extension(
        self, metrics: Mapping[str, Any], context: LoadReportContext,
    ) -> Mapping[str, Any]:
        del metrics, context
        # The accepted Apple report has no source-specific extension block.
        return {}

    def curated_profile_record(
        self, entry: Mapping[str, Any], installed_fingerprint: str | None,
    ) -> Mapping[str, Any]:
        from wearable_project.DataLoaders.apple_healthkit.curated import curated_profile_record

        return curated_profile_record(entry, installed_fingerprint)

    def curated_profile_summary(
        self, entries: Sequence[Mapping[str, Any]], installed_fingerprint: str | None,
    ) -> Mapping[str, Any]:
        from wearable_project.DataLoaders.apple_healthkit.curated import curated_profile_summary

        return curated_profile_summary(tuple(entries), installed_fingerprint)

    def curated_profile_columns(self) -> Mapping[str, str]:
        from wearable_project.DataLoaders.apple_healthkit.curated import curated_profile_columns

        return curated_profile_columns()

    def validate_frame(self, frame: Any, path: Path, feature: str, phase: str) -> None:
        del frame, path, feature, phase
        # Apple content validation remains exactly the production generic/numeric/timestamp checks.
        return None

    def participant_statuses(self, root: Path) -> dict[str, str]:
        del root
        # Legacy Apple loading has never classified complete-empty source participants in load reports.
        return {}

    def contract_metadata(self, feature: str, phase: str) -> dict[str, Any]:
        del feature, phase
        # Keep the accepted Apple load-report schema unchanged.
        return {}

    def registry_unit(self, feature: str, column: str) -> str | None:
        from wearable_project.DataLoaders.apple_healthkit.units import stored_unit

        return stored_unit(feature, column).unit

    def column_metadata(
        self, feature: str, column: str, phase: str, schema: Any, *, dense: bool,
    ) -> ColumnMetadataContract:
        from wearable_project.DataLoaders.apple_healthkit.schema import column_metadata

        return column_metadata(feature, column, phase, schema, dense=dense)

    def undeclared_columns_warning(self, feature: str, columns: Sequence[str]) -> str:
        return (
            f"{len(columns)} column(s) in {feature} files have no declared role in curation.schema: "
            f"{', '.join(sorted(columns))}. They are returned only by projection='full' or an explicit "
            "columns= request. Add them to COLUMN_ROLES."
        )

    def overview_info(self, *, native_root: Path, curated_root: Path) -> InfoDocument:
        from wearable_project.DataLoaders.apple_healthkit.info import _overview_report

        return _overview_report(native_root=native_root, curated_root=curated_root)

    def feature_info(self, loader: Any, *, include_evidence: bool) -> InfoDocument:
        from wearable_project.DataLoaders.apple_healthkit.info import _feature_report

        return _feature_report(loader, include_evidence=include_evidence)

    def with_harmonized_values(self, data: Any, feature: str, phase: str) -> Any:
        from wearable_project.DataLoaders.apple_healthkit.derived import with_harmonized_values

        return with_harmonized_values(data, feature, phase)
