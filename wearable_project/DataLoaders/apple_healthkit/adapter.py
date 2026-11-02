"""
Wearable Data Processing and Modeling project
Apple HealthKit adapter for the source-neutral DataLoader seam.
"""


from __future__ import annotations
from pathlib import Path
from typing import Any
from wearable_project.DataLoaders._contracts import DateContract, LoaderFeatureContract, RootIdentity
from wearable_project.DataLoaders._sources import (
    DEFAULT_APPLE_HEALTHKIT_CURATED_ROOT, DEFAULT_APPLE_HEALTHKIT_NATIVE_ROOT, WearableSource,
    supported_features_for_source,
)


class AppleHealthKitAdapter:
    """
    Compatibility adapter for the accepted Apple loader implementation.
    The mature Apple algorithms still execute in ``DataLoaders._base``. This adapter owns source-qualified
    catalog, roots, and temporal contracts while exposing the normalized contract required by the shared engine.
    Apple parsing, projections, dtypes, state reconciliation, units, and metadata remain byte-for-byte compatible
    with the production implementation.
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
        return feature in self.features(phase)

    def date_contract(self, feature: str) -> DateContract:
        if feature == "ActivitySummary":
            return DateContract("datetime", "outer_summary_datetime_not_canonical_summary_day")
        return DateContract("start_date", "event_start")

    def projection_schema(self, feature: str, phase: str) -> Any:
        from wearable_project.curation.apple_healthkit import schema

        return schema

    def feature_contract(self, feature: str, phase: str) -> LoaderFeatureContract:
        from wearable_project.curation.apple_healthkit import schema
        from wearable_project.processing.apple_healthkit.registry import get_feature_spec

        spec = get_feature_spec(feature)
        categorical = schema.categorical_columns(feature)
        measurements = {
            column
            for column in spec.measurement_columns
            if column not in categorical and schema.role_of(column) is not schema.ColumnRole.PAYLOAD
        }
        numeric = measurements | {"canonical_value"} | set(schema.INTEGER_COLUMNS)
        # Keep the accepted Apple parser's temporal universe. Some columns are absent from most feature files;
        # listing them here changes no stored value and preserves the exact historical parsing behavior.
        temporal = (
            "start_date",
            "end_date",
            "datetime",
            "created_at",
            "updated_at",
            "modified_date",
        )
        return LoaderFeatureContract(
            feature=feature,
            date=self.date_contract(feature),
            timestamp_columns=temporal,
            calendar_date_columns=(),
            text_columns=frozenset(categorical),
            categorical_columns=frozenset(categorical),
            string_columns=frozenset(),
            numeric_columns=frozenset(numeric),
            integer_columns=frozenset(schema.INTEGER_COLUMNS),
            boolean_columns=frozenset(schema.NULLABLE_BOOLEAN_COLUMNS),
            measurement_columns=tuple(spec.measurement_columns),
            required_columns=(),
            physical_columns=(),
        )

    def validate_root_identity(self, identity: RootIdentity, phase: str) -> None:
        # Backward-compatible Apple roots predate source metadata and are already recognized by their state schema.
        return None

    def validate_frame(self, frame: Any, path: Path, feature: str, phase: str) -> None:
        # Apple validation remains exactly where it has always lived in DataLoaders._base.
        return None

    def participant_statuses(self, root: Path) -> dict[str, str]:
        # Legacy Apple loading has never classified complete-empty source participants in load reports.
        return {}

    def contract_metadata(self, feature: str, phase: str) -> dict[str, Any]:
        # Keep the accepted Apple load-report schema unchanged.
        return {}

    def registry_unit(self, feature: str, column: str) -> str | None:
        from wearable_project.DataLoaders._units import stored_unit

        established = stored_unit(feature, column)
        return established.unit
