"""
Wearable Data Processing and Modeling project
Apple HealthKit source-qualified feature and temporal contracts for the shared DataLoader engine.
"""


from __future__ import annotations
from wearable_project.DataLoaders._contracts import DateContract, LoaderFeatureContract


def native_features() -> tuple[str, ...]:
    """Complete Apple processing registry, loaded only for native-only extension classes."""

    from wearable_project.processing.apple_healthkit.registry import known_features

    return tuple(known_features())


def date_contract(feature: str) -> DateContract:
    if feature == "ActivitySummary":
        return DateContract("datetime", "outer_summary_datetime_not_canonical_summary_day")
    return DateContract("start_date", "event_start")


def feature_contract(feature: str, phase: str) -> LoaderFeatureContract:
    # Deferred imports keep construction and catalog discovery independent of Apple processing/curation modules.
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
    # Preserve the accepted Apple parser's temporal universe. Some columns are absent from most files; listing
    # them changes no stored value and preserves the production parsing behavior exactly.
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
        date=date_contract(feature),
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
