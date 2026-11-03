"""
Wearable Data Processing and Modeling project
Apple HealthKit projection and returned-column metadata contracts.
"""


from __future__ import annotations
from typing import Any
from wearable_project.DataLoaders._contracts import ColumnMetadataContract
from wearable_project.DataLoaders.apple_healthkit.units import stored_unit


DENSE_CURATED_DEFAULTS: dict[str, Any] = {
    "acquisition_method": "unclassified",
    "curation_status": "pass",
    "curation_flags": "",
    "include_by_default": True,
}
DENSE_FILL_BLANKS: frozenset[str] = frozenset({"acquisition_method", "curation_flags"})
CURATION_FLAG_SEPARATOR = ";"


def projection_schema() -> Any:
    from wearable_project.curation.apple_healthkit import schema

    return schema


def column_metadata(
    feature: str, column: str, phase: str, schema: Any, *, dense: bool,
) -> ColumnMetadataContract:
    """Return the exact metadata contract historically emitted by the production Apple loader."""

    from wearable_project.processing.apple_healthkit.registry import get_feature_spec

    spec = get_feature_spec(feature)
    unit_policy = spec.unit_policy
    role = schema.role_of(column)
    registry_unit = None
    if unit_policy is not None:
        if role is schema.ColumnRole.MEASUREMENT and column in spec.measurement_columns:
            registry_unit = stored_unit(feature, column).unit
        elif column == "canonical_value":
            registry_unit = unit_policy.canonical_unit
    return ColumnMetadataContract(
        role=role.value if role is not None else "undeclared",
        description=(
            schema.ROLE_DESCRIPTIONS[role]
            if role is not None
            else "No declared role in curation.schema."
        ),
        registry_unit=registry_unit,
        dense_default=DENSE_CURATED_DEFAULTS.get(column) if dense else None,
    )
