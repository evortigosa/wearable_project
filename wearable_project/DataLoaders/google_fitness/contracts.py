"""
Wearable Data Processing and Modeling project
Google Fitness feature, temporal, and versioned DataLoader contracts.
"""


from __future__ import annotations
from typing import Any
from wearable_project.DataLoaders._contracts import DateContract, LoaderFeatureContract


def date_contract(feature: str) -> DateContract:
    if feature == "StepCount":
        return DateContract("outer_datetime", "source_outer_summary_datetime_not_canonical_event_day")
    if feature in {"Height", "Weight"}:
        return DateContract("end_date", "source_reading_interval_end")
    return DateContract("start_date", "event_start")


def feature_contract(feature: str, phase: str) -> LoaderFeatureContract:
    from wearable_project.DataLoaders.google_fitness.schema import get_schema
    from wearable_project.processing.google_fitness.registry import get_feature_spec

    spec = get_feature_spec(feature)
    schema = get_schema(feature, phase)
    physical_columns = tuple(spec.column_names)
    required_columns = tuple(spec.required_columns)
    measurement_columns = tuple(spec.measurement_columns)
    if phase == "curated":
        from wearable_project.curation.google_fitness.registry import get_policy

        policy = get_policy(feature, allow_fallback=False)
        physical_columns = (*physical_columns, *policy.output.materialized_columns)
        required_columns = (*required_columns, *policy.output.fixed_columns)
        if "canonical_value" in policy.output.virtual_columns:
            measurement_columns = (*measurement_columns, "canonical_value")
    return LoaderFeatureContract(
        feature=feature,
        date=date_contract(feature),
        timestamp_columns=schema.TIMESTAMP_COLUMNS,
        calendar_date_columns=schema.CALENDAR_DATE_COLUMNS,
        text_columns=schema.TEXT_COLUMNS,
        categorical_columns=schema.CATEGORICAL_COLUMNS,
        string_columns=schema.STRING_COLUMNS,
        numeric_columns=schema.NUMERIC_COLUMNS,
        integer_columns=schema.INTEGER_COLUMNS,
        boolean_columns=schema.NULLABLE_BOOLEAN_COLUMNS,
        measurement_columns=measurement_columns,
        required_columns=required_columns,
        physical_columns=physical_columns,
        unknown_columns_as_text=True,
        index_sort_kind="mergesort",
    )


def contract_metadata(feature: str, phase: str, root_contract: Any | None = None) -> dict[str, Any]:
    from wearable_project.DataLoaders.google_fitness.schema import (
        CURATED_DATALOADER_SCHEMA_VERSION,
        SCHEMA_VERSION,
    )
    from wearable_project.processing.google_fitness.registry import REGISTRY_VERSION, get_feature_spec
    from wearable_project.processing.google_fitness.writer import STATE_SCHEMA_VERSION

    spec = get_feature_spec(feature)
    result: dict[str, Any] = {
        "dataloader_schema_version": SCHEMA_VERSION if phase == "native" else CURATED_DATALOADER_SCHEMA_VERSION,
        "native_registry_version": REGISTRY_VERSION,
        "feature_family": spec.family.value,
        "supported_collecting_method_versions": list(spec.supported_versions),
    }
    if phase == "native":
        result["native_state_schema_version"] = STATE_SCHEMA_VERSION
        return result

    from wearable_project.curation.google_fitness.contract import (
        CONTRACT_SCHEMA_VERSION,
        CURATED_SCHEMA_VERSION,
        contract_fingerprint,
    )
    from wearable_project.curation.google_fitness.registry import (
        CURATION_REGISTRY_VERSION,
        POLICY_CONTRACT_VERSION,
    )
    from wearable_project.curation.google_fitness.state import CURATION_STATE_SCHEMA_VERSION

    result.update({
        "curated_schema_version": CURATED_SCHEMA_VERSION,
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "curation_registry_version": CURATION_REGISTRY_VERSION,
        "policy_contract_version": POLICY_CONTRACT_VERSION,
        "curation_state_schema_version": CURATION_STATE_SCHEMA_VERSION,
        "contract_fingerprint": contract_fingerprint(),
    })
    if root_contract is not None:
        result["root_contract"] = dict(root_contract.metadata)
    return result
