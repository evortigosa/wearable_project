"""
Wearable Data Processing and Modeling project
Google Fitness adapter for the source-neutral DataLoader seam.
"""


from __future__ import annotations
from contextlib import closing
from pathlib import Path
import sqlite3
from typing import Any
import numpy as np
import pandas as pd
from wearable_project.DataLoaders._contracts import DateContract, LoaderFeatureContract, RootIdentity
from wearable_project.DataLoaders._sources import (
    DEFAULT_GOOGLE_FITNESS_CURATED_ROOT, DEFAULT_GOOGLE_FITNESS_NATIVE_ROOT, WearableSource,
    supported_features_for_source,
)
from wearable_project.DataLoaders.google_fitness.schema import SCHEMA_VERSION, get_schema
from wearable_project.exceptions import DataLoaderConfigurationError, DataLoaderReadError
from wearable_project.processing.google_fitness.registry import (
    ColumnType, NativeFeatureFamily, REGISTRY_VERSION, SLEEP_STATE_MAPPING, get_feature_spec,
)
from wearable_project.processing.google_fitness.writer import STATE_SCHEMA_VERSION


class GoogleFitnessAdapter:
    """
    Google Fitness source contract for the shared DataLoader engine.
    Native loading is implemented for every registered output feature. Curated schema-v2 loading remains disabled
    until the next checkpoint, so an explicit curated Google loader cannot accidentally interpret sparse physical
    decisions or contract-defined virtual columns using the Apple curation machinery.
    """

    source = WearableSource.GOOGLE_FITNESS
    implemented_phases = frozenset({"native"})

    @property
    def native_root(self) -> Path:
        return DEFAULT_GOOGLE_FITNESS_NATIVE_ROOT

    @property
    def curated_root(self) -> Path:
        return DEFAULT_GOOGLE_FITNESS_CURATED_ROOT

    def features(self, phase: str) -> tuple[str, ...]:
        return supported_features_for_source(self.source, phase)

    def supports_feature(self, feature: str, phase: str) -> bool:
        return feature in self.features(phase)

    def date_contract(self, feature: str) -> DateContract:
        if feature == "StepCount":
            return DateContract("outer_datetime", "source_outer_summary_datetime_not_canonical_event_day")
        if feature in {"Height", "Weight"}:
            return DateContract("end_date", "source_reading_interval_end")
        return DateContract("start_date", "event_start")

    def projection_schema(self, feature: str, phase: str) -> Any:
        if phase != "native":
            raise DataLoaderConfigurationError(
                "Google Fitness curated projection schema is not enabled until the schema-v2 contract checkpoint"
            )
        return get_schema(feature)

    def feature_contract(self, feature: str, phase: str) -> LoaderFeatureContract:
        if phase != "native":
            raise DataLoaderConfigurationError(
                "Google Fitness curated feature contracts are not enabled until the schema-v2 contract checkpoint"
            )
        spec = get_feature_spec(feature)
        schema = get_schema(feature)
        return LoaderFeatureContract(
            feature=feature,
            date=self.date_contract(feature),
            timestamp_columns=schema.TIMESTAMP_COLUMNS,
            calendar_date_columns=schema.CALENDAR_DATE_COLUMNS,
            text_columns=schema.TEXT_COLUMNS,
            categorical_columns=schema.CATEGORICAL_COLUMNS,
            string_columns=schema.STRING_COLUMNS,
            numeric_columns=schema.NUMERIC_COLUMNS,
            integer_columns=schema.INTEGER_COLUMNS,
            boolean_columns=schema.NULLABLE_BOOLEAN_COLUMNS,
            measurement_columns=tuple(spec.measurement_columns),
            required_columns=tuple(spec.required_columns),
            physical_columns=tuple(spec.column_names),
        )

    def validate_root_identity(self, identity: RootIdentity, phase: str) -> None:
        """Reject managed Google roots written by an incompatible native state or registry contract."""

        from wearable_project.DataLoaders._base import DataLoaderStateError

        if phase != "native" or not identity.managed:
            return
        if identity.source is not self.source or not identity.verified:
            raise DataLoaderStateError(
                "managed Google Fitness native state could not establish the requested source contract: "
                f"{identity.evidence}"
            )
        state_version = identity.metadata.get("state_schema_version")
        registry_version = identity.metadata.get("registry_version")
        if state_version != STATE_SCHEMA_VERSION:
            raise DataLoaderStateError(
                "incompatible Google Fitness native state schema: "
                f"root records {state_version!r}, installed loader requires {STATE_SCHEMA_VERSION!r}"
            )
        if registry_version != REGISTRY_VERSION:
            raise DataLoaderStateError(
                "incompatible Google Fitness native registry: "
                f"root records {registry_version!r}, installed loader requires {REGISTRY_VERSION!r}"
            )

    def validate_frame(self, frame: pd.DataFrame, path: Path, feature: str, phase: str) -> None:
        """Apply registered Google native schema and version-specific invariants after temporal parsing."""

        if phase != "native":
            return
        spec = get_feature_spec(feature)
        expected = list(spec.column_names)
        expected_set = set(expected)
        missing = [column for column in expected if column not in frame.columns]
        if missing:
            raise DataLoaderReadError(
                f"{path} is missing registered Google Fitness column(s): {', '.join(missing)}"
            )
        registered_order = [column for column in frame.columns if column in expected_set]
        if registered_order != expected:
            raise DataLoaderReadError(
                f"{path} does not preserve the registered Google Fitness column order for {feature}: "
                f"expected {expected!r}, found {registered_order!r}"
            )

        for column in spec.required_columns:
            absent = frame[column].isna()
            if bool(absent.any()):
                row = int(absent[absent].index[0])
                raise DataLoaderReadError(
                    f"{path}: required column {column!r} is blank at data row {row + 1}"
                )

        versions = frame["collecting_method_version"].astype("string")
        invalid_version = versions.notna() & ~versions.isin(spec.supported_versions)
        if bool(invalid_version.any()):
            row = int(invalid_version[invalid_version].index[0])
            raise DataLoaderReadError(
                f"{path}: unsupported collecting_method_version {versions.loc[row]!r} at data row {row + 1}; "
                f"registered versions are {', '.join(spec.supported_versions)}"
            )

        for column_spec in spec.columns:
            column = column_spec.name
            values = frame[column]
            if column_spec.data_type in {ColumnType.NUMBER, ColumnType.INTEGER}:
                present = values.notna()
                if bool(present.any()):
                    numeric = pd.to_numeric(values.loc[present], errors="raise")
                    finite = np.isfinite(numeric.to_numpy(dtype=float))
                    if not bool(finite.all()):
                        bad_position = int(np.flatnonzero(~finite)[0])
                        row = int(numeric.index[bad_position])
                        raise DataLoaderReadError(
                            f"{path}: column {column!r} must contain finite values; data row {row + 1} is non-finite"
                        )
                    if column_spec.data_type is ColumnType.INTEGER:
                        integral = np.equal(numeric.to_numpy(dtype=float), np.floor(numeric.to_numpy(dtype=float)))
                        if not bool(integral.all()):
                            bad_position = int(np.flatnonzero(~integral)[0])
                            row = int(numeric.index[bad_position])
                            raise DataLoaderReadError(
                                f"{path}: column {column!r} must contain whole numbers; "
                                f"data row {row + 1} holds {values.loc[row]!r}"
                            )

        if "payload_index" in frame.columns:
            bad = frame["payload_index"].notna() & (pd.to_numeric(frame["payload_index"]) < 0)
            if bool(bad.any()):
                row = int(bad[bad].index[0])
                raise DataLoaderReadError(f"{path}: payload_index is negative at data row {row + 1}")

        if {"start_date", "end_date"} <= set(frame.columns):
            present = frame["start_date"].notna() & frame["end_date"].notna()
            if spec.family in {NativeFeatureFamily.ACTIVITY_INTERVAL, NativeFeatureFamily.SLEEP_STATE_INTERVAL}:
                bad = present & ~(frame["end_date"] > frame["start_date"])
                relation = "strictly after"
            else:
                bad = present & (frame["end_date"] < frame["start_date"])
                relation = "at or after"
            if bool(bad.any()):
                row = int(bad[bad].index[0])
                raise DataLoaderReadError(
                    f"{path}: end_date must be {relation} start_date; data row {row + 1} violates the contract"
                )

        if feature == "Sleep":
            allowed = frozenset(SLEEP_STATE_MAPPING.values())
            states = frame["value"].astype("string")
            bad = states.notna() & ~states.isin(allowed)
            if bool(bad.any()):
                row = int(bad[bad].index[0])
                raise DataLoaderReadError(
                    f"{path}: unregistered Google Fitness sleep state {states.loc[row]!r} at data row {row + 1}"
                )

        if feature == "StepCount":
            self._validate_step_count_versions(frame, path)

    @staticmethod
    def _validate_step_count_versions(frame: pd.DataFrame, path: Path) -> None:
        versions = frame["collecting_method_version"].astype("string")
        legacy = versions.eq("1.0").fillna(False)
        modern = versions.eq("2.0").fillna(False)
        modern_required = (
            "summary_date",
            "start_date",
            "end_date",
            "raw_value",
            "aggregation_source",
            "origin_data_source_id",
        )
        if bool(modern.any()):
            for column in modern_required:
                bad = modern & frame[column].isna()
                if bool(bad.any()):
                    row = int(bad[bad].index[0])
                    raise DataLoaderReadError(
                        f"{path}: StepCount 2.0 requires {column!r}; data row {row + 1} is blank"
                    )
        if bool(legacy.any()):
            modern_only = (
                "summary_date",
                "start_date",
                "end_date",
                "raw_value",
                "aggregation_source",
                "origin_data_source_id",
                "app_name",
                "app_package_name",
                "device_manufacturer",
                "device_model",
                "device_type",
                "device_uid",
            )
            for column in modern_only:
                bad = legacy & frame[column].notna()
                if bool(bad.any()):
                    row = int(bad[bad].index[0])
                    raise DataLoaderReadError(
                        f"{path}: StepCount 1.0 must not invent {column!r}; data row {row + 1} contains a value"
                    )
            bad_index = legacy & (frame["payload_index"] != 0)
            if bool(bad_index.any()):
                row = int(bad_index[bad_index].index[0])
                raise DataLoaderReadError(
                    f"{path}: StepCount 1.0 payload_index must be zero; data row {row + 1} is not"
                )

    def participant_statuses(self, root: Path) -> dict[str, str]:
        """Participant completion status from the managed native state, without touching feature CSVs."""

        database = root / ".wearable_state.sqlite"
        if not database.is_file():
            return {}
        uri = f"{database.resolve().as_uri()}?immutable=1"
        try:
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                return {
                    str(participant_id): str(status)
                    for participant_id, status in connection.execute(
                        "SELECT participant_id, status FROM participants"
                    ).fetchall()
                }
        except sqlite3.Error as exc:
            from wearable_project.DataLoaders._base import DataLoaderStateError

            raise DataLoaderStateError(f"could not read participant statuses from {database}: {exc}") from exc

    def contract_metadata(self, feature: str, phase: str) -> dict[str, Any]:
        spec = get_feature_spec(feature)
        return {
            "dataloader_schema_version": SCHEMA_VERSION,
            "native_registry_version": REGISTRY_VERSION,
            "native_state_schema_version": STATE_SCHEMA_VERSION,
            "feature_family": spec.family.value,
            "supported_collecting_method_versions": list(spec.supported_versions),
        }

    def registry_unit(self, feature: str, column: str) -> str | None:
        # Count is established by the StepCount source contract. Exporter-unit mappings for every other Google
        # measurement are deliberately unresolved and remain None until curation establishes them.
        if feature == "StepCount" and column in {"value", "raw_value"}:
            return "count"
        return None
