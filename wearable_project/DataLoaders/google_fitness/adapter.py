"""
Wearable Data Processing and Modeling project
Google Fitness adapter for source-qualified native and curated schema-v2 DataLoaders.
"""


from __future__ import annotations
from pathlib import Path
from collections import Counter
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from wearable_project.DataLoaders._contracts import (
    ColumnMetadataContract, CuratedFramePreparation, DateContract, InfoDocument, LoadReportContext,
    LoaderFeatureContract, RootIdentity, StateManifestContract,
)
from wearable_project.DataLoaders._sources import (
    DEFAULT_GOOGLE_FITNESS_CURATED_ROOT, DEFAULT_GOOGLE_FITNESS_NATIVE_ROOT, GOOGLE_FITNESS_PUBLIC_FEATURES,
    WearableSource,
)
from wearable_project.exceptions import DataLoaderConfigurationError, DataLoaderReadError


_BOOLEAN_TEXT: dict[str, bool] = {
    "true": True, "t": True, "yes": True, "y": True, "1": True, "1.0": True,
    "false": False, "f": False, "no": False, "n": False, "0": False, "0.0": False,
}


def _as_nullable_boolean(values: pd.Series) -> pd.Series | None:
    """Interpret the registered boolean vocabulary without depending on the concrete loader class."""

    if pd.api.types.is_bool_dtype(values):
        return values.astype("boolean")
    missing = values.isna().to_numpy(dtype=bool)
    result = np.zeros(len(values), dtype=bool)
    if (~missing).any():
        present = values.to_numpy(dtype=object)[~missing]
        mapping: dict[Any, bool] = {}
        for value in pd.unique(present):
            if isinstance(value, (bool, np.bool_)):
                token: bool | None = bool(value)
            elif isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
                token = bool(value)
            else:
                token = _BOOLEAN_TEXT.get(str(value).strip().lower())
            if token is None:
                return None
            mapping[value] = token
        result[~missing] = [mapping[value] for value in present]
    return pd.Series(pd.arrays.BooleanArray(result, missing), index=values.index, name=values.name)


class GoogleFitnessAdapter:
    """Google Fitness scientific/source contract consumed by the shared DataLoader engine."""

    source = WearableSource.GOOGLE_FITNESS
    implemented_phases = frozenset({"native", "curated"})

    def __init__(self) -> None:
        self._root_contract: Any | None = None

    @property
    def native_root(self) -> Path:
        return DEFAULT_GOOGLE_FITNESS_NATIVE_ROOT

    @property
    def curated_root(self) -> Path:
        return DEFAULT_GOOGLE_FITNESS_CURATED_ROOT

    def features(self, phase: str) -> tuple[str, ...]:
        del phase
        return GOOGLE_FITNESS_PUBLIC_FEATURES

    def supports_feature(self, feature: str, phase: str) -> bool:
        del phase
        return feature in GOOGLE_FITNESS_PUBLIC_FEATURES

    def date_contract(self, feature: str) -> DateContract:
        from wearable_project.DataLoaders.google_fitness.contracts import date_contract

        return date_contract(feature)

    def projection_schema(self, feature: str, phase: str) -> Any:
        from wearable_project.DataLoaders.google_fitness.schema import get_schema

        return get_schema(feature, phase)

    def feature_contract(self, feature: str, phase: str) -> LoaderFeatureContract:
        from wearable_project.DataLoaders.google_fitness.contracts import feature_contract

        return feature_contract(feature, phase)

    def validate_root_identity(self, identity: RootIdentity, phase: str) -> None:
        from wearable_project.DataLoaders.google_fitness.state import validate_root_identity

        validate_root_identity(identity, phase)

    def validate_root_contract(self, root: Path, phase: str) -> Mapping[str, Any]:
        """Validate and cache schema-v2 root contract before curated files can be interpreted."""

        if phase != "curated":
            self._root_contract = None
            return {}
        from wearable_project.DataLoaders.google_fitness.curated import validate_root_contract

        self._root_contract = validate_root_contract(root)
        return self._root_contract.metadata

    def state_manifest_contract(self, phase: str) -> StateManifestContract:
        from wearable_project.DataLoaders.google_fitness.state import manifest_contract

        return manifest_contract(phase)

    def required_root_files(self, phase: str) -> tuple[str, ...]:
        # Curated schema-v2 sparse defaults and virtual projections are defined by this immutable root contract.
        return (".curation_contract.json",) if phase == "curated" else ()

    def prepare_curated_frame(self, frame: pd.DataFrame, path: Path, feature: str,) -> CuratedFramePreparation:
        if self._root_contract is None:
            raise DataLoaderConfigurationError(
                "Google Fitness curated frame preparation requires a validated root curation contract"
            )
        from wearable_project.DataLoaders.google_fitness.curated import prepare_curated_frame

        return prepare_curated_frame(frame, path, feature)

    def reconcile_curated_frame(
        self, prepared: CuratedFramePreparation, path: Path, feature: str, entry: Mapping[str, Any],
    ) -> bool:
        from wearable_project.DataLoaders.google_fitness.curated import reconcile_curated_frame

        return reconcile_curated_frame(prepared, path, feature, entry)

    def installed_policy_fingerprint(self, feature: str, phase: str) -> str | None:
        if phase != "curated":
            return None
        from wearable_project.DataLoaders.google_fitness.curated import installed_policy_fingerprint

        return installed_policy_fingerprint(feature)

    def requires_installed_policy_without_state(self, phase: str) -> bool:
        # Curated schema-v2 virtual projections are executable installed policy. Their fingerprint remains
        # relevant even when callers disable file-state reconciliation.
        return phase == "curated"

    def dense_default(self, feature: str, column: str, phase: str) -> Any:
        if phase != "curated":
            return None
        from wearable_project.DataLoaders.google_fitness.curated import dense_default

        return dense_default(feature, column)

    def curated_logical_counts(self, frame: pd.DataFrame, feature: str) -> Mapping[str, Any]:
        del feature
        from wearable_project.DataLoaders.google_fitness.curated import logical_counts

        return logical_counts(frame)

    def frame_load_metrics(self, frame: pd.DataFrame, feature: str, phase: str) -> Mapping[str, Any]:
        """Source-qualified returned-row counters used only in Google load reports."""

        metrics: dict[str, Any] = {}
        if "collecting_method_version" in frame.columns:
            metrics["collecting_method_version_counts"] = Counter(
                str(value) for value in frame["collecting_method_version"].dropna()
            )
        if "tracked" in frame.columns:
            tracked = _as_nullable_boolean(frame["tracked"])
            if tracked is not None:
                metrics["tracking_counts"] = Counter({
                    "true": int(tracked.eq(True).fillna(False).sum()),  # noqa: E712 - pandas scalar comparison
                    "false": int(tracked.eq(False).fillna(False).sum()),  # noqa: E712
                    "missing": int(tracked.isna().sum()),
                })
        if phase == "curated":
            metrics.update(self.curated_logical_counts(frame, feature))
        return metrics

    def load_report_extension(
        self, metrics: Mapping[str, Any], context: LoadReportContext,
    ) -> Mapping[str, Any]:
        """Finalize Google-specific report fields without exposing their policy to the shared engine."""

        report: dict[str, Any] = {
            "requested_participants_empty_for_source": list(context.requested_empty_for_source),
            "collecting_method_version_counts": dict(sorted(
                metrics.get("collecting_method_version_counts", {}).items()
            )),
            "tracking_counts": dict(sorted(metrics.get("tracking_counts", {}).items())),
        }
        if not context.root_contract_metadata:
            return report

        physical_observed: dict[str, None] = {}
        for participant_id in sorted(context.physical_columns_by_participant):
            physical_observed.update(dict.fromkeys(context.physical_columns_by_participant[participant_id]))
        report.update({
            "curation_contract_validated": True,
            "curation_contract": dict(context.root_contract_metadata),
            "columns_stored": list(physical_observed),
            "columns_reconstructed": list(context.reconstructed_columns),
            "columns_virtual": list(context.virtual_columns),
            "curation_status_counts": dict(sorted(metrics.get("curation_status_counts", {}).items())),
            "step_count_role_counts": dict(sorted(metrics.get("step_count_role_counts", {}).items())),
            "derivation_method_counts": dict(sorted(metrics.get("derivation_method_counts", {}).items())),
            "tracking_status_counts": dict(sorted(metrics.get("tracking_status_counts", {}).items())),
            "curation_unit_status_counts": dict(sorted(
                metrics.get("curation_unit_status_counts", {}).items()
            )),
            "unit_evidence_counts": dict(sorted(metrics.get("unit_evidence_counts", {}).items())),
            "canonical_value_rows": int(metrics.get("canonical_value_rows", 0)),
        })
        return report

    def curated_profile_record(
        self, entry: Mapping[str, Any], installed_fingerprint: str | None,
    ) -> Mapping[str, Any]:
        from wearable_project.DataLoaders.google_fitness.curated import curated_profile_record

        return curated_profile_record(entry, installed_fingerprint)

    def curated_profile_summary(
        self, entries: Sequence[Mapping[str, Any]], installed_fingerprint: str | None,
    ) -> Mapping[str, Any]:
        from wearable_project.DataLoaders.google_fitness.curated import curated_profile_summary

        return curated_profile_summary(entries, installed_fingerprint)

    def curated_profile_columns(self) -> Mapping[str, str]:
        from wearable_project.DataLoaders.google_fitness.curated import curated_profile_columns

        return curated_profile_columns()

    def validate_frame(self, frame: pd.DataFrame, path: Path, feature: str, phase: str) -> None:
        """Apply registered native-prefix and version-specific invariants after temporal parsing."""

        del phase
        from wearable_project.processing.google_fitness.registry import (
            ColumnType, NativeFeatureFamily, SLEEP_STATE_MAPPING, get_feature_spec,
        )

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
                            f"{path}: column {column!r} must contain finite values; "
                            f"data row {row + 1} is non-finite"
                        )
                    if column_spec.data_type is ColumnType.INTEGER:
                        numeric_array = numeric.to_numpy(dtype=float)
                        integral = np.equal(numeric_array, np.floor(numeric_array))
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
                    f"{path}: end_date must be {relation} start_date; "
                    f"data row {row + 1} violates the contract"
                )

        if feature == "Sleep":
            allowed = frozenset(SLEEP_STATE_MAPPING.values())
            states = frame["value"].astype("string")
            bad = states.notna() & ~states.isin(allowed)
            if bool(bad.any()):
                row = int(bad[bad].index[0])
                raise DataLoaderReadError(
                    f"{path}: unregistered Google Fitness sleep state {states.loc[row]!r} "
                    f"at data row {row + 1}"
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
                        f"{path}: StepCount 1.0 must not invent {column!r}; "
                        f"data row {row + 1} contains a value"
                    )
            bad_index = legacy & (frame["payload_index"] != 0)
            if bool(bad_index.any()):
                row = int(bad_index[bad_index].index[0])
                raise DataLoaderReadError(
                    f"{path}: StepCount 1.0 payload_index must be zero; data row {row + 1} is not"
                )

    def participant_statuses(self, root: Path) -> dict[str, str]:
        from wearable_project.DataLoaders.google_fitness.state import participant_statuses

        return participant_statuses(root)

    def contract_metadata(self, feature: str, phase: str) -> Mapping[str, Any]:
        from wearable_project.DataLoaders.google_fitness.contracts import contract_metadata

        return contract_metadata(feature, phase, self._root_contract)

    def registry_unit(self, feature: str, column: str) -> str | None:
        from wearable_project.DataLoaders.google_fitness.units import registry_unit

        return registry_unit(feature, column)

    def column_metadata(
        self, feature: str, column: str, phase: str, schema: Any, *, dense: bool,
    ) -> ColumnMetadataContract:
        role = schema.role_of(column)
        contract = self.feature_contract(feature, phase)
        return ColumnMetadataContract(
            role=role.value if role is not None else "undeclared",
            description=(
                schema.ROLE_DESCRIPTIONS[role]
                if role is not None
                else "No declared role in the google_fitness DataLoader schema."
            ),
            registry_unit=(self.registry_unit(feature, column) if column in contract.measurement_columns else None),
            dense_default=(self.dense_default(feature, column, phase) if dense else None),
        )

    def undeclared_columns_warning(self, feature: str, columns: Sequence[str]) -> str:
        return (
            f"{len(columns)} column(s) in {feature} files have no declared role in the google_fitness "
            f"DataLoader schema: {', '.join(sorted(columns))}. They are returned only by projection='full' "
            "or an explicit columns= request. Add them to the source schema."
        )

    def overview_info(self, *, native_root: Path, curated_root: Path) -> InfoDocument:
        from wearable_project.DataLoaders.google_fitness.info import overview_report

        return overview_report(native_root=native_root, curated_root=curated_root)

    def feature_info(self, loader: Any, *, include_evidence: bool) -> InfoDocument:
        from wearable_project.DataLoaders.google_fitness.info import feature_report

        return feature_report(loader, include_evidence=include_evidence)

    def with_harmonized_values(self, data: Any, feature: str, phase: str) -> Any:
        from wearable_project.DataLoaders.google_fitness.derived import with_harmonized_values

        return with_harmonized_values(data, feature, phase)
