"""
Wearable Data Processing and Modeling project
Google Fitness curated schema-v2 DataLoader contracts and real-sample acceptance.
"""


from __future__ import annotations
from collections import Counter
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Iterable, Mapping
import numpy as np
import pandas as pd
import pytest
from wearable_project.DataLoaders import DataLoaderStateError
from wearable_project.DataLoaders._base import AppleHealthFeatureLoader
from wearable_project.DataLoaders._sources import APPLE_HEALTHKIT_PUBLIC_FEATURES
from wearable_project.DataLoaders.google_fitness.schema import CURATED_DATALOADER_SCHEMA_VERSION, get_schema
from wearable_project.DataLoaders.StepCountLoader import StepCountLoader
from wearable_project.curation.google_fitness.contract import (
    CONTRACT_SCHEMA_VERSION, CURATED_SCHEMA_VERSION, CURATION_CONTRACT_FILENAME, ROW_DEFAULTS,
    contract_fingerprint, virtual_row_values, write_contract_atomic,
)
from wearable_project.curation.google_fitness.engine import (
    ENGINE_VERSION, CuratedFeatureInfo, curate_feature_file,
)
from wearable_project.curation.google_fitness.pipeline import PIPELINE_VERSION
from wearable_project.curation.google_fitness.registry import (
    CURATION_REGISTRY_VERSION, POLICY_CONTRACT_VERSION, get_policy,
)
from wearable_project.curation.google_fitness.state import (
    CURATION_STATE_FILENAME, CURATION_STATE_SCHEMA_VERSION, CurationStateDatabase, CuratedOutputState,
)
from wearable_project.exceptions import DataLoaderConfigurationError, DataLoaderReadError
from wearable_project.processing.google_fitness.registry import (
    ColumnType, NativeFeatureFamily, REGISTRY_VERSION, STEP_COUNT_DATA_SOURCE_ID, feature_names, get_feature_spec,
)


GOOGLE_FEATURES = tuple(feature_names())
SHARED_FEATURES = frozenset(GOOGLE_FEATURES) & frozenset(APPLE_HEALTHKIT_PUBLIC_FEATURES)
GOOGLE_CURATED_SAMPLE = Path(os.environ.get(
    "WEARABLE_GOOGLE_CURATED_SAMPLE",
    "/home/evortigosa/Desktop/postdoc/code/cluster/wearable_data/curated_google_fitness",
))
needs_google_curated_sample = pytest.mark.skipif(
    not GOOGLE_CURATED_SAMPLE.is_dir(), reason="a managed Google Fitness curated sample root is required",
)


def _loader_class(feature: str) -> type[AppleHealthFeatureLoader]:
    module = importlib.import_module(f"wearable_project.DataLoaders.{feature}Loader")
    return getattr(module, f"{feature}Loader")


def _loader(feature: str, root: Path, **kwargs: Any) -> AppleHealthFeatureLoader:
    cls = _loader_class(feature)
    source = {"source": "google_fitness"} if feature in SHARED_FEATURES else {}
    return cls(phase="curated", root=root, **source, **kwargs)


def _base_row(feature: str, *, version: str | None = None, offset: int = 0, **overrides: Any,) -> dict[str, Any]:
    spec = get_feature_spec(feature)
    selected = version or spec.supported_versions[0]
    start = pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(hours=offset)
    end = start + pd.Timedelta(minutes=30)
    values: dict[str, Any] = {
        "start_date": start.isoformat().replace("+00:00", "Z"),
        "end_date": end.isoformat().replace("+00:00", "Z"),
        "outer_datetime": start.isoformat().replace("+00:00", "Z"),
        "summary_date": start.strftime("%Y-%m-%d"),
        "value": 17,
        "raw_value": 17,
        "calories": 1.25,
        "distance": 2.5,
        "quantity": 3.75,
        "tracked": True,
        "device": "phone",
        "source_name": "Google Fit",
        "source_id": "com.google.android.apps.fitness",
        "source_day_label": "יום שני",
        "added_by": "com.google.android.apps.fitness",
        "aggregation_source": "estimated_steps",
        "origin_data_source_id": STEP_COUNT_DATA_SOURCE_ID,
        "app_name": "Google Fit",
        "app_package_name": "com.google.android.apps.fitness",
        "device_manufacturer": "Acme",
        "device_model": "Model 1",
        "device_type": "phone",
        "device_uid": "000-device-uid",
        "collecting_method_version": selected,
        "outer_id": f"outer-{offset}",
        "payload_index": offset,
    }
    if feature == "Sleep":
        values["value"] = "SLEEP"
    elif feature in {"Calories", "Distance", "Height", "Weight"}:
        values["value"] = 1.5
    elif feature == "StepCount" and selected == "1.0":
        for name in (
            "summary_date", "start_date", "end_date", "raw_value", "aggregation_source",
            "origin_data_source_id", "app_name", "app_package_name", "device_manufacturer",
            "device_model", "device_type", "device_uid",
        ):
            values[name] = None
        values["payload_index"] = 0
    values.update(overrides)
    return {column: values.get(column) for column in spec.column_names}


def _write_native(path: Path, feature: str, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=get_feature_spec(feature).column_names).to_csv(path, index=False)
    return path


def _output_state(info: CuratedFeatureInfo) -> CuratedOutputState:
    return CuratedOutputState(
        feature=info.feature,
        filename=info.filename,
        native_rows=info.native_rows,
        curated_rows=info.curated_rows,
        size_bytes=info.size_bytes,
        sha256=info.curated_sha256,
        native_sha256=info.native_sha256,
        policy_fingerprint=info.policy_fingerprint,
        active_materialized_columns=info.active_materialized_columns,
        pass_rows=info.pass_rows,
        review_rows=info.review_rows,
        exclude_default_rows=info.exclude_default_rows,
        included_by_default_rows=info.included_by_default_rows,
        excluded_by_default_rows=info.excluded_by_default_rows,
        canonical_value_rows=info.canonical_value_rows,
        flag_counts=info.flag_counts,
        acquisition_counts=info.acquisition_counts,
        derivation_counts=info.derivation_counts,
        tracking_counts=info.tracking_counts,
        step_count_role_counts=info.step_count_role_counts,
        unit_status_counts=info.unit_status_counts,
        unit_evidence_counts=info.unit_evidence_counts,
    )


def _write_contract(root: Path) -> None:
    write_contract_atomic(
        root / CURATION_CONTRACT_FILENAME,
        engine_version=ENGINE_VERSION,
        pipeline_version=PIPELINE_VERSION,
        state_schema_version=CURATION_STATE_SCHEMA_VERSION,
    )


def _build_root(
    tmp_path: Path, products: Iterable[tuple[str, str, list[dict[str, Any]]]], *, with_state: bool = True,
) -> tuple[Path, dict[tuple[str, str], CuratedFeatureInfo]]:
    """Curate exact native fixtures with the accepted engine, then optionally write managed state."""

    root = tmp_path / "curated-google"
    native_root = tmp_path / "native-google"
    root.mkdir(parents=True, exist_ok=True)
    infos: dict[tuple[str, str], CuratedFeatureInfo] = {}
    by_participant: dict[str, list[CuratedOutputState]] = {}
    for participant, feature, rows in products:
        native = _write_native(native_root / participant / f"{feature}.csv", feature, rows)
        destination = root / participant / f"{feature}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        info = curate_feature_file(native, destination, feature).info
        infos[(participant, feature)] = info
        by_participant.setdefault(participant, []).append(_output_state(info))
    _write_contract(root)
    if with_state:
        with CurationStateDatabase(root / CURATION_STATE_FILENAME) as state:
            for participant, outputs in sorted(by_participant.items()):
                state.commit_participant(
                    participant,
                    native_signature=f"native:{participant}",
                    policy_signature=f"policy:{participant}",
                    native_files=(),
                    outputs=outputs,
                )
            state.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    for suffix in ("-wal", "-shm", "-journal"):
        (root / (CURATION_STATE_FILENAME + suffix)).unlink(missing_ok=True)
    return root, infos


def _edit_state(root: Path, sql: str, parameters: tuple[Any, ...] = ()) -> None:
    connection = sqlite3.connect(root / CURATION_STATE_FILENAME)
    connection.execute(sql, parameters)
    connection.commit()
    connection.close()
    for suffix in ("-wal", "-shm", "-journal"):
        (root / (CURATION_STATE_FILENAME + suffix)).unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_virtual(feature: str, physical_row: Mapping[str, str]) -> dict[str, str]:
    return virtual_row_values(feature, physical_row, get_policy(feature, allow_fallback=False))


def _assert_virtual_row(feature: str, loaded: pd.Series, physical_row: Mapping[str, str]) -> None:
    expected = _expected_virtual(feature, physical_row)
    for column in get_policy(feature, allow_fallback=False).output.virtual_columns:
        actual = loaded[column]
        if pd.isna(actual):
            actual_text = ""
        elif column == "canonical_value":
            number = float(actual)
            actual_text = str(int(number)) if number.is_integer() else str(number)
        else:
            actual_text = str(actual)
        assert actual_text == expected.get(column, ""), (feature, column, expected, actual_text)


# --------------------------------------------------------------------- complete source catalog
@pytest.mark.parametrize("feature", GOOGLE_FEATURES)
def test_every_registered_google_feature_supports_curated_loading(feature: str):
    loader = _loader_class(feature)(
        source="google_fitness" if feature in SHARED_FEATURES else None,
        phase="curated",
    )
    assert loader.phase == "curated"
    assert loader.source.value == "google_fitness"
    assert loader._source_adapter.supports_feature(feature, "curated")


@pytest.mark.parametrize("feature", GOOGLE_FEATURES)
def test_every_registered_feature_round_trips_curated_logical_schema(tmp_path: Path, feature: str):
    version = "2.0" if feature == "Height" else None
    root, _ = _build_root(tmp_path, [("001", feature, [_base_row(feature, version=version)])], with_state=False)
    data = _loader(feature, root, state_validation="off").get_data(projection="full")
    policy = get_policy(feature, allow_fallback=False)
    expected_columns = [
        *get_feature_spec(feature).column_names,
        *policy.output.materialized_columns,
        *policy.output.virtual_columns,
    ]
    assert data.df.columns.tolist() == expected_columns
    assert data.load_report["columns_undeclared"] == []
    assert data.load_report["curation_contract_validated"] is True
    assert data.load_report["source_contract"]["dataloader_schema_version"] == CURATED_DATALOADER_SCHEMA_VERSION
    physical_path = root / "001" / f"{feature}.csv"
    with physical_path.open(newline="", encoding="utf-8") as handle:
        physical_row = next(csv.DictReader(handle))
    _assert_virtual_row(feature, data.df.iloc[0], physical_row)


def test_native_and_curated_google_schema_caches_are_phase_isolated_in_both_orders(tmp_path: Path):
    native_root = tmp_path / "native"
    _write_native(native_root / "001" / "StepCount.csv", "StepCount", [_base_row("StepCount")])
    curated_root, _ = _build_root(
        tmp_path / "curated-fixture",
        [("001", "StepCount", [_base_row("StepCount")])],
        with_state=False,
    )

    for first, second in (("native", "curated"), ("curated", "native")):
        loaded: dict[str, Any] = {}
        for phase in (first, second):
            root = native_root if phase == "native" else curated_root
            loaded[phase] = StepCountLoader(
                source="google_fitness", phase=phase, root=root, state_validation="off",
            ).get_data(projection="full", sort_index=False)
        assert "step_count_role" not in loaded["native"].df.columns
        assert "canonical_value" not in loaded["native"].df.columns
        assert "step_count_role" in loaded["curated"].df.columns
        assert "canonical_value" in loaded["curated"].df.columns
        assert loaded["native"].load_report["source_contract"]["dataloader_schema_version"] != (
            loaded["curated"].load_report["source_contract"]["dataloader_schema_version"]
        )


# ------------------------------------------------------------------ root contract and source guards
def test_curated_contract_is_required_even_when_state_validation_is_off(tmp_path: Path):
    root, _ = _build_root(
        tmp_path, [("001", "Walking", [_base_row("Walking")])], with_state=False,
    )
    (root / CURATION_CONTRACT_FILENAME).unlink()
    with pytest.raises(DataLoaderStateError, match="requires .curation_contract.json"):
        _loader("Walking", root, state_validation="off").get_data()


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("invalid_json", "invalid Google Fitness curation contract"),
        ("duplicate_key", "repeats JSON key"),
        ("fingerprint", "contract fingerprint differs"),
        ("semantic", "contract field differs: features"),
        ("state_version", "implementation state schema differs"),
    ],
)
def test_incompatible_root_contract_is_rejected_before_csv_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, match: str,
):
    root, _ = _build_root(tmp_path, [("001", "Walking", [_base_row("Walking")])])
    path = root / CURATION_CONTRACT_FILENAME
    if mutation == "invalid_json":
        path.write_text("{", encoding="utf-8")
    elif mutation == "duplicate_key":
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("{", '{"source_key":"google_fitness",', 1), encoding="utf-8")
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "fingerprint":
            payload["contract_fingerprint"] = "different"
        elif mutation == "semantic":
            payload["features"]["Walking"]["row_meaning"] = "changed"
        else:
            payload["implementation"]["state_schema_version"] = "old-state"
        path.write_text(json.dumps(payload), encoding="utf-8")

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("CSV parsing occurred before root-contract validation")

    monkeypatch.setattr(AppleHealthFeatureLoader, "_read_feature_file", forbidden)
    with pytest.raises(DataLoaderStateError, match=match):
        _loader("Walking", root, state_validation="off").get_data()


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("source_key", "other", "could not establish the requested source contract"),
        ("product_kind", "native", "incompatible Google Fitness curated-root state metadata"),
        ("state_schema_version", "old", "incompatible Google Fitness curated-root state metadata"),
        ("curated_schema_version", "old", "incompatible Google Fitness curated-root state metadata"),
        ("contract_fingerprint", "old", "incompatible Google Fitness curated-root state metadata"),
    ],
)
def test_incompatible_curated_state_metadata_is_rejected_before_csv_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: str, match: str,
):
    root, _ = _build_root(tmp_path, [("001", "Walking", [_base_row("Walking")])])
    _edit_state(root, "UPDATE metadata SET value=? WHERE key=?", (value, key))

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("CSV parsing occurred before state metadata validation")

    monkeypatch.setattr(AppleHealthFeatureLoader, "_read_feature_file", forbidden)
    with pytest.raises((DataLoaderStateError, DataLoaderConfigurationError), match=match):
        _loader("Walking", root, state_validation="off").get_data()


# ---------------------------------------------------------- physical schema and sparse reconstruction
def test_absent_sparse_columns_are_reconstructed_from_contract_defaults(tmp_path: Path):
    root, _ = _build_root(tmp_path, [("001", "Walking", [_base_row("Walking")])], with_state=False)
    physical = pd.read_csv(root / "001" / "Walking.csv", keep_default_na=False)
    assert physical.columns.tolist()[-1:] == ["acquisition_method"]
    data = _loader("Walking", root, state_validation="off").get_data(projection="full")
    assert data.df["curation_status"].astype(str).tolist() == [ROW_DEFAULTS["curation_status"]]
    assert data.df["curation_flags"].astype(str).tolist() == [""]
    assert data.df["include_by_default"].tolist() == [True]
    assert data.load_report["rows_reconstructed"] == {
        "curation_flags": 1, "curation_status": 1, "include_by_default": 1,
    }
    assert data.load_report["columns_reconstructed"] == [
        "curation_status", "curation_flags", "include_by_default",
    ]


def test_blank_sparse_cells_take_defaults_but_explicit_decisions_remain(tmp_path: Path):
    rows = [_base_row("Calories", value=1.0), _base_row("Calories", offset=1, value=-1.0)]
    root, _ = _build_root(tmp_path, [("001", "Calories", rows)], with_state=False)
    path = root / "001" / "Calories.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    # Preserve the negative row's explicit decision while materializing blank defaults for the positive row.
    for column in ("curation_status", "curation_flags", "include_by_default"):
        if column not in frame:
            frame[column] = ""
    frame.loc[0, ["curation_status", "curation_flags", "include_by_default"]] = ""
    native = get_feature_spec("Calories").column_names
    policy = get_policy("Calories", allow_fallback=False)
    frame.loc[:, [*native, *policy.output.materialized_columns]].to_csv(path, index=False)
    data = _loader("Calories", root, state_validation="off").get_data(projection="full")
    assert data.df["curation_status"].astype(str).tolist()[0] == "pass"
    assert data.df["include_by_default"].tolist()[0] is True
    assert data.df["include_by_default"].tolist()[1] is False
    assert data.load_report["rows_reconstructed"]["include_by_default"] == 1


@pytest.mark.parametrize(
    "edit,match",
    [
        ("unknown_suffix", "outside the curated schema-v2 materialized contract"),
        ("wrong_order", "materialized-column order"),
        ("missing_acquisition", "missing fixed curated schema-v2"),
        ("bad_status", "unrecognized curation_status"),
        ("bad_boolean", "must hold true/false flags"),
        ("bad_acquisition", "schema-v2 contract reconstructs"),
    ],
)
def test_malformed_physical_curated_schema_is_refused(tmp_path: Path, edit: str, match: str,):
    root, _ = _build_root(tmp_path, [("001", "Walking", [_base_row("Walking")])], with_state=False)
    path = root / "001" / "Walking.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    if edit == "unknown_suffix":
        frame["future_decision"] = "x"
    elif edit == "wrong_order":
        frame["curation_status"] = ""
        frame = frame.loc[:, [*frame.columns[:-2], "curation_status", "acquisition_method"]]
    elif edit == "missing_acquisition":
        frame = frame.drop(columns=["acquisition_method"])
    elif edit == "bad_status":
        frame["curation_status"] = "unknown-status"
    elif edit == "bad_boolean":
        frame["include_by_default"] = "maybe"
    else:
        frame["acquisition_method"] = "unknown"
    frame.to_csv(path, index=False)
    with pytest.raises(DataLoaderReadError, match=match):
        _loader("Walking", root, state_validation="off").get_data()


def test_native_prefix_order_and_values_remain_authoritative(tmp_path: Path):
    root, _ = _build_root(tmp_path, [("001", "Distance", [_base_row("Distance")])], with_state=False)
    path = root / "001" / "Distance.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    order = frame.columns.tolist()
    order[0], order[1] = order[1], order[0]
    frame.loc[:, order].to_csv(path, index=False)
    with pytest.raises(DataLoaderReadError, match="native prefix"):
        _loader("Distance", root, state_validation="off").get_data()


# ----------------------------------------------------------- projections, virtual values, and dtypes
def test_curated_activity_projections_preserve_analysis_fields_and_privacy(tmp_path: Path):
    row = _base_row("Walking", source_id="0012", device="watch", source_name="Personal Watch")
    root, _ = _build_root(tmp_path, [("001", "Walking", [row])], with_state=False)
    loader = _loader("Walking", root, state_validation="off")
    default = loader.get_data().df
    analysis = loader.get_data(projection="analysis").df
    full = loader.get_data(projection="full").df
    for name in ("calories", "distance", "quantity", "acquisition_method", "derivation_method", "tracking_status"):
        assert name in default
    for name in ("source_id", "source_name", "outer_id"):
        assert name not in default
    assert "device" in analysis and "outer_id" in analysis and "payload_index" in analysis
    assert "source_id" not in analysis and "source_name" not in analysis
    assert "source_id" in full and "source_name" in full


def test_curated_step_count_projection_and_dtypes(tmp_path: Path):
    rows = [
        _base_row("StepCount", version="1.0", value=11),
        _base_row("StepCount", version="2.0", offset=1, value=20, raw_value=20),
        _base_row(
            "StepCount", version="2.0", offset=1, value=18, raw_value=18,
            aggregation_source="merge_step_deltas", payload_index=1,
        ),
    ]
    root, _ = _build_root(tmp_path, [("001", "StepCount", rows)], with_state=False)
    loader = _loader("StepCount", root, state_validation="off")
    default = loader.get_data()
    analysis = loader.get_data(projection="analysis")
    full = loader.get_data(projection="full")
    assert str(default.df["value"].dtype) == "Int64"
    assert str(default.df["canonical_value"].dtype) == "Int64"
    assert str(default.df["include_by_default"].dtype) == "boolean"
    assert str(default.df["summary_date"].dtype) == "datetime64[ns]"
    assert isinstance(default.df["outer_datetime"].dtype, pd.DatetimeTZDtype)
    assert "unit_evidence" not in default.df
    assert "unit_evidence" in analysis.df
    for restricted in ("origin_data_source_id", "app_package_name", "device_uid"):
        assert restricted not in analysis.df and restricted in full.df
    assert default.load_report["date_column"] == "outer_datetime"
    assert default.load_report["date_semantics"] == "source_outer_summary_datetime_not_canonical_event_day"


def test_single_batch_and_empty_curated_dtypes_are_stable(tmp_path: Path):
    products = [
        ("001", "Calories", [_base_row("Calories", value=1.0)]),
        ("002", "Calories", [_base_row("Calories", offset=1, value=-1.0)]),
    ]
    root, _ = _build_root(tmp_path, products, with_state=False)
    loader = _loader("Calories", root, state_validation="off")
    one = loader.get_data(registration_codes="001")
    batch = loader.get_data(registration_codes=["002", "001"])
    empty = loader.get_data(start_date="2099-01-01")
    assert one.df.columns.tolist() == batch.df.columns.tolist() == empty.df.columns.tolist()
    assert one.df.dtypes.astype(str).to_dict() == batch.df.dtypes.astype(str).to_dict()
    assert batch.df.dtypes.astype(str).to_dict() == empty.df.dtypes.astype(str).to_dict()


@pytest.mark.parametrize(
    "feature,version,expected_status,expected_derivation",
    [
        ("Calories", None, "withheld_exporter_mapping_pending", "application_summary"),
        ("Distance", None, "withheld_exporter_mapping_pending", "application_summary"),
        ("Weight", "1.0", "withheld_exporter_mapping_pending", "raw"),
        ("Height", "2.0", "withheld_exporter_mapping_pending", "raw"),
        ("StepCount", "1.0", "resolved_source_convention", "legacy_summary"),
        ("StepCount", "2.0", "resolved_payload_explicit", "estimated"),
    ],
)
def test_scalar_virtual_unit_and_derivation_contracts(
    tmp_path: Path, feature: str, version: str | None, expected_status: str, expected_derivation: str,
):
    root, _ = _build_root(
        tmp_path, [("001", feature, [_base_row(feature, version=version)])], with_state=False,
    )
    row = _loader(feature, root, state_validation="off").get_data(projection="full").df.iloc[0]
    assert str(row["curation_unit_status"]) == expected_status
    assert str(row["derivation_method"]) == expected_derivation
    if feature == "StepCount":
        assert row["canonical_value"] == row["value"]
        assert str(row["canonical_unit"]) == "count"
    else:
        assert pd.isna(row["canonical_value"]) and pd.isna(row["canonical_unit"])


# ---------------------------------------------------------- StepCount group roles and inclusion policy
def _step_group_rows(kind: str) -> list[dict[str, Any]]:
    base = dict(version="2.0", offset=0, outer_id="group", summary_date="2024-01-01")
    if kind == "preferred_pair":
        return [
            _base_row("StepCount", value=20, raw_value=20, aggregation_source="estimated_steps", payload_index=0, **base),
            _base_row("StepCount", value=18, raw_value=18, aggregation_source="merge_step_deltas", payload_index=1, **base),
        ]
    if kind == "merge_fallback":
        return [_base_row("StepCount", value=18, raw_value=18, aggregation_source="merge_step_deltas", payload_index=0, **base)]
    if kind == "third_party_fallback":
        return [_base_row("StepCount", value=18, raw_value=18, aggregation_source="vendor_sum", payload_index=0, **base)]
    if kind == "ambiguous":
        return [
            _base_row("StepCount", value=20, raw_value=20, aggregation_source="estimated_steps", payload_index=0, **base),
            _base_row("StepCount", value=18, raw_value=18, aggregation_source="estimated_steps", payload_index=1, **base),
        ]
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind,roles,included,statuses",
    [
        ("preferred_pair", ["fit_app_preferred", "parallel_nonpreferred"], [True, False], ["review", "review"]),
        ("merge_fallback", ["merge_fallback"], [True], ["review"]),
        ("third_party_fallback", ["third_party_fallback"], [True], ["review"]),
        ("ambiguous", ["ambiguous_parallel", "ambiguous_parallel"], [False, False], ["review", "review"]),
    ],
)
def test_step_count_group_roles_and_default_inclusion(
    tmp_path: Path, kind: str, roles: list[str], included: list[bool], statuses: list[str],
):
    root, _ = _build_root(tmp_path, [("001", "StepCount", _step_group_rows(kind))], with_state=False)
    loader = _loader("StepCount", root, state_validation="off")
    data = loader.get_data(projection="full", sort_index=False)
    assert data.df["step_count_role"].astype(str).tolist() == roles
    assert data.df["include_by_default"].tolist() == included
    assert data.df["curation_status"].astype(str).tolist() == statuses
    filtered = loader.get_data(default_inclusion_only=True, sort_index=False)
    assert len(filtered.df) == sum(included)
    assert filtered.load_report["default_inclusion_rows_dropped"] == len(included) - sum(included)


def test_tampered_step_count_role_is_refused_without_state(tmp_path: Path):
    root, _ = _build_root(tmp_path, [("001", "StepCount", _step_group_rows("preferred_pair"))], with_state=False)
    path = root / "001" / "StepCount.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    frame.loc[0, "step_count_role"] = "ambiguous_parallel"
    frame.to_csv(path, index=False)
    with pytest.raises(DataLoaderReadError, match="group contract reconstructs"):
        _loader("StepCount", root, state_validation="off").get_data()


# ---------------------------------------------------------------------- managed-state verification
def test_managed_curated_file_verifies_every_persisted_counter(tmp_path: Path):
    root, infos = _build_root(
        tmp_path,
        [("001", "Calories", [_base_row("Calories", value=1.0), _base_row("Calories", offset=1, value=-1.0)])],
    )
    data = _loader("Calories", root, state_validation="required").get_data(projection="full")
    info = infos[("001", "Calories")]
    assert data.load_report["state_verified"] is True
    assert data.load_report["files_verified"] == 1
    assert data.load_report["curation_contract"]["contract_fingerprint"] == contract_fingerprint()
    assert data.load_report["curation_status_counts"] == {
        key: value
        for key, value in {
            "exclude_default": info.exclude_default_rows,
            "pass": info.pass_rows,
            "review": info.review_rows,
        }.items()
        if value
    }
    assert data.load_report["canonical_value_rows"] == info.canonical_value_rows


@pytest.mark.parametrize(
    "sql,match",
    [
        ("UPDATE curation_outputs SET curated_rows=curated_rows+1", "curated rows"),
        ("UPDATE curation_outputs SET canonical_value_rows=canonical_value_rows+1", "canonical-value rows"),
        ("UPDATE curation_outputs SET policy_fingerprint='old'", "policy fingerprint"),
        ("UPDATE curation_outputs SET active_materialized_columns_json='[]'", "active materialized columns"),
        ("UPDATE curation_outputs SET derivation_counts_json='{}'", "derivation counts"),
        ("UPDATE curation_outputs SET unit_status_counts_json='{}'", "unit-status counts"),
    ],
)
def test_curated_state_disagreement_is_rejected(tmp_path: Path, sql: str, match: str):
    root, _ = _build_root(tmp_path, [("001", "Calories", [_base_row("Calories")])])
    _edit_state(root, sql)
    with pytest.raises(DataLoaderStateError, match=match):
        _loader("Calories", root, state_validation="required").get_data()


def test_same_size_curated_edit_is_caught_by_sha256(tmp_path: Path):
    root, _ = _build_root(tmp_path, [("001", "Calories", [_base_row("Calories", value=10.0)])])
    path = root / "001" / "Calories.csv"
    original = path.read_text(encoding="utf-8")
    edited = original.replace(",10.0,", ",11.0,", 1)
    assert edited != original and len(edited) == len(original)
    path.write_text(edited, encoding="utf-8")
    with pytest.raises(DataLoaderStateError, match="sha256"):
        _loader("Calories", root, state_validation="required").get_data()


def test_state_validation_off_skips_hash_but_not_contract_or_semantic_guards(tmp_path: Path):
    root, _ = _build_root(tmp_path, [("001", "Calories", [_base_row("Calories", value=10.0)])])
    path = root / "001" / "Calories.csv"
    text = path.read_text(encoding="utf-8").replace(",10.0,", ",11.0,", 1)
    path.write_text(text, encoding="utf-8")
    data = _loader("Calories", root, state_validation="off").get_data()
    assert data.load_report["state_verified"] is False
    assert data.df["value"].tolist() == [11.0]
    assert data.load_report["curation_contract_validated"] is True


# ---------------------------------------------------------------- helper and profile behavior
def test_curated_step_count_harmonizes_from_curation_virtuals(tmp_path: Path):
    root, _ = _build_root(tmp_path, [("001", "StepCount", [_base_row("StepCount", version="1.0", value=11)])])
    result = _loader("StepCount", root, state_validation="required").get_data().with_harmonized_values()
    assert result.df["harmonized_value"].tolist() == [11.0]
    assert result.df["harmonized_unit"].astype(str).tolist() == ["count"]
    assert result.df["harmonized_unit_source"].astype(str).tolist() == ["curation"]
    report = result.load_report["derived"]["harmonized_values"]
    assert report["curation_unit_status_counts"] == {"resolved_source_convention": 1}


@pytest.mark.parametrize(
    "feature,candidate",
    [("Calories", "kcal"), ("Distance", "m"), ("Weight", "kg"), ("Height", "m")],
)
def test_curated_withheld_units_remain_unresolved(tmp_path: Path, feature: str, candidate: str):
    version = "2.0" if feature == "Height" else None
    root, _ = _build_root(tmp_path, [("001", feature, [_base_row(feature, version=version)])])
    result = _loader(feature, root, state_validation="required").get_data().with_harmonized_values()
    assert result.df["harmonized_value"].isna().all()
    assert result.df["harmonized_unit"].isna().all()
    assert result.df["harmonized_unit_source"].astype(str).tolist() == ["unresolved"]
    report = result.load_report["derived"]["harmonized_values"]
    assert report["unit_withheld_by_curation"] is True
    assert report["unit_candidates"] == [candidate]


def test_curated_profile_uses_state_only_and_exposes_schema_v2_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root, _ = _build_root(tmp_path, [("001", "StepCount", _step_group_rows("preferred_pair"))])

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("profile parsed a CSV")

    monkeypatch.setattr(AppleHealthFeatureLoader, "_read_feature_file", forbidden)
    profile = _loader("StepCount", root, state_validation="required").profile()
    summary = profile.summary
    assert summary["rows"] == 2
    assert summary["step_count_roles"] == {"fit_app_preferred": 1, "parallel_nonpreferred": 1}
    assert summary["provenance"]["derivation_method_counts"] == {"estimated": 1, "merged": 1}
    assert summary["unit_resolution"]["canonical_value_rows"] == 2
    assert summary["unit_resolution"]["resolved_fraction"] == 1.0
    assert summary["materialized_columns"]["file_counts"]["step_count_role"] == 1
    assert summary["curation_contract"]["contract_fingerprint"] == contract_fingerprint()
    assert str(profile.participants["canonical_value_rows"].dtype) == "Int64"
    assert "StepCount roles" in profile.text
    assert "Curation flags" in profile.text


def test_curated_profile_state_off_remains_zero_csv_and_contract_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root, _ = _build_root(tmp_path, [("001", "Walking", [_base_row("Walking")])])

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("profile parsed a CSV")

    monkeypatch.setattr(AppleHealthFeatureLoader, "_read_feature_file", forbidden)
    profile = _loader("Walking", root, state_validation="off").profile()
    assert profile.summary["source"] == "filesystem"
    assert profile.summary["rows"] is None
    assert profile.summary["curation_contract"]["contract_fingerprint"] == contract_fingerprint()
    assert profile.summary["source_contract"]["root_contract"]["contract_fingerprint"] == contract_fingerprint()
    assert "StepCount roles" not in profile.text


# -------------------------------------------------------------------------- real sample acceptance
def _sample_outputs(root: Path) -> list[tuple[str, str, int]]:
    connection = sqlite3.connect(f"file:{root / CURATION_STATE_FILENAME}?immutable=1", uri=True)
    rows = connection.execute(
        "SELECT participant_id,feature,curated_rows FROM curation_outputs ORDER BY feature,participant_id"
    ).fetchall()
    connection.close()
    return [(str(participant), str(feature), int(rows)) for participant, feature, rows in rows]


@needs_google_curated_sample
def test_real_curated_sample_every_observed_feature_verifies_profiles_and_preserves_rows():
    outputs = _sample_outputs(GOOGLE_CURATED_SAMPLE)
    expected = Counter()
    for _, feature, rows in outputs:
        expected[feature] += rows
    assert len(expected) == 16
    assert sum(expected.values()) == 145_613
    loaded = 0
    for feature, rows in sorted(expected.items()):
        loader = _loader(feature, GOOGLE_CURATED_SAMPLE, state_validation="required")
        data = loader.get_data(projection="full", max_rows=None)
        profile = loader.profile()
        assert len(data.df) == rows, feature
        assert profile.summary["rows"] == rows, feature
        assert data.load_report["state_verified"] is True, feature
        assert data.load_report["files_verified"] == data.load_report["files_read"], feature
        assert data.load_report["curation_contract_validated"] is True, feature
        assert data.load_report["columns_undeclared"] == [], feature
        loaded += len(data.df)
    assert loaded == 145_613


@needs_google_curated_sample
def test_real_curated_full_projection_preserves_every_physical_cell_and_virtual_oracle():
    outputs = _sample_outputs(GOOGLE_CURATED_SAMPLE)
    participants_by_feature: dict[str, list[str]] = {}
    for participant, feature, _ in outputs:
        participants_by_feature.setdefault(feature, []).append(participant)

    for feature, participants in sorted(participants_by_feature.items()):
        spec = get_feature_spec(feature)
        policy = get_policy(feature, allow_fallback=False)
        data = _loader(feature, GOOGLE_CURATED_SAMPLE, state_validation="required").get_data(
            registration_codes=sorted(participants), projection="full", sort_index=False, max_rows=None,
        )
        physical_frames: list[pd.DataFrame] = []
        raw_rows: list[dict[str, str]] = []
        physical_columns: list[str] = []
        for participant in sorted(participants):
            path = GOOGLE_CURATED_SAMPLE / participant / f"{feature}.csv"
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                raw_rows.extend(rows)
                if not physical_columns:
                    physical_columns = list(reader.fieldnames or [])
            physical_frames.append(pd.read_csv(
                path,
                keep_default_na=False,
                na_values=[""],
                dtype={
                    column.name: str
                    for column in spec.columns
                    if column.data_type in {ColumnType.STRING, ColumnType.DATE}
                },
                float_precision="round_trip",
            ))
        raw = pd.concat(physical_frames, ignore_index=True, sort=False)
        assert len(data.df) == len(raw) == len(raw_rows), feature
        for column in spec.column_names:
            actual = data.df[column].reset_index(drop=True)
            expected = raw[column]
            column_spec = next(item for item in spec.columns if item.name == column)
            if column_spec.data_type is ColumnType.TIMESTAMP_UTC:
                expected = pd.to_datetime(expected, utc=True, errors="raise", format="mixed")
                assert actual.equals(expected), (feature, column)
            elif column_spec.data_type is ColumnType.DATE:
                expected = pd.to_datetime(expected, errors="raise", format="%Y-%m-%d")
                assert actual.equals(expected), (feature, column)
            elif column_spec.data_type in {ColumnType.NUMBER, ColumnType.INTEGER}:
                assert np.array_equal(
                    actual.astype("Float64").to_numpy(dtype=float, na_value=np.nan),
                    pd.to_numeric(expected, errors="raise").astype("Float64").to_numpy(dtype=float, na_value=np.nan),
                    equal_nan=True,
                ), (feature, column)
            elif column_spec.data_type is ColumnType.BOOLEAN:
                expected_bool = expected.map({True: True, False: False, "True": True, "False": False, 1: True, 0: False})
                assert actual.astype("boolean").reset_index(drop=True).equals(
                    expected_bool.astype("boolean").reset_index(drop=True)
                ), (feature, column)
            else:
                assert actual.astype("string").fillna("<NA>").equals(
                    expected.astype("string").fillna("<NA>")
                ), (feature, column)

        # Check every virtual value against the row-level scientific oracle, not against loader implementation.
        loaded = data.df.reset_index(drop=True)
        for index, raw_row in enumerate(raw_rows):
            _assert_virtual_row(feature, loaded.iloc[index], raw_row)
        assert data.df.columns.tolist() == [
            *spec.column_names, *policy.output.materialized_columns, *policy.output.virtual_columns,
        ]


@needs_google_curated_sample
def test_real_step_count_roles_inclusion_units_and_versions():
    loader = _loader("StepCount", GOOGLE_CURATED_SAMPLE, state_validation="required")
    data = loader.get_data(projection="full", max_rows=None)
    assert data.df["step_count_role"].astype(str).value_counts().to_dict() == {
        "legacy_v1": 3_128,
        "fit_app_preferred": 1_869,
        "parallel_nonpreferred": 1_796,
    }
    assert data.df["collecting_method_version"].astype(str).value_counts().to_dict() == {
        "2.0": 3_665, "1.0": 3_128,
    }
    assert data.df["canonical_value"].notna().sum() == 6_793
    filtered = loader.get_data(default_inclusion_only=True, max_rows=None)
    assert len(filtered.df) == 4_997
    assert filtered.df["step_count_role"].astype(str).value_counts().to_dict() == {
        "legacy_v1": 3_128, "fit_app_preferred": 1_869,
    }


@needs_google_curated_sample
def test_real_negative_calories_default_inclusion_policy():
    loader = _loader("Calories", GOOGLE_CURATED_SAMPLE, state_validation="required")
    all_rows = loader.get_data(projection="full", max_rows=None)
    filtered = loader.get_data(default_inclusion_only=True, max_rows=None)
    assert len(all_rows.df) == 7_765
    assert len(filtered.df) == 7_359
    assert all_rows.load_report["curation_status_counts"] == {
        "exclude_default": 164, "pass": 7_359, "review": 242,
    }
    assert filtered.load_report["default_inclusion_rows_dropped"] == 406


@needs_google_curated_sample
def test_real_curated_loader_is_read_only():
    tracked = [
        GOOGLE_CURATED_SAMPLE / CURATION_STATE_FILENAME,
        GOOGLE_CURATED_SAMPLE / CURATION_CONTRACT_FILENAME,
        *sorted(GOOGLE_CURATED_SAMPLE.glob("*/*.csv")),
    ]
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns, _sha256(path)) for path in tracked}
    _loader("Walking", GOOGLE_CURATED_SAMPLE, state_validation="required").get_data(max_rows=None)
    _loader("StepCount", GOOGLE_CURATED_SAMPLE, state_validation="required").profile()
    after = {path: (path.stat().st_size, path.stat().st_mtime_ns, _sha256(path)) for path in tracked}
    assert after == before
    assert not list(GOOGLE_CURATED_SAMPLE.glob(f"{CURATION_STATE_FILENAME}-*"))


# Static version assertions make accidental contract drift a reviewed change rather than a silent one.
def test_curated_loader_reports_exact_accepted_contract_versions(tmp_path: Path):
    root, _ = _build_root(tmp_path, [("001", "Walking", [_base_row("Walking")])])
    report = _loader("Walking", root, state_validation="required").get_data().load_report
    contract = report["source_contract"]
    assert contract == {
        "dataloader_schema_version": CURATED_DATALOADER_SCHEMA_VERSION,
        "native_registry_version": REGISTRY_VERSION,
        "feature_family": "activity_interval",
        "supported_collecting_method_versions": ["1.0"],
        "curated_schema_version": CURATED_SCHEMA_VERSION,
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "curation_registry_version": CURATION_REGISTRY_VERSION,
        "policy_contract_version": POLICY_CONTRACT_VERSION,
        "curation_state_schema_version": CURATION_STATE_SCHEMA_VERSION,
        "contract_fingerprint": contract_fingerprint(),
        "root_contract": report["curation_contract"],
    }
