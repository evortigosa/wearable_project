"""
Wearable Data Processing and Modeling project
Native Google Fitness DataLoader contracts, synthetic edge cases, and real-sample acceptance.
"""


from __future__ import annotations
from collections import Counter
import csv
import hashlib
import importlib
import os
from pathlib import Path
import sqlite3
from typing import Any
import numpy as np
import pandas as pd
import pytest
from wearable_project import DataLoaders
from wearable_project.DataLoaders import DataLoaderSizeError, DataLoaderStateError
from wearable_project.DataLoaders._base import AppleHealthFeatureLoader, WearableFeatureLoader
from wearable_project.DataLoaders._sources import APPLE_HEALTHKIT_PUBLIC_FEATURES, WearableSource
from wearable_project.DataLoaders.google_fitness.schema import get_schema
from wearable_project.DataLoaders.HeightLoader import HeightLoader
from wearable_project.DataLoaders.SleepLoader import SleepLoader
from wearable_project.DataLoaders.StepCountLoader import StepCountLoader
from wearable_project.DataLoaders.WeightLoader import WeightLoader
from wearable_project.exceptions import DataLoaderConfigurationError, DataLoaderReadError, DataLoaderSourceError
from wearable_project.processing.google_fitness.registry import (
    ColumnType, NativeFeatureFamily, REGISTRY_VERSION, STEP_COUNT_DATA_SOURCE_ID, feature_names, get_feature_spec,
)
from wearable_project.processing.google_fitness.writer import STATE_SCHEMA_VERSION


GOOGLE_FEATURES = tuple(feature_names())
SHARED_FEATURES = tuple(sorted(set(GOOGLE_FEATURES) & set(APPLE_HEALTHKIT_PUBLIC_FEATURES)))
GOOGLE_ONLY_FEATURES = tuple(feature for feature in GOOGLE_FEATURES if feature not in SHARED_FEATURES)
ACTIVITY_FEATURES = tuple(
    feature
    for feature in GOOGLE_FEATURES
    if get_feature_spec(feature).family is NativeFeatureFamily.ACTIVITY_INTERVAL
)

GOOGLE_NATIVE_SAMPLE = Path(os.environ.get(
    "WEARABLE_GOOGLE_NATIVE_SAMPLE",
    "/home/evortigosa/Desktop/postdoc/code/cluster/wearable_data/cleaned_google_fitness",
))
needs_google_sample = pytest.mark.skipif(
    not GOOGLE_NATIVE_SAMPLE.is_dir(),
    reason="a managed Google Fitness native sample root is required",
)


def _loader_class(feature: str) -> type[AppleHealthFeatureLoader]:
    module = importlib.import_module(f"wearable_project.DataLoaders.{feature}Loader")
    return getattr(module, f"{feature}Loader")


def _loader(feature: str, root: Path, **kwargs: Any) -> AppleHealthFeatureLoader:
    cls = _loader_class(feature)
    if feature in SHARED_FEATURES:
        return cls(source="google_fitness", phase="native", root=root, **kwargs)
    return cls(phase="native", root=root, **kwargs)


def _base_row(feature: str, *, version: str | None = None, offset: int = 0, **overrides: Any) -> dict[str, Any]:
    """One valid row in the exact registered physical schema for ``feature``."""

    spec = get_feature_spec(feature)
    selected_version = version or spec.supported_versions[0]
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
        "source_name": "None",
        "source_id": "0012",
        "source_day_label": "יום שני",
        "added_by": "com.example.writer",
        "aggregation_source": "estimated_steps",
        "origin_data_source_id": STEP_COUNT_DATA_SOURCE_ID,
        "app_name": "Google Fit",
        "app_package_name": "com.google.android.apps.fitness",
        "device_manufacturer": "Acme",
        "device_model": "Model 1",
        "device_type": "phone",
        "device_uid": "000-device-uid",
        "collecting_method_version": selected_version,
        "outer_id": f"000{offset + 1}",
        "payload_index": offset,
    }
    if feature == "Sleep":
        values["value"] = "SLEEP"
    elif feature in {"Calories", "Distance", "Height", "Weight"}:
        values["value"] = 1.5
    elif feature == "StepCount" and selected_version == "1.0":
        for name in (
            "summary_date", "start_date", "end_date", "raw_value", "aggregation_source",
            "origin_data_source_id", "app_name", "app_package_name", "device_manufacturer",
            "device_model", "device_type", "device_uid",
        ):
            values[name] = None
        values["payload_index"] = 0
    values.update(overrides)
    return {column: values.get(column) for column in spec.column_names}


def _write_feature(
    root: Path, feature: str, rows: list[dict[str, Any]], *, participant: str = "00123",
    extra_columns: dict[str, list[Any] | Any] | None = None,
) -> Path:
    path = root / participant / f"{feature}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = get_feature_spec(feature)
    frame = pd.DataFrame(rows, columns=spec.column_names)
    for name, values in (extra_columns or {}).items():
        if isinstance(values, list):
            frame[name] = values
        else:
            frame[name] = values
    frame.to_csv(path, index=False)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def _write_state(
    root: Path, *, statuses: dict[str, str] | None = None, metadata_overrides: dict[str, str] | None = None,
    output_overrides: dict[tuple[str, str], dict[str, Any]] | None = None,
    omit_outputs: set[tuple[str, str]] | None = None,
) -> Path:
    """Create the strict subset of native state consumed by the DataLoader."""

    database = root / ".wearable_state.sqlite"
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    metadata = {
        "source_key": "google_fitness",
        "state_schema_version": STATE_SCHEMA_VERSION,
        "registry_version": REGISTRY_VERSION,
    }
    metadata.update(metadata_overrides or {})
    connection.executemany("INSERT INTO metadata VALUES (?, ?)", sorted(metadata.items()))
    connection.execute("CREATE TABLE participants (participant_id TEXT PRIMARY KEY, status TEXT NOT NULL)")
    connection.execute(
        "CREATE TABLE feature_outputs ("
        "participant_id TEXT NOT NULL, feature TEXT NOT NULL, filename TEXT NOT NULL, "
        "row_count INTEGER NOT NULL, size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, "
        "represented_occurrences INTEGER NOT NULL, PRIMARY KEY (participant_id, feature))"
    )

    discovered: dict[str, str] = {}
    for path in sorted(root.glob("*/*.csv")):
        participant = path.parent.name
        feature = path.stem
        discovered[participant] = "complete"
        key = (participant, feature)
        if key in (omit_outputs or set()):
            continue
        entry: dict[str, Any] = {
            "participant_id": participant,
            "feature": feature,
            "filename": path.name,
            "row_count": _csv_rows(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "represented_occurrences": _csv_rows(path),
        }
        entry.update((output_overrides or {}).get(key, {}))
        connection.execute(
            "INSERT INTO feature_outputs VALUES (:participant_id, :feature, :filename, :row_count, "
            ":size_bytes, :sha256, :represented_occurrences)",
            entry,
        )
    discovered.update(statuses or {})
    connection.executemany(
        "INSERT INTO participants VALUES (?, ?)", sorted(discovered.items()),
    )
    connection.commit()
    connection.close()
    for suffix in ("-wal", "-shm", "-journal"):
        (root / (database.name + suffix)).unlink(missing_ok=True)
    return database


def _managed_feature_root(
    tmp_path: Path, feature: str, rows: list[dict[str, Any]], *, participant: str = "00123",
) -> Path:
    root = tmp_path / "google-native"
    _write_feature(root, feature, rows, participant=participant)
    _write_state(root)
    return root


# -------------------------------------------------------------------------------- public catalog
def test_google_catalog_has_one_top_level_public_loader_per_registered_feature():
    assert len(GOOGLE_FEATURES) == 67
    assert SHARED_FEATURES == ("Height", "Sleep", "StepCount", "Weight")
    for feature in GOOGLE_FEATURES:
        cls = _loader_class(feature)
        assert cls.feature_name == feature
        assert getattr(DataLoaders, f"{feature}Loader") is cls
        assert DataLoaders.__all__.count(f"{feature}Loader") == 1
    assert len(DataLoaders.__all__) == len(set(DataLoaders.__all__))


def test_google_implementation_package_does_not_expose_competing_feature_loader_modules():
    package = Path(DataLoaders.__file__).parent / "google_fitness"
    assert not list(package.glob("*Loader.py"))


@pytest.mark.parametrize("feature", GOOGLE_ONLY_FEATURES)
def test_google_only_public_loaders_default_to_google_source(feature: str):
    cls = _loader_class(feature)
    loader = cls(phase="native")
    assert cls.__mro__ == (cls, WearableFeatureLoader, AppleHealthFeatureLoader, object)
    assert loader.source is WearableSource.GOOGLE_FITNESS
    assert loader.phase == "native"
    with pytest.raises(DataLoaderSourceError, match="not available.*apple_healthkit"):
        cls(source="apple_healthkit", phase="native")


@pytest.mark.parametrize("feature", SHARED_FEATURES)
def test_shared_public_loaders_keep_apple_default_and_select_google_explicitly(feature: str):
    cls = _loader_class(feature)
    assert cls().source is WearableSource.APPLE_HEALTHKIT
    google = cls(source="google_fitness", phase="native")
    assert google.source is WearableSource.GOOGLE_FITNESS


def test_google_curated_loading_requires_the_schema_v2_root_contract(tmp_path: Path):
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(DataLoaderStateError, match="requires .curation_contract.json"):
        StepCountLoader(source="google_fitness", root=tmp_path).get_data()


# -------------------------------------------------------------------------- activity schema family
@pytest.mark.parametrize("feature", ACTIVITY_FEATURES)
def test_every_registered_activity_loader_reads_its_exact_schema(tmp_path: Path, feature: str):
    row = _base_row(feature)
    _write_feature(tmp_path, feature, [row])
    data = _loader(feature, tmp_path, state_validation="off").get_data(projection="full")
    spec = get_feature_spec(feature)
    assert list(data.df.columns) == list(spec.column_names)
    assert len(data.df) == 1
    assert data.load_report["columns_undeclared"] == []
    assert data.load_report["date_column"] == "start_date"
    assert isinstance(data.df.index.get_level_values("Date").dtype, pd.DatetimeTZDtype)
    for measurement in spec.measurement_columns:
        assert data.df_columns_metadata.loc[measurement, "role"] == "measurement"
        assert data.df_columns_metadata.loc[measurement, "registry_unit"] is None


def test_activity_projections_keep_measurements_and_withhold_identifiers(tmp_path: Path):
    feature = "Walking"
    _write_feature(tmp_path, feature, [_base_row(feature)])
    loader = _loader(feature, tmp_path, state_validation="off")
    default = loader.get_data().df
    analysis = loader.get_data(projection="analysis").df
    full = loader.get_data(projection="full").df
    assert list(default.columns) == [
        "start_date", "end_date", "calories", "distance", "quantity", "tracked",
        "collecting_method_version",
    ]
    assert list(analysis.columns) == [
        "start_date", "end_date", "calories", "distance", "quantity", "tracked", "device",
        "collecting_method_version", "outer_id", "payload_index",
    ]
    assert list(full.columns) == list(get_feature_spec(feature).column_names)
    assert {"source_name", "source_id"}.isdisjoint(analysis.columns)
    assert str(default["tracked"].dtype) == "boolean"
    assert str(default["collecting_method_version"].dtype) == "category"
    assert all(str(default[name].dtype) == "float64" for name in ("calories", "distance", "quantity"))


def test_activity_full_projection_preserves_unicode_literal_none_and_leading_zero_ids(tmp_path: Path):
    feature = "Walking"
    row = _base_row(feature, source_name="None", source_id="0012", device="שעון", outer_id="0007")
    _write_feature(tmp_path, feature, [row])
    frame = _loader(feature, tmp_path, state_validation="off").get_data(projection="full").df
    assert frame.iloc[0]["source_name"] == "None"
    assert frame.iloc[0]["source_id"] == "0012"
    assert str(frame.iloc[0]["device"]) == "שעון"
    assert frame.iloc[0]["outer_id"] == "0007"


def test_unknown_future_activity_column_is_withheld_but_reachable_without_lexical_loss(tmp_path: Path):
    feature = "Walking"
    _write_feature(
        tmp_path,
        feature,
        [_base_row(feature, offset=0), _base_row(feature, offset=1)],
        extra_columns={"future_field": ["0012", "None"]},
    )
    loader = _loader(feature, tmp_path, state_validation="off")
    with pytest.warns(UserWarning, match="future_field"):
        normal = loader.get_data()
    assert "future_field" not in normal.df.columns
    assert normal.load_report["columns_undeclared"] == ["future_field"]
    with pytest.warns(UserWarning, match="future_field"):
        full = loader.get_data(projection="full")
    assert full.df["future_field"].tolist() == ["0012", "None"]
    assert str(full.df["future_field"].dtype) == "object"
    assert full.df_columns_metadata.loc["future_field", "role"] == "undeclared"


# -------------------------------------------------------------------------------------- Sleep
def test_sleep_retains_state_and_optional_measurements_with_declared_dtypes(tmp_path: Path):
    rows = [
        _base_row("Sleep", offset=0, value="SLEEP", calories=None, distance=None, quantity=None),
        _base_row("Sleep", offset=1, value="REM", calories=1.0, distance=2.0, quantity=3.0),
    ]
    _write_feature(tmp_path, "Sleep", rows)
    data = _loader("Sleep", tmp_path, state_validation="off").get_data()
    assert list(data.df["value"].astype(str)) == ["SLEEP", "REM"]
    assert str(data.df["value"].dtype) == "category"
    assert all(str(data.df[name].dtype) == "float64" for name in ("calories", "distance", "quantity"))
    assert data.df_columns_metadata.loc["value", "role"] == "state"


@pytest.mark.parametrize("state", ["INBED", "UNKNOWN", ""])
def test_unregistered_or_blank_sleep_state_is_rejected(tmp_path: Path, state: str):
    _write_feature(tmp_path, "Sleep", [_base_row("Sleep", value=state or None)])
    with pytest.raises(DataLoaderReadError, match="required column|unregistered Google Fitness sleep state"):
        _loader("Sleep", tmp_path, state_validation="off").get_data()


def test_sleep_interval_must_have_positive_duration(tmp_path: Path):
    timestamp = "2024-01-01T00:00:00Z"
    _write_feature(tmp_path, "Sleep", [_base_row("Sleep", start_date=timestamp, end_date=timestamp)])
    with pytest.raises(DataLoaderReadError, match="strictly after"):
        _loader("Sleep", tmp_path, state_validation="off").get_data()


# ------------------------------------------------------------------- Calories and Distance
@pytest.mark.parametrize("feature", ["Calories", "Distance"])
@pytest.mark.parametrize("version", ["1.0", "2.0"])
def test_interval_total_versions_preserve_native_values_and_day_labels(
    tmp_path: Path, feature: str, version: str,
):
    value = -0.25 if feature == "Calories" else 123.5
    _write_feature(tmp_path, feature, [_base_row(feature, version=version, value=value)])
    data = _loader(feature, tmp_path, state_validation="off").get_data(projection="full")
    assert data.df["value"].tolist() == [value]
    assert list(data.df["source_day_label"].astype(str)) == ["יום שני"]
    assert list(data.df["collecting_method_version"].astype(str)) == [version]
    assert str(data.df["value"].dtype) == "float64"
    assert str(data.df["source_day_label"].dtype) == "category"
    assert data.df_columns_metadata.loc["value", "registry_unit"] is None


@pytest.mark.parametrize("feature,candidates", [("Calories", ["kcal"]), ("Distance", ["m"])])
def test_unresolved_google_interval_units_are_not_borrowed_from_apple(
    tmp_path: Path, feature: str, candidates: list[str],
):
    _write_feature(tmp_path, feature, [_base_row(feature)])
    result = _loader(feature, tmp_path, state_validation="off").get_data().with_harmonized_values()
    assert result.df["harmonized_value"].isna().all()
    assert result.df["harmonized_unit"].isna().all()
    assert result.df["harmonized_unit_source"].astype(str).tolist() == ["unresolved"]
    report = result.load_report["derived"]["harmonized_values"]
    assert report["registry_unit"] is None
    assert report["unit_withheld_by_curation"] is True
    assert report["unit_candidates"] == candidates


# ------------------------------------------------------------------------ Weight and Height
@pytest.mark.parametrize("feature,version", [("Weight", "1.0"), ("Weight", "2.0"), ("Height", "2.0")])
def test_anthropometric_features_use_interval_end_as_the_mechanical_date(
    tmp_path: Path, feature: str, version: str,
):
    row = _base_row(feature, version=version, value=70.5)
    _write_feature(tmp_path, feature, [row])
    data = _loader(feature, tmp_path, state_validation="off").get_data(projection="full")
    assert data.load_report["date_column"] == "end_date"
    assert data.load_report["date_semantics"] == "source_reading_interval_end"
    assert data.df.index.get_level_values("Date")[0] == pd.Timestamp(row["end_date"])
    assert data.df.iloc[0]["start_date"] < data.df.iloc[0]["end_date"]
    assert data.df_columns_metadata.loc["value", "registry_unit"] is None
    assert "added_by" in data.load_report["columns_withheld"] or "added_by" in data.df.columns


@pytest.mark.parametrize("feature,candidates", [("Weight", ["kg"]), ("Height", ["m"])])
def test_anthropometric_harmonization_remains_unresolved(tmp_path: Path, feature: str, candidates: list[str]):
    version = "2.0" if feature == "Height" else "1.0"
    _write_feature(tmp_path, feature, [_base_row(feature, version=version)])
    result = _loader(feature, tmp_path, state_validation="off").get_data().with_harmonized_values()
    assert result.df["harmonized_value"].isna().all()
    assert result.load_report["derived"]["harmonized_values"]["unit_candidates"] == candidates


# ----------------------------------------------------------------------------------- StepCount
def _step_count_rows() -> list[dict[str, Any]]:
    shared_outer = "2024-01-02T00:00:00Z"
    return [
        _base_row("StepCount", version="1.0", offset=0, value=11, outer_id="legacy"),
        _base_row(
            "StepCount", version="2.0", offset=24, outer_datetime=shared_outer, value=20, raw_value=20,
            aggregation_source="estimated_steps", payload_index=0, outer_id="modern",
        ),
        _base_row(
            "StepCount", version="2.0", offset=24, outer_datetime=shared_outer, value=18, raw_value=18,
            aggregation_source="merge_step_deltas", payload_index=1, outer_id="modern",
        ),
    ]


def test_step_count_mixed_versions_preserve_all_temporal_contexts_and_alternatives(tmp_path: Path):
    _write_feature(tmp_path, "StepCount", _step_count_rows())
    data = _loader("StepCount", tmp_path, state_validation="off").get_data(projection="full")
    assert len(data.df) == 3
    assert data.load_report["date_column"] == "outer_datetime"
    assert data.load_report["date_semantics"] == "source_outer_summary_datetime_not_canonical_event_day"
    assert isinstance(data.df.index.get_level_values("Date").dtype, pd.DatetimeTZDtype)
    assert isinstance(data.df["outer_datetime"].dtype, pd.DatetimeTZDtype)
    assert isinstance(data.df["start_date"].dtype, pd.DatetimeTZDtype)
    assert str(data.df["summary_date"].dtype) == "datetime64[ns]"
    assert data.df["summary_date"].dt.tz is None
    assert str(data.df["value"].dtype) == "Int64"
    assert str(data.df["raw_value"].dtype) == "Int64"
    assert data.df.iloc[0]["collecting_method_version"] == "1.0"
    assert pd.isna(data.df.iloc[0]["summary_date"])
    modern = data.df[data.df["collecting_method_version"].astype(str) == "2.0"]
    assert modern["aggregation_source"].astype(str).tolist() == ["estimated_steps", "merge_step_deltas"]
    assert modern.index.get_level_values("Date").nunique() == 1
    assert modern["payload_index"].tolist() == [0, 1]


def test_step_count_default_projection_withholds_persistent_identifiers_but_keeps_context(tmp_path: Path):
    _write_feature(tmp_path, "StepCount", _step_count_rows())
    loader = _loader("StepCount", tmp_path, state_validation="off")
    default = loader.get_data().df
    analysis = loader.get_data(projection="analysis").df
    full = loader.get_data(projection="full").df
    assert list(default.columns) == [
        "outer_datetime", "summary_date", "start_date", "end_date", "value", "raw_value",
        "aggregation_source", "collecting_method_version",
    ]
    assert {"origin_data_source_id", "app_package_name", "device_uid"}.isdisjoint(analysis.columns)
    assert {"app_name", "device_manufacturer", "device_model", "device_type", "outer_id", "payload_index"} <= set(
        analysis.columns
    )
    assert list(full.columns) == list(get_feature_spec("StepCount").column_names)


def test_step_count_harmonizes_to_count_without_changing_native_integer_values(tmp_path: Path):
    _write_feature(tmp_path, "StepCount", _step_count_rows())
    data = _loader("StepCount", tmp_path, state_validation="off").get_data()
    harmonized = data.with_harmonized_values()
    assert harmonized.df["harmonized_value"].tolist() == [11.0, 20.0, 18.0]
    assert harmonized.df["harmonized_unit"].astype(str).tolist() == ["count"] * 3
    assert harmonized.df["harmonized_unit_source"].astype(str).tolist() == ["registry"] * 3
    assert harmonized.df["value"].equals(data.df["value"])
    assert data.df_columns_metadata.loc["value", "registry_unit"] == "count"
    assert data.df_columns_metadata.loc["raw_value", "registry_unit"] == "count"


def test_step_count_date_filter_uses_outer_datetime_not_summary_date_or_interval(tmp_path: Path):
    rows = _step_count_rows()
    rows[1]["summary_date"] = "1999-01-01"
    rows[1]["start_date"] = "1999-01-01T00:00:00Z"
    rows[1]["end_date"] = "1999-01-01T01:00:00Z"
    _write_feature(tmp_path, "StepCount", rows)
    data = _loader("StepCount", tmp_path, state_validation="off").get_data(start_date="2024-01-02")
    assert data.df["value"].tolist() == [20, 18]


@pytest.mark.parametrize(
    "row,match",
    [
        (_base_row("StepCount", version="1.0", summary_date="2024-01-01"), "must not invent 'summary_date'"),
        (_base_row("StepCount", version="1.0", payload_index=1), "payload_index must be zero"),
        (_base_row("StepCount", version="2.0", aggregation_source=None), "requires 'aggregation_source'"),
        (_base_row("StepCount", version="2.0", value=1.5), "must contain whole numbers"),
        (_base_row("StepCount", version="2.0", payload_index=-1), "payload_index is negative"),
        (_base_row("StepCount", version="3.0"), "unsupported collecting_method_version"),
    ],
)
def test_invalid_step_count_version_contracts_are_rejected(tmp_path: Path, row: dict[str, Any], match: str):
    _write_feature(tmp_path, "StepCount", [row])
    with pytest.raises(DataLoaderReadError, match=match):
        _loader("StepCount", tmp_path, state_validation="off").get_data()


def test_step_count_calendar_date_must_be_a_calendar_label(tmp_path: Path):
    _write_feature(tmp_path, "StepCount", [_base_row("StepCount", version="2.0", summary_date="2024/01/01")])
    with pytest.raises(DataLoaderReadError, match="must use YYYY-MM-DD"):
        _loader("StepCount", tmp_path, state_validation="off").get_data()


# ------------------------------------------------------------------------ common parsing/dtypes
def test_apple_and_google_loader_contracts_do_not_leak_across_call_order(tmp_path: Path):
    apple_root = tmp_path / "apple-native"
    apple_path = apple_root / "00123" / "StepCount.csv"
    apple_path.parent.mkdir(parents=True)
    pd.DataFrame([
        {
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-01T01:00:00Z",
            "value": 10.5,
            "data_source": "AppleHealthkit",
        }
    ]).to_csv(apple_path, index=False)

    google_root = tmp_path / "google-native"
    _write_feature(google_root, "StepCount", [
        _base_row("StepCount", version="1.0", value=11),
        _base_row("StepCount", version="2.0", offset=1, value=12, raw_value=12),
    ])
    _write_state(google_root)

    def load_apple():
        return StepCountLoader(
            source="apple_healthkit", phase="native", root=apple_root, state_validation="off",
        ).get_data(projection="full")

    def load_google():
        return StepCountLoader(
            source="google_fitness", phase="native", root=google_root, state_validation="required",
        ).get_data(projection="full")

    apple_before = load_apple()
    google_middle = load_google()
    apple_after = load_apple()
    pd.testing.assert_frame_equal(apple_before.df, apple_after.df)
    pd.testing.assert_frame_equal(apple_before.df_columns_metadata, apple_after.df_columns_metadata)
    assert apple_before.load_report["date_column"] == apple_after.load_report["date_column"] == "start_date"
    assert "outer_datetime" not in apple_after.load_report["columns_available"]

    google_before = load_google()
    _ = load_apple()
    google_after = load_google()
    pd.testing.assert_frame_equal(google_before.df, google_after.df)
    pd.testing.assert_frame_equal(google_before.df_columns_metadata, google_after.df_columns_metadata)
    assert google_middle.load_report["date_column"] == google_after.load_report["date_column"] == "outer_datetime"
    assert str(google_after.df["value"].dtype) == "Int64"
    assert str(apple_after.df["value"].dtype) == "float64"


def test_single_and_multi_participant_google_dtypes_and_column_order_are_identical(tmp_path: Path):
    _write_feature(tmp_path, "Walking", [_base_row("Walking", offset=0)], participant="001")
    _write_feature(tmp_path, "Walking", [_base_row("Walking", offset=1, source_name="Other")], participant="002")
    loader = _loader("Walking", tmp_path, state_validation="off")
    one = loader.get_data(registration_codes="001")
    both = loader.get_data(registration_codes=["002", "001"])
    assert one.df.columns.tolist() == both.df.columns.tolist()
    assert one.df.dtypes.astype(str).to_dict() == both.df.dtypes.astype(str).to_dict()
    assert both.load_report["participant_ids_loaded"] == ["001", "002"]


def test_empty_date_window_keeps_the_populated_google_schema_and_dtypes(tmp_path: Path):
    _write_feature(tmp_path, "StepCount", _step_count_rows())
    loader = _loader("StepCount", tmp_path, state_validation="off")
    populated = loader.get_data()
    empty = loader.get_data(start_date="2099-01-01")
    assert empty.df.empty
    assert empty.df.columns.tolist() == populated.df.columns.tolist()
    assert empty.df.dtypes.astype(str).to_dict() == populated.df.dtypes.astype(str).to_dict()
    assert str(empty.df.index.get_level_values("Date").dtype) == str(
        populated.df.index.get_level_values("Date").dtype
    )


def test_explicit_google_columns_do_not_force_the_date_anchor_into_data_columns(tmp_path: Path):
    _write_feature(tmp_path, "Walking", [_base_row("Walking")])
    data = _loader("Walking", tmp_path, state_validation="off").get_data(columns=["quantity"])
    assert data.df.columns.tolist() == ["quantity"]
    assert data.df.index.get_level_values("Date")[0] == pd.Timestamp("2024-01-01T00:00:00Z")


@pytest.mark.parametrize(
    "feature,overrides,match",
    [
        ("Walking", {"tracked": "maybe"}, "must hold true/false flags"),
        ("Walking", {"quantity": "not-a-number"}, "must hold numbers"),
        ("Walking", {"start_date": "not-a-date"}, "failed parsing timestamp"),
        ("Walking", {"end_date": "2023-12-31T23:00:00Z"}, "strictly after"),
        ("Calories", {"collecting_method_version": "9.0"}, "unsupported collecting_method_version"),
        ("Weight", {"end_date": "2023-12-31T23:00:00Z"}, "at or after"),
    ],
)
def test_malformed_google_native_content_is_refused(
    tmp_path: Path, feature: str, overrides: dict[str, Any], match: str,
):
    version = "2.0" if feature == "Height" else None
    _write_feature(tmp_path, feature, [_base_row(feature, version=version, **overrides)])
    with pytest.raises(DataLoaderReadError, match=match):
        _loader(feature, tmp_path, state_validation="off").get_data()


def test_registered_google_column_order_is_enforced(tmp_path: Path):
    feature = "Distance"
    spec = get_feature_spec(feature)
    path = _write_feature(tmp_path, feature, [_base_row(feature)])
    frame = pd.read_csv(path, keep_default_na=False)
    order = list(spec.column_names)
    order[0], order[1] = order[1], order[0]
    frame.loc[:, order].to_csv(path, index=False)
    with pytest.raises(DataLoaderReadError, match="registered Google Fitness column order"):
        _loader(feature, tmp_path, state_validation="off").get_data()


# ------------------------------------------------------------------------- managed-state integrity
def test_managed_google_native_state_verifies_hash_size_rows_and_filename(tmp_path: Path):
    root = _managed_feature_root(tmp_path, "Walking", [_base_row("Walking")])
    data = _loader("Walking", root, state_validation="required").get_data()
    assert data.load_report["source_verified"] is True
    assert data.load_report["state_verified"] is True
    assert data.load_report["files_verified"] == 1
    assert data.load_report["size_estimate"]["source"] == "native_state"
    assert data.load_report["source_contract"] == {
        "dataloader_schema_version": "google-fitness-dataloader-native-schema-1",
        "native_registry_version": REGISTRY_VERSION,
        "native_state_schema_version": STATE_SCHEMA_VERSION,
        "feature_family": "activity_interval",
        "supported_collecting_method_versions": ["1.0"],
    }
    assert data.load_report["collecting_method_version_counts"] == {"1.0": 1}
    assert data.load_report["tracking_counts"] == {"false": 0, "missing": 0, "true": 1}
    assert data.df_metadata["state_verified"].tolist() == [True]


def test_google_same_size_edit_is_caught_by_sha256(tmp_path: Path):
    root = _managed_feature_root(tmp_path, "StepCount", [_base_row("StepCount", version="1.0", value=10)])
    path = root / "00123" / "StepCount.csv"
    original = path.read_text()
    edited = original.replace(",10,", ",11,", 1)
    assert edited != original and len(edited) == len(original)
    path.write_text(edited)
    with pytest.raises(DataLoaderStateError, match="sha256"):
        _loader("StepCount", root, state_validation="required").get_data()


@pytest.mark.parametrize(
    "override,match",
    [
        ({"row_count": 99}, "rows: file"),
        ({"size_bytes": 1}, "bytes"),
        ({"filename": "Other.csv"}, "state records filename"),
    ],
)
def test_google_manifest_mismatch_is_rejected(tmp_path: Path, override: dict[str, Any], match: str,):
    root = tmp_path / "root"
    _write_feature(root, "Walking", [_base_row("Walking")])
    _write_state(root, output_overrides={("00123", "Walking"): override})
    with pytest.raises(DataLoaderStateError, match=match):
        _loader("Walking", root, state_validation="required").get_data()


@pytest.mark.parametrize(
    "metadata,match",
    [
        ({"source_key": "unknown_source"}, "could not establish the requested source contract"),
        ({"state_schema_version": "old-state"}, "incompatible Google Fitness native state schema"),
        ({"registry_version": "old-registry"}, "incompatible Google Fitness native registry"),
    ],
)
def test_incompatible_google_state_or_registry_version_is_rejected_before_csv_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata: dict[str, str], match: str,
):
    root = tmp_path / "root"
    _write_feature(root, "Walking", [_base_row("Walking")])
    _write_state(root, metadata_overrides=metadata)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("CSV parsing happened before managed-root compatibility validation")

    monkeypatch.setattr(AppleHealthFeatureLoader, "_read_feature_file", forbidden)
    with pytest.raises(DataLoaderStateError, match=match):
        _loader("Walking", root, state_validation="off").get_data()


def test_google_state_required_and_off_modes(tmp_path: Path):
    root = tmp_path / "root"
    path = _write_feature(root, "Walking", [_base_row("Walking")])
    with pytest.raises(DataLoaderStateError, match="required"):
        _loader("Walking", root, state_validation="required").get_data()
    _write_state(root)
    path.write_text(path.read_text().replace("1.25", "9.25", 1))
    off = _loader("Walking", root, state_validation="off").get_data()
    assert off.load_report["source_verified"] is True
    assert off.load_report["state_verified"] is False
    assert off.df["calories"].tolist() == [9.25]


def test_google_stray_file_without_feature_outputs_row_is_rejected(tmp_path: Path):
    root = tmp_path / "root"
    _write_feature(root, "Walking", [_base_row("Walking")])
    _write_state(root, omit_outputs={("00123", "Walking")})
    with pytest.raises(DataLoaderStateError, match="no feature_outputs row"):
        _loader("Walking", root, state_validation="required").get_data()


def test_google_missing_manifest_file_is_reported_without_parsing(tmp_path: Path):
    root = _managed_feature_root(tmp_path, "Walking", [_base_row("Walking")])
    (root / "00123" / "Walking.csv").unlink()
    with pytest.warns(UserWarning, match="no longer on disk"):
        data = _loader("Walking", root, state_validation="required").get_data()
    assert data.df.empty
    assert data.load_report["manifest_files_missing"] == ["00123"]


def test_google_max_rows_preflight_uses_state_before_csv_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,):
    root = _managed_feature_root(tmp_path, "Walking", [_base_row("Walking", offset=i) for i in range(2)])

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("CSV was parsed before max_rows preflight")

    monkeypatch.setattr(AppleHealthFeatureLoader, "_read_feature_file", forbidden)
    with pytest.raises(DataLoaderSizeError, match="above max_rows=1"):
        _loader("Walking", root, state_validation="required").get_data(max_rows=1)


def test_google_profile_uses_state_and_stat_without_parsing_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,):
    root = _managed_feature_root(tmp_path, "Walking", [_base_row("Walking", offset=i) for i in range(2)])

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("profile parsed a CSV")

    monkeypatch.setattr(AppleHealthFeatureLoader, "_read_feature_file", forbidden)
    profile = _loader("Walking", root, state_validation="required").profile()
    assert profile.summary["rows"] == 2
    assert profile.summary["source"] == "native_state"
    assert profile.summary["wearable_source"] == "google_fitness"
    assert profile.summary["source_verified"] is True
    assert profile.summary["source_contract"]["native_registry_version"] == REGISTRY_VERSION
    assert profile.summary["source_contract"]["native_state_schema_version"] == STATE_SCHEMA_VERSION
    assert profile.summary["integrity"] == {
        "files_missing_on_disk": 0,
        "files_without_state": 0,
        "files_size_mismatch": 0,
    }


def test_google_complete_empty_unknown_and_feature_absent_participants_are_distinct(tmp_path: Path):
    root = tmp_path / "root"
    _write_feature(root, "Walking", [_base_row("Walking")], participant="with-data")
    (root / "empty-source-with-dir").mkdir(parents=True)
    (root / "without-feature").mkdir(parents=True)
    _write_state(
        root,
        statuses={
            "empty-source": "complete_empty",
            "empty-source-with-dir": "complete_empty",
            "without-feature": "complete",
        },
    )
    loader = _loader("Walking", root, state_validation="required")
    data = loader.get_data(
        registration_codes=[
            "with-data", "empty-source", "empty-source-with-dir", "without-feature", "unknown",
        ]
    )
    report = data.load_report
    assert report["participant_ids_loaded"] == ["with-data"]
    assert report["requested_participants_empty_for_source"] == [
        "empty-source", "empty-source-with-dir",
    ]
    assert report["requested_participants_without_file"] == ["without-feature"]
    assert report["requested_participants_missing_from_root"] == ["unknown"]


# ----------------------------------------------------------------------------- helper behavior
def test_google_local_time_refuses_to_invent_offsets(tmp_path: Path):
    _write_feature(tmp_path, "Walking", [_base_row("Walking")])
    data = _loader("Walking", tmp_path, state_validation="off").get_data()
    with pytest.raises(DataLoaderConfigurationError, match="stores no utc_offset_minutes"):
        data.with_local_time()


def test_multi_measurement_activity_cannot_use_singular_harmonization_helper(tmp_path: Path):
    _write_feature(tmp_path, "Walking", [_base_row("Walking")])
    data = _loader("Walking", tmp_path, state_validation="off").get_data()
    with pytest.raises(DataLoaderConfigurationError, match="different measurement contract"):
        data.with_harmonized_values()


def test_google_sleep_state_is_not_treated_as_a_numeric_harmonization_value(tmp_path: Path):
    _write_feature(tmp_path, "Sleep", [_base_row("Sleep")])
    data = _loader("Sleep", tmp_path, state_validation="off").get_data()
    with pytest.raises(DataLoaderConfigurationError, match="numeric values"):
        data.with_harmonized_values()


# ----------------------------------------------------------------------------- real sample acceptance
def _sample_outputs(root: Path) -> list[tuple[str, str, int]]:
    connection = sqlite3.connect(f"file:{root / '.wearable_state.sqlite'}?immutable=1", uri=True)
    rows = connection.execute(
        "SELECT participant_id, feature, row_count FROM feature_outputs ORDER BY feature, participant_id"
    ).fetchall()
    connection.close()
    return [(str(participant), str(feature), int(rows)) for participant, feature, rows in rows]


@needs_google_sample
def test_real_google_native_sample_every_observed_feature_verifies_and_profiles():
    outputs = _sample_outputs(GOOGLE_NATIVE_SAMPLE)
    expected_by_feature = Counter()
    for _, feature, rows in outputs:
        expected_by_feature[feature] += rows
    assert len(expected_by_feature) == 16
    assert sum(expected_by_feature.values()) == 145_613
    loaded = 0
    for feature, expected_rows in sorted(expected_by_feature.items()):
        loader = _loader(feature, GOOGLE_NATIVE_SAMPLE, state_validation="required")
        data = loader.get_data(projection="full", max_rows=None)
        profile = loader.profile()
        assert len(data.df) == expected_rows, feature
        assert profile.summary["rows"] == expected_rows, feature
        assert data.load_report["state_verified"] is True, feature
        assert data.load_report["files_verified"] == data.load_report["files_read"], feature
        assert data.load_report["columns_undeclared"] == [], feature
        assert list(data.df.columns) == list(get_feature_spec(feature).column_names), feature
        loaded += len(data.df)
    assert loaded == 145_613


@needs_google_sample
def test_real_google_step_count_versions_and_alternatives_are_all_retained():
    data = _loader("StepCount", GOOGLE_NATIVE_SAMPLE, state_validation="required").get_data(
        projection="full", max_rows=None,
    )
    versions = data.df["collecting_method_version"].astype(str).value_counts().to_dict()
    assert versions == {"2.0": 3_665, "1.0": 3_128}
    v2 = data.df[data.df["collecting_method_version"].astype(str) == "2.0"]
    assert v2["outer_id"].nunique() == 1_869
    group_sizes = v2.groupby("outer_id", observed=True).size().value_counts().to_dict()
    assert group_sizes == {2: 1_796, 1: 73}
    assert len(data.df) == 6_793


@needs_google_sample
def test_real_google_single_and_batch_dtypes_are_stable_for_every_observed_feature():
    outputs = _sample_outputs(GOOGLE_NATIVE_SAMPLE)
    by_feature: dict[str, list[str]] = {}
    for participant, feature, _ in outputs:
        by_feature.setdefault(feature, []).append(participant)
    for feature, participants in sorted(by_feature.items()):
        loader = _loader(feature, GOOGLE_NATIVE_SAMPLE, state_validation="required")
        one = loader.get_data(registration_codes=participants[0], max_rows=None)
        batch = loader.get_data(registration_codes=list(reversed(participants)), max_rows=None)
        assert one.df.columns.tolist() == batch.df.columns.tolist(), feature
        assert one.df.dtypes.astype(str).to_dict() == batch.df.dtypes.astype(str).to_dict(), feature
        assert str(one.df.index.get_level_values("Date").dtype) == str(
            batch.df.index.get_level_values("Date").dtype
        ), feature


@needs_google_sample
def test_real_google_full_projection_preserves_every_stored_cell_semantically():
    outputs = _sample_outputs(GOOGLE_NATIVE_SAMPLE)
    participants_by_feature: dict[str, list[str]] = {}
    for participant, feature, _ in outputs:
        participants_by_feature.setdefault(feature, []).append(participant)

    for feature, participants in sorted(participants_by_feature.items()):
        spec = get_feature_spec(feature)
        data = _loader(feature, GOOGLE_NATIVE_SAMPLE, state_validation="required").get_data(
            registration_codes=sorted(participants), projection="full", sort_index=False, max_rows=None,
        )
        raw = pd.concat(
            [
                pd.read_csv(
                    GOOGLE_NATIVE_SAMPLE / participant / f"{feature}.csv",
                    keep_default_na=False,
                    na_values=[""],
                    dtype={column.name: str for column in spec.columns if column.data_type in {ColumnType.STRING, ColumnType.DATE}},
                    float_precision="round_trip",
                )
                for participant in sorted(participants)
            ],
            ignore_index=True,
        )
        assert list(data.df.columns) == list(raw.columns), feature
        assert len(data.df) == len(raw), feature
        for column_spec in spec.columns:
            name = column_spec.name
            actual = data.df[name].reset_index(drop=True)
            expected = raw[name]
            if column_spec.data_type is ColumnType.TIMESTAMP_UTC:
                expected = pd.to_datetime(expected, utc=True, errors="raise", format="mixed")
                assert actual.equals(expected), (feature, name)
            elif column_spec.data_type is ColumnType.DATE:
                expected = pd.to_datetime(expected, errors="raise", format="%Y-%m-%d")
                assert actual.equals(expected), (feature, name)
            elif column_spec.data_type in {ColumnType.NUMBER, ColumnType.INTEGER}:
                expected_numeric = pd.to_numeric(expected, errors="raise")
                assert np.array_equal(
                    actual.astype("Float64").to_numpy(dtype=float, na_value=np.nan),
                    expected_numeric.astype("Float64").to_numpy(dtype=float, na_value=np.nan),
                    equal_nan=True,
                ), (feature, name)
            elif column_spec.data_type is ColumnType.BOOLEAN:
                expected_bool = expected.map({True: True, False: False, "True": True, "False": False, 1: True, 0: False})
                assert actual.astype("boolean").reset_index(drop=True).equals(
                    expected_bool.astype("boolean").reset_index(drop=True)
                ), (feature, name)
            else:
                left = actual.astype("string").fillna("<NA>")
                right = expected.astype("string").fillna("<NA>")
                assert left.equals(right), (feature, name)


@needs_google_sample
def test_real_google_loader_is_read_only():
    tracked = [GOOGLE_NATIVE_SAMPLE / ".wearable_state.sqlite", *sorted(GOOGLE_NATIVE_SAMPLE.glob("*/*.csv"))]
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns, _sha256(path)) for path in tracked}
    _loader("Walking", GOOGLE_NATIVE_SAMPLE, state_validation="required").get_data(max_rows=None)
    _loader("StepCount", GOOGLE_NATIVE_SAMPLE, state_validation="required").profile()
    after = {path: (path.stat().st_size, path.stat().st_mtime_ns, _sha256(path)) for path in tracked}
    assert after == before
    assert not list(GOOGLE_NATIVE_SAMPLE.glob(".wearable_state.sqlite-*"))
