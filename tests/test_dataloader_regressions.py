"""
Wearable Data Processing and Modeling project
Each test pins one defect so it cannot return silently: reader fidelity (floats, NA tokens), typed empty results,
explicit sparse columns, a schema fixed by the files read, participant-code and date-bound validation, argument
normalization, hidden folders, and damaged state.
"""


from __future__ import annotations
import csv
import datetime as dt
import os
import shutil
import sqlite3
import sys
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pytest
from wearable_project.DataLoaders import DataLoaderStateError
from wearable_project.DataLoaders._base import AppleHealthFeatureLoader
from wearable_project.DataLoaders.ActivitySummaryLoader import ActivitySummaryLoader
from wearable_project.DataLoaders.BloodGlucoseLoader import BloodGlucoseLoader
from wearable_project.DataLoaders.BMILoader import BMILoader
from wearable_project.DataLoaders.HeartRateLoader import HeartRateLoader
from wearable_project.DataLoaders.StepCountLoader import StepCountLoader
from wearable_project.DataLoaders.WeightLoader import WeightLoader
from wearable_project.exceptions import DataLoaderConfigurationError


CURATED_SAMPLE = Path(os.environ.get(
    "WEARABLE_CURATED_SAMPLE",
    "/home/evortigosa/Desktop/postdoc/code/PostdocProject/cluster_data/EV_curated_apple_healthkit"
))
NATIVE_SAMPLE = Path(os.environ.get(
    "WEARABLE_NATIVE_SAMPLE",
    "/home/evortigosa/Desktop/postdoc/code/PostdocProject/cluster_data/EV_cleaned_apple_healthkit"
))
needs_sample = pytest.mark.skipif(
    not CURATED_SAMPLE.is_dir() or not NATIVE_SAMPLE.is_dir(),
    reason="curated and native sample roots are required for end-to-end regression tests",
)
ROOTS = {"curated": CURATED_SAMPLE, "native": NATIVE_SAMPLE}
P1, P2 = "1420269268", "1235738253"
csv.field_size_limit(sys.maxsize)


def _raw_column(path: Path, column: str) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        index = header.index(column)
        return [row[index] for row in reader]


def _write_steps(root: Path, participant: str, rows: list[dict]) -> None:
    folder = root / participant
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "StepCount.csv", index=False)


def _row(**extra) -> dict:
    row = {"start_date": "2024-01-01T00:00:00Z", "end_date": "2024-01-01T01:00:00Z", "value": 1.0,
           "datetime": "2024-01-01T00:00:00Z"}
    row.update(extra)
    return row


# ============================================================================ reader fidelity
@needs_sample
@pytest.mark.parametrize("phase", ["curated", "native"])
def test_dexcom_none_trend_arrow_is_kept_as_text(phase):
    """pandas' default NA tokens turned 104 legitimate "None" trend arrows into missing values."""
    arrows = BloodGlucoseLoader(phase=phase, root=ROOTS[phase]).get_data().df["trend_arrow"]
    assert int((arrows.astype(object) == "None").sum()) == 104
    assert "None" in arrows.cat.categories


@needs_sample
@pytest.mark.parametrize("phase", ["curated", "native"])
def test_floats_come_back_exactly_as_stored(phase):
    """pandas' default float parser mis-rounds about 3% of stored values by one unit in the last place."""
    for participant, feature in ((P2, "HeartRate"), ("8776545505", "BloodGlucose"), (P1, "StepCount")):
        loader = __import__(f"wearable_project.DataLoaders.{feature}Loader", fromlist=[f"{feature}Loader"])
        cls = getattr(loader, f"{feature}Loader")
        loaded = cls(phase=phase, root=ROOTS[phase]).get_data(
            registration_codes=participant, columns=["value"], sort_index=False
        ).df["value"].tolist()
        stored = [float(text) for text in _raw_column(ROOTS[phase] / participant / f"{feature}.csv", "value")]
        assert loaded == stored, feature


def test_a_value_the_default_parser_mis_rounds_is_read_exactly(tmp_path: Path):
    text = "0.9899999999999999"
    _write_steps(tmp_path, "123", [_row(value=float(text))])
    assert _raw_column(tmp_path / "123" / "StepCount.csv", "value") == [text]
    value = StepCountLoader(phase="native", root=tmp_path).get_data().df["value"].iloc[0]
    assert value == float(text)


def test_only_empty_cells_are_missing(tmp_path: Path):
    folder = tmp_path / "123"
    folder.mkdir()
    (folder / "StepCount.csv").write_text(
        "start_date,end_date,value,datetime,device\n"
        + "".join(f"2024-01-01T0{i}:00:00Z,2024-01-01T0{i}:30:00Z,1.0,2024-01-01T00:00:00Z,{token}\n"
                  for i, token in enumerate(["None", "NA", "null", "n/a", ""]))
    )
    device = StepCountLoader(phase="native", root=tmp_path).get_data(
        columns=["device"], sort_index=False
    ).df["device"].astype(object)
    assert device.iloc[:4].tolist() == ["None", "NA", "null", "n/a"]
    assert pd.isna(device.iloc[4])


# ======================================================================== typed empty results
@needs_sample
@pytest.mark.parametrize("phase", ["curated", "native"])
@pytest.mark.parametrize("loader", [HeartRateLoader, WeightLoader, ActivitySummaryLoader, StepCountLoader])
def test_empty_result_has_the_dtypes_of_a_populated_one(phase, loader):
    populated = loader(phase=phase, root=ROOTS[phase]).get_data().df
    empty = loader(phase=phase, root=ROOTS[phase]).get_data(start_date="2099-01-01").df
    assert list(empty.columns) == list(populated.columns)
    assert empty.dtypes.astype(str).to_dict() == populated.dtypes.astype(str).to_dict()
    for level in ("RegistrationCode", "Date"):
        assert str(empty.index.get_level_values(level).dtype) == str(populated.index.get_level_values(level).dtype)


@needs_sample
def test_concatenating_an_empty_result_keeps_non_categorical_dtypes():
    populated = HeartRateLoader(root=CURATED_SAMPLE).get_data().df
    empty = HeartRateLoader(root=CURATED_SAMPLE).get_data(start_date="2099-01-01").df
    # pandas 2 announces that concatenation will stop ignoring empty frames when choosing dtypes; pandas 3 made
    # that change. This is the caller's concat, and the assertion below must hold under both behaviors.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation", category=FutureWarning)
        combined = pd.concat([empty, populated])
    for column in populated.columns:
        if not isinstance(populated[column].dtype, pd.CategoricalDtype):
            assert str(combined[column].dtype) == str(populated[column].dtype), column


# =================================================================== explicit sparse columns
@needs_sample
@pytest.mark.parametrize("loader, column", [(WeightLoader, "canonical_value"),
                                            (HeartRateLoader, "heart_rate_motion_context")])
def test_named_sparse_numeric_column_keeps_its_numeric_dtype(loader, column):
    """Filling absent columns per file with pd.NA made them object under pandas 3."""
    projected = loader(root=CURATED_SAMPLE).get_data(projection="full").df[column]
    named = loader(root=CURATED_SAMPLE).get_data(columns=[column]).df[column]
    from wearable_project.curation import schema as _schema
    # Whole-number columns are nullable integers; every other numeric column is float64.
    assert str(named.dtype) == str(projected.dtype) == ("Int64" if column in _schema.INTEGER_COLUMNS else "float64")
    assert named.tolist() == projected.tolist() or named.isna().equals(projected.isna())


# ======================================================================== schema from files read
@needs_sample
def test_a_date_window_does_not_change_the_columns():
    """Columns must not depend on which participants happen to have rows inside the window."""
    everyone = BMILoader(root=CURATED_SAMPLE).get_data()
    dates = everyone.df.index.get_level_values("Date")
    window = BMILoader(root=CURATED_SAMPLE).get_data(start_date=dates.max(), end_date=dates.max())
    assert window.df.index.get_level_values(0).nunique() < everyone.df.index.get_level_values(0).nunique()
    assert list(window.df.columns) == list(everyone.df.columns)


@needs_sample
def test_the_inclusion_filter_does_not_change_the_columns():
    everyone = WeightLoader(root=CURATED_SAMPLE).get_data()
    included = WeightLoader(root=CURATED_SAMPLE).get_data(default_inclusion_only=True)
    assert list(included.df.columns) == list(everyone.df.columns)


@needs_sample
def test_a_participant_subset_only_omits_columns_it_has_no_data_for():
    everyone = WeightLoader(root=CURATED_SAMPLE).get_data(projection="full").df
    subset = WeightLoader(root=CURATED_SAMPLE).get_data(projection="full", registration_codes=P1).df
    restricted = everyone[everyone.index.get_level_values(0) == f"10K_{P1}"]
    assert set(subset.columns) <= set(everyone.columns)
    for column in set(everyone.columns) - set(subset.columns):
        assert restricted[column].isna().all(), column


# ================================================================ participant-code validation
@needs_sample
@pytest.mark.parametrize("code", [np.int64(P1), float(P1), np.float64(P1), f"10k_{P1}", f"  10K_{P1} ", int(P1)])
def test_accepted_participant_code_forms(code):
    assert len(StepCountLoader(root=CURATED_SAMPLE).get_data(registration_codes=code).df) == 19_328


@pytest.mark.parametrize("code", [None, float("nan"), np.nan, pd.NA, 1.5, True, b"123", [["123"]], object()])
def test_malformed_participant_codes_are_rejected(tmp_path: Path, code):
    (tmp_path / "123").mkdir()
    codes = code if isinstance(code, list) else ["123", code]
    with pytest.raises(DataLoaderConfigurationError):
        StepCountLoader(root=tmp_path).get_data(registration_codes=codes)


@pytest.mark.parametrize("code", ["10K_123", "10k_123", "123", 123, np.int64(123), 123.0, " 123 "])
def test_registration_code_mapping_is_consistent(code):
    assert AppleHealthFeatureLoader._to_registration_code(code) == "10K_123"
    assert AppleHealthFeatureLoader._to_participant_id(code) == "123"


# ======================================================================= date-bound validation
@pytest.mark.parametrize("bound", ["", float("nan"), np.nan, pd.NaT, np.datetime64("NaT"), 20240101, 2024.0,
                                   np.int64(20240101), True])
def test_ambiguous_or_missing_bounds_are_rejected(tmp_path: Path, bound):
    """An integer bound was read as nanoseconds since 1970 and silently returned everything."""
    (tmp_path / "123").mkdir()
    with pytest.raises(DataLoaderConfigurationError, match="start_date"):
        StepCountLoader(root=tmp_path).get_data(start_date=bound)


@needs_sample
@pytest.mark.parametrize("bound", [
    dt.date(2020, 1, 1), dt.datetime(2020, 1, 1, tzinfo=ZoneInfo("Asia/Jerusalem")),
    np.datetime64("2020-01-01"), pd.Timestamp("2020-01-01", tz="UTC"), "2020-01-01T00:00:00+03:00",
])
def test_real_timestamp_bounds_are_accepted(bound):
    StepCountLoader(root=CURATED_SAMPLE).get_data(start_date=bound)


# ================================================================== argument normalisation
@needs_sample
def test_a_bare_column_name_is_one_column():
    assert list(StepCountLoader(root=CURATED_SAMPLE).get_data(columns="value").df.columns) == ["value"]


@needs_sample
def test_projection_names_are_normalised_like_phase():
    assert StepCountLoader(root=CURATED_SAMPLE).get_data(projection=" Full ").load_report["projection"] == "full"


# ========================================================================= discovery and state
@needs_sample
def test_hidden_folders_are_not_participants(tmp_path: Path):
    root = tmp_path / "native"
    shutil.copytree(NATIVE_SAMPLE, root)
    shutil.copytree(root / P1, root / ".scratch_copy")
    data = StepCountLoader(phase="native", root=root).get_data()
    assert not any(".scratch" in code for code in data.df_metadata.index)
    assert StepCountLoader(phase="native", root=root).profile().summary["participants"] == 10


@needs_sample
@pytest.mark.parametrize("damage", ["zero_bytes", "garbage", "no_outputs_table"])
def test_damaged_state_database_is_refused(tmp_path: Path, damage):
    root = tmp_path / "curated"
    for path in CURATED_SAMPLE.glob("*/StepCount.csv"):
        (root / path.parent.name).mkdir(parents=True)
        shutil.copy2(path, root / path.parent.name / path.name)
    database = root / ".wearable_curation_state.sqlite"
    shutil.copy2(CURATED_SAMPLE / database.name, database)
    if damage == "zero_bytes":
        database.write_bytes(b"")
    elif damage == "garbage":
        database.write_bytes(os.urandom(4096))
    else:
        connection = sqlite3.connect(database)
        connection.execute("drop table curation_outputs")
        connection.commit()
        connection.execute("pragma wal_checkpoint(TRUNCATE)")
        connection.close()
    for suffix in ("-wal", "-shm"):
        (root / (database.name + suffix)).unlink(missing_ok=True)
    with pytest.raises(DataLoaderStateError):
        StepCountLoader(root=root).get_data()


# ====================================================== datetime resolution of empty results
@needs_sample
def test_a_result_with_no_files_read_has_the_populated_date_dtype():
    """Under pandas 3 an empty DatetimeIndex defaults to seconds, while parsed timestamps are microseconds."""
    populated = StepCountLoader(root=CURATED_SAMPLE).get_data().df
    nothing = StepCountLoader(root=CURATED_SAMPLE).get_data(registration_codes="10K_0000000000").df
    assert nothing.empty
    assert str(nothing.index.get_level_values("Date").dtype) == str(populated.index.get_level_values("Date").dtype)


@needs_sample
@pytest.mark.parametrize("kwargs", [{"start_date": "2099-01-01"}, {"registration_codes": "10K_0000000000"}])
def test_empty_df_metadata_dates_have_the_populated_dtype(kwargs):
    populated = StepCountLoader(root=CURATED_SAMPLE).get_data().df_metadata
    empty = StepCountLoader(root=CURATED_SAMPLE).get_data(**kwargs).df_metadata
    for column in ("first_date", "last_date"):
        assert str(empty[column].dtype) == str(populated[column].dtype)
