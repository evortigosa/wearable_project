"""
Wearable Data Processing and Modeling project
Tests for native verification, the HPP ``reg_ids``/``cols`` aliases, local wall-clock time, and harmonized values.
"""


from __future__ import annotations
import glob
import os
import shutil
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from wearable_project.DataLoaders import DataLoaderStateError, LoaderData
from wearable_project.DataLoaders._base import AppleHealthFeatureLoader
from wearable_project.DataLoaders.ActivitySummaryLoader import ActivitySummaryLoader
from wearable_project.DataLoaders.EnergyConsumedLoader import EnergyConsumedLoader
from wearable_project.DataLoaders.HeartRateLoader import HeartRateLoader
from wearable_project.DataLoaders.OxygenSaturationLoader import OxygenSaturationLoader
from wearable_project.DataLoaders.SleepLoader import SleepLoader
from wearable_project.DataLoaders.StepCountLoader import StepCountLoader
from wearable_project.DataLoaders.WeightLoader import WeightLoader
from wearable_project.exceptions import DataLoaderConfigurationError
from wearable_project.processing.registry import get_feature_spec


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
    reason="curated and native sample roots are required for these tests",
)
ROOTS = {"curated": CURATED_SAMPLE, "native": NATIVE_SAMPLE}
NATIVE_STATE = ".wearable_state.sqlite"
P1 = "1420269268"
NOT_APPLICABLE = {"ActivitySummary", "BloodPressure", "Electrocardiogram", "Mindful", "Sleep"}


def _loader(feature: str):
    module = __import__(f"wearable_project.DataLoaders.{feature}Loader", fromlist=[f"{feature}Loader"])
    return getattr(module, f"{feature}Loader")


def _features() -> list[str]:
    return sorted({Path(p).stem for p in glob.glob(str(CURATED_SAMPLE / "*" / "*.csv"))})


def _native_copy(tmp_path: Path, feature: str = "StepCount", state: bool = True) -> Path:
    root = tmp_path / "native"
    for path in NATIVE_SAMPLE.glob(f"*/{feature}.csv"):
        (root / path.parent.name).mkdir(parents=True)
        shutil.copy2(path, root / path.parent.name / path.name)
    if state:
        shutil.copy2(NATIVE_SAMPLE / NATIVE_STATE, root / NATIVE_STATE)
    return root


@pytest.fixture(autouse=True)
def _no_row_limit():
    saved = AppleHealthFeatureLoader.default_max_rows
    AppleHealthFeatureLoader.default_max_rows = None
    yield
    AppleHealthFeatureLoader.default_max_rows = saved


# =========================================================================== native verification
@needs_sample
def test_every_native_feature_verifies_against_the_processing_state():
    for feature in _features():
        report = _loader(feature)(phase="native", root=NATIVE_SAMPLE).get_data().load_report
        assert report["state_verified"] is True, feature
        assert report["files_verified"] == report["files_read"], feature


@needs_sample
def test_native_same_length_edit_is_caught_by_the_hash(tmp_path: Path):
    root = _native_copy(tmp_path)
    path = root / P1 / "StepCount.csv"
    original = path.read_text()
    edited = original.replace(",39.0,", ",40.0,", 1)
    assert edited != original and len(edited) == len(original)
    path.write_text(edited)
    with pytest.raises(DataLoaderStateError, match="sha256"):
        StepCountLoader(phase="native", root=root).get_data()


@needs_sample
def test_native_truncated_file_is_refused(tmp_path: Path):
    root = _native_copy(tmp_path)
    path = root / P1 / "StepCount.csv"
    path.write_text("\n".join(path.read_text().splitlines()[:-20]) + "\n")
    with pytest.raises(DataLoaderStateError, match="rows: file"):
        StepCountLoader(phase="native", root=root).get_data()


@needs_sample
def test_native_stray_folder_is_refused(tmp_path: Path):
    """A copied participant folder has no processing record, so it can no longer load as a participant."""
    root = _native_copy(tmp_path)
    shutil.copytree(root / P1, root / "backup")
    with pytest.raises(DataLoaderStateError, match="no feature_outputs row"):
        StepCountLoader(phase="native", root=root).get_data()


@needs_sample
def test_native_missing_file_is_reported_not_refused(tmp_path: Path):
    root = _native_copy(tmp_path)
    (root / P1 / "StepCount.csv").unlink()
    with pytest.warns(UserWarning, match="no longer on disk"):
        data = StepCountLoader(phase="native", root=root).get_data()
    assert data.load_report["manifest_files_missing"] == [P1]
    assert data.load_report["state_verified"] is True


@needs_sample
def test_native_validation_modes(tmp_path: Path):
    unmanaged = _native_copy(tmp_path / "a", state=False)
    assert StepCountLoader(phase="native", root=unmanaged).get_data().load_report["state_verified"] is False
    with pytest.raises(DataLoaderStateError, match="required"):
        StepCountLoader(phase="native", root=unmanaged, state_validation="required").get_data()
    managed = _native_copy(tmp_path / "b")
    assert StepCountLoader(phase="native", root=managed, state_validation="required").get_data().load_report[
        "state_verified"] is True
    path = managed / P1 / "StepCount.csv"
    path.write_text(path.read_text().replace(",39.0,", ",40.0,", 1))
    off = StepCountLoader(phase="native", root=managed, state_validation="off").get_data()
    assert off.load_report["state_verified"] is False
    assert off.load_report["size_estimate"] is None


@needs_sample
def test_native_size_estimate_comes_from_the_verified_manifest():
    estimate = StepCountLoader(phase="native", root=NATIVE_SAMPLE).get_data().load_report["size_estimate"]
    assert estimate["source"] == "native_state"
    assert estimate["rows_expected"] == 52_247


@needs_sample
def test_native_profile_honours_state_validation_off():
    assert StepCountLoader(phase="native", root=NATIVE_SAMPLE, state_validation="off").profile().summary[
        "source"] == "filesystem"


# ==================================================================================== aliases
@needs_sample
def test_reg_ids_and_cols_are_exact_aliases():
    long = StepCountLoader(root=CURATED_SAMPLE).get_data(registration_codes=[f"10K_{P1}"], columns=["value"])
    short = StepCountLoader(root=CURATED_SAMPLE).get_data(reg_ids=[f"10K_{P1}"], cols=["value"])
    assert long.df.equals(short.df)
    assert StepCountLoader(root=CURATED_SAMPLE).get_data(cols="value").df.columns.tolist() == ["value"]


@needs_sample
def test_profile_accepts_reg_ids():
    assert StepCountLoader(root=CURATED_SAMPLE).profile(reg_ids=P1).summary["rows"] == 19_328


@pytest.mark.parametrize("kwargs", [
    {"registration_codes": "123", "reg_ids": "123"},
    {"columns": ["value"], "cols": ["value"]},
    {"registration_codes": [], "reg_ids": ["123"]},
])
def test_an_alias_and_its_long_form_together_are_rejected(tmp_path: Path, kwargs):
    (tmp_path / "123").mkdir()
    with pytest.raises(DataLoaderConfigurationError, match="not both"):
        StepCountLoader(root=tmp_path).get_data(**kwargs)


def test_profile_rejects_both_forms(tmp_path: Path):
    (tmp_path / "123").mkdir()
    with pytest.raises(DataLoaderConfigurationError, match="not both"):
        StepCountLoader(root=tmp_path).profile(registration_codes="123", reg_ids="123")


# ================================================================================= local time
@needs_sample
@pytest.mark.parametrize("phase", ["curated", "native"])
def test_local_time_is_utc_plus_the_row_offset(phase):
    data = HeartRateLoader(phase=phase, root=ROOTS[phase]).get_data()
    local = data.with_local_time().df
    shift = pd.to_timedelta(local["utc_offset_minutes"], unit="min")
    for column in ("start_date", "end_date"):
        expected = (local[column] + shift).dt.tz_localize(None)
        assert local[f"{column}_local"].equals(expected)
        assert local[f"{column}_local"].dt.tz is None
        assert local[column].equals(data.df[column])


@needs_sample
def test_local_time_follows_daylight_saving_offsets():
    df = SleepLoader(root=CURATED_SAMPLE).get_data().with_local_time().df
    for offset in (120, 180):
        row = df[df["utc_offset_minutes"] == offset].iloc[0]
        assert row["start_date_local"] - row["start_date"].tz_localize(None) == pd.Timedelta(minutes=offset)


@needs_sample
def test_local_time_puts_bedtime_near_local_midnight():
    df = SleepLoader(root=CURATED_SAMPLE).get_data().with_local_time().df
    inbed = df[df["value"].astype(str) == "INBED"]
    assert inbed["start_date_local"].dt.hour.mode().iloc[0] in {23, 0}
    assert inbed["start_date"].dt.hour.mode().iloc[0] in {20, 21, 22}


@needs_sample
def test_local_time_leaves_the_original_result_unchanged():
    data = HeartRateLoader(root=CURATED_SAMPLE).get_data()
    before = data.df.copy()
    local = data.with_local_time()
    assert isinstance(local, LoaderData) and local is not data
    assert data.df.equals(before)
    assert "start_date_local" not in data.df.columns


@needs_sample
def test_local_time_metadata_and_report():
    local = HeartRateLoader(root=CURATED_SAMPLE).get_data().with_local_time()
    assert list(local.df_columns_metadata.index) == list(local.df.columns)
    assert local.df_columns_metadata.loc["start_date_local", "role"] == "derived"
    assert local.load_report["derived"]["local_time"] == {"columns": ["start_date_local", "end_date_local"]}


@needs_sample
def test_local_time_refusals():
    with pytest.raises(DataLoaderConfigurationError, match="stores no utc_offset_minutes"):
        ActivitySummaryLoader(root=CURATED_SAMPLE).get_data().with_local_time()
    with pytest.raises(DataLoaderConfigurationError, match="omits"):
        HeartRateLoader(root=CURATED_SAMPLE).get_data(columns=["start_date"]).with_local_time()
    with pytest.raises(DataLoaderConfigurationError, match="day key"):
        HeartRateLoader(root=CURATED_SAMPLE).get_data().with_local_time(["datetime"])
    with pytest.raises(DataLoaderConfigurationError, match="not a timezone-aware"):
        HeartRateLoader(root=CURATED_SAMPLE).get_data().with_local_time(["value"])


@needs_sample
def test_local_time_on_one_named_column_and_on_an_empty_result():
    one = HeartRateLoader(root=CURATED_SAMPLE).get_data().with_local_time("start_date")
    assert "start_date_local" in one.df.columns and "end_date_local" not in one.df.columns
    empty = HeartRateLoader(root=CURATED_SAMPLE).get_data(start_date="2099-01-01").with_local_time()
    assert empty.df.empty and "start_date_local" in empty.df.columns


# =========================================================================== harmonized values
@needs_sample
def test_harmonization_invariants_for_every_feature_and_phase():
    """Every harmonized value must equal the column its source names, and be empty when unresolved."""
    for phase, root in ROOTS.items():
        for feature in _features():
            data = _loader(feature)(phase=phase, root=root).get_data()
            if feature in NOT_APPLICABLE:
                with pytest.raises(DataLoaderConfigurationError):
                    data.with_harmonized_values()
                continue
            df = data.with_harmonized_values().df
            source = df["harmonized_unit_source"].astype(str)
            value, unit = df["harmonized_value"], df["harmonized_unit"]
            policy = get_feature_spec(feature).unit_policy
            fixed = policy.canonical_unit if policy and policy.raw_unit == policy.canonical_unit else None
            canonical = source.isin(["curation", "processing"])
            if canonical.any():
                assert (value[canonical] == df.loc[canonical, "canonical_value"]).all(), (phase, feature)
            registry = source == "registry"
            assert (value[registry] == df.loc[registry, "value"]).all(), (phase, feature)
            assert (unit[registry].astype(str) == str(fixed)).all(), (phase, feature)
            assert value[source == "unresolved"].isna().all(), (phase, feature)
            assert (value.notna() == unit.notna()).all(), (phase, feature)
            assert phase == "curated" or not (source == "curation").any(), (phase, feature)
            assert df["value"].equals(data.df["value"]), (phase, feature)


@needs_sample
@pytest.mark.parametrize("loader, phase, expected", [
    (WeightLoader, "curated", {"curation": 82, "unresolved": 375}),
    (WeightLoader, "native", {"unresolved": 457}),
    (EnergyConsumedLoader, "curated", {"unresolved": 832}),
    (EnergyConsumedLoader, "native", {"unresolved": 832}),
    (OxygenSaturationLoader, "curated", {"curation": 4_757}),
    (OxygenSaturationLoader, "native", {"processing": 4_757}),
    (HeartRateLoader, "curated", {"registry": 379_245}),
])
def test_harmonization_sources(loader, phase, expected):
    report = loader(phase=phase, root=ROOTS[phase]).get_data().with_harmonized_values().load_report
    assert report["derived"]["harmonized_values"]["sources"] == expected


@needs_sample
def test_curation_withholding_of_energy_consumed_beats_the_registry_unit():
    """The registry says kcal, but curation found that label untrustworthy, so no row is harmonized."""
    df = EnergyConsumedLoader(root=CURATED_SAMPLE).get_data().with_harmonized_values().df
    assert df["harmonized_value"].isna().all()
    assert set(df["curation_unit_status"].astype(str)) == {"ambiguous"}


@needs_sample
def test_weight_is_harmonized_to_kilograms_only_where_curation_resolved_it():
    df = WeightLoader(root=CURATED_SAMPLE).get_data().with_harmonized_values().df
    resolved = df["harmonized_unit_source"] == "curation"
    assert set(df.loc[resolved, "harmonized_unit"].astype(str)) == {"kg"}
    assert df.loc[resolved, "curation_unit_status"].astype(str).str.startswith("resolved").all()


@needs_sample
def test_canonical_value_is_never_filled():
    data = HeartRateLoader(root=CURATED_SAMPLE).get_data(projection="full")
    harmonized = data.with_harmonized_values()
    assert "canonical_value" not in data.df.columns or data.df["canonical_value"].equals(
        harmonized.df["canonical_value"])
    assert harmonized.df["harmonized_value"].notna().all()


@needs_sample
def test_dropping_a_unit_verdict_is_refused_rather_than_guessed():
    with pytest.raises(DataLoaderConfigurationError, match="curation_unit_status"):
        EnergyConsumedLoader(root=CURATED_SAMPLE).get_data(columns=["value"]).with_harmonized_values()
    with pytest.raises(DataLoaderConfigurationError, match="omits"):
        WeightLoader(root=CURATED_SAMPLE).get_data(columns=["value"]).with_harmonized_values()
    # HeartRate stores no unit inputs, so a narrow request is still sound.
    narrow = HeartRateLoader(root=CURATED_SAMPLE).get_data(columns=["value"]).with_harmonized_values()
    assert (narrow.df["harmonized_unit_source"] == "registry").all()


@needs_sample
def test_harmonized_source_concatenates_across_calls():
    """Fixed categories keep the source column categorical when separate results are concatenated."""
    a = WeightLoader(root=CURATED_SAMPLE).get_data(reg_ids="10K_2719597610").with_harmonized_values().df
    b = HeartRateLoader(root=CURATED_SAMPLE).get_data(reg_ids=f"10K_{P1}").with_harmonized_values().df
    combined = pd.concat([a[["harmonized_unit_source"]], b[["harmonized_unit_source"]]])
    assert isinstance(combined["harmonized_unit_source"].dtype, pd.CategoricalDtype)


@needs_sample
def test_harmonization_metadata_report_and_empty_result():
    data = WeightLoader(root=CURATED_SAMPLE).get_data().with_harmonized_values()
    cm = data.df_columns_metadata
    assert list(cm.index) == list(data.df.columns)
    assert cm.loc["harmonized_value", "role"] == "derived"
    assert str(data.df["harmonized_value"].dtype) == "float64"
    empty = WeightLoader(root=CURATED_SAMPLE).get_data(start_date="2099-01-01").with_harmonized_values()
    assert empty.df.empty and "harmonized_value" in empty.df.columns
    assert empty.load_report["derived"]["harmonized_values"]["sources"] == {}


@needs_sample
def test_derived_columns_chain_and_recompute_cleanly():
    data = WeightLoader(root=CURATED_SAMPLE).get_data()
    chained = data.with_local_time().with_harmonized_values().with_harmonized_values()
    assert list(chained.df.columns).count("harmonized_value") == 1
    assert set(chained.load_report["derived"]) == {"local_time", "harmonized_values"}


# ============================================================================ unit reconciliation
from wearable_project.DataLoaders._units import stored_unit  # noqa: E402
from wearable_project.curation.registry import get_policy  # noqa: E402
from wearable_project.processing.registry import SPECS  # noqa: E402


def test_curation_withholds_energy_consumed_units_in_the_resolver():
    unit = stored_unit("EnergyConsumed", "value")
    assert unit.unit is None and unit.withheld_by_curation
    assert unit.candidates == ("kcal", "kJ")
    assert unit.canonical_unit == "kcal" and not unit.already_canonical


@pytest.mark.parametrize("feature, column, expected", [
    ("HeartRate", "value", "beats/min"),
    ("StepCount", "value", "count"),
    ("BloodPressure", "blood_pressure_systolic_value", "mmHg"),
])
def test_processing_units_stand_where_curation_does_not_withhold_them(feature, column, expected):
    unit = stored_unit(feature, column)
    assert unit.unit == expected and not unit.withheld_by_curation and unit.already_canonical


def test_resolver_reports_withholding_even_where_processing_names_no_unit():
    unit = stored_unit("Weight", "value")
    assert unit.unit is None and unit.withheld_by_curation and unit.candidates == ("kg", "lb")


def test_resolver_ignores_columns_that_are_not_measurements():
    assert stored_unit("HeartRate", "source_name") == stored_unit("HeartRate", "not_a_column")
    assert stored_unit("HeartRate", "source_name").unit is None


def test_the_registries_never_name_conflicting_units():
    """Curation may withhold a unit or name one processing leaves open, but never contradict processing's label."""
    conflicts = []
    for feature, spec in SPECS.items():
        try:
            declarations = get_policy(feature, allow_fallback=False).units.measurements
        except KeyError:
            continue
        for declaration in declarations:
            if declaration.measurement in spec.measurement_columns and spec.unit_policy is not None \
                    and declaration.raw_unit is not None and declaration.raw_unit != spec.unit_policy.raw_unit:
                conflicts.append((feature, declaration.measurement))
    assert conflicts == []


def test_only_energy_consumed_loses_a_processing_unit_today():
    """If this changes, a registry changed: review the newly withheld feature before relying on its values."""
    withdrawn = {
        feature for feature, spec in SPECS.items() if spec.unit_policy is not None
        for column in spec.measurement_columns if stored_unit(feature, column).withheld_by_curation
    }
    assert withdrawn == {"EnergyConsumed"}


@needs_sample
@pytest.mark.parametrize("phase", ["curated", "native"])
def test_energy_consumed_is_unresolved_in_both_phases_and_keeps_its_stored_label(phase):
    data = EnergyConsumedLoader(phase=phase, root=ROOTS[phase]).get_data()
    harmonized = data.with_harmonized_values()
    assert harmonized.df["harmonized_value"].isna().all()
    assert set(harmonized.df["harmonized_unit_source"].astype(str)) == {"unresolved"}
    assert harmonized.df["value"].equals(data.df["value"])
    assert set(harmonized.df["raw_unit"].astype(str)) == {"kcal"}
    report = harmonized.load_report["derived"]["harmonized_values"]
    assert report["unit_withheld_by_curation"] is True and report["unit_candidates"] == ["kcal", "kJ"]
    assert data.df_columns_metadata.loc["value", "registry_unit"] is None


@needs_sample
def test_metadata_and_harmonizer_agree_on_the_established_unit():
    data = HeartRateLoader(root=CURATED_SAMPLE).get_data()
    report = data.with_harmonized_values().load_report["derived"]["harmonized_values"]
    assert data.df_columns_metadata.loc["value", "registry_unit"] == report["registry_unit"] == "beats/min"
    assert report["unit_withheld_by_curation"] is False and report["unit_candidates"] == []
