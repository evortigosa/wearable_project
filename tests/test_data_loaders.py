"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import importlib
from pathlib import Path
import pandas as pd
import pytest
from wearable_project.DataLoaders._base import DEFAULT_CURATED_ROOT, DEFAULT_NATIVE_ROOT
from wearable_project.DataLoaders.ActivitySummaryLoader import ActivitySummaryLoader
from wearable_project.DataLoaders.StepCountLoader import StepCountLoader
from wearable_project.DataLoaders.WeightLoader import WeightLoader
from wearable_project.exceptions import DataLoaderConfigurationError, DataLoaderPathError, DataLoaderReadError
from wearable_project.curation.registry import COHORT_OBSERVED_FEATURES
from wearable_project import DataLoaders


CLASS_NAMES = {
    "ActiveEnergyBurned": "ActiveEnergyBurnedLoader",
    "ActivitySummary": "ActivitySummaryLoader",
    "BMI": "BMILoader",
    "BasalEnergyBurned": "BasalEnergyBurnedLoader",
    "BloodAlcoholContent": "BloodAlcoholContentLoader",
    "BloodGlucose": "BloodGlucoseLoader",
    "BloodPressure": "BloodPressureLoader",
    "BodyFatPercentage": "BodyFatPercentageLoader",
    "BodyTemperature": "BodyTemperatureLoader",
    "Carbohydrates": "CarbohydratesLoader",
    "DailyDistanceCycling": "DailyDistanceCyclingLoader",
    "DailyDistanceSwimming": "DailyDistanceSwimmingLoader",
    "DistanceWalkingRunning": "DistanceWalkingRunningLoader",
    "Electrocardiogram": "ElectrocardiogramLoader",
    "EnergyConsumed": "EnergyConsumedLoader",
    "FlightsClimbed": "FlightsClimbedLoader",
    "HeartRate": "HeartRateLoader",
    "HeartRateVariability": "HeartRateVariabilityLoader",
    "Height": "HeightLoader",
    "LeanBodyMass": "LeanBodyMassLoader",
    "Mindful": "MindfulLoader",
    "OxygenSaturation": "OxygenSaturationLoader",
    "PeakFlow": "PeakFlowLoader",
    "Protein": "ProteinLoader",
    "RespiratoryRate": "RespiratoryRateLoader",
    "RestingHeartRate": "RestingHeartRateLoader",
    "Sleep": "SleepLoader",
    "StepCount": "StepCountLoader",
    "TotalFat": "TotalFatLoader",
    "Vo2Max": "Vo2MaxLoader",
    "WaistCircumference": "WaistCircumferenceLoader",
    "WalkingHeartRate": "WalkingHeartRateLoader",
    "Weight": "WeightLoader",
}


def _write_csv(root: Path, participant: str, feature: str, rows: list[dict]) -> None:
    target = root / participant
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target / f"{feature}.csv", index=False)


def _base_rows() -> list[dict]:
    return [
        {
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-01T00:05:00Z",
            "value": 10,
            "datetime": "2024-01-01T00:00:00Z",
            "created_at": "2024-01-02T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
            "data_source": "AppleHealthkit",
            "collecting_method_version": 2,
        },
        {
            "start_date": "2024-01-02T00:00:00.123000Z",
            "end_date": "2024-01-02T00:05:00.123000Z",
            "value": 20,
            "datetime": "2024-01-02T00:00:00Z",
            "created_at": "2024-01-03T00:00:00Z",
            "updated_at": "2024-01-03T00:00:00Z",
            "data_source": "AppleHealthkit",
            "collecting_method_version": 2,
        },
    ]


def test_exactly_one_public_loader_for_each_observed_cohort_feature():
    assert set(CLASS_NAMES) == set(COHORT_OBSERVED_FEATURES)
    assert len(CLASS_NAMES) == 33
    for feature, class_name in CLASS_NAMES.items():
        module = importlib.import_module(f"wearable_project.DataLoaders.{feature}Loader")
        cls = getattr(module, class_name)
        assert cls.feature_name == feature
        assert cls()._data_index_names == ["RegistrationCode", "Date"]


def test_default_phase_and_hpp_roots():
    loader = StepCountLoader()
    assert loader.phase == "curated"
    assert loader.native_root == DEFAULT_NATIVE_ROOT
    assert loader.curated_root == DEFAULT_CURATED_ROOT
    assert loader.data_root == DEFAULT_CURATED_ROOT


def test_root_override_and_native_phase(tmp_path: Path):
    loader = StepCountLoader(phase="native", root=tmp_path)
    assert loader.phase == "native"
    assert loader.data_root == tmp_path


def test_invalid_phase_fails():
    with pytest.raises(DataLoaderConfigurationError):
        StepCountLoader(phase="silver")


def test_missing_root_fails(tmp_path: Path):
    with pytest.raises(DataLoaderPathError):
        StepCountLoader(root=tmp_path / "missing").get_data()


def test_get_data_returns_hpp_style_multiindex_and_10k_codes(tmp_path: Path):
    _write_csv(tmp_path, "123", "StepCount", _base_rows())
    _write_csv(tmp_path, "456", "StepCount", _base_rows()[:1])

    result = StepCountLoader(root=tmp_path).get_data()
    df = result.df

    assert df.index.names == ["RegistrationCode", "Date"]
    assert set(df.index.get_level_values("RegistrationCode")) == {
        "10K_123",
        "10K_456",
    }
    assert str(df.index.get_level_values("Date").dtype) == "datetime64[ns, UTC]"
    assert len(df) == 3
    assert result.metadata["feature"] == "StepCount"
    assert result.metadata["phase"] == "curated"
    assert result.metadata["files_read"] == 2


def test_registration_code_and_date_filters(tmp_path: Path):
    _write_csv(tmp_path, "123", "StepCount", _base_rows())
    _write_csv(tmp_path, "456", "StepCount", _base_rows())

    df = StepCountLoader(root=tmp_path).get_data(
        registration_codes="10K_123",
        start_date="2024-01-02",
        end_date="2024-01-02 23:59:59",
    ).df

    assert len(df) == 1
    assert df.index[0][0] == "10K_123"
    assert df.index[0][1] == pd.Timestamp("2024-01-02T00:00:00.123000Z")


def test_columns_selection_does_not_force_date_anchor_column(tmp_path: Path):
    _write_csv(tmp_path, "123", "StepCount", _base_rows())
    df = StepCountLoader(root=tmp_path).get_data(registration_codes="123", columns=["value"]).df
    assert df.columns.tolist() == ["value"]
    assert len(df) == 2


def test_curated_loader_does_not_silently_drop_excluded_rows(tmp_path: Path):
    rows = _base_rows()
    rows[0].update(
        {
            "canonical_value": 10,
            "canonical_unit": "kg",
            "curation_status": "pass",
            "include_by_default": 1,
        }
    )
    rows[1].update(
        {
            "canonical_value": pd.NA,
            "canonical_unit": pd.NA,
            "curation_status": "review",
            "include_by_default": 0,
        }
    )
    _write_csv(tmp_path, "123", "Weight", rows)

    loader = WeightLoader(root=tmp_path)
    assert len(loader.get_data().df) == 2
    filtered = loader.get_data(default_inclusion_only=True).df
    assert len(filtered) == 1
    assert filtered.iloc[0]["value"] == 10


def test_default_inclusion_filter_is_not_valid_for_native(tmp_path: Path):
    _write_csv(tmp_path, "123", "StepCount", _base_rows())
    with pytest.raises(DataLoaderConfigurationError):
        StepCountLoader(phase="native", root=tmp_path).get_data(default_inclusion_only=True)


def test_activity_summary_uses_outer_datetime_without_claiming_canonical_day(tmp_path: Path,):
    rows = [
        {
            "datetime": "2024-02-01T00:00:00Z",
            "apple_stand_hours": 5,
            "apple_exercise_time": 30,
            "active_energy_burned": 400,
            "apple_stand_hours_goal": 12,
            "apple_exercise_time_goal": 30,
            "active_energy_burned_goal": 500,
            "created_at": "2024-02-02T00:00:00Z",
            "updated_at": "2024-02-02T00:00:00Z",
            "data_source": "AppleHealthkit",
            "collecting_method_version": 2,
            "curation_status": "review",
            "include_by_default": 0,
        }
    ]
    _write_csv(tmp_path, "123", "ActivitySummary", rows)
    result = ActivitySummaryLoader(root=tmp_path).get_data()
    assert result.df.index[0] == ("10K_123", pd.Timestamp("2024-02-01T00:00:00Z"),)
    assert (
        result.metadata["date_semantics"] == "outer_summary_datetime_not_canonical_summary_day"
    )


def test_missing_feature_for_requested_participant_returns_empty_hpp_frame(tmp_path: Path,):
    (tmp_path / "123").mkdir(parents=True)
    result = StepCountLoader(root=tmp_path).get_data("10K_123")
    assert result.df.empty
    assert result.df.index.names == ["RegistrationCode", "Date"]


def test_requested_missing_participant_is_reported_without_failure(tmp_path: Path):
    (tmp_path / "123").mkdir(parents=True)
    result = StepCountLoader(root=tmp_path).get_data("10K_999")
    assert result.df.empty
    assert result.metadata["requested_participants_missing_from_root"] == ["999"]


def test_sparse_curated_columns_are_filled_across_participants(tmp_path: Path):
    first = _base_rows()[:1]
    first[0]["canonical_value"] = 10.0
    second = _base_rows()[:1]
    _write_csv(tmp_path, "123", "Weight", first)
    _write_csv(tmp_path, "456", "Weight", second)

    df = WeightLoader(root=tmp_path).get_data(
        columns=["value", "canonical_value"]
    ).df
    assert len(df) == 2
    assert df.loc[("10K_123", pd.Timestamp("2024-01-01T00:00:00Z")), "canonical_value"] == 10.0
    assert pd.isna(df.loc[("10K_456", pd.Timestamp("2024-01-01T00:00:00Z")), "canonical_value"])


def test_requested_column_missing_from_every_feature_file_is_rejected(tmp_path: Path):
    _write_csv(tmp_path, "123", "StepCount", _base_rows())

    with pytest.raises(DataLoaderReadError):
        StepCountLoader(root=tmp_path).get_data(columns=["definitely_not_a_column"])


def test_dataloaders_info_overview_is_complete_and_read_only():
    report = DataLoaders.info()
    payload = report.as_dict()
    assert report.kind == "overview"
    assert payload["feature_count"] == 33
    assert {item["feature"] for item in payload["features"]} == set(COHORT_OBSERVED_FEATURES)
    assert payload["defaults"]["phase"] == "curated"
    assert payload["defaults"]["index_names"] == ["RegistrationCode", "Date"]
    assert "never silently filtered" in str(report)
    assert "StepCount" in str(report)


def test_feature_info_combines_loader_processing_guidance_and_curation(tmp_path: Path):
    report = DataLoaders.info("stepcount", phase="native", root=tmp_path)
    payload = report.as_dict()
    assert report.kind == "feature"
    assert payload["feature"] == "StepCount"
    assert payload["loader"]["class"] == "StepCountLoader"
    assert payload["loader"]["phase"] == "native"
    assert payload["loader"]["resolved_root"] == str(tmp_path)
    assert payload["processing"]["family"] == "interval_total"
    assert payload["guidance"]["short_description"]
    assert payload["curation"]["policy"]["identity"]["canonical_name"] == "StepCount"
    assert payload["loader_behavior"]["all_curated_rows_by_default"] is True
    assert "Future resampling declaration" in str(report)


def test_loader_instance_info_reflects_instance_configuration(tmp_path: Path):
    loader = WeightLoader(phase="curated", root=tmp_path)
    report = loader.info()
    assert report.as_dict()["loader"]["resolved_root"] == str(tmp_path)
    assert report.as_dict()["feature"] == "Weight"
    assert "Weight" in str(report)


def test_feature_info_can_include_evidence_catalog_details():
    report = DataLoaders.info("Sleep", include_evidence=True)
    payload = report.as_dict()
    assert "apple_sleep_analysis" in payload["evidence_refs"]
    assert payload["evidence"]["apple_sleep_analysis"]["organization"] == "Apple Developer Documentation"


def test_feature_info_rejects_unknown_feature():
    with pytest.raises(DataLoaderConfigurationError):
        DataLoaders.info("DefinitelyNotAFeature")
