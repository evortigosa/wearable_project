"""
Wearable Data Processing and Modeling project
Unit tests for scripts/build_visit_features.py on synthetic daily frames; no data root needed.
"""


from __future__ import annotations
import importlib.util
from pathlib import Path
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_visit_features.py"
spec = importlib.util.spec_from_file_location("build_visit_features", SCRIPT)
vf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vf)


def visits(*rows: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["RegistrationCode", "research_stage", "visit_date"]).assign(
        visit_date=lambda frame: pd.to_datetime(frame["visit_date"]))


def daily(dates: list[str], code: str = "10K_1", **columns) -> pd.DataFrame:
    defaults = {"records": 24, "hours_with_data": 24, "observed_minutes": 1440.0, "values_below_range": 0.0,
                "values_above_range": 0.0, "value_sum": 8000.0, "value_median": 60.0}
    frame = pd.DataFrame({"RegistrationCode": code, "local_date": dates})
    for name, value in {**defaults, **columns}.items():
        frame[name] = value
    return frame


def test_window_is_inclusive_and_repeats_a_day_in_overlapping_windows():
    v = visits(("10K_1", "00_00_visit", "2020-01-01"), ("10K_1", "02_00_visit", "2021-06-01"))
    days = daily(["2018-12-31", "2019-01-01", "2020-12-31", "2021-01-01", "2022-06-01", "2022-06-02"])
    windowed = vf.attach_windows(days, v, "local_date")
    by_stage = windowed.groupby("research_stage")["local_date"].apply(list).to_dict()
    assert by_stage["00_00_visit"] == ["2019-01-01", "2020-12-31"]
    assert by_stage["02_00_visit"] == ["2020-12-31", "2021-01-01", "2022-06-01"]
    assert windowed.loc[windowed["local_date"] == "2019-01-01", "days_from_visit"].item() == -365


def test_window_ignores_other_participants():
    v = visits(("10K_2", "00_00_visit", "2020-01-01"))
    assert vf.attach_windows(daily(["2020-01-01"]), v, "local_date").empty


def test_whole_day_single_record_passes_coverage_by_observed_minutes():
    days = daily(["2016-05-11", "2016-05-12"], records=1, hours_with_data=1, observed_minutes=[930.0, 300.0])
    assert vf.day_drop_reason("StepCount", days).tolist() == ["", "low_coverage"]


def test_total_bounds_drop_days_without_clipping():
    days = daily(["d1", "d2", "d3"], value_sum=[70_000.0, 12_000.0, -1.0])
    assert vf.day_drop_reason("StepCount", days).tolist() == ["above_bound", "", "below_bound"]
    assert days["value_sum"].tolist() == [70_000.0, 12_000.0, -1.0]


def test_basal_energy_lower_bound_drops_partial_days():
    days = daily(["d1", "d2"], value_sum=[200.0, 1500.0])
    assert vf.day_drop_reason("BasalEnergyBurned", days).tolist() == ["below_bound", ""]


def test_out_of_range_share_takes_precedence():
    days = daily(["d1", "d2"], records=100, values_above_range=[10.0, 2.0], hours_with_data=[24, 5])
    assert vf.day_drop_reason("HeartRate", days).tolist() == ["out_of_range", "low_coverage"]


def test_level_feature_without_value_is_dropped():
    days = daily(["d1", "d2"], value_median=[float("nan"), 62.0])
    assert vf.day_drop_reason("RestingHeartRate", days).tolist() == ["no_value", ""]


def test_flights_fill_zero_on_valid_step_days_only():
    key = {"RegistrationCode": "10K_1", "research_stage": "00_00_visit"}
    steps = pd.DataFrame([{**key, "local_date": d, "days_from_visit": i, "drop_reason": r}
                          for i, (d, r) in enumerate([("d1", ""), ("d2", ""), ("d3", "low_coverage")])])
    flights = pd.DataFrame([{**key, "local_date": "d1", "days_from_visit": 0, "visit_date": pd.Timestamp("2020-01-01"),
                             "records": 2, "values_below_range": 0.0, "values_above_range": 0.0, "value_sum": 5.0},
                            {**key, "local_date": "d3", "days_from_visit": 2, "visit_date": pd.Timestamp("2020-01-01"),
                             "records": 1, "values_below_range": 0.0, "values_above_range": 0.0, "value_sum": 9.0}])
    days = vf.flights_on_valid_step_days(flights, steps)
    assert days.set_index("local_date")["value_sum"].to_dict() == {"d1": 5.0, "d2": 0.0}


def test_flights_skip_participants_without_any_flights_record():
    steps = pd.DataFrame([{"RegistrationCode": "10K_1", "research_stage": "00_00_visit", "local_date": "d1",
                           "days_from_visit": 0, "drop_reason": ""}])
    flights = pd.DataFrame(columns=[*vf.KEY, "local_date", "days_from_visit", "visit_date", "records",
                                    "values_below_range", "values_above_range", "value_sum"])
    assert vf.flights_on_valid_step_days(flights, steps).empty


def windowed_days(values: list[float], start: str = "2020-01-05") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({"local_date": dates.strftime("%Y-%m-%d"), "value_sum": values, "value_median": values,
                         "days_from_visit": range(len(values)), "readings_below_90_percent": 0})


def test_minimum_valid_days():
    assert vf.summarize_feature_window("StepCount", windowed_days([5000.0] * 6)) == {}
    assert vf.summarize_feature_window("StepCount", windowed_days([5000.0] * 7))["steps_median"] == 5000.0
    assert vf.summarize_feature_window("Vo2Max", windowed_days([40.0]))["vo2max_median"] == 40.0


def test_step_statistics_use_the_israeli_weekend():
    # 2020-01-05 is a Sunday; two weeks, Friday and Saturday at 2,000 steps, other days at 12,000.
    dates = pd.date_range("2020-01-05", periods=14, freq="D")
    values = [2000.0 if d.weekday() in (4, 5) else 12000.0 for d in dates]
    stats = vf.summarize_feature_window("StepCount", windowed_days(values))
    assert stats["steps_weekday_minus_weekend"] == 10000.0
    assert stats["steps_share_days_below_5k"] == pytest.approx(4 / 14)
    assert stats["steps_share_days_above_10k"] == pytest.approx(10 / 14)


def test_weekday_contrast_needs_enough_days_on_each_side():
    dates = pd.date_range("2020-01-05", periods=7, freq="D")
    stats = vf.summarize_feature_window("StepCount", windowed_days([1000.0] * len(dates)))
    assert "steps_weekday_minus_weekend" not in stats


def test_resting_hr_slope_needs_a_long_enough_span():
    short = windowed_days([60.0 + i for i in range(30)])
    assert "resting_hr_slope_per_year" not in vf.summarize_feature_window("RestingHeartRate", short)
    long = short.assign(days_from_visit=[i * 10 for i in range(30)], value_median=[60.0 + i * 10 / 365.25 for i in range(30)])
    slope = vf.summarize_feature_window("RestingHeartRate", long)["resting_hr_slope_per_year"]
    assert slope == pytest.approx(1.0)


def test_vo2max_nearest_visit():
    days = windowed_days([30.0, 40.0, 50.0]).assign(days_from_visit=[-100, 5, 200])
    assert vf.summarize_feature_window("Vo2Max", days)["vo2max_nearest_visit"] == 40.0


def nights(n: int, **columns) -> pd.DataFrame:
    defaults = {"asleep_recorded": True, "in_bed_recorded": True, "staged": False, "sleep_period_minutes": 450.0,
                "asleep_minutes": 420.0, "in_bed_minutes": 460.0, "midpoint_hours_after_noon": 15.0,
                "efficiency": 0.93, "waso_minutes": 30.0, "rem_share": float("nan"), "deep_share": float("nan")}
    return pd.DataFrame({name: [value] * n for name, value in {**defaults, **columns}.items()})


def test_night_drop_reasons():
    frame = pd.concat([nights(1, asleep_recorded=False), nights(1, sleep_period_minutes=100.0),
                       nights(1, sleep_period_minutes=1000.0), nights(1)], ignore_index=True)
    assert vf.night_drop_reason(frame).tolist() == ["no_asleep_time", "below_bound", "above_bound", ""]


def test_in_bed_only_nights_are_kept_for_in_bed_features():
    frame = nights(14, asleep_recorded=False, sleep_period_minutes=float("nan"))
    assert (vf.night_drop_reason(frame) == "no_asleep_time").all()
    assert (vf.in_bed_drop_reason(frame) == "").all()
    assert vf.summarize_in_bed_window(frame)["sleep_in_bed_minutes_median"] == 460.0


def test_sleep_summary_needs_minimum_nights_and_reports_stages_only_when_enough():
    assert vf.summarize_sleep_window(nights(13)) == {}
    stats = vf.summarize_sleep_window(nights(14))
    assert stats["sleep_asleep_minutes_median"] == 420.0 and stats["sleep_staged_share"] == 0.0
    assert "sleep_rem_share_median" not in stats


def test_midpoint_after_midnight_is_continuous():
    frame = nights(14).assign(midpoint_hours_after_noon=[11.5, 12.5] * 7)  # 23:30 and 00:30
    stats = vf.summarize_sleep_window(frame)
    assert stats["sleep_midpoint_hours_after_noon_median"] == 12.0
    assert stats["sleep_midpoint_hours_sd"] == pytest.approx(0.519, abs=1e-3)


def test_cohort_outlier_flag_is_per_visit_and_never_removes():
    frame = pd.DataFrame({"research_stage": ["00_00_visit"] * 6 + ["02_00_visit"] * 6,
                          "steps_median": [5000, 5100, 4900, 5050, 4950, 30000, 30000, 30100, 29900, 30050, 29950, 30000]})
    flags = vf.flag_cohort_outliers(frame, ["steps_median"])
    assert flags.tolist() == [""] * 5 + ["steps_median"] + [""] * 6
    assert len(frame) == 12


def test_cohort_outlier_flag_skips_columns_without_spread():
    frame = pd.DataFrame({"research_stage": ["00_00_visit"] * 3, "steps_median": [5000.0, 5000.0, 9000.0]})
    assert vf.flag_cohort_outliers(frame, ["steps_median"]).tolist() == ["", "", ""]


def test_every_feature_column_is_described():
    prefixes = [*vf.FEATURE_UNITS]
    for prefix in prefixes:
        for suffix in vf.STATISTIC_SUFFIXES:
            definition, unit = vf.describe_column(prefix + suffix)
            assert definition and unit
    for column in vf.SPECIFIC_COLUMNS:
        assert all(vf.describe_column(column))
