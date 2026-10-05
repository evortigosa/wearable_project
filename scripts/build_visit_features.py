"""
Wearable Data Processing and Modeling project
Per-visit wearable summary features: one row per participant and HPP visit, summarizing the days within
WINDOW_DAYS of the visit date. Reads a written daily-statistics run and a written domain-metrics run (sleep nights);
visit dates come from the pheno snapshot's clinic-visit tables.
Outliers are handled in three layers, each counted in outlier_audit.csv: days dominated by implausible readings are
dropped; days with too little coverage, or beyond a physiological cap, are dropped (never clipped); and a
participant-visit is reported only with enough valid days, then flagged, never removed, when far from its cohort.

Usage:
    python scripts/build_visit_features.py --daily-run RUN/daily --domain-run RUN/domain --out RUN/features
"""


from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from wearable_project.utils import data_statistics as ds
from wearable_project.utils import domain_metrics as dm


SNAPSHOT = Path("/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/Pheno/snapshots/pheno_data_snapshots/out_dir")
# Clinic tables measured on the visit day itself; their dates agree for 99% of participant-visits. Sleep, CGM and
# fundus are excluded because they are taken over days at home or on a separate appointment.
VISIT_TABLES = ("anthropometrics/anthropometrics", "blood_pressure/blood_pressure", "ABI/ABI", "ECG/ECG")
VISIT_STAGES = ("00_00_visit", "02_00_visit", "04_00_visit", "06_00_visit")
WINDOW_DAYS = 365

# Layer 1: a day whose records are more than this share outside ds.PLAUSIBLE_RANGES is dropped. Below it, the day's
# median stands, since a median is robust to an occasional bad reading.
MAX_OUT_OF_RANGE_SHARE = 0.05

# Layer 2: day validity. Totals need 10 h of recorded time. observed_minutes, not hours_with_data, because up to
# about 2022 roughly half of all step-days are one record spanning most of the day (hours_with_data == 1).
MIN_OBSERVED_MINUTES_FOR_TOTALS = 600
MIN_HOURS_WITH_DATA = {"HeartRate": 10}
# Flights are recorded only when climbing, so their own records never cover a day. Their days are the valid StepCount
# days instead, and a valid step day without a flights record counts as 0: a carried phone records every climb.
FLIGHTS_FEATURE = "FlightsClimbed"
# Physiological bounds on a day's total; a day outside them is dropped, not clipped.
DAILY_TOTAL_BOUNDS = {
    "StepCount": (0.0, 60_000.0),
    "ActiveEnergyBurned": (0.0, 5_000.0),        # kcal
    "BasalEnergyBurned": (500.0, 4_000.0),       # kcal
    "DistanceWalkingRunning": (0.0, 60_000.0),   # m
    "FlightsClimbed": (0.0, 300.0),
}
TOTAL_FEATURES = tuple(DAILY_TOTAL_BOUNDS)
LEVEL_FEATURES = ("HeartRate", "RestingHeartRate", "WalkingHeartRate", "HeartRateVariability", "OxygenSaturation",
                  "RespiratoryRate", "Vo2Max")
# Bounds on a night's main sleep period (nights with asleep time, mostly Apple Watch) and on time in bed (nights
# from the iPhone bedtime schedule, which record time in bed only).
SLEEP_PERIOD_MINUTES_BOUNDS = (180.0, 960.0)
IN_BED_MINUTES_BOUNDS = (180.0, 960.0)

# Layer 3: participant-visit validity and cohort outlier flags.
MIN_VALID_DAYS = 7
MIN_VALID_DAYS_BY_FEATURE = {"Vo2Max": 1}
MIN_VALID_NIGHTS = 14
# A trend needs valid days spread over this span; a contrast needs this many valid days on each side.
MIN_SLOPE_SPAN_DAYS = 180
MIN_DAYS_PER_WEEK_PART = 4
OUTLIER_MAD_THRESHOLD = 4.0
MAD_TO_SD = 1.4826

STEPS_LOW_DAY = 5_000
STEPS_HIGH_DAY = 10_000
WEEKEND_WEEKDAYS = (4, 5)  # Friday and Saturday, the Israeli weekend

FEATURE_PREFIX = {
    "StepCount": "steps", "ActiveEnergyBurned": "active_energy", "BasalEnergyBurned": "basal_energy",
    "DistanceWalkingRunning": "distance", "FlightsClimbed": "flights", "HeartRate": "hr",
    "RestingHeartRate": "resting_hr", "WalkingHeartRate": "walking_hr", "HeartRateVariability": "hrv",
    "OxygenSaturation": "spo2", "RespiratoryRate": "resp_rate", "Vo2Max": "vo2max",
}
KEY = ["RegistrationCode", "research_stage"]
DAILY_COLUMNS = ["RegistrationCode", "local_date", "records", "hours_with_data", "observed_minutes",
                 "values_below_range", "values_above_range", "value_sum", "value_median", "readings_below_90_percent",
                 "minutes"]


# ------------------------------------------------------------------------------------------------ visits and windows
def load_visit_dates(snapshot: Path = SNAPSHOT) -> pd.DataFrame:
    """One date per (RegistrationCode, research_stage): the most common date across the clinic-visit tables."""

    frames = [pd.read_parquet(snapshot / table / "df_metadata.parquet", columns=["research_stage"]).reset_index()
              for table in VISIT_TABLES]
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[rows["research_stage"].isin(VISIT_STAGES)]
    rows["RegistrationCode"] = rows["RegistrationCode"].str.replace("10k_", "10K_", regex=False)
    rows["date"] = rows["date"].dt.tz_localize(None).dt.normalize()
    visits = (rows.groupby(KEY)["date"].agg(lambda dates: dates.mode().min())
              .rename("visit_date").reset_index())
    assert not visits.duplicated(KEY).any()
    return visits


def attach_windows(days: pd.DataFrame, visits: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Each day once per visit whose window contains it, with research_stage and days_from_visit."""

    dated = days.assign(_date=pd.to_datetime(days[date_column]))
    joined = dated.merge(visits, on="RegistrationCode", how="inner")
    joined["days_from_visit"] = (joined["_date"] - joined["visit_date"]).dt.days
    inside = joined["days_from_visit"].abs() <= WINDOW_DAYS
    return joined.loc[inside].drop(columns="_date").reset_index(drop=True)


def read_windowed_daily(daily_run: Path, feature: str, visits: pd.DataFrame) -> pd.DataFrame:
    """A feature's daily rows within any visit window, read one participant at a time to bound memory."""

    with_visits = set(visits["RegistrationCode"])
    frames = []
    for participant in ds.iter_daily(daily_run, feature):
        if participant["RegistrationCode"].iloc[0] not in with_visits:
            continue
        columns = [c for c in DAILY_COLUMNS if c in participant.columns]
        windowed = attach_windows(participant[columns], visits, "local_date")
        if not windowed.empty:
            frames.append(windowed)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ------------------------------------------------------------------------------------------------ day validity
def day_drop_reason(feature: str, days: pd.DataFrame) -> pd.Series:
    """Why each day is dropped, in layer order (first reason wins); empty string for a valid day."""

    reason = pd.Series("", index=days.index, dtype=object)
    out_of_range = days["values_below_range"].fillna(0) + days["values_above_range"].fillna(0)
    reason[out_of_range / days["records"].clip(lower=1) > MAX_OUT_OF_RANGE_SHARE] = "out_of_range"
    if feature in TOTAL_FEATURES:
        low, high = DAILY_TOTAL_BOUNDS[feature]
        if feature != FLIGHTS_FEATURE:
            reason[(reason == "") & (days["observed_minutes"] < MIN_OBSERVED_MINUTES_FOR_TOTALS)] = "low_coverage"
        reason[(reason == "") & (days["value_sum"] < low)] = "below_bound"
        reason[(reason == "") & (days["value_sum"] > high)] = "above_bound"
    elif feature in MIN_HOURS_WITH_DATA:
        reason[(reason == "") & (days["hours_with_data"] < MIN_HOURS_WITH_DATA[feature])] = "low_coverage"
    reason[(reason == "") & days[day_value_column(feature)].isna()] = "no_value"
    return reason


def night_drop_reason(nights: pd.DataFrame) -> pd.Series:
    low, high = SLEEP_PERIOD_MINUTES_BOUNDS
    reason = pd.Series("", index=nights.index, dtype=object)
    reason[~nights["asleep_recorded"].astype(bool)] = "no_asleep_time"
    reason[(reason == "") & (nights["sleep_period_minutes"] < low)] = "below_bound"
    reason[(reason == "") & (nights["sleep_period_minutes"] > high)] = "above_bound"
    return reason


def in_bed_drop_reason(nights: pd.DataFrame) -> pd.Series:
    low, high = IN_BED_MINUTES_BOUNDS
    reason = pd.Series("", index=nights.index, dtype=object)
    reason[~nights["in_bed_recorded"].astype(bool)] = "no_in_bed_time"
    reason[(reason == "") & (nights["in_bed_minutes"] < low)] = "below_bound"
    reason[(reason == "") & (nights["in_bed_minutes"] > high)] = "above_bound"
    return reason


def flights_on_valid_step_days(flights: pd.DataFrame, step_days: pd.DataFrame) -> pd.DataFrame:
    """Flights per valid step day, for participant-visits with any flights record in the window; 0 where none."""

    with_flights = flights[KEY].drop_duplicates()
    valid_steps = step_days.loc[step_days["drop_reason"] == "", [*KEY, "local_date", "days_from_visit"]]
    days = valid_steps.merge(with_flights, on=KEY).merge(
        flights.drop(columns=["days_from_visit", "visit_date"]), on=[*KEY, "local_date"], how="left")
    days["records"] = days["records"].fillna(0)
    days["value_sum"] = days["value_sum"].fillna(0.0)
    return days


def day_value_column(feature: str) -> str:
    return "value_sum" if feature in TOTAL_FEATURES else "value_median"


# ------------------------------------------------------------------------------------------------ summaries
def robust_stats(values: pd.Series, prefix: str) -> dict[str, float]:
    q10, q25, q50, q75, q90 = values.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return {f"{prefix}_median": q50, f"{prefix}_iqr": q75 - q25, f"{prefix}_p10": q10, f"{prefix}_p90": q90}


def summarize_feature_window(feature: str, valid: pd.DataFrame) -> dict[str, float]:
    """Summary of one participant-visit's valid days of one feature; empty below the minimum number of days."""

    prefix = FEATURE_PREFIX[feature]
    if len(valid) < MIN_VALID_DAYS_BY_FEATURE.get(feature, MIN_VALID_DAYS):
        return {}
    values = valid[day_value_column(feature)]
    stats = robust_stats(values, prefix)
    if feature == "StepCount":
        weekday = pd.to_datetime(valid["local_date"]).dt.weekday
        weekend = weekday.isin(WEEKEND_WEEKDAYS)
        stats["steps_share_days_below_5k"] = float((values < STEPS_LOW_DAY).mean())
        stats["steps_share_days_above_10k"] = float((values >= STEPS_HIGH_DAY).mean())
        if min(weekend.sum(), (~weekend).sum()) >= MIN_DAYS_PER_WEEK_PART:
            stats["steps_weekday_minus_weekend"] = values[~weekend].median() - values[weekend].median()
    span = valid["days_from_visit"].max() - valid["days_from_visit"].min()
    if feature == "RestingHeartRate" and span >= MIN_SLOPE_SPAN_DAYS:
        years = valid["days_from_visit"] / 365.25
        stats["resting_hr_slope_per_year"] = float(np.polyfit(years, values, 1)[0])
    if feature == "OxygenSaturation":
        stats["spo2_share_days_below_90"] = float((valid["readings_below_90_percent"] > 0).mean())
    if feature == "Vo2Max":
        nearest = valid.loc[valid["days_from_visit"].abs().idxmin()]
        stats["vo2max_nearest_visit"] = nearest["value_median"]
    return stats


def summarize_sleep_window(valid: pd.DataFrame) -> dict[str, float]:
    if len(valid) < MIN_VALID_NIGHTS:
        return {}
    stats = robust_stats(valid["asleep_minutes"], "sleep_asleep_minutes")
    staged = valid[valid["staged"].astype(bool)]
    stats.update({
        "sleep_midpoint_hours_after_noon_median": valid["midpoint_hours_after_noon"].median(),
        "sleep_midpoint_hours_sd": valid["midpoint_hours_after_noon"].std(),
        "sleep_efficiency_median": valid["efficiency"].median(),
        "sleep_waso_minutes_median": valid["waso_minutes"].median(),
        "sleep_staged_share": len(staged) / len(valid),
        "sleep_staged_nights": float(len(staged)),
    })
    if len(staged) >= MIN_VALID_NIGHTS:
        stats["sleep_rem_share_median"] = staged["rem_share"].median()
        stats["sleep_deep_share_median"] = staged["deep_share"].median()
    return stats


def summarize_in_bed_window(valid: pd.DataFrame) -> dict[str, float]:
    return robust_stats(valid["in_bed_minutes"], "sleep_in_bed_minutes") if len(valid) >= MIN_VALID_NIGHTS else {}


def summarize_wear_window(step_days: pd.DataFrame) -> dict[str, float]:
    valid = step_days[step_days["drop_reason"] == ""]
    return {
        "wear_step_days_recorded": float(len(step_days)),
        "wear_step_days_valid": float(len(valid)),
        "wear_step_valid_share": len(valid) / len(step_days),
        "wear_first_day_from_visit": float(step_days["days_from_visit"].min()),
        "wear_last_day_from_visit": float(step_days["days_from_visit"].max()),
    }


def summarize_mindful_window(mindful_days: pd.DataFrame, recorded_days: float) -> dict[str, float]:
    if not recorded_days:
        return {}
    return {"mindful_minutes_per_week": mindful_days["minutes"].sum() / (recorded_days / 7.0),
            "mindful_days": float(len(mindful_days))}


# ------------------------------------------------------------------------------------------------ outlier flags
def flag_cohort_outliers(features: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Per row, the columns more than OUTLIER_MAD_THRESHOLD robust SDs from that visit's cohort median."""

    flags = pd.DataFrame(False, index=features.index, columns=columns)
    for _, group in features.groupby("research_stage"):
        for column in columns:
            values = group[column]
            median = values.median()
            spread = MAD_TO_SD * (values - median).abs().median()
            if not spread > 0:
                continue
            flags.loc[group.index, column] = ((values - median).abs() / spread > OUTLIER_MAD_THRESHOLD).to_numpy()
    return flags.apply(lambda row: ";".join(row.index[row]), axis=1)


# ------------------------------------------------------------------------------------------------ assembly
def audit_days(feature: str, days: pd.DataFrame, reported: dict[tuple[str, str], bool]) -> pd.DataFrame:
    """Per visit: days in windows, days dropped by each reason, valid days, participant-visits with data and reported."""

    by_stage = days.groupby("research_stage")
    audit = pd.crosstab(days["research_stage"], days["drop_reason"])
    audit.columns = ["valid_days" if reason == "" else f"dropped_{reason}" for reason in audit.columns]
    audit.insert(0, "days_in_windows", by_stage.size())
    audit["participant_visits_with_data"] = by_stage["RegistrationCode"].nunique()
    reported_by_stage = pd.Series(reported, dtype=bool).groupby(level=1).sum()
    audit["participant_visits_reported"] = reported_by_stage.reindex(audit.index, fill_value=0)
    return audit.reset_index().assign(feature=feature)


def build_visit_features(daily_run: Path, domain_run: Path,
                         visits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: dict[tuple[str, str], dict[str, float]] = {}
    audits = []

    def add(key: tuple[str, str], stats: dict[str, float]) -> None:
        summaries.setdefault(key, {}).update(stats)

    def summarize_family(name: str, days: pd.DataFrame, summarize, count_column: str) -> None:
        reported = {}
        for key, window in days.groupby(KEY):
            stats = summarize(window[window["drop_reason"] == ""])
            reported[key] = bool(stats)
            stats[count_column] = float((window["drop_reason"] == "").sum())
            add(key, stats)
        audits.append(audit_days(name, days, reported))
        print(f"{name}: {len(days):,} windowed days", flush=True)

    step_days = pd.DataFrame()
    for feature in (*TOTAL_FEATURES, *LEVEL_FEATURES):
        days = read_windowed_daily(daily_run, feature, visits)
        if days.empty:
            continue
        if feature == FLIGHTS_FEATURE:
            days = flights_on_valid_step_days(days, step_days)
        days["drop_reason"] = day_drop_reason(feature, days)
        summarize_family(feature, days, lambda valid, f=feature: summarize_feature_window(f, valid),
                         f"{FEATURE_PREFIX[feature]}_valid_days")
        if feature == "StepCount":
            step_days = days
            for key, window in days.groupby(KEY):
                add(key, summarize_wear_window(window))

    recorded_step_days = step_days.groupby(KEY).size().astype(float).to_dict()

    mindful = read_windowed_daily(daily_run, "Mindful", visits)
    if not mindful.empty:
        for key, window in mindful.groupby(KEY):
            add(key, summarize_mindful_window(window, recorded_step_days.get(key, 0.0)))

    nights = attach_windows(dm.read_domain_table(domain_run, "sleep_nights"), visits, "night_date")
    summarize_family("SleepNight", nights.assign(drop_reason=night_drop_reason(nights)), summarize_sleep_window,
                     "sleep_valid_nights")
    summarize_family("SleepInBed", nights.assign(drop_reason=in_bed_drop_reason(nights)), summarize_in_bed_window,
                     "sleep_in_bed_valid_nights")

    features = pd.DataFrame.from_dict(summaries, orient="index").rename_axis(KEY).reset_index()
    features = visits.merge(features, on=KEY, how="inner", validate="one_to_one")
    flagged_columns = [c for c in features.columns if c.endswith(("_median", "_iqr", "_slope_per_year", "_nearest_visit",
                                                                  "_per_week", "_sd"))]
    features["outlier_features"] = flag_cohort_outliers(features, flagged_columns)
    features["n_outlier_features"] = features["outlier_features"].str.count(";") + (features["outlier_features"] != "")
    audit = pd.concat(audits, ignore_index=True).fillna(0)
    flag_counts = (features.assign(flag=features["outlier_features"].str.split(";")).explode("flag")
                   .query("flag != ''").groupby(["research_stage", "flag"]).size().rename("participant_visits_flagged"))
    return features.sort_values(KEY).reset_index(drop=True), audit, flag_counts.reset_index()


FEATURE_UNITS = {
    "steps": ("daily steps", "steps/day"), "active_energy": ("daily active energy", "kcal/day"),
    "basal_energy": ("daily basal energy", "kcal/day"), "distance": ("daily walking and running distance", "m/day"),
    "flights": ("daily flights climbed", "flights/day"), "hr": ("daily median heart rate", "beats/min"),
    "resting_hr": ("daily resting heart rate", "beats/min"), "walking_hr": ("daily walking heart rate", "beats/min"),
    "hrv": ("daily median HRV (SDNN)", "ms"), "spo2": ("daily median oxygen saturation", "%"),
    "resp_rate": ("daily median respiratory rate", "breaths/min"), "vo2max": ("VO2max estimate", "mL/(kg*min)"),
    "sleep_asleep_minutes": ("minutes asleep in the main sleep period, per valid night", "min"),
    "sleep_in_bed_minutes": ("minutes in bed, per valid night (includes iPhone bedtime-only nights)", "min"),
}
STATISTIC_SUFFIXES = {
    "_median": "median", "_iqr": "interquartile range", "_p10": "10th percentile", "_p90": "90th percentile",
}
SPECIFIC_COLUMNS = {
    "visit_date": ("visit date, from the snapshot clinic tables", "date"),
    "steps_share_days_below_5k": ("share of valid days under 5,000 steps", "fraction"),
    "steps_share_days_above_10k": ("share of valid days with at least 10,000 steps", "fraction"),
    "steps_weekday_minus_weekend": ("median weekday steps minus median Friday-Saturday steps", "steps/day"),
    "resting_hr_slope_per_year": ("linear trend of daily resting heart rate across the window", "beats/min per year"),
    "spo2_share_days_below_90": ("share of valid days with any reading below 90%", "fraction"),
    "vo2max_nearest_visit": ("VO2max estimate on the valid day nearest the visit", "mL/(kg*min)"),
    "sleep_midpoint_hours_after_noon_median": ("median sleep midpoint, in hours after noon (14 = 02:00)", "h"),
    "sleep_midpoint_hours_sd": ("SD of the sleep midpoint across valid nights; lower is more regular", "h"),
    "sleep_efficiency_median": ("median sleep efficiency", "fraction"),
    "sleep_waso_minutes_median": ("median wake after sleep onset", "min"),
    "sleep_staged_share": ("share of valid nights with sleep stages", "fraction"),
    "sleep_staged_nights": ("valid nights with sleep stages", "nights"),
    "sleep_rem_share_median": ("median REM share of staged sleep, staged nights only", "fraction"),
    "sleep_deep_share_median": ("median deep share of staged sleep, staged nights only", "fraction"),
    "sleep_valid_nights": ("valid nights with asleep time in the window", "nights"),
    "sleep_in_bed_valid_nights": ("valid nights with time in bed in the window", "nights"),
    "wear_step_days_recorded": ("days in the window with any step record", "days"),
    "wear_step_days_valid": ("days in the window that pass the step-day rules", "days"),
    "wear_step_valid_share": ("valid step days as a share of recorded step days", "fraction"),
    "wear_first_day_from_visit": ("first recorded step day, relative to the visit", "days"),
    "wear_last_day_from_visit": ("last recorded step day, relative to the visit", "days"),
    "mindful_minutes_per_week": ("mindful-session minutes per week of recorded step days", "min/week"),
    "mindful_days": ("days with a mindful session in the window", "days"),
    "outlier_features": ("columns more than 4 robust SDs from that visit's cohort median (flag only)", "list"),
    "n_outlier_features": ("number of flagged columns", "count"),
}


def describe_column(column: str) -> tuple[str, str]:
    """Definition and unit of one output column."""

    if column in SPECIFIC_COLUMNS:
        return SPECIFIC_COLUMNS[column]
    if column.endswith("_valid_days"):
        return f"valid days of {column.removesuffix('_valid_days')} in the window", "days"
    for suffix, statistic in STATISTIC_SUFFIXES.items():
        prefix = column.removesuffix(suffix)
        if column.endswith(suffix) and prefix in FEATURE_UNITS:
            quantity, unit = FEATURE_UNITS[prefix]
            return f"{statistic} of {quantity} over valid days", unit
    return "", ""


def write_dictionary(columns: list[str], out: Path) -> None:
    rows = [{"column": c, "definition": d, "unit": u} for c in columns for d, u in [describe_column(c)]]
    pd.DataFrame(rows).to_csv(out / "feature_dictionary.csv", index=False)
    rules = {
        "window_days": WINDOW_DAYS, "max_out_of_range_share": MAX_OUT_OF_RANGE_SHARE,
        "min_observed_minutes_for_totals": MIN_OBSERVED_MINUTES_FOR_TOTALS, "min_hours_with_data": MIN_HOURS_WITH_DATA,
        "daily_total_bounds": DAILY_TOTAL_BOUNDS, "sleep_period_minutes_bounds": SLEEP_PERIOD_MINUTES_BOUNDS,
        "min_valid_days": MIN_VALID_DAYS, "min_valid_days_by_feature": MIN_VALID_DAYS_BY_FEATURE,
        "min_valid_nights": MIN_VALID_NIGHTS, "outlier_mad_threshold": OUTLIER_MAD_THRESHOLD,
        "weekend_weekdays": "Friday, Saturday", "in_bed_minutes_bounds": IN_BED_MINUTES_BOUNDS,
        "min_slope_span_days": MIN_SLOPE_SPAN_DAYS, "min_days_per_week_part": MIN_DAYS_PER_WEEK_PART,
        "flights_days": "valid StepCount days; no flights record on such a day counts as 0",
    }
    pd.Series(rules).astype(str).rename("value").rename_axis("rule").to_csv(out / "rules.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    parser.add_argument("--daily-run", type=Path, required=True, help="out= directory of compute_daily_statistics")
    parser.add_argument("--domain-run", type=Path, required=True, help="out= directory of compute_domain_metrics")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    visits = load_visit_dates(args.snapshot)
    features, audit, flag_counts = build_visit_features(args.daily_run, args.domain_run, visits)
    features.to_parquet(args.out / "visit_features.parquet", index=False)
    features.to_csv(args.out / "visit_features.csv", index=False)
    audit.to_csv(args.out / "outlier_audit.csv", index=False)
    flag_counts.to_csv(args.out / "outlier_flags.csv", index=False)
    write_dictionary(list(features.columns), args.out)
    print(features.groupby("research_stage").size().to_string())


if __name__ == "__main__":
    main()
