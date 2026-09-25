"""
Wearable Data Processing and Modeling project
Participant and cohort summaries, built on the daily tables of ``data_statistics``.
``data_statistics`` describes every participant-day. This module decides which days are usable and summarizes
participants and the cohort over them:
- Valid days. A day counts when it meets its feature's ``ValidDayRule``. The defaults are conventions, stated here
  and overridable per feature: heart rate needs at least 10 hours with data, a common wear-time criterion; glucose
  from a continuous monitor needs at least 70% of its expected readings, the international consensus criterion for
  CGM data; every other feature needs at least one record.
- Completeness. A day's records over those expected. It exists only for features whose sensor samples on a fixed
  period when worn continuously (declared in ``FIXED_CADENCE``), and only for participants whose data shows that
  cadence. Occasional finger-stick readings are never judged against a monitor's 288 readings a day.
- Adherence and gaps, per participant and feature: follow-up span, valid days, adherence (valid days over the span),
  the longest run of consecutive valid days, and the gaps between valid days.
- Participant summaries over valid days: for each feature's headline metrics (chosen by its measurement kind) and
  coverage metrics, the number of days, mean, standard deviation, median, 10th, 25th, 75th and 90th percentiles,
  minimum and maximum.
- A cohort Table: the distribution across participants of their medians, with valid days and adherence.
- Retention: how many participants still contribute valid days k days after their first.
``summarize`` accepts the result of ``compute_daily_statistics`` or the directory of a written run, which it reads one
participant at a time, so memory stays bounded at cohort scale.
"""


from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from wearable_project.exceptions import DataLoaderConfigurationError
from wearable_project.utils import data_statistics as ds


@dataclass(frozen=True)
class ValidDayRule:
    """
    When a participant-day counts as valid. ``min_hours_with_data`` applies to features with event times;
    ``min_completeness`` applies only where the participant's data has a fixed cadence (see ``FIXED_CADENCE``).
    """

    min_records: int = 1
    min_hours_with_data: int | None = None
    min_completeness: float | None = None

    def __post_init__(self) -> None:
        if self.min_records < 0:
            raise DataLoaderConfigurationError("min_records cannot be negative")
        if self.min_hours_with_data is not None and not 0 <= self.min_hours_with_data <= 24:
            raise DataLoaderConfigurationError("min_hours_with_data must be between 0 and 24")
        if self.min_completeness is not None and not 0 <= self.min_completeness <= 1:
            raise DataLoaderConfigurationError("min_completeness must be between 0 and 1")

    def describe(self) -> str:
        parts = [f"records >= {self.min_records}"]
        if self.min_hours_with_data is not None:
            parts.append(f"hours_with_data >= {self.min_hours_with_data}")
        if self.min_completeness is not None:
            parts.append(f"completeness >= {self.min_completeness:g} (fixed-cadence data only)")
        return " and ".join(parts)


@dataclass(frozen=True)
class FixedCadence:
    """A participant's data has a fixed cadence when its typical interval and regularity meet these bounds."""

    max_typical_gap_minutes: float
    min_regular_share: float


FIXED_CADENCE: dict[str, FixedCadence] = {
    # A continuous glucose monitor reads every 5 minutes (some models every 15); finger-stick readings are irregular.
    "BloodGlucose": FixedCadence(max_typical_gap_minutes=15.0, min_regular_share=0.9),
}
DEFAULT_RULE = ValidDayRule()
DEFAULT_RULES: dict[str, ValidDayRule] = {
    "HeartRate": ValidDayRule(min_hours_with_data=10),
    "BloodGlucose": ValidDayRule(min_completeness=0.70),
}
STATISTICS = ("n_days", "mean", "sd", "median", "p10", "p25", "p75", "p90", "min", "max")
_HEADLINE = {
    "extensive_total": ["value_sum"], "event_amount": ["value_sum"],
    "intensive_value": ["value_mean"], "ratio": ["value_mean"], "summary_statistic": ["value_mean"],
    "categorical_state": ["minutes_asleep_total", "minutes_inbed"], "duration": ["minutes"],
}
_INTERVAL_KINDS = {"extensive_total", "categorical_state", "duration", "signal"}
_FIXED_UNITS = {"records": "records", "hours_with_data": "h", "observed_minutes": "min", "minutes": "min",
                "minutes_asleep_total": "min", "minutes_inbed": "min"}


def expected_records_per_day(feature: str, typical_gap_minutes: float | None, regular_share: float | None) -> float | None:
    """Readings a full day holds for a fixed-cadence participant-feature; ``None`` where completeness does not apply."""

    cadence = FIXED_CADENCE.get(feature)
    if cadence is None or typical_gap_minutes is None or regular_share is None:
        return None
    if not (np.isfinite(typical_gap_minutes) and np.isfinite(regular_share)) or typical_gap_minutes <= 0:
        return None
    if typical_gap_minutes > cadence.max_typical_gap_minutes or regular_share < cadence.min_regular_share:
        return None
    return 1440.0 / typical_gap_minutes


def headline_metrics(feature: str, columns: Iterable[str]) -> list[str]:
    """The daily columns a feature is summarized by: value metrics chosen by its measurement kind, then coverage."""

    columns = list(columns)
    kind = ds.measurement_kind(feature)
    if kind in ("multivariate_point", "multivariate_summary", "signal"):
        values = [c for c in columns if c.endswith("_mean")]
    else:
        values = [c for c in _HEADLINE.get(kind, []) if c in columns]
    coverage = [c for c in ("records", "hours_with_data") if c in columns]
    if kind in _INTERVAL_KINDS and "observed_minutes" in columns:
        coverage.append("observed_minutes")
    return values + coverage


def _describe(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = int(len(values))
    if not n:
        return {"n_days": 0, **{s: np.nan for s in STATISTICS[1:]}}
    q = values.quantile([0.10, 0.25, 0.75, 0.90])
    return {"n_days": n, "mean": float(values.mean()), "sd": float(values.std(ddof=1)) if n > 1 else np.nan,
            "median": float(values.median()), "p10": float(q.loc[0.10]), "p25": float(q.loc[0.25]),
            "p75": float(q.loc[0.75]), "p90": float(q.loc[0.90]), "min": float(values.min()), "max": float(values.max())}


def _runs(days: list) -> tuple[int, list[int]]:
    """Longest run of consecutive days, and the lengths of the gaps (missing days) between the given days."""

    if not days:
        return 0, []
    ordinals = sorted({d.toordinal() for d in days})
    longest = current = 1
    gaps = []
    for previous, day in zip(ordinals, ordinals[1:]):
        if day - previous == 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
            gaps.append(day - previous - 1)
    return longest, gaps


def mark_valid_days(feature: str, daily: pd.DataFrame, rule: ValidDayRule, expected: float | None) -> pd.DataFrame:
    """One participant's daily rows for a feature, with ``completeness`` and ``valid`` added."""

    daily = daily.copy()
    daily["completeness"] = (daily["records"] / expected).clip(upper=1.0) if expected else np.nan
    valid = daily["records"] >= rule.min_records
    if rule.min_hours_with_data is not None and "hours_with_data" in daily:
        valid &= daily["hours_with_data"].fillna(0) >= rule.min_hours_with_data
    if rule.min_completeness is not None and expected:
        valid &= daily["completeness"] >= rule.min_completeness
    daily["valid"] = valid.astype(bool)
    return daily


def _unit_of(metric: str, daily: pd.DataFrame, units: dict[str, dict[str, str | None]]) -> str | None:
    if metric in _FIXED_UNITS:
        return _FIXED_UNITS[metric]
    if metric.startswith("value_"):
        found = daily["value_unit"].dropna().astype(str).unique() if "value_unit" in daily else []
        if len(found) == 1:
            return found[0]
        declared = units.get("value", {})
        return declared.get("established") or declared.get("declared_by_curation")
    column = metric[: -len("_mean")] if metric.endswith("_mean") else metric
    declared = units.get(column, {})
    return declared.get("established") or declared.get("declared_by_curation")


def summarize_participant(feature: str, daily: pd.DataFrame, cadence: dict[str, Any] | None,
                          rule: ValidDayRule) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Metric summaries and adherence for one participant's daily rows of one feature."""

    cadence = cadence or {}
    code = str(daily["RegistrationCode"].iloc[0])
    expected = expected_records_per_day(feature, cadence.get("typical_gap_minutes"), cadence.get("regular_share"))
    marked = mark_valid_days(feature, daily, rule, expected)
    valid = marked[marked["valid"]]
    days = list(marked["local_date"])
    valid_days = list(valid["local_date"])
    longest, gaps = _runs(valid_days)
    first, last = min(days), max(days)
    span = (last - first).days + 1
    adherence = {
        "RegistrationCode": code, "feature": feature, "rule": rule.describe(),
        "first_day": first, "last_day": last, "span_days": span, "days_with_data": len(days),
        "valid_days": len(valid_days), "adherence": len(valid_days) / span,
        "first_valid_day": min(valid_days) if valid_days else None, "last_valid_day": max(valid_days) if valid_days else None,
        "longest_valid_run": longest, "gaps": len(gaps), "longest_gap_days": max(gaps) if gaps else 0,
        "median_gap_days": float(np.median(gaps)) if gaps else np.nan,
        "typical_gap_minutes": cadence.get("typical_gap_minutes"), "regular_share": cadence.get("regular_share"),
        "expected_records_per_day": expected,
        "median_completeness": float(marked["completeness"].median()) if expected else np.nan,
    }
    units = ds.column_units(feature)
    rows = []
    for metric in headline_metrics(feature, daily.columns):
        rows.append({"RegistrationCode": code, "feature": feature, "metric": metric,
                     "unit": _unit_of(metric, valid if len(valid) else marked, units), **_describe(valid[metric])})
    return rows, adherence


@dataclass
class Summaries:
    """Participant metrics (long format), adherence per participant and feature, the cohort table, and retention."""

    participant_metrics: pd.DataFrame
    adherence: pd.DataFrame
    cohort: pd.DataFrame
    retention: pd.DataFrame
    rules: dict[str, str] = field(default_factory=dict)

    def wide(self, statistic: str = "median") -> pd.DataFrame:
        """One row per participant and one column per feature and metric, holding the chosen statistic."""

        table = self.participant_metrics.assign(column=lambda d: d["feature"] + "." + d["metric"])
        return table.pivot_table(index="RegistrationCode", columns="column", values=statistic, aggfunc="first")


def cohort_table(participant_metrics: pd.DataFrame, adherence: pd.DataFrame) -> pd.DataFrame:
    """Table: per feature and metric, the distribution across participants of their medians over valid days."""

    rows = []

    def row(feature, metric, unit, values):
        values = pd.to_numeric(values, errors="coerce").dropna().astype(float)
        if not len(values):
            return
        q = values.quantile([0.05, 0.25, 0.75, 0.95])
        rows.append({"feature": feature, "metric": metric, "unit": unit, "participants": int(len(values)),
                     "mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                     "p5": float(q.loc[0.05]), "p25": float(q.loc[0.25]), "median": float(values.median()),
                     "p75": float(q.loc[0.75]), "p95": float(q.loc[0.95]),
                     "min": float(values.min()), "max": float(values.max())})

    for (feature, metric), group in participant_metrics.groupby(["feature", "metric"], sort=False):
        units = group["unit"].dropna().unique()
        row(feature, metric, units[0] if len(units) == 1 else ("mixed" if len(units) else None), group["median"])
    for feature, group in adherence.groupby("feature", sort=False):
        with_valid = group[group["valid_days"] > 0]
        row(feature, "valid_days", "days", with_valid["valid_days"])
        row(feature, "adherence", "fraction", with_valid["adherence"])
    return pd.DataFrame(rows, columns=["feature", "metric", "unit", "participants", "mean", "sd", "p5", "p25",
                                       "median", "p75", "p95", "min", "max"])


def retention(adherence: pd.DataFrame) -> pd.DataFrame:
    """Per feature: participants still contributing valid days k days after their first valid day, k = 0, 1, ..."""

    rows = []
    for feature, group in adherence.dropna(subset=["first_valid_day"]).groupby("feature", sort=False):
        spans = np.array([(last - first).days for first, last in zip(group["first_valid_day"], group["last_valid_day"])])
        total = len(spans)
        for k in range(int(spans.max()) + 1):
            still = int((spans >= k).sum())
            rows.append({"feature": feature, "days_since_first_valid": k, "participants": still, "fraction": still / total})
    return pd.DataFrame(rows, columns=["feature", "days_since_first_valid", "participants", "fraction"])


def _check_rules(rules: dict[str, ValidDayRule] | None) -> None:
    """A rule for a feature that does not exist would silently never apply, so it is refused."""

    unknown = sorted(set(rules or {}) - set(ds.available_features()))
    if unknown:
        raise DataLoaderConfigurationError(f"rules name unknown feature(s): {', '.join(unknown)}")


def summarize(source: "ds.DailyStatistics | str | Path", *, rules: dict[str, ValidDayRule] | None = None,
              features: Iterable[str] | None = None, out: str | Path | None = None,
              chunksize: int = 200_000) -> Summaries:
    """
    Summaries of a daily-statistics run: the object ``compute_daily_statistics`` returned, or the directory it wrote.
    ``rules`` overrides the valid-day rule for the features it names. With ``out``, the tables are written there as CSV.
    """

    if isinstance(source, ds.DailyStatistics):
        cadence_table = source.participant_feature
        available = list(source.daily)

        def participants_of(feature):
            table = source.daily.get(feature)
            return [] if table is None else (g.reset_index(drop=True) for _, g in table.groupby("RegistrationCode", sort=True))
    else:
        directory = Path(source)
        if not (directory / "run.json").is_file():
            raise DataLoaderConfigurationError(f"{directory} is not a finished data_statistics run (no run.json)")
        cadence_table = ds.read_table(directory, "participant_feature")
        available = sorted(p.name.replace(".csv.gz", "") for p in (directory / "daily").glob("*.csv.gz"))

        def participants_of(feature):
            return ds.iter_daily(directory, feature, chunksize=chunksize)

    chosen = [f for f in available if features is None or f in set(features)]
    _check_rules(rules)
    chosen_rules = {f: (rules or {}).get(f, DEFAULT_RULES.get(f, DEFAULT_RULE)) for f in chosen}
    cadence = {(r["feature"], r["RegistrationCode"]): r for r in cadence_table.to_dict("records")}
    metric_rows, adherence_rows = [], []
    for feature in chosen:
        for daily in participants_of(feature):
            code = str(daily["RegistrationCode"].iloc[0])
            rows, adherence = summarize_participant(feature, daily, cadence.get((feature, code)), chosen_rules[feature])
            metric_rows.extend(rows)
            adherence_rows.append(adherence)
    participant_metrics = pd.DataFrame(metric_rows, columns=["RegistrationCode", "feature", "metric", "unit", *STATISTICS])
    adherence = pd.DataFrame(adherence_rows)
    result = Summaries(participant_metrics, adherence, cohort_table(participant_metrics, adherence),
                       retention(adherence) if len(adherence) else pd.DataFrame(),
                       {f: r.describe() for f, r in chosen_rules.items()})
    if out is not None:
        directory = Path(out).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        result.participant_metrics.to_csv(directory / "participant_metrics.csv", index=False)
        result.adherence.to_csv(directory / "adherence.csv", index=False)
        result.cohort.to_csv(directory / "cohort_table.csv", index=False)
        result.retention.to_csv(directory / "retention.csv", index=False)
        pd.Series(result.rules, name="rule").rename_axis("feature").to_csv(directory / "valid_day_rules.csv")
    return result


# ------------------------------------------------------------------------------------ context
# Friday and Saturday (Monday = 0): the Israeli weekend, the HPP cohort's, where the working week runs Sunday to
# Thursday. Pass weekend_days=(5, 6) for a Saturday-Sunday weekend.
WEEKEND_DAYS = (4, 5)
# The home zone is the participant's most common UTC offset, together with the offset 60 minutes from it that is
# next most common, when that offset covers at least this share of days: the zone's daylight-saving time.
DST_PARTNER_MIN_SHARE = 0.10


class _Source:
    """Uniform access to a daily-statistics run, in memory or written, one participant at a time."""

    def __init__(self, source, chunksize: int = 200_000) -> None:
        self.chunksize = chunksize
        if isinstance(source, ds.DailyStatistics):
            self.memory, self.dir = source, None
            self.features = list(source.daily)
        else:
            self.memory, self.dir = None, Path(source)
            if not (self.dir / "run.json").is_file():
                raise DataLoaderConfigurationError(f"{self.dir} is not a finished data_statistics run (no run.json)")
            self.features = sorted(p.name.replace(".csv.gz", "") for p in (self.dir / "daily").glob("*.csv.gz"))

    def participants_of(self, feature: str):
        if self.memory is not None:
            table = self.memory.daily.get(feature)
            if table is not None:
                for _, group in table.groupby("RegistrationCode", sort=True):
                    yield group.reset_index(drop=True)
        else:
            yield from ds.iter_daily(self.dir, feature, chunksize=self.chunksize)

    def table(self, name: str) -> pd.DataFrame:
        if self.memory is not None:
            return {"participant_feature": self.memory.participant_feature, "participant_days": self.memory.participant_days,
                    "hourly_profile": self.memory.hourly, "daily_provenance": self.memory.provenance,
                    "daily_curation": self.memory.curation}[name]
        path = ds._table_path(self.dir, name)
        return ds.read_table(self.dir, name) if path.is_file() else pd.DataFrame()

    def grouped(self, name: str):
        """A participant-level table one participant at a time."""

        if self.memory is not None:
            table = self.table(name)
            if len(table):
                for _, group in table.groupby("RegistrationCode", sort=True):
                    yield group.reset_index(drop=True)
        else:
            path = ds._table_path(self.dir, name)
            if path.is_file():
                yield from ds._iter_participants(path, self.chunksize)


def _spread(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if not len(values):
        return {"participants": 0, "median": np.nan, "p25": np.nan, "p75": np.nan}
    return {"participants": int(len(values)), "median": float(values.median()),
            "p25": float(values.quantile(0.25)), "p75": float(values.quantile(0.75))}


@dataclass
class TemporalPatterns:
    """Participant medians by day of week, weekday or weekend, and month, over valid days; and cohort spreads."""

    day_of_week: pd.DataFrame
    weekday_weekend: pd.DataFrame
    month_of_year: pd.DataFrame
    cohort_day_of_week: pd.DataFrame
    cohort_weekday_weekend: pd.DataFrame
    cohort_month_of_year: pd.DataFrame
    weekend_days: tuple[int, ...]


def temporal_patterns(source, *, rules: dict[str, ValidDayRule] | None = None, features: Iterable[str] | None = None,
                      weekend_days: tuple[int, ...] = WEEKEND_DAYS, chunksize: int = 200_000) -> TemporalPatterns:
    """
    Each participant's median of each headline metric by day of week (Monday = 0), by weekday or weekend, and by
    month, over valid days only; then, across participants, the median and interquartile range of those medians.
    """

    if not weekend_days or any(d not in range(7) for d in weekend_days):
        raise DataLoaderConfigurationError("weekend_days must be weekday numbers from 0 (Monday) to 6 (Sunday)")
    _check_rules(rules)
    src = _Source(source, chunksize)
    cadence = {(r["feature"], r["RegistrationCode"]): r for r in src.table("participant_feature").to_dict("records")}
    chosen = [f for f in src.features if features is None or f in set(features)]
    dow, month, split = [], [], []
    for feature in chosen:
        rule = (rules or {}).get(feature, DEFAULT_RULES.get(feature, DEFAULT_RULE))
        for daily in src.participants_of(feature):
            code = str(daily["RegistrationCode"].iloc[0])
            c = cadence.get((feature, code), {})
            valid = mark_valid_days(feature, daily, rule, expected_records_per_day(
                feature, c.get("typical_gap_minutes"), c.get("regular_share")))
            valid = valid[valid["valid"]]
            if valid.empty:
                continue
            dates = pd.to_datetime(valid["local_date"])
            weekday, months = dates.dt.weekday.to_numpy(), dates.dt.month.to_numpy()
            weekend = np.isin(weekday, weekend_days)
            for metric in headline_metrics(feature, daily.columns):
                values = pd.to_numeric(valid[metric], errors="coerce").to_numpy(dtype=float)
                for key, labels, rows in (("weekday", weekday, dow), ("month", months, month)):
                    for label in np.unique(labels):
                        chosen_values = values[(labels == label) & ~np.isnan(values)]
                        if len(chosen_values):
                            rows.append({"RegistrationCode": code, "feature": feature, "metric": metric, key: int(label),
                                         "n_days": len(chosen_values), "median": float(np.median(chosen_values))})
                work, rest = values[~weekend & ~np.isnan(values)], values[weekend & ~np.isnan(values)]
                split.append({"RegistrationCode": code, "feature": feature, "metric": metric,
                              "weekday_days": len(work), "weekday_median": float(np.median(work)) if len(work) else np.nan,
                              "weekend_days": len(rest), "weekend_median": float(np.median(rest)) if len(rest) else np.nan})
    dow_t = pd.DataFrame(dow, columns=["RegistrationCode", "feature", "metric", "weekday", "n_days", "median"])
    month_t = pd.DataFrame(month, columns=["RegistrationCode", "feature", "metric", "month", "n_days", "median"])
    split_t = pd.DataFrame(split, columns=["RegistrationCode", "feature", "metric", "weekday_days", "weekday_median",
                                           "weekend_days", "weekend_median"])
    split_t["weekend_minus_weekday"] = split_t["weekend_median"] - split_t["weekday_median"]

    def cohort(table, by, value):
        rows = [{**dict(zip(by, key if isinstance(key, tuple) else (key,))), **_spread(g[value])}
                for key, g in table.groupby(by, sort=True)]
        return pd.DataFrame(rows, columns=[*by, "participants", "median", "p25", "p75"])

    return TemporalPatterns(dow_t, split_t, month_t, cohort(dow_t, ["feature", "metric", "weekday"], "median"),
                            cohort(split_t, ["feature", "metric"], "weekend_minus_weekday"),
                            cohort(month_t, ["feature", "metric", "month"], "median"), tuple(weekend_days))


def hour_of_day(source) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ``(participants, cohort)`` by local hour of day. Per participant: records per day with data at that hour, the
    average amount per day at that hour for totals and event amounts, and the mean value for levels. Per feature
    and hour: the median and interquartile range of those across participants.
    """

    src = _Source(source)
    hourly = src.table("hourly_profile")
    days = src.table("participant_feature")[["RegistrationCode", "feature", "days_with_data"]]
    if hourly.empty:
        return pd.DataFrame(), pd.DataFrame()
    table = hourly.merge(days, on=["RegistrationCode", "feature"], how="left")
    table["records_per_day"] = table["records"] / table["days_with_data"]
    table["value_per_day"] = table["value_sum"] / table["days_with_data"]
    participants = table[["RegistrationCode", "feature", "hour", "records_per_day", "value_per_day", "value_mean"]]
    rows = []
    for (feature, hour), group in participants.groupby(["feature", "hour"], sort=True):
        row = {"feature": feature, "hour": int(hour)}
        for column in ("records_per_day", "value_per_day", "value_mean"):
            spread = _spread(group[column])
            row.update({f"{column}_{k}": v for k, v in spread.items() if k != "participants"})
            row[f"{column}_participants"] = spread["participants"]
        rows.append(row)
    return participants.reset_index(drop=True), pd.DataFrame(rows)


def time_zones(source) -> pd.DataFrame:
    """
    Per participant: the home zone (``DST_PARTNER_MIN_SHARE``), the days with any offset outside it, and the trips,
    runs of consecutive such days. A day on which the clocks change carries both home offsets, and is not away.
    """

    rows = []
    for days in _Source(source).grouped("participant_days"):
        code = str(days["RegistrationCode"].iloc[0])
        per_day = {}
        for day, text in zip(days["local_date"], days["utc_offsets"]):
            offsets = {int(float(x)) for x in str(text).split(";") if x not in ("", "nan", "None")}
            if offsets:
                per_day[day] = offsets
        if not per_day:
            continue
        counts: dict[int, int] = {}
        for offsets in per_day.values():
            for offset in offsets:
                counts[offset] = counts.get(offset, 0) + 1
        home = max(sorted(counts), key=lambda o: counts[o])
        partners = [o for o in (home - 60, home + 60) if counts.get(o, 0) >= DST_PARTNER_MIN_SHARE * len(per_day)]
        home_set = {home} | ({max(partners, key=lambda o: counts[o])} if partners else set())
        away = sorted(d for d, offsets in per_day.items() if offsets - home_set)
        runs, current = [], 1
        for previous, day in zip(away, away[1:]):
            if (day - previous).days == 1:
                current += 1
            else:
                runs.append(current)
                current = 1
        if away:
            runs.append(current)
        distances = [abs(o - home) for d in away for o in per_day[d] - home_set]
        rows.append({"RegistrationCode": code, "days_with_offsets": len(per_day),
                     "home_offsets": ";".join(str(o) for o in sorted(home_set)),
                     "distinct_offsets": len(counts), "days_away": len(away), "trips": len(runs),
                     "longest_trip_days": max(runs) if runs else 0,
                     "max_offset_difference_hours": max(distances) / 60 if distances else 0.0})
    return pd.DataFrame(rows, columns=["RegistrationCode", "days_with_offsets", "home_offsets", "distinct_offsets",
                                       "days_away", "trips", "longest_trip_days", "max_offset_difference_hours"])


def day_overlap(source) -> pd.DataFrame:
    """
    For every pair of features, the participant-days holding both, those holding either, and their ratio (the
    Jaccard index): how often two modalities can be analyzed on the same day. The diagonal gives each feature's days.
    """

    single: dict[str, int] = {}
    pairs: dict[tuple[str, str], int] = {}
    for days in _Source(source).grouped("participant_days"):
        for text in days["features"]:
            present = sorted(str(text).split(";"))
            for f in present:
                single[f] = single.get(f, 0) + 1
            for i, a in enumerate(present):
                for b in present[i + 1:]:
                    pairs[(a, b)] = pairs.get((a, b), 0) + 1
    rows = []
    names = sorted(single)
    for i, a in enumerate(names):
        for b in names[i:]:
            both = single[a] if a == b else pairs.get((a, b), 0)
            either = single[a] + single[b] - both if a != b else single[a]
            rows.append({"feature_a": a, "feature_b": b, "days_both": both, "days_a": single[a], "days_b": single[b],
                         "jaccard": both / either if either else np.nan})
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "days_both", "days_a", "days_b", "jaccard"])


def days_with(source, features: Iterable[str]) -> pd.DataFrame:
    """Per participant, the days on which every one of ``features`` has data, with the first and last such day."""

    wanted = {features} if isinstance(features, str) else set(features)
    if not wanted:
        raise DataLoaderConfigurationError("days_with needs at least one feature")
    unknown = sorted(wanted - set(ds.available_features()))
    if unknown:
        raise DataLoaderConfigurationError(f"unknown feature(s): {', '.join(unknown)}")
    rows = []
    for days in _Source(source).grouped("participant_days"):
        chosen = [d for d, text in zip(days["local_date"], days["features"]) if wanted <= set(str(text).split(";"))]
        if chosen:
            rows.append({"RegistrationCode": str(days["RegistrationCode"].iloc[0]), "days": len(chosen),
                         "first_day": min(chosen), "last_day": max(chosen)})
    return pd.DataFrame(rows, columns=["RegistrationCode", "days", "first_day", "last_day"])


@dataclass
class QualityReport:
    """Per feature: coverage, provenance and plausibility totals; acquisition methods; curation statuses and flags."""

    features: pd.DataFrame
    provenance: pd.DataFrame
    curation: pd.DataFrame


def quality_report(source) -> QualityReport:
    """
    Totals per feature: records, participant-days, records flagged user-entered, observed and redundant minutes
    (coverage duplicated across devices), and values below or above the declared plausible range, with the days on
    which plausibility could be assessed; the share of records by acquisition method; and, in the curated phase, the
    records per curation status and per flag. Counts only, never names.
    """

    src = _Source(source)
    rows = []
    for feature in src.features:
        totals = {"feature": feature, "participants": 0, "participant_days": 0, "records": 0,
                  "records_user_entered": 0.0, "observed_minutes": 0.0, "redundant_minutes": 0.0,
                  "values_below_range": 0.0, "values_above_range": 0.0, "days_assessed": 0}
        seen = {"records_user_entered": False, "observed_minutes": False, "redundant_minutes": False,
                "values_below_range": False}
        for daily in src.participants_of(feature):
            totals["participants"] += 1
            totals["participant_days"] += len(daily)
            totals["records"] += int(daily["records"].sum())
            for column in ("records_user_entered", "observed_minutes", "redundant_minutes", "values_below_range",
                           "values_above_range"):
                if column in daily and daily[column].notna().any():
                    totals[column] += float(daily[column].sum())
                    seen[column if column != "values_above_range" else "values_below_range"] = True
            if "values_below_range" in daily:
                totals["days_assessed"] += int(daily["values_below_range"].notna().sum())
        for column, present in seen.items():
            if not present:
                totals[column] = np.nan
                if column == "values_below_range":
                    totals["values_above_range"] = np.nan
        totals["redundant_share"] = (totals["redundant_minutes"] / totals["observed_minutes"]
                                     if totals["observed_minutes"] and not np.isnan(totals["redundant_minutes"]) else np.nan)
        rows.append(totals)
    features = pd.DataFrame(rows)
    provenance = src.table("daily_provenance")
    if len(provenance):
        provenance = provenance.groupby(["feature", "acquisition_method"], sort=True)[["records", "records_user_entered"]].sum().reset_index()
        provenance["share"] = provenance["records"] / provenance.groupby("feature")["records"].transform("sum")
    curation = src.table("daily_curation")
    if len(curation):
        curation = curation.groupby(["feature", "kind", "name"], sort=True)["records"].sum().reset_index()
        record_totals = features.set_index("feature")["records"]
        curation["share_of_records"] = curation["records"] / curation["feature"].map(record_totals)
    return QualityReport(features, provenance, curation)
