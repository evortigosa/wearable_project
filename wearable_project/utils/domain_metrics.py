"""
Wearable Data Processing and Modeling project
Domain metrics: sleep by night, and continuous-glucose-monitor (CGM) metrics, built on the DataLoaders and on
``data_statistics``. Every threshold and definition is a named constant below, recorded in each run's ``run.json``.
Sleep nights
    A night is the noon-to-noon window starting on its ``night_date``, in the participant's local time; a record that
    crosses noon is split at it. Asleep records (asleep unspecified, core, deep, REM) separated by at most
    ``EPISODE_GAP_MINUTES`` form one sleep episode; the episode with the most asleep time is the night's main sleep,
    and the others count as naps. On the samples, gaps between asleep records are at most 120 minutes within a night
    and over 360 minutes between nights. From the main sleep: onset and offset, the sleep period between them, total
    sleep time (the union of asleep records, so overlapping devices count once), wake after sleep onset (the period
    minus sleep time), time in bed (the period together with every in-bed record overlapping it), efficiency (sleep
    time over time in bed, or over the period where no in-bed record exists), and minutes and shares per stage where
    stages were recorded. A night holding only in-bed records has its in-bed time but no sleep metrics: its sleep was
    not measured, which is not the same as no sleep. Clock times are also given in hours after the night's noon, so
    they can be averaged across nights without wrapping at midnight.
CGM metrics
    Only for participants whose glucose data has a continuous monitor's fixed cadence (``data_summaries``), so
    finger-stick readings are never mixed in. Readings are classified in mg/dL, the unit of the consensus ranges,
    converting stored mmol/L with HealthKit's factor ``MGDL_PER_MMOL`` (the molar mass of glucose, 180.15588 g/mol);
    on the samples every stored reading converts to a whole mg/dL value within 1e-10. Rounded mmol/L cutoffs such as
    3.9 would misclassify the many readings exactly at 70 mg/dL (3.8855 mmol/L). Per day: readings, completeness,
    mean, SD, coefficient of variation, and the percentage of readings in each consensus range. Per participant,
    pooled over valid days (at least 70% of expected readings): the same, plus the Glucose Management Indicator.
"""


from __future__ import annotations
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from wearable_project.exceptions import DataLoaderConfigurationError
from wearable_project.utils import data_statistics as ds
from wearable_project.utils import data_summaries as sm


# ------------------------------------------------------------------------------------------ declarations
NIGHT_START_HOUR = 12
EPISODE_GAP_MINUTES = 120
ASLEEP_STATES = ds.ASLEEP_STATES
STAGE_STATES = ("CORE", "DEEP", "REM")
MGDL_PER_MMOL = 18.015588
CGM_MIN_DAY_COMPLETENESS = sm.DEFAULT_RULES["BloodGlucose"].min_completeness
CGM_SUFFICIENT_DAYS = 14
# Consensus ranges in mg/dL, each a half-open interval, so every reading falls in exactly one.
GLUCOSE_RANGES = (
    ("very_low_percent", None, 54.0),      # below 54 mg/dL: level 2 hypoglycemia
    ("low_percent", 54.0, 70.0),           # 54 to 69 mg/dL: level 1 hypoglycemia
    ("in_range_percent", 70.0, 180.0),     # 70 to 180 mg/dL: target range (upper bound inclusive, see below)
    ("high_percent", 180.0, 250.0),        # 181 to 250 mg/dL: level 1 hyperglycaemia
    ("very_high_percent", 250.0, None),    # above 250 mg/dL: level 2 hyperglycaemia
)


@dataclass(frozen=True)
class NightRule:
    """When a night counts as valid: its sleep was measured, and the main sleep lasted at least this long."""

    min_asleep_minutes: float = 180.0

    def __post_init__(self) -> None:
        if not 0 <= self.min_asleep_minutes <= 1440:
            raise DataLoaderConfigurationError("min_asleep_minutes must be between 0 and 1440")

    def describe(self) -> str:
        return f"asleep recorded and asleep_minutes >= {self.min_asleep_minutes:g}"


def definitions() -> dict[str, Any]:
    """The declared constants, as recorded in ``run.json``."""

    return {"night_start_hour": NIGHT_START_HOUR, "episode_gap_minutes": EPISODE_GAP_MINUTES,
            "asleep_states": sorted(ASLEEP_STATES), "stage_states": list(STAGE_STATES),
            "mgdl_per_mmol": MGDL_PER_MMOL, "cgm_min_day_completeness": CGM_MIN_DAY_COMPLETENESS,
            "cgm_sufficient_days": CGM_SUFFICIENT_DAYS,
            "glucose_ranges_mgdl": [[n, lo, hi] for n, lo, hi in GLUCOSE_RANGES],
            "glucose_range_rule": "low <= mg/dL < high, except in_range, which includes 180; fixed cadence per "
                                  "data_summaries.FIXED_CADENCE"}


# ------------------------------------------------------------------------------------------------- sleep
NIGHT_COLUMNS = [
    "RegistrationCode", "night_date", "records", "records_without_offset", "sources", "asleep_recorded", "staged",
    "in_bed_recorded", "onset_local", "offset_local", "midpoint_local", "onset_hours_after_noon",
    "offset_hours_after_noon", "midpoint_hours_after_noon", "sleep_period_minutes", "asleep_minutes",
    "waso_minutes", "awake_recorded_minutes", "in_bed_minutes", "efficiency", "efficiency_basis", "core_minutes",
    "deep_minutes", "rem_minutes", "asleep_unspecified_minutes", "staged_minutes", "staging_sources", "core_share",
    "deep_share", "rem_share", "episodes", "nap_minutes",
]


def _night_pieces(df: pd.DataFrame) -> pd.DataFrame:
    """Records split at local noon: one piece per record and night window, with its state."""

    shift = pd.Timedelta(NIGHT_START_HOUR, unit="h")
    pieces = ds._pieces(df["local_start"] - shift, df["local_end"] - shift)
    # Timedeltas are built with an explicit unit: under pandas 2 with numpy 2.4 or later, keyword construction such as
    # pd.Timedelta(hours=12) uses a "generic" numpy unit that numpy has deprecated and will make an error.
    dtype = df["local_start"].dtype
    pieces["start"] = pd.to_datetime(pieces["start"]).astype(dtype) + shift
    pieces["end"] = pd.to_datetime(pieces["end"]).astype(dtype) + shift
    # The same dtype as night dates computed from the records: pandas 2.1 cannot concatenate date columns of
    # different resolutions.
    pieces["night"] = pd.to_datetime(pieces["day"]).astype(df["local_start"].dtype)
    pieces["RegistrationCode"] = df["RegistrationCode"].to_numpy()[pieces["row"]]
    pieces["state"] = df["value"].astype(str).to_numpy()[pieces["row"]]
    return pieces


def sleep_nights(data) -> pd.DataFrame:
    """One row per participant and night from a Sleep result (one or several participants)."""

    if data.df.empty:
        return pd.DataFrame(columns=NIGHT_COLUMNS)
    df, _ = ds._local_frame(data)
    keys = ["RegistrationCode", "night"]
    shift = pd.Timedelta(NIGHT_START_HOUR, unit="h")
    pieces = _night_pieces(df)
    counts = df.assign(night=(df["local_start"] - shift).dt.normalize()).groupby(keys).agg(
        records=("local_start", "size"), records_without_offset=("without_offset", "sum"))
    index = pd.MultiIndex.from_frame(
        pd.concat([counts.reset_index()[keys], pieces[keys]]).drop_duplicates().sort_values(keys))
    table = pd.DataFrame(index=index)
    table["records"] = counts["records"].reindex(index).fillna(0).astype(int)
    table["records_without_offset"] = counts["records_without_offset"].reindex(index).fillna(0).astype(int)
    if "source_id" in df:
        pieces["source"] = df["source_id"].astype(str).to_numpy()[pieces["row"]]
        table["sources"] = pieces.groupby(keys)["source"].nunique().reindex(index).fillna(0).astype(int)
    else:
        table["sources"] = np.nan

    positive = pieces[pieces["end"] > pieces["start"]]
    in_bed_all = positive[positive["state"] == "INBED"]
    in_bed_whole_night = ds._union_minutes(in_bed_all, keys) if not in_bed_all.empty else pd.Series(dtype=float)
    asleep = positive[positive["state"].isin(ASLEEP_STATES)].sort_values(keys + ["start"])
    metrics = pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=keys))
    if not asleep.empty:
        running = asleep.groupby(keys, sort=False)["end"].cummax()
        previous = running.groupby([asleep[k] for k in keys], sort=False).shift()
        new = previous.isna() | ((asleep["start"] - previous) > pd.Timedelta(EPISODE_GAP_MINUTES, unit="m"))
        asleep = asleep.assign(episode=new.cumsum().to_numpy())
        per_episode = ds._union_minutes(asleep, keys + ["episode"])
        main = {index[2] for index in per_episode.groupby(level=[0, 1]).idxmax()}  # ties: the earliest episode
        main_pieces = asleep[asleep["episode"].isin(main)]
        bounds = main_pieces.groupby(keys).agg(onset=("start", "min"), offset=("end", "max"))
        idx = bounds.index

        def union(frame: pd.DataFrame) -> pd.Series:
            return (ds._union_minutes(frame, keys).reindex(idx) if not frame.empty else pd.Series(0.0, index=idx)).fillna(0.0)

        # merge, not join: joining an empty frame on columns would also make them index levels
        periods = bounds.reset_index()

        def within_period(frame: pd.DataFrame) -> pd.DataFrame:
            joined = frame.merge(periods, on=keys, how="inner")
            joined["start"] = joined[["start", "onset"]].max(axis=1)
            joined["end"] = joined[["end", "offset"]].min(axis=1)
            return joined[joined["end"] > joined["start"]]

        asleep_minutes = per_episode[per_episode.index.get_level_values(2).isin(main)].droplevel(2).reindex(idx)
        period = (bounds["offset"] - bounds["onset"]).dt.total_seconds() / 60.0
        in_bed = in_bed_all.merge(periods, on=keys, how="inner")
        in_bed = in_bed[(in_bed["start"] < in_bed["offset"]) & (in_bed["end"] > in_bed["onset"])]
        overlapping = (in_bed.groupby(keys).size().reindex(idx).fillna(0) > 0)
        bed = pd.concat([bounds.reset_index().rename(columns={"onset": "start", "offset": "end"})[keys + ["start", "end"]],
                         in_bed[keys + ["start", "end"]]], ignore_index=True)
        bed_minutes = ds._union_minutes(bed, keys).reindex(idx)
        # Stages come from one source per night, the one that staged the most sleep: two devices staging the same
        # minutes differently would otherwise count those minutes under two stages. Total sleep still unions all.
        stage_pieces = main_pieces[main_pieces["state"].isin(STAGE_STATES)]
        if "source" in stage_pieces and not stage_pieces.empty:
            per_source = ds._union_minutes(stage_pieces, keys + ["source"])
            chosen = set(per_source.groupby(level=[0, 1]).idxmax())  # ties: the first source identifier
            staging_sources = per_source.groupby(level=[0, 1]).size().reindex(idx).fillna(0).astype(int)
            stage_pieces = stage_pieces[[key in chosen for key in zip(*(stage_pieces[k] for k in keys + ["source"]))]]
        else:
            staging_sources = (stage_pieces.groupby(keys).size().reindex(idx).fillna(0) > 0).astype(int)
        staged = staging_sources > 0
        staged_minutes = union(stage_pieces).where(staged)
        stages = {column: union(stage_pieces[stage_pieces["state"] == state]) for state, column in (
            ("CORE", "core_minutes"), ("DEEP", "deep_minutes"), ("REM", "rem_minutes"))}
        stages["asleep_unspecified_minutes"] = union(main_pieces[main_pieces["state"] == "ASLEEP"])
        denominator = bed_minutes.where(overlapping, period)
        metrics = pd.DataFrame({
            "onset_local": bounds["onset"], "offset_local": bounds["offset"],
            "sleep_period_minutes": period, "asleep_minutes": asleep_minutes,
            "waso_minutes": period - asleep_minutes,
            "awake_recorded_minutes": union(within_period(positive[positive["state"] == "AWAKE"])),
            "bed_minutes": bed_minutes, "in_bed_overlapping": overlapping,
            "efficiency": asleep_minutes / denominator.where(denominator > 0),
            "efficiency_basis": np.where(overlapping, "in_bed", "sleep_period"),
            **stages, "staged_minutes": staged_minutes, "staging_sources": staging_sources,
            **{share: (stages[stage] / staged_minutes).where(staged) for stage, share in (
                ("core_minutes", "core_share"), ("deep_minutes", "deep_share"), ("rem_minutes", "rem_share"))},
            "staged": staged,
            "episodes": per_episode.groupby(level=[0, 1]).size().reindex(idx),
            "nap_minutes": (ds._union_minutes(asleep, keys).reindex(idx) - asleep_minutes).clip(lower=0),
        }, index=idx)

    table = table.join(metrics, how="left")
    for column in NIGHT_COLUMNS:
        if column not in table and column not in ("RegistrationCode", "night_date"):
            table[column] = np.nan
    asleep_recorded = table["asleep_minutes"].notna()
    table["asleep_recorded"] = asleep_recorded
    whole_night = in_bed_whole_night.reindex(table.index)
    bed = table["bed_minutes"] if "bed_minutes" in table else pd.Series(np.nan, index=table.index)
    table["in_bed_minutes"] = bed.where(asleep_recorded, whole_night)  # nights without sleep keep their in-bed time
    overlapping = table["in_bed_overlapping"] if "in_bed_overlapping" in table else pd.Series(False, index=table.index)
    # eq(True) gives booleans directly, without relying on pandas' deprecated downcasting of object columns.
    table["in_bed_recorded"] = overlapping.eq(True).where(asleep_recorded, whole_night.notna())
    table["staged"] = table["staged"].eq(True) if "staged" in table else False
    table["episodes"] = table["episodes"].fillna(0).astype(int)
    table["staging_sources"] = table["staging_sources"].fillna(0).astype(int)
    table = table.reset_index().rename(columns={"night": "night_date"})
    noon = pd.to_datetime(table["night_date"]) + shift
    # The records' resolution, also for a participant whose nights are all in-bed only (whose times are all missing).
    table["onset_local"] = pd.to_datetime(table["onset_local"]).astype(df["local_start"].dtype)
    table["offset_local"] = pd.to_datetime(table["offset_local"]).astype(df["local_start"].dtype)
    table["midpoint_local"] = table["onset_local"] + (table["offset_local"] - table["onset_local"]) / 2
    for name in ("onset", "offset", "midpoint"):
        table[f"{name}_hours_after_noon"] = (table[f"{name}_local"] - noon).dt.total_seconds() / 3600.0
    table["night_date"] = pd.to_datetime(table["night_date"]).dt.date
    for column in ("asleep_recorded", "staged", "in_bed_recorded"):
        table[column] = table[column].astype(bool)
    # One dtype per column whatever the participant: text here even where every night lacks it; missing stays NaN,
    # as a written run reads it back.
    table["efficiency_basis"] = table["efficiency_basis"].astype(object)
    return table[NIGHT_COLUMNS]


# --------------------------------------------------------------------------------------------------- CGM
CGM_DAY_COLUMNS = ["RegistrationCode", "local_date", "readings", "completeness", "valid", "mean_mgdl", "mean_mmol",
                   "sd_mgdl", "cv_percent", *[n for n, _, _ in GLUCOSE_RANGES]]
CGM_PERIOD_COLUMNS = ["RegistrationCode", "cgm", "typical_gap_minutes", "expected_readings_per_day",
                      "unresolved_readings", "first_day",
                      "last_day", "span_days", "valid_days", "readings", "active_percent", "span_active_percent",
                      "sufficient", "mean_mgdl", "mean_mmol", "gmi_percent", "sd_mgdl", "cv_percent",
                      *[n for n, _, _ in GLUCOSE_RANGES]]


def _period_frame(period: dict[str, Any]) -> pd.DataFrame:
    """The one-row period table, typed as a written run reads it back: numbers as floats, missing values as NaN."""

    frame = pd.DataFrame([period], columns=CGM_PERIOD_COLUMNS)
    for column in CGM_PERIOD_COLUMNS:
        if column not in ("RegistrationCode", "cgm", "sufficient", "first_day", "last_day"):
            frame[column] = pd.to_numeric(frame[column]).astype(float)
    for column in ("first_day", "last_day"):  # dates or None, one dtype whether or not the participant has CGM days
        value = frame.at[0, column]
        frame[column] = pd.Series([pd.Timestamp(value).date() if pd.notna(value) else pd.NaT], dtype=object)
    frame["cgm"] = frame["cgm"].astype(bool)
    frame["sufficient"] = frame["sufficient"].astype(bool)
    return frame


def glucose_mgdl(mmol: pd.Series | np.ndarray) -> np.ndarray:
    """Stored mmol/L as mg/dL, rounded to 1e-6 so a whole-number reading compares exactly with its cutoff."""

    return np.round(np.asarray(mmol, dtype=float) * MGDL_PER_MMOL, 6)


def glucose_range(mgdl: np.ndarray) -> np.ndarray:
    """The consensus range of each reading, by name; the target range includes both 70 and 180 mg/dL."""

    out = np.empty(len(mgdl), dtype=object)
    out[:] = "in_range_percent"
    out[mgdl < 54.0] = "very_low_percent"
    out[(mgdl >= 54.0) & (mgdl < 70.0)] = "low_percent"
    out[(mgdl > 180.0) & (mgdl <= 250.0)] = "high_percent"
    out[mgdl > 250.0] = "very_high_percent"
    return out


def _describe_glucose(mgdl: np.ndarray) -> dict[str, float]:
    n = len(mgdl)
    if not n:
        return {"mean_mgdl": np.nan, "mean_mmol": np.nan, "sd_mgdl": np.nan, "cv_percent": np.nan,
                **{name: np.nan for name, _, _ in GLUCOSE_RANGES}}
    mean = float(np.mean(mgdl))
    sd = float(np.std(mgdl, ddof=1)) if n > 1 else np.nan
    ranges = glucose_range(mgdl)
    return {"mean_mgdl": mean, "mean_mmol": mean / MGDL_PER_MMOL, "sd_mgdl": sd, "cv_percent": 100.0 * sd / mean,
            **{name: 100.0 * float(np.mean(ranges == name)) for name, _, _ in GLUCOSE_RANGES}}


def cgm_metrics(data) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ``(days, period)`` for one participant's BloodGlucose result. ``period`` always has one row; its ``cgm`` column
    says whether the data has a continuous monitor's cadence, and only then are the metrics filled.
    """

    code = str(data.df.index.get_level_values("RegistrationCode")[0]) if not data.df.empty else None
    cadence = ds.sampling_cadence(data)
    expected = sm.expected_records_per_day("BloodGlucose", cadence["typical_gap_minutes"], cadence["regular_share"])
    period = {c: np.nan for c in CGM_PERIOD_COLUMNS}
    period.update({"RegistrationCode": code, "cgm": expected is not None,
                   "typical_gap_minutes": cadence["typical_gap_minutes"], "expected_readings_per_day": expected,
                   "sufficient": False})
    if expected is None:
        return pd.DataFrame(columns=CGM_DAY_COLUMNS), _period_frame(period)
    harmonized = data.with_harmonized_values().df.reset_index()
    units = set(harmonized["harmonized_unit"].dropna().astype(str))
    if units - {"mmol/L"}:
        raise DataLoaderConfigurationError(f"CGM metrics need glucose in mmol/L; found {sorted(units)}")
    frame, _ = ds._local_frame(data)
    frame["mgdl"] = glucose_mgdl(harmonized["harmonized_value"])
    period["unresolved_readings"] = int((~np.isfinite(frame["mgdl"])).sum())  # excluded: unit not established
    frame = frame[np.isfinite(frame["mgdl"])]
    frame["day"] = frame["local_start"].dt.date
    rows = []
    for day, group in frame.groupby("day", sort=True):
        completeness = min(len(group) / expected, 1.0)
        rows.append({"RegistrationCode": code, "local_date": day, "readings": len(group), "completeness": completeness,
                     "valid": completeness >= CGM_MIN_DAY_COMPLETENESS, **_describe_glucose(group["mgdl"].to_numpy())})
    days = pd.DataFrame(rows, columns=CGM_DAY_COLUMNS)
    period.update({"valid_days": 0, "readings": 0})  # a monitor's cadence, but perhaps no reading with a usable unit
    if len(days):
        valid = days[days["valid"]]
        span = (days["local_date"].max() - days["local_date"].min()).days + 1
        pooled = frame[frame["day"].isin(set(valid["local_date"]))]["mgdl"].to_numpy()
        described = _describe_glucose(pooled)
        period.update({"first_day": days["local_date"].min(), "last_day": days["local_date"].max(), "span_days": span,
                       "valid_days": int(len(valid)), "readings": int(len(pooled)),
                       "active_percent": 100.0 * len(pooled) / (expected * len(valid)) if len(valid) else np.nan,
                       "span_active_percent": 100.0 * min(len(frame) / (expected * span), 1.0),
                       "sufficient": bool(len(valid) >= CGM_SUFFICIENT_DAYS), **described,
                       "gmi_percent": 3.31 + 0.02392 * described["mean_mgdl"] if len(pooled) else np.nan})
    return days, _period_frame(period)


# ------------------------------------------------------------------------------------------ cohort runs
DOMAIN_TABLES = ("sleep_nights", "cgm_days", "cgm_periods")


def read_domain_table(out: str | Path, name: str) -> pd.DataFrame:
    """A table of a written domain run, typed as the in-memory result: dates as dates, clock times as timestamps."""

    frame = ds.read_table(out, name)
    if "night_date" in frame:
        frame["night_date"] = pd.to_datetime(frame["night_date"]).dt.date
    for column in ("onset_local", "offset_local", "midpoint_local"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], format="ISO8601")  # seconds with or without fractions
    return frame


@dataclass
class DomainMetrics:
    """Sleep nights, CGM days and CGM periods. A written run's tables are read back with ``read_domain_table``."""

    sleep_nights: pd.DataFrame
    cgm_days: pd.DataFrame
    cgm_periods: pd.DataFrame
    errors: list[dict[str, str]]
    run: dict[str, Any]


def _domain_task(args) -> dict[str, Any]:
    code, phase, root, state_validation, default_inclusion_only = args
    out = {"code": code, "tables": {n: pd.DataFrame() for n in DOMAIN_TABLES}, "errors": []}
    extra = {"default_inclusion_only": True} if default_inclusion_only else {}
    for feature in ("Sleep", "BloodGlucose"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loader = ds._loader(feature, phase, root, state_validation=state_validation)
                # Sleep needs source identifiers to count devices, which only the full projection returns; source
                # names are loaded with them but never read or written.
                projection = "full" if feature == "Sleep" else "default"
                data = loader.get_data(registration_codes=code, max_rows=None, projection=projection, **extra)
            if data.df.empty:
                continue
            if feature == "Sleep":
                out["tables"]["sleep_nights"] = sleep_nights(data)
            else:
                out["tables"]["cgm_days"], out["tables"]["cgm_periods"] = cgm_metrics(data)
        except Exception as exc:
            out["errors"].append({"RegistrationCode": code, "feature": feature, "error": f"{type(exc).__name__}: {exc}"})
    return out


def compute_domain_metrics(
    phase: str = "curated", *, root: str | Path | None = None, participants: Iterable[str] | None = None,
    default_inclusion_only: bool = False, state_validation: str = "auto", workers: int = 1,
    out: str | Path | None = None, resume: bool = False,
) -> DomainMetrics:
    """
    Sleep nights and CGM metrics for every participant with Sleep or BloodGlucose data, streamed one participant at a
    time, with the same written, resumable outputs as ``data_statistics.compute_daily_statistics``.
    """

    phase = ds._check_phase(phase)
    if default_inclusion_only and phase != "curated":
        raise DataLoaderConfigurationError("default_inclusion_only applies to the curated phase only")
    root_path = Path(root).expanduser() if root is not None else Path(
        ds.DEFAULT_CURATED_ROOT if phase == "curated" else ds.DEFAULT_NATIVE_ROOT)
    started, clock = datetime.now(timezone.utc), time.perf_counter()
    out_dir = ds._guard_output(Path(out), [root_path]) if out is not None else None
    if participants is None:
        coverage = ds.compute_coverage(phase, root=root_path, features=["Sleep", "BloodGlucose"])
        codes = sorted(coverage.participant_feature["RegistrationCode"].unique())
    else:
        codes = sorted({c if c.startswith("10K_") else f"10K_{c}" for c in (str(p).strip() for p in participants)})
    parameters = {"tool": "domain_metrics", "phase": phase, "root": str(root_path.resolve()), "participants": codes,
                  "default_inclusion_only": default_inclusion_only, "state_validation": state_validation,
                  "definitions": definitions()}
    outputs = ds._Outputs(out_dir, parameters, resume, tables=DOMAIN_TABLES) if out_dir is not None else None
    pending = [c for c in codes if outputs is None or c not in outputs.completed]
    tasks = [(code, phase, root_path, state_validation, default_inclusion_only) for code in pending]
    memory: dict[str, list] = {n: [] for n in DOMAIN_TABLES}
    errors: list[dict[str, str]] = []

    def consume(result):
        if outputs is not None:
            outputs.write(result, result["tables"])
            return
        errors.extend(result["errors"])
        for name, frame in result["tables"].items():
            if not frame.empty:
                memory[name].append(frame)

    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_domain_task, tasks, chunksize=1):
                consume(result)
    else:
        for task in tasks:
            consume(_domain_task(task))

    columns = {"sleep_nights": NIGHT_COLUMNS, "cgm_days": CGM_DAY_COLUMNS, "cgm_periods": CGM_PERIOD_COLUMNS}
    if outputs is None:
        tables = {n: pd.concat(memory[n], ignore_index=True) if memory[n] else pd.DataFrame(columns=columns[n])
                  for n in DOMAIN_TABLES}
    else:
        tables = {n: read_domain_table(out_dir, n) if (out_dir / f"{n}.csv.gz").is_file() else pd.DataFrame(columns=columns[n])
                  for n in DOMAIN_TABLES}
        errors_path = out_dir / "errors.csv"
        errors = pd.read_csv(errors_path, dtype=str).to_dict("records") if errors_path.is_file() else []
    with_data = set(tables["sleep_nights"]["RegistrationCode"]) | set(tables["cgm_periods"]["RegistrationCode"])
    run = {"tool": "domain_metrics", **ds._provenance(phase, root_path),
           "participants_without_data": sorted(set(codes) - with_data),
           "parameters": {**{k: v for k, v in parameters.items() if k != "participants"}, "participants": len(codes)},
           "workers": workers, "resumed": bool(outputs is not None and len(pending) < len(codes)),
           "participants_processed_now": len(pending), "errors": len(errors),
           "started": started.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
           "seconds": round(time.perf_counter() - clock, 2)}
    if out_dir is not None:
        import json
        (out_dir / "run.json").write_text(json.dumps(run, indent=2, default=str))
    return DomainMetrics(tables["sleep_nights"], tables["cgm_days"], tables["cgm_periods"], errors, run)


# ------------------------------------------------------------------------------------------------ summaries
SLEEP_SUMMARY_METRICS = {
    "asleep_minutes": "min", "sleep_period_minutes": "min", "in_bed_minutes": "min", "efficiency": "fraction",
    "waso_minutes": "min", "onset_hours_after_noon": "h after noon", "offset_hours_after_noon": "h after noon",
    "midpoint_hours_after_noon": "h after noon", "core_share": "fraction", "deep_share": "fraction",
    "rem_share": "fraction", "nap_minutes": "min",
}
CGM_SUMMARY_METRICS = {"mean_mgdl": "mg/dL", "mean_mmol": "mmol/L", "gmi_percent": "%", "cv_percent": "%",
                       "sd_mgdl": "mg/dL", "active_percent": "%", **{n: "%" for n, _, _ in GLUCOSE_RANGES}}


def summarize_domain(metrics: DomainMetrics, *, night_rule: NightRule = NightRule()) -> sm.Summaries:
    """
    Participant summaries over valid nights (``night_rule``), CGM period metrics for sufficient participants, and
    their cohort table, in the same long format as ``data_summaries``. Sleep appears as feature ``SleepNight``;
    CGM as ``CGM``, one value per participant, so its n is 1.
    """

    rows, adherence = [], []
    nights = metrics.sleep_nights
    for code, group in nights.groupby("RegistrationCode", sort=True):
        valid = group[group["asleep_recorded"].astype(bool) & (group["asleep_minutes"] >= night_rule.min_asleep_minutes)]
        days = list(group["night_date"])
        longest, gaps = sm._runs(list(valid["night_date"]))
        span = (max(days) - min(days)).days + 1
        adherence.append({"RegistrationCode": code, "feature": "SleepNight", "rule": night_rule.describe(),
                          "first_day": min(days), "last_day": max(days), "span_days": span, "days_with_data": len(days),
                          "valid_days": len(valid), "adherence": len(valid) / span,
                          "first_valid_day": min(valid["night_date"]) if len(valid) else None,
                          "last_valid_day": max(valid["night_date"]) if len(valid) else None,
                          "longest_valid_run": longest, "gaps": len(gaps), "longest_gap_days": max(gaps) if gaps else 0,
                          "median_gap_days": float(np.median(gaps)) if gaps else np.nan,
                          "nights_in_bed_only": int((~group["asleep_recorded"].astype(bool) & group["in_bed_recorded"].astype(bool)).sum())})
        for metric, unit in SLEEP_SUMMARY_METRICS.items():
            rows.append({"RegistrationCode": code, "feature": "SleepNight", "metric": metric, "unit": unit,
                         **sm._describe(valid[metric])})
    for record in metrics.cgm_periods.to_dict("records"):
        if not (record["cgm"] and record["sufficient"]):
            continue
        for metric, unit in CGM_SUMMARY_METRICS.items():
            value = pd.Series([record[metric]], dtype=float)
            rows.append({"RegistrationCode": record["RegistrationCode"], "feature": "CGM", "metric": metric,
                         "unit": unit, **sm._describe(value)})
    participant_metrics = pd.DataFrame(rows, columns=["RegistrationCode", "feature", "metric", "unit", *sm.STATISTICS])
    adherence_table = pd.DataFrame(adherence)
    return sm.Summaries(participant_metrics, adherence_table, sm.cohort_table(participant_metrics, adherence_table),
                        sm.retention(adherence_table) if len(adherence_table) else pd.DataFrame(),
                        {"SleepNight": night_rule.describe(),
                         "CGM": f"cgm cadence, >= {CGM_SUFFICIENT_DAYS} valid days of >= {CGM_MIN_DAY_COMPLETENESS:g} completeness"})
