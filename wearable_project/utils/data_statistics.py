"""
Wearable Data Processing and Modeling project
Cohort statistics computed through the DataLoaders, for the native or the curated phase.
Two tiers, chosen by what each question costs at cohort scale:
1. Coverage (``compute_coverage``). Who has which data, how much, and in what curation state. Built from each
   loader's ``profile()``, which reads the state database and one ``stat`` per file, so the whole cohort is covered
   without parsing a single CSV.
2. Daily summaries (``compute_daily_statistics``). One row per participant, local day and feature, streamed
   participant by participant through the loaders, so memory stays bounded however large the cohort. From these rows
   come the cohort tables: participants with data per day, and per-participant coverage spans.
The rules, each derived from a declaration rather than assumed:
- A day is the participant's local calendar day: the local time of an event is its UTC time plus the row's
  ``utc_offset_minutes``, the rule ``with_local_time()`` applies. A row without an offset falls back to its UTC day and
  is counted in ``records_without_offset``. ActivitySummary has no event times; its day is the export's day key.
- Intervals are split at local midnight. Each piece belongs to its day.
- Values are combined as the curation registry's measurement kind requires. Totals (``extensive_total``) are summed,
  each interval contributing to a day in proportion to its overlap with it, the allocation the registry declares, so
  daily sums add up to the stored total. Event amounts are summed on the event's day. Levels, rates, proportions and
  device summaries are described by mean, median, minimum and maximum on the sample's day; they are never summed.
- Time is never counted twice. Observed minutes, sleep-state minutes and mindful minutes are the length of the union of
  intervals, so overlapping records from several devices are counted once. Sleep has one ``minutes_<state>`` column
  per state and ``minutes_asleep_total``, the union across the asleep states (unspecified, core, deep and REM).
- Where a unit can be harmonized, values are harmonized first, so a feature's values share one unit; a value whose
  unit could not be established is left out of the value statistics and counted in ``values_unresolved``.
``python -m wearable_project.utils.data_statistics coverage|daily --phase curated --out DIR`` runs either tier from the
command line. Outputs are written only outside the data roots.
"""


from __future__ import annotations
import argparse
import json
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from wearable_project import __release_label__, __version__
from wearable_project.curation import schema
from wearable_project.curation.registry import get_policy
from wearable_project.DataLoaders import available_features
from wearable_project.DataLoaders._base import DEFAULT_CURATED_ROOT, DEFAULT_NATIVE_ROOT
from wearable_project.exceptions import DataLoaderConfigurationError, DataLoaderError
from wearable_project.processing.registry import get_feature_spec


PHASES = ("curated", "native")
# HealthKit's sleep-analysis categories, and those that mean the participant was asleep (unspecified, core, deep,
# REM). Every Sleep day carries a column for each, so all participants' tables share one set of columns.
SLEEP_STATES = ("INBED", "ASLEEP", "AWAKE", "CORE", "DEEP", "REM")
ASLEEP_STATES = frozenset({"ASLEEP", "CORE", "DEEP", "REM"})
_US = np.timedelta64(1, "us")
_DAY = np.timedelta64(1, "D")


# ------------------------------------------------------------------------------------------------ helpers
def _loader(feature: str, phase: str, root: Path | None, **kwargs):
    module = __import__(f"wearable_project.DataLoaders.{feature}Loader", fromlist=[f"{feature}Loader"])
    cls = getattr(module, f"{feature}Loader")
    return cls(phase=phase, root=root, **kwargs) if root is not None else cls(phase=phase, **kwargs)


def _check_phase(phase: str) -> str:
    phase = str(phase).strip().lower()
    if phase not in PHASES:
        raise DataLoaderConfigurationError(f"phase must be 'curated' or 'native'; received {phase!r}")
    return phase


def _features(features: Iterable[str] | None) -> tuple[str, ...]:
    known = available_features()
    if features is None:
        return known
    chosen = [features] if isinstance(features, str) else list(features)
    unknown = sorted(set(chosen) - set(known))
    if unknown:
        raise DataLoaderConfigurationError(f"unknown feature(s): {', '.join(unknown)}")
    return tuple(f for f in known if f in set(chosen))


def column_units(feature: str) -> dict[str, dict[str, str | None]]:
    """
    For each numeric measurement column: the unit the loaders establish for its stored values, and the unit the
    curation registry declares. They differ where curation withholds a unit or the processing registry names none.
    Statistics of ``value`` use harmonized values, whose unit is reported per row in ``value_unit``.
    """

    from wearable_project.DataLoaders._units import _curation_declaration, stored_unit

    categorical = schema.categorical_columns(feature)
    units = {}
    for column in get_feature_spec(feature).measurement_columns:
        if column in categorical or schema.role_of(column) is schema.ColumnRole.PAYLOAD:
            continue
        declaration = _curation_declaration(feature, column)
        units[column] = {"established": stored_unit(feature, column).unit,
                         "declared_by_curation": declaration.raw_unit if declaration is not None else None}
    return units


def measurement_kind(feature: str) -> str:
    """The measurement kind the curation registry declares for a feature."""

    return get_policy(feature, allow_fallback=False).as_dict()["semantics"]["measurement_kind"]


def _guard_output(out: Path, roots: Iterable[Path]) -> Path:
    out = Path(out).expanduser().resolve()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if out == root or root in out.parents:
            raise DataLoaderConfigurationError(f"{out} lies inside the data root {root}; outputs are never written there")
    return out


# ---------------------------------------------------------------------------------------------- coverage
@dataclass
class Coverage:
    """Who has which data. ``participant_feature`` has one row per participant and feature with a file."""

    phase: str
    participant_feature: pd.DataFrame
    features: pd.DataFrame
    participants: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    def presence(self) -> pd.DataFrame:
        """Participants by features: True where the participant has a file for the feature."""

        table = self.participant_feature.assign(present=True).pivot_table(
            index="RegistrationCode", columns="feature", values="present", aggfunc="any", fill_value=False)
        return table.reindex(columns=list(self.features.index), fill_value=False).astype(bool)

    def co_availability(self, features: Iterable[str]) -> list[str]:
        """The participants who have every one of the given features."""

        wanted = list(features)
        presence = self.presence()
        missing = [f for f in wanted if f not in presence.columns]
        if missing:
            return []
        return sorted(presence.index[presence[wanted].all(axis=1)])


def compute_coverage(phase: str = "curated", *, root: str | Path | None = None,
                     features: Iterable[str] | None = None) -> Coverage:
    """
    Coverage of every feature from the state databases, without parsing any CSV. ``root`` defaults to the phase's
    permanent HPP root. Curated coverage adds the curation status, default-inclusion and unit-resolution counts.
    """

    phase = _check_phase(phase)
    rows, notes = [], []
    for feature in _features(features):
        report = _loader(feature, phase, Path(root) if root is not None else None).profile()
        table = report.participants
        present = table[table["on_disk"].astype(bool)]
        for code, entry in present.iterrows():
            row = {"RegistrationCode": str(code), "feature": feature,
                   "rows": entry.get("rows"), "bytes_on_disk": entry.get("bytes_on_disk"),
                   "verifiable": bool(entry.get("has_state", False))}
            if phase == "curated":
                for column in ("pass_rows", "review_rows", "exclude_default_rows", "included_by_default_rows",
                               "excluded_by_default_rows", "canonical_value_rows", "ambiguous_unit_rows",
                               "acquisition_classified_fraction"):
                    row[column] = entry.get(column)
            rows.append(row)
        unverifiable = int((~present["has_state"].astype(bool)).sum()) if "has_state" in present else len(present)
        if unverifiable:
            notes.append(f"{feature}: {unverifiable} file(s) have no state record; their rows are unknown")
    participant_feature = pd.DataFrame(rows)
    if participant_feature.empty:
        participant_feature = pd.DataFrame(columns=["RegistrationCode", "feature", "rows", "bytes_on_disk", "verifiable"])
    participant_feature["rows"] = pd.to_numeric(participant_feature["rows"]).astype("Int64")
    participant_feature["bytes_on_disk"] = pd.to_numeric(participant_feature["bytes_on_disk"]).astype("Int64")

    grouped = participant_feature.groupby("feature", sort=False)
    per_feature = pd.DataFrame({
        "participants": grouped["RegistrationCode"].nunique(),
        "rows": grouped["rows"].sum(min_count=1),
        "bytes_on_disk": grouped["bytes_on_disk"].sum(min_count=1),
        "median_rows_per_participant": grouped["rows"].median(),
        "max_rows_per_participant": grouped["rows"].max(),
    }).reindex(list(_features(features)))
    per_feature["participants"] = per_feature["participants"].fillna(0).astype(int)
    if phase == "curated" and not participant_feature.empty:
        for column in ("pass_rows", "review_rows", "exclude_default_rows", "included_by_default_rows",
                       "excluded_by_default_rows", "canonical_value_rows", "ambiguous_unit_rows"):
            per_feature[column] = grouped[column].sum(min_count=1)
    per_feature.index.name = "feature"

    by_participant = participant_feature.groupby("RegistrationCode")
    per_participant = pd.DataFrame({
        "features": by_participant["feature"].nunique(),
        "rows": by_participant["rows"].sum(min_count=1),
        "bytes_on_disk": by_participant["bytes_on_disk"].sum(min_count=1),
    }).sort_index()
    return Coverage(phase, participant_feature, per_feature, per_participant, notes)


# ------------------------------------------------------------------------------------------ daily summaries
def _local_frame(data) -> tuple[pd.DataFrame, str]:
    """
    The rows of one participant's result with naive local start and end times, the basis used for days, and the
    count of rows that had no UTC offset. Inverted intervals (end before start) are treated as instants.
    """

    df = data.df.reset_index()
    if data.load_report["date_column"] != "start_date":  # ActivitySummary: the export's day key
        day = df["Date"].dt.tz_convert(None)
        df["local_start"] = day
        df["local_end"] = day
        df["without_offset"] = False
        return df, "export_day_key"
    local = data.with_local_time().df.reset_index()
    fallback_start = df["start_date"].dt.tz_convert(None)
    fallback_end = df["end_date"].dt.tz_convert(None) if "end_date" in df else fallback_start
    df["without_offset"] = local["start_date_local"].isna().to_numpy()
    df["local_start"] = local["start_date_local"].where(local["start_date_local"].notna(), fallback_start)
    end = local["end_date_local"] if "end_date_local" in local else df["local_start"]
    end = end.where(end.notna(), fallback_end if "end_date" in df else df["local_start"])
    df["local_end"] = end.where(end >= df["local_start"], df["local_start"])  # inverted -> instant
    return df, "local"


def _pieces(start: pd.Series, end: pd.Series) -> pd.DataFrame:
    """Split each [start, end] interval at local midnights: one row per (source row, day) piece."""

    s = start.to_numpy(dtype="datetime64[us]")
    e = end.to_numpy(dtype="datetime64[us]")
    first = s.astype("datetime64[D]")
    last = np.where(e > s, (e - _US).astype("datetime64[D]"), first)
    count = (last - first).astype("timedelta64[D]").astype(np.int64) + 1
    source = np.repeat(np.arange(len(s)), count)
    offset = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
    day = first[source] + offset.astype("timedelta64[D]")
    piece_start = np.maximum(s[source], day.astype("datetime64[us]"))
    piece_end = np.minimum(e[source], (day + _DAY).astype("datetime64[us]"))
    length = (e - s)[source]
    share = np.where(length > np.timedelta64(0, "us"), (piece_end - piece_start) / np.where(length > np.timedelta64(0, "us"), length, _US), 1.0)
    return pd.DataFrame({"row": source, "day": day, "start": piece_start, "end": piece_end, "share": share})


def _union_minutes(pieces: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Minutes covered by the union of the pieces' intervals, per group: overlapping time is counted once."""

    frame = pieces[pieces["end"] > pieces["start"]].sort_values(keys + ["start"])
    if frame.empty:
        return pd.Series(dtype="float64")
    running = frame.groupby(keys, sort=False)["end"].cummax()
    previous = running.groupby([frame[k] for k in keys], sort=False).shift()
    new_segment = previous.isna() | (frame["start"] > previous)
    segment = new_segment.cumsum()
    spans = frame.assign(segment=segment.to_numpy(), running=running.to_numpy()).groupby(
        keys + ["segment"], sort=False).agg(start=("start", "min"), end=("running", "max"))
    minutes = (spans["end"] - spans["start"]).dt.total_seconds() / 60.0
    return minutes.groupby(level=list(range(len(keys)))).sum()


def summarize_result(data, feature: str | None = None, *, harmonize: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Daily summary and hour-of-day profile of one loaded result, typically one participant's. Returns
    ``(daily, hourly)``: ``daily`` has one row per participant and local day; ``hourly`` counts records per local hour.
    The value columns depend on the feature's measurement kind; see the module docstring. With ``harmonize=False``
    value statistics use the stored values as they are; ``column_units`` says which unit that is.
    """

    feature = feature or data.load_report["feature"]
    kind = measurement_kind(feature)
    categorical = schema.categorical_columns(feature)
    spec = get_feature_spec(feature)
    if data.df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df, basis = _local_frame(data)
    df["day"] = df["local_start"].dt.normalize()
    keys = ["RegistrationCode", "day"]
    pieces = _pieces(df["local_start"], df["local_end"])
    pieces["RegistrationCode"] = df["RegistrationCode"].to_numpy()[pieces["row"]]
    # The same dtype as the start days: pandas 2.1 cannot concatenate date columns of different resolutions.
    pieces["day"] = pd.to_datetime(pieces["day"]).astype(df["day"].dtype)
    # A day has a row wherever any piece of data falls, including the part of a night's sleep after midnight;
    # records are counted on the day they start, so such a day can hold time or value with zero records.
    days = pd.concat([df[keys], pieces[keys]]).drop_duplicates().sort_values(keys)
    counted = df.groupby(keys, sort=True).agg(records=("local_start", "size"),
                                              records_without_offset=("without_offset", "sum"))
    daily = counted.reindex(pd.MultiIndex.from_frame(days))
    daily["records"] = daily["records"].fillna(0).astype(int)
    daily["records_without_offset"] = daily["records_without_offset"].fillna(0).astype(int)
    daily["day_basis"] = basis
    if "include_by_default" in df:
        daily["records_included"] = df.groupby(keys)["include_by_default"].sum().reindex(daily.index).fillna(0).astype(int)

    # A day key carries no time of day, so hour-of-day statistics exist only for event times.
    if basis == "local":
        hours = df.assign(hour=df["local_start"].dt.hour)
        daily["hours_with_data"] = hours.groupby(keys)["hour"].nunique().reindex(daily.index).fillna(0).astype(int)
        hourly = hours.groupby(["RegistrationCode", "hour"]).size().rename("records").reset_index()
        hourly.insert(1, "feature", feature)
    else:
        hourly = pd.DataFrame(columns=["RegistrationCode", "feature", "hour", "records"])

    if basis == "local":
        daily["observed_minutes"] = _union_minutes(pieces, keys).reindex(daily.index).fillna(0.0)

    # Values: harmonized where the unit can be harmonized, so every value of the feature shares one unit.
    # Declared columns, whether or not this participant's file stores them, so every participant's table has the
    # same columns; a column the file lacks gives missing statistics.
    value_columns = [c for c in spec.measurement_columns
                     if c not in categorical and schema.role_of(c) is not schema.ColumnRole.PAYLOAD]
    for column in value_columns:
        if column not in df:
            df[column] = np.nan
    unit = None
    if "value" in value_columns and harmonize:
        daily["values_unresolved"] = np.nan
        try:
            harmonized = data.with_harmonized_values().df.reset_index()
            df["value"] = harmonized["harmonized_value"].to_numpy()
            units = harmonized["harmonized_unit"].dropna().astype(str).unique()
            unit = units[0] if len(units) == 1 else (None if not len(units) else "mixed")
            daily["values_unresolved"] = df.assign(
                missing=harmonized["harmonized_unit_source"].astype(str).eq("unresolved").to_numpy()
            ).groupby(keys)["missing"].sum()
        except DataLoaderError:
            pass
    daily["value_unit"] = unit

    if kind == "extensive_total" and "value" in df:
        pieces["value"] = df["value"].to_numpy()[pieces["row"]] * pieces["share"]
        daily["value_sum"] = pieces.groupby(keys)["value"].sum(min_count=1)
    elif kind == "event_amount" and "value" in df:
        daily["value_sum"] = df.groupby(keys)["value"].sum(min_count=1)
        daily["value_count"] = df.groupby(keys)["value"].count()
    elif kind in ("intensive_value", "ratio", "summary_statistic") and "value" in df:
        grouped = df.groupby(keys)["value"]
        for name, fn in (("value_mean", "mean"), ("value_median", "median"), ("value_min", "min"),
                         ("value_max", "max"), ("value_count", "count")):
            daily[name] = getattr(grouped, fn)()
    elif kind in ("multivariate_point", "multivariate_summary", "signal"):
        for column in value_columns:
            values = pd.to_numeric(df[column], errors="coerce")
            daily[f"{column}_mean"] = values.groupby([df[k] for k in keys]).mean()
            if kind == "multivariate_point":
                daily[f"{column}_count"] = values.groupby([df[k] for k in keys]).count()
    elif kind == "categorical_state" and "value" in df:
        pieces["state"] = df["value"].astype(str).to_numpy()[pieces["row"]]
        by_state = _union_minutes(pieces, keys + ["state"])
        table = by_state.unstack("state", fill_value=0.0) if not by_state.empty else pd.DataFrame(index=daily.index)
        for state in SLEEP_STATES:
            column = table[state].reindex(daily.index) if state in table.columns else None
            daily[f"minutes_{state.lower()}"] = column.fillna(0.0) if column is not None else 0.0
        others = pieces[~pieces["state"].isin(SLEEP_STATES)]
        daily["minutes_other_states"] = _union_minutes(others, keys).reindex(daily.index).fillna(0.0) if not others.empty else 0.0
        # The union across every asleep state. Named apart from minutes_asleep, which is HealthKit's single
        # "asleep, stage unspecified" state.
        asleep = pieces[pieces["state"].isin(ASLEEP_STATES)]
        daily["minutes_asleep_total"] = _union_minutes(asleep, keys).reindex(daily.index).fillna(0.0) if not asleep.empty else 0.0
    elif kind == "duration":
        daily["sessions"] = daily["records"]
        daily["minutes"] = _union_minutes(pieces, keys).reindex(daily.index).fillna(0.0)

    daily = daily.reset_index().rename(columns={"day": "local_date"})
    daily["local_date"] = daily["local_date"].dt.date
    daily.insert(1, "feature", feature)
    return daily, hourly


# ------------------------------------------------------------------------------------------ the whole run
@dataclass
class DailyStatistics:
    """The outcome of ``compute_daily_statistics``. Tables are in memory unless the run wrote them to ``out``."""

    phase: str
    daily: dict[str, pd.DataFrame]
    hourly: pd.DataFrame
    participant_feature: pd.DataFrame
    cohort_daily: pd.DataFrame
    active_participants: pd.DataFrame
    errors: list[dict[str, str]]
    run: dict[str, Any]


def _participant_task(args) -> dict[str, Any]:
    """Every requested feature for one participant; runs in a worker process when ``workers > 1``."""

    code, phase, root, features, default_inclusion_only, state_validation, harmonize = args
    out = {"code": code, "daily": {}, "hourly": [], "errors": []}
    for feature in features:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = _loader(feature, phase, root, state_validation=state_validation).get_data(
                    registration_codes=code, max_rows=None,
                    **({"default_inclusion_only": True} if default_inclusion_only else {}))
            if data.df.empty:
                continue
            daily, hourly = summarize_result(data, feature, harmonize=harmonize)
            out["daily"][feature] = daily
            out["hourly"].append(hourly)
        except Exception as exc:  # recorded per participant and feature; the run continues
            out["errors"].append({"RegistrationCode": code, "feature": feature, "error": f"{type(exc).__name__}: {exc}"})
    return out


def compute_daily_statistics(
    phase: str = "curated", *, root: str | Path | None = None, features: Iterable[str] | None = None,
    participants: Iterable[str] | None = None, default_inclusion_only: bool = False, harmonize: bool = True,
    state_validation: str = "auto", workers: int = 1, out: str | Path | None = None,
) -> DailyStatistics:
    """
    Daily summaries for every participant and feature, streamed one participant at a time.
    ``participants`` defaults to everyone with data. ``default_inclusion_only`` (curated phase) restricts every
    statistic to the policy's default subset. ``harmonize=False`` computes value statistics from stored values rather
    than harmonized ones. With ``out``, each feature's daily table is written to
    ``out/daily/<Feature>.csv.gz`` as participants finish, in participant order, and only the cohort tables stay in
    memory; without it, everything is returned in memory, which suits a few participants. ``workers > 1`` processes
    participants in parallel. A participant-feature that cannot be loaded is recorded in ``errors`` and skipped.
    """

    phase = _check_phase(phase)
    if default_inclusion_only and phase != "curated":
        raise DataLoaderConfigurationError("default_inclusion_only applies to the curated phase only")
    chosen = _features(features)
    root_path = Path(root).expanduser() if root is not None else Path(DEFAULT_CURATED_ROOT if phase == "curated" else DEFAULT_NATIVE_ROOT)
    started = time.perf_counter()
    if participants is None:
        coverage = compute_coverage(phase, root=root_path, features=chosen)
        codes = sorted(coverage.participant_feature["RegistrationCode"].unique())
    else:
        codes = sorted({c if str(c).startswith("10K_") else f"10K_{c}" for c in (str(p).strip() for p in participants)})
    out_dir = _guard_output(Path(out), [root_path]) if out is not None else None
    if out_dir is not None:
        (out_dir / "daily").mkdir(parents=True, exist_ok=True)

    daily_frames: dict[str, list[pd.DataFrame]] = {f: [] for f in chosen}
    written: set[str] = set()
    headers: dict[str, list[str]] = {}
    hourly_frames, errors, pf_rows = [], [], []
    cohort: Counter = Counter()
    cohort_records: Counter = Counter()
    cohort_value_sum: dict[tuple[str, Any], float] = {}
    active: Counter = Counter()
    tasks = [(code, phase, root_path, chosen, default_inclusion_only, state_validation, harmonize) for code in codes]

    def consume(result: dict[str, Any]) -> None:
        errors.extend(result["errors"])
        if result["hourly"]:
            hourly_frames.append(pd.concat(result["hourly"], ignore_index=True))
        days_any: set = set()
        for feature, daily in result["daily"].items():
            days_any.update(daily["local_date"])
            pf_rows.append({"RegistrationCode": result["code"], "feature": feature, "days_with_data": len(daily),
                            "first_day": daily["local_date"].min(), "last_day": daily["local_date"].max(),
                            "records": int(daily["records"].sum())})
            for day, records in zip(daily["local_date"], daily["records"]):
                cohort[(feature, day)] += 1
                cohort_records[(feature, day)] += int(records)
            if "value_sum" in daily:
                for day, value in zip(daily["local_date"], daily["value_sum"]):
                    if pd.notna(value):
                        cohort_value_sum[(feature, day)] = cohort_value_sum.get((feature, day), 0.0) + float(value)
            if out_dir is not None:
                if feature in headers and list(daily.columns) != headers[feature]:
                    raise RuntimeError(  # never append rows under a header they do not match
                        f"{feature}: participant {result['code']} produced columns {list(daily.columns)}, "
                        f"not {headers[feature]}")
                headers.setdefault(feature, list(daily.columns))
                path = out_dir / "daily" / f"{feature}.csv.gz"
                daily.to_csv(path, mode="a" if feature in written else "w", header=feature not in written, index=False)
                written.add(feature)
            else:
                daily_frames[feature].append(daily)
        for day in days_any:
            active[day] += 1

    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_participant_task, tasks, chunksize=1):  # results arrive in participant order
                consume(result)
    else:
        for task in tasks:
            consume(_participant_task(task))

    cohort_daily = pd.DataFrame(
        [{"feature": f, "local_date": d, "participants": n, "records": cohort_records[(f, d)],
          "value_sum": cohort_value_sum.get((f, d))} for (f, d), n in cohort.items()],
        columns=["feature", "local_date", "participants", "records", "value_sum"],
    ).sort_values(["feature", "local_date"], ignore_index=True)
    cohort_daily["value_sum_per_participant"] = cohort_daily["value_sum"] / cohort_daily["participants"]
    active_participants = pd.DataFrame(sorted(active.items()), columns=["local_date", "participants"])
    participant_feature = pd.DataFrame(pf_rows, columns=["RegistrationCode", "feature", "days_with_data",
                                                         "first_day", "last_day", "records"])
    hourly = pd.concat(hourly_frames, ignore_index=True) if hourly_frames else pd.DataFrame(
        columns=["RegistrationCode", "feature", "hour", "records"])
    daily = {} if out_dir is not None else {
        f: pd.concat(frames, ignore_index=True) for f, frames in daily_frames.items() if frames}
    run = {"tool": "data_statistics", "package": {"version": __version__, "release_label": __release_label__},
           "pandas": pd.__version__, "phase": phase, "root": str(root_path), "features": list(chosen),
           "participants": len(codes), "default_inclusion_only": default_inclusion_only, "harmonize": harmonize,
           "state_validation": state_validation, "workers": workers, "errors": len(errors),
           "measurement_kinds": {f: measurement_kind(f) for f in chosen},
           "units": {f: column_units(f) for f in chosen},
           "seconds": round(time.perf_counter() - started, 2),
           "finished": datetime.now(timezone.utc).isoformat()}
    if out_dir is not None:
        hourly.to_csv(out_dir / "hourly_profile.csv.gz", index=False)
        participant_feature.to_csv(out_dir / "participant_feature_days.csv", index=False)
        cohort_daily.to_csv(out_dir / "cohort_daily.csv", index=False)
        active_participants.to_csv(out_dir / "active_participants.csv", index=False)
        pd.DataFrame(errors, columns=["RegistrationCode", "feature", "error"]).to_csv(out_dir / "errors.csv", index=False)
        (out_dir / "run.json").write_text(json.dumps(run, indent=2, default=str))
    return DailyStatistics(phase, daily, hourly, participant_feature, cohort_daily, active_participants, errors, run)


# ------------------------------------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cohort statistics through the DataLoaders.")
    parser.add_argument("tier", choices=["coverage", "daily"])
    parser.add_argument("--phase", default="curated", choices=list(PHASES))
    parser.add_argument("--root", type=Path, default=None, help="data root (default: the phase's HPP root)")
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--participants", nargs="+", default=None, help="registration codes (default: everyone)")
    parser.add_argument("--default-inclusion-only", action="store_true")
    parser.add_argument("--stored-values", action="store_true", help="value statistics from stored, not harmonized, values")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True, help="output directory, outside the data roots")
    args = parser.parse_args(argv)
    root = args.root or Path(DEFAULT_CURATED_ROOT if args.phase == "curated" else DEFAULT_NATIVE_ROOT)
    try:
        out = _guard_output(args.out, [root])
        if args.tier == "coverage":
            coverage = compute_coverage(args.phase, root=root, features=args.features)
            out.mkdir(parents=True, exist_ok=True)
            coverage.participant_feature.to_csv(out / "coverage_participant_feature.csv", index=False)
            coverage.features.to_csv(out / "coverage_features.csv")
            coverage.participants.to_csv(out / "coverage_participants.csv")
            print(coverage.features.to_string())
        else:
            result = compute_daily_statistics(
                args.phase, root=root, features=args.features, participants=args.participants,
                default_inclusion_only=args.default_inclusion_only, harmonize=not args.stored_values,
                workers=args.workers, out=out)
            print(json.dumps(result.run, indent=2, default=str))
    except DataLoaderConfigurationError as exc:
        parser.error(str(exc))
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
