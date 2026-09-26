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
Written runs are made for the full cohort: each participant's rows are appended as it finishes, an interrupted run
resumes where it stopped (``resume=True``), ``read_daily``, ``iter_daily`` and ``read_table`` read the tables back one
participant at a time, and ``run.json`` records versions, registry fingerprints and the state database's checksum.
``python -m wearable_project.utils.data_statistics coverage|daily --phase curated --out DIR`` runs either tier from the
command line. Outputs are written only outside the data roots.
"""


from __future__ import annotations
import re
import os
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
from wearable_project.DataLoaders._base import DEFAULT_CURATED_ROOT, DEFAULT_NATIVE_ROOT, _PHASE_STATE_FILES
from wearable_project.exceptions import DataLoaderConfigurationError, DataLoaderError
from wearable_project.processing.registry import get_feature_spec


PHASES = ("curated", "native")
HOURLY_COLUMNS = ["RegistrationCode", "feature", "hour", "records", "value_sum", "value_mean", "value_count"]
PROVENANCE_COLUMNS = ["RegistrationCode", "feature", "local_date", "acquisition_method", "records", "records_user_entered"]
CURATION_COLUMNS = ["RegistrationCode", "feature", "local_date", "kind", "name", "records"]
# HealthKit's sleep-analysis categories, and those that mean the participant was asleep (unspecified, core, deep,
# REM). Every Sleep day carries a column for each, so all participants' tables share one set of columns.
SLEEP_STATES = ("INBED", "ASLEEP", "AWAKE", "CORE", "DEEP", "REM")
ASLEEP_STATES = frozenset({"ASLEEP", "CORE", "DEEP", "REM"})
# Broad physiological limits, in each feature's harmonized unit, for reporting values that are likely errors; they
# are conventions, not clinical thresholds, and values outside them are counted, never altered or removed. A range
# is applied only when the values are in its unit. Totals and event amounts are never negative.
PLAUSIBLE_RANGES: dict[str, dict[str, tuple[float | None, float | None, str | None]]] = {
    "HeartRate": {"value": (25.0, 250.0, "beats/min")},
    "RestingHeartRate": {"value": (25.0, 150.0, "beats/min")},
    "WalkingHeartRate": {"value": (30.0, 220.0, "beats/min")},
    "HeartRateVariability": {"value": (1.0, 300.0, "ms")},
    "OxygenSaturation": {"value": (50.0, 100.0, "%")},
    "RespiratoryRate": {"value": (4.0, 60.0, "breaths/min")},
    "BodyTemperature": {"value": (32.0, 43.0, "Cel")},
    "BloodGlucose": {"value": (1.1, 33.3, "mmol/L")},
    "Weight": {"value": (25.0, 300.0, "kg")},
    "Height": {"value": (1.0, 2.5, "m")},
    "BMI": {"value": (10.0, 80.0, "kg/m2")},
    "BodyFatPercentage": {"value": (2.0, 75.0, "%")},
    "LeanBodyMass": {"value": (15.0, 150.0, "kg")},
    "WaistCircumference": {"value": (0.4, 2.0, "m")},
    "Vo2Max": {"value": (5.0, 100.0, "mL/(kg*min)")},
    "PeakFlow": {"value": (50.0, 1000.0, "L/min")},
    "BloodAlcoholContent": {"value": (0.0, 0.5, "%")},
    "BloodPressure": {"blood_pressure_systolic_value": (50.0, 300.0, "mmHg"),
                      "blood_pressure_diastolic_value": (20.0, 200.0, "mmHg")},
}
_NON_NEGATIVE_KINDS = {"extensive_total", "event_amount"}

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
    if not chosen:
        raise DataLoaderConfigurationError("no features chosen; pass None for every feature")
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


# Outputs never go into a data root, so that a root holds only wearable data: not statistics, summaries, figures or
# reports. Protected are both phases' permanent roots, every root read in this session, and the root a written run
# was computed from once that run is opened.
_ROOTS_READ: set[Path] = set()


def _remember_root(root: str | Path | None) -> None:
    if root is not None:
        _ROOTS_READ.add(Path(root).expanduser().resolve())


def protected_roots(*roots: str | Path | None) -> list[Path]:
    """
    The data roots no output is ever written into: both phases' permanent roots, every root read in this session
    (by ``compute_coverage``, ``compute_daily_statistics`` or ``compute_domain_metrics``, or recorded by a written run
    that summaries or figures were drawn from), and any given.
    """

    candidates = [DEFAULT_NATIVE_ROOT, DEFAULT_CURATED_ROOT, *_ROOTS_READ, *(r for r in roots if r is not None)]
    return sorted({Path(r).expanduser().resolve() for r in candidates})


_PARTICIPANT_FOLDER = re.compile(r"^(10K_)?\d{6,}$")


def _holds_wearable_data(directory: Path, scan: int = 500) -> bool:
    """
    Whether a directory is recognizably a data root by its content: a processing state database at its top, or a
    participant folder holding a feature file. Only the first ``scan`` entries are examined, stopping at a match.
    """

    if any((directory / name).is_file() for name in _PHASE_STATE_FILES.values()):
        return True
    features = set(available_features())
    try:
        with os.scandir(directory) as entries:
            for count, entry in enumerate(entries):
                if count >= scan:
                    break
                if _PARTICIPANT_FOLDER.match(entry.name) and entry.is_dir():
                    with os.scandir(entry.path) as inner:
                        if any(e.name.endswith(".csv") and e.name[:-4] in features for e in inner):
                            return True
    except OSError:
        return False
    return False


def guard_output(path: str | Path, *roots: str | Path | None) -> Path:
    """
    ``path`` resolved (symbolic links followed), or an error if it lies inside a data root: a protected root
    (``protected_roots``), or any directory recognizably holding wearable data, whether this session read it.
    """

    out = Path(path).expanduser().resolve()
    advice = ("statistics, summaries and figures are never written into a data root, so that it holds only wearable "
              "data. Choose a folder outside it, such as ~/wearable_statistics")
    for root in protected_roots(*roots):
        if out == root or root in out.parents:
            raise DataLoaderConfigurationError(f"{out} lies inside the data root {root}; {advice}")
    for directory in (out, *out.parents):
        if directory.is_dir() and _holds_wearable_data(directory):
            raise DataLoaderConfigurationError(f"{out} lies inside {directory}, which holds wearable data; {advice}")
    return out


def _guard_output(out: Path, roots: Iterable[Path]) -> Path:
    return guard_output(out, *roots)


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
        _remember_root(root)
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


def _pieces(start: pd.Series, end: pd.Series, unit: str = "D") -> pd.DataFrame:
    """
    Split each [start, end] interval at unit boundaries, local midnights by default or hours with ``unit="h"``: one
    row per (source row, unit) piece; ``day`` holds the start of the piece's day or hour.
    """

    step = np.timedelta64(1, unit)
    s = start.to_numpy(dtype="datetime64[us]")
    e = end.to_numpy(dtype="datetime64[us]")
    first = s.astype(f"datetime64[{unit}]")
    last = np.where(e > s, (e - _US).astype(f"datetime64[{unit}]"), first)
    count = (last - first).astype(f"timedelta64[{unit}]").astype(np.int64) + 1
    source = np.repeat(np.arange(len(s)), count)
    offset = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
    day = first[source] + offset.astype(f"timedelta64[{unit}]")
    piece_start = np.maximum(s[source], day.astype("datetime64[us]"))
    piece_end = np.minimum(e[source], (day + step).astype("datetime64[us]"))
    length = (e - s)[source]
    share = np.where(length > np.timedelta64(0, "us"), (piece_end - piece_start) / np.where(length > np.timedelta64(0, "us"), length, _US), 1.0)
    return pd.DataFrame({"row": source, "day": day, "start": piece_start, "end": piece_end, "share": share})


def _union_spans(pieces: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """The union of the pieces' intervals per group, as disjoint spans: overlapping or touching intervals merge."""

    frame = pieces[pieces["end"] > pieces["start"]].sort_values(keys + ["start"])
    if frame.empty:
        return pd.DataFrame(columns=[*keys, "start", "end"])
    running = frame.groupby(keys, sort=False)["end"].cummax()
    previous = running.groupby([frame[k] for k in keys], sort=False).shift()
    new_segment = previous.isna() | (frame["start"] > previous)
    segment = new_segment.cumsum()
    spans = frame.assign(segment=segment.to_numpy(), running=running.to_numpy()).groupby(
        keys + ["segment"], sort=False).agg(start=("start", "min"), end=("running", "max"))
    return spans.reset_index().drop(columns="segment")


def _union_minutes(pieces: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Minutes covered by the union of the pieces' intervals, per group: overlapping time is counted once."""

    spans = _union_spans(pieces, keys)
    if spans.empty:
        return pd.Series(dtype="float64")
    minutes = (spans["end"] - spans["start"]).dt.total_seconds() / 60.0
    return minutes.groupby([spans[k] for k in keys]).sum()


def _minutes_covered_at_least(spans: pd.DataFrame, keys: list[str], count: int) -> pd.Series:
    """Minutes per group covered by at least ``count`` of the given spans, by a sweep over their start and end times."""

    if spans.empty:
        return pd.Series(dtype="float64")
    events = pd.concat([spans[keys].assign(time=spans["start"], delta=1), spans[keys].assign(time=spans["end"], delta=-1)],
                       ignore_index=True).sort_values(keys + ["time", "delta"], ignore_index=True)
    running = events.groupby(keys, sort=False)["delta"].cumsum()
    following = events.groupby(keys, sort=False)["time"].shift(-1)
    length = (following - events["time"]).dt.total_seconds().fillna(0.0) / 60.0
    return length.where(running >= count, 0.0).groupby([events[k] for k in keys]).sum()


def _out_of_range(feature, kind, df, keys, unit, harmonize):
    """Values below and above the feature's declared range, per day; ``(None, None)`` where it cannot be assessed."""

    declared = dict(PLAUSIBLE_RANGES.get(feature, {}))
    if kind in _NON_NEGATIVE_KINDS and "value" in df and "value" not in declared:
        declared["value"] = (0.0, None, None)
    groups = [df[k] for k in keys]
    below = above = assessed = None
    for column, (low, high, required) in declared.items():
        if column not in df:
            continue
        if required is not None:  # a range applies only to values in its unit
            if column == "value" and harmonize:
                established = unit
            else:
                established = column_units(feature).get(column, {}).get("established")
            if established != required:
                continue
        values = pd.to_numeric(df[column], errors="coerce")
        low_hits = (values < low) if low is not None else pd.Series(False, index=values.index)
        high_hits = (values > high) if high is not None else pd.Series(False, index=values.index)
        b = low_hits.astype(int).groupby(groups).sum()
        a = high_hits.astype(int).groupby(groups).sum()
        n = values.notna().astype(int).groupby(groups).sum()
        below = b if below is None else below.add(b, fill_value=0)
        above = a if above is None else above.add(a, fill_value=0)
        assessed = n if assessed is None else assessed.add(n, fill_value=0)
    if assessed is None:
        return None, None
    # A day with no value that could be checked is not assessed, which is not the same as nothing out of range.
    return below.where(assessed > 0), above.where(assessed > 0)


def _hourly_values(hourly: pd.DataFrame, df: pd.DataFrame, kind: str, feature: str) -> pd.DataFrame:
    """
    Value statistics per local hour of day, pooled over all days: totals split at hour boundaries in proportion to
    overlap, event amounts summed at their start hour, levels averaged at their start hour.
    """

    keys = ["RegistrationCode", "hour"]
    frame = hourly.set_index(keys)
    frame["value_sum"] = np.nan
    frame["value_mean"] = np.nan
    frame["value_count"] = np.nan
    if "value" in df:
        values = pd.to_numeric(df["value"], errors="coerce")
        if kind == "extensive_total":
            pieces = _pieces(df["local_start"], df["local_end"], unit="h")
            pieces["RegistrationCode"] = df["RegistrationCode"].to_numpy()[pieces["row"]]
            pieces["hour"] = pd.to_datetime(pieces["day"]).dt.hour
            pieces["value"] = values.to_numpy()[pieces["row"]] * pieces["share"]
            sums = pieces.groupby(keys)["value"].sum(min_count=1)
            frame = frame.reindex(frame.index.union(sums.index))
            frame["records"] = frame["records"].fillna(0).astype(int)
            frame["value_sum"] = sums.reindex(frame.index)
        else:
            groups = [df["RegistrationCode"], df["local_start"].dt.hour.rename("hour")]
            if kind == "event_amount":
                frame["value_sum"] = values.groupby(groups).sum(min_count=1).reindex(frame.index)
                frame["value_count"] = values.groupby(groups).count().reindex(frame.index)
            elif kind in ("intensive_value", "ratio", "summary_statistic"):
                frame["value_mean"] = values.groupby(groups).mean().reindex(frame.index)
                frame["value_count"] = values.groupby(groups).count().reindex(frame.index)
    frame = frame.reset_index()
    frame["feature"] = feature
    return frame[HOURLY_COLUMNS]


def provenance_tables(data, feature: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ``(provenance, curation)`` of one result in long format: per participant, local day and acquisition method, the
    records and those flagged user-entered; and, in the curated phase, the records per curation status and per flag.
    A file with no acquisition method gives ``not_recorded``. Only categories and counts, never source names.
    """

    feature = feature or data.load_report["feature"]
    if data.df.empty:
        return pd.DataFrame(columns=PROVENANCE_COLUMNS), pd.DataFrame(columns=CURATION_COLUMNS)
    df, _ = _local_frame(data)
    df["local_date"] = df["local_start"].dt.date
    df["method"] = df["acquisition_method"].astype(str) if "acquisition_method" in df else "not_recorded"
    df["flagged"] = df["was_user_entered"].eq(True).fillna(False).astype(int) if "was_user_entered" in df else 0
    provenance = df.groupby(["RegistrationCode", "local_date", "method"], sort=True).agg(
        records=("method", "size"), records_user_entered=("flagged", "sum")).reset_index().rename(
        columns={"method": "acquisition_method"})
    provenance.insert(1, "feature", feature)
    parts = []
    if "curation_status" in df:
        status = df.groupby(["RegistrationCode", "local_date", df["curation_status"].astype(str).rename("name")]).size()
        parts.append(status.rename("records").reset_index().assign(kind="status"))
    if "curation_flags" in df:
        flags = df[["RegistrationCode", "local_date"]].assign(name=df["curation_flags"].astype(str).str.split(";")).explode("name")
        flags = flags[flags["name"].notna() & (flags["name"] != "") & (flags["name"] != "nan")]
        if len(flags):
            parts.append(flags.groupby(["RegistrationCode", "local_date", "name"]).size().rename("records").reset_index().assign(kind="flag"))
    curation = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=CURATION_COLUMNS)
    curation["feature"] = feature
    return provenance[PROVENANCE_COLUMNS], curation.sort_values(["RegistrationCode", "local_date", "kind", "name"],
                                                               ignore_index=True)[CURATION_COLUMNS]


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
        starts = df[keys + ["local_start"]].drop_duplicates().sort_values(keys + ["local_start"])
        gaps = starts.groupby(keys, sort=False)["local_start"].diff().dt.total_seconds() / 60.0
        gaps = starts.assign(gap=gaps.to_numpy())
        gaps = gaps[gaps["gap"] > 0]
        daily["median_gap_minutes"] = gaps.groupby(keys)["gap"].median().reindex(daily.index)
        daily["max_gap_minutes"] = gaps.groupby(keys)["gap"].max().reindex(daily.index)
    else:
        hourly = pd.DataFrame(columns=HOURLY_COLUMNS)

    if basis == "local":
        daily["observed_minutes"] = _union_minutes(pieces, keys).reindex(daily.index).fillna(0.0)

    # Values: harmonized where the unit can be harmonized, so every value of the feature shares one unit.
    # Declared columns, whether this participant's file stores them, so every participant's table has the
    # same columns; a column the file lacks gives missing statistics.
    value_columns = [c for c in spec.measurement_columns
                     if c not in categorical and schema.role_of(c) is not schema.ColumnRole.PAYLOAD]
    for column in value_columns:
        if column not in df:
            df[column] = np.nan
    unit = None
    if "value" in value_columns and harmonize:
        # No fallback: if harmonization fails, the participant-feature is recorded as an error, rather than being
        # summarized silently from stored values in another unit.
        harmonized = data.with_harmonized_values().df.reset_index()
        df["value"] = harmonized["harmonized_value"].to_numpy()
        units = harmonized["harmonized_unit"].dropna().astype(str).unique()
        unit = units[0] if len(units) == 1 else (None if not len(units) else "mixed")
        daily["values_unresolved"] = df.assign(
            missing=harmonized["harmonized_unit_source"].astype(str).eq("unresolved").to_numpy()
        ).groupby(keys)["missing"].sum()
    daily["value_unit"] = unit

    # Where the data come from, and how plausible it is.
    if "source_id" in df:
        # Distinct known sources; a day on which no record names its source is not assessed. For redundancy, records
        # without a source form one group of their own, so their coverage is not lost.
        known = df["source_id"].astype(object).where(df["source_id"].notna(), None).to_numpy()[pieces["row"]]
        counted = pd.Series(known, index=pieces.index).groupby([pieces[k] for k in keys]).nunique()
        daily["sources"] = counted.where(counted > 0).reindex(daily.index)
        pieces["source"] = pd.Series(known, index=pieces.index).fillna("unknown").astype(str).to_numpy()
    else:
        daily["sources"] = np.nan
    if "was_user_entered" in df:  # the flag is set only where true
        daily["records_user_entered"] = df.assign(flagged=df["was_user_entered"].eq(True).fillna(False).astype(int)).groupby(
            keys)["flagged"].sum().reindex(daily.index).fillna(0).astype(int)
    else:
        daily["records_user_entered"] = np.nan
    if basis == "local" and "source" in pieces:
        # Minutes recorded by at least two devices: each device's coverage is merged first, so a device overlapping
        # itself is not redundancy, and three devices at once count once, never twice.
        per_source = _union_spans(pieces, keys + ["source"])
        daily["redundant_minutes"] = _minutes_covered_at_least(per_source, keys, 2).reindex(daily.index).fillna(0.0)
    else:
        daily["redundant_minutes"] = np.nan
    below, above = _out_of_range(feature, kind, df, keys, unit, harmonize)
    daily["values_below_range"] = below.reindex(daily.index) if below is not None else np.nan
    daily["values_above_range"] = above.reindex(daily.index) if above is not None else np.nan
    if basis == "local":
        hourly = _hourly_values(hourly, df, kind, feature)

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
PARTICIPANT_TABLES = ("participant_feature", "participant_days", "hourly_profile", "daily_provenance", "daily_curation")
_PARTICIPANT_FEATURE_COLUMNS = ["RegistrationCode", "feature", "days_with_data", "first_day", "last_day", "records",
                                "typical_gap_minutes", "regular_share"]
_DATE_COLUMNS = ("local_date", "first_day", "last_day")
# Semicolon-separated lists: read as text, so a column whose every cell holds one offset is not taken for numbers.
_TEXT_COLUMNS = {"features": str, "utc_offsets": str}


@dataclass
class DailyStatistics:
    """
    The outcome of ``compute_daily_statistics``. In memory, every table is here. When the run wrote to ``out``, the
    per-participant tables stay on disk (read them with ``read_daily`` and ``read_table``) and only
    ``participant_feature`` and the cohort tables are returned.
    """

    phase: str
    daily: dict[str, pd.DataFrame]
    hourly: pd.DataFrame
    participant_feature: pd.DataFrame
    participant_days: pd.DataFrame
    cohort_daily: pd.DataFrame
    active_participants: pd.DataFrame
    errors: list[dict[str, str]]
    run: dict[str, Any]
    provenance: pd.DataFrame = field(default_factory=pd.DataFrame)
    curation: pd.DataFrame = field(default_factory=pd.DataFrame)


def sampling_cadence(data) -> dict[str, float]:
    """
    How regularly a result's records arrive: the median interval between consecutive distinct record starts, and the
    share of intervals within 10% of it. A continuous glucose monitor gives 5 minutes and a share near 1; adaptive
    sources, such as the heart rate, give a low share. Day-key features have no event times, so no cadence.
    """

    if data.load_report["date_column"] != "start_date" or data.df.empty:
        return {"typical_gap_minutes": np.nan, "regular_share": np.nan}
    starts = np.unique(data.df["start_date"].dt.tz_convert(None).to_numpy(dtype="datetime64[us]"))
    gaps = np.diff(starts).astype(np.int64) / 60e6
    gaps = gaps[gaps > 0]
    if not len(gaps):
        return {"typical_gap_minutes": np.nan, "regular_share": np.nan}
    typical = float(np.median(gaps))
    return {"typical_gap_minutes": typical, "regular_share": float(np.mean(np.abs(gaps - typical) <= 0.1 * typical))}


def _participant_task(args) -> dict[str, Any]:
    """Every requested feature for one participant; runs in a worker process when ``workers > 1``."""

    code, phase, root, features, default_inclusion_only, state_validation, harmonize = args
    out = {"code": code, "daily": {}, "hourly": [], "cadence": {}, "provenance": [], "curation": [], "offsets": {},
           "errors": []}
    for feature in features:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # The full projection carries source identifiers, needed to count devices; source names come
                # with them but are never read or written.
                data = _loader(feature, phase, root, state_validation=state_validation).get_data(
                    registration_codes=code, max_rows=None, projection="full",
                    **({"default_inclusion_only": True} if default_inclusion_only else {}))
            if data.df.empty:
                continue
            daily, hourly = summarize_result(data, feature, harmonize=harmonize)
            out["daily"][feature] = daily
            out["hourly"].append(hourly)
            out["cadence"][feature] = sampling_cadence(data)
            provenance, curation = provenance_tables(data, feature)
            out["provenance"].append(provenance)
            out["curation"].append(curation)
            if data.load_report["date_column"] == "start_date" and "utc_offset_minutes" in data.df:
                local, _ = _local_frame(data)
                pairs = pd.DataFrame({"day": local["local_start"].dt.date.to_numpy(),
                                      "offset": pd.to_numeric(data.df["utc_offset_minutes"]).to_numpy()})
                for day, offset in pairs.dropna().drop_duplicates().itertuples(index=False):
                    out["offsets"].setdefault(day, set()).add(int(offset))
        except Exception as exc:  # recorded per participant and feature; the run continues
            out["errors"].append({"RegistrationCode": code, "feature": feature, "error": f"{type(exc).__name__}: {exc}"})
    return out


def _participant_tables(result: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """One participant's rows for the tables beside the daily ones."""

    rows, days = [], {}
    for feature, daily in result["daily"].items():
        cadence = result["cadence"].get(feature, {})
        rows.append({"RegistrationCode": result["code"], "feature": feature, "days_with_data": len(daily),
                     "first_day": daily["local_date"].min(), "last_day": daily["local_date"].max(),
                     "records": int(daily["records"].sum()),
                     "typical_gap_minutes": cadence.get("typical_gap_minutes"), "regular_share": cadence.get("regular_share")})
        for day in daily["local_date"]:
            days.setdefault(day, []).append(feature)
    hourly = [h for h in result["hourly"] if not h.empty]
    provenance = [f for f in result.get("provenance", []) if not f.empty]
    curation = [f for f in result.get("curation", []) if not f.empty]
    offsets = result.get("offsets", {})
    return {
        "participant_feature": pd.DataFrame(rows, columns=_PARTICIPANT_FEATURE_COLUMNS),
        "participant_days": pd.DataFrame(
            [{"RegistrationCode": result["code"], "local_date": day, "features": ";".join(sorted(features)),
              "utc_offsets": ";".join(str(o) for o in sorted(offsets.get(day, ())))}
             for day, features in sorted(days.items())], columns=["RegistrationCode", "local_date", "features", "utc_offsets"]),
        "hourly_profile": pd.concat(hourly, ignore_index=True) if hourly else pd.DataFrame(columns=HOURLY_COLUMNS),
        "daily_provenance": pd.concat(provenance, ignore_index=True) if provenance else pd.DataFrame(columns=PROVENANCE_COLUMNS),
        "daily_curation": pd.concat(curation, ignore_index=True) if curation else pd.DataFrame(columns=CURATION_COLUMNS),
    }


def _cohort_tables(daily_frames: Iterable[tuple[str, pd.DataFrame]],
                   day_frames: Iterable[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cohort tables from participant-grouped frames: participants, records and summed totals per feature and day, and
    participants with any data per day. Each participant's rows arrive in one frame, so counts add across frames.
    """

    participants: Counter = Counter()
    records: Counter = Counter()
    sums: dict[tuple[str, Any], float] = {}
    for feature, frame in daily_frames:
        grouped = frame.groupby("local_date")
        for day, n in grouped["RegistrationCode"].nunique().items():
            participants[(feature, day)] += int(n)
        for day, n in grouped["records"].sum().items():
            records[(feature, day)] += int(n)
        if "value_sum" in frame:
            for day, value in grouped["value_sum"].sum(min_count=1).items():
                if pd.notna(value):
                    sums[(feature, day)] = sums.get((feature, day), 0.0) + float(value)
    cohort = pd.DataFrame(
        [{"feature": f, "local_date": d, "participants": n, "records": records[(f, d)], "value_sum": sums.get((f, d))}
         for (f, d), n in participants.items()],
        columns=["feature", "local_date", "participants", "records", "value_sum"],
    ).sort_values(["feature", "local_date"], ignore_index=True)
    cohort["value_sum"] = pd.to_numeric(cohort["value_sum"])
    cohort["value_sum_per_participant"] = cohort["value_sum"] / cohort["participants"]
    active: Counter = Counter()
    for frame in day_frames:
        for day, n in frame.groupby("local_date")["RegistrationCode"].nunique().items():
            active[day] += int(n)
    return cohort, pd.DataFrame(sorted(active.items()), columns=["local_date", "participants"])


# ------------------------------------------------------------------------------------------ written outputs
def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    for column in _DATE_COLUMNS:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column]).dt.date
    if "RegistrationCode" in frame:
        frame["RegistrationCode"] = frame["RegistrationCode"].astype(str)
    return frame


def _compression(path: Path) -> dict[str, Any] | None:
    """Gzip with a fixed header time: identical results give identical bytes, so a run's files can be checksummed."""

    return {"method": "gzip", "mtime": 0} if str(path).endswith(".gz") else None


def _table_path(out: Path, name: str) -> Path:
    """Participant-level tables are compressed; cohort-level ones are plain CSV."""

    compressed = Path(out) / f"{name}.csv.gz"
    return compressed if name in PARTICIPANT_TABLES or compressed.is_file() else Path(out) / f"{name}.csv"


def _iter_participants(path: Path, chunksize: int):
    """A participant-grouped file, one participant's rows at a time, reading at most ``chunksize`` rows at once."""

    carry = None
    # round_trip parses each float to the value that was written; pandas' default parser can miss by one unit in
    # the last place, which would make statistics read back from files differ from the in-memory ones.
    for chunk in pd.read_csv(path, chunksize=chunksize, keep_default_na=False, na_values=[""], low_memory=False,
                             float_precision="round_trip", dtype=_TEXT_COLUMNS):
        chunk = _typed(chunk)
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last = chunk["RegistrationCode"].iloc[-1]
        carry = chunk[chunk["RegistrationCode"] == last]
        ready = chunk[chunk["RegistrationCode"] != last]
        for _, frame in ready.groupby("RegistrationCode", sort=False):
            yield frame.reset_index(drop=True)
    if carry is not None and len(carry):
        yield carry.reset_index(drop=True)


def iter_daily(out: str | Path, feature: str, chunksize: int = 200_000):
    """A written run's daily table for one feature, one participant at a time, so memory stays bounded."""

    path = Path(out) / "daily" / f"{feature}.csv.gz"
    parameters = Path(out) / "run_parameters.json"
    if parameters.is_file() and feature not in json.loads(parameters.read_text()).get("features", [feature]):
        raise DataLoaderConfigurationError(f"{feature} was not among the features of the run in {out}")
    if not path.is_file():
        return
    yield from _iter_participants(path, chunksize)


def read_daily(out: str | Path, feature: str, participants: Iterable[str] | None = None) -> pd.DataFrame:
    """A written run's daily table for one feature, optionally for some participants only."""

    wanted = None if participants is None else {c if str(c).startswith("10K_") else f"10K_{c}" for c in map(str, participants)}
    frames = [f for f in iter_daily(out, feature) if wanted is None or f["RegistrationCode"].iloc[0] in wanted]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_table(out: str | Path, name: str) -> pd.DataFrame:
    """One of a written run's other tables, by name, for example ``"participant_feature"`` or ``"cohort_daily"``."""

    path = _table_path(Path(out), name)
    if not path.is_file():
        raise DataLoaderConfigurationError(f"{path} does not exist")
    return _typed(pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False, float_precision="round_trip",
                              dtype=_TEXT_COLUMNS))


def export_parquet(out: str | Path) -> list[Path]:
    """
    Write a Parquet copy beside every table of a written run, for faster reading. It needs pyarrow (or fastparquet),
    which the package does not require; the CSV files remain the run's record.
    """
    guard_output(out)

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise DataLoaderConfigurationError("Parquet export needs pyarrow: pip install pyarrow") from exc
    out = Path(out)
    written = []
    for path in sorted(out.glob("daily/*.csv.gz")):
        target = path.with_name(path.name.replace(".csv.gz", ".parquet"))
        read_daily(out, path.name.replace(".csv.gz", "")).to_parquet(target, index=False)
        written.append(target)
    for name in PARTICIPANT_TABLES + ("cohort_daily", "active_participants"):
        if _table_path(out, name).is_file():
            target = out / f"{name}.parquet"
            read_table(out, name).to_parquet(target, index=False)
            written.append(target)
    return written


class _Outputs:
    """
    Writes a run participant by participant, so it can resume after an interruption. A participant counts as done
    once all of its rows are written and its code is appended to ``progress.csv``. Resuming first removes any rows of
    participants that were not done, so every table holds each finished participant exactly once, and then continues
    with the rest. A resumed run must have the same parameters as the interrupted one.
    """

    def __init__(self, directory: Path, parameters: dict[str, Any], resume: bool,
                 tables: tuple[str, ...] = PARTICIPANT_TABLES) -> None:
        self.dir = directory
        self.tables = tables
        marker = directory / "run_parameters.json"
        self.progress = directory / "progress.csv"
        self.headers: dict[Path, list[str]] = {}
        if marker.exists():
            if not resume:
                raise DataLoaderConfigurationError(
                    f"{directory} already holds a statistics run; pass resume=True to continue it, or choose an "
                    "empty directory")
            previous = json.loads(marker.read_text())
            if previous != parameters:
                changed = sorted(k for k in set(previous) | set(parameters) if previous.get(k) != parameters.get(k))
                raise DataLoaderConfigurationError(
                    f"cannot resume {directory}: {', '.join(changed)} differ from the interrupted run's")
            self.completed = set(pd.read_csv(self.progress, dtype=str)["RegistrationCode"]) if self.progress.exists() else set()
            self._repair()
        else:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "daily").mkdir(exist_ok=True)
            marker.write_text(json.dumps(parameters, indent=2, default=str))
            self.progress.write_text("RegistrationCode\n")
            self.completed = set()
        (directory / "daily").mkdir(exist_ok=True)

    def _files(self) -> list[Path]:
        return sorted(self.dir.glob("daily/*.csv.gz")) + [
            self.dir / f"{n}.csv.gz" for n in self.tables] + [self.dir / "errors.csv"]

    def _repair(self) -> None:
        for path in self._files():
            if not path.is_file():
                continue
            kept = pd.read_csv(path, dtype=str, keep_default_na=False)
            kept = kept[kept["RegistrationCode"].isin(self.completed)] if "RegistrationCode" in kept else kept
            # Compression is stated, not inferred from the temporary name, so a .gz file stays gzip.
            temporary = path.with_name(path.name + ".repair")
            kept.to_csv(temporary, index=False, compression=_compression(path))
            temporary.replace(path)

    def _append(self, path: Path, frame: pd.DataFrame) -> None:
        exists = path.is_file() and path.stat().st_size > 0
        if exists and path not in self.headers:
            self.headers[path] = list(pd.read_csv(path, nrows=0).columns)
        if exists and list(frame.columns) != self.headers[path]:
            raise RuntimeError(f"{path.name}: rows with columns {list(frame.columns)} cannot follow {self.headers[path]}")
        frame.to_csv(path, mode="a" if exists else "w", header=not exists, index=False, compression=_compression(path))
        self.headers.setdefault(path, list(frame.columns))

    def write(self, result: dict[str, Any], tables: dict[str, pd.DataFrame]) -> None:
        for feature, daily in result.get("daily", {}).items():
            self._append(self.dir / "daily" / f"{feature}.csv.gz", daily)
        for name, frame in tables.items():
            if not frame.empty:
                self._append(self.dir / f"{name}.csv.gz", frame)
        if result["errors"]:
            self._append(self.dir / "errors.csv", pd.DataFrame(result["errors"], columns=["RegistrationCode", "feature", "error"]))
        with open(self.progress, "a") as handle:  # the participant is done only once this line is written
            handle.write(f"{result['code']}\n")
        self.completed.add(result["code"])


def _provenance(phase: str, root: Path) -> dict[str, Any]:
    """What the run was computed from: versions, registry fingerprints, and the state database's checksum."""

    import hashlib

    from wearable_project.curation.decisions import decisions_fingerprint
    from wearable_project.curation.guidance import GUIDANCE_VERSION, guidance_fingerprint
    from wearable_project.curation.registry import CURATION_REGISTRY_VERSION, registry_fingerprint
    from wearable_project.DataLoaders._base import _PHASE_STATE_FILES
    from wearable_project.processing.registry import REGISTRY_VERSION

    database = root / _PHASE_STATE_FILES[phase]
    checksum = hashlib.sha256(database.read_bytes()).hexdigest() if database.is_file() else None
    return {
        "package": {"version": __version__, "release_label": __release_label__},
        "python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
        "processing_registry": REGISTRY_VERSION, "curation_registry": CURATION_REGISTRY_VERSION,
        "curation_registry_fingerprint": registry_fingerprint(), "guidance": GUIDANCE_VERSION,
        "guidance_fingerprint": guidance_fingerprint(), "decisions_fingerprint": decisions_fingerprint(),
        "state_database": {"file": str(database), "sha256": checksum},
    }


def compute_daily_statistics(
    phase: str = "curated", *, root: str | Path | None = None, features: Iterable[str] | None = None,
    participants: Iterable[str] | None = None, default_inclusion_only: bool = False, harmonize: bool = True,
    state_validation: str = "auto", workers: int = 1, out: str | Path | None = None, resume: bool = False,
) -> DailyStatistics:
    """
    Daily summaries for every participant and feature, streamed one participant at a time.
    ``participants`` defaults to everyone with data. ``default_inclusion_only`` (curated phase) restricts every
    statistic to the policy's default subset. ``harmonize=False`` computes value statistics from stored values rather
    than harmonized ones. Without ``out`` everything is returned in memory, which suits a few participants. With
    ``out``, each participant's rows are written as it finishes, in participant order, and the cohort tables are then
    computed from the written files; ``resume=True`` continues an interrupted run in the same directory, skipping the
    participants it finished. ``workers > 1`` processes participants in parallel. A participant-feature that cannot be
    loaded is recorded in ``errors`` and skipped.
    """

    phase = _check_phase(phase)
    if default_inclusion_only and phase != "curated":
        raise DataLoaderConfigurationError("default_inclusion_only applies to the curated phase only")
    chosen = _features(features)
    root_path = Path(root).expanduser() if root is not None else Path(DEFAULT_CURATED_ROOT if phase == "curated" else DEFAULT_NATIVE_ROOT)
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    _remember_root(root_path)
    out_dir = _guard_output(Path(out), [root_path]) if out is not None else None
    if participants is None:
        coverage = compute_coverage(phase, root=root_path, features=chosen)
        codes = sorted(coverage.participant_feature["RegistrationCode"].unique())
    else:
        codes = sorted({c if c.startswith("10K_") else f"10K_{c}" for c in (str(p).strip() for p in participants)})
    parameters = {"phase": phase, "root": str(root_path.resolve()), "features": list(chosen), "participants": codes,
                  "default_inclusion_only": default_inclusion_only, "harmonize": harmonize,
                  "state_validation": state_validation}
    outputs = _Outputs(out_dir, parameters, resume) if out_dir is not None else None
    pending = [c for c in codes if outputs is None or c not in outputs.completed]
    tasks = [(code, phase, root_path, chosen, default_inclusion_only, state_validation, harmonize) for code in pending]

    memory: dict[str, list] = {"daily": [], "tables": {n: [] for n in PARTICIPANT_TABLES}, "errors": []}

    def consume(result: dict[str, Any]) -> None:
        tables = _participant_tables(result)
        if outputs is not None:
            outputs.write(result, tables)
            return
        memory["errors"].extend(result["errors"])
        memory["daily"].extend(result["daily"].items())
        for name, frame in tables.items():
            memory["tables"][name].append(frame)

    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(_participant_task, tasks, chunksize=1):  # results arrive in participant order
                consume(result)
    else:
        for task in tasks:
            consume(_participant_task(task))

    if outputs is None:
        tables = {n: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                  for n, frames in memory["tables"].items()}
        daily = {}
        for feature, frame in memory["daily"]:
            daily.setdefault(feature, []).append(frame)
        daily = {f: pd.concat(daily[f], ignore_index=True) for f in chosen if f in daily}
        cohort_daily, active = _cohort_tables(
            ((f, frame) for f, frame in memory["daily"]), [tables["participant_days"]] if not tables["participant_days"].empty else [])
        errors = memory["errors"]
        participant_feature = tables["participant_feature"] if not tables["participant_feature"].empty else pd.DataFrame(
            columns=_PARTICIPANT_FEATURE_COLUMNS)
    else:
        daily_frames = ((f, frame) for f in chosen for frame in iter_daily(out_dir, f))
        day_path = _table_path(out_dir, "participant_days")
        day_frames = _iter_participants(day_path, 200_000) if day_path.is_file() else []
        cohort_daily, active = _cohort_tables(daily_frames, day_frames)
        cohort_daily.to_csv(out_dir / "cohort_daily.csv", index=False)
        active.to_csv(out_dir / "active_participants.csv", index=False)
        errors_path = out_dir / "errors.csv"
        errors = pd.read_csv(errors_path, dtype=str).to_dict("records") if errors_path.is_file() else []
        feature_path = _table_path(out_dir, "participant_feature")
        participant_feature = read_table(out_dir, "participant_feature") if feature_path.is_file() else pd.DataFrame(
            columns=_PARTICIPANT_FEATURE_COLUMNS)
        tables = {"hourly_profile": pd.DataFrame(), "participant_days": pd.DataFrame(),
                  "daily_provenance": pd.DataFrame(), "daily_curation": pd.DataFrame()}
        daily = {}

    with_data = set(participant_feature["RegistrationCode"]) if len(participant_feature) else set()
    run = {"tool": "data_statistics", **_provenance(phase, root_path),
           "participants_without_data": sorted(set(codes) - with_data), "parameters": {
               **{k: v for k, v in parameters.items() if k != "participants"}, "participants": len(codes)},
           "workers": workers, "resumed": bool(outputs is not None and len(pending) < len(codes)),
           "participants_processed_now": len(pending), "errors": len(errors),
           "measurement_kinds": {f: measurement_kind(f) for f in chosen},
           "units": {f: column_units(f) for f in chosen},
           "started": started.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
           "seconds": round(time.perf_counter() - clock, 2)}
    if out_dir is not None:
        (out_dir / "run.json").write_text(json.dumps(run, indent=2, default=str))
    return DailyStatistics(phase, daily, tables["hourly_profile"], participant_feature, tables["participant_days"],
                           cohort_daily, active, errors, run, tables["daily_provenance"], tables["daily_curation"])


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
    parser.add_argument("--resume", action="store_true", help="continue an interrupted daily run in --out")
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
                workers=args.workers, out=out, resume=args.resume)
            print(json.dumps(result.run, indent=2, default=str))
    except DataLoaderConfigurationError as exc:
        parser.error(str(exc))
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
