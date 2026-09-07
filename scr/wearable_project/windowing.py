"""
Wearable Data Processing and Modeling project
Scientifically explicit fixed-width windowing for interval and point events.
"""

from __future__ import annotations
from collections.abc import Mapping
import logging
from typing import Any
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick
from .cleaning import value_columns
from .exceptions import ConfigurationError, SchemaError
from .policies import FeaturePolicy, resolve_feature_policy
from .utils import canonical_cell, stable_json

LOGGER= logging.getLogger(__name__)

_INTERNAL_COLUMNS= {
    "_event_id", "_overlap_start", "_overlap_end", "_overlap_seconds", "_allocation_fraction", "_weight",
}


def fixed_window_timedelta(frequency:str) -> pd.Timedelta:
    """Validate a fixed-width frequency and return its duration."""

    try:
        offset= to_offset(frequency)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid window frequency {frequency!r}: {exc}") from exc
    if not isinstance(offset, Tick):
        raise ConfigurationError(
            f"Window frequency {frequency!r} is calendar-based; use a fixed duration such as '5min' or '1h'."
        )
    return pd.Timedelta(offset.nanos, unit="ns")


def event_window_overlaps(start:Any, end:Any, frequency:str= "5min", *, max_windows:int= 100_000,) -> pd.DataFrame:
    """
    Return exact overlap between one event and every fixed window it touches.
    Intervals use half-open semantics ``[start, end)``. Point events (``start == end``) are assigned to their
    containing window with zero observed duration and allocation fraction one.
    """

    start_ts= pd.Timestamp(start)
    end_ts= pd.Timestamp(end)
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise SchemaError("Windowing requires non-missing start and end timestamps.")
    if start_ts.tzinfo is None or end_ts.tzinfo is None:
        raise SchemaError("Windowing requires timezone-aware timestamps.")
    end_ts= end_ts.tz_convert(start_ts.tz)
    if end_ts < start_ts:
        raise SchemaError(f"Event end {end_ts} precedes start {start_ts}.")

    width= fixed_window_timedelta(frequency)
    first_start= start_ts.floor(frequency)

    if end_ts == start_ts:
        return pd.DataFrame({
            "start_date": [first_start],
            "end_date": [first_start + width],
            "overlap_start": [start_ts],
            "overlap_end": [end_ts],
            "overlap_seconds": [0.0],
            "allocation_fraction": [1.0],
        })

    final_end= end_ts.ceil(frequency)
    if final_end <= first_start:
        final_end= first_start + width
    estimated= int((final_end - first_start) / width)
    if estimated > max_windows:
        raise ConfigurationError(f"Event spans {estimated:,} windows, exceeding max_windows={max_windows:,}.")

    boundaries= pd.date_range(start=first_start, end=final_end, freq=frequency)
    total_seconds= (end_ts - start_ts).total_seconds()
    rows:list[dict[str, Any]]= []
    for window_start, window_end in zip(boundaries[:-1], boundaries[1:], strict=True):
        overlap_start= max(start_ts, window_start)
        overlap_end= min(end_ts, window_end)
        overlap_seconds= (overlap_end - overlap_start).total_seconds()
        if overlap_seconds <= 0:
            continue
        rows.append({
            "start_date": window_start,
            "end_date": window_end,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "overlap_seconds": float(overlap_seconds),
            "allocation_fraction": float(overlap_seconds / total_seconds),
        })
    return pd.DataFrame(rows)


def _numeric_view(series:pd.Series) -> tuple[pd.Series, bool]:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float), True
    converted= pd.to_numeric(series, errors="coerce")
    original_non_missing= int(series.notna().sum())
    converted_non_missing= int(converted.notna().sum())
    if original_non_missing == converted_non_missing:
        return converted.astype(float), True
    return series, False


def _mode_any(series:pd.Series) -> Any:
    non_missing= series.loc[series.notna()]
    if non_missing.empty:
        return pd.NA
    canonical= non_missing.map(canonical_cell)
    counts= canonical.value_counts(dropna=False, sort=True)
    winner= counts.index[0]
    return non_missing.loc[canonical.eq(winner)].iloc[0]


def aggregate_values_by_type(series:pd.Series, aggregation:str= "median") -> Any:
    """ Aggregate numeric, categorical, and nested values without unhashable-mode errors. """

    numeric, is_numeric= _numeric_view(series)
    if is_numeric:
        if aggregation == "sum":
            return numeric.sum(min_count=1)
        if aggregation == "mean" or aggregation == "weighted_mean":
            return numeric.mean()
        if aggregation == "median":
            return numeric.median()
        if aggregation == "mode":
            return _mode_any(numeric)
        raise ConfigurationError(f"Unsupported aggregation {aggregation!r}.")
    return _mode_any(series)


def _weighted_mean(values:pd.Series, weights:pd.Series) -> float:
    numeric, is_numeric= _numeric_view(values)
    if not is_numeric:
        return float("nan")
    valid= numeric.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return float(numeric.mean())
    return float(np.average(numeric.loc[valid], weights=weights.loc[valid]))


def _union_seconds(starts:pd.Series, ends:pd.Series) -> float:
    intervals= sorted((
        (pd.Timestamp(start), pd.Timestamp(end))
        for start, end in zip(starts, ends, strict=True)
        if pd.notna(start) and pd.notna(end) and end > start
    ), key=lambda item: (item[0], item[1]),)
    if not intervals:
        return 0.0
    total= 0.0
    current_start, current_end= intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end= max(current_end, end)
        else:
            total += (current_end - current_start).total_seconds()
            current_start, current_end= start, end
    total += (current_end - current_start).total_seconds()
    return float(total)


def _consistent_or_explicitly_mixed(series:pd.Series) -> Any:
    values= series.loc[series.notna()]
    if values.empty:
        return pd.NA
    canonical= values.map(canonical_cell)
    unique_keys= canonical.drop_duplicates()
    if len(unique_keys) == 1:
        return values.iloc[0]
    # Preserve the fact that metadata disagreed instead of silently choosing the first
    # device/unit/source. This remains machine-readable when exported as CSV.
    unique_values= [values.loc[canonical.eq(key)].iloc[0] for key in unique_keys]
    return stable_json(unique_values)


def window_feature(frame:pd.DataFrame, feature_name:str, frequency:str= "5min", *, policy:FeaturePolicy | None= None,
                   policy_overrides:Mapping[str, FeaturePolicy | Mapping[str, Any]] | None= None,
                   max_windows_per_event:int= 100_000,) -> pd.DataFrame:
    """
    Place a normalized feature frame into fixed windows.
    Cumulative numeric values are allocated by exact temporal overlap and then summed, so their total is conserved.
    Discrete values are never divided across windows.
    """

    required= {"start_date", "end_date"}
    missing= required.difference(frame.columns)
    if missing:
        raise SchemaError(f"Cannot window {feature_name!r}; missing columns {sorted(missing)}.")
    chosen_policy= policy or resolve_feature_policy(feature_name, policy_overrides)
    if not chosen_policy.windowable:
        LOGGER.info("Feature %s is configured as non-windowable; preserving raw records.", feature_name)
        result= frame.copy()
        result["is_windowed"]= False
        return result
    if frame.empty:
        result= frame.copy()
        for column, dtype in (
            ("observed_duration_seconds", "float64"),
            ("feature_observed_duration_seconds", "float64"),
            ("overlap_seconds_sum", "float64"),
            ("event_count", "int64"),
            ("is_windowed", "bool"),
        ):
            result[column]= pd.Series(dtype=dtype)
        return result

    values= value_columns(frame)
    if not values:
        raise SchemaError(
            f"Cannot window {feature_name!r}: no supported value column was found. "
            "Expected 'value' or a known multi-value HealthKit schema."
        )
    working= frame.copy()
    numeric_columns:dict[str, bool]= {}
    for column in values:
        original_non_missing= int(working[column].notna().sum())
        converted_candidate= pd.to_numeric(working[column], errors="coerce")
        converted_non_missing= int(converted_candidate.notna().sum())
        if 0 < converted_non_missing < original_non_missing:
            invalid_count= original_non_missing - converted_non_missing
            raise SchemaError(
                f"Feature {feature_name!r} column {column!r} mixes numeric and "
                f"non-numeric values ({invalid_count} non-numeric records)."
            )
        converted, is_numeric= _numeric_view(working[column])
        numeric_columns[column]= is_numeric
        if is_numeric:
            working[column]= converted
        elif chosen_policy.aggregation != "mode":
            raise SchemaError(
                f"Feature {feature_name!r} column {column!r} is non-numeric, but "
                f"its policy requests aggregation={chosen_policy.aggregation!r}. "
                "Use aggregation='mode' or mark the feature non-windowable."
            )

    # Never combine incompatible units into one numeric aggregate.  Multiple units
    # remain as separate rows for the same time window so downstream code must make
    # an explicit conversion or selection.
    unit_columns= [column for column in ("unit", "units") if column in working.columns]
    for column in unit_columns:
        normalized= working[column].astype("string").str.strip()
        working[column]= normalized.mask(normalized.eq(""), pd.NA)

    # HealthKit can contain overlapping samples from multiple apps/devices. Keep
    # pseudonymous sources as separate partitions so the parser never silently sums
    # or averages across sources before the study defines an adjudication rule.
    partition_columns= [*unit_columns]
    if "source_key" in working.columns:
        partition_columns.append("source_key")

    metadata_columns= [
        column
        for column in working.columns if column not in {"start_date", "end_date", *values, *partition_columns}
    ]
    expanded_rows:list[dict[str, Any]]= []
    for event_id, row in working.iterrows():
        overlaps= event_window_overlaps(
            row["start_date"], row["end_date"], frequency, max_windows=max_windows_per_event,
        )
        for overlap in overlaps.itertuples(index=False):
            record:dict[str, Any]= {
                "start_date": overlap.start_date,
                "end_date": overlap.end_date,
                "_overlap_start": overlap.overlap_start,
                "_overlap_end": overlap.overlap_end,
                "_overlap_seconds": overlap.overlap_seconds,
                "_allocation_fraction": overlap.allocation_fraction,
                "_weight": overlap.overlap_seconds if overlap.overlap_seconds > 0 else 1.0,
                "_event_id": event_id,
            }
            for column in values:
                value= row[column]
                if chosen_policy.cumulative and numeric_columns[column] and pd.notna(value):
                    record[column]= float(value) * overlap.allocation_fraction
                else:
                    record[column]= value
            for column in partition_columns:
                record[column]= row[column]
            for column in metadata_columns:
                record[column]= row[column]
            expanded_rows.append(record)

    if not expanded_rows:
        columns= list(frame.columns)
        for column in (
            "observed_duration_seconds", "feature_observed_duration_seconds", "overlap_seconds_sum",
            "event_count", "is_windowed",
        ):
            if column not in columns:
                columns.append(column)
        return pd.DataFrame(columns=columns)

    expanded= pd.DataFrame(expanded_rows)
    feature_coverage_by_window= {
        key: _union_seconds(group["_overlap_start"], group["_overlap_end"])
        for key, group in expanded.groupby(["start_date", "end_date"], sort=False, dropna=False)
    }
    output_rows:list[dict[str, Any]]= []
    group_columns= ["start_date", "end_date", *partition_columns]
    grouped= expanded.groupby(group_columns, sort=True, dropna=False)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key= (group_key,)
        window_start, window_end, *partition_values= group_key
        output:dict[str, Any]= {
            "start_date": window_start,
            "end_date": window_end,
            "observed_duration_seconds": _union_seconds(group["_overlap_start"], group["_overlap_end"]),
            "feature_observed_duration_seconds": feature_coverage_by_window[(window_start, window_end)],
            "overlap_seconds_sum": float(group["_overlap_seconds"].sum()),
            "event_count": int(group["_event_id"].nunique()),
            "is_windowed": True,
        }
        output.update(dict(zip(partition_columns, partition_values, strict=True)))
        for column in values:
            if numeric_columns[column]:
                if chosen_policy.cumulative or chosen_policy.aggregation == "sum":
                    output[column]= group[column].sum(min_count=1)
                elif chosen_policy.aggregation == "weighted_mean":
                    output[column]= _weighted_mean(group[column], group["_weight"])
                elif chosen_policy.aggregation == "mean":
                    output[column]= group[column].mean()
                elif chosen_policy.aggregation == "median":
                    output[column]= group[column].median()
                else:
                    output[column]= _mode_any(group[column])
            else:
                output[column]= _mode_any(group[column])
        for column in metadata_columns:
            output[column]= _consistent_or_explicitly_mixed(group[column])
        output_rows.append(output)

    result= pd.DataFrame(output_rows)
    preferred= [
        "start_date", "end_date", *partition_columns, *values, "observed_duration_seconds",
        "feature_observed_duration_seconds", "overlap_seconds_sum", "event_count", "is_windowed",
    ]
    ordered= [column for column in preferred if column in result.columns]
    ordered.extend(column for column in result.columns if column not in ordered)
    return result.loc[:, ordered].sort_values(["start_date", "end_date"], kind="mergesort").reset_index(drop=True)
