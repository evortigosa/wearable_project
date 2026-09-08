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
from ..exceptions import ConfigurationError, SchemaError
from .policies import FeaturePolicy, resolve_feature_policy
from ..utils.serialization import canonical_cell, stable_json

LOGGER= logging.getLogger(__name__)
_INTERNAL_COLUMNS= {
    "_event_id", "_group_id", "_window_group_id", "_overlap_start", "_overlap_end", "_overlap_seconds",
    "_allocation_fraction", "_weight",
}


def fixed_window_timedelta(frequency:str) -> pd.Timedelta:
    """ Validate a fixed-width frequency and return its duration. """
    try:
        offset= to_offset(frequency)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid window frequency {frequency!r}: {exc}") from exc
    if not isinstance(offset, Tick):
        raise ConfigurationError(
            f"Window frequency {frequency!r} is calendar-based; use a fixed duration such as '5min' or '1h'."
        )
    duration= pd.Timedelta(offset.nanos, unit="ns")
    if duration <= pd.Timedelta(0):
        raise ConfigurationError("Window frequency must be positive.")
    return duration


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
        if aggregation in {"mean", "weighted_mean"}:
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


def _timezone_aware_series(values:pd.Series, *, name:str) -> pd.Series:
    """ Return a timezone-aware series while retaining one common timezone. """
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        result= values.copy()
    else:
        try:
            result= pd.to_datetime(values, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            result= pd.to_datetime(values, errors="coerce")
    if not isinstance(result.dtype, pd.DatetimeTZDtype):
        raise SchemaError(f"Windowing requires timezone-aware {name} timestamps.")
    if result.isna().any():
        raise SchemaError(f"Windowing requires non-missing {name} timestamps.")
    return result


def _timestamp_series_from_ns(values:np.ndarray, timezone:Any) -> pd.Series:
    index= pd.to_datetime(values, utc=True)
    if str(timezone) != "UTC":
        index= index.tz_convert(timezone)
    return pd.Series(index)


def _expand_events_vectorized(working:pd.DataFrame, *, frequency:str, value_names:list[str],
                              numeric_columns:Mapping[str, bool], policy:FeaturePolicy,
                              max_windows_per_event:int,) -> pd.DataFrame:
    """ Expand events into intersected windows using vectorized index arithmetic. """
    starts= _timezone_aware_series(working["start_date"], name="start")
    timezone= starts.dt.tz
    ends= _timezone_aware_series(working["end_date"], name="end").dt.tz_convert(timezone)
    invalid_order= ends.lt(starts)
    if invalid_order.any():
        raise SchemaError(f"Windowing received {int(invalid_order.sum())} event(s) whose end precedes start.")

    width= fixed_window_timedelta(frequency)
    width_ns= int(width.value)
    first_windows= starts.dt.floor(frequency)
    final_boundaries= ends.dt.ceil(frequency)
    point_events= ends.eq(starts).to_numpy(dtype=bool)

    first_ns= first_windows.astype("int64").to_numpy()
    final_ns= final_boundaries.astype("int64").to_numpy()
    counts= np.floor_divide(final_ns - first_ns, width_ns).astype(np.int64)
    counts[point_events]= 1
    if (counts <= 0).any():
        raise SchemaError("Window expansion produced a non-positive window count.")
    largest= int(counts.max(initial=0))
    if largest > max_windows_per_event:
        event_position= int(np.argmax(counts))
        raise ConfigurationError(
            f"Event at row {event_position} spans {largest:,} windows, exceeding "
            f"max_windows_per_event={max_windows_per_event:,}."
        )

    total_rows= int(counts.sum())
    event_positions= np.repeat(np.arange(len(working), dtype=np.int64), counts)
    repeated_block_starts= np.repeat(np.cumsum(counts) - counts, counts)
    window_offsets= np.arange(total_rows, dtype=np.int64) - repeated_block_starts
    window_start_ns= first_ns[event_positions] + window_offsets * width_ns
    window_end_ns= window_start_ns + width_ns

    start_ns= starts.astype("int64").to_numpy()[event_positions]
    end_ns= ends.astype("int64").to_numpy()[event_positions]
    overlap_start_ns= np.maximum(start_ns, window_start_ns)
    overlap_end_ns= np.minimum(end_ns, window_end_ns)
    overlap_ns= np.maximum(overlap_end_ns - overlap_start_ns, 0)

    repeated_points= point_events[event_positions]
    valid= (overlap_ns > 0)|repeated_points
    if not bool(np.all(valid)):
        event_positions= event_positions[valid]
        window_start_ns= window_start_ns[valid]
        window_end_ns= window_end_ns[valid]
        overlap_start_ns= overlap_start_ns[valid]
        overlap_end_ns= overlap_end_ns[valid]
        overlap_ns= overlap_ns[valid]
        repeated_points= repeated_points[valid]

    duration_ns= (ends.astype("int64").to_numpy() - starts.astype("int64").to_numpy())[event_positions]
    allocation_fraction= np.ones(len(event_positions), dtype=float)
    interval_mask= ~repeated_points
    allocation_fraction[interval_mask]= (
        overlap_ns[interval_mask].astype(float) / duration_ns[interval_mask].astype(float)
    )
    overlap_seconds= overlap_ns.astype(float) / 1_000_000_000.0

    expanded= working.iloc[event_positions].reset_index(drop=True).copy()
    expanded["start_date"]= _timestamp_series_from_ns(window_start_ns, timezone)
    expanded["end_date"]= _timestamp_series_from_ns(window_end_ns, timezone)
    expanded["_overlap_start"]= _timestamp_series_from_ns(overlap_start_ns, timezone)
    expanded["_overlap_end"]= _timestamp_series_from_ns(overlap_end_ns, timezone)
    expanded["_overlap_seconds"]= overlap_seconds
    expanded["_allocation_fraction"]= allocation_fraction
    expanded["_weight"]= np.where(overlap_seconds > 0, overlap_seconds, 1.0)
    expanded["_event_id"]= event_positions

    if policy.cumulative:
        for column in value_names:
            if numeric_columns[column]:
                expanded[column]= expanded[column].astype(float) * allocation_fraction
    return expanded


def _union_seconds_by_group(group_ids:pd.Series, starts:pd.Series, ends:pd.Series,) -> pd.Series:
    """ Calculate interval-union duration for each integer group without Python loops. """
    intervals= pd.DataFrame({
        "_group": group_ids.to_numpy(dtype=np.int64),
        "_start": starts.astype("int64").to_numpy(),
        "_end": ends.astype("int64").to_numpy(),
    }).sort_values(["_group", "_start", "_end"], kind="mergesort")
    running_end= intervals.groupby("_group", sort=False)["_end"].cummax()
    previous_end= running_end.groupby(intervals["_group"], sort=False).shift()
    effective_start= intervals["_start"].to_numpy(copy=True)
    has_previous= previous_end.notna().to_numpy()
    effective_start[has_previous]= np.maximum(
        effective_start[has_previous], previous_end.loc[has_previous].to_numpy(dtype=np.int64),
    )
    contribution= np.maximum(intervals["_end"].to_numpy() - effective_start, 0,)
    return (
        pd.Series(contribution, index=intervals.index)
        .groupby(intervals["_group"], sort=True)
        .sum()
        .astype(float)
        / 1_000_000_000.0
    )


def _weighted_mean_by_group(values:pd.Series, weights:pd.Series, group_ids:pd.Series,) -> pd.Series:
    numeric= pd.to_numeric(values, errors="coerce")
    valid= numeric.notna() & weights.notna() & weights.gt(0)
    numerator= (numeric * weights).where(valid).groupby(group_ids, sort=True).sum(min_count=1)
    denominator= weights.where(valid).groupby(group_ids, sort=True).sum(min_count=1)
    weighted= numerator / denominator
    fallback= numeric.groupby(group_ids, sort=True).mean()
    return weighted.fillna(fallback)


def _aggregate_metadata_column(values:pd.Series, group_ids:pd.Series, group_index:pd.Index,) -> pd.Series:
    """ Keep one value when consistent; serialize the unique set when it disagrees. """
    result= pd.Series(pd.NA, index=group_index, dtype="object")
    non_missing= values.notna()
    if not non_missing.any():
        return result

    valid_values= values.loc[non_missing]
    valid_groups= group_ids.loc[non_missing]
    first= valid_values.groupby(valid_groups, sort=True).first()
    result.loc[first.index]= first.astype("object")

    canonical= valid_values.map(canonical_cell)
    unique_rows= pd.DataFrame({
        "_group": valid_groups.to_numpy(dtype=np.int64),
        "_canonical": canonical.to_numpy(dtype=object),
        "_value": valid_values.to_numpy(dtype=object),
    }).drop_duplicates(["_group", "_canonical"], keep="first")
    counts= unique_rows.groupby("_group", sort=True).size()
    mixed_groups= counts.index[counts.gt(1)]
    if len(mixed_groups):
        mixed= unique_rows.loc[unique_rows["_group"].isin(mixed_groups)]
        serialized= mixed.groupby("_group", sort=True)["_value"].agg(lambda group: stable_json(group.tolist()))
        result.loc[serialized.index]= serialized
    return result


def window_feature(frame:pd.DataFrame, feature_name:str, frequency:str= "5min", *, policy:FeaturePolicy|None= None,
                   policy_overrides:Mapping[str, FeaturePolicy|Mapping[str, Any]]|None= None,
                   max_windows_per_event:int= 100_000,) -> pd.DataFrame:
    """
    Place a normalized feature frame into fixed windows. Cumulative numeric values are allocated by exact temporal
    overlap and then summed, so their total is conserved. Discrete values are repeated into intersected windows
    and aggregated according to the feature policy. Point events are assigned exactly once. Units, source
    partitions, and manual-entry status are never merged silently.
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

    unit_columns= [column for column in ("unit", "units") if column in working.columns]
    for column in unit_columns:
        normalized= working[column].astype("string").str.strip()
        working[column]= normalized.mask(normalized.eq(""), pd.NA)

    partition_columns= [*unit_columns]
    if "source_key" in working.columns:
        partition_columns.append("source_key")
    if "is_user_entered" in working.columns:
        partition_columns.append("is_user_entered")

    metadata_columns= [
        column
        for column in working.columns
        if column not in {"start_date", "end_date", *values, *partition_columns}
    ]
    expanded= _expand_events_vectorized(
        working, frequency=frequency, value_names=values, numeric_columns=numeric_columns, policy=chosen_policy,
        max_windows_per_event=max_windows_per_event,
    )
    if expanded.empty:
        columns= list(frame.columns)
        for column in (
            "observed_duration_seconds",
            "feature_observed_duration_seconds",
            "overlap_seconds_sum",
            "event_count",
            "is_windowed",
        ):
            if column not in columns:
                columns.append(column)
        return pd.DataFrame(columns=columns)

    group_columns= ["start_date", "end_date", *partition_columns]
    expanded["_group_id"]= expanded.groupby(
        group_columns, sort=True, dropna=False, observed=True,
    ).ngroup()
    expanded["_window_group_id"]= expanded.groupby(
        ["start_date", "end_date"], sort=True, dropna=False, observed=True,
    ).ngroup()

    group_keys= (
        expanded[["_group_id", *group_columns]]
        .drop_duplicates("_group_id", keep="first")
        .sort_values("_group_id")
        .set_index("_group_id")
    )
    group_index= group_keys.index
    output= group_keys.copy()

    observed= _union_seconds_by_group(
        expanded["_group_id"], expanded["_overlap_start"], expanded["_overlap_end"],
    )
    feature_observed= _union_seconds_by_group(
        expanded["_window_group_id"], expanded["_overlap_start"], expanded["_overlap_end"],
    )
    group_to_window= (
        expanded[["_group_id", "_window_group_id"]]
        .drop_duplicates("_group_id", keep="first")
        .set_index("_group_id")["_window_group_id"]
    )

    output["observed_duration_seconds"]= observed.reindex(group_index).fillna(0.0)
    output["feature_observed_duration_seconds"]= (
        group_to_window.reindex(group_index).map(feature_observed).fillna(0.0)
    )
    output["overlap_seconds_sum"]= (
        expanded["_overlap_seconds"]
        .groupby(expanded["_group_id"], sort=True)
        .sum()
        .reindex(group_index)
        .fillna(0.0)
    )
    output["event_count"]= (
        expanded["_event_id"]
        .groupby(expanded["_group_id"], sort=True)
        .nunique()
        .reindex(group_index)
        .fillna(0)
        .astype("int64")
    )
    output["is_windowed"]= True

    for column in values:
        grouped_values= expanded[column].groupby(expanded["_group_id"], sort=True)
        if numeric_columns[column]:
            if chosen_policy.cumulative or chosen_policy.aggregation == "sum":
                aggregated= grouped_values.sum(min_count=1)
            elif chosen_policy.aggregation == "weighted_mean":
                aggregated= _weighted_mean_by_group(
                    expanded[column], expanded["_weight"], expanded["_group_id"]
                )
            elif chosen_policy.aggregation == "mean":
                aggregated= grouped_values.mean()
            elif chosen_policy.aggregation == "median":
                aggregated= grouped_values.median()
            else:
                aggregated= grouped_values.agg(_mode_any)
        else:
            aggregated= grouped_values.agg(_mode_any)
        output[column]= aggregated.reindex(group_index)

    for column in metadata_columns:
        output[column]= _aggregate_metadata_column(
            expanded[column], expanded["_group_id"], group_index
        )

    result= output.reset_index(drop=True)
    preferred= [
        "start_date", "end_date", *partition_columns, *values, "observed_duration_seconds",
        "feature_observed_duration_seconds", "overlap_seconds_sum", "event_count", "is_windowed",
    ]
    ordered= [column for column in preferred if column in result.columns]
    ordered.extend(column for column in result.columns if column not in ordered)
    return (
        result.loc[:, ordered]
        .sort_values(["start_date", "end_date", *partition_columns], kind="mergesort")
        .reset_index(drop=True)
    )
