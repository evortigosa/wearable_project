"""
Wearable Data Processing and Modeling project
Non-destructive quality and coverage statistics for processed wearable data.
"""

from __future__ import annotations
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Literal
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..processing.cleaning import normalize_datetime_series
from ..exceptions import ConfigurationError
from ..processing.policies import canonical_feature_name
from ..processing.pipeline import MANIFEST_NAME, QUALITY_REPORT_NAME
from .filesystem import atomic_write_dataframe, atomic_write_json
from .runtime import cap_workers, ensure_disjoint_roots, utc_now_iso

LOGGER= logging.getLogger(__name__)
DayBasis= Literal["utc", "timezone", "source_offset"]


@dataclass(slots=True)
class StatisticsConfig:
    """ Controls date grouping and participant-level parallelism. """
    max_workers:int|None= 4
    day_basis:DayBasis= "utc"
    timezone:str= "UTC"

    def __post_init__(self) -> None:
        if self.day_basis not in {"utc", "timezone", "source_offset"}:
            raise ConfigurationError("day_basis must be 'utc', 'timezone', or 'source_offset'.")
        cap_workers(self.max_workers)
        if self.day_basis == "timezone":
            try:
                pd.Timestamp("2024-01-01", tz=self.timezone)
            except Exception as exc:
                raise ConfigurationError(f"Invalid timezone {self.timezone!r}: {exc}") from exc


def _read_feature_map(participant_dir:Path) -> dict[str, Path]:
    participant_root= participant_dir.resolve()
    manifest_path= participant_dir / MANIFEST_NAME
    try:
        manifest= json.loads(manifest_path.read_text(encoding="utf-8"))
        features= manifest.get("features", {})
        mapped:dict[str, Path]= {}
        for feature, details in features.items():
            if not isinstance(details, dict) or not isinstance(details.get("file"), str):
                continue
            candidate= (participant_dir / details["file"]).resolve()
            if candidate.is_relative_to(participant_root) and candidate.is_file():
                mapped[str(feature)]= candidate
        if mapped:
            return mapped
    except (OSError, json.JSONDecodeError, TypeError):
        pass

    ignored= {QUALITY_REPORT_NAME, MANIFEST_NAME}
    mapped:dict[str, Path]= {}
    for path in sorted(participant_dir.iterdir()):
        if not path.is_file() or path.name in ignored:
            continue
        if path.suffix.casefold() not in {".csv", ".parquet"}:
            continue
        if path.name.startswith(("logfile", "statistics_", "processing_")):
            continue
        resolved= path.resolve()
        if resolved.is_relative_to(participant_root):
            mapped[path.stem]= resolved
    return mapped


def _read_table(path:Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.casefold() == ".parquet" else pd.read_csv(path, low_memory=False)


def _local_clock(timestamps:pd.Series, *, config:StatisticsConfig, offsets:pd.Series|None= None,) -> pd.Series:
    parsed, _, _= normalize_datetime_series(timestamps, naive_timezone="UTC")
    if config.day_basis == "utc":
        return parsed
    if config.day_basis == "timezone":
        return parsed.dt.tz_convert(config.timezone)
    numeric_offsets= (
        pd.to_numeric(offsets, errors="coerce")
        if offsets is not None
        else pd.Series(np.nan, index=parsed.index)
    )
    # Retaining UTC tzinfo while shifting the clock is deliberate: only local date/hour
    # components are consumed, and heterogeneous fixed offsets cannot share one dtype.
    return parsed + pd.to_timedelta(numeric_offsets.fillna(0), unit="m")


def _interval_timezone(config:StatisticsConfig, offset_minutes:float|None) -> Any:
    if config.day_basis == "utc":
        return timezone.utc
    if config.day_basis == "timezone":
        return config.timezone
    if offset_minutes is None or not np.isfinite(offset_minutes):
        return timezone.utc
    return timezone(timedelta(minutes=float(offset_minutes)))


def _split_interval_by_local_day(start:pd.Timestamp, end:pd.Timestamp, *, config:StatisticsConfig,
                                 offset_minutes:float|None,) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    if pd.isna(start) or pd.isna(end) or end <= start:
        return []
    tz= _interval_timezone(config, offset_minutes)
    local_start= start.tz_convert(tz)
    local_end= end.tz_convert(tz)
    pieces:list[tuple[str, pd.Timestamp, pd.Timestamp]]= []
    cursor= local_start
    while cursor < local_end:
        next_date= cursor.date() + timedelta(days=1)
        boundary= pd.Timestamp(datetime.combine(next_date, time.min), tz=tz)
        piece_end= min(local_end, boundary)
        pieces.append((
            cursor.date().isoformat(),
            cursor.tz_convert("UTC"),
            piece_end.tz_convert("UTC"),
        ))
        cursor= piece_end
    return pieces


def _union_interval_seconds(intervals:Iterable[tuple[pd.Timestamp, pd.Timestamp]]) -> float:
    ordered= sorted(intervals, key=lambda pair: (pair[0], pair[1]))
    if not ordered:
        return 0.0
    total= 0.0
    current_start, current_end= ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end= max(current_end, end)
        else:
            total += (current_end - current_start).total_seconds()
            current_start, current_end= start, end
    total += (current_end - current_start).total_seconds()
    return float(total)


def coverage_by_day(frame:pd.DataFrame, *, config:StatisticsConfig|None= None,) -> pd.DataFrame:
    """ Compute non-overlapping temporal coverage per local day for one feature. """

    config= config or StatisticsConfig(max_workers=1)
    if frame.empty or not {"start_date", "end_date"}.issubset(frame.columns):
        return pd.DataFrame(columns=["date", "coverage_seconds"])
    start, _, _= normalize_datetime_series(frame["start_date"], naive_timezone="UTC")
    end, _, _= normalize_datetime_series(frame["end_date"], naive_timezone="UTC")
    valid= start.notna() & end.notna() & end.ge(start)
    working= frame.loc[valid].copy()
    working["start_date"]= start.loc[valid]
    working["end_date"]= end.loc[valid]

    is_windowed= (
        working.get("is_windowed", pd.Series(False, index=working.index))
        .astype("string")
        .str.casefold()
        .isin({"true", "1"})
    )
    coverage_column= (
        "feature_observed_duration_seconds"
        if "feature_observed_duration_seconds" in working.columns
        else "observed_duration_seconds"
    )
    if is_windowed.all() and coverage_column in working.columns:
        # Unit-specific window rows may repeat the same feature-level coverage. Reduce to one row per window
        # before daily accumulation. When only the historical group-specific coverage field exists, max() is the
        # conservative lower-bound estimate for the union across duplicated unit/source rows.
        offsets_column= working.get("start_timezone_offset_minutes", pd.Series(np.nan, index=working.index))
        window_table= pd.DataFrame({
            "start_date": working["start_date"],
            "end_date": working["end_date"],
            "coverage_seconds": pd.to_numeric(working[coverage_column], errors="coerce"),
            "offset_minutes": pd.to_numeric(offsets_column, errors="coerce"),
        })
        window_table= (
            window_table.groupby(["start_date", "end_date"], as_index=False, dropna=False)
            .agg(coverage_seconds=("coverage_seconds", "max"), offset_minutes=("offset_minutes", "first"))
        )
        by_day:dict[str, float]= defaultdict(float)
        for row in window_table.itertuples(index=False):
            span_seconds= max(0.0, float((row.end_date - row.start_date).total_seconds()))
            observed_seconds= float(row.coverage_seconds) if pd.notna(row.coverage_seconds) else 0.0
            observed_seconds= min(max(observed_seconds, 0.0), span_seconds)
            if span_seconds == 0 or observed_seconds == 0:
                continue
            pieces= _split_interval_by_local_day(
                row.start_date, row.end_date, config=config,
                offset_minutes=(float(row.offset_minutes) if pd.notna(row.offset_minutes) else None),
            )
            for date, piece_start, piece_end in pieces:
                piece_seconds= float((piece_end - piece_start).total_seconds())
                # Aggregated windows no longer contain exact sub-window observation
                # locations. Proportional allocation is deterministic and conserves
                # the reported observed duration when a large window crosses midnight.
                by_day[date] += observed_seconds * piece_seconds / span_seconds
        rows= [
            {"date": date, "coverage_seconds": min(seconds, 86_400.0)}
            for date, seconds in sorted(by_day.items())
        ]
        return pd.DataFrame(rows, columns=["date", "coverage_seconds"])

    intervals_by_day:dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]= defaultdict(list)
    offsets= pd.to_numeric(
        working.get("start_timezone_offset_minutes", pd.Series(np.nan, index=working.index)), errors="coerce",
    )
    for index, row in working.iterrows():
        offset= offsets.loc[index] if index in offsets.index else np.nan
        pieces= _split_interval_by_local_day(
            row["start_date"], row["end_date"], config=config,
            offset_minutes=float(offset) if pd.notna(offset) else None,
        )
        for date, piece_start, piece_end in pieces:
            intervals_by_day[date].append((piece_start, piece_end))
    rows= [
        {"date": date, "coverage_seconds": _union_interval_seconds(intervals)}
        for date, intervals in sorted(intervals_by_day.items())
    ]

    return pd.DataFrame(rows, columns=["date", "coverage_seconds"])


def _summarize_participant(participant_dir:str, config:StatisticsConfig,) -> dict[str, Any]:
    path= Path(participant_dir)
    participant_id= path.name
    feature_map= _read_feature_map(path)
    feature_summary:list[dict[str, Any]]= []
    daily_counts:list[dict[str, Any]]= []
    daily_coverage:list[dict[str, Any]]= []
    hourly_counts:list[dict[str, Any]]= []
    active_dates:set[str]= set()
    feature_storage_bytes= 0
    folder_storage_bytes= sum(candidate.stat().st_size for candidate in path.rglob("*") if candidate.is_file())
    feature_error_count= 0

    for feature_name, file_path in sorted(feature_map.items(), key=lambda item: item[0].casefold()):
        try:
            file_size= file_path.stat().st_size
        except OSError as exc:
            feature_error_count += 1
            feature_summary.append({
                "participant_id": participant_id,
                "feature": feature_name,
                "file": file_path.name,
                "status": "read_failed",
                "error": str(exc),
                "rows": 0,
                "invalid_timestamp_rows": 0,
                "first_timestamp_utc": None,
                "last_timestamp_utc": None,
                "windowed": False,
                "coverage_seconds": 0.0,
                "file_size_bytes": 0,
            })
            continue
        feature_storage_bytes += file_size
        try:
            frame= _read_table(file_path)
        except Exception as exc:
            feature_error_count += 1
            feature_summary.append({
                "participant_id": participant_id,
                "feature": feature_name,
                "file": file_path.name,
                "status": "read_failed",
                "error": str(exc),
                "rows": 0,
                "invalid_timestamp_rows": 0,
                "first_timestamp_utc": None,
                "last_timestamp_utc": None,
                "windowed": False,
                "coverage_seconds": 0.0,
                "file_size_bytes": file_size,
            })
            continue

        total_rows= int(len(frame))
        invalid_timestamps= 0
        first_timestamp:str|None= None
        last_timestamp:str|None= None
        windowed= False
        total_coverage= 0.0
        feature_status= "completed"
        feature_error:str|None= None

        if {"start_date", "end_date"}.issubset(frame.columns):
            start, _, _= normalize_datetime_series(frame["start_date"], naive_timezone="UTC")
            end, _, _= normalize_datetime_series(frame["end_date"], naive_timezone="UTC")
            valid= start.notna() & end.notna() & end.ge(start)
            invalid_timestamps= int((~valid).sum())
            valid_frame= frame.loc[valid].copy()
            valid_frame["start_date"]= start.loc[valid]
            valid_frame["end_date"]= end.loc[valid]
            offsets= valid_frame.get("start_timezone_offset_minutes")
            local= _local_clock(valid_frame["start_date"], config=config, offsets=offsets)
            count_table= pd.DataFrame({
                "date": local.dt.date.astype("string"),
                "hour": local.dt.hour,
            }, index=valid_frame.index,).dropna(subset=["date"])
            daily_group= count_table.groupby("date").size()
            contribution= pd.to_numeric(
                valid_frame.get("event_count", pd.Series(np.nan, index=valid_frame.index)), errors="coerce",
            )
            contribution_table= pd.DataFrame(
                {"date": local.dt.date.astype("string"), "contribution": contribution}, index=valid_frame.index,
            ).groupby("date")["contribution"].sum(min_count=1)
            for date, count in daily_group.items():
                daily_counts.append({
                    "participant_id": participant_id,
                    "date": str(date),
                    "feature": feature_name,
                    "row_count": int(count),
                    "source_event_contributions": (
                        float(contribution_table.get(date))
                        if date in contribution_table.index and pd.notna(contribution_table.get(date))
                        else np.nan
                    ),
                })
                active_dates.add(str(date))
            for (date, hour), count in count_table.groupby(["date", "hour"]).size().items():
                hourly_counts.append({
                    "participant_id": participant_id,
                    "date": str(date),
                    "hour": int(hour),
                    "feature": feature_name,
                    "row_count": int(count),
                })
            coverage= coverage_by_day(valid_frame, config=config)
            for row in coverage.itertuples(index=False):
                daily_coverage.append({
                    "participant_id": participant_id,
                    "date": str(row.date),
                    "feature": feature_name,
                    "coverage_seconds": float(row.coverage_seconds),
                })
                active_dates.add(str(row.date))
            total_coverage= float(coverage["coverage_seconds"].sum()) if not coverage.empty else 0.0
            if valid.any():
                first_timestamp= start.loc[valid].min().isoformat()
                last_timestamp= end.loc[valid].max().isoformat()
            windowed= (
                frame.get("is_windowed", pd.Series(False, index=frame.index))
                .astype("string")
                .str.casefold()
                .isin({"true", "1"})
                .any()
            )
        elif "datetime" in frame.columns:
            timestamp, _, _= normalize_datetime_series(frame["datetime"], naive_timezone="UTC")
            valid= timestamp.notna()
            invalid_timestamps= int((~valid).sum())
            valid_frame= frame.loc[valid].copy()
            valid_frame["datetime"]= timestamp.loc[valid]
            offsets= valid_frame.get("datetime_timezone_offset_minutes")
            local= _local_clock(valid_frame["datetime"], config=config, offsets=offsets)
            count_table= pd.DataFrame(
                {"date": local.dt.date.astype("string"), "hour": local.dt.hour}, index=valid_frame.index,
            ).dropna(subset=["date"])
            for date, count in count_table.groupby("date").size().items():
                daily_counts.append({
                    "participant_id": participant_id,
                    "date": str(date),
                    "feature": feature_name,
                    "row_count": int(count),
                    "source_event_contributions": np.nan,
                })
                active_dates.add(str(date))
            for (date, hour), count in count_table.groupby(["date", "hour"]).size().items():
                hourly_counts.append({
                    "participant_id": participant_id,
                    "date": str(date),
                    "hour": int(hour),
                    "feature": feature_name,
                    "row_count": int(count),
                })
            if valid.any():
                first_timestamp= timestamp.loc[valid].min().isoformat()
                last_timestamp= timestamp.loc[valid].max().isoformat()
        else:
            invalid_timestamps= total_rows
            feature_status= "missing_timestamp_columns"
            feature_error= "Expected start_date/end_date or datetime columns."
            feature_error_count += 1

        if feature_status == "completed" and invalid_timestamps:
            feature_status= "completed_with_invalid_timestamps"
            feature_error= f"{invalid_timestamps} rows have invalid or reversed timestamps."
            feature_error_count += 1

        feature_summary.append({
            "participant_id": participant_id,
            "feature": feature_name,
            "file": file_path.name,
            "status": feature_status,
            "error": feature_error,
            "rows": total_rows,
            "invalid_timestamp_rows": invalid_timestamps,
            "first_timestamp_utc": first_timestamp,
            "last_timestamp_utc": last_timestamp,
            "windowed": bool(windowed),
            "coverage_seconds": total_coverage,
            "file_size_bytes": file_size,
        })

    canonical_features= {canonical_feature_name(name) for name in feature_map}
    return {
        "participant_id": participant_id,
        "feature_summary": feature_summary,
        "daily_counts": daily_counts,
        "daily_coverage": daily_coverage,
        "hourly_counts": hourly_counts,
        "active_dates": sorted(active_dates),
        "storage_bytes": folder_storage_bytes,
        "folder_storage_bytes": folder_storage_bytes,
        "feature_storage_bytes": feature_storage_bytes,
        "features": sorted(feature_map),
        "has_height_and_weight": int(
            "height" in canonical_features and bool({"weight", "bodymass"} & canonical_features)
        ),
        "has_bmi": int(bool({"bmi", "bodymassindex"} & canonical_features)),
        "status": "completed_with_errors" if feature_error_count else "completed",
        "feature_error_count": feature_error_count,
    }


def _statistics_worker(participant_dir:str, config:StatisticsConfig) -> dict[str, Any]:
    try:
        return _summarize_participant(participant_dir, config)
    except Exception as exc:
        LOGGER.exception("Statistics failed for %s", participant_dir)
        return {
            "participant_id": Path(participant_dir).name,
            "status": "failed",
            "error": str(exc),
            "feature_summary": [],
            "daily_counts": [],
            "daily_coverage": [],
            "hourly_counts": [],
            "active_dates": [],
            "storage_bytes": 0,
            "features": [],
            "has_height_and_weight": 0,
            "has_bmi": 0,
        }


def _records_frame(results:list[dict[str, Any]], key:str, columns:list[str]) -> pd.DataFrame:
    records= [record for result in results for record in result.get(key, [])]
    return pd.DataFrame(records, columns=columns)


def _write_legacy_outputs(output_root:Path, *, feature_summary:pd.DataFrame, daily_counts:pd.DataFrame,
                          daily_coverage:pd.DataFrame, hourly_counts:pd.DataFrame, active:pd.DataFrame,
                          participant_storage:pd.DataFrame, participant_features:pd.DataFrame,) -> None:
    atomic_write_dataframe(
        participant_storage.rename(columns={"storage_bytes": "folder_size_bytes"}),
        output_root / "logfile_id_folder_sizes.csv", "csv",
    )
    if feature_summary.empty:
        feature_presence= pd.DataFrame(columns=["feature", "count"])
    else:
        feature_presence= (
            feature_summary.loc[feature_summary["status"].eq("completed")]
            .groupby("feature")["participant_id"]
            .nunique()
            .sort_values(ascending=False)
            .rename("count")
            .reset_index()
        )
    atomic_write_dataframe(feature_presence, output_root / "logfile_features_summary.csv", "csv")
    atomic_write_dataframe(participant_features, output_root / "logfile_features_by_id_folder.csv", "csv")

    if daily_counts.empty:
        counts_wide= pd.DataFrame()
    else:
        counts_wide= daily_counts.pivot_table(
            index="date", columns="feature", values="row_count", aggfunc="sum", fill_value=0
        ).sort_index()
    atomic_write_dataframe(
        counts_wide.rename_axis("date").reset_index(), output_root / "logfile_activity_counts.csv", "csv",
    )

    if daily_coverage.empty:
        coverage_wide= pd.DataFrame()
    else:
        coverage_wide= daily_coverage.pivot_table(
            index="date", columns="feature", values="coverage_seconds", aggfunc="sum", fill_value=0
        ).sort_index()
    atomic_write_dataframe(
        coverage_wide.rename_axis("date").reset_index(), output_root / "logfile_activity_durations.csv", "csv",
    )

    if hourly_counts.empty:
        hourly_wide= pd.DataFrame(columns=list(range(24)))
    else:
        hourly_wide= (
            hourly_counts.groupby(["date", "hour"])["row_count"]
            .sum()
            .unstack("hour", fill_value=0)
            .reindex(columns=range(24), fill_value=0)
            .sort_index()
        )
    atomic_write_dataframe(
        hourly_wide.rename_axis("date").reset_index(), output_root / "logfile_all_hourly_data.csv", "csv",
    )

    active_legacy= active.copy()
    if "active_participants" in active_legacy.columns:
        active_legacy["active_devices"]= active_legacy["active_participants"]
    atomic_write_dataframe(active_legacy, output_root / "logfile_active_devices.csv", "csv")


def summarize_dataset(
    input_root:str|Path, output_root:str|Path, config:StatisticsConfig|None= None,
) -> dict[str, pd.DataFrame]:
    """ Generate long-form, auditable statistics without rewriting feature files. """

    config= config or StatisticsConfig()
    input_path= Path(input_root)
    output_path= Path(output_root)
    if not input_path.is_dir():
        raise ConfigurationError(f"Input root is not a directory: {input_path}")
    ensure_disjoint_roots(input_path, output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    participants= sorted(
        [path for path in input_path.iterdir() if path.is_dir()], key=lambda path: path.name.casefold()
    )
    workers= cap_workers(config.max_workers)
    results:list[dict[str, Any]]= []
    if workers == 1:
        for participant in tqdm(participants, desc="Computing statistics", unit="participant"):
            results.append(_statistics_worker(str(participant), config))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"),
        ) as executor:
            futures= {
                executor.submit(_statistics_worker, str(participant), config): participant.name
                for participant in participants
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Computing statistics", unit="participant",
            ):
                results.append(future.result())
    results.sort(key=lambda item: item["participant_id"].casefold())

    feature_summary= _records_frame(
        results,
        "feature_summary",
        [
            "participant_id",
            "feature",
            "file",
            "status",
            "error",
            "rows",
            "invalid_timestamp_rows",
            "first_timestamp_utc",
            "last_timestamp_utc",
            "windowed",
            "coverage_seconds",
            "file_size_bytes",
        ],
    )
    daily_counts= _records_frame(
        results,
        "daily_counts",
        ["participant_id", "date", "feature", "row_count", "source_event_contributions"],
    )
    daily_coverage= _records_frame(
        results,
        "daily_coverage",
        ["participant_id", "date", "feature", "coverage_seconds"],
    )
    hourly_counts= _records_frame(
        results,
        "hourly_counts",
        ["participant_id", "date", "hour", "feature", "row_count"],
    )

    active_counts:dict[str, int]= defaultdict(int)
    for result in results:
        for date in result.get("active_dates", []):
            active_counts[date] += 1
    active= pd.DataFrame([
        {"date": date, "active_participants": count}
        for date, count in sorted(active_counts.items())
    ], columns=["date", "active_participants"],)
    participant_storage= pd.DataFrame([
        {
            "participant_id": result["participant_id"],
            "storage_bytes": result.get("storage_bytes", 0),
            "folder_storage_bytes": result.get(
                "folder_storage_bytes", result.get("storage_bytes", 0)
            ),
            "feature_storage_bytes": result.get("feature_storage_bytes", 0),
            "feature_count": len(result.get("features", [])),
            "has_height_and_weight": result.get("has_height_and_weight", 0),
            "has_bmi": result.get("has_bmi", 0),
            "status": result.get("status", "unknown"),
        }
        for result in results
    ])
    all_features= sorted(
        {feature for result in results for feature in result.get("features", [])}, key=str.casefold
    )
    presence_rows= []
    for result in results:
        present= set(result.get("features", []))
        presence_rows.append({
            "participant_id": result["participant_id"], **{name: int(name in present) for name in all_features}
        })
    participant_features= pd.DataFrame(presence_rows, columns=["participant_id", *all_features])
    error_records= [
        {
            "participant_id": result["participant_id"],
            "feature": None,
            "file": None,
            "status": result.get("status"),
            "error": result.get("error"),
        }
        for result in results
        if result.get("status") == "failed"
    ]
    if not feature_summary.empty:
        feature_errors= feature_summary.loc[~feature_summary["status"].eq("completed")]
        error_records.extend(
            feature_errors.loc[:, ["participant_id", "feature", "file", "status", "error"]].to_dict(orient="records")
        )
    errors= pd.DataFrame(error_records, columns=["participant_id", "feature", "file", "status", "error"],)

    outputs= {
        "feature_summary": feature_summary,
        "daily_counts": daily_counts,
        "daily_coverage": daily_coverage,
        "hourly_counts": hourly_counts,
        "active_participants": active,
        "participant_storage": participant_storage,
        "participant_features": participant_features,
        "errors": errors,
    }
    for name, frame in outputs.items():
        atomic_write_dataframe(frame, output_path / f"statistics_{name}.csv", "csv")
    _write_legacy_outputs(
        output_path,
        feature_summary=feature_summary,
        daily_counts=daily_counts,
        daily_coverage=daily_coverage,
        hourly_counts=hourly_counts,
        active=active,
        participant_storage=participant_storage,
        participant_features=participant_features,
    )
    atomic_write_json(
        output_path / "statistics_summary.json",
        {
            "created_at_utc": utc_now_iso(),
            "input_root": str(input_path.resolve()),
            "output_root": str(output_path.resolve()),
            "participants": len(participants),
            "failed_participants": int(
                sum(result.get("status") == "failed" for result in results)
            ),
            "participants_with_errors": int(
                sum(result.get("status") != "completed" for result in results)
            ),
            "reported_errors": int(len(errors)),
            "features": len(all_features),
            "day_basis": config.day_basis,
            "timezone": config.timezone,
            "coverage_definition": (
                "union of non-overlapping raw intervals per participant, feature, and day; "
                "for pre-aggregated windows crossing a day boundary, observed duration is "
                "allocated in proportion to each day-overlap"
            ),
        },
    )
    return outputs


# compatibility helpers retained for notebooks that imported the original functions.
def accumulate_feature_time(feature:str, frame:pd.DataFrame,) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config= StatisticsConfig(max_workers=1)
    working= frame.copy()
    start, _, _= normalize_datetime_series(working["start_date"], naive_timezone="UTC")
    end, _, _= normalize_datetime_series(working["end_date"], naive_timezone="UTC")
    valid= start.notna() & end.notna() & end.ge(start)
    working= working.loc[valid].copy()
    working["start_date"]= start.loc[valid]
    working["end_date"]= end.loc[valid]
    local= _local_clock(working["start_date"], config=config)
    dates= local.dt.date.astype("string")
    hours= local.dt.hour
    counts= dates.value_counts().sort_index().rename(feature).rename_axis("date").reset_index()
    coverage= coverage_by_day(working, config=config).rename(columns={"coverage_seconds": feature})
    hourly= (
        pd.DataFrame({"date": dates, "hour": hours})
        .groupby(["date", "hour"])
        .size()
        .unstack("hour", fill_value=0)
        .reindex(columns=range(24), fill_value=0)
        .reset_index()
    )
    return coverage, counts, hourly


def list_of_dfs_to_df(frames:list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    result= frames[0].copy()
    for frame in frames[1:]:
        result= result.merge(frame, on="date", how="outer")

    return result.sort_values("date").fillna(0).set_index("date")


def merge_hourly_dataframes(frames:list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).groupby("date", as_index=False).sum(numeric_only=True)


def merge_dfs_and_sum_features(frames:list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["date"])

    normalized= [frame.reset_index() if "date" not in frame.columns else frame.copy() for frame in frames]
    combined= pd.concat(normalized, ignore_index=True)
    numeric= [column for column in combined.select_dtypes(include="number").columns if column != "date"]

    return combined.groupby("date", as_index=False)[numeric].sum(min_count=1).sort_values("date")
