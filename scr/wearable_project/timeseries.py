"""
Wearable Data Processing and Modeling project
Continuous-segment construction without concurrent writes to a shared CSV.
"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal, Sequence
import numpy as np
import pandas as pd
from tqdm import tqdm
from .cleaning import normalize_datetime_series
from .exceptions import ConfigurationError, SchemaError
from .policies import canonical_feature_name
from .processing import MANIFEST_NAME
from .utils import (
    atomic_write_json,
    cap_workers,
    ensure_disjoint_roots,
    safe_feature_stem,
    utc_now_iso,
)

LOGGER= logging.getLogger(__name__)
DurationBasis= Literal["span", "coverage"]


@dataclass(slots=True)
class SegmentConfig:
    """Definition of a valid continuous time-series segment."""

    min_duration:str= "60min"
    tolerance:str= "1min"
    duration_basis:DurationBasis= "span"
    min_coverage_fraction:float= 0.0
    max_workers:int|None= 4

    def __post_init__(self) -> None:
        try:
            minimum= pd.Timedelta(self.min_duration)
            tolerance= pd.Timedelta(self.tolerance)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid segment duration: {exc}") from exc
        if minimum < pd.Timedelta(0):
            raise ConfigurationError("min_duration cannot be negative.")
        if tolerance < pd.Timedelta(0):
            raise ConfigurationError("tolerance cannot be negative.")
        if self.duration_basis not in {"span", "coverage"}:
            raise ConfigurationError("duration_basis must be 'span' or 'coverage'.")
        if not 0.0 <= self.min_coverage_fraction <= 1.0:
            raise ConfigurationError("min_coverage_fraction must be between 0 and 1.")
        cap_workers(self.max_workers)


def _union_seconds(frame:pd.DataFrame) -> float:
    intervals= sorted([
        (pd.Timestamp(start), pd.Timestamp(end))
        for start, end in zip(frame["start_date"], frame["end_date"], strict=True)
        if pd.notna(start) and pd.notna(end) and end > start
    ], key=lambda item: (item[0], item[1]),)
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
    return float(total + (current_end - current_start).total_seconds())


def segment_continuous_intervals(frame:pd.DataFrame, config:SegmentConfig|None= None, *,
                                 participant_id:str|None= None, feature_name:str|None= None,) -> pd.DataFrame:
    """
    Label rows belonging to segments that satisfy span/coverage requirements.
    The reach of preceding intervals is tracked with a cumulative maximum end time.
    This correctly handles nested and overlapping records; comparing only with the
    immediately preceding row can split a genuinely continuous segment.
    """

    config= config or SegmentConfig(max_workers=1)
    missing= {"start_date", "end_date"}.difference(frame.columns)
    if missing:
        raise SchemaError(f"Time-series segmentation requires columns {sorted(missing)}.")
    helper_columns= [
        "segment_index", "segment_id", "segment_start", "segment_end", "segment_span_seconds",
        "segment_coverage_seconds", "segment_gap_seconds", "segment_coverage_fraction",
    ]
    if frame.empty:
        result= frame.copy()
        for column in helper_columns:
            result[column]= pd.Series(dtype="object")
        return result

    start, _, _= normalize_datetime_series(frame["start_date"], naive_timezone="UTC")
    end, _, _= normalize_datetime_series(frame["end_date"], naive_timezone="UTC")
    valid= start.notna() & end.notna() & end.ge(start)
    working= frame.loc[valid].copy()
    working["start_date"]= start.loc[valid]
    working["end_date"]= end.loc[valid]
    if working.empty:
        result= frame.iloc[0:0].copy()
        for column in helper_columns:
            result[column]= pd.Series(dtype="object")
        return result

    working= working.sort_values(["start_date", "end_date"], kind="mergesort").reset_index(drop=True)
    previous_reach= working["end_date"].cummax().shift(1)
    tolerance= pd.Timedelta(config.tolerance)
    begins_segment= previous_reach.isna()|working["start_date"].gt(previous_reach + tolerance)
    working["_segment"]= begins_segment.cumsum().astype(int) - 1

    minimum_seconds= pd.Timedelta(config.min_duration).total_seconds()
    accepted:list[pd.DataFrame]= []
    accepted_index= 0
    for _, group in working.groupby("_segment", sort=True):
        segment_start= group["start_date"].min()
        segment_end= group["end_date"].max()
        span_seconds= float((segment_end - segment_start).total_seconds())
        coverage_column= (
            "feature_observed_duration_seconds"
            if "feature_observed_duration_seconds" in group.columns
            else "observed_duration_seconds"
        )
        observed= (
            pd.to_numeric(group[coverage_column], errors="coerce")
            if coverage_column in group.columns
            else None
        )
        is_windowed= (
            group.get("is_windowed", pd.Series(False, index=group.index))
            .astype("string")
            .str.casefold()
            .isin({"true", "1"})
            .any()
        )
        if is_windowed and observed is not None and observed.notna().any():
            window_coverage= pd.DataFrame({
                "start_date": group["start_date"],
                "end_date": group["end_date"],
                "coverage": observed,
            })
            # Unit-specific rows repeat feature-level coverage for the same window.
            # Reduce those duplicates before summing adjacent fixed windows.
            window_coverage= window_coverage.groupby(
                ["start_date", "end_date"], as_index=False, dropna=False
            )["coverage"].max()
            coverage_seconds= float(window_coverage["coverage"].clip(lower=0).sum())
            coverage_seconds= min(coverage_seconds, span_seconds)
        else:
            coverage_seconds= _union_seconds(group)
        gap_seconds= max(0.0, span_seconds - coverage_seconds)
        coverage_fraction= 1.0 if span_seconds == 0 else coverage_seconds / span_seconds
        duration_value= span_seconds if config.duration_basis == "span" else coverage_seconds
        if duration_value < minimum_seconds or coverage_fraction < config.min_coverage_fraction:
            continue

        segment= group.drop(columns=["_segment"]).copy()
        segment["segment_index"]= accepted_index
        prefix= participant_id or "participant"
        feature= feature_name or "feature"
        segment["segment_id"]= f"{prefix}:{feature}:{accepted_index:06d}"
        segment["segment_start"]= segment_start
        segment["segment_end"]= segment_end
        segment["segment_span_seconds"]= span_seconds
        segment["segment_coverage_seconds"]= coverage_seconds
        segment["segment_gap_seconds"]= gap_seconds
        segment["segment_coverage_fraction"]= coverage_fraction
        if participant_id is not None:
            segment["participant_id"]= str(participant_id)
        if feature_name is not None:
            segment["feature"]= str(feature_name)
        accepted.append(segment)
        accepted_index += 1

    if not accepted:
        result= working.iloc[0:0].drop(columns=["_segment"]).copy()
        for column in helper_columns:
            if column not in result.columns:
                result[column]= pd.Series(dtype="object")
        return result
    return pd.concat(accepted, ignore_index=True, sort=False)


def _find_feature_file(participant_dir:Path, feature_name:str) -> Path|None:
    requested_key= canonical_feature_name(feature_name)
    participant_root= participant_dir.resolve()
    try:
        manifest= json.loads((participant_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        features= manifest.get("features", {})
        if isinstance(features, dict):
            for recorded_name, details in features.items():
                if canonical_feature_name(recorded_name) != requested_key:
                    continue
                if isinstance(details, dict) and isinstance(details.get("file"), str):
                    candidate= (participant_dir / details["file"]).resolve()
                    if candidate.is_relative_to(participant_root) and candidate.is_file():
                        return candidate
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pass
    stem= safe_feature_stem(feature_name)
    for suffix in (".csv", ".parquet"):
        candidate= participant_dir / f"{stem}{suffix}"
        if candidate.is_file():
            resolved= candidate.resolve()
            if resolved.is_relative_to(participant_root):
                return resolved
    for candidate in sorted(participant_dir.iterdir()):
        if (
            candidate.is_file() and candidate.suffix.casefold() in {".csv", ".parquet"}
            and canonical_feature_name(candidate.stem) == requested_key
        ):
            resolved= candidate.resolve()
            if resolved.is_relative_to(participant_root):
                return resolved
    return None


def _read_table(path:Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.casefold() == ".parquet" else pd.read_csv(path)


def collect_time_series(directory_path:str|Path, activity:str, min_duration:str, tolerance:str, *,
                        duration_basis:DurationBasis= "span",
                        min_coverage_fraction:float= 0.0,) -> pd.DataFrame|None:
    """ Compatibility API for segmenting one participant-feature table. """

    participant_dir= Path(directory_path)
    source= _find_feature_file(participant_dir, activity)
    if source is None:
        return None
    frame= _read_table(source)
    result= segment_continuous_intervals(
        frame,
        SegmentConfig(
            min_duration=min_duration, tolerance=tolerance, duration_basis=duration_basis,
            min_coverage_fraction=min_coverage_fraction, max_workers=1,
        ),
        participant_id=participant_dir.name, feature_name=activity,
    )
    return None if result.empty else result


def _segment_worker(participant_dir:str, feature_name:str, config:SegmentConfig, temporary_root:str,) -> dict[str, Any]:
    participant= Path(participant_dir)
    source= _find_feature_file(participant, feature_name)
    if source is None:
        return {
            "participant_id": participant.name,
            "status": "feature_missing",
            "rows": 0,
            "segments": 0,
            "coverage_seconds": 0.0,
            "invalid_rows_dropped": 0,
            "path": None,
        }
    try:
        frame= _read_table(source)
        start, _, _= normalize_datetime_series(frame.get("start_date", pd.Series(dtype="object")))
        end, _, _= normalize_datetime_series(frame.get("end_date", pd.Series(dtype="object")))
        invalid_rows= int((~(start.notna() & end.notna() & end.ge(start))).sum())
        segmented= segment_continuous_intervals(
            frame, config, participant_id=participant.name, feature_name=feature_name,
        )
        if segmented.empty:
            return {
                "participant_id": participant.name,
                "status": "no_valid_segments",
                "rows": 0,
                "segments": 0,
                "coverage_seconds": 0.0,
                "invalid_rows_dropped": invalid_rows,
                "path": None,
            }
        destination= Path(temporary_root) / f"{safe_feature_stem(participant.name)}.csv"
        segmented.to_csv(destination, index=False)
        segment_summary= segmented.drop_duplicates("segment_id")
        return {
            "participant_id": participant.name,
            "status": "completed",
            "rows": int(len(segmented)),
            "segments": int(segment_summary["segment_id"].nunique()),
            "coverage_seconds": float(segment_summary["segment_coverage_seconds"].sum()),
            "invalid_rows_dropped": invalid_rows,
            "path": str(destination),
        }
    except Exception as exc:
        LOGGER.exception("Time-series construction failed for %s/%s", participant.name, feature_name)
        return {
            "participant_id": participant.name,
            "status": "failed",
            "error": str(exc),
            "rows": 0,
            "segments": 0,
            "coverage_seconds": 0.0,
            "invalid_rows_dropped": 0,
            "path": None,
        }


def _combine_segment_files(paths:list[Path], destination:Path) -> None:
    preferred= [
        "participant_id", "feature", "segment_id", "segment_index", "segment_start", "segment_end",
        "segment_span_seconds", "segment_coverage_seconds", "segment_gap_seconds", "segment_coverage_fraction",
        "start_date", "end_date",
    ]
    all_columns:list[str]= []
    for path in paths:
        columns= pd.read_csv(path, nrows=0).columns.tolist()
        for column in columns:
            if column not in all_columns:
                all_columns.append(column)
    ordered= [column for column in preferred if column in all_columns]
    ordered.extend(column for column in all_columns if column not in ordered)

    temporary= destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    wrote_header= False
    if not paths:
        pd.DataFrame(columns=ordered or preferred).to_csv(temporary, index=False)
        wrote_header= True
    for path in paths:
        for chunk in pd.read_csv(path, chunksize=100_000):
            chunk.reindex(columns=ordered).to_csv(
                temporary, mode="a" if wrote_header else "w", header=not wrote_header, index=False,
            )
            wrote_header= True
    os.replace(temporary, destination)


def build_time_series(input_root:str|Path, output_root:str|Path, features:Sequence[str],
                      config:SegmentConfig|None= None,) -> dict[str, list[dict[str, Any]]]:
    """ Build deterministic combined feature CSVs using per-worker temporary files. """

    config= config or SegmentConfig()
    input_path= Path(input_root)
    output_path= Path(output_root)
    if not input_path.is_dir():
        raise ConfigurationError(f"Input root is not a directory: {input_path}")
    if not features:
        raise ConfigurationError("At least one feature must be requested.")
    canonical_features= [canonical_feature_name(feature) for feature in features]
    if any(not key for key in canonical_features):
        raise ConfigurationError("Feature names cannot be empty.")
    if len(set(canonical_features)) != len(canonical_features):
        raise ConfigurationError("Feature requests must be unique after canonicalization.")
    ensure_disjoint_roots(input_path, output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    participants= sorted(
        [path for path in input_path.iterdir() if path.is_dir()], key=lambda path: path.name.casefold()
    )
    workers= cap_workers(config.max_workers)
    all_reports:dict[str, list[dict[str, Any]]]= {}

    for feature_name in features:
        temporary_root= Path(
            tempfile.mkdtemp(prefix=f".{safe_feature_stem(feature_name)}.segments-", dir=output_path)
        )
        reports:list[dict[str, Any]]= []
        try:
            if workers == 1:
                for participant in tqdm(
                    participants, desc=f"Building {feature_name}", unit="participant",
                ):
                    reports.append(
                        _segment_worker(str(participant), feature_name, config, str(temporary_root))
                    )
            else:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=mp.get_context("spawn"),
                ) as executor:
                    futures= {
                        executor.submit(
                            _segment_worker, str(participant), feature_name, config, str(temporary_root),
                        ): participant.name
                        for participant in participants
                    }
                    for future in tqdm(
                        as_completed(futures), total=len(futures), desc=f"Building {feature_name}", unit="participant",
                    ):
                        reports.append(future.result())
            reports.sort(key=lambda item: item["participant_id"].casefold())
            paths= [Path(item["path"]) for item in reports if item.get("path")]
            destination= output_path / f"{safe_feature_stem(feature_name)}_timeseries.csv"
            _combine_segment_files(paths, destination)
            atomic_write_json(
                output_path / f"{safe_feature_stem(feature_name)}_timeseries_summary.json",
                {
                    "created_at_utc": utc_now_iso(),
                    "feature": feature_name,
                    "configuration": asdict(config),
                    "participants": len(participants),
                    "status_counts": pd.Series([item["status"] for item in reports])
                    .value_counts()
                    .to_dict(),
                    "rows": int(sum(item.get("rows", 0) for item in reports)),
                    "segments": int(sum(item.get("segments", 0) for item in reports)),
                    "coverage_hours": float(
                        sum(item.get("coverage_seconds", 0.0) for item in reports) / 3600
                    ),
                    "reports": [{key: value for key, value in item.items() if key != "path"} for item in reports],
                },
            )
            all_reports[feature_name]= reports
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)
    return all_reports
