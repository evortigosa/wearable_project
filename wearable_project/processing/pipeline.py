"""
Wearable Data Processing and Modeling project
End-to-end processing of participant-level Apple HealthKit-style CSV exports.
"""

from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import json
import logging
import multiprocessing as mp
from numbers import Integral
import os
from pathlib import Path
import shutil
import tempfile
import traceback
from typing import Any
import pandas as pd
from tqdm import tqdm
from wearable_project import __version__
from wearable_project.processing.cleaning import (
    DEFAULT_SENSITIVE_COLUMNS,
    DEFAULT_SOURCE_IDENTITY_COLUMNS,
    normalize_feature_frame,
    parse_serialized_payload,
    validate_naive_timezone,
)
from wearable_project.exceptions import ConfigurationError, PayloadParseError, SchemaError
from wearable_project.processing.policies import (
    FeaturePolicy,
    canonical_feature_name,
    feature_policy_origin,
    policies_to_jsonable,
    resolve_feature_policy,
)
from wearable_project.utils import (
    MONTHLY_FILE_RE,
    atomic_write_dataframe,
    atomic_write_json,
    cap_workers,
    config_digest,
    ensure_disjoint_roots,
    file_fingerprint,
    safe_feature_stem,
    utc_now_iso,
)
from wearable_project.processing.windowing import fixed_window_timedelta, window_feature

LOGGER= logging.getLogger(__name__)
MANIFEST_NAME= ".wearable_manifest.json"
QUALITY_REPORT_NAME= "quality_report.json"
MAX_ISSUE_EXAMPLES= 100


@dataclass(slots=True)
class ProcessingConfig:
    """ Configuration for raw export processing. """

    target_data_source:str= "applehealthkit"
    window:str | None= "5min"
    max_workers:int | None= 4
    resume:bool= True
    strict:bool= False
    chunksize:int= 10_000
    output_format:str= "csv"
    naive_timezone:str= "UTC"
    retain_participant_id:bool= True
    sensitive_columns:tuple[str, ...]= DEFAULT_SENSITIVE_COLUMNS
    separate_sources:bool= True
    source_identity_columns:tuple[str, ...]= DEFAULT_SOURCE_IDENTITY_COLUMNS
    fingerprint_mode:str= "metadata"
    monthly_filename_only:bool= True
    max_windows_per_event:int= 100_000
    feature_policies:Mapping[str, FeaturePolicy | Mapping[str, Any]]= field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.target_data_source, str):
            raise ConfigurationError("target_data_source must be a string.")
        self.target_data_source= self.target_data_source.strip().casefold()
        if not self.target_data_source:
            raise ConfigurationError("target_data_source cannot be empty.")
        if self.window is not None:
            if not isinstance(self.window, str):
                raise ConfigurationError("window must be a fixed-duration string or None.")
            self.window= self.window.strip() or None
            if self.window is not None:
                fixed_window_timedelta(self.window)
        if isinstance(self.chunksize, bool) or not isinstance(self.chunksize, Integral):
            raise ConfigurationError("chunksize must be a positive integer.")
        if self.chunksize <= 0:
            raise ConfigurationError("chunksize must be positive.")
        if self.output_format not in {"csv", "parquet"}:
            raise ConfigurationError("output_format must be 'csv' or 'parquet'.")
        if self.fingerprint_mode not in {"metadata", "sha256"}:
            raise ConfigurationError("fingerprint_mode must be 'metadata' or 'sha256'.")
        if (
            isinstance(self.max_windows_per_event, bool)
            or not isinstance(self.max_windows_per_event, Integral)
            or self.max_windows_per_event <= 0
        ):
            raise ConfigurationError("max_windows_per_event must be positive.")
        try:
            self.naive_timezone= validate_naive_timezone(self.naive_timezone)
        except SchemaError as exc:
            raise ConfigurationError(str(exc)) from exc
        if not all(str(column).strip() for column in self.source_identity_columns):
            raise ConfigurationError("source_identity_columns cannot contain empty names.")
        # Validate early, even when processing runs in worker subprocesses.
        cap_workers(self.max_workers)
        policies_to_jsonable(self.feature_policies)

    @property
    def extension(self) -> str:
        return ".parquet" if self.output_format == "parquet" else ".csv"

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "target_data_source": self.target_data_source,
            "window": self.window,
            "strict": self.strict,
            "output_format": self.output_format,
            "naive_timezone": self.naive_timezone,
            "retain_participant_id": self.retain_participant_id,
            "sensitive_columns": list(self.sensitive_columns),
            "separate_sources": self.separate_sources,
            "source_identity_columns": list(self.source_identity_columns),
            "fingerprint_mode": self.fingerprint_mode,
            "monthly_filename_only": self.monthly_filename_only,
            "max_windows_per_event": self.max_windows_per_event,
            "feature_policies": policies_to_jsonable(self.feature_policies),
            "pipeline_version": __version__,
        }


@dataclass(slots=True)
class FileParseResult:
    path:str
    status:str= "pending"
    rows_seen:int= 0
    rows_selected:int= 0
    payload_records:int= 0
    rejected_rows:int= 0
    issues:list[dict[str, Any]]= field(default_factory=list)

    def add_issue(self, *, row:int | None, code:str, message:str) -> None:
        self.rejected_rows += 1
        if len(self.issues) < MAX_ISSUE_EXAMPLES:
            self.issues.append({"row": row, "code": code, "message": message})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_input_files(participant_dir:Path, *, monthly_only:bool= True) -> tuple[list[Path], list[str]]:
    """ Return deterministic input files and names skipped by the monthly-file rule. """

    csv_files= sorted(path for path in participant_dir.glob("*.csv") if path.is_file())
    if not monthly_only:
        return csv_files, []
    selected:list[Path]= []
    skipped:list[str]= []
    for path in csv_files:
        if MONTHLY_FILE_RE.fullmatch(path.name):
            selected.append(path)
        else:
            skipped.append(path.name)
    return selected, skipped


def _context_value(row:pd.Series, name:str) -> Any:
    value= row.get(name, pd.NA)
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    return value


def parse_monthly_file(path:Path, participant_id:str,
                       config:ProcessingConfig,) -> tuple[dict[str, list[pd.DataFrame]], FileParseResult]:
    """ Parse one monthly export without mutating source data. """

    result= FileParseResult(path=path.name)
    by_feature:dict[str, list[pd.DataFrame]]= defaultdict(list)
    reader:Any|None= None
    try:
        reader= pd.read_csv(path, chunksize=config.chunksize, on_bad_lines="error")
        for chunk in reader:
            required= {"data_source", "name", "data"}
            missing= required.difference(chunk.columns)
            if missing:
                raise SchemaError(f"Missing required top-level columns: {sorted(missing)}.")
            result.rows_seen += int(len(chunk))
            source= chunk["data_source"].astype("string").str.strip().str.casefold()
            selected= chunk.loc[source.eq(config.target_data_source)]
            result.rows_selected += int(len(selected))

            for source_index, row in selected.iterrows():
                line_number= int(source_index) + 2 if isinstance(source_index, Integral) else None
                raw_name= row.get("name", pd.NA)
                if pd.isna(raw_name) or not str(raw_name).strip():
                    result.add_issue(row=line_number, code="missing_feature_name", message="Feature name is empty.")
                    if config.strict:
                        raise SchemaError(f"Missing feature name in {path} at row {line_number}.")
                    continue
                feature_name= str(raw_name).strip()
                feature_key= canonical_feature_name(feature_name)
                if feature_key == "mindful":
                    # The legacy export represents this marker without processable values.
                    continue
                try:
                    records= parse_serialized_payload(row.get("data", pd.NA))
                except PayloadParseError as exc:
                    result.add_issue(row=line_number, code="payload_parse_error", message=str(exc))
                    if config.strict:
                        raise
                    continue
                payload= pd.DataFrame.from_records(records)
                if payload.empty:
                    result.add_issue(row=line_number, code="empty_payload", message="Decoded payload has no records.")
                    if config.strict:
                        raise PayloadParseError(f"Empty payload in {path} at row {line_number}.")
                    continue

                top_level_participant= _context_value(row, "participant_id")
                if "participant_id" not in payload.columns and top_level_participant is not pd.NA:
                    payload["participant_id"]= top_level_participant
                payload["data_source"]= config.target_data_source
                payload["source_file"]= path.name
                payload["source_month"]= path.stem
                payload["source_row"]= line_number

                export_datetime= _context_value(row, "datetime")
                if (
                    feature_key == "activitysummary"
                    and "datetime" not in payload.columns
                    and export_datetime is not pd.NA
                ):
                    payload["datetime"]= export_datetime
                payload["source_export_datetime"]= export_datetime
                payload["source_created_at"]= _context_value(row, "created_at")
                payload["source_updated_at"]= _context_value(row, "updated_at")

                by_feature[feature_name].append(payload)
                result.payload_records += int(len(payload))
    except Exception as exc:
        result.status= "failed"
        if not result.issues or result.issues[-1].get("message") != str(exc):
            if len(result.issues) < MAX_ISSUE_EXAMPLES:
                result.issues.append({"row": None, "code": "file_error", "message": str(exc)})
        # Row-level, expected payload problems are handled above.  A file-level
        # exception means the file was not completely inspected, so returning a
        # partial table would make the manifest incorrectly certify incomplete data.
        raise
    finally:
        if reader is not None and hasattr(reader, "close"):
            reader.close()

    result.status= "completed_with_rejections" if result.rejected_rows else "completed"
    return by_feature, result


def parse_participant_directory(participant_dir:str|Path,
                                config:ProcessingConfig,) -> tuple[dict[str, pd.DataFrame], dict[str, Any], list[dict[str, Any]]]:
    """ Parse all monthly files for one participant and return cleaned feature tables. """

    participant_path= Path(participant_dir)
    participant_id= participant_path.name
    files, skipped_files= discover_input_files(participant_path, monthly_only=config.monthly_filename_only)
    fingerprints= [file_fingerprint(path, config.fingerprint_mode) for path in files]
    raw_by_feature:dict[str, list[pd.DataFrame]]= defaultdict(list)
    display_names:dict[str, str]= {}
    feature_aliases:dict[str, set[str]]= defaultdict(set)
    file_reports:list[dict[str, Any]]= []

    for path in files:
        parsed, file_report= parse_monthly_file(path, participant_id, config)
        file_reports.append(file_report.to_dict())
        for feature_name, frames in parsed.items():
            # Case-only variation is export noise, not a distinct HealthKit type.
            # Punctuation is retained to avoid conflating genuinely different custom
            # study variables such as ``A-B`` and ``AB``.
            key= feature_name.casefold()
            display_names.setdefault(key, feature_name)
            feature_aliases[key].add(feature_name)
            raw_by_feature[key].extend(frames)

    feature_tables:dict[str, pd.DataFrame]= {}
    feature_reports:dict[str, dict[str, Any]]= {}
    for feature_key in sorted(raw_by_feature):
        feature_name= display_names[feature_key]
        combined= pd.concat(raw_by_feature[feature_key], ignore_index=True, sort=False)
        try:
            normalized, cleaning_report= normalize_feature_frame(
                feature_name,
                combined,
                participant_id=participant_id,
                retain_participant_id=config.retain_participant_id,
                naive_timezone=config.naive_timezone,
                sensitive_columns=config.sensitive_columns,
                separate_sources=config.separate_sources,
                source_identity_columns=config.source_identity_columns,
            )
        except SchemaError as exc:
            feature_reports[feature_name]= {
                "status": "rejected", "error": str(exc), "input_records": int(len(combined)),
            }
            if config.strict:
                raise
            continue

        if config.strict and cleaning_report["participant_mismatch_records"]:
            raise SchemaError(
                f"Feature {feature_name!r} contains "
                f"{cleaning_report['participant_mismatch_records']} participant IDs that "
                f"do not match folder {participant_id!r}."
            )

        chosen_policy= resolve_feature_policy(feature_name, config.feature_policies)
        policy_origin= feature_policy_origin(feature_name, config.feature_policies)
        unit_columns= [column for column in ("unit", "units") if column in normalized.columns]
        unit_values= sorted(
            {
                str(value).strip()
                for column in unit_columns
                for value in normalized[column].dropna().tolist()
                if str(value).strip()
            },
            key=str.casefold,
        )
        windowed= False
        if config.window is not None and canonical_feature_name(feature_name) != "activitysummary":
            try:
                normalized= window_feature(
                    normalized, feature_name, config.window, policy=chosen_policy,
                    max_windows_per_event=config.max_windows_per_event,
                )
                windowed= bool(chosen_policy.windowable)
            except (ConfigurationError, SchemaError) as exc:
                if config.strict:
                    raise
                # Preserve validated raw events instead of silently deleting an
                # entire feature because an aggregation assumption was unsafe.
                normalized= normalized.copy()
                normalized["is_windowed"]= False
                feature_tables[feature_name]= normalized
                feature_reports[feature_name]= {
                    **cleaning_report,
                    "status": "preserved_after_windowing_failure",
                    "error": str(exc),
                    "windowed": False,
                    "output_records": int(len(normalized)),
                    "policy": chosen_policy.to_dict(),
                    "policy_origin": policy_origin,
                    "aliases": sorted(
                        feature_aliases[feature_key], key=lambda value: (value.casefold(), value)
                    ),
                    "unit_values": unit_values,
                    "source_adjudication_required": (
                        cleaning_report.get("source_identity_count", 0) > 1
                    ),
                }
                continue
        elif "is_windowed" not in normalized.columns:
            normalized["is_windowed"]= False

        feature_tables[feature_name]= normalized
        feature_reports[feature_name]= {
            **cleaning_report,
            "status": "completed",
            "windowed": windowed,
            "output_records": int(len(normalized)),
            "policy": chosen_policy.to_dict(),
            "policy_origin": policy_origin,
            "aliases": sorted(feature_aliases[feature_key], key=lambda value: (value.casefold(), value)),
            "unit_values": unit_values,
            "source_adjudication_required": (cleaning_report.get("source_identity_count", 0) > 1),
        }

    quality_report:dict[str, Any]= {
        "participant_id": participant_id,
        "created_at_utc": utc_now_iso(),
        "input_file_count": len(files),
        "skipped_non_monthly_csv_files": skipped_files,
        "files": file_reports,
        "features": feature_reports,
        "summary": {
            "top_level_rows_seen": sum(item["rows_seen"] for item in file_reports),
            "top_level_rows_selected": sum(item["rows_selected"] for item in file_reports),
            "payload_records_decoded": sum(item["payload_records"] for item in file_reports),
            "rejected_top_level_rows": sum(item["rejected_rows"] for item in file_reports),
            "features_written": len(feature_tables),
            "records_written": sum(len(table) for table in feature_tables.values()),
        },
    }
    final_files, _= discover_input_files(participant_path, monthly_only=config.monthly_filename_only)
    final_fingerprints= [file_fingerprint(path, config.fingerprint_mode) for path in final_files]
    if final_fingerprints != fingerprints:
        raise ConfigurationError(
            f"Raw files changed while participant {participant_id!r} was being processed; "
            "no output was committed. Re-run after the export directory is stable."
        )
    return feature_tables, quality_report, final_fingerprints


def _manifest_matches(participant_output:Path, *, input_fingerprints:list[dict[str, Any]],
                      config_hash:str,) -> bool:
    manifest_path= participant_output / MANIFEST_NAME
    try:
        manifest= json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("config_digest") != config_hash:
        return False
    if manifest.get("input_files") != input_fingerprints:
        return False
    features= manifest.get("features", {})
    if not isinstance(features, dict):
        return False
    root= participant_output.resolve()
    for details in features.values():
        if not isinstance(details, dict) or not isinstance(details.get("file"), str):
            return False
        candidate= (participant_output / details["file"]).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            return False
    return True


def _swap_participant_directory(temporary:Path, destination:Path) -> None:
    """ Replace a participant directory with rollback on commit failure. """

    backup= destination.with_name(f".{safe_feature_stem(destination.name)}.backup-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    moved_old= False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved_old= True
        os.replace(temporary, destination)
    except Exception:
        if moved_old and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def process_participant(participant_dir:str|Path, output_root:str|Path,
                        config:ProcessingConfig|None= None,) -> dict[str, Any]:
    """ Process one participant transactionally and return a summary dictionary. """

    config= config or ProcessingConfig()
    participant_path= Path(participant_dir)
    if not participant_path.is_dir():
        raise ConfigurationError(f"Participant directory does not exist: {participant_path}")
    output_path= Path(output_root)
    ensure_disjoint_roots(participant_path, output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    participant_id= participant_path.name
    files, skipped= discover_input_files(participant_path, monthly_only=config.monthly_filename_only)
    input_fingerprints= [file_fingerprint(path, config.fingerprint_mode) for path in files]
    config_payload= config.manifest_payload()
    config_hash= config_digest(config_payload)
    destination= output_path / participant_id

    if config.resume and _manifest_matches(
        destination, input_fingerprints=input_fingerprints, config_hash=config_hash,
    ):
        return {
            "participant_id": participant_id,
            "status": "skipped_unchanged",
            "input_files": len(files),
            "skipped_non_monthly_csv_files": len(skipped),
            "features_written": 0,
            "records_written": 0,
        }

    if not files:
        return {
            "participant_id": participant_id,
            "status": "no_input_files",
            "input_files": 0,
            "skipped_non_monthly_csv_files": len(skipped),
            "features_written": 0,
            "records_written": 0,
        }

    temporary= Path(
        tempfile.mkdtemp(
            prefix=f".{safe_feature_stem(participant_id)}.tmp-", dir=output_path,
        )
    )
    try:
        feature_tables, quality_report, parsed_fingerprints= parse_participant_directory(participant_path, config)
        feature_manifest:dict[str, dict[str, Any]]= {}
        used_filenames:set[str]= set()
        for feature_name in sorted(feature_tables, key=str.casefold):
            stem= safe_feature_stem(feature_name)
            filename= f"{stem}{config.extension}"
            if filename in used_filenames:
                raise ConfigurationError(f"Feature filename collision for {feature_name!r}: {filename}")
            used_filenames.add(filename)
            table= feature_tables[feature_name]
            atomic_write_dataframe(table, temporary / filename, config.output_format)
            feature_manifest[feature_name]= {
                "file": filename, "rows": int(len(table)),
                "windowed": bool(table.get("is_windowed", pd.Series([False])).astype(bool).any()),
            }

        manifest= {
            "manifest_schema_version": 1,
            "pipeline_version": __version__,
            "participant_id": participant_id,
            "created_at_utc": utc_now_iso(),
            "config": config_payload,
            "config_digest": config_hash,
            "input_files": parsed_fingerprints,
            "features": feature_manifest,
            "quality_report": QUALITY_REPORT_NAME,
        }
        feature_warnings= any(
            details.get("status") != "completed"
            or details.get("participant_mismatch_records", 0) > 0
            or details.get("policy_origin") == "unknown_conservative"
            or details.get("source_adjudication_required", False)
            for details in quality_report["features"].values()
        )
        run_status= (
            "completed_with_warnings"
            if (
                quality_report["summary"]["rejected_top_level_rows"] or feature_warnings or not feature_tables
            )
            else "completed"
        )
        quality_report["status"]= run_status
        atomic_write_json(temporary / QUALITY_REPORT_NAME, quality_report)
        manifest["status"]= run_status
        atomic_write_json(temporary / MANIFEST_NAME, manifest)
        _swap_participant_directory(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "participant_id": participant_id,
        "status": run_status,
        "input_files": len(files),
        "skipped_non_monthly_csv_files": len(skipped),
        "features_written": len(feature_tables),
        "records_written": int(sum(len(table) for table in feature_tables.values())),
        "rejected_top_level_rows": quality_report["summary"]["rejected_top_level_rows"],
    }


def _participant_worker(participant_dir:str, output_root:str, config:ProcessingConfig,) -> dict[str, Any]:
    try:
        return process_participant(participant_dir, output_root, config)
    except Exception as exc:
        LOGGER.exception("Participant processing failed for %s", participant_dir)
        return {
            "participant_id": Path(participant_dir).name,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=20),
            "input_files": 0,
            "features_written": 0,
            "records_written": 0,
        }


def process_dataset(input_root:str|Path, output_root:str|Path, config:ProcessingConfig|None= None, *,
                    participant_ids:Sequence[str]|None= None, select_root:str|Path|None= None,) -> list[dict[str, Any]]:
    """ Process all participant directories, continuing after participant-level failures. """

    config= config or ProcessingConfig()
    input_path= Path(input_root)
    output_path= Path(output_root)
    if not input_path.is_dir():
        raise ConfigurationError(f"Input root does not exist or is not a directory: {input_path}")
    ensure_disjoint_roots(input_path, output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    available= {path.name:path for path in input_path.iterdir() if path.is_dir()}
    if participant_ids is not None:
        requested= set(participant_ids)
    elif select_root is not None:
        selection= Path(select_root)
        if not selection.is_dir():
            raise ConfigurationError(f"Selection root is not a directory: {selection}")
        requested= {path.name for path in selection.iterdir() if path.is_dir()}
    else:
        requested= set(available)
    selected= [available[name] for name in sorted(requested.intersection(available), key=str.casefold)]
    missing_requested= sorted(requested.difference(available), key=str.casefold)

    reports:list[dict[str, Any]]= []
    workers= cap_workers(config.max_workers)
    if workers == 1:
        for participant in tqdm(selected, desc="Processing participants", unit="participant"):
            reports.append(_participant_worker(str(participant), str(output_path), config))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),) as executor:
            futures= {
                executor.submit(_participant_worker, str(participant), str(output_path), config): participant.name
                for participant in selected
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing participants", unit="participant",
            ):
                reports.append(future.result())

    reports.sort(key=lambda item: item["participant_id"].casefold())
    summary= {
        "created_at_utc": utc_now_iso(),
        "input_root": str(input_path.resolve()),
        "output_root": str(output_path.resolve()),
        "participants_discovered": len(available),
        "participants_requested": len(requested),
        "participants_selected": len(selected),
        "participants_missing": missing_requested,
        "status_counts": pd.Series([item["status"] for item in reports]).value_counts().to_dict(),
        "records_written": int(sum(item.get("records_written", 0) for item in reports)),
        "reports": reports,
    }
    atomic_write_json(output_path / "processing_summary.json", summary)
    atomic_write_dataframe(
        pd.DataFrame(reports).drop(columns=["traceback"], errors="ignore"), output_path / "processing_summary.csv", "csv",
    )
    return reports
