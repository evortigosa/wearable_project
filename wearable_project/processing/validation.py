"""
Wearable Data Processing and Modeling project
Fast structural validation for raw participant/month export trees.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import pandas as pd
from .cleaning import normalize_datetime_series, parse_serialized_payload, validate_naive_timezone
from ..exceptions import ConfigurationError, PayloadParseError, SchemaError
from .policies import canonical_feature_name
from .pipeline import discover_input_files
from ..utils.filesystem import atomic_write_json
from ..utils.runtime import utc_now_iso


@dataclass(slots=True)
class ValidationConfig:
    target_data_source:str= "applehealthkit"
    sample_rows_per_file:int= 25
    monthly_filename_only:bool= True
    naive_timezone:str= "UTC"

    def __post_init__(self) -> None:
        if not isinstance(self.target_data_source, str):
            raise ConfigurationError("target_data_source must be a string.")
        self.target_data_source= self.target_data_source.strip().casefold()
        if not self.target_data_source:
            raise ConfigurationError("target_data_source cannot be empty.")
        if self.sample_rows_per_file <= 0:
            raise ConfigurationError("sample_rows_per_file must be positive.")
        try:
            self.naive_timezone= validate_naive_timezone(self.naive_timezone)
        except SchemaError as exc:
            raise ConfigurationError(str(exc)) from exc


def validate_raw_dataset(input_root:str | Path, config:ValidationConfig | None= None, *,
                         output_json:str | Path | None= None,) -> dict[str, Any]:
    """ Sample each raw CSV and report structural/payload problems before a full run. """

    config= config or ValidationConfig()
    root= Path(input_root)
    if not root.is_dir():
        raise ConfigurationError(f"Input root is not a directory: {root}")
    participants= sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: p.name.casefold())
    file_reports:list[dict[str, Any]]= []
    participants_without_monthly_files:list[str]= []

    for participant in participants:
        files, skipped= discover_input_files(participant, monthly_only=config.monthly_filename_only)
        if not files:
            participants_without_monthly_files.append(participant.name)
        for skipped_name in skipped:
            file_reports.append({
                "participant_id": participant.name,
                "file": skipped_name,
                "status": "skipped_non_monthly_name",
                "sampled_rows": 0,
                "selected_rows": 0,
                "payload_failures": 0,
                "schema_failures": 0,
                "invalid_timestamp_records": 0,
                "participant_mismatch_records": 0,
                "error": None,
            })
        for path in files:
            report= {
                "participant_id": participant.name,
                "file": path.name,
                "status": "valid_sample",
                "sampled_rows": 0,
                "selected_rows": 0,
                "payload_failures": 0,
                "schema_failures": 0,
                "invalid_timestamp_records": 0,
                "participant_mismatch_records": 0,
                "error": None,
                "size_bytes": path.stat().st_size,
            }
            try:
                sample= pd.read_csv(path, nrows=config.sample_rows_per_file)
                report["sampled_rows"]= int(len(sample))
                missing= {"data_source", "name", "data"}.difference(sample.columns)
                if missing:
                    raise ValueError(f"missing columns {sorted(missing)}")
                source= sample["data_source"].astype("string").str.strip().str.casefold()
                selected= sample.loc[source.eq(config.target_data_source)]
                report["selected_rows"]= int(len(selected))
                payload_failures= 0
                schema_failures= 0
                invalid_timestamps= 0
                participant_mismatches= 0
                for _, row in selected.iterrows():
                    raw_name= row.get("name", pd.NA)
                    if pd.isna(raw_name) or not str(raw_name).strip():
                        schema_failures += 1
                        continue
                    try:
                        records= parse_serialized_payload(row.get("data", pd.NA))
                    except PayloadParseError:
                        payload_failures += 1
                        continue
                    if not records:
                        schema_failures += 1
                        continue
                    payload= pd.DataFrame.from_records(records)
                    feature_key= canonical_feature_name(str(raw_name))
                    if "participant_id" in payload.columns:
                        identifiers= payload["participant_id"].astype("string")
                        participant_mismatches += int(
                            (identifiers.notna() & identifiers.ne(participant.name)).sum()
                        )
                    if feature_key == "activitysummary":
                        if "datetime" not in payload.columns and pd.notna(row.get("datetime", pd.NA)):
                            payload["datetime"]= row.get("datetime")
                        if "datetime" not in payload.columns:
                            schema_failures += 1
                            continue
                        parsed, _, _= normalize_datetime_series(
                            payload["datetime"], naive_timezone=config.naive_timezone
                        )
                        invalid_timestamps += int(parsed.isna().sum())
                    else:
                        if not {"start_date", "end_date"}.issubset(payload.columns):
                            schema_failures += 1
                            continue
                        start, _, _= normalize_datetime_series(
                            payload["start_date"], naive_timezone=config.naive_timezone
                        )
                        end, _, _= normalize_datetime_series(
                            payload["end_date"], naive_timezone=config.naive_timezone
                        )
                        invalid_timestamps += int(
                            (start.isna() | end.isna() | end.lt(start)).sum()
                        )
                report["payload_failures"]= payload_failures
                report["schema_failures"]= schema_failures
                report["invalid_timestamp_records"]= invalid_timestamps
                report["participant_mismatch_records"]= participant_mismatches
                if not len(selected):
                    report["status"]= "no_target_rows_in_sample"
                elif payload_failures or schema_failures or invalid_timestamps or participant_mismatches:
                    report["status"]= "sample_issues"
            except Exception as exc:
                report["status"]= "invalid"
                report["error"]= str(exc)
            file_reports.append(report)

    status_counts= pd.Series([item["status"] for item in file_reports]).value_counts().to_dict()
    issue_statuses= {"invalid", "sample_issues"}
    sampled_issues= int(sum(status_counts.get(status, 0) for status in issue_statuses))
    structural_issues= len(participants_without_monthly_files)
    report= {
        "created_at_utc": utc_now_iso(),
        "input_root": str(root.resolve()),
        "configuration": asdict(config),
        "participants": len(participants),
        "participants_without_monthly_files": participants_without_monthly_files,
        "files": len(file_reports),
        "status_counts": status_counts,
        "sampled_issues": sampled_issues,
        "structural_issues": structural_issues,
        "issues_found": sampled_issues + structural_issues,
        "file_reports": file_reports,
    }
    if output_json is not None:
        atomic_write_json(Path(output_json), report)
    return report
