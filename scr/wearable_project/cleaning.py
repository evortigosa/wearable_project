"""
Wearable Data Processing and Modeling project
Safe payload parsing, timestamp normalization and exact deduplication.
"""

from __future__ import annotations
import ast
from collections.abc import Iterable, Mapping
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import numpy as np
import pandas as pd
from .exceptions import PayloadParseError, SchemaError
from .policies import canonical_feature_name
from .utils import atomic_write_dataframe, canonical_cell, stable_json

LOGGER= logging.getLogger(__name__)
DEFAULT_SENSITIVE_COLUMNS= ("id", "metadata", "source_id", "source_name")
DEFAULT_SOURCE_IDENTITY_COLUMNS= ("source_id", "source_name")
PROVENANCE_COLUMNS= (
    "source_file", "source_month", "source_row", "source_export_datetime", "source_created_at", "source_updated_at",
)
ACTIVITY_SUMMARY_VALUES= (
    "apple_stand_hours", "apple_exercise_time", "active_energy_burned", "apple_stand_hours_goal",
    "apple_exercise_time_goal", "active_energy_burned_goal",
)
ECG_VALUES= ("average_heart_rate", "sampling_frequency", "voltage_measurements")
BLOOD_PRESSURE_VALUES= (
    "blood_pressure_systolic_value", "blood_pressure_diastolic_value",
)
_OFFSET_RE= re.compile(
    r"(?:(?P<z>[zZ])|(?P<sign>[+-])(?P<hours>\d{2}):?(?P<minutes>\d{2}))$"
)


def _is_missing_scalar(value:Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    if isinstance(value, str):
        return not value.strip()
    return False


def _canonical_identity_value(value:Any) -> Any:
    """ Return a stable, privacy-conscious representation for source-key hashing. """

    if _is_missing_scalar(value):
        return None
    if isinstance(value, str):
        # Exporters frequently vary only in whitespace/case between monthly files.
        return " ".join(value.split()).casefold()
    return canonical_cell(value)


def pseudonymous_source_keys(frame:pd.DataFrame, *, participant_id:str|None,
                             identity_columns:Iterable[str]= DEFAULT_SOURCE_IDENTITY_COLUMNS,) -> tuple[pd.Series, int]:
    """
    Create participant-scoped source keys without retaining raw source labels.
    The same source receives the same key across features/months for one participant,
    while an identical source label for another participant receives a different key.
    This is pseudonymization, not anonymization or cryptographic access control.
    """

    requested= {
        canonical_feature_name(column):str(column)
        for column in identity_columns if str(column).strip()
    }
    available:dict[str, str]= {}
    for column in frame.columns:
        key= canonical_feature_name(column)
        if key in requested and key not in available:
            available[key]= column

    keys= pd.Series(pd.NA, index=frame.index, dtype="string")
    if not available:
        return keys, 0

    namespace= str(participant_id or "")
    ordered_columns= sorted(available.items())
    encoded= pd.DataFrame(index=frame.index)
    missing_token= "m:"
    for canonical_name, column in ordered_columns:
        encoded[canonical_name]= frame[column].map(
            lambda value: (
                missing_token
                if (normalized := _canonical_identity_value(value)) is None
                else f"v:{stable_json(normalized)}"
            )
        )

    # Hash each distinct source identity only once; factorization avoids a Python
    # row loop on very large payloads while keeping deterministic cross-file keys.
    identities= pd.MultiIndex.from_frame(encoded)
    codes, unique_identities= pd.factorize(identities, sort=False)
    unique_keys:list[str|None]= []
    column_names= [name for name, _ in ordered_columns]
    for identity_values in unique_identities:
        if not isinstance(identity_values, tuple):
            identity_values= (identity_values,)
        identity= {
            name: value[2:]
            for name, value in zip(column_names, identity_values, strict=True) if value != missing_token
        }
        if not identity:
            unique_keys.append(None)
            continue
        payload= stable_json({"participant": namespace, "source": identity})
        digest= hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        unique_keys.append(f"src_{digest}")
    key_values= np.asarray(unique_keys, dtype=object)[codes]
    keys= pd.Series(key_values, index=frame.index, dtype="string")
    return keys, int(keys.dropna().nunique())


def _validate_payload_object(value:Any) -> list[dict[str, Any]] | None:
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
        return [dict(item) for item in value]
    return None


def parse_serialized_payload(value:Any, *, max_decode_depth:int= 4) -> list[dict[str, Any]]:
    """
    Decode a payload that may be JSON, a Python literal, or multiply encoded.
    No characters are removed speculatively. This prevents the silent corruption caused
    by unconditional ``value[1:-1]`` slicing.
    """

    validated= _validate_payload_object(value)
    if validated is not None:
        return validated
    if _is_missing_scalar(value):
        raise PayloadParseError("Payload is empty or missing.")
    if isinstance(value, bytes):
        try:
            current:Any= value.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise PayloadParseError(f"Payload is not valid UTF-8: {exc}") from exc
    elif isinstance(value, str):
        current= value.strip().lstrip("\ufeff")
    else:
        raise PayloadParseError(f"Unsupported payload type: {type(value).__name__}.")

    errors:list[str]= []
    for _ in range(max_decode_depth):
        validated= _validate_payload_object(current)
        if validated is not None:
            return validated
        if not isinstance(current, str):
            break
        text= current.strip()
        if not text:
            break

        decoded= False
        for parser_name, parser in (("json", json.loads), ("literal", ast.literal_eval)):
            try:
                candidate= parser(text)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError) as exc:
                errors.append(f"{parser_name}: {exc}")
                continue
            current= candidate
            decoded= True
            break
        if not decoded:
            break

    validated= _validate_payload_object(current)
    if validated is not None:
        return validated
    detail= "; ".join(errors[-2:]) if errors else f"decoded type {type(current).__name__}"
    raise PayloadParseError(f"Payload must decode to a dictionary or list of dictionaries ({detail}).")


def _to_datetime_mixed(values:pd.Series, *, utc:bool) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce", utc=utc, format="mixed")
    except (TypeError, ValueError):
        # Compatibility fallback for older pandas versions or unusual object inputs.
        return pd.to_datetime(values, errors="coerce", utc=utc)


def validate_naive_timezone(naive_timezone:str) -> str:
    """ Validate and normalize a naive-timestamp policy. """

    if not isinstance(naive_timezone, str) or not naive_timezone.strip():
        raise SchemaError("naive_timezone must be an IANA timezone name or 'reject'.")
    normalized= naive_timezone.strip()
    if normalized.casefold() == "reject":
        return "reject"
    try:
        ZoneInfo(normalized)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise SchemaError(f"Invalid naive timezone {normalized!r}: {exc}") from exc
    return normalized


def _offsets_from_strings(values:pd.Series) -> pd.Series:
    strings= values.astype("string").str.strip()
    extracted= strings.str.extract(_OFFSET_RE)
    offsets= pd.Series(pd.NA, index=values.index, dtype="Float64")
    z_mask= extracted["z"].notna()
    offsets.loc[z_mask]= 0.0
    numeric_mask= extracted["sign"].notna()
    if numeric_mask.any():
        hours= pd.to_numeric(extracted.loc[numeric_mask, "hours"], errors="coerce")
        minutes= pd.to_numeric(extracted.loc[numeric_mask, "minutes"], errors="coerce")
        sign= extracted.loc[numeric_mask, "sign"].map({"+": 1, "-": -1}).astype(float)
        offsets.loc[numeric_mask]= sign * (hours * 60 + minutes)
    return offsets


def normalize_datetime_series(values:pd.Series, *, naive_timezone:str= "UTC",) -> tuple[pd.Series, pd.Series, int]:
    """
    Normalize a timestamp series to UTC and retain original UTC offsets.
    ``naive_timezone='reject'`` converts naive values to ``NaT``. Any other value is
    interpreted as an IANA timezone name accepted by pandas.
    """

    naive_timezone= validate_naive_timezone(naive_timezone)
    if not isinstance(values, pd.Series):
        values= pd.Series(values)
    strings= values.astype("string").str.strip()
    explicit_offsets= _offsets_from_strings(values)
    aware_mask= explicit_offsets.notna()
    non_missing= strings.notna() & strings.ne("")
    naive_mask= non_missing & ~aware_mask

    result= pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    if aware_mask.any():
        aware= _to_datetime_mixed(values.loc[aware_mask], utc=True)
        result.loc[aware_mask]= aware

    if naive_mask.any() and naive_timezone.casefold() != "reject":
        if naive_timezone.upper() == "UTC":
            parsed_naive= _to_datetime_mixed(values.loc[naive_mask], utc=True)
            result.loc[naive_mask]= parsed_naive
            explicit_offsets.loc[naive_mask & result.notna()]= 0.0
        else:
            parsed= _to_datetime_mixed(values.loc[naive_mask], utc=False)
            try:
                localized= parsed.dt.tz_localize(
                    naive_timezone, ambiguous="NaT", nonexistent="NaT",
                )
            except (TypeError, ValueError) as exc:
                raise SchemaError(f"Cannot localize naive timestamps in {naive_timezone!r}: {exc}") from exc
            result.loc[naive_mask]= localized.dt.tz_convert("UTC")
            valid_localized= localized.notna()
            for index, timestamp in localized.loc[valid_localized].items():
                offset= timestamp.utcoffset()
                explicit_offsets.loc[index]= (
                    offset.total_seconds() / 60 if offset is not None else pd.NA
                )

    return result, explicit_offsets, int(naive_mask.sum())


def value_columns(frame:pd.DataFrame) -> list[str]:
    """ Infer value-bearing columns without guessing an ActivitySummary schema. """

    if "value" in frame.columns:
        return ["value"]
    if all(column in frame.columns for column in BLOOD_PRESSURE_VALUES):
        return list(BLOOD_PRESSURE_VALUES)
    ecg= [column for column in ECG_VALUES if column in frame.columns]
    if ecg:
        return ecg
    summary= [column for column in ACTIVITY_SUMMARY_VALUES if column in frame.columns]
    if summary:
        return summary
    return []


def drop_exact_duplicates(frame:pd.DataFrame, *,
                          ignore_columns:Iterable[str]= PROVENANCE_COLUMNS,) -> tuple[pd.DataFrame, int]:
    """
    Remove exact records after canonicalizing nested values.
    Simultaneous records with different values are retained. This avoids silently averaging measurements from
    different devices or sources merely because their timestamps match.
    """

    if frame.empty:
        return frame.copy(), 0
    ignored= set(ignore_columns)
    key_columns= [column for column in frame.columns if column not in ignored]
    if not key_columns:
        return frame.iloc[:1].copy(), max(0, len(frame) - 1)
    canonical= pd.DataFrame(index=frame.index)
    for column in key_columns:
        canonical[column]= frame[column].map(canonical_cell)
    duplicate_mask= canonical.duplicated(keep="first")
    removed= int(duplicate_mask.sum())

    return frame.loc[~duplicate_mask].copy(), removed


def normalize_feature_frame(feature_name:str, frame:pd.DataFrame, *, participant_id:str|None= None,
                            retain_participant_id:bool= True, naive_timezone:str= "UTC",
                            sensitive_columns:Iterable[str]= DEFAULT_SENSITIVE_COLUMNS,
                            separate_sources:bool= True,
                            source_identity_columns:Iterable[str]= DEFAULT_SOURCE_IDENTITY_COLUMNS,
                            ) -> tuple[pd.DataFrame, dict[str, int]]:
    """ Validate, pseudonymize, timestamp-normalize and exactly deduplicate a feature. """

    report= {
        "input_records": int(len(frame)),
        "invalid_timestamp_records": 0,
        "negative_duration_records": 0,
        "naive_timestamp_values": 0,
        "participant_mismatch_records": 0,
        "source_identity_count": 0,
        "source_identity_records": 0,
        "duplicates_removed": 0,
        "output_records": 0,
    }
    df= frame.copy()

    if participant_id is not None and "participant_id" in df.columns:
        source_ids= df["participant_id"].astype("string")
        mismatch= source_ids.notna() & source_ids.ne(str(participant_id))
        report["participant_mismatch_records"]= int(mismatch.sum())
    if retain_participant_id and participant_id is not None:
        df["participant_id"]= str(participant_id)
    elif not retain_participant_id:
        df= df.drop(columns=["participant_id"], errors="ignore")

    source_keys, source_identity_count= pseudonymous_source_keys(
        df, participant_id=participant_id, identity_columns=source_identity_columns,
    )
    report["source_identity_count"]= source_identity_count
    report["source_identity_records"]= int(source_keys.notna().sum())
    if separate_sources and source_identity_count:
        df["source_key"]= source_keys

    df= df.drop(columns=list(sensitive_columns), errors="ignore")

    # Top-level export fields are often inconsistently padded or cased across
    # monthly files.  Canonicalize them before exact-record deduplication so the
    # same HealthKit event is not counted twice merely because one export says
    # ``" AppleHealthKit "`` and another says ``"AppleHealthKit"``.
    if "data_source" in df.columns:
        df["data_source"]= (df["data_source"].astype("string").str.strip().str.casefold())
    for unit_column in ("unit", "units"):
        if unit_column in df.columns:
            normalized_units= df[unit_column].astype("string").str.strip()
            df[unit_column]= normalized_units.mask(normalized_units.eq(""), pd.NA)

    feature_key= canonical_feature_name(feature_name)

    if feature_key == "activitysummary":
        if "datetime" not in df.columns:
            raise SchemaError("ActivitySummary records require a 'datetime' column.")
        parsed, offsets, naive_count= normalize_datetime_series(df["datetime"], naive_timezone=naive_timezone)
        df["datetime"]= parsed
        df["datetime_timezone_offset_minutes"]= offsets
        report["naive_timestamp_values"] += naive_count
        invalid= df["datetime"].isna()
        report["invalid_timestamp_records"]= int(invalid.sum())
        df= df.loc[~invalid].copy()
        sort_columns= ["datetime"]
    else:
        missing= [column for column in ("start_date", "end_date") if column not in df.columns]
        if missing:
            raise SchemaError(
                f"Feature {feature_name!r} requires columns {missing}; found {list(df.columns)}."
            )
        start, start_offsets, start_naive= normalize_datetime_series(
            df["start_date"], naive_timezone=naive_timezone
        )
        end, end_offsets, end_naive= normalize_datetime_series(
            df["end_date"], naive_timezone=naive_timezone
        )
        df["start_date"]= start
        df["end_date"]= end
        df["start_timezone_offset_minutes"]= start_offsets
        df["end_timezone_offset_minutes"]= end_offsets
        report["naive_timestamp_values"] += start_naive + end_naive

        invalid= df["start_date"].isna() | df["end_date"].isna()
        report["invalid_timestamp_records"]= int(invalid.sum())
        df= df.loc[~invalid].copy()
        negative= df["end_date"] < df["start_date"]
        report["negative_duration_records"]= int(negative.sum())
        df= df.loc[~negative].copy()
        sort_columns= ["start_date", "end_date"]

    df, duplicates_removed= drop_exact_duplicates(df)
    report["duplicates_removed"]= duplicates_removed
    if sort_columns and not df.empty:
        df= df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    else:
        df= df.reset_index(drop=True)
    report["output_records"]= int(len(df))
    return df, report


def clean_processed_file(feature_name:str, path:str|Path, *, participant_id:str|None= None,
                         rewrite:bool= False, naive_timezone:str= "UTC",) -> tuple[pd.DataFrame, dict[str, int]]:
    """ Clean a processed CSV/Parquet file, optionally replacing it atomically. """

    table_path= Path(path)
    if table_path.suffix.casefold() == ".parquet":
        frame= pd.read_parquet(table_path)
        output_format= "parquet"
    else:
        frame= pd.read_csv(table_path)
        output_format= "csv"
    cleaned, report= normalize_feature_frame(
        feature_name, frame, participant_id=participant_id, naive_timezone=naive_timezone,
    )
    if rewrite:
        atomic_write_dataframe(cleaned, table_path, output_format)
    return cleaned, report
