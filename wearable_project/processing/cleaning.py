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
from ..exceptions import PayloadParseError, SchemaError
from .policies import canonical_feature_name
from ..utils.filesystem import atomic_write_dataframe
from ..utils.serialization import canonical_cell, stable_json

LOGGER= logging.getLogger(__name__)
DEFAULT_SENSITIVE_COLUMNS= ("id", "metadata", "source_id", "source_name")
DEFAULT_SOURCE_IDENTITY_COLUMNS= ("source_id", "source_name")
DEFAULT_SAMPLE_IDENTITY_COLUMNS= ("id",)
USER_ENTERED_METADATA_KEYS= {"hkwasuserentered", "wasuserentered", "isuserentered"}
PROVENANCE_COLUMNS= (
    "source_file", "source_month", "source_row", "source_export_datetime", "source_created_at",
    "source_updated_at",
)
ACTIVITY_SUMMARY_VALUES= (
    "apple_stand_hours", "apple_exercise_time", "active_energy_burned", "apple_stand_hours_goal",
    "apple_exercise_time_goal", "active_energy_burned_goal",
)
ECG_VALUES= ("average_heart_rate", "sampling_frequency", "voltage_measurements")
BLOOD_PRESSURE_VALUES= ("blood_pressure_systolic_value", "blood_pressure_diastolic_value",)
_OFFSET_RE= re.compile(r"(?:(?P<z>[zZ])|(?P<sign>[+-])(?P<hours>\d{2}):?(?P<minutes>\d{2}))$")


def _is_missing_scalar(value:Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    if isinstance(value, str):
        return not value.strip()
    return False


def _canonical_identity_value(value:Any, *, casefold_strings:bool) -> Any:
    """ Return a stable, privacy-conscious representation for identity hashing. """
    if _is_missing_scalar(value):
        return None
    if isinstance(value, str):
        normalized= " ".join(value.split())
        return normalized.casefold() if casefold_strings else normalized
    return canonical_cell(value)


def _identity_columns_present(frame:pd.DataFrame, identity_columns:Iterable[str],) -> list[tuple[str, str]]:
    """ Return ``(canonical_name, actual_column)`` pairs in deterministic order. """
    requested= {
        canonical_feature_name(column): str(column)
        for column in identity_columns
        if str(column).strip()
    }
    available:dict[str, str]= {}
    for column in frame.columns:
        key= canonical_feature_name(column)
        if key in requested and key not in available:
            available[key]= column
    return sorted(available.items())


def _pseudonymous_identity_keys(
    frame:pd.DataFrame, *, participant_id:str|None, identity_columns:Iterable[str], identity_kind:str,
    prefix:str, casefold_strings:bool,
) -> tuple[pd.Series, int, list[str]]:
    """ Hash stable raw identities into participant-scoped pseudonymous keys. """
    ordered_columns= _identity_columns_present(frame, identity_columns)
    keys= pd.Series(pd.NA, index=frame.index, dtype="string")
    if not ordered_columns:
        return keys, 0, []

    namespace= str(participant_id or "")
    encoded= pd.DataFrame(index=frame.index)
    missing_token= "m:"
    for canonical_name, column in ordered_columns:
        encoded[canonical_name]= frame[column].map(
            lambda value: (
                missing_token
                if (normalized := _canonical_identity_value(value, casefold_strings=casefold_strings)) is None
                else f"v:{stable_json(normalized)}"
            )
        )

    identities= pd.MultiIndex.from_frame(encoded)
    codes, unique_identities= pd.factorize(identities, sort=False)
    unique_keys:list[str|None]= []
    column_names= [name for name, _ in ordered_columns]
    for identity_values in unique_identities:
        if not isinstance(identity_values, tuple):
            identity_values= (identity_values,)
        identity= {
            name: value[2:]
            for name, value in zip(column_names, identity_values, strict=True)
            if value != missing_token
        }
        if not identity:
            unique_keys.append(None)
            continue
        payload= stable_json({"participant": namespace, identity_kind: identity,})
        digest= hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        unique_keys.append(f"{prefix}{digest}")
    key_values= np.asarray(unique_keys, dtype=object)[codes]
    keys= pd.Series(key_values, index=frame.index, dtype="string")
    return keys, int(keys.dropna().nunique()), [column for _, column in ordered_columns]


def pseudonymous_source_keys(frame:pd.DataFrame, *, participant_id:str|None,
                             identity_columns:Iterable[str]= DEFAULT_SOURCE_IDENTITY_COLUMNS,) -> tuple[pd.Series, int]:
    """
    Create participant-scoped source keys without retaining raw source labels. The same source receives the same
    key across features/months for one participant, while an identical source label for another participant receives
    a different key. This is pseudonymization, not anonymization or cryptographic access control.
    """

    keys, count, _= _pseudonymous_identity_keys(
        frame, participant_id=participant_id, identity_columns=identity_columns, identity_kind="source",
        prefix="src_", casefold_strings=True,
    )
    return keys, count


def pseudonymous_sample_keys(frame:pd.DataFrame, *, participant_id:str|None,
                             identity_columns:Iterable[str]= DEFAULT_SAMPLE_IDENTITY_COLUMNS,) -> tuple[pd.Series, int]:
    """ Create stable participant-scoped keys for payload sample identities. """
    keys, count, _= _pseudonymous_identity_keys(
        frame, participant_id=participant_id, identity_columns=identity_columns, identity_kind="sample",
        prefix="sample_", casefold_strings=False,
    )
    return keys, count


def _validate_payload_object(value:Any) -> list[dict[str, Any]] | None:
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
        return [dict(item) for item in value]
    return None


def parse_serialized_payload(value:Any, *, max_decode_depth:int= 4) -> list[dict[str, Any]]:
    """
    Decode a payload that may be JSON, a Python literal, or multiply encoded. No characters are removed speculatively.
    This prevents the silent corruption caused by unconditional ``value[1:-1]`` slicing.
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
    Normalize a timestamp series to UTC and retain original UTC offsets. ``naive_timezone='reject'``
    converts naive values to ``NaT``. Any other value is interpreted as an IANA timezone name accepted by pandas.
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


def _canonical_content_frame(frame:pd.DataFrame, *, ignore_columns:Iterable[str],) -> pd.DataFrame:
    ignored= set(ignore_columns)
    key_columns= [column for column in frame.columns if column not in ignored]
    canonical= pd.DataFrame(index=frame.index)
    for column in key_columns:
        canonical[column]= frame[column].map(canonical_cell)
    return canonical


def drop_exact_duplicates(
    frame:pd.DataFrame, *, ignore_columns:Iterable[str]= PROVENANCE_COLUMNS,
) -> tuple[pd.DataFrame, int]:
    """
    Remove exact content duplicates.
    This legacy helper is retained for API compatibility. New parsing code uses :func:`drop_identity_aware_duplicates`,
    which gives stable sample identities precedence over content equality.
    """
    if frame.empty:
        return frame.copy(), 0
    canonical= _canonical_content_frame(frame, ignore_columns=ignore_columns)
    if canonical.shape[1] == 0:
        return frame.iloc[:1].copy(), max(0, len(frame) - 1)
    duplicate_mask= canonical.duplicated(keep="first")
    removed= int(duplicate_mask.sum())
    return frame.loc[~duplicate_mask].copy(), removed


def drop_identity_aware_duplicates(
    frame:pd.DataFrame, *, sample_key_column:str= "sample_key", deduplicate_no_id_content:bool= False,
    ignore_columns:Iterable[str]= PROVENANCE_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Deduplicate records without conflating distinct stable sample identities.
    Rows carrying a stable ``sample_key`` are removed only when both the key and normalized record content are
    repeated. Conflicting records sharing one key are retained and reported. Rows without a stable key are retained
    by default; exact content fallback is available only through an explicit opt-in.
    """

    report= {
        "stable_id_records": 0,
        "stable_id_duplicate_records_removed": 0,
        "stable_id_conflict_groups": 0,
        "ambiguous_no_id_duplicate_candidates": 0,
        "ambiguous_no_id_duplicate_groups": 0,
        "content_fallback_duplicates_removed": 0,
        "duplicates_removed": 0,
    }
    if frame.empty:
        return frame.copy(), report

    if sample_key_column in frame.columns:
        sample_keys= frame[sample_key_column].astype("string")
    else:
        sample_keys= pd.Series(pd.NA, index=frame.index, dtype="string")
    stable_mask= sample_keys.notna() & sample_keys.str.strip().ne("")
    report["stable_id_records"]= int(stable_mask.sum())

    canonical= _canonical_content_frame(
        frame,
        ignore_columns={*ignore_columns, sample_key_column},
    )
    if canonical.shape[1] == 0:
        canonical= pd.DataFrame({"_empty_record": 0}, index=frame.index)

    remove_mask= pd.Series(False, index=frame.index)
    if stable_mask.any():
        stable_identity= pd.concat(
            [sample_keys.loc[stable_mask].rename(sample_key_column), canonical.loc[stable_mask]],
            axis=1,
        )
        stable_duplicates= stable_identity.duplicated(keep="first")
        duplicate_indices= stable_duplicates.index[stable_duplicates]
        remove_mask.loc[duplicate_indices]= True
        report["stable_id_duplicate_records_removed"]= int(stable_duplicates.sum())

        distinct_versions= stable_identity.drop_duplicates()
        version_counts= distinct_versions.groupby(sample_key_column, dropna=False).size()
        report["stable_id_conflict_groups"]= int(version_counts.gt(1).sum())

    no_id_mask= ~stable_mask
    if no_id_mask.any():
        no_id_content= canonical.loc[no_id_mask]
        no_id_duplicates= no_id_content.duplicated(keep="first")
        report["ambiguous_no_id_duplicate_candidates"]= int(no_id_duplicates.sum())
        if bool(no_id_duplicates.any()):
            duplicate_groups= no_id_content.loc[
                no_id_content.duplicated(keep=False)
            ].drop_duplicates()
            report["ambiguous_no_id_duplicate_groups"]= int(len(duplicate_groups))
        if deduplicate_no_id_content:
            duplicate_indices= no_id_duplicates.index[no_id_duplicates]
            remove_mask.loc[duplicate_indices]= True
            report["content_fallback_duplicates_removed"]= int(no_id_duplicates.sum())

    report["duplicates_removed"]= int(remove_mask.sum())
    return frame.loc[~remove_mask].copy(), report


def _coerce_user_entered_value(value:Any) -> Any:
    if _is_missing_scalar(value):
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
        return bool(int(value))
    if isinstance(value, str):
        normalized= value.strip().casefold()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return pd.NA


def _metadata_user_entered_value(value:Any) -> Any:
    if not isinstance(value, Mapping):
        return pd.NA
    for key, candidate in value.items():
        if canonical_feature_name(key) in USER_ENTERED_METADATA_KEYS:
            return _coerce_user_entered_value(candidate)
    return pd.NA


def extract_user_entered_flags(frame:pd.DataFrame) -> pd.Series:
    """ Extract HealthKit's manual-entry marker into a bounded boolean column. """
    result= pd.Series(pd.NA, index=frame.index, dtype="boolean")
    if "is_user_entered" in frame.columns:
        result= frame["is_user_entered"].map(_coerce_user_entered_value).astype("boolean")
    if "metadata" in frame.columns:
        metadata_values= frame["metadata"].map(_metadata_user_entered_value).astype("boolean")
        result= result.fillna(metadata_values)
    return result


def normalize_feature_frame(
    feature_name:str, frame:pd.DataFrame, *, participant_id:str|None= None, retain_participant_id:bool= True,
    naive_timezone:str= "UTC", sensitive_columns:Iterable[str]= DEFAULT_SENSITIVE_COLUMNS, separate_sources:bool= True,
    source_identity_columns:Iterable[str]= DEFAULT_SOURCE_IDENTITY_COLUMNS,
    sample_identity_columns:Iterable[str]= DEFAULT_SAMPLE_IDENTITY_COLUMNS, deduplicate_no_id_content:bool= False,
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
        "sample_identity_count": 0,
        "sample_identity_records": 0,
        "user_entered_true_records": 0,
        "user_entered_false_records": 0,
        "stable_id_records": 0,
        "stable_id_duplicate_records_removed": 0,
        "stable_id_conflict_groups": 0,
        "ambiguous_no_id_duplicate_candidates": 0,
        "ambiguous_no_id_duplicate_groups": 0,
        "content_fallback_duplicates_removed": 0,
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

    sample_keys, sample_identity_count= pseudonymous_sample_keys(
        df, participant_id=participant_id, identity_columns=sample_identity_columns,
    )
    report["sample_identity_count"]= sample_identity_count
    report["sample_identity_records"]= int(sample_keys.notna().sum())
    if sample_identity_count:
        df["sample_key"]= sample_keys

    user_entered= extract_user_entered_flags(df)
    if user_entered.notna().any():
        df["is_user_entered"]= user_entered
        report["user_entered_true_records"]= int(user_entered.eq(True).sum())
        report["user_entered_false_records"]= int(user_entered.eq(False).sum())

    raw_source_columns= [
        column for _, column in _identity_columns_present(df, source_identity_columns)
    ]
    raw_sample_columns= [
        column for _, column in _identity_columns_present(df, sample_identity_columns)
    ]
    columns_to_drop= set(sensitive_columns)|set(raw_source_columns)|set(raw_sample_columns)
    df= df.drop(columns=list(columns_to_drop), errors="ignore")

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

    df, duplicate_report= drop_identity_aware_duplicates(
        df, deduplicate_no_id_content=deduplicate_no_id_content,
    )
    report.update(duplicate_report)
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
