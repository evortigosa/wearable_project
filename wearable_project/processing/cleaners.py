"""
Wearable Data Processing and Modeling project
Feature-aware cleaning at each feature's native temporal resolution.
"""


from __future__ import annotations
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import pandas as pd
from wearable_project.processing.parser import canonical_json
from wearable_project.processing.registry import DedupStrategy, FeatureFamily, FeatureSpec, get_feature_spec


FLATTENED_METADATA_KEYS = {
    "h_k_time_zone", "time_zone", "h_k_was_user_entered", "was_user_entered",
    "h_k_metadata_key_heart_rate_motion_context", "heart_rate_motion_context",
    "status", "trend_arrow", "trend_rate", "h_k_device_name", "device_name",
    "h_k_metadata_key_sync_version", "h_k_v_o2_max_test_type",
}

INTERNAL_OUTPUT_COLUMNS = {
    "event_id", "content_id", "participant_id", "feature", "start_date_local", "end_date_local",
    "start_date_iana", "end_date_iana", "duration_seconds", "unit_evidence", "metadata_raw",
    "metadata_present", "outer_id", "source_file", "source_month", "source_file_sha256",
    "outer_row_number", "start_date_raw", "end_date_raw", "datetime_raw", "created_at_raw",
    "updated_at_raw", "interval_group_id", "quality_flags_raw", "_from_existing_output", "unit", "units",
}

NOISY_QUALITY_FLAGS = {
    "interval_feature_has_zero_duration", "point_feature_has_nonzero_duration",
    "outer_bucket_differs_from_local_start_date", "unit_inferred_not_explicit",
}

PROVENANCE_COLUMNS = [
    "record_id", "source_id", "source_name", "device", "utc_offset_minutes", "time_zone", "was_user_entered",
    "acquisition_method",
]
CONTEXT_COLUMNS = [
    "heart_rate_motion_context", "status", "trend_arrow", "trend_rate", "metadata_device_name",
    "metadata_sync_version", "vo2_max_test_type", "metadata",
]
UNIT_COLUMNS = [
    "raw_unit", "canonical_value", "canonical_unit", "unit_status",
]
AUDIT_COLUMNS = [
    "occurrence_count", "duplicate_count", "revision_count", "duplicate_record_ids", "duplicate_details",
    "revision_details", "conflict_group_id", "quality_flags",
]


@dataclass(slots=True)
class CleanResult:
    dataframe: pd.DataFrame
    input_rows: int
    output_rows: int
    exact_duplicates_removed: int
    revisions_resolved: int
    unresolved_conflicts: int
    invalid_timestamp_rows: int


def missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def text(value: Any) -> str | None:
    return None if missing(value) else str(value)


def flags_from(value: Any) -> set[str]:
    if missing(value):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if str(item)}
    raw = str(value).strip()
    if raw.startswith("["):
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                return {str(item) for item in loaded if str(item)}
        except json.JSONDecodeError:
            pass
    return {item for item in raw.split(";") if item}


def flags_text(flags: Iterable[str]) -> str | None:
    values = sorted({str(item) for item in flags if str(item)})
    return ";".join(values) if values else None


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value)) if not missing(value) else default
    except (TypeError, ValueError):
        return default


def list_from_json(value: Any) -> list[Any]:
    if missing(value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        return [str(value)]


def normalized_scalar(value: Any) -> Any:
    if missing(value):
        return None
    if isinstance(value, bool):
        return value
    raw = str(value).strip()
    try:
        number = Decimal(raw)
        if number.is_finite():
            return format(number.normalize(), "f")
    except (InvalidOperation, ValueError):
        pass
    return raw


def hash_object(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def parse_timestamp(value: Any, assume_utc_if_naive: bool = True) -> tuple[str | None, str | None, int | None, datetime | None, set[str]]:
    flags: set[str] = set()
    if missing(value):
        return None, None, None, None, flags
    raw = str(value).strip()
    candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        converted = pd.to_datetime(raw, errors="coerce", utc=False)
        if pd.isna(converted):
            return None, raw, None, None, {"timestamp_parse_failed"}
        parsed = pd.Timestamp(converted).to_pydatetime()
    if parsed.tzinfo is None:
        if not assume_utc_if_naive:
            return None, parsed.isoformat(), None, parsed, {"timestamp_missing_offset"}
        flags.add("timestamp_assumed_utc")
        parsed = parsed.replace(tzinfo=timezone.utc)
    offset = parsed.utcoffset()
    offset_minutes = int(offset.total_seconds() / 60) if offset is not None else None
    local_iso = parsed.isoformat()
    utc_dt = parsed.astimezone(timezone.utc)
    utc_iso = utc_dt.isoformat().replace("+00:00", "Z")
    return utc_iso, local_iso, offset_minutes, utc_dt, flags


def parse_outer_timestamp(value: Any) -> tuple[str | None, datetime | None, set[str]]:
    # The outer datetime and backend lifecycle fields are defined by this export as UTC bucket/backend timestamps
    # even though they are serialized without an explicit offset. Treat that as a known source convention rather
    # than flagging every valid event as an assumption. Parse failures still surface.
    utc_iso, _, _, dt, flags = parse_timestamp(value, assume_utc_if_naive=True)
    normalized = {flag.replace("timestamp", "outer_timestamp", 1) for flag in flags}
    normalized.discard("outer_timestamp_assumed_utc")
    return utc_iso, dt, normalized


def numeric(value: Any) -> float | None:
    if missing(value):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def nullable_bool(value: Any) -> bool | None:
    if missing(value):
        return None
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    return None


def standardize_time_fields(row: dict[str, Any], spec: FeatureSpec) -> None:
    flags = flags_from(row.get("quality_flags")) | flags_from(row.get("quality_flags_raw"))
    start_source = row.get("start_date_raw") if not missing(row.get("start_date_raw")) else row.get("start_date")
    end_source = row.get("end_date_raw") if not missing(row.get("end_date_raw")) else row.get("end_date")
    start_utc, start_local, offset, start_dt, start_flags = parse_timestamp(start_source)
    end_utc, end_local, _, end_dt, end_flags = parse_timestamp(end_source)

    # Compact participant-feature CSVs retain UTC timestamps plus the original numeric offset instead of repeating
    # UTC, local, raw, and IANA-rendered timestamp strings. During a later incremental merge, reconstruct the
    # local representation internally so boundary-revision logic remains equivalent to a from-scratch clean.
    existing_offset = row.get("utc_offset_minutes")
    if row.get("_from_existing_output") and not missing(existing_offset):
        try:
            offset = int(float(existing_offset))
            fixed_zone = timezone(timedelta(minutes=offset))
            if start_dt is not None:
                start_local = start_dt.astimezone(fixed_zone).isoformat()
            if end_dt is not None:
                end_local = end_dt.astimezone(fixed_zone).isoformat()
        except (TypeError, ValueError, OverflowError):
            flags.add("invalid_utc_offset_minutes")
    flags |= start_flags | end_flags
    row.update({
        "start_date": start_utc,
        "end_date": end_utc,
        "start_date_local": start_local,
        "end_date_local": end_local,
        "utc_offset_minutes": offset,
    })
    for raw_name, clean_name in (("datetime_raw", "datetime"), ("created_at_raw", "created_at"), ("updated_at_raw", "updated_at")):
        source = row.get(raw_name) if not missing(row.get(raw_name)) else row.get(clean_name)
        clean, parsed, more_flags = parse_outer_timestamp(source)
        row[clean_name] = clean
        flags |= more_flags
        if clean_name == "datetime":
            outer_dt = parsed
    outer_dt = locals().get("outer_dt")
    if start_dt is not None and end_dt is not None:
        duration = (end_dt - start_dt).total_seconds()
        row["duration_seconds"] = duration
        if duration < 0:
            flags.add("negative_interval")
    else:
        row["duration_seconds"] = None
        if spec.family != FeatureFamily.DAILY_SUMMARY:
            flags.add("missing_event_timestamp")
    if start_dt is not None and outer_dt is not None:
        if start_dt.date() != outer_dt.date():
            flags.add("outer_bucket_differs_from_utc_start_date")
    zone_name = text(row.get("time_zone"))
    if zone_name and start_dt is not None:
        try:
            zone = ZoneInfo(zone_name)
            zoned = start_dt.astimezone(zone)
            row["start_date_iana"] = zoned.isoformat()
            if end_dt is not None:
                row["end_date_iana"] = end_dt.astimezone(zone).isoformat()
            zone_offset = zoned.utcoffset()
            if offset is not None and zone_offset is not None and int(zone_offset.total_seconds() / 60) != offset:
                flags.add("time_zone_offset_mismatch")
        except ZoneInfoNotFoundError:
            flags.add("invalid_iana_time_zone")
    row["quality_flags"] = flags_text(flags)


def unit_alias(value: str | None) -> str | None:
    if value is None:
        return None
    key = "".join(value.strip().lower().replace("°", "deg").split())
    return {
        "percent": "%", "percentage": "%", "fraction": "fraction",
        "celsius": "cel", "degc": "cel", "fahrenheit": "degf",
        "inch": "in", "inches": "in", "centimeter": "cm", "centimeters": "cm",
        "centimetre": "cm", "centimetres": "cm", "meter": "m", "meters": "m",
        "metre": "m", "metres": "m", "pounds": "lb", "lbs": "lb",
        "kilograms": "kg", "ml/kg/min": "ml/(kg*min)",
    }.get(key, key)


def explicit_conversion(value: float, raw_unit: str, canonical_unit: str) -> float | None:
    raw, target = unit_alias(raw_unit), unit_alias(canonical_unit)
    if raw == target:
        return value
    functions = {
        ("fraction", "%"): lambda x: x * 100.0,
        ("in", "m"): lambda x: x * 0.0254,
        ("cm", "m"): lambda x: x / 100.0,
        ("lb", "kg"): lambda x: x * 0.45359237,
        ("degf", "cel"): lambda x: (x - 32.0) * 5.0 / 9.0,
    }
    fn = functions.get((raw, target))
    return None if fn is None else float(fn(value))


def apply_unit_policy(row: dict[str, Any], spec: FeatureSpec) -> None:
    flags = flags_from(row.get("quality_flags"))

    # Values loaded from an already committed compact CSV have already passed this registry version's unit policy.
    # Preserve the recorded status and conversion rather than reclassifying an inferred source convention as an
    # explicit payload unit during an incremental merge.
    if row.get("_from_existing_output"):
        for column in ("value", "canonical_value", "canonical_systolic_value", "canonical_diastolic_value"):
            if column in row:
                converted = numeric(row.get(column))
                if converted is not None:
                    row[column] = converted
        row["quality_flags"] = flags_text(flags)
        return

    explicit = next((text(row.get(name)) for name in ("unit", "units") if text(row.get(name))), None)
    policy = spec.unit_policy
    if policy is None:
        if explicit:
            row.update({"raw_unit": explicit, "unit_status": "explicit_unresolved", "unit_evidence": "payload_unit"})
        elif spec.measurement_columns and spec.family not in {
            FeatureFamily.STATE_INTERVAL, FeatureFamily.WAVEFORM,
            FeatureFamily.DAILY_SUMMARY, FeatureFamily.DURATION_EVENT,
        }:
            row.update({"unit_status": "unknown", "unit_evidence": None})
        row["quality_flags"] = flags_text(flags)
        return
    row["raw_unit"] = explicit or policy.raw_unit
    row["canonical_unit"] = policy.canonical_unit
    row["unit_status"] = "explicit" if explicit else policy.status
    row["unit_evidence"] = "payload_unit" if explicit else policy.evidence
    if spec.measurement_columns == ("value",):
        value = numeric(row.get("value"))
        if value is None:
            row["canonical_value"] = None
        elif explicit:
            converted = explicit_conversion(value, explicit, policy.canonical_unit)
            row["canonical_value"] = converted
            if converted is None:
                flags.add("unsupported_explicit_unit")
        else:
            row["canonical_value"] = policy.convert(value)
    elif spec.name == "BloodPressure":
        row["canonical_systolic_value"] = numeric(row.get("blood_pressure_systolic_value"))
        row["canonical_diastolic_value"] = numeric(row.get("blood_pressure_diastolic_value"))
    row["quality_flags"] = flags_text(flags)


def measurement_signature(row: dict[str, Any], spec: FeatureSpec) -> dict[str, Any]:
    return {column: normalized_scalar(row.get(column)) for column in spec.measurement_columns}


def source_key(row: dict[str, Any]) -> str | None:
    return text(row.get("source_id")) or text(row.get("source_name"))


def content_payload(row: dict[str, Any], spec: FeatureSpec) -> dict[str, Any]:
    contextual_metadata = {
        name: normalized_scalar(row.get(name))
        for name in (
            "time_zone", "was_user_entered", "heart_rate_motion_context",
            "status", "trend_arrow", "trend_rate", "metadata_device_name",
            "metadata_sync_version", "vo2_max_test_type", "acquisition_method",
        )
        if not missing(row.get(name))
    }
    payload: dict[str, Any] = {
        "participant_id": text(row.get("participant_id")),
        "feature": spec.name,
        "source": source_key(row),
        "start_date": text(row.get("start_date")),
        "end_date": text(row.get("end_date")),
        "metadata": text(row.get("metadata")),
        "context": contextual_metadata,
        "device": text(row.get("device")),
        "measurements": measurement_signature(row, spec),
    }
    if spec.family == FeatureFamily.DAILY_SUMMARY:
        payload.update({"datetime": text(row.get("datetime")), "payload_index": normalized_scalar(row.get("payload_index"))})
    return payload


def set_ids(row: dict[str, Any], spec: FeatureSpec) -> None:
    content = content_payload(row, spec)
    row["content_id"] = hash_object(content)
    record_id = text(row.get("record_id"))
    identity = {
        "participant_id": text(row.get("participant_id")),
        "feature": spec.name,
        "source": source_key(row),
        "record_id": record_id,
    } if record_id else content
    row["event_id"] = hash_object(identity)
    if row.get("start_date") or row.get("end_date"):
        row["interval_group_id"] = hash_object({
            "participant_id": text(row.get("participant_id")), "feature": spec.name,
            "source": source_key(row), "start_date": text(row.get("start_date")),
            "end_date": text(row.get("end_date")),
        })
    else:
        row["interval_group_id"] = None


def acquisition_method(row: dict[str, Any], feature: str) -> str | None:
    sid = (text(row.get("source_id")) or "").lower()
    if feature == "BloodAlcoholContent" and "intellidrink" in sid:
        return "calculator_estimate"
    # Manual entry is already represented losslessly by was_user_entered. Ordinary device/application imports are
    # recoverable from source_id and source_name. Avoid repeating a low-information string on every row.
    return None


def compact_metadata(row: dict[str, Any]) -> None:
    """
    Keep only non-null metadata that is not already represented in columns. The parser intentionally exposes
    frequently used metadata as ordinary columns. Keeping the same values again in both ``metadata`` and
    ``metadata_raw`` made dense files, especially HeartRate and CGM, much larger than their source exports.
    The normalized residual dictionary keeps unmodelled information without duplicating the flattened fields.
    """

    value = row.get("metadata")
    if missing(value):
        row["metadata"] = None
        return
    if isinstance(value, dict):
        metadata = dict(value)
    else:
        try:
            loaded = json.loads(str(value))
            metadata = dict(loaded) if isinstance(loaded, dict) else {"_metadata_value": loaded}
        except (json.JSONDecodeError, TypeError, ValueError):
            metadata = {"_unparsed_metadata": str(value)}
    residual = {
        key: item
        for key, item in metadata.items()
        if key not in FLATTENED_METADATA_KEYS and not missing(item)
    }
    row["metadata"] = canonical_json(residual) if residual else None


def clinical_flags(row: dict[str, Any], spec: FeatureSpec) -> None:
    flags = flags_from(row.get("quality_flags"))
    value = numeric(row.get("canonical_value"))
    if value is None:
        value = numeric(row.get("value"))
    ranges = {
        "HeartRate": (20.0, 250.0), "RestingHeartRate": (20.0, 220.0),
        "WalkingHeartRate": (20.0, 250.0), "RespiratoryRate": (2.0, 80.0),
        "OxygenSaturation": (50.0, 100.0), "Vo2Max": (5.0, 100.0),
        "BodyTemperature": (30.0, 45.0), "PeakFlow": (20.0, 1000.0),
        "PeakExpiratoryFlow": (20.0, 1000.0),
        "BloodAlcoholContent": (0.0, 0.5),
    }
    if value is not None and spec.name in ranges:
        low, high = ranges[spec.name]
        if not low <= value <= high:
            flags.add("clinical_range_warning")
    if spec.name == "BloodPressure":
        systolic, diastolic = numeric(row.get("blood_pressure_systolic_value")), numeric(row.get("blood_pressure_diastolic_value"))
        if systolic is not None and diastolic is not None:
            if systolic <= diastolic:
                flags.add("blood_pressure_pair_order_warning")
            if not 50 <= systolic <= 260 or not 30 <= diastolic <= 160:
                flags.add("clinical_range_warning")
    if spec.name == "Sleep" and not missing(row.get("value")):
        state = str(row["value"]).strip().upper()
        row["value"] = state
        if state not in {"INBED", "ASLEEP", "AWAKE", "CORE", "DEEP", "REM"}:
            flags.add("unknown_sleep_state")
    if spec.family == FeatureFamily.DAILY_SUMMARY:
        flags.add("summary_date_assignment_ambiguous")
    if spec.name == "Electrocardiogram" and not missing(row.get("voltage_measurements")):
        try:
            points = json.loads(str(row["voltage_measurements"]))
            if isinstance(points, list):
                row["waveform_sample_count"] = len(points)
                frequency, duration = numeric(row.get("sampling_frequency")), numeric(row.get("duration_seconds"))
                if frequency and duration is not None and abs(len(points) - round(frequency * duration)) > 1:
                    flags.add("waveform_sample_count_mismatch")
        except (json.JSONDecodeError, TypeError):
            flags.add("waveform_parse_warning")
    row["quality_flags"] = flags_text(flags)


def standardize_record(raw: dict[str, Any], spec: FeatureSpec) -> dict[str, Any]:
    row = dict(raw)
    row["feature"] = spec.name
    row["was_user_entered"] = nullable_bool(row.get("was_user_entered"))
    row["occurrence_count"] = max(1, integer(row.get("occurrence_count"), 1))
    row["duplicate_count"] = integer(row.get("duplicate_count"), 0)
    row["revision_count"] = integer(row.get("revision_count"), 0)
    standardize_time_fields(row, spec)
    compact_metadata(row)
    for column in spec.measurement_columns:
        if column not in row:
            row[column] = None
        if column != "voltage_measurements" and spec.family != FeatureFamily.STATE_INTERVAL:
            converted = numeric(row.get(column))
            if converted is not None:
                row[column] = converted
    apply_unit_policy(row, spec)
    if not row.get("_from_existing_output"):
        row["acquisition_method"] = acquisition_method(row, spec.name)
    clinical_flags(row, spec)
    set_ids(row, spec)
    return row


def preference(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        text(row.get("updated_at")) or "",
        text(row.get("created_at")) or "",
        text(row.get("datetime")) or "",
        text(row.get("source_month")) or "",
        integer(row.get("outer_row_number"), -1),
        integer(row.get("payload_index"), -1),
        text(row.get("source_file")) or "",
    )


def occurrence_detail(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": text(row.get("record_id")),
        "source_month": text(row.get("source_month")),
        "payload_index": integer(row.get("payload_index"), -1),
    }


def revision_detail(row: dict[str, Any], spec: FeatureSpec) -> dict[str, Any]:
    return {
        "record_id": text(row.get("record_id")),
        "datetime": text(row.get("datetime")),
        "source_month": text(row.get("source_month")), "measurements": measurement_signature(row, spec),
    }


def merge_rows(a: dict[str, Any], b: dict[str, Any], spec: FeatureSpec, reason: str, revision: bool = False) -> dict[str, Any]:
    winner, loser = (a, b) if preference(a) >= preference(b) else (b, a)
    winner = dict(winner)
    winner["occurrence_count"] = integer(a.get("occurrence_count"), 1) + integer(b.get("occurrence_count"), 1)
    winner["duplicate_count"] = integer(a.get("duplicate_count"), 0) + integer(b.get("duplicate_count"), 0) + (0 if revision else 1)
    winner["revision_count"] = integer(a.get("revision_count"), 0) + integer(b.get("revision_count"), 0) + (1 if revision else 0)
    ids = {str(item) for item in [text(a.get("record_id")), text(b.get("record_id"))] if item}
    ids.update(str(item) for item in list_from_json(a.get("duplicate_record_ids")) if not missing(item))
    ids.update(str(item) for item in list_from_json(b.get("duplicate_record_ids")) if not missing(item))
    winner["duplicate_record_ids"] = canonical_json(sorted(ids)) if ids else None
    duplicate_details = [*list_from_json(a.get("duplicate_details")), *list_from_json(b.get("duplicate_details"))]
    if not revision:
        duplicate_details.append(occurrence_detail(loser))
    winner["duplicate_details"] = canonical_json(duplicate_details) if duplicate_details else None
    details = [*list_from_json(a.get("revision_details")), *list_from_json(b.get("revision_details"))]
    if revision:
        details.append(revision_detail(loser, spec))
    winner["revision_details"] = canonical_json(details) if details else None
    flags = flags_from(a.get("quality_flags")) | flags_from(b.get("quality_flags")) | {reason}
    winner["quality_flags"] = flags_text(flags)
    return winner


def merge_into_preferred(preferred: dict[str, Any], other: dict[str, Any], spec: FeatureSpec, reason: str) -> dict[str, Any]:
    winner = dict(preferred)
    winner["occurrence_count"] = integer(preferred.get("occurrence_count"), 1) + integer(other.get("occurrence_count"), 1)
    winner["duplicate_count"] = integer(preferred.get("duplicate_count"), 0) + integer(other.get("duplicate_count"), 0)
    winner["revision_count"] = integer(preferred.get("revision_count"), 0) + integer(other.get("revision_count"), 0) + 1
    ids = {str(item) for item in [text(preferred.get("record_id")), text(other.get("record_id"))] if item}
    ids.update(str(item) for item in list_from_json(preferred.get("duplicate_record_ids")) if not missing(item))
    ids.update(str(item) for item in list_from_json(other.get("duplicate_record_ids")) if not missing(item))
    winner["duplicate_record_ids"] = canonical_json(sorted(ids)) if ids else None
    duplicate_details = [*list_from_json(preferred.get("duplicate_details")), *list_from_json(other.get("duplicate_details"))]
    winner["duplicate_details"] = canonical_json(duplicate_details) if duplicate_details else None
    details = [*list_from_json(preferred.get("revision_details")), *list_from_json(other.get("revision_details")), revision_detail(other, spec)]
    winner["revision_details"] = canonical_json(details)
    winner["quality_flags"] = flags_text(flags_from(preferred.get("quality_flags")) | flags_from(other.get("quality_flags")) | {reason})
    return winner


def dedup_record_ids(rows: list[dict[str, Any]], spec: FeatureSpec) -> tuple[list[dict[str, Any]], int, int]:
    by_id: dict[tuple[str | None, str], dict[str, Any]] = {}
    without_id: list[dict[str, Any]] = []
    duplicates = revisions = 0
    for row in rows:
        record_id = text(row.get("record_id"))
        if not record_id:
            without_id.append(row)
            continue
        key = (source_key(row), record_id)
        previous = by_id.get(key)
        if previous is None:
            by_id[key] = row
        elif previous["content_id"] == row["content_id"]:
            by_id[key] = merge_rows(previous, row, spec, "record_id_duplicate")
            duplicates += 1
        else:
            by_id[key] = merge_rows(previous, row, spec, "record_id_revision_resolved", revision=True)
            revisions += 1
    return [*by_id.values(), *without_id], duplicates, revisions


def dedup_content(rows: list[dict[str, Any]], spec: FeatureSpec) -> tuple[list[dict[str, Any]], int]:
    if spec.preserve_content_duplicates or spec.dedup_strategy == DedupStrategy.RECORD_ID_ONLY:
        return rows, 0
    by_content: dict[str, dict[str, Any]] = {}
    removed = 0
    for row in rows:
        key = row["content_id"]
        if key not in by_content:
            by_content[key] = row
        else:
            by_content[key] = merge_rows(by_content[key], row, spec, "exact_content_duplicate")
            removed += 1
    return list(by_content.values()), removed


def iso_date(value: Any) -> str | None:
    return str(value)[:10] if not missing(value) and len(str(value)) >= 10 else None


def resolve_interval_revisions(rows: list[dict[str, Any]], spec: FeatureSpec) -> tuple[list[dict[str, Any]], int, int]:
    if spec.dedup_strategy != DedupStrategy.INTERVAL_REVISION:
        return rows, 0, 0
    groups: dict[str | None, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get("interval_group_id"), []).append(row)
    output: list[dict[str, Any]] = []
    resolved = unresolved = 0
    for group_id, group in groups.items():
        if group_id is None or len(group) == 1:
            output.extend(group)
            continue
        local_matches = [row for row in group if iso_date(row.get("datetime")) == iso_date(row.get("start_date_local"))]
        if spec.resolve_boundary_revisions and len(local_matches) == 1:
            winner = local_matches[0]
            for row in group:
                if row is not local_matches[0]:
                    winner = merge_into_preferred(winner, row, spec, "boundary_revision_resolved")
            output.append(winner)
            resolved += len(group) - 1
        else:
            unresolved += 1
            for row in group:
                row = dict(row)
                row["conflict_group_id"] = group_id
                row["quality_flags"] = flags_text(flags_from(row.get("quality_flags")) | {"same_interval_conflict"})
                output.append(row)
    return output, resolved, unresolved


def numerically_equal(left: Any, right: Any) -> bool:
    a, b = numeric(left), numeric(right)
    if a is None or b is None:
        return normalized_scalar(left) == normalized_scalar(right)
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def compact_output_row(row: dict[str, Any], spec: FeatureSpec) -> dict[str, Any]:
    compact = dict(row)

    # Remove flags that describe normal geometry or duplicate information that is already captured by explicit
    # unit-status columns.
    remaining_flags = flags_from(compact.get("quality_flags")) - NOISY_QUALITY_FLAGS
    compact["quality_flags"] = flags_text(remaining_flags)

    # Keep raw and canonical values together only when a real conversion has occurred. Identity conversions
    # otherwise repeat value and unit in every row without adding information.
    raw_unit = text(compact.get("raw_unit"))
    canonical_unit = text(compact.get("canonical_unit"))
    canonical_value = compact.get("canonical_value")
    raw_value = compact.get("value")
    converted = (
        not missing(canonical_value)
        and (
            not numerically_equal(raw_value, canonical_value)
            or unit_alias(raw_unit) != unit_alias(canonical_unit)
        )
    )
    if not converted:
        compact["canonical_value"] = None
        compact["canonical_unit"] = None
        if text(compact.get("unit_status")) in {
            "source_convention", "validated_source_convention",
            "feature_convention", "explicit",
        }:
            compact["unit_status"] = None

    # Blood-pressure canonical components use an identity mmHg policy and are exact copies of the paired
    # source measurements.
    compact["canonical_systolic_value"] = None
    compact["canonical_diastolic_value"] = None

    if spec.family != FeatureFamily.DAILY_SUMMARY:
        compact["payload_index"] = None

    # Audit counters are exceptional information. Leave ordinary rows blank when the surrounding file needs the
    # column because another row contains a duplicate or revision.
    if integer(compact.get("occurrence_count"), 1) == 1:
        compact["occurrence_count"] = None
    if integer(compact.get("duplicate_count"), 0) == 0:
        compact["duplicate_count"] = None
    if integer(compact.get("revision_count"), 0) == 0:
        compact["revision_count"] = None

    for name in INTERNAL_OUTPUT_COLUMNS:
        compact.pop(name, None)
    return compact


def output_dataframe(rows: list[dict[str, Any]], spec: FeatureSpec) -> pd.DataFrame:
    if spec.family == FeatureFamily.DAILY_SUMMARY:
        rows.sort(key=lambda row: (
            text(row.get("datetime")) or "", integer(row.get("payload_index"), -1),
            text(row.get("event_id")) or "",
        ))
    else:
        rows.sort(key=lambda row: (
            text(row.get("start_date")) or "", text(row.get("end_date")) or "",
            source_key(row) or "", text(row.get("record_id")) or "",
            text(row.get("event_id")) or "",
        ))

    frame = pd.DataFrame([compact_output_row(row, spec) for row in rows])
    if frame.empty:
        return frame

    # Columns that are completely absent or consist only of default audit values are not useful in
    # participant-feature files.
    for column, default in (
        ("occurrence_count", 1),
        ("duplicate_count", 0),
        ("revision_count", 0),
    ):
        if column in frame.columns and all(integer(value, default) == default for value in frame[column].tolist()):
            frame = frame.drop(columns=[column])

    empty_columns = [
        column for column in frame.columns
        if all(missing(value) for value in frame[column].tolist())
    ]
    frame = frame.drop(columns=empty_columns)

    measurement_columns = [name for name in spec.measurement_columns if name in frame.columns]
    core_columns = (
        ["datetime", *measurement_columns, "created_at", "updated_at", "data_source", "collecting_method_version"]
        if spec.family == FeatureFamily.DAILY_SUMMARY
        else ["start_date", "end_date", *measurement_columns, "datetime", "created_at", "updated_at", "data_source", "collecting_method_version"]
    )
    feature_specific = [
        name for name in ("classification", "algorithm_version", "waveform_sample_count", "payload_index")
        if name in frame.columns
    ]
    order: list[str] = []
    for name in [
        *core_columns, *PROVENANCE_COLUMNS, *CONTEXT_COLUMNS,
        *UNIT_COLUMNS, *feature_specific, *AUDIT_COLUMNS,
    ]:
        if name in frame.columns and name not in order:
            order.append(name)
    order.extend(name for name in frame.columns if name not in order)
    return frame.loc[:, order]


def clean_feature(feature: str, raw_rows: list[dict[str, Any]]) -> CleanResult:
    input_rows = len(raw_rows)
    if not raw_rows:
        return CleanResult(pd.DataFrame(), 0, 0, 0, 0, 0, 0)
    all_columns = {key for row in raw_rows for key in row}
    spec = get_feature_spec(feature, all_columns)
    rows = [standardize_record(row, spec) for row in raw_rows]
    rows, record_duplicates, record_revisions = dedup_record_ids(rows, spec)
    rows, content_duplicates = dedup_content(rows, spec)
    rows, interval_revisions, unresolved = resolve_interval_revisions(rows, spec)
    event_ids = [text(row.get("event_id")) for row in rows]
    if any(value is None for value in event_ids) or len(set(event_ids)) != len(event_ids):
        raise ValueError(f"Cleaner produced missing or duplicate internal event IDs for {feature}")
    invalid = sum("timestamp_parse_failed" in flags_from(row.get("quality_flags")) or "missing_event_timestamp" in flags_from(row.get("quality_flags")) for row in rows)
    frame = output_dataframe(rows, spec)
    return CleanResult(frame, input_rows, len(frame), record_duplicates + content_duplicates, record_revisions + interval_revisions, unresolved, invalid)
