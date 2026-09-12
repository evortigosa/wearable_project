"""
Wearable Data Processing and Modeling project
Streaming parser for cumulative monthly Apple HealthKit CSV exports.
"""


from __future__ import annotations
import ast
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from wearable_project.exceptions import DuplicateMonthError, ParticipantProcessingError, PayloadDecodeError


APPLE_SOURCE = "applehealthkit"
MONTH_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{1,2})\.csv$")
EXPECTED_COLUMNS = {
    "id", "participant_id", "data_source", "name", "datetime", "data",
    "collecting_method_version", "created_at", "updated_at",
}


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: Path
    canonical_month: str
    sha256: str
    size_bytes: int


@dataclass(slots=True)
class ParseDiagnostic:
    source_file: str
    row_number: int | None
    feature: str | None
    stage: str
    message: str


@dataclass(slots=True)
class ParseResult:
    records_by_feature: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    diagnostics: list[ParseDiagnostic] = field(default_factory=list)
    outer_rows_seen: int = 0
    apple_rows_seen: int = 0
    payload_items_seen: int = 0


def canonical_month_from_name(name: str) -> str | None:
    match = MONTH_RE.match(name)
    if match is None:
        return None
    year, month = int(match.group("year")), int(match.group("month"))
    return f"{year:04d}-{month:02d}" if 1 <= month <= 12 else None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_month_files(participant_dir: Path) -> list[SourceFile]:
    by_month: dict[str, SourceFile] = {}
    for path in sorted(participant_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        month = canonical_month_from_name(path.name)
        if month is None:
            continue
        source = SourceFile(path, month, sha256_file(path), path.stat().st_size)
        if month in by_month:
            raise DuplicateMonthError(
                f"Participant {participant_dir.name!r} contains both "
                f"{by_month[month].path.name!r} and {path.name!r} for {month}."
            )
        by_month[month] = source
    return [by_month[key] for key in sorted(by_month)]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _load_literal(text: str) -> Any:
    json_error: Exception | None = None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        json_error = exc
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError) as exc:
        raise PayloadDecodeError(f"JSON error: {json_error}; Python-literal error: {exc}") from exc


def decode_payload(value: Any, max_layers: int = 8) -> Any:
    """Decode repeated JSON/Python-literal string layers safely.

    Some exported cells contain literal outer quote characters while an inner
    user-entered string contains unescaped double quotes. In that exact shape,
    stripping one visibly paired outer quote reproduces parse_v4's successful
    `data[1:-1]` behavior without applying it unconditionally.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise PayloadDecodeError("Payload is null")
    current = value
    for _ in range(max_layers):
        if not isinstance(current, str):
            return current
        text = current.strip()
        if not text:
            raise PayloadDecodeError("Payload is empty")
        try:
            decoded = _load_literal(text)
        except PayloadDecodeError:
            if len(text) >= 2 and text[0] == text[-1] and text[0] in {"\"", "'"}:
                current = text[1:-1]
                continue
            raise
        if decoded == current:
            break
        current = decoded
    if isinstance(current, str):
        raise PayloadDecodeError("Payload remained a string after all decoding layers")
    return current


def iter_payload_items(payload: Any) -> Iterator[dict[str, Any]]:
    if isinstance(payload, dict):
        yield payload
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
            else:
                yield {"value": item, "_payload_shape_warning": "non_dict_list_item"}
    else:
        yield {"value": payload, "_payload_shape_warning": "scalar_payload"}


def normalize_metadata_key(key: Any) -> str:
    text = re.sub(r"\s*_\s*", "_", str(key).strip())
    text = re.sub(r"\s+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_metadata(value: Any) -> tuple[dict[str, Any], list[str]]:
    if value in (None, ""):
        return {}, []
    metadata = value
    if isinstance(metadata, str):
        try:
            metadata = decode_payload(metadata)
        except PayloadDecodeError:
            return {"_unparsed_metadata": metadata}, ["metadata_unparsed"]
    if not isinstance(metadata, dict):
        return {"_non_dict_metadata": metadata}, ["metadata_not_dict"]
    normalized: dict[str, Any] = {}
    flags: list[str] = []
    for raw_key, raw_value in metadata.items():
        key = normalize_metadata_key(raw_key)
        if key in normalized and normalized[key] != raw_value:
            flags.append(f"metadata_key_collision:{key}")
            previous = normalized[key]
            normalized[key] = [previous, raw_value] if not isinstance(previous, list) else [*previous, raw_value]
        else:
            normalized[key] = raw_value
    return normalized, flags


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _nullable_bool(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def _csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _serialize_nested(value: Any) -> Any:
    return canonical_json(value) if isinstance(value, (dict, list, tuple)) else value


def build_event_record(
    item: dict[str, Any], outer: dict[str, str], participant_id: str,
    feature: str, source: SourceFile, row_number: int, payload_index: int,
) -> dict[str, Any]:
    event = dict(item)
    flags: list[str] = []
    record_id = event.pop("id", None)
    metadata_sentinel = object()
    metadata_raw = event.pop("metadata", metadata_sentinel)
    metadata_present = metadata_raw is not metadata_sentinel
    metadata_value = None if not metadata_present else metadata_raw
    metadata, metadata_flags = normalize_metadata(metadata_value)
    flags.extend(metadata_flags)
    start_raw, end_raw = event.pop("start_date", None), event.pop("end_date", None)
    warning = event.pop("_payload_shape_warning", None)
    if warning:
        flags.append(str(warning))
    for key, value in list(event.items()):
        event[key] = _serialize_nested(value)
    event.update({
        "record_id": None if record_id in (None, "") else str(record_id),
        "start_date_raw": start_raw,
        "end_date_raw": end_raw,
        "metadata_present": metadata_present,
        "metadata_raw": canonical_json(metadata_value) if metadata_present else None,
        "metadata": canonical_json(metadata) if metadata_present else None,
        "time_zone": _metadata_value(metadata, "h_k_time_zone", "time_zone"),
        "was_user_entered": _nullable_bool(_metadata_value(metadata, "h_k_was_user_entered", "was_user_entered")),
        "heart_rate_motion_context": _metadata_value(metadata, "h_k_metadata_key_heart_rate_motion_context", "heart_rate_motion_context"),
        "status": _metadata_value(metadata, "status"),
        "trend_arrow": _metadata_value(metadata, "trend_arrow"),
        "trend_rate": _metadata_value(metadata, "trend_rate"),
        "metadata_device_name": _metadata_value(metadata, "h_k_device_name", "device_name"),
        "metadata_sync_version": _metadata_value(metadata, "h_k_metadata_key_sync_version"),
        "vo2_max_test_type": _metadata_value(metadata, "h_k_v_o2_max_test_type"),
        "participant_id": participant_id,
        "feature": feature,
        "data_source": outer.get("data_source"),
        "collecting_method_version": outer.get("collecting_method_version"),
        "datetime_raw": outer.get("datetime"),
        "created_at_raw": outer.get("created_at"),
        "updated_at_raw": outer.get("updated_at"),
        "outer_id": outer.get("id"),
        "source_file": source.path.name,
        "source_month": source.canonical_month,
        "source_file_sha256": source.sha256,
        "outer_row_number": row_number,
        "payload_index": payload_index,
        "quality_flags_raw": flags,
    })
    return event


def parse_month_file(source: SourceFile, participant_id: str, *, row_error_policy: str = "fail-participant") -> ParseResult:
    _csv_field_limit()
    result = ParseResult(records_by_feature=defaultdict(list))
    try:
        handle = source.path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ParticipantProcessingError(f"Could not open {source.path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        missing = EXPECTED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ParticipantProcessingError(f"{source.path} is missing outer columns: {sorted(missing)}")
        for row_number, outer in enumerate(reader, start=2):
            result.outer_rows_seen += 1
            if str(outer.get("data_source", "")).strip().lower() != APPLE_SOURCE:
                continue
            result.apple_rows_seen += 1
            feature = str(outer.get("name", "")).strip()
            if not feature:
                diagnostic = ParseDiagnostic(source.path.name, row_number, None, "outer-row", "Missing feature name")
                result.diagnostics.append(diagnostic)
                if row_error_policy == "fail-participant":
                    raise ParticipantProcessingError(str(diagnostic))
                continue
            outer_participant = str(outer.get("participant_id", "")).strip()
            if outer_participant and outer_participant != participant_id:
                message = f"Outer participant_id {outer_participant!r} differs from folder {participant_id!r}"
                result.diagnostics.append(ParseDiagnostic(source.path.name, row_number, feature, "outer-row", message))
                if row_error_policy == "fail-participant":
                    raise ParticipantProcessingError(message)
                continue
            try:
                payload = decode_payload(outer.get("data"))
                items = list(iter_payload_items(payload))
            except Exception as exc:
                result.diagnostics.append(ParseDiagnostic(source.path.name, row_number, feature, "payload-decode", str(exc)))
                if row_error_policy == "fail-participant":
                    raise ParticipantProcessingError(f"{participant_id}/{source.path.name}:{row_number} {feature}: {exc}") from exc
                continue
            for payload_index, item in enumerate(items):
                result.records_by_feature[feature].append(
                    build_event_record(item, outer, participant_id, feature, source, row_number, payload_index)
                )
                result.payload_items_seen += 1
    result.records_by_feature = dict(result.records_by_feature)
    return result


def merge_parse_results(results: list[ParseResult]) -> ParseResult:
    merged = ParseResult(records_by_feature=defaultdict(list))
    for result in results:
        merged.outer_rows_seen += result.outer_rows_seen
        merged.apple_rows_seen += result.apple_rows_seen
        merged.payload_items_seen += result.payload_items_seen
        merged.diagnostics.extend(result.diagnostics)
        for feature, records in result.records_by_feature.items():
            merged.records_by_feature[feature].extend(records)
    merged.records_by_feature = dict(merged.records_by_feature)
    return merged
