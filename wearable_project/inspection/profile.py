"""
Wearable Data Processing and Modeling project
Mixed-source, bounded-memory profiler for HPP raw wearable exports.
"""


from __future__ import annotations
import ast
import collections
import csv
import hashlib
import json
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from .input_source import ParticipantFiles, RawFile, RawInput
from .sketches import BoundedCounter, NumericSketch, RankedExamples, stable_rank


TOOL_VERSION = "mixed-source-profiler-1.2.0"
CHECKPOINT_COMPATIBLE_TOOL_VERSIONS = frozenset({TOOL_VERSION})
EXPECTED_COLUMNS = (
    "id",
    "participant_id",
    "data_source",
    "name",
    "datetime",
    "data",
    "collecting_method_version",
    "created_at",
    "updated_at",
)
PROVENANCE_DIMENSION_BY_LEAF = {
    "device": "coarse_device",
    "device_name": "coarse_device",
    "device_manufacturer": "device_manufacturer",
    "device_model": "device_model",
    "device_type": "device_type",
    "device_uid": "device_uid",
    "app_package_name": "app_package",
    "package_name": "app_package",
    "app_name": "app_name",
    "application": "app_name",
    "app": "app_name",
    "added_by": "added_by",
    "source_id": "source_id",
    "source_name": "source_name",
    "data_source_id": "data_source_id",
    "origin_data_source_id": "origin_data_source_id",
    "tracked": "tracked",
    "was_user_entered": "user_entered",
    "user_entered": "user_entered",
}
NESTED_DEVICE_DIMENSION_BY_LEAF = {
    "manufacturer": "device_manufacturer",
    "model": "device_model",
    "type": "device_type",
    "uid": "device_uid",
    "name": "coarse_device",
}
DEVICE_DETAIL_KEY_ALIASES = {
    "manufacturer": "manufacturer",
    "device_manufacturer": "manufacturer",
    "model": "model",
    "device_model": "model",
    "type": "type",
    "device_type": "type",
    "uid": "uid",
    "device_uid": "uid",
    "device_name": "name",
}
IDENTITY_LEAF_KEYS = {
    "id",
    "record_id",
    "uuid",
    "identifier",
    "activity_name",
    "source",
    "source_id",
    "source_name",
    "data_source_id",
    "origin_data_source_id",
    "start",
    "end",
    "start_date",
    "end_date",
    "start_time",
    "end_time",
    "date",
    "day",
}


def normalize_source(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().casefold()) or "<missing>"


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
        raise ValueError(f"JSON error: {json_error}; Python-literal error: {exc}") from exc


def decode_payload(value: Any, max_layers: int = 8) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("payload is null")
    current = value
    for _ in range(max_layers):
        if not isinstance(current, str):
            return current
        text = current.strip()
        if not text:
            raise ValueError("payload is empty")
        decoded: Any | None = None
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
            inner = text[1:-1].strip()
            if len(inner) >= 2 and (inner[0], inner[-1]) in {("[", "]"), ("{", "}"), ("(", ")")}:
                try:
                    decoded = _load_literal(inner)
                except ValueError:
                    decoded = None
        if decoded is None:
            try:
                decoded = _load_literal(text)
            except ValueError:
                if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
                    current = text[1:-1]
                    continue
                raise
        if decoded == current:
            break
        current = decoded
    if isinstance(current, str):
        raise ValueError("payload remained a string after all decoding layers")
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


@dataclass(frozen=True)
class ParsedTime:
    utc: datetime
    local: datetime
    offset_minutes: int | None
    offset_explicit: bool


def parse_time_info(value: Any) -> ParsedTime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        try:
            if abs(number) > 10_000_000_000:
                number /= 1000.0
            parsed = datetime.fromtimestamp(number, tz=timezone.utc)
            return ParsedTime(parsed, parsed, None, False)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    candidates = [text, text.replace("Z", "+00:00")]
    if re.search(r"[+-]\d{4}$", text):
        candidates.append(text[:-5] + text[-5:-2] + ":" + text[-2:])
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        explicit = parsed.tzinfo is not None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        offset = parsed.utcoffset()
        offset_minutes = int(offset.total_seconds() // 60) if explicit and offset is not None else None
        return ParsedTime(
            utc=parsed.astimezone(timezone.utc),
            local=parsed,
            offset_minutes=offset_minutes,
            offset_explicit=explicit,
        )
    return None


def parse_time(value: Any) -> datetime | None:
    parsed = parse_time_info(value)
    return parsed.utc if parsed is not None else None


def parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if not text:
        return None
    # Most Google summary dates are ISO dates. Accept an ISO timestamp as a fallback without treating its
    # timezone as evidence for the summary date.
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        parsed = parse_time_info(text)
        return parsed.local.date() if parsed is not None else None


def _normalized_weekday_token(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = text.replace("’", "'").replace("׳", "'")
    text = re.sub(r"[.\s]+$", "", text)
    return text


_WEEKDAY_INDEX: dict[str, int] = {}
for index, names in enumerate((
    ("mon", "monday", "пн", "seg", "lun", "mo", "mån"),
    ("tue", "tuesday", "вт", "ter", "mar", "di", "tis"),
    ("wed", "wednesday", "ср", "qua", "mer", "mié", "mie", "mi", "ons"),
    ("thu", "thursday", "чт", "qui", "jeu", "jue", "do", "tors"),
    ("fri", "friday", "пт", "sex", "ven", "vie", "fr", "fre"),
    ("sat", "saturday", "сб", "sáb", "sab", "sam", "sa", "lör"),
    ("sun", "sunday", "вс", "нд", "dom", "dim", "so", "sön"),
)):
    for name in names:
        _WEEKDAY_INDEX[_normalized_weekday_token(name)] = index
_WEEKDAY_INDEX.update({
    _normalized_weekday_token("יום ב׳"): 0,
    _normalized_weekday_token("יום ג׳"): 1,
    _normalized_weekday_token("יום ד׳"): 2,
    _normalized_weekday_token("יום ה׳"): 3,
    _normalized_weekday_token("יום ו׳"): 4,
    _normalized_weekday_token("שבת"): 5,
    _normalized_weekday_token("יום א׳"): 6,
})


def weekday_index(value: Any) -> int | None:
    return _WEEKDAY_INDEX.get(_normalized_weekday_token(value))


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _safe_text(value: Any, max_length: int = 512) -> str:
    text = value if isinstance(value, str) else canonical_json(value)
    if len(text) <= max_length:
        return text
    digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
    return f"{text[:160]}…[sha256={digest};length={len(text)}]"


def _bounded_example(value: Any, depth: int = 0, max_depth: int = 6, max_list_items: int = 12) -> Any:
    """Create a deterministic, bounded representative suitable for JSON reports."""
    if depth >= max_depth:
        return f"<{_type_name(value)} omitted at depth {depth}>"
    if isinstance(value, dict):
        return {str(key): _bounded_example(child, depth + 1, max_depth, max_list_items) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        shown = [_bounded_example(child, depth + 1, max_depth, max_list_items) for child in value[:max_list_items]]
        if len(value) > max_list_items:
            shown.append({"_omitted_items": len(value) - max_list_items})
        return shown
    if isinstance(value, str) and len(value) > 1000:
        digest = hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()
        return f"{value[:400]}…[sha256={digest};length={len(value)}]"
    return value



def _fingerprint(value: Any) -> str:
    return hashlib.blake2b(
        canonical_json(value).encode("utf-8", "surrogatepass"),
        digest_size=16,
    ).hexdigest()


def _identity_projection(value: Any) -> dict[str, Any]:
    projection: dict[str, Any] = {}

    def visit(current: Any, path: str, depth: int) -> None:
        if depth > 8:
            return
        if isinstance(current, dict):
            for raw_key in sorted(current, key=lambda item: str(item)):
                key = str(raw_key)
                child = current[raw_key]
                child_path = f"{path}.{key}" if path else key
                leaf = key.casefold()
                if leaf in IDENTITY_LEAF_KEYS and not isinstance(child, (dict, list)):
                    projection[child_path] = child
                visit(child, child_path, depth + 1)
        elif isinstance(current, list):
            for child in current:
                visit(child, f"{path}[]", depth + 1)

    visit(value, "", 0)
    return projection


def _record_identifier(item: Mapping[str, Any]) -> tuple[str, Any] | None:
    for key in ("record_id", "uuid", "identifier", "id"):
        value = item.get(key)
        if value not in (None, ""):
            return key, value
    return None


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _first_numeric(item: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in item:
            number = _numeric_value(item.get(key))
            if number is not None:
                return number
    return None


def _signed_day_delta(left: date, right: date) -> int:
    """Return ``right - left`` in whole calendar days."""
    return (right - left).days


def _normalize_weekday_delta(value: int) -> int:
    while value > 3:
        value -= 7
    while value < -3:
        value += 7
    return value


def _negative_calorie_bucket(value: float) -> str:
    magnitude = abs(value)
    if magnitude <= 0.001:
        return "negative_tiny_abs_le_0.001"
    if magnitude <= 1:
        return "negative_small_abs_gt_0.001_le_1"
    if magnitude <= 100:
        return "negative_material_abs_gt_1_le_100"
    return "negative_extreme_abs_gt_100"


def _duration_bucket(seconds: float) -> str:
    magnitude = abs(seconds)
    if seconds == 0:
        return "zero_duration"
    if seconds < 0:
        return "negative_duration"
    if magnitude <= 1:
        return "positive_le_1_second"
    if magnitude <= 60:
        return "positive_gt_1_second_le_1_minute"
    if magnitude <= 3600:
        return "positive_gt_1_minute_le_1_hour"
    if magnitude <= 86400:
        return "positive_gt_1_hour_le_24_hours"
    return "positive_gt_24_hours"


def _difference_bucket(value: float) -> str:
    magnitude = abs(value)
    if magnitude == 0:
        return "equal"
    direction = "positive" if value > 0 else "negative"
    if magnitude <= 10:
        scale = "abs_le_10"
    elif magnitude <= 100:
        scale = "abs_gt_10_le_100"
    elif magnitude <= 1000:
        scale = "abs_gt_100_le_1000"
    elif magnitude <= 10000:
        scale = "abs_gt_1000_le_10000"
    else:
        scale = "abs_gt_10000"
    return f"{direction}_{scale}"


def _month_index(month: str) -> int:
    year, number = month.split("-", 1)
    return int(year) * 12 + int(number) - 1


def _missing_months(months: set[str]) -> list[str]:
    if len(months) < 2:
        return []
    indices = sorted(_month_index(month) for month in months)
    missing: list[str] = []
    for index in range(indices[0], indices[-1] + 1):
        year, month0 = divmod(index, 12)
        month = f"{year:04d}-{month0 + 1:02d}"
        if month not in months:
            missing.append(month)
    return missing


@dataclass
class DiagnosticCollector:
    max_examples_per_code: int = 100
    counts: collections.Counter[str] = field(default_factory=collections.Counter)
    examples: dict[str, RankedExamples] = field(default_factory=dict)

    def add(self, code: str, context: Mapping[str, Any] | None = None, count: int = 1) -> None:
        self.counts[code] += count
        if context is not None and self.max_examples_per_code > 0:
            bucket = self.examples.setdefault(
                code,
                RankedExamples(capacity=self.max_examples_per_code),
            )
            bucket.add(dict(context))

    def merge(self, other: "DiagnosticCollector") -> None:
        self.counts.update(other.counts)
        for code, incoming in other.examples.items():
            bucket = self.examples.setdefault(
                code,
                RankedExamples(capacity=self.max_examples_per_code),
            )
            bucket.merge(incoming)


@dataclass
class SourceSlice:
    rows: int = 0
    months: set[str] = field(default_factory=set)
    features: set[str] = field(default_factory=set)
    versions: set[str] = field(default_factory=set)
    first_outer_datetime: str | None = None
    last_outer_datetime: str | None = None

    def observe_datetime(self, value: str) -> None:
        if not value:
            return
        self.first_outer_datetime = value if self.first_outer_datetime is None else min(self.first_outer_datetime, value)
        self.last_outer_datetime = value if self.last_outer_datetime is None else max(self.last_outer_datetime, value)

    def merge(self, other: "SourceSlice") -> None:
        self.rows += other.rows
        self.months.update(other.months)
        self.features.update(other.features)
        self.versions.update(other.versions)
        if other.first_outer_datetime is not None:
            self.first_outer_datetime = (
                other.first_outer_datetime
                if self.first_outer_datetime is None
                else min(self.first_outer_datetime, other.first_outer_datetime)
            )
        if other.last_outer_datetime is not None:
            self.last_outer_datetime = (
                other.last_outer_datetime
                if self.last_outer_datetime is None
                else max(self.last_outer_datetime, other.last_outer_datetime)
            )


@dataclass
class ParticipantStats:
    participant_id: str
    file_count: int = 0
    source_bytes: int = 0
    export_ended_present: bool = False
    months: set[str] = field(default_factory=set)
    outer_rows: int = 0
    decoded_outer_rows: int = 0
    payload_items: int = 0
    source_slices: dict[str, SourceSlice] = field(default_factory=dict)
    raw_source_labels: dict[str, str] = field(default_factory=dict)
    month_source_rows: collections.Counter[tuple[str, str]] = field(default_factory=collections.Counter)
    month_source_features: dict[tuple[str, str], set[str]] = field(default_factory=lambda: collections.defaultdict(set))


@dataclass
class FeatureStats:
    raw_source_label: str
    source_key: str
    feature: str
    version: str
    participants: set[str] = field(default_factory=set)
    files: set[tuple[str, str]] = field(default_factory=set)
    months: set[str] = field(default_factory=set)
    outer_rows: int = 0
    decoded_outer_rows: int = 0
    decode_failures: int = 0
    payload_items: int = 0
    payload_top_types: collections.Counter[str] = field(default_factory=collections.Counter)
    list_lengths: BoundedCounter = field(default_factory=lambda: BoundedCounter(capacity=100))
    first_outer_datetime: str | None = None
    last_outer_datetime: str | None = None

    def observe_datetime(self, value: str) -> None:
        if not value:
            return
        self.first_outer_datetime = value if self.first_outer_datetime is None else min(self.first_outer_datetime, value)
        self.last_outer_datetime = value if self.last_outer_datetime is None else max(self.last_outer_datetime, value)

    def merge(self, other: "FeatureStats") -> None:
        self.participants.update(other.participants)
        self.files.update(other.files)
        self.months.update(other.months)
        self.outer_rows += other.outer_rows
        self.decoded_outer_rows += other.decoded_outer_rows
        self.decode_failures += other.decode_failures
        self.payload_items += other.payload_items
        self.payload_top_types.update(other.payload_top_types)
        self.list_lengths.merge(other.list_lengths)
        if other.first_outer_datetime is not None:
            self.first_outer_datetime = (
                other.first_outer_datetime
                if self.first_outer_datetime is None
                else min(self.first_outer_datetime, other.first_outer_datetime)
            )
        if other.last_outer_datetime is not None:
            self.last_outer_datetime = (
                other.last_outer_datetime
                if self.last_outer_datetime is None
                else max(self.last_outer_datetime, other.last_outer_datetime)
            )


@dataclass
class FieldStats:
    raw_source_label: str
    source_key: str
    feature: str
    version: str
    path: str
    present_count: int = 0
    null_count: int = 0
    types: collections.Counter[str] = field(default_factory=collections.Counter)
    numeric: NumericSketch = field(default_factory=NumericSketch)
    categorical: BoundedCounter = field(default_factory=BoundedCounter)
    string_too_long_count: int = 0
    nested_list_truncated_count: int = 0
    list_lengths: NumericSketch = field(default_factory=lambda: NumericSketch(sample_size=2048))

    def merge(self, other: "FieldStats") -> None:
        self.present_count += other.present_count
        self.null_count += other.null_count
        self.types.update(other.types)
        self.numeric.merge(other.numeric)
        self.categorical.merge(other.categorical)
        self.string_too_long_count += other.string_too_long_count
        self.nested_list_truncated_count += other.nested_list_truncated_count
        self.list_lengths.merge(other.list_lengths)


@dataclass
class TemporalStats:
    raw_source_label: str
    source_key: str
    feature: str
    version: str
    item_count: int = 0
    start_present: int = 0
    end_present: int = 0
    invalid_start: int = 0
    invalid_end: int = 0
    negative_duration: int = 0
    zero_duration: int = 0
    duration: NumericSketch = field(default_factory=NumericSketch)

    def merge(self, other: "TemporalStats") -> None:
        self.item_count += other.item_count
        self.start_present += other.start_present
        self.end_present += other.end_present
        self.invalid_start += other.invalid_start
        self.invalid_end += other.invalid_end
        self.negative_duration += other.negative_duration
        self.zero_duration += other.zero_duration
        self.duration.merge(other.duration)


@dataclass
class ActivityStats:
    label: str
    count: int = 0
    participants: set[str] = field(default_factory=set)
    files: set[tuple[str, str]] = field(default_factory=set)
    field_presence: collections.Counter[str] = field(default_factory=collections.Counter)
    schemas: collections.Counter[str] = field(default_factory=collections.Counter)
    duration: NumericSketch = field(default_factory=NumericSketch)

    def merge(self, other: "ActivityStats") -> None:
        self.count += other.count
        self.participants.update(other.participants)
        self.files.update(other.files)
        self.field_presence.update(other.field_presence)
        self.schemas.update(other.schemas)
        self.duration.merge(other.duration)


@dataclass
class FeatureVersionMonthStats:
    outer_rows: int = 0
    decoded_outer_rows: int = 0
    payload_items: int = 0
    first_outer_datetime: str | None = None
    last_outer_datetime: str | None = None

    def observe_datetime(self, value: str) -> None:
        if not value:
            return
        self.first_outer_datetime = value if self.first_outer_datetime is None else min(self.first_outer_datetime, value)
        self.last_outer_datetime = value if self.last_outer_datetime is None else max(self.last_outer_datetime, value)

    def merge(self, other: "FeatureVersionMonthStats") -> None:
        self.outer_rows += other.outer_rows
        self.decoded_outer_rows += other.decoded_outer_rows
        self.payload_items += other.payload_items
        if other.first_outer_datetime is not None:
            self.first_outer_datetime = other.first_outer_datetime if self.first_outer_datetime is None else min(self.first_outer_datetime, other.first_outer_datetime)
        if other.last_outer_datetime is not None:
            self.last_outer_datetime = other.last_outer_datetime if self.last_outer_datetime is None else max(self.last_outer_datetime, other.last_outer_datetime)


@dataclass
class DuplicateCalibrationStats:
    observations: int = 0
    unique_identity_keys: int = 0
    distinct_content_fingerprints: int = 0
    repeated_groups: int = 0
    repeated_occurrences: int = 0
    exact_repeat_occurrences: int = 0
    changed_content_occurrences: int = 0
    within_month_repeat_occurrences: int = 0
    cross_month_repeat_occurrences: int = 0
    missing_identity_observations: int = 0

    def merge(self, other: "DuplicateCalibrationStats") -> None:
        self.observations += other.observations
        self.unique_identity_keys += other.unique_identity_keys
        self.distinct_content_fingerprints += other.distinct_content_fingerprints
        self.repeated_groups += other.repeated_groups
        self.repeated_occurrences += other.repeated_occurrences
        self.exact_repeat_occurrences += other.exact_repeat_occurrences
        self.changed_content_occurrences += other.changed_content_occurrences
        self.within_month_repeat_occurrences += other.within_month_repeat_occurrences
        self.cross_month_repeat_occurrences += other.cross_month_repeat_occurrences
        self.missing_identity_observations += other.missing_identity_observations


@dataclass
class _SeenIdentity:
    content_fingerprints: set[str]
    months: set[str]
    repeated_group_counted: bool = False


@dataclass
class ActivitySubmeasurementStats:
    label: str
    version: str
    item_count: int = 0
    participants: set[str] = field(default_factory=set)
    files: set[tuple[str, str]] = field(default_factory=set)
    field_presence: collections.Counter[str] = field(default_factory=collections.Counter)
    numeric_fields: dict[str, NumericSketch] = field(default_factory=dict)
    tracked_values: BoundedCounter = field(default_factory=BoundedCounter)
    device_values: BoundedCounter = field(default_factory=BoundedCounter)
    source_id_values: BoundedCounter = field(default_factory=BoundedCounter)
    source_name_values: BoundedCounter = field(default_factory=BoundedCounter)

    def numeric(self, name: str, sample_size: int) -> NumericSketch:
        stats = self.numeric_fields.get(name)
        if stats is None:
            stats = NumericSketch(sample_size=sample_size)
            self.numeric_fields[name] = stats
        return stats

    def merge(self, other: "ActivitySubmeasurementStats") -> None:
        self.item_count += other.item_count
        self.participants.update(other.participants)
        self.files.update(other.files)
        self.field_presence.update(other.field_presence)
        for name, incoming in other.numeric_fields.items():
            current = self.numeric_fields.get(name)
            if current is None:
                self.numeric_fields[name] = incoming
            else:
                current.merge(incoming)
        self.tracked_values.merge(other.tracked_values)
        self.device_values.merge(other.device_values)
        self.source_id_values.merge(other.source_id_values)
        self.source_name_values.merge(other.source_name_values)


@dataclass
class FocusedDistributionStats:
    count: int = 0
    participants: set[str] = field(default_factory=set)
    values: NumericSketch = field(default_factory=NumericSketch)

    def merge(self, other: "FocusedDistributionStats") -> None:
        self.count += other.count
        self.participants.update(other.participants)
        self.values.merge(other.values)


@dataclass
class WeightCalibrationStats:
    version: str
    added_by: str
    item_count: int = 0
    participants: set[str] = field(default_factory=set)
    malformed_count: int = 0
    point_count: int = 0
    nonpoint_count: int = 0
    invalid_time_count: int = 0
    value: NumericSketch = field(default_factory=NumericSketch)
    duration: NumericSketch = field(default_factory=NumericSketch)

    def merge(self, other: "WeightCalibrationStats") -> None:
        self.item_count += other.item_count
        self.participants.update(other.participants)
        self.malformed_count += other.malformed_count
        self.point_count += other.point_count
        self.nonpoint_count += other.nonpoint_count
        self.invalid_time_count += other.invalid_time_count
        self.value.merge(other.value)
        self.duration.merge(other.duration)


@dataclass
class StepAlternativeStats:
    outer_rows: int = 0
    rows_with_multiple_alternatives: int = 0
    rows_with_estimated: int = 0
    rows_with_merge: int = 0
    rows_with_both: int = 0
    rows_equal: int = 0
    rows_different: int = 0
    difference_merge_minus_estimated: NumericSketch = field(default_factory=NumericSketch)
    absolute_difference: NumericSketch = field(default_factory=NumericSketch)
    difference_buckets: collections.Counter[str] = field(default_factory=collections.Counter)
    alternative_names: collections.Counter[str] = field(default_factory=collections.Counter)
    alternatives_per_outer_row: collections.Counter[int] = field(default_factory=collections.Counter)
    summary_raw_pairs: int = 0
    summary_raw_equal: int = 0
    summary_raw_different: int = 0
    summary_minus_raw: NumericSketch = field(default_factory=NumericSketch)
    summary_raw_difference_buckets: collections.Counter[str] = field(default_factory=collections.Counter)

    def merge(self, other: "StepAlternativeStats") -> None:
        self.outer_rows += other.outer_rows
        self.rows_with_multiple_alternatives += other.rows_with_multiple_alternatives
        self.rows_with_estimated += other.rows_with_estimated
        self.rows_with_merge += other.rows_with_merge
        self.rows_with_both += other.rows_with_both
        self.rows_equal += other.rows_equal
        self.rows_different += other.rows_different
        self.difference_merge_minus_estimated.merge(other.difference_merge_minus_estimated)
        self.absolute_difference.merge(other.absolute_difference)
        self.difference_buckets.update(other.difference_buckets)
        self.alternative_names.update(other.alternative_names)
        self.alternatives_per_outer_row.update(other.alternatives_per_outer_row)
        self.summary_raw_pairs += other.summary_raw_pairs
        self.summary_raw_equal += other.summary_raw_equal
        self.summary_raw_different += other.summary_raw_different
        self.summary_minus_raw.merge(other.summary_minus_raw)
        self.summary_raw_difference_buckets.update(other.summary_raw_difference_buckets)


@dataclass
class ProfilerConfig:
    decode_non_apple: bool = True
    decode_source_keys: set[str] = field(default_factory=set)
    apple_source_key: str = "applehealthkit"
    numeric_sample_size: int = 4096
    category_top_k: int = 200
    max_nested_list_items: int = 128
    max_category_length: int = 256
    diagnostic_examples: int = 100

    def should_decode(self, source_key: str) -> bool:
        if self.decode_source_keys:
            return source_key in self.decode_source_keys
        return self.decode_non_apple and source_key != self.apple_source_key


@dataclass
class ProfilerState:
    config: ProfilerConfig
    participants: dict[str, ParticipantStats] = field(default_factory=dict)
    features: dict[tuple[str, str, str, str], FeatureStats] = field(default_factory=dict)
    fields: dict[tuple[str, str, str, str, str], FieldStats] = field(default_factory=dict)
    temporal: dict[tuple[str, str, str, str], TemporalStats] = field(default_factory=dict)
    temporal_alignment_counts: collections.Counter[tuple[str, str, str, str, str, str]] = field(default_factory=collections.Counter)
    participant_feature_version_months: dict[tuple[str, str, str, str, str, str], FeatureVersionMonthStats] = field(default_factory=dict)
    duplicate_calibration: dict[tuple[str, str, str, str, str, str], DuplicateCalibrationStats] = field(default_factory=dict)
    schema_counts: collections.Counter[tuple[str, str, str, str, str]] = field(default_factory=collections.Counter)
    schema_participants: dict[tuple[str, str, str, str, str], set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    representative_payloads: dict[tuple[str, str, str, str, str], dict[str, Any]] = field(default_factory=dict)
    source_labels: collections.Counter[tuple[str, str]] = field(default_factory=collections.Counter)
    activity: dict[str, ActivityStats] = field(default_factory=dict)
    activity_submeasurements: dict[tuple[str, str], ActivitySubmeasurementStats] = field(default_factory=dict)
    participant_activity: collections.Counter[tuple[str, str]] = field(default_factory=collections.Counter)
    provenance: collections.Counter[tuple[str, str, str, str, str, str, str, str]] = field(default_factory=collections.Counter)
    participant_provenance_values: dict[tuple[str, str, str], set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    participant_devices: dict[tuple[str, str], set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    participant_device_details: dict[tuple[str, str], set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    participant_apps: dict[tuple[str, str], set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    participant_streams: dict[tuple[str, str], set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    step_alternatives: dict[tuple[str, str], StepAlternativeStats] = field(default_factory=dict)
    focused_distributions: dict[tuple[str, str, str, str, str, str], FocusedDistributionStats] = field(default_factory=dict)
    weight_calibration: dict[tuple[str, str], WeightCalibrationStats] = field(default_factory=dict)
    malformed_item_counts: collections.Counter[tuple[str, str, str, str]] = field(default_factory=collections.Counter)
    malformed_item_participants: dict[tuple[str, str, str, str], set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    diagnostics: DiagnosticCollector = field(default_factory=DiagnosticCollector)
    outer_schema_issues: list[dict[str, Any]] = field(default_factory=list)
    invalid_files: list[dict[str, Any]] = field(default_factory=list)
    completed_participants: set[str] = field(default_factory=set)
    total_files: int = 0
    total_source_bytes: int = 0
    total_outer_rows: int = 0
    total_decoded_outer_rows: int = 0
    total_payload_items: int = 0
    participant_failures: dict[str, dict[str, Any]] = field(default_factory=dict)
    participant_runtime_seconds: dict[str, float] = field(default_factory=dict)
    participant_peak_rss_bytes: dict[str, int] = field(default_factory=dict)
    workers_requested: int = 1
    workers_effective: int = 1
    max_in_flight: int = 1
    execution_mode: str = "sequential"
    started_at_epoch: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.diagnostics.max_examples_per_code = self.config.diagnostic_examples


def merge_profiler_state(target: ProfilerState, source: ProfilerState) -> None:
    """
    Merge one fully scanned participant state into the cohort state. The executor calls this in a deterministic
    task order. Participant-owned dictionaries therefore cannot collide, while cohort aggregates are reduced
    through explicit merge operations rather than shared mutable state.
    """
    duplicate_participants = set(target.participants) & set(source.participants)
    if duplicate_participants:
        raise ValueError(
            "Participant profile merged more than once: "
            + ", ".join(sorted(duplicate_participants))
        )
    target.participants.update(source.participants)

    for key, incoming in source.features.items():
        existing = target.features.get(key)
        if existing is None:
            target.features[key] = incoming
        else:
            existing.merge(incoming)
    for key, incoming in source.fields.items():
        existing = target.fields.get(key)
        if existing is None:
            target.fields[key] = incoming
        else:
            existing.merge(incoming)
    for key, incoming in source.temporal.items():
        existing = target.temporal.get(key)
        if existing is None:
            target.temporal[key] = incoming
        else:
            existing.merge(incoming)
    target.temporal_alignment_counts.update(source.temporal_alignment_counts)
    for key, incoming in source.participant_feature_version_months.items():
        existing = target.participant_feature_version_months.get(key)
        if existing is None:
            target.participant_feature_version_months[key] = incoming
        else:
            existing.merge(incoming)
    for key, incoming in source.duplicate_calibration.items():
        existing = target.duplicate_calibration.get(key)
        if existing is None:
            target.duplicate_calibration[key] = incoming
        else:
            existing.merge(incoming)

    target.schema_counts.update(source.schema_counts)
    for key, participants in source.schema_participants.items():
        target.schema_participants[key].update(participants)
    for key, representative in source.representative_payloads.items():
        existing = target.representative_payloads.get(key)
        if existing is None or representative.get("stable_rank", "") < existing.get("stable_rank", ""):
            target.representative_payloads[key] = representative

    target.source_labels.update(source.source_labels)
    for key, incoming in source.activity.items():
        existing = target.activity.get(key)
        if existing is None:
            target.activity[key] = incoming
        else:
            existing.merge(incoming)
    for key, incoming in source.activity_submeasurements.items():
        existing = target.activity_submeasurements.get(key)
        if existing is None:
            target.activity_submeasurements[key] = incoming
        else:
            existing.merge(incoming)
    target.participant_activity.update(source.participant_activity)
    target.provenance.update(source.provenance)
    for key, values in source.participant_provenance_values.items():
        target.participant_provenance_values[key].update(values)
    for key, values in source.participant_devices.items():
        target.participant_devices[key].update(values)
    for key, values in source.participant_device_details.items():
        target.participant_device_details[key].update(values)
    for key, values in source.participant_apps.items():
        target.participant_apps[key].update(values)
    for key, values in source.participant_streams.items():
        target.participant_streams[key].update(values)
    for key, incoming in source.step_alternatives.items():
        existing = target.step_alternatives.get(key)
        if existing is None:
            target.step_alternatives[key] = incoming
        else:
            existing.merge(incoming)
    for key, incoming in source.focused_distributions.items():
        existing = target.focused_distributions.get(key)
        if existing is None:
            target.focused_distributions[key] = incoming
        else:
            existing.merge(incoming)
    for key, incoming in source.weight_calibration.items():
        existing = target.weight_calibration.get(key)
        if existing is None:
            target.weight_calibration[key] = incoming
        else:
            existing.merge(incoming)
    target.malformed_item_counts.update(source.malformed_item_counts)
    for key, participants in source.malformed_item_participants.items():
        target.malformed_item_participants[key].update(participants)

    target.diagnostics.merge(source.diagnostics)
    target.outer_schema_issues.extend(source.outer_schema_issues)
    target.invalid_files.extend(source.invalid_files)
    target.completed_participants.update(source.completed_participants)
    target.total_files += source.total_files
    target.total_source_bytes += source.total_source_bytes
    target.total_outer_rows += source.total_outer_rows
    target.total_decoded_outer_rows += source.total_decoded_outer_rows
    target.total_payload_items += source.total_payload_items


class MixedSourceProfiler:
    def __init__(self, raw_input: RawInput, state: ProfilerState):
        self.raw_input = raw_input
        self.state = state
        self._duplicate_seen: dict[tuple[str, str, str, str, str, str], _SeenIdentity] = {}

    def scan_participant(self, participant_files: ParticipantFiles) -> None:
        pid = participant_files.participant_id
        # Duplicate/revision calibration is participant-scoped. Keeping these identity maps out of ProfilerState
        # bounds parent/checkpoint memory.
        self._duplicate_seen = {}
        participant = self.state.participants.setdefault(pid, ParticipantStats(pid))
        participant.export_ended_present = participant_files.export_ended_present
        for invalid in participant_files.invalid_csv_names:
            self.state.invalid_files.append({
                "participant_id": pid,
                "kind": "invalid_month_filename",
                "path": invalid,
                "details": "CSV filename does not match YYYY-M.csv or YYYY-MM.csv",
            })
        for month, names in participant_files.duplicate_months:
            self.state.invalid_files.append({
                "participant_id": pid,
                "kind": "duplicate_canonical_month",
                "path": ";".join(names),
                "details": month,
            })
            self.state.diagnostics.add("duplicate_canonical_month", {"participant_id": pid, "month": month, "files": names})
        for file in participant_files.files:
            participant.file_count += 1
            participant.source_bytes += file.size_bytes
            participant.months.add(file.canonical_month)
            self.state.total_files += 1
            self.state.total_source_bytes += file.size_bytes
            self._scan_file(file, participant)
        if not participant_files.export_ended_present:
            self.state.diagnostics.add("export_ended_missing", {"participant_id": pid})
        self.state.completed_participants.add(pid)

    def _observe_duplicate_candidate(
        self, *, raw_source: str, source_key: str, feature: str, version: str, level: str, candidate: str,
        identity_value: Any | None, content_value: Any, identity_digest: str | None = None,
        content_digest: str | None = None, canonical_month: str, context: Mapping[str, Any],
    ) -> None:
        stats_key = (raw_source, source_key, feature, version, level, candidate)
        stats = self.state.duplicate_calibration.get(stats_key)
        if stats is None:
            stats = DuplicateCalibrationStats()
            self.state.duplicate_calibration[stats_key] = stats
        stats.observations += 1
        if identity_value in (None, ""):
            stats.missing_identity_observations += 1
            return

        identity_digest = identity_digest or _fingerprint(identity_value)
        content_digest = content_digest or _fingerprint(content_value)
        tracker_key = (source_key, feature, version, level, candidate, identity_digest)
        seen = self._duplicate_seen.get(tracker_key)
        if seen is None:
            self._duplicate_seen[tracker_key] = _SeenIdentity(
                content_fingerprints={content_digest},
                months={canonical_month},
            )
            stats.unique_identity_keys += 1
            stats.distinct_content_fingerprints += 1
            return

        stats.repeated_occurrences += 1
        if not seen.repeated_group_counted:
            stats.repeated_groups += 1
            seen.repeated_group_counted = True
        location = "within_month" if canonical_month in seen.months else "cross_month"
        if location == "within_month":
            stats.within_month_repeat_occurrences += 1
        else:
            stats.cross_month_repeat_occurrences += 1

        exact = content_digest in seen.content_fingerprints
        if exact:
            stats.exact_repeat_occurrences += 1
            outcome = "exact_repeat"
        else:
            stats.changed_content_occurrences += 1
            stats.distinct_content_fingerprints += 1
            seen.content_fingerprints.add(content_digest)
            outcome = "changed_content"
        previous_months = sorted(seen.months)
        seen.months.add(canonical_month)
        self.state.diagnostics.add(
            f"duplicate_calibration_{level}_{candidate}_{outcome}",
            {
                **dict(context),
                "candidate_level": level,
                "candidate_definition": candidate,
                "identity_fingerprint": identity_digest,
                "content_fingerprint": content_digest,
                "repeat_location": location,
                "previous_months": previous_months,
                "current_month": canonical_month,
            },
        )

    def _scan_file(self, file: RawFile, participant: ParticipantStats) -> None:
        try:
            with self.raw_input.open_text(file) as handle:
                reader = csv.DictReader(handle)
                header = tuple(reader.fieldnames or ())
                missing = sorted(set(EXPECTED_COLUMNS) - set(header))
                extra = sorted(set(header) - set(EXPECTED_COLUMNS))
                if missing or extra:
                    issue = {
                        "participant_id": file.participant_id,
                        "canonical_month": file.canonical_month,
                        "source_file": file.member_path,
                        "missing_columns": ";".join(missing),
                        "extra_columns": ";".join(extra),
                    }
                    self.state.outer_schema_issues.append(issue)
                    self.state.diagnostics.add("outer_schema_issue", issue)
                for row_number, row in enumerate(reader, start=2):
                    if None in row:
                        self.state.diagnostics.add("outer_row_extra_fields", {
                            "participant_id": file.participant_id,
                            "source_file": file.member_path,
                            "row_number": row_number,
                            "extra_field_count": len(row.get(None) or []),
                        })
                    self._scan_outer_row(file, row_number, row, participant)
        except (csv.Error, UnicodeError, OSError) as exc:
            self.state.diagnostics.add("monthly_file_read_failure", {
                "participant_id": file.participant_id,
                "source_file": file.member_path,
                "error": f"{type(exc).__name__}: {exc}",
            })

    def _scan_outer_row(
        self, file: RawFile, row_number: int, row: Mapping[str | None, Any], participant: ParticipantStats,
    ) -> None:
        raw_source = str(row.get("data_source") or "<missing>").strip() or "<missing>"
        source_key = normalize_source(raw_source)
        feature = str(row.get("name") or "<missing>").strip() or "<missing>"
        version = str(row.get("collecting_method_version") or "<missing>").strip() or "<missing>"
        outer_datetime = str(row.get("datetime") or "").strip()
        participant_id_value = str(row.get("participant_id") or "").strip()
        if participant_id_value and participant_id_value != file.participant_id:
            self.state.diagnostics.add("participant_id_mismatch", {
                "folder_participant_id": file.participant_id,
                "row_participant_id": participant_id_value,
                "source_file": file.member_path,
                "row_number": row_number,
            })

        participant.outer_rows += 1
        self.state.total_outer_rows += 1
        self.state.source_labels[(raw_source, source_key)] += 1
        source_slice = participant.source_slices.setdefault(source_key, SourceSlice())
        participant.raw_source_labels[source_key] = raw_source
        source_slice.rows += 1
        source_slice.months.add(file.canonical_month)
        source_slice.features.add(feature)
        source_slice.versions.add(version)
        source_slice.observe_datetime(outer_datetime)
        participant.month_source_rows[(file.canonical_month, source_key)] += 1
        participant.month_source_features[(file.canonical_month, source_key)].add(feature)

        feature_key = (raw_source, source_key, feature, version)
        stats = self.state.features.get(feature_key)
        if stats is None:
            stats = FeatureStats(raw_source, source_key, feature, version)
            self.state.features[feature_key] = stats
        stats.outer_rows += 1
        stats.participants.add(file.participant_id)
        stats.files.add((file.participant_id, file.canonical_month))
        stats.months.add(file.canonical_month)
        stats.observe_datetime(outer_datetime)

        version_month_key = (
            file.participant_id, raw_source, source_key, feature, version, file.canonical_month,
        )
        version_month = self.state.participant_feature_version_months.get(version_month_key)
        if version_month is None:
            version_month = FeatureVersionMonthStats()
            self.state.participant_feature_version_months[version_month_key] = version_month
        version_month.outer_rows += 1
        version_month.observe_datetime(outer_datetime)

        if not self.state.config.should_decode(source_key):
            return

        row_content = {column: row.get(column) for column in EXPECTED_COLUMNS}
        row_without_id = {key: value for key, value in row_content.items() if key != "id"}
        row_content_digest = _fingerprint(row_content)
        row_without_id_digest = _fingerprint(row_without_id)
        duplicate_context = {
            "participant_id": file.participant_id,
            "source_file": file.member_path,
            "row_number": row_number,
            "canonical_month": file.canonical_month,
            "data_source": raw_source,
            "feature": feature,
            "collecting_method_version": version,
            "outer_id": row.get("id"),
            "outer_datetime": outer_datetime,
        }
        self._observe_duplicate_candidate(
            raw_source=raw_source,
            source_key=source_key,
            feature=feature,
            version=version,
            level="outer_row",
            candidate="outer_id",
            identity_value=row.get("id"),
            content_value=row_content,
            content_digest=row_content_digest,
            canonical_month=file.canonical_month,
            context=duplicate_context,
        )
        self._observe_duplicate_candidate(
            raw_source=raw_source,
            source_key=source_key,
            feature=feature,
            version=version,
            level="outer_row",
            candidate="exact_content_without_id",
            identity_value=row_without_id,
            content_value=row_without_id,
            identity_digest=row_without_id_digest,
            content_digest=row_without_id_digest,
            canonical_month=file.canonical_month,
            context=duplicate_context,
        )
        self._observe_duplicate_candidate(
            raw_source=raw_source,
            source_key=source_key,
            feature=feature,
            version=version,
            level="outer_row",
            candidate="feature_datetime",
            identity_value={"feature": feature, "version": version, "datetime": outer_datetime},
            content_value=row_content,
            content_digest=row_content_digest,
            canonical_month=file.canonical_month,
            context=duplicate_context,
        )

        payload_text = row.get("data")
        try:
            payload = decode_payload(payload_text)
        except Exception as exc:
            stats.decode_failures += 1
            self.state.diagnostics.add("payload_decode_failure", {
                "participant_id": file.participant_id,
                "source_file": file.member_path,
                "row_number": row_number,
                "data_source": raw_source,
                "feature": feature,
                "collecting_method_version": version,
                "error": f"{type(exc).__name__}: {exc}",
            })
            return

        participant.decoded_outer_rows += 1
        self.state.total_decoded_outer_rows += 1
        stats.decoded_outer_rows += 1
        top_type = _type_name(payload)
        stats.payload_top_types[top_type] += 1
        if isinstance(payload, list):
            stats.list_lengths.add(len(payload))
        items = list(iter_payload_items(payload))
        stats.payload_items += len(items)
        participant.payload_items += len(items)
        self.state.total_payload_items += len(items)
        version_month.decoded_outer_rows += 1
        version_month.payload_items += len(items)

        self._profile_step_alternatives(
            file=file,
            row_number=row_number,
            raw_source=raw_source,
            source_key=source_key,
            feature=feature,
            version=version,
            items=items,
        )
        for payload_index, item in enumerate(items):
            self._profile_item(
                file=file,
                row_number=row_number,
                payload_index=payload_index,
                raw_source=raw_source,
                source_key=source_key,
                feature=feature,
                version=version,
                outer_datetime=outer_datetime,
                outer_id=str(row.get("id") or ""),
                item=item,
            )

    def _profile_item(
        self, *, file: RawFile, row_number: int, payload_index: int, raw_source: str, source_key: str, feature: str,
        version: str, outer_datetime: str, outer_id: str, item: dict[str, Any],
    ) -> None:
        signature = self._schema_signature(item)
        schema_key = (raw_source, source_key, feature, version, signature)
        self.state.schema_counts[schema_key] += 1
        self.state.schema_participants[schema_key].add(file.participant_id)
        representative = {
            "raw_source_label": raw_source,
            "source_key": source_key,
            "feature": feature,
            "collecting_method_version": version,
            "participant_id": file.participant_id,
            "canonical_month": file.canonical_month,
            "source_file": file.member_path,
            "row_number": row_number,
            "payload_index": payload_index,
            "item_schema": signature,
            "item": _bounded_example(item),
        }
        # Rank on source-independent scientific context. The source member path remains in the emitted example
        # for traceability, but moving the same raw root or reading it through a ZIP must not change which
        # payload is selected for a schema.
        representative_rank = stable_rank({
            key: value
            for key, value in representative.items()
            if key != "source_file"
        })
        representative["stable_rank"] = representative_rank
        previous = self.state.representative_payloads.get(schema_key)
        if previous is None or representative_rank < previous.get("stable_rank", ""):
            self.state.representative_payloads[schema_key] = representative

        context_base = f"{file.participant_id}|{file.canonical_month}|{file.basename}|{row_number}|{payload_index}"
        item_context = {
            "participant_id": file.participant_id,
            "source_file": file.member_path,
            "canonical_month": file.canonical_month,
            "row_number": row_number,
            "payload_index": payload_index,
            "data_source": raw_source,
            "feature": feature,
            "collecting_method_version": version,
            "outer_id": outer_id,
            "item_schema": signature,
        }
        item_digest = _fingerprint(item)
        self._observe_duplicate_candidate(
            raw_source=raw_source,
            source_key=source_key,
            feature=feature,
            version=version,
            level="payload_item",
            candidate="exact_content",
            identity_value=item,
            content_value=item,
            identity_digest=item_digest,
            content_digest=item_digest,
            canonical_month=file.canonical_month,
            context=item_context,
        )
        self._observe_duplicate_candidate(
            raw_source=raw_source,
            source_key=source_key,
            feature=feature,
            version=version,
            level="payload_item",
            candidate="outer_id_payload_index",
            identity_value={"outer_id": outer_id, "payload_index": payload_index} if outer_id else None,
            content_value=item,
            content_digest=item_digest,
            canonical_month=file.canonical_month,
            context=item_context,
        )
        record_identifier = _record_identifier(item)
        self._observe_duplicate_candidate(
            raw_source=raw_source,
            source_key=source_key,
            feature=feature,
            version=version,
            level="payload_item",
            candidate="record_identifier",
            identity_value={record_identifier[0]: record_identifier[1]} if record_identifier else None,
            content_value=item,
            content_digest=item_digest,
            canonical_month=file.canonical_month,
            context=item_context,
        )
        semantic_projection = _identity_projection(item)
        self._observe_duplicate_candidate(
            raw_source=raw_source,
            source_key=source_key,
            feature=feature,
            version=version,
            level="payload_item",
            candidate="semantic_time_and_provenance",
            identity_value=semantic_projection or None,
            content_value=item,
            content_digest=item_digest,
            canonical_month=file.canonical_month,
            context=item_context,
        )

        shape_warning = item.get("_payload_shape_warning")
        if shape_warning not in (None, ""):
            self.state.diagnostics.add("payload_shape_warning", {**item_context, "warning": shape_warning})
        malformed_reasons: set[str] = set()
        if shape_warning and not (feature == "StepCount" and version == "1.0" and shape_warning == "scalar_payload"):
            malformed_reasons.add(str(shape_warning))
        if feature in {"Weight", "Height"} and _numeric_value(item.get("value")) is None:
            malformed_reasons.add("missing_or_nonnumeric_value")
        for malformed_reason in sorted(malformed_reasons):
            malformed_key = (raw_source, feature, version, malformed_reason)
            self.state.malformed_item_counts[malformed_key] += 1
            self.state.malformed_item_participants[malformed_key].add(file.participant_id)
            self.state.diagnostics.add(
                "malformed_payload_item",
                {**item_context, "reason": malformed_reason, "item": _bounded_example(item)},
            )

        self._walk_value(
            value=item,
            path="",
            depth=0,
            context_base=context_base,
            raw_source=raw_source,
            source_key=source_key,
            feature=feature,
            version=version,
            participant_id=file.participant_id,
        )
        self._profile_temporal(
            raw_source, source_key, feature, version, item, context_base, outer_datetime, item_context,
        )
        if source_key == "googlefitness":
            self._profile_google_item(
                file, row_number, payload_index, raw_source, source_key, feature, version, item,
                signature, context_base,
            )

    def _schema_signature(self, item: Any) -> str:
        paths: set[str] = set()

        def visit(value: Any, path: str, depth: int) -> None:
            kind = _type_name(value)
            paths.add(f"{path or '<item>'}:{kind}")
            if depth >= 6:
                return
            if isinstance(value, dict):
                for key in sorted(value, key=lambda x: str(x)):
                    child = f"{path}.{key}" if path else str(key)
                    visit(value[key], child, depth + 1)
            elif isinstance(value, list):
                for child_value in value[: self.state.config.max_nested_list_items]:
                    visit(child_value, f"{path}[]", depth + 1)

        visit(item, "", 0)
        digest_source = "|".join(sorted(paths))
        if len(digest_source) <= 4000:
            return digest_source
        digest = hashlib.sha256(digest_source.encode("utf-8", "surrogatepass")).hexdigest()
        return f"{digest_source[:3000]}…[sha256={digest}]"

    def _walk_value(
        self, *, value: Any, path: str, depth: int, context_base: str, raw_source: str, source_key: str,
        feature: str, version: str, participant_id: str,
    ) -> None:
        if path:
            key = (raw_source, source_key, feature, version, path)
            stats = self.state.fields.get(key)
            if stats is None:
                stats = FieldStats(
                    raw_source, source_key, feature, version, path,
                    numeric=NumericSketch(sample_size=self.state.config.numeric_sample_size),
                    categorical=BoundedCounter(capacity=self.state.config.category_top_k),
                )
                self.state.fields[key] = stats
            stats.present_count += 1
            kind = _type_name(value)
            stats.types[kind] += 1
            if value is None:
                stats.null_count += 1
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                stats.numeric.add(value, f"{context_base}|{path}|{value!r}")
            elif isinstance(value, (str, bool)):
                text = str(value)
                if len(text) <= self.state.config.max_category_length:
                    stats.categorical.add(text)
                else:
                    stats.string_too_long_count += 1
            elif isinstance(value, list):
                stats.list_lengths.add(len(value), f"{context_base}|{path}|length")
                if len(value) > self.state.config.max_nested_list_items:
                    stats.nested_list_truncated_count += 1
            self._capture_provenance(
                participant_id=participant_id,
                raw_source=raw_source,
                source_key=source_key,
                feature=feature,
                version=version,
                path=path,
                value=value,
            )

        if depth >= 8:
            return
        if isinstance(value, dict):
            self._capture_device_dict(participant_id, source_key, path, value)
            for raw_key, child in value.items():
                child_path = f"{path}.{raw_key}" if path else str(raw_key)
                self._walk_value(
                    value=child,
                    path=child_path,
                    depth=depth + 1,
                    context_base=context_base,
                    raw_source=raw_source,
                    source_key=source_key,
                    feature=feature,
                    version=version,
                    participant_id=participant_id,
                )
        elif isinstance(value, list):
            limit = self.state.config.max_nested_list_items
            for child in value[:limit]:
                self._walk_value(
                    value=child,
                    path=f"{path}[]",
                    depth=depth + 1,
                    context_base=context_base,
                    raw_source=raw_source,
                    source_key=source_key,
                    feature=feature,
                    version=version,
                    participant_id=participant_id,
                )

    def _capture_provenance(
        self, *, participant_id: str, raw_source: str, source_key: str, feature: str, version: str, path: str,
        value: Any,
    ) -> None:
        leaf = path.rsplit(".", 1)[-1].replace("[]", "").casefold()
        dimension = PROVENANCE_DIMENSION_BY_LEAF.get(leaf)
        if dimension is None:
            parent_tokens = [
                token.replace("[]", "").casefold()
                for token in path.split(".")[:-1]
            ]
            if any("device" in token for token in parent_tokens):
                dimension = NESTED_DEVICE_DIMENSION_BY_LEAF.get(leaf)
        if dimension is None or isinstance(value, (dict, list)) or value in (None, ""):
            return
        text = _safe_text(value)
        self.state.provenance[
            (raw_source, source_key, feature, version, dimension, path, text, participant_id)
        ] += 1
        participant_key = (participant_id, source_key)
        self.state.participant_provenance_values[(participant_id, source_key, dimension)].add(text)

        # Retain the original coarse groups for report compatibility, but do not conflate source streams with
        # applications in the corrected reports.
        if dimension == "coarse_device":
            self.state.participant_devices[participant_key].add(text)
        if dimension in {"app_package", "app_name", "added_by"}:
            self.state.participant_apps[participant_key].add(text)
        if dimension in {"source_id", "data_source_id", "origin_data_source_id"}:
            self.state.participant_streams[participant_key].add(text)

    def _capture_device_dict(self, participant_id: str, source_key: str, path: str, value: dict[str, Any]) -> None:
        normalized = {str(key).casefold(): child for key, child in value.items()}
        subset: dict[str, Any] = {}
        for raw_key, canonical_key in DEVICE_DETAIL_KEY_ALIASES.items():
            child = normalized.get(raw_key)
            if child not in (None, ""):
                subset[canonical_key] = child
        # Require at least one device-specific attribute. A generic top-level ``type`` field alone is not
        # sufficient to declare a device identity.
        if not subset or not ({"manufacturer", "model", "uid", "name"} & set(subset)):
            return
        text = _safe_text({key: subset[key] for key in sorted(subset)})
        participant_key = (participant_id, source_key)
        self.state.participant_device_details[participant_key].add(text)
        self.state.participant_provenance_values[(participant_id, source_key, "detailed_device")].add(text)

    def _record_temporal_alignment(
        self, *, raw_source: str, source_key: str, feature: str, version: str, relationship: str, value: Any,
        context: Mapping[str, Any] | None = None, discrepancy: bool = False,
    ) -> None:
        text = str(value)
        self.state.temporal_alignment_counts[
            (raw_source, source_key, feature, version, relationship, text)
        ] += 1
        if discrepancy and context is not None:
            self.state.diagnostics.add(
                "temporal_alignment_discrepancy",
                {**dict(context), "relationship": relationship, "alignment_value": text},
            )

    def _profile_temporal(
        self, raw_source: str, source_key: str, feature: str, version: str, item: Mapping[str, Any],
        context_base: str, outer_datetime: str, context: Mapping[str, Any],
    ) -> None:
        key = (raw_source, source_key, feature, version)
        stats = self.state.temporal.get(key)
        if stats is None:
            stats = TemporalStats(
                raw_source, source_key, feature, version,
                duration=NumericSketch(sample_size=self.state.config.numeric_sample_size),
            )
            self.state.temporal[key] = stats
        stats.item_count += 1
        start_raw = item.get("start_date", item.get("start"))
        end_raw = item.get("end_date", item.get("end"))
        start_info = parse_time_info(start_raw)
        end_info = parse_time_info(end_raw)
        outer_info = parse_time_info(outer_datetime)
        if start_raw not in (None, ""):
            stats.start_present += 1
            if start_info is None:
                stats.invalid_start += 1
        if end_raw not in (None, ""):
            stats.end_present += 1
            if end_info is None:
                stats.invalid_end += 1
        if start_info is not None and end_info is not None:
            duration = (end_info.utc - start_info.utc).total_seconds()
            stats.negative_duration += int(duration < 0)
            stats.zero_duration += int(duration == 0)
            stats.duration.add(duration, f"{context_base}|duration|{duration!r}")
            span = _signed_day_delta(start_info.utc.date(), end_info.utc.date())
            self._record_temporal_alignment(
                raw_source=raw_source,
                source_key=source_key,
                feature=feature,
                version=version,
                relationship="start_to_end_utc_date_delta_days",
                value=span,
                context=context,
                discrepancy=span < 0,
            )

        for name, parsed in (("outer", outer_info), ("start", start_info), ("end", end_info)):
            if parsed is not None:
                offset_value = parsed.offset_minutes if parsed.offset_explicit else "<not_explicit>"
                self._record_temporal_alignment(
                    raw_source=raw_source,
                    source_key=source_key,
                    feature=feature,
                    version=version,
                    relationship=f"{name}_timezone_offset_minutes",
                    value=offset_value,
                )

        if outer_info is not None and start_info is not None:
            utc_delta = _signed_day_delta(outer_info.utc.date(), start_info.utc.date())
            local_delta = _signed_day_delta(outer_info.local.date(), start_info.local.date())
            for relationship, delta in (
                ("outer_to_start_utc_date_delta_days", utc_delta),
                ("outer_to_start_local_date_delta_days", local_delta),
            ):
                self._record_temporal_alignment(
                    raw_source=raw_source,
                    source_key=source_key,
                    feature=feature,
                    version=version,
                    relationship=relationship,
                    value=delta,
                    context=context,
                    discrepancy=delta != 0,
                )
        if outer_info is not None and end_info is not None:
            utc_delta = _signed_day_delta(outer_info.utc.date(), end_info.utc.date())
            local_delta = _signed_day_delta(outer_info.local.date(), end_info.local.date())
            for relationship, delta in (
                ("outer_to_end_utc_date_delta_days", utc_delta),
                ("outer_to_end_local_date_delta_days", local_delta),
            ):
                self._record_temporal_alignment(
                    raw_source=raw_source,
                    source_key=source_key,
                    feature=feature,
                    version=version,
                    relationship=relationship,
                    value=delta,
                    context=context,
                    discrepancy=abs(delta) > 1,
                )

        summary_dates: list[tuple[str, date]] = []
        for field_name in ("summary_date", "date"):
            parsed_date = parse_date(item.get(field_name))
            if parsed_date is not None:
                summary_dates.append((field_name, parsed_date))
        steps = item.get("steps")
        if isinstance(steps, list):
            for index, step in enumerate(steps):
                if isinstance(step, Mapping):
                    parsed_date = parse_date(step.get("date"))
                    if parsed_date is not None:
                        summary_dates.append((f"steps[{index}].date", parsed_date))
        seen_summary_dates: set[tuple[str, date]] = set()
        for date_path, summary_date in summary_dates:
            if (date_path, summary_date) in seen_summary_dates:
                continue
            seen_summary_dates.add((date_path, summary_date))
            anchors = (
                ("outer_utc", outer_info.utc.date() if outer_info else None),
                ("outer_local", outer_info.local.date() if outer_info else None),
                ("start_utc", start_info.utc.date() if start_info else None),
                ("start_local", start_info.local.date() if start_info else None),
            )
            for anchor_name, anchor_date in anchors:
                if anchor_date is None:
                    continue
                delta = _signed_day_delta(anchor_date, summary_date)
                self._record_temporal_alignment(
                    raw_source=raw_source,
                    source_key=source_key,
                    feature=feature,
                    version=version,
                    relationship=f"{date_path}_to_{anchor_name}_date_delta_days",
                    value=delta,
                    context={**dict(context), "summary_date": summary_date.isoformat(), "date_path": date_path},
                    discrepancy=delta != 0,
                )

        day_label = item.get("day")
        if day_label not in (None, ""):
            label_index = weekday_index(day_label)
            self._record_temporal_alignment(
                raw_source=raw_source,
                source_key=source_key,
                feature=feature,
                version=version,
                relationship="weekday_label_recognition",
                value="recognized" if label_index is not None else "unrecognized",
                context={**dict(context), "day_label": day_label},
                discrepancy=label_index is None,
            )
            if label_index is None:
                self.state.diagnostics.add(
                    "weekday_label_unrecognized", {**dict(context), "day_label": day_label},
                )
            else:
                for anchor_name, parsed in (("outer_local", outer_info), ("start_local", start_info)):
                    if parsed is None:
                        continue
                    delta = _normalize_weekday_delta(label_index - parsed.local.weekday())
                    self._record_temporal_alignment(
                        raw_source=raw_source,
                        source_key=source_key,
                        feature=feature,
                        version=version,
                        relationship=f"weekday_label_to_{anchor_name}_delta_days",
                        value=delta,
                        context={**dict(context), "day_label": day_label, "anchor_date": parsed.local.date().isoformat()},
                        discrepancy=delta != 0,
                    )
                    if delta != 0:
                        self.state.diagnostics.add(
                            "weekday_label_date_mismatch",
                            {
                                **dict(context),
                                "day_label": day_label,
                                "anchor": anchor_name,
                                "anchor_date": parsed.local.date().isoformat(),
                                "signed_delta_days": delta,
                            },
                        )

    def _observe_focused_distribution(
        self, *, raw_source: str, source_key: str, feature: str, version: str, distribution: str, bucket: str,
        participant_id: str, value: float | None, context: str,
    ) -> None:
        key = (raw_source, source_key, feature, version, distribution, bucket)
        stats = self.state.focused_distributions.get(key)
        if stats is None:
            stats = FocusedDistributionStats(values=NumericSketch(sample_size=self.state.config.numeric_sample_size))
            self.state.focused_distributions[key] = stats
        stats.count += 1
        stats.participants.add(participant_id)
        if value is not None:
            stats.values.add(value, context)

    def _profile_google_item(
        self, file: RawFile, row_number: int, payload_index: int, raw_source: str, source_key: str, feature: str,
        version: str, item: Mapping[str, Any], signature: str, context_base: str,
    ) -> None:
        if feature == "Activities":
            label = str(item.get("activity_name") or "<missing>")
            stats = self.state.activity.get(label)
            if stats is None:
                stats = ActivityStats(label, duration=NumericSketch(sample_size=self.state.config.numeric_sample_size))
                self.state.activity[label] = stats
            stats.count += 1
            stats.participants.add(file.participant_id)
            stats.files.add((file.participant_id, file.canonical_month))
            stats.schemas[signature] += 1
            for key, value in item.items():
                if value is not None:
                    stats.field_presence[str(key)] += 1
            self.state.participant_activity[(file.participant_id, label)] += 1

            detailed_key = (label, version)
            detailed = self.state.activity_submeasurements.get(detailed_key)
            if detailed is None:
                detailed = ActivitySubmeasurementStats(
                    label=label,
                    version=version,
                    tracked_values=BoundedCounter(capacity=self.state.config.category_top_k),
                    device_values=BoundedCounter(capacity=self.state.config.category_top_k),
                    source_id_values=BoundedCounter(capacity=self.state.config.category_top_k),
                    source_name_values=BoundedCounter(capacity=self.state.config.category_top_k),
                )
                self.state.activity_submeasurements[detailed_key] = detailed
            detailed.item_count += 1
            detailed.participants.add(file.participant_id)
            detailed.files.add((file.participant_id, file.canonical_month))
            for field_name in ("calories", "distance", "quantity", "tracked", "device", "source_id", "source_name"):
                if field_name in item and item.get(field_name) is not None:
                    detailed.field_presence[field_name] += 1
            for field_name in ("calories", "distance", "quantity"):
                number = _numeric_value(item.get(field_name))
                if number is not None:
                    detailed.numeric(field_name, self.state.config.numeric_sample_size).add(
                        number,
                        f"{context_base}|activity_{field_name}|{number!r}",
                    )
            if item.get("tracked") not in (None, ""):
                detailed.tracked_values.add(item.get("tracked"))
            if item.get("device") not in (None, ""):
                detailed.device_values.add(item.get("device"))
            if item.get("source_id") not in (None, ""):
                detailed.source_id_values.add(item.get("source_id"))
            if item.get("source_name") not in (None, ""):
                detailed.source_name_values.add(item.get("source_name"))

            start = parse_time(item.get("start"))
            end = parse_time(item.get("end"))
            if start is not None and end is not None:
                duration = (end - start).total_seconds()
                stats.duration.add(duration, f"{context_base}|activity_duration|{duration!r}")
                detailed.numeric("duration_seconds", self.state.config.numeric_sample_size).add(
                    duration,
                    f"{context_base}|activity_duration|{duration!r}",
                )

        elif feature == "Calories":
            value = _first_numeric(item, "calorie", "calories", "value")
            if value is not None and value < 0:
                bucket = _negative_calorie_bucket(value)
                self._observe_focused_distribution(
                    raw_source=raw_source,
                    source_key=source_key,
                    feature=feature,
                    version=version,
                    distribution="negative_calorie_value",
                    bucket=bucket,
                    participant_id=file.participant_id,
                    value=value,
                    context=f"{context_base}|negative_calorie|{value!r}",
                )
                self.state.diagnostics.add("negative_calorie_value", {
                    "participant_id": file.participant_id,
                    "source_file": file.member_path,
                    "row_number": row_number,
                    "payload_index": payload_index,
                    "collecting_method_version": version,
                    "value": value,
                    "magnitude_bucket": bucket,
                })

        elif feature == "Weight":
            added_by = str(item.get("added_by") or "<missing>")
            weight_key = (version, added_by)
            calibration = self.state.weight_calibration.get(weight_key)
            if calibration is None:
                calibration = WeightCalibrationStats(
                    version=version,
                    added_by=added_by,
                    value=NumericSketch(sample_size=self.state.config.numeric_sample_size),
                    duration=NumericSketch(sample_size=self.state.config.numeric_sample_size),
                )
                self.state.weight_calibration[weight_key] = calibration
            calibration.item_count += 1
            calibration.participants.add(file.participant_id)
            value = _numeric_value(item.get("value"))
            if value is None:
                calibration.malformed_count += 1
            else:
                calibration.value.add(value, f"{context_base}|weight_value|{value!r}")

            start = parse_time(item.get("start_date", item.get("start")))
            end = parse_time(item.get("end_date", item.get("end")))
            if start is None or end is None:
                calibration.invalid_time_count += 1
                return
            duration = (end - start).total_seconds()
            calibration.duration.add(duration, f"{context_base}|weight_duration|{duration!r}")
            bucket = _duration_bucket(duration)
            if duration == 0:
                calibration.point_count += 1
            else:
                calibration.nonpoint_count += 1
                self.state.diagnostics.add("weight_nonpoint_interval", {
                    "participant_id": file.participant_id,
                    "source_file": file.member_path,
                    "row_number": row_number,
                    "payload_index": payload_index,
                    "collecting_method_version": version,
                    "added_by": added_by,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "duration_seconds": duration,
                    "duration_bucket": bucket,
                })
            self._observe_focused_distribution(
                raw_source=raw_source,
                source_key=source_key,
                feature=feature,
                version=version,
                distribution="weight_interval_duration_seconds",
                bucket=bucket,
                participant_id=file.participant_id,
                value=duration,
                context=f"{context_base}|weight_duration|{duration!r}",
            )

    def _profile_step_alternatives(
        self, *, file: RawFile, row_number: int, raw_source: str, source_key: str, feature: str,
        version: str, items: list[dict[str, Any]],
    ) -> None:
        if source_key != "googlefitness" or feature != "StepCount":
            return
        stats = self.state.step_alternatives.setdefault(
            (raw_source, version),
            StepAlternativeStats(
                difference_merge_minus_estimated=NumericSketch(sample_size=self.state.config.numeric_sample_size),
                absolute_difference=NumericSketch(sample_size=self.state.config.numeric_sample_size),
                summary_minus_raw=NumericSketch(sample_size=self.state.config.numeric_sample_size),
            ),
        )
        stats.outer_rows += 1
        stats.alternatives_per_outer_row[len(items)] += 1
        values: dict[str, list[float]] = collections.defaultdict(list)
        for item_index, item in enumerate(items):
            name = str(item.get("source") or ("legacy_scalar" if version == "1.0" else "<missing>"))
            stats.alternative_names[name] += 1
            step_values = item.get("steps")
            summary_numbers: list[float] = []
            if isinstance(step_values, list):
                for entry in step_values:
                    if isinstance(entry, Mapping):
                        number = _numeric_value(entry.get("value"))
                        if number is not None:
                            summary_numbers.append(number)
            else:
                number = _numeric_value(item.get("value"))
                if number is not None:
                    summary_numbers.append(number)
            if summary_numbers:
                values[name].append(sum(summary_numbers))

            raw_numbers: list[float] = []
            raw_steps = item.get("raw_steps")
            if isinstance(raw_steps, list):
                for entry in raw_steps:
                    if isinstance(entry, Mapping):
                        number = _first_numeric(entry, "steps", "value")
                        if number is not None:
                            raw_numbers.append(number)
            if summary_numbers and raw_numbers:
                summary_total = sum(summary_numbers)
                raw_total = sum(raw_numbers)
                difference = summary_total - raw_total
                stats.summary_raw_pairs += 1
                if difference == 0:
                    stats.summary_raw_equal += 1
                else:
                    stats.summary_raw_different += 1
                bucket = _difference_bucket(difference)
                stats.summary_raw_difference_buckets[bucket] += 1
                stats.summary_minus_raw.add(
                    difference,
                    f"{file.participant_id}|{file.member_path}|{row_number}|{item_index}|summary_raw|{difference!r}",
                )
                if difference != 0:
                    self.state.diagnostics.add(
                        "step_summary_raw_disagree",
                        {
                            "participant_id": file.participant_id,
                            "source_file": file.member_path,
                            "row_number": row_number,
                            "item_index": item_index,
                            "alternative_name": name,
                            "summary_total": summary_total,
                            "raw_total": raw_total,
                            "difference": difference,
                            "difference_bucket": bucket,
                        },
                    )

        if len(items) > 1:
            stats.rows_with_multiple_alternatives += 1
        estimated_values = [
            value
            for name, numbers in values.items()
            if name.endswith(":estimated_steps") or name == "estimated_steps"
            for value in numbers
        ]
        merged_values = [
            value
            for name, numbers in values.items()
            if name.endswith(":merge_step_deltas") or name == "merge_step_deltas"
            for value in numbers
        ]
        estimated = estimated_values[0] if estimated_values else None
        merged = merged_values[0] if merged_values else None
        stats.rows_with_estimated += int(estimated is not None)
        stats.rows_with_merge += int(merged is not None)
        if estimated is not None and merged is not None:
            stats.rows_with_both += 1
            difference = merged - estimated
            bucket = _difference_bucket(difference)
            stats.difference_buckets[bucket] += 1
            if difference == 0:
                stats.rows_equal += 1
            else:
                stats.rows_different += 1
                self.state.diagnostics.add("step_aggregation_alternatives_disagree", {
                    "participant_id": file.participant_id,
                    "source_file": file.member_path,
                    "row_number": row_number,
                    "estimated_steps": estimated,
                    "merge_step_deltas": merged,
                    "difference": difference,
                    "difference_bucket": bucket,
                })
            context = f"{file.participant_id}|{file.member_path}|{row_number}|step_difference|{difference!r}"
            stats.difference_merge_minus_estimated.add(difference, context)
            stats.absolute_difference.add(abs(difference), context + "|absolute")
            self._observe_focused_distribution(
                raw_source=raw_source,
                source_key=source_key,
                feature=feature,
                version=version,
                distribution="merge_minus_estimated_steps",
                bucket=bucket,
                participant_id=file.participant_id,
                value=difference,
                context=context,
            )

    def source_membership_counts(self) -> collections.Counter[str]:
        counts: collections.Counter[str] = collections.Counter()
        for participant in self.state.participants.values():
            source_set = tuple(sorted(participant.source_slices))
            counts[" + ".join(source_set) if source_set else "<no_rows>"] += 1
        return counts

    def summary(self) -> dict[str, Any]:
        participant_directories = len(self.state.participants)
        data_bearing = sum(participant.outer_rows > 0 for participant in self.state.participants.values())
        empty_directories = sum(
            participant.file_count == 0 and participant.outer_rows == 0
            for participant in self.state.participants.values()
        )
        files_without_rows = sum(
            participant.file_count > 0 and participant.outer_rows == 0
            for participant in self.state.participants.values()
        )
        mixed = sum(len(participant.source_slices) > 1 for participant in self.state.participants.values())
        decoded_sources = sorted({key[1] for key, stats in self.state.features.items() if stats.decoded_outer_rows})
        runtimes = list(self.state.participant_runtime_seconds.values())
        peak_values = list(self.state.participant_peak_rss_bytes.values())
        status = "complete" if not self.state.participant_failures else "complete_with_errors"
        return {
            "tool_version": TOOL_VERSION,
            "status": status,
            # ``participants`` is retained as a compatibility alias. New reports
            # use the explicit directory/data-bearing terminology below.
            "participants": participant_directories,
            "participant_directories": participant_directories,
            "data_bearing_participants": data_bearing,
            "empty_export_directories": empty_directories,
            "participant_directories_with_month_files_but_no_rows": files_without_rows,
            "participants_completed": len(self.state.completed_participants),
            "participants_failed": len(self.state.participant_failures),
            "participants_attempted": len(self.state.completed_participants) + len(self.state.participant_failures),
            "monthly_files": self.state.total_files,
            "source_bytes": self.state.total_source_bytes,
            "outer_rows": self.state.total_outer_rows,
            "decoded_outer_rows": self.state.total_decoded_outer_rows,
            "payload_items": self.state.total_payload_items,
            "mixed_source_participants": mixed,
            "source_membership": dict(sorted(self.source_membership_counts().items())),
            "source_labels": {
                raw: count
                for (raw, _), count in sorted(self.state.source_labels.items(), key=lambda item: item[0])
            },
            "decoded_source_keys": decoded_sources,
            "payload_decode_failures": self.state.diagnostics.counts.get("payload_decode_failure", 0),
            "outer_schema_issue_files": len(self.state.outer_schema_issues),
            "invalid_or_duplicate_month_files": len(self.state.invalid_files),
            "export_ended_missing_participants": self.state.diagnostics.counts.get("export_ended_missing", 0),
            "workers_requested": self.state.workers_requested,
            "workers_effective": self.state.workers_effective,
            "max_in_flight": self.state.max_in_flight,
            "execution_mode": self.state.execution_mode,
            "participant_runtime_seconds_sum": sum(runtimes),
            "participant_runtime_seconds_max": max(runtimes) if runtimes else None,
            "participant_peak_rss_bytes_max": max(peak_values) if peak_values else None,
            "elapsed_seconds": time.time() - self.state.started_at_epoch,
        }
