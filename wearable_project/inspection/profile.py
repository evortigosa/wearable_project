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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from .input_source import ParticipantFiles, RawFile, RawInput
from .sketches import BoundedCounter, NumericSketch


TOOL_VERSION = "mixed-source-profiler-1.1.1"
CHECKPOINT_COMPATIBLE_TOOL_VERSIONS = frozenset({"mixed-source-profiler-1.1.0", TOOL_VERSION,})
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

PROVENANCE_KEYS = {
    "device",
    "device_name",
    "manufacturer",
    "model",
    "type",
    "uid",
    "source_id",
    "source_name",
    "data_source_id",
    "origin_data_source_id",
    "added_by",
    "package_name",
    "application",
    "app",
    "tracked",
    "was_user_entered",
    "user_entered",
}
DEVICE_KEYS = {"manufacturer", "model", "type", "uid", "device_name"}
APP_KEYS = {"source_id", "source_name", "added_by", "package_name", "application", "app"}
STREAM_KEYS = {"data_source_id", "origin_data_source_id"}


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


def parse_time(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        try:
            # Google payload timestamps in the sample are epoch milliseconds.
            if abs(number) > 10_000_000_000:
                number /= 1000.0
            return datetime.fromtimestamp(number, tz=timezone.utc)
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
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


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
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: collections.defaultdict(list))


    def add(self, code: str, context: Mapping[str, Any] | None = None, count: int = 1) -> None:
        self.counts[code] += count
        if context is not None and len(self.examples[code]) < self.max_examples_per_code:
            self.examples[code].append(dict(context))


    def merge(self, other: "DiagnosticCollector") -> None:
        self.counts.update(other.counts)
        for code in sorted(other.examples):
            remaining = self.max_examples_per_code - len(self.examples[code])
            if remaining > 0:
                self.examples[code].extend(other.examples[code][:remaining])


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
class StepAlternativeStats:
    outer_rows: int = 0
    rows_with_multiple_alternatives: int = 0
    rows_with_estimated: int = 0
    rows_with_merge: int = 0
    rows_with_both: int = 0
    rows_equal: int = 0
    rows_different: int = 0
    difference_merge_minus_estimated: NumericSketch = field(default_factory=NumericSketch)
    alternative_names: collections.Counter[str] = field(default_factory=collections.Counter)


    def merge(self, other: "StepAlternativeStats") -> None:
        self.outer_rows += other.outer_rows
        self.rows_with_multiple_alternatives += other.rows_with_multiple_alternatives
        self.rows_with_estimated += other.rows_with_estimated
        self.rows_with_merge += other.rows_with_merge
        self.rows_with_both += other.rows_with_both
        self.rows_equal += other.rows_equal
        self.rows_different += other.rows_different
        self.difference_merge_minus_estimated.merge(other.difference_merge_minus_estimated)
        self.alternative_names.update(other.alternative_names)


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
    schema_counts: collections.Counter[tuple[str, str, str, str, str]] = field(default_factory=collections.Counter)
    schema_participants: dict[tuple[str, str, str, str, str], set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    representative_payloads: dict[tuple[str, str, str, str, str], dict[str, Any]] = field(default_factory=dict)
    source_labels: collections.Counter[tuple[str, str]] = field(default_factory=collections.Counter)
    activity: dict[str, ActivityStats] = field(default_factory=dict)
    participant_activity: collections.Counter[tuple[str, str]] = field(default_factory=collections.Counter)
    provenance: collections.Counter[tuple[str, str, str, str, str]] = field(default_factory=collections.Counter)
    participant_devices: dict[tuple[str, str], set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    participant_device_details: dict[tuple[str, str], set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    participant_apps: dict[tuple[str, str], set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    participant_streams: dict[tuple[str, str], set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    step_alternatives: dict[tuple[str, str], StepAlternativeStats] = field(default_factory=dict)
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

    target.schema_counts.update(source.schema_counts)
    for key, participants in source.schema_participants.items():
        target.schema_participants[key].update(participants)
    for key, representative in source.representative_payloads.items():
        target.representative_payloads.setdefault(key, representative)

    target.source_labels.update(source.source_labels)
    for key, incoming in source.activity.items():
        existing = target.activity.get(key)
        if existing is None:
            target.activity[key] = incoming
        else:
            existing.merge(incoming)
    target.participant_activity.update(source.participant_activity)
    target.provenance.update(source.provenance)
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


    def scan_participant(self, participant_files: ParticipantFiles) -> None:
        pid = participant_files.participant_id
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

        if not self.state.config.should_decode(source_key):
            return
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
                item=item,
            )


    def _profile_item(
        self, *, file: RawFile, row_number: int, payload_index: int, raw_source: str, source_key: str, feature: str,
        version: str, outer_datetime: str, item: dict[str, Any],
    ) -> None:
        signature = self._schema_signature(item)
        schema_key = (raw_source, source_key, feature, version, signature)
        self.state.schema_counts[schema_key] += 1
        self.state.schema_participants[schema_key].add(file.participant_id)
        if schema_key not in self.state.representative_payloads:
            self.state.representative_payloads[schema_key] = {
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
        context_base = f"{file.participant_id}|{file.canonical_month}|{file.basename}|{row_number}|{payload_index}"
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
        self._profile_temporal(raw_source, source_key, feature, version, item, context_base, outer_datetime)
        if source_key == "googlefitness":
            self._profile_google_item(file, row_number, payload_index, feature, version, item, signature, context_base)

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
        self, *, participant_id: str, raw_source: str, source_key: str, feature: str, path: str, value: Any,
    ) -> None:
        leaf = path.rsplit(".", 1)[-1].replace("[]", "").casefold()
        if leaf not in PROVENANCE_KEYS or isinstance(value, (dict, list)) or value in (None, ""):
            return
        text = _safe_text(value)
        self.state.provenance[(raw_source, feature, path, text, participant_id)] += 1
        participant_key = (participant_id, source_key)
        if leaf == "device" or leaf == "device_name":
            self.state.participant_devices[participant_key].add(text)
        if leaf in APP_KEYS:
            self.state.participant_apps[participant_key].add(text)
        if leaf in STREAM_KEYS:
            self.state.participant_streams[participant_key].add(text)

    def _capture_device_dict(self, participant_id: str, source_key: str, path: str, value: dict[str, Any]) -> None:
        normalized = {str(key).casefold(): child for key, child in value.items()}
        if not (set(normalized) & DEVICE_KEYS):
            return
        subset = {key: normalized[key] for key in sorted(set(normalized) & DEVICE_KEYS) if normalized[key] not in (None, "")}
        if subset:
            self.state.participant_device_details[(participant_id, source_key)].add(_safe_text(subset))


    def _profile_temporal(
        self, raw_source: str, source_key: str, feature: str, version: str, item: Mapping[str, Any],
        context_base: str, outer_datetime: str,
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
        start = parse_time(start_raw)
        end = parse_time(end_raw)
        if start_raw not in (None, ""):
            stats.start_present += 1
            if start is None:
                stats.invalid_start += 1
        if end_raw not in (None, ""):
            stats.end_present += 1
            if end is None:
                stats.invalid_end += 1
        if start is not None and end is not None:
            duration = (end - start).total_seconds()
            stats.negative_duration += int(duration < 0)
            stats.zero_duration += int(duration == 0)
            stats.duration.add(duration, f"{context_base}|duration|{duration!r}")
        day = item.get("day")
        outer = parse_time(outer_datetime)
        if day not in (None, "") and outer is not None:
            english = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
            hebrew = ("יום ב׳", "יום ג׳", "יום ד׳", "יום ה׳", "יום ו׳", "שבת", "יום א׳")
            expected = {english[outer.weekday()], hebrew[outer.weekday()]}
            if str(day).strip() not in expected:
                self.state.diagnostics.add("day_label_outer_datetime_mismatch", {
                    "data_source": raw_source,
                    "feature": feature,
                    "day_label": day,
                    "outer_datetime": outer_datetime,
                    "expected_labels": sorted(expected),
                })


    def _profile_google_item(
        self, file: RawFile, row_number: int, payload_index: int, feature: str, version: str,
        item: Mapping[str, Any], signature: str, context_base: str,
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
            start = parse_time(item.get("start"))
            end = parse_time(item.get("end"))
            if start is not None and end is not None:
                duration = (end - start).total_seconds()
                stats.duration.add(duration, f"{context_base}|activity_duration|{duration!r}")
        elif feature == "Calories":
            value = item.get("calorie", item.get("calories"))
            if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) < 0:
                self.state.diagnostics.add("negative_calorie_value", {
                    "participant_id": file.participant_id,
                    "source_file": file.member_path,
                    "row_number": row_number,
                    "payload_index": payload_index,
                    "value": value,
                })
        elif feature == "Weight":
            start = parse_time(item.get("start_date", item.get("start")))
            end = parse_time(item.get("end_date", item.get("end")))
            if start is not None and end is not None and start != end:
                self.state.diagnostics.add("weight_nonpoint_interval", {
                    "participant_id": file.participant_id,
                    "source_file": file.member_path,
                    "row_number": row_number,
                    "payload_index": payload_index,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "duration_seconds": (end - start).total_seconds(),
                })


    def _profile_step_alternatives(
        self, *, file: RawFile, row_number: int, raw_source: str, source_key: str, feature: str,
        version: str, items: list[dict[str, Any]],
    ) -> None:
        if source_key != "googlefitness" or feature != "StepCount":
            return
        stats = self.state.step_alternatives.setdefault((raw_source, version), StepAlternativeStats())
        stats.outer_rows += 1
        values: dict[str, float] = {}
        for item in items:
            name = str(item.get("source") or ("legacy_scalar" if version == "1.0" else "<missing>"))
            stats.alternative_names[name] += 1
            step_values = item.get("steps")
            value: Any = None
            if isinstance(step_values, list) and step_values and isinstance(step_values[0], dict):
                value = step_values[0].get("value")
            elif isinstance(item.get("value"), (int, float)):
                value = item.get("value")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[name] = float(value)
        if len(items) > 1:
            stats.rows_with_multiple_alternatives += 1
        estimated = next((value for name, value in values.items() if name.endswith(":estimated_steps") or name == "estimated_steps"), None)
        merged = next((value for name, value in values.items() if name.endswith(":merge_step_deltas") or name == "merge_step_deltas"), None)
        stats.rows_with_estimated += int(estimated is not None)
        stats.rows_with_merge += int(merged is not None)
        if estimated is not None and merged is not None:
            stats.rows_with_both += 1
            difference = merged - estimated
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
                })
            stats.difference_merge_minus_estimated.add(
                difference,
                f"{file.participant_id}|{file.member_path}|{row_number}|step_difference|{difference!r}",
            )


    def source_membership_counts(self) -> collections.Counter[str]:
        counts: collections.Counter[str] = collections.Counter()
        for participant in self.state.participants.values():
            source_set = tuple(sorted(participant.source_slices))
            counts[" + ".join(source_set) if source_set else "<no_rows>"] += 1
        return counts


    def summary(self) -> dict[str, Any]:
        mixed = sum(len(participant.source_slices) > 1 for participant in self.state.participants.values())
        decoded_sources = sorted({key[1] for key, stats in self.state.features.items() if stats.decoded_outer_rows})
        runtimes = list(self.state.participant_runtime_seconds.values())
        peak_values = list(self.state.participant_peak_rss_bytes.values())
        status = "complete" if not self.state.participant_failures else "complete_with_errors"
        return {
            "tool_version": TOOL_VERSION,
            "status": status,
            "participants": len(self.state.participants),
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
