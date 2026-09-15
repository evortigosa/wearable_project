"""
Wearable Data Processing and Modeling project
Non-destructive feature-file curation engine. Native columns and row order are preserved. The engine performs
reviewed unit canonicalization and adds sparse, policy-derived annotations; it never deletes, clips,
interpolates, or resamples a native observation.
"""

from __future__ import annotations
import ast
import csv
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from wearable_project.curation.models import (
    CurationStatus, EventKind, InclusionPolicy, PolicyExecutionMode, UnresolvedUnitAction,
)
from wearable_project.curation.registry import get_policy, known_curation_features
from wearable_project.curation.unit_resolution import (
    ParticipantUnitContext, UnitAnnotation, fixed_unit_annotation,
)
from wearable_project.processing.parser import sha256_file
from wearable_project.processing.writer import drop_file_cache, link_or_copy


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_field_limit()


DERIVED_COLUMN_ORDER: tuple[str, ...] = (
    "canonical_value",
    "canonical_unit",
    "curation_unit_status",
    "unit_evidence",
    "unit_epoch_id",
    "acquisition_method",
    "curation_status",
    "curation_flags",
    "include_by_default",
)

_STATUS_RANK = {"pass": 0, "review": 1, "exclude_default": 2}
_KNOWN_SLEEP_STATES = {"INBED", "ASLEEP", "AWAKE", "CORE", "DEEP", "REM"}
_DETAILED_SLEEP_STATES = {"AWAKE", "CORE", "DEEP", "REM"}
_KNOWN_CGM_STATUS = {"IN_RANGE", "HIGH", "LOW", "UNKNOWN", ""}
_KNOWN_CGM_TRENDS = {
    "Flat", "FortyFiveUp", "FortyFiveDown", "SingleUp", "SingleDown",
    "DoubleUp", "DoubleDown", "NotComputable", "RateOutOfRange", "",
}


@dataclass(slots=True)
class RowAnnotation:
    values: dict[str, str] = field(default_factory=dict)
    status: str = "pass"
    include: bool = True
    flags: set[str] = field(default_factory=set)

    def escalate(self, status: str, *, exclude: bool = False, flag: str | None = None) -> None:
        if _STATUS_RANK[status] > _STATUS_RANK[self.status]:
            self.status = status
        if exclude:
            self.include = False
        if flag:
            self.flags.add(flag)

    def as_sparse_dict(self) -> dict[str, str]:
        result = dict(self.values)
        if self.status != "pass":
            result["curation_status"] = self.status
        if not self.include:
            result["include_by_default"] = "0"
        if self.flags:
            result["curation_flags"] = ";".join(sorted(self.flags))
        return result


@dataclass(frozen=True, slots=True)
class CuratedFeatureInfo:
    feature: str
    filename: str
    native_rows: int
    curated_rows: int
    size_bytes: int
    sha256: str
    pass_rows: int
    review_rows: int
    exclude_default_rows: int
    canonical_value_rows: int
    ambiguous_unit_rows: int
    scale_transition_rows: int
    flag_counts: dict[str, int]
    acquisition_counts: dict[str, int]
    native_sha256: str
    policy_fingerprint: str


@dataclass(slots=True)
class FeatureCurationResult:
    info: CuratedFeatureInfo
    unit_epochs: list[dict[str, Any]] = field(default_factory=list)


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _time(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        result = datetime.fromisoformat(text)
    except ValueError:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _is_true(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _format_number(value: float) -> str:
    return format(float(value), ".15g")


def _acquisition_method(feature: str, row: Mapping[str, str]) -> str | None:
    if _is_true(row.get("was_user_entered")):
        return "user_entered"
    source_id = str(row.get("source_id", "") or "").lower()
    source_name = str(row.get("source_name", "") or "").lower()
    if feature == "BloodAlcoholContent" and (
        "intellidrink" in source_id or "intellidrink" in source_name
    ):
        return "calculator_estimate"
    if feature == "ActivitySummary":
        return "application_summary"
    if feature in {"RestingHeartRate", "WalkingHeartRate", "Vo2Max", "ActiveEnergyBurned", "BasalEnergyBurned"}:
        return "device_estimate"
    if source_id and "apple" not in source_id:
        return "third_party_import"
    return None


def _add_unit(annotation: RowAnnotation, unit: UnitAnnotation | None) -> None:
    if unit is None:
        return
    if unit.canonical_value is not None:
        annotation.values["canonical_value"] = _format_number(unit.canonical_value)
        if unit.canonical_unit:
            annotation.values["canonical_unit"] = unit.canonical_unit
    if unit.unit_status not in {"", "not_applicable"}:
        annotation.values["curation_unit_status"] = unit.unit_status
    if unit.unit_evidence:
        annotation.values["unit_evidence"] = unit.unit_evidence
    if unit.unit_epoch_id:
        annotation.values["unit_epoch_id"] = unit.unit_epoch_id
    if unit.scale_transition:
        annotation.flags.add("possible_unit_transition")
    if unit.unit_status in {"ambiguous", "invalid_numeric"}:
        annotation.escalate("review", flag="unit_unresolved")


def _generic_rules(feature: str, row: Mapping[str, str], annotation: RowAnnotation) -> None:
    policy = get_policy(feature)
    measurements = [field.column for field in policy.schema.measurements if field.required]
    numeric_values: list[float] = []
    for column in measurements:
        raw = row.get(column)
        if raw in (None, ""):
            annotation.escalate("exclude_default", exclude=True, flag="missing_required_measurement")
            continue
        field = next(item for item in policy.schema.measurements if item.column == column)
        if field.value_kind == "numeric":
            value = _float(raw)
            if value is None:
                annotation.escalate("exclude_default", exclude=True, flag="nonfinite_measurement")
            else:
                numeric_values.append(value)

    start = _time(row.get("start_date"))
    end = _time(row.get("end_date"))
    if start is not None and end is not None:
        if end < start:
            annotation.escalate("exclude_default", exclude=True, flag="negative_native_interval")
        duration = (end - start).total_seconds()
        if policy.semantics.event_kind is EventKind.POINT and duration != 0:
            annotation.escalate("review", flag="point_has_nonzero_duration")
        if policy.semantics.event_kind in {
            EventKind.INTERVAL, EventKind.STATE_INTERVAL, EventKind.LONG_SUMMARY_INTERVAL,
            EventKind.DURATION_EVENT, EventKind.WAVEFORM,
        } and duration <= 0:
            annotation.escalate("review", flag="interval_has_zero_duration")

    if "nonnegative_measurement" in policy.curation.rule_ids and any(value < 0 for value in numeric_values):
        annotation.escalate("review", flag="negative_measurement")
    if "positive_measurement" in policy.curation.rule_ids and any(value <= 0 for value in numeric_values):
        annotation.escalate("review", flag="nonpositive_measurement")

    quality = str(row.get("quality_flags", "") or "")
    if row.get("conflict_group_id") or "same_interval_conflict" in quality:
        annotation.escalate(
            "review", exclude=True, flag="unresolved_same_interval_conflict"
        )


def _validate_timezone(row: Mapping[str, str], annotation: RowAnnotation) -> None:
    value = str(row.get("time_zone", "") or "").strip()
    if not value:
        return
    try:
        ZoneInfo(value)
    except ZoneInfoNotFoundError:
        annotation.escalate("review", flag="invalid_cgm_time_zone")


def _parse_waveform(value: str) -> list[Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except Exception:
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            return None
    return parsed if isinstance(parsed, list) else None


def _ecg_rules(row: Mapping[str, str], annotation: RowAnnotation) -> None:
    waveform = _parse_waveform(str(row.get("voltage_measurements", "") or ""))
    if not waveform:
        annotation.escalate("exclude_default", exclude=True, flag="invalid_ecg_waveform_shape")
        return
    times: list[float] = []
    amplitudes: list[float] = []
    valid_shape = True
    for pair in waveform:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            valid_shape = False
            break
        time_value, amplitude = _float(pair[0]), _float(pair[1])
        if time_value is None or amplitude is None:
            valid_shape = False
            break
        times.append(time_value)
        amplitudes.append(amplitude)
    if not valid_shape:
        annotation.escalate("exclude_default", exclude=True, flag="invalid_ecg_waveform_shape")
        return
    if any(right <= left for left, right in zip(times, times[1:])):
        annotation.escalate("exclude_default", exclude=True, flag="ecg_relative_time_not_monotonic")
    if not all(math.isfinite(value) for value in amplitudes):
        annotation.escalate("exclude_default", exclude=True, flag="nonfinite_ecg_amplitude")
    start, end = _time(row.get("start_date")), _time(row.get("end_date"))
    frequency = _float(row.get("sampling_frequency"))
    if start and end and frequency and frequency > 0:
        expected = max(0.0, (end - start).total_seconds() * frequency)
        if expected and abs(len(waveform) - expected) > max(2.0, expected * 0.01):
            annotation.escalate("review", flag="ecg_sample_count_mismatch")


def _feature_rules(feature: str, row: Mapping[str, str], annotation: RowAnnotation) -> None:
    if feature == "ActivitySummary":
        annotation.escalate("review", exclude=True, flag="summary_date_assignment_ambiguous")
        for column in (
            "apple_stand_hours", "apple_exercise_time", "active_energy_burned",
            "apple_stand_hours_goal", "apple_exercise_time_goal", "active_energy_burned_goal",
        ):
            value = _float(row.get(column))
            if value is not None and value < 0:
                annotation.escalate("review", flag="negative_activity_summary_value")
    elif feature == "BloodPressure":
        systolic = _float(row.get("blood_pressure_systolic_value"))
        diastolic = _float(row.get("blood_pressure_diastolic_value"))
        if systolic is None or diastolic is None:
            annotation.escalate("exclude_default", exclude=True, flag="incomplete_blood_pressure_pair")
        elif systolic <= diastolic:
            annotation.escalate("review", flag="blood_pressure_pair_order_warning")
    elif feature == "BloodGlucose":
        status = str(row.get("status", "") or "")
        trend = str(row.get("trend_arrow", "") or "")
        if status not in _KNOWN_CGM_STATUS:
            annotation.escalate("review", flag="unknown_cgm_status")
        if trend not in _KNOWN_CGM_TRENDS:
            annotation.escalate("review", flag="unknown_cgm_trend")
        _validate_timezone(row, annotation)
    elif feature == "OxygenSaturation":
        if not row.get("source_id"):
            annotation.escalate("review", flag="pulse_ox_source_context_limited")
    elif feature == "BodyTemperature":
        metadata = str(row.get("metadata", "") or "")
        if "sensor_location" not in metadata and "temperature_sensor" not in metadata:
            annotation.escalate("review", flag="temperature_sensor_location_unknown")
    elif feature == "PeakFlow":
        annotation.escalate("review", flag="peak_flow_session_context_unknown")
    elif feature == "Vo2Max" and not row.get("vo2_max_test_type"):
        annotation.escalate("review", flag="vo2max_test_type_unknown")
    elif feature == "BloodAlcoholContent":
        method = _acquisition_method(feature, row)
        if method == "calculator_estimate":
            annotation.escalate("review", exclude=True, flag="calculator_estimate")
    elif feature == "WaistCircumference":
        annotation.escalate("review", flag="waist_measurement_protocol_unknown")
    elif feature == "Electrocardiogram":
        _ecg_rules(row, annotation)
    elif feature == "Mindful":
        start, end = _time(row.get("start_date")), _time(row.get("end_date"))
        if start is None or end is None or end < start:
            annotation.escalate("exclude_default", exclude=True, flag="invalid_mindful_interval")


def _sleep_overlap_flags(rows: list[Mapping[str, str]]) -> dict[int, set[str]]:
    flags: dict[int, set[str]] = defaultdict(set)
    intervals: list[tuple[datetime, datetime, str, int]] = []
    for index, row in enumerate(rows):
        start, end = _time(row.get("start_date")), _time(row.get("end_date"))
        state = str(row.get("value", "") or "").upper()
        if state not in _KNOWN_SLEEP_STATES:
            flags[index].add("unknown_sleep_state")
        if start is not None and end is not None and end >= start:
            intervals.append((start, end, state, index))
    intervals.sort(key=lambda value: (value[0], value[1], value[3]))

    active: list[tuple[datetime, datetime, str, int]] = []
    for current in intervals:
        start, end, state, index = current
        active = [item for item in active if item[1] > start]
        for other_start, other_end, other_state, other_index in active:
            if min(end, other_end) <= max(start, other_start):
                continue
            if state == other_state:
                flags[index].add("overlapping_same_sleep_state")
                flags[other_index].add("overlapping_same_sleep_state")
            elif state in _DETAILED_SLEEP_STATES and other_state in _DETAILED_SLEEP_STATES:
                flags[index].add("overlapping_detailed_sleep_states")
                flags[other_index].add("overlapping_detailed_sleep_states")
        active.append(current)

    inbed = [(start, end) for start, end, state, _ in intervals if state == "INBED"]
    inbed.sort()
    for start, end, state, index in intervals:
        if state not in _DETAILED_SLEEP_STATES or not inbed:
            continue
        if not any(min(end, bed_end) > max(start, bed_start) for bed_start, bed_end in inbed):
            flags[index].add("sleep_state_outside_inbed")
    return flags


def _initial_annotation(feature: str) -> RowAnnotation:
    policy = get_policy(feature)
    return RowAnnotation(
        status=policy.curation.default_status.value,
        include=policy.curation.default_inclusion is InclusionPolicy.INCLUDE,
    )


def _annotation_for_row(
    feature: str, row: Mapping[str, str], row_index: int, unit_context: ParticipantUnitContext,
    precomputed_flags: Mapping[int, set[str]] | None = None,
) -> RowAnnotation:
    if feature not in known_curation_features():
        annotation = RowAnnotation(status="review", include=False)
        annotation.flags.add("unknown_feature_policy")
        return annotation

    annotation = _initial_annotation(feature)
    _generic_rules(feature, row, annotation)

    policy = get_policy(feature)
    unit = unit_context.annotation(feature, row_index)
    # Fixed scalar conversions apply only when the policy's first unit-bearing measurement is the ordinary
    # ``value`` column. Multivariate summaries, BloodPressure, and ECG must not be treated as missing scalar values.
    unit_measurement = (
        policy.units.measurements[0].measurement
        if policy.units.measurements else None
    )
    if unit is None and unit_measurement == "value" and "value" in row:
        unit = fixed_unit_annotation(feature, _float(row.get("value")))
    _add_unit(annotation, unit)
    # Milestone 1 may already contain a compact canonical value. Native columns are immutable, so disagreements
    # are surfaced for review rather than overwritten or duplicated.
    if unit is not None and unit.canonical_value is not None:
        native_canonical = _float(row.get("canonical_value"))
        if native_canonical is not None and not math.isclose(
            native_canonical, unit.canonical_value, rel_tol=1e-9, abs_tol=1e-12,
        ):
            annotation.escalate(
                "review", flag="native_canonical_value_conflicts_with_curation_policy"
            )
        native_unit = str(row.get("canonical_unit", "") or "").strip()
        if native_unit and unit.canonical_unit and native_unit != unit.canonical_unit:
            annotation.escalate(
                "review", flag="native_canonical_unit_conflicts_with_curation_policy"
            )

    if policy.calibration.execution_mode is PolicyExecutionMode.BLOCK_DERIVATION:
        annotation.escalate("review", exclude=True, flag="derivation_blocked_by_policy")
    elif (
        policy.calibration.execution_mode is PolicyExecutionMode.CONSERVATIVE_ANNOTATION
        and policy.units.measurements
        and policy.units.measurements[0].canonical_unit is not None
        and (unit is None or unit.canonical_value is None)
    ):
        annotation.escalate("review", flag="canonical_unit_withheld")
    elif (
        policy.calibration.execution_mode is PolicyExecutionMode.SOURCE_SPECIFIC_EXECUTION
        and policy.units.measurements
        and policy.units.measurements[0].unresolved_action
        is UnresolvedUnitAction.EXCLUDE_FROM_DEFAULT_CANONICAL_ANALYSIS
        and (unit is None or unit.canonical_value is None)
    ):
        annotation.escalate("review", exclude=True, flag="unit_unresolved")

    method = _acquisition_method(feature, row)
    if method:
        annotation.values["acquisition_method"] = method

    _feature_rules(feature, row, annotation)
    if precomputed_flags:
        for flag in precomputed_flags.get(row_index, ()):
            annotation.escalate("review", flag=flag)
    return annotation


def _native_rows(path: Path) -> tuple[list[str], Iterable[dict[str, str]]]:
    handle = path.open("r", encoding="utf-8-sig", newline="")
    reader = csv.DictReader(handle)
    fields = list(reader.fieldnames or [])

    def generator() -> Iterable[dict[str, str]]:
        try:
            yield from reader
        finally:
            handle.close()

    return fields, generator()


def _write_projected(
    native_path: Path, destination: Path, annotations_path: Path, active_columns: list[str],
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with native_path.open("r", encoding="utf-8-sig", newline="") as source, \
         annotations_path.open("r", encoding="utf-8") as annotations, \
         destination.open("w", encoding="utf-8", newline="") as target:
        reader = csv.DictReader(source)
        native_fields = list(reader.fieldnames or [])
        # Native columns are immutable. Do not duplicate or overwrite a native processing column that happens to
        # share a curation field name.
        append_columns = [column for column in active_columns if column not in native_fields]
        writer = csv.DictWriter(
            target, fieldnames=native_fields + append_columns,
            lineterminator="\n", quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        next_annotation = annotations.readline()
        current = json.loads(next_annotation) if next_annotation else None
        count = 0
        for index, row in enumerate(reader):
            derived: dict[str, str] = {}
            if current is not None and current["index"] == index:
                derived = current["values"]
                line = annotations.readline()
                current = json.loads(line) if line else None
            if "curation_status" in append_columns and "curation_status" not in derived:
                derived["curation_status"] = "pass"
            if "include_by_default" in append_columns and "include_by_default" not in derived:
                derived["include_by_default"] = "1"
            output = dict(row)
            for column in append_columns:
                output[column] = derived.get(column, "")
            writer.writerow(output)
            count += 1
        target.flush()
        os.fsync(target.fileno())
    return count


def curate_feature_file(
    native_path: Path, destination: Path, feature: str, unit_context: ParticipantUnitContext,
) -> FeatureCurationResult:
    """Curate one native feature CSV without changing its rows or native fields."""

    policy = get_policy(feature)
    native_sha = sha256_file(native_path)
    sleep_rows: list[dict[str, str]] | None = None
    sleep_flags: dict[int, set[str]] | None = None
    if feature == "Sleep":
        with native_path.open("r", encoding="utf-8-sig", newline="") as handle:
            sleep_rows = list(csv.DictReader(handle))
        sleep_flags = _sleep_overlap_flags(sleep_rows)

    active_columns: set[str] = set()
    flag_counts: Counter[str] = Counter()
    acquisition_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    canonical_rows = 0
    ambiguous_rows = 0
    transition_rows = 0
    native_rows = 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, annotation_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".annotations.jsonl", dir=destination.parent,
    )
    os.close(fd)
    annotations_path = Path(annotation_name)
    try:
        with annotations_path.open("w", encoding="utf-8") as annotation_file:
            if sleep_rows is not None:
                iterator: Iterable[Mapping[str, str]] = sleep_rows
            else:
                _, iterator = _native_rows(native_path)
            for index, row in enumerate(iterator):
                annotation = _annotation_for_row(
                    feature, row, index, unit_context, sleep_flags,
                )
                sparse = annotation.as_sparse_dict()
                if annotation.values.get("canonical_value") not in (None, ""):
                    canonical_rows += 1
                if annotation.values.get("curation_unit_status") == "ambiguous":
                    ambiguous_rows += 1
                if "possible_unit_transition" in annotation.flags:
                    transition_rows += 1
                status_counts[annotation.status] += 1
                flag_counts.update(annotation.flags)
                method = annotation.values.get("acquisition_method")
                if method:
                    acquisition_counts[method] += 1
                if sparse:
                    active_columns.update(sparse)
                    annotation_file.write(json.dumps(
                        {"index": index, "values": sparse},
                        ensure_ascii=False, separators=(",", ":"),
                    ) + "\n")
                native_rows += 1

        if not active_columns:
            link_or_copy(native_path, destination)
            curated_rows = native_rows
        else:
            ordered = [column for column in DERIVED_COLUMN_ORDER if column in active_columns]
            curated_rows = _write_projected(
                native_path, destination, annotations_path, ordered,
            )
        drop_file_cache(destination)
        info = CuratedFeatureInfo(
            feature=feature,
            filename=destination.name,
            native_rows=native_rows,
            curated_rows=curated_rows,
            size_bytes=destination.stat().st_size,
            sha256=sha256_file(destination),
            pass_rows=status_counts["pass"],
            review_rows=status_counts["review"],
            exclude_default_rows=status_counts["exclude_default"],
            canonical_value_rows=canonical_rows,
            ambiguous_unit_rows=ambiguous_rows,
            scale_transition_rows=transition_rows,
            flag_counts=dict(sorted(flag_counts.items())),
            acquisition_counts=dict(sorted(acquisition_counts.items())),
            native_sha256=native_sha,
            policy_fingerprint=policy.fingerprint(),
        )
        return FeatureCurationResult(info=info)
    finally:
        annotations_path.unlink(missing_ok=True)
