"""
Wearable Data Processing and Modeling project
Read-only policy-calibration audit for curation. The audit reads native processing participant/feature CSVs and
emits compact cohort evidence tables. It never modifies native files and never creates a curated participant
data product. Its purpose is to support human review of provisional unit, source, and date-assignment policies
before execute those policies.
"""

from __future__ import annotations
from collections import defaultdict
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import statistics
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence
import pandas as pd
from tqdm import tqdm
from wearable_project.curation.guidance import guidance_fingerprint
from wearable_project.curation.models import PolicyMaturity
from wearable_project.curation.registry import (
    CURATION_POLICIES, CURATION_REGISTRY_VERSION, get_policy, registry_fingerprint,
)


AUDIT_VERSION = "0.2.0a1.1-audit-1"
DEFAULT_RESERVOIR_SIZE = 20_000
_CONTEXT_COLUMNS = (
    "collecting_method_version", "source_id", "source_name", "device", "metadata_device_name",
    "time_zone", "was_user_entered", "acquisition_method", "metadata_algorithm_version",
    "metadata_app_version", "temperature_sensor_location",
)


class AuditError(RuntimeError):
    """Raised when an audit cannot be completed safely."""


@dataclass(slots=True)
class OnlineDistribution:
    """Numerically stable online distribution with bounded quantile sample."""

    sample_size: int = DEFAULT_RESERVOIR_SIZE
    seed: int = 0
    count: int = 0
    missing: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float = 0.0
    m2: float = 0.0
    sample: list[float] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def add(self, raw: Any) -> None:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            self.missing += 1
            return
        if not math.isfinite(value):
            self.missing += 1
            return
        self.count += 1
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        if len(self.sample) < self.sample_size:
            self.sample.append(value)
        else:
            index = self._rng.randrange(self.count)
            if index < self.sample_size:
                self.sample[index] = value

    def summary(self) -> dict[str, Any]:
        ordered = sorted(self.sample)
        return {
            "value_count": self.count,
            "missing_value_count": self.missing,
            "value_min": self.minimum,
            "value_q05": _quantile(ordered, 0.05),
            "value_median": _quantile(ordered, 0.50),
            "value_q95": _quantile(ordered, 0.95),
            "value_max": self.maximum,
            "value_mean": self.mean if self.count else None,
            "value_std": math.sqrt(self.m2 / (self.count - 1)) if self.count > 1 else None,
            "quantile_sample_size": len(ordered),
        }


@dataclass(slots=True)
class ContextAccumulator:
    participant_id: str
    feature: str
    context: tuple[str, ...]
    seed: int
    rows: int = 0
    first_start: str | None = None
    last_start: str | None = None
    context_segments: int = 0
    values: OnlineDistribution = field(init=False)
    durations: OnlineDistribution = field(init=False)
    implied_rates: OnlineDistribution = field(init=False)

    def __post_init__(self) -> None:
        self.values = OnlineDistribution(seed=self.seed)
        self.durations = OnlineDistribution(seed=self.seed ^ 0xA5A5A5)
        self.implied_rates = OnlineDistribution(seed=self.seed ^ 0x5A5A5A)

    def add(self, row: Mapping[str, str], *, begins_segment: bool) -> None:
        self.rows += 1
        if begins_segment:
            self.context_segments += 1
        start = row.get("start_date") or row.get("datetime") or ""
        if start:
            if self.first_start is None or start < self.first_start:
                self.first_start = start
            if self.last_start is None or start > self.last_start:
                self.last_start = start
        raw_value = row.get("value")
        duration = _duration_seconds(row.get("start_date"), row.get("end_date"))
        self.values.add(raw_value)
        self.durations.add(duration)
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            numeric_value = math.nan
        if duration is not None and duration > 0 and math.isfinite(numeric_value):
            self.implied_rates.add(numeric_value / duration)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "participant_id": self.participant_id,
            "feature": self.feature,
            "context_id": _context_id(self.context),
            "rows": self.rows,
            "context_segments": self.context_segments,
            "first_start": self.first_start,
            "last_start": self.last_start,
        }
        row.update(dict(zip(_CONTEXT_COLUMNS, self.context)))
        row.update(self.values.summary())
        duration = self.durations.summary()
        row.update({f"duration_{key.removeprefix('value_')}": value for key, value in duration.items()})
        rates = self.implied_rates.summary()
        row.update({f"implied_rate_per_second_{key.removeprefix('value_')}": value for key, value in rates.items()})
        return row


@dataclass(frozen=True, slots=True)
class ParticipantAuditResult:
    participant_id: str
    feature_files_read: int
    rows_read: int
    source_context_rows: tuple[dict[str, Any], ...]
    unit_candidate_rows: tuple[dict[str, Any], ...]
    temporal_scale_rows: tuple[dict[str, Any], ...]
    activity_summary_rows: tuple[dict[str, Any], ...]
    cross_feature_rows: tuple[dict[str, Any], ...]
    diagnostics: tuple[dict[str, Any], ...]
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class AuditSummary:
    audit_version: str
    status: str
    input_native: str
    output: str
    started_at: str
    finished_at: str
    wall_clock_seconds: float
    workers: int
    policy_scope: str
    features_requested: tuple[str, ...]
    participants_discovered: int
    participants_completed: int
    participants_failed: int
    feature_files_read: int
    rows_read: int
    source_context_groups: int
    unit_candidate_rows: int
    temporal_scale_rows: int
    activity_summary_rows: int
    cross_feature_rows: int
    diagnostic_count: int
    registry_version: str
    registry_fingerprint: str
    guidance_fingerprint: str
    output_files: tuple[str, ...]
    failures: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _duration_seconds(start: str | None, end: str | None) -> float | None:
    first = _parse_iso(start)
    second = _parse_iso(end)
    if first is None or second is None:
        return None
    return (second - first).total_seconds()


def _parse_metadata(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(raw)
        except Exception:
            continue
        if isinstance(value, dict):
            return {str(key): item for key, item in value.items()}
    return {}


def _metadata_value(metadata: Mapping[str, Any], *needles: str) -> str:
    normalized_needles = tuple(needle.lower().replace(" ", "_") for needle in needles)
    for key, value in metadata.items():
        normalized = str(key).lower().replace(" ", "_")
        if all(needle in normalized for needle in normalized_needles):
            return "" if value is None else str(value)
    return ""


def _context_tuple(row: Mapping[str, str]) -> tuple[str, ...]:
    metadata = _parse_metadata(row.get("metadata"))
    derived = {
        "metadata_algorithm_version": _metadata_value(metadata, "algorithm", "version"),
        "metadata_app_version": _metadata_value(metadata, "app", "version") or _metadata_value(metadata, "application", "version"),
        "temperature_sensor_location": _metadata_value(metadata, "body", "temperature", "sensor", "location") or _metadata_value(metadata, "temperature", "sensor", "location"),
    }
    return tuple((derived.get(column, row.get(column)) or "").strip() for column in _CONTEXT_COLUMNS)


def _context_id(context: tuple[str, ...]) -> str:
    encoded = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _seed(participant: str, feature: str, context: tuple[str, ...]) -> int:
    payload = f"{participant}\0{feature}\0{_context_id(context)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _candidate_conversion(raw_unit: str, canonical_unit: str | None, value: float) -> float | None:
    unit = raw_unit.strip().lower()
    canonical = (canonical_unit or "").strip().lower()
    if unit in {"kg", "kilogram", "kilograms"} and canonical == "kg":
        return value
    if unit in {"lb", "lbs", "pound", "pounds"} and canonical == "kg":
        return value * 0.45359237
    if unit in {"m", "meter", "metre"} and canonical == "m":
        return value
    if unit in {"cm", "centimeter", "centimetre"} and canonical == "m":
        return value / 100.0
    if unit in {"in", "inch", "inches"} and canonical == "m":
        return value * 0.0254
    if unit == "mg/dl" and canonical == "mmol/l":
        return value / 18.0182
    if unit == "mmol/l" and canonical == "mmol/l":
        return value
    if unit in {"cel", "c", "degc", "°c"} and canonical in {"cel", "c", "degc", "°c"}:
        return value
    if unit in {"degf", "f", "°f"} and canonical in {"cel", "c", "degc", "°c"}:
        return (value - 32.0) * 5.0 / 9.0
    if unit == "m" and canonical == "m":
        return value
    if unit == "km" and canonical == "m":
        return value * 1000.0
    if unit in {"mi", "mile", "miles"} and canonical == "m":
        return value * 1609.344
    if unit == "kcal" and canonical == "kcal":
        return value
    if unit == "kj" and canonical == "kcal":
        return value / 4.184
    if unit == "s" and canonical == "ms":
        return value * 1000.0
    if unit == "ms" and canonical == "ms":
        return value
    if unit in {"l/min", "l/minute"} and canonical.lower() == "l/min":
        return value
    if unit in {"fraction", "1"} and canonical == "%":
        return value * 100.0
    if unit == "%" and canonical == "%":
        return value
    if unit == canonical and canonical:
        return value
    return None


def _candidate_rows(context_row: Mapping[str, Any], feature: str) -> list[dict[str, Any]]:
    policy = get_policy(feature, allow_fallback=False)
    output: list[dict[str, Any]] = []
    sample_values = context_row.get("_sample_values") or []
    for unit_policy in policy.units.measurements:
        candidates = unit_policy.raw_unit_candidates or ((unit_policy.raw_unit,) if unit_policy.raw_unit else ())
        if not candidates:
            continue
        for candidate in candidates:
            converted = [
                result for value in sample_values
                if (result := _candidate_conversion(candidate, unit_policy.canonical_unit, value)) is not None
            ]
            converted.sort()
            output.append({
                "participant_id": context_row["participant_id"],
                "feature": feature,
                "context_id": context_row["context_id"],
                "measurement": unit_policy.measurement,
                "candidate_raw_unit": candidate,
                "canonical_unit": unit_policy.canonical_unit,
                "conversion_rule": unit_policy.conversion_rule,
                "candidate_value_count": len(converted),
                "canonical_min": converted[0] if converted else None,
                "canonical_q05": _quantile(converted, 0.05),
                "canonical_median": _quantile(converted, 0.50),
                "canonical_q95": _quantile(converted, 0.95),
                "canonical_max": converted[-1] if converted else None,
                "evidence_grade": policy.calibration.evidence_grade.value,
                "execution_mode": policy.calibration.execution_mode.value,
                "candidate_is_selected": False,
                "note": "Read-only candidate distribution; this audit does not select a unit.",
            })
    return output


def _audit_feature_file(
    participant: str, path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int, list[dict[str, Any]]]:
    feature = path.stem
    groups: dict[tuple[str, ...], ContextAccumulator] = {}
    monthly_groups: dict[tuple[tuple[str, ...], str], OnlineDistribution] = {}
    previous_context: tuple[str, ...] | None = None
    rows_read = 0
    diagnostics: list[dict[str, Any]] = []
    has_value_column = False
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise AuditError("CSV has no header")
            has_value_column = "value" in reader.fieldnames
            for line_number, row in enumerate(reader, start=2):
                rows_read += 1
                context = _context_tuple(row)
                accumulator = groups.get(context)
                if accumulator is None:
                    accumulator = ContextAccumulator(participant, feature, context, _seed(participant, feature, context))
                    groups[context] = accumulator
                accumulator.add(row, begins_segment=(context != previous_context))
                start_text = row.get("start_date") or row.get("datetime") or ""
                month = start_text[:7] if len(start_text) >= 7 else "unknown"
                month_key = (context, month)
                month_distribution = monthly_groups.get(month_key)
                if month_distribution is None:
                    month_distribution = OnlineDistribution(seed=_seed(participant, feature + month, context))
                    monthly_groups[month_key] = month_distribution
                month_distribution.add(row.get("value"))
                previous_context = context
    except Exception as exc:  # preserve participant-level audit progress
        diagnostics.append({
            "participant_id": participant,
            "feature": feature,
            "source_file": str(path),
            "stage": "source-context-audit",
            "message": f"{type(exc).__name__}: {exc}",
        })

    source_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    for accumulator in groups.values():
        result = accumulator.to_row()
        result["_sample_values"] = sorted(accumulator.values.sample)
        if has_value_column:
            candidate_rows.extend(_candidate_rows(result, feature))
        result.pop("_sample_values", None)
        source_rows.append(result)
    for (context, month), distribution in monthly_groups.items():
        row = {
            "participant_id": participant,
            "feature": feature,
            "context_id": _context_id(context),
            "year_month": month,
        }
        row.update(dict(zip(_CONTEXT_COLUMNS, context)))
        row.update(distribution.summary())
        temporal_rows.append(row)
    return source_rows, candidate_rows, temporal_rows, rows_read, diagnostics


def _numeric_vector(row: Mapping[str, str], columns: Sequence[str]) -> tuple[float, ...] | None:
    output: list[float] = []
    for column in columns:
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        output.append(value)
    return tuple(output)


def _activity_summary_alignment(participant: str, path: Path) -> dict[str, Any]:
    measures = (
        "apple_stand_hours", "apple_exercise_time", "active_energy_burned",
        "apple_stand_hours_goal", "apple_exercise_time_goal", "active_energy_burned_goal",
    )
    per_date: dict[str, list[tuple[int, tuple[float, ...]]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            vector = _numeric_vector(row, measures)
            if vector is None:
                continue
            date = (row.get("datetime") or "")[:10]
            try:
                payload_index = int(float(row.get("payload_index") or 0))
            except ValueError:
                payload_index = 0
            per_date[date].append((payload_index, vector))
    dates = sorted(per_date)
    transitions = 0
    exact_matches = 0
    first_to_next_first = 0
    second_to_next_first = 0
    for current, following in zip(dates, dates[1:]):
        try:
            if (datetime.fromisoformat(following) - datetime.fromisoformat(current)).days != 1:
                continue
        except ValueError:
            continue
        current_rows = sorted(per_date[current])
        next_rows = sorted(per_date[following])
        if not current_rows or not next_rows:
            continue
        transitions += 1
        if current_rows[0][1] == next_rows[0][1]:
            first_to_next_first += 1
        if len(current_rows) > 1 and current_rows[1][1] == next_rows[0][1]:
            second_to_next_first += 1
            exact_matches += 1
    payload_counts = [len(per_date[date]) for date in dates]
    return {
        "participant_id": participant,
        "feature": "ActivitySummary",
        "outer_dates": len(dates),
        "payload_rows": sum(payload_counts),
        "dates_with_one_item": sum(count == 1 for count in payload_counts),
        "dates_with_two_items": sum(count == 2 for count in payload_counts),
        "dates_with_other_item_count": sum(count not in {1, 2} for count in payload_counts),
        "adjacent_day_transitions": transitions,
        "second_item_equals_next_first": second_to_next_first,
        "second_item_equals_next_first_fraction": second_to_next_first / transitions if transitions else None,
        "first_item_equals_next_first": first_to_next_first,
        "date_assignment_resolved": False,
        "note": "Sliding-pair evidence only; no calendar date is assigned by the audit.",
    }


def _load_measurements(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["start_date", "value", "source_id"])
    usecols = None
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    if "start_date" not in frame or "value" not in frame:
        return pd.DataFrame(columns=["start_date", "value", "source_id"])
    frame["start_date"] = pd.to_datetime(frame["start_date"], format="mixed", utc=True, errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["start_date", "value"]).sort_values("start_date")
    if "source_id" not in frame:
        frame["source_id"] = ""
    return frame[["start_date", "value", "source_id"]]


def _nearest_merge(left: pd.DataFrame, right: pd.DataFrame, name: str) -> pd.DataFrame:
    if left.empty or right.empty:
        return pd.DataFrame()
    renamed = right.rename(columns={"value": name, "source_id": f"{name}_source", "start_date": f"{name}_time"})
    working = renamed.copy()
    working["merge_time"] = working[f"{name}_time"]
    merged = pd.merge_asof(
        left.sort_values("start_date"), working.sort_values("merge_time"),
        left_on="start_date", right_on="merge_time", direction="nearest",
    )
    merged[f"{name}_delta_seconds"] = (
        merged["start_date"] - merged[f"{name}_time"]
    ).abs().dt.total_seconds()
    return merged.drop(columns=["merge_time"])


def _bmi_consistency(participant: str, directory: Path) -> list[dict[str, Any]]:
    bmi = _load_measurements(directory / "BMI.csv")
    height = _load_measurements(directory / "Height.csv")
    weight = _load_measurements(directory / "Weight.csv")
    if bmi.empty or height.empty or weight.empty:
        return []
    base = bmi.rename(columns={"value": "observed_bmi", "source_id": "bmi_source"})
    merged = _nearest_merge(base, height, "height")
    if merged.empty:
        return []
    merged = _nearest_merge(merged, weight, "weight")
    if merged.empty:
        return []
    combinations = (("m", "kg"), ("m", "lb"), ("cm", "kg"), ("cm", "lb"), ("in", "kg"), ("in", "lb"))
    rows: list[dict[str, Any]] = []
    for height_unit, weight_unit in combinations:
        h = merged["height"].map(lambda value: _candidate_conversion(height_unit, "m", float(value)))
        w = merged["weight"].map(lambda value: _candidate_conversion(weight_unit, "kg", float(value)))
        predicted = w / (h * h)
        error = (predicted - merged["observed_bmi"]).abs()
        relative = error / merged["observed_bmi"].abs().replace(0, pd.NA)
        valid = pd.DataFrame({
            "error": error,
            "relative": relative,
            "height_delta": merged["height_delta_seconds"],
            "weight_delta": merged["weight_delta_seconds"],
        }).dropna()
        if valid.empty:
            continue
        rows.append({
            "participant_id": participant,
            "check": "bmi_height_weight_consistency",
            "candidate_height_unit": height_unit,
            "candidate_weight_unit": weight_unit,
            "match_count": int(len(valid)),
            "exact_timestamp_matches": int(((valid["height_delta"] == 0) & (valid["weight_delta"] == 0)).sum()),
            "matches_within_1_day": int(((valid["height_delta"] <= 86400) & (valid["weight_delta"] <= 86400)).sum()),
            "median_absolute_error": float(valid["error"].median()),
            "median_relative_error": float(valid["relative"].median()),
            "median_height_delta_seconds": float(valid["height_delta"].median()),
            "median_weight_delta_seconds": float(valid["weight_delta"].median()),
            "candidate_selected": False,
            "note": "Nearest native observations; audit evidence only, not an automatic unit decision.",
        })
    return rows


def _body_composition_consistency(participant: str, directory: Path) -> list[dict[str, Any]]:
    weight = _load_measurements(directory / "Weight.csv")
    fat = _load_measurements(directory / "BodyFatPercentage.csv")
    lean = _load_measurements(directory / "LeanBodyMass.csv")
    if weight.empty or fat.empty or lean.empty:
        return []
    merged = weight.rename(columns={"value": "weight", "source_id": "weight_source"})
    merged = _nearest_merge(merged, fat, "body_fat")
    if merged.empty:
        return []
    merged = _nearest_merge(merged, lean, "lean")
    if merged.empty:
        return []
    rows: list[dict[str, Any]] = []
    for weight_unit in ("kg", "lb"):
        for lean_unit in ("kg", "lb"):
            for fat_unit in ("fraction", "%"):
                w = merged["weight"].map(lambda value: _candidate_conversion(weight_unit, "kg", float(value)))
                l = merged["lean"].map(lambda value: _candidate_conversion(lean_unit, "kg", float(value)))
                fat_fraction = merged["body_fat"] if fat_unit == "fraction" else merged["body_fat"] / 100.0
                expected = w * (1.0 - fat_fraction)
                error = (expected - l).abs()
                relative = error / l.abs().replace(0, pd.NA)
                valid = pd.DataFrame({
                    "error": error,
                    "relative": relative,
                    "fat_delta": merged["body_fat_delta_seconds"],
                    "lean_delta": merged["lean_delta_seconds"],
                }).dropna()
                if valid.empty:
                    continue
                rows.append({
                    "participant_id": participant,
                    "check": "lean_mass_weight_body_fat_consistency",
                    "candidate_weight_unit": weight_unit,
                    "candidate_lean_unit": lean_unit,
                    "candidate_body_fat_unit": fat_unit,
                    "match_count": int(len(valid)),
                    "exact_timestamp_matches": int(((valid["fat_delta"] == 0) & (valid["lean_delta"] == 0)).sum()),
                    "matches_within_5_minutes": int(((valid["fat_delta"] <= 300) & (valid["lean_delta"] <= 300)).sum()),
                    "median_absolute_error_kg": float(valid["error"].median()),
                    "median_relative_error": float(valid["relative"].median()),
                    "candidate_selected": False,
                    "note": "Nearest native observations; audit evidence only, not an automatic unit decision.",
                })
    return rows


def _audit_participant(arguments: tuple[str, str, tuple[str, ...]]) -> ParticipantAuditResult:
    participant, directory_text, features = arguments
    directory = Path(directory_text)
    started = time.perf_counter()
    source_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    activity_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    files_read = 0
    rows_read = 0
    for feature in features:
        path = directory / f"{feature}.csv"
        if not path.exists():
            continue
        files_read += 1
        rows, candidates, temporal, count, problems = _audit_feature_file(participant, path)
        source_rows.extend(rows)
        candidate_rows.extend(candidates)
        temporal_rows.extend(temporal)
        rows_read += count
        diagnostics.extend(problems)
        if feature == "ActivitySummary":
            try:
                activity_rows.append(_activity_summary_alignment(participant, path))
            except Exception as exc:
                diagnostics.append({
                    "participant_id": participant, "feature": feature,
                    "source_file": str(path), "stage": "activity-summary-audit",
                    "message": f"{type(exc).__name__}: {exc}",
                })
    try:
        cross_rows.extend(_bmi_consistency(participant, directory))
        cross_rows.extend(_body_composition_consistency(participant, directory))
    except Exception as exc:
        diagnostics.append({
            "participant_id": participant, "feature": "anthropometrics",
            "source_file": str(directory), "stage": "cross-feature-audit",
            "message": f"{type(exc).__name__}: {exc}",
        })
    return ParticipantAuditResult(
        participant_id=participant,
        feature_files_read=files_read,
        rows_read=rows_read,
        source_context_rows=tuple(source_rows),
        unit_candidate_rows=tuple(candidate_rows),
        temporal_scale_rows=tuple(temporal_rows),
        activity_summary_rows=tuple(activity_rows),
        cross_feature_rows=tuple(cross_rows),
        diagnostics=tuple(diagnostics),
        runtime_seconds=time.perf_counter() - started,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str] = (),) -> None:
    ordered_fields: list[str] = list(fieldnames)
    seen: set[str] = set(ordered_fields)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                ordered_fields.append(key)
    if not ordered_fields:
        raise AuditError(f"No CSV schema declared for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _selected_features(scope: str, explicit: Iterable[str] | None) -> tuple[str, ...]:
    if explicit:
        unknown = sorted(set(explicit) - set(CURATION_POLICIES))
        if unknown:
            raise AuditError(f"Unknown feature(s): {', '.join(unknown)}")
        return tuple(sorted(set(explicit)))
    if scope == "all":
        return tuple(sorted(CURATION_POLICIES))
    return tuple(sorted(
        name for name, policy in CURATION_POLICIES.items()
        if policy.identity.maturity is PolicyMaturity.PROVISIONAL
    ))


_AUDIT_OUTPUT_MAP: dict[str, tuple[str, str]] = {
    "activity_summary_sliding_pair_alignment": ("activity_summary_alignment.csv", "activity"),
    "exporter_date_component_review": ("manual_exporter_review", "manual"),
    "bac_source_regime_inventory": ("source_context_epochs.csv", "source"),
    "bac_fraction_encoding_confirmation": ("unit_candidate_distributions.csv", "unit"),
    "glucose_source_epoch_unit_audit": ("unit_candidate_distributions.csv", "unit"),
    "glucose_scale_transition_audit": ("temporal_scale_windows.csv", "temporal"),
    "temperature_source_epoch_unit_audit": ("unit_candidate_distributions.csv", "unit"),
    "temperature_sensor_location_coverage": ("source_context_epochs.csv", "source"),
    "cycling_distance_unit_audit": ("unit_candidate_distributions.csv", "unit"),
    "cycling_implied_speed_audit": ("source_context_epochs.csv", "implied_rate"),
    "dietary_energy_unit_audit": ("unit_candidate_distributions.csv", "unit"),
    "nutrition_source_regime_audit": ("source_context_epochs.csv", "source"),
    "hrv_source_version_unit_audit": ("unit_candidate_distributions.csv", "unit"),
    "hrv_algorithm_epoch_audit": ("source_context_epochs.csv", "source"),
    "anthropometric_unit_epoch_audit": ("temporal_scale_windows.csv", "temporal"),
    "bmi_height_weight_consistency": ("cross_feature_consistency.csv", "bmi_cross"),
    "lean_mass_weight_body_fat_consistency": ("cross_feature_consistency.csv", "body_cross"),
    "peak_flow_unit_inventory": ("unit_candidate_distributions.csv", "unit"),
    "peak_flow_source_context_audit": ("source_context_epochs.csv", "source"),
    "waist_unit_epoch_audit": ("temporal_scale_windows.csv", "temporal"),
    "waist_protocol_context_audit": ("manual_protocol_review", "manual"),
}


def _audit_coverage_rows(
    features: Sequence[str], source_rows: Sequence[Mapping[str, Any]], candidate_rows: Sequence[Mapping[str, Any]],
    temporal_rows: Sequence[Mapping[str, Any]], activity_rows: Sequence[Mapping[str, Any]],
    cross_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source_counts: dict[str, int] = defaultdict(int)
    unit_counts: dict[str, int] = defaultdict(int)
    temporal_counts: dict[str, int] = defaultdict(int)
    implied_counts: dict[str, int] = defaultdict(int)
    for row in source_rows:
        feature = str(row.get("feature"))
        source_counts[feature] += 1
        if int(row.get("implied_rate_per_second_count") or 0) > 0:
            implied_counts[feature] += 1
    for row in candidate_rows:
        unit_counts[str(row.get("feature"))] += 1
    for row in temporal_rows:
        temporal_counts[str(row.get("feature"))] += 1
    activity_count = len(activity_rows)
    bmi_cross = sum(row.get("check") == "bmi_height_weight_consistency" for row in cross_rows)
    body_cross = sum(row.get("check") == "lean_mass_weight_body_fat_consistency" for row in cross_rows)

    rows: list[dict[str, Any]] = []
    for feature in features:
        policy = get_policy(feature, allow_fallback=False)
        for audit_id in policy.calibration.required_audits:
            output_table, audit_kind = _AUDIT_OUTPUT_MAP.get(audit_id, ("unmapped", "unmapped"))
            if audit_kind == "source":
                count = source_counts.get(feature, 0)
            elif audit_kind == "unit":
                count = unit_counts.get(feature, 0)
            elif audit_kind == "temporal":
                count = temporal_counts.get(feature, 0)
            elif audit_kind == "implied_rate":
                count = implied_counts.get(feature, 0)
            elif audit_kind == "activity":
                count = activity_count if feature == "ActivitySummary" else 0
            elif audit_kind == "bmi_cross":
                count = bmi_cross if feature in {"Height", "Weight"} else 0
            elif audit_kind == "body_cross":
                count = body_cross if feature in {"Weight", "LeanBodyMass"} else 0
            else:
                count = 0
            if audit_kind == "manual":
                status = "manual_review_required"
                note = "This decision requires exporter/source/protocol review outside the native CSV evidence."
            elif audit_kind == "unmapped":
                status = "unmapped_audit"
                note = "No audit-output mapping has been declared."
            elif count:
                status = "evidence_generated"
                note = "Evidence rows are available; human review is still required."
            else:
                status = "no_evidence_observed"
                note = "The selected native root produced no evidence rows for this audit."
            rows.append({
                "feature": feature,
                "audit_id": audit_id,
                "audit_kind": audit_kind,
                "output_table": output_table,
                "evidence_rows": count,
                "status": status,
                "note": note,
            })
    return rows


def _policy_decision_rows(
    features: Sequence[str], source_counts: Mapping[str, int], candidate_counts: Mapping[str, int],
    coverage_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    coverage_by_feature: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for coverage in coverage_rows:
        coverage_by_feature[str(coverage.get("feature"))].append(coverage)
    for feature in features:
        policy = get_policy(feature, allow_fallback=False)
        feature_coverage = coverage_by_feature.get(feature, [])
        rows.append({
            "feature": feature,
            "maturity": policy.identity.maturity.value,
            "evidence_grade": policy.calibration.evidence_grade.value,
            "execution_mode": policy.calibration.execution_mode.value,
            "source_scope": policy.calibration.source_scope,
            "required_audits": ";".join(policy.calibration.required_audits),
            "safe_fallback": policy.calibration.safe_fallback,
            "source_context_groups_observed": source_counts.get(feature, 0),
            "unit_candidate_rows_generated": candidate_counts.get(feature, 0),
            "required_audits_with_evidence": sum(item.get("status") == "evidence_generated" for item in feature_coverage),
            "required_audits_manual": sum(item.get("status") == "manual_review_required" for item in feature_coverage),
            "required_audits_without_evidence": sum(item.get("status") in {"no_evidence_observed", "unmapped_audit"} for item in feature_coverage),
            "policy_fingerprint": policy.fingerprint(),
            "decision_changed_by_audit": False,
            "review_status": "pending_human_review" if policy.identity.maturity is PolicyMaturity.PROVISIONAL else "reviewed_policy_reference",
        })
    return rows


_CSV_BASE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "source_context_epochs.csv": ("participant_id", "feature", "context_id", "rows", "context_segments", "first_start", "last_start"),
    "unit_candidate_distributions.csv": ("participant_id", "feature", "context_id", "measurement", "candidate_raw_unit", "canonical_unit"),
    "temporal_scale_windows.csv": ("participant_id", "feature", "context_id", "year_month"),
    "activity_summary_alignment.csv": ("participant_id", "feature", "outer_dates", "payload_rows"),
    "cross_feature_consistency.csv": ("participant_id", "check", "candidate_selected", "note"),
    "provisional_policy_decisions.csv": ("feature", "maturity", "evidence_grade", "execution_mode", "review_status"),
    "audit_coverage.csv": ("feature", "audit_id", "audit_kind", "output_table", "evidence_rows", "status", "note"),
    "audit_diagnostics.csv": ("participant_id", "feature", "source_file", "stage", "message"),
}


def run_curation_audit(
    input_native: Path, output: Path, *, workers: int = 4, max_in_flight: int | None = None,
    policy_scope: str = "provisional", features: Iterable[str] | None = None,
    selected_participants: set[str] | None = None, overwrite: bool = False,
) -> AuditSummary:
    """Run a read-only calibration audit over a native processing root."""

    native = input_native.expanduser().resolve()
    target = output.expanduser().resolve()
    if not native.is_dir():
        raise AuditError(f"Native input directory does not exist: {native}")
    if native == target or _is_relative_to(target, native) or _is_relative_to(native, target):
        raise AuditError("Native and audit roots must be separate and non-nested")
    selected_features = _selected_features(policy_scope, features)
    participants = sorted(
        path for path in native.iterdir()
        if path.is_dir() and not path.name.startswith(".")
        and (selected_participants is None or path.name in selected_participants)
    )
    if target.exists() and not overwrite:
        raise AuditError(f"Audit output already exists: {target}; pass overwrite=True to replace it")

    started_at = _iso_now()
    start_clock = time.perf_counter()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    source_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    activity_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    files_read = 0
    rows_read = 0
    completed = 0
    effective_workers = max(1, min(int(workers), os.cpu_count() or 1, max(1, len(participants))))
    in_flight = max_in_flight or effective_workers
    in_flight = max(effective_workers, int(in_flight))

    try:
        if participants:
            tasks = [(path.name, str(path), selected_features) for path in participants]

            def consume(result: ParticipantAuditResult) -> None:
                nonlocal completed, files_read, rows_read
                completed += 1
                files_read += result.feature_files_read
                rows_read += result.rows_read
                source_rows.extend(result.source_context_rows)
                candidate_rows.extend(result.unit_candidate_rows)
                temporal_rows.extend(result.temporal_scale_rows)
                activity_rows.extend(result.activity_summary_rows)
                cross_rows.extend(result.cross_feature_rows)
                diagnostics.extend(result.diagnostics)

            if effective_workers == 1:
                with tqdm(total=len(tasks), desc="Auditing participants", unit="participant") as progress:
                    for task in tasks:
                        participant = task[0]
                        try:
                            consume(_audit_participant(task))
                        except Exception as exc:
                            failures[participant] = f"{type(exc).__name__}: {exc}"
                        progress.update(1)
            else:
                context = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=effective_workers, mp_context=context) as executor:
                    iterator = iter(tasks)
                    pending: dict[Any, str] = {}
                    for _ in range(min(in_flight, len(tasks))):
                        task = next(iterator, None)
                        if task is None:
                            break
                        pending[executor.submit(_audit_participant, task)] = task[0]
                    with tqdm(total=len(tasks), desc="Auditing participants", unit="participant") as progress:
                        while pending:
                            future = next(as_completed(pending))
                            participant = pending.pop(future)
                            try:
                                consume(future.result())
                            except Exception as exc:
                                failures[participant] = f"{type(exc).__name__}: {exc}"
                            progress.update(1)
                            task = next(iterator, None)
                            if task is not None:
                                pending[executor.submit(_audit_participant, task)] = task[0]

        source_rows.sort(key=lambda row: (str(row.get("feature")), str(row.get("participant_id")), str(row.get("context_id"))))
        candidate_rows.sort(key=lambda row: (str(row.get("feature")), str(row.get("participant_id")), str(row.get("context_id")), str(row.get("candidate_raw_unit"))))
        temporal_rows.sort(key=lambda row: (str(row.get("feature")), str(row.get("participant_id")), str(row.get("context_id")), str(row.get("year_month"))))
        activity_rows.sort(key=lambda row: str(row.get("participant_id")))
        cross_rows.sort(key=lambda row: (str(row.get("check")), str(row.get("participant_id")), str(row)))
        diagnostics.sort(key=lambda row: (str(row.get("participant_id")), str(row.get("feature")), str(row.get("stage"))))

        source_counts: dict[str, int] = defaultdict(int)
        candidate_counts: dict[str, int] = defaultdict(int)
        for row in source_rows:
            source_counts[str(row["feature"])] += 1
        for row in candidate_rows:
            candidate_counts[str(row["feature"])] += 1
        coverage_rows = _audit_coverage_rows(
            selected_features, source_rows, candidate_rows, temporal_rows, activity_rows, cross_rows
        )
        decisions = _policy_decision_rows(
            selected_features, source_counts, candidate_counts, coverage_rows
        )

        outputs = (
            "source_context_epochs.csv", "unit_candidate_distributions.csv", "temporal_scale_windows.csv",
            "activity_summary_alignment.csv", "cross_feature_consistency.csv", "provisional_policy_decisions.csv",
            "audit_coverage.csv", "audit_diagnostics.csv", "policy_audit_summary.json",
        )
        _write_csv(staging / outputs[0], source_rows, fieldnames=_CSV_BASE_SCHEMAS[outputs[0]])
        _write_csv(staging / outputs[1], candidate_rows, fieldnames=_CSV_BASE_SCHEMAS[outputs[1]])
        _write_csv(staging / outputs[2], temporal_rows, fieldnames=_CSV_BASE_SCHEMAS[outputs[2]])
        _write_csv(staging / outputs[3], activity_rows, fieldnames=_CSV_BASE_SCHEMAS[outputs[3]])
        _write_csv(staging / outputs[4], cross_rows, fieldnames=_CSV_BASE_SCHEMAS[outputs[4]])
        _write_csv(staging / outputs[5], decisions, fieldnames=_CSV_BASE_SCHEMAS[outputs[5]])
        _write_csv(staging / outputs[6], coverage_rows, fieldnames=_CSV_BASE_SCHEMAS[outputs[6]])
        _write_csv(staging / outputs[7], diagnostics, fieldnames=_CSV_BASE_SCHEMAS[outputs[7]])

        finished_at = _iso_now()
        summary = AuditSummary(
            audit_version=AUDIT_VERSION,
            status="complete" if not failures else "complete_with_errors",
            input_native=str(native), output=str(target), started_at=started_at,
            finished_at=finished_at, wall_clock_seconds=time.perf_counter() - start_clock,
            workers=effective_workers, policy_scope=policy_scope,
            features_requested=selected_features,
            participants_discovered=len(participants), participants_completed=completed,
            participants_failed=len(failures), feature_files_read=files_read,
            rows_read=rows_read, source_context_groups=len(source_rows),
            unit_candidate_rows=len(candidate_rows), temporal_scale_rows=len(temporal_rows),
            activity_summary_rows=len(activity_rows),
            cross_feature_rows=len(cross_rows), diagnostic_count=len(diagnostics),
            registry_version=CURATION_REGISTRY_VERSION,
            registry_fingerprint=registry_fingerprint(),
            guidance_fingerprint=guidance_fingerprint(),
            output_files=outputs, failures=failures,
        )
        (staging / "policy_audit_summary.json").write_text(
            json.dumps(summary.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        backup: Path | None = None
        try:
            if target.exists():
                backup = target.parent / f".{target.name}.backup-{int(time.time() * 1_000_000)}"
                os.replace(target, backup)
            os.replace(staging, target)
        except Exception:
            if backup is not None and backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        else:
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def format_audit_summary(summary: AuditSummary | Mapping[str, Any]) -> str:
    payload = summary.as_dict() if isinstance(summary, AuditSummary) else dict(summary)
    failures = payload.get("failures", {})
    lines = [
        f"Curation policy audit: {payload.get('status')}",
        f"Audit version: {payload.get('audit_version')}",
        f"Participants: {payload.get('participants_completed')}/{payload.get('participants_discovered')} completed; "
        f"{payload.get('participants_failed')} failed",
        f"Feature files read: {payload.get('feature_files_read')}",
        f"Rows read: {payload.get('rows_read')}",
        f"Source-context groups: {payload.get('source_context_groups')}",
        f"Unit-candidate rows: {payload.get('unit_candidate_rows')}",
        f"Temporal scale rows: {payload.get('temporal_scale_rows')}",
        f"ActivitySummary audit rows: {payload.get('activity_summary_rows')}",
        f"Cross-feature audit rows: {payload.get('cross_feature_rows')}",
        f"Diagnostics: {payload.get('diagnostic_count')}",
        f"Wall time: {float(payload.get('wall_clock_seconds') or 0):.2f} s",
        f"Output: {payload.get('output')}",
    ]
    if failures:
        lines.append("Failures:")
        lines.extend(f"  {participant}: {message}" for participant, message in sorted(failures.items()))
    return "\n".join(lines)
