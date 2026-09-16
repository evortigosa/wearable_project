"""
Wearable Data Processing and Modeling project
Participant/source/scale-aware unit resolution for curation. The resolver is intentionally conservative.
It never mutates native values and emits a canonical value only when the reviewed feature policy or bounded
participant/source evidence resolves the raw unit. Stable source context is not assumed to imply stable
numerical scale: context runs are split into unit epochs when monthly scale labels or reviewed conversion-factor
transitions change.
"""

from __future__ import annotations
import bisect
import csv
import hashlib
import math
import sys
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from wearable_project.curation.models import PolicyExecutionMode
from wearable_project.curation.registry import get_policy


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_field_limit()


UNIT_SENSITIVE_FEATURES: tuple[str, ...] = (
    "BloodGlucose",
    "EnergyConsumed",
    "Height",
    "LeanBodyMass",
    "Weight",
)

_CONTEXT_COLUMNS: tuple[str, ...] = (
    "collecting_method_version",
    "source_id",
    "source_name",
    "device",
    "metadata_device_name",
    "time_zone",
    "was_user_entered",
)


@dataclass(frozen=True, slots=True)
class UnitAnnotation:
    """Resolved or unresolved unit information for one native row."""

    raw_unit: str | None
    canonical_value: float | None
    canonical_unit: str | None
    unit_status: str
    unit_evidence: str
    unit_epoch_id: str | None
    scale_transition: bool = False


@dataclass(frozen=True, slots=True)
class UnitEpoch:
    participant_id: str
    feature: str
    epoch_id: str
    context_id: str
    start_index: int
    end_index: int
    first_time: str | None
    last_time: str | None
    row_count: int
    raw_unit: str | None
    canonical_unit: str | None
    status: str
    evidence: str
    scale_transition: bool
    median_value: float | None


@dataclass(slots=True)
class ParticipantUnitContext:
    participant_id: str
    annotations: dict[str, list[UnitAnnotation]] = field(default_factory=dict)
    epochs: list[UnitEpoch] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    def annotation(self, feature: str, row_index: int) -> UnitAnnotation | None:
        values = self.annotations.get(feature)
        if values is None or row_index >= len(values):
            return None
        return values[row_index]


@dataclass(frozen=True, slots=True)
class _Observation:
    index: int
    time: datetime | None
    raw_time: str
    value: float | None
    row: Mapping[str, str]
    context_id: str
    month: str


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


def _context_id(row: Mapping[str, str]) -> str:
    payload = "\x1f".join(str(row.get(column, "") or "").strip() for column in _CONTEXT_COLUMNS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _month(value: datetime | None) -> str:
    return value.strftime("%Y-%m") if value is not None else "unknown"


def _read_feature(path: Path) -> list[_Observation]:
    result: list[_Observation] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            timestamp = _time(row.get("start_date") or row.get("datetime"))
            result.append(_Observation(
                index=index,
                time=timestamp,
                raw_time=str(row.get("start_date") or row.get("datetime") or ""),
                value=_float(row.get("value")),
                row=row,
                context_id=_context_id(row),
                month=_month(timestamp),
            ))
    return result


def _median(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return float(statistics.median(clean)) if clean else None


def _convert(value: float, raw_unit: str, canonical_unit: str) -> float:
    if raw_unit == canonical_unit:
        return value
    if raw_unit == "lb" and canonical_unit == "kg":
        return value * 0.45359237
    if raw_unit == "kg" and canonical_unit == "kg":
        return value
    if raw_unit == "in" and canonical_unit == "m":
        return value * 0.0254
    if raw_unit == "cm" and canonical_unit == "m":
        return value / 100.0
    if raw_unit == "m" and canonical_unit == "m":
        return value
    if raw_unit == "mg/dL" and canonical_unit == "mmol/L":
        return value / 18.0182
    if raw_unit == "mmol/L" and canonical_unit == "mmol/L":
        return value
    if raw_unit == "kJ" and canonical_unit == "kcal":
        return value / 4.184
    if raw_unit == "kcal" and canonical_unit == "kcal":
        return value
    raise ValueError(f"Unsupported unit conversion: {raw_unit} -> {canonical_unit}")


def _candidate_from_scale(feature: str, median: float | None) -> str | None:
    if median is None or median <= 0:
        return None
    if feature == "Height":
        if 0.5 <= median <= 2.5:
            return "m"
        if 35 <= median < 100:
            return "in"
        if 100 <= median <= 250:
            return "cm"
        return None
    if feature == "Weight":
        # Deliberately leave the overlap between plausible heavy kilograms and common pounds unresolved
        # unless bounded BMI evidence settles it.
        if 25 <= median <= 115:
            return "kg"
        if 135 <= median <= 700:
            return "lb"
        return None
    if feature == "LeanBodyMass":
        if 20 <= median <= 90:
            return "kg"
        if 120 <= median <= 500:
            return "lb"
        return None
    if feature == "BloodGlucose":
        if 1 <= median <= 40:
            return "mmol/L"
        if 40 < median <= 700:
            return "mg/dL"
        return None
    return None


def _canonical_unit(feature: str) -> str | None:
    return {
        "Height": "m",
        "Weight": "kg",
        "LeanBodyMass": "kg",
        "BloodGlucose": "mmol/L",
        "EnergyConsumed": "kcal",
    }.get(feature)


def _nearest(observations: list[_Observation], target: datetime, *, seconds: float) -> _Observation | None:
    timed = [(item.time, item) for item in observations if item.time is not None and item.value is not None]
    if not timed:
        return None
    times = [item[0] for item in timed]
    position = bisect.bisect_left(times, target)
    candidates = []
    if position < len(timed):
        candidates.append(timed[position])
    if position:
        candidates.append(timed[position - 1])
    if not candidates:
        return None
    when, observation = min(candidates, key=lambda pair: abs((pair[0] - target).total_seconds()))
    if abs((when - target).total_seconds()) > seconds:
        return None
    return observation


def _bmi_preference(
    heights: list[_Observation], weights: list[_Observation], bmis: list[_Observation]
) -> tuple[str, str, float, int] | None:
    """
    Return a bounded participant-level height/weight unit preference. Evidence is restricted to observations
    within one day. The winner must have a small median relative error and a meaningful margin over the second
    candidate. The result is evidence for unit epochs, not a global cohort default.
    """

    errors: dict[tuple[str, str], list[float]] = defaultdict(list)
    for bmi in bmis:
        if bmi.time is None or bmi.value is None or bmi.value <= 0:
            continue
        height = _nearest(heights, bmi.time, seconds=86400)
        weight = _nearest(weights, bmi.time, seconds=86400)
        if height is None or weight is None or height.value is None or weight.value is None:
            continue
        for height_unit in ("m", "cm", "in"):
            for weight_unit in ("kg", "lb"):
                try:
                    h = _convert(height.value, height_unit, "m")
                    w = _convert(weight.value, weight_unit, "kg")
                except ValueError:
                    continue
                if h <= 0:
                    continue
                predicted = w / (h * h)
                errors[(height_unit, weight_unit)].append(abs(predicted - bmi.value) / bmi.value)
    ranked = sorted(
        ((statistics.median(values), len(values), key) for key, values in errors.items() if values),
        key=lambda item: item[0],
    )
    if not ranked:
        return None
    best_error, count, (height_unit, weight_unit) = ranked[0]
    second_error = ranked[1][0] if len(ranked) > 1 else float("inf")
    if best_error <= 0.10 and (second_error >= best_error * 2.0 or second_error - best_error >= 0.05):
        return height_unit, weight_unit, float(best_error), count
    return None


def _month_labels(
    feature: str, observations: list[_Observation], preferred_unit: str | None,
) -> dict[tuple[str, str], str | None]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for observation in observations:
        if observation.value is not None:
            values[(observation.context_id, observation.month)].append(observation.value)
    labels: dict[tuple[str, str], str | None] = {}
    for key, month_values in values.items():
        median = float(statistics.median(month_values))
        inferred = _candidate_from_scale(feature, median)
        if preferred_unit is not None:
            # A bounded cross-feature preference may resolve otherwise overlapping scales, but do not override
            # a clearly incompatible monthly scale.
            if inferred is None or inferred == preferred_unit:
                inferred = preferred_unit
        labels[key] = inferred
    return labels


def _stable_epoch_id(participant: str, feature: str, context: str, number: int, unit: str | None) -> str:
    payload = f"{participant}|{feature}|{context}|{number}|{unit or 'unresolved'}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _resolve_feature(
    participant_id: str, feature: str, observations: list[_Observation], *, preferred_unit: str | None = None,
) -> tuple[list[UnitAnnotation], list[UnitEpoch]]:
    #policy = get_policy(feature)
    #measurement_policy = policy.units.measurements[0] if policy.units.measurements else None
    canonical = _canonical_unit(feature)
    labels = _month_labels(feature, observations, preferred_unit)

    grouped: dict[str, list[_Observation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.context_id].append(observation)

    annotation_slots: list[UnitAnnotation | None] = [None] * len(observations)
    epochs: list[UnitEpoch] = []
    for context_id, context_rows in grouped.items():
        ordered = sorted(
            context_rows, key=lambda obs: (obs.time or datetime.max.replace(tzinfo=timezone.utc), obs.index)
        )
        epoch_rows: list[_Observation] = []
        epoch_unit: str | None = None
        epoch_number = 0
        previous_time: datetime | None = None

        def flush(*, transition: bool) -> None:
            nonlocal epoch_rows, epoch_unit, epoch_number
            if not epoch_rows:
                return
            epoch_number += 1
            epoch_id = _stable_epoch_id(participant_id, feature, context_id, epoch_number, epoch_unit)
            status = "resolved_source_epoch" if epoch_unit is not None else "ambiguous"
            evidence = (
                "bounded_bmi_consistency" if preferred_unit is not None and epoch_unit == preferred_unit
                else "monthly_scale_regime" if epoch_unit is not None
                else "insufficient_epoch_evidence"
            )
            for erow in epoch_rows:
                canonical_value = None
                if erow.value is not None and epoch_unit is not None and canonical is not None:
                    try:
                        canonical_value = _convert(erow.value, epoch_unit, canonical)
                    except ValueError:
                        canonical_value = None
                        status = "ambiguous"
                annotation_slots[erow.index] = UnitAnnotation(
                    raw_unit=epoch_unit,
                    canonical_value=canonical_value,
                    canonical_unit=canonical if canonical_value is not None else None,
                    unit_status=status,
                    unit_evidence=evidence,
                    unit_epoch_id=epoch_id,
                    scale_transition=transition,
                )
            epochs.append(UnitEpoch(
                participant_id=participant_id,
                feature=feature,
                epoch_id=epoch_id,
                context_id=context_id,
                start_index=min(row.index for row in epoch_rows),
                end_index=max(row.index for row in epoch_rows),
                first_time=min((row.raw_time for row in epoch_rows if row.raw_time), default=None),
                last_time=max((row.raw_time for row in epoch_rows if row.raw_time), default=None),
                row_count=len(epoch_rows),
                raw_unit=epoch_unit,
                canonical_unit=canonical if epoch_unit is not None else None,
                status=status,
                evidence=evidence,
                scale_transition=transition,
                median_value=_median(row.value for row in epoch_rows),
            ))
            epoch_rows = []

        #last_unit: str | None | object = object()
        transition_pending = False
        for item in ordered:
            current_unit = labels.get((context_id, item.month))
            long_gap = bool(previous_time and item.time and (item.time - previous_time).days > 120)
            changed_scale = bool(epoch_rows) and current_unit != epoch_unit and current_unit is not None and epoch_unit is not None
            if epoch_rows and (long_gap or changed_scale):
                flush(transition=changed_scale)
                transition_pending = changed_scale
            if not epoch_rows:
                epoch_unit = current_unit
            elif epoch_unit is None and current_unit is not None:
                # Split unresolved and resolved portions rather than silently back-propagating a later scale decision.
                flush(transition=True)
                epoch_unit = current_unit
                transition_pending = True
            epoch_rows.append(item)
            previous_time = item.time or previous_time
            #last_unit = current_unit
        flush(transition=transition_pending)

    default = UnitAnnotation(None, None, None, "ambiguous", "insufficient_epoch_evidence", None)
    return [item if item is not None else default for item in annotation_slots], epochs


def build_participant_unit_context(participant_dir: Path) -> ParticipantUnitContext:
    """Build unit decisions for the unit-sensitive features of one participant."""

    participant_id = participant_dir.name
    context = ParticipantUnitContext(participant_id)
    loaded: dict[str, list[_Observation]] = {}
    for feature in (*UNIT_SENSITIVE_FEATURES, "BMI", "BodyFatPercentage"):
        path = participant_dir / f"{feature}.csv"
        if path.is_file():
            try:
                loaded[feature] = _read_feature(path)
            except (OSError, csv.Error) as exc:
                context.diagnostics.append(f"{feature}: {type(exc).__name__}: {exc}")

    preferred_height: str | None = None
    preferred_weight: str | None = None
    if all(name in loaded for name in ("Height", "Weight", "BMI")):
        preference = _bmi_preference(loaded["Height"], loaded["Weight"], loaded["BMI"])
        if preference is not None:
            preferred_height, preferred_weight, error, count = preference
            context.diagnostics.append(
                f"BMI evidence: height={preferred_height}, weight={preferred_weight}, "
                f"median_relative_error={error:.6g}, matches={count}"
            )

    for feature in UNIT_SENSITIVE_FEATURES:
        observations = loaded.get(feature)
        if observations is None:
            continue
        preferred = None
        if feature == "Height":
            preferred = preferred_height
        elif feature == "Weight":
            preferred = preferred_weight
        elif feature == "LeanBodyMass":
            preferred = preferred_weight
        elif feature == "EnergyConsumed":
            # Human calibration explicitly forbids automatic kcal/kJ selection.
            context.annotations[feature] = [
                UnitAnnotation(None, None, None, "ambiguous", "energy_unit_unresolved", None)
                for _ in observations
            ]
            continue
        context.annotations[feature], epochs = _resolve_feature(
            participant_id, feature, observations, preferred_unit=preferred,
        )
        context.epochs.extend(epochs)
    return context


def fixed_unit_annotation(feature: str, value: float | None) -> UnitAnnotation | None:
    """Return the reviewed fixed conversion for a non-epoch feature."""

    policy = get_policy(feature)
    if not policy.units.measurements:
        return None
    unit = policy.units.measurements[0]
    if unit.canonical_unit is None or unit.conversion_rule == "not_applicable":
        return None
    if value is None:
        return UnitAnnotation(
            raw_unit=unit.raw_unit,
            canonical_value=None,
            canonical_unit=None,
            unit_status="invalid_numeric",
            unit_evidence="missing_or_nonfinite_value",
            unit_epoch_id=None,
        )
    conversion = unit.conversion_rule
    converted: float | None
    if conversion == "identity":
        # Identity canonical values and their unit status are already encoded by the native feature contract.
        # Avoid repeating constant annotations in every row of very large files.
        return None
    elif conversion == "fraction_to_percent":
        converted = value * 100.0
    elif conversion == "seconds_to_milliseconds":
        converted = value * 1000.0
    elif conversion == "inches_to_metres":
        converted = value * 0.0254
    elif conversion == "fahrenheit_to_celsius":
        converted = (value - 32.0) * 5.0 / 9.0
    else:
        return None
    return UnitAnnotation(
        raw_unit=unit.raw_unit,
        canonical_value=converted,
        canonical_unit=unit.canonical_unit if converted is not None else None,
        unit_status="resolved_reviewed",
        unit_evidence=unit.unit_resolution_strategy,
        unit_epoch_id=None,
    )
