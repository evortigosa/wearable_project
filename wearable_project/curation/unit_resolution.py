"""
Wearable Data Processing and Modeling project
Participant-, source-, and scale-epoch-aware unit resolution. The resolver is deliberately conservative. Native
values are never mutated and a canonical value is emitted only when the reviewed policy or temporally bounded
cross-feature evidence resolves the raw unit. Stable provenance does not imply a stable scale: each source-context
run is split whenever the monthly scale becomes resolved, unresolved, or changes to another reviewed unit.
"""


from __future__ import annotations
import bisect
import csv
import hashlib
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
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

_EVIDENCE_PRIORITY = {
    "bounded_bmi_consistency_exact": 0,
    "bounded_bmi_consistency_5min": 1,
    "bounded_bmi_consistency_1day": 2,
    "bounded_body_composition_consistency": 3,
    "bounded_cross_feature_epoch_propagation": 4,
    "monthly_scale_regime": 5,
    "insufficient_epoch_evidence": 9,
}


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
    transition_reason: str | None = None


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
    transition_from_unit: str | None = None
    transition_reason: str | None = None


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


@dataclass(frozen=True, slots=True)
class _UnitPreference:
    unit: str
    evidence: str
    median_relative_error: float
    match_count: int
    maximum_time_delta_seconds: float

    @property
    def rank(self) -> tuple[int, float, int]:
        return (
            _EVIDENCE_PRIORITY.get(self.evidence, 8),
            self.median_relative_error,
            -self.match_count,
        )


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
    if raw_unit == "in" and canonical_unit == "m":
        return value * 0.0254
    if raw_unit == "cm" and canonical_unit == "m":
        return value / 100.0
    if raw_unit == "mg/dL" and canonical_unit == "mmol/L":
        return value / 18.0182
    if raw_unit == "kJ" and canonical_unit == "kcal":
        return value / 4.184
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
        # Values between 115 and 135 are deliberately unresolved. Values in the overlapping 25--115 range can
        # be overridden only by bounded BMI evidence, never by a participant-global preference.
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


def _timed(observations: list[_Observation]) -> list[tuple[datetime, _Observation]]:
    return sorted(
        ((item.time, item) for item in observations if item.time is not None and item.value is not None),
        key=lambda pair: (pair[0], pair[1].index),
    )


def _nearest_with_delta(
    observations: list[_Observation], target: datetime, *, seconds: float,
) -> tuple[_Observation | None, float | None]:
    timed = _timed(observations)
    if not timed:
        return None, None
    times = [item[0] for item in timed]
    position = bisect.bisect_left(times, target)
    candidates: list[tuple[datetime, _Observation]] = []
    if position < len(timed):
        candidates.append(timed[position])
    if position:
        candidates.append(timed[position - 1])
    when, observation = min(candidates, key=lambda pair: abs((pair[0] - target).total_seconds()))
    delta = abs((when - target).total_seconds())
    if delta > seconds:
        return None, None
    return observation, delta


def _preference_evidence(maximum_delta: float) -> str:
    if maximum_delta <= 1.0:
        return "bounded_bmi_consistency_exact"
    if maximum_delta <= 300.0:
        return "bounded_bmi_consistency_5min"
    return "bounded_bmi_consistency_1day"


def _select_preference(candidates: list[_UnitPreference]) -> _UnitPreference | None:
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda item: item.rank)
    units = {item.unit for item in ordered}
    if len(units) == 1:
        return ordered[0]
    best = ordered[0]
    second = ordered[1]
    if best.rank[0] < second.rank[0]:
        return best
    if best.median_relative_error <= second.median_relative_error / 2.0:
        return best
    return None


def _bmi_preferences_by_context_month(
    heights: list[_Observation], weights: list[_Observation], bmis: list[_Observation],
) -> tuple[dict[tuple[str, str], _UnitPreference], dict[tuple[str, str], _UnitPreference]]:
    """
    Resolve bounded Height and Weight regimes from BMI consistency. Evidence is attached to the actual Height
    and Weight context-months that participated in the match. It is never promoted to a participant-global
    preference, which prevents an exact 100-lb BMI match from being overridden by a broad kilogram magnitude
    threshold in a different source epoch.
    """

    grouped_errors: dict[
        tuple[str, str, str, str],
        dict[tuple[str, str], list[float]],
    ] = defaultdict(lambda: defaultdict(list))
    grouped_deltas: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)

    for bmi in bmis:
        if bmi.time is None or bmi.value is None or bmi.value <= 0:
            continue
        height, height_delta = _nearest_with_delta(heights, bmi.time, seconds=86400)
        weight, weight_delta = _nearest_with_delta(weights, bmi.time, seconds=86400)
        if (
            height is None or weight is None or height.value is None or weight.value is None
            or height_delta is None or weight_delta is None
        ):
            continue
        key = (height.context_id, height.month, weight.context_id, weight.month)
        grouped_deltas[key].append(max(height_delta, weight_delta))
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
                grouped_errors[key][(height_unit, weight_unit)].append(
                    abs(predicted - bmi.value) / bmi.value
                )

    height_candidates: dict[tuple[str, str], list[_UnitPreference]] = defaultdict(list)
    weight_candidates: dict[tuple[str, str], list[_UnitPreference]] = defaultdict(list)
    for key, candidate_errors in grouped_errors.items():
        ranked = sorted(
            [
                (float(statistics.median(errors)), len(errors), candidate)
                for candidate, errors in candidate_errors.items() if errors
            ],
            key=lambda item: item[0],
        )
        if not ranked:
            continue
        best_error, count, (height_unit, weight_unit) = ranked[0]
        second_error = ranked[1][0] if len(ranked) > 1 else float("inf")
        max_delta = max(grouped_deltas[key])
        # Exact/near-exact evidence may resolve one observation; one-day evidence requires at least two matches.
        # All accepted evidence must have a clinically negligible reconstruction error and clear margin.
        enough_matches = max_delta <= 300.0 or count >= 2
        clear_margin = second_error >= best_error * 2.0 or second_error - best_error >= 0.03
        if not (enough_matches and best_error <= 0.05 and clear_margin):
            continue
        evidence = _preference_evidence(max_delta)
        h_pref = _UnitPreference(height_unit, evidence, best_error, count, max_delta)
        w_pref = _UnitPreference(weight_unit, evidence, best_error, count, max_delta)
        height_candidates[(key[0], key[1])].append(h_pref)
        weight_candidates[(key[2], key[3])].append(w_pref)

    return (
        {key: selected for key, values in height_candidates.items() if (selected := _select_preference(values))},
        {key: selected for key, values in weight_candidates.items() if (selected := _select_preference(values))},
    )


def _body_fat_fraction(value: float | None) -> float | None:
    if value is None or value < 0:
        return None
    if value <= 1.5:
        return value
    if value <= 100:
        return value / 100.0
    return None


def _lean_mass_preferences_by_context_month(
    lean: list[_Observation], weights: list[_Observation], weight_annotations: list[UnitAnnotation],
    body_fat: list[_Observation],
) -> dict[tuple[str, str], _UnitPreference]:
    candidates: dict[tuple[str, str], list[_UnitPreference]] = defaultdict(list)
    for item in lean:
        if item.time is None or item.value is None:
            continue
        weight, weight_delta = _nearest_with_delta(weights, item.time, seconds=86400)
        fat, fat_delta = _nearest_with_delta(body_fat, item.time, seconds=86400)
        if weight is None or fat is None or weight_delta is None or fat_delta is None:
            continue
        if weight.index >= len(weight_annotations):
            continue
        weight_unit = weight_annotations[weight.index].raw_unit
        fraction = _body_fat_fraction(fat.value)
        if weight_unit not in {"kg", "lb"} or weight.value is None or fraction is None:
            continue
        expected = weight.value * (1.0 - fraction)
        denominator = max(abs(item.value), 1e-12)
        error = abs(expected - item.value) / denominator
        max_delta = max(weight_delta, fat_delta)
        if error > 0.05 or (max_delta > 300 and max_delta <= 86400 and error > 0.02):
            continue
        candidates[(item.context_id, item.month)].append(_UnitPreference(
            weight_unit,
            "bounded_body_composition_consistency",
            error,
            1,
            max_delta,
        ))
    return {
        key: selected for key, values in candidates.items()
        if (selected := _select_preference(values))
    }


def _month_ordinal(month: str) -> int | None:
    try:
        year, number = month.split("-", 1)
        return int(year) * 12 + int(number) - 1
    except (AttributeError, TypeError, ValueError):
        return None


def _same_scale_months(feature: str, left: float, right: float) -> bool:
    if left <= 0 or right <= 0:
        return False
    ratio = max(left, right) / min(left, right)
    if feature == "Height":
        return ratio <= 1.20
    if feature in {"Weight", "LeanBodyMass"}:
        return ratio <= 1.60
    if feature == "BloodGlucose":
        return ratio <= 3.0
    return ratio <= 1.60


def _propagate_bounded_preferences(
    feature: str, medians: Mapping[tuple[str, str], float], labels: dict[tuple[str, str], str | None],
    evidence: dict[tuple[str, str], str], overrides: Mapping[tuple[str, str], _UnitPreference],
) -> None:
    """
    Propagate bounded cross-feature evidence within stable scale runs. An exact or temporally bounded
    BMI/body-composition match applies to the source context and numerical scale epoch, not only to the
    calendar month containing the matched observation. Propagation stops at long month gaps, abrupt scale
    changes, or conflicting bounded anchors.
    """

    by_context: dict[str, list[str]] = defaultdict(list)
    for context_id, month in medians:
        if month != "unknown":
            by_context[context_id].append(month)
    for context_id, months in by_context.items():
        ordered = sorted(set(months), key=lambda value: _month_ordinal(value) or -1)
        segments: list[list[str]] = []
        current: list[str] = []
        for month in ordered:
            if not current:
                current = [month]
                continue
            previous = current[-1]
            prev_ordinal = _month_ordinal(previous)
            ordinal = _month_ordinal(month)
            gap = None if prev_ordinal is None or ordinal is None else ordinal - prev_ordinal
            stable = _same_scale_months(
                feature, medians[(context_id, previous)], medians[(context_id, month)],
            )
            if gap is None or gap > 4 or not stable:
                segments.append(current)
                current = [month]
            else:
                current.append(month)
        if current:
            segments.append(current)

        for segment in segments:
            anchors = [
                overrides[(context_id, month)]
                for month in segment if (context_id, month) in overrides
            ]
            anchor_units = {anchor.unit for anchor in anchors}
            if len(anchor_units) != 1:
                continue
            unit = next(iter(anchor_units))
            for month in segment:
                key = (context_id, month)
                if key in overrides:
                    continue
                labels[key] = unit
                evidence[key] = "bounded_cross_feature_epoch_propagation"


def _month_labels(
    feature: str, observations: list[_Observation], overrides: Mapping[tuple[str, str], _UnitPreference] | None = None,
) -> tuple[dict[tuple[str, str], str | None], dict[tuple[str, str], str]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for observation in observations:
        if observation.value is not None:
            values[(observation.context_id, observation.month)].append(observation.value)
    medians = {key: float(statistics.median(month_values)) for key, month_values in values.items()}
    labels: dict[tuple[str, str], str | None] = {}
    evidence: dict[tuple[str, str], str] = {}
    overrides = overrides or {}
    for key, median in medians.items():
        preference = overrides.get(key)
        if preference is not None:
            labels[key] = preference.unit
            evidence[key] = preference.evidence
            continue
        labels[key] = _candidate_from_scale(feature, median)
        evidence[key] = (
            "monthly_scale_regime" if labels[key] is not None else "insufficient_epoch_evidence"
        )
    _propagate_bounded_preferences(feature, medians, labels, evidence, overrides)
    return labels, evidence


def _stable_epoch_id(participant: str, feature: str, context: str, number: int, unit: str | None) -> str:
    payload = f"{participant}|{feature}|{context}|{number}|{unit or 'unresolved'}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _best_evidence(values: Iterable[str]) -> str:
    return min(values, key=lambda value: _EVIDENCE_PRIORITY.get(value, 8), default="insufficient_epoch_evidence")


def _resolve_feature(
    participant_id: str, feature: str, observations: list[_Observation], *,
    overrides: Mapping[tuple[str, str], _UnitPreference] | None = None,
) -> tuple[list[UnitAnnotation], list[UnitEpoch]]:
    canonical = _canonical_unit(feature)
    labels, label_evidence = _month_labels(feature, observations, overrides)

    grouped: dict[str, list[_Observation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.context_id].append(observation)

    annotation_slots: list[UnitAnnotation | None] = [None] * len(observations)
    epochs: list[UnitEpoch] = []
    for context_id, context_rows in grouped.items():
        ordered = sorted(
            context_rows, key=lambda obs: (obs.time or datetime.max.replace(tzinfo=timezone.utc), obs.index,),
        )
        epoch_rows: list[_Observation] = []
        epoch_unit: str | None = None
        epoch_evidence: set[str] = set()
        epoch_number = 0
        previous_time: datetime | None = None
        epoch_transition = False
        transition_from_unit: str | None = None
        transition_reason: str | None = None

        def flush() -> None:
            nonlocal epoch_rows, epoch_number
            if not epoch_rows:
                return
            epoch_number += 1
            epoch_id = _stable_epoch_id(participant_id, feature, context_id, epoch_number, epoch_unit)
            status = "resolved_source_epoch" if epoch_unit is not None else "ambiguous"
            evidence = _best_evidence(epoch_evidence)
            if epoch_unit is None:
                evidence = "insufficient_epoch_evidence"
            for obs in epoch_rows:
                canonical_value = None
                row_status = status
                if obs.value is not None and epoch_unit is not None and canonical is not None:
                    try:
                        canonical_value = _convert(obs.value, epoch_unit, canonical)
                    except ValueError:
                        canonical_value = None
                        row_status = "ambiguous"
                annotation_slots[obs.index] = UnitAnnotation(
                    raw_unit=epoch_unit if row_status != "ambiguous" else None,
                    canonical_value=canonical_value,
                    canonical_unit=canonical if canonical_value is not None else None,
                    unit_status=row_status,
                    unit_evidence=evidence if row_status != "ambiguous" else "insufficient_epoch_evidence",
                    unit_epoch_id=epoch_id,
                    scale_transition=epoch_transition,
                    transition_reason=transition_reason,
                )
            parsed_times = [obs.time for obs in epoch_rows if obs.time is not None]
            epochs.append(UnitEpoch(
                participant_id=participant_id,
                feature=feature,
                epoch_id=epoch_id,
                context_id=context_id,
                start_index=min(obs.index for obs in epoch_rows),
                end_index=max(obs.index for obs in epoch_rows),
                first_time=min(parsed_times).isoformat() if parsed_times else None,
                last_time=max(parsed_times).isoformat() if parsed_times else None,
                row_count=len(epoch_rows),
                raw_unit=epoch_unit,
                canonical_unit=canonical if epoch_unit is not None else None,
                status=status,
                evidence=evidence,
                scale_transition=epoch_transition,
                median_value=_median(obs.value for obs in epoch_rows),
                transition_from_unit=transition_from_unit,
                transition_reason=transition_reason,
            ))
            epoch_rows = []

        for item in ordered:
            key = (context_id, item.month)
            current_unit = labels.get(key)
            current_evidence = label_evidence.get(key, "insufficient_epoch_evidence")
            long_gap = bool(previous_time and item.time and (item.time - previous_time).days > 120)
            unit_changed = bool(epoch_rows and current_unit != epoch_unit)
            if epoch_rows and (long_gap or unit_changed):
                old_unit = epoch_unit
                flush()
                epoch_unit = current_unit
                epoch_evidence = {current_evidence}
                epoch_transition = unit_changed
                transition_from_unit = old_unit if unit_changed else None
                if unit_changed:
                    if old_unit is None:
                        transition_reason = "scale_became_resolved"
                    elif current_unit is None:
                        transition_reason = "scale_became_unresolved"
                    else:
                        transition_reason = "resolved_unit_changed"
                else:
                    transition_reason = "long_gap"
            elif not epoch_rows:
                epoch_unit = current_unit
                epoch_evidence = {current_evidence}
            else:
                epoch_evidence.add(current_evidence)
            epoch_rows.append(item)
            previous_time = item.time or previous_time
        flush()

    default = UnitAnnotation(None, None, None, "ambiguous", "insufficient_epoch_evidence", None,)
    return [item if item is not None else default for item in annotation_slots], epochs


def build_participant_unit_context(participant_dir: Path) -> ParticipantUnitContext:
    """Build unit decisions for unit-sensitive features of one participant."""

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

    height_overrides: dict[tuple[str, str], _UnitPreference] = {}
    weight_overrides: dict[tuple[str, str], _UnitPreference] = {}
    if all(name in loaded for name in ("Height", "Weight", "BMI")):
        height_overrides, weight_overrides = _bmi_preferences_by_context_month(
            loaded["Height"], loaded["Weight"], loaded["BMI"],
        )
        if height_overrides or weight_overrides:
            context.diagnostics.append(
                "Bounded BMI evidence resolved "
                f"{len(height_overrides)} height context-months and "
                f"{len(weight_overrides)} weight context-months"
            )

    # Resolve Height and Weight before LeanBodyMass because LeanBodyMass can inherit a bounded,
    # body-composition-consistent Weight scale.
    for feature, overrides in (
        ("Height", height_overrides), ("Weight", weight_overrides), ("BloodGlucose", {}),
    ):
        observations = loaded.get(feature)
        if observations is None:
            continue
        context.annotations[feature], epochs = _resolve_feature(
            participant_id, feature, observations, overrides=overrides,
        )
        context.epochs.extend(epochs)

    lean = loaded.get("LeanBodyMass")
    if lean is not None:
        lean_overrides: dict[tuple[str, str], _UnitPreference] = {}
        weights = loaded.get("Weight")
        body_fat = loaded.get("BodyFatPercentage")
        weight_annotations = context.annotations.get("Weight")
        if weights is not None and body_fat is not None and weight_annotations is not None:
            lean_overrides = _lean_mass_preferences_by_context_month(
                lean, weights, weight_annotations, body_fat,
            )
        context.annotations["LeanBodyMass"], epochs = _resolve_feature(
            participant_id, "LeanBodyMass", lean, overrides=lean_overrides,
        )
        context.epochs.extend(epochs)

    energy = loaded.get("EnergyConsumed")
    if energy is not None:
        # The reviewed decision intentionally withholds a kcal/kJ choice. Emit explicit ambiguous epochs
        # so the report can account for all rows.
        context.annotations["EnergyConsumed"], epochs = _resolve_feature(participant_id, "EnergyConsumed", energy)
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
        # Identity canonical values are already represented by the native value and source contract; do not
        # repeat them in large feature files.
        return None
    if conversion == "fraction_to_percent":
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
