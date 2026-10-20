"""
Wearable Data Processing and Modeling project
Deterministic aggregation primitives for the Google Fitness CWP2 native audit. The classes in this module are
intentionally independent of CSV I/O. Worker processes build compact participant-local aggregates and the
parent merges them in participant order, so scientific reports do not depend on worker completion order.
"""


from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
import heapq
import math
from typing import Any, Iterable


AUDIT_VERSION = "google-fitness-curation-audit-cwp2-0.1.0"
REPORT_SCHEMA_VERSION = "google-fitness-curation-audit-report-0.1.0"
DEFAULT_SAMPLE_SIZE = 4096


def _stable_rank(key: str) -> int:
    return int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "big")


@dataclass(slots=True)
class NumericSketch:
    """Exact moments plus deterministic bounded samples for quantiles."""

    sample_size: int = DEFAULT_SAMPLE_SIZE
    seen: int = 0
    missing: int = 0
    nonnumeric: int = 0
    nonfinite: int = 0
    finite: int = 0
    negative: int = 0
    zero: int = 0
    positive: int = 0
    minimum: float | None = None
    maximum: float | None = None
    total: float = 0.0
    total_sq: float = 0.0
    samples: dict[int, float] = field(default_factory=dict)
    _max_heap: list[int] = field(default_factory=list, repr=False)

    def add(self, value: Any, *, key: str) -> float | None:
        self.seen += 1
        if value is None or value == "":
            self.missing += 1
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            self.nonnumeric += 1
            return None
        if not math.isfinite(number):
            self.nonfinite += 1
            return None
        self.finite += 1
        if number < 0:
            self.negative += 1
        elif number > 0:
            self.positive += 1
        else:
            self.zero += 1
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)
        self.total += number
        self.total_sq += number * number
        rank = _stable_rank(key)
        if rank in self.samples:
            return number
        if len(self.samples) < self.sample_size:
            self.samples[rank] = number
            heapq.heappush(self._max_heap, -rank)
        else:
            largest = -self._max_heap[0]
            if rank < largest:
                heapq.heapreplace(self._max_heap, -rank)
                del self.samples[largest]
                self.samples[rank] = number
        return number

    def merge(self, other: "NumericSketch") -> None:
        self.seen += other.seen
        self.missing += other.missing
        self.nonnumeric += other.nonnumeric
        self.nonfinite += other.nonfinite
        self.finite += other.finite
        self.negative += other.negative
        self.zero += other.zero
        self.positive += other.positive
        if other.minimum is not None:
            self.minimum = other.minimum if self.minimum is None else min(self.minimum, other.minimum)
        if other.maximum is not None:
            self.maximum = other.maximum if self.maximum is None else max(self.maximum, other.maximum)
        self.total += other.total
        self.total_sq += other.total_sq
        combined = {**self.samples, **other.samples}
        retained = sorted(combined.items())[: self.sample_size]
        self.samples = dict(retained)
        self._max_heap = [-rank for rank, _ in retained]
        heapq.heapify(self._max_heap)

    @property
    def mean(self) -> float | None:
        return self.total / self.finite if self.finite else None

    @property
    def std(self) -> float | None:
        if not self.finite:
            return None
        mean = self.total / self.finite
        variance = max(0.0, self.total_sq / self.finite - mean * mean)
        return math.sqrt(variance)

    def quantile(self, probability: float) -> float | None:
        values = sorted(self.samples.values())
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        position = probability * (len(values) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return values[lower]
        fraction = position - lower
        return values[lower] * (1.0 - fraction) + values[upper] * fraction

    def summary(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "missing": self.missing,
            "nonnumeric": self.nonnumeric,
            "nonfinite": self.nonfinite,
            "finite": self.finite,
            "negative": self.negative,
            "zero": self.zero,
            "positive": self.positive,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "p01": self.quantile(0.01),
            "p05": self.quantile(0.05),
            "p25": self.quantile(0.25),
            "p50": self.quantile(0.50),
            "p75": self.quantile(0.75),
            "p95": self.quantile(0.95),
            "p99": self.quantile(0.99),
            "sample_size": len(self.samples),
        }


@dataclass(slots=True)
class PairSketch:
    """Exact pair moments and deterministic difference/ratio summaries."""

    pairs: int = 0
    exact_equal: int = 0
    within_1pct: int = 0
    within_5pct: int = 0
    left_zero: int = 0
    right_zero: int = 0
    sum_left: float = 0.0
    sum_right: float = 0.0
    sum_left_sq: float = 0.0
    sum_right_sq: float = 0.0
    sum_cross: float = 0.0
    difference: NumericSketch = field(default_factory=NumericSketch)
    absolute_difference: NumericSketch = field(default_factory=NumericSketch)
    ratio: NumericSketch = field(default_factory=NumericSketch)

    def add(self, left: float, right: float, *, key: str) -> None:
        if not (math.isfinite(left) and math.isfinite(right)):
            return
        self.pairs += 1
        self.sum_left += left
        self.sum_right += right
        self.sum_left_sq += left * left
        self.sum_right_sq += right * right
        self.sum_cross += left * right
        if left == 0:
            self.left_zero += 1
        if right == 0:
            self.right_zero += 1
        difference = left - right
        absolute = abs(difference)
        if difference == 0:
            self.exact_equal += 1
        scale = max(abs(left), abs(right), 1e-12)
        if absolute <= 0.01 * scale:
            self.within_1pct += 1
        if absolute <= 0.05 * scale:
            self.within_5pct += 1
        self.difference.add(difference, key=f"{key}|diff")
        self.absolute_difference.add(absolute, key=f"{key}|abs")
        if right != 0:
            self.ratio.add(left / right, key=f"{key}|ratio")

    def merge(self, other: "PairSketch") -> None:
        self.pairs += other.pairs
        self.exact_equal += other.exact_equal
        self.within_1pct += other.within_1pct
        self.within_5pct += other.within_5pct
        self.left_zero += other.left_zero
        self.right_zero += other.right_zero
        self.sum_left += other.sum_left
        self.sum_right += other.sum_right
        self.sum_left_sq += other.sum_left_sq
        self.sum_right_sq += other.sum_right_sq
        self.sum_cross += other.sum_cross
        self.difference.merge(other.difference)
        self.absolute_difference.merge(other.absolute_difference)
        self.ratio.merge(other.ratio)

    @property
    def correlation(self) -> float | None:
        if self.pairs < 2:
            return None
        n = float(self.pairs)
        numerator = n * self.sum_cross - self.sum_left * self.sum_right
        left_term = n * self.sum_left_sq - self.sum_left * self.sum_left
        right_term = n * self.sum_right_sq - self.sum_right * self.sum_right
        denominator = math.sqrt(max(0.0, left_term) * max(0.0, right_term))
        return numerator / denominator if denominator else None

    def summary(self) -> dict[str, Any]:
        return {
            "pairs": self.pairs,
            "exact_equal": self.exact_equal,
            "within_1pct": self.within_1pct,
            "within_5pct": self.within_5pct,
            "left_zero": self.left_zero,
            "right_zero": self.right_zero,
            "correlation": self.correlation,
            "difference": self.difference.summary(),
            "absolute_difference": self.absolute_difference.summary(),
            "ratio": self.ratio.summary(),
        }


@dataclass(slots=True)
class ParticipantAuditResult:
    participant_id: str
    files_read: int = 0
    rows_read: int = 0
    bytes_read: int = 0
    feature_files: Counter[str] = field(default_factory=Counter)
    feature_rows: Counter[str] = field(default_factory=Counter)
    feature_versions: Counter[tuple[str, str]] = field(default_factory=Counter)
    unit_stats: dict[tuple[str, str, str], NumericSketch] = field(default_factory=dict)
    activity_quantity: dict[tuple[str, str], NumericSketch] = field(default_factory=dict)
    activity_quantity_distance: dict[tuple[str, str], PairSketch] = field(default_factory=dict)
    activity_quantity_calories: dict[tuple[str, str], PairSketch] = field(default_factory=dict)
    activity_quantity_stepcount: dict[tuple[str, str], PairSketch] = field(default_factory=dict)
    activity_quantity_stepcount_coverage: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    activity_concordance: dict[tuple[str, str], PairSketch] = field(default_factory=dict)
    activity_concordance_coverage: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    activity_cross_midnight: Counter[str] = field(default_factory=Counter)
    step_preference_compositions: Counter[tuple[tuple[str, int], ...]] = field(default_factory=Counter)
    step_role_counts: Counter[str] = field(default_factory=Counter)
    step_group_counts: Counter[str] = field(default_factory=Counter)
    step_cross_membership: Counter[str] = field(default_factory=Counter)
    step_cross_pairs: dict[str, PairSketch] = field(default_factory=dict)
    step_value_raw_pairs: dict[str, PairSketch] = field(default_factory=dict)
    step_temporal_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    step_temporal_duration: dict[str, NumericSketch] = field(default_factory=dict)
    negative_calories: Counter[tuple[str, str]] = field(default_factory=Counter)
    negative_calories_values: dict[tuple[str, str], NumericSketch] = field(default_factory=dict)
    weight_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    weight_duration: dict[str, NumericSketch] = field(default_factory=dict)
    weight_values: dict[str, NumericSketch] = field(default_factory=dict)
    weight_added_by: Counter[tuple[str, str]] = field(default_factory=Counter)
    anthropometric_counts: Counter[str] = field(default_factory=Counter)
    anthropometric_bmi: NumericSketch = field(default_factory=NumericSketch)
    anthropometric_time_delta_days: NumericSketch = field(default_factory=NumericSketch)
    provenance_counts: Counter[tuple[str, str, str, str]] = field(default_factory=Counter)
    source_day_counts: Counter[tuple[str, str, str, str, str]] = field(default_factory=Counter)
    sleep_overlap_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    sleep_overlap_rows_affected: Counter[tuple[str, str]] = field(default_factory=Counter)
    quality_rule_counts: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    sleep_overlap_duration: dict[tuple[str, str], NumericSketch] = field(default_factory=dict)
    schema_issues: list[str] = field(default_factory=list)

    def sketch(self, mapping: dict[Any, NumericSketch], key: Any) -> NumericSketch:
        if key not in mapping:
            mapping[key] = NumericSketch()
        return mapping[key]

    def pair_sketch(self, mapping: dict[Any, PairSketch], key: Any) -> PairSketch:
        if key not in mapping:
            mapping[key] = PairSketch()
        return mapping[key]

    def merge(self, other: "ParticipantAuditResult") -> None:
        self.files_read += other.files_read
        self.rows_read += other.rows_read
        self.bytes_read += other.bytes_read
        for target, source in (
            (self.feature_files, other.feature_files),
            (self.feature_rows, other.feature_rows),
            (self.feature_versions, other.feature_versions),
            (self.activity_concordance_coverage, other.activity_concordance_coverage),
            (self.activity_quantity_stepcount_coverage, other.activity_quantity_stepcount_coverage),
            (self.activity_cross_midnight, other.activity_cross_midnight),
            (self.step_preference_compositions, other.step_preference_compositions),
            (self.step_role_counts, other.step_role_counts),
            (self.step_group_counts, other.step_group_counts),
            (self.step_cross_membership, other.step_cross_membership),
            (self.step_temporal_counts, other.step_temporal_counts),
            (self.negative_calories, other.negative_calories),
            (self.weight_counts, other.weight_counts),
            (self.weight_added_by, other.weight_added_by),
            (self.anthropometric_counts, other.anthropometric_counts),
            (self.provenance_counts, other.provenance_counts),
            (self.source_day_counts, other.source_day_counts),
            (self.sleep_overlap_counts, other.sleep_overlap_counts),
            (self.sleep_overlap_rows_affected, other.sleep_overlap_rows_affected),
            (self.quality_rule_counts, other.quality_rule_counts),
        ):
            target.update(source)
        for target, source in (
            (self.unit_stats, other.unit_stats),
            (self.activity_quantity, other.activity_quantity),
            (self.step_temporal_duration, other.step_temporal_duration),
            (self.negative_calories_values, other.negative_calories_values),
            (self.weight_duration, other.weight_duration),
            (self.weight_values, other.weight_values),
            (self.sleep_overlap_duration, other.sleep_overlap_duration),
        ):
            for key, value in source.items():
                if key not in target:
                    target[key] = value
                else:
                    target[key].merge(value)
        self.anthropometric_bmi.merge(other.anthropometric_bmi)
        self.anthropometric_time_delta_days.merge(other.anthropometric_time_delta_days)
        for target, source in (
            (self.activity_quantity_distance, other.activity_quantity_distance),
            (self.activity_quantity_calories, other.activity_quantity_calories),
            (self.activity_quantity_stepcount, other.activity_quantity_stepcount),
            (self.activity_concordance, other.activity_concordance),
            (self.step_cross_pairs, other.step_cross_pairs),
            (self.step_value_raw_pairs, other.step_value_raw_pairs),
        ):
            for key, value in source.items():
                if key not in target:
                    target[key] = value
                else:
                    target[key].merge(value)
        self.schema_issues.extend(other.schema_issues)
        if len(self.schema_issues) > 1000:
            self.schema_issues[:] = sorted(set(self.schema_issues))[:1000]


def flatten_summary(prefix: str, value: Any) -> dict[str, Any]:
    """Flatten nested numeric summaries for stable CSV rendering."""

    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}_{key}" if prefix else str(key)
            result.update(flatten_summary(child, item))
    else:
        result[prefix] = value
    return result


def sorted_counter_rows(counter: Counter[Any]) -> Iterable[tuple[Any, int]]:
    return sorted(counter.items(), key=lambda item: repr(item[0]))
