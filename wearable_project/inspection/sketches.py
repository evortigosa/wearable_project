"""
Wearable Data Processing and Modeling project
Bounded deterministic sketches used by the raw-export profiler.
"""


from __future__ import annotations
import hashlib
import heapq
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable


def _hash64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "big")


def stable_rank(value: Any) -> str:
    """
    Return a process-order-independent rank for JSON-like evidence. ``default=str`` mirrors the profiler's
    report serialization. SHA-256 is intentionally used here rather than Python's randomized ``hash`` so the
    selected examples are stable across workers, processes, checkpoints, and interpreter invocations.
    """
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()


@dataclass
class RankedExamples:
    """Bounded deterministic examples selected by the smallest stable ranks."""

    capacity: int = 100
    entries: dict[str, Any] = field(default_factory=dict)

    def add(self, value: Any) -> str:
        rank = stable_rank(value)
        if self.capacity <= 0:
            return rank
        self.entries.setdefault(rank, value)
        if len(self.entries) > self.capacity:
            for discarded in sorted(self.entries, reverse=True)[: len(self.entries) - self.capacity]:
                del self.entries[discarded]
        return rank

    def merge(self, other: "RankedExamples") -> None:
        for rank, value in other.entries.items():
            self.entries.setdefault(rank, value)
        if len(self.entries) > self.capacity:
            for discarded in sorted(self.entries, reverse=True)[: len(self.entries) - self.capacity]:
                del self.entries[discarded]

    def items(self) -> list[tuple[str, Any]]:
        return [(rank, self.entries[rank]) for rank in sorted(self.entries)]

    def values(self) -> list[Any]:
        return [value for _, value in self.items()]

    def __len__(self) -> int:
        return len(self.entries)


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


@dataclass
class NumericSketch:
    """Exact counts/range plus a deterministic bottom-k sample for quantiles."""

    sample_size: int = 4096
    count: int = 0
    finite_count: int = 0
    zero_count: int = 0
    negative_count: int = 0
    nonfinite_count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    total: float = 0.0
    # Max-heap represented as (-hash, value). The largest retained hash is at index 0.
    _sample_heap: list[tuple[int, float]] = field(default_factory=list)

    def add(self, value: Any, context: str) -> None:
        if isinstance(value, bool):
            return
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return
        self.count += 1
        if not math.isfinite(number):
            self.nonfinite_count += 1
            return
        self.finite_count += 1
        self.total += number
        self.zero_count += int(number == 0)
        self.negative_count += int(number < 0)
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)
        if self.sample_size <= 0:
            return
        rank = _hash64(context)
        entry = (-rank, number)
        if len(self._sample_heap) < self.sample_size:
            heapq.heappush(self._sample_heap, entry)
        elif rank < -self._sample_heap[0][0]:
            heapq.heapreplace(self._sample_heap, entry)

    def merge(self, other: "NumericSketch") -> None:
        self.count += other.count
        self.finite_count += other.finite_count
        self.zero_count += other.zero_count
        self.negative_count += other.negative_count
        self.nonfinite_count += other.nonfinite_count
        self.total += other.total
        if other.minimum is not None:
            self.minimum = other.minimum if self.minimum is None else min(self.minimum, other.minimum)
        if other.maximum is not None:
            self.maximum = other.maximum if self.maximum is None else max(self.maximum, other.maximum)
        for negative_rank, value in other._sample_heap:
            rank = -negative_rank
            entry = (-rank, value)
            if len(self._sample_heap) < self.sample_size:
                heapq.heappush(self._sample_heap, entry)
            elif rank < -self._sample_heap[0][0]:
                heapq.heapreplace(self._sample_heap, entry)

    @property
    def sample_values(self) -> list[float]:
        return [value for _, value in self._sample_heap]

    def summary(self) -> dict[str, Any]:
        sample = self.sample_values
        return {
            "numeric_count": self.count,
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "numeric_min": self.minimum,
            "numeric_p01": _quantile(sample, 0.01),
            "numeric_p25": _quantile(sample, 0.25),
            "numeric_median": _quantile(sample, 0.50),
            "numeric_p75": _quantile(sample, 0.75),
            "numeric_p99": _quantile(sample, 0.99),
            "numeric_max": self.maximum,
            "numeric_mean": self.total / self.finite_count if self.finite_count else None,
            "numeric_zero_count": self.zero_count,
            "numeric_negative_count": self.negative_count,
            "quantile_sample_size": len(sample),
            "quantile_method": "exact" if self.finite_count <= self.sample_size else "deterministic_bottom_k",
        }


@dataclass
class BoundedCounter:
    """
    A bounded heavy-hitter counter. Counts are exact until pruning occurs. After pruning, emitted counts are
    lower bounds; this is explicitly reported. Low-cardinality fields remain exact.
    """

    capacity: int = 200
    counts: Counter[str] = field(default_factory=Counter)
    pruned_mass: int = 0
    prune_operations: int = 0

    def add(self, value: Any, count: int = 1) -> None:
        text = str(value)
        self.counts[text] += count
        if len(self.counts) > max(2 * self.capacity, self.capacity + 32):
            self._prune()

    def _prune(self) -> None:
        # Resolve equal-count ties lexicographically so pruning does not depend on process completion order or
        # Counter insertion order.
        retained = sorted(
            self.counts.items(), key=lambda item: (-item[1], item[0])
        )[: self.capacity]
        retained_keys = {key for key, _ in retained}
        self.pruned_mass += sum(count for key, count in self.counts.items() if key not in retained_keys)
        self.counts = Counter(dict(retained))
        self.prune_operations += 1

    def merge(self, other: "BoundedCounter") -> None:
        """
        Merge another bounded counter. Counts retained by either input remain lower bounds after a prior prune.
        ``pruned_mass`` therefore accumulates both already-discarded mass and any additional mass discarded
        while combining the retained candidates.
        """
        self.pruned_mass += other.pruned_mass
        self.prune_operations += other.prune_operations
        for key, count in sorted(other.counts.items()):
            self.counts[key] += count
        if len(self.counts) > max(2 * self.capacity, self.capacity + 32):
            self._prune()

    def items(self) -> Iterable[tuple[str, int]]:
        if len(self.counts) > self.capacity:
            self._prune()
        return sorted(
            self.counts.items(), key=lambda item: (-item[1], item[0])
        )[: self.capacity]

    @property
    def approximate(self) -> bool:
        return self.prune_operations > 0
