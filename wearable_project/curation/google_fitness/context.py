"""
Wearable Data Processing and Modeling project
Group-aware curation context for Google Fitness feature files. Group-aware logic is kept separate from row-local
rule execution. The context builders never delete, reorder, or modify native rows. They return immutable per-row
annotations that the curation engine may append as derived metadata.
"""


from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Mapping, Sequence
from .models import CurationStatus, StepCountRole


_DETAILED_SLEEP_STATES = frozenset({"DEEP", "LIGHT", "REM"})
_SLEEP_STATES = frozenset({"SLEEP", "AWAKE", *_DETAILED_SLEEP_STATES})


@dataclass(frozen=True, slots=True)
class StepCountRowContext:
    """Group-derived role, inclusion, and flags for one StepCount row."""

    role: StepCountRole
    status: CurationStatus
    include_by_default: bool
    flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SleepRowContext:
    """Overlap flags for one native Sleep row."""

    flags: tuple[str, ...] = ()


def parse_datetime(value: str | None) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def finite_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def aggregation_kind(value: str | None) -> str:
    """Normalize exporter-qualified StepCount source names for policy logic."""

    text = (value or "").strip()
    folded = text.casefold()
    if "estimated_steps" in folded:
        return "estimated_steps"
    if "merge_step_deltas" in folded:
        return "merge_step_deltas"
    return text or "<missing>"


def _different_numeric_values(rows: Sequence[Mapping[str, str]]) -> bool:
    values = [finite_float(row.get("value")) for row in rows]
    finite = [value for value in values if value is not None]
    return len(set(finite)) > 1


def build_step_count_context(rows: Sequence[Mapping[str, str]]) -> dict[int, StepCountRowContext]:
    """Build deterministic version-aware StepCount roles without dropping rows."""

    result: dict[int, StepCountRowContext] = {}
    v2_groups: dict[str, list[int]] = defaultdict(list)

    for index, row in enumerate(rows):
        version = (row.get("collecting_method_version") or "").strip()
        if version == "1.0":
            result[index] = StepCountRowContext(
                role=StepCountRole.LEGACY_V1,
                status=CurationStatus.PASS,
                include_by_default=True,
            )
        elif version == "2.0":
            outer_id = (row.get("outer_id") or "").strip()
            # Missing IDs must never merge unrelated rows.  Keep each row in its
            # own deterministic synthetic group and mark its structure invalid.
            key = outer_id if outer_id else f"__missing_outer_id__:{index}"
            v2_groups[key].append(index)
        else:
            result[index] = StepCountRowContext(
                role=StepCountRole.AMBIGUOUS_PARALLEL,
                status=CurationStatus.REVIEW,
                include_by_default=False,
                flags=("invalid_step_count_group_structure",),
            )

    for group_key in sorted(v2_groups):
        indices = v2_groups[group_key]
        group_rows = [rows[index] for index in indices]
        estimated = [index for index in indices if aggregation_kind(rows[index].get("aggregation_source")) == "estimated_steps"]
        merged = [index for index in indices if aggregation_kind(rows[index].get("aggregation_source")) == "merge_step_deltas"]

        preferred: int | None = None
        preferred_role: StepCountRole | None = None
        if len(estimated) == 1:
            preferred = estimated[0]
            preferred_role = StepCountRole.FIT_APP_PREFERRED
        elif not estimated and len(merged) == 1:
            preferred = merged[0]
            preferred_role = StepCountRole.MERGE_FALLBACK
        elif len(indices) == 1:
            preferred = indices[0]
            preferred_role = StepCountRole.THIRD_PARTY_FALLBACK

        disagreement = _different_numeric_values(group_rows)
        malformed_group = (
            group_key.startswith("__missing_outer_id__:")
            or len(estimated) > 1
            or (not estimated and len(merged) > 1)
        )

        for index in indices:
            flags: set[str] = set()
            if disagreement:
                flags.add("step_count_alternatives_disagree")
            if malformed_group:
                flags.add("invalid_step_count_group_structure")

            if preferred is None:
                role = StepCountRole.AMBIGUOUS_PARALLEL
                status = CurationStatus.REVIEW
                include = False
            elif index == preferred:
                assert preferred_role is not None
                role = preferred_role
                status = (
                    CurationStatus.REVIEW
                    if role in {StepCountRole.MERGE_FALLBACK, StepCountRole.THIRD_PARTY_FALLBACK}
                    else CurationStatus.PASS
                )
                include = True
            else:
                role = StepCountRole.PARALLEL_NONPREFERRED
                status = CurationStatus.PASS
                include = False

            if disagreement and status is CurationStatus.PASS:
                status = CurationStatus.REVIEW
            if malformed_group:
                status = CurationStatus.REVIEW
                include = False

            result[index] = StepCountRowContext(
                role=role,
                status=status,
                include_by_default=include,
                flags=tuple(sorted(flags)),
            )

    return result


def _overlap(left_start: datetime, left_end: datetime, right_start: datetime, right_end: datetime) -> bool:
    return left_start < right_end and right_start < left_end


def _sleep_pair_flag(left_state: str, right_state: str) -> str | None:
    if left_state == right_state:
        return "overlapping_same_sleep_state"
    pair = {left_state, right_state}
    if "AWAKE" in pair and pair & {"SLEEP", *_DETAILED_SLEEP_STATES}:
        return "awake_overlaps_sleep_state"
    if "SLEEP" in pair and pair & _DETAILED_SLEEP_STATES:
        return "generic_sleep_overlaps_detailed_state"
    if left_state in _DETAILED_SLEEP_STATES and right_state in _DETAILED_SLEEP_STATES:
        return "overlapping_detailed_sleep_states"
    return None


def build_sleep_context(rows: Sequence[Mapping[str, str]]) -> dict[int, SleepRowContext]:
    """Classify same-provenance Sleep overlaps while retaining every interval."""

    flags_by_index: dict[int, set[str]] = defaultdict(set)
    groups: dict[tuple[str, str, str, str], list[tuple[datetime, datetime, int, str]]] = defaultdict(list)

    for index, row in enumerate(rows):
        start = parse_datetime(row.get("start_date"))
        end = parse_datetime(row.get("end_date"))
        state = (row.get("value") or "").strip()
        if start is None or end is None or end <= start or state not in _SLEEP_STATES:
            continue
        key = tuple((row.get(column) or "").strip() for column in (
            "collecting_method_version", "source_id", "source_name", "device"
        ))
        groups[key].append((start, end, index, state))

    for key in sorted(groups):
        intervals = sorted(groups[key], key=lambda item: (item[0], item[1], item[2]))
        active: list[tuple[datetime, datetime, int, str]] = []
        for current in intervals:
            start, end, index, state = current
            active = [item for item in active if item[1] > start]
            for other_start, other_end, other_index, other_state in active:
                if not _overlap(start, end, other_start, other_end):
                    continue
                flag = _sleep_pair_flag(state, other_state)
                if flag:
                    flags_by_index[index].add(flag)
                    flags_by_index[other_index].add(flag)
            active.append(current)

    return {
        index: SleepRowContext(tuple(sorted(flags)))
        for index, flags in sorted(flags_by_index.items())
    }
