"""
Wearable Data Processing and Modeling project
Deterministic native cleaning and reconciliation for Google Fitness WP3. The WP2 parser has already decoded and
normalized supported source envelopes into records that match the immutable WP1 native schemas. This module performs
the next semantic step without widening those schemas:

* validate every candidate against its registered feature contract;
* reconcile repeated physical source occurrences;
* resolve changed occurrences only when source lifecycle evidence gives a unique later winner;
* collapse exact same-version semantic duplicates;
* preserve divergent same-identity observations as explicit conflicts;
* never reconcile across collecting-method versions; and
* emit a total deterministic row order independent of input/month order.

Audit and conflict evidence is returned in a parallel ledger for the future Google Fitness state database. It is
deliberately not repeated in feature CSVs.
"""


from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from enum import Enum
from hashlib import sha256
import json
import math
from typing import Any, Iterable, Mapping, Sequence
from wearable_project.processing.google_fitness.models import (
    CANDIDATE_LINEAGE_VERSION, PARSER_VERSION, CandidateLineage, ParseResult, SourceOccurrence,
)
from wearable_project.processing.google_fitness.registry import (
    EXPECTED_REGISTRY_FINGERPRINT, REGISTRY_VERSION, ColumnSpec, ColumnType, IdentityMaturity,
    NativeFeatureFamily, NativeFeatureSpec, SLEEP_STATE_MAPPING, UNREGISTERED_ACTIVITY_POLICY,
    get_feature_spec,
)


CLEANER_VERSION = "google-fitness-cleaner-wp3-0.1.0"
_SLEEP_STATES = frozenset(SLEEP_STATE_MAPPING.values())


class ReconciliationStage(str, Enum):
    OCCURRENCE = "occurrence"
    SEMANTIC = "semantic"


class ReconciliationAction(str, Enum):
    EXACT_DUPLICATE = "exact_duplicate"
    REVISION_RESOLVED = "revision_resolved"
    CONFLICT_PRESERVED = "conflict_preserved"


@dataclass(frozen=True, slots=True)
class CleanedRowAudit:
    """State-ledger evidence for one final native row."""

    row_id: str
    winner_candidate_key: str
    winner_source_occurrence_key: str
    candidate_keys: tuple[str, ...]
    source_occurrence_keys: tuple[str, ...]
    occurrence_count: int
    duplicate_count: int
    revision_count: int
    conflict_group_ids: tuple[str, ...]
    content_fingerprint: str
    semantic_identity_fingerprint: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "winner_candidate_key": self.winner_candidate_key,
            "winner_source_occurrence_key": self.winner_source_occurrence_key,
            "candidate_keys": list(self.candidate_keys),
            "source_occurrence_keys": list(self.source_occurrence_keys),
            "occurrence_count": self.occurrence_count,
            "duplicate_count": self.duplicate_count,
            "revision_count": self.revision_count,
            "conflict_group_ids": list(self.conflict_group_ids),
            "content_fingerprint": self.content_fingerprint,
            "semantic_identity_fingerprint": self.semantic_identity_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ReconciliationDecision:
    """One deterministic duplicate, revision, or conflict decision."""

    feature: str
    stage: ReconciliationStage
    action: ReconciliationAction
    group_id: str
    identity_json: str
    representative_candidate_keys: tuple[str, ...]
    removed_candidate_keys: tuple[str, ...]
    all_candidate_keys: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "stage": self.stage.value,
            "action": self.action.value,
            "group_id": self.group_id,
            "identity_json": self.identity_json,
            "representative_candidate_keys": list(self.representative_candidate_keys),
            "removed_candidate_keys": list(self.removed_candidate_keys),
            "all_candidate_keys": list(self.all_candidate_keys),
        }


@dataclass(frozen=True, slots=True)
class FeatureCleanResult:
    """Final rows and exact reconciliation accounting for one feature."""

    feature: str
    rows: tuple[dict[str, Any], ...]
    audits: tuple[CleanedRowAudit, ...]
    decisions: tuple[ReconciliationDecision, ...]
    input_candidates: int
    output_rows: int
    represented_occurrences: int
    occurrence_duplicates_removed: int
    semantic_duplicates_removed: int
    revisions_resolved: int
    occurrence_conflict_groups: int
    semantic_conflict_groups: int
    conflict_rows: int
    output_fingerprint: str

    @property
    def exact_duplicates_removed(self) -> int:
        return self.occurrence_duplicates_removed + self.semantic_duplicates_removed

    @property
    def unresolved_conflict_groups(self) -> int:
        return self.occurrence_conflict_groups + self.semantic_conflict_groups

    def accounting_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        if len(self.rows) != self.output_rows:
            issues.append("rows length != output_rows")
        if len(self.audits) != self.output_rows:
            issues.append("audit length != output_rows")
        if self.input_candidates != (
            self.output_rows + self.exact_duplicates_removed + self.revisions_resolved
        ):
            issues.append("input_candidates != output_rows + exact_duplicates_removed + revisions_resolved")
        if self.represented_occurrences != self.input_candidates:
            issues.append("represented_occurrences != input_candidates")
        if sum(item.occurrence_count for item in self.audits) != self.input_candidates:
            issues.append("audit occurrence counts do not close")
        if sum(item.duplicate_count for item in self.audits) != self.exact_duplicates_removed:
            issues.append("audit duplicate counts do not close")
        if sum(item.revision_count for item in self.audits) != self.revisions_resolved:
            issues.append("audit revision counts do not close")
        if len({item.row_id for item in self.audits}) != len(self.audits):
            issues.append("final row ids are not unique")
        actual_conflict_rows = sum(bool(item.conflict_group_ids) for item in self.audits)
        if actual_conflict_rows != self.conflict_rows:
            issues.append("conflict row count does not match audit ledger")
        return tuple(issues)

    def assert_accounting_closed(self) -> None:
        issues = self.accounting_issues()
        if issues:
            raise AssertionError(f"{self.feature}: " + "; ".join(issues))

    def accounting_payload(self) -> dict[str, Any]:
        return {
            "cleaner_version": CLEANER_VERSION,
            "feature": self.feature,
            "input_candidates": self.input_candidates,
            "output_rows": self.output_rows,
            "represented_occurrences": self.represented_occurrences,
            "occurrence_duplicates_removed": self.occurrence_duplicates_removed,
            "semantic_duplicates_removed": self.semantic_duplicates_removed,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "revisions_resolved": self.revisions_resolved,
            "occurrence_conflict_groups": self.occurrence_conflict_groups,
            "semantic_conflict_groups": self.semantic_conflict_groups,
            "unresolved_conflict_groups": self.unresolved_conflict_groups,
            "conflict_rows": self.conflict_rows,
            "output_fingerprint": self.output_fingerprint,
            "accounting_issues": list(self.accounting_issues()),
        }


@dataclass(frozen=True, slots=True)
class ParticipantCleanResult:
    """All feature results and parser-to-native accounting for one participant."""

    features: Mapping[str, FeatureCleanResult]
    candidate_items_seen: int
    valid_candidates: int
    malformed_candidates: int
    input_candidates: int
    output_rows: int
    represented_occurrences: int
    exact_duplicates_removed: int
    revisions_resolved: int
    unresolved_conflict_groups: int
    conflict_rows: int
    output_fingerprint: str

    def accounting_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        values = tuple(self.features.values())
        if self.candidate_items_seen != self.valid_candidates + self.malformed_candidates:
            issues.append("candidate_items_seen != valid_candidates + malformed_candidates")
        if self.valid_candidates != self.input_candidates:
            issues.append("valid_candidates != cleaner input_candidates")
        expected = {
            "input_candidates": sum(item.input_candidates for item in values),
            "output_rows": sum(item.output_rows for item in values),
            "represented_occurrences": sum(item.represented_occurrences for item in values),
            "exact_duplicates_removed": sum(item.exact_duplicates_removed for item in values),
            "revisions_resolved": sum(item.revisions_resolved for item in values),
            "unresolved_conflict_groups": sum(item.unresolved_conflict_groups for item in values),
            "conflict_rows": sum(item.conflict_rows for item in values),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                issues.append(f"{name} does not equal the feature sum")
        if self.input_candidates != (
            self.output_rows + self.exact_duplicates_removed + self.revisions_resolved
        ):
            issues.append("input_candidates != output_rows + exact_duplicates_removed + revisions_resolved")
        for feature, result in self.features.items():
            for issue in result.accounting_issues():
                issues.append(f"{feature}: {issue}")
        return tuple(issues)

    def assert_accounting_closed(self) -> None:
        issues = self.accounting_issues()
        if issues:
            raise AssertionError("; ".join(issues))

    def accounting_payload(self) -> dict[str, Any]:
        return {
            "registry_version": REGISTRY_VERSION,
            "registry_fingerprint": EXPECTED_REGISTRY_FINGERPRINT,
            "parser_version": PARSER_VERSION,
            "candidate_lineage_version": CANDIDATE_LINEAGE_VERSION,
            "cleaner_version": CLEANER_VERSION,
            "candidate_items_seen": self.candidate_items_seen,
            "valid_candidates": self.valid_candidates,
            "malformed_candidates": self.malformed_candidates,
            "input_candidates": self.input_candidates,
            "output_rows": self.output_rows,
            "represented_occurrences": self.represented_occurrences,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "revisions_resolved": self.revisions_resolved,
            "unresolved_conflict_groups": self.unresolved_conflict_groups,
            "conflict_rows": self.conflict_rows,
            "output_fingerprint": self.output_fingerprint,
            "features": {name: result.accounting_payload() for name, result in sorted(self.features.items())},
            "accounting_issues": list(self.accounting_issues()),
        }


@dataclass(frozen=True, slots=True)
class _FeatureContract:
    feature: str
    source_feature: str
    columns: tuple[ColumnSpec, ...]
    column_names: tuple[str, ...]
    supported_versions: tuple[str, ...]
    occurrence_identity_columns: tuple[str, ...]
    semantic_identity_by_version: Mapping[str, tuple[IdentityMaturity, tuple[str, ...]]]
    sort_columns: tuple[str, ...]
    deduplicate_across_versions: bool
    strict_positive_interval: bool


@dataclass(frozen=True, slots=True)
class _RowState:
    record: dict[str, Any]
    source: SourceOccurrence
    audit: CleanedRowAudit
    record_json: str
    semantic_content_json: str
    occurrence_identity_json: str
    semantic_identity_json: str | None
    semantic_group_key: str | None
    revision_preference: tuple[Any, ...]
    deterministic_preference: tuple[Any, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _fingerprint_json(serialized: str) -> str:
    return sha256(serialized.encode("utf-8")).hexdigest()


def _fingerprint(value: Any) -> str:
    return _fingerprint_json(_canonical_json(value))


def _feature_contract(feature: str) -> _FeatureContract:
    if feature == UNREGISTERED_ACTIVITY_POLICY.native_feature:
        return _FeatureContract(
            feature=feature,
            source_feature="Activities",
            columns=UNREGISTERED_ACTIVITY_POLICY.columns,
            column_names=tuple(item.name for item in UNREGISTERED_ACTIVITY_POLICY.columns),
            supported_versions=("1.0",),
            occurrence_identity_columns=(*UNREGISTERED_ACTIVITY_POLICY.occurrence_identity_columns,),
            semantic_identity_by_version={"1.0": (IdentityMaturity.SOURCE_OCCURRENCE_ONLY, ())},
            sort_columns=UNREGISTERED_ACTIVITY_POLICY.sort_columns,
            deduplicate_across_versions=False,
            strict_positive_interval=True,
        )
    spec: NativeFeatureSpec = get_feature_spec(feature)
    return _FeatureContract(
        feature=feature,
        source_feature=spec.source_feature,
        columns=spec.columns,
        column_names=tuple(item.name for item in spec.columns),
        supported_versions=spec.supported_versions,
        occurrence_identity_columns=spec.occurrence_identity_columns,
        semantic_identity_by_version={
            item.version: (item.maturity, item.semantic_columns) for item in spec.semantic_identity_by_version
        },
        sort_columns=spec.sort_columns,
        deduplicate_across_versions=spec.deduplicate_across_versions,
        strict_positive_interval=spec.family in {
            NativeFeatureFamily.ACTIVITY_INTERVAL, NativeFeatureFamily.SLEEP_STATE_INTERVAL,
        },
    )


def _normalize_iso_datetime(value: str) -> datetime:
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(candidate)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_value(column: ColumnSpec, value: Any) -> None:
    if value is None:
        if not column.nullable:
            raise ValueError(f"{column.name}: required value is null")
        return
    if column.data_type is ColumnType.TIMESTAMP_UTC:
        if not isinstance(value, str):
            raise ValueError(f"{column.name}: expected UTC timestamp string")
        try:
            parsed = _normalize_iso_datetime(value)
        except ValueError as exc:
            raise ValueError(f"{column.name}: invalid timestamp {value!r}") from exc
        if parsed.utcoffset() is None or not value.endswith("Z"):
            raise ValueError(f"{column.name}: timestamp is not normalized UTC")
        return
    if column.data_type is ColumnType.DATE:
        if not isinstance(value, str):
            raise ValueError(f"{column.name}: expected ISO date string")
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{column.name}: invalid date {value!r}") from exc
        return
    if column.data_type is ColumnType.NUMBER:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{column.name}: expected finite number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{column.name}: non-finite number")
        return
    if column.data_type is ColumnType.INTEGER:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{column.name}: expected integer")
        return
    if column.data_type is ColumnType.BOOLEAN:
        if not isinstance(value, bool):
            raise ValueError(f"{column.name}: expected boolean")
        return
    if column.data_type is ColumnType.STRING:
        if not isinstance(value, str):
            raise ValueError(f"{column.name}: expected string")
        return
    raise AssertionError(f"Unhandled column type: {column.data_type}")


def _validate_record(
    feature: str, record: Mapping[str, Any], lineage: CandidateLineage, source: SourceOccurrence,
    contract: _FeatureContract,
) -> dict[str, Any]:
    if tuple(record) != contract.column_names:
        raise ValueError(
            f"{feature}: candidate columns {tuple(record)} do not match registered columns {contract.column_names}"
        )
    normalized = dict(record)
    for column in contract.columns:
        _validate_value(column, normalized[column.name])
    version = normalized.get("collecting_method_version")
    if version not in contract.supported_versions:
        raise ValueError(f"{feature}: unsupported candidate version {version!r}")
    if lineage.collecting_method_version != version:
        raise ValueError(f"{feature}: candidate/version lineage mismatch")
    if lineage.outer_id != normalized.get("outer_id"):
        raise ValueError(f"{feature}: candidate/outer-id lineage mismatch")
    if "payload_index" in normalized and lineage.payload_index != normalized["payload_index"]:
        raise ValueError(f"{feature}: candidate/payload-index lineage mismatch")
    if source.outer_id != lineage.outer_id:
        raise ValueError(f"{feature}: source occurrence outer id mismatch")
    if source.collecting_method_version != lineage.collecting_method_version:
        raise ValueError(f"{feature}: source occurrence version mismatch")
    if source.source_feature != contract.source_feature:
        raise ValueError(
            f"{feature}: source feature {source.source_feature!r} does not match "
            f"registered source feature {contract.source_feature!r}"
        )
    if source.decoded_payload_sha256 is None:
        raise ValueError(f"{feature}: valid candidate lacks decoded payload lineage")
    if source.payload_item_count is not None and not 0 <= lineage.payload_index < source.payload_item_count:
        raise ValueError(f"{feature}: payload index is outside source cardinality")
    if contract.deduplicate_across_versions:
        raise ValueError(f"{feature}: cross-version reconciliation is forbidden in WP3")

    start = normalized.get("start_date")
    end = normalized.get("end_date")
    if start is not None and end is not None:
        start_dt = _normalize_iso_datetime(start)
        end_dt = _normalize_iso_datetime(end)
        if end_dt < start_dt:
            raise ValueError(f"{feature}: end_date precedes start_date")
        if contract.strict_positive_interval and end_dt <= start_dt:
            raise ValueError(f"{feature}: interval must have positive duration")
    if feature == "Sleep" and normalized.get("value") not in _SLEEP_STATES:
        raise ValueError(f"Sleep: unregistered native state {normalized.get('value')!r}")
    if feature == "StepCount":
        if version == "1.0":
            forbidden = (
                "summary_date",
                "start_date",
                "end_date",
                "raw_value",
                "aggregation_source",
                "origin_data_source_id",
                "app_name",
                "app_package_name",
                "device_manufacturer",
                "device_model",
                "device_type",
                "device_uid",
            )
            populated = [name for name in forbidden if normalized.get(name) is not None]
            if populated:
                raise ValueError("StepCount 1.0 contains invented version-2 fields: " + ", ".join(populated))
        elif version == "2.0":
            required = (
                "summary_date",
                "start_date",
                "end_date",
                "raw_value",
                "aggregation_source",
            )
            missing = [name for name in required if normalized.get(name) is None]
            if missing:
                raise ValueError("StepCount 2.0 is missing required evidence: " + ", ".join(missing))
    return normalized


def _timestamp_preference(value: str) -> tuple[int, int]:
    text = value.strip()
    if not text:
        return (0, 0)
    candidates = [text]
    if text.endswith("Z"):
        candidates.append(text[:-1] + "+00:00")
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        parsed = parsed.astimezone(timezone.utc)
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        delta = parsed - epoch
        micros = (delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds)
        return 1, micros
    # Invalid lifecycle strings are retained in SourceOccurrence but do not establish revision precedence.
    # Canonical snapshot month remains the final supported lifecycle signal.
    return 0, 0


def _revision_preference(source: SourceOccurrence) -> tuple[Any, ...]:
    """Lifecycle evidence allowed to resolve changed same-occurrence content."""

    return (
        _timestamp_preference(source.updated_at_raw),
        _timestamp_preference(source.created_at_raw),
        source.canonical_month,
    )


def _unique_revision_winner(states: Sequence[_RowState]) -> _RowState | None:
    """
    Return a unique later row only from comparable lifecycle evidence. A valid timestamp is not considered
    later merely because another variant has a missing or invalid timestamp. A lifecycle field participates only
    when every still-tied variant has a valid value. Canonical snapshot month is always comparable and is the
    final supported signal.
    """

    candidates = list(states)
    for index in (0, 1):
        values = [item.revision_preference[index] for item in candidates]
        if any(valid == 0 for valid, _ in values):
            continue
        maximum = max(micros for _, micros in values)
        candidates = [item for item in candidates if item.revision_preference[index][1] == maximum]
        if len(candidates) == 1:
            return candidates[0]
    maximum_month = max(item.revision_preference[2] for item in candidates)
    candidates = [item for item in candidates if item.revision_preference[2] == maximum_month]
    return candidates[0] if len(candidates) == 1 else None


def _deterministic_preference(source: SourceOccurrence) -> tuple[Any, ...]:
    """Total tie-break used only to select a representative, never infer a revision."""

    return (
        *_revision_preference(source),
        source.source_file,
        source.outer_row_number,
        source.occurrence_key,
    )


def _sort_token(value: Any) -> tuple[int, int, Any]:
    if value is None:
        return 1, 0, ""
    if isinstance(value, bool):
        return 0, 0, int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 0, 1, float(value)
    return 0, 2, str(value)


def _row_sort_key(state: _RowState, contract: _FeatureContract) -> tuple[Any, ...]:
    names = [*contract.sort_columns]
    for name in ("collecting_method_version", "outer_id", "payload_index"):
        if name in contract.column_names and name not in names:
            names.append(name)
    return (
        *(_sort_token(state.record.get(name)) for name in names),
        state.record_json, state.audit.winner_candidate_key,
    )


def _identity_json(columns: Sequence[str], record: Mapping[str, Any]) -> str:
    return _canonical_json([[name, record.get(name)] for name in columns])


def _row_id(feature: str, record: Mapping[str, Any]) -> str:
    return _fingerprint({"feature": feature, "record": record})


def _semantic_content(record: Mapping[str, Any], contract: _FeatureContract) -> dict[str, Any]:
    omitted = set(contract.occurrence_identity_columns)
    return {
        name: record[name] for name in contract.column_names if name not in omitted
    }


def _semantic_identity(record: Mapping[str, Any], contract: _FeatureContract) -> tuple[str, tuple[str, ...], str] | None:
    version = str(record["collecting_method_version"])
    maturity, columns = contract.semantic_identity_by_version[version]
    if maturity is IdentityMaturity.SOURCE_OCCURRENCE_ONLY:
        return None
    identity_json = _identity_json(columns, record)
    return version, columns, identity_json


def _initial_state(
    feature: str, record: dict[str, Any], lineage: CandidateLineage, source: SourceOccurrence,
    contract: _FeatureContract,
) -> _RowState:
    record_json = _canonical_json(record)
    semantic_content_json = _canonical_json(_semantic_content(record, contract))
    occurrence_identity_json = _identity_json(
        ("collecting_method_version", *contract.occurrence_identity_columns), record,
    )
    semantic = _semantic_identity(record, contract)
    if semantic is None:
        semantic_identity_json = None
        semantic_group_key = None
        semantic_fingerprint = None
    else:
        version, _, semantic_identity_json = semantic
        semantic_group_key = _canonical_json([version, json.loads(semantic_identity_json)])
        semantic_fingerprint = _fingerprint_json(semantic_identity_json)
    revision_preference = _revision_preference(source)
    deterministic_preference = (
        *revision_preference,
        source.source_file,
        source.outer_row_number,
        lineage.source_occurrence_key,
    )
    candidate_key = lineage.candidate_key
    audit = CleanedRowAudit(
        row_id=_fingerprint_json(_canonical_json({"feature": feature, "record": record})),
        winner_candidate_key=candidate_key,
        winner_source_occurrence_key=lineage.source_occurrence_key,
        candidate_keys=(candidate_key,),
        source_occurrence_keys=(lineage.source_occurrence_key,),
        occurrence_count=1,
        duplicate_count=0,
        revision_count=0,
        conflict_group_ids=(),
        content_fingerprint=_fingerprint_json(semantic_content_json),
        semantic_identity_fingerprint=semantic_fingerprint,
    )
    return _RowState(
        record=record,
        source=source,
        audit=audit,
        record_json=record_json,
        semantic_content_json=semantic_content_json,
        occurrence_identity_json=occurrence_identity_json,
        semantic_identity_json=semantic_identity_json,
        semantic_group_key=semantic_group_key,
        revision_preference=revision_preference,
        deterministic_preference=deterministic_preference,
    )


def _merge_audits(
    feature: str, winner: _RowState, states: Sequence[_RowState], *, added_duplicates: int = 0,
    added_revisions: int = 0, conflict_group_id: str | None = None,
) -> CleanedRowAudit:
    candidate_keys = tuple(sorted({key for state in states for key in state.audit.candidate_keys}))
    source_occurrence_keys = tuple(sorted({key for state in states for key in state.audit.source_occurrence_keys}))
    conflicts = {group for state in states for group in state.audit.conflict_group_ids}
    if conflict_group_id is not None:
        conflicts.add(conflict_group_id)
    return CleanedRowAudit(
        row_id=_row_id(feature, winner.record),
        winner_candidate_key=winner.audit.winner_candidate_key,
        winner_source_occurrence_key=winner.audit.winner_source_occurrence_key,
        candidate_keys=candidate_keys,
        source_occurrence_keys=source_occurrence_keys,
        occurrence_count=len(candidate_keys),
        duplicate_count=(sum(state.audit.duplicate_count for state in states) + added_duplicates),
        revision_count=(sum(state.audit.revision_count for state in states) + added_revisions),
        conflict_group_ids=tuple(sorted(conflicts)),
        content_fingerprint=winner.audit.content_fingerprint,
        semantic_identity_fingerprint=winner.audit.semantic_identity_fingerprint,
    )


def _representative(states: Sequence[_RowState]) -> _RowState:
    return max(states, key=lambda item: item.deterministic_preference)


def _decision(
    *, feature: str, stage: ReconciliationStage, action: ReconciliationAction, identity_json: str,
    representatives: Sequence[_RowState], removed: Sequence[_RowState], all_states: Sequence[_RowState],
) -> ReconciliationDecision:
    group_id = _fingerprint(
        {
            "feature": feature,
            "stage": stage.value,
            "identity": identity_json,
        }
    )
    return ReconciliationDecision(
        feature=feature,
        stage=stage,
        action=action,
        group_id=group_id,
        identity_json=identity_json,
        representative_candidate_keys=tuple(sorted(item.audit.winner_candidate_key for item in representatives)),
        removed_candidate_keys=tuple(sorted({key for item in removed for key in item.audit.candidate_keys})),
        all_candidate_keys=tuple(sorted({key for item in all_states for key in item.audit.candidate_keys})),
    )


def _collapse_exact_content(
    *, feature: str, stage: ReconciliationStage, identity_json: str, states: Sequence[_RowState], content_key,
    decisions: list[ReconciliationDecision],
) -> tuple[list[_RowState], int]:
    by_content: defaultdict[str, list[_RowState]] = defaultdict(list)
    for state in states:
        by_content[content_key(state)].append(state)
    retained: list[_RowState] = []
    removed_count = 0
    for content_json in sorted(by_content):
        group = by_content[content_json]
        content_fingerprint = _fingerprint_json(content_json)
        winner = _representative(group)
        duplicates = len(group) - 1
        if duplicates:
            removed = [item for item in group if item is not winner]
            decision = _decision(
                feature=feature,
                stage=stage,
                action=ReconciliationAction.EXACT_DUPLICATE,
                identity_json=_canonical_json(
                    {
                        "identity": json.loads(identity_json),
                        "content_fingerprint": content_fingerprint,
                    }
                ),
                representatives=(winner,),
                removed=removed,
                all_states=group,
            )
            decisions.append(decision)
            winner = replace(
                winner, audit=_merge_audits(feature, winner, group, added_duplicates=duplicates,),
            )
            removed_count += duplicates
        retained.append(winner)
    return retained, removed_count


def _reconcile_occurrences(
    feature: str, states: Sequence[_RowState], contract: _FeatureContract,
) -> tuple[list[_RowState], list[ReconciliationDecision], int, int, int]:
    grouped: defaultdict[str, list[_RowState]] = defaultdict(list)
    for state in states:
        grouped[state.occurrence_identity_json].append(state)

    retained: list[_RowState] = []
    decisions: list[ReconciliationDecision] = []
    duplicate_count = 0
    revision_count = 0
    conflict_groups = 0

    for identity_json in sorted(grouped):
        group = grouped[identity_json]
        content_rows, duplicates = _collapse_exact_content(
            feature=feature,
            stage=ReconciliationStage.OCCURRENCE,
            identity_json=identity_json,
            states=group,
            content_key=lambda state: state.record_json,
            decisions=decisions,
        )
        duplicate_count += duplicates
        if len(content_rows) == 1:
            retained.extend(content_rows)
            continue

        winner = _unique_revision_winner(content_rows)
        if winner is not None:
            losers = [item for item in content_rows if item is not winner]
            decision = _decision(
                feature=feature,
                stage=ReconciliationStage.OCCURRENCE,
                action=ReconciliationAction.REVISION_RESOLVED,
                identity_json=identity_json,
                representatives=(winner,),
                removed=losers,
                all_states=content_rows,
            )
            decisions.append(decision)
            winner = replace(
                winner, audit=_merge_audits(feature, winner, content_rows, added_revisions=len(losers),),
            )
            revision_count += len(losers)
            retained.append(winner)
        else:
            conflict_id = _fingerprint(
                {
                    "feature": feature,
                    "stage": ReconciliationStage.OCCURRENCE.value,
                    "identity": json.loads(identity_json),
                }
            )
            decision = _decision(
                feature=feature,
                stage=ReconciliationStage.OCCURRENCE,
                action=ReconciliationAction.CONFLICT_PRESERVED,
                identity_json=identity_json,
                representatives=content_rows,
                removed=(),
                all_states=content_rows,
            )
            decisions.append(decision)
            conflict_groups += 1
            retained.extend(
                replace( item, audit=_merge_audits(feature, item, (item,), conflict_group_id=conflict_id,),)
                for item in content_rows
            )
    return retained, decisions, duplicate_count, revision_count, conflict_groups


def _reconcile_semantic(
    feature: str, states: Sequence[_RowState], contract: _FeatureContract,
) -> tuple[list[_RowState], list[ReconciliationDecision], int, int]:
    passthrough: list[_RowState] = []
    grouped: defaultdict[str, list[_RowState]] = defaultdict(list)
    identity_json_by_key: dict[str, str] = {}
    for state in states:
        if state.semantic_group_key is None:
            passthrough.append(state)
            continue
        grouped[state.semantic_group_key].append(state)
        assert state.semantic_identity_json is not None
        identity_json_by_key[state.semantic_group_key] = state.semantic_identity_json

    retained = list(passthrough)
    decisions: list[ReconciliationDecision] = []
    duplicate_count = 0
    conflict_groups = 0
    for key in sorted(grouped):
        group = grouped[key]
        identity_json = identity_json_by_key[key]
        content_rows, duplicates = _collapse_exact_content(
            feature=feature,
            stage=ReconciliationStage.SEMANTIC,
            identity_json=identity_json,
            states=group,
            content_key=lambda state: state.semantic_content_json,
            decisions=decisions,
        )
        duplicate_count += duplicates
        if len(content_rows) == 1:
            retained.extend(content_rows)
            continue

        existing_conflicts = [set(item.audit.conflict_group_ids) for item in content_rows]
        common_existing = (
            set.intersection(*existing_conflicts) if existing_conflicts and all(existing_conflicts) else set()
        )
        if common_existing:
            # The divergent rows are already the preserved variants of one same-occurrence conflict. Do not count
            # the same uncertainty a second time merely because the semantic identity is identical.
            retained.extend(content_rows)
            continue

        conflict_id = _fingerprint(
            {
                "feature": feature,
                "stage": ReconciliationStage.SEMANTIC.value,
                "identity": json.loads(identity_json),
            }
        )
        decision = _decision(
            feature=feature,
            stage=ReconciliationStage.SEMANTIC,
            action=ReconciliationAction.CONFLICT_PRESERVED,
            identity_json=identity_json,
            representatives=content_rows,
            removed=(),
            all_states=content_rows,
        )
        decisions.append(decision)
        conflict_groups += 1
        retained.extend(
            replace(item, audit=_merge_audits(feature, item, (item,), conflict_group_id=conflict_id,),)
            for item in content_rows
        )
    return retained, decisions, duplicate_count, conflict_groups


def clean_feature(
    feature: str, records: Sequence[Mapping[str, Any]], lineages: Sequence[CandidateLineage],
    source_occurrences: Mapping[str, SourceOccurrence] | Iterable[SourceOccurrence],
) -> FeatureCleanResult:
    """Validate, reconcile, and deterministically order one native feature."""

    if len(records) != len(lineages):
        raise ValueError(f"{feature}: {len(records)} candidates but {len(lineages)} lineage records")
    contract = _feature_contract(feature)
    if isinstance(source_occurrences, Mapping):
        occurrence_by_key = dict(source_occurrences)
        if any(key != item.occurrence_key for key, item in occurrence_by_key.items()):
            raise ValueError("source occurrence mapping keys do not match records")
    else:
        occurrence_list = list(source_occurrences)
        occurrence_by_key = {item.occurrence_key: item for item in occurrence_list}
        if len(occurrence_by_key) != len(occurrence_list):
            raise ValueError("source occurrence keys are not unique")

    states: list[_RowState] = []
    for record, lineage in zip(records, lineages, strict=True):
        try:
            source = occurrence_by_key[lineage.source_occurrence_key]
        except KeyError as exc:
            raise ValueError(
                f"{feature}: unknown source occurrence {lineage.source_occurrence_key}"
            ) from exc
        validated = _validate_record(feature, record, lineage, source, contract)
        states.append(_initial_state(feature, validated, lineage, source, contract))

    occurrence_rows, occurrence_decisions, occurrence_duplicates, revisions, occurrence_conflicts = _reconcile_occurrences(
        feature, states, contract
    )
    semantic_rows, semantic_decisions, semantic_duplicates, semantic_conflicts = _reconcile_semantic(
        feature, occurrence_rows, contract
    )
    ordered = sorted(semantic_rows, key=lambda item: _row_sort_key(item, contract))
    rows = tuple(dict(item.record) for item in ordered)
    audits = tuple(item.audit for item in ordered)
    decisions = tuple(
        sorted(
            [*occurrence_decisions, *semantic_decisions],
            key=lambda item: (item.stage.value, item.action.value, item.group_id,),
        )
    )
    result = FeatureCleanResult(
        feature=feature,
        rows=rows,
        audits=audits,
        decisions=decisions,
        input_candidates=len(records),
        output_rows=len(rows),
        represented_occurrences=sum(item.occurrence_count for item in audits),
        occurrence_duplicates_removed=occurrence_duplicates,
        semantic_duplicates_removed=semantic_duplicates,
        revisions_resolved=revisions,
        occurrence_conflict_groups=occurrence_conflicts,
        semantic_conflict_groups=semantic_conflicts,
        conflict_rows=sum(bool(item.conflict_group_ids) for item in audits),
        output_fingerprint=_fingerprint({"feature": feature, "rows": rows}),
    )
    result.assert_accounting_closed()
    return result


def clean_parse_result(parsed: ParseResult) -> ParticipantCleanResult:
    """Clean every feature in a fully accounted WP2/WP3 parse result."""

    parsed.assert_accounting_closed()
    occurrence_by_key = {item.occurrence_key: item for item in parsed.source_occurrences}
    if len(occurrence_by_key) != len(parsed.source_occurrences):
        raise ValueError("source occurrence keys are not unique")

    features: dict[str, FeatureCleanResult] = {}
    for feature in sorted(parsed.records_by_feature):
        features[feature] = clean_feature(
            feature,
            parsed.records_by_feature[feature],
            parsed.candidate_lineage_by_feature.get(feature, ()),
            occurrence_by_key,
        )
    result = ParticipantCleanResult(
        features=features,
        candidate_items_seen=parsed.candidate_items_seen,
        valid_candidates=parsed.valid_candidates,
        malformed_candidates=parsed.malformed_candidates,
        input_candidates=sum(item.input_candidates for item in features.values()),
        output_rows=sum(item.output_rows for item in features.values()),
        represented_occurrences=sum(item.represented_occurrences for item in features.values()),
        exact_duplicates_removed=sum(item.exact_duplicates_removed for item in features.values()),
        revisions_resolved=sum(item.revisions_resolved for item in features.values()),
        unresolved_conflict_groups=sum(item.unresolved_conflict_groups for item in features.values()),
        conflict_rows=sum(item.conflict_rows for item in features.values()),
        output_fingerprint=_fingerprint(
            {
                name: item.output_fingerprint for name, item in sorted(features.items())
            }
        ),
    )
    if result.input_candidates != parsed.valid_candidates:
        raise AssertionError("cleaner input candidates do not match parser valid candidates")
    result.assert_accounting_closed()
    return result
