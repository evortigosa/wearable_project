"""
Wearable Data Processing and Modeling project
Immutable parser lineage and exact accounting models for Google Fitness WP2/WP3.
"""


from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any


PARSER_VERSION = "google-fitness-parser-wp2-0.1.0"
CANDIDATE_LINEAGE_VERSION = "google-fitness-candidate-lineage-wp3-0.1.0"


def source_occurrence_key(
    participant_id: str, canonical_month: str, source_file: str, source_file_sha256: str, outer_row_number: int,
    outer_id: str,
) -> str:
    """Return the stable identity of one physical outer-row occurrence."""

    material = "\0".join(
        (
            participant_id,
            canonical_month,
            source_file,
            source_file_sha256,
            str(outer_row_number),
            outer_id,
        )
    )
    return sha256(material.encode("utf-8")).hexdigest()


def candidate_occurrence_key(
    source_occurrence_key_value: str, outer_id: str, payload_index: int, collecting_method_version: str,
) -> str:
    """Return the stable identity of one decoded valid-candidate occurrence."""

    material = (
        f"{source_occurrence_key_value}\0{outer_id}\0{payload_index}\0"
        f"{collecting_method_version}"
    )
    return sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One canonical participant-month source file."""

    path: Path
    canonical_month: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ParseDiagnostic:
    """A deterministic parser diagnostic tied to source lineage."""

    source_file: str
    canonical_month: str
    row_number: int | None
    outer_id: str | None
    feature: str | None
    collecting_method_version: str | None
    payload_index: int | None
    stage: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class SourceOccurrence:
    """Outer-row lineage retained once rather than repeated in every native row."""

    participant_id: str
    outer_participant_id: str
    raw_source_label: str
    source_feature: str
    collecting_method_version: str
    outer_id: str
    outer_datetime_raw: str
    created_at_raw: str
    updated_at_raw: str
    source_file: str
    canonical_month: str
    source_file_sha256: str
    outer_row_number: int
    raw_payload_sha256: str
    decoded_payload_sha256: str | None
    payload_item_count: int | None

    @property
    def occurrence_key(self) -> str:
        return source_occurrence_key(
            self.participant_id,
            self.canonical_month,
            self.source_file,
            self.source_file_sha256,
            self.outer_row_number,
            self.outer_id,
        )


@dataclass(frozen=True, slots=True)
class CandidateLineage:
    """Parallel lineage for one valid native-shaped candidate."""

    source_occurrence_key: str
    outer_id: str
    payload_index: int
    collecting_method_version: str

    @property
    def candidate_key(self) -> str:
        return candidate_occurrence_key(
            self.source_occurrence_key,
            self.outer_id,
            self.payload_index,
            self.collecting_method_version,
        )


@dataclass(frozen=True, slots=True)
class MalformedCandidate:
    """One decoded source item that could not become a native candidate row."""

    source_file: str
    canonical_month: str
    row_number: int
    outer_id: str
    source_feature: str
    collecting_method_version: str
    payload_index: int
    reason_codes: tuple[str, ...]
    item_json: str


@dataclass(slots=True)
class ParseResult:
    """Candidate records, lineage, diagnostics, and exact parser accounting."""

    records_by_feature: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    candidate_lineage_by_feature: dict[str, list[CandidateLineage]] = field(default_factory=lambda: defaultdict(list))
    source_occurrences: list[SourceOccurrence] = field(default_factory=list)
    malformed_items: list[MalformedCandidate] = field(default_factory=list)
    diagnostics: list[ParseDiagnostic] = field(default_factory=list)

    outer_rows_seen: int = 0
    google_rows_seen: int = 0
    irrelevant_rows_seen: int = 0
    supported_outer_rows: int = 0
    unsupported_outer_rows: int = 0
    payloads_decoded: int = 0
    payload_decode_failures: int = 0
    candidate_items_seen: int = 0
    valid_candidates: int = 0
    malformed_candidates: int = 0

    activity_items_seen: int = 0
    activity_items_routed: int = 0
    registered_activity_items_routed: int = 0
    unregistered_activity_items_routed: int = 0
    activity_items_malformed: int = 0

    step_count_v2_items_seen: int = 0
    step_count_v2_alternatives_emitted: int = 0
    step_count_v2_items_malformed: int = 0

    source_contract_outer_rows: Counter[str] = field(default_factory=Counter)
    source_contract_candidate_items: Counter[str] = field(default_factory=Counter)
    source_contract_valid_candidates: Counter[str] = field(default_factory=Counter)
    source_contract_malformed_candidates: Counter[str] = field(default_factory=Counter)
    native_feature_candidates: Counter[str] = field(default_factory=Counter)
    malformed_reason_counts: Counter[str] = field(default_factory=Counter)
    outer_id_counts: Counter[str] = field(default_factory=Counter)

    unknown_activity_labels: set[str] = field(default_factory=set)
    unsupported_contracts: set[str] = field(default_factory=set)
    schema_warning_features: set[str] = field(default_factory=set)

    @property
    def payload_items_seen(self) -> int:
        """Compatibility/readability alias for decoded candidate items."""

        return self.candidate_items_seen

    @property
    def duplicate_outer_ids(self) -> tuple[str, ...]:
        return tuple(sorted(key for key, count in self.outer_id_counts.items() if count > 1))

    @property
    def total_record_count(self) -> int:
        return sum(len(records) for records in self.records_by_feature.values())

    @property
    def total_candidate_lineage_count(self) -> int:
        return sum(len(items) for items in self.candidate_lineage_by_feature.values())

    def accounting_issues(self) -> tuple[str, ...]:
        """Return every failed exact-accounting invariant."""

        issues: list[str] = []
        if self.outer_rows_seen != self.google_rows_seen + self.irrelevant_rows_seen:
            issues.append("outer_rows_seen != google_rows_seen + irrelevant_rows_seen")
        if self.google_rows_seen != self.supported_outer_rows + self.unsupported_outer_rows:
            issues.append("google_rows_seen != supported_outer_rows + unsupported_outer_rows")
        if self.supported_outer_rows != self.payloads_decoded + self.payload_decode_failures:
            issues.append("supported_outer_rows != payloads_decoded + payload_decode_failures")
        if sum(self.source_contract_outer_rows.values()) != self.supported_outer_rows:
            issues.append("source-contract outer-row counts do not close")
        if sum(self.source_contract_candidate_items.values()) != self.candidate_items_seen:
            issues.append("source-contract candidate-item counts do not close")
        contract_keys = set(self.source_contract_candidate_items)
        contract_keys.update(self.source_contract_valid_candidates)
        contract_keys.update(self.source_contract_malformed_candidates)
        for key in sorted(contract_keys):
            if self.source_contract_candidate_items[key] != (
                self.source_contract_valid_candidates[key] + self.source_contract_malformed_candidates[key]
            ):
                issues.append(f"source-contract candidate accounting does not close: {key}")
        if sum(self.source_contract_valid_candidates.values()) != self.valid_candidates:
            issues.append("source-contract valid-candidate counts do not close")
        if sum(self.source_contract_malformed_candidates.values()) != self.malformed_candidates:
            issues.append("source-contract malformed-candidate counts do not close")
        if self.candidate_items_seen != self.valid_candidates + self.malformed_candidates:
            issues.append("candidate_items_seen != valid_candidates + malformed_candidates")
        if self.total_record_count != self.valid_candidates:
            issues.append("records_by_feature row count != valid_candidates")
        if self.total_candidate_lineage_count != self.valid_candidates:
            issues.append("candidate-lineage row count != valid_candidates")
        feature_names = set(self.records_by_feature) | set(self.candidate_lineage_by_feature)
        for feature in sorted(feature_names):
            if len(self.records_by_feature.get(feature, ())) != len(self.candidate_lineage_by_feature.get(feature, ())):
                issues.append(f"candidate lineage does not align with records: {feature}")
        occurrence_keys = [item.occurrence_key for item in self.source_occurrences]
        if len(set(occurrence_keys)) != len(occurrence_keys):
            issues.append("source occurrence keys are not unique")
        known_occurrences = set(occurrence_keys)
        candidate_keys = [
            lineage.candidate_key for items in self.candidate_lineage_by_feature.values() for lineage in items
        ]
        if len(set(candidate_keys)) != len(candidate_keys):
            issues.append("candidate occurrence keys are not unique")
        missing_occurrences = sorted(
            {
                lineage.source_occurrence_key
                for items in self.candidate_lineage_by_feature.values()
                for lineage in items
            }
            - known_occurrences
        )
        if missing_occurrences:
            issues.append("candidate lineage references unknown source occurrences")
        for feature in sorted(feature_names):
            for record, lineage in zip(
                self.records_by_feature.get(feature, ()), self.candidate_lineage_by_feature.get(feature, ()),
            ):
                if record.get("outer_id") != lineage.outer_id:
                    issues.append(f"candidate outer id does not match lineage: {feature}")
                    break
                if record.get("collecting_method_version") != lineage.collecting_method_version:
                    issues.append(f"candidate version does not match lineage: {feature}")
                    break
                if "payload_index" in record and record.get("payload_index") != lineage.payload_index:
                    issues.append(f"candidate payload index does not match lineage: {feature}")
                    break
        if sum(self.native_feature_candidates.values()) != self.valid_candidates:
            issues.append("native-feature candidate counts do not close")
        if len(self.malformed_items) != self.malformed_candidates:
            issues.append("malformed-item ledger count != malformed_candidates")
        if self.activity_items_seen != self.activity_items_routed + self.activity_items_malformed:
            issues.append("activity candidate accounting does not close")
        if self.activity_items_routed != (
            self.registered_activity_items_routed + self.unregistered_activity_items_routed
        ):
            issues.append("registered/unregistered activity routing does not close")
        if self.step_count_v2_items_seen != (
            self.step_count_v2_alternatives_emitted + self.step_count_v2_items_malformed
        ):
            issues.append("StepCount 2.0 alternative accounting does not close")
        if len(self.source_occurrences) != self.google_rows_seen:
            issues.append("source occurrence ledger count != google_rows_seen")
        return tuple(issues)

    def assert_accounting_closed(self) -> None:
        issues = self.accounting_issues()
        if issues:
            raise AssertionError("; ".join(issues))

    def accounting_payload(self) -> dict[str, Any]:
        """Stable machine-readable parser accounting."""

        return {
            "parser_version": PARSER_VERSION,
            "outer_rows_seen": self.outer_rows_seen,
            "google_rows_seen": self.google_rows_seen,
            "irrelevant_rows_seen": self.irrelevant_rows_seen,
            "supported_outer_rows": self.supported_outer_rows,
            "unsupported_outer_rows": self.unsupported_outer_rows,
            "payloads_decoded": self.payloads_decoded,
            "payload_decode_failures": self.payload_decode_failures,
            "candidate_items_seen": self.candidate_items_seen,
            "valid_candidates": self.valid_candidates,
            "candidate_lineage_records": self.total_candidate_lineage_count,
            "malformed_candidates": self.malformed_candidates,
            "activity_items_seen": self.activity_items_seen,
            "activity_items_routed": self.activity_items_routed,
            "registered_activity_items_routed": self.registered_activity_items_routed,
            "unregistered_activity_items_routed": self.unregistered_activity_items_routed,
            "activity_items_malformed": self.activity_items_malformed,
            "step_count_v2_items_seen": self.step_count_v2_items_seen,
            "step_count_v2_alternatives_emitted": self.step_count_v2_alternatives_emitted,
            "step_count_v2_items_malformed": self.step_count_v2_items_malformed,
            "source_contract_outer_rows": dict(sorted(self.source_contract_outer_rows.items())),
            "source_contract_candidate_items": dict(sorted(self.source_contract_candidate_items.items())),
            "source_contract_valid_candidates": dict(sorted(self.source_contract_valid_candidates.items())),
            "source_contract_malformed_candidates": dict(sorted(self.source_contract_malformed_candidates.items())),
            "native_feature_candidates": dict(sorted(self.native_feature_candidates.items())),
            "malformed_reason_counts": dict(sorted(self.malformed_reason_counts.items())),
            "unknown_activity_labels": sorted(self.unknown_activity_labels),
            "unsupported_contracts": sorted(self.unsupported_contracts),
            "duplicate_outer_ids": list(self.duplicate_outer_ids),
            "accounting_issues": list(self.accounting_issues()),
        }
