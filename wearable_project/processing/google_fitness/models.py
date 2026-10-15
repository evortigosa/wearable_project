"""
Wearable Data Processing and Modeling project
Immutable lineage records and exact accounting models for Google Fitness WP2.
"""


from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PARSER_VERSION = "google-fitness-parser-wp2-0.1.0"


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
        if sum(self.native_feature_candidates.values()) != self.valid_candidates:
            issues.append("native-feature candidate counts do not close")
        if len(self.malformed_items) != self.malformed_candidates:
            issues.append("malformed-item ledger count != malformed_candidates")
        if self.activity_items_seen != self.activity_items_routed + self.activity_items_malformed:
            issues.append("activity candidate accounting does not close")
        if self.activity_items_routed != (
            self.registered_activity_items_routed
            + self.unregistered_activity_items_routed
        ):
            issues.append("registered/unregistered activity routing does not close")
        if self.step_count_v2_items_seen != (
            self.step_count_v2_alternatives_emitted
            + self.step_count_v2_items_malformed
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
