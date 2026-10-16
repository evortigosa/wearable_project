"""
Wearable Data Processing and Modeling project
Version-aware parser for cumulative Google Fitness CSV exports. WP2 converts supported outer rows into
native-shaped candidate records while preserving source-occurrence lineage and closing candidate accounting
exactly. WP3 adds a parallel candidate-lineage ledger so reconciliation can distinguish physical repetitions
without widening the registered CSV schemas. The parser still does not reconcile duplicates, choose revisions,
write feature files, infer units, or select preferred StepCount alternatives.
"""


from __future__ import annotations
import ast
import csv
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence
from wearable_project.exceptions import (
    DuplicateMonthError, ParticipantProcessingError, PayloadDecodeError,
)
from wearable_project.processing.google_fitness.candidate_decoders import (
    _CandidateContext, _CandidateOutcome, _candidate_items_for_contract, _decode_candidate, _unique_sorted,
)
from wearable_project.processing.google_fitness.models import (
    PARSER_VERSION, CandidateLineage, MalformedCandidate, ParseDiagnostic, ParseResult, SourceFile,
    SourceOccurrence,
)
from wearable_project.processing.google_fitness.registry import (
    NORMALIZED_RAW_SOURCE, OUTER_SCHEMA_COLUMNS, UNREGISTERED_ACTIVITY_POLICY, get_source_feature_version_spec,
    normalize_source_label,
)


MONTH_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{1,2})\.csv$")
EXPECTED_COLUMNS = frozenset(OUTER_SCHEMA_COLUMNS)
ROW_ERROR_POLICIES = frozenset({"fail-participant", "skip-row"})


def canonical_month_from_name(name: str) -> str | None:
    match = MONTH_RE.match(name)
    if match is None:
        return None
    year = int(match.group("year"))
    month = int(match.group("month"))
    if not 1 <= month <= 12:
        return None
    return f"{year:04d}-{month:02d}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_month_files(participant_dir: Path) -> list[SourceFile]:
    by_month: dict[str, SourceFile] = {}
    for path in sorted(participant_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        month = canonical_month_from_name(path.name)
        if month is None:
            continue
        source = SourceFile(
            path=path,
            canonical_month=month,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
        if month in by_month:
            raise DuplicateMonthError(
                f"Participant {participant_dir.name!r} contains both "
                f"{by_month[month].path.name!r} and {path.name!r} for {month}."
            )
        by_month[month] = source
    return [by_month[key] for key in sorted(by_month)]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _load_literal(text: str) -> Any:
    json_error: Exception | None = None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        json_error = exc
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError) as exc:
        raise PayloadDecodeError(
            f"JSON error: {json_error}; Python-literal error: {exc}"
        ) from exc


def _decode_structured_quote_envelope(text: str) -> Any | None:
    if len(text) < 2 or text[0] != text[-1] or text[0] not in {'"', "'"}:
        return None
    inner = text[1:-1].strip()
    if len(inner) < 2 or (inner[0], inner[-1]) not in {
        ("[", "]"),
        ("{", "}"),
        ("(", ")"),
    }:
        return None
    try:
        return _load_literal(inner)
    except PayloadDecodeError:
        return None


def decode_payload(value: Any, max_layers: int = 8) -> Any:
    """Decode repeated JSON/Python-literal layers without source interpretation."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise PayloadDecodeError("Payload is null")
    current = value
    for _ in range(max_layers):
        if not isinstance(current, str):
            return current
        text = current.strip()
        if not text:
            raise PayloadDecodeError("Payload is empty")
        envelope = _decode_structured_quote_envelope(text)
        if envelope is not None:
            decoded = envelope
        else:
            try:
                decoded = _load_literal(text)
            except PayloadDecodeError:
                if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
                    current = text[1:-1]
                    continue
                raise
        if decoded == current:
            break
        current = decoded
    if isinstance(current, str):
        raise PayloadDecodeError("Payload remained a string after all decoding layers")
    return current


def _csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _contract_key(feature: str, version: str) -> str:
    return f"{feature}|{version}"


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _validate_row_error_policy(row_error_policy: str) -> None:
    if row_error_policy not in ROW_ERROR_POLICIES:
        raise ValueError(
            f"row_error_policy must be one of {sorted(ROW_ERROR_POLICIES)}, not {row_error_policy!r}"
        )


def _make_source_occurrence(
    *, source: SourceFile, participant_id: str, row_number: int, outer: dict[str, Any], outer_id: str,
    outer_participant_id: str, raw_source_label: str, feature: str, version: str, outer_datetime_raw: str,
    raw_payload_sha256: str, decoded_payload_sha256: str | None, payload_item_count: int | None,
) -> SourceOccurrence:
    return SourceOccurrence(
        participant_id=participant_id,
        outer_participant_id=outer_participant_id,
        raw_source_label=raw_source_label,
        source_feature=feature,
        collecting_method_version=version,
        outer_id=outer_id,
        outer_datetime_raw=outer_datetime_raw,
        created_at_raw=str(outer.get("created_at") or ""),
        updated_at_raw=str(outer.get("updated_at") or ""),
        source_file=source.path.name,
        canonical_month=source.canonical_month,
        source_file_sha256=source.sha256,
        outer_row_number=row_number,
        raw_payload_sha256=raw_payload_sha256,
        decoded_payload_sha256=decoded_payload_sha256,
        payload_item_count=payload_item_count,
    )


def parse_month_file(
    source: SourceFile, participant_id: str, *, row_error_policy: str = "fail-participant",
    result: ParseResult | None = None,
) -> ParseResult:
    """
    Parse one participant-month file. Supported malformed payload items are retained in ``malformed_items`` and
    do not abort the participant. Outer-row contract failures follow ``row_error_policy``.  Passing an existing
    ``result`` preserves partial counters and diagnostics if strict processing raises.
    """

    _validate_row_error_policy(row_error_policy)
    _csv_field_limit()
    if result is None:
        result = ParseResult()
    else:
        if not isinstance(result.records_by_feature, defaultdict):
            result.records_by_feature = defaultdict(list, result.records_by_feature)
        if not isinstance(result.candidate_lineage_by_feature, defaultdict):
            result.candidate_lineage_by_feature = defaultdict(
                list, result.candidate_lineage_by_feature
            )

    seen_diagnostics: set[tuple[Any, ...]] = set()

    def add_diagnostic(
        *, row_number: int | None, outer_id: str | None, feature: str | None, version: str | None,
        payload_index: int | None, stage: str, code: str, message: str, once_key: tuple[Any, ...] | None = None,
    ) -> None:
        if once_key is not None:
            if once_key in seen_diagnostics:
                return
            seen_diagnostics.add(once_key)
        result.diagnostics.append(
            ParseDiagnostic(
                source_file=source.path.name,
                canonical_month=source.canonical_month,
                row_number=row_number,
                outer_id=outer_id,
                feature=feature,
                collecting_method_version=version,
                payload_index=payload_index,
                stage=stage,
                code=code,
                message=message,
            )
        )

    def row_error(
        *, row_number: int, outer_id: str | None, feature: str | None, version: str | None, stage: str,
        code: str, message: str, cause: Exception | None = None,
    ) -> None:
        add_diagnostic(
            row_number=row_number,
            outer_id=outer_id,
            feature=feature,
            version=version,
            payload_index=None,
            stage=stage,
            code=code,
            message=message,
        )
        if row_error_policy == "fail-participant":
            error = ParticipantProcessingError(
                f"{participant_id}/{source.path.name}:{row_number} "
                f"{feature or '<unknown>'}/{version or '<unknown>'}: {message}"
            )
            if cause is not None:
                raise error from cause
            raise error

    try:
        handle = source.path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ParticipantProcessingError(f"Could not open {source.path}: {exc}") from exc

    with handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = EXPECTED_COLUMNS.difference(fieldnames)
        if missing:
            raise ParticipantProcessingError(f"{source.path} is missing outer columns: {sorted(missing)}")
        extras = sorted(set(fieldnames).difference(EXPECTED_COLUMNS))
        if extras:
            add_diagnostic(
                row_number=None,
                outer_id=None,
                feature=None,
                version=None,
                payload_index=None,
                stage="outer-schema",
                code="additional_outer_columns",
                message=f"Additional outer columns were ignored: {extras}",
                once_key=("additional_outer_columns", tuple(extras)),
            )

        for row_number, outer in enumerate(reader, start=2):
            result.outer_rows_seen += 1
            raw_source_label = str(outer.get("data_source") or "")
            if normalize_source_label(raw_source_label) != NORMALIZED_RAW_SOURCE:
                result.irrelevant_rows_seen += 1
                continue

            result.google_rows_seen += 1
            outer_id = str(outer.get("id") or "").strip()
            feature = str(outer.get("name") or "").strip()
            version = str(outer.get("collecting_method_version") or "").strip()
            outer_participant_id = str(outer.get("participant_id") or "").strip()
            outer_datetime_raw = str(outer.get("datetime") or "").strip()
            raw_payload = outer.get("data")
            raw_payload_text = "" if raw_payload is None else str(raw_payload)
            raw_payload_sha256 = _sha256_text(raw_payload_text)

            if not outer_id:
                result.unsupported_outer_rows += 1
                result.source_occurrences.append(
                    _make_source_occurrence(
                        source=source,
                        participant_id=participant_id,
                        row_number=row_number,
                        outer=outer,
                        outer_id=outer_id,
                        outer_participant_id=outer_participant_id,
                        raw_source_label=raw_source_label,
                        feature=feature,
                        version=version,
                        outer_datetime_raw=outer_datetime_raw,
                        raw_payload_sha256=raw_payload_sha256,
                        decoded_payload_sha256=None,
                        payload_item_count=None,
                    )
                )
                row_error(
                    row_number=row_number,
                    outer_id=None,
                    feature=feature or None,
                    version=version or None,
                    stage="outer-row",
                    code="missing_outer_id",
                    message="Missing outer row id",
                )
                continue

            result.outer_id_counts[outer_id] += 1

            if outer_participant_id and outer_participant_id != participant_id:
                result.unsupported_outer_rows += 1
                result.source_occurrences.append(
                    _make_source_occurrence(
                        source=source,
                        participant_id=participant_id,
                        row_number=row_number,
                        outer=outer,
                        outer_id=outer_id,
                        outer_participant_id=outer_participant_id,
                        raw_source_label=raw_source_label,
                        feature=feature,
                        version=version,
                        outer_datetime_raw=outer_datetime_raw,
                        raw_payload_sha256=raw_payload_sha256,
                        decoded_payload_sha256=None,
                        payload_item_count=None,
                    )
                )
                row_error(
                    row_number=row_number,
                    outer_id=outer_id,
                    feature=feature or None,
                    version=version or None,
                    stage="outer-row",
                    code="participant_id_mismatch",
                    message=(
                        f"Outer participant_id {outer_participant_id!r} differs from folder {participant_id!r}"
                    ),
                )
                continue

            try:
                get_source_feature_version_spec(feature, version)
            except KeyError as exc:
                contract_key = _contract_key(feature or "<missing>", version or "<missing>")
                result.unsupported_outer_rows += 1
                result.unsupported_contracts.add(contract_key)
                result.source_occurrences.append(
                    _make_source_occurrence(
                        source=source,
                        participant_id=participant_id,
                        row_number=row_number,
                        outer=outer,
                        outer_id=outer_id,
                        outer_participant_id=outer_participant_id,
                        raw_source_label=raw_source_label,
                        feature=feature,
                        version=version,
                        outer_datetime_raw=outer_datetime_raw,
                        raw_payload_sha256=raw_payload_sha256,
                        decoded_payload_sha256=None,
                        payload_item_count=None,
                    )
                )
                row_error(
                    row_number=row_number,
                    outer_id=outer_id,
                    feature=feature or None,
                    version=version or None,
                    stage="source-contract",
                    code="unsupported_feature_version",
                    message=str(exc),
                    cause=exc,
                )
                continue

            contract_key = _contract_key(feature, version)
            result.supported_outer_rows += 1
            result.source_contract_outer_rows[contract_key] += 1

            try:
                payload = decode_payload(raw_payload)
                items, envelope_reasons = _candidate_items_for_contract(payload, feature, version)
            except PayloadDecodeError as exc:
                result.payload_decode_failures += 1
                result.source_occurrences.append(
                    _make_source_occurrence(
                        source=source,
                        participant_id=participant_id,
                        row_number=row_number,
                        outer=outer,
                        outer_id=outer_id,
                        outer_participant_id=outer_participant_id,
                        raw_source_label=raw_source_label,
                        feature=feature,
                        version=version,
                        outer_datetime_raw=outer_datetime_raw,
                        raw_payload_sha256=raw_payload_sha256,
                        decoded_payload_sha256=None,
                        payload_item_count=None,
                    )
                )
                row_error(
                    row_number=row_number,
                    outer_id=outer_id,
                    feature=feature,
                    version=version,
                    stage="payload-decode",
                    code="payload_decode_failure",
                    message=str(exc),
                    cause=exc,
                )
                continue

            result.payloads_decoded += 1
            decoded_payload_sha256 = _sha256_text(canonical_json(payload))
            source_occurrence = _make_source_occurrence(
                source=source,
                participant_id=participant_id,
                row_number=row_number,
                outer=outer,
                outer_id=outer_id,
                outer_participant_id=outer_participant_id,
                raw_source_label=raw_source_label,
                feature=feature,
                version=version,
                outer_datetime_raw=outer_datetime_raw,
                raw_payload_sha256=raw_payload_sha256,
                decoded_payload_sha256=decoded_payload_sha256,
                payload_item_count=len(items),
            )
            result.source_occurrences.append(source_occurrence)
            source_occurrence_key_value = source_occurrence.occurrence_key

            if not items:
                result.schema_warning_features.add(feature)
                add_diagnostic(
                    row_number=row_number,
                    outer_id=outer_id,
                    feature=feature,
                    version=version,
                    payload_index=None,
                    stage="payload-contract",
                    code="decoded_payload_has_no_candidate_items",
                    message="; ".join(envelope_reasons) or "Decoded payload is empty",
                )

            result.candidate_items_seen += len(items)
            result.source_contract_candidate_items[contract_key] += len(items)
            if feature == "Activities":
                result.activity_items_seen += len(items)
            if feature == "StepCount" and version == "2.0":
                result.step_count_v2_items_seen += len(items)

            for payload_index, item in enumerate(items):
                context = _CandidateContext(
                    outer_id=outer_id,
                    source_feature=feature,
                    version=version,
                    outer_datetime=outer_datetime_raw,
                    payload_index=payload_index,
                )
                outcome = _decode_candidate(item, context)
                if envelope_reasons:
                    outcome = _CandidateOutcome(
                        native_feature=None,
                        record=None,
                        reason_codes=_unique_sorted([*envelope_reasons, *outcome.reason_codes]),
                    )
                if outcome.record is None or outcome.native_feature is None:
                    reasons = _unique_sorted(
                        outcome.reason_codes or ("unspecified_malformed_candidate",)
                    )
                    result.malformed_candidates += 1
                    result.source_contract_malformed_candidates[contract_key] += 1
                    if feature == "Activities":
                        result.activity_items_malformed += 1
                    if feature == "StepCount" and version == "2.0":
                        result.step_count_v2_items_malformed += 1
                    result.schema_warning_features.add(feature)
                    for reason in reasons:
                        result.malformed_reason_counts[reason] += 1
                    malformed = MalformedCandidate(
                        source_file=source.path.name,
                        canonical_month=source.canonical_month,
                        row_number=row_number,
                        outer_id=outer_id,
                        source_feature=feature,
                        collecting_method_version=version,
                        payload_index=payload_index,
                        reason_codes=reasons,
                        item_json=canonical_json(item),
                    )
                    result.malformed_items.append(malformed)
                    add_diagnostic(
                        row_number=row_number,
                        outer_id=outer_id,
                        feature=feature,
                        version=version,
                        payload_index=payload_index,
                        stage="candidate-validation",
                        code="malformed_candidate",
                        message="; ".join(reasons),
                    )
                    continue

                result.valid_candidates += 1
                result.source_contract_valid_candidates[contract_key] += 1
                result.records_by_feature[outcome.native_feature].append(outcome.record)
                result.candidate_lineage_by_feature[outcome.native_feature].append(
                    CandidateLineage(
                        source_occurrence_key=source_occurrence_key_value,
                        outer_id=outer_id,
                        payload_index=payload_index,
                        collecting_method_version=version,
                    )
                )
                result.native_feature_candidates[outcome.native_feature] += 1
                if feature == "Activities":
                    result.activity_items_routed += 1
                    if outcome.native_feature == UNREGISTERED_ACTIVITY_POLICY.native_feature:
                        result.unregistered_activity_items_routed += 1
                        raw_label = str(outcome.record.get("raw_activity_name") or "")
                        result.unknown_activity_labels.add(raw_label)
                    else:
                        result.registered_activity_items_routed += 1
                if feature == "StepCount" and version == "2.0":
                    result.step_count_v2_alternatives_emitted += 1

                if outcome.warning is not None:
                    warning_code, warning_message = outcome.warning
                    add_diagnostic(
                        row_number=row_number,
                        outer_id=outer_id,
                        feature=feature,
                        version=version,
                        payload_index=payload_index,
                        stage="candidate-routing",
                        code=warning_code,
                        message=warning_message,
                        once_key=(warning_code, warning_message),
                    )

    result.records_by_feature = dict(result.records_by_feature)
    return result


def merge_parse_results(results: Sequence[ParseResult]) -> ParseResult:
    """Merge month-level results in caller-provided deterministic order."""

    merged = ParseResult()
    records: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        merged.source_occurrences.extend(result.source_occurrences)
        merged.malformed_items.extend(result.malformed_items)
        merged.diagnostics.extend(result.diagnostics)

        for name in (
            "outer_rows_seen",
            "google_rows_seen",
            "irrelevant_rows_seen",
            "supported_outer_rows",
            "unsupported_outer_rows",
            "payloads_decoded",
            "payload_decode_failures",
            "candidate_items_seen",
            "valid_candidates",
            "malformed_candidates",
            "activity_items_seen",
            "activity_items_routed",
            "registered_activity_items_routed",
            "unregistered_activity_items_routed",
            "activity_items_malformed",
            "step_count_v2_items_seen",
            "step_count_v2_alternatives_emitted",
            "step_count_v2_items_malformed",
        ):
            setattr(merged, name, getattr(merged, name) + getattr(result, name))

        merged.source_contract_outer_rows.update(result.source_contract_outer_rows)
        merged.source_contract_candidate_items.update(result.source_contract_candidate_items)
        merged.source_contract_valid_candidates.update(result.source_contract_valid_candidates)
        merged.source_contract_malformed_candidates.update(result.source_contract_malformed_candidates)
        merged.native_feature_candidates.update(result.native_feature_candidates)
        merged.malformed_reason_counts.update(result.malformed_reason_counts)
        merged.outer_id_counts.update(result.outer_id_counts)
        merged.unknown_activity_labels.update(result.unknown_activity_labels)
        merged.unsupported_contracts.update(result.unsupported_contracts)
        merged.schema_warning_features.update(result.schema_warning_features)
        for feature, feature_records in result.records_by_feature.items():
            records[feature].extend(feature_records)
        for feature, feature_lineage in result.candidate_lineage_by_feature.items():
            merged.candidate_lineage_by_feature[feature].extend(feature_lineage)

    merged.records_by_feature = dict(records)
    merged.candidate_lineage_by_feature = dict(merged.candidate_lineage_by_feature)
    return merged


def parse_participant_directory(
    participant_dir: Path, *, row_error_policy: str = "fail-participant",
) -> tuple[list[SourceFile], ParseResult]:
    """Discover and parse all canonical monthly files for one participant."""

    participant_dir = Path(participant_dir)
    sources = discover_month_files(participant_dir)
    results = [
        parse_month_file(source, participant_dir.name, row_error_policy=row_error_policy,)
        for source in sources
    ]
    merged = merge_parse_results(results)
    merged.assert_accounting_closed()
    return sources, merged
