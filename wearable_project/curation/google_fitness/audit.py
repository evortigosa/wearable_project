"""
Wearable Data Processing and Modeling project
Read-only CWP2 calibration audit for an accepted Google Fitness native root. The audit consumes the immutable
CWP1 policy contracts and the accepted native state database. It never creates or modifies a curated root.
Participant feature CSVs are read in spawned worker processes; compact aggregates are merged deterministically
by participant ID and rendered into calibration reports that can be reviewed before executable curation is frozen.
"""


from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4
from tqdm import tqdm
from wearable_project.curation.google_fitness.audit_models import (
    AUDIT_VERSION, REPORT_SCHEMA_VERSION, NumericSketch, PairSketch, ParticipantAuditResult, flatten_summary,
)
from wearable_project.curation.google_fitness.decisions import DECISIONS, decisions_fingerprint, validate_decisions
from wearable_project.curation.google_fitness.evidence import evidence_fingerprint, validate_evidence
from wearable_project.curation.google_fitness.guidance import guidance_fingerprint, validate_guidance
from wearable_project.curation.google_fitness.models import EventKind
from wearable_project.curation.google_fitness.registry import (
    CURATION_POLICIES, CURATION_REGISTRY_VERSION, get_policy, registry_fingerprint, validate_registry,
)
from wearable_project.curation.google_fitness.rules import RULES, rules_fingerprint, validate_rules
from wearable_project.processing.google_fitness.registry import REGISTRY_VERSION as NATIVE_REGISTRY_VERSION, SOURCE_KEY
from wearable_project.processing.google_fitness.writer import STATE_FILENAME


# Native CSVs are small-field tabular files, but raising the limit prevents the audit from inheriting
# platform-specific defaults.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


REQUIRED_REPORTS = (
    "native_snapshot.json",
    "policy_coverage.csv",
    "unit_contract_matrix.csv",
    "activity_measurement_concordance.csv",
    "activity_quantity_by_label_audit.csv",
    "activity_quantity_stepcount_concordance.csv",
    "anthropometric_cross_feature_audit.csv",
    "stepcount_preference_audit.csv",
    "stepcount_value_raw_audit.csv",
    "stepcount_cross_version_audit.csv",
    "stepcount_temporal_audit.csv",
    "negative_calories_audit.csv",
    "weight_interval_audit.csv",
    "provenance_classification_coverage.csv",
    "source_day_label_audit.csv",
    "sleep_overlap_audit.csv",
    "quality_rule_trigger_audit.csv",
    "calibration_decisions.json",
    "cwp2_acceptance.json",
    "audit_summary.json",
    "REPORT.md",
)


@dataclass(frozen=True, slots=True)
class NativeFileRecord:
    participant_id: str
    feature: str
    filename: str
    row_count: int
    size_bytes: int
    sha256: str
    represented_occurrences: int


@dataclass(frozen=True, slots=True)
class ParticipantAuditInput:
    participant_id: str
    native_root: str
    files: tuple[NativeFileRecord, ...]
    verify_native_hashes: bool


@dataclass(frozen=True, slots=True)
class NativeSnapshot:
    snapshot_id: str
    source_key: str
    metadata: Mapping[str, str]
    participant_status_counts: Mapping[str, int]
    participant_count: int
    participants_with_files: int
    feature_file_count: int
    native_rows: int
    represented_occurrences: int
    output_bytes: int
    source_occurrences: int
    malformed_candidates: int
    parser_diagnostics: int
    reconciliation_decisions: int
    malformed_breakdown: tuple[dict[str, Any], ...]
    feature_totals: tuple[dict[str, Any], ...]
    file_manifest_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "source_key": self.source_key,
            "metadata": dict(self.metadata),
            "participant_status_counts": dict(self.participant_status_counts),
            "participant_count": self.participant_count,
            "participants_with_files": self.participants_with_files,
            "feature_file_count": self.feature_file_count,
            "native_rows": self.native_rows,
            "represented_occurrences": self.represented_occurrences,
            "output_bytes": self.output_bytes,
            "source_occurrences": self.source_occurrences,
            "malformed_candidates": self.malformed_candidates,
            "parser_diagnostics": self.parser_diagnostics,
            "reconciliation_decisions": self.reconciliation_decisions,
            "malformed_breakdown": list(self.malformed_breakdown),
            "feature_totals": list(self.feature_totals),
            "file_manifest_digest": self.file_manifest_digest,
        }


_WEEKDAY_LABELS: dict[str, int] = {
    # English
    "mon": 0, "mon.": 0, "monday": 0,
    "tue": 1, "tue.": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "wed.": 2, "wednesday": 2,
    "thu": 3, "thu.": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "fri.": 4, "friday": 4,
    "sat": 5, "sat.": 5, "saturday": 5,
    "sun": 6, "sun.": 6, "sunday": 6,
    # Hebrew
    "יום ב׳": 0, "יום ג׳": 1, "יום ד׳": 2, "יום ה׳": 3,
    "יום ו׳": 4, "שבת": 5, "יום א׳": 6,
    # Russian
    "пн": 0, "вт": 1, "ср": 2, "чт": 3, "пт": 4, "сб": 5, "вс": 6,
    # Portuguese
    "seg": 0, "seg.": 0, "ter": 1, "ter.": 1, "qua": 2, "qua.": 2,
    "qui": 3, "qui.": 3, "sex": 4, "sex.": 4, "sáb": 5, "sáb.": 5,
    "dom": 6, "dom.": 6,
    # French
    "lun": 0, "lun.": 0, "mar": 1, "mar.": 1, "mer": 2, "mer.": 2,
    "jeu": 3, "jeu.": 3, "ven": 4, "ven.": 4, "sam": 5, "sam.": 5,
    "dim": 6, "dim.": 6,
    # Spanish (shared abbreviations intentionally map to the same weekdays)
    "mié": 2, "mié.": 2, "jue": 3, "jue.": 3, "vie": 4, "vie.": 4,
    # German
    "mo.": 0, "di.": 1, "mi.": 2, "do.": 3, "fr.": 4, "sa.": 5, "so.": 6,
    # Swedish
    "mån": 0, "tis": 1, "ons": 2, "tors": 3, "fre": 4, "lör": 5, "sön": 6,
    # Bulgarian abbreviation observed in Phase 0
    "нд": 6,
}


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    materialized = [dict(row) for row in rows]
    if fieldnames is None:
        names: list[str] = []
        seen: set[str] = set()
        for row in materialized:
            for name in row:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        fieldnames = names
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in materialized:
            writer.writerow({name: _render_csv_value(row.get(name)) for name in fieldnames})


def _render_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (tuple, list, set, frozenset, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_datetime(value: str | None) -> datetime | None:
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


def _parse_date(value: str | None) -> date | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool_status(value: str | None) -> str:
    text = (value or "").strip().casefold()
    if text in {"1", "true", "yes"}:
        return "tracked"
    if text in {"0", "false", "no"}:
        return "untracked"
    return "unknown"


def _is_google_identifier(value: str | None) -> bool:
    text = (value or "").strip().casefold()
    return bool(text) and (
        text.startswith("com.google.")
        or text.startswith("derived:com.google.")
        or "google fitness" in text
        or "google fit" in text
    )


def _activity_provenance(row: Mapping[str, str]) -> tuple[str, str, str, dict[str, bool]]:
    source_id = row.get("source_id", "")
    source_name = row.get("source_name", "")
    device = row.get("device", "")
    tracking = _bool_status(row.get("tracked"))
    if source_id and not _is_google_identifier(source_id):
        acquisition = "third_party_import"
    elif device or source_name:
        acquisition = "device_estimate"
    elif source_id and _is_google_identifier(source_id):
        acquisition = "health_import_unknown"
    else:
        acquisition = "unknown"
    return acquisition, "activity_classification", tracking, {
        "source_id": bool(source_id),
        "source_name": bool(source_name),
        "device": bool(device),
        "tracked": bool((row.get("tracked") or "").strip()),
    }


def _aggregation_kind(value: str | None) -> str:
    """Normalize exporter-qualified StepCount source names for policy logic."""

    text = (value or "").strip()
    folded = text.casefold()
    if "estimated_steps" in folded:
        return "estimated_steps"
    if "merge_step_deltas" in folded:
        return "merge_step_deltas"
    return text or "<missing>"


def _step_provenance(row: Mapping[str, str]) -> tuple[str, str, str, dict[str, bool]]:
    version = row.get("collecting_method_version", "")
    aggregation = _aggregation_kind(row.get("aggregation_source"))
    package = (row.get("app_package_name") or "").strip()
    device_present = any((row.get(name) or "").strip() for name in (
        "device_manufacturer", "device_model", "device_type", "device_uid"
    ))
    if version == "1.0":
        acquisition = "health_import_unknown"
        derivation = "legacy_summary"
    else:
        if package and not _is_google_identifier(package):
            acquisition = "third_party_import"
        elif device_present:
            acquisition = "device_estimate"
        else:
            acquisition = "health_import_unknown"
        if aggregation == "estimated_steps":
            derivation = "estimated"
        elif aggregation == "merge_step_deltas":
            derivation = "merged"
        elif aggregation:
            derivation = "third_party_aggregate"
        else:
            derivation = "unknown"
    return acquisition, derivation, "not_applicable", {
        "app_package_name": bool(package),
        "app_name": bool((row.get("app_name") or "").strip()),
        "origin_data_source_id": bool((row.get("origin_data_source_id") or "").strip()),
        "device": device_present,
    }


def _anthropometric_provenance(row: Mapping[str, str]) -> tuple[str, str, str, dict[str, bool]]:
    added_by = (row.get("added_by") or "").strip()
    if added_by and not _is_google_identifier(added_by):
        acquisition = "third_party_import"
    elif added_by:
        acquisition = "health_import_unknown"
    else:
        acquisition = "unknown"
    return acquisition, "raw", "not_applicable", {"added_by": bool(added_by)}


def _outer_summary_provenance(row: Mapping[str, str]) -> tuple[str, str, str, dict[str, bool]]:
    return "unknown", "application_summary", "not_applicable", {}


def _classify_provenance(feature: str, policy_kind: EventKind, row: Mapping[str, str]) -> tuple[str, str, str, dict[str, bool]]:
    if policy_kind in {EventKind.ACTIVITY_INTERVAL, EventKind.STATE_INTERVAL}:
        return _activity_provenance(row)
    if feature == "StepCount":
        return _step_provenance(row)
    if feature in {"Weight", "Height"}:
        return _anthropometric_provenance(row)
    return _outer_summary_provenance(row)


def _weekday_relation(label: str, start: datetime | None, end: datetime | None) -> tuple[str, str]:
    normalized = label.strip().casefold()
    expected = _WEEKDAY_LABELS.get(normalized)
    if expected is None:
        return "unrecognized", ""
    if start is None and end is None:
        return "recognized_no_timestamp", str(expected)
    start_weekday = start.weekday() if start else None
    end_weekday = end.weekday() if end else None
    if expected == start_weekday and expected == end_weekday:
        return "matches_both", str(expected)
    if expected == start_weekday:
        return "matches_start", str(expected)
    if expected == end_weekday:
        return "matches_end", str(expected)
    adjacent = set()
    for item in (start_weekday, end_weekday):
        if item is not None:
            adjacent.update({(item - 1) % 7, (item + 1) % 7})
    if expected in adjacent:
        return "adjacent", str(expected)
    return "nonadjacent", str(expected)


def _day_delta(left: date | None, right: date | None) -> str:
    if left is None or right is None:
        return "missing"
    delta = (left - right).days
    if delta == 0:
        return "same_day"
    if delta == 1:
        return "plus_one_day"
    if delta == -1:
        return "minus_one_day"
    return "other_day_difference"


def _calorie_bucket(value: float | None) -> str:
    if value is None:
        return "invalid_or_missing"
    if value >= 0:
        return "zero" if value == 0 else "positive"
    magnitude = abs(value)
    if magnitude <= 0.001:
        return "negative_tiny"
    if magnitude <= 1.0:
        return "negative_small"
    if magnitude <= 100.0:
        return "negative_material"
    return "negative_extreme"


def _duration_bucket(seconds: float | None) -> str:
    if seconds is None:
        return "invalid_or_missing"
    if seconds < 0:
        return "negative"
    if seconds == 0:
        return "zero"
    if seconds <= 60:
        return "positive_le_1_minute"
    if seconds <= 3600:
        return "positive_le_1_hour"
    if seconds <= 86400:
        return "positive_le_24_hours"
    return "positive_gt_24_hours"


def _native_state_connection(native_root: Path) -> sqlite3.Connection:
    state = native_root / STATE_FILENAME
    if not state.is_file():
        raise FileNotFoundError(f"Google Fitness native state not found: {state}")
    wal = Path(str(state) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError(f"native state has an active WAL file: {wal}; finish/checkpoint processing before auditing")
    # immutable=1 prevents a read-only audit from creating SQLite -wal/-shm sidecars inside the protected native root.
    connection = sqlite3.connect(f"file:{state.as_posix()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    row = connection.execute("SELECT value FROM metadata WHERE key='source_key'").fetchone()
    if row is None or row["value"] != SOURCE_KEY:
        connection.close()
        raise ValueError(
            f"native state source is {None if row is None else row['value']!r}; expected {SOURCE_KEY!r}"
        )
    return connection


def _load_native_snapshot(native_root: Path) -> tuple[NativeSnapshot, tuple[ParticipantAuditInput, ...]]:
    connection = _native_state_connection(native_root)
    try:
        metadata = {
            row["key"]: row["value"] for row in connection.execute("SELECT key,value FROM metadata ORDER BY key")
        }
        status_counts = {
            row["status"]: int(row["count"])
            for row in connection.execute(
                "SELECT status,COUNT(*) AS count FROM participants GROUP BY status ORDER BY status"
            )
        }
        feature_totals = tuple(
            {
                "feature": row["feature"],
                "files": int(row["files"]),
                "rows": int(row["rows"]),
                "bytes": int(row["bytes"]),
                "represented_occurrences": int(row["represented"]),
            }
            for row in connection.execute(
                """
                SELECT feature, COUNT(*) AS files, SUM(row_count) AS rows,
                       SUM(size_bytes) AS bytes,
                       SUM(represented_occurrences) AS represented
                FROM feature_outputs
                GROUP BY feature
                ORDER BY feature
                """
            )
        )
        file_rows = list(
            connection.execute(
                """
                SELECT participant_id,feature,filename,row_count,size_bytes,sha256,
                       represented_occurrences
                FROM feature_outputs
                ORDER BY participant_id,feature
                """
            )
        )
        file_manifest_hasher = sha256()
        for row in file_rows:
            file_manifest_hasher.update(
                (
                    f"{row['participant_id']}\0{row['feature']}\0{row['filename']}\0"
                    f"{row['row_count']}\0{row['size_bytes']}\0{row['sha256']}\0"
                    f"{row['represented_occurrences']}\n"
                ).encode("utf-8")
            )
        file_manifest_digest = file_manifest_hasher.hexdigest()
        by_participant: dict[str, list[NativeFileRecord]] = defaultdict(list)
        for row in file_rows:
            by_participant[row["participant_id"]].append(
                NativeFileRecord(
                    participant_id=row["participant_id"],
                    feature=row["feature"],
                    filename=row["filename"],
                    row_count=int(row["row_count"]),
                    size_bytes=int(row["size_bytes"]),
                    sha256=row["sha256"],
                    represented_occurrences=int(row["represented_occurrences"]),
                )
            )
        malformed = tuple(
            {
                "source_feature": row["source_feature"],
                "collecting_method_version": row["collecting_method_version"],
                "reason_codes": json.loads(row["reason_codes_json"]),
                "count": int(row["count"]),
            }
            for row in connection.execute(
                """
                SELECT source_feature, collecting_method_version, reason_codes_json,
                       COUNT(*) AS count
                FROM malformed_candidates
                GROUP BY source_feature, collecting_method_version, reason_codes_json
                ORDER BY source_feature, collecting_method_version, reason_codes_json
                """
            )
        )
        source_occurrences = int(connection.execute("SELECT COUNT(*) FROM source_occurrences").fetchone()[0])
        malformed_count = int(connection.execute("SELECT COUNT(*) FROM malformed_candidates").fetchone()[0])
        diagnostics = int(connection.execute("SELECT COUNT(*) FROM parser_diagnostics").fetchone()[0])
        reconciliations = int(connection.execute("SELECT COUNT(*) FROM reconciliation_decisions").fetchone()[0])
        stable_payload = {
            "source_key": SOURCE_KEY,
            "metadata": metadata,
            "participant_status_counts": status_counts,
            "features": feature_totals,
            "source_occurrences": source_occurrences,
            "malformed_breakdown": malformed,
            "parser_diagnostics": diagnostics,
            "reconciliation_decisions": reconciliations,
            "file_manifest_digest": file_manifest_digest,
        }
        snapshot_id = sha256(
            json.dumps(stable_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        snapshot = NativeSnapshot(
            snapshot_id=snapshot_id,
            source_key=SOURCE_KEY,
            metadata=metadata,
            participant_status_counts=status_counts,
            participant_count=sum(status_counts.values()),
            participants_with_files=len(by_participant),
            feature_file_count=len(file_rows),
            native_rows=sum(int(row["row_count"]) for row in file_rows),
            represented_occurrences=sum(int(row["represented_occurrences"]) for row in file_rows),
            output_bytes=sum(int(row["size_bytes"]) for row in file_rows),
            source_occurrences=source_occurrences,
            malformed_candidates=malformed_count,
            parser_diagnostics=diagnostics,
            reconciliation_decisions=reconciliations,
            malformed_breakdown=malformed,
            feature_totals=feature_totals,
            file_manifest_digest=file_manifest_digest,
        )
        inputs = tuple(
            ParticipantAuditInput(
                participant_id=participant,
                native_root=str(native_root),
                files=tuple(files),
                verify_native_hashes=False,
            )
            for participant, files in sorted(by_participant.items())
        )
        return snapshot, inputs
    finally:
        connection.close()


def _validate_file(path: Path, record: NativeFileRecord, verify_hash: bool) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    if stat.st_size != record.size_bytes:
        raise ValueError(f"size mismatch for {path}: state={record.size_bytes}, disk={stat.st_size}")
    if verify_hash:
        observed = _sha256_file(path)
        if observed != record.sha256:
            raise ValueError(f"SHA-256 mismatch for {path}: state={record.sha256}, disk={observed}")


def _record_provenance(
    result: ParticipantAuditResult, feature: str, version: str, policy_kind: EventKind, row: Mapping[str, str],
) -> tuple[str, str, str]:
    acquisition, derivation, tracking, raw_presence = _classify_provenance(feature, policy_kind, row)
    result.provenance_counts[(feature, version, "acquisition_method", acquisition)] += 1
    result.provenance_counts[(feature, version, "derivation_method", derivation)] += 1
    result.provenance_counts[(feature, version, "tracking_status", tracking)] += 1
    for field, present in raw_presence.items():
        result.provenance_counts[(feature, version, f"raw_field:{field}", "present" if present else "missing")] += 1
    return acquisition, derivation, tracking


def _record_source_day(
    result: ParticipantAuditResult, feature: str, version: str, row: Mapping[str, str], start: datetime | None,
    end: datetime | None,
) -> None:
    label = (row.get("source_day_label") or "").strip()
    if not label:
        result.source_day_counts[(feature, version, "", "missing", "")] += 1
        return
    relation, normalized = _weekday_relation(label, start, end)
    result.source_day_counts[(feature, version, label, relation, normalized)] += 1


def _record_unit_stats(
    result: ParticipantAuditResult, feature: str, version: str, row: Mapping[str, str], row_key: str,
) -> None:
    policy = get_policy(feature)
    for measurement in policy.measurements:
        sketch = result.sketch(result.unit_stats, (feature, measurement.column, version))
        sketch.add(row.get(measurement.column), key=f"{row_key}|{measurement.column}")


def _preferred_step_rows(rows: Sequence[Mapping[str, str]]) -> tuple[list[Mapping[str, str]], Counter[str], str]:
    roles: Counter[str] = Counter()
    estimated = [row for row in rows if _aggregation_kind(row.get("aggregation_source")) == "estimated_steps"]
    merged = [row for row in rows if _aggregation_kind(row.get("aggregation_source")) == "merge_step_deltas"]
    preferred: list[Mapping[str, str]] = []
    if len(estimated) == 1:
        preferred = estimated
        roles["fit_app_preferred"] += 1
        roles["parallel_nonpreferred"] += len(rows) - 1
        outcome = "estimated_preferred"
    elif not estimated and len(merged) == 1:
        preferred = merged
        roles["merge_fallback"] += 1
        roles["parallel_nonpreferred"] += len(rows) - 1
        outcome = "merge_fallback"
    elif len(rows) == 1:
        preferred = [rows[0]]
        roles["third_party_fallback"] += 1
        outcome = "sole_fallback"
    else:
        roles["ambiguous_parallel"] += len(rows)
        outcome = "ambiguous_parallel"
    return preferred, roles, outcome


def _audit_stepcount(
    result: ParticipantAuditResult, rows: Sequence[Mapping[str, str]], participant_id: str,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    v1_by_outer_date: dict[str, list[float]] = defaultdict(list)
    v2_by_outer_id: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        version = row.get("collecting_method_version", "")
        if version == "1.0":
            dt = _parse_datetime(row.get("outer_datetime"))
            value = _float(row.get("value"))
            if dt and value is not None:
                v1_by_outer_date[dt.date().isoformat()].append(value)
            result.step_role_counts["legacy_v1"] += 1
            result.quality_rule_counts[("step_count_v1_legacy_context", "StepCount", "1.0")] += 1
            continue
        if version != "2.0":
            result.step_group_counts["unknown_version_rows"] += 1
            continue
        v2_by_outer_id[row.get("outer_id", "")].append(row)
        aggregation = _aggregation_kind(row.get("aggregation_source"))
        value = _float(row.get("value"))
        raw_value = _float(row.get("raw_value"))
        if value is not None and raw_value is not None:
            result.pair_sketch(result.step_value_raw_pairs, aggregation).add(
                value, raw_value,
                key=f"{participant_id}|step_value_raw|{row.get('outer_id')}|{row.get('payload_index')}",
            )
        outer_dt = _parse_datetime(row.get("outer_datetime"))
        summary_date = _parse_date(row.get("summary_date"))
        start = _parse_datetime(row.get("start_date"))
        end = _parse_datetime(row.get("end_date"))
        result.step_temporal_counts[(aggregation, f"summary_vs_outer:{_day_delta(summary_date, outer_dt.date() if outer_dt else None)}")] += 1
        result.step_temporal_counts[(aggregation, f"raw_start_vs_outer:{_day_delta(start.date() if start else None, outer_dt.date() if outer_dt else None)}")] += 1
        result.step_temporal_counts[(aggregation, f"raw_end_vs_outer:{_day_delta(end.date() if end else None, outer_dt.date() if outer_dt else None)}")] += 1
        result.step_temporal_counts[(aggregation, f"summary_vs_raw_start:{_day_delta(summary_date, start.date() if start else None)}")] += 1
        if start and end:
            seconds = (end - start).total_seconds()
            result.sketch(result.step_temporal_duration, aggregation).add(
                seconds, key=f"{participant_id}|step|{row.get('outer_id')}|{row.get('payload_index')}|duration",
            )
            result.step_temporal_counts[(aggregation, "raw_interval_valid" if seconds >= 0 else "raw_interval_negative")] += 1
        else:
            result.step_temporal_counts[(aggregation, "raw_interval_missing")]+=1

    preferred_by_outer_date: dict[str, list[float]] = defaultdict(list)
    preferred_by_summary_date: dict[str, list[float]] = defaultdict(list)
    for outer_id, group in sorted(v2_by_outer_id.items()):
        composition_counter = Counter(_aggregation_kind(row.get("aggregation_source")) for row in group)
        composition = tuple(sorted(composition_counter.items()))
        result.step_preference_compositions[composition] += 1
        invalid_group_structure = (
            not outer_id
            or "<missing>" in composition_counter
            or any(count > 1 for count in composition_counter.values())
        )
        if invalid_group_structure:
            result.quality_rule_counts[("step_count_v2_group_structure", "StepCount", "2.0")] += len(group)
            result.step_group_counts["groups_with_invalid_structure"] += 1
        result.step_group_counts["v2_groups"] += 1
        result.step_group_counts[f"group_size:{len(group)}"] += 1
        values = [_float(row.get("value")) for row in group]
        finite = [value for value in values if value is not None]
        if len(finite) == len(group) and len(set(finite)) > 1:
            result.step_group_counts["groups_with_disagreement"] += 1
            result.quality_rule_counts[("step_count_alternatives_disagree", "StepCount", "2.0")] += len(group)
        elif len(finite) == len(group):
            result.step_group_counts["groups_all_values_equal"] += 1
        else:
            result.step_group_counts["groups_with_invalid_values"] += 1
            if not invalid_group_structure:
                result.quality_rule_counts[("step_count_v2_group_structure", "StepCount", "2.0")] += len(group)
        preferred, roles, outcome = _preferred_step_rows(group)
        result.step_role_counts.update(roles)
        result.step_group_counts[f"preference_outcome:{outcome}"] += 1
        if outcome == "ambiguous_parallel":
            result.quality_rule_counts[("step_count_ambiguous_parallel", "StepCount", "2.0")] += len(group)
        elif outcome == "estimated_preferred":
            result.quality_rule_counts[("step_count_estimated_preferred", "StepCount", "2.0")] += 1
            result.quality_rule_counts[("step_count_parallel_nonpreferred", "StepCount", "2.0")] += max(0, len(group) - 1)
        elif outcome == "merge_fallback":
            result.quality_rule_counts[("step_count_merge_fallback", "StepCount", "2.0")] += 1
            result.quality_rule_counts[("step_count_parallel_nonpreferred", "StepCount", "2.0")] += max(0, len(group) - 1)
        elif outcome == "sole_fallback":
            result.quality_rule_counts[("step_count_sole_fallback", "StepCount", "2.0")] += 1
        for row in preferred:
            value = _float(row.get("value"))
            outer = _parse_datetime(row.get("outer_datetime"))
            summary = _parse_date(row.get("summary_date"))
            if value is not None and outer:
                preferred_by_outer_date[outer.date().isoformat()].append(value)
            if value is not None and summary:
                preferred_by_summary_date[summary.isoformat()].append(value)

    has_v1 = bool(v1_by_outer_date)
    has_v2 = bool(v2_by_outer_id)
    membership = "both" if has_v1 and has_v2 else "v1_only" if has_v1 else "v2_only" if has_v2 else "none"
    result.step_cross_membership[membership] += 1
    if has_v1 and has_v2:
        result.quality_rule_counts[("step_count_cross_version_separate", "StepCount", "1.0")] += sum(len(values) for values in v1_by_outer_date.values())
        result.quality_rule_counts[("step_count_cross_version_separate", "StepCount", "2.0")] += sum(len(group) for group in v2_by_outer_id.values())
        outer_shared = set(v1_by_outer_date) & set(preferred_by_outer_date)
        summary_shared = set(v1_by_outer_date) & set(preferred_by_summary_date)
        result.step_cross_membership["participants_with_outer_date_overlap" if outer_shared else "participants_without_outer_date_overlap"] += 1
        result.step_cross_membership["participants_with_summary_date_overlap" if summary_shared else "participants_without_summary_date_overlap"] += 1
        for basis, shared, right in (
            ("outer_date", outer_shared, preferred_by_outer_date),
            ("summary_date", summary_shared, preferred_by_summary_date),
        ):
            for day in sorted(shared):
                left_values = v1_by_outer_date[day]
                right_values = right[day]
                if len(left_values) == 1 and len(right_values) == 1:
                    result.pair_sketch(result.step_cross_pairs, basis).add(
                        left_values[0], right_values[0], key=f"{participant_id}|{basis}|{day}"
                    )
                    result.step_cross_membership[f"{basis}:comparable_dates"] += 1
                else:
                    result.step_cross_membership[f"{basis}:ambiguous_multiplicity_dates"] += 1
    return preferred_by_outer_date, preferred_by_summary_date


def _audit_sleep_overlaps(
    result: ParticipantAuditResult, rows: Sequence[Mapping[str, str]], participant_id: str,
) -> None:
    intervals: list[tuple[datetime, datetime, str, tuple[str, ...], int]] = []
    for index, row in enumerate(rows):
        start = _parse_datetime(row.get("start_date"))
        end = _parse_datetime(row.get("end_date"))
        if not start or not end or end <= start:
            continue
        provenance = tuple((row.get(name) or "") for name in (
            "collecting_method_version", "source_id", "source_name", "device"
        ))
        intervals.append((start, end, row.get("value", ""), provenance, index))
    intervals.sort(key=lambda item: (item[0], item[1], item[2], item[4]))
    active: list[tuple[datetime, datetime, str, tuple[str, ...], int]] = []
    affected_rows: dict[tuple[str, str], set[int]] = defaultdict(set)
    detailed = {"DEEP", "LIGHT", "REM"}
    sleep_states = {"SLEEP", "DEEP", "LIGHT", "REM"}
    for current in intervals:
        start, end, state, provenance, index = current
        active = [item for item in active if item[1] > start]
        for other in active:
            overlap = min(end, other[1]) - max(start, other[0])
            seconds = overlap.total_seconds()
            if seconds <= 0:
                continue
            scope = "same_provenance" if provenance == other[3] else "cross_provenance"
            other_state = other[2]
            if state == other_state:
                category = "same_state"
            elif "AWAKE" in {state, other_state} and ({state, other_state} & sleep_states):
                category = "awake_sleep"
            elif state in detailed and other_state in detailed:
                category = "detailed_state_conflict"
            elif "SLEEP" in {state, other_state}:
                category = "generic_detail_context"
            else:
                category = "other_overlap"
            result.sleep_overlap_counts[(scope, category)] += 1
            affected_rows[(scope, category)].update((other[4], index))
            if scope == "same_provenance":
                sleep_rule = {
                    "same_state": "sleep_same_state_overlap",
                    "awake_sleep": "sleep_awake_overlap",
                    "detailed_state_conflict": "sleep_detailed_state_overlap",
                    "generic_detail_context": "sleep_generic_detail_overlap_context",
                }.get(category)
                if sleep_rule:
                    result.quality_rule_counts[(sleep_rule, "Sleep", "1.0")] += 2
            result.sketch(result.sleep_overlap_duration, (scope, category)).add(
                seconds,
                key=f"{participant_id}|sleep|{other[4]}|{index}",
            )
        active.append(current)
    for key, indexes in affected_rows.items():
        result.sleep_overlap_rows_affected[key] += len(indexes)


def _audit_participant(task: ParticipantAuditInput) -> ParticipantAuditResult:
    result = ParticipantAuditResult(task.participant_id)
    root = Path(task.native_root)
    activity_daily: dict[str, dict[str, float]] = {
        "calories": defaultdict(float),
        "distance": defaultdict(float),
    }
    activity_quantity_daily: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    outer_daily: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    step_rows: list[dict[str, str]] = []
    sleep_rows: list[dict[str, str]] = []
    weight_observations: list[tuple[datetime, float, str]] = []
    height_observations: list[tuple[datetime, float, str]] = []

    for record in task.files:
        path = root / record.participant_id / record.filename
        _validate_file(path, record, task.verify_native_hashes)
        policy = get_policy(record.feature)
        active_rule_ids = frozenset(policy.curation.rule_ids)
        expected_header = tuple(policy.native_schema.native_columns)
        result.files_read += 1
        result.bytes_read += record.size_bytes
        result.feature_files[record.feature] += 1
        row_count = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            observed_header = tuple(reader.fieldnames or ())
            if observed_header != expected_header:
                result.schema_issues.append(
                    f"{record.participant_id}/{record.filename}: header {observed_header!r} != {expected_header!r}"
                )
                continue
            for row_index, row in enumerate(reader):
                row_count += 1
                result.rows_read += 1
                version = (row.get("collecting_method_version") or "").strip()
                result.feature_rows[record.feature] += 1
                result.feature_versions[(record.feature, version)] += 1
                row_key = f"{record.participant_id}|{record.feature}|{row_index}"
                _record_unit_stats(result, record.feature, version, row, row_key)
                acquisition, derivation, tracking = _record_provenance(
                    result, record.feature, version, policy.event_kind, row
                )

                start = _parse_datetime(row.get("start_date"))
                end = _parse_datetime(row.get("end_date"))
                invalid_native_interval = False
                if policy.temporal.start_column and policy.temporal.end_column:
                    invalid_native_interval = start is None or end is None or end < start
                    if "native_interval_valid" in active_rule_ids and invalid_native_interval:
                        result.quality_rule_counts[("native_interval_valid", record.feature, version)] += 1
                    if (
                        "activity_interval_valid" in active_rule_ids
                        and policy.event_kind in {EventKind.ACTIVITY_INTERVAL, EventKind.STATE_INTERVAL}
                        and (invalid_native_interval or (start is not None and end is not None and end <= start))
                    ):
                        result.quality_rule_counts[("activity_interval_valid", record.feature, version)] += 1
                if "finite_measurement" in active_rule_ids:
                    for measurement in policy.measurements:
                        raw_measurement = row.get(measurement.column)
                        if raw_measurement not in (None, "") and _float(raw_measurement) is None:
                            result.quality_rule_counts[("finite_measurement", record.feature, version)] += 1
                            break
                if "source_day_label" in row:
                    source_day_label = (row.get("source_day_label") or "").strip()
                    if source_day_label and "source_day_label_context" in active_rule_ids:
                        result.quality_rule_counts[("source_day_label_context", record.feature, version)] += 1
                    relation, _ = _weekday_relation(source_day_label, start, end)
                    if relation == "unrecognized" and "source_day_label_unrecognized" in active_rule_ids:
                        result.quality_rule_counts[("source_day_label_unrecognized", record.feature, version)] += 1
                    elif relation == "nonadjacent" and "source_day_label_nonadjacent_disagreement" in active_rule_ids:
                        result.quality_rule_counts[("source_day_label_nonadjacent_disagreement", record.feature, version)] += 1
                    _record_source_day(result, record.feature, version, row, start, end)

                if policy.event_kind in {EventKind.ACTIVITY_INTERVAL, EventKind.STATE_INTERVAL}:
                    if "activity_tracking_status" in active_rule_ids:
                        result.quality_rule_counts[("activity_tracking_status", record.feature, version)] += 1
                    if "activity_provenance_classification" in active_rule_ids:
                        result.quality_rule_counts[("activity_provenance_classification", record.feature, version)] += 1
                    if acquisition == "unknown" and "activity_provenance_unknown" in active_rule_ids:
                        result.quality_rule_counts[("activity_provenance_unknown", record.feature, version)] += 1
                    if (
                        "activity_quantity_unresolved" in active_rule_ids
                        and "quantity" in row
                        and (row.get("quantity") or "").strip()
                    ):
                        result.quality_rule_counts[("activity_quantity_unresolved", record.feature, version)] += 1
                    if start and end and start.date() != end.date():
                        result.activity_cross_midnight[record.feature] += 1
                    day = start.date().isoformat() if start else None
                    activity_values = [_float(row.get(name)) for name in ("calories", "distance", "quantity") if name in row]
                    if (
                        "activity_measurements_nonnegative" in active_rule_ids
                        and any(value is not None and value < 0 for value in activity_values)
                    ):
                        result.quality_rule_counts[("activity_measurements_nonnegative", record.feature, version)] += 1
                    for measurement in ("calories", "distance"):
                        value = _float(row.get(measurement))
                        if day and value is not None:
                            activity_daily[measurement][day] += value
                    if "quantity" in row:
                        quantity = result.sketch(result.activity_quantity, (record.feature, version)).add(
                            row.get("quantity"), key=f"{row_key}|quantity"
                        )
                        distance = _float(row.get("distance"))
                        calories = _float(row.get("calories"))
                        if quantity is not None and distance is not None:
                            result.pair_sketch(result.activity_quantity_distance, (record.feature, version)).add(
                                quantity, distance, key=f"{row_key}|quantity_distance"
                            )
                        if quantity is not None and calories is not None:
                            result.pair_sketch(result.activity_quantity_calories, (record.feature, version)).add(
                                quantity, calories, key=f"{row_key}|quantity_calories"
                            )
                        if day and quantity is not None:
                            activity_quantity_daily[record.feature][day] += quantity
                    if record.feature == "Sleep":
                        if "sleep_no_apple_equivalence" in active_rule_ids:
                            result.quality_rule_counts[("sleep_no_apple_equivalence", record.feature, version)] += 1
                        value = (row.get("value") or "").strip()
                        if value not in policy.native_schema.state_vocabulary and "sleep_state_vocabulary" in active_rule_ids:
                            result.quality_rule_counts[("sleep_state_vocabulary", record.feature, version)] += 1
                        sleep_rows.append(dict(row))

                if (
                    "unit_mapping_pending" in active_rule_ids
                    and record.feature in {"Calories", "Distance", "Weight", "Height"}
                    and _float(row.get("value")) is not None
                ):
                    result.quality_rule_counts[("unit_mapping_pending", record.feature, version)] += 1
                if (
                    "anthropometric_added_by_context" in active_rule_ids
                    and record.feature in {"Weight", "Height"}
                    and (row.get("added_by") or "").strip()
                ):
                    result.quality_rule_counts[("anthropometric_added_by_context", record.feature, version)] += 1

                if record.feature in {"Calories", "Distance"}:
                    day = start.date().isoformat() if start else None
                    value = _float(row.get("value"))
                    if day and value is not None:
                        outer_daily[(record.feature.casefold(), version)][day] += value

                if record.feature == "Calories":
                    value = _float(row.get("value"))
                    bucket = _calorie_bucket(value)
                    result.negative_calories[(version, bucket)] += 1
                    bucket_rule = {
                        "negative_tiny": "calories_negative_tiny",
                        "negative_small": "calories_negative_small",
                        "negative_material": "calories_negative_material",
                        "negative_extreme": "calories_negative_extreme",
                    }.get(bucket)
                    if bucket_rule and bucket_rule in active_rule_ids:
                        result.quality_rule_counts[(bucket_rule, record.feature, version)] += 1
                    result.sketch(result.negative_calories_values, (version, bucket)).add(
                        row.get("value"), key=f"{row_key}|calories"
                    )

                if record.feature == "Weight":
                    seconds = (end - start).total_seconds() if start and end else None
                    bucket = _duration_bucket(seconds)
                    result.weight_counts[(version, bucket)] += 1
                    if (
                        "weight_nonpoint_interval" in active_rule_ids
                        and seconds is not None
                        and 0 < seconds <= 86400
                    ):
                        result.quality_rule_counts[("weight_nonpoint_interval", record.feature, version)] += 1
                    elif (
                        "weight_long_interval" in active_rule_ids
                        and seconds is not None
                        and seconds > 86400
                    ):
                        result.quality_rule_counts[("weight_long_interval", record.feature, version)] += 1
                    result.sketch(result.weight_duration, version).add(
                        seconds, key=f"{row_key}|weight_duration"
                    )
                    value = result.sketch(result.weight_values, version).add(
                        row.get("value"), key=f"{row_key}|weight_value"
                    )
                    if value is not None:
                        below = value < 0
                        above = value > 1000
                        result.weight_counts[(version, "value_below_source_range")] += int(below)
                        result.weight_counts[(version, "value_above_source_range")] += int(above)
                        if (below or above) and "weight_source_range" in active_rule_ids:
                            result.quality_rule_counts[("weight_source_range", record.feature, version)] += 1
                    added_by = (row.get("added_by") or "").strip() or "<missing>"
                    result.weight_added_by[(version, added_by)] += 1
                    anchor = end or start
                    if anchor is not None and value is not None:
                        weight_observations.append((anchor, value, row_key))

                if record.feature == "Height":
                    height_value = _float(row.get("value"))
                    anchor = end or start
                    if anchor is not None and height_value is not None:
                        height_observations.append((anchor, height_value, row_key))

                if record.feature == "Distance":
                    value = _float(row.get("value"))
                    seconds = (end - start).total_seconds() if start and end else None
                    if value is not None and value < 0 and "distance_nonnegative" in active_rule_ids:
                        result.quality_rule_counts[("distance_nonnegative", record.feature, version)] += 1
                    if (
                        "distance_positive_zero_duration" in active_rule_ids
                        and value is not None
                        and value > 0
                        and seconds == 0
                    ):
                        result.quality_rule_counts[("distance_positive_zero_duration", record.feature, version)] += 1
                    if (
                        "distance_source_rate_limit" in active_rule_ids
                        and value is not None
                        and seconds is not None
                        and seconds > 0
                        and value / seconds > 100
                    ):
                        result.quality_rule_counts[("distance_source_rate_limit", record.feature, version)] += 1
                if record.feature == "Height":
                    value = _float(row.get("value"))
                    if (
                        "height_source_range" in active_rule_ids
                        and value is not None
                        and (value < 0 or value > 3)
                    ):
                        result.quality_rule_counts[("height_source_range", record.feature, version)] += 1
                if record.feature == "StepCount":
                    value = _float(row.get("value"))
                    if version == "2.0":
                        if "step_count_temporal_context" in active_rule_ids:
                            result.quality_rule_counts[("step_count_temporal_context", record.feature, version)] += 1
                        if "unit_payload_explicit" in active_rule_ids:
                            result.quality_rule_counts[("unit_payload_explicit", record.feature, version)] += 1
                    if value is not None and value < 0 and "step_count_nonnegative" in active_rule_ids:
                        result.quality_rule_counts[("step_count_nonnegative", record.feature, version)] += 1
                    if version == "2.0" and start and end and value is not None:
                        seconds = (end - start).total_seconds()
                        if (
                            "step_count_source_rate_limit" in active_rule_ids
                            and seconds > 0
                            and value / seconds > 10
                        ):
                            result.quality_rule_counts[("step_count_source_rate_limit", record.feature, version)] += 1
                    step_rows.append(dict(row))
        if row_count != record.row_count:
            result.schema_issues.append(
                f"{record.participant_id}/{record.filename}: state rows={record.row_count}, observed={row_count}"
            )

    for measurement, activity_values in activity_daily.items():
        for outer_version in ("1.0", "2.0"):
            outer_values = outer_daily[(measurement, outer_version)]
            activity_days = set(activity_values)
            outer_days = set(outer_values)
            shared = activity_days & outer_days
            coverage_key = (measurement, outer_version)
            result.activity_concordance_coverage[(measurement, outer_version, "activity_days")] += len(activity_days)
            result.activity_concordance_coverage[(measurement, outer_version, "outer_days")] += len(outer_days)
            result.activity_concordance_coverage[(measurement, outer_version, "shared_days")] += len(shared)
            result.activity_concordance_coverage[(measurement, outer_version, "activity_only_days")] += len(activity_days - outer_days)
            result.activity_concordance_coverage[(measurement, outer_version, "outer_only_days")] += len(outer_days - activity_days)
            sketch = result.pair_sketch(result.activity_concordance, coverage_key)
            for day in sorted(shared):
                sketch.add(
                    activity_values[day],
                    outer_values[day],
                    key=f"{task.participant_id}|{measurement}|{outer_version}|{day}",
                )

    preferred_step_outer: dict[str, list[float]] = {}
    preferred_step_summary: dict[str, list[float]] = {}
    if step_rows:
        preferred_step_outer, preferred_step_summary = _audit_stepcount(
            result, step_rows, task.participant_id
        )
    for feature, quantities in sorted(activity_quantity_daily.items()):
        for basis, step_values in (
            ("outer_date", preferred_step_outer),
            ("summary_date", preferred_step_summary),
        ):
            quantity_days = set(quantities)
            step_days = set(step_values)
            shared = quantity_days & step_days
            result.activity_quantity_stepcount_coverage[(feature, basis, "quantity_days")] += len(quantity_days)
            result.activity_quantity_stepcount_coverage[(feature, basis, "step_days")] += len(step_days)
            result.activity_quantity_stepcount_coverage[(feature, basis, "shared_days")] += len(shared)
            for day in sorted(shared):
                candidates = step_values[day]
                if len(candidates) == 1:
                    result.pair_sketch(
                        result.activity_quantity_stepcount, (feature, basis)
                    ).add(
                        quantities[day], candidates[0],
                        key=f"{task.participant_id}|quantity_stepcount|{feature}|{basis}|{day}",
                    )
                    result.activity_quantity_stepcount_coverage[(feature, basis, "comparable_days")] += 1
                else:
                    result.activity_quantity_stepcount_coverage[(feature, basis, "ambiguous_step_multiplicity_days")] += 1

    if height_observations:
        result.anthropometric_counts["participants_with_height"] += 1
        result.anthropometric_counts["height_observations"] += len(height_observations)
    if weight_observations:
        result.anthropometric_counts["participants_with_weight"] += 1
        result.anthropometric_counts["weight_observations"] += len(weight_observations)
    if height_observations and weight_observations:
        result.anthropometric_counts["participants_with_both"] += 1
        ordered_weights = sorted(weight_observations, key=lambda item: (item[0], item[1], item[2]))
        for height_time, height_value, height_key in height_observations:
            nearest = min(
                ordered_weights,
                key=lambda item: (abs((item[0] - height_time).total_seconds()), item[0], item[1], item[2]),
            )
            weight_time, weight_value, _ = nearest
            delta_days = abs((weight_time - height_time).total_seconds()) / 86400.0
            result.anthropometric_counts["height_observations_with_nearest_weight"] += 1
            if delta_days <= 30:
                result.anthropometric_counts["nearest_weight_within_30_days"] += 1
            if delta_days <= 365:
                result.anthropometric_counts["nearest_weight_within_365_days"] += 1
            result.anthropometric_time_delta_days.add(
                delta_days, key=f"{task.participant_id}|anthropometric|{height_key}|days"
            )
            if height_value > 0 and weight_value >= 0:
                bmi = weight_value / (height_value * height_value)
                result.anthropometric_bmi.add(
                    bmi, key=f"{task.participant_id}|anthropometric|{height_key}|bmi"
                )
                result.anthropometric_counts["bmi_assumption_rows"] += 1
                if 10 <= bmi <= 80:
                    result.anthropometric_counts["bmi_assumption_plausible_10_80"] += 1
                else:
                    result.anthropometric_counts["bmi_assumption_outside_10_80"] += 1

    if sleep_rows:
        _audit_sleep_overlaps(result, sleep_rows, task.participant_id)
    return result


def _replace_verify(tasks: Sequence[ParticipantAuditInput], verify: bool) -> tuple[ParticipantAuditInput, ...]:
    return tuple(
        ParticipantAuditInput(
            participant_id=task.participant_id,
            native_root=task.native_root,
            files=task.files,
            verify_native_hashes=verify,
        )
        for task in tasks
    )


def _run_workers(
    tasks: Sequence[ParticipantAuditInput], *, workers: int, max_in_flight: int, progress: bool,
) -> tuple[ParticipantAuditResult, list[dict[str, str]]]:
    aggregate = ParticipantAuditResult("__cohort__")
    failures: list[dict[str, str]] = []
    if not tasks:
        return aggregate, failures
    workers = max(1, min(workers, len(tasks)))
    max_in_flight = max(workers, max_in_flight)
    task_by_id = {task.participant_id: task for task in tasks}
    ordered_ids = sorted(task_by_id)
    next_submit = 0
    next_reduce = 0
    buffered: dict[str, ParticipantAuditResult] = {}
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        future_to_id: dict[Any, str] = {}
        bar = tqdm(
            total=len(tasks),
            desc="Auditing Google Fitness native participants",
            unit="participant",
            disable=not progress,
        )
        try:
            while next_submit < len(ordered_ids) or future_to_id:
                while (
                    next_submit < len(ordered_ids)
                    and len(future_to_id) + len(buffered) < max_in_flight
                ):
                    participant_id = ordered_ids[next_submit]
                    future = pool.submit(_audit_participant, task_by_id[participant_id])
                    future_to_id[future] = participant_id
                    next_submit += 1
                done, _ = wait(future_to_id, return_when=FIRST_COMPLETED)
                for future in done:
                    participant_id = future_to_id.pop(future)
                    try:
                        buffered[participant_id] = future.result()
                    except Exception as exc:  # participant-isolated evidence collection
                        failures.append(
                            {
                                "participant_id": participant_id,
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            }
                        )
                    bar.update(1)
                    bar.set_postfix(failed=len(failures), workers=workers)
                while next_reduce < len(ordered_ids):
                    participant_id = ordered_ids[next_reduce]
                    if participant_id in buffered:
                        aggregate.merge(buffered.pop(participant_id))
                        next_reduce += 1
                    elif any(item["participant_id"] == participant_id for item in failures):
                        next_reduce += 1
                    else:
                        break
        finally:
            bar.close()
    return aggregate, sorted(failures, key=lambda item: item["participant_id"])


def _policy_coverage_rows(snapshot: NativeSnapshot, aggregate: ParticipantAuditResult) -> list[dict[str, Any]]:
    state_by_feature = {item["feature"]: item for item in snapshot.feature_totals}
    issue_features = Counter()
    for issue in aggregate.schema_issues:
        for feature in CURATION_POLICIES:
            if f"/{feature}.csv" in issue:
                issue_features[feature] += 1
                break
    rows = []
    for feature, policy in CURATION_POLICIES.items():
        state = state_by_feature.get(feature, {})
        versions = sorted(
            version for observed_feature, version in aggregate.feature_versions if observed_feature == feature
        )
        rows.append(
            {
                "feature": feature,
                "event_kind": policy.event_kind.value,
                "policy_maturity": policy.calibration.maturity.value,
                "execution_mode": policy.calibration.execution_mode.value,
                "native_files": int(state.get("files", 0)),
                "native_rows": int(state.get("rows", 0)),
                "audited_files": aggregate.feature_files.get(feature, 0),
                "audited_rows": aggregate.feature_rows.get(feature, 0),
                "observed_versions": ";".join(versions),
                "expected_versions": ";".join(policy.native_schema.supported_versions),
                "schema_issues": issue_features.get(feature, 0),
                "required_audits": ";".join(policy.calibration.required_audits),
                "decision_ids": ";".join(policy.calibration.decision_ids),
                "policy_present": 1,
            }
        )
    return rows


def _unit_contract_rows(aggregate: ParticipantAuditResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature, policy in CURATION_POLICIES.items():
        for measurement in policy.measurements:
            for unit_policy in measurement.unit_by_version:
                for version in unit_policy.versions:
                    sketch = aggregate.unit_stats.get((feature, measurement.column, version), NumericSketch())
                    row = {
                        "feature": feature,
                        "measurement": measurement.column,
                        "collecting_method_version": version,
                        "measurement_kind": measurement.kind.value,
                        "required": measurement.required,
                        "raw_projection_enabled": measurement.raw_projection_enabled,
                        "canonical_projection_enabled": measurement.canonical_projection_enabled,
                        "raw_unit_status": unit_policy.raw_unit_status.value,
                        "raw_unit": unit_policy.raw_unit,
                        "proposed_canonical_unit": unit_policy.canonical_unit,
                        "evidence_grade": unit_policy.evidence_grade.value,
                        "execution_mode": unit_policy.execution_mode.value,
                        "canonicalization": unit_policy.canonicalization.value,
                        "evidence_refs": ";".join(unit_policy.evidence_refs),
                        "limitations": " | ".join(unit_policy.limitations),
                    }
                    row.update(flatten_summary("observed", sketch.summary()))
                    rows.append(row)
    return rows


def _activity_concordance_rows(aggregate: ParticipantAuditResult) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(aggregate.activity_concordance) | {
        ("calories", "1.0"), ("calories", "2.0"), ("distance", "1.0"), ("distance", "2.0")
    }):
        measurement, version = key
        sketch = aggregate.activity_concordance.get(key, PairSketch())
        row = {
            "activity_measurement": measurement,
            "outer_feature": measurement.title(),
            "outer_collecting_method_version": version,
            "alignment_basis": "participant_utc_start_date",
            "activity_days": aggregate.activity_concordance_coverage[(measurement, version, "activity_days")],
            "outer_days": aggregate.activity_concordance_coverage[(measurement, version, "outer_days")],
            "shared_days": aggregate.activity_concordance_coverage[(measurement, version, "shared_days")],
            "activity_only_days": aggregate.activity_concordance_coverage[(measurement, version, "activity_only_days")],
            "outer_only_days": aggregate.activity_concordance_coverage[(measurement, version, "outer_only_days")],
            "activity_rows_crossing_utc_midnight": sum(aggregate.activity_cross_midnight.values()),
            "interpretation": "cohort_concordance_only_not_exporter_mapping_proof",
        }
        row.update(flatten_summary("comparison", sketch.summary()))
        rows.append(row)
    return rows


def _activity_quantity_rows(aggregate: ParticipantAuditResult) -> list[dict[str, Any]]:
    rows = []
    for feature, policy in CURATION_POLICIES.items():
        if "quantity" not in policy.measurement_columns:
            continue
        for version in policy.native_schema.supported_versions:
            sketch = aggregate.activity_quantity.get((feature, version), NumericSketch())
            row = {
                "feature": feature,
                "collecting_method_version": version,
                "unit_status": "unresolved",
                "canonical_projection_enabled": 0,
                "interpretation": "activity_dependent_source_value",
            }
            row.update(flatten_summary("quantity", sketch.summary()))
            row.update(flatten_summary(
                "quantity_vs_distance",
                aggregate.activity_quantity_distance.get((feature, version), PairSketch()).summary(),
            ))
            row.update(flatten_summary(
                "quantity_vs_calories",
                aggregate.activity_quantity_calories.get((feature, version), PairSketch()).summary(),
            ))
            rows.append(row)
    return rows


def _activity_quantity_stepcount_rows(aggregate: ParticipantAuditResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    activity_features = [
        feature for feature, policy in CURATION_POLICIES.items() if "quantity" in policy.measurement_columns
    ]
    for feature in activity_features:
        for basis in ("outer_date", "summary_date"):
            row = {
                "feature": feature,
                "step_count_date_basis": basis,
                "alignment_basis": "participant_utc_start_date",
                "quantity_days": aggregate.activity_quantity_stepcount_coverage[(feature, basis, "quantity_days")],
                "step_days": aggregate.activity_quantity_stepcount_coverage[(feature, basis, "step_days")],
                "shared_days": aggregate.activity_quantity_stepcount_coverage[(feature, basis, "shared_days")],
                "comparable_days": aggregate.activity_quantity_stepcount_coverage[(feature, basis, "comparable_days")],
                "ambiguous_step_multiplicity_days": aggregate.activity_quantity_stepcount_coverage[(feature, basis, "ambiguous_step_multiplicity_days")],
                "interpretation": "exploratory_only_quantity_semantics_unresolved",
            }
            row.update(flatten_summary(
                "comparison",
                aggregate.activity_quantity_stepcount.get((feature, basis), PairSketch()).summary(),
            ))
            rows.append(row)
    return rows


def _anthropometric_rows(aggregate: ParticipantAuditResult) -> list[dict[str, Any]]:
    row = {
        "interpretation": "exploratory_identity_unit_assumption_not_exporter_mapping_proof",
        **dict(sorted(aggregate.anthropometric_counts.items())),
    }
    row.update(flatten_summary("nearest_time_delta_days", aggregate.anthropometric_time_delta_days.summary()))
    row.update(flatten_summary("bmi_assuming_kg_per_m2", aggregate.anthropometric_bmi.summary()))
    return [row]


def _step_value_raw_rows(aggregate: ParticipantAuditResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sorted(aggregate.step_value_raw_pairs):
        row = {
            "aggregation_source": source,
            "interpretation": "source_concepts_kept_separate_even_when_equal",
        }
        row.update(flatten_summary("comparison", aggregate.step_value_raw_pairs[source].summary()))
        rows.append(row)
    return rows


def _step_preference_rows(aggregate: ParticipantAuditResult) -> list[dict[str, Any]]:
    rows = []
    for composition, count in sorted(aggregate.step_preference_compositions.items(), key=lambda item: repr(item[0])):
        names = {name: value for name, value in composition}
        estimated = names.get("estimated_steps", 0)
        merged = names.get("merge_step_deltas", 0)
        size = sum(names.values())
        if estimated == 1:
            outcome = "estimated_preferred"
        elif estimated == 0 and merged == 1:
            outcome = "merge_fallback"
        elif size == 1:
            outcome = "sole_fallback"
        else:
            outcome = "ambiguous_parallel"
        rows.append(
            {
                "composition": json.dumps(names, sort_keys=True, separators=(",", ":")),
                "group_size": size,
                "group_count": count,
                "estimated_steps_count": estimated,
                "merge_step_deltas_count": merged,
                "third_party_count": size - estimated - merged,
                "preference_outcome": outcome,
            }
        )
    for role, count in sorted(aggregate.step_role_counts.items()):
        rows.append(
            {
                "composition": "__role_total__",
                "group_size": "",
                "group_count": count,
                "estimated_steps_count": "",
                "merge_step_deltas_count": "",
                "third_party_count": "",
                "preference_outcome": role,
            }
        )
    for metric, count in sorted(aggregate.step_group_counts.items()):
        rows.append(
            {
                "composition": "__metric__",
                "group_size": "",
                "group_count": count,
                "estimated_steps_count": "",
                "merge_step_deltas_count": "",
                "third_party_count": "",
                "preference_outcome": metric,
            }
        )
    return rows


def _step_cross_version_rows(aggregate: ParticipantAuditResult) -> list[dict[str, Any]]:
    rows = [
        {"section": "participant_membership", "metric": metric, "count": count}
        for metric, count in sorted(aggregate.step_cross_membership.items())
    ]
    for basis in ("outer_date", "summary_date"):
        sketch = aggregate.step_cross_pairs.get(basis, PairSketch())
        row = {"section": "value_comparison", "metric": basis, "count": sketch.pairs}
        row.update(flatten_summary("comparison", sketch.summary()))
        rows.append(row)
    return rows


def _step_temporal_rows(aggregate: ParticipantAuditResult) -> list[dict[str, Any]]:
    rows = []
    sources = sorted({source for source, _ in aggregate.step_temporal_counts} | set(aggregate.step_temporal_duration))
    for source in sources:
        metrics = {
            metric: count
            for (observed_source, metric), count in aggregate.step_temporal_counts.items()
            if observed_source == source
        }
        row = {"aggregation_source": source}
        row.update(metrics)
        row.update(flatten_summary(
            "raw_interval_duration_seconds",
            aggregate.step_temporal_duration.get(source, NumericSketch()).summary(),
        ))
        rows.append(row)
    return rows


def _negative_calories_rows(aggregate: ParticipantAuditResult) -> list[dict[str, Any]]:
    rows = []
    versions = sorted({version for version, _ in aggregate.negative_calories})
    order = (
        "positive", "zero", "negative_tiny", "negative_small",
        "negative_material", "negative_extreme", "invalid_or_missing",
    )
    for version in versions:
        for bucket in order:
            count = aggregate.negative_calories[(version, bucket)]
            sketch = aggregate.negative_calories_values.get((version, bucket), NumericSketch())
            row = {
                "collecting_method_version": version,
                "bucket": bucket,
                "count": count,
                "proposed_status_effect": "pending_human_review",
                "native_value_action": "preserve_unchanged",
            }
            row.update(flatten_summary("value", sketch.summary()))
            rows.append(row)
    return rows


def _weight_interval_rows(aggregate: ParticipantAuditResult) -> list[dict[str, Any]]:
    rows = []
    versions = sorted({version for version, _ in aggregate.weight_counts} | set(aggregate.weight_duration))
    for version in versions:
        row = {
            "collecting_method_version": version,
            "distinct_added_by": len({value for observed_version, value in aggregate.weight_added_by if observed_version == version}),
            "added_by_missing": aggregate.weight_added_by[(version, "<missing>")],
            "value_below_source_range": aggregate.weight_counts[(version, "value_below_source_range")],
            "value_above_source_range": aggregate.weight_counts[(version, "value_above_source_range")],
        }
        for bucket in (
            "negative", "zero", "positive_le_1_minute", "positive_le_1_hour",
            "positive_le_24_hours", "positive_gt_24_hours", "invalid_or_missing",
        ):
            row[f"duration_{bucket}"] = aggregate.weight_counts[(version, bucket)]
        row.update(flatten_summary("duration_seconds", aggregate.weight_duration.get(version, NumericSketch()).summary()))
        row.update(flatten_summary("value", aggregate.weight_values.get(version, NumericSketch()).summary()))
        rows.append(row)
    return rows


def _provenance_rows(aggregate: ParticipantAuditResult) -> list[dict[str, Any]]:
    totals: Counter[tuple[str, str, str]] = Counter()
    for (feature, version, dimension, category), count in aggregate.provenance_counts.items():
        totals[(feature, version, dimension)] += count
    rows = []
    for (feature, version, dimension, category), count in sorted(aggregate.provenance_counts.items()):
        total = totals[(feature, version, dimension)]
        rows.append(
            {
                "feature": feature,
                "collecting_method_version": version,
                "dimension": dimension,
                "category": category,
                "count": count,
                "fraction": count / total if total else None,
                "classifier_status": "provisional",
            }
        )
    return rows


def _source_day_rows(aggregate: ParticipantAuditResult) -> list[dict[str, Any]]:
    rows = []
    for (feature, version, label, relation, normalized), count in sorted(aggregate.source_day_counts.items()):
        rows.append(
            {
                "feature": feature,
                "collecting_method_version": version,
                "source_day_label": label,
                "normalized_weekday_index": normalized,
                "relation_to_native_boundaries": relation,
                "count": count,
                "canonical_date_allowed": 0,
            }
        )
    return rows


def _sleep_overlap_rows(aggregate: ParticipantAuditResult) -> list[dict[str, Any]]:
    rows = []
    keys = sorted(set(aggregate.sleep_overlap_counts) | set(aggregate.sleep_overlap_duration))
    for scope, category in keys:
        row = {
            "overlap_scope": scope,
            "overlap_category": category,
            "pair_count": aggregate.sleep_overlap_counts[(scope, category)],
            "rows_affected": aggregate.sleep_overlap_rows_affected[(scope, category)],
            "resolution": "preserve_all_intervals",
        }
        row.update(flatten_summary(
            "overlap_duration_seconds",
            aggregate.sleep_overlap_duration.get((scope, category), NumericSketch()).summary(),
        ))
        rows.append(row)
    return rows


def _rule_count_semantics(rule_id: str) -> str:
    if rule_id.startswith("sleep_") and rule_id.endswith("overlap"):
        return "overlap_pair_row_incidence"
    if rule_id == "sleep_generic_detail_overlap_context":
        return "overlap_pair_row_incidence"
    if rule_id.startswith("step_count_") and rule_id in {
        "step_count_estimated_preferred",
        "step_count_merge_fallback",
        "step_count_sole_fallback",
        "step_count_parallel_nonpreferred",
        "step_count_ambiguous_parallel",
        "step_count_alternatives_disagree",
        "step_count_v2_group_structure",
    }:
        return "rows_in_version2_groups"
    if rule_id in {
        "activity_quantity_unresolved",
        "activity_tracking_status",
        "activity_provenance_classification",
        "sleep_no_apple_equivalence",
        "source_day_label_context",
        "unit_mapping_pending",
        "anthropometric_added_by_context",
        "step_count_temporal_context",
        "step_count_cross_version_separate",
        "unit_payload_explicit",
        "step_count_v1_legacy_context",
    }:
        return "rows_with_policy_context"
    return "rows_matching_condition"


def _quality_rule_rows(aggregate: ParticipantAuditResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature, policy in CURATION_POLICIES.items():
        versions = policy.native_schema.supported_versions or ("",)
        for rule_id in policy.curation.rule_ids:
            definition = RULES[rule_id]
            for version in versions:
                rows.append(
                    {
                        "rule_id": rule_id,
                        "feature": feature,
                        "collecting_method_version": version,
                        "trigger_count": aggregate.quality_rule_counts[(rule_id, feature, version)],
                        "count_semantics": _rule_count_semantics(rule_id),
                        "calibration_only": 1,
                        "rule_class": definition.rule_class.value,
                        "severity": definition.severity.value,
                        "status_effect": definition.status_effect.value,
                        "inclusion_effect": definition.inclusion_effect.value,
                        "flag": definition.flag,
                        "parameters": dict(definition.parameters),
                    }
                )
    # Defensive path: if any implementation bug records a rule that is not attached to the feature policy, keep it
    # visible so the acceptance gate can fail loudly.
    represented = {(row["rule_id"], row["feature"], row["collecting_method_version"]) for row in rows}
    for key, count in sorted(aggregate.quality_rule_counts.items()):
        if key in represented:
            continue
        rule_id, feature, version = key
        definition = RULES[rule_id]
        rows.append(
            {
                "rule_id": rule_id,
                "feature": feature,
                "collecting_method_version": version,
                "trigger_count": count,
                "count_semantics": _rule_count_semantics(rule_id),
                "calibration_only": 1,
                "rule_class": definition.rule_class.value,
                "severity": definition.severity.value,
                "status_effect": definition.status_effect.value,
                "inclusion_effect": definition.inclusion_effect.value,
                "flag": definition.flag,
                "parameters": dict(definition.parameters),
            }
        )
    return sorted(rows, key=lambda row: (row["rule_id"], row["feature"], row["collecting_method_version"]))


def _decision_findings(decision_id: str, aggregate: ParticipantAuditResult) -> dict[str, Any]:
    if decision_id == "negative_calories_buckets":
        return {
            "counts_by_version_and_bucket": {
                f"{version}|{bucket}": count
                for (version, bucket), count in sorted(aggregate.negative_calories.items())
            }
        }
    if decision_id in {"weight_nonpoint_preserved", "weight_unit_mapping"}:
        return {
            "duration_counts": {
                f"{version}|{bucket}": count
                for (version, bucket), count in sorted(aggregate.weight_counts.items())
                if bucket.startswith("positive") or bucket in {"zero", "negative", "invalid_or_missing"}
            },
            "value_stats": {
                version: sketch.summary()
                for version, sketch in sorted(aggregate.weight_values.items())
            },
            "weight_height_assumption_check": {
                "counts": dict(sorted(aggregate.anthropometric_counts.items())),
                "bmi_assuming_kg_per_m2": aggregate.anthropometric_bmi.summary(),
                "nearest_time_delta_days": aggregate.anthropometric_time_delta_days.summary(),
            },
        }
    if decision_id == "step_count_preference":
        return {
            "group_metrics": dict(sorted(aggregate.step_group_counts.items())),
            "role_counts": dict(sorted(aggregate.step_role_counts.items())),
        }
    if decision_id in {"step_count_preserve_all_rows", "step_count_no_canonical_day", "step_count_unit_mapping"}:
        return {
            "cross_version_membership": dict(sorted(aggregate.step_cross_membership.items())),
            "temporal_counts": {
                f"{source}|{metric}": count
                for (source, metric), count in sorted(aggregate.step_temporal_counts.items())
            },
            "value_raw_comparison": {
                source: sketch.summary()
                for source, sketch in sorted(aggregate.step_value_raw_pairs.items())
            },
        }
    if decision_id in {"provenance_classifier", "provenance_dimensions"}:
        acquisition = Counter()
        derivation = Counter()
        tracking = Counter()
        for (_, _, dimension, category), count in aggregate.provenance_counts.items():
            if dimension == "acquisition_method":
                acquisition[category] += count
            elif dimension == "derivation_method":
                derivation[category] += count
            elif dimension == "tracking_status":
                tracking[category] += count
        return {
            "acquisition_method": dict(sorted(acquisition.items())),
            "derivation_method": dict(sorted(derivation.items())),
            "tracking_status": dict(sorted(tracking.items())),
        }
    if decision_id in {"calories_unit_mapping", "distance_unit_mapping"}:
        measurement = "calories" if decision_id.startswith("calories") else "distance"
        return {
            "activity_to_outer_comparison": {
                version: aggregate.activity_concordance.get((measurement, version), PairSketch()).summary()
                for version in ("1.0", "2.0")
            }
        }
    if decision_id == "height_unit_mapping":
        return {
            "height_value_stats": {
                version: aggregate.unit_stats.get(("Height", "value", version), NumericSketch()).summary()
                for version in ("2.0",)
            },
            "weight_height_assumption_check": {
                "counts": dict(sorted(aggregate.anthropometric_counts.items())),
                "bmi_assuming_kg_per_m2": aggregate.anthropometric_bmi.summary(),
                "nearest_time_delta_days": aggregate.anthropometric_time_delta_days.summary(),
            },
        }
    if decision_id == "activity_quantity_unresolved":
        return {
            "features_with_quantity": len(aggregate.activity_quantity),
            "rows_with_finite_quantity": sum(sketch.finite for sketch in aggregate.activity_quantity.values()),
        }
    if decision_id == "activity_multimeasurement_rows":
        return {
            "activity_rows": sum(
                count
                for feature, count in aggregate.feature_rows.items()
                if get_policy(feature).event_kind in {EventKind.ACTIVITY_INTERVAL, EventKind.STATE_INTERVAL}
            ),
            "cross_midnight_rows": sum(aggregate.activity_cross_midnight.values()),
        }
    if decision_id == "source_day_label_context_only":
        relations = Counter()
        for (_, _, _, relation, _), count in aggregate.source_day_counts.items():
            relations[relation] += count
        return {"relations": dict(sorted(relations.items()))}
    if decision_id == "sleep_native_states_preserved":
        return {
            "sleep_rows": aggregate.feature_rows.get("Sleep", 0),
            "overlap_counts": {
                f"{scope}|{category}": count
                for (scope, category), count in sorted(aggregate.sleep_overlap_counts.items())
            },
        }
    if decision_id == "unknown_feature_fallback":
        observed = set(aggregate.feature_rows)
        return {
            "unknown_observed_features": sorted(observed - set(CURATION_POLICIES)),
            "registered_observed_features": len(observed & set(CURATION_POLICIES)),
        }
    return {}


def _decision_payload(
    snapshot: NativeSnapshot, aggregate: ParticipantAuditResult, *, full_native_coverage: bool,
) -> dict[str, Any]:
    generated_reports = {
        "unit_contract_matrix",
        "activity_measurement_concordance",
        "activity_quantity_by_label_audit",
        "activity_quantity_stepcount_concordance",
        "anthropometric_cross_feature_audit",
        "stepcount_preference_audit",
        "stepcount_value_raw_audit",
        "stepcount_cross_version_audit",
        "stepcount_temporal_audit",
        "negative_calories_audit",
        "weight_interval_audit",
        "provenance_classification_coverage",
        "source_day_label_audit",
        "sleep_overlap_audit",
        "policy_coverage",
        "quality_rule_trigger_audit",
    }
    items = []
    for decision_id, decision in DECISIONS.items():
        missing = sorted(set(decision.required_audits) - generated_reports)
        if missing:
            recommendation = "audit_incomplete"
            rationale = f"Missing required audit outputs: {', '.join(missing)}"
        elif not full_native_coverage and decision.status.value != "accepted":
            recommendation = "subset_evidence_only"
            rationale = "The audit covered a participant subset; provisional decisions require a complete native-root run."
        elif decision.status.value == "accepted":
            recommendation = "retain_accepted"
            rationale = "Required audit evidence is present; CWP2 does not reopen an accepted non-destructive decision."
        elif decision_id in {"calories_unit_mapping", "distance_unit_mapping", "weight_unit_mapping", "height_unit_mapping"}:
            recommendation = "remain_provisional"
            rationale = "Native cohort concordance cannot by itself prove the exporter query-to-Google-type mapping."
        elif decision_id == "provenance_classifier":
            recommendation = "human_review_required"
            rationale = "Coverage is measured, but raw package and stream identifiers do not uniquely prove acquisition method."
        else:
            recommendation = "ready_for_human_freeze_review"
            rationale = "The required full-native calibration report is complete; a reviewer must approve final status and inclusion effects."
        items.append(
            {
                "decision_id": decision_id,
                "current_status": decision.status.value,
                "required_audits": list(decision.required_audits),
                "missing_audits": missing,
                "recommendation": recommendation,
                "rationale": rationale,
                "selected_option": decision.selected_option,
                "affected_features": list(decision.affected_features),
                "automated_findings": _decision_findings(decision_id, aggregate),
            }
        )
    return {
        "tool": "google_fitness_curation_calibration",
        "audit_version": AUDIT_VERSION,
        "snapshot_id": snapshot.snapshot_id,
        "audit_scope": "full_native_root" if full_native_coverage else "participant_subset",
        "policy_fingerprints": {
            "registry": registry_fingerprint(),
            "evidence": evidence_fingerprint(),
            "rules": rules_fingerprint(),
            "decisions": decisions_fingerprint(),
            "guidance": guidance_fingerprint(),
        },
        "decisions": items,
        "structural_findings": {
            "schema_issue_count": len(aggregate.schema_issues),
            "native_rows_audited": aggregate.rows_read,
            "native_files_audited": aggregate.files_read,
            "native_snapshot_rows": snapshot.native_rows,
            "native_snapshot_files": snapshot.feature_file_count,
        },
    }


def _policy_catalog_issues() -> tuple[str, ...]:
    issues: list[str] = []
    registry_validation = validate_registry()
    issues.extend(f"registry: {item}" for item in registry_validation.issues)
    issues.extend(f"evidence: {item}" for item in validate_evidence())
    issues.extend(f"rules: {item}" for item in validate_rules())
    issues.extend(f"decisions: {item}" for item in validate_decisions())
    issues.extend(f"guidance: {item}" for item in validate_guidance())
    return tuple(sorted(set(issues)))


def _version_contract_issues(aggregate: ParticipantAuditResult) -> tuple[str, ...]:
    issues: list[str] = []
    for (feature, version), count in sorted(aggregate.feature_versions.items()):
        policy = CURATION_POLICIES.get(feature)
        if policy is None:
            issues.append(f"unregistered observed feature {feature!r} ({count} rows)")
        elif version not in policy.native_schema.supported_versions:
            issues.append(
                f"{feature}: unsupported collecting_method_version {version!r} ({count} rows)"
            )
    return tuple(issues)


def _quality_rule_policy_alignment_issues(aggregate: ParticipantAuditResult,) -> tuple[str, ...]:
    """Return nonzero rule counts that are not registered for their feature."""

    issues: list[str] = []
    for (rule_id, feature, version), count in sorted(aggregate.quality_rule_counts.items()):
        if count <= 0:
            continue
        policy = CURATION_POLICIES.get(feature)
        if policy is None:
            issues.append(
                f"{feature}|{version}: observed {count} trigger(s) for {rule_id!r} without a registered policy"
            )
        elif rule_id not in policy.curation.rule_ids:
            issues.append(
                f"{feature}|{version}: observed {count} trigger(s) for unregistered rule {rule_id!r}"
            )
    return tuple(issues)


def _acceptance_payload(
    snapshot: NativeSnapshot, aggregate: ParticipantAuditResult, failures: Sequence[Mapping[str, str]],
    output_dir: Path, *, expected_files: int, expected_rows: int, full_native_coverage: bool,
) -> dict[str, Any]:
    policy_issues = _policy_catalog_issues()
    version_issues = _version_contract_issues(aggregate)
    quality_rule_policy_alignment_issues = _quality_rule_policy_alignment_issues(aggregate)
    feature_totals_rows = sum(int(item["rows"]) for item in snapshot.feature_totals)
    feature_totals_files = sum(int(item["files"]) for item in snapshot.feature_totals)
    checks = [
        ("source_key", snapshot.source_key == SOURCE_KEY, snapshot.source_key),
        (
            "native_registry_version",
            snapshot.metadata.get("registry_version") == NATIVE_REGISTRY_VERSION,
            f"{snapshot.metadata.get('registry_version')!r}/{NATIVE_REGISTRY_VERSION!r}",
        ),
        ("policy_catalog_integrity", not policy_issues, f"{len(policy_issues)} issue(s)"),
        ("participant_workers", not failures, f"{len(failures)} failure(s)"),
        ("schema_validation", not aggregate.schema_issues, f"{len(aggregate.schema_issues)} issue(s)"),
        ("version_contracts", not version_issues, f"{len(version_issues)} issue(s)"),
        (
            "quality_rule_policy_alignment",
            not quality_rule_policy_alignment_issues,
            f"{len(quality_rule_policy_alignment_issues)} issue(s)",
        ),
        (
            "state_feature_file_accounting",
            feature_totals_files == snapshot.feature_file_count,
            f"{feature_totals_files}/{snapshot.feature_file_count}",
        ),
        (
            "state_feature_row_accounting",
            feature_totals_rows == snapshot.native_rows,
            f"{feature_totals_rows}/{snapshot.native_rows}",
        ),
        ("file_count", aggregate.files_read == expected_files, f"{aggregate.files_read}/{expected_files}"),
        ("row_count", aggregate.rows_read == expected_rows, f"{aggregate.rows_read}/{expected_rows}"),
        ("full_native_coverage", full_native_coverage, "complete native root" if full_native_coverage else "participant subset"),
        ("policy_coverage", set(item["feature"] for item in snapshot.feature_totals) <= set(CURATION_POLICIES), "all observed features registered"),
        ("required_reports", all((output_dir / name).is_file() for name in REQUIRED_REPORTS if name not in {"cwp2_acceptance.json", "REPORT.md"}), "required report files present"),
    ]
    records = [
        {"check": name, "passed": bool(passed), "detail": detail}
        for name, passed, detail in checks
    ]
    failed = [item for item in records if not item["passed"]]
    structural_failed = [item for item in failed if item["check"] != "full_native_coverage"]
    status = "fail" if structural_failed else "pass" if full_native_coverage else "partial_pass"
    return {
        "status": status,
        "scope": "full_native_root" if full_native_coverage else "participant_subset",
        "full_native_coverage": full_native_coverage,
        "check_count": len(records),
        "passed_check_count": len(records) - len(failed),
        "failed_check_count": len(failed),
        "checks": records,
        "policy_catalog_issues": list(policy_issues),
        "version_contract_issues": list(version_issues),
        "quality_rule_policy_alignment_issues": list(quality_rule_policy_alignment_issues),
    }


def _render_report_md(
    snapshot: NativeSnapshot, aggregate: ParticipantAuditResult, failures: Sequence[Mapping[str, str]],
    acceptance: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# Google Fitness CWP2 native calibration audit",
            "",
            f"- Audit version: `{AUDIT_VERSION}`",
            f"- Native snapshot: `{snapshot.snapshot_id}`",
            f"- Acceptance: **{str(acceptance['status']).upper()}**",
            f"- Audit scope: `{acceptance['scope']}`",
            f"- Native participants with files: {snapshot.participants_with_files:,}",
            f"- Native feature files: {snapshot.feature_file_count:,}",
            f"- Native rows: {snapshot.native_rows:,}",
            f"- Audited files: {aggregate.files_read:,}",
            f"- Audited rows: {aggregate.rows_read:,}",
            f"- Worker failures: {len(failures):,}",
            f"- Schema issues: {len(aggregate.schema_issues):,}",
            "",
            "The audit is read-only. It writes no curated root and changes no native file.",
            "Snapshot-specific counts are stored in `native_snapshot.json`, not as timeless policy constants.",
            "",
            "## Calibration reports",
            "",
            *[f"- `{name}`" for name in REQUIRED_REPORTS if name not in {"REPORT.md", "cwp2_acceptance.json"}],
            "",
            "## Interpretation boundary",
            "",
            "The reports measure the accepted native product and support human review of provisional policies. ",
            "They do not prove exporter mappings that are absent from the payload, and they do not modify CWP1 decisions automatically.",
            "",
        ]
    )


def _write_reports(
    stage: Path, snapshot: NativeSnapshot, aggregate: ParticipantAuditResult, failures: Sequence[Mapping[str, str]],
    *, operational: Mapping[str, Any], expected_files: int, expected_rows: int, full_native_coverage: bool,
) -> dict[str, Any]:
    _json_dump(stage / "native_snapshot.json", snapshot.as_dict())
    _csv_write(stage / "policy_coverage.csv", _policy_coverage_rows(snapshot, aggregate))
    _csv_write(stage / "unit_contract_matrix.csv", _unit_contract_rows(aggregate))
    _csv_write(stage / "activity_measurement_concordance.csv", _activity_concordance_rows(aggregate))
    _csv_write(stage / "activity_quantity_by_label_audit.csv", _activity_quantity_rows(aggregate))
    _csv_write(stage / "activity_quantity_stepcount_concordance.csv", _activity_quantity_stepcount_rows(aggregate))
    _csv_write(stage / "anthropometric_cross_feature_audit.csv", _anthropometric_rows(aggregate))
    _csv_write(stage / "stepcount_preference_audit.csv", _step_preference_rows(aggregate))
    _csv_write(stage / "stepcount_value_raw_audit.csv", _step_value_raw_rows(aggregate))
    _csv_write(stage / "stepcount_cross_version_audit.csv", _step_cross_version_rows(aggregate))
    _csv_write(stage / "stepcount_temporal_audit.csv", _step_temporal_rows(aggregate))
    _csv_write(stage / "negative_calories_audit.csv", _negative_calories_rows(aggregate))
    _csv_write(stage / "weight_interval_audit.csv", _weight_interval_rows(aggregate))
    _csv_write(stage / "provenance_classification_coverage.csv", _provenance_rows(aggregate))
    _csv_write(stage / "source_day_label_audit.csv", _source_day_rows(aggregate))
    _csv_write(stage / "quality_rule_trigger_audit.csv", _quality_rule_rows(aggregate))
    _csv_write(
        stage / "sleep_overlap_audit.csv",
        _sleep_overlap_rows(aggregate),
        fieldnames=(
            "overlap_scope", "overlap_category", "pair_count", "rows_affected", "resolution",
            "overlap_duration_seconds_seen", "overlap_duration_seconds_missing",
            "overlap_duration_seconds_nonnumeric", "overlap_duration_seconds_nonfinite",
            "overlap_duration_seconds_finite", "overlap_duration_seconds_negative",
            "overlap_duration_seconds_zero", "overlap_duration_seconds_positive",
            "overlap_duration_seconds_minimum", "overlap_duration_seconds_maximum",
            "overlap_duration_seconds_mean", "overlap_duration_seconds_std",
            "overlap_duration_seconds_p01", "overlap_duration_seconds_p05",
            "overlap_duration_seconds_p25", "overlap_duration_seconds_p50",
            "overlap_duration_seconds_p75", "overlap_duration_seconds_p95",
            "overlap_duration_seconds_p99", "overlap_duration_seconds_sample_size",
        ),
    )
    _json_dump(
        stage / "calibration_decisions.json",
        _decision_payload(snapshot, aggregate, full_native_coverage=full_native_coverage),
    )
    _json_dump(stage / "participant_failures.json", list(failures))
    _json_dump(stage / "schema_issues.json", sorted(set(aggregate.schema_issues)))
    summary = {
        "tool": "google_fitness_curation_calibration",
        "audit_version": AUDIT_VERSION,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "snapshot_id": snapshot.snapshot_id,
        "native_files": snapshot.feature_file_count,
        "native_rows": snapshot.native_rows,
        "audit_scope": "full_native_root" if full_native_coverage else "participant_subset",
        "expected_audited_files": expected_files,
        "expected_audited_rows": expected_rows,
        "audited_files": aggregate.files_read,
        "audited_rows": aggregate.rows_read,
        "participants_with_files": snapshot.participants_with_files,
        "worker_failures": len(failures),
        "schema_issues": len(aggregate.schema_issues),
        "policy_count": len(CURATION_POLICIES),
        "curation_registry_version": CURATION_REGISTRY_VERSION,
        "policy_fingerprints": {
            "registry": registry_fingerprint(),
            "evidence": evidence_fingerprint(),
            "rules": rules_fingerprint(),
            "decisions": decisions_fingerprint(),
            "guidance": guidance_fingerprint(),
        },
        "malformed_native_source_items": snapshot.malformed_candidates,
        "operational": dict(operational),
    }
    _json_dump(stage / "audit_summary.json", summary)
    # Acceptance requires all other files, so write it after the scientific reports.
    acceptance = _acceptance_payload(
        snapshot, aggregate, failures, stage,
        expected_files=expected_files,
        expected_rows=expected_rows,
        full_native_coverage=full_native_coverage,
    )
    _json_dump(stage / "cwp2_acceptance.json", acceptance)
    (stage / "REPORT.md").write_text(
        _render_report_md(snapshot, aggregate, failures, acceptance), encoding="utf-8"
    )
    return {**summary, "acceptance": acceptance}


def _safe_output(native_root: Path, output: Path) -> None:
    native_root = native_root.resolve()
    output = output.resolve()
    if output == native_root or native_root in output.parents or output in native_root.parents:
        raise ValueError(f"audit output {output} must be separate from native root {native_root}")


def run_audit(
    native_root: Path, output: Path, *, workers: int = 4, max_in_flight: int | None = None,
    participants: set[str] | None = None, verify_native_hashes: bool = False, overwrite: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    native_root = Path(native_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not native_root.is_dir():
        raise FileNotFoundError(native_root)
    _safe_output(native_root, output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"audit output already exists: {output}")
    snapshot, all_tasks = _load_native_snapshot(native_root)
    full_native_coverage = participants is None
    unmatched_participants: tuple[str, ...] = ()
    if participants is None:
        tasks = all_tasks
    else:
        available = {task.participant_id for task in all_tasks}
        unmatched_participants = tuple(sorted(participants - available))
        tasks = tuple(task for task in all_tasks if task.participant_id in participants)
        if not tasks:
            raise ValueError("participant filter selected no participants with native feature files")
    tasks = _replace_verify(tasks, verify_native_hashes)
    expected_files = sum(len(task.files) for task in tasks)
    expected_rows = sum(record.row_count for task in tasks for record in task.files)
    effective_workers = max(1, min(int(workers), len(tasks) or 1))
    effective_in_flight = max(effective_workers, int(max_in_flight or effective_workers))
    stage = output.parent / f".{output.name}.stage-{uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        aggregate, failures = _run_workers(
            tasks,
            workers=effective_workers,
            max_in_flight=effective_in_flight,
            progress=progress,
        )
        operational = {
            "workers_requested": workers,
            "workers_effective": effective_workers,
            "max_in_flight": effective_in_flight,
            "verify_native_hashes": verify_native_hashes,
            "participant_filter_applied": participants is not None,
            "participants_requested": None if participants is None else len(participants),
            "participants_selected": len(tasks),
            "participants_unmatched": list(unmatched_participants),
            "audit_scope": "full_native_root" if full_native_coverage else "participant_subset",
            "expected_audited_files": expected_files,
            "expected_audited_rows": expected_rows,
            "elapsed_seconds": time.perf_counter() - started,
        }
        summary = _write_reports(
            stage, snapshot, aggregate, failures, operational=operational,
            expected_files=expected_files, expected_rows=expected_rows,
            full_native_coverage=full_native_coverage,
        )
        if output.exists():
            shutil.rmtree(output)
        os.replace(stage, output)
        return {**summary, "output": str(output)}
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _participants_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the read-only Google Fitness CWP2 native calibration audit.")
    parser.add_argument("--input-native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--participants-file", type=Path)
    parser.add_argument("--verify-native-hashes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--json-summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_audit(
            args.input_native,
            args.output,
            workers=args.workers,
            max_in_flight=args.max_in_flight,
            participants=_participants_file(args.participants_file),
            verify_native_hashes=args.verify_native_hashes,
            overwrite=args.overwrite,
            progress=not args.no_progress,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json_summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        acceptance = summary["acceptance"]
        print(
            f"Google Fitness CWP2 audit {acceptance['status'].upper()}: "
            f"{summary['audited_rows']:,} rows in {summary['audited_files']:,} files; "
            f"reports written to {summary['output']}"
        )
    return 0 if summary["acceptance"]["status"] in {"pass", "partial_pass"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
