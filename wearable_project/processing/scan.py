"""
Wearable Data Processing and Modeling project
Read-only planning scan for native cumulative HealthKit processing. The scanner evaluates the exact
participant-level planner used by ``process`` without creating an output directory, mutating
``.wearable_state.sqlite``, or writing participant feature files.  It is intended to answer a concrete
operational question before an incremental run:
    What would be skipped, incrementally updated, rebuilt, or blocked if the native processor were executed
    now with these options?
"""


from __future__ import annotations
import hashlib
import json
import os
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from tqdm import tqdm
from wearable_project import __version__
from wearable_project.exceptions import InputLayoutError
from wearable_project.processing.parser import SourceFile, canonical_month_from_name, discover_month_files
from wearable_project.processing.environment import processing_environment_manifest
from wearable_project.processing.pipeline import (
    PARSER_VERSION, PlanAction, discover_participants, plan_participant, resolve_export_root,
)
from wearable_project.processing.registry import REGISTRY_VERSION
from wearable_project.processing.writer import (
    FeatureOutputInfo, StoredParticipantState, existing_output_is_usable, source_signature,
)


SCAN_VERSION = "native-process-scan-2"
NATIVE_STATE_FILENAME = ".wearable_state.sqlite"
CURATION_STATE_FILENAME = ".wearable_curation_state.sqlite"
_REQUIRED_STATE_TABLES = {"participants", "source_files", "feature_outputs"}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceDiscoveryResult:
    participant_id: str
    participant_dir: Path
    sources: tuple[SourceFile, ...] = ()
    candidate_files: int = 0
    candidate_bytes: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ParticipantScan:
    participant_id: str
    planned_action: str
    reason: str
    reason_code: str
    current_state: str | None
    current_source_signature: str | None
    committed_source_signature: str | None
    source_files_discovered: int
    source_bytes_discovered: int
    source_files_to_parse: int
    source_bytes_to_parse: int
    committed_source_files: int
    added_months: tuple[str, ...] = ()
    later_added_months: tuple[str, ...] = ()
    recovered_historical_months: tuple[str, ...] = ()
    changed_months: tuple[str, ...] = ()
    missing_months: tuple[str, ...] = ()
    unchanged_months: tuple[str, ...] = ()
    existing_output_usable: bool | None = None
    stored_parser_version: str | None = None
    stored_registry_version: str | None = None
    parser_version_changed: bool = False
    registry_version_changed: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProcessingScanReport:
    scan_id: str
    started_at: str
    finished_at: str
    wall_clock_seconds: float
    status: str
    input_root: str
    export_root: str
    output_root: str
    state_file: str
    state_file_exists: bool
    scan_version: str
    package_version: str
    parser_version: str
    registry_version: str
    mode: str
    snapshot_policy: str
    verify_existing_hashes: bool
    workers_requested: int
    workers_effective: int
    participants_requested: int | None
    participants_missing_from_selection: tuple[str, ...]
    participant_counts: dict[str, int]
    source_file_counts: dict[str, int]
    source_byte_counts: dict[str, int]
    reason_counts: dict[str, int]
    processing_needed: bool
    safe_to_process: bool
    no_op: bool
    plan_fingerprint: str
    state_versions: dict[str, Any]
    processing_environment: dict[str, Any]
    warnings: tuple[str, ...] = ()
    participant_details: list[ParticipantScan] = field(default_factory=list)

    def as_dict(self, *, include_details: bool = False) -> dict[str, Any]:
        payload = {
            "scan_id": self.scan_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_clock_seconds": self.wall_clock_seconds,
            "status": self.status,
            "paths": {
                "input_root": self.input_root,
                "export_root": self.export_root,
                "output_root": self.output_root,
                "state_file": self.state_file,
                "state_file_exists": self.state_file_exists,
            },
            "versions": {
                "scan": self.scan_version,
                "package": self.package_version,
                "parser": self.parser_version,
                "registry": self.registry_version,
            },
            "configuration": {
                "mode": self.mode,
                "snapshot_policy": self.snapshot_policy,
                "verify_existing_hashes": self.verify_existing_hashes,
                "workers_requested": self.workers_requested,
                "workers_effective": self.workers_effective,
                "participants_requested": self.participants_requested,
            },
            "participants_missing_from_selection": list(self.participants_missing_from_selection),
            "participants": dict(self.participant_counts),
            "source_files": dict(self.source_file_counts),
            "source_bytes": dict(self.source_byte_counts),
            "reasons": dict(self.reason_counts),
            "processing_needed": self.processing_needed,
            "safe_to_process": self.safe_to_process,
            "no_op": self.no_op,
            "plan_fingerprint": self.plan_fingerprint,
            "state_versions": dict(self.state_versions),
            "processing_environment": dict(self.processing_environment),
            "warnings": list(self.warnings),
        }
        if include_details:
            payload["participant_details"] = [item.as_dict() for item in self.participant_details]
        return payload


class ReadOnlyNativeState:
    """Read processing state without creating or mutating SQLite."""

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.connection: sqlite3.Connection | None = None
        self.exists = self.path.is_file()
        self.feature_has_represented_occurrences = False

    def __enter__(self) -> "ReadOnlyNativeState":
        if not self.exists:
            return self
        uri = self.path.as_uri() + "?mode=ro"
        try:
            self.connection = sqlite3.connect(uri, uri=True)
        except sqlite3.Error as exc:
            raise InputLayoutError(f"Could not open native processing state read-only: {self.path}: {exc}") from exc
        self.connection.row_factory = sqlite3.Row
        tables = {
            row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        missing = _REQUIRED_STATE_TABLES.difference(tables)
        if missing:
            self.connection.close()
            self.connection = None
            raise InputLayoutError(f"Native processing state {self.path} lacks required tables: {sorted(missing)}")
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(feature_outputs)")}
        self.feature_has_represented_occurrences = "represented_occurrences" in columns
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def participant_ids(self) -> set[str]:
        if self.connection is None:
            return set()
        return {
            str(row[0]) for row in self.connection.execute("SELECT participant_id FROM participants")
        }

    def get_participant(self, participant_id: str) -> StoredParticipantState | None:
        if self.connection is None:
            return None
        row = self.connection.execute(
            "SELECT * FROM participants WHERE participant_id=?", (participant_id,)
        ).fetchone()
        if row is None:
            return None
        sources = {
            item["canonical_month"]: {
                "filename": item["filename"],
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            }
            for item in self.connection.execute(
                "SELECT * FROM source_files WHERE participant_id=?", (participant_id,)
            )
        }
        represented_column = (
            ", represented_occurrences" if self.feature_has_represented_occurrences else ""
        )
        outputs: dict[str, FeatureOutputInfo] = {}
        for item in self.connection.execute(
            "SELECT feature,filename,row_count,size_bytes,sha256"
            f"{represented_column} FROM feature_outputs WHERE participant_id=?",
            (participant_id,),
        ):
            represented = (
                item["represented_occurrences"] if self.feature_has_represented_occurrences else None
            )
            outputs[item["feature"]] = FeatureOutputInfo(
                item["feature"], item["filename"], item["row_count"],
                item["size_bytes"], item["sha256"], represented,
            )
        return StoredParticipantState(
            row["participant_id"], row["status"], row["parser_version"],
            row["registry_version"], row["source_signature"], row["last_error"],
            sources, outputs,
        )


def _candidate_file_stats(participant_dir: Path) -> tuple[int, int]:
    count = 0
    size = 0
    for path in participant_dir.iterdir():
        if not path.is_file() or canonical_month_from_name(path.name) is None:
            continue
        count += 1
        try:
            size += path.stat().st_size
        except OSError:
            pass
    return count, size


def _discover_one(participant_id: str, participant_dir: Path) -> SourceDiscoveryResult:
    candidate_files, candidate_bytes = _candidate_file_stats(participant_dir)
    try:
        sources = tuple(discover_month_files(participant_dir))
        return SourceDiscoveryResult(
            participant_id, participant_dir, sources,
            candidate_files, candidate_bytes, None,
        )
    except Exception as exc:
        return SourceDiscoveryResult(
            participant_id, participant_dir, (),
            candidate_files, candidate_bytes, str(exc),
        )


def _discover_all(
    participants: dict[str, Path], *, workers: int, show_progress: bool,
) -> dict[str, SourceDiscoveryResult]:
    if not participants:
        return {}
    effective = max(1, min(int(workers), max(1, os.cpu_count() or 1), len(participants)))
    if effective == 1:
        iterator: Iterable[tuple[str, Path]] = participants.items()
        if show_progress:
            iterator = tqdm(iterator, total=len(participants), desc="Scanning source months", unit="participant")
        return {participant_id: _discover_one(participant_id, path) for participant_id, path in iterator}

    results: dict[str, SourceDiscoveryResult] = {}
    with ThreadPoolExecutor(max_workers=effective, thread_name_prefix="wearable-scan") as executor:
        futures = {
            executor.submit(_discover_one, participant_id, path): participant_id
            for participant_id, path in participants.items()
        }
        completed = as_completed(futures)
        if show_progress:
            completed = tqdm(completed, total=len(futures), desc="Scanning source months", unit="participant")
        for future in completed:
            result = future.result()
            results[result.participant_id] = result
    return results


def _source_delta(
    sources: Iterable[SourceFile], stored: StoredParticipantState | None,
) -> dict[str, tuple[str, ...]]:
    current = {item.canonical_month: item for item in sources}
    old = stored.sources if stored is not None else {}
    current_months, old_months = set(current), set(old)
    added = sorted(current_months - old_months)
    missing = sorted(old_months - current_months)
    changed = sorted(
        month for month in current_months & old_months if current[month].sha256 != old[month]["sha256"]
    )
    unchanged = sorted(
        month for month in current_months & old_months if current[month].sha256 == old[month]["sha256"]
    )
    max_old = max(old_months) if old_months else None
    historical = sorted(
        month for month in added if max_old is not None and month <= max_old
    )
    later = sorted(month for month in added if month not in historical)
    return {
        "added": tuple(added),
        "later": tuple(later),
        "historical": tuple(historical),
        "changed": tuple(changed),
        "missing": tuple(missing),
        "unchanged": tuple(unchanged),
    }


def _reason_code(action: str, reason: str) -> str:
    rules = (
        ("forced rebuild", "forced_rebuild"),
        ("has no committed state", "new_participant"),
        ("previous state", "previous_state_not_complete"),
        ("parser or feature registry version changed", "parser_or_registry_changed"),
        ("committed output is missing or invalid", "output_missing_or_invalid"),
        ("participant is absent", "participant_missing_from_snapshot"),
        ("snapshot omits committed months", "missing_committed_months"),
        ("authoritative snapshot removed months", "authoritative_month_removal"),
        ("append-only snapshot omits old months", "append_only_snapshot_conflict"),
        ("historical monthly content changed", "historical_content_changed"),
        ("snapshot unchanged", "snapshot_unchanged"),
        ("recovered historical months", "recovered_historical_months"),
        ("only later months were added", "later_months_added"),
        ("Source discovery or planning failed", "planning_error"),
    )
    for phrase, code in rules:
        if phrase in reason:
            return code
    return f"{action}_other"


def _validate_scan_roots(output_root: Path) -> None:
    if (output_root / CURATION_STATE_FILENAME).exists():
        raise InputLayoutError(
            f"The native processing output path appears to be a curated root "
            f"because it contains {CURATION_STATE_FILENAME}: {output_root}"
        )


def scan_processing_plan(
    input_root: Path, output_root: Path, *, workers: int = 4, mode: str = "auto",
    snapshot_policy: str = "strict-cumulative", selected_participants: set[str] | None = None,
    verify_existing_hashes: bool = False, state_file: Path | None = None, show_progress: bool = True,
) -> ProcessingScanReport:
    """Compute the native-processing plan without mutating input, output, or state."""

    started_clock = time.perf_counter()
    started_at = _now_utc()
    export_root = resolve_export_root(input_root)
    output_root = output_root.expanduser().resolve()
    _validate_scan_roots(output_root)
    resolved_state = (state_file or output_root / NATIVE_STATE_FILENAME).expanduser().resolve()
    participants = discover_participants(export_root, selected_participants)
    workers_requested = int(workers)
    workers_effective = max(
        1, min(workers_requested, max(1, os.cpu_count() or 1), max(1, len(participants)))
    )
    warnings: list[str] = []
    if not output_root.exists():
        warnings.append("The output root does not exist; every discovered participant will be planned as a new build.")
    elif not resolved_state.exists():
        warnings.append(
            f"No native processing state was found at {resolved_state}; existing participant folders cannot be trusted for incremental skipping."
        )
        if any(path.is_dir() and not path.name.startswith(".") for path in output_root.iterdir()):
            warnings.append(
                "The output root already contains participant-like directories but has no managed native state; the processor would rebuild discovered participants."
            )

    requested_missing = tuple(sorted((selected_participants or set()).difference(participants)))
    if requested_missing:
        warnings.append(
            f"{len(requested_missing)} requested participant(s) were not found in the current export snapshot."
        )

    discoveries = _discover_all(participants, workers=workers_effective, show_progress=show_progress,)
    details: list[ParticipantScan] = []

    with ReadOnlyNativeState(resolved_state) as state:
        stored_ids = state.participant_ids()
        if snapshot_policy == "strict-cumulative" and selected_participants is None:
            for participant_id in sorted(stored_ids.difference(participants)):
                stored = state.get_participant(participant_id)
                old_months = tuple(sorted(stored.sources)) if stored is not None else ()
                details.append(ParticipantScan(
                    participant_id=participant_id,
                    planned_action=PlanAction.BLOCK.value,
                    reason="participant is absent from the new cumulative snapshot",
                    reason_code="participant_missing_from_snapshot",
                    current_state=stored.status if stored is not None else None,
                    current_source_signature=None,
                    committed_source_signature=(stored.source_signature if stored is not None else None),
                    source_files_discovered=0,
                    source_bytes_discovered=0,
                    source_files_to_parse=0,
                    source_bytes_to_parse=0,
                    committed_source_files=len(old_months),
                    missing_months=old_months,
                    stored_parser_version=(stored.parser_version if stored is not None else None),
                    stored_registry_version=(stored.registry_version if stored is not None else None),
                    parser_version_changed=(stored is not None and stored.parser_version != PARSER_VERSION),
                    registry_version_changed=(stored is not None and stored.registry_version != REGISTRY_VERSION),
                ))

        for participant_id in sorted(participants):
            discovery = discoveries[participant_id]
            stored = state.get_participant(participant_id)
            if discovery.error is not None:
                reason = f"Source discovery or planning failed: {discovery.error}"
                details.append(ParticipantScan(
                    participant_id=participant_id,
                    planned_action="planning_error",
                    reason=reason,
                    reason_code="planning_error",
                    current_state=stored.status if stored is not None else None,
                    current_source_signature=None,
                    committed_source_signature=(stored.source_signature if stored is not None else None),
                    source_files_discovered=discovery.candidate_files,
                    source_bytes_discovered=discovery.candidate_bytes,
                    source_files_to_parse=0,
                    source_bytes_to_parse=0,
                    committed_source_files=len(stored.sources) if stored is not None else 0,
                    stored_parser_version=(stored.parser_version if stored is not None else None),
                    stored_registry_version=(stored.registry_version if stored is not None else None),
                    parser_version_changed=(stored is not None and stored.parser_version != PARSER_VERSION),
                    registry_version_changed=(stored is not None and stored.registry_version != REGISTRY_VERSION),
                    error=discovery.error,
                ))
                continue

            sources = list(discovery.sources)
            delta = _source_delta(sources, stored)
            try:
                plan = plan_participant(
                    participant_id,
                    discovery.participant_dir,
                    sources,
                    stored,
                    output_root,
                    mode=mode,
                    snapshot_policy=snapshot_policy,
                    verify_existing_hashes=verify_existing_hashes,
                )
            except Exception as exc:
                reason = f"Source discovery or planning failed: {exc}"
                details.append(ParticipantScan(
                    participant_id=participant_id,
                    planned_action="planning_error",
                    reason=reason,
                    reason_code="planning_error",
                    current_state=stored.status if stored is not None else None,
                    current_source_signature=source_signature(sources),
                    committed_source_signature=(stored.source_signature if stored is not None else None),
                    source_files_discovered=len(sources),
                    source_bytes_discovered=sum(item.size_bytes for item in sources),
                    source_files_to_parse=0,
                    source_bytes_to_parse=0,
                    committed_source_files=len(stored.sources) if stored is not None else 0,
                    added_months=delta["added"],
                    later_added_months=delta["later"],
                    recovered_historical_months=delta["historical"],
                    changed_months=delta["changed"],
                    missing_months=delta["missing"],
                    unchanged_months=delta["unchanged"],
                    stored_parser_version=(stored.parser_version if stored is not None else None),
                    stored_registry_version=(stored.registry_version if stored is not None else None),
                    parser_version_changed=(stored is not None and stored.parser_version != PARSER_VERSION),
                    registry_version_changed=(stored is not None and stored.registry_version != REGISTRY_VERSION),
                    error=str(exc),
                ))
                continue

            if stored is None:
                output_usable: bool | None = None
            elif stored.status == "complete_empty":
                output_usable = True
            elif stored.status == "complete":
                output_usable = existing_output_is_usable(
                    output_root / participant_id, stored, verify_existing_hashes
                )
            else:
                output_usable = False

            details.append(ParticipantScan(
                participant_id=participant_id,
                planned_action=plan.action.value,
                reason=plan.reason,
                reason_code=_reason_code(plan.action.value, plan.reason),
                current_state=stored.status if stored is not None else None,
                current_source_signature=source_signature(sources),
                committed_source_signature=(stored.source_signature if stored is not None else None),
                source_files_discovered=len(sources),
                source_bytes_discovered=sum(item.size_bytes for item in sources),
                source_files_to_parse=len(plan.sources_to_parse),
                source_bytes_to_parse=sum(item.size_bytes for item in plan.sources_to_parse),
                committed_source_files=len(stored.sources) if stored is not None else 0,
                added_months=delta["added"],
                later_added_months=delta["later"],
                recovered_historical_months=delta["historical"],
                changed_months=delta["changed"],
                missing_months=delta["missing"],
                unchanged_months=delta["unchanged"],
                existing_output_usable=output_usable,
                stored_parser_version=(stored.parser_version if stored is not None else None),
                stored_registry_version=(stored.registry_version if stored is not None else None),
                parser_version_changed=(stored is not None and stored.parser_version != PARSER_VERSION),
                registry_version_changed=(stored is not None and stored.registry_version != REGISTRY_VERSION),
            ))

    details.sort(key=lambda item: item.participant_id)

    stored_parser_versions = Counter(
        item.stored_parser_version or "<missing>" for item in details if item.current_state is not None
    )
    stored_registry_versions = Counter(
        item.stored_registry_version or "<missing>" for item in details if item.current_state is not None
    )
    mismatch_only = [
        item for item in details
        if item.reason_code == "parser_or_registry_changed"
        and item.current_source_signature is not None
        and item.current_source_signature == item.committed_source_signature
        and item.existing_output_usable is True
        and not item.added_months
        and not item.changed_months
        and not item.missing_months
    ]
    state_versions = {
        "current_parser_version": PARSER_VERSION,
        "current_registry_version": REGISTRY_VERSION,
        "stored_parser_version_counts": dict(sorted(stored_parser_versions.items())),
        "stored_registry_version_counts": dict(sorted(stored_registry_versions.items())),
        "parser_mismatch_participants": sum(item.parser_version_changed for item in details),
        "registry_mismatch_participants": sum(item.registry_version_changed for item in details),
        "version_mismatch_only_participants": len(mismatch_only),
    }
    environment = processing_environment_manifest().as_dict()
    warnings.extend(environment.get("warnings", ()))
    if mismatch_only:
        warnings.append(
            f"{len(mismatch_only)} participant(s) require rebuild solely because the parser/registry identity "
            f"stored in native state differs from the current implementation. Source signatures are unchanged "
            f"and committed outputs are usable. Do not rebuild solely from this result until the originating "
            f"processing version has been verified. Possible causes include a historical local checkout, a "
            f"semantic-version metadata change, or a genuinely different processing implementation."
        )

    action_counts = Counter(item.planned_action for item in details)
    reason_counts = Counter(item.reason_code for item in details)
    source_file_counts = {
        "discovered": sum(item.source_files_discovered for item in details),
        "committed": sum(item.committed_source_files for item in details),
        "to_parse": sum(item.source_files_to_parse for item in details),
        "added": sum(len(item.added_months) for item in details),
        "later_added": sum(len(item.later_added_months) for item in details),
        "recovered_historical": sum(len(item.recovered_historical_months) for item in details),
        "changed": sum(len(item.changed_months) for item in details),
        "missing": sum(len(item.missing_months) for item in details),
        "unchanged": sum(len(item.unchanged_months) for item in details),
    }
    source_byte_counts = {
        "discovered": sum(item.source_bytes_discovered for item in details),
        "to_parse": sum(item.source_bytes_to_parse for item in details),
    }
    participant_counts = {
        "discovered_in_snapshot": len(participants),
        "stored_in_state": len({item.participant_id for item in details if item.current_state is not None}),
        "skip": action_counts.get(PlanAction.SKIP.value, 0),
        "incremental": action_counts.get(PlanAction.INCREMENTAL.value, 0),
        "rebuild": action_counts.get(PlanAction.REBUILD.value, 0),
        "block": action_counts.get(PlanAction.BLOCK.value, 0),
        "planning_error": action_counts.get("planning_error", 0),
        "total_plan_records": len(details),
    }
    processing_needed = bool(
        participant_counts["incremental"] or participant_counts["rebuild"]
    )
    safe_to_process = not bool(
        participant_counts["block"] or participant_counts["planning_error"]
    )
    no_op = (
        safe_to_process
        and not processing_needed
        and participant_counts["skip"] == participant_counts["total_plan_records"]
    )
    status = "complete" if safe_to_process else "complete_with_issues"
    plan_payload = [
        {
            "participant_id": item.participant_id,
            "action": item.planned_action,
            "reason_code": item.reason_code,
            "stored_parser_version": item.stored_parser_version,
            "stored_registry_version": item.stored_registry_version,
            "current_source_signature": item.current_source_signature,
            "committed_source_signature": item.committed_source_signature,
            "source_files_to_parse": item.source_files_to_parse,
            "added_months": item.added_months,
            "changed_months": item.changed_months,
            "missing_months": item.missing_months,
        }
        for item in details
    ]
    finished_at = _now_utc()
    return ProcessingScanReport(
        scan_id=_fingerprint({"started_at": started_at, "plan": plan_payload})[:32],
        started_at=started_at,
        finished_at=finished_at,
        wall_clock_seconds=time.perf_counter() - started_clock,
        status=status,
        input_root=str(Path(input_root).expanduser().resolve()),
        export_root=str(export_root),
        output_root=str(output_root),
        state_file=str(resolved_state),
        state_file_exists=resolved_state.is_file(),
        scan_version=SCAN_VERSION,
        package_version=__version__,
        parser_version=PARSER_VERSION,
        registry_version=REGISTRY_VERSION,
        mode=mode,
        snapshot_policy=snapshot_policy,
        verify_existing_hashes=verify_existing_hashes,
        workers_requested=workers_requested,
        workers_effective=workers_effective,
        participants_requested=(len(selected_participants) if selected_participants is not None else None),
        participants_missing_from_selection=requested_missing,
        participant_counts=participant_counts,
        source_file_counts=source_file_counts,
        source_byte_counts=source_byte_counts,
        reason_counts=dict(sorted(reason_counts.items())),
        processing_needed=processing_needed,
        safe_to_process=safe_to_process,
        no_op=no_op,
        plan_fingerprint=_fingerprint(plan_payload),
        state_versions=state_versions,
        processing_environment=environment,
        warnings=tuple(dict.fromkeys(warnings)),
        participant_details=details,
    )


def _human_bytes(value: int) -> str:
    amount = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            return f"{amount:,.2f} {unit}"
        amount /= 1024.0
    return f"{value:,} B"


def format_processing_scan(report: ProcessingScanReport | dict[str, Any], *, max_listed: int = 50,) -> str:
    payload = report.as_dict(include_details=True) if isinstance(report, ProcessingScanReport) else report
    participants = payload["participants"]
    files = payload["source_files"]
    byte_counts = payload["source_bytes"]
    lines = [
        "Native processing scan",
        f"Status: {payload['status']}",
        f"Input snapshot: {payload['paths']['export_root']}",
        f"Native output: {payload['paths']['output_root']}",
        f"State file: {payload['paths']['state_file']} "
        f"({'found' if payload['paths']['state_file_exists'] else 'not found'})",
        "",
        "Processing identity:",
        f"  imported package:        {payload.get('processing_environment', {}).get('imported_package_path')}",
        f"  import source:           {payload.get('processing_environment', {}).get('import_source_kind')}",
        f"  current parser:          {payload.get('state_versions', {}).get('current_parser_version')}",
        f"  current registry:        {payload.get('state_versions', {}).get('current_registry_version')}",
        f"  parser mismatches:       {payload.get('state_versions', {}).get('parser_mismatch_participants', 0):,}",
        f"  registry mismatches:     {payload.get('state_versions', {}).get('registry_mismatch_participants', 0):,}",
        f"  mismatch-only rebuilds:  {payload.get('state_versions', {}).get('version_mismatch_only_participants', 0):,}",
        "",
        "Participant plan:",
        f"  discovered in snapshot: {participants['discovered_in_snapshot']:,}",
        f"  skip:                   {participants['skip']:,}",
        f"  incremental update:     {participants['incremental']:,}",
        f"  rebuild:                {participants['rebuild']:,}",
        f"  blocked:                {participants['block']:,}",
        f"  planning errors:        {participants['planning_error']:,}",
        "",
        "Source-month scope:",
        f"  discovered:             {files['discovered']:,} ({_human_bytes(byte_counts['discovered'])})",
        f"  would be parsed:        {files['to_parse']:,} ({_human_bytes(byte_counts['to_parse'])})",
        f"  later months added:     {files['later_added']:,}",
        f"  historical recoveries:  {files['recovered_historical']:,}",
        f"  changed historical:     {files['changed']:,}",
        f"  missing committed:      {files['missing']:,}",
        "",
        f"Processing needed: {'yes' if payload['processing_needed'] else 'no'}",
        f"Safe to process:   {'yes' if payload['safe_to_process'] else 'no'}",
        f"No-op plan:        {'yes' if payload['no_op'] else 'no'}",
        f"Plan fingerprint:  {payload['plan_fingerprint']}",
        f"Scan wall time:    {payload['wall_clock_seconds']:.2f} seconds",
    ]
    state_versions = payload.get("state_versions", {})
    if state_versions.get("stored_parser_version_counts") or state_versions.get("stored_registry_version_counts"):
        lines.extend(["", "Stored state versions:"])
        for version, count in sorted(state_versions.get("stored_parser_version_counts", {}).items()):
            lines.append(f"  parser {version}: {count:,}")
        for version, count in sorted(state_versions.get("stored_registry_version_counts", {}).items()):
            lines.append(f"  registry {version}: {count:,}")
    if payload.get("warnings"):
        lines.extend(["", "Warnings:"])
        lines.extend(f"  - {warning}" for warning in payload["warnings"])
    if payload.get("reasons"):
        lines.extend(["", "Plan reasons:"])
        lines.extend(
            f"  {reason}: {count:,}" for reason, count in sorted(payload["reasons"].items())
        )
    actionable = [
        item for item in payload.get("participant_details", [])
        if item["planned_action"] != PlanAction.SKIP.value
    ]
    if actionable:
        lines.extend(["", "Participants requiring attention:"])
        limit = max(0, int(max_listed))
        for item in actionable[:limit]:
            lines.append(
                f"  {item['participant_id']}: {item['planned_action']} — {item['reason']} "
                f"(files to parse: {item['source_files_to_parse']})"
            )
        if len(actionable) > limit:
            lines.append(
                f"  ... {len(actionable) - limit:,} additional participant(s); use --json --include-details for the complete plan."
            )
    return "\n".join(lines) + "\n"
