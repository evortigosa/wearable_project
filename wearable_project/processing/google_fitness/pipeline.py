"""
Wearable Data Processing and Modeling project
Participant-owned Google Fitness native-processing orchestration for WP4.
"""


from __future__ import annotations
import concurrent.futures
from collections import Counter
import ctypes
import gc
import multiprocessing as mp
import os
import shutil
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping
from tqdm import tqdm
from wearable_project.exceptions import InputLayoutError, OutputValidationError, ParticipantProcessingError
from wearable_project.processing.google_fitness.cleaners import CLEANER_VERSION, clean_parse_result
from wearable_project.processing.google_fitness.models import PARSER_VERSION, SourceFile
from wearable_project.processing.google_fitness.parser import (
    canonical_month_from_name, discover_month_files, merge_parse_results, parse_month_file,
)
from wearable_project.processing.google_fitness.registry import REGISTRY_VERSION
from wearable_project.processing.google_fitness.tracker import build_processing_report, peak_rss_bytes
from wearable_project.processing.google_fitness.writer import (
    DirectorySwap, FeatureOutputInfo, STATE_FILENAME, StateDatabase, existing_output_is_usable, read_ledger,
    safe_feature_filename, source_signature, validate_participant_stage, write_feature_csv, write_ledger,
)


PIPELINE_VERSION = "google-fitness-pipeline-wp4-0.1.0"


def _release_process_memory() -> None:
    """Best-effort return of freed parent allocations on Linux."""

    gc.collect()
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return
    try:
        trim(0)
    except Exception:
        pass


class PlanAction(str, Enum):
    SKIP = "skip"
    INCREMENTAL = "incremental"
    REBUILD = "rebuild"
    BLOCK = "block"


@dataclass(slots=True)
class ParticipantPlan:
    participant_id: str
    participant_dir: Path
    action: PlanAction
    reason: str
    all_sources: list[SourceFile]


@dataclass(slots=True)
class ParticipantTask:
    participant_id: str
    sources: list[SourceFile]
    stage_dir: Path
    ledger_path: Path
    row_error_policy: str


@dataclass(slots=True)
class BuildResult:
    participant_id: str
    stage_dir: Path
    ledger_path: Path
    output_infos: list[FeatureOutputInfo] = field(default_factory=list)
    feature_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_files_read: int = 0
    source_bytes_read: int = 0
    outer_rows: int = 0
    google_rows: int = 0
    irrelevant_rows: int = 0
    supported_outer_rows: int = 0
    unsupported_outer_rows: int = 0
    payloads_decoded: int = 0
    payload_decode_failures: int = 0
    candidate_items: int = 0
    valid_candidates: int = 0
    malformed_candidates: int = 0
    activity_items_routed: int = 0
    step_count_v2_alternatives: int = 0
    output_rows: int = 0
    represented_occurrences: int = 0
    exact_duplicates_removed: int = 0
    revisions_resolved: int = 0
    unresolved_conflict_groups: int = 0
    conflict_rows: int = 0
    source_occurrences: int = 0
    diagnostics: int = 0
    diagnostics_by_code: dict[str, int] = field(default_factory=dict)
    unknown_activity_labels: list[str] = field(default_factory=list)
    unsupported_contracts: list[str] = field(default_factory=list)
    runtime_seconds: float | None = None
    peak_rss_bytes: int | None = None
    empty: bool = False
    error: str | None = None
    traceback_text: str | None = None

    def result_payload(self) -> dict[str, Any]:
        return {
            "source_files_read": self.source_files_read,
            "source_bytes_read": self.source_bytes_read,
            "outer_rows": self.outer_rows,
            "google_rows": self.google_rows,
            "irrelevant_rows": self.irrelevant_rows,
            "supported_outer_rows": self.supported_outer_rows,
            "unsupported_outer_rows": self.unsupported_outer_rows,
            "payloads_decoded": self.payloads_decoded,
            "payload_decode_failures": self.payload_decode_failures,
            "candidate_items": self.candidate_items,
            "valid_candidates": self.valid_candidates,
            "malformed_candidates": self.malformed_candidates,
            "activity_items_routed": self.activity_items_routed,
            "step_count_v2_alternatives": self.step_count_v2_alternatives,
            "output_rows": self.output_rows,
            "represented_occurrences": self.represented_occurrences,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "revisions_resolved": self.revisions_resolved,
            "unresolved_conflict_groups": self.unresolved_conflict_groups,
            "conflict_rows": self.conflict_rows,
            "source_occurrences": self.source_occurrences,
            "diagnostics": self.diagnostics,
            "diagnostics_by_code": dict(sorted(self.diagnostics_by_code.items())),
            "feature_files": len(self.output_infos),
            "output_size_bytes": sum(item.size_bytes for item in self.output_infos),
            "runtime_seconds": self.runtime_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "unknown_activity_labels": list(self.unknown_activity_labels),
            "unsupported_contracts": list(self.unsupported_contracts),
        }


@dataclass(slots=True)
class RunSummary:
    run_id: str
    discovered: int = 0
    skipped: int = 0
    incremental: int = 0
    rebuilt: int = 0
    committed: int = 0
    empty: int = 0
    blocked: int = 0
    failed: int = 0
    blocks: dict[str, str] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] | None = None

    def basic_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "discovered": self.discovered,
            "skipped": self.skipped,
            "incremental": self.incremental,
            "rebuilt": self.rebuilt,
            "committed": self.committed,
            "empty": self.empty,
            "blocked": self.blocked,
            "failed": self.failed,
            "blocks": dict(self.blocks),
            "failures": dict(self.failures),
            "configuration": dict(self.configuration),
        }

    def as_dict(self) -> dict[str, Any]:
        return self.report if self.report is not None else self.basic_dict()


def contains_month_file(path: Path) -> bool:
    return path.is_dir() and any(
        child.is_file() and canonical_month_from_name(child.name) is not None
        for child in path.iterdir()
    )


def resolve_export_root(input_root: Path) -> Path:
    root = input_root.expanduser().resolve()
    if not root.is_dir():
        raise InputLayoutError(f"input is not a directory: {root}")
    if any(
        contains_month_file(child)
        for child in root.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    ):
        return root
    children = [
        child
        for child in root.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    ]
    if len(children) == 1 and any(
        contains_month_file(child)
        for child in children[0].iterdir()
        if child.is_dir() and not child.name.startswith(".")
    ):
        return children[0]
    raise InputLayoutError(f"no participant directories with YYYY-M.csv or YYYY-MM.csv were found under {root}")


def discover_participants(root: Path, selected: set[str] | None = None) -> dict[str, Path]:
    return {
        child.name: child
        for child in sorted(root.iterdir(), key=lambda item: item.name)
        if child.is_dir()
        and not child.name.startswith(".")
        and (selected is None or child.name in selected)
        and contains_month_file(child)
    }


def _map_sources(sources: Iterable[SourceFile]) -> dict[str, SourceFile]:
    return {item.canonical_month: item for item in sources}


def plan_participant(
    participant_id: str, participant_dir: Path, sources: list[SourceFile], stored: Any, output_root: Path,
    *, mode: str, snapshot_policy: str, verify_existing_hashes: bool,
) -> ParticipantPlan:
    def rebuild(reason: str) -> ParticipantPlan:
        return ParticipantPlan(participant_id, participant_dir, PlanAction.REBUILD, reason, sources)

    if mode == "rebuild":
        return rebuild("forced rebuild")
    if stored is None:
        return rebuild("participant has no committed Google Fitness state")
    if stored.status not in {"complete", "complete_empty"}:
        return rebuild(f"previous state is {stored.status!r}")
    if (
        stored.parser_version != PARSER_VERSION
        or stored.registry_version != REGISTRY_VERSION
        or stored.cleaner_version != CLEANER_VERSION
        or stored.pipeline_version != PIPELINE_VERSION
    ):
        return rebuild("parser, registry, cleaner, or pipeline version changed")
    if not existing_output_is_usable(
        output_root / participant_id,
        stored,
        verify_hashes=verify_existing_hashes,
    ):
        return rebuild("committed output is missing or invalid")

    current = _map_sources(sources)
    old = stored.sources
    old_months = set(old)
    current_months = set(current)
    missing = sorted(old_months - current_months)
    added = sorted(current_months - old_months)
    changed = sorted(
        month for month in old_months & current_months if old[month]["sha256"] != current[month].sha256
    )

    if missing:
        if snapshot_policy == "strict-cumulative":
            return ParticipantPlan(
                participant_id,
                participant_dir,
                PlanAction.BLOCK,
                f"snapshot omits committed months: {missing}",
                sources,
            )
        if snapshot_policy == "authoritative":
            return rebuild(f"authoritative snapshot removed months: {missing}")
        return ParticipantPlan(
            participant_id,
            participant_dir,
            PlanAction.BLOCK,
            "append-only cannot preserve absent months while WP4 uses full-history recomputation",
            sources,
        )
    if changed:
        return rebuild(f"historical monthly content changed: {changed}")
    if not added:
        return ParticipantPlan(
            participant_id,
            participant_dir,
            PlanAction.SKIP,
            "snapshot unchanged",
            sources,
        )
    max_old = max(old_months) if old_months else None
    historical = [month for month in added if max_old is not None and month <= max_old]
    if historical:
        return rebuild(f"recovered historical months: {historical}")
    return ParticipantPlan(
        participant_id,
        participant_dir,
        PlanAction.INCREMENTAL,
        f"only later months were added: {added}; execution uses full-history recomputation",
        sources,
    )


def _hash_participant_sources(item: tuple[str, str],) -> tuple[str, list[SourceFile] | None, str | None]:
    participant_id, participant_dir = item
    try:
        return participant_id, discover_month_files(Path(participant_dir)), None
    except Exception as exc:  # returned so one directory does not kill the pre-pass
        return participant_id, None, f"{type(exc).__name__}: {exc}"


def prehash_participant_sources(
    participants: Mapping[str, Path], workers: int, *, show_progress: bool = True,
) -> dict[str, list[SourceFile] | str]:
    items = [(participant_id, str(path)) for participant_id, path in participants.items()]
    if workers <= 1 or len(items) < 2:
        result: dict[str, list[SourceFile] | str] = {}
        iterator = tqdm(
            items,
            desc="Hashing participant sources",
            unit="participant",
            disable=not show_progress,
        )
        for item in iterator:
            participant_id, sources, error = _hash_participant_sources(item)
            result[participant_id] = error if error is not None else (sources or [])
        return result

    result: dict[str, list[SourceFile] | str] = {}
    # Source discovery and SHA-256 calculation are I/O-heavy and hashlib releases the GIL. Threads avoid a
    # second process pool (and its semaphore/resource tracker lifecycle) while participant parsing and cleaning
    # remain process-owned.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, len(items)),
        thread_name_prefix="google-source-hash",
    ) as executor:
        futures = [executor.submit(_hash_participant_sources, item) for item in items]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Hashing participant sources",
            unit="participant",
            disable=not show_progress,
        ):
            participant_id, sources, error = future.result()
            result[participant_id] = error if error is not None else (sources or [])
    return result


def _ledger_payload(participant_id: str, parsed: Any, cleaned: Any) -> dict[str, Any]:
    source_occurrences = []
    for item in parsed.source_occurrences:
        payload = asdict(item)
        payload["occurrence_key"] = item.occurrence_key
        source_occurrences.append(payload)
    decisions: list[dict[str, Any]] = []
    exceptional: list[dict[str, Any]] = []
    for feature, result in sorted(cleaned.features.items()):
        decisions.extend(item.as_dict() for item in result.decisions)
        for audit in result.audits:
            if (
                audit.occurrence_count > 1
                or audit.duplicate_count
                or audit.revision_count
                or audit.conflict_group_ids
            ):
                payload = audit.as_dict()
                payload["feature"] = feature
                exceptional.append(payload)
    return {
        "participant_id": participant_id,
        "source_occurrences": source_occurrences,
        "malformed_candidates": [asdict(item) for item in parsed.malformed_items],
        "parser_diagnostics": [asdict(item) for item in parsed.diagnostics],
        "reconciliation_decisions": decisions,
        "exceptional_row_audits": exceptional,
        "parser_accounting": {
            "outer_rows": parsed.outer_rows_seen,
            "google_rows": parsed.google_rows_seen,
            "irrelevant_rows": parsed.irrelevant_rows_seen,
            "candidate_items": parsed.candidate_items_seen,
            "valid_candidates": parsed.valid_candidates,
            "malformed_candidates": parsed.malformed_candidates,
        },
        "cleaner_accounting": cleaned.accounting_payload(),
    }


def build_participant(task: ParticipantTask) -> BuildResult:
    started = time.perf_counter()
    result = BuildResult(task.participant_id, task.stage_dir, task.ledger_path)
    try:
        shutil.rmtree(task.stage_dir, ignore_errors=True)
        task.stage_dir.mkdir(parents=True, exist_ok=True)
        task.ledger_path.unlink(missing_ok=True)
        month_results = [
            parse_month_file(
                source,
                task.participant_id,
                row_error_policy=task.row_error_policy,
            )
            for source in sorted(task.sources, key=lambda item: item.canonical_month)
        ]
        parsed = merge_parse_results(month_results)
        parsed.assert_accounting_closed()
        cleaned = clean_parse_result(parsed)
        cleaned.assert_accounting_closed()

        output_infos: list[FeatureOutputInfo] = []
        feature_stats: dict[str, dict[str, Any]] = {}
        for feature, feature_result in sorted(cleaned.features.items()):
            filename = safe_feature_filename(feature)
            info = write_feature_csv(
                feature,
                feature_result.rows,
                task.stage_dir / filename,
                represented_occurrences=feature_result.represented_occurrences,
            )
            output_infos.append(info)
            payload = feature_result.accounting_payload()
            payload.update(info.as_dict())
            feature_stats[feature] = payload

        write_ledger(task.ledger_path, _ledger_payload(task.participant_id, parsed, cleaned),)
        result.output_infos = output_infos
        result.feature_stats = feature_stats
        result.source_files_read = len(task.sources)
        result.source_bytes_read = sum(item.size_bytes for item in task.sources)
        result.outer_rows = parsed.outer_rows_seen
        result.google_rows = parsed.google_rows_seen
        result.irrelevant_rows = parsed.irrelevant_rows_seen
        result.supported_outer_rows = parsed.supported_outer_rows
        result.unsupported_outer_rows = parsed.unsupported_outer_rows
        result.payloads_decoded = parsed.payloads_decoded
        result.payload_decode_failures = parsed.payload_decode_failures
        result.candidate_items = parsed.candidate_items_seen
        result.valid_candidates = parsed.valid_candidates
        result.malformed_candidates = parsed.malformed_candidates
        result.activity_items_routed = parsed.activity_items_routed
        result.step_count_v2_alternatives = parsed.step_count_v2_alternatives_emitted
        result.output_rows = cleaned.output_rows
        result.represented_occurrences = cleaned.represented_occurrences
        result.exact_duplicates_removed = cleaned.exact_duplicates_removed
        result.revisions_resolved = cleaned.revisions_resolved
        result.unresolved_conflict_groups = cleaned.unresolved_conflict_groups
        result.conflict_rows = cleaned.conflict_rows
        result.source_occurrences = len(parsed.source_occurrences)
        result.diagnostics = len(parsed.diagnostics)
        result.diagnostics_by_code = dict(
            sorted(Counter(item.code for item in parsed.diagnostics).items())
        )
        result.unknown_activity_labels = sorted(parsed.unknown_activity_labels)
        result.unsupported_contracts = sorted(parsed.unsupported_contracts)
        result.empty = not output_infos
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.traceback_text = traceback.format_exc()
    finally:
        result.runtime_seconds = time.perf_counter() - started
        result.peak_rss_bytes = peak_rss_bytes()
        gc.collect()
    return result


def worker_entry(task: ParticipantTask, connection: Any) -> None:
    """Run exactly one participant in a fresh spawned process."""

    result = build_participant(task)
    try:
        connection.send(result)
    finally:
        connection.close()


def _validate_roots(input_root: Path, output_root: Path, state_file: Path) -> None:
    if output_root == input_root or input_root in output_root.parents:
        raise OutputValidationError(f"output root {output_root} must not lie inside input root {input_root}")
    if output_root in input_root.parents:
        raise OutputValidationError(f"input root {input_root} must not lie inside output root {output_root}")
    if state_file == input_root or input_root in state_file.parents:
        raise OutputValidationError("state database must not be written inside the raw root")
    if output_root.exists() and not state_file.exists():
        visible = [item for item in output_root.iterdir() if not item.name.startswith(".")]
        if visible:
            raise OutputValidationError(f"nonempty output root lacks Google Fitness state: {output_root}")


def _remove_output_then_commit(final_dir: Path, backup_dir: Path, commit: Any,) -> None:
    shutil.rmtree(backup_dir, ignore_errors=True)
    had_previous = final_dir.exists()
    if had_previous:
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(final_dir, backup_dir)
    try:
        commit()
    except Exception:
        if had_previous and backup_dir.exists() and not final_dir.exists():
            os.replace(backup_dir, final_dir)
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)


def process_dataset(
    input_root: Path, output_root: Path, *, workers: int = 4, max_in_flight: int | None = None,
    mode: str = "auto", snapshot_policy: str = "strict-cumulative", row_error_policy: str = "fail-participant",
    selected_participants: set[str] | None = None, verify_existing_hashes: bool = False, fail_fast: bool = False,
    state_file: Path | None = None,
) -> RunSummary:
    if workers < 1:
        raise ValueError("workers must be at least one")
    if mode not in {"auto", "rebuild"}:
        raise ValueError(f"unsupported mode: {mode}")
    if snapshot_policy not in {"strict-cumulative", "authoritative", "append-only"}:
        raise ValueError(f"unsupported snapshot policy: {snapshot_policy}")
    if row_error_policy not in {"fail-participant", "skip-row"}:
        raise ValueError(f"unsupported row error policy: {row_error_policy}")

    input_root = resolve_export_root(Path(input_root))
    output_root = Path(output_root).expanduser().resolve()
    state_file = (
        Path(state_file).expanduser().resolve()
        if state_file is not None
        else output_root / STATE_FILENAME
    )
    _validate_roots(input_root, output_root, state_file)
    output_root.mkdir(parents=True, exist_ok=True)

    participants = discover_participants(input_root, selected_participants)
    run_id = uuid.uuid4().hex
    run_started = time.perf_counter()
    concurrency = max(1, min(max_in_flight or workers, workers, max(1, len(participants))))
    summary = RunSummary(
        run_id=run_id,
        discovered=len(participants),
        configuration={
            "mode": mode,
            "snapshot_policy": snapshot_policy,
            "row_error_policy": row_error_policy,
            "verify_existing_hashes": bool(verify_existing_hashes),
            "existing_output_validation": (
                "file_set_size_and_sha256"
                if verify_existing_hashes
                else "file_set_and_size"
            ),
            "source_validation": "sha256_all_discovered_month_files",
            "workers_requested": workers,
            "max_in_flight_requested": max_in_flight,
            "max_in_flight_effective": concurrency,
            "fail_fast": bool(fail_fast),
        },
    )

    temp_root = output_root.parent / f".{output_root.name}.wp4_tmp" / run_id
    stage_root = temp_root / "stage"
    ledger_root = temp_root / "ledger"
    backup_root = temp_root / "backup"
    for path in (stage_root, ledger_root, backup_root):
        path.mkdir(parents=True, exist_ok=True)

    source_discovery = prehash_participant_sources(participants, workers)

    with StateDatabase(state_file) as state:
        state.begin_run(run_id, input_root, output_root)
        plans: list[ParticipantPlan] = []
        try:
            for participant_id, participant_dir in participants.items():
                discovered = source_discovery.get(participant_id)
                if isinstance(discovered, str):
                    summary.failed += 1
                    summary.failures[participant_id] = discovered
                    state.mark_failed(participant_id, discovered)
                    state.record_run_participant(
                        run_id,
                        participant_id,
                        planned_action=PlanAction.REBUILD.value,
                        plan_reason="source discovery failed",
                        final_status="failed",
                        result={},
                        error_message=discovered,
                    )
                    if fail_fast:
                        raise ParticipantProcessingError(f"{participant_id}: {discovered}")
                    continue
                sources = list(discovered or [])
                plan = plan_participant(
                    participant_id,
                    participant_dir,
                    sources,
                    state.get_participant(participant_id),
                    output_root,
                    mode=mode,
                    snapshot_policy=snapshot_policy,
                    verify_existing_hashes=verify_existing_hashes,
                )
                if plan.action is PlanAction.SKIP:
                    summary.skipped += 1
                    state.record_run_participant(
                        run_id,
                        participant_id,
                        planned_action=plan.action.value,
                        plan_reason=plan.reason,
                        final_status="skipped",
                        result={"source_files_discovered": len(sources)},
                    )
                    continue
                if plan.action is PlanAction.BLOCK:
                    summary.blocked += 1
                    summary.blocks[participant_id] = plan.reason
                    state.record_run_participant(
                        run_id,
                        participant_id,
                        planned_action=plan.action.value,
                        plan_reason=plan.reason,
                        final_status="blocked",
                        result={"source_files_discovered": len(sources)},
                        error_message=plan.reason,
                    )
                    continue
                state.mark_in_progress(
                    participant_id,
                    parser_version=PARSER_VERSION,
                    registry_version=REGISTRY_VERSION,
                    cleaner_version=CLEANER_VERSION,
                    pipeline_version=PIPELINE_VERSION,
                )
                plans.append(plan)

            plans.sort(
                key=lambda item: sum(source.size_bytes for source in item.all_sources), reverse=True,
            )
            def task_for(plan: ParticipantPlan) -> ParticipantTask:
                return ParticipantTask(
                    participant_id=plan.participant_id,
                    sources=plan.all_sources,
                    stage_dir=stage_root / plan.participant_id,
                    ledger_path=ledger_root / f"{plan.participant_id}.json.gz",
                    row_error_policy=row_error_policy,
                )

            def handle_result(plan: ParticipantPlan, result: BuildResult) -> None:
                if result.error:
                    summary.failed += 1
                    summary.failures[plan.participant_id] = result.error
                    state.mark_failed(plan.participant_id, result.error)
                    state.record_run_participant(
                        run_id,
                        plan.participant_id,
                        planned_action=plan.action.value,
                        plan_reason=plan.reason,
                        final_status="failed",
                        result=result.result_payload(),
                        feature_stats=result.feature_stats,
                        error_message=result.error,
                    )
                    shutil.rmtree(result.stage_dir, ignore_errors=True)
                    result.ledger_path.unlink(missing_ok=True)
                    if fail_fast:
                        raise ParticipantProcessingError(
                            f"{plan.participant_id}: {result.error}\n"
                            f"{result.traceback_text or ''}"
                        )
                    return

                ledger: dict[str, Any] = {}
                final_dir = output_root / plan.participant_id
                backup_dir = backup_root / plan.participant_id
                try:
                    ledger = read_ledger(result.ledger_path)
                    if ledger.get("participant_id") != plan.participant_id:
                        raise OutputValidationError(f"worker ledger participant mismatch for {plan.participant_id}")
                    if result.empty:
                        _remove_output_then_commit(
                            final_dir,
                            backup_dir,
                            lambda: state.commit_participant(
                                participant_id=plan.participant_id,
                                parser_version=PARSER_VERSION,
                                registry_version=REGISTRY_VERSION,
                                cleaner_version=CLEANER_VERSION,
                                pipeline_version=PIPELINE_VERSION,
                                signature=source_signature(plan.all_sources),
                                sources=plan.all_sources,
                                outputs=[],
                                ledger=ledger,
                                status="complete_empty",
                            ),
                        )
                        shutil.rmtree(result.stage_dir, ignore_errors=True)
                        summary.empty += 1
                        final_status = "complete_empty"
                    else:
                        outputs = validate_participant_stage(
                            result.stage_dir,
                            plan.participant_id,
                            result.output_infos,
                        )
                        swap = DirectorySwap(final_dir, result.stage_dir, backup_dir)
                        swap.apply()
                        try:
                            state.commit_participant(
                                participant_id=plan.participant_id,
                                parser_version=PARSER_VERSION,
                                registry_version=REGISTRY_VERSION,
                                cleaner_version=CLEANER_VERSION,
                                pipeline_version=PIPELINE_VERSION,
                                signature=source_signature(plan.all_sources),
                                sources=plan.all_sources,
                                outputs=outputs,
                                ledger=ledger,
                                status="complete",
                            )
                        except Exception:
                            swap.rollback()
                            raise
                        swap.finalize()
                        summary.committed += 1
                        final_status = "committed"
                    if plan.action is PlanAction.INCREMENTAL:
                        summary.incremental += 1
                    else:
                        summary.rebuilt += 1
                    state.record_run_participant(
                        run_id,
                        plan.participant_id,
                        planned_action=plan.action.value,
                        plan_reason=plan.reason,
                        final_status=final_status,
                        result=result.result_payload(),
                        feature_stats=result.feature_stats,
                    )
                except Exception as exc:
                    summary.failed += 1
                    summary.failures[plan.participant_id] = str(exc)
                    state.mark_failed(plan.participant_id, str(exc))
                    state.record_run_participant(
                        run_id,
                        plan.participant_id,
                        planned_action=plan.action.value,
                        plan_reason=plan.reason,
                        final_status="failed",
                        result=result.result_payload(),
                        feature_stats=result.feature_stats,
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                    shutil.rmtree(result.stage_dir, ignore_errors=True)
                    if fail_fast:
                        raise
                finally:
                    result.ledger_path.unlink(missing_ok=True)
                    try:
                        ledger.clear()
                    except (NameError, AttributeError):
                        pass
                    _release_process_memory()

            context = mp.get_context("spawn")
            plan_iterator = iter(plans)
            running: dict[str, tuple[Any, Any, ParticipantPlan]] = {}

            def start_more() -> None:
                while len(running) < concurrency:
                    try:
                        plan = next(plan_iterator)
                    except StopIteration:
                        return
                    parent_connection, child_connection = context.Pipe(duplex=False)
                    process = context.Process(
                        target=worker_entry,
                        args=(task_for(plan), child_connection),
                        name=f"google-fitness-{plan.participant_id}",
                    )
                    process.start()
                    child_connection.close()
                    running[plan.participant_id] = (process, parent_connection, plan)

            with tqdm(
                total=len(plans), desc="Processing Google Fitness participants", unit="participant",
            ) as progress:
                start_more()
                try:
                    while running:
                        completed_id: str | None = None
                        for participant_id, (process, connection, _) in running.items():
                            if connection.poll() or not process.is_alive():
                                completed_id = participant_id
                                break
                        if completed_id is None:
                            time.sleep(0.05)
                            continue
                        process, connection, plan = running.pop(completed_id)
                        try:
                            if connection.poll():
                                result = connection.recv()
                            else:
                                result = BuildResult(
                                    participant_id=plan.participant_id,
                                    stage_dir=stage_root / plan.participant_id,
                                    ledger_path=ledger_root / f"{plan.participant_id}.json.gz",
                                    error=(
                                        "worker exited without returning a result "
                                        f"(exit code {process.exitcode})"
                                    ),
                                )
                        except EOFError:
                            result = BuildResult(
                                participant_id=plan.participant_id,
                                stage_dir=stage_root / plan.participant_id,
                                ledger_path=ledger_root / f"{plan.participant_id}.json.gz",
                                error="worker pipe closed unexpectedly (exit code {process.exitcode})",
                            )
                        finally:
                            connection.close()
                            process.join(timeout=10)
                            if process.is_alive():
                                process.kill()
                                process.join()
                        progress.update(1)
                        handle_result(plan, result)
                        start_more()
                finally:
                    for process, connection, _ in running.values():
                        connection.close()
                        if process.is_alive():
                            process.terminate()
                        process.join(timeout=10)

            status = (
                "complete"
                if summary.failed == 0 and summary.blocked == 0
                else "complete_with_errors"
            )
            preliminary = summary.basic_dict()
            state.finish_run(run_id, status, preliminary)
            summary.report = build_processing_report(state.connection, run_id)
        except BaseException:
            try:
                state.finish_run(run_id, "failed", summary.basic_dict())
            except Exception:
                pass
            raise
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
            try:
                temp_root.parent.rmdir()
            except OSError:
                pass
    return summary
