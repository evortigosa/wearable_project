"""
Wearable Data Processing and Modeling project
Participant-level orchestration, incremental planning, and atomic commits.
"""


from __future__ import annotations
import multiprocessing as mp
import os
import shutil
import traceback
import uuid
import time
import gc
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable
from tqdm import tqdm
from wearable_project import __version__
from wearable_project.exceptions import InputLayoutError, ParticipantProcessingError
from wearable_project.processing.parser import (
    ParseResult, SourceFile, canonical_month_from_name, discover_month_files,
    merge_parse_results, parse_month_file, sha256_file,
)
from wearable_project.processing.registry import REGISTRY_VERSION, get_feature_spec, is_registered_feature
from wearable_project.processing.tracker import peak_rss_bytes
from wearable_project.processing.writer import (
    DirectorySwap, FeatureOutputInfo, StateDatabase, atomic_write_csv, drop_file_cache, existing_output_is_usable,
    load_csv_records, prepare_stage, safe_feature_filename, source_signature, validate_participant_stage,
)


PARSER_VERSION = f"native-parser-{__version__}"


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
    sources_to_parse: list[SourceFile]
    effective_sources: list[SourceFile]
    existing_outputs: dict[str, FeatureOutputInfo] = field(default_factory=dict)


@dataclass(slots=True)
class ParticipantTask:
    participant_id: str
    action: str
    sources_to_parse: list[SourceFile]
    existing_dir: Path | None
    stage_dir: Path
    row_error_policy: str
    existing_outputs: dict[str, FeatureOutputInfo] = field(default_factory=dict)


@dataclass(slots=True)
class BuildResult:
    participant_id: str
    stage_dir: Path
    feature_stats: dict[str, dict[str, Any]]
    source_files_read: int = 0
    source_files_completed: int = 0
    source_bytes_read: int = 0
    outer_rows: int = 0
    apple_rows: int = 0
    payloads_decoded: int = 0
    payload_decode_failures: int = 0
    payload_items: int = 0
    unknown_features: list[str] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    output_infos: list[FeatureOutputInfo] = field(default_factory=list)
    runtime_seconds: float | None = None
    peak_rss_bytes: int | None = None
    worker_exit_code: int | None = None
    empty: bool = False
    error: str | None = None
    traceback_text: str | None = None


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
    report: dict[str, Any] | None = None

    def basic_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "discovered": self.discovered,
            "skipped": self.skipped, "incremental": self.incremental,
            "rebuilt": self.rebuilt, "committed": self.committed,
            "empty": self.empty, "blocked": self.blocked, "failed": self.failed,
            "blocks": self.blocks, "failures": self.failures,
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
        raise InputLayoutError(f"Input is not a directory: {root}")
    if any(contains_month_file(child) for child in root.iterdir() if child.is_dir()):
        return root
    children = [child for child in root.iterdir() if child.is_dir() and not child.name.startswith(".")]
    if len(children) == 1 and any(contains_month_file(child) for child in children[0].iterdir() if child.is_dir()):
        return children[0]
    raise InputLayoutError(f"No participant directories with YYYY-M.csv or YYYY-MM.csv were found under {root}")


def discover_participants(root: Path, selected: set[str] | None = None) -> dict[str, Path]:
    return {
        child.name: child for child in sorted(root.iterdir(), key=lambda item: item.name)
        if child.is_dir() and not child.name.startswith(".")
        and (selected is None or child.name in selected) and contains_month_file(child)
    }


def map_sources(sources: Iterable[SourceFile]) -> dict[str, SourceFile]:
    return {item.canonical_month: item for item in sources}


def stored_source(month: str, value: dict[str, Any]) -> SourceFile:
    return SourceFile(Path(value["filename"]), month, value["sha256"], int(value["size_bytes"]))


def plan_participant(
    participant_id: str, participant_dir: Path, sources: list[SourceFile], stored: Any,
    output_root: Path, *, mode: str, snapshot_policy: str, verify_existing_hashes: bool,
) -> ParticipantPlan:
    rebuild = lambda reason: ParticipantPlan(participant_id, participant_dir, PlanAction.REBUILD, reason, sources, sources, sources)
    if mode == "rebuild":
        return rebuild("forced rebuild")
    if stored is None:
        return rebuild("participant has no committed state")
    if stored.status not in {"complete", "complete_empty"}:
        return rebuild(f"previous state is {stored.status!r}")
    if stored.parser_version != PARSER_VERSION or stored.registry_version != REGISTRY_VERSION:
        return rebuild("parser or feature registry version changed")
    if stored.status == "complete" and not existing_output_is_usable(output_root / participant_id, stored, verify_existing_hashes):
        return rebuild("committed output is missing or invalid")

    current, old = map_sources(sources), stored.sources
    old_months, current_months = set(old), set(current)
    missing = sorted(old_months - current_months)
    added = sorted(current_months - old_months)
    changed = sorted(month for month in old_months & current_months if old[month]["sha256"] != current[month].sha256)

    if missing:
        if snapshot_policy == "strict-cumulative":
            return ParticipantPlan(participant_id, participant_dir, PlanAction.BLOCK, f"snapshot omits committed months: {missing}", sources, [], sources)
        if snapshot_policy == "authoritative":
            return rebuild(f"authoritative snapshot removed months: {missing}")
        if changed:
            return ParticipantPlan(participant_id, participant_dir, PlanAction.BLOCK, f"append-only snapshot omits old months and changes {changed}", sources, [], sources)
    if changed:
        return rebuild(f"historical monthly content changed: {changed}")
    if not added:
        return ParticipantPlan(participant_id, participant_dir, PlanAction.SKIP, "snapshot unchanged", sources, [], sources)
    max_old = max(old_months) if old_months else None
    historical = [month for month in added if max_old is not None and month <= max_old]
    if historical:
        return rebuild(f"recovered historical months: {historical}")
    effective = sources
    if missing and snapshot_policy == "append-only":
        union = {month: stored_source(month, value) for month, value in old.items()}
        union.update(current)
        effective = [union[month] for month in sorted(union)]
    return ParticipantPlan(
        participant_id, participant_dir, PlanAction.INCREMENTAL,
        f"only later months were added: {added}", sources,
        [current[month] for month in added], effective, stored.outputs,
    )


def hash_participant_sources(item: tuple[str, str]) -> tuple[str, list[SourceFile] | None, str | None]:
    """
    Hash one participant's month files. Body of the parallel pre-pass below.
    Read-only by construction: it walks a single participant directory and hashes the month files it finds.
    It never opens the state database, the tracker, or the output tree, which is what makes it safe to run
    in a separate process. Failures are returned rather than raised so the planning loop can re-raise
    them inside its own handler and keep the existing failure accounting.
    """
    participant_id, participant_dir = item
    try:
        return participant_id, discover_month_files(Path(participant_dir)), None
    except Exception as exc:
        return participant_id, None, f"{type(exc).__name__}: {exc}"


def prehash_participant_sources(
    participants: dict[str, Path], workers: int,
) -> dict[str, list[SourceFile] | str]:
    """
    Hash every participant's month files up front, in parallel.
    Returns ``{participant_id: sources}`` on success and ``{participant_id: error_text}`` on failure.
    An empty dict means the caller should fall back to hashing inline, so a pool that cannot start
    degrades to the previous sequential behavior instead of failing the run.
    Each worker holds one streaming hash state, so the memory cost per worker is negligible.
    """
    if workers <= 1 or len(participants) < 2:
        return {}
    items = [(participant_id, str(path)) for participant_id, path in participants.items()]
    results: dict[str, list[SourceFile] | str] = {}
    context = mp.get_context("spawn")
    try:
        with context.Pool(processes=workers) as pool:
            # chunksize=1 keeps scheduling dynamic. Folder sizes in this cohort span three orders of
            # magnitude, so static chunking would leave workers idle behind one oversized participant.
            for participant_id, sources, error_text in tqdm(
                pool.imap_unordered(hash_participant_sources, items, chunksize=1),
                total=len(items), desc="Hashing source months", unit="participant",
            ):
                results[participant_id] = error_text if error_text is not None else sources
    except Exception:
        return {}
    return results


def feature_filename_map(features: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for feature in features:
        filename = safe_feature_filename(feature)
        if filename in reverse and reverse[filename] != feature:
            raise ParticipantProcessingError(f"Feature filename collision: {feature!r} and {reverse[filename]!r} -> {filename}")
        mapping[feature], reverse[filename] = filename, feature
    return mapping


def build_participant(task: ParticipantTask) -> BuildResult:
    started = time.perf_counter()
    gc_was_enabled = gc.isenabled()
    # A worker processes one participant and exits. Parsed event dictionaries are overwhelmingly acyclic; disabling
    # cyclic GC prevents long full-heap scans on dense HeartRate/CGM participants while reference counting still
    # releases each feature after it is written.
    gc.disable()
    parse_results: list[ParseResult] = []
    parsed: ParseResult | None = None
    stats: dict[str, dict[str, Any]] = {}
    manifest: dict[str, FeatureOutputInfo] = {}
    source_files_read = 0
    source_files_completed = 0
    source_bytes_read = 0
    build_result: BuildResult
    try:
        # Keep pandas and feature-cleaning machinery out of the parent process. Under spawn, this materially reduces
        # peak memory because only workers import the heavy tabular stack.
        from wearable_project.processing.cleaners import clean_feature

        incremental = task.action == PlanAction.INCREMENTAL.value
        prepare_stage(task.stage_dir, task.existing_dir, incremental)
        manifest = dict(task.existing_outputs) if incremental else {}

        for source in task.sources_to_parse:
            source_files_read += 1
            source_bytes_read += source.size_bytes
            partial = ParseResult()
            try:
                parse_month_file(
                    source, task.participant_id, row_error_policy=task.row_error_policy, result=partial
                )
                source_files_completed += 1
            finally:
                # parse_month_file mutates the supplied result before raising, allowing failed participants to
                # retain partial telemetry.
                parse_results.append(partial)
                drop_file_cache(source.path)

        parsed = merge_parse_results(parse_results)
        del parse_results
        filenames = feature_filename_map(parsed.records_by_feature)
        for feature in sorted(list(parsed.records_by_feature)):
            incoming = parsed.records_by_feature.pop(feature)
            destination = task.stage_dir / filenames[feature]
            if incremental and destination.exists():
                existing = load_csv_records(destination)
                for row in existing:
                    # participant_id and feature are encoded by the directory and filename in compact output files.
                    # Reattach them only for the internal reconciliation pass.
                    row["participant_id"] = task.participant_id
                    row["feature"] = feature
                    row["_from_existing_output"] = True
                combined = existing + incoming
            else:
                combined = incoming

            spec = get_feature_spec(feature, {key for row in combined for key in row})
            clean_result = clean_feature(feature, combined)
            registered = is_registered_feature(feature)
            if not registered:
                schema_status = "unknown_feature"
            elif feature in parsed.schema_warning_features:
                schema_status = "warning"
            else:
                schema_status = "known"
            stats[feature] = {
                "registered_policy": registered,
                "feature_family": spec.family.value,
                "schema_status": schema_status,
                "input_rows": clean_result.input_rows,
                "output_rows": clean_result.output_rows,
                "represented_occurrences": clean_result.represented_occurrences,
                "exact_duplicates_removed": clean_result.exact_duplicates_removed,
                "revisions_resolved": clean_result.revisions_resolved,
                "unresolved_conflicts": clean_result.unresolved_conflicts,
                "invalid_timestamp_rows": clean_result.invalid_timestamp_rows,
            }
            if clean_result.dataframe.empty:
                destination.unlink(missing_ok=True)
                manifest.pop(feature, None)
                del combined, clean_result, incoming
                continue
            atomic_write_csv(clean_result.dataframe, destination)
            output_hash = sha256_file(destination)
            manifest[feature] = FeatureOutputInfo(
                feature, destination.name, clean_result.output_rows,
                destination.stat().st_size, output_hash, clean_result.represented_occurrences,
            )
            drop_file_cache(destination)
            del combined, clean_result, incoming

        build_result = BuildResult(
            participant_id=task.participant_id,
            stage_dir=task.stage_dir,
            feature_stats=stats,
            source_files_read=source_files_read,
            source_files_completed=source_files_completed,
            source_bytes_read=source_bytes_read,
            outer_rows=parsed.outer_rows_seen,
            apple_rows=parsed.apple_rows_seen,
            payloads_decoded=parsed.payloads_decoded,
            payload_decode_failures=parsed.payload_decode_failures,
            payload_items=parsed.payload_items_seen,
            unknown_features=sorted(parsed.unknown_features),
            diagnostics=[asdict(item) for item in parsed.diagnostics],
            output_infos=[manifest[key] for key in sorted(manifest)],
            empty=not any(task.stage_dir.glob("*.csv")),
        )
    except Exception as exc:
        if parsed is None:
            parsed = merge_parse_results(parse_results)
        build_result = BuildResult(
            participant_id=task.participant_id,
            stage_dir=task.stage_dir,
            feature_stats=stats,
            source_files_read=source_files_read,
            source_files_completed=source_files_completed,
            source_bytes_read=source_bytes_read,
            outer_rows=parsed.outer_rows_seen,
            apple_rows=parsed.apple_rows_seen,
            payloads_decoded=parsed.payloads_decoded,
            payload_decode_failures=parsed.payload_decode_failures,
            payload_items=parsed.payload_items_seen,
            unknown_features=sorted(parsed.unknown_features),
            diagnostics=[asdict(item) for item in parsed.diagnostics],
            output_infos=[manifest[key] for key in sorted(manifest)],
            error=str(exc),
            traceback_text=traceback.format_exc(),
        )
    finally:
        if gc_was_enabled:
            gc.enable()

    build_result.runtime_seconds = time.perf_counter() - started
    build_result.peak_rss_bytes = peak_rss_bytes()
    return build_result


def worker_entry(task: ParticipantTask, connection: Any) -> None:
    """Run exactly one participant in a fresh spawned process."""
    try:
        connection.send(build_participant(task))
    except BaseException as exc:
        connection.send(BuildResult(
            participant_id=task.participant_id,
            stage_dir=task.stage_dir,
            feature_stats={},
            runtime_seconds=None,
            peak_rss_bytes=peak_rss_bytes(),
            error=str(exc),
            traceback_text=traceback.format_exc(),
        ))
    finally:
        connection.close()


def remove_output_then_commit(final_dir: Path, backup_dir: Path, commit: Any) -> None:
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
    mode: str = "auto", snapshot_policy: str = "strict-cumulative",
    row_error_policy: str = "fail-participant", selected_participants: set[str] | None = None,
    verify_existing_hashes: bool = False, fail_fast: bool = False, state_file: Path | None = None,
) -> RunSummary:
    run_started = time.perf_counter()
    export_root = resolve_export_root(input_root)
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_file = state_file or output_root / ".wearable_state.sqlite"
    run_id = uuid.uuid4().hex
    temp_parent = output_root / ".wearable_tmp"
    # Participant commits are transactional; staging directories from an interrupted earlier process are never
    # authoritative and can be removed safely before a new single-node run starts.
    shutil.rmtree(temp_parent, ignore_errors=True)
    temp_root = temp_parent / run_id
    stage_root, backup_root = temp_root / "staging", temp_root / "backups"
    stage_root.mkdir(parents=True, exist_ok=True)
    participants = discover_participants(export_root, selected_participants)
    summary = RunSummary(run_id, discovered=len(participants))
    workers_requested = int(workers)
    workers_effective = max(1, min(workers_requested, max(1, (os.cpu_count() or 2) - 1)))
    max_in_flight_effective = max(1, int(max_in_flight or workers_effective))
    concurrency = min(workers_effective, max_in_flight_effective)
    # Run the hashing pre-pass before the state database is opened. No SQLite handle exists in this process
    # yet, so nothing database-related can be inherited by the pool workers.
    prehashed = prehash_participant_sources(participants, concurrency)

    with StateDatabase(state_file) as state:
        state.begin_run(run_id, export_root, output_root)
        tracker = state.tracker
        tracker.begin_run(
            run_id, mode=mode, snapshot_policy=snapshot_policy,
            row_error_policy=row_error_policy, workers_requested=workers_requested,
            workers_effective=workers_effective, max_in_flight=max_in_flight_effective,
        )
        try:
            plans: list[ParticipantPlan] = []
            if snapshot_policy == "strict-cumulative" and selected_participants is None:
                missing_participants = sorted(state.participant_ids().difference(participants))
                for participant_id in missing_participants:
                    reason = "participant is absent from the new cumulative snapshot"
                    summary.blocked += 1
                    summary.blocks[participant_id] = reason
                    tracker.record_plan(
                        run_id, participant_id, action="missing_participant", reason=reason,
                        final_status="blocked", source_files_discovered=0,
                        source_files_planned=0, source_bytes_discovered=0,
                    )

            for participant_id, participant_dir in tqdm(
                participants.items(), desc="Planning participants", unit="participant"
            ):
                planning_started = time.perf_counter()
                candidates = [
                    path for path in participant_dir.iterdir()
                    if path.is_file() and canonical_month_from_name(path.name) is not None
                ]
                candidate_bytes = 0
                for path in candidates:
                    try:
                        candidate_bytes += path.stat().st_size
                    except OSError:
                        pass
                try:
                    # pop, not get: releases each participant's SourceFile list as soon as it is planned, so
                    # the pre-pass results do not stay resident for the whole run.
                    precomputed = prehashed.pop(participant_id, None)
                    if isinstance(precomputed, str):
                        raise ParticipantProcessingError(precomputed)
                    sources = precomputed if precomputed is not None else discover_month_files(participant_dir)
                    plan = plan_participant(
                        participant_id, participant_dir, sources, state.get_participant(participant_id),
                        output_root, mode=mode, snapshot_policy=snapshot_policy,
                        verify_existing_hashes=verify_existing_hashes,
                    )
                except Exception as exc:
                    message = f"Source discovery or planning failed: {exc}"
                    summary.failed += 1
                    summary.failures[participant_id] = message
                    tracker.record_plan(
                        run_id, participant_id, action="planning_error", reason=message,
                        final_status="failed", source_files_discovered=len(candidates),
                        source_files_planned=0, source_bytes_discovered=candidate_bytes,
                    )
                    planning_result = BuildResult(
                        participant_id=participant_id, stage_dir=stage_root / participant_id,
                        feature_stats={}, runtime_seconds=time.perf_counter() - planning_started,
                        error=message, traceback_text=traceback.format_exc(),
                    )
                    state.mark_failed(participant_id, message)
                    tracker.record_participant_result(
                        run_id, participant_id, final_status="failed", result=planning_result,
                        worker_exit_code=None, committed_output_size_bytes=0,
                    )
                    if fail_fast:
                        raise ParticipantProcessingError(f"{participant_id}: {message}") from exc
                    continue

                initial_status = (
                    "skipped" if plan.action == PlanAction.SKIP else
                    "blocked" if plan.action == PlanAction.BLOCK else "planned"
                )
                tracker.record_plan(
                    run_id, participant_id, action=plan.action.value, reason=plan.reason,
                    final_status=initial_status, source_files_discovered=len(sources),
                    source_files_planned=len(plan.sources_to_parse),
                    source_bytes_discovered=sum(source.size_bytes for source in sources),
                )
                if plan.action == PlanAction.SKIP:
                    summary.skipped += 1
                elif plan.action == PlanAction.BLOCK:
                    summary.blocked += 1
                    summary.blocks[participant_id] = plan.reason
                else:
                    plans.append(plan)
                    state.mark_in_progress(participant_id, PARSER_VERSION, REGISTRY_VERSION)

            # Process the largest participant snapshots first. On memory-limited single nodes this avoids
            # accumulating page cache and allocator residue from many smaller participants before the largest task.
            plans.sort(
                key=lambda item: sum(source.size_bytes for source in item.sources_to_parse), reverse=True,
            )

            context = mp.get_context("spawn")
            task_iterator = iter(plans)
            running: dict[str, tuple[Any, Any, ParticipantPlan]] = {}

            def make_task(plan: ParticipantPlan) -> ParticipantTask:
                return ParticipantTask(
                    plan.participant_id, plan.action.value, plan.sources_to_parse,
                    output_root / plan.participant_id if plan.action == PlanAction.INCREMENTAL else None,
                    stage_root / plan.participant_id, row_error_policy, plan.existing_outputs,
                )

            def start_more() -> None:
                while len(running) < concurrency:
                    try:
                        plan = next(task_iterator)
                    except StopIteration:
                        break
                    parent_connection, child_connection = context.Pipe(duplex=False)
                    process = context.Process(
                        target=worker_entry, args=(make_task(plan), child_connection),
                        name=f"wearable-{plan.participant_id}",
                    )
                    tracker.mark_participant_started(run_id, plan.participant_id)
                    process.start()
                    child_connection.close()
                    running[plan.participant_id] = (process, parent_connection, plan)

            with tqdm(total=len(plans), desc="Processing participants", unit="participant") as progress:
                start_more()
                try:
                    while running:
                        completed_id = None
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
                                    feature_stats={},
                                    error=f"Worker exited with code {process.exitcode} without returning a result",
                                )
                        except EOFError:
                            result = BuildResult(
                                participant_id=plan.participant_id,
                                stage_dir=stage_root / plan.participant_id,
                                feature_stats={},
                                error=f"Worker pipe closed unexpectedly (exit code {process.exitcode})",
                            )
                        finally:
                            connection.close()
                            process.join(timeout=5)
                            if process.is_alive():
                                process.kill()
                                process.join()
                        result.worker_exit_code = process.exitcode
                        progress.update(1)

                        if result.error:
                            summary.failed += 1
                            summary.failures[plan.participant_id] = result.error
                            state.mark_failed(plan.participant_id, result.error)
                            tracker.record_participant_result(
                                run_id, plan.participant_id, final_status="failed", result=result,
                                worker_exit_code=result.worker_exit_code, committed_output_size_bytes=0,
                            )
                            shutil.rmtree(result.stage_dir, ignore_errors=True)
                            if fail_fast:
                                raise ParticipantProcessingError(
                                    f"{plan.participant_id}: {result.error}\n"
                                    f"{result.traceback_text or ''}"
                                )
                            start_more()
                            continue

                        final_dir = output_root / plan.participant_id
                        backup_dir = backup_root / plan.participant_id
                        try:
                            if result.empty:
                                remove_output_then_commit(
                                    final_dir, backup_dir,
                                    lambda: state.commit_participant(
                                        plan.participant_id, PARSER_VERSION, REGISTRY_VERSION,
                                        source_signature(plan.effective_sources),
                                        plan.effective_sources, [], status="complete_empty",
                                    ),
                                )
                                shutil.rmtree(result.stage_dir, ignore_errors=True)
                                summary.empty += 1
                                final_status = "complete_empty"
                                output_size = 0
                            else:
                                outputs = validate_participant_stage(
                                    result.stage_dir, plan.participant_id, result.output_infos
                                )
                                swap = DirectorySwap(final_dir, result.stage_dir, backup_dir)
                                swap.apply()
                                try:
                                    state.commit_participant(
                                        plan.participant_id, PARSER_VERSION, REGISTRY_VERSION,
                                        source_signature(plan.effective_sources),
                                        plan.effective_sources, outputs,
                                    )
                                except Exception:
                                    swap.rollback()
                                    raise
                                swap.finalize()
                                summary.committed += 1
                                final_status = "committed"
                                output_size = sum(item.size_bytes for item in outputs)
                            if plan.action == PlanAction.INCREMENTAL:
                                summary.incremental += 1
                            else:
                                summary.rebuilt += 1
                            tracker.record_participant_result(
                                run_id, plan.participant_id, final_status=final_status, result=result,
                                worker_exit_code=result.worker_exit_code,
                                committed_output_size_bytes=output_size,
                            )
                        except Exception as exc:
                            summary.failed += 1
                            summary.failures[plan.participant_id] = str(exc)
                            state.mark_failed(plan.participant_id, str(exc))
                            tracker.record_participant_result(
                                run_id, plan.participant_id, final_status="failed", result=result,
                                worker_exit_code=result.worker_exit_code, committed_output_size_bytes=0,
                                error_message=str(exc), traceback_text=traceback.format_exc(),
                            )
                            shutil.rmtree(result.stage_dir, ignore_errors=True)
                            if fail_fast:
                                raise
                        start_more()
                finally:
                    for process, connection, _ in running.values():
                        connection.close()
                        if process.is_alive():
                            process.terminate()
                        process.join(timeout=5)

            status = "complete" if not summary.failed and not summary.blocked else "complete_with_errors"
            tracker.finish_run(
                run_id, status=status, wall_clock_seconds=time.perf_counter() - run_started
            )
            summary.report = tracker.build_report(run_id)
            state.finish_run(run_id, status, summary.report)
        except BaseException:
            tracker.finish_run(
                run_id, status="failed", wall_clock_seconds=time.perf_counter() - run_started
            )
            summary.report = tracker.build_report(run_id)
            state.finish_run(run_id, "failed", summary.report)
            raise
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
            try:
                temp_parent.rmdir()
            except OSError:
                pass
    return summary
