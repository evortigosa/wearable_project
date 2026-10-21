"""
Wearable Data Processing and Modeling project
Transactional participant-level Google Fitness curation pipeline.
"""


from __future__ import annotations
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sqlite3
import time
import traceback
from typing import Any, Iterable
from tqdm import tqdm
from wearable_project.curation.google_fitness.decisions import decisions_fingerprint
from wearable_project.curation.google_fitness.engine import (
    ENGINE_VERSION, CuratedFeatureInfo, curate_feature_file, feature_policy_fingerprint, sha256_file,
    validate_curated_feature_file,
)
from wearable_project.curation.google_fitness.evidence import evidence_fingerprint
from wearable_project.curation.google_fitness.guidance import guidance_fingerprint
from wearable_project.curation.google_fitness.registry import get_policy, registry_fingerprint
from wearable_project.curation.google_fitness.rules import rules_fingerprint
from wearable_project.curation.google_fitness.state import (
    CURATION_STATE_FILENAME, CuratedOutputState, CurationRunSummary, CurationStateDatabase, NativeFileState,
    format_curation_report,
)
from wearable_project.exceptions import InputLayoutError, OutputValidationError
from wearable_project.processing.google_fitness.registry import SOURCE_KEY
from wearable_project.processing.google_fitness.tracker import peak_rss_bytes
from wearable_project.processing.google_fitness.writer import DirectorySwap, STATE_FILENAME as NATIVE_STATE_FILENAME


PIPELINE_VERSION = "google-fitness-curation-pipeline-cwp4-0.1.0"
TEMP_DIRNAME = ".wearable_curation_tmp"


@dataclass(frozen=True, slots=True)
class NativeFeatureInput:
    feature: str
    filename: str
    path: Path
    row_count: int
    size_bytes: int
    sha256: str
    policy_fingerprint: str


@dataclass(frozen=True, slots=True)
class ParticipantPlan:
    participant_id: str
    native_dir: Path
    action: str
    reason: str
    features: tuple[NativeFeatureInput, ...]
    native_signature: str
    policy_signature: str


@dataclass(slots=True)
class CurationTask:
    participant_id: str
    stage_dir: Path
    features: tuple[NativeFeatureInput, ...]


@dataclass(slots=True)
class CurationBuildResult:
    participant_id: str
    stage_dir: Path
    outputs: list[CuratedFeatureInfo] = field(default_factory=list)
    runtime_seconds: float | None = None
    peak_rss_bytes: int | None = None
    error: str | None = None
    traceback_text: str | None = None


def _native_state_connection(native_root: Path) -> sqlite3.Connection:
    state_path = native_root / NATIVE_STATE_FILENAME
    if not state_path.is_file():
        raise InputLayoutError(f"managed Google Fitness native root lacks {NATIVE_STATE_FILENAME}: {native_root}")
    wal = state_path.with_name(state_path.name + "-wal")
    if wal.is_file() and wal.stat().st_size:
        raise InputLayoutError(f"native state has an active WAL; finish native processing before curation: {wal}")
    connection = sqlite3.connect(f"file:{state_path}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        metadata = {
            row["key"]: row["value"] for row in connection.execute("SELECT key,value FROM metadata")
        }
    except sqlite3.Error as exc:
        connection.close()
        raise InputLayoutError(f"invalid Google Fitness native state: {state_path}") from exc
    if metadata.get("source_key") != SOURCE_KEY:
        connection.close()
        raise InputLayoutError(f"native state source is {metadata.get('source_key')!r}, expected {SOURCE_KEY!r}")
    return connection


def validate_native_input_root(
    input_root: Path, *, allow_unmanaged_native_root: bool = False,
) -> Path:
    source = Path(input_root).expanduser().resolve()
    if not source.is_dir():
        raise InputLayoutError(f"native input root is not a directory: {source}")
    if (source / CURATION_STATE_FILENAME).is_file():
        raise InputLayoutError(f"input appears to be a curated root: {source}")
    if (source / NATIVE_STATE_FILENAME).is_file():
        with _native_state_connection(source):
            pass
        return source
    if not allow_unmanaged_native_root:
        raise InputLayoutError(
            f"native input root lacks {NATIVE_STATE_FILENAME}: {source}; "
            "use --allow-unmanaged-native-root only for an intentional test copy"
        )
    return source


def _ensure_disjoint_roots(
    input_native: Path, output: Path, *, allow_unmanaged_native_root: bool,
) -> tuple[Path, Path]:
    native_root = validate_native_input_root(input_native, allow_unmanaged_native_root=allow_unmanaged_native_root)
    output_root = Path(output).expanduser().resolve()
    if (
        native_root == output_root
        or native_root in output_root.parents
        or output_root in native_root.parents
    ):
        raise InputLayoutError(
            f"native and curated roots must be distinct and non-nested: native={native_root}, curated={output_root}"
        )
    if (output_root / NATIVE_STATE_FILENAME).is_file():
        raise InputLayoutError(f"curated output points to a native processing root: {output_root}")
    return native_root, output_root


def discover_native_participants(native_root: Path, selected: set[str] | None = None,) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for child in sorted(native_root.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if selected is not None and child.name not in selected:
            continue
        if any(path.is_file() and path.suffix.lower() == ".csv" for path in child.iterdir()):
            result[child.name] = child
    return result


def _native_manifest(native_root: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    state_path = native_root / NATIVE_STATE_FILENAME
    if not state_path.is_file():
        return {}, {
            "managed": False,
            "source_occurrences": None,
            "malformed_candidates": None,
            "parser_diagnostics": None,
            "reconciliation_decisions": None,
        }
    with _native_state_connection(native_root) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(feature_outputs)")}
        required = {"participant_id", "feature", "filename", "row_count", "size_bytes", "sha256"}
        if not required.issubset(columns):
            raise InputLayoutError("native state feature_outputs schema is incomplete")
        manifest = {
            (row["participant_id"], row["filename"]): dict(row)
            for row in connection.execute(
                "SELECT participant_id,feature,filename,row_count,size_bytes,sha256 FROM feature_outputs"
            )
        }
        def count(table: str) -> int | None:
            try:
                return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            except sqlite3.Error:
                return None
        diagnostics = {
            "managed": True,
            "source_occurrences": count("source_occurrences"),
            "malformed_candidates": count("malformed_candidates"),
            "parser_diagnostics": count("parser_diagnostics"),
            "reconciliation_decisions": count("reconciliation_decisions"),
        }
        return manifest, diagnostics


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def discover_native_features(
    participant_id: str, participant_dir: Path, manifest: dict[tuple[str, str], dict[str, Any]], *,
    verify_native_hashes: bool,
) -> tuple[NativeFeatureInput, ...]:
    paths = sorted(participant_dir.glob("*.csv"), key=lambda path: path.name)
    expected_names = {
        filename for (stored_participant, filename) in manifest if stored_participant == participant_id
    }
    actual_names = {path.name for path in paths}
    if expected_names and actual_names != expected_names:
        raise OutputValidationError(
            f"native file set differs from state for {participant_id}: "
            f"expected={sorted(expected_names)}, actual={sorted(actual_names)}"
        )
    features: list[NativeFeatureInput] = []
    for path in paths:
        stored = manifest.get((participant_id, path.name))
        size = path.stat().st_size
        if stored is not None:
            if size != int(stored["size_bytes"]):
                raise OutputValidationError(f"native file size differs from state: {path}")
            row_count = int(stored["row_count"])
            digest = str(stored["sha256"])
            if verify_native_hashes:
                observed = sha256_file(path)
                if observed != digest:
                    raise OutputValidationError(f"native file hash differs from state: {path}")
        else:
            row_count = _count_csv_rows(path)
            digest = sha256_file(path)
        policy = get_policy(path.stem)
        features.append(
            NativeFeatureInput(
                feature=path.stem,
                filename=path.name,
                path=path,
                row_count=row_count,
                size_bytes=size,
                sha256=digest,
                policy_fingerprint=feature_policy_fingerprint(policy),
            )
        )
    return tuple(features)


def _signature(features: Iterable[NativeFeatureInput], *, include_policy: bool) -> str:
    payload: list[dict[str, Any]] = []
    for item in sorted(features, key=lambda value: value.filename):
        row: dict[str, Any] = {
            "feature": item.feature,
            "filename": item.filename,
            "rows": item.row_count,
            "size": item.size_bytes,
            "sha256": item.sha256,
        }
        if include_policy:
            row["policy_fingerprint"] = item.policy_fingerprint
        payload.append(row)
    if include_policy:
        payload.extend(
            [
                {"registry": registry_fingerprint()},
                {"evidence": evidence_fingerprint()},
                {"rules": rules_fingerprint()},
                {"decisions": decisions_fingerprint()},
                {"guidance": guidance_fingerprint()},
                {"engine": ENGINE_VERSION},
                {"pipeline": PIPELINE_VERSION},
            ]
        )
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _stored_outputs_usable(output_dir: Path, stored: Any, *, verify_hashes: bool,) -> bool:
    if stored.status != "complete" or not output_dir.is_dir() or not stored.outputs:
        return False
    expected = {item.filename for item in stored.outputs.values()}
    actual = {path.name for path in output_dir.glob("*.csv")}
    if expected != actual:
        return False
    for item in stored.outputs.values():
        path = output_dir / item.filename
        if not path.is_file() or path.stat().st_size != item.size_bytes:
            return False
        if verify_hashes and sha256_file(path) != item.sha256:
            return False
    return True


def _recover_stale_transactions(
    output_root: Path, temporary_root: Path, state: CurationStateDatabase,
) -> tuple[str, ...]:
    """
    Recover or discard artifacts left by an interrupted parent process. A backup is removed only when the currently
    published directory still matches committed state. Otherwise, the backup is restored before the participant is
    replanned. Uncommitted staging directories are always discarded. Concurrent curation runs against one output
    root are intentionally unsupported.
    """

    actions: list[str] = []
    if not temporary_root.is_dir():
        return ()
    for run_dir in sorted(temporary_root.iterdir(), key=lambda path: path.name):
        if not run_dir.is_dir():
            run_dir.unlink(missing_ok=True)
            continue
        for backup in sorted(run_dir.glob(".*.backup"), key=lambda path: path.name):
            participant_id = backup.name[1:-len(".backup")]
            final_dir = output_root / participant_id
            stored = state.get_participant(participant_id)
            published_is_committed = (
                stored is not None and _stored_outputs_usable(final_dir, stored, verify_hashes=True)
            )
            if published_is_committed:
                shutil.rmtree(backup, ignore_errors=True)
                actions.append(f"discarded_redundant_backup:{participant_id}")
            else:
                shutil.rmtree(final_dir, ignore_errors=True)
                os.replace(backup, final_dir)
                actions.append(f"restored_backup:{participant_id}")
        shutil.rmtree(run_dir, ignore_errors=True)
    try:
        temporary_root.rmdir()
    except OSError:
        pass
    return tuple(actions)


def plan_participant(
    participant_id: str, participant_dir: Path, features: tuple[NativeFeatureInput, ...], state: CurationStateDatabase,
    output_root: Path, *, mode: str, verify_existing_hashes: bool,
) -> ParticipantPlan:
    native_signature = _signature(features, include_policy=False)
    policy_signature = _signature(features, include_policy=True)
    stored = state.get_participant(participant_id)
    if mode == "rebuild":
        reason = "forced rebuild"
    elif stored is None:
        reason = "participant has no committed curation state"
    elif stored.status != "complete":
        reason = f"previous curation state is {stored.status!r}"
    elif stored.engine_version != ENGINE_VERSION:
        reason = "curation engine version changed"
    elif stored.native_signature != native_signature:
        reason = "native participant input changed"
    elif stored.policy_signature != policy_signature:
        reason = "relevant feature policy changed"
    elif not _stored_outputs_usable(
        output_root / participant_id, stored, verify_hashes=verify_existing_hashes
    ):
        reason = "curated participant output is missing or invalid"
    else:
        return ParticipantPlan(
            participant_id, participant_dir, "skip", "native input and policies unchanged",
            features, native_signature, policy_signature,
        )
    return ParticipantPlan(
        participant_id, participant_dir, "rebuild", reason, features, native_signature, policy_signature,
    )


def _build_participant(task: CurationTask) -> CurationBuildResult:
    started = time.perf_counter()
    try:
        shutil.rmtree(task.stage_dir, ignore_errors=True)
        task.stage_dir.mkdir(parents=True, exist_ok=False)
        outputs: list[CuratedFeatureInfo] = []
        for native in task.features:
            result = curate_feature_file(
                native.path,
                task.stage_dir / native.filename,
                native.feature,
            )
            info = result.info
            if info.native_rows != native.row_count:
                raise OutputValidationError(
                    f"{native.path}: state rows={native.row_count}, curation rows={info.native_rows}"
                )
            if info.native_sha256 != native.sha256:
                raise OutputValidationError(f"{native.path}: native hash changed or differs from state")
            if info.curated_rows != info.native_rows:
                raise OutputValidationError(f"{native.path}: curation changed row count")
            outputs.append(info)
        expected = {item.filename for item in task.features}
        actual = {path.name for path in task.stage_dir.glob("*.csv")}
        if expected != actual:
            raise OutputValidationError(
                f"{task.participant_id}: staged file set mismatch: expected={sorted(expected)}, actual={sorted(actual)}"
            )
        return CurationBuildResult(
            participant_id=task.participant_id,
            stage_dir=task.stage_dir,
            outputs=outputs,
            runtime_seconds=time.perf_counter() - started,
            peak_rss_bytes=peak_rss_bytes(),
        )
    except Exception as exc:
        return CurationBuildResult(
            participant_id=task.participant_id,
            stage_dir=task.stage_dir,
            runtime_seconds=time.perf_counter() - started,
            peak_rss_bytes=peak_rss_bytes(),
            error=f"{type(exc).__name__}: {exc}",
            traceback_text=traceback.format_exc(),
        )


def _native_state(item: NativeFeatureInput) -> NativeFileState:
    return NativeFileState(
        feature=item.feature,
        filename=item.filename,
        row_count=item.row_count,
        size_bytes=item.size_bytes,
        sha256=item.sha256,
        policy_fingerprint=item.policy_fingerprint,
    )


def _output_state(info: CuratedFeatureInfo) -> CuratedOutputState:
    return CuratedOutputState(
        feature=info.feature,
        filename=info.filename,
        native_rows=info.native_rows,
        curated_rows=info.curated_rows,
        size_bytes=info.size_bytes,
        sha256=info.curated_sha256,
        native_sha256=info.native_sha256,
        policy_fingerprint=info.policy_fingerprint,
        active_derived_columns=info.active_derived_columns,
        pass_rows=info.pass_rows,
        review_rows=info.review_rows,
        exclude_default_rows=info.exclude_default_rows,
        included_by_default_rows=info.included_by_default_rows,
        excluded_by_default_rows=info.excluded_by_default_rows,
        canonical_value_rows=info.canonical_value_rows,
        flag_counts=info.flag_counts,
        acquisition_counts=info.acquisition_counts,
        derivation_counts=info.derivation_counts,
        tracking_counts=info.tracking_counts,
        step_count_role_counts=info.step_count_role_counts,
        unit_status_counts=info.unit_status_counts,
        unit_evidence_counts=info.unit_evidence_counts,
    )


def _validate_staged_outputs(
    stage_dir: Path, features: tuple[NativeFeatureInput, ...], outputs: Iterable[CuratedFeatureInfo],
) -> None:
    outputs = tuple(outputs)
    expected_names = {item.filename for item in features}
    if {item.filename for item in outputs} != expected_names:
        raise OutputValidationError("worker output manifest differs from native feature set")
    if {path.name for path in stage_dir.glob("*.csv")} != expected_names:
        raise OutputValidationError("staged CSV file set differs from native feature set")
    native_by_feature = {item.feature: item for item in features}
    for info in outputs:
        native = native_by_feature[info.feature]
        path = stage_dir / info.filename
        if path.stat().st_size != info.size_bytes or sha256_file(path) != info.curated_sha256:
            raise OutputValidationError(f"staged curated file differs from worker manifest: {path}")
        count = validate_curated_feature_file(
            native.path,
            path,
            info.feature,
            expected_derived_columns=info.active_derived_columns,
        )
        if count != info.curated_rows:
            raise OutputValidationError(f"staged row count differs for {info.feature}")


def curate_dataset(
    input_native: Path, output: Path, *, workers: int = 4, max_in_flight: int | None = None, mode: str = "auto",
    selected_participants: set[str] | None = None, verify_existing_hashes: bool = False,
    verify_native_hashes: bool = False, fail_fast: bool = False, state_file: Path | None = None,
    allow_unmanaged_native_root: bool = False,
) -> CurationRunSummary:
    if mode not in {"auto", "rebuild"}:
        raise ValueError(f"unsupported curation mode: {mode}")
    native_root, output_root = _ensure_disjoint_roots(
        input_native, output,
        allow_unmanaged_native_root=allow_unmanaged_native_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = Path(state_file).expanduser().resolve() if state_file else output_root / CURATION_STATE_FILENAME
    if state_path == native_root / NATIVE_STATE_FILENAME or native_root in state_path.parents:
        raise InputLayoutError("curation state cannot be stored inside the native root")
    existing_items = [item for item in output_root.iterdir() if item.name != TEMP_DIRNAME]
    if existing_items and not state_path.is_file():
        raise InputLayoutError(f"nonempty curated output root lacks its curation state database: {output_root}")

    participants = discover_native_participants(native_root, selected_participants)
    manifest, native_diagnostics = _native_manifest(native_root)
    worker_count = max(1, min(int(workers), os.cpu_count() or 1, len(participants) or 1))
    inflight_limit = max(1, int(max_in_flight or worker_count))
    temporary_root = output_root / TEMP_DIRNAME
    temporary_root.mkdir(parents=True, exist_ok=True)
    configuration = {
        "mode": mode,
        "workers_requested": int(workers),
        "workers_effective": worker_count,
        "max_in_flight": inflight_limit,
        "verify_existing_hashes": bool(verify_existing_hashes),
        "verify_native_hashes": bool(verify_native_hashes),
        "fail_fast": bool(fail_fast),
        "engine_version": ENGINE_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "registry_fingerprint": registry_fingerprint(),
        "evidence_fingerprint": evidence_fingerprint(),
        "rules_fingerprint": rules_fingerprint(),
        "decisions_fingerprint": decisions_fingerprint(),
        "guidance_fingerprint": guidance_fingerprint(),
        "native_input_diagnostics": native_diagnostics,
    }

    with CurationStateDatabase(state_path) as state:
        interrupted_runs = state.mark_interrupted_runs()
        recovery_actions = _recover_stale_transactions(output_root, temporary_root, state)
        temporary_root.mkdir(parents=True, exist_ok=True)
        configuration["interrupted_runs_recovered"] = list(interrupted_runs)
        configuration["stale_transaction_recovery"] = list(recovery_actions)
        run_id = state.begin_run(native_root, output_root, configuration)
        summary = CurationRunSummary(run_id=run_id, discovered=len(participants))
        plans: list[ParticipantPlan] = []
        try:
            for participant_id, participant_dir in participants.items():
                features = discover_native_features(
                    participant_id,
                    participant_dir,
                    manifest,
                    verify_native_hashes=verify_native_hashes,
                )
                if not features:
                    continue
                plan = plan_participant(
                    participant_id,
                    participant_dir,
                    features,
                    state,
                    output_root,
                    mode=mode,
                    verify_existing_hashes=verify_existing_hashes,
                )
                if plan.action == "skip":
                    summary.skipped += 1
                    stored = state.get_participant(participant_id)
                    stored_outputs = tuple(stored.outputs.values()) if stored else ()
                    state.record_run_participant(
                        run_id,
                        participant_id,
                        planned_action="skip",
                        plan_reason=plan.reason,
                        final_status="skipped",
                        native_files=len(features),
                        native_rows=sum(item.row_count for item in features),
                        curated_rows=sum(item.curated_rows for item in stored_outputs),
                        output_size_bytes=sum(item.size_bytes for item in stored_outputs),
                    )
                else:
                    summary.rebuilt += 1
                    state.mark_in_progress(participant_id)
                    plans.append(plan)

            ctx = mp.get_context("spawn")
            pending_plans = iter(plans)
            futures: dict[Future[CurationBuildResult], ParticipantPlan] = {}
            aborted = False
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
                def submit_next() -> bool:
                    try:
                        plan = next(pending_plans)
                    except StopIteration:
                        return False
                    task = CurationTask(
                        participant_id=plan.participant_id,
                        stage_dir=temporary_root / run_id / plan.participant_id,
                        features=plan.features,
                    )
                    futures[executor.submit(_build_participant, task)] = plan
                    return True

                while len(futures) < inflight_limit and submit_next():
                    pass
                progress = tqdm(total=len(plans), desc="Curating Google Fitness participants", unit="participant")
                try:
                    while futures:
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            plan = futures.pop(future)
                            result = future.result()
                            if result.error:
                                summary.failed += 1
                                summary.failures[plan.participant_id] = result.error
                                state.mark_failed(plan.participant_id, result.error)
                                state.record_run_participant(
                                    run_id,
                                    plan.participant_id,
                                    planned_action=plan.action,
                                    plan_reason=plan.reason,
                                    final_status="failed",
                                    native_files=len(plan.features),
                                    native_rows=sum(item.row_count for item in plan.features),
                                    runtime_seconds=result.runtime_seconds,
                                    peak_rss_bytes=result.peak_rss_bytes,
                                    error_message=result.error,
                                    traceback_text=result.traceback_text,
                                )
                                shutil.rmtree(result.stage_dir, ignore_errors=True)
                                aborted = aborted or fail_fast
                            else:
                                final_dir = output_root / plan.participant_id
                                backup_dir = temporary_root / run_id / f".{plan.participant_id}.backup"
                                swap = DirectorySwap(final_dir, result.stage_dir, backup_dir)
                                try:
                                    _validate_staged_outputs(result.stage_dir, plan.features, result.outputs)
                                    swap.apply()
                                    output_states = tuple(_output_state(item) for item in result.outputs)
                                    state.commit_participant(
                                        plan.participant_id,
                                        plan.native_signature,
                                        plan.policy_signature,
                                        tuple(_native_state(item) for item in plan.features),
                                        output_states,
                                    )
                                    state.record_run_participant(
                                        run_id,
                                        plan.participant_id,
                                        planned_action=plan.action,
                                        plan_reason=plan.reason,
                                        final_status="committed",
                                        native_files=len(plan.features),
                                        native_rows=sum(item.native_rows for item in result.outputs),
                                        curated_rows=sum(item.curated_rows for item in result.outputs),
                                        output_size_bytes=sum(item.size_bytes for item in result.outputs),
                                        runtime_seconds=result.runtime_seconds,
                                        peak_rss_bytes=result.peak_rss_bytes,
                                        outputs=output_states,
                                    )
                                    swap.finalize()
                                    summary.committed += 1
                                except Exception as exc:
                                    swap.rollback()
                                    message = f"{type(exc).__name__}: {exc}"
                                    summary.failed += 1
                                    summary.failures[plan.participant_id] = message
                                    state.mark_failed(plan.participant_id, message)
                                    state.record_run_participant(
                                        run_id,
                                        plan.participant_id,
                                        planned_action=plan.action,
                                        plan_reason=plan.reason,
                                        final_status="failed",
                                        native_files=len(plan.features),
                                        native_rows=sum(item.row_count for item in plan.features),
                                        runtime_seconds=result.runtime_seconds,
                                        peak_rss_bytes=result.peak_rss_bytes,
                                        error_message=message,
                                        traceback_text=traceback.format_exc(),
                                    )
                                    aborted = aborted or fail_fast
                            progress.update(1)
                            if not aborted:
                                while len(futures) < inflight_limit and submit_next():
                                    pass
                        if aborted:
                            for future in futures:
                                future.cancel()
                            break
                finally:
                    progress.close()

            shutil.rmtree(temporary_root / run_id, ignore_errors=True)
            status = "complete" if not summary.failed else "complete_with_errors"
            state.connection.execute(
                "UPDATE curation_runs SET finished_at=?,status=? WHERE run_id=?",
                (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), status, run_id),
            )
            state.connection.commit()
            report = state.build_report(run_id)
            report["status"] = status
            report["participants"]["discovered"] = summary.discovered
            report["native_input_diagnostics"] = native_diagnostics
            state.finish_run(run_id, status, report)
            summary.report = report
            return summary
        except Exception:
            state.finish_run(run_id, "failed", {"error": traceback.format_exc(), "configuration": configuration})
            raise
        finally:
            shutil.rmtree(temporary_root / run_id, ignore_errors=True)
            try:
                temporary_root.rmdir()
            except OSError:
                pass


def format_summary(summary: CurationRunSummary) -> str:
    return format_curation_report(summary.as_dict())
