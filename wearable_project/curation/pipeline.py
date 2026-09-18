"""
Wearable Data Processing and Modeling project
Participant-level non-destructive curation pipeline.
"""


from __future__ import annotations
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import shutil
import sqlite3
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from tqdm import tqdm
from wearable_project.curation.decisions import decisions_fingerprint
from wearable_project.curation.engine import CuratedFeatureInfo, curate_feature_file
from wearable_project.curation.registry import get_policy, registry_fingerprint
from wearable_project.curation.state import (
    CURATION_ENGINE_VERSION, CuratedOutputState, CurationRunSummary, CurationStateDatabase,
    NativeFileState, format_curation_report,
)
from wearable_project.curation.unit_resolution import build_participant_unit_context
from wearable_project.exceptions import CurationError, InputLayoutError, OutputValidationError
from wearable_project.processing.parser import sha256_file
from wearable_project.processing.tracker import peak_rss_bytes
from wearable_project.processing.writer import DirectorySwap, drop_file_cache


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_field_limit()


@dataclass(frozen=True, slots=True)
class NativeFeatureInput:
    feature: str
    filename: str
    path: Path
    row_count: int
    size_bytes: int
    sha256: str
    policy_fingerprint: str


@dataclass(slots=True)
class CurationTask:
    participant_id: str
    native_dir: Path
    stage_dir: Path
    features: list[NativeFeatureInput]


@dataclass(slots=True)
class CurationBuildResult:
    participant_id: str
    stage_dir: Path
    outputs: list[CuratedFeatureInfo] = field(default_factory=list)
    unit_epochs: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    runtime_seconds: float | None = None
    peak_rss_bytes: int | None = None
    error: str | None = None
    traceback_text: str | None = None


@dataclass(frozen=True, slots=True)
class ParticipantPlan:
    participant_id: str
    native_dir: Path
    action: str
    reason: str
    features: list[NativeFeatureInput]
    native_signature: str
    policy_signature: str


NATIVE_STATE_FILENAME = ".wearable_state.sqlite"
CURATION_STATE_FILENAME = ".wearable_curation_state.sqlite"
CURATION_ONLY_COLUMNS = frozenset({
    "curation_status",
    "curation_flags",
    "include_by_default",
    "curation_unit_status",
})


def _find_curated_csv_header(input_root: Path) -> tuple[Path, tuple[str, ...]] | None:
    """
    Return the first feature CSV that contains curation-only columns. This scan is used only when the normal
    hidden state markers are absent. A managed native processing root returns before this path, so routine
    curation does not pay the cost of opening feature files merely to classify the root.
    """

    for participant_dir in sorted(
        (path for path in input_root.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: path.name,
    ):
        for csv_path in sorted(participant_dir.glob("*.csv"), key=lambda path: path.name):
            try:
                with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    header = next(csv.reader(handle), [])
            except (OSError, UnicodeError, csv.Error):
                # Root classification must not replace the feature reader's full diagnostics. Continue until
                # a decisive curated header is found.
                continue
            found = tuple(sorted(CURATION_ONLY_COLUMNS.intersection(header)))
            if found:
                return csv_path, found
    return None


def validate_native_input_root(input_root: Path, *, allow_unmanaged_native_root: bool = False,) -> Path:
    """
    Validate that ``input_root`` is a native processing dataset. A curated root is rejected before any
    participant discovery or output-state mutation. Managed native roots are identified by
    ``.wearable_state.sqlite``. The explicit override is reserved for deliberately unmanaged copies; it
    does not permit a root carrying a curation marker or Milestone 2-only columns.
    """

    source = input_root.expanduser().resolve()
    if not source.is_dir():
        raise InputLayoutError(f"Native input root is not a directory: {source}")

    curated_state = source / CURATION_STATE_FILENAME
    native_state = source / NATIVE_STATE_FILENAME
    if curated_state.is_file():
        raise InputLayoutError(
            "The --input-native path appears to be a Milestone 2 curated root "
            f"because it contains {CURATION_STATE_FILENAME}. Expected a Milestone 1 "
            f"native root containing {NATIVE_STATE_FILENAME}: {source}"
        )

    if native_state.is_file():
        return source

    curated_header = _find_curated_csv_header(source)
    if curated_header is not None:
        csv_path, columns = curated_header
        relative = csv_path.relative_to(source)
        raise InputLayoutError(
            "The --input-native path appears to contain Milestone 2 curated CSVs. "
            f"Found curation-only column(s) {', '.join(columns)} in {relative}. "
            f"Expected a Milestone 1 native root containing {NATIVE_STATE_FILENAME}: {source}"
        )

    if not allow_unmanaged_native_root:
        raise InputLayoutError(
            "The --input-native path is not a recognized managed Milestone 1 native root "
            f"because {NATIVE_STATE_FILENAME} is missing: {source}. "
            "Use --allow-unmanaged-native-root only for a deliberately unmanaged copy "
            "whose feature CSVs are known to be Milestone 1 native outputs."
        )
    return source


def _ensure_disjoint_roots(
    input_root: Path, output_root: Path, *, allow_unmanaged_native_root: bool = False,
) -> tuple[Path, Path]:
    source = validate_native_input_root(
        input_root, allow_unmanaged_native_root=allow_unmanaged_native_root,
    )
    destination = output_root.expanduser().resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise InputLayoutError(
            "Native and curated roots must be distinct and non-nested: "
            f"native={source}, curated={destination}"
        )
    return source, destination


def discover_native_participants(input_root: Path, selected: set[str] | None = None,) -> dict[str, Path]:
    participants: dict[str, Path] = {}
    for child in sorted(input_root.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if selected is not None and child.name not in selected:
            continue
        if any(path.is_file() and path.suffix.lower() == ".csv" for path in child.iterdir()):
            participants[child.name] = child
    return participants


def _native_state_manifest(input_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    state_file = input_root / ".wearable_state.sqlite"
    if not state_file.is_file():
        return {}
    manifest: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        with sqlite3.connect(state_file) as connection:
            connection.row_factory = sqlite3.Row
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(feature_outputs)")
            }
            required = {"participant_id", "feature", "filename", "row_count", "size_bytes", "sha256"}
            if not required.issubset(columns):
                return {}
            for row in connection.execute(
                "SELECT participant_id,feature,filename,row_count,size_bytes,sha256 FROM feature_outputs"
            ):
                manifest[(row["participant_id"], row["filename"])] = dict(row)
    except sqlite3.Error:
        return {}
    return manifest


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def discover_native_features(
    participant_id: str, participant_dir: Path, manifest: dict[tuple[str, str], dict[str, Any]],
) -> list[NativeFeatureInput]:
    features: list[NativeFeatureInput] = []
    for path in sorted(participant_dir.glob("*.csv"), key=lambda item: item.name):
        feature = path.stem
        stored = manifest.get((participant_id, path.name))
        if stored and path.stat().st_size == int(stored["size_bytes"]):
            row_count = int(stored["row_count"])
            digest = str(stored["sha256"])
        else:
            row_count = _count_csv_rows(path)
            digest = sha256_file(path)
        policy = get_policy(feature)
        features.append(NativeFeatureInput(
            feature=feature,
            filename=path.name,
            path=path,
            row_count=row_count,
            size_bytes=path.stat().st_size,
            sha256=digest,
            policy_fingerprint=policy.fingerprint(),
        ))
    return features


def _signature(features: Iterable[NativeFeatureInput], *, include_policy: bool) -> str:
    payload = []
    for item in sorted(features, key=lambda value: value.filename):
        row = {
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
        payload.extend([
            {"registry": registry_fingerprint()},
            {"decisions": decisions_fingerprint()},
            {"engine": CURATION_ENGINE_VERSION},
        ])
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stored_outputs_usable(output_dir: Path, stored: Any, *, verify_hashes: bool) -> bool:
    if not output_dir.is_dir() or not stored.outputs:
        return False
    for output in stored.outputs.values():
        path = output_dir / output.filename
        if not path.is_file() or path.stat().st_size != output.size_bytes:
            return False
        if verify_hashes and sha256_file(path) != output.sha256:
            return False
    return True


def plan_participant(
    participant_id: str, participant_dir: Path, features: list[NativeFeatureInput], state: CurationStateDatabase,
    output_root: Path, *, mode: str, verify_existing_hashes: bool,
) -> ParticipantPlan:
    native_signature = _signature(features, include_policy=False)
    policy_signature = _signature(features, include_policy=True)
    stored = state.get_participant(participant_id)
    if mode == "rebuild":
        return ParticipantPlan(
            participant_id, participant_dir, "rebuild", "forced rebuild",
            features, native_signature, policy_signature,
        )
    if stored is None:
        reason = "participant has no committed curation state"
    elif stored.status != "complete":
        reason = f"previous curation state is {stored.status!r}"
    elif stored.engine_version != CURATION_ENGINE_VERSION:
        reason = "curation engine version changed"
    elif stored.native_signature != native_signature:
        reason = "native participant input changed"
    elif stored.policy_signature != policy_signature:
        reason = "relevant feature policy changed"
    elif not _stored_outputs_usable(
        output_root / participant_id, stored, verify_hashes=verify_existing_hashes,
    ):
        reason = "curated participant output is missing or invalid"
    else:
        return ParticipantPlan(
            participant_id, participant_dir, "skip", "native input and policies unchanged",
            features, native_signature, policy_signature,
        )
    return ParticipantPlan(
        participant_id, participant_dir, "rebuild", reason,
        features, native_signature, policy_signature,
    )


def _build_participant(task: CurationTask) -> CurationBuildResult:
    started = time.perf_counter()
    try:
        if task.stage_dir.exists():
            shutil.rmtree(task.stage_dir)
        task.stage_dir.mkdir(parents=True)
        unit_context = build_participant_unit_context(task.native_dir)
        outputs: list[CuratedFeatureInfo] = []
        for native in task.features:
            destination = task.stage_dir / native.filename
            result = curate_feature_file(
                native.path, destination, native.feature, unit_context,
            )
            if result.info.native_rows != native.row_count:
                raise OutputValidationError(
                    f"{native.path}: manifest rows={native.row_count}, "
                    f"curation rows={result.info.native_rows}"
                )
            if result.info.curated_rows != result.info.native_rows:
                raise OutputValidationError(
                    f"{native.path}: native rows={result.info.native_rows}, "
                    f"curated rows={result.info.curated_rows}"
                )
            outputs.append(result.info)
        expected = {item.filename for item in task.features}
        actual = {path.name for path in task.stage_dir.glob("*.csv")}
        if expected != actual:
            raise OutputValidationError(
                f"{task.participant_id}: staged file set mismatch: "
                f"expected={sorted(expected)}, actual={sorted(actual)}"
            )
        return CurationBuildResult(
            participant_id=task.participant_id,
            stage_dir=task.stage_dir,
            outputs=outputs,
            unit_epochs=[asdict(epoch) for epoch in unit_context.epochs],
            diagnostics=list(unit_context.diagnostics),
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


def _output_state(info: CuratedFeatureInfo) -> CuratedOutputState:
    return CuratedOutputState(**asdict(info))


def _native_state(item: NativeFeatureInput) -> NativeFileState:
    return NativeFileState(
        feature=item.feature, filename=item.filename, row_count=item.row_count,
        size_bytes=item.size_bytes, sha256=item.sha256,
        policy_fingerprint=item.policy_fingerprint,
    )


def curate_dataset(
    input_native: Path, output: Path, *, workers: int = 4, max_in_flight: int | None = None, mode: str = "auto",
    selected_participants: set[str] | None = None, verify_existing_hashes: bool = False, fail_fast: bool = False,
    state_file: Path | None = None, allow_unmanaged_native_root: bool = False,
) -> CurationRunSummary:
    native_root, output_root = _ensure_disjoint_roots(
        input_native, output, allow_unmanaged_native_root=allow_unmanaged_native_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = state_file or (output_root / ".wearable_curation_state.sqlite")
    participants = discover_native_participants(native_root, selected_participants)
    manifest = _native_state_manifest(native_root)
    worker_count = max(1, min(int(workers), os.cpu_count() or 1))
    inflight_limit = max(1, int(max_in_flight or worker_count))
    temporary_root = output_root / ".wearable_curation_tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)

    configuration = {
        "mode": mode,
        "workers_requested": workers,
        "workers_effective": worker_count,
        "max_in_flight": inflight_limit,
        "verify_existing_hashes": verify_existing_hashes,
        "engine_version": CURATION_ENGINE_VERSION,
        "registry_fingerprint": registry_fingerprint(),
        "decisions_fingerprint": decisions_fingerprint(),
    }

    with CurationStateDatabase(state_path) as state:
        run_id = state.begin_run(native_root, output_root, configuration)
        summary = CurationRunSummary(run_id=run_id, discovered=len(participants))
        plans: list[ParticipantPlan] = []
        for participant_id, participant_dir in participants.items():
            features = discover_native_features(participant_id, participant_dir, manifest)
            if not features:
                continue
            plan = plan_participant(
                participant_id, participant_dir, features, state, output_root,
                mode=mode, verify_existing_hashes=verify_existing_hashes,
            )
            if plan.action == "skip":
                summary.skipped += 1
                stored = state.get_participant(participant_id)
                native_rows = sum(item.row_count for item in plan.features)
                curated_rows = sum(item.curated_rows for item in stored.outputs.values()) if stored else 0
                output_bytes = sum(item.size_bytes for item in stored.outputs.values()) if stored else 0
                state.record_run_participant(
                    run_id, participant_id, planned_action="skip", plan_reason=plan.reason,
                    final_status="skipped", native_files=len(plan.features), native_rows=native_rows,
                    curated_rows=curated_rows, output_size_bytes=output_bytes,
                )
            else:
                plans.append(plan)
                summary.rebuilt += 1
                state.mark_in_progress(participant_id)

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
                stage_dir = temporary_root / run_id / plan.participant_id
                task = CurationTask(
                    participant_id=plan.participant_id,
                    native_dir=plan.native_dir,
                    stage_dir=stage_dir,
                    features=plan.features,
                )
                futures[executor.submit(_build_participant, task)] = plan
                return True

            while len(futures) < inflight_limit and submit_next():
                pass

            progress = tqdm(total=len(plans), desc="Curating participants", unit="participant")
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
                                run_id, plan.participant_id,
                                planned_action=plan.action, plan_reason=plan.reason,
                                final_status="failed", native_files=len(plan.features),
                                native_rows=sum(item.row_count for item in plan.features),
                                runtime_seconds=result.runtime_seconds,
                                peak_rss_bytes=result.peak_rss_bytes,
                                error_message=result.error,
                                traceback_text=result.traceback_text,
                            )
                            shutil.rmtree(result.stage_dir, ignore_errors=True)
                            if fail_fast:
                                aborted = True
                        else:
                            final_dir = output_root / plan.participant_id
                            backup_dir = temporary_root / run_id / f".{plan.participant_id}.backup"
                            swap = DirectorySwap(final_dir, result.stage_dir, backup_dir)
                            try:
                                swap.apply()
                                outputs = [_output_state(item) for item in result.outputs]
                                state.commit_participant(
                                    plan.participant_id, plan.native_signature, plan.policy_signature,
                                    [_native_state(item) for item in plan.features], outputs,
                                    result.unit_epochs, run_id=run_id,
                                )
                                swap.finalize()
                                summary.committed += 1
                                state.record_run_participant(
                                    run_id, plan.participant_id,
                                    planned_action=plan.action, plan_reason=plan.reason,
                                    final_status="committed", native_files=len(plan.features),
                                    native_rows=sum(item.native_rows for item in result.outputs),
                                    curated_rows=sum(item.curated_rows for item in result.outputs),
                                    output_size_bytes=sum(item.size_bytes for item in result.outputs),
                                    runtime_seconds=result.runtime_seconds,
                                    peak_rss_bytes=result.peak_rss_bytes,
                                )
                            except Exception as exc:
                                swap.rollback()
                                summary.failed += 1
                                message = f"{type(exc).__name__}: {exc}"
                                summary.failures[plan.participant_id] = message
                                state.mark_failed(plan.participant_id, message)
                                state.record_run_participant(
                                    run_id, plan.participant_id,
                                    planned_action=plan.action, plan_reason=plan.reason,
                                    final_status="failed", native_files=len(plan.features),
                                    native_rows=sum(item.row_count for item in plan.features),
                                    runtime_seconds=result.runtime_seconds,
                                    peak_rss_bytes=result.peak_rss_bytes,
                                    error_message=message,
                                    traceback_text=traceback.format_exc(),
                                )
                                if fail_fast:
                                    aborted = True
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
        report_status = "complete" if not summary.failed else "complete_with_errors"
        state.mark_run_finished(run_id, report_status)
        report = state.build_report(run_id)
        report["status"] = report_status
        report["participants"]["discovered"] = summary.discovered
        state.finish_run(run_id, report_status, report)
        summary.report = report
        return summary


def format_summary(summary: CurationRunSummary) -> str:
    return format_curation_report(summary.as_dict())
