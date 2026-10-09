"""
Wearable Data Processing and Modeling project
Participant-owned multiprocessing for the mixed-source raw profiler.
"""


from __future__ import annotations
import atexit
import gc
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
import multiprocessing as mp
from .input_source import ParticipantFiles, RawInput, WorkerRawInput, open_worker_input, raise_csv_field_limit
from .profile import MixedSourceProfiler, ProfilerConfig, ProfilerState


_WORKER_INPUT: WorkerRawInput | None = None
_WORKER_CONFIG: ProfilerConfig | None = None


@dataclass(frozen=True, slots=True)
class ParticipantProfileTask:
    sequence: int
    participant: ParticipantFiles


@dataclass(slots=True)
class ParticipantProfileResult:
    sequence: int
    participant_id: str
    state: ProfilerState | None
    runtime_seconds: float
    peak_rss_bytes: int | None
    error: str | None = None
    traceback_text: str | None = None


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if value <= 0:
            return None
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, AttributeError, OSError, ValueError):
        return None


def _close_worker_input() -> None:
    global _WORKER_INPUT
    if _WORKER_INPUT is not None:
        _WORKER_INPUT.close()
        _WORKER_INPUT = None


def _worker_initializer(kind: str, input_path: str, config: ProfilerConfig) -> None:
    global _WORKER_INPUT, _WORKER_CONFIG
    raise_csv_field_limit()
    _WORKER_INPUT = open_worker_input(kind, Path(input_path))
    _WORKER_CONFIG = config
    atexit.register(_close_worker_input)


def _scan_worker(task: ParticipantProfileTask) -> ParticipantProfileResult:
    started = time.perf_counter()
    participant_id = task.participant.participant_id
    try:
        if _WORKER_INPUT is None or _WORKER_CONFIG is None:
            raise RuntimeError("Profiler worker was not initialized")
        local_state = ProfilerState(config=_WORKER_CONFIG)
        MixedSourceProfiler(_WORKER_INPUT, local_state).scan_participant(task.participant)
        return ParticipantProfileResult(
            sequence=task.sequence,
            participant_id=participant_id,
            state=local_state,
            runtime_seconds=time.perf_counter() - started,
            peak_rss_bytes=_peak_rss_bytes(),
        )
    except BaseException as exc:
        return ParticipantProfileResult(
            sequence=task.sequence,
            participant_id=participant_id,
            state=None,
            runtime_seconds=time.perf_counter() - started,
            peak_rss_bytes=_peak_rss_bytes(),
            error=f"{type(exc).__name__}: {exc}",
            traceback_text=traceback.format_exc(),
        )
    finally:
        # Participant-local dictionaries can be large. A pool worker is reused, so make collection explicit
        # before it accepts its next participant.
        gc.collect()


def _scan_local(
    raw_input: RawInput, config: ProfilerConfig, task: ParticipantProfileTask,
) -> ParticipantProfileResult:
    started = time.perf_counter()
    participant_id = task.participant.participant_id
    try:
        local_state = ProfilerState(config=config)
        MixedSourceProfiler(raw_input, local_state).scan_participant(task.participant)
        return ParticipantProfileResult(
            sequence=task.sequence,
            participant_id=participant_id,
            state=local_state,
            runtime_seconds=time.perf_counter() - started,
            peak_rss_bytes=_peak_rss_bytes(),
        )
    except BaseException as exc:
        return ParticipantProfileResult(
            sequence=task.sequence,
            participant_id=participant_id,
            state=None,
            runtime_seconds=time.perf_counter() - started,
            peak_rss_bytes=_peak_rss_bytes(),
            error=f"{type(exc).__name__}: {exc}",
            traceback_text=traceback.format_exc(),
        )
    finally:
        gc.collect()


def ordered_participants(participants: Sequence[ParticipantFiles]) -> list[ParticipantFiles]:
    """Largest snapshots first, then participant ID for deterministic scheduling."""
    return sorted(
        participants,
        key=lambda participant: (
            -sum(file.size_bytes for file in participant.files),
            participant.participant_id,
        ),
    )


def iter_participant_results(
    *, raw_input: RawInput, participants: Sequence[ParticipantFiles], config: ProfilerConfig,
    workers: int, max_in_flight: int,
) -> Iterator[ParticipantProfileResult]:
    """
    Yield participant results in deterministic task order.
    Workers scan participants independently. Results may finish out of order, but
    the parent buffers at most ``max_in_flight`` results and yields them in the
    deterministic largest-first order. This keeps representative examples,
    bounded sketches, checkpoints, and resumed runs reproducible.
    """
    tasks = [
        ParticipantProfileTask(sequence=index, participant=participant)
        for index, participant in enumerate(ordered_participants(participants))
    ]
    if not tasks:
        return

    worker_count = max(1, int(workers))
    inflight_limit = max(1, int(max_in_flight))
    if worker_count == 1 or len(tasks) == 1:
        for task in tasks:
            yield _scan_local(raw_input, config, task)
        return

    descriptor = raw_input.descriptor()
    input_kind = str(descriptor["kind"])
    input_path = str(descriptor["input_path"])
    context = mp.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_worker_initializer,
        initargs=(input_kind, input_path, config),
    )
    futures: dict[Future[ParticipantProfileResult], int] = {}
    buffered: dict[int, ParticipantProfileResult] = {}
    next_submit = 0
    next_yield = 0

    def submit_more() -> None:
        nonlocal next_submit
        # Bound both running futures and completed-but-not-yet-reduced results. Without the buffered term, one
        # slow early task could allow thousands of later participant states to accumulate in the parent process.
        while (
            next_submit < len(tasks)
            and len(futures) + len(buffered) < inflight_limit
        ):
            task = tasks[next_submit]
            futures[executor.submit(_scan_worker, task)] = task.sequence
            next_submit += 1

    try:
        submit_more()
        while futures or buffered or next_submit < len(tasks):
            if futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    sequence = futures.pop(future)
                    task = tasks[sequence]
                    try:
                        result = future.result()
                    except BaseException as exc:
                        result = ParticipantProfileResult(
                            sequence=sequence,
                            participant_id=task.participant.participant_id,
                            state=None,
                            runtime_seconds=0.0,
                            peak_rss_bytes=None,
                            error=f"{type(exc).__name__}: {exc}",
                            traceback_text=traceback.format_exc(),
                        )
                    buffered[sequence] = result
            while next_yield in buffered:
                yield buffered.pop(next_yield)
                next_yield += 1
                submit_more()
            if not futures and next_submit < len(tasks):
                submit_more()
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
