"""
Wearable Data Processing and Modeling project
CLI for the mixed-source raw wearable profiler.
"""


from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
import pickle
import platform
import shutil
import sys
import time
from pathlib import Path
from tqdm import tqdm
from .executor import iter_participant_results
from .input_source import open_raw_input, raise_csv_field_limit
from .profile import (
    CHECKPOINT_COMPATIBLE_TOOL_VERSIONS, MixedSourceProfiler, ProfilerConfig, ProfilerState, TOOL_VERSION,
    merge_profiler_state, normalize_source,
)
from .reports import write_reports


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wearable_project.inspection",
        description=(
            "Read-only, mixed-source census and bounded deep profile of HPP participant/month raw exports. "
            "All outer sources are inventoried; selected sources are payload-decoded."
        ),
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw cohort root or ZIP archive")
    parser.add_argument("--output", type=Path, required=True, help="New report directory")
    parser.add_argument("--label", help="Human-readable run label")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--outer-only", action="store_true", help="Do not decode any payload")
    mode.add_argument(
        "--decode-source", action="append", default=[], metavar="LABEL",
        help="Decode this source label/key; repeat for multiple sources",
    )
    mode.add_argument(
        "--decode-non-apple", action="store_true",
        help="Decode every source except --apple-source (recommended for the mixed cohort)",
    )
    parser.add_argument("--apple-source", default="AppleHealthkit", help="Outer source label treated as Apple")
    parser.add_argument("--participants-file", type=Path, help="Optional participant IDs, one per line")
    parser.add_argument("--max-participants", type=int, help="Testing only: stop after this many selected participants")
    parser.add_argument("--numeric-sample-size", type=int, default=4096)
    parser.add_argument("--category-top-k", type=int, default=200)
    parser.add_argument("--max-nested-list-items", type=int, default=128)
    parser.add_argument("--diagnostic-examples", type=int, default=100)
    parser.add_argument(
        "--workers", type=int, default=4, help="Participant worker processes (bounded by available CPUs)",
    )
    parser.add_argument(
        "--max-in-flight", type=int, help="Maximum submitted participant tasks; defaults to --workers",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop after the first participant worker failure",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=1000,
        help="Write a full-state checkpoint after this many participants; 0 disables",
    )
    parser.add_argument(
        "--progress-every", type=int, default=0,
        help=(
            "Write a detailed progress snapshot after this many reduced participants; "
            "0 disables snapshots (the tqdm bar remains enabled)"
        ),
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable the tqdm participant progress bar",
    )
    parser.add_argument(
        "--require-phase0-pass", action="store_true",
        help=("Exit nonzero after writing reports unless phase0_acceptance.json has status 'pass'"),
    )
    parser.add_argument("--resume", action="store_true", help="Resume OUTPUT.partial/checkpoint.pkl.gz")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output/partial directory")
    parser.add_argument("--version", action="version", version=TOOL_VERSION)

    return parser


def _selected_participants(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _config_from_args(args: argparse.Namespace) -> ProfilerConfig:
    source_keys = {normalize_source(value) for value in args.decode_source}
    if args.outer_only:
        decode_non_apple = False
    elif source_keys:
        decode_non_apple = False
    else:
        # The command's purpose is the non-Apple extension; make the useful full-census mode the default.
        decode_non_apple = True
    if args.decode_non_apple:
        decode_non_apple = True
    return ProfilerConfig(
        decode_non_apple=decode_non_apple,
        decode_source_keys=source_keys,
        apple_source_key=normalize_source(args.apple_source),
        numeric_sample_size=args.numeric_sample_size,
        category_top_k=args.category_top_k,
        max_nested_list_items=args.max_nested_list_items,
        diagnostic_examples=args.diagnostic_examples,
    )


def _config_signature(
    config: ProfilerConfig, selected: set[str] | None, max_participants: int | None = None,
    *, tool_version: str = TOOL_VERSION,
) -> str:
    payload = {
        "tool_version": tool_version,
        "config": {
            "decode_non_apple": config.decode_non_apple,
            "decode_source_keys": sorted(config.decode_source_keys),
            "apple_source_key": config.apple_source_key,
            "numeric_sample_size": config.numeric_sample_size,
            "category_top_k": config.category_top_k,
            "max_nested_list_items": config.max_nested_list_items,
            "max_category_length": config.max_category_length,
            "diagnostic_examples": config.diagnostic_examples,
        },
        "selected_participants": sorted(selected) if selected is not None else None,
        "max_participants": max_participants,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _assert_safe_paths(input_path: Path, output: Path) -> None:
    input_resolved = input_path.expanduser().resolve()
    output_resolved = output.expanduser().resolve()
    dangerous = {Path("/").resolve(), Path.home().resolve(), input_resolved}
    if output_resolved in dangerous:
        raise ValueError(f"Refusing dangerous output path: {output_resolved}")
    if input_resolved.is_dir() and output_resolved.is_relative_to(input_resolved):
        raise ValueError("Output must not be inside the raw input root")
    if input_resolved.is_dir() and input_resolved.is_relative_to(output_resolved):
        raise ValueError("Output must not be an ancestor of the raw input root")


def _save_checkpoint(path: Path, state: ProfilerState, signature: str, input_descriptor: dict[str, object]) -> None:
    payload = {
        "tool_version": TOOL_VERSION,
        "config_signature": signature,
        "input_descriptor": input_descriptor,
        "state": state,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Checkpoints contain the complete reduced state. Level 1 materially lowers checkpoint CPU time on the full
    # cohort while retaining useful compression.
    with gzip.open(temporary, "wb", compresslevel=1) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path, compatible_signatures: set[str], input_descriptor: dict[str, object],
) -> ProfilerState:
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    checkpoint_version = payload.get("tool_version")
    if checkpoint_version not in CHECKPOINT_COMPATIBLE_TOOL_VERSIONS:
        raise ValueError("Checkpoint tool version differs from the current profiler")
    if payload.get("config_signature") not in compatible_signatures:
        raise ValueError("Checkpoint configuration differs from this command")
    old_descriptor = payload.get("input_descriptor", {})
    keys = {"kind", "input_path", "participant_count", "monthly_file_count"}
    if input_descriptor.get("kind") == "directory":
        keys.add("source_bytes")
    else:
        keys.update({"archive_size_bytes", "archive_mtime_ns", "uncompressed_monthly_bytes"})
    for key in sorted(keys):
        if old_descriptor.get(key) != input_descriptor.get(key):
            raise ValueError(f"Checkpoint input descriptor differs at {key!r}")
    state = payload.get("state")
    if not isinstance(state, ProfilerState):
        raise ValueError("Checkpoint does not contain a valid profiler state")
    return state


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.max_in_flight is not None and args.max_in_flight < 1:
        raise ValueError("--max-in-flight must be at least 1")
    if args.max_participants is not None and args.max_participants < 1:
        raise ValueError("--max-participants must be at least 1")
    if args.numeric_sample_size < 0:
        raise ValueError("--numeric-sample-size must be non-negative")
    if args.category_top_k < 1:
        raise ValueError("--category-top-k must be at least 1")
    if args.max_nested_list_items < 1:
        raise ValueError("--max-nested-list-items must be at least 1")
    if args.diagnostic_examples < 0:
        raise ValueError("--diagnostic-examples must be non-negative")
    if args.checkpoint_every < 0 or args.progress_every < 0:
        raise ValueError("--checkpoint-every and --progress-every must be non-negative")
    raise_csv_field_limit()
    _assert_safe_paths(args.input, args.output)
    output = args.output.expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    checkpoint = partial / "checkpoint.pkl.gz"

    if args.overwrite and not args.resume:
        if output.exists():
            shutil.rmtree(output)
        if partial.exists():
            shutil.rmtree(partial)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if partial.exists() and not args.resume:
        raise FileExistsError(f"Partial output already exists; use --resume or --overwrite: {partial}")
    partial.mkdir(parents=True, exist_ok=True)

    selected = _selected_participants(args.participants_file)
    config = _config_from_args(args)
    signature = _config_signature(config, selected, args.max_participants)
    compatible_signatures = {
        _config_signature(config, selected, args.max_participants, tool_version=version,)
        for version in CHECKPOINT_COMPATIBLE_TOOL_VERSIONS
    }
    started_wall = time.time()
    phase0_acceptance_status = "not_generated"

    with open_raw_input(args.input) as raw_input:
        descriptor = raw_input.descriptor()
        if args.resume:
            if not checkpoint.exists():
                raise FileNotFoundError(f"No checkpoint found: {checkpoint}")
            state = _load_checkpoint(checkpoint, compatible_signatures, descriptor)
        else:
            state = ProfilerState(config=config)
        profiler = MixedSourceProfiler(raw_input, state)
        participants = [entry for entry in raw_input.participants() if selected is None or entry.participant_id in selected]
        if args.max_participants is not None:
            participants = participants[: args.max_participants]
        total = len(participants)
        remaining = [
            participant
            for participant in participants if participant.participant_id not in state.completed_participants
        ]
        workers_requested = int(args.workers)
        workers_effective = max(1, min(workers_requested, os.cpu_count() or 1, max(1, len(remaining)),),)
        max_in_flight = max(1, int(args.max_in_flight or workers_effective))
        state.workers_requested = workers_requested
        state.workers_effective = workers_effective
        state.max_in_flight = max_in_flight
        state.execution_mode = "multiprocessing" if workers_effective > 1 else "sequential"

        newly_attempted = 0
        first_failure: tuple[str, str] | None = None
        progress_failures = 0
        initial_completed = len(state.completed_participants)
        progress = tqdm(
            total=total,
            initial=initial_completed,
            desc="Inspecting participants",
            unit="participant",
            file=sys.stderr,
            disable=args.no_progress,
            dynamic_ncols=True,
        )

        def participant_finished(result) -> None:
            nonlocal progress_failures
            if result.error is not None or result.state is None:
                progress_failures += 1
            progress.update(1)
            progress.set_postfix(
                workers=workers_effective,
                failed=progress_failures,
                refresh=False,
            )

        results = iter_participant_results(
            raw_input=raw_input,
            participants=remaining,
            config=config,
            workers=workers_effective,
            max_in_flight=max_in_flight,
            on_complete=participant_finished,
        )
        try:
            for result in results:
                newly_attempted += 1
                participant_id = result.participant_id
                state.participant_runtime_seconds[participant_id] = result.runtime_seconds
                if result.peak_rss_bytes is not None:
                    state.participant_peak_rss_bytes[participant_id] = result.peak_rss_bytes
                if result.error is not None or result.state is None:
                    message = result.error or "Worker returned no participant state"
                    state.participant_failures[participant_id] = {
                        "participant_id": participant_id,
                        "error": message,
                        "traceback": result.traceback_text,
                        "runtime_seconds": result.runtime_seconds,
                        "peak_rss_bytes": result.peak_rss_bytes,
                    }
                    if first_failure is None:
                        first_failure = (participant_id, message)
                else:
                    merge_profiler_state(state, result.state)
                    state.participant_failures.pop(participant_id, None)

                if args.progress_every and newly_attempted % args.progress_every == 0:
                    elapsed = max(time.time() - started_wall, 1e-9)
                    attempted_total = len(state.completed_participants) + len(state.participant_failures)
                    tqdm.write(
                        f"[{attempted_total}/{total}] completed={len(state.completed_participants)} "
                        f"failed={len(state.participant_failures)} files={state.total_files} "
                        f"outer_rows={state.total_outer_rows} decoded_items={state.total_payload_items} "
                        f"workers={workers_effective} in_flight={max_in_flight} elapsed={elapsed:.1f}s",
                        file=sys.stderr,
                    )
                if args.checkpoint_every and newly_attempted % args.checkpoint_every == 0:
                    _save_checkpoint(checkpoint, state, signature, descriptor)
                if first_failure is not None and args.fail_fast:
                    _save_checkpoint(checkpoint, state, signature, descriptor)
                    break
        finally:
            close_results = getattr(results, "close", None)
            if close_results is not None:
                close_results()
            progress.close()

        _save_checkpoint(checkpoint, state, signature, descriptor)
        manifest = {
            "tool_version": TOOL_VERSION,
            "label": args.label,
            "input": descriptor,
            "config_signature": signature,
            "config": {
                "decode_non_apple": config.decode_non_apple,
                "decode_source_keys": sorted(config.decode_source_keys),
                "apple_source_key": config.apple_source_key,
                "numeric_sample_size": config.numeric_sample_size,
                "category_top_k": config.category_top_k,
                "max_nested_list_items": config.max_nested_list_items,
                "diagnostic_examples": config.diagnostic_examples,
            },
            "execution": {
                "workers_requested": workers_requested,
                "workers_effective": workers_effective,
                "max_in_flight": max_in_flight,
                "start_method": "spawn" if workers_effective > 1 else None,
                "task_unit": "participant",
                "task_order": "descending_source_bytes_then_participant_id",
                "deterministic_parent_reduction": True,
                "fail_fast": bool(args.fail_fast),
                "progress_bar_enabled": not args.no_progress,
                "progress_every": args.progress_every,
                "require_phase0_pass": bool(args.require_phase0_pass),
            },
            "python": sys.version,
            "platform": platform.platform(),
            "started_epoch": state.started_at_epoch,
            "finished_epoch": time.time(),
            "selected_participant_count": total,
            "completed_participant_count": len(state.completed_participants),
            "failed_participant_count": len(state.participant_failures),
        }
        write_reports(partial, profiler, manifest)
        acceptance_path = partial / "phase0_acceptance.json"
        if acceptance_path.exists():
            phase0_acceptance_status = json.loads(
                acceptance_path.read_text(encoding="utf-8")
            ).get("status", "unknown")

    checkpoint.unlink(missing_ok=True)
    marker = partial / ".profile_complete"
    status = "complete" if not state.participant_failures else "complete_with_errors"
    marker.write_text(f"{TOOL_VERSION}\n{status}\n", encoding="utf-8")
    os.replace(partial, output)
    print(
        json.dumps(
            {"output": str(output), "phase0_acceptance_status": phase0_acceptance_status, **profiler.summary(),},
            sort_keys=True,
        ),
        flush=True,
    )
    if first_failure is not None and args.fail_fast:
        participant_id, message = first_failure
        print(f"ERROR: participant {participant_id}: {message}", file=sys.stderr)
    if state.participant_failures:
        return 1
    if args.require_phase0_pass and phase0_acceptance_status != "pass":
        print(
            "ERROR: Phase 0 acceptance status is "
            f"{phase0_acceptance_status!r}; see {output / 'phase0_acceptance.json'}",
            file=sys.stderr,
        )
        return 2
    return 0
