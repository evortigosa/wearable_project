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
from .input_source import open_raw_input, raise_csv_field_limit
from .profile import MixedSourceProfiler, ProfilerConfig, ProfilerState, TOOL_VERSION, normalize_source
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
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Checkpoint after this many participants; 0 disables")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress after this many participants; 0 disables")
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


def _config_signature(config: ProfilerConfig, selected: set[str] | None) -> str:
    payload = {
        "tool_version": TOOL_VERSION,
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
    with gzip.open(temporary, "wb", compresslevel=3) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _load_checkpoint(path: Path, signature: str, input_descriptor: dict[str, object]) -> ProfilerState:
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if payload.get("tool_version") != TOOL_VERSION:
        raise ValueError("Checkpoint tool version differs from the current profiler")
    if payload.get("config_signature") != signature:
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
    signature = _config_signature(config, selected)
    started_wall = time.time()

    with open_raw_input(args.input) as raw_input:
        descriptor = raw_input.descriptor()
        if args.resume:
            if not checkpoint.exists():
                raise FileNotFoundError(f"No checkpoint found: {checkpoint}")
            state = _load_checkpoint(checkpoint, signature, descriptor)
        else:
            state = ProfilerState(config=config)
        profiler = MixedSourceProfiler(raw_input, state)
        participants = [entry for entry in raw_input.participants() if selected is None or entry.participant_id in selected]
        if args.max_participants is not None:
            participants = participants[: args.max_participants]
        total = len(participants)
        newly_completed = 0
        for index, participant_files in enumerate(participants, start=1):
            if participant_files.participant_id in state.completed_participants:
                continue
            profiler.scan_participant(participant_files)
            newly_completed += 1
            if args.progress_every and newly_completed % args.progress_every == 0:
                elapsed = max(time.time() - started_wall, 1e-9)
                print(
                    f"[{index}/{total}] participants={len(state.completed_participants)} "
                    f"files={state.total_files} outer_rows={state.total_outer_rows} "
                    f"decoded_items={state.total_payload_items} elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
            if args.checkpoint_every and newly_completed % args.checkpoint_every == 0:
                _save_checkpoint(checkpoint, state, signature, descriptor)
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
            "python": sys.version,
            "platform": platform.platform(),
            "started_epoch": state.started_at_epoch,
            "finished_epoch": time.time(),
            "selected_participant_count": total,
            "completed_participant_count": len(state.completed_participants),
        }
        write_reports(partial, profiler, manifest)

    checkpoint.unlink(missing_ok=True)
    marker = partial / ".profile_complete"
    marker.write_text(TOOL_VERSION + "\n", encoding="utf-8")
    os.replace(partial, output)
    print(json.dumps({"output": str(output), **profiler.summary()}, sort_keys=True), flush=True)

    return 0
