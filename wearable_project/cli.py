"""
Wearable Data Processing and Modeling project
CLI for native participant-feature processing.
"""


from __future__ import annotations
import argparse
import json
from pathlib import Path
from wearable_project import __version__
from wearable_project.processing.pipeline import process_dataset
from wearable_project.processing.registry import REGISTRY_VERSION, known_features


def participant_selection(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wearable-project",
        description="Clean cumulative Apple HealthKit exports into participant/feature CSV folders.",
    )
    parser.add_argument("--version", action="version", version=f"wearable-project {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    process = commands.add_parser("process", help="Build or safely update native feature CSVs")
    process.add_argument("--input", type=Path, required=True, help="Cumulative export directory")
    process.add_argument("--output", type=Path, required=True, help="Cleaned participant output root")
    process.add_argument("--workers", type=int, default=4, help="Participant worker processes")
    process.add_argument("--max-in-flight", type=int, help="Maximum submitted unfinished participant tasks")
    process.add_argument("--mode", choices=("auto", "rebuild"), default="auto")
    process.add_argument(
        "--snapshot-policy", choices=("strict-cumulative", "authoritative", "append-only"),
        default="strict-cumulative",
    )
    process.add_argument(
        "--row-error-policy", choices=("fail-participant", "skip-row"), default="fail-participant",
    )
    process.add_argument("--participants-file", type=Path, help="Optional participant folder names, one per line")
    process.add_argument("--verify-existing-hashes", action="store_true")
    process.add_argument("--fail-fast", action="store_true")
    process.add_argument("--state-file", type=Path, help="Defaults to OUTPUT/.wearable_state.sqlite")
    process.add_argument("--json-summary", action="store_true")

    registry = commands.add_parser("registry", help="List explicitly registered feature policies")
    registry.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "registry":
        payload = {"registry_version": REGISTRY_VERSION, "features": list(known_features())}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Registry version: {REGISTRY_VERSION}")
            print("\n".join(payload["features"]))
        return 0

    summary = process_dataset(
        args.input, args.output, workers=args.workers, max_in_flight=args.max_in_flight,
        mode=args.mode, snapshot_policy=args.snapshot_policy,
        row_error_policy=args.row_error_policy,
        selected_participants=participant_selection(args.participants_file),
        verify_existing_hashes=args.verify_existing_hashes,
        fail_fast=args.fail_fast, state_file=args.state_file,
    )
    if args.json_summary:
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    else:
        print(
            f"discovered={summary.discovered} committed={summary.committed} "
            f"incremental={summary.incremental} rebuilt={summary.rebuilt} "
            f"skipped={summary.skipped} empty={summary.empty} "
            f"blocked={summary.blocked} failed={summary.failed}"
        )
        for participant, reason in sorted(summary.blocks.items()):
            print(f"BLOCKED {participant}: {reason}")
        for participant, reason in sorted(summary.failures.items()):
            print(f"FAILED {participant}: {reason}")
    return 1 if summary.failed or summary.blocked else 0
