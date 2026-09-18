"""
Wearable Data Processing and Modeling project
CLI integration for curation commands.
"""


from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any
from wearable_project.curation.pipeline import curate_dataset, format_summary
from wearable_project.curation.state import format_curation_report, load_curation_report
from wearable_project.processing.pipeline import process_dataset
from wearable_project.processing.tracker import format_processing_report


def _participants(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def add_curation_commands(commands: Any) -> None:
    curate = commands.add_parser(
        "curate", help="Curate a native processing participant-feature root without changing it",
    )
    curate.add_argument("--input-native", type=Path, required=True)
    curate.add_argument("--output", type=Path, required=True)
    curate.add_argument("--workers", type=int, default=4)
    curate.add_argument("--max-in-flight", type=int)
    curate.add_argument("--mode", choices=("auto", "rebuild"), default="auto")
    curate.add_argument("--participants-file", type=Path)
    curate.add_argument("--verify-existing-hashes", action="store_true")
    curate.add_argument(
        "--allow-unmanaged-native-root", action="store_true",
        help=(
            "Allow a native-copy root without .wearable_state.sqlite. Curated roots "
            "are still rejected by state-marker and CSV-header checks."
        ),
    )
    curate.add_argument("--fail-fast", action="store_true")
    curate.add_argument("--state-file", type=Path)
    curate.add_argument("--json-summary", action="store_true")

    run = commands.add_parser("run", help="Run native processing and then curation",)
    run.add_argument("--input", type=Path, required=True, help="Raw cumulative export root")
    run.add_argument("--native-output", type=Path, required=True)
    run.add_argument("--curated-output", type=Path, required=True)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--max-in-flight", type=int)
    run.add_argument("--native-mode", choices=("auto", "rebuild"), default="auto")
    run.add_argument("--curation-mode", choices=("auto", "rebuild"), default="auto")
    run.add_argument(
        "--snapshot-policy", choices=("strict-cumulative", "authoritative", "append-only"),
        default="strict-cumulative",
    )
    run.add_argument("--row-error-policy", choices=("fail-participant", "skip-row"), default="fail-participant",)
    run.add_argument("--participants-file", type=Path)
    run.add_argument("--verify-existing-hashes", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.add_argument("--json-summary", action="store_true")

    report = commands.add_parser("curation-report", help="Read a persisted Milestone 2 curation report",)
    location = report.add_mutually_exclusive_group(required=True)
    location.add_argument("--output", type=Path)
    location.add_argument("--state-file", type=Path)
    report.add_argument("--run-id")
    report.add_argument("--json", action="store_true")
    report.add_argument("--include-details", action="store_true")


def handle_curation_command(args: argparse.Namespace) -> int | None:
    if args.command == "curate":
        try:
            summary = curate_dataset(
                args.input_native, args.output,
                workers=args.workers,
                max_in_flight=args.max_in_flight,
                mode=args.mode,
                selected_participants=_participants(args.participants_file),
                verify_existing_hashes=args.verify_existing_hashes,
                fail_fast=args.fail_fast,
                state_file=args.state_file,
                allow_unmanaged_native_root=args.allow_unmanaged_native_root,
            )
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        if args.json_summary:
            print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(format_summary(summary), end="")
            for participant, message in sorted(summary.failures.items()):
                print(f"FAILED {participant}: {message}")
        return 1 if summary.failed else 0

    if args.command == "curation-report":
        state_file = args.state_file or (
            args.output.expanduser().resolve() / ".wearable_curation_state.sqlite"
        )
        try:
            report = load_curation_report(
                state_file, args.run_id, include_details=args.include_details,
            )
        except (FileNotFoundError, KeyError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(format_curation_report(report), end="")
            for participant, message in sorted(report.get("failures", {}).items()):
                print(f"FAILED {participant}: {message}")
        return 1 if report["participants"]["failed"] else 0

    if args.command == "run":
        selected = _participants(args.participants_file)
        try:
            native_summary = process_dataset(
                args.input, args.native_output,
                workers=args.workers,
                max_in_flight=args.max_in_flight,
                mode=args.native_mode,
                snapshot_policy=args.snapshot_policy,
                row_error_policy=args.row_error_policy,
                selected_participants=selected,
                verify_existing_hashes=args.verify_existing_hashes,
                fail_fast=args.fail_fast,
            )
            if native_summary.failed or native_summary.blocked:
                payload = {"native": native_summary.as_dict(), "curation": None}
                if args.json_summary:
                    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
                else:
                    print(format_processing_report(native_summary.as_dict()))
                    print("Curation was not started because native processing did not complete cleanly.")
                return 1
            curated_summary = curate_dataset(
                args.native_output, args.curated_output,
                workers=args.workers,
                max_in_flight=args.max_in_flight,
                mode=args.curation_mode,
                selected_participants=selected,
                verify_existing_hashes=args.verify_existing_hashes,
                fail_fast=args.fail_fast,
            )
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        payload = {
            "native": native_summary.as_dict(), "curation": curated_summary.as_dict(),
        }
        if args.json_summary:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print("Milestone 1 native processing")
            print(format_processing_report(native_summary.as_dict()))
            print("Milestone 2 curation")
            print(format_summary(curated_summary), end="")
        return 1 if curated_summary.failed else 0

    return None
