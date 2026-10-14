"""
Wearable Data Processing and Modeling project
Source-aware command-line integration for native processing.
"""


from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


APPLE_HEALTHKIT_SOURCE = "apple-healthkit"


def _participant_selection(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _add_apple_healthkit_source(
    command: argparse.ArgumentParser, *, handler: Callable[[argparse.Namespace], int], help_text: str,
) -> argparse.ArgumentParser:
    sources = command.add_subparsers(
        dest="source",
        required=True,
        metavar="{apple-healthkit}",
        title="sources",
        description="Select the source-specific native-processing implementation.",
    )
    apple = sources.add_parser(APPLE_HEALTHKIT_SOURCE, help=help_text)
    apple.set_defaults(_handler=handler)
    return apple


def add_processing_commands(commands: Any) -> None:
    process_command = commands.add_parser("process", help="Build or safely update source-specific native feature CSVs",)
    process = _add_apple_healthkit_source(
        process_command, handler=_process_apple_healthkit,
        help_text="Process Apple HealthKit rows from a cumulative export root",
    )
    process.add_argument("--input", type=Path, required=True, help="Cumulative export directory")
    process.add_argument("--output", type=Path, required=True, help="Cleaned participant output root")
    process.add_argument("--workers", type=int, default=4, help="Participant worker processes")
    process.add_argument("--max-in-flight", type=int, help="Maximum submitted unfinished participant tasks")
    process.add_argument("--mode", choices=("auto", "rebuild"), default="auto")
    process.add_argument(
        "--snapshot-policy", choices=("strict-cumulative", "authoritative", "append-only"), default="strict-cumulative",
    )
    process.add_argument(
        "--row-error-policy", choices=("fail-participant", "skip-row"), default="fail-participant",
    )
    process.add_argument("--participants-file", type=Path, help="Optional participant folder names, one per line")
    process.add_argument("--verify-existing-hashes", action="store_true")
    process.add_argument("--fail-fast", action="store_true")
    process.add_argument("--state-file", type=Path, help="Defaults to OUTPUT/.wearable_state.sqlite")
    process.add_argument("--json-summary", action="store_true", help="Print the complete processing report as JSON")

    scan_command = commands.add_parser(
        "process-scan", help="Read-only scan of the next source-specific native processing plan",
    )
    process_scan = _add_apple_healthkit_source(
        scan_command, handler=_process_scan_apple_healthkit, help_text="Scan the Apple HealthKit processing plan",
    )
    process_scan.add_argument("--input", type=Path, required=True, help="Cumulative export directory")
    process_scan.add_argument("--output", type=Path, required=True, help="Existing or intended native output root")
    process_scan.add_argument("--workers", type=int, default=4, help="Parallel source-hashing workers")
    process_scan.add_argument("--mode", choices=("auto", "rebuild"), default="auto")
    process_scan.add_argument(
        "--snapshot-policy", choices=("strict-cumulative", "authoritative", "append-only"), default="strict-cumulative",
    )
    process_scan.add_argument("--participants-file", type=Path, help="Optional participant folder names, one per line")
    process_scan.add_argument("--verify-existing-hashes", action="store_true")
    process_scan.add_argument("--state-file", type=Path, help="Defaults to OUTPUT/.wearable_state.sqlite")
    process_scan.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    process_scan.add_argument(
        "--include-details", action="store_true", help="Include one planning record per participant in JSON output",
    )
    process_scan.add_argument(
        "--report-output", type=Path, help="Write the rendered report to a file instead of stdout",
    )
    process_scan.add_argument(
        "--max-listed", type=int, default=50,
        help="Maximum non-skip participants listed in the human-readable report",
    )
    process_scan.add_argument(
        "--fail-if-work-needed", action="store_true",
        help="Return exit code 3 when the scan is safe but incremental work or rebuilds are needed",
    )
    process_scan.add_argument("--no-progress", action="store_true", help="Disable the hashing progress bar")

    environment_command = commands.add_parser(
        "processing-environment", help="Report source-specific native processing versions and integrity",
    )
    processing_environment = _add_apple_healthkit_source(
        environment_command, handler=_processing_environment_apple_healthkit,
        help_text="Inspect the Apple HealthKit processing environment",
    )
    processing_environment.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    report_command = commands.add_parser("report", help="Read a persisted source-specific processing report")
    report = _add_apple_healthkit_source(
        report_command, handler=_report_apple_healthkit, help_text="Read an Apple HealthKit processing report",
    )
    location = report.add_mutually_exclusive_group(required=True)
    location.add_argument("--output", type=Path, help="Cleaned output root containing .wearable_state.sqlite")
    location.add_argument("--state-file", type=Path, help="Explicit processing-state SQLite file")
    report.add_argument("--run-id", help="Specific run ID; defaults to the latest tracked run")
    report.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    report.add_argument(
        "--include-details", action="store_true",
        help="Include participant, feature, and diagnostic rows in JSON output",
    )

    registry_command = commands.add_parser("registry", help="List source-specific native processing feature policies")
    registry = _add_apple_healthkit_source(
        registry_command, handler=_registry_apple_healthkit, help_text="List Apple HealthKit native feature policies",
    )
    registry.add_argument("--json", action="store_true")


def _process_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.processing.apple_healthkit.environment import processing_environment_manifest
    from wearable_project.processing.apple_healthkit.pipeline import process_dataset
    from wearable_project.processing.apple_healthkit.tracker import format_processing_report

    processing_env = processing_environment_manifest()
    for warning in processing_env.warnings:
        print(f"PROCESSING ENVIRONMENT WARNING: {warning}", file=sys.stderr)

    summary = process_dataset(
        args.input,
        args.output,
        workers=args.workers,
        max_in_flight=args.max_in_flight,
        mode=args.mode,
        snapshot_policy=args.snapshot_policy,
        row_error_policy=args.row_error_policy,
        selected_participants=_participant_selection(args.participants_file),
        verify_existing_hashes=args.verify_existing_hashes,
        fail_fast=args.fail_fast,
        state_file=args.state_file,
    )
    if args.json_summary:
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    else:
        print(format_processing_report(summary.as_dict()))
        for participant, reason in sorted(summary.blocks.items()):
            print(f"BLOCKED {participant}: {reason}")
        for participant, reason in sorted(summary.failures.items()):
            print(f"FAILED {participant}: {reason}")
    return 1 if summary.failed or summary.blocked else 0


def _process_scan_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.processing.apple_healthkit.scan import format_processing_scan, scan_processing_plan

    try:
        scan = scan_processing_plan(
            args.input,
            args.output,
            workers=args.workers,
            mode=args.mode,
            snapshot_policy=args.snapshot_policy,
            selected_participants=_participant_selection(args.participants_file),
            verify_existing_hashes=args.verify_existing_hashes,
            state_file=args.state_file,
            show_progress=not args.no_progress,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        rendered = json.dumps(
            scan.as_dict(include_details=args.include_details),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    else:
        rendered = format_processing_scan(scan, max_listed=args.max_listed)
    if args.report_output:
        target = args.report_output.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if not scan.safe_to_process:
        return 1
    if args.fail_if_work_needed and scan.processing_needed:
        return 3
    return 0


def _processing_environment_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.processing.apple_healthkit.environment import processing_environment_manifest

    payload = processing_environment_manifest().as_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"Package version: {payload['package_version']}")
        print(f"Release label: {payload['package_release_label']}")
        print(f"Working directory: {payload['current_working_directory']}")
        print(f"Imported package: {payload['imported_package_path']}")
        print(f"Installed distribution: {payload.get('distribution_package_path')}")
        print(f"Import source: {payload['import_source_kind']}")
        print(f"Parser version: {payload['parser_version']}")
        print(f"Registry version: {payload['registry_version']}")
        invalid = [
            name for name, valid in payload.get("core_module_integrity", {}).items() if not valid
        ]
        print("Frozen processing modules: " + ("valid" if not invalid else "mismatch: " + ", ".join(invalid)))
        if payload.get("git_commit"):
            print(f"Git commit: {payload['git_commit']}")
        if payload.get("warnings"):
            print("Warnings:")
            for warning in payload["warnings"]:
                print(f"  - {warning}")
    return 1 if payload.get("warnings") else 0


def _report_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.processing.apple_healthkit.tracker import load_processing_report

    state_file = args.state_file or (args.output.expanduser().resolve() / ".wearable_state.sqlite")
    try:
        report_payload = load_processing_report(
            state_file,
            args.run_id,
            include_details=args.include_details,
        )
    except (FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report_payload, indent=2, sort_keys=True))
    else:
        print(format_processing_report(report_payload))
        for participant, reason in sorted(
            report_payload["failure_behavior"]["blocked_participants"].items()
        ):
            print(f"BLOCKED {participant}: {reason}")
        for participant, reason in sorted(
            report_payload["failure_behavior"]["failed_participants"].items()
        ):
            print(f"FAILED {participant}: {reason}")
    return 1 if (
        report_payload["participants"]["failed"]
        or report_payload["participants"]["blocked"]
    ) else 0


def _registry_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.processing.apple_healthkit.registry import REGISTRY_VERSION, known_features

    payload = {
        "registry_version": REGISTRY_VERSION, "features": list(known_features()),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Registry version: {REGISTRY_VERSION}")
        print("\n".join(payload["features"]))
    return 0
