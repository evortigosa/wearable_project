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
GOOGLE_FITNESS_SOURCE = "google-fitness"


def _participant_selection(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _add_processing_sources(
    command: argparse.ArgumentParser, *, apple_handler: Callable[[argparse.Namespace], int],
    google_handler: Callable[[argparse.Namespace], int], apple_help: str, google_help: str,
) -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    sources = command.add_subparsers(
        dest="source",
        required=True,
        metavar="{apple-healthkit,google-fitness}",
        title="sources",
        description="Select the source-specific native-processing implementation.",
    )
    apple = sources.add_parser(APPLE_HEALTHKIT_SOURCE, help=apple_help)
    apple.set_defaults(_handler=apple_handler)
    google = sources.add_parser(GOOGLE_FITNESS_SOURCE, help=google_help)
    google.set_defaults(_handler=google_handler)
    return apple, google


def _add_process_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="Cumulative export directory")
    parser.add_argument("--output", type=Path, required=True, help="Cleaned participant output root")
    parser.add_argument("--workers", type=int, default=4, help="Participant worker processes")
    parser.add_argument("--max-in-flight", type=int, help="Maximum submitted unfinished participant tasks")
    parser.add_argument("--mode", choices=("auto", "rebuild"), default="auto")
    parser.add_argument(
        "--snapshot-policy", choices=("strict-cumulative", "authoritative", "append-only"),
        default="strict-cumulative",
    )
    parser.add_argument(
        "--row-error-policy", choices=("fail-participant", "skip-row"), default="fail-participant",
    )
    parser.add_argument("--participants-file", type=Path, help="Optional participant folder names, one per line")
    parser.add_argument("--verify-existing-hashes", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--state-file", type=Path, help="Defaults to OUTPUT/.wearable_state.sqlite")
    parser.add_argument("--json-summary", action="store_true", help="Print the complete processing report as JSON")


def _add_scan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="Cumulative export directory")
    parser.add_argument("--output", type=Path, required=True, help="Existing or intended native output root")
    parser.add_argument("--workers", type=int, default=4, help="Parallel source-hashing workers")
    parser.add_argument("--mode", choices=("auto", "rebuild"), default="auto")
    parser.add_argument(
        "--snapshot-policy", choices=("strict-cumulative", "authoritative", "append-only"), default="strict-cumulative",
    )
    parser.add_argument("--participants-file", type=Path, help="Optional participant folder names, one per line")
    parser.add_argument("--verify-existing-hashes", action="store_true")
    parser.add_argument("--state-file", type=Path, help="Defaults to OUTPUT/.wearable_state.sqlite")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--include-details", action="store_true", help="Include one planning record per participant in JSON output",
    )
    parser.add_argument(
        "--report-output", type=Path, help="Write the rendered report to a file instead of stdout",
    )
    parser.add_argument(
        "--max-listed", type=int, default=50,
        help="Maximum non-skip participants listed in the human-readable report",
    )
    parser.add_argument(
        "--fail-if-work-needed", action="store_true",
        help="Return exit code 3 when the scan is safe but work is needed",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable hashing progress")


def _add_report_arguments(parser: argparse.ArgumentParser) -> None:
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--output", type=Path, help="Cleaned output root containing .wearable_state.sqlite")
    location.add_argument("--state-file", type=Path, help="Explicit processing-state SQLite file")
    parser.add_argument("--run-id", help="Specific run ID; defaults to the latest tracked run")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--include-details", action="store_true", help="Include participant and feature rows in JSON output",
    )


def add_processing_commands(commands: Any) -> None:
    process_command = commands.add_parser("process", help="Build or safely update source-specific native feature CSVs")
    apple, google = _add_processing_sources(
        process_command,
        apple_handler=_process_apple_healthkit,
        google_handler=_process_google_fitness,
        apple_help="Process Apple HealthKit rows from a cumulative export root",
        google_help="Process Google Fitness rows from a cumulative export root",
    )
    _add_process_arguments(apple)
    _add_process_arguments(google)

    scan_command = commands.add_parser(
        "process-scan", help="Read-only scan of the next source-specific native processing plan"
    )
    apple, google = _add_processing_sources(
        scan_command,
        apple_handler=_process_scan_apple_healthkit,
        google_handler=_process_scan_google_fitness,
        apple_help="Scan the Apple HealthKit processing plan",
        google_help="Scan the Google Fitness processing plan",
    )
    _add_scan_arguments(apple)
    _add_scan_arguments(google)

    environment_command = commands.add_parser(
        "processing-environment", help="Report source-specific native processing versions and integrity",
    )
    apple, google = _add_processing_sources(
        environment_command,
        apple_handler=_processing_environment_apple_healthkit,
        google_handler=_processing_environment_google_fitness,
        apple_help="Inspect the Apple HealthKit processing environment",
        google_help="Inspect the Google Fitness processing environment",
    )
    apple.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    google.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    report_command = commands.add_parser(
        "report", help="Read a persisted source-specific processing report"
    )
    apple, google = _add_processing_sources(
        report_command,
        apple_handler=_report_apple_healthkit,
        google_handler=_report_google_fitness,
        apple_help="Read an Apple HealthKit processing report",
        google_help="Read a Google Fitness processing report",
    )
    _add_report_arguments(apple)
    _add_report_arguments(google)

    registry_command = commands.add_parser(
        "registry", help="List source-specific native processing feature policies"
    )
    apple, google = _add_processing_sources(
        registry_command,
        apple_handler=_registry_apple_healthkit,
        google_handler=_registry_google_fitness,
        apple_help="List Apple HealthKit native feature policies",
        google_help="List Google Fitness native feature policies",
    )
    apple.add_argument("--json", action="store_true")
    google.add_argument("--json", action="store_true")


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


def _process_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.processing.google_fitness.environment import processing_environment_manifest
    from wearable_project.processing.google_fitness.pipeline import process_dataset
    from wearable_project.processing.google_fitness.tracker import format_processing_report

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


def _run_scan(args: argparse.Namespace, scan_function: Any, formatter: Any) -> int:
    try:
        scan = scan_function(
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
        rendered = formatter(scan, max_listed=args.max_listed)
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


def _process_scan_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.processing.apple_healthkit.scan import format_processing_scan, scan_processing_plan
    return _run_scan(args, scan_processing_plan, format_processing_scan)


def _process_scan_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.processing.google_fitness.scan import format_processing_scan, scan_processing_plan
    return _run_scan(args, scan_processing_plan, format_processing_scan)


def _print_environment(payload: dict[str, Any], as_json: bool) -> int:
    if as_json:
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
        if payload.get("cleaner_version"):
            print(f"Cleaner version: {payload['cleaner_version']}")
        if payload.get("pipeline_version"):
            print(f"Pipeline version: {payload['pipeline_version']}")
        invalid = [
            name for name, valid in payload.get("core_module_integrity", {}).items() if not valid
        ]
        print("Frozen processing modules: " + ("valid" if not invalid else "mismatch: " + ", ".join(invalid)))
        if payload.get("warnings"):
            print("Warnings:")
            for warning in payload["warnings"]:
                print(f"  - {warning}")
    return 1 if payload.get("warnings") else 0


def _processing_environment_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.processing.apple_healthkit.environment import processing_environment_manifest
    return _print_environment(processing_environment_manifest().as_dict(), args.json)


def _processing_environment_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.processing.google_fitness.environment import processing_environment_manifest
    return _print_environment(processing_environment_manifest().as_dict(), args.json)


def _run_report(args: argparse.Namespace, loader: Any, formatter: Any) -> int:
    state_file = args.state_file or (args.output.expanduser().resolve() / ".wearable_state.sqlite")
    try:
        payload = loader(state_file, args.run_id, include_details=args.include_details)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(formatter(payload))
        for participant, reason in sorted(payload["failure_behavior"]["blocked_participants"].items()):
            print(f"BLOCKED {participant}: {reason}")
        for participant, reason in sorted(payload["failure_behavior"]["failed_participants"].items()):
            print(f"FAILED {participant}: {reason}")
    return 1 if payload["participants"]["failed"] or payload["participants"]["blocked"] else 0


def _report_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.processing.apple_healthkit.tracker import format_processing_report, load_processing_report
    return _run_report(args, load_processing_report, format_processing_report)


def _report_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.processing.google_fitness.tracker import format_processing_report, load_processing_report
    return _run_report(args, load_processing_report, format_processing_report)


def _registry_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.processing.apple_healthkit.registry import REGISTRY_VERSION, known_features
    payload = {"registry_version": REGISTRY_VERSION, "features": list(known_features())}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Registry version: {REGISTRY_VERSION}")
        print("\n".join(payload["features"]))
    return 0


def _registry_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.processing.google_fitness.registry import registry_payload
    payload = registry_payload()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Registry version: {payload['registry_version']}")
        print(f"Registry fingerprint: {payload['registry_fingerprint']}")
        for item in payload["native_features"]:
            print(item["name"])
    return 0
