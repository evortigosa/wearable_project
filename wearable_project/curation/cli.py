"""
Wearable Data Processing and Modeling project
Source-aware command-line integration for curation and combined runs.
"""


from __future__ import annotations
import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Callable


APPLE_HEALTHKIT_SOURCE = "apple-healthkit"
GOOGLE_FITNESS_SOURCE = "google-fitness"


def _participants(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _add_apple_only_source(
    command: argparse.ArgumentParser, *, handler: Callable[[argparse.Namespace], int], help_text: str,
) -> argparse.ArgumentParser:
    sources = command.add_subparsers(
        dest="source", required=True, metavar="{apple-healthkit}", title="sources",
        description="Select the source-specific curation implementation.",
    )
    apple = sources.add_parser(APPLE_HEALTHKIT_SOURCE, help=help_text)
    apple.set_defaults(_handler=handler)
    return apple


def _add_google_only_source(
    command: argparse.ArgumentParser, *, handler: Callable[[argparse.Namespace], int], help_text: str,
) -> argparse.ArgumentParser:
    sources = command.add_subparsers(
        dest="source", required=True, metavar="{google-fitness}", title="sources",
        description="Select the source-specific curation implementation.",
    )
    google = sources.add_parser(GOOGLE_FITNESS_SOURCE, help=help_text)
    google.set_defaults(_handler=handler)
    return google


def _add_source_pair(
    command: argparse.ArgumentParser, *, apple_handler: Callable[[argparse.Namespace], int],
    google_handler: Callable[[argparse.Namespace], int], apple_help: str, google_help: str,
) -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    sources = command.add_subparsers(
        dest="source", required=True,
        metavar="{apple-healthkit,google-fitness}", title="sources",
        description="Select the source-specific curation implementation.",
    )
    apple = sources.add_parser(APPLE_HEALTHKIT_SOURCE, help=apple_help)
    apple.set_defaults(_handler=apple_handler)
    google = sources.add_parser(GOOGLE_FITNESS_SOURCE, help=google_help)
    google.set_defaults(_handler=google_handler)
    return apple, google


def _add_curate_arguments(parser: argparse.ArgumentParser, *, google: bool = False) -> None:
    parser.add_argument("--input-native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--mode", choices=("auto", "rebuild"), default="auto")
    parser.add_argument("--participants-file", type=Path)
    parser.add_argument("--verify-existing-hashes", action="store_true")
    if google:
        parser.add_argument(
            "--verify-native-hashes", action="store_true",
            help="Recompute every native CSV SHA-256 instead of trusting matching native state metadata",
        )
    parser.add_argument(
        "--allow-unmanaged-native-root", action="store_true",
        help=(
            "Allow a deliberately unmanaged native-copy root without .wearable_state.sqlite. "
            "Curated roots are still rejected."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--json-summary", action="store_true")


def _add_report_arguments(parser: argparse.ArgumentParser) -> None:
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--output", type=Path)
    location.add_argument("--state-file", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--include-details", action="store_true")


def _add_registry_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    parser.add_argument("--feature", action="append", dest="features")
    parser.add_argument("--include-evidence", action="store_true")
    parser.add_argument("--include-rules", action="store_true")
    parser.add_argument("--output", type=Path)


def _add_describe_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("feature", nargs="?", help="Feature name; omit only with --all")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    parser.add_argument("--include-rules", action="store_true")
    parser.add_argument("--no-sources", action="store_true")
    parser.add_argument("--output", type=Path)


def _add_evidence_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--feature")
    parser.add_argument("--kind")
    parser.add_argument("--format", choices=("table", "json", "markdown"), default="table")
    parser.add_argument("--output", type=Path)


def _add_rule_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("rule_id")
    parser.add_argument("--format", choices=("text", "json"), default="text")


def _add_unit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("feature")
    parser.add_argument("--format", choices=("text", "json"), default="text")


def _add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true")


def _add_decision_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("table", "json", "csv"), default="table")
    parser.add_argument("--feature", action="append", dest="decision_features")
    parser.add_argument("--output", type=Path)


def add_curation_commands(commands: Any) -> None:
    curate_command = commands.add_parser(
        "curate", help="Curate a source-specific native participant-feature root without changing it"
    )
    apple, google = _add_source_pair(
        curate_command,
        apple_handler=_curate_apple_healthkit,
        google_handler=_curate_google_fitness,
        apple_help="Curate an Apple HealthKit native root",
        google_help="Curate a Google Fitness native root",
    )
    _add_curate_arguments(apple)
    _add_curate_arguments(google, google=True)

    run_command = commands.add_parser("run", help="Run source-specific native processing and then curation")
    run = _add_apple_only_source(
        run_command, handler=_run_apple_healthkit,
        help_text="Run Apple HealthKit processing followed by Apple HealthKit curation",
    )
    run.add_argument("--input", type=Path, required=True, help="Raw cumulative export root")
    run.add_argument("--native-output", type=Path, required=True)
    run.add_argument("--curated-output", type=Path, required=True)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--max-in-flight", type=int)
    run.add_argument("--native-mode", choices=("auto", "rebuild"), default="auto")
    run.add_argument("--curation-mode", choices=("auto", "rebuild"), default="auto")
    run.add_argument("--snapshot-policy", choices=("strict-cumulative", "authoritative", "append-only"), default="strict-cumulative")
    run.add_argument("--row-error-policy", choices=("fail-participant", "skip-row"), default="fail-participant")
    run.add_argument("--participants-file", type=Path)
    run.add_argument("--verify-existing-hashes", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.add_argument("--json-summary", action="store_true")

    report_command = commands.add_parser("curation-report", help="Read a persisted source-specific curation report")
    apple, google = _add_source_pair(
        report_command,
        apple_handler=_curation_report_apple_healthkit,
        google_handler=_curation_report_google_fitness,
        apple_help="Read an Apple HealthKit curation report",
        google_help="Read a Google Fitness curation report",
    )
    _add_report_arguments(apple)
    _add_report_arguments(google)

    registry_command = commands.add_parser("curation-registry", help="Inspect a source-specific curation policy matrix")
    apple, google = _add_source_pair(
        registry_command,
        apple_handler=_curation_registry_apple_healthkit,
        google_handler=_curation_registry_google_fitness,
        apple_help="Inspect the Apple HealthKit curation policy registry",
        google_help="Inspect the Google Fitness curation policy registry",
    )
    _add_registry_arguments(apple)
    _add_registry_arguments(google)

    describe_command = commands.add_parser("describe-feature", help="Explain source-specific feature semantics and policy")
    apple, google = _add_source_pair(
        describe_command,
        apple_handler=_describe_feature_apple_healthkit,
        google_handler=_describe_feature_google_fitness,
        apple_help="Explain Apple HealthKit feature policies",
        google_help="Explain Google Fitness feature policies",
    )
    _add_describe_arguments(apple)
    _add_describe_arguments(google)

    evidence_command = commands.add_parser("evidence", help="Inspect a source-specific curation evidence catalog")
    apple, google = _add_source_pair(
        evidence_command,
        apple_handler=_evidence_apple_healthkit,
        google_handler=_evidence_google_fitness,
        apple_help="Inspect Apple HealthKit curation evidence",
        google_help="Inspect Google Fitness curation evidence",
    )
    _add_evidence_arguments(apple)
    _add_evidence_arguments(google)

    rule_command = commands.add_parser("explain-rule", help="Explain one source-specific curation rule")
    apple, google = _add_source_pair(
        rule_command,
        apple_handler=_explain_rule_apple_healthkit,
        google_handler=_explain_rule_google_fitness,
        apple_help="Explain an Apple HealthKit curation rule",
        google_help="Explain a Google Fitness curation rule",
    )
    _add_rule_arguments(apple)
    _add_rule_arguments(google)

    unit_command = commands.add_parser("explain-unit-policy", help="Explain one source-specific unit policy")
    apple, google = _add_source_pair(
        unit_command,
        apple_handler=_explain_unit_policy_apple_healthkit,
        google_handler=_explain_unit_policy_google_fitness,
        apple_help="Explain an Apple HealthKit unit policy",
        google_help="Explain a Google Fitness unit policy",
    )
    _add_unit_arguments(apple)
    _add_unit_arguments(google)

    audit_command = commands.add_parser("curation-audit", help="Run a source-specific read-only calibration audit")
    apple, google = _add_source_pair(
        audit_command,
        apple_handler=_curation_audit_apple_healthkit,
        google_handler=_curation_audit_google_fitness,
        apple_help="Audit an Apple HealthKit native root",
        google_help="Audit a Google Fitness native root",
    )
    apple.add_argument("--input-native", type=Path, required=True)
    apple.add_argument("--output", type=Path, required=True)
    apple.add_argument("--workers", type=int, default=4)
    apple.add_argument("--max-in-flight", type=int)
    apple.add_argument("--policies", choices=("provisional", "calibration", "all"), default="calibration")
    apple.add_argument("--feature", action="append", dest="audit_features")
    apple.add_argument("--participants-file", type=Path)
    apple.add_argument("--allow-unmanaged-native-root", action="store_true")
    apple.add_argument("--overwrite", action="store_true")
    apple.add_argument("--json-summary", action="store_true")
    google.add_argument("--input-native", type=Path, required=True)
    google.add_argument("--output", type=Path, required=True)
    google.add_argument("--workers", type=int, default=4)
    google.add_argument("--max-in-flight", type=int)
    google.add_argument("--participants-file", type=Path)
    google.add_argument("--verify-native-hashes", action="store_true")
    google.add_argument("--overwrite", action="store_true")
    google.add_argument("--no-progress", action="store_true")
    google.add_argument("--json-summary", action="store_true")

    acceptance_command = commands.add_parser(
        "curation-acceptance", help="Run formal source-specific native-to-curated acceptance checks",
    )
    google = _add_google_only_source(
        acceptance_command, handler=_curation_acceptance_google_fitness,
        help_text="Accept a Google Fitness native/curated product pair",
    )
    google.add_argument("--input-native", type=Path, required=True)
    google.add_argument("--input-curated", type=Path, required=True)
    google.add_argument("--output", type=Path, required=True)
    google.add_argument("--workers", type=int, default=4)
    google.add_argument("--max-in-flight", type=int)
    google.add_argument("--reference-manifest", type=Path)
    google.add_argument("--no-verify-hashes", action="store_true")
    google.add_argument("--overwrite", action="store_true")
    google.add_argument("--no-progress", action="store_true")
    google.add_argument("--json-summary", action="store_true")

    cohort_acceptance_command = commands.add_parser(
        "curation-cohort-acceptance",
        help="Run formal full-cohort source-specific native-to-curated acceptance",
    )
    google = _add_google_only_source(
        cohort_acceptance_command, handler=_curation_cohort_acceptance_google_fitness,
        help_text="Accept a complete Google Fitness native/curated cohort pair",
    )
    google.add_argument("--input-native", type=Path, required=True)
    google.add_argument("--input-curated", type=Path, required=True)
    google.add_argument("--output", type=Path, required=True)
    google.add_argument("--expected-snapshot", type=Path, required=True)
    google.add_argument("--workers", type=int, default=8)
    google.add_argument("--max-in-flight", type=int)
    google.add_argument("--overwrite", action="store_true")
    google.add_argument("--no-progress", action="store_true")
    google.add_argument("--json-summary", action="store_true")

    migration_command = commands.add_parser(
        "curation-schema-migration-audit",
        help="Compare frozen Google Fitness curation schema v1 with schema v2",
    )
    google = _add_google_only_source(
        migration_command,
        handler=_curation_schema_migration_google_fitness,
        help_text="Audit semantic equivalence between Google Fitness curated schema v1 and v2",
    )
    google.add_argument("--input-native", type=Path, required=True)
    google.add_argument("--input-curated-v1", type=Path, required=True)
    google.add_argument("--input-curated-v2", type=Path, required=True)
    google.add_argument("--output", type=Path, required=True)
    google.add_argument("--workers", type=int, default=8)
    google.add_argument("--max-in-flight", type=int)
    google.add_argument("--max-examples", type=int, default=100)
    google.add_argument("--overwrite", action="store_true")
    google.add_argument("--no-progress", action="store_true")
    google.add_argument("--json-summary", action="store_true")

    environment_command = commands.add_parser("curation-environment", help="Report curation fingerprints and source integrity")
    apple, google = _add_source_pair(
        environment_command,
        apple_handler=_curation_environment_apple_healthkit,
        google_handler=_curation_environment_google_fitness,
        apple_help="Inspect the Apple HealthKit curation environment",
        google_help="Inspect the Google Fitness curation environment",
    )
    _add_environment_arguments(apple)
    _add_environment_arguments(google)

    decisions_command = commands.add_parser("policy-decisions", help="Inspect source-specific calibration decisions")
    apple, google = _add_source_pair(
        decisions_command,
        apple_handler=_policy_decisions_apple_healthkit,
        google_handler=_policy_decisions_google_fitness,
        apple_help="Inspect Apple HealthKit calibration decisions",
        google_help="Inspect Google Fitness calibration decisions",
    )
    _add_decision_arguments(apple)
    _add_decision_arguments(google)

def _curate_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.curation.apple_healthkit.pipeline import curate_dataset, format_summary

    try:
        summary = curate_dataset(
            args.input_native,
            args.output,
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


def _curation_report_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.curation.apple_healthkit.state import format_curation_report, load_curation_report

    state_file = args.state_file or (args.output.expanduser().resolve() / ".wearable_curation_state.sqlite")
    try:
        report = load_curation_report(
            state_file,
            args.run_id,
            include_details=args.include_details,
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


def _run_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.curation.apple_healthkit.pipeline import curate_dataset, format_summary
    from wearable_project.processing.apple_healthkit.pipeline import process_dataset
    from wearable_project.processing.apple_healthkit.tracker import format_processing_report

    selected = _participants(args.participants_file)
    try:
        native_summary = process_dataset(
            args.input,
            args.native_output,
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
            args.native_output,
            args.curated_output,
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
        "native": native_summary.as_dict(),
        "curation": curated_summary.as_dict(),
    }
    if args.json_summary:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("Phase 1 native processing")
        print(format_processing_report(native_summary.as_dict()))
        print("Phase 2 curation")
        print(format_summary(curated_summary), end="")
    return 1 if curated_summary.failed else 0


def _describe_feature_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.curation.apple_healthkit.explain import (
        feature_explanation, render_all_guides_markdown, render_feature_markdown, render_feature_text, render_json,
    )
    from wearable_project.curation.apple_healthkit.guidance import known_guides

    if args.all and args.feature:
        print("ERROR: provide either FEATURE or --all, not both", file=sys.stderr)
        return 2
    if not args.all and not args.feature:
        print("ERROR: FEATURE is required unless --all is used", file=sys.stderr)
        return 2
    include_sources = not args.no_sources
    try:
        if args.all:
            if args.format == "markdown":
                rendered = render_all_guides_markdown(
                    include_rules=args.include_rules,
                    include_sources=include_sources,
                )
            elif args.format == "json":
                payload = {
                    "features": {
                        name: feature_explanation(
                            name,
                            include_rules=args.include_rules,
                            include_sources=include_sources,
                        )
                        for name in known_guides()
                    }
                }
                rendered = render_json(payload)
            else:
                rendered = "\n".join(
                    render_feature_text(
                        name,
                        include_rules=args.include_rules,
                        include_sources=include_sources,
                    ).rstrip()
                    for name in known_guides()
                ) + "\n"
        elif args.format == "json":
            rendered = render_json(
                feature_explanation(
                    args.feature,
                    include_rules=args.include_rules,
                    include_sources=include_sources,
                )
            )
        elif args.format == "markdown":
            rendered = render_feature_markdown(
                args.feature,
                include_rules=args.include_rules,
                include_sources=include_sources,
            )
        else:
            rendered = render_feature_text(
                args.feature,
                include_rules=args.include_rules,
                include_sources=include_sources,
            )
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.output:
        target = args.output.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def _evidence_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.curation.apple_healthkit.evidence import EVIDENCE
    from wearable_project.curation.apple_healthkit.explain import evidence_for_feature, render_json

    try:
        sources = (
            list(evidence_for_feature(args.feature))
            if args.feature
            else list(EVIDENCE.values())
        )
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.kind:
        sources = [source for source in sources if source.kind == args.kind]
    sources.sort(key=lambda source: source.evidence_id)
    if args.format == "json":
        rendered = render_json(
            {
                source.evidence_id: (
                    source.__dict__
                    if hasattr(source, "__dict__")
                    else {
                        field: getattr(source, field) for field in source.__dataclass_fields__
                    }
                )
                for source in sources
            }
        )
    elif args.format == "markdown":
        lines = ["# Curation evidence catalog", ""]
        for source in sources:
            lines.extend(
                [
                    f"## {source.evidence_id}",
                    "",
                    f"**{source.citation_text}**",
                    "",
                    source.brief_summary,
                    "",
                    f"Source: {source.locator}",
                    "",
                    f"Kind: `{source.kind}`",
                    "",
                ]
            )
            if source.limitations:
                lines.extend([f"Limitation: {source.limitations}", ""])
        rendered = "\n".join(lines).rstrip() + "\n"
    else:
        rendered = "\n".join(
            f"{source.evidence_id}  {source.kind}  {source.citation_text}  {source.locator}"
            for source in sources
        ) + ("\n" if sources else "")
    if args.output:
        target = args.output.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def _explain_rule_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.curation.apple_healthkit.explain import render_json, rule_explanation

    try:
        payload = rule_explanation(args.rule_id)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(render_json(payload), end="")
    else:
        rule = payload["rule"]
        print(f"Rule: {args.rule_id}")
        print(f"Class: {rule['rule_class']}")
        print(f"Severity: {rule['severity']}")
        print(f"Flag: {rule['flag']}")
        print(f"Description: {rule['description']}")
        if payload["sources"]:
            print("Sources:")
            for key, source in payload["sources"].items():
                print(f"  {key}: {source['locator']}")
    return 0


def _explain_unit_policy_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.curation.apple_healthkit.explain import render_json, unit_policy_explanation

    try:
        payload = unit_policy_explanation(args.feature)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(render_json(payload), end="")
    else:
        print(f"Feature: {args.feature}")
        print(f"Maturity: {payload['maturity']}")
        print(f"Evidence grade: {payload['calibration']['evidence_grade']}")
        print(f"Execution mode: {payload['calibration']['execution_mode']}")
        print(f"Safe fallback: {payload['safe_fallback']}")
        for measurement in payload["measurements"]:
            print(
                f"  {measurement['measurement']}: "
                f"raw={measurement['raw_unit'] or measurement['raw_unit_candidates']} "
                f"canonical={measurement['canonical_unit']} "
                f"rule={measurement['conversion_rule']}"
            )
    return 0


def _curation_audit_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.curation.apple_healthkit.audit import AuditError, format_audit_summary, run_curation_audit

    try:
        summary = run_curation_audit(
            args.input_native,
            args.output,
            workers=args.workers,
            max_in_flight=args.max_in_flight,
            policy_scope=args.policies,
            features=args.audit_features,
            selected_participants=_participants(args.participants_file),
            overwrite=args.overwrite,
            allow_unmanaged_native_root=args.allow_unmanaged_native_root,
        )
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json_summary:
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    else:
        print(format_audit_summary(summary))
    return 1 if summary.participants_failed else 0


def _curation_environment_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.curation.apple_healthkit.environment import environment_manifest

    payload = environment_manifest().as_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"Package version: {payload['package_version']}")
        print(f"Release label: {payload['package_release_label']}")
        print(f"Working directory: {payload['current_working_directory']}")
        print(f"Imported package: {payload['imported_package_path']}")
        print(f"Installed distribution: {payload.get('distribution_package_path')}")
        print(f"Import source: {payload['import_source_kind']}")
        print(f"Registry fingerprint: {payload['registry_fingerprint']}")
        print(f"Guidance fingerprint: {payload['guidance_fingerprint']}")
        print(f"Decisions fingerprint: {payload['decisions_fingerprint']}")
        if payload.get("git_commit"):
            print(f"Git commit: {payload['git_commit']}")
        if payload.get("warnings"):
            print("Warnings:")
            for warning in payload["warnings"]:
                print(f"  - {warning}")
    return 1 if payload.get("warnings") else 0


def _policy_decisions_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.curation.apple_healthkit.decisions import (
        decisions_fingerprint, get_decision, known_decisions, validate_decisions,
    )

    selected = tuple(args.decision_features or known_decisions())
    unknown = sorted(set(selected) - set(known_decisions()))
    if unknown:
        print(f"ERROR: unknown calibration decision feature(s): {', '.join(unknown)}", file=sys.stderr,)
        return 2
    errors = validate_decisions()
    rows = [
        get_decision(name).as_dict()
        | {"decision_fingerprint": get_decision(name).fingerprint()}
        for name in sorted(set(selected))
    ]
    if args.format == "json":
        rendered = json.dumps(
            {
                "decision_set_fingerprint": decisions_fingerprint(),
                "validation_errors": list(errors),
                "decisions": {row["feature"]: row for row in rows},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    elif args.format == "csv":
        buffer = io.StringIO()
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ";".join(value) if isinstance(value, (tuple, list)) else value
                    for key, value in row.items()
                }
            )
        rendered = buffer.getvalue()
    else:
        lines = [
            "Feature                    Maturity     Grade          Execution mode",
            "-------------------------  -----------  -------------  --------------------------",
        ]
        for row in rows:
            lines.append(
                f"{row['feature']:<25}  {row['maturity']:<11}  "
                f"{row['evidence_grade']:<13}  {row['execution_mode']}"
            )
        rendered = "\n".join(lines) + "\n"
    if args.output:
        target = args.output.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 1 if errors else 0


def _curation_registry_apple_healthkit(args: argparse.Namespace) -> int:
    from wearable_project.curation.apple_healthkit.registry import (
        get_policy, matrix_csv, matrix_table, registry_payload, validate_registry,
    )

    features = tuple(args.features) if args.features else None
    if features:
        unknown = [
            feature
            for feature in features
            if get_policy(feature).identity.maturity.value == "unknown"
        ]
        if unknown:
            print(f"ERROR: unknown curation feature(s): {', '.join(unknown)}", file=sys.stderr,)
            return 2
    validation = validate_registry()
    if args.format == "json":
        rendered = json.dumps(
            registry_payload(
                features=features,
                include_evidence=args.include_evidence,
                include_rules=args.include_rules,
            ),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ) + "\n"
    elif args.format == "csv":
        if args.include_evidence or args.include_rules:
            print("ERROR: --include-evidence/--include-rules require --format json", file=sys.stderr,)
            return 2
        rendered = matrix_csv(features)
    else:
        if args.include_evidence or args.include_rules:
            print("ERROR: --include-evidence/--include-rules require --format json", file=sys.stderr,)
            return 2
        rendered = matrix_table(features) + "\n"

    if args.output is not None:
        target = args.output.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if not validation.valid:
        for error in validation.errors:
            print(f"REGISTRY ERROR: {error}", file=sys.stderr)
        return 1
    return 0


# ------------------------------------------------------------- Google Fitness curation handlers
def _write_or_print(rendered: str, output: Path | None) -> None:
    if output is None:
        print(rendered, end="")
        return
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")


def _curate_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.curation.google_fitness.pipeline import curate_dataset, format_summary

    try:
        summary = curate_dataset(
            args.input_native,
            args.output,
            workers=args.workers,
            max_in_flight=args.max_in_flight,
            mode=args.mode,
            selected_participants=_participants(args.participants_file),
            verify_existing_hashes=args.verify_existing_hashes,
            verify_native_hashes=args.verify_native_hashes,
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


def _curation_acceptance_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.curation.google_fitness.acceptance import run_acceptance

    try:
        result = run_acceptance(
            args.input_native,
            args.input_curated,
            args.output,
            workers=args.workers,
            max_in_flight=args.max_in_flight,
            reference_manifest=args.reference_manifest,
            verify_hashes=not args.no_verify_hashes,
            overwrite=args.overwrite,
            no_progress=args.no_progress,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json_summary:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"Google Fitness CWP5 acceptance {result['status'].upper()}: "
            f"{result['passed_check_count']}/{result['check_count']} checks passed; "
            f"{result['snapshot']['feature_files']} files; "
            f"{result['snapshot']['curated_rows']} rows"
        )
    return 0 if result["status"] == "pass" else 1


def _curation_cohort_acceptance_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.curation.google_fitness.cohort_acceptance import run_cohort_acceptance

    try:
        result = run_cohort_acceptance(
            args.input_native,
            args.input_curated,
            args.output,
            expected_snapshot=args.expected_snapshot,
            workers=args.workers,
            max_in_flight=args.max_in_flight,
            overwrite=args.overwrite,
            no_progress=args.no_progress,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json_summary:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        snapshot = result["product_acceptance"]["snapshot"]
        print(
            f"Google Fitness CWP6 cohort acceptance {result['status'].upper()}: "
            f"{result['passed_check_count']}/{result['check_count']} checks passed; "
            f"{snapshot['feature_files']} files; {snapshot['curated_rows']} rows"
        )
    return 0 if result["status"] == "pass" else 1


def _curation_schema_migration_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.curation.google_fitness.schema_migration import run_schema_migration_audit

    try:
        result = run_schema_migration_audit(
            args.input_native,
            args.input_curated_v1,
            args.input_curated_v2,
            args.output,
            workers=args.workers,
            max_in_flight=args.max_in_flight,
            max_examples=args.max_examples,
            overwrite=args.overwrite,
            no_progress=args.no_progress,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json_summary:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            f"Google Fitness curation schema migration audit {result['status'].upper()}: "
            f"{result['participants_compared']:,} participants; {result['rows']:,} rows; {result['issue_count']} issue(s)"
        )
    return 0 if result["status"] == "pass" else 1


def _curation_report_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.curation.google_fitness.state import (
        CURATION_STATE_FILENAME, format_curation_report, load_curation_report,
    )
    from wearable_project.exceptions import WearableProjectError

    state_file = args.state_file or (args.output.expanduser().resolve() / CURATION_STATE_FILENAME)
    try:
        report = load_curation_report(
            state_file,
            args.run_id,
            include_details=args.include_details,
        )
    except (FileNotFoundError, KeyError, ValueError, WearableProjectError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_curation_report(report), end="")
        for participant, message in sorted(report.get("failures", {}).items()):
            print(f"FAILED {participant}: {message}")
    return 1 if report["participants"]["failed"] else 0


def _curation_registry_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.curation.google_fitness.evidence import evidence_payload
    from wearable_project.curation.google_fitness.registry import (
        get_policy, matrix_csv, matrix_rows, registry_payload, validate_registry,
    )
    from wearable_project.curation.google_fitness.rules import rules_payload

    features = tuple(args.features) if args.features else None
    if features:
        unknown = []
        for feature in features:
            try:
                get_policy(feature, allow_fallback=False)
            except KeyError:
                unknown.append(feature)
        if unknown:
            print(f"ERROR: unknown curation feature(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
    validation = validate_registry()
    if args.format == "json":
        payload = registry_payload()
        if features is not None:
            wanted = set(features)
            payload["policies"] = [row for row in payload["policies"] if row["feature"] in wanted]
        if args.include_evidence:
            payload["evidence_catalog"] = evidence_payload()
        if args.include_rules:
            payload["rule_registry"] = rules_payload()
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    elif args.format == "csv":
        if args.include_evidence or args.include_rules:
            print("ERROR: --include-evidence/--include-rules require --format json", file=sys.stderr)
            return 2
        rendered = matrix_csv(features)
    else:
        if args.include_evidence or args.include_rules:
            print("ERROR: --include-evidence/--include-rules require --format json", file=sys.stderr)
            return 2
        rows = matrix_rows(features)
        lines = [
            "Feature                    Event kind                    Maturity     Execution mode",
            "-------------------------  ----------------------------  -----------  --------------------------",
        ]
        for row in rows:
            lines.append(
                f"{str(row['feature']):<25}  {str(row['event_kind']):<28}  "
                f"{str(row['maturity']):<11}  {row['execution_mode']}"
            )
        rendered = "\n".join(lines) + "\n"
    _write_or_print(rendered, args.output)
    if not validation.valid:
        for issue in validation.issues:
            print(f"REGISTRY ERROR: {issue}", file=sys.stderr)
        return 1
    return 0


def _describe_feature_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.curation.google_fitness.explain import feature_explanation, render_json
    from wearable_project.curation.google_fitness.guidance import known_guides

    if args.all and args.feature:
        print("ERROR: supply a feature or --all, not both", file=sys.stderr)
        return 2
    if not args.all and not args.feature:
        print("ERROR: feature is required unless --all is used", file=sys.stderr)
        return 2
    names = known_guides() if args.all else (args.feature,)
    try:
        payloads = [feature_explanation(name, include_rules=args.include_rules) for name in names]
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.no_sources:
        for payload in payloads:
            payload.pop("sources", None)
    if args.format == "json":
        rendered = render_json(payloads if args.all else payloads[0])
    elif args.format == "markdown":
        sections: list[str] = []
        for payload in payloads:
            guide = payload["guide"]
            sections.extend([
                f"# {payload['feature']}", "", guide["row_meaning"], "",
                f"**Recommended use:** {guide['recommended_use']}", "",
                f"**Default projection:** {guide['default_projection']}", "",
                f"**Unit interpretation:** {guide['unit_interpretation']}", "",
                f"**Temporal interpretation:** {guide['temporal_interpretation']}", "",
                f"**Provenance interpretation:** {guide['provenance_interpretation']}", "",
            ])
            if guide["limitations"]:
                sections.extend(["## Limitations", *[f"- {item}" for item in guide["limitations"]], ""])
        rendered = "\n".join(sections).rstrip() + "\n"
    else:
        blocks = []
        for payload in payloads:
            guide = payload["guide"]
            blocks.append("\n".join([
                f"Feature: {payload['feature']}",
                f"Row meaning: {guide['row_meaning']}",
                f"Recommended use: {guide['recommended_use']}",
                f"Units: {guide['unit_interpretation']}",
                f"Time: {guide['temporal_interpretation']}",
                f"Provenance: {guide['provenance_interpretation']}",
            ]))
        rendered = "\n\n".join(blocks) + "\n"
    _write_or_print(rendered, args.output)
    return 0


def _evidence_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.curation.google_fitness.evidence import get_evidence, known_evidence
    from wearable_project.curation.google_fitness.registry import get_policy

    ids = list(known_evidence())
    if args.feature:
        try:
            policy = get_policy(args.feature, allow_fallback=False)
        except KeyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        used: set[str] = set(policy.evidence_refs) | set(policy.provenance.evidence_refs)
        for measurement in policy.measurements:
            for unit in measurement.unit_by_version:
                used.update(unit.evidence_refs)
        ids = [item for item in ids if item in used]
    sources = [get_evidence(item) for item in ids]
    if args.kind:
        sources = [source for source in sources if source.kind.value == args.kind]
    if args.format == "json":
        rendered = json.dumps(
            {source.evidence_id: source.as_dict() for source in sources},
            ensure_ascii=False, indent=2, sort_keys=True,
        ) + "\n"
    elif args.format == "markdown":
        lines = ["# Google Fitness curation evidence", ""]
        for source in sources:
            lines.extend([
                f"## {source.evidence_id}", "", f"**{source.title}**", "",
                source.summary, "", f"Authority: {source.authority}", "",
                f"Kind: `{source.kind.value}`", "",
            ])
            if source.url:
                lines.extend([f"URL: {source.url}", ""])
            if source.limitations:
                lines.extend(["Limitations:", *[f"- {item}" for item in source.limitations], ""])
        rendered = "\n".join(lines).rstrip() + "\n"
    else:
        rendered = "\n".join(
            f"{source.evidence_id}  {source.kind.value}  {source.title}"
            for source in sources
        ) + ("\n" if sources else "")
    _write_or_print(rendered, args.output)
    return 0


def _explain_rule_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.curation.google_fitness.explain import render_json, rule_explanation

    try:
        payload = rule_explanation(args.rule_id)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(render_json(payload), end="")
    else:
        rule = payload["rule"]
        print(f"Rule: {args.rule_id}")
        print(f"Class: {rule['rule_class']}")
        print(f"Severity: {rule['severity']}")
        print(f"Flag: {rule['flag']}")
        print(f"Description: {rule['description']}")
    return 0


def _explain_unit_policy_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.curation.google_fitness.explain import render_json, unit_policy_explanation

    try:
        payload = unit_policy_explanation(args.feature)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(render_json(payload), end="")
    else:
        print(f"Feature: {payload['feature']}")
        print(f"Maturity: {payload['maturity']}")
        print(f"Evidence grade: {payload['evidence_grade']}")
        print(f"Execution mode: {payload['execution_mode']}")
        print(f"Safe fallback: {payload['safe_fallback']}")
        for measurement in payload["measurements"]:
            print(f"  {measurement['column']}: {measurement['unit_by_version']}")
    return 0


def _curation_audit_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.curation.google_fitness.audit import run_audit

    try:
        summary = run_audit(
            args.input_native,
            args.output,
            workers=args.workers,
            max_in_flight=args.max_in_flight,
            participants=_participants(args.participants_file),
            verify_native_hashes=args.verify_native_hashes,
            overwrite=args.overwrite,
            progress=not args.no_progress,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json_summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        acceptance = summary["acceptance"]
        print(
            f"Google Fitness CWP2 audit {acceptance['status'].upper()}: "
            f"{summary['audited_rows']:,} rows in {summary['audited_files']:,} files; "
            f"reports written to {summary['output']}"
        )
    return 0 if summary["acceptance"]["status"] in {"pass", "partial_pass"} else 1


def _curation_environment_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.curation.google_fitness.environment import environment_manifest

    payload = environment_manifest().as_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"Package version: {payload['package_version']}")
        print(f"Release label: {payload['package_release_label']}")
        print(f"Import source: {payload['import_source_kind']}")
        print(f"Registry fingerprint: {payload['registry_fingerprint']}")
        print(f"Evidence fingerprint: {payload['evidence_fingerprint']}")
        print(f"Rules fingerprint: {payload['rules_fingerprint']}")
        print(f"Decisions fingerprint: {payload['decisions_fingerprint']}")
        print(f"Guidance fingerprint: {payload['guidance_fingerprint']}")
        if payload["warnings"]:
            print("Warnings:")
            for warning in payload["warnings"]:
                print(f"  - {warning}")
    return 1 if payload["warnings"] else 0


def _policy_decisions_google_fitness(args: argparse.Namespace) -> int:
    from wearable_project.curation.google_fitness.decisions import (
        decisions_fingerprint, get_decision, known_decisions, validate_decisions,
    )

    selected = tuple(args.decision_features or known_decisions())
    unknown = sorted(set(selected) - set(known_decisions()))
    if unknown:
        print(f"ERROR: unknown decision(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    rows = [get_decision(name).as_dict() for name in sorted(set(selected))]
    issues = validate_decisions()
    if args.format == "json":
        rendered = json.dumps(
            {
                "decision_set_fingerprint": decisions_fingerprint(),
                "validation_issues": list(issues),
                "decisions": {row["decision_id"]: row for row in rows},
            },
            ensure_ascii=False, indent=2, sort_keys=True,
        ) + "\n"
    elif args.format == "csv":
        buffer = io.StringIO()
        fields = list(rows[0]) if rows else []
        writer = csv.DictWriter(buffer, fieldnames=fields)
        if rows:
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: ";".join(str(item) for item in value) if isinstance(value, list) else value
                    for key, value in row.items()
                })
        rendered = buffer.getvalue()
    else:
        lines = [
            "Decision                              Status       Selected option",
            "------------------------------------  -----------  ----------------------------------------",
        ]
        for row in rows:
            lines.append(
                f"{row['decision_id']:<36}  {row['status']:<11}  {row['selected_option']}"
            )
        rendered = "\n".join(lines) + "\n"
    _write_or_print(rendered, args.output)
    return 1 if issues else 0
