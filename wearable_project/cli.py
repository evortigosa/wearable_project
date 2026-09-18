"""
Wearable Data Processing and Modeling project
CLI for native participant-feature processing.
"""


from __future__ import annotations
import argparse
import csv
import io
import json
import sys
from pathlib import Path
from wearable_project import __version__
from wearable_project.processing.pipeline import process_dataset
from wearable_project.processing.registry import REGISTRY_VERSION, known_features
from wearable_project.processing.tracker import format_processing_report, load_processing_report
from wearable_project.curation.curate_cli import add_curation_commands, handle_curation_command


def participant_selection(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wearable-project", description="Clean cumulative Apple HealthKit exports into participant/feature CSV folders.",
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

    process_scan = commands.add_parser(
        "process-scan", help="Read-only scan of what the next native process run would skip, update, rebuild, or block",
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
        "--max-listed", type=int, default=50, help="Maximum non-skip participants listed in the human-readable report",
    )
    process_scan.add_argument(
        "--fail-if-work-needed", action="store_true",
        help="Return exit code 3 when the scan is safe but incremental work or rebuilds are needed",
    )
    process_scan.add_argument("--no-progress", action="store_true", help="Disable the hashing progress bar")

    processing_environment = commands.add_parser(
        "processing-environment", help="Report the imported native processing source, versions, and frozen-module integrity",
    )
    processing_environment.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    report = commands.add_parser("report", help="Read a persisted processing report")
    location = report.add_mutually_exclusive_group(required=True)
    location.add_argument("--output", type=Path, help="Cleaned output root containing .wearable_state.sqlite")
    location.add_argument("--state-file", type=Path, help="Explicit processing-state SQLite file")
    report.add_argument("--run-id", help="Specific run ID; defaults to the latest tracked run")
    report.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    report.add_argument(
        "--include-details", action="store_true", help="Include participant, feature, and diagnostic rows in JSON output",
    )

    registry = commands.add_parser("registry", help="List native processing feature policies")
    registry.add_argument("--json", action="store_true")

    curation_registry = commands.add_parser(
        "curation-registry", help="Inspect and validate the curation feature-policy matrix",
    )
    curation_registry.add_argument(
        "--format", choices=("table", "json", "csv"), default="table", help="Output representation",
    )
    curation_registry.add_argument(
        "--feature", action="append", dest="features", help="Restrict output to one feature; repeat for several features",
    )
    curation_registry.add_argument(
        "--include-evidence", action="store_true", help="Include the evidence catalog in JSON output",
    )
    curation_registry.add_argument(
        "--include-rules", action="store_true", help="Include rule declarations in JSON output",
    )
    curation_registry.add_argument(
        "--output", type=Path, help="Write output to this file instead of stdout",
    )

    describe = commands.add_parser(
        "describe-feature", help="Explain one or all HealthKit features, their native semantics, caveats, and evidence",
    )
    describe.add_argument("feature", nargs="?", help="Feature name; omit only with --all")
    describe.add_argument("--all", action="store_true", help="Render the complete feature guide")
    describe.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    describe.add_argument("--include-rules", action="store_true")
    describe.add_argument("--no-sources", action="store_true", help="Omit evidence-source details")
    describe.add_argument("--output", type=Path)

    evidence = commands.add_parser("evidence", help="Inspect the packaged curation evidence catalog")
    evidence.add_argument("--feature", help="Restrict to evidence used by one feature")
    evidence.add_argument("--kind", help="Restrict to one evidence kind")
    evidence.add_argument("--format", choices=("table", "json", "markdown"), default="table")
    evidence.add_argument("--output", type=Path)

    explain_rule = commands.add_parser("explain-rule", help="Explain one curation rule and its evidence")
    explain_rule.add_argument("rule_id")
    explain_rule.add_argument("--format", choices=("text", "json"), default="text")

    explain_unit = commands.add_parser("explain-unit-policy", help="Explain one feature's unit policy and calibration gate")
    explain_unit.add_argument("feature")
    explain_unit.add_argument("--format", choices=("text", "json"), default="text")

    audit = commands.add_parser(
        "curation-audit", help="Run a read-only policy-calibration audit over a native processing root",
    )
    audit.add_argument("--input-native", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--workers", type=int, default=4)
    audit.add_argument("--max-in-flight", type=int)
    audit.add_argument("--policies", choices=("provisional", "calibration", "all"), default="calibration")
    audit.add_argument("--feature", action="append", dest="audit_features")
    audit.add_argument("--participants-file", type=Path)
    audit.add_argument(
        "--allow-unmanaged-native-root", action="store_true",
        help=(
            "Allow a native-copy root without .wearable_state.sqlite. Curated roots "
            "are still rejected by state-marker and CSV-header checks."
        ),
    )
    audit.add_argument("--overwrite", action="store_true")
    audit.add_argument("--json-summary", action="store_true")

    environment = commands.add_parser(
        "curation-environment", help="Report the imported curation source, wheel location, fingerprints, and module integrity",
    )
    environment.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    decisions = commands.add_parser(
        "policy-decisions", help="Inspect the versioned human calibration decisions used by curation",
    )
    decisions.add_argument("--format", choices=("table", "json", "csv"), default="table")
    decisions.add_argument("--feature", action="append", dest="decision_features")
    decisions.add_argument("--output", type=Path)

    add_curation_commands(commands)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    handled = handle_curation_command(args)
    if handled is not None:
        return handled

    if args.command == "describe-feature":
        from wearable_project.curation.explain import (
            feature_explanation, render_all_guides_markdown, render_feature_markdown,
            render_feature_text, render_json,
        )
        from wearable_project.curation.guidance import known_guides
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
                        include_rules=args.include_rules, include_sources=include_sources,
                    )
                elif args.format == "json":
                    payload = {
                        "features": {
                            name: feature_explanation(
                                name, include_rules=args.include_rules, include_sources=include_sources,
                            ) for name in known_guides()
                        }
                    }
                    rendered = render_json(payload)
                else:
                    rendered = "\n".join(
                        render_feature_text(
                            name, include_rules=args.include_rules, include_sources=include_sources,
                        ).rstrip() for name in known_guides()
                    ) + "\n"
            elif args.format == "json":
                rendered = render_json(feature_explanation(
                    args.feature, include_rules=args.include_rules, include_sources=include_sources,
                ))
            elif args.format == "markdown":
                rendered = render_feature_markdown(
                    args.feature, include_rules=args.include_rules, include_sources=include_sources,
                )
            else:
                rendered = render_feature_text(
                    args.feature, include_rules=args.include_rules, include_sources=include_sources,
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

    if args.command == "evidence":
        from wearable_project.curation.evidence import EVIDENCE, evidence_by_kind
        from wearable_project.curation.explain import evidence_for_feature, render_json
        try:
            sources = list(evidence_for_feature(args.feature)) if args.feature else list(EVIDENCE.values())
        except KeyError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if args.kind:
            sources = [source for source in sources if source.kind == args.kind]
        sources.sort(key=lambda source: source.evidence_id)
        if args.format == "json":
            rendered = render_json({source.evidence_id: source.__dict__ if hasattr(source, "__dict__") else {
                field: getattr(source, field) for field in source.__dataclass_fields__
            } for source in sources})
        elif args.format == "markdown":
            lines = ["# Curation evidence catalog", ""]
            for source in sources:
                lines.extend([
                    f"## {source.evidence_id}", "",
                    f"**{source.citation_text}**", "", source.brief_summary, "",
                    f"Source: {source.locator}", "",
                    f"Kind: `{source.kind}`", "",
                ])
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

    if args.command == "explain-rule":
        from wearable_project.curation.explain import render_json, rule_explanation
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

    if args.command == "explain-unit-policy":
        from wearable_project.curation.explain import render_json, unit_policy_explanation
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
                    f"  {measurement['measurement']}: raw={measurement['raw_unit'] or measurement['raw_unit_candidates']} "
                    f"canonical={measurement['canonical_unit']} rule={measurement['conversion_rule']}"
                )
        return 0

    if args.command == "processing-environment":
        from wearable_project.processing.environment import processing_environment_manifest
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
            invalid = [name for name, valid in payload.get('core_module_integrity', {}).items() if not valid]
            print(f"Frozen processing modules: {'valid' if not invalid else 'mismatch: ' + ', '.join(invalid)}")
            if payload.get('git_commit'):
                print(f"Git commit: {payload['git_commit']}")
            if payload.get('warnings'):
                print("Warnings:")
                for warning in payload['warnings']:
                    print(f"  - {warning}")
        return 1 if payload.get("warnings") else 0

    if args.command == "curation-environment":
        from wearable_project.curation.environment import environment_manifest
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
            if payload.get('git_commit'):
                print(f"Git commit: {payload['git_commit']}")
            if payload.get('warnings'):
                print("Warnings:")
                for warning in payload['warnings']:
                    print(f"  - {warning}")
        return 1 if payload.get("warnings") else 0

    if args.command == "policy-decisions":
        from wearable_project.curation.decisions import (
            decisions_fingerprint, get_decision, known_decisions, validate_decisions,
        )
        selected = tuple(args.decision_features or known_decisions())
        unknown = sorted(set(selected) - set(known_decisions()))
        if unknown:
            print(f"ERROR: unknown calibration decision feature(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        errors = validate_decisions()
        rows = [get_decision(name).as_dict() | {
            "decision_fingerprint": get_decision(name).fingerprint(),
        } for name in sorted(set(selected))]
        if args.format == "json":
            rendered = json.dumps({
                "decision_set_fingerprint": decisions_fingerprint(),
                "validation_errors": list(errors),
                "decisions": {row["feature"]: row for row in rows},
            }, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        elif args.format == "csv":
            buffer = io.StringIO()
            fields = []
            for row in rows:
                for key in row:
                    if key not in fields:
                        fields.append(key)
            writer = csv.DictWriter(buffer, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: ";".join(value) if isinstance(value, (tuple, list)) else value
                    for key, value in row.items()
                })
            rendered = buffer.getvalue()
        else:
            lines = [
                "Feature                    Maturity     Grade          Execution mode",
                "-------------------------  -----------  -------------  --------------------------",
            ]
            for row in rows:
                lines.append(
                    f"{row['feature']:<25}  {row['maturity']:<11}  {row['evidence_grade']:<13}  {row['execution_mode']}"
                )
            rendered = "\n".join(lines) + "\n"
        if args.output:
            target = args.output.expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 1 if errors else 0

    if args.command == "curation-audit":
        from wearable_project.curation.audit import (
            AuditError, format_audit_summary, run_curation_audit,
        )
        try:
            summary = run_curation_audit(
                args.input_native, args.output, workers=args.workers,
                max_in_flight=args.max_in_flight, policy_scope=args.policies,
                features=args.audit_features,
                selected_participants=participant_selection(args.participants_file),
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

    if args.command == "curation-registry":
        from wearable_project.curation.registry import (
            get_policy, matrix_csv, matrix_table, registry_payload, validate_registry,
        )

        features = tuple(args.features) if args.features else None
        if features:
            unknown = [feature for feature in features if get_policy(feature).identity.maturity.value == "unknown"]
            if unknown:
                print(f"ERROR: unknown curation feature(s): {', '.join(unknown)}", file=sys.stderr)
                return 2
        validation = validate_registry()
        if args.format == "json":
            rendered = json.dumps(
                registry_payload(
                    features=features, include_evidence=args.include_evidence, include_rules=args.include_rules,
                ),
                indent=2, sort_keys=True, ensure_ascii=False,
            ) + "\n"
        elif args.format == "csv":
            if args.include_evidence or args.include_rules:
                print("ERROR: --include-evidence/--include-rules require --format json", file=sys.stderr)
                return 2
            rendered = matrix_csv(features)
        else:
            if args.include_evidence or args.include_rules:
                print("ERROR: --include-evidence/--include-rules require --format json", file=sys.stderr)
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

    if args.command == "registry":
        payload = {"registry_version": REGISTRY_VERSION, "features": list(known_features())}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Registry version: {REGISTRY_VERSION}")
            print("\n".join(payload["features"]))
        return 0

    if args.command == "process-scan":
        from wearable_project.processing.scan import format_processing_scan, scan_processing_plan
        try:
            scan = scan_processing_plan(
                args.input, args.output, workers=args.workers, mode=args.mode,
                snapshot_policy=args.snapshot_policy,
                selected_participants=participant_selection(args.participants_file),
                verify_existing_hashes=args.verify_existing_hashes,
                state_file=args.state_file, show_progress=not args.no_progress,
            )
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if args.json:
            rendered = json.dumps(
                scan.as_dict(include_details=args.include_details),
                ensure_ascii=False, indent=2, sort_keys=True,
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

    if args.command == "report":
        state_file = args.state_file or (args.output.expanduser().resolve() / ".wearable_state.sqlite")
        try:
            report_payload = load_processing_report(
                state_file, args.run_id, include_details=args.include_details
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
            report_payload["participants"]["failed"] or report_payload["participants"]["blocked"]
        ) else 0

    from wearable_project.processing.environment import processing_environment_manifest
    processing_env = processing_environment_manifest()
    for warning in processing_env.warnings:
        print(f"PROCESSING ENVIRONMENT WARNING: {warning}", file=sys.stderr)

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
        print(format_processing_report(summary.as_dict()))
        for participant, reason in sorted(summary.blocks.items()):
            print(f"BLOCKED {participant}: {reason}")
        for participant, reason in sorted(summary.failures.items()):
            print(f"FAILED {participant}: {reason}")
    return 1 if summary.failed or summary.blocked else 0
