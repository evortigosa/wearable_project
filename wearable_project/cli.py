"""
Wearable Data Processing and Modeling project
CLI for native participant-feature processing.
"""


from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from wearable_project import __version__
from wearable_project.processing.pipeline import process_dataset
from wearable_project.processing.registry import REGISTRY_VERSION, known_features
from wearable_project.processing.tracker import format_processing_report, load_processing_report
from wearable_project.curation.explain import (
    feature_explanation, render_all_guides_markdown, render_feature_markdown, render_feature_text,
    render_json, evidence_for_feature, rule_explanation, unit_policy_explanation
)
from wearable_project.curation.audit import (
    AuditError, format_audit_summary, run_curation_audit,
)
from wearable_project.curation.registry import (
    get_policy, matrix_csv, matrix_table, registry_payload, validate_registry,
)
from wearable_project.curation.guidance import known_guides
from wearable_project.curation.evidence import EVIDENCE, evidence_by_kind


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
    audit.add_argument("--policies", choices=("provisional", "all"), default="provisional")
    audit.add_argument("--feature", action="append", dest="audit_features")
    audit.add_argument("--participants-file", type=Path)
    audit.add_argument("--overwrite", action="store_true")
    audit.add_argument("--json-summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "describe-feature":
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

    if args.command == "curation-audit":
        try:
            summary = run_curation_audit(
                args.input_native, args.output, workers=args.workers,
                max_in_flight=args.max_in_flight, policy_scope=args.policies,
                features=args.audit_features,
                selected_participants=participant_selection(args.participants_file),
                overwrite=args.overwrite,
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
