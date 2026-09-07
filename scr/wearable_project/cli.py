"""
Wearable Data Processing and Modeling project
Command-line interface for the wearable processing project.
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Sequence
import pandas as pd
from . import __version__
from .cleaning import DEFAULT_SENSITIVE_COLUMNS, DEFAULT_SOURCE_IDENTITY_COLUMNS
from .policies import load_feature_policies
from .processing import ProcessingConfig, process_dataset
from .statistics import StatisticsConfig, summarize_dataset
from .timeseries import SegmentConfig, build_time_series
from .validation import ValidationConfig, validate_raw_dataset


def _parser() -> argparse.ArgumentParser:
    parser= argparse.ArgumentParser(
        prog="wearable-project", description="Parse, validate, summarize, and segment Apple HealthKit-style exports.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    subparsers= parser.add_subparsers(dest="command", required=True)

    validate= subparsers.add_parser("validate", help="Sample raw files and validate structure/payloads.")
    validate.add_argument("input_root", type=Path)
    validate.add_argument("--output", type=Path, default=Path("validation_report.json"))
    validate.add_argument("--sample-rows", type=int, default=25)
    validate.add_argument("--data-source", default="applehealthkit")
    validate.add_argument(
        "--naive-timezone", default="UTC", help="IANA zone for naive timestamps, or 'reject'.",
    )
    validate.add_argument("--include-non-monthly-csv", action="store_true")
    # ---- parse -----------------------------------------------------------
    parse= subparsers.add_parser("parse", help="Parse raw monthly exports into feature tables.")
    parse.add_argument("input_root", type=Path)
    parse.add_argument("output_root", type=Path)
    parse.add_argument("--data-source", default="applehealthkit")
    parse.add_argument("--window", default="5min", help="Fixed window width; use 'none' to preserve events.")
    parse.add_argument("--max-workers", type=int, default=4)
    parse.add_argument("--chunksize", type=int, default=10_000)
    parse.add_argument("--strict", action="store_true")
    parse.add_argument("--no-resume", action="store_true")
    parse.add_argument("--output-format", choices=["csv", "parquet"], default="csv")
    parse.add_argument("--naive-timezone", default="UTC", help="IANA zone for naive timestamps, or 'reject'.")
    parse.add_argument("--fingerprint", choices=["metadata", "sha256"], default="metadata")
    parse.add_argument("--feature-policies", type=Path)
    parse.add_argument("--drop-participant-column", action="store_true")
    parse.add_argument(
        "--sensitive-column", action="append", default=[],
        help="Additional payload column to remove; repeat for multiple columns.",
    )
    parse.add_argument(
        "--source-identity-column", action="append", default=[],
        help=(
            "Additional raw source column used to derive participant-scoped source_key; repeat for multiple columns."
        ),
    )
    parse.add_argument(
        "--merge-sources", action="store_true",
        help=(
            "Do not keep source_key partitions. Use only after defining a study-level source adjudication/merge rule."
        ),
    )
    parse.add_argument("--include-non-monthly-csv", action="store_true")
    parse.add_argument("--max-windows-per-event", type=int, default=100_000)
    # ---- stats -----------------------------------------------------------
    stats= subparsers.add_parser("stats", help="Generate non-destructive coverage and quality statistics.")
    stats.add_argument("input_root", type=Path)
    stats.add_argument("output_root", type=Path)
    stats.add_argument("--max-workers", type=int, default=4)
    stats.add_argument("--day-basis", choices=["utc", "timezone", "source_offset"], default="utc")
    stats.add_argument("--timezone", default="UTC")
    # ---- series ----------------------------------------------------------
    timeseries= subparsers.add_parser("timeseries", help="Build continuous feature time-series datasets.")
    timeseries.add_argument("input_root", type=Path)
    timeseries.add_argument("output_root", type=Path)
    timeseries.add_argument("features", nargs="+")
    timeseries.add_argument("--min-duration", default="60min")
    timeseries.add_argument("--tolerance", default="1min")
    timeseries.add_argument("--duration-basis", choices=["span", "coverage"], default="span")
    timeseries.add_argument("--min-coverage-fraction", type=float, default=0.0)
    timeseries.add_argument("--max-workers", type=int, default=4)

    return parser


def main(argv: Sequence[str] | None= None) -> int:
    args= _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s %(processName)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "validate":
        report= validate_raw_dataset(
            args.input_root,
            ValidationConfig(
                target_data_source=args.data_source, sample_rows_per_file=args.sample_rows,
                monthly_filename_only=not args.include_non_monthly_csv, naive_timezone=args.naive_timezone,
            ),
            output_json=args.output,
        )
        print(json.dumps(report["status_counts"], indent=2, sort_keys=True))
        print(f"Validation report: {args.output}")
        return 1 if report.get("issues_found", 0) else 0

    if args.command == "parse":
        window= None if str(args.window).casefold() in {"none", "off", "false"} else args.window
        reports= process_dataset(
            args.input_root,
            args.output_root,
            ProcessingConfig(
                target_data_source=args.data_source,
                window=window,
                max_workers=args.max_workers,
                resume=not args.no_resume,
                strict=args.strict,
                chunksize=args.chunksize,
                output_format=args.output_format,
                naive_timezone=args.naive_timezone,
                retain_participant_id=not args.drop_participant_column,
                sensitive_columns=tuple(dict.fromkeys([*DEFAULT_SENSITIVE_COLUMNS, *args.sensitive_column])),
                separate_sources=not args.merge_sources,
                source_identity_columns=tuple(dict.fromkeys([*DEFAULT_SOURCE_IDENTITY_COLUMNS, *args.source_identity_column])),
                fingerprint_mode=args.fingerprint,
                monthly_filename_only=not args.include_non_monthly_csv,
                max_windows_per_event=args.max_windows_per_event,
                feature_policies=load_feature_policies(args.feature_policies),
            ),
        )
        failures= sum(item["status"] == "failed" for item in reports)
        statuses= pd.Series([item["status"] for item in reports]).value_counts().to_dict()
        print(json.dumps(statuses, indent=2, sort_keys=True))
        print(f"Processed {len(reports)} participants; failures: {failures}")
        return 1 if failures else 0

    if args.command == "stats":
        outputs= summarize_dataset(
            args.input_root, args.output_root,
            StatisticsConfig(max_workers=args.max_workers, day_basis=args.day_basis, timezone=args.timezone,),
        )
        print(
            f"Feature summary rows: {len(outputs['feature_summary'])}; reported errors: {len(outputs['errors'])}"
        )
        return 1 if not outputs["errors"].empty else 0

    if args.command == "timeseries":
        reports= build_time_series(
            args.input_root, args.output_root, args.features,
            SegmentConfig(
                min_duration=args.min_duration, tolerance=args.tolerance, duration_basis=args.duration_basis,
                min_coverage_fraction=args.min_coverage_fraction, max_workers=args.max_workers,
            ),
        )
        failures= sum(
            report["status"] == "failed"
            for feature_reports in reports.values()
            for report in feature_reports
        )
        print(f"Built {len(reports)} feature datasets; participant-feature failures: {failures}")
        return 1 if failures else 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
