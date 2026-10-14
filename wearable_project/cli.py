"""
Wearable Data Processing and Modeling project
Root command-line composition for source-specific processing and curation.
"""


from __future__ import annotations
import argparse
from collections.abc import Sequence
from wearable_project import __version__
from wearable_project.curation.cli import add_curation_commands
from wearable_project.processing.cli import add_processing_commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wearable-project",
        description="Process and curate source-specific wearable exports into participant/feature data products.",
    )
    parser.add_argument("--version", action="version", version=f"wearable-project {__version__}",)
    commands = parser.add_subparsers(dest="command", required=True)
    add_processing_commands(commands)
    add_curation_commands(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.error("the selected command does not define a source handler")
    return int(handler(args))
