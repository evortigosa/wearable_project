"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import pytest
from wearable_project.cli import build_parser, main


APPLE = "apple-healthkit"
GOOGLE = "google-fitness"


def test_root_cli_import_is_source_neutral() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; import wearable_project.cli; "
            "assert 'wearable_project.processing.apple_healthkit.pipeline' not in sys.modules; "
            "assert 'wearable_project.curation.apple_healthkit.pipeline' not in sys.modules"
        ),
    ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_processing_command_dispatches_to_processing_stage() -> None:
    args = build_parser().parse_args(["registry", APPLE, "--json"])
    assert args.command == "registry"
    assert args.source == APPLE
    assert args._handler.__module__ == "wearable_project.processing.cli"


def test_curation_command_dispatches_to_curation_stage() -> None:
    args = build_parser().parse_args(
        ["curation-registry", APPLE, "--feature", "Sleep", "--format", "json"]
    )
    assert args.command == "curation-registry"
    assert args.source == APPLE
    assert args._handler.__module__ == "wearable_project.curation.cli"


@pytest.mark.parametrize(
    "command",
    [
        "process",
        "process-scan",
        "processing-environment",
        "report",
        "registry",
        "curate",
        "run",
        "curation-report",
        "curation-registry",
        "describe-feature",
        "evidence",
        "explain-rule",
        "explain-unit-policy",
        "curation-audit",
        "curation-environment",
        "policy-decisions",
    ],
)
def test_source_is_required(command: str) -> None:
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args([command])
    assert caught.value.code == 2


def test_google_fitness_is_registered_for_processing_and_curation_but_not_combined_run() -> None:
    args = build_parser().parse_args(["registry", GOOGLE, "--json"])
    assert args.source == GOOGLE
    assert args._handler.__module__ == "wearable_project.processing.cli"
    curate = build_parser().parse_args([
        "curate", GOOGLE, "--input-native", "native", "--output", "curated"
    ])
    assert curate.source == GOOGLE
    assert curate._handler.__module__ == "wearable_project.curation.cli"
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(["run", GOOGLE])
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "removed_command",
    [
        "curation-acceptance",
        "curation-cohort-acceptance",
        "curation-schema-migration-audit",
    ],
)
def test_release_only_curation_commands_are_not_registered(removed_command: str) -> None:
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args([removed_command, GOOGLE])
    assert caught.value.code == 2


def test_processing_registry_works_through_explicit_source(capsys) -> None:
    code = main(["registry", APPLE, "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["registry_version"]
    assert "Sleep" in payload["features"]


def test_root_help_and_source_help_are_available() -> None:
    root = subprocess.run(
        [sys.executable, "-m", "wearable_project", "--help"],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )
    assert root.returncode == 0
    assert "source-specific wearable exports" in root.stdout
    assert "process" in root.stdout
    assert "curate" in root.stdout

    source = subprocess.run(
        [sys.executable, "-m", "wearable_project", "process", "--help"],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )
    assert source.returncode == 0
    assert "apple-healthkit" in source.stdout
    assert "google-fitness" in source.stdout
