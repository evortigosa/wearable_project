"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import csv
import io
import json
from pathlib import Path
from wearable_project.cli import main


def test_curation_registry_json(capsys) -> None:
    code = main(["curation-registry", "--feature", "Sleep", "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert list(payload["policies"]) == ["Sleep"]
    assert payload["validation"]["valid"] is True


def test_curation_registry_csv(capsys) -> None:
    code = main(["curation-registry", "--format", "csv"])
    assert code == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(rows) == 33


def test_curation_registry_output_file(tmp_path: Path) -> None:
    target = tmp_path / "matrix.csv"
    code = main(["curation-registry", "--format", "csv", "--output", str(target)])
    assert code == 0
    assert target.exists()
    assert len(list(csv.DictReader(target.open(encoding="utf-8")))) == 33


def test_curation_registry_rejects_unknown_requested_feature(capsys) -> None:
    code = main(["curation-registry", "--feature", "DoesNotExist", "--format", "json"])
    assert code == 2
    assert "unknown curation feature" in capsys.readouterr().err


def test_evidence_requires_json(capsys) -> None:
    code = main(["curation-registry", "--format", "table", "--include-evidence"])
    assert code == 2
    assert "require --format json" in capsys.readouterr().err


def test_describe_feature_json(capsys) -> None:
    code = main(["describe-feature", "Sleep", "--format", "json", "--include-rules"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["feature"] == "Sleep"
    assert "sleep_detailed_state_overlap" in payload["rules"]


def test_describe_feature_markdown_file(tmp_path: Path) -> None:
    target = tmp_path / "sleep.md"
    code = main(["describe-feature", "Sleep", "--format", "markdown", "--output", str(target)])
    assert code == 0
    assert "# Sleep" in target.read_text()


def test_evidence_feature_filter(capsys) -> None:
    code = main(["evidence", "--feature", "BloodPressure", "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "apple_blood_pressure" in payload


def test_explain_unit_policy(capsys) -> None:
    code = main(["explain-unit-policy", "BloodGlucose", "--format", "json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["calibration"]["execution_mode"] == "source_specific_execution"


def test_curation_environment_json(capsys) -> None:
    code = main(["curation-environment", "--json"])
    assert code in {0, 1}
    payload = json.loads(capsys.readouterr().out)
    assert payload["registry_fingerprint"]
    assert payload["import_source_kind"]


def test_policy_decisions_json(capsys) -> None:
    code = main(["policy-decisions", "--format", "json", "--feature", "DailyDistanceCycling"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert list(payload["decisions"]) == ["DailyDistanceCycling"]
    assert payload["decisions"]["DailyDistanceCycling"]["execution_mode"] == "reviewed_execution"
