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
