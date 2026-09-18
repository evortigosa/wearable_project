"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import csv
from pathlib import Path
import pytest
from wearable_project.cli import main
from wearable_project.curation.audit import AuditError, run_curation_audit
from wearable_project.curation.pipeline import (
    CURATION_STATE_FILENAME, NATIVE_STATE_FILENAME, curate_dataset, validate_native_input_root,
)
from wearable_project.exceptions import InputLayoutError


def _write_csv(path: Path, fieldnames: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def _minimal_native_csv(root: Path, *, curated_column: str | None = None) -> Path:
    fields = [
        "start_date", "end_date", "value", "datetime", "created_at", "updated_at",
        "data_source", "collecting_method_version",
    ]
    row = {
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-01-01T00:00:00Z",
        "value": "70",
        "datetime": "2024-01-01T00:00:00Z",
        "created_at": "2024-02-01T00:00:00Z",
        "updated_at": "2024-02-01T00:00:00Z",
        "data_source": "AppleHealthkit",
        "collecting_method_version": "2.0",
    }
    if curated_column is not None:
        fields.append(curated_column)
        row[curated_column] = "pass"
    path = root / "p1" / "HeartRate.csv"
    _write_csv(path, fields, row)
    return path


def test_managed_native_root_is_accepted(tmp_path: Path) -> None:
    root = tmp_path / "native"
    _minimal_native_csv(root)
    (root / NATIVE_STATE_FILENAME).touch()
    assert validate_native_input_root(root) == root.resolve()


def test_curated_state_marker_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "curated"
    _minimal_native_csv(root)
    (root / CURATION_STATE_FILENAME).touch()
    with pytest.raises(InputLayoutError, match="Milestone 2 curated root") as caught:
        validate_native_input_root(root)
    assert NATIVE_STATE_FILENAME in str(caught.value)


def test_curated_csv_header_is_rejected_without_state_marker(tmp_path: Path) -> None:
    root = tmp_path / "curated-copy"
    path = _minimal_native_csv(root, curated_column="curation_status")
    with pytest.raises(InputLayoutError, match="curation-only column") as caught:
        validate_native_input_root(root, allow_unmanaged_native_root=True)
    assert str(path.relative_to(root)) in str(caught.value)
    assert "curation_status" in str(caught.value)


def test_unmanaged_native_copy_requires_explicit_override(tmp_path: Path) -> None:
    root = tmp_path / "native-copy"
    _minimal_native_csv(root)
    with pytest.raises(InputLayoutError, match=NATIVE_STATE_FILENAME):
        validate_native_input_root(root)
    assert validate_native_input_root(root, allow_unmanaged_native_root=True,) == root.resolve()


def test_curate_rejects_curated_input_before_creating_output(tmp_path: Path) -> None:
    root = tmp_path / "curated"
    output = tmp_path / "out"
    _minimal_native_csv(root)
    (root / CURATION_STATE_FILENAME).touch()
    with pytest.raises(InputLayoutError, match="Milestone 2 curated root"):
        curate_dataset(root, output, workers=1, max_in_flight=1)
    assert not output.exists()


def test_audit_rejects_curated_input(tmp_path: Path) -> None:
    root = tmp_path / "curated"
    output = tmp_path / "audit"
    _minimal_native_csv(root)
    (root / CURATION_STATE_FILENAME).touch()
    with pytest.raises(AuditError, match="Milestone 2 curated root"):
        run_curation_audit(root, output, workers=1)
    assert not output.exists()


def test_cli_returns_layout_error_for_curated_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "curated"
    output = tmp_path / "out"
    _minimal_native_csv(root)
    (root / CURATION_STATE_FILENAME).touch()
    result = main([
        "curate", "--input-native", str(root), "--output", str(output), "--workers", "1",
    ])
    captured = capsys.readouterr()
    assert result == 2
    assert "Milestone 2 curated root" in captured.err
    assert not output.exists()
