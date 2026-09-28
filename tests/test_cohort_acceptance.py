"""
Wearable Data Processing and Modeling project
Tests for ``utils.cohort_acceptance``: it must pass healthy roots, fail damaged ones, warn without failing where
the loaders deliberately report instead of refusing, and never write into a data root.
"""


from __future__ import annotations
import hashlib
import json
import os
import shutil
from pathlib import Path
import pytest
from wearable_project.utils import cohort_acceptance


NATIVE = Path(os.environ.get(
    "WEARABLE_NATIVE_SAMPLE",
    "/home/evortigosa/Desktop/postdoc/code/cluster/wearable_data/cleaned_apple_healthkit"
))
CURATED = Path(os.environ.get(
    "WEARABLE_CURATED_SAMPLE",
    "/home/evortigosa/Desktop/postdoc/code/cluster/wearable_data/curated_apple_healthkit"
))
needs_samples = pytest.mark.skipif(not (NATIVE.is_dir() and CURATED.is_dir()), reason="sample roots required")


def _mirror(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True)
    for entry in source.iterdir():
        if entry.is_file():
            shutil.copy2(entry, destination / entry.name)
        elif entry.is_dir() and not entry.name.startswith("."):
            (destination / entry.name).mkdir()
            for file in entry.iterdir():
                (destination / entry.name / file.name).symlink_to(file)
    return destination


def _fingerprint(root: Path) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def _run(tmp_path: Path, native: Path, curated: Path, *extra: str):
    out = tmp_path / "report"
    code = cohort_acceptance.main(["--native", str(native), "--curated", str(curated), "--out", str(out), *extra])
    assert (out / "cohort_acceptance.txt").read_text().startswith("Cohort acceptance")
    return code, json.loads((out / "cohort_acceptance.json").read_text())


def _checks(result, level):
    return {(f["phase"], f["feature"], f["check"]) for f in result["findings"] if f["level"] == level}


@needs_samples
def test_the_healthy_samples_pass_and_are_left_untouched(tmp_path):
    before = {root: _fingerprint(root) for root in (NATIVE, CURATED)}
    code, result = _run(tmp_path, NATIVE, CURATED, "--participants", "10")
    assert code == 0 and result["verdict"] == "PASS" and result["findings"] == []
    for phase in ("native", "curated"):
        assert len(result["profiles"][phase]) == len(result["loads"][phase]) == len(result["features"]) == 33
    assert len(result["participants_sampled"]) == 10
    assert {root: _fingerprint(root) for root in (NATIVE, CURATED)} == before


@needs_samples
def test_a_damaged_file_fails_the_run(tmp_path):
    native = _mirror(NATIVE, tmp_path / "native")
    victim = sorted(p for p in native.iterdir() if p.is_dir() and (p / "StepCount.csv").exists())[0] / "StepCount.csv"
    lines = victim.read_text().split("\n")
    last = lines[-2]
    lines[-2] = last[:-1] + ("7" if last[-1] != "7" else "3")  # same size, still valid
    victim.unlink()
    victim.write_text("\n".join(lines))
    code, result = _run(tmp_path, native, CURATED, "--features", "StepCount", "--participants", "10")
    assert code == 1 and result["verdict"] == "FAIL"
    assert ("native", "StepCount", "verified load") in _checks(result, "FAIL")


@needs_samples
def test_an_unrecorded_participant_fails_the_run(tmp_path):
    native = _mirror(NATIVE, tmp_path / "native")
    (native / "9999999999").mkdir()
    shutil.copy2(next(native.glob("*/StepCount.csv")).resolve(), native / "9999999999" / "StepCount.csv")
    code, result = _run(tmp_path, native, CURATED, "--features", "StepCount", "--profile-only")
    assert code == 1 and ("native", "StepCount", "unrecorded files") in _checks(result, "FAIL")


@needs_samples
def test_a_recorded_file_missing_from_disk_warns_without_failing(tmp_path):
    curated = _mirror(CURATED, tmp_path / "curated")
    victim = sorted(p for p in curated.iterdir() if p.is_dir() and (p / "StepCount.csv").exists())[0]
    (victim / "StepCount.csv").unlink()
    code, result = _run(tmp_path, NATIVE, curated, "--features", "StepCount", "--participants", "10")
    assert code == 0 and result["verdict"] == "PASS WITH WARNINGS"
    warned = _checks(result, "WARN")
    assert ("curated", "StepCount", "recorded files missing from disk") in warned
    assert (None, "StepCount", "native files not yet curated") in warned


@needs_samples
def test_profile_only_runs_no_loads(tmp_path):
    code, result = _run(tmp_path, NATIVE, CURATED, "--features", "Weight", "Sleep", "--profile-only")
    assert code == 0 and result["loads"] == {} and sorted(result["features"]) == ["Sleep", "Weight"]


@needs_samples
@pytest.mark.parametrize("argv", [
    ["--features", "NotAFeature"],
    ["--out", str(NATIVE / "report")],
    ["--out", str(CURATED)],
])


def test_invalid_arguments_and_reports_inside_a_root_are_refused(tmp_path, argv):
    base = ["--native", str(NATIVE), "--curated", str(CURATED)]
    if "--out" not in argv:
        base += ["--out", str(tmp_path / "report")]
    with pytest.raises(SystemExit) as caught:
        cohort_acceptance.main(base + argv)
    assert caught.value.code == 2
    assert not (NATIVE / "report").exists()
