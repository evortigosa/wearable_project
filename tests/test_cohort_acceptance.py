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
import stat
import subprocess
import sys
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
GOOGLE_NATIVE = Path(os.environ.get(
    "WEARABLE_GOOGLE_NATIVE_SAMPLE",
    "/home/evortigosa/Desktop/postdoc/code/cluster/wearable_data/cleaned_google_fitness",
))
GOOGLE_CURATED = Path(os.environ.get(
    "WEARABLE_GOOGLE_CURATED_SAMPLE",
    "/home/evortigosa/Desktop/postdoc/code/cluster/wearable_data/curated_google_fitness",
))
needs_samples = pytest.mark.skipif(not (NATIVE.is_dir() and CURATED.is_dir()), reason="sample roots required")
needs_google_samples = pytest.mark.skipif(
    not (GOOGLE_NATIVE.is_dir() and GOOGLE_CURATED.is_dir()),
    reason="Google Fitness sample roots required",
)


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


def _run(tmp_path: Path, native: Path, curated: Path, *extra: str, source: str | None = None,):
    out = tmp_path / "report"
    args = ["--native", str(native), "--curated", str(curated), "--out", str(out)]
    if source is not None:
        args.extend(("--source", source))
    code = cohort_acceptance.main([*args, *extra])
    assert (out / "cohort_acceptance.txt").read_text().startswith("Cohort acceptance")
    return code, json.loads((out / "cohort_acceptance.json").read_text())


def _run_subprocess(tmp_path: Path, native: Path, curated: Path, *extra: str, source: str,):
    """Exercise the real CLI in an isolated process, as full-cohort deployment does."""

    out = tmp_path / "report"
    command = [
        sys.executable,
        "-m",
        "wearable_project.utils.cohort_acceptance",
        "--source",
        source,
        "--native",
        str(native),
        "--curated",
        str(curated),
        "--out",
        str(out),
        *extra,
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert (out / "cohort_acceptance.txt").read_text().startswith("Cohort acceptance"), completed.stderr
    return completed.returncode, json.loads((out / "cohort_acceptance.json").read_text())


def _checks(result, level):
    return {(f["phase"], f["feature"], f["check"]) for f in result["findings"] if f["level"] == level}


def _sharing_warnings(
    features, native: Path = NATIVE, curated: Path = CURATED, *, source: str = "apple_healthkit",
) -> set:
    """The sharing warnings the samples' own permission bits call for: none where they are open to their group."""

    expected = set()
    for phase, root in (("native", native), ("curated", curated)):
        folders = [root, *(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))]
        for feature in features:
            if any(not os.stat(p).st_mode & stat.S_IRGRP for p in root.glob(f"[!.]*/{feature}.csv")):
                expected.add((phase, feature, "files not readable by their group"))
        if any(os.stat(p).st_mode & (stat.S_IRGRP | stat.S_IXGRP) != stat.S_IRGRP | stat.S_IXGRP for p in folders):
            expected.add((phase, None, "folders not open to their group"))
        database = root / {"native": ".wearable_state.sqlite", "curated": ".wearable_curation_state.sqlite"}[phase]
        if database.is_file() and not os.stat(database).st_mode & stat.S_IRGRP:
            expected.add((phase, None, "state database not readable by its group"))
        if source == "google_fitness" and phase == "curated":
            contract = root / ".curation_contract.json"
            if contract.is_file() and not os.stat(contract).st_mode & stat.S_IRGRP:
                expected.add((phase, None, "root support file not readable by its group"))
    return expected


@needs_samples
def test_the_healthy_samples_pass_and_are_left_untouched(tmp_path):
    before = {root: _fingerprint(root) for root in (NATIVE, CURATED)}
    code, result = _run_subprocess(
        tmp_path,
        NATIVE,
        CURATED,
        "--participants",
        "10",
        source="apple_healthkit",
    )
    expected = _sharing_warnings(result["features"])  # what the samples' own file modes call for, if anything
    assert code == 0 and result["verdict"] == ("PASS WITH WARNINGS" if expected else "PASS")
    assert result["wearable_source"] == "apple_healthkit"
    assert {(f["phase"], f["feature"], f["check"]) for f in result["findings"]} == expected
    assert all(f["level"] == "WARN" for f in result["findings"])
    for phase in ("native", "curated"):
        assert len(result["profiles"][phase]) == len(result["loads"][phase]) == len(result["features"]) == 33
    assert len(result["participants_sampled"]) == 10
    assert {root: _fingerprint(root) for root in (NATIVE, CURATED)} == before


@needs_google_samples
def test_the_healthy_google_samples_pass_all_source_qualified_features_and_are_left_untouched(tmp_path):
    before = {root: _fingerprint(root) for root in (GOOGLE_NATIVE, GOOGLE_CURATED)}
    code, result = _run_subprocess(
        tmp_path,
        GOOGLE_NATIVE,
        GOOGLE_CURATED,
        "--participants",
        "0",
        source="google_fitness",
    )
    expected = _sharing_warnings(
        result["features"],
        GOOGLE_NATIVE,
        GOOGLE_CURATED,
        source="google_fitness",
    )
    assert code == 0 and result["verdict"] == ("PASS WITH WARNINGS" if expected else "PASS")
    assert result["wearable_source"] == "google_fitness"
    assert {(f["phase"], f["feature"], f["check"]) for f in result["findings"]} == expected
    assert all(f["level"] == "WARN" for f in result["findings"])
    assert len(result["features"]) == 67
    for phase in ("native", "curated"):
        assert len(result["profiles"][phase]) == len(result["loads"][phase]) == 67
        assert result["profiles"][phase]["StepCount"]["rows"] == 6793
        assert result["loads"][phase]["StepCount"]["rows"] == 6793
        assert result["profiles"][phase]["Walking"]["rows"] == 68751
    assert len(result["participants_sampled"]) == 4
    assert {root: _fingerprint(root) for root in (GOOGLE_NATIVE, GOOGLE_CURATED)} == before


@needs_samples
def test_a_damaged_file_fails_the_run(tmp_path):
    native = _mirror(NATIVE, tmp_path / "native")
    victim = sorted(p for p in native.iterdir() if p.is_dir() and (p / "StepCount.csv").exists())[0] / "StepCount.csv"
    lines = victim.read_text().split("\n")
    last = lines[-2]
    lines[-2] = last[:-1] + ("7" if last[-1] != "7" else "3")  # same size, still valid
    victim.unlink()
    victim.write_text("\n".join(lines))
    # Every participant is loaded, so the damaged file is always among them, whatever the sample's size.
    code, result = _run(tmp_path, native, CURATED, "--features", "StepCount", "--participants", "0")
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


@needs_google_samples
def test_google_curated_contract_group_permission_is_checked(tmp_path):
    curated = _mirror(GOOGLE_CURATED, tmp_path / "curated")
    contract = curated / ".curation_contract.json"
    contract.chmod(contract.stat().st_mode & ~stat.S_IRGRP)
    code, result = _run_subprocess(
        tmp_path,
        GOOGLE_NATIVE,
        curated,
        "--features",
        "StepCount",
        "--profile-only",
        source="google_fitness",
    )
    assert code == 0 and result["verdict"] == "PASS WITH WARNINGS"
    assert ("curated", None, "root support file not readable by its group") in _checks(result, "WARN")


@needs_google_samples
def test_google_feature_catalog_rejects_apple_only_features(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "wearable_project.utils.cohort_acceptance",
            "--source",
            "google_fitness",
            "--native",
            str(GOOGLE_NATIVE),
            "--curated",
            str(GOOGLE_CURATED),
            "--features",
            "HeartRate",
            "--out",
            str(tmp_path / "report"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 2
    assert "unknown feature(s): HeartRate" in completed.stderr
    assert not (tmp_path / "report").exists()


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
