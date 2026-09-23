"""
Wearable Data Processing and Modeling project
Tests for ``profile()``, the ``max_rows`` size guard, and acquisition-method coverage reporting.
``profile()`` must describe exactly what an unfiltered ``get_data()`` returns while reading only the phase's
state database and one ``stat`` per file. The size guard must refuse an oversized request before parsing
any CSV when the state gives an exact count, and otherwise stop as soon as the retained rows exceed the
limit. Coverage must describe the returned rows whatever columns are requested.
"""


from __future__ import annotations
import glob
import hashlib
import json
import math
import os
import shutil
import sqlite3
import warnings
from pathlib import Path
import pandas as pd
import pytest
from wearable_project.DataLoaders import DataLoaderSizeError, DataLoaderStateError, ProfileReport
from wearable_project.DataLoaders._base import AppleHealthFeatureLoader
from wearable_project.DataLoaders._profile import ACQUISITION_CAVEAT
from wearable_project.DataLoaders.BloodAlcoholContentLoader import BloodAlcoholContentLoader
from wearable_project.DataLoaders.HeartRateLoader import HeartRateLoader
from wearable_project.DataLoaders.OxygenSaturationLoader import OxygenSaturationLoader
from wearable_project.DataLoaders.StepCountLoader import StepCountLoader
from wearable_project.DataLoaders.WeightLoader import WeightLoader
from wearable_project.exceptions import (
    DataLoaderConfigurationError, DataLoaderError, DataLoaderPathError, DataLoaderReadError,
)


CURATED_SAMPLE = Path(os.environ.get(
    "WEARABLE_CURATED_SAMPLE",
    "/home/evortigosa/Desktop/postdoc/code/PostdocProject/cluster_data/EV_curated_apple_healthkit"
))
NATIVE_SAMPLE = Path(os.environ.get(
    "WEARABLE_NATIVE_SAMPLE",
    "/home/evortigosa/Desktop/postdoc/code/PostdocProject/cluster_data/EV_cleaned_apple_healthkit"
))
needs_sample = pytest.mark.skipif(
    not CURATED_SAMPLE.is_dir() or not NATIVE_SAMPLE.is_dir(),
    reason="curated and native sample roots are required for end-to-end profile tests",
)
CURATED_STATE = ".wearable_curation_state.sqlite"
NATIVE_STATE = ".wearable_state.sqlite"
PARTICIPANT = "1420269268"


def _loader(feature: str):
    module = __import__(f"wearable_project.DataLoaders.{feature}Loader", fromlist=[f"{feature}Loader"])
    return getattr(module, f"{feature}Loader")


def _features() -> list[str]:
    return sorted({os.path.basename(p)[:-4] for p in glob.glob(str(CURATED_SAMPLE / "*" / "*.csv"))})


def _copy_root(tmp_path: Path, source: Path, feature: str, state: str | None) -> Path:
    root = tmp_path / source.name
    for path in source.glob(f"*/{feature}.csv"):
        (root / path.parent.name).mkdir(parents=True)
        shutil.copy2(path, root / path.parent.name / path.name)
    if state is not None:
        shutil.copy2(source / state, root / state)
    return root


def _edit_state(root: Path, state: str, sql: str) -> None:
    connection = sqlite3.connect(root / state)
    connection.execute(sql)
    connection.commit()
    connection.close()
    for suffix in ("-wal", "-shm", "-journal"):
        (root / (state + suffix)).unlink(missing_ok=True)


@pytest.fixture
def count_csv_reads(monkeypatch):
    """Count how many feature CSVs the loader parses."""
    calls = {"n": 0}
    original = AppleHealthFeatureLoader._read_feature_file

    def counting(self, path):
        calls["n"] += 1
        return original(self, path)

    monkeypatch.setattr(AppleHealthFeatureLoader, "_read_feature_file", counting)
    return calls


@pytest.fixture
def restore_default_max_rows():
    saved = AppleHealthFeatureLoader.default_max_rows
    yield
    AppleHealthFeatureLoader.default_max_rows = saved


# ============================================================ profile(): agreement with get_data
@needs_sample
def test_profile_describes_exactly_what_get_data_returns_for_every_feature_and_phase():
    """The central promise of profile(), checked across all 33 features in both phases."""
    disagreements = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for phase, root in (("curated", CURATED_SAMPLE), ("native", NATIVE_SAMPLE)):
            for feature in _features():
                loader = _loader(feature)
                profile = loader(phase=phase, root=root).profile()
                data = loader(phase=phase, root=root).get_data()
                summary, report = profile.summary, data.load_report
                checks = {
                    "rows": summary["rows"] == len(data.df),
                    "estimate": report["size_estimate"]["rows_expected"] == len(data.df),
                    "bytes": summary["bytes_on_disk"] == report["size_estimate"]["bytes_on_disk"],
                    "per_participant": (
                        {code: int(v) for code, v in profile.participants["rows"].items()}
                        == {code: int(v) for code, v in data.df_metadata["rows"].items()}
                    ),
                }
                if phase == "curated":
                    fraction = summary["acquisition"]["classified_fraction"]
                    checks["acquisition_counts"] = (
                        summary["acquisition"]["counts"] == report["acquisition_method_counts"]
                    )
                    checks["classified_fraction"] = math.isclose(
                        fraction, report["acquisition_classified_fraction"]
                    )
                disagreements += [(phase, feature, name) for name, ok in checks.items() if not ok]
    assert disagreements == []


@needs_sample
def test_profile_inclusion_matches_the_default_inclusion_subset():
    profile = WeightLoader(root=CURATED_SAMPLE).profile()
    subset = WeightLoader(root=CURATED_SAMPLE).get_data(default_inclusion_only=True)
    assert profile.summary["default_inclusion"] == {"included": 82, "excluded": 375, "unverifiable_files": 0}
    assert len(subset.df) == 82
    assert subset.load_report["size_estimate"]["rows_expected"] == 82


# ================================================================== profile(): content
@needs_sample
def test_profile_returns_a_report_with_summary_participants_and_text():
    profile = HeartRateLoader(root=CURATED_SAMPLE).profile()
    assert isinstance(profile, ProfileReport)
    assert profile.summary["source"] == "curation_state"
    assert profile.participants.index.name == "RegistrationCode"
    assert list(profile.participants.index) == sorted(profile.participants.index)
    assert str(profile) == profile.text
    assert "HeartRate profile (curated" in profile.text


@needs_sample
def test_profile_as_dict_is_json_serialisable():
    payload = WeightLoader(root=CURATED_SAMPLE).profile().as_dict()
    decoded = json.loads(json.dumps(payload))
    assert decoded["summary"]["rows"] == 457
    assert len(decoded["participants"]) == 7


@needs_sample
def test_profile_participant_table_has_stable_nullable_dtypes():
    participants = WeightLoader(root=CURATED_SAMPLE).profile().participants
    assert str(participants["rows"].dtype) == "Int64"
    assert str(participants["bytes_on_disk"].dtype) == "Int64"
    assert str(participants["on_disk"].dtype) == "boolean"
    assert str(participants["policy_current"].dtype) == "boolean"
    assert str(participants["acquisition_classified_fraction"].dtype) == "Float64"


@needs_sample
def test_profile_reports_acquisition_coverage_and_the_caveat():
    profile = HeartRateLoader(root=CURATED_SAMPLE).profile()
    acquisition = profile.summary["acquisition"]
    assert acquisition["counts"]["unclassified"] == 326_290
    assert math.isclose(acquisition["classified_fraction"], (379_245 - 326_290) / 379_245)
    assert ACQUISITION_CAVEAT in profile.text


@needs_sample
def test_profile_unit_resolution_is_not_applicable_where_the_registry_fixes_the_unit():
    """0% would misreport HeartRate as unresolved; its unit is fixed, not undetermined."""
    assert HeartRateLoader(root=CURATED_SAMPLE).profile().summary["unit_resolution"]["resolved_fraction"] is None
    weight = WeightLoader(root=CURATED_SAMPLE).profile().summary["unit_resolution"]
    assert weight["canonical_value_rows"] == 82 and weight["ambiguous_unit_rows"] == 375
    assert math.isclose(weight["resolved_fraction"], 82 / 457)
    oxygen = OxygenSaturationLoader(root=CURATED_SAMPLE).profile().summary["unit_resolution"]
    assert oxygen["resolved_fraction"] == 1.0
    assert "not applicable" in HeartRateLoader(root=CURATED_SAMPLE).profile().text


@needs_sample
def test_profile_status_counts_match_the_curated_rows():
    summary = WeightLoader(root=CURATED_SAMPLE).profile().summary
    assert sum(summary["status_counts"].values()) == summary["rows"]


@needs_sample
def test_native_profile_reports_rows_but_no_curation_sections():
    summary = StepCountLoader(phase="native", root=NATIVE_SAMPLE).profile().summary
    assert summary["source"] == "native_state"
    assert summary["rows"] == 52_247
    for section in ("status_counts", "default_inclusion", "acquisition", "unit_resolution", "policy"):
        assert summary[section] is None


@needs_sample
def test_profile_limited_to_requested_participants():
    profile = StepCountLoader(root=CURATED_SAMPLE).profile(registration_codes=["10K_1235738253", "1420269268"])
    assert list(profile.participants.index) == ["10K_1235738253", "10K_1420269268"]
    data = StepCountLoader(root=CURATED_SAMPLE).get_data(registration_codes=["10K_1235738253", "1420269268"])
    assert profile.summary["rows"] == len(data.df)


@needs_sample
def test_profile_never_parses_a_feature_csv(count_csv_reads):
    HeartRateLoader(root=CURATED_SAMPLE).profile()
    StepCountLoader(phase="native", root=NATIVE_SAMPLE).profile()
    assert count_csv_reads["n"] == 0


@needs_sample
def test_profile_never_writes(tmp_path: Path):
    root = _copy_root(tmp_path, CURATED_SAMPLE, "HeartRate", CURATED_STATE)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    HeartRateLoader(root=root).profile()
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    assert after == before


# ======================================================== profile(): fallbacks and integrity
@needs_sample
def test_profile_without_state_falls_back_to_the_filesystem(tmp_path: Path):
    root = _copy_root(tmp_path, CURATED_SAMPLE, "StepCount", None)
    profile = StepCountLoader(root=root).profile()
    summary = profile.summary
    assert summary["source"] == "filesystem"
    assert summary["rows"] is None
    assert summary["bytes_on_disk"] > 0 and summary["files_on_disk"] == 10
    assert summary["load"]["default_get_data_rows"] is None
    assert "not checked" in profile.text


@needs_sample
def test_profile_honours_state_validation_off(tmp_path: Path):
    summary = StepCountLoader(root=CURATED_SAMPLE, state_validation="off").profile().summary
    assert summary["source"] == "filesystem"
    assert "state_validation='off'" in summary["source_note"]


@needs_sample
def test_profile_reports_a_file_missing_from_disk(tmp_path: Path):
    root = _copy_root(tmp_path, CURATED_SAMPLE, "StepCount", CURATED_STATE)
    (root / PARTICIPANT / "StepCount.csv").unlink()
    profile = StepCountLoader(root=root).profile()
    assert profile.summary["integrity"]["files_missing_on_disk"] == 1
    assert profile.participants.loc[f"10K_{PARTICIPANT}", "on_disk"] == False  # noqa: E712
    data = StepCountLoader(root=root)
    with pytest.warns(UserWarning, match="no longer on disk"):
        loaded = data.get_data()
    assert profile.summary["rows"] == len(loaded.df)


@needs_sample
def test_profile_reports_a_size_mismatch(tmp_path: Path):
    root = _copy_root(tmp_path, CURATED_SAMPLE, "StepCount", CURATED_STATE)
    path = root / PARTICIPANT / "StepCount.csv"
    path.write_text(path.read_text() + "\n")
    assert StepCountLoader(root=root).profile().summary["integrity"]["files_size_mismatch"] == 1


@needs_sample
def test_profile_reports_a_file_without_state(tmp_path: Path):
    root = _copy_root(tmp_path, NATIVE_SAMPLE, "StepCount", NATIVE_STATE)
    _edit_state(root, NATIVE_STATE,
                f"delete from feature_outputs where participant_id = '{PARTICIPANT}' and feature = 'StepCount'")
    summary = StepCountLoader(phase="native", root=root).profile().summary
    assert summary["integrity"]["files_without_state"] == 1
    assert summary["load"]["default_get_data_rows"] is None


@needs_sample
def test_profile_counts_policy_divergence(tmp_path: Path):
    root = _copy_root(tmp_path, CURATED_SAMPLE, "StepCount", CURATED_STATE)
    _edit_state(root, CURATED_STATE, f"update curation_outputs set policy_fingerprint = 'older' "
                                     f"where participant_id = '{PARTICIPANT}' and feature = 'StepCount'")
    profile = StepCountLoader(root=root).profile()
    assert profile.summary["policy"]["files_diverged"] == 1
    assert profile.participants.loc[f"10K_{PARTICIPANT}", "policy_current"] == False  # noqa: E712


@needs_sample
def test_profile_flags_migrated_inclusion_counts(tmp_path: Path):
    root = _copy_root(tmp_path, CURATED_SAMPLE, "StepCount", CURATED_STATE)
    _edit_state(root, CURATED_STATE, "update curation_outputs set included_by_default_rows = 0, "
                                     "excluded_by_default_rows = 0 where feature = 'StepCount'")
    profile = StepCountLoader(root=root).profile()
    assert profile.summary["default_inclusion"]["unverifiable_files"] == 10
    assert profile.participants["included_by_default_rows"].isna().all()


@needs_sample
def test_profile_flags_a_request_that_exceeds_the_limit(restore_default_max_rows):
    AppleHealthFeatureLoader.default_max_rows = 100_000
    profile = HeartRateLoader(root=CURATED_SAMPLE).profile()
    assert profile.summary["load"]["exceeds_max_rows"] is True
    assert "exceeds the limit" in profile.text


@needs_sample
def test_profile_checks_phase_and_root_like_get_data():
    with pytest.raises(DataLoaderConfigurationError, match="contains native data"):
        StepCountLoader(phase="curated", root=NATIVE_SAMPLE).profile()


def test_profile_on_a_missing_root_raises(tmp_path: Path):
    with pytest.raises(DataLoaderPathError):
        StepCountLoader(root=tmp_path / "absent").profile()


# ====================================================================== size guard
@needs_sample
def test_oversized_request_is_refused_before_any_csv_is_parsed(count_csv_reads):
    with pytest.raises(DataLoaderSizeError, match="would return 379,245 rows"):
        HeartRateLoader(root=CURATED_SAMPLE).get_data(max_rows=100_000)
    assert count_csv_reads["n"] == 0


@needs_sample
def test_native_oversized_request_is_refused_from_native_state(count_csv_reads):
    with pytest.raises(DataLoaderSizeError, match="would return 52,247 rows"):
        StepCountLoader(phase="native", root=NATIVE_SAMPLE).get_data(max_rows=10_000)
    assert count_csv_reads["n"] == 0


@needs_sample
def test_date_filtered_request_is_not_refused_up_front():
    """Every file is read but few rows are kept, so only the retained rows count against the limit."""
    data = HeartRateLoader(root=CURATED_SAMPLE).get_data(
        max_rows=100_000, start_date="2024-06-01", end_date="2024-06-07"
    )
    assert 0 < len(data.df) <= 100_000
    assert data.load_report["size_estimate"]["rows_expected"] is None


@needs_sample
def test_retained_rows_are_limited_incrementally(count_csv_reads):
    with pytest.raises(DataLoaderSizeError, match="exceeded max_rows=100,000"):
        HeartRateLoader(root=CURATED_SAMPLE).get_data(max_rows=100_000, start_date="2015-01-01")
    assert 0 < count_csv_reads["n"] < 9


@needs_sample
def test_limit_applies_without_state_through_the_incremental_check(tmp_path: Path):
    root = _copy_root(tmp_path, CURATED_SAMPLE, "HeartRate", None)
    with pytest.raises(DataLoaderSizeError, match="exceeded"):
        HeartRateLoader(root=root).get_data(max_rows=100_000)
    data = HeartRateLoader(root=root).get_data(max_rows=None)
    assert data.load_report["size_estimate"] is None
    assert data.load_report["size_estimate_note"] == "curation state database absent"


@needs_sample
def test_state_validation_off_reads_no_state_for_the_estimate():
    data = StepCountLoader(root=CURATED_SAMPLE, state_validation="off").get_data()
    assert data.load_report["size_estimate"] is None


@needs_sample
def test_none_disables_the_limit():
    data = HeartRateLoader(root=CURATED_SAMPLE).get_data(max_rows=None)
    assert len(data.df) == 379_245
    assert data.load_report["max_rows"] is None


@needs_sample
def test_default_inclusion_estimate_counts_only_included_rows(count_csv_reads):
    data = WeightLoader(root=CURATED_SAMPLE).get_data(default_inclusion_only=True, max_rows=82)
    assert len(data.df) == 82
    with pytest.raises(DataLoaderSizeError):
        WeightLoader(root=CURATED_SAMPLE).get_data(default_inclusion_only=True, max_rows=81)


@needs_sample
def test_session_default_can_be_lowered_and_per_call_overrides_it(restore_default_max_rows):
    AppleHealthFeatureLoader.default_max_rows = 100_000
    with pytest.raises(DataLoaderSizeError):
        HeartRateLoader(root=CURATED_SAMPLE).get_data()
    assert len(HeartRateLoader(root=CURATED_SAMPLE).get_data(max_rows=None).df) == 379_245


@needs_sample
def test_session_default_can_be_disabled(restore_default_max_rows):
    AppleHealthFeatureLoader.default_max_rows = None
    assert HeartRateLoader(root=CURATED_SAMPLE).get_data().load_report["max_rows"] is None


@pytest.mark.parametrize("value", [0, -5, True, "10", 1.5])
def test_invalid_limits_are_rejected(tmp_path: Path, value):
    (tmp_path / "123").mkdir()
    with pytest.raises(DataLoaderConfigurationError, match="max_rows must be a positive integer or None"):
        StepCountLoader(root=tmp_path).get_data(max_rows=value)


def test_default_limit_is_published():
    from wearable_project.DataLoaders import DEFAULT_MAX_ROWS
    assert AppleHealthFeatureLoader.default_max_rows == DEFAULT_MAX_ROWS == 25_000_000


def test_size_error_is_distinct_from_read_errors():
    assert issubclass(DataLoaderSizeError, DataLoaderError)
    assert not issubclass(DataLoaderSizeError, DataLoaderReadError)


@needs_sample
def test_size_estimate_is_reported():
    report = StepCountLoader(root=CURATED_SAMPLE).get_data().load_report
    estimate = report["size_estimate"]
    assert estimate["source"] == "curation_state"
    assert estimate["rows_expected"] == estimate["rows_stored"] == 52_247
    assert estimate["participants"] == 10
    assert report["max_rows"] == 25_000_000


# ========================================================= writer-activity guard, both journals
@needs_sample
def test_native_hot_journal_is_refused_like_a_curated_one(tmp_path: Path):
    root = _copy_root(tmp_path, NATIVE_SAMPLE, "StepCount", NATIVE_STATE)
    (root / (NATIVE_STATE + "-journal")).write_bytes(b"\0" * 512)
    with pytest.raises(DataLoaderStateError, match="uncommitted writes"):
        StepCountLoader(phase="native", root=root).get_data()
    data = StepCountLoader(phase="native", root=root, state_validation="off").get_data()
    assert len(data.df) == 52_247
    assert data.load_report["size_estimate"] is None


@needs_sample
def test_curated_hot_rollback_journal_is_refused_like_a_wal(tmp_path: Path):
    root = _copy_root(tmp_path, CURATED_SAMPLE, "StepCount", CURATED_STATE)
    (root / (CURATED_STATE + "-journal")).write_bytes(b"\0" * 512)
    with pytest.raises(DataLoaderStateError, match="uncommitted writes"):
        StepCountLoader(root=root).get_data()


# ============================================================ acquisition coverage in get_data
@needs_sample
def test_get_data_reports_acquisition_coverage():
    report = HeartRateLoader(root=CURATED_SAMPLE).get_data().load_report
    assert report["acquisition_method_counts"]["unclassified"] == 326_290
    assert sum(report["acquisition_method_counts"].values()) == 379_245
    assert math.isclose(report["acquisition_classified_fraction"], 52_955 / 379_245)


@needs_sample
def test_coverage_describes_returned_rows_whatever_columns_are_requested():
    full = HeartRateLoader(root=CURATED_SAMPLE).get_data().load_report
    narrow = HeartRateLoader(root=CURATED_SAMPLE).get_data(columns=["value"]).load_report
    assert narrow["acquisition_method_counts"] == full["acquisition_method_counts"]


@needs_sample
def test_coverage_follows_row_filters():
    data = HeartRateLoader(root=CURATED_SAMPLE).get_data(start_date="2024-01-01")
    assert sum(data.load_report["acquisition_method_counts"].values()) == len(data.df)
    subset = WeightLoader(root=CURATED_SAMPLE).get_data(default_inclusion_only=True)
    assert sum(subset.load_report["acquisition_method_counts"].values()) == 82


@needs_sample
def test_df_metadata_carries_per_participant_coverage():
    data = HeartRateLoader(root=CURATED_SAMPLE).get_data()
    fraction = data.df_metadata["acquisition_classified_fraction"]
    assert str(fraction.dtype) == "Float64"
    for code, value in fraction.items():
        methods = data.df.xs(code, level="RegistrationCode")["acquisition_method"].astype(str)
        assert math.isclose(value, float((methods != "unclassified").mean()))


@needs_sample
def test_native_coverage_uses_stored_values_without_manufacturing_labels():
    blood_alcohol = BloodAlcoholContentLoader(phase="native", root=NATIVE_SAMPLE).get_data().load_report
    assert set(blood_alcohol["acquisition_method_counts"]) <= {"calculator_estimate", "unclassified"}
    steps = StepCountLoader(phase="native", root=NATIVE_SAMPLE).get_data().load_report
    assert steps["acquisition_method_counts"] == {"unclassified": 52_247}
    assert steps["acquisition_classified_fraction"] == 0.0


@needs_sample
def test_empty_result_has_no_coverage():
    report = HeartRateLoader(root=CURATED_SAMPLE).get_data(start_date="2099-01-01").load_report
    assert report["acquisition_method_counts"] == {}
    assert report["acquisition_classified_fraction"] is None
