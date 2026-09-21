"""
Wearable Data Processing and Modeling project
Tests for the 0.2.0 DataLoader contract.
1. Drift guard: stored columns without a declared role are reported and never silently merged with
   deliberately withheld columns.
2. Dense curated schema, reconciled against the curation state database, with a policy-fingerprint check.
3. The HPP ``LoaderData`` triple: ``df``, ``df_metadata``, ``df_columns_metadata``, plus ``load_report``.
4. Declared dtypes: categoricals for low-cardinality text, nullable booleans for true/false columns.
End-to-end tests run against copies of the curated and native sample roots; they never modify the samples.
"""


from __future__ import annotations
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import warnings
from pathlib import Path
import pandas as pd
import pytest
from wearable_project.DataLoaders import DataLoaderStateError, LoaderData
from wearable_project.DataLoaders._base import AppleHealthFeatureLoader
from wearable_project.DataLoaders.ActivitySummaryLoader import ActivitySummaryLoader
from wearable_project.DataLoaders.BloodAlcoholContentLoader import BloodAlcoholContentLoader
from wearable_project.DataLoaders.HeartRateLoader import HeartRateLoader
from wearable_project.DataLoaders.OxygenSaturationLoader import OxygenSaturationLoader
from wearable_project.DataLoaders.SleepLoader import SleepLoader
from wearable_project.DataLoaders.StepCountLoader import StepCountLoader
from wearable_project.DataLoaders.WeightLoader import WeightLoader
from wearable_project.exceptions import DataLoaderConfigurationError, DataLoaderReadError


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
    reason="curated and native sample roots are required for end-to-end contract tests",
)

STATE_DB = ".wearable_curation_state.sqlite"
DENSE = ("acquisition_method", "curation_status", "curation_flags", "include_by_default")
PARTICIPANT = "1420269268"


# ------------------------------------------------------------------------------------------ helpers
def _curated_copy(tmp_path: Path, feature: str = "StepCount", *, with_state: bool = True) -> Path:
    """A copy of one feature across all sample participants, optionally with the state database."""
    root = tmp_path / "curated"
    for path in CURATED_SAMPLE.glob(f"*/{feature}.csv"):
        (root / path.parent.name).mkdir(parents=True)
        shutil.copy2(path, root / path.parent.name / path.name)
    if with_state:
        shutil.copy2(CURATED_SAMPLE / STATE_DB, root / STATE_DB)
    return root


def _edit_state(root: Path, sql: str) -> None:
    """Modify a copied state database, then remove the side files the write leaves behind."""
    connection = sqlite3.connect(root / STATE_DB)
    connection.execute(sql)
    connection.commit()
    connection.close()
    for suffix in ("-wal", "-shm"):
        (root / (STATE_DB + suffix)).unlink(missing_ok=True)


def _write_minimal_curated(root: Path, participant: str, rows: list[dict]) -> None:
    folder = root / participant
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "StepCount.csv", index=False)


def _minimal_row(**extra) -> dict:
    row = {
        "start_date": "2024-01-01T00:00:00Z", "end_date": "2024-01-01T01:00:00Z", "value": 10.0,
        "datetime": "2024-01-01T00:00:00Z", "data_source": "AppleHealthkit",
    }
    row.update(extra)
    return row


def _snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    state = {}
    for directory, _, files in os.walk(root):
        for name in files:
            path = os.path.join(directory, name)
            stat = os.stat(path)
            state[path] = (stat.st_size, stat.st_mtime_ns, hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return state


# ------------------------------------------------------------------------ item 1: drift guard
@needs_sample
def test_undeclared_column_is_reported_apart_from_withheld_columns(tmp_path: Path):
    root = tmp_path / "drift"
    (root / "1235738253").mkdir(parents=True)
    frame = pd.read_csv(CURATED_SAMPLE / "1235738253" / "HeartRate.csv", dtype=str, keep_default_na=False)
    frame["future_column"] = "x"
    frame.to_csv(root / "1235738253" / "HeartRate.csv", index=False)

    with pytest.warns(UserWarning, match="no declared role in curation.schema: future_column"):
        data = HeartRateLoader(root=root).get_data()
    report = data.load_report
    assert report["columns_undeclared"] == ["future_column"]
    assert "future_column" not in report["columns_withheld"]
    assert "record_id" in report["columns_withheld"]
    assert "future_column" not in data.df.columns


@needs_sample
def test_undeclared_column_is_returned_and_labelled_under_full(tmp_path: Path):
    root = tmp_path / "drift"
    (root / "1235738253").mkdir(parents=True)
    frame = pd.read_csv(CURATED_SAMPLE / "1235738253" / "HeartRate.csv", dtype=str, keep_default_na=False)
    frame["future_column"] = "x"
    frame.to_csv(root / "1235738253" / "HeartRate.csv", index=False)

    with pytest.warns(UserWarning, match="future_column"):
        data = HeartRateLoader(root=root).get_data(projection="full")
    assert "future_column" in data.df.columns
    assert data.df_columns_metadata.loc["future_column", "role"] == "undeclared"
    assert data.load_report["columns_withheld"] == []


@needs_sample
def test_sample_roots_have_no_undeclared_columns():
    for phase, root in (("curated", CURATED_SAMPLE), ("native", NATIVE_SAMPLE)):
        for feature in ("HeartRate", "Sleep", "BloodGlucose", "Weight", "Electrocardiogram"):
            loader = __import__(
                f"wearable_project.DataLoaders.{feature}Loader", fromlist=[f"{feature}Loader"]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                data = getattr(loader, f"{feature}Loader")(phase=phase, root=root).get_data()
            assert data.load_report["columns_undeclared"] == []


# ------------------------------------------------------------- item 2: dense curated schema
@needs_sample
def test_every_curated_frame_carries_the_dense_columns_without_missing_values():
    for loader in (StepCountLoader, HeartRateLoader, SleepLoader, WeightLoader, ActivitySummaryLoader):
        data = loader(root=CURATED_SAMPLE).get_data(columns=list(DENSE))
        assert list(data.df.columns) == list(DENSE)
        assert not data.df.isna().any().any(), loader.feature_name


@needs_sample
def test_dense_view_reproduces_the_state_database_totals():
    """The loader's dense logical view must agree with the counts the curation writer recorded."""
    connection = sqlite3.connect(f"file:{CURATED_SAMPLE / STATE_DB}?immutable=1", uri=True)
    recorded = connection.execute(
        "select sum(curated_rows), sum(pass_rows), sum(review_rows), "
        "sum(included_by_default_rows), sum(excluded_by_default_rows) from curation_outputs "
        "where feature in ('StepCount', 'HeartRate', 'Weight')"
    ).fetchone()
    connection.close()

    frames = [
        loader(root=CURATED_SAMPLE).get_data(columns=["curation_status", "include_by_default"]).df
        for loader in (StepCountLoader, HeartRateLoader, WeightLoader)
    ]
    combined = pd.concat([frame.astype({"curation_status": str}) for frame in frames])
    assert len(combined) == recorded[0]
    assert (combined["curation_status"] == "pass").sum() == recorded[1]
    assert (combined["curation_status"] == "review").sum() == recorded[2]
    assert combined["include_by_default"].sum() == recorded[3]
    assert (~combined["include_by_default"]).sum() == recorded[4]


@needs_sample
def test_heart_rate_acquisition_matches_the_corrected_unclassified_count():
    counts = HeartRateLoader(root=CURATED_SAMPLE).get_data(
        columns=["acquisition_method"]
    ).df["acquisition_method"].value_counts()
    assert int(counts["unclassified"]) == 326_290
    assert int(counts.sum()) == 379_245


@needs_sample
def test_rows_reconstructed_are_reported():
    report = StepCountLoader(root=CURATED_SAMPLE).get_data().load_report
    assert report["dense_reconstruction"] is True
    # StepCount stores none of the four curation columns, so every row of every file is reconstructed.
    assert report["rows_reconstructed"] == {column: 52_247 for column in DENSE}


def test_blank_acquisition_and_flags_are_filled_but_blank_status_is_not(tmp_path: Path):
    """Blanks mean the default only for acquisition_method and curation_flags."""
    root = tmp_path / "curated"
    _write_minimal_curated(root, "123", [
        _minimal_row(acquisition_method="device_estimate", curation_flags="a;b", curation_status="review"),
        _minimal_row(start_date="2024-01-01T01:00:00Z", acquisition_method=None, curation_flags=None,
                     curation_status=None),
    ])
    df = StepCountLoader(root=root).get_data(columns=list(DENSE)).df
    assert list(df["acquisition_method"].astype(str)) == ["device_estimate", "unclassified"]
    assert list(df["curation_flags"].astype(str)) == ["a;b", ""]
    # A blank status inside a written column is a writer anomaly; it is left for reconciliation to report.
    assert df["curation_status"].isna().tolist() == [False, True]


def test_include_by_default_is_a_plain_boolean(tmp_path: Path):
    root = tmp_path / "curated"
    _write_minimal_curated(root, "123", [
        _minimal_row(include_by_default=1),
        _minimal_row(start_date="2024-01-01T01:00:00Z", include_by_default=0),
    ])
    column = StepCountLoader(root=root).get_data().df["include_by_default"]
    assert column.dtype == bool
    assert column.tolist() == [True, False]


@needs_sample
def test_native_phase_is_never_densified():
    data = StepCountLoader(phase="native", root=NATIVE_SAMPLE).get_data(projection="full")
    for column in DENSE:
        assert column not in data.df.columns
    assert data.load_report["dense_reconstruction"] is False
    assert data.load_report["rows_reconstructed"] == {}
    assert data.load_report["state_verified"] is None


@needs_sample
def test_native_blood_alcohol_acquisition_is_left_as_stored():
    """The one native file carrying acquisition_method keeps its raw values; nothing is manufactured."""
    df = BloodAlcoholContentLoader(phase="native", root=NATIVE_SAMPLE).get_data().df
    assert set(df["acquisition_method"].dropna().astype(str)) == {"calculator_estimate"}


@needs_sample
def test_requesting_a_dense_column_works_in_curated_and_fails_in_native():
    data = StepCountLoader(root=CURATED_SAMPLE).get_data(columns=["include_by_default"])
    assert data.df["include_by_default"].all()
    with pytest.raises(DataLoaderReadError, match="include_by_default"):
        StepCountLoader(phase="native", root=NATIVE_SAMPLE).get_data(columns=["include_by_default"])


# ------------------------------------------------------------------ item 2: reconciliation
@needs_sample
def test_sample_root_verifies_cleanly():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        report = StepCountLoader(root=CURATED_SAMPLE).get_data().load_report
    assert report["state_verified"] is True
    assert report["state_verification_note"] is None
    assert report["files_verified"] == report["files_read"] == 10
    assert report["participants_with_diverged_policy"] == []
    assert report["manifest_files_missing"] == []


@needs_sample
def test_verification_runs_on_whole_files_so_filtered_calls_still_verify():
    for kwargs in ({"start_date": "2023-01-01"}, {"default_inclusion_only": True},
                   {"registration_codes": "10K_1420269268"}):
        report = WeightLoader(root=CURATED_SAMPLE).get_data(**kwargs).load_report
        assert report["state_verified"] is True, kwargs


@needs_sample
def test_truncated_file_is_rejected(tmp_path: Path):
    root = _curated_copy(tmp_path)
    path = root / PARTICIPANT / "StepCount.csv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-50]) + "\n")
    with pytest.raises(DataLoaderStateError, match="rows: file"):
        StepCountLoader(root=root).get_data()


@needs_sample
def test_same_length_value_edit_is_caught_by_the_hash(tmp_path: Path):
    """Size and every count are unchanged here; only the content hash can detect the edit."""
    root = _curated_copy(tmp_path)
    path = root / PARTICIPANT / "StepCount.csv"
    original = path.read_text()
    edited = original.replace(",39.0,", ",40.0,", 1)
    assert edited != original and len(edited) == len(original)
    path.write_text(edited)
    with pytest.raises(DataLoaderStateError, match="sha256"):
        StepCountLoader(root=root).get_data()


@needs_sample
def test_stale_state_database_is_rejected(tmp_path: Path):
    root = _curated_copy(tmp_path)
    _edit_state(root, f"update curation_outputs set pass_rows = pass_rows + 1 "
                      f"where participant_id = '{PARTICIPANT}' and feature = 'StepCount'")
    with pytest.raises(DataLoaderStateError, match="pass rows"):
        StepCountLoader(root=root).get_data()


@needs_sample
def test_file_without_a_manifest_row_is_rejected(tmp_path: Path):
    root = _curated_copy(tmp_path)
    _edit_state(root, f"delete from curation_outputs where participant_id = '{PARTICIPANT}' "
                      f"and feature = 'StepCount'")
    with pytest.raises(DataLoaderStateError, match="has no curation_outputs row"):
        StepCountLoader(root=root).get_data()


@needs_sample
def test_state_error_is_a_read_error(tmp_path: Path):
    """Existing handlers for DataLoaderReadError keep catching the new, more specific error."""
    root = _curated_copy(tmp_path)
    _edit_state(root, "update curation_outputs set curated_rows = curated_rows + 1 where feature = 'StepCount'")
    with pytest.raises(DataLoaderReadError):
        StepCountLoader(root=root).get_data()


@needs_sample
def test_manifest_file_missing_from_disk_warns_but_loads(tmp_path: Path):
    root = _curated_copy(tmp_path)
    (root / PARTICIPANT / "StepCount.csv").unlink()
    with pytest.warns(UserWarning, match="no longer on disk"):
        data = StepCountLoader(root=root).get_data()
    assert data.load_report["manifest_files_missing"] == [PARTICIPANT]
    assert data.load_report["state_verified"] is True


@needs_sample
def test_manifest_missing_check_respects_requested_participants(tmp_path: Path):
    root = _curated_copy(tmp_path)
    (root / PARTICIPANT / "StepCount.csv").unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        data = StepCountLoader(root=root).get_data(registration_codes="10K_1235738253")
    assert data.load_report["manifest_files_missing"] == []


@needs_sample
def test_diverged_policy_fingerprint_warns_and_is_reflected_per_participant(tmp_path: Path):
    root = _curated_copy(tmp_path)
    _edit_state(root, f"update curation_outputs set policy_fingerprint = 'older' "
                      f"where participant_id = '{PARTICIPANT}' and feature = 'StepCount'")
    with pytest.warns(UserWarning, match="differs from the installed curation registry"):
        data = StepCountLoader(root=root).get_data()
    assert data.load_report["participants_with_diverged_policy"] == [PARTICIPANT]
    assert data.df_metadata.loc[f"10K_{PARTICIPANT}", "policy_current"] is False or \
        not bool(data.df_metadata.loc[f"10K_{PARTICIPANT}", "policy_current"])
    others = data.df_metadata.drop(index=f"10K_{PARTICIPANT}")["policy_current"]
    assert others.all()


@needs_sample
def test_migrated_database_skips_inclusion_counts_and_flags_unverified_inclusion(tmp_path: Path):
    """A database migrated from an engine without inclusion counts stores 0/0; that is not a mismatch."""
    root = _curated_copy(tmp_path)
    _edit_state(root, "update curation_outputs set included_by_default_rows = 0, "
                      "excluded_by_default_rows = 0 where feature = 'StepCount'")
    report = StepCountLoader(root=root).get_data().load_report
    assert report["state_verified"] is True
    assert report["inclusion_counts_unverifiable"] == 10

    with pytest.warns(UserWarning, match="without count verification"):
        filtered = StepCountLoader(root=root).get_data(default_inclusion_only=True)
    assert filtered.load_report["default_inclusion_rows_unverified"] == 52_247


@needs_sample
def test_non_empty_write_ahead_log_is_refused(tmp_path: Path):
    root = _curated_copy(tmp_path)
    (root / (STATE_DB + "-wal")).write_bytes(b"\0" * 64)
    with pytest.raises(DataLoaderStateError, match="uncommitted writes"):
        StepCountLoader(root=root).get_data()


@needs_sample
def test_empty_write_ahead_log_is_harmless(tmp_path: Path):
    root = _curated_copy(tmp_path)
    (root / (STATE_DB + "-wal")).write_bytes(b"")
    assert StepCountLoader(root=root).get_data().load_report["state_verified"] is True


@needs_sample
def test_absent_state_database_under_auto_loads_unverified(tmp_path: Path):
    root = _curated_copy(tmp_path, with_state=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        data = StepCountLoader(root=root).get_data()
    assert data.load_report["state_verified"] is False
    assert data.load_report["state_verification_note"] == "curation state database absent"
    assert not data.df_metadata["state_verified"].any()
    assert data.df_metadata["policy_current"].isna().all()


@needs_sample
def test_absent_state_database_under_required_is_rejected(tmp_path: Path):
    root = _curated_copy(tmp_path, with_state=False)
    with pytest.raises(DataLoaderStateError, match="required"):
        StepCountLoader(root=root, state_validation="required").get_data()


@needs_sample
def test_off_skips_verification_even_for_a_tampered_file(tmp_path: Path):
    root = _curated_copy(tmp_path)
    path = root / PARTICIPANT / "StepCount.csv"
    path.write_text(path.read_text().replace(",39.0,", ",40.0,", 1))
    data = StepCountLoader(root=root, state_validation="off").get_data()
    assert data.load_report["state_verified"] is False
    assert "state_validation='off'" in data.load_report["state_verification_note"]
    # Dense reconstruction still happens with verification off.
    assert set(DENSE) <= set(data.df.columns)


def test_invalid_state_validation_is_rejected():
    with pytest.raises(DataLoaderConfigurationError, match="state_validation must be one of"):
        StepCountLoader(state_validation="sometimes")


def test_required_validation_is_rejected_for_native():
    with pytest.raises(DataLoaderConfigurationError, match="applies only to phase='curated'"):
        StepCountLoader(phase="native", state_validation="required")


def test_constructor_performs_no_filesystem_access(tmp_path: Path):
    """The phase roots may be unreachable at construction time; only get_data touches the filesystem."""
    StepCountLoader(root=tmp_path / "does" / "not" / "exist", state_validation="required")


@needs_sample
def test_loading_never_writes_to_the_data_tree(tmp_path: Path):
    """Reading the WAL-mode state database must not leave -shm or -wal side files behind."""
    root = _curated_copy(tmp_path, feature="HeartRate")
    before = _snapshot(root)
    HeartRateLoader(root=root).get_data(projection="full")
    HeartRateLoader(root=root).get_data(default_inclusion_only=True)
    assert _snapshot(root) == before


@pytest.mark.skipif(os.geteuid() != 0, reason="needs root to drop privileges to an unprivileged user")
@needs_sample
def test_state_database_is_readable_from_a_read_only_directory():
    """
    ``mode=ro`` fails on a WAL database in a read-only directory because SQLite tries to create the
    ``-shm`` side file. The loader's ``immutable=1`` read must succeed there, as an unprivileged user.
    pytest's own tmp_path sits under a root-only directory, so this test builds a world-traversable one.
    """
    import tempfile
    base = Path(tempfile.mkdtemp(prefix="wearable_ro_", dir="/tmp"))
    try:
        os.chmod(base, 0o755)
        package = Path(__file__).resolve().parents[1] / "wearable_project"
        shutil.copytree(package, base / "pkg" / "wearable_project",
                        ignore=shutil.ignore_patterns("__pycache__"))
        root = _curated_copy(base, feature="StepCount")
        for directory, _, files in os.walk(base):
            os.chmod(directory, 0o555)
            for name in files:
                os.chmod(os.path.join(directory, name), 0o444)
        script = (
            f"import sys; sys.dont_write_bytecode = True; sys.path.insert(0, {str(base / 'pkg')!r})\n"
            "from wearable_project.DataLoaders.StepCountLoader import StepCountLoader\n"
            f"print(StepCountLoader(root={str(root)!r}).get_data().load_report['state_verified'])\n"
        )
        result = subprocess.run(
            ["su", "nobody", "-s", "/bin/sh", "-c", f"{sys.executable} -B -c \"{script}\""],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True"
        assert not list(root.glob(f"{STATE_DB}-*")), "the read left side files in the data tree"
    finally:
        for directory, _, files in os.walk(base):
            os.chmod(directory, 0o755)
            for name in files:
                os.chmod(os.path.join(directory, name), 0o644)
        shutil.rmtree(base, ignore_errors=True)


# ----------------------------------------------------------------- item 3: LoaderData triple
@needs_sample
def test_loader_data_exposes_the_hpp_triple_and_a_load_report():
    data = WeightLoader(root=CURATED_SAMPLE).get_data()
    assert isinstance(data, LoaderData)
    assert isinstance(data.df, pd.DataFrame)
    assert isinstance(data.df_metadata, pd.DataFrame)
    assert isinstance(data.df_columns_metadata, pd.DataFrame)
    assert isinstance(data.load_report, dict)


@needs_sample
def test_df_metadata_has_one_row_per_participant_in_df():
    data = WeightLoader(root=CURATED_SAMPLE).get_data(start_date="2020-01-01")
    md = data.df_metadata
    assert md.index.name == "RegistrationCode"
    assert list(md.index) == sorted(set(data.df.index.get_level_values("RegistrationCode")))
    assert int(md["rows"].sum()) == len(data.df)
    for code, row in md.iterrows():
        dates = data.df.xs(code, level="RegistrationCode").index
        assert row["first_date"] == dates.min() and row["last_date"] == dates.max()
        stored = len(pd.read_csv(CURATED_SAMPLE / row["participant_id"] / "Weight.csv"))
        assert row["stored_rows"] == stored
        assert row["rows"] <= row["stored_rows"]


@needs_sample
def test_df_metadata_dtypes_are_stable():
    md = WeightLoader(root=CURATED_SAMPLE).get_data().df_metadata
    assert str(md["rows"].dtype) == "int64"
    assert str(md["stored_rows"].dtype) == "int64"
    assert str(md["state_verified"].dtype) == "boolean"
    assert str(md["policy_current"].dtype) == "boolean"


@needs_sample
def test_df_metadata_in_the_native_phase_marks_verification_not_applicable():
    md = WeightLoader(phase="native", root=NATIVE_SAMPLE).get_data().df_metadata
    assert md["state_verified"].isna().all()
    assert md["policy_current"].isna().all()


@needs_sample
def test_df_columns_metadata_describes_every_returned_column():
    data = HeartRateLoader(root=CURATED_SAMPLE).get_data()
    cm = data.df_columns_metadata
    assert list(cm.index) == list(data.df.columns)
    assert set(cm.columns) == {"role", "description", "dtype", "non_null", "registry_unit", "dense_default"}
    for column in data.df.columns:
        assert cm.loc[column, "dtype"] == str(data.df[column].dtype)
        assert cm.loc[column, "non_null"] == int(data.df[column].notna().sum())
    assert cm.loc["value", "registry_unit"] == "beats/min"
    assert cm.loc["acquisition_method", "dense_default"] == "unclassified"
    assert cm.loc["include_by_default", "dense_default"] is True


@needs_sample
def test_df_columns_metadata_reports_canonical_units_from_the_registry():
    cm = OxygenSaturationLoader(root=CURATED_SAMPLE).get_data().df_columns_metadata
    assert cm.loc["value", "registry_unit"] == "fraction"
    assert cm.loc["canonical_value", "registry_unit"] == "%"


@needs_sample
def test_empty_result_carries_empty_metadata_tables():
    data = WeightLoader(root=CURATED_SAMPLE).get_data(start_date="2099-01-01")
    assert data.df.empty
    assert data.df_metadata.empty and data.df_metadata.index.name == "RegistrationCode"
    assert list(data.df_metadata.columns) == [
        "participant_id", "rows", "first_date", "last_date", "stored_rows", "state_verified", "policy_current",
        "acquisition_classified_fraction",
    ]
    assert list(data.df_columns_metadata.index) == list(data.df.columns)


@needs_sample
def test_metadata_is_a_deprecated_alias_of_load_report():
    data = StepCountLoader(root=CURATED_SAMPLE).get_data()
    with pytest.warns(DeprecationWarning, match="use LoaderData.load_report"):
        alias = data.metadata
    assert alias is data.load_report


# ----------------------------------------------------------------------- item 4: dtypes
@needs_sample
def test_declared_categorical_columns_are_categorical():
    df = HeartRateLoader(root=CURATED_SAMPLE).get_data(projection="full").df
    for column in ("acquisition_method", "curation_status", "curation_flags", "raw_unit",
                   "source_id", "source_name", "data_source"):
        assert isinstance(df[column].dtype, pd.CategoricalDtype), column
    for column in ("record_id", "metadata"):
        assert not isinstance(df[column].dtype, pd.CategoricalDtype), column
    assert df["value"].dtype == "float64"


@needs_sample
def test_sleep_value_is_categorical_but_numeric_values_are_not():
    assert isinstance(SleepLoader(root=CURATED_SAMPLE).get_data().df["value"].dtype, pd.CategoricalDtype)
    assert StepCountLoader(root=CURATED_SAMPLE).get_data().df["value"].dtype == "float64"


@needs_sample
def test_was_user_entered_is_a_nullable_boolean_that_keeps_missing_values():
    df = WeightLoader(root=CURATED_SAMPLE).get_data().df
    column = df["was_user_entered"]
    assert str(column.dtype) == "boolean"
    assert column.isna().any() and column.notna().any()


@needs_sample
def test_dtypes_are_identical_across_different_participant_subsets():
    """Declared, not inferred: the same column has the same dtype whichever participants are loaded."""
    a = WeightLoader(root=CURATED_SAMPLE).get_data(registration_codes="10K_2719597610").df
    b = WeightLoader(root=CURATED_SAMPLE).get_data(registration_codes="10K_3120605962").df
    for column in set(a.columns) & set(b.columns):
        assert type(a[column].dtype) is type(b[column].dtype), column


@needs_sample
def test_typing_does_not_change_any_value():
    typed = HeartRateLoader(root=CURATED_SAMPLE).get_data(
        registration_codes="10K_1235738253", projection="full"
    ).df
    raw = pd.read_csv(CURATED_SAMPLE / "1235738253" / "HeartRate.csv", low_memory=False)
    for column in ("acquisition_method", "raw_unit", "source_name"):
        if column in raw.columns:
            left = typed[column].astype(object).where(typed[column].notna(), None).tolist()
            right = raw[column].astype(object).where(raw[column].notna(), None)
            right = right.fillna("unclassified").tolist() if column == "acquisition_method" else right.tolist()
            assert sorted(map(str, left)) == sorted(map(str, right)), column


@needs_sample
def test_declared_dtypes_reduce_memory_substantially():
    df = HeartRateLoader(root=CURATED_SAMPLE).get_data().df
    untyped = df.copy()
    for column in untyped.columns:
        if isinstance(untyped[column].dtype, pd.CategoricalDtype) or str(untyped[column].dtype) == "boolean":
            untyped[column] = untyped[column].astype(object)
    ratio = untyped.memory_usage(deep=True).sum() / df.memory_usage(deep=True).sum()
    assert ratio > 3.0, f"expected a substantial reduction, got {ratio:.2f}x"


def test_unrecognised_boolean_token_leaves_the_column_untouched():
    converted = AppleHealthFeatureLoader._as_nullable_boolean(pd.Series([True, "maybe", None], dtype=object))
    assert converted is None


def test_boolean_normalisation_accepts_the_on_disk_forms():
    converted = AppleHealthFeatureLoader._as_nullable_boolean(
        pd.Series([True, False, None, "1", "false", 0, 1.0], dtype=object)
    )
    assert str(converted.dtype) == "boolean"
    assert converted.tolist() == [True, False, pd.NA, True, False, False, True]


# --------------------------------------------------------------------- import hygiene holds
def test_loader_import_still_avoids_the_curation_package():
    script = (
        "import sys\n"
        "import wearable_project.DataLoaders.StepCountLoader\n"
        "from wearable_project.DataLoaders import DataLoaderStateError, LoaderData\n"
        "loaded = [m for m in sys.modules if m.startswith('wearable_project.curation')]\n"
        "assert not loaded, loaded\n"
        "print('clean')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
