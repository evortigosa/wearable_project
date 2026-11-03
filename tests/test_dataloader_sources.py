"""
Wearable Data Processing and Modeling project
Source-selection and managed-root identity guards for the source-aware DataLoader foundation.
"""


from __future__ import annotations
import inspect
import sqlite3
import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest
from wearable_project import DataLoaders
from wearable_project.DataLoaders._base import AppleHealthFeatureLoader, WearableFeatureLoader
from wearable_project.DataLoaders._sources import (
    APPLE_HEALTHKIT_PUBLIC_FEATURES, DEFAULT_APPLE_HEALTHKIT_CURATED_ROOT, DEFAULT_APPLE_HEALTHKIT_NATIVE_ROOT,
    DEFAULT_GOOGLE_FITNESS_CURATED_ROOT, DEFAULT_GOOGLE_FITNESS_NATIVE_ROOT, WearableSource,
    features_for_source, normalize_source, probe_root_identity,
)
from wearable_project.DataLoaders.HeartRateLoader import HeartRateLoader
from wearable_project.DataLoaders.StepCountLoader import StepCountLoader
from wearable_project.curation.apple_healthkit.registry import COHORT_OBSERVED_FEATURES
from wearable_project.exceptions import DataLoaderConfigurationError, DataLoaderSourceError
from wearable_project.processing.google_fitness.registry import (
    REGISTRY_VERSION as GOOGLE_REGISTRY_VERSION, feature_names as google_feature_names,
)
from wearable_project.processing.google_fitness.writer import STATE_SCHEMA_VERSION as GOOGLE_STATE_SCHEMA_VERSION


def _state(root: Path, phase: str, source: WearableSource) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    name = ".wearable_state.sqlite" if phase == "native" else ".wearable_curation_state.sqlite"
    database = root / name
    connection = sqlite3.connect(database)
    if source is WearableSource.GOOGLE_FITNESS:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        metadata = {"source_key": "google_fitness"}
        if phase == "native":
            metadata.update({
                "state_schema_version": GOOGLE_STATE_SCHEMA_VERSION,
                "registry_version": GOOGLE_REGISTRY_VERSION,
            })
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", sorted(metadata.items()))
        if phase == "native":
            connection.execute(
                "CREATE TABLE feature_outputs ("
                "participant_id TEXT, feature TEXT, filename TEXT, row_count INTEGER, "
                "size_bytes INTEGER, sha256 TEXT, represented_occurrences INTEGER)"
            )
            connection.execute("CREATE TABLE participants (participant_id TEXT, status TEXT)")
        else:
            connection.execute("CREATE TABLE curation_outputs (participant_id TEXT, feature TEXT)")
    elif phase == "native":
        connection.execute("CREATE TABLE feature_outputs (participant_id TEXT, feature TEXT)")
        connection.execute("CREATE TABLE run_tracking (run_id TEXT)")
        connection.execute("CREATE TABLE run_diagnostics (diagnostic_id TEXT)")
    else:
        connection.execute("CREATE TABLE curation_outputs (participant_id TEXT, feature TEXT)")
        connection.execute("CREATE TABLE curation_unit_epochs (participant_id TEXT)")
    connection.commit()
    connection.close()
    return database


def _minimal_native(root: Path) -> None:
    participant = root / "123"
    participant.mkdir(parents=True)
    pd.DataFrame([
        {
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-01T01:00:00Z",
            "value": 10.0,
            "data_source": "AppleHealthkit",
        }
    ]).to_csv(participant / "StepCount.csv", index=False)


def test_source_aliases_normalize_to_canonical_values():
    assert normalize_source("apple-healthkit") is WearableSource.APPLE_HEALTHKIT
    assert normalize_source("AppleHealthkit") is WearableSource.APPLE_HEALTHKIT
    assert normalize_source("google fitness") is WearableSource.GOOGLE_FITNESS
    assert normalize_source("google-fit") is WearableSource.GOOGLE_FITNESS
    assert normalize_source(WearableSource.GOOGLE_FITNESS) is WearableSource.GOOGLE_FITNESS
    with pytest.raises(DataLoaderConfigurationError, match="unknown wearable source"):
        normalize_source("other")


def test_existing_loader_defaults_remain_apple_healthkit():
    loader = StepCountLoader()
    assert loader.source is WearableSource.APPLE_HEALTHKIT
    assert loader.native_root == DEFAULT_APPLE_HEALTHKIT_NATIVE_ROOT
    assert loader.curated_root == DEFAULT_APPLE_HEALTHKIT_CURATED_ROOT
    assert loader.date_column == "start_date"
    assert loader.date_semantics == "event_start"
    assert "date_column" not in loader.__dict__
    assert "date_semantics" not in loader.__dict__
    assert isinstance(loader, AppleHealthFeatureLoader)


def test_legacy_base_ownership_and_public_mro_are_preserved():
    assert StepCountLoader.__mro__ == (StepCountLoader, AppleHealthFeatureLoader, object)
    assert "__init__" in AppleHealthFeatureLoader.__dict__
    parameters = tuple(inspect.signature(StepCountLoader.__init__).parameters)
    assert parameters == (
        "self", "phase", "native_root", "curated_root", "root", "state_validation", "source",
    )


def test_default_apple_construction_does_not_import_google_processing_modules():
    script = """
import sys
from wearable_project.DataLoaders.StepCountLoader import StepCountLoader
StepCountLoader()
loaded = sorted(name for name in sys.modules if name.startswith('wearable_project.processing.google_fitness'))
assert not loaded, loaded
print('clean')
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "clean"


def test_explicit_apple_is_constructor_equivalent_to_omitted_source():
    implicit = StepCountLoader()
    explicit = StepCountLoader(source="apple_healthkit")
    assert implicit.source == explicit.source
    assert implicit.phase == explicit.phase
    assert implicit.native_root == explicit.native_root
    assert implicit.curated_root == explicit.curated_root
    assert implicit.date_column == explicit.date_column
    assert implicit.date_semantics == explicit.date_semantics


def test_shared_loader_selects_google_roots_and_temporal_contract_without_io(tmp_path: Path):
    missing = tmp_path / "does-not-exist"
    loader = StepCountLoader(source="google_fitness", root=missing)
    assert loader.source is WearableSource.GOOGLE_FITNESS
    assert loader.native_root == DEFAULT_GOOGLE_FITNESS_NATIVE_ROOT
    assert loader.curated_root == DEFAULT_GOOGLE_FITNESS_CURATED_ROOT
    assert loader.data_root == missing
    assert loader.date_column == "outer_datetime"
    assert loader.date_semantics == "source_outer_summary_datetime_not_canonical_event_day"


def test_unsupported_feature_source_pair_fails_during_construction(monkeypatch):
    touched = {"filesystem": False}

    def forbidden(*args, **kwargs):
        touched["filesystem"] = True
        raise AssertionError("constructor touched the filesystem")

    monkeypatch.setattr(Path, "is_dir", forbidden)
    with pytest.raises(DataLoaderSourceError, match="HeartRate.*not available.*google_fitness"):
        HeartRateLoader(source="google_fitness")
    assert touched["filesystem"] is False


def test_existing_apple_native_extension_points_remain_available():
    class BasalBodyTemperatureLoader(AppleHealthFeatureLoader):
        feature_name = "BasalBodyTemperature"

    native = BasalBodyTemperatureLoader(phase="native")
    assert native.source is WearableSource.APPLE_HEALTHKIT
    with pytest.raises(DataLoaderSourceError, match="not available.*apple_healthkit"):
        BasalBodyTemperatureLoader(phase="curated")


def test_neutral_base_supports_future_google_only_top_level_loaders():
    assert issubclass(WearableFeatureLoader, AppleHealthFeatureLoader)
    class WalkingLoader(WearableFeatureLoader):
        feature_name = "Walking"
        default_source = WearableSource.GOOGLE_FITNESS

    assert WalkingLoader().source is WearableSource.GOOGLE_FITNESS
    with pytest.raises(DataLoaderSourceError, match="Walking.*not available.*apple_healthkit"):
        WalkingLoader(source="apple_healthkit")


def test_feature_catalogs_match_the_accepted_source_registries():
    assert APPLE_HEALTHKIT_PUBLIC_FEATURES == tuple(COHORT_OBSERVED_FEATURES)
    assert features_for_source("google_fitness") == tuple(google_feature_names())
    assert DataLoaders.available_features() == tuple(COHORT_OBSERVED_FEATURES)
    assert DataLoaders.available_features(source="google-fitness") == tuple(google_feature_names())
    assert DataLoaders.available_sources("StepCount") == ("apple_healthkit", "google_fitness")
    assert DataLoaders.available_sources("HeartRate") == ("apple_healthkit",)
    assert DataLoaders.available_sources("Walking") == ("google_fitness",)


@pytest.mark.parametrize("phase", ["native", "curated"])
@pytest.mark.parametrize("source", list(WearableSource))
def test_managed_root_probe_identifies_source_and_phase(tmp_path: Path, phase: str, source: WearableSource):
    root = tmp_path / f"{source.value}-{phase}"
    database = _state(root, phase, source)
    identity = probe_root_identity(root, phase)
    assert identity.source is source
    assert identity.phase == phase
    assert identity.managed is True
    assert identity.verified is True
    assert identity.state_database == database
    assert StepCountLoader.detect_source(root) is source


def test_unmanaged_root_makes_no_source_claim(tmp_path: Path):
    identity = probe_root_identity(tmp_path, None)
    assert identity.source is None
    assert identity.managed is False
    assert identity.verified is False
    assert StepCountLoader.detect_source(tmp_path) is None


def test_wrong_managed_source_is_rejected_before_csv_read_even_when_state_validation_is_off(
    tmp_path: Path, monkeypatch,
):
    root = tmp_path / "google"
    _state(root, "native", WearableSource.GOOGLE_FITNESS)

    def forbidden(*args, **kwargs):
        raise AssertionError("feature CSV was parsed before source validation")

    monkeypatch.setattr(AppleHealthFeatureLoader, "_read_feature_file", forbidden)
    with pytest.raises(DataLoaderSourceError, match="managed 'google_fitness' native root"):
        StepCountLoader(
            source="apple_healthkit", phase="native", root=root, state_validation="off",
        ).get_data()


def test_correct_google_native_root_reaches_the_google_engine(tmp_path: Path):
    root = tmp_path / "google"
    _state(root, "native", WearableSource.GOOGLE_FITNESS)

    data = StepCountLoader(
        source="google_fitness", phase="native", root=root, state_validation="required",
    ).get_data()
    assert data.df.empty
    assert data.load_report["wearable_source"] == "google_fitness"
    assert data.load_report["source_verified"] is True
    assert data.load_report["state_verified"] is True
    assert data.load_report["files_read"] == 0


def test_apple_loader_reports_unverified_source_for_an_unmanaged_trimmed_tree(tmp_path: Path):
    _minimal_native(tmp_path)
    result = StepCountLoader(phase="native", root=tmp_path, state_validation="off").get_data()
    assert len(result.df) == 1
    assert result.load_report["wearable_source"] == "apple_healthkit"
    assert result.load_report["source_verified"] is False
    assert result.load_report["source_identity_evidence"] == "no unique native/curated state marker"

    profile = StepCountLoader(phase="native", root=tmp_path, state_validation="off").profile()
    assert profile.summary["wearable_source"] == "apple_healthkit"
    assert profile.summary["source_verified"] is False


def test_source_qualified_information_dispatches_to_google_contracts():
    report = DataLoaders.info("StepCount", source="google_fitness")
    assert report.payload["wearable_source"] == "google_fitness"
    assert report.payload["feature"] == "StepCount"
    assert report.payload["date_index"]["column"] == "outer_datetime"
    assert report.payload["harmonization"]["established_unit"] == "count"
