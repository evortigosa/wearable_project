"""
Wearable Data Processing and Modeling project
Architectural acceptance for the source-aware DataLoader framework.
These tests protect the production Apple compatibility surface while proving that source-specific scientific
contracts are isolated behind adapters and that managed-root identity is never weakened by optional state-file
verification.
"""


from __future__ import annotations
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest
from wearable_project import DataLoaders
from wearable_project.DataLoaders._base import AppleHealthFeatureLoader, DataLoaderStateError
from wearable_project.DataLoaders._contracts import SourceAdapter
from wearable_project.DataLoaders._sources import GOOGLE_FITNESS_PUBLIC_FEATURES, WearableSource, get_source_adapter
from wearable_project.DataLoaders.StepCountLoader import StepCountLoader
from wearable_project.DataLoaders.info import info as static_info
from wearable_project.curation.google_fitness.registry import CURATION_POLICIES
from wearable_project.exceptions import DataLoaderConfigurationError
from wearable_project.processing.google_fitness.registry import (
    REGISTRY_VERSION as GOOGLE_REGISTRY_VERSION, feature_names as google_feature_names,
)
from wearable_project.processing.google_fitness.writer import STATE_SCHEMA_VERSION as GOOGLE_STATE_SCHEMA_VERSION


REPO_ROOT = Path(__file__).resolve().parents[1]
APPLE_INFO_HASHES = Path(__file__).parent / "fixtures" / "apple_dataloader_info_hashes.json"


def _apple_state(root: Path, phase: str = "native") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    name = ".wearable_state.sqlite" if phase == "native" else ".wearable_curation_state.sqlite"
    database = root / name
    with sqlite3.connect(database) as connection:
        if phase == "native":
            connection.execute(
                "CREATE TABLE feature_outputs (participant_id TEXT, feature TEXT, row_count INTEGER, "
                "size_bytes INTEGER, sha256 TEXT)"
            )
            connection.execute("CREATE TABLE run_tracking (run_id TEXT)")
            connection.execute("CREATE TABLE run_diagnostics (diagnostic_id TEXT)")
        else:
            connection.execute("CREATE TABLE curation_outputs (participant_id TEXT, feature TEXT)")
            connection.execute("CREATE TABLE curation_unit_epochs (participant_id TEXT)")
    return database


def _google_native_state(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    database = root / ".wearable_state.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            sorted({
                "source_key": "google_fitness",
                "state_schema_version": GOOGLE_STATE_SCHEMA_VERSION,
                "registry_version": GOOGLE_REGISTRY_VERSION,
            }.items()),
        )
        connection.execute(
            "CREATE TABLE feature_outputs (participant_id TEXT, feature TEXT, filename TEXT, "
            "row_count INTEGER, size_bytes INTEGER, sha256 TEXT, represented_occurrences INTEGER)"
        )
        connection.execute("CREATE TABLE participants (participant_id TEXT, status TEXT)")
    return database


def _minimal_apple_curated(root: Path) -> None:
    participant = root / "123"
    participant.mkdir(parents=True)
    pd.DataFrame([{
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-01-01T01:00:00Z",
        "value": 10.0,
        "data_source": "AppleHealthkit",
    }]).to_csv(participant / "StepCount.csv", index=False)


def _run_isolation_script(script: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _canonical_payload_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _with_increment3_release_identity(value: object) -> object:
    """Normalize the intentional 0.4.0 release identity before comparing Apple behavior with Increment 3."""

    if isinstance(value, dict):
        return {key: _with_increment3_release_identity(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_with_increment3_release_identity(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_with_increment3_release_identity(item) for item in value)
    return "0.3.0" if value == "0.4.0" else value


def test_source_adapters_satisfy_the_runtime_contract():
    assert isinstance(get_source_adapter("apple_healthkit"), SourceAdapter)
    assert isinstance(get_source_adapter("google_fitness"), SourceAdapter)


def test_source_adapters_own_state_and_csv_behavior_contracts():
    apple = get_source_adapter("apple_healthkit")
    google = get_source_adapter("google_fitness")

    assert apple.state_manifest_contract("native").query == (
        "SELECT participant_id, row_count, size_bytes, sha256 "
        "FROM feature_outputs WHERE feature = ?"
    )
    assert google.state_manifest_contract("native").query == (
        "SELECT * FROM feature_outputs WHERE feature = ?"
    )
    assert apple.state_manifest_contract("curated").row_count_column == "curated_rows"
    assert google.state_manifest_contract("curated").row_count_column == "curated_rows"
    assert apple.required_root_files("native") == ()
    assert apple.required_root_files("curated") == ()
    assert google.required_root_files("native") == ()
    assert google.required_root_files("curated") == (".curation_contract.json",)
    assert apple.state_manifest_contract("native").required_columns == (
        "participant_id", "feature", "row_count", "size_bytes", "sha256",
    )
    assert google.state_manifest_contract("native").required_columns == (
        "participant_id", "feature", "filename", "row_count", "size_bytes", "sha256",
    )

    apple_contract = apple.feature_contract("StepCount", "native")
    google_contract = google.feature_contract("StepCount", "native")
    assert apple_contract.unknown_columns_as_text is False
    assert apple_contract.index_sort_kind is None
    assert google_contract.unknown_columns_as_text is True
    assert google_contract.index_sort_kind == "mergesort"


def test_google_public_catalog_matches_native_and_curation_registries():
    assert GOOGLE_FITNESS_PUBLIC_FEATURES == tuple(google_feature_names())
    assert set(GOOGLE_FITNESS_PUBLIC_FEATURES) == set(CURATION_POLICIES)
    assert len(GOOGLE_FITNESS_PUBLIC_FEATURES) == 67


def test_package_import_and_loader_construction_do_not_import_scientific_source_stacks():
    script = r'''
import sys
from wearable_project import DataLoaders
from wearable_project.DataLoaders.StepCountLoader import StepCountLoader
from wearable_project.DataLoaders.WalkingLoader import WalkingLoader
StepCountLoader()
StepCountLoader(source="google_fitness")
WalkingLoader()
DataLoaders.available_features()
DataLoaders.available_features(source="google_fitness")
prefixes = (
    "wearable_project.processing.apple_healthkit",
    "wearable_project.curation.apple_healthkit",
    "wearable_project.processing.google_fitness",
    "wearable_project.curation.google_fitness",
)
loaded = sorted(name for name in sys.modules if name.startswith(prefixes))
assert not loaded, loaded
print("isolated")
'''
    completed = _run_isolation_script(script)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "isolated"


def test_google_static_info_does_not_import_apple_scientific_modules():
    script = r'''
import sys
from wearable_project import DataLoaders
DataLoaders.info(source="google_fitness")
DataLoaders.info("StepCount", source="google_fitness", phase="curated")
loaded = sorted(
    name for name in sys.modules
    if name.startswith((
        "wearable_project.processing.apple_healthkit",
        "wearable_project.curation.apple_healthkit",
    ))
)
assert not loaded, loaded
print("google-only")
'''
    completed = _run_isolation_script(script)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "google-only"


def test_apple_static_info_does_not_import_google_scientific_modules():
    script = r'''
import sys
from wearable_project import DataLoaders
DataLoaders.info()
DataLoaders.info("StepCount", phase="curated")
loaded = sorted(
    name for name in sys.modules
    if name.startswith((
        "wearable_project.processing.google_fitness",
        "wearable_project.curation.google_fitness",
    ))
)
assert not loaded, loaded
print("apple-only")
'''
    completed = _run_isolation_script(script)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "apple-only"


def test_google_static_info_covers_every_feature_and_phase_without_root_io(monkeypatch: pytest.MonkeyPatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("static info attempted data-root I/O")

    monkeypatch.setattr(AppleHealthFeatureLoader, "_checked_root", forbidden)
    overview = static_info(source="google_fitness")
    assert overview.payload["feature_count"] == 67
    assert overview.payload["catalog_consistency"] == {
        "public_equals_native_registry": True,
        "public_equals_curation_registry": True,
    }

    for feature in GOOGLE_FITNESS_PUBLIC_FEATURES:
        for phase in ("native", "curated"):
            report = static_info(feature, source="google_fitness", phase=phase)
            assert report.payload["wearable_source"] == "google_fitness"
            assert report.payload["feature"] == feature
            assert report.payload["phase"] == phase
            assert report.payload["date_index"]["column"]
            assert report.payload["projections"]["default"]["columns"]
            assert report.payload["integrity"]["source_and_phase_checked_before_csv"] is True


def test_google_info_describes_step_count_and_withheld_scalar_units():
    step = static_info("StepCount", source="google_fitness", phase="curated", include_evidence=True).payload
    assert step["date_index"]["column"] == "outer_datetime"
    assert step["contract_versions"]["info_contract_version"] == "dataloader-info-google-fitness-2"
    assert step["date_index"]["canonical_day_derived"] is False
    assert step["group_policy"]["role_column"] == "step_count_role"
    assert step["columns"]["curation_virtual"] == [
        "canonical_value", "canonical_unit", "curation_unit_status", "unit_evidence", "derivation_method",
    ]
    assert step["harmonization"]["supported"] is True
    assert step["harmonization"]["resolved_values_available"] is True
    assert step["harmonization"]["established_unit"] == "count"
    assert step["harmonization"]["primary_measurement"] == "value"
    assert step["harmonization"]["auxiliary_measurements"] == ["raw_value"]
    assert step["harmonization"]["multi_measurement_feature"] is False
    assert step["harmonization"]["result_policy"] == "resolved"
    assert "singular with_harmonized_values() helper is intentionally inapplicable" not in static_info(
        "StepCount", source="google_fitness", phase="curated"
    ).text
    assert step["evidence"]

    calories = static_info("Calories", source="google_fitness", phase="curated").payload
    assert calories["harmonization"] == {
        "supported": True,
        "resolved_values_available": False,
        "established_unit": None,
        "unit_candidates_only": ["kcal"],
        "primary_measurement": "value",
        "auxiliary_measurements": [],
        "multi_measurement_feature": False,
        "result_policy": "withheld_by_policy",
    }
    assert calories["measurements"][0]["canonical_projection_enabled"] is False
    assert calories["measurements"][0]["version_units"][0]["canonicalization"] == "withhold"

    walking = static_info("Walking", source="google_fitness", phase="curated").payload
    assert walking["harmonization"]["supported"] is False
    assert walking["harmonization"]["primary_measurement"] is None
    assert walking["harmonization"]["multi_measurement_feature"] is True
    assert walking["harmonization"]["result_policy"] == "not_applicable"
    assert walking["columns"]["measurement"] == ["calories", "distance", "quantity"]


def test_apple_static_info_remains_byte_and_payload_equivalent_to_increment3():
    expected = json.loads(APPLE_INFO_HASHES.read_text(encoding="utf-8"))
    reports = {"overview": static_info()}
    for feature in DataLoaders.available_features():
        for phase in ("curated", "native"):
            reports[f"{feature}:{phase}"] = static_info(feature, phase=phase)

    assert set(reports) == set(expected)
    for name, report in reports.items():
        baseline_text = report.text.replace("0.4.0", "0.3.0")
        baseline_payload = _with_increment3_release_identity(report.payload)
        assert hashlib.sha256(baseline_text.encode("utf-8")).hexdigest() == expected[name]["text_sha256"], name
        assert _canonical_payload_hash(baseline_payload) == expected[name]["payload_sha256"], name


def test_both_state_markers_are_rejected_before_csv_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "ambiguous"
    _apple_state(root, "native")
    _apple_state(root, "curated")

    def forbidden(*args, **kwargs):
        raise AssertionError("feature CSV was parsed before phase identity")

    monkeypatch.setattr(AppleHealthFeatureLoader, "_read_feature_file", forbidden)
    with pytest.raises(DataLoaderConfigurationError, match="both native and curated state markers"):
        StepCountLoader(root=root, state_validation="off").get_data()


@pytest.mark.parametrize("marker", [".wearable_state.sqlite", ".wearable_curation_state.sqlite"])
def test_malformed_state_marker_is_not_treated_as_an_unmanaged_tree(tmp_path: Path, marker: str):
    root = tmp_path / "malformed-marker"
    (root / marker).mkdir(parents=True)
    phase = "native" if marker == ".wearable_state.sqlite" else "curated"
    with pytest.raises(DataLoaderStateError, match="is not a state database file"):
        StepCountLoader(phase=phase, root=root, state_validation="off").get_data()


def test_present_but_unrecognized_managed_state_is_not_downgraded_under_off(tmp_path: Path):
    root = tmp_path / "unknown"
    root.mkdir()
    with sqlite3.connect(root / ".wearable_state.sqlite") as connection:
        connection.execute("CREATE TABLE future_state (value TEXT)")
    with pytest.raises(DataLoaderStateError, match="no recognized source signature"):
        StepCountLoader(phase="native", root=root, state_validation="off").get_data()


def test_present_but_unreadable_managed_state_is_not_downgraded_under_off(tmp_path: Path):
    root = tmp_path / "corrupt"
    root.mkdir()
    (root / ".wearable_state.sqlite").write_bytes(b"not a sqlite database")
    with pytest.raises(DataLoaderStateError, match="state identity unreadable"):
        StepCountLoader(phase="native", root=root, state_validation="off").get_data()


@pytest.mark.parametrize("source", [WearableSource.APPLE_HEALTHKIT, WearableSource.GOOGLE_FITNESS])
def test_managed_manifest_missing_required_columns_fails_before_file_use(
    tmp_path: Path, source: WearableSource,
):
    root = tmp_path / source.value
    database = _apple_state(root) if source is WearableSource.APPLE_HEALTHKIT else _google_native_state(root)
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE feature_outputs")
        connection.execute(
            "CREATE TABLE feature_outputs (participant_id TEXT, feature TEXT, row_count INTEGER, size_bytes INTEGER)"
        )
    with pytest.raises(DataLoaderStateError, match="missing required loader column.*sha256"):
        StepCountLoader(
            source=source,
            phase="native",
            root=root,
            state_validation="required",
        ).get_data()


@pytest.mark.parametrize("source", [WearableSource.APPLE_HEALTHKIT, WearableSource.GOOGLE_FITNESS])
def test_duplicate_participant_feature_state_rows_are_not_silently_overwritten(
    tmp_path: Path, source: WearableSource,
):
    root = tmp_path / source.value
    database = _apple_state(root) if source is WearableSource.APPLE_HEALTHKIT else _google_native_state(root)
    with sqlite3.connect(database) as connection:
        if source is WearableSource.APPLE_HEALTHKIT:
            rows = [("123", "StepCount", 0, 0, "0" * 64)] * 2
            connection.executemany(
                "INSERT INTO feature_outputs (participant_id, feature, row_count, size_bytes, sha256) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        else:
            rows = [("123", "StepCount", "StepCount.csv", 0, 0, "0" * 64, 0)] * 2
            connection.executemany(
                "INSERT INTO feature_outputs VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
    with pytest.raises(DataLoaderStateError, match="duplicate rows for participant '123'"):
        StepCountLoader(
            source=source,
            phase="native",
            root=root,
            state_validation="required",
        ).get_data()


@pytest.mark.parametrize("source", [WearableSource.APPLE_HEALTHKIT, WearableSource.GOOGLE_FITNESS])
@pytest.mark.parametrize("suffix", ["-wal", "-journal"])
def test_active_managed_state_is_rejected_even_when_file_verification_is_off(
    tmp_path: Path, source: WearableSource, suffix: str,
):
    root = tmp_path / f"{source.value}-{suffix[1:]}"
    database = _apple_state(root) if source is WearableSource.APPLE_HEALTHKIT else _google_native_state(root)
    Path(f"{database}{suffix}").write_bytes(b"active")
    with pytest.raises(DataLoaderStateError, match="uncommitted writes"):
        StepCountLoader(
            source=source,
            phase="native",
            root=root,
            state_validation="off",
        ).get_data()


def test_unmanaged_trimmed_apple_tree_still_uses_the_adapter_curated_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _minimal_apple_curated(tmp_path)
    loader = StepCountLoader(root=tmp_path, state_validation="off")
    called = {"prepare": False}
    original = loader._source_adapter.prepare_curated_frame

    def wrapped(frame, path, feature):
        called["prepare"] = True
        return original(frame, path, feature)

    monkeypatch.setattr(loader._source_adapter, "prepare_curated_frame", wrapped)
    result = loader.get_data()
    assert called["prepare"] is True
    assert len(result.df) == 1
    assert result.load_report["source_verified"] is False
    assert result.df["include_by_default"].all()


@pytest.mark.parametrize(
    "registration_code",
    ["../outside", "a/b", r"a\b", ".", "..", "/tmp/outside", r"C:\outside"],
)
def test_registration_codes_must_be_one_safe_participant_component(
    tmp_path: Path, registration_code: str,
):
    with pytest.raises(DataLoaderConfigurationError, match="exactly one participant directory"):
        StepCountLoader(phase="native", root=tmp_path, state_validation="off").get_data(
            registration_codes=[registration_code]
        )


@pytest.mark.parametrize("source", [WearableSource.APPLE_HEALTHKIT, WearableSource.GOOGLE_FITNESS])
def test_managed_state_participant_ids_cannot_escape_the_product_root(
    tmp_path: Path, source: WearableSource,
):
    root = tmp_path / source.value
    database = _apple_state(root) if source is WearableSource.APPLE_HEALTHKIT else _google_native_state(root)
    with sqlite3.connect(database) as connection:
        if source is WearableSource.APPLE_HEALTHKIT:
            connection.execute(
                "INSERT INTO feature_outputs VALUES (?, ?, ?, ?, ?)",
                ("../outside", "StepCount", 0, 0, "0" * 64),
            )
        else:
            connection.execute(
                "INSERT INTO feature_outputs VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("../outside", "StepCount", "StepCount.csv", 0, 0, "0" * 64, 0),
            )
    with pytest.raises(DataLoaderStateError, match="participant_id.*only one directory"):
        StepCountLoader(
            source=source, phase="native", root=root, state_validation="required",
        ).get_data()


@pytest.mark.parametrize("source", [WearableSource.APPLE_HEALTHKIT, WearableSource.GOOGLE_FITNESS])
def test_managed_state_participant_ids_are_not_silently_trimmed(
    tmp_path: Path, source: WearableSource,
):
    root = tmp_path / f"{source.value}-whitespace"
    database = _apple_state(root) if source is WearableSource.APPLE_HEALTHKIT else _google_native_state(root)
    with sqlite3.connect(database) as connection:
        if source is WearableSource.APPLE_HEALTHKIT:
            connection.execute(
                "INSERT INTO feature_outputs VALUES (?, ?, ?, ?, ?)",
                (" 123 ", "StepCount", 0, 0, "0" * 64),
            )
        else:
            connection.execute(
                "INSERT INTO feature_outputs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (" 123 ", "StepCount", "StepCount.csv", 0, 0, "0" * 64, 0),
            )
    with pytest.raises(DataLoaderStateError, match="leading or trailing whitespace"):
        StepCountLoader(
            source=source, phase="native", root=root, state_validation="required",
        ).get_data()


@pytest.mark.parametrize("source", [WearableSource.APPLE_HEALTHKIT, WearableSource.GOOGLE_FITNESS])
@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("row_count", -1, "nonnegative integer"),
        ("row_count", "not-an-integer", "nonnegative integer"),
        ("sha256", "not-a-digest", "64-digit SHA-256"),
    ],
)
def test_managed_state_manifest_values_are_validated_before_use(
    tmp_path: Path, source: WearableSource, column: str, value: object, message: str,
):
    root = tmp_path / f"{source.value}-{column}"
    database = _apple_state(root) if source is WearableSource.APPLE_HEALTHKIT else _google_native_state(root)
    payload = {
        "participant_id": "123",
        "feature": "StepCount",
        "filename": "StepCount.csv",
        "row_count": 0,
        "size_bytes": 0,
        "sha256": "0" * 64,
        "represented_occurrences": 0,
    }
    payload[column] = value
    with sqlite3.connect(database) as connection:
        if source is WearableSource.APPLE_HEALTHKIT:
            connection.execute(
                "INSERT INTO feature_outputs VALUES (?, ?, ?, ?, ?)",
                tuple(payload[name] for name in ("participant_id", "feature", "row_count", "size_bytes", "sha256")),
            )
        else:
            connection.execute(
                "INSERT INTO feature_outputs VALUES (?, ?, ?, ?, ?, ?, ?)",
                tuple(payload[name] for name in (
                    "participant_id", "feature", "filename", "row_count", "size_bytes", "sha256",
                    "represented_occurrences",
                )),
            )
    with pytest.raises(DataLoaderStateError, match=message):
        StepCountLoader(
            source=source, phase="native", root=root, state_validation="required",
        ).get_data()


def test_state_source_metadata_must_agree_with_known_schema_signature(tmp_path: Path):
    root = tmp_path / "spoofed-source"
    database = _google_native_state(root)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE source_occurrences (occurrence_id TEXT)")
        connection.execute("CREATE TABLE malformed_candidates (candidate_id TEXT)")
        connection.execute("UPDATE metadata SET value = 'apple_healthkit' WHERE key = 'source_key'")
    with pytest.raises(DataLoaderStateError, match="disagrees with Google native state schema signature"):
        StepCountLoader(phase="native", root=root, state_validation="off").get_data()


def test_conflicting_known_state_schema_signatures_are_rejected(tmp_path: Path):
    root = tmp_path / "conflicting-state"
    database = _apple_state(root)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE source_occurrences (occurrence_id TEXT)")
        connection.execute("CREATE TABLE malformed_candidates (candidate_id TEXT)")
    with pytest.raises(DataLoaderStateError, match="conflicting source signatures"):
        StepCountLoader(phase="native", root=root, state_validation="off").get_data()


@pytest.mark.parametrize(
    ("metadata_sql", "rows", "message"),
    [
        (
            "CREATE TABLE metadata (key TEXT, value TEXT)",
            [("source_key", "google_fitness"), ("source_key", "google_fitness")],
            "repeats key 'source_key'",
        ),
        (
            "CREATE TABLE metadata (key TEXT)",
            [("source_key",)],
            "lacks required key/value columns",
        ),
    ],
)
def test_malformed_state_metadata_is_not_trusted(
    tmp_path: Path, metadata_sql: str, rows: list[tuple[object, ...]], message: str,
):
    root = tmp_path / "malformed-metadata"
    root.mkdir()
    database = root / ".wearable_state.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(metadata_sql)
        placeholders = ", ".join("?" for _ in rows[0])
        connection.executemany(f"INSERT INTO metadata VALUES ({placeholders})", rows)
        connection.execute(
            "CREATE TABLE feature_outputs (participant_id TEXT, feature TEXT, filename TEXT, "
            "row_count INTEGER, size_bytes INTEGER, sha256 TEXT, represented_occurrences INTEGER)"
        )
        connection.execute("CREATE TABLE source_occurrences (occurrence_id TEXT)")
        connection.execute("CREATE TABLE malformed_candidates (candidate_id TEXT)")
    with pytest.raises(DataLoaderStateError, match=message):
        StepCountLoader(
            source="google_fitness", phase="native", root=root, state_validation="off",
        ).get_data()


def test_shared_curated_run_tables_do_not_make_apple_state_look_google(tmp_path: Path):
    root = tmp_path / "apple-curated-shared-tables"
    database = _apple_state(root, "curated")
    with sqlite3.connect(database) as connection:
        # Both accepted curation implementations use this orchestration table name. It is not source-unique.
        connection.execute("CREATE TABLE curation_run_features (run_id TEXT)")
    loader = StepCountLoader(phase="curated", root=root, state_validation="off")
    result = loader.get_data()
    assert result.load_report["wearable_source"] == "apple_healthkit"
    assert result.load_report["source_verified"] is True
    assert result.load_report["source_identity_evidence"] == "legacy Apple curated state schema signature"


def test_google_curated_column_signature_corroborates_and_guards_source_metadata(tmp_path: Path):
    root = tmp_path / "google-curated-spoof"
    root.mkdir()
    database = root / ".wearable_curation_state.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO metadata VALUES ('source_key', 'apple_healthkit')")
        connection.execute(
            "CREATE TABLE curation_outputs ("
            "participant_id TEXT, feature TEXT, filename TEXT, native_rows INTEGER, curated_rows INTEGER, "
            "size_bytes INTEGER, sha256 TEXT, policy_fingerprint TEXT, "
            "active_materialized_columns_json TEXT, derivation_counts_json TEXT, tracking_counts_json TEXT, "
            "step_count_role_counts_json TEXT)"
        )
    with pytest.raises(DataLoaderStateError, match="disagrees with Google curated state schema signature"):
        StepCountLoader(phase="curated", root=root, state_validation="off").get_data()


@pytest.mark.parametrize(
    "payload",
    ['{"x": -1}', '{"x": true}', '{"x": 1.5}', '{"x": 1, "x": 2}'],
)
def test_curated_state_json_counters_reject_negative_noninteger_and_duplicate_values(payload: str):
    from wearable_project.DataLoaders.apple_healthkit.curated import _json_counter as apple_counter
    from wearable_project.DataLoaders.google_fitness.curated import _json_counter as google_counter

    for decoder in (apple_counter, google_counter):
        with pytest.raises(DataLoaderStateError, match="JSON counter|nonnegative integer"):
            decoder({"counts": payload}, "counts")


def test_google_participant_status_state_is_validated_before_empty_source_classification(tmp_path: Path):
    root = tmp_path / "invalid-participant-status"
    database = _google_native_state(root)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO participants VALUES (?, ?)",
            ("../outside", "complete_empty"),
        )
    with pytest.raises(DataLoaderStateError, match="participants contains participant_id.*path"):
        StepCountLoader(
            source="google_fitness", phase="native", root=root, state_validation="required",
        ).get_data(registration_codes=["123"])
