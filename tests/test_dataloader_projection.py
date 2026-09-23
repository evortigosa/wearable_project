"""
Wearable Data Processing and Modeling project
Tests for the column-role schema and the DataLoader projection it drives. The schema in ``curation/schema.py``
restates column names that stable processing and curation modules already declare for other purposes. It was
added without editing those modules, so the correspondence is asserted here instead of enforced by an import.
If ``processing/cleaners.py`` gains a column group entry, or the registry gains a measurement column, the
consistency tests below fail until the role table is updated.
"""


from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import pytest
from wearable_project.curation import schema as S
from wearable_project.curation.schema import ColumnRole
from wearable_project.processing.cleaners import (
    AUDIT_COLUMNS, CONTEXT_COLUMNS, PROVENANCE_COLUMNS, UNIT_COLUMNS,
)
from wearable_project.processing.registry import SPECS
from wearable_project.DataLoaders.ActivitySummaryLoader import ActivitySummaryLoader
from wearable_project.DataLoaders.BloodGlucoseLoader import BloodGlucoseLoader
from wearable_project.DataLoaders.BloodPressureLoader import BloodPressureLoader
from wearable_project.DataLoaders.ElectrocardiogramLoader import ElectrocardiogramLoader
from wearable_project.DataLoaders.HeartRateLoader import HeartRateLoader
from wearable_project.DataLoaders.SleepLoader import SleepLoader
from wearable_project.DataLoaders.StepCountLoader import StepCountLoader
from wearable_project.DataLoaders.WeightLoader import WeightLoader
from wearable_project.exceptions import DataLoaderConfigurationError, DataLoaderReadError


# Sample roots are optional: the consistency and unit tests run anywhere, the end-to-end tests need data.
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
    reason="curated and native sample roots are required for end-to-end projection tests",
)

# Columns the curation layer appends, per the non-destructive contract in README.md.
CURATION_APPENDED = frozenset({
    "canonical_value", "canonical_unit", "curation_unit_status", "unit_evidence", "unit_epoch_id",
    "acquisition_method", "curation_status", "curation_flags", "include_by_default",
})
# Core columns fixed by processing.cleaners.output_dataframe.
CORE_COLUMNS = frozenset({
    "start_date", "end_date", "datetime", "created_at", "updated_at",
    "data_source", "collecting_method_version",
})
FEATURE_SPECIFIC_COLUMNS = frozenset({
    "classification", "algorithm_version", "waveform_sample_count", "payload_index",
})


def _sample_columns(root: Path) -> set[str]:
    found: set[str] = set()
    for path in root.glob("*/*.csv"):
        found |= set(pd.read_csv(path, nrows=0).columns)
    return found


# ------------------------------------------------------- schema / stable-module consistency
def test_every_declared_column_is_known_to_a_stable_module():
    """No invented names: each role entry is a column some stable module already declares."""
    stable = (
        CORE_COLUMNS | FEATURE_SPECIFIC_COLUMNS | CURATION_APPENDED
        | set(PROVENANCE_COLUMNS) | set(CONTEXT_COLUMNS) | set(UNIT_COLUMNS) | set(AUDIT_COLUMNS)
        | S.declared_measurement_columns()
    )
    invented = sorted(set(S.COLUMN_ROLES) - stable)
    assert invented == [], f"role table declares columns no stable module emits: {invented}"


def test_every_stable_column_has_a_declared_role():
    """The reverse direction: a new column in cleaners.py or the registry must be given a role."""
    stable = (
        CORE_COLUMNS | FEATURE_SPECIFIC_COLUMNS | CURATION_APPENDED
        | set(PROVENANCE_COLUMNS) | set(CONTEXT_COLUMNS) | set(UNIT_COLUMNS) | set(AUDIT_COLUMNS)
        | S.declared_measurement_columns()
    )
    undeclared = sorted(stable - set(S.COLUMN_ROLES))
    assert undeclared == [], f"columns without a declared role: {undeclared}"


def test_every_registry_measurement_column_has_a_role():
    for spec in SPECS.values():
        for column in spec.measurement_columns:
            assert column in S.COLUMN_ROLES, f"{spec.name}.{column} has no declared role"


def test_measurement_columns_are_measurements_except_the_recorded_override():
    """voltage_measurements is the one measurement column deliberately demoted, for its size."""
    demoted = {
        column for column in S.declared_measurement_columns()
        if S.COLUMN_ROLES[column] is not ColumnRole.MEASUREMENT
    }
    assert demoted == {"voltage_measurements"}
    assert S.role_of("voltage_measurements") is ColumnRole.PAYLOAD


@needs_sample
def test_no_column_in_either_sample_root_is_undeclared():
    observed = _sample_columns(CURATED_SAMPLE) | _sample_columns(NATIVE_SAMPLE)
    assert S.undeclared_columns(observed) == ()


# ------------------------------------------------------------------ projection structure
def test_projections_are_cumulative():
    assert S.DEFAULT_ROLES < S.ANALYSIS_ROLES < S.FULL_ROLES


def test_full_admits_every_role():
    assert S.FULL_ROLES == frozenset(ColumnRole)


def test_available_projections_are_ordered_widest_last():
    assert S.available_projections() == ("default", "analysis", "full")
    sizes = [len(S.projection_roles(name)) for name in S.available_projections()]
    assert sizes == sorted(sizes)


def test_each_projection_has_a_summary():
    for name in S.available_projections():
        assert S.describe_projection(name).strip()


def test_sensitive_and_bulky_roles_are_withheld_from_default():
    for role in (ColumnRole.IDENTIFIER, ColumnRole.PAYLOAD, ColumnRole.INGEST,
                 ColumnRole.TIME_ZONE, ColumnRole.RECONCILIATION, ColumnRole.CURATION_AUDIT):
        assert role not in S.DEFAULT_ROLES


def test_identifiers_and_payload_are_withheld_from_analysis_too():
    for role in (ColumnRole.IDENTIFIER, ColumnRole.PAYLOAD, ColumnRole.INGEST):
        assert role not in S.ANALYSIS_ROLES


def test_analysis_adds_exactly_the_documented_roles():
    assert S.ANALYSIS_ROLES - S.DEFAULT_ROLES == {
        ColumnRole.CURATION_AUDIT, ColumnRole.DEVICE,
        ColumnRole.RECONCILIATION, ColumnRole.TIME_ZONE,
    }


def test_columns_for_projection_preserves_input_order():
    available = ["end_date", "value", "record_id", "start_date"]
    assert S.columns_for_projection("default", available) == ["end_date", "value", "start_date"]


def test_columns_for_projection_full_returns_everything_including_unknowns():
    available = ["value", "record_id", "a_future_column"]
    assert S.columns_for_projection("full", available) == available


def test_unknown_columns_are_withheld_from_default_and_analysis():
    available = ["value", "a_future_column"]
    assert S.columns_for_projection("default", available) == ["value"]
    assert S.columns_for_projection("analysis", available) == ["value"]
    assert S.undeclared_columns(available) == ("a_future_column",)


def test_unknown_projection_name_raises_keyerror_in_the_schema():
    with pytest.raises(KeyError):
        S.projection_roles("everything")
    assert not S.is_known_projection("everything")


# --------------------------------------------------------------------- loader behaviour
@needs_sample
def test_full_projection_returns_every_stored_column_plus_the_dense_schema():
    """
    Since 0.2.0 ``full`` means every column of the curated logical schema, not the stored bytes: sparse
    curation columns absent from a file are reconstructed, so full is a superset of what is on disk.
    """
    data = StepCountLoader(root=CURATED_SAMPLE).get_data(projection="full")
    stored = set()
    for path in CURATED_SAMPLE.glob("*/StepCount.csv"):
        stored |= set(pd.read_csv(path, nrows=0).columns)
    dense = {"acquisition_method", "curation_status", "curation_flags", "include_by_default"}
    assert set(data.df.columns) == stored | dense
    assert data.load_report["columns_withheld"] == []


@needs_sample
def test_default_projection_withholds_identifiers_and_ingest():
    data = HeartRateLoader(root=CURATED_SAMPLE).get_data()
    returned = set(data.df.columns)
    for column in ("record_id", "source_id", "source_name", "metadata",
                   "created_at", "updated_at", "data_source"):
        assert column not in returned
    assert {"record_id", "source_id", "source_name"} <= set(data.load_report["columns_withheld"])


@needs_sample
def test_default_projection_keeps_the_analytical_core():
    data = HeartRateLoader(root=CURATED_SAMPLE).get_data()
    returned = set(data.df.columns)
    for column in ("start_date", "end_date", "value", "raw_unit",
                   "utc_offset_minutes", "acquisition_method"):
        assert column in returned


@needs_sample
def test_default_projection_reduces_memory_substantially():
    full = HeartRateLoader(root=CURATED_SAMPLE).get_data(projection="full").df
    default = HeartRateLoader(root=CURATED_SAMPLE).get_data().df
    assert len(full) == len(default)
    ratio = full.memory_usage(deep=True).sum() / default.memory_usage(deep=True).sum()
    assert ratio > 2.5, f"expected a meaningful reduction, got {ratio:.2f}x"


@needs_sample
def test_ecg_waveform_is_withheld_by_default_and_reachable_explicitly():
    default = ElectrocardiogramLoader(root=CURATED_SAMPLE).get_data()
    assert "voltage_measurements" not in default.df.columns
    assert "waveform_sample_count" in default.df.columns

    full = ElectrocardiogramLoader(root=CURATED_SAMPLE).get_data(projection="full")
    assert "voltage_measurements" in full.df.columns

    named = ElectrocardiogramLoader(root=CURATED_SAMPLE).get_data(columns=["voltage_measurements"])
    assert list(named.df.columns) == ["voltage_measurements"]

    # The waveform dominates the feature, which is why it is not a default column.
    assert full.df.memory_usage(deep=True).sum() > 100 * default.df.memory_usage(deep=True).sum()


@needs_sample
def test_blood_pressure_keeps_its_coupled_pair_by_default():
    data = BloodPressureLoader(root=CURATED_SAMPLE).get_data()
    assert "blood_pressure_systolic_value" in data.df.columns
    assert "blood_pressure_diastolic_value" in data.df.columns


@needs_sample
def test_activity_summary_keeps_its_measurements_and_disambiguator():
    data = ActivitySummaryLoader(root=CURATED_SAMPLE).get_data()
    for column in ("apple_stand_hours", "apple_exercise_time", "active_energy_burned",
                   "apple_stand_hours_goal", "payload_index", "datetime"):
        assert column in data.df.columns, column


@needs_sample
def test_cgm_status_and_trend_survive_the_default_projection():
    """status marks readings pinned at the sensor reporting limits, so it qualifies the value."""
    data = BloodGlucoseLoader(root=CURATED_SAMPLE).get_data()
    for column in ("status", "trend_arrow", "trend_rate", "canonical_value", "canonical_unit"):
        assert column in data.df.columns, column


@needs_sample
def test_sleep_categorical_value_survives():
    data = SleepLoader(root=CURATED_SAMPLE).get_data()
    assert "value" in data.df.columns
    assert set(data.df["value"].dropna().unique()) <= {
        "INBED", "ASLEEP", "AWAKE", "CORE", "DEEP", "REM",
    }


@needs_sample
def test_curation_verdicts_are_default_but_the_audit_trail_is_not():
    data = WeightLoader(root=CURATED_SAMPLE).get_data()
    assert "include_by_default" in data.df.columns
    assert "curation_status" in data.df.columns
    assert "curation_unit_status" in data.df.columns
    assert "unit_evidence" not in data.df.columns
    assert "unit_epoch_id" not in data.df.columns

    wider = WeightLoader(root=CURATED_SAMPLE).get_data(projection="analysis")
    assert "unit_evidence" in wider.df.columns


@needs_sample
def test_explicit_columns_override_the_projection():
    """An explicit list reaches columns the projection withholds, which is why it takes precedence."""
    data = HeartRateLoader(root=CURATED_SAMPLE).get_data(
        columns=["value", "record_id"], projection="default"
    )
    assert list(data.df.columns) == ["value", "record_id"]
    assert data.load_report["projection"] == "columns"
    assert data.load_report["columns_withheld"] == []
    assert "record_id" not in HeartRateLoader(root=CURATED_SAMPLE).get_data().df.columns


@needs_sample
def test_unknown_projection_raises_a_configuration_error():
    with pytest.raises(DataLoaderConfigurationError, match="unknown projection"):
        StepCountLoader(root=CURATED_SAMPLE).get_data(projection="everything")


@needs_sample
def test_native_phase_supports_projections():
    data = StepCountLoader(phase="native", root=NATIVE_SAMPLE).get_data()
    assert len(data.df) > 0
    assert "record_id" not in data.df.columns
    assert "value" in data.df.columns


@needs_sample
def test_projection_does_not_change_row_content():
    full = StepCountLoader(root=CURATED_SAMPLE).get_data(projection="full").df
    default = StepCountLoader(root=CURATED_SAMPLE).get_data().df
    assert full.index.equals(default.index)
    for column in default.columns:
        assert full[column].equals(default[column]), column


@needs_sample
def test_date_anchor_is_unaffected_by_the_projection():
    full = StepCountLoader(root=CURATED_SAMPLE).get_data(projection="full").df
    default = StepCountLoader(root=CURATED_SAMPLE).get_data().df
    assert full.index.get_level_values("Date").equals(default.index.get_level_values("Date"))


# ------------------------------------------------- interaction with the earlier correctness fixes
@needs_sample
def test_empty_projected_result_keeps_the_projected_schema():
    populated = StepCountLoader(root=CURATED_SAMPLE).get_data()
    empty = StepCountLoader(root=CURATED_SAMPLE).get_data(start_date="2099-01-01")
    assert len(empty.df) == 0
    assert list(empty.df.columns) == list(populated.df.columns)


@needs_sample
def test_empty_projected_result_concatenates_with_a_populated_one():
    populated = StepCountLoader(root=CURATED_SAMPLE).get_data(
        registration_codes="10K_1235738253"
    ).df
    empty = StepCountLoader(root=CURATED_SAMPLE).get_data(start_date="2099-01-01").df
    combined = pd.concat([empty, populated])
    assert list(combined.columns) == list(populated.columns)
    assert len(combined) == len(populated)


@needs_sample
def test_empty_full_result_keeps_the_full_schema():
    populated = StepCountLoader(root=CURATED_SAMPLE).get_data(projection="full")
    empty = StepCountLoader(root=CURATED_SAMPLE).get_data(
        projection="full", start_date="2099-01-01"
    )
    assert list(empty.df.columns) == list(populated.df.columns)


@needs_sample
def test_absent_column_still_raises_under_a_projection():
    with pytest.raises(DataLoaderReadError, match="not_a_column"):
        StepCountLoader(root=CURATED_SAMPLE).get_data(columns=["not_a_column"])


@needs_sample
def test_default_inclusion_only_still_filters_under_the_projection():
    """
    The inclusion filter runs before the projection, so the two are independent. The sample root carries
    its state database, so reconstructed inclusion values are verified and no warning is expected.
    """
    projected = WeightLoader(root=CURATED_SAMPLE).get_data(default_inclusion_only=True)
    full = WeightLoader(root=CURATED_SAMPLE).get_data(default_inclusion_only=True, projection="full")
    assert projected.load_report["rows"] == full.load_report["rows"] == 82
    assert projected.load_report["default_inclusion_rows_dropped"] == 375


@needs_sample
def test_phase_mismatch_is_still_detected_under_a_projection():
    with pytest.raises(DataLoaderConfigurationError, match="contains native data"):
        StepCountLoader(phase="curated", root=NATIVE_SAMPLE).get_data()


# ------------------------------------------------------------------------ metadata reporting
@needs_sample
def test_metadata_reports_the_projection_and_what_it_withheld():
    data = HeartRateLoader(root=CURATED_SAMPLE).get_data()
    assert data.load_report["projection"] == "default"
    withheld = data.load_report["columns_withheld"]
    assert withheld == sorted(withheld)
    assert set(withheld).isdisjoint(set(data.df.columns))


@needs_sample
def test_withheld_plus_returned_equals_the_stored_column_set():
    data = HeartRateLoader(root=CURATED_SAMPLE).get_data()
    full = HeartRateLoader(root=CURATED_SAMPLE).get_data(projection="full")
    assert set(data.df.columns) | set(data.load_report["columns_withheld"]) == set(full.df.columns)


@needs_sample
def test_full_projection_withholds_nothing():
    data = StepCountLoader(root=CURATED_SAMPLE).get_data(projection="full")
    assert data.load_report["projection"] == "full"
    assert data.load_report["columns_withheld"] == []


# ---------------------------------------------------------------------- import hygiene
def test_importing_a_loader_does_not_pull_the_curation_package():
    """The schema import is deferred so a loader import stays free of the policy and engine stack."""
    import subprocess
    import sys
    script = (
        "import sys\n"
        "import wearable_project.DataLoaders.StepCountLoader\n"
        "loaded = [m for m in sys.modules if m.startswith('wearable_project.curation')]\n"
        "assert not loaded, loaded\n"
        "print('clean')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


@needs_sample
def test_schema_is_imported_on_first_get_data():
    import subprocess
    import sys
    script = (
        "import sys\n"
        "from wearable_project.DataLoaders.StepCountLoader import StepCountLoader\n"
        "assert 'wearable_project.curation.schema' not in sys.modules\n"
        f"StepCountLoader(root={str(CURATED_SAMPLE)!r}).get_data(registration_codes='10K_0000000000')\n"
        "assert 'wearable_project.curation.schema' in sys.modules\n"
        "print('deferred')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "deferred" in result.stdout
