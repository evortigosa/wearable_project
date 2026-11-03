"""
Wearable Data Processing and Modeling project
Source identifiers, compatibility aliases, source-qualified roots, feature catalogs, and managed-root probing.
Source selection is never inferred from a feature filename. A caller selects a scientific source contract and
managed-root metadata verifies that choice before a feature CSV is parsed. Unmanaged trimmed trees remain
loadable, but are explicitly reported as source-unverified; a present but unsafe or unknown managed state is
refused rather than silently downgraded to unmanaged.
"""


from __future__ import annotations
from contextlib import closing
from enum import Enum
from functools import lru_cache
from pathlib import Path
import sqlite3
from wearable_project.DataLoaders._contracts import RootIdentity, SourceAdapter
from wearable_project.exceptions import DataLoaderConfigurationError


class WearableSource(str, Enum):
    """Supported source contracts for processed wearable products."""

    APPLE_HEALTHKIT = "apple_healthkit"
    GOOGLE_FITNESS = "google_fitness"


DEFAULT_APPLE_HEALTHKIT_NATIVE_ROOT = Path(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/third-party/EV_cleaned_apple_healthkit"
)
DEFAULT_APPLE_HEALTHKIT_CURATED_ROOT = Path(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/third-party/EV_curated_apple_healthkit"
)
DEFAULT_GOOGLE_FITNESS_NATIVE_ROOT = Path(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/third-party/EV_cleaned_google_fitness"
)
DEFAULT_GOOGLE_FITNESS_CURATED_ROOT = Path(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/third-party/EV_curated_google_fitness"
)

# The existing public Apple loader catalog is a compatibility API, not the complete Apple processing registry.
# Keep it explicit and test it against COHORT_OBSERVED_FEATURES so adding a processing feature cannot silently add
# a public DataLoader.
APPLE_HEALTHKIT_PUBLIC_FEATURES: tuple[str, ...] = (
    "ActiveEnergyBurned", "ActivitySummary", "BMI", "BasalEnergyBurned", "BloodAlcoholContent",
    "BloodGlucose", "BloodPressure", "BodyFatPercentage", "BodyTemperature", "Carbohydrates",
    "DailyDistanceCycling", "DailyDistanceSwimming", "DistanceWalkingRunning", "Electrocardiogram",
    "EnergyConsumed", "FlightsClimbed", "HeartRate", "HeartRateVariability", "Height", "LeanBodyMass",
    "Mindful", "OxygenSaturation", "PeakFlow", "Protein", "RespiratoryRate", "RestingHeartRate", "Sleep",
    "StepCount", "TotalFat", "Vo2Max", "WaistCircumference", "WalkingHeartRate", "Weight",
)


GOOGLE_FITNESS_PUBLIC_FEATURES: tuple[str, ...] = (
    "Aerobics", "Basketball", "Biking", "BikingHand", "BikingMountain", "BikingRoad",
    "BikingSpinning", "BikingStationary", "BikingUtility", "Boxing", "Calisthenics", "Calories",
    "CircuitTraining", "Crossfit", "Dancing", "Distance", "Elliptical", "Ergometer", "Gardening",
    "GuidedBreathing", "Gymnastics", "Handball", "Height", "Hiking", "HorsebackRiding", "InVehicle",
    "IntervalTraining", "IntervalTrainingHighIntensity", "JumpRope", "Kayaking", "KickScooter",
    "Kickboxing", "MartialArts", "Meditation", "OtherActivity", "Pilates", "RockClimbing", "Rowing",
    "RowingMachine", "Running", "RunningJogging", "RunningSand", "RunningTreadmill", "SkatingCross",
    "SkiingCrossCountry", "Sleep", "StairClimbing", "StairClimbingMachine", "StepCount", "Still",
    "StrengthTraining", "Swimming", "SwimmingOpenWater", "SwimmingPool", "Tennis", "UnknownActivity",
    "Volleyball", "VolleyballBeach", "VolleyballIndoor", "Walking", "WalkingFitness", "WalkingNordic",
    "WalkingTreadmill", "Weight", "Weightlifting", "Yoga", "Zumba",
)

_SOURCE_ALIASES: dict[str, WearableSource] = {
    "apple": WearableSource.APPLE_HEALTHKIT,
    "applehealth": WearableSource.APPLE_HEALTHKIT,
    "applehealthkit": WearableSource.APPLE_HEALTHKIT,
    "apple_health": WearableSource.APPLE_HEALTHKIT,
    "apple_healthkit": WearableSource.APPLE_HEALTHKIT,
    "google": WearableSource.GOOGLE_FITNESS,
    "googlefit": WearableSource.GOOGLE_FITNESS,
    "google_fit": WearableSource.GOOGLE_FITNESS,
    "googlefitness": WearableSource.GOOGLE_FITNESS,
    "google_fitness": WearableSource.GOOGLE_FITNESS,
}


def _source_token(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_source(
    value: WearableSource | str | None, *, default: WearableSource | str | None = None,
) -> WearableSource:
    """Return a canonical source enum, applying ``default`` only when the caller omitted ``value``."""

    selected: WearableSource | str | None = default if value is None else value
    if isinstance(selected, WearableSource):
        return selected
    if selected is None:
        raise DataLoaderConfigurationError(
            "source is required for this loader; choose 'apple_healthkit' or 'google_fitness'"
        )
    token = _source_token(selected)
    canonical = _SOURCE_ALIASES.get(token)
    if canonical is None:
        choices = ", ".join(repr(item.value) for item in WearableSource)
        raise DataLoaderConfigurationError(f"unknown wearable source {selected!r}; choose one of {choices}")
    return canonical


@lru_cache(maxsize=1)
def _apple_native_features() -> tuple[str, ...]:
    # Deferred through the Apple adapter package so this source-neutral router never imports a scientific
    # processing registry directly.
    from wearable_project.DataLoaders.apple_healthkit.contracts import native_features

    return native_features()


def _google_features() -> tuple[str, ...]:
    # Checked against the processing and curation registries by release tests. Keeping the public catalog here
    # lets construction and discovery remain independent of the processing package's intentionally broad
    # ``__init__`` imports.
    return GOOGLE_FITNESS_PUBLIC_FEATURES


def public_features_for_source(source: WearableSource | str) -> tuple[str, ...]:
    """Accepted feature catalog exposed by the source-aware public DataLoader API."""

    canonical = normalize_source(source)
    if canonical is WearableSource.APPLE_HEALTHKIT:
        return APPLE_HEALTHKIT_PUBLIC_FEATURES
    return _google_features()


def supported_features_for_source(source: WearableSource | str, phase: str) -> tuple[str, ...]:
    """Features the source adapter can validly represent in ``phase``."""

    canonical = normalize_source(source)
    normalized_phase = str(phase).strip().lower()
    if normalized_phase not in {"native", "curated"}:
        raise DataLoaderConfigurationError(f"unknown processing phase {phase!r}")
    if canonical is WearableSource.APPLE_HEALTHKIT:
        # Native processing supports more types than the accepted Apple curation/public-loader catalog.
        return _apple_native_features() if normalized_phase == "native" else APPLE_HEALTHKIT_PUBLIC_FEATURES
    return _google_features()


def features_for_source(source: WearableSource | str) -> tuple[str, ...]:
    """Convenience name for the source-qualified public feature catalog."""

    return public_features_for_source(source)


def sources_for_feature(feature: str, *, phase: str | None = None) -> tuple[WearableSource, ...]:
    """Every source whose public catalog, or phase-specific support catalog, contains ``feature``."""

    name = str(feature).strip()
    return tuple(
        source for source in WearableSource
        if name in (
            public_features_for_source(source) if phase is None
            else supported_features_for_source(source, phase)
        )
    )


def get_source_adapter(source: WearableSource | str) -> SourceAdapter:
    """Construct the adapter for ``source`` without importing the other source's implementation."""

    canonical = normalize_source(source)
    if canonical is WearableSource.APPLE_HEALTHKIT:
        from wearable_project.DataLoaders.apple_healthkit.adapter import AppleHealthKitAdapter

        return AppleHealthKitAdapter()
    from wearable_project.DataLoaders.google_fitness.adapter import GoogleFitnessAdapter

    return GoogleFitnessAdapter()


_STATE_ACTIVE_SUFFIXES = ("-wal", "-journal")


def _active_state_sidecar(database: Path) -> Path | None:
    for suffix in _STATE_ACTIVE_SUFFIXES:
        candidate = Path(f"{database}{suffix}")
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        except OSError:
            # The state database is managed evidence. If its sidecar cannot be inspected, identity is not safe.
            return candidate
    return None


def _read_state_identity(
    database: Path,
) -> tuple[set[str], dict[str, frozenset[str]], dict[str, str], str | None]:
    """Read table/column structure and key/value metadata without creating SQLite side files."""

    active = _active_state_sidecar(database)
    if active is not None:
        return set(), {}, {}, f"active state sidecar has uncommitted writes: {active}"

    uri = f"{database.resolve().as_uri()}?immutable=1"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            columns_by_table = {
                table: frozenset(
                    str(row[1])
                    for row in connection.execute(
                        f'PRAGMA table_info("{table.replace(chr(34), chr(34) * 2)}")'
                    ).fetchall()
                )
                for table in tables
            }
            metadata: dict[str, str] = {}
            if "metadata" in tables:
                columns = columns_by_table["metadata"]
                if not {"key", "value"} <= columns:
                    return tables, columns_by_table, {}, "metadata table lacks required key/value columns"
                for raw_key, raw_value in connection.execute("SELECT key, value FROM metadata").fetchall():
                    if raw_key is None or not str(raw_key).strip():
                        return tables, columns_by_table, {}, "metadata table contains a blank key"
                    key = str(raw_key)
                    if key in metadata:
                        return tables, columns_by_table, {}, f"metadata table repeats key {key!r}"
                    if raw_value is None:
                        return tables, columns_by_table, {}, f"metadata key {key!r} has a NULL value"
                    metadata[key] = str(raw_value)
            return tables, columns_by_table, metadata, None
    except sqlite3.Error as exc:
        return set(), {}, {}, str(exc)


def _schema_source_signature(
    tables: set[str], columns_by_table: dict[str, frozenset[str]], phase: str,
) -> tuple[WearableSource | None, str | None]:
    """Establish source from genuinely source-unique state structure; reject contradictory signatures."""

    signatures: list[tuple[WearableSource, str]] = []
    if phase == "native":
        if {"feature_outputs", "run_tracking", "run_diagnostics"} <= tables:
            signatures.append((WearableSource.APPLE_HEALTHKIT, "legacy Apple native state schema signature"))
        if {"feature_outputs", "source_occurrences", "malformed_candidates"} <= tables:
            signatures.append((WearableSource.GOOGLE_FITNESS, "Google native state schema signature"))
    elif phase == "curated":
        output_columns = columns_by_table.get("curation_outputs", frozenset())
        apple_unique = {
            "ambiguous_unit_rows", "scale_transition_rows", "raw_unit_counts_json",
            "canonical_unit_counts_json", "unknown_category_counts_json",
        }
        google_unique = {
            "active_materialized_columns_json", "derivation_counts_json", "tracking_counts_json",
            "step_count_role_counts_json",
        }
        if "curation_unit_epochs" in tables or apple_unique <= output_columns:
            signatures.append((WearableSource.APPLE_HEALTHKIT, "legacy Apple curated state schema signature"))
        if google_unique <= output_columns:
            signatures.append((WearableSource.GOOGLE_FITNESS, "Google curated state schema signature"))
    sources = {source for source, _ in signatures}
    if len(sources) > 1:
        descriptions = "; ".join(description for _, description in signatures)
        return None, f"state structure matches conflicting source signatures: {descriptions}"
    return signatures[0] if signatures else (None, None)


def probe_root_identity(root: Path, phase: str | None) -> RootIdentity:
    """
    Establish source identity from managed-root state metadata/schema without reading feature CSVs.
    ``phase`` should come from the state filename. When no state marker exists, the root is an unmanaged trimmed
    tree and no source claim is made. Ambiguous or malformed markers are rejected by the loader before this probe.
    """

    if phase not in {"native", "curated"}:
        return RootIdentity(
            source=None, phase=phase, managed=False, verified=False,
            evidence="no unique native/curated state marker",
        )
    filename = ".wearable_state.sqlite" if phase == "native" else ".wearable_curation_state.sqlite"
    database = root / filename
    if not database.is_file():
        return RootIdentity(
            source=None, phase=phase, managed=False, verified=False,
            evidence=f"{filename} absent",
        )

    tables, columns_by_table, metadata, error = _read_state_identity(database)
    if error is not None:
        return RootIdentity(
            source=None, phase=phase, managed=True, verified=False,
            evidence=f"state identity unreadable: {error}", state_database=database,
        )

    schema_source, schema_evidence = _schema_source_signature(tables, columns_by_table, phase)
    if schema_source is None and schema_evidence is not None:
        return RootIdentity(
            source=None, phase=phase, managed=True, verified=False,
            evidence=schema_evidence, state_database=database, metadata=metadata, tables=tuple(sorted(tables)),
        )

    declared = metadata.get("source_key")
    if declared is not None:
        try:
            source = normalize_source(declared)
        except DataLoaderConfigurationError:
            return RootIdentity(
                source=None, phase=phase, managed=True, verified=False,
                evidence=f"unrecognized state source_key={declared!r}", state_database=database,
                metadata=metadata, tables=tuple(sorted(tables)),
            )
        if schema_source is not None and schema_source is not source:
            return RootIdentity(
                source=None, phase=phase, managed=True, verified=False,
                evidence=(
                    f"state metadata source_key={declared!r} disagrees with {schema_evidence}"
                ),
                state_database=database, metadata=metadata, tables=tuple(sorted(tables)),
            )
        evidence = f"state metadata source_key={declared!r}"
        if schema_evidence is not None:
            evidence += f" corroborated by {schema_evidence}"
        return RootIdentity(
            source=source, phase=phase, managed=True, verified=True, evidence=evidence,
            state_database=database, metadata=metadata, tables=tuple(sorted(tables)),
        )

    if schema_source is None:
        return RootIdentity(
            source=None, phase=phase, managed=True, verified=False,
            evidence="state schema has no recognized source signature", state_database=database,
            metadata=metadata, tables=tuple(sorted(tables)),
        )
    return RootIdentity(
        source=schema_source, phase=phase, managed=True, verified=True, evidence=str(schema_evidence),
        state_database=database, metadata=metadata, tables=tuple(sorted(tables)),
    )


__all__ = [
    "APPLE_HEALTHKIT_PUBLIC_FEATURES",
    "DEFAULT_APPLE_HEALTHKIT_CURATED_ROOT",
    "DEFAULT_APPLE_HEALTHKIT_NATIVE_ROOT",
    "DEFAULT_GOOGLE_FITNESS_CURATED_ROOT",
    "DEFAULT_GOOGLE_FITNESS_NATIVE_ROOT",
    "GOOGLE_FITNESS_PUBLIC_FEATURES",
    "WearableSource",
    "features_for_source",
    "get_source_adapter",
    "normalize_source",
    "probe_root_identity",
    "public_features_for_source",
    "sources_for_feature",
    "supported_features_for_source",
]
