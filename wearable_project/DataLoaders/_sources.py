"""
Wearable Data Processing and Modeling project
Source identifiers, compatibility aliases, source-qualified roots, feature catalogs, and managed-root probing.
Source selection is never inferred from a feature filename. A caller selects a scientific source contract and
managed-root metadata verifies that choice before a feature CSV is parsed. Unmanaged trimmed trees remain
loadable, but are explicitly reported as source-unverified.
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
    # Deferred so the public catalog remains cheap and independent from curation imports.
    from wearable_project.processing.apple_healthkit.registry import known_features

    return tuple(known_features())


@lru_cache(maxsize=1)
def _google_features() -> tuple[str, ...]:
    # Deferred so importing an Apple feature loader does not import the Google processing stack.
    from wearable_project.processing.google_fitness.registry import feature_names

    return tuple(feature_names())


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


def _read_state_identity(database: Path) -> tuple[set[str], dict[str, str], str | None]:
    """Read only table names and key/value metadata from a state database; never create SQLite side files."""

    uri = f"{database.resolve().as_uri()}?immutable=1"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            metadata: dict[str, str] = {}
            if "metadata" in tables:
                columns = {
                    str(row[1]) for row in connection.execute("PRAGMA table_info(metadata)").fetchall()
                }
                if {"key", "value"} <= columns:
                    metadata = {
                        str(key): str(value)
                        for key, value in connection.execute("SELECT key, value FROM metadata").fetchall()
                    }
            return tables, metadata, None
    except sqlite3.Error as exc:
        return set(), {}, str(exc)


def probe_root_identity(root: Path, phase: str | None) -> RootIdentity:
    """
    Establish source identity from managed-root state metadata/schema without reading feature CSVs.
    ``phase`` should come from the state filename.  When no unique phase marker exists, the root is treated as
    unmanaged/ambiguous and no source claim is made, preserving the accepted behavior for trimmed sample trees.
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

    tables, metadata, error = _read_state_identity(database)
    if error is not None:
        return RootIdentity(
            source=None, phase=phase, managed=True, verified=False,
            evidence=f"state identity unreadable: {error}", state_database=database,
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
        return RootIdentity(
            source=source, phase=phase, managed=True, verified=True,
            evidence=f"state metadata source_key={declared!r}", state_database=database,
            metadata=metadata, tables=tuple(sorted(tables)),
        )

    # Legacy Apple state predates source_key metadata.  Its source is nevertheless unambiguous from tables that
    # do not occur in the Google state schemas.  Keep the signatures narrow so an unknown future schema is
    # reported unverified instead of guessed.
    if phase == "native" and {"feature_outputs", "run_tracking", "run_diagnostics"} <= tables:
        source = WearableSource.APPLE_HEALTHKIT
        evidence = "legacy Apple native state schema signature"
    elif phase == "curated" and {"curation_outputs", "curation_unit_epochs"} <= tables:
        source = WearableSource.APPLE_HEALTHKIT
        evidence = "legacy Apple curated state schema signature"
    elif phase == "native" and {"feature_outputs", "source_occurrences", "malformed_candidates"} <= tables:
        source = WearableSource.GOOGLE_FITNESS
        evidence = "Google native state schema signature"
    else:
        return RootIdentity(
            source=None, phase=phase, managed=True, verified=False,
            evidence="state schema has no recognized source signature", state_database=database,
            metadata=metadata, tables=tuple(sorted(tables)),
        )

    return RootIdentity(
        source=source, phase=phase, managed=True, verified=True, evidence=evidence,
        state_database=database, metadata=metadata, tables=tuple(sorted(tables)),
    )


__all__ = [
    "APPLE_HEALTHKIT_PUBLIC_FEATURES",
    "DEFAULT_APPLE_HEALTHKIT_CURATED_ROOT",
    "DEFAULT_APPLE_HEALTHKIT_NATIVE_ROOT",
    "DEFAULT_GOOGLE_FITNESS_CURATED_ROOT",
    "DEFAULT_GOOGLE_FITNESS_NATIVE_ROOT",
    "WearableSource",
    "features_for_source",
    "get_source_adapter",
    "normalize_source",
    "probe_root_identity",
    "public_features_for_source",
    "sources_for_feature",
    "supported_features_for_source",
]
