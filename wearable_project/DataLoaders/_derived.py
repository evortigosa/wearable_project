"""
Wearable Data Processing and Modeling project
Derived columns for DataLoader results: local wall-clock time and harmonized values.
Both are computed from columns a load already returned and are appended as new columns; nothing stored is
overwritten. In particular ``canonical_value`` is never filled, because its absence records that a unit was
not resolved. Each function takes a ``LoaderData`` and returns a new one whose ``df_columns_metadata``
describes the added columns and whose ``load_report["derived"]`` records how they were produced.
A derived value is only as sound as its inputs, so each function checks ``load_report["columns_available"]``:
an input the feature never stores means the derivation does not apply, while an input the request left out
(through a projection or ``columns=``) raises rather than silently producing a less careful result.
"""


from __future__ import annotations
from collections.abc import Sequence
from dataclasses import replace
from typing import Any
import numpy as np
import pandas as pd
from wearable_project.exceptions import DataLoaderConfigurationError
from wearable_project.DataLoaders._units import stored_unit


LOCAL_SUFFIX = "_local"
LOCAL_DEFAULT_COLUMNS = ("start_date", "end_date")
OFFSET_COLUMN = "utc_offset_minutes"

HARMONIZED_VALUE = "harmonized_value"
HARMONIZED_UNIT = "harmonized_unit"
HARMONIZED_SOURCE = "harmonized_unit_source"
# Fixed categories, so the source column concatenates across separate calls without falling back to text.
SOURCE_CATEGORIES = pd.CategoricalDtype(["curation", "processing", "registry", "unresolved"])
_CURATION_UNIT_INPUTS = ("canonical_value", "canonical_unit", "curation_unit_status")
_NATIVE_UNIT_INPUTS = ("canonical_value", "canonical_unit")


_GOOGLE_NATIVE_UNIT_CANDIDATES: dict[str, tuple[str, ...]] = {
    # These are official-type candidates recorded by curation, not accepted exporter mappings. They are
    # reported only as unresolved evidence; no candidate is assigned to harmonized_unit.
    "Calories": ("kcal",),
    "Distance": ("m",),
    "Height": ("m",),
    "Weight": ("kg",),
}


def _wearable_source(data: Any) -> str:
    """Source recorded by get_data; legacy LoaderData without the additive field remains Apple-compatible."""

    return str(data.load_report.get("wearable_source", "apple_healthkit"))


def _require(data: Any, needed: Sequence[str], purpose: str) -> None:
    """Raise when an input the stored files carry is missing from this result."""

    available = set(data.load_report.get("columns_available", data.df.columns))
    omitted = [name for name in needed if name in available and name not in data.df.columns]
    if omitted:
        raise DataLoaderConfigurationError(
            f"{purpose} needs {', '.join(omitted)}, which this result omits; reload with the default "
            "projection, or add them to columns="
        )


def _metadata_rows(frame: pd.DataFrame, descriptions: dict[str, str], registry_units: dict[str, Any]) -> pd.DataFrame:
    records = [{
        "column": name, "role": "derived", "description": description,
        "dtype": str(frame[name].dtype), "non_null": int(frame[name].notna().sum()),
        "registry_unit": registry_units.get(name), "dense_default": None,
    } for name, description in descriptions.items()]
    return pd.DataFrame.from_records(records).set_index("column")


def _with_columns(data: Any, frame: pd.DataFrame, descriptions: dict[str, str],
                  registry_units: dict[str, Any], report_key: str, report: dict[str, Any]) -> Any:
    existing = data.df_columns_metadata.drop(index=[n for n in descriptions if n in data.df_columns_metadata.index])
    # Recomputing a column keeps its place in the frame, so the metadata is realigned to the frame's column order
    # rather than left with the recomputed rows at the end; df_columns_metadata always describes df column by column.
    metadata = pd.concat([existing, _metadata_rows(frame, descriptions, registry_units)])
    metadata = metadata.reindex(pd.Index(frame.columns, name="column"))
    load_report = dict(data.load_report)
    load_report["derived"] = {**load_report.get("derived", {}), report_key: report}
    return replace(data, df=frame, df_columns_metadata=metadata, load_report=load_report)


def with_local_time(data: Any, columns: Sequence[str] | None = None) -> Any:
    """
    Add local wall-clock versions of event timestamps, named ``<column>_local``.
    Stored timestamps are UTC. Each row's ``utc_offset_minutes`` (local minus UTC) recovers the local clock,
    which is what diurnal analyses such as hour-of-day profiles need. The added columns are timezone-naive
    because a single pandas column cannot hold the per-row offsets that daylight saving time produces; the UTC
    columns remain for anything that needs the absolute instant. Each row's offset applies to both its start
    and end, so an interval that crosses a clock change keeps the offset of its start.
    ``columns`` defaults to whichever of ``start_date`` and ``end_date`` the result holds. ``datetime`` is
    refused: it is the export's UTC day key rather than an event time, and has no local time.
    """

    feature = data.load_report.get("feature", "this feature")
    available = set(data.load_report.get("columns_available", data.df.columns))
    if OFFSET_COLUMN not in available:
        raise DataLoaderConfigurationError(
            f"{feature} stores no {OFFSET_COLUMN}, so local time cannot be recovered"
        )
    requested = list(columns) if columns is not None else [c for c in LOCAL_DEFAULT_COLUMNS if c in data.df.columns]
    if isinstance(columns, str):
        requested = [columns]
    if not requested:
        missing = "start_date or end_date" + (f", and {OFFSET_COLUMN}" if OFFSET_COLUMN not in data.df.columns else "")
        raise DataLoaderConfigurationError(
            f"local time needs {missing}, which this result omits; reload with the default projection, or add "
            "them to columns="
        )
    _require(data, [OFFSET_COLUMN], "local time")
    for name in requested:
        if name == "datetime":
            raise DataLoaderConfigurationError(
                "datetime is the export's UTC day key rather than an event time, so it has no local time"
            )
        if name not in data.df.columns:
            raise DataLoaderConfigurationError(f"{name!r} is not a column of this result")
        if not isinstance(data.df[name].dtype, pd.DatetimeTZDtype):
            raise DataLoaderConfigurationError(f"{name!r} is not a timezone-aware timestamp column")

    frame = data.df.copy()
    offsets = pd.to_timedelta(frame[OFFSET_COLUMN], unit="min")
    descriptions = {}
    for name in requested:
        target = f"{name}{LOCAL_SUFFIX}"
        frame[target] = (frame[name] + offsets).dt.tz_localize(None)
        descriptions[target] = (
            f"Local wall-clock {name}: the UTC value shifted by the row's {OFFSET_COLUMN}; timezone-naive."
        )
    return _with_columns(data, frame, descriptions, {}, "local_time", {"columns": list(descriptions)})


def _with_google_harmonized_values(data: Any, feature: str, phase: str) -> Any:
    """Google harmonization under the source-qualified native or curated contract."""

    available = set(data.load_report.get("columns_available", data.df.columns))
    if "value" not in available:
        raise DataLoaderConfigurationError(
            f"harmonized values apply to features with one numeric value column; {feature} uses a different "
            "measurement contract"
        )
    inputs = ["value"]
    if phase == "curated":
        inputs.extend(("canonical_value", "canonical_unit", "curation_unit_status"))
    _require(data, inputs, "harmonization")
    values = data.df["value"]
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        raise DataLoaderConfigurationError(
            f"harmonized values apply to numeric values; {feature}'s value holds {values.dtype}"
        )

    frame = data.df.copy()
    size = len(frame)
    fixed = "count" if feature == "StepCount" else None
    candidates = _GOOGLE_NATIVE_UNIT_CANDIDATES.get(feature, ())
    harmonized = np.full(size, np.nan, dtype=float)
    units = np.full(size, None, dtype=object)
    sources = np.full(size, "unresolved", dtype=object)
    status_counts: dict[str, int] = {}

    if phase == "native":
        if fixed is not None:
            harmonized[:] = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
            units[:] = fixed
            sources[:] = "registry"
    elif phase == "curated":
        canonical = frame["canonical_value"]
        canonical_unit = frame["canonical_unit"].astype(object)
        verdict = frame["curation_unit_status"].astype("string")
        status_counts = {
            str(name): int(count)
            for name, count in verdict.fillna("unresolved").value_counts().items()
        }
        resolved = verdict.str.startswith("resolved", na=False).to_numpy(dtype=bool)
        has_canonical = (canonical.notna() & canonical_unit.notna()).to_numpy(dtype=bool)
        by_curation = resolved & has_canonical
        if bool(by_curation.any()):
            harmonized[by_curation] = pd.to_numeric(canonical, errors="coerce").to_numpy(dtype=float)[by_curation]
            units[by_curation] = canonical_unit.to_numpy(dtype=object)[by_curation]
            sources[by_curation] = "curation"
    else:  # pragma: no cover - constructor validates phase
        raise DataLoaderConfigurationError(f"unknown Google Fitness phase {phase!r}")

    frame[HARMONIZED_VALUE] = pd.Series(harmonized, index=frame.index, dtype="float64")
    frame[HARMONIZED_UNIT] = pd.Series(pd.Categorical(units), index=frame.index)
    frame[HARMONIZED_SOURCE] = pd.Series(
        pd.Categorical(sources, dtype=SOURCE_CATEGORIES), index=frame.index,
    )
    counts = {name: int(count) for name, count in pd.Series(sources).value_counts().items()}
    descriptions = {
        HARMONIZED_VALUE: "The value in a trusted unit, or missing where no layer established one.",
        HARMONIZED_UNIT: "The unit of harmonized_value.",
        HARMONIZED_SOURCE: "The layer that established the unit: curation, processing, registry, or unresolved.",
    }
    report = {
        "sources": dict(sorted(counts.items())),
        "registry_unit": fixed,
        "unit_withheld_by_curation": bool(candidates) if phase == "native" else any(
            name.startswith("withheld") for name in status_counts
        ),
        "unit_candidates": list(candidates),
    }
    if phase == "curated":
        report["curation_unit_status_counts"] = dict(sorted(status_counts.items()))
    return _with_columns(
        data, frame, descriptions, {HARMONIZED_VALUE: fixed}, "harmonized_values", report,
    )


def with_harmonized_values(data: Any) -> Any:
    """
    Add each row's value in a trusted unit: ``harmonized_value``, ``harmonized_unit``, and
    ``harmonized_unit_source`` recording which layer established the unit.
    Precedence, row by row:
    - ``curation``: in the curated phase a curation unit verdict decides. A resolved verdict with a stored
      canonical value gives that value and unit; any other verdict, such as ``ambiguous``, leaves the row
      unresolved even where the registry names a unit, so curation's withholding of EnergyConsumed's kcal
      label is respected.
    - ``processing``: without a verdict, a canonical value that native processing stored, such as a fraction
      converted to percent.
    - ``registry``: without either, the processing registry's unit, where it fixes a single unit and the
      stored value is therefore already canonical, unless the curation registry declares the stored unit
      unknown. The unit is resolved by ``DataLoaders._units.stored_unit``, the one place both registries meet.
    - ``unresolved``: otherwise, with a missing value and unit.
    Applies to features with a numeric ``value`` column. The native phase has no curation verdicts, so its rows
    rely on the registry path, which still honors the curation registry's declaration: EnergyConsumed, which
    curation declares kcal-or-kJ, is unresolved in both phases while ``value`` and ``raw_unit`` are kept.
    """

    feature = data.load_report.get("feature")
    phase = data.load_report.get("phase")
    if feature is None or phase is None:
        raise DataLoaderConfigurationError(
            "harmonized values need the feature and phase recorded by get_data(); this LoaderData carries none"
        )
    source = _wearable_source(data)
    if source == "google_fitness":
        return _with_google_harmonized_values(data, str(feature), str(phase))
    if source != "apple_healthkit":
        raise DataLoaderConfigurationError(
            f"harmonized values do not support wearable source {source!r}"
        )
    available = set(data.load_report.get("columns_available", data.df.columns))
    if "value" not in available:
        raise DataLoaderConfigurationError(
            f"harmonized values apply to features with a numeric value column; {feature} has none"
        )
    inputs = _CURATION_UNIT_INPUTS if phase == "curated" else _NATIVE_UNIT_INPUTS
    _require(data, ["value", *inputs], "harmonization")
    values = data.df["value"]
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        raise DataLoaderConfigurationError(
            f"harmonized values apply to numeric values; {feature}'s value holds {values.dtype}"
        )

    established = stored_unit(feature, "value")
    fixed = established.unit if established.already_canonical else None
    frame = data.df.copy()
    size = len(frame)
    canonical = frame["canonical_value"] if "canonical_value" in frame else pd.Series(np.nan, index=frame.index)
    canonical_unit = (frame["canonical_unit"].astype(object) if "canonical_unit" in frame
                      else pd.Series([None] * size, index=frame.index, dtype=object))
    has_canonical = (canonical.notna() & canonical_unit.notna()).to_numpy(dtype=bool)
    if phase == "curated" and "curation_unit_status" in frame:
        verdict = frame["curation_unit_status"].astype(object)
        has_verdict = verdict.notna().to_numpy(dtype=bool)
        resolved = has_verdict & verdict.astype(str).str.startswith("resolved").to_numpy(dtype=bool)
    else:
        has_verdict = np.zeros(size, dtype=bool)
        resolved = np.zeros(size, dtype=bool)

    by_curation = resolved & has_canonical
    by_processing = ~has_verdict & has_canonical
    by_registry = ~has_verdict & ~has_canonical & (fixed is not None)
    from_canonical = by_curation | by_processing

    source = np.full(size, "unresolved", dtype=object)
    source[by_curation], source[by_processing], source[by_registry] = "curation", "processing", "registry"
    harmonized = np.full(size, np.nan)
    harmonized[from_canonical] = pd.to_numeric(canonical, errors="coerce").to_numpy(dtype=float)[from_canonical]
    harmonized[by_registry] = values.to_numpy(dtype=float)[by_registry]
    units = np.full(size, None, dtype=object)
    units[from_canonical] = canonical_unit.to_numpy()[from_canonical]
    units[by_registry] = fixed

    frame[HARMONIZED_VALUE] = pd.Series(harmonized, index=frame.index, dtype="float64")
    frame[HARMONIZED_UNIT] = pd.Series(pd.Categorical(units), index=frame.index)
    frame[HARMONIZED_SOURCE] = pd.Series(pd.Categorical(source, dtype=SOURCE_CATEGORIES), index=frame.index)
    counts = {name: int(count) for name, count in pd.Series(source).value_counts().items()}
    descriptions = {
        HARMONIZED_VALUE: "The value in a trusted unit, or missing where no layer established one.",
        HARMONIZED_UNIT: "The unit of harmonized_value.",
        HARMONIZED_SOURCE: "The layer that established the unit: curation, processing, registry, or unresolved.",
    }
    return _with_columns(
        data, frame, descriptions, {HARMONIZED_VALUE: fixed}, "harmonized_values",
        {"sources": dict(sorted(counts.items())), "registry_unit": fixed,
         "unit_withheld_by_curation": established.withheld_by_curation,
         "unit_candidates": list(established.candidates)},
    )
