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
import pandas as pd
from wearable_project.exceptions import DataLoaderConfigurationError


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


def with_harmonized_values(data: Any) -> Any:
    """
    Add source-qualified ``harmonized_value``, ``harmonized_unit``, and ``harmonized_unit_source`` columns.
    The configured source adapter owns the scientific policy. Apple retains its accepted row-wise precedence
    across curation, processing, registry, and unresolved evidence. Google exposes canonical counts for StepCount,
    deliberately withholds the unverified exporter mappings for Calories, Distance, Weight, and Height, and rejects
    singular harmonization for multi-measurement activity rows. Stored and canonical columns are never overwritten.
    """

    feature = data.load_report.get("feature")
    phase = data.load_report.get("phase")
    if feature is None or phase is None:
        raise DataLoaderConfigurationError(
            "harmonized values need the feature and phase recorded by get_data(); this LoaderData carries none"
        )
    source = _wearable_source(data)
    from wearable_project.DataLoaders._sources import get_source_adapter

    try:
        adapter = get_source_adapter(source)
    except DataLoaderConfigurationError as exc:
        raise DataLoaderConfigurationError(
            f"harmonized values do not support wearable source {source!r}"
        ) from exc
    return adapter.with_harmonized_values(data, str(feature), str(phase))
