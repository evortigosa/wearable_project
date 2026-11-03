"""
Wearable Data Processing and Modeling project
Apple HealthKit post-load value harmonization.
"""


from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from wearable_project.DataLoaders.apple_healthkit.units import stored_unit
from wearable_project.exceptions import DataLoaderConfigurationError


def with_harmonized_values(data: Any, feature: str, phase: str) -> Any:
    """Add the accepted Apple harmonized-value projection without changing stored or canonical columns."""

    from wearable_project.DataLoaders._derived import (
        HARMONIZED_SOURCE,
        HARMONIZED_UNIT,
        HARMONIZED_VALUE,
        SOURCE_CATEGORIES,
        _CURATION_UNIT_INPUTS,
        _NATIVE_UNIT_INPUTS,
        _require,
        _with_columns,
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
    canonical_unit = (
        frame["canonical_unit"].astype(object)
        if "canonical_unit" in frame
        else pd.Series([None] * size, index=frame.index, dtype=object)
    )
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

    sources = np.full(size, "unresolved", dtype=object)
    sources[by_curation], sources[by_processing], sources[by_registry] = "curation", "processing", "registry"
    harmonized = np.full(size, np.nan)
    harmonized[from_canonical] = pd.to_numeric(canonical, errors="coerce").to_numpy(dtype=float)[from_canonical]
    harmonized[by_registry] = values.to_numpy(dtype=float)[by_registry]
    units = np.full(size, None, dtype=object)
    units[from_canonical] = canonical_unit.to_numpy()[from_canonical]
    units[by_registry] = fixed

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
    return _with_columns(
        data,
        frame,
        descriptions,
        {HARMONIZED_VALUE: fixed},
        "harmonized_values",
        {
            "sources": dict(sorted(counts.items())),
            "registry_unit": fixed,
            "unit_withheld_by_curation": established.withheld_by_curation,
            "unit_candidates": list(established.candidates),
        },
    )
