"""
Wearable Data Processing and Modeling project
Google Fitness post-load value harmonization under the accepted source contract.
"""


from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from wearable_project.DataLoaders.google_fitness.units import registry_unit, unresolved_unit_candidates
from wearable_project.exceptions import DataLoaderConfigurationError


def with_harmonized_values(data: Any, feature: str, phase: str) -> Any:
    """Add trusted Google harmonized values, withholding unresolved exporter-unit mappings."""

    from wearable_project.DataLoaders._derived import (
        HARMONIZED_SOURCE,
        HARMONIZED_UNIT,
        HARMONIZED_VALUE,
        SOURCE_CATEGORIES,
        _require,
        _with_columns,
    )

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
    fixed = registry_unit(feature, "value")
    candidates = unresolved_unit_candidates(feature, "value")
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
        data,
        frame,
        descriptions,
        {HARMONIZED_VALUE: fixed},
        "harmonized_values",
        report,
    )
