"""
Wearable Data Processing and Modeling project
Google Fitness DataLoader unit evidence and established-unit decisions.
"""


from __future__ import annotations


def registry_unit(feature: str, column: str) -> str | None:
    """Return only units established for the exported representation, never proposed official-type mappings."""

    if feature == "StepCount" and column in {"value", "raw_value", "canonical_value"}:
        return "count"
    return None


def unresolved_unit_candidates(feature: str, column: str = "value") -> tuple[str, ...]:
    """Official-type candidate units retained as evidence while exporter mapping remains withheld."""

    from wearable_project.curation.google_fitness.registry import get_policy

    try:
        policy = get_policy(feature, allow_fallback=False)
    except KeyError:
        return ()
    candidates: list[str] = []
    for measurement in policy.measurements:
        if measurement.column != column:
            continue
        for version in measurement.unit_by_version:
            if version.canonical_unit is not None and version.raw_unit is None:
                candidates.append(version.canonical_unit)
    return tuple(dict.fromkeys(candidates))
