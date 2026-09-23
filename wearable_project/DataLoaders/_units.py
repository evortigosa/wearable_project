"""
Wearable Data Processing and Modeling project
The stored unit the project establishes for a feature's measurement column.
Two registries describe units. ``processing/registry.py`` declares the unit convention native processing applied;
``curation/registry.py`` declares what the project knows about the stored unit after reviewing evidence. This
module reconciles them in one place, so every consumer gives the same answer: the harmonized values added by
``LoaderData.with_harmonized_values`` and the ``registry_unit`` column of ``df_columns_metadata``.
The reconciliation is deliberately narrow. The processing registry names the unit, and the curation registry can
only withdraw that endorsement, by declaring the stored unit unknown (``raw_unit`` of None, with the candidate
units it could not choose between). Curation never substitutes a unit of its own here, so no unit label changes.
Today the withdrawal matters for one feature: EnergyConsumed, which processing labels kcal while curation cannot
tell kcal from kJ. It applies in both phases, because it is a declaration about the stored data rather than a
per-row curation verdict; the stored ``value`` and ``raw_unit`` are left exactly as written.
Curation measurement declarations are matched to stored columns by name, which covers ``value``. Declarations whose
names differ from the stored columns, such as BloodPressure's ``systolic_pressure``, leave processing's unit in place.
Where processing declares no unit, curation sometimes names one (DailyDistanceCycling in metres, for instance);
adopting those would be a positive endorsement rather than a withdrawal, and is intentionally not done here.
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from wearable_project.processing.registry import get_feature_spec


@dataclass(frozen=True)
class StoredUnit:
    """The established unit of one stored measurement column."""

    unit: str | None
    """Unit the stored values are in, or None where no registry establishes one."""
    canonical_unit: str | None
    """Unit the processing registry harmonizes this measurement to."""
    withheld_by_curation: bool
    """True where the curation registry declares the stored unit unknown."""
    candidates: tuple[str, ...]
    """The units curation could not choose between, when it withholds the unit."""

    @property
    def already_canonical(self) -> bool:
        """Whether stored values are already in the canonical unit, so they need no conversion."""

        return self.unit is not None and self.unit == self.canonical_unit


def _curation_declaration(feature: str, column: str) -> Any | None:
    """The curation registry's unit declaration for ``column``, or None when it declares nothing for it."""

    # Imported here so that importing a loader never pulls the curation package; by the time a result is
    # described or harmonized, get_data has already loaded it.
    from wearable_project.curation.registry import get_policy

    try:
        policy = get_policy(feature, allow_fallback=False)
    except KeyError:
        return None
    for declaration in policy.units.measurements:
        if declaration.measurement == column:
            return declaration
    return None


def stored_unit(feature: str, column: str) -> StoredUnit:
    """Return the unit the registries establish for ``feature``'s stored ``column``."""

    spec = get_feature_spec(feature)
    if column not in spec.measurement_columns:
        return StoredUnit(None, None, False, ())
    declaration = _curation_declaration(feature, column)
    withheld = declaration is not None and declaration.raw_unit is None
    candidates = tuple(declaration.raw_unit_candidates) if withheld else ()
    policy = spec.unit_policy
    if policy is None:
        return StoredUnit(None, None, withheld, candidates)
    if withheld:
        return StoredUnit(None, policy.canonical_unit, True, candidates)
    return StoredUnit(policy.raw_unit, policy.canonical_unit, False, ())
