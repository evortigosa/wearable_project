"""
Wearable Data Processing and Modeling project
Measurement-level unit annotations for Google Fitness curation. The engine executes only reviewed unit decisions.
StepCount is resolved to counts. Calories, Distance, Weight, Height, and activity submeasurements remain raw
because their cohort-exporter mappings are not proven. No value is converted or guessed.
"""


from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping
from .models import FeaturePolicy


@dataclass(frozen=True, slots=True)
class UnitResolution:
    canonical_value: str | None = None
    canonical_unit: str | None = None
    unit_status: str | None = None
    unit_evidence: str | None = None

    def as_values(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if self.canonical_value not in (None, ""):
            result["canonical_value"] = str(self.canonical_value)
        if self.canonical_unit not in (None, ""):
            result["canonical_unit"] = str(self.canonical_unit)
        if self.unit_status not in (None, ""):
            result["curation_unit_status"] = str(self.unit_status)
        if self.unit_evidence not in (None, ""):
            result["unit_evidence"] = str(self.unit_evidence)
        return result


def _finite_nonnegative_text(value: str | None) -> str | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return text


def resolve_units(feature: str, row: Mapping[str, str], policy: FeaturePolicy) -> UnitResolution | None:
    """Return the reviewed row-level unit annotation, or ``None`` when not applicable."""

    if feature == "StepCount":
        version = (row.get("collecting_method_version") or "").strip()
        canonical = _finite_nonnegative_text(row.get("value"))
        if version == "2.0":
            return UnitResolution(
                canonical_value=canonical,
                canonical_unit="count",
                unit_status="resolved_payload_explicit",
                unit_evidence="payload_explicit:com.google.step_count.delta",
            )
        if version == "1.0":
            return UnitResolution(
                canonical_value=canonical,
                canonical_unit="count",
                unit_status="resolved_source_convention",
                unit_evidence="source_convention:google_fitness_step_count_v1",
            )
        return UnitResolution(
            canonical_value=None,
            canonical_unit="count",
            unit_status="unresolved_collecting_method_version",
            unit_evidence="unsupported_collecting_method_version",
        )

    if feature in {"Calories", "Distance", "Weight", "Height"}:
        proposed = {
            "Calories": "kcal",
            "Distance": "m",
            "Weight": "kg",
            "Height": "m",
        }[feature]
        return UnitResolution(
            canonical_value=None,
            canonical_unit=None,
            unit_status="withheld_exporter_mapping_pending",
            unit_evidence=f"exporter_mapping_pending:proposed_{proposed}",
        )

    # Activity submeasurements remain multiple native fields. A single scalar canonical_value would misrepresent
    # the row, so no global unit columns are emitted for activity and Sleep policies.
    return None
