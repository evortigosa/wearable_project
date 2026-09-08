"""
Wearable Data Processing and Modeling project
Feature aggregation policies.

HealthKit quantity types do not all have the same aggregation semantics. Cumulative
quantities (for example, steps or distance) are allocated proportionally and summed.
Discrete quantities are summarized without dividing their values across windows.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal, Mapping
from ..exceptions import ConfigurationError

Aggregation= Literal["sum", "weighted_mean", "mean", "median", "mode"]


@dataclass(frozen=True, slots=True)
class FeaturePolicy:
    """
    Rules used when a feature is placed into fixed-width windows.
    Parameters
    ----------
    aggregation:
        Aggregation applied to each value column inside a window.
    cumulative:
        Whether a numeric value represents a total over its source interval. A cumulative value is allocated
        in proportion to temporal overlap before it is summed, which conserves the original event total.
    windowable:
        Whether the feature can be safely represented in fixed-width windows.
    """
    aggregation:Aggregation= "weighted_mean"
    cumulative:bool= False
    windowable:bool= True

    def __post_init__(self) -> None:
        allowed= {"sum", "weighted_mean", "mean", "median", "mode"}
        if self.aggregation not in allowed:
            raise ConfigurationError(
                f"Unsupported aggregation {self.aggregation!r}; expected one of {sorted(allowed)}."
            )
        if self.cumulative and self.aggregation != "sum":
            raise ConfigurationError("Cumulative features must use aggregation='sum'.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Canonical names plus aliases observed in common export pipelines. The registry is
# intentionally overrideable because exporters may rename HealthKit identifiers.
_CUMULATIVE_FEATURES= {
    "activeenergyburned", "basalenergyburned", "carbohydrates", "dailydistancecycling", "dailydistanceswimming",
    "dietarycarbohydrates", "dietaryenergyconsumed", "dietaryfattotal", "dietaryprotein", "distancecycling",
    "distanceswimming", "distancewalkingrunning", "energyconsumed", "flightsclimbed", "inhalerusage",
    "protein", "stepcount", "totalfat",
}

# Sparse point measurements, complex records, and categorical sleep intervals are
# preserved at native resolution unless a study-approved override is supplied. Sleep
# is intentionally explicit here: the cohort can contain INBED, ASLEEP, DEEP, and REM,
# and overlapping hierarchical labels must not be collapsed by a generic mode rule.
_NON_WINDOWABLE_FEATURES= {
    "activitysummary", "bmi", "bodyfatpercentage", "bodymass", "bodymassindex", "electrocardiogram", "ecg",
    "height", "leanbodymass", "sleep", "sleepanalysis", "waistcircumference", "weight",
}

# These are measurements or ratios, not interval totals. Listing them explicitly
# prevents accidental treatment as cumulative if an old configuration is reused.
_DISCRETE_FEATURES= {
    "bloodalcoholcontent", "bloodglucose", "bloodpressure", "bodytemperature", "heartrate",
    "heartratevariability", "heartratevariabilitysdnn", "oxygensaturation", "peakflow", "respiratoryrate",
    "restingheartrate", "vo2max", "walkingheartrate", "walkingheartrateaverage",
}


def canonical_feature_name(name:str) -> str:
    """ Return a comparison key insensitive to punctuation and case. """

    return "".join(ch for ch in str(name).casefold() if ch.isalnum())


def default_feature_policy(feature_name:str) -> FeaturePolicy:
    """ Resolve the built-in policy for a feature name. """

    key= canonical_feature_name(feature_name)
    if key in _CUMULATIVE_FEATURES:
        return FeaturePolicy(aggregation="sum", cumulative=True, windowable=True)
    if key in _NON_WINDOWABLE_FEATURES:
        return FeaturePolicy(aggregation="weighted_mean", cumulative=False, windowable=False)
    if key in _DISCRETE_FEATURES:
        return FeaturePolicy(aggregation="weighted_mean", cumulative=False, windowable=True)
    # Unknown features are preserved as events.  Their HealthKit aggregation style
    # cannot be inferred safely from the export label alone, so averaging or summing
    # them would be an undocumented scientific transformation.
    return FeaturePolicy(aggregation="weighted_mean", cumulative=False, windowable=False)


def feature_policy_origin(
    feature_name:str, overrides:Mapping[str, FeaturePolicy | Mapping[str, Any]] | None= None,
) -> Literal["override", "built_in", "unknown_conservative"]:
    """
    Describe why a feature received its policy.
    Quality reports expose this value so researchers can identify feature names that still need an explicit,
    study-approved policy rather than silently accepting an inferred aggregation rule.
    """

    key= canonical_feature_name(feature_name)
    if key in normalize_policy_mapping(overrides):
        return "override"
    if key in _CUMULATIVE_FEATURES | _NON_WINDOWABLE_FEATURES | _DISCRETE_FEATURES:
        return "built_in"
    return "unknown_conservative"


def _coerce_policy(value:FeaturePolicy | Mapping[str, Any]) -> FeaturePolicy:
    if isinstance(value, FeaturePolicy):
        return value
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Feature policy must be an object, got {type(value).__name__}.")
    allowed= {"aggregation", "cumulative", "windowable"}
    unknown= set(value).difference(allowed)
    if unknown:
        raise ConfigurationError(f"Unknown feature-policy fields: {sorted(unknown)}.")
    for field_name in ("cumulative", "windowable"):
        if field_name in value and not isinstance(value[field_name], bool):
            raise ConfigurationError(
                f"Feature policy field {field_name!r} must be a JSON boolean, not {type(value[field_name]).__name__}."
            )
    return FeaturePolicy(
        aggregation=value.get("aggregation", "weighted_mean"), cumulative=value.get("cumulative", False),
        windowable=value.get("windowable", True),
    )


def normalize_policy_mapping(
    policies:Mapping[str, FeaturePolicy | Mapping[str, Any]] | None,
) -> dict[str, FeaturePolicy]:
    """ Normalize override keys and values for fast, deterministic lookup. """

    if not policies:
        return {}
    normalized:dict[str, FeaturePolicy]= {}
    original_names:dict[str, str]= {}
    for name, value in policies.items():
        key= canonical_feature_name(name)
        if not key:
            raise ConfigurationError("Feature policy names cannot be empty.")
        policy= _coerce_policy(value)
        if key in normalized and normalized[key] != policy:
            raise ConfigurationError(
                f"Conflicting policies for {original_names[key]!r} and {name!r}; "
                "their canonical feature names are identical."
            )
        normalized[key]= policy
        original_names[key]= str(name)
    return normalized


def resolve_feature_policy(
    feature_name:str, overrides:Mapping[str, FeaturePolicy | Mapping[str, Any]] | None= None,
) -> FeaturePolicy:
    """ Return an override when present, otherwise the conservative built-in policy. """

    normalized= normalize_policy_mapping(overrides)
    return normalized.get(canonical_feature_name(feature_name), default_feature_policy(feature_name))


def load_feature_policies(path:str | Path | None) -> dict[str, FeaturePolicy]:
    """
    Load feature-policy overrides from a JSON object.
    The file maps feature names to objects containing ``aggregation``, ``cumulative`` and ``windowable`` fields.
    """

    if path is None:
        return {}
    policy_path= Path(path)
    try:
        payload= json.loads(policy_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"Cannot read feature policy file {policy_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in feature policy file {policy_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError("Feature policy JSON must contain a top-level object.")
    return normalize_policy_mapping(payload)


def policies_to_jsonable(
    policies:Mapping[str, FeaturePolicy | Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """ Return policies in a stable representation suitable for manifests. """

    normalized= normalize_policy_mapping(policies)
    return {name: normalized[name].to_dict() for name in sorted(normalized)}
