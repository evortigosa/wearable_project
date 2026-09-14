"""
Wearable Data Processing and Modeling project
Defines and validates the policy framework only. It does not modify or curate participant feature files.
"""


from .registry import (
    COHORT_OBSERVED_FEATURES, CURATION_POLICIES, CURATION_REGISTRY_VERSION, UNKNOWN_POLICY,
    get_policy, known_curation_features, registry_fingerprint, validate_registry,
)

__all__ = [
    "COHORT_OBSERVED_FEATURES", "CURATION_POLICIES", "CURATION_REGISTRY_VERSION", "UNKNOWN_POLICY",
    "get_policy", "known_curation_features", "registry_fingerprint", "validate_registry",
]
