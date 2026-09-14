"""
Wearable Data Processing and Modeling project
Defines and validates the policy framework only. It does not modify or curate participant feature files.
"""


from .evidence import EVIDENCE, EvidenceSource, evidence_by_kind
from .guidance import (
    FEATURE_GUIDES, FeatureGuide, get_feature_guide, guidance_fingerprint, known_guides,
)
from .models import EvidenceGrade, PolicyExecutionMode
from .registry import (
    COHORT_OBSERVED_FEATURES, CURATION_POLICIES, CURATION_REGISTRY_VERSION, UNKNOWN_POLICY,
    get_policy, known_curation_features, registry_fingerprint, validate_registry,
)

__all__ = [
    "COHORT_OBSERVED_FEATURES", "CURATION_POLICIES", "CURATION_REGISTRY_VERSION", "EVIDENCE",
    "EvidenceGrade", "EvidenceSource", "FEATURE_GUIDES", "FeatureGuide", "PolicyExecutionMode",
    "UNKNOWN_POLICY", "evidence_by_kind", "get_feature_guide", "get_policy", "guidance_fingerprint",
    "known_curation_features", "known_guides", "registry_fingerprint", "validate_registry",
]
