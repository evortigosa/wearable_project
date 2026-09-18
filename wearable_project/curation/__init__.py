"""
Wearable Data Processing and Modeling project
Defines curation policies, guidance, audit and execution engine.
"""


from .evidence import EVIDENCE, EvidenceSource, evidence_by_kind
from .guidance import FEATURE_GUIDES, FeatureGuide, get_feature_guide, guidance_fingerprint, known_guides
from .decisions import CalibrationDecision, decisions_fingerprint, get_decision, known_decisions
from .environment import EnvironmentManifest, environment_manifest
from .models import EvidenceGrade, PolicyExecutionMode
from .registry import (
    COHORT_OBSERVED_FEATURES, CURATION_POLICIES, CURATION_REGISTRY_VERSION, UNKNOWN_POLICY,
    get_policy, known_curation_features, registry_fingerprint, validate_registry,
)
from .pipeline import curate_dataset
from .state import CURATION_ENGINE_VERSION, load_curation_report


__all__ = [
    "COHORT_OBSERVED_FEATURES",
    "CURATION_ENGINE_VERSION",
    "CURATION_POLICIES",
    "CURATION_REGISTRY_VERSION",
    "CalibrationDecision",
    "EVIDENCE",
    "EnvironmentManifest",
    "EvidenceGrade",
    "EvidenceSource",
    "FEATURE_GUIDES",
    "FeatureGuide",
    "PolicyExecutionMode",
    "UNKNOWN_POLICY",
    "curate_dataset",
    "decisions_fingerprint",
    "environment_manifest",
    "evidence_by_kind",
    "get_decision",
    "get_feature_guide",
    "get_policy",
    "guidance_fingerprint",
    "known_curation_features",
    "known_decisions",
    "known_guides",
    "load_curation_report",
    "registry_fingerprint",
    "validate_registry",
]
