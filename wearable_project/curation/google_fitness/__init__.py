"""
Wearable Data Processing and Modeling project
Google Fitness curation policy foundation.
"""


from .audit_models import AUDIT_VERSION, REPORT_SCHEMA_VERSION
from .decisions import DECISIONS, CalibrationDecision, decisions_fingerprint, get_decision, known_decisions
from .evidence import EVIDENCE, EvidenceKind, EvidenceSource, evidence_fingerprint, get_evidence, known_evidence
from .guidance import FEATURE_GUIDES, FeatureGuide, get_feature_guide, guidance_fingerprint, known_guides
from .models import CurationStatus, DecisionStatus, EvidenceGrade, PolicyExecutionMode, PolicyMaturity
from .rules import RULES, RuleDefinition, get_rule, known_rules, rules_fingerprint
from .engine import (
    ENGINE_VERSION, CuratedFeatureInfo, FeatureCurationResult, curate_feature_file, feature_policy_fingerprint,
    validate_curated_feature_file, validate_engine_contract,
)
from .registry import (
    CURATION_POLICIES, CURATION_REGISTRY_VERSION, POLICY_CONTRACT_VERSION, UNKNOWN_POLICY, get_policy,
    known_curation_features, registry_fingerprint, registry_payload, validate_registry,
)


__all__ = [
    "AUDIT_VERSION",
    "ENGINE_VERSION",
    "CuratedFeatureInfo",
    "FeatureCurationResult",
    "CURATION_POLICIES",
    "CURATION_REGISTRY_VERSION",
    "CurationStatus",
    "DECISIONS",
    "DecisionStatus",
    "EVIDENCE",
    "EvidenceGrade",
    "EvidenceKind",
    "EvidenceSource",
    "FEATURE_GUIDES",
    "FeatureGuide",
    "POLICY_CONTRACT_VERSION",
    "PolicyExecutionMode",
    "PolicyMaturity",
    "REPORT_SCHEMA_VERSION",
    "RULES",
    "RuleDefinition",
    "UNKNOWN_POLICY",
    "CalibrationDecision",
    "curate_feature_file",
    "decisions_fingerprint",
    "evidence_fingerprint",
    "feature_policy_fingerprint",
    "get_decision",
    "get_evidence",
    "get_feature_guide",
    "get_policy",
    "get_rule",
    "guidance_fingerprint",
    "known_curation_features",
    "known_decisions",
    "known_evidence",
    "known_guides",
    "known_rules",
    "registry_fingerprint",
    "registry_payload",
    "rules_fingerprint",
    "run_native_calibration_audit",
    "validate_curated_feature_file",
    "validate_engine_contract",
    "validate_registry",
]


def run_native_calibration_audit(*args, **kwargs):
    """Run the read-only CWP2 audit without importing audit machinery eagerly."""

    from .audit import run_audit

    return run_audit(*args, **kwargs)
