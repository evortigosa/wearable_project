"""
Wearable Data Processing and Modeling project
Google Fitness curation policy foundation.
"""


from .audit_models import AUDIT_VERSION, REPORT_SCHEMA_VERSION
from .decisions import DECISIONS, CalibrationDecision, decisions_fingerprint, get_decision, known_decisions
from .contract import (
    CONTRACT_SCHEMA_VERSION, CURATED_SCHEMA_VERSION, CURATION_CONTRACT_FILENAME, MATERIALIZED_COLUMN_ORDER,
    POLICY_LEVEL_FACTS, PROJECTION_DEFINITIONS, ROW_DEFAULTS, VIRTUAL_COLUMN_ORDER, contract_document,
    contract_fingerprint, effective_row_decisions, expected_acquisition_method, virtual_row_values,
)
from .engine import (
    ENGINE_VERSION, CuratedFeatureInfo, FeatureCurationResult, curate_feature_file, feature_policy_fingerprint,
    validate_curated_feature_file, validate_engine_contract,
)
from .evidence import (
    EVIDENCE, EvidenceKind, EvidenceSource, evidence_fingerprint, get_evidence, known_evidence,
)
from .evidence import EVIDENCE, EvidenceKind, EvidenceSource, evidence_fingerprint, get_evidence, known_evidence
from .guidance import FEATURE_GUIDES, FeatureGuide, get_feature_guide, guidance_fingerprint, known_guides
from .models import CurationStatus, DecisionStatus, EvidenceGrade, PolicyExecutionMode, PolicyMaturity
from .registry import (
    CURATION_POLICIES, CURATION_REGISTRY_VERSION, POLICY_CONTRACT_VERSION, UNKNOWN_POLICY, get_policy,
    known_curation_features, registry_fingerprint, registry_payload, validate_registry,
)
from .rules import RULES, RuleDefinition, get_rule, known_rules, rules_fingerprint
from .schema_migration import SCHEMA_MIGRATION_AUDIT_VERSION, SCHEMA_MIGRATION_REPORT_VERSION
from .pipeline import PIPELINE_VERSION, curate_dataset, format_summary
from .state import (
    CURATION_STATE_FILENAME, CURATION_STATE_SCHEMA_VERSION, CurationRunSummary, CurationStateDatabase,
    format_curation_report, load_curation_report,
)


__all__ = [
    "AUDIT_VERSION",
    "PIPELINE_VERSION",
    "CURATION_STATE_FILENAME",
    "CURATION_STATE_SCHEMA_VERSION",
    "CONTRACT_SCHEMA_VERSION",
    "CURATED_SCHEMA_VERSION",
    "CURATION_CONTRACT_FILENAME",
    "CurationRunSummary",
    "CurationStateDatabase",
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
    "MATERIALIZED_COLUMN_ORDER",
    "POLICY_LEVEL_FACTS",
    "PROJECTION_DEFINITIONS",
    "ROW_DEFAULTS",
    "VIRTUAL_COLUMN_ORDER",
    "PolicyExecutionMode",
    "PolicyMaturity",
    "REPORT_SCHEMA_VERSION",
    "SCHEMA_MIGRATION_AUDIT_VERSION",
    "SCHEMA_MIGRATION_REPORT_VERSION",
    "RULES",
    "RuleDefinition",
    "UNKNOWN_POLICY",
    "CalibrationDecision",
    "curate_feature_file",
    "curate_dataset",
    "contract_document",
    "contract_fingerprint",
    "decisions_fingerprint",
    "evidence_fingerprint",
    "effective_row_decisions",
    "expected_acquisition_method",
    "feature_policy_fingerprint",
    "format_curation_report",
    "format_summary",
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
    "load_curation_report",
    "registry_fingerprint",
    "registry_payload",
    "rules_fingerprint",
    "run_native_calibration_audit",
    "validate_curated_feature_file",
    "validate_engine_contract",
    "validate_registry",
    "virtual_row_values",
    "run_curation_acceptance",
    "run_cohort_acceptance",
    "run_schema_migration_audit",
]


def run_native_calibration_audit(*args, **kwargs):
    """Run the read-only CWP2 audit without importing audit machinery eagerly."""

    from .audit import run_audit

    return run_audit(*args, **kwargs)


def run_curation_acceptance(*args, **kwargs):
    """Run CWP5 acceptance without importing acceptance machinery eagerly."""

    from .acceptance import run_acceptance

    return run_acceptance(*args, **kwargs)


def run_cohort_acceptance(*args, **kwargs):
    """Run CWP6 full-cohort acceptance without importing it eagerly."""

    from .cohort_acceptance import run_cohort_acceptance as _run

    return _run(*args, **kwargs)


def run_schema_migration_audit(*args, **kwargs):
    """Audit schema-v1/schema-v2 semantic equivalence lazily."""

    from .schema_migration import run_schema_migration_audit as _run

    return _run(*args, **kwargs)
