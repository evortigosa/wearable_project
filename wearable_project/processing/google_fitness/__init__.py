"""
Wearable Data Processing and Modeling project
Google Fitness native-processing contracts, parser, and native cleaner. WP1 defines immutable schemas and
routes. WP2 parses supported source/version envelopes into native-shaped candidates with exact accounting.
WP3 adds parallel candidate lineage, deterministic duplicate/revision/conflict reconciliation, and stable
native row ordering.
"""


from wearable_project.processing.google_fitness.cleaners import (
    CLEANER_VERSION, CleanedRowAudit, FeatureCleanResult, ParticipantCleanResult, ReconciliationAction,
    ReconciliationDecision, ReconciliationStage, clean_feature, clean_parse_result,
)
from wearable_project.processing.google_fitness.models import (
    CANDIDATE_LINEAGE_VERSION, PARSER_VERSION, CandidateLineage, MalformedCandidate, ParseDiagnostic,
    ParseResult, SourceFile, SourceOccurrence,
)
from wearable_project.processing.google_fitness.parser import (
    canonical_month_from_name, decode_payload, discover_month_files, merge_parse_results, parse_month_file,
    parse_participant_directory,
)
from wearable_project.processing.google_fitness.registry import (
    ACTIVITY_ROUTE_BY_LABEL, ACTIVITY_ROUTES, EXPECTED_REGISTRY_FINGERPRINT, NATIVE_FEATURE_SPECS,
    NORMALIZED_RAW_SOURCE, RAW_SOURCE_LABELS, REGISTRY_VERSION, SLEEP_STATE_MAPPING, SOURCE_FEATURE_VERSION_BY_KEY,
    SOURCE_FEATURE_VERSION_SPECS, SOURCE_KEY, UNREGISTERED_ACTIVITY_POLICY, activity_feature_names,
    feature_names, get_activity_route, get_feature_spec, get_source_feature_version_spec, registry_fingerprint,
    registry_payload, validate_registry,
)

__all__ = [
    "ACTIVITY_ROUTE_BY_LABEL",
    "ACTIVITY_ROUTES",
    "CANDIDATE_LINEAGE_VERSION",
    "CLEANER_VERSION",
    "CandidateLineage",
    "CleanedRowAudit",
    "EXPECTED_REGISTRY_FINGERPRINT",
    "FeatureCleanResult",
    "MalformedCandidate",
    "NATIVE_FEATURE_SPECS",
    "NORMALIZED_RAW_SOURCE",
    "PARSER_VERSION",
    "ParticipantCleanResult",
    "ParseDiagnostic",
    "ParseResult",
    "RAW_SOURCE_LABELS",
    "REGISTRY_VERSION",
    "ReconciliationAction",
    "ReconciliationDecision",
    "ReconciliationStage",
    "SLEEP_STATE_MAPPING",
    "SOURCE_FEATURE_VERSION_BY_KEY",
    "SOURCE_FEATURE_VERSION_SPECS",
    "SOURCE_KEY",
    "SourceFile",
    "SourceOccurrence",
    "UNREGISTERED_ACTIVITY_POLICY",
    "activity_feature_names",
    "canonical_month_from_name",
    "clean_feature",
    "clean_parse_result",
    "decode_payload",
    "discover_month_files",
    "feature_names",
    "get_activity_route",
    "get_feature_spec",
    "get_source_feature_version_spec",
    "merge_parse_results",
    "parse_month_file",
    "parse_participant_directory",
    "registry_fingerprint",
    "registry_payload",
    "validate_registry",
]
