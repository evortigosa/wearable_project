"""
Wearable Data Processing and Modeling project
Google Fitness native-processing contracts and version-aware candidate parser. WP1 defines the immutable registry
and native schemas. WP2 parses supported source/version envelopes into native-shaped candidates with exact accounting
and source-occurrence lineage.
"""

from wearable_project.processing.google_fitness.models import (
    PARSER_VERSION, MalformedCandidate, ParseDiagnostic, ParseResult, SourceFile, SourceOccurrence,
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
    "EXPECTED_REGISTRY_FINGERPRINT",
    "MalformedCandidate",
    "NATIVE_FEATURE_SPECS",
    "NORMALIZED_RAW_SOURCE",
    "PARSER_VERSION",
    "ParseDiagnostic",
    "ParseResult",
    "RAW_SOURCE_LABELS",
    "REGISTRY_VERSION",
    "SLEEP_STATE_MAPPING",
    "SOURCE_FEATURE_VERSION_BY_KEY",
    "SOURCE_FEATURE_VERSION_SPECS",
    "SOURCE_KEY",
    "SourceFile",
    "SourceOccurrence",
    "UNREGISTERED_ACTIVITY_POLICY",
    "activity_feature_names",
    "canonical_month_from_name",
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
