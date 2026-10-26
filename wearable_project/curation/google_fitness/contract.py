"""
Wearable Data Processing and Modeling project
Versioned Google Fitness curated-schema contract. The curated CSVs materialize only row-level decisions and the
reviewed ``acquisition_method`` interpretation.  Deterministic feature/version facts (unit evidence, canonical
identity projection, derivation, and tracking) live in this root-level contract and are exposed as virtual projections.
"""


from __future__ import annotations
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
from wearable_project.exceptions import CurationError
from wearable_project.processing.google_fitness.registry import SOURCE_KEY
from .decisions import decisions_fingerprint
from .evidence import evidence_fingerprint
from .guidance import guidance_fingerprint
from .models import EventKind, FeaturePolicy, to_primitive
from .provenance import classify_provenance
from .registry import (
    CURATION_POLICIES, CURATION_REGISTRY_VERSION, POLICY_CONTRACT_VERSION, get_policy, registry_fingerprint,
)
from .rules import RULES, rules_fingerprint
from .unit_resolution import resolve_units


CURATED_SCHEMA_VERSION = "google-fitness-curated-schema-v2-0.1.0"
CONTRACT_SCHEMA_VERSION = "google-fitness-curation-contract-v2-0.1.0"
CURATION_CONTRACT_FILENAME = ".curation_contract.json"

MATERIALIZED_COLUMN_ORDER: tuple[str, ...] = (
    "acquisition_method",
    "step_count_role",
    "curation_status",
    "curation_flags",
    "include_by_default",
)
VIRTUAL_COLUMN_ORDER: tuple[str, ...] = (
    "canonical_value",
    "canonical_unit",
    "curation_unit_status",
    "unit_evidence",
    "derivation_method",
    "tracking_status",
)

ROW_DEFAULTS: dict[str, object] = {
    "curation_status": "pass",
    "curation_flags": [],
    "include_by_default": 1,
}

POLICY_LEVEL_FACTS: tuple[dict[str, object], ...] = (
    {
        "fact_id": "activity_quantity_semantics_unresolved",
        "scope": "measurement_policy",
        "applies_to": "activity-derived features containing quantity",
        "meaning": "The native quantity field is retained but has no defensible common canonical unit or meaning.",
        "former_row_flag": "activity_quantity_semantics_unresolved",
    },
    {
        "fact_id": "exporter_mapping_pending",
        "scope": "feature_version_measurement_policy",
        "applies_to": ["Calories.value", "Distance.value", "Weight.value", "Height.value"],
        "meaning": "Official Google units are proposed, but canonical projection remains withheld until the exporter mapping is proven.",
        "former_row_flag": "canonical_unit_withheld_pending_exporter_mapping",
    },
)

PROJECTION_DEFINITIONS: dict[str, object] = {
    "acquisition_method": {
        "storage": "materialized",
        "classifier": "feature-aware conservative provenance classifier",
        "inputs": [
            "source_id", "source_name", "device", "app_package_name",
            "device_manufacturer", "device_model", "device_type",
            "device_uid", "added_by",
        ],
        "values": [
            "device_estimate", "health_import_unknown",
            "third_party_import", "unknown",
        ],
        "status_effect": "none",
        "inclusion_effect": "none",
    },
    "derivation_method": {
        "storage": "virtual",
        "mapping": {
            "activity_or_sleep": "activity_classification",
            "Calories_or_Distance": "application_summary",
            "Weight_or_Height": "raw",
            "StepCount_1.0": "legacy_summary",
            "StepCount_2.0_estimated_steps": "estimated",
            "StepCount_2.0_merge_step_deltas": "merged",
            "StepCount_2.0_other_named_source": "third_party_aggregate",
            "otherwise": "unknown",
        },
    },
    "tracking_status": {
        "storage": "virtual",
        "source_column": "tracked",
        "mapping": {"true": "tracked", "false": "untracked", "blank": "unknown"},
        "non_activity_default": "not_applicable",
    },
    "canonical_value": {
        "storage": "virtual",
        "StepCount": "identity projection of finite nonnegative native value",
        "other_features": "withheld",
    },
    "canonical_unit": {
        "storage": "virtual",
        "StepCount": "count",
        "other_features": "withheld",
    },
    "curation_unit_status": {
        "storage": "virtual",
        "mapping": {
            "StepCount_1.0": "resolved_source_convention",
            "StepCount_2.0": "resolved_payload_explicit",
            "Calories_Distance_Weight_Height": "withheld_exporter_mapping_pending",
            "activity_and_sleep": "not_applicable",
        },
    },
    "unit_evidence": {
        "storage": "virtual",
        "mapping": {
            "StepCount_1.0": "source_convention:google_fitness_step_count_v1",
            "StepCount_2.0": "payload_explicit:com.google.step_count.delta",
            "Calories": "exporter_mapping_pending:proposed_kcal",
            "Distance_or_Height": "exporter_mapping_pending:proposed_m",
            "Weight": "exporter_mapping_pending:proposed_kg",
            "activity_and_sleep": "none",
        },
    },
    "step_count_role": {
        "storage": "materialized",
        "scope": "StepCount outer-row alternative group",
        "reason": "group-aware preference is a central analytical decision and is not a trivial row-local recoding",
    },
    "curation_status": {
        "storage": "sparse_materialized",
        "blank_default": "pass",
    },
    "curation_flags": {
        "storage": "sparse_materialized",
        "scope": "row- or group-specific findings only",
        "policy_level_facts": "stored in policy_level_facts instead of repeated on rows",
    },
    "include_by_default": {
        "storage": "sparse_materialized",
        "blank_default": 1,
        "excluded_value": 0,
    },
}


def _feature_contract(policy: FeaturePolicy) -> dict[str, object]:
    measurements: list[dict[str, object]] = []
    for measurement in policy.measurements:
        versions: list[dict[str, object]] = []
        for unit in measurement.unit_by_version:
            versions.append(
                {
                    "versions": list(unit.versions),
                    "raw_unit_status": unit.raw_unit_status.value,
                    "raw_unit": unit.raw_unit,
                    "canonical_unit": unit.canonical_unit,
                    "evidence_grade": unit.evidence_grade.value,
                    "execution_mode": unit.execution_mode.value,
                    "canonicalization": unit.canonicalization.value,
                    "evidence_refs": list(unit.evidence_refs),
                    "limitations": list(unit.limitations),
                }
            )
        measurements.append(
            {
                "column": measurement.column,
                "kind": measurement.kind.value,
                "required": measurement.required,
                "raw_projection_enabled": measurement.raw_projection_enabled,
                "canonical_projection_enabled": measurement.canonical_projection_enabled,
                "unit_by_version": versions,
                "description": measurement.description,
            }
        )

    rules = []
    for rule_id in policy.curation.rule_ids:
        rule = RULES[rule_id]
        rules.append(
            {
                "rule_id": rule_id,
                "flag": rule.flag,
                "flag_scope": "row_finding" if rule.materialize_flag else "policy_context",
                "status_effect": rule.status_effect.value,
                "inclusion_effect": rule.inclusion_effect.value,
            }
        )

    return {
        "event_kind": policy.event_kind.value,
        "row_meaning": policy.native_schema.row_meaning,
        "native_columns": list(policy.native_schema.native_columns),
        "supported_versions": list(policy.native_schema.supported_versions),
        "measurements": measurements,
        "materialized_columns": list(policy.output.materialized_columns),
        "virtual_columns": list(policy.output.virtual_columns),
        "fixed_materialized_columns": list(policy.output.fixed_columns),
        "sparse_materialized_columns": list(policy.output.sparse_columns),
        "provenance": {
            "raw_columns": list(policy.provenance.raw_columns),
            "annotation_dimensions": list(policy.provenance.output_dimensions),
            "classifier_id": policy.provenance.classifier_id,
            "limitations": list(policy.provenance.limitations),
        },
        "grouping": to_primitive(policy.grouping),
        "rules": rules,
        "default_status": policy.curation.default_status.value,
        "default_inclusion": policy.curation.default_inclusion.value,
    }


def contract_definition() -> dict[str, object]:
    """Return the deterministic scientific schema contract."""

    return {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "curated_schema_version": CURATED_SCHEMA_VERSION,
        "source_key": SOURCE_KEY,
        "curation_registry_version": CURATION_REGISTRY_VERSION,
        "policy_contract_version": POLICY_CONTRACT_VERSION,
        "row_defaults": ROW_DEFAULTS,
        "materialized_column_order": list(MATERIALIZED_COLUMN_ORDER),
        "virtual_column_order": list(VIRTUAL_COLUMN_ORDER),
        "policy_level_facts": list(POLICY_LEVEL_FACTS),
        "projection_definitions": PROJECTION_DEFINITIONS,
        "features": {
            feature: _feature_contract(policy)
            for feature, policy in sorted(CURATION_POLICIES.items())
        },
        "semantic_fingerprints": {
            "registry": registry_fingerprint(),
            "evidence": evidence_fingerprint(),
            "rules": rules_fingerprint(),
            "decisions": decisions_fingerprint(),
            "guidance": guidance_fingerprint(),
        },
    }


def contract_fingerprint() -> str:
    rendered = json.dumps(
        contract_definition(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(rendered).hexdigest()


def contract_document(
    *,
    engine_version: str,
    pipeline_version: str,
    state_schema_version: str,
) -> dict[str, object]:
    payload = contract_definition()
    payload.update(
        {
            "contract_fingerprint": contract_fingerprint(),
            "implementation": {
                "engine_version": engine_version,
                "pipeline_version": pipeline_version,
                "state_schema_version": state_schema_version,
            },
        }
    )
    return payload


def contract_text(
    *,
    engine_version: str,
    pipeline_version: str,
    state_schema_version: str,
) -> str:
    return json.dumps(
        contract_document(
            engine_version=engine_version,
            pipeline_version=pipeline_version,
            state_schema_version=state_schema_version,
        ),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def write_contract_atomic(
    path: Path,
    *,
    engine_version: str,
    pipeline_version: str,
    state_schema_version: str,
) -> str:
    """Write the root contract atomically and return its SHA-256."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = contract_text(
        engine_version=engine_version,
        pipeline_version=pipeline_version,
        state_schema_version=state_schema_version,
    )
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(path.read_bytes()).hexdigest()


def validate_contract_document(
    payload: Mapping[str, Any],
    *,
    engine_version: str,
    pipeline_version: str,
    state_schema_version: str,
) -> tuple[str, ...]:
    expected = contract_document(
        engine_version=engine_version,
        pipeline_version=pipeline_version,
        state_schema_version=state_schema_version,
    )
    issues: list[str] = []
    for key in (
        "contract_schema_version",
        "curated_schema_version",
        "source_key",
        "curation_registry_version",
        "policy_contract_version",
        "contract_fingerprint",
        "row_defaults",
        "materialized_column_order",
        "virtual_column_order",
        "policy_level_facts",
        "projection_definitions",
        "features",
        "semantic_fingerprints",
        "implementation",
    ):
        if payload.get(key) != expected.get(key):
            issues.append(f"contract field differs: {key}")
    return tuple(issues)


def validate_contract_file(
    path: Path,
    *,
    engine_version: str,
    pipeline_version: str,
    state_schema_version: str,
) -> tuple[str, ...]:
    path = Path(path)
    if not path.is_file():
        return (f"missing curation contract: {path}",)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (f"invalid curation contract: {type(exc).__name__}: {exc}",)
    if not isinstance(payload, dict):
        return ("curation contract must be a JSON object",)
    return validate_contract_document(
        payload,
        engine_version=engine_version,
        pipeline_version=pipeline_version,
        state_schema_version=state_schema_version,
    )


def virtual_row_values(
    feature: str,
    row: Mapping[str, str],
    policy: FeaturePolicy | None = None,
) -> dict[str, str]:
    """Reconstruct virtual derived values from native evidence and policy."""

    policy = policy or get_policy(feature)
    if policy.unknown_fallback:
        return {"derivation_method": "unknown"}

    annotation = classify_provenance(feature, policy.event_kind, row)
    values = annotation.as_values(policy.provenance.output_dimensions)
    values.pop("acquisition_method", None)  # Materialized in schema v2.

    unit = resolve_units(feature, row, policy)
    if unit is not None:
        values.update(unit.as_values())

    return {
        column: str(values[column])
        for column in policy.output.virtual_columns
        if values.get(column) not in (None, "")
    }


def expected_acquisition_method(
    feature: str,
    row: Mapping[str, str],
    policy: FeaturePolicy | None = None,
) -> str:
    """Recompute the one broadly materialized provenance interpretation."""

    policy = policy or get_policy(feature)
    if policy.unknown_fallback:
        return "unknown"
    return classify_provenance(feature, policy.event_kind, row).acquisition_method.value


def virtual_default(column: str) -> str:
    if column == "tracking_status":
        return "not_applicable"
    if column == "derivation_method":
        return "unknown"
    if column == "curation_unit_status":
        return "not_applicable"
    if column == "unit_evidence":
        return "none"
    return ""


def effective_row_decisions(row: Mapping[str, str]) -> dict[str, object]:
    """Return sparse-default-aware row decisions from a schema-v2 curated row."""

    status = (row.get("curation_status") or "").strip() or "pass"
    include = (row.get("include_by_default") or "").strip()
    flags = tuple(item for item in (row.get("curation_flags") or "").split(";") if item)
    return {
        "curation_status": status,
        "curation_flags": flags,
        "include_by_default": include != "0",
    }


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "CURATED_SCHEMA_VERSION",
    "CURATION_CONTRACT_FILENAME",
    "MATERIALIZED_COLUMN_ORDER",
    "POLICY_LEVEL_FACTS",
    "PROJECTION_DEFINITIONS",
    "ROW_DEFAULTS",
    "VIRTUAL_COLUMN_ORDER",
    "contract_definition",
    "contract_document",
    "contract_fingerprint",
    "contract_text",
    "effective_row_decisions",
    "expected_acquisition_method",
    "validate_contract_document",
    "validate_contract_file",
    "virtual_default",
    "virtual_row_values",
    "write_contract_atomic",
]
