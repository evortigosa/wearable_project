"""
Wearable Data Processing and Modeling project
Human- and machine-readable explanations for Google Fitness curation policy.
"""


from __future__ import annotations
import json
from typing import Any
from .decisions import get_decision
from .evidence import get_evidence
from .guidance import get_feature_guide
from .models import to_primitive
from .registry import get_policy
from .rules import get_rule


def render_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _evidence_payload(ids: tuple[str, ...] | list[str]) -> dict[str, object]:
    return {evidence_id: get_evidence(evidence_id).as_dict() for evidence_id in ids}


def rule_explanation(rule_id: str) -> dict[str, object]:
    rule = get_rule(rule_id)
    return {
        "rule_id": rule_id,
        "rule": rule.as_dict(),
        "sources": _evidence_payload(rule.evidence_refs),
    }


def unit_policy_explanation(feature: str) -> dict[str, object]:
    policy = get_policy(feature, allow_fallback=False)
    measurements = []
    evidence_ids: list[str] = []
    for measurement in policy.measurements:
        units = []
        for unit in measurement.unit_by_version:
            units.append(to_primitive(unit))
            evidence_ids.extend(unit.evidence_refs)
        measurements.append({
            "column": measurement.column,
            "measurement_kind": measurement.kind.value,
            "required": measurement.required,
            "raw_projection_enabled": measurement.raw_projection_enabled,
            "canonical_projection_enabled": measurement.canonical_projection_enabled,
            "unit_by_version": units,
        })
    return {
        "feature": feature,
        "maturity": policy.calibration.maturity.value,
        "evidence_grade": policy.calibration.evidence_grade.value,
        "execution_mode": policy.calibration.execution_mode.value,
        "measurements": measurements,
        "safe_fallback": "; ".join(policy.calibration.limitations) or "Use the native value and withhold unsupported canonicalization.",
        "required_audits": list(policy.calibration.required_audits),
        "sources": _evidence_payload(tuple(dict.fromkeys(evidence_ids))),
    }


def feature_explanation(feature: str, *, include_rules: bool = False) -> dict[str, object]:
    policy = get_policy(feature, allow_fallback=False)
    guide = get_feature_guide(feature)
    evidence_ids: list[str] = []
    evidence_ids.extend(guide.evidence_refs)
    evidence_ids.extend(policy.provenance.evidence_refs)
    evidence_ids.extend(policy.evidence_refs)
    for measurement in policy.measurements:
        for unit in measurement.unit_by_version:
            evidence_ids.extend(unit.evidence_refs)
    payload: dict[str, object] = {
        "feature": feature,
        "guide": guide.as_dict(),
        "policy": to_primitive(policy),
        "decisions": {
            decision_id: get_decision(decision_id).as_dict() for decision_id in policy.calibration.decision_ids
        },
        "sources": _evidence_payload(tuple(dict.fromkeys(evidence_ids))),
    }
    if include_rules:
        payload["rules"] = {
            rule_id: get_rule(rule_id).as_dict() for rule_id in policy.curation.rule_ids
        }
    return payload
