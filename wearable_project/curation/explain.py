"""
Wearable Data Processing and Modeling project
Read-only explanation helpers for feature policies, rules, units and evidence.
"""

from __future__ import annotations
from dataclasses import asdict
import json
from typing import Iterable
from wearable_project.curation.evidence import EVIDENCE, EvidenceSource
from wearable_project.curation.guidance import (
    FEATURE_GUIDES, FeatureGuide, get_feature_guide, guidance_fingerprint,
)
from wearable_project.curation.models import FeaturePolicy, RuleDefinition, to_primitive
from wearable_project.curation.registry import get_policy, registry_fingerprint
from wearable_project.curation.rules import RULES, get_rule


def _policy_evidence_refs(policy: FeaturePolicy) -> tuple[str, ...]:
    refs: list[str] = list(policy.evidence_refs)
    for unit in policy.units.measurements:
        refs.extend(unit.evidence_refs)
    for source_rule in policy.provenance.source_rules:
        refs.extend(source_rule.evidence_refs)
    for rule_id in policy.curation.rule_ids:
        rule = RULES.get(rule_id)
        if rule is not None:
            refs.extend(rule.evidence_refs)
    return tuple(dict.fromkeys(refs))


def evidence_for_feature(feature: str) -> tuple[EvidenceSource, ...]:
    policy = get_policy(feature, allow_fallback=False)
    guide = get_feature_guide(feature)
    refs = tuple(dict.fromkeys(_policy_evidence_refs(policy) + guide.evidence_refs))
    return tuple(EVIDENCE[ref] for ref in refs)


def feature_explanation(feature: str, *, include_rules: bool = False, include_sources: bool = True,) -> dict[str, object]:
    policy = get_policy(feature, allow_fallback=False)
    guide = get_feature_guide(feature)
    payload: dict[str, object] = {
        "feature": feature,
        "guide": guide.as_dict(),
        "calibration": to_primitive(policy.calibration),
        "policy": {
            "maturity": policy.identity.maturity.value,
            "policy_version": policy.identity.policy_version,
            "feature_family": policy.schema.feature_family,
            "event_kind": policy.semantics.event_kind.value,
            "measurement_kind": policy.semantics.measurement_kind.value,
            "duration_model": policy.semantics.duration_model.value,
            "default_status": policy.curation.default_status.value,
            "default_inclusion": policy.curation.default_inclusion.value,
            "rule_ids": list(policy.curation.rule_ids),
            "resampling": to_primitive(policy.resampling),
            "policy_fingerprint": policy.fingerprint(),
        },
        "registry_fingerprint": registry_fingerprint(),
        "guidance_fingerprint": guidance_fingerprint(),
    }
    if include_rules:
        payload["rules"] = {
            rule_id: to_primitive(get_rule(rule_id)) for rule_id in policy.curation.rule_ids
        }
    if include_sources:
        payload["sources"] = {
            source.evidence_id: to_primitive(source) for source in evidence_for_feature(feature)
        }
    return payload


def rule_explanation(rule_id: str) -> dict[str, object]:
    rule = get_rule(rule_id)
    return {
        "rule": to_primitive(rule), "sources": {ref: to_primitive(EVIDENCE[ref]) for ref in rule.evidence_refs},
    }


def unit_policy_explanation(feature: str) -> dict[str, object]:
    policy = get_policy(feature, allow_fallback=False)
    return {
        "feature": feature,
        "maturity": policy.identity.maturity.value,
        "calibration": to_primitive(policy.calibration),
        "measurements": [to_primitive(item) for item in policy.units.measurements],
        "cross_feature_rules": list(policy.cross_feature.rule_ids),
        "safe_fallback": policy.calibration.safe_fallback,
        "sources": {source.evidence_id: to_primitive(source) for source in evidence_for_feature(feature)},
    }


def _source_lines(sources: Iterable[EvidenceSource]) -> list[str]:
    lines: list[str] = []
    for source in sources:
        lines.append(f"- [{source.evidence_id}] {source.citation_text}")
        lines.append(f"  {source.brief_summary}")
        lines.append(f"  Source: {source.locator}")
        if source.limitations:
            lines.append(f"  Limitation: {source.limitations}")
    return lines


def render_feature_text(feature: str, *, include_rules: bool = False, include_sources: bool = True,) -> str:
    policy = get_policy(feature, allow_fallback=False)
    guide = get_feature_guide(feature)
    lines = [
        f"Feature: {feature}",
        f"Category: {guide.category}",
        f"Policy maturity: {policy.identity.maturity.value}",
        f"Evidence grade: {policy.calibration.evidence_grade.value}",
        f"Execution mode: {policy.calibration.execution_mode.value}",
        "",
        "Description:",
        f"  {guide.short_description}",
        "",
        "One native row means:",
        f"  {guide.one_row_means}",
        "",
        "Native time semantics:",
        f"  {guide.native_time_semantics}",
        "",
        "Units:",
        f"  {guide.unit_summary}",
        "",
        "Acquisition context:",
        f"  {guide.acquisition_summary}",
        "",
        "Curation approach:",
        f"  {guide.curation_summary}",
        "",
        "Calibration rationale:",
        f"  {policy.calibration.rationale}",
        "",
        "Safe fallback:",
        f"  {policy.calibration.safe_fallback}",
    ]
    if policy.calibration.required_audits:
        lines.extend(("", "Required calibration audits:"))
        lines.extend(f"  - {audit}" for audit in policy.calibration.required_audits)
    if guide.important_caveats:
        lines.extend(("", "Important caveats:"))
        lines.extend(f"  - {item}" for item in guide.important_caveats)
    if guide.not_equivalent_to:
        lines.extend(("", "Do not treat this feature as:"))
        lines.extend(f"  - {item}" for item in guide.not_equivalent_to)
    if include_rules:
        lines.extend(("", "Declared curation rules:"))
        for rule_id in policy.curation.rule_ids:
            rule = RULES[rule_id]
            lines.append(f"  - {rule_id}: {rule.description}")
    if include_sources:
        lines.extend(("", "Evidence sources:"))
        lines.extend(_source_lines(evidence_for_feature(feature)))
    return "\n".join(lines) + "\n"


def render_feature_markdown(feature: str, *, include_rules: bool = False, include_sources: bool = True,) -> str:
    policy = get_policy(feature, allow_fallback=False)
    guide = get_feature_guide(feature)
    lines = [
        f"# {feature}",
        "",
        f"**Category:** {guide.category}  ",
        f"**Policy maturity:** {policy.identity.maturity.value}  ",
        f"**Evidence grade:** {policy.calibration.evidence_grade.value}  ",
        f"**Execution mode:** {policy.calibration.execution_mode.value}",
        "",
        "## Description",
        "",
        guide.short_description,
        "",
        "## One native row means",
        "",
        guide.one_row_means,
        "",
        "## Native time semantics",
        "",
        guide.native_time_semantics,
        "",
        "## Units",
        "",
        guide.unit_summary,
        "",
        "## Acquisition context",
        "",
        guide.acquisition_summary,
        "",
        "## Curation approach",
        "",
        guide.curation_summary,
        "",
        "## Calibration",
        "",
        f"**Rationale:** {policy.calibration.rationale}",
        "",
        f"**Safe fallback:** {policy.calibration.safe_fallback}",
    ]
    if policy.calibration.required_audits:
        lines.extend(("", "**Required audits:**"))
        lines.extend(f"- `{item}`" for item in policy.calibration.required_audits)
    if guide.important_caveats:
        lines.extend(("", "## Important caveats", ""))
        lines.extend(f"- {item}" for item in guide.important_caveats)
    if guide.not_equivalent_to:
        lines.extend(("", "## Do not treat this feature as", ""))
        lines.extend(f"- {item}" for item in guide.not_equivalent_to)
    if include_rules:
        lines.extend(("", "## Declared curation rules", ""))
        for rule_id in policy.curation.rule_ids:
            lines.append(f"- **`{rule_id}`:** {RULES[rule_id].description}")
    if include_sources:
        lines.extend(("", "## Evidence sources", ""))
        for source in evidence_for_feature(feature):
            lines.append(f"- **[{source.evidence_id}]({source.locator}) — {source.citation_text}**")
            lines.append(f"  - {source.brief_summary}")
            if source.limitations:
                lines.append(f"  - *Limitation:* {source.limitations}")
    return "\n".join(lines) + "\n"


def render_all_guides_markdown(*, include_rules: bool = False, include_sources: bool = True,) -> str:
    sections = [
        "# wearable-project feature guide",
        "",
        "This guide explains native row semantics and curation limitations. It is not medical advice and does not replace source documentation or clinical measurement standards.",
        "",
        f"Guidance fingerprint: `{guidance_fingerprint()}`",
        "",
    ]
    for feature in sorted(FEATURE_GUIDES):
        rendered = render_feature_markdown(feature, include_rules=include_rules, include_sources=include_sources)
        sections.append(rendered.replace(f"# {feature}", f"## {feature}", 1).rstrip())
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def render_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
