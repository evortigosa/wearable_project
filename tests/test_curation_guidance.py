"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import json
from wearable_project.curation.evidence import EVIDENCE
from wearable_project.curation.explain import (
    evidence_for_feature, feature_explanation, render_all_guides_markdown, render_feature_text,
)
from wearable_project.curation.guidance import (
    FEATURE_GUIDES, get_feature_guide, guidance_fingerprint,
)
from wearable_project.curation.registry import COHORT_OBSERVED_FEATURES


def test_guidance_covers_exactly_the_full_cohort_features() -> None:
    assert set(FEATURE_GUIDES) == set(COHORT_OBSERVED_FEATURES)
    assert len(FEATURE_GUIDES) == 33


def test_every_guide_is_complete_and_evidence_refs_exist() -> None:
    for feature, guide in FEATURE_GUIDES.items():
        assert guide.feature == feature
        assert guide.category
        assert guide.short_description
        assert guide.one_row_means
        assert guide.native_time_semantics
        assert guide.unit_summary
        assert guide.acquisition_summary
        assert guide.curation_summary
        assert guide.evidence_refs
        assert not (set(guide.evidence_refs) - set(EVIDENCE))


def test_evidence_catalog_is_enriched_and_offline_reproducible() -> None:
    for source in EVIDENCE.values():
        assert source.brief_summary
        assert source.limitations
        assert source.last_verified
        assert source.citation_text
        assert source.locator.startswith(("https://", "http://", "project://"))


def test_guidance_fingerprint_is_deterministic_and_separate() -> None:
    assert guidance_fingerprint() == guidance_fingerprint()
    assert len(guidance_fingerprint()) == 64


def test_feature_explanation_links_policy_rules_and_sources() -> None:
    payload = feature_explanation("Sleep", include_rules=True, include_sources=True)
    assert payload["feature"] == "Sleep"
    assert payload["calibration"]["execution_mode"] == "reviewed_execution"
    assert "sleep_detailed_state_overlap" in payload["rules"]
    assert "apple_sleep_analysis" in payload["sources"]


def test_human_feature_guide_contains_caveat_and_url() -> None:
    rendered = render_feature_text("BloodAlcoholContent", include_sources=True)
    assert "calculator" in rendered.lower()
    assert "https://" in rendered


def test_all_guides_markdown_has_one_section_per_feature() -> None:
    rendered = render_all_guides_markdown(include_sources=False)
    for feature in COHORT_OBSERVED_FEATURES:
        assert f"## {feature}" in rendered
