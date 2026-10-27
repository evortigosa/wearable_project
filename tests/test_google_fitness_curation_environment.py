"""
Wearable Data Processing and Modeling project
Centralized source-integrity checks for Google Fitness curation (CWP4).
"""


from __future__ import annotations
from wearable_project.curation.google_fitness.environment import _TRACKED_MODULES, environment_manifest
from wearable_project.utils.regenerate_manifest import compute
from wearable_project.utils.release_manifest import EXPECTED_GFIT_CURATION_MODULE_SHA256


def test_google_curation_environment_integrity_is_complete() -> None:
    expected_modules = {
        "__init__.py", "audit.py", "audit_models.py", "context.py", "contract.py",
        "decisions.py", "engine.py", "environment.py", "evidence.py", "explain.py",
        "guidance.py", "models.py", "pipeline.py", "provenance.py", "registry.py",
        "rules.py", "state.py", "strategies.py", "unit_resolution.py",
    }
    assert set(_TRACKED_MODULES) == expected_modules
    manifest = environment_manifest()
    assert set(manifest.module_integrity) == set(_TRACKED_MODULES)
    assert all(manifest.module_integrity.values())
    assert all(manifest.semantic_integrity.values())
    assert manifest.audit_version == "google-fitness-curation-audit-1.0.0"
    assert manifest.audit_report_schema_version == "google-fitness-curation-audit-report-1.0.0"
    assert not [warning for warning in manifest.warnings if "release manifest" in warning]


def test_central_generator_tracks_google_curation_modules() -> None:
    values = compute()
    assert values["EXPECTED_GFIT_CURATION_MODULE_SHA256"] == EXPECTED_GFIT_CURATION_MODULE_SHA256
    assert set(EXPECTED_GFIT_CURATION_MODULE_SHA256) == set(_TRACKED_MODULES)
