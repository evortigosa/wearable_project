"""
Wearable Data Processing and Modeling project
Google Fitness curation environment, semantic fingerprints, and source integrity.
"""


from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
from pathlib import Path
import subprocess
from typing import Any
import wearable_project
from wearable_project import __release_label__, __version__
from .audit_models import AUDIT_VERSION, REPORT_SCHEMA_VERSION
from .contract import CURATED_SCHEMA_VERSION, CONTRACT_SCHEMA_VERSION, contract_fingerprint
from .decisions import decisions_fingerprint
from .engine import ENGINE_VERSION
from .evidence import evidence_fingerprint
from .guidance import guidance_fingerprint
from .pipeline import PIPELINE_VERSION
from .registry import CURATION_REGISTRY_VERSION, POLICY_CONTRACT_VERSION, registry_fingerprint
from .rules import rules_fingerprint
from .state import CURATION_STATE_SCHEMA_VERSION

try:
    from wearable_project.utils.release_manifest import (
        EXPECTED_GFIT_CURATION_DECISIONS_FINGERPRINT, EXPECTED_GFIT_CURATION_EVIDENCE_FINGERPRINT,
        EXPECTED_GFIT_CURATION_GUIDANCE_FINGERPRINT, EXPECTED_GFIT_CURATION_MODULE_SHA256,
        EXPECTED_GFIT_CURATION_REGISTRY_FINGERPRINT, EXPECTED_GFIT_CURATION_RULES_FINGERPRINT,
    )
except ImportError:  # Allows the centralized generator to add the first Google curation block.
    EXPECTED_GFIT_CURATION_MODULE_SHA256 = {}
    EXPECTED_GFIT_CURATION_REGISTRY_FINGERPRINT = ""
    EXPECTED_GFIT_CURATION_EVIDENCE_FINGERPRINT = ""
    EXPECTED_GFIT_CURATION_RULES_FINGERPRINT = ""
    EXPECTED_GFIT_CURATION_DECISIONS_FINGERPRINT = ""
    EXPECTED_GFIT_CURATION_GUIDANCE_FINGERPRINT = ""


_TRACKED_MODULES = (
    "__init__.py",
    "audit.py",
    "audit_models.py",
    "context.py",
    "contract.py",
    "decisions.py",
    "engine.py",
    "environment.py",
    "evidence.py",
    "explain.py",
    "guidance.py",
    "models.py",
    "pipeline.py",
    "provenance.py",
    "registry.py",
    "rules.py",
    "state.py",
    "strategies.py",
    "unit_resolution.py",
)


@dataclass(frozen=True, slots=True)
class CurationEnvironmentManifest:
    package_version: str
    package_release_label: str
    current_working_directory: str
    imported_package_path: str
    distribution_package_path: str | None
    import_source_kind: str
    audit_version: str
    audit_report_schema_version: str
    curation_registry_version: str
    policy_contract_version: str
    engine_version: str
    pipeline_version: str
    state_schema_version: str
    curated_schema_version: str
    contract_schema_version: str
    contract_fingerprint: str
    registry_fingerprint: str
    evidence_fingerprint: str
    rules_fingerprint: str
    decisions_fingerprint: str
    guidance_fingerprint: str
    semantic_integrity: dict[str, bool]
    module_integrity: dict[str, bool]
    module_sha256: dict[str, str | None]
    git_commit: str | None
    git_dirty: bool | None
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_info(root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def environment_manifest() -> CurationEnvironmentManifest:
    package_root = Path(wearable_project.__file__).resolve().parent
    source_root = package_root / "curation" / "google_fitness"
    repository_root = package_root.parent
    try:
        distribution = metadata.distribution("wearable-project")
        distribution_path: str | None = str(Path(distribution.locate_file("")))
    except metadata.PackageNotFoundError:
        distribution_path = None

    actual: dict[str, str | None] = {}
    integrity: dict[str, bool] = {}
    for name in _TRACKED_MODULES:
        path = source_root / name
        observed = _hash(path) if path.is_file() else None
        expected = EXPECTED_GFIT_CURATION_MODULE_SHA256.get(name)
        actual[name] = observed
        integrity[name] = expected is not None and observed == expected

    semantics = {
        "registry": registry_fingerprint() == EXPECTED_GFIT_CURATION_REGISTRY_FINGERPRINT,
        "evidence": evidence_fingerprint() == EXPECTED_GFIT_CURATION_EVIDENCE_FINGERPRINT,
        "rules": rules_fingerprint() == EXPECTED_GFIT_CURATION_RULES_FINGERPRINT,
        "decisions": decisions_fingerprint() == EXPECTED_GFIT_CURATION_DECISIONS_FINGERPRINT,
        "guidance": guidance_fingerprint() == EXPECTED_GFIT_CURATION_GUIDANCE_FINGERPRINT,
    }
    commit, dirty = _git_info(repository_root)
    warnings: list[str] = []
    invalid_modules = [name for name, valid in integrity.items() if not valid]
    if invalid_modules:
        warnings.append(
            "Google Fitness curation source differs from its centralized release manifest: "
            + ", ".join(invalid_modules)
        )
    manifest_only = sorted(set(EXPECTED_GFIT_CURATION_MODULE_SHA256) - set(_TRACKED_MODULES))
    if manifest_only:
        warnings.append(
            "Google Fitness curation manifest contains untracked modules: "
            + ", ".join(manifest_only)
        )
    invalid_semantics = [name for name, valid in semantics.items() if not valid]
    if invalid_semantics:
        warnings.append(
            "Google Fitness curation semantic fingerprints differ from the release manifest: "
            + ", ".join(invalid_semantics)
        )

    imported = Path(wearable_project.__file__).resolve()
    if distribution_path is None:
        source_kind = "local_checkout"
    else:
        try:
            imported.relative_to(Path(distribution_path).resolve())
            source_kind = "installed_distribution"
        except ValueError:
            source_kind = "local_checkout"
            warnings.append(
                "wearable_project is imported from a local checkout rather than the installed distribution"
            )

    return CurationEnvironmentManifest(
        package_version=__version__,
        package_release_label=__release_label__,
        current_working_directory=str(Path.cwd()),
        imported_package_path=str(imported),
        distribution_package_path=distribution_path,
        import_source_kind=source_kind,
        audit_version=AUDIT_VERSION,
        audit_report_schema_version=REPORT_SCHEMA_VERSION,
        curation_registry_version=CURATION_REGISTRY_VERSION,
        policy_contract_version=POLICY_CONTRACT_VERSION,
        engine_version=ENGINE_VERSION,
        pipeline_version=PIPELINE_VERSION,
        state_schema_version=CURATION_STATE_SCHEMA_VERSION,
        curated_schema_version=CURATED_SCHEMA_VERSION,
        contract_schema_version=CONTRACT_SCHEMA_VERSION,
        contract_fingerprint=contract_fingerprint(),
        registry_fingerprint=registry_fingerprint(),
        evidence_fingerprint=evidence_fingerprint(),
        rules_fingerprint=rules_fingerprint(),
        decisions_fingerprint=decisions_fingerprint(),
        guidance_fingerprint=guidance_fingerprint(),
        semantic_integrity=semantics,
        module_integrity=integrity,
        module_sha256=actual,
        git_commit=commit,
        git_dirty=dirty,
        warnings=tuple(warnings),
    )
