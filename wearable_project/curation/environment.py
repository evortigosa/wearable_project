"""
Wearable Data Processing and Modeling project
Reproducibility and import-source diagnostics for curation releases.
"""


from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
import site
import subprocess
import sys
import sysconfig
from typing import Any, Mapping
import pandas as pd
import wearable_project
from wearable_project.curation.decisions import decisions_fingerprint
from wearable_project.curation.guidance import guidance_fingerprint
from wearable_project.curation.registry import registry_fingerprint


_TRACKED_MODULES = (
    "audit.py",
    "curate_cli.py",
    "decisions.py",
    "engine.py",
    "environment.py",
    "evidence.py",
    "explain.py",
    "guidance.py",
    "models.py",
    "pipeline.py",
    "registry.py",
    "rules.py",
    "state.py",
    "strategies.py",
    "unit_resolution.py",
)


@dataclass(frozen=True, slots=True)
class EnvironmentManifest:
    package_version: str
    package_release_label: str
    imported_package_path: str
    current_working_directory: str
    sys_path_head: tuple[str, ...]
    distribution_version: str | None
    distribution_root: str | None
    distribution_package_path: str | None
    distribution_candidates: tuple[Mapping[str, str | None], ...]
    site_package_roots: tuple[str, ...]
    import_source_kind: str
    python_version: str
    pandas_version: str
    platform: str
    registry_fingerprint: str
    guidance_fingerprint: str
    decisions_fingerprint: str
    module_sha256: Mapping[str, str]
    expected_module_sha256: Mapping[str, str]
    module_integrity: Mapping[str, bool]
    git_commit: str | None
    git_dirty: bool | None
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git_info(root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=2,
        ).stdout.strip())
        return commit or None, dirty
    except Exception:
        return None, None



def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True

def environment_manifest() -> EnvironmentManifest:
    module_file = wearable_project.__file__
    if module_file is None:
        raise RuntimeError("wearable_project has no __file__ (namespace package?)")
    package_path = Path(module_file).resolve()
    package_root = package_path.parent
    curation_root = package_root / "curation"
    source_root = package_root.parent
    candidates: list[dict[str, str | None]] = []
    for candidate in metadata.distributions():
        name = (candidate.metadata.get("Name") or "").lower().replace("_", "-")
        if name != "wearable-project":
            continue
        try:
            candidate_root = Path(str(candidate.locate_file(""))).resolve()
            candidate_package = Path(str(candidate.locate_file("wearable_project/__init__.py"))).resolve()
        except Exception:
            continue
        candidates.append({
            "version": candidate.version,
            "root": str(candidate_root),
            "package_path": str(candidate_package),
        })
    primary = candidates[0] if candidates else None
    dist_version = primary["version"] if primary else None
    dist_root = Path(primary["root"]) if primary and primary.get("root") else None
    dist_package = Path(primary["package_path"]) if primary and primary.get("package_path") else None

    roots: set[Path] = set()
    for value in site.getsitepackages() if hasattr(site, "getsitepackages") else ():
        roots.add(Path(value).resolve())
    user_site = site.getusersitepackages()
    if user_site:
        roots.add(Path(user_site).resolve())
    for key in ("purelib", "platlib"):
        value = sysconfig.get_paths().get(key)
        if value:
            roots.add(Path(value).resolve())
    site_roots = tuple(sorted(str(item) for item in roots))

    cwd = Path.cwd().resolve()
    local_checkout = (cwd / "pyproject.toml").is_file() and (cwd / "wearable_project" / "__init__.py").is_file() and (package_path == (cwd / "wearable_project" / "__init__.py").resolve())
    in_site_packages = any(_is_relative_to(package_path, root) for root in roots)
    if local_checkout:
        source_kind = "local_checkout"
    elif dist_package is not None and package_path == dist_package:
        # The imported package is exactly the package registered by the installed distribution. This is true for
        # a normal wheel install and for isolated --target validation directories, even when the target is not
        # one of Python's conventional site-packages roots.
        source_kind = "installed_distribution"
    elif in_site_packages:
        source_kind = "installed_distribution"
    elif dist_package is None:
        source_kind = "unregistered_source_tree"
    else:
        source_kind = "local_source_shadowing_installed_distribution"

    actual = {
        name: _hash(curation_root / name) for name in _TRACKED_MODULES if (curation_root / name).is_file()
    }
    try:
        from wearable_project.curation.release_manifest import (
            EXPECTED_CURATION_MODULE_SHA256,
            EXPECTED_DECISIONS_FINGERPRINT,
            EXPECTED_GUIDANCE_FINGERPRINT,
            EXPECTED_REGISTRY_FINGERPRINT,
        )
        expected = dict(EXPECTED_CURATION_MODULE_SHA256)
    except Exception:
        expected = {}
        EXPECTED_REGISTRY_FINGERPRINT = None
        EXPECTED_GUIDANCE_FINGERPRINT = None
        EXPECTED_DECISIONS_FINGERPRINT = None
    integrity = {
        name: actual.get(name) == expected_hash for name, expected_hash in expected.items()
    }
    warnings: list[str] = []
    if source_kind in {"local_checkout", "local_source_shadowing_installed_distribution"}:
        warnings.append(
            "Commands are importing the local checkout rather than a site-packages wheel. Run from outside the checkout to test the wheel itself."
        )
    mismatched = sorted(name for name, valid in integrity.items() if not valid)
    if mismatched:
        warnings.append("Curation source differs from the release manifest: " + ", ".join(mismatched))
    if dist_version is not None and dist_version != wearable_project.__version__:
        warnings.append(
            f"Imported package version {wearable_project.__version__} differs from installed distribution {dist_version}."
        )
    current_registry = registry_fingerprint()
    current_guidance = guidance_fingerprint()
    current_decisions = decisions_fingerprint()
    if EXPECTED_REGISTRY_FINGERPRINT and current_registry != EXPECTED_REGISTRY_FINGERPRINT:
        warnings.append("Registry fingerprint differs from the release manifest.")
    if EXPECTED_GUIDANCE_FINGERPRINT and current_guidance != EXPECTED_GUIDANCE_FINGERPRINT:
        warnings.append("Guidance fingerprint differs from the release manifest.")
    if EXPECTED_DECISIONS_FINGERPRINT and current_decisions != EXPECTED_DECISIONS_FINGERPRINT:
        warnings.append("Calibration-decision fingerprint differs from the release manifest.")
    git_commit, git_dirty = _git_info(source_root)
    if git_dirty:
        warnings.append(
            "The enclosing Git working tree has uncommitted changes; imported package module integrity is reported separately."
        )

    return EnvironmentManifest(
        package_version=wearable_project.__version__,
        package_release_label=getattr(wearable_project, "__release_label__", wearable_project.__version__),
        imported_package_path=str(package_path),
        current_working_directory=str(cwd),
        sys_path_head=tuple(sys.path[:5]),
        distribution_version=dist_version,
        distribution_root=str(dist_root) if dist_root is not None else None,
        distribution_package_path=str(dist_package) if dist_package is not None else None,
        distribution_candidates=tuple(candidates),
        site_package_roots=site_roots,
        import_source_kind=source_kind,
        python_version=sys.version.split()[0],
        pandas_version=pd.__version__,
        platform=platform.platform(),
        registry_fingerprint=current_registry,
        guidance_fingerprint=current_guidance,
        decisions_fingerprint=current_decisions,
        module_sha256=actual,
        expected_module_sha256=expected,
        module_integrity=integrity,
        git_commit=git_commit,
        git_dirty=git_dirty,
        warnings=tuple(warnings),
    )


def environment_json() -> str:
    return json.dumps(environment_manifest().as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
