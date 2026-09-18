"""
Wearable Data Processing and Modeling project
Reproducibility diagnostics for Phase 1 native processing. The native processing state stores semantic
parser and registry versions. This module reports the code that is currently imported so users can
distinguish a real state-version mismatch from an invocation mismatch such as:
    python -m wearable_project process       # local checkout on sys.path
    wearable-project process-scan            # installed wheel console script
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import platform
import site
import subprocess
import sys
import sysconfig
from typing import Any, Mapping
import wearable_project
from wearable_project.processing.pipeline import PARSER_VERSION
from wearable_project.processing.registry import REGISTRY_VERSION
from wearable_project.utils.release_manifest import EXPECTED_CORE_PROCESSING_MODULE_SHA256


_TRACKED_CORE_MODULES = (
    "__init__.py",
    "cleaners.py",
    "parser.py",
    "pipeline.py",
    "registry.py",
    "resampling.py",
    "tracker.py",
    "writer.py",
)

@dataclass(frozen=True, slots=True)
class ProcessingEnvironmentManifest:
    package_version: str
    package_release_label: str
    imported_package_path: str
    current_working_directory: str
    sys_path_head: tuple[str, ...]
    distribution_version: str | None
    distribution_root: str | None
    distribution_package_path: str | None
    distribution_candidates: tuple[Mapping[str, str | None], ...]
    import_source_kind: str
    python_version: str
    platform: str
    parser_version: str
    registry_version: str
    core_module_sha256: Mapping[str, str]
    expected_core_module_sha256: Mapping[str, str]
    core_module_integrity: Mapping[str, bool]
    git_commit: str | None
    git_dirty: bool | None
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


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


def processing_environment_manifest() -> ProcessingEnvironmentManifest:
    module_file = wearable_project.__file__
    if module_file is None:
        raise RuntimeError("wearable_project has no __file__ (namespace package?)")
    package_path = Path(module_file).resolve()
    package_root = package_path.parent
    processing_root = package_root / "processing"
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

    cwd = Path.cwd().resolve()
    local_checkout = (
        (cwd / "pyproject.toml").is_file()
        and (cwd / "wearable_project" / "__init__.py").is_file()
        and package_path == (cwd / "wearable_project" / "__init__.py").resolve()
    )
    in_site_packages = any(_is_relative_to(package_path, root) for root in roots)
    if local_checkout:
        source_kind = "local_checkout"
    elif dist_package is not None and package_path == dist_package:
        source_kind = "installed_distribution"
    elif in_site_packages:
        source_kind = "installed_distribution"
    elif dist_package is None:
        source_kind = "unregistered_source_tree"
    else:
        source_kind = "local_source_shadowing_installed_distribution"

    actual = {
        name: _hash(processing_root / name)
        for name in _TRACKED_CORE_MODULES if (processing_root / name).is_file()
    }
    expected = dict(EXPECTED_CORE_PROCESSING_MODULE_SHA256)
    integrity = {
        name: actual.get(name) == expected_hash
        for name, expected_hash in expected.items()
    }

    warnings: list[str] = []
    if source_kind in {"local_checkout", "local_source_shadowing_installed_distribution"}:
        warnings.append(
            "Commands are importing a local checkout rather than the installed wheel. "
            "Using `python -m wearable_project ...` from a repository can therefore differ "
            "from the `wearable-project ...` console script."
        )
    mismatched = sorted(name for name, valid in integrity.items() if not valid)
    if mismatched:
        warnings.append(
            "Core native processing source differs from the frozen release: "
            + ", ".join(mismatched)
        )
    if dist_version is not None and dist_version != wearable_project.__version__:
        warnings.append(
            f"Imported package version {wearable_project.__version__} differs from "
            f"installed distribution {dist_version}."
        )

    git_commit, git_dirty = _git_info(source_root)
    if git_dirty and source_kind != "installed_distribution":
        warnings.append("The imported processing source is from a Git working tree with uncommitted changes.")

    return ProcessingEnvironmentManifest(
        package_version=wearable_project.__version__,
        package_release_label=getattr(wearable_project, "__release_label__", wearable_project.__version__),
        imported_package_path=str(package_path),
        current_working_directory=str(cwd),
        sys_path_head=tuple(sys.path[:5]),
        distribution_version=dist_version,
        distribution_root=str(dist_root) if dist_root is not None else None,
        distribution_package_path=str(dist_package) if dist_package is not None else None,
        distribution_candidates=tuple(candidates),
        import_source_kind=source_kind,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        parser_version=PARSER_VERSION,
        registry_version=REGISTRY_VERSION,
        core_module_sha256=actual,
        expected_core_module_sha256=expected,
        core_module_integrity=integrity,
        git_commit=git_commit,
        git_dirty=git_dirty,
        warnings=tuple(warnings),
    )
