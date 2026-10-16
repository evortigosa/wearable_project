"""
Wearable Data Processing and Modeling project
Google Fitness WP4 processing environment and source-integrity report.
"""


from __future__ import annotations
import hashlib
import subprocess
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any
import wearable_project
from wearable_project import __release_label__, __version__
from wearable_project.processing.google_fitness.cleaners import CLEANER_VERSION
from wearable_project.processing.google_fitness.models import PARSER_VERSION
from wearable_project.processing.google_fitness.pipeline import PIPELINE_VERSION
from wearable_project.processing.google_fitness.registry import REGISTRY_VERSION, registry_fingerprint
from wearable_project.utils.release_manifest import GFIT_EXPECTED_MODULE_SHA256
from wearable_project.processing.google_fitness.writer import STATE_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class ProcessingEnvironmentManifest:
    package_version: str
    package_release_label: str
    current_working_directory: str
    imported_package_path: str
    distribution_package_path: str | None
    import_source_kind: str
    parser_version: str
    registry_version: str
    registry_fingerprint: str
    cleaner_version: str
    pipeline_version: str
    state_schema_version: str
    core_module_integrity: dict[str, bool]
    core_module_sha256: dict[str, str | None]
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
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def processing_environment_manifest() -> ProcessingEnvironmentManifest:
    package_root = Path(wearable_project.__file__).resolve().parent
    source_root = package_root / "processing" / "google_fitness"
    repository_root = package_root.parent
    distribution_path: str | None = None
    try:
        distribution = metadata.distribution("wearable-project")
        distribution_path = str(Path(distribution.locate_file("")))
    except metadata.PackageNotFoundError:
        distribution = None

    actual: dict[str, str | None] = {}
    integrity: dict[str, bool] = {}
    for name, expected in sorted(GFIT_EXPECTED_MODULE_SHA256.items()):
        path = source_root / name
        observed = _hash(path) if path.is_file() else None
        actual[name] = observed
        integrity[name] = observed == expected

    commit, dirty = _git_info(repository_root)
    warnings: list[str] = []
    invalid = [name for name, valid in integrity.items() if not valid]
    if invalid:
        warnings.append("Google Fitness WP4 source differs from its release manifest: " + ", ".join(invalid))
    if registry_fingerprint() == "":
        warnings.append("Google Fitness registry fingerprint is empty")
    imported = Path(wearable_project.__file__).resolve()
    if distribution_path is None:
        source_kind = "local_checkout"
    else:
        dist_root = Path(distribution_path).resolve()
        try:
            imported.relative_to(dist_root)
            source_kind = "installed_distribution"
        except ValueError:
            source_kind = "local_checkout"
            warnings.append(
                "wearable_project is imported from a local checkout rather than the installed distribution"
            )

    return ProcessingEnvironmentManifest(
        package_version=__version__,
        package_release_label=__release_label__,
        current_working_directory=str(Path.cwd()),
        imported_package_path=str(imported),
        distribution_package_path=distribution_path,
        import_source_kind=source_kind,
        parser_version=PARSER_VERSION,
        registry_version=REGISTRY_VERSION,
        registry_fingerprint=registry_fingerprint(),
        cleaner_version=CLEANER_VERSION,
        pipeline_version=PIPELINE_VERSION,
        state_schema_version=STATE_SCHEMA_VERSION,
        core_module_integrity=integrity,
        core_module_sha256=actual,
        git_commit=commit,
        git_dirty=dirty,
        warnings=tuple(warnings),
    )
