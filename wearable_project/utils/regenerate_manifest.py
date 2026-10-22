"""
Wearable Data Processing and Modeling project
Regenerate the centralized release-integrity manifest from the current source tree. Run after changing any tracked
Apple HealthKit or Google Fitness processing module, any tracked Apple curation module, or the package version::

    python -m wearable_project.utils.regenerate_manifest --write
    python -m wearable_project.utils.regenerate_manifest --check

For backward convenience, omitting both ``--write`` and ``--check`` also writes the manifest. The generated
``utils/release_manifest.py`` contains:

- hashes for the Apple HealthKit curation modules tracked by
  ``curation.apple_healthkit.environment._TRACKED_MODULES``;
- hashes for the Apple HealthKit processing modules tracked by
  ``processing.apple_healthkit.environment._TRACKED_CORE_MODULES``;
- hashes for the Google Fitness processing modules tracked by
  ``processing.google_fitness.environment._TRACKED_CORE_MODULES``;
- Apple curation registry, guidance, and calibration-decision fingerprints;
- the package version and release label.

The hashes certify exactly the source tree present when this command runs. Review all intended source changes
before writing the manifest. ``--check`` writes nothing and exits with status 1 when the manifest is out of
date, so it can be used as a release/CI guard.
"""


from __future__ import annotations
import argparse
import hashlib
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


_SCALAR_NAMES = (
    "EXPECTED_PACKAGE_VERSION",
    "EXPECTED_RELEASE_LABEL",
    "EXPECTED_REGISTRY_FINGERPRINT",
    "EXPECTED_GUIDANCE_FINGERPRINT",
    "EXPECTED_DECISIONS_FINGERPRINT",
    "EXPECTED_GFIT_CURATION_REGISTRY_FINGERPRINT",
    "EXPECTED_GFIT_CURATION_EVIDENCE_FINGERPRINT",
    "EXPECTED_GFIT_CURATION_RULES_FINGERPRINT",
    "EXPECTED_GFIT_CURATION_DECISIONS_FINGERPRINT",
    "EXPECTED_GFIT_CURATION_GUIDANCE_FINGERPRINT",
)
_MAPPING_NAMES = (
    "EXPECTED_CURATION_MODULE_SHA256",
    "EXPECTED_CORE_PROCESSING_MODULE_SHA256",
    "EXPECTED_GFIT_MODULE_SHA256",
    "EXPECTED_GFIT_CURATION_MODULE_SHA256",
)
_MANIFEST_NAMES = frozenset((*_SCALAR_NAMES, *_MAPPING_NAMES))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_hashes(root: Path, names: Iterable[str], *, label: str) -> dict[str, str]:
    tracked = tuple(sorted(set(names)))
    missing = [name for name in tracked if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"tracked {label} modules are missing: {missing}")
    return {name: _sha256(root / name) for name in tracked}


def compute(release_label: str | None = None) -> dict[str, Any]:
    """Return the current value for every centralized manifest constant."""

    import wearable_project
    from wearable_project.curation.apple_healthkit import environment as curation_environment
    from wearable_project.curation.apple_healthkit.decisions import decisions_fingerprint
    from wearable_project.curation.apple_healthkit.guidance import guidance_fingerprint
    from wearable_project.curation.apple_healthkit.registry import registry_fingerprint
    from wearable_project.processing.apple_healthkit import environment as apple_processing_environment
    from wearable_project.processing.google_fitness import environment as google_processing_environment
    from wearable_project.curation.google_fitness import environment as google_curation_environment
    from wearable_project.curation.google_fitness.decisions import decisions_fingerprint as gfit_decisions_fingerprint
    from wearable_project.curation.google_fitness.evidence import evidence_fingerprint as gfit_evidence_fingerprint
    from wearable_project.curation.google_fitness.guidance import guidance_fingerprint as gfit_guidance_fingerprint
    from wearable_project.curation.google_fitness.registry import registry_fingerprint as gfit_registry_fingerprint
    from wearable_project.curation.google_fitness.rules import rules_fingerprint as gfit_rules_fingerprint

    module_file = wearable_project.__file__
    if module_file is None:
        raise RuntimeError("wearable_project has no __file__ (namespace package?)")
    package = Path(module_file).resolve().parent

    return {
        "EXPECTED_PACKAGE_VERSION": wearable_project.__version__,
        "EXPECTED_RELEASE_LABEL": release_label or wearable_project.__version__,
        "EXPECTED_REGISTRY_FINGERPRINT": registry_fingerprint(),
        "EXPECTED_GUIDANCE_FINGERPRINT": guidance_fingerprint(),
        "EXPECTED_DECISIONS_FINGERPRINT": decisions_fingerprint(),
        "EXPECTED_GFIT_CURATION_REGISTRY_FINGERPRINT": gfit_registry_fingerprint(),
        "EXPECTED_GFIT_CURATION_EVIDENCE_FINGERPRINT": gfit_evidence_fingerprint(),
        "EXPECTED_GFIT_CURATION_RULES_FINGERPRINT": gfit_rules_fingerprint(),
        "EXPECTED_GFIT_CURATION_DECISIONS_FINGERPRINT": gfit_decisions_fingerprint(),
        "EXPECTED_GFIT_CURATION_GUIDANCE_FINGERPRINT": gfit_guidance_fingerprint(),
        "EXPECTED_CURATION_MODULE_SHA256": _module_hashes(
            package / "curation" / "apple_healthkit",
            curation_environment._TRACKED_MODULES,
            label="Apple HealthKit curation",
        ),
        "EXPECTED_CORE_PROCESSING_MODULE_SHA256": _module_hashes(
            package / "processing" / "apple_healthkit",
            apple_processing_environment._TRACKED_CORE_MODULES,
            label="Apple HealthKit processing",
        ),
        "EXPECTED_GFIT_MODULE_SHA256": _module_hashes(
            package / "processing" / "google_fitness",
            google_processing_environment._TRACKED_CORE_MODULES,
            label="Google Fitness processing",
        ),
        "EXPECTED_GFIT_CURATION_MODULE_SHA256": _module_hashes(
            package / "curation" / "google_fitness",
            google_curation_environment._TRACKED_MODULES,
            label="Google Fitness curation",
        ),
    }


def _dict_block(name: str, mapping: Mapping[str, str]) -> str:
    lines = [f"{name} = {{"]
    lines.extend(f'    "{key}": "{mapping[key]}",' for key in sorted(mapping))
    lines.append("}")
    return "\n".join(lines)


def render_manifest(values: Mapping[str, Any]) -> str:
    missing = sorted(_MANIFEST_NAMES - set(values))
    extra = sorted(set(values) - _MANIFEST_NAMES)
    if missing or extra:
        raise ValueError(f"invalid manifest value set; missing={missing}, extra={extra}")

    scalars = "\n".join(f'{name} = "{values[name]}"' for name in _SCALAR_NAMES)
    mappings = "\n\n".join(_dict_block(name, values[name]) for name in _MAPPING_NAMES)
    return (
        '"""\n'
        "Wearable Data Processing and Modeling project\n"
        f"Generated integrity manifest for the {values['EXPECTED_RELEASE_LABEL']} release.\n"
        "Regenerate with `python -m wearable_project.utils.regenerate_manifest --write`; do not edit by hand.\n"
        '"""\n\n\n'
        "from __future__ import annotations\n\n\n"
        f"{scalars}\n\n"
        f"{mappings}\n"
    )


def _current_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    namespace: dict[str, Any] = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return {name: namespace[name] for name in _MANIFEST_NAMES if name in namespace}


def _short(value: Any) -> str:
    return "<missing>" if value is None else str(value)[:16]


def _differences(current: Mapping[str, Any], computed: Mapping[str, Any]) -> list[str]:
    changes: list[str] = []
    for name in sorted(set(current) | set(computed)):
        if name not in computed:
            changes.append(f"{name}: {_short(current[name])} -> <removed>")
            continue
        if name not in current:
            changes.append(f"{name}: <missing> -> {_short(computed[name])}")
            continue

        new_value = computed[name]
        old_value = current[name]
        if isinstance(new_value, Mapping):
            old_mapping = old_value if isinstance(old_value, Mapping) else {}
            for module in sorted(set(new_value) | set(old_mapping)):
                old_hash = old_mapping.get(module)
                new_hash = new_value.get(module)
                if old_hash != new_hash:
                    changes.append(
                        f"{name}[{module}]: {_short(old_hash)} -> {_short(new_hash)}"
                    )
        elif old_value != new_value:
            changes.append(f"{name}: {_short(old_value)} -> {_short(new_value)}")
    return changes


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the centralized release-integrity manifest.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write the updated manifest (default mode)")
    mode.add_argument("--check", action="store_true", help="report differences and write nothing")
    parser.add_argument("--release-label", help="release label, defaulting to the package version")
    args = parser.parse_args(argv)

    import wearable_project

    module_file = wearable_project.__file__
    if module_file is None:
        raise RuntimeError("wearable_project has no __file__ (namespace package?)")
    package = Path(module_file).resolve().parent
    manifest_path = package / "utils" / "release_manifest.py"
    computed = compute(args.release_label)
    changes = _differences(_current_manifest(manifest_path), computed)

    if not changes:
        print("release manifest is up to date")
        return 0

    print(f"{len(changes)} value(s) differ from the current sources:")
    for line in changes:
        print(f"  {line}")
    if args.check:
        return 1

    _atomic_write(manifest_path, render_manifest(computed))
    remaining = _differences(_current_manifest(manifest_path), computed)
    if remaining:
        raise RuntimeError(f"written manifest did not verify: {remaining}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
