"""
Wearable Data Processing and Modeling project
Source-aware static documentation for wearable feature DataLoaders.
``info`` performs no filesystem access.  It dispatches to source-specific executable contracts while preserving the
production Apple HealthKit report byte-for-byte.  Use ``profile()`` for root-specific inventories and counts.
"""


from __future__ import annotations
import sys
import types
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any
from wearable_project.DataLoaders._base import (
    AppleHealthFeatureLoader, DEFAULT_CURATED_ROOT, DEFAULT_NATIVE_ROOT, ProcessingPhase,
)
from wearable_project.DataLoaders._contracts import InfoDocument
from wearable_project.DataLoaders._sources import (
    WearableSource, features_for_source, get_source_adapter, normalize_source, sources_for_feature,
)
from wearable_project.exceptions import DataLoaderConfigurationError


@dataclass(frozen=True, slots=True)
class InfoReport:
    """Human-readable plus machine-readable DataLoader documentation."""

    kind: str
    title: str
    payload: dict[str, Any]
    text: str

    def as_dict(self) -> dict[str, Any]:
        """Return the structured report payload."""

        return self.payload

    def __str__(self) -> str:
        return self.text

    def _repr_markdown_(self) -> str:  # pragma: no cover - notebook display hook
        return "```text\n" + self.text + "\n```"


def _public_report(document: InfoDocument) -> InfoReport:
    return InfoReport(
        kind=document.kind,
        title=document.title,
        payload=document.payload,
        text=document.text,
    )


def available_features(source: WearableSource | str | None = None) -> tuple[str, ...]:
    """
    Return the accepted feature catalog for ``source``. Omitting ``source`` preserves the production Apple
    HealthKit catalog; it does not return a cross-source union.
    """

    canonical = normalize_source(source, default=WearableSource.APPLE_HEALTHKIT)
    return features_for_source(canonical)


def available_sources(feature: str) -> tuple[str, ...]:
    """Return canonical source names that provide ``feature``; matching is case-insensitive."""

    value = str(feature).strip()
    union = {name.casefold(): name for source in WearableSource for name in features_for_source(source)}
    canonical = union.get(value.casefold())
    if canonical is None:
        raise DataLoaderConfigurationError(
            f"unknown DataLoader feature {feature!r}; choose a feature from available_features(source=...)"
        )
    return tuple(source.value for source in sources_for_feature(canonical))


def _normalize_feature(feature: str, *, source: WearableSource | str) -> str:
    names = features_for_source(source)
    value = str(feature).strip()
    if value in names:
        return value
    folded = {name.casefold(): name for name in names}
    canonical = folded.get(value.casefold())
    if canonical is None:
        raise DataLoaderConfigurationError(
            f"unknown DataLoader feature {feature!r} for source {normalize_source(source).value!r}; choose one of: "
            + ", ".join(names)
        )
    return canonical


def _loader_class(feature: str) -> type[AppleHealthFeatureLoader]:
    module = import_module(f"wearable_project.DataLoaders.{feature}Loader")
    candidates: list[type[AppleHealthFeatureLoader]] = []
    for value in vars(module).values():
        if not isinstance(value, type) or value is AppleHealthFeatureLoader:
            continue
        try:
            if issubclass(value, AppleHealthFeatureLoader) and value.feature_name == feature:
                candidates.append(value)
        except AttributeError:
            continue
    if len(candidates) != 1:  # pragma: no cover - protected by release tests
        raise RuntimeError(f"expected exactly one public loader class for {feature}, found {len(candidates)}")
    return candidates[0]


def _resolved_roots(
    source: WearableSource, native_root: str | Path, curated_root: str | Path,
) -> tuple[Path, Path]:
    """Apply the same legacy-Apple-default sentinel behavior as loader construction, without touching disk."""

    adapter = get_source_adapter(source)
    native = Path(native_root).expanduser()
    curated = Path(curated_root).expanduser()
    if native == DEFAULT_NATIVE_ROOT and adapter.native_root != DEFAULT_NATIVE_ROOT:
        native = adapter.native_root
    if curated == DEFAULT_CURATED_ROOT and adapter.curated_root != DEFAULT_CURATED_ROOT:
        curated = adapter.curated_root
    return native, curated


def info(
    feature: str | None = None, *, phase: ProcessingPhase | str = "curated",
    native_root: str | Path = DEFAULT_NATIVE_ROOT, curated_root: str | Path = DEFAULT_CURATED_ROOT,
    root: str | Path | None = None, include_evidence: bool = False, source: WearableSource | str | None = None,
) -> InfoReport:
    """Return source-qualified static DataLoader usage and scientific-contract information."""

    canonical_source = normalize_source(source, default=WearableSource.APPLE_HEALTHKIT)
    resolved_native, resolved_curated = _resolved_roots(canonical_source, native_root, curated_root)
    if feature is None:
        document = get_source_adapter(canonical_source).overview_info(
            native_root=resolved_native, curated_root=resolved_curated,
        )
        return _public_report(document)

    canonical = _normalize_feature(feature, source=canonical_source)
    loader_cls = _loader_class(canonical)
    loader = loader_cls(
        phase=phase,
        native_root=native_root,
        curated_root=curated_root,
        root=root,
        source=canonical_source,
    )
    return _feature_report(loader, include_evidence=include_evidence)


def _feature_report(loader: AppleHealthFeatureLoader, *, include_evidence: bool) -> InfoReport:
    """Dispatch a configured loader to its source-specific, no-I/O static report builder."""

    return _public_report(
        loader._source_adapter.feature_info(loader, include_evidence=include_evidence)
    )


class _CallableModule(types.ModuleType):
    def __call__(self, *args: Any, **kwargs: Any) -> InfoReport:
        return info(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
