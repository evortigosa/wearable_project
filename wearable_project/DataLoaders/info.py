"""
Wearable Data Processing and Modeling project
Introspective help for the HPP-style wearable feature loaders. This module intentionally derives feature
definitions from the same native processing registry and curation registry/guidance used by the pipeline.
It is therefore documentation *of the executable contracts*, not a second hand-maintained feature dictionary.
Typical notebook use:

    from wearable_project import DataLoaders

    DataLoaders.info()                 # package overview
    DataLoaders.info("StepCount")      # feature-specific report
    DataLoaders.info("Sleep").as_dict()

Every feature loader also exposes ``loader.info()`` with the instance's phase and root settings.
"""


from __future__ import annotations
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any
from wearable_project import __release_label__, __version__
from wearable_project.DataLoaders._base import (
    AppleHealthFeatureLoader, DEFAULT_CURATED_ROOT, DEFAULT_NATIVE_ROOT, ProcessingPhase,
)
from wearable_project.exceptions import DataLoaderConfigurationError
from wearable_project.curation.decisions import decisions_fingerprint
from wearable_project.curation.evidence import EVIDENCE
from wearable_project.curation.guidance import GUIDANCE_VERSION, get_feature_guide, guidance_fingerprint
from wearable_project.curation.registry import (
    COHORT_OBSERVED_FEATURES, CURATION_REGISTRY_VERSION, get_policy, registry_fingerprint,
)
from wearable_project.processing.registry import (
    REGISTRY_VERSION as PROCESSING_REGISTRY_VERSION, FeatureSpec, get_feature_spec,
)


INFO_CONTRACT_VERSION = "dataloader-info-1"


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


def available_features() -> tuple[str, ...]:
    """Return the 33 feature names with public DataLoader classes."""

    return tuple(COHORT_OBSERVED_FEATURES)


def info(
    feature: str | None = None, *, phase: ProcessingPhase | str = "curated",
    native_root: str | Path = DEFAULT_NATIVE_ROOT, curated_root: str | Path = DEFAULT_CURATED_ROOT,
    root: str | Path | None = None, include_evidence: bool = False,
) -> InfoReport:
    """
    Return DataLoader usage and feature-contract information.
    Parameters
    ----------
    feature:
        ``None`` returns a package-level overview. Otherwise, provide one of the 33 cohort-observed feature
        names, for example ``"StepCount"``.
    phase:
        Loader phase reflected in feature-specific examples and resolved root. ``"curated"`` is the default.
    native_root, curated_root, root:
        Same path options accepted by every feature loader. They are reported but never accessed by ``info``;
        calling this function performs no data I/O.
    include_evidence:
        Include structured evidence-catalog entries referenced by the feature. Evidence identifiers are always
        reported even when this is false.
    """

    if feature is None:
        return _overview_report(
            native_root=Path(native_root).expanduser(), curated_root=Path(curated_root).expanduser(),
        )

    canonical = _normalize_feature(feature)
    loader_cls = _loader_class(canonical)
    loader = loader_cls(
        phase=phase,
        native_root=native_root,
        curated_root=curated_root,
        root=root,
    )
    return _feature_report(loader, include_evidence=include_evidence)


def _normalize_feature(feature: str) -> str:
    value = str(feature).strip()
    if value in COHORT_OBSERVED_FEATURES:
        return value
    folded = {name.casefold(): name for name in COHORT_OBSERVED_FEATURES}
    canonical = folded.get(value.casefold())
    if canonical is None:
        raise DataLoaderConfigurationError(
            f"unknown DataLoader feature {feature!r}; choose one of: "
            + ", ".join(COHORT_OBSERVED_FEATURES)
        )
    return canonical


def _loader_class(feature: str) -> type[AppleHealthFeatureLoader]:
    module = import_module(f"wearable_project.DataLoaders.{feature}Loader")
    candidates: list[type[AppleHealthFeatureLoader]] = []
    for value in vars(module).values():
        if not isinstance(value, type):
            continue
        if value is AppleHealthFeatureLoader:
            continue
        try:
            if issubclass(value, AppleHealthFeatureLoader) and value.feature_name == feature:
                candidates.append(value)
        except AttributeError:
            continue
    if len(candidates) != 1:  # pragma: no cover - protected by release tests
        raise RuntimeError(
            f"expected exactly one public loader class for {feature}, found {len(candidates)}"
        )
    return candidates[0]


def _processing_spec_payload(spec: FeatureSpec) -> dict[str, Any]:
    unit = None
    if spec.unit_policy is not None:
        unit = {
            "raw_unit": spec.unit_policy.raw_unit,
            "canonical_unit": spec.unit_policy.canonical_unit,
            "scale": spec.unit_policy.scale,
            "offset": spec.unit_policy.offset,
            "status": spec.unit_policy.status,
            "evidence": spec.unit_policy.evidence,
        }
    return {
        "registry_version": PROCESSING_REGISTRY_VERSION,
        "feature": spec.name,
        "family": spec.family.value,
        "measurement_columns": list(spec.measurement_columns),
        "dedup_strategy": spec.dedup_strategy.value,
        "native_unit_policy": unit,
        "preserve_content_duplicates": spec.preserve_content_duplicates,
        "resolve_boundary_revisions": spec.resolve_boundary_revisions,
        "provisional": spec.provisional,
        "scope_note": (
            "Native processing preserves native-resolution events and does not perform fixed-window resampling."
        ),
    }


def _overview_report(*, native_root: Path, curated_root: Path) -> InfoReport:
    feature_rows: list[dict[str, Any]] = []
    for feature in COHORT_OBSERVED_FEATURES:
        guide = get_feature_guide(feature)
        spec = get_feature_spec(feature)
        loader_cls = _loader_class(feature)
        feature_rows.append(
            {
                "feature": feature,
                "loader": loader_cls.__name__,
                "module": f"wearable_project.DataLoaders.{feature}Loader",
                "category": guide.category,
                "processing_family": spec.family.value,
                "short_description": guide.short_description,
            }
        )

    payload: dict[str, Any] = {
        "kind": "overview",
        "info_contract_version": INFO_CONTRACT_VERSION,
        "package": {
            "version": __version__,
            "release_label": __release_label__,
        },
        "contract_versions": {
            "processing_registry": PROCESSING_REGISTRY_VERSION,
            "curation_registry": CURATION_REGISTRY_VERSION,
            "guidance": GUIDANCE_VERSION,
            "curation_registry_fingerprint": registry_fingerprint(),
            "guidance_fingerprint": guidance_fingerprint(),
            "decisions_fingerprint": decisions_fingerprint(),
        },
        "defaults": {
            "phase": "curated",
            "native_root": str(native_root),
            "curated_root": str(curated_root),
            "index_names": ["RegistrationCode", "Date"],
            "registration_code_rule": "participant folder ID prefixed with '10K_'",
        },
        "phase_semantics": {
            "native": {
                "meaning": (
                    "Native processed Apple HealthKit rows at native event/interval resolution."
                ),
                "important_behavior": [
                    "No fixed-window resampling, interpolation, imputation, or daily broadcasting.",
                    "Native feature semantics, timestamps, provenance, and reconciled record identity are retained.",
                    "Native processing unit labels describe the native processing contract and are not automatically proof of payload-explicit units.",
                ],
            },
            "curated": {
                "meaning": (
                    "Curation non-destructive curation of the native rows with policy annotations and reviewed canonical values where justified."
                ),
                "important_behavior": [
                    "All curated rows are loaded by default, including review and include_by_default=false rows.",
                    "Use default_inclusion_only=True only when an analyst explicitly wants the policy's default analytical subset.",
                    "Canonical/derived columns are intentionally sparse and may be absent in participant-feature files when not applicable.",
                    "No fixed-window resampling, interpolation, imputation, exclusive sleep hypnogram, or model tensors are created by the loader.",
                ],
            },
        },
        "usage": {
            "basic": (
                "from wearable_project.DataLoaders.StepCountLoader import StepCountLoader\n"
                "df = StepCountLoader().get_data().df"
            ),
            "native": "df = StepCountLoader(phase='native').get_data().df",
            "root_override": (
                "df = StepCountLoader(root='/path/to/sample').get_data().df"
            ),
            "participant_filter": (
                "df = StepCountLoader().get_data(registration_codes='10K_1235738253').df"
            ),
            "date_filter": (
                "df = HeartRateLoader().get_data(start_date='2024-01-01', end_date='2024-12-31').df"
            ),
            "curated_default_subset": (
                "df = WeightLoader().get_data(default_inclusion_only=True).df"
            ),
            "feature_help": (
                "from wearable_project import DataLoaders\n"
                "print(DataLoaders.info('StepCount'))"
            ),
        },
        "loading_behavior": {
            "dates": "Date filters are inclusive and normalized to UTC.",
            "column_projection": (
                "columns=[...] projects analytical columns while retaining the hidden date anchor needed to build the HPP index."
            ),
            "sparse_columns": (
                "Curated columns requested but absent for one participant are filled with NA when they exist elsewhere for the feature."
            ),
            "missing_requested_participant": (
                "Returns an empty HPP-indexed frame and reports the missing participant in LoaderData.metadata rather than failing."
            ),
            "storage": "DataLoaders are read-only and never modify native or curated roots.",
        },
        "scalability": {
            "materialization": (
                "get_data().df materializes the selected feature rows in pandas memory."
            ),
            "recommendation": (
                "For very large features such as full-cohort HeartRate, restrict participants/date ranges/columns until a future lazy or chunked backend is added."
            ),
            "no_hidden_sampling": "Loaders never silently sample, truncate, aggregate, or downsample rows.",
        },
        "feature_count": len(feature_rows),
        "features": feature_rows,
    }

    lines = [
        "Wearable DataLoaders",
        f"Package: {__version__} ({__release_label__})",
        f"Features: {len(feature_rows)}",
        "Default phase: curated",
        "Index: RegistrationCode, Date",
        "",
        "Core contract:",
        "  - FeatureLoader().get_data().df returns an HPP-style pandas DataFrame.",
        "  - Curated rows are never silently filtered; review/excluded rows remain visible by default.",
        "  - Loaders are read-only and do not resample, interpolate, impute, or aggregate.",
        "  - Date filters are inclusive UTC bounds.",
        "  - Full-cohort large features can require substantial RAM; filter explicitly when needed.",
        "",
        "Default roots:",
        f"  native:  {native_root}",
        f"  curated: {curated_root}",
        "",
        "Usage:",
        "  from wearable_project.DataLoaders.StepCountLoader import StepCountLoader",
        "  df = StepCountLoader().get_data().df",
        "  df = StepCountLoader(phase='native').get_data().df",
        "  df = StepCountLoader(root='/path/to/sample').get_data().df",
        "",
        "Feature-specific help:",
        "  from wearable_project import DataLoaders",
        "  print(DataLoaders.info('StepCount'))",
        "  payload = DataLoaders.info('StepCount').as_dict()",
        "",
        "Available features:",
    ]
    for row in feature_rows:
        lines.append(f"  {row['feature']}: {row['processing_family']} — {row['short_description']}")
    return InfoReport(
        kind="overview",
        title="Wearable DataLoaders",
        payload=payload,
        text="\n".join(lines),
    )


def _feature_report(loader: AppleHealthFeatureLoader, *, include_evidence: bool,) -> InfoReport:
    feature = loader.feature_name
    guide = get_feature_guide(feature)
    processing_spec = get_feature_spec(feature)
    policy = get_policy(feature, allow_fallback=False)
    policy_payload = policy.as_dict()

    evidence_ids = list(
        dict.fromkeys(
            list(guide.evidence_refs)
            + list(policy.evidence_refs)
            + [ref for unit in policy.units.measurements for ref in unit.evidence_refs]
        )
    )
    evidence_payload = None
    if include_evidence:
        evidence_payload = {
            evidence_id: asdict(EVIDENCE[evidence_id]) for evidence_id in evidence_ids if evidence_id in EVIDENCE
        }

    import_path = (
        f"from wearable_project.DataLoaders.{feature}Loader import {loader.__class__.__name__}"
    )
    ctor = loader.__class__.__name__
    root_literal = str(loader.data_root)

    payload: dict[str, Any] = {
        "kind": "feature",
        "info_contract_version": INFO_CONTRACT_VERSION,
        "package": {
            "version": __version__,
            "release_label": __release_label__,
        },
        "feature": feature,
        "loader": {
            "class": ctor,
            "module": f"wearable_project.DataLoaders.{feature}Loader",
            "phase": loader.phase,
            "resolved_root": root_literal,
            "filename": loader.filename,
            "index_names": loader._data_index_names,
            "registration_prefix": loader.registration_prefix,
            "date_column": loader.date_column,
            "date_semantics": loader.date_semantics,
            "read_only": True,
        },
        "guidance": guide.as_dict(),
        "processing": _processing_spec_payload(processing_spec),
        "curation": {
            "registry_version": CURATION_REGISTRY_VERSION,
            "policy_fingerprint": policy.fingerprint(),
            "policy": policy_payload,
            "scope_note": (
                "Curated files preserve native rows and add policy annotations/derived values; the DataLoader does not execute curation."
            ),
        },
        "loader_behavior": {
            "all_curated_rows_by_default": True,
            "default_inclusion_filter": (
                "Pass default_inclusion_only=True explicitly for phase='curated'."
            ),
            "date_filtering": "Inclusive UTC start_date/end_date bounds on the loader Date anchor.",
            "column_projection": "Pass columns=[...] to reduce returned analytical columns.",
            "sparse_curated_columns": (
                "Requested curated derived columns can be NA for participant files where they were not applicable."
            ),
            "resampling": (
                "The curation registry may declare future resampling semantics, but this loader does not resample data."
            ),
            "memory": (
                "get_data().df materializes selected rows in pandas memory; use participant/date/column filters for large features."
            ),
        },
        "usage": {
            "import": import_path,
            "current_phase": f"df = {ctor}(phase={loader.phase!r}).get_data().df",
            "current_root": f"df = {ctor}(phase={loader.phase!r}, root={root_literal!r}).get_data().df",
            "participant": (
                f"df = {ctor}().get_data(registration_codes='10K_1235738253').df"
            ),
            "date_range": (
                f"df = {ctor}().get_data(start_date='2024-01-01', end_date='2024-12-31').df"
            ),
            "columns": (
                f"df = {ctor}().get_data(columns={list(processing_spec.measurement_columns)!r}).df"
            ),
            "default_curated_subset": (
                f"df = {ctor}().get_data(default_inclusion_only=True).df"
            ),
        },
        "evidence_refs": evidence_ids,
        "evidence": evidence_payload,
        "contract_versions": {
            "processing_registry": PROCESSING_REGISTRY_VERSION,
            "curation_registry": CURATION_REGISTRY_VERSION,
            "guidance": GUIDANCE_VERSION,
            "curation_registry_fingerprint": registry_fingerprint(),
            "guidance_fingerprint": guidance_fingerprint(),
            "decisions_fingerprint": decisions_fingerprint(),
        },
    }

    native_unit = _native_unit_text(processing_spec)
    resampling = policy_payload["resampling"]
    calibration = policy_payload["calibration"]
    lines = [
        f"{feature} — {ctor}",
        f"Category: {guide.category}",
        f"Phase: {loader.phase}",
        f"Root: {loader.data_root}",
        f"Index: {', '.join(loader._data_index_names)}",
        f"Date anchor: {loader.date_column} ({loader.date_semantics})",
        "",
        "Definition:",
        f"  {guide.short_description}",
        "",
        "One row means:",
        f"  {guide.one_row_means}",
        "",
        "Native time semantics:",
        f"  {guide.native_time_semantics}",
        "",
        "Units:",
        f"  Guidance: {guide.unit_summary}",
        f"  Native processing registry: {native_unit}",
        "",
        "Acquisition/provenance:",
        f"  {guide.acquisition_summary}",
        "",
        "Curation:",
        f"  {guide.curation_summary}",
        f"  Policy maturity: {policy_payload['identity']['maturity']}",
        f"  Default status: {policy_payload['curation']['default_status']}",
        f"  Default inclusion: {policy_payload['curation']['default_inclusion']}",
        f"  Execution mode: {calibration['execution_mode']}",
        "",
        "Loader behavior:",
        "  All curated rows are returned by default; review/excluded rows are not silently dropped.",
        "  default_inclusion_only=True is an explicit curated-only filter.",
        "  Date filters are inclusive UTC bounds.",
        "  columns=[...] can reduce memory use; no hidden sampling or resampling occurs.",
        "",
        "Future resampling declaration (not executed by this loader):",
        f"  support={resampling['support']}; strategy={resampling['strategy']}; aggregation={resampling['aggregation']}",
        "",
        "Important caveats:",
    ]
    if guide.important_caveats:
        lines.extend(f"  - {item}" for item in guide.important_caveats)
    else:
        lines.append("  - None recorded in feature guidance.")
    if guide.not_equivalent_to:
        lines.extend(("", "Do not interpret as:"))
        lines.extend(f"  - {item}" for item in guide.not_equivalent_to)

    lines.extend(
        (
            "",
            "Usage:",
            f"  {import_path}",
            f"  df = {ctor}(phase={loader.phase!r}).get_data().df",
            f"  df = {ctor}(root='/path/to/sample').get_data(registration_codes='10K_1235738253').df",
            "",
            "Evidence references:",
        )
    )
    lines.extend(f"  - {evidence_id}" for evidence_id in evidence_ids)
    if include_evidence and evidence_payload:
        lines.extend(("", "Evidence details:"))
        for evidence_id, source in evidence_payload.items():
            lines.append(f"  {evidence_id}: {source['title']} — {source['organization']} ({source['kind']})")
            if source.get("brief_summary"):
                lines.append(f"    {source['brief_summary']}")
            if source.get("limitations"):
                lines.append(f"    Limitations: {source['limitations']}")
            if source.get("locator"):
                lines.append(f"    {source['locator']}")

    return InfoReport(
        kind="feature",
        title=f"{feature} DataLoader information",
        payload=payload,
        text="\n".join(lines),
    )


def _native_unit_text(spec: FeatureSpec) -> str:
    unit = spec.unit_policy
    if unit is None:
        return "no fixed native processing unit conversion; interpretation remains feature/source dependent"
    if unit.raw_unit == unit.canonical_unit and unit.scale == 1.0 and unit.offset == 0.0:
        return f"{unit.raw_unit} ({unit.status})"
    return (
        f"{unit.raw_unit} -> {unit.canonical_unit}; scale={unit.scale:g}, offset={unit.offset:g} ({unit.status})"
    )
