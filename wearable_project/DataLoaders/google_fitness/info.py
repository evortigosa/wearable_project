"""
Wearable Data Processing and Modeling project
Static, source-qualified documentation for Google Fitness native and curated DataLoaders.
The report is generated entirely from installed executable contracts and performs no filesystem access. Use
``profile()`` for root-specific inventories and counts.
"""


from __future__ import annotations
from pathlib import Path
import textwrap
from typing import Any
from wearable_project import __release_label__, __version__
from wearable_project.DataLoaders._contracts import InfoDocument
from wearable_project.DataLoaders._profile import ACQUISITION_CAVEAT
from wearable_project.DataLoaders._sources import GOOGLE_FITNESS_PUBLIC_FEATURES


INFO_CONTRACT_VERSION = "dataloader-info-google-fitness-1"
_WIDTH = 100
_EXAMPLE_CODE = "10K_1235738253"


def _wrap(text: object, indent: str = "  ", width: int = _WIDTH) -> list[str]:
    return textwrap.wrap(
        str(text), width=width, initial_indent=indent, subsequent_indent=indent + "  ",
    ) or [indent]


def _primitive(value: Any) -> Any:
    from wearable_project.curation.google_fitness.models import to_primitive

    return to_primitive(value)


def _contract_versions() -> dict[str, Any]:
    from wearable_project.DataLoaders.google_fitness.schema import (
        CURATED_DATALOADER_SCHEMA_VERSION, SCHEMA_VERSION,
    )
    from wearable_project.curation.google_fitness.contract import (
        CONTRACT_SCHEMA_VERSION, CURATED_SCHEMA_VERSION, contract_fingerprint, decisions_fingerprint,
        evidence_fingerprint, guidance_fingerprint, registry_fingerprint, rules_fingerprint,
    )
    from wearable_project.curation.google_fitness.registry import (
        CURATION_REGISTRY_VERSION, POLICY_CONTRACT_VERSION,
    )
    from wearable_project.curation.google_fitness.state import CURATION_STATE_SCHEMA_VERSION
    from wearable_project.processing.google_fitness.registry import REGISTRY_VERSION
    from wearable_project.processing.google_fitness.writer import STATE_SCHEMA_VERSION

    return {
        "package_version": __version__,
        "release_label": __release_label__,
        "info_contract_version": INFO_CONTRACT_VERSION,
        "native_dataloader_schema_version": SCHEMA_VERSION,
        "curated_dataloader_schema_version": CURATED_DATALOADER_SCHEMA_VERSION,
        "native_registry_version": REGISTRY_VERSION,
        "native_state_schema_version": STATE_SCHEMA_VERSION,
        "curation_registry_version": CURATION_REGISTRY_VERSION,
        "policy_contract_version": POLICY_CONTRACT_VERSION,
        "curated_schema_version": CURATED_SCHEMA_VERSION,
        "curation_contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "curation_state_schema_version": CURATION_STATE_SCHEMA_VERSION,
        "curation_contract_fingerprint": contract_fingerprint(),
        "curation_registry_fingerprint": registry_fingerprint(),
        "curation_rules_fingerprint": rules_fingerprint(),
        "curation_decisions_fingerprint": decisions_fingerprint(),
        "curation_evidence_fingerprint": evidence_fingerprint(),
        "curation_guidance_fingerprint": guidance_fingerprint(),
    }


def _evidence_payload(evidence_ids: tuple[str, ...], include_evidence: bool) -> dict[str, Any]:
    if not include_evidence:
        return {}
    from wearable_project.curation.google_fitness.evidence import EVIDENCE

    return {
        evidence_id: EVIDENCE[evidence_id].as_dict()
        for evidence_id in evidence_ids
        if evidence_id in EVIDENCE
    }


def _projection_payload(schema: Any, available: tuple[str, ...]) -> dict[str, Any]:
    return {
        name: {
            "description": schema.describe_projection(name),
            "columns": schema.columns_for_projection(name, available),
        }
        for name in schema.available_projections()
    }


def _date_anchor_summary() -> dict[str, str]:
    return {
        "activity_intervals": "start_date",
        "sleep": "start_date",
        "calories": "start_date",
        "distance": "start_date",
        "weight": "end_date",
        "height": "end_date",
        "step_count": "outer_datetime",
    }


def overview_report(*, native_root: Path, curated_root: Path) -> InfoDocument:
    """Describe the complete Google Fitness public loader catalog without touching either root."""

    from wearable_project.curation.google_fitness.contract import (
        MATERIALIZED_COLUMN_ORDER, ROW_DEFAULTS, VIRTUAL_COLUMN_ORDER,
    )
    from wearable_project.DataLoaders.google_fitness.schema import (
        CURATED_PROJECTION_SUMMARIES, NATIVE_PROJECTION_SUMMARIES,
    )
    from wearable_project.curation.google_fitness.registry import CURATION_POLICIES
    from wearable_project.processing.google_fitness.registry import feature_names

    public = tuple(GOOGLE_FITNESS_PUBLIC_FEATURES)
    registered = tuple(feature_names())
    policy_features = tuple(sorted(CURATION_POLICIES))
    shared = ("Height", "Sleep", "StepCount", "Weight")
    payload = {
        "wearable_source": "google_fitness",
        "title": "Google Fitness DataLoaders",
        "accepted_features": list(public),
        "feature_count": len(public),
        "shared_public_features": list(shared),
        "catalog_consistency": {
            "public_equals_native_registry": public == registered,
            "public_equals_curation_registry": set(public) == set(policy_features),
        },
        "phases": ["native", "curated"],
        "default_roots": {
            "native": str(native_root),
            "curated": str(curated_root),
        },
        "source_selection": {
            "explicit": "Pass source='google_fitness' for shared loaders; Google-only loaders default to Google Fitness.",
            "one_source_per_instance": True,
            "auto_detection": False,
            "managed_root_source_guard": True,
        },
        "date_anchors": _date_anchor_summary(),
        "projections": {
            "native": dict(NATIVE_PROJECTION_SUMMARIES),
            "curated": dict(CURATED_PROJECTION_SUMMARIES),
        },
        "curated_schema_v2": {
            "native_prefix_preserved": True,
            "row_order_preserved": True,
            "row_count_preserved": True,
            "materialized_column_order": list(MATERIALIZED_COLUMN_ORDER),
            "virtual_column_order": list(VIRTUAL_COLUMN_ORDER),
            "sparse_row_defaults": dict(ROW_DEFAULTS),
            "root_contract_required": True,
        },
        "units": {
            "established": {"StepCount": "count"},
            "withheld_exporter_mappings": {
                "Calories": "kcal",
                "Distance": "m",
                "Weight": "kg",
                "Height": "m",
            },
            "activity_submeasurements": "Raw calories, distance and quantity remain source evidence; no common canonical mapping is asserted.",
        },
        "step_count": {
            "versions": ["1.0", "2.0"],
            "all_representations_preserved": True,
            "default_date_anchor": "outer_datetime",
            "canonical_day_derived": False,
            "curated_roles": [
                "legacy_v1", "fit_app_preferred", "merge_fallback", "third_party_fallback",
                "parallel_nonpreferred", "ambiguous_parallel",
            ],
        },
        "integrity": {
            "native_state": ".wearable_state.sqlite",
            "curated_state": ".wearable_curation_state.sqlite",
            "curated_contract": ".curation_contract.json",
            "checks_before_filtering": [
                "source", "phase", "state schema", "registry/contract version", "filename", "size",
                "SHA-256", "row count", "source-specific logical counters",
            ],
            "active_wal_or_journal_refused": True,
        },
        "privacy": {
            "default_withholds": [
                "persistent source identifiers", "package identifiers", "device_uid",
                "free-text source/application labels", "detailed lineage",
            ],
            "full_projection_required": True,
        },
        "contract_versions": _contract_versions(),
        "usage": {
            "shared": (
                "from wearable_project.DataLoaders.StepCountLoader import StepCountLoader\n"
                "loader = StepCountLoader(source='google_fitness', phase='curated')"
            ),
            "google_only": (
                "from wearable_project.DataLoaders.WalkingLoader import WalkingLoader\n"
                "loader = WalkingLoader(phase='native')"
            ),
            "profile": "print(loader.profile())",
            "load": f"data = loader.get_data(registration_codes=['{_EXAMPLE_CODE}'])",
        },
    }

    lines = [
        "Google Fitness DataLoaders",
        f"{len(public)} accepted feature loaders · native and curated schema-v2 products",
        "",
        "Source selection",
        "  Shared feature classes remain at their production top-level paths. Pass source='google_fitness' for",
        "    Height, Sleep, StepCount and Weight. Google-only feature loaders default to Google Fitness.",
        "  Source is selected explicitly and then verified against managed-root metadata; it is never guessed from",
        "    a filename or overlapping physical schema.",
        "",
        "Products and integrity",
        f"  Native default root: {native_root}",
        f"  Curated default root: {curated_root}",
        "  Native files are verified against .wearable_state.sqlite. Curated files are verified against",
        "    .wearable_curation_state.sqlite and the exact .curation_contract.json before sparse or virtual",
        "    fields are interpreted. Verification precedes date, inclusion and projection filtering.",
        "",
        "Curated schema v2",
        "  The complete native prefix, row order and row count are preserved. Sparse physical decisions are",
        "    reconstructed from contract defaults, and deterministic virtual projections are materialized by the",
        "    source adapter only after contract validation.",
        "",
        "Temporal anchors",
        "  Activities, Sleep, Calories and Distance use start_date; Weight and Height use end_date; StepCount uses",
        "    outer_datetime as a mechanical index anchor. The StepCount anchor is not a manufactured canonical day.",
        "",
        "Units",
        "  StepCount is established as count. Calories, Distance, Weight and Height retain proposed official-type",
        "    units as evidence, but harmonized/canonical values remain withheld until exporter mapping is accepted.",
        "  Activity calories, distance and quantity remain multiple raw measurements rather than one synthetic value.",
        "",
        "Projections",
        "  default: analytical timing, values/states, source context, reviewed provenance and curation decisions.",
        "  analysis: default plus detailed evidence, non-unique device descriptors and lineage.",
        "  full: every stored, reconstructed and virtual column, including restricted identifiers.",
        "",
        "Usage",
        "  from wearable_project.DataLoaders.StepCountLoader import StepCountLoader",
        "  loader = StepCountLoader(source='google_fitness', phase='curated')",
        f"  data = loader.get_data(registration_codes=['{_EXAMPLE_CODE}'])",
        "  print(loader.profile())",
        "",
        "Accepted features",
    ]
    lines.extend(_wrap(", ".join(public)))
    return InfoDocument(
        kind="overview",
        title="Google Fitness DataLoaders",
        payload=payload,
        text="\n".join(lines),
    )


def feature_report(loader: Any, *, include_evidence: bool) -> InfoDocument:
    """Describe one configured Google Fitness loader from installed executable contracts only."""

    from wearable_project.curation.google_fitness.contract import ROW_DEFAULTS
    from wearable_project.DataLoaders.google_fitness.units import registry_unit, unresolved_unit_candidates
    from wearable_project.curation.google_fitness.guidance import get_feature_guide
    from wearable_project.curation.google_fitness.registry import get_policy
    from wearable_project.processing.google_fitness.registry import get_feature_spec

    feature = loader.feature_name
    phase = loader.phase
    spec = get_feature_spec(feature)
    policy = get_policy(feature, allow_fallback=False)
    guide = get_feature_guide(feature)
    schema = loader._loader_schema()
    contract = loader._feature_loader_contract()
    available = tuple(dict.fromkeys(contract.physical_columns))
    if phase == "curated":
        available = tuple(dict.fromkeys((*available, *policy.output.virtual_columns)))

    evidence_ids = tuple(dict.fromkeys((*guide.evidence_refs, *policy.evidence_refs)))
    evidence = _evidence_payload(evidence_ids, include_evidence)
    measurements: list[dict[str, Any]] = []
    for measurement in policy.measurements:
        units = []
        for version in measurement.unit_by_version:
            units.append({
                "versions": list(version.versions),
                "raw_unit_status": version.raw_unit_status.value,
                "raw_unit": version.raw_unit,
                "canonical_unit": version.canonical_unit,
                "evidence_grade": version.evidence_grade.value,
                "execution_mode": version.execution_mode.value,
                "canonicalization": version.canonicalization.value,
                "evidence_refs": list(version.evidence_refs),
                "limitations": list(version.limitations),
            })
        measurements.append({
            "column": measurement.column,
            "kind": measurement.kind.value,
            "required": measurement.required,
            "raw_projection_enabled": measurement.raw_projection_enabled,
            "canonical_projection_enabled": measurement.canonical_projection_enabled,
            "established_registry_unit": registry_unit(feature, measurement.column),
            "unresolved_unit_candidates": list(unresolved_unit_candidates(feature, measurement.column)),
            "description": measurement.description,
            "version_units": units,
        })

    physical_materialized = list(policy.output.materialized_columns) if phase == "curated" else []
    virtual = list(policy.output.virtual_columns) if phase == "curated" else []
    sparse_defaults = {
        name: ROW_DEFAULTS[name]
        for name in policy.output.sparse_columns
        if phase == "curated" and name in ROW_DEFAULTS
    }
    projections = _projection_payload(schema, available)
    date = loader._source_adapter.date_contract(feature)
    source_contract = dict(loader._source_adapter.contract_metadata(feature, phase))
    root = loader.data_root
    loader_name = type(loader).__name__
    usage = {
        "construct": (
            f"from wearable_project.DataLoaders.{feature}Loader import {loader_name}\n"
            f"loader = {loader_name}(source='google_fitness', phase={phase!r}, root={str(root)!r})"
        ),
        "load": f"data = loader.get_data(registration_codes=['{_EXAMPLE_CODE}'])",
        "profile": "print(loader.profile())",
        "full": "full = loader.get_data(projection='full')",
    }
    if phase == "curated":
        usage["default_subset"] = "included = loader.get_data(default_inclusion_only=True)"

    payload = {
        "wearable_source": "google_fitness",
        "feature": feature,
        "loader": loader_name,
        "phase": phase,
        "root": str(root),
        "date_index": {
            "column": date.column,
            "semantics": date.semantics,
            "all_timestamp_columns": list(contract.timestamp_columns),
            "calendar_date_columns": list(contract.calendar_date_columns),
            "canonical_day_derived": False,
        },
        "native_schema": spec.as_dict(),
        "row_meaning": guide.row_meaning,
        "recommended_use": guide.recommended_use,
        "measurements": measurements,
        "state_columns": list(spec.state_columns),
        "supported_collecting_method_versions": list(spec.supported_versions),
        "columns": {
            "native_physical": list(spec.column_names),
            "curation_materialized": physical_materialized,
            "curation_sparse": list(policy.output.sparse_columns) if phase == "curated" else [],
            "sparse_defaults": sparse_defaults,
            "curation_virtual": virtual,
            "logical_available": list(available),
            "required": list(contract.required_columns),
            "measurement": list(contract.measurement_columns),
        },
        "projections": projections,
        "dtypes": {
            "timestamps_utc": list(contract.timestamp_columns),
            "calendar_dates_naive": list(contract.calendar_date_columns),
            "categorical": sorted(contract.categorical_columns),
            "string": sorted(contract.string_columns),
            "nullable_integer": sorted(contract.integer_columns),
            "nullable_boolean": sorted(contract.boolean_columns),
            "numeric": sorted(contract.numeric_columns),
        },
        "temporal_policy": _primitive(policy.temporal),
        "provenance_policy": _primitive(policy.provenance),
        "group_policy": _primitive(policy.grouping),
        "curation_policy": _primitive(policy.curation),
        "output_policy": _primitive(policy.output),
        "resampling_policy": _primitive(policy.resampling),
        "calibration": _primitive(policy.calibration),
        "guidance": guide.as_dict(),
        "integrity": {
            "state_database": (
                ".wearable_state.sqlite" if phase == "native" else ".wearable_curation_state.sqlite"
            ),
            "root_contract": ".curation_contract.json" if phase == "curated" else None,
            "source_and_phase_checked_before_csv": True,
            "file_checks_before_filtering": ["filename", "size", "SHA-256", "row count"],
            "logical_state_counts_checked": phase == "curated",
            "active_wal_or_journal_refused": True,
        },
        "inclusion": {
            "all_rows_returned_by_default": True,
            "default_subset_available": phase == "curated",
            "filter": "default_inclusion_only=True" if phase == "curated" else None,
            "stored_decision_not_recomputed_by_filter": phase == "curated",
        },
        "local_time": {
            "available": False,
            "reason": "The Google product does not retain a row-level UTC offset from which local wall time can be reconstructed.",
        },
        "harmonization": {
            "supported": feature == "StepCount",
            "established_unit": registry_unit(feature, "value"),
            "unit_candidates_only": list(unresolved_unit_candidates(feature, "value")),
            "multi_measurement_feature": len(spec.measurement_columns) > 1,
        },
        "evidence_refs": list(evidence_ids),
        "evidence": evidence,
        "source_contract": source_contract,
        "contract_versions": _contract_versions(),
        "usage": usage,
        "limitations": list(guide.limitations),
        "acquisition_caveat": ACQUISITION_CAVEAT,
    }

    lines = [
        f"{feature} — {loader_name}",
        f"Google Fitness · {spec.family.value} · phase {phase}",
        "",
        "What one row means",
        *_wrap(guide.row_meaning),
        *_wrap(f"Recommended use: {guide.recommended_use}"),
        "",
        "Time",
        *_wrap(
            f"The RegistrationCode/Date index uses {date.column!r}: {date.semantics}. "
            f"All original temporal contexts remain columns when admitted by the projection."
        ),
        *_wrap(guide.temporal_interpretation),
        "",
        "Measurements and units",
    ]
    if measurements:
        for measurement in measurements:
            unit_bits = []
            for version in measurement["version_units"]:
                unit_bits.append(
                    f"{','.join(version['versions'])}: {version['canonical_unit'] or 'unresolved'} "
                    f"({version['raw_unit_status']}; {version['execution_mode']})"
                )
            lines.extend(_wrap(f"{measurement['column']}: " + "; ".join(unit_bits), indent="  - "))
    else:
        lines.append("  - No numeric measurement column; the feature is interpreted through its state/context fields.")
    lines.extend(_wrap(guide.unit_interpretation))
    if feature in {"Calories", "Distance", "Weight", "Height"}:
        lines.extend(_wrap(
            "The official-type unit is retained only as a candidate. The cohort exporter mapping is unverified, "
            "so canonical and harmonized values remain missing rather than being silently labeled."
        ))
    if len(spec.measurement_columns) > 1:
        lines.extend(_wrap(
            "This is a multi-measurement row. The singular with_harmonized_values() helper is intentionally "
            "inapplicable; calories, distance and quantity remain separate evidence."
        ))

    lines.extend(["", "Columns and projections"])
    if phase == "curated":
        lines.extend(_wrap(
            "The native prefix is physically preserved. Materialized curation columns are "
            f"{', '.join(physical_materialized) or 'none'}; virtual fields are {', '.join(virtual) or 'none'}."
        ))
        if sparse_defaults:
            lines.extend(_wrap(
                "Sparse defaults: " + ", ".join(f"{key}={value!r}" for key, value in sparse_defaults.items()) + "."
            ))
    for name, projection in projections.items():
        marker = " (default)" if name == "default" else ""
        lines.extend(_wrap(f"projection={name!r}{marker}: {projection['description']}"))

    lines.extend(["", "Curation and provenance"])
    if phase == "curated":
        lines.extend(_wrap(
            "All rows are returned unless default_inclusion_only=True is requested. That filter applies the stored "
            "reviewed inclusion decision and never deletes rows from the curated product."
        ))
        if feature == "StepCount":
            lines.extend(_wrap(
                "Every version-1 and version-2 representation remains present. step_count_role records the reviewed "
                "preference/fallback role; outer_datetime is only the mechanical index anchor."
            ))
    else:
        lines.extend(_wrap("Native loading exposes source evidence only; curation decisions and virtual fields are absent."))
    lines.extend(_wrap(guide.provenance_interpretation))
    lines.extend(_wrap(ACQUISITION_CAVEAT))

    lines.extend(["", "Integrity"])
    lines.extend(_wrap(
        f"The loader validates the managed {phase} root's source and phase before parsing a feature CSV, then checks "
        "the complete physical file against state before any date, inclusion or projection filter."
    ))
    if phase == "curated":
        lines.extend(_wrap(
            ".curation_contract.json must match the installed schema-v2 contract even when state_validation='off', "
            "because sparse defaults and virtual projections otherwise have no safe interpretation."
        ))

    lines.extend(["", "Important limitations"])
    if guide.limitations:
        for item in guide.limitations:
            lines.extend(_wrap(item, indent="  - "))
    else:
        lines.append("  - None recorded in source guidance.")

    lines.extend(["", "Usage"])
    for value in usage.values():
        lines.extend(f"  {line}" for line in value.splitlines())

    lines.extend(["", "Evidence references"])
    lines.extend(f"  - {evidence_id}" for evidence_id in evidence_ids)
    if include_evidence and evidence:
        lines.extend(["", "Evidence details"])
        for evidence_id, source in evidence.items():
            lines.append(
                f"  {evidence_id}: {source['title']} — {source['authority']} "
                f"({source['kind']}; {source['grade']})"
            )
            lines.extend(_wrap(source["summary"], indent="    "))
            for limitation in source.get("limitations", []):
                lines.extend(_wrap(f"Limitation: {limitation}", indent="    "))
            locator = source.get("url") or source.get("artifact")
            if locator:
                lines.append(f"    {locator}")

    return InfoDocument(
        kind="feature",
        title=f"{feature} Google Fitness DataLoader information",
        payload=payload,
        text="\n".join(lines),
    )
