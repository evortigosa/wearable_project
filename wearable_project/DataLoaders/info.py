"""
Wearable Data Processing and Modeling project
Introspective help for the HPP-style wearable feature loaders.
Every statement here is derived from an executable contract rather than written by hand: the native processing
registry, the curation registry and its guidance, the column roles and projections in ``curation.schema``, the
stored-unit reconciliation in ``DataLoaders._units``, and the loader classes themselves. When a contract changes,
this documentation changes with it, and ``tests/test_dataloader_info.py`` holds its claims against what the loaders
actually do on the representative samples. Typical notebook use:

    from wearable_project import DataLoaders

    DataLoaders.info()                 # package overview and capability matrix
    DataLoaders.info("StepCount")      # one feature
    DataLoaders.info("Sleep").as_dict()

``info`` describes the static contract and never touches data. To see what a particular root holds, use a loader's
``profile()``. Every feature loader also exposes ``loader.info()`` with the instance's phase and root.
"""


from __future__ import annotations
import textwrap
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any
from wearable_project import __release_label__, __version__
from wearable_project.DataLoaders._base import (
    AppleHealthFeatureLoader, DEFAULT_CURATED_ROOT, DEFAULT_NATIVE_ROOT, ProcessingPhase,
    _DENSE_CURATED_DEFAULTS, _PHASE_STATE_FILES,
)
from wearable_project.DataLoaders._profile import ACQUISITION_CAVEAT
from wearable_project.DataLoaders._units import _curation_declaration, stored_unit
from wearable_project.exceptions import DataLoaderConfigurationError
from wearable_project.curation import schema
from wearable_project.curation.decisions import decisions_fingerprint
from wearable_project.curation.evidence import EVIDENCE
from wearable_project.curation.guidance import GUIDANCE_VERSION, get_feature_guide, guidance_fingerprint
from wearable_project.curation.registry import (
    COHORT_OBSERVED_FEATURES, CURATION_REGISTRY_VERSION, get_policy, registry_fingerprint,
)
from wearable_project.processing.registry import (
    REGISTRY_VERSION as PROCESSING_REGISTRY_VERSION, FeatureSpec, get_feature_spec,
)


INFO_CONTRACT_VERSION = "dataloader-info-2"

_WIDTH = 100
_EXAMPLE_CODE = "10K_1235738253"
_DATE_SEMANTICS = {
    "event_start": "the start of each event, in UTC",
    "outer_summary_datetime_not_canonical_summary_day": (
        "the export's outer UTC day key, which is not proven to be the summary's local calendar day"
    ),
}
_STATE_TABLES = {"curated": "curation_outputs", "native": "feature_outputs"}
_STATE_CHECKS = {
    "curated": [
        "SHA-256", "size", "row count", "pass/review/exclude_default counts", "default-inclusion counts",
        "acquisition-method counts", "curation flag counts",
    ],
    "native": ["SHA-256", "size", "row count"],
}
_STATE_VALIDATION = {
    "auto": "default; verifies every file whenever the state database exists, and reports the load as unverified otherwise",
    "required": "fails when the state database is absent",
    "off": "skips verification; for deliberately assembled sample trees",
}
_MEASUREMENT_KINDS = {
    # measurement_kind declared by the curation registry -> (matrix label, how to read and combine the values)
    "extensive_total": (
        "total: sum",
        "Each row is a total accumulated over its stored interval, not a reading at an instant. A row can hold "
        "a fractional share of a source sample that spans more than one interval, so fractional values are "
        "expected. Combine rows by summing them.",
    ),
    "intensive_value": (
        "level: average",
        "Each row is a level or a rate, not an amount, so a sum of rows has no meaning. Summarize over time with "
        "averages or other statistics.",
    ),
    "ratio": (
        "proportion: average",
        "Each row is a proportion, not an amount, so a sum of rows has no meaning. Summarize over time with "
        "averages or other statistics.",
    ),
    "event_amount": (
        "event amount: sum",
        "Each row is an amount logged for one event. Summing the events in a period gives the total logged "
        "in that period.",
    ),
    "summary_statistic": (
        "device summary",
        "Each row is a summary the device computed over a longer window, not a raw reading. Use it as reported "
        "rather than aggregating it like raw readings.",
    ),
    "multivariate_point": (
        "reading set",
        "Each row is one reading of several values that belong together. Keep them together, and never sum them.",
    ),
    "multivariate_summary": (
        "daily summary",
        "Each row is one summary object holding several measures; read the caveats before treating a row as a "
        "calendar day.",
    ),
    "categorical_state": (
        "state",
        "value names a state, not a quantity. Time spent in each state comes from start_date and end_date.",
    ),
    "duration": (
        "duration: union",
        "The amount is the session's duration, end_date minus start_date. Merge overlapping sessions rather "
        "than adding them, or shared time is counted twice.",
    ),
    "signal": (
        "waveform",
        "Each row is one recording: the waveform samples and summary values computed from them.",
    ),
}
_FLOAT_NOTE = (
    "Values are float64, exactly as stored: whole numbers appear as x.0, and some carry floating-point noise "
    "from earlier arithmetic, such as 61.99999999999999 for 62. Round only for presentation, after any aggregation."
)
_WITHHELD_ROLES = [role for role in schema.ColumnRole if role not in schema.DEFAULT_ROLES]


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
    """Return the names of every feature with a public DataLoader class."""

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
        ``None`` returns a package overview with a capability matrix. Otherwise, provide one of the names from
        ``available_features()``, for example ``"StepCount"``; matching ignores case.
    phase:
        Loader phase the feature report describes. ``"curated"`` is the default. Curation annotations, the dense
        curated schema and ``default_inclusion_only`` apply only to the curated phase, and the report says so.
    native_root, curated_root, root:
        Same path options accepted by every feature loader. They are reported, and used to make the usage
        examples runnable as configured, but never accessed: ``info`` performs no data I/O.
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


# ------------------------------------------------------------------------------------------ helpers
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


def _wrap(text: str, indent: str = "  ", width: int = _WIDTH) -> list[str]:
    """Wrap one paragraph, indenting continuation lines under the first."""

    return textwrap.wrap(str(text), width=width, initial_indent=indent, subsequent_indent=indent + "  ") or [indent]


def _plain(value: Any) -> Any:
    """Registry enums render as their value, not as ``EnumName.MEMBER``."""

    return getattr(value, "value", value)


def _join(units: tuple[str, ...] | list[str], conjunction: str) -> str:
    units = list(units)
    if len(units) <= 1:
        return units[0] if units else "an unknown unit"
    return ", ".join(units[:-1]) + f" {conjunction} " + units[-1]


def _either(units: tuple[str, ...] | list[str]) -> str:
    units = list(units)
    if not units:
        return "an unknown unit"
    if len(units) == 1:
        return units[0]
    return ", ".join(units[:-1]) + " or " + units[-1]


def _constructor(loader: AppleHealthFeatureLoader) -> str:
    """The loader call that reproduces this instance's phase and root, so examples run as configured."""

    arguments = []
    if loader.phase != "curated":
        arguments.append(f"phase={loader.phase!r}")
    if loader.uses_root_override:
        arguments.append(f"root={str(loader.data_root)!r}")
    return f"{loader.__class__.__name__}({', '.join(arguments)})"


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


# ------------------------------------------------------------------------- derived statements
def _units(feature: str, spec: FeatureSpec) -> list[dict[str, Any]]:
    """What each registry declares for each measurement column, and the unit the loaders establish."""

    rows = []
    for column in spec.measurement_columns:
        established = stored_unit(feature, column)
        declaration = _curation_declaration(feature, column)
        processing = spec.unit_policy
        rows.append({
            "column": column,
            "processing_registry": None if processing is None else {
                "raw_unit": processing.raw_unit, "canonical_unit": processing.canonical_unit,
                "status": _plain(processing.status),
            },
            "curation_registry": None if declaration is None else {
                "raw_unit": declaration.raw_unit, "raw_unit_status": _plain(declaration.raw_unit_status),
                "candidates": list(declaration.raw_unit_candidates),
                "canonical_unit": declaration.canonical_unit,
            },
            "established_unit": established.unit,
            "canonical_unit": established.canonical_unit,
            "already_canonical": established.already_canonical,
            "withheld_by_curation": established.withheld_by_curation,
        })
    return rows


def _unit_line(row: dict[str, Any]) -> list[str]:
    """One wrapped line per measurement column: each registry's declaration, then the unit the loaders use."""

    processing, curation = row["processing_registry"], row["curation_registry"]
    if processing is None:
        processing_text = "no unit"
    else:
        processing_text = processing["raw_unit"]
        if processing["canonical_unit"] != processing["raw_unit"]:
            processing_text += f" converted to {processing['canonical_unit']}"
        processing_text += f" ({processing['status']})"
    if curation is None:
        curation_text = "no declaration"
    elif curation["raw_unit"] is None:
        curation_text = f"unknown, {_either(curation['candidates'])} ({curation['raw_unit_status']})"
    else:
        curation_text = f"{curation['raw_unit']} ({curation['raw_unit_status']})"
    if row["established_unit"] is not None:
        used = (f"{row['established_unit']}, already canonical" if row["already_canonical"]
                else f"{row['established_unit']}, canonical {row['canonical_unit']} through canonical_value")
    elif row["withheld_by_curation"] and processing is not None:
        used = "none, because curation withholds the processing registry's label"
    elif row["withheld_by_curation"]:
        used = "none before curation; resolved per row where curation can"
    elif curation is not None and curation["raw_unit"]:
        used = "none, because the loaders adopt units only from the processing registry"
    else:
        used = "none"
    return _wrap(
        f"{row['column']}: processing registry {processing_text} · curation registry {curation_text} · "
        f"used by the loaders: {used}"
    )


def _harmonization(feature: str, spec: FeatureSpec) -> dict[str, Any]:
    """
    What ``with_harmonized_values()`` produces for this feature in each phase, derived from the same declarations
    the helper reads. ``sources`` lists every unit source a row can receive, so it is a bound, not a count.
    """

    if "value" not in spec.measurement_columns:
        measurements = ", ".join(spec.measurement_columns) or "none"
        return {
            "available": False, "sources": {"curated": [], "native": []}, "unit": None,
            "summary": f"not applicable: the feature has no single numeric value column (measurements: {measurements})",
            "matrix": "not applicable",
        }
    if "value" in schema.categorical_columns(feature):
        return {
            "available": False, "sources": {"curated": [], "native": []}, "unit": None,
            "summary": "not applicable: value holds categories (states), not quantities",
            "matrix": "not applicable",
        }
    unit = stored_unit(feature, "value")
    declaration = _curation_declaration(feature, "value")
    processing = spec.unit_policy
    if unit.withheld_by_curation and processing is not None:
        return {
            "available": True, "sources": {"curated": ["unresolved"], "native": ["unresolved"]}, "unit": None,
            "summary": (
                f"unresolved in both phases: curation cannot tell {_join(unit.candidates, 'and')} apart, so the "
                f"processing registry's {processing.raw_unit} label is not endorsed; value and raw_unit stay as stored"
            ),
            "matrix": f"unresolved ({_either(unit.candidates)})",
        }
    if unit.already_canonical:
        return {
            "available": True, "sources": {"curated": ["registry"], "native": ["registry"]}, "unit": unit.unit,
            "summary": f"registry, in both phases: stored values are already in {unit.unit}",
            "matrix": f"registry, {unit.unit}",
        }
    if processing is not None:
        return {
            "available": True, "sources": {"curated": ["curation", "unresolved"], "native": ["processing"]},
            "unit": processing.canonical_unit,
            "summary": (
                f"native: processing's conversion from {processing.raw_unit} to {processing.canonical_unit}; "
                "curated: the same canonical values, row by row as curation reviewed them, otherwise unresolved"
            ),
            "matrix": f"reviewed conversion, {processing.canonical_unit}",
        }
    if declaration is not None and declaration.canonical_unit and declaration.raw_unit != declaration.canonical_unit:
        target = declaration.canonical_unit
        return {
            "available": True, "sources": {"curated": ["curation", "unresolved"], "native": ["unresolved"]},
            "unit": target,
            "summary": (
                f"curated: {target} for each row whose unit curation resolved, otherwise unresolved; native: "
                "unresolved, since no unit is established before curation"
            ),
            "matrix": f"curation per row, {target}",
        }
    note = (f"; curation declares {declaration.raw_unit}, but harmonization adopts only units the processing "
            "registry establishes" if declaration is not None and declaration.raw_unit else "")
    return {
        "available": True, "sources": {"curated": ["unresolved"], "native": ["unresolved"]}, "unit": None,
        "summary": "unresolved in both phases: the processing registry establishes no unit" + note,
        "matrix": "unresolved",
    }


def _values(feature: str, spec: FeatureSpec, policy: dict[str, Any]) -> dict[str, Any]:
    """How to read and combine this feature's values, from its declared measurement kind and units."""

    kind = policy["semantics"]["measurement_kind"]
    label, reading = _MEASUREMENT_KINDS.get(kind, (kind, ""))
    numeric = [
        column for column in spec.measurement_columns
        if column not in schema.categorical_columns(feature) and schema.role_of(column) is not schema.ColumnRole.PAYLOAD
    ]
    counts = spec.unit_policy is not None and spec.unit_policy.raw_unit == "count"
    guidance = [reading] if reading else []
    if kind == "extensive_total" and counts:
        guidance.append(
            f"{feature} counts whole units, but stored rows need not be whole numbers. Sum rows first, and round "
            "the total if whole numbers are needed: rounding each row changes the totals."
        )
    if numeric:
        guidance.append(_FLOAT_NOTE)
    return {
        "measurement_kind": kind,
        "aggregation": policy["resampling"]["aggregation"],
        "combine": label,
        "extensive_total": kind == "extensive_total",
        "count_unit": counts,
        "numeric_columns": numeric,
        "dtype": "float64" if numeric else None,
        "guidance": guidance,
    }


def _local_time(loader: AppleHealthFeatureLoader) -> dict[str, Any]:
    if loader.date_column == "start_date":
        return {
            "available": True, "columns": ["start_date_local", "end_date_local"],
            "summary": (
                "with_local_time() adds start_date_local and end_date_local, timezone-naive local wall-clock "
                "times from each row's utc_offset_minutes; the UTC columns are kept"
            ),
        }
    return {
        "available": False, "columns": [],
        "summary": (
            "not available: the exporter encodes no event times or UTC offset for this feature"
        ),
    }


def _columns(feature: str, spec: FeatureSpec, phase: str) -> dict[str, Any]:
    by_role = schema.columns_by_role()
    withheld = {role.value: list(by_role[role]) for role in _WITHHELD_ROLES}
    payload_columns = [c for c in spec.measurement_columns if schema.role_of(c) is schema.ColumnRole.PAYLOAD]
    return {
        "default_projection": schema.DEFAULT_PROJECTION,
        "projections": {name: schema.describe_projection(name) for name in schema.available_projections()},
        "measurement_columns": list(spec.measurement_columns),
        "withheld_by_default": withheld,
        "withheld_measurement_columns": payload_columns,
        "dense_curated_columns": dict(_DENSE_CURATED_DEFAULTS) if phase == "curated" else {},
    }


def _dtypes(feature: str, phase: str) -> dict[str, Any]:
    categorical = sorted(schema.categorical_columns(feature))
    return {
        "categorical": categorical,
        "categorical_value": "value" in categorical,
        "nullable_boolean": sorted(schema.NULLABLE_BOOLEAN_COLUMNS),
        "boolean": ["include_by_default"] if phase == "curated" else [],
        "timestamps": "timezone-aware UTC",
    }


def _analytical_status(policy: dict[str, Any], phase: str) -> dict[str, Any]:
    curation, calibration, identity = policy["curation"], policy["calibration"], policy["identity"]
    if curation["default_inclusion"] == "exclude":
        subset = "keeps only rows that a rule explicitly includes, because the policy excludes rows by default"
    else:
        subset = "drops only rows that a rule excluded, because the policy includes rows by default"
    return {
        "curation_applied": phase == "curated",
        "returned_by_default": (
            "every row, including review and excluded rows" if phase == "curated" else "every stored row"
        ),
        "policy_default_status": curation["default_status"],
        "policy_default_inclusion": curation["default_inclusion"],
        "default_subset": subset,
        "maturity": identity["maturity"],
        "evidence_grade": calibration.get("evidence_grade"),
        "execution_mode": calibration.get("execution_mode"),
        "rationale": calibration.get("rationale"),
        "safe_fallback": calibration.get("safe_fallback"),
        "required_audits": list(calibration.get("required_audits") or []),
    }


def _integrity(phase: str) -> dict[str, Any]:
    return {
        "state_database": _PHASE_STATE_FILES[phase],
        "table": _STATE_TABLES[phase],
        "checks": list(_STATE_CHECKS[phase]),
        "policy_fingerprint_checked": phase == "curated",
        "state_validation": dict(_STATE_VALIDATION),
        "read_only": True,
    }


def _usage(loader: AppleHealthFeatureLoader, spec: FeatureSpec, local: dict, harmonized: dict) -> dict[str, str]:
    """Examples that run as written for this loader's configuration."""

    call, cls = _constructor(loader), loader.__class__.__name__
    selectable = [c for c in spec.measurement_columns if schema.role_of(c) is not schema.ColumnRole.PAYLOAD]
    columns = selectable[:3] or ["start_date", "end_date"]
    usage = {
        "import": f"from wearable_project.DataLoaders.{spec.name}Loader import {cls}",
        "load": f"data = {call}.get_data()",
        "participant_and_dates": (
            f"df = {call}.get_data(reg_ids={_EXAMPLE_CODE!r}, start_date='2024-01-01', "
            "end_date='2024-12-31 23:59:59').df"
        ),
        "columns": f"df = {call}.get_data(cols={columns!r}).df",
    }
    if harmonized["available"]:
        usage["harmonized_values"] = f"df = {call}.get_data().with_harmonized_values().df"
    if local["available"]:
        usage["local_time"] = f"df = {call}.get_data().with_local_time().df"
    if loader.phase == "curated":
        usage["default_subset"] = f"df = {call}.get_data(default_inclusion_only=True).df"
    usage["profile"] = f"print({call}.profile())"
    return usage


def _contract_versions() -> dict[str, Any]:
    return {
        "processing_registry": PROCESSING_REGISTRY_VERSION,
        "curation_registry": CURATION_REGISTRY_VERSION,
        "guidance": GUIDANCE_VERSION,
        "curation_registry_fingerprint": registry_fingerprint(),
        "guidance_fingerprint": guidance_fingerprint(),
        "decisions_fingerprint": decisions_fingerprint(),
    }


# --------------------------------------------------------------------------------------- overview
def _overview_report(*, native_root: Path, curated_root: Path) -> InfoReport:
    feature_rows: list[dict[str, Any]] = []
    for feature in COHORT_OBSERVED_FEATURES:
        guide = get_feature_guide(feature)
        spec = get_feature_spec(feature)
        loader_cls = _loader_class(feature)
        local = _local_time(loader_cls())
        harmonized = _harmonization(feature, spec)
        values = _values(feature, spec, get_policy(feature, allow_fallback=False).as_dict())
        feature_rows.append({
            "feature": feature,
            "loader": loader_cls.__name__,
            "module": f"wearable_project.DataLoaders.{feature}Loader",
            "category": guide.category,
            "processing_family": spec.family.value,
            "short_description": guide.short_description,
            "local_time": local["available"],
            "harmonized_values": harmonized["matrix"],
            "values": values["combine"],
            "extensive_total": values["extensive_total"],
        })
    totals = [row["feature"] for row in feature_rows if row["extensive_total"]]
    max_rows = AppleHealthFeatureLoader.default_max_rows
    limit_text = "no limit" if max_rows is None else f"{max_rows:,} rows"

    payload: dict[str, Any] = {
        "kind": "overview",
        "info_contract_version": INFO_CONTRACT_VERSION,
        "package": {"version": __version__, "release_label": __release_label__},
        "contract_versions": _contract_versions(),
        "defaults": {
            "phase": "curated",
            "native_root": str(native_root),
            "curated_root": str(curated_root),
            "index_names": ["RegistrationCode", "Date"],
            "registration_code_rule": "participant folder ID prefixed with '10K_'",
            "projection": schema.DEFAULT_PROJECTION,
            "state_validation": "auto",
            "max_rows": max_rows,
        },
        "phase_semantics": {
            "native": {
                "meaning": "Native processed Apple HealthKit rows at native event or interval resolution.",
                "important_behavior": [
                    "No fixed-window resampling, interpolation, imputation, or daily broadcasting.",
                    "Native feature semantics, timestamps, provenance, and reconciled record identity are retained.",
                    "Each file is verified against feature_outputs in the processing state database.",
                    "No curation annotations exist, so default_inclusion_only raises in this phase.",
                ],
            },
            "curated": {
                "meaning": (
                    "Non-destructive curation of the native rows, with policy annotations and reviewed canonical "
                    "values where justified."
                ),
                "important_behavior": [
                    "All curated rows are loaded by default, including review and include_by_default=False rows.",
                    "Use default_inclusion_only=True only when an analyst explicitly wants the policy's default subset.",
                    "acquisition_method, curation_status, curation_flags and include_by_default are always present, "
                    "reconstructed where a file stores none; canonical unit fields stay missing where unresolved.",
                    "Each file is verified against curation_outputs in the curation state database.",
                ],
            },
        },
        "loading_behavior": {
            "projections": {name: schema.describe_projection(name) for name in schema.available_projections()},
            "columns": "columns=[...] (alias cols=) selects exact columns and overrides the projection.",
            "participants": "registration_codes=[...] (alias reg_ids=) accepts codes with or without the 10K_ prefix.",
            "dates": (
                "start_date and end_date are inclusive UTC bounds; a bare date means midnight UTC, so "
                "end_date='2024-12-31' stops at the first instant of that day."
            ),
            "exact_values": "Values come back exactly as stored; only empty cells are missing.",
            "missing_requested_participant": (
                "Returns an HPP-indexed frame without that participant and lists it in "
                "load_report['requested_participants_missing_from_root']."
            ),
            "storage": "DataLoaders are read-only and never modify native or curated roots.",
        },
        "values": {
            "dtype": "float64",
            "exactness": _FLOAT_NOTE,
            "extensive_totals": totals,
            "totals_guidance": (
                "Totals are stored per interval, and a row can hold a fractional share of a source sample that spans "
                "more than one interval, even for counts. Sum rows first, and round the total if whole numbers are "
                "needed: rounding each row changes the totals."
            ),
            "never_sum": "Levels, rates and proportions are summarized with averages or other statistics, never sums.",
        },
        "result": {
            "df": "rows indexed by RegistrationCode and Date",
            "df_metadata": "one row per participant: row counts, date span, verification, coverage",
            "df_columns_metadata": "one row per column: role, description, dtype, non-null count, registry unit",
            "load_report": "provenance of the call: root, phase, filters, projection, verification, size, coverage",
            "helpers": {
                "with_local_time": "local wall-clock columns from each row's utc_offset_minutes",
                "with_harmonized_values": "each row's value in a trusted unit, with the layer that established it",
            },
        },
        "scalability": {
            "max_rows": max_rows,
            "behavior": (
                "get_data() refuses requests above max_rows, before parsing any file when the state database "
                "gives an exact count, otherwise as rows are retained."
            ),
            "profile": "FeatureLoader().profile() sizes a request and describes a root without loading data.",
            "no_hidden_sampling": "Loaders never silently sample, truncate, aggregate, or downsample rows.",
        },
        "usage": {
            "basic": (
                "from wearable_project.DataLoaders.StepCountLoader import StepCountLoader\n"
                "df = StepCountLoader().get_data().df"
            ),
            "native": "df = StepCountLoader(phase='native').get_data().df",
            "root_override": "df = StepCountLoader(root='/path/to/sample').get_data().df",
            "participant_filter": f"df = StepCountLoader().get_data(reg_ids={_EXAMPLE_CODE!r}).df",
            "date_filter": (
                "df = StepCountLoader().get_data(start_date='2024-01-01', end_date='2024-12-31 23:59:59').df"
            ),
            "curated_default_subset": "df = StepCountLoader().get_data(default_inclusion_only=True).df",
            "profile": "print(StepCountLoader().profile())",
            "feature_help": "from wearable_project import DataLoaders\nprint(DataLoaders.info('StepCount'))",
        },
        "feature_count": len(feature_rows),
        "features": feature_rows,
    }

    lines = [
        f"Wearable DataLoaders — {__version__} ({__release_label__})",
        f"{len(feature_rows)} features · default phase curated · index RegistrationCode, Date",
        "",
        "Core contract",
        "  - FeatureLoader().get_data() returns a LoaderData whose .df is an HPP-style pandas DataFrame.",
        "  - Curated rows are never silently filtered; review and excluded rows stay visible by default.",
        "    default_inclusion_only=True is the explicit way to take the policy's default subset.",
        "  - Values come back exactly as stored. Loaders are read-only and never resample, interpolate,",
        "    impute, aggregate, or sample.",
        "  - Every file is verified against its phase's state database before use.",
        "",
        "Getting the data",
    ]
    for name in schema.available_projections():
        lines.extend(_wrap(f"projection={name!r:11s} {schema.describe_projection(name)}"))
    lines.extend([
        "  columns=[...] (alias cols=) selects exact columns and overrides the projection.",
        "  registration_codes=[...] (alias reg_ids=) accepts codes with or without the 10K_ prefix.",
        "  start_date/end_date are inclusive UTC bounds. A bare date means midnight, so end a year with",
        "  end_date='2024-12-31 23:59:59'.",
        "",
        "Values",
        *_wrap(_FLOAT_NOTE),
        *_wrap(
            f"Totals ({', '.join(totals)}) are stored per interval, and a row can hold a fractional share of a "
            "source sample that spans more than one interval, even for counts. Sum rows first, and round the total "
            "if whole numbers are needed: rounding each row changes the totals."
        ),
        *_wrap("Levels, rates and proportions, such as HeartRate, are summarized with averages or other statistics, "
               "never sums. The matrix below says how each feature's values combine."),
        "",
        "What a result holds",
        "  .df                   rows, indexed by RegistrationCode and Date",
        "  .df_metadata          one row per participant: row counts, date span, verification, coverage",
        "  .df_columns_metadata  one row per column: role, description, dtype, non-null count, registry unit",
        "  .load_report          provenance: root, phase, filters, projection, verification, size, coverage",
        "  .with_local_time()          adds local wall-clock columns from each row's utc_offset_minutes",
        "  .with_harmonized_values()   adds each row's value in a trusted unit, and which layer established it",
        "",
        "Phases",
        "  curated  non-destructive curation of the native rows. acquisition_method, curation_status,",
        "           curation_flags and include_by_default are always present; canonical unit fields stay",
        "           missing where a unit was not resolved.",
        "  native   processed rows before curation: no curation annotations, and default_inclusion_only raises.",
        "",
        "Integrity",
        "  curated  each file is checked against curation_outputs: SHA-256, size, row count, and curation counts.",
        "  native   each file is checked against feature_outputs: SHA-256, size and row count.",
        "  state_validation='auto' (default) verifies whenever the state database exists; 'required' fails",
        "  without it; 'off' skips it. The database is read without ever writing to the data tree.",
        "",
        "Scale",
        f"  get_data() refuses requests above max_rows ({limit_text} in this session); pass max_rows=None",
        "  to lift it for one call. FeatureLoader().profile() sizes a request without loading any data.",
        "",
        "info versus profile",
        "  info() describes the static contract and never touches data; profile() reports what a root holds.",
        "",
        #"Default roots",
        #f"  native:  {native_root}",
        #f"  curated: {curated_root}",
        #"",
        "Usage",
        "  from wearable_project.DataLoaders.StepCountLoader import StepCountLoader",
        "  df = StepCountLoader().get_data().df",
        "  df = StepCountLoader(phase='native').get_data().df",
        "  df = StepCountLoader(root='/path/to/sample').get_data(reg_ids='10K_1235738253').df",
        "  print(StepCountLoader().profile())",
        "",
        "Feature-specific help",
        "  from wearable_project import DataLoaders",
        "  print(DataLoaders.info('StepCount'))",
        "  payload = DataLoaders.info('StepCount').as_dict()",
        "",
        "Features: how their values combine, and what the helpers give in the curated phase",
        f"  {'feature':24s} {'values':21s} {'local time':11s} harmonized values",
    ])
    for row in feature_rows:
        lines.append(
            f"  {row['feature']:24s} {row['values']:21s} "
            f"{'yes' if row['local_time'] else 'no':11s} {row['harmonized_values']}"
        )
    return InfoReport(kind="overview", title="Wearable DataLoaders", payload=payload, text="\n".join(lines))


# ---------------------------------------------------------------------------------- feature report
def _feature_report(loader: AppleHealthFeatureLoader, *, include_evidence: bool,) -> InfoReport:
    feature = loader.feature_name
    phase = loader.phase
    curated = phase == "curated"
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

    ctor = loader.__class__.__name__
    units = _units(feature, processing_spec)
    harmonized = _harmonization(feature, processing_spec)
    local = _local_time(loader)
    values = _values(feature, processing_spec, policy_payload)
    columns = _columns(feature, processing_spec, phase)
    dtypes = _dtypes(feature, phase)
    status = _analytical_status(policy_payload, phase)
    integrity = _integrity(phase)
    usage = _usage(loader, processing_spec, local, harmonized)
    max_rows = AppleHealthFeatureLoader.default_max_rows
    date_text = _DATE_SEMANTICS.get(loader.date_semantics, loader.date_semantics)

    payload: dict[str, Any] = {
        "kind": "feature",
        "info_contract_version": INFO_CONTRACT_VERSION,
        "package": {"version": __version__, "release_label": __release_label__},
        "feature": feature,
        "loader": {
            "class": ctor,
            "module": f"wearable_project.DataLoaders.{feature}Loader",
            "phase": phase,
            "resolved_root": str(loader.data_root),
            "root_override": loader.uses_root_override,
            "filename": loader.filename,
            "index_names": loader._data_index_names,
            "registration_prefix": loader.registration_prefix,
            "date_column": loader.date_column,
            "date_semantics": loader.date_semantics,
            "date_semantics_text": date_text,
            "read_only": True,
        },
        "guidance": guide.as_dict(),
        "processing": _processing_spec_payload(processing_spec),
        "curation": {
            "registry_version": CURATION_REGISTRY_VERSION,
            "policy_fingerprint": policy.fingerprint(),
            "policy": policy_payload,
            "applies_in_phase": curated,
            "scope_note": (
                "Curated files preserve native rows and add policy annotations and derived values; the DataLoader "
                "does not execute curation."
            ),
        },
        "values": values,
        "columns": columns,
        "units": units,
        "derived_helpers": {"with_local_time": local, "with_harmonized_values": harmonized},
        "analytical_status": status,
        "acquisition": {
            "summary": guide.acquisition_summary,
            "strategy": policy_payload["provenance"].get("acquisition_method_strategy"),
            "coverage": (
                "acquisition_method is 'unclassified' where curation could not establish how a row was obtained; "
                "profile() reports the classified fraction for a root. " + ACQUISITION_CAVEAT
            ),
        },
        "integrity": integrity,
        "dtypes": dtypes,
        "scale": {"max_rows": max_rows, "profile": usage["profile"]},
        "loader_behavior": {
            "all_curated_rows_by_default": True,
            "default_inclusion_filter": (
                "Pass default_inclusion_only=True explicitly; curated phase only."
            ),
            "date_filtering": (
                "Inclusive UTC start_date/end_date bounds on the Date anchor; a bare date means midnight UTC."
            ),
            "resampling": (
                "The curation registry may declare future resampling semantics, but this loader does not resample."
            ),
        },
        "usage": usage,
        "evidence_refs": evidence_ids,
        "evidence": evidence_payload,
        "contract_versions": _contract_versions(),
    }

    resampling = policy_payload["resampling"]
    lines = [
        f"{feature} — {ctor}",
        f"{guide.category} · {processing_spec.family.value} · phase {phase}",
        #f"Root: {loader.data_root}",
        "",
        "What it is",
        *_wrap(guide.short_description),
        *_wrap(f"One row: {guide.one_row_means}"),
        "",
        "Values",
        *[line for item in values["guidance"] for line in _wrap(item)],
        "",
        "Time",
        f"  Index: {', '.join(loader._data_index_names)}. Date is {date_text}.",
        *_wrap(f"Native time semantics: {guide.native_time_semantics}"),
        *_wrap(f"Local wall-clock time: {local['summary']}."),
        "",
        "Getting the data",
    ]
    for name in schema.available_projections():
        marker = " (default)" if name == schema.DEFAULT_PROJECTION else ""
        lines.extend(_wrap(f"projection={name!r}{marker}: {schema.describe_projection(name)}"))
    lines.extend(_wrap(f"Measurements: {', '.join(processing_spec.measurement_columns) or 'none'}."))
    if columns["withheld_measurement_columns"]:
        heavy = ", ".join(columns["withheld_measurement_columns"])
        lines.extend(_wrap(
            f"{heavy} is withheld by default because of its size; request it with projection='full' or "
            f"columns={columns['withheld_measurement_columns']!r}."
        ))
    lines.extend(_wrap(
        "load_report['columns_withheld'] lists what a call withheld, and columns=[...] (alias cols=) returns "
        "any stored column by name."
    ))
    if curated:
        dense = ", ".join(f"{name}={value!r}" for name, value in columns["dense_curated_columns"].items())
        lines.extend(_wrap(f"Always present in the curated phase, reconstructed where a file stores none: {dense}."))

    lines.extend(["", "Units and harmonized values"])
    for row in units:
        lines.extend(_unit_line(row))
    lines.extend(_wrap(f"with_harmonized_values(): {harmonized['summary']}."))

    lines.extend(["", "Curation and inclusion"])
    if curated:
        lines.extend(_wrap(f"Returned by default: {status['returned_by_default']}."))
        lines.extend(_wrap(
            f"Policy defaults, before row-level rules: status {status['policy_default_status']}, inclusion "
            f"{status['policy_default_inclusion']}. profile() shows this root's actual outcome."
        ))
        lines.extend(_wrap(f"default_inclusion_only=True: {status['default_subset']}."))
        lines.extend(_wrap(
            f"Policy maturity {status['maturity']} · evidence grade {status['evidence_grade']} · "
            f"execution {status['execution_mode']}"
        ))
        if status["rationale"]:
            lines.extend(_wrap(f"Why: {status['rationale']}"))
        if status["safe_fallback"]:
            lines.extend(_wrap(f"Fallback: {status['safe_fallback']}"))
        if status["required_audits"]:
            lines.extend(_wrap(f"Open audits: {', '.join(status['required_audits'])}"))
        lines.extend(_wrap(f"Summary: {guide.curation_summary}"))
    else:
        lines.extend(_wrap(
            "Curation has not run in the native phase: rows carry no curation status, flags or default-subset "
            "decision, and default_inclusion_only raises. Load phase='curated' for them."
        ))

    lines.extend(["", "Acquisition and provenance", *_wrap(guide.acquisition_summary)])
    if curated:
        lines.extend(_wrap(payload["acquisition"]["coverage"]))
    else:
        lines.extend(_wrap(
            "In the native phase acquisition_method is present only where processing assigned it; curation "
            "classifies the rest."
        ))

    lines.extend(["", "Integrity"])
    lines.extend(_wrap(
        f"Each file is checked against {integrity['table']} in {integrity['state_database']} before use: "
        + ", ".join(integrity["checks"])
        + (". The policy fingerprint is compared with the installed registry, and a difference is reported."
           if curated else ".")
    ))
    lines.extend(_wrap(
        "state_validation='auto' (default) verifies whenever the state database exists; 'required' fails without "
        "it; 'off' skips it. The database is read without ever writing to the data tree."
    ))

    lines.extend(["", "Returned dtypes"])
    if values["dtype"]:
        lines.extend(_wrap(f"Measurements ({', '.join(values['numeric_columns'])}): float64; see Values above."))
    lines.extend(_wrap(f"Categorical, where present: {', '.join(dtypes['categorical'])}."))
    if dtypes["categorical_value"]:
        lines.extend(_wrap("value is categorical for this feature, because it holds states rather than quantities."))
    lines.extend(_wrap(
        "Nullable boolean: " + ", ".join(dtypes["nullable_boolean"])
        + ("; boolean: include_by_default" if curated else "") + ". Timestamps are timezone-aware UTC."
    ))

    lines.extend(["", "Scale"])
    limit_text = "no limit" if max_rows is None else f"{max_rows:,} rows"
    lines.extend(_wrap(
        f"get_data() refuses requests above max_rows ({limit_text} in this session); pass max_rows=None to "
        f"lift it for one call. Size a request first with {usage['profile'][len('print('):-1]}."
    ))

    lines.extend([
        "",
        "Future resampling declaration (not executed by this loader)",
        f"  support={resampling['support']}; strategy={resampling['strategy']}; aggregation={resampling['aggregation']}",
        "",
        "Important caveats",
    ])
    if guide.important_caveats:
        for item in guide.important_caveats:
            lines.extend(_wrap(item, indent="  - "))
    else:
        lines.append("  - None recorded in feature guidance.")
    if guide.not_equivalent_to:
        lines.extend(("", "Do not interpret as"))
        for item in guide.not_equivalent_to:
            lines.extend(_wrap(item, indent="  - "))

    lines.extend(["", "Usage"])
    lines.extend(f"  {example}" for example in usage.values())

    lines.extend(("", "Evidence references"))
    lines.extend(f"  - {evidence_id}" for evidence_id in evidence_ids)
    if include_evidence and evidence_payload:
        lines.extend(("", "Evidence details"))
        for evidence_id, source in evidence_payload.items():
            lines.append(f"  {evidence_id}: {source['title']} — {source['organization']} ({source['kind']})")
            if source.get("brief_summary"):
                lines.extend(_wrap(source["brief_summary"], indent="    "))
            if source.get("limitations"):
                lines.extend(_wrap(f"Limitations: {source['limitations']}", indent="    "))
            if source.get("locator"):
                lines.append(f"    {source['locator']}")

    return InfoReport(
        kind="feature",
        title=f"{feature} DataLoader information",
        payload=payload,
        text="\n".join(lines),
    )
