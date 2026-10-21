"""
Wearable Data Processing and Modeling project

Non-destructive Google Fitness feature-file curation engine (CWP3).

The engine consumes one accepted native feature CSV and writes one curated CSV.
Native columns remain an unchanged prefix, native row order and row count are
preserved, and only reviewed policy-derived columns are appended.  No row is
deleted, clipped, repaired, resampled, or imputed.
"""


from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from itertools import zip_longest
from typing import Iterable, Mapping, Sequence

from wearable_project.exceptions import CurationError

from .context import build_sleep_context, build_step_count_context
from .decisions import decisions_fingerprint, get_decision
from .evidence import evidence_fingerprint
from .guidance import guidance_fingerprint
from .models import DecisionStatus, FeaturePolicy, StepCountRole, to_primitive
from .registry import get_policy, registry_fingerprint
from .rules import rules_fingerprint
from .strategies import RowAnnotation, annotate_row, validate_engine_rule_coverage


ENGINE_VERSION = "google-fitness-curation-engine-cwp3-0.1.0"
DERIVED_COLUMN_ORDER: tuple[str, ...] = (
    "canonical_value",
    "canonical_unit",
    "curation_unit_status",
    "unit_evidence",
    "acquisition_method",
    "derivation_method",
    "tracking_status",
    "step_count_role",
    "curation_status",
    "curation_flags",
    "include_by_default",
)

_STATUS_VALUES = frozenset({"pass", "review", "exclude_default", ""})
_INCLUDE_VALUES = frozenset({"0", "1", ""})
_ROLE_VALUES = frozenset({item.value for item in StepCountRole} | {""})


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_field_limit()


@dataclass(frozen=True, slots=True)
class CuratedFeatureInfo:
    feature: str
    filename: str
    native_rows: int
    curated_rows: int
    native_sha256: str
    curated_sha256: str
    size_bytes: int
    active_derived_columns: tuple[str, ...]
    pass_rows: int
    review_rows: int
    exclude_default_rows: int
    included_by_default_rows: int
    excluded_by_default_rows: int
    canonical_value_rows: int
    flag_counts: dict[str, int]
    acquisition_counts: dict[str, int]
    derivation_counts: dict[str, int]
    tracking_counts: dict[str, int]
    step_count_role_counts: dict[str, int]
    unit_status_counts: dict[str, int]
    unit_evidence_counts: dict[str, int]
    engine_version: str
    policy_fingerprint: str
    registry_fingerprint: str
    evidence_fingerprint: str
    rules_fingerprint: str
    decisions_fingerprint: str
    guidance_fingerprint: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureCurationResult:
    info: CuratedFeatureInfo

    def as_dict(self) -> dict[str, object]:
        return {"info": self.info.as_dict()}


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_policy_fingerprint(policy: FeaturePolicy) -> str:
    payload = json.dumps(
        to_primitive(policy),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


_REQUIRED_ACCEPTED_DECISIONS: tuple[str, ...] = (
    "activity_multimeasurement_rows",
    "activity_quantity_unresolved",
    "negative_calories_buckets",
    "non_destructive_curation",
    "provenance_dimensions",
    "sleep_native_states_preserved",
    "source_day_label_context_only",
    "status_inclusion_separate",
    "step_count_no_canonical_day",
    "step_count_preference",
    "step_count_preserve_all_rows",
    "step_count_unit_mapping",
    "unknown_feature_fallback",
    "weight_nonpoint_preserved",
)
_REQUIRED_PROVISIONAL_DECISIONS: tuple[str, ...] = (
    "calories_unit_mapping",
    "distance_unit_mapping",
    "height_unit_mapping",
    "provenance_classifier",
    "weight_unit_mapping",
)


def validate_engine_contract() -> tuple[str, ...]:
    """Validate the exact reviewed decision boundary executed by CWP3."""

    issues = list(validate_engine_rule_coverage())
    for decision_id in _REQUIRED_ACCEPTED_DECISIONS:
        if get_decision(decision_id).status is not DecisionStatus.ACCEPTED:
            issues.append(f"CWP3 executable decision is not accepted: {decision_id}")
    for decision_id in _REQUIRED_PROVISIONAL_DECISIONS:
        if get_decision(decision_id).status is not DecisionStatus.PROVISIONAL:
            issues.append(f"CWP3 abstention boundary changed from provisional: {decision_id}")
    return tuple(sorted(set(issues)))


def _read_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return tuple(csv.reader(handle).__next__())


def _validate_native_header(feature: str, native_fields: tuple[str, ...], policy: FeaturePolicy) -> None:
    if policy.unknown_fallback:
        collisions = sorted(set(native_fields) & set(policy.derived_columns))
        if collisions:
            raise CurationError(f"Unknown feature {feature!r} collides with curation columns: {collisions}")
        if not native_fields:
            raise CurationError(f"Native CSV for unknown feature {feature!r} has no header")
        return
    if native_fields != policy.native_schema.native_columns:
        raise CurationError(
            f"Native schema mismatch for {feature}: expected {policy.native_schema.native_columns}, observed {native_fields}"
        )


def _load_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        return fields, list(reader)


def _stream_rows(path: Path) -> tuple[tuple[str, ...], Iterable[dict[str, str]]]:
    handle = path.open("r", encoding="utf-8-sig", newline="")
    reader = csv.DictReader(handle)
    fields = tuple(reader.fieldnames or ())

    def iterator() -> Iterable[dict[str, str]]:
        try:
            yield from reader
        finally:
            handle.close()

    return fields, iterator()


def _ordered_active_columns(policy: FeaturePolicy, active: set[str]) -> tuple[str, ...]:
    invalid = sorted(active - set(policy.derived_columns))
    if invalid:
        raise CurationError(f"Engine produced columns outside policy for {policy.feature}: {invalid}")
    return tuple(column for column in DERIVED_COLUMN_ORDER if column in active)


def _annotation_values(annotation: RowAnnotation, policy: FeaturePolicy) -> dict[str, str]:
    values = annotation.as_sparse_dict(policy)
    unexpected = sorted(set(values) - set(policy.derived_columns))
    if unexpected:
        raise CurationError(f"Annotation contains undeclared columns for {policy.feature}: {unexpected}")
    return values


def _write_projected(
    native_path: Path,
    output_path: Path,
    annotation_path: Path,
    append_columns: Sequence[str],
) -> int:
    with native_path.open("r", encoding="utf-8-sig", newline="") as source, \
         annotation_path.open("r", encoding="utf-8") as annotations, \
         output_path.open("w", encoding="utf-8", newline="") as target:
        reader = csv.DictReader(source)
        native_fields = list(reader.fieldnames or [])
        writer = csv.DictWriter(
            target,
            fieldnames=native_fields + list(append_columns),
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
            extrasaction="raise",
        )
        writer.writeheader()
        count = 0
        for index, row in enumerate(reader):
            line = annotations.readline()
            if not line:
                raise CurationError(f"Missing annotation for {native_path} row {index}")
            payload = json.loads(line)
            if payload.get("index") != index:
                raise CurationError(f"Annotation index mismatch for {native_path}: {payload.get('index')} != {index}")
            derived = payload.get("values", {})
            if "curation_status" in append_columns and "curation_status" not in derived:
                derived["curation_status"] = "pass"
            if "include_by_default" in append_columns and "include_by_default" not in derived:
                derived["include_by_default"] = "1"
            output = dict(row)
            for column in append_columns:
                output[column] = derived.get(column, "")
            writer.writerow(output)
            count += 1
        if annotations.readline():
            raise CurationError(f"Annotation stream has more rows than native file {native_path}")
        target.flush()
        os.fsync(target.fileno())
    return count


def validate_curated_feature_file(
    native_path: Path,
    curated_path: Path,
    feature: str | None = None,
    *,
    expected_derived_columns: Sequence[str] | None = None,
) -> int:
    """Validate native-prefix, row-order, row-count, and derived-column contracts."""

    feature = feature or native_path.stem
    policy = get_policy(feature)
    with native_path.open("r", encoding="utf-8-sig", newline="") as native_handle, \
         curated_path.open("r", encoding="utf-8-sig", newline="") as curated_handle:
        native_reader = csv.DictReader(native_handle)
        curated_reader = csv.DictReader(curated_handle)
        native_fields = tuple(native_reader.fieldnames or ())
        curated_fields = tuple(curated_reader.fieldnames or ())
        if curated_fields[: len(native_fields)] != native_fields:
            raise CurationError(f"Curated native prefix differs for {feature}")
        derived = curated_fields[len(native_fields):]
        if expected_derived_columns is not None and tuple(expected_derived_columns) != derived:
            raise CurationError(
                f"Derived columns differ for {feature}: expected {tuple(expected_derived_columns)}, observed {derived}"
            )
        if not set(derived) <= set(policy.derived_columns):
            raise CurationError(f"Curated file has undeclared derived columns for {feature}: {sorted(set(derived)-set(policy.derived_columns))}")

        count = 0
        for index, pair in enumerate(zip_longest(native_reader, curated_reader), start=0):
            native_row, curated_row = pair
            if native_row is None or curated_row is None:
                raise CurationError(f"Native/curated row-count mismatch for {feature} at row {index}")
            for column in native_fields:
                if curated_row.get(column, "") != native_row.get(column, ""):
                    raise CurationError(f"Native value changed for {feature} row {index} column {column}")
            status = curated_row.get("curation_status", "")
            include = curated_row.get("include_by_default", "")
            role = curated_row.get("step_count_role", "")
            if status not in _STATUS_VALUES:
                raise CurationError(f"Invalid curation_status {status!r} for {feature} row {index}")
            if include not in _INCLUDE_VALUES:
                raise CurationError(f"Invalid include_by_default {include!r} for {feature} row {index}")
            if role not in _ROLE_VALUES:
                raise CurationError(f"Invalid step_count_role {role!r} for {feature} row {index}")
            count += 1
        return count


def curate_feature_file(native_path: Path, destination: Path, feature: str | None = None) -> FeatureCurationResult:
    """Curate one native Google Fitness feature CSV without modifying the input."""

    native_path = Path(native_path).resolve()
    destination = Path(destination).resolve()
    if native_path == destination:
        raise CurationError("Native and curated feature paths must differ")
    if not native_path.is_file():
        raise CurationError(f"Native feature file does not exist: {native_path}")

    feature = feature or native_path.stem
    policy = get_policy(feature)
    native_sha = sha256_file(native_path)

    if feature in {"StepCount", "Sleep"}:
        native_fields, materialized_rows = _load_rows(native_path)
        iterator: Iterable[Mapping[str, str]] = materialized_rows
    else:
        native_fields, iterator = _stream_rows(native_path)
        materialized_rows = []
    _validate_native_header(feature, native_fields, policy)

    step_context = build_step_count_context(materialized_rows) if feature == "StepCount" else {}
    sleep_context = build_sleep_context(materialized_rows) if feature == "Sleep" else {}

    active_columns: set[str] = set(policy.output.fixed_columns)
    status_counts: Counter[str] = Counter()
    inclusion_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    acquisition_counts: Counter[str] = Counter()
    derivation_counts: Counter[str] = Counter()
    tracking_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    unit_status_counts: Counter[str] = Counter()
    unit_evidence_counts: Counter[str] = Counter()
    canonical_rows = 0
    native_rows = 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    annotation_fd, annotation_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".annotations.jsonl", dir=destination.parent
    )
    os.close(annotation_fd)
    annotations_path = Path(annotation_name)
    output_fd, output_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".curated.tmp", dir=destination.parent
    )
    os.close(output_fd)
    output_path = Path(output_name)

    try:
        with annotations_path.open("w", encoding="utf-8") as annotation_file:
            for index, row in enumerate(iterator):
                annotation = annotate_row(
                    feature,
                    row,
                    policy,
                    step_context=step_context.get(index),
                    sleep_context=sleep_context.get(index),
                )
                values = _annotation_values(annotation, policy)
                active_columns.update(column for column, value in values.items() if value not in (None, ""))
                annotation_file.write(json.dumps(
                    {"index": index, "values": values},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ) + "\n")

                status_counts[annotation.status.value] += 1
                inclusion_counts["included" if annotation.include_by_default else "excluded"] += 1
                flag_counts.update(annotation.flags)
                acquisition_counts[annotation.values.get("acquisition_method", "unclassified")] += 1
                derivation_counts[annotation.values.get("derivation_method", "unclassified")] += 1
                tracking_counts[annotation.values.get("tracking_status", "not_applicable")] += 1
                role_counts[annotation.values.get("step_count_role", "not_applicable")] += 1
                unit_status_counts[annotation.values.get("curation_unit_status", "not_applicable")] += 1
                unit_evidence_counts[annotation.values.get("unit_evidence", "none")] += 1
                if annotation.values.get("canonical_value") not in (None, ""):
                    canonical_rows += 1
                native_rows += 1
            annotation_file.flush()
            os.fsync(annotation_file.fileno())

        ordered_columns = _ordered_active_columns(policy, active_columns)
        if ordered_columns:
            curated_rows = _write_projected(native_path, output_path, annotations_path, ordered_columns)
        else:
            shutil.copyfile(native_path, output_path)
            curated_rows = native_rows

        validate_curated_feature_file(
            native_path,
            output_path,
            feature,
            expected_derived_columns=ordered_columns,
        )
        if curated_rows != native_rows:
            raise CurationError(f"Row-count mismatch for {feature}: native={native_rows}, curated={curated_rows}")
        if sha256_file(native_path) != native_sha:
            raise CurationError(f"Native feature changed during curation: {native_path}")

        os.replace(output_path, destination)
        curated_sha = sha256_file(destination)
        info = CuratedFeatureInfo(
            feature=feature,
            filename=destination.name,
            native_rows=native_rows,
            curated_rows=curated_rows,
            native_sha256=native_sha,
            curated_sha256=curated_sha,
            size_bytes=destination.stat().st_size,
            active_derived_columns=ordered_columns,
            pass_rows=status_counts["pass"],
            review_rows=status_counts["review"],
            exclude_default_rows=status_counts["exclude_default"],
            included_by_default_rows=inclusion_counts["included"],
            excluded_by_default_rows=inclusion_counts["excluded"],
            canonical_value_rows=canonical_rows,
            flag_counts=dict(sorted(flag_counts.items())),
            acquisition_counts=dict(sorted(acquisition_counts.items())),
            derivation_counts=dict(sorted(derivation_counts.items())),
            tracking_counts=dict(sorted(tracking_counts.items())),
            step_count_role_counts=dict(sorted(role_counts.items())),
            unit_status_counts=dict(sorted(unit_status_counts.items())),
            unit_evidence_counts=dict(sorted(unit_evidence_counts.items())),
            engine_version=ENGINE_VERSION,
            policy_fingerprint=feature_policy_fingerprint(policy),
            registry_fingerprint=registry_fingerprint(),
            evidence_fingerprint=evidence_fingerprint(),
            rules_fingerprint=rules_fingerprint(),
            decisions_fingerprint=decisions_fingerprint(),
            guidance_fingerprint=guidance_fingerprint(),
        )
        return FeatureCurationResult(info)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        annotations_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


_ENGINE_CONTRACT_ISSUES = validate_engine_contract()
if _ENGINE_CONTRACT_ISSUES:
    raise RuntimeError("Invalid Google Fitness CWP3 engine contract: " + "; ".join(_ENGINE_CONTRACT_ISSUES))
