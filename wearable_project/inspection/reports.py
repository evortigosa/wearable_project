"""
Wearable Data Processing and Modeling project
Deterministic report writers for the mixed-source raw-export profiler.
"""


from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from .profile import MixedSourceProfiler, ProfilerState, canonical_json


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _empty_numeric_summary() -> dict[str, Any]:
    return {
        "numeric_count": 0,
        "finite_count": 0,
        "nonfinite_count": 0,
        "numeric_min": None,
        "numeric_p01": None,
        "numeric_p25": None,
        "numeric_median": None,
        "numeric_p75": None,
        "numeric_p99": None,
        "numeric_max": None,
        "numeric_mean": None,
        "numeric_zero_count": 0,
        "numeric_negative_count": 0,
        "quantile_sample_size": 0,
        "quantile_method": "no_values",
    }


def _source_sequence(participant) -> list[dict[str, Any]]:
    months = sorted(participant.months)
    result = []
    for month in months:
        sources = sorted(source for (seen_month, source), rows in participant.month_source_rows.items() if seen_month == month and rows)
        if sources:
            result.append({"month": month, "sources": sources})
    return result


def _source_set_change_count(sequence: list[dict[str, Any]]) -> int:
    changes = 0
    previous: tuple[str, ...] | None = None
    for entry in sequence:
        current = tuple(entry["sources"])
        if previous is not None and current != previous:
            changes += 1
        previous = current
    return changes


def write_reports(output_dir: Path, profiler: MixedSourceProfiler, manifest: Mapping[str, Any]) -> None:
    state = profiler.state
    summary = profiler.summary()
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "run_manifest.json", dict(manifest))
    _write_participants(output_dir, state)
    _write_participant_sources(output_dir, state)
    _write_participant_source_months(output_dir, state)
    _write_version_transitions(output_dir, state)
    _write_source_labels(output_dir, state)
    _write_features(output_dir, state)
    _write_schemas(output_dir, state)
    _write_representatives(output_dir, state)
    _write_fields(output_dir, state)
    _write_temporal(output_dir, state)
    _write_temporal_alignment(output_dir, state)
    _write_duplicate_calibration(output_dir, state)
    _write_activity(output_dir, state)
    _write_provenance(output_dir, state)
    _write_steps(output_dir, state)
    _write_focused_anomalies(output_dir, state)
    _write_diagnostics(output_dir, state)
    _write_input_issues(output_dir, state)
    _write_execution(output_dir, state)
    (output_dir / "REPORT.md").write_text(_markdown_report(summary, state), encoding="utf-8")


def _write_participants(output_dir: Path, state: ProfilerState) -> None:
    rows = []
    for participant in state.participants.values():
        source_keys = sorted(participant.source_slices)
        sequence = _source_sequence(participant)
        rows.append({
            "participant_id": participant.participant_id,
            "data_bearing": int(participant.outer_rows > 0),
            "empty_export_directory": int(participant.file_count == 0 and participant.outer_rows == 0),
            "has_month_files": int(participant.file_count > 0),
            "month_files_without_outer_rows": int(participant.file_count > 0 and participant.outer_rows == 0),
            "month_file_count": participant.file_count,
            "first_month": min(participant.months) if participant.months else None,
            "last_month": max(participant.months) if participant.months else None,
            "missing_months_within_range": ";".join(_missing_months_local(participant.months)),
            "export_ended_present": int(participant.export_ended_present),
            "source_bytes": participant.source_bytes,
            "outer_rows": participant.outer_rows,
            "decoded_outer_rows": participant.decoded_outer_rows,
            "payload_items": participant.payload_items,
            "source_count": len(source_keys),
            "source_keys": ";".join(source_keys),
            "raw_source_labels": ";".join(sorted(participant.raw_source_labels.values())),
            "mixed_outer_sources": int(len(source_keys) > 1),
            "months_with_multiple_sources": sum(len(entry["sources"]) > 1 for entry in sequence),
            "source_set_change_count": _source_set_change_count(sequence),
            "source_sequence_json": canonical_json(sequence),
        })
    fields = (
        "participant_id", "data_bearing", "empty_export_directory", "has_month_files",
        "month_files_without_outer_rows", "month_file_count", "first_month", "last_month", "missing_months_within_range",
        "export_ended_present", "source_bytes", "outer_rows", "decoded_outer_rows", "payload_items",
        "source_count", "source_keys", "raw_source_labels", "mixed_outer_sources",
        "months_with_multiple_sources", "source_set_change_count", "source_sequence_json",
    )
    write_csv(output_dir / "participants.csv", sorted(rows, key=lambda row: row["participant_id"]), fields)


def _missing_months_local(months: set[str]) -> list[str]:
    if len(months) < 2:
        return []
    def idx(value: str) -> int:
        year, month = value.split("-")
        return int(year) * 12 + int(month) - 1
    indices = sorted(idx(value) for value in months)
    missing = []
    for index in range(indices[0], indices[-1] + 1):
        year, month0 = divmod(index, 12)
        value = f"{year:04d}-{month0 + 1:02d}"
        if value not in months:
            missing.append(value)
    return missing


def _write_participant_sources(output_dir: Path, state: ProfilerState) -> None:
    rows = []
    for participant in state.participants.values():
        for source_key, stats in participant.source_slices.items():
            rows.append({
                "participant_id": participant.participant_id,
                "source_key": source_key,
                "raw_source_label": participant.raw_source_labels.get(source_key),
                "outer_rows": stats.rows,
                "month_count": len(stats.months),
                "first_month": min(stats.months) if stats.months else None,
                "last_month": max(stats.months) if stats.months else None,
                "feature_count": len(stats.features),
                "features": ";".join(sorted(stats.features)),
                "collecting_method_versions": ";".join(sorted(stats.versions)),
                "first_outer_datetime": stats.first_outer_datetime,
                "last_outer_datetime": stats.last_outer_datetime,
            })
    fields = (
        "participant_id", "source_key", "raw_source_label", "outer_rows", "month_count", "first_month",
        "last_month", "feature_count", "features", "collecting_method_versions", "first_outer_datetime",
        "last_outer_datetime",
    )
    write_csv(output_dir / "participant_sources.csv", sorted(rows, key=lambda r: (r["participant_id"], r["source_key"])), fields)


def _write_participant_source_months(output_dir: Path, state: ProfilerState) -> None:
    rows = []
    for participant in state.participants.values():
        for (month, source_key), count in participant.month_source_rows.items():
            rows.append({
                "participant_id": participant.participant_id,
                "canonical_month": month,
                "source_key": source_key,
                "raw_source_label": participant.raw_source_labels.get(source_key),
                "outer_rows": count,
                "feature_count": len(participant.month_source_features[(month, source_key)]),
                "features": ";".join(sorted(participant.month_source_features[(month, source_key)])),
            })
    fields = ("participant_id", "canonical_month", "source_key", "raw_source_label", "outer_rows", "feature_count", "features")
    write_csv(output_dir / "participant_source_months.csv", sorted(rows, key=lambda r: (r["participant_id"], r["canonical_month"], r["source_key"])), fields)


def _write_version_transitions(output_dir: Path, state: ProfilerState) -> None:
    versions_by_month: dict[tuple[str, str, str, str, str], set[str]] = {}
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    version_aggregate: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    for key, stats in state.participant_feature_version_months.items():
        participant_id, raw_source, source_key, feature, version, month = key
        versions_by_month.setdefault(
            (participant_id, raw_source, source_key, feature, month), set()
        ).add(version)
        group = grouped.setdefault(
            (participant_id, raw_source, source_key, feature), {}
        )
        month_entry = group.setdefault(month, {})
        month_entry[version] = stats
        aggregate = version_aggregate.setdefault(
            (participant_id, raw_source, source_key, feature, version),
            {
                "months": set(),
                "outer_rows": 0,
                "decoded_outer_rows": 0,
                "payload_items": 0,
                "first_outer_datetime": None,
                "last_outer_datetime": None,
            },
        )
        aggregate["months"].add(month)
        aggregate["outer_rows"] += stats.outer_rows
        aggregate["decoded_outer_rows"] += stats.decoded_outer_rows
        aggregate["payload_items"] += stats.payload_items
        if stats.first_outer_datetime is not None:
            aggregate["first_outer_datetime"] = (
                stats.first_outer_datetime
                if aggregate["first_outer_datetime"] is None
                else min(aggregate["first_outer_datetime"], stats.first_outer_datetime)
            )
        if stats.last_outer_datetime is not None:
            aggregate["last_outer_datetime"] = (
                stats.last_outer_datetime
                if aggregate["last_outer_datetime"] is None
                else max(aggregate["last_outer_datetime"], stats.last_outer_datetime)
            )

    month_rows = []
    for key, stats in state.participant_feature_version_months.items():
        participant_id, raw_source, source_key, feature, version, month = key
        versions = sorted(versions_by_month[(participant_id, raw_source, source_key, feature, month)])
        month_rows.append({
            "participant_id": participant_id,
            "raw_source_label": raw_source,
            "source_key": source_key,
            "feature": feature,
            "collecting_method_version": version,
            "canonical_month": month,
            "outer_rows": stats.outer_rows,
            "decoded_outer_rows": stats.decoded_outer_rows,
            "payload_items": stats.payload_items,
            "first_outer_datetime": stats.first_outer_datetime,
            "last_outer_datetime": stats.last_outer_datetime,
            "versions_in_same_month": ";".join(versions),
            "multiple_versions_same_month": int(len(versions) > 1),
        })
    write_csv(
        output_dir / "participant_feature_version_months.csv",
        sorted(
            month_rows,
            key=lambda row: (
                row["participant_id"], row["source_key"], row["feature"],
                row["canonical_month"], row["collecting_method_version"],
            ),
        ),
        (
            "participant_id", "raw_source_label", "source_key", "feature",
            "collecting_method_version", "canonical_month", "outer_rows",
            "decoded_outer_rows", "payload_items", "first_outer_datetime",
            "last_outer_datetime", "versions_in_same_month",
            "multiple_versions_same_month",
        ),
    )

    version_rows = []
    for key, aggregate in version_aggregate.items():
        participant_id, raw_source, source_key, feature, version = key
        months = sorted(aggregate["months"])
        overlap_months = [
            month
            for month in months
            if len(versions_by_month[(participant_id, raw_source, source_key, feature, month)]) > 1
        ]
        version_rows.append({
            "participant_id": participant_id,
            "raw_source_label": raw_source,
            "source_key": source_key,
            "feature": feature,
            "collecting_method_version": version,
            "month_count": len(months),
            "first_month": months[0] if months else None,
            "last_month": months[-1] if months else None,
            "same_month_overlap_count": len(overlap_months),
            "same_month_overlap_months": ";".join(overlap_months),
            "outer_rows": aggregate["outer_rows"],
            "decoded_outer_rows": aggregate["decoded_outer_rows"],
            "payload_items": aggregate["payload_items"],
            "first_outer_datetime": aggregate["first_outer_datetime"],
            "last_outer_datetime": aggregate["last_outer_datetime"],
        })
    write_csv(
        output_dir / "participant_feature_versions.csv",
        sorted(
            version_rows,
            key=lambda row: (
                row["participant_id"], row["source_key"], row["feature"],
                row["collecting_method_version"],
            ),
        ),
        (
            "participant_id", "raw_source_label", "source_key", "feature",
            "collecting_method_version", "month_count", "first_month", "last_month",
            "same_month_overlap_count", "same_month_overlap_months", "outer_rows",
            "decoded_outer_rows", "payload_items", "first_outer_datetime",
            "last_outer_datetime",
        ),
    )

    transition_rows = []
    for group_key, month_map in grouped.items():
        participant_id, raw_source, source_key, feature = group_key
        months = sorted(month_map)
        sequence = [
            {"month": month, "versions": sorted(month_map[month])}
            for month in months
        ]
        all_versions = sorted({version for entry in month_map.values() for version in entry})
        transition_count = sum(
            tuple(sequence[index]["versions"]) != tuple(sequence[index - 1]["versions"])
            for index in range(1, len(sequence))
        )
        overlap_months = [entry["month"] for entry in sequence if len(entry["versions"]) > 1]
        blocks: dict[str, int] = {}
        first_last: dict[str, dict[str, str]] = {}
        for version in all_versions:
            present_flags = [version in entry["versions"] for entry in sequence]
            blocks[version] = sum(
                present and (index == 0 or not present_flags[index - 1])
                for index, present in enumerate(present_flags)
            )
            present_months = [entry["month"] for entry in sequence if version in entry["versions"]]
            first_last[version] = {
                "first_month": present_months[0],
                "last_month": present_months[-1],
            }
        has_reversion = any(value > 1 for value in blocks.values())
        if len(all_versions) <= 1:
            transition_class = "single_version"
        elif overlap_months and len(overlap_months) == len(months):
            transition_class = "concurrent_versions"
        elif has_reversion:
            transition_class = "interleaved_or_reversion"
        elif overlap_months:
            transition_class = "overlapping_transition"
        else:
            transition_class = "sequential_transition"
        transition_rows.append({
            "participant_id": participant_id,
            "raw_source_label": raw_source,
            "source_key": source_key,
            "feature": feature,
            "version_count": len(all_versions),
            "versions": ";".join(all_versions),
            "month_count": len(months),
            "first_month": months[0] if months else None,
            "last_month": months[-1] if months else None,
            "version_set_transition_count": transition_count,
            "same_month_multiple_version_count": len(overlap_months),
            "same_month_multiple_version_months": ";".join(overlap_months),
            "has_version_reversion": int(has_reversion),
            "transition_class": transition_class,
            "version_blocks_json": canonical_json(blocks),
            "version_first_last_json": canonical_json(first_last),
            "version_month_sequence_json": canonical_json(sequence),
        })
    write_csv(
        output_dir / "participant_feature_version_transitions.csv",
        sorted(
            transition_rows,
            key=lambda row: (row["participant_id"], row["source_key"], row["feature"]),
        ),
        (
            "participant_id", "raw_source_label", "source_key", "feature",
            "version_count", "versions", "month_count", "first_month", "last_month",
            "version_set_transition_count", "same_month_multiple_version_count",
            "same_month_multiple_version_months", "has_version_reversion",
            "transition_class", "version_blocks_json", "version_first_last_json",
            "version_month_sequence_json",
        ),
    )


def _write_source_labels(output_dir: Path, state: ProfilerState) -> None:
    rows = [
        {"raw_source_label": raw, "source_key": key, "outer_rows": count}
        for (raw, key), count in state.source_labels.items()
    ]
    write_csv(output_dir / "source_labels.csv", sorted(rows, key=lambda r: (r["source_key"], r["raw_source_label"])), ("raw_source_label", "source_key", "outer_rows"))


def _write_features(output_dir: Path, state: ProfilerState) -> None:
    rows = []
    for stats in state.features.values():
        rows.append({
            "raw_source_label": stats.raw_source_label,
            "source_key": stats.source_key,
            "feature": stats.feature,
            "collecting_method_version": stats.version,
            "participant_count": len(stats.participants),
            "source_file_count": len(stats.files),
            "month_count": len(stats.months),
            "first_month": min(stats.months) if stats.months else None,
            "last_month": max(stats.months) if stats.months else None,
            "outer_rows": stats.outer_rows,
            "decoded_outer_rows": stats.decoded_outer_rows,
            "payload_decode_failures": stats.decode_failures,
            "payload_items": stats.payload_items,
            "payload_items_per_decoded_outer_row": stats.payload_items / stats.decoded_outer_rows if stats.decoded_outer_rows else None,
            "payload_top_types_json": canonical_json(dict(sorted(stats.payload_top_types.items()))),
            "list_length_distribution_json": canonical_json(dict(stats.list_lengths.items())),
            "list_length_distribution_approximate": int(stats.list_lengths.approximate),
            "first_outer_datetime": stats.first_outer_datetime,
            "last_outer_datetime": stats.last_outer_datetime,
        })
    fields = (
        "raw_source_label", "source_key", "feature", "collecting_method_version", "participant_count",
        "source_file_count", "month_count", "first_month", "last_month", "outer_rows", "decoded_outer_rows",
        "payload_decode_failures", "payload_items", "payload_items_per_decoded_outer_row",
        "payload_top_types_json", "list_length_distribution_json", "list_length_distribution_approximate",
        "first_outer_datetime", "last_outer_datetime",
    )
    write_csv(output_dir / "source_feature_versions.csv", sorted(rows, key=lambda r: (r["source_key"], r["feature"], r["collecting_method_version"], r["raw_source_label"])), fields)


def _write_schemas(output_dir: Path, state: ProfilerState) -> None:
    rows = []
    for key, count in state.schema_counts.items():
        raw, source_key, feature, version, signature = key
        rows.append({
            "raw_source_label": raw,
            "source_key": source_key,
            "feature": feature,
            "collecting_method_version": version,
            "item_schema": signature,
            "item_count": count,
            "participant_count": len(state.schema_participants[key]),
        })
    fields = ("raw_source_label", "source_key", "feature", "collecting_method_version", "item_schema", "item_count", "participant_count")
    write_csv(output_dir / "payload_schema_signatures.csv", sorted(rows, key=lambda r: (r["source_key"], r["feature"], r["collecting_method_version"], -int(r["item_count"]), r["item_schema"])), fields)


def _write_representatives(output_dir: Path, state: ProfilerState) -> None:
    representatives = [
        state.representative_payloads[key]
        for key in sorted(state.representative_payloads, key=lambda item: (item[1], item[2], item[3], item[4]))
    ]
    write_json(output_dir / "representative_payloads.json", representatives)


def _write_fields(output_dir: Path, state: ProfilerState) -> None:
    field_rows = []
    category_rows = []
    for stats in state.fields.values():
        numeric = stats.numeric.summary()
        list_lengths = stats.list_lengths.summary()
        field_rows.append({
            "raw_source_label": stats.raw_source_label,
            "source_key": stats.source_key,
            "feature": stats.feature,
            "collecting_method_version": stats.version,
            "field_path": stats.path,
            "present_count": stats.present_count,
            "null_count": stats.null_count,
            "observed_types_json": canonical_json(dict(sorted(stats.types.items()))),
            "categorical_retained_unique_count": len(stats.categorical.counts),
            "categorical_approximate": int(stats.categorical.approximate),
            "categorical_pruned_mass": stats.categorical.pruned_mass,
            "string_too_long_count": stats.string_too_long_count,
            "nested_list_truncated_count": stats.nested_list_truncated_count,
            "list_length_count": list_lengths["numeric_count"],
            "list_length_min": list_lengths["numeric_min"],
            "list_length_median": list_lengths["numeric_median"],
            "list_length_max": list_lengths["numeric_max"],
            **numeric,
        })
        for value, count in stats.categorical.items():
            category_rows.append({
                "raw_source_label": stats.raw_source_label,
                "source_key": stats.source_key,
                "feature": stats.feature,
                "collecting_method_version": stats.version,
                "field_path": stats.path,
                "value": value,
                "count_lower_bound": count,
                "approximate": int(stats.categorical.approximate),
            })
    field_fields = (
        "raw_source_label", "source_key", "feature", "collecting_method_version", "field_path",
        "present_count", "null_count", "observed_types_json", "categorical_retained_unique_count",
        "categorical_approximate", "categorical_pruned_mass", "string_too_long_count",
        "nested_list_truncated_count", "list_length_count", "list_length_min", "list_length_median",
        "list_length_max", "numeric_count", "finite_count", "nonfinite_count", "numeric_min", "numeric_p01",
        "numeric_p25", "numeric_median", "numeric_p75", "numeric_p99", "numeric_max", "numeric_mean",
        "numeric_zero_count", "numeric_negative_count", "quantile_sample_size", "quantile_method",
    )
    write_csv(output_dir / "field_profile.csv", sorted(field_rows, key=lambda r: (r["source_key"], r["feature"], r["collecting_method_version"], r["field_path"])), field_fields)
    category_fields = ("raw_source_label", "source_key", "feature", "collecting_method_version", "field_path", "value", "count_lower_bound", "approximate")
    write_csv(output_dir / "categorical_values.csv", sorted(category_rows, key=lambda r: (r["source_key"], r["feature"], r["collecting_method_version"], r["field_path"], -int(r["count_lower_bound"]), r["value"])), category_fields)


def _write_temporal(output_dir: Path, state: ProfilerState) -> None:
    rows = []
    for stats in state.temporal.values():
        duration = stats.duration.summary()
        rows.append({
            "raw_source_label": stats.raw_source_label,
            "source_key": stats.source_key,
            "feature": stats.feature,
            "collecting_method_version": stats.version,
            "item_count": stats.item_count,
            "start_present": stats.start_present,
            "end_present": stats.end_present,
            "invalid_start": stats.invalid_start,
            "invalid_end": stats.invalid_end,
            "negative_duration": stats.negative_duration,
            "zero_duration": stats.zero_duration,
            **{f"duration_{key}": value for key, value in duration.items()},
        })
    fields = (
        "raw_source_label", "source_key", "feature", "collecting_method_version", "item_count",
        "start_present", "end_present", "invalid_start", "invalid_end", "negative_duration", "zero_duration",
        "duration_numeric_count", "duration_finite_count", "duration_nonfinite_count", "duration_numeric_min",
        "duration_numeric_p01", "duration_numeric_p25", "duration_numeric_median", "duration_numeric_p75",
        "duration_numeric_p99", "duration_numeric_max", "duration_numeric_mean", "duration_numeric_zero_count",
        "duration_numeric_negative_count", "duration_quantile_sample_size", "duration_quantile_method",
    )
    write_csv(output_dir / "timestamp_profile.csv", sorted(rows, key=lambda r: (r["source_key"], r["feature"], r["collecting_method_version"])), fields)


def _write_temporal_alignment(output_dir: Path, state: ProfilerState) -> None:
    rows = [
        {
            "raw_source_label": raw_source,
            "source_key": source_key,
            "feature": feature,
            "collecting_method_version": version,
            "relationship": relationship,
            "value": value,
            "count": count,
        }
        for (
            raw_source, source_key, feature, version, relationship, value,
        ), count in state.temporal_alignment_counts.items()
    ]
    write_csv(
        output_dir / "temporal_alignment_counts.csv",
        sorted(
            rows,
            key=lambda row: (
                row["source_key"], row["feature"], row["collecting_method_version"],
                row["relationship"], row["value"],
            ),
        ),
        (
            "raw_source_label", "source_key", "feature", "collecting_method_version",
            "relationship", "value", "count",
        ),
    )


def _write_duplicate_calibration(output_dir: Path, state: ProfilerState) -> None:
    rows = []
    for key, stats in state.duplicate_calibration.items():
        raw_source, source_key, feature, version, level, candidate = key
        rows.append({
            "raw_source_label": raw_source,
            "source_key": source_key,
            "feature": feature,
            "collecting_method_version": version,
            "candidate_level": level,
            "candidate_definition": candidate,
            "observations": stats.observations,
            "unique_identity_keys": stats.unique_identity_keys,
            "distinct_content_fingerprints": stats.distinct_content_fingerprints,
            "repeated_groups": stats.repeated_groups,
            "repeated_occurrences": stats.repeated_occurrences,
            "exact_repeat_occurrences": stats.exact_repeat_occurrences,
            "changed_content_occurrences": stats.changed_content_occurrences,
            "within_month_repeat_occurrences": stats.within_month_repeat_occurrences,
            "cross_month_repeat_occurrences": stats.cross_month_repeat_occurrences,
            "missing_identity_observations": stats.missing_identity_observations,
        })
    write_csv(
        output_dir / "duplicate_revision_calibration.csv",
        sorted(
            rows,
            key=lambda row: (
                row["source_key"], row["feature"], row["collecting_method_version"],
                row["candidate_level"], row["candidate_definition"],
            ),
        ),
        (
            "raw_source_label", "source_key", "feature", "collecting_method_version",
            "candidate_level", "candidate_definition", "observations",
            "unique_identity_keys", "distinct_content_fingerprints", "repeated_groups",
            "repeated_occurrences", "exact_repeat_occurrences",
            "changed_content_occurrences", "within_month_repeat_occurrences",
            "cross_month_repeat_occurrences", "missing_identity_observations",
        ),
    )

    examples = []
    for code, bucket in state.diagnostics.examples.items():
        if not code.startswith("duplicate_calibration_"):
            continue
        for rank, context in bucket.items():
            examples.append({
                "diagnostic_code": code,
                "stable_rank": rank,
                "context_json": canonical_json(context),
            })
    write_csv(
        output_dir / "duplicate_revision_examples.csv",
        sorted(examples, key=lambda row: (row["diagnostic_code"], row["stable_rank"])),
        ("diagnostic_code", "stable_rank", "context_json"),
    )


def _candidate_activity_file(label: str) -> tuple[str, str]:
    sleep_labels = {"sleep", "sleep.light", "sleep.deep", "sleep.rem", "sleep.awake"}
    if label in sleep_labels or label.startswith("sleep."):
        return "Sleep.csv", "sleep_state"
    special = {
        "in_vehicle": "InVehicle.csv",
        "strength_training": "StrengthTraining.csv",
        "stair_climbing": "StairClimbing.csv",
        "unknown": "UnknownActivity.csv",
        "other": "OtherActivity.csv",
    }
    if label in special:
        return special[label], "one_label_per_feature"
    words = re_split_label(label)
    name = "".join(word[:1].upper() + word[1:] for word in words if word) or "UnnamedActivity"
    return f"{name}.csv", "one_label_per_feature"


def re_split_label(label: str) -> list[str]:
    import re
    return re.split(r"[^A-Za-z0-9]+", label)


def _write_activity(output_dir: Path, state: ProfilerState) -> None:
    rows = []
    schema_rows = []
    for label, stats in state.activity.items():
        duration = stats.duration.summary()
        candidate_file, candidate_rule = _candidate_activity_file(label)
        rows.append({
            "activity_name": label,
            "item_count": stats.count,
            "participant_count": len(stats.participants),
            "source_file_count": len(stats.files),
            "field_presence_json": canonical_json(dict(sorted(stats.field_presence.items()))),
            "schema_count": len(stats.schemas),
            "candidate_feature_file": candidate_file,
            "candidate_partition_rule": candidate_rule,
            **{f"duration_{key}": value for key, value in duration.items()},
        })
        for signature, count in stats.schemas.items():
            schema_rows.append({"activity_name": label, "item_schema": signature, "item_count": count})
    fields = (
        "activity_name", "item_count", "participant_count", "source_file_count", "field_presence_json",
        "schema_count", "candidate_feature_file", "candidate_partition_rule", "duration_numeric_count",
        "duration_finite_count", "duration_nonfinite_count", "duration_numeric_min", "duration_numeric_p01",
        "duration_numeric_p25", "duration_numeric_median", "duration_numeric_p75", "duration_numeric_p99",
        "duration_numeric_max", "duration_numeric_mean", "duration_numeric_zero_count",
        "duration_numeric_negative_count", "duration_quantile_sample_size", "duration_quantile_method",
    )
    write_csv(output_dir / "activity_label_inventory.csv", sorted(rows, key=lambda r: (-int(r["item_count"]), r["activity_name"])), fields)
    write_csv(output_dir / "activity_label_schemas.csv", sorted(schema_rows, key=lambda r: (r["activity_name"], -int(r["item_count"]), r["item_schema"])), ("activity_name", "item_schema", "item_count"))
    participant_rows = [
        {"participant_id": participant, "activity_name": label, "item_count": count}
        for (participant, label), count in state.participant_activity.items()
    ]
    write_csv(output_dir / "participant_activity_labels.csv", sorted(participant_rows, key=lambda r: (r["participant_id"], r["activity_name"])), ("participant_id", "activity_name", "item_count"))

    measurement_rows = []
    context_rows = []
    for (label, version), detailed in state.activity_submeasurements.items():
        candidate_file, candidate_rule = _candidate_activity_file(label)
        for measurement in ("duration_seconds", "calories", "distance", "quantity"):
            numeric = detailed.numeric_fields.get(measurement)
            summary = numeric.summary() if numeric is not None else _empty_numeric_summary()
            present_count = detailed.field_presence.get(measurement, 0)
            if measurement == "duration_seconds":
                present_count = summary["numeric_count"]
            measurement_rows.append({
                "activity_name": label,
                "collecting_method_version": version,
                "candidate_feature_file": candidate_file,
                "candidate_partition_rule": candidate_rule,
                "measurement": measurement,
                "item_count": detailed.item_count,
                "participant_count": len(detailed.participants),
                "source_file_count": len(detailed.files),
                "present_count": present_count,
                "missing_count": detailed.item_count - int(present_count),
                **summary,
            })
        dimensions = {
            "tracked": detailed.tracked_values,
            "device": detailed.device_values,
            "source_id": detailed.source_id_values,
            "source_name": detailed.source_name_values,
        }
        for dimension, counter in dimensions.items():
            for value, count in counter.items():
                context_rows.append({
                    "activity_name": label,
                    "collecting_method_version": version,
                    "candidate_feature_file": candidate_file,
                    "dimension": dimension,
                    "value": value,
                    "count_lower_bound": count,
                    "approximate": int(counter.approximate),
                })
    measurement_fields = (
        "activity_name", "collecting_method_version", "candidate_feature_file",
        "candidate_partition_rule", "measurement", "item_count", "participant_count",
        "source_file_count", "present_count", "missing_count", "numeric_count",
        "finite_count", "nonfinite_count", "numeric_min", "numeric_p01", "numeric_p25",
        "numeric_median", "numeric_p75", "numeric_p99", "numeric_max", "numeric_mean",
        "numeric_zero_count", "numeric_negative_count", "quantile_sample_size",
        "quantile_method",
    )
    write_csv(
        output_dir / "activity_submeasurement_profile.csv",
        sorted(
            measurement_rows,
            key=lambda row: (
                row["activity_name"], row["collecting_method_version"], row["measurement"]
            ),
        ),
        measurement_fields,
    )
    write_csv(
        output_dir / "activity_context_values.csv",
        sorted(
            context_rows,
            key=lambda row: (
                row["activity_name"], row["collecting_method_version"], row["dimension"],
                -int(row["count_lower_bound"]), row["value"],
            ),
        ),
        (
            "activity_name", "collecting_method_version", "candidate_feature_file",
            "dimension", "value", "count_lower_bound", "approximate",
        ),
    )


def _write_provenance(output_dir: Path, state: ProfilerState) -> None:
    aggregate: dict[tuple[str, str, str, str, str, str, str], dict[str, Any]] = {}
    for (
        raw_source, source_key, feature, version, dimension, path, value, participant,
    ), count in state.provenance.items():
        key = (raw_source, source_key, feature, version, dimension, path, value)
        row = aggregate.setdefault(key, {"event_count": 0, "participants": set()})
        row["event_count"] += count
        row["participants"].add(participant)
    rows = [
        {
            "raw_source_label": key[0],
            "source_key": key[1],
            "feature": key[2],
            "collecting_method_version": key[3],
            "provenance_dimension": key[4],
            "field_path": key[5],
            "value": key[6],
            "event_count": value["event_count"],
            "participant_count": len(value["participants"]),
        }
        for key, value in aggregate.items()
    ]
    write_csv(
        output_dir / "provenance_values.csv",
        sorted(
            rows,
            key=lambda row: (
                row["source_key"], row["feature"], row["collecting_method_version"],
                row["provenance_dimension"], row["field_path"], -int(row["event_count"]),
                row["value"],
            ),
        ),
        (
            "raw_source_label", "source_key", "feature", "collecting_method_version",
            "provenance_dimension", "field_path", "value", "event_count",
            "participant_count",
        ),
    )

    value_rows = []
    participant_source_keys: set[tuple[str, str]] = set()
    for (participant_id, source_key, dimension), values in state.participant_provenance_values.items():
        participant_source_keys.add((participant_id, source_key))
        for value in sorted(values):
            value_rows.append({
                "participant_id": participant_id,
                "source_key": source_key,
                "provenance_dimension": dimension,
                "value": value,
            })
    write_csv(
        output_dir / "participant_provenance_values.csv",
        sorted(
            value_rows,
            key=lambda row: (
                row["participant_id"], row["source_key"], row["provenance_dimension"],
                row["value"],
            ),
        ),
        ("participant_id", "source_key", "provenance_dimension", "value"),
    )

    dimensions = (
        "coarse_device",
        "detailed_device",
        "device_manufacturer",
        "device_model",
        "device_type",
        "device_uid",
        "app_package",
        "app_name",
        "added_by",
        "source_name",
        "source_id",
        "data_source_id",
        "origin_data_source_id",
        "tracked",
        "user_entered",
    )
    summaries = []
    for participant_id, source_key in sorted(participant_source_keys):
        values_by_dimension = {
            dimension: sorted(
                state.participant_provenance_values.get(
                    (participant_id, source_key, dimension), set()
                )
            )
            for dimension in dimensions
        }
        app_union = set(values_by_dimension["app_package"]) | set(values_by_dimension["app_name"]) | set(values_by_dimension["added_by"])
        stream_union = set(values_by_dimension["source_id"]) | set(values_by_dimension["data_source_id"]) | set(values_by_dimension["origin_data_source_id"])
        detailed = values_by_dimension["detailed_device"]
        coarse = values_by_dimension["coarse_device"]
        summaries.append({
            "participant_id": participant_id,
            "source_key": source_key,
            "coarse_device_count": len(coarse),
            "detailed_device_count": len(detailed),
            "device_manufacturer_count": len(values_by_dimension["device_manufacturer"]),
            "device_model_count": len(values_by_dimension["device_model"]),
            "device_type_count": len(values_by_dimension["device_type"]),
            "device_uid_count": len(values_by_dimension["device_uid"]),
            "app_package_count": len(values_by_dimension["app_package"]),
            "app_name_count": len(values_by_dimension["app_name"]),
            "added_by_count": len(values_by_dimension["added_by"]),
            "application_identity_count": len(app_union),
            "source_name_count": len(values_by_dimension["source_name"]),
            "source_id_count": len(values_by_dimension["source_id"]),
            "data_source_id_count": len(values_by_dimension["data_source_id"]),
            "origin_data_source_id_count": len(values_by_dimension["origin_data_source_id"]),
            "stream_identity_count": len(stream_union),
            "user_entered_value_count": len(values_by_dimension["user_entered"]),
            "apparent_multi_device": int(len(detailed) > 1 or (not detailed and len(coarse) > 1)),
            "apparent_multi_application": int(len(app_union) > 1),
            "coarse_devices_json": canonical_json(coarse),
            "detailed_devices_json": canonical_json(detailed),
            "device_manufacturers_json": canonical_json(values_by_dimension["device_manufacturer"]),
            "device_models_json": canonical_json(values_by_dimension["device_model"]),
            "device_types_json": canonical_json(values_by_dimension["device_type"]),
            "device_uids_json": canonical_json(values_by_dimension["device_uid"]),
            "app_packages_json": canonical_json(values_by_dimension["app_package"]),
            "app_names_json": canonical_json(values_by_dimension["app_name"]),
            "added_by_json": canonical_json(values_by_dimension["added_by"]),
            "source_names_json": canonical_json(values_by_dimension["source_name"]),
            "source_ids_json": canonical_json(values_by_dimension["source_id"]),
            "data_source_ids_json": canonical_json(values_by_dimension["data_source_id"]),
            "origin_data_source_ids_json": canonical_json(values_by_dimension["origin_data_source_id"]),
            "user_entered_values_json": canonical_json(values_by_dimension["user_entered"]),
        })
    fields = (
        "participant_id", "source_key", "coarse_device_count", "detailed_device_count",
        "device_manufacturer_count", "device_model_count", "device_type_count",
        "device_uid_count", "app_package_count", "app_name_count", "added_by_count",
        "application_identity_count", "source_name_count", "source_id_count",
        "data_source_id_count", "origin_data_source_id_count", "stream_identity_count",
        "user_entered_value_count", "apparent_multi_device", "apparent_multi_application",
        "coarse_devices_json", "detailed_devices_json", "device_manufacturers_json",
        "device_models_json", "device_types_json", "device_uids_json", "app_packages_json",
        "app_names_json", "added_by_json", "source_names_json", "source_ids_json",
        "data_source_ids_json", "origin_data_source_ids_json", "user_entered_values_json",
    )
    write_csv(output_dir / "participant_provenance_summary.csv", summaries, fields)



def _write_steps(output_dir: Path, state: ProfilerState) -> None:
    rows = []
    bucket_rows = []
    for (raw_source, version), stats in state.step_alternatives.items():
        difference = stats.difference_merge_minus_estimated.summary()
        absolute = stats.absolute_difference.summary()
        summary_raw = stats.summary_minus_raw.summary()
        rows.append({
            "raw_source_label": raw_source,
            "collecting_method_version": version,
            "outer_rows": stats.outer_rows,
            "rows_with_multiple_alternatives": stats.rows_with_multiple_alternatives,
            "rows_with_estimated_steps": stats.rows_with_estimated,
            "rows_with_merge_step_deltas": stats.rows_with_merge,
            "rows_with_both": stats.rows_with_both,
            "rows_equal": stats.rows_equal,
            "rows_different": stats.rows_different,
            "alternative_names_json": canonical_json(dict(sorted(stats.alternative_names.items()))),
            "alternatives_per_outer_row_json": canonical_json(dict(sorted(stats.alternatives_per_outer_row.items()))),
            "difference_buckets_json": canonical_json(dict(sorted(stats.difference_buckets.items()))),
            "summary_raw_pairs": stats.summary_raw_pairs,
            "summary_raw_equal": stats.summary_raw_equal,
            "summary_raw_different": stats.summary_raw_different,
            "summary_raw_difference_buckets_json": canonical_json(dict(sorted(stats.summary_raw_difference_buckets.items()))),
            **{f"difference_{key}": value for key, value in difference.items()},
            **{f"absolute_difference_{key}": value for key, value in absolute.items()},
            **{f"summary_minus_raw_{key}": value for key, value in summary_raw.items()},
        })
        for comparison, buckets in (
            ("merge_minus_estimated", stats.difference_buckets),
            ("summary_minus_raw", stats.summary_raw_difference_buckets),
        ):
            for bucket, count in buckets.items():
                bucket_rows.append({
                    "raw_source_label": raw_source,
                    "collecting_method_version": version,
                    "comparison": comparison,
                    "difference_bucket": bucket,
                    "count": count,
                })
    base_fields = (
        "raw_source_label", "collecting_method_version", "outer_rows",
        "rows_with_multiple_alternatives", "rows_with_estimated_steps",
        "rows_with_merge_step_deltas", "rows_with_both", "rows_equal",
        "rows_different", "alternative_names_json", "alternatives_per_outer_row_json",
        "difference_buckets_json", "summary_raw_pairs", "summary_raw_equal",
        "summary_raw_different", "summary_raw_difference_buckets_json",
    )
    numeric_suffixes = (
        "numeric_count", "finite_count", "nonfinite_count", "numeric_min", "numeric_p01",
        "numeric_p25", "numeric_median", "numeric_p75", "numeric_p99", "numeric_max",
        "numeric_mean", "numeric_zero_count", "numeric_negative_count", "quantile_sample_size",
        "quantile_method",
    )
    fields = (
        *base_fields,
        *(f"difference_{suffix}" for suffix in numeric_suffixes),
        *(f"absolute_difference_{suffix}" for suffix in numeric_suffixes),
        *(f"summary_minus_raw_{suffix}" for suffix in numeric_suffixes),
    )
    write_csv(
        output_dir / "stepcount_alternative_summary.csv",
        sorted(rows, key=lambda row: (row["raw_source_label"], row["collecting_method_version"])),
        fields,
    )
    write_csv(
        output_dir / "stepcount_difference_buckets.csv",
        sorted(
            bucket_rows,
            key=lambda row: (
                row["raw_source_label"], row["collecting_method_version"],
                row["comparison"], row["difference_bucket"],
            ),
        ),
        (
            "raw_source_label", "collecting_method_version", "comparison",
            "difference_bucket", "count",
        ),
    )


def _write_focused_anomalies(output_dir: Path, state: ProfilerState) -> None:
    rows = []
    for key, stats in state.focused_distributions.items():
        raw_source, source_key, feature, version, distribution, bucket = key
        rows.append({
            "raw_source_label": raw_source,
            "source_key": source_key,
            "feature": feature,
            "collecting_method_version": version,
            "distribution": distribution,
            "bucket": bucket,
            "count": stats.count,
            "participant_count": len(stats.participants),
            **stats.values.summary(),
        })
    numeric_fields = (
        "numeric_count", "finite_count", "nonfinite_count", "numeric_min", "numeric_p01",
        "numeric_p25", "numeric_median", "numeric_p75", "numeric_p99", "numeric_max",
        "numeric_mean", "numeric_zero_count", "numeric_negative_count", "quantile_sample_size",
        "quantile_method",
    )
    fields = (
        "raw_source_label", "source_key", "feature", "collecting_method_version",
        "distribution", "bucket", "count", "participant_count", *numeric_fields,
    )
    ordered = sorted(
        rows,
        key=lambda row: (
            row["source_key"], row["feature"], row["collecting_method_version"],
            row["distribution"], row["bucket"],
        ),
    )
    write_csv(output_dir / "focused_anomaly_distributions.csv", ordered, fields)
    write_csv(
        output_dir / "calories_negative_distribution.csv",
        [row for row in ordered if row["distribution"] == "negative_calorie_value"],
        fields,
    )
    write_csv(
        output_dir / "weight_interval_duration_distribution.csv",
        [row for row in ordered if row["distribution"] == "weight_interval_duration_seconds"],
        fields,
    )

    weight_rows = []
    for (version, added_by), stats in state.weight_calibration.items():
        weight_rows.append({
            "collecting_method_version": version,
            "added_by": added_by,
            "item_count": stats.item_count,
            "participant_count": len(stats.participants),
            "malformed_count": stats.malformed_count,
            "point_count": stats.point_count,
            "nonpoint_count": stats.nonpoint_count,
            "invalid_time_count": stats.invalid_time_count,
            **{f"value_{key}": value for key, value in stats.value.summary().items()},
            **{f"duration_{key}": value for key, value in stats.duration.summary().items()},
        })
    weight_fields = (
        "collecting_method_version", "added_by", "item_count", "participant_count",
        "malformed_count", "point_count", "nonpoint_count", "invalid_time_count",
        *(f"value_{suffix}" for suffix in numeric_fields),
        *(f"duration_{suffix}" for suffix in numeric_fields),
    )
    write_csv(
        output_dir / "weight_calibration.csv",
        sorted(weight_rows, key=lambda row: (row["collecting_method_version"], row["added_by"])),
        weight_fields,
    )

    malformed_rows = []
    for key, count in state.malformed_item_counts.items():
        raw_source, feature, version, reason = key
        malformed_rows.append({
            "raw_source_label": raw_source,
            "feature": feature,
            "collecting_method_version": version,
            "reason": reason,
            "item_count": count,
            "participant_count": len(state.malformed_item_participants[key]),
        })
    write_csv(
        output_dir / "malformed_item_summary.csv",
        sorted(
            malformed_rows,
            key=lambda row: (
                row["raw_source_label"], row["feature"], row["collecting_method_version"],
                row["reason"],
            ),
        ),
        (
            "raw_source_label", "feature", "collecting_method_version", "reason",
            "item_count", "participant_count",
        ),
    )



def _write_diagnostics(output_dir: Path, state: ProfilerState) -> None:
    summary_rows = [
        {
            "diagnostic_code": code,
            "count": count,
            "example_count": len(state.diagnostics.examples.get(code, ())),
        }
        for code, count in state.diagnostics.counts.items()
    ]
    write_csv(
        output_dir / "diagnostic_summary.csv",
        sorted(summary_rows, key=lambda row: (-int(row["count"]), row["diagnostic_code"])),
        ("diagnostic_code", "count", "example_count"),
    )
    example_rows = []
    for code, bucket in state.diagnostics.examples.items():
        for index, (rank, example) in enumerate(bucket.items(), start=1):
            example_rows.append({
                "diagnostic_code": code,
                "example_index": index,
                "stable_rank": rank,
                "context_json": canonical_json(example),
            })
    write_csv(
        output_dir / "diagnostic_examples.csv",
        sorted(
            example_rows,
            key=lambda row: (row["diagnostic_code"], row["stable_rank"]),
        ),
        ("diagnostic_code", "example_index", "stable_rank", "context_json"),
    )



def _write_input_issues(output_dir: Path, state: ProfilerState) -> None:
    write_csv(
        output_dir / "outer_schema_issues.csv",
        sorted(
            state.outer_schema_issues,
            key=lambda row: (
                str(row.get("participant_id", "")),
                str(row.get("canonical_month", "")),
                str(row.get("source_file", "")),
            ),
        ),
        ("participant_id", "canonical_month", "source_file", "missing_columns", "extra_columns"),
    )
    write_csv(
        output_dir / "invalid_or_duplicate_month_files.csv",
        sorted(
            state.invalid_files,
            key=lambda row: (
                str(row.get("participant_id", "")), str(row.get("kind", "")), str(row.get("path", "")),
            ),
        ),
        ("participant_id", "kind", "path", "details"),
    )


def _write_execution(output_dir: Path, state: ProfilerState) -> None:
    participant_ids = sorted(
        set(state.participant_runtime_seconds)
        | set(state.participant_peak_rss_bytes)
        | set(state.completed_participants)
        | set(state.participant_failures)
    )
    telemetry = []
    for participant_id in participant_ids:
        failure = state.participant_failures.get(participant_id)
        telemetry.append({
            "participant_id": participant_id,
            "status": "failed" if failure else "complete",
            "runtime_seconds": state.participant_runtime_seconds.get(participant_id),
            "peak_rss_bytes": state.participant_peak_rss_bytes.get(participant_id),
            "error": failure.get("error") if failure else None,
        })
    write_csv(
        output_dir / "participant_execution.csv",
        telemetry,
        ("participant_id", "status", "runtime_seconds", "peak_rss_bytes", "error"),
    )

    failures = [
        {
            "participant_id": participant_id,
            "error": value.get("error"),
            "runtime_seconds": value.get("runtime_seconds"),
            "peak_rss_bytes": value.get("peak_rss_bytes"),
            "traceback": value.get("traceback"),
        }
        for participant_id, value in sorted(state.participant_failures.items())
    ]
    write_csv(
        output_dir / "participant_failures.csv",
        failures,
        ("participant_id", "error", "runtime_seconds", "peak_rss_bytes", "traceback"),
    )


def _markdown_report(summary: Mapping[str, Any], state: ProfilerState) -> str:
    memberships = "\n".join(f"- `{key}`: {value:,}" for key, value in summary["source_membership"].items()) or "- None"
    sources = "\n".join(f"- `{key}`: {value:,} outer rows" for key, value in summary["source_labels"].items()) or "- None"
    activities = sorted(state.activity.values(), key=lambda item: (-item.count, item.label))
    activity_lines = "\n".join(
        f"- `{item.label}`: {item.count:,} items across {len(item.participants):,} participants"
        for item in activities[:30]
    ) or "- No decoded `Activities` labels"
    diagnostics = "\n".join(
        f"- `{code}`: {count:,}" for code, count in state.diagnostics.counts.most_common(30)
    ) or "- No diagnostics"
    return f"""# Mixed-source raw wearable census and Phase 0 calibration

## Headline inventory

- Participant directories discovered: **{summary['participant_directories']:,}**
- Data-bearing participants: **{summary['data_bearing_participants']:,}**
- Empty export directories: **{summary['empty_export_directories']:,}**
- Directories with month files but no outer rows: **{summary['participant_directories_with_month_files_but_no_rows']:,}**
- Monthly files: **{summary['monthly_files']:,}**
- Outer rows: **{summary['outer_rows']:,}**
- Participants with more than one outer `data_source`: **{summary['mixed_source_participants']:,}**
- Deep-decoded outer rows: **{summary['decoded_outer_rows']:,}**
- Deep-decoded payload items: **{summary['payload_items']:,}**
- Payload decode failures: **{summary['payload_decode_failures']:,}**
- Participant worker failures: **{summary['participants_failed']:,}**

## Execution

- Mode: **{summary['execution_mode']}**
- Workers requested/effective: **{summary['workers_requested']:,} / {summary['workers_effective']:,}**
- Maximum in flight: **{summary['max_in_flight']:,}**
- Wall-clock elapsed: **{summary['elapsed_seconds']:.1f} seconds**
- Sum of participant worker runtimes: **{summary['participant_runtime_seconds_sum']:.1f} seconds**
- Maximum observed worker high-water RSS: **{summary['participant_peak_rss_bytes_max'] or 0:,} bytes**

## Raw source labels

{sources}

## Participant source membership

{memberships}

## Decoded activity labels

{activity_lines}

The `activity_label_inventory.csv` report includes a **candidate**, not final, partition into files such as
`Walking.csv`, `Biking.csv`, and consolidated `Sleep.csv`. The raw label remains the authority until the
processing registry is calibrated against the complete cohort.

## Phase 0 calibration reports

- `participant_feature_version_months.csv`, `participant_feature_versions.csv`, and
  `participant_feature_version_transitions.csv` expose sequential, overlapping, and reverting exporter versions.
- `duplicate_revision_calibration.csv` and `duplicate_revision_examples.csv` compare alternative outer-row and
  payload-item identity definitions without dropping or reconciling any observation.
- `activity_submeasurement_profile.csv` and `activity_context_values.csv` characterize measurements and provenance
  attached to each routed activity label.
- `participant_provenance_values.csv` and `participant_provenance_summary.csv` keep device, application, and source
  stream identities separate.
- `temporal_alignment_counts.csv` records exact date, weekday-label, and timezone-offset relationships.
- `focused_anomaly_distributions.csv`, `calories_negative_distribution.csv`,
  `weight_interval_duration_distribution.csv`, `weight_calibration.csv`, and `malformed_item_summary.csv` provide
  evidence for feature-specific native and curation policies.
- `representative_payloads.json` and `diagnostic_examples.csv` use stable cryptographic ranking rather than
  first-seen process order.

## Diagnostics

{diagnostics}

## Interpretation boundary

This command inventories and profiles the raw export. It does not clean, reconcile duplicates, infer units,
select competing step aggregates, or alter Apple HealthKit processing. All source-feature-version keys remain
source-qualified.
"""
