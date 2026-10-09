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
    _write_source_labels(output_dir, state)
    _write_features(output_dir, state)
    _write_schemas(output_dir, state)
    _write_representatives(output_dir, state)
    _write_fields(output_dir, state)
    _write_temporal(output_dir, state)
    _write_activity(output_dir, state)
    _write_provenance(output_dir, state)
    _write_steps(output_dir, state)
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
        "participant_id", "month_file_count", "first_month", "last_month", "missing_months_within_range",
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


def _write_provenance(output_dir: Path, state: ProfilerState) -> None:
    aggregate: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for (raw_source, feature, path, value, participant), count in state.provenance.items():
        key = (raw_source, feature, path, value)
        row = aggregate.setdefault(key, {"event_count": 0, "participants": set()})
        row["event_count"] += count
        row["participants"].add(participant)
    rows = [
        {
            "raw_source_label": key[0],
            "feature": key[1],
            "field_path": key[2],
            "value": key[3],
            "event_count": value["event_count"],
            "participant_count": len(value["participants"]),
        }
        for key, value in aggregate.items()
    ]
    write_csv(output_dir / "provenance_values.csv", sorted(rows, key=lambda r: (r["raw_source_label"], r["feature"], r["field_path"], -int(r["event_count"]), r["value"])), ("raw_source_label", "feature", "field_path", "value", "event_count", "participant_count"))

    participant_source_keys = sorted(
        set(state.participant_devices) | set(state.participant_device_details) | set(state.participant_apps) | set(state.participant_streams)
    )
    summaries = []
    for participant_id, source_key in participant_source_keys:
        coarse = sorted(state.participant_devices.get((participant_id, source_key), set()))
        details = sorted(state.participant_device_details.get((participant_id, source_key), set()))
        apps = sorted(state.participant_apps.get((participant_id, source_key), set()))
        streams = sorted(state.participant_streams.get((participant_id, source_key), set()))
        summaries.append({
            "participant_id": participant_id,
            "source_key": source_key,
            "coarse_device_count": len(coarse),
            "detailed_device_count": len(details),
            "application_or_source_count": len(apps),
            "stream_count": len(streams),
            "apparent_multi_device": int(len(details) > 1 or (not details and len(coarse) > 1)),
            "apparent_multi_application": int(len(apps) > 1),
            "coarse_devices_json": canonical_json(coarse),
            "detailed_devices_json": canonical_json(details),
            "applications_or_sources_json": canonical_json(apps),
            "streams_json": canonical_json(streams),
        })
    fields = (
        "participant_id", "source_key", "coarse_device_count", "detailed_device_count",
        "application_or_source_count", "stream_count", "apparent_multi_device", "apparent_multi_application",
        "coarse_devices_json", "detailed_devices_json", "applications_or_sources_json", "streams_json",
    )
    write_csv(output_dir / "participant_provenance_summary.csv", summaries, fields)


def _write_steps(output_dir: Path, state: ProfilerState) -> None:
    rows = []
    for (raw_source, version), stats in state.step_alternatives.items():
        difference = stats.difference_merge_minus_estimated.summary()
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
            **{f"difference_{key}": value for key, value in difference.items()},
        })
    fields = (
        "raw_source_label", "collecting_method_version", "outer_rows", "rows_with_multiple_alternatives",
        "rows_with_estimated_steps", "rows_with_merge_step_deltas", "rows_with_both", "rows_equal",
        "rows_different", "alternative_names_json", "difference_numeric_count", "difference_finite_count",
        "difference_nonfinite_count", "difference_numeric_min", "difference_numeric_p01", "difference_numeric_p25",
        "difference_numeric_median", "difference_numeric_p75", "difference_numeric_p99", "difference_numeric_max",
        "difference_numeric_mean", "difference_numeric_zero_count", "difference_numeric_negative_count",
        "difference_quantile_sample_size", "difference_quantile_method",
    )
    write_csv(output_dir / "stepcount_alternative_summary.csv", sorted(rows, key=lambda r: (r["raw_source_label"], r["collecting_method_version"])), fields)


def _write_diagnostics(output_dir: Path, state: ProfilerState) -> None:
    summary_rows = [
        {"diagnostic_code": code, "count": count, "example_count": len(state.diagnostics.examples.get(code, []))}
        for code, count in state.diagnostics.counts.items()
    ]
    write_csv(output_dir / "diagnostic_summary.csv", sorted(summary_rows, key=lambda r: (-int(r["count"]), r["diagnostic_code"])), ("diagnostic_code", "count", "example_count"))
    example_rows = []
    for code, examples in state.diagnostics.examples.items():
        for index, example in enumerate(examples, start=1):
            example_rows.append({"diagnostic_code": code, "example_index": index, "context_json": canonical_json(example)})
    write_csv(output_dir / "diagnostic_examples.csv", sorted(example_rows, key=lambda r: (r["diagnostic_code"], int(r["example_index"]))), ("diagnostic_code", "example_index", "context_json"))


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
                str(row.get("participant_id", "")),
                str(row.get("kind", "")),
                str(row.get("path", "")),
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
        telemetry, ("participant_id", "status", "runtime_seconds", "peak_rss_bytes", "error"),
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
        failures, ("participant_id", "error", "runtime_seconds", "peak_rss_bytes", "traceback"),
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
    return f"""# Mixed-source raw wearable census

## Headline inventory

- Participants: **{summary['participants']:,}**
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

## Diagnostics

{diagnostics}

## Interpretation boundary

This command inventories and profiles the raw export. It does not clean, reconcile duplicates, infer units,
select competing step aggregates, or alter Apple HealthKit processing. All source-feature-version keys remain
source-qualified.
"""
