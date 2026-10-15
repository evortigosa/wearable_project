"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import ast
import csv
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from typing import Any
import pytest
from wearable_project.exceptions import DuplicateMonthError, ParticipantProcessingError
from wearable_project.processing.google_fitness.parser import (
    PARSER_VERSION, ParseResult, SourceFile, canonical_month_from_name, decode_payload, discover_month_files,
    merge_parse_results, parse_month_file, parse_participant_directory,
)
from wearable_project.processing.google_fitness.registry import (
    EXPECTED_REGISTRY_FINGERPRINT, OUTER_SCHEMA_COLUMNS, STEP_COUNT_DATA_SOURCE_ID, STEP_COUNT_DATA_TYPE_NAME,
    UNREGISTERED_ACTIVITY_POLICY, get_feature_spec, registry_fingerprint,
)


FIXTURE_PATH = (Path(__file__).parent / "fixtures" / "google_fitness_phase0_representative_items.json")


def _outer_row(
    *, outer_id: str, participant_id: str = "P001", data_source: str = "GoogleFitness", feature: str,
    version: str, payload: Any, outer_datetime: str = "2024-01-01 00:00:00",
) -> dict[str, str]:
    return {
        "id": outer_id,
        "participant_id": participant_id,
        "data_source": data_source,
        "name": feature,
        "datetime": outer_datetime,
        "data": repr(payload),
        "collecting_method_version": version,
        "created_at": "2024-01-02 00:00:00",
        "updated_at": "2024-01-03 00:00:00",
    }


def _write_month(
    root: Path, rows: list[dict[str, str]], *, participant_id: str = "P001", filename: str = "2024-1.csv",
) -> tuple[Path, SourceFile]:
    participant = root / participant_id
    participant.mkdir(parents=True, exist_ok=True)
    path = participant / filename
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTER_SCHEMA_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    sources = discover_month_files(participant)
    source = next(item for item in sources if item.path.name == filename)
    return participant, source


def _step_alternative(
    *, source: str, summary_date: str, value: int, start: int, end: int, raw_value: int | None = None,
    origin: str = "raw:com.google.step_count.delta:test:stream",
) -> dict[str, Any]:
    return {
        "source": source,
        "steps": [{"date": summary_date, "value": value}],
        "raw_steps": [
            {
                "steps": value if raw_value is None else raw_value,
                "start_date": start,
                "end_date": end,
                "data_source_id": STEP_COUNT_DATA_SOURCE_ID,
                "data_type_name": STEP_COUNT_DATA_TYPE_NAME,
                "origin_data_source_id": origin,
                "app_name": "Example App",
                "app_package_name": "org.example.app",
                "device_manufacturer": "Example",
                "device_model": "Model 1",
                "device_type": "watch",
                "device_uid": "abc123",
            }
        ],
    }


def _source_payload(feature: str, version: str, raw_item: Any) -> Any:
    if feature == "Weight" and version == "1.0":
        return raw_item
    if feature == "StepCount" and version == "1.0":
        return raw_item
    return [raw_item]


def _parse_rows(
    tmp_path: Path, rows: list[dict[str, str]], *, participant_id: str = "P001",
    row_error_policy: str = "fail-participant",
) -> ParseResult:
    _, source = _write_month(tmp_path, rows, participant_id=participant_id,)
    result = parse_month_file(source, participant_id, row_error_policy=row_error_policy,)
    result.assert_accounting_closed()
    return result


def test_wp2_parser_version_and_wp1_registry_fingerprint_are_stable() -> None:
    assert PARSER_VERSION == "google-fitness-parser-wp2-0.1.0"
    assert registry_fingerprint() == EXPECTED_REGISTRY_FINGERPRINT


def test_decode_payload_supports_export_layers_and_expected_scalar() -> None:
    payload = [{"value": 1, "tracked": True}]
    encoded = json.dumps(repr(payload))
    assert decode_payload(encoded) == payload
    assert decode_payload(repr({"value": 75})) == {"value": 75}
    assert decode_payload("59") == 59


def test_month_discovery_canonicalizes_names_and_rejects_duplicates(tmp_path: Path,) -> None:
    participant = tmp_path / "P001"
    participant.mkdir()
    (participant / "2024-1.csv").write_text("x\n", encoding="utf-8")
    (participant / "README.txt").write_text("ignored\n", encoding="utf-8")
    sources = discover_month_files(participant)
    assert [item.canonical_month for item in sources] == ["2024-01"]
    assert canonical_month_from_name("2024-01.csv") == "2024-01"
    assert canonical_month_from_name("2024-13.csv") is None

    (participant / "2024-01.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(DuplicateMonthError):
        discover_month_files(participant)


def test_mixed_source_file_filters_apple_before_payload_decoding_and_keeps_lineage(tmp_path: Path,) -> None:
    rows = [
        _outer_row(
            outer_id="A1",
            data_source="AppleHealthkit",
            feature="Electrocardiogram",
            version="1.0",
            payload="not even a valid payload",
        ),
        _outer_row(
            outer_id="G1",
            data_source="Google Fitness",
            feature="Calories",
            version="1.0",
            payload=[
                {
                    "start_date": "2024-01-01T00:00:00.000Z",
                    "end_date": "2024-01-01T23:59:59.999Z",
                    "calorie": -2.5,
                    "day": "Mon",
                }
            ],
        ),
    ]
    result = _parse_rows(tmp_path, rows)

    assert result.outer_rows_seen == 2
    assert result.google_rows_seen == 1
    assert result.irrelevant_rows_seen == 1
    assert result.payloads_decoded == 1
    assert result.payload_decode_failures == 0
    assert result.valid_candidates == 1
    assert result.records_by_feature["Calories"][0]["value"] == -2.5

    occurrence = result.source_occurrences[0]
    assert occurrence.raw_source_label == "Google Fitness"
    assert occurrence.source_feature == "Calories"
    assert occurrence.outer_id == "G1"
    assert occurrence.source_file == "2024-1.csv"
    assert occurrence.canonical_month == "2024-01"
    assert occurrence.raw_payload_sha256
    assert occurrence.decoded_payload_sha256
    assert occurrence.payload_item_count == 1


def test_all_47_phase0_representative_schema_families_are_accounted(tmp_path: Path,) -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for index, entry in enumerate(fixture):
        rows.append(
            _outer_row(
                outer_id=f"R{index:03d}",
                feature=entry["feature"],
                version=entry["collecting_method_version"],
                payload=_source_payload(
                    entry["feature"],
                    entry["collecting_method_version"],
                    entry["raw_item"],
                ),
            )
        )

    result = _parse_rows(tmp_path, rows)
    assert result.outer_rows_seen == 47
    assert result.google_rows_seen == 47
    assert result.payloads_decoded == 47
    assert result.candidate_items_seen == 47
    assert result.valid_candidates == 46
    assert result.malformed_candidates == 1
    assert result.activity_items_seen == 26
    assert result.activity_items_routed == 26
    assert result.step_count_v2_items_seen == 6
    assert result.step_count_v2_alternatives_emitted == 6
    assert result.malformed_reason_counts == {"missing_or_nonnumeric_value": 1, "non_dict_list_item": 1,}
    assert "Activities" not in result.records_by_feature

    for feature, records in result.records_by_feature.items():
        spec = get_feature_spec(feature)
        for record in records:
            assert tuple(record) == spec.column_names


def test_activity_routing_is_exactly_once_and_sleep_states_are_normalized(tmp_path: Path,) -> None:
    activities = [
        {
            "activity_name": "walking",
            "start": 1704067200000,
            "end": 1704067260000,
            "calories": 4.5,
            "distance": 80,
            "quantity": 100,
            "tracked": True,
            "device": "Android",
            "source_name": "Android",
            "source_id": "com.google.android.gms",
        },
        {
            "activity_name": "sleep.deep",
            "start": 1704067260000,
            "end": 1704070860000,
            "calories": 50,
            "quantity": 60,
            "tracked": False,
            "device": "Android Wear",
            "source_name": "Android Wear",
            "source_id": None,
        },
    ]
    result = _parse_rows(
        tmp_path,
        [
            _outer_row(
                outer_id="ACT1",
                feature="Activities",
                version="1.0",
                payload=activities,
            )
        ],
    )

    assert result.candidate_items_seen == 2
    assert result.activity_items_seen == 2
    assert result.activity_items_routed == 2
    assert result.valid_candidates == 2
    assert set(result.records_by_feature) == {"Walking", "Sleep"}
    assert result.records_by_feature["Sleep"][0]["value"] == "DEEP"
    assert "raw_activity_name" not in result.records_by_feature["Sleep"][0]
    assert "duration" not in result.records_by_feature["Walking"][0]
    assert "Activities" not in result.records_by_feature


def test_unregistered_activity_uses_lossless_fallback_and_warning(tmp_path: Path,) -> None:
    item = {
        "activity_name": "future.activity",
        "start": 1704067200000,
        "end": 1704067260000,
        "calories": 2,
        "distance": 3,
        "quantity": 4,
        "tracked": True,
        "device": "Android",
        "source_name": "Android",
        "source_id": "org.example.future",
    }
    result = _parse_rows(
        tmp_path,
        [
            _outer_row(
                outer_id="ACT2",
                feature="Activities",
                version="1.0",
                payload=[item],
            )
        ],
    )

    assert result.unknown_activity_labels == {"future.activity"}
    assert result.unregistered_activity_items_routed == 1
    assert result.registered_activity_items_routed == 0
    record = result.records_by_feature[UNREGISTERED_ACTIVITY_POLICY.native_feature][0]
    assert tuple(record) == UNREGISTERED_ACTIVITY_POLICY.column_names
    assert record["raw_activity_name"] == "future.activity"
    assert record["distance"] == 3
    assert any(item.code == "unregistered-activity" for item in result.diagnostics)


def test_registered_activity_schema_drift_is_malformed_not_silently_dropped(tmp_path: Path,) -> None:
    item = {
        "activity_name": "rock_climbing",
        "start": 1704067200000,
        "end": 1704067260000,
        "calories": 2,
        "distance": 3,
        "quantity": 4,
        "tracked": True,
        "device": "Android",
        "source_name": "Android",
        "source_id": "org.example",
    }
    result = _parse_rows(
        tmp_path,
        [
            _outer_row(
                outer_id="ACT3",
                feature="Activities",
                version="1.0",
                payload=[item],
            )
        ],
    )

    assert result.valid_candidates == 0
    assert result.malformed_candidates == 1
    assert result.activity_items_malformed == 1
    assert result.malformed_reason_counts["distance:unregistered_measurement_for_activity"] == 1
    assert "RockClimbing" not in result.records_by_feature


def test_stepcount_v2_emits_one_row_per_alternative_without_preference(tmp_path: Path,) -> None:
    payload = [
        _step_alternative(
            source="com.google.android.gms:estimated_steps",
            summary_date="2024-01-01",
            value=1000,
            raw_value=999,
            start=1704067200000,
            end=1704153599999,
            origin="raw:estimated",
        ),
        _step_alternative(
            source="com.google.android.gms:merge_step_deltas",
            summary_date="2024-01-01",
            value=1200,
            raw_value=1200,
            start=1704067200000,
            end=1704153599999,
            origin="raw:merged",
        ),
    ]
    result = _parse_rows(
        tmp_path,
        [
            _outer_row(
                outer_id="S2",
                feature="StepCount",
                version="2.0",
                payload=payload,
            )
        ],
    )

    assert result.step_count_v2_items_seen == 2
    assert result.step_count_v2_alternatives_emitted == 2
    rows = result.records_by_feature["StepCount"]
    assert len(rows) == 2
    assert [row["aggregation_source"] for row in rows] == [
        "com.google.android.gms:estimated_steps",
        "com.google.android.gms:merge_step_deltas",
    ]
    assert [row["payload_index"] for row in rows] == [0, 1]
    assert rows[0]["value"] == 1000
    assert rows[0]["raw_value"] == 999
    assert rows[0]["summary_date"] == "2024-01-01"
    assert rows[0]["start_date"] == "2024-01-01T00:00:00.000000Z"
    assert rows[0]["origin_data_source_id"] == "raw:estimated"
    assert rows[0]["app_package_name"] == "org.example.app"


def test_stepcount_v1_equal_values_and_cross_version_rows_remain_distinct(tmp_path: Path,) -> None:
    rows = [
        _outer_row(
            outer_id="S10",
            feature="StepCount",
            version="1.0",
            payload=500,
            outer_datetime="2024-01-01 00:00:00",
        ),
        _outer_row(
            outer_id="S11",
            feature="StepCount",
            version="1.0",
            payload=500,
            outer_datetime="2024-01-02 00:00:00",
        ),
        _outer_row(
            outer_id="S20",
            feature="StepCount",
            version="2.0",
            payload=[
                _step_alternative(
                    source="com.google.android.gms:estimated_steps",
                    summary_date="2024-01-01",
                    value=500,
                    start=1704067200000,
                    end=1704153599999,
                )
            ],
        ),
    ]
    result = _parse_rows(tmp_path, rows)

    step_rows = result.records_by_feature["StepCount"]
    assert len(step_rows) == 3
    assert [row["collecting_method_version"] for row in step_rows] == ["1.0", "1.0", "2.0",]
    assert [row["outer_id"] for row in step_rows] == ["S10", "S11", "S20"]
    assert result.valid_candidates == 3


def test_negative_calories_and_nonpoint_weight_are_preserved(tmp_path: Path) -> None:
    rows = [
        _outer_row(
            outer_id="C1",
            feature="Calories",
            version="2.0",
            payload=[
                {
                    "start_date": "2024-01-01T00:00:00.000Z",
                    "end_date": "2024-01-01T23:59:59.999Z",
                    "calorie": -123.5,
                    "day": "Mon",
                }
            ],
        ),
        _outer_row(
            outer_id="W1",
            feature="Weight",
            version="1.0",
            payload={
                "start_date": "2024-01-01T10:00:00.000Z",
                "end_date": "2024-01-01T10:30:00.000Z",
                "value": 75,
                "day": "Mon",
                "added_by": "org.example.scale",
            },
        ),
    ]
    result = _parse_rows(tmp_path, rows)
    assert result.records_by_feature["Calories"][0]["value"] == -123.5
    weight = result.records_by_feature["Weight"][0]
    assert weight["start_date"] != weight["end_date"]
    assert "duration" not in weight


def test_four_style_weight_malformed_item_is_counted_once_with_two_reasons(tmp_path: Path,) -> None:
    result = _parse_rows(
        tmp_path,
        [
            _outer_row(
                outer_id="W2",
                feature="Weight",
                version="2.0",
                payload=[None],
            )
        ],
    )

    assert result.candidate_items_seen == 1
    assert result.valid_candidates == 0
    assert result.malformed_candidates == 1
    assert len(result.malformed_items) == 1
    assert result.malformed_items[0].reason_codes == ("missing_or_nonnumeric_value", "non_dict_list_item",)
    assert result.malformed_reason_counts == {"missing_or_nonnumeric_value": 1, "non_dict_list_item": 1,}
    assert "Weight" not in result.records_by_feature


def test_envelope_and_cardinality_drift_remain_exactly_counted(tmp_path: Path,) -> None:
    item = {
        "start_date": "2024-01-01T00:00:00.000Z",
        "end_date": "2024-01-01T23:59:59.999Z",
        "calorie": 1,
        "day": "Mon",
    }
    result = _parse_rows(
        tmp_path,
        [
            _outer_row(
                outer_id="C2",
                feature="Calories",
                version="1.0",
                payload=[item, item],
            )
        ],
    )

    assert result.payloads_decoded == 1
    assert result.payload_decode_failures == 0
    assert result.candidate_items_seen == 2
    assert result.valid_candidates == 0
    assert result.malformed_candidates == 2
    reason = "payload_cardinality:above_maximum_1_found_2"
    assert result.malformed_reason_counts[reason] == 2


def test_stepcount_registry_constant_mismatch_is_not_silently_omitted(tmp_path: Path,) -> None:
    alternative = _step_alternative(
        source="com.google.android.gms:estimated_steps",
        summary_date="2024-01-01",
        value=100,
        start=1704067200000,
        end=1704153599999,
    )
    alternative["raw_steps"][0]["data_source_id"] = "unexpected"
    result = _parse_rows(
        tmp_path,
        [
            _outer_row(
                outer_id="S3",
                feature="StepCount",
                version="2.0",
                payload=[alternative],
            )
        ],
    )
    assert result.valid_candidates == 0
    assert result.malformed_candidates == 1
    assert result.malformed_reason_counts["raw_steps.data_source_id:registry_constant_mismatch"] == 1


def test_unsupported_source_contract_strict_failure_and_skip_accounting(tmp_path: Path,) -> None:
    row = _outer_row(
        outer_id="U1",
        feature="Calories",
        version="9.9",
        payload=[{}],
    )
    _, source = _write_month(tmp_path, [row])
    partial = ParseResult()
    with pytest.raises(ParticipantProcessingError):
        parse_month_file(source, "P001", result=partial)
    assert partial.google_rows_seen == 1
    assert partial.unsupported_outer_rows == 1
    assert partial.unsupported_contracts == {"Calories|9.9"}

    skipped = parse_month_file(source, "P001", row_error_policy="skip-row")
    skipped.assert_accounting_closed()
    assert skipped.supported_outer_rows == 0
    assert skipped.unsupported_outer_rows == 1
    assert skipped.valid_candidates == 0
    assert len(skipped.source_occurrences) == 1


def test_payload_decode_failure_skip_policy_closes_outer_accounting(tmp_path: Path,) -> None:
    row = _outer_row(
        outer_id="BAD1",
        feature="Calories",
        version="1.0",
        payload=[],
    )
    row["data"] = "not a valid structured payload"
    _, source = _write_month(tmp_path, [row])

    result = parse_month_file(source, "P001", row_error_policy="skip-row")
    result.assert_accounting_closed()

    assert result.outer_rows_seen == 1
    assert result.google_rows_seen == 1
    assert result.supported_outer_rows == 1
    assert result.payloads_decoded == 0
    assert result.payload_decode_failures == 1
    assert result.candidate_items_seen == 0
    assert result.valid_candidates == 0
    assert result.malformed_candidates == 0
    assert len(result.source_occurrences) == 1
    assert result.source_occurrences[0].decoded_payload_sha256 is None
    assert result.source_occurrences[0].payload_item_count is None


def test_duplicate_outer_ids_are_reported_without_collapsing_candidates(tmp_path: Path,) -> None:
    rows = [
        _outer_row(
            outer_id="DUP1",
            feature="StepCount",
            version="1.0",
            payload=100,
            outer_datetime="2024-01-01 00:00:00",
        ),
        _outer_row(
            outer_id="DUP1",
            feature="StepCount",
            version="1.0",
            payload=200,
            outer_datetime="2024-01-02 00:00:00",
        ),
    ]
    result = _parse_rows(tmp_path, rows)

    assert result.duplicate_outer_ids == ("DUP1",)
    assert result.valid_candidates == 2
    assert [row["value"] for row in result.records_by_feature["StepCount"]] == [100, 200,]


def test_merge_parse_results_preserves_input_order_and_closes_accounting(tmp_path: Path,) -> None:
    participant = tmp_path / "P001"
    rows_january = [
        _outer_row(
            outer_id="J1",
            feature="StepCount",
            version="1.0",
            payload=1,
            outer_datetime="2024-01-01 00:00:00",
        )
    ]
    rows_february = [
        _outer_row(
            outer_id="F1",
            feature="StepCount",
            version="1.0",
            payload=2,
            outer_datetime="2024-02-01 00:00:00",
        )
    ]
    _, january = _write_month(tmp_path, rows_january, filename="2024-1.csv")
    _, february = _write_month(tmp_path, rows_february, filename="2024-2.csv")
    result_january = parse_month_file(january, "P001")
    result_february = parse_month_file(february, "P001")
    merged = merge_parse_results([result_january, result_february])
    merged.assert_accounting_closed()

    assert [row["outer_id"] for row in merged.records_by_feature["StepCount"]] == ["J1", "F1",]
    assert merged.outer_rows_seen == 2
    assert merged.candidate_items_seen == 2
    assert merged.valid_candidates == 2
    assert [item.outer_id for item in merged.source_occurrences] == ["J1", "F1"]
    assert participant.is_dir()


def test_parse_participant_directory_uses_canonical_month_order(tmp_path: Path) -> None:
    _write_month(
        tmp_path,
        [
            _outer_row(
                outer_id="F1",
                feature="StepCount",
                version="1.0",
                payload=2,
                outer_datetime="2024-02-01 00:00:00",
            )
        ],
        filename="2024-2.csv",
    )
    _write_month(
        tmp_path,
        [
            _outer_row(
                outer_id="J1",
                feature="StepCount",
                version="1.0",
                payload=1,
                outer_datetime="2024-01-01 00:00:00",
            )
        ],
        filename="2024-1.csv",
    )
    sources, result = parse_participant_directory(tmp_path / "P001")
    assert [source.canonical_month for source in sources] == ["2024-01", "2024-02"]
    assert [row["outer_id"] for row in result.records_by_feature["StepCount"]] == ["J1", "F1",]


def test_google_parser_has_no_apple_scientific_imports() -> None:
    parser_path = (
        Path(__file__).parents[1] / "wearable_project" / "processing" / "google_fitness" / "parser.py"
    )
    tree = ast.parse(parser_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not [name for name in imported if ".apple_healthkit" in name]


def test_parser_data_classes_are_immutable_where_they_are_lineage_records(tmp_path: Path,) -> None:
    _, source = _write_month(
        tmp_path,
        [
            _outer_row(
                outer_id="I1",
                feature="StepCount",
                version="1.0",
                payload=1,
            )
        ],
    )
    result = parse_month_file(source, "P001")
    with pytest.raises(FrozenInstanceError):
        setattr(result.source_occurrences[0], "outer_id", "changed")
