"""
Wearable Data Processing and Modeling project
"""


from __future__ import annotations
import ast
import copy
import csv
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
from typing import Any
import pytest
from wearable_project.processing.google_fitness.cleaners import (
    CLEANER_VERSION, ReconciliationAction, ReconciliationStage, clean_feature, clean_parse_result,
)
from wearable_project.processing.google_fitness.models import ParseResult
from wearable_project.processing.google_fitness.parser import (
    discover_month_files, merge_parse_results, parse_month_file, parse_participant_directory,
)
from wearable_project.processing.google_fitness.registry import (
    OUTER_SCHEMA_COLUMNS, STEP_COUNT_DATA_SOURCE_ID, STEP_COUNT_DATA_TYPE_NAME, get_feature_spec,
)


def _outer_row(
    *, outer_id: str, feature: str, version: str, payload: Any, participant_id: str = "P001",
    outer_datetime: str = "2024-01-01 00:00:00", created_at: str = "2024-01-02 00:00:00",
    updated_at: str = "2024-01-03 00:00:00", data_source: str = "GoogleFitness",
) -> dict[str, str]:
    return {
        "id": outer_id,
        "participant_id": participant_id,
        "data_source": data_source,
        "name": feature,
        "datetime": outer_datetime,
        "data": repr(payload),
        "collecting_method_version": version,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _write_month(
    root: Path, rows: list[dict[str, str]], *, participant_id: str = "P001", filename: str = "2024-1.csv",
) -> Path:
    participant = root / participant_id
    participant.mkdir(parents=True, exist_ok=True)
    path = participant / filename
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTER_SCHEMA_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return participant


def _calorie_payload(value: float, *, day: str = "Mon") -> list[dict[str, Any]]:
    return [
        {
            "start_date": "2024-01-01T00:00:00.000Z",
            "end_date": "2024-01-01T23:59:59.999Z",
            "calorie": value,
            "day": day,
        }
    ]


def _distance_payload(value: float, *, day: str = "Mon") -> list[dict[str, Any]]:
    return [
        {
            "start_date": "2024-01-01T00:00:00.000Z",
            "end_date": "2024-01-01T23:59:59.999Z",
            "distance": value,
            "day": day,
        }
    ]


def _step_alternative(
    *, source: str, value: int = 100, summary_date: str = "2024-01-01", start: int = 1704067200000,
    end: int = 1704153599999,
) -> dict[str, Any]:
    return {
        "source": source,
        "steps": [{"date": summary_date, "value": value}],
        "raw_steps": [
            {
                "steps": value,
                "start_date": start,
                "end_date": end,
                "data_source_id": STEP_COUNT_DATA_SOURCE_ID,
                "data_type_name": STEP_COUNT_DATA_TYPE_NAME,
                "origin_data_source_id": "raw:test:stream",
                "app_name": "Example App",
                "app_package_name": "org.example.app",
                "device_manufacturer": "Example",
                "device_model": "Model 1",
                "device_type": "watch",
                "device_uid": "uid-1",
            }
        ],
    }


def _spawn_parse_and_clean(participant: str) -> tuple[str, dict[str, Any]]:
    _, parsed = parse_participant_directory(Path(participant))
    cleaned = clean_parse_result(parsed)
    return cleaned.output_fingerprint, cleaned.accounting_payload()


def _parse_participant(root: Path, rows_by_month: list[tuple[str, list[dict[str, str]]]]) -> ParseResult:
    participant: Path | None = None
    for filename, rows in rows_by_month:
        participant = _write_month(root, rows, filename=filename)
    assert participant is not None
    _, parsed = parse_participant_directory(participant)
    parsed.assert_accounting_closed()
    return parsed


def test_wp3_cleaner_version_is_stable() -> None:
    assert CLEANER_VERSION == "google-fitness-cleaner-wp3-0.1.0"


def test_candidate_lineage_is_parallel_and_distinguishes_payload_items(tmp_path: Path,) -> None:
    parsed = _parse_participant(
        tmp_path,
        [
            (
                "2024-1.csv",
                [
                    _outer_row(
                        outer_id="S20",
                        feature="StepCount",
                        version="2.0",
                        payload=[
                            _step_alternative(source="estimated_steps"),
                            _step_alternative(source="merge_step_deltas"),
                        ],
                    )
                ],
            )
        ],
    )
    records = parsed.records_by_feature["StepCount"]
    lineage = parsed.candidate_lineage_by_feature["StepCount"]
    assert len(records) == len(lineage) == 2
    assert lineage[0].source_occurrence_key == lineage[1].source_occurrence_key
    assert lineage[0].candidate_key != lineage[1].candidate_key
    assert [item.payload_index for item in lineage] == [0, 1]


def test_exact_repeated_source_occurrence_is_collapsed_and_fully_represented(tmp_path: Path,) -> None:
    row = _outer_row(
        outer_id="C1",
        feature="Calories",
        version="1.0",
        payload=_calorie_payload(10.0),
    )
    parsed = _parse_participant(tmp_path, [("2024-1.csv", [row]), ("2024-2.csv", [row])],)
    cleaned = clean_parse_result(parsed)
    feature = cleaned.features["Calories"]
    assert feature.input_candidates == 2
    assert feature.output_rows == 1
    assert feature.occurrence_duplicates_removed == 1
    assert feature.semantic_duplicates_removed == 0
    assert feature.revisions_resolved == 0
    assert feature.represented_occurrences == 2
    assert feature.audits[0].occurrence_count == 2
    assert feature.audits[0].duplicate_count == 1
    assert len(feature.audits[0].candidate_keys) == 2
    assert any(
        item.stage is ReconciliationStage.OCCURRENCE
        and item.action is ReconciliationAction.EXACT_DUPLICATE
        for item in feature.decisions
    )


def test_changed_same_occurrence_uses_unique_later_lifecycle_evidence(tmp_path: Path,) -> None:
    first = _outer_row(
        outer_id="C1",
        feature="Calories",
        version="1.0",
        payload=_calorie_payload(10.0),
        updated_at="2024-01-03 00:00:00",
    )
    revised = _outer_row(
        outer_id="C1",
        feature="Calories",
        version="1.0",
        payload=_calorie_payload(12.0),
        updated_at="2024-02-03 00:00:00",
    )
    parsed = _parse_participant(tmp_path, [("2024-1.csv", [first]), ("2024-2.csv", [revised])],)
    feature = clean_parse_result(parsed).features["Calories"]
    assert feature.output_rows == 1
    assert feature.rows[0]["value"] == 12.0
    assert feature.revisions_resolved == 1
    assert feature.exact_duplicates_removed == 0
    assert feature.audits[0].revision_count == 1
    assert any(
        item.stage is ReconciliationStage.OCCURRENCE
        and item.action is ReconciliationAction.REVISION_RESOLVED
        for item in feature.decisions
    )


def test_changed_same_occurrence_without_lifecycle_order_is_preserved_as_conflict(tmp_path: Path,) -> None:
    rows = [
        _outer_row(
            outer_id="C1",
            feature="Calories",
            version="1.0",
            payload=_calorie_payload(10.0),
        ),
        _outer_row(
            outer_id="C1",
            feature="Calories",
            version="1.0",
            payload=_calorie_payload(12.0),
        ),
    ]
    parsed = _parse_participant(tmp_path, [("2024-1.csv", rows)])
    feature = clean_parse_result(parsed).features["Calories"]
    assert feature.output_rows == 2
    assert feature.revisions_resolved == 0
    assert feature.occurrence_conflict_groups == 1
    assert feature.semantic_conflict_groups == 0
    assert feature.conflict_rows == 2
    assert all(item.conflict_group_ids for item in feature.audits)


def test_same_version_semantic_exact_duplicate_is_collapsed(tmp_path: Path) -> None:
    rows = [
        _outer_row(
            outer_id="C1",
            feature="Calories",
            version="1.0",
            payload=_calorie_payload(10.0),
        ),
        _outer_row(
            outer_id="C2",
            feature="Calories",
            version="1.0",
            payload=_calorie_payload(10.0),
        ),
    ]
    feature = clean_parse_result(
        _parse_participant(tmp_path, [("2024-1.csv", rows)])
    ).features["Calories"]
    assert feature.output_rows == 1
    assert feature.occurrence_duplicates_removed == 0
    assert feature.semantic_duplicates_removed == 1
    assert feature.represented_occurrences == 2
    assert any(
        item.stage is ReconciliationStage.SEMANTIC
        and item.action is ReconciliationAction.EXACT_DUPLICATE
        for item in feature.decisions
    )


def test_same_version_semantic_value_disagreement_is_preserved(tmp_path: Path) -> None:
    rows = [
        _outer_row(
            outer_id="C1",
            feature="Calories",
            version="1.0",
            payload=_calorie_payload(10.0),
        ),
        _outer_row(
            outer_id="C2",
            feature="Calories",
            version="1.0",
            payload=_calorie_payload(12.0),
        ),
    ]
    feature = clean_parse_result(
        _parse_participant(tmp_path, [("2024-1.csv", rows)])
    ).features["Calories"]
    assert feature.output_rows == 2
    assert feature.semantic_duplicates_removed == 0
    assert feature.semantic_conflict_groups == 1
    assert sorted(row["value"] for row in feature.rows) == [10.0, 12.0]


def test_repeated_equal_stepcount_v1_values_are_not_semantic_duplicates(tmp_path: Path,) -> None:
    rows = [
        _outer_row(
            outer_id="S1",
            feature="StepCount",
            version="1.0",
            payload=100,
        ),
        _outer_row(
            outer_id="S2",
            feature="StepCount",
            version="1.0",
            payload=100,
        ),
    ]
    feature = clean_parse_result(
        _parse_participant(tmp_path, [("2024-1.csv", rows)])
    ).features["StepCount"]
    assert feature.output_rows == 2
    assert feature.exact_duplicates_removed == 0
    assert feature.unresolved_conflict_groups == 0


def test_collecting_versions_are_never_reconciled(tmp_path: Path) -> None:
    rows = [
        _outer_row(
            outer_id="D1",
            feature="Distance",
            version="1.0",
            payload=_distance_payload(100.0),
        ),
        _outer_row(
            outer_id="D1",
            feature="Distance",
            version="2.0",
            payload=_distance_payload(100.0),
        ),
    ]
    feature = clean_parse_result(
        _parse_participant(tmp_path, [("2024-1.csv", rows)])
    ).features["Distance"]
    assert feature.output_rows == 2
    assert {row["collecting_method_version"] for row in feature.rows} == {"1.0", "2.0",}
    assert feature.exact_duplicates_removed == 0
    assert feature.revisions_resolved == 0


def test_stepcount_v2_alternatives_remain_distinct(tmp_path: Path) -> None:
    parsed = _parse_participant(
        tmp_path,
        [
            (
                "2024-1.csv",
                [
                    _outer_row(
                        outer_id="S2",
                        feature="StepCount",
                        version="2.0",
                        payload=[
                            _step_alternative(source="estimated_steps"),
                            _step_alternative(source="merge_step_deltas"),
                        ],
                    )
                ],
            )
        ],
    )
    feature = clean_parse_result(parsed).features["StepCount"]
    assert feature.output_rows == 2
    assert [row["aggregation_source"] for row in feature.rows] == ["estimated_steps", "merge_step_deltas",]
    assert feature.exact_duplicates_removed == 0


def test_unregistered_activity_remains_source_occurrence_only(tmp_path: Path) -> None:
    item = {
        "activity_name": "future.activity",
        "start": 1704067200000,
        "end": 1704067260000,
        "calories": 1.0,
        "distance": None,
        "quantity": 1.0,
        "tracked": True,
        "device": "Android",
        "source_name": "Android",
        "source_id": "org.example",
    }
    rows = [
        _outer_row(
            outer_id="A1",
            feature="Activities",
            version="1.0",
            payload=[item],
        ),
        _outer_row(
            outer_id="A2",
            feature="Activities",
            version="1.0",
            payload=[item],
        ),
    ]
    feature = clean_parse_result(
        _parse_participant(tmp_path, [("2024-1.csv", rows)])
    ).features["UnregisteredActivity"]
    assert feature.output_rows == 2
    assert feature.exact_duplicates_removed == 0


def test_cleaning_is_independent_of_candidate_and_source_input_order(tmp_path: Path,) -> None:
    rows = [
        _outer_row(
            outer_id="C3",
            feature="Calories",
            version="1.0",
            payload=_calorie_payload(30.0),
            outer_datetime="2024-01-03 00:00:00",
        ),
        _outer_row(
            outer_id="C1",
            feature="Calories",
            version="1.0",
            payload=_calorie_payload(10.0),
            outer_datetime="2024-01-01 00:00:00",
        ),
        _outer_row(
            outer_id="C2",
            feature="Calories",
            version="1.0",
            payload=_calorie_payload(20.0),
            outer_datetime="2024-01-02 00:00:00",
        ),
    ]
    parsed = _parse_participant(tmp_path, [("2024-1.csv", rows)])
    first = clean_parse_result(parsed)

    shuffled = copy.deepcopy(parsed)
    for feature in shuffled.records_by_feature:
        paired = list(
            zip(
                shuffled.records_by_feature[feature],
                shuffled.candidate_lineage_by_feature[feature],
                strict=True,
            )
        )
        paired.reverse()
        shuffled.records_by_feature[feature] = [item[0] for item in paired]
        shuffled.candidate_lineage_by_feature[feature] = [item[1] for item in paired]
    shuffled.source_occurrences.reverse()
    second = clean_parse_result(shuffled)

    assert second.output_fingerprint == first.output_fingerprint
    assert second.features["Calories"].rows == first.features["Calories"].rows
    assert second.features["Calories"].audits == first.features["Calories"].audits
    assert second.features["Calories"].decisions == first.features["Calories"].decisions


def test_cleaner_preserves_registered_schema_and_does_not_add_audit_columns(tmp_path: Path,) -> None:
    parsed = _parse_participant(
        tmp_path,
        [
            (
                "2024-1.csv",
                [
                    _outer_row(
                        outer_id="C1",
                        feature="Calories",
                        version="2.0",
                        payload=_calorie_payload(-12.5),
                    )
                ],
            )
        ],
    )
    before = copy.deepcopy(parsed.records_by_feature["Calories"])
    feature = clean_parse_result(parsed).features["Calories"]
    assert parsed.records_by_feature["Calories"] == before
    assert tuple(feature.rows[0]) == get_feature_spec("Calories").column_names
    assert not {
        "occurrence_count", "duplicate_count", "revision_count", "conflict_group_id",
    } & set(feature.rows[0])
    assert feature.rows[0]["value"] == -12.5


def test_merge_month_results_and_cleaning_are_deterministic(tmp_path: Path) -> None:
    participant = tmp_path / "P001"
    _write_month(
        tmp_path,
        [
            _outer_row(
                outer_id="D2",
                feature="Distance",
                version="1.0",
                payload=_distance_payload(2.0),
            )
        ],
        filename="2024-2.csv",
    )
    _write_month(
        tmp_path,
        [
            _outer_row(
                outer_id="D1",
                feature="Distance",
                version="1.0",
                payload=_distance_payload(1.0),
            )
        ],
        filename="2024-1.csv",
    )
    sources = discover_month_files(participant)
    january = parse_month_file(sources[0], "P001")
    february = parse_month_file(sources[1], "P001")
    forward = clean_parse_result(merge_parse_results([january, february]))
    reverse = clean_parse_result(merge_parse_results([february, january]))
    assert forward.output_fingerprint == reverse.output_fingerprint
    assert forward.features["Distance"].rows == reverse.features["Distance"].rows


def test_cleaner_has_no_apple_scientific_imports() -> None:
    path = (Path(__file__).parents[1] / "wearable_project" / "processing" / "google_fitness" / "cleaners.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not [name for name in imported if ".apple_healthkit" in name]


def test_all_phase0_representative_valid_candidates_clean_without_loss(tmp_path: Path,) -> None:
    fixture = (Path(__file__).parent / "fixtures" / "google_fitness_phase0_representative_items.json")
    import json

    entries = json.loads(fixture.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        item = entry["raw_item"]
        feature = entry["feature"]
        version = entry["collecting_method_version"]
        payload = item if (feature, version) in {("Weight", "1.0"), ("StepCount", "1.0")} else [item]
        rows.append(
            _outer_row(
                outer_id=f"R{index:03d}",
                feature=feature,
                version=version,
                payload=payload,
            )
        )
    parsed = _parse_participant(tmp_path, [("2024-1.csv", rows)])
    cleaned = clean_parse_result(parsed)
    assert parsed.valid_candidates == 46
    assert parsed.malformed_candidates == 1
    assert cleaned.input_candidates == 46
    assert cleaned.output_rows == 46
    assert cleaned.exact_duplicates_removed == 0
    assert cleaned.revisions_resolved == 0
    assert cleaned.unresolved_conflict_groups == 0
    cleaned.assert_accounting_closed()


def test_clean_feature_rejects_misaligned_parallel_lineage(tmp_path: Path) -> None:
    parsed = _parse_participant(
        tmp_path,
        [
            (
                "2024-1.csv",
                [
                    _outer_row(
                        outer_id="C1",
                        feature="Calories",
                        version="1.0",
                        payload=_calorie_payload(1.0),
                    )
                ],
            )
        ],
    )
    with pytest.raises(ValueError, match="candidates but 0 lineage"):
        clean_feature(
            "Calories",
            parsed.records_by_feature["Calories"],
            (),
            parsed.source_occurrences,
        )


def test_duplicate_then_revision_accounting_closes_exactly(tmp_path: Path) -> None:
    old = _outer_row(
        outer_id="C1",
        feature="Calories",
        version="1.0",
        payload=_calorie_payload(10.0),
        updated_at="2024-01-03 00:00:00",
    )
    new = _outer_row(
        outer_id="C1",
        feature="Calories",
        version="1.0",
        payload=_calorie_payload(12.0),
        updated_at="2024-03-03 00:00:00",
    )
    parsed = _parse_participant(
        tmp_path,[("2024-1.csv", [old]), ("2024-2.csv", [old]), ("2024-3.csv", [new]),],
    )
    feature = clean_parse_result(parsed).features["Calories"]
    assert feature.input_candidates == 3
    assert feature.output_rows == 1
    assert feature.exact_duplicates_removed == 1
    assert feature.revisions_resolved == 1
    assert feature.rows[0]["value"] == 12.0
    assert feature.audits[0].occurrence_count == 3
    assert feature.audits[0].duplicate_count == 1
    assert feature.audits[0].revision_count == 1
    feature.assert_accounting_closed()


def test_identical_payload_items_use_candidate_keys_not_outer_row_count(tmp_path: Path,) -> None:
    item = {
        "activity_name": "future.activity",
        "start": 1704067200000,
        "end": 1704067260000,
        "calories": 1.0,
        "distance": None,
        "quantity": 1.0,
        "tracked": True,
        "device": "Android",
        "source_name": "Android",
        "source_id": "org.example",
    }
    # Fallback activities are source-occurrence-only and therefore remain two rows; the audit still proves that
    # the two candidate occurrences are distinct even though they came from one outer row.
    parsed = _parse_participant(
        tmp_path,
        [
            (
                "2024-1.csv",
                [
                    _outer_row(
                        outer_id="A1",
                        feature="Activities",
                        version="1.0",
                        payload=[item, item],
                    )
                ],
            )
        ],
    )
    feature = clean_parse_result(parsed).features["UnregisteredActivity"]
    assert feature.input_candidates == 2
    assert feature.output_rows == 2
    assert sum(item.occurrence_count for item in feature.audits) == 2
    assert len({key for audit in feature.audits for key in audit.candidate_keys}) == 2
    assert len({key for audit in feature.audits for key in audit.source_occurrence_keys}) == 1


def test_payload_index_is_sorted_numerically_not_lexically(tmp_path: Path) -> None:
    activities = []
    for index in range(11):
        activities.append(
            {
                "activity_name": f"future.activity.{index:02d}",
                "start": 1704067200000,
                "end": 1704067260000,
                "calories": 1.0,
                "distance": None,
                "quantity": 1.0,
                "tracked": True,
                "device": "Android",
                "source_name": "Android",
                "source_id": "org.example",
            }
        )
    parsed = _parse_participant(
        tmp_path,
        [
            (
                "2024-1.csv",
                [
                    _outer_row(
                        outer_id="A1",
                        feature="Activities",
                        version="1.0",
                        payload=activities,
                    )
                ],
            )
        ],
    )
    feature = clean_parse_result(parsed).features["UnregisteredActivity"]
    # raw_activity_name precedes payload_index in the registered sort key, so make the labels equal before
    # invoking clean_feature directly to exercise the numeric final tie-break.
    records = [dict(item) for item in parsed.records_by_feature["UnregisteredActivity"]]
    for record in records:
        record["raw_activity_name"] = "future.activity"
    result = clean_feature(
        "UnregisteredActivity",
        list(reversed(records)),
        list(reversed(parsed.candidate_lineage_by_feature["UnregisteredActivity"])),
        parsed.source_occurrences,
    )
    assert [row["payload_index"] for row in result.rows] == list(range(11))


def test_canonical_month_is_a_supported_revision_signal_when_timestamps_tie(tmp_path: Path,) -> None:
    first = _outer_row(
        outer_id="C1",
        feature="Calories",
        version="1.0",
        payload=_calorie_payload(10.0),
        created_at="",
        updated_at="",
    )
    later = _outer_row(
        outer_id="C1",
        feature="Calories",
        version="1.0",
        payload=_calorie_payload(11.0),
        created_at="",
        updated_at="",
    )
    feature = clean_parse_result(
        _parse_participant(tmp_path, [("2024-1.csv", [first]), ("2024-2.csv", [later])],)
    ).features["Calories"]
    assert feature.output_rows == 1
    assert feature.revisions_resolved == 1
    assert feature.rows[0]["value"] == 11.0


def test_created_at_resolves_revision_when_updated_at_ties(tmp_path: Path) -> None:
    first = _outer_row(
        outer_id="C1",
        feature="Calories",
        version="1.0",
        payload=_calorie_payload(10.0),
        created_at="2024-01-01 00:00:00",
        updated_at="",
    )
    later = _outer_row(
        outer_id="C1",
        feature="Calories",
        version="1.0",
        payload=_calorie_payload(11.0),
        created_at="2024-01-02 00:00:00",
        updated_at="",
    )
    feature = clean_parse_result(
        _parse_participant(tmp_path, [("2024-1.csv", [first, later])])
    ).features["Calories"]
    assert feature.output_rows == 1
    assert feature.revisions_resolved == 1
    assert feature.rows[0]["value"] == 11.0


def test_invalid_lifecycle_strings_do_not_create_revision_order(tmp_path: Path) -> None:
    first = _outer_row(
        outer_id="C1",
        feature="Calories",
        version="1.0",
        payload=_calorie_payload(10.0),
        created_at="not-a-date-a",
        updated_at="not-a-date-a",
    )
    second = _outer_row(
        outer_id="C1",
        feature="Calories",
        version="1.0",
        payload=_calorie_payload(11.0),
        created_at="not-a-date-z",
        updated_at="not-a-date-z",
    )
    feature = clean_parse_result(
        _parse_participant(tmp_path, [("2024-1.csv", [first, second])])
    ).features["Calories"]
    assert feature.output_rows == 2
    assert feature.revisions_resolved == 0
    assert feature.occurrence_conflict_groups == 1


def test_stepcount_v2_same_alternative_exact_semantic_duplicate_collapses(tmp_path: Path,) -> None:
    rows = [
        _outer_row(
            outer_id="S1",
            feature="StepCount",
            version="2.0",
            payload=[_step_alternative(source="estimated_steps", value=100)],
        ),
        _outer_row(
            outer_id="S2",
            feature="StepCount",
            version="2.0",
            payload=[_step_alternative(source="estimated_steps", value=100)],
        ),
    ]
    feature = clean_parse_result(
        _parse_participant(tmp_path, [("2024-1.csv", rows)])
    ).features["StepCount"]
    assert feature.output_rows == 1
    assert feature.semantic_duplicates_removed == 1
    assert feature.unresolved_conflict_groups == 0


def test_stepcount_v2_same_identity_value_disagreement_is_preserved(tmp_path: Path,) -> None:
    rows = [
        _outer_row(
            outer_id="S1",
            feature="StepCount",
            version="2.0",
            payload=[_step_alternative(source="estimated_steps", value=100)],
        ),
        _outer_row(
            outer_id="S2",
            feature="StepCount",
            version="2.0",
            payload=[_step_alternative(source="estimated_steps", value=101)],
        ),
    ]
    feature = clean_parse_result(
        _parse_participant(tmp_path, [("2024-1.csv", rows)])
    ).features["StepCount"]
    assert feature.output_rows == 2
    assert feature.semantic_duplicates_removed == 0
    assert feature.semantic_conflict_groups == 1
    assert sorted(row["value"] for row in feature.rows) == [100, 101]


def test_participant_accounting_closes_valid_and_malformed_candidates(tmp_path: Path,) -> None:
    valid_weight = {
        "start_date": "2024-01-01T00:00:00.000Z",
        "end_date": "2024-01-01T00:00:00.000Z",
        "value": 70.0,
        "day": "Mon",
        "added_by": "org.example.scale",
    }
    parsed = _parse_participant(
        tmp_path,
        [
            (
                "2024-1.csv",
                [
                    _outer_row(
                        outer_id="W1",
                        feature="Weight",
                        version="2.0",
                        payload=[valid_weight],
                    ),
                    _outer_row(
                        outer_id="W2",
                        feature="Weight",
                        version="2.0",
                        payload=[None],
                    )
                ],
            )
        ],
    )
    cleaned = clean_parse_result(parsed)
    assert cleaned.candidate_items_seen == 2
    assert cleaned.valid_candidates == cleaned.input_candidates == 1
    assert cleaned.malformed_candidates == 1
    assert cleaned.output_rows == 1
    payload = cleaned.accounting_payload()
    assert payload["candidate_items_seen"] == 2
    assert payload["valid_candidates"] == 1
    assert payload["malformed_candidates"] == 1
    assert payload["accounting_issues"] == []



def test_cleaner_models_and_results_are_spawn_process_safe(tmp_path: Path) -> None:
    participant = _write_month(
        tmp_path,
        [
            _outer_row(
                outer_id="C1",
                feature="Calories",
                version="1.0",
                payload=_calorie_payload(10.0),
            ),
            _outer_row(
                outer_id="S1",
                feature="StepCount",
                version="2.0",
                payload=[
                    _step_alternative(source="estimated_steps", value=100),
                    _step_alternative(source="merge_step_deltas", value=101),
                ],
            ),
        ],
    )
    _, parsed = parse_participant_directory(participant)
    local = clean_parse_result(parsed)
    with ProcessPoolExecutor(
        max_workers=1, mp_context=mp.get_context("spawn"),
    ) as executor:
        remote_fingerprint, remote_accounting = executor.submit(
            _spawn_parse_and_clean, str(participant)
        ).result(timeout=60)
    assert remote_fingerprint == local.output_fingerprint
    assert remote_accounting == local.accounting_payload()



def test_valid_lifecycle_timestamp_does_not_beat_missing_evidence_by_itself(tmp_path: Path,) -> None:
    known = _outer_row(
        outer_id="C1",
        feature="Calories",
        version="1.0",
        payload=_calorie_payload(10.0),
        created_at="",
        updated_at="2024-01-03 00:00:00",
    )
    unknown = _outer_row(
        outer_id="C1",
        feature="Calories",
        version="1.0",
        payload=_calorie_payload(11.0),
        created_at="",
        updated_at="not-a-date",
    )
    feature = clean_parse_result(
        _parse_participant(tmp_path, [("2024-1.csv", [known, unknown])])
    ).features["Calories"]
    assert feature.output_rows == 2
    assert feature.revisions_resolved == 0
    assert feature.occurrence_conflict_groups == 1



def test_reconciliation_ledgers_are_json_serializable(tmp_path: Path) -> None:
    rows = [
        _outer_row(
            outer_id="C1",
            feature="Calories",
            version="1.0",
            payload=_calorie_payload(10.0),
        ),
        _outer_row(
            outer_id="C2",
            feature="Calories",
            version="1.0",
            payload=_calorie_payload(10.0),
        ),
    ]
    feature = clean_parse_result(
        _parse_participant(tmp_path, [("2024-1.csv", rows)])
    ).features["Calories"]
    import json

    serialized = json.dumps(
        {
            "audits": [item.as_dict() for item in feature.audits],
            "decisions": [item.as_dict() for item in feature.decisions],
            "accounting": feature.accounting_payload(),
        },
        sort_keys=True,
    )
    assert "exact_duplicate" in serialized


def test_clean_feature_rejects_source_mapping_key_mismatch(tmp_path: Path) -> None:
    parsed = _parse_participant(
        tmp_path,
        [
            (
                "2024-1.csv",
                [
                    _outer_row(
                        outer_id="C1",
                        feature="Calories",
                        version="1.0",
                        payload=_calorie_payload(1.0),
                    )
                ],
            )
        ],
    )
    with pytest.raises(ValueError, match="mapping keys"):
        clean_feature(
            "Calories",
            parsed.records_by_feature["Calories"],
            parsed.candidate_lineage_by_feature["Calories"],
            {"not-the-occurrence-key": parsed.source_occurrences[0]},
        )


def test_clean_feature_rejects_registered_schema_drift(tmp_path: Path) -> None:
    parsed = _parse_participant(
        tmp_path,
        [
            (
                "2024-1.csv",
                [
                    _outer_row(
                        outer_id="C1",
                        feature="Calories",
                        version="1.0",
                        payload=_calorie_payload(1.0),
                    )
                ],
            )
        ],
    )
    record = dict(parsed.records_by_feature["Calories"][0])
    record.pop("source_day_label")
    with pytest.raises(ValueError, match="candidate columns"):
        clean_feature(
            "Calories",
            [record],
            parsed.candidate_lineage_by_feature["Calories"],
            parsed.source_occurrences,
        )


def test_clean_feature_rejects_nonpositive_activity_interval(tmp_path: Path) -> None:
    item = {
        "activity_name": "walking",
        "start": 1704067200000,
        "end": 1704067260000,
        "calories": 1.0,
        "distance": 10.0,
        "quantity": 5.0,
        "tracked": True,
        "device": "Android",
        "source_name": "Android",
        "source_id": "org.example",
    }
    parsed = _parse_participant(
        tmp_path,
        [
            (
                "2024-1.csv",
                [
                    _outer_row(
                        outer_id="A1",
                        feature="Activities",
                        version="1.0",
                        payload=[item],
                    )
                ],
            )
        ],
    )
    record = dict(parsed.records_by_feature["Walking"][0])
    record["end_date"] = record["start_date"]
    with pytest.raises(ValueError, match="positive duration"):
        clean_feature(
            "Walking",
            [record],
            parsed.candidate_lineage_by_feature["Walking"],
            parsed.source_occurrences,
        )


def test_stepcount_v1_rejects_invented_version2_evidence(tmp_path: Path) -> None:
    parsed = _parse_participant(
        tmp_path,
        [
            (
                "2024-1.csv",
                [
                    _outer_row(
                        outer_id="S1",
                        feature="StepCount",
                        version="1.0",
                        payload=100,
                    )
                ],
            )
        ],
    )
    record = dict(parsed.records_by_feature["StepCount"][0])
    record["summary_date"] = "2024-01-01"
    with pytest.raises(ValueError, match="invented version-2 fields"):
        clean_feature(
            "StepCount",
            [record],
            parsed.candidate_lineage_by_feature["StepCount"],
            parsed.source_occurrences,
        )
