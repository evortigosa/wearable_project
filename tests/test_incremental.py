"""
Wearable Data Processing and Modeling project
"""


from pathlib import Path
from wearable_project.processing.parser import SourceFile
from wearable_project.processing.pipeline import PARSER_VERSION, PlanAction, plan_participant
from wearable_project.processing.registry import REGISTRY_VERSION
from wearable_project.processing.writer import StoredParticipantState


def source(month, digest):
    return SourceFile(Path(f"{month}.csv"), month, digest, 10)


def stored(months):
    return StoredParticipantState(
        "P1", "complete_empty", PARSER_VERSION, REGISTRY_VERSION, "x", None,
        {month: {"filename": f"{month}.csv", "sha256": digest, "size_bytes": 10} for month, digest in months.items()}, {},
    )


def plan(tmp_path, sources, state, policy="strict-cumulative"):
    return plan_participant("P1", tmp_path, sources, state, tmp_path, mode="auto", snapshot_policy=policy, verify_existing_hashes=False)


def test_only_future_month_is_incremental(tmp_path):
    result = plan(tmp_path, [source("2024-12", "a"), source("2025-01", "b")], stored({"2024-12": "a"}))
    assert result.action == PlanAction.INCREMENTAL
    assert [item.canonical_month for item in result.sources_to_parse] == ["2025-01"]


def test_changed_month_rebuilds(tmp_path):
    assert plan(tmp_path, [source("2024-12", "b")], stored({"2024-12": "a"})).action == PlanAction.REBUILD


def test_recovered_historical_month_rebuilds(tmp_path):
    result = plan(tmp_path, [source("2024-11", "b"), source("2024-12", "a")], stored({"2024-12": "a"}))
    assert result.action == PlanAction.REBUILD


def test_missing_month_blocks_strict_snapshot(tmp_path):
    assert plan(tmp_path, [source("2025-01", "b")], stored({"2024-12": "a"})).action == PlanAction.BLOCK


def test_older_audit_count_contract_forces_rebuild(tmp_path):
    state = stored({"2024-12": "a"})
    state.parser_version = "native-parser-0.1.1"
    result = plan(tmp_path, [source("2024-12", "a")], state)
    assert result.action == PlanAction.REBUILD
    assert "parser or feature registry version changed" in result.reason


def test_failed_v013_participant_rebuilds_under_v014_compatibility_patch(tmp_path):
    state = stored({"2024-12": "a"})
    state.status = "failed"
    result = plan(tmp_path, [source("2024-12", "a")], state)
    assert result.action == PlanAction.REBUILD
    assert "previous state" in result.reason
