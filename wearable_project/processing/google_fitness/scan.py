"""
Wearable Data Processing and Modeling project
Read-only planning for Google Fitness WP4 native processing.
"""


from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from wearable_project.exceptions import OutputValidationError
from wearable_project.processing.google_fitness.pipeline import (
    PlanAction, discover_participants, plan_participant, prehash_participant_sources, resolve_export_root,
)
from wearable_project.processing.google_fitness.registry import SOURCE_KEY
from wearable_project.processing.google_fitness.writer import (
    FeatureOutputInfo, STATE_FILENAME, StoredParticipantState, read_only_mode,
)


@dataclass(frozen=True, slots=True)
class ParticipantScan:
    participant_id: str
    action: str
    reason: str
    source_files: int
    source_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "action": self.action,
            "reason": self.reason,
            "source_files": self.source_files,
            "source_bytes": self.source_bytes,
        }


@dataclass(frozen=True, slots=True)
class ProcessingScanReport:
    input_root: str
    output_root: str
    state_file: str
    participants_discovered: int
    mode: str
    snapshot_policy: str
    verify_existing_hashes: bool
    workers_requested: int
    workers_effective: int
    counts: dict[str, int]
    safe_to_process: bool
    processing_needed: bool
    no_op: bool
    errors: tuple[str, ...]
    participants: tuple[ParticipantScan, ...]

    def as_dict(self, *, include_details: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_key": SOURCE_KEY,
            "input_root": self.input_root,
            "output_root": self.output_root,
            "state_file": self.state_file,
            "participants_discovered": self.participants_discovered,
            "configuration": {
                "mode": self.mode,
                "snapshot_policy": self.snapshot_policy,
                "verify_existing_hashes": self.verify_existing_hashes,
                "workers_requested": self.workers_requested,
                "workers_effective": self.workers_effective,
            },
            "validation": {
                "source_files": "sha256",
                "existing_outputs": (
                    "file_set_size_and_sha256"
                    if self.verify_existing_hashes
                    else "file_set_and_size"
                ),
            },
            "counts": dict(self.counts),
            "safe_to_process": self.safe_to_process,
            "processing_needed": self.processing_needed,
            "no_op": self.no_op,
            "errors": list(self.errors),
        }
        if include_details:
            payload["participants"] = [item.as_dict() for item in self.participants]
        return payload


class ReadOnlyState:
    def __init__(self, path: Path):
        self.path = path
        self.connection: sqlite3.Connection | None = None
        if path.is_file():
            self.connection = sqlite3.connect(f"file:{path}?{read_only_mode(path)}", uri=True)
            self.connection.row_factory = sqlite3.Row
            try:
                row = self.connection.execute("SELECT value FROM metadata WHERE key='source_key'").fetchone()
            except sqlite3.Error as exc:
                raise OutputValidationError(f"not a Google Fitness WP4 state database: {path}") from exc
            if row is None or row["value"] != SOURCE_KEY:
                raise OutputValidationError(
                    f"output state is managed by {None if row is None else row['value']!r}, not {SOURCE_KEY!r}"
                )

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()

    def get_participant(self, participant_id: str) -> StoredParticipantState | None:
        if self.connection is None:
            return None
        row = self.connection.execute(
            "SELECT * FROM participants WHERE participant_id=?", (participant_id,)
        ).fetchone()
        if row is None:
            return None
        sources = {
            item["canonical_month"]: {
                "filename": item["filename"],
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            }
            for item in self.connection.execute(
                "SELECT * FROM source_files WHERE participant_id=?", (participant_id,)
            )
        }
        outputs = {
            item["feature"]: FeatureOutputInfo(
                item["feature"],
                item["filename"],
                item["row_count"],
                item["size_bytes"],
                item["sha256"],
                item["represented_occurrences"],
            )
            for item in self.connection.execute(
                "SELECT * FROM feature_outputs WHERE participant_id=?", (participant_id,)
            )
        }
        return StoredParticipantState(
            participant_id=row["participant_id"],
            status=row["status"],
            parser_version=row["parser_version"],
            registry_version=row["registry_version"],
            cleaner_version=row["cleaner_version"],
            pipeline_version=row["pipeline_version"],
            source_signature=row["source_signature"],
            last_error=row["last_error"],
            sources=sources,
            outputs=outputs,
        )


def scan_processing_plan(
    input_root: Path, output_root: Path, *, workers: int = 4, mode: str = "auto",
    snapshot_policy: str = "strict-cumulative", selected_participants: set[str] | None = None,
    verify_existing_hashes: bool = False, state_file: Path | None = None, show_progress: bool = True,
) -> ProcessingScanReport:
    input_root = resolve_export_root(Path(input_root))
    output_root = Path(output_root).expanduser().resolve()
    state_file = (
        Path(state_file).expanduser().resolve()
        if state_file is not None
        else output_root / STATE_FILENAME
    )
    errors: list[str] = []
    if output_root == input_root or input_root in output_root.parents:
        errors.append("output root lies inside the raw input root")
    if output_root in input_root.parents:
        errors.append("raw input root lies inside the output root")
    if output_root.exists() and not state_file.exists():
        visible = [item for item in output_root.iterdir() if not item.name.startswith(".")]
        if visible:
            errors.append("nonempty output root lacks Google Fitness state")

    participants = discover_participants(input_root, selected_participants)
    source_results = prehash_participant_sources(participants, workers, show_progress=show_progress)
    details: list[ParticipantScan] = []
    state: ReadOnlyState | None = None
    try:
        try:
            state = ReadOnlyState(state_file)
        except Exception as exc:
            errors.append(str(exc))
            state = None
        for participant_id, participant_dir in participants.items():
            value = source_results.get(participant_id)
            if isinstance(value, str):
                errors.append(f"{participant_id}: {value}")
                details.append(ParticipantScan(participant_id, "error", value, 0, 0))
                continue
            sources = list(value or [])
            if state is None and state_file.exists():
                details.append(
                    ParticipantScan(
                        participant_id,
                        "error",
                        "state database could not be opened",
                        len(sources),
                        sum(item.size_bytes for item in sources),
                    )
                )
                continue
            plan = plan_participant(
                participant_id,
                participant_dir,
                sources,
                None if state is None else state.get_participant(participant_id),
                output_root,
                mode=mode,
                snapshot_policy=snapshot_policy,
                verify_existing_hashes=verify_existing_hashes,
            )
            details.append(
                ParticipantScan(
                    participant_id,
                    plan.action.value,
                    plan.reason,
                    len(sources),
                    sum(item.size_bytes for item in sources),
                )
            )
    finally:
        if state is not None:
            state.close()

    counts: dict[str, int] = {}
    for item in details:
        counts[item.action] = counts.get(item.action, 0) + 1
    safe = not errors and counts.get("error", 0) == 0
    needed = bool(counts.get(PlanAction.REBUILD.value, 0) or counts.get(PlanAction.INCREMENTAL.value, 0))
    no_op = safe and not needed and not counts.get(PlanAction.BLOCK.value, 0)
    return ProcessingScanReport(
        input_root=str(input_root),
        output_root=str(output_root),
        state_file=str(state_file),
        participants_discovered=len(participants),
        mode=mode,
        snapshot_policy=snapshot_policy,
        verify_existing_hashes=bool(verify_existing_hashes),
        workers_requested=workers,
        workers_effective=(min(workers, len(participants)) if participants else 0),
        counts=dict(sorted(counts.items())),
        safe_to_process=safe,
        processing_needed=needed,
        no_op=no_op,
        errors=tuple(errors),
        participants=tuple(sorted(details, key=lambda item: item.participant_id)),
    )


def format_processing_scan(report: ProcessingScanReport | dict[str, Any], *, max_listed: int = 50,) -> str:
    payload = report.as_dict(include_details=True) if isinstance(report, ProcessingScanReport) else report
    lines = [
        "Google Fitness processing scan",
        f"Input: {payload['input_root']}",
        f"Output: {payload['output_root']}",
        f"Participants: {payload['participants_discovered']}",
        "Configuration: "
        f"mode={payload.get('configuration', {}).get('mode')}, "
        f"snapshot_policy={payload.get('configuration', {}).get('snapshot_policy')}, "
        f"verify_existing_hashes={payload.get('configuration', {}).get('verify_existing_hashes')}",
        "Validation: "
        f"source_files={payload.get('validation', {}).get('source_files')}, "
        f"existing_outputs={payload.get('validation', {}).get('existing_outputs')}",
        "Actions: " + ", ".join(f"{name}={count}" for name, count in sorted(payload.get("counts", {}).items())),
        f"Safe to process: {payload['safe_to_process']}",
        f"Processing needed: {payload['processing_needed']}",
        f"No-op: {payload['no_op']}",
    ]
    if payload.get("errors"):
        lines.append("Errors:")
        lines.extend(f"  - {item}" for item in payload["errors"])
    non_skip = [item for item in payload.get("participants", []) if item.get("action") != PlanAction.SKIP.value]
    if non_skip:
        lines.append("Non-skip participants:")
        for item in non_skip[:max_listed]:
            lines.append(f"  {item['participant_id']}: {item['action']} — {item['reason']}")
        if len(non_skip) > max_listed:
            lines.append(f"  ... {len(non_skip) - max_listed} more")
    return "\n".join(lines) + "\n"
