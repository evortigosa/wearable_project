"""
Wearable Data Processing and Modeling project
Formal full-cohort Google Fitness curation acceptance (CWP6). CWP5 proves the native-to-curated contract for
every physical participant-feature file. CWP6 composes that product-level audit with cohort-release checks that
are meaningful only after a complete curation run exists:

* the native participant-status ledger closes, including ``complete_empty`` rows;
* the current native snapshot matches an explicit, versioned snapshot manifest;
* the curated participant set equals the native ``complete`` participant set;
* a successful build run exists;
* the latest completed curation run is a hash-verified true no-op;
* current policies classify every participant as ``skip``;
* the acceptance process is read-only for both state databases.

The module does not load Apple DataLoaders and does not replace the later source-aware DataLoader acceptance layer.
It validates the persisted Google Fitness products directly from their state databases and physical CSV files.
"""


from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
from wearable_project.utils.gfit_acceptance import run_acceptance
from wearable_project.curation.google_fitness.contract import CURATED_SCHEMA_VERSION, contract_fingerprint
from wearable_project.curation.google_fitness.decisions import decisions_fingerprint
from wearable_project.curation.google_fitness.engine import ENGINE_VERSION, sha256_file
from wearable_project.curation.google_fitness.evidence import evidence_fingerprint
from wearable_project.curation.google_fitness.guidance import guidance_fingerprint
from wearable_project.curation.google_fitness.pipeline import PIPELINE_VERSION
from wearable_project.curation.google_fitness.registry import registry_fingerprint
from wearable_project.curation.google_fitness.rules import rules_fingerprint
from wearable_project.curation.google_fitness.state import CURATION_STATE_SCHEMA_VERSION
from wearable_project.curation.google_fitness.state import CURATION_STATE_FILENAME
from wearable_project.exceptions import InputLayoutError
from wearable_project.processing.google_fitness.registry import SOURCE_KEY
from wearable_project.processing.google_fitness.writer import STATE_FILENAME as NATIVE_STATE_FILENAME


COHORT_ACCEPTANCE_VERSION = "google-fitness-curation-cohort-acceptance-schema-v2-0.2.0"
COHORT_REPORT_SCHEMA_VERSION = "google-fitness-curation-cohort-acceptance-report-v2-0.2.0"
EXPECTED_SNAPSHOT_SCHEMA_VERSION = "google-fitness-curation-expected-native-snapshot-v2-0.2.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _state_quiet(path: Path, label: str) -> None:
    wal = path.with_name(path.name + "-wal")
    if wal.is_file() and wal.stat().st_size:
        raise InputLayoutError(f"{label} state has an active WAL: {wal}")


def _connect_immutable(path: Path, label: str) -> sqlite3.Connection:
    if not path.is_file():
        raise InputLayoutError(f"{label} state does not exist: {path}")
    _state_quiet(path, label)
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _guard_roots(native_root: Path, curated_root: Path, output: Path) -> tuple[Path, Path, Path]:
    native = Path(native_root).expanduser().resolve()
    curated = Path(curated_root).expanduser().resolve()
    report = Path(output).expanduser().resolve()
    if not native.is_dir():
        raise InputLayoutError(f"native root is not a directory: {native}")
    if not curated.is_dir():
        raise InputLayoutError(f"curated root is not a directory: {curated}")
    if native == curated or native in curated.parents or curated in native.parents:
        raise InputLayoutError(f"native and curated roots must be distinct and non-nested: {native}, {curated}")
    for protected in (native, curated):
        if report == protected or protected in report.parents or report in protected.parents:
            raise InputLayoutError(f"cohort-acceptance output must be outside both data roots: {report}")
    return native, curated, report


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        return {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key,value FROM metadata")
        }
    except sqlite3.Error as exc:
        raise InputLayoutError("state database lacks a readable metadata table") from exc


def _safe_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_native_snapshot(state_path: Path) -> dict[str, Any]:
    with _connect_immutable(state_path, "native") as connection:
        metadata = _metadata(connection)
        if metadata.get("source_key") != SOURCE_KEY:
            raise InputLayoutError(
                f"native state source is {metadata.get('source_key')!r}, expected {SOURCE_KEY!r}"
            )
        participant_status_counts = {
            str(row["status"]): int(row["count"])
            for row in connection.execute(
                "SELECT status,COUNT(*) AS count FROM participants GROUP BY status ORDER BY status"
            )
        }
        feature_row = connection.execute(
            "SELECT COUNT(*) AS feature_files,COUNT(DISTINCT participant_id) AS participants_with_files,"
            "COALESCE(SUM(row_count),0) AS native_rows,COALESCE(SUM(size_bytes),0) AS native_size_bytes,"
            "COALESCE(SUM(represented_occurrences),0) AS represented_occurrences "
            "FROM feature_outputs"
        ).fetchone()
        feature_names = [
            str(row["feature"])
            for row in connection.execute(
                "SELECT DISTINCT feature FROM feature_outputs ORDER BY feature"
            )
        ]
        malformed_breakdown = []
        for row in connection.execute(
            "SELECT source_feature,collecting_method_version,reason_codes_json,COUNT(*) AS count "
            "FROM malformed_candidates GROUP BY source_feature,collecting_method_version,reason_codes_json "
            "ORDER BY source_feature,collecting_method_version,reason_codes_json"
        ):
            try:
                reasons = json.loads(row["reason_codes_json"])
            except (TypeError, json.JSONDecodeError):
                reasons = row["reason_codes_json"]
            malformed_breakdown.append({
                "source_feature": str(row["source_feature"]),
                "collecting_method_version": str(row["collecting_method_version"]),
                "reason_codes": reasons,
                "count": int(row["count"]),
            })
        diagnostics_by_code = {
            str(row["code"]): int(row["count"])
            for row in connection.execute(
                "SELECT code,COUNT(*) AS count FROM parser_diagnostics GROUP BY code ORDER BY code"
            )
        }
        active_runs = int(connection.execute(
            "SELECT COUNT(*) FROM runs WHERE status='in_progress'"
        ).fetchone()[0])
        source_occurrences = int(connection.execute(
            "SELECT COUNT(*) FROM source_occurrences"
        ).fetchone()[0])
        malformed_candidates = int(connection.execute(
            "SELECT COUNT(*) FROM malformed_candidates"
        ).fetchone()[0])
        parser_diagnostics = int(connection.execute(
            "SELECT COUNT(*) FROM parser_diagnostics"
        ).fetchone()[0])
        reconciliation_decisions = int(connection.execute(
            "SELECT COUNT(*) FROM reconciliation_decisions"
        ).fetchone()[0])

    complete = int(participant_status_counts.get("complete", 0))
    complete_empty = int(participant_status_counts.get("complete_empty", 0))
    participant_records = sum(participant_status_counts.values())
    return {
        "source_key": metadata.get("source_key"),
        "state_schema_version": metadata.get("state_schema_version"),
        "registry_version": metadata.get("registry_version"),
        "participant_status_counts": dict(sorted(participant_status_counts.items())),
        "participant_records": participant_records,
        "complete_participants": complete,
        "complete_empty_participants": complete_empty,
        "participants_with_feature_files": int(feature_row["participants_with_files"]),
        "feature_files": int(feature_row["feature_files"]),
        "feature_types": len(feature_names),
        "feature_names": feature_names,
        "activities_feature_present": "Activities" in feature_names,
        "native_rows": int(feature_row["native_rows"]),
        "native_size_bytes": int(feature_row["native_size_bytes"]),
        "represented_occurrences": int(feature_row["represented_occurrences"]),
        "source_occurrences": source_occurrences,
        "malformed_candidates": malformed_candidates,
        "malformed_breakdown": malformed_breakdown,
        "parser_diagnostics": parser_diagnostics,
        "parser_diagnostics_by_code": diagnostics_by_code,
        "reconciliation_decisions": reconciliation_decisions,
        "active_processing_runs": active_runs,
    }


def _curation_run_history(state_path: Path) -> list[dict[str, Any]]:
    with _connect_immutable(state_path, "curated") as connection:
        metadata = _metadata(connection)
        if metadata.get("source_key") != SOURCE_KEY:
            raise InputLayoutError(
                f"curated state source is {metadata.get('source_key')!r}, expected {SOURCE_KEY!r}"
            )
        rows = list(connection.execute(
            "SELECT run_id,started_at,finished_at,input_native_root,output_root,status,"
            "configuration_json,summary_json FROM curation_runs ORDER BY started_at,run_id"
        ))
        participant_counts = {
            (str(row["run_id"]), str(row["final_status"])): int(row["count"])
            for row in connection.execute(
                "SELECT run_id,final_status,COUNT(*) AS count FROM curation_run_participants "
                "GROUP BY run_id,final_status ORDER BY run_id,final_status"
            )
        }
    history: list[dict[str, Any]] = []
    for row in rows:
        run_id = str(row["run_id"])
        configuration = _safe_json(row["configuration_json"])
        summary = _safe_json(row["summary_json"])
        participants = summary.get("participants")
        if not isinstance(participants, Mapping):
            participants = {}
        final_status_counts = {
            status: count
            for (stored_run_id, status), count in participant_counts.items()
            if stored_run_id == run_id
        }
        history.append({
            "run_id": run_id,
            "started_at": str(row["started_at"]),
            "finished_at": row["finished_at"],
            "input_native_root": str(row["input_native_root"]),
            "output_root": str(row["output_root"]),
            "status": str(row["status"]),
            "participants": {
                "discovered": int(participants.get("discovered", 0)),
                "skipped": int(participants.get("skipped", 0)),
                "committed": int(participants.get("committed", 0)),
                "failed": int(participants.get("failed", 0)),
            },
            "final_status_counts": dict(sorted(final_status_counts.items())),
            "verify_native_hashes": bool(configuration.get("verify_native_hashes", False)),
            "verify_existing_hashes": bool(configuration.get("verify_existing_hashes", False)),
            "engine_version": configuration.get("engine_version"),
            "pipeline_version": configuration.get("pipeline_version"),
            "registry_fingerprint": configuration.get("registry_fingerprint"),
            "evidence_fingerprint": configuration.get("evidence_fingerprint"),
            "rules_fingerprint": configuration.get("rules_fingerprint"),
            "decisions_fingerprint": configuration.get("decisions_fingerprint"),
            "guidance_fingerprint": configuration.get("guidance_fingerprint"),
        })
    return history


def current_policy_fingerprints() -> dict[str, str]:
    """Return the exact curation implementation identity used by CWP6."""

    return {
        "registry": registry_fingerprint(),
        "evidence": evidence_fingerprint(),
        "rules": rules_fingerprint(),
        "decisions": decisions_fingerprint(),
        "guidance": guidance_fingerprint(),
        "engine": ENGINE_VERSION,
        "pipeline": PIPELINE_VERSION,
        "state_schema": CURATION_STATE_SCHEMA_VERSION,
        "curated_schema": CURATED_SCHEMA_VERSION,
        "contract": contract_fingerprint(),
    }


def build_expected_snapshot_from_native_root(reference_id: str, native_root: Path) -> dict[str, Any]:
    """Build a release manifest for one explicit cumulative native snapshot."""

    root = Path(native_root).expanduser().resolve()
    return build_expected_snapshot_manifest(
        reference_id,
        _expected_view(
            read_native_snapshot(root / NATIVE_STATE_FILENAME),
            current_policy_fingerprints(),
        ),
    )


def _expected_view(native_snapshot: Mapping[str, Any], fingerprints: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_key": SOURCE_KEY,
        "native_snapshot": dict(native_snapshot),
        "fingerprints": dict(fingerprints),
    }


def build_expected_snapshot_manifest(reference_id: str, expected_view: Mapping[str, Any]) -> dict[str, Any]:
    """
    Create a snapshot-specific CWP6 expected-native manifest. The manifest is a release artifact, not a scientific
    policy constant. A future cumulative snapshot should receive a new reference ID and manifest.
    """

    return {
        "expected_snapshot_schema_version": EXPECTED_SNAPSHOT_SCHEMA_VERSION,
        "reference_id": reference_id,
        "expected": dict(expected_view),
    }


def _compare_expected(expected_manifest: Mapping[str, Any], actual: Mapping[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if expected_manifest.get("expected_snapshot_schema_version") != EXPECTED_SNAPSHOT_SCHEMA_VERSION:
        return [{
            "path": "expected_snapshot_schema_version",
            "expected": EXPECTED_SNAPSHOT_SCHEMA_VERSION,
            "observed": expected_manifest.get("expected_snapshot_schema_version"),
        }]
    expected = expected_manifest.get("expected")
    if not isinstance(expected, Mapping):
        return [{"path": "expected", "expected": "mapping", "observed": type(expected).__name__}]

    def compare(path: str, wanted: Any, observed: Any) -> None:
        if isinstance(wanted, Mapping):
            if not isinstance(observed, Mapping):
                issues.append({"path": path, "expected": wanted, "observed": observed})
                return
            for key in sorted(wanted):
                compare(f"{path}.{key}" if path else str(key), wanted[key], observed.get(key))
        elif isinstance(wanted, list):
            if wanted != observed:
                issues.append({"path": path, "expected": wanted, "observed": observed})
        elif wanted != observed:
            issues.append({"path": path, "expected": wanted, "observed": observed})

    compare("", expected, actual)
    return issues


def _check(check_id: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"check_id": check_id, "status": "pass" if passed else "fail", "detail": detail}


def _publish(stage: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise InputLayoutError(f"cohort-acceptance output already exists; use --overwrite: {output}")
        backup = output.with_name(f".{output.name}.backup")
        shutil.rmtree(backup, ignore_errors=True)
        os.replace(output, backup)
        try:
            os.replace(stage, output)
        except Exception:
            os.replace(backup, output)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, output)


def _render(payload: Mapping[str, Any]) -> str:
    native = payload["native_snapshot"]
    product = payload["product_acceptance"]["snapshot"]
    lines = [
        "# Google Fitness CWP6 full-cohort curation acceptance",
        "",
        f"- Status: **{str(payload['status']).upper()}**",
        f"- Cohort acceptance version: `{payload['cohort_acceptance_version']}`",
        f"- Expected snapshot: `{payload['expected_snapshot']['reference_id']}`",
        f"- Data-bearing participants: {native['participant_records']:,}",
        f"- Participants with Google feature files: {native['participants_with_feature_files']:,}",
        f"- Participant-feature files: {product['feature_files']:,}",
        f"- Native rows: {product['native_rows']:,}",
        f"- Curated rows: {product['curated_rows']:,}",
        f"- Failed checks: {payload['failed_check_count']}",
        "",
        "## Checks",
        "",
    ]
    for item in payload["checks"]:
        marker = "PASS" if item["status"] == "pass" else "FAIL"
        lines.append(f"- **{marker}** `{item['check_id']}` — {item['detail']}")
    lines.extend([
        "",
        "## Native snapshot",
        "",
        "```json",
        json.dumps(native, indent=2, sort_keys=True),
        "```",
        "",
        "## Curation run evidence",
        "",
        "```json",
        json.dumps(payload["run_evidence"], indent=2, sort_keys=True),
        "```",
        "",
    ])
    return "\n".join(lines)


def run_cohort_acceptance(
    native_root: Path, curated_root: Path, output: Path, *, expected_snapshot: Path, workers: int = 8,
    max_in_flight: int | None = None, overwrite: bool = False, no_progress: bool = False,
) -> dict[str, Any]:
    """
    Run formal CWP6 full-cohort acceptance. The function always enables native and curated SHA-256 verification
    through the CWP5 product audit. ``expected_snapshot`` is mandatory so that a formal release is tied to an
    explicit cumulative native snapshot rather than to hard-coded source constants.
    """

    native, curated, report_root = _guard_roots(native_root, curated_root, output)
    expected_path = Path(expected_snapshot).expanduser().resolve()
    if not expected_path.is_file():
        raise InputLayoutError(f"expected snapshot manifest does not exist: {expected_path}")
    expected_payload = _json_read(expected_path)

    native_state_path = native / NATIVE_STATE_FILENAME
    curated_state_path = curated / CURATION_STATE_FILENAME
    native_state_before = sha256_file(native_state_path)
    curated_state_before = sha256_file(curated_state_path)
    started_at = _now()
    started = time.perf_counter()

    stage_parent = report_root.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{report_root.name}.stage-", dir=stage_parent))
    try:
        product_dir = stage / "product_acceptance"
        product = run_acceptance(
            native,
            curated,
            product_dir,
            workers=workers,
            max_in_flight=max_in_flight,
            reference_manifest=None,
            verify_hashes=True,
            overwrite=False,
            no_progress=no_progress,
        )
        native_snapshot = read_native_snapshot(native_state_path)
        history = _curation_run_history(curated_state_path)
        latest_run = history[-1] if history else None
        build_runs = [
            run for run in history
            if run["status"] == "complete"
            and run["participants"]["committed"] > 0
            and run["participants"]["failed"] == 0
            and run["verify_native_hashes"]
        ]
        latest_build = build_runs[-1] if build_runs else None
        current_participants = int(product["snapshot"]["participants"])
        latest_noop = bool(
            latest_run
            and latest_run["status"] == "complete"
            and latest_run["participants"] == {
                "discovered": current_participants,
                "skipped": current_participants,
                "committed": 0,
                "failed": 0,
            }
            and latest_run["final_status_counts"] == {"skipped": current_participants}
            and latest_run["verify_native_hashes"]
            and latest_run["verify_existing_hashes"]
        )
        noop_after_build = bool(
            latest_noop
            and latest_build
            and str(latest_run["started_at"]) >= str(latest_build["started_at"])
        )

        actual_expected_view = _expected_view(native_snapshot, product["fingerprints"])
        expected_issues = _compare_expected(expected_payload, actual_expected_view)
        participant_statuses = set(native_snapshot["participant_status_counts"])
        accepted_statuses = {"complete", "complete_empty"}
        complete_matches_files = (
            native_snapshot["complete_participants"]
            == native_snapshot["participants_with_feature_files"]
            == product["snapshot"]["participants"]
        )
        planner_noop = product["planner_action_counts"] == {"skip": current_participants}
        no_activities = not native_snapshot["activities_feature_present"]

        checks = [
            _check(
                "cwp5_product_acceptance",
                product["status"] == "pass",
                f"CWP5 product acceptance status: {product['status']}",
            ),
            _check(
                "expected_snapshot_reference_id",
                bool(expected_payload.get("reference_id")),
                f"expected snapshot reference ID: {expected_payload.get('reference_id')!r}",
            ),
            _check(
                "expected_snapshot_match",
                not expected_issues,
                f"expected snapshot mismatches: {len(expected_issues)}",
            ),
            _check(
                "native_participant_statuses",
                participant_statuses <= accepted_statuses,
                f"native participant statuses: {native_snapshot['participant_status_counts']}",
            ),
            _check(
                "native_participant_accounting",
                native_snapshot["participant_records"]
                == native_snapshot["complete_participants"] + native_snapshot["complete_empty_participants"],
                "complete + complete_empty equals all native participant records",
            ),
            _check(
                "curated_participants_match_native_complete",
                complete_matches_files,
                "curated participants equal native complete participants with feature files",
            ),
            _check(
                "native_feature_coverage",
                native_snapshot["feature_files"] == product["snapshot"]["feature_files"],
                f"native={native_snapshot['feature_files']}, curated={product['snapshot']['feature_files']}",
            ),
            _check(
                "no_activities_output",
                no_activities,
                "native feature set contains no Activities output",
            ),
            _check(
                "successful_build_run_exists",
                latest_build is not None,
                "at least one complete hash-verified run committed participants",
            ),
            _check(
                "latest_run_hash_verified_noop",
                latest_noop,
                "latest curation run skipped every current participant with both hash checks enabled",
            ),
            _check(
                "noop_follows_build",
                noop_after_build,
                "latest hash-verified no-op occurred after a successful build run",
            ),
            _check(
                "current_policy_planner_noop",
                planner_noop,
                f"current planner actions: {product['planner_action_counts']}",
            ),
            _check(
                "native_state_quiescent",
                native_snapshot["active_processing_runs"] == 0,
                f"active native runs: {native_snapshot['active_processing_runs']}",
            ),
            _check(
                "row_parity",
                product["snapshot"]["native_rows"] == product["snapshot"]["curated_rows"],
                "native and curated row totals are equal",
            ),
            _check(
                "status_accounting",
                bool(product["snapshot"]["status_accounting_complete"]),
                "pass + review + exclude_default equals curated rows",
            ),
            _check(
                "default_inclusion_accounting",
                bool(product["snapshot"]["default_inclusion_accounting_complete"]),
                "included + excluded equals curated rows",
            ),
            _check(
                "hash_verification",
                next(
                    (item["status"] == "pass" for item in product["checks"] if item["check_id"] == "hash_verification"),
                    False,
                ),
                "CWP5 verified native and curated SHA-256 values",
            ),
        ]

        native_state_after = sha256_file(native_state_path)
        curated_state_after = sha256_file(curated_state_path)
        state_unchanged = (
            native_state_before == native_state_after
            and curated_state_before == curated_state_after
        )
        checks.append(_check(
            "state_databases_unchanged",
            state_unchanged,
            "native and curated state database hashes are unchanged",
        ))

        failed = [item for item in checks if item["status"] == "fail"]
        status = "pass" if not failed else "fail"
        run_evidence = {
            "run_count": len(history),
            "successful_build_run_count": len(build_runs),
            "latest_successful_build_run": latest_build,
            "latest_run": latest_run,
        }
        release_manifest = {
            "cohort_release_schema_version": COHORT_REPORT_SCHEMA_VERSION,
            "source_key": SOURCE_KEY,
            "expected_snapshot_reference_id": expected_payload.get("reference_id"),
            "native_snapshot": native_snapshot,
            "curated_snapshot": product["snapshot"],
            "fingerprints": product["fingerprints"],
            "native_csv_tree_sha256": product["native_csv_tree_sha256"],
            "curated_csv_tree_sha256": product["curated_csv_tree_sha256"],
            "native_state_semantic_sha256": product["native_state_semantic_sha256"],
            "curated_state_semantic_sha256": product["curated_state_semantic_sha256"],
            "latest_hash_verified_noop_run_id": latest_run["run_id"] if latest_noop else None,
        }
        payload = {
            "tool": "google_fitness_curation_cohort_acceptance",
            "source_key": SOURCE_KEY,
            "cohort_acceptance_version": COHORT_ACCEPTANCE_VERSION,
            "report_schema_version": COHORT_REPORT_SCHEMA_VERSION,
            "status": status,
            "check_count": len(checks),
            "passed_check_count": sum(item["status"] == "pass" for item in checks),
            "failed_check_count": sum(item["status"] == "fail" for item in checks),
            "checks": checks,
            "expected_snapshot": {
                "path": str(expected_path),
                "reference_id": expected_payload.get("reference_id"),
                "mismatch_count": len(expected_issues),
                "mismatches": expected_issues,
            },
            "native_snapshot": native_snapshot,
            "product_acceptance": {
                "status": product["status"],
                "acceptance_version": product["acceptance_version"],
                "check_count": product["check_count"],
                "passed_check_count": product["passed_check_count"],
                "failed_check_count": product["failed_check_count"],
                "snapshot": product["snapshot"],
                "planner_action_counts": product["planner_action_counts"],
                "fingerprints": product["fingerprints"],
                "native_csv_tree_sha256": product["native_csv_tree_sha256"],
                "curated_csv_tree_sha256": product["curated_csv_tree_sha256"],
            },
            "run_evidence": run_evidence,
            "cohort_release_manifest": release_manifest,
        }

        history_rows = []
        for run in history:
            history_rows.append({
                "run_id": run["run_id"],
                "started_at": run["started_at"],
                "finished_at": run["finished_at"] or "",
                "status": run["status"],
                "discovered": run["participants"]["discovered"],
                "skipped": run["participants"]["skipped"],
                "committed": run["participants"]["committed"],
                "failed": run["participants"]["failed"],
                "verify_native_hashes": int(run["verify_native_hashes"]),
                "verify_existing_hashes": int(run["verify_existing_hashes"]),
                "final_status_counts": json.dumps(run["final_status_counts"], sort_keys=True),
            })

        _json_write(stage / "cwp6_acceptance.json", payload)
        _json_write(stage / "native_cohort_snapshot.json", native_snapshot)
        _json_write(stage / "expected_snapshot_comparison.json", {
            "reference_id": expected_payload.get("reference_id"),
            "mismatches": expected_issues,
        })
        _json_write(stage / "cohort_release_manifest.json", release_manifest)
        _csv_write(stage / "curation_run_history.csv", history_rows, (
            "run_id", "started_at", "finished_at", "status", "discovered", "skipped",
            "committed", "failed", "verify_native_hashes", "verify_existing_hashes",
            "final_status_counts",
        ))
        (stage / "REPORT.md").write_text(_render(payload), encoding="utf-8")
        _json_write(stage / "run_manifest.json", {
            "tool": "google_fitness_curation_cohort_acceptance",
            "cohort_acceptance_version": COHORT_ACCEPTANCE_VERSION,
            "started_at": started_at,
            "finished_at": _now(),
            "native_root": str(native),
            "curated_root": str(curated),
            "output": str(report_root),
            "expected_snapshot": str(expected_path),
            "workers_requested": int(workers),
            "max_in_flight": int(max_in_flight or workers),
            "wall_clock_seconds": round(time.perf_counter() - started, 6),
            "native_state_sha256_before": native_state_before,
            "native_state_sha256_after": native_state_after,
            "curated_state_sha256_before": curated_state_before,
            "curated_state_sha256_after": curated_state_after,
        })
        _publish(stage, report_root, overwrite)
        return payload
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run formal full-cohort Google Fitness native-to-curated acceptance."
    )
    parser.add_argument("--input-native", type=Path, required=True)
    parser.add_argument("--input-curated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-snapshot", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--json-summary", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_cohort_acceptance(
            args.input_native,
            args.input_curated,
            args.output,
            expected_snapshot=args.expected_snapshot,
            workers=args.workers,
            max_in_flight=args.max_in_flight,
            overwrite=args.overwrite,
            no_progress=args.no_progress,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.json_summary:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        snapshot = result["product_acceptance"]["snapshot"]
        print(
            f"Google Fitness CWP6 cohort acceptance {result['status'].upper()}: "
            f"{result['passed_check_count']}/{result['check_count']} checks passed; "
            f"{snapshot['feature_files']} files; {snapshot['curated_rows']} rows"
        )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
