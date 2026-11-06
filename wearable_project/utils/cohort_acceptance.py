"""
Wearable Data Processing and Modeling project
Cohort acceptance check for one source's DataLoaders, run on the real native and curated roots. The loader test
suites run on a representative sample. This script checks what only the full cohort can show, without loading all
of it:
1. Profiles. ``profile()`` for every feature in both phases. It reads only the state databases and one ``stat``
   per file, so the whole cohort is covered in minutes. Unrecorded files and size mismatches fail; recorded files
   missing from disk are reported. The two phases must agree on each participant's rows.
2. Inventory closure. Every state-recorded feature and every participant CSV is classified as a public feature, a
   known non-public source product, or an unexpected product. State/disk pairs must close exactly. Optional
   ``--verify-all-hashes`` checks every committed output's size and SHA-256 without parsing every cohort row.
3. Verified loads. A deterministic, feature-specific participant sample is selected independently for every
   feature and phase. Every observed feature is therefore materially loaded with ``state_validation="required"``;
   rare features cannot disappear behind a cohort-wide sample. Dtypes, profiles, row counts and helpers are checked.
4. Sharing. The data owner can read every file whatever its mode, so the owner's own run passes the checks above
   even where colleagues are refused. The permission bits are checked instead: the files of the checked features,
   every participant folder, each root and its state database must be open to their group. Anything that is not
   is reported, with the command that fixes it. ``--required-group`` additionally verifies the owning group.
Usage:

    python -m wearable_project.utils.cohort_acceptance \
        --source apple_healthkit --participants 50 --seed 1 --out ~/acceptance/apple

    python -m wearable_project.utils.cohort_acceptance \
        --source google_fitness --participants 50 --seed 1 --out ~/acceptance/google

The roots default to the selected source's permanent HPP roots; ``--native`` and ``--curated`` point elsewhere.
Omitting ``--source`` remains backward-compatible and selects Apple HealthKit. Reports are written as
``cohort_acceptance.json`` and ``cohort_acceptance.txt`` in ``--out``, which must lie outside every selected data
root. The exit status is 0 when the selected gate passes, 1 when it fails, and 2 for invalid arguments.
"""


from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import os
import sqlite3
import stat
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from wearable_project import __release_label__, __version__
from wearable_project.DataLoaders import available_features
from wearable_project.DataLoaders._contracts import participant_component_issue
from wearable_project.DataLoaders._base import _PHASE_STATE_FILES, AppleHealthFeatureLoader, _account_name
from wearable_project.DataLoaders._sources import (
    DEFAULT_APPLE_HEALTHKIT_CURATED_ROOT, DEFAULT_APPLE_HEALTHKIT_NATIVE_ROOT, DEFAULT_GOOGLE_FITNESS_CURATED_ROOT,
    DEFAULT_GOOGLE_FITNESS_NATIVE_ROOT, WearableSource, get_source_adapter, normalize_source, probe_root_identity,
)
from wearable_project.exceptions import DataLoaderError


PHASES = ("native", "curated")
COHORT_ACCEPTANCE_VERSION = "dataloader-cohort-acceptance-3"
EARLIEST_PLAUSIBLE = pd.Timestamp("2000-01-01", tz="UTC")
OFFSET_RANGE = (-720, 840)  # UTC-12:00 to UTC+14:00, in minutes
DETERMINISM_PROBES = 3
DEFAULT_HASH_WORKERS = min(16, max(1, os.cpu_count() or 1))
DEFAULT_SOURCE = WearableSource.APPLE_HEALTHKIT
_ALL_DEFAULT_ROOTS = (
    DEFAULT_APPLE_HEALTHKIT_NATIVE_ROOT,
    DEFAULT_APPLE_HEALTHKIT_CURATED_ROOT,
    DEFAULT_GOOGLE_FITNESS_NATIVE_ROOT,
    DEFAULT_GOOGLE_FITNESS_CURATED_ROOT,
)


class Findings:
    """Every FAIL and WARN, with where it was found."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, level: str, phase: str | None, feature: str | None, check: str, detail: str) -> None:
        self.items.append({"level": level, "phase": phase, "feature": feature, "check": check, "detail": detail})

    def count(self, level: str) -> int:
        return sum(1 for item in self.items if item["level"] == level)


def _loader(
    feature: str, phase: str, root: Path, source: WearableSource | str = DEFAULT_SOURCE, **kwargs,
) -> AppleHealthFeatureLoader:
    module = __import__(f"wearable_project.DataLoaders.{feature}Loader", fromlist=[f"{feature}Loader"])
    return getattr(module, f"{feature}Loader")(
        phase=phase,
        root=root,
        source=source,
        **kwargs,
    )


def _source_argument(value: str) -> WearableSource:
    try:
        return normalize_source(value)
    except DataLoaderError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _required_group_argument(value: str) -> tuple[int, str]:
    """Resolve a group name or numeric gid for deployment-sharing checks."""

    try:
        import grp
    except ImportError as exc:  # pragma: no cover - the HPP deployment is POSIX
        raise argparse.ArgumentTypeError("--required-group is available only on POSIX systems") from exc
    token = str(value).strip()
    if not token:
        raise argparse.ArgumentTypeError("required group must not be blank")
    if token.isdigit():
        gid = int(token)
        try:
            group = grp.getgrgid(gid)
        except KeyError:
            # Numeric ownership remains verifiable even where the local NSS cannot resolve the name.
            return gid, token
        return int(group.gr_gid), str(group.gr_name)
    try:
        group = grp.getgrnam(token)
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unknown group {value!r}") from exc
    return int(group.gr_gid), str(group.gr_name)


def _protected_roots(*roots: str | Path | None) -> tuple[Path, ...]:
    """Known permanent and selected roots into which acceptance reports must never be written."""

    candidates = [*_ALL_DEFAULT_ROOTS, *(root for root in roots if root is not None)]
    return tuple(sorted({Path(root).expanduser().resolve() for root in candidates}))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _same_dtype(a: Any, b: Any) -> bool:
    return (isinstance(a, pd.CategoricalDtype) and isinstance(b, pd.CategoricalDtype)) or a == b


# ----------------------------------------------------------------------------------------------- inventory
def _stable_sample(
    participants: list[str] | tuple[str, ...] | set[str], count: int, *, seed: int, source: WearableSource,
    feature: str,
) -> list[str]:
    """Choose a process-independent deterministic sample for one source-qualified feature."""

    values = sorted(set(map(str, participants)))
    if count <= 0 or count >= len(values):
        return values

    def rank(participant: str) -> tuple[bytes, str]:
        token = f"{seed}\0{source.value}\0{feature}\0{participant}".encode("utf-8")
        return hashlib.sha256(token).digest(), participant

    return sorted(sorted(values, key=rank)[:count])


def select_feature_samples(
    features, rows_by_participant: dict[str, dict[str, dict[str, int | None]]], participants: int, seed: int,
    source: WearableSource | str = DEFAULT_SOURCE,
) -> dict[str, dict[str, list[str]]]:
    """Select up to ``participants`` from each feature's own participants in each phase."""

    canonical = normalize_source(source)
    return {
        feature: {
            phase: _stable_sample(
                set(rows_by_participant[phase].get(feature, {})),
                participants,
                seed=seed,
                source=canonical,
                feature=feature,
            )
            for phase in PHASES
        }
        for feature in features
    }


def _empty_inventory_summary(source: WearableSource, error: str) -> dict[str, Any]:
    """Stable report shape when managed inventory cannot be read safely."""

    return {
        "state_rows": 0,
        "state_features": [],
        "public_features_present": [],
        "public_features_absent": list(available_features(source=source)),
        "known_non_public_features_present": [],
        "unexpected_features_present": [],
        "disk_csv_files": 0,
        "unrecorded_disk_files": 0,
        "state_files_missing": 0,
        "filename_mismatches": 0,
        "error": error,
    }


def _state_inventory(
    root: Path, phase: str, findings: Findings, *, source: WearableSource,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read and close one managed state/disk inventory without parsing a feature CSV."""

    adapter = get_source_adapter(source)
    identity = probe_root_identity(root, phase)
    try:
        adapter.validate_root_identity(identity, phase)
        adapter.validate_root_contract(root, phase)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        findings.add("FAIL", phase, None, "managed-root inventory", detail)
        return _empty_inventory_summary(source, detail), []

    database = root / _PHASE_STATE_FILES[phase]
    if not database.is_file():
        detail = f"state database is absent: {database}"
        findings.add("FAIL", phase, None, "managed-root inventory", detail)
        return _empty_inventory_summary(source, detail), []

    active = AppleHealthFeatureLoader._active_write_file(database)
    if active is not None:
        detail = f"active state sidecar: {active}"
        findings.add("FAIL", phase, None, "managed-root inventory", detail)
        return _empty_inventory_summary(source, detail), []

    contract = adapter.state_manifest_contract(phase)
    uri = f"{database.resolve().as_uri()}?immutable=1"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            AppleHealthFeatureLoader._validate_state_manifest_columns(connection, database, contract)
            quoted = contract.table.replace('"', '""')
            rows = connection.execute(
                f'SELECT * FROM "{quoted}" ORDER BY feature, participant_id'
            ).fetchall()
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        findings.add("FAIL", phase, None, "managed-root inventory", detail)
        return _empty_inventory_summary(source, detail), []

    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    invalid_feature_rows = 0
    for row in rows:
        raw = row["feature"]
        raw_text = "" if raw is None else str(raw)
        feature = raw_text.strip()
        issue = participant_component_issue(feature)
        if raw_text != feature or issue is not None:
            invalid_feature_rows += 1
            detail = (
                f"feature {raw!r} has leading/trailing whitespace"
                if raw_text != feature
                else f"feature {raw!r} {issue}"
            )
            findings.add("FAIL", phase, None, "invalid state feature", detail)
            continue
        grouped[feature].append(row)

    entries: list[dict[str, Any]] = []
    filename_mismatches: list[str] = []
    for feature, feature_rows in sorted(grouped.items()):
        try:
            manifest = AppleHealthFeatureLoader._manifest_from_rows(feature_rows, database, contract)
        except Exception as exc:
            findings.add("FAIL", phase, feature, "state inventory", f"{type(exc).__name__}: {exc}")
            continue
        for participant_id, entry in sorted(manifest.items()):
            expected_filename = f"{feature}.csv"
            raw_filename = entry.get("filename", expected_filename)
            raw_text = "" if raw_filename is None else str(raw_filename)
            filename = raw_text.strip()
            issue = participant_component_issue(filename)
            if raw_text != filename or issue is not None or filename != expected_filename:
                filename_mismatches.append(
                    f"{participant_id}/{raw_filename!s} (feature {feature!r}, expected {expected_filename!r})"
                )
                # Keep the expected safe path for disk closure; never follow a state-controlled unsafe path.
                filename = expected_filename
            entries.append({
                "participant_id": participant_id,
                "feature": feature,
                "filename": filename,
                "row_count": int(entry[contract.row_count_column]),
                "size_bytes": int(entry["size_bytes"]),
                "sha256": str(entry["sha256"]),
                "path": root / participant_id / filename,
            })
    if filename_mismatches:
        findings.add(
            "FAIL",
            phase,
            None,
            "state filename does not match feature",
            f"{len(filename_mismatches)}: {filename_mismatches[:5]}",
        )

    state_pairs = {(entry["participant_id"], entry["feature"]): entry for entry in entries}
    disk_pairs: dict[tuple[str, str], Path] = {}
    try:
        participant_folders = sorted(
            path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
        )
    except OSError as exc:
        findings.add("FAIL", phase, None, "disk inventory", f"could not list {root}: {exc}")
        participant_folders = []
    for folder in participant_folders:
        raw_participant = folder.name
        participant_id = raw_participant.strip()
        issue = participant_component_issue(participant_id)
        if raw_participant != participant_id or issue is not None:
            findings.add(
                "FAIL",
                phase,
                None,
                "invalid participant directory",
                f"{raw_participant!r}: {'leading/trailing whitespace' if raw_participant != participant_id else issue}",
            )
            continue
        try:
            files = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".csv")
        except OSError as exc:
            findings.add("FAIL", phase, None, "disk inventory", f"could not list {folder}: {exc}")
            continue
        for path in files:
            pair = (participant_id, path.stem)
            if pair in disk_pairs:
                findings.add("FAIL", phase, path.stem, "duplicate disk product", str(path))
            disk_pairs[pair] = path

    unrecorded = sorted(set(disk_pairs) - set(state_pairs))
    missing = sorted(set(state_pairs) - set(disk_pairs))
    if unrecorded:
        examples = [f"{participant}/{feature}.csv" for participant, feature in unrecorded[:5]]
        findings.add(
            "FAIL", phase, None, "unrecorded product files",
            f"{len(unrecorded)} participant-feature file(s), e.g. {examples}",
        )
    if missing:
        examples = [f"{participant}/{feature}.csv" for participant, feature in missing[:5]]
        findings.add(
            "WARN", phase, None, "state-recorded product files missing from disk",
            f"{len(missing)} participant-feature file(s), e.g. {examples}",
        )

    public = set(available_features(source=source))
    known_contracts = {item.feature: item for item in adapter.non_public_output_contracts(phase)}
    state_features = set(grouped)
    known_present = sorted(state_features & set(known_contracts))
    unexpected = sorted(state_features - public - set(known_contracts))
    for feature in known_present:
        contract_item = known_contracts[feature]
        count = sum(1 for entry in entries if entry["feature"] == feature)
        findings.add(
            "WARN",
            phase,
            feature,
            f"known non-public output: {contract_item.classification}",
            f"{count} file(s); {contract_item.detail}",
        )
    if unexpected:
        findings.add(
            "FAIL",
            phase,
            None,
            "unexpected state feature products",
            f"{len(unexpected)}: {unexpected}",
        )

    known_summary = [
        {
            "feature": feature,
            "classification": known_contracts[feature].classification,
            "detail": known_contracts[feature].detail,
            "files": sum(1 for entry in entries if entry["feature"] == feature),
            "rows": sum(entry["row_count"] for entry in entries if entry["feature"] == feature),
        }
        for feature in known_present
    ]
    summary = {
        "state_database": str(database),
        "state_table": contract.table,
        "state_rows": len(entries),
        "invalid_feature_rows": invalid_feature_rows,
        "state_features": sorted(state_features),
        "public_features_present": sorted(state_features & public),
        "public_features_absent": sorted(public - state_features),
        "known_non_public_features_present": known_summary,
        "unexpected_features_present": unexpected,
        "disk_csv_files": len(disk_pairs),
        "unrecorded_disk_files": len(unrecorded),
        "state_files_missing": len(missing),
        "filename_mismatches": len(filename_mismatches),
    }
    return summary, entries


def check_inventory(roots, findings: Findings, source: WearableSource | str = DEFAULT_SOURCE,) -> dict[str, Any]:
    """Classify all managed outputs and close every state row against the participant tree."""

    canonical = normalize_source(source)
    summaries: dict[str, Any] = {}
    entries: dict[str, list[dict[str, Any]]] = {}
    for phase in PHASES:
        summaries[phase], entries[phase] = _state_inventory(
            roots[phase], phase, findings, source=canonical,
        )
    return {"summary": summaries, "entries": entries}


def _hash_one(entry: dict[str, Any]) -> dict[str, Any]:
    path = Path(entry["path"])
    result = {
        "participant_id": entry["participant_id"],
        "feature": entry["feature"],
        "path": str(path),
        "expected_size": entry["size_bytes"],
        "expected_sha256": entry["sha256"],
        "status": "verified",
    }
    try:
        size = path.stat().st_size
    except OSError as exc:
        result.update(status="missing_or_unreadable", error=f"{type(exc).__name__}: {exc}")
        return result
    result["actual_size"] = int(size)
    if int(size) != int(entry["size_bytes"]):
        result["status"] = "size_mismatch"
        return result
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        result.update(status="missing_or_unreadable", error=f"{type(exc).__name__}: {exc}")
        return result
    result["actual_sha256"] = digest.hexdigest()
    if result["actual_sha256"] != entry["sha256"]:
        result["status"] = "sha256_mismatch"
    return result


def verify_inventory_hashes(
    entries_by_phase: dict[str, list[dict[str, Any]]], findings: Findings, *, workers: int,
) -> dict[str, Any]:
    """Verify every state-recorded output's size and SHA-256 with bounded I/O concurrency."""

    summary: dict[str, Any] = {}
    for phase in PHASES:
        entries = sorted(
            entries_by_phase.get(phase, ()),
            key=lambda item: (item["feature"], item["participant_id"]),
        )
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"accept-{phase}") as executor:
            results = list(executor.map(_hash_one, entries))
        counts = Counter(result["status"] for result in results)
        failures = [result for result in results if result["status"] != "verified"]
        grouped_failures: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for result in failures:
            grouped_failures[(result["feature"], result["status"])].append(result)
        for (feature, status), items in sorted(grouped_failures.items()):
            examples = [item["participant_id"] for item in items[:5]]
            findings.add(
                "FAIL",
                phase,
                feature,
                "exhaustive output hash verification",
                f"{len(items)} file(s) with {status}, e.g. participants {examples}",
            )
        summary[phase] = {
            "enabled": True,
            "workers": workers,
            "files_recorded": len(entries),
            "files_verified": int(counts.get("verified", 0)),
            "bytes_verified": int(sum(
                result.get("actual_size", 0) for result in results if result["status"] == "verified"
            )),
            "status_counts": dict(sorted(counts.items())),
            "failures": failures[:100],
            "failure_count": len(failures),
            "seconds": round(time.perf_counter() - started, 3),
        }
    return summary


# ------------------------------------------------------------------------------------------------ profiles
def check_profiles(
    features, roots, findings: Findings, source: WearableSource | str = DEFAULT_SOURCE,
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {phase: {} for phase in PHASES}
    rows_by_participant: dict[str, dict[str, dict[str, int]]] = {phase: {} for phase in PHASES}
    for phase in PHASES:
        for feature in features:
            started = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    report = _loader(feature, phase, roots[phase], source=source).profile()
            except Exception as exc:  # recorded, never raised: one feature must not stop the run
                findings.add("FAIL", phase, feature, "profile", f"{type(exc).__name__}: {exc}")
                continue
            try:
                profiles[phase][feature] = _summarize_profile(
                    report, phase, feature, rows_by_participant, findings, caught, started)
            except Exception as exc:  # an unexpected shape is a finding, never a crash
                findings.add("FAIL", phase, feature, "profile", f"could not interpret the profile: {type(exc).__name__}: {exc}")
    _compare_phases(features, rows_by_participant, findings)
    return {"profiles": profiles, "rows_by_participant": rows_by_participant}


def _summarize_profile(report, phase, feature, rows_by_participant, findings, caught, started) -> dict[str, Any]:
    """Turn one profile into its findings and a summary; row counts a file has no state for stay unknown."""

    table = report.participants
    unrecorded = table.index[table["on_disk"].astype(bool) & ~table["has_state"].astype(bool)].tolist()
    missing = table.index[~table["on_disk"].astype(bool) & table["has_state"].astype(bool)].tolist()
    mismatched = table.index[table["on_disk"].astype(bool) & table["has_state"].astype(bool)
                             & ~table["size_matches"].fillna(True).astype(bool)].tolist()
    if unrecorded:
        findings.add("FAIL", phase, feature, "unrecorded files", f"{len(unrecorded)}: {unrecorded[:5]}")
    if mismatched:
        findings.add("FAIL", phase, feature, "size differs from the state database", f"{len(mismatched)}: {mismatched[:5]}")
    if missing:
        findings.add("WARN", phase, feature, "recorded files missing from disk", f"{len(missing)}: {missing[:5]}")
    for warning in caught:
        findings.add("WARN", phase, feature, "profile warning", str(warning.message)[:300])
    present = table[table["on_disk"].astype(bool)]
    # A file with no state row has no recorded row count; it is already reported as unrecorded.
    rows_by_participant[phase][feature] = {
        str(code): (int(n) if pd.notna(n) else None) for code, n in present["rows"].items()
    }
    return {
        "participants": int(len(present)), "rows": int(pd.to_numeric(present["rows"]).fillna(0).sum()),
        "rows_unknown": int(present["rows"].isna().sum()),
        "unrecorded": len(unrecorded), "missing_from_disk": len(missing), "size_mismatches": len(mismatched),
        "seconds": round(time.perf_counter() - started, 3), "summary": _jsonable(report.summary),
    }


def _compare_phases(features, rows_by_participant, findings: Findings) -> None:
    for feature in features:
        native, curated = rows_by_participant["native"].get(feature), rows_by_participant["curated"].get(feature)
        if native is None or curated is None:
            continue
        only_curated = sorted(set(curated) - set(native))
        not_curated = sorted(set(native) - set(curated))
        differing = sorted(code for code in set(native) & set(curated)
                           if native[code] is not None and curated[code] is not None and native[code] != curated[code])
        if only_curated:
            findings.add("FAIL", None, feature, "curated files without native files", f"{len(only_curated)}: {only_curated[:5]}")
        if differing:
            findings.add("FAIL", None, feature, "curated rows differ from native rows", f"{len(differing)}: {differing[:5]}")
        if not_curated:
            findings.add("WARN", None, feature, "native files not yet curated", f"{len(not_curated)}: {not_curated[:5]}")


# --------------------------------------------------------------------------------------------------- loads
def check_load(
    feature, phase, root, codes, expected_rows_by_participant, findings: Findings,
    source: WearableSource | str = DEFAULT_SOURCE,
) -> dict[str, Any]:
    started = time.perf_counter()
    loader = _loader(feature, phase, root, source=source, state_validation="required")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = loader.get_data(registration_codes=codes, projection="analysis")
    except Exception as exc:
        findings.add("FAIL", phase, feature, "verified load", f"{type(exc).__name__}: {exc}")
        return {"error": f"{type(exc).__name__}: {exc}"}
    seconds = time.perf_counter() - started
    for warning in caught:
        findings.add("WARN", phase, feature, "load warning", str(warning.message)[:300])
    df, report = data.df, data.load_report
    expected_known = {
        code: expected_rows_by_participant.get(code)
        for code in codes
        if expected_rows_by_participant.get(code) is not None
    }
    expected_rows = int(sum(expected_known.values()))
    result: dict[str, Any] = {
        "rows": int(len(df)),
        "participants": int(report["participants_loaded"]),
        "participants_requested": list(codes),
        "expected_rows_from_profile": expected_rows,
        "expected_rows_complete": len(expected_known) == len(codes),
        "observed_feature": bool(expected_rows_by_participant),
        "seconds": round(seconds, 3),
        "rows_per_second": int(len(df) / seconds) if seconds else None,
    }
    if expected_known and len(expected_known) == len(codes) and int(len(df)) != expected_rows:
        findings.add(
            "FAIL",
            phase,
            feature,
            "sampled rows differ from profile/state",
            f"loaded {len(df):,} row(s), expected {expected_rows:,} for {list(codes)!r}",
        )
    if expected_rows > 0 and len(df) == 0:
        findings.add(
            "FAIL",
            phase,
            feature,
            "observed feature was not materially loaded",
            f"the selected participants have {expected_rows:,} recorded row(s), but the load returned none",
        )

    date = df.index.get_level_values("Date").dtype
    if list(df.index.names) != ["RegistrationCode", "Date"] or not (isinstance(date, pd.DatetimeTZDtype) and str(date.tz) == "UTC"):
        findings.add("FAIL", phase, feature, "index", f"names {list(df.index.names)}, Date dtype {date}")
    if report["state_verified"] is not True:
        findings.add("FAIL", phase, feature, "verification", "the load was not verified against the state database")

    contract = loader._source_adapter.feature_contract(feature, phase)
    projection_schema = loader._source_adapter.projection_schema(feature, phase)
    categorical = contract.categorical_columns
    for column in df.columns:
        dtype = df[column].dtype
        if column in categorical and not isinstance(dtype, pd.CategoricalDtype):
            findings.add("FAIL", phase, feature, "declared categorical column", f"{column} is {dtype}")
        if column in contract.boolean_columns and str(dtype) != "boolean":
            findings.add("FAIL", phase, feature, "declared boolean column", f"{column} is {dtype}")
        if column in contract.integer_columns and str(dtype) != "Int64":
            fractional = int((df[column].dropna() % 1 != 0).sum()) if pd.api.types.is_numeric_dtype(dtype) else None
            findings.add("WARN", phase, feature, "whole-number column holds fractions",
                         f"{column} is {dtype}; {fractional} non-whole value(s); its dtype now depends on the participants read")
    undeclared = projection_schema.undeclared_columns(report["columns_available"])
    if undeclared:
        findings.add("WARN", phase, feature, "columns the representative sample never showed", ", ".join(map(str, undeclared)))
    result["undeclared_columns"] = list(map(str, undeclared))

    present = list(dict.fromkeys(df.index.get_level_values("RegistrationCode").astype(str)))
    for code in present[:DETERMINISM_PROBES]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                alone = loader.get_data(registration_codes=code, projection="analysis").df
        except Exception as exc:
            findings.add("FAIL", phase, feature, "single-participant load", f"{code}: {type(exc).__name__}: {exc}")
            continue
        for column in alone.columns:
            if column in df.columns and not _same_dtype(alone[column].dtype, df[column].dtype):
                findings.add("FAIL", phase, feature, "dtype depends on the participants read",
                             f"{column}: {alone[column].dtype} for {code}, {df[column].dtype} for the set")

    try:
        if loader.date_column == "start_date" and "utc_offset_minutes" in df.columns:
            data.with_local_time()
        if "value" in df.columns and "value" not in categorical:
            sources = data.with_harmonized_values().df["harmonized_unit_source"].astype(str)
            result["harmonized_sources"] = dict(Counter(sources))
    except DataLoaderError as exc:
        findings.add("FAIL", phase, feature, "helpers", f"{type(exc).__name__}: {exc}")

    anomalies: dict[str, int] = {}
    if {"start_date", "end_date"} <= set(df.columns):
        anomalies["end_before_start"] = int((df["end_date"] < df["start_date"]).sum())
    dates = df.index.get_level_values("Date")
    anomalies["dates_before_2000"] = int((dates < EARLIEST_PLAUSIBLE).sum())
    anomalies["dates_in_the_future"] = int((dates > pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).sum())
    if "utc_offset_minutes" in df.columns:
        offsets = df["utc_offset_minutes"].dropna()
        anomalies["offsets_out_of_range"] = int(((offsets < OFFSET_RANGE[0]) | (offsets > OFFSET_RANGE[1])).sum())
    for name, count in anomalies.items():
        if count:
            findings.add("WARN", phase, feature, f"plausibility: {name.replace('_', ' ')}", f"{count} row(s)")
    result["anomalies"] = anomalies
    return result


def check_loads(
    features, roots, samples_by_feature, rows_by_participant, findings: Findings,
    source: WearableSource | str = DEFAULT_SOURCE,
) -> dict[str, Any]:
    loads: dict[str, dict[str, Any]] = {phase: {} for phase in PHASES}
    for phase in PHASES:
        for feature in features:
            codes = samples_by_feature.get(feature, {}).get(phase, [])
            loads[phase][feature] = check_load(
                feature,
                phase,
                roots[phase],
                codes,
                rows_by_participant[phase].get(feature, {}),
                findings,
                source=source,
            )
    return loads


# -------------------------------------------------------------------------------------------------- report
def check_sharing(
    features, roots, findings: Findings, source: WearableSource | str = DEFAULT_SOURCE,
    required_group: tuple[int, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Whether each root is open to its group: files readable, folders readable and searchable, so that colleagues in
    the group can list and read what the loaders read. Checked from the permission bits, since the data owner can
    read everything whatever the bits say.
    """

    summary: dict[str, dict[str, Any]] = {}
    adapter = get_source_adapter(source)
    for phase in PHASES:
        root = roots[phase]
        fix = f"the data owner can open the root to its group with: chmod -R g+rX {root}"
        names = {f"{feature}.csv": feature for feature in features}
        checked: Counter[str] = Counter()
        closed_files: dict[str, list[str]] = {feature: [] for feature in features}
        groups: set[int] = set()
        wrong_group_paths: set[str] = set()

        def record_group(path: Path, info: os.stat_result | None = None) -> None:
            try:
                stat_result = info or path.stat()
            except OSError:
                return
            groups.add(stat_result.st_gid)
            if required_group is not None and stat_result.st_gid != required_group[0]:
                try:
                    label = str(path.relative_to(root)) or "."
                except ValueError:
                    label = str(path)
                wrong_group_paths.add(label)

        try:
            closed_folders = [] if _open_to_group(root, folder=True) else [str(root)]
            record_group(root)
            folders = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError as exc:
            findings.add("FAIL", phase, None, "sharing", f"could not list {root}: {exc}")
            continue
        for folder in folders:
            record_group(folder)
            if not _open_to_group(folder, folder=True):
                closed_folders.append(folder.name)
            try:
                with os.scandir(folder) as entries:
                    for entry in entries:
                        feature = names.get(entry.name)
                        if feature is None:
                            continue
                        try:
                            info = entry.stat()
                        except OSError:  # a broken link: the profile reports the file as missing
                            continue
                        checked[feature] += 1
                        record_group(Path(entry.path), info)
                        if not info.st_mode & stat.S_IRGRP:
                            closed_files[feature].append(folder.name)
            except OSError:
                closed_folders.append(folder.name)
        for feature in features:
            closed = closed_files[feature]
            if closed:
                findings.add("WARN", phase, feature, "files not readable by their group",
                             f"{len(closed):,} of {checked[feature]:,}, e.g. {closed[:3]}; {fix}")
        if closed_folders:
            findings.add("WARN", phase, None, "folders not open to their group",
                         f"{len(closed_folders):,}, e.g. {closed_folders[:3]}; {fix}")
        database = root / _PHASE_STATE_FILES[phase]
        database_open = _open_to_group(database, folder=False) if database.is_file() else None
        if database.is_file():
            record_group(database)
        if database_open is False:
            findings.add("WARN", phase, None, "state database not readable by its group",
                         f"{database.name}: the loaders cannot verify any file without it; {fix}")
        support_files: dict[str, bool | None] = {}
        for filename in adapter.required_root_files(phase):
            path = root / filename
            readable = _open_to_group(path, folder=False) if path.is_file() else None
            support_files[filename] = readable
            if path.is_file():
                record_group(path)
            if readable is None:
                findings.add(
                    "FAIL",
                    phase,
                    None,
                    "required root support file missing",
                    f"{filename}: the {adapter.source.value} "
                    "loader cannot interpret this managed product without it",
                )
            elif readable is False:
                findings.add(
                    "WARN",
                    phase,
                    None,
                    "root support file not readable by its group",
                    f"{filename}: colleagues cannot interpret the product; {fix}",
                )
        if wrong_group_paths and required_group is not None:
            examples = sorted(wrong_group_paths)[:5]
            findings.add(
                "WARN",
                phase,
                None,
                "paths not owned by required group",
                f"{len(wrong_group_paths):,} path(s), e.g. {examples}; required group is "
                f"{required_group[1]!r} (gid {required_group[0]})",
            )
        summary[phase] = {
            "groups": sorted(_account_name(gid, "group") for gid in groups),
            "required_group": (
                {"name": required_group[1], "gid": required_group[0]}
                if required_group is not None else None
            ),
            "paths_not_owned_by_required_group": len(wrong_group_paths),
            "files_checked": sum(checked.values()),
            "files_not_group_readable": {feature: len(v) for feature, v in closed_files.items() if v},
            "folders_not_open_to_group": len(closed_folders),
            "state_database_group_readable": database_open,
            "required_root_files_group_readable": support_files,
        }
    return summary


def _open_to_group(path: Path, folder: bool) -> bool:
    needed = stat.S_IRGRP | stat.S_IXGRP if folder else stat.S_IRGRP
    return path.stat().st_mode & needed == needed


def verdict(findings: Findings) -> str:
    if findings.count("FAIL"):
        return "FAIL"
    return "PASS WITH WARNINGS" if findings.count("WARN") else "PASS"


def release_gate_verdict(findings: Findings, *, fail_on_warnings: bool) -> str:
    """Binary deployment-gate result while retaining the diagnostic three-state verdict."""

    if findings.count("FAIL") or (fail_on_warnings and findings.count("WARN")):
        return "FAIL"
    return "PASS"


def render(result: dict[str, Any]) -> str:
    findings = result["findings"]
    participant_limit = result["participants_requested"]
    sample_scope = (
        "all participants per feature/phase"
        if participant_limit <= 0
        else f"up to {participant_limit} per feature/phase"
    )
    lines = [
        f"Cohort acceptance — wearable_project {result['package']['version']} ({result['package']['release_label']})",
        f"Acceptance contract: {result['cohort_acceptance_version']}",
        f"Verdict: {result['verdict']}   ({sum(f['level'] == 'FAIL' for f in findings)} failed, "
        f"{sum(f['level'] == 'WARN' for f in findings)} warnings)",
        f"Release gate:  {result['release_gate_verdict']}"
        + (" (warnings are fatal)" if result["fail_on_warnings"] else ""),
        f"Source:       {result['wearable_source']}",
        f"Native root:  {result['roots']['native']}", f"Curated root: {result['roots']['curated']}",
        f"Started {result['started']}, finished {result['finished']}",
        f"Features: {len(result['features'])}; distinct participants selected across feature-specific samples: "
        f"{len(result['participants_sampled'])} ({sample_scope}; seed {result['seed']})", "",
    ]
    for level in ("FAIL", "WARN"):
        chosen = [f for f in findings if f["level"] == level]
        if chosen:
            lines.append("Failures" if level == "FAIL" else "Warnings")
            for f in chosen:
                where = " ".join(x for x in (f["feature"], f"({f['phase']})" if f["phase"] else None) if x)
                lines.append(f"  - {where}: {f['check']}: {f['detail']}")
            lines.append("")
    lines.append("Profiles (participants, rows)")
    for feature in result["features"]:
        cells = []
        for phase in PHASES:
            p = result["profiles"].get(phase, {}).get(feature)
            cells.append(f"{phase} {p['participants']:>6,} / {p['rows']:>13,}" if p else f"{phase} {'—':>22}")
        lines.append(f"  {feature:24s} " + "   ".join(cells))
    if result.get("loads"):
        lines.extend(["", "Verified loads (rows, seconds)"])
        for feature in result["features"]:
            cells = []
            for phase in PHASES:
                load = result["loads"][phase].get(feature, {})
                cells.append(f"{phase} {load['rows']:>11,} in {load['seconds']:>7.2f}s" if "rows" in load else f"{phase} failed")
            lines.append(f"  {feature:24s} " + "   ".join(cells))
    if result.get("hash_verification"):
        lines.extend(["", "Exhaustive output hashes"])
        for phase in PHASES:
            item = result["hash_verification"].get(phase, {})
            if not item.get("enabled"):
                lines.append(f"  {phase:8s} disabled")
            else:
                lines.append(
                    f"  {phase:8s} {item.get('files_verified', 0):,}/{item.get('files_recorded', 0):,} files, "
                    f"{item.get('bytes_verified', 0):,} bytes, {item.get('failure_count', 0)} failure(s)"
                )
    return "\n".join(lines) + "\n"


def run(
    native: Path, curated: Path, participants: int, seed: int, features, profile_only: bool,
    source: WearableSource | str = DEFAULT_SOURCE, verify_all_hashes: bool = False,
    hash_workers: int = DEFAULT_HASH_WORKERS, fail_on_warnings: bool = False,
    required_group: tuple[int, str] | None = None,
) -> dict[str, Any]:
    source = normalize_source(source)
    roots = {"native": native, "curated": curated}
    findings = Findings()
    started = datetime.now(timezone.utc)
    inventory = check_inventory(roots, findings, source=source)
    profiled = check_profiles(features, roots, findings, source=source)
    sharing = check_sharing(
        features,
        roots,
        findings,
        source=source,
        required_group=required_group,
    )
    samples_by_feature = select_feature_samples(
        features,
        profiled["rows_by_participant"],
        participants,
        seed,
        source=source,
    )
    codes = sorted({
        code
        for by_phase in samples_by_feature.values()
        for phase_codes in by_phase.values()
        for code in phase_codes
    })
    loads = {} if profile_only else check_loads(
        features,
        roots,
        samples_by_feature,
        profiled["rows_by_participant"],
        findings,
        source=source,
    )
    hash_verification = (
        verify_inventory_hashes(inventory["entries"], findings, workers=hash_workers)
        if verify_all_hashes
        else {phase: {"enabled": False} for phase in PHASES}
    )
    base_verdict = verdict(findings)
    gate_verdict = release_gate_verdict(findings, fail_on_warnings=fail_on_warnings)
    return {
        "tool": "cohort_acceptance",
        "cohort_acceptance_version": COHORT_ACCEPTANCE_VERSION,
        "package": {"version": __version__, "release_label": __release_label__},
        "wearable_source": source.value,
        "pandas": pd.__version__, "python": sys.version.split()[0],
        "roots": {k: str(v) for k, v in roots.items()}, "seed": seed, "features": list(features),
        "participants_requested": participants, "participants_sampled": codes,
        "participants_sampled_by_feature": samples_by_feature,
        "participant_sampling": {
            "strategy": "sha256_rank_by_source_feature_seed",
            "limit_per_feature_phase": participants,
            "zero_means_all": True,
        },
        "verify_all_hashes": bool(verify_all_hashes),
        "hash_workers": int(hash_workers),
        "fail_on_warnings": bool(fail_on_warnings),
        "required_group": (
            {"gid": required_group[0], "name": required_group[1]}
            if required_group is not None else None
        ),
        "started": started.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
        "inventory": inventory["summary"],
        "profiles": profiled["profiles"],
        "sharing": sharing,
        "loads": loads,
        "hash_verification": hash_verification,
        "findings": findings.items,
        "verdict": base_verdict,
        "release_gate_verdict": gate_verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        type=_source_argument,
        default=DEFAULT_SOURCE,
        metavar="SOURCE",
        help="wearable source: apple_healthkit or google_fitness (default: apple_healthkit)",
    )
    parser.add_argument("--native", type=Path, default=None, help="native root (default: selected source root)")
    parser.add_argument("--curated", type=Path, default=None, help="curated root (default: selected source root)")
    parser.add_argument("--participants", type=int, default=50, help="participants to load; 0 loads all (default 50)")
    parser.add_argument("--seed", type=int, default=1, help="seed for choosing participants (default 1)")
    parser.add_argument("--features", nargs="+", help="limit to these features (default: all)")
    parser.add_argument("--profile-only", action="store_true", help="run the profiles only")
    parser.add_argument(
        "--verify-all-hashes",
        action="store_true",
        help="verify size and SHA-256 for every state-recorded output file without parsing every row",
    )
    parser.add_argument(
        "--hash-workers",
        type=int,
        default=DEFAULT_HASH_WORKERS,
        help=f"bounded workers for --verify-all-hashes (default: {DEFAULT_HASH_WORKERS})",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="return exit status 1 when any warning remains; recommended for release qualification",
    )
    parser.add_argument(
        "--required-group",
        type=_required_group_argument,
        default=None,
        metavar="GROUP",
        help="require every checked product path to be owned by this group name or gid",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="report directory (default: ./cohort_acceptance_<source>_<time>)",
    )
    args = parser.parse_args(argv)

    if args.participants < 0:
        parser.error("--participants must be zero or a positive integer")
    if args.hash_workers <= 0 or args.hash_workers > 256:
        parser.error("--hash-workers must be between 1 and 256")

    source = normalize_source(args.source)
    adapter = get_source_adapter(source)
    args.native = (args.native or adapter.native_root).expanduser()
    args.curated = (args.curated or adapter.curated_root).expanduser()
    features = available_features(source=source)
    if args.features:
        unknown = sorted(set(args.features) - set(features))
        if unknown:
            parser.error(f"unknown feature(s): {', '.join(unknown)}")
        features = tuple(f for f in features if f in set(args.features))
    for name in ("native", "curated"):
        if not getattr(args, name).is_dir():
            parser.error(f"--{name} {getattr(args, name)} is not a directory")
    out = (
        args.out
        or Path.cwd() / f"cohort_acceptance_{source.value}_{datetime.now():%Y%m%d-%H%M%S}"
    ).expanduser().resolve()
    for root in _protected_roots(args.native, args.curated):
        if out == root or root in out.parents:
            parser.error(f"--out {out} lies inside the data root {root}; reports are never written into a data root")

    result = run(
        args.native,
        args.curated,
        args.participants,
        args.seed,
        features,
        args.profile_only,
        source=source,
        verify_all_hashes=args.verify_all_hashes,
        hash_workers=args.hash_workers,
        fail_on_warnings=args.fail_on_warnings,
        required_group=args.required_group,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "cohort_acceptance.json").write_text(json.dumps(_jsonable(result), indent=2))
    text = render(result)
    (out / "cohort_acceptance.txt").write_text(text)
    print(text)
    print(f"Reports written to {out}")
    return 0 if result["release_gate_verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
