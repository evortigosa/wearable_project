"""
Wearable Data Processing and Modeling project
Cohort acceptance check for the DataLoaders, run on the real native and curated roots. The loader test suites
run on a representative sample. This script checks what only the full cohort can show, without loading all of it:
1. Profiles. ``profile()`` for every feature in both phases. It reads only the state databases and one ``stat``
   per file, so the whole cohort is covered in minutes. Unrecorded files and size mismatches fail; recorded files
   missing from disk are reported. The two phases must agree on each participant's rows.
2. Verified loads. For a seeded random set of participants, every feature is loaded in both phases with
   ``state_validation="required"``. Each load is checked: dtypes identical whether one participant or the set is
   loaded; declared categorical and whole-number columns typed as promised; curated rows equal to native rows per
   participant; both helpers working wherever they apply. Columns the representative sample never showed, and
   implausible values, are reported.
Usage:

    python -m wearable_project.utils.cohort_acceptance --participants 50 --seed 1 --out ~/acceptance

The roots default to the permanent HPP roots; ``--native`` and ``--curated`` point elsewhere. Reports are written as
``cohort_acceptance.json`` and ``cohort_acceptance.txt`` in ``--out``, which must lie outside both roots. The exit
status is 0 when nothing failed (warnings allowed), 1 when a check failed, and 2 for invalid arguments.
"""


from __future__ import annotations
import argparse
import json
import random
import sys
import time
import traceback
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from wearable_project import __release_label__, __version__
from wearable_project.curation import schema
from wearable_project.DataLoaders import available_features
from wearable_project.DataLoaders._base import DEFAULT_CURATED_ROOT, DEFAULT_NATIVE_ROOT, AppleHealthFeatureLoader
from wearable_project.exceptions import DataLoaderError


PHASES = ("native", "curated")
EARLIEST_PLAUSIBLE = pd.Timestamp("2000-01-01", tz="UTC")
OFFSET_RANGE = (-720, 840)  # UTC-12:00 to UTC+14:00, in minutes
DETERMINISM_PROBES = 3


class Findings:
    """Every FAIL and WARN, with where it was found."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, level: str, phase: str | None, feature: str | None, check: str, detail: str) -> None:
        self.items.append({"level": level, "phase": phase, "feature": feature, "check": check, "detail": detail})

    def count(self, level: str) -> int:
        return sum(1 for item in self.items if item["level"] == level)


def _loader(feature: str, phase: str, root: Path, **kwargs) -> AppleHealthFeatureLoader:
    module = __import__(f"wearable_project.DataLoaders.{feature}Loader", fromlist=[f"{feature}Loader"])
    return getattr(module, f"{feature}Loader")(phase=phase, root=root, **kwargs)


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


# ------------------------------------------------------------------------------------------------ profiles
def check_profiles(features, roots, findings: Findings) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {phase: {} for phase in PHASES}
    rows_by_participant: dict[str, dict[str, dict[str, int]]] = {phase: {} for phase in PHASES}
    for phase in PHASES:
        for feature in features:
            started = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    report = _loader(feature, phase, roots[phase]).profile()
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
def check_load(feature, phase, root, codes, findings: Findings) -> dict[str, Any]:
    started = time.perf_counter()
    loader = _loader(feature, phase, root, state_validation="required")
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
    result: dict[str, Any] = {"rows": int(len(df)), "participants": int(report["participants_loaded"]),
                              "seconds": round(seconds, 3), "rows_per_second": int(len(df) / seconds) if seconds else None}

    date = df.index.get_level_values("Date").dtype
    if list(df.index.names) != ["RegistrationCode", "Date"] or not (isinstance(date, pd.DatetimeTZDtype) and str(date.tz) == "UTC"):
        findings.add("FAIL", phase, feature, "index", f"names {list(df.index.names)}, Date dtype {date}")
    if report["state_verified"] is not True:
        findings.add("FAIL", phase, feature, "verification", "the load was not verified against the state database")

    categorical = schema.categorical_columns(feature)
    for column in df.columns:
        dtype = df[column].dtype
        if column in categorical and not isinstance(dtype, pd.CategoricalDtype):
            findings.add("FAIL", phase, feature, "declared categorical column", f"{column} is {dtype}")
        if column in schema.INTEGER_COLUMNS and str(dtype) != "Int64":
            fractional = int((df[column].dropna() % 1 != 0).sum()) if pd.api.types.is_numeric_dtype(dtype) else None
            findings.add("WARN", phase, feature, "whole-number column holds fractions",
                         f"{column} is {dtype}; {fractional} non-whole value(s); its dtype now depends on the participants read")
    undeclared = schema.undeclared_columns(report["columns_available"])
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


def check_loads(features, roots, codes, rows_by_participant, findings: Findings) -> dict[str, Any]:
    loads: dict[str, dict[str, Any]] = {phase: {} for phase in PHASES}
    for phase in PHASES:
        for feature in features:
            loads[phase][feature] = check_load(feature, phase, roots[phase], codes, findings)
    return loads


# -------------------------------------------------------------------------------------------------- report
def verdict(findings: Findings) -> str:
    if findings.count("FAIL"):
        return "FAIL"
    return "PASS WITH WARNINGS" if findings.count("WARN") else "PASS"


def render(result: dict[str, Any]) -> str:
    findings = result["findings"]
    lines = [
        f"Cohort acceptance — wearable_project {result['package']['version']} ({result['package']['release_label']})",
        f"Verdict: {result['verdict']}   ({sum(f['level'] == 'FAIL' for f in findings)} failed, "
        f"{sum(f['level'] == 'WARN' for f in findings)} warnings)",
        f"Native root:  {result['roots']['native']}", f"Curated root: {result['roots']['curated']}",
        f"Started {result['started']}, finished {result['finished']}",
        f"Features: {len(result['features'])}; participants loaded: {len(result['participants_sampled'])} "
        f"(seed {result['seed']})", "",
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
    return "\n".join(lines) + "\n"


def run(native: Path, curated: Path, participants: int, seed: int, features, profile_only: bool) -> dict[str, Any]:
    roots = {"native": native, "curated": curated}
    findings = Findings()
    started = datetime.now(timezone.utc)
    profiled = check_profiles(features, roots, findings)
    pool = sorted({code for phase in PHASES for rows in profiled["rows_by_participant"][phase].values() for code in rows})
    count = len(pool) if participants <= 0 else min(participants, len(pool))
    codes = sorted(random.Random(seed).sample(pool, count)) if pool else []
    loads = {} if profile_only else check_loads(features, roots, codes, profiled["rows_by_participant"], findings)
    return {
        "tool": "cohort_acceptance", "package": {"version": __version__, "release_label": __release_label__},
        "pandas": pd.__version__, "python": sys.version.split()[0],
        "roots": {k: str(v) for k, v in roots.items()}, "seed": seed, "features": list(features),
        "participants_requested": participants, "participants_sampled": codes,
        "started": started.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
        "profiles": profiled["profiles"], "loads": loads, "findings": findings.items, "verdict": verdict(findings),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--native", type=Path, default=Path(DEFAULT_NATIVE_ROOT), help="native root")
    parser.add_argument("--curated", type=Path, default=Path(DEFAULT_CURATED_ROOT), help="curated root")
    parser.add_argument("--participants", type=int, default=50, help="participants to load; 0 loads all (default 50)")
    parser.add_argument("--seed", type=int, default=1, help="seed for choosing participants (default 1)")
    parser.add_argument("--features", nargs="+", help="limit to these features (default: all)")
    parser.add_argument("--profile-only", action="store_true", help="run the profiles only")
    parser.add_argument("--out", type=Path, default=None, help="report directory (default: ./cohort_acceptance_<time>)")
    args = parser.parse_args(argv)

    features = available_features()
    if args.features:
        unknown = sorted(set(args.features) - set(features))
        if unknown:
            parser.error(f"unknown feature(s): {', '.join(unknown)}")
        features = tuple(f for f in features if f in set(args.features))
    for name in ("native", "curated"):
        if not getattr(args, name).is_dir():
            parser.error(f"--{name} {getattr(args, name)} is not a directory")
    out = (args.out or Path.cwd() / f"cohort_acceptance_{datetime.now():%Y%m%d-%H%M%S}").expanduser().resolve()
    for root in (args.native, args.curated):
        if out == root.resolve() or root.resolve() in out.parents:
            parser.error(f"--out {out} lies inside a data root; reports are never written into the roots")

    result = run(args.native, args.curated, args.participants, args.seed, features, args.profile_only)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cohort_acceptance.json").write_text(json.dumps(_jsonable(result), indent=2))
    text = render(result)
    (out / "cohort_acceptance.txt").write_text(text)
    print(text)
    print(f"Reports written to {out}")
    return 1 if result["verdict"] == "FAIL" else 0


if __name__ == "__main__":  # pragma: no cover
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
