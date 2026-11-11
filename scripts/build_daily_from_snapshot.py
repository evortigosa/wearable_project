"""
Daily Apple Health summaries from the HPP pheno snapshot (out_dir/physical_activity), in the layout of a
``data_statistics`` run, so ``build_visit_features.py`` reads them unchanged.

The snapshot holds record-level Apple Health tables re-exported centrally (more participants and later data than the
curated per-participant files). Each record becomes a row of the pipeline's record format and is summarized per
participant and local day by the pipeline's own ``summarize_result``; sleep nights come from ``sleep_nights``.

Adapter rules, where the snapshot differs from the curated files:
  - Times: the snapshot stores local wall-clock times and ``utc_offset_min`` (local minus UTC); UTC start and end are
    recovered as local minus offset, so the pipeline's local times equal the snapshot's.
  - Records ending before they start are dropped, as curation drops them; exact duplicate rows are dropped.
  - Totals (steps, energy, distance, flights) repeat some hourly bins at UTC midnight with different values: the
    revisions of one bin from two export days. The native rule needs the export day, which the snapshot lacks; the
    largest value is kept, since a cumulative bin's revisions only grow.
  - No watch/phone source filter, as in the curated build. Units are those the curated run established.

Usage:
    python scripts/build_daily_from_snapshot.py --out ../wearable_visit_features/snapshot_2026_10
    then: python scripts/build_visit_features.py --daily-run OUT/daily --domain-run OUT/domain --out OUT/features
Layout, as data_statistics writes it: the daily run's root is OUT/daily (run_parameters.json, daily/<Feature>.csv.gz);
the domain run's root is OUT/domain (sleep_nights.csv.gz).
"""


from __future__ import annotations
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from wearable_project.DataLoaders._base import LoaderData
from wearable_project.utils import data_statistics as ds
from wearable_project.utils import domain_metrics as dm


SNAPSHOT = Path("/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/Pheno/snapshots/pheno_data_snapshots/out_dir")
# feature: (snapshot value column, unit established by the curated run; None for features without a value unit)
FEATURES = {
    "ActiveEnergyBurned": ("active_kcal", "kcal"),
    "BasalEnergyBurned": ("basal_kcal", "kcal"),
    "DistanceWalkingRunning": ("distance_m", "m"),
    "FlightsClimbed": ("flights", "count"),
    "HeartRate": ("heart_rate", "beats/min"),
    "HeartRateVariability": ("hrv_sdnn_ms", "ms"),
    "Mindful": ("duration_min", None),
    "OxygenSaturation": ("spo2_pct", "%"),
    "RespiratoryRate": ("respiratory_rate", "breaths/min"),
    "RestingHeartRate": ("resting_heart_rate", "beats/min"),
    "StepCount": ("steps", "count"),
    "Vo2Max": ("vo2max", "mL/(kg*min)"),
    "WalkingHeartRate": ("walking_heart_rate", "beats/min"),
}
SLEEP = ("Sleep", "sleep_stage")
TOTALS = {feature for feature in FEATURES if ds.measurement_kind(feature) == "extensive_total"}
TIME_COLUMNS = ["start_time", "end_time", "utc_offset_min"]
OPTIONAL_COLUMNS = ["source_id"]


@dataclass
class SnapshotData(LoaderData):
    """A LoaderData whose values are already in the unit the curated run established for the feature."""

    unit: str | None = None

    def with_harmonized_values(self) -> "SnapshotData":
        frame = self.df.copy()
        frame["harmonized_value"] = frame["value"]
        frame["harmonized_unit"] = self.unit
        frame["harmonized_unit_source"] = "processing"
        return replace(self, df=frame)


def to_records(frame: pd.DataFrame, value_column: str, feature: str) -> pd.DataFrame:
    """Snapshot rows of one participant as pipeline records: UTC start/end, offset, value, optional source."""

    frame = frame.drop_duplicates()
    frame = frame[frame["end_time"] >= frame["start_time"]]
    if feature in TOTALS:
        frame = frame.sort_values(value_column).drop_duplicates(["RegistrationCode", "start_time", "end_time"], keep="last")
    offset = pd.to_timedelta(frame["utc_offset_min"].astype("int64"), unit="min")
    records = pd.DataFrame({
        "RegistrationCode": frame["RegistrationCode"].astype(str).str.replace("10k_", "10K_", regex=False),
        "start_date": (frame["start_time"] - offset).dt.tz_localize("UTC"),
        "end_date": (frame["end_time"] - offset).dt.tz_localize("UTC"),
        "utc_offset_minutes": frame["utc_offset_min"].astype("int64"),
        "value": frame[value_column].to_numpy(),
    })
    if "source_id" in frame:
        records["source_id"] = frame["source_id"].astype(str).to_numpy()
    records = records.sort_values("start_date")
    records["Date"] = records["start_date"]
    return records.set_index(["RegistrationCode", "Date"])


def loader_data(records: pd.DataFrame, feature: str, unit: str | None) -> SnapshotData:
    report = {"feature": feature, "phase": "snapshot", "date_column": "start_date",
              "columns_available": list(records.columns)}
    return SnapshotData(df=records, load_report=report, unit=unit)


def participants(path: Path, columns: list[str]):
    """The table one participant at a time; the snapshot is sorted by participant, so row groups are read in order."""

    file = pq.ParquetFile(path)
    available = [c for c in columns + OPTIONAL_COLUMNS if c in file.schema_arrow.names]
    carry = None
    for group in range(file.num_row_groups):
        chunk = file.read_row_group(group, columns=available + ["RegistrationCode"]).to_pandas(ignore_metadata=True)
        chunk["RegistrationCode"] = chunk["RegistrationCode"].astype(str)
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last = chunk["RegistrationCode"].iloc[-1]
        carry = chunk[chunk["RegistrationCode"] == last]
        for _, frame in chunk[chunk["RegistrationCode"] != last].groupby("RegistrationCode", sort=False):
            yield frame
    if carry is not None and len(carry):
        yield carry


def build_feature(feature: str, out: Path, limit: int | None) -> int:
    """Write daily/<feature>.csv.gz; one gzip member per participant keeps memory flat and participants contiguous."""

    value_column, unit = FEATURES[feature]
    path = out / "daily" / "daily" / f"{feature}.csv.gz"   # the daily run's root is OUT/daily
    columns = None
    written = 0
    with open(path, "wb") as handle:
        for frame in participants(SNAPSHOT / "physical_activity" / f"apple_{feature}" / "df.parquet",
                                  TIME_COLUMNS + [value_column]):
            records = to_records(frame, value_column, feature)
            if records.empty:
                continue
            daily, _ = ds.summarize_result(loader_data(records, feature, unit), feature, harmonize=unit is not None)
            if daily.empty:
                continue
            columns = columns or list(daily.columns)
            daily.reindex(columns=columns).to_csv(handle, header=written == 0, index=False,
                                                  compression={"method": "gzip", "mtime": 0})
            written += 1
            if limit and written >= limit:
                break
    return written


def build_sleep(out: Path, limit: int | None) -> int:
    feature, value_column = SLEEP
    path = out / "domain" / "sleep_nights.csv.gz"
    written = 0
    with open(path, "wb") as handle:
        for frame in participants(SNAPSHOT / "physical_activity" / f"apple_{feature}" / "df.parquet",
                                  TIME_COLUMNS + [value_column]):
            records = to_records(frame, value_column, feature)
            if records.empty:
                continue
            nights = dm.sleep_nights(loader_data(records, feature, None))
            if nights.empty:
                continue
            nights.to_csv(handle, header=written == 0, index=False, compression={"method": "gzip", "mtime": 0})
            written += 1
            if limit and written >= limit:
                break
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--features", nargs="*", default=[*FEATURES, SLEEP[0]])
    parser.add_argument("--limit", type=int, default=None, help="participants per feature, for a trial run")
    parser.add_argument("--jobs", type=int, default=1, help="features built in parallel")
    args = parser.parse_args()
    (args.out / "daily" / "daily").mkdir(parents=True, exist_ok=True)
    (args.out / "domain").mkdir(parents=True, exist_ok=True)
    parameters = {"output_schema": ds.OUTPUT_SCHEMA, "phase": "snapshot", "root": str(SNAPSHOT / "physical_activity"),
                  "features": [f for f in args.features if f in FEATURES], "harmonize": True,
                  "builder": "build_daily_from_snapshot.py", "limit": args.limit}
    (args.out / "daily" / "run_parameters.json").write_text(json.dumps(parameters, indent=2))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        counts = pool.map(build_one, args.features, [args.out] * len(args.features), [args.limit] * len(args.features))
        for feature, count in zip(args.features, counts):
            print(f"{feature}: {count} participants", flush=True)


def build_one(feature: str, out: Path, limit: int | None) -> int:
    return build_sleep(out, limit) if feature == SLEEP[0] else build_feature(feature, out, limit)


if __name__ == "__main__":
    main()
