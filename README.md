# Wearable Data Processing and Modeling project

This research focuses on data collected from wearable devices like smartwatches. These devices continuously monitor health
and lifestyle information provided by consenting participants, such as physical activity, heart rate, and sleep patterns.
Using this data, we are building cutting-edge machine learning models based on the Transformer architecture
(the same technology behind tools like ChatGPT) to better understand how daily habits impact health.
This research has the potential to provide deeper insights into disease prevention and health optimization.

Under development at the Segal Lab, Department of Computer Science and Applied Mathematics, Weizmann Institute of Science.

---

This repository provides an auditable pipeline for parsing, cleaning, validating, optionally regularizing, summarizing, and
segmenting participant-level Apple HealthKit-style exports. Raw data are expected as one directory per participant and one
CSV per month.

## Repository layout

```text
wearable_project/                         # repository root
├── wearable_project/                     # importable Python package
│   ├── processing/
│   │   ├── cleaning.py                   # payload decoding, timestamps, identity and deduplication
│   │   ├── policies.py                   # feature aggregation policies
│   │   ├── pipeline.py                   # participant/dataset parsing and transactional output
│   │   ├── validation.py                 # raw-data sampling and schema checks
│   │   ├── windowing.py                  # vectorized exact-overlap allocation and aggregation
│   │   └── timeseries.py                 # continuous-segment construction
│   ├── loaders/                          # reserved for future feature/model loaders
│   ├── utils/
│   │   ├── filesystem.py                 # atomic I/O, fingerprints and safe filenames
│   │   ├── runtime.py                    # workers, timestamps and path-safety checks
│   │   ├── serialization.py              # deterministic nested-value serialization
│   │   ├── statistics.py                 # coverage and quality statistics
│   │   └── plotting.py                   # plotting helpers
│   ├── cli.py                            # `wearable-project` command
│   ├── exceptions.py                     # shared exception hierarchy
│   ├── _version.py                       # single package-version source
│   └── __main__.py                       # `python -m wearable_project`
└── pyproject.toml
```

Import directly from the module that owns each implementation:

```python
from wearable_project.processing.cleaning import normalize_feature_frame
from wearable_project.processing.pipeline import ProcessingConfig, process_dataset
from wearable_project.processing.policies import FeaturePolicy
from wearable_project.processing.timeseries import SegmentConfig, build_time_series
from wearable_project.utils.statistics import StatisticsConfig, summarize_dataset
```

## Expected raw-data layout

```text
raw_root/
├── participant_0001/
│   ├── 2023-11.csv
│   ├── 2023-12.csv
│   └── 2024-1.csv
├── participant_0002/
│   ├── 2024-01.csv
│   └── 2024-02.csv
└── ...
```

### Top-level monthly CSV contract

Required columns:

| Column        | Meaning                                                                                                      |
|---------------|--------------------------------------------------------------------------------------------------------------|
| `data_source` | Export-source selector. The default target is `applehealthkit`; comparison is stripped and case-insensitive. |
| `name`        | Feature name, for example `StepCount`, `HeartRate`, or `Sleep`.                                              |
| `data`        | Serialized mapping or list of mappings containing feature records.                                           |

Optional contextual columns include `participant_id`, `datetime`, `created_at`, and `updated_at`. Their values are copied to
bounded provenance columns where appropriate.

A `data` cell can contain:

- a dictionary or list of dictionaries when using the Python API;
- JSON;
- a Python literal representation;
- a JSON/Python string encoded more than once, up to the configured decode depth.

The decoder never strips first/last characters speculatively. A payload that cannot be decoded into a mapping or list of
mappings is rejected and reported.

### Payload schemas

Most features are interval records and require:

```text
start_date, end_date
```

The cohort context also contains important exceptions:

- `ActivitySummary` is a daily/contextual record with `datetime` rather than `start_date`/`end_date`;
- `BloodPressure` uses systolic and diastolic value columns rather than a single generic `value`;
- `Electrocardiogram` can contain `average_heart_rate`, `sampling_frequency`, and `voltage_measurements`;
- `Sleep` is categorical and can include labels such as `INBED`, `ASLEEP`, `DEEP`, and `REM`.

The parser preserves every categorical sleep value. Generic fixed-window mode aggregation is not applied to sleep 
by default because stage labels can overlap hierarchically and require an explicit study definition.

Known scalar and multi-value layouts are recognized by `cleaning.value_columns`. Unrecognized columns are retained in native
event output. A feature is fixed-windowed only when both its value schema and approved policy support that transformation.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Verify the installation:

```bash
wearable-project --version
python -m wearable_project --help
```

## Keep raw and derived roots disjoint

Input and output trees must be disjoint. This prevents recursive processing and accidental replacement of raw data.

```text
study_root/
├── raw/
├── processed_events/
├── processed_5min/
├── validation/
├── statistics/
└── timeseries/
```

Do not place processed, statistics, or time-series outputs inside the raw root.

## End-to-end command-line workflow

### 1. Validate a sample of every raw file

```bash
wearable-project validate \
  /data/study/raw \
  --output /data/study/validation/raw_validation.json \
  --sample-rows 100 \
  --naive-timezone reject
```

Validation samples the beginning of each selected file. It checks top-level columns, payload decoding, required timestamp
fields, invalid/reversed timestamps, participant-ID mismatches, and participant folders with no eligible monthly files. It is
a screening step, not proof that unsampled rows are valid. Full parsing remains authoritative. Use the same naive-timezone
policy for validation and parsing.

A nonzero exit status means sampled issues were found.

### 2. Parse and clean native events

The default preserves validated events at their exported temporal resolution:

```bash
wearable-project parse \
  /data/study/raw \
  /data/study/processed_events \
  --max-workers 8 \
  --chunksize 10000 \
  --fingerprint sha256 \
  --naive-timezone reject
```

This is the recommended first production-stage representation because it preserves the evidence needed to review units,
sources, duplicate candidates, categorical labels, and event durations before imposing model-specific assumptions.

### 3. Optionally regularize approved features

Request fixed windows explicitly:

```bash
wearable-project parse \
  /data/study/raw \
  /data/study/processed_5min \
  --window 5min \
  --feature-policies examples/feature_policies.example.json \
  --max-workers 8 \
  --fingerprint sha256
```

A five-minute window is not inherently correct for every analysis. The preliminary project experiments used fixed windows,
but the processing layer now treats regularization as an explicit derived representation. Non-windowable features such as
sleep, ECG, weight, BMI, height, and Activity Summary remain native events even within a windowed run unless an override is
supplied.

Useful parse options:

| Option                            | Effect                                                                                                            |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `--window none`                   | Preserve normalized events. This is the default.                                                                  |
| `--strict`                        | Fail a participant on the first payload/schema/windowing problem rather than preserving valid rows with warnings. |
| `--no-resume`                     | Rebuild every selected participant.                                                                               |
| `--output-format parquet`         | Write Parquet feature tables; install the `parquet` extra.                                                        |
| `--naive-timezone Europe/London`  | Localize naive timestamps in an explicit IANA timezone before UTC conversion.                                     |
| `--naive-timezone reject`         | Reject records containing naive timestamps.                                                                       |
| `--drop-participant-column`       | Omit `participant_id` from feature tables; directory/manifests still identify the participant.                    |
| `--sensitive-column COLUMN`       | Remove an additional raw payload column; repeat as needed.                                                        |
| `--source-identity-column COLUMN` | Include an additional source field when deriving `source_key`; repeat as needed.                                  |
| `--sample-identity-column COLUMN` | Include an additional stable sample-ID field when deriving `sample_key`; repeat as needed.                        |
| `--deduplicate-no-id-content`     | Opt into exact-content fallback for records without stable identity.                                              |
| `--merge-sources`                 | Suppress source partitions. Use only after defining an approved source adjudication/merge rule.                   |
| `--include-non-monthly-csv`       | Include other CSV filenames that follow the same contract.                                                        |
| `--max-windows-per-event N`       | Bound expansion of a single unusually long interval.                                                              |

### 4. Generate non-destructive statistics

```bash
wearable-project stats \
  /data/study/processed_events \
  /data/study/statistics \
  --day-basis timezone \
  --timezone Asia/Jerusalem \
  --max-workers 8
```

`--day-basis` controls calendar grouping:

- `utc`: UTC calendar day;
- `timezone`: convert timestamps to the supplied IANA timezone;
- `source_offset`: use the numeric offset retained from each raw timestamp.

An IANA timezone is preferred for a known locale because it includes daylight-saving rules. A numeric offset records only the
offset present on one event.

### 5. Build continuous feature time series

```bash
wearable-project timeseries \
  /data/study/processed_5min \
  /data/study/timeseries \
  StepCount HeartRate \
  --min-duration 60min \
  --tolerance 1min \
  --duration-basis coverage \
  --min-coverage-fraction 0.80 \
  --max-workers 8
```

A new segment starts when the next interval begins later than the cumulative maximum prior end plus `--tolerance`. This handles
nested and overlapping intervals correctly.

The presentation's early single-modal prototype used at least five hours and allowed one missing five-minute window. Those are
historical experiment settings, not hard-coded scientific defaults. Reproduce them explicitly, for example:

```bash
wearable-project timeseries \
  /data/study/processed_5min \
  /data/study/timeseries_5h \
  StepCount \
  --min-duration 5h \
  --tolerance 5min \
  --duration-basis coverage
```

Use participant-level splitting downstream to prevent leakage across training, validation, and test sets.

## Fixed-window semantics

Intervals use half-open semantics: `[start_date, end_date)`. Adjacent intervals/windows therefore meet without double-counting
the shared endpoint.

For a cumulative `StepCount` event:

```text
start:  00:04
end:    00:14
value:  100 steps
window: 5 minutes
```

The exact allocation is:

| Window           | Overlap | Fraction | Output value |
|------------------|--------:|---------:|-------------:|
| `[00:00, 00:05)` |    60 s |     0.10 |     10 steps |
| `[00:05, 00:10)` |   300 s |     0.50 |     50 steps |
| `[00:10, 00:15)` |   240 s |     0.40 |     40 steps |

The output sums to 100. Equal division across the three touched windows would misplace the total.

This proportional allocation assumes uniform accumulation inside the exported interval. It conserves cumulative values but
cannot recover when activity actually happened within an hour-scale source event. Five-minute rows created from one-hour
StepCount intervals should be described as allocated estimates, not direct five-minute observations.

For a discrete/ratio measurement, the original value is not divided. The default policy computes an overlap-weighted mean.
A point event (`start_date == end_date`) is assigned to one containing window with zero observed duration and allocation
fraction one.

### Window quality columns

| Column                              | Definition                                                                            |
|-------------------------------------|---------------------------------------------------------------------------------------|
| `observed_duration_seconds`         | Union coverage inside this unit/source/manual-entry partition and window.             |
| `feature_observed_duration_seconds` | Union coverage across all partitions for the feature/window.                          |
| `overlap_seconds_sum`               | Sum of event overlap contributions; can exceed union coverage when intervals overlap. |
| `event_count`                       | Number of source events contributing to the output row.                               |
| `is_windowed`                       | `True` for aggregated windows; `False` for preserved native events.                   |

Coverage and overlap intensity are intentionally separate quantities.

## Feature policies

Feature names are canonicalized by removing punctuation and ignoring case. The built-in registry covers the feature labels
identified in the preliminary cohort presentation, while remaining overrideable because exporter names and study semantics can
change.

| Feature family                                                      | Default aggregation | Cumulative | Windowable |
|---------------------------------------------------------------------|---------------------|-----------:|-----------:|
| Step/distance/energy/flights/nutrition totals                       | `sum`               |        yes |        yes |
| Heart rate/HRV/oxygen/respiratory/temperature/glucose/VO2/peak flow | `weighted_mean`     |         no |        yes |
| Weight/height/BMI/body composition/waist circumference              | preserved           |         no |         no |
| Sleep/SleepAnalysis                                                 | preserved           |         no |         no |
| ECG/ActivitySummary                                                 | preserved           |         no |         no |
| Unknown feature                                                     | preserved           |         no |         no |

The explicit built-in inventory includes:

```text
ActiveEnergyBurned, ActivitySummary, BasalEnergyBurned, BloodAlcoholContent,
BloodGlucose, BloodPressure, BodyFatPercentage, BodyTemperature, BMI,
Carbohydrates, DailyDistanceCycling, DailyDistanceSwimming,
DistanceWalkingRunning, Electrocardiogram, EnergyConsumed, FlightsClimbed,
HeartRate, HeartRateVariability, Height, LeanBodyMass, OxygenSaturation,
PeakFlow, Protein, RespiratoryRate, RestingHeartRate, Sleep, StepCount,
TotalFat, Vo2Max, WaistCircumference, WalkingHeartRate, Weight
```

A built-in entry means the software has a conservative default, not that the feature is automatically suitable for a model.
Unknown names receive `policy_origin: "unknown_conservative"` in quality reports.

Policy override format:

```json
{
  "StepCount": {
    "aggregation": "sum",
    "cumulative": true,
    "windowable": true
  },
  "HeartRate": {
    "aggregation": "weighted_mean",
    "cumulative": false,
    "windowable": true
  },
  "LabSpecificScore": {
    "aggregation": "mean",
    "cumulative": false,
    "windowable": true
  }
}
```

Allowed aggregations are `sum`, `weighted_mean`, `mean`, `median`, and `mode`. A cumulative policy must use `sum`.
Categorical data should remain native unless the study defines how simultaneous/overlapping categories are resolved.

## Sleep handling

Sleep requires separate treatment from ordinary scalar quantities:

- labels are preserved exactly after basic normalization;
- `INBED` is not treated as equivalent to physiological sleep;
- `ASLEEP`, `DEEP`, and `REM` are not removed or remapped to `INBED`;
- native intervals remain the default representation;
- generic window `mode` is not used automatically because overlapping labels can represent nested concepts;
- duration summaries should use a study-defined label taxonomy and interval-union rules;
- source overlap and duplicated sleep sessions require explicit adjudication.

## Units

The pipeline strips surrounding whitespace from unit labels but performs no automatic conversion. Mixed units produce separate
partitions. This is deliberate: converting `mg/dL`, `mmol/L`, `m`, `km`, `kg`, `lb`, `cm`, `in`, `count/min`, and
exporter-specific labels requires an approved feature/unit contract.

Before modeling:

1. inventory `unit_values` from participant quality reports;
2. define one canonical unit and dimensional family per feature;
3. implement explicit conversions with unit tests;
4. reject unknown or dimensionally incompatible labels;
5. version the conversion registry and record its digest in model-dataset metadata.

Missing units remain missing. The pipeline does not infer units from feature names or value magnitudes.

## Sources and devices

When source fields such as `source_id` or `source_name` exist, the parser:

1. creates a deterministic participant-scoped SHA-256-derived `source_key`;
2. removes raw source fields by default;
3. keeps different source keys as separate aggregation partitions;
4. sets `source_adjudication_required: true` when a feature has multiple observed source identities.

This is pseudonymization, not anonymization. Do not blindly sum or average source partitions. A valid rule may depend on HealthKit
source priority, device class, study protocol, wear-state evidence, completeness, or exporter behavior. `--merge-sources`
removes the safeguard but does not implement a scientifically valid source-priority algorithm.

## Outliers and plausibility checks

The preliminary analysis identified extreme values, including step-count spikes. The preparation pipeline intentionally does not
apply a universal outlier filter because physiological ranges, device artifacts, manual entries, units, and interval durations
are feature-specific.

Recommended design:

1. preserve the normalized event table;
2. attach non-destructive quality flags rather than deleting rows immediately;
3. define feature-, unit-, source-, and duration-aware rules;
4. distinguish impossible values from unusual but possible observations;
5. evaluate participant-local and cohort-level distributions separately;
6. retain raw-to-clean lineage and version every rule;
7. compare model results with and without uncertain records.

Outlier handling belongs in a reviewed QC/analysis stage after units and sources are resolved.

## Output layout and provenance

```text
processed_root/
├── processing_summary.csv
├── processing_summary.json
├── participant_0001/
│   ├── .wearable_manifest.json
│   ├── quality_report.json
│   ├── StepCount.csv
│   ├── HeartRate.csv
│   └── ...
└── participant_0002/
    └── ...
```

### Participant manifest

`.wearable_manifest.json` records:

- pipeline and manifest-schema versions;
- processing configuration and digest;
- complete input-file fingerprints;
- output filenames, row counts, and windowed flags;
- participant completion state;
- quality-report location.

Resume skips a participant only when the configuration digest, ordered input fingerprints, and all manifest-declared output
files still match. `metadata` fingerprints use filename, byte size, and nanosecond modification time. `sha256` also hashes
content and is preferred for a frozen production export.

## Statistics outputs

The `stats` command writes long-form auditable tables:

| File                                  | Purpose                                                                   |
|---------------------------------------|---------------------------------------------------------------------------|
| `statistics_feature_summary.csv`      | Participant/feature rows, timestamp range, coverage, storage, and errors. |
| `statistics_daily_counts.csv`         | Processed rows and source-event contributions by day.                     |
| `statistics_daily_coverage.csv`       | Interval-union coverage by participant, feature, and day.                 |
| `statistics_hourly_counts.csv`        | Processed row counts by day and hour.                                     |
| `statistics_active_participants.csv`  | Participants with events/coverage on each date.                           |
| `statistics_participant_storage.csv`  | Participant-folder and feature-file storage.                              |
| `statistics_participant_features.csv` | Participant-by-feature presence matrix.                                   |
| `statistics_errors.csv`               | Participant- and feature-level failures.                                  |
| `statistics_summary.json`             | Run metadata and coverage definition.                                     |

Legacy `logfile_*.csv` files are also written where retained for analysis compatibility. The statistics stage never modifies
processed feature files.

## Time-series outputs

For each requested feature:

```text
timeseries_root/
├── StepCount_timeseries.csv
├── StepCount_timeseries_summary.json
├── HeartRate_timeseries.csv
└── HeartRate_timeseries_summary.json
```

Rows contain participant/feature identifiers, deterministic segment IDs, segment boundaries, elapsed span, interval-union
coverage, gap seconds, coverage fraction, and original feature columns. Workers write private temporary files; the parent
combines them deterministically and atomically. No workers concurrently append to one final CSV.

## Python API

```python
from wearable_project.processing.pipeline import ProcessingConfig, process_dataset
from wearable_project.processing.timeseries import SegmentConfig, build_time_series
from wearable_project.utils.statistics import StatisticsConfig, summarize_dataset

processing_reports = process_dataset(
    "/data/study/raw",
    "/data/study/processed_events",
    ProcessingConfig(
        window=None,
        max_workers=8,
        strict=False,
        fingerprint_mode="sha256",
        naive_timezone="reject",
    ),
)

statistics = summarize_dataset(
    "/data/study/processed_events",
    "/data/study/statistics",
    StatisticsConfig(
        day_basis="timezone",
        timezone="Asia/Jerusalem",
        max_workers=8,
    ),
)

segment_reports = build_time_series(
    "/data/study/processed_5min",
    "/data/study/timeseries",
    ["StepCount", "HeartRate"],
    SegmentConfig(
        min_duration="5h",
        tolerance="5min",
        duration_basis="coverage",
        min_coverage_fraction=0.8,
        max_workers=8,
    ),
)
```

Programmatic callers should inspect all returned statuses and JSON quality artifacts rather than treating file existence as
success.

## Important limitations

- **Uniform-within-interval assumption:** proportional cumulative allocation conserves totals but cannot reconstruct intra-event
  behavior.
- **No automatic unit conversion:** mixed and missing units remain explicit.
- **No automatic source priority:** source partitions remain separate and are flagged.
- **Sleep is not generically windowed:** a study-specific overlap/stage policy is required for a regularized sleep representation.
- **Top-level chunking is not payload streaming:** one unusually large serialized `data` cell is decoded in memory.
- **Offset is not timezone:** a numeric offset does not encode historical daylight-saving rules.
- **Validation samples rows:** passing validation does not certify unsampled rows; full parsing is authoritative.
- **CSV is lossy for types:** Parquet is preferable when downstream tooling supports it.
- **Multimodal missingness is structural:** absence of a feature or month must not automatically become numeric zero.

## License

MIT. See [`LICENSE`](LICENSE).
