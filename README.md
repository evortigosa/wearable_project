# Wearable Data Processing and Modeling project

This project turns the Apple HealthKit exports of the Human Phenotype Project (HPP, the "10K" cohort) into verified,
analysis-ready data. It provides:

- one file per participant and feature;
- a curated copy that marks what to review or leave out, without changing any value;
- loaders that read either copy into pandas;
- tools for cohort statistics, sleep and glucose metrics, figures and reports.

Developed at the Segal Lab, Department of Computer Science and Applied Mathematics, Weizmann Institute of Science.

## How it fits together

```text
raw exports ──process──▶ native root ──curate──▶ curated root
                             │                        │
                             └────── DataLoaders ─────┘
                                          │
                utils: statistics, sleep and CGM metrics, figures, reports
```

1. **Processing** (`wearable-project process`) reads the cumulative monthly CSV exports (one folder per
   participant). It writes one CSV per participant and feature, cleaned at the feature's native time resolution.
   Runs are incremental and atomic, and a state database records every file written.
2. **Curation** (`wearable-project curate`) writes a second root with the same rows and values. It adds a curation
   status, flags, a default-inclusion decision, the acquisition method and resolved units, and never changes a
   native value.
3. **DataLoaders** read either root: one loader per feature, with the HPP interface, verified against the state
   database.
4. **`wearable_project.utils`** computes statistics, summaries, sleep and CGM metrics, figures and PDF reports
   through the DataLoaders, and documents itself with `utils.info()`.

The HPP roots are already processed and curated, so most users need only steps 3 and 4.

## Repository layout

```text
wearable_project/
├── wearable_project/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── exceptions.py
│   ├── DataLoaders/               # 33 feature loaders, one loader per feature
│   │   ├── StepCountLoader.py
│   │   ├── StepCountLoader.py
│   │   ├── HeartRateLoader.py
│   │   ├── SleepLoader.py
│   │   ├── ...
│   │   ├── WeightLoader.py
│   │   ├── _base.py               # The shared loader: reading, filters, projections, verification, size limit
│   │   ├── _derived.py            # with_local_time() and with_harmonized_values()
│   │   ├── _profile.py            # profile(): what a root holds, without parsing any file
│   │   ├── _units.py              # The unit each feature's measurement is stored in
│   │   └── info.py                # DataLoaders.info(): help on every feature
│   ├── processing/                # Stage 1: raw exports -> native root
│   │   ├── parser.py              # Streaming parser for the cumulative monthly CSV exports
│   │   ├── registry.py            # Feature families and unit policies for native rows
│   │   ├── cleaners.py            # Feature-aware cleaning at each feature's native resolution
│   │   ├── pipeline.py            # Per-participant orchestration, incremental planning, atomic commits
│   │   ├── writer.py              # Atomic participant-folder writing and the processing state database
│   │   ├── scan.py                # Read-only planning scan (`process-scan`)
│   │   ├── tracker.py             # Run telemetry and processing reports
│   │   ├── environment.py         # Version and module-integrity diagnostics
│   │   └── resampling.py          # Reserved for a later resampling stage
│   ├── curation/                  # Stage 2: native root -> curated root, non-destructively
│   │   ├── registry.py            # The feature-policy registry, the source of truth for curation
│   │   ├── models.py              # Typed policy contracts
│   │   ├── rules.py               # Curation rules and flags
│   │   ├── strategies.py          # Named strategies referenced by policies
│   │   ├── decisions.py           # Versioned human calibration decisions
│   │   ├── evidence.py            # Evidence catalog behind the policies
│   │   ├── guidance.py            # User-facing guidance for every feature
│   │   ├── explain.py             # Explanations of policies, rules and unit policies
│   │   ├── unit_resolution.py     # Participant-, source- and scale-aware unit resolution
│   │   ├── engine.py              # Row-preserving curation of one feature file
│   │   ├── pipeline.py            # Participant-level curation, staging and atomic commits
│   │   ├── state.py               # Curation state database and reports
│   │   ├── schema.py              # Column roles, projections and returned dtypes for the DataLoaders
│   │   ├── audit.py               # Read-only policy-calibration audit
│   │   ├── environment.py         # Version, fingerprint and module-integrity diagnostics
│   │   └── curate_cli.py          # Curation commands
│   └── utils/                     # Analysis through the DataLoaders
│       ├── data_statistics.py     # Coverage, daily statistics, written runs
│       ├── data_summaries.py      # Valid days, adherence, summaries, patterns, quality
│       ├── domain_metrics.py      # Sleep by night, CGM metrics and the ambulatory glucose profile
│       ├── data_plots.py          # Figures and reports
│       ├── info.py                # utils.info(): help on the four modules above
│       └── cohort_acceptance.py   # Checks the DataLoaders on the full cohort roots
├── tests/
├── README.md
├── LICENSE
└── pyproject.toml
```

## Installation

From a release wheel, or from a source checkout (the `plots` extra adds matplotlib):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install ./wearable_project-0.3.0-py3-none-any.whl     # a release wheel
```

**Requirements:** Python 3.10 or newer, pandas 2.0 or newer, matplotlib 3.5 or newer, and tqdm.

**Optional:**
- pytest, for the tests (the `dev` extra);
- pyarrow, for Parquet exports.

It is tested on Python 3.10 with pandas 2.0, and on Python 3.12 with pandas 2.1, 2.2 and 3.0.

To check an installation, run from outside the source folder:

```bash
wearable-project --version                # wearable-project 0.3.0
wearable-project curation-environment     # import source, versions, module integrity
```

## Quick start

```python
from wearable_project import DataLoaders
from wearable_project.DataLoaders import StepCountLoader

print(DataLoaders.info("StepCount"))       # what a row is, its units, curation and examples

data = StepCountLoader().get_data(registration_codes=["10K_XXXXXXXXXX"])
data.df.head()                             # indexed by RegistrationCode and Date
data.load_report["state_verified"]         # True: checked against the state database
```

```python
from wearable_project import utils
from wearable_project.utils import data_statistics as ds, data_summaries as sm, data_plots as dp

daily = ds.compute_daily_statistics("curated", features=["StepCount", "HeartRate"],
                                    participants=["10K_XXXXXXXXXX"])
summaries = sm.summarize(daily)            # valid days, adherence, participant and cohort summaries
dp.plot_hour_of_day(daily, "HeartRate", path="heart_rate_by_hour.png")
utils.info("compute_daily_statistics")     # help on any tool
```

## Loading features

There are 33 features, each with a loader named `<Feature>Loader`: ActiveEnergyBurned, ActivitySummary, BMI,
BasalEnergyBurned, BloodAlcoholContent, BloodGlucose, BloodPressure, BodyFatPercentage, BodyTemperature,
Carbohydrates, DailyDistanceCycling, DailyDistanceSwimming, DistanceWalkingRunning, Electrocardiogram, EnergyConsumed,
FlightsClimbed, HeartRate, HeartRateVariability, Height, LeanBodyMass, Mindful, OxygenSaturation, PeakFlow, Protein,
RespiratoryRate, RestingHeartRate, Sleep, StepCount, TotalFat, Vo2Max, WaistCircumference, WalkingHeartRate and
Weight.

```python
from wearable_project.DataLoaders import HeartRateLoader

loader = HeartRateLoader(phase="curated", state_validation="auto")
data = loader.get_data(
    registration_codes=["10K_XXXXXXXXXX"],
    start_date="2024-01-01",
    end_date="2024-12-31 23:59:59",
)
```

**The loader** takes these arguments:

| Argument                              | Meaning                                                                                                                        |
|---------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `phase`                               | `"curated"` (the default) or `"native"`                                                                                        |
| `root`, `native_root`, `curated_root` | Read from another root                                                                                                         |
| `state_validation`                    | `"auto"` (the default) verifies files when a state database exists; `"required"` fails without one; `"off"` skips verification |

**`get_data()`** takes these arguments:

| Argument                               | Meaning                                                    |
|----------------------------------------|------------------------------------------------------------|
| `registration_codes` (alias `reg_ids`) | Participants to read; `None` for all                       |
| `start_date`, `end_date`               | Inclusive UTC bounds; a bare date means midnight UTC       |
| `columns` (alias `cols`)               | Exactly these columns; overrides `projection`              |
| `projection`                           | `"default"`, `"analysis"` or `"full"` (see below)          |
| `default_inclusion_only`               | Curated phase only: keep only the rows included by default |
| `max_rows`                             | Refuse larger requests (see below); `None` for no limit    |
| `sort_index`                           | Sort by `RegistrationCode` and `Date` (the default)        |

### What comes back

```python
data.df                   # the rows, indexed by RegistrationCode and Date
data.df_metadata          # per participant: returned and stored rows, date span, verification
data.df_columns_metadata  # per column: role, description, dtype, unit
data.load_report          # provenance: root, phase, files, filters, projection, verification, request size
```

- **Values come back exactly as stored.** Only empty cells are missing.
- **Sum first, round last.** Totals such as steps, distances and energy are stored per interval, and a row can hold
  a fractional share of a sample, so sum the rows before rounding. Levels and rates such as heart rate are averaged,
  never summed. `DataLoaders.info()` states how each feature's values combine.
- **Dtypes are declared, not inferred,** so they never depend on which participants a call reads. Whole-number
  columns are `Int64`, other numbers are `float64`, and identifiers, device names, units and sleep states are
  categorical. Under pandas 2.x, `groupby` on a categorical column includes unobserved categories unless
  `observed=True` is passed.
- **`Date` is each event's `start_date`.** For ActivitySummary, it is the export's UTC day key.
- **Columns can differ between participants,** because each participant's file stores only the columns it uses.
  Pass `columns=[...]` for a fixed schema.
- **A file is read exactly or not at all.** If reading it would change what it says, `DataLoaderReadError` names the
  file and, where there is one, the row and value.
- **What a call could not deliver is reported** in `load_report`: requested participants without a folder, or
  without a file for the feature, and recorded files missing from disk.

### Projections

A projection is a named set of columns:

| Projection   | Returns                                                                                                                                                      |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `"default"`  | Everything except persistent identifiers, free-text device labels, the raw metadata payload, the ECG waveform, deduplication bookkeeping and ingest metadata |
| `"analysis"` | The default, plus the unit audit trail, device descriptors, bookkeeping and the IANA time zone                                                               |
| `"full"`     | Every column, for authorised provenance work inside the protected environment                                                                                |

`load_report["columns_withheld"]` lists what a projection left out.

### Curated columns and default inclusion

Curation never alters native data, which is the non-destructive contract. Every curated file has the same rows, in
the same order, with the same native columns and values. It only appends columns:

```text
canonical_value, canonical_unit, curation_unit_status, unit_evidence, unit_epoch_id,
acquisition_method, curation_status, curation_flags, include_by_default
```

`curation_status` (`pass`, `review` or `exclude_default`) and `include_by_default` (`True` or `False`) answer
different questions. Loaders return every row, and the default subset must be asked for explicitly:

```python
from wearable_project.DataLoaders import WeightLoader

df = WeightLoader().get_data(default_inclusion_only=True).df
```

Curated files store the four status columns only where a row needs a non-default value. Loaders restore them for
every row:

| Column               | Restored as      |
|----------------------|------------------|
| `acquisition_method` | `"unclassified"` |
| `curation_status`    | `"pass"`         |
| `curation_flags`     | `""` (no flags)  |
| `include_by_default` | `True`           |

Unit columns are never filled in, because a missing unit means it was not resolved. `DataLoaders.info(feature)`
explains a feature's curation policy and its rationale.

### Verification

Before any filter is applied, each file is checked against the phase's state database:

- **Every file:** SHA-256, size and row count.
- **Curated files, in addition:** the status, inclusion, acquisition-method and flag counts.

A disagreement, or a file the database does not record, raises `DataLoaderStateError`.

- **Samples without a database** load under `"auto"`, with `load_report["state_verified"]` set to `False`.
- **Files curated under a policy that differs from the installed one** load with a warning, until the curated root
  is rebuilt.
- **The database is opened read-only,** so loading never writes into the data tree. It refuses to verify while a
  processing or curation run is in progress, or after an unclean shutdown.

### Size limit

`get_data()` refuses requests above `max_rows` rows, 50,000,000 by default, raising `DataLoaderSizeError`. Near the
limit, one call can use roughly 20 to 60 GB at peak. Where the state database gives exact counts, an oversized
request is refused before any file is parsed. The request size is in `load_report["size_estimate"]`.

```python
from wearable_project.DataLoaders import AppleHealthFeatureLoader

HeartRateLoader().get_data(max_rows=5_000_000)       # this call
AppleHealthFeatureLoader.default_max_rows = None     # the whole session: no limit
```

### Local time and harmonized values

Stored timestamps are UTC. Two helpers add derived columns and keep the originals:

```python
from wearable_project.DataLoaders import SleepLoader

sleep = SleepLoader().get_data().with_local_time().df        # adds start_date_local, end_date_local
weight = WeightLoader().get_data().with_harmonized_values().df
```

- **`with_local_time()`** uses each row's `utc_offset_minutes`. ActivitySummary has no offset, so it is refused.
- **`with_harmonized_values()`** adds `harmonized_value`, `harmonized_unit` and `harmonized_unit_source`, which is
  one of `curation`, `processing`, `registry` or `unresolved`. A unit no layer establishes stays unresolved, and is
  never guessed.

### Profiling a root before loading

`profile()` describes what a root holds for a feature, from the state database and one file `stat` per file,
without parsing any CSV:

```python
print(HeartRateLoader().profile())
profile = WeightLoader().profile(registration_codes=["10K_XXXXXXXXXX"])
profile.summary          # rows, files, bytes, and in the curated phase status, inclusion and unit coverage
profile.participants     # one row per participant
```

### Errors

All loader errors derive from `DataLoaderError`:

| Error                          | Raised when                                                                  |
|--------------------------------|------------------------------------------------------------------------------|
| `DataLoaderConfigurationError` | An invalid phase, filter or option is given                                  |
| `DataLoaderPathError`          | The configured data root is unavailable                                      |
| `DataLoaderReadError`          | A file cannot be read exactly                                                |
| `DataLoaderStateError`         | A file disagrees with its state database, or the database does not record it |
| `DataLoaderSizeError`          | A call would return more than `max_rows` rows                                |

### Help on a feature

```python
print(DataLoaders.info())                                   # the loading contract and every feature
print(DataLoaders.info("Sleep", include_evidence=True))     # one feature, with its evidence
print(HeartRateLoader(phase="native").info())               # the same, for this loader's phase and root
```

A feature report explains:
- what one row is, and its time anchor;
- which columns each projection returns;
- its units in each phase;
- its curation policy;
- how its files are verified;
- its dtypes and caveats;
- runnable examples.

`info()` states the contract; `profile()` reports what a particular root holds.

## Statistics, metrics, figures and reports

```python
from wearable_project.utils import data_statistics as ds, data_summaries as sm
from wearable_project.utils import domain_metrics as dm, data_plots as dp
```

| Module            | Provides                                                                                                                                                   |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `data_statistics` | Coverage of the cohort, from `profile()` without parsing files; daily statistics per participant and local day                                             |
| `data_summaries`  | Valid days, adherence and retention, participant and cohort summaries (Table 1), weekly and hourly patterns, time zones, data quality, clinical thresholds |
| `domain_metrics`  | Sleep by noon-to-noon night, CGM consensus metrics, the ambulatory glucose profile                                                                         |
| `data_plots`      | 52 figures, and PDF reports                                                                                                                                |

```python
coverage = ds.compute_coverage("curated")               # participants, rows and bytes per feature
daily = ds.compute_daily_statistics("curated", participants=["10K_XXXXXXXXXX"])
daily.daily["StepCount"]                                 # one row per participant and local day

summaries = sm.summarize(daily)
summaries.cohort                                         # Table 1, counting each participant once
patterns = sm.temporal_patterns(daily)                   # day of week, weekday and weekend, month

metrics = dm.compute_domain_metrics("curated", participants=["10K_XXXXXXXXXX"])
metrics.sleep_nights                                     # one row per participant and night
metrics.cgm_periods                                      # CGM metrics, one row per participant

figure = dp.plot_daily_values(daily, "StepCount", participant="10K_XXXXXXXXXX")
figure.data                                              # the exact table the figure draws
dp.report(daily, "report.pdf", domain=metrics)           # the standard figures in one PDF
```

**How the statistics work:**
- **Days are the participant's local calendar days,** from each row's UTC offset.
- **Values combine as their feature requires.** Totals are summed and split at midnight in proportion; levels are
  described, never summed.
- **Time is counted once** when several devices record it.
- **Summaries use valid days only.** Each feature has a valid-day rule, and `rules=` overrides it.
- **Every threshold and rule is a named constant.** `utils.info("defaults")` shows each one's value, source and
  basis.

**Outputs:**
- **Nothing is written unless asked.** Results are returned in memory; `out=` writes a run to a folder, and `path=`
  saves a figure.
- **A written run** is crash-safe, resumable (`resume=True`) and reproducible byte for byte, apart from the times in
  its `run.json`. Every summary and figure function accepts the run's folder in place of the in-memory result.
- **At cohort scale,** compute once into a folder, and use `workers=` to process participants in parallel:
  ```bash
  python -m wearable_project.utils.data_statistics daily --phase curated --out runs/daily --workers 8
  ```

**Figures:**
- **Figures for sharing:** `min_participants=` hides any aggregate resting on fewer participants, and removes it from
  `figure.data` too. Participant identifiers appear only with `show_ids=True`.
- **Figures never need a display.** Each is a matplotlib `Figure`, saved when given `path=`.

### Help on the tools

`utils.info()` explains every function, figure, parameter, output table and column. It also explains the concepts
the tools rest on, every default with its source, and complete workflows as runnable code:

```python
utils.info()                          # the map: modules, pipeline, tools by question
utils.info("plot_agp")                # one figure: what it shows, options, its data
utils.info("min_participants")        # one parameter, and every tool that takes it
utils.info("DomainMetrics.sleep_nights")  # one output table, column by column
utils.info("valid days")              # one concept, with the rules in force
utils.info("defaults")                # every threshold and rule, with its source
utils.info("shareable report")        # one workflow, as code
utils.info("jetlag")                  # anything else: a ranked search
```

From a shell: `python -m wearable_project.utils.info plot_agp`, with `--json` for structured output.

## Processing and curation from the command line

```bash
wearable-project process-scan --input /data/exports --output /data/native --workers 16   # what a run would do
wearable-project process --input /data/exports --output /data/native --workers 8
wearable-project curate --input-native /data/native --output /data/curated --workers 8 --max-in-flight 8
wearable-project run --input /data/exports --native-output /data/native --curated-output /data/curated
```

- **Runs are incremental.** Unchanged participants are skipped, changed ones are updated or rebuilt, and every commit
  is atomic. `--mode rebuild` rebuilds everything (for `run`: `--native-mode`, `--curation-mode`). `--json-summary`
  prints the run's report as JSON.
- **`process-scan` is read-only.** It never creates or changes anything. It exits with 0 when no work is needed, 3
  with `--fail-if-work-needed` when some is, and 1 on a blocking or planning error.
- **The roots must be distinct and not nested.** `--input-native` must be a native root: a curated root is refused,
  by its state database or by its curation columns. `--allow-unmanaged-native-root` admits a native copy without a
  state database.

| Command                                                   | Purpose                                                                        |
|-----------------------------------------------------------|--------------------------------------------------------------------------------|
| `process`, `process-scan`, `run`                          | Build or update the native root; plan a run; process then curate               |
| `curate`                                                  | Build or update the curated root from a native root                            |
| `report`, `curation-report`                               | Read a persisted processing or curation report (`--json`, `--include-details`) |
| `processing-environment`, `curation-environment`          | The imported source, versions and module integrity                             |
| `registry`, `curation-registry`                           | The native processing policies; the curation policy matrix                     |
| `describe-feature`, `explain-rule`, `explain-unit-policy` | Explain a feature, a curation rule, a unit policy                              |
| `evidence`, `policy-decisions`                            | The evidence catalog; the versioned calibration decisions                      |
| `curation-audit`                                          | A read-only policy-calibration audit of a native root                          |

`wearable-project <command> --help` gives each command's options. `python -m wearable_project` is equivalent to
`wearable-project`.

## Checking the cohort

The test suites run on a small representative sample. `cohort_acceptance` checks the real roots for what only the
full cohort can show:

```bash
python -m wearable_project.utils.cohort_acceptance --participants 50 --seed 1 --out ~/acceptance
```

1. **It profiles every feature in both phases,** from the state databases only. It fails on unrecorded files and
   size mismatches, and on phases that disagree on a participant's rows.
2. **It loads every feature in both phases** for a seeded random set of participants, with verification required,
   and checks dtypes, declared types and both helpers.

It writes `cohort_acceptance.txt` and `cohort_acceptance.json` to `--out`, and exits with 0 when nothing failed, 1
when a check failed, and 2 for invalid arguments. `--features` restricts the run, `--participants 0` loads every
participant who has the chosen features, and `--native` and `--curated` point to other roots.

## Running the tests

```bash
python -m pip install -e ".[dev]"
export WEARABLE_NATIVE_SAMPLE=/path/to/native_sample
export WEARABLE_CURATED_SAMPLE=/path/to/curated_sample
python -m pytest -rs
```

Tests that need the sample roots are skipped without them, and `-rs` shows why a test was skipped. The Parquet
tests need pyarrow.

## License

MIT. See [`LICENSE`](LICENSE).
