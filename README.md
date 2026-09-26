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
wearable_project/
├── wearable_project/
│   ├── DataLoaders/
│   │   ├── StepCountLoader.py
│   │   ├── HeartRateLoader.py
│   │   ├── SleepLoader.py
│   │   ├── ...
│   │   ├── WeightLoader.py
│   │   ├── info.py
│   │   ├── _base.py
│   │   └── _profile.py
│   ├── processing/
│   │   ├── cleaners.py         # Apply feature-specific cleaning and deduplication
│   │   ├── environment.py      # Reproducibility diagnostics for native processing
│   │   ├── parser.py           # Read raw CSVs and decode nested feature payloads
│   │   ├── pipeline.py         # Process one participant and orchestrate multiprocessing
│   │   ├── registry.py         # Map each feature to its schema and processing policy
│   │   ├── resampling.py       # Optional time transformations
│   │   ├── scan.py             # Run and report participant-level planning logic
│   │   ├── tracker.py          # Track and report processing metrics
│   │   └── writer.py           # Write one feature CSV atomically
│   ├── curation/
│   │   ├── audit.py            # Read-only calibration analyses
│   │   ├── curate_cli.py       # Curate, run, report, and engine-aware commands
│   │   ├── decisions.py        # Versioned human calibration decisions
│   │   ├── engine.py           # Applies the resolved feature policy to native rows
│   │   ├── environment.py      # Package, module, and source integrity verification
│   │   ├── evidence.py         # Literature and project evidence catalog
│   │   ├── explain.py          # Assemble policy, rules, and sources
│   │   ├── guidance.py         # User-facing feature descriptions
│   │   ├── models.py           # Typed policy contracts
│   │   ├── pipeline.py         # Staging, incremental planning, and atomic commits
│   │   ├── registry.py         # All feature policies
│   │   ├── rules.py            # Standardized rule and flag definitions
│   │   ├── schema.py           # Column roles, projections, and returned dtypes for DataLoaders
│   │   ├── state.py            # Curation state, hashes, policies, manifests, and reports
│   │   ├── strategies.py       # Named strategy declarations
│   │   └── unit_resolution.py  # Context-, scale-, participant-, and cross-feature unit res.
│   ├── utils/
│   │   ├── cohort_acceptance.py    # Check a cohort release on the real roots
│   │   ├── data_statistics.py      # Cohort coverage and daily statistics
│   │   ├── data_summaries.py       # Valid days, adherence, participant summaries
│   │   ├── domain_metrics.py       # Sleep nights and CGM consensus metrics
│   │   └── release_manifest.py     # Release fingerprints and module hashes
│   ├── cli.py              # Expose simple commands
│   ├── exceptions.py
│   └── __main__.py
├── tests/
├── README.md
└── pyproject.toml
```

## Release 0.2.0rc5

This release candidate is built directly from the accepted `0.2.0rc4` source.
It adds introspective DataLoader information derived from the existing processing
registry, curation registry, and evidence-linked feature guidance while leaving
Milestone 1 processing, and Milestone 2 curation semantics behavior unchanged.

The installable PEP 440 version is:

```text
0.2.0rc5
```

### HPP-style feature DataLoaders

Loaders live under `wearable_project/DataLoaders/` and provide one public class
per cohort feature. The basic interface follows the HPP pattern:

```python
from wearable_project.DataLoaders.StepCountLoader import StepCountLoader

step_count = StepCountLoader().get_data().df
```

Choose the native representation explicitly:

```python
step_count = StepCountLoader(phase="native").get_data().df
```

For smoke tests or alternate roots, override either permanent root or the active
root directly:

```python
loader = StepCountLoader(
    phase="curated",
    curated_root="/tmp/curated_sample",
)

# Equivalent single-root override for the selected phase.
loader = StepCountLoader(phase="curated", root="/tmp/curated_sample")
```

The returned DataFrame uses the HPP-style MultiIndex:

```text
RegistrationCode, Date
```

Participant folders such as `1235738253` are exposed as
`RegistrationCode="10K_1235738253"`. `Date` is the event `start_date` for all
features except `ActivitySummary`, where it is the retained outer `datetime`.
This does not assert a canonical ActivitySummary day; that derivation remains
blocked by the Milestone 2 policy.

#### What `get_data()` returns

`get_data()` returns a `LoaderData` object shaped like the HPP one:

```python
data = StepCountLoader().get_data()

data.df                   # rows, indexed by RegistrationCode and Date
data.df_metadata          # one row per participant present in df
data.df_columns_metadata  # one row per column of df
data.load_report          # provenance of this call
```

`df_metadata` is indexed by `RegistrationCode` and reports, per participant, the
returned and stored row counts, the returned date span, and, in the curated
phase, whether the participant's file was verified against the curation state
database and curated under the installed policy, together with the fraction of
its returned rows whose acquisition method is classified. `df_columns_metadata` gives each
returned column's declared role and description, dtype, non-null count, the
registry unit for measurement columns, and the default used when a curated column
is reconstructed. `load_report` records the root, phase, files read, filters,
projection, verification outcome, request size, and acquisition-method coverage.
`LoaderData.metadata` remains as a deprecated alias of `load_report`.

Values come back exactly as stored. Floats are parsed to the nearest double, and
only empty cells are missing, so text such as the Dexcom trend arrow `"None"` is
kept rather than read as a missing value.

Values are `float64`. Totals such as StepCount, FlightsClimbed, distances and
energy are stored per interval, and a row can hold a fractional share of a source
sample that spans more than one interval, even for counts: sum rows first, and
round the total if whole numbers are needed, because rounding each row changes the
totals. Levels, rates and proportions such as HeartRate are summarized with
averages or other statistics, never sums. Some values also carry floating-point
noise from earlier arithmetic, such as `61.99999999999999` for 62, so round only
for presentation, after any aggregation. `DataLoaders.info()` states how each
feature's values combine.

Dtypes never depend on which participants a call reads. Whole-number columns
(counts, codes and offsets such as `utc_offset_minutes`) are pandas nullable
integers (`Int64`), and every other numeric column is `float64`. Columns the schema
declares categorical, such as identifiers, device names and units, are read as text
and returned as categories, so an identifier stored as `"0012"` keeps its text.
Columns keep the order of the stored files: the first participant's columns, then
each column another participant adds, merged in participant-id order whatever
order the participants were requested in.

Files are read exactly or not at all. A file is refused, with a `DataLoaderReadError`
naming the file and, where there is one, the row and value, when reading it would
change what it says: a row with more fields than the header (pandas would shift
every value into the wrong column), a header naming a column twice, text in a
numeric column, an unrecognized true/false flag, or a row without its date. On a
file that could not be verified against the state database, a row with fewer fields
than the header is refused too; a verified file is byte-identical to what the
pipeline wrote, so it skips that pass. A broken link or a directory in place of a
feature file is refused rather than treated as missing data. What a call could not
deliver is reported: `load_report["requested_participants_missing_from_root"]` lists
requested participants without a folder, `load_report["requested_participants_without_file"]`
those whose folder holds no file for the feature, and
`load_report["manifest_files_missing"]`, together with a warning, files the state
database records but the disk no longer holds. Timestamps written without an offset
are read as UTC, and a blank inclusion flag means the row is not included by default.

A call's columns are fixed by the files it reads. Date bounds and
`default_inclusion_only` never change them, and an empty result carries the same
columns and dtypes as a populated one. Because each participant's file stores
only the columns it uses, different participant subsets can return different
columns; pass `columns=[...]` for a fixed schema.

#### Column projections

Each call returns a named column set. The default leaves out persistent
identifiers, free-text device labels, the raw metadata payload, the ECG waveform,
deduplication bookkeeping, and ingest metadata:

```python
StepCountLoader().get_data()                       # projection="default"
StepCountLoader().get_data(projection="analysis")  # adds the unit audit trail, device
                                                   # descriptors, bookkeeping, IANA time zone
StepCountLoader().get_data(projection="full")      # every column
```

An explicit `columns=[...]` list overrides the projection, so every stored column
remains reachable by name. `load_report["columns_withheld"]` lists what a
projection left out. Column roles, projections, and returned dtypes are declared in
`curation/schema.py`. A stored column with no declared role is listed separately
in `load_report["columns_undeclared"]`, raises a warning, and is returned only by
`projection="full"` or by name.

`projection="full"` suits authorised provenance work inside the protected
environment; prefer the default when producing derived exports.

#### The curated logical schema

The curation writer stores four columns sparsely: a column is written only where
at least one row in that participant-feature file needs a non-default value.
Curated loaders restore the logical schema, so every curated frame carries all
four with no missing values:

```text
acquisition_method   absent or blank  ->  "unclassified"
curation_status      absent           ->  "pass"
curation_flags       absent or blank  ->  ""   (no flags)
include_by_default   absent           ->  True
```

`include_by_default` is returned as a boolean. Unit fields such as
`canonical_value` and `canonical_unit` are never filled, because their absence
means the unit was not resolved. Native loaders are never reconstructed.

Curated loaders do **not** silently discard review or excluded-by-default rows.
`curation_status="review"` and `include_by_default=False` answer different
questions. An analyst who wants the policy default subset requests it explicitly:

```python
df = WeightLoader().get_data(default_inclusion_only=True).df
```

#### Verification against the state database

Each file is checked against its row in the phase's state database before any
filter is applied. Curated files are compared with `curation_outputs` in
`.wearable_curation_state.sqlite` on SHA-256, size, and row count; on their
`pass`, `review`, and `exclude_default` counts; on their inclusion counts; and on
their acquisition-method and flag counts. Native files are compared with
`feature_outputs` in `.wearable_state.sqlite` on SHA-256, size, and row count.
The hash proves the file is exactly what the writer produced; the curated counts
additionally prove the loader's dense reconstruction agrees with the logical
state that was recorded. Any disagreement raises `DataLoaderStateError`, a
subclass of `DataLoaderReadError`, and so does a file the state database does not
record at all, such as a stray copy of a participant folder.

```python
StepCountLoader(state_validation="auto")      # default: verify when the database exists
StepCountLoader(state_validation="required")  # fail when the database is absent
StepCountLoader(state_validation="off")       # skip verification
```

Sample trees without a state database load under `"auto"` and report
`load_report["state_verified"]` as `False`. Files curated under a policy
fingerprint that differs from the installed registry, and files the database
records but that are no longer on disk, are reported and raise a warning without
failing the load. A database migrated from an engine that predates inclusion
counts is recognized; those two checks are skipped and reported.

The state database is opened with SQLite's `immutable=1`, so loading never writes
side files into the data tree and works from read-only directories. If a
non-empty `-wal` or `-journal` file shows a curation or processing run in
progress or an unclean shutdown, the loader refuses to verify rather than read
stale state.

#### Returned dtypes

Low-cardinality text columns are returned as pandas categoricals, as is Sleep's
`value`, which holds sleep states. `was_user_entered` is a nullable boolean.
These dtypes are declared rather than inferred, so they do not depend on which
participants a call covers. Three pandas behaviors follow: assigning a value
outside a column's categories raises; concatenating the results of separate
calls falls back to a plain text dtype when their categories differ; and under
pandas 2.x, `groupby` on a categorical includes unobserved categories unless
`observed=True` is passed.

#### Filters

Useful first-stage filters are available without changing stored data:

```python
df = StepCountLoader().get_data(
    registration_codes=["10K_1235738253", "10K_9870822060"],
    start_date="2024-01-01",
    end_date="2024-12-31 23:59:59",
    columns=["value", "curation_status", "include_by_default"],
).df
```

Date bounds are inclusive UTC instants. A bare date means midnight UTC, so
`end_date="2024-12-31"` stops at the first instant of that day. Bounds may be
dates, datetimes, numpy `datetime64` values, or strings; timezone-aware values
are converted to UTC. Numbers are rejected, because pandas would read
`20240101` as nanoseconds since 1970, and so are missing values; pass `None` for
no bound.

`reg_ids` and `cols` are accepted as exact aliases of `registration_codes` and
`columns`, for the HPP spelling; passing both an alias and its long form raises.

Registration codes may be strings or integers, including numpy integers and
whole-number floats, so a column of codes can be passed directly. The `10K_`
prefix is optional and case-insensitive. Missing or fractional codes raise
rather than being reported as absent participants. Hidden directories in a root
are never treated as participants.

#### Acquisition-method coverage

Every call reports how the returned rows were obtained, whatever columns were
requested:

```python
report = HeartRateLoader().get_data().load_report
report["acquisition_method_counts"]        # rows per acquisition method
report["acquisition_classified_fraction"]  # share of rows with a classified method
```

`df_metadata["acquisition_classified_fraction"]` gives the same figure per
participant. Curation classifies acquisition conservatively rather than guessing,
so much of the cohort is `unclassified`. Models conditioning on acquisition method
are restricted to the classified subset and may face substantial source-selection
bias.

#### Profiling a feature before loading

`profile()` describes what a root holds for a feature without parsing any feature
CSV. It reads the phase's state database and one `stat` per file, so it is cheap
enough to run before deciding what to load:

```python
print(HeartRateLoader().profile())

profile = WeightLoader().profile(registration_codes=["10K_1235738253"])
profile.summary        # aggregate figures, as a dictionary
profile.participants   # one row per participant, indexed by RegistrationCode
```

Its figures cover the files `get_data()` would read, so `summary["rows"]` equals
the row count of an unfiltered `get_data()` call. Curated profiles add curation
status, default inclusion, acquisition-method coverage, unit-resolution coverage,
and whether each file was curated under the installed policy. Unit resolution is
reported as not applicable, rather than as zero, for features whose unit the
registry fixes. Integrity checks compare each file's presence and size with its
state record. Without a readable state database, or with
`state_validation="off"`, the profile falls back to the filesystem and reports
bytes but not rows. `DataLoaders.info()` describes a feature's static contract;
`profile()` describes what a particular root holds.

#### Request size limit

`get_data()` refuses requests that would return more than `max_rows` rows,
25,000,000 by default, raising `DataLoaderSizeError`. On the representative
samples, peak memory while loading was roughly 350 to 1,200 bytes per returned
row depending on the feature's width, so the default keeps a single call to the
order of 10 to 30 GB at peak. Adjust it to the machine:

```python
HeartRateLoader().get_data(max_rows=None)       # no limit for this call
HeartRateLoader().get_data(max_rows=5_000_000)  # a tighter limit for this call

from wearable_project.DataLoaders import AppleHealthFeatureLoader
AppleHealthFeatureLoader.default_max_rows = None  # no limit for the session
```

When the state database gives an exact row count, an oversized request is
refused before any feature file is parsed. With date bounds, or without a
readable state database, the limit is enforced as rows are retained instead, so
a narrow date window over a large cohort still loads. The request's size is
reported in `load_report["size_estimate"]`.

#### Local wall-clock time

Stored timestamps are UTC, and each row's `utc_offset_minutes` recovers the local
clock that diurnal analyses need:

```python
df = SleepLoader().get_data().with_local_time().df
df["start_date_local"].dt.hour.value_counts()   # bedtime near local midnight
```

`with_local_time()` returns a new result with `start_date_local` and
`end_date_local` added and the UTC columns kept. The added columns are
timezone-naive, because one pandas column cannot hold the per-row offsets that
daylight saving time produces, and each row's offset applies to both its start
and end. ActivitySummary stores no offset, and `datetime` is the export's UTC day
key rather than an event time, so both are refused.

#### Harmonized values

`value` holds whatever unit a row was stored in, which for features such as
Weight differs between participants. `with_harmonized_values()` adds each row's
value in a trusted unit, without touching `value` or `canonical_value`:

```python
df = WeightLoader().get_data().with_harmonized_values().df
df.loc[df["harmonized_unit_source"] != "unresolved", ["harmonized_value", "harmonized_unit"]]
```

`harmonized_unit_source` records which layer established the unit, row by row:

```text
curation     a curation unit verdict resolved it; the value is canonical_value
processing   native processing converted it, such as a fraction to percent
registry     the registry fixes a single unit, so the stored value is canonical
unresolved   no layer established a unit; value and unit are missing
```

A curation verdict decides first, so a unit curation withheld stays unresolved
even where a registry names one. The registry path is also governed by what the
curation registry declares about the stored unit, in both phases: EnergyConsumed
is labeled kcal by processing, but curation declares its stored unit unknown
(kcal or kJ), so it is unresolved whether loaded native or curated, while its
stored `value` and `raw_unit` stay exactly as written. The same reconciliation
decides the `registry_unit` shown in `df_columns_metadata`, so the metadata and
the harmonized values always agree. Features without a numeric `value` are
refused, and so is a result whose unit verdict a projection or `columns=` list
dropped, rather than harmonizing on weaker evidence.

`load_report["derived"]` records what each helper produced, including the count
of rows per unit source.

#### Changes from 0.2.0rc5

The default projection no longer returns every stored column; pass
`projection="full"` for the previous column set. `full` now returns the curated
logical schema rather than the stored bytes, so it includes reconstructed
curation columns. `LoaderData` gains `df_metadata` and `df_columns_metadata`, and
`metadata` becomes `load_report`, kept as a deprecated alias. The inclusion
counters `default_inclusion_rows_unflagged` and
`participants_without_inclusion_flag` are replaced by
`default_inclusion_rows_reconstructed` and `default_inclusion_rows_unverified`.
Declared dtypes replace plain text and `object` columns. `get_data()` now
refuses requests above `max_rows`, 25,000,000 rows by default, raising
`DataLoaderSizeError`; pass `max_rows=None` for the previous unlimited
behavior. `profile()`, acquisition-method coverage, and the reported request
size are new, as are `with_local_time()`, `with_harmonized_values()`, and the
`reg_ids` and `cols` aliases. Native files are now verified against the
processing state database as curated files are against the curation state, so
`state_validation="required"` applies to both phases and a damaged or unrecorded
native file is refused rather than loaded. Values now load exactly as stored: about 3% of float values differ
from earlier releases in the last digit, and Dexcom `"None"` trend arrows are no
longer read as missing. Malformed participant codes and date bounds now raise
instead of silently returning nothing or everything. Stored data are unchanged,
and higher-level HPP properties, chunked or lazy backends, and feature-specific
helper methods remain additive to this contract.

## Previous release: 0.2.0rc3

RC3 hardened the read-only native-processing scan by exposing the parser and
registry versions stored in `.wearable_state.sqlite`, reporting
version-mismatch-only rebuilds, and adding `processing-environment`.

### Read-only native processing scan

Use `process-scan` to determine what a native processing run would do without
performing any processing or changing `.wearable_state.sqlite`:

```bash
wearable-project process-scan \
  --input /data/exported-at-latest \
  --output /data/apple_cleaned_native \
  --workers 16
```

The report separates participants that would be:

```text
skipped
incrementally updated
rebuilt
blocked
failed during planning
```

It also reports source months and bytes that would be parsed, later-month
additions, recovered historical months, changed historical content, missing
committed months, planner/version mismatches, and output-integrity failures.

Machine-readable detail can be saved with:

```bash
wearable-project process-scan \
  --input /data/exported-at-latest \
  --output /data/apple_cleaned_native \
  --workers 16 \
  --json --include-details \
  --report-output native-processing-plan.json
```

The scanner uses the same `plan_participant()` implementation as `process`,
opens the state database in SQLite read-only mode, and never creates the output
root. A plan remains a snapshot: if source files, output files, or state change
after the scan, the eventual processing plan may differ.

For automation, `--fail-if-work-needed` returns exit code `3` when the scan is
safe but incremental processing or rebuilding would be required. Blocking or
planning errors return exit code `1`. A successful scan returns `0`.

`process-scan` also rejects a Milestone 2 curated output root containing
`.wearable_curation_state.sqlite`.

### Native-root safety guard

`--input-native` must identify a Milestone 1 native root. The program now rejects
a Milestone 2 curated root before participant discovery or output-state creation.

The primary guard rejects a root containing:

```text
.wearable_curation_state.sqlite
```

A secondary guard scans feature headers when state markers are absent and rejects
Milestone 2-only columns such as:

```text
curation_status
curation_flags
include_by_default
curation_unit_status
```

Managed native roots are identified by:

```text
.wearable_state.sqlite
```

A deliberately unmanaged copy can be used only with the explicit
`--allow-unmanaged-native-root` option. That override never permits a root with a
curation marker or curation-only columns. The same safety contract applies to
`curate` and `curation-audit`.

Because `CURATION_ENGINE_VERSION` and all policy fingerprints remain unchanged,
upgrading to this RC does not invalidate the accepted `0.2.0a2.3` curated state
or force a cohort rebuild.

## Correction in 0.2.0a2.3

### Explicit unit provenance

Milestone 1 can persist `raw_unit` from either an explicit source-event unit or
a feature-registry convention. The curation engine now treats a unit as
payload-explicit only when the retained native row also contains explicit unit
provenance (`unit_status=explicit`, `unit_status=explicit_unresolved`, or
`unit_evidence=payload_unit`).

Consequently, `EnergyConsumed` remains conservative when its native rows merely
contain the Milestone 1 `raw_unit=kcal` convention: raw values are retained,
canonical values are withheld, and the rows remain under review. A genuinely
payload-explicit supported unit may still authorize conversion.

The mass-unit and Sleep corrections from `0.2.0a2.2` are preserved unchanged.

## Corrections in 0.2.0a2.2

### Conservative absolute mass-unit resolution

Weight and LeanBodyMass are no longer resolved from broad magnitude ranges.
Kilograms and pounds overlap too strongly for a value such as `80` or `100` to
identify a unit safely.

Mass units can now be resolved only through reviewed evidence such as:

```text
explicit raw unit
reviewed source/exporter convention
bounded Height/Weight/BMI consistency
reviewed metric-to-imperial conversion fingerprint
stable propagation from one of those trusted anchors
```

Characteristic values such as:

```text
110.23113109243879 lb -> exactly 50.0 kg
```

can be identified by a strict conversion fingerprint. Ordinary integer or
one-decimal values are not treated as unit evidence by themselves.

LeanBodyMass can inherit a Weight unit only when the Weight epoch was itself
resolved through trusted absolute-unit evidence and the temporally matched body
composition relationship is consistent. A magnitude-only or ambiguous Weight
decision cannot propagate to LeanBodyMass.

Ambiguous mass epochs retain the native value, omit `canonical_value`, receive
`curation_status=review`, and are excluded from default canonical-unit analyses.

### Sleep provenance semantics

Sleep intervals remain non-destructive and source-aware.

```text
cross_source_sleep_overlap
```

is now informational provenance context and does not by itself escalate a row to
review. Different sources can legitimately export overlapping representations.

Detailed stages are now split into two distinct INBED conditions:

```text
source_has_no_inbed_state
    The compatible source/device epoch never emits INBED. Informational only.

detailed_state_outside_available_inbed
    The source does emit INBED, but this detailed interval overlaps none of it.
    Review condition.
```

Same-source detailed-stage conflicts and same-source same-state overlap remain
review conditions. No sleep state is selected, removed, or resolved lexically.

### Unit-epoch report provenance

Detailed unit-epoch reports now expose the source context used to construct the
epoch:

```text
collecting_method_version
source_id
source_name
device
metadata_device_name
time_zone
was_user_entered
```

### Acquisition-method accounting

Every curated row is now represented in acquisition reporting. Rows without a
reviewed acquisition classification are counted as:

```text
unclassified
```

Reports expose classified, unclassified, and accounting-complete totals. This
does not add a repeated `acquisition_method=unclassified` column to every row.

### State migration

The curation SQLite state is migrated additively to schema version 3. The
curation engine version is:

```text
curation-engine-0.2.0a2.2
```

Running in `mode=auto` against an earlier curated root safely rebuilds affected
participants because the engine version changed. A new curated root remains the
cleanest comparison strategy.

## Non-destructive contract

For every participant-feature file:

```text
curated row count == native row count
row order is unchanged
all native columns remain first and unchanged
native measurements and timestamps are unchanged
record/source/metadata fields are unchanged
Milestone 1 quality_flags are unchanged
```

The curation engine may append compact derived columns:

```text
canonical_value
canonical_unit
curation_unit_status
unit_evidence
unit_epoch_id
acquisition_method
curation_status
curation_flags
include_by_default
```

No native value is clipped, replaced, or deleted.

## Status and inclusion semantics

`curation_status` and `include_by_default` answer different questions:

```text
curation_status = pass | review | exclude_default
include_by_default = 1 | 0
```

Reports expose:

```text
pass_rows
review_rows
exclude_default_status_rows
included_by_default_rows
excluded_by_default_rows
```

When `curation_flags` exists, an empty cell means no Milestone 2 curation rule
triggered for that row under the recorded policy version.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ./wearable_project-0.2.0rc5-py3-none-any.whl
```

The package requires Python 3.10 or newer and pandas 2.0 or newer. The DataLoaders
are tested on Python 3.10 with pandas 2.0, and on Python 3.12 with pandas 2.1, 2.2
and 3.0.

Verify outside a source checkout:

```bash
cd /tmp
wearable-project --version
wearable-project curation-environment
```

Expected:

```text
wearable-project 0.2.0rc5
Import source: installed_distribution
Warnings: none
```

## Curate an existing native root

A new output root is recommended for the acceptance rerun:

```bash
wearable-project curate \
  --input-native /data/apple_cleaned_native \
  --output /data/apple_cleaned_curated \
  --workers 8 \
  --max-in-flight 8
```

The native and curated roots must be distinct and non-nested.

Machine-readable run summary:

```bash
wearable-project curate \
  --input-native /data/apple_cleaned_native \
  --output /data/apple_cleaned_curated \
  --workers 8 \
  --max-in-flight 8 \
  --json-summary > curation-run.json
```

## Run Milestones 1 and 2 from raw input

`run` is an orchestrator. It invokes the unchanged native processor and then the
same curation engine.

```bash
wearable-project run \
  --input /data/exported-at-latest \
  --native-output /data/apple_cleaned_native \
  --curated-output /data/apple_cleaned_curated \
  --workers 8 \
  --max-in-flight 8
```

## Read a persisted curation report

```bash
wearable-project curation-report \
  --output /data/apple_cleaned_curated
```

Detailed JSON:

```bash
wearable-project curation-report \
  --output /data/apple_cleaned_curated \
  --json \
  --include-details > curation-report-detailed.json
```

The detailed report includes unit epochs, transition boundaries, unit evidence,
source context, acquisition classification, curation flags, and inclusion
counts.

## Unit epochs and scale changes

Stable source metadata does not imply a stable numerical unit. Unit-sensitive
features are resolved at:

```text
participant x feature x source-context epoch x numerical-scale regime
```

For Weight and LeanBodyMass, magnitude alone is never sufficient. For Height and
BloodGlucose, bounded source-epoch and scale evidence remain available under
their reviewed policies. EnergyConsumed remains conservative when kcal versus kJ
is unresolved.

## Core feature behavior

- Sleep intervals are preserved. Cross-source overlap and a source that never
  emits INBED are informational; same-source detailed-stage conflict and a
  detailed state outside available same-source INBED are review conditions.
- BloodPressure systolic and diastolic values remain coupled.
- ECG remains one waveform event and is validated without ordinary resampling.
- ActivitySummary date derivation remains blocked.
- CGM status, trend, source/device, and time-zone context remain available.
- BAC calculator estimates remain distinguishable from user-entered values.
- RestingHeartRate and WalkingHeartRate remain source-defined summary events,
  represented as either points or intervals.
- Unknown future features are copied through with a safe review/exclusion
  fallback and no guessed conversion.

## Policy, evidence, and guidance

```bash
wearable-project curation-registry
wearable-project describe-feature Sleep --include-rules
wearable-project explain-unit-policy Weight
wearable-project evidence --feature BloodPressure
wearable-project policy-decisions --format json
wearable-project curation-environment --json
```

## Runtime module boundaries

The policy/evidence layer remains separated:

```text
models.py        typed contracts
strategies.py    named strategy identifiers
rules.py         curation rule definitions and execution classification
registry.py      33-feature policy source of truth
decisions.py     versioned human calibration decisions
evidence.py      evidence catalog
guidance.py      feature explanations
explain.py       helper API/CLI assembly
audit.py         read-only policy calibration
environment.py   reproducibility and module integrity
```

The curation engine consists of:

```text
unit_resolution.py  source-, scale-, and bounded-evidence-aware unit epochs
engine.py           row-preserving feature curation
pipeline.py         participant multiprocessing and atomic commits
state.py            transactional curation state and reports
curate_cli.py       curation commands
```

## Scope boundary

This release does not implement Milestone 3. Fixed-window resampling remains a
separate future layer and will never overwrite native or curated roots.


### Diagnosing parser/registry state mismatches

The native state stores semantic parser and registry versions for every participant.
A scan now reports both the currently imported versions and the versions stored in
state. This is especially useful when a processing run used a local checkout through
`python -m wearable_project`, while a later scan used the installed-wheel console
script.

Compare the two invocation paths directly:

```bash
python -m wearable_project processing-environment
wearable-project processing-environment
```

Both commands should report the same imported package path, parser version, registry
version, and valid frozen processing-module hashes.

When source signatures are unchanged, committed outputs are usable, and the only
rebuild reason is a stored version mismatch, the scan emits a dedicated warning. It
does not suppress the rebuild automatically: a version mismatch remains a real safety
condition until the processing environments are reconciled.

## Cohort acceptance check

The DataLoader test suites run on a representative sample. Before announcing a new
cohort release, check the real roots for what only the full cohort can show:

```bash
python -m wearable_project.utils.cohort_acceptance --participants 50 --seed 1 --out ~/acceptance
```

It first profiles every feature in both phases, reading only the state databases,
so the whole cohort is covered in minutes: unrecorded files and size mismatches
fail, recorded files missing from disk are reported, and the two phases must agree
on each participant's rows. It then loads every feature in both phases for a seeded
random set of participants with `state_validation="required"`, checking that dtypes
are identical whether one participant or the set is loaded and that both helpers
work where they apply. Columns the representative sample never showed, and
implausible values, are reported as warnings. The roots default to the permanent HPP
roots; `--native` and `--curated` point elsewhere, `--features` limits the run, and
`--profile-only` skips the loads. It writes `cohort_acceptance.json` and
`cohort_acceptance.txt` to `--out`, never inside a data root, and exits with 0 when
nothing failed, 1 when a check failed, and 2 for invalid arguments.

## Cohort statistics

`wearable_project.utils.data_statistics` computes statistics through the DataLoaders,
for either phase, in two tiers:

```python
from wearable_project.utils import data_statistics as ds

coverage = ds.compute_coverage("curated")              # no CSV is parsed
coverage.features                                      # participants, rows, bytes per feature
coverage.co_availability(["Height", "Weight", "BMI"])  # participants with all three

daily = ds.compute_daily_statistics("curated", participants=["10K_1235738253"])
daily.daily["StepCount"]                               # one row per participant and local day
```

Coverage comes from each loader's `profile()`, so the whole cohort takes minutes.
Daily statistics stream one participant at a time, so memory stays bounded; with
`out=` each feature's table is written as participants finish, and `workers=`
processes participants in parallel. A day is the participant's local calendar day,
from each row's `utc_offset_minutes`; intervals are split at local midnight; values
are combined as the feature's declared measurement kind requires (totals summed in
proportion to their overlap with each day, levels described by mean, median, minimum
and maximum, never summed); time is never counted twice when records overlap; and
values are harmonized first, unless `harmonize=False`. The command line runs either
tier: `python -m wearable_project.utils.data_statistics daily --phase curated --out DIR`.
Nothing is written unless you ask. Without `out=`, statistics, summaries and domain
metrics are computed in memory and returned. With `out=` (on the command line `--out`
is required), they are written to that folder, crash-safe and resumable, and any
summary or figure function accepts the folder in place of the in-memory result.
Figures are saved only when given `path=`.

None of these is ever written into a data root, so that the roots hold only wearable
data. A location inside one is refused, before anything is created, with an
explanation. The protected roots are:

- both phases' permanent HPP roots;
- every root read in the session;
- the root a written run was computed from;
- any directory recognizable as a data root by its content: a processing state
  database, or participant folders holding feature files. A local copy is therefore
  protected too.

Symbolic links and relative paths are resolved first. A folder beside the roots, such
as their parent, stays usable. `data_statistics.protected_roots()` lists the protected
roots, and `data_statistics.guard_output(path)` applies the same check to your own
files.

Written runs are reproducible: running again with the same data and parameters gives
byte-identical files (apart from the times in `run.json`), so a run can be verified
by checksum. `run.json` also lists `participants_without_data`, requested
participants for whom no feature had data. Arguments that would silently mislead,
such as a rule naming a misspelled feature or an impossible weekday, are refused
with an explanation.

Daily rows also describe sampling cadence (the median and longest gap between
records), and each participant-feature its typical interval and regularity. A
written run is made for the full cohort: an interrupted run continues with
`resume=True` (`--resume`), `read_daily`, `iter_daily` and `read_table` read the
tables back one participant at a time, `export_parquet` adds Parquet copies where
pyarrow is installed, and `run.json` records the versions, registry fingerprints and
state-database checksum the statistics were computed from.

`wearable_project.utils.data_summaries` builds on those tables:

```python
from wearable_project.utils import data_summaries as sm

summaries = sm.summarize(daily)           # or the directory of a written run
summaries.adherence                        # valid days, adherence, runs and gaps
summaries.participant_metrics              # per participant: n, mean, SD, median, percentiles
summaries.cohort                           # Table 1 across participants
summaries.wide()                           # one row per participant, one column per metric
```

A day counts when it meets its feature's valid-day rule. The defaults are
conventions, overridable with `rules=`: heart rate needs at least 10 hours with
data; glucose from a continuous monitor needs at least 70% of its expected readings,
the consensus CGM criterion; every other feature needs at least one record.
Completeness applies only to data with a fixed cadence, so finger-stick glucose
readings are never judged against a monitor's 288 readings a day. Summaries use
valid days only, with each feature's metric chosen by its measurement kind.

Daily rows also carry context. `sources` counts the distinct devices that recorded
data that day, `records_user_entered` the records flagged as entered by hand, and
`redundant_minutes` the time recorded by more than one device. `values_below_range` and
`values_above_range` count values outside broad physiological limits
(`data_statistics.PLAUSIBLE_RANGES`, declared in each feature's harmonized unit):
values are counted, never altered, and a day with nothing that can be checked is
reported as not assessed rather than as zero. The hourly profile holds values as well
as records, with totals split at hour boundaries. Two long tables,
`daily_provenance` and `daily_curation`, count records per acquisition method and per
curation status and flag; `participant_days` lists each day's UTC offsets.

```python
summaries_by_time = sm.temporal_patterns(daily)   # day of week, weekday/weekend, month
people, cohort = sm.hour_of_day(daily)            # per local hour of day
sm.time_zones(daily)                              # home zone, days away, trips
sm.day_overlap(daily)                             # participant-days shared by each pair of features
sm.days_with(daily, ["HeartRate", "Sleep"])       # days on which all given features have data
sm.quality_report(daily)                          # provenance, redundancy, plausibility, curation
```

The weekend defaults to Friday and Saturday, the cohort's weekend in Israel; pass
`weekend_days=(5, 6)` for Saturday and Sunday. A participant's home zone is their most
common UTC offset together with the offset an hour away that covers at least 10% of
their days, their daylight-saving time, so a clock change is never mistaken for travel.

`wearable_project.utils.domain_metrics` adds sleep by night and CGM metrics:

```python
from wearable_project.utils import domain_metrics as dm

metrics = dm.compute_domain_metrics("curated")   # also out=, resume=, workers=
metrics.sleep_nights                              # one row per participant and night
metrics.cgm_periods                               # one row per participant
dm.summarize_domain(metrics).cohort               # Table 1 rows for sleep and CGM
```

A night is the local noon-to-noon window starting on its `night_date`. Asleep records
at most 120 minutes apart from one episode; the longest is the main sleep, the rest
are naps. Each night reports onset, offset and midpoint (also as hours after noon, so
they average without wrapping at midnight), sleep period, total sleep time counted once
across devices, wake after sleep onset, time in bed, efficiency, and stage minutes and
shares where stages were recorded, taken from the night's stage source (the device
that staged the most sleep) so that two devices staging the same minutes never count
them twice; `staging_sources` shows how many devices staged each night. A night with only in-bed records has no sleep
metrics, since its sleep was not measured. CGM metrics are computed only for data with
a continuous monitor's cadence, classifying readings in mg/dL (the consensus ranges'
unit) with HealthKit's conversion factor 18.015588, so readings at exactly 70 or 54
mg/dL fall in the right range. Per participant, pooled over days with at least 70% of
expected readings: mean glucose, GMI, SD, CV and time in each consensus range, with a
flag for the 14-day sufficiency criterion. Every threshold is a named constant,
recorded in the run's `run.json`.

## Figures

`wearable_project.utils.data_plots` draws from the statistics outputs, never from raw
files, so every figure uses local days, kind-aware values, harmonized units and valid
days. matplotlib is optional:

```bash
pip install "wearable_project[plots]"
```

```python
from wearable_project.utils import data_plots as dp

figure = dp.plot_hour_of_day(daily, "HeartRate", path="heart_rate_by_hour.png")
figure.data        # the exact table the figure draws
```

Each function returns a matplotlib `Figure` built without `pyplot` (nothing is shown,
no global state, no display needed) and saves it when given `path=`. Every figure
carries the table it draws as `figure.data`. Every histogram bin is drawn; nothing is
clipped unless `clip_quantile` is given, and then the figure states how many values lie
beyond; participant identifiers appear only with `show_ids=True`.


## DataLoader introspection and feature help

`DataLoaders.info()` documents the loaders from the same executable contracts they
run on: the processing registry, the curation registry and its guidance, the column
roles and projections, and the stored-unit reconciliation. It never inspects or
modifies the data roots. `profile()` is its empirical counterpart: `info()` states
the contract, `profile()` reports what a particular root holds.

```python
from wearable_project import DataLoaders

print(DataLoaders.info())
print(DataLoaders.info("StepCount"))

report = DataLoaders.info("Sleep", include_evidence=True)
structured = report.as_dict()
```

Every feature loader also exposes the same feature report with its configured
phase and root reflected in the output, including in its usage examples, which run
as printed:

```python
from wearable_project.DataLoaders.WeightLoader import WeightLoader

loader = WeightLoader(phase="curated", root="/path/to/sample")
print(loader.info())
policy_and_usage = loader.info().as_dict()
```

The overview summarizes the loading contract (projections, filters, the result's
tables and helpers, phases, verification, and the size limit) and ends with a
capability matrix showing whether local time is available and
what harmonized values the curated phase yields.

A feature report explains what one row is; its time anchor and whether local
wall-clock time can be recovered; which columns the projections return; what each
registry declares about its units, the unit the loaders actually use, and what
`with_harmonized_values()` yields in each phase; its curation policy, including the
policy's own rationale, fallback and open audits; how files are verified; the
returned dtypes; the size limit; caveats; and runnable usage examples. Reports are
phase-aware: a native-phase report says that curation has not run rather than
presenting curated behavior. Resampling declarations are informative only:
DataLoaders never resample, interpolate, impute, or silently aggregate the stored
rows.

`tests/test_dataloader_info.py` holds these statements against the loaders'
behavior on the representative samples, so the documentation cannot drift from
what the code does.

## License

MIT. See [`LICENSE`](LICENSE).
