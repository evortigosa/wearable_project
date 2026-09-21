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
│   │   └── _base.py
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
│   │   ├── release_manifest.py
│   │   ├── 
│   │   └── 
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

The default processing phase is curated. The permanent HPP roots are:

```text
native:
/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/third-party/EV_cleaned_apple_healthkit

curated:
/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/third-party/EV_curated_apple_healthkit
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
database and curated under the installed policy. `df_columns_metadata` gives each
returned column's declared role and description, dtype, non-null count, the
registry unit for measurement columns, and the default used when a curated column
is reconstructed. `load_report` records the root, phase, files read, filters,
projection, and verification outcome. `LoaderData.metadata` remains as a
deprecated alias of `load_report`.

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

#### Verification against the curation state

In the curated phase, each file is checked against its row in
`.wearable_curation_state.sqlite` before any filter is applied: its SHA-256, size,
and row count; its `pass`, `review`, and `exclude_default` counts; its inclusion
counts; and its acquisition-method and flag counts. The hash proves the file is
exactly what curation wrote; the counts prove the loader's reconstruction agrees
with the logical state the writer recorded. Any disagreement raises
`DataLoaderStateError`, a subclass of `DataLoaderReadError`.

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
counts is recognized; those two checks are skipped and reported. The native phase
has no curation state, so `"required"` is rejected there.

The state database is opened with SQLite's `immutable=1`, so loading never writes
side files into the data tree and works from read-only directories. If a
non-empty `-wal` file shows a curation run in progress or an unclean shutdown,
the loader refuses to verify rather than read stale state.

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
`end_date="2024-12-31"` stops at the first instant of that day.

#### Changes from 0.2.0rc5

The default projection no longer returns every stored column; pass
`projection="full"` for the previous column set. `full` now returns the curated
logical schema rather than the stored bytes, so it includes reconstructed
curation columns. `LoaderData` gains `df_metadata` and `df_columns_metadata`, and
`metadata` becomes `load_report`, kept as a deprecated alias. The inclusion
counters `default_inclusion_rows_unflagged` and
`participants_without_inclusion_flag` are replaced by
`default_inclusion_rows_reconstructed` and `default_inclusion_rows_unverified`.
Declared dtypes replace plain text and `object` columns. Stored data are
unchanged, and higher-level HPP properties, chunked or lazy backends, and
feature-specific helper methods remain additive to this contract.

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

## DataLoader introspection and feature help

The DataLoader package exposes read-only documentation derived from the same
processing registry, curation registry, and user-facing feature guidance used
by the pipeline.  It does not inspect or modify the data roots.

```python
from wearable_project import DataLoaders

print(DataLoaders.info())
print(DataLoaders.info("StepCount"))

report = DataLoaders.info("Sleep", include_evidence=True)
structured = report.as_dict()
```

Every feature loader also exposes the same feature report with its configured
phase and root reflected in the output:

```python
from wearable_project.DataLoaders.WeightLoader import WeightLoader

loader = WeightLoader(phase="curated", root="/path/to/sample")
print(loader.info())
policy_and_usage = loader.info().as_dict()
```

The overview documents phase semantics, default roots, HPP index conventions,
participant/date/column filtering, curated default-inclusion behavior, sparse
derived columns, read-only guarantees, and memory/scalability considerations.
A feature report additionally includes the Milestone-1 feature family/unit and
deduplication contract, the Milestone-2 curation policy and future resampling
declaration, evidence-linked feature guidance, caveats, and runnable usage
examples.  Resampling declarations are informative only: DataLoaders never
resample, interpolate, impute, or silently aggregate the stored rows.

## License

MIT. See [`LICENSE`](LICENSE).
