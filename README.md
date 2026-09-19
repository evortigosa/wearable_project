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
│   │   ├── state.py            # Curation state, hashes, policies, manifests, and reports
│   │   ├── strategies.py       # Named strategy declarations
│   │   ├── unit_resolution.py  # Context-, scale-, participant-, and cross-feature unit res.
│   │   └── 
│   ├── utils/
│   │   ├── 
│   │   ├── 
│   │   └── 
│   ├── cli.py              # Expose simple commands
│   ├── exceptions.py
│   └── __main__.py
├── tests/
├── README.md
└── pyproject.toml
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## First processing run

```bash
wearable-project process \
  --input /data/exported-at-2024-12 \
  --output /data/cleaned_apple_healthkit \
  --workers 8 \
  --max-in-flight 8
```

The input may either be the participant root itself or a directory containing
exactly one participant root.

## Ordinary cumulative update

```bash
wearable-project process \
  --input /data/exported-at-2025-01 \
  --output /data/cleaned_apple_healthkit \
  --workers 8
```

Under the default `strict-cumulative` policy:

- unchanged historical monthly files are skipped by SHA-256;
- only newly added later months take the incremental fast path;
- a changed historical month triggers a complete rebuild of that participant;
- a recovered older month triggers a complete rebuild of that participant;
- an omitted committed month blocks that participant and leaves its previous
  output untouched;
- a failed or interrupted participant is rebuilt on the next run.

The required correctness invariant is:

```text
process(December), then update(January)
    == process January cumulative export from scratch
```

Historical corrections replace old values through participant rebuilds. They
are never combined through a median or another aggregate.

## Snapshot policies

```text
strict-cumulative  default; missing committed months block the participant

authoritative     newest snapshot is complete truth; omissions trigger rebuild

append-only       preserves missing old months and appends only safe later data
```

Use `authoritative` only when the newest export is deliberately intended to
replace prior source history.

## Rebuild and selected participants

```bash
wearable-project process \
  --input /data/exported-at-2025-01 \
  --output /data/cleaned_apple_healthkit \
  --mode rebuild \
  --workers 8
```

A participant allow-list can be supplied with one folder name per line:

```bash
wearable-project process ... --participants-file participants.txt
```

## Failure behavior

The default `fail-participant` row policy preserves transactional safety:
malformed data prevents that participant from being committed, while other
participants continue. The previous committed participant directory remains
unchanged. `--fail-fast` stops the full run after the first participant error.

`--row-error-policy skip-row` is available for exploratory recovery, but it
can produce an intentionally incomplete participant output and should not be
the default laboratory policy.

## Native output semantics

Participant identity is encoded by the parent directory and feature identity
by the CSV filename; those strings are not repeated in every row. Internal
event hashes and source-file bookkeeping are used during reconciliation but
are not written into the analytical feature CSVs.

For ordinary scalar features, critical columns are written first:

```text
start_date, end_date, value, datetime, created_at, updated_at,
data_source, collecting_method_version
```

They are followed only by feature-relevant provenance, metadata, unit, and
exceptional duplicate/revision fields. UTC timestamps plus
`utc_offset_minutes` and, when available, an IANA `time_zone` preserve local
time reconstruction without storing multiple equivalent timestamp strings.
Columns that are empty for a complete participant-feature file are omitted.

Feature families are treated differently:

| Family              | Core behavior                                                 |
|---------------------|---------------------------------------------------------------|
| Interval totals     | Preserve original interval and total; no five-minute division |
| Intensive intervals | Preserve every native observation and duration                |
| Point scalars       | Preserve zero-duration point measurement                      |
| Point vectors       | Keep components coupled, e.g. systolic/diastolic pressure     |
| Sleep states        | Preserve all state intervals; never categorical mode          |
| Daily summaries     | Preserve every payload item; never average ambiguous pairs    |
| Long summaries      | Preserve one long-summary event; never broadcast it           |
| ECG                 | Preserve waveform in the feature CSV as one event             |
| Unknown features    | Preserve generic payload fields and mark policy provisional   |

Useful metadata is retained, including source IDs, manual-entry status,
HealthKit time zone, heart-rate motion context, and CGM status/trend fields.
Frequently used metadata is flattened into columns. The `metadata` column,
when present, contains only normalized residual keys not already represented
elsewhere; a duplicate raw metadata dictionary is not written.

### Audit count columns

The reconciliation counters are optional at the file level:

```text
occurrence_count
duplicate_count
revision_count
```

If every row in a participant-feature file has the ordinary defaults
`(1, 0, 0)`, these columns are omitted. If at least one row represents a
duplicate or revision, the needed columns are retained and **every row has an
explicit integer value**:

```text
occurrence_count default = 1
duplicate_count  default = 0
revision_count   default = 0
```

Consequently, `occurrence_count.sum()` is the number of represented source
occurrences for that file, while the duplicate and revision sums recover the
corresponding reconciled occurrence counts. Blank cells never stand for an
implicit count.

### `quality_flags` scope

`quality_flags` is deliberately a sparse column of retained, non-redundant
warnings. It is **not an exhaustive QC record**. An absent column or empty
cell means only that the row has no retained flags; it does not certify that
every possible structural or clinical check passed.

The native compact writer intentionally omits deterministic conditions that
can be reconstructed from other persisted fields, including:

```text
interval_feature_has_zero_duration
point_feature_has_nonzero_duration
outer_bucket_differs_from_local_start_date
unit_inferred_not_explicit
```

For example, zero/nonzero duration is recoverable from `start_date` and
`end_date`, local-date displacement is recoverable from UTC timestamps plus
`utc_offset_minutes` or `time_zone`, and inferred-unit status is represented
by the unit columns. Milestone 2 will add a reviewed feature-specific quality
status; the milestone-1 `quality_flags` column should not be used as a single
boolean definition of a clean row.

## Deduplication and revision policy

The cleaner separates:

1. repeated record UUIDs;
2. exact-content duplicates with different UUIDs;
3. conflicting revisions of the same aggregate interval;
4. distinct observations that merely occur close together.

The first three can be reconciled according to feature policy. The fourth is
always preserved in native output. Sleep states and blood-pressure pairs are
never fused by a universal median or mode. Nutrition entries are deduplicated
by record identity only because two identical items at the same time may be
legitimate.

For the observed Apple boundary-revision pattern in additive interval totals,
the occurrence whose outer bucket date matches the interval's local start date
is selected, while alternative occurrence details and counts remain in the
output.

## Units

The raw numeric value is always retained. Canonical conversion is performed
only for reviewed feature/source conventions. `canonical_value` and
`canonical_unit` are written only when they differ from the raw value/unit.
Inferred or unresolved policies retain `unit_status`; repetitive evidence text
remains in the versioned registry rather than every event row.

## Internal state, atomic commits, and processing telemetry

`.wearable_state.sqlite` remains the only non-feature object in the cleaned
output root. It stores source-file hashes, parser/registry versions,
participant status, committed feature checksums, and run-scoped processing
telemetry. It is operational state, not a physiological analysis product.

A participant is built under `.wearable_tmp`, validated, and swapped into place
as one directory. The state database is committed only after the new directory
is in place. A failed worker cannot partially append to a live feature CSV.

Telemetry through the dedicated module:

```text
wearable_project/processing/tracker.py
```

The tracker does not alter participant feature CSVs, feature-cleaning policies,
or incremental planning. The parser and cleaner expose counters that already
exist during processing; the parent process persists those counters in the
same hidden SQLite database.

### Two report scopes

Every report separates two quantities that must not be conflated.

**Run activity** describes only work performed by the selected invocation:

```text
monthly files discovered, planned, read, and completed
outer and Apple rows read
payloads decoded and payload failures
raw payload observations parsed
features touched
canonical rows written for touched features
duplicates and revisions reconciled
unresolved conflicts and invalid timestamps
unknown feature names and schema warnings
```

**Current snapshot** describes the complete committed output after the run:

```text
committed and Apple-empty participants
committed monthly files
feature files
canonical rows
represented source occurrences
total output size
largest participant output
```

For example, an unchanged no-op run correctly reports zero monthly files read,
while the current snapshot still reports all committed files and rows.

### Performance and failure telemetry

Each participant worker records:

```text
worker processing time
peak resident memory high-water mark
source bytes and files read
output size
worker exit code
error message and traceback when applicable
```

The run report includes wall-clock time, median and 95th-percentile participant
runtime and peak RSS, the slowest participant, the highest-memory participant,
and the largest participant output. On Linux, the memory ceiling is the smaller
of physical RAM and an active cgroup limit when one exists.

The safe-worker value is deliberately an advisory:

```text
min(CPU count - 1,
    floor(70% of effective memory limit / maximum observed worker RSS))
```

It is a conservative first estimate for the same machine and dataset shape,
not a guarantee. The first complete cohort run should still be monitored for
page cache, parent-process memory, filesystem pressure, and unusually large
future participants.

Automatic retries remain disabled. Deterministic source or schema errors should
not be hidden by immediately repeating the same operation. Failed participants
retain their previous committed outputs and are rebuilt on the next invocation.
The tracker records attempt number, retry count, worker exit code, error, and
traceback.

### Report commands

Print the complete summary from the processing command:

```bash
wearable-project process \
  --input /data/exported-at-latest \
  --output /data/cleaned_apple_healthkit \
  --workers 8 \
  --max-in-flight 8 \
  --json-summary > processing-report.json
```

Read the latest persisted report without processing data again:

```bash
wearable-project report \
  --output /data/cleaned_apple_healthkit
```

Machine-readable summary:

```bash
wearable-project report \
  --output /data/cleaned_apple_healthkit \
  --json > processing-report.json
```

Include every participant row, touched feature row, and exceptional diagnostic:

```bash
wearable-project report \
  --output /data/cleaned_apple_healthkit \
  --json \
  --include-details > processing-report-detailed.json
```

Select a historical run:

```bash
wearable-project report \
  --output /data/cleaned_apple_healthkit \
  --run-id RUN_ID \
  --json
```

The detailed records are stored in four additive tables:

| Table              | Purpose                                                    |
|--------------------|------------------------------------------------------------|
| `run_tracking`     | Run configuration, system resources, status, and wall time |
| `run_participants` | One participant plan/result row per invocation             |
| `run_features`     | One touched participant-feature result per invocation      |
| `run_diagnostics`  | Exceptional row/schema/parse diagnostics only              |

The pre-existing `runs`, `participants`, `source_files`, and `feature_outputs`
tables remain the authoritative incremental state.

Because telemetry did not exist during earlier runs, upgrading an existing
v0.1.1 output and immediately receiving a no-op skip can only report the
current snapshot plus the no-op activity. To populate complete parse, cleaning,
runtime, and worker-memory telemetry for the production acceptance run, perform
one explicit rebuild:

```bash
wearable-project process \
  --input /data/exported-at-latest \
  --output /data/cleaned_apple_healthkit \
  --mode rebuild \
  --workers 8 \
  --max-in-flight 8 \
  --json-summary > first-full-acceptance-report.json
```

## Commands

```bash
wearable-project --help
wearable-project process --help
wearable-project report --help
wearable-project registry
python -m wearable_project process --help
```

## Tests

```bash
pytest
```

The suite covers payload decoding, malformed metadata keys, sleep-state
preservation, paired blood pressure, boundary revisions, units, exact
waist-circumference duplicates, incremental/from-scratch equivalence,
historical corrections, strict cumulative-snapshot blocking, 
run/snapshot telemetry, persisted detailed reports, and
participant-level planning failures.

## License

MIT. See [`LICENSE`](LICENSE).
