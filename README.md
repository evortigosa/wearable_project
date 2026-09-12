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
│   ├── processing/
│   │   ├── parser.py
│   │   ├── registry.py
│   │   ├── cleaners.py
│   │   ├── resampling.py
│   │   ├── writer.py
│   │   └── pipeline.py
│   ├── cli.py
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

## Internal state and atomic commits

`.wearable_state.sqlite` contains source-file hashes, parser/registry versions,
participant status, and committed feature checksums. It is implementation
state, not an analysis product.

A participant is built under `.wearable_tmp`, validated, and swapped into place
as one directory. The state database is committed only after the new directory
is in place. A crash leaves the previous commit recoverable and forces a safe
participant rebuild on the next run.

## Commands

```bash
wearable-project --help
wearable-project process --help
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
historical corrections, and strict cumulative-snapshot blocking.

## License

MIT. See [`LICENSE`](LICENSE).
