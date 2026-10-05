# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Apple HealthKit exports of the HPP ("10K") cohort, turned into one CSV per participant per feature, plus a
non-destructive curated copy and HPP-style pandas loaders. `README.md` is the user-facing manual and is
authoritative for the public API; this file covers the cross-file invariants you need before changing code.

## Commands

```bash
python -m pip install -e ".[dev]"

# Tests need two sample roots; tests that touch them are skipped without these.
export WEARABLE_NATIVE_SAMPLE=/path/to/native_sample
export WEARABLE_CURATED_SAMPLE=/path/to/curated_sample

python -m pytest -rs                                  # -rs prints the skip reasons
python -m pytest tests/test_dataloader_contract.py
python -m pytest tests/test_curation_registry.py::test_name
python -m pytest -k "projection and not profile"
```

`pyproject.toml` sets `addopts = "-q"` and `testpaths = ["tests"]`. There is no conftest.py: each loader test
module defines its own `needs_sample` skip marker from the env vars above.

```bash
wearable-project --version                 # also: python -m wearable_project
wearable-project <command> --help
wearable-project curation-environment      # import source, versions, module integrity

# Acceptance against the real cohort roots (not part of pytest):
python -m wearable_project.utils.cohort_acceptance --participants 50 --seed 1 --out ~/acceptance

# Per-visit summary features (±1 year around HPP visits) from written daily and domain runs; outputs stay outside the repo:
python scripts/build_visit_features.py --daily-run RUN/daily --domain-run RUN/domain --out RUN/features

# Built-in help systems, usable as modules:
python -m wearable_project.utils.info plot_agp --json
python -c "from wearable_project import DataLoaders; print(DataLoaders.info('Sleep'))"
```

## Architecture

Three stages, two data roots, two SQLite state databases:

```
raw monthly exports ──process──▶ native root ──curate──▶ curated root
                                      │                        │
                                      └────── DataLoaders ─────┘
```

- `processing/` — streaming parser → feature-aware cleaning at native resolution → atomic per-participant
  commit. `writer.py` owns staging and `.wearable_state.sqlite`.
- `curation/` — reads the native root, appends curation columns, writes the curated root with
  `.wearable_curation_state.sqlite`. Never changes a native row, value, or column order.
- `DataLoaders/` — one thin subclass per feature over `_base.AppleHealthFeatureLoader`; all behavior lives in
  `_base.py`.
- `utils/` — statistics, summaries, domain metrics, plots, all going through the DataLoaders.

The HPP roots are already built, so most work is in `DataLoaders/` and `utils/`. Default roots are hardcoded in
`DataLoaders/_base.py` (`DEFAULT_NATIVE_ROOT`, `DEFAULT_CURATED_ROOT`) under
`.../LabData/Data/10K/aws_lab_files/third-party/`.

### Invariants that span files

- **Registries are the source of truth, not the code that reads them.** `processing/registry.py` (`SPECS`,
  `FeatureSpec`, `FeatureFamily`, `DedupStrategy`, unit convention applied while writing) and
  `curation/registry.py` (`_POLICIES`, `FeaturePolicy`, what is known about the stored unit after reviewing
  evidence). Behavior changes belong in a registry entry, not in a branch inside an engine.
- **`DataLoaders/_units.py` reconciles the two registries in exactly one place.** Curation may only *withdraw*
  processing's unit claim (declare it unknown); it never substitutes a unit of its own. Today this matters only
  for EnergyConsumed (kcal vs kJ). Do not widen this.
- **Curated columns are stored sparsely, restored densely.** A curation column is physically written only when
  some row needs a non-default value. `_base._DENSE_CURATED_DEFAULTS` restores `acquisition_method`
  ("unclassified"), `curation_status` ("pass"), `curation_flags` (""), `include_by_default` (True) for every
  row. Unit columns are deliberately *not* filled — missing means unresolved.
- **The state database is opened `immutable=1`, never `mode=ro`.** `mode=ro` would create `-shm`/`-wal` side
  files in the data tree. A non-empty `-wal` or `-journal` means a writer is active or crashed, and the loader
  refuses rather than read stale state (`_STATE_ACTIVE_SUFFIXES` in `_base.py`). Loading must never write into a
  data root.
- **Phase detection comes from which state file is present** (`_PHASE_STATE_FILES`), so a mislabeled root is
  caught instead of trusted. Trimmed sample trees have neither; detection returns `None` and no claim is made.
- **Dtypes are declared, not inferred** (`curation/schema.py`): whole numbers `Int64`, other numbers `float64`,
  identifiers/devices/units/sleep states categorical. Results must not depend on which participants a call read.
  Under pandas 2.x, `groupby` on a categorical includes unobserved categories unless `observed=True`.
- **Projections (`default` / `analysis` / `full`) are defined by column *roles*** in `curation/schema.py`, not by
  name lists at call sites. A new column gets a role; the projections follow.
- **`curation/schema.py` is imported lazily inside `_base._projection_schema()`** so importing one feature loader
  does not pull the whole curation package (its `__init__` eagerly loads policy, evidence and engine). Keep that
  import deferred.
- **Sum first, round last.** Interval totals (steps, distance, energy) are stored per interval and a row may hold
  a fractional share of a sample. Levels and rates are averaged, never summed. The feature's `FeatureFamily`
  says which it is.
- **Day boundaries are the participant's local calendar day**, derived from each row's `utc_offset_minutes`.
  Stored timestamps are UTC; `Date` is the event's `start_date` (for ActivitySummary, the export's UTC day key).
- **Every threshold and rule is a named constant** surfaced by `utils.info("defaults")` with its source. Do not
  inline a magic number.
- **A file is read exactly or not at all.** If reading would change what the file says, raise
  `DataLoaderReadError` naming the file and, where known, the row and value. Exceptions all live in
  `wearable_project/exceptions.py` and derive from `WearableProjectError` / `DataLoaderError`.

### Touching a feature

A feature is defined across several files; changing one alone leaves the project inconsistent:

1. `processing/registry.py` — `FeatureSpec` (family, measurement columns, dedup strategy, unit policy).
2. `curation/registry.py` — a `FeaturePolicy` via `_register(...)`, plus evidence refs and, where calibrated,
   an entry in `_CALIBRATION_AUDITS` / `POLICY_CALIBRATION_DECISIONS`.
3. `curation/guidance.py`, `curation/evidence.py`, `curation/decisions.py` — user-facing guidance, the evidence
   behind the policy, and the versioned human decision.
4. `DataLoaders/<Feature>Loader.py` — a subclass setting only `feature_name`, exported from
   `DataLoaders/__init__.py`.
5. `DataLoaders/info.py` and `utils/info.py` — the help text for the feature and any tool that covers it.

Registry and policy versions (`REGISTRY_VERSION`, `CURATION_REGISTRY_VERSION`, `POLICY_CONTRACT_VERSION`,
`SCHEMA_VERSION`) and the fingerprints in `curation/audit.py` gate rebuilds: a curated root built under a
different policy fingerprint loads with a warning. Bump them when the contract changes.

### Writing and runs

Nothing is written unless asked. `out=` writes a run folder that must be crash-safe, resumable (`resume=True`)
and byte-for-byte reproducible apart from the timestamps in `run.json`. Every summary and figure function must
accept that folder in place of the in-memory result.

Figures are matplotlib `Figure` objects and must never require a display. `min_participants=` must remove
suppressed aggregates from `figure.data` too, not just from the drawing; participant identifiers appear only
with `show_ids=True`.

## Conventions

- Every module starts with the docstring header `"""\nWearable Data Processing and Modeling project\n<what it
  does>\n"""`, often with a short rationale paragraph. Match it.
- Comments in this codebase explain *why* a non-obvious constraint exists (the `immutable=1` note is the model).
  Keep that density; do not add narration of what the code plainly does.
- Contract and policy types are `@dataclass(frozen=True, slots=True)`; enums subclass `str, Enum`.
- The version appears in `pyproject.toml`, `wearable_project/__init__.py` (`__version__`,
  `__release_label__`) and the README install line. Keep them in sync.
- Dependencies are pandas, tqdm, matplotlib only; pytest (`dev` extra) and pyarrow (Parquet tests) are optional.
  Python >= 3.10, pandas >= 2.0.
