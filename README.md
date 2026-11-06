# wearable-project

`wearable-project` turns Apple HealthKit and Google Fitness exports from the Human Phenotype Project (HPP, the
"10K" cohort) into verified, analysis-ready participant/feature tables.

For most researchers, the package starts and ends with **DataLoaders**: one public `<Feature>Loader` class per
feature, pandas output with stable schemas and dtypes, native or curated data, and state-backed integrity checks.
Processing and curation are included for the small number of data engineers who maintain the cohort products.

Developed at the Segal Lab, Department of Computer Science and Applied Mathematics, Weizmann Institute of Science.
A project overview is available in [this presentation](https://docs.google.com/presentation/d/1dbA7ByPcUTwcUZf5fLVpNBrlMeY5RP5Myy0WmeKwETI/edit?usp=drivesdk).

| Package                  | Sources                            |                                       Public feature loaders | Default phase | Python |
|--------------------------|------------------------------------|-------------------------------------------------------------:|---------------|--------|
| `wearable-project 0.4.0` | Apple HealthKit and Google Fitness | 96 unique modules: 33 Apple + 67 Google, with 4 shared names | `curated`     | 3.10+  |

## Contents

- [Install](#install)
- [Start here: load wearable data](#start-here-load-wearable-data)
- [Choose a source, feature and phase](#choose-a-source-feature-and-phase)
- [Available features and HPP cohort coverage](#available-features-and-hpp-cohort-coverage)
- [DataLoader reference](#dataloader-reference)
- [Source-specific scientific contracts](#source-specific-scientific-contracts)
- [Statistics, metrics, figures and reports](#statistics-metrics-figures-and-reports)
- [Processing and curation for data engineers](#processing-and-curation-for-data-engineers)
- [Cohort acceptance and release qualification](#cohort-acceptance-and-release-qualification)
- [Repository layout and developer documentation](#repository-layout-and-developer-documentation)
- [Running the tests](#running-the-tests)

## Install

Install the release wheel in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install ./wearable_project-0.4.0-py3-none-any.whl
```

For development from a checkout:

```bash
python -m pip install -e ".[dev]"
```

Runtime requirements are Python 3.10 or newer, pandas 2.0 or newer, matplotlib 3.5 or newer, and tqdm 4.66 or
newer. `pyarrow` is optional and is needed only for Parquet exports and their tests.

Check the installed artifact from a directory outside the source checkout:

```bash
wearable-project --version
# wearable-project 0.4.0

python - <<'PY'
from wearable_project import __version__
assert __version__ == "0.4.0"
print("wearable-project", __version__)
PY
```

## Start here: load wearable data

The permanent HPP native and curated roots are already configured as defaults in the laboratory environment. A
normal analysis therefore needs only a feature class and, for Google Fitness, an explicit source when the feature
name is shared with Apple.

```python
from wearable_project import DataLoaders
from wearable_project.DataLoaders import StepCountLoader, WalkingLoader

# Existing Apple loaders remain backward compatible: omitted source means Apple HealthKit.
apple_steps = StepCountLoader(state_validation="required").get_data(
    registration_codes=["10K_1235738253"],
    start_date="2024-01-01",
    end_date="2024-12-31 23:59:59",
)

# The same public class reads Google Fitness when source is explicit.
google_steps = StepCountLoader(
    source="google_fitness",
    state_validation="required",
).get_data(registration_codes=["10K_8790614327"])

# Google-only loaders default to Google Fitness, although explicit source is clearer in shared code.
google_walking = WalkingLoader(
    source="google_fitness",
    state_validation="required",
).get_data(projection="analysis")

apple_steps.df.head()
google_steps.load_report["wearable_source"]             # "google_fitness"
google_steps.load_report["state_verified"]              # True
google_steps.load_report["curation_contract_validated"] # True for Google curated schema v2
```

Every call returns a `LoaderData` container:

```python
data = google_steps
data.df                   # feature rows, indexed by RegistrationCode and Date
data.df_metadata          # per-participant returned/stored rows, date span and verification
data.df_columns_metadata  # per-column role, description, dtype, non-null count, unit and dense default
data.load_report          # source, root, phase, files, filters, projection and verification provenance
```

> **One load always represents one source.** Version 0.4.0 does not merge Apple and Google observations, select a
> preferred source, deduplicate across sources, reconcile source intervals, or map both Sleep ontologies into one
> representation. Those are later harmonization decisions, not DataLoader behavior.

## Choose a source, feature and phase

A loader is defined by three independent choices:

1. **Source** — `"apple_healthkit"` or `"google_fitness"`.
2. **Feature** — the top-level `<Feature>Loader` class.
3. **Phase** — `"curated"` by default, or `"native"` for the source-faithful product.

### Source defaults and compatibility

| Loader kind                | Example                                                          | Omitted `source` | Other source                   |
|----------------------------|------------------------------------------------------------------|------------------|--------------------------------|
| Existing Apple-only loader | `HeartRateLoader`                                                | Apple HealthKit  | rejected during construction   |
| Shared loader name         | `StepCountLoader`, `SleepLoader`, `HeightLoader`, `WeightLoader` | Apple HealthKit  | pass `source="google_fitness"` |
| Google-only loader         | `WalkingLoader`, `CaloriesLoader`, `BikingLoader`                | Google Fitness   | Apple is rejected              |

The four shared feature names have one public module and one class each; there is no competing
`google_fitness.StepCountLoader`. Existing laboratory imports remain valid:

```python
from wearable_project.DataLoaders.StepCountLoader import StepCountLoader
```

For new code, package-level imports are shorter:

```python
from wearable_project.DataLoaders import StepCountLoader, WearableSource

loader = StepCountLoader(source=WearableSource.GOOGLE_FITNESS)
```

### Native and curated phases

| Phase     | Meaning                                                                                                                                             |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `native`  | Source-faithful normalized rows at their retained native time support; no ordinary loading-time resampling, source selection or curation filtering  |
| `curated` | The same rows, order, native prefix and values, plus reviewed annotations, default-inclusion decisions and accepted unit/provenance interpretations |

Curated is the default because it exposes the review decisions without deleting evidence. Loaders still return every
row unless `default_inclusion_only=True` is requested explicitly.

### Discover the catalog and inspect contracts

```python
from wearable_project import DataLoaders

DataLoaders.available_features()                              # 33 Apple features (legacy default)
DataLoaders.available_features(source="apple_healthkit")      # 33
DataLoaders.available_features(source="google_fitness")       # 67
DataLoaders.available_sources("StepCount")                    # both sources
DataLoaders.available_sources("HeartRate")                    # Apple only

print(DataLoaders.info("StepCount"))                          # Apple curated contract
print(DataLoaders.info("StepCount", source="google_fitness")) # Google curated contract
print(DataLoaders.info("Walking", source="google_fitness", phase="native"))
```

`info()` is static: it reads executable package contracts and never opens a data root. `profile()` describes one
actual root from its state database and file metadata without parsing feature CSVs:

```python
from wearable_project.DataLoaders import HeartRateLoader

profile = HeartRateLoader(state_validation="required").profile()
print(profile)
profile.summary       # rows, files, bytes, integrity and curated outcome coverage
profile.participants  # one row per participant
```

## Available features and HPP cohort coverage

The final full-cohort acceptance run on **5 November 2026** found every public feature in both native and curated
roots, with identical participant and row counts across phases. The counts below are therefore per phase.

| Source          | Public features | Feature files |        Rows | Native size | Curated size | Largest participant coverage    |
|-----------------|----------------:|--------------:|------------:|------------:|-------------:|---------------------------------|
| Apple HealthKit |              33 |        23,335 | 451,653,688 |    103.9 GB |     107.9 GB | `StepCount`: 4,006 participants |
| Google Fitness  |              67 |        32,707 |  13,476,849 |     1.64 GB |      1.89 GB | `Calories`: 5,558 participants  |

Bars in the participant charts are linear. Row charts use a logarithmic scale because feature sizes span many
orders of magnitude. These are cohort snapshots, not hard-coded loader limits; use `profile()` for the current root.

### Apple HealthKit coverage

Participants with each feature:

```text
StepCount              ██████████████████████████████████████████████████ 4,006
ActiveEnergyBurned     ████████████████████████ 1,888
BasalEnergyBurned      ██████████████████████ 1,744
Height                 ████████████████████▌ 1,639
DistanceWalkingRunning ████████████████████ 1,602
FlightsClimbed         ███████████████████ 1,570
Weight                 ██████████████████ 1,553
ActivitySummary        ██████████████ 1,085
HeartRate              █████████████ 1,055
Sleep                  ████████████ 1,032
RestingHeartRate       █████████ 753
Vo2Max                 ███████ 573
HeartRateVariability   ███████ 568
WalkingHeartRate       ██████ 540
Mindful                ██████ 514
OxygenSaturation       ██████ 460
RespiratoryRate        █████ 425
DailyDistanceCycling   █████ 405
Electrocardiogram      ████ 349
BMI                    ████ 324
DailyDistanceSwimming  ███ 253
BodyFatPercentage      ███ 224
LeanBodyMass           ██ 157
EnergyConsumed         █ 106
Protein                █ 103
Carbohydrates          █ 101
BloodPressure          █ 97
TotalFat               █ 95
WaistCircumference     ▋ 52
BloodGlucose           ▌ 39
BodyTemperature        ▎ 20
BloodAlcoholContent    ▏ 2
PeakFlow               ▏ 1
```

<details>
<summary>Rows per feature (log scale)</summary>

```text
HeartRate              ███████████████████████████████████████████████████ 275,540,794
StepCount              ███████████████████████████████████████████████ 60,014,907
DistanceWalkingRunning ██████████████████████████████████████████████ 40,879,374
BasalEnergyBurned      █████████████████████████████████████████████ 28,792,402
ActiveEnergyBurned     ████████████████████████████████████████████ 19,812,185
FlightsClimbed         ██████████████████████████████████████████ 9,876,659
Sleep                  █████████████████████████████████████████ 5,469,021
RespiratoryRate        ████████████████████████████████████████ 3,604,591
HeartRateVariability   ███████████████████████████████████████ 2,250,623
OxygenSaturation       ██████████████████████████████████████ 1,672,342
ActivitySummary        █████████████████████████████████████▏ 1,530,301
RestingHeartRate       ███████████████████████████████████ 570,414
BloodGlucose           ███████████████████████████████████ 546,853
WalkingHeartRate       ██████████████████████████████████ 384,732
Weight                 ███████████████████████████████ 113,700
Vo2Max                 ██████████████████████████████ 92,210
DailyDistanceCycling   █████████████████████████████ 68,704
BMI                    ████████████████████████████ 68,337
BodyFatPercentage      ███████████████████████████ 44,719
EnergyConsumed         ███████████████████████████ 44,174
Protein                ███████████████████████████ 39,091
LeanBodyMass           ███████████████████████████ 38,992
Carbohydrates          ███████████████████████████ 38,855
TotalFat               ███████████████████████████ 38,101
Mindful                ███████████████████████████ 36,761
DailyDistanceSwimming  ██████████████████████████ 28,248
BodyTemperature        ██████████████████████████ 26,469
Height                 ████████████████████████ 12,080
BloodPressure          ████████████████████████ 10,346
Electrocardiogram      ██████████████████████ 6,519
BloodAlcoholContent    ████████████████ 835
WaistCircumference     ██████████████ 346
PeakFlow               ██ 3
```

</details>

### Google Fitness coverage

Participants with each feature:

```text
Calories                      ██████████████████████████████████████████████████ 5,558
StepCount                     ████████████████████████████████████████████████▌ 5,394
Weight                        ████████████████████████████████████████████████ 5,340
Distance                      ███████████████████████████████████████████████ 5,267
UnknownActivity               █████████████████████ 2,354
Walking                       ████████████████████ 2,243
InVehicle                     ████████████████████ 2,229
Running                       ██████████████ 1,633
Biking                        ██████████ 1,100
Still                         ██████████ 1,093
OtherActivity                 ▌ 61
Pilates                       ▎ 34
Sleep                         ▎ 34
StrengthTraining              ▎ 29
Calisthenics                  ▎ 27
Swimming                      ▎ 21
BikingMountain                ▏ 20
Weightlifting                 ▏ 20
WalkingFitness                ▏ 18
SwimmingPool                  ▏ 17
Yoga                          ▏ 17
Aerobics                      ▏ 14
Elliptical                    ▏ 11
StairClimbing                 ▏ 11
Hiking                        ▏ 10
IntervalTrainingHighIntensity ▏ 10
WalkingTreadmill              ▏ 10
Meditation                    ▏ 9
RunningJogging                ▏ 9
RunningTreadmill              ▏ 8
BikingSpinning                ▏ 7
CircuitTraining               ▏ 7
Gardening                     ▏ 7
BikingUtility                 ▏ 6
Gymnastics                    ▏ 6
IntervalTraining              ▏ 6
BikingStationary              ▏ 5
Crossfit                      ▏ 5
Dancing                       ▏ 5
Tennis                        ▏ 5
BikingRoad                    ▏ 4
MartialArts                   ▏ 4
SwimmingOpenWater             ▏ 4
BikingHand                    ▏ 3
Boxing                        ▏ 3
Ergometer                     ▏ 2
Kayaking                      ▏ 2
Rowing                        ▏ 2
StairClimbingMachine          ▏ 2
Volleyball                    ▏ 2
WalkingNordic                 ▏ 2
Zumba                         ▏ 2
Basketball                    ▏ 1
GuidedBreathing               ▏ 1
Handball                      ▏ 1
Height                        ▏ 1
HorsebackRiding               ▏ 1
JumpRope                      ▏ 1
KickScooter                   ▏ 1
Kickboxing                    ▏ 1
RockClimbing                  ▏ 1
RowingMachine                 ▏ 1
RunningSand                   ▏ 1
SkatingCross                  ▏ 1
SkiingCrossCountry            ▏ 1
VolleyballBeach               ▏ 1
VolleyballIndoor              ▏ 1
```

<details>
<summary>Rows per feature (log scale)</summary>

```text
Calories                      ██████████████████████████████████████████████████ 4,059,476
Walking                       █████████████████████████████████████████████████ 2,633,002
StepCount                     ████████████████████████████████████████████████ 2,174,908
UnknownActivity               ███████████████████████████████████████████████ 1,710,270
Distance                      ██████████████████████████████████████████████ 1,369,648
Still                         █████████████████████████████████████████████ 930,098
InVehicle                     ███████████████████████████████████████████ 460,997
Weight                        ████████████████████████████████████ 61,939
Running                       ███████████████████████████████████ 51,558
Biking                        ████████████████████████████████▌ 19,816
OtherActivity                 ████████████████████████ 1,540
Sleep                         ███████████████████████ 1,173
StrengthTraining              ███████████████████ 286
Pilates                       █████████████████▌ 209
Yoga                          █████████████████ 198
RunningTreadmill              █████████████████ 190
Calisthenics                  █████████████████ 177
BikingMountain                ████████████████▌ 154
StairClimbing                 ████████████████ 118
WalkingFitness                ████████████████ 115
Weightlifting                 ███████████████ 99
BikingSpinning                ███████████████ 81
SwimmingPool                  ███████████████ 78
Elliptical                    ██████████████ 69
Swimming                      ██████████████ 66
CircuitTraining               ██████████████ 63
IntervalTrainingHighIntensity ██████████████ 58
Aerobics                      █████████████ 51
WalkingTreadmill              █████████████ 48
BikingStationary              █████████████ 47
Meditation                    ████████████ 39
Hiking                        ███████████ 30
RunningJogging                ███████████ 26
WalkingNordic                 ██████████▌ 24
Crossfit                      ██████████ 22
BikingUtility                 ██████████ 21
MartialArts                   ██████████ 21
Tennis                        ██████████ 20
IntervalTraining              █████████ 15
BikingRoad                    ████████ 10
Gymnastics                    ████████ 10
Gardening                     ███████ 9
Zumba                         ███████ 8
Basketball                    ██████ 7
Dancing                       ██████ 7
Ergometer                     ██████ 6
SwimmingOpenWater             ██████ 6
Kayaking                      █████ 5
BikingHand                    ███ 3
Boxing                        ███ 3
Rowing                        ███ 3
StairClimbingMachine          ███ 3
Volleyball                    ███ 3
Height                        ██ 2
RockClimbing                  ██ 2
GuidedBreathing               ▏ 1
Handball                      ▏ 1
HorsebackRiding               ▏ 1
JumpRope                      ▏ 1
KickScooter                   ▏ 1
Kickboxing                    ▏ 1
RowingMachine                 ▏ 1
RunningSand                   ▏ 1
SkatingCross                  ▏ 1
SkiingCrossCountry            ▏ 1
VolleyballBeach               ▏ 1
VolleyballIndoor              ▏ 1
```

</details>

`UnknownActivity` is an accepted Google activity label and has a public loader. `UnregisteredActivity` is a separate
exceptional fallback for future, unknown exporter labels; it is intentionally not part of the public catalog, and
`cohort_acceptance` surfaces it for review rather than silently accepting it.

## DataLoader reference

### Construct a loader

```python
from wearable_project.DataLoaders import HeartRateLoader, StepCountLoader

apple = HeartRateLoader(
    phase="curated",
    state_validation="required",
)

google = StepCountLoader(
    source="google_fitness",
    phase="native",
    root="/path/to/another/google/native/root",
    state_validation="required",
)
```

| Constructor argument          | Meaning                                                                                                    |
|-------------------------------|------------------------------------------------------------------------------------------------------------|
| `source`                      | `"apple_healthkit"` or `"google_fitness"`; omission uses the class compatibility default described above   |
| `phase`                       | `"curated"` (default) or `"native"`                                                                        |
| `root`                        | Override the active root for the selected phase                                                            |
| `native_root`, `curated_root` | Override one or both phase-specific roots while keeping normal phase selection                             |
| `state_validation`            | `"auto"` verifies when state exists; `"required"` fails without state; `"off"` skips per-file state checks |

Source identity is always checked for managed roots, even when `state_validation="off"`. Google curated roots must
also provide a compatible `.curation_contract.json`, because sparse defaults and virtual fields cannot be interpreted
safely without it.

### Read rows with `get_data()`

```python
result = apple.get_data(
    registration_codes=["10K_1235738253", "8790614327"], # prefixed or raw IDs are accepted
    start_date="2024-01-01",
    end_date="2024-12-31 23:59:59",
    projection="analysis",
    default_inclusion_only=False,
    max_rows=5_000_000,
)
```

| `get_data()` argument            | Meaning                                                                                          |
|----------------------------------|--------------------------------------------------------------------------------------------------|
| `registration_codes` (`reg_ids`) | One participant or an iterable; `None` discovers all participants with that feature              |
| `start_date`, `end_date`         | Inclusive bounds applied to the feature's mechanical `Date` anchor; bare dates mean midnight UTC |
| `columns` (`cols`)               | Return exactly these logical columns; overrides the named projection                             |
| `projection`                     | `"default"`, `"analysis"` or `"full"`                                                            |
| `default_inclusion_only`         | Curated phase only: retain rows whose reviewed decision includes them by default                 |
| `sort_index`                     | Sort by `RegistrationCode` and `Date` (default `True`)                                           |
| `max_rows`                       | Per-call safety ceiling; default 50,000,000; `None` disables it explicitly                       |

The resulting `DataFrame` has a two-level index:

```text
RegistrationCode
Date
```

`Date` is a source-qualified mechanical anchor, not a promise that every feature has the same temporal meaning.
Duplicate index keys are legitimate—for example, parallel Google StepCount alternatives can share an outer timestamp.

### Result metadata

`LoaderData` keeps the data and its audit trail together:

- `df` contains only requested rows and projected columns.
- `df_metadata` reports returned rows, stored rows, date span, state verification, policy currency and acquisition
  coverage for each participant present in the result.
- `df_columns_metadata` reports every returned column's declared role, description, dtype, non-null count, registry
  unit and reconstructed default where applicable.
- `load_report` records source, phase, resolved root, source identity, participants, files, filters, projections,
  withheld/undeclared columns, state verification, size estimates and source-specific contract information.

`LoaderData.metadata` is a deprecated alias of `load_report` retained for older callers.

### Projections and protected provenance

| Projection | Returns                                                                                                                                                                  |
|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `default`  | Core timing, measurements/states, interpretation context, reviewed provenance and curation decisions; persistent identifiers and bulky/free-text provenance are withheld |
| `analysis` | `default` plus source-specific audit, device, algorithm and lineage fields useful for stratified analysis; persistent identifiers remain withheld                        |
| `full`     | Every logical column, including restricted identifiers; Google curated data also include reconstructed sparse defaults and contract-defined virtual fields               |

`load_report["columns_withheld"]` lists what the projection omitted. Use `full` only for authorized provenance work
inside the protected environment; do not use it casually for derived exports.

### Native fidelity and curated inclusion

Ordinary loading does not:

- resample or fill time slots;
- merge devices or sources;
- deduplicate rows beyond what native processing already committed;
- collapse Google StepCount alternatives;
- infer local time when offset evidence is absent;
- assign a canonical unit that curation deliberately withheld.

Curation is non-destructive. Every curated CSV retains the same row count, row order, native prefix and parsed native
values as its corresponding native CSV. Curation adds decisions and interpretations.

`curation_status` and `include_by_default` answer different questions:

- `curation_status`: `pass`, `review`, or `exclude_default`;
- `include_by_default`: the reviewed binary decision used by `default_inclusion_only=True`.

```python
from wearable_project.DataLoaders import StepCountLoader

all_google_rows = StepCountLoader(source="google_fitness").get_data()
default_google_rows = StepCountLoader(source="google_fitness").get_data(
    default_inclusion_only=True
)
```

Apple curated files can omit dense logical defaults physically; the loader reconstructs the accepted values such as
`acquisition_method="unclassified"`, `curation_status="pass"`, no flags and inclusion by default.

Google curated schema v2 stores a compact physical decision suffix and exposes deterministic virtual fields from the
validated root contract. Missing or blank `curation_status`, `curation_flags` and `include_by_default` mean
`"pass"`, no flags and `True`. `load_report` distinguishes stored, reconstructed, virtual and available columns.

### Exact reads, stable dtypes and verification

The loader is deliberately stricter than a normal `pandas.read_csv()` call:

- only empty cells become missing; literal text such as `"None"` stays text;
- floating-point values use round-trip parsing;
- duplicate headers, malformed row widths, invalid numeric/Boolean values and malformed timestamps fail loudly;
- declared dtypes are applied after participant concatenation, so one-participant and cohort loads agree;
- controlled vocabularies are categorical, whole-number fields use nullable `Int64`, Booleans use pandas Boolean
  dtypes, persistent identifiers remain strings, and instants remain timezone-aware UTC;
- Apple StepCount remains floating point because fractional attributed shares are legitimate;
- Google StepCount values and payload indices use nullable `Int64`.

Before date, inclusion or projection filters are applied, managed files are checked against source-specific state:

- every file: filename, byte size, SHA-256 and row count;
- Apple curated files: curation status, inclusion, acquisition method and flag counts;
- Google curated files: root contract, policy fingerprint, active physical columns, status/inclusion/provenance,
  StepCount-role and unit counters, and canonical-value count.

The state database is opened read-only. Loading refuses active SQLite sidecars, source/phase conflicts, unsafe
participant paths and malformed managed-state evidence. A mismatch raises `DataLoaderStateError` before data are
returned.

### Time semantics

HealthKit and Google Fitness define feature semantics, not one universal sample cadence. The loader preserves the
stored timestamps and does not hard-code assumptions such as "StepCount is always hourly" or "HeartRate is always
one-minute data".

| Source / feature family                      | Mechanical `Date` anchor | Interpretation                                                                         |
|----------------------------------------------|--------------------------|----------------------------------------------------------------------------------------|
| Apple ordinary point/interval features       | `start_date`             | Event/interval start in UTC                                                            |
| Apple `ActivitySummary`                      | `datetime`               | Outer summary timestamp; not silently converted into a canonical local day             |
| Google activities, Sleep, Calories, Distance | `start_date`             | Source interval start in UTC                                                           |
| Google Weight, Height                        | `end_date`               | Source reading-interval end                                                            |
| Google StepCount                             | `outer_datetime`         | Outer summary timestamp shared across versions/alternatives; not a canonical event day |

Google `summary_date` remains a timezone-naive calendar label. It is not converted to midnight UTC and is not used
as the default index.

### Local time and harmonized values

Derived helpers return a new `LoaderData` and preserve the stored columns:

```python
from wearable_project.DataLoaders import SleepLoader, WeightLoader

apple_sleep = SleepLoader().get_data().with_local_time()
apple_weight = WeightLoader().get_data().with_harmonized_values()
google_weight = WeightLoader(source="google_fitness").get_data().with_harmonized_values()
```

- `with_local_time()` uses Apple row-level `utc_offset_minutes` to add timezone-naive local wall-clock columns.
  Google products do not retain equivalent row-level offsets, so the helper refuses to invent them.
- `with_harmonized_values()` adds `harmonized_value`, `harmonized_unit` and `harmonized_unit_source` according to
  the selected source's accepted policy.
- Google StepCount resolves to `count`.
- Google Calories, Distance, Weight and Height remain unresolved until the exporter-to-official-type mapping is
  accepted; proposed official units are evidence, not canonical output.
- Google activity rows can contain `calories`, `distance` and `quantity` together, so the singular harmonization
  helper is intentionally inapplicable.

### Profile before loading and control large requests

`profile()` is the normal first step for dense features:

```python
from wearable_project.DataLoaders import HeartRateLoader

loader = HeartRateLoader(state_validation="required")
profile = loader.profile()
print(profile)

# Filter or batch before materializing a very large frame.
participant_batch = profile.participants.index[:50].tolist()
data = loader.get_data(registration_codes=participant_batch)
```

`get_data()` refuses requests above 50,000,000 rows by default and raises `DataLoaderSizeError`. Where state provides
exact counts, refusal occurs before a CSV is parsed. Full-cohort Apple HeartRate (275,540,794 rows) and StepCount
(60,014,907 rows) exceed this default; load them in participant/date batches or pass an explicit `max_rows` override
for a suitably provisioned job.

```python
loader.get_data(max_rows=150_000_000)  # explicit high-memory decision for this call
loader.get_data(max_rows=None)         # explicit removal of the per-call ceiling
```

The conservative package default is intentional and should not be raised merely because one cluster has high memory.

### Errors

All loader errors derive from `DataLoaderError`:

| Error                          | Raised when                                                                          |
|--------------------------------|--------------------------------------------------------------------------------------|
| `DataLoaderConfigurationError` | A phase, source, filter, projection, column or option is invalid                     |
| `DataLoaderSourceError`        | A feature is unavailable for the source, or a managed root belongs to another source |
| `DataLoaderPathError`          | The configured root is unavailable or unsafe                                         |
| `DataLoaderReadError`          | A CSV cannot be read exactly, or permissions prevent reading it                      |
| `DataLoaderStateError`         | Managed files, state, policy or root contracts disagree                              |
| `DataLoaderSizeError`          | A request exceeds `max_rows`                                                         |

## Source-specific scientific contracts

### Apple HealthKit

The 33 established public Apple loaders preserve production behavior from earlier releases.

- Quantity samples retain their original point or interval support; no fixed cadence is inferred from the feature
  name or unit.
- Cumulative quantities such as StepCount, distance and energy are interval totals. Apple StepCount is `float64`
  because attributed/reconciled rows can be fractional.
- HeartRate and other discrete quantities can be points or short source-defined intervals. `beats/min` is a unit,
  not a promise that each row covers one minute.
- Sleep is a variable-duration state interval. `ActivitySummary` is a calendar-summary product with separate time
  semantics. ECG records retain session timing and waveform sampling metadata.
- Row-level UTC offsets support `with_local_time()`.
- Apple curation can physically materialize canonical values, units, unit evidence and unit epochs where policy
  resolution succeeds.

Use `DataLoaders.info(feature, source="apple_healthkit", include_evidence=True)` for the full row meaning, unit
policy, caveats and evidence for one feature.

### Google Fitness

Google output features fall into five loader schema families:

| Family             | Features                      | Row meaning                                                   | Main measurements/state                                 | `Date` anchor    |
|--------------------|-------------------------------|---------------------------------------------------------------|---------------------------------------------------------|------------------|
| Activity intervals | 61 registered activity labels | One retained activity/state interval                          | `calories`, `distance`, `quantity`, tracking/provenance | `start_date`     |
| Sleep              | `Sleep`                       | One retained sleep-state interval                             | categorical `value` plus optional activity measurements | `start_date`     |
| Scalar intervals   | `Calories`, `Distance`        | One source interval total/value                               | `value`, source day label, version                      | `start_date`     |
| Anthropometrics    | `Weight`, `Height`            | One point or retained source interval                         | `value`, adding application, version                    | `end_date`       |
| Step count         | `StepCount`                   | One version-1 scalar or one version-2 aggregation alternative | `value`, `raw_value`, aggregation/app/device context    | `outer_datetime` |

#### Google StepCount representations

The loader keeps every representation. Curation adds a reviewed `step_count_role` rather than deleting alternatives:

```text
legacy_v1
fit_app_preferred
merge_fallback
third_party_fallback
parallel_nonpreferred
ambiguous_parallel
```

`default_inclusion_only=True` selects the reviewed default subset. It does not rewrite the curated product or merge
version 1 with version 2.

#### Google units

| Feature    | Current accepted loader behavior                               |
|------------|----------------------------------------------------------------|
| StepCount  | Canonical/harmonized `count`                                   |
| Calories   | Candidate `kcal` documented; canonical mapping withheld        |
| Distance   | Candidate `m` documented; canonical mapping withheld           |
| Weight     | Candidate `kg` documented; canonical mapping withheld          |
| Height     | Candidate `m` documented; canonical mapping withheld           |
| Activities | Multiple measurement dimensions; no singular harmonized scalar |

Detailed contracts are available in
[`docs/GOOGLE_FITNESS_NATIVE_DATALOADERS.md`](docs/GOOGLE_FITNESS_NATIVE_DATALOADERS.md) and
[`docs/GOOGLE_FITNESS_CURATED_DATALOADERS.md`](docs/GOOGLE_FITNESS_CURATED_DATALOADERS.md).

## Statistics, metrics, figures and reports

The source-aware DataLoaders read Apple and Google independently. The higher-level modules below remain **Apple
HealthKit analytical contracts** in 0.4.0 because they depend on Apple local-time evidence, feature semantics and
unit policies. They do not merge sources or reinterpret Google data as Apple data.

| Module            | Main outputs                                                                        |
|-------------------|-------------------------------------------------------------------------------------|
| `data_statistics` | Coverage, participant-day statistics, valid-day rules and reproducible written runs |
| `data_summaries`  | Adherence, participant/cohort summaries and temporal patterns                       |
| `domain_metrics`  | Sleep-by-night, CGM consensus metrics and ambulatory glucose profiles               |
| `data_plots`      | Traceable matplotlib figures and PDF reports                                        |

```python
from wearable_project import utils
from wearable_project.utils import data_statistics as ds
from wearable_project.utils import data_summaries as sm
from wearable_project.utils import domain_metrics as dm
from wearable_project.utils import data_plots as dp

# Apple HealthKit analytical workflow.
daily = ds.compute_daily_statistics(
    "curated",
    features=["StepCount", "HeartRate"],
    participants=["10K_1235738253"],
)
summaries = sm.summarize(daily)
metrics = dm.compute_domain_metrics("curated", participants=["10K_1235738253"])
figure = dp.plot_daily_values(daily, "StepCount", participant="10K_1235738253")
dp.report(daily, "report.pdf", domain=metrics)

utils.info("compute_daily_statistics")
utils.info("valid days")
utils.info("defaults")
```

Key analytical rules:

- days are participant-local Apple calendar days recovered from each row's UTC offset;
- extensive totals are summed and split at midnight in proportion; intensive levels are summarized, never summed;
- overlapping device time is counted once;
- summaries use feature-specific valid-day rules;
- results stay in memory unless `out=` or `path=` is supplied;
- written runs are crash-safe, resumable and reproducible apart from timestamps in `run.json`.

At cohort scale, compute once into a run folder and use workers:

```bash
python -m wearable_project.utils.data_statistics daily \
  --phase curated --out runs/daily --workers 8
```

## Processing and curation for data engineers

Most users should not rebuild the permanent products. This section is for maintainers working from the cumulative
monthly HPP export.

```text
mixed raw export
   ├── process apple-healthkit ──▶ managed Apple native root ──▶ curated Apple root
   └── process google-fitness  ──▶ managed Google native root ─▶ curated Google root
                                              │
                                     source-aware DataLoaders
```

Each processor reads only its own rows from the mixed export. Native processing writes one file per participant and
feature and records lineage, counts and hashes in `.wearable_state.sqlite`. Curation writes an independent root,
preserves all native rows/values, and records decisions in `.wearable_curation_state.sqlite`.

### Apple HealthKit pipeline

```bash
RAW=/path/to/exported-at-YYYY-MM
APPLE_NATIVE=/path/to/cleaned_apple_healthkit
APPLE_CURATED=/path/to/curated_apple_healthkit

wearable-project process apple-healthkit \
  --input "$RAW" --output "$APPLE_NATIVE" \
  --workers 32 --max-in-flight 32 \
  --snapshot-policy strict-cumulative \
  --verify-existing-hashes --json-summary

wearable-project curate apple-healthkit \
  --input-native "$APPLE_NATIVE" --output "$APPLE_CURATED" \
  --workers 32 --max-in-flight 32 \
  --verify-existing-hashes --json-summary
```

Apple also provides the combined command:

```bash
wearable-project run apple-healthkit \
  --input "$RAW" \
  --native-output "$APPLE_NATIVE" \
  --curated-output "$APPLE_CURATED" \
  --workers 32 --max-in-flight 32 \
  --snapshot-policy strict-cumulative \
  --verify-existing-hashes --json-summary
```

### Google Fitness pipeline

```bash
RAW=/path/to/exported-at-YYYY-MM
GOOGLE_NATIVE=/path/to/cleaned_google_fitness
GOOGLE_CURATED=/path/to/curated_google_fitness

wearable-project process google-fitness \
  --input "$RAW" --output "$GOOGLE_NATIVE" \
  --workers 32 --max-in-flight 32 \
  --snapshot-policy strict-cumulative \
  --verify-existing-hashes --json-summary

wearable-project curate google-fitness \
  --input-native "$GOOGLE_NATIVE" --output "$GOOGLE_CURATED" \
  --workers 32 --max-in-flight 32 \
  --verify-native-hashes --verify-existing-hashes --json-summary
```

There is no combined `run google-fitness` command in 0.4.0; run the two explicit stages above.

### Operational guarantees and inspection commands

- Runs are participant-transactional and atomic. Workers stage outputs; the parent process publishes a complete
  participant directory and commits state only after validation.
- Unchanged participants are skipped; changed participants are recomputed from their complete retained source
  history. `--mode rebuild` rebuilds everything.
- `strict-cumulative` blocks missing committed months and rebuilds when historical input changes.
- Native and curated roots must be distinct and must not be nested.
- `process-scan <source>` is read-only and reports the next plan without creating or changing files.
- `--json-summary` emits the complete machine-readable run report.

```bash
wearable-project process-scan apple-healthkit --input "$RAW" --output "$APPLE_NATIVE"
wearable-project process-scan google-fitness --input "$RAW" --output "$GOOGLE_NATIVE"
wearable-project processing-environment apple-healthkit
wearable-project processing-environment google-fitness
wearable-project curation-environment apple-healthkit
wearable-project curation-environment google-fitness
```

Use `wearable-project <command> --help` for all options. `python -m wearable_project` is equivalent to the console
command.

## Cohort acceptance and release qualification

`cohort_acceptance` is the full-product DataLoader gate. It is source-aware and complements the unit/representative
sample test suite.

It:

1. profiles every public feature in native and curated state without parsing CSVs;
2. closes the complete state/disk inventory and surfaces non-public fallbacks or unexpected products;
3. loads every observed feature for a deterministic, feature-specific participant sample;
4. verifies exact sampled row counts, schemas, dtypes, helpers and native/curated agreement;
5. optionally recomputes every state-recorded file SHA-256 independently of the sample;
6. validates group ownership and read/search access for roots, participant folders, state databases and required
   support contracts.

### Permanent-root gate

Run the checked-in wrapper from the repository root:

```bash
set -euo pipefail

export LAB_GROUP=segalg
export APPLE_NATIVE=/path/to/cleaned_apple_healthkit
export APPLE_CURATED=/path/to/curated_apple_healthkit
export GOOGLE_NATIVE=/path/to/cleaned_google_fitness
export GOOGLE_CURATED=/path/to/curated_google_fitness
export PARTICIPANTS=50
export SEED=1
export HASH_WORKERS=16
export OUT=/path/outside/the/data/acceptance_0_4_0

bash scripts/release_0_4_0/RUN_FULL_COHORT_GATES.sh \
  2>&1 | tee "$OUT.run.log"
```

The wrapper invokes both sources with `--verify-all-hashes`, `--required-group` and `--fail-on-warnings`. Stable
release evidence requires, for both reports:

```json
{
  "verdict": "PASS",
  "release_gate_verdict": "PASS",
  "findings": []
}
```

`--participants 50` means up to 50 participants independently for each observed feature and phase. Rare features are
therefore loaded fully, while `--verify-all-hashes` still covers every committed file. Use `--participants 0` only
when every stored row must also be parsed.

The final 5 November 2026 permanent-root run passed both sources with zero failures and zero warnings and verified:

```text
Apple native + curated: 46,670 files, 211.8 GB
Google native + curated: 65,414 files, 3.53 GB
Combined:                112,084 files, 215.3 GB, 0 hash failures
```

### Python and pandas compatibility gate

Package metadata declares Python 3.10+ and pandas 2.0+. Before publishing a stable wheel, execute the full suite and
installed-wheel smoke test in the supported matrix:

| Python | pandas                     | Purpose                           |
|--------|----------------------------|-----------------------------------|
| 3.10   | 2.0.x                      | Oldest declared boundary          |
| 3.12   | 2.1.x                      | Historical laboratory environment |
| 3.12   | 2.2.x                      | Current pandas 2.x behavior       |
| 3.12   | 3.0.x                      | pandas 3 timestamp/dtype behavior |
| 3.13   | 2.2.x or current supported | Newest Python/build environment   |

Use the checked-in isolated runner for one matrix row:

```bash
PYTHON=/path/to/python3.12 \
PANDAS_SPEC='pandas==2.2.*' \
WEARABLE_NATIVE_SAMPLE=/path/to/cleaned_apple_healthkit \
WEARABLE_CURATED_SAMPLE=/path/to/curated_apple_healthkit \
WEARABLE_GOOGLE_NATIVE_SAMPLE=/path/to/cleaned_google_fitness \
WEARABLE_GOOGLE_CURATED_SAMPLE=/path/to/curated_google_fitness \
scripts/release_0_4_0/RUN_COMPATIBILITY_ENV.sh
```

### Build and release-integrity gate

After every tracked source change and before building final artifacts:

```bash
python -m wearable_project.utils.regenerate_manifest --write --release-label 0.4.0
python -m wearable_project.utils.regenerate_manifest --check --release-label 0.4.0
python -m pip install --upgrade build
python -m build
```

Install both wheel and source distribution outside the checkout, run representative Apple and Google loads, and
record SHA-256 digests for source archive, wheel, source distribution, test evidence and cohort reports.

## Repository layout and developer documentation

```text
wearable_project/
├── wearable_project/
│   ├── DataLoaders/                 # 96 top-level feature modules and the shared source-aware engine
│   │   ├── apple_healthkit/         # Internal Apple contracts, state, schemas, units, derived behavior and info
│   │   ├── google_fitness/          # Internal Google contracts, schema-v2 virtual fields, units and info
│   │   ├── _base.py                 # Exact reads, filtering, verification, projections and size protection
│   │   ├── _contracts.py            # Source-neutral adapter and feature/root contracts
│   │   ├── _sources.py              # Source catalogs, default roots and managed-root identity
│   │   └── info.py                  # Source-aware static discovery and feature reports
│   ├── processing/
│   │   ├── apple_healthkit/
│   │   └── google_fitness/
│   ├── curation/
│   │   ├── apple_healthkit/
│   │   └── google_fitness/
│   ├── utils/                       # Apple analytical utilities plus source-aware cohort acceptance
│   ├── cli.py
│   └── exceptions.py
├── tests/
├── docs/
├── README.md
├── LICENSE
└── pyproject.toml
```

## Running the tests

```bash
python -m pip install -e ".[dev]"

export WEARABLE_NATIVE_SAMPLE=/path/to/cleaned_apple_healthkit
export WEARABLE_CURATED_SAMPLE=/path/to/curated_apple_healthkit
export WEARABLE_GOOGLE_NATIVE_SAMPLE=/path/to/cleaned_google_fitness
export WEARABLE_GOOGLE_CURATED_SAMPLE=/path/to/curated_google_fitness

python -m pytest -rs
```

Tests that require sample roots are skipped when the corresponding environment variable is absent; `-rs` reports
why. Parquet tests additionally require `pyarrow`.

## License

MIT. See [`LICENSE`](LICENSE).
