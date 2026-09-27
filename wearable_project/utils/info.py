"""
Wearable Data Processing and Modeling project
Introspective help for the statistics and figure tools: ``data_statistics``, ``data_summaries``, ``domain_metrics``
and ``data_plots``.
As with ``DataLoaders.info``, whatever the code can state for itself is derived from it: signatures, docstrings,
the options each figure accepts, the fields of every result, the output schema versions and the command line's own
help. What the code cannot state is written here: the question each tool answers, how the tools chain, what each
parameter and field means, which participants a figure shows, and an example. ``tests/test_utils_info.py`` holds
those claims against the code. Every public function and class has a card and every parameter and field is
explained. Every example runs as written, and every figure's announced level and panels match what it draws.
Typical notebook use:

    from wearable_project import utils

    utils.info()                             # the map: modules, pipeline, tools by question
    utils.info("compute_daily_statistics")   # one tool
    utils.info("plot_agp")                   # one figure
    utils.info("min_participants")           # one parameter, and the tools that take it
    utils.info("sleep")                      # the tools for one question
    utils.info("jetlag")                     # anything else: a ranked search
    utils.info("plot_agp").as_dict()         # the same, structured

``info`` describes the tools and never touches data. From a shell: ``python -m wearable_project.utils.info plot_agp``.
"""


from __future__ import annotations
import argparse
import contextlib
import dataclasses
import difflib
import importlib
import inspect
import io
import json
import re
import sys
import textwrap
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from wearable_project import __version__
from wearable_project.DataLoaders.info import InfoReport
from wearable_project.exceptions import DataLoaderConfigurationError


WIDTH = 110
# The four modules users call, with the alias every example uses and what each is for.
MODULES = {
    "data_statistics": ("ds", "Per participant-day statistics from the loaders: coverage of the roots, the daily run "
                              "every other tool draws from, written runs, and the guard that keeps outputs out of "
                              "data roots."),
    "data_summaries": ("sm", "Summaries of a run: valid days, adherence and retention, participant metrics, "
                             "temporal patterns, hour of day, time zones, data quality and clinical thresholds."),
    "domain_metrics": ("dm", "Sleep and glucose: nights with their stages and naps, regularity, CGM days and "
                             "periods, the Ambulatory Glucose Profile, and one participant's segments or readings."),
    "data_plots": ("dp", "Figures of all of the above, each carrying the table it draws, and a report that gathers "
                         "them into one PDF."),
}
# The questions the tools answer, in the order the overview lists them: key, title, what the group is for.
GROUPS = (
    ("runs", "Runs and inputs", "Compute, write and read the statistics every other tool draws from."),
    ("coverage", "Coverage and availability", "Who has which feature, how much of it, and when."),
    ("quality", "Data quality and trust", "How far the data can be trusted: valid-day rules, cadence, provenance, "
                                          "curation, redundancy and plausibility."),
    ("activity", "Activity over time", "How data accrue and change over calendar or study time."),
    ("patterns", "Daily and weekly patterns", "When things happen: hour of day, day of week, month of year, time "
                                              "zones."),
    ("values", "Values across participants", "What the values are: participant metrics, their distributions and "
                                             "spread."),
    ("adherence", "Adherence and retention", "How consistently participants contribute valid days, and for how "
                                             "long."),
    ("sleep", "Sleep", "Nights, stages, naps, regularity and timing."),
    ("glucose", "Glucose", "CGM days and periods, glucose profiles and consensus targets."),
    ("clinical", "Clinical references", "Values against clinical categories, guidelines and thresholds."),
    ("multimodal", "Multimodal", "How features relate: within one participant over time, and between "
                                 "participants."),
    ("identity", "Feature order and colors", "The order and color every figure gives a feature."),
    ("output", "Reports, files and safety", "Reports, the command line, and where outputs may and may not go."),
)
TIERS = ("entry point", "building block")
# Options many figures share; a figure's card lists those its signature accepts.
SHARED_OPTIONS = ("groups", "min_participants", "ci", "seed", "unit", "time_axis", "show_ids", "path", "dpi")

# The names every example assumes. Each card shows the blocks its example needs; ``{phase}``, ``{root}`` and
# ``{out}`` are filled from ``info(..., phase=, root=, out=)``, so each example runs as printed.
SETUP = {
    "base": ('import numpy as np\n'
             'from wearable_project import DataLoaders\n'
             'from wearable_project.utils import data_statistics as ds, data_summaries as sm, domain_metrics as dm, '
             'data_plots as dp\n\n'
             'coverage = ds.compute_coverage("{phase}"{root})\n'
             'daily = ds.compute_daily_statistics("{phase}"{root})\n'
             'summaries, patterns, quality = sm.summarize(daily), sm.temporal_patterns(daily), sm.quality_report(daily)\n'
             'metrics = dm.compute_domain_metrics("{phase}"{root})\n'
             'code = coverage.participant_feature.query("feature == \'HeartRate\'")["RegistrationCode"].iloc[0]\n'
             'sleeper = metrics.sleep_nights["RegistrationCode"].iloc[0]\n'
             'wearer = metrics.cgm_periods.loc[metrics.cgm_periods["cgm"].astype(bool), "RegistrationCode"].iloc[0]'),
    "one": ('segments = dm.load_sleep_segments(sleeper, "{phase}"{root})\n'
            'readings = dm.load_cgm_readings(wearer, "{phase}"{root})'),
    "loaded": ('heart = DataLoaders.HeartRateLoader(phase="{phase}"{root}).get_data(registration_codes=code)\n'
               'sleep = DataLoaders.SleepLoader(phase="{phase}"{root}).get_data(registration_codes=sleeper, '
               'projection="full")\n'
               'glucose = DataLoaders.BloodGlucoseLoader(phase="{phase}"{root}).get_data(registration_codes=wearer)'),
    "written": ('run_dir, domain_dir = "{out}/wearable_run", "{out}/domain_run"\n'
                'ds.compute_daily_statistics("{phase}"{root}, features=["HeartRate", "Weight"], out=run_dir)\n'
                'dm.compute_domain_metrics("{phase}"{root}, out=domain_dir)'),
}
SETUP_NOTES = {
    "base": "the conventional names: coverage, daily, summaries, patterns, quality, metrics, and participants code "
            "(with heart rate), sleeper (with sleep) and wearer (with a CGM). At cohort scale, compute the run once "
            "with out= and pass its directory wherever daily appears, since every tool that takes a run accepts one",
    "one": "one participant's sleep segments and CGM readings",
    "loaded": "loaded results, as a DataLoader returns them, for the building blocks that summarize one result",
    "written": "a written statistics run and a written domain run, in a directory outside every data root",
}

# What each parameter means, wherever it appears; a tool's own notes override these (``_Tool.notes``).
PARAMETERS = {
    "adherence": "The adherence table, summarize(...).adherence: per participant and feature, valid days, "
                 "follow-up, the longest run of valid days and the gaps.",
    "argv": "Command-line arguments as a list of strings; None reads them from the command line.",
    "bin_minutes": "Width of the time-of-day bins in minutes; it must divide the day's 1,440 minutes, such as 5, 15 "
                   "or 60.",
    "by": "What data volume is counted by: 'participant', a histogram of each participant's bytes on disk, or "
          "'feature', the total per feature.",
    "cadence": "The participant-feature's row of participant_feature (typical gap and regular share), or None when "
               "unknown.",
    "chunksize": "Rows read at a time from a written run, which bounds memory at cohort scale.",
    "ci": "Level of a seeded bootstrap interval of the cohort median, such as 0.95, drawn distinctly from the "
          "interquartile band (which shows spread, not uncertainty); None draws none.",
    "clip_quantile": "Stop the value axis at this quantile, stating how many values lie beyond it; None, the default, "
                     "draws every value.",
    "columns": "The columns of the feature's daily table.",
    "coverage": "What compute_coverage(...) returned.",
    "daily": "One participant's daily rows of one feature: a slice of a run's daily table.",
    "data": "A loaded result, as a DataLoader's get_data(...) returns it, typically for one participant.",
    "default_inclusion_only": "Curated phase: keep only the records the curation includes by default. The native "
                              "phase has no such verdicts.",
    "diastolic": "Diastolic pressures in mmHg, aligned with systolic.",
    "domain": "Domain metrics (compute_domain_metrics), so that sleep is drawn per night and glucose per CGM day; "
              "the 'sleep' series needs them.",
    "dpi": "Resolution of a saved raster image such as PNG; vector formats (PDF, SVG) ignore it.",
    "expected": "Readings a full day holds for this participant (expected_records_per_day), or None where "
                "completeness does not apply.",
    "feature": "One feature's name, such as 'HeartRate'; DataLoaders.available_features() lists them.",
    "features": "The feature names to include; None includes every feature the input holds.",
    "gaps": "'kept' draws nights on a continuous axis with the gaps between them; 'removed' draws only nights with "
            "data, one after another.",
    "groups": "A mapping from participant to label (a dict, a Series, or a table with RegistrationCode and group), "
              "drawing one line, box or bar per group with its size stated. Participants without a label are left "
              "out, and the figure says how many.",
    "guideline": "The blood-pressure guideline: 'esc_esh_home' (the default: hypertension from 135/85 mmHg, for "
                 "readings taken at home), 'esc_esh_office' (optimal to grade 3) or 'acc_aha' (normal to stage 2).",
    "harmonize": "Convert every value to its feature's one declared unit before summarizing (the default); False "
                 "keeps the stored units.",
    "include_individual": "Include figures of individual participants. By default they are included unless "
                          "min_participants is above 1, since they cannot respect a small-cell threshold.",
    "lag_days": "Days from x to y: x on day d is paired with y on day d + lag_days (a whole number, at most 30).",
    "max_combinations": "How many feature combinations to draw, the most common first.",
    "measure": "What the figure draws; each figure lists its choices.",
    "method": "The correlation: 'spearman' (the default, rank-based) or 'pearson'.",
    "metric": "The daily column to draw; None chooses the feature's headline metric, by its measurement kind.",
    "metrics": "Domain metrics, as compute_domain_metrics(...) returns them.",
    "mgdl": "Glucose readings in mg/dL.",
    "min_days": "Paired days a participant needs to be included.",
    "min_participants": "The small-cell threshold. A point, bar, cell, bin or combination resting on fewer "
                        "participants is not drawn, its values are removed from figure.data too, and the figure "
                        "says how many were hidden. Use it for figures leaving the lab; 1, the default, hides nothing.",
    "min_valid_days": "The k of the curves 'participants with at least k valid days'.",
    "mmol": "Glucose readings in mmol/L.",
    "name": "The table's name, such as 'participant_feature'.",
    "night_rule": "Which nights count: NightRule(min_asleep_minutes=180) by default.",
    "nights": "The sleep_nights table of domain metrics.",
    "order_by": "Order participants by the hour of their lowest value ('trough', the default) or highest ('peak').",
    "out": "A directory for a written run, which must be new or empty (unless resuming) and outside every data root; "
           "None keeps everything in memory.",
    "paired": "Draw a line per participant from their weekday to their weekend median (individual-level, the "
              "default); False draws only the distribution of the differences, an aggregate.",
    "participant": "One participant's registration code, with or without the 10K_ prefix.",
    "participant_metrics": "summarize(...).participant_metrics: each participant's median, quartiles and days per "
                           "feature and metric.",
    "participants": "Registration codes to include; None includes everyone with data.",
    "path": "Save the figure there, in the format of its suffix (.png, .pdf, .svg); None saves nothing. A data root is "
            "refused as destination.",
    "patterns": "What temporal_patterns(...) returned.",
    "period": "'month', 'quarter', or 'auto' (the default: months up to 36 of them, quarters beyond).",
    "phase": "'curated' (the default) or 'native'.",
    "points": "Draw each participant as a point (individual-level, the default); False keeps the figure aggregate.",
    "readings": "CGM readings, as load_cgm_readings or cgm_readings return them.",
    "regular_share": "The share of gaps close to the typical one, from participant_feature.",
    "relative": "Divide each participant's profile by their own mean (the default), so profiles in different ranges "
                "compare; False draws the values themselves.",
    "report": "What quality_report(...) returned.",
    "resume": "Continue an interrupted written run in out; a run written with another output schema is refused.",
    "rolling": "Add a centred rolling mean over this many days.",
    "root": "The data root to read; None reads the phase's permanent HPP root.",
    "roots": "More directories to protect, beyond the permanent roots and those read in this session.",
    "rule": "The rule deciding which days (a ValidDayRule; None uses the feature's default) or nights (a NightRule) "
            "count.",
    "rules": "Valid-day rules by feature, replacing the defaults: {\"HeartRate\": ValidDayRule(...)}.",
    "sections": "The report's parts to keep: 'cohort', 'sleep', 'glucose', 'values', 'clinical' and 'feature' (one "
                "dossier per feature); None keeps them all.",
    "seed": "Seed of the bootstrap, so that intervals are reproducible.",
    "segments": "One participant's sleep segments, as load_sleep_segments or sleep_segments return them.",
    "show_ids": "Label participants with their registration codes; by default they are Participant 1, 2 and so on.",
    "sort": "Order the lines by first day ('start', the default) or by length ('length').",
    "source": "A daily-statistics run: the object compute_daily_statistics returned, or the directory it wrote.",
    "state_validation": "How the loader checks the root's state database before reading: 'auto' (the default), "
                        "'required' or 'off'.",
    "sufficient_only": "Only CGM participants with the 14 valid days the consensus requires (the default).",
    "summaries": "What summarize(...) returned.",
    "systolic": "Systolic pressures in mmHg, aligned with diastolic.",
    "thresholds": "The threshold values to try; None chooses a range suited to the rule's criterion.",
    "time_axis": "'calendar' (dates) or 'study' (days since each participant's first). A figure of one participant "
                 "defaults to study days, so that real dates appear only when asked for.",
    "typical_gap_minutes": "The typical gap between records, from participant_feature.",
    "unit": "What counts once in a cohort distribution: 'participant' (the default: each participant's median day "
            "or night), or each day or night pooled ('participant_day', 'night').",
    "valid_only": "Draw valid days only (the default for cohort figures).",
    "weekend_days": "The weekend, Monday being 0: (4, 5), Friday and Saturday, by default. A night is free when it "
                    "ends on a weekend day, and weeks start the day after the weekend.",
    "workers": "Worker processes; 1, the default, computes in this process.",
    "x": "The first daily series: a feature of the run, or 'sleep' (total sleep of the night starting on day d, which "
         "needs domain).",
    "y": "The second daily series, taken lag_days after x: a feature of the run, or 'sleep'.",
}

# What each field of the result classes holds.
FIELDS = {
    ("Coverage", "phase"): "The phase covered.",
    ("Coverage", "participant_feature"): "One row per participant and feature: rows, bytes on disk and, curated, the "
                                         "curation verdicts.",
    ("Coverage", "features"): "One row per feature: participants, rows and bytes.",
    ("Coverage", "participants"): "One row per participant: features, rows and bytes.",
    ("Coverage", "notes"): "What the coverage could not establish, and why.",
    ("DailyStatistics", "phase"): "The phase computed.",
    ("DailyStatistics", "daily"): "Per feature, one row per participant and local day: records, values combined as "
                                  "the feature's measurement kind requires, coverage and hours.",
    ("DailyStatistics", "hourly"): "One row per participant, feature and local hour of day, all 24 hours present.",
    ("DailyStatistics", "participant_feature"): "One row per participant and feature: days, first and last day, "
                                                "cadence and totals.",
    ("DailyStatistics", "participant_days"): "One row per participant and local day: the features with data and the "
                                             "UTC offsets.",
    ("DailyStatistics", "cohort_daily"): "Per feature and local day, across participants.",
    ("DailyStatistics", "active_participants"): "Participants with any data per local day.",
    ("DailyStatistics", "errors"): "Participant-features that failed, with the reason; the run continues past them.",
    ("DailyStatistics", "run"): "The run's record (run.json): parameters, output schema, versions and timing.",
    ("DailyStatistics", "provenance"): "Records per participant, feature, day and acquisition method.",
    ("DailyStatistics", "curation"): "Curated phase: records per participant, feature, day and curation status or "
                                     "flag.",
    ("ValidDayRule", "min_records"): "Records a day needs.",
    ("ValidDayRule", "min_hours_with_data"): "Distinct local hours with a record that a day needs; None does not "
                                             "require any.",
    ("ValidDayRule", "min_completeness"): "Share of the expected readings a day needs, for participants with a fixed "
                                          "cadence; None does not require any.",
    ("FixedCadence", "max_typical_gap_minutes"): "Largest typical gap for which a cadence counts as fixed.",
    ("FixedCadence", "min_regular_share"): "Smallest share of regular gaps for which a cadence counts as fixed.",
    ("Summaries", "participant_metrics"): "Each participant's median, quartiles and valid days per feature and "
                                          "metric.",
    ("Summaries", "adherence"): "Per participant and feature: valid days, follow-up, longest run and gaps.",
    ("Summaries", "cohort"): "Table 1: per feature and metric, the distribution of participants' medians.",
    ("Summaries", "retention"): "Per feature, the share of participants still contributing k days after their first "
                                "valid day.",
    ("Summaries", "rules"): "The valid-day rule applied to each feature.",
    ("TemporalPatterns", "day_of_week"): "Each participant's median per day of week.",
    ("TemporalPatterns", "weekday_weekend"): "Each participant's weekday and weekend medians, and their difference.",
    ("TemporalPatterns", "month_of_year"): "Each participant's median per month of year.",
    ("TemporalPatterns", "cohort_day_of_week"): "Across participants, per day of week.",
    ("TemporalPatterns", "cohort_weekday_weekend"): "Across participants, weekdays against the weekend.",
    ("TemporalPatterns", "cohort_month_of_year"): "Across participants, per month of year.",
    ("TemporalPatterns", "weekend_days"): "The weekend days used, Monday being 0.",
    ("QualityReport", "features"): "Per feature: records, participant-days, user-entered records, redundancy, values "
                                   "out of range and records without a UTC offset.",
    ("QualityReport", "provenance"): "Per feature and acquisition method, the records and their share.",
    ("QualityReport", "curation"): "Per feature, curation status or flag, the records and their share.",
    ("NightRule", "min_asleep_minutes"): "Sleep a night needs to count as valid, in minutes.",
    ("DomainMetrics", "sleep_nights"): "One row per participant and night (noon to noon): time in bed and asleep, "
                                       "stages, timing, wake, naps and sources.",
    ("DomainMetrics", "cgm_days"): "One row per CGM participant and day: readings, completeness, validity and "
                                   "glucose metrics.",
    ("DomainMetrics", "cgm_periods"): "One row per participant with BloodGlucose: whether a CGM, valid days, "
                                      "sufficiency and the consensus metrics over valid days.",
    ("DomainMetrics", "errors"): "Participants that failed, with the reason.",
    ("DomainMetrics", "run"): "The run's record: parameters, output schema, definitions and timing.",
    ("DomainMetrics", "cgm_profile"): "Each CGM participant's glucose percentiles per 15-minute bin of the day, over "
                                      "valid days (the AGP).",
}


@dataclass(frozen=True)
class _Tool:
    """What the code cannot say about a tool; everything else is derived."""

    module: str
    group: str
    tier: str
    does: str
    returns: str
    example: str = ""
    needs: tuple[str, ...] = ()
    related: tuple[str, ...] = ()
    notes: tuple[tuple[str, str], ...] = ()
    # figures only: the level of the figure the example draws, when it depends on arguments, and its panels
    individual: bool | None = None
    level_note: str = ""
    panels: tuple[str, ...] = ()
    data: str = ""


# ------------------------------------------------------------------------------------------------ catalogue
S, M, D, P = "data_statistics", "data_summaries", "domain_metrics", "data_plots"
E, B = "entry point", "building block"
_WRITE_NOTE = "A new directory for the written run, outside every data root; None keeps the run in memory."
_READ_NOTE = "The directory a run was written to."
_NIGHT_RULE = "The night rule: NightRule(min_asleep_minutes=180) by default."
_HOUR_MEASURE = ("'value' (the default: the mean value, or the amount per day for totals and event amounts), "
                 "'records' per day, or 'coverage', the share of the participant's days with data in the hour.")


def _t(module: str, group: str, tier: str, does: str, returns: str = "", example: str = "", **extra) -> _Tool:
    notes = tuple(extra.pop("notes", {}).items())
    return _Tool(module, group, tier, does, returns, example, notes=notes, **extra)


TOOLS: dict[str, _Tool] = {
    # ---------------------------------------------------------------------------------- data_statistics
    "compute_coverage": _t(
        S, "coverage", E, "What each phase's root holds for every feature, read from the state databases alone, "
        "without parsing any CSV: which participant has which feature, with rows, bytes and curation verdicts.",
        "Coverage: .participant_feature, .features, .participants, .notes, and .presence(), participants by "
        "features.", 'ds.compute_coverage("{phase}"{root})',
        related=("compute_daily_statistics", "plot_feature_presence", "plot_participants_per_feature",
                 "plot_feature_combinations"),
        notes={"features": "The features to cover; None covers every feature."}),
    "compute_daily_statistics": _t(
        S, "runs", E, "Per participant and local day, the statistics of every feature: records, values combined as "
        "the feature's measurement kind requires (sums for totals, means for levels), coverage and hours with data, "
        "provenance and curation. Streamed one participant at a time, in memory or written as a resumable run; "
        "every summary and figure draws from it.",
        "DailyStatistics: .daily (per feature), .hourly, .participant_feature, .participant_days, .cohort_daily, "
        ".active_participants, .provenance, .curation, .errors and .run. With out=, the same tables are also "
        "written, and the directory can stand in for the object anywhere.",
        'ds.compute_daily_statistics("{phase}"{root}, features=["HeartRate", "StepCount"])',
        related=("summarize", "temporal_patterns", "quality_report", "compute_domain_metrics", "report"),
        notes={"out": _WRITE_NOTE, "features": "The features to compute; None computes every feature."}),
    "read_daily": _t(
        S, "runs", B, "Read one feature's daily table from a written run, optionally for some participants.",
        "A DataFrame of daily rows, typed as in memory.", 'ds.read_daily(run_dir, "HeartRate")', needs=("written",),
        related=("iter_daily", "read_table", "compute_daily_statistics"),
        notes={"out": _READ_NOTE, "participants": "The participants to read; None reads everyone."}),
    "iter_daily": _t(
        S, "runs", B, "Read one feature's daily table from a written run one participant at a time, so memory stays "
                      "bounded at cohort scale.",
        "An iterator of DataFrames, one participant each.", 'next(ds.iter_daily(run_dir, "HeartRate"))',
        needs=("written",), related=("read_daily",), notes={"out": _READ_NOTE}),
    "read_table": _t(
        S, "runs", B, "Read one of a written run's tables other than the daily ones, by name.",
        "A DataFrame, typed as in memory.", 'ds.read_table(run_dir, "participant_feature")', needs=("written",),
        related=("read_daily", "compute_daily_statistics"),
        notes={"out": _READ_NOTE, "name": "The table, such as 'participant_feature', 'participant_days', 'hourly' "
                                          "or 'cohort_daily'."}),
    "export_parquet": _t(
        S, "runs", B, "Write a Parquet copy beside every table of a written run, for faster reading. Needs pyarrow "
                      "(or fastparquet), which the package does not require.",
        "The Parquet files written.", 'ds.export_parquet(run_dir)', needs=("written", "pyarrow"),
        related=("read_table",), notes={"out": _READ_NOTE}),
    "guard_output": _t(
        S, "output", B, "Check a destination before writing: refuse any path inside a data root, whether a permanent "
                        "HPP root, a root read in this session, or a directory that holds participant data.",
        "The path, resolved, or a DataLoaderConfigurationError.", 'ds.guard_output("{out}/figure.png")',
        related=("protected_roots",), notes={"path": "The destination to check."}),
    "protected_roots": _t(
        S, "output", B, "List the directories no output is ever written into.", "A list of paths.",
        "ds.protected_roots()", related=("guard_output",)),
    "measurement_kind": _t(
        S, "runs", B, "The measurement kind the curation registry declares for a feature, which decides how its "
                      "values combine: summed (totals), averaged (levels), and so on.",
        "A string such as 'extensive_total' or 'intensive_value'.", 'ds.measurement_kind("StepCount")',
        related=("headline_metrics", "column_units")),
    "column_units": _t(
        S, "quality", B, "For each numeric measurement of a feature, the unit the loaders establish for its stored "
                         "values and the unit the curation registry declares.",
        "A dict per column: the established and declared units.", 'ds.column_units("BloodGlucose")',
        related=("measurement_kind",)),
    "sampling_cadence": _t(
        S, "quality", B, "How regularly a result's records arrive: the typical gap between records and the share of "
                         "gaps close to it.",
        "A dict with typical_gap_minutes and regular_share.", "ds.sampling_cadence(heart)", needs=("loaded",),
        related=("plot_sampling_cadence", "expected_records_per_day")),
    "summarize_result": _t(
        S, "runs", B, "The daily summary and hour-of-day profile of one loaded result, as the run computes them per "
                      "participant.",
        "(daily, hourly): two DataFrames.", "ds.summarize_result(heart)", needs=("loaded",),
        related=("compute_daily_statistics",),
        notes={"feature": "The feature, when the result does not record its own."}),
    "provenance_tables": _t(
        S, "quality", B, "Records of one loaded result by participant, local day and acquisition method, and by "
                         "curation status and flag.",
        "(provenance, curation): two DataFrames in long format.", "ds.provenance_tables(heart)", needs=("loaded",),
        related=("quality_report",), notes={"feature": "The feature, when the result does not record its own."}),
    "main": _t(
        S, "output", E, "The command line: compute coverage or a written run from a shell, with the same options.",
        "An exit code.", "$ python -m wearable_project.utils.data_statistics --help",
        related=("compute_daily_statistics", "compute_coverage")),
    "Coverage": _t(S, "coverage", E, "What compute_coverage returns.", "", "coverage.presence()",
                   related=("compute_coverage",)),
    "DailyStatistics": _t(S, "runs", E, "What compute_daily_statistics returns: the run.", "",
                          "daily.participant_feature.head()", related=("compute_daily_statistics",)),
    # ----------------------------------------------------------------------------------- data_summaries
    "summarize": _t(
        M, "values", E, "Summaries of a run over valid days: each participant's median, quartiles and days per "
        "feature and metric; adherence (valid days, follow-up, runs and gaps); retention; and the cohort's Table 1.",
        "Summaries: .participant_metrics, .adherence, .cohort, .retention, .rules.", "sm.summarize(daily)",
        related=("temporal_patterns", "plot_adherence", "plot_retention", "plot_metric_distributions"),
        notes={"out": "A directory to write the summary tables to, outside every data root; None keeps them in "
                      "memory.", "features": "The features to summarize; None summarizes every feature of the run."}),
    "temporal_patterns": _t(
        M, "patterns", E, "Each participant's median per day of week, weekdays against the weekend, and per month "
                          "of year, over valid days; then the same across participants.",
        "TemporalPatterns: participant and cohort tables for each, and the weekend used.",
        "sm.temporal_patterns(daily, weekend_days=(5, 6))",
        related=("plot_weekly_pattern", "plot_monthly_pattern", "plot_weekday_weekend"),
        notes={"features": "The features to summarize; None summarizes every feature of the run."}),
    "quality_report": _t(
        M, "quality", E, "Totals per feature: records, participant-days, user-entered records, time recorded by "
                         "more than one device, values outside plausible ranges and records without a UTC offset; "
                         "and records by acquisition method and by curation status and flag.",
        "QualityReport: .features, .provenance, .curation.", "sm.quality_report(daily)",
        related=("plot_quality", "plot_acquisition", "plot_curation", "plot_curation_flags")),
    "hour_of_day": _t(
        M, "patterns", E, "Each participant's profile by local hour of day (records per day, amount per day, mean "
                          "value, and the share of days with data in the hour), and the cohort's quartiles.",
        "(participants, cohort): two DataFrames.", "sm.hour_of_day(daily)",
        related=("plot_hour_of_day", "plot_hour_profiles", "plot_hourly_coverage", "plot_daily_rhythms")),
    "clinical_thresholds": _t(
        M, "clinical", E, "Readings beyond clinical thresholds, such as SpO2 below 90% or a temperature of at least "
                          "38.0 C, per participant and for the cohort; counted only where the values are in the "
                          "threshold's unit.",
        "(participants, cohort): two DataFrames.", "sm.clinical_thresholds(daily)",
        related=("plot_clinical_thresholds",)),
    "time_zones": _t(
        M, "patterns", E, "Per participant, the home zone (a zone with its daylight-saving partner), the days spent "
                          "away from it, and the trips: runs of consecutive days away.",
        "A DataFrame, one row per participant.", "sm.time_zones(daily)", related=("plot_time_zones",)),
    "day_overlap": _t(
        M, "multimodal", E, "For each pair of features, the participant-days holding both, those holding either, "
                            "and their ratio (the Jaccard index): how often two modalities can be analysed together.",
        "A DataFrame, one row per pair.", "sm.day_overlap(daily)", related=("plot_co_availability", "days_with")),
    "days_with": _t(
        M, "multimodal", E, "Per participant, the days on which every one of several features has data, with the "
                            "first and last such day.",
        "A DataFrame, one row per participant with at least one such day.",
        'sm.days_with(daily, ["HeartRate", "StepCount"])', related=("plot_multimodal_days", "day_overlap"),
        notes={"features": "The features that must all hold data on a day."}),
    "retention": _t(
        M, "adherence", B, "Per feature, the share of participants still contributing valid days k days after their "
                           "first valid day, for every k.",
        "A DataFrame: feature, days since the first valid day, participants and fraction.",
        "sm.retention(summaries.adherence)", related=("plot_retention", "summarize")),
    "cohort_table": _t(
        M, "values", B, "Table 1: per feature and metric, the distribution across participants of their medians.",
        "A DataFrame.", "sm.cohort_table(summaries.participant_metrics, summaries.adherence)",
        related=("summarize", "plot_metric_distributions")),
    "summarize_participant": _t(
        M, "values", B, "Metric summaries and adherence of one participant's daily rows of one feature.",
        "(metric rows, adherence row).",
        'rows = daily.daily["HeartRate"]; sm.summarize_participant("HeartRate", rows[rows["RegistrationCode"] == '
        'code], None, sm.DEFAULT_RULES["HeartRate"])', related=("summarize", "mark_valid_days"),
        notes={"rule": "The valid-day rule to apply."}),
    "mark_valid_days": _t(
        M, "quality", B, "Decide which of one participant's days are valid under a rule, adding completeness where "
                         "a cadence is expected.",
        "The daily rows with completeness and valid added.",
        'rows = daily.daily["HeartRate"]; sm.mark_valid_days("HeartRate", rows[rows["RegistrationCode"] == code], '
        'sm.DEFAULT_RULES["HeartRate"], None)', related=("ValidDayRule", "plot_valid_day_rule"),
        notes={"rule": "The valid-day rule to apply."}),
    "expected_records_per_day": _t(
        M, "quality", B, "Readings a full day holds for a participant with a fixed cadence, such as a CGM every 5 "
                         "minutes; completeness is measured against it.",
        "A number, or None where completeness does not apply.",
        'sm.expected_records_per_day("BloodGlucose", 5.0, 0.95)', related=("FixedCadence", "mark_valid_days")),
    "headline_metrics": _t(
        M, "values", B, "The daily columns a feature is summarized by, chosen by its measurement kind, headline "
                        "first.",
        "A list of column names.", 'sm.headline_metrics("StepCount", daily.daily["StepCount"].columns)',
        related=("measurement_kind",)),
    "ValidDayRule": _t(M, "quality", E, "Which days count as valid: records, hours with data and completeness.",
                       "", "sm.ValidDayRule(min_records=1, min_hours_with_data=12)",
                       related=("mark_valid_days", "plot_valid_day_rule", "summarize")),
    "FixedCadence": _t(M, "quality", B, "When a participant's cadence counts as fixed, so that completeness "
                                        "applies.", "", "sm.FIXED_CADENCE", related=("expected_records_per_day",)),
    "Summaries": _t(M, "values", E, "What summarize returns.", "", "summaries.adherence.head()",
                    related=("summarize",)),
    "TemporalPatterns": _t(M, "patterns", E, "What temporal_patterns returns.", "", "patterns.weekday_weekend.head()",
                           related=("temporal_patterns",)),
    "QualityReport": _t(M, "quality", E, "What quality_report returns.", "", "quality.features.head()",
                        related=("quality_report",)),
    # ----------------------------------------------------------------------------------- domain_metrics
    "compute_domain_metrics": _t(
        D, "runs", E, "Sleep nights and CGM metrics for every participant with Sleep or BloodGlucose data, streamed "
        "one participant at a time, in memory or written as a resumable run.",
        "DomainMetrics: .sleep_nights, .cgm_days, .cgm_periods, .cgm_profile, .errors and .run.",
        'dm.compute_domain_metrics("{phase}"{root})',
        related=("summarize_domain", "plot_sleep", "plot_agp", "plot_glycemic_cohort"),
        notes={"out": _WRITE_NOTE}),
    "summarize_domain": _t(
        D, "values", E, "Participant summaries over valid nights and of CGM periods, with their cohort table, in "
                        "the same form as summarize.",
        "Summaries.", "dm.summarize_domain(metrics)", related=("summarize", "compute_domain_metrics")),
    "read_domain_table": _t(
        D, "runs", B, "Read one table of a written domain run, typed as in memory.", "A DataFrame.",
        'dm.read_domain_table(domain_dir, "sleep_nights")', needs=("written",), related=("compute_domain_metrics",),
        notes={"out": "The directory a domain run was written to.",
               "name": "'sleep_nights', 'cgm_days', 'cgm_periods' or 'cgm_profile'."}),
    "definitions": _t(
        D, "runs", B, "The constants the domain metrics rest on (night start, episode gap, the sleep states, the "
                      "CGM thresholds and the glucose ranges), as recorded in every run.",
        "A dict.", "dm.definitions()", related=("compute_domain_metrics",)),
    "load_sleep_segments": _t(
        D, "sleep", E, "One participant's sleep records split into the nights they belong to, marking what the night "
                       "metrics count: the main sleep period, and stages from the device that staged most.",
        "A DataFrame of segments.", 'dm.load_sleep_segments(sleeper, "{phase}"{root})',
        related=("plot_sleep_raster", "sleep_segments")),
    "load_cgm_readings": _t(
        D, "glucose", E, "One participant's CGM readings with local time, minute of the day, mg/dL, and each day's "
                         "completeness and validity.",
        "A DataFrame of readings.", 'dm.load_cgm_readings(wearer, "{phase}"{root})',
        related=("plot_glucose_days", "cgm_profile", "cgm_readings")),
    "sleep_nights": _t(
        D, "sleep", B, "One row per participant and night, from a loaded Sleep result.", "A DataFrame.",
        "dm.sleep_nights(sleep)", needs=("loaded",), related=("compute_domain_metrics",)),
    "sleep_segments": _t(
        D, "sleep", B, "One participant's sleep segments, from a loaded Sleep result.", "A DataFrame.",
        "dm.sleep_segments(sleep)", needs=("loaded",), related=("load_sleep_segments",)),
    "sleep_regularity": _t(
        D, "sleep", E, "Per participant over valid nights: the variability of sleep timing and duration, and social "
                       "jetlag, the mean midpoint on free nights minus that on work nights.",
        "A DataFrame, one row per participant.", "dm.sleep_regularity(metrics.sleep_nights)",
        related=("plot_sleep_regularity",), notes={"rule": _NIGHT_RULE}),
    "cgm_metrics": _t(
        D, "glucose", B, "One participant's CGM days and period metrics from a loaded BloodGlucose result.",
        "(days, period): two DataFrames.", "dm.cgm_metrics(glucose)", needs=("loaded",),
        related=("compute_domain_metrics",)),
    "cgm_readings": _t(
        D, "glucose", B, "One participant's CGM readings from a loaded BloodGlucose result.", "A DataFrame.",
        "dm.cgm_readings(glucose)", needs=("loaded",), related=("load_cgm_readings",)),
    "cgm_profile": _t(
        D, "glucose", E, "The Ambulatory Glucose Profile: over valid days, the 5th to 95th percentiles of glucose "
                         "in each time-of-day bin, with the readings and days behind each.",
        "A DataFrame, one row per participant and bin.", "dm.cgm_profile(readings, bin_minutes=60)",
        needs=("one",), related=("plot_agp",)),
    "glucose_mgdl": _t(D, "glucose", B, "Convert glucose from mmol/L to mg/dL.", "An array of mg/dL.",
                       "dm.glucose_mgdl(np.array([3.9, 10.0]))", related=("glucose_range",)),
    "glucose_range": _t(D, "glucose", B, "Name the consensus range of each reading, from very low to very high.",
                        "An array of range names.",
                        "dm.glucose_range(np.array([50.0, 65.0, 120.0, 200.0, 300.0]))",
                        related=("glucose_mgdl", "plot_cgm_ranges")),
    "NightRule": _t(D, "sleep", E, "Which nights count as valid.", "", "dm.NightRule(min_asleep_minutes=240)",
                    related=("plot_sleep", "sleep_regularity")),
    "DomainMetrics": _t(D, "runs", E, "What compute_domain_metrics returns.", "", "metrics.cgm_periods.head()",
                        related=("compute_domain_metrics",)),
    # ----------------------------------------------------------------------------------------- data_plots
    "plot_feature_presence": _t(
        P, "coverage", E, "Which participant has which feature: participants (rows) by features (columns).",
        example="dp.plot_feature_presence(coverage)", individual=True, data="participant, feature, present",
        related=("plot_participants_per_feature", "plot_feature_combinations")),
    "plot_participants_per_feature": _t(
        P, "coverage", E, "How many participants have each feature, with the share of the cohort.",
        example="dp.plot_participants_per_feature(coverage, min_participants=5)", individual=False,
        data="feature, category, participants, share", related=("plot_features_per_participant",)),
    "plot_features_per_participant": _t(
        P, "coverage", E, "How many features participants have.", example="dp.plot_features_per_participant(coverage)",
        individual=False, data="features, participants", related=("plot_participants_per_feature",)),
    "plot_data_volume": _t(
        P, "coverage", E, "How much data is stored, per participant or per feature.",
        example='dp.plot_data_volume(coverage, by="feature")', individual=False,
        data="bytes per participant or per feature", related=("compute_coverage",)),
    "plot_feature_combinations": _t(
        P, "coverage", E, "Which exact combinations of features participants have (an UpSet chart).",
        example='dp.plot_feature_combinations(coverage, ["HeartRate", "StepCount", "Sleep"])', individual=False,
        panels=("sets", "all_combinations"), data="the drawn combinations and their participants",
        related=("plot_multimodal_days", "days_with"),
        notes={"features": "The features to combine; None takes the six most common."}),
    "plot_availability_raster": _t(
        P, "coverage", E, "Days with data per participant and month, in calendar or study time.",
        example='dp.plot_availability_raster(daily, time_axis="study")', individual=True,
        data="participant, month, days", related=("plot_follow_up",),
        notes={"features": "The features whose days count; None counts any feature."}),
    "plot_follow_up": _t(
        P, "coverage", E, "Each participant's span from first to last day with data, colored by the share of its "
                          "days holding data.",
        example='dp.plot_follow_up(daily, "Weight")', individual=True,
        data="participant, span, days with data, density", related=("plot_availability_raster",),
        notes={"feature": "One feature, or None for any feature."}),
    "plot_valid_day_rule": _t(
        P, "quality", E, "How a valid-day rule shapes the data before committing to it: its criterion's "
                         "distribution, and the days and participants kept as the threshold moves.",
        example='dp.plot_valid_day_rule(daily, "HeartRate")', individual=False, panels=("distribution", "sensitivity"),
        data="the sensitivity: per threshold, days kept and participants with at least k valid days",
        related=("ValidDayRule", "mark_valid_days"),
        notes={"rule": "The valid-day rule to examine; None examines the feature's default."}),
    "plot_sampling_cadence": _t(
        P, "quality", E, "Each participant's typical gap between records against its regularity: devices show as "
                         "clusters.",
        example='dp.plot_sampling_cadence(daily, ["HeartRate", "BloodGlucose"])', individual=True,
        data="participant, feature, typical gap, regular share", related=("sampling_cadence",),
        notes={"features": "The features to draw; None draws every feature."}),
    "plot_hourly_coverage": _t(
        P, "quality", E, "The share of each participant's days with data in each local hour: when devices are off.",
        example='dp.plot_hourly_coverage(daily, "HeartRate")', individual=True, panels=("cohort",),
        data="participant, hour, share of days", related=("hour_of_day", "plot_hour_of_day")),
    "plot_quality": _t(
        P, "quality", E, "Four quality indicators per feature: values out of range, time on more than one device, "
                         "records entered by hand, records without a UTC offset.",
        example="dp.plot_quality(quality)", individual=False, data="the indicators per feature",
        related=("quality_report",)),
    "plot_curation_flags": _t(
        P, "quality", E, "The share of each feature's records carrying each curation flag (curated phase).",
        example="dp.plot_curation_flags(quality)", individual=False, data="feature, flag, share of records",
        related=("quality_report", "plot_curation")),
    "plot_acquisition": _t(
        P, "quality", E, "The share of each feature's records by acquisition method.",
        example="dp.plot_acquisition(quality)", individual=False, data="feature, method, share",
        related=("plot_acquisition_over_time",)),
    "plot_acquisition_over_time": _t(
        P, "quality", E, "Records by acquisition method per calendar month: device and app transitions.",
        example='dp.plot_acquisition_over_time(daily, "HeartRate")', individual=False,
        data="month, method, records, share", related=("plot_acquisition",),
        notes={"feature": "One feature, or None for all.",
               "measure": "'share' of each month's records (the default) or 'records'."}),
    "plot_curation": _t(
        P, "quality", E, "The share of each feature's records by curation status (curated phase).",
        example="dp.plot_curation(quality)", individual=False, data="feature, status, share",
        related=("plot_curation_flags",)),
    "plot_active_participants": _t(
        P, "activity", E, "Participants with any data per day.", example="dp.plot_active_participants(daily, rolling=28)",
        individual=False, data="local date, participants", related=("plot_feature_activity",)),
    "plot_feature_activity": _t(
        P, "activity", E, "Participants (or participant-days) per feature and calendar month.",
        example="dp.plot_feature_activity(daily)", individual=False, data="feature, month, count",
        related=("plot_active_participants",),
        notes={"measure": "'participants' (the default) or 'participant_days', per feature and month."}),
    "plot_daily_values": _t(
        P, "activity", E, "A feature's daily values over time: one participant's days, or the cohort's median per "
                          "day.",
        example='dp.plot_daily_values(daily, "HeartRate", code)', individual=True,
        level_note="with participant=None, the cohort's median per day, an aggregate",
        data="the plotted days", related=("plot_monthly_distribution",),
        notes={"participant": "One participant, drawn on study days by default (individual-level); None draws the "
                              "cohort's median per day."}),
    "plot_monthly_distribution": _t(
        P, "activity", E, "The distribution of daily values per month or quarter, each participant counting once by "
                          "default.",
        example='dp.plot_monthly_distribution(daily, "HeartRate")', individual=False,
        data="per period: participants, median and quartiles", related=("plot_daily_values",),
        notes={"unit": "'participant' (the default: each participant's median day) or 'participant_day'."}),
    "plot_hour_of_day": _t(
        P, "patterns", E, "The cohort's median by local hour of day, with the participants behind each hour.",
        example='dp.plot_hour_of_day(daily, "HeartRate", ci=0.95)', individual=False,
        data="per hour: median, quartiles, participants", related=("hour_of_day", "plot_hour_profiles"),
        notes={"measure": _HOUR_MEASURE}),
    "plot_weekly_pattern": _t(
        P, "patterns", E, "The cohort's median of participants' medians by day of week, weekend shaded.",
        example='dp.plot_weekly_pattern(patterns, "StepCount")', individual=False,
        data="per weekday: median, quartiles, participants", related=("temporal_patterns", "plot_weekday_weekend")),
    "plot_monthly_pattern": _t(
        P, "patterns", E, "The cohort's median of participants' medians by month of year.",
        example='dp.plot_monthly_pattern(patterns, "HeartRate")', individual=False,
        data="per month: median, quartiles, participants", related=("temporal_patterns",)),
    "plot_weekday_weekend": _t(
        P, "patterns", E, "Weekdays against the weekend, per participant, and the distribution of the differences.",
        example='dp.plot_weekday_weekend(patterns, "StepCount")', individual=True,
        level_note="with paired=False, only the differences, an aggregate",
        data="participant: weekday and weekend medians and their difference", related=("temporal_patterns",)),
    "plot_hour_profiles": _t(
        P, "patterns", E, "Each participant's day by local hour, ordered by the hour of their lowest (or highest) "
                          "value: a chronotype-like view.",
        example='dp.plot_hour_profiles(daily, "HeartRate")', individual=True, panels=("order",),
        data="participant, hour, relative value", related=("hour_of_day", "plot_daily_rhythms"),
        notes={"measure": _HOUR_MEASURE}),
    "plot_daily_rhythms": _t(
        P, "patterns", E, "Small multiples of the cohort's daily rhythm per feature, relative to each participant's "
                          "mean, so features in different units compare.",
        example='dp.plot_daily_rhythms(daily, ["HeartRate", "StepCount"])', individual=False,
        data="feature, hour, median and quartiles of relative values", related=("plot_hour_profiles",),
        notes={"measure": _HOUR_MEASURE, "features": "The features to draw; None draws every feature with hours."}),
    "plot_time_zones": _t(
        P, "patterns", E, "Home zones, days away and trips across the cohort, or one participant's UTC offsets day by "
                          "day.",
        example="dp.plot_time_zones(daily)", individual=False, panels=("home_zones", "days_away", "trips"),
        level_note="with a participant, that participant's offsets, individual-level and without panels",
        data="participant, home zone, days away, trips", related=("time_zones",),
        notes={"participant": "One participant's offsets day by day; None draws the cohort."}),
    "plot_metric_distributions": _t(
        P, "values", E, "A visual Table 1: the distribution of participants' medians of each feature's headline "
                        "metric, with median and interquartile range stated.",
        example='dp.plot_metric_distributions(summaries, ["StepCount", "HeartRate", "Weight"])', individual=False,
        panels=("table1",), data="participant, feature, median", related=("cohort_table", "plot_caterpillar"),
        notes={"features": "The features to draw; None draws every feature with participant medians."}),
    "plot_caterpillar": _t(
        P, "values", E, "Each participant's median and interquartile range, sorted.",
        example='dp.plot_caterpillar(summaries, "HeartRate")', individual=True,
        data="participant: median and quartiles", related=("plot_metric_distributions",)),
    "plot_adherence": _t(
        P, "adherence", E, "Valid days per participant, and adherence over their follow-up.",
        example='dp.plot_adherence(summaries, "HeartRate")', individual=False,
        data="participant: valid days and adherence", related=("summarize", "plot_retention")),
    "plot_retention": _t(
        P, "adherence", E, "The share of participants still contributing, by days since their first valid day.",
        example="dp.plot_retention(summaries)", individual=False, data="feature, day, participants, share",
        related=("retention", "plot_adherence"),
        notes={"features": "The features to draw; None draws every feature."}),
    "plot_sleep": _t(
        P, "sleep", E, "Total sleep time, sleep midpoint and efficiency, each participant counting once by default.",
        example="dp.plot_sleep(metrics)", individual=False, data="participant or night: sleep, midpoint, efficiency",
        related=("plot_sleep_regularity", "plot_sleep_timing"),
        notes={"rule": _NIGHT_RULE, "unit": "'participant' (the default: each participant's median night) or "
                                            "'night'."}),
    "plot_sleep_raster": _t(
        P, "sleep", E, "One participant's nights from noon to noon: in bed, awake, asleep and stages, with what the "
                       "metrics leave out drawn faded.",
        example='dp.plot_sleep_raster(segments, gaps="removed")', needs=("one",), individual=True,
        data="the segments drawn", related=("load_sleep_segments",)),
    "plot_sleep_regularity": _t(
        P, "sleep", E, "Variability of the sleep midpoint, social jetlag and extra sleep on free nights, per "
                       "participant.",
        example="dp.plot_sleep_regularity(metrics)", individual=False, data="participant: the regularity measures",
        related=("sleep_regularity",), notes={"rule": _NIGHT_RULE}),
    "plot_sleep_timing": _t(
        P, "sleep", E, "Sleep midpoint against total sleep time.", example="dp.plot_sleep_timing(metrics)",
        individual=True, data="participant or night: midpoint and sleep", related=("plot_sleep",),
        notes={"rule": _NIGHT_RULE, "unit": "'participant' (the default) or 'night'."}),
    "plot_sleep_architecture": _t(
        P, "sleep", E, "Stage composition, wake after sleep onset and naps, per participant.",
        example="dp.plot_sleep_architecture(metrics)", individual=False,
        data="participant: stage shares, WASO, nap share", related=("plot_sleep",), notes={"rule": _NIGHT_RULE}),
    "plot_sleep_recording": _t(
        P, "sleep", E, "What each participant's nights record: measured sleep, time in bed only, or neither.",
        example="dp.plot_sleep_recording(metrics)", individual=True, data="participant: nights of each kind",
        related=("plot_sleep",)),
    "plot_cgm_ranges": _t(
        P, "glucose", E, "Each CGM participant's time in the consensus glucose ranges.",
        example="dp.plot_cgm_ranges(metrics, sufficient_only=False)", individual=True,
        data="participant: share of readings per range", related=("glucose_range", "plot_glycemic_cohort")),
    "plot_agp": _t(
        P, "glucose", E, "One participant's Ambulatory Glucose Profile with the consensus metrics, or the cohort's "
                         "glucose by time of day.",
        example="dp.plot_agp(metrics, wearer)", individual=True, panels=("metrics",),
        level_note="with participant=None, the cohort's glucose by time of day, an aggregate",
        data="percentiles per time-of-day bin", related=("cgm_profile", "plot_glucose_days"),
        notes={"participant": "One participant's AGP; None draws the cohort's glucose by time of day."}),
    "plot_glucose_days": _t(
        P, "glucose", E, "One participant's glucose day by day, with the median over the days.",
        example="dp.plot_glucose_days(readings)", needs=("one",), individual=True, panels=("median",),
        data="day, minute of day, glucose", related=("load_cgm_readings", "plot_agp")),
    "plot_glycemic_cohort": _t(
        P, "glucose", E, "Glycaemic variability, mean glucose with GMI, and the share of participants meeting each "
                         "consensus target.",
        example="dp.plot_glycemic_cohort(metrics, sufficient_only=False)", individual=False, panels=("targets",),
        data="participant: CV, mean glucose, GMI, ranges", related=("plot_cgm_ranges",)),
    "plot_cgm_wear": _t(
        P, "glucose", E, "Each CGM participant's days, colored by completeness: sensor changes and gaps.",
        example="dp.plot_cgm_wear(metrics)", individual=True, data="participant, day, readings, completeness",
        related=("plot_agp",)),
    "plot_bmi_categories": _t(
        P, "clinical", E, "Participants per WHO adult BMI class.", example="dp.plot_bmi_categories(summaries)",
        individual=False, data="class, participants, share", related=("bmi_categories",)),
    "plot_step_categories": _t(
        P, "clinical", E, "Participants per daily-step category (Tudor-Locke).",
        example="dp.plot_step_categories(summaries)", individual=False, data="category, participants, share",
        related=("step_categories",)),
    "plot_activity_goals": _t(
        P, "clinical", E, "Weekly exercise against the WHO's 150 minutes, and Apple ring goals met.",
        example="dp.plot_activity_goals(daily)", individual=False, panels=("weeks",),
        data="participant: weekly minutes, weeks meeting the guideline, goals met", related=("activity_goals",)),
    "plot_blood_pressure": _t(
        P, "clinical", E, "Blood-pressure categories under a selectable guideline, and each participant's medians.",
        example='dp.plot_blood_pressure(summaries, guideline="acc_aha")', individual=True, panels=("categories",),
        level_note="with points=False, the categories alone, an aggregate",
        data="participant: systolic, diastolic, category", related=("blood_pressure", "bp_category")),
    "plot_weight_trajectories": _t(
        P, "clinical", E, "Each participant's weight as change from their first measurement.",
        example="dp.plot_weight_trajectories(daily)", individual=True, panels=("cohort",),
        data="participant, study day, weight, percent change", related=("plot_daily_values",)),
    "plot_clinical_thresholds": _t(
        P, "clinical", E, "The share of each participant's readings beyond clinical thresholds, such as SpO2 below "
                          "90%.",
        example="dp.plot_clinical_thresholds(daily)", individual=False, panels=("cohort",),
        data="participant, threshold, readings beyond", related=("clinical_thresholds",)),
    "plot_co_availability": _t(
        P, "multimodal", E, "For each pair of features, the share of their participant-days they share.",
        example="dp.plot_co_availability(daily)", individual=False, data="feature pair, Jaccard index",
        related=("day_overlap",), notes={"features": "The features to pair; None pairs every feature."}),
    "plot_multimodal_days": _t(
        P, "multimodal", E, "Each participant's days holding every one of several features, and how many "
                            "participants each requirement keeps.",
        example='dp.plot_multimodal_days(daily, ["HeartRate", "StepCount"])', individual=False, panels=("curve",),
        data="participant, complete days", related=("days_with",),
        notes={"features": "The features that must all hold data on a day (at least two)."}),
    "plot_participant_overview": _t(
        P, "multimodal", E, "One participant on one time axis: which features have data each day, then steps, "
                            "resting heart rate, sleep, weight and glucose.",
        example="dp.plot_participant_overview(daily, sleeper, domain=metrics)", individual=True,
        panels=("availability",), data="feature, day, value, valid", related=("plot_availability_raster",),
        notes={"participant": "The participant to draw.",
               "features": "The features to draw; None draws steps, resting heart rate, sleep, weight and glucose."}),
    "plot_lagged_association": _t(
        P, "multimodal", E, "Within-person association between two daily series with an explicit lag: deviations "
                            "from each participant's own means.",
        example='dp.plot_lagged_association(daily, "sleep", "RestingHeartRate", lag_days=1, domain=metrics)',
        individual=True, panels=("participants", "summary"),
        level_note="with points=False, the per-participant correlations alone, an aggregate",
        data="paired days: x, y and their deviations", related=("plot_feature_correlations",)),
    "plot_feature_correlations": _t(
        P, "multimodal", E, "Correlations across participants between features' medians, with the participants "
                            "behind each pair.",
        example="dp.plot_feature_correlations(summaries)", individual=False,
        data="feature pair, correlation, participants", related=("plot_lagged_association",),
        notes={"features": "The features to correlate; None correlates every feature with participant medians."}),
    "report": _t(
        P, "output", E, "The standard figures in one PDF: a cohort overview, sleep and glucose pages, values and "
                        "clinical pages, and a dossier per feature.",
        "A DataFrame with one row per figure: its section, feature, title and whether it was drawn, skipped (with "
        "the reason) or left out.",
        'dp.report(daily, "{out}/report.pdf", features=["HeartRate"], sections=["values", "clinical"])',
        related=("plot_feature_presence", "plot_quality"),
        notes={"source": "The run to report on: the object compute_daily_statistics returned, or its directory.",
               "out": "The PDF to write, outside every data root.",
               "features": "The features whose pages to draw; None draws every feature of the run.",
               "coverage": "Coverage for the coverage pages; None computes it from the run's root when it can be read.",
               "domain": "Domain metrics for the sleep and glucose pages; None computes them from the run's root "
                         "when it can be read.",
               "dpi": "Resolution of the report's raster images."}),
    "activity_goals": _t(
        P, "clinical", B, "Weekly exercise minutes over complete weeks and Apple ring goals met, per participant.",
        "(participants, weeks): two DataFrames.", "dp.activity_goals(daily)", related=("plot_activity_goals",)),
    "blood_pressure": _t(
        P, "clinical", B, "Each participant's median systolic and diastolic pressure, and their category.",
        "A DataFrame, one row per participant.", "dp.blood_pressure(summaries)", related=("plot_blood_pressure",)),
    "bmi_categories": _t(
        P, "clinical", B, "Participants per WHO adult BMI class.", "A DataFrame.", "dp.bmi_categories(summaries)",
        related=("plot_bmi_categories",)),
    "bp_category": _t(
        P, "clinical", B, "The blood-pressure category of each systolic and diastolic pair: the highest either "
                          "pressure reaches.",
        "An array of category names.", 'dp.bp_category([128, 142], [79, 91], "esc_esh_office")',
        related=("blood_pressure",)),
    "step_categories": _t(
        P, "clinical", B, "Participants per daily-step category.", "A DataFrame.", "dp.step_categories(summaries)",
        related=("plot_step_categories",)),
    "feature_order": _t(
        P, "identity", B, "Features in figure order: by category, then by name.", "A list of feature names.",
        'dp.feature_order(["Sleep", "StepCount", "HeartRate"])', related=("feature_color", "feature_category"),
        notes={"features": "The features to order; None orders every feature."}),
    "feature_color": _t(
        P, "identity", B, "The color a feature has in every figure: its category's hue, in a shade of its own.",
        "A hex color.", 'dp.feature_color("HeartRate")', related=("feature_order",)),
    "feature_category": _t(
        P, "identity", B, "The category a feature's guide declares, such as 'Heart and cardiovascular'.",
        "A string.", 'dp.feature_category("HeartRate")', related=("feature_order",)),
}


# ------------------------------------------------------------------------------------------------ discovery
def _module(name: str) -> types.ModuleType:
    return importlib.import_module(f"wearable_project.utils.{name}")


def tools() -> dict[str, tuple[str, Any]]:
    """Every public function and class of the four modules, by name: ``(module, object)``."""

    found: dict[str, tuple[str, Any]] = {}
    for module_name in MODULES:
        module = _module(module_name)
        for name, obj in vars(module).items():
            if name.startswith("_"):
                continue
            target = getattr(obj, "__wrapped__", obj)  # functions wrapped by functools.lru_cache
            defined_here = getattr(target, "__module__", None) == module.__name__
            if defined_here and (inspect.isfunction(target) or inspect.isclass(obj)):
                found[name] = (module_name, obj)
    return found


def _is_figure(name: str) -> bool:
    return name.startswith("plot_")


def _parameters(obj) -> list[inspect.Parameter]:
    try:
        return list(inspect.signature(obj).parameters.values())
    except (TypeError, ValueError):
        return []


def _explain(tool: str, parameter: str) -> tuple[str, str]:
    """What a parameter means for a tool, and where that is written: the tool's notes or the shared glossary."""

    notes = dict(TOOLS[tool].notes) if tool in TOOLS else {}
    if parameter in notes:
        return notes[parameter], "tool"
    if parameter in PARAMETERS:
        return PARAMETERS[parameter], "shared"
    return "", "missing"


def _default(value) -> str:
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "..."


def _call(name: str) -> str:
    """The call as users write it, without type annotations."""

    module_name, obj = tools()[name]
    parts, starred = [], False
    for p in _parameters(obj):
        if p.kind is p.VAR_POSITIONAL:
            parts.append(f"*{p.name}")
            starred = True
            continue
        if p.kind is p.KEYWORD_ONLY and not starred:
            parts.append("*")
            starred = True
        if p.kind is p.VAR_KEYWORD:
            parts.append(f"**{p.name}")
            continue
        parts.append(p.name if p.default is p.empty else f"{p.name}={_default(p.default)}")
    return f"{MODULES[module_name][0]}.{name}({', '.join(parts)})"


def _fields(obj) -> list[str]:
    return [f.name for f in dataclasses.fields(obj)] if dataclasses.is_dataclass(obj) else []


def _methods(obj) -> list[str]:
    return [n for n, v in vars(obj).items() if not n.startswith("_") and inspect.isfunction(v)]


def _first_sentence(text: str) -> str:
    flat = " ".join(text.split())
    match = re.match(r"(.+?[.!?])(\s|$)", flat)
    return match.group(1) if match else flat


def _fill(template: str, phase: str, root, out) -> str:
    root_text = f", root={str(root)!r}" if root is not None else ""
    return (template.replace("{phase}", phase).replace("{root}", root_text)
            .replace("{out}", str(out) if out is not None else "outputs"))


def _setup_blocks(tool: str) -> list[str]:
    return ["base", *[n for n in TOOLS[tool].needs if n in SETUP]]


def _options(name: str) -> list[str]:
    return [p.name for p in _parameters(tools()[name][1]) if p.name in SHARED_OPTIONS]


def _group_title(key: str) -> str:
    return next(title for k, title, _ in GROUPS if k == key)


def _cli_help() -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.suppress(SystemExit):
        _module("data_statistics").main(["--help"])
    return buffer.getvalue().strip()


# ------------------------------------------------------------------------------------------------ rendering
def _wrap(text: str, indent: int = 2) -> list[str]:
    return textwrap.wrap(" ".join(str(text).split()), width=WIDTH, initial_indent=" " * indent,
                         subsequent_indent=" " * (indent + 2)) or [" " * indent]


def _paragraphs(text: str, indent: int = 2) -> list[str]:
    lines: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        if lines:
            lines.append("")
        lines.extend(_wrap(paragraph, indent))
    return lines


def _code(text: str, indent: int = 2) -> list[str]:
    return [" " * indent + line if line else "" for line in text.splitlines()]


def _document(title: str, subtitle: str, sections: list[tuple[str, list[str]]]) -> str:
    lines = [title, subtitle, ""]
    for heading, body in sections:
        if body:
            lines.extend([heading, *body, ""])
    return "\n".join(lines).rstrip()


def _report(kind: str, title: str, payload: dict[str, Any], text: str) -> InfoReport:
    return InfoReport(kind=kind, title=title, payload=payload, text=text)


# ---------------------------------------------------------------------------------------------------- cards
def _tool_payload(name: str, phase: str, root, out) -> dict[str, Any]:
    module_name, obj = tools()[name]
    meta = TOOLS[name]
    kind = "class" if inspect.isclass(obj) else "figure" if _is_figure(name) else "function"
    parameters = []
    for p in _parameters(obj) if kind != "class" else []:
        meaning, source = _explain(name, p.name)
        parameters.append({"name": p.name, "default": None if p.default is p.empty else _default(p.default),
                           "meaning": meaning, "explained_by": source})
    payload: dict[str, Any] = {
        "name": name, "module": module_name, "alias": MODULES[module_name][0], "kind": kind, "group": meta.group,
        "group_title": _group_title(meta.group), "tier": meta.tier, "call": _call(name), "does": meta.does,
        "returns": meta.returns, "parameters": parameters, "related": list(meta.related),
        "docstring": inspect.getdoc(obj) or "",
        "example": {"setup": {block: _fill(SETUP[block], phase, root, out) for block in _setup_blocks(name)},
                    "code": _fill(meta.example, phase, root, out), "needs_pyarrow": "pyarrow" in meta.needs},
    }
    if kind == "figure":
        payload.update({"options": _options(name), "individual_level": meta.individual, "level_note": meta.level_note,
                        "panels": list(meta.panels), "data": meta.data})
    if kind == "class":
        payload.update({"fields": [{"name": f, "meaning": FIELDS.get((name, f), "")} for f in _fields(obj)],
                        "methods": [{"name": m, "does": _first_sentence(inspect.getdoc(getattr(obj, m)) or "")}
                                    for m in _methods(obj)]})
    if name == "main":
        payload["cli_help"] = _cli_help()
    return payload


def _tool_text(p: dict[str, Any]) -> str:
    kind_label = {"figure": "figure", "class": "class", "function": "function"}[p["kind"]]
    sections: list[tuple[str, list[str]]] = [("Call", _wrap(p["call"]))]
    sections.append(("What it shows" if p["kind"] == "figure" else "What it is" if p["kind"] == "class" else
                     "What it does", _wrap(p["does"])))
    if p["kind"] == "figure":
        first = p["parameters"][0] if p["parameters"] else None
        if first:
            sections.append(("Draws from", _wrap(f"{first['name']}: {first['meaning']}")))
        level = "yes" if p["individual_level"] else "no"
        note = f" ({p['level_note']})" if p["level_note"] else ""
        sections.append(("Participants", _wrap(f"Shows individual participants: {level}, as in the example{note}. "
                                                f"figure.individual_level says so on every figure.")))
        if p["options"]:
            sections.append(("Shared options", _wrap(", ".join(p["options"]) + ". utils.info(\"<option>\") explains "
                                                                                 "each.")))
        panels = ", ".join(p["panels"]) if p["panels"] else "none"
        sections.append(("What it returns", _wrap(f"A matplotlib Figure. figure.data: {p['data']}. figure.panels: "
                                                   f"{panels}.")))
    elif p["returns"]:
        sections.append(("What it returns", _wrap(p["returns"])))
    if p["kind"] == "class":
        sections.append(("Fields", [line for f in p["fields"] for line in _wrap(f"{f['name']}: {f['meaning']}")]))
        if p["methods"]:
            sections.append(("Methods", [line for m in p["methods"] for line in _wrap(f"{m['name']}(): {m['does']}")]))
    if p["parameters"]:
        sections.append(("Parameters", [line for q in p["parameters"] for line in _wrap(f"{q['name']}: {q['meaning']}")]))
    if p.get("cli_help"):
        sections.append(("Command line", _code(p["cli_help"])))
    example = p["example"]
    if example["code"]:
        body = _wrap("Assumes the conventional names set up under Conventions in utils.info(): coverage, daily, "
                     "summaries, patterns, quality, metrics, code, sleeper and wearer.")
        for block in example["setup"]:
            if block != "base":
                body.extend(_wrap(f"And {SETUP_NOTES[block]}:"))
                body.extend(_code(example["setup"][block], 4))
        body.extend(["  Then:", *_code(example["code"], 4)])
        if example["needs_pyarrow"]:
            body.extend(_wrap("Needs pyarrow (or fastparquet), which the package does not require."))
        sections.append(("Example", body))
    if p["related"]:
        sections.append(("Related", _wrap(", ".join(p["related"]))))
    if p["docstring"]:
        sections.append(("Documentation", _paragraphs(p["docstring"])))
    subtitle = f"{p['group_title']} · {kind_label} · {p['tier']}"
    return _document(f"{p['name']} — {p['module']}", subtitle, sections)


def _parameter_report(name: str) -> InfoReport:
    takers = []
    for tool, (module_name, obj) in sorted(tools().items()):
        if any(p.name == name for p in _parameters(obj)):
            meaning, source = _explain(tool, name)
            takers.append({"tool": tool, "module": module_name, "meaning": meaning if source == "tool" else None})
    figures = [t["tool"] for t in takers if _is_figure(t["tool"])]
    payload = {"name": name, "meaning": PARAMETERS[name], "tools": takers, "figures": figures}
    sections = [("Meaning", _wrap(PARAMETERS[name])),
                ("Taken by", _wrap(", ".join(t["tool"] for t in takers) + f" ({len(takers)} tools, {len(figures)} "
                                                                         f"figures)."))]
    specific = [line for t in takers if t["meaning"] for line in _wrap(f"{t['tool']}: {t['meaning']}")]
    if specific:
        sections.append(("Where it means something more specific", specific))
    return _report("parameter", name, payload, _document(f"{name} — parameter", "shared by the statistics and figure "
                                                                                "tools", sections))


def _catalogue(names: list[str]) -> list[str]:
    lines: list[str] = []
    for tier in TIERS:
        chosen = [n for n in names if TOOLS[n].tier == tier]
        if chosen:
            lines.extend(_wrap(f"{tier.capitalize()}s:"))
            for n in chosen:
                lines.extend(_wrap(f"{n} — {TOOLS[n].does}", 4))
    return lines


def _group_report(key: str, module: str | None = None) -> InfoReport:
    title = _group_title(key)
    description = next(d for k, _, d in GROUPS if k == key)
    names = [n for n in TOOLS if TOOLS[n].group == key and (module is None or TOOLS[n].module == module)]
    payload = {"group": key, "title": title, "description": description,
               "tools": [{"name": n, "tier": TOOLS[n].tier, "does": TOOLS[n].does} for n in names]}
    text = _document(title, description, [("Tools", _catalogue(names))])
    return _report("group", title, payload, text)


def _resolve_group(question: str | None) -> str | None:
    if question is None:
        return None
    folded = question.strip().casefold()
    for key, title, _ in GROUPS:
        if folded in (key, title.casefold()):
            return key
    raise DataLoaderConfigurationError(f"unknown question {question!r}; choose one of: "
                                       + ", ".join(k for k, _, _ in GROUPS))


def _resolve_module(module: str | None) -> str | None:
    if module is None:
        return None
    folded = module.strip().casefold()
    for name, (alias, _) in MODULES.items():
        if folded in (name, alias):
            return name
    raise DataLoaderConfigurationError(f"unknown module {module!r}; choose one of: " + ", ".join(MODULES))


def _section_report(section: str, question: str | None, module: str | None) -> InfoReport:
    group, module_name = _resolve_group(question), _resolve_module(module)
    names = [n for n in TOOLS if (group is None or TOOLS[n].group == group)
             and (module_name is None or TOOLS[n].module == module_name)]
    if section == "figures":
        names = [n for n in names if _is_figure(n)]
        rows = [{"name": n, "shows": TOOLS[n].does, "options": _options(n), "individual_level": TOOLS[n].individual,
                 "level_note": TOOLS[n].level_note, "group": TOOLS[n].group} for n in names]
        body = []
        for r in rows:
            level = "individual" if r["individual_level"] else "cohort"
            extra = f"; {r['level_note']}" if r["level_note"] else ""
            body.extend(_wrap(f"{r['name']} — {r['shows']} [{level}{extra}; options: {', '.join(r['options']) or 'none'}]"))
        text = _document("Figures", f"{len(rows)} figures" + (f" · {_group_title(group)}" if group else ""),
                         [("Figures", body)])
        return _report("figures", "Figures", {"figures": rows}, text)
    if section == "parameters":
        rows = [{"name": n, "meaning": m, "tools": sum(any(p.name == n for p in _parameters(o)) for _, o in tools().values())}
                for n, m in sorted(PARAMETERS.items())]
        body = [line for r in rows for line in _wrap(f"{r['name']} ({r['tools']} tools): {r['meaning']}")]
        return _report("parameters", "Parameters", {"parameters": rows},
                       _document("Parameters", f"{len(rows)} parameters, each explained once", [("Parameters", body)]))
    sections = []
    for key, title, description in GROUPS:
        chosen = [n for n in names if TOOLS[n].group == key]
        if chosen:
            sections.append((title, _wrap(description) + _catalogue(chosen)))
    payload = {"groups": [{"group": k, "title": t, "tools": [n for n in names if TOOLS[n].group == k]}
                          for k, t, _ in GROUPS if any(TOOLS[n].group == k for n in names)]}
    return _report("tools", "Tools", payload, _document("Tools", f"{len(names)} tools, by question", sections))


def _overview(phase: str, root, out) -> InfoReport:
    found = tools()
    functions = [n for n, (_, o) in found.items() if not inspect.isclass(o)]
    classes = [n for n, (_, o) in found.items() if inspect.isclass(o)]
    figures = [n for n in functions if _is_figure(n)]
    ds, dm = _module("data_statistics"), _module("domain_metrics")
    pipeline = [
        ("coverage = ds.compute_coverage(phase)", "what each root holds, from the state databases alone"),
        ("daily = ds.compute_daily_statistics(phase)", "the run: per participant-day statistics of every feature"),
        ("summaries = sm.summarize(daily)", "participant metrics, adherence, retention and Table 1"),
        ("patterns = sm.temporal_patterns(daily)", "day of week, weekdays and weekend, month of year"),
        ("quality = sm.quality_report(daily)", "provenance, redundancy, plausibility and curation"),
        ("metrics = dm.compute_domain_metrics(phase)", "sleep nights, CGM days and periods, glucose profiles"),
        (f"dp.plot_...(any of the above)", f"{len(figures)} figures, each carrying the table it draws"),
        ('dp.report(daily, "report.pdf")', "the standard figures in one PDF"),
    ]
    options = {o: sum(o in _options(f) for f in figures) for o in SHARED_OPTIONS}
    schemas = {"statistics": ds.OUTPUT_SCHEMA, "domain": dm.DOMAIN_OUTPUT_SCHEMA}
    payload = {
        "version": __version__, "modules": {n: {"alias": a, "purpose": d, "tools": sum(1 for m, _ in found.values() if m == n)}
                                            for n, (a, d) in MODULES.items()},
        "counts": {"functions": len(functions), "classes": len(classes), "figures": len(figures)},
        "pipeline": [{"call": c, "what": w} for c, w in pipeline], "options": options, "schemas": schemas,
        "groups": [{"group": k, "title": t, "entry_points": [n for n in TOOLS if TOOLS[n].group == k and TOOLS[n].tier == E],
                    "building_blocks": [n for n in TOOLS if TOOLS[n].group == k and TOOLS[n].tier == B]} for k, t, _ in GROUPS],
        "setup": _fill(SETUP["base"], phase, root, out),
    }
    width = max(len(c) for c, _ in pipeline) + 2
    groups_body = []
    for g in payload["groups"]:
        blocks = len(g["building_blocks"])
        if g["entry_points"]:
            listed = ", ".join(g["entry_points"]) + (f" (and {blocks} building block{'s' if blocks != 1 else ''})" if blocks else "")
        else:
            listed = ", ".join(g["building_blocks"]) + " (building blocks)"
        groups_body.extend(_wrap(f"{g['title']}: {listed}"))
    sections = [
        ("The modules", [line for n, (a, d) in MODULES.items() for line in _wrap(f"{a} = {n}: {d}")]),
        ("The pipeline", [f"  {c.ljust(width)}{w}" for c, w in pipeline]),
        ("Tools by question", groups_body),
        ("Options many figures share", _wrap("; ".join(f"{o} ({'every figure' if n == len(figures) else f'{n} figures'})"
                                                       for o, n in options.items()) + ". Each figure's card lists those "
                                             "it accepts, and utils.info(\"<option>\") explains each.")),
        ("Written runs", _wrap(f"Statistics runs record output schema {schemas['statistics']} and domain runs "
                               f"{schemas['domain']}. A run written with another schema cannot be resumed: compute "
                               f"it again. Outputs are never written inside a data root.")),
        ("Conventions of the examples", _wrap(f"Every example assumes {SETUP_NOTES['base']}") + _code(payload["setup"], 4)),
        ("Asking", _wrap('utils.info("name") for a function, class, figure, parameter or question (such as "sleep"); '
                         'utils.info("tools"), utils.info("figures") and utils.info("parameters") for the catalogues, '
                         'which question= and module= filter; anything else searches. .as_dict() gives each report '
                         'structured.')),
    ]
    subtitle = (f"{len(MODULES)} modules · {len(functions)} functions, {len(figures)} of them figures · "
                f"{len(classes)} classes · version {__version__}")
    return _report("overview", "wearable_project.utils", payload,
                   _document("wearable_project.utils — statistics and figures", subtitle, sections))


# ---------------------------------------------------------------------------------------------------- search
def search(term: str, limit: int = 15) -> list[dict[str, Any]]:
    """Topics matching ``term``, best first: tools, parameters and questions."""

    words = [w for w in re.findall(r"[a-z0-9]+", term.casefold()) if len(w) > 1]
    phrase = " ".join(words)
    if not words:
        return []
    entries = []
    for name, meta in TOOLS.items():
        obj = tools()[name][1]
        entries.append(("tool", name, f"{meta.does} {meta.returns} {meta.data} {meta.level_note}",
                        inspect.getdoc(obj) or ""))
    entries += [("parameter", n, m, "") for n, m in PARAMETERS.items()]
    entries += [("question", k, f"{t} {d}", "") for k, t, d in GROUPS]
    results = []
    for kind, name, text, doc in entries:
        label, body, extra = name.casefold().replace("_", " "), text.casefold(), doc.casefold()
        score = 0
        if phrase == label or phrase.replace(" ", "_") == name.casefold():
            score += 100
        elif phrase in label:
            score += 60
        if phrase in body:
            score += 40
        for w in words:
            score += 15 * (w in label.split()) + 6 * body.count(w) + 1 * extra.count(w)
        if score:
            results.append({"topic": name, "kind": kind, "score": score,
                            "summary": TOOLS[name].does if kind == "tool" else text})
    return sorted(results, key=lambda r: (-r["score"], r["topic"]))[:limit]


def _search_report(term: str) -> InfoReport:
    results = search(term)
    body = [line for r in results for line in _wrap(f"{r['topic']} ({r['kind']}) — {r['summary']}")]
    return _report("search", term, {"term": term, "results": results},
                   _document(f"Search: {term}", f"{len(results)} topics, best first", [("Topics", body)]))


# ------------------------------------------------------------------------------------------------------ info
SECTIONS = ("overview", "tools", "figures", "parameters")


def info(topic: str | None = None, *, question: str | None = None, module: str | None = None,
         phase: str = "curated", root: str | Path | None = None, out: str | Path | None = None) -> InfoReport:
    """
    Help on the statistics and figure tools.
    Parameters
    ----------
    topic:
        None for the overview; a function, class or figure name (``"plot_agp"``, also ``"dp.plot_agp"``); a parameter
        (``"min_participants"``); a question (``"sleep"``); ``"tools"``, ``"figures"`` or ``"parameters"`` for the
        catalogues. Anything else searches. Matching ignores case.
    question, module:
        Filter the catalogues, such as ``info("figures", question="sleep")`` or ``info("tools", module="dm")``.
    phase, root, out:
        Fill the examples, so that each runs as printed for this phase and root, writing into ``out``. They are never
        accessed: ``info`` performs no data I/O.
    """

    if phase not in ("curated", "native"):
        raise DataLoaderConfigurationError("phase must be 'curated' or 'native'")
    if topic is None:
        return _section_report("tools", question, module) if (question or module) else _overview(phase, root, out)
    key = str(topic).strip()
    folded = key.casefold()
    for prefix in [f"{n}." for n in MODULES] + [f"{a}." for a, _ in MODULES.values()]:
        if folded.startswith(prefix):
            folded = folded[len(prefix):]
    if folded == "overview":
        return _overview(phase, root, out)
    if folded in SECTIONS:
        return _section_report(folded, question, module)
    by_name = {n.casefold(): n for n in TOOLS}
    if folded in by_name:
        name = by_name[folded]
        payload = _tool_payload(name, phase, root, out)
        return _report(payload["kind"], name, payload, _tool_text(payload))
    if folded in PARAMETERS:
        return _parameter_report(folded)
    group = next((k for k, t, _ in GROUPS if folded in (k, t.casefold())), None)
    if group is not None:
        return _group_report(group, _resolve_module(module))
    if search(key):
        return _search_report(key)
    candidates = [*TOOLS, *PARAMETERS, *(k for k, _, _ in GROUPS)]
    close = difflib.get_close_matches(folded, [c.casefold() for c in candidates], n=5, cutoff=0.6)
    hint = f"; did you mean {', '.join(close)}?" if close else "; utils.info() lists every topic"
    raise DataLoaderConfigurationError(f"no topic or match for {topic!r}{hint}")


def topics() -> list[str]:
    """Every topic ``info`` answers by name."""

    return [*SECTIONS, *TOOLS, *PARAMETERS, *(k for k, _, _ in GROUPS)]


# ----------------------------------------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m wearable_project.utils.info",
                                     description="Help on the statistics and figure tools.")
    parser.add_argument("topic", nargs="?", help="a function, figure, class, parameter or question; none for the overview")
    parser.add_argument("--question", help="filter the catalogues by question, such as sleep")
    parser.add_argument("--module", help="filter the catalogues by module: ds, sm, dm or dp")
    parser.add_argument("--phase", default="curated", choices=("curated", "native"))
    parser.add_argument("--root", help="fill the examples with this root")
    parser.add_argument("--json", action="store_true", help="print the structured report")
    args = parser.parse_args(argv)
    try:
        report = info(args.topic, question=args.question, module=args.module, phase=args.phase, root=args.root)
    except DataLoaderConfigurationError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(json.dumps(report.as_dict(), indent=2, default=str) if args.json else report.text)
    return 0


class _CallableModule(types.ModuleType):
    """``utils.info`` is this module; calling it calls ``info``, whatever the order things were imported in."""

    def __call__(self, *args, **kwargs) -> InfoReport:
        return info(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule


if __name__ == "__main__":
    raise SystemExit(main())
