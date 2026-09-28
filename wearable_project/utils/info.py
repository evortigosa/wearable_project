"""
Wearable Data Processing and Modeling project
Introspective help for the statistics and figure tools: ``data_statistics``, ``data_summaries``, ``domain_metrics``
and ``data_plots``.
As with ``DataLoaders.info``, whatever the code can state for itself is derived from it: signatures, docstrings,
the options each figure accepts, the fields of every result, the declared column lists, the value of every rule and
threshold, the output schema versions and the command line's own help. What the code cannot state is written here:
the question each tool answers, how the tools chain, what each parameter, field and column means, which participants
a figure shows, the concepts the tools rest on, the reason for each default, and workflows from start to finish.
Three test files hold those claims against the code and the samples: ``test_utils_info.py`` (every tool and
parameter, and every example runs), ``test_utils_info_tables.py`` (every table and column against real outputs in
both phases) and ``test_utils_info_guide.py`` (every workflow runs as printed, every default is read live and
complete, and every name opens its own card).
Typical notebook use:
    from wearable_project import utils

    utils.info()                              # the map: modules, pipeline, tools by question
    utils.info("compute_daily_statistics")    # one tool
    utils.info("plot_agp")                    # one figure
    utils.info("min_participants")            # one parameter, and the tools that take it
    utils.info("sleep")                       # the tools for one question
    utils.info("DomainMetrics.sleep_nights")  # one output table, column by column
    utils.info("valid days")                  # one concept, with the current values of its rules
    utils.info("CGM_SUFFICIENT_DAYS")         # one default: value, source, basis, the tools it governs
    utils.info("shareable report")            # one workflow, as code that runs as printed
    utils.info("jetlag")                      # anything else: a ranked search
    utils.info("plot_agp").as_dict()          # the same, structured
``info`` describes the tools and never touches data. From a shell: ``python -m wearable_project.utils.info plot_agp``.
"""


from __future__ import annotations
import argparse
import calendar
import contextlib
import dataclasses
import difflib
import functools
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
# Options many figures share; a figure's card lists its signature accepts.
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
    "lag_days": "Days from x to y: x on day d is paired with y on day d + lag_days (a whole number of days from -30 to 30; negative when y comes first).",
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
    "night_rule": "Which nights count: NightRule(min_asleep_minutes={d:night_rule}) by default.",
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
    "sufficient_only": "Only CGM participants with sufficient data, at least {d:cgm_sufficient_days} valid days (the default).",
    "summaries": "What summarize(...) returned.",
    "systolic": "Systolic pressures in mmHg, aligned with diastolic.",
    "thresholds": "The threshold values to try; None chooses a range suited to the rule's criterion.",
    "time_axis": "'calendar' (dates) or 'study' (days since each participant's first). A figure of one participant "
                 "defaults to study days, so that real dates appear only when asked for.",
    "typical_gap_minutes": "The typical gap between records, from participant_feature.",
    "unit": "What counts once in a cohort distribution: 'participant' (the default: each participant's median day "
            "or night), or each day or night pooled ('participant_day', 'night').",
    "valid_only": "Draw valid days only (the default for cohort figures).",
    "weekend_days": "The weekend, Monday being 0: {c:WEEKEND_DAYS}, {d:weekend_days}, by default. A night is free when it "
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
    ("DomainMetrics", "cgm_profile"): "Each CGM participant's glucose percentiles per {c:AGP_BIN_MINUTES}-minute bin of the day, over "
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
_NIGHT_RULE = "The night rule: NightRule(min_asleep_minutes={d:night_rule}) by default."
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
                          "38.0 °C, per participant and for the cohort; counted only where the values are in the "
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


# --------------------------------------------------------------------------------------------- output tables
@dataclass(frozen=True)
class _Table:
    """An output table: what a row is, its columns, and which tools return it."""

    rows: str
    columns: tuple[str, ...] = ()
    returned_by: tuple[str, ...] = ()
    derived_from: str = ""            # "module.CONSTANT" holding the column list, when the module declares one
    extends: str = ""                 # another table whose columns come first
    curated_only: tuple[str, ...] = ()
    optional: tuple[tuple[str, str], ...] = ()   # (column, when it is present)
    dynamic: str = ""                 # columns named at run time, explained
    index: tuple[str, ...] = ()
    notes: tuple[tuple[str, str], ...] = ()      # meanings specific to this table
    read_back: str = ""               # the call reading it from a written run


# The daily tables, one per feature, have columns that depend on the feature's measurement kind and the phase.
DAILY_COMMON = ("RegistrationCode", "feature", "local_date", "day_basis", "records", "records_user_entered",
                "records_without_offset", "redundant_minutes", "sources", "value_unit", "values_above_range",
                "values_below_range")
DAILY_CURATED_ONLY = ("records_included",)
_DAILY_TIME = ("hours_with_data", "median_gap_minutes", "max_gap_minutes", "observed_minutes")
_DAILY_LEVEL = _DAILY_TIME + ("value_mean", "value_median", "value_min", "value_max", "value_count", "values_unresolved")
DAILY_BY_KIND = {
    "categorical_state": _DAILY_TIME + ("minutes_asleep", "minutes_asleep_total", "minutes_awake", "minutes_core",
                                        "minutes_deep", "minutes_inbed", "minutes_other_states", "minutes_rem"),
    "duration": _DAILY_TIME + ("minutes", "sessions"),
    "event_amount": _DAILY_TIME + ("value_sum", "value_count", "values_unresolved"),
    "extensive_total": _DAILY_TIME + ("value_sum", "values_unresolved"),
    "intensive_value": _DAILY_LEVEL,
    "ratio": _DAILY_LEVEL,
    "summary_statistic": _DAILY_LEVEL,
    "multivariate_point": _DAILY_TIME,
    "multivariate_summary": (),
    "signal": _DAILY_TIME,
}
DAILY_BY_FEATURE = {
    "BloodPressure": ("blood_pressure_systolic_value_mean", "blood_pressure_systolic_value_count",
                      "blood_pressure_diastolic_value_mean", "blood_pressure_diastolic_value_count"),
    "ActivitySummary": ("active_energy_burned_mean", "active_energy_burned_goal_mean", "apple_exercise_time_mean",
                        "apple_exercise_time_goal_mean", "apple_stand_hours_mean", "apple_stand_hours_goal_mean"),
    "Electrocardiogram": ("average_heart_rate_mean", "sampling_frequency_mean"),
}


def daily_columns(feature: str, phase: str = "curated") -> list[str]:
    """The columns of a feature's daily table in a phase: shared ones, then its measurement kind's, then its own."""

    from wearable_project.utils import data_statistics as ds
    kind = ds.measurement_kind(feature)
    thresholds = [name for name, *_ in ds.CLINICAL_THRESHOLDS.get(feature, ())]
    return [*DAILY_COMMON, *(DAILY_CURATED_ONLY if phase == "curated" else ()), *DAILY_BY_KIND.get(kind, ()),
            *DAILY_BY_FEATURE.get(feature, ()), *thresholds]


_CI = (("ci_low", "with ci="), ("ci_high", "with ci="))
_HOUR_STATS = tuple(f"{m}_{s}" for m in ("records_per_day", "value_per_day", "value_mean", "day_share")
                    for s in ("median", "p25", "p75", "participants"))
_COHORT_POINT = ("group", "participants", "median", "p25", "p75", "suppressed")

TABLES: dict[str, _Table] = {
    # ------------------------------------------------------------------------------------- the statistics run
    "DailyStatistics.daily": _Table(
        "one per participant and local day, in one table per feature (daily[feature])",
        read_back='ds.read_daily(run_dir, "<feature>")', returned_by=(
            "compute_daily_statistics", "read_daily", "iter_daily", "summarize_result"),
        dynamic="Its columns depend on the feature and the phase: utils.info.daily_columns(feature, phase) lists them."),
    "DailyStatistics.hourly": _Table(
        "one per participant, feature and local hour of day, all 24 hours present",
        read_back='ds.read_table(run_dir, "hourly_profile")', derived_from="data_statistics.HOURLY_COLUMNS",
        returned_by=("compute_daily_statistics", "read_table", "summarize_result"),
        notes=(("records", "Records starting in the hour, over all the participant's days."),
               ("days", "Days on which the hour holds data: an event in it, or an interval overlapping it."),
               ("value_sum", "The amount recorded in the hour over all days, for totals and event amounts (0 where "
                             "none was recorded); NaN for levels."),
               ("value_mean", "The mean value of the readings starting in the hour, for levels."),
               ("value_count", "Values recorded in the hour."))),
    "DailyStatistics.participant_feature": _Table(
        "one per participant and feature", ("RegistrationCode", "feature", "days_with_data", "first_day", "last_day",
                                            "records", "typical_gap_minutes", "regular_share"),
        returned_by=("compute_daily_statistics", "read_table"), read_back='ds.read_table(run_dir, "participant_feature")',
        notes=(("records", "The participant's records of the feature."),)),
    "DailyStatistics.participant_days": _Table(
        "one per participant and local day with any data", read_back='ds.read_table(run_dir, "participant_days")',
        columns=("RegistrationCode", "local_date", "features", "utc_offsets"),
        returned_by=("compute_daily_statistics", "read_table"),
        notes=(("features", "The features with data that day, separated by semicolons."),)),
    "DailyStatistics.cohort_daily": _Table(
        "one per feature and local day, across participants", ("feature", "local_date", "participants", "records",
                                                               "value_sum", "value_sum_per_participant"),
        returned_by=("compute_daily_statistics", "read_table"), read_back='ds.read_table(run_dir, "cohort_daily")',
        notes=(("value_sum", "The day's total across participants, for totals and event amounts."),)),
    "DailyStatistics.active_participants": _Table(
        "one per local day", ("local_date", "participants"), returned_by=("compute_daily_statistics", "read_table"),
        read_back='ds.read_table(run_dir, "active_participants")',
        notes=(("participants", "Participants with any data that day."),)),
    "DailyStatistics.provenance": _Table(
        "one per participant, feature, local day and acquisition method",
        read_back='ds.read_table(run_dir, "daily_provenance")', derived_from="data_statistics.PROVENANCE_COLUMNS",
        returned_by=("compute_daily_statistics", "read_table", "provenance_tables")),
    "DailyStatistics.curation": _Table(
        "one per participant, feature, local day and curation status or flag (curated phase; empty in native)",
        read_back='ds.read_table(run_dir, "daily_curation")',
        derived_from="data_statistics.CURATION_COLUMNS", returned_by=("compute_daily_statistics", "read_table",
                                                                     "provenance_tables")),
    # ------------------------------------------------------------------------------------------------ coverage
    "Coverage.participant_feature": _Table(
        "one per participant and feature with a file", ("RegistrationCode", "feature", "rows", "bytes_on_disk", "verifiable"),
        returned_by=("compute_coverage",), notes=(("rows", "Rows in the participant's file for the feature."),),
        curated_only=("pass_rows", "review_rows", "exclude_default_rows", "included_by_default_rows",
                      "excluded_by_default_rows", "canonical_value_rows", "ambiguous_unit_rows",
                      "acquisition_classified_fraction")),
    "Coverage.features": _Table(
        "one per feature", ("participants", "rows", "bytes_on_disk", "median_rows_per_participant",
                                                "max_rows_per_participant"), returned_by=("compute_coverage",),
        index=("feature",), notes=(("participants", "Participants with a file for the feature."),
                                   ("rows", "Rows across participants' files."),
                                   ("bytes_on_disk", "Bytes the feature's files take on disk.")), curated_only=("pass_rows", "review_rows", "exclude_default_rows", "included_by_default_rows",
                                          "excluded_by_default_rows", "canonical_value_rows", "ambiguous_unit_rows")),
    "Coverage.participants": _Table(
        "one per participant", ("features", "rows", "bytes_on_disk"),
        returned_by=("compute_coverage",), index=("RegistrationCode",),
        notes=(("features", "The number of features the participant has."),
               ("rows", "Rows across the participant's files."),
               ("bytes_on_disk", "Bytes the participant's files take on disk."))),
    "Coverage.presence": _Table(
        "one per participant: Coverage.presence()", returned_by=("Coverage",),
        index=("RegistrationCode",), dynamic="One column per feature: True where the participant has a file for it."),
    # ---------------------------------------------------------------------------------------------- summaries
    "Summaries.participant_metrics": _Table(
        "one per participant, feature and metric, over valid days", ("RegistrationCode", "feature", "metric", "unit",
        "n_days", "mean", "sd", "median", "p10", "p25", "p75", "p90", "min", "max"), returned_by=("summarize",),
        notes=(("mean", "The mean of the participant's valid days."), ("sd", "The standard deviation over valid days."),
               ("median", "The participant's median valid day."), ("min", "The lowest valid day."),
               ("max", "The highest valid day."))),
    "Summaries.adherence": _Table(
        "one per participant and feature", ("RegistrationCode", "feature", "rule", "first_day", "last_day", "span_days",
        "days_with_data", "valid_days", "adherence", "first_valid_day", "last_valid_day", "longest_valid_run", "gaps",
        "longest_gap_days", "median_gap_days", "typical_gap_minutes", "regular_share", "expected_records_per_day",
        "median_completeness"), returned_by=("summarize",)),
    "Summaries.cohort": _Table(
        "Table 1: one per feature and metric, across participants' medians", ("feature", "metric", "unit",
        "participants", "mean", "sd", "p5", "p25", "median", "p75", "p95", "min", "max"),
        returned_by=("summarize", "cohort_table"),
        notes=(("mean", "The mean of participants' medians."), ("sd", "The standard deviation of participants' medians."),
               ("median", "The median of participants' medians."), ("min", "The lowest participant median."),
               ("max", "The highest participant median."))),
    "Summaries.retention": _Table(
        "one per feature and day since the first valid day", ("feature", "days_since_first_valid", "participants",
                                                              "fraction"), returned_by=("summarize", "retention"),
        notes=(("participants", "Participants still contributing valid days that many days after their first."),)),
    "summarize_domain.participant_metrics": _Table(
        "one per participant and night or CGM metric", extends="Summaries.participant_metrics",
        returned_by=("summarize_domain",)),
    "summarize_domain.cohort": _Table(
        "Table 1 for nights and CGM days: one per domain metric, across participants' medians",
        extends="Summaries.cohort", returned_by=("summarize_domain",)),
    "summarize_domain.retention": _Table(
        "one per domain and day since the first valid night or CGM day", extends="Summaries.retention",
        returned_by=("summarize_domain",)),
    "summarize_domain.adherence": _Table(
        "one per participant and domain (nights, CGM days)", ("RegistrationCode", "feature", "rule", "first_day",
        "last_day", "span_days", "days_with_data", "valid_days", "adherence", "first_valid_day", "last_valid_day",
        "longest_valid_run", "gaps", "longest_gap_days", "median_gap_days", "nights_in_bed_only"),
        returned_by=("summarize_domain",)),
    # ----------------------------------------------------------------------------------------------- patterns
    "TemporalPatterns.day_of_week": _Table(
        "one per participant, feature, metric and weekday", ("RegistrationCode", "feature", "metric", "weekday",
                                                            "n_days", "median"), returned_by=("temporal_patterns",),
        notes=(("median", "The participant's median valid day on that weekday."),)),
    "TemporalPatterns.weekday_weekend": _Table(
        "one per participant, feature and metric", ("RegistrationCode", "feature", "metric", "weekday_days",
        "weekday_median", "weekend_days", "weekend_median", "weekend_minus_weekday"), returned_by=("temporal_patterns",)),
    "TemporalPatterns.month_of_year": _Table(
        "one per participant, feature, metric and month of year", ("RegistrationCode", "feature", "metric", "month",
                                                                   "n_days", "median"), returned_by=("temporal_patterns",),
        notes=(("month", "The month of year, 1 to 12."), ("median", "The participant's median valid day in that month."))),
    "TemporalPatterns.cohort_day_of_week": _Table(
        "one per feature, metric and weekday, across participants' medians", ("feature", "metric", "weekday",
        "participants", "median", "p25", "p75"), returned_by=("temporal_patterns",)),
    "TemporalPatterns.cohort_weekday_weekend": _Table(
        "one per feature and metric: participants' weekend-minus-weekday differences", ("feature", "metric",
        "participants", "median", "p25", "p75"), returned_by=("temporal_patterns",)),
    "TemporalPatterns.cohort_month_of_year": _Table(
        "one per feature, metric and month of year, across participants' medians", ("feature", "metric", "month",
        "participants", "median", "p25", "p75"), returned_by=("temporal_patterns",),
        notes=(("month", "The month of year, 1 to 12."),)),
    # ------------------------------------------------------------------------------------------------ quality
    "QualityReport.features": _Table(
        "one per feature", ("feature", "participants", "participant_days", "records", "records_user_entered",
        "observed_minutes", "redundant_minutes", "values_below_range", "values_above_range", "days_assessed",
        "records_without_offset", "without_offset_share", "redundant_share"), returned_by=("quality_report",)),
    "QualityReport.provenance": _Table(
        "one per feature and acquisition method", ("feature", "acquisition_method", "records", "records_user_entered",
                                                   "share"), returned_by=("quality_report",),
        notes=(("share", "The method's share of the feature's records."),)),
    "QualityReport.curation": _Table(
        "one per feature and curation status or flag (curated phase; empty in native)", ("feature", "kind", "name",
        "records", "share_of_records"), returned_by=("quality_report",)),
    "hour_of_day.participants": _Table(
        "one per participant, feature and local hour", ("RegistrationCode", "feature", "hour", "records_per_day",
        "value_per_day", "value_mean", "day_share"), returned_by=("hour_of_day",)),
    "hour_of_day.cohort": _Table(
        "one per feature and local hour, across participants", ("feature", "hour", *_HOUR_STATS),
        returned_by=("hour_of_day",)),
    "clinical_thresholds.participants": _Table(
        "one per participant, feature and threshold", derived_from="data_summaries.CLINICAL_COLUMNS",
        returned_by=("clinical_thresholds",)),
    "clinical_thresholds.cohort": _Table(
        "one per feature and threshold, across participants", ("feature", "threshold", "participants",
        "participants_beyond", "readings", "beyond", "days", "days_beyond", "share_beyond"),
        returned_by=("clinical_thresholds",)),
    "time_zones": _Table(
        "one per participant with UTC offsets", ("RegistrationCode", "days_with_offsets", "home_offsets",
        "distinct_offsets", "days_away", "trips", "longest_trip_days", "max_offset_difference_hours"),
        returned_by=("time_zones",)),
    "day_overlap": _Table(
        "one per pair of features", ("feature_a", "feature_b", "days_both", "days_a", "days_b", "jaccard"),
        returned_by=("day_overlap",)),
    "days_with": _Table(
        "one per participant with at least one day holding every feature", ("RegistrationCode", "days", "first_day",
                                                                             "last_day"), returned_by=("days_with",),
        notes=(("days", "Days on which every one of the features has data."),
               ("first_day", "The first such day."), ("last_day", "The last such day."))),
    "mark_valid_days": _Table(
        "the participant's daily rows, with two columns added", extends="DailyStatistics.daily",
        columns=("completeness", "valid"), returned_by=("mark_valid_days",)),
    # ------------------------------------------------------------------------------------------ domain metrics
    "DomainMetrics.sleep_nights": _Table(
        "one per participant and night, from noon to the next noon", derived_from="domain_metrics.NIGHT_COLUMNS",
        read_back='dm.read_domain_table(domain_dir, "sleep_nights")',
        returned_by=("compute_domain_metrics", "sleep_nights", "read_domain_table"),
        notes=(("records", "Sleep records in the night."),)),
    "DomainMetrics.cgm_days": _Table(
        "one per CGM participant and local day", derived_from="domain_metrics.CGM_DAY_COLUMNS",
        read_back='dm.read_domain_table(domain_dir, "cgm_days")',
        returned_by=("compute_domain_metrics", "cgm_metrics", "read_domain_table"),
        notes=(("readings", "Readings that day."),)),
    "DomainMetrics.cgm_periods": _Table(
        "one per participant with BloodGlucose", derived_from="domain_metrics.CGM_PERIOD_COLUMNS",
        read_back='dm.read_domain_table(domain_dir, "cgm_periods")',
        returned_by=("compute_domain_metrics", "cgm_metrics", "read_domain_table"),
        notes=(("readings", "Readings on valid days."),)),
    "DomainMetrics.cgm_profile": _Table(
        "one per CGM participant and time-of-day bin", derived_from="domain_metrics.CGM_PROFILE_COLUMNS",
        read_back='dm.read_domain_table(domain_dir, "cgm_profile")',
        returned_by=("compute_domain_metrics", "cgm_profile", "read_domain_table"),
        notes=(("readings", "Readings in the bin over valid days."), ("days", "Valid days with readings in the bin."))),
    "load_sleep_segments": _Table(
        "one per piece of a sleep record, split at local noon", derived_from="domain_metrics.SEGMENT_COLUMNS",
        returned_by=("load_sleep_segments", "sleep_segments"),
        notes=(("minutes", "The piece's length."),)),
    "load_cgm_readings": _Table(
        "one per CGM reading with a usable unit", derived_from="domain_metrics.CGM_READING_COLUMNS",
        returned_by=("load_cgm_readings", "cgm_readings")),
    "sleep_regularity": _Table(
        "one per participant with valid nights", derived_from="domain_metrics.REGULARITY_COLUMNS",
        returned_by=("sleep_regularity",), notes=(("nights", "Valid nights."),)),
    # ------------------------------------------------------------------------------- tables figure helpers return
    "activity_goals.participants": _Table(
        "one per participant with ActivitySummary", ("RegistrationCode", "days", "complete_weeks",
        "weekly_exercise_minutes", "weeks_meeting_who", "move_goal_days", "move_goal_met", "exercise_goal_days",
        "exercise_goal_met", "stand_goal_days", "stand_goal_met"), returned_by=("activity_goals",),
        notes=(("days", "Days with ActivitySummary."),)),
    "activity_goals.weeks": _Table(
        "one per participant and week", ("week_start", "days", "known", "exercise_minutes", "complete",
                                         "RegistrationCode"), returned_by=("activity_goals",),
        notes=(("days", "Days of the week with ActivitySummary."),)),
    "blood_pressure": _Table(
        "one per participant with blood pressure", ("RegistrationCode", "systolic", "diastolic", "days", "category"),
        returned_by=("blood_pressure",), notes=(("days", "Valid days with blood pressure."),)),
    "bmi_categories": _Table(
        "one per group and WHO adult BMI class", ("group", "category", "lower", "upper", "participants", "share"),
        returned_by=("bmi_categories",), notes=(("category", "The BMI class."),)),
    "step_categories": _Table(
        "one per group and daily-step category", ("group", "category", "lower", "upper", "participants", "share"),
        returned_by=("step_categories",), notes=(("category", "The daily-step category."),)),
    "report": _Table(
        "one per page the report considered", ("section", "feature", "figure", "status", "reason"),
        returned_by=("report",)),
    # ---------------------------------------------------------------------------------------- figures: data
    "plot_feature_presence.data": _Table("one per participant and feature", ("RegistrationCode", "feature", "present"),
                                         returned_by=("plot_feature_presence",)),
    "plot_participants_per_feature.data": _Table(
        "one per feature with participants", ("feature", "participants", "category", "share", "suppressed"),
        returned_by=("plot_participants_per_feature",), notes=(("category", "The feature's category."),
                                                               ("share", "The feature's participants as a share of the cohort."))),
    "plot_features_per_participant.data": _Table(
        "one per group and number of features", ("features", "participants", "group", "suppressed"),
        returned_by=("plot_features_per_participant",), notes=(("features", "A number of features."),
                                                               ("participants", "Participants with that many features."))),
    "plot_data_volume.data": _Table(
        "one per feature or per participant", returned_by=("plot_data_volume",),
        optional=(("feature", "by='feature'"), ("bytes_on_disk", "by='feature'"), ("RegistrationCode", "by='participant'"),
                  ("bytes", "by='participant'"), ("band", "by='participant'"))),
    "plot_feature_combinations.data": _Table(
        "one per drawn combination of features", ("participants", "size", "combination", "suppressed"),
        returned_by=("plot_feature_combinations",), dynamic="One column per chosen feature: whether the combination "
                                                            "includes it."),
    "plot_availability_raster.data": _Table(
        "one per participant and month", ("RegistrationCode", "days"), returned_by=("plot_availability_raster",),
        optional=(("month", "time_axis='calendar', the default"), ("study_month", "time_axis='study'")),
        notes=(("days", "Days with data in the month."), ("month", "The calendar month."))),
    "plot_follow_up.data": _Table(
        "one per participant", ("RegistrationCode", "first_day", "last_day", "span_days", "days_with_data", "density"),
        returned_by=("plot_follow_up",)),
    "plot_valid_day_rule.data": _Table(
        "one per threshold and measure", ("threshold", "measure", "k", "count", "share", "suppressed", "feature",
                                          "criterion"), returned_by=("plot_valid_day_rule",),
        notes=(("measure", "Either participant-days kept, or participants with at least k valid days."),
               ("count", "The participant-days kept, or the participants with at least k valid days."),
               ("share", "count as a share of all participant-days, or of all participants."),
               ("criterion", "The rule's criterion: hours_with_data, completeness or records."),
               ("k", "The k of 'participants with at least k valid days'; empty for days kept."))),
    "plot_sampling_cadence.data": _Table(
        "one per participant and feature with a cadence", ("RegistrationCode", "feature", "typical_gap_minutes",
        "regular_share", "days_with_data", "records"), returned_by=("plot_sampling_cadence",)),
    "plot_hourly_coverage.data": _Table("one per participant and local hour", ("RegistrationCode", "hour", "day_share"),
                                        returned_by=("plot_hourly_coverage",)),
    "plot_quality.data": _Table(
        "one per feature", ("feature", "category", "records", "below_per_1000", "above_per_1000", "redundant_share",
                            "user_entered_share", "without_offset_share"), returned_by=("plot_quality",),
        notes=(("category", "The feature's category."),)),
    "plot_curation_flags.data": _Table("one per feature and flag present", ("feature", "flag", "share_of_records"),
                                       returned_by=("plot_curation_flags",)),
    "plot_acquisition.data": _Table("one per feature and acquisition method", ("feature", "acquisition_method", "share"),
                                    returned_by=("plot_acquisition",),
                                    notes=(("share", "The method's share of the feature's records."),)),
    "plot_acquisition_over_time.data": _Table(
        "one per calendar month and acquisition method", ("month", "acquisition_method", "records", "share"),
        returned_by=("plot_acquisition_over_time",),
        notes=(("month", "The calendar month."), ("share", "The method's share of the month's records."))),
    "plot_curation.data": _Table("one per feature and curation status", ("feature", "status", "share"),
                                 returned_by=("plot_curation",),
                                 notes=(("share", "The status's share of the feature's records."),)),
    "plot_active_participants.data": _Table(
        "one per local day with data", ("local_date", "participants", "suppressed"), returned_by=("plot_active_participants",),
        optional=(("rolling_mean", "with rolling="),), notes=(("participants", "Participants with any data that day."),)),
    "plot_feature_activity.data": _Table(
        "one per feature and calendar month", ("feature", "month", "suppressed"), returned_by=("plot_feature_activity",),
        optional=(("participants", "measure='participants'"), ("participant_days", "measure='participant_days'")),
        notes=(("month", "The calendar month."), ("participants", "Participants with the feature that month."))),
    "plot_daily_values.data": _Table(
        "one per day drawn: the participant's days, or the cohort's days", returned_by=("plot_daily_values",),
        optional=(("RegistrationCode", "with a participant"), ("value", "with a participant"),
                  ("valid", "with a participant"), ("group", "for the cohort"), ("participants", "for the cohort"),
                  ("median", "for the cohort"), ("p25", "for the cohort"), ("p75", "for the cohort"),
                  ("suppressed", "for the cohort"), ("feature", "for the cohort"), ("metric", "for the cohort"),
                  ("study_day", "with a participant"),
                  ("local_date", "for the cohort, or with a participant and time_axis='calendar'")),
        notes=(("value", "The participant's value that day."),)),
    "plot_monthly_distribution.data": _Table(
        "one per group and period", ("group", "period", "participants", "values", "n_days", "median", "p25", "p75",
                                     "suppressed"), returned_by=("plot_monthly_distribution",),
        notes=(("period", "The calendar month or quarter."), ("values", "The values drawn in the box."),
               ("n_days", "Participant-days in the period."))),
    "plot_hour_of_day.data": _Table(
        "one per group and local hour", ("group", "hour", "participants", "median", "p25", "p75", "suppressed",
                                         "feature", "measure"), returned_by=("plot_hour_of_day",), optional=_CI,
        notes=(("measure", "What was drawn: value, records or coverage."),)),
    "plot_weekly_pattern.data": _Table(
        "one per group and weekday", ("group", "weekday", "participants", "median", "p25", "p75", "suppressed",
                                      "feature", "metric"), returned_by=("plot_weekly_pattern",), optional=_CI),
    "plot_monthly_pattern.data": _Table(
        "one per group and month of year", ("group", "month", "participants", "median", "p25", "p75", "suppressed",
                                            "feature", "metric"), returned_by=("plot_monthly_pattern",), optional=_CI,
        notes=(("month", "The month of year, 1 to 12."),)),
    "plot_weekday_weekend.data": _Table("one per participant drawn", extends="TemporalPatterns.weekday_weekend",
                                        columns=("group",), returned_by=("plot_weekday_weekend",)),
    "plot_hour_profiles.data": _Table(
        "one per participant and local hour", ("RegistrationCode", "hour"), returned_by=("plot_hour_profiles",),
        optional=(("relative", "relative=True"), ("value_mean", "relative=False, measure='value' for levels"),
                  ("value_per_day", "relative=False, measure='value' for totals"),
                  ("records_per_day", "relative=False, measure='records'"), ("day_share", "relative=False, measure='coverage'"))),
    "plot_daily_rhythms.data": _Table("one per feature, group and local hour", (*_COHORT_POINT[:1], "hour",
                                      *_COHORT_POINT[1:], "feature"), returned_by=("plot_daily_rhythms",),
                                      notes=(("median", "The median across participants of their relative value."),)),
    "plot_time_zones.data": _Table(
        "one per participant, or one per day of one participant", returned_by=("plot_time_zones",),
        optional=(*((c, "for the cohort") for c in ("RegistrationCode", "days_with_offsets", "home_offsets",
                  "distinct_offsets", "days_away", "trips", "longest_trip_days", "max_offset_difference_hours",
                  "home_zone")), ("offsets", "with a participant"), ("offset_hours", "with a participant"),
                  ("home", "with a participant"), ("study_day", "with a participant"),
                  ("local_date", "with a participant, time_axis='calendar'"))),
    "plot_metric_distributions.data": _Table(
        "one per participant and feature", ("RegistrationCode", "median", "feature", "metric", "unit", "group"),
        returned_by=("plot_metric_distributions",), notes=(("median", "The participant's median valid day."),)),
    "plot_caterpillar.data": _Table(
        "one per participant drawn", ("RegistrationCode", "feature", "metric", "unit", "n_days", "median", "p25", "p75"),
        returned_by=("plot_caterpillar",), notes=(("median", "The participant's median valid day."),
                                                  ("p25", "The participant's 25th-percentile valid day."),
                                                  ("p75", "The participant's 75th-percentile valid day."))),
    "plot_adherence.data": _Table("one per participant", extends="Summaries.adherence", columns=("group",),
                                  returned_by=("plot_adherence",)),
    "plot_retention.data": _Table("one per feature or group and day", extends="Summaries.retention",
                                  columns=("group", "suppressed"), returned_by=("plot_retention",)),
    "plot_sleep.data": _Table(
        "one per participant (their median night) or per night", ("RegistrationCode", "asleep_minutes",
        "midpoint_hours_after_noon", "efficiency", "group"), returned_by=("plot_sleep",),
        optional=(("nights", "unit='participant'"), ("night_date", "unit='night'"))),
    "plot_sleep_raster.data": _Table(
        "one per segment drawn", ("night_index", "state", "source", "start_hours_after_noon", "end_hours_after_noon",
                                  "minutes", "main_period", "counted"), returned_by=("plot_sleep_raster",),
        optional=(("night_date", "time_axis='calendar'"), ("start_local", "time_axis='calendar'"),
                  ("end_local", "time_axis='calendar'")), notes=(("minutes", "The segment's length."),)),
    "plot_sleep_regularity.data": _Table("one per participant", extends="sleep_regularity",
                                         columns=("extra_sleep_free_minutes", "group"), returned_by=("plot_sleep_regularity",)),
    "plot_sleep_timing.data": _Table(
        "one per participant (their median night) or per night", ("RegistrationCode", "midpoint_hours_after_noon",
        "asleep_minutes", "group"), returned_by=("plot_sleep_timing",), optional=(("night_date", "unit='night'"),)),
    "plot_sleep_architecture.data": _Table(
        "one per participant", ("RegistrationCode", "nights", "waso_minutes", "nap_share", "core_share", "deep_share",
                                "rem_share", "staged_nights", "group"), returned_by=("plot_sleep_architecture",),
        notes=(("nights", "Valid nights."), ("waso_minutes", "The participant's median wake after sleep onset."),
               ("core_share", "The median core share over staged nights."),
               ("deep_share", "The median deep share over staged nights."),
               ("rem_share", "The median REM share over staged nights."))),
    "plot_sleep_recording.data": _Table(
        "one per participant with nights", ("RegistrationCode", "sleep_recorded_nights", "in_bed_only_nights",
        "neither_nights", "sleep_recorded_share", "in_bed_only_share", "neither_share"),
        returned_by=("plot_sleep_recording",)),
    "plot_cgm_ranges.data": _Table(
        "one per CGM participant drawn", ("RegistrationCode", "valid_days", "readings", "very_low_percent",
        "low_percent", "in_range_percent", "high_percent", "very_high_percent"), returned_by=("plot_cgm_ranges",),
        notes=(("readings", "Readings on valid days."),)),
    "plot_agp.data": _Table(
        "one per time-of-day bin: the participant's percentiles, or the cohort's", returned_by=("plot_agp",),
        optional=(*((c, "with a participant") for c in ("RegistrationCode", "minute", "readings", "days", "p5", "p50",
                                                         "p95")),
                  *((c, "for the cohort") for c in ("group", "hour", "participants", "median", "suppressed")),
                  ("p25", "with a participant, or for the cohort"), ("p75", "with a participant, or for the cohort")),
        notes=(("hour", "The bin's centre, in hours after midnight."),
               ("median", "The median across participants of their median in the bin."),
               ("readings", "Readings in the bin over valid days."), ("days", "Valid days with readings in the bin."))),
    "plot_glucose_days.data": _Table("one per reading drawn", ("day_index", "minute_of_day", "mgdl", "valid_day"),
                                     returned_by=("plot_glucose_days",)),
    "plot_glycemic_cohort.data": _Table(
        "one per CGM participant drawn", ("RegistrationCode", "valid_days", "mean_mgdl", "gmi_percent", "cv_percent",
        "very_low_percent", "low_percent", "in_range_percent", "high_percent", "very_high_percent", "below_70_percent",
        "above_180_percent", "meets_in_range", "meets_below_70", "meets_below_54", "meets_above_180",
        "meets_above_250", "meets_cv", "group"), returned_by=("plot_glycemic_cohort",)),
    "plot_cgm_wear.data": _Table(
        "one per CGM participant and day", ("RegistrationCode", "study_day", "readings", "completeness", "valid"),
        returned_by=("plot_cgm_wear",), optional=(("local_date", "time_axis='calendar'"),),
        notes=(("readings", "Readings that day."),)),
    "plot_bmi_categories.data": _Table("one per group and class", extends="bmi_categories", columns=("suppressed",),
                                       returned_by=("plot_bmi_categories",)),
    "plot_step_categories.data": _Table("one per group and category", extends="step_categories", columns=("suppressed",),
                                        returned_by=("plot_step_categories",)),
    "plot_activity_goals.data": _Table("one per participant", extends="activity_goals.participants", columns=("group",),
                                       returned_by=("plot_activity_goals",)),
    "plot_blood_pressure.data": _Table("one per participant", extends="blood_pressure", columns=("group",),
                                       returned_by=("plot_blood_pressure",)),
    "plot_weight_trajectories.data": _Table(
        "one per participant and valid day", ("RegistrationCode", "study_day", "value", "percent_change"),
        returned_by=("plot_weight_trajectories",), optional=(("local_date", "time_axis='calendar'"),),
        notes=(("value", "The day's weight."),)),
    "plot_clinical_thresholds.data": _Table("one per participant, feature and threshold",
                                            extends="clinical_thresholds.participants",
                                            returned_by=("plot_clinical_thresholds",)),
    "plot_co_availability.data": _Table("one per ordered pair of features drawn", ("feature_a", "feature_b", "jaccard"),
                                        returned_by=("plot_co_availability",)),
    "plot_multimodal_days.data": _Table("one per participant with any of the features",
                                        ("RegistrationCode", "complete_days", "group"), returned_by=("plot_multimodal_days",)),
    "plot_participant_overview.data": _Table(
        "one per point drawn in the value panels", ("x", "value", "valid", "feature", "label"),
        returned_by=("plot_participant_overview",),
        notes=(("x", "The day's position: days since the first day with data, or the date."),
               ("value", "The day's headline value (total sleep per night, mean glucose per CGM day with domain "
                         "metrics)."))),
    "plot_lagged_association.data": _Table(
        "one per paired day", ("RegistrationCode", "x", "y", "x_dev", "y_dev"), returned_by=("plot_lagged_association",),
        notes=(("x", "The first series' value on day d."), ("y", "The second series' value on day d + lag_days."))),
    "plot_feature_correlations.data": _Table("one per ordered pair of features", ("feature_a", "feature_b", "r",
                                             "participants", "suppressed"), returned_by=("plot_feature_correlations",),
                                             notes=(("participants", "Participants with both features."),)),
    # -------------------------------------------------------------------------------------- figures: panels
    "plot_feature_combinations.panels.sets": _Table("one per chosen feature", ("feature", "participants"),
                                                    returned_by=("plot_feature_combinations",)),
    "plot_feature_combinations.panels.all_combinations": _Table(
        "one per combination, drawn or not", ("participants", "size", "combination", "suppressed"),
        returned_by=("plot_feature_combinations",), dynamic="One column per chosen feature: whether the combination "
                                                            "includes it."),
    "plot_valid_day_rule.panels.distribution": _Table(
        "one per participant-day the criterion governs", ("RegistrationCode", "local_date", "criterion", "valid"),
        returned_by=("plot_valid_day_rule",), notes=(("criterion", "The day's value of the rule's criterion."),)),
    "plot_valid_day_rule.panels.sensitivity": _Table(
        "one per threshold and measure", ("threshold", "measure", "k", "count", "share", "suppressed"),
        returned_by=("plot_valid_day_rule",),
        notes=(("measure", "Either participant-days kept, or participants with at least k valid days."),
               ("count", "The participant-days kept, or the participants with at least k valid days."),
               ("share", "count as a share of all participant-days, or of all participants."),
               ("k", "The k of 'participants with at least k valid days'; empty for days kept."))),
    "plot_hourly_coverage.panels.cohort": _Table(
        "one per local hour", ("hour", "day_share_median", "day_share_p25", "day_share_p75", "day_share_participants"),
        returned_by=("plot_hourly_coverage",)),
    "plot_hour_profiles.panels.order": _Table(
        "one per participant drawn", ("RegistrationCode",), returned_by=("plot_hour_profiles",),
        optional=(("trough_hour", "order_by='trough'"), ("peak_hour", "order_by='peak'"))),
    "plot_time_zones.panels.home_zones": _Table("one per home zone", ("home_zone", "participants", "suppressed"),
                                                returned_by=("plot_time_zones",)),
    "plot_time_zones.panels.days_away": _Table("one per number of days away", ("days_away", "participants", "suppressed"),
                                               returned_by=("plot_time_zones",),
                                               notes=(("days_away", "A number of days away."),)),
    "plot_time_zones.panels.trips": _Table("one per number of trips", ("trips", "participants", "suppressed"),
                                           returned_by=("plot_time_zones",), notes=(("trips", "A number of trips."),)),
    "plot_metric_distributions.panels.table1": _Table(
        "one per feature and group: Table 1", ("feature", "metric", "unit", "group", "participants", "median", "p25",
                                               "p75", "suppressed"), returned_by=("plot_metric_distributions",),
        notes=(("median", "The median of participants' medians."),)),
    "plot_agp.panels.metrics": _Table("the participant's consensus metrics", extends="DomainMetrics.cgm_periods",
                                      returned_by=("plot_agp",)),
    "plot_glucose_days.panels.median": _Table("one per {c:AGP_BIN_MINUTES}-minute bin", extends="DomainMetrics.cgm_profile",
                                              returned_by=("plot_glucose_days",)),
    "plot_glycemic_cohort.panels.targets": _Table(
        "one per group and consensus target", ("group", "target", "participants", "meeting", "share", "suppressed"),
        returned_by=("plot_glycemic_cohort",), notes=(("share", "The share of participants meeting the target."),)),
    "plot_activity_goals.panels.weeks": _Table("one per participant and week", extends="activity_goals.weeks",
                                               returned_by=("plot_activity_goals",)),
    "plot_blood_pressure.panels.categories": _Table(
        "one per group and category", ("group", "category", "participants", "share", "suppressed"),
        returned_by=("plot_blood_pressure",), notes=(("category", "The blood-pressure category."),)),
    "plot_weight_trajectories.panels.cohort": _Table(
        "one per 30-day period since the first measurement", ("period", "participants", "median"),
        returned_by=("plot_weight_trajectories",),
        notes=(("period", "The 30-day period since each participant's first valid measurement, from 0."),
               ("median", "The median change from the first measurement, in percent."))),
    "plot_clinical_thresholds.panels.cohort": _Table("one per feature and threshold", extends="clinical_thresholds.cohort",
                                                     columns=("suppressed",), returned_by=("plot_clinical_thresholds",)),
    "plot_multimodal_days.panels.curve": _Table(
        "one per group and required number of complete days", ("group", "k", "participants", "share", "suppressed"),
        returned_by=("plot_multimodal_days",),
        notes=(("k", "A required number of complete days."), ("participants", "Participants with at least k complete days."),
               ("share", "Those participants as a share of the group."))),
    "plot_participant_overview.panels.availability": _Table(
        "one per day with any data", ("x", "features"), returned_by=("plot_participant_overview",),
        notes=(("x", "The day's position: days since the first day with data, or the date."),
               ("features", "The features with data that day, separated by semicolons."))),
    "plot_lagged_association.panels.participants": _Table(
        "one per participant included", ("RegistrationCode", "days", "r"), returned_by=("plot_lagged_association",),
        notes=(("days", "Paired days."), ("r", "The participant's own correlation between the two series."))),
    "plot_lagged_association.panels.summary": _Table(
        "one row", ("x", "y", "lag_days", "participants", "paired_days", "within_slope", "within_r"),
        returned_by=("plot_lagged_association",), notes=(("x", "The first series' name."), ("y", "The second series' name."))),
}


# A meaning for every column name. A table can override one with its own notes, where a name means something more
# specific there; utils.info("<table>") shows the meaning that applies.
COLUMNS = {
    # who, what, when
    "RegistrationCode": "The participant's registration code.",
    "feature": "The feature (data type), such as StepCount or HeartRate.",
    "feature_a": "The first feature of the pair.",
    "feature_b": "The second feature of the pair.",
    "local_date": "The participant's local calendar day.",
    "night_date": "The night, named by the local date of the noon it starts at.",
    "hour": "The local hour of day, 0 to 23.",
    "weekday": "The day of the week.",
    "month": "The month.",
    "period": "The period the row summarizes.",
    "study_day": "Days since the participant's first day with data, from 0.",
    "study_month": "Months since the participant's first day with data, from 0.",
    "day_index": "The day's position among the days drawn, from 0.",
    "night_index": "The night's position among the nights drawn, from 0.",
    "week_start": "The Monday the week starts on.",
    "local_time": "The reading's local date and time.",
    "minute": "The start of the time-of-day bin, in minutes after local midnight.",
    "minute_of_day": "The reading's local time of day, in minutes after midnight.",
    "start_local": "The segment's local start time.",
    "end_local": "The segment's local end time.",
    "first_day": "The first local day with data.",
    "last_day": "The last local day with data.",
    "first_valid_day": "The first valid day.",
    "last_valid_day": "The last valid day.",
    "x": "The value on the horizontal axis.",
    "y": "The value on the vertical axis.",
    # what a row describes
    "metric": "The daily quantity summarized: a column of the daily table (such as value_sum), or a domain metric.",
    "unit": "The unit of the metric.",
    "value_unit": "The unit the day's values are in, after harmonization.",
    "rule": "The valid-day rule applied, as text.",
    "group": "The group from groups=, or 'all participants'.",
    "category": "The category the row counts.",
    "threshold": "The threshold: a valid-day threshold, or a clinical threshold named after its column.",
    "kind": "Whether the row counts a curation status or a curation flag.",
    "name": "The curation status or flag.",
    "status": "The record's curation status: pass, review or exclude_default.",
    "flag": "The curation flag.",
    "acquisition_method": "How the records were acquired, such as automatic device recording or manual entry.",
    "state": "The sleep state recorded, such as ASLEEP, AWAKE, INBED or a sleep stage.",
    "source": "The source (device or app) that recorded it.",
    "criterion": "The valid-day rule's criterion.",
    "measure": "What the row measures.",
    "label": "The label drawn.",
    "target": "The consensus target, as text.",
    "combination": "The features in the combination, as text.",
    "section": "The report section.",
    "figure": "The figure function.",
    "reason": "Why the page was not drawn; empty when it was.",
    "band": "The participant's data-volume band, such as 1-10 MB.",
    "day_basis": "How the row's day was assigned: local (from the participant's local time) or export_day_key (the "
                 "export's own day, for ActivitySummary).",
    "efficiency_basis": "What efficiency divides by: in_bed (recorded time in bed overlapping the sleep) or "
                        "sleep_period; empty without asleep time.",
    "cgm": "Whether the BloodGlucose readings have the regular cadence of a continuous glucose monitor.",
    "home": "Whether the day's UTC offset is the participant's home offset.",
    "home_zone": "The participant's home UTC offsets, as a label.",
    "features": "The features, as text or as a number.",
    "utc_offsets": "The day's UTC offsets in minutes, separated by semicolons.",
    "offsets": "The day's UTC offsets in minutes, separated by semicolons.",
    "home_offsets": "The UTC offsets counted as the participant's home, in minutes, separated by semicolons (such as "
                    "standard and daylight-saving time).",
    # counts
    "records": "The number of stored records the row counts.",
    "records_user_entered": "Records the participant entered by hand.",
    "records_without_offset": "Records without a UTC offset, placed on a day by their UTC time.",
    "records_included": "Records the curation includes by default.",
    "value_count": "The number of values recorded.",
    "values_unresolved": "Values left out of the value statistics because their unit could not be established.",
    "values_below_range": "Values below the feature's plausible range.",
    "values_above_range": "Values above the feature's plausible range.",
    "readings": "The number of readings the row counts.",
    "unresolved_readings": "Readings left out because their unit could not be established.",
    "days": "The number of days the row counts.",
    "days_with_data": "Days with data.",
    "valid_days": "Days meeting the valid-day rule.",
    "days_assessed": "Participant-days whose values could be checked against the plausible range.",
    "days_with_offsets": "Days with a known UTC offset.",
    "days_away": "Days spent away from the home UTC offset.",
    "days_beyond": "Days with at least one reading beyond the threshold.",
    "days_both": "Participant-days with both features.",
    "days_a": "Participant-days with the first feature.",
    "days_b": "Participant-days with the second feature.",
    "n_days": "Valid days behind the value.",
    "nights": "The number of nights the row counts.",
    "free_nights": "Valid nights ending on a weekend day (utils.info(\"free nights\") gives the current weekend).",
    "work_nights": "Valid nights ending on a workday.",
    "staged_nights": "Valid nights with sleep stages.",
    "participants": "The number of participants the row counts.",
    "participants_beyond": "Participants with at least one reading beyond the threshold.",
    "participant_days": "Participant-days.",
    "sessions": "Sessions recorded.",
    "episodes": "Separate sleep episodes in the night.",
    "trips": "Stays away from the home offset.",
    "gaps": "Runs of days without a valid day, between the first and last valid day.",
    "complete_weeks": "Weeks with all seven days recorded.",
    "complete_days": "Days on which every one of the features has a valid day.",
    "weeks_meeting_who": "Complete weeks with at least 150 minutes of exercise.",
    "move_goal_days": "Days with a move goal and a value.",
    "exercise_goal_days": "Days with an exercise goal and a value.",
    "stand_goal_days": "Days with a stand goal and a value.",
    "known": "Days of the week with the exercise value known.",
    "beyond": "Readings beyond the threshold.",
    "count": "The count the row reports.",
    "k": "A number of days.",
    "size": "The number of features in the combination.",
    "values": "The number of values the row counts.",
    "meeting": "Participants meeting the target.",
    "paired_days": "Days with both series paired at the lag.",
    "rows": "The number of rows stored.",
    "pass_rows": "Rows the curation passed.",
    "review_rows": "Rows the curation marked for review.",
    "exclude_default_rows": "Rows the curation excludes by default.",
    "included_by_default_rows": "Rows included by default.",
    "excluded_by_default_rows": "Rows excluded by default.",
    "canonical_value_rows": "Rows whose value is in the feature's canonical unit.",
    "ambiguous_unit_rows": "Rows whose unit is ambiguous.",
    "bytes_on_disk": "Bytes the files take on disk.",
    "bytes": "Bytes the participant's files take on disk.",
    "median_rows_per_participant": "The median participant's rows.",
    "max_rows_per_participant": "The largest participant's rows.",
    "distinct_offsets": "Distinct UTC offsets seen.",
    "sources": "Distinct sources (devices or apps) that recorded that day, where known.",
    "staging_sources": "Sources that recorded sleep stages that night.",
    "sleep_recorded_nights": "Nights with asleep time recorded.",
    "in_bed_only_nights": "Nights with time in bed but no asleep time.",
    "neither_nights": "Nights in the participant's span with neither asleep time nor time in bed.",
    "readings_below_90_percent": "Oxygen saturation readings below 90% that day.",
    "readings_at_least_38_celsius": "Body temperature readings of at least 38 °C that day.",
    "blood_pressure_systolic_value_count": "Systolic readings that day.",
    "blood_pressure_diastolic_value_count": "Diastolic readings that day.",
    "nights_in_bed_only": "Nights with time in bed but no asleep time.",
    # durations
    "hours_with_data": "Local hours of the day holding data: an event in the hour, or an interval overlapping it.",
    "median_gap_minutes": "The median gap between consecutive records that day, in minutes.",
    "max_gap_minutes": "The longest gap between consecutive records that day, in minutes.",
    "observed_minutes": "Minutes of the day covered by at least one record.",
    "redundant_minutes": "Minutes covered by records from two or more sources at once.",
    "typical_gap_minutes": "The most common gap between consecutive records, in minutes: the sampling cadence.",
    "span_days": "Days from the first to the last day, inclusive.",
    "longest_valid_run": "The longest run of consecutive valid days.",
    "longest_gap_days": "The longest run of days without a valid day.",
    "median_gap_days": "The median run of days without a valid day.",
    "days_since_first_valid": "Days since the participant's first valid day.",
    "longest_trip_days": "The longest stay away from the home offset, in days.",
    "max_offset_difference_hours": "The largest difference from the home offset, in hours.",
    "offset_hours": "The day's UTC offset, in hours.",
    "minutes": "The row's duration, in minutes.",
    "minutes_asleep": "Minutes recorded as asleep without a stage (HealthKit's asleep state).",
    "minutes_asleep_total": "Minutes asleep in any asleep state: unspecified, core, deep or REM.",
    "minutes_awake": "Minutes recorded as awake.",
    "minutes_core": "Minutes of core (light) sleep.",
    "minutes_deep": "Minutes of deep sleep.",
    "minutes_inbed": "Minutes recorded as in bed.",
    "minutes_other_states": "Minutes in states other than these.",
    "minutes_rem": "Minutes of REM sleep.",
    "sleep_period_minutes": "The main sleep period, from sleep onset to final waking, in minutes.",
    "asleep_minutes": "Minutes asleep within the main sleep period.",
    "waso_minutes": "Wake after sleep onset: the sleep period minus the minutes asleep.",
    "awake_recorded_minutes": "Minutes recorded as awake within the sleep period.",
    "in_bed_minutes": "Minutes recorded as in bed.",
    "core_minutes": "Minutes of core sleep in the sleep period.",
    "deep_minutes": "Minutes of deep sleep in the sleep period.",
    "rem_minutes": "Minutes of REM sleep in the sleep period.",
    "asleep_unspecified_minutes": "Minutes asleep without a stage in the sleep period.",
    "staged_minutes": "Minutes asleep with a stage in the sleep period.",
    "nap_minutes": "Minutes asleep outside the main sleep period.",
    "asleep_sd_minutes": "The standard deviation of minutes asleep across valid nights.",
    "asleep_free_minutes": "The mean minutes asleep on free nights.",
    "asleep_work_minutes": "The mean minutes asleep on work nights.",
    "extra_sleep_free_minutes": "Mean minutes asleep on free nights minus work nights.",
    "weekly_exercise_minutes": "The median exercise minutes per complete week.",
    "exercise_minutes": "Exercise minutes that week.",
    "lag_days": "The lag: the second series is taken this many days after the first.",
    # clock times
    "onset_local": "Sleep onset: the local start of the main sleep period.",
    "offset_local": "Final waking: the local end of the main sleep period.",
    "midpoint_local": "The local midpoint of the main sleep period.",
    "onset_hours_after_noon": "Sleep onset, in hours after the night's noon.",
    "offset_hours_after_noon": "Final waking, in hours after the night's noon.",
    "midpoint_hours_after_noon": "The sleep midpoint, in hours after the night's noon.",
    "start_hours_after_noon": "The segment's start, in hours after the night's noon.",
    "end_hours_after_noon": "The segment's end, in hours after the night's noon.",
    "midpoint_sd_hours": "The standard deviation of the sleep midpoint across valid nights, in hours.",
    "onset_sd_hours": "The standard deviation of sleep onset across valid nights, in hours.",
    "offset_sd_hours": "The standard deviation of final waking across valid nights, in hours.",
    "midpoint_free": "The mean sleep midpoint on free nights, in hours after noon.",
    "midpoint_work": "The mean sleep midpoint on work nights, in hours after noon.",
    "social_jetlag_hours": "The free-night minus the work-night sleep midpoint, in hours.",
    "trough_hour": "The hour of the participant's lowest value.",
    "peak_hour": "The hour of the participant's highest value.",
    # values and statistics
    "value": "The value drawn.",
    "value_sum": "The total amount recorded, for totals and event amounts.",
    "value_mean": "The mean value, for levels.",
    "value_median": "The median value that day.",
    "value_min": "The lowest value that day.",
    "value_max": "The highest value that day.",
    "value_per_day": "The amount per day recorded in the hour, over the participant's days.",
    "records_per_day": "Records per day starting in the hour, over the participant's days.",
    "value_sum_per_participant": "value_sum divided by the participants contributing that day.",
    "mean": "The mean.",
    "sd": "The standard deviation.",
    "median": "The median.",
    "p5": "The 5th percentile.",
    "p10": "The 10th percentile.",
    "p25": "The 25th percentile.",
    "p50": "The 50th percentile (the median).",
    "p75": "The 75th percentile.",
    "p90": "The 90th percentile.",
    "p95": "The 95th percentile.",
    "min": "The lowest value.",
    "max": "The highest value.",
    "ci_low": "The lower bound of the bootstrap confidence interval of the median.",
    "ci_high": "The upper bound of the bootstrap confidence interval of the median.",
    "rolling_mean": "The centred rolling mean over the chosen number of days.",
    "relative": "The value relative to the participant's own mean over the 24 hours (1 is their average).",
    "r": "The Pearson correlation.",
    "within_slope": "The pooled within-participant slope of y on x.",
    "within_r": "The pooled within-participant correlation.",
    "x_dev": "x minus the participant's own mean.",
    "y_dev": "y minus the participant's own mean.",
    "jaccard": "Days with both, as a share of days with either (Jaccard index).",
    "density": "Days with data as a share of the span.",
    "percent_change": "The change from the participant's first valid measurement, in percent.",
    "systolic": "The participant's median systolic pressure over valid days, in mmHg.",
    "diastolic": "The participant's median diastolic pressure over valid days, in mmHg.",
    "blood_pressure_systolic_value_mean": "The day's mean systolic pressure, in mmHg.",
    "blood_pressure_diastolic_value_mean": "The day's mean diastolic pressure, in mmHg.",
    "active_energy_burned_mean": "The day's active energy (Move ring), in kcal.",
    "active_energy_burned_goal_mean": "The day's Move goal, in kcal.",
    "apple_exercise_time_mean": "The day's exercise minutes (Exercise ring).",
    "apple_exercise_time_goal_mean": "The day's Exercise goal, in minutes.",
    "apple_stand_hours_mean": "The day's stand hours (Stand ring).",
    "apple_stand_hours_goal_mean": "The day's Stand goal, in hours.",
    "average_heart_rate_mean": "The mean heart rate the day's recordings report, in beats per minute.",
    "sampling_frequency_mean": "The mean sampling frequency of the day's recordings, in Hz.",
    "lower": "The category's lower bound (inclusive).",
    "upper": "The category's upper bound (exclusive).",
    "weekday_median": "The participant's median valid weekday.",
    "weekend_median": "The participant's median valid weekend day.",
    "weekend_minus_weekday": "weekend_median minus weekday_median.",
    "weekday_days": "Valid weekdays.",
    "weekend_days": "Valid weekend days.",
    # shares, between 0 and 1 unless the name says percent
    "share": "A share, between 0 and 1.",
    "fraction": "The participants as a share of those who started.",
    "day_share": "The share of the participant's days on which the hour holds data.",
    "regular_share": "The share of gaps within 10% of the typical gap: how regular the cadence is.",
    "completeness": "The day's completeness: the valid-day criterion's value relative to its expected level.",
    "adherence": "Valid days as a share of the span.",
    "median_completeness": "The median day's completeness.",
    "share_of_records": "The status or flag's share of the feature's records.",
    "share_beyond": "Readings beyond the threshold, as a share of readings.",
    "redundant_share": "Redundant minutes as a share of observed minutes.",
    "without_offset_share": "Records without a UTC offset, as a share of records.",
    "user_entered_share": "Records entered by hand, as a share of records.",
    "below_per_1000": "Values below the plausible range per 1,000 values assessed.",
    "above_per_1000": "Values above the plausible range per 1,000 values assessed.",
    "acquisition_classified_fraction": "The share of rows whose acquisition method is classified.",
    "expected_records_per_day": "The records a complete day would hold, from the typical gap.",
    "expected_readings_per_day": "The readings a complete day would hold, from the typical gap.",
    "efficiency": "Minutes asleep as a share of the efficiency basis.",
    "core_share": "Core sleep as a share of staged sleep.",
    "deep_share": "Deep sleep as a share of staged sleep.",
    "rem_share": "REM sleep as a share of staged sleep.",
    "nap_share": "The share of valid nights with a nap.",
    "active_percent": "Readings on valid days as a percent of the readings expected on them.",
    "span_active_percent": "Readings over the whole span as a percent of the readings expected over it.",
    "sleep_recorded_share": "Nights with asleep time recorded, as a share of the participant's span.",
    "in_bed_only_share": "Nights with time in bed only, as a share of the participant's span.",
    "neither_share": "Nights with neither, as a share of the participant's span.",
    # true or false
    "present": "Whether the participant has a file for the feature.",
    "valid": "Whether the day meets the valid-day rule.",
    "valid_day": "Whether the reading's day meets the valid CGM day rule.",
    "suppressed": "Whether the row's participant count is below min_participants, so its values are hidden.",
    "counted": "Whether the segment counts toward the night's metrics.",
    "main_period": "Whether the segment lies within the main sleep period.",
    "complete": "Whether all seven days of the week are recorded.",
    "sufficient": "Whether the participant has at least the consensus minimum of valid CGM days.",
    "staged": "Whether the night has sleep stages.",
    "asleep_recorded": "Whether the night has asleep time recorded.",
    "in_bed_recorded": "Whether the night has time in bed recorded.",
    "verifiable": "Whether the file carries the state needed to verify it against the manifest.",
    "move_goal_met": "The share of days meeting the Move goal.",
    "exercise_goal_met": "The share of days meeting the Exercise goal.",
    "stand_goal_met": "The share of days meeting the Stand goal.",
    "meets_in_range": "Whether time in range (70–180 mg/dL) is above 70%.",
    "meets_below_70": "Whether time below 70 mg/dL is under 4%.",
    "meets_below_54": "Whether time below 54 mg/dL is under 1%.",
    "meets_above_180": "Whether time above 180 mg/dL is under 25%.",
    "meets_above_250": "Whether time above 250 mg/dL is under 5%.",
    "meets_cv": "Whether glucose variability (CV) is at most 36%.",
    # glucose
    "mgdl": "The glucose reading, in mg/dL.",
    "mean_mgdl": "Mean glucose, in mg/dL.",
    "mean_mmol": "Mean glucose, in mmol/L.",
    "sd_mgdl": "The standard deviation of glucose, in mg/dL.",
    "cv_percent": "Glucose variability: the coefficient of variation, in percent.",
    "gmi_percent": "The glucose management indicator, in percent (an estimate of HbA1c).",
    "very_low_percent": "Time below 54 mg/dL, in percent.",
    "low_percent": "Time from 54 to below 70 mg/dL, in percent.",
    "in_range_percent": "Time from 70 to 180 mg/dL, in percent.",
    "high_percent": "Time above 180 up to 250 mg/dL, in percent.",
    "very_high_percent": "Time above 250 mg/dL, in percent.",
    "below_70_percent": "Time below 70 mg/dL, in percent (very low plus low).",
    "above_180_percent": "Time above 180 mg/dL, in percent (high plus very high).",
}

# Families of columns named by a pattern: the cohort statistics of hour_of_day's per-participant measures.
_MEASURES = {"records_per_day": "records per day", "value_per_day": "amount per day", "value_mean": "mean value",
             "day_share": "share of days with data"}
_STATS = {"median": "The median across participants of their {m} in the hour.",
          "p25": "The 25th percentile across participants of their {m} in the hour.",
          "p75": "The 75th percentile across participants of their {m} in the hour.",
          "participants": "Participants with a {m} in the hour."}
_HOUR_STATISTIC = re.compile(rf"^({'|'.join(_MEASURES)})_({'|'.join(_STATS)})$")


def _pattern_meaning(column: str) -> str:
    match = _HOUR_STATISTIC.match(column)
    return _STATS[match.group(2)].format(m=_MEASURES[match.group(1)]) if match else ""


# ---------------------------------------------------------------------------------------------------- concepts
@dataclass(frozen=True)
class _Concept:
    """An idea the tools rest on. ``{d:key}`` in the text is the current value of the default ``key``."""

    title: str
    group: str
    summary: str
    body: tuple[str, ...]
    tools: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    defaults: tuple[str, ...] = ()
    related: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.body, str):
            object.__setattr__(self, "body", (self.body,))


CONCEPTS: dict[str, _Concept] = {
    # ---------------------------------------------------------------------------------------- runs and inputs
    "phases": _Concept(
        "Phases: curated and native", "runs",
        "Every tool reads one of two copies of the data: the native export, or the curated one.",
        ("The native phase is the export as HealthKit wrote it, cleaned into one file per participant and feature. The "
         "curated phase is the same data after curation: each record has a status (pass, review or exclude_default), "
         "flags, and a default-inclusion decision, and units are resolved where they can be. The phases are "
         "{d:phases}, and each has a default root ({d:data_roots}).",
         "Curation adds information; it removes nothing. Curation columns (such as records_included) and tables "
         "(DailyStatistics.curation) exist only in the curated phase, and figures of curation refuse the native one. "
         "default_inclusion_only=True restricts a curated run to the records the curation includes by default."),
        tools=("compute_coverage", "compute_daily_statistics", "compute_domain_metrics", "plot_curation"),
        parameters=("phase", "root", "default_inclusion_only"), columns=("records_included",),
        defaults=("phases", "data_roots"), related=("curation", "data-roots"), aliases=("curated phase", "native phase", "curated and native")),
    "local-days": _Concept(
        "Local days", "runs",
        "A day is the participant's local calendar day, not the UTC day.",
        ("An event's local time is its UTC time plus the record's UTC offset, so a run at 23:30 in Tel Aviv belongs to "
         "that evening, not to the next UTC day. A record without an offset falls back to its UTC day and is counted "
         "in records_without_offset, so you can see how often that happens. ActivitySummary has no event times: its "
         "day is the export's own day key, which day_basis records.",
         "Intervals are split at local midnight, and each piece belongs to its day: a night's sleep contributes to "
         "two local days, which is why the sleep tools use nights instead."),
        tools=("compute_daily_statistics", "time_zones"), columns=("local_date", "day_basis", "records_without_offset"),
        related=("noon-to-noon-night", "time-zones"), aliases=("local day", "local time")),
    "measurement-kinds": _Concept(
        "Measurement kinds", "runs",
        "How a feature's values combine into a day depends on what they measure.",
        ("Each feature has a measurement kind, declared by the curation registry. Totals (extensive_total, such as "
         "steps) are summed, each interval contributing to a day in proportion to its overlap with it, so daily sums "
         "add up to the stored total. Event amounts (such as a meal's carbohydrates) are summed on the event's day. "
         "Levels, rates, proportions and device summaries (heart rate, oxygen saturation, weight) are described by "
         "mean, median, minimum and maximum, and never summed. Sleep and mindful sessions are counted in minutes.",
         "The kind decides the daily columns (utils.info.daily_columns lists them) and each feature's headline metric: "
         "value_sum for totals, value_mean for levels."),
        tools=("compute_daily_statistics", "summarize"), columns=("value_sum", "value_mean", "value_median"),
        related=("time-counted-once", "units"), aliases=("measurement kind",)),
    "time-counted-once": _Concept(
        "Time counted once", "quality",
        "Overlapping records, such as a watch and a phone recording the same walk, are never counted twice.",
        ("Observed minutes, sleep-state minutes and mindful minutes are the length of the union of intervals, so "
         "overlapping records from several devices count once. What the overlap was is kept, too: redundant_minutes "
         "counts the minutes covered by two or more sources at once, and QualityReport.features gives each feature's "
         "redundant share.",
         "Totals such as steps are allocated by overlap with each day, not deduplicated: when two devices record the "
         "same steps, both are summed. The redundancy columns show where that can matter."),
        tools=("compute_daily_statistics", "quality_report", "plot_quality"),
        columns=("observed_minutes", "redundant_minutes", "redundant_share"),
        related=("measurement-kinds",), aliases=("redundancy", "duplicate devices")),
    "units": _Concept(
        "Units", "runs",
        "Values are harmonized to one unit per feature before any statistic.",
        ("Where a unit can be harmonized, it is, so a feature's values share one unit (value_unit). A value whose unit "
         "could not be established is left out of the value statistics and counted in values_unresolved, never "
         "guessed. harmonize=False computes from stored values instead.",
         "Glucose is classified in mg/dL, the unit of the consensus ranges, converting stored mmol/L with HealthKit's "
         "factor ({d:mgdl_per_mmol}). Rounded mmol/L cutoffs such as 3.9 would misclassify the many readings exactly "
         "at 70 mg/dL."),
        tools=("compute_daily_statistics", "glucose_mgdl"), parameters=("harmonize",),
        columns=("value_unit", "values_unresolved"), defaults=("mgdl_per_mmol",),
        related=("measurement-kinds",), aliases=("harmonization", "harmonized units")),
    "written-runs": _Concept(
        "Written runs", "output",
        "At cohort scale, compute once into a directory, then read it back one participant at a time.",
        ("With out=, compute_daily_statistics and compute_domain_metrics write each participant's rows as it finishes, "
         "so memory stays bounded; resume=True continues an interrupted run where it stopped. Every tool that takes a "
         "run accepts its directory, and read_daily, iter_daily, read_table and read_domain_table read it back. "
         "run.json records versions, registry fingerprints and the state database's checksum.",
         "Each run records its output schema ({d:output_schemas}). A run written with another schema cannot be "
         "resumed: compute it again. Three statistics tables are written under other names than their fields; each "
         "table's card shows how to read it back."),
        tools=("compute_daily_statistics", "compute_domain_metrics", "read_daily", "iter_daily", "read_table",
               "read_domain_table", "export_parquet"),
        parameters=("out", "resume", "workers"), defaults=("output_schemas",),
        related=("data-roots",), aliases=("written run", "output schema", "run directory")),
    "data-roots": _Concept(
        "Data roots are read-only", "output",
        "Nothing is ever written inside a data root.",
        ("Every writing tool checks its destination first (data_statistics.guard_output), and refuses a path inside "
         "any data root it knows of, such as {d:data_roots}. This holds for runs, figures (path=) and reports."),
        tools=("guard_output", "protected_roots", "report"), parameters=("out", "path"), defaults=("data_roots",),
        related=("written-runs",), aliases=("data root", "read only roots")),
    # ------------------------------------------------------------------------------------------------ coverage
    "coverage-tiers": _Concept(
        "Two tiers: coverage and daily statistics", "coverage",
        "Who has which data is cheap to answer; what the data say is not.",
        ("compute_coverage answers who has which feature, how many rows and bytes, and in what curation state, from "
         "the state databases and one file stat per file, without parsing a single CSV: it covers the whole cohort in "
         "seconds. compute_daily_statistics reads the records, one participant at a time, and describes every "
         "participant-day. Start with coverage to decide what to compute."),
        tools=("compute_coverage", "compute_daily_statistics", "plot_participants_per_feature", "plot_feature_presence"),
        related=("written-runs",), aliases=("tiers", "coverage tiers")),
    "curation": _Concept(
        "Curation status and default inclusion", "quality",
        "In the curated phase each record carries a verdict, and a default decision to include it or not.",
        ("A record's status is pass, review or exclude_default, and it may carry flags naming what the curation "
         "noticed. Its default inclusion follows from both. Statistics count every record unless "
         "default_inclusion_only=True; records_included counts those included by default, so the effect of the "
         "choice is visible before making it. Figures of curation show the statuses and flags per feature."),
        tools=("compute_coverage", "quality_report", "plot_curation", "plot_curation_flags"),
        parameters=("default_inclusion_only",), columns=("records_included", "status", "flag", "pass_rows"),
        related=("phases",), aliases=("curation status", "default inclusion", "curation flags")),
    "provenance": _Concept(
        "Provenance", "quality",
        "How each record was acquired, and whether a person typed it in.",
        ("Records are counted by acquisition method (such as automatic device recording or manual entry), and those "
         "the participant entered by hand are counted separately (records_user_entered). A feature that is mostly "
         "hand-entered, such as weight, reads differently from one a sensor records all day."),
        tools=("quality_report", "provenance_tables", "plot_acquisition", "plot_acquisition_over_time"),
        columns=("acquisition_method", "records_user_entered", "user_entered_share"),
        aliases=("acquisition", "user entered records")),
    "plausible-ranges": _Concept(
        "Plausible ranges", "quality",
        "Values outside a feature's plausible range are counted, not removed.",
        ("{d:plausible_ranges} have a declared plausible range. Values below or above it are counted per day "
         "(values_below_range, values_above_range) and summarized per feature, so implausible values are visible "
         "without anything being silently dropped. Deciding what to exclude stays with the analysis."),
        tools=("quality_report", "plot_quality"), columns=("values_below_range", "values_above_range", "days_assessed"),
        defaults=("plausible_ranges",), aliases=("plausible range", "implausible values")),
    "sampling-cadence": _Concept(
        "Sampling cadence", "quality",
        "How often a participant's device records a feature, and how regularly.",
        ("typical_gap_minutes is the most common gap between consecutive records, and regular_share the share of "
         "gaps within 10% of it. A continuous monitor shows a short, very regular gap; hand-entered data shows long, "
         "irregular ones. A feature declared with a fixed cadence ({d:fixed_cadence}) gets expected records per day, "
         "hence completeness, only for participants whose data show that cadence."),
        tools=("summarize", "plot_sampling_cadence"), columns=("typical_gap_minutes", "regular_share",
                                                                 "expected_records_per_day"),
        defaults=("fixed_cadence",), related=("day-completeness",), aliases=("fixed cadence",)),
    # ------------------------------------------------------------------------------------ adherence and retention
    "valid-days": _Concept(
        "Valid days", "adherence",
        "Only days that meet their feature's valid-day rule enter summaries, patterns and most figures.",
        ("A day with a single stray record says little about that day. Each feature has a ValidDayRule: a minimum "
         "number of records, hours with data, or completeness. The defaults are conventions, overridable per "
         "feature: {d:valid_day_rules}.",
         "Pass rules={feature: sm.ValidDayRule(...)} to summarize, temporal_patterns and the figures that take "
         "rules. plot_valid_day_rule shows, before committing, how many participant-days and participants a "
         "threshold keeps."),
        tools=("summarize", "temporal_patterns", "mark_valid_days", "plot_valid_day_rule", "ValidDayRule"),
        parameters=("rules", "rule"), columns=("valid", "valid_days", "hours_with_data"),
        defaults=("valid_day_rules",), related=("day-completeness", "adherence-and-gaps", "valid-nights"),
        aliases=("valid day", "valid-day rule", "valid day rule", "wear time")),
    "day-completeness": _Concept(
        "Day completeness", "adherence",
        "A day's records over those expected, for sensors that sample on a fixed period.",
        ("Completeness exists only for features whose sensor samples on a fixed period when worn continuously, and "
         "only for participants whose data shows that cadence ({d:fixed_cadence}). Occasional finger-stick glucose "
         "readings are therefore never judged against a monitor's expected readings. For CGM, a day with at least "
         "{d:cgm_day_completeness} of its expected readings is valid, the consensus criterion."),
        tools=("summarize", "mark_valid_days", "cgm_metrics"),
        columns=("completeness", "median_completeness", "expected_records_per_day"),
        defaults=("fixed_cadence", "cgm_day_completeness"), related=("sampling-cadence", "valid-days"),
        aliases=("expected records", "day completeness")),
    "adherence-and-gaps": _Concept(
        "Adherence and gaps", "adherence",
        "How much of a participant's follow-up actually produced valid days.",
        ("Per participant and feature: the follow-up span from the first to the last day with data, the valid days, "
         "adherence (valid days over the span), the longest run of consecutive valid days, and the gaps between valid "
         "days. A participant with 30 valid days over 30 days and one with 30 over a year differ in ways the count of "
         "valid days alone would hide."),
        tools=("summarize", "plot_adherence", "plot_follow_up"),
        columns=("adherence", "span_days", "longest_valid_run", "gaps", "longest_gap_days"),
        related=("valid-days", "retention-curve"), aliases=("adherence and gaps", "follow up")),
    "retention-curve": _Concept(
        "Retention curves", "adherence",
        "How many participants still contribute valid days, day by day after their own start.",
        ("Retention counts, for each day k after each participant's first valid day, the participants who still have "
         "a valid day at k or later, as a share of those who started. It aligns everyone on their own start, so "
         "enrolment dates do not blur how long people keep wearing a device."),
        tools=("retention", "summarize", "plot_retention"), columns=("days_since_first_valid", "fraction"),
        related=("adherence-and-gaps", "study-time"), aliases=("retention curve",)),
    "study-time": _Concept(
        "Study time and calendar time", "adherence",
        "Figures can place days by date, or by days since each participant's first day.",
        ("time_axis=\"study\" counts days since the participant's first day with data (study_day), which aligns "
         "participants and keeps real dates out of the figure; time_axis=\"calendar\" uses dates, which shows "
         "seasons and enrolment waves. Figures of one participant default to study time, so that real dates appear "
         "only when asked for."),
        parameters=("time_axis",), columns=("study_day", "study_month", "local_date"),
        tools=("plot_daily_values", "plot_availability_raster", "plot_participant_overview"),
        aliases=("study time", "calendar time")),
    # ---------------------------------------------------------------------------------------------- patterns
    "weekday-weekend": _Concept(
        "Weekdays and weekends", "patterns",
        "The weekend is {d:weekend_days}, as in Israel.",
        ("temporal_patterns compares each participant's median valid weekday with their median valid weekend day, and "
         "the cohort figures show the spread of those differences. For a cohort elsewhere, pass weekend_days to "
         "temporal_patterns (Monday is 0).",
         "Free and work nights in the sleep tools follow the same weekend: the nights before a weekend day are free."),
        tools=("temporal_patterns", "plot_weekday_weekend", "plot_weekly_pattern", "sleep_regularity"),
        parameters=("weekend_days",), columns=("weekday_median", "weekend_median", "weekend_minus_weekday"),
        defaults=("weekend_days",), related=("free-nights",), aliases=("weekend", "weekday and weekend")),
    "hour-of-day": _Concept(
        "Hour of day", "patterns",
        "When in the day a feature is recorded, and what it measures then.",
        ("hour_of_day gives, per participant and local hour: records per day, the amount per day for totals, the mean "
         "value for levels, and day_share, the share of the participant's days on which the hour holds data. Hours "
         "without data are true zeros in the hourly table, not missing rows. The figures draw one of three measures: "
         "value, records or coverage.",
         "Relative profiles divide each participant's hours by their own daily mean, so that rhythms can be compared "
         "between people whose levels differ."),
        tools=("hour_of_day", "plot_hour_of_day", "plot_hourly_coverage", "plot_hour_profiles", "plot_daily_rhythms"),
        parameters=("measure", "relative"), columns=("day_share", "records_per_day", "value_per_day", "relative"),
        related=("local-days",), aliases=("daily rhythm", "time of day")),
    "time-zones": _Concept(
        "Time zones and travel", "patterns",
        "Each participant's home zone, and the days they spent away from it.",
        ("The home zone is the participant's most common UTC offset, together with the offset 60 minutes from it "
         "when that covers at least {d:home_zone_share} of the days: the zone's daylight-saving time. A day with any other offset is away; consecutive away days form a trip. "
         "A day on which the clocks change carries both home offsets, and is not away. Travel shifts local days, so "
         "long trips are worth knowing before reading daily patterns."),
        tools=("time_zones", "plot_time_zones"), columns=("home_offsets", "days_away", "trips", "longest_trip_days"),
        defaults=("home_zone_share",), related=("local-days",), aliases=("time zone", "travel", "home zone")),
    # ------------------------------------------------------------------------------------------------ values
    "participant-medians": _Concept(
        "Participants counted once", "values",
        "A cohort distribution gives each participant one value: their median over valid days.",
        ("A participant with 300 valid days would otherwise weigh 300 times more than one with a single day. Cohort "
         "tables and most cohort figures therefore summarize each participant first (their median valid day) and "
         "then describe those medians. unit=\"participant_day\" pools days instead, where that is the question.",
         "Summaries.cohort is the resulting Table 1: per feature and metric, the distribution of participants' "
         "medians ({d:summary_statistics} per participant)."),
        tools=("summarize", "cohort_table", "plot_metric_distributions", "plot_caterpillar"),
        parameters=("unit",), columns=("median", "p25", "p75", "n_days"), defaults=("summary_statistics",),
        related=("valid-days", "small-cells"), aliases=("participant median", "counting participants once")),
    "clinical-thresholds": _Concept(
        "Clinical thresholds", "clinical",
        "Readings beyond a clinical threshold are counted per participant and day.",
        ("The thresholds are {d:clinical_thresholds}. They are counts, not diagnoses: a consumer device's single "
         "reading below a threshold is a prompt to look, and the tables show how many readings and days are involved. "
         "Participants whose values are not in the threshold's unit are left out, since their counts are unknown.",
         "Population references are also drawn from established classifications: WHO adult BMI classes, daily-step "
         "categories, blood-pressure guidelines and the WHO activity recommendation, each shown with its source."),
        tools=("clinical_thresholds", "plot_clinical_thresholds", "plot_bmi_categories", "plot_step_categories",
               "plot_blood_pressure", "plot_activity_goals"),
        columns=("share_beyond", "days_beyond", "readings_below_90_percent"),
        defaults=("clinical_thresholds", "bmi_classes", "step_categories", "bp_guidelines", "who_weekly_minutes"),
        aliases=("clinical threshold",)),
    # ------------------------------------------------------------------------------------------------- sleep
    "noon-to-noon-night": _Concept(
        "The noon-to-noon night", "sleep",
        "A night is the window from noon to the next noon, named by its first date.",
        ("Sleep crosses midnight, so local days would split it in two. A night starts at {d:night_start_hour}:00 local "
         "time on night_date, and a record crossing noon is split at it. Clock times are also given in hours after "
         "the night's noon, so onset, midpoint and final waking can be averaged across nights without wrapping at "
         "midnight."),
        tools=("sleep_nights", "compute_domain_metrics", "plot_sleep_raster"),
        columns=("night_date", "onset_hours_after_noon", "midpoint_hours_after_noon"), defaults=("night_start_hour",),
        related=("sleep-episodes", "local-days"), aliases=("night", "noon to noon", "hours after noon")),
    "sleep-episodes": _Concept(
        "Sleep episodes, main sleep and naps", "sleep",
        "Asleep records close together form an episode; the longest episode is the night's main sleep.",
        ("Asleep records separated by at most {d:episode_gap} minutes form one sleep episode. The episode with the most "
         "asleep time is the main sleep; the others are naps. On the samples, gaps between asleep records are at most "
         "120 minutes within a night and over 360 minutes between nights. From the main sleep come onset, final "
         "waking, the sleep period between them and total sleep time; nap_minutes holds the rest."),
        tools=("sleep_nights", "plot_sleep_architecture"),
        columns=("episodes", "sleep_period_minutes", "asleep_minutes", "nap_minutes", "main_period"),
        defaults=("episode_gap", "sleep_states"), related=("noon-to-noon-night", "sleep-staging"),
        aliases=("sleep episode", "main sleep", "naps")),
    "sleep-staging": _Concept(
        "Sleep stages", "sleep",
        "Stage minutes and shares come from one device per night.",
        ("Two devices can stage the same minutes differently. Stages are therefore taken from one source per night, "
         "the device that staged the most sleep, and the shares are fractions of that source's staged time. The "
         "states are {d:sleep_states}."),
        tools=("sleep_nights", "plot_sleep_architecture", "plot_sleep_raster"),
        columns=("core_share", "deep_share", "rem_share", "staged", "staging_sources"), defaults=("sleep_states",),
        related=("sleep-episodes",), aliases=("sleep stages", "staging")),
    "sleep-efficiency": _Concept(
        "Efficiency and wake after sleep onset", "sleep",
        "Sleep time over time in bed, or over the sleep period when no time in bed was recorded.",
        ("Wake after sleep onset (WASO) is the sleep period minus total sleep time. Efficiency is total sleep time over "
         "time in bed, where an in-bed record overlaps the sleep; otherwise over the sleep period, which reads higher. "
         "efficiency_basis says which, so the two are never compared unawares."),
        tools=("sleep_nights", "plot_sleep"), columns=("efficiency", "efficiency_basis", "waso_minutes", "in_bed_minutes"),
        related=("in-bed-only",), aliases=("sleep efficiency", "waso", "wake after sleep onset")),
    "in-bed-only": _Concept(
        "Nights with time in bed only", "sleep",
        "A night holding only in-bed records has no measured sleep, which is not the same as no sleep.",
        ("Some sources record time in bed but no sleep. Such a night keeps its in-bed time but gets no sleep metrics, "
         "so it cannot pass for a short night. On the curated sample, about a third of nights are like this, which is "
         "why plot_sleep_recording shows each participant's mix of nights."),
        tools=("sleep_nights", "plot_sleep_recording", "summarize_domain"),
        columns=("in_bed_recorded", "asleep_recorded", "nights_in_bed_only", "in_bed_only_nights"),
        related=("valid-nights",), aliases=("in bed only", "time in bed only")),
    "valid-nights": _Concept(
        "Valid nights", "sleep",
        "A night counts when its sleep was measured and the main sleep lasted long enough.",
        ("A night is valid when it has asleep records and its main sleep lasted at least {d:night_rule} minutes "
         "(NightRule). Short fragments and in-bed-only nights are kept in the tables but left out of sleep summaries "
         "and figures. summarize_domain takes night_rule=, and sleep_regularity and the sleep figures take rule=."),
        tools=("NightRule", "summarize_domain", "sleep_regularity", "plot_sleep"), parameters=("night_rule",),
        defaults=("night_rule",), related=("in-bed-only", "valid-days"), aliases=("valid night", "night rule")),
    "free-nights": _Concept(
        "Free nights and social jetlag", "sleep",
        "Sleep before a free day often differs from sleep before a workday.",
        ("A night is free when the day it ends on is a weekend day ({d:weekend_days}), and a work night otherwise. Social jetlag is "
         "the free-night sleep midpoint minus the work-night one, in hours: how far the body clock drifts when the "
         "alarm is off. sleep_regularity also gives the night-to-night variability of onset, offset and midpoint."),
        tools=("sleep_regularity", "plot_sleep_regularity"),
        columns=("social_jetlag_hours", "midpoint_free", "midpoint_work", "free_nights", "work_nights"),
        defaults=("weekend_days",), related=("weekday-weekend",), aliases=("social jetlag", "free night", "sleep regularity")),
    # ------------------------------------------------------------------------------------------------ glucose
    "cgm-validity": _Concept(
        "CGM data: who, which days, and enough", "glucose",
        "Consensus CGM metrics need a continuous monitor, valid days, and enough of them.",
        ("Only participants whose glucose data shows a continuous monitor's fixed cadence get CGM metrics, so "
         "finger-stick readings are never mixed in (cgm). A CGM day is valid with at least {d:cgm_day_completeness} of "
         "its expected readings, and a participant's metrics are sufficient with at least {d:cgm_sufficient_days} "
         "valid days, the consensus requirement. active_percent reports wear over valid days; span_active_percent "
         "over the whole span."),
        tools=("cgm_metrics", "compute_domain_metrics", "plot_cgm_wear", "plot_glycemic_cohort"),
        parameters=("sufficient_only",), columns=("cgm", "valid", "sufficient", "active_percent", "span_active_percent"),
        defaults=("cgm_day_completeness", "cgm_sufficient_days", "fixed_cadence"),
        related=("day-completeness", "glucose-ranges"), aliases=("cgm validity", "cgm data", "sufficient cgm data")),
    "glucose-ranges": _Concept(
        "Glucose ranges and consensus targets", "glucose",
        "Time in each glucose range, and the targets the consensus sets for them.",
        ("Readings are classified into five ranges ({d:glucose_ranges}), and each range's time is the share of "
         "readings in it. The consensus targets for most adults with diabetes are {d:glucose_targets}. For people "
         "without diabetes these targets are a reference, not a goal.",
         "Variability is the coefficient of variation (sd over mean); at or below the stability threshold, glucose "
         "counts as stable."),
        tools=("glucose_range", "cgm_metrics", "plot_cgm_ranges", "plot_glycemic_cohort"),
        columns=("in_range_percent", "very_low_percent", "cv_percent", "meets_in_range"),
        defaults=("glucose_ranges", "glucose_targets"), related=("cgm-validity", "gmi"),
        aliases=("time in range", "glucose range", "consensus targets")),
    "agp": _Concept(
        "The ambulatory glucose profile (AGP)", "glucose",
        "One participant's glucose by time of day, over all their valid days.",
        ("The AGP stacks a participant's valid days on one 24-hour axis, in bins of {d:agp}, and draws the median "
         "with the interquartile and outer percentile bands. It shows when in the day glucose runs high or varies, "
         "which daily means hide. The cohort version draws the median across participants of their own medians."),
        tools=("cgm_profile", "plot_agp", "plot_glucose_days"), columns=("minute", "p5", "p50", "p95"),
        defaults=("agp",), related=("cgm-validity",), aliases=("ambulatory glucose profile", "glucose profile")),
    "gmi": _Concept(
        "The glucose management indicator (GMI)", "glucose",
        "An estimate of HbA1c from mean CGM glucose.",
        ("GMI (%) = 3.31 + 0.02392 × mean glucose in mg/dL (Bergenstal et al., Diabetes Care 2018), computed over "
         "valid days. It estimates what an HbA1c test would show; the two can differ in a given person."),
        tools=("cgm_metrics", "plot_glycemic_cohort"), columns=("gmi_percent", "mean_mgdl"),
        related=("glucose-ranges",), aliases=("glucose management indicator",)),
    # ------------------------------------------------------------------------------------ reports and safety
    "small-cells": _Concept(
        "Small-cell suppression", "output",
        "Aggregates resting on too few participants are hidden, for figures leaving the lab.",
        ("With min_participants=k, any point, bar, cell or bin resting on fewer than k participants is not drawn, its "
         "values are removed from figure.data too, and the figure states how many were hidden. The default is "
         "{o:min_participants}; 1 hides nothing. Choose k by the data-sharing rules that apply. Reports with k above 1 also leave out every "
         "individual-level page, since those cannot respect a threshold."),
        tools=("report", "plot_hour_of_day", "plot_metric_distributions", "plot_feature_correlations"),
        parameters=("min_participants",), columns=("suppressed",), related=("individual-level",),
        aliases=("small cell", "suppression", "small cell suppression")),
    "individual-level": _Concept(
        "Individual and cohort figures", "output",
        "Some figures show participants one by one; others only aggregates.",
        ("figure.individual_level says which: rows, points or bars per person, or cohort aggregates. Participant "
         "identifiers appear only with show_ids=True, and real dates only with time_axis=\"calendar\". A shareable "
         "report (include_individual=False, or min_participants above 1) leaves the individual pages out."),
        tools=("report", "plot_participant_overview", "plot_sleep_raster"), parameters=("show_ids", "include_individual"),
        related=("small-cells",), aliases=("individual level", "privacy")),
    "group-labels": _Concept(
        "Groups: labels and sizes", "values",
        "One line, box or bar per group, each with its size stated.",
        ("groups= maps each participant to a label: a dict, a Series, or a table with RegistrationCode and group. "
         "Registration codes are matched in their 10K_ form, with or without the prefix. Participants without a "
         "label are left out, and the figure says how many; groups smaller than min_participants are hidden."),
        tools=("plot_metric_distributions", "plot_glycemic_cohort", "plot_retention"), parameters=("groups",),
        columns=("group",), related=("small-cells",), aliases=("group labels",)),
    "uncertainty": _Concept(
        "Spread and uncertainty", "values",
        "The interquartile band shows how participants differ; a confidence interval shows how sure the median is.",
        ("Cohort figures draw the median with its interquartile band, the spread between participants. ci= adds a "
         "bootstrap interval of the cohort median ({d:bootstrap_samples} resamples, seeded by seed), which narrows as "
         "participants are added while the band does not. They answer different questions and are drawn "
         "differently."),
        tools=("plot_hour_of_day", "plot_weekly_pattern", "plot_monthly_pattern"), parameters=("ci", "seed"),
        columns=("ci_low", "ci_high", "p25", "p75"), defaults=("bootstrap_samples",),
        aliases=("confidence interval", "bootstrap", "spread and uncertainty")),
    "figure-data": _Concept(
        "Every figure carries its data", "output",
        "What a figure draws is available as a table, exactly.",
        ("figure.data holds the table the figure draws, and figure.panels each panel's table, with suppressed values "
         "removed. A number read off a figure can therefore be checked, cited or re-plotted. utils.info(\"<figure>\") "
         "lists their columns."),
        tools=("plot_agp", "report"), related=("small-cells",), aliases=("figure data",)),
    # ---------------------------------------------------------------------------------------------- multimodal
    "co-availability": _Concept(
        "Co-availability of modalities", "multimodal",
        "A multimodal analysis needs days on which every chosen feature has data.",
        ("day_overlap gives, for every pair of features, the participant-days holding both and their Jaccard index; "
         "days_with gives each participant's days holding every one of a set of features. How many participants keep "
         "at least k such days shrinks quickly as features are added: plot_multimodal_days shows that curve before a "
         "dataset is fixed."),
        tools=("day_overlap", "days_with", "plot_co_availability", "plot_multimodal_days", "plot_feature_combinations"),
        columns=("jaccard", "days_both", "complete_days"), related=("within-person",),
        aliases=("co availability", "modalities together")),
    "within-person": _Concept(
        "Within-person associations", "multimodal",
        "Whether a person's own good days in one series go with their good days in another.",
        ("plot_lagged_association pairs day d of one series with day d + lag of another, and uses each participant's "
         "deviations from their own means, so that differences between people cannot pose as effects within them. "
         "The pooled within-participant slope and correlation summarize it; each participant's own correlation is "
         "kept too. \"sleep\" names total sleep of the night starting on day d."),
        tools=("plot_lagged_association", "plot_feature_correlations"), parameters=("lag_days",),
        columns=("within_slope", "within_r", "x_dev", "y_dev"), related=("co-availability",),
        aliases=("lagged association", "within person")),
}


# ---------------------------------------------------------------------------------------------------- defaults
def _num(value) -> str:
    """A number as a person writes it: 70 rather than 70.0, 18.5 as is."""

    if isinstance(value, float) and value == float("inf"):
        return "∞"
    return f"{value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def _pct(fraction: float) -> str:
    return f"{fraction * 100:g}%"


def _and(items) -> str:
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _rule_text(rule) -> str:
    parts = []
    if rule.min_hours_with_data is not None:
        parts.append(f"at least {_num(rule.min_hours_with_data)} hours with data")
    if rule.min_completeness is not None:
        parts.append(f"at least {_pct(rule.min_completeness)} of its expected records")
    if rule.min_records > 1 or not parts:
        parts.append(f"at least {rule.min_records} record" + ("" if rule.min_records == 1 else "s"))
    return " and ".join(parts)


def _bands(entries, unit: str) -> str:
    out = []
    for name, low, high in entries:
        if high == float("inf"):
            out.append(f"{name} from {_num(low)}")
        elif not low:
            out.append(f"{name} below {_num(high)}")
        else:
            out.append(f"{name} {_num(low)} to below {_num(high)}")
    return "; ".join(out) + f" ({unit})"


_OPERATORS = {"<": "below", "<=": "at most", ">": "above", ">=": "at least"}
_UNITS = {"Cel": "°C"}


@dataclass(frozen=True)
class _Default:
    """A threshold, rule or setting, read from the code whenever it is shown."""

    title: str
    group: str
    sources: tuple[str, ...]          # "module.CONSTANT", or "module.Class.field" for a dataclass default
    meaning: str
    basis: str
    used_by: tuple[str, ...]
    change: str
    inline: Any                       # the live values -> compact text


DEFAULTS: dict[str, _Default] = {
    "valid_day_rules": _Default(
        "Valid-day rules", "adherence", ("data_summaries.DEFAULT_RULES", "data_summaries.DEFAULT_RULE"),
        "When a participant-day counts as valid, per feature.",
        "Conventions: at least 10 hours with data is a common wear-time criterion for heart rate; at least 70% of "
        "expected readings is the international consensus criterion for CGM (Battelino et al., Diabetes Care 2019); "
        "any other feature needs at least one record.",
        ("summarize", "temporal_patterns", "plot_valid_day_rule", "plot_daily_values"),
        "rules={feature: sm.ValidDayRule(...)} on summarize, temporal_patterns and the figures that take rules.",
        lambda rules, default: "; ".join(f"{f} {_rule_text(r)}" for f, r in rules.items())
        + f"; every other feature {_rule_text(default)}"),
    "fixed_cadence": _Default(
        "Fixed cadence", "adherence", ("data_summaries.FIXED_CADENCE",),
        "Which features sample on a fixed period, and when a participant's data shows it. Only then are expected "
        "records per day, and completeness, defined.",
        "A continuous glucose monitor records every 5 or 15 minutes; finger-stick readings do not. Requiring a short, "
        "regular typical gap keeps occasional readings from being judged against a monitor's expected readings.",
        ("summarize", "mark_valid_days", "cgm_metrics"),
        "A module constant; change it in data_summaries only with a reason recorded.",
        lambda cadence: "; ".join(f"{f}, when the typical gap is at most {_num(c.max_typical_gap_minutes)} minutes "
                                  f"and at least {_pct(c.min_regular_share)} of gaps are within 10% of it"
                                  for f, c in cadence.items())),
    "night_rule": _Default(
        "Valid nights", "sleep", ("domain_metrics.NightRule.min_asleep_minutes",),
        "The minimum main sleep for a night to count in sleep summaries and figures.",
        "A convention: three hours of main sleep separate a night's sleep from fragments and naps recorded alone.",
        ("summarize_domain", "sleep_regularity", "plot_sleep", "plot_sleep_timing", "plot_sleep_architecture",
         "plot_sleep_regularity"),
        "night_rule=dm.NightRule(min_asleep_minutes=...) on summarize_domain; rule= on sleep_regularity and the sleep "
        "figures.", lambda minutes: _num(minutes)),
    "night_start_hour": _Default(
        "Night start", "sleep", ("domain_metrics.NIGHT_START_HOUR",),
        "The local hour at which a night begins and the previous one ends.",
        "A common convention: noon lies outside almost all night-time sleep, so no night is split.",
        ("sleep_nights", "load_sleep_segments", "compute_domain_metrics"), "A module constant, recorded in run.json.",
        lambda hour: _num(hour)),
    "episode_gap": _Default(
        "Sleep episode gap", "sleep", ("domain_metrics.EPISODE_GAP_MINUTES",),
        "Asleep records separated by at most this many minutes form one sleep episode.",
        "On the samples, gaps between asleep records are at most 120 minutes within a night and over 360 minutes "
        "between nights.", ("sleep_nights", "compute_domain_metrics"), "A module constant, recorded in run.json.",
        lambda minutes: _num(minutes)),
    "sleep_states": _Default(
        "Sleep states", "sleep", ("data_statistics.SLEEP_STATES", "data_statistics.ASLEEP_STATES",
                                  "domain_metrics.ASLEEP_STATES", "domain_metrics.STAGE_STATES"),
        "The sleep states HealthKit records, which of them count as asleep, and which are stages.",
        "HealthKit's own states: asleep without a stage, and the core, deep and REM stages of devices that stage "
        "sleep.", ("compute_daily_statistics", "sleep_nights"), "Fixed by HealthKit.",
        lambda states, asleep, _asleep, stages: f"{_and(states)}; asleep means "
        f"{_and(s for s in states if s in asleep)}; the stages are {_and(stages)}"),
    "mgdl_per_mmol": _Default(
        "Glucose unit conversion", "glucose", ("domain_metrics.MGDL_PER_MMOL",),
        "mg/dL per mmol/L, used to classify stored mmol/L readings in the consensus ranges' unit.",
        "HealthKit's factor, from the molar mass of glucose (180.15588 g/mol). On the samples every stored reading "
        "converts to a whole mg/dL value within 1e-10.", ("glucose_mgdl", "cgm_metrics", "load_cgm_readings"),
        "Fixed.", lambda factor: f"{factor:g} mg/dL per mmol/L"),
    "cgm_day_completeness": _Default(
        "Valid CGM day", "glucose", ("domain_metrics.CGM_MIN_DAY_COMPLETENESS",),
        "The share of a day's expected readings a CGM day needs to be valid.",
        "The international consensus on CGM metrics (Battelino et al., Diabetes Care 2019).",
        ("cgm_metrics", "compute_domain_metrics", "load_cgm_readings"), "A module constant, recorded in run.json.",
        lambda share: _pct(share)),
    "cgm_sufficient_days": _Default(
        "Sufficient CGM data", "glucose", ("domain_metrics.CGM_SUFFICIENT_DAYS",),
        "The valid CGM days a participant needs for sufficient consensus metrics.",
        "The international consensus: 14 days of CGM data approximate three months of glycaemia (Battelino et al., "
        "Diabetes Care 2019).", ("cgm_metrics", "plot_glycemic_cohort", "plot_cgm_ranges"),
        "sufficient_only=False on the glucose figures includes insufficient participants.", lambda days: _num(days)),
    "glucose_ranges": _Default(
        "Glucose ranges", "glucose", ("domain_metrics.GLUCOSE_RANGES",),
        "The five consensus glucose ranges, in mg/dL; a reading on a boundary of the target range counts as in range.",
        "The international consensus on time in ranges (Battelino et al., Diabetes Care 2019).",
        ("cgm_metrics", "plot_cgm_ranges", "plot_agp"), "Fixed by the consensus.",
        lambda ranges: _and(f"below {_num(hi)}" if lo is None else f"above {_num(lo)}" if hi is None
                            else f"{_num(lo)} to {_num(hi)}" for _, lo, hi in ranges) + " mg/dL"),
    "glucose_targets": _Default(
        "Glucose targets", "glucose", ("data_plots.GLUCOSE_TARGETS", "data_plots.CV_STABILITY_THRESHOLD",
                                       "data_plots.TIME_IN_RANGE_TARGET"),
        "The consensus targets for time in ranges, and the variability at or below which glucose counts as stable.",
        "Targets for most adults with diabetes (Battelino et al., Diabetes Care 2019); the CV threshold from Danne et "
        "al., Diabetes Care 2017. For people without diabetes they are a reference, not a goal.",
        ("plot_glycemic_cohort", "plot_cgm_ranges"), "Fixed by the consensus.",
        lambda targets, cv, tir: "; ".join(name for name, *_ in targets)),
    "agp": _Default(
        "AGP bins and percentiles", "glucose", ("domain_metrics.AGP_BIN_MINUTES", "domain_metrics.AGP_PERCENTILES"),
        "The width of the AGP's time-of-day bins, and the percentiles drawn.",
        "The standard ambulatory glucose profile.", ("cgm_profile", "plot_agp", "plot_glucose_days"),
        "bin_minutes= on cgm_profile.", lambda minutes, percentiles: f"{_num(minutes)}-minute bins, with the "
        f"{_and(_num(p) for p in percentiles)}th percentiles"),
    "clinical_thresholds": _Default(
        "Clinical thresholds", "clinical", ("data_statistics.CLINICAL_THRESHOLDS",),
        "Readings beyond these are counted per day, per participant and for the cohort.",
        "SpO2 below 90% (hypoxaemia) and a body temperature of 38.0 °C or more (fever), counted where the values are "
        "in the threshold's unit; counts, not diagnoses.", ("compute_daily_statistics", "clinical_thresholds", "plot_clinical_thresholds"),
        "A module constant; each threshold becomes a daily column.",
        lambda thresholds: "; ".join(f"{feature} {_OPERATORS[op]} {_num(limit)} {_UNITS.get(unit, unit)}"
                                     for feature, entries in thresholds.items() for _, op, limit, unit in entries)),
    "plausible_ranges": _Default(
        "Plausible ranges", "quality", ("data_statistics.PLAUSIBLE_RANGES",),
        "Per feature, the range outside which values are counted as implausible (never removed). The daily statistics "
        "count them; quality_report and plot_quality summarize those counts.",
        "Broad physiological limits in each feature's harmonized unit, for reporting values that are likely errors: "
        "conventions, not clinical thresholds, applied only when the values are in their unit.",
        ("compute_daily_statistics",), "A module constant.",
        lambda ranges: f"{len(ranges)} features"),
    "weekend_days": _Default(
        "Weekend", "patterns", ("data_summaries.WEEKEND_DAYS",),
        "The days of the week that count as weekend (Monday is 0). Free nights are those ending on them.",
        "Friday and Saturday, the weekend in Israel, where the cohort lives.",
        ("temporal_patterns", "activity_goals", "sleep_regularity", "plot_weekday_weekend", "plot_sleep_regularity"),
        "weekend_days= on temporal_patterns, activity_goals, sleep_regularity and the figures that take it.",
        lambda days: _and(calendar.day_name[d] for d in days)),
    "home_zone_share": _Default(
        "Home zone", "patterns", ("data_summaries.DST_PARTNER_MIN_SHARE",),
        "The share of days the offset 60 minutes from a participant's most common one must cover to join the home "
        "zone as its daylight-saving time.", "A daylight-saving season covers a large share of days; a short trip an hour away does not.", ("time_zones", "plot_time_zones"),
        "A module constant.", lambda share: _pct(share)),
    "bmi_classes": _Default(
        "BMI classes", "clinical", ("data_plots.BMI_CLASSES",), "The adult BMI classes, in kg/m².",
        "The WHO classification of adult BMI; each class runs from its lower bound up to, not including, its upper.", ("bmi_categories", "plot_bmi_categories"), "Fixed by the WHO.",
        lambda classes: _bands(classes, "kg/m²")),
    "step_categories": _Default(
        "Daily-step categories", "clinical", ("data_plots.STEP_CATEGORIES",), "Categories of median daily steps.",
        "Tudor-Locke and Bassett (2004) and Tudor-Locke et al. (2008); each category runs from its lower bound up to, "
        "not including, its upper.",
        ("step_categories", "plot_step_categories"), "Fixed.", lambda categories: _bands(categories, "steps a day")),
    "who_weekly_minutes": _Default(
        "Weekly activity recommendation", "clinical", ("data_plots.WHO_WEEKLY_MINUTES",),
        "The weekly minutes of exercise a complete week is compared with.",
        "The WHO's 2020 guidelines: 150 to 300 minutes of moderate activity a week for adults; this is the lower bound.",
        ("activity_goals", "plot_activity_goals"), "Fixed by the WHO.", lambda minutes: f"{_num(minutes)} minutes"),
    "bp_guidelines": _Default(
        "Blood-pressure guidelines", "clinical", ("data_plots.BP_GUIDELINES",),
        "The categories a participant's median blood pressure falls in, under each guideline.",
        "ESC/ESH defines hypertension at home from 135/85 mmHg (2018 and 2023 guidelines), stricter than in the office; "
        "ACC/AHA 2017 has its own categories. A participant falls in the highest category either pressure reaches, the "
        "guidelines' and/or. The figures default to the ESC/ESH home thresholds.",
        ("blood_pressure", "plot_blood_pressure"), "guideline= on blood_pressure and plot_blood_pressure.",
        lambda guidelines: _and(guidelines)),
    "activity_rings": _Default(
        "Activity rings", "clinical", ("data_plots.ACTIVITY_RINGS",),
        "The three Apple activity rings, each with its daily value and goal in ActivitySummary.",
        "Apple's Move, Exercise and Stand rings.", ("activity_goals", "plot_activity_goals"), "Fixed by Apple.",
        lambda rings: _and(name for name, *_ in rings)),
    "bootstrap_samples": _Default(
        "Bootstrap resamples", "values", ("data_plots.BOOTSTRAP_SAMPLES",),
        "The resamples behind a confidence interval of the cohort median (ci=).",
        "A common choice for percentile intervals; seed= makes them reproducible.",
        ("plot_hour_of_day", "plot_weekly_pattern", "plot_monthly_pattern"), "A module constant.",
        lambda n: _num(n)),
    "summary_statistics": _Default(
        "Participant statistics", "values", ("data_summaries.STATISTICS",),
        "What participant summaries report for each metric over valid days.", "A distribution's usual description.",
        ("summarize", "summarize_domain"), "Fixed.", lambda statistics: _and(statistics)),
    "domain_summary_metrics": _Default(
        "Summarized night and CGM metrics", "sleep", ("domain_metrics.SLEEP_SUMMARY_METRICS",
                                                      "domain_metrics.CGM_SUMMARY_METRICS"),
        "The night and CGM metrics summarize_domain describes per participant, with their units.",
        "The metrics the sleep and CGM literature reports.", ("summarize_domain",), "Fixed.",
        lambda sleep, cgm: f"{len(sleep)} night metrics and {len(cgm)} CGM metrics"),
    "overview_features": _Default(
        "One participant's overview", "values", ("data_plots.OVERVIEW_FEATURES",),
        "The features plot_participant_overview draws by default.", "Activity, heart, sleep, body and glucose.",
        ("plot_participant_overview",), "features= on plot_participant_overview.", lambda features: _and(features)),
    "volume_bands": _Default(
        "Data-volume bands", "coverage", ("data_plots.VOLUME_BANDS",), "The size bands of participants' data.",
        "Decades of bytes.", ("plot_data_volume",), "Fixed.", lambda bands: _and(label for *_, label in bands)),
    "report_sections": _Default(
        "Report sections", "output", ("data_plots.REPORT_SECTIONS",), "The parts sections= can keep in a report.",
        "The report's own structure.", ("report",), "sections= on report.", lambda sections: _and(sorted(sections))),
    "feature_order": _Default(
        "Feature categories", "identity", ("data_plots.CATEGORY_ORDER",),
        "The order of feature categories in every figure; each feature also keeps one color.",
        "From activity to nutrition, so related features sit together.",
        ("plot_participants_per_feature",), "Fixed.",
        lambda order: _and(order)),
    "output_schemas": _Default(
        "Output schemas", "output", ("data_statistics.OUTPUT_SCHEMA", "domain_metrics.DOMAIN_OUTPUT_SCHEMA"),
        "The version of the written tables' layout, recorded in run.json.",
        "Raised whenever a written table changes, so that a run is never resumed into a different layout; the module's "
        "comment records what each version changed.",
        ("compute_daily_statistics", "compute_domain_metrics"), "Fixed; compute an older run again.",
        lambda statistics, domain: f"{statistics} for statistics runs and {domain} for domain runs"),
    "phases": _Default(
        "Phases", "runs", ("data_statistics.PHASES",), "The two copies of the data a tool can read.",
        "The native export and its curated copy.", ("compute_coverage", "compute_daily_statistics",
                                                    "compute_domain_metrics"),
        "phase= on each of them.", lambda phases: _and(phases)),
    "data_roots": _Default(
        "Data roots", "runs", ("data_statistics.DEFAULT_CURATED_ROOT", "data_statistics.DEFAULT_NATIVE_ROOT"),
        "Where each phase is read from when root= is not given; outputs are never written inside them.",
        "The permanent HPP roots of the two phases.",
        ("compute_coverage", "compute_daily_statistics", "compute_domain_metrics", "guard_output"),
        "root= on the tools that read data.", lambda curated, native: f"{curated.name} (curated) and {native.name} "
                                                                     f"(native), under {curated.parent}"),
}

# Public constants that are not analytical defaults, with the reason.
NOT_DEFAULTS = {
    **dict.fromkeys(("HOURLY_COLUMNS", "PROVENANCE_COLUMNS", "CURATION_COLUMNS", "PARTICIPANT_TABLES",
                     "CLINICAL_COLUMNS", "NIGHT_COLUMNS", "CGM_DAY_COLUMNS", "CGM_PERIOD_COLUMNS",
                     "CGM_READING_COLUMNS", "CGM_PROFILE_COLUMNS", "SEGMENT_COLUMNS", "REGULARITY_COLUMNS",
                     "DOMAIN_TABLES", "TARGET_COLUMNS"), "column lists and table names, documented as output tables"),
    **dict.fromkeys(("WEEKDAY_NAMES", "MONTH_NAMES", "PALETTE", "SINGLE", "BAND", "GLUCOSE_COLORS", "GLUCOSE_LABELS",
                     "CATEGORY_COLORS", "SLEEP_COLORS", "SLEEP_LABELS", "THRESHOLD_LABELS", "ALL_PARTICIPANTS"),
                    "presentation: names, labels and colors"),
    "TYPE_CHECKING": "typing",
}


# --------------------------------------------------------------------------------------------------- workflows
@dataclass(frozen=True)
class _Workflow:
    """A task from start to finish, as code that runs as printed."""

    title: str
    group: str
    question: str
    steps: tuple[tuple[str, str], ...]
    reading: tuple[str, ...]
    concepts: tuple[str, ...] = ()
    phases: tuple[str, ...] = ("curated", "native")
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.reading, str):
            object.__setattr__(self, "reading", (self.reading,))


_START = '''from pathlib import Path
from wearable_project.utils import data_statistics as ds, data_summaries as sm, domain_metrics as dm, data_plots as dp

out = Path("{out}")
out.mkdir(parents=True, exist_ok=True)'''

WORKFLOWS: dict[str, _Workflow] = {
    "first-look": _Workflow(
        "A first look at a cohort", "coverage",
        "Who has which data, how much, and over what period?",
        (("Coverage answers who has which feature for the whole cohort in seconds, without reading the records.",
          _START + '''
coverage = ds.compute_coverage("{phase}"{root})
coverage.features.sort_values("participants", ascending=False).head(10)'''),
         ("Draw who has what: participants per feature, each participant's number of features, and the combinations "
          "that occur together.",
          '''dp.plot_participants_per_feature(coverage, path=out / "participants_per_feature.png")
dp.plot_features_per_participant(coverage, path=out / "features_per_participant.png")
dp.plot_feature_combinations(coverage, ["StepCount", "HeartRate", "Sleep", "Weight"], path=out / "combinations.png")'''),
         ("Then when the data exist: daily statistics for the features of interest, and the figures of time.",
          '''daily = ds.compute_daily_statistics("{phase}"{root}, features=["StepCount", "HeartRate", "Sleep"])
dp.plot_availability_raster(daily, path=out / "availability.png")
dp.plot_follow_up(daily, path=out / "follow_up.png")
dp.plot_active_participants(daily, rolling=7, path=out / "active_participants.png")''')),
        ("The participants with a feature are the ceiling of every analysis that uses it; the combinations show how "
         "fast that ceiling falls when features are required together.",
         "The availability raster shows each participant's months with data, so enrolment waves, drop-out and gaps "
         "appear at once. The follow-up figure gives each participant's span and how densely it is filled.",
         "At cohort scale, compute the daily statistics once into a directory (the workflow \"cohort run\")."),
        concepts=("coverage-tiers", "local-days", "co-availability", "study-time"), aliases=("first look", "start")),
    "quality-review": _Workflow(
        "A data-quality review", "quality",
        "Can the data be trusted, and where must they be read with care?",
        (("Daily statistics for the features to review, and their quality report.",
          _START + '''
daily = ds.compute_daily_statistics("{phase}"{root}, features=["HeartRate", "StepCount", "Weight", "OxygenSaturation"])
quality = sm.quality_report(daily)
quality.features[["feature", "records", "values_below_range", "values_above_range", "without_offset_share",
                  "redundant_share"]]'''),
         ("Implausible values, redundant devices, hand-entered records and how they were acquired.",
          '''dp.plot_quality(quality, path=out / "quality.png")
dp.plot_acquisition(quality, path=out / "acquisition.png")'''),
         ("How regularly each participant's devices record, which hours hold data, and travel.",
          '''dp.plot_sampling_cadence(daily, path=out / "sampling_cadence.png")
dp.plot_hourly_coverage(daily, "HeartRate", path=out / "hourly_coverage.png")
dp.plot_time_zones(daily, path=out / "time_zones.png")'''),
         ("In the curated phase, what the curation questioned.",
          '''if len(quality.curation):
    dp.plot_curation(quality, path=out / "curation.png")
    dp.plot_curation_flags(quality, path=out / "curation_flags.png")''')),
        ("Values beyond the plausible range are counted, not removed: decide what to exclude, and say so.",
         "A high redundant share means several devices recorded the same time. Time is counted once, but totals such "
         "as steps are summed across devices.",
         "Records without a UTC offset fall on their UTC day; a large share of them shifts daily patterns.",
         "In the curated phase, default_inclusion_only=True restricts every statistic to what the curation keeps by "
         "default; records_included shows the effect before choosing."),
        concepts=("plausible-ranges", "time-counted-once", "provenance", "curation", "sampling-cadence", "time-zones"),
        aliases=("quality review", "data quality", "trust")),
    "choosing-valid-days": _Workflow(
        "Choosing a valid-day rule", "adherence",
        "Which days are good enough to use, and what does the choice cost?",
        (("Daily statistics for the feature, and the rule's effect before committing to it.",
          _START + '''
daily = ds.compute_daily_statistics("{phase}"{root}, features=["HeartRate", "StepCount"])
dp.plot_valid_day_rule(daily, "HeartRate", thresholds=range(0, 25, 2), path=out / "valid_day_rule.png")'''),
         ("Apply the chosen rule, and see each participant's valid days and adherence.",
          '''rules = {"HeartRate": sm.ValidDayRule(min_hours_with_data=12)}
summaries = sm.summarize(daily, rules=rules)
summaries.adherence.query("feature == 'HeartRate'")[["RegistrationCode", "valid_days", "adherence",
                                                     "longest_valid_run"]]'''),
         ("Adherence and retention under that rule.",
          '''dp.plot_adherence(summaries, "HeartRate", path=out / "adherence.png")
dp.plot_retention(summaries, ["HeartRate"], path=out / "retention.png")''')),
        ("The left panel shows the criterion over all participant-days; a threshold where few days lie loses little. "
         "The right panel shows, as the threshold moves, the share of days kept and of participants keeping at least "
         "k valid days.",
         "summaries.rules records the rules applied: report them with every result."),
        concepts=("valid-days", "day-completeness", "adherence-and-gaps", "retention-curve"),
        aliases=("choosing a valid day rule", "valid day workflow")),
    "cohort-run": _Workflow(
        "Computing the whole cohort once", "runs",
        "How is the full cohort computed, resumed and read back without running out of memory?",
        (("Write the runs into directories. Each participant's rows are written as it finishes; resume=True continues "
          "an interrupted run.",
          '''from pathlib import Path
from wearable_project.utils import data_statistics as ds, data_summaries as sm, domain_metrics as dm

run_dir, domain_dir = Path("{out}") / "statistics_run", Path("{out}") / "domain_run"
ds.compute_daily_statistics("{phase}"{root}, features=["StepCount", "HeartRate"], out=run_dir, resume=True)
dm.compute_domain_metrics("{phase}"{root}, out=domain_dir, resume=True)'''),
         ("Read the tables back, whole or one chunk at a time.",
          '''participant_feature = ds.read_table(run_dir, "participant_feature")
steps = ds.read_daily(run_dir, "StepCount")
for chunk in ds.iter_daily(run_dir, "HeartRate"):
    pass  # a bounded number of rows at a time
nights = dm.read_domain_table(domain_dir, "sleep_nights")'''),
         ("Every tool that takes a run accepts its directory, and reads it one participant at a time.",
          '''summaries = sm.summarize(run_dir)
patterns = sm.temporal_patterns(run_dir)
summaries.cohort.head()''')),
        ("workers= computes several participants at once. The same runs are available from the command line: python "
         "-m wearable_project.utils.data_statistics daily --phase curated --out DIR.",
         "run.json records versions, registry fingerprints and the output schema; a run with another schema cannot "
         "be resumed. export_parquet(run_dir) adds Parquet copies for faster reading (it needs pyarrow).",
         "Outputs are refused inside any data root."),
        concepts=("written-runs", "data-roots", "coverage-tiers"), aliases=("cohort run", "full cohort", "at scale")),
    "table-one": _Workflow(
        "A cohort Table 1", "values",
        "What are typical values in the cohort, counting each participant once?",
        (("Participant summaries over valid days, and the cohort table of their medians.",
          _START + '''
daily = ds.compute_daily_statistics("{phase}"{root}, features=["StepCount", "RestingHeartRate", "Weight"])
summaries = sm.summarize(daily)
summaries.cohort.query("metric in ['value_sum', 'value_mean']")[["feature", "metric", "unit", "participants",
                                                                   "median", "p25", "p75"]]'''),
         ("The same as a figure, whose table1 panel holds Table 1, and every participant's median with its "
          "interquartile range.",
          '''figure = dp.plot_metric_distributions(summaries, ["StepCount", "RestingHeartRate"], path=out / "table1.png")
figure.panels["table1"]
dp.plot_caterpillar(summaries, "StepCount", path=out / "caterpillar.png")''')),
        ("Each participant contributes one value, their median valid day, so heavy wearers do not dominate.",
         "n_days in participant_metrics says how many valid days stand behind each participant's median."),
        concepts=("participant-medians", "valid-days", "figure-data"), aliases=("table 1", "table one")),
    "daily-weekly-patterns": _Workflow(
        "Finding daily and weekly patterns", "patterns",
        "How do values change over the day, the week and the year?",
        (("Temporal patterns over valid days.",
          _START + '''
daily = ds.compute_daily_statistics("{phase}"{root}, features=["HeartRate", "StepCount"])
patterns = sm.temporal_patterns(daily)'''),
         ("The week: each weekday, then weekends against weekdays, paired within participants.",
          '''dp.plot_weekly_pattern(patterns, "StepCount", ci=0.95, seed=1, path=out / "weekly.png")
dp.plot_weekday_weekend(patterns, "StepCount", path=out / "weekday_weekend.png")'''),
         ("The day: the cohort's hours, several features' rhythms side by side, and each participant's profile.",
          '''participants, cohort = sm.hour_of_day(daily)
dp.plot_hour_of_day(daily, "HeartRate", path=out / "hour_of_day.png")
dp.plot_daily_rhythms(daily, ["HeartRate", "StepCount"], path=out / "daily_rhythms.png")
dp.plot_hour_profiles(daily, "HeartRate", path=out / "hour_profiles.png")''')),
        ("The shaded band is the spread between participants; ci= adds the uncertainty of the median, which is a "
         "different thing.",
         "The weekend is {d:weekend_days} by default; weekend_days= changes it.",
         "Relative profiles divide each participant's hours by their own mean, so rhythms compare across levels."),
        concepts=("weekday-weekend", "hour-of-day", "uncertainty", "time-zones"),
        aliases=("patterns workflow", "rhythms")),
    "sleep-analysis": _Workflow(
        "Sleep analysis", "sleep",
        "How long, when, how regularly and how well do participants sleep?",
        (("Nights from noon to noon, and what each night recorded.",
          '''from pathlib import Path
from wearable_project.utils import domain_metrics as dm, data_plots as dp

out = Path("{out}")
out.mkdir(parents=True, exist_ok=True)
metrics = dm.compute_domain_metrics("{phase}"{root})
nights = metrics.sleep_nights
nights[["asleep_recorded", "in_bed_recorded", "staged"]].mean()'''),
         ("Per participant, over valid nights.",
          '''summary = dm.summarize_domain(metrics)
summary.cohort.head(12)'''),
         ("What nights record, then duration, timing, stages and regularity.",
          '''dp.plot_sleep_recording(metrics, path=out / "sleep_recording.png")
dp.plot_sleep(metrics, path=out / "sleep.png")
dp.plot_sleep_timing(metrics, path=out / "sleep_timing.png")
dp.plot_sleep_architecture(metrics, path=out / "sleep_architecture.png")
dp.plot_sleep_regularity(metrics, path=out / "sleep_regularity.png")'''),
         ("One participant's nights.",
          '''sleeper = nights["RegistrationCode"].value_counts().index[0]
segments = dm.load_sleep_segments(sleeper, "{phase}"{root})
dp.plot_sleep_raster(segments, path=out / "sleep_raster.png")''')),
        ("Nights with time in bed only have no measured sleep; they are kept in the tables and left out of sleep "
         "metrics. Look at plot_sleep_recording first.",
         "Efficiency over time in bed and over the sleep period differ; efficiency_basis says which.",
         "Social jetlag compares free nights (ending on a weekend day) with work nights."),
        concepts=("noon-to-noon-night", "sleep-episodes", "valid-nights", "in-bed-only", "sleep-efficiency",
                  "sleep-staging", "free-nights"),
        aliases=("sleep workflow",)),
    "cgm-analysis": _Workflow(
        "Continuous glucose monitoring", "glucose",
        "What do participants' CGM data show against the consensus metrics?",
        (("CGM metrics for participants whose glucose data has a monitor's cadence.",
          '''from pathlib import Path
from wearable_project.utils import domain_metrics as dm, data_plots as dp

out = Path("{out}")
out.mkdir(parents=True, exist_ok=True)
metrics = dm.compute_domain_metrics("{phase}"{root})
periods = metrics.cgm_periods[metrics.cgm_periods["cgm"].astype(bool)]
periods[["RegistrationCode", "valid_days", "sufficient", "mean_mgdl", "gmi_percent", "in_range_percent",
         "cv_percent"]]'''),
         ("The cohort against the consensus targets, time in ranges, and wear.",
          '''dp.plot_glycemic_cohort(metrics, sufficient_only=False, path=out / "glycemic_cohort.png")
dp.plot_cgm_ranges(metrics, sufficient_only=False, path=out / "cgm_ranges.png")
dp.plot_cgm_wear(metrics, path=out / "cgm_wear.png")'''),
         ("One participant's ambulatory glucose profile and daily traces.",
          '''wearer = periods["RegistrationCode"].iloc[0]
dp.plot_agp(metrics, wearer, path=out / "agp.png")
readings = dm.load_cgm_readings(wearer, "{phase}"{root})
dp.plot_glucose_days(readings, path=out / "glucose_days.png")''')),
        ("Consensus metrics need at least {d:cgm_sufficient_days} valid days (sufficient); sufficient_only=False includes the others, "
         "which the figures say.",
         "The targets are those for most adults with diabetes: for others, a reference.",
         "GMI estimates HbA1c from mean glucose; the two can differ in a given person.",
         "The native phase has no usable glucose units, so these figures need the curated phase."),
        concepts=("cgm-validity", "glucose-ranges", "agp", "gmi", "day-completeness"), phases=("curated",),
        aliases=("cgm workflow", "glucose workflow")),
    "comparing-groups": _Workflow(
        "Comparing groups", "values",
        "Do groups of participants differ, and are the groups large enough to say?",
        (("Participant summaries, and a label for each participant. Here the label comes from the data; in practice "
          "it comes from outside it, such as sex, an age band or a clinical label.",
          _START + '''
daily = ds.compute_daily_statistics("{phase}"{root}, features=["StepCount", "RestingHeartRate", "HeartRate"])
summaries = sm.summarize(daily)
steps = summaries.participant_metrics.query("feature == 'StepCount' and metric == 'value_sum'")
cut = steps["median"].median()
groups = {code: "more steps" if median >= cut else "fewer steps"
          for code, median in zip(steps["RegistrationCode"], steps["median"])}'''),
         ("The same figures, one line, box or bar per group, each with its size.",
          '''dp.plot_metric_distributions(summaries, ["RestingHeartRate"], groups=groups, path=out / "by_group.png")
dp.plot_hour_of_day(daily, "HeartRate", groups=groups, ci=0.95, seed=1, path=out / "hour_by_group.png")
dp.plot_retention(summaries, ["HeartRate"], groups=groups, path=out / "retention_by_group.png")''')),
        ("Each group's size is stated; participants without a label are left out and counted.",
         "Groups defined from the data being compared make the comparison partly circular, as here: this is an "
         "illustration of the mechanics.",
         "For figures leaving the lab, add min_participants so that small groups are hidden."),
        concepts=("group-labels", "participant-medians", "uncertainty", "small-cells"),
        aliases=("compare groups", "group comparison")),
    "shareable-report": _Workflow(
        "A report that can leave the lab", "output",
        "How is a report produced that shows no individual and no small group?",
        (("Coverage and daily statistics for the report.",
          _START + '''
coverage = ds.compute_coverage("{phase}"{root})
daily = ds.compute_daily_statistics("{phase}"{root}, features=["StepCount", "HeartRate", "RestingHeartRate"])'''),
         ("The report, with small cells suppressed and individual pages left out.",
          '''status = dp.report(daily, out / "cohort_report.pdf", coverage=coverage, sections=["cohort", "values"],
                   min_participants=5, include_individual=False)
status["status"].value_counts()'''),
         ("What was not drawn, and why.",
          '''status.query("status != 'drawn'")[["section", "figure", "status", "reason"]]''')),
        ("Choose min_participants by the data-sharing rules that apply; every aggregate resting on fewer participants "
         "is hidden, in the figures and in their data.",
         "Individual pages are left out, identifiers never appear without show_ids=True, and the status table "
         "accounts for every page considered.",
         "The report is written only outside the data roots."),
        concepts=("small-cells", "individual-level", "figure-data", "data-roots"),
        aliases=("shareable report", "sharing", "report for sharing")),
    "multimodal-dataset": _Workflow(
        "Sizing a multimodal dataset", "multimodal",
        "How many participants and days have several modalities together?",
        (("How often pairs of features share a day.",
          _START + '''
daily = ds.compute_daily_statistics("{phase}"{root}, features=["StepCount", "HeartRate", "Sleep", "RestingHeartRate"])
overlap = sm.day_overlap(daily)
overlap.query("feature_a == 'StepCount'").sort_values("jaccard", ascending=False)
dp.plot_co_availability(daily, path=out / "co_availability.png")'''),
         ("Days on which a whole set of features has data, and how many participants keep at least k of them.",
          '''complete = sm.days_with(daily, ["StepCount", "HeartRate", "Sleep"])
dp.plot_multimodal_days(daily, ["StepCount", "HeartRate", "Sleep"], path=out / "multimodal_days.png")'''),
         ("Whether one modality goes with another within participants: steps on a day against sleep that night.",
          '''metrics = dm.compute_domain_metrics("{phase}"{root})
dp.plot_lagged_association(daily, "StepCount", "sleep", domain=metrics, path=out / "steps_and_sleep.png")''')),
        ("Each feature added shrinks the complete days; the curve shows the dataset each requirement leaves.",
         "days_with counts days with data; apply valid-day rules for days of usable quality.",
         "The within-person association uses deviations from each participant's own means."),
        concepts=("co-availability", "within-person", "valid-days"),
        aliases=("multimodal dataset", "sizing a multimodal dataset")),
    "one-participant": _Workflow(
        "One participant, in depth", "values",
        "What does one participant's record look like, day by day?",
        (("Daily statistics and domain metrics, and a participant.",
          _START + '''
daily = ds.compute_daily_statistics("{phase}"{root}, features=["StepCount", "RestingHeartRate", "Sleep", "Weight"])
metrics = dm.compute_domain_metrics("{phase}"{root})
participant = daily.daily["StepCount"]["RegistrationCode"].iloc[0]'''),
         ("Their features over one time axis, their daily values and their travel.",
          '''dp.plot_participant_overview(daily, participant, domain=metrics, path=out / "overview.png")
dp.plot_daily_values(daily, "StepCount", participant=participant, path=out / "steps.png")
dp.plot_time_zones(daily, participant=participant, path=out / "travel.png")''')),
        ("These figures show one person: keep them in the lab. Identifiers appear only with show_ids=True, and real "
         "dates only with time_axis=\"calendar\".",
         "Valid days are drawn filled, other days hollow."),
        concepts=("individual-level", "study-time", "valid-days"), aliases=("one participant", "dossier")),
    "clinical-references": _Workflow(
        "Checking clinical references", "clinical",
        "How does the cohort sit against clinical thresholds and population references?",
        (("Readings beyond the clinical thresholds.",
          _START + '''
daily = ds.compute_daily_statistics("{phase}"{root}, features=["OxygenSaturation", "BodyTemperature", "BMI",
                                                               "StepCount", "BloodPressure", "ActivitySummary"])
summaries = sm.summarize(daily)
participants, cohort = sm.clinical_thresholds(daily)
dp.plot_clinical_thresholds(daily, path=out / "clinical_thresholds.png")'''),
         ("Body size, activity and blood pressure against their references.",
          '''dp.plot_bmi_categories(summaries, path=out / "bmi.png")
dp.plot_step_categories(summaries, path=out / "steps.png")
dp.plot_blood_pressure(summaries, guideline="esc_esh_home", path=out / "blood_pressure.png")
dp.plot_activity_goals(daily, path=out / "activity_goals.png")''')),
        ("Counts beyond a threshold are prompts to look, not diagnoses.",
         "Each reference names its source; plot_blood_pressure's guideline= chooses among ESC/ESH home, ESC/ESH office "
         "and ACC/AHA."),
        concepts=("clinical-thresholds", "valid-days"), aliases=("clinical workflow", "references")),
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
    argv = sys.argv[:]          # argparse names the program after sys.argv[0]: in a notebook, the kernel's launcher
    sys.argv[:] = ["python -m wearable_project.utils.data_statistics", *argv[1:]]
    try:
        with contextlib.redirect_stdout(buffer), contextlib.suppress(SystemExit):
            _module("data_statistics").main(["--help"])
    finally:
        sys.argv[:] = argv
    return buffer.getvalue().strip()


# ------------------------------------------------------------------------------------------------ rendering
def _wrap(text: str, indent: int = 2) -> list[str]:
    return textwrap.wrap(" ".join(_values(str(text)).split()), width=WIDTH, initial_indent=" " * indent,
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
    return InfoReport(kind=kind, title=title, payload=_filled(payload), text=text)


def _filled(value):
    """A payload with every current value filled in, like the text."""

    if isinstance(value, str):
        return _values(value) if "{" in value else value
    if isinstance(value, dict):
        return {k: _filled(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_filled(v) for v in value)
    return value


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
    payload["tables"] = _tool_tables(name)
    payload["concepts"] = _concepts_of("tool", name)
    payload["workflows"] = _workflows_calling(name)
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
    if p["tables"]:
        title = {"figure": "Data and panels", "class": "Tables it holds"}.get(p["kind"], "Tables it returns")
        sections.append((title, _tool_tables_lines(p)))
    if p["concepts"]:
        sections.append(("Concepts", _wrap(", ".join(f"{_values(CONCEPTS[c].title)} ({c})" for c in p["concepts"]))))
    if p["workflows"]:
        sections.append(("In workflows", _wrap(", ".join(f"{WORKFLOWS[w].title} ({w})" for w in p["workflows"]))))
    if p["related"]:
        sections.append(("Related", _wrap(", ".join(p["related"]))))
    if p["docstring"]:
        sections.append(("Documentation", _paragraphs(p["docstring"])))
    subtitle = f"{p['group_title']} · {kind_label} · {p['tier']}"
    return _document(f"{p['name']} — {p['module']}", subtitle, sections)


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _parameter_report(name: str) -> InfoReport:
    takers = []
    for tool, (module_name, obj) in sorted(tools().items()):
        if any(p.name == name for p in _parameters(obj)):
            meaning, source = _explain(tool, name)
            takers.append({"tool": tool, "module": module_name, "meaning": meaning if source == "tool" else None})
    figures = [t["tool"] for t in takers if _is_figure(t["tool"])]
    payload = {"name": name, "meaning": PARAMETERS[name], "tools": takers, "figures": figures}
    sections = [("Meaning", _wrap(PARAMETERS[name])),
                ("Taken by", _wrap(", ".join(t["tool"] for t in takers) + f" ({_count(len(takers), 'tool')}, "
                                                                         f"{_count(len(figures), 'figure')})."))]
    specific = [line for t in takers if t["meaning"] for line in _wrap(f"{t['tool']}: {t['meaning']}")]
    if specific:
        sections.append(("Where it means something more specific", specific))
    concepts = _concepts_of("parameter", name)
    payload["concepts"] = concepts
    if concepts:
        sections.append(("Concepts", _wrap(", ".join(f"{_values(CONCEPTS[c].title)} ({c})" for c in concepts))))
    if name in _all_columns():
        tables = _column_tables(name)
        payload["column"] = {"meaning": COLUMNS.get(name) or _pattern_meaning(name), "tables": tables}
        sections.append(("Also a column", _wrap(f"{payload['column']['meaning']} In {len(tables)} tables: "
                                                  f"{', '.join(tables)}.")))
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
    if section == "tables":
        return _tables_section_report(question, module)
    if section == "columns":
        return _columns_section_report()
    if section in ("concepts", "defaults", "workflows"):
        return _guide_section_report(section, question, module)
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
        ("Output tables", _wrap(f"The tools return {len(TABLES)} tables, from the statistics run to each figure's data and "
                                 f"panels. utils.info(\"tables\") lists them, utils.info(\"<table>\") explains every column "
                                 f"(such as utils.info(\"DomainMetrics.sleep_nights\")), and utils.info.table_columns() "
                                 f"gives a table's columns in a phase. A function's or figure's card lists the columns of "
                                 f"the tables it returns.")),
        ("Concepts, defaults and workflows", _wrap(
            f"{len(CONCEPTS)} concepts explain the ideas the tools rest on, with the current values of their rules: "
            f"utils.info(\"concepts\"), or one such as utils.info(\"valid days\"). utils.info(\"defaults\") gives "
            f"every threshold and rule ({len(DEFAULTS)}), read from the code, with its source and basis, and every "
            f"option's default. {len(WORKFLOWS)} workflows run a task from start to finish as code: "
            f"utils.info(\"workflows\"), or one such as utils.info(\"shareable report\").")),
        ("Conventions of the examples", _wrap(f"Every example assumes {SETUP_NOTES['base']}") + _code(payload["setup"], 4)),
        ("Asking", _wrap('utils.info("name") for a function, class, figure, parameter, question (such as "sleep"), '
                         'output table or column; utils.info("tools"), utils.info("figures"), utils.info("parameters"), '
                         'utils.info("tables") and utils.info("columns") for the catalogues, which question= and module= '
                         'filter; anything else searches. .as_dict() gives each report structured.')),
    ]
    subtitle = (f"{len(MODULES)} modules · {len(functions)} functions, {len(figures)} of them figures · "
                f"{len(classes)} classes · version {__version__}")
    return _report("overview", "wearable_project.utils", payload,
                   _document("wearable_project.utils — statistics and figures", subtitle, sections))



# ------------------------------------------------------------------------------------------ tables: machinery
def _daily_universe() -> list[str]:
    from wearable_project.utils import data_statistics as ds
    names = [*DAILY_COMMON, *DAILY_CURATED_ONLY]
    for extra in (*DAILY_BY_KIND.values(), *DAILY_BY_FEATURE.values(),
                  *([n for n, *_ in entries] for entries in ds.CLINICAL_THRESHOLDS.values())):
        names += [c for c in extra if c not in names]
    return names


def table_columns(table: str, phase: str = "curated", *, feature: str | None = None,
                  optional: bool = False) -> list[str]:
    """
    The documented columns of an output table in a phase.
    Parameters
    ----------
    table:
        A table, such as ``"DomainMetrics.sleep_nights"``; ``utils.info("tables")`` lists them.
    phase:
        ``"curated"`` or ``"native"``: some columns exist only in the curated phase.
    feature:
        For the daily tables, whose columns depend on the feature. Without it, only the columns every feature has.
    optional:
        Also the columns present only under a condition, such as ``ci_low`` with ``ci=``.
    """

    if table not in TABLES:
        close = difflib.get_close_matches(table, list(TABLES), n=3, cutoff=0.6)
        raise DataLoaderConfigurationError(f"no output table {table!r}" + (f"; did you mean {', '.join(close)}?" if close else ""))
    if phase not in ("curated", "native"):
        raise DataLoaderConfigurationError("phase must be 'curated' or 'native'")
    spec = TABLES[table]
    if table == "DailyStatistics.daily":
        columns = daily_columns(feature, phase) if feature else [*DAILY_COMMON, *(DAILY_CURATED_ONLY if phase == "curated" else ())]
    else:
        columns = table_columns(spec.extends, phase, feature=feature) if spec.extends else []
        if spec.derived_from:
            module, constant = spec.derived_from.split(".")
            columns += [c for c in getattr(_module(module), constant) if c not in columns]
        columns += [c for c in spec.columns if c not in columns]
        if phase == "curated":
            columns += [c for c in spec.curated_only if c not in columns]
    if optional:
        columns += [c for c, _ in spec.optional if c not in columns]
    return columns


def _column_meaning(table: str | None, column: str) -> str:
    key = table
    while key:
        notes = dict(TABLES[key].notes)
        if column in notes:
            return notes[column]
        key = TABLES[key].extends or None
    return COLUMNS.get(column) or _pattern_meaning(column)


def _tables_of(tool: str) -> list[str]:
    return [key for key, spec in TABLES.items() if tool in spec.returned_by]


def _table_group(key: str) -> str:
    """The question a table answers: that of the first tool returning it outside runs and inputs."""

    groups = [TOOLS[tool].group for tool in TABLES[key].returned_by]
    return next((g for g in groups if g != "runs"), groups[0])


def _column_rows(key: str) -> list[dict[str, str]]:
    spec, native = TABLES[key], set(table_columns(key, "native"))
    rows = [{"name": c, "meaning": _column_meaning(key, c), "when": "" if c in native else "curated phase only"}
            for c in table_columns(key, "curated")]
    whens: dict[str, list[str]] = {}
    for column, when in spec.optional:
        whens.setdefault(column, []).append(when)
    listed = {r["name"] for r in rows}
    rows += [{"name": c, "meaning": _column_meaning(key, c), "when": "; ".join(w)} for c, w in whens.items()
             if c not in listed]
    return rows


def _table_payload(key: str) -> dict[str, Any]:
    spec = TABLES[key]
    payload: dict[str, Any] = {
        "table": key, "rows": spec.rows, "returned_by": list(spec.returned_by), "group": _table_group(key),
        "group_title": _group_title(_table_group(key)), "derived_from": spec.derived_from, "extends": spec.extends,
        "index": list(spec.index), "columns": _column_rows(key), "dynamic": spec.dynamic, "read_back": spec.read_back,
    }
    if key == "DailyStatistics.daily":
        from wearable_project.utils import data_statistics as ds
        extras = {f: list(c) for f, c in DAILY_BY_FEATURE.items()}
        for feature, entries in ds.CLINICAL_THRESHOLDS.items():
            extras.setdefault(feature, []).extend(n for n, *_ in entries)
        payload["by_kind"] = {k: list(c) for k, c in DAILY_BY_KIND.items()}
        payload["by_feature"] = extras
        payload["kind_columns"] = [{"name": c, "meaning": _column_meaning(key, c)} for c in _daily_universe()
                                   if c not in DAILY_COMMON and c not in DAILY_CURATED_ONLY]
    return payload


def _column_lines(rows: list[dict[str, str]], indent: int = 2) -> list[str]:
    lines = []
    for r in rows:
        when = f" [{r['when']}]" if r.get("when") else ""
        lines.extend(_wrap(f"{r['name']}{when} — {r['meaning']}", indent))
    return lines


def _table_text(p: dict[str, Any]) -> str:
    sections: list[tuple[str, list[str]]] = [("Rows", _wrap(p["rows"][0].upper() + p["rows"][1:] + "."))]
    sections.append(("Returned by", _wrap(", ".join(p["returned_by"]))))
    if p["index"]:
        sections.append(("Index", _wrap(", ".join(p["index"]))))
    source = f" (from {p['derived_from']})" if p["derived_from"] else ""
    if p["extends"]:
        source = f" (those of {p['extends']}" + (", then these)" if TABLES[p["table"]].columns else ")")
    if p["table"] == "DailyStatistics.daily":
        sections.append(("Every feature, both phases", _column_lines([r for r in p["columns"] if not r["when"]])))
        sections.append(("Curated phase only", _column_lines([{**r, "when": ""} for r in p["columns"] if r["when"]])))
        sections.append(("By measurement kind", [line for k, c in p["by_kind"].items()
                                                  for line in _wrap(f"{k}: {', '.join(c) or 'none beyond the above'}")]))
        sections.append(("Only some features", [line for f, c in p["by_feature"].items() for line in _wrap(f"{f}: {', '.join(c)}")]))
        sections.append(("What those columns mean", _column_lines(p["kind_columns"])))
    else:
        sections.append((f"Columns{source}", _column_lines(p["columns"])))
    if p["dynamic"]:
        title = "A given feature's columns" if p["table"] == "DailyStatistics.daily" else "Columns named at run time"
        sections.append((title, _wrap(p["dynamic"])))
    if p["read_back"]:
        directory = ("domain_dir, the out= directory of compute_domain_metrics" if "domain_dir" in p["read_back"]
                     else "run_dir, the out= directory of compute_daily_statistics")
        sections.append(("In a written run", _wrap(f"{p['read_back']}, with {directory}.")))
    if p["table"] == "DailyStatistics.daily":
        count = "columns by feature"
    elif p["dynamic"]:
        count = f"{len(p['columns'])} columns and more named at run time" if p["columns"] else "columns named at run time"
    else:
        count = f"{len(p['columns'])} columns"
    return _document(f"{p['table']} — output table", f"{p['group_title']} · {count}", sections)


def _table_report(key: str) -> InfoReport:
    payload = _table_payload(key)
    return _report("table", key, payload, _table_text(payload))


def _column_tables(name: str) -> list[str]:
    found = []
    for key in TABLES:
        names = _daily_universe() if key == "DailyStatistics.daily" else table_columns(key, "curated", optional=True)
        if name in names:
            found.append(key)
    return found


def _all_columns() -> list[str]:
    names: list[str] = []
    for key in TABLES:
        for c in (_daily_universe() if key == "DailyStatistics.daily" else table_columns(key, "curated", optional=True)):
            if c not in names:
                names.append(c)
    return names


def _column_report(name: str) -> InfoReport:
    tables = _column_tables(name)
    general = COLUMNS.get(name) or _pattern_meaning(name)
    specific = [{"table": t, "meaning": _column_meaning(t, name)} for t in tables if _column_meaning(t, name) != general]
    body = _wrap(general)
    if specific:
        body += [""] + [line for s in specific for line in _wrap(f"In {s['table']}: {s['meaning']}")]
    payload = {"column": name, "meaning": general, "tables": tables, "specific": specific}
    text = _document(f"{name} — column", f"in {len(tables)} tables",
                     [("Meaning", body), ("Tables", _wrap(", ".join(tables)))])
    return _report("column", name, payload, text)


def _tool_tables(name: str) -> list[dict[str, Any]]:
    keys = _tables_of(name) + [k for k in TABLES if k.startswith(f"{name}.") and name not in TABLES[k].returned_by]
    return [{"table": k, "rows": TABLES[k].rows, "columns": _column_rows(k), "dynamic": TABLES[k].dynamic,
             "index": list(TABLES[k].index)} for k in keys]


def _tool_tables_lines(p: dict[str, Any]) -> list[str]:
    body: list[str] = []
    detailed = p["kind"] == "figure" or len(p["tables"]) <= 2
    for t in p["tables"]:
        body.extend(_wrap(f"{t['table']} — {t['rows']}"))
        if t["table"] == "DailyStatistics.daily":
            body.extend(_wrap("Columns depend on the feature and the phase: utils.info.daily_columns(feature, phase) "
                              "lists them.", 4))
        elif detailed:
            body.extend(_column_lines(t["columns"], 4))
        else:
            body.extend(_wrap(f"Columns: {', '.join(r['name'] for r in t['columns'])}", 4))
        if t["dynamic"]:
            body.extend(_wrap(t["dynamic"], 4))
    if p["tables"] and not detailed:
        body.extend(_wrap("utils.info('<table>') explains every column of a table, such as "
                          f"utils.info({p['tables'][0]['table']!r})."))
    return body


def _tables_section_report(question: str | None, module: str | None) -> InfoReport:
    group, module_name = _resolve_group(question), _resolve_module(module)
    keys = [k for k in TABLES if (group is None or _table_group(k) == group)
            and (module_name is None or any(TOOLS[t].module == module_name for t in TABLES[k].returned_by))]
    sections = []
    for g, title, _ in GROUPS:
        chosen = [k for k in keys if _table_group(k) == g]
        if chosen:
            sections.append((title, [line for k in chosen for line in _wrap(f"{k} — {TABLES[k].rows}")]))
    rows = [{"table": k, "rows": TABLES[k].rows, "group": _table_group(k), "returned_by": list(TABLES[k].returned_by)}
            for k in keys]
    return _report("tables", "Tables", {"tables": rows},
                   _document("Output tables", f"{len(keys)} tables; utils.info('<table>') explains its columns", sections))


def _columns_section_report() -> InfoReport:
    names = sorted(_all_columns(), key=str.casefold)
    rows = [{"column": n, "meaning": COLUMNS.get(n) or _pattern_meaning(n), "tables": len(_column_tables(n))} for n in names]
    body = [line for r in rows for line in _wrap(f"{r['column']} ({r['tables']} tables) — {r['meaning']}")]
    return _report("columns", "Columns", {"columns": rows},
                   _document("Columns", f"{len(rows)} column names across the output tables", [("Columns", body)]))


# ------------------------------------------------------------------------- concepts, defaults, workflows: machinery
_PLACEHOLDER = re.compile(r"\{d:([a-z_]+)\}")
_OPTION = re.compile(r"\{o:([a-z_]+)\}")
_CONSTANT = re.compile(r"\{c:([A-Za-z_][A-Za-z0-9_.]*)\}")


def _live(source: str):
    """The current value of a default's source: a module constant, or a dataclass field's default."""

    module_name, *path = source.split(".")
    obj: Any = _module(module_name)
    for i, name in enumerate(path):
        if inspect.isclass(obj) and dataclasses.is_dataclass(obj) and i == len(path) - 1:
            return {f.name: f.default for f in dataclasses.fields(obj)}[name]
        obj = getattr(obj, name)
    return obj


def _inline(key: str) -> str:
    spec = DEFAULTS[key]
    return spec.inline(*(_live(s) for s in spec.sources))


def _values(text: str) -> str:
    """Text with current values: {d:key} a default in words, {c:CONSTANT} a constant's value, {o:parameter} its defaults."""

    text = _PLACEHOLDER.sub(lambda m: _inline(m.group(1)), text)
    text = _CONSTANT.sub(lambda m: _constant_value(m.group(1)), text)
    return _OPTION.sub(lambda m: _option_inline(m.group(1)), text)


def _constant_value(name: str) -> str:
    """A constant's current value, by the name it answers to (CONSTANT, module.CONSTANT or alias.CONSTANT)."""

    key = _default_by_constant()[name]
    constant = name.split(".")[-1] if name.count(".") == 1 and name.split(".")[0] in {**MODULES, **{a: 0 for a, *_ in MODULES.values()}} else name
    return _short(_live(next(s for s in DEFAULTS[key].sources if s.split(".", 1)[1] == constant)))


def _option_inline(name: str) -> str:
    """A parameter's defaults, from the signatures: the usual value, then each exception with its tools."""

    row = next(r for r in _option_defaults() if r["parameter"] == name)
    usual, *others = sorted(row["defaults"], key=lambda d: -len(d["tools"]))
    if not others:
        return usual["value"]
    return usual["value"] + " everywhere except " + _and(f"{', '.join(d['tools'])} ({d['value']})" for d in others)


def _short(value) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_short(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (frozenset, set)):
        return ", ".join(sorted(map(str, value)))
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(_short(v) for v in value) + ")"
    if isinstance(value, float):
        return _num(value)
    return str(value)


def _detail(value) -> list[str]:
    if isinstance(value, dict):
        return [f"{k}: {_short(v)}" for k, v in value.items()]
    if isinstance(value, (tuple, list)) and value and all(isinstance(v, (tuple, list)) for v in value):
        return [", ".join(_short(x) for x in v) for v in value]
    return [_short(value)]


def _constant_names(key: str) -> list[str]:
    return [source.split(".", 1)[1] for source in DEFAULTS[key].sources]


def _default_by_constant() -> dict[str, str]:
    """Every name a default answers to: CONSTANT, module.CONSTANT and alias.CONSTANT."""

    names: dict[str, str] = {}
    for key, spec in DEFAULTS.items():
        for source in spec.sources:
            module_name, constant = source.split(".", 1)
            for name in (constant, source, f"{MODULES[module_name][0]}.{constant}"):
                names[name] = key
    return names


def _default_payload(key: str) -> dict[str, Any]:
    spec = DEFAULTS[key]
    return {"default": key, "title": spec.title, "group": spec.group, "group_title": _group_title(spec.group),
            "value": _inline(key), "sources": list(spec.sources),
            "values": {source: _short(_live(source)) for source in spec.sources},
            "detail": {source: _detail(_live(source)) for source in spec.sources},
            "meaning": spec.meaning, "basis": spec.basis, "used_by": list(spec.used_by), "change": spec.change,
            "concepts": [c for c, s in CONCEPTS.items() if key in s.defaults]}


def _default_report(key: str) -> InfoReport:
    p = _default_payload(key)
    detail = [line for source, lines in p["detail"].items()
              for line in (_wrap(f"{source}:") + [l for x in lines for l in _wrap(x, 4)])]
    sections = [("Value", _wrap(p["value"])), ("In detail", detail), ("What it governs", _wrap(p["meaning"])),
                ("Why this value", _wrap(p["basis"])), ("Used by", _wrap(", ".join(p["used_by"]))),
                ("To use another value", _wrap(p["change"]))]
    if p["concepts"]:
        sections.append(("Concepts", _wrap(", ".join(f"{CONCEPTS[c].title} ({c})" for c in p["concepts"]))))
    return _report("default", _constant_names(key)[0], p,
                   _document(f"{p['title']} — default", f"{p['group_title']} · read from the code", sections))


def _concept_payload(key: str) -> dict[str, Any]:
    spec = CONCEPTS[key]
    return {"concept": key, "title": _values(spec.title), "group": spec.group, "group_title": _group_title(spec.group),
            "summary": _values(spec.summary), "body": [_values(p) for p in spec.body],
            "values": [{"default": d, "title": DEFAULTS[d].title, "value": _inline(d), "constants": _constant_names(d)}
                       for d in spec.defaults],
            "tools": list(spec.tools), "parameters": list(spec.parameters), "columns": list(spec.columns),
            "related": list(spec.related), "aliases": list(spec.aliases),
            "workflows": [w for w, s in WORKFLOWS.items() if key in s.concepts]}


def _concept_report(key: str) -> InfoReport:
    p = _concept_payload(key)
    sections = [("In short", _wrap(p["summary"])), ("Explanation", [l for para in p["body"] for l in _wrap(para) + [""]][:-1])]
    if p["values"]:
        sections.append(("Current values", [line for v in p["values"] for line in
                                            _wrap(f"{v['title']}: {v['value']} (utils.info({v['constants'][0]!r}))")]))
    for title, names in (("Tools", p["tools"]), ("Parameters", p["parameters"]), ("Columns", p["columns"])):
        if names:
            sections.append((title, _wrap(", ".join(names))))
    if p["related"]:
        sections.append(("Related concepts", _wrap(", ".join(f"{CONCEPTS[c].title} ({c})" for c in p["related"]))))
    if p["workflows"]:
        sections.append(("Workflows", _wrap(", ".join(f"{WORKFLOWS[w].title} ({w})" for w in p["workflows"]))))
    return _report("concept", key, p, _document(f"{p['title']} — concept", p["group_title"], sections))


def _workflow_payload(key: str, phase: str, root, out) -> dict[str, Any]:
    spec = WORKFLOWS[key]
    return {"workflow": key, "title": spec.title, "group": spec.group, "group_title": _group_title(spec.group),
            "question": spec.question, "phases": list(spec.phases), "aliases": list(spec.aliases),
            "steps": [{"does": does, "code": _fill(code, phase, root, out)} for does, code in spec.steps],
            "reading": list(spec.reading), "concepts": list(spec.concepts)}


def _workflow_report(key: str, phase: str, root, out) -> InfoReport:
    p = _workflow_payload(key, phase, root, out)
    steps: list[str] = []
    for n, step in enumerate(p["steps"], 1):
        steps.extend(_wrap(f"{n}. {step['does']}") + _code(step["code"], 4) + [""])
    sections = [("The question", _wrap(p["question"])), ("Steps", steps[:-1]),
                ("Reading the results", [l for para in p["reading"] for l in _wrap(para) + [""]][:-1]),
                ("Concepts", _wrap(", ".join(f"{CONCEPTS[c].title} ({c})" for c in p["concepts"]))),
                ("Runs in", _wrap(" and ".join(f"the {ph} phase" for ph in p["phases"])
                                  + ("" if len(p["phases"]) > 1 else " only")))]
    return _report("workflow", key, p, _document(f"{p['title']} — workflow", p["group_title"], sections))


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _guide_names(kind: str, key: str) -> set[str]:
    spec = CONCEPTS[key] if kind == "concept" else WORKFLOWS[key]
    names = {_norm(key), _norm(_values(spec.title)), *(_norm(a) for a in spec.aliases)}
    return names | {n[:-1] for n in names if n.endswith("s") and len(n) > 3}


def _guide_index() -> dict[str, tuple[str, str]]:
    """Every name a concept or workflow answers to, normalized: its key, title and aliases, singular or plural."""

    index: dict[str, tuple[str, str]] = {}
    for kind, catalogue in (("concept", CONCEPTS), ("workflow", WORKFLOWS)):
        for key in catalogue:
            for name in _guide_names(kind, key):
                index.setdefault(name, (kind, key))
    return index


def _find_guide(topic: str) -> tuple[str, str] | None:
    index, name = _guide_index(), _norm(topic)
    return index.get(name) or (index.get(name[:-1]) if name.endswith("s") else None)


def _tool_modules(names) -> set[str]:
    return {TOOLS[n].module for n in names if n in TOOLS}


def _guide_section_report(section: str, question: str | None, module: str | None) -> InfoReport:
    group, module_name = _resolve_group(question), _resolve_module(module)
    if section == "defaults":
        keys = [k for k, s in DEFAULTS.items() if (group is None or s.group == group)
                and (module_name is None or any(src.split(".")[0] == module_name for src in s.sources))]
        line = lambda k: f"{DEFAULTS[k].title}: {_inline(k)} [{', '.join(_constant_names(k))}]"
        rows = [_default_payload(k) for k in keys]
        options = _option_defaults() if group is None and module_name is None else []
        extra = [("Option defaults", [l for o in options for l in _wrap(
            f"{o['parameter']}: " + "; ".join(f"{d['value']} ({_count(len(d['tools']), 'tool')})" if len(d["tools"]) > 3
                                              else f"{d['value']} ({', '.join(d['tools'])})" for d in o["defaults"]))])] \
            if options else []
        title, subtitle = "Defaults", f"{len(keys)} defaults, read from the code; utils.info(\"<CONSTANT>\") explains one"
        payload: dict[str, Any] = {"defaults": rows, "options": options}
    elif section == "concepts":
        keys = [k for k, s in CONCEPTS.items() if (group is None or s.group == group)
                and (module_name is None or module_name in _tool_modules(s.tools))]
        line = lambda k: f"{_values(CONCEPTS[k].title)} ({k}) — {_values(CONCEPTS[k].summary)}"
        title, subtitle, extra = "Concepts", f"{len(keys)} concepts; utils.info(\"<concept>\") explains one", []
        payload = {"concepts": [{"concept": k, "title": _values(CONCEPTS[k].title), "group": CONCEPTS[k].group,
                                 "summary": _values(CONCEPTS[k].summary)} for k in keys]}
    else:
        keys = [k for k, s in WORKFLOWS.items() if (group is None or s.group == group)
                and (module_name is None or re.search(rf"\b{MODULES[module_name][0]}\.", "".join(c for _, c in s.steps)))]
        line = lambda k: f"{WORKFLOWS[k].title} ({k}) — {WORKFLOWS[k].question}"
        title, subtitle, extra = "Workflows", f"{len(keys)} workflows; utils.info(\"<workflow>\") runs one as code", []
        payload = {"workflows": [{"workflow": k, "title": WORKFLOWS[k].title, "group": WORKFLOWS[k].group,
                                  "question": WORKFLOWS[k].question, "phases": list(WORKFLOWS[k].phases)} for k in keys]}
    catalogue = CONCEPTS if section == "concepts" else WORKFLOWS if section == "workflows" else DEFAULTS
    sections = [(t, [l for k in keys if catalogue[k].group == g for l in _wrap(line(k))]) for g, t, _ in GROUPS
                if any(catalogue[k].group == g for k in keys)]
    return _report(section, title, payload, _document(title, subtitle, sections + extra))


@functools.lru_cache(maxsize=1)
def _option_defaults() -> list[dict[str, Any]]:
    """Every parameter's default, from the signatures: each distinct value and the tools that use it."""

    rows = []
    for name in sorted(PARAMETERS):
        values: dict[str, list[str]] = {}
        for tool, (_, obj) in tools().items():
            if inspect.isclass(obj):
                continue
            for p in _parameters(obj):
                if p.name == name and p.default is not p.empty:
                    values.setdefault(_default(p.default), []).append(tool)
        if values:
            rows.append({"parameter": name, "defaults": [{"value": v, "tools": sorted(t)} for v, t in values.items()]})
    return rows


def _concepts_of(kind: str, name: str) -> list[str]:
    field = {"tool": "tools", "parameter": "parameters"}[kind]
    return [k for k, s in CONCEPTS.items() if name in getattr(s, field)]


def _workflows_calling(tool: str) -> list[str]:
    pattern = re.compile(rf"\b(ds|sm|dm|dp)\.{re.escape(tool)}\(")
    return [k for k, s in WORKFLOWS.items() if any(pattern.search(code) for _, code in s.steps)]

# ---------------------------------------------------------------------------------------------------- search
def search(term: str, limit: int = 15) -> list[dict[str, Any]]:
    """Topics matching ``term``, best first: tools, parameters, questions, tables, columns, concepts, workflows, defaults."""

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
    entries += [("table", k, " ".join([s.rows, *(_daily_universe() if k == "DailyStatistics.daily"
                                                 else table_columns(k, "curated", optional=True))]), "")
                for k, s in TABLES.items()]
    entries += [("column", c, COLUMNS.get(c) or _pattern_meaning(c), "") for c in _all_columns()]
    entries += [("concept", k, " ".join([_values(s.title), _values(s.summary), *map(_values, s.body), *s.aliases]), "")
                for k, s in CONCEPTS.items()]
    entries += [("workflow", k, " ".join([s.title, s.question, *s.reading, *(d for d, _ in s.steps), *s.aliases]), "")
                for k, s in WORKFLOWS.items()]
    entries += [("default", _constant_names(k)[0], " ".join([s.title, s.meaning, s.basis, *_constant_names(k)]), "")
                for k, s in DEFAULTS.items()]
    entries = [(kind, name, _values(text), extra) for kind, name, text, extra in entries]
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
                            "summary": _search_summary(kind, name, text)})
    return sorted(results, key=lambda r: (-r["score"], r["topic"]))[:limit]


def _search_summary(kind: str, name: str, text: str) -> str:
    if kind == "tool":
        return _values(TOOLS[name].does)
    if kind == "table":
        return _values(TABLES[name].rows)
    if kind == "concept":
        return _values(CONCEPTS[name].summary)
    if kind == "workflow":
        return WORKFLOWS[name].question
    if kind == "default":
        key = _default_by_constant()[name]
        return f"{DEFAULTS[key].title}: {_inline(key)}"
    return _values(text)


def _search_report(term: str) -> InfoReport:
    results = search(term)
    body = [line for r in results for line in _wrap(f"{r['topic']} ({r['kind']}) — {r['summary']}")]
    return _report("search", term, {"term": term, "results": results},
                   _document(f"Search: {term}", f"{len(results)} topics, best first", [("Topics", body)]))


# ------------------------------------------------------------------------------------------------------ info
SECTIONS = ("overview", "tools", "figures", "parameters", "tables", "columns", "concepts", "defaults", "workflows")


def info(topic: str | None = None, *, question: str | None = None, module: str | None = None,
         phase: str = "curated", root: str | Path | None = None, out: str | Path | None = None) -> InfoReport:
    """
    Help on the statistics and figure tools.
    Parameters
    ----------
    topic:
        None for the overview; a function, class or figure name (``"plot_agp"``, also ``"dp.plot_agp"``); a parameter
        (``"min_participants"``); a question (``"sleep"``); an output table (``"DomainMetrics.sleep_nights"``,
        ``"plot_agp.data"``) or a column (``"gmi_percent"``); a concept (``"valid days"``), a workflow
        (``"shareable report"``) or a default by its constant (``"CGM_SUFFICIENT_DAYS"``); ``"tools"``, ``"figures"``,
        ``"parameters"``, ``"tables"``, ``"columns"``, ``"concepts"``, ``"defaults"`` or ``"workflows"`` for the
        catalogues. Anything else searches. Matching ignores case, except for constants.
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
    constants = _default_by_constant()
    if key in constants:          # exact constant names first: STEP_CATEGORIES is a default, step_categories a tool
        return _default_report(constants[key])
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
    by_table = {k.casefold(): k for k in TABLES}
    if key.casefold() in by_table:
        return _table_report(by_table[key.casefold()])
    by_column = {c.casefold(): c for c in _all_columns()}
    if folded in by_column:
        return _column_report(by_column[folded])
    found = _find_guide(key)                  # concepts and workflows, by key, title or alias
    if found is not None:
        kind, name = found
        return _concept_report(name) if kind == "concept" else _workflow_report(name, phase, root, out)
    if search(key):
        return _search_report(key)
    candidates = [*TOOLS, *PARAMETERS, *(k for k, _, _ in GROUPS), *TABLES, *CONCEPTS, *WORKFLOWS, *_default_by_constant()]
    close = difflib.get_close_matches(folded, [c.casefold() for c in candidates], n=5, cutoff=0.6)
    hint = f"; did you mean {', '.join(close)}?" if close else "; utils.info() lists every topic"
    raise DataLoaderConfigurationError(f"no topic or match for {topic!r}{hint}")


def topics() -> list[str]:
    """Every topic ``info`` answers by name."""

    named = [*SECTIONS, *TOOLS, *PARAMETERS, *(k for k, _, _ in GROUPS), *TABLES]
    taken = {n.casefold() for n in named}
    constants = [c for c in _default_by_constant() if "." not in c]
    columns = [c for c in _all_columns() if c.casefold() not in taken]
    return named + constants + columns + [*CONCEPTS, *WORKFLOWS]


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
