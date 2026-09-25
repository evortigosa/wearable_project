"""
Wearable Data Processing and Modeling project
Figures from the statistics modules: coverage, activity over time, daily patterns, values, adherence, context,
sleep, glucose and BMI.
Every function draws from the output of ``data_statistics``, ``data_summaries`` or ``domain_metrics``, never from raw
files, so each figure inherits their rules: local calendar days, values combined as the feature's measurement kind
requires, harmonized units, valid days, time counted once across devices. Each returns a matplotlib ``Figure`` built
with the object-oriented API: nothing is shown, no global state is touched, and it works without a display. Pass
``path=`` to save it; the format follows the file's suffix.
Every figure carries the exact table it draws as ``figure.data``, so the numbers behind any plot can be exported,
checked or re-plotted. The axes are honest: every histogram bin is drawn, nothing is clipped unless asked (and then
the number of clipped values is stated on the figure), units come from the statistics tables, and each figure states
how many participants it describes. Participant identifiers never appear on a figure unless ``show_ids=True``.
matplotlib is an optional dependency: ``pip install "wearable_project[plots]"``.
"""


from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from wearable_project.exceptions import DataLoaderConfigurationError
from wearable_project.utils import data_statistics as ds
from wearable_project.utils import data_summaries as sm
from wearable_project.utils import domain_metrics as dm


WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
# Data volume bands, in bytes, for the size distribution.
VOLUME_BANDS = ((0, 1e6, "< 1 MB"), (1e6, 1e7, "1-10 MB"), (1e7, 1e8, "10-100 MB"), (1e8, 1e9, "100 MB-1 GB"),
                (1e9, np.inf, ">= 1 GB"))
# WHO adult BMI classes, kg/m2, each a half-open interval [lower, upper).
BMI_CLASSES = (("Underweight", 0.0, 18.5), ("Normal weight", 18.5, 25.0), ("Pre-obesity", 25.0, 30.0),
               ("Obesity class I", 30.0, 35.0), ("Obesity class II", 35.0, 40.0), ("Obesity class III", 40.0, np.inf))
# Colours for the consensus glucose ranges, from very low to very high.
GLUCOSE_COLOURS = {"very_low_percent": "#8e0000", "low_percent": "#e53935", "in_range_percent": "#43a047",
                   "high_percent": "#fdd835", "very_high_percent": "#fb8c00"}
GLUCOSE_LABELS = {"very_low_percent": "< 54 mg/dL", "low_percent": "54-69 mg/dL", "in_range_percent": "70-180 mg/dL",
                  "high_percent": "181-250 mg/dL", "very_high_percent": "> 250 mg/dL"}
# Okabe-Ito, a palette distinguishable with the common colour-vision deficiencies.
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000", "#999999", "#882255")
TIME_IN_RANGE_TARGET = 70.0  # consensus target for most adults with diabetes, % of readings in 70-180 mg/dL


# ------------------------------------------------------------------------------------------------ helpers
def _figure(width: float = 9.0, height: float = 5.0, nrows: int = 1, ncols: int = 1):
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise DataLoaderConfigurationError('figures need matplotlib: pip install "wearable_project[plots]"') from exc
    figure = Figure(figsize=(width, height), layout="constrained")
    return figure, figure.subplots(nrows, ncols, squeeze=False)


def _finish(figure, data: pd.DataFrame, path: str | Path | None, dpi: int) -> Any:
    figure.data = data.reset_index(drop=True)
    if path is not None:
        figure.savefig(Path(path).expanduser(), dpi=dpi)
    return figure


def _titles(ax, title: str, caption: str) -> None:
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_title(caption, loc="right", fontsize=8, color="0.4")


def _participants(n: int) -> str:
    return f"{n:,} participant{'s' if n != 1 else ''}"


def _phase(source) -> str | None:
    if isinstance(source, ds.DailyStatistics):
        return source.phase
    run = Path(source) / "run_parameters.json"
    if run.is_file():
        import json
        return json.loads(run.read_text()).get("phase")
    return None


def _caption(source, n: int, extra: str = "") -> str:
    phase = _phase(source) if not isinstance(source, (ds.Coverage,)) else source.phase
    parts = [p for p in (f"{phase} phase" if phase else None, _participants(n), extra) if p]
    return " · ".join(parts)


def _labels(codes: Iterable[str], show_ids: bool) -> list[str]:
    codes = list(codes)
    return codes if show_ids else [f"Participant {i + 1}" for i in range(len(codes))]


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise DataLoaderConfigurationError(message)


def _metric(feature: str, columns: Iterable[str], metric: str | None) -> str:
    """The feature's headline metric (``data_summaries.headline_metrics``) unless one is named."""

    available = sm.headline_metrics(feature, columns)
    if metric is None:
        _need(bool(available), f"{feature} has no metric to plot")
        return available[0]
    _need(metric in set(columns), f"{metric!r} is not a daily column of {feature}")
    return metric


def _unit(feature: str, metric: str, frame: pd.DataFrame) -> str:
    if metric in sm._FIXED_UNITS:
        return sm._FIXED_UNITS[metric]
    unit = sm._unit_of(metric, frame, ds.column_units(feature))
    return unit or "unit not established"


# ----------------------------------------------------------------------------------------------- coverage
def plot_feature_presence(coverage: "ds.Coverage", *, show_ids: bool = False, path=None, dpi: int = 150):
    """Which participant has which feature: participants (rows, most features first) by features (columns)."""

    presence = coverage.presence()
    _need(not presence.empty, "the coverage holds no participant")
    presence = presence[presence.sum().sort_values(ascending=False, kind="stable").index]
    order = presence.assign(_n=presence.sum(axis=1)).sort_values(["_n"], ascending=False, kind="stable").index
    presence = presence.loc[order]
    from matplotlib.colors import ListedColormap
    fig, axes = _figure(max(8.0, 0.28 * presence.shape[1] + 3), max(4.0, min(12.0, 0.18 * presence.shape[0] + 2)))
    ax = axes[0, 0]
    ax.imshow(presence.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap=ListedColormap(["#f2f2f2", "#1f4e79"]), vmin=0, vmax=1)
    ax.set_xticks(range(presence.shape[1]))
    ax.set_xticklabels(presence.columns, rotation=90, fontsize=8)
    ax.set_ylabel("participants, most features first")
    if show_ids and presence.shape[0] <= 60:
        ax.set_yticks(range(presence.shape[0]))
        ax.set_yticklabels(presence.index, fontsize=7)
    else:
        ax.set_yticks([])
    _titles(ax, "Features present per participant", _caption(coverage, presence.shape[0]))
    data = presence.reset_index().melt(id_vars="RegistrationCode", var_name="feature", value_name="present")
    return _finish(fig, data, path, dpi)


def plot_participants_per_feature(coverage: "ds.Coverage", *, path=None, dpi: int = 150):
    """How many participants have each feature, most common first, with the share of the cohort."""

    table = coverage.features[["participants"]].reset_index()
    table = table[table["participants"] > 0].sort_values("participants", ascending=True, kind="stable")
    _need(not table.empty, "the coverage holds no participant")
    cohort = len(coverage.participants)
    table["share"] = table["participants"] / cohort
    fig, axes = _figure(8.0, max(3.5, 0.3 * len(table) + 1.5))
    ax = axes[0, 0]
    bars = ax.barh(table["feature"], table["participants"], color="#1f4e79")
    ax.bar_label(bars, labels=[f"{n:,} ({s:.0%})" for n, s in zip(table["participants"], table["share"])], fontsize=7, padding=2)
    ax.set_xlabel("participants")
    ax.set_xlim(0, table["participants"].max() * 1.18)
    _titles(ax, "Participants with each feature", _caption(coverage, cohort))
    return _finish(fig, table.iloc[::-1], path, dpi)


def plot_features_per_participant(coverage: "ds.Coverage", *, path=None, dpi: int = 150):
    """How many features participants have: every count from 1 to the maximum is drawn, including empty ones."""

    counts = coverage.participants["features"]
    _need(len(counts) > 0, "the coverage holds no participant")
    table = pd.DataFrame({"features": range(1, int(counts.max()) + 1)})
    table["participants"] = table["features"].map(counts.value_counts()).fillna(0).astype(int)
    fig, axes = _figure(8.0, 4.5)
    ax = axes[0, 0]
    bars = ax.bar(table["features"], table["participants"], color="#1f4e79")
    ax.bar_label(bars, fontsize=7)
    ax.set_xticks(table["features"])
    ax.set_xlabel("number of features")
    ax.set_ylabel("participants")
    _titles(ax, "Features per participant", _caption(coverage, len(counts)))
    return _finish(fig, table, path, dpi)


def plot_data_volume(coverage: "ds.Coverage", *, by: str = "participant", path=None, dpi: int = 150):
    """
    Stored data per participant (a histogram on a logarithmic axis, with the share of participants in each size
    band) or per feature (total bytes). Sizes are the files' bytes on disk.
    """

    _need(by in ("participant", "feature"), "by must be 'participant' or 'feature'")
    fig, axes = _figure(9.0, 4.8)
    ax = axes[0, 0]
    if by == "feature":
        table = coverage.features[["bytes_on_disk"]].reset_index().dropna()
        table = table[table["bytes_on_disk"] > 0].sort_values("bytes_on_disk", kind="stable")
        _need(not table.empty, "the coverage records no file size")
        ax.barh(table["feature"], table["bytes_on_disk"] / 1e6, color="#1f4e79")
        ax.set_xscale("log")
        ax.set_xlabel("MB on disk (log scale)")
        _titles(ax, "Stored data per feature", _caption(coverage, len(coverage.participants)))
        return _finish(fig, table.iloc[::-1], path, dpi)
    sizes = coverage.participants["bytes_on_disk"].dropna().astype(float)
    sizes = sizes[sizes > 0]
    _need(len(sizes) > 0, "the coverage records no file size")
    table = sizes.rename("bytes").reset_index()
    table["band"] = pd.cut(table["bytes"], [b[0] for b in VOLUME_BANDS] + [np.inf], right=False,
                           labels=[b[2] for b in VOLUME_BANDS]).astype(str)
    edges = np.logspace(np.floor(np.log10(sizes.min())), np.ceil(np.log10(sizes.max())) + 1e-9, 30)
    ax.hist(sizes, bins=edges, color="#1f4e79")
    ax.set_xscale("log")
    for low, high, label in VOLUME_BANDS:
        if low > 0 and sizes.min() <= low <= sizes.max():
            ax.axvline(low, color="0.5", linestyle=":", linewidth=1)
    shares = table["band"].value_counts(normalize=True)
    ax.text(0.99, 0.97, "\n".join(f"{label}: {shares.get(label, 0):.0%}" for _, _, label in VOLUME_BANDS),
            transform=ax.transAxes, ha="right", va="top", fontsize=8, family="monospace")
    ax.set_xlabel("bytes on disk per participant (log scale)")
    ax.set_ylabel("participants")
    _titles(ax, "Stored data per participant", _caption(coverage, len(sizes)))
    return _finish(fig, table, path, dpi)


# -------------------------------------------------------------------------------------- activity over time
def plot_active_participants(source, *, rolling: int | None = None, path=None, dpi: int = 150):
    """Participants with any data on each local day; ``rolling`` adds a centred mean over that many days."""

    active = source.active_participants if isinstance(source, ds.DailyStatistics) else ds.read_table(source, "active_participants")
    _need(not active.empty, "the run holds no active day")
    table = active.assign(local_date=pd.to_datetime(active["local_date"])).sort_values("local_date")
    fig, axes = _figure(10.0, 4.5)
    ax = axes[0, 0]
    ax.plot(table["local_date"], table["participants"], color="#9ecae1" if rolling else "#1f4e79", linewidth=0.8, label="per day")
    if rolling:
        _need(int(rolling) >= 2, "rolling must be at least 2 days")
        table["rolling_mean"] = table.set_index("local_date")["participants"].asfreq("D", fill_value=0).rolling(
            int(rolling), center=True).mean().reindex(table["local_date"]).to_numpy()
        ax.plot(table["local_date"], table["rolling_mean"], color="#1f4e79", linewidth=1.6, label=f"{rolling}-day mean")
        ax.legend(fontsize=8, loc="upper left")
    ax.set_ylabel("participants with data")
    ax.set_xlabel("local date")
    _titles(ax, "Participants with data per day", _caption(source, _count_participants(source), f"at most {int(table['participants'].max()):,} on one day"))
    return _finish(fig, table, path, dpi)


def plot_feature_activity(source, *, measure: str = "participants", features: Iterable[str] | None = None,
                          path=None, dpi: int = 150):
    """
    Features (rows) by calendar month (columns): the participants with any data that month, or the participant-days
    (``measure="participant_days"``). Months are counted from local dates, and every month in the span is drawn.
    """

    _need(measure in ("participants", "participant_days"), "measure must be 'participants' or 'participant_days'")
    wanted = None if features is None else set(features)
    counts: dict[tuple[str, pd.Period], int] = {}
    for days in sm._Source(source).grouped("participant_days"):
        months = pd.to_datetime(days["local_date"]).dt.to_period("M")
        seen: set[tuple[str, pd.Period]] = set()
        for month, text in zip(months, days["features"]):
            for feature in str(text).split(";"):
                if wanted is not None and feature not in wanted:
                    continue
                key = (feature, month)
                if measure == "participant_days":
                    counts[key] = counts.get(key, 0) + 1
                elif key not in seen:
                    seen.add(key)
                    counts[key] = counts.get(key, 0) + 1
    _need(bool(counts), "the run holds no participant-day for these features")
    table = pd.DataFrame([(f, m, n) for (f, m), n in counts.items()], columns=["feature", "month", measure])
    span = pd.period_range(table["month"].min(), table["month"].max(), freq="M")
    matrix = table.pivot_table(index="feature", columns="month", values=measure, aggfunc="sum", fill_value=0).reindex(columns=span, fill_value=0)
    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False, kind="stable").index]
    fig, axes = _figure(min(16.0, max(9.0, 0.12 * len(span) + 4)), max(3.5, 0.3 * len(matrix) + 1.5))
    ax = axes[0, 0]
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="viridis")
    fig.colorbar(image, ax=ax, label=measure.replace("_", "-"))
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    step = max(1, len(span) // 12)
    ax.set_xticks(range(0, len(span), step))
    ax.set_xticklabels([str(p) for p in span[::step]], rotation=90, fontsize=7)
    _titles(ax, f"{measure.replace('_', '-').capitalize()} per feature and month", _caption(source, _count_participants(source)))
    data = matrix.reset_index().melt(id_vars="feature", var_name="month", value_name=measure)
    data["month"] = data["month"].astype(str)
    return _finish(fig, data, path, dpi)


def _count_participants(source) -> int:
    table = sm._Source(source).table("participant_feature")
    return int(table["RegistrationCode"].nunique()) if len(table) else 0


# --------------------------------------------------------------------------------------------- daily patterns
def plot_hour_of_day(source, feature: str, *, measure: str = "value", path=None, dpi: int = 150):
    """
    The cohort's median, with interquartile band, by local hour of day: the mean value for levels, the average
    amount per day for totals and event amounts (``measure="value"``), or records per day (``measure="records"``).
    """

    _need(measure in ("value", "records"), "measure must be 'value' or 'records'")
    participants, cohort = sm.hour_of_day(source)
    _need(len(cohort) and feature in set(cohort["feature"]), f"no hour-of-day data for {feature}")
    rows = cohort[cohort["feature"] == feature].sort_values("hour")
    kind = ds.measurement_kind(feature)
    if measure == "records":
        column, label = "records_per_day", "records per day"
    elif kind in ("extensive_total", "event_amount"):
        column = "value_per_day"
        unit = _hour_unit(source, feature)
        label = f"amount per day ({unit})"
    else:
        column = "value_mean"
        label = f"mean value ({_hour_unit(source, feature)})"
    _need(rows[f"{column}_median"].notna().any(), f"{feature} has no {measure} by hour of day")
    table = rows[["hour", f"{column}_median", f"{column}_p25", f"{column}_p75", f"{column}_participants"]].rename(
        columns=lambda c: c.replace(f"{column}_", ""))
    fig, axes = _figure(9.0, 4.5)
    ax = axes[0, 0]
    ax.fill_between(table["hour"], table["p25"], table["p75"], color="#9ecae1", alpha=0.6, label="interquartile range")
    ax.plot(table["hour"], table["median"], color="#1f4e79", marker="o", markersize=3, label="median")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("local hour of day")
    ax.set_ylabel(label)
    ax.legend(fontsize=8)
    _titles(ax, f"{feature} by hour of day", _caption(source, int(table["participants"].max())))
    return _finish(fig, table, path, dpi)


def _hour_unit(source, feature: str) -> str:
    for daily in sm._Source(source).participants_of(feature):
        units = daily["value_unit"].dropna().astype(str).unique() if "value_unit" in daily else []
        if len(units):
            return units[0]
    return "unit not established"


def _pattern(patterns: "sm.TemporalPatterns", feature: str, metric: str | None, by: str) -> tuple[pd.DataFrame, str]:
    table = patterns.cohort_day_of_week if by == "weekday" else patterns.cohort_month_of_year
    rows = table[table["feature"] == feature]
    _need(len(rows) > 0, f"no {by} pattern for {feature}")
    if metric is None:  # the headline metric, chosen by measurement kind, never whichever sorts first
        ranked = sm.headline_metrics(feature, list(dict.fromkeys(rows["metric"])))
        _need(bool(ranked), f"{feature} has no metric to plot")
        metric = ranked[0]
    rows = rows[rows["metric"] == metric]
    _need(len(rows) > 0, f"no {by} pattern for {feature} {metric}")
    return rows.sort_values(by), metric


def plot_weekly_pattern(patterns: "sm.TemporalPatterns", feature: str, metric: str | None = None, *, path=None, dpi: int = 150):
    """The cohort's median of participants' medians by day of week, with interquartile range; weekend shaded."""

    rows, metric = _pattern(patterns, feature, metric, "weekday")
    fig, axes = _figure(8.0, 4.5)
    ax = axes[0, 0]
    for day in patterns.weekend_days:
        ax.axvspan(day - 0.5, day + 0.5, color="0.92", zorder=0)
    ax.errorbar(rows["weekday"], rows["median"], yerr=[rows["median"] - rows["p25"], rows["p75"] - rows["median"]],
                fmt="o-", color="#1f4e79", capsize=3)
    ax.set_xticks(range(7))
    ax.set_xticklabels(WEEKDAY_NAMES)
    ax.set_ylabel(f"{metric} (median of participants' medians)")
    _titles(ax, f"{feature} by day of week (weekend shaded)", _participants(int(rows["participants"].max())))
    return _finish(fig, rows, path, dpi)


def plot_monthly_pattern(patterns: "sm.TemporalPatterns", feature: str, metric: str | None = None, *, path=None, dpi: int = 150):
    """The cohort's median of participants' medians by month of year, with interquartile range."""

    rows, metric = _pattern(patterns, feature, metric, "month")
    fig, axes = _figure(8.0, 4.5)
    ax = axes[0, 0]
    ax.errorbar(rows["month"], rows["median"], yerr=[rows["median"] - rows["p25"], rows["p75"] - rows["median"]],
                fmt="o-", color="#1f4e79", capsize=3)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_ylabel(f"{metric} (median of participants' medians)")
    _titles(ax, f"{feature} by month of year", _participants(int(rows["participants"].max())))
    return _finish(fig, rows, path, dpi)


# -------------------------------------------------------------------------------------------------- values
def _daily_rows(source, feature: str, metric: str | None, valid_only: bool, rules, participant: str | None):
    """(rows with local_date, RegistrationCode, value, valid), the metric and its unit."""

    src = sm._Source(source)
    _need(feature in src.features, f"{feature} is not among the run's features")
    cadence = {(r["feature"], r["RegistrationCode"]): r for r in src.table("participant_feature").to_dict("records")}
    rule = (rules or {}).get(feature, sm.DEFAULT_RULES.get(feature, sm.DEFAULT_RULE))
    wanted = None if participant is None else (participant if str(participant).startswith("10K_") else f"10K_{participant}")
    frames, chosen, unit = [], metric, None
    for daily in src.participants_of(feature):
        code = str(daily["RegistrationCode"].iloc[0])
        if wanted is not None and code != wanted:
            continue
        chosen = _metric(feature, daily.columns, metric)
        c = cadence.get((feature, code), {})
        marked = sm.mark_valid_days(feature, daily, rule, sm.expected_records_per_day(
            feature, c.get("typical_gap_minutes"), c.get("regular_share")))
        unit = unit or _unit(feature, chosen, marked)
        frame = marked[["RegistrationCode", "local_date", "valid"]].assign(value=pd.to_numeric(marked[chosen], errors="coerce"))
        frames.append(frame[frame["valid"]] if valid_only else frame)
    _need(bool(frames), f"no daily rows for {feature}" + (f" and participant {wanted}" if wanted else ""))
    rows = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    _need(len(rows) > 0, f"{feature} has no {chosen} value to plot")
    return rows, chosen, unit


def plot_daily_values(source, feature: str, participant: str | None = None, *, metric: str | None = None,
                      valid_only: bool = False, rules=None, show_ids: bool = False, path=None, dpi: int = 150):
    """
    One participant's daily headline metric over time, invalid days drawn hollow; or, with ``participant=None``, the
    cohort's median per day with interquartile band, over valid days.
    """

    rows, metric, unit = _daily_rows(source, feature, metric, valid_only or participant is None, rules, participant)
    fig, axes = _figure(10.0, 4.5)
    ax = axes[0, 0]
    dates = pd.to_datetime(rows["local_date"])
    if participant is not None:
        table = rows.assign(local_date=dates).sort_values("local_date")
        valid, invalid = table[table["valid"]], table[~table["valid"]]
        ax.plot(table["local_date"], table["value"], color="0.7", linewidth=0.6)
        ax.plot(valid["local_date"], valid["value"], "o", color="#1f4e79", markersize=3, label="valid day")
        if len(invalid):
            ax.plot(invalid["local_date"], invalid["value"], "o", markerfacecolor="none", color="#1f4e79", markersize=3, label="other day")
        ax.legend(fontsize=8)
        who = table["RegistrationCode"].iloc[0] if show_ids else "one participant"
        caption = f"{who} · {len(valid):,} valid of {len(table):,} days"
    else:
        grouped = rows.assign(local_date=dates).groupby("local_date")["value"]
        table = pd.DataFrame({"median": grouped.median(), "p25": grouped.quantile(0.25), "p75": grouped.quantile(0.75),
                              "participants": grouped.count()}).reset_index()
        ax.fill_between(table["local_date"], table["p25"], table["p75"], color="#9ecae1", alpha=0.6, label="interquartile range")
        ax.plot(table["local_date"], table["median"], color="#1f4e79", linewidth=0.9, label="median")
        ax.legend(fontsize=8)
        caption = _caption(source, int(rows["RegistrationCode"].nunique()), "valid days")
    ax.set_ylabel(f"{metric} ({unit})")
    ax.set_xlabel("local date")
    _titles(ax, f"{feature}: daily {metric}", caption)
    return _finish(fig, table, path, dpi)


def plot_monthly_distribution(source, feature: str, *, metric: str | None = None, valid_only: bool = True, rules=None,
                              period: str = "auto", clip_quantile: float | None = None, path=None, dpi: int = 150):
    """
    Boxplots of participant-days' values per calendar month or quarter: the median, quartiles, and whiskers at 1.5
    times the interquartile range. Every period in the span is drawn, so gaps in the data stay visible. ``period`` is
    ``"month"``, ``"quarter"``, or ``"auto"``: months up to 36 of them, quarters beyond, as the title states. Nothing is
    clipped unless ``clip_quantile`` is given; then the axis stops at that quantile and the number of values beyond it
    is stated on the figure.
    """

    _need(period in ("auto", "month", "quarter"), "period must be 'auto', 'month' or 'quarter'")
    rows, metric, unit = _daily_rows(source, feature, metric, valid_only, rules, None)
    dates = pd.to_datetime(rows["local_date"])
    span_months = len(pd.period_range(dates.min().to_period("M"), dates.max().to_period("M"), freq="M"))
    freq = "M" if period == "month" or (period == "auto" and span_months <= 36) else "Q"
    rows = rows.assign(month=dates.dt.to_period(freq))
    months = pd.period_range(rows["month"].min(), rows["month"].max(), freq=freq)
    groups = [rows.loc[rows["month"] == m, "value"].to_numpy() for m in months]
    table = pd.DataFrame({"period": [str(m) for m in months], "n_days": [len(g) for g in groups],
                          "median": [np.median(g) if len(g) else np.nan for g in groups],
                          "p25": [np.quantile(g, 0.25) if len(g) else np.nan for g in groups],
                          "p75": [np.quantile(g, 0.75) if len(g) else np.nan for g in groups]})
    fig, axes = _figure(min(16.0, max(8.0, 0.35 * len(months) + 3)), 5.0)
    ax = axes[0, 0]
    positions = [i for i, g in enumerate(groups) if len(g)]
    ax.boxplot([groups[i] for i in positions], positions=positions, widths=0.6, showfliers=True,
               flierprops={"markersize": 2, "alpha": 0.4})
    step = max(1, len(months) // 24)
    ax.set_xticks(range(0, len(months), step))
    ax.set_xticklabels([str(m) for m in months[::step]], rotation=90, fontsize=7)
    ax.set_xlim(-0.7, len(months) - 0.3)
    caption = f"{rows['RegistrationCode'].nunique():,} participants · {len(rows):,} {'valid ' if valid_only else ''}days"
    if clip_quantile is not None:
        _need(0 < clip_quantile < 1, "clip_quantile must be between 0 and 1")
        limit = float(np.quantile(rows["value"], clip_quantile))
        beyond = int((rows["value"] > limit).sum())
        ax.set_ylim(top=limit)
        caption += f" · axis clipped at the {clip_quantile:.0%} quantile: {beyond:,} values above"
    ax.set_ylabel(f"{metric} ({unit})")
    _titles(ax, f"{feature}: daily {metric} by {'month' if freq == 'M' else 'quarter'}", caption)
    return _finish(fig, table, path, dpi)


# ------------------------------------------------------------------------------------------------ adherence
def plot_adherence(summaries: "sm.Summaries", feature: str, *, path=None, dpi: int = 150):
    """Valid days per participant, and adherence (valid days over the follow-up span), for one feature."""

    table = summaries.adherence[summaries.adherence["feature"] == feature]
    _need(len(table) > 0, f"no adherence for {feature}")
    fig, axes = _figure(10.0, 4.2, ncols=2)
    left, right = axes[0, 0], axes[0, 1]
    left.hist(table["valid_days"], bins=min(30, max(5, len(table))), color="#1f4e79")
    left.set_xlabel("valid days")
    left.set_ylabel("participants")
    right.hist(table["adherence"], bins=np.linspace(0, 1, 21), color="#1f4e79")
    right.set_xlabel("adherence (valid days / follow-up days)")
    _titles(left, f"{feature}: valid days", _participants(len(table)))
    _titles(right, "Adherence", str(table["rule"].iloc[0]) if "rule" in table else "")
    return _finish(fig, table, path, dpi)


def plot_retention(summaries: "sm.Summaries", features: Iterable[str] | None = None, *, path=None, dpi: int = 150):
    """The share of participants still contributing valid days, by days since their first valid day."""

    table = summaries.retention
    _need(len(table) > 0, "the summaries hold no retention")
    chosen = list(dict.fromkeys(table["feature"])) if features is None else list(features)
    table = table[table["feature"].isin(chosen)]
    _need(len(table) > 0, "no retention for these features")
    fig, axes = _figure(9.0, 5.0)
    ax = axes[0, 0]
    for feature, rows in table.groupby("feature", sort=False):
        ax.step(rows["days_since_first_valid"], rows["fraction"], where="post", label=f"{feature} (n={int(rows['participants'].iloc[0])})")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("days since first valid day")
    ax.set_ylabel("share still contributing")
    ax.legend(fontsize=7, ncol=2 if len(chosen) > 8 else 1)
    _titles(ax, "Retention", f"{len(chosen)} feature{'s' if len(chosen) != 1 else ''}")
    return _finish(fig, table, path, dpi)


# -------------------------------------------------------------------------------------------------- context
def plot_co_availability(source, features: Iterable[str] | None = None, *, path=None, dpi: int = 150):
    """For each pair of features, the share of their participant-days they share (the Jaccard index)."""

    overlap = sm.day_overlap(source)
    _need(len(overlap) > 0, "the run holds no participant-day")
    names = sorted(set(overlap["feature_a"])) if features is None else sorted(set(features))
    matrix = pd.DataFrame(np.nan, index=names, columns=names)
    for row in overlap.itertuples():
        if row.feature_a in matrix.index and row.feature_b in matrix.index:
            matrix.loc[row.feature_a, row.feature_b] = matrix.loc[row.feature_b, row.feature_a] = row.jaccard
    fig, axes = _figure(max(6.0, 0.35 * len(names) + 3), max(5.0, 0.35 * len(names) + 2))
    ax = axes[0, 0]
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Jaccard index of participant-days")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    if len(names) <= 12:
        for i in range(len(names)):
            for j in range(len(names)):
                if not np.isnan(matrix.iat[i, j]):
                    ax.text(j, i, f"{matrix.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if matrix.iat[i, j] < 0.6 else "black")
    _titles(ax, "Days shared by pairs of features", _caption(source, _count_participants(source)))
    data = matrix.rename_axis("feature_a").reset_index().melt(id_vars="feature_a", var_name="feature_b", value_name="jaccard")
    return _finish(fig, data, path, dpi)


def _stacked_shares(table: pd.DataFrame, category: str, title: str, caption: str, path, dpi):
    shares = table.pivot_table(index="feature", columns=category, values="share", aggfunc="sum", fill_value=0.0)
    shares = shares.loc[sorted(shares.index, reverse=True)]
    order = shares.sum().sort_values(ascending=False, kind="stable").index
    fig, axes = _figure(9.0, max(3.5, 0.3 * len(shares) + 1.5))
    ax = axes[0, 0]
    left = np.zeros(len(shares))
    for i, name in enumerate(order):
        values = shares[name].to_numpy()
        ax.barh(shares.index, values, left=left, label=str(name), color=PALETTE[i % len(PALETTE)])
        left += values
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of records")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _titles(ax, title, caption)
    data = shares.reset_index().melt(id_vars="feature", var_name=category, value_name="share")
    return _finish(fig, data, path, dpi)


def plot_acquisition(report: "sm.QualityReport", *, path=None, dpi: int = 150):
    """For each feature, the share of records by acquisition method."""

    _need(len(report.provenance) > 0, "the quality report holds no provenance")
    return _stacked_shares(report.provenance, "acquisition_method", "How records were acquired",
                           f"{report.provenance['feature'].nunique()} features", path, dpi)


def plot_curation(report: "sm.QualityReport", *, path=None, dpi: int = 150):
    """For each feature, the share of records by curation status (curated phase)."""

    curation = report.curation
    _need(len(curation) > 0 and (curation["kind"] == "status").any(), "the quality report holds no curation status")
    statuses = curation[curation["kind"] == "status"].rename(columns={"name": "status", "share_of_records": "share"})
    return _stacked_shares(statuses, "status", "Curation status of records", f"{statuses['feature'].nunique()} features", path, dpi)


# --------------------------------------------------------------------------------------------------- domain
def plot_sleep(metrics: "dm.DomainMetrics", *, rule: "dm.NightRule" = dm.NightRule(), path=None, dpi: int = 150):
    """Over valid nights: total sleep time, the clock time of the sleep midpoint, and sleep efficiency."""

    nights = metrics.sleep_nights
    valid = nights[nights["asleep_recorded"].astype(bool) & (nights["asleep_minutes"] >= rule.min_asleep_minutes)]
    _need(len(valid) > 0, "no valid night: " + rule.describe())
    fig, axes = _figure(12.0, 4.0, ncols=3)
    hours = valid["asleep_minutes"] / 60
    axes[0, 0].hist(hours, bins=np.arange(np.floor(hours.min()), np.ceil(hours.max()) + 0.5, 0.5), color="#1f4e79")
    axes[0, 0].set_xlabel("total sleep time (h)")
    axes[0, 0].set_ylabel("nights")
    midpoint = valid["midpoint_hours_after_noon"]
    axes[0, 1].hist(midpoint, bins=np.arange(np.floor(midpoint.min()), np.ceil(midpoint.max()) + 0.5, 0.5), color="#1f4e79")
    ticks = np.arange(np.floor(midpoint.min()), np.ceil(midpoint.max()) + 1)
    axes[0, 1].set_xticks(ticks)
    axes[0, 1].set_xticklabels([f"{int((12 + t) % 24):02d}:00" for t in ticks], rotation=90, fontsize=7)
    axes[0, 1].set_xlabel("sleep midpoint (local clock time)")
    efficiency = valid["efficiency"].dropna()
    axes[0, 2].hist(efficiency, bins=np.linspace(max(0.0, efficiency.min() - 0.02), 1.0, 25), color="#1f4e79")
    axes[0, 2].set_xlabel("sleep efficiency")
    _titles(axes[0, 0], "Total sleep time", f"{len(valid):,} valid nights")
    _titles(axes[0, 1], "Sleep midpoint", _participants(valid["RegistrationCode"].nunique()))
    _titles(axes[0, 2], "Efficiency", rule.describe())
    columns = ["RegistrationCode", "night_date", "asleep_minutes", "midpoint_hours_after_noon", "efficiency"]
    return _finish(fig, valid[columns], path, dpi)


def plot_cgm_ranges(metrics: "dm.DomainMetrics", *, sufficient_only: bool = True, show_ids: bool = False,
                    path=None, dpi: int = 150):
    """
    For each CGM participant, the percentage of readings in each consensus range, pooled over valid days; the
    dashed line marks the consensus target of more than 70% in range.
    """

    periods = metrics.cgm_periods
    chosen = periods[periods["cgm"].astype(bool) & (periods["valid_days"] > 0)]
    if sufficient_only:
        chosen = chosen[chosen["sufficient"].astype(bool)]
    _need(len(chosen) > 0, "no CGM participant" + (" with the 14 days the consensus requires" if sufficient_only else ""))
    chosen = chosen.sort_values("in_range_percent", ascending=True, kind="stable")
    labels = _labels(chosen["RegistrationCode"], show_ids)
    fig, axes = _figure(9.0, max(3.0, 0.35 * len(chosen) + 1.8))
    ax = axes[0, 0]
    left = np.zeros(len(chosen))
    for name, colour in GLUCOSE_COLOURS.items():
        values = chosen[name].to_numpy(dtype=float)
        ax.barh(labels, values, left=left, color=colour, label=GLUCOSE_LABELS[name])
        left += values
    ax.axvline(TIME_IN_RANGE_TARGET, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of readings (valid days pooled)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _titles(ax, "Time in glucose ranges", f"{_participants(len(chosen))} · target > {TIME_IN_RANGE_TARGET:.0f}% in range")
    columns = ["RegistrationCode", "valid_days", "readings", *GLUCOSE_COLOURS]
    return _finish(fig, chosen[columns], path, dpi)


def bmi_categories(summaries: "sm.Summaries") -> pd.DataFrame:
    """Participants per WHO adult BMI class, by each participant's median BMI over valid days."""

    metrics = summaries.participant_metrics
    bmi = metrics[(metrics["feature"] == "BMI") & (metrics["metric"] == "value_mean")].dropna(subset=["median"])
    rows = []
    for name, lower, upper in BMI_CLASSES:
        n = int(((bmi["median"] >= lower) & (bmi["median"] < upper)).sum())
        rows.append({"category": name, "lower": lower, "upper": upper, "participants": n})
    table = pd.DataFrame(rows)
    table["share"] = table["participants"] / table["participants"].sum() if len(bmi) else np.nan
    return table


def _bmi_range(lower: float, upper: float) -> str:
    if lower == 0:
        return f"< {upper:g}"
    if np.isinf(upper):
        return f">= {lower:g}"
    return f"{lower:g} to < {upper:g}"


def plot_bmi_categories(summaries: "sm.Summaries", *, path=None, dpi: int = 150):
    """Participants per WHO adult BMI class, from each participant's median BMI over valid days."""

    table = bmi_categories(summaries)
    _need(table["participants"].sum() > 0, "no participant with a BMI value")
    fig, axes = _figure(9.0, 4.5)
    ax = axes[0, 0]
    labels = [f"{name}\n{_bmi_range(lower, upper)}" for name, lower, upper in zip(table["category"], table["lower"], table["upper"])]
    bars = ax.bar(labels, table["participants"], color="#1f4e79")
    ax.bar_label(bars, labels=[f"{n} ({s:.0%})" for n, s in zip(table["participants"], table["share"])], fontsize=7)
    ax.set_ylabel("participants")
    ax.tick_params(axis="x", labelsize=7)
    _titles(ax, "BMI categories (WHO, kg/m2)", _participants(int(table["participants"].sum())))
    return _finish(fig, table, path, dpi)
