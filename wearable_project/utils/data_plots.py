"""
Wearable Data Processing and Modeling project
Figures from the statistics modules: coverage and availability, data quality, activity over time, daily patterns,
values, adherence, context, sleep, glucose and BMI, and a report that gathers them.
Every function draws from the output of ``data_statistics``, ``data_summaries`` or ``domain_metrics``, never from raw
files, so each figure inherits their rules: local calendar days, values combined as the feature's measurement kind
requires, harmonized units, valid days, time counted once across devices. Each returns a matplotlib ``Figure`` built
with the object-oriented API: nothing is shown, no global state is touched, and it works without a display. Pass
``path=`` to save it; the format follows the file's suffix, and a data root is refused as a destination
(``data_statistics.guard_output``).
Every figure carries the exact table it draws as ``figure.data``; a figure of several panels also carries each panel's
table in ``figure.panels``. ``figure.individual_level`` says whether it shows individual participants (rows, points or
bars per person), as opposed to cohort aggregates.
Shared options, where they apply:
- ``groups``: a mapping from participant to a label (a dict, a Series, or a table with ``RegistrationCode`` and
  ``group``), drawing one line, box or bar per group with its size stated. Participants without a label are left
  out, and the figure says how many.
- ``min_participants``: a point, bar, cell or bin resting on fewer participants is not drawn, its values are removed
  from ``figure.data`` too, and the figure states how many were hidden. For figures leaving the lab.
- ``ci``: a bootstrap interval of the cohort median (seeded by ``seed``), distinct from the interquartile band, which
  shows spread rather than uncertainty.
- ``unit``: whether a cohort distribution counts each participant once (the default) or pools participant-days.
- ``time_axis``: ``"calendar"`` or ``"study"`` (days since each participant's first day). Figures of one participant
  default to study time, so that real dates appear only when asked for.
Features are ordered by their category (physical activity, heart, sleep, glucose, respiratory, body, nutrition,
other) and keep one color in every figure (``feature_color``). Axes are honest: every histogram bin and calendar
period is drawn, nothing is clipped unless asked (and then the figure says how many values lie beyond), units come
from the statistics, and each figure states how many participants it describes. Participant identifiers appear only
with ``show_ids=True``.
"""


from __future__ import annotations
import json
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping
import numpy as np
import pandas as pd
from wearable_project.curation.guidance import get_feature_guide
from wearable_project.exceptions import DataLoaderConfigurationError
from wearable_project.utils import data_statistics as ds
from wearable_project.utils import data_summaries as sm
from wearable_project.utils import domain_metrics as dm
if TYPE_CHECKING:  # what the figure functions return, for type checkers; at runtime a plain matplotlib Figure
    from matplotlib.figure import Figure

    class PlotFigure(Figure):
        data: pd.DataFrame
        panels: dict[str, pd.DataFrame]
        individual_level: bool


WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
# Data volume bands, in bytes, for the size distribution.
VOLUME_BANDS = ((0, 1e6, "< 1 MB"), (1e6, 1e7, "1-10 MB"), (1e7, 1e8, "10-100 MB"), (1e8, 1e9, "100 MB-1 GB"),
                (1e9, np.inf, ">= 1 GB"))
# WHO adult BMI classes, kg/m2, each a half-open interval [lower, upper).
BMI_CLASSES = (("Underweight", 0.0, 18.5), ("Normal weight", 18.5, 25.0), ("Pre-obesity", 25.0, 30.0),
               ("Obesity class I", 30.0, 35.0), ("Obesity class II", 35.0, 40.0), ("Obesity class III", 40.0, np.inf))
# Colors for the consensus glucose ranges, from very low to very high.
GLUCOSE_COLORS = {"very_low_percent": "#8e0000", "low_percent": "#e53935", "in_range_percent": "#43a047",
                   "high_percent": "#fdd835", "very_high_percent": "#fb8c00"}
GLUCOSE_LABELS = {"very_low_percent": "< 54 mg/dL", "low_percent": "54-69 mg/dL", "in_range_percent": "70-180 mg/dL",
                  "high_percent": "181-250 mg/dL", "very_high_percent": "> 250 mg/dL"}
# Okabe-Ito, a palette distinguishable with the common color-vision deficiencies.
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000", "#999999", "#882255")
SINGLE = "#1f4e79"   # the color of an ungrouped series
BAND = "#9ecae1"     # the color of an ungrouped interquartile band
TIME_IN_RANGE_TARGET = 70.0  # consensus target for most adults with diabetes, % of readings in 70-180 mg/dL
ALL_PARTICIPANTS = "all participants"
BOOTSTRAP_SAMPLES = 1000
# Feature categories, as the feature guides declare them, in the order figures list them, each with a base color.
CATEGORY_ORDER = ("Physical activity and energy", "Heart and cardiovascular", "Sleep and recovery",
                  "Glucose and metabolism", "Respiratory and vital signs", "Anthropometrics and body composition",
                  "Nutrition", "Other health data")
CATEGORY_COLORS = {"Physical activity and energy": "#009E73", "Heart and cardiovascular": "#D55E00",
                    "Sleep and recovery": "#0072B2", "Glucose and metabolism": "#CC79A7",
                    "Respiratory and vital signs": "#56B4E9", "Anthropometrics and body composition": "#E69F00",
                    "Nutrition": "#8C6D1F", "Other health data": "#777777"}


# --------------------------------------------------------------------------------------- feature identity
@lru_cache(maxsize=None)
def feature_category(feature: str) -> str:
    """The category the feature's guide declares, such as "Heart and cardiovascular"."""

    try:
        return get_feature_guide(feature).category
    except Exception:
        return "Other health data"


def feature_order(features: Iterable[str] | None = None) -> list[str]:
    """Features in figure order: by category (``CATEGORY_ORDER``), then by name."""

    names = ds.available_features() if features is None else list(dict.fromkeys(features))
    rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    return sorted(names, key=lambda f: (rank.get(feature_category(f), len(rank)), f))


def _shade(color: str, amount: float) -> str:
    rgb = np.array([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    rgb = rgb * (1 + amount) if amount < 0 else rgb + (1 - rgb) * amount
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in np.clip(rgb, 0, 1))


@lru_cache(maxsize=None)
def feature_color(feature: str) -> str:
    """One color per feature in every figure: its category's hue, darker or lighter by its place in the category."""

    category = feature_category(feature)
    members = sorted(f for f in ds.available_features() if feature_category(f) == category)
    base = CATEGORY_COLORS.get(category, "#777777")
    if len(members) <= 1 or feature not in members:
        return base
    return _shade(base, -0.35 + 0.8 * members.index(feature) / (len(members) - 1))


# ------------------------------------------------------------------------------------------------ helpers
def _figure(width: float = 9.0, height: float = 5.0, nrows: int = 1, ncols: int = 1, **grid):
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise DataLoaderConfigurationError('figures need matplotlib: pip install "wearable_project[plots]"') from exc
    figure = Figure(figsize=(width, height), layout="constrained")
    return figure, figure.subplots(nrows, ncols, squeeze=False, gridspec_kw=grid or None)


def _figure_with_strip(width: float = 9.0, height: float = 5.2):
    """A main axes above a strip, sharing the x axis, for the participants behind each point."""

    fig, axes = _figure(width, height, 2, 1, height_ratios=[4, 1])
    main, strip = axes[0, 0], axes[1, 0]
    strip.sharex(main)
    return fig, main, strip


def _finish(figure, data: pd.DataFrame, path: str | Path | None, dpi: int, *, individual: bool = False,
            panels: Mapping[str, pd.DataFrame] | None = None) -> "PlotFigure":
    figure.data = data.reset_index(drop=True)
    figure.panels = {name: table.reset_index(drop=True) for name, table in (panels or {}).items()}
    figure.individual_level = individual
    if path is not None:  # never into a data root (data_statistics.protected_roots)
        figure.savefig(ds.guard_output(path), dpi=dpi)
    return figure


def _titles(ax, title: str, caption: str) -> None:
    """The title at the top left and the caption beneath it, wrapped to the panel's width so that nothing overlaps."""

    inches = ax.get_position().width * ax.figure.get_figwidth()
    lines = textwrap.wrap(caption, width=max(24, int(inches * 16)), break_on_hyphens=False) if caption else []
    ax.set_title(title, loc="left", fontsize=11, pad=5 + 10 * len(lines))
    if lines:
        ax.annotate("\n".join(lines), xy=(0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                    ha="left", va="bottom", fontsize=7.5, color="0.4")


def _participants(n: int) -> str:
    return f"{n:,} participant{'s' if n != 1 else ''}"


def _phase(source) -> str | None:
    if isinstance(source, (ds.DailyStatistics, ds.Coverage)):
        return source.phase
    try:
        run = Path(source) / "run_parameters.json"
    except TypeError:
        return None
    return json.loads(run.read_text()).get("phase") if run.is_file() else None


def _caption(source, n: int, *extra: str) -> str:
    phase = _phase(source)
    return " · ".join(p for p in (f"{phase} phase" if phase else None, _participants(n), *extra) if p)


def _labels(codes: Iterable[str], show_ids: bool) -> list[str]:
    codes = list(codes)
    return codes if show_ids else [f"Participant {i + 1}" for i in range(len(codes))]


def _need(condition: "bool | np.bool_", message: str) -> None:
    if not condition:
        raise DataLoaderConfigurationError(message)


def _check_min(min_participants: int) -> int:
    _need(isinstance(min_participants, (int, np.integer)) and min_participants >= 1, "min_participants must be a whole number of at least 1")
    return int(min_participants)


def _hidden(count: int, min_participants: int, what: str = "point") -> str:
    if not count:
        return ""
    return f"{count:,} {what}{'s' if count != 1 else ''} hidden (fewer than {min_participants} participants)"


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


def _code(code) -> str:
    text = str(code).strip()
    return text if text.startswith("10K_") else f"10K_{text}"


def _group_map(groups) -> pd.Series | None:
    """``groups`` as a Series from registration code to label, validated."""

    if groups is None:
        return None
    if isinstance(groups, pd.DataFrame):
        _need({"RegistrationCode", "group"} <= set(groups.columns), "groups as a table needs RegistrationCode and group columns")
        series = groups.set_index("RegistrationCode")["group"]
    elif isinstance(groups, pd.Series):
        series = groups
    else:
        series = pd.Series(dict(groups), dtype=object)
    series = series.dropna()
    _need(len(series) > 0, "groups is empty")
    series.index = pd.Index([_code(c) for c in series.index])
    _need(series.index.is_unique, "a participant appears more than once in groups")
    return series.astype(str)


def _grouped(frame: pd.DataFrame, groups) -> tuple[pd.DataFrame, list[str], int]:
    """The frame with a ``group`` column, the group labels in order, and the participants left out for lacking one."""

    mapping = _group_map(groups)
    if mapping is None:
        return frame.assign(group=ALL_PARTICIPANTS), [ALL_PARTICIPANTS], 0
    group = frame["RegistrationCode"].map(mapping)
    excluded = int(frame.loc[group.isna(), "RegistrationCode"].nunique())
    kept = frame.assign(group=group)[group.notna()]
    _need(len(kept) > 0, "no participant here has a group")
    return kept, sorted(kept["group"].unique()), excluded


def _group_color(labels: list[str], label: str, fallback: str = SINGLE) -> str:
    return fallback if labels == [ALL_PARTICIPANTS] else PALETTE[labels.index(label) % len(PALETTE)]


def _group_legend(label: str, n: int) -> str:
    return f"{label} (n={n:,})"


def _left_out(excluded: int) -> str:
    return f"{excluded:,} without a group left out" if excluded else ""


def _bootstrap(values: np.ndarray, level: float, seed: int, samples: int = BOOTSTRAP_SAMPLES) -> tuple[float, float]:
    """A percentile bootstrap interval of the median, resampling participants."""

    n = len(values)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    chunk = max(1, min(samples, 2_000_000 // n))
    medians, done = [], 0
    while done < samples:
        k = min(chunk, samples - done)
        medians.append(np.median(values[rng.integers(0, n, size=(k, n))], axis=1))
        done += k
    low, high = np.quantile(np.concatenate(medians), [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(low), float(high)


def _cohort(frame: pd.DataFrame, x: str, value: str, min_participants: int, ci: float | None, seed: int) -> pd.DataFrame:
    """
    Per group and x: the median and quartiles of one value per participant, the participants behind them, and with
    ``ci`` a bootstrap interval of the median. Points resting on fewer than ``min_participants`` participants are
    suppressed: their statistics and count are removed.
    """

    if ci is not None:
        _need(0 < ci < 1, "ci must be between 0 and 1, such as 0.95")
    rows = []
    for (group, key), part in frame.groupby(["group", x], sort=True):
        v = part[value].dropna().to_numpy(dtype=float)
        n = int(part.loc[part[value].notna(), "RegistrationCode"].nunique())
        row = {"group": group, x: key, "participants": n,
               "median": float(np.median(v)) if len(v) else np.nan,
               "p25": float(np.quantile(v, 0.25)) if len(v) else np.nan,
               "p75": float(np.quantile(v, 0.75)) if len(v) else np.nan}
        if ci is not None:
            row["ci_low"], row["ci_high"] = _bootstrap(v, ci, seed)
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["suppressed"] = (table["participants"] < min_participants) & (table["participants"] > 0)
    statistics = [c for c in ("median", "p25", "p75", "ci_low", "ci_high") if c in table]
    table.loc[table["suppressed"], statistics] = np.nan
    table["participants"] = table["participants"].astype(float)
    table.loc[table["suppressed"], "participants"] = np.nan
    return table


def _strip(ax, table: pd.DataFrame, x: str, labels: list[str], width: float = 0.8) -> None:
    """Stacked bars of the participants behind each point, by group."""

    xs = sorted(table[x].unique())
    bottom = np.zeros(len(xs))
    for label in labels:
        counts = table[table["group"] == label].set_index(x)["participants"].reindex(xs).fillna(0).to_numpy()
        ax.bar(xs, counts, bottom=bottom, width=width, color=_group_color(labels, label, "0.6"), linewidth=0)
        bottom += counts
    ax.set_ylabel("participants", fontsize=8)
    ax.tick_params(labelsize=7)


def _study_day(frame: pd.DataFrame, column: str = "local_date") -> pd.Series:
    dates = pd.to_datetime(frame[column])
    return (dates - dates.groupby(frame["RegistrationCode"]).transform("min")).dt.days


def _bars_masked(ax, edges: np.ndarray, counts: np.ndarray, color: str, *, participants: np.ndarray | None = None,
                 min_participants: int = 1, label: str | None = None) -> int:
    """Histogram bars from edges and counts, bins resting on too few participants left undrawn; returns how many."""

    people = counts if participants is None else participants
    hide = (people > 0) & (people < min_participants)
    keep = ~hide
    ax.bar(edges[:-1][keep], counts[keep], width=np.diff(edges)[keep], align="edge", color=color, label=label, linewidth=0)
    return int(hide.sum())


def _hist_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.histogram(values, bins=edges)[0]


# ----------------------------------------------------------------------------------------------- coverage
def plot_feature_presence(coverage: "ds.Coverage", *, show_ids: bool = False, path=None, dpi: int = 150):
    """Which participant has which feature: participants (rows, most features first) by features (columns)."""

    presence = coverage.presence()
    _need(not presence.empty, "the coverage holds no participant")
    presence = presence[feature_order(presence.columns)]
    order = presence.assign(_n=presence.sum(axis=1)).sort_values(["_n"], ascending=False, kind="stable").index
    presence = presence.loc[order]
    from matplotlib.colors import ListedColormap
    fig, axes = _figure(max(8.0, 0.28 * presence.shape[1] + 3), max(4.0, min(12.0, 0.18 * presence.shape[0] + 2)))
    ax = axes[0, 0]
    ax.imshow(presence.to_numpy(dtype=float), aspect="auto", interpolation="nearest",
              cmap=ListedColormap(["#f2f2f2", SINGLE]), vmin=0, vmax=1)
    ax.set_xticks(range(presence.shape[1]))
    ax.set_xticklabels(presence.columns, rotation=90, fontsize=8)
    for label, feature in zip(ax.get_xticklabels(), presence.columns):
        label.set_color(feature_color(feature))
    ax.set_ylabel("participants, most features first")
    if show_ids and presence.shape[0] <= 60:
        ax.set_yticks(range(presence.shape[0]))
        ax.set_yticklabels(presence.index, fontsize=7)
    else:
        ax.set_yticks([])
    _titles(ax, "Features present per participant", _caption(coverage, presence.shape[0]))
    data = presence.reset_index().melt(id_vars="RegistrationCode", var_name="feature", value_name="present")
    return _finish(fig, data, path, dpi, individual=True)


def plot_participants_per_feature(coverage: "ds.Coverage", *, min_participants: int = 1, path=None, dpi: int = 150):
    """How many participants have each feature, by category, with the share of the cohort."""

    k = _check_min(min_participants)
    table = coverage.features[["participants"]].reset_index()
    table = table[table["participants"] > 0]
    _need(not table.empty, "the coverage holds no participant")
    cohort = len(coverage.participants)
    table["category"] = table["feature"].map(feature_category)
    rank = {f: i for i, f in enumerate(feature_order(table["feature"]))}
    table = table.assign(_r=table["feature"].map(rank)).sort_values(["_r"]).drop(columns="_r")
    table["participants"] = table["participants"].astype(float)
    table["share"] = table["participants"] / cohort
    table["suppressed"] = table["participants"] < k
    table.loc[table["suppressed"], ["participants", "share"]] = np.nan
    shown = table[~table["suppressed"]]
    fig, axes = _figure(8.0, max(3.5, 0.3 * len(table) + 1.5))
    ax = axes[0, 0]
    y = np.arange(len(table))[::-1]
    position = dict(zip(table["feature"], y))
    bars = ax.barh([position[f] for f in shown["feature"]], shown["participants"],
                   color=[feature_color(f) for f in shown["feature"]])
    ax.bar_label(bars, labels=[f"{int(n):,} ({s:.0%})" for n, s in zip(shown["participants"], shown["share"])], fontsize=7, padding=2)
    ax.set_yticks(y)
    ax.set_yticklabels(table["feature"], fontsize=8)
    for label, feature in zip(ax.get_yticklabels(), table["feature"]):
        label.set_color(feature_color(feature))
    ax.set_xlabel("participants")
    ax.set_xlim(0, (shown["participants"].max() if len(shown) else 1) * 1.2)
    _titles(ax, "Participants with each feature", " · ".join(p for p in (
        _caption(coverage, cohort), _hidden(int(table["suppressed"].sum()), k, "feature")) if p))
    return _finish(fig, table, path, dpi)


def plot_features_per_participant(coverage: "ds.Coverage", *, groups=None, min_participants: int = 1, path=None,
                                  dpi: int = 150):
    """How many features participants have: every count from 1 to the maximum is drawn, including empty ones."""

    k = _check_min(min_participants)
    counts = coverage.participants["features"]
    _need(len(counts) > 0, "the coverage holds no participant")
    frame, labels, excluded = _grouped(counts.rename("features").reset_index(), groups)
    xs = list(range(1, int(frame["features"].max()) + 1))
    rows = [{"group": label, "features": x, "participants": int(((frame["group"] == label) & (frame["features"] == x)).sum())}
            for label in labels for x in xs]
    table = pd.DataFrame(rows)
    table["participants"] = table["participants"].astype(float)
    table["suppressed"] = (table["participants"] > 0) & (table["participants"] < k)
    table.loc[table["suppressed"], "participants"] = np.nan
    fig, axes = _figure(8.0, 4.5)
    ax = axes[0, 0]
    width = 0.8 / len(labels)
    for i, label in enumerate(labels):
        part = table[table["group"] == label]
        offset = (i - (len(labels) - 1) / 2) * width
        bars = ax.bar(np.array(xs) + offset, part["participants"].fillna(0), width=width,
                      color=_group_color(labels, label), label=_group_legend(label, int((frame["group"] == label).sum())))
        if len(labels) == 1:
            ax.bar_label(bars, fontsize=7)
    if len(labels) > 1:
        ax.legend(fontsize=8)
    ax.set_xticks(xs)
    ax.set_xlabel("number of features")
    ax.set_ylabel("participants")
    _titles(ax, "Features per participant", " · ".join(p for p in (
        _caption(coverage, int(frame["RegistrationCode"].nunique())), _left_out(excluded),
        _hidden(int(table["suppressed"].sum()), k, "bar")) if p))
    data = table[["features", "participants", "group", "suppressed"]]
    return _finish(fig, data, path, dpi)


def plot_data_volume(coverage: "ds.Coverage", *, by: str = "participant", min_participants: int = 1, path=None,
                     dpi: int = 150):
    """
    Stored data per participant (a histogram on a logarithmic axis, with the share of participants in each size
    band) or per feature (total bytes). Sizes are the files' bytes on disk.
    """

    _need(by in ("participant", "feature"), "by must be 'participant' or 'feature'")
    k = _check_min(min_participants)
    fig, axes = _figure(9.0, 4.8)
    ax = axes[0, 0]
    if by == "feature":
        table = coverage.features[["bytes_on_disk", "participants"]].reset_index().dropna(subset=["bytes_on_disk"])
        table = table[(table["bytes_on_disk"] > 0) & (table["participants"] >= k)]
        _need(not table.empty, "the coverage records no file size")
        table = table.sort_values("bytes_on_disk", kind="stable")
        ax.barh(table["feature"], table["bytes_on_disk"] / 1e6, color=[feature_color(f) for f in table["feature"]])
        ax.set_xscale("log")
        ax.set_xlabel("MB on disk (log scale)")
        _titles(ax, "Stored data per feature", _caption(coverage, len(coverage.participants)))
        return _finish(fig, table[["feature", "bytes_on_disk"]].iloc[::-1], path, dpi)
    sizes = coverage.participants["bytes_on_disk"].dropna().astype(float)
    sizes = sizes[sizes > 0]
    _need(len(sizes) > 0, "the coverage records no file size")
    table = sizes.rename("bytes").reset_index()
    table["band"] = pd.cut(table["bytes"], [b[0] for b in VOLUME_BANDS] + [np.inf], right=False,
                           labels=[b[2] for b in VOLUME_BANDS]).astype(str)
    edges = np.logspace(np.floor(np.log10(sizes.min())), np.ceil(np.log10(sizes.max())) + 1e-9, 30)
    hidden = _bars_masked(ax, edges, _hist_counts(sizes.to_numpy(), edges), SINGLE, min_participants=k)
    ax.set_xscale("log")
    for low, _, _ in VOLUME_BANDS:
        if low > 0 and sizes.min() <= low <= sizes.max():
            ax.axvline(low, color="0.5", linestyle=":", linewidth=1)
    shares = table["band"].value_counts(normalize=True)
    ax.text(0.99, 0.97, "\n".join(f"{label}: {shares.get(label, 0):.0%}" for _, _, label in VOLUME_BANDS),
            transform=ax.transAxes, ha="right", va="top", fontsize=8, family="monospace")
    ax.set_xlabel("bytes on disk per participant (log scale)")
    ax.set_ylabel("participants")
    _titles(ax, "Stored data per participant", " · ".join(p for p in (_caption(coverage, len(sizes)), _hidden(hidden, k, "bin")) if p))
    return _finish(fig, table, path, dpi)


# -------------------------------------------------------------------------------------- activity over time
def _count_participants(source) -> int:
    table = sm._Source(source).table("participant_feature")
    return int(table["RegistrationCode"].nunique()) if len(table) else 0


def plot_active_participants(source, *, rolling: int | None = None, min_participants: int = 1, path=None, dpi: int = 150):
    """Participants with any data on each local day; ``rolling`` adds a centred mean over that many days."""

    k = _check_min(min_participants)
    active = source.active_participants if isinstance(source, ds.DailyStatistics) else ds.read_table(source, "active_participants")
    _need(not active.empty, "the run holds no active day")
    table = active.assign(local_date=pd.to_datetime(active["local_date"])).sort_values("local_date")
    table["participants"] = table["participants"].astype(float)
    if rolling:
        _need(int(rolling) >= 2, "rolling must be at least 2 days")
        table["rolling_mean"] = table.set_index("local_date")["participants"].asfreq("D", fill_value=0).rolling(
            int(rolling), center=True).mean().reindex(table["local_date"]).to_numpy()
    table["suppressed"] = table["participants"] < k
    table.loc[table["suppressed"], [c for c in ("participants", "rolling_mean") if c in table]] = np.nan
    fig, axes = _figure(10.0, 4.5)
    ax = axes[0, 0]
    ax.plot(table["local_date"], table["participants"], color=BAND if rolling else SINGLE, linewidth=0.8, label="per day")
    if rolling:
        ax.plot(table["local_date"], table["rolling_mean"], color=SINGLE, linewidth=1.6, label=f"{rolling}-day mean")
        ax.legend(fontsize=8, loc="upper left")
    ax.set_ylabel("participants with data")
    ax.set_xlabel("local date")
    most = table["participants"].max()
    _titles(ax, "Participants with data per day", " · ".join(p for p in (
        _caption(source, _count_participants(source), f"at most {int(most):,} on one day" if most == most else ""),
        _hidden(int(table["suppressed"].sum()), k, "day")) if p))
    return _finish(fig, table, path, dpi)


def plot_feature_activity(source, *, measure: str = "participants", features: Iterable[str] | None = None,
                          min_participants: int = 1, path=None, dpi: int = 150):
    """
    Features (rows, by category) by calendar month (columns): the participants with any data that month, or the
    participant-days (``measure="participant_days"``). Every month in the span is drawn; a cell resting on fewer
    than ``min_participants`` participants is left blank.
    """

    _need(measure in ("participants", "participant_days"), "measure must be 'participants' or 'participant_days'")
    k = _check_min(min_participants)
    wanted = None if features is None else set(features)
    days_count: dict[tuple[str, pd.Period], int] = {}
    people: dict[tuple[str, pd.Period], int] = {}
    for days in sm._Source(source).grouped("participant_days"):
        months = pd.to_datetime(days["local_date"]).dt.to_period("M")
        seen: set[tuple[str, pd.Period]] = set()
        for month, text in zip(months, days["features"]):
            for feature in str(text).split(";"):
                if wanted is not None and feature not in wanted:
                    continue
                key = (feature, month)
                days_count[key] = days_count.get(key, 0) + 1
                if key not in seen:
                    seen.add(key)
                    people[key] = people.get(key, 0) + 1
    _need(bool(days_count), "the run holds no participant-day for these features")
    table = pd.DataFrame([(f, m, days_count[(f, m)], people[(f, m)]) for (f, m) in days_count],
                         columns=["feature", "month", "participant_days", "participants"])
    span = pd.period_range(table["month"].min(), table["month"].max(), freq="M")
    matrix = table.pivot_table(index="feature", columns="month", values=measure, aggfunc="sum", fill_value=0).reindex(columns=span, fill_value=0)
    persons = table.pivot_table(index="feature", columns="month", values="participants", aggfunc="sum", fill_value=0).reindex(columns=span, fill_value=0)
    order = feature_order(matrix.index)
    matrix, persons = matrix.loc[order].astype(float), persons.loc[order]
    hide = (persons > 0) & (persons < k)
    matrix = matrix.mask(hide)
    fig, axes = _figure(min(16.0, max(9.0, 0.12 * len(span) + 4)), max(3.5, 0.3 * len(matrix) + 1.5))
    ax = axes[0, 0]
    image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)), aspect="auto", interpolation="nearest", cmap="viridis")
    fig.colorbar(image, ax=ax, label=measure.replace("_", "-"))
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    for label, feature in zip(ax.get_yticklabels(), matrix.index):
        label.set_color(feature_color(feature))
    step = max(1, len(span) // 12)
    ax.set_xticks(range(0, len(span), step))
    ax.set_xticklabels([str(p) for p in span[::step]], rotation=90, fontsize=7)
    _titles(ax, f"{measure.replace('_', '-').capitalize()} per feature and month", " · ".join(p for p in (
        _caption(source, _count_participants(source)), _hidden(int(hide.to_numpy().sum()), k, "cell")) if p))
    data = matrix.reset_index().melt(id_vars="feature", var_name="month", value_name=measure)
    data["month"] = data["month"].astype(str)
    data["suppressed"] = hide.reset_index().melt(id_vars="feature", var_name="month", value_name="s")["s"].to_numpy()
    return _finish(fig, data, path, dpi)


# --------------------------------------------------------------------------------------------- daily patterns
def _hour_unit(source, feature: str) -> str:
    for daily in sm._Source(source).participants_of(feature):
        units = daily["value_unit"].dropna().astype(str).unique() if "value_unit" in daily else []
        if len(units):
            return units[0]
    return "unit not established"


def _draw_cohort(ax, table: pd.DataFrame, x: str, labels: list[str], sizes: Mapping[Any, int], *, ci: float | None,
                 marker: str | None = "o", band: bool = True, errorbars: bool = False, spread: float = 0.0) -> None:
    """Median lines per group with their interquartile band (or error bars), and bootstrap intervals when asked."""

    single = labels == [ALL_PARTICIPANTS]
    for i, label in enumerate(labels):
        part = table[table["group"] == label].sort_values(x)
        color = _group_color(labels, label)
        xs = part[x] + (i - (len(labels) - 1) / 2) * spread if spread else part[x]
        name = "median" if single else _group_legend(label, sizes.get(label, 0))
        if errorbars:
            ax.errorbar(xs, part["median"], yerr=[part["median"] - part["p25"], part["p75"] - part["median"]],
                        fmt="o-", color=color, capsize=3, label=name)
        else:
            if band:
                ax.fill_between(xs, part["p25"], part["p75"], color=BAND if single else color, alpha=0.6 if single else 0.18,
                                linewidth=0, label="interquartile range" if single else None)
            ax.plot(xs, part["median"], color=color, marker=marker, markersize=3, linewidth=1.2 if marker else 0.9, label=name)
        if ci is not None and "ci_low" in part:
            ax.errorbar(xs, part["median"], yerr=[part["median"] - part["ci_low"], part["ci_high"] - part["median"]],
                        fmt="none", ecolor=color, elinewidth=2.2, alpha=0.55, capsize=0,
                        label=f"{ci:.0%} interval of the median" if (single or i == 0) else None)


def plot_hour_of_day(source, feature: str, *, measure: str = "value", groups=None, min_participants: int = 1,
                     ci: float | None = None, seed: int = 0, path=None, dpi: int = 150):
    """
    The cohort's median, with interquartile band, by local hour of day, over every participant with the feature:
    the mean value for levels, the average amount per day for totals and event amounts (``measure="value"``),
    records per day (``measure="records"``), or the share of the participant's days with data in that hour
    (``measure="coverage"``). The strip below counts the participants behind each hour.
    """

    _need(measure in ("value", "records", "coverage"), "measure must be 'value', 'records' or 'coverage'")
    k = _check_min(min_participants)
    participants, _ = sm.hour_of_day(source)
    _need(len(participants) > 0 and feature in set(participants["feature"]), f"no hour-of-day data for {feature}")
    rows = participants[participants["feature"] == feature]
    kind = ds.measurement_kind(feature)
    if measure == "records":
        column, label = "records_per_day", "records per day"
    elif measure == "coverage":
        column, label = "day_share", "share of days with data in the hour"
        _need(rows[column].notna().any(), "this run has no hourly days (written before output schema statistics-4); compute it again")
    elif kind in ("extensive_total", "event_amount"):
        column, label = "value_per_day", f"amount per day ({_hour_unit(source, feature)})"
    else:
        column, label = "value_mean", f"mean value ({_hour_unit(source, feature)})"
    _need(rows[column].notna().any(), f"{feature} has no {measure} by hour of day")
    frame, labels, excluded = _grouped(rows[["RegistrationCode", "hour", column]], groups)
    table = _cohort(frame, "hour", column, k, ci, seed)
    sizes = frame.groupby("group")["RegistrationCode"].nunique().to_dict()
    fig, ax, strip = _figure_with_strip(9.0, 5.2)
    _draw_cohort(ax, table, "hour", labels, sizes, ci=ci)
    _strip(strip, table, "hour", labels)
    strip.set_xticks(range(0, 24, 2))
    strip.set_xlabel("local hour of day")
    ax.set_ylabel(label)
    if measure == "coverage":
        ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    _titles(ax, f"{feature} by hour of day", " · ".join(p for p in (
        _caption(source, int(frame["RegistrationCode"].nunique())), _left_out(excluded),
        _hidden(int(table["suppressed"].sum()), k)) if p))
    return _finish(fig, table.assign(feature=feature, measure=measure), path, dpi)


def _pattern_rows(patterns: "sm.TemporalPatterns", feature: str, metric: str | None, by: str) -> tuple[pd.DataFrame, str]:
    table = patterns.day_of_week if by == "weekday" else patterns.month_of_year
    rows = table[table["feature"] == feature]
    _need(len(rows) > 0, f"no {by} pattern for {feature}")
    if metric is None:  # the headline metric, chosen by measurement kind, never whichever sorts first
        ranked = sm.headline_metrics(feature, list(dict.fromkeys(rows["metric"])))
        _need(bool(ranked), f"{feature} has no metric to plot")
        metric = ranked[0]
    rows = rows[rows["metric"] == metric]
    _need(len(rows) > 0, f"no {by} pattern for {feature} {metric}")
    return rows, metric


def _plot_pattern(patterns, feature, metric, by, groups, min_participants, ci, seed, path, dpi):
    k = _check_min(min_participants)
    rows, metric = _pattern_rows(patterns, feature, metric, by)
    frame, labels, excluded = _grouped(rows[["RegistrationCode", by, "median"]].rename(columns={"median": "value"}), groups)
    table = _cohort(frame, by, "value", k, ci, seed)
    sizes = frame.groupby("group")["RegistrationCode"].nunique().to_dict()
    fig, ax, strip = _figure_with_strip(8.0, 5.0)
    if by == "weekday":
        for day in patterns.weekend_days:
            ax.axvspan(day - 0.5, day + 0.5, color="0.92", zorder=0)
    _draw_cohort(ax, table, by, labels, sizes, ci=ci, errorbars=True, spread=0.08 if len(labels) > 1 else 0.0)
    _strip(strip, table, by, labels)
    if by == "weekday":
        strip.set_xticks(range(7))
        strip.set_xticklabels(WEEKDAY_NAMES)
    else:
        strip.set_xticks(range(1, 13))
        strip.set_xticklabels(MONTH_NAMES)
    ax.set_ylabel(f"{metric} (median of participants' medians)")
    if len(labels) > 1 or ci is not None:
        ax.legend(fontsize=8)
    title = f"{feature} by day of week (weekend shaded)" if by == "weekday" else f"{feature} by month of year"
    _titles(ax, title, " · ".join(p for p in (_participants(int(frame["RegistrationCode"].nunique())), _left_out(excluded),
                                               _hidden(int(table["suppressed"].sum()), k)) if p))
    return _finish(fig, table.assign(feature=feature, metric=metric), path, dpi)


def plot_weekly_pattern(patterns: "sm.TemporalPatterns", feature: str, metric: str | None = None, *, groups=None,
                        min_participants: int = 1, ci: float | None = None, seed: int = 0, path=None, dpi: int = 150):
    """The cohort's median of participants' medians by day of week, with interquartile range; weekend shaded."""

    return _plot_pattern(patterns, feature, metric, "weekday", groups, min_participants, ci, seed, path, dpi)


def plot_monthly_pattern(patterns: "sm.TemporalPatterns", feature: str, metric: str | None = None, *, groups=None,
                         min_participants: int = 1, ci: float | None = None, seed: int = 0, path=None, dpi: int = 150):
    """The cohort's median of participants' medians by month of year, with interquartile range."""

    return _plot_pattern(patterns, feature, metric, "month", groups, min_participants, ci, seed, path, dpi)


# -------------------------------------------------------------------------------------------------- values
def _daily_rows(source, feature: str, metric: str | None, valid_only: bool, rules,
                participant: str | None) -> tuple[pd.DataFrame, str, str]:
    """(rows with local_date, RegistrationCode, value, valid), the metric and its unit."""

    src = sm._Source(source)
    _need(feature in src.features, f"{feature} is not among the run's features")
    cadence = {(r["feature"], r["RegistrationCode"]): r for r in src.table("participant_feature").to_dict("records")}
    rule = (rules or {}).get(feature, sm.DEFAULT_RULES.get(feature, sm.DEFAULT_RULE))
    wanted = None if participant is None else _code(participant)
    frames: list[pd.DataFrame] = []
    chosen, unit = metric, None
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
    rows["local_date"] = pd.to_datetime(rows["local_date"])
    return rows, str(chosen), str(unit)


def plot_daily_values(source, feature: str, participant: str | None = None, *, metric: str | None = None,
                      valid_only: bool = False, rules=None, time_axis: str | None = None, groups=None,
                      min_participants: int = 1, show_ids: bool = False, path=None, dpi: int = 150):
    """
    One participant's daily headline metric over time, invalid days drawn hollow, on study days by default (so the
    real dates stay off the figure unless ``time_axis="calendar"``); or, with ``participant=None``, the cohort's median
    per day with interquartile band over valid days, by calendar date or by study day, with the participants behind
    each day in the strip below.
    """

    time_axis = time_axis or ("study" if participant is not None else "calendar")
    _need(time_axis in ("calendar", "study"), "time_axis must be 'calendar' or 'study'")
    k = _check_min(min_participants)
    rows, metric, unit = _daily_rows(source, feature, metric, valid_only or participant is None, rules, participant)
    rows["study_day"] = _study_day(rows)
    x = "local_date" if time_axis == "calendar" else "study_day"
    xlabel = "local date" if time_axis == "calendar" else "days since the first day with data"
    if participant is not None:
        table = rows.sort_values(x)
        fig, axes = _figure(10.0, 4.5)
        ax = axes[0, 0]
        valid, invalid = table[table["valid"]], table[~table["valid"]]
        ax.plot(table[x], table["value"], color="0.7", linewidth=0.6)
        ax.plot(valid[x], valid["value"], "o", color=SINGLE, markersize=3, label="valid day")
        if len(invalid):
            ax.plot(invalid[x], invalid["value"], "o", markerfacecolor="none", color=SINGLE, markersize=3, label="other day")
        ax.legend(fontsize=8)
        who = table["RegistrationCode"].iloc[0] if show_ids else "one participant"
        ax.set_ylabel(f"{metric} ({unit})")
        ax.set_xlabel(xlabel)
        _titles(ax, f"{feature}: daily {metric}", f"{who} · {len(valid):,} valid of {len(table):,} days")
        columns = ["RegistrationCode", x, "value", "valid"] + (["study_day"] if x == "local_date" else [])
        return _finish(fig, table[columns], path, dpi, individual=True)
    frame, labels, excluded = _grouped(rows[["RegistrationCode", x, "value"]], groups)
    table = _cohort(frame, x, "value", k, None, 0)
    # Every day in the span, so that the lines break where no participant has data instead of bridging the gap.
    full = (pd.date_range(table[x].min(), table[x].max(), freq="D") if x == "local_date"
            else np.arange(int(table[x].min()), int(table[x].max()) + 1))
    table = pd.concat([part.set_index(x).reindex(full).rename_axis(x).reset_index().assign(group=label)
                       for label, part in table.groupby("group", sort=False)], ignore_index=True)
    empty = table["suppressed"].isna()
    table.loc[empty, "participants"] = 0.0
    table["suppressed"] = table["suppressed"].eq(True)
    sizes = frame.groupby("group")["RegistrationCode"].nunique().to_dict()
    fig, ax, strip = _figure_with_strip(10.0, 5.2)
    _draw_cohort(ax, table, x, labels, sizes, ci=None, marker=None)
    _strip(strip, table, x, labels, width=1.0)
    ax.legend(fontsize=8)
    ax.set_ylabel(f"{metric} ({unit})")
    strip.set_xlabel(xlabel)
    _titles(ax, f"{feature}: daily {metric}", " · ".join(p for p in (
        _caption(source, int(frame["RegistrationCode"].nunique()), "valid days"), _left_out(excluded),
        _hidden(int(table["suppressed"].sum()), k, "day")) if p))
    return _finish(fig, table.assign(feature=feature, metric=metric), path, dpi)


def plot_monthly_distribution(source, feature: str, *, metric: str | None = None, valid_only: bool = True, rules=None,
                              period: str = "auto", unit: str = "participant", groups=None, min_participants: int = 1,
                              clip_quantile: float | None = None, path=None, dpi: int = 150):
    """
    Boxplots per calendar month or quarter: the median, quartiles, and whiskers at 1.5 times the interquartile range.
    With ``unit="participant"`` (the default) each participant counts once per period, by their median day, so that
    participants with many days do not dominate; ``unit="participant_day"`` pools every participant-day. Every period
    in the span is drawn, so gaps stay visible; ``period`` is ``"month"``, ``"quarter"``, or ``"auto"`` (months up to 36
    of them, quarters beyond, as the title states). Nothing is clipped unless ``clip_quantile`` is given; then the axis
    stops at that quantile and the number of values beyond it is stated on the figure.
    """

    _need(period in ("auto", "month", "quarter"), "period must be 'auto', 'month' or 'quarter'")
    _need(unit in ("participant", "participant_day"), "unit must be 'participant' or 'participant_day'")
    k = _check_min(min_participants)
    rows, metric, unit_label = _daily_rows(source, feature, metric, valid_only, rules, None)
    dates = rows["local_date"]
    span_months = len(pd.period_range(dates.min().to_period("M"), dates.max().to_period("M"), freq="M"))
    freq = "M" if period == "month" or (period == "auto" and span_months <= 36) else "Q"
    rows = rows.assign(period=dates.dt.to_period(freq))
    periods = pd.period_range(rows["period"].min(), rows["period"].max(), freq=freq)
    days_per_period = rows.groupby("period").size().reindex(periods, fill_value=0)
    values = (rows.groupby(["RegistrationCode", "period"])["value"].median().reset_index() if unit == "participant"
              else rows[["RegistrationCode", "period", "value"]])
    frame, labels, excluded = _grouped(values, groups)
    records, boxes = [], []
    for label in labels:
        part = frame[frame["group"] == label]
        for i, p in enumerate(periods):
            v = part.loc[part["period"] == p, "value"].to_numpy(dtype=float)
            n = int(part.loc[part["period"] == p, "RegistrationCode"].nunique())
            hide = 0 < n < k
            records.append({"group": label, "period": str(p), "participants": np.nan if hide else float(n),
                            "values": np.nan if hide else float(len(v)), "n_days": int(days_per_period.iloc[i]),
                            "median": np.nan if hide or not len(v) else float(np.median(v)),
                            "p25": np.nan if hide or not len(v) else float(np.quantile(v, 0.25)),
                            "p75": np.nan if hide or not len(v) else float(np.quantile(v, 0.75)), "suppressed": hide})
            boxes.append((label, i, None if hide or not len(v) else v))
    table = pd.DataFrame(records)
    fig, ax, strip = _figure_with_strip(min(16.0, max(8.0, 0.35 * len(periods) + 3)), 5.6)
    width = 0.6 / len(labels)
    for g, label in enumerate(labels):
        chosen = [(i, v) for lab, i, v in boxes if lab == label and v is not None]
        if not chosen:
            continue
        offset = (g - (len(labels) - 1) / 2) * width
        parts = ax.boxplot([v for _, v in chosen], positions=[i + offset for i, _ in chosen], widths=width * 0.9,
                           showfliers=True, flierprops={"markersize": 2, "alpha": 0.4}, patch_artist=True)
        color = _group_color(labels, label, "white")
        for box in parts["boxes"]:
            box.set_facecolor(color if len(labels) > 1 else "white")
            box.set_alpha(0.8)
        if len(labels) > 1:
            parts["boxes"][0].set_label(_group_legend(label, int(frame.loc[frame["group"] == label, "RegistrationCode"].nunique())))
    strip_table = table.assign(position=table["period"].map({str(p): i for i, p in enumerate(periods)}))
    _strip(strip, strip_table, "position", labels)
    step = max(1, len(periods) // 24)
    strip.set_xticks(range(0, len(periods), step))
    strip.set_xticklabels([str(p) for p in periods[::step]], rotation=90, fontsize=7)
    ax.set_xlim(-0.7, len(periods) - 0.3)
    how = "one value per participant and period, their median day" if unit == "participant" else "every participant-day"
    notes = [f"{frame['RegistrationCode'].nunique():,} participants · {len(rows):,} {'valid ' if valid_only else ''}days", how,
             _left_out(excluded), _hidden(int(table["suppressed"].sum()), k, "box")]
    if clip_quantile is not None:
        _need(0 < clip_quantile < 1, "clip_quantile must be between 0 and 1")
        plotted = frame["value"].to_numpy(dtype=float)
        limit = float(np.quantile(plotted, clip_quantile))
        beyond = int((plotted > limit).sum())
        ax.set_ylim(top=limit)
        notes.append(f"axis clipped at the {clip_quantile:.0%} quantile: {beyond:,} values above")
    if len(labels) > 1:
        ax.legend(fontsize=8)
    ax.set_ylabel(f"{metric} ({unit_label})")
    _titles(ax, f"{feature}: daily {metric} by {'month' if freq == 'M' else 'quarter'}", " · ".join(n for n in notes if n))
    return _finish(fig, table, path, dpi)


# ------------------------------------------------------------------------------------------------ adherence
def _histogram_panel(ax, frame: pd.DataFrame, column: str, edges: np.ndarray, labels: list[str], k: int) -> int:
    hidden = 0
    for label in labels:
        part = frame[frame["group"] == label]
        counts = _hist_counts(part[column].to_numpy(dtype=float), edges)
        if labels == [ALL_PARTICIPANTS]:
            hidden += _bars_masked(ax, edges, counts, SINGLE, min_participants=k)
        else:
            shown = np.where((counts > 0) & (counts < k), np.nan, counts)
            hidden += int(((counts > 0) & (counts < k)).sum())
            ax.stairs(np.nan_to_num(shown), edges, color=_group_color(labels, label), linewidth=1.6,
                      label=_group_legend(label, int(part["RegistrationCode"].nunique())))
    return hidden


def plot_adherence(summaries: "sm.Summaries", feature: str, *, groups=None, min_participants: int = 1, path=None,
                   dpi: int = 150):
    """Valid days per participant, and adherence (valid days over the follow-up span), for one feature."""

    k = _check_min(min_participants)
    table = summaries.adherence[summaries.adherence["feature"] == feature]
    _need(len(table) > 0, f"no adherence for {feature}")
    frame, labels, excluded = _grouped(table, groups)
    fig, axes = _figure(10.0, 4.2, ncols=2)
    left, right = axes[0, 0], axes[0, 1]
    edges = np.histogram_bin_edges(frame["valid_days"].to_numpy(dtype=float), bins=min(30, max(5, len(frame))))
    hidden = _histogram_panel(left, frame, "valid_days", edges, labels, k)
    left.set_xlabel("valid days")
    left.set_ylabel("participants")
    hidden += _histogram_panel(right, frame, "adherence", np.linspace(0, 1, 21), labels, k)
    right.set_xlabel("adherence (valid days / follow-up days)")
    if len(labels) > 1:
        left.legend(fontsize=7)
    _titles(left, f"{feature}: valid days", " · ".join(p for p in (_participants(int(frame["RegistrationCode"].nunique())),
                                                                   _left_out(excluded), _hidden(hidden, k, "bin")) if p))
    _titles(right, "Adherence", str(table["rule"].iloc[0]) if "rule" in table else "")
    return _finish(fig, frame, path, dpi)


def plot_retention(summaries: "sm.Summaries", features: Iterable[str] | None = None, *, groups=None,
                   min_participants: int = 1, path=None, dpi: int = 150):
    """
    The share of participants still contributing valid days, by days since their first valid day: one line per
    feature, or with ``groups`` one line per group for a single feature. A point resting on fewer than
    ``min_participants`` participants is not drawn.
    """

    k = _check_min(min_participants)
    _need(len(summaries.retention) > 0, "the summaries hold no retention")
    chosen = list(dict.fromkeys(summaries.retention["feature"])) if features is None else list(features)
    if groups is not None:
        _need(len(chosen) == 1, "retention by group is drawn for one feature at a time")
        adherence = summaries.adherence[summaries.adherence["feature"] == chosen[0]]
        frame, labels, excluded = _grouped(adherence, groups)
        table = pd.concat([sm.retention(frame[frame["group"] == label].drop(columns="group")).assign(group=label)
                           for label in labels], ignore_index=True)
    else:
        table = summaries.retention[summaries.retention["feature"].isin(chosen)].assign(group=ALL_PARTICIPANTS)
        labels, excluded = [ALL_PARTICIPANTS], 0
    _need(len(table) > 0, "no retention for these features")
    table = table.copy()
    table["participants"] = table["participants"].astype(float)
    table["suppressed"] = (table["participants"] > 0) & (table["participants"] < k)
    table.loc[table["suppressed"], ["participants", "fraction"]] = np.nan
    fig, axes = _figure(9.0, 5.0)
    ax = axes[0, 0]
    if groups is None:
        for feature, rows in table.groupby("feature", sort=False):
            first = rows["participants"].dropna()
            ax.step(rows["days_since_first_valid"], rows["fraction"], where="post", color=feature_color(feature),
                    label=f"{feature} (n={int(first.iloc[0]) if len(first) else 0})")
    else:
        for label, rows in table.groupby("group", sort=True):
            first = rows["participants"].dropna()
            ax.step(rows["days_since_first_valid"], rows["fraction"], where="post", color=_group_color(labels, str(label)),
                    label=_group_legend(str(label), int(first.iloc[0]) if len(first) else 0))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("days since first valid day")
    ax.set_ylabel("share still contributing")
    ax.legend(fontsize=7, ncol=2 if len(chosen) > 8 else 1)
    caption = f"{len(chosen)} feature{'s' if len(chosen) != 1 else ''}" if groups is None else chosen[0]
    _titles(ax, "Retention", " · ".join(p for p in (caption, _left_out(excluded), _hidden(int(table["suppressed"].sum()), k)) if p))
    return _finish(fig, table, path, dpi)


# -------------------------------------------------------------------------------------------------- context
def plot_co_availability(source, features: Iterable[str] | None = None, *, path=None, dpi: int = 150):
    """For each pair of features, the share of their participant-days they share (the Jaccard index)."""

    overlap = sm.day_overlap(source)
    _need(len(overlap) > 0, "the run holds no participant-day")
    names = feature_order(set(overlap["feature_a"]) if features is None else set(features))
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
    for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
        for label, feature in zip(labels, names):
            label.set_color(feature_color(feature))
    if len(names) <= 12:
        cells = matrix.to_numpy(dtype=float)
        for i in range(len(names)):
            for j in range(len(names)):
                value = cells[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if value < 0.6 else "black")
    _titles(ax, "Days shared by pairs of features", _caption(source, _count_participants(source)))
    data = matrix.rename_axis("feature_a").reset_index().melt(id_vars="feature_a", var_name="feature_b", value_name="jaccard")
    return _finish(fig, data, path, dpi)


def _name_colors(names: Iterable[str]) -> dict[str, str]:
    """A stable color per category name (such as an acquisition method), the same in every figure."""

    return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(sorted(set(map(str, names))))}


def _stacked_shares(table: pd.DataFrame, category: str, title: str, caption: str, path, dpi):
    shares = table.pivot_table(index="feature", columns=category, values="share", aggfunc="sum", fill_value=0.0)
    shares = shares.loc[list(reversed(feature_order(shares.index)))]
    order = shares.sum().sort_values(ascending=False, kind="stable").index
    colors = _name_colors(shares.columns)
    fig, axes = _figure(9.0, max(3.5, 0.3 * len(shares) + 1.5))
    ax = axes[0, 0]
    left = np.zeros(len(shares))
    for name in order:
        values = shares[name].to_numpy()
        ax.barh(shares.index, values, left=left, label=str(name), color=colors[str(name)])
        left += values
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of records")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    for label, feature in zip(ax.get_yticklabels(), shares.index):
        label.set_color(feature_color(feature))
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
def plot_sleep(metrics: "dm.DomainMetrics", *, rule: "dm.NightRule" = dm.NightRule(), unit: str = "participant",
               groups=None, min_participants: int = 1, path=None, dpi: int = 150):
    """
    Over valid nights: total sleep time, the clock time of the sleep midpoint, and sleep efficiency. With
    ``unit="participant"`` (the default) each participant counts once, by their median night; ``unit="night"`` pools
    every valid night.
    """

    _need(unit in ("participant", "night"), "unit must be 'participant' or 'night'")
    k = _check_min(min_participants)
    nights = metrics.sleep_nights
    valid = nights[nights["asleep_recorded"].astype(bool) & (nights["asleep_minutes"] >= rule.min_asleep_minutes)]
    _need(len(valid) > 0, "no valid night: " + rule.describe())
    columns = ["asleep_minutes", "midpoint_hours_after_noon", "efficiency"]
    if unit == "participant":
        values = valid.groupby("RegistrationCode")[columns].median().join(
            valid.groupby("RegistrationCode").size().rename("nights")).reset_index()
    else:
        values = valid[["RegistrationCode", "night_date", *columns]]
    frame, labels, excluded = _grouped(values, groups)
    fig, axes = _figure(12.0, 4.2, ncols=3)
    hours = frame["asleep_minutes"] / 60
    midpoint = frame["midpoint_hours_after_noon"]
    efficiency = frame["efficiency"].dropna()
    panels = [(axes[0, 0], frame.assign(v=hours), np.arange(np.floor(hours.min()), np.ceil(hours.max()) + 0.5, 0.5)),
              (axes[0, 1], frame.assign(v=midpoint), np.arange(np.floor(midpoint.min()), np.ceil(midpoint.max()) + 0.5, 0.5)),
              (axes[0, 2], frame.assign(v=frame["efficiency"]),
               np.linspace(max(0.0, (efficiency.min() if len(efficiency) else 0.5) - 0.02), 1.0, 25))]
    hidden = 0
    for ax, table, edges in panels:
        hidden += _histogram_panel(ax, table.dropna(subset=["v"]), "v", edges, labels, k)
    axes[0, 0].set_xlabel("total sleep time (h)")
    axes[0, 0].set_ylabel("participants" if unit == "participant" else "nights")
    ticks = np.arange(np.floor(midpoint.min()), np.ceil(midpoint.max()) + 1)
    axes[0, 1].set_xticks(ticks)
    axes[0, 1].set_xticklabels([f"{int((12 + t) % 24):02d}:00" for t in ticks], rotation=90, fontsize=7)
    axes[0, 1].set_xlabel("sleep midpoint (local clock time)")
    axes[0, 2].set_xlabel("sleep efficiency")
    if len(labels) > 1:
        axes[0, 0].legend(fontsize=7)
    how = "each participant's median night" if unit == "participant" else "every valid night"
    _titles(axes[0, 0], "Total sleep time", f"{len(valid):,} valid nights")
    _titles(axes[0, 1], "Sleep midpoint", " · ".join(p for p in (_participants(int(frame["RegistrationCode"].nunique())), how) if p))
    _titles(axes[0, 2], "Efficiency", " · ".join(p for p in (rule.describe(), _left_out(excluded), _hidden(hidden, k, "bin")) if p))
    return _finish(fig, frame, path, dpi)


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
    for name, color in GLUCOSE_COLORS.items():
        values = chosen[name].to_numpy(dtype=float)
        ax.barh(labels, values, left=left, color=color, label=GLUCOSE_LABELS[name])
        left += values
    ax.axvline(TIME_IN_RANGE_TARGET, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of readings (valid days pooled)")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _titles(ax, "Time in glucose ranges", f"{_participants(len(chosen))} · target > {TIME_IN_RANGE_TARGET:.0f}% in range")
    columns = ["RegistrationCode", "valid_days", "readings", *GLUCOSE_COLORS]
    return _finish(fig, chosen[columns], path, dpi, individual=True)


def bmi_categories(summaries: "sm.Summaries", *, groups=None) -> pd.DataFrame:
    """Participants per WHO adult BMI class, by each participant's median BMI over valid days (per group if given)."""

    metrics = summaries.participant_metrics
    bmi = metrics[(metrics["feature"] == "BMI") & (metrics["metric"] == "value_mean")].dropna(subset=["median"])
    frame, labels, _ = _grouped(bmi, groups) if len(bmi) else (bmi.assign(group=ALL_PARTICIPANTS), [ALL_PARTICIPANTS], 0)
    rows = []
    for label in labels:
        part = frame[frame["group"] == label]
        total = len(part)
        for name, lower, upper in BMI_CLASSES:
            n = int(((part["median"] >= lower) & (part["median"] < upper)).sum())
            rows.append({"group": label, "category": name, "lower": lower, "upper": upper, "participants": n,
                         "share": n / total if total else np.nan})
    return pd.DataFrame(rows)


def _bmi_range(lower: float, upper: float) -> str:
    if lower == 0:
        return f"< {upper:g}"
    if np.isinf(upper):
        return f">= {lower:g}"
    return f"{lower:g} to < {upper:g}"


def plot_bmi_categories(summaries: "sm.Summaries", *, groups=None, min_participants: int = 1, path=None, dpi: int = 150):
    """Participants per WHO adult BMI class, from each participant's median BMI over valid days."""

    k = _check_min(min_participants)
    table = bmi_categories(summaries, groups=groups)
    _need(table["participants"].sum() > 0, "no participant with a BMI value")
    table = table.copy()
    table["participants"] = table["participants"].astype(float)
    table["suppressed"] = (table["participants"] > 0) & (table["participants"] < k)
    table.loc[table["suppressed"], ["participants", "share"]] = np.nan
    labels = list(dict.fromkeys(table["group"]))
    fig, axes = _figure(9.0, 4.5)
    ax = axes[0, 0]
    names = [f"{n}\n{_bmi_range(l, u)}" for n, l, u in BMI_CLASSES]
    width = 0.8 / len(labels)
    for i, label in enumerate(labels):
        part = table[table["group"] == label]
        offset = (i - (len(labels) - 1) / 2) * width
        bars = ax.bar(np.arange(len(names)) + offset, part["participants"].fillna(0), width=width,
                      color=_group_color(labels, label), label=None if len(labels) == 1 else label)
        if len(labels) == 1:
            ax.bar_label(bars, labels=["" if s else f"{int(n)} ({sh:.0%})" for n, sh, s in
                                       zip(part["participants"].fillna(0), part["share"].fillna(0), part["suppressed"])], fontsize=7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("participants")
    if len(labels) > 1:
        ax.legend(fontsize=8)
    _titles(ax, "BMI categories (WHO, kg/m2)", " · ".join(p for p in (
        _participants(int(table["participants"].sum())), _hidden(int(table["suppressed"].sum()), k, "bar")) if p))
    return _finish(fig, table, path, dpi)


# ------------------------------------------------------------------------------------- trust: valid days
def _criterion(rule: "sm.ValidDayRule") -> str:
    if rule.min_completeness is not None:
        return "completeness"
    if rule.min_hours_with_data is not None:
        return "hours_with_data"
    return "records"


def _rule_threshold(rule: "sm.ValidDayRule", criterion: str) -> float:
    return {"completeness": rule.min_completeness, "hours_with_data": rule.min_hours_with_data,
            "records": rule.min_records}[criterion]


def _valid_at(days: pd.DataFrame, rule: "sm.ValidDayRule", criterion: str, threshold: float) -> np.ndarray:
    """
    Whether each day is valid under ``rule`` with the criterion's threshold set to ``threshold``: the same test as
    ``data_summaries.mark_valid_days``, in which a completeness threshold applies only where a cadence is expected.
    """

    ok = days["records"].to_numpy(dtype=float) >= (threshold if criterion == "records" else rule.min_records)
    hours = threshold if criterion == "hours_with_data" else rule.min_hours_with_data
    if hours is not None and "hours_with_data" in days:
        ok &= days["hours_with_data"].fillna(0).to_numpy(dtype=float) >= hours
    completeness = threshold if criterion == "completeness" else rule.min_completeness
    if completeness is not None:
        expected = days["_expected"].to_numpy(dtype=float)
        applies = np.isfinite(expected) & (expected > 0)
        ok &= ~applies | (days["completeness"].to_numpy(dtype=float) >= completeness)
    return ok


def plot_valid_day_rule(source, feature: str, *, rule: "sm.ValidDayRule | None" = None,
                        thresholds: Iterable[float] | None = None, min_valid_days: Iterable[int] = (1, 7, 14, 30),
                        min_participants: int = 1, path=None, dpi: int = 150):
    """
    How a valid-day rule shapes the data, before committing to it. Left: the distribution of the rule's criterion
    (hours with data, completeness or records) over participant-days, with the rule's threshold and the share of days
    it keeps. Right: as that threshold varies, the share of participant-days kept, and the share of participants
    keeping at least k valid days for each k in ``min_valid_days``. A completeness criterion governs only participants
    with a fixed cadence (``data_summaries.FIXED_CADENCE``), and the figure says how many.
    """

    k = _check_min(min_participants)
    src = sm._Source(source)
    _need(feature in src.features, f"{feature} is not among the run's features")
    valid_rule = rule if rule is not None else sm.DEFAULT_RULES.get(feature, sm.DEFAULT_RULE)
    criterion = _criterion(valid_rule)
    cadence = {(r["feature"], r["RegistrationCode"]): r for r in src.table("participant_feature").to_dict("records")}
    frames = []
    for daily in src.participants_of(feature):
        code = str(daily["RegistrationCode"].iloc[0])
        c = cadence.get((feature, code), {})
        expected = sm.expected_records_per_day(feature, c.get("typical_gap_minutes"), c.get("regular_share"))
        marked = sm.mark_valid_days(feature, daily, valid_rule, expected)
        keep = ["RegistrationCode", "local_date", "records", *(["hours_with_data"] if "hours_with_data" in marked else []),
                "completeness", "valid"]
        frames.append(marked[keep].assign(_expected=float(expected) if expected else np.nan))
    _need(bool(frames), f"no daily rows for {feature}")
    days = pd.concat(frames, ignore_index=True)
    if criterion == "hours_with_data" and "hours_with_data" not in days:
        criterion = "records"  # a day key has no hours; the rule then rests on records alone
    governed = days[np.isfinite(days["_expected"].to_numpy(dtype=float))] if criterion == "completeness" else days
    _need(len(governed) > 0, f"no {feature} participant has the fixed cadence a completeness rule applies to")
    values = governed[criterion].fillna(0).to_numpy(dtype=float)
    if thresholds is None:
        if criterion == "hours_with_data":
            thresholds = np.arange(0, 25)
        elif criterion == "completeness":
            thresholds = np.round(np.linspace(0, 1, 21), 2)
        else:
            thresholds = np.unique(np.round(np.geomspace(1, max(2.0, float(np.quantile(values, 0.99))), 30)))
    thresholds = np.asarray(sorted(set(float(t) for t in thresholds)))
    at_rule = _rule_threshold(valid_rule, criterion)
    people = int(days["RegistrationCode"].nunique())
    ks = sorted(set(int(v) for v in min_valid_days))
    rows = []
    for t in thresholds:
        ok = _valid_at(days, valid_rule, criterion, t)
        per = pd.Series(ok).groupby(days["RegistrationCode"].to_numpy()).sum()
        rows.append({"threshold": t, "measure": "participant-days kept", "k": np.nan, "count": float(ok.sum()),
                     "share": float(ok.mean())})
        for kk in ks:
            n = int((per >= kk).sum())
            rows.append({"threshold": t, "measure": f"participants with at least {kk} valid days", "k": kk,
                         "count": float(n), "share": n / people})
    table = pd.DataFrame(rows)
    table["suppressed"] = table["k"].notna() & (table["count"] > 0) & (table["count"] < k)
    table.loc[table["suppressed"], ["count", "share"]] = np.nan
    fig, axes = _figure(12.0, 4.6, ncols=2)
    left, right = axes[0, 0], axes[0, 1]
    if criterion == "hours_with_data":
        edges = np.arange(-0.5, 25.0, 1.0)
    elif criterion == "completeness":
        edges = np.linspace(0, 1, 21)
    else:
        edges = np.unique(np.geomspace(max(1.0, values.min() if values.size else 1.0), max(2.0, values.max() + 1), 30))
    bins = np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)
    counts = np.bincount(bins, minlength=len(edges) - 1)
    persons = pd.Series(governed["RegistrationCode"].to_numpy()).groupby(bins).nunique().reindex(range(len(edges) - 1), fill_value=0).to_numpy()
    hidden = _bars_masked(left, edges, counts, SINGLE, participants=persons, min_participants=k)
    if criterion == "records":
        left.set_xscale("log")
    kept_share = float(_valid_at(days, valid_rule, criterion, at_rule).mean())
    left.axvline(at_rule, color="black", linestyle="--", linewidth=1.2)
    left.set_xlabel({"hours_with_data": "hours with data in the day", "completeness": "completeness of the day",
                     "records": "records in the day (log scale)"}[criterion])
    left.set_ylabel("participant-days")
    kept = table[table["measure"] == "participant-days kept"]
    right.plot(kept["threshold"], kept["share"], color="black", linestyle="--", label="participant-days kept")
    for i, kk in enumerate(ks):
        part = table[table["k"] == kk]
        right.plot(part["threshold"], part["share"], color=PALETTE[i % len(PALETTE)], marker="o", markersize=2.5,
                   label=f"participants with >= {kk} valid days")
    right.axvline(at_rule, color="black", linestyle=":", linewidth=1)
    right.set_ylim(0, 1.02)
    right.set_xlabel(f"threshold on {criterion.replace('_', ' ')}")
    right.set_ylabel("share")
    right.legend(fontsize=7)
    if criterion == "records":
        right.set_xscale("log")
    governed_people = int(governed["RegistrationCode"].nunique())
    note = (f"the criterion governs the {governed_people:,} of {people:,} participants with a fixed cadence"
            if criterion == "completeness" else "")
    _titles(left, f"{feature}: the valid-day rule", " · ".join(p for p in (
        valid_rule.describe(), f"the rule keeps {kept_share:.0%} of days", _hidden(hidden, k, "bin")) if p))
    _titles(right, "Sensitivity to the threshold", " · ".join(p for p in (_caption(source, people), note,
                                                                           _hidden(int(table["suppressed"].sum()), k)) if p))
    distribution = governed[["RegistrationCode", "local_date", criterion, "valid"]].rename(columns={criterion: "criterion"})
    return _finish(fig, table.assign(feature=feature, criterion=criterion), path, dpi,
                   panels={"distribution": distribution, "sensitivity": table})


# --------------------------------------------------------------------------------- trust: cadence and hours
def plot_sampling_cadence(source, features: Iterable[str] | None = None, *, path=None, dpi: int = 150):
    """
    Each participant's sampling cadence per feature: the typical gap between records (logarithmic axis) against
    the share of gaps close to it. Separate clusters reveal different devices or ways of measuring, such as a CGM's
    5 minutes against a finger-stick's hours, which one cohort median would blend.
    """

    table = sm._Source(source).table("participant_feature")
    chosen = feature_order(table["feature"].unique() if features is None else features)
    rows = table[table["feature"].isin(chosen)].dropna(subset=["typical_gap_minutes"])
    _need(len(rows) > 0, "no participant has a cadence for these features")
    fig, axes = _figure(10.0, 5.2)
    ax = axes[0, 0]
    for feature in chosen:
        part = rows[rows["feature"] == feature]
        if len(part):
            ax.scatter(part["typical_gap_minutes"], part["regular_share"], s=18, alpha=0.75, color=feature_color(feature),
                       label=f"{feature} ({len(part):,})")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    for minutes, name in ((1, "1 min"), (5, "5 min"), (15, "15 min"), (60, "1 h"), (1440, "1 day")):
        ax.axvline(minutes, color="0.85", linewidth=1, zorder=0)
        ax.text(minutes, 1.03, name, ha="center", va="bottom", fontsize=7, color="0.4", transform=ax.get_xaxis_transform())
    ax.set_xlabel("typical gap between records (minutes, log scale)")
    ax.set_ylabel("share of gaps close to the typical one")
    if len(chosen) <= 16:
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    else:  # too many features to name: their categories, whose hues the features share
        from matplotlib.lines import Line2D
        present = [c for c in CATEGORY_ORDER if any(feature_category(f) == c for f in rows["feature"].unique())]
        ax.legend([Line2D([], [], marker="o", linestyle="", color=CATEGORY_COLORS[c]) for c in present], present, fontsize=7,
                  loc="center left", bbox_to_anchor=(1.01, 0.5), title="category (shades tell features apart)", title_fontsize=7)
    _titles(ax, "Sampling cadence", _caption(source, int(rows["RegistrationCode"].nunique())))
    columns = ["RegistrationCode", "feature", "typical_gap_minutes", "regular_share", "days_with_data", "records"]
    return _finish(fig, rows[columns], path, dpi, individual=True)


def plot_hourly_coverage(source, feature: str, *, show_ids: bool = False, path=None, dpi: int = 150):
    """
    For each participant (rows, most complete first), the share of their days with data that hold data in each
    local hour of day; above, the cohort's median and interquartile range. Hours when devices are off, such as a
    watch charging at night, show as dark columns.
    """

    participants, cohort = sm.hour_of_day(source)
    rows = participants[participants["feature"] == feature] if len(participants) else participants
    _need(len(rows) > 0, f"no hour-of-day data for {feature}")
    _need(rows["day_share"].notna().any(), "this run has no hourly days (written before output schema statistics-4); compute it again")
    matrix = rows.pivot(index="RegistrationCode", columns="hour", values="day_share").reindex(columns=range(24))
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False, kind="stable").index]
    fig, axes = _figure(9.5, max(4.8, min(12.0, 0.16 * len(matrix) + 3.2)), 2, 1, height_ratios=[1, 3])
    top, heat = axes[0, 0], axes[1, 0]
    summary = cohort[cohort["feature"] == feature].sort_values("hour")
    top.fill_between(summary["hour"], summary["day_share_p25"], summary["day_share_p75"], color=BAND, alpha=0.6, linewidth=0)
    top.plot(summary["hour"], summary["day_share_median"], color=SINGLE, marker="o", markersize=3)
    top.set_ylim(0, 1.02)
    top.set_xlim(-0.5, 23.5)
    top.set_ylabel("cohort", fontsize=8)
    image = heat.imshow(matrix.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(image, ax=[top, heat], label="share of days with data in the hour")
    heat.set_xticks(range(0, 24, 2))
    heat.set_xlabel("local hour of day")
    if show_ids and len(matrix) <= 60:
        heat.set_yticks(range(len(matrix)))
        heat.set_yticklabels(matrix.index, fontsize=7)
    else:
        heat.set_yticks([])
        heat.set_ylabel("participants, most complete first")
    _titles(top, f"{feature}: when data exist", _caption(source, len(matrix)))
    data = matrix.reset_index().melt(id_vars="RegistrationCode", var_name="hour", value_name="day_share")
    cohort_columns = ["hour", "day_share_median", "day_share_p25", "day_share_p75", "day_share_participants"]
    return _finish(fig, data, path, dpi, individual=True, panels={"cohort": summary[cohort_columns]})


# ---------------------------------------------------------------------------------------- trust: quality
def plot_quality(report: "sm.QualityReport", *, path=None, dpi: int = 150):
    """
    Four quality indicators per feature: values outside the plausible range per 1,000 records, the share of observed
    time recorded by more than one device, the share of records entered by hand, and the share of records without a
    UTC offset (whose local day is uncertain). A blank bar is a measured zero; "n/a" means the indicator does not apply.
    """

    f = report.features.copy()
    _need(len(f) > 0, "the quality report holds no feature")
    f = f.set_index("feature").loc[feature_order(f["feature"])].reset_index()
    f["category"] = f["feature"].map(feature_category)
    f["below_per_1000"] = f["values_below_range"] / f["records"] * 1000
    f["above_per_1000"] = f["values_above_range"] / f["records"] * 1000
    f["user_entered_share"] = f["records_user_entered"] / f["records"]
    if "without_offset_share" not in f:
        f["without_offset_share"] = f.get("records_without_offset", np.nan) / f["records"]
    y = np.arange(len(f))[::-1]
    fig, axes = _figure(13.0, max(4.0, 0.28 * len(f) + 1.8), 1, 4)
    colors = [feature_color(x) for x in f["feature"]]
    specs = [("below_per_1000", "out of range per 1,000 records"), ("redundant_share", "share of time on more than one device"),
             ("user_entered_share", "share of records entered by hand"), ("without_offset_share", "share of records without offset")]
    for i, (column, label) in enumerate(specs):
        ax = axes[0, i]
        values = f[column].to_numpy(dtype=float)
        if column == "below_per_1000":
            above = f["above_per_1000"].to_numpy(dtype=float)
            ax.barh(y, np.nan_to_num(values), color=colors, label="below")
            ax.barh(y, np.nan_to_num(above), left=np.nan_to_num(values), color=[_shade(c, 0.55) for c in colors], label="above")
            ax.legend(fontsize=7, loc="lower right")
            missing = np.isnan(values)
        else:
            ax.barh(y, np.nan_to_num(values), color=colors)
            ax.set_xlim(0, max(0.05, float(np.nanmax(values)) * 1.15) if np.isfinite(values).any() else 1)
            missing = np.isnan(values)
        for row, gone in zip(y, missing):
            if gone:
                ax.text(0, row, " n/a", va="center", fontsize=6, color="0.5")
        ax.set_title(label, fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(f["feature"] if i == 0 else [], fontsize=7)
        if i == 0:
            for tick, feature in zip(ax.get_yticklabels(), f["feature"]):
                tick.set_color(feature_color(feature))
    fig.suptitle(f"Data quality by feature · {len(f)} features", x=0.01, ha="left", fontsize=11)
    columns = ["feature", "category", "records", "below_per_1000", "above_per_1000", "redundant_share",
               "user_entered_share", "without_offset_share"]
    return _finish(fig, f[columns], path, dpi)


def plot_curation_flags(report: "sm.QualityReport", *, path=None, dpi: int = 150):
    """
    Features (rows) by curation flag (columns): the share of the feature's records carrying the flag, on a
    logarithmic color scale so that rare flags stay visible; blank cells were never flagged. Curated phase only.
    """

    flags = report.curation[report.curation["kind"] == "flag"] if len(report.curation) else report.curation
    _need(len(flags) > 0, "the quality report holds no curation flag (flags exist in the curated phase)")
    matrix = flags.pivot_table(index="feature", columns="name", values="share_of_records", aggfunc="sum")
    matrix = matrix.loc[feature_order(matrix.index)]
    matrix = matrix[matrix.sum().sort_values(ascending=False, kind="stable").index]
    from matplotlib.colors import LogNorm
    positive = matrix.to_numpy(dtype=float)
    low = float(np.nanmin(positive[positive > 0])) if (positive > 0).any() else 1e-4
    fig, axes = _figure(max(8.0, 0.45 * matrix.shape[1] + 4), max(4.0, 0.32 * matrix.shape[0] + 2.5))
    ax = axes[0, 0]
    image = ax.imshow(np.ma.masked_invalid(positive), aspect="auto", interpolation="nearest", cmap="magma_r",
                      norm=LogNorm(vmin=low, vmax=1.0))
    fig.colorbar(image, ax=ax, label="share of records (log scale)")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([c.replace("_", " ") for c in matrix.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=7)
    for tick, feature in zip(ax.get_yticklabels(), matrix.index):
        tick.set_color(feature_color(feature))
    if matrix.size <= 400:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = positive[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1%}" if v >= 0.001 else "<0.1%", ha="center", va="center", fontsize=5.5,
                            color="white" if v > 0.2 else "black")
    _titles(ax, "Curation flags", f"{matrix.shape[0]} features · {matrix.shape[1]} flags")
    data = matrix.reset_index().melt(id_vars="feature", var_name="flag", value_name="share_of_records").dropna()
    return _finish(fig, data, path, dpi)


def plot_acquisition_over_time(source, feature: str | None = None, *, measure: str = "share", path=None, dpi: int = 150):
    """
    Records by acquisition method per calendar month, for one feature or all: shares of each month's records
    (``measure="share"``) or record counts. Device and app transitions show as changes in the mix; every month in
    the span is drawn, and months without records stay empty.
    """

    _need(measure in ("share", "records"), "measure must be 'share' or 'records'")
    counts: dict[tuple[pd.Period, str], float] = {}
    for table in sm._Source(source).grouped("daily_provenance"):
        if feature is not None:
            table = table[table["feature"] == feature]
        if table.empty:
            continue
        months = pd.to_datetime(table["local_date"]).dt.to_period("M")
        summed = table.groupby([months, table["acquisition_method"].astype(str)])["records"].sum()
        for month, method, n in summed.reset_index().itertuples(index=False, name=None):
            counts[(month, method)] = counts.get((month, method), 0.0) + float(n)
    _need(bool(counts), "no provenance for " + (feature or "any feature"))
    frame = pd.DataFrame([(m, a, n) for (m, a), n in counts.items()], columns=["month", "acquisition_method", "records"])
    span = pd.period_range(frame["month"].min(), frame["month"].max(), freq="M")
    matrix = frame.pivot_table(index="month", columns="acquisition_method", values="records", aggfunc="sum",
                               fill_value=0.0).reindex(span, fill_value=0.0)
    totals = matrix.sum(axis=1)
    shares = matrix.div(totals.where(totals > 0), axis=0)
    methods = list(matrix.sum().sort_values(ascending=False, kind="stable").index)
    colors = _name_colors(matrix.columns)
    fig, axes = _figure(min(16.0, max(9.0, 0.1 * len(span) + 5)), 4.8)
    ax = axes[0, 0]
    shown = shares if measure == "share" else matrix
    bottom = np.zeros(len(span))
    for m in methods:  # bars, not areas: an area would interpolate across months without records
        heights = shown[m].fillna(0).to_numpy(dtype=float)
        ax.bar(np.arange(len(span)), heights, bottom=bottom, width=0.9, color=colors[m], label=m, linewidth=0)
        bottom += heights
    step = max(1, len(span) // 12)
    ax.set_xticks(range(0, len(span), step))
    ax.set_xticklabels([str(p) for p in span[::step]], rotation=90, fontsize=7)
    ax.set_xlim(-0.6, len(span) - 0.4)
    ax.set_ylabel("share of the month's records" if measure == "share" else "records")
    if measure == "share":
        ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    empty = int((totals == 0).sum())
    _titles(ax, f"Acquisition methods over time{': ' + feature if feature else ''}", " · ".join(p for p in (
        f"{len(span)} months", f"{empty} without records" if empty else "") if p))
    matrix.index.name = shares.index.name = "month"
    data = matrix.reset_index().melt(id_vars="month", var_name="acquisition_method", value_name="records")
    data["share"] = shares.reset_index().melt(id_vars="month", var_name="acquisition_method", value_name="share")["share"].to_numpy()
    data["month"] = data["month"].astype(str)
    return _finish(fig, data, path, dpi)


# ------------------------------------------------------------------------------------------ time zones
def _offset_label(minutes: int) -> str:
    sign = "+" if minutes >= 0 else "-"
    return f"UTC{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"


def _offsets(text) -> list[int]:
    return sorted({int(float(x)) for x in str(text).split(";") if x not in ("", "nan", "None")})


def plot_time_zones(source, participant: str | None = None, *, time_axis: str | None = None, min_participants: int = 1,
                    show_ids: bool = False, path=None, dpi: int = 150):
    """
    Time zones. For the cohort: participants per home zone (a zone together with its daylight-saving partner), the
    days each spent away from it, and their trips (runs of consecutive days away). For one ``participant``: the UTC
    offsets of each day, against the home zone, on study days unless ``time_axis="calendar"``.
    """

    k = _check_min(min_participants)
    zones = sm.time_zones(source)
    _need(len(zones) > 0, "no participant has UTC offsets (day-key features carry none)")
    if participant is not None:
        code = _code(participant)
        axis = time_axis or "study"
        _need(axis in ("calendar", "study"), "time_axis must be 'calendar' or 'study'")
        days = None
        for part in sm._Source(source).grouped("participant_days"):
            if str(part["RegistrationCode"].iloc[0]) == code:
                days = part
                break
        if days is None or not len(zones[zones["RegistrationCode"] == code]):
            raise DataLoaderConfigurationError(f"no UTC offsets for participant {code}")
        home = _offsets(zones.set_index("RegistrationCode").loc[code, "home_offsets"])
        rows = days.dropna(subset=["utc_offsets"]).assign(local_date=lambda d: pd.to_datetime(d["local_date"]))
        rows = rows.assign(offsets=rows["utc_offsets"].map(_offsets)).explode("offsets").dropna(subset=["offsets"])
        rows["offset_hours"] = rows["offsets"].astype(float) / 60
        rows["study_day"] = (rows["local_date"] - rows["local_date"].min()).dt.days
        rows["home"] = rows["offsets"].astype(int).isin(home)
        x = "study_day" if axis == "study" else "local_date"
        fig, axes = _figure(10.0, 3.8)
        ax = axes[0, 0]
        for minutes in home:
            ax.axhline(minutes / 60, color="0.85", linewidth=6, zorder=0)
        ax.scatter(rows.loc[rows["home"], x], rows.loc[rows["home"], "offset_hours"], s=6, color=SINGLE, label="home zone")
        away = rows[~rows["home"]]
        if len(away):
            ax.scatter(away[x], away["offset_hours"], s=10, color=PALETTE[5], label="away")
        ax.set_ylabel("UTC offset (hours)")
        ax.set_xlabel("days since the first day with an offset" if axis == "study" else "local date")
        ax.legend(fontsize=8)
        who = code if show_ids else "one participant"
        _titles(ax, "UTC offsets by day", f"{who} · home {' / '.join(_offset_label(m) for m in home)}")
        columns = [x, "offsets", "offset_hours", "home"] + (["study_day"] if x == "local_date" else [])
        return _finish(fig, rows[columns], path, dpi, individual=True)
    zones = zones.assign(home_zone=zones["home_offsets"].map(lambda t: " / ".join(_offset_label(m) for m in _offsets(t))))
    homes = zones["home_zone"].value_counts().rename_axis("home_zone").reset_index(name="participants")
    away = zones["days_away"].astype(int).value_counts().sort_index()
    away = away.reindex(range(0, int(away.index.max()) + 1), fill_value=0).rename_axis("days_away").reset_index(name="participants")
    trips = zones["trips"].astype(int).value_counts().sort_index()
    trips = trips.reindex(range(0, int(trips.index.max()) + 1), fill_value=0).rename_axis("trips").reset_index(name="participants")
    hidden = 0
    for table in (homes, away, trips):
        table["participants"] = table["participants"].astype(float)
        table["suppressed"] = (table["participants"] > 0) & (table["participants"] < k)
        hidden += int(table["suppressed"].sum())
        table.loc[table["suppressed"], "participants"] = np.nan
    fig, axes = _figure(13.0, 4.2, 1, 3)
    axes[0, 0].barh(homes["home_zone"], homes["participants"].fillna(0), color=SINGLE)
    axes[0, 0].set_xlabel("participants")
    axes[0, 1].bar(away["days_away"], away["participants"].fillna(0), color=SINGLE, width=0.9)
    axes[0, 1].set_xlabel("days away from the home zone")
    axes[0, 2].bar(trips["trips"], trips["participants"].fillna(0), color=SINGLE, width=0.9)
    axes[0, 2].set_xlabel("trips (runs of consecutive days away)")
    from matplotlib.ticker import MaxNLocator
    for ax, top in ((axes[0, 1], int(away["days_away"].max())), (axes[0, 2], int(trips["trips"].max()))):
        if top <= 12:  # the integer locator falls back to fractions when fewer than two integers are in view
            ax.set_xticks(range(0, top + 1))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(-0.6, top + 0.6)
        ax.set_ylabel("participants")
    travelling = float((zones["days_away"] > 0).mean())
    _titles(axes[0, 0], "Home zones", _participants(len(zones)))
    _titles(axes[0, 1], "Days away", f"{travelling:.0%} of participants were ever away")
    _titles(axes[0, 2], "Trips", _hidden(hidden, k, "bar"))
    return _finish(fig, zones, path, dpi, panels={"home_zones": homes, "days_away": away, "trips": trips})


# ---------------------------------------------------------------------------------- coverage: who has what
def plot_feature_combinations(coverage: "ds.Coverage", features: Iterable[str] | None = None, *,
                              max_combinations: int = 15, min_participants: int = 1, path=None, dpi: int = 150):
    """
    Which exact combinations of features participants have (an UpSet chart): bars count the participants whose
    features, among those chosen, are exactly the combination marked in the matrix below; the bars on the left count
    each feature's participants. By default the six most common features.
    """

    k = _check_min(min_participants)
    presence = coverage.presence()
    _need(not presence.empty, "the coverage holds no participant")
    if features is None:
        chosen = list(presence.sum().sort_values(ascending=False, kind="stable").index[:6])
    else:
        chosen = list(dict.fromkeys(features))
        unknown = sorted(set(chosen) - set(presence.columns))
        _need(not unknown, f"no participant has {', '.join(unknown)}")
    chosen = feature_order(chosen)
    sub = presence[chosen]
    sub = sub[sub.any(axis=1)]
    _need(len(sub) > 0, "no participant has any of these features")
    keys = sub.apply(lambda r: tuple(bool(v) for v in r), axis=1)
    counts = keys.value_counts()
    table = pd.DataFrame({"members": list(counts.index), "participants": counts.to_numpy(dtype=float)})
    table["size"] = table["members"].map(sum)
    table["combination"] = [" + ".join(f for f, on in zip(chosen, members) if on) for members in table["members"]]
    table = table.sort_values(["participants", "size"], ascending=[False, False], kind="stable").reset_index(drop=True)
    table["suppressed"] = table["participants"] < k
    table.loc[table["suppressed"], "participants"] = np.nan
    top = table[~table["suppressed"]].head(int(max_combinations))
    _need(len(top) > 0, f"every combination rests on fewer than {k} participants")
    fig, axes = _figure(max(8.0, 0.5 * len(top) + 4.5), 3.2 + 0.32 * len(chosen), 2, 2, height_ratios=[3, max(1.2, 0.4 * len(chosen))],
                        width_ratios=[1.2, max(3.0, 0.5 * len(top))])
    axes[0, 0].axis("off")
    bars = axes[0, 1].bar(range(len(top)), top["participants"], color=SINGLE)
    axes[0, 1].bar_label(bars, labels=[f"{int(n):,}" for n in top["participants"]], fontsize=7)
    axes[0, 1].set_ylabel("participants")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_xlim(-0.6, len(top) - 0.4)
    matrix = axes[1, 1]
    for j, members in enumerate(top["members"]):
        on = [i for i, m in enumerate(members) if m]
        matrix.scatter([j] * len(chosen), range(len(chosen)), s=40, color="0.88", zorder=1)
        matrix.scatter([j] * len(on), on, s=40, color=SINGLE, zorder=3)
        if len(on) > 1:
            matrix.plot([j, j], [min(on), max(on)], color=SINGLE, linewidth=2, zorder=2)
    matrix.set_xlim(-0.6, len(top) - 0.4)
    matrix.set_ylim(-0.6, len(chosen) - 0.4)
    matrix.invert_yaxis()
    matrix.set_yticks(range(len(chosen)))
    matrix.set_yticklabels(chosen, fontsize=8)
    for tick, feature in zip(matrix.get_yticklabels(), chosen):
        tick.set_color(feature_color(feature))
    matrix.set_xticks([])
    for side in ("top", "right", "bottom"):
        matrix.spines[side].set_visible(False)
    sets = sub.sum().reindex(chosen).rename_axis("feature").reset_index(name="participants")
    axes[1, 0].barh(range(len(chosen)), sets["participants"], color=[feature_color(f) for f in chosen])
    axes[1, 0].set_ylim(-0.6, len(chosen) - 0.4)
    axes[1, 0].invert_yaxis()
    axes[1, 0].invert_xaxis()
    axes[1, 0].set_yticks([])
    axes[1, 0].set_xlabel("participants", fontsize=8)
    notes = [_caption(coverage, len(sub)), f"{len(top)} of {len(table)} combinations shown",
             _hidden(int(table["suppressed"].sum()), k, "combination")]
    _titles(axes[0, 1], "Feature combinations per participant", " · ".join(n for n in notes if n))
    data = top.drop(columns="members").assign(**{f: [m[i] for m in top["members"]] for i, f in enumerate(chosen)})
    everything = table.drop(columns="members").assign(**{f: [m[i] for m in table["members"]] for i, f in enumerate(chosen)})
    return _finish(fig, data, path, dpi, panels={"sets": sets, "all_combinations": everything})


def plot_multimodal_days(source, features: Iterable[str], *, groups=None, min_participants: int = 1, path=None,
                         dpi: int = 150):
    """
    For a set of features: each participant's days on which all of them have data (complete days), and the share of
    participants with at least k complete days, by k: how large a multimodal dataset each requirement leaves. The
    participants counted are those with any of the features; those with none of these days count as zero.
    """

    chosen = feature_order(features)
    _need(len(chosen) >= 2, "multimodal days need at least two features")
    k = _check_min(min_participants)
    complete = sm.days_with(source, chosen)
    table = sm._Source(source).table("participant_feature")
    base = pd.DataFrame({"RegistrationCode": sorted(table.loc[table["feature"].isin(chosen), "RegistrationCode"].unique())})
    _need(len(base) > 0, "no participant has any of these features")
    base = base.merge(complete[["RegistrationCode", "days"]], on="RegistrationCode", how="left")
    base["complete_days"] = base.pop("days").fillna(0).astype(int)
    frame, labels, excluded = _grouped(base, groups)
    most = int(frame["complete_days"].max())
    ks = np.arange(1, max(2, most + 1))
    curve = []
    for label in labels:
        part = frame[frame["group"] == label]
        for kk in ks:
            n = int((part["complete_days"] >= kk).sum())
            hide = 0 < n < k
            curve.append({"group": label, "k": int(kk), "participants": np.nan if hide else float(n),
                          "share": np.nan if hide else n / len(part), "suppressed": hide})
    curve = pd.DataFrame(curve)
    fig, axes = _figure(12.0, 4.6, ncols=2)
    left, right = axes[0, 0], axes[0, 1]
    edges = np.unique(np.concatenate([[-0.5, 0.5], np.linspace(0.5, most + 0.5, min(30, max(2, most)) + 1)]))
    hidden = _histogram_panel(left, frame, "complete_days", edges, labels, k)
    left.set_xlabel("complete days per participant")
    left.set_ylabel("participants")
    for label in labels:
        part = curve[curve["group"] == label]
        right.step(part["k"], part["share"], where="post", color=_group_color(labels, label),
                   label="participants" if labels == [ALL_PARTICIPANTS] else _group_legend(label, int((frame["group"] == label).sum())))
    right.set_ylim(0, 1.02)
    right.set_xlabel("required complete days (k)")
    right.set_ylabel("share with at least k complete days")
    if len(labels) > 1:
        right.legend(fontsize=7)
    zero = int((frame["complete_days"] == 0).sum())
    _titles(left, "Complete days: " + " + ".join(chosen), " · ".join(p for p in (
        f"{zero:,} of {len(frame):,} participants have none", _left_out(excluded), _hidden(hidden, k, "bin")) if p))
    _titles(right, "How many participants each requirement keeps", " · ".join(p for p in (
        _caption(source, len(frame)), _hidden(int(curve["suppressed"].sum()), k)) if p))
    return _finish(fig, frame, path, dpi, panels={"curve": curve})


def plot_availability_raster(source, *, features: Iterable[str] | None = None, time_axis: str = "calendar",
                             show_ids: bool = False, path=None, dpi: int = 150):
    """
    Participants (rows) by months (columns): the days in each month with data in any of ``features`` (all by
    default). Rows are sorted by first month; with ``time_axis="study"`` months count from each participant's first.
    Blank cells are months without data; every month in the span is drawn.
    """

    _need(time_axis in ("calendar", "study"), "time_axis must be 'calendar' or 'study'")
    wanted = None if features is None else set(features)
    per: dict[str, pd.Series] = {}
    for days in sm._Source(source).grouped("participant_days"):
        if wanted is not None:
            days = days[days["features"].map(lambda s: bool(wanted & set(str(s).split(";"))))]
        if days.empty:
            continue
        per[str(days["RegistrationCode"].iloc[0])] = pd.to_datetime(days["local_date"]).dt.to_period("M").value_counts()
    _need(bool(per), "no participant-day for these features")
    first = {code: s.index.min() for code, s in per.items()}
    if time_axis == "calendar":
        columns = list(pd.period_range(min(first.values()), max(s.index.max() for s in per.values()), freq="M"))
        rows = {code: s.reindex(columns, fill_value=0) for code, s in per.items()}
    else:
        rows = {code: pd.Series(s.to_numpy(), index=[p.ordinal - first[code].ordinal for p in s.index]) for code, s in per.items()}
        columns = list(range(0, max(int(s.index.max()) for s in rows.values()) + 1))
        rows = {code: s.groupby(level=0).sum().reindex(columns, fill_value=0) for code, s in rows.items()}
    matrix = pd.DataFrame(rows).T.reindex(columns=columns).fillna(0).astype(int)
    order = sorted(matrix.index, key=lambda c: (first[c].ordinal if time_axis == "calendar" else 0, -int(matrix.loc[c].sum()), c))
    matrix = matrix.loc[order]
    fig, axes = _figure(min(16.0, max(9.0, 0.1 * len(columns) + 5)), max(4.0, min(12.0, 0.16 * len(matrix) + 2.5)))
    ax = axes[0, 0]
    image = ax.imshow(np.ma.masked_equal(matrix.to_numpy(dtype=float), 0), aspect="auto", interpolation="nearest",
                      cmap="viridis", vmin=1, vmax=31)
    fig.colorbar(image, ax=ax, label="days with data in the month")
    step = max(1, len(columns) // 12)
    ax.set_xticks(range(0, len(columns), step))
    ax.set_xticklabels([str(c) for c in columns[::step]], rotation=90, fontsize=7)
    ax.set_xlabel("month" if time_axis == "calendar" else "months since the first month with data")
    if show_ids and len(matrix) <= 60:
        ax.set_yticks(range(len(matrix)))
        ax.set_yticklabels(matrix.index, fontsize=7)
    else:
        ax.set_yticks([])
        ax.set_ylabel("participants" + (", by first month" if time_axis == "calendar" else ", most days first"))
    what = "any feature" if features is None else " or ".join(feature_order(features))
    _titles(ax, f"Data availability: {what}", _caption(source, len(matrix)))
    name = "month" if time_axis == "calendar" else "study_month"
    data = matrix.rename_axis("RegistrationCode").reset_index().melt(id_vars="RegistrationCode", var_name=name, value_name="days")
    data[name] = data[name].astype(str) if time_axis == "calendar" else data[name].astype(int)
    return _finish(fig, data, path, dpi, individual=True)


def plot_follow_up(source, feature: str | None = None, *, sort: str = "start", time_axis: str = "calendar",
                   show_ids: bool = False, path=None, dpi: int = 150):
    """
    Each participant's follow-up, from their first to their last day with data (for one feature, or any), colored
    by density: the share of the span's days that hold data. ``sort`` is ``"start"`` or ``"length"``; with
    ``time_axis="study"`` every line starts at zero and the real dates stay off the figure and out of its data.
    """

    _need(sort in ("start", "length"), "sort must be 'start' or 'length'")
    _need(time_axis in ("calendar", "study"), "time_axis must be 'calendar' or 'study'")
    src = sm._Source(source)
    table = src.table("participant_feature")
    if feature is not None:
        _need(feature in set(table["feature"]), f"no participant has {feature}")
        rows = table[table["feature"] == feature][["RegistrationCode", "first_day", "last_day", "days_with_data"]].copy()
    else:
        rows = table.groupby("RegistrationCode").agg(first_day=("first_day", "min"), last_day=("last_day", "max")).reset_index()
        days = {str(d["RegistrationCode"].iloc[0]): len(d) for d in src.grouped("participant_days")}
        rows["days_with_data"] = rows["RegistrationCode"].map(days).fillna(0).astype(int)
    _need(len(rows) > 0, "no participant has data")
    rows["first_day"], rows["last_day"] = pd.to_datetime(rows["first_day"]), pd.to_datetime(rows["last_day"])
    rows["span_days"] = (rows["last_day"] - rows["first_day"]).dt.days + 1
    rows["density"] = rows["days_with_data"] / rows["span_days"]
    rows = rows.sort_values(["first_day", "span_days"] if sort == "start" else ["span_days", "first_day"],
                            ascending=[True, False] if sort == "start" else [False, True], kind="stable").reset_index(drop=True)
    import matplotlib
    from matplotlib import cm
    from matplotlib.colors import Normalize
    norm = Normalize(0, 1)
    cmap = matplotlib.colormaps["viridis"]
    colors = cmap(norm(rows["density"].to_numpy(dtype=float)))
    fig, axes = _figure(10.0, max(4.0, min(12.0, 0.12 * len(rows) + 2.5)))
    ax = axes[0, 0]
    y = np.arange(len(rows))
    if time_axis == "calendar":
        ax.hlines(y, rows["first_day"], rows["last_day"], colors=colors, linewidth=2)
        single = rows["span_days"] == 1
        ax.scatter(rows.loc[single, "first_day"], y[single.to_numpy()], s=6, color=colors[single.to_numpy()])
        ax.set_xlabel("local date")
    else:
        ax.hlines(y, 0, rows["span_days"] - 1, colors=colors, linewidth=2)
        ax.set_xlabel("days since the first day with data")
    ax.invert_yaxis()
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="share of the span's days with data")
    if show_ids and len(rows) <= 60:
        ax.set_yticks(y)
        ax.set_yticklabels(rows["RegistrationCode"], fontsize=7)
    else:
        ax.set_yticks([])
        ax.set_ylabel("participants, " + ("earliest start first" if sort == "start" else "longest first"))
    _titles(ax, f"Follow-up: {feature or 'any feature'}", " · ".join(p for p in (
        _caption(source, len(rows)), f"median span {int(rows['span_days'].median()):,} days") if p))
    columns = ["RegistrationCode", "span_days", "days_with_data", "density"]
    if time_axis == "calendar":
        columns[1:1] = ["first_day", "last_day"]
    return _finish(fig, rows[columns], path, dpi, individual=True)


# -------------------------------------------------------------------------------------------------- report
def _coverage_for(source, features: list[str]) -> "ds.Coverage | None":
    """The coverage of the root the run was computed from, if that root can still be read."""

    if isinstance(source, ds.DailyStatistics):
        parameters = source.run.get("parameters", {})
    else:
        parameters = json.loads((Path(source) / "run.json").read_text()).get("parameters", {})
    root = parameters.get("root")
    if not root or not Path(root).is_dir():
        return None
    return ds.compute_coverage(parameters.get("phase", "curated"), root=root, features=features)


REPORT_SECTIONS = frozenset({"cohort", "sleep", "glucose", "values", "clinical", "feature"})


def _domain_for(source) -> "dm.DomainMetrics | None":
    """The sleep and CGM metrics of the root the run was computed from, if that root can still be read."""

    if isinstance(source, ds.DailyStatistics):
        parameters = source.run.get("parameters", {})
    else:
        parameters = json.loads((Path(source) / "run.json").read_text()).get("parameters", {})
    root = parameters.get("root")
    if not root or not Path(root).is_dir():
        return None
    return dm.compute_domain_metrics(parameters.get("phase", "curated"), root=root)


def report(source, out, *, features: Iterable[str] | None = None, coverage: "ds.Coverage | None" = None,
           domain: "dm.DomainMetrics | None" = None, rules=None,
           groups=None, min_participants: int = 1, include_individual: bool | None = None,
           sections: Iterable[str] | None = None, dpi: int = 100) -> pd.DataFrame:
    """
    A multi-page PDF of the standard figures: a cohort overview (coverage, availability, quality, context), then a
    dossier per feature (valid-day rule, hour of day, when data exist, weekly and monthly patterns, distribution
    over time, daily values, adherence, participants' medians, weekdays and weekend, each participant's day), with
    Table 1, daily rhythms, correlations and clinical reference pages. With Sleep or BloodGlucose, sleep and glucose pages from
    ``domain`` (computed from the run's root when not given, if it can be read). ``out`` is the PDF's path, outside
    every data root.

    ``groups`` and ``min_participants`` apply to every figure that supports them. With ``min_participants`` above 1,
    for figures leaving the lab, figures of individual participants are left out, since they cannot respect a
    threshold. ``sections`` keeps only some parts: "cohort", "sleep", "glucose", "values", "clinical" and "feature"
    (the per-feature dossiers). Figures that do not apply (such as hours for a day-key feature) are skipped. Returns one row per
    figure: its section, feature, title, and whether it was drawn, skipped (with the reason) or left out.
    """

    import inspect

    target = ds.guard_output(out)
    _need(target.suffix.lower() == ".pdf", "the report is a PDF: give out a file name ending in .pdf")
    k = _check_min(min_participants)
    include = (k == 1) if include_individual is None else bool(include_individual)
    _need(not (include and k > 1), "figures of individual participants cannot respect min_participants; set include_individual=False")
    src = sm._Source(source)
    chosen = feature_order(src.features if features is None else features)
    unknown = sorted(set(chosen) - set(src.features))
    _need(not unknown, f"not among the run's features: {', '.join(unknown)}")
    coverage = coverage if coverage is not None else _coverage_for(source, chosen)
    summaries = sm.summarize(source, rules=rules, features=chosen)
    patterns = sm.temporal_patterns(source, rules=rules, features=chosen)
    quality = sm.quality_report(source)
    shared = {"groups": groups, "min_participants": k, "rules": rules}

    def draw(function, *args, **extra):
        accepted = inspect.signature(function).parameters
        return function(*args, **{n: v for n, v in {**shared, **extra}.items() if n in accepted})

    table = src.table("participant_feature")
    common = [f for f in table.groupby("feature")["RegistrationCode"].nunique().sort_values(ascending=False).index if f in chosen][:3]
    pages: list[tuple[str, str, str, bool, Any]] = []
    add = lambda section, feature, title, individual, thunk: pages.append((section, feature, title, individual, thunk))
    if coverage is not None:
        add("cohort", "", "Participants with each feature", False, lambda: draw(plot_participants_per_feature, coverage))
        add("cohort", "", "Features per participant", False, lambda: draw(plot_features_per_participant, coverage))
        add("cohort", "", "Feature combinations", False, lambda: draw(plot_feature_combinations, coverage))
        add("cohort", "", "Features present per participant", True, lambda: draw(plot_feature_presence, coverage))
        add("cohort", "", "Stored data per participant", False, lambda: draw(plot_data_volume, coverage))
    add("cohort", "", "Data availability", True, lambda: draw(plot_availability_raster, source))
    add("cohort", "", "Follow-up", True, lambda: draw(plot_follow_up, source))
    add("cohort", "", "Participants with data per day", False, lambda: draw(plot_active_participants, source, rolling=28))
    add("cohort", "", "Participants per feature and month", False, lambda: draw(plot_feature_activity, source, features=chosen))
    add("cohort", "", "Days shared by pairs of features", False, lambda: draw(plot_co_availability, source, chosen))
    if len(common) >= 2:
        add("cohort", "", "Complete days", False, lambda: draw(plot_multimodal_days, source, common))
    add("cohort", "", "Retention", False, lambda: draw(plot_retention, summaries, chosen))
    add("cohort", "", "Data quality by feature", False, lambda: draw(plot_quality, quality))
    add("cohort", "", "Curation flags", False, lambda: draw(plot_curation_flags, quality))
    add("cohort", "", "How records were acquired", False, lambda: draw(plot_acquisition, quality))
    add("cohort", "", "Acquisition methods over time", False, lambda: draw(plot_acquisition_over_time, source))
    add("cohort", "", "Time zones", False, lambda: draw(plot_time_zones, source))
    add("cohort", "", "Sampling cadence", True, lambda: draw(plot_sampling_cadence, source, chosen))
    if domain is None and {"Sleep", "BloodGlucose"} & set(chosen):
        domain = _domain_for(source)
    if domain is not None and "Sleep" in chosen:
        add("sleep", "Sleep", "Sleep duration, timing and efficiency", False, lambda: draw(plot_sleep, domain))
        add("sleep", "Sleep", "Regularity and social jetlag", False, lambda: draw(plot_sleep_regularity, domain))
        add("sleep", "Sleep", "Stages, wake and naps", False, lambda: draw(plot_sleep_architecture, domain))
        add("sleep", "Sleep", "Timing and duration", True, lambda: draw(plot_sleep_timing, domain))
        add("sleep", "Sleep", "What nights record", True, lambda: draw(plot_sleep_recording, domain))
    if domain is not None and "BloodGlucose" in chosen:
        add("glucose", "BloodGlucose", "Glucose by time of day", False, lambda: draw(plot_agp, domain))
        add("glucose", "BloodGlucose", "Glycaemia and consensus targets", False, lambda: draw(plot_glycemic_cohort, domain))
        add("glucose", "BloodGlucose", "Time in glucose ranges", True, lambda: draw(plot_cgm_ranges, domain))
        add("glucose", "BloodGlucose", "CGM wear", True, lambda: draw(plot_cgm_wear, domain))
    add("values", "", "Participants' medians (Table 1)", False, lambda: draw(plot_metric_distributions, summaries, chosen))
    add("values", "", "Daily rhythms across features", False, lambda: draw(plot_daily_rhythms, source, chosen))
    add("values", "", "Correlations across participants", False, lambda: draw(plot_feature_correlations, summaries, chosen))
    add("clinical", "", "Readings beyond clinical thresholds", False, lambda: draw(plot_clinical_thresholds, source))
    if "StepCount" in chosen:
        add("clinical", "StepCount", "Daily step categories", False, lambda: draw(plot_step_categories, summaries))
    if "ActivitySummary" in chosen:
        add("clinical", "ActivitySummary", "Activity guideline and ring goals", False, lambda: draw(plot_activity_goals, source))
    if "BloodPressure" in chosen:
        add("clinical", "BloodPressure", "Blood-pressure categories", False, lambda: draw(plot_blood_pressure, summaries, points=False))
    if "Weight" in chosen:
        add("clinical", "Weight", "Weight trajectories", True, lambda: draw(plot_weight_trajectories, source))
    for feature in chosen:
        f = feature
        add("feature", f, "The valid-day rule", False, lambda f=f: draw(plot_valid_day_rule, source, f, rule=(rules or {}).get(f)))
        add("feature", f, "By hour of day", False, lambda f=f: draw(plot_hour_of_day, source, f))
        add("feature", f, "Share of days with data by hour", False, lambda f=f: draw(plot_hour_of_day, source, f, measure="coverage"))
        add("feature", f, "When data exist", True, lambda f=f: draw(plot_hourly_coverage, source, f))
        add("feature", f, "By day of week", False, lambda f=f: draw(plot_weekly_pattern, patterns, f))
        add("feature", f, "By month of year", False, lambda f=f: draw(plot_monthly_pattern, patterns, f))
        add("feature", f, "Distribution over time", False, lambda f=f: draw(plot_monthly_distribution, source, f))
        add("feature", f, "Daily values", False, lambda f=f: draw(plot_daily_values, source, f))
        add("feature", f, "Adherence", False, lambda f=f: draw(plot_adherence, summaries, f))
        add("feature", f, "Participants' medians and spread", True, lambda f=f: draw(plot_caterpillar, summaries, f))
        add("feature", f, "Weekdays and weekend", False, lambda f=f: draw(plot_weekday_weekend, patterns, f, paired=False))
        add("feature", f, "Each participant's day", True, lambda f=f: draw(plot_hour_profiles, source, f))
    if sections is not None:
        wanted = set(sections)
        _need(wanted <= REPORT_SECTIONS, f"sections must be among {', '.join(sorted(REPORT_SECTIONS))}")
        pages = [page for page in pages if page[0] in wanted]
    from matplotlib.backends.backend_pdf import PdfPages
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    phase = _phase(source) or ""
    with PdfPages(target, metadata={"Title": f"Wearable data report ({phase} phase)".replace(" ( phase)", ""),
                                    "Creator": "wearable_project data_plots", "CreationDate": None}) as pdf:
        for section, feature, title, individual, thunk in pages:
            if individual and not include:
                rows.append({"section": section, "feature": feature, "figure": title, "status": "left out",
                             "reason": "shows individual participants"})
                continue
            try:
                pdf.savefig(thunk())
                rows.append({"section": section, "feature": feature, "figure": title, "status": "drawn", "reason": ""})
            except DataLoaderConfigurationError as exc:
                rows.append({"section": section, "feature": feature, "figure": title, "status": "skipped", "reason": str(exc)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ sleep
SLEEP_COLORS = {"INBED": "#d9d9d9", "AWAKE": "#fdae6b", "ASLEEP": "#6baed6", "CORE": "#4292c6", "DEEP": "#08306b",
                 "REM": "#9e9ac8"}
SLEEP_LABELS = {"INBED": "in bed", "AWAKE": "awake", "ASLEEP": "asleep (unstaged)", "CORE": "core", "DEEP": "deep",
                "REM": "REM"}
# The consensus targets for most adults with diabetes (Battelino et al., Diabetes Care 2019), and the coefficient of
# variation at or below which glucose counts as stable (Danne et al., 2017).
CV_STABILITY_THRESHOLD = 36.0
GLUCOSE_TARGETS = (("time in range 70-180 > 70%", "in_range_percent", ">", 70.0),
                   ("time below 70 < 4%", "below_70_percent", "<", 4.0),
                   ("time below 54 < 1%", "very_low_percent", "<", 1.0),
                   ("time above 180 < 25%", "above_180_percent", "<", 25.0),
                   ("time above 250 < 5%", "very_high_percent", "<", 5.0),
                   (f"CV <= {CV_STABILITY_THRESHOLD:g}%", "cv_percent", "<=", CV_STABILITY_THRESHOLD))


def _clock_ticks(ax, start: int = 12) -> None:
    """Hours after ``start`` o'clock on the x-axis, labeled with local clock times every three hours."""

    ticks = np.arange(0, 25, 3)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int((start + t) % 24):02d}:00" for t in ticks], fontsize=7)


def _rectangles(ax, x0, x1, y, height: float, color: str, alpha: float = 1.0) -> None:
    from matplotlib.collections import PolyCollection
    if len(x0) == 0:
        return
    vertices = [[(a, b - height / 2), (a, b + height / 2), (c, b + height / 2), (c, b - height / 2)] for a, c, b in zip(x0, x1, y)]
    ax.add_collection(PolyCollection(vertices, facecolors=color, edgecolors="none", alpha=alpha))


def _valid_nights(metrics: "dm.DomainMetrics", rule: "dm.NightRule") -> pd.DataFrame:
    nights = metrics.sleep_nights
    _need(len(nights) > 0, "the domain metrics hold no night")
    valid = nights[nights["asleep_recorded"].astype(bool) & (nights["asleep_minutes"] >= rule.min_asleep_minutes)]
    _need(len(valid) > 0, "no valid night: " + rule.describe())
    return valid


def plot_sleep_raster(segments: pd.DataFrame, *, time_axis: str = "study", gaps: str = "kept", show_ids: bool = False,
                      path=None, dpi: int = 150):
    """
    One participant's nights (rows) from noon to noon: time in bed, awake, asleep and sleep stages, from
    ``domain_metrics.load_sleep_segments``. Stages the night metrics leave out (recorded by a second device, or outside
    the main sleep period) are drawn faded, and nights on which two devices staged sleep are marked on the right. Rows
    count nights since the first, gaps kept, unless ``time_axis="calendar"``; ``gaps="removed"`` draws only the nights
    with segments, one after another, so that long follow-ups with breaks stay readable.
    """

    _need(time_axis in ("calendar", "study"), "time_axis must be 'calendar' or 'study'")
    _need(gaps in ("kept", "removed"), "gaps must be 'kept' or 'removed'")
    _need(len(segments) > 0, "no sleep segments")
    _need(segments["RegistrationCode"].nunique() == 1, "the sleep raster draws one participant: filter the segments first")
    table = segments.copy()
    nights = sorted(table["night_date"].unique())
    index = ({n: (pd.Timestamp(n) - pd.Timestamp(nights[0])).days for n in nights} if gaps == "kept"
             else {n: i for i, n in enumerate(nights)})
    table["night_index"] = table["night_date"].map(index)
    last = max(index.values())
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, axes = _figure(10.5, max(4.0, min(14.0, 0.02 * (last + 1) + 3.0)))
    ax = axes[0, 0]
    handles = []
    known = ("INBED", "AWAKE", "ASLEEP", "CORE", "DEEP", "REM")
    for state in known + tuple(sorted(set(table["state"]) - set(known))):
        part = table[table["state"] == state]
        if part.empty:
            continue
        color = SLEEP_COLORS.get(state, "#bdbdbd")
        for counted, alpha in ((True, 1.0), (False, 0.3)):
            sub = part[part["counted"].astype(bool) == counted]
            _rectangles(ax, sub["start_hours_after_noon"].to_numpy(), sub["end_hours_after_noon"].to_numpy(),
                        sub["night_index"].to_numpy(), 0.9 if state == "INBED" else 0.6, color, alpha)
        handles.append(Patch(color=color, label=SLEEP_LABELS.get(state, state.lower())))
    if (~table["counted"].astype(bool)).any():
        handles.append(Patch(color=SLEEP_COLORS["CORE"], alpha=0.3, label="stage not counted"))
    # Sources staging within the main sleep period, as the night metrics count them (staging_sources).
    stages = table[table["state"].isin(dm.STAGE_STATES) & table["main_period"].astype(bool)]
    multi = stages.groupby("night_index")["source"].nunique() if len(stages) else pd.Series(dtype=int)
    multi = multi[multi > 1].index.to_numpy()
    if len(multi):
        ax.scatter(np.full(len(multi), 24.35), multi, marker="<", s=14, color=PALETTE[5], clip_on=False)
        handles.append(Line2D([], [], marker="<", linestyle="", color=PALETTE[5], label="staged by two devices"))
    ax.set_xlim(0, 24)
    ax.set_ylim(last + 1, -1)
    _clock_ticks(ax)
    ax.set_xlabel("local time, noon to noon")
    if time_axis == "calendar":
        step = max(1, len(nights) // 15)
        ax.set_yticks([index[n] for n in nights[::step]])
        ax.set_yticklabels([str(n) for n in nights[::step]], fontsize=7)
    else:
        ax.set_ylabel("nights since the first" if gaps == "kept" else "nights with data, in order")
    ax.legend(handles=handles, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    asleep = table[table["state"].isin(dm.ASLEEP_STATES)]
    naps = asleep.loc[~asleep["main_period"].astype(bool), "night_index"].nunique()
    who = table["RegistrationCode"].iloc[0] if show_ids else "one participant"
    _titles(ax, "Sleep by night", f"{who} · {len(nights):,} nights, {asleep['night_index'].nunique():,} with sleep recorded · "
                                  f"{naps:,} with naps · {len(multi):,} staged by two devices")
    columns = ["night_index", "state", "source", "start_hours_after_noon", "end_hours_after_noon", "minutes", "main_period", "counted"]
    if time_axis == "calendar":
        columns = ["night_date", "start_local", "end_local"] + columns
    return _finish(fig, table[columns], path, dpi, individual=True)


def plot_sleep_regularity(metrics: "dm.DomainMetrics", *, rule: "dm.NightRule" = dm.NightRule(),
                          weekend_days: Iterable[int] = sm.WEEKEND_DAYS, groups=None, min_participants: int = 1,
                          path=None, dpi: int = 150):
    """
    Sleep regularity over valid nights, one value per participant (``domain_metrics.sleep_regularity``): the standard
    deviation of the sleep midpoint; social jetlag, the mean midpoint on free nights minus that on work nights; and
    the extra sleep on free nights. A night is free when it ends on a weekend day (``weekend_days``, Monday 0): with
    the default Friday-Saturday weekend, the Thursday and Friday nights.
    """

    k = _check_min(min_participants)
    table = dm.sleep_regularity(_valid_nights(metrics, rule), rule=rule, weekend_days=weekend_days)
    table["extra_sleep_free_minutes"] = table["asleep_free_minutes"] - table["asleep_work_minutes"]
    frame, labels, excluded = _grouped(table, groups)
    fig, axes = _figure(13.0, 4.4, ncols=3)
    specs = (("midpoint_sd_hours", "SD of the sleep midpoint (hours)", 0.25, None),
             ("social_jetlag_hours", "free minus work midpoint (hours)", 0.25, 0.0),
             ("extra_sleep_free_minutes", "free minus work total sleep (minutes)", 15.0, 0.0))
    hidden = 0
    for ax, (column, label, width, zero) in zip(axes[0], specs):
        rows = frame.dropna(subset=[column])
        ax.set_xlabel(label)
        if rows.empty:
            ax.text(0.5, 0.5, "no participant has\nboth kinds of night", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            continue
        values = rows[column].to_numpy(dtype=float)
        edges = np.arange(np.floor(values.min() / width) * width, np.ceil(values.max() / width) * width + width * 1.5, width)
        hidden += _histogram_panel(ax, rows, column, edges, labels, k)
        if zero is not None:
            ax.axvline(zero, color="black", linewidth=0.8)
    axes[0, 0].set_ylabel("participants")
    if len(labels) > 1:
        axes[0, 0].legend(fontsize=7)
    free = ", ".join(WEEKDAY_NAMES[(int(d) - 1) % 7] for d in sorted(weekend_days))
    both = int(frame["social_jetlag_hours"].notna().sum())
    _titles(axes[0, 0], "Regularity", " · ".join(p for p in (_participants(int(frame["RegistrationCode"].nunique())), _left_out(excluded)) if p))
    _titles(axes[0, 1], "Social jetlag", f"free nights: {free} · {both:,} participants with both kinds")
    _titles(axes[0, 2], "Sleep on free nights", " · ".join(p for p in (rule.describe(), _hidden(hidden, k, "bin")) if p))
    return _finish(fig, frame, path, dpi)


def plot_sleep_timing(metrics: "dm.DomainMetrics", *, rule: "dm.NightRule" = dm.NightRule(), unit: str = "participant",
                      groups=None, path=None, dpi: int = 150):
    """
    Sleep timing against duration over valid nights: the sleep midpoint (local clock time) against total sleep time,
    one point per participant (their median night) or per night (``unit="night"``), with the cohort medians marked.
    """

    _need(unit in ("participant", "night"), "unit must be 'participant' or 'night'")
    valid = _valid_nights(metrics, rule)
    columns = ["midpoint_hours_after_noon", "asleep_minutes"]
    values = (valid.groupby("RegistrationCode")[columns].median().reset_index() if unit == "participant"
              else valid[["RegistrationCode", "night_date", *columns]])
    frame, labels, excluded = _grouped(values, groups)
    fig, axes = _figure(8.5, 5.5)
    ax = axes[0, 0]
    for label in labels:
        part = frame[frame["group"] == label]
        ax.scatter(part["midpoint_hours_after_noon"], part["asleep_minutes"] / 60, s=22 if unit == "participant" else 6,
                   alpha=0.8 if unit == "participant" else 0.35, color=_group_color(labels, label),
                   label=None if len(labels) == 1 else _group_legend(label, int(part["RegistrationCode"].nunique())))
    ax.axvline(frame["midpoint_hours_after_noon"].median(), color="0.5", linestyle="--", linewidth=0.8)
    ax.axhline(frame["asleep_minutes"].median() / 60, color="0.5", linestyle="--", linewidth=0.8)
    ticks = np.arange(np.floor(frame["midpoint_hours_after_noon"].min()), np.ceil(frame["midpoint_hours_after_noon"].max()) + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int((12 + t) % 24):02d}:00" for t in ticks], fontsize=7)
    ax.set_xlabel("sleep midpoint (local clock time)")
    ax.set_ylabel("total sleep time (h)")
    if len(labels) > 1:
        ax.legend(fontsize=7)
    how = "each participant's median night" if unit == "participant" else "every valid night"
    _titles(ax, "Sleep timing and duration", " · ".join(p for p in (
        _participants(int(frame["RegistrationCode"].nunique())), how, "dashed: cohort medians", _left_out(excluded)) if p))
    return _finish(fig, frame, path, dpi, individual=True)


def plot_sleep_architecture(metrics: "dm.DomainMetrics", *, rule: "dm.NightRule" = dm.NightRule(), groups=None,
                            min_participants: int = 1, path=None, dpi: int = 150):
    """
    Stage composition, wake after sleep onset and naps over valid nights, one value per participant: the median share
    of core, deep and REM sleep over staged nights, the median WASO, and the share of nights with a nap. Stages come
    from one device per night, the one that staged the most; the figure counts the nights two devices staged.
    """

    k = _check_min(min_participants)
    valid = _valid_nights(metrics, rule)
    per = valid.groupby("RegistrationCode").agg(nights=("asleep_minutes", "size"), waso_minutes=("waso_minutes", "median"),
                                                nap_share=("nap_minutes", lambda s: float((s.fillna(0) > 0).mean())))
    staged = valid[valid["staged"].astype(bool)]
    per = per.join(staged.groupby("RegistrationCode")[["core_share", "deep_share", "rem_share"]].median())
    per = per.join(staged.groupby("RegistrationCode").size().rename("staged_nights")).fillna({"staged_nights": 0}).reset_index()
    per["staged_nights"] = per["staged_nights"].astype(int)
    frame, labels, excluded = _grouped(per, groups)
    fig, axes = _figure(13.0, 4.4, ncols=3)
    ax = axes[0, 0]
    width = 0.7 / len(labels)
    hidden = 0
    for g, label in enumerate(labels):
        part = frame[frame["group"] == label]
        chosen = []
        for i, column in enumerate(("core_share", "deep_share", "rem_share")):
            v = part[column].dropna().to_numpy(dtype=float)
            if 0 < len(v) < k:
                hidden += 1
            elif len(v):
                chosen.append((i + (g - (len(labels) - 1) / 2) * width, v))
        if chosen:
            boxes = ax.boxplot([v for _, v in chosen], positions=[p for p, _ in chosen], widths=width * 0.85, patch_artist=True,
                               flierprops={"markersize": 2})
            for box in boxes["boxes"]:
                box.set_facecolor(_group_color(labels, label, SLEEP_COLORS["CORE"]))
                box.set_alpha(0.7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["core", "deep", "REM"])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("share of staged sleep (participant medians)")
    waso = frame.dropna(subset=["waso_minutes"])
    if len(waso):
        hidden += _histogram_panel(axes[0, 1], waso, "waso_minutes", np.arange(0, np.ceil(waso["waso_minutes"].max() / 10) * 10 + 20, 10), labels, k)
    axes[0, 1].set_xlabel("wake after sleep onset (minutes, participant medians)")
    axes[0, 1].set_ylabel("participants")
    hidden += _histogram_panel(axes[0, 2], frame, "nap_share", np.linspace(0, 1, 11), labels, k)
    axes[0, 2].set_xlabel("share of nights with a nap")
    if len(labels) > 1:
        axes[0, 1].legend(fontsize=7)
    two = int((valid["staging_sources"].fillna(0) > 1).sum())
    _titles(ax, "Sleep stages", f"{int((frame['staged_nights'] > 0).sum()):,} participants with staged nights · {len(staged):,} of {len(valid):,} valid nights staged")
    _titles(axes[0, 1], "Wake after sleep onset", " · ".join(p for p in (_participants(int(frame["RegistrationCode"].nunique())), _left_out(excluded)) if p))
    _titles(axes[0, 2], "Naps", " · ".join(p for p in (f"{two:,} nights staged by two devices: stages from the one that staged most",
                                                        _hidden(hidden, k, "box or bin")) if p))
    return _finish(fig, frame, path, dpi)


def plot_sleep_recording(metrics: "dm.DomainMetrics", *, show_ids: bool = False, path=None, dpi: int = 150):
    """
    What each participant's nights record (rows, most nights with time in bed only first): measured sleep, time in bed
    only, or neither, so that nights recording only time in bed are never mistaken for measured sleep.
    """

    nights = metrics.sleep_nights
    _need(len(nights) > 0, "the domain metrics hold no night")
    asleep, in_bed = nights["asleep_recorded"].astype(bool), nights["in_bed_recorded"].astype(bool)
    kind = np.where(asleep, "sleep recorded", np.where(in_bed, "time in bed only", "neither"))
    order = ["sleep recorded", "time in bed only", "neither"]
    counts = pd.crosstab(nights["RegistrationCode"], kind).reindex(columns=order, fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    rank = shares.sort_values(["time in bed only", "sleep recorded"], ascending=[False, True], kind="stable").index
    counts, shares = counts.loc[rank], shares.loc[rank]
    labels = _labels(counts.index, show_ids)
    fig, axes = _figure(9.0, max(3.0, 0.32 * len(counts) + 1.8))
    ax = axes[0, 0]
    left = np.zeros(len(counts))
    colors = {"sleep recorded": SLEEP_COLORS["CORE"], "time in bed only": SLEEP_COLORS["INBED"], "neither": "#fdd0a2"}
    for name in order:
        ax.barh(labels, shares[name].to_numpy(), left=left, color=colors[name], label=name)
        left += shares[name].to_numpy()
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the participant's nights")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    only = int(counts["time in bed only"].sum())
    total = int(counts.to_numpy().sum())
    _titles(ax, "What nights record", f"{only:,} of {total:,} nights ({only / total:.0%}) record time in bed only, not sleep")
    data = counts.add_suffix("_nights").join(shares.add_suffix("_share")).rename_axis("RegistrationCode").reset_index()
    return _finish(fig, data, path, dpi, individual=True)


# ------------------------------------------------------------------------------------------- glucose
def _glucose_axes(ax, top: float) -> None:
    ax.axhspan(70, 180, color="#e5f5e0", zorder=0)
    for level, style in ((54, ":"), (70, "--"), (180, "--"), (250, ":")):
        ax.axhline(level, color="0.55", linestyle=style, linewidth=0.8, zorder=1)
    ax.set_xlim(0, 24)
    ticks = np.arange(0, 25, 3)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t) % 24:02d}:00" for t in ticks], fontsize=7)
    ax.set_ylim(40, max(300.0, top))
    ax.set_xlabel("local time of day")
    ax.set_ylabel("glucose (mg/dL)")


def plot_agp(metrics: "dm.DomainMetrics", participant: str | None = None, *, groups=None, min_participants: int = 1,
             show_ids: bool = False, path=None, dpi: int = 150):
    """
    The Ambulatory Glucose Profile of one participant: over valid days, the 5th-95th and 25th-75th percentile bands and
    the median by time of day (``domain_metrics.cgm_profile``, unsmoothed), the 70-180 mg/dL target range shaded, and
    the consensus metrics beside it. With ``participant=None``, the cohort's glucose by time of day instead: the median
    across CGM participants of each one's median, with the interquartile range across participants. That is not an
    AGP, which describes one person, and the title says so.
    """

    k = _check_min(min_participants)
    profile = metrics.cgm_profile
    _need(len(profile) > 0, "the domain metrics hold no glucose profile: no CGM participant, or computed before output schema domain-2")
    step = int(profile["minute"].drop_duplicates().sort_values().diff().dropna().min()) if profile["minute"].nunique() > 1 else 1440
    profile = profile.assign(hour=(profile["minute"] + step / 2) / 60)
    if participant is not None:
        code = _code(participant)
        rows = profile[profile["RegistrationCode"] == code]
        _need(len(rows) > 0, f"no glucose profile for participant {code}")
        matches = metrics.cgm_periods[metrics.cgm_periods["RegistrationCode"] == code]
        _need(len(matches) > 0, f"no CGM metrics for participant {code}")
        period = matches.iloc[0]
        fig, axes = _figure(11.5, 5.0, 1, 2, width_ratios=[4, 1.25])
        ax, panel = axes[0, 0], axes[0, 1]
        ax.fill_between(rows["hour"], rows["p5"], rows["p95"], color="#c6dbef", label="5th-95th percentile", zorder=2)
        ax.fill_between(rows["hour"], rows["p25"], rows["p75"], color="#6baed6", label="25th-75th percentile", zorder=3)
        ax.plot(rows["hour"], rows["p50"], color="#08306b", linewidth=2, label="median", zorder=4)
        _glucose_axes(ax, float(np.nanmax(rows["p95"])) + 20)
        ax.legend(fontsize=7, loc="upper left")
        tbr = period["very_low_percent"] + period["low_percent"]
        tar = period["high_percent"] + period["very_high_percent"]
        lines = [f"valid days: {int(period['valid_days'])}", f"CGM active: {period['active_percent']:.0f}%",
                 f"mean glucose: {period['mean_mgdl']:.0f} mg/dL", f"GMI: {period['gmi_percent']:.1f}%",
                 f"CV: {period['cv_percent']:.1f}% (target <= {CV_STABILITY_THRESHOLD:g}%)", "",
                 f"time > 250: {period['very_high_percent']:.1f}%", f"time > 180: {tar:.1f}%",
                 f"time 70-180: {period['in_range_percent']:.1f}%", f"time < 70: {tbr:.1f}%", f"time < 54: {period['very_low_percent']:.1f}%"]
        if not bool(period["sufficient"]):
            lines += ["", f"fewer than the {dm.CGM_SUFFICIENT_DAYS} days", "the consensus requires"]
        panel.axis("off")
        panel.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8, family="monospace", transform=panel.transAxes)
        who = code if show_ids else "one participant"
        _titles(ax, "Ambulatory Glucose Profile", f"{who} · {int(period['valid_days'])} valid days · {step}-minute bins, unsmoothed")
        return _finish(fig, rows.drop(columns="hour"), path, dpi, individual=True,
                       panels={"metrics": matches.iloc[[0]].reset_index(drop=True)})
    frame, labels, excluded = _grouped(profile[["RegistrationCode", "minute", "hour", "p50"]], groups)
    table = _cohort(frame, "hour", "p50", k, None, 0)
    sizes = frame.groupby("group")["RegistrationCode"].nunique().to_dict()
    fig, ax, strip = _figure_with_strip(10.0, 5.4)
    _draw_cohort(ax, table, "hour", labels, sizes, ci=None, marker=None)
    _glucose_axes(ax, float(np.nanmax(table["p75"])) + 30 if table["p75"].notna().any() else 300.0)
    ax.set_xlabel("")
    _strip(strip, table, "hour", labels, width=step / 60 * 0.9)
    strip.set_xlabel("local time of day")
    ax.legend(fontsize=7, loc="upper left")
    _titles(ax, "Glucose by time of day across CGM participants (not an AGP)", " · ".join(p for p in (
        _participants(int(frame["RegistrationCode"].nunique())), "median and interquartile range of participants' medians",
        _left_out(excluded), _hidden(int(table["suppressed"].sum()), k, "bin")) if p))
    return _finish(fig, table, path, dpi)


def plot_glucose_days(readings: pd.DataFrame, *, valid_only: bool = True, show_ids: bool = False, path=None, dpi: int = 150):
    """
    One participant's glucose by time of day, one line per day, over valid days or every day, with the median by
    15-minute bin over them and the 70-180 mg/dL target range shaded.
    From ``domain_metrics.load_cgm_readings``.
    """

    _need(len(readings) > 0, "no glucose readings")
    _need(readings["RegistrationCode"].nunique() == 1, "daily glucose overlays draw one participant: filter the readings first")
    rows = readings[readings["valid_day"].astype(bool)] if valid_only else readings
    _need(len(rows) > 0, "no valid day of glucose readings")
    rows = rows.sort_values("local_time").copy()
    days = sorted(rows["local_date"].unique())
    rows["day_index"] = rows["local_date"].map({d: (pd.Timestamp(d) - pd.Timestamp(days[0])).days for d in days})
    rows["hour"] = rows["minute_of_day"] / 60
    fig, axes = _figure(10.0, 5.2)
    ax = axes[0, 0]
    for _, day in rows.groupby("day_index", sort=True):
        x, y = day["hour"].to_numpy(dtype=float), day["mgdl"].to_numpy(dtype=float)
        gaps = np.where(np.diff(x) > 0.5)[0] + 1
        x, y = np.insert(x, gaps, np.nan), np.insert(y, gaps, np.nan)
        ax.plot(x, y, color="0.45", linewidth=0.5, alpha=0.3)
    median = dm.cgm_profile(rows.assign(valid_day=True))
    ax.plot((median["minute"] + 7.5) / 60, median["p50"], color="#08306b", linewidth=2.2, label="median by 15 minutes")
    _glucose_axes(ax, float(rows["mgdl"].quantile(0.999)) + 20)
    ax.legend(fontsize=7, loc="upper left")
    who = rows["RegistrationCode"].iloc[0] if show_ids else "one participant"
    _titles(ax, "Glucose day by day", f"{who} · {len(days):,} {'valid ' if valid_only else ''}days · {len(rows):,} readings")
    return _finish(fig, rows[["day_index", "minute_of_day", "mgdl", "valid_day"]], path, dpi, individual=True,
                   panels={"median": median})


def plot_glycemic_cohort(metrics: "dm.DomainMetrics", *, sufficient_only: bool = True, groups=None,
                         min_participants: int = 1, path=None, dpi: int = 150):
    """
    CGM participants' glycaemia, one value per participant over valid days: the coefficient of variation against the
    36% stability threshold; mean glucose, with the glucose management indicator (a linear function of it) on the top
    axis; and the share of participants meeting each consensus target. The targets are those for most adults with
    diabetes. By default, only participants with the 14 valid days the consensus requires.
    """

    k = _check_min(min_participants)
    periods = metrics.cgm_periods
    cgm = periods[periods["cgm"].astype(bool) & (periods["valid_days"].fillna(0) > 0)] if len(periods) else periods
    if sufficient_only and len(cgm):
        cgm = cgm[cgm["sufficient"].astype(bool)]
    _need(len(cgm) > 0, "no CGM participant" + (" with the 14 valid days the consensus requires" if sufficient_only else ""))
    table = cgm[["RegistrationCode", "valid_days", "mean_mgdl", "gmi_percent", "cv_percent", *GLUCOSE_COLORS]].copy()
    table["below_70_percent"] = table["very_low_percent"] + table["low_percent"]
    table["above_180_percent"] = table["high_percent"] + table["very_high_percent"]
    compare = {">": np.greater, "<": np.less, "<=": np.less_equal}
    for name, column, op, limit in GLUCOSE_TARGETS:
        table[name] = compare[op](table[column].to_numpy(dtype=float), limit)
    frame, labels, excluded = _grouped(table, groups)
    fig, axes = _figure(14.0, 4.6, ncols=3)
    cv, mean, targets = axes[0, 0], axes[0, 1], axes[0, 2]
    values = frame["cv_percent"].to_numpy(dtype=float)
    hidden = _histogram_panel(cv, frame, "cv_percent", np.arange(np.floor(values.min() / 2) * 2, max(values.max(), CV_STABILITY_THRESHOLD) + 4, 2.0), labels, k)
    cv.axvline(CV_STABILITY_THRESHOLD, color="black", linestyle="--", linewidth=1)
    cv.set_xlabel("coefficient of variation (%)")
    cv.set_ylabel("participants")
    means = frame["mean_mgdl"].to_numpy(dtype=float)
    hidden += _histogram_panel(mean, frame, "mean_mgdl", np.arange(np.floor(means.min() / 5) * 5, means.max() + 10, 5.0), labels, k)
    mean.set_xlabel("mean glucose (mg/dL)")
    gmi = mean.secondary_xaxis("top", functions=(lambda m: 3.31 + 0.02392 * m, lambda g: (g - 3.31) / 0.02392))
    gmi.set_xlabel("GMI (%) = 3.31 + 0.02392 x mean glucose", fontsize=7)
    rows = []
    for label in labels:
        part = frame[frame["group"] == label]
        for name, *_ in GLUCOSE_TARGETS:
            n, met = len(part), int(part[name].sum())
            hide = 0 < n < k
            rows.append({"group": label, "target": name, "participants": np.nan if hide else float(n),
                         "meeting": np.nan if hide else float(met), "share": np.nan if hide else met / n, "suppressed": hide})
    shares = pd.DataFrame(rows)
    height = 0.8 / len(labels)
    names = [n for n, *_ in GLUCOSE_TARGETS]
    for i, label in enumerate(labels):
        part = shares[shares["group"] == label].set_index("target").reindex(names)
        offset = (i - (len(labels) - 1) / 2) * height
        bars = targets.barh(np.arange(len(names)) + offset, part["share"].fillna(0), height=height, color=_group_color(labels, label),
                            label=None if len(labels) == 1 else _group_legend(label, int((frame["group"] == label).sum())))
        if len(labels) == 1:
            targets.bar_label(bars, labels=["" if s else f"{int(m)} of {int(n)}" for m, n, s in
                                            zip(part["meeting"].fillna(0), part["participants"].fillna(0), part["suppressed"])], fontsize=7, padding=2)
    targets.set_yticks(range(len(names)))
    targets.set_yticklabels(names, fontsize=7)
    targets.invert_yaxis()
    targets.set_xlim(0, 1.25)
    targets.set_xlabel("share of participants meeting the target")
    if len(labels) > 1:
        targets.legend(fontsize=7)
    _titles(cv, "Glycaemic variability", " · ".join(p for p in (_participants(int(frame["RegistrationCode"].nunique())),
                                                               "dashed: 36% stability threshold", _left_out(excluded)) if p))
    mean.set_title("Mean glucose", loc="left", fontsize=11, pad=30)  # above the GMI axis
    _titles(targets, "Consensus targets", " · ".join(p for p in ("for most adults with diabetes",
                                                                  _hidden(hidden + int(shares["suppressed"].sum()), k, "bar or bin")) if p))
    return _finish(fig, frame, path, dpi, panels={"targets": shares})


def plot_cgm_wear(metrics: "dm.DomainMetrics", *, time_axis: str = "study", show_ids: bool = False, path=None, dpi: int = 150):
    """
    CGM wear: participants (rows) by day (columns), colored by the day's completeness, the share of expected
    readings present (valid from 70%). Sensor changes and gaps show as breaks. Days count from each participant's
    first unless ``time_axis="calendar"``.
    """

    _need(time_axis in ("calendar", "study"), "time_axis must be 'calendar' or 'study'")
    days = metrics.cgm_days
    _need(len(days) > 0, "the domain metrics hold no CGM day")
    days = days.assign(local_date=pd.to_datetime(days["local_date"]))
    days["study_day"] = (days["local_date"] - days.groupby("RegistrationCode")["local_date"].transform("min")).dt.days
    if time_axis == "calendar":
        columns = pd.date_range(days["local_date"].min(), days["local_date"].max(), freq="D")
        key = "local_date"
    else:
        columns = np.arange(0, int(days["study_day"].max()) + 1)
        key = "study_day"
    matrix = days.pivot_table(index="RegistrationCode", columns=key, values="completeness", aggfunc="first").reindex(columns=columns)
    matrix = matrix.loc[matrix.notna().sum(axis=1).sort_values(ascending=False, kind="stable").index]
    fig, axes = _figure(min(16.0, max(9.0, 0.012 * len(columns) + 6)), max(3.0, min(10.0, 0.35 * len(matrix) + 2)))
    ax = axes[0, 0]
    image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)), aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    bar = fig.colorbar(image, ax=ax, label="completeness of the day")
    threshold = dm.CGM_MIN_DAY_COMPLETENESS
    if threshold is not None:  # the completeness rule's threshold, always set for BloodGlucose
        bar.ax.axhline(threshold, color="white", linewidth=1.5)
    step = max(1, len(columns) // 12)
    ax.set_xticks(range(0, len(columns), step))
    ax.set_xticklabels([str(c.date()) if time_axis == "calendar" else str(c) for c in columns[::step]], rotation=90, fontsize=7)
    ax.set_xlabel("local date" if time_axis == "calendar" else "days since the first CGM day")
    labels = _labels(matrix.index, show_ids)
    ax.set_yticks(range(len(matrix)))
    ax.set_yticklabels(labels, fontsize=7)
    valid = int((days["completeness"] >= dm.CGM_MIN_DAY_COMPLETENESS).sum())
    _titles(ax, "CGM wear", f"{_participants(len(matrix))} · {valid:,} of {len(days):,} days valid (completeness >= {dm.CGM_MIN_DAY_COMPLETENESS:.0%})")
    data = days[["RegistrationCode", key, "readings", "completeness", "valid"] + (["study_day"] if key == "local_date" else [])]
    return _finish(fig, data, path, dpi, individual=True)


# ------------------------------------------------------------------------------------------ values
def _headline(feature: str, metrics: Iterable[str]) -> str:
    ranked = sm.headline_metrics(feature, list(dict.fromkeys(metrics)))
    _need(bool(ranked), f"{feature} has no metric to summarize")
    return ranked[0]


def _medians(summaries: "sm.Summaries", feature: str, metric: str | None = None) -> tuple[pd.DataFrame, str]:
    """Each participant's median of the feature's metric (its headline one by default) over valid days."""

    table = summaries.participant_metrics
    rows = table[table["feature"] == feature]
    _need(len(rows) > 0, f"the summaries hold no participant metric for {feature}")
    metric = metric or _headline(feature, rows["metric"])
    rows = rows[rows["metric"] == metric].dropna(subset=["median"])
    _need(len(rows) > 0, f"no participant median of {feature} {metric}")
    return rows, metric


def _unit_label(rows: pd.DataFrame) -> str:
    units = rows["unit"].dropna().astype(str) if "unit" in rows else pd.Series(dtype=str)
    return units.iloc[0] if len(units) else ""


def _edges(values: np.ndarray, bins: int = 20) -> np.ndarray:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.array([0.0, 1.0])
    if np.ptp(values) == 0:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    return np.histogram_bin_edges(values, bins=min(bins, max(5, len(values))))


def plot_metric_distributions(summaries: "sm.Summaries", features: Iterable[str] | None = None, *, groups=None,
                              min_participants: int = 1, path=None, dpi: int = 150):
    """
    A visual Table 1: for each feature, the distribution of participants' medians of its headline metric over valid
    days, one value per participant, with the median and interquartile range stated. ``figure.panels["table1"]``
    holds the numbers, per group when ``groups`` is given.
    """

    k = _check_min(min_participants)
    _need(len(summaries.participant_metrics) > 0, "the summaries hold no participant metric")
    names = feature_order(summaries.participant_metrics["feature"].unique() if features is None else features)
    chosen = []
    for feature in names:
        try:
            chosen.append(_medians(summaries, feature))
        except DataLoaderConfigurationError:
            continue
    _need(bool(chosen), "none of these features has participant medians")
    ncols = min(4, len(chosen))
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = _figure(3.4 * ncols, 3.0 * nrows, nrows, ncols)
    frames, stats, hidden, excluded_total = [], [], 0, 0
    for ax, (rows, metric) in zip(axes.flat, chosen):
        feature, unit = rows["feature"].iloc[0], _unit_label(rows)
        frame, labels, excluded = _grouped(rows[["RegistrationCode", "median"]].assign(feature=feature, metric=metric, unit=unit), groups)
        excluded_total = max(excluded_total, excluded)
        values = frame["median"].to_numpy(dtype=float)
        hidden += _histogram_panel(ax, frame, "median", _edges(values), labels, k)
        for label in labels:
            part = frame.loc[frame["group"] == label, "median"].to_numpy(dtype=float)
            hide = 0 < len(part) < k
            q = np.quantile(part, [0.25, 0.5, 0.75]) if len(part) and not hide else [np.nan] * 3
            stats.append({"feature": feature, "metric": metric, "unit": unit, "group": label,
                          "participants": np.nan if hide else float(len(part)), "median": q[1], "p25": q[0], "p75": q[2], "suppressed": hide})
        ax.set_xlabel(f"{metric} ({unit})" if unit else metric, fontsize=7)
        ax.tick_params(labelsize=7)
        q = np.quantile(values, [0.25, 0.5, 0.75])
        _titles(ax, feature, f"n={len(values):,} · median {q[1]:.3g} [{q[0]:.3g}, {q[2]:.3g}]" if len(values) >= k else f"fewer than {k} participants")
        if len(labels) > 1:
            ax.legend(fontsize=6)
        frames.append(frame)
    for ax in list(axes.flat)[len(chosen):]:
        ax.axis("off")
    notes = ["participants' medians over valid days", _left_out(excluded_total), _hidden(hidden, k, "bin")]
    fig.suptitle("Table 1 · " + " · ".join(n for n in notes if n), x=0.01, ha="left", fontsize=10)
    return _finish(fig, pd.concat(frames, ignore_index=True), path, dpi, panels={"table1": pd.DataFrame(stats)})


def plot_caterpillar(summaries: "sm.Summaries", feature: str, metric: str | None = None, *, show_ids: bool = False,
                     path=None, dpi: int = 150):
    """
    Each participant's median of the feature's headline metric over valid days (dot) and interquartile range (bar),
    sorted by median: how many participants differ, and how much each varies. The dashed line marks the cohort median.
    """

    rows, metric = _medians(summaries, feature, metric)
    rows = rows.sort_values(["median", "RegistrationCode"], kind="stable").reset_index(drop=True)
    unit = _unit_label(rows)
    fig, axes = _figure(8.5, max(3.5, min(12.0, 0.16 * len(rows) + 2.2)))
    ax = axes[0, 0]
    y = np.arange(len(rows))
    ax.hlines(y, rows["p25"], rows["p75"], color=feature_color(feature), linewidth=2, alpha=0.75)
    ax.plot(rows["median"], y, "o", color=SINGLE, markersize=3.5)
    ax.axvline(rows["median"].median(), color="0.5", linestyle="--", linewidth=1)
    if len(rows) <= 60:
        ax.set_yticks(y)
        ax.set_yticklabels(_labels(rows["RegistrationCode"], show_ids), fontsize=6)
    else:
        ax.set_yticks([])
        ax.set_ylabel("participants, by median")
    ax.set_xlabel(f"{metric} ({unit})" if unit else metric)
    _titles(ax, f"{feature}: each participant's median and interquartile range",
            f"{_participants(len(rows))} · valid days · dashed: cohort median")
    columns = ["RegistrationCode", "feature", "metric", "unit", "n_days", "median", "p25", "p75"]
    return _finish(fig, rows[columns], path, dpi, individual=True)


def plot_weekday_weekend(patterns: "sm.TemporalPatterns", feature: str, metric: str | None = None, *, paired: bool = True,
                         groups=None, min_participants: int = 1, path=None, dpi: int = 150):
    """
    Weekdays against the weekend, per participant (``temporal_patterns``' weekday and weekend medians): with
    ``paired``, a line per participant from their weekday median to their weekend median, and always the distribution
    of their weekend-minus-weekday differences. ``paired=False`` draws the differences alone, an aggregate.
    """

    k = _check_min(min_participants)
    table = patterns.weekday_weekend
    rows = table[table["feature"] == feature] if len(table) else table
    _need(len(rows) > 0, f"no weekday and weekend medians for {feature}")
    metric = metric or _headline(feature, rows["metric"])
    rows = rows[rows["metric"] == metric].dropna(subset=["weekday_median", "weekend_median"])
    _need(len(rows) > 0, f"no participant has both weekday and weekend {feature} {metric}")
    frame, labels, excluded = _grouped(rows, groups)
    fig, axes = _figure(11.5 if paired else 6.5, 4.6, 1, 2 if paired else 1)
    histogram = axes[0, -1]
    if paired:
        from matplotlib.collections import LineCollection
        ax = axes[0, 0]
        for label in labels:
            part = frame[frame["group"] == label]
            color = _group_color(labels, label)
            ax.add_collection(LineCollection([[(0, a), (1, b)] for a, b in zip(part["weekday_median"], part["weekend_median"])],
                                             colors=color, alpha=0.3, linewidths=0.8))
            ax.plot([0, 1], [part["weekday_median"].median(), part["weekend_median"].median()], color=color, linewidth=2.5, marker="o",
                    label="cohort median" if len(labels) == 1 else _group_legend(label, len(part)))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["weekdays", "weekend"])
        ax.set_xlim(-0.2, 1.2)
        low = min(frame["weekday_median"].min(), frame["weekend_median"].min())
        high = max(frame["weekday_median"].max(), frame["weekend_median"].max())
        ax.set_ylim(low - 0.05 * (high - low + 1), high + 0.05 * (high - low + 1))
        ax.set_ylabel(f"{metric} (participant medians)")
        ax.legend(fontsize=7)
        _titles(ax, f"{feature}: weekdays and weekend", _participants(len(frame)))
    differences = frame["weekend_minus_weekday"].to_numpy(dtype=float)
    hidden = _histogram_panel(histogram, frame, "weekend_minus_weekday", _edges(differences), labels, k)
    histogram.axvline(0, color="black", linewidth=0.8)
    histogram.set_xlabel(f"weekend minus weekdays ({metric})")
    histogram.set_ylabel("participants")
    if len(labels) > 1 and not paired:
        histogram.legend(fontsize=7)
    weekend = ", ".join(WEEKDAY_NAMES[d] for d in sorted(patterns.weekend_days))
    _titles(histogram, "Weekend minus weekdays", " · ".join(p for p in (
        f"weekend: {weekend}", f"median difference {np.median(differences):.3g}" if len(differences) >= k else "",
        _left_out(excluded), _hidden(hidden, k, "bin")) if p))
    return _finish(fig, frame, path, dpi, individual=paired)


def _hour_measure(source, feature: str, measure: str) -> tuple[str, str]:
    """The hourly column a measure reads (as ``plot_hour_of_day`` chooses it) and its label."""

    _need(measure in ("value", "records", "coverage"), "measure must be 'value', 'records' or 'coverage'")
    if measure == "records":
        return "records_per_day", "records per day"
    if measure == "coverage":
        return "day_share", "share of days with data"
    if ds.measurement_kind(feature) in ("extensive_total", "event_amount"):
        return "value_per_day", f"amount per day ({_hour_unit(source, feature)})"
    return "value_mean", f"mean value ({_hour_unit(source, feature)})"


def _profiles(participants: pd.DataFrame, feature: str, column: str) -> pd.DataFrame:
    rows = participants[participants["feature"] == feature] if len(participants) else participants
    if rows.empty:
        return pd.DataFrame()
    return rows.pivot(index="RegistrationCode", columns="hour", values=column).reindex(columns=range(24)).astype(float).dropna(how="all")


def plot_hour_profiles(source, feature: str, *, measure: str = "value", order_by: str = "trough", relative: bool = True,
                       show_ids: bool = False, path=None, dpi: int = 150):
    """
    Each participant's profile over the local hours of the day (rows), relative to their own mean by default, ordered
    by the hour of their lowest value (``order_by="trough"``, such as heart rate's nightly minimum) or highest
    (``"peak"``), marked on each row: a chronotype-like view of when each participant's day happens. Blank: no data.
    """

    _need(order_by in ("trough", "peak"), "order_by must be 'trough' or 'peak'")
    column, label = _hour_measure(source, feature, measure)
    participants, _ = sm.hour_of_day(source)
    _need(len(participants) > 0 and feature in set(participants["feature"]), f"no hour-of-day data for {feature}")
    matrix = _profiles(participants, feature, column)
    _need(len(matrix) > 0, f"no {measure} by hour of day for {feature}")
    shown = matrix.div(matrix.mean(axis=1).where(lambda m: m != 0), axis=0) if relative else matrix
    shown = shown.dropna(how="all")
    _need(len(shown) > 0, f"no participant has a {measure} profile of {feature} to draw")
    values = shown.to_numpy(dtype=float)
    key = np.where(np.isnan(values), np.inf if order_by == "trough" else -np.inf, values)
    extreme = key.argmin(axis=1) if order_by == "trough" else key.argmax(axis=1)
    order = np.lexsort((np.arange(len(shown)), extreme))
    shown, extreme = shown.iloc[order], extreme[order]
    fig, axes = _figure(9.5, max(4.0, min(12.0, 0.16 * len(shown) + 2.5)))
    ax = axes[0, 0]
    data = shown.to_numpy(dtype=float)
    if relative:
        spread = float(np.nanquantile(np.abs(data - 1), 0.98)) if np.isfinite(data).any() else 0.5
        spread = spread if np.isfinite(spread) and spread > 0 else 0.5
        image = ax.imshow(np.ma.masked_invalid(data), aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=1 - spread, vmax=1 + spread)
        fig.colorbar(image, ax=ax, label=f"{label}, relative to the participant's mean")
    else:
        image = ax.imshow(np.ma.masked_invalid(data), aspect="auto", interpolation="nearest", cmap="viridis")
        fig.colorbar(image, ax=ax, label=label)
    ax.scatter(extreme, np.arange(len(shown)), s=10, color="black", marker="|")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("local hour of day")
    if show_ids and len(shown) <= 60:
        ax.set_yticks(range(len(shown)))
        ax.set_yticklabels(shown.index, fontsize=7)
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"participants, by the hour of their {'lowest' if order_by == 'trough' else 'highest'} value")
    _titles(ax, f"{feature}: each participant's day", f"{_participants(len(shown))} · marks: the hour of each one's "
                                                     f"{'lowest' if order_by == 'trough' else 'highest'} value")
    long = shown.reset_index().melt(id_vars="RegistrationCode", var_name="hour", value_name="relative" if relative else column)
    return _finish(fig, long, path, dpi, individual=True,
                   panels={"order": pd.DataFrame({"RegistrationCode": shown.index, f"{order_by}_hour": extreme})})


def plot_daily_rhythms(source, features: Iterable[str] | None = None, *, measure: str = "value", min_participants: int = 1,
                       path=None, dpi: int = 150):
    """
    Small multiples of the cohort's daily rhythm per feature: each participant's hour-of-day profile relative to their
    own mean (1 is their average hour), then the median and interquartile range across participants at each hour, so
    that features in different units share one scale. Features without the measure are named, not drawn.
    """

    k = _check_min(min_participants)
    participants, _ = sm.hour_of_day(source)
    _need(len(participants) > 0, "the run holds no hour-of-day data")
    names = feature_order(participants["feature"].unique() if features is None else features)
    panels, skipped = [], []
    for feature in names:
        column, _ = _hour_measure(source, feature, measure)
        matrix = _profiles(participants, feature, column)
        matrix = matrix[matrix.mean(axis=1) > 0] if len(matrix) else matrix
        if matrix.empty:
            skipped.append(feature)
            continue
        relative = matrix.div(matrix.mean(axis=1), axis=0).reset_index().melt(id_vars="RegistrationCode", var_name="hour", value_name="relative")
        relative = relative.dropna().assign(group=ALL_PARTICIPANTS)
        panels.append((feature, _cohort(relative, "hour", "relative", k, None, 0).assign(feature=feature)))
    _need(bool(panels), f"no feature has a {measure} profile by hour of day")
    ncols = min(4, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = _figure(3.3 * ncols, 2.6 * nrows, nrows, ncols)
    for ax, (feature, table) in zip(axes.flat, panels):
        color = feature_color(feature)
        ax.fill_between(table["hour"], table["p25"], table["p75"], color=color, alpha=0.25, linewidth=0)
        ax.plot(table["hour"], table["median"], color=color, linewidth=1.6)
        ax.axhline(1.0, color="0.6", linewidth=0.8, linestyle=":")
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 6))
        ax.tick_params(labelsize=7)
        n = int(table["participants"].max()) if table["participants"].notna().any() else 0
        _titles(ax, feature, f"n={n:,}" + (" · some hours hidden" if table["suppressed"].any() else ""))
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    note = f" · no {measure} profile: {', '.join(skipped)}" if skipped else ""
    fig.suptitle(f"Daily rhythms: {measure} by local hour, relative to each participant's mean{note}", x=0.01, ha="left", fontsize=10)
    return _finish(fig, pd.concat([t for _, t in panels], ignore_index=True), path, dpi)


# ---------------------------------------------------------------------------------- clinical references
# Daily steps, each a half-open interval [lower, upper).
STEP_CATEGORIES = (("Sedentary", 0.0, 5000.0), ("Low active", 5000.0, 7500.0), ("Somewhat active", 7500.0, 10000.0),
                   ("Active", 10000.0, 12500.0), ("Highly active", 12500.0, np.inf))
# The WHO's weekly moderate activity for adults (2020 guidelines: 150-300 minutes).
WHO_WEEKLY_MINUTES = 150.0
# ActivitySummary's rings: the day's value and goal columns.
ACTIVITY_RINGS = (("move", "active_energy_burned_mean", "active_energy_burned_goal_mean"),
                  ("exercise", "apple_exercise_time_mean", "apple_exercise_time_goal_mean"),
                  ("stand", "apple_stand_hours_mean", "apple_stand_hours_goal_mean"))
# Blood-pressure categories, each with the systolic and diastolic pressure (mmHg) at which it starts. A participant
# falls in the highest category either pressure reaches, the guidelines' "and/or"; None: no diastolic criterion.
# ESC/ESH defines hypertension at home from 135/85 mmHg (2018 and 2023 guidelines), stricter than in the office.
BP_GUIDELINES = {
    "esc_esh_home": (("Normal (home)", 0.0, 0.0), ("Hypertension (home)", 135.0, 85.0)),
    "esc_esh_office": (("Optimal", 0.0, 0.0), ("Normal", 120.0, 80.0), ("High normal", 130.0, 85.0),
                       ("Grade 1 hypertension", 140.0, 90.0), ("Grade 2 hypertension", 160.0, 100.0),
                       ("Grade 3 hypertension", 180.0, 110.0)),
    "acc_aha": (("Normal", 0.0, 0.0), ("Elevated", 120.0, None), ("Stage 1 hypertension", 130.0, 80.0),
                ("Stage 2 hypertension", 140.0, 90.0)),
}


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce") if name in frame else pd.Series(np.nan, index=frame.index)


def _category_table(values: pd.DataFrame, column: str, categories, groups) -> pd.DataFrame:
    frame, labels, _ = _grouped(values, groups)
    rows = []
    for label in labels:
        part = frame[frame["group"] == label]
        for name, lower, upper in categories:
            n = int(((part[column] >= lower) & (part[column] < upper)).sum())
            rows.append({"group": label, "category": name, "lower": lower, "upper": upper, "participants": n,
                         "share": n / len(part) if len(part) else np.nan})
    return pd.DataFrame(rows)


def _category_bars(ax, table: pd.DataFrame, names: list[str], ticks: list[str], k: int) -> pd.DataFrame:
    table = table.copy()
    table["participants"] = table["participants"].astype(float)
    table["suppressed"] = (table["participants"] > 0) & (table["participants"] < k)
    table.loc[table["suppressed"], ["participants", "share"]] = np.nan
    labels = list(dict.fromkeys(table["group"]))
    width = 0.8 / len(labels)
    for i, label in enumerate(labels):
        part = table[table["group"] == label].set_index("category").reindex(names)
        bars = ax.bar(np.arange(len(names)) + (i - (len(labels) - 1) / 2) * width, part["participants"].fillna(0), width=width,
                      color=_group_color(labels, label), label=None if len(labels) == 1 else label)
        if len(labels) == 1:
            ax.bar_label(bars, labels=["" if s else f"{int(n)} ({sh:.0%})" for n, sh, s in
                                       zip(part["participants"].fillna(0), part["share"].fillna(0), part["suppressed"])], fontsize=7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(ticks, fontsize=7)
    ax.set_ylabel("participants")
    if len(labels) > 1:
        ax.legend(fontsize=7)
    return table


def _range_label(lower: float, upper: float) -> str:
    if lower == 0:
        return f"< {upper:,.0f}"
    if np.isinf(upper):
        return f">= {lower:,.0f}"
    return f"{lower:,.0f} to < {upper:,.0f}"


def step_categories(summaries: "sm.Summaries", *, groups=None) -> pd.DataFrame:
    """Participants per daily-step category (``STEP_CATEGORIES``), from each participant's median daily steps."""

    rows, _ = _medians(summaries, "StepCount")
    return _category_table(rows[["RegistrationCode", "median"]], "median", STEP_CATEGORIES, groups)


def plot_step_categories(summaries: "sm.Summaries", *, groups=None, min_participants: int = 1, path=None, dpi: int = 150):
    """Participants per daily-step category (Tudor-Locke), from each participant's median daily steps over valid days."""

    k = _check_min(min_participants)
    table = step_categories(summaries, groups=groups)
    fig, axes = _figure(9.0, 4.5)
    ax = axes[0, 0]
    names = [n for n, *_ in STEP_CATEGORIES]
    table = _category_bars(ax, table, names, [f"{n}\n{_range_label(lo, hi)}" for n, lo, hi in STEP_CATEGORIES], k)
    _titles(ax, "Daily steps (Tudor-Locke categories)", " · ".join(p for p in (
        _participants(int(table["participants"].sum())), "each participant's median over valid days",
        _hidden(int(table["suppressed"].sum()), k, "bar")) if p))
    return _finish(fig, table, path, dpi)


def activity_goals(source, *, weekend_days: Iterable[int] = sm.WEEKEND_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ``(participants, weeks)`` from ActivitySummary. Weeks start the day after the weekend (Sunday with the default
    Friday-Saturday weekend) and are complete when all seven days have exercise minutes. Per participant: complete
    weeks, their median weekly exercise minutes, the share reaching ``WHO_WEEKLY_MINUTES``, and for each ring the days
    with a goal set (a goal of 0 is none) and the share of them on which it was met.
    """

    src = sm._Source(source)
    _need("ActivitySummary" in src.features, "ActivitySummary is not among the run's features")
    start = (max(int(d) for d in weekend_days) + 1) % 7
    people, weeks = [], []
    for daily in src.participants_of("ActivitySummary"):
        code = str(daily["RegistrationCode"].iloc[0])
        dates = pd.to_datetime(daily["local_date"])
        week = dates - pd.to_timedelta((dates.dt.weekday - start) % 7, unit="D")
        exercise = _column(daily, "apple_exercise_time_mean")
        per_week = pd.DataFrame({"week": week.to_numpy(), "minutes": exercise.to_numpy()}).groupby("week").agg(
            days=("minutes", "size"), known=("minutes", "count"), minutes=("minutes", "sum"))
        per_week["complete"] = (per_week["days"] == 7) & (per_week["known"] == 7)
        complete = per_week[per_week["complete"]]
        weeks.append(per_week.reset_index().assign(RegistrationCode=code))
        row = {"RegistrationCode": code, "days": len(daily), "complete_weeks": len(complete),
               "weekly_exercise_minutes": float(complete["minutes"].median()) if len(complete) else np.nan,
               "weeks_meeting_who": float((complete["minutes"] >= WHO_WEEKLY_MINUTES).mean()) if len(complete) else np.nan}
        for ring, value, goal in ACTIVITY_RINGS:
            v, g = _column(daily, value), _column(daily, goal)
            has = (g > 0).to_numpy()
            row[f"{ring}_goal_days"] = int(has.sum())
            row[f"{ring}_goal_met"] = float((v[has] >= g[has]).mean()) if has.any() else np.nan
        people.append(row)
    weeks = pd.concat(weeks, ignore_index=True) if weeks else pd.DataFrame(columns=["week", "days", "known", "minutes", "complete", "RegistrationCode"])
    return pd.DataFrame(people), weeks.rename(columns={"week": "week_start", "minutes": "exercise_minutes"})


def plot_activity_goals(source, *, weekend_days: Iterable[int] = sm.WEEKEND_DAYS, groups=None, min_participants: int = 1,
                        path=None, dpi: int = 150):
    """
    Activity against the WHO guideline and Apple's rings (``activity_goals``). Left: each participant's median weekly
    exercise minutes over complete weeks, against the WHO's 150 minutes a week; Apple's exercise minutes, brisk activity,
    approximate moderate-to-vigorous activity. Middle: the share of each participant's complete weeks reaching 150.
    Right: the share of days each ring's goal was met, over days with a goal set.
    """

    k = _check_min(min_participants)
    people, weeks = activity_goals(source, weekend_days=weekend_days)
    _need(len(people) > 0, "no participant has ActivitySummary")
    frame, labels, excluded = _grouped(people, groups)
    fig, axes = _figure(14.0, 4.5, 1, 3)
    hidden = 0
    weekly = frame.dropna(subset=["weekly_exercise_minutes"])
    if len(weekly):
        top = max(float(weekly["weekly_exercise_minutes"].max()), WHO_WEEKLY_MINUTES) + 30
        hidden += _histogram_panel(axes[0, 0], weekly, "weekly_exercise_minutes", np.arange(0, top + 30, 30.0), labels, k)
    axes[0, 0].axvline(WHO_WEEKLY_MINUTES, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_xlabel("median weekly exercise minutes")
    axes[0, 0].set_ylabel("participants")
    share = frame.dropna(subset=["weeks_meeting_who"])
    if len(share):
        hidden += _histogram_panel(axes[0, 1], share, "weeks_meeting_who", np.linspace(0, 1, 11), labels, k)
    axes[0, 1].set_xlabel("share of complete weeks with at least 150 minutes")
    ax = axes[0, 2]
    width = 0.7 / len(labels)
    for g, label in enumerate(labels):
        part = frame[frame["group"] == label]
        boxes = []
        for i, (ring, *_) in enumerate(ACTIVITY_RINGS):
            v = part[f"{ring}_goal_met"].dropna().to_numpy(dtype=float)
            if 0 < len(v) < k:
                hidden += 1
            elif len(v):
                boxes.append((i + (g - (len(labels) - 1) / 2) * width, v))
        if boxes:
            drawn = ax.boxplot([v for _, v in boxes], positions=[p for p, _ in boxes], widths=width * 0.85, patch_artist=True, flierprops={"markersize": 2})
            for box in drawn["boxes"]:
                box.set_facecolor(_group_color(labels, label, CATEGORY_COLORS["Physical activity and energy"]))
                box.set_alpha(0.7)
    ax.set_xticks(range(len(ACTIVITY_RINGS)))
    ax.set_xticklabels([r for r, *_ in ACTIVITY_RINGS])
    ax.set_xlim(-0.6, len(ACTIVITY_RINGS) - 0.4)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("share of days the goal was met")
    complete = int(frame["complete_weeks"].sum())
    _titles(axes[0, 0], "Weekly exercise (WHO: 150 minutes)", " · ".join(p for p in (
        _participants(int(frame["RegistrationCode"].nunique())), f"{complete:,} complete weeks", _left_out(excluded)) if p))
    _titles(axes[0, 1], "Weeks reaching the guideline", "Apple exercise minutes approximate moderate-to-vigorous activity")
    _titles(ax, "Apple ring goals", " · ".join(p for p in ("days with a goal set", _hidden(hidden, k, "box or bin")) if p))
    return _finish(fig, frame, path, dpi, panels={"weeks": weeks})


def bp_category(systolic, diastolic, guideline: str = "esc_esh_home") -> np.ndarray:
    """The ``BP_GUIDELINES`` category of each systolic and diastolic pair (mmHg): the highest either pressure reaches."""

    _need(guideline in BP_GUIDELINES, f"guideline must be one of {', '.join(BP_GUIDELINES)}")
    categories = BP_GUIDELINES[guideline]
    s, d = np.asarray(systolic, dtype=float), np.asarray(diastolic, dtype=float)
    index = np.zeros(len(s), dtype=int)
    for i, (_, systolic_from, diastolic_from) in enumerate(categories[1:], start=1):
        reached = (s >= systolic_from) | ((d >= diastolic_from) if diastolic_from is not None else False)
        index = np.where(reached, i, index)
    names = np.array([c[0] for c in categories], dtype=object)[index]
    names[np.isnan(s) | np.isnan(d)] = None
    return names


def blood_pressure(summaries: "sm.Summaries", *, guideline: str = "esc_esh_home") -> pd.DataFrame:
    """Each participant's median systolic and diastolic pressure over valid days, and their category."""

    metrics = summaries.participant_metrics
    pressure = metrics[metrics["feature"] == "BloodPressure"] if len(metrics) else metrics
    systolic = pressure[pressure["metric"] == "blood_pressure_systolic_value_mean"].set_index("RegistrationCode")
    diastolic = pressure[pressure["metric"] == "blood_pressure_diastolic_value_mean"].set_index("RegistrationCode")
    _need(len(systolic) > 0 and len(diastolic) > 0, "the summaries hold no blood pressure")
    table = pd.DataFrame({"systolic": systolic["median"], "diastolic": diastolic["median"], "days": systolic["n_days"]}).dropna(
        subset=["systolic", "diastolic"]).rename_axis("RegistrationCode").reset_index()
    table["category"] = bp_category(table["systolic"], table["diastolic"], guideline)
    return table


def plot_blood_pressure(summaries: "sm.Summaries", *, guideline: str = "esc_esh_home", points: bool = True, groups=None,
                        min_participants: int = 1, path=None, dpi: int = 150):
    """
    Blood-pressure categories from each participant's median systolic and diastolic pressure over valid days, under a
    selectable guideline: ESC/ESH home thresholds (the default, since HealthKit readings are mostly taken at home:
    hypertension from 135/85 mmHg), ESC/ESH office grades, or ACC/AHA stages. With ``points``, each participant's
    medians against the thresholds.
    """

    k = _check_min(min_participants)
    table = blood_pressure(summaries, guideline=guideline)
    frame, labels, excluded = _grouped(table, groups)
    names = [c[0] for c in BP_GUIDELINES[guideline]]
    counts = pd.DataFrame([{"group": label, "category": name, "participants": int((frame.loc[frame["group"] == label, "category"] == name).sum()),
                            "share": float((frame.loc[frame["group"] == label, "category"] == name).mean())}
                           for label in labels for name in names])
    fig, axes = _figure(13.0 if points else 8.0, 4.8, 1, 2 if points else 1)
    counts = _category_bars(axes[0, 0], counts, names, [n.replace(" hypertension", "\nhypertension").replace(" (home)", "\n(home)") for n in names], k)
    if points:
        ax = axes[0, 1]
        for label in labels:
            part = frame[frame["group"] == label]
            ax.scatter(part["diastolic"], part["systolic"], s=24, color=_group_color(labels, label),
                       label=None if len(labels) == 1 else _group_legend(label, len(part)))
        for name, systolic_from, diastolic_from in BP_GUIDELINES[guideline][1:]:
            ax.axhline(systolic_from, color="0.7", linewidth=0.8, linestyle="--")
            if diastolic_from is not None:
                ax.axvline(diastolic_from, color="0.7", linewidth=0.8, linestyle="--")
        systolic_from = [s for _, s, _ in BP_GUIDELINES[guideline][1:]]
        diastolic_from = [d for _, _, d in BP_GUIDELINES[guideline][1:] if d is not None]
        ax.set_ylim(min(frame["systolic"].min(), min(systolic_from)) - 5, max(frame["systolic"].max(), max(systolic_from)) + 5)
        ax.set_xlim(min(frame["diastolic"].min(), min(diastolic_from)) - 5, max(frame["diastolic"].max(), max(diastolic_from)) + 5)
        ax.set_xlabel("diastolic (mmHg, participant median)")
        ax.set_ylabel("systolic (mmHg, participant median)")
        if len(labels) > 1:
            ax.legend(fontsize=7)
        _titles(ax, "Each participant's medians", "dashed: the guideline's thresholds")
    title = {"esc_esh_home": "ESC/ESH home", "esc_esh_office": "ESC/ESH office", "acc_aha": "ACC/AHA"}[guideline]
    _titles(axes[0, 0], f"Blood pressure ({title})", " · ".join(p for p in (
        _participants(len(frame)), "median over valid days", _left_out(excluded), _hidden(int(counts["suppressed"].sum()), k, "bar")) if p))
    return _finish(fig, frame, path, dpi, individual=points, panels={"categories": counts})


def plot_weight_trajectories(source, *, time_axis: str = "study", rules=None, show_ids: bool = False, path=None, dpi: int = 150):
    """
    Each participant's weight as percentage change from their first valid measurement, over study days (or calendar
    dates), with the cohort's median change per 30-day period since each participant's first measurement.
    """

    _need(time_axis in ("calendar", "study"), "time_axis must be 'calendar' or 'study'")
    rows, _, unit = _daily_rows(source, "Weight", None, True, rules, None)
    rows = rows.sort_values(["RegistrationCode", "local_date"], kind="stable").reset_index(drop=True)
    first = rows.groupby("RegistrationCode")["value"].transform("first")
    rows["percent_change"] = 100.0 * (rows["value"] / first - 1.0)
    rows["study_day"] = _study_day(rows)
    rows["period"] = rows["study_day"] // 30
    cohort = rows.groupby("period").agg(participants=("RegistrationCode", "nunique"), median=("percent_change", "median"))
    # Every 30-day period since the first, so that the cohort line breaks where no participant was weighed.
    cohort = cohort.reindex(range(int(rows["period"].max()) + 1)).rename_axis("period").reset_index()
    cohort["participants"] = cohort["participants"].fillna(0).astype(int)
    from matplotlib.collections import LineCollection
    from matplotlib.dates import date2num
    fig, axes = _figure(10.0, 5.0)
    ax = axes[0, 0]
    x = rows["study_day"].to_numpy(dtype=float) if time_axis == "study" else date2num(rows["local_date"])
    segments = [np.column_stack([x[idx], rows["percent_change"].to_numpy()[idx]]) for idx in rows.groupby("RegistrationCode").indices.values()]
    ax.add_collection(LineCollection(segments, colors="0.55", linewidths=0.8, alpha=0.6))
    ax.plot(x, rows["percent_change"], ".", color="0.4", markersize=2)
    if time_axis == "study":
        ax.plot(cohort["period"] * 30 + 15, cohort["median"], color=feature_color("Weight"), linewidth=2.5, marker="o", markersize=3,
                label="cohort median per 30 days")
        ax.legend(fontsize=7)
        ax.set_xlabel("days since the first valid measurement")
    else:
        ax.xaxis_date()
        ax.set_xlabel("local date")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.autoscale_view()
    ax.set_ylabel(f"change from the first measurement (%, weight in {unit})")
    _titles(ax, "Weight trajectories", f"{_participants(rows['RegistrationCode'].nunique())} · {len(rows):,} valid days")
    columns = ["RegistrationCode", "study_day", "value", "percent_change"] + (["local_date"] if time_axis == "calendar" else [])
    return _finish(fig, rows[columns], path, dpi, individual=True, panels={"cohort": cohort})


THRESHOLD_LABELS = {"readings_below_90_percent": "SpO2 below 90%", "readings_at_least_38_celsius": "Body temperature at least 38.0 C"}


def plot_clinical_thresholds(source, *, min_participants: int = 1, path=None, dpi: int = 150):
    """
    Readings beyond clinical thresholds (``data_statistics.CLINICAL_THRESHOLDS``, such as SpO2 below 90%): for each,
    the distribution of the share of each participant's readings beyond it, with how many participants had any and the
    cohort's total. Wrist sensors produce occasional artifacts, so isolated readings deserve scrutiny before
    interpretation.
    """

    k = _check_min(min_participants)
    participants, cohort = sm.clinical_thresholds(source)
    _need(len(participants) > 0, "no participant has readings with a clinical threshold in its unit")
    names = list(dict.fromkeys(participants["threshold"]))
    fig, axes = _figure(5.6 * len(names), 4.4, 1, len(names))
    hidden = 0
    for ax, name in zip(axes[0], names):
        part = participants[participants["threshold"] == name].assign(group=ALL_PARTICIPANTS)
        part = part.assign(percent=100.0 * part["share_beyond"])
        top = max(1.0, float(np.nanmax(part["percent"]))) if part["percent"].notna().any() else 1.0
        hidden += _histogram_panel(ax, part.dropna(subset=["percent"]), "percent", np.linspace(0, top, 21), [ALL_PARTICIPANTS], k)
        ax.set_xlabel("% of the participant's readings beyond the threshold")
        c = cohort[cohort["threshold"] == name].iloc[0]
        anyone = int(c["participants_beyond"])
        told = f"{anyone:,} of {int(c['participants']):,} participants had any" if anyone == 0 or anyone >= k else f"fewer than {k} participants had any"
        _titles(ax, str(THRESHOLD_LABELS.get(name, name)), f"{told} · {int(c['beyond']):,} of {int(c['readings']):,} readings ({c['share_beyond']:.1%})")
    axes[0, 0].set_ylabel("participants")
    if hidden:
        fig.suptitle(_hidden(hidden, k, "bin"), x=0.99, ha="right", fontsize=8)
    return _finish(fig, participants, path, dpi, panels={"cohort": cohort})


# ------------------------------------------------------------------------------------------ multimodal
OVERVIEW_FEATURES = ("StepCount", "RestingHeartRate", "Sleep", "Weight", "BloodGlucose")


def plot_participant_overview(source, participant: str, *, features: Iterable[str] | None = None,
                              domain: "dm.DomainMetrics | None" = None, rules=None, time_axis: str = "study",
                              show_ids: bool = False, path=None, dpi: int = 150):
    """
    One participant on one time axis: which features have data each day (top), then the daily headline values of the
    chosen features (by default steps, resting heart rate, sleep, weight and glucose), valid days filled and other days
    hollow. With ``domain``, sleep is total sleep per night and glucose each CGM day's mean. Study days unless
    ``time_axis="calendar"``.
    """

    _need(time_axis in ("calendar", "study"), "time_axis must be 'calendar' or 'study'")
    code = _code(participant)
    src = sm._Source(source)
    days = next((d for d in src.grouped("participant_days") if str(d["RegistrationCode"].iloc[0]) == code), None)
    if days is None:
        raise DataLoaderConfigurationError(f"no data for participant {code}")
    first = pd.to_datetime(days["local_date"]).min()
    position = (lambda dates: (pd.to_datetime(dates) - first).dt.days.to_numpy(dtype=float)) if time_axis == "study" else (lambda dates: pd.to_datetime(dates).to_numpy())
    present = feature_order({f for text in days["features"] for f in str(text).split(";") if f})
    chosen = [f for f in (OVERVIEW_FEATURES if features is None else features) if f in present]
    series = []
    for feature in chosen:
        if feature == "Sleep" and domain is not None and (domain.sleep_nights["RegistrationCode"] == code).any():
            nights = domain.sleep_nights[(domain.sleep_nights["RegistrationCode"] == code) & domain.sleep_nights["asleep_recorded"].astype(bool)]
            table = pd.DataFrame({"x": position(nights["night_date"]), "value": nights["asleep_minutes"].to_numpy() / 60,
                                  "valid": (nights["asleep_minutes"] >= dm.NightRule().min_asleep_minutes).to_numpy()})
            series.append((feature, "total sleep per night (h)", table))
        elif feature == "BloodGlucose" and domain is not None and (domain.cgm_days["RegistrationCode"] == code).any():
            cgm = domain.cgm_days[domain.cgm_days["RegistrationCode"] == code]
            series.append((feature, "mean glucose per day (mg/dL)", pd.DataFrame({"x": position(cgm["local_date"]), "value": cgm["mean_mgdl"].to_numpy(),
                                                                                  "valid": cgm["valid"].astype(bool).to_numpy()})))
        else:
            try:
                rows, metric, unit = _daily_rows(source, feature, None, False, rules, code)
            except DataLoaderConfigurationError:
                continue
            series.append((feature, f"{metric} ({unit})", pd.DataFrame({"x": position(rows["local_date"]), "value": rows["value"].to_numpy(),
                                                                         "valid": rows["valid"].to_numpy()})))
    heights = [max(1.2, 0.22 * len(present))] + [1.0] * len(series)
    fig, axes = _figure(11.0, 1.2 + 1.6 * len(series) + 0.2 * len(present), 1 + len(series), 1, height_ratios=heights)
    top = axes[0, 0]
    for i, feature in enumerate(present):
        mask = days["features"].map(lambda s, f=feature: f in str(s).split(";")).to_numpy()
        top.plot(position(days.loc[mask, "local_date"]), np.full(int(mask.sum()), i), "|", color=feature_color(feature), markersize=6)
    top.set_yticks(range(len(present)))
    top.set_yticklabels(present, fontsize=6)
    top.set_ylim(len(present) - 0.5, -0.5)
    long = []
    for ax, (feature, label, table) in zip(axes[1:, 0], series):
        ax.sharex(top)
        color = feature_color(feature)
        order = np.argsort(table["x"].to_numpy())
        table = table.iloc[order]
        ax.plot(table["x"], table["value"], color=color, linewidth=0.6, alpha=0.6)
        ax.plot(table.loc[table["valid"], "x"], table.loc[table["valid"], "value"], "o", color=color, markersize=2.2)
        other = table[~table["valid"]]
        if len(other):
            ax.plot(other["x"], other["value"], "o", markerfacecolor="none", color=color, markersize=2.2)
        ax.set_ylabel(label, fontsize=7)
        ax.tick_params(labelsize=7)
        long.append(table.assign(feature=feature, label=label))
    axes[-1, 0].set_xlabel("days since the first day with data" if time_axis == "study" else "local date")
    who = code if show_ids else "one participant"
    _titles(top, "Participant overview", f"{who} · {len(days):,} days with data · {len(present)} features")
    data = pd.concat(long, ignore_index=True) if long else pd.DataFrame(columns=["x", "value", "valid", "feature", "label"])
    availability = days.assign(x=position(days["local_date"]))[["x", "features"]]
    return _finish(fig, data, path, dpi, individual=True, panels={"availability": availability})


def _day_series(source, name: str, domain, rules) -> tuple[pd.DataFrame, str]:
    if name == "sleep":
        _need(domain is not None, "the sleep series needs domain metrics: pass domain=compute_domain_metrics(...)")
        nights = domain.sleep_nights
        valid = nights[nights["asleep_recorded"].astype(bool) & (nights["asleep_minutes"] >= dm.NightRule().min_asleep_minutes)]
        return (pd.DataFrame({"RegistrationCode": valid["RegistrationCode"].astype(str).to_numpy(), "date": pd.to_datetime(valid["night_date"]).to_numpy(),
                              "value": (valid["asleep_minutes"] / 60).to_numpy()}), "total sleep of the night (h)")
    rows, metric, unit = _daily_rows(source, name, None, True, rules, None)
    return rows.rename(columns={"local_date": "date"})[["RegistrationCode", "date", "value"]], f"{name} {metric} ({unit})"


def plot_lagged_association(source, x: str, y: str, *, lag_days: int = 0, domain: "dm.DomainMetrics | None" = None,
                            rules=None, points: bool = True, min_days: int = 10, min_participants: int = 1, path=None,
                            dpi: int = 150):
    """
    The within-person association between two daily series, ``x`` on day d and ``y`` on day d + ``lag_days``: each
    participant's deviations from their own means over the paired days, so that differences between people cannot pose
    as effects within them. ``"sleep"`` names total sleep of the night starting on day d (from ``domain``); other names
    are features of the run over valid days. With ``lag_days=0``, steps on day d against sleep that night; with
    ``x="sleep"`` and ``lag_days=1``, sleep against the next day's resting heart rate. Left: the pooled deviations and
    the within-person slope; right: each participant's own correlation. Participants need ``min_days`` paired days.
    """

    k = _check_min(min_participants)
    _need(isinstance(lag_days, (int, np.integer)) and abs(int(lag_days)) <= 30, "lag_days must be a whole number of days, at most 30")
    xs, xlabel = _day_series(source, x, domain, rules)
    ys, ylabel = _day_series(source, y, domain, rules)
    shifted = ys.assign(date=ys["date"] - pd.Timedelta(int(lag_days), unit="D"))
    pairs = xs.merge(shifted, on=["RegistrationCode", "date"], suffixes=("_x", "_y")).rename(columns={"value_x": "x", "value_y": "y"})
    counts = pairs.groupby("RegistrationCode").size()
    pairs = pairs[pairs["RegistrationCode"].isin(counts[counts >= min_days].index)].copy()
    _need(len(pairs) > 0, f"no participant has {min_days} days with both {x} and {y}")
    pairs["x_dev"] = pairs["x"] - pairs.groupby("RegistrationCode")["x"].transform("mean")
    pairs["y_dev"] = pairs["y"] - pairs.groupby("RegistrationCode")["y"].transform("mean")
    denominator = float((pairs["x_dev"] ** 2).sum())
    slope = float((pairs["x_dev"] * pairs["y_dev"]).sum() / denominator) if denominator > 0 else np.nan
    within_r = float(np.corrcoef(pairs["x_dev"], pairs["y_dev"])[0, 1]) if len(pairs) > 2 and pairs["x_dev"].std() > 0 and pairs["y_dev"].std() > 0 else np.nan
    per = []
    for code, part in pairs.groupby("RegistrationCode", sort=True):
        r = float(np.corrcoef(part["x"], part["y"])[0, 1]) if part["x"].std() > 0 and part["y"].std() > 0 else np.nan
        per.append({"RegistrationCode": code, "days": len(part), "r": r})
    per = pd.DataFrame(per).assign(group=ALL_PARTICIPANTS)
    fig, axes = _figure(12.0 if points else 6.5, 4.8, 1, 2 if points else 1)
    if points:
        ax = axes[0, 0]
        ax.scatter(pairs["x_dev"], pairs["y_dev"], s=5, alpha=0.25, color=SINGLE)
        if np.isfinite(slope):
            span = np.array([pairs["x_dev"].min(), pairs["x_dev"].max()])
            ax.plot(span, slope * span, color=PALETTE[5], linewidth=2, label=f"within-person slope {slope:.3g}")
            ax.legend(fontsize=7)
        ax.axhline(0, color="0.6", linewidth=0.8)
        ax.axvline(0, color="0.6", linewidth=0.8)
        ax.set_xlabel(f"{xlabel}, deviation from own mean", fontsize=8)
        ax.set_ylabel(f"{ylabel}, deviation from own mean", fontsize=8)
        _titles(ax, "Within-person association", f"{x} on day d, {y} on day d{int(lag_days):+d} · {len(pairs):,} paired days")
    histogram = axes[0, -1]
    hidden = _histogram_panel(histogram, per.dropna(subset=["r"]), "r", np.linspace(-1, 1, 21), [ALL_PARTICIPANTS], k)
    histogram.axvline(0, color="black", linewidth=0.8)
    histogram.set_xlabel("each participant's correlation")
    histogram.set_ylabel("participants")
    _titles(histogram, "Per participant", " · ".join(p for p in (
        _participants(len(per)), f"pooled within-person r {within_r:.2f}" if np.isfinite(within_r) else "",
        _hidden(hidden, k, "bin")) if p))
    summary = pd.DataFrame([{"x": x, "y": y, "lag_days": int(lag_days), "participants": len(per), "paired_days": len(pairs),
                             "within_slope": slope, "within_r": within_r}])
    return _finish(fig, pairs[["RegistrationCode", "x", "y", "x_dev", "y_dev"]], path, dpi, individual=points,
                   panels={"participants": per.drop(columns="group"), "summary": summary})


def plot_feature_correlations(summaries: "sm.Summaries", features: Iterable[str] | None = None, *, method: Literal["spearman", "pearson"] = "spearman",
                              min_participants: int = 3, path=None, dpi: int = 150):
    """
    Correlations across participants between features' headline metrics (each participant's median over valid days),
    with the participants behind each pair; a pair resting on fewer than ``min_participants`` (at least 3) is left
    blank. These are between-person associations: they say nothing about how features move together within a person
    (``plot_lagged_association``).
    """

    _need(method in ("spearman", "pearson"), "method must be 'spearman' or 'pearson'")
    k = max(3, _check_min(min_participants))
    names = feature_order(summaries.participant_metrics["feature"].unique() if features is None else features)
    columns = {}
    for feature in names:
        try:
            rows, _ = _medians(summaries, feature)
        except DataLoaderConfigurationError:
            continue
        columns[feature] = rows.set_index("RegistrationCode")["median"]
    _need(len(columns) >= 2, "correlations need at least two features with participant medians")
    wide = pd.DataFrame(columns)
    names = list(wide.columns)
    present = wide.notna().astype(int)
    n = present.T.dot(present)
    r = wide.corr(method=method).where(n >= k)
    fig, axes = _figure(max(6.0, 0.45 * len(names) + 3), max(5.0, 0.45 * len(names) + 2))
    ax = axes[0, 0]
    image = ax.imshow(r.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    fig.colorbar(image, ax=ax, label=f"{method} correlation of participant medians")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    if len(names) <= 14:
        cells = r.to_numpy(dtype=float)
        for i in range(len(names)):
            for j in range(len(names)):
                value = cells[i, j]
                if i != j and np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}\nn={n.iat[i, j]}", ha="center", va="center", fontsize=5.5,
                            color="white" if abs(value) > 0.6 else "black")
    blank = int(((n < k) & ~np.eye(len(names), dtype=bool)).to_numpy().sum() // 2)
    _titles(ax, "Correlations across participants", " · ".join(p for p in (
        f"{wide.shape[0]:,} participants", "between people, not within them",
        f"{blank} pairs with fewer than {k} participants left blank" if blank else "") if p))
    data = r.rename_axis("feature_a").reset_index().melt(id_vars="feature_a", var_name="feature_b", value_name="r")
    data["participants"] = n.rename_axis("feature_a").reset_index().melt(id_vars="feature_a", var_name="feature_b", value_name="n")["n"].to_numpy()
    data["suppressed"] = data["participants"] < k
    return _finish(fig, data, path, dpi)
