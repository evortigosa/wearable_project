"""
Wearable Data Processing and Modeling project
Visualization helpers that validate inputs, avoid mutating callers, and return figures.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Literal
import numpy as np
import pandas as pd
from .policies import canonical_feature_name


def _plot_imports() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtransforms
        import seaborn as sns
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:
        raise ImportError(
            "Plotting requires optional dependencies. Install with `pip install -e '.[plots]'`."
        ) from exc
    return plt, sns, mdates, mtransforms, MaxNLocator


def _require_columns(frame:pd.DataFrame, columns:set[str], context:str) -> None:
    missing= columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{context} is missing required columns: {sorted(missing)}")


def _utc_boundary(value:Any) -> pd.Timestamp:
    timestamp= pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _read_interval_values(file_path:str|Path, start_date:Any, end_date:Any, *, day_sum:bool,
                          value_threshold:float|None,) -> pd.DataFrame:
    frame= pd.read_csv(file_path)
    _require_columns(frame, {"start_date", "end_date", "value"}, str(file_path))
    frame= frame.copy()
    frame["start_date"]= pd.to_datetime(frame["start_date"], errors="coerce", utc=True)
    frame["end_date"]= pd.to_datetime(frame["end_date"], errors="coerce", utc=True)
    frame["value"]= pd.to_numeric(frame["value"], errors="coerce")
    frame= frame.dropna(subset=["start_date", "end_date", "value"])
    lower, upper= _utc_boundary(start_date), _utc_boundary(end_date)
    if upper <= lower:
        raise ValueError("end_date must be later than start_date.")
    # Include any interval that overlaps the requested range.
    interval_overlap= frame["end_date"].gt(lower) & frame["start_date"].lt(upper)
    point_in_range= (
        frame["start_date"].eq(frame["end_date"])
        & frame["start_date"].ge(lower)
        & frame["start_date"].lt(upper)
    )
    frame= frame.loc[interval_overlap|point_in_range].copy()
    if day_sum:
        frame= (
            frame.assign(start_date=frame["start_date"].dt.floor("D"))
            .groupby("start_date", as_index=False)["value"]
            .sum(min_count=1)
        )
    if value_threshold is not None:
        frame["value"]= frame["value"].clip(upper=value_threshold)
    frame["year_month"]= (frame["start_date"].dt.tz_localize(None).dt.to_period("M").astype(str))

    return frame


def map_features_by_id(file_path:str|Path, n_top:int= 50, random_sample:bool= False, *, show:bool= True,):
    plt, sns, _, _, _= _plot_imports()
    frame= pd.read_csv(file_path)
    _require_columns(frame, {"participant_id"}, str(file_path))
    if frame.empty:
        raise ValueError("Feature-presence table is empty.")
    if n_top <= 0:
        raise ValueError("n_top must be positive.")
    feature_columns= [
        column for column in frame.columns if column not in {"participant_id", "feature_count"}
    ]
    numeric= frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    frame= frame.assign(feature_count=numeric.sum(axis=1))
    sample_size= min(max(1, n_top), len(frame))
    sample= (
        frame.sample(n=sample_size, random_state=42)
        if random_sample
        else frame.nlargest(sample_size, "feature_count")
    )
    sample_features= numeric.loc[sample.index].set_axis(sample["participant_id"], axis=0)
    fig, ax= plt.subplots(figsize=(12, 8))
    sns.heatmap(sample_features, cbar=False, ax=ax)
    ax.set(title="Feature presence by participant", xlabel="Feature", ylabel="Participant ID")
    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax


def vis_ft_count_per_user(file_path:str|Path, *, show:bool= True):
    plt, sns, _, _, _= _plot_imports()
    frame= pd.read_csv(file_path)
    lower= {column.casefold(): column for column in frame.columns}
    if "feature" not in lower or "count" not in lower:
        raise ValueError("Expected columns named 'feature' and 'count'.")
    feature, count= lower["feature"], lower["count"]
    frame= frame.sort_values(count, ascending=True)
    fig, ax= plt.subplots(figsize=(10, 6))
    sns.barplot(data=frame, x=count, y=feature, ax=ax)
    ax.set(title="Participants containing each feature", xlabel="Participant count", ylabel="Feature")
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def vis_active_devices(active_file:str|Path, start_date:Any, end_date:Any, yyyy_mm_x:bool= True, *, show:bool= True,):
    plt, _, mdates, _, MaxNLocator= _plot_imports()
    frame= pd.read_csv(active_file)
    _require_columns(frame, {"date"}, str(active_file))
    value_column= "active_participants" if "active_participants" in frame.columns else "active_devices"
    _require_columns(frame, {value_column}, str(active_file))
    frame= frame.copy()
    frame["date"]= pd.to_datetime(frame["date"], errors="coerce")
    frame[value_column]= pd.to_numeric(frame[value_column], errors="coerce")
    frame= frame.dropna(subset=["date", value_column])
    frame= frame.loc[
        frame["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date), inclusive="both")
    ]
    fig, ax= plt.subplots(figsize=(12, 6))
    ax.fill_between(frame["date"], frame[value_column], alpha=0.35)
    ax.plot(frame["date"], frame[value_column], linewidth=1.5)
    ax.set(title="Daily active participants", xlabel="Date", ylabel="Active participants")
    if yyyy_mm_x:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.autofmt_xdate()
    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax


def activities_per_folder(file_path:str|Path, n_bins_to_keep:int|None= None, *, show:bool= True,):
    plt, sns, _, _, _= _plot_imports()
    frame= pd.read_csv(file_path)
    _require_columns(frame, {"participant_id"}, str(file_path))
    feature_columns= [column for column in frame.columns if column != "participant_id"]
    feature_columns= [column for column in feature_columns if column != "feature_count"]
    counts_per_participant= frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    histogram= counts_per_participant.value_counts().sort_index()
    if n_bins_to_keep is not None and n_bins_to_keep > 0 and len(histogram) > n_bins_to_keep:
        histogram= histogram.iloc[-n_bins_to_keep:]
    fig, ax= plt.subplots(figsize=(10, 6))
    sns.barplot(x=histogram.index.astype(str), y=histogram.values, ax=ax)
    ax.set(
        title="Distribution of feature counts per participant",
        xlabel="Features present",
        ylabel="Participants",
    )
    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax


def dual_feature_month_boxplot(file1:str|Path, file2:str|Path, start_date:Any, end_date:Any, label1:str,
                               label2:str, day_sum:bool= False, value_threshold1:float|None= None,
                               value_threshold2:float|None= None, *, show:bool= True,):
    plt, sns, _, _, _= _plot_imports()
    first= _read_interval_values(
        file1, start_date, end_date, day_sum=day_sum, value_threshold=value_threshold1
    ).assign(feature=label1)
    second= _read_interval_values(
        file2, start_date, end_date, day_sum=day_sum, value_threshold=value_threshold2
    ).assign(feature=label2)
    combined= pd.concat([first, second], ignore_index=True)
    fig, ax= plt.subplots(figsize=(14, 7))
    sns.boxplot(x="year_month", y="value", hue="feature", data=combined, showfliers=True, ax=ax)
    ax.set(title=f"Monthly distribution of {label1} and {label2}", xlabel="Month", ylabel="Value")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax


def feature_month_boxplot(file:str|Path, start_date:Any, end_date:Any, label:str, day_sum:bool= False,
                          value_threshold:float|None= None, *, show:bool= True,):
    plt, sns, _, _, _= _plot_imports()
    frame= _read_interval_values(
        file, start_date, end_date, day_sum=day_sum, value_threshold=value_threshold
    )
    fig, ax= plt.subplots(figsize=(12, 6))
    sns.boxplot(x="year_month", y="value", data=frame, showfliers=True, ax=ax)
    ax.set(title=f"Monthly distribution of {label}", xlabel="Month", ylabel=label)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax


def feature_value_evolving(file:str|Path, start_date:Any, end_date:Any, ylabel:str= "Steps",
                           value_threshold:float|None= None, interactive:bool= True, *,
                           aggregation:Literal["sum", "mean", "median"]= "sum", show:bool= True,):
    if aggregation not in {"sum", "mean", "median"}:
        raise ValueError("aggregation must be 'sum', 'mean', or 'median'.")
    frame= _read_interval_values(
        file, start_date, end_date, day_sum=False, value_threshold=value_threshold
    )
    daily= (
        frame.assign(date=frame["start_date"].dt.floor("D"))
        .groupby("date", as_index=False)["value"]
        .agg(aggregation)
    )
    if interactive:
        try:
            import plotly.express as px
        except ImportError as exc:
            raise ImportError("Interactive plots require `pip install -e '.[plots]'`.") from exc
        fig= px.area(
            daily,
            x="date",
            y="value",
            title=f"Daily {aggregation} of {ylabel}",
            labels={"date": "Date", "value": ylabel},
        )
        if show:
            fig.show()
        return fig

    plt, _, _, _, _= _plot_imports()
    fig, ax= plt.subplots(figsize=(10, 6))
    ax.fill_between(daily["date"], daily["value"], alpha=0.35)
    ax.plot(daily["date"], daily["value"], linewidth=1.5)
    ax.set(title=f"Daily {aggregation} of {ylabel}", xlabel="Date", ylabel=ylabel)
    fig.autofmt_xdate()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def activ_duration_feature_evolving(df:pd.DataFrame, feature_name:str, start_date:Any, end_date:Any, *,
                                    show:bool= True,):
    plt, _, _, _, _= _plot_imports()
    frame= df.copy()
    _require_columns(frame, {feature_name}, "duration DataFrame")
    frame.index= pd.to_datetime(frame.index, errors="coerce")
    frame= frame.loc[frame.index.to_series().between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    values= pd.to_numeric(frame[feature_name], errors="coerce") / 3600
    fig, ax= plt.subplots(figsize=(12, 5))
    ax.fill_between(frame.index, values, alpha=0.35)
    ax.plot(frame.index, values, linewidth=1.5)
    ax.set(title=f"{feature_name} coverage over time", xlabel="Date", ylabel="Coverage (hours)")
    fig.autofmt_xdate()
    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax


def plot_evolution(df:pd.DataFrame, start_date:Any, end_date:Any, top_n:int= 10, *, show:bool= True,):
    plt, sns, _, _, _= _plot_imports()
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    frame= df.copy()
    frame.index= pd.to_datetime(frame.index, errors="coerce")
    frame= frame.loc[frame.index.notna()].sort_index()
    frame= frame.loc[frame.index.to_series().between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    monthly= frame.resample("ME").sum(numeric_only=True)
    if monthly.empty or monthly.shape[1] == 0:
        raise ValueError("No numeric data are available for the selected period.")
    top= monthly.sum().nlargest(top_n).index
    fig, ax= plt.subplots(figsize=(12, 6))
    sns.heatmap(monthly.loc[:, top].T, ax=ax)
    ax.set(title="Feature coverage over time", xlabel="Month", ylabel="Feature")
    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax


def plot_agg_totals(df_counts:pd.DataFrame, df_durations:pd.DataFrame, start_date:Any, end_date:Any,
                    top_n:int= 10, *, show:bool= True,):
    plt, _, _, _, _= _plot_imports()
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    counts= df_counts.copy()
    durations= df_durations.copy()
    counts.index= pd.to_datetime(counts.index, errors="coerce")
    durations.index= pd.to_datetime(durations.index, errors="coerce")
    lower, upper= pd.Timestamp(start_date), pd.Timestamp(end_date)
    counts= counts.loc[counts.index.to_series().between(lower, upper)]
    durations= durations.loc[durations.index.to_series().between(lower, upper)]
    top_counts= counts.sum(numeric_only=True).nlargest(top_n).sort_values()
    top_durations= (durations.sum(numeric_only=True) / 3600).nlargest(top_n).sort_values()
    fig, axes= plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(top_counts.index, top_counts.values)
    axes[0].set(title="Top features by row count", xlabel="Rows")
    axes[1].barh(top_durations.index, top_durations.values)
    axes[1].set(title="Top features by temporal coverage", xlabel="Coverage (hours)")
    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes


def plot_heatmap(df:pd.DataFrame, start_date:Any, end_date:Any, *, show:bool= True,):
    plt, sns, _, _, _= _plot_imports()
    frame= df.copy()
    frame.index= pd.to_datetime(frame.index, errors="coerce")
    frame= frame.loc[frame.index.to_series().between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    if frame.empty:
        raise ValueError("No data are available for the selected period.")
    fig, ax= plt.subplots(figsize=(14, 6))
    sns.heatmap(frame.T, ax=ax)
    ax.set(title=f"Data collection heatmap ({start_date} to {end_date})", xlabel="Date", ylabel="Hour")
    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax


def data_sizes_by_folders(file_path:str|Path, *, show:bool= True):
    plt, sns, _, mtransforms, _= _plot_imports()
    frame= pd.read_csv(file_path)
    lower= {column.casefold(): column for column in frame.columns}
    if "folder_size_bytes" not in lower and "storage_bytes" not in lower:
        raise ValueError("Expected a 'folder_size_bytes' or 'storage_bytes' column.")
    size_column= lower.get("folder_size_bytes", lower.get("storage_bytes"))
    sizes= pd.to_numeric(frame[size_column], errors="coerce").dropna().clip(lower=0)
    if sizes.empty:
        raise ValueError("No valid folder-size values are available.")

    figures= []
    fig_hist, ax_hist= plt.subplots(figsize=(12, 6))
    sns.histplot(x=sizes, bins=min(100, max(10, int(np.sqrt(len(sizes))))), kde=len(sizes) > 1, ax=ax_hist)
    ax_hist.set(title="Distribution of participant storage", xlabel="Bytes", ylabel="Participants")
    fig_hist.tight_layout()
    figures.append((fig_hist, ax_hist))

    fig_box, ax_box= plt.subplots(figsize=(8, 4))
    sns.boxplot(x=sizes, ax=ax_box)
    if sizes.gt(0).any():
        ax_box.set_xscale("log")
    ax_box.set(title="Participant storage distribution", xlabel="Bytes")
    fig_box.tight_layout()
    figures.append((fig_box, ax_box))

    sorted_sizes= np.sort(sizes.to_numpy())
    count= len(sorted_sizes)
    thresholds= [1e6, 10e6, 100e6, 500e6]
    fig_scatter, ax_scatter= plt.subplots(figsize=(12, 6))
    ax_scatter.plot(sorted_sizes, marker="o", linestyle="", markersize=2)
    if np.any(sorted_sizes > 0):
        ax_scatter.set_yscale("log")
    ax_scatter.set(title="Participant storage, sorted", xlabel="Participant rank", ylabel="Bytes")
    for threshold in thresholds:
        index= int(np.searchsorted(sorted_sizes, threshold, side="right"))
        percentage= index / count * 100
        ax_scatter.axvline(index, linestyle="--", alpha=0.5)
        offset= mtransforms.ScaledTranslation(0.05, 0, fig_scatter.dpi_scale_trans)
        ax_scatter.text(index, max(float(sorted_sizes.min()), 1.0), f"<{threshold / 1e6:g} MB: {percentage:.1f}%", rotation=90, transform=ax_scatter.transData + offset)
    fig_scatter.tight_layout()
    figures.append((fig_scatter, ax_scatter))
    if show:
        plt.show()

    return figures


def plot_monthly_data_trends(df:pd.DataFrame, start_date:Any, end_date:Any, remove_first_year_label:bool= True, *,
                             show:bool= True,):
    """ Plots the evolution of data volume over time for multiple participants. """
    plt, sns, _, _, _= _plot_imports()
    frame= df.copy()
    frame.columns= pd.to_datetime(frame.columns, format="%Y-%m", errors="coerce")
    frame= frame.loc[:, frame.columns.notna()]
    lower, upper= pd.Timestamp(start_date), pd.Timestamp(end_date)
    frame= frame.loc[:, (frame.columns >= lower) & (frame.columns <= upper)]
    monthly= frame.apply(pd.to_numeric, errors="coerce").sum(axis=0)
    if monthly.empty:
        raise ValueError("No monthly columns are available for the selected period.")
    fig, ax= plt.subplots(figsize=(14, 6))
    sns.lineplot(x=monthly.index, y=monthly.values, marker="o", ax=ax)
    years= monthly.index.year
    unique_years= list(dict.fromkeys(years))
    if remove_first_year_label and len(unique_years) > 1:
        unique_years= unique_years[1:]
    ticks= [monthly.index[years == year][0] for year in unique_years]
    ax.set_xticks(ticks, labels=unique_years)
    ax.set(title="Total data volume over time", xlabel="Year", ylabel="Bytes")
    fig.tight_layout()
    if show:
        plt.show()

    return fig, ax


def get_bmi_stats(data_dir_path:str|Path, *, print_summary:bool= True) -> pd.DataFrame:
    """
    Return adult BMI categories from each participant's median BMI.
    These population cutoffs are descriptive and are not a clinical diagnosis. The
    function separates underweight values instead of classifying all BMI < 25 as healthy.
    """

    path= Path(data_dir_path)
    if path.is_file():
        file_path= path
    else:
        candidates= [
            candidate
            for candidate in path.glob("*_timeseries.csv")
            if canonical_feature_name(candidate.stem.removesuffix("_timeseries"))
            in {"bmi", "bodymassindex"}
        ]
        if not candidates:
            raise FileNotFoundError(f"No BMI time-series CSV was found in {path}.")
        file_path= sorted(candidates, key=lambda candidate: candidate.name.casefold())[0]
    frame= pd.read_csv(file_path)
    _require_columns(frame, {"participant_id", "value"}, str(file_path))
    values= pd.to_numeric(frame["value"], errors="coerce")
    median= frame.assign(value=values).dropna(subset=["value"]).groupby("participant_id")["value"].median()
    categories= pd.cut(
        median,
        bins=[-np.inf, 18.5, 25.0, 30.0, np.inf],
        right=False,
        labels=["Underweight", "Healthy range", "Overweight", "Obesity"],
    )
    counts= categories.value_counts(sort=False).rename("participants").reset_index(names="category")
    total= int(counts["participants"].sum())
    counts["percentage"]= counts["participants"].div(total).mul(100) if total else 0.0
    if print_summary:
        print(counts.to_string(index=False, formatters={"percentage": "{:.1f}%".format}))

    return counts


def norm_df_by_active_devices(df_to_norm:pd.DataFrame, df_active_devices:pd.DataFrame, *,
                              missing:Literal["nan", "zero"]= "nan",) -> pd.DataFrame:
    """ Normalize numeric columns by active participants using vectorized division. """

    values= df_to_norm.copy()
    active= df_active_devices.copy()
    if "date" in values.columns and not isinstance(values.index, pd.DatetimeIndex):
        values= values.set_index("date")
    if "date" in active.columns and not isinstance(active.index, pd.DatetimeIndex):
        active= active.set_index("date")
    values.index= pd.to_datetime(values.index, errors="coerce")
    active.index= pd.to_datetime(active.index, errors="coerce")
    denominator_column= (
        "active_participants" if "active_participants" in active.columns else "active_devices"
    )
    _require_columns(active, {denominator_column}, "active-participant DataFrame")
    denominator= pd.to_numeric(active[denominator_column], errors="coerce")
    denominator= denominator.groupby(level=0).max().reindex(values.index)
    denominator= denominator.where(denominator.gt(0))
    numeric_columns= values.select_dtypes(include="number").columns
    values.loc[:, numeric_columns]= (
        values.loc[:, numeric_columns].astype(float).div(denominator, axis=0)
    )
    if missing == "zero":
        values.loc[:, numeric_columns]= values.loc[:, numeric_columns].fillna(0)
    elif missing != "nan":
        raise ValueError("missing must be 'nan' or 'zero'.")

    return values
