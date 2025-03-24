import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms
import seaborn as sns
import plotly.express as px
from matplotlib.ticker import MaxNLocator


def map_features_by_id(file_path, n_top=50, random_sample=False):
    """
    - n_top (int): Define the number of top participants to sample.
    - random_sample (bool): If True, sample a subset of rows, otherwise, 
    select the top n_top folders with the largest feature_count.
    """
    df= pd.read_csv(file_path)

    feature_cols = df.columns.drop("participant_id")
    df["feature_count"] = df[feature_cols].sum(axis=1)

    if random_sample:
        # Sample a subset of rows (e.g., 50 participants)
        sample_df = df.sample(n=n_top, random_state=42).set_index("participant_id")
    else:
        # Select the top n_top folders with the largest feature_count.
        sample_df = df.nlargest(n_top, 'feature_count').set_index("participant_id")
    # Only include the binary feature columns
    sample_features = sample_df[feature_cols]

    plt.figure(figsize=(12, 8))
    sns.heatmap(sample_features, cmap="Blues", cbar=False)
    plt.title("Heatmap of Feature Presence (Sample of Participants)")
    plt.xlabel("Feature")
    plt.ylabel("Participant ID")
    plt.tight_layout()
    plt.show()


def vis_ft_count_per_user(ft_file):
    df_ft_count= pd.read_csv(ft_file)
    col_feature= [col for col in df_ft_count.columns if col.lower() == "feature"][0]
    col_count  = [col for col in df_ft_count.columns if col.lower() == "count"][0]
    # Sort DataFrame by count ascending (for horizontal bar chart, smallest at the bottom)
    df_sorted = df_ft_count.sort_values(col_count, ascending=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_sorted, x=col_count, y=col_feature, hue=col_feature, palette="viridis", 
        dodge=False
    )
    plt.legend([], [], frameon=False)
    plt.title("Feature Appearance Count per User Folder")
    plt.xlabel("Count")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def vis_active_devices(active_file, start_date, end_date, yyyy_mm_x=True):
    df_active = pd.read_csv(active_file)
    df_active['date'] = pd.to_datetime(df_active['date'])
    # Filter the DataFrame based on provided date interval
    df_active = df_active[df_active['date'] >= pd.to_datetime(start_date)]
    df_active = df_active[df_active['date'] <= pd.to_datetime(end_date)]

    plt.figure(figsize=(12, 6))
    plt.fill_between(df_active['date'], df_active['active_devices'], color="skyblue", alpha=0.6)
    plt.plot(df_active['date'], df_active['active_devices'], color="Slateblue", linewidth=2)
    plt.title("Daily Active Devices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Active Devices")
    
    ax = plt.gca()
    if yyyy_mm_x:
        # Set x-axis: one tick per month with format YYYY-MM
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        # Set x-axis: one tick per year with format YYYY
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # Set y-axis to display only integer ticks
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def activities_per_folder(file_path, n_bins_to_keep=None):
    """
    n_bins_to_keep (int, optional): If provided and the total number of bins (unique activity 
    counts) is greater than this number, then the lower bins will be set to zero (i.e. removed)
    so that only the highest bins are highlighted in the plot
    """
    features_per_folder= pd.read_csv(file_path)
    feature_cols = features_per_folder.columns.drop("participant_id")
    # Compute the number of features (activities) present for each folder.
    features_per_folder['feature_count'] = features_per_folder[feature_cols].sum(axis=1)
    # Compute the counts per bin (each unique feature_count value)
    counts = features_per_folder['feature_count'].value_counts().sort_index()
    # If n_bins_to_keep is provided and there are more bins than desired,
    # then determine how many of the lowest bins to remove.
    if n_bins_to_keep is not None and counts.size > n_bins_to_keep:
        bins_to_remove = counts.size - n_bins_to_keep
        # Set the counts for the lower bins to 0.
        counts.iloc[:bins_to_remove] = 0

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, color="royalblue")
    plt.xlabel("Number of Activities Present")
    plt.ylabel("Number of Folders")
    plt.title("Distribution of Activities per Folder")
    plt.xticks(range(0, counts.size))
    plt.tight_layout()
    plt.show()


def dual_feature_month_boxplot(
    file1, file2, start_date, end_date, label1, label2, day_sum=False, value_threshold1=None, 
    value_threshold2=None
):
    def load_and_process(file, value_threshold):
        df = pd.read_csv(file)
        df['start_date']= pd.to_datetime(df['start_date'])
        df['end_date']= pd.to_datetime(df['end_date'])
        df_filtered= df[(df['start_date'] >= start_date) & (df['end_date'] <= end_date)].copy()
        
        if day_sum:
            df_filtered = df_filtered.groupby(df_filtered['start_date'].dt.date)['value'].sum().reset_index()
            df_filtered.rename(columns={"index": "start_date"}, inplace=True)
            df_filtered['start_date'] = pd.to_datetime(df_filtered['start_date'])
        
        df_filtered['year_month'] = df_filtered['start_date'].dt.to_period("M").astype(str)
        
        if value_threshold is not None:
            df_filtered['value'] = df_filtered['value'].clip(upper=value_threshold)
        
        return df_filtered

    # Load both datasets and combine them
    df1 = load_and_process(file1, value_threshold1)
    df1["feature"] = label1
    df2 = load_and_process(file2, value_threshold2)
    df2["feature"] = label2
    df_combined = pd.concat([df1, df2], ignore_index=True)
    
    # Create a boxplot
    plt.figure(figsize=(14, 7))
    sns.boxplot(x="year_month", y="value", hue="feature", data=df_combined, showfliers=True)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.title(f"Monthly Distribution of {label1} & {label2}")
    plt.legend(title="Feature")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def feature_month_boxplot(
    file, start_date, end_date, label, day_sum=False, value_threshold=None
):
    df = pd.read_csv(file)

    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date']= pd.to_datetime(df['end_date'])
    df_filtered= df[(df['start_date'] >= start_date) & (df['end_date'] <= end_date)].copy()
    
    if day_sum:
        # aggregate by day (summing 'value')
        df_filtered= df_filtered.groupby(df_filtered['start_date'].dt.date)['value'].sum().reset_index()
        df_filtered['start_date'] = pd.to_datetime(df_filtered['start_date'])
    # extract "Year-Month" for grouping
    df_filtered['year_month']= df_filtered['start_date'].dt.to_period("M").astype(str)  # Format: YYYY-MM
    # threshold if provided
    if value_threshold is not None:
        df_filtered['value']= df_filtered['value'].clip(upper=value_threshold)

    # create a boxplot of heart rate over time
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="year_month", y="value", data=df_filtered, showfliers=True,
                dodge=True, width=0.92, gap=0.14)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Month")
    plt.ylabel(label)
    plt.title("Monthly Distribution of " + label)
    plt.show()


def feature_value_evolving(
    file, start_date, end_date, ylabel='Steps', value_threshold=None, interactive=True
):    
    df= pd.read_csv(file)

    df['start_date']= pd.to_datetime(df['start_date'])
    df['end_date']= pd.to_datetime(df['end_date'])
    df_filtered= df[(df['start_date'] >= start_date) & (df['end_date'] <= end_date)].copy()

    # aggregate by day (summing up the 'value' column)
    df_daily = df_filtered.groupby(df_filtered['start_date'].dt.date)['value'].sum().reset_index()
    df_daily.columns = ['date', 'value']

    if value_threshold is not None:
        df_daily['value']= df_daily['value'].clip(upper=value_threshold)

    if interactive:
        # interactive area chart with Plotly
        fig = px.area(
            df_daily, x='date', y='value', 
            title=f"Total Daily {ylabel} Over Time",
            labels={'Date': 'date', f'{ylabel}': 'value'},
            hover_data={'value': True, 'date': True}
        )

        fig.update_layout(
            xaxis=dict(title="Date", tickangle=0), 
            yaxis_title=f"Total {ylabel}",
            hovermode="x unified"
        )  # Ensures the tooltip appears for all points at the same x-axis value
        fig.show()
    else:
        # plotting the area chart
        plt.figure(figsize=(10, 6))
        plt.fill_between(df_daily['date'], df_daily['value'], color="skyblue", alpha=0.4)
        plt.plot(df_daily['date'], df_daily['value'], color="Slateblue", alpha=0.6)
        plt.title(f'Total Daily {ylabel} Over Time')
        plt.xlabel('Date')
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def activ_duration_feature_evolving(df, feature_name, start_date, end_date):
    # Filter for the selected year and month
    df.index= pd.to_datetime(df.index, format="%Y-%m-%d")
    df_filtered= df[(df.index >= start_date) & (df.index <= end_date)].copy()

    # Convert heart rate from seconds to hours
    df_filtered[feature_name]= df_filtered[feature_name] / 3600  # Convert from sec to hours

    plt.figure(figsize=(12, 5))
    plt.fill_between(df_filtered.index, df_filtered[feature_name], color="blue", alpha=0.5)
    plt.plot(df_filtered.index, df_filtered[feature_name], color="blue", linewidth=1.5)

    max_value= df_filtered[feature_name].max()
    y_max= max(max_value, 200)
    
    plt.xlabel("Time")
    plt.ylabel(feature_name + " (TS Duration -- Hours)")
    plt.title(feature_name + " -- TS Duration Over Time")
    if y_max== 200:
        ticks= np.arange(0, 201, 25)
        plt.yticks(ticks)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.show()


def plot_evolution(df, start_date, end_date, top_n=10):
    # Filter for the selected year and month
    df.index= pd.to_datetime(df.index, format="%Y-%m-%d")
    df_filtered= df[(df.index >= start_date) & (df.index <= end_date)]
    # Sum up monthly durations
    df_monthly= df_filtered.resample("ME").sum()
    # Select top activities
    top_activities = df_monthly.sum().nlargest(top_n).index
    df_monthly_filtered = df_monthly[top_activities]
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_monthly_filtered.T, cmap="YlGnBu", linewidths=0.0, annot=False)
    plt.title(f"Activity Duration Over Time ({start_date} to {end_date})")
    plt.xlabel("Date")
    plt.ylabel("Activity")
    x_labels = [date.strftime('%Y-%m') if date.month == 1 else '' for date in df_monthly.index]
    plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=0)
    plt.yticks(rotation=0)
    plt.show()


def plot_agg_totals(df_counts, df_durations, start_date, end_date, top_n=10):
    # Filter for the selected year and month
    df_counts.index= pd.to_datetime(df_counts.index, format="%Y-%m-%d")
    df_counts_filtered= df_counts[(df_counts.index >= start_date) & (df_counts.index <= end_date)]

    df_durations.index= pd.to_datetime(df_durations.index, format="%Y-%m-%d")
    df_durations_filtered= df_durations[(df_durations.index >= start_date) & (df_durations.index <= end_date)]

    # Aggregate total occurrences and durations
    total_counts = df_counts_filtered.sum().sort_values(ascending=False)
    total_durations = df_durations_filtered.sum().sort_values(ascending=False)
    
    top_counts= total_counts[:top_n]
    top_durations= total_durations[:top_n] / 60  # Convert to hours
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count Plot
    axes[0].barh(top_counts.index, top_counts.values, color="skyblue")
    axes[0].set_title("Top Activities by Count")
    axes[0].set_xlabel("Number of Occurrences")
    axes[0].invert_yaxis()
    # Duration Plot
    axes[1].barh(top_durations.index, top_durations.values, color="lightcoral")
    axes[1].set_title("Top Activities by Duration (Hours)")
    axes[1].set_xlabel("Total Duration (Hours)")
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def plot_heatmap(df, start_date, end_date):
    # Filter for the selected year and month
    df.index= pd.to_datetime(df.index, format="%Y-%m-%d")
    df_filtered= df[(df.index >= start_date) & (df.index <= end_date)]

    if df_filtered.empty:
        print("No data available for the selected period.")
        return
    # Pivot for heatmap (Days as columns, Hours as rows)
    heatmap_data= df_filtered.T  # Transpose so hours are on the y-axis

    # Plot heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.0)
    plt.title(f"Data Collection Heatmap ({start_date} to {end_date})")
    plt.xlabel("Day of Month")
    plt.ylabel("Hour of Day")
    x_labels = [
        date.strftime('%Y') if date.month == 1 and date.day == 1 else '' 
        for date in df_filtered.index
    ]
    plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=45)
    plt.yticks(rotation=0)
    plt.show()


def data_sizes_by_folders(ft_file):
    df_id_sizes= pd.read_csv(ft_file)
    col_size= [
        col for col in df_id_sizes.columns if col.lower() == "folder_size_bytes"
    ][0]

    # 1. Histogram with KDE
    plt.figure(figsize=(12, 6))
    ax= sns.histplot(
        x=df_id_sizes[col_size], bins=100, kde=True, 
        color="royalblue",           # Histogram color
        edgecolor="black",           # Edge color for better separation
    )
    ax.lines[0].set_color('crimson')
    plt.title("Distribution of Folder Sizes (Bytes)")
    plt.xlabel("Folder Size (Bytes)")
    plt.ylabel("Count")
    plt.show()

    # 2. Boxplot (with log scale on x-axis for better visibility)
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df_id_sizes[col_size], color="lightgreen")
    plt.xscale("log")
    plt.title("Folder Sizes (Log Scale)")
    plt.xlabel("Folder Size (Bytes)")
    plt.show()

    # 3. Sorted Scatter Plot (each point represents a participant)
    df_sorted = df_id_sizes.sort_values(col_size)
    folder_sizes= df_sorted[col_size].values
    N= len(folder_sizes)
    # Define thresholds in bytes (1 MB = 1e6 bytes, etc.)
    thresholds= [1e6, 10e6, 100e6, 500e6]
    threshold_labels= [" 1 MB", " 10 MB", " 100 MB", " 500 MB"]
    
    # Compute cumulative percentages for each threshold
    cumulative= []
    for thresh in thresholds:
        idx= np.searchsorted(folder_sizes, thresh, side="right")
        cumulative.append(idx / N * 100)

    # Compute bin percentages (differences)
    bin_percentages= []
    bin_percentages.append(cumulative[0])  # Folders below 1MB
    for i in range(1, len(cumulative)):
        # For subsequent bins, subtract the previous cumulative percentage
        bin_percentages.append(cumulative[i] - cumulative[i - 1])
    # For the last bin: folders above the highest threshold (500 MB)
    above_pct= 100 - cumulative[-1]
    bin_percentages.append(above_pct)
    
    plt.figure(figsize=(12, 6))
    plt.plot(folder_sizes, marker='o', linestyle='', markersize=2)
    plt.yscale("log")
    plt.title("Folder Sizes per Participant (Sorted, Log Scale)")
    plt.xlabel("Participants (sorted)")
    plt.ylabel("Folder Size (Bytes)")
    plt.grid(alpha=0.5)
    ax = plt.gca()  # get the current axis

    # For annotation, compute the x-index for each threshold
    indices = {}
    for label, thresh in zip(threshold_labels, thresholds):
        idx = np.searchsorted(folder_sizes, thresh, side="right")
        indices[label] = idx

    # Annotate each bin (for bins below 500 MB)
    y_min = plt.ylim()[0]
    for label, pct in zip(threshold_labels, bin_percentages):
        idx = indices[label]
        plt.axvline(x=idx, color='black', linestyle='--', alpha=0.8)
        # Annotate near the bottom of the plot; use a small offset if needed
        plt.text(
            idx, y_min, f"{label}: {pct:.1f}%", rotation=90,
            verticalalignment='bottom', horizontalalignment='right', 
            color='darkred', fontsize=11
        )
    # Annotate the "500 MB+" percentage near the end
    offset= mtransforms.ScaledTranslation(0.2, 0, plt.gcf().dpi_scale_trans)
    plt.text(
        N, plt.ylim()[0], f" 500 MB+: {bin_percentages[-1]:.1f}%", rotation=90, 
        verticalalignment='bottom', horizontalalignment='right', 
        color='darkred', fontsize=11, 
        transform=ax.transData + offset
    )
    plt.show()


def plot_monthly_data_trends(df, start_date, end_date, remove_first_year_label=True):
    """
    Plots the evolution of data volume over time for multiple participants.
    Parameters:
    - df (pd.DataFrame): DataFrame with participants as rows and months (YYYY-MM) as columns.
    """
    plt.figure(figsize=(14, 6))

    df.columns= pd.to_datetime(df.columns, format="%Y-%m")
    # Filter for the selected date range
    df_filtered= df.loc[:, (df.columns>= start_date) & (df.columns<= end_date)]
    # Aggregate data across participants for each month
    monthly_totals= df_filtered.sum(axis=0)
    # Convert YYYY-MM to datetime
    monthly_totals.index= pd.to_datetime(monthly_totals.index, format='%Y-%m')
    
    # Plot total data volume evolution
    sns.lineplot(
        x=monthly_totals.index, y=monthly_totals.values, marker='o', 
        label="Total Data Volume"
    )
    # Formatting x-axis to show only year transitions
    years= monthly_totals.index.to_series().dt.year.drop_duplicates().values
    # Remove the first year from x-axis labels
    if remove_first_year_label and len(years)> 1: years= years[1:]
    plt.xticks(
        ticks=[monthly_totals.index[monthly_totals.index.year== y][0] for y in years],
        labels=years,
    )
    plt.xlabel("Year")
    plt.ylabel("Total Data Volume (Bytes)")
    plt.title("Total Data Volume Over Time")
    plt.grid(True)
    plt.show()


def get_bmi_stats(data_dir_path):
    # Load the dataset
    file_path = data_dir_path + "BMI_timeseries.csv"
    df = pd.read_csv(file_path)

    # Compute the median BMI for each user
    median_bmi_per_user = df.groupby("participant_id")["value"].median()

    # Categorize users into BMI classes
    bmi_categories = {
        "Healthy": (median_bmi_per_user < 25),
        "Overweight": (median_bmi_per_user.between(25, 30, inclusive="left")),
        "Obese": (median_bmi_per_user >= 30)
    }
    # Count users in each category
    bmi_distribution = {category: median_bmi_per_user[mask].count() for category, mask in bmi_categories.items()}
    # Compute percentage distribution
    total_users = median_bmi_per_user.count()
    bmi_percentage = {category: (count / total_users) * 100 for category, count in bmi_distribution.items()}

    # Print the distribution
    print("BMI Distribution:")
    total_users= 0
    total_percent= 0
    for category, count in bmi_distribution.items():
        print(f"{category}: {count} users ({bmi_percentage[category]:.1f}%)")
        total_users += count
        total_percent += bmi_percentage[category]

    print("------------------------------")
    print(f"Total: {total_users} users ({int(total_percent)}%)")
