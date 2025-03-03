import pandas as pd
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from parse_v4 import clean_up_duplicated
from tqdm import tqdm


LOG_FILE_PREFIX= "logfile"


def accumulate_feature_time(feature, df):
    """
    Accumulates time spent on a given feature and counts the number of collections per day.
    Parameters:
    - feature (str): The name of the feature being analyzed.
    - df (pd.DataFrame): A DataFrame containing the following columns:
        - 'start_date' (datetime): Start time of the feature occurrence.
        - 'end_date' (datetime): End time of the feature occurrence.
        - 'datetime' (datetime): Timestamp indicating the date of collection.
    Returns:
    - daily_duration (pd.DataFrame): A DataFrame with total accumulated duration (in seconds) per day.
    - daily_counts (pd.DataFrame): A DataFrame with the count of feature occurrences per day.
    - hourly_counts (pd.DataFrame): A DataFrame with the count of feature occurrences per day.
    """
    # Reset index to maintain order -- assume 'start_date' and 'end_date' previously converted to datetime
    df = df.sort_values(by=["start_date", "end_date"]).reset_index(drop=True)
    # Extract date and hour from 'start_date'
    df["date"] = df["start_date"].dt.date
    df["hour"] = df["start_date"].dt.hour
    # Count collections per hour per day
    hourly_counts = (
        df.groupby(["date", "hour"]).size().reset_index(name=f"{feature}_hourly")
    )
    # Pivot to have 'date' as rows and hours (0-23) as columns.
    hourly_counts = hourly_counts.pivot(
        index="date", columns="hour", values=f"{feature}_hourly"
    ).fillna(0)
    # Ensure all 24 hours (0-23) exist as columns, filling missing ones with 0
    hourly_counts = hourly_counts.reindex(columns=range(24), fill_value=0)
    # Reorder columns to ensure 0-23 order
    hourly_counts = hourly_counts.reset_index().rename_axis(None, axis=1)

    # Extract only the date
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.date
    # Count collections per day
    daily_counts = df.groupby("datetime").size().reset_index()
    daily_counts.columns = ["date", feature]

    # Calculate duration in seconds
    df["duration"] = (df["end_date"] - df["start_date"]).dt.total_seconds()
    # Aggregate total seconds per day
    daily_duration = df.groupby("datetime")["duration"].sum().reset_index()
    daily_duration.columns = ["date", feature]

    return daily_duration, daily_counts, hourly_counts


def list_of_dfs_to_df(list_dfs):
    """
    Merges a list of DataFrames on 'date', resulting in one DataFrame with columns:
    [date, ft_df0, ft_df1, ... ft_dfN]. Fills missing values with 0 and sorts by date.
    Parameters:
    - list_dfs (list of pd.DataFrame): List of DataFrames, each with a 'date' column.
    Returns:
    - pd.DataFrame: Merged DataFrame with date as index and numeric columns as integers.
    """
    if not list_dfs:
        return pd.DataFrame()
    df_final = list_dfs[0]

    for df in list_dfs[1:]:
        df_final = pd.merge(df_final, df, on="date", how="outer")

    df_final = df_final.sort_values(by="date").fillna(0).set_index("date")
    # ensure numeric columns are cast to int.
    return df_final.astype(
        {col: "int" for col in df_final.select_dtypes(include=["number"]).columns}
    )


def merge_hourly_dataframes(list_dfs):
    """
    Merges a list of hourly DataFrames on 'date' by summing overlapping values.
    Parameters:
    - list_dfs (list of pd.DataFrame): List of DataFrames with a 'date' column.
    Returns:
    - pd.DataFrame: Merged DataFrame with summed values.
    """
    if not list_dfs:
        return pd.DataFrame()

    # Concatenate all dataframes along rows
    df_final = pd.concat(list_dfs, axis=0)
    # Sum values for the same date
    df_final = df_final.groupby("date", as_index=False).sum()

    return df_final.astype(
        {col: "int" for col in df_final.select_dtypes(include=["number"]).columns}
    )


def merge_dfs_and_sum_features(list_dfs):
    """
    Merges a list of DataFrames on 'date' and sums values for overlapping features.
    Args:
    - list_dfs (list of pd.DataFrame): List of DataFrames with a 'date' column.
    Returns:
    - pd.DataFrame: Merged DataFrame with dates as rows and summed feature values.
    """
    if not list_dfs:
        return pd.DataFrame()

    # ensure each DataFrame has 'date' as a column.
    for i, df in enumerate(list_dfs):
        if 'date' not in df.columns:
            list_dfs[i] = df.reset_index()

    df_all = pd.concat(list_dfs, ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])

    # Identify numeric columns (excluding 'date')
    numeric_cols = df_all.select_dtypes(include=["number"]).columns.tolist()
    if 'date' in numeric_cols:
        numeric_cols.remove('date')

    # Group by the 'date' column
    df_final = df_all.groupby("date", as_index=False)[numeric_cols].sum()

    df_final[numeric_cols] = df_final[numeric_cols].fillna(0).astype(int)
    df_final = df_final.sort_values(by="date").reset_index(drop=True)

    return df_final


def get_id_statistics(input_parent_dir, id_dir, fts_to_skip_time_accum):
    """
    Processes a directory of CSV files corresponding to a single ID.
    Parameters:
    - input_parent_dir (str): Path to the parent directory with ID subdirectories.
    - id_dir (str): The ID directory under processing.
    - fts_to_skip_time_accum (list): Features to skip for time accumulation.
    Returns:
    - id_dir (str): The ID directory under processing.
    - feature_presence (dict): Dictionary mapping features to their occurrence count.
    - dir_size (int): Total size of files in the directory (in bytes).
    - ft_evolution (pd.DataFrame): DataFrame of feature evolution (accumulated durations).
    - ft_daily_counts (pd.DataFrame): DataFrame of daily feature counts.
    - ft_hourly_counts (pd.DataFrame): DataFrame of hourly feature counts.
    - active_dates: a set of dates (as datetime.date objects) on which at least one 
    activity is recorded.
    """
    dir_size = 0
    feature_presence = {}
    # Lists to hold event counts and durations for each day
    ft_duration_dfs = []
    ft_daily_counts_dfs = []
    ft_daily_hourly_dfs = []
    active_dates = set()

    print(f"\nProcessing ID: {id_dir}")
    # Access the ID directory under processing.
    directory_path= os.path.join(input_parent_dir, id_dir)
    # Use glob to find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    for file in csv_files:
        if os.path.isfile(file):  # Ensure it's a file
            dir_size += os.path.getsize(file)
            feature_name = file.replace(directory_path + os.sep, "").replace(".csv", "")

            if LOG_FILE_PREFIX in feature_name.split("_"):
                print("Skipping log file.")
                continue

            # track feature presence
            feature_presence[feature_name] = feature_presence.get(feature_name, 0) + 1
            # certify that there are no duplicates
            df = clean_up_duplicated(feature_name, file)
            # collect the dates on which at least one activity is recorded
            if 'datetime' in df.columns:
                dates = pd.to_datetime(
                    df['datetime'], errors='coerce', utc=True
                ).dt.date.dropna().unique()
                active_dates.update(dates)
            else:
                print(f"'datetime' field not found in {feature_name}")
            
            # fts_to_skip_time_accum features are not proper time varying data
            if feature_name not in fts_to_skip_time_accum:
                ft_duration, ft_daily_counts, ft_hourly_counts = (
                    accumulate_feature_time(feature_name, df)
                )

                ft_duration_dfs.append(ft_duration)
                ft_daily_counts_dfs.append(ft_daily_counts)
                ft_daily_hourly_dfs.append(ft_hourly_counts)
            else:
                print(f"Skip {feature_name} in time accumulation -- Not a time series.")
                continue

    ft_evolution = list_of_dfs_to_df(ft_duration_dfs)
    ft_daily_counts = list_of_dfs_to_df(ft_daily_counts_dfs)
    ft_hourly_counts = merge_hourly_dataframes(ft_daily_hourly_dfs)

    return id_dir, feature_presence, dir_size, ft_evolution, ft_daily_counts, ft_hourly_counts, active_dates


def process_all_ids(input_parent_dir, output_parent_dir, fts_to_skip_time_accum, max_workers=4):
    """
    Processes all ID directories within the input_parent_dir, aggregates their data,
    and saves summary CSV files in the output_parent_dir.
    Parameters:
    - input_parent_dir (str): Path to the parent directory with ID subdirectories.
    - output_parent_dir (str): Path to save the output CSV files.
    - fts_to_skip_time_accum (list): List of feature names to skip time accumulation.
    """
    # Check the number of CPUs/cores available
    max_workers= max(1, os.cpu_count() - 1) if max_workers> max(1, os.cpu_count() - 1) else max_workers

    # Aggregates daily and hourly data
    all_ids_daily_counts = []
    all_ids_hourly_counts = []
    # Aggregates activity evolution (count and duration)
    all_ids_ft_evol = []
    # Store information per ID folder
    participant_sizes = []
    participant_features = {}
    # Store the count of active users per day
    active_counts = {}
    aggregated_features = {}

    # Ensure the output parent directory exists
    os.makedirs(output_parent_dir, exist_ok=True)
    # List all subdirectories in the input_parent_dir
    id_dirs = [
        d for d in os.listdir(input_parent_dir)
        if os.path.isdir(os.path.join(input_parent_dir, d))
    ]
    if not id_dirs:
        print(f"No subdirectories found in input parent directory: {input_parent_dir}")
        return
    print(f"Found {len(id_dirs)} ID directories to process.")

    # Processes all ID folders in parallel using a ProcessPoolExecutor.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(  # Call the statistics function
                get_id_statistics, input_parent_dir, id_dir, fts_to_skip_time_accum
            ): id_dir for id_dir in id_dirs
        }
        # as_completed iterator with tqdm for progress monitoring.
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing ID directories"):
            (
                id_dir, dir_features, dir_size, dir_ft_evol, dir_ft_daily_count, dir_ft_hourly_counts, 
                user_active_dates
            ) = future.result()
            # The output directory for this ID
            output_dir = os.path.join(output_parent_dir, id_dir)
            os.makedirs(output_dir, exist_ok=True)

            all_ids_ft_evol.append(dir_ft_evol)
            all_ids_daily_counts.append(dir_ft_daily_count)
            all_ids_hourly_counts.append(dir_ft_hourly_counts)

            for d in user_active_dates:
                active_counts[d] = active_counts.get(d, 0) + 1

            # Store size results per ID folder
            participant_sizes.append({"participant_id": id_dir, "folder_size_bytes": dir_size})
            # Features present in each ID folder
            participant_features[id_dir] = dir_features

            # Aggregate feature counts across directories
            for feature, count in dir_features.items():
                aggregated_features[feature] = aggregated_features.get(feature, 0) + count

            try:
                dir_ft_evol.to_csv(os.path.join(output_dir, LOG_FILE_PREFIX + "_activity_durations.csv"))
                dir_ft_daily_count.to_csv(os.path.join(output_dir, LOG_FILE_PREFIX + "_activity_counts.csv"))
                print("Saving per-participant log files.")
            except Exception as e:
                print(f"Error saving log file for ID {id_dir}: {e}")

    try:
        print("Saving processing log files...")
        # Convert aggregated sizes to a DataFrame and save
        df_sizes = pd.DataFrame(participant_sizes)
        # Sort by folder size
        df_sizes = df_sizes.sort_values(by="folder_size_bytes", ascending=False)
        df_sizes.to_csv(
            os.path.join(output_parent_dir, LOG_FILE_PREFIX + "_id_folder_sizes.csv"), index=False,
        )
        # print(f"\nFolders sizes:")
        # print(df_sizes)

        # Create a DataFrame summarizing feature presence across IDs
        df_features = pd.DataFrame(list(aggregated_features.items()), columns=["feature", "count"])
        # Sort features by count
        df_features = df_features.sort_values(by="count", ascending=False).reset_index(drop=True)
        df_features.to_csv(
            os.path.join(output_parent_dir, LOG_FILE_PREFIX + "_features_summary.csv"), index=False,
        )
        # print(f"\nFeature Presence Across {total_participants} Participants:")
        # print(df_features)

        # Build list of presence of features for each ID
        id_features = []
        for participant, features in participant_features.items():
            row = {"participant_id": participant}
            row.update({feature: features.get(feature, 0) for feature in df_features["feature"]})
            id_features.append(row)

        df_presence = pd.DataFrame(id_features)
        df_presence.to_csv(
            os.path.join(output_parent_dir, LOG_FILE_PREFIX + "_features_by_id_folder.csv"), index=False
        )
        # print(f"\nFeature Presence For Each Participant:")
        # print(df_presence)

        # Merge hourly and daily DataFrames and save.
        df_hourly_counts = (merge_hourly_dataframes(all_ids_hourly_counts)).set_index("date")
        df_hourly_counts.to_csv(os.path.join(output_parent_dir, LOG_FILE_PREFIX + "_all_hourly_data.csv"))

        df_daily_counts = merge_dfs_and_sum_features(all_ids_daily_counts).set_index('date')
        df_daily_counts.to_csv(os.path.join(output_parent_dir, LOG_FILE_PREFIX + "_activity_counts.csv"))
        df_durations = merge_dfs_and_sum_features(all_ids_ft_evol).set_index('date')
        df_durations.to_csv(os.path.join(output_parent_dir, LOG_FILE_PREFIX + "_activity_durations.csv"))

        # Convert active_counts to a DataFrame and sort by date
        df_active = pd.DataFrame(list(active_counts.items()), columns=["date", "active_devices"])
        df_active["date"] = pd.to_datetime(df_active["date"])
        df_active = df_active.sort_values("date")
        df_active.to_csv(os.path.join(output_parent_dir, LOG_FILE_PREFIX + "_active_devices.csv"), index=False)

        total_per_activity = df_durations.sum()
        overall_total = (total_per_activity.sum() / 3600).round(0).astype(int)

        print("\nTotal duration of data (in hours):", overall_total)
        print("All ID directories have been processed.")
    except Exception as e:
        print(f"Error saving log file: {e}")


if __name__ == "__main__":
    current_folder = os.getcwd()
    # Specify the parent input directory containing ID subdirectories
    input_parent_directory = current_folder + "/EV_aggregated_from_apple_24_12_w05min_update/"

    # Specify the parent output directory where statistics data will be saved
    output_parent_directory = input_parent_directory

    fts_to_skip_time_accum = ["ActivitySummary", "Height", "Weight"]

    # Process all ID directories
    process_all_ids(
        input_parent_directory, output_parent_directory, fts_to_skip_time_accum, max_workers=4
    )
