import numpy as np
import pandas as pd
import ast
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


LOG_FILE_PREFIX= "logfile"
LOG_FILE_NAME= LOG_FILE_PREFIX + "_aggregated_files.csv"


def activities_to_average_window(ft_name):
    """
    Define the activities to average their values by the number of windows during the aggregation.
    """
    to_average= [
        'StepCount','DistanceWalkingRunning','BasalEnergyBurned','ActiveEnergyBurned','FlightsClimbed',
        'DailyDistanceCycling','BloodGlucose','DailyDistanceSwimming','EnergyConsumed','Carbohydrates',
        'Protein','TotalFat','BloodAlcoholContent',
    ]
    if ft_name in to_average:
        return True
    return False


def features_to_aggregate(df):
    """
    Function to determine the column values to aggregate according to the feature under processing
    """
    if "value" in df.columns:
        values = ["value"]
    elif (
        "blood_pressure_systolic_value" in df.columns and "blood_pressure_diastolic_value" in df.columns
    ):
        # we are handling BloodPressure
        values = ["blood_pressure_systolic_value", "blood_pressure_diastolic_value"]
    elif "average_heart_rate" in df.columns:
        # we are handling ECG/Electrocardiogram
        values = ["average_heart_rate", "sampling_frequency", "voltage_measurements"]
    else:
        # we are handling ActivitySummary
        values = [
            "apple_stand_hours", "apple_exercise_time", "active_energy_burned", "apple_stand_hours_goal",
            "apple_exercise_time_goal", "active_energy_burned_goal",
        ]

    return values


def aggregate_values_by_type(series, agg_type='median'):
    """
    Function to handle both numeric and categorical values during aggregation
    """
    if pd.api.types.is_numeric_dtype(series):
        if (agg_type=='median'):
            return series.median()  # Median for numerical data
        else:
            return series.mean()    # Mean for numerical data

    if len(series) > 1 and isinstance(series.iloc[0], list) and all(
        isinstance(item, list) for item in series.iloc[0]
    ):
        # here we have ECG voltage_measurements
        max_len = max(len(p) for p in series)
        pw_list = []

        for i in range(max_len):  # combine the lists by index
            value_pairs = [row[i] for row in series if i < len(row)]  # sum each corresponding pair

            if value_pairs:  # ensure there are values to compute median
                if (agg_type=='median'):
                    pw_list.append(np.median(value_pairs, axis=0).tolist())
                else:
                    pw_list.append(np.mean(value_pairs, axis=0).tolist())
        
        return pw_list

    # Mode for categorical
    mode_values = series.mode()
    return mode_values.iloc[0] if not mode_values.empty else series.iloc[0]


def anonymize_and_remove_duplicates(ft_name, df):
    """
    This function anonymizes and cleans the input DataFrame by:
    1. Dropping specific columns ('id', 'metadata', 'source_id', 'source_name') that are deemed sensitive
    2. Removing duplicate rows
    3. Resetting the index of the DataFrame after cleaning
    Parameters:
    - ft_name (str): label of the feature under processing
    - df (pandas.DataFrame): The input DataFrame to be cleaned
    Returns:
    - pandas.DataFrame: The cleaned DataFrame
    """
    if ft_name == "ActivitySummary":
        df["datetime"] = pd.to_datetime(
            df["datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce", utc=True
        )
        value_features = features_to_aggregate(df)

        # Group by 'datetime', then take mean values
        df = df.groupby(["datetime"], as_index=False).agg({
            **{col: "mean" for col in value_features},
            # Keep other columns
            **{col: "first" for col in df.columns
               if col not in ["datetime", *value_features]},
        })
        df = df.sort_values(by=["datetime"]).reset_index(drop=True)
    else:
        # Drop sensitive/useless columns
        df = df.drop(
            columns=["id", "metadata", "source_id", "source_name"], errors="ignore"
        )

        # Convert start_date and end_date to datetime
        df["start_date"] = pd.to_datetime(
            df["start_date"], format="%Y-%m-%dT%H:%M:%S.%f%z", errors="coerce", utc=True
        )
        df["end_date"] = pd.to_datetime(
            df["end_date"], format="%Y-%m-%dT%H:%M:%S.%f%z", errors="coerce", utc=True
        )

        value_features = features_to_aggregate(df)

        # Group by 'start_date' and 'end_date', keeping the first occurrence of other columns
        df = df.groupby(["start_date", "end_date"], as_index=False).agg({
            # Compute the mean/mode of 'value' for duplicate timestamps
            **{col: aggregate_values_by_type for col in value_features},
            # Keep other columns
            **{col: "first" for col in df.columns
               if col not in ["start_date", "end_date", *value_features]},
        })
        df = df.sort_values(by=["start_date", "end_date"]).reset_index(drop=True)

    return df


def clean_up_duplicated(feature, file_path):
    """
    Reads a CSV file and removes duplicated rows based on specific columns,
    depending on the feature type. If duplicates are found, the data is anonymized and
    duplicates removed before saving the updated CSV.
    Parameters:
    - feature (str): Name of the feature.
    - file_path (str): Path to the CSV file.
    Returns:
    - pd.DataFrame: Cleaned DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        
        # checking for duplicates
        if feature == "ActivitySummary":
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
            duplicated = df[["datetime"]].duplicated().sum() > 0
        else:
            df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
            df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce", utc=True)
            duplicated = df[["start_date", "end_date"]].duplicated().sum() > 0

        if duplicated:
            # anonymize and remove duplicates (if any present)
            df = anonymize_and_remove_duplicates(feature, df)

            try:
                df.to_csv(file_path, index=False)
                print(f"Updating data for '{feature}' to {file_path}")
            except Exception as e:
                print(f"Error saving {file_path}: {e}")

    except Exception as e:
        print(f"clean_up_duplicated -- Error reading {file_path}: {e}")
        df= pd.DataFrame()

    return df


def create_event_windows(start, end, window="5min"):
    """
    Distributes an event's duration, count, and additional attributes evenly across fixed time windows.
    This function assumes that each row (event) in the input DataFrame has at least the following columns:
    - 'start_date': The datetime when the event begins.
    - 'end_date': The datetime when the event ends.
    It creates fixed windows (default: 5-minute windows) covering the period from the floored start_date to the
    ceiled end_date. The event's total duration (in seconds) is divided equally among these windows, and an event
    count of 1 is assigned to each window. In addition, all other attributes in the row  (except 'start_date'
    and 'end_date') are replicated into each window's row.
    Parameters
    - row (pd.Series): A row from the input DataFrame representing an event. Must include 'start_date' and
    'end_date' as datetime objects.
    - window (str, optional): A string representing the resampling frequency (e.g., '5min' for five minutes).
    Default is '5min'.
    Returns
    pd.DataFrame or None (if no windows are generated)
    - A DataFrame with the following columns:
        - 'start_window': The start of the fixed time window.
        - 'end_window': The end of the fixed time window.
        - 'duration': The allocated duration (in seconds) of the event for that window.
        - 'event_count': The event count (set to 1 for each window).
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    window_start = start.floor(window)
    window_end = end.ceil(window)

    if window_start == window_end:  # Ensure at least one valid window
        window_end = window_start + pd.Timedelta(window)

    windows = pd.date_range(start=window_start, end=window_end, freq=window, tz=start.tz)
    n_windows = len(windows) - 1

    if n_windows == 0:
        return None

    total_duration = (end - start).total_seconds()
    duration_per_window = total_duration / max(n_windows, 1)

    return windows[:-1], windows[1:], duration_per_window, n_windows


def distribute_events(df, ft_name, window="5min"):
    """
    Distributes all events in a DataFrame into fixed time windows, preserving all additional attributes.
    Designed to build time windows and solve the issues with data overlappings.
    Returns
    - pd.DataFrame with one row per fixed time window that an event covers. The columns include:
        - 'start_window': The start of the fixed time window.
        - 'end_window': The end of the fixed time window.
        - 'value/value_feature: The mean(numeric feature)/mode(categorical feature) of the agg values
        - Additional attributes (replicated from the original event row).
    """
    results = []
    values = []

    start_dates = df["start_date"].values
    end_dates = df["end_date"].values
    additional_cols = {
        col: df[col].values
        for col in df.columns
        if col not in ["start_date", "end_date"]
    }

    for i in range(len(df)):
        start, end = start_dates[i], end_dates[i]
        result = create_event_windows(start, end, window)
        if result is not None:
            start_windows, end_windows, duration, event_count = result

            for j in range(len(start_windows)):
                results.append((start_windows[j], end_windows[j], duration, event_count))
                values.append({col: additional_cols[col][i] for col in additional_cols})

    if not results:
        return pd.DataFrame()

    df_distributed = pd.DataFrame(results, columns=["start_window", "end_window", "duration", "event_count"])
    df_values = pd.DataFrame(values)
    df_distributed = pd.concat([df_distributed, df_values], axis=1)

    value_features = features_to_aggregate(df)
    
    if activities_to_average_window(ft_name) is True:
        for col in value_features:
            df_distributed[col]= (df_distributed[col] / df_distributed["event_count"])

    df_result = (df_distributed.groupby(["start_window", "end_window"]).agg({
        # Compute the mean/mode of 'value' for duplicate time windows
        **{col: aggregate_values_by_type for col in value_features},
        # Keep other columns
        **{col: "first" for col in df_distributed.columns
            if col not in [
                "start_window", "end_window", *value_features, "duration", "event_count",
            ]},
        # 'duration': 'sum',
        # 'event_count': 'sum',
    }).reset_index())

    df_result.rename(columns={"start_window": "start_date", "end_window": "end_date"}, inplace=True)
    df_result = df_result.sort_values(by=["start_date"]).reset_index(drop=True)

    return df_result


def resume_from_log_files(csv_files, input_dir, output_dir):
    """
    Here we identify the log file with previously processed files, read it, and skip 
    processed files
    - csv_files (list of str): Paths of all CSV files in the input_dir
    - input_dir (str): Path to the directory containing CSV files.
    - output_dir (str): Path to the parent directory where aggregated data will be saved.
    Returns:
    - csv_files (list of str): Updated csv_files list only with paths to not processed files
    - processed_files (list of str): List of processed files from a log file in output_dir
    """
    # Hold the name of processed files to track the updated files 
    processed_files= []

    try:
        id_number= csv_files[0].split(os.sep)[-2]
        log_path= os.path.join(output_dir, id_number, LOG_FILE_NAME)
        log_processed= pd.read_csv(log_path, header=None, dtype=str)
        log_processed= log_processed.iloc[:, 0].dropna().tolist()
        filtered_files= []

        for file_path in csv_files:
            # get the relative file name (without directory and extension)
            file_name= os.path.splitext(os.path.relpath(file_path, input_dir))[0]

            if file_name in log_processed:
                # track processed files and skip those already processed.
                processed_files.append(file_name)
                print(f"Skiping file {file_name}: Previously processed.")
            else:
                # keep only non-processed elements
                filtered_files.append(file_path)
        
        csv_files= filtered_files  # update csv_files to only include new files
    except Exception as e:
        print(f"Error reading processed logfile: {e}.\nThis directory will be processed from scratch.")

    return csv_files, processed_files


def aggregate_apple_healthkit_data(directory_path, output_parent_dir, resume_from_log=True):
    """
    Aggregates data from all CSV files in the specified directory where data_source is AppleHealthkit.
    Parameters:
    - directory_path (str): Path to the directory containing CSV files.
    - output_parent_dir (str): Path to the parent directory where aggregated data will be saved.
    - resume_from_log (bool): If true, inspect each ID folder in output_parent_dir for a log file holding the  
    list of processed files in input_parent_dir to skip during the processing; otherwise, process each ID folder  
    in input_parent_dir from scratch
    Returns:
    - aggregated_dfs (dict): Dictionary where keys are unique 'name' values and values are aggregated DataFrames.
    """
    # Hold the name of processed files to track the updated files 
    processed_files= []
    # Initialize the dictionary to hold aggregated DataFrames
    aggregated_dfs = {}
    # Use glob to find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    # Skip possible log files
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith(LOG_FILE_PREFIX)]

    if not csv_files:
        print(f"No CSV files found in directory: {directory_path}")
        return aggregated_dfs, processed_files

    print(f"Found {len(csv_files)} CSV files in directory: {directory_path}")

    if resume_from_log:
        csv_files, processed_files= resume_from_log_files(
            csv_files, directory_path, output_parent_dir
        )
    print(f"--> {len(csv_files)} CSV files will be processed.")

    # iterate over each CSV file with a progress bar
    for file_path in tqdm(csv_files, desc="Processing CSV files"):
        file_name= os.path.splitext(os.path.relpath(file_path, directory_path))[0]

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue  # Skip this file if there's an error

        # Track processed files
        processed_files.append(file_name)

        # Ensure 'data_source' column exists and filter correctly
        if "data_source" in df.columns:
            # Filter rows where data_source is AppleHealthkit
            apple_health_df = df[df["data_source"].str.strip().str.lower() == "applehealthkit"]
        else:
            print(f"'data_source' column not found in {file_path}. Skipping.")
            continue

        if apple_health_df.empty:
            print(f"No AppleHealthkit data in {file_path}. Skipping.")
            continue  # Skip this file if no relevant data

        # Get all unique names in this subset
        unique_names = apple_health_df["name"].unique()

        for name in unique_names:
            if name == "Mindful":
                print(f"{name} is a feature with no processing values. Skipping.")
                continue

            # Filter the DataFrame for the current 'name'
            subset = apple_health_df[apple_health_df["name"] == name]
            # List to hold parsed DataFrames for the current 'name'
            list_of_dfs = []

            for index, row in subset.iterrows():
                data_str = row.get("data", "")[1:-1]
                if not isinstance(data_str, str):
                    print(f"Invalid data format in file {file_path} at index {index}. Skipping row.")
                    continue

                # Attempt to parse the data string
                try:
                    parsed_data = ast.literal_eval(data_str)
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing data in file {file_path} at index {index}: {e}")
                    continue  # Skip this row if parsing fails

                # Ensure the parsed data is a list of dictionaries
                if isinstance(parsed_data, list) and all(
                    isinstance(item, dict) for item in parsed_data
                ):
                    df_data = pd.DataFrame(parsed_data)
                # if parsed_data is a dictionary, convert it to a DataFrame
                elif isinstance(parsed_data, dict):
                    df_data = pd.DataFrame([parsed_data])
                else:
                    print(f"Data in file {file_path} at index {index} is not a list of dictionaries.")
                    continue  # Skip this row if data format is invalid

                # Optionally, add contextual columns from the main DataFrame
                # Convert date strings to datetime objects
                for date_col in ["datetime", "created_at", "updated_at"]:
                    if date_col in row and pd.notnull(row[date_col]):
                        try:
                            df_data[date_col] = pd.to_datetime(row[date_col])
                        except Exception as e:
                            print(f"Error converting '{date_col}' in file {file_path} at index {index}: {e}")
                            df_data[date_col] = (pd.NaT)  # Assign Not-a-Time if conversion fails

                # Add other contextual columns
                df_data["participant_id"] = row.get("participant_id", None)
                df_data["data_source"] = row.get("data_source", None)

                try:
                    df_data = anonymize_and_remove_duplicates(name, df_data)
                except Exception as e:
                    print(f"Error removing duplicates of '{name}' in file {file_path}: {e}")
                    continue  # Skip this row if anonymizing fails

                list_of_dfs.append(df_data)

            if list_of_dfs:
                # Concatenate all DataFrames for the current 'name'
                combined_df = pd.concat(list_of_dfs, ignore_index=True)
                # If the 'name' already exists in aggregated_dfs, append to it
                if name in aggregated_dfs:
                    aggregated_dfs[name] = pd.concat([aggregated_dfs[name], combined_df], ignore_index=True)
                else:
                    aggregated_dfs[name] = combined_df
            else:
                print(f"No valid data found for name: {name} in file {file_path}")

    print("Aggregation complete.")
    return aggregated_dfs, processed_files


def process_one_id_dir(
    id_dir, input_parent_dir, output_parent_dir, save_aggregated_dfs, window, resume_from_log
):
    """
    Process all CSV files in a given ID folder.
    Parameters:
    - id_dir (str): Path to the ID directory under processing.
    - input_parent_dir (str): Path to the parent directory containing ID subdirectories.
    - output_parent_dir (str): Path to the parent directory where aggregated data will be saved.
    - save_aggregated_dfs (bool): Whether to save the aggregated DataFrames to CSV files.
    - window (str, optional): A string representing the resampling frequency (e.g., '5min' for five minutes).
    Default is '5min'. If None, no time windowing on the processed data.
    - resume_from_log (bool): If true, inspect each ID folder in output_parent_dir for a log file holding the  
    list of processed files in input_parent_dir to skip during the processing; otherwise, process each ID folder  
    in input_parent_dir from scratch
    """
    participant_count= 0

    id_dir_path = os.path.join(input_parent_dir, id_dir)
    print(f"\nProcessing ID: {id_dir}")

    # Call the aggregation function
    aggregated_data, processed_files= aggregate_apple_healthkit_data(
        id_dir_path, output_parent_dir, resume_from_log
    )

    if save_aggregated_dfs and aggregated_data:
        participant_count= 1
        # Define the output directory for this ID
        output_dir = os.path.join(output_parent_dir, id_dir)
        os.makedirs(output_dir, exist_ok=True)

        for name, aggregated_df in aggregated_data.items():
            # Create a valid filename by replacing spaces and other unwanted characters
            safe_name = "".join([c if c.isalnum() or c in ("_", "-") else "_" for c in name])
            filename = f"{safe_name}.csv"
            file_path = os.path.join(output_dir, filename)

            if window and name != "ActivitySummary":
                print(f"Creating {window} windows for '{name}'")
                aggregated_df = distribute_events(aggregated_df, name, window=window)

            try:
                if os.path.exists(file_path):
                    # append new data to the existing file (without header)
                    aggregated_df.to_csv(file_path, mode='a', header=False, index=False)
                    # final check for duplicates after appending into the file due to events
                    # in the last day of the previous month that can last after midnight
                    aggregated_df = clean_up_duplicated(name, file_path)
                    print(f"Appended aggregated data for '{name}' to {file_path}")
                else:
                    # file doesn't exist: create a new file with headers
                    aggregated_df.to_csv(file_path, index=False)
                    print(f"Saved aggregated data for '{name}' to {file_path}")
            except Exception as e:
                print(f"Error saving {file_path}: {e}")

        try:
            processed_files= pd.DataFrame(
                processed_files, columns=["filename"]
            ).sort_values(by=["filename"]).reset_index(drop=True)

            processed_files.to_csv(os.path.join(output_dir, LOG_FILE_NAME), index=False, header=True)
            print(f"Creating processed log file to {id_dir}")
        except Exception as e:
            print(f"Error saving processed logfile: {e}")

    return participant_count


def process_all_ids(
    input_parent_dir, output_parent_dir, select_dir=None, save_aggregated_dfs=True,
    window="5min", skip_dirs_in_output_parent_dir=False, resume_from_log=True, max_workers=4
):
    """
    Processes all ID directories within the input_parent_dir and aggregates their AppleHealthkit data.
    Parameters:
    - input_parent_dir (str): Path to the parent directory containing ID subdirectories.
    - output_parent_dir (str): Path to the parent directory where aggregated data will be saved.
    - select_dir (str): Path to a directory with known subdirectories to select and process in the input_parent_dir
    - save_aggregated_dfs (bool): Whether to save the aggregated DataFrames to CSV files.
    - window (str, optional): A string representing the resampling frequency (e.g., '5min' for five minutes).
    Default is '5min'. If None, no time windowing on the processed data.
    - skip_dirs_in_output_parent_dir (bool): If True, skip the dirs present in output_parent_dir during the
    processing of dirs in input_parent_dir; otherwise, process input_parent_dir from scratch
    - resume_from_log (bool): If true, inspect each ID folder in output_parent_dir for a log file holding the  
    list of processed files in input_parent_dir to skip during the processing; otherwise, process each ID folder  
    in input_parent_dir from scratch
    """
    # Counter for participants with saved data
    participant_count = 0
    # Ensure the output parent directory exists
    os.makedirs(output_parent_dir, exist_ok=True)
    # List all subdirectories in the output_parent_dir
    out_id_dirs = [
        d for d in os.listdir(output_parent_dir) 
        if os.path.isdir(os.path.join(output_parent_dir, d))
    ]
    if select_dir is None:
        # List all subdirectories in the input_parent_dir
        id_dirs = [
            d for d in os.listdir(input_parent_dir) 
            if os.path.isdir(os.path.join(input_parent_dir, d))
        ]
    else:
        # List all subdirectories in the select_dir to process only the same in the input_parent_dir
        id_dirs = [
            d for d in os.listdir(select_dir) 
            if os.path.isdir(os.path.join(select_dir, d))
        ]
    print(f"Found a total of {len(id_dirs)} ID directories.")

    # This step acts like a 'resume' in processing by skipping dirs already processed
    if out_id_dirs and skip_dirs_in_output_parent_dir:
        id_dirs = [item for item in id_dirs if item not in out_id_dirs]
        print(f"Skipping {len(out_id_dirs)} ID directories previously processed.")

    if not id_dirs:
        print(f"No subdirectories found in input parent directory: {input_parent_dir}")
        return
    print(f"--> {len(id_dirs)} ID directories will be processed.")


    # Processes all ID folders in parallel using a ProcessPoolExecutor.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(  # task with multiple parameters
                process_one_id_dir, id_dir, input_parent_dir, output_parent_dir, save_aggregated_dfs, 
                window, resume_from_log
            ): id_dir for id_dir in id_dirs
        }
        # as_completed iterator with tqdm for progress monitoring.
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing ID directories"):
            participant_count += future.result()


    print(f"Aggregated participants: {participant_count} out {len(id_dirs)} ID directories processed.")
    print("All ID directories have been processed.")


if __name__ == "__main__":
    current_folder = os.getcwd()
    # Specify the parent input directory containing ID subdirectories
    input_parent_directory = current_folder + "/exported-at-2024-12/"

    # Specify the parent directory containing ID subdirectories to select and process in the input_parent_directory
    input_selected = None  # current_folder + "/EV_aggregated_from_apple_24_12_w05min/"

    # Specify the parent output directory where aggregated data will be saved
    output_parent_directory = current_folder + "/EV_aggregated_from_apple_24_12_w05min_update/"

    # Process all ID directories
    process_all_ids(
        input_parent_directory, output_parent_directory, input_selected, save_aggregated_dfs=True,
        window="5min", skip_dirs_in_output_parent_dir=False, resume_from_log=True, max_workers=24
    )
