import os
import pandas as pd
from tqdm import tqdm


def collect_time_series(directory_path, activity, min_duration, tolerance):
    """
    Processes one ID folder for a specific activity.
    Reads the activity CSV (assumed to be named as "<activity>.csv"), sorts the data chronologically, 
    and splits it into continuous segments. A continuous segment is defined as a sequence of rows 
    where the gap between the current row's start_date and the previous row's end_date is less than 
    or equal to the given tolerance.
    Only segments with a total duration at least equal to min_duration (e.g., '60min') are returned.

    Returns:
    - valid_df (pd.DataFrame): DataFrame containing only the rows that belong to continuous segments
    meeting the duration requirement. If no valid segment is found, returns None.
    """
    # Construct the file path for the activity CSV
    csv_file = os.path.join(directory_path, f"{activity}.csv")

    if not os.path.exists(csv_file):
        print(f"No {activity} file found in directory: {directory_path}")
        return None
    
    try:
        # Assume it has 'start_date' and 'end_date' columns
        # The data was processed with a fixed window size (in minutes)
        df = pd.read_csv(csv_file, parse_dates=['start_date', 'end_date'])
        df = df.sort_values("start_date").reset_index(drop=True)
        # Compute the previous record's end time (shift down by 1)
        df['prev_end'] = df['end_date'].shift(1)
        df.loc[df.index[0], 'prev_end'] = df.loc[df.index[0], 'start_date']
        # Compute the gap between the current start and previous end
        df['gap'] = df['start_date'] - df['prev_end']

        # If the gap is less than or equal to tolerance, we consider it continuous
        tolerance = pd.Timedelta(tolerance)
        # Mark the beginning of a new segment: first row or gap > tolerance
        df['segment'] = (df['gap'] > tolerance).cumsum()
        # Compute the duration of each segment (last end - first start)
        segment_duration = df.groupby('segment').agg(
            start_date=('start_date', 'first'), end_date=('end_date', 'last')
        )
        segment_duration['duration'] = segment_duration['end_date'] - segment_duration['start_date']

        # Convert the minimum time length string into a Timedelta
        min_duration = pd.Timedelta(min_duration)
        # Identify segments meeting the minimum duration requirement
        valid_segments = segment_duration[segment_duration['duration'] >= min_duration].index
        # Filter df to include only rows in valid segments
        valid_df = df[df['segment'].isin(valid_segments)].copy()
        # Drop helper columns to clean up the output
        valid_df.drop(columns=['prev_end', 'gap', 'segment'], inplace=True)

        if valid_df.empty:
            print(f"No valid continuous segment of at least {min_duration} found in {csv_file}")
            return None
        
        return valid_df

    except Exception as e:
        print(f"Error processing file {csv_file}: {e}")
        return None


def process_all_ids(input_parent_dir, output_parent_dir, fts_to_build, 
                    min_time_duration='60min', continuity_tolerance='1min'):
    """
    Processes all ID directories within input_parent_dir.
    For each target activity, it finds the corresponding CSV file in each ID folder,
    collects only the time series that meet the minimum continuous duration requirement,
    and concatenates them into a single CSV file for each activity.

    Parameters:
    - input_parent_dir (str): Parent folder containing subfolders for each user.
    - output_parent_dir (str): Folder where the concatenated time series files will be saved.
    - fts_to_build (list): List of activity names to process (e.g. ["StepCount"]).
    - min_time_duration (str): Minimum continuous duration required (e.g., '60min')
    - continuity_tolerance (str): Maximum gap allowed between consecutive records (e.g., '1min').
    """
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

    # Dictionary to store valid time series for each activity.
    activity_series = {activity: [] for activity in fts_to_build}

    # Iterate over each ID directory
    for id_dir in tqdm(id_dirs, desc="Processing ID directories"):
        id_dir_path = os.path.join(input_parent_dir, id_dir)
        print(f"\nProcessing ID: {id_dir}")

        for activity in fts_to_build:
            df_series = collect_time_series(
                id_dir_path, activity, min_time_duration, continuity_tolerance
            )
            if df_series is not None:
                activity_series[activity].append(df_series)

    # For each activity, concatenate the valid time series and save to CSV.
    for activity, df_list in activity_series.items():
        if df_list:
            combined_df = pd.concat(df_list, ignore_index=True)
            output_file = os.path.join(output_parent_dir, f"{activity}_timeseries.csv")
            
            try:
                combined_df.to_csv(output_file, index=False)
                # Calculate the accumulated duration in hours.
                total_seconds = (
                    combined_df['end_date'] - combined_df['start_date']
                ).dt.total_seconds().sum()
                total_hours = total_seconds / 3600
                
                print(f"Saved {activity} time series with accumulated duration of {total_hours:.0f} hours to {output_file}")
            except Exception as e:
                print(f"Error saving {output_file}: {e}")
        else:
            print(f"No valid time series found for activity: {activity}")

    print("All ID directories have been processed.")


if __name__ == "__main__":
    current_folder = os.getcwd()
    # Specify the input directory containing processed ID subdirectories
    input_parent_directory = current_folder + "/test_processed/"  # "/EV_aggregated_from_apple_24_12_w05min_update/"

    # Specify the parent output directory where the time series data will be saved
    output_parent_directory = input_parent_directory

    # Specify the target activities to build the time series
    fts_to_build = ["StepCount"]

    # Process all ID directories
    process_all_ids(
        input_parent_directory, output_parent_directory, fts_to_build, min_time_duration='300min', 
        continuity_tolerance='1min'
    )
