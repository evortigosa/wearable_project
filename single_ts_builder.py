import os
import pandas as pd
import multiprocessing as mp
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
        # The data was processed with a fixed window size (in minutes)
        df = pd.read_csv(csv_file)
        # Assume it has 'start_date' and 'end_date' columns
        df["start_date"]= pd.to_datetime(df["start_date"], errors="coerce", utc=True)
        df["end_date"]  = pd.to_datetime(df["end_date"], errors="coerce", utc=True)
        # Drop rows where conversion failed
        df = df.dropna(subset=["start_date", "end_date"])
        
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
    

def process_single_id(id_dir, input_parent_dir, output_parent_dir, activity, min_time_duration, 
                      continuity_tolerance):
    """
    Processes a single ID directory and appends results to the corresponding CSV file.
    """
    id_dir_path = os.path.join(input_parent_dir, id_dir)
    output_file = os.path.join(output_parent_dir, f"{activity}_timeseries.csv")
    
    df_series = collect_time_series(
        id_dir_path, activity, min_time_duration, continuity_tolerance
    )
    total_hours = 0

    if df_series is not None:
        try:
            lock = mp.Lock()  # Ensures safe write to file in parallel execution
            with lock:
                df_series.to_csv(                       # Ensures header is written only once
                    output_file, mode='a', index=False, header=(not os.path.exists(output_file))
                )

            # Calculate the accumulated duration in hours
            total_seconds = (df_series['end_date'] - df_series['start_date']).dt.total_seconds().sum()
            total_hours += total_seconds / 3600

            print(f"Appended {len(df_series)} rows from {id_dir} to {output_file}")
        except Exception as e:
            print(f"Error saving data from {id_dir} to {output_file}: {e}")

    return total_hours


def arguments_wrapper(args):
    """
    Helper function to unpack arguments.
    """
    return process_single_id(*args)


def process_all_ids(input_parent_dir, output_parent_dir, fts_to_build, 
                    min_time_duration='60min', continuity_tolerance='1min', max_workers=4):
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
    # Check the number of CPUs/cores available
    max_workers= min(max_workers, max(1, os.cpu_count() - 1))
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

    for activity in fts_to_build:
        output_file = os.path.join(output_parent_dir, f"{activity}_timeseries.csv")
        # Clear previous data
        if os.path.exists(output_file):
            os.remove(output_file)

        total_hours = 0

        # Create multiprocessing pool
        with mp.Pool(processes=max_workers) as pool:
            task_args = [
                (id_dir, input_parent_dir, output_parent_dir, activity, min_time_duration, continuity_tolerance)
                for id_dir in id_dirs
            ]
            with tqdm(total=len(id_dirs), desc=f"Processing {activity}", unit="file") as pbar:
                for hours in pool.imap_unordered(arguments_wrapper, task_args):
                    total_hours += hours
                    pbar.update(1)

        total_hours= int(total_hours)
        print(f"Finished processing {activity} time series.\
            \n- Continuity tolerance between consecutive records: {continuity_tolerance}\
            \n- Minimum continuous duration: {min_time_duration}\
            \n- Accumulated duration of {total_hours} hours\
            \nResults saved to {output_file}")
        
    print("All ID directories have been processed.")


if __name__ == "__main__":
    current_folder = os.getcwd()
    # Specify the input directory containing processed ID subdirectories
    input_parent_directory = current_folder + "/EV_aggregated_from_apple_24_12_w05min_update/"

    # Specify the parent output directory where the time series data will be saved
    output_parent_directory = input_parent_directory

    # Specify the target activities to build the time series
    fts_to_build = ["StepCount"]

    # Process all ID directories
    process_all_ids(
        input_parent_directory, output_parent_directory, fts_to_build, min_time_duration='300min', 
        continuity_tolerance='6min', max_workers=8
    )
