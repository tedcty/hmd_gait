import pandas as pd
import numpy as np
import os
from ptb.util.io.helper import StorageIO
from typing import Dict
from event_constants import (WHOLE_EVENTS, REPETITIVE_EVENTS, EVENT_SIZES)

# Cache for computed std (samples) per task for whole events
STD_CACHE: Dict[str, Dict[str, int]] = {}

STD_CACHE_GLOBAL: Dict[str, int] = {}


def read_offset_times(file_path):
    # Read offset times Excel file (with a two-level header and PID column as index)
    offsets_df = pd.read_excel(file_path, header=[0, 1], index_col=0)

    # Extract the first two levels into arrays
    lvl0 = offsets_df.columns.get_level_values(0)
    lvl1 = offsets_df.columns.get_level_values(1)

    # Forward-fill the top level by using a Series
    lvl0_filled = pd.Series(lvl0).ffill().values

    # Rebuild the MultiIndex
    offsets_df.columns = pd.MultiIndex.from_arrays([lvl0_filled, lvl1], names=offsets_df.columns.names)

    return offsets_df


def read_event_labels(file_path):

    # Define the different tasks completed by each participant
    tasks = ['Combination', 'Defined', 'Free', 'Obstacles', 'Stairs', 'Straight']

    # Initialize an empty dictionary to hold DataFrames for each task
    events_df = {}

    for task in tasks:
        # Read event labels Excel file (with a two-level header and PID column as index)
        events_df[task] = pd.read_excel(file_path, header=[0, 1], index_col=[0, 1], sheet_name=task)

        # Extract the first two levels into arrays
        lvl0 = events_df[task].columns.get_level_values(0)
        lvl1 = events_df[task].columns.get_level_values(1)

        # Forward-fill the top level by using a Series
        lvl0_filled = pd.Series(lvl0).ffill().values

        # Rebuild the MultiIndex
        events_df[task].columns = pd.MultiIndex.from_arrays([lvl0_filled, lvl1], names=events_df[task].columns.names)

    return events_df


def sync_imu_with_kinematics(offsets_df):
    # Initialize an empty dictionary to hold synced data
    synced = {}
    # Loop through each participant's data in the offsets DataFrame
    for pid, trial in offsets_df.iterrows():
        # Loop through each task and condition for the current participant
        for (task, cond), offset in trial.items():
            # Search for specific task's IMU folder in participant folder
            imu_participant_folder = os.path.join('Z:/Upper Body/IMU', pid)
            for folder in os.listdir(imu_participant_folder):
                if task.lower() in folder.lower() and cond.lower() in folder.lower():
                    imu_task_folder = os.path.join(imu_participant_folder, folder)
                    # Find the IMU CSV files in the task folder
                    imu_files = [f for f in os.listdir(imu_task_folder) if f.endswith('.csv') and not f.endswith('.sto.csv')]

            # Search for specific task's kinematic MOT file in participant folder
            kinematic_participant_folder = os.path.join('Z:/Upper Body/Kinematics', pid)
            for file in os.listdir(kinematic_participant_folder):
                if task.lower() in file.lower() and cond.lower() in file.lower() and file.endswith('.mot'):
                    kinematic_file = os.path.join(kinematic_participant_folder, file)

            # Read kinematics data from the MOT file
            kin_storage = StorageIO.load(kinematic_file)
            kin_df = kin_storage.data  # DataFrame with kinematic data

            # Read IMU data for each CSV file in the task folder
            imu_streams = {}  # Dictionary to hold multiple IMU streams
            for imu_file in imu_files:
                imu_path = os.path.join(imu_task_folder, imu_file)
                imu_df = pd.read_csv(imu_path)

                # Shift IMU times (backwards) so they line up with kinematics
                imu_df['time'] = imu_df['time'] - offset

                # Remove non-overlapping times (that would be before the first kinematic time or after the last kinematic time)
                imu_df = imu_df[(imu_df['time'] >= kin_df['time'].min()) & (imu_df['time'] <= kin_df['time'].max())]

                # Store the IMU stream in the dictionary
                imu_streams[imu_file] = imu_df

            # Store the synced IMU and kinematics data in the synced dictionary
            synced[(pid, task, cond)] = {
                'kin': kin_df,
                'imu': imu_streams
            }
    
    return synced


def compute_whole_event_std_global(events_dict, sample_rate=100):
    # Return cached if available
    if STD_CACHE_GLOBAL:
        return STD_CACHE_GLOBAL

    std_map: Dict[str, int] = {}
    for base_event in WHOLE_EVENTS:
        all_durations = []
        for _, event_df in events_dict.items():  # iterate every task sheet
            event_cols = [col[0] for col in event_df.columns if col[0].startswith(base_event)]
            for event in event_cols:
                d = event_df[(event, 'Duration')].dropna().values
                d = d[d > 0]
                all_durations.extend(d)
        if all_durations:
            std_map[base_event] = np.std(all_durations) * sample_rate

    # cache and return
    STD_CACHE_GLOBAL.update(std_map)
    return std_map


def compute_whole_event_std(task, events_dict, sample_rate=100):
    if task in STD_CACHE:
        return STD_CACHE[task]
    
    std_map: Dict[str, int] = {}

    # For each whole event, collect durations across ALL tasks
    for base_event in WHOLE_EVENTS:
        all_durations = []
        # Look through all tasks for this event
        for task_sheet, event_df in events_dict.items():
            # Get all columns that start with this event name
            event_cols = [col[0] for col in event_df.columns if col[0].startswith(base_event)]
            for event in event_cols:
                # Get durations directly from Duration column
                durations = event_df[(event, 'Duration')].dropna().values
                durations = durations[durations > 0]  # Filter out any non-positive durations
                all_durations.extend(durations)

        if len(all_durations) > 0:
            # Convert to samples and get the standard deviation
            std_samples = np.std(all_durations) * sample_rate
            std_map[base_event] = std_samples  # Store under base event name
    
    STD_CACHE[task] = std_map
    return std_map


def trim_events(df, events_dict, pid, task, cond, time_col='time', sample_rate=100):
    # Grab the event‐labels DataFrame for this task
    event_df = events_dict[task]
    trimmed = {}
    # List of events
    events = event_df.columns.levels[0]
    # Sort once per call
    df_sorted = df.sort_values(time_col).reset_index(drop=True)

    # Get standard deviations for whole events
    std_map = compute_whole_event_std(task, events_dict, sample_rate)

    # Loop through each event
    for event in events:
        # Get base event name (remove numbering if present)
        base_event = event.split(' ', 1)[0]  # Get first word
        for known_event in WHOLE_EVENTS | REPETITIVE_EVENTS:
            if event.startswith(known_event):
                base_event = known_event
                break
        # Check if the event exists for this participant and condition
        raw_start = event_df.loc[(pid, cond), (event, 'Start')]
        raw_end   = event_df.loc[(pid, cond), (event, 'End')]
        if pd.isna(raw_start) or pd.isna(raw_end):
            continue

        if base_event in WHOLE_EVENTS:
            # For whole events, use mean window + std with buffers
            event_size = EVENT_SIZES[base_event]  # Already in frames
            std_size = std_map.get(base_event, 0)  # Already in frames
            # Round the sum to nearest whole number of frames
            center_window = round(event_size + std_size)  
            center_window_seconds = center_window / sample_rate
            
            # Center the window around the middle of the event
            event_mid = (raw_start + raw_end) / 2
            start_time = event_mid - (center_window_seconds / 2) - center_window_seconds  # Left buffer
            end_time = event_mid + (center_window_seconds / 2) + center_window_seconds    # Right buffer
            
        elif base_event in REPETITIVE_EVENTS:
            # For repetitive events, keep full duration plus one cycle buffer
            cycle_size = EVENT_SIZES[base_event] / sample_rate
            start_time = raw_start - cycle_size  # Buffer of one cycle
            end_time = raw_end + cycle_size      # Buffer of one cycle

        # Convert to sample indices and ensure within bounds
        start_idx = max(0, int(np.ceil(start_time * sample_rate)))
        end_idx = min(len(df_sorted) - 1, int(np.floor(end_time * sample_rate)))
        
        # Slice by position
        trimmed[event] = df_sorted.iloc[start_idx:end_idx + 1].copy()

    return trimmed


def trim_all_streams(synced_data, events_df, time_col='time'):
    trimmed = {}
    # Loop through each participant's data in the synced data
    for (pid, task, cond), streams in synced_data.items():
        # Check if the task and condition exist in the event labels
        if task not in events_df:
            print(f"Skipping {pid} {task} {cond} as it is not in the event labels.")
            continue
        print(f"Trimming data for {pid} {task} {cond}")
        # Use the index‐based trim for kinematics and each IMU
        kin_windows = trim_events(streams['kin'],  events_df, pid, task, cond, time_col=time_col, sample_rate=100)
        imu_windows = {}
        for imu_file, imu_df in streams['imu'].items():
            imu_windows[imu_file] = trim_events(imu_df, events_df, pid, task, cond, time_col=time_col, sample_rate=100)
        trimmed[(pid, task, cond)] = {'kin': kin_windows, 'imu': imu_windows}
    return trimmed


if __name__ == "__main__":

    # Read offset times from offset times Excel file into a DataFrame
    offsets_df = read_offset_times(file_path="Z:/Upper Body/IMU-Kinematics Offset.xlsx")
    print("Offset times read successfully.")

    # Read event labels from event labels Excel file into a DataFrame
    events_df = read_event_labels(file_path="Z:/Upper Body/Event labels.xlsx")
    print("Event labels read successfully.")

    # Sync IMU data with kinematics data using the offset times
    synced_data = sync_imu_with_kinematics(offsets_df)
    print("IMU data synced with kinematics data successfully.")

    # Trim all streams from synced data based on the event labels
    trimmed_data = trim_all_streams(synced_data, events_df, time_col='time')
    print("All streams trimmed based on event labels successfully.")

    # Save the trimmed files
    for (pid, task, cond), streams in trimmed_data.items():
        # Make participant folders if it doesn't exist
        kin_out_dir = os.path.join('Z:/Upper Body/Events/Kinematics', pid)
        imu_out_dir = os.path.join('Z:/Upper Body/Events/IMU', pid)
        os.makedirs(kin_out_dir, exist_ok=True)
        os.makedirs(imu_out_dir, exist_ok=True)

        # Save each event as its own file
        # Kinematics
        for event, kin_df in streams['kin'].items():
            kin_file_name = f"{pid}_{task}_{cond}_{event}.mot.csv"  # Use .mot.csv to indicate it's a kinematic file
            kin_df.to_csv(os.path.join(kin_out_dir, kin_file_name), index=False)
        # IMU
        for imu_file, imu_dict in streams['imu'].items():
            for event, imu_df in imu_dict.items():
                base, _ = os.path.splitext(imu_file)
                imu_file_name = f"{pid}_{task}_{cond}_{event}_{base}.csv"
                imu_df.to_csv(os.path.join(imu_out_dir, imu_file_name), index=False)

        print(f"Saved trimmed data for {pid} {task} {cond}")