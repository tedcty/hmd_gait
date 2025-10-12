import os
import glob
import pandas as pd
from scipy.signal import resample_poly
import numpy as np

skipped_files = []

excluded_participants = ["P001" ,"P010", "P011", "P012", "P014", "P016", "P017", "P019", "P026", "P043"]

def upsample_csv(input_path, output_path, orig_freq=60, target_freq=100):
    df = pd.read_csv(input_path)
    time_col = 'time' if 'time' in df.columns else df.columns[0]

    # Skip files with no time data
    if df[time_col].isnull().all():
        skipped_files.append(input_path)
        return

    # Compute new sample count
    n_samples = len(df)
    new_n_samples = int(np.ceil(n_samples * target_freq / orig_freq))

    # Zero-based exact 100 Hz time vector
    new_time = np.arange(new_n_samples) / target_freq
    
    # Resample all other data columns via polyphase filtering
    data_cols = [col for col in df.columns if col != time_col]
    resampled_data = {}
    for col in data_cols:
        resampled_data[col] = resample_poly(df[col].values, target_freq, orig_freq)

    # Assemble
    resampled_df = pd.DataFrame({time_col: new_time})
    for col in data_cols:
        resampled_df[col] = resampled_data[col]

    # Write to output CSV
    resampled_df.to_csv(output_path, index=False)


def upsample_all_csvs(root_folder, orig_freq=60, target_freq=100):
    for csv_file in glob.glob(os.path.join(root_folder, '**', '*.csv'), recursive=True):
        if not csv_file.endswith('.sto.csv'):
            # Extract participant ID from the file path
            path_parts = csv_file.split(os.sep)
            participant_id = None
            
            # Look for participant folder pattern (e.g., P001, P010, etc.)
            for part in path_parts:
                if part.startswith('P') and len(part) == 4 and part[1:].isdigit():
                    participant_id = part
                    break
            
            # Skip if participant is in excluded list
            if participant_id in excluded_participants:
                continue
                
            print(f"Upsampling {csv_file}")
            upsample_csv(csv_file, csv_file, orig_freq, target_freq)

if __name__ == "__main__":
    root_folder = fr'Z:\Upper Body\IMU'
    upsample_all_csvs(root_folder)
    if skipped_files:
        print("Skipped files due to empty time column:")
        for f in skipped_files:
            print(f)