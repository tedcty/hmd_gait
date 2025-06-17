import os
import glob
import pandas as pd
from scipy.signal import resample
import numpy as np

skipped_files = []

def upsample_csv(input_path, output_path, orig_freq=60, target_freq=100):
    df = pd.read_csv(input_path)
    if 'time' in df.columns:
        time_col = 'time'
    else:
        time_col = df.columns[0]  # Assume first column is time if not named

    # Check if time column has any values
    if df[time_col].isnull().all():
        skipped_files.append(input_path)
        return

    # Calculate duration and new number of samples
    duration = (df[time_col].iloc[-1] - df[time_col].iloc[0])
    n_samples = len(df)
    new_n_samples = int(np.round(n_samples * target_freq / orig_freq))

    # Resample all columns except time
    data_cols = [col for col in df.columns if col != time_col]
    resampled_data = {}
    for col in data_cols:
        resampled_data[col] = resample(df[col].values, new_n_samples)

    # Create new time vector starting from 0 and round to 2 decimal places
    new_time = np.round(
        np.linspace(0, df[time_col].iloc[-1] - df[time_col].iloc[0], new_n_samples),
        2
    )
    resampled_df = pd.DataFrame({time_col: new_time})
    for col in data_cols:
        resampled_df[col] = resampled_data[col]

    resampled_df.to_csv(output_path, index=False)

def upsample_all_csvs(root_folder, orig_freq=60, target_freq=100):
    for csv_file in glob.glob(os.path.join(root_folder, '**', '*.csv'), recursive=True):
        if not csv_file.endswith('.sto.csv'):
            print(f"Upsampling {csv_file}")
            upsample_csv(csv_file, csv_file, orig_freq, target_freq)

if __name__ == "__main__":
    root_folder = fr'Z:\Upper Body\IMU\P026'
    upsample_all_csvs(root_folder)
    if skipped_files:
        print("Skipped files due to empty time column:")
        for f in skipped_files:
            print(f)