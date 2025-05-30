import pandas as pd
import os
import glob

def check_imu(input_path):

    df = pd.read_csv(input_path)
    
    # Check if dataframe is empty (only headers, no data)
    if df.empty or len(df) == 0:
        print(f"{input_path} has no data (only headers).")
        return input_path
        
if __name__ == "__main__":
    root_folder = fr'Z:\Mocap\Movella_Re'
    empty_input_paths = []
    for csv_file in glob.glob(os.path.join(root_folder, '**', '*.csv'), recursive=True):
        empty_input_path = check_imu(csv_file)
        if empty_input_path:
            empty_input_paths.append(empty_input_path)
    # Write empty input paths to a file
    if empty_input_paths:
        with open(r'Z:\Upper Body\empty_imu_files.txt', 'w') as f:
            for path in empty_input_paths:
                f.write(f"{path}\n")
        print(f"Found {len(empty_input_paths)} empty IMU files. Paths written to empty_imu_files.txt.")
    else:
        print("No empty IMU files found.")
