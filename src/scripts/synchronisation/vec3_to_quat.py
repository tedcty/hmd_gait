from scipy.spatial.transform import Rotation as R
import glob
import csv
import numpy as np
import os

def vec3_to_quat(input_csv):
    # Check if input file ends with correct suffix
    if not input_csv.endswith('_imu_vec3_2.csv'):
        return
        
    # Create output filename by replacing suffix
    output_csv = input_csv.replace('_imu_vec3_2.csv', '_imu_ori.csv')
    
    # Read input CSV
    data = []
    with open(input_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        for row in reader:
            data.append([float(x) for x in row])
    
    data = np.array(data)
    
    # Convert Euler angles to quaternions
    # Assuming Euler angles are in columns 1,2,3 in radians
    rot = R.from_euler('xyz', data[:, 1:4]) # NOTE: Change to columns 4, 5, 6 if angular velocity instead of angular acceleration!
    quats = rot.as_quat()  # Returns (w,x,y,z) format
    
    # Write output CSV with quaternions
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'w', 'i', 'j', 'k'])
        for i in range(len(data)):
            writer.writerow([data[i,0], *quats[i]])
    
    print(f"Created {output_csv}")

if __name__ == "__main__":
    root_folder = fr'Z:\Upper Body\IMU\P001\Combination AR 3'
    for csv_file in glob.glob(os.path.join(root_folder, '**', '*_imu_vec3_2.csv'), recursive=True):
        print(f"Processing {csv_file}")
        vec3_to_quat(csv_file)