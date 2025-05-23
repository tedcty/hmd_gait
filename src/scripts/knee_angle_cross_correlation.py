import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(w, i, j, k):
    r = R.from_quat([i, j, k, w])
    return r.as_euler('xyz', degrees=True)

participant_id = 'P001' # Replace with the actual participant ID
session_id = 'Combination 1' # Replace with the actual session ID

def read_euler_angles(imu_location):
    input_csv = fr'Z:\Upper Body\IMU\{participant_id}\{session_id}\{imu_location}_imu_ori.csv' # Path to the IMU quaternions CSV file
    time = []
    rolls = []
    pitches = []
    yaws = []
    with open(input_csv, 'r', newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            w = float(row['w'])
            i = float(row['i'])
            j = float(row['j'])
            k = float(row['k'])
            roll, pitch, yaw = quaternion_to_euler(w, i, j, k)
            rolls.append(roll)
            pitches.append(pitch)
            yaws.append(yaw)
            if 'time' in row:
                time.append(float(row['time']))
            else:
                time.append(len(time))
    return np.array(time), np.array(rolls), np.array(pitches), np.array(yaws)

# Read both IMUs
time1, rolls1, pitches1, yaws1 = read_euler_angles('RightLowerLeg')
time2, rolls2, pitches2, yaws2 = read_euler_angles('RightUpperLeg')

# Ensure both arrays are the same length
# Shift time so it starts at 0
time1 = time1 - time1[0]
time2 = time2 - time2[0]
min_len = min(len(time1), len(time2))
time = time1[:min_len]
roll_diff = rolls1[:min_len] - rolls2[:min_len]
pitch_diff = pitches1[:min_len] - pitches2[:min_len]
yaw_diff = yaws1[:min_len] - yaws2[:min_len]

plt.figure(figsize=(10, 6))
plt.plot(time, pitch_diff, label='Pitch Difference')
# plt.plot(time, yaw_diff, label='Yaw Difference')
plt.xlabel('Time')
plt.ylabel('Angle Difference (degrees)')
plt.title('Euler Angle Differences (RightLowerLeg - RightUpperLeg) Over Time')
plt.legend()
plt.tight_layout()
plt.show()