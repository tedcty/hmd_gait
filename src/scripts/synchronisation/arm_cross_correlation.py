import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import correlate

def quaternion_to_euler(w, i, j, k):
    r = R.from_quat([i, j, k, w])
    return r.as_euler('xyz', degrees=True)

def read_euler_angles(imu_location, participant_id, session_id):
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

def read_kinematic_data(kinematic, participant_id, session_id):
    input_mot = fr'Z:\Upper Body\Kinematics\{participant_id}\{session_id}.mot'
    time = []
    knee_angles = []
    with open(input_mot, 'r') as infile:
        # Skip to endheader
        for line in infile:
            if line.strip().lower() == 'endheader':
                break
        # Grab the header names
        header_line = infile.readline().strip()
        # Build a DictReader using those names
        fieldnames = header_line.split('\t')
        reader = csv.DictReader(infile, fieldnames=fieldnames, delimiter='\t')
        for row in reader:
            # row is now a dict: row['time'], row[kinematic]
            time.append(float(row['time']))
            knee_angles.append(float(row[kinematic]))
    return np.array(time), np.array(knee_angles)


participant_id = 'P001' # NOTE: Replace with the actual participant ID
session_id = 'Defined normal 1' # NOTE: Replace with the actual session ID

# Read both IMUs
# NOTE: Change to left arm if needed
time, rolls, pitches, yaws = read_euler_angles('RightUpperArm', participant_id, session_id)

# Shift time so it starts at 0
time -= time[0]

# Read kinematic data
# NOTE: Change to left arm if needed (or even to flexion/extension instead of abduction/adduction)
time_kinematic, knee_angles = read_kinematic_data('arm_add_r', participant_id, session_id)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot arm IMU
axs[0].plot(time, pitches, label='Upper Arm IMU Pitch')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Angle Difference (degrees)')
axs[0].set_title('Pitch Difference between IMUs (RightLowerLeg - RightUpperLeg)')
axs[0].legend()
axs[0].grid(True)

# Plot knee angle
axs[1].plot(time_kinematic, knee_angles, label='Knee Angle', color='orange')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Knee Angle (degrees)')
axs[1].set_title('Knee Angle Over Time')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# User input if they are happy with the plots and to continue
happy = input("Are you happy with the plots? (y/n): ")
if happy.lower() == 'y':
    print("Continuing with the next steps...")
elif happy.lower() == 'n':
    print("Exiting the script. Please check the plots and try again.")
    exit()

# Calculate cross-correlation
correlation = correlate(pitches, knee_angles, mode='full')

# Find lag at which correlation is maximum
lag_index = np.argmax(correlation)
lag_value = lag_index - (len(knee_angles) - 1)
# Calculate time lag in seconds
time_lag = lag_value / 100  # Assuming the data is sampled at 100 Hz

print(f"Maximum cross-correlation occurs at lag: {lag_value} samples, which is {time_lag} seconds.")