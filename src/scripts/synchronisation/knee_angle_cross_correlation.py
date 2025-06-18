import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import correlate, correlation_lags

def quaternion_to_euler(w, i, j, k):
    r = R.from_quat([i, j, k, w])
    return r.as_euler('xyz', degrees=True)

def read_euler_angles(imu_location, participant_id, session_id):
    input_csv = fr'Z:\Upper Body\IMU\{participant_id}\{session_id}\{imu_location}_imu_ori.csv' # Path to the IMU quaternions CSV file
    time, rolls, pitches, yaws = [], [], [], []
    with open(input_csv, 'r', newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            w,i,j,k = float(row['w']), float(row['i']), float(row['j']), float(row['k'])
            roll, pitch, yaw = quaternion_to_euler(w, i, j, k)
            rolls.append(roll); pitches.append(pitch); yaws.append(yaw)
            time.append(float(row['time']) if 'time' in row else len(time))
    return np.array(time), np.array(rolls), np.array(pitches), np.array(yaws)

def read_kinematic_data(kinematic, participant_id, session_id):
    input_mot = fr'Z:\Upper Body\Kinematics\{participant_id}\{session_id}.mot'
    time, values = [], []
    with open(input_mot, 'r') as infile:
        # Skip to endheader
        for line in infile:
            if line.strip().lower() == 'endheader':
                break
        # Grab the header names
        header = infile.readline().strip().split('\t')
        # Build a DictReader using those names
        reader = csv.DictReader(infile, fieldnames=header, delimiter='\t')
        for row in reader:
            # row is now a dict: row['time'], row[kinematic]
            time.append(float(row['time']))
            values.append(float(row[kinematic]))
    t = np.array(time)
    return t - t[0], np.array(values)


if __name__ == '__main__':
    participant_id = 'P026' # NOTE: Replace with the actual participant ID
    session_id = 'Straight vr 01' # NOTE: Replace with the actual session ID

    # Read both IMUs and kinematic data
    # NOTE: Change to left leg if needed
    time1, rolls1, pitches1, yaws1 = read_euler_angles('RightLowerLeg', participant_id, session_id)
    time2, rolls2, pitches2, yaws2 = read_euler_angles('RightUpperLeg', participant_id, session_id)
    time_kin, knee_angles = read_kinematic_data('knee_angle_r', participant_id, session_id)

    # Ensure both IMU arrays are the same length
    min_len = min(len(time1), len(time2))
    time = time1[:min_len]
    roll_diff = rolls1[:min_len] - rolls2[:min_len]
    pitch_diff = pitches1[:min_len] - pitches2[:min_len]
    yaw_diff = yaws1[:min_len] - yaws2[:min_len]

    # Initial plots
    ig, axs = plt.subplots(1, 2, figsize=(14,6))
    axs[0].plot(time, pitch_diff, label='Pitch Difference')
    axs[0].set(xlabel='Time (s)', ylabel='Angle Difference (°)', title='Pitch Difference between IMUs (LowerLeg - UpperLeg)')
    axs[0].legend(); axs[0].grid(True)

    axs[1].plot(time_kin, knee_angles, label='Knee Angle', color='orange')
    axs[1].set(xlabel='Time (s)', ylabel='Knee Angle (°)', title='Kinematic Knee Angle (Flexion/Extension)')
    axs[1].legend(); axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    # User input if they are happy with the plots and to continue
    if input("Are you happy with the plots? (y/n): ").lower() != 'y':
        print("Exiting. Inspect plots and rerun.")
        exit()

    # Subtract each signal's mean before cross-correlation
    x = pitch_diff - np.mean(pitch_diff)
    y = knee_angles - np.mean(knee_angles)
    
    # Calculate cross-correlation and lag estimation
    corr = correlate(x, y, mode='full')
    lags = correlation_lags(len(x), len(y), mode='full')
    lag_samp = lags[np.argmax(corr)]
    lag_sec = lag_samp / 100  # Assuming the data is sampled at 100 Hz

    print(f"Max correlation at {lag_samp} samples, which is {lag_sec} seconds.")

    # Plot correlation vs lag
    plt.figure(figsize=(8,4))
    plt.plot(lags, corr, label='Cross-correlation')
    plt.axvline(lag_samp, color='r', linestyle='--',
                label=f'Peak at {lag_sec:.3f}s')
    plt.xlabel('Lag (samples)')
    plt.ylabel('Correlation')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show()