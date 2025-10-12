from sync_utils import read_euler_angles, read_kinematic_data, compute_coarse_offset, refine_offset_upsample
from scipy.signal import correlate, correlation_lags
import numpy as np
import matplotlib.pyplot as plt
from ptb.util.math.filters import Butterworth

if __name__ == '__main__':
    participant_id = 'P003' # NOTE: Replace with the actual participant ID
    session_id = 'Stairs VR 1' # NOTE: Replace with the actual session ID
    limb = 'L'  # 'L' for left leg, 'R' for right leg
    
    if limb == 'L':
        lower_leg_sensor = 'LeftLowerLeg'
        upper_leg_sensor = 'LeftUpperLeg'
        knee_kinematic = 'knee_angle_l'
    elif limb == 'R':
        lower_leg_sensor = 'RightLowerLeg'
        upper_leg_sensor = 'RightUpperLeg'
        knee_kinematic = 'knee_angle_r'

    # Read both IMUs and kinematic data
    time1, rolls1, pitches1, yaws1 = read_euler_angles(lower_leg_sensor, participant_id, session_id)
    time2, rolls2, pitches2, yaws2 = read_euler_angles(upper_leg_sensor, participant_id, session_id)
    time_kin, knee_angles = read_kinematic_data(knee_kinematic, participant_id, session_id)

    # Ensure both IMU arrays are the same length
    min_len = min(len(time1), len(time2))
    time = time1[:min_len]
    roll_diff = rolls1[:min_len] - rolls2[:min_len]
    pitch_diff = pitches1[:min_len] - pitches2[:min_len]
    yaw_diff = yaws1[:min_len] - yaws2[:min_len]

    # Apply Butterworth low-pass filter to the pitch difference and knee angles
    pitch_diff = Butterworth.butter_low_filter(data=pitch_diff, fs=100, cut=3)
    # pitch_diff = pitch_diff / np.max(np.abs(pitch_diff))
    knee_angles = Butterworth.butter_low_filter(data=knee_angles, fs=100, cut=3)
    # knee_angles = knee_angles / np.max(np.abs(knee_angles))

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

    # Visual check of alignment
    aligned_time_imu = time - lag_sec
    plt.plot(aligned_time_imu, pitch_diff, label='IMU pitch difference')
    plt.plot(time_kin, knee_angles, label='Knee angle (shifted)')
    plt.legend(); plt.grid(True)
    plt.title(f'Peak‐based alignment (offset = {lag_sec:.3f}s)')
    plt.show()

    # Precise refinement via upsampling
    precise_lag = refine_offset_upsample(x, y, fs=100, up_fs=1000, win_sec=0.1)
    print(f"Refined offset via upsampling: {precise_lag:.6f} seconds")