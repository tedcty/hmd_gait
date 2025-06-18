from sync_utils import read_euler_angles, read_kinematic_data, compute_coarse_offset, refine_offset_upsample
from scipy.signal import correlate, correlation_lags
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    participant_id = 'P026' # NOTE: Replace with the actual participant ID
    session_id = 'Combo AR 02' # NOTE: Replace with the actual session ID

    # Read both IMUs and kinematic data
    # NOTE: Change to left leg if needed
    time1, rolls1, pitches1, yaws1 = read_euler_angles('RightUpperArm', participant_id, session_id)
    time2, rolls2, pitches2, yaws2 = read_euler_angles('T8', participant_id, session_id)  # sternum
    time_kin, knee_angles = read_kinematic_data('arm_add_r', participant_id, session_id)

    # Ensure both IMU arrays are the same length
    min_len = min(len(time1), len(time2))
    time = time1[:min_len]
    roll_diff = rolls1[:min_len] - rolls2[:min_len]
    pitch_diff = pitches1[:min_len] - pitches2[:min_len]
    yaw_diff = yaws1[:min_len] - yaws2[:min_len]

    # Initial plots
    ig, axs = plt.subplots(1, 2, figsize=(14,6))
    axs[0].plot(time, pitch_diff, label='Pitch Difference')
    axs[0].set(xlabel='Time (s)', ylabel='Angle Difference (°)', title='Pitch Difference between IMUs (UpperArm - Sternum)')
    axs[0].legend(); axs[0].grid(True)

    axs[1].plot(time_kin, knee_angles, label='Knee Angle', color='orange')
    axs[1].set(xlabel='Time (s)', ylabel='Arm Adduction Angle (°)', title='Arm Adduction Over Time')
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

    # Precise refinement via upsampling
    precise_lag = refine_offset_upsample(x, y, fs=100, up_fs=1000, win_sec=0.1)
    print(f"Refined offset via upsampling: {precise_lag:.6f} seconds")