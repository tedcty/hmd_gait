import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import correlate, correlation_lags
from knee_angle_cross_correlation import quaternion_to_euler, read_euler_angles, read_kinematic_data

def optimise_lag(time_x, x, time_y, y, lb, ub, new_freq=1000):
    # Extract window
    mask_x = (time_x >= lb) & (time_x <= ub)
    mask_y = (time_y >= lb) & (time_y <= ub)
    tx, xx = time_x[mask_x], x[mask_x]
    ty, yy = time_y[mask_y], y[mask_y]
    # New common 1000 Hz axis
    common_t = np.arange(lb, ub, 1/new_freq)
    xi = np.interp(common_t, tx, xx)
    yi = np.interp(common_t, ty, yy)
    # Plot two segments side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax1.plot(common_t, xi)
    ax1.set_title('Upsampled IMU segment')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Signal amplitude')

    ax2.plot(common_t, yi)
    ax2.set_title('Upsampled Kinematics segment')
    ax2.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()
    
    # Zero-mean
    xi -= np.mean(xi)
    yi -= np.mean(yi)
    # Cross-correlation
    corr = correlate(xi, yi, mode='full')
    lags = correlation_lags(len(xi), len(yi), mode='full')
    lag_samp = lags[np.argmax(corr)]
    lag_sec = lag_samp / new_freq
    return lag_sec, corr, lags, common_t, xi, yi

if __name__ == '__main__':
    participant_id = 'P010'  # NOTE: Replace with the actual participant ID
    session_id = 'Obstacles VR 1'  # NOTE: Replace with the actual session ID

    # Read both IMUs and kinematic data
    time1, rolls1, pitches1, yaws1 = read_euler_angles('LeftLowerLeg', participant_id, session_id)
    time2, rolls2, pitches2, yaws2 = read_euler_angles('LeftUpperLeg', participant_id, session_id)
    time_kin, knee_angles = read_kinematic_data('knee_angle_l', participant_id, session_id)

    # Difference signal
    min_len = min(len(time1), len(time2))
    time = time1[:min_len]
    pitch_diff = pitches1[:min_len] - pitches2[:min_len]

    lb = float(input("Enter lower bound (s): "))
    ub = float(input("Enter upper bound (s): "))

    # Optimise lag
    lag_sec, corr, lags, common_t, xi, yi = optimise_lag(time, pitch_diff, time_kin, knee_angles, lb, ub)

    print(f"Optimised lag: {lag_sec:.4f} seconds")

    # Plotting
    plt.figure(figsize=(8, 4))
    plt.plot(lags/1000, corr)
    plt.axvline(lag_sec, color='r', linestyle='--', label=f'Optimised Lag: {lag_sec:.4f}s')
    plt.xlabel('Lag (s)')
    plt.ylabel('Correlation')
    plt.legend(); plt.grid(True)
    plt.show()