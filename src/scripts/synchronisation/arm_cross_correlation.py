from sync_utils import read_euler_angles, read_kinematic_data
from scipy.signal import correlate, correlation_lags, find_peaks, resample
import numpy as np
import matplotlib.pyplot as plt
from ptb.util.math.filters import Butterworth

if __name__ == '__main__':
    participant_id = 'P026' # NOTE: Replace with the actual participant ID
    session_id = 'Combo vr 01' # NOTE: Replace with the actual session ID

    # Read both IMUs and kinematic data
    # NOTE: Change to left leg if needed
    time1, rolls1, pitches1, yaws1 = read_euler_angles('LeftUpperArm', participant_id, session_id)
    time2, rolls2, pitches2, yaws2 = read_euler_angles('T8', participant_id, session_id)  # sternum
    time_kin, knee_angles = read_kinematic_data('arm_add_l', participant_id, session_id)

    # Ensure both IMU arrays are the same length
    min_len = min(len(time1), len(time2))
    time = time1[:min_len]
    roll_diff = rolls1[:min_len] - rolls2[:min_len]
    pitch_diff = pitches1[:min_len] - pitches2[:min_len]
    yaw_diff = yaws1[:min_len] - yaws2[:min_len]

    # # Apply Butterworth low-pass filter to the pitch difference and knee angles
    # pitch_diff = Butterworth.butter_low_filter(data=pitch_diff, fs=100, cut=3)
    # # pitch_diff = pitch_diff / np.max(np.abs(pitch_diff))
    # knee_angles = Butterworth.butter_low_filter(data=knee_angles, fs=100, cut=3)
    # # knee_angles = knee_angles / np.max(np.abs(knee_angles))

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

    # Find peaks from arm raise during T-pose at the start and end of the session
    peaks_imu, _ = find_peaks(-pitch_diff, height=30, distance=10)
    peaks_kin, _ = find_peaks(-knee_angles, height=np.max(np.abs(knee_angles)) * 0.8, distance=10)

    # Get the very first and last peaks
    if len(peaks_imu) >= 2 and len(peaks_kin) >= 2:
        peaks_imu = np.array([peaks_imu[0], peaks_imu[-1]])
        peaks_kin = np.array([peaks_kin[0], peaks_kin[-1]])
    # else:
    #     print("Not enough peaks detected. Ensure the data has sufficient variation.")
    #     exit()
       
    print("IMU peaks at times:", time[peaks_imu])
    print("Kin peaks at times:", time_kin[peaks_kin])

    # Plot peaks on the IMU pitch difference
    ig, axs = plt.subplots(1, 2, figsize=(14,6))
    axs[0].plot(time, pitch_diff, label='Pitch Difference')
    axs[0].plot(time[peaks_imu], pitch_diff[peaks_imu], 'x')
    axs[0].set(xlabel='Time (s)', ylabel='Angle Difference (°)', title='Pitch Difference between IMUs (UpperArm - Sternum)')
    axs[0].legend(); axs[0].grid(True)

    axs[1].plot(time_kin, knee_angles, label='Knee Angle', color='orange')
    axs[1].plot(time_kin[peaks_kin], knee_angles[peaks_kin], 'x')
    axs[1].set(xlabel='Time (s)', ylabel='Arm Adduction Angle (°)', title='Arm Adduction Over Time')
    axs[1].legend(); axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    # User input if they are happy with the peaks and to continue
    if input("Have the first and last peaks been identified correctly? (y/n): ").lower() != 'y':
        print("Exiting. Inspect plots and rerun.")
        exit()

    # Calculate mean offset by averaging the time differences between peaks
    if len(peaks_imu) != len(peaks_kin):
        print("Warning: Unequal number of peaks detected in IMU and kinematic data.")
    else:
        offsets = time[peaks_imu] - time_kin[peaks_kin]
        print(f"Offsets between IMU and kinematic peaks: {offsets}")
        mean_offset = np.mean(offsets)
        print(f"Mean offset between IMU and kinematic data: {mean_offset:.3f} seconds")

    # Visual check of alignment
    aligned_time_imu = time - mean_offset
    plt.plot(aligned_time_imu, pitch_diff, label='IMU pitch difference')
    plt.plot(time_kin, knee_angles, label='Arm adduction (shifted)')
    plt.legend(); plt.grid(True)
    plt.title(f'Peak‐based alignment (offset = {mean_offset:.3f}s)')
    plt.show()

    fs = 1.0/(time[1] - time[0])  # Sampling frequency from time array
    half_win = 0.5  # Half window size in seconds

    local_offsets = []
    for peak_imu, peak_kin in zip(peaks_imu, peaks_kin):
        # Slice out each half-window segment around the peaks
        t_imu = time[peak_imu]
        t_kin = time_kin[peak_kin]
        mask_imu = (time >= t_imu - half_win) & (time <= t_imu + half_win)
        mask_kin = (time_kin >= t_kin - half_win) & (time_kin <= t_kin + half_win)
        seg_imu = pitch_diff[mask_imu]
        seg_kin = knee_angles[mask_kin]

        # Upsample both segments
        M = int(len(seg_imu) * 1000 / fs)  # samples at up_fs
        imu_up = resample(seg_imu, M)
        kin_up = resample(seg_kin, M)

        # High-resolution cross-correlation
        cu = correlate(imu_up - imu_up.mean(), kin_up - kin_up.mean(), mode='full')
        lu = correlation_lags(len(imu_up), len(kin_up), mode='full')
        sub = lu[np.argmax(cu)] / 1000
        local_offsets.append(sub)

    local_offsets = np.array(local_offsets)
    refined_offsets = local_offsets + offsets  # Adjust by the mean offset
    print(f"Refined offsets for each peak: {refined_offsets}")