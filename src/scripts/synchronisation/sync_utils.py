import csv
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import correlate, correlation_lags, resample

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

def compute_coarse_offset(x, y, fs=100):
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    corr = correlate(xc, yc, mode='full')
    lags = correlation_lags(len(x), len(y), mode='full')
    return lags[np.argmax(corr)]

def refine_offset_upsample(x, y, fs=100, up_fs=1000, win_sec=0.1):
    # Coarse lag
    lag_samp = compute_coarse_offset(x, y, fs=fs)
    coarse_time = lag_samp / fs

    # Align & window
    if lag_samp >= 0:
        x_al = x[lag_samp:]
        y_al = y[:len(x_al)]
    else:
        y_al = y[-lag_samp:]
        x_al = x[:len(y_al)]
    half_win = int(win_sec * fs / 2)
    center   = len(x_al) // 2
    seg_x    = x_al[center-half_win : center+half_win]
    seg_y    = y_al[center-half_win : center+half_win]

    # Upsample both segments
    M    = int(len(seg_x) * up_fs / fs)  # samples at up_fs
    x_up = resample(seg_x, M)
    y_up = resample(seg_y, M)

    # High-resolution cross-correlation
    cu  = correlate(x_up - x_up.mean(), y_up - y_up.mean(), mode='full')
    lu  = correlation_lags(len(x_up), len(y_up), mode='full')
    sub = lu[np.argmax(cu)] / up_fs

    return coarse_time + sub