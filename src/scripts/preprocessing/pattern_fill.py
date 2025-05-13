from yatpkg.util.data import TRC, Yatsdo
import numpy as np

participant_id = "P017"  # Replace with the actual participant ID
session_id = "Reactive AR 2_Reconstructed"  # Replace with the actual session ID

# Load TRC (no automatic filling)
trc = TRC.read(filename=fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}.trc", delimiter="\t", headers=True, fill_data=False)
df  = trc.to_panda()

# Identify source and target columns
time_col = 'Time'
src_cols = ['R_PSIS_X8', 'R_PSIS_Y8', 'R_PSIS_Z8']
tgt_cols = ['R_ASIS_X5', 'R_ASIS_Y5', 'R_ASIS_Z5']
# Note: The columns are assumed to be named as per the original TRC file. Adjust if necessary.

# Pick a calibration frame where both markers are present
cal_idx = df.dropna(subset=src_cols + tgt_cols).index[0]
src_cal = df.loc[cal_idx, src_cols].to_numpy()
tgt_cal = df.loc[cal_idx, tgt_cols].to_numpy()

# Compute the fixed offset vector
offset = tgt_cal - src_cal   # shape (3,)

# Build a Yatsdo spline on the source marker
y_src = Yatsdo(
    df[[time_col] + src_cols],
    col_names=[time_col] + src_cols,
    fill_data=True    # build spline ignoring NaNs
)

# Find the times where the target is missing
missing = df[tgt_cols[0]].isna()
times_m = df.loc[missing, time_col].to_list()

# Interpolate the source at those times
filled_src = y_src.get_samples(times_m, assume_time_first_col=True)  # ndarray shape (n_missing,4)
# ignore column 0 (time)
filled_src_xyz = filled_src[:,1:4]  # shape (n_missing,3)

# Reconstruct the missing target
reconstructed = filled_src_xyz + offset  # broadcast shape (n_missing,3)

# Write those back into your DataFrame
df.loc[missing, tgt_cols] = reconstructed

# Inject back into TRC and write out
trc.data = df.to_numpy()
trc.update()
trc.write(fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}_Reconstructed.trc")
