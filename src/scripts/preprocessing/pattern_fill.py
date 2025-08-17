from yatpkg.util.data import TRC, Yatsdo
import numpy as np
setattr(np, 'NaN', np.nan)  # Ensure NaN is set for numpy

participant_id = "P021"  # Replace with the actual participant ID
session_id = "Straight AR 01_Reconstructed"  # Replace with the actual session ID

# Load TRC (no automatic filling)
trc = TRC.read(filename=fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}.trc", delimiter="\t", headers=True, fill_data=False)
df  = trc.to_panda()

# Identify source and target columns
time_col = 'Time'
src_cols = ['R_ASIS_X5', 'R_ASIS_Y5', 'R_ASIS_Z5']
tgt_cols = ['L_ASIS_X6', 'L_ASIS_Y6', 'L_ASIS_Z6']
# NOTE: The columns are assumed to be named as per the original TRC file. Adjust if necessary.

# Find frames where both source and target are visible
both_ok = df.dropna(subset=src_cols + tgt_cols)
if both_ok.empty:
    raise ValueError("No valid frames found with both source and target markers.")

# First/last calibration frames
cal_idx_start = both_ok.index[0]
cal_idx_end = both_ok.index[-1]

# Times to normalise interpolation across the trial
t0 = df.loc[cal_idx_start, time_col]
t1 = df.loc[cal_idx_end, time_col]

# Start/end calibration samples and offsets
src_cal_start = df.loc[cal_idx_start, src_cols].to_numpy()
tgt_cal_start = df.loc[cal_idx_start, tgt_cols].to_numpy()
offset_start = tgt_cal_start - src_cal_start

# If only one valid frame, fall back to constant offset
if cal_idx_end == cal_idx_start or t1 == t0:
    offset_end = offset_start.copy()
else:
    src_cal_end = df.loc[cal_idx_end, src_cols].to_numpy()
    tgt_cal_end = df.loc[cal_idx_end, tgt_cols].to_numpy()
    offset_end = tgt_cal_end - src_cal_end

# Build a Yatsdo spline on the source marker
y_src = Yatsdo(
    df[[time_col] + src_cols],
    col_names=[time_col] + src_cols,
    fill_data=True    # build spline ignoring NaNs
)

# Find the times where the target is missing
missing_mask = df[tgt_cols[0]].isna()
if not missing_mask.any():
    # Nothing to fill; still write out a clean copy if you want
    trc.data = df.to_numpy()
    trc.update()
    trc.write(fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}_Reconstructed.trc")
else:
    times_m = df.loc[missing_mask, time_col].to_numpy()

    # Interpolate the source at those times
    filled_src = y_src.get_samples(times_m, assume_time_first_col=True)
    filled_src_xyz = filled_src[:, 1:4]

    # Compute per-time interpolated offsets
    if t1 == t0:
        # Single anchor: constant offset
        w = np.zeros_like(times_m, dtype=float)
    else:
        w = (times_m - t0) / (t1 - t0)
        w = np.clip(w, 0.0, 1.0)

    # Broadcast to (n_missing, 3)
    offsets_interp = (1.0 - w)[:, None] * offset_start[None, :] + w[:, None] * offset_end[None, :]

    # Reconstruct the missing target
    reconstructed = filled_src_xyz + offsets_interp  

    # Write those back into your DataFrame
    df.loc[missing_mask, tgt_cols] = reconstructed

    # Inject back into TRC and write out
    trc.data = df.to_numpy()
    trc.update()
    trc.write(fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}_Reconstructed.trc")