from yatpkg.util.data import TRC, Yatsdo
import numpy as np
setattr(np, 'NaN', np.nan)  # Ensure NaN is set for numpy

if __name__ == "__main__":

    # Configuration - Change these parameters as needed
    participant_id = "P021"  # Replace with the actual participant ID
    session_id = "Straight AR 01_Reconstructed"  # Replace with the actual session ID
    source_marker = "RASIS"  # Source marker to use for reconstruction
    target_marker = "LASIS"  # Target marker to reconstruct

    # Load TRC (no automatic filling)
    trc = TRC.read(filename=fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}.trc", delimiter="\t", headers=True, fill_data=False)
    df  = trc.to_panda()

    # Helper to find columns containing marker names
    def find_marker_columns(df, marker_patterns):
        """
        Find X, Y, Z columns for a marker based on patterns in column names.
        Returns tuple of (x_col, y_col, z_col) or raises ValueError if not found.
        """
        x_col = None
        for col in df.columns:
            if any(pattern.upper() in col.upper() for pattern in marker_patterns) and 'X' in col:
                x_col = col
                break
        
        if x_col is None:
            raise ValueError(f"No X column found for patterns: {marker_patterns}")
        
        # Generate Y and Z column names based on X column
        y_col = x_col.replace('X', 'Y')
        z_col = x_col.replace('X', 'Z')
        
        # Verify Y and Z columns exist
        if y_col not in df.columns or z_col not in df.columns:
            raise ValueError(f"Y or Z columns not found for marker with X column: {x_col}")
        
        return x_col, y_col, z_col

    # Define marker search patterns
    marker_patterns = {
        "RASIS": ["RASIS", "R_ASIS"],
        "LASIS": ["LASIS", "L_ASIS"],
        "RPSIS": ["RPSIS", "R_PSIS"],
        "LPSIS": ["LPSIS", "L_PSIS"]
    }

    # Validate markers
    if source_marker not in marker_patterns:
        raise ValueError(f"Source marker '{source_marker}' must be one of: {list(marker_patterns.keys())}")
    if target_marker not in marker_patterns:
        raise ValueError(f"Target marker '{target_marker}' must be one of: {list(marker_patterns.keys())}")

    # Find source and target columns dynamically
    src_x, src_y, src_z = find_marker_columns(df, marker_patterns[source_marker])
    tgt_x, tgt_y, tgt_z = find_marker_columns(df, marker_patterns[target_marker])

    # Identify source and target columns
    time_col = 'Time'
    src_cols = [src_x, src_y, src_z]
    tgt_cols = [tgt_x, tgt_y, tgt_z]

    print(f"Using source marker {source_marker}: {src_cols}")
    print(f"Reconstructing target marker {target_marker}: {tgt_cols}")

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