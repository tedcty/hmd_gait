from ptb.util.data import TRC
import numpy as np
setattr(np, 'NaN', np.nan)  # Ensure NaN is set for numpy
from ptb.util.math.transformation import Cloud


if __name__ == "__main__":

    # Configuration - Change these parameters as needed
    participant_id = "P026"  # Replace with the actual participant ID
    session_id = "Obstacle AR 01_Reconstructed"  # Replace with the actual session ID
    target_marker = "LPSIS"  # Change to "RASIS", "LASIS", "RPSIS", or "LPSIS"

    # Read in the TRC file
    trc_data = TRC.read(filename=fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}.trc", delimiter="\t", headers=True, fill_data=False)  # Add path to specific TRC file
    df = trc_data.to_panda()  # Convert to pandas dataframe

    # Helper to pick between alternative column names
    def pick_column_name(df, *candidates):
        """
        Return the first column name that exists in the DataFrame.
        """
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(f"None of the columns {candidates} exist in the DataFrame.")

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

    # Validate target marker
    if target_marker not in marker_patterns:
        raise ValueError(f"Target marker '{target_marker}' must be one of: {list(marker_patterns.keys())}")

    # Resolve actual X, Y, Z column names for each marker
    cols = {}
    for name, patterns in marker_patterns.items():
        cols[name] = find_marker_columns(df, patterns)

    # Get the three reference markers (excluding the target marker)
    all_markers = ["RASIS", "LASIS", "RPSIS", "LPSIS"]
    reference_markers = [marker for marker in all_markers if marker != target_marker]

    print(f"Reconstructing missing {target_marker} marker using reference markers: {reference_markers}")

    # Identify frames where target marker is present vs missing
    target_x, target_y, target_z = cols[target_marker]
    valid_idxs = df.index[df[target_x].notna()]
    missing_idxs = df.index[df[target_x].isna()]

    print(f"Found {len(missing_idxs)} missing frames for {target_marker}")

    if len(missing_idxs) == 0:
        print(f"No missing data found for {target_marker}. Exiting.")
        exit()

    # If the missing frames are before the first valid, use first valid; otherwise use last valid
    if missing_idxs.min() < valid_idxs.min():
        calib_idx = valid_idxs.min()
    else:
        calib_idx = valid_idxs.max()

    calibration_frame = df.loc[calib_idx]

    # Extract calibration marker positions
    calibration_markers = {
        name: [
            calibration_frame[x], 
            calibration_frame[y], 
            calibration_frame[z]
        ]
        for name, (x, y, z) in cols.items()
    }

    # Apply rigid body transformation to the missing target marker
    reconstructed_positions = []

    for idx in missing_idxs:
        row = df.loc[idx]
        
        # Extract current marker positions for the three reference markers
        current_markers = np.array([
            [row[cols[ref_marker][0]], row[cols[ref_marker][1]], row[cols[ref_marker][2]]]
            for ref_marker in reference_markers
        ]).T

        # Extract calibration marker positions (excluding target marker)
        calibration_markers_current = np.array([
            calibration_markers[ref_marker] for ref_marker in reference_markers
        ]).T

        # Compute transformation matrix
        transformation_matrix = Cloud.rigid_body_transform(calibration_markers_current, current_markers, rowpoints=False)

        # Apply transformation to the missing target marker
        target_calib = np.array([calibration_markers[target_marker] + [1]])
        target_new = transformation_matrix @ target_calib.T

        # Store the reconstructed target marker
        reconstructed_positions.append(target_new[:3].flatten())  # Take only x, y, z

    # Add the reconstructed target marker to the dataframe
    for idx, (x, y, z) in zip(missing_idxs, reconstructed_positions):
        df.loc[idx, [target_x, target_y, target_z]] = [x, y, z]

    print(f"Successfully reconstructed {len(missing_idxs)} frames for {target_marker}")

    # Replace trc_data's data with the updated dataframe
    trc_data.data = df.to_numpy()
    trc_data.update()

    # Save the reconstructed TRC file
    output_filename = fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}_Reconstructed.trc"
    trc_data.write(output_filename)
    print(f"Saved reconstructed data to: {output_filename}")