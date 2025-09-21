from yatpkg.util.data import TRC
import numpy as np
setattr(np, 'NaN', np.nan)  # Ensure NaN is set for numpy
from yatpkg.math.transformation import Cloud

participant_id = "P026"  # Replace with the actual participant ID
session_id = "Define Normal 03_Reconstructed"  # Replace with the actual session ID

# Read in the TRC file
trc_data = TRC.read(filename=fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}.trc", delimiter="\t", headers=True, fill_data=False)  # Add path to specific TRC file
df = trc_data.to_panda()  # Convert to pandas dataframe

# # Export to CSV
# output_csv_path = r"C:\Users\vibha\OneDrive\Documents\Masters Project\Data Processing\Reactive Normal 01.csv"
# df.to_csv(output_csv_path, index=False)

# Helper to pick between alternative column names
def pick_column_name(df, *candidates):
    """
    Return the first column name that exists in the DataFrame.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of the columns {candidates} exist in the DataFrame.")

# Define possible naming conventions for each marker's X coordinate
marker_cols = {
    "RASIS": ("R_ASIS_X5", "R_ASIS_X7", "pelvis:RASIS_X21", "pelvis:RASIS_X26"),
    "LASIS": ("L_ASIS_X6", "L_ASIS_X8", "pelvis:LASIS_X22", "pelvis:LASIS_X27"),
    "RPSIS": ("R_PSIS_X8", "R_PSIS_X5", "pelvis:RPSIS_X19", "pelvis:RPSIS_X24"),
    "LPSIS": ("L_PSIS_X7", "L_PSIS_X6", "pelvis:LPSIS_X20", "pelvis:LPSIS_X25")
}

# Resolve actual X, Y, Z column names for each marker
cols = {}
for name, x_candidates in marker_cols.items():
    x_col = pick_column_name(df, *x_candidates)
    y_col = x_col.replace("X", "Y")
    z_col = x_col.replace("X", "Z")
    cols[name] = (x_col, y_col, z_col)

# Identify frames where RPSIS is present vs missing
rpsis_x, rpsis_y, rpsis_z = cols["RPSIS"]
valid_idxs = df.index[df[rpsis_x].notna()]
missing_idxs = df.index[df[rpsis_x].isna()]

# If the missing frames are before the first valid, use first valid; otherwise use last valid
if missing_idxs.min() < valid_idxs.min():
    calib_idx = valid_idxs.min()
else:
    calib_idx = valid_idxs.max()

calibration_frame = df.loc[calib_idx]

# Extract callibration marker positions
calibration_markers = {
    name: [
        calibration_frame[x], 
        calibration_frame[y], 
        calibration_frame[z]
    ]
    for name, (x, y, z) in cols.items()
}

# Apply rigid body transformation to the missing RPSIS marker

reconstructed_rpsis = []

for idx in missing_idxs:
    row = df.loc[idx]
    # Extract current marker positions for RASIS, LPSIS and LASIS
    current_markers = np.array([
        [row[cols["RASIS"][0]], row[cols["RASIS"][1]], row[cols["RASIS"][2]]],
        [row[cols["LPSIS"][0]], row[cols["LPSIS"][1]], row[cols["LPSIS"][2]]],
        [row[cols["LASIS"][0]], row[cols["LASIS"][1]], row[cols["LASIS"][2]]],
    ]).T

    # Extract calibration marker positions (excluding RPSIS)
    calibration_markers_current = np.array([calibration_markers["RASIS"], calibration_markers["LPSIS"], calibration_markers["LASIS"]]).T

    # Compute transformation matrix
    transformation_matrix = Cloud.rigid_body_transform(calibration_markers_current, current_markers, rowpoints=False)

    # Apply transformation to the missing RPSIS marker
    rpsis_calib = np.array([calibration_markers["RPSIS"] + [1]])
    rpsis_new = transformation_matrix @ rpsis_calib.T

    # Store the reconstructed RPSIS marker
    reconstructed_rpsis.append(rpsis_new[:3].flatten())  # Take only x, y, z

# Add the reconstructed RPSIS marker to the dataframe
for idx, (x, y, z) in zip(missing_idxs, reconstructed_rpsis):
    df.loc[idx, [rpsis_x, rpsis_y, rpsis_z]] = [x, y, z]

# Replace trc_data's data with the updated dataframe
trc_data.data = df.to_numpy()

trc_data.update()

trc_data.write(fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}_Reconstructed.trc")  # Add path to save the reconstructed TRC file