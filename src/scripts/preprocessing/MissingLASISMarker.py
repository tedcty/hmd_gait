from yatpkg.util.data import TRC
import numpy as np
setattr(np, 'NaN', np.nan)  # Ensure NaN is set for numpy
from yatpkg.math.transformation import Cloud

participant_id = "P011"  # Replace with the actual participant ID
session_id = "Reactive AR 1_Reconstructed"  # Replace with the actual session ID

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
    "RASIS": ("R_ASIS_X5", "pelvis:RASIS_X21", "pelvis:RASIS_X26"),
    "LASIS": ("L_ASIS_X6", "pelvis:LASIS_X22", "pelvis:LASIS_X27"),
    "RPSIS": ("R_PSIS_X8", "pelvis:RPSIS_X19", "pelvis:RPSIS_X24"),
    "LPSIS": ("L_PSIS_X7", "pelvis:LPSIS_X20", "pelvis:LPSIS_X25")
}

# Resolve actual X, Y, Z column names for each marker
cols = {}
for name, x_candidates in marker_cols.items():
    x_col = pick_column_name(df, *x_candidates)
    y_col = x_col.replace("X", "Y")
    z_col = x_col.replace("X", "Z")
    cols[name] = (x_col, y_col, z_col)

# Identify frames where LASIS is present vs missing
lasis_x, lasis_y, lasis_z = cols["LASIS"]
valid_idxs = df.index[df[lasis_x].notna()]
missing_idxs = df.index[df[lasis_x].isna()]

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

# Apply rigid body transformation to the missing LASIS marker

reconstructed_lasis = []

for idx in missing_idxs:
    row = df.loc[idx]
    # Extract current marker positions for RASIS, RPSIS and LPSIS
    current_markers = np.array([
        [row[cols["RASIS"][0]], row[cols["RASIS"][1]], row[cols["RASIS"][2]]],
        [row[cols["RPSIS"][0]], row[cols["RPSIS"][1]], row[cols["RPSIS"][2]]],
        [row[cols["LPSIS"][0]], row[cols["LPSIS"][1]], row[cols["LPSIS"][2]]],
    ]).T

    # Extract calibration marker positions (excluding LASIS)
    calibration_markers_current = np.array([calibration_markers["RASIS"], calibration_markers["RPSIS"], calibration_markers["LPSIS"]]).T

    # Compute transformation matrix
    transformation_matrix = Cloud.rigid_body_transform(calibration_markers_current, current_markers, rowpoints=False)

    # Apply transformation to the missing LASIS marker
    lasis_calib = np.array([calibration_markers["LASIS"] + [1]])
    lasis_new = transformation_matrix @ lasis_calib.T

    # Store the reconstructed LASIS marker
    reconstructed_lasis.append(lasis_new[:3].flatten())  # Take only x, y, z

# Add the reconstructed LASIS marker to the dataframe
for idx, (x, y, z) in zip(missing_idxs, reconstructed_lasis):
    df.loc[idx, [lasis_x, lasis_y, lasis_z]] = [x, y, z]

# Replace trc_data's data with the updated dataframe
trc_data.data = df.to_numpy()

trc_data.update()

trc_data.write(fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}_Reconstructed.trc")  # Add path to save the reconstructed TRC file