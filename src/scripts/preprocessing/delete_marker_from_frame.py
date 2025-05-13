from yatpkg.util.data import TRC
import numpy as np
from yatpkg.math.transformation import Cloud

participant_id = "P001"  # Replace with the actual participant ID
session_id = "Defined AR 3_Reconstructed"  # Replace with the actual session ID

# Read in the TRC file
trc_data = TRC.read(filename=fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}.trc", delimiter="\t", headers=True, fill_data=False)  # Add path to specific TRC file
df = trc_data.to_panda()  # Convert to pandas dataframe

# if your DataFrame index is the frame number
marker_cols = ['R_Radius_X15', 'R_Radius_Y15', 'R_Radius_Z15']

# set frames 247 and 248 to missing
df.loc[0:42, marker_cols] = np.nan

# Replace trc_data's data with the updated dataframe
trc_data.data = df.to_numpy()

trc_data.update()

trc_data.write(fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}_Reconstructed.trc")  # Add path to save the reconstructed TRC file