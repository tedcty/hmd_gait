# Change header names in TRC files to match the expected format for OpenSim models during Inverse Kinematics (IK) analysis.

import pandas as pd

participant_id = "P010"  # Replace with the actual participant ID
session_id = "Obstacles VR 1"  # Replace with the actual session ID

infile = fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}.trc"
outfile = fr"Z:\Upper Body\Mocap\{participant_id}\{session_id}.trc"

# Rename mapping
rename_map = {
    'torso:Head': 'Head',
    'torso:Sternum': 'Sternum',
    'torso:R_Acromion': 'R_Acromion',
    'torso:L_Acromion': 'L_Acromion',
    'r_leg:R_FibHead': 'R_FibHead',
    'r_leg:R_LatKnee': 'R_LatKnee',
    'r_leg:R_MedKnee': 'R_MedKnee',
    'r_leg:R_MidShank': 'R_MidShank',
    'r_foot:R_Heel': 'R_Heel',
    'r_foot:R_MT5': 'R_MT5',
    'r_foot:R_MT2': 'R_MT2',
    'r_foot:R_DP1': 'R_DP1',
    'r_foot:R_MedAnkle': 'R_MedAnkle',
    'r_foot:R_LatAnkle': 'R_LatAnkle',
    'r_arm:R_Lat_HumEpicondyle': 'R_Lat_HumEpicondyle',
    'r_arm:R_Med_HumEpicondyle': 'R_Med_HumEpicondyle',
    'r_arm:R_Radius': 'R_Radius',
    'r_arm:R_Ulna': 'R_Ulna',
    'pelvis:RPSIS': 'R_PSIS',
    'pelvis:LPSIS': 'L_PSIS',
    'pelvis:RASIS': 'R_ASIS',
    'pelvis:LASIS': 'L_ASIS',
    'l_leg:L_FibHead': 'L_FibHead',
    'l_leg:L_LatKnee': 'L_LatKnee',
    'l_leg:L_MatKnee': 'L_MedKnee',
    'l_leg:L_MidShank': 'L_MidShank',
    'l_foot:L_Heel': 'L_Heel',
    'l_foot:L_MT5': 'L_MT5',
    'l_foot:L_MT2': 'L_MT2',
    'l_foot:L_DP1': 'L_DP1',
    'l_foot:L_LatAnkle': 'L_LatAnkle',
    'l_foot:L_LMedAnkle': 'L_MedAnkle',
    'l_arm:L_Lat_HumEpicondyle': 'L_Lat_HumEpicondyle',
    'l_arm:L_Med_HumEpicondyle': 'L_Med_HumEpicondyle',
    'l_arm:L_Radius': 'L_Radius',
    'l_arm:L_Ulna': 'L_Ulna'
}

# Read & update the header
with open(infile, 'r') as f:
    header_lines = [next(f) for _ in range(5)]

# header_lines[3] is the "Frame#  Time  Marker1  ... MarkerN" row
cols = header_lines[3].rstrip('\n').split('\t')
cols_renamed = [rename_map.get(c, c) for c in cols]
header_lines[3] = '\t'.join(cols_renamed) + '\n'

# Load the rest of the data as plain text
df = pd.read_csv(infile,
                 sep='\t',
                 skiprows=5,
                 header=None,
                 dtype=str,    # preserve formatting
                 na_filter=False)

# Write out the new file
with open(outfile, 'w', newline='') as f:
    f.writelines(header_lines)
    df.to_csv(f, sep='\t', header=False, index=False)