import os
import pandas as pd
from enum import Enum
from ptb.ml.ml_util import MLOperations
from ptb.util.math.filters import Butterworth


class UpperBodyClassifier:

    @staticmethod
    def upper_body_imu_for_event(event, root_dir="Z:/Upper Body/Events/IMU"):
        # Create DataFrame for upper body IMU data based on the event that can be used for tsfresh feature extraction
        # Search for the event in each participant folder in the root directory
        imu_data = pd.DataFrame()
        for pid in os.listdir(root_dir):
            pid_path = os.path.join(root_dir, pid)
            if os.path.isdir(pid_path):
                for file in os.listdir(pid_path):
                    # Use vec3_raw IMU data and only select upper body IMU
                    if (
                        event in file and 
                        file.endswith("_imu_vec3_2_raw.csv") and 
                        any(imu.value in file for imu in UpperBodyIMU)
                    ):
                        file_path = os.path.join(pid_path, file)
                        # Read selected IMU file
                        imu_df = pd.read_csv(file_path)
                        # Shift time so the event always starts at 0
                        imu_df['time'] = imu_df['time'] - imu_df['time'].min()
                        # Append to the main DataFrame with filename under new 'id' column
                        imu_df['id'] = file.replace('_imu_vec3_2_raw.csv', '')
                        # Reorder columns to make 'id' the first column
                        cols = ['id'] + [col for col in imu_df.columns if col != 'id']
                        imu_df = imu_df[cols]
                        imu_data = pd.concat([imu_data, imu_df], ignore_index=True)
        return imu_data
    
    @staticmethod
    def upper_body_kinematics_for_event(event, root_dir="Z:/Upper Body/Events/Kinematics"):
        # Create DataFrame for upper body kinematics data based on the event that can be used for tsfresh feature extraction
        # Flatten the list of all upper body kinematics columns
        kin_cols = [col_name for kinematic in UpperBodyKinematics for col_name in kinematic.value]
        all_dfs = []
        # Search for the event in each participant folder in the root directory
        for pid in os.listdir(root_dir):
            pid_path = os.path.join(root_dir, pid)
            if os.path.isdir(pid_path):
                for file in os.listdir(pid_path):
                    if event in file and file.endswith(".mot.csv"):
                        file_path = os.path.join(pid_path, file)
                        # Read selected kinematics file
                        kinematics_df = pd.read_csv(file_path)
                        # Shift time so the event always starts at 0
                        kinematics_df['time'] = kinematics_df['time'] - kinematics_df['time'].min()
                        # Append to the main DataFrame with filename under new 'id' column
                        kinematics_df['id'] = file.replace('.mot.csv', '')
                        # Keep only time and upper body kinematics columns
                        cols_to_keep = ['time'] + [c for c in kin_cols if c in kinematics_df.columns]
                        kinematics_df = kinematics_df[cols_to_keep].copy()
                        # Reorder columns to make 'id' the first column
                        kinematics_df['id'] = file.replace('.mot.csv', '')
                        kinematics_df = kinematics_df[['id'] + cols_to_keep]

                        all_dfs.append(kinematics_df)
        return pd.concat(all_dfs, ignore_index=True)


class UpperBodyKinematics(Enum):
    pelvis = ["pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz"]
    lumbar = ["lumbar_extension", "lumbar_bending", "lumbar_rotation"]
    left_arm = ["arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l"]
    right_arm = ["arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r"]


class UpperBodyIMU(Enum):
    head = "Head"
    left_forearm = "LeftForeArm"
    right_forearm = "RightForeArm"
    left_hand = "LeftHand"
    right_hand = "RightHand"
    left_shoulder = "LeftShoulder"
    right_shoulder = "RightShoulder"
    left_upper_arm = "LeftUpperArm"
    right_upper_arm = "RightUpperArm"
    pelvis = "Pelvis"
    sternum = "T8"


if __name__ == "__main__":
    imu_data = UpperBodyClassifier.upper_body_imu_for_event(event="Dribbling basketball")
    print(imu_data.head())
    # Export to CSV
    imu_data.to_csv("Z:/Upper Body/upper_body_imu_data.csv", index=False)  # Just to check