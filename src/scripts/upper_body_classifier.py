import os
import pandas as pd
import numpy as np
from enum import Enum
import re
from ptb.ml.ml_util import MLOperations
from ptb.util.math.filters import Butterworth
from tsfresh.transformers import FeatureSelector
from sklearn.ensemble import RandomForestClassifier


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
                        # Apply 4th order Butterworth low-pass filter (6 Hz cut-off frequency) to all columns except 'time' and 'id'
                        cols_to_filter = [col for col in imu_df.columns if col not in ['time', 'id']]
                        for col in cols_to_filter:
                            imu_df[col] = Butterworth.butter_low_filter(imu_df[col], cut=6, fs=100, order=4)
                        # Append to the main DataFrame with filename under new 'id' column
                        # Extract id: remove everything before and including '_Normal_', '_AR_', or '_VR_' and the suffix
                        id_str = file.replace('_imu_vec3_2_raw.csv', '')
                        id_str = re.sub(r'^.*?_(Normal|AR|VR)_', '', id_str)
                        imu_df['id'] = id_str
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
                        # Apply 4th order Butterworth low-pass filter (6 Hz cut-off frequency) to all columns except 'time' and 'id'
                        cols_to_filter = [col for col in kinematics_df.columns if col not in ['time', 'id']]
                        for col in cols_to_filter:
                            kinematics_df[col] = Butterworth.butter_low_filter(kinematics_df[col], cut=6, fs=100, order=4)
                        # Extract id: remove everything before and including '_Normal_', '_AR_', or '_VR_' and the suffix
                        id_str = file.replace('.mot.csv', '')
                        id_str = re.sub(r'^.*?_(Normal|AR|VR)_', '', id_str)
                        kinematics_df['id'] = id_str
                        # Keep only time and upper body kinematics columns
                        cols_to_keep = ['time'] + [c for c in kin_cols if c in kinematics_df.columns]
                        kinematics_df = kinematics_df[cols_to_keep].copy()
                        # Reorder columns to make 'id' the first column
                        kinematics_df['id'] = id_str
                        kinematics_df = kinematics_df[['id'] + cols_to_keep]

                        all_dfs.append(kinematics_df)
        return pd.concat(all_dfs, ignore_index=True)    

    @staticmethod
    def y_label_column(df, start_buffer=0.2, end_buffer=0.2, fs=100):
        # Each instance where time == 0 is a start of a new trial
        trial_breaks = df['time'].eq(0).cumsum()
        # Initialise all zeros
        labels = pd.Series(0, index=df.index, name='y')
        # Group each (id, trial) separately
        groups = df.groupby([df['id'], trial_breaks]).groups
        for (_, _), idx in groups.items():
            n = len(idx)
            # How many samples to skip at start and end
            skip_start = int(np.round(start_buffer * fs))
            skip_end   = int(np.round(end_buffer   * fs))

            # Make a zero array, then set 1’s in the core window
            mask = np.zeros(n, dtype=int)
            mask[skip_start : n - skip_end] = 1

            # Assign back into the labels Series
            labels.loc[idx] = mask

        return labels
    
    @staticmethod
    def sliding_window(df, window_size, stride=1):
        windows = []
        # Assign a trial index within each id
        df = df.copy()
        df['trial'] = df['time'].eq(0).cumsum()

        # Group by 'id' and 'trial'
        for (id_val, trial_idx), group in df.groupby(['id', 'trial']):
            group = group.reset_index(drop=True)
            n = len(group)
            # For every possible start
            for start in range(0, n - window_size + 1, stride):
                w = group.iloc[start:start + window_size].copy()
                windows.append(w)
        return windows
    
    @staticmethod
    def feature_extraction(data, y):
        # Merge y labels into the data
        data['y'] = y
        # Window combined data
        windows = UpperBodyClassifier.sliding_window(data, window_size=100, stride=1)
        # Extract features using tsfresh on each window and at the same time do a majority-vote on y labels
        feature_dfs = []
        y_window = {}
        for window in windows:
            wid = window['id'].iat[0]
            # Majority vote on y labels
            vals = window['y'].to_numpy()
            # If more than half the samples are 1, then the window is labelled as 1
            y_window[wid] = int(vals.mean() > 0.5)
            # Drop y labels from the window
            X = window.drop(columns=['y'])
            # Extract features using tsfresh
            features = MLOperations.extract_features_from_x(X)  # NOTE: Figure out whether 'id' column is right
            feature_dfs.append(features)
        # Concatenate into a single DataFrame
        X_feat = pd.concat(feature_dfs)
        y_feat = pd.Series(y_window, name='y')

        return X_feat, y_feat
    
    @staticmethod
    def feature_selection(X, y):
        # Select features using tsfresh's FeatureSelector transformer
        selector = FeatureSelector()
        X_selected = selector.fit_transform(X, y)
        # Fit a RandomForestClassifier on the selected features
        clf = RandomForestClassifier()
        clf.fit(X_selected, y)
        # Get feature importances
        fi = pd.Series(clf.feature_importances_, X_selected.columns)
        return {
            "model": clf,
            "feature": X_selected.columns.tolist(),
            "feature_importance": fi,
            "fc_selected": X_selected
        }


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


class EventWindowSize(Enum):
    events = {
        "Straight walk": 110,
        "Stair up": 130,
        "Stair down": 110,
        "Pick up basketball": 150,
        "Dribbling basketball": 80,
        "Put down basketball": 180,
        "Put ping pong ball in cup": 250,
        "Step over cone": 160
    }


if __name__ == "__main__":

    ## IMU data
    # Create DataFrame for upper body IMU data for tsfresh
    imu_data = UpperBodyClassifier.upper_body_imu_for_event(event="Dribbling basketball")
    y_imu = UpperBodyClassifier.y_label_column(imu_data)
    print("IMU DataFrame and y labels created.")
    # Extract features (tsfresh) from windowed data
    X_imu, y_imu = UpperBodyClassifier.feature_extraction(imu_data, y_imu)
    print("IMU features extracted.")
    # Select features by hypothesis testing (tsfresh)
    imu_fs = UpperBodyClassifier.feature_selection(X_imu, y_imu)

    ## Kinematics data
    # Create DataFrame for upper body kinematics data for tsfresh
    kin_data = UpperBodyClassifier.upper_body_kinematics_for_event(event="Dribbling basketball")
    y_kin = UpperBodyClassifier.y_label_column(kin_data)
    print("Kinematics DataFrame and y labels created.")
    # Extract features from windowed data
    X_kin, y_kin = UpperBodyClassifier.feature_extraction(kin_data, y_kin)
    print("Kinematics features extracted.")
    # Select features by hypothesis testing (tsfresh)
    kin_fs = UpperBodyClassifier.feature_selection(X_kin, y_kin)
