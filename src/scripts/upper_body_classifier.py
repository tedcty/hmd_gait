import os
import pandas as pd
import numpy as np
import dask.dataframe as dd
from enum import Enum
from joblib import dump, Parallel, delayed
from ptb.ml.ml_util import MLOperations
from ptb.util.math.filters import Butterworth
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.transformers import FeatureSelector
from tsfresh.utilities.dataframe_functions import roll_time_series
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing


class UpperBodyClassifier:

    @staticmethod
    def upper_body_imu_for_event(event, root_dir):
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
                        # Extract id: remove the suffix
                        id_str = file.replace('_imu_vec3_2_raw.csv', '')
                        imu_df['id'] = id_str
                        # Reorder columns to make 'id' the first column
                        cols = ['id'] + [col for col in imu_df.columns if col != 'id']
                        imu_df = imu_df[cols]
                        imu_data = pd.concat([imu_data, imu_df], ignore_index=True)
        return imu_data
    
    @staticmethod
    def upper_body_kinematics_for_event(event, root_dir):
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
                        # Extract id: remove the suffix
                        id_str = file.replace('.mot.csv', '')
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

        # Group by original 'id' and 'trial'
        for (eid, _), group in df.groupby(['id', 'trial']):
            group = group.reset_index(drop=True)
            n = len(group)
            # Slide the window
            for start in range(0, n - window_size + 1, stride):
                w = group.iloc[start:start + window_size].copy()
                window_id = (eid, start)
                windows.append((window_id, w))
        return windows
    
    @staticmethod
    def feature_extraction(data, y, event):
        # Merge y labels into the data
        df = data.copy()
        df['y'] = y
        
        # Pick window size based on event type from EventWindowSize
        window_size = EventWindowSize.events.value[event]
        
        features, idx, y_vals = [], [], []
        for (eid, start), w in UpperBodyClassifier.sliding_window(df, window_size):
            w = w.reset_index(drop=True)
            # Tag every row with window composite id
            w["id"] = [(eid, start)] * len(w)
            # Feature extraction
            Xw, _ = MLOperations.extract_features_from_x(w, n_jobs=3)
            features.append(Xw.iloc[0])
            # Majority vote: label = 1 if more than half of samples are 1
            label = int(w['y'].mean() > 0.5)
            idx.append((eid, start))
            y_vals.append(label)

        X_feat = pd.DataFrame(
            features,
            index=pd.MultiIndex.from_tuples(idx, names=["id", "start"])
        )

        y_feat = pd.Series(
            y_vals,
            index=X_feat.index,
            name='y'
        )

        return X_feat, y_feat
    
    @staticmethod
    def feature_selection(X, y):
        # Select statistically significant features using tsfresh's FeatureSelector transformer
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
            "X_selected": X_selected
        }
    
    @staticmethod
    def export_features_with_importance(selected, top_100, results_path):
        # Slice original importance Series to only top-100
        fi = selected["feature_importance"]
        top_list = top_100["pfx"]
        top_imp_dict = fi.loc[top_list].to_dict()
        # Export as JSON using helper
        MLOperations.export_features_json(top_imp_dict, results_path)

    @staticmethod
    def train_and_test_classifier(X_imu, X_kin, y_imu, y_kin, results_path):
        # Align and combine final feature sets of IMU and kinematics
        X_combined = pd.concat([X_imu, X_kin], axis=1)
        # Check y_imu and y_kin are equal
        assert y_imu.equals(y_kin), "IMU and kinematics labels must match."
        y = y_imu
        # Split data into training and test sets for classification
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
        # Fit Random Forest Classifier
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        # Evaluate the classification model
        y_pred = clf.predict(X_test)
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)
        # Save metrics to file
        os.makedirs(results_path, exist_ok=True)
        # Save accuracy and AUC scores
        metrics_path = os.path.join(results_path, "classifier_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"AUC: {auc_score:.4f}\n")
            f.write("\n")
        # Save classification report separately
        report_path = os.path.join(results_path, "classifier_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        # Visualise and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = os.path.join(results_path, "confusion_matrix.png")
        plt.savefig(cm_path)

        return clf
    
    @staticmethod
    def export_model(clf, results_path, event):
        model_path = os.path.join(results_path, f"{event}_rf_classifier.pkl")
        dump(clf, model_path)

    @staticmethod
    def feature_engineering(event, root_dir, datatype):
        # Combine all functions involving DataFrame creation, feature extraction and selection

        # Folder directory for specific data
        root_dir = os.path.join(root_dir, datatype)
        # Create DataFrame for upper body data for tsfresh (IMU or kinematics)
        if datatype == "IMU":
            X = UpperBodyClassifier.upper_body_imu_for_event(event, root_dir)
        elif datatype == "Kinematics":
            X = UpperBodyClassifier.upper_body_kinematics_for_event(event, root_dir)
        y = UpperBodyClassifier.y_label_column(X)
        print(f"{datatype} DataFrame and y labels created.")
        # Extract features (tsfresh) from windowed data
        X_feat, y_feat = UpperBodyClassifier.feature_extraction(X, y, event)
        print(f"{datatype} features extracted.")
        # Select features by hypothesis testing (tsfresh)
        X_sel = UpperBodyClassifier.feature_selection(X_feat, y_feat)
        print(f"{datatype} features selected by hypothesis testing.")
        # Pick the top-100 features by importance
        top100 = MLOperations.select_top_features_from_x(X_sel["feature_importance"], num_of_feat=100)
        # Export IMU top features with importances to JSON
        UpperBodyClassifier.export_features_with_importance(X_sel, top100, filepath=f"Z:/Upper Body/Results/10 Participants/{datatype}_top100_features.json")
        print(f"Found Top 100 {datatype} features by importance.")

        return X_sel["X_selected"][top100["pfx"]], y_feat
    

class UpperBodyPipeline:

    @staticmethod
    def process_event(event, root_dir, results_base):
        
        print(f"\n=== Starting EVENT: {event} ===")

        # Build a results folder per event
        safe_name = event.replace(" ", "_")
        results_dir = os.path.join(results_base, safe_name)
        os.makedirs(results_dir, exist_ok=True)

        # IMU feature engineering algorithm outputting the final feature set and associated labels
        print(f"[{event}] --> Loading IMU data ...")
        X_imu_top100, y_imu = UpperBodyClassifier.feature_engineering(event, root_dir, datatype="IMU")
        print(f"[{event}] --> IMU feature matrix: {X_imu_top100.shape}, labels: {y_imu.shape}")
        
        # Kinematics feature engineering algorithm outputting the final feature set and associated labels
        print(f"[{event}] --> Loading kinematics data ...")
        X_kin_top100, y_kin = UpperBodyClassifier.feature_engineering(event, root_dir, datatype="Kinematics")
        print(f"[{event}] --> Kinematics feature matrix: {X_kin_top100.shape}, labels: {y_kin.shape}")

        # Train and test Random Forest Classifier model using top-100 features from both IMU and kinematics
        print(f"[{event}] --> Training & testing classifier ...")
        clf = UpperBodyClassifier.train_and_test_classifier(X_imu_top100, X_kin_top100, y_imu, y_kin, results_dir)

        # Export the final model
        UpperBodyClassifier.export_model(clf, results_dir, event)
        print(f"[{event}] --> Classifier trained and saved.")

        print(f"=== Finished EVENT: {event} ===\n")
        return f"{event} done."


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
        "Place ping pong ball in cup": 250,
        "Step over cone": 160
    }


if __name__ == "__main__":
    # Set up inputs
    root_dir = "Z:/Upper Body/Events/"
    results_base = "Z:/Upper Body/Results/10 Participants"

    # Get all event names from the enum
    events = list(EventWindowSize.events.value.keys())

    cores = max(1, multiprocessing.cpu_count()//2-1)
    outputs = Parallel(n_jobs=cores)(
        delayed(UpperBodyPipeline.process_event)(ev, root_dir, results_base)
        for ev in events
    )

    for o in outputs:
        print(o)
