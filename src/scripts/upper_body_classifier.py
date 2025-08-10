import os
import pandas as pd
import numpy as np
from enum import Enum
from joblib import dump, Parallel, delayed
from ptb.ml.ml_util import MLOperations
from ptb.util.math.filters import Butterworth
from ptb.ml.tags import MLKeys
from tsfresh.transformers import FeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from event_constants import (WHOLE_EVENTS, REPETITIVE_EVENTS, EVENT_SIZES, EventWindowSize)
from trim_events import compute_whole_event_std_global, read_event_labels


class UpperBodyClassifier:

    @staticmethod
    def upper_body_imu_for_event(event, root_dir):
        # Create DataFrame for upper body IMU data based on the event that can be used for tsfresh feature extraction
        # Search for the event in each participant folder in the root directory
        for pid in os.listdir(root_dir):
            pid_path = os.path.join(root_dir, pid)
            if os.path.isdir(pid_path):
                for file in os.listdir(pid_path):
                    # Use vec3_raw IMU data and only select upper body IMU
                    if (event in file and 
                        file.endswith("_imu_vec3_2_raw.csv") and 
                        any(imu.value in file for imu in UpperBodyIMU)):
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
                        # Extract sensor id
                        for imu in UpperBodyIMU:
                            if imu.value in file:
                                id_str = f"{imu.value}_imu"
                                break
                        imu_df['id'] = id_str
                        # Prefix all signal columns with sensor id (keep time/id as is)
                        rename_map = {c: f"{id_str}_{c}" for c in imu_df.columns if c not in ['time', 'id']}
                        imu_df = imu_df.rename(columns=rename_map)
                        # Put 'id' first
                        cols = ['id'] + [col for col in imu_df.columns if col != 'id']
                        imu_df = imu_df[cols]

                        yield imu_df
    
    @staticmethod
    def upper_body_kinematics_for_event(event, root_dir):
        # Create DataFrame for upper body kinematics data based on the event that can be used for tsfresh feature extraction
        # Flatten the list of all upper body kinematics columns
        kin_cols = [col_name for kinematic in UpperBodyKinematics for col_name in kinematic.value]
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
                        id_str = 'kinematics'
                        kinematics_df['id'] = id_str
                        # Keep only time and upper body kinematics columns
                        cols_to_keep = ['time'] + [c for c in kin_cols if c in kinematics_df.columns]
                        kinematics_df = kinematics_df[['id'] + cols_to_keep].copy()

                        yield kinematics_df

    @staticmethod
    def y_label_column(df, event, events_dict=None, fs=100):
        # Initialize all zeros
        labels = pd.Series(0, index=df.index, name='y')
        
        # Get event type
        base_event = event
        for known_event in WHOLE_EVENTS | REPETITIVE_EVENTS:
            if event.startswith(known_event):
                base_event = known_event
                break

        # Get window sizes
        event_size = EVENT_SIZES[base_event]  # Already in frames
        
        if base_event in WHOLE_EVENTS and events_dict is not None:
            # Get standard deviation from cache or compute it
            std_map = compute_whole_event_std_global(events_dict, fs)
            std_size = std_map.get(base_event, 0)  # Get std or default to 0
            # Round the sum to nearest whole number of frames
            center_window = round(event_size + std_size)
            # Convert to samples for labeling
            n = len(df)
            center_start = (n - center_window) // 2
            center_end = center_start + center_window
            # Set 1's in the center window only
            mask = np.zeros(n, dtype=int)
            mask[center_start:center_end] = 1
            labels[:] = mask
            
        elif base_event in REPETITIVE_EVENTS:
            # For repetitive events, label the full duration (excluding cycle buffers)
            cycle_size = event_size  # one cycle in frames
            # Skip the buffer zones
            n = len(df)
            mask = np.zeros(n, dtype=int)
            if n > 2 * cycle_size:
                mask[cycle_size:-cycle_size] = 1  # Label everything except buffer zones
            labels[:] = mask

        return labels
    
    @staticmethod
    def sliding_window(df, event, events_dict=None, fs=100, stride=1):
        windows = []
        # Get base event name
        base_event = event
        for known_event in WHOLE_EVENTS | REPETITIVE_EVENTS:
            if event.startswith(known_event):
                base_event = known_event
                break
        
        # Get base window size
        window_size = EVENT_SIZES[base_event]  # Already in frames
        
        # Adjust window size for whole events using std
        if base_event in WHOLE_EVENTS and events_dict is not None:
            std_map = compute_whole_event_std_global(events_dict, fs)
            std_size = int(std_map.get(base_event, 0))
            window_size = round(window_size + std_size)  # Round to nearest frame
        
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
    def feature_extraction(data, y, event, n_jobs, events_dict=None):
        # Merge y labels into the data
        df = data.copy()
        df['y'] = y

        # Create lists to store window data
        windows_data = []
        window_ids = []
        y_windows = []

        for (eid, start), w in UpperBodyClassifier.sliding_window(df, event, events_dict=events_dict, stride=1):
            w = w.reset_index(drop=True)
            # Tag every row with window composite id
            w["id"] = f"{eid}|{start}"  # to all rows in w
            windows_data.append(w)
            window_ids.append(f"{eid}|{start}")
            # Majority vote: label = 1 if more than 80% of samples are 1
            label = int(w['y'].mean() > 0.8)
            y_windows.append(label)

        if not windows_data:
            # If no windows were created, return empty DataFrames
            return pd.DataFrame(), pd.Series(dtype=int)

        # Stack all windows into a single DataFrame
        combined_windows = pd.concat(windows_data, ignore_index=True)

        # Drop y before extracting features
        combined_windows = combined_windows.drop(columns=['y', 'trial'], errors='ignore')

        # Extract features using tsfresh  NOTE: Currently using MinimalFCParameters
        X_feat, _ = MLOperations.extract_features_from_x(
            combined_windows,
            fc_parameters=MLKeys.MFCParameters,
            n_jobs=n_jobs)
        
        # Ensure index type matches string ids and align order
        X_feat.index = X_feat.index.astype(str)
        valid_ids = [i for i in window_ids if i in X_feat.index]
        X_feat = X_feat.loc[valid_ids]

        # Labels aligned to the same valid_ids
        y_feat = pd.Series(
            [y for i, y in zip(window_ids, y_windows) if i in X_feat.index],
            index=valid_ids,
            name='y'
        )

        return X_feat, y_feat
    
    @staticmethod
    def feature_selection(X_train, y_train):
        # Select statistically significant features using tsfresh's FeatureSelector transformer
        selector = FeatureSelector()
        X_train_sel = selector.fit_transform(X_train, y_train)
        # Fit a RandomForestClassifier on the selected features
        rf = RandomForestClassifier()
        rf.fit(X_train_sel, y_train)
        # Get feature importances
        importances = pd.Series(rf.feature_importances_, X_train_sel.columns)
        return selector, X_train_sel, importances

    @staticmethod
    def train_and_test_classifier(X, y, results_path, event, datatype):
        # Split data into training and test sets for classification
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Feature selection on training set
        selector, X_train_sel, importances = UpperBodyClassifier.feature_selection(X_train, y_train)
        X_test_sel = selector.transform(X_test)
        print(f"{datatype} features selected by hypothesis testing.")
        
        # Pick the top-100 features by importance
        top100 = MLOperations.select_top_features_from_x(importances, num_of_feat=100)
        top100 = top100["pfx"]
        X_train_top = X_train_sel[top100]
        X_test_top = X_test_sel[top100]
        # Export IMU top features with importances to JSON
        final_feat_path = os.path.join(results_path, f"{datatype}_{event}_top100_features.json")
        MLOperations.export_features_json(importances[top100].to_dict(), final_feat_path)
        print(f"Found Top 100 {datatype} features by importance.")
        
        # Fit Random Forest Classifier
        clf = RandomForestClassifier()
        clf.fit(X_train_top, y_train)
        # Evaluate the classification model
        y_pred = clf.predict(X_test_top)
        y_proba = clf.predict_proba(X_test_top)[:, 1]  # Get probabilities for ROC AUC
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_proba)
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
        plt.close()

        return clf
    
    @staticmethod
    def export_model(clf, results_path, event, datatype):
        model_path = os.path.join(results_path, f"{datatype}_{event}_rf_classifier.pkl")
        dump(clf, model_path)

    @staticmethod
    def feature_engineering(event, root_dir, datatype):
        # Combine all functions involving DataFrame creation, feature extraction (no selection here)

        # Folder directory for intermediate features
        feat_dir = os.path.join("Z:/Upper Body/Results/10 Participants", datatype, "intermediate_features", event)
        os.makedirs(feat_dir, exist_ok=True)
        # Process each trial separately
        all_features = []
        all_labels = []
        # Create DataFrame for upper body data for tsfresh (IMU or kinematics)
        if datatype == "IMU":
            data_iter = UpperBodyClassifier.upper_body_imu_for_event(event, os.path.join(root_dir, datatype))
        elif datatype == "Kinematics":
            data_iter = UpperBodyClassifier.upper_body_kinematics_for_event(event, os.path.join(root_dir, datatype))
        # Get number of jobs for feature extraction
        n_jobs_features = max(1, (multiprocessing.cpu_count() - 2) // 2)        
        
        # Read event labels from the Excel file
        events_dict = read_event_labels("Z:/Upper Body/Event labels.xlsx")
        # Process each trial
        for i, trial_df in enumerate(data_iter):
            # Get labels for this trial
            y = UpperBodyClassifier.y_label_column(trial_df, event, events_dict=events_dict)
            # Extract features from windowed data
            X_feat, y_feat = UpperBodyClassifier.feature_extraction(trial_df, y, event, n_jobs=n_jobs_features, events_dict=events_dict)
            # Save intermediate features
            feat_file = os.path.join(feat_dir, f"trial_{i}_features.parquet")
            X_feat.to_parquet(feat_file)            
            all_features.append(X_feat)
            all_labels.append(y_feat)            
            print(f"Processed trial {i} - Shape: {X_feat.shape}")        
        # Combine all features
        X_combined = pd.concat(all_features, axis=0)
        y_combined = pd.concat(all_labels, axis=0)
        if y_combined.nunique() < 2:
            raise RuntimeError("Labelling produces a single class. Check threshold and window size settings.")

        return X_combined, y_combined
    

class UpperBodyPipeline:

    @staticmethod
    def process_event(event, root_dir, datatype, results_base):
        
        print(f"\n=== Starting EVENT: {event} ===")

        # Build a results folder per event
        safe_name = event.replace(" ", "_")
        results_dir = os.path.join(results_base, safe_name)
        os.makedirs(results_dir, exist_ok=True)

        # IMU feature engineering algorithm outputting the extracted features and labels
        print(f"[{event}] --> Loading {datatype} data ...")
        X, y = UpperBodyClassifier.feature_engineering(event, root_dir, datatype)
        print(f"[{event}] --> {datatype} feature matrix: {X.shape}, labels: {y.shape}")

        # Train and test Random Forest Classifier model using top-100 features
        print(f"[{event}] --> Training & testing {datatype} classifier ...")
        clf = UpperBodyClassifier.train_and_test_classifier(X, y, results_dir, event, datatype)

        # Export the final model
        UpperBodyClassifier.export_model(clf, results_dir, event, datatype)
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
    # left_forearm = "LeftForeArm"
    # right_forearm = "RightForeArm"
    # left_hand = "LeftHand"
    # right_hand = "RightHand"
    # left_shoulder = "LeftShoulder"
    # right_shoulder = "RightShoulder"
    # left_upper_arm = "LeftUpperArm"
    # right_upper_arm = "RightUpperArm"
    # pelvis = "Pelvis"
    # sternum = "T8"


if __name__ == "__main__":
    # Set up inputs
    root_dir = "Z:/Upper Body/Events/"
    datatypes = ["IMU", "Kinematics"]

    # Get all event names from the enum
    events = list(EventWindowSize.events.value.keys())

    # Reserve 2 cores for system stability, use remaining cores for processing
    total_cores = multiprocessing.cpu_count()
    cores = max(1, total_cores - 2)  # Use all cores except 2 for system

    # Set parallel jobs for feature extraction
    n_jobs_features = max(1, cores // 2)  # Use half of available cores for feature extraction
    
    print(f"System has {total_cores} cores. Using {cores} cores for processing, {n_jobs_features} for feature extraction.")
    
    # Create status file to indicate processing has started
    status_file = "Z:/Upper Body/Results/10 Participants/processing_status.txt"
    with open(status_file, 'w') as f:
        f.write("Processing started: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")) 
    
    # Process one datatype at a time to manage memory
    try:
        for datatype in datatypes:
            print(f"\nProcessing {datatype} data...")
            try:
                outputs = Parallel(n_jobs=cores, prefer="threads")(
                    delayed(UpperBodyPipeline.process_event)(
                        ev, 
                        root_dir, 
                        datatype,
                        f"Z:/Upper Body/Results/10 Participants/{datatype}"
                    )
                    for ev in events
                )
                
                # Force garbage collection between datatypes
                import gc
                gc.collect()
                
                for o in outputs:
                    print(o)
                    # Update status file after each datatype
                    with open(status_file, 'a') as f:
                        f.write(f"\nCompleted {datatype}: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
                    
            except MemoryError:
                print(f"Memory error encountered. Reducing parallel processing...")
                # Fallback to fewer cores if memory error occurs
                cores_fallback = max(1, cores // 2)
                outputs = Parallel(n_jobs=cores_fallback, prefer="threads")(
                    delayed(UpperBodyPipeline.process_event)(
                        ev, 
                        root_dir, 
                        datatype,
                        f"Z:/Upper Body/Results/10 Participants/{datatype}"
                    )
                    for ev in events
                )
                
                for o in outputs:
                    print(o)

        # Mark processing as complete
        with open(status_file, 'a') as f:
            f.write("\nProcessing completed successfully: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

    except Exception as e:
        # Log any errors that occurred
        with open(status_file, 'a') as f:
            f.write(f"\nError occurred: {str(e)}\n" + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        raise  # Re-raise the exception after logging
