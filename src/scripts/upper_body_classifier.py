import os
import pandas as pd
import numpy as np
from enum import Enum
import json
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
                    if (event in file
                        and "imu_vec3" in file
                        and file.endswith("_raw.csv")
                        and any(imu.value in file for imu in UpperBodyIMU)):
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
                        # Add participant and trial name
                        imu_df['participant'] = pid
                        # Remove either "_imu_vec3_2_raw.csv" or "_imu_vec3_raw.csv" as appropriate
                        if file.endswith("_imu_vec3_2_raw.csv"):
                            trial_name = file.removesuffix("_imu_vec3_2_raw.csv")
                        elif file.endswith("_imu_vec3_raw.csv"):
                            trial_name = file.removesuffix("_imu_vec3_raw.csv")
                        else:
                            trial_name = file  # fallback, should not happen
                        imu_df['trial_name'] = trial_name
                        # Put 'id' first
                        cols = ['id', 'participant', 'trial_name'] + [col for col in imu_df.columns if col not in ['id', 'participant', 'trial_name']]
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
                        kinematics_df['id'] = 'kinematics'
                        # Add participant and trial name
                        kinematics_df['participant'] = pid
                        trial_name = file.removesuffix(".mot.csv")
                        kinematics_df['trial_name'] = trial_name
                        # Keep only time and upper body kinematics columns
                        cols_to_keep = ['time'] + [c for c in kin_cols if c in kinematics_df.columns]
                        kinematics_df = kinematics_df[['id', 'participant', 'trial_name'] + cols_to_keep].copy()

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
        combined_windows = combined_windows.drop(columns=['y', 'trial', 'participant', 'trial_name'], errors='ignore')

        # Extract features using tsfresh  NOTE: Currently using ComprehensiveFCParameters
        X_feat, _ = MLOperations.extract_features_from_x(
            combined_windows,
            fc_parameters=MLKeys.CFCParameters,
            n_jobs=n_jobs
        )

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
    def extract_and_save_features(event, root_dir, datatype, out_root, n_jobs_features, events_dict):
        # Choose data type
        if datatype == "IMU":
            data_iter = UpperBodyClassifier.upper_body_imu_for_event(event, os.path.join(root_dir, datatype))
        elif datatype == "Kinematics":
            data_iter = UpperBodyClassifier.upper_body_kinematics_for_event(event, os.path.join(root_dir, datatype))
        else:
            raise ValueError("data type must be 'IMU' or 'Kinematics'")

        for trial_df in data_iter:
            pid = trial_df['participant'].iloc[0]
            sensor = trial_df['id'].iloc[0]
            trial_name = trial_df['trial_name'].iloc[0]

            y = UpperBodyClassifier.y_label_column(trial_df, event, events_dict=events_dict)
            X_feat, y_feat = UpperBodyClassifier.feature_extraction(
                trial_df, y, event, n_jobs=n_jobs_features, events_dict=events_dict
                )
            if X_feat.empty:
                print(f"Skip {pid}/{sensor}/{trial_name} - No windows")
                continue

            dst_dir = os.path.join(out_root, datatype, event.replace(" ", "_"), pid, sensor)
            os.makedirs(dst_dir, exist_ok=True)
            X_path = os.path.join(dst_dir, f"{trial_name}_X.csv")
            y_path = os.path.join(dst_dir, f"{trial_name}_y.csv")

            X_feat.to_csv(X_path, index=True, index_label="window_id")
            y_feat.to_frame("y").to_csv(y_path, index=True, index_label="window_id")

            print(f"Saved: {X_path}   ({X_feat.shape[0]} windows, {X_feat.shape[1]} features)")

    @staticmethod
    def load_features_for_participants(out_root, datatype, event, participants):
        base = os.path.join(out_root, datatype, event.replace(" ", "_"))
        X_list, y_list = [], []
        all_cols_set = set()
        
        # First pass: collect all DataFrames and determine all columns
        for pid in participants:
            pid_path = os.path.join(base, pid)
            if not os.path.isdir(pid_path):
                continue
            for sensor in os.listdir(pid_path):
                sensor_path = os.path.join(pid_path, sensor)
                if not os.path.isdir(sensor_path):
                    continue
                for fname in os.listdir(sensor_path):
                    if fname.endswith("_X.csv"):
                        stem = fname[:-6]  # Remove _X.csv
                        Xp = os.path.join(sensor_path, f"{stem}_X.csv")
                        Yp = os.path.join(sensor_path, f"{stem}_y.csv")
                        if not os.path.exists(Yp):
                            continue
                        X_df = pd.read_csv(Xp, index_col="window_id")
                        y_df = pd.read_csv(Yp, index_col="window_id")
                        y_sr = y_df.iloc[:, 0].astype(np.int8)
                        common_idx = X_df.index.intersection(y_sr.index)
                        if len(common_idx) == 0:
                            continue
                        X_df = X_df.loc[common_idx]
                        y_sr = y_sr.loc[common_idx]
                        
                        # Convert numeric columns to float32 immediately
                        numeric_cols = X_df.select_dtypes(include=[np.number]).columns
                        X_df[numeric_cols] = X_df[numeric_cols].astype(np.float32)
                        
                        all_cols_set.update(X_df.columns)
                        X_list.append(X_df)
                        y_list.append(y_sr)
        
        if not X_list:
            raise RuntimeError("No CSV features found for given participants.")
        
        all_cols = sorted(all_cols_set)
        
        # Reindex all DataFrames to have consistent columns
        X_list_reindexed = [df.reindex(columns=all_cols, fill_value=np.float32(0.0)) for df in X_list]
        
        # Ensure all data is float32 before concatenation
        for i, df in enumerate(X_list_reindexed):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].astype(np.float32)
            X_list_reindexed[i] = df
        
        # Concatenate all DataFrames at once
        X_all = pd.concat(X_list_reindexed, axis=0)
        y_all = pd.concat(y_list, axis=0).astype(np.int8)
        
        print(f"Loaded features shape: {X_all.shape}")
        
        return X_all, y_all
    
    @staticmethod
    def split_participants(base_dir, train_size=0.8, random_state=42):
        # Split participant folders in base_dir into train/test sets
        pids = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if len(pids) < 2:
            raise ValueError("Need at least 2 participants for splitting.")
        train_pids, test_pids = train_test_split(sorted(pids), train_size=train_size, random_state=random_state)
        return train_pids, test_pids

    @staticmethod
    def feature_selection(X_train, y_train, n_jobs=1):
        # Select statistically significant features using tsfresh's FeatureSelector transformer
        selector = FeatureSelector(n_jobs=n_jobs)  # Add n_jobs parameter
        X_train_sel = selector.fit_transform(X_train, y_train)
        # Fit a RandomForestClassifier on the selected features
        rf = RandomForestClassifier(n_jobs=n_jobs)  # Also parallelize RF
        rf.fit(X_train_sel, y_train)
        # Get feature importances
        importances = pd.Series(rf.feature_importances_, X_train_sel.columns)
        return selector, X_train_sel, importances
    
    @staticmethod
    def export_model(clf, results_path, event, datatype):
        model_path = os.path.join(results_path, f"{datatype}_{event}_rf_classifier.pkl")
        dump(clf, model_path)

    @staticmethod
    def select_and_export_top_features(out_root, datatype, event, results_dir, selector_n_jobs=1, train_pids=None, test_pids=None):
        # Load saved CSV features, do train-only selection, export top-100
        os.makedirs(results_dir, exist_ok=True)

        base = os.path.join(out_root, datatype, event.replace(" ", "_"))
        if train_pids is None or test_pids is None:
            train_pids, test_pids = UpperBodyClassifier.split_participants(base, train_size=0.8, random_state=42)
        
        with open(os.path.join(results_dir, "train_pids.json"), "w", encoding="utf-8") as f:
            json.dump(sorted(list(train_pids)), f, indent=2)

        with open(os.path.join(results_dir, "test_pids.json"), "w", encoding="utf-8") as f:
            json.dump(sorted(list(test_pids)), f, indent=2)

        # Load features only for train participants
        X_train, y_train = UpperBodyClassifier.load_features_for_participants(out_root, datatype, event, train_pids)

        # Feature selection with parallel processing
        selector, X_train_sel, importances = UpperBodyClassifier.feature_selection(X_train, y_train, n_jobs=selector_n_jobs)

        # Filter out duplicate feature types, keeping only the highest importance
        def get_feature_base_name(feature_name):
            import re
            parts = feature_name.split('__')
            cleaned = []
            for p in parts:
                if '_' in p:
                    key, val = p.rsplit('_', 1)  # paramName_value
                    if re.fullmatch(r'-?\d+(?:\.\d+)?', val):
                        p = key  # drop numeric value, keep param name (e.g., 'chunk_len')
                cleaned.append(p)
            return '__'.join(cleaned)

        
        # Group features by their base name and keep only the one with highest importance
        feature_groups = {}
        for feature_name, importance in importances.items():
            base_name = get_feature_base_name(feature_name)
            if base_name not in feature_groups or importance > feature_groups[base_name][1]:
                feature_groups[base_name] = (feature_name, importance)  # Keep original full name
        
        # Create filtered importances series with only the best feature from each group
        # This preserves the original complete feature names
        filtered_importances = pd.Series(
            {original_name: importance for original_name, importance in feature_groups.values()},
            name=importances.name
        )
        
        # Sort by importance and take top 100
        top100_filtered = filtered_importances.sort_values(ascending=False).head(100)

        # Export filtered top features with importances to JSON
        final_feat_path = os.path.join(results_dir, f"{datatype}_{event}_top100_features.json")
        
        # Modified function from ptb to order features by importance not alphabetical
        def export_features_json(ef: dict, filepath):
            with open(filepath, 'w') as outfile:
                json.dump(ef, outfile, indent=4)

        export_features_json(top100_filtered.to_dict(), final_feat_path)
        print(f"Found Top 100 {datatype} features by importance (filtered for duplicates).")

        return train_pids, test_pids
    
    @staticmethod
    def train_and_test_classifier(out_root, datatype, event, results_dir, train_pids=None, test_pids=None):
        os.makedirs(results_dir, exist_ok=True)
        
        # If split not provided, try load it; if not present, run selection now
        train_path = os.path.join(results_dir, "train_pids.json")
        test_path = os.path.join(results_dir, "test_pids.json")
        if train_pids is None or test_pids is None:
            if os.path.exists(train_path) and os.path.exists(test_path):
                with open(train_path, "r", encoding="utf-8") as f:
                    train_pids = set(json.load(f))
                with open(test_path, "r", encoding="utf-8") as f:
                    test_pids = set(json.load(f))
            else:
                train_pids, test_pids = UpperBodyClassifier.select_and_export_top_features(out_root, datatype, event, results_dir)
        # Load top-100 importances
        top100_path = os.path.join(results_dir, f"{datatype}_{event}_top100_features.json")
        if not os.path.exists(top100_path):
            train_pids, test_pids = UpperBodyClassifier.select_and_export_top_features(out_root, datatype, event, results_dir, train_pids, test_pids)
        with open(top100_path, "r", encoding="utf-8") as f:
            feat_imp = json.load(f)
        top_names = sorted(feat_imp.keys(), key=lambda x: feat_imp[x], reverse=True)
        # Load train/test features
        X_train, y_train = UpperBodyClassifier.load_features_for_participants(out_root, datatype, event, train_pids)
        X_test, y_test = UpperBodyClassifier.load_features_for_participants(out_root, datatype, event, test_pids)
        # Restrict to top 100 columns
        X_train = X_train.reindex(columns=top_names, fill_value=0)
        X_test = X_test.reindex(columns=top_names, fill_value=0)

        # Train and evaluate
        # Fit Random Forest Classifier
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        # Evaluate the classification model
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]  # Get probabilities for ROC AUC
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_proba)
        # Save metrics to file
        os.makedirs(results_dir, exist_ok=True)
        # Save accuracy and AUC scores
        metrics_path = os.path.join(results_dir, "classifier_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"AUC: {auc_score:.4f}\n")
            f.write("\n")
        # Save classification report separately
        report_path = os.path.join(results_dir, "classifier_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        # Visualise and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = os.path.join(results_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        # Export model
        UpperBodyClassifier.export_model(clf, results_dir, event, datatype)

        return clf, (train_pids, test_pids)

    @staticmethod
    def select_train_and_test_classifier(out_root, datatype, event, results_dir, selector_n_jobs=1, train_pids=None, test_pids=None):
        
        os.makedirs(results_dir, exist_ok=True)

        base = os.path.join(out_root, datatype, event.replace(" ", "_"))
        if train_pids is None or test_pids is None:
            train_pids, test_pids = UpperBodyClassifier.split_participants(base, train_size=0.8, random_state=42)
        
        # Save participant splits
        with open(os.path.join(results_dir, "train_pids.json"), "w", encoding="utf-8") as f:
            json.dump(sorted(list(train_pids)), f, indent=2)
        with open(os.path.join(results_dir, "test_pids.json"), "w", encoding="utf-8") as f:
            json.dump(sorted(list(test_pids)), f, indent=2)

        # Load features for both train and test sets
        X_train, y_train = UpperBodyClassifier.load_features_for_participants(out_root, datatype, event, train_pids)
        X_test, y_test = UpperBodyClassifier.load_features_for_participants(out_root, datatype, event, test_pids)

        # Feature selection on training data only
        selector, X_train_sel, importances = UpperBodyClassifier.feature_selection(X_train, y_train, n_jobs=selector_n_jobs)

        # Filter out duplicate feature types, keeping only the highest importance
        def get_feature_base_name(feature_name):
            import re
            parts = feature_name.split('__')
            cleaned = []
            for p in parts:
                if '_' in p:
                    key, val = p.rsplit('_', 1)  # paramName_value
                    if re.fullmatch(r'-?\d+(?:\.\d+)?', val):
                        p = key  # drop numeric value, keep param name (e.g., 'chunk_len')
                cleaned.append(p)
            return '__'.join(cleaned)

        # Group features by their base name and keep only the one with highest importance
        feature_groups = {}
        for feature_name, importance in importances.items():
            base_name = get_feature_base_name(feature_name)
            if base_name not in feature_groups or importance > feature_groups[base_name][1]:
                feature_groups[base_name] = (feature_name, importance)  # Keep original full name
        
        # Create filtered importances series with only the best feature from each group
        filtered_importances = pd.Series(
            {original_name: importance for original_name, importance in feature_groups.values()},
            name=importances.name
        )
        
        # Sort by importance and take top 100
        top100_filtered = filtered_importances.sort_values(ascending=False).head(100)

        # Export filtered top features with importances to JSON
        final_feat_path = os.path.join(results_dir, f"{datatype}_{event}_top100_features.json")
        
        def export_features_json(ef: dict, filepath):
            with open(filepath, 'w') as outfile:
                json.dump(ef, outfile, indent=4)

        export_features_json(top100_filtered.to_dict(), final_feat_path)
        print(f"Found Top 100 {datatype} features by importance (filtered for duplicates).")

        # Get top feature names in order of importance
        top_names = list(top100_filtered.index)
        
        # Restrict both train and test to top 100 columns
        X_train_top = X_train.reindex(columns=top_names, fill_value=0)
        X_test_top = X_test.reindex(columns=top_names, fill_value=0)

        # Train and evaluate classifier
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
        metrics_path = os.path.join(results_dir, "classifier_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"AUC: {auc_score:.4f}\n")
            f.write("\n")
        
        # Save classification report separately
        report_path = os.path.join(results_dir, "classifier_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        
        # Visualise and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = os.path.join(results_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        # Export model
        UpperBodyClassifier.export_model(clf, results_dir, event, datatype)

        return clf, (train_pids, test_pids)


class UpperBodyKinematics(Enum):
    pelvis = ["pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz"]
    lumbar = ["lumbar_extension", "lumbar_bending", "lumbar_rotation"]
    left_arm = ["arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l"]
    right_arm = ["arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r"]


class UpperBodyIMU(Enum):
    # head = "Head"
    # left_forearm = "LeftForeArm"
    # right_forearm = "RightForeArm"
    # left_hand = "LeftHand"
    right_hand = "RightHand"
    # left_shoulder = "LeftShoulder"
    # right_shoulder = "RightShoulder"
    # left_upper_arm = "LeftUpperArm"
    # right_upper_arm = "RightUpperArm"
    # pelvis = "Pelvis"
    # sternum = "T8"


if __name__ == "__main__":
    from datetime import datetime

    # Set up inputs
    root_dir = "Z:/Upper Body/Events/"
    # datatypes = ["IMU", "Kinematics"]
    datatypes = ["IMU"]

    # Get all event names from the enum
    events = list(EventWindowSize.events.value.keys())

    out_root = "Z:/Upper Body/Results/10 Participants/features"
    models_root = "Z:/Upper Body/Results/10 Participants/models"
    status_file = "Z:/Upper Body/Results/10 Participants/processing_status.txt"

    # Toggles
    RUN_EXTRACT = True
    RUN_SELECT_AND_TRAIN = True

    # 2 events in parallel, more cores per tsfresh
    total_cores = multiprocessing.cpu_count()
    events_n_jobs = 2
    tsfresh_n_jobs = max(1, (total_cores - 2) // events_n_jobs)
    # Use even more cores for feature selection (since it's less memory intensive than extraction)
    selector_n_jobs = min(total_cores - 1, tsfresh_n_jobs * 2)  # Cap at total-1 cores

    def write_status(path, msg):
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")

    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    with open(status_file, 'w') as f:
        f.write(f"Run start: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # Load event labels once at the start
    events_dict = read_event_labels("Z:/Upper Body/Event labels.xlsx")

    try:
        # Extract
        if RUN_EXTRACT:
            for datatype in datatypes:
                # Extract
                write_status(status_file, f"[EXTRACT] {datatype} BEGIN")
                Parallel(n_jobs=events_n_jobs, prefer="threads")(
                    delayed(UpperBodyClassifier.extract_and_save_features)(
                        ev, root_dir, datatype, out_root, tsfresh_n_jobs, events_dict
                    ) for ev in events
                )
                write_status(status_file, f"[EXTRACT] {datatype} END")

        # Combined select and train/test
        if RUN_SELECT_AND_TRAIN:
            for datatype in datatypes:
                write_status(status_file, f"[SELECT+TRAIN] {datatype} BEGIN")
                Parallel(n_jobs=events_n_jobs, prefer="threads")(
                    delayed(UpperBodyClassifier.select_train_and_test_classifier)(
                        out_root, 
                        datatype, 
                        ev, 
                        os.path.join(models_root, datatype, ev.replace(" ", "_")),
                        selector_n_jobs
                    ) for ev in events
                )
                write_status(status_file, f"[SELECT+TRAIN] {datatype} END")

        write_status(status_file, f"Run success: {datetime.now():%Y-%m-%d %H:%M:%S}")

    except Exception as e:
        write_status(status_file, f"Run failed: {e}")
        raise
