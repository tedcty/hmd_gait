import os
import pandas as pd
import numpy as np
from enum import Enum
import json
from joblib import Parallel, delayed
from ptb.ml.ml_util import MLOperations
from ptb.util.math.filters import Butterworth
from ptb.ml.tags import MLKeys
from tsfresh.transformers import FeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from event_constants import (WHOLE_EVENTS, REPETITIVE_EVENTS, EVENT_SIZES, EventWindowSize)
from trim_events import compute_whole_event_std_global, read_event_labels
from datetime import datetime
import re
import psutil


class UpperBodyClassifier:

    @staticmethod
    def upper_body_imu_for_event(event, root_dir):
        """
        Generate DataFrames for upper body IMU data for a specific event.
        """
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
        """
        Generate DataFrames for upper body kinematics data for a specific event.
        """
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
        """
        Generate binary labels for event detection based on event type and timing.
        """
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
        """
        Create sliding windows from time series data for feature extraction.
        """
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
        """
        Extract time series features from windowed data using tsfresh.
        """
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
        """
        Extract and save features for all trials of a specific event and data type.
        """
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
    def load_features_for_participants(out_root, datatype, event, participants, return_groups=False, n_jobs_loading=8):
        """
        Load and combine features from multiple participants for a specific event.
        """
        base = os.path.join(out_root, datatype, event.replace(" ", "_"))
        
        def print_memory_usage():
            process = psutil.Process()
            memory_gb = process.memory_info().rss / (1024**3)
            print(f"Memory usage: {memory_gb:.2f} GB")
        
        print("Scanning files and collecting column names...")
        file_list = []
        all_cols_set = set()
        
        scan_start = pd.Timestamp.now()
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
                        stem = fname[:-6]
                        Xp = os.path.join(sensor_path, f"{stem}_X.csv")
                        Yp = os.path.join(sensor_path, f"{stem}_y.csv")
                        if os.path.exists(Yp):
                            file_list.append((pid, Xp, Yp))
                            
                            # Read only header for column names
                            sample_df = pd.read_csv(Xp, index_col="window_id", nrows=0)
                            all_cols_set.update(sample_df.columns)
    
        scan_time = (pd.Timestamp.now() - scan_start).total_seconds()
        print(f"File scan completed in {scan_time:.1f}s - Found {len(file_list)} files with {len(all_cols_set)} features")
        
        if not file_list:
            raise RuntimeError("No CSV features found for given participants.")
        
        all_cols = sorted(all_cols_set)
        print_memory_usage()
        
        # Data loading with progress tracking
        X_list, y_list, groups_list = [], [], []
        load_start = pd.Timestamp.now()
        
        def load_single_file(pid, Xp, Yp, all_cols):
            """Helper function to load a single file pair"""
            try:
                X_df = pd.read_csv(Xp, index_col="window_id", dtype=np.float32)
                y_df = pd.read_csv(Yp, index_col="window_id")
                y_sr = y_df.iloc[:, 0].astype(np.int8)
                
                # Align indices
                common_idx = X_df.index.intersection(y_sr.index)
                if len(common_idx) == 0:
                    return None, None, None
                    
                X_df = X_df.loc[common_idx]
                y_sr = y_sr.loc[common_idx]
                
                # Reindex immediately
                X_df = X_df.reindex(columns=all_cols, fill_value=np.float32(0.0))
                
                groups_data = [pid] * len(X_df) if return_groups else []
                
                return X_df, y_sr, groups_data
            except Exception as e:
                print(f"Error loading {Xp}: {e}")
                return None, None, None
        
        # Parallel file loading
        print(f"Loading {len(file_list)} files in parallel with {n_jobs_loading} workers...")
        load_start = pd.Timestamp.now()
        
        results = Parallel(n_jobs=n_jobs_loading, prefer="threads")(
            delayed(load_single_file)(pid, Xp, Yp, all_cols) 
            for pid, Xp, Yp in file_list
        )
        
        # Filter out failed loads and separate results
        X_list, y_list, groups_list = [], [], []
        for X_df, y_sr, groups_data in results:
            if X_df is not None:
                X_list.append(X_df)
                y_list.append(y_sr)
                if return_groups:
                    groups_list.extend(groups_data)
        
        load_time = (pd.Timestamp.now() - load_start).total_seconds()
        print(f"Parallel loading completed in {load_time:.1f}s")
        
        # Single concatenation
        concat_start = pd.Timestamp.now()
        print("Concatenating DataFrames...")
        X_all = pd.concat(X_list, axis=0)
        y_all = pd.concat(y_list, axis=0).astype(np.int8)
        concat_time = (pd.Timestamp.now() - concat_start).total_seconds()
        
        print(f"Concatenation completed in {concat_time:.1f}s")
        print(f"Final features shape: {X_all.shape}")
        print_memory_usage()
        
        total_time = scan_time + load_time + concat_time
        print(f"Total loading time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        
        if return_groups:
            groups = np.array(groups_list)
            print(f"Groups shape: {groups.shape}, Unique participants: {len(np.unique(groups))}")
            return X_all, y_all, groups
        else:
            return X_all, y_all

    @staticmethod
    def feature_selection(X_train, y_train, n_jobs=1):
        """
        Perform feature selection using tsfresh statistical tests and Random Forest importance.
        """
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
    def k_fold_cross_validation(out_root, datatype, event, results_dir, combined_features, k=5, rf_n_jobs=1, random_state=42, top_k=None):
        """
        Perform participant-based k-fold cross-validation using the combined feature set.
        Includes comprehensive metrics and plots.
        """
        base = os.path.join(out_root, datatype, event.replace(" ", "_"))
        all_pids = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        
        if len(all_pids) < k:
            raise ValueError(f"Need at least {k} participants for {k}-fold CV. Found: {len(all_pids)}")
        
        # Load all features WITH group labels
        X_all, y_all, groups = UpperBodyClassifier.load_features_for_participants(
            out_root, datatype, event, all_pids, return_groups=True
        )
        
        # Restrict to combined features
        X_all = X_all.reindex(columns=combined_features, fill_value=0)
        
        # Participant-based K-fold cross-validation
        gkf = GroupKFold(n_splits=k)
        
        # Storage for comprehensive results
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        fold_metrics = {}
        fold_confusion_matrices = []
        fold_feature_importances = []
        roc_curves_data = []
        
        # Add top_k info to print statements if available
        top_k_str = f" (top {top_k})" if top_k else ""
        print(f"Running participant-based {k}-fold cross-validation for {datatype} - {event}{top_k_str}")
        print(f"Total samples: {len(X_all)}, Total participants: {len(np.unique(groups))}")
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_all, groups)):
            print(f"Processing fold {fold + 1}/{k}")
            
            X_train_fold = X_all.iloc[train_idx]
            X_test_fold = X_all.iloc[test_idx]
            y_train_fold = y_all.iloc[train_idx]
            y_test_fold = y_all.iloc[test_idx]
            
            # Get participant info for this fold
            train_participants = list(np.unique(groups[train_idx]))
            test_participants = list(np.unique(groups[test_idx]))
            
            print(f"  Train participants: {train_participants}")
            print(f"  Test participants: {test_participants}")
            
            # Train classifier
            clf = RandomForestClassifier(n_jobs=rf_n_jobs, random_state=random_state)
            clf.fit(X_train_fold, y_train_fold)
            
            # Evaluate
            y_pred = clf.predict(X_test_fold)
            y_proba = clf.predict_proba(X_test_fold)[:, 1]
            
            # Store for aggregate analysis
            all_y_true.extend(y_test_fold.tolist())
            all_y_pred.extend(y_pred.tolist())
            all_y_proba.extend(y_proba.tolist())
            
            # Calculate fold metrics (only AUC - not in classification_report)
            auc_score = roc_auc_score(y_test_fold, y_proba) if len(np.unique(y_test_fold)) > 1 else 0.0
            
            # Confusion matrix
            cm = confusion_matrix(y_test_fold, y_pred)
            fold_confusion_matrices.append(cm)
            
            # Feature importance
            fold_feature_importances.append(clf.feature_importances_)
            
            # ROC curve data (only if we have both classes)
            if len(np.unique(y_test_fold)) > 1:
                fpr, tpr, _ = roc_curve(y_test_fold, y_proba)
                roc_curves_data.append({'fpr': fpr, 'tpr': tpr, 'auc': auc_score})
            
            fold_metrics[f'fold_{fold + 1}'] = {
                'auc': auc_score,
                'train_size': len(X_train_fold),
                'test_size': len(X_test_fold),
                'train_participants': train_participants,
                'test_participants': test_participants,
                'confusion_matrix': cm.tolist()
            }
            
            print(f"  Fold {fold + 1}: AUC={auc_score:.4f}")
            print(f"  Train size: {len(X_train_fold)}, Test size: {len(X_test_fold)}")

        # Calculate additional metrics not included in classification_report
        overall_auc = roc_auc_score(all_y_true, all_y_proba)
        overall_cm = confusion_matrix(all_y_true, all_y_pred)
        
        # Calculate summary statistics for AUC (only additional metric)
        fold_aucs = [fold_metrics[f'fold_{i+1}']['auc'] for i in range(k)]
        
        # Get scikit-learn classification report as dict
        sklearn_report = classification_report(all_y_true, all_y_pred, output_dict=True)
        
        summary_stats = {
            'sklearn_classification_report': sklearn_report,
            'additional_metrics': {
                'overall_auc': overall_auc,
                'cv_auc_mean': np.mean(fold_aucs),
                'cv_auc_std': np.std(fold_aucs),
                'overall_confusion_matrix': overall_cm.tolist()
            },
            'individual_folds': fold_metrics
        }
        
        # Generate comprehensive plots
        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Add top_k suffix to all file names if available
        file_suffix = f"_top{top_k}" if top_k else ""
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        title_suffix = f" (top {top_k})" if top_k else ""
        plt.title(f'Overall Confusion Matrix - {datatype} {event}{title_suffix}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{datatype}_{event}_confusion_matrix{file_suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curves for all folds
        plt.figure(figsize=(10, 8))
        for i, roc_data in enumerate(roc_curves_data):
            plt.plot(roc_data['fpr'], roc_data['tpr'], alpha=0.7, 
                    label=f'Fold {i+1} (AUC = {roc_data["auc"]:.3f})')
        
        # Overall ROC curve
        overall_fpr, overall_tpr, _ = roc_curve(all_y_true, all_y_proba)
        overall_roc_auc = auc(overall_fpr, overall_tpr)
        plt.plot(overall_fpr, overall_tpr, 'k-', linewidth=3,
                label=f'Overall (AUC = {overall_roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {datatype} {event}{title_suffix}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{datatype}_{event}_roc_curves{file_suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature Prevalence (top 10 by prevalence percentage)
        # Load the prevalence analysis data
        prevalence_analysis_path = os.path.join(results_dir, f"{datatype}_{event}_prevalence_analysis.csv")
        if os.path.exists(prevalence_analysis_path):
            prevalence_df = pd.read_csv(prevalence_analysis_path)
            top_10_prevalence = prevalence_df.head(10)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=top_10_prevalence, x='prevalence_percentage', y='feature', orient='h')
            plt.title(f'Top 10 Features by Prevalence - {datatype} {event}{title_suffix}')
            plt.xlabel('Prevalence Percentage (%)')
            plt.ylabel('Feature')
            
            # Add prevalence count annotations to the bars
            for i, (idx, row) in enumerate(top_10_prevalence.iterrows()):
                plt.text(row['prevalence_percentage'] + 1, i, 
                        f"{row['appeared_in_iterations']}", 
                        va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{datatype}_{event}_feature_prevalence{file_suffix}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            # Fallback to importance-based plot if prevalence data not available
            avg_importances = np.mean(fold_feature_importances, axis=0)
            feature_importance_df = pd.DataFrame({
                'feature': combined_features,
                'importance': avg_importances
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            top_10_features = feature_importance_df.head(10)
            sns.barplot(data=top_10_features, x='importance', y='feature', orient='h')
            plt.title(f'Top 10 Feature Importances - {datatype} {event}{title_suffix}')
            plt.xlabel('Average Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{datatype}_{event}_feature_prevalence{file_suffix}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Cross-Validation AUC Box Plot
        cv_metrics_df = pd.DataFrame({
            'AUC': fold_aucs
        })
        
        plt.figure(figsize=(6, 6))
        cv_metrics_df.boxplot()
        plt.title(f'Cross-Validation AUC Distribution - {datatype} {event}{title_suffix}')
        plt.ylabel('AUC Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{datatype}_{event}_cv_auc_boxplot{file_suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Save feature importances
        if 'feature_importance_df' not in locals():
            avg_importances = np.mean(fold_feature_importances, axis=0)
            feature_importance_df = pd.DataFrame({
                'feature': combined_features,
                'importance': avg_importances
            }).sort_values('importance', ascending=False)
        
        feature_importance_df.to_csv(
            os.path.join(results_dir, f"{datatype}_{event}_feature_importances{file_suffix}.csv"),
            index=False
        )

        # Save comprehensive text report
        report_path = os.path.join(results_dir, f"{datatype}_{event}_classification_report{file_suffix}.txt")
        
        def fold_lines():
            lines = []
            for i in range(k):
                fkey = f'fold_{i+1}'
                f = fold_metrics[fkey]
                tn, fp = f['confusion_matrix'][0]
                fn, tp = f['confusion_matrix'][1]
                lines.append(
                    f"  Fold {i+1}:\n"
                    f"    AUC: {f['auc']:.4f}\n"
                    f"    Train size: {f['train_size']}   Test size: {f['test_size']}\n"
                    f"    Train participants: {', '.join(map(str, f['train_participants']))}\n"
                    f"    Test participants: {', '.join(map(str, f['test_participants']))}\n"
                    f"    Confusion matrix (TN, FP, FN, TP): {tn}, {fp}, {fn}, {tp}\n\n"
                )
            return "".join(lines)

        with open(report_path, "w") as f:
            f.write(f"Classification Report - {datatype} {event}{title_suffix}\n")
            f.write("=" * 50 + "\n\n")
            f.write(classification_report(all_y_true, all_y_pred))
            f.write("\n" + "-" * 50 + "\n")
            f.write("Summary Metrics\n")
            f.write("-" * 50 + "\n")
            f.write("AUC Metrics:\n")
            f.write(f"  Overall AUC: {overall_auc:.4f}\n")
            f.write(f"  Cross-Validation AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}\n\n")
            f.write("Confusion Matrix (Overall):\n")
            f.write(f"  True Negative: {overall_cm[0,0]}    False Positive: {overall_cm[0,1]}\n")
            f.write(f"  False Negative: {overall_cm[1,0]}   True Positive: {overall_cm[1,1]}\n\n")
            f.write("Cross-Validation Details:\n")
            f.write(f"  Number of folds: {k}\n")
            f.write(f"  Total samples: {len(all_y_true)}\n")
            f.write(f"  Total participants: {len(np.unique(groups))}\n")
            f.write(f"  Features used: {len(combined_features)}\n\n")
            f.write("Per-Fold Breakdown:\n")
            f.write(fold_lines())
    
        print(f"\nComprehensive K-Fold CV Results{top_k_str}:")
        print(f"Overall Accuracy: {sklearn_report['accuracy']:.4f}")
        print(f"Overall AUC: {overall_auc:.4f}")
        print(f"Overall F1-Score: {sklearn_report['weighted avg']['f1-score']:.4f}")
        print(f"CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
        print(f"Classification report: {report_path}")
        print(f"Plots saved to: {plots_dir}")
        
        # Return just the important data for programmatic use
        return {
            'overall_auc': overall_auc,
            'cv_auc_mean': np.mean(fold_aucs),
            'cv_auc_std': np.std(fold_aucs),
            'accuracy': sklearn_report['accuracy'],
            'f1_score': sklearn_report['weighted avg']['f1-score']
        }

    @staticmethod
    def lopo_feature_selection_and_cv(out_root, datatype, event, results_dir, 
                                  selector_n_jobs=1, rf_n_jobs=1, k_folds=5, 
                                  top_k_values=[100, 50, 20, 10], 
                                  loading_n_jobs=8):
        """
        Complete pipeline: LOPO feature selection + K-fold CV evaluation for multiple top_k values.
        """
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"Starting LOPO feature selection and CV for {datatype} - {event} (top_k values: {top_k_values})")
        
        # Load data once for all top_k values
        base = os.path.join(out_root, datatype, event.replace(" ", "_"))
        all_pids = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        
        if len(all_pids) < 3:
            raise ValueError(f"Need at least 3 participants for LOPO feature selection. Found: {len(all_pids)}")
        
        print(f"Loading all data for {len(all_pids)} participants...")
        
        # Load ALL data at once - shared across all top_k values
        X_all, y_all, groups = UpperBodyClassifier.load_features_for_participants(
            out_root, datatype, event, all_pids, return_groups=True, 
            n_jobs_loading=loading_n_jobs
        )
        
        print(f"Loaded {X_all.shape[0]} samples with {X_all.shape[1]} features")
        
        # Run LOPO once and collect all feature importance data
        logo = LeaveOneGroupOut()
        all_importance_dicts = []
        
        def get_feature_base_name(feature_name):
            parts = feature_name.split('__')
            cleaned = []
            for p in parts:
                if '_' in p:
                    key, val = p.rsplit('_', 1)
                    if re.fullmatch(r'-?\d+(?:\.\d+)?', val):
                        p = key
                cleaned.append(p)
            return '__'.join(cleaned)
        
        print(f"Running LOPO feature selection...")
        
        for i, (train_idx, test_idx) in enumerate(logo.split(X_all, y_all, groups)):
            held_out_participants = list(np.unique(groups[test_idx]))
            print(f"LOPO iteration {i+1}/{len(all_pids)}: Holding out {held_out_participants}")
            
            X_train = X_all.iloc[train_idx]
            y_train = y_all.iloc[train_idx]
            
            selector, X_train_sel, importances = UpperBodyClassifier.feature_selection(
                X_train, y_train, n_jobs=selector_n_jobs
            )
            
            # Filter duplicates
            feature_groups = {}
            for feature_name, importance in importances.items():
                base_name = get_feature_base_name(feature_name)
                if base_name not in feature_groups or importance > feature_groups[base_name][1]:
                    feature_groups[base_name] = (feature_name, importance)
            
            filtered_importances = pd.Series(
                {original_name: importance for original_name, importance in feature_groups.values()},
                name=importances.name
            )
            
            all_importance_dicts.append(filtered_importances.to_dict())
            print(f"  Selected {len(filtered_importances)} features from {len(X_train)} training samples")
    
        # Process each top_k value using the same LOPO results
        results = {}
        
        for top_k in top_k_values:
            print(f"\nProcessing top_{top_k} features...")
            
            # Create subfolder for this top_k value
            results_dir_topk = os.path.join(results_dir, f"top_{top_k}")
            os.makedirs(results_dir_topk, exist_ok=True)
            
            # Generate top_k feature sets for each LOPO iteration
            top_k_feature_sets = []
            for importance_dict in all_importance_dicts:
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                top_k_features = [feature_name for feature_name, _ in sorted_features[:top_k]]
                top_k_feature_sets.append(set(top_k_features))
            
            # Calculate prevalence-based combined feature set
            feature_prevalence = {}
            for feature_set in top_k_feature_sets:
                for feature in feature_set:
                    feature_prevalence[feature] = feature_prevalence.get(feature, 0) + 1
            
            def calculate_max_importance(feature_name, importance_dicts):
                scores = [d.get(feature_name, 0) for d in importance_dicts]
                return max(scores) if scores else 0
            
            feature_ranking = []
            for feature, prevalence in feature_prevalence.items():
                max_importance = calculate_max_importance(feature, all_importance_dicts)
                feature_ranking.append((feature, prevalence, max_importance))
            
            feature_ranking.sort(key=lambda x: (x[1], x[2]), reverse=True)
            
            top_k_prevalent = feature_ranking[:top_k]
            prevalence_features = [feature for feature, _, _ in top_k_prevalent]
            
            prevalence_importances = {
                feature: calculate_max_importance(feature, all_importance_dicts) 
                for feature in prevalence_features
            }
            
            # Save results
            def export_features_json(ef: dict, filepath):
                with open(filepath, 'w') as outfile:
                    json.dump(ef, outfile, indent=4)
            
            prevalence_path = os.path.join(results_dir_topk, f"{datatype}_{event}_prevalence_top{top_k}_features.json")
            export_features_json(prevalence_importances, prevalence_path)
            
            prevalence_analysis = []
            for feature, prevalence, max_importance in top_k_prevalent:
                prevalence_analysis.append({
                    'feature': feature,
                    'prevalence': prevalence,
                    'prevalence_percentage': (prevalence / len(all_pids)) * 100,
                    'max_importance': max_importance,
                    'appeared_in_iterations': f"{prevalence}/{len(all_pids)}"
                })
            
            prevalence_df = pd.DataFrame(prevalence_analysis)
            prevalence_analysis_path = os.path.join(results_dir_topk, f"{datatype}_{event}_prevalence_analysis.csv")
            prevalence_df.to_csv(prevalence_analysis_path, index=False)
            
            print(f"Top {top_k} results: {len(prevalence_features)} features -> {prevalence_path}")
            
            # Run K-fold CV for this top_k
            cv_results = UpperBodyClassifier.k_fold_cross_validation(
                out_root, datatype, event, results_dir_topk, prevalence_features, k_folds, rf_n_jobs, top_k=top_k
            )
            
            results[f'top_{top_k}'] = {
                'features': prevalence_features,
                'cv_results': cv_results
            }
        
        return results


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
    RUN_EXTRACT = False
    RUN_SELECT_AND_TRAIN = True

    total_cores = multiprocessing.cpu_count()
    print(f"Total available cores: {total_cores}")
    
    if RUN_EXTRACT:
        # For extraction: parallel events with cores for tsfresh
        events_n_jobs = 2
        tsfresh_n_jobs = max(1, (total_cores - 2) // events_n_jobs)
        print(f"Extraction: {events_n_jobs} events in parallel, {tsfresh_n_jobs} cores per tsfresh")
    
    if RUN_SELECT_AND_TRAIN:
        # Optimise for different workload types        
        loading_n_jobs = min(12, max(4, total_cores // 4))
        remaining_cores = total_cores - 2  # Leave 2 for system
        selector_n_jobs = max(4, remaining_cores // 2)
        rf_n_jobs = max(4, remaining_cores // 2)
        
        print(f"Core allocation:")
        print(f"  Loading: {loading_n_jobs} cores")
        print(f"  Feature selection: {selector_n_jobs} cores") 
        print(f"  Random Forest: {rf_n_jobs} cores")

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
                write_status(status_file, f"[EXTRACT] {datatype} BEGIN")
                Parallel(n_jobs=events_n_jobs, prefer="threads")(
                    delayed(UpperBodyClassifier.extract_and_save_features)(
                        ev, root_dir, datatype, out_root, tsfresh_n_jobs, events_dict
                    ) for ev in events
                )
                write_status(status_file, f"[EXTRACT] {datatype} END")

        # Select and train/test for current TOP_K value
        if RUN_SELECT_AND_TRAIN:
            for datatype in datatypes:
                write_status(status_file, f"[SELECT+TRAIN] {datatype} BEGIN")
                # Run feature selection and cross-validation in series instead of parallel
                for ev in events:
                    UpperBodyClassifier.lopo_feature_selection_and_cv(
                        out_root, 
                        datatype, 
                        ev, 
                        os.path.join(models_root, datatype, ev.replace(" ", "_")),
                        selector_n_jobs,
                        rf_n_jobs,
                        5,  # k_folds
                        [100, 50, 20, 10]  # top_k_values
                    )
                write_status(status_file, f"[SELECT+TRAIN] {datatype} END")

        write_status(status_file, f"Run success: {datetime.now():%Y-%m-%d %H:%M:%S}")

    except Exception as e:
        write_status(status_file, f"Run failed: {e}")
        raise
