import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import pandas as pd
import numpy as np
import json
from datetime import datetime
import gc
from tqdm import tqdm
from joblib import Parallel, delayed
import psutil
import multiprocessing
import traceback

# Import reusable functions from upper_body_classifier.py
from upper_body_classifier import UpperBodyClassifier


class MinimalIMUClassifier:
    """
    Classifier for minimal IMU subset using same feature selection as full classifier.
    Only loads features from specified IMU sensors, then runs identical pipeline.
    """
    
    @staticmethod
    def filter_columns_by_imu_subset(columns, imu_subset):
        """
        Filter column list to only include features from the specified IMU subset.
        """
        filtered_columns = []
        for col in columns:
            # Check if column starts with any of the allowed IMU sensor IDs
            if any(col.startswith(f"{imu}_") for imu in imu_subset):
                filtered_columns.append(col)
        return filtered_columns
    
    @staticmethod
    def load_features_for_participants_filtered(out_root, datatype, event, participants, 
                                                imu_subset, return_groups=False, n_jobs_loading=8):
        """
        Load features but filter to only include specified IMU sensors.
        Similar to UpperBodyClassifier.load_features_for_participants but with IMU filtering.
        """
        
        base = os.path.join(out_root, datatype, event.replace(" ", "_"))
        
        def print_memory_usage():
            process = psutil.Process()
            memory_gb = process.memory_info().rss / (1024**3)
            print(f"Memory usage: {memory_gb:.2f} GB")
        
        print("Scanning files and collecting column names...")
        file_list = []
        all_cols_set = set()
        
        for pid in participants:
            pid_path = os.path.join(base, pid)
            if not os.path.isdir(pid_path):
                continue
            for sensor in os.listdir(pid_path):
                # Filter at sensor directory level
                if sensor not in imu_subset:
                    continue
                    
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
        
        if not file_list:
            raise RuntimeError(f"No CSV features found for IMU subset {imu_subset}")
        
        # Filter columns to only those from the IMU subset
        all_cols = sorted(all_cols_set)
        filtered_cols = MinimalIMUClassifier.filter_columns_by_imu_subset(all_cols, imu_subset)
        
        print(f"File scan completed - Found {len(file_list)} files")
        print(f"Total columns: {len(all_cols)}, Filtered to IMU subset: {len(filtered_cols)}")
        print_memory_usage()
        
        def load_single_file(pid, Xp, Yp, filtered_cols):
            """Helper function to load a single file pair with column filtering"""
            try:
                X_df = pd.read_csv(Xp, index_col="window_id")
                
                # Filter to only IMU subset columns
                X_df = X_df[[col for col in X_df.columns if col in filtered_cols]]
                
                # Convert to float32
                for col in X_df.columns:
                    X_df[col] = pd.to_numeric(X_df[col], errors='coerce').astype(np.float32)
                
                y_df = pd.read_csv(Yp, index_col="window_id")
                y_sr = y_df.iloc[:, 0].astype(np.int8)
                
                # Align indices
                common_idx = X_df.index.intersection(y_sr.index)
                if len(common_idx) == 0:
                    return None, None, None
                    
                X_df = X_df.loc[common_idx]
                y_sr = y_sr.loc[common_idx]
                
                # Reindex to ensure consistent column order
                X_df = X_df.reindex(columns=filtered_cols, fill_value=np.float32(0.0))
                
                groups_data = [pid] * len(X_df) if return_groups else []
                
                return X_df, y_sr, groups_data
            except Exception as e:
                print(f"Error loading {Xp}: {e}")
                return None, None, None
        
        # Parallel file loading
        print(f"Loading {len(file_list)} files in parallel with {n_jobs_loading} workers...")
        
        results = Parallel(n_jobs=n_jobs_loading, prefer="processes")(
            delayed(load_single_file)(pid, Xp, Yp, filtered_cols) 
            for pid, Xp, Yp in tqdm(file_list, desc="Loading files", unit="file")
        )
        
        # Filter out failed loads
        X_list, y_list, groups_list = [], [], []
        for X_df, y_sr, groups_data in results:
            if X_df is not None:
                X_list.append(X_df)
                y_list.append(y_sr)
                if return_groups:
                    groups_list.extend(groups_data)
        
        print("Concatenating DataFrames...")
        X_all = pd.concat(X_list, axis=0)
        y_all = pd.concat(y_list, axis=0).astype(np.int8)
        
        print(f"Final features shape: {X_all.shape}")
        print_memory_usage()
        
        if return_groups:
            groups = np.array(groups_list)
            print(f"Groups shape: {groups.shape}, Unique participants: {len(np.unique(groups))}")
            return X_all, y_all, groups
        else:
            return X_all, y_all
    
    @staticmethod
    def participant_level_feature_selection_filtered(out_root, datatype, event, imu_subset, selector_n_jobs=1):
        """
        Run feature selection on each participant with filtered IMU subset.
        Uses UpperBodyClassifier's feature selection but loads filtered data.
        """
        base = os.path.join(out_root, datatype, event.replace(" ", "_"))
        all_pids = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        
        participant_features = {}
        
        for pid in all_pids:
            print(f"Feature selection for participant {pid} (IMU subset: {imu_subset})...")
            
            try:
                # Load filtered data for this participant
                X_participant, y_participant = MinimalIMUClassifier.load_features_for_participants_filtered(
                    out_root, datatype, event, [pid], imu_subset, return_groups=False
                )
                
                if X_participant.empty:
                    print(f"  No data found for participant {pid} with IMU subset")
                    continue
                
                # Use the exact same feature selection as full classifier
                selector, X_train_sel, importances = UpperBodyClassifier.feature_selection(
                    X_participant, y_participant, n_jobs=selector_n_jobs
                )
                
                # Store the selected features and importances
                participant_features[pid] = importances.to_dict()
                
                # Cleanup
                del X_participant, y_participant, selector, X_train_sel, importances
                gc.collect()
                
                print(f"  Selected {len(participant_features[pid])} features")
                
            except Exception as e:
                print(f"Error processing participant {pid}: {e}")
                continue
        
        return participant_features
    
    @staticmethod
    def lopo_pipeline_minimal_imu(out_root, datatype, event, imu_subset, results_dir,
                                  selector_n_jobs=1, rf_n_jobs=1, k_folds=5,
                                  top_k_values=[100],
                                  loading_n_jobs=8, subset_name=None):
        """
        Run the complete LOPO pipeline for minimal IMU subset.
        Identical to UpperBodyClassifier.lopo_pipeline but with filtered data loading.
        """
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"LOPO Pipeline for Minimal IMU Classifier")
        print(f"Event: {event}")
        print(f"IMU Subset: {imu_subset}")
        if subset_name:
            print(f"Subset Name: {subset_name}")
        print(f"{'='*80}\n")
        
        # Step 1: Participant-level feature selection with filtered data
        print(f"Step 1: Running participant-level feature selection for {datatype} - {event}")
        participant_features = MinimalIMUClassifier.participant_level_feature_selection_filtered(
            out_root, datatype, event, imu_subset, selector_n_jobs
        )
        
        if len(participant_features) < 3:
            raise ValueError(f"Need at least 3 participants for feature selection. Found: {len(participant_features)}")
        
        # Step 2: Use UpperBodyClassifier's prevalence analysis (same logic)
        print(f"Step 2: Analysing feature prevalence across participants")
        top_k_results, master_ranking = UpperBodyClassifier.lopo_feature_prevalence_analysis(
            participant_features, top_k_values
        )
        
        # Step 3: Get participant list
        base = os.path.join(out_root, datatype, event.replace(" ", "_"))
        all_pids = list(participant_features.keys())
        
        # Step 4: Run CV for each top_k
        results = {}
        for top_k in top_k_values:
            print(f"Step 4: Running {k_folds}-fold CV for top_{top_k}")
            
            results_dir_topk = os.path.join(results_dir, f"top_{top_k}")
            os.makedirs(results_dir_topk, exist_ok=True)
            
            prevalence_features = top_k_results[f'top_{top_k}']
            
            # Save prevalence analysis
            top_k_prevalent = master_ranking[:top_k]
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
            
            # Save feature list
            prevalence_importances = {
                feature: max_importance for feature, _, max_importance in top_k_prevalent
            }
            prevalence_path = os.path.join(results_dir_topk, f"{datatype}_{event}_prevalence_top{top_k}_features.json")
            with open(prevalence_path, 'w') as outfile:
                json.dump(prevalence_importances, outfile, indent=4)
            
            # Load filtered data for CV
            print(f"Loading data with only {len(prevalence_features)} selected features from IMU subset...")
            X_all, y_all, groups = MinimalIMUClassifier.load_features_for_participants_filtered(
                out_root, datatype, event, all_pids, imu_subset, 
                return_groups=True, n_jobs_loading=loading_n_jobs
            )
            
            # Filter to only prevalence features
            available_features = [f for f in prevalence_features if f in X_all.columns]
            X_all = X_all[available_features]
            
            print(f"Filtered data shape: {X_all.shape}")
            
            # Use suffix that includes subset name if provided
            cv_suffix = f"{subset_name}_top{top_k}" if subset_name else f"top{top_k}"
            
            # Run k-fold CV using UpperBodyClassifier's method
            cv_results = UpperBodyClassifier.k_fold_cross_validation(
                out_root, datatype, event, results_dir_topk, available_features,
                k_folds, rf_n_jobs, top_k=cv_suffix,
                preloaded_data=(X_all, y_all, groups)
            )
            
            results[f'top_{top_k}'] = {
                'features': available_features,
                'cv_results': cv_results
            }
            
            # Cleanup
            del X_all, y_all, groups
            gc.collect()
        
        return results


if __name__ == "__main__":
    
    # Configuration
    out_root = "/hpc/vlee669/Results/30 Participants/features"
    results_root = "/hpc/vlee669/Results/30 Participants/minimal_imu_models"
    
    # Core allocation
    total_cores = multiprocessing.cpu_count()
    print(f"Total available cores: {total_cores}")
    half_cores = max(1, total_cores // 2)
    
    loading_n_jobs = min(16, half_cores)
    selector_n_jobs = half_cores
    rf_n_jobs = 16
    
    print(f"Core allocation:")
    print(f"  Loading: {loading_n_jobs} cores")
    print(f"  Feature Selection: {selector_n_jobs} cores")
    print(f"  Random Forest: {rf_n_jobs} cores")
    
    # Define the event you want to train
    event = "Dribbling basketball"
    datatype = "IMU"
    
    # Define both IMU subsets to test
    imu_subsets = {
        "minimal_analysis": ["LeftHand_imu", "RightHand_imu"],  # Based on analysis (normalised sum of importance scores)
        "product_usecase_left": ["Head_imu", "LeftForeArm_imu"],   # Based on product use-case considerations
        "product_usecase_right": ["Head_imu", "RightForeArm_imu"]   # Another product use-case
    }
    
    print(f"\nTraining event: {event}")
    print(f"Testing {len(imu_subsets)} IMU subsets")
    print(f"Start time: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    
    results_comparison = []
    
    for subset_name, imu_subset in imu_subsets.items():
        print(f"\n{'='*80}")
        print(f"Training with {subset_name}: {imu_subset}")
        print(f"{'='*80}\n")
        
        subset_results_dir = os.path.join(
            results_root,
            event.replace(" ", "_"),
            subset_name
        )
        os.makedirs(subset_results_dir, exist_ok=True)
        
        try:
            # Run complete pipeline
            results = MinimalIMUClassifier.lopo_pipeline_minimal_imu(
                out_root=out_root,
                datatype=datatype,
                event=event,
                imu_subset=imu_subset,
                results_dir=subset_results_dir,
                selector_n_jobs=selector_n_jobs,
                rf_n_jobs=rf_n_jobs,
                k_folds=5,
                top_k_values=[100],
                loading_n_jobs=loading_n_jobs,
                subset_name=subset_name
            )
            
            # Store top 100 results for comparison
            top_100_results = results['top_100']['cv_results']
            
            # Metrics are already at the top level of cv_results
            results_comparison.append({
                'subset_name': subset_name,
                'imus': ', '.join(imu_subset),
                'num_imus': len(imu_subset),
                'overall_auc': top_100_results['overall_auc'],
                'cv_auc_mean': top_100_results['cv_auc_mean'],
                'cv_auc_std': top_100_results['cv_auc_std'],
                'accuracy': top_100_results['accuracy'],
                'f1_score': top_100_results['f1_score']
            })
            
        except Exception as e:
            print(f"\nPipeline failed for {subset_name}: {e}")
            traceback.print_exc()
            results_comparison.append({
                'subset_name': subset_name,
                'imus': ', '.join(imu_subset),
                'num_imus': len(imu_subset),
                'overall_auc': None,
                'cv_auc_mean': None,
                'cv_auc_std': None,
                'accuracy': None,
                'f1_score': None,
                'error': str(e)
            })
    
    # Save comparison report
    comparison_df = pd.DataFrame(results_comparison)
    comparison_path = os.path.join(
        results_root,
        f"{event.replace(' ', '_')}_imu_subset_comparison.csv"
    )
    comparison_df.to_csv(comparison_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"End time: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Training completed for all subsets!")
    print(f"{'='*80}\n")
    
    print("Comparison Results:")
    print(comparison_df.to_string(index=False))
    print(f"\nComparison saved to: {comparison_path}")
