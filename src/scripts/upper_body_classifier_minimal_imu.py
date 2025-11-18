import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import gc

# Import reusable functions from upper_body_classifier.py
from upper_body_classifier import UpperBodyClassifier
from event_constants import EventWindowSize


class MinimalIMUClassifier:
    """
    Classifier for minimal IMU subset using pre-selected top 100 features.
    Skips feature extraction and selection, goes straight to training.
    """
    
    @staticmethod
    def filter_features_by_imu_subset(features_list, imu_subset):
        """
        Filter feature list to only include features from the specified IMU subset.
        """
        filtered_features = []
        for feature in features_list:
            # Check if feature starts with any of the allowed IMU sensor IDs
            if any(feature.startswith(f"{imu}_") for imu in imu_subset):
                filtered_features.append(feature)
        
        return filtered_features
    
    @staticmethod
    def load_top_100_features(models_dir, datatype, event):
        """
        Load the top 100 selected features from the prevalence analysis.

        """
        prevalence_file = os.path.join(
            models_dir, datatype, event.replace(" ", "_"), 
            "top_100", f"{datatype}_{event}_prevalence_top100_features.json"
        )
        
        if not os.path.exists(prevalence_file):
            raise FileNotFoundError(f"Top 100 features file not found: {prevalence_file}")
        
        with open(prevalence_file, 'r') as f:
            feature_importances = json.load(f)
        
        # Return feature names (keys of the dictionary)
        return list(feature_importances.keys())
    
    @staticmethod
    def train_minimal_imu_model(out_root, models_dir, event, imu_subset, 
                                results_dir, k_folds=5, rf_n_jobs=1, 
                                loading_n_jobs=8):
        """
        Train a Random Forest classifier using top 100 features on minimal IMU subset.
        """
        datatype = "IMU"  # Only IMU for this classifier
        
        print(f"\n{'='*80}")
        print(f"Training Minimal IMU Classifier for: {event}")
        print(f"Using IMU subset: {imu_subset}")
        print(f"{'='*80}\n")
        
        # Step 1: Load top 100 features
        print("Step 1: Loading top 100 features...")
        top_100_features = MinimalIMUClassifier.load_top_100_features(
            models_dir, datatype, event
        )
        print(f"  Loaded {len(top_100_features)} features")
        
        # Step 2: Filter to minimal IMU subset
        print(f"Step 2: Filtering features to minimal IMU subset...")
        filtered_features = MinimalIMUClassifier.filter_features_by_imu_subset(
            top_100_features, imu_subset
        )
        print(f"  Filtered to {len(filtered_features)} features from {len(imu_subset)} IMUs")
        
        if len(filtered_features) == 0:
            raise ValueError(f"No features found for IMU subset {imu_subset}. Check feature names and IMU IDs.")
        
        # Step 3: Load data with only the filtered features
        print("Step 3: Loading data with filtered features...")
        base = os.path.join(out_root, datatype, event.replace(" ", "_"))
        all_pids = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        
        X_all, y_all, groups = UpperBodyClassifier.load_features_for_participants_selective(
            out_root, datatype, event, all_pids, filtered_features, 
            return_groups=True, n_jobs_loading=loading_n_jobs
        )
        print(f"  Loaded data shape: {X_all.shape}")
        print(f"  Participants: {len(np.unique(groups))}")
        
        # Step 4: Run k-fold cross-validation
        print(f"Step 4: Running {k_folds}-fold cross-validation...")
        cv_results = UpperBodyClassifier.k_fold_cross_validation(
            out_root, datatype, event, results_dir, filtered_features,
            k=k_folds, rf_n_jobs=rf_n_jobs, top_k=None,
            preloaded_data=(X_all, y_all, groups)
        )
        
        # Step 5: Save minimal IMU configuration
        config_path = os.path.join(results_dir, f"{event.replace(' ', '_')}_minimal_imu_config.json")
        config = {
            'event': event,
            'imu_subset': imu_subset,
            'num_features': len(filtered_features),
            'features': filtered_features,
            'cv_results': {
                'overall_auc': float(cv_results['overall_auc']),
                'cv_auc_mean': float(cv_results['cv_auc_mean']),
                'cv_auc_std': float(cv_results['cv_auc_std']),
                'accuracy': float(cv_results['accuracy']),
                'f1_score': float(cv_results['f1_score'])
            },
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"\nSaved minimal IMU configuration: {config_path}")
        print(f"\nResults Summary:")
        print(f"  Overall AUC: {cv_results['overall_auc']:.4f}")
        print(f"  CV AUC: {cv_results['cv_auc_mean']:.4f} ± {cv_results['cv_auc_std']:.4f}")
        print(f"  Accuracy: {cv_results['accuracy']:.4f}")
        print(f"  F1-Score: {cv_results['f1_score']:.4f}")
        
        # Cleanup
        del X_all, y_all, groups
        gc.collect()
        
        return cv_results
    
    @staticmethod
    def batch_train_all_events(out_root, models_dir, results_root, 
                               imu_subset_mapping, k_folds=5, 
                               rf_n_jobs=1, loading_n_jobs=8):
        """
        Train minimal IMU classifiers for all events with specified IMU subsets.
        """
        os.makedirs(results_root, exist_ok=True)
        
        results_summary = []
        
        for event, imu_subset in imu_subset_mapping.items():
            print(f"\n{'#'*80}")
            print(f"Processing Event: {event}")
            print(f"{'#'*80}")
            
            try:
                # Create results directory for this event
                event_results_dir = os.path.join(
                    results_root, event.replace(" ", "_")
                )
                os.makedirs(event_results_dir, exist_ok=True)
                
                # Train model
                cv_results = MinimalIMUClassifier.train_minimal_imu_model(
                    out_root, models_dir, event, imu_subset,
                    event_results_dir, k_folds, rf_n_jobs, loading_n_jobs
                )
                
                # Add to summary
                results_summary.append({
                    'event': event,
                    'imu_subset': ', '.join(imu_subset),
                    'num_imus': len(imu_subset),
                    'overall_auc': cv_results['overall_auc'],
                    'cv_auc_mean': cv_results['cv_auc_mean'],
                    'cv_auc_std': cv_results['cv_auc_std'],
                    'accuracy': cv_results['accuracy'],
                    'f1_score': cv_results['f1_score'],
                    'status': 'Success'
                })
                
            except Exception as e:
                print(f"ERROR processing {event}: {e}")
                results_summary.append({
                    'event': event,
                    'imu_subset': ', '.join(imu_subset),
                    'num_imus': len(imu_subset),
                    'overall_auc': None,
                    'cv_auc_mean': None,
                    'cv_auc_std': None,
                    'accuracy': None,
                    'f1_score': None,
                    'status': f'Failed: {str(e)}'
                })
                continue
            
            # Cleanup between events
            gc.collect()
        
        # Save summary report
        summary_df = pd.DataFrame(results_summary)
        summary_path = os.path.join(results_root, "minimal_imu_training_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*80}")
        print(f"Batch Training Complete!")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*80}\n")
        
        # Print summary table
        print(summary_df.to_string(index=False))
        
        return summary_df


if __name__ == "__main__":
    
    # Configuration
    out_root = "/hpc/vlee669/Results/30 Participants/features"
    models_root = "/hpc/vlee669/Results/30 Participants/models"
    results_root = "/hpc/vlee669/Results/30 Participants/minimal_imu_models"
    
    # Core allocation
    import multiprocessing
    total_cores = multiprocessing.cpu_count()
    print(f"Total available cores: {total_cores}")
    half_cores = max(1, total_cores // 2)
    
    loading_n_jobs = min(16, half_cores)
    rf_n_jobs = 16
    
    print(f"Core allocation:")
    print(f"  Loading: {loading_n_jobs} cores")
    print(f"  Random Forest: {rf_n_jobs} cores")
    
    # Define the event you want to train
    event = "Straight walk"  # Change this to your desired event
    
    # Define the minimal IMU subset for this event
    imu_subset = [
        "Head_imu",
        "LeftForeArm_imu", 
        "RightForeArm_imu"
    ]
    
    # Create results directory for this event
    event_results_dir = os.path.join(results_root, event.replace(" ", "_"))
    os.makedirs(event_results_dir, exist_ok=True)
    
    print(f"\nTraining single event: {event}")
    print(f"Start time: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    
    try:
        cv_results = MinimalIMUClassifier.train_minimal_imu_model(
            out_root=out_root,
            models_dir=models_root,
            event=event,
            imu_subset=imu_subset,
            results_dir=event_results_dir,
            k_folds=5,
            rf_n_jobs=rf_n_jobs,
            loading_n_jobs=loading_n_jobs
        )
        
        print(f"\nEnd time: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"Training completed successfully for {event}!")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
