import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gias3.learning.PCA import PCA
import traceback


class NormativePCAModel:
    
    @staticmethod
    def get_available_event_conditions(out_root, datatype):
        """
        Get all available event-condition combinations for a given datatype.
        """
        base_dir = os.path.join(out_root, datatype)
        
        if not os.path.exists(base_dir):
            return {}
        
        event_conditions = {}
        
        for event_dir in os.listdir(base_dir):
            event_path = os.path.join(base_dir, event_dir)
            if os.path.isdir(event_path):
                # Convert directory name back to event name (replace _ with space)
                event_name = event_dir.replace("_", " ")
                
                # Find all condition combinations for this event
                conditions_found = set()
                participants = {}
                
                for participant_dir in os.listdir(event_path):
                    participant_path = os.path.join(event_path, participant_dir)
                    if os.path.isdir(participant_path):
                        
                        # Look through all IMU locations for this participant
                        for imu_location in os.listdir(participant_path):
                            imu_path = os.path.join(participant_path, imu_location)
                            if os.path.isdir(imu_path):
                                
                                # Check CSV files to extract conditions
                                for file in os.listdir(imu_path):
                                    if file.endswith('_X.csv'):
                                        # Extract condition from filename
                                        # Format: P001_Combination_AR_Stair down 1_Head_X.csv
                                        parts = file.replace('.csv', '').split('_')
                                        
                                        if len(parts) >= 3:
                                            # Extract condition (AR, VR, Normal)
                                            condition = parts[2]  # AR, VR, etc.
                                            
                                            # Create event-condition combination
                                            event_condition = f"{event_name} {condition}"
                                            conditions_found.add(event_condition)
                                            
                                            if event_condition not in participants:
                                                participants[event_condition] = set()
                                            participants[event_condition].add(participant_dir)
                
                # Convert sets to sorted lists
                for event_condition in conditions_found:
                    if event_condition in participants:
                        event_conditions[event_condition] = sorted(list(participants[event_condition]))
        
        return event_conditions

    @staticmethod
    def get_cv_folds(participants, k=5, folds_json_path=None):
        """
        Get k-fold cross-validation splits for participants.
        
        If folds_json_path is provided and exists, load pre-defined folds from JSON.
        """
        if folds_json_path and os.path.exists(folds_json_path):
            print(f"Loading CV folds from {folds_json_path}")
            with open(folds_json_path, 'r') as f:
                folds_dict = json.load(f)
            
            # Convert to expected format
            folds = []
            all_participants = set(participants)
            for fold_name in sorted(folds_dict.keys()):
                test_participants = [p for p in folds_dict[fold_name] if p in all_participants]
                train_participants = [p for p in all_participants if p not in test_participants]
                folds.append({
                    'train': train_participants,
                    'test': test_participants
                })
            
            print(f"Loaded {len(folds)} folds from JSON")
            return folds
        
        # Deterministic round-robin split
        print(f"Creating deterministic {k}-fold split")
        sorted_participants = sorted(participants)
        folds = [{'train': [], 'test': []} for _ in range(k)]
        
        # Assign participants to folds round-robin
        for i, participant in enumerate(sorted_participants):
            fold_idx = i % k
            folds[fold_idx]['test'].append(participant)
        
        # Fill in training sets (all participants except test)
        for i in range(k):
            test_set = set(folds[i]['test'])
            folds[i]['train'] = [p for p in sorted_participants if p not in test_set]
        
        return folds

    @staticmethod
    def load_top_features(results_dir, datatype, event_condition, top_k=100):
        """
        Load the top 100 features from the prevalence-based feature selection results.
        """
        # Parse event and condition from event_condition
        parts = event_condition.split()
        condition = parts[-1]  # Last part is condition (AR, VR, Normal)
        event = " ".join(parts[:-1])  # Everything except last part is event name
        
        # Format paths to match the directory structure
        event_filename = event.replace(' ', '_')
        
        # Path structure: models/IMU/Event_name/top_100/IMU_Event_name_Condition_prevalence_top100_features.json
        feature_file = os.path.join(
            results_dir, 
            datatype,
            event_filename,
            "top_100",
            f"{datatype}_{event}_prevalence_top100_features.json"
        )
        
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Feature selection file not found: {feature_file}")
        
        with open(feature_file, 'r') as f:
            features_dict = json.load(f)
        
        # Sort features by importance (descending) and take top 100
        sorted_features = sorted(features_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature_name for feature_name, _ in sorted_features[:100]]
        
        print(f"Loaded top 100 features for {datatype} - {event_condition}")
        return top_features

    @staticmethod
    def load_features_for_event_condition(out_root, datatype, event, condition, participants, top_features=None, return_groups=True):
        """
        Load features for a specific event-condition combination by filtering files based on filename.
        Only loads files that contain the specific condition in the filename.
        """
        print(f"Loading features for {event} - {condition}")
        print(f"Participants: {participants}")
        if top_features is not None:
            print(f"Loading only top 100 features")
        
        all_features = []
        all_labels = []
        all_groups = []
        
        # Build the path to the event directory
        event_dir = event.replace(" ", "_")
        event_path = os.path.join(out_root, datatype, event_dir)
        
        if not os.path.exists(event_path):
            raise FileNotFoundError(f"Event directory not found: {event_path}")
        
        for participant in participants:
            participant_path = os.path.join(event_path, participant)
            
            if not os.path.exists(participant_path):
                print(f"Warning: Participant directory not found: {participant_path}")
                continue
            
            # Go through each IMU location for this participant
            for imu_location in os.listdir(participant_path):
                imu_path = os.path.join(participant_path, imu_location)
                
                if not os.path.isdir(imu_path):
                    continue
                
                # Look for files that contain the specific condition in the filename
                for file in os.listdir(imu_path):
                    if file.endswith('_X.csv'):
                        # Check if this file contains our condition
                        if f"_{condition}_" in file:
                            feature_file = os.path.join(imu_path, file)
                            label_file = os.path.join(imu_path, file.replace('_X.csv', '_y.csv'))
                            
                            if os.path.exists(feature_file) and os.path.exists(label_file):
                                try:
                                    # Load feature data
                                    features_df = pd.read_csv(feature_file)
                                    
                                    # Filter to only top features if specified
                                    if top_features is not None:
                                        # Only keep columns that exist in both the file and top_features list
                                        available_features = [f for f in top_features if f in features_df.columns]
                                        if len(available_features) == 0:
                                            print(f"Warning: No top features found in {file}")
                                            continue
                                        features_df = features_df[available_features]
                                    
                                    labels_df = pd.read_csv(label_file)
                                    
                                    # Extract only the 'label' column from labels_df
                                    if 'label' in labels_df.columns:
                                        labels = labels_df['label'].values
                                    else:
                                        labels = labels_df.iloc[:, -1].values
                                    
                                    # Ensure labels and features have the same length
                                    if len(features_df) != len(labels):
                                        print(f"Warning: Length mismatch in {file}: features={len(features_df)}, labels={len(labels)}")
                                        min_len = min(len(features_df), len(labels))
                                        features_df = features_df.iloc[:min_len]
                                        labels = labels[:min_len]
                                    
                                    # Add to collections
                                    all_features.append(features_df)
                                    all_labels.extend(labels)
                                    all_groups.extend([participant] * len(features_df))
                                    
                                except Exception as e:
                                    print(f"Error loading {feature_file}: {e}")
                                    continue
        
        if not all_features:
            raise ValueError(f"No feature files found for {event} - {condition}")
        
        # Combine all features
        X_combined = pd.concat(all_features, ignore_index=True)
        y_combined = np.array(all_labels)
        groups_combined = np.array(all_groups)
        
        # Handle NaN values by filling with zeros
        nan_count = X_combined.isna().sum().sum()
        if nan_count > 0:
            X_combined = X_combined.fillna(0)
        
        print(f"Total samples loaded for {event} - {condition}: {len(X_combined)}")
        print(f"Participants in data: {np.unique(groups_combined)}")
        
        if return_groups:
            return X_combined, y_combined, groups_combined
        else:
            return X_combined, y_combined

    @staticmethod
    def load_event_condition_data(out_root, datatype, event_condition, participants, top_features, filter_event_only=False):
        """
        Load and filter data for a specific event-condition combination.
        """
        print(f"Loading data for {datatype} - {event_condition}")
        print(f"Participants: {participants}")
        print(f"Number of features to load: {len(top_features)}")
        
        # Parse event and condition
        parts = event_condition.split()
        condition = parts[-1]
        event = " ".join(parts[:-1])
        
        # Load features with top_features filtering applied during loading
        X, y, groups = NormativePCAModel.load_features_for_event_condition(
            out_root, datatype, event, condition, participants, 
            top_features=top_features, return_groups=True
        )
        
        # Filter to only include the available top features
        available_features = [f for f in top_features if f in X.columns]
        missing_features = [f for f in top_features if f not in X.columns]
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features not found in data:")
            for feature in missing_features[:5]:
                print(f"  - {feature}")
            if len(missing_features) > 5:
                print(f"  ... and {len(missing_features) - 5} more")
        
        if not available_features:
            raise ValueError("None of the specified top features are available in the data")
        
        print(f"Using {len(available_features)} out of {len(top_features)} requested features")
        
        # Select only the available top features
        X_filtered = X[available_features].copy()
        
        if filter_event_only:
            # Filter to only keep event windows (label = 1)
            event_mask = (y == 1)
            X_filtered = X_filtered[event_mask]
            y_filtered = y[event_mask]
            groups_filtered = groups[event_mask]
            
            print(f"After filtering for event windows only:")
            print(f"  Samples: {len(X_filtered)} (from {len(X)} total)")
            print(f"  Event windows: {np.sum(y_filtered)} / {len(y_filtered)}")
        else:
            y_filtered = y
            groups_filtered = groups
        
        print(f"Final data shape: {X_filtered.shape}")
        print(f"Participants in filtered data: {len(np.unique(groups_filtered))}")
        
        return X_filtered, y_filtered, groups_filtered, available_features

    @staticmethod
    def create_normative_pca_models_cv(out_root, datatype, event_condition, results_dir, 
                                       n_components=None, folds_json_path=None):
        """
        Create normative PCA models using 5-fold cross-validation across participants.
        
        For each fold:
        - Train PCA on 4/5 of participants (training set)
        - Project test participants' data into the PCA space
        - Save per-fold results and test projections
        
        After all folds:
        - Aggregate explained variance statistics across folds
        """
        print(f"Creating CV-based normative PCA models for {datatype} - {event_condition}")
        
        # Parse event and condition from event_condition
        parts = event_condition.split()
        condition = parts[-1]
        event = " ".join(parts[:-1])
        event_filename = event.replace(' ', '_')
        
        # Create CV results directory
        cv_dir = os.path.join(results_dir, "pca_models", event_filename, condition, "top100_cv")
        os.makedirs(cv_dir, exist_ok=True)
        
        # Load top 100 features
        top_features = NormativePCAModel.load_top_features(results_dir, datatype, event_condition, top_k=100)
        
        # Get available participants
        available_data = NormativePCAModel.get_available_event_conditions(out_root, datatype)
        if event_condition not in available_data:
            raise ValueError(f"Event-condition '{event_condition}' not found in available data")
        
        all_participants = available_data[event_condition]
        print(f"Total participants available: {len(all_participants)}")
        
        # Get CV folds
        folds = NormativePCAModel.get_cv_folds(all_participants, k=5, folds_json_path=folds_json_path)
        print(f"Created {len(folds)} folds for cross-validation")
        
        # Store results for each fold
        fold_results = []
        
        # Process each fold
        for fold_idx, fold in enumerate(folds):
            print(f"\n{'='*60}")
            print(f"Processing Fold {fold_idx}")
            print(f"{'='*60}")
            
            train_participants = fold['train']
            test_participants = fold['test']
            
            print(f"Training participants ({len(train_participants)}): {train_participants}")
            print(f"Test participants ({len(test_participants)}): {test_participants}")
            
            # Determine number of components for this fold
            if n_components is None:
                n_components_fold = len(train_participants) - 1
            else:
                n_components_fold = min(n_components, len(train_participants) - 1)
            
            print(f"Number of components for this fold: {n_components_fold}")
            
            try:
                # Load training data (event windows only)
                X_train, y_train, groups_train, feature_names = NormativePCAModel.load_event_condition_data(
                    out_root, datatype, event_condition, train_participants, top_features, filter_event_only=True
                )
                
                if len(X_train) == 0:
                    print(f"Warning: No training data for fold {fold_idx}")
                    continue
                
                # Fit PCA on training data
                pca = PCA()
                pca.setData(X_train.values.T)  # Transpose to (features, samples)
                pca.inc_svd_decompose(n_components_fold)
                pc = pca.PC
                
                # Extract explained variance
                if hasattr(pc, 'explained_variance_ratio_') and pc.explained_variance_ratio_ is not None:
                    evr = np.array(pc.explained_variance_ratio_, dtype=float)
                else:
                    evr = np.array(pc.getNormSpectrum(), dtype=float)
                
                cum_var = evr.cumsum()
                
                # Save fold-specific loadings
                loadings_df = pd.DataFrame(
                    pc.modes, 
                    index=feature_names, 
                    columns=[f"PC{i+1}" for i in range(pc.modes.shape[1])]
                )
                loadings_path = os.path.join(cv_dir, f"{datatype}_{event_filename}_{condition}_fold{fold_idx}_loadings.csv")
                loadings_df.to_csv(loadings_path)
                print(f"Saved loadings to {loadings_path}")
                
                # Save fold-specific explained variance
                ev_data = []
                for i in range(len(evr)):
                    ev_data.append({
                        'Component': i + 1,
                        'Explained_Variance_Ratio': evr[i],
                        'Cumulative_Variance': cum_var[i]
                    })
                ev_df = pd.DataFrame(ev_data)
                ev_path = os.path.join(cv_dir, f"{datatype}_{event_filename}_{condition}_fold{fold_idx}_explained_variance.csv")
                ev_df.to_csv(ev_path, index=False)
                print(f"Saved explained variance to {ev_path}")
                
                # Store fold results
                fold_results.append({
                    'fold': fold_idx,
                    'n_components': n_components_fold,
                    'evr': evr,
                    'cum_var': cum_var,
                    'n_train_samples': len(X_train),
                    'n_train_participants': len(train_participants)
                })
                
                # Project test data into PCA space
                if len(test_participants) > 0:
                    try:
                        X_test, y_test, groups_test, _ = NormativePCAModel.load_event_condition_data(
                            out_root, datatype, event_condition, test_participants, top_features, filter_event_only=True
                        )
                        
                        if len(X_test) > 0:
                            # Project test data
                            # GIAS3 PCA expects data as (features, samples)
                            test_scores = pc.project(X_test.values.T).T  # Result is (samples, components)
                            
                            # Save test projection scores
                            test_scores_df = pd.DataFrame(
                                test_scores,
                                columns=[f"PC{i+1}" for i in range(test_scores.shape[1])]
                            )
                            test_scores_df.insert(0, 'Participant', groups_test)
                            
                            test_proj_path = os.path.join(cv_dir, f"{datatype}_{event_filename}_{condition}_fold{fold_idx}_test_projection.csv")
                            test_scores_df.to_csv(test_proj_path, index=False)
                            print(f"Saved test projections to {test_proj_path}")
                            
                            # Compute reconstruction and variance metrics
                            # Reconstruct from PCA space: X_reconstructed = mean + (scores @ loadings.T)
                            X_reconstructed = pc.mean.reshape(-1, 1) + (pc.modes @ test_scores.T)
                            X_reconstructed = X_reconstructed.T  # Back to (samples, features)
                            
                            # Compute reconstruction error per sample
                            reconstruction_errors = np.sqrt(((X_test.values - X_reconstructed) ** 2).sum(axis=1))
                            

                            # Compute variance captured (squared norm of scores)
                            variance_captured = (test_scores ** 2).sum(axis=1)
                            
                            # Save variance metrics
                            variance_metrics = pd.DataFrame({
                                'Participant': groups_test,
                                'Reconstruction_Error': reconstruction_errors,
                                'Variance_Captured': variance_captured
                            })
                            
                            variance_path = os.path.join(cv_dir, f"{datatype}_{event_filename}_{condition}_fold{fold_idx}_test_variance_metrics.csv")
                            variance_metrics.to_csv(variance_path, index=False)
                            print(f"Saved variance metrics to {variance_path}")
                            
                            fold_results[-1]['n_test_samples'] = len(X_test)
                            fold_results[-1]['n_test_participants'] = len(test_participants)
                            fold_results[-1]['mean_reconstruction_error'] = reconstruction_errors.mean()
                            fold_results[-1]['mean_variance_captured'] = variance_captured.mean()
                        else:
                            print(f"Warning: No test data available for fold {fold_idx}")
                    
                    except Exception as e:
                        print(f"Error projecting test data for fold {fold_idx}: {e}")
                        traceback.print_exc()
                
            except Exception as e:
                print(f"Error processing fold {fold_idx}: {e}")
                traceback.print_exc()
                continue
        
        # Aggregate results across folds
        if not fold_results:
            print("Warning: No successful folds")
            return None
        
        print(f"\n{'='*60}")
        print("Aggregating results across folds")
        print(f"{'='*60}")
        
        # Find maximum number of components across folds
        max_components = max(result['n_components'] for result in fold_results)
        
        # Pad shorter EVR arrays with NaN
        evr_matrix = np.full((len(fold_results), max_components), np.nan)
        cum_matrix = np.full((len(fold_results), max_components), np.nan)
        
        for i, result in enumerate(fold_results):
            n_comp = len(result['evr'])
            evr_matrix[i, :n_comp] = result['evr']
            cum_matrix[i, :n_comp] = result['cum_var']
        
        # Compute mean and std, ignoring NaNs
        evr_mean = np.nanmean(evr_matrix, axis=0)
        evr_std = np.nanstd(evr_matrix, axis=0)
        cum_mean = np.nanmean(cum_matrix, axis=0)
        cum_std = np.nanstd(cum_matrix, axis=0)
        
        # Save aggregated explained variance
        agg_data = []
        for i in range(max_components):
            agg_data.append({
                'Component': i + 1,
                'EVR_mean': evr_mean[i],
                'EVR_sd': evr_std[i],
                'Cum_mean': cum_mean[i],
                'Cum_sd': cum_std[i]
            })
        
        agg_df = pd.DataFrame(agg_data)
        agg_path = os.path.join(cv_dir, f"{datatype}_{event_filename}_{condition}_cv_aggregated_explained_variance.csv")
        agg_df.to_csv(agg_path, index=False)
        print(f"Saved aggregated results to {agg_path}")
        
        # Create visualization of aggregated results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Explained variance with error bars
        components = np.arange(1, max_components + 1)
        ax1.errorbar(components, evr_mean * 100, yerr=evr_std * 100, 
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=6)
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Explained Variance (%)")
        ax1.set_title(f"Explained Variance (Mean ± SD)\n{event_condition} - 5-Fold CV")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative variance with error bars
        ax2.errorbar(components, cum_mean * 100, yerr=cum_std * 100,
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=6)
        ax2.set_xlabel("Number of Principal Components")
        ax2.set_ylabel("Cumulative Explained Variance (%)")
        ax2.set_title(f"Cumulative Explained Variance (Mean ± SD)\n{event_condition} - 5-Fold CV")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(cv_dir, f"{datatype}_{event_filename}_{condition}_cv_explained_variance.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved aggregated plot to {plot_path}")
        
        # Return summary
        return {
            'event_condition': event_condition,
            'n_folds': len(fold_results),
            'n_components_per_fold': [r['n_components'] for r in fold_results],
            'evr_mean': evr_mean,
            'evr_sd': evr_std,
            'cum_mean': cum_mean,
            'cum_sd': cum_std,
            'total_variance_explained': cum_mean[-1] if len(cum_mean) > 0 else 0,
            'fold_results': fold_results
        }


if __name__ == "__main__":
    # Set up paths
    out_root = "Z:/Upper Body/Results/30 Participants/features"
    results_dir = "Z:/Upper Body/Results/30 Participants/models"
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Parameters
    datatype = "IMU"
    n_components = None  # Automatically set to n_participants - 1
    
    print("Running in CROSS-VALIDATION mode")
    
    # Path to pre-defined folds (if available)
    folds_json_path = os.path.join(results_dir, "classifier_folds.json")
    if not os.path.exists(folds_json_path):
        folds_json_path = None
        print("No pre-defined folds found; will use deterministic split")
    
    # Get all available event-condition combinations
    available_data = NormativePCAModel.get_available_event_conditions(out_root, datatype)
    print("\nAvailable event-condition combinations:")
    for event_condition, participants in available_data.items():
        print(f"  {event_condition}: {len(participants)} participants")
    
    # Store performance results for comparison
    performance_results = []
    
    # Cross-validation mode
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION MODE: Training 5-fold CV PCA models")
    print(f"{'='*60}\n")
    
    for event_condition in available_data.keys():
        participants = available_data[event_condition]
        print(f"\n{'='*50}")
        print(f"Processing event-condition: {event_condition}")
        print(f"Available participants: {participants}")
        
        try:
            cv_result = NormativePCAModel.create_normative_pca_models_cv(
                out_root, datatype, event_condition, results_dir, 
                n_components=n_components, folds_json_path=folds_json_path
            )
            
            if cv_result:
                performance_results.append(cv_result)
                print(f"Successfully processed {event_condition} with CV")
            
        except Exception as e:
            print(f"Error processing {event_condition}: {e}")
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"CV-based PCA model creation completed for all event-condition combinations.")
    print(f"Results saved to {results_dir}/pca_models/")
