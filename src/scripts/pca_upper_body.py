import os
import json
import pandas as pd
import numpy as np, csv, json
import matplotlib.pyplot as plt
from gias3.learning.PCA import PCA


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
    def load_top_features(results_dir, datatype, event_condition, top_k=10):
        """
        Load the top k features from the prevalence-based feature selection results.
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
        
        # Sort features by importance (descending) and take top k
        sorted_features = sorted(features_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature_name for feature_name, _ in sorted_features[:top_k]]
        
        print(f"Loaded top {len(top_features)} features for {datatype} - {event_condition}")
        return top_features

    @staticmethod
    def load_features_for_event_condition(out_root, datatype, event, condition, participants, return_groups=True):
        """
        Load features for a specific event-condition combination by filtering files based on filename.
        Only loads files that contain the specific condition in the filename.
        """
        print(f"Loading features for {event} - {condition}")
        print(f"Participants: {participants}")
        
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
                        # Format: P001_Combination_AR_Stair down 1_Head_X.csv
                        if f"_{condition}_" in file:
                            feature_file = os.path.join(imu_path, file)
                            label_file = os.path.join(imu_path, file.replace('_X.csv', '_y.csv'))
                            
                            if os.path.exists(feature_file) and os.path.exists(label_file):
                                # Load feature data
                                try:
                                    features_df = pd.read_csv(feature_file)
                                    labels_df = pd.read_csv(label_file)
                                    
                                    # Extract only the 'label' column from labels_df
                                    if 'label' in labels_df.columns:
                                        labels = labels_df['label'].values
                                    else:
                                        # Fallback: assume the last column is the label
                                        labels = labels_df.iloc[:, -1].values
                                    
                                    # Ensure labels and features have the same length
                                    if len(features_df) != len(labels):
                                        print(f"Warning: Length mismatch in {file}: features={len(features_df)}, labels={len(labels)}")
                                    
                                    # Add to collections
                                    all_features.append(features_df)
                                    all_labels.extend(labels)
                                    all_groups.extend([participant] * len(features_df))
                                    
                                    print(f"Loaded: {file} ({len(features_df)} samples)")
                                    
                                except Exception as e:
                                    print(f"Error loading {feature_file}: {e}")
                                    continue
        
        if not all_features:
            raise ValueError(f"No feature files found for {event} - {condition}")
        
        # Combine all features
        X_combined = pd.concat(all_features, ignore_index=True)
        y_combined = np.array(all_labels)
        groups_combined = np.array(all_groups)
        
        print(f"Total samples loaded for {event} - {condition}: {len(X_combined)}")
        print(f"Participants in data: {np.unique(groups_combined)}")
        
        if return_groups:
            return X_combined, y_combined, groups_combined
        else:
            return X_combined, y_combined

    @staticmethod
    def load_event_condition_data(out_root, datatype, event_condition, participants, top_features, 
                                  filter_event_only=True):
        """
        Load and filter IMU feature data for a specific event-condition combination.
        """
        print(f"Loading data for {datatype} - {event_condition}")
        print(f"Participants: {participants}")
        print(f"Number of features to load: {len(top_features)}")
        
        # Parse event and condition from event_condition
        parts = event_condition.split()
        condition = parts[-1]  # Last part is condition (AR, VR, Normal)
        event = " ".join(parts[:-1])  # Everything except last part is event name
        
        # Load features filtered by condition
        X_all, y_all, groups = NormativePCAModel.load_features_for_event_condition(
            out_root, datatype, event, condition, participants, return_groups=True
        )
        
        # Filter to only include the available top features
        available_features = [f for f in top_features if f in X_all.columns]
        missing_features = [f for f in top_features if f not in X_all.columns]
        
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
        X_filtered = X_all[available_features].copy()
        
        if filter_event_only:
            # Filter to only keep event windows (label = 1)
            event_mask = (y_all == 1)
            X_filtered = X_filtered[event_mask]
            y_filtered = y_all[event_mask]
            groups_filtered = groups[event_mask]
            
            print(f"After filtering for event windows only:")
            print(f"  Samples: {len(X_filtered)} (from {len(X_all)} total)")
            print(f"  Event windows: {np.sum(y_filtered)} / {len(y_filtered)}")
        else:
            y_filtered = y_all
            groups_filtered = groups
        
        print(f"Final data shape: {X_filtered.shape}")
        print(f"Participants in filtered data: {len(np.unique(groups_filtered))}")
        
        return X_filtered, y_filtered, groups_filtered, available_features

    @staticmethod
    def create_normative_pca_model(out_root, datatype, event_condition, results_dir, n_components=None, 
                                   top_k_features=10, exclude_participants=None):
        """
        Create a normative PCA model using participant data for specific event-condition.
        """
        print(f"Creating normative PCA model for {datatype} - {event_condition}")
        
        # Parse event and condition from event_condition
        parts = event_condition.split()
        condition = parts[-1]  # Last part is condition (AR, VR, Normal)
        event = " ".join(parts[:-1])  # Everything except last part is event name
        event_filename = event.replace(' ', '_')
        
        # Create organized directory structure: pca_models/Event/Condition
        pc_model_dir = os.path.join(results_dir, "pca_models", event_filename, condition)
        os.makedirs(pc_model_dir, exist_ok=True)
        
        # Create output filename (simplified since it's in organized folders)
        outname = f"{datatype}_{event_filename}_{condition}_top{top_k_features}_pca"
        
        # Load top features from prevalence-based feature selection
        top_features = NormativePCAModel.load_top_features(results_dir, datatype, event_condition, top_k_features)
        
        # Get available participants for this event-condition
        available_data = NormativePCAModel.get_available_event_conditions(out_root, datatype)
        if event_condition not in available_data:
            raise ValueError(f"Event-condition '{event_condition}' not found in available data")
        
        all_participants = available_data[event_condition]
        
        # Exclude specified participants if any
        if exclude_participants:
            participants = [p for p in all_participants if p not in exclude_participants]
            print(f"Excluding participants: {exclude_participants}")
        else:
            participants = all_participants
        
        # Calculate n_components as n_participants - 1
        n_participants = len(participants)
        if n_components is None:
            n_components = n_participants - 1
        
        print(f"Using {n_participants} participants for normative model: {participants}")
        print(f"Number of components to compute: {n_components}")
        
        # Load and filter data (only event windows)
        X_filtered, y_filtered, groups_filtered, feature_names = NormativePCAModel.load_event_condition_data(
            out_root, datatype, event_condition, participants, top_features, filter_event_only=True
        )
        
        if len(X_filtered) == 0:
            raise ValueError("No event windows found for the specified participants")
        
        # Create PCA model
        pca = PCA()
        pca.setData(X_filtered.T)  # Transpose to (features, samples) format
        pca.inc_svd_decompose(n_components)
        pc = pca.PC

        # Explained variance
        if hasattr(pc, 'explained_variance_ratio_') and pc.explained_variance_ratio_ is not None:
            ratio = np.array(pc.explained_variance_ratio_, dtype=float)
            expl = np.array(pc.explained_variance_, dtype=float)
        else:
            ratio = np.array(pc.getNormSpectrum(), dtype=float)
            expl = ratio * ratio.sum()  # Calculate absolute variance from ratio
        
        cum = ratio.cumsum()

        # Save explained variance to CSV
        csv_path = os.path.join(pc_model_dir, f"{outname}_explained_variance.csv")
        with open(csv_path, 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Component", "Explained Variance", "Explained Variance Ratio", "Cumulative Variance"])
            for i, (e, r, c) in enumerate(zip(expl, ratio, cum), start=1):
                writer.writerow([i, e, r, c])
        print(f"Explained variance saved to {csv_path}")

        # Save mean and PC loadings with feature names
        np.savetxt(os.path.join(pc_model_dir, f"{outname}_mean.csv"), pc.mean, delimiter=",", 
                   header="feature_means", comments="")
        
        # Save loadings with feature names as header
        loadings_df = pd.DataFrame(pc.modes, 
                                   index=feature_names, 
                                   columns=[f"PC{i+1}" for i in range(pc.modes.shape[1])])
        loadings_df.to_csv(os.path.join(pc_model_dir, f"{outname}_loadings.csv"))

        # Save projected weights (scores)
        scores = None
        if pc.projectedWeights is not None:
            scores = pc.projectedWeights.T  # Transpose to (samples, components)
            scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
            scores_df.to_csv(os.path.join(pc_model_dir, f"{outname}_scores.csv"), index=False)

        # PCA plots
        k_plot = len(ratio)
        
        # Bar plot of explained variance %
        plt.figure(figsize=(10, 6))
        plt.bar([f"PC{i}" for i in range(1, k_plot + 1)], ratio * 100)
        plt.ylabel("Explained Variance (%)")
        plt.xlabel("Principal Component")
        plt.title(f"Explained Variance - {event_condition} (top{top_k_features})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(pc_model_dir, f"{outname}_explained_variance.png"), dpi=300)
        plt.close()
        
        # Cumulative variance plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, k_plot + 1), cum * 100, marker='o', linewidth=2, markersize=6)
        plt.ylabel("Cumulative Explained Variance (%)")
        plt.xlabel("Number of Principal Components")
        plt.title(f"Cumulative Explained Variance - {event_condition} (top{top_k_features})")
        plt.grid(True, linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(pc_model_dir, f"{outname}_cumulative_explained_variance.png"), dpi=300)
        plt.close()
        
        # PC1 vs PC2 scores scatter plot (if available)
        if scores is not None and scores.shape[1] >= 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(scores[:, 0], scores[:, 1], s=12, alpha=0.6)
            plt.xlabel("PC1 Score")
            plt.ylabel("PC2 Score")
            plt.title(f"PC1 vs PC2 Scores - {event_condition} (top{top_k_features})")
            plt.grid(True, linewidth=0.5, alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(pc_model_dir, f"{outname}_pc1_pc2_scatter.png"), dpi=300)
            plt.close()

        # Save PCA model files
        pc.save(os.path.join(pc_model_dir, outname))
        pc.savemat(os.path.join(pc_model_dir, f"{outname}.mat"))
        print(f"Done {outname}")
        
        # Return performance metrics for comparison
        return {
            'event_condition': event_condition,
            'n_components': len(ratio),
            'explained_variance_ratio': ratio,
            'cumulative_variance': cum,
            'total_variance_explained': cum[-1] if len(cum) > 0 else 0,
            'n_samples': X_filtered.shape[0],
            'n_features': X_filtered.shape[1]
        }

    @staticmethod
    def compare_models_performance(performance_results, results_dir):
        """
        Compare PCA models across event-condition combinations by variance explained.
        Creates separate comparison plots for each event.
        """
        if not performance_results:
            return
        
        # Use pca_models directory for comparisons too
        pca_models_dir = os.path.join(results_dir, "pca_models")
        
        # Group results by event
        events_data = {}
        for result in performance_results:
            event_condition = result['event_condition']
            parts = event_condition.split()
            event = " ".join(parts[:-1])  # Event name
            condition = parts[-1]         # Condition (AR, VR, Normal)
            
            if event not in events_data:
                events_data[event] = []
            
            events_data[event].append({
                'condition': condition,
                'total_variance_explained': result['total_variance_explained'],
                'event_condition': event_condition,
                'result': result
            })
        
        # Create separate plots for each event (save in event folder)
        for event, conditions_data in events_data.items():
            event_filename = event.replace(" ", "_").replace("/", "_")
            event_dir = os.path.join(pca_models_dir, event_filename)
            
            # Sort conditions for consistent ordering
            condition_order = ['Normal', 'AR', 'VR']
            conditions_data.sort(key=lambda x: condition_order.index(x['condition']) 
                               if x['condition'] in condition_order else 999)
            
            conditions = [cd['condition'] for cd in conditions_data]
            total_var_explained = [cd['total_variance_explained'] for cd in conditions_data]
            
            # Create individual event comparison plot (saved in event folder)
            plt.figure(figsize=(10, 6))
            bars = plt.bar(conditions, [v*100 for v in total_var_explained], 
                          color=['skyblue', 'lightcoral', 'lightgreen'])
            plt.ylabel("Total Variance Explained (%)")
            plt.xlabel("Condition")
            plt.title(f"PCA Model Performance Comparison - {event}")
            plt.ylim(0, 100)
            
            # Add value labels on bars
            for bar, val in zip(bars, total_var_explained):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(event_dir, f"pca_comparison_{event_filename}.png"), dpi=300)
            plt.close()
            
            print(f"Created comparison plot for {event}")
        
        # Create overall comparison plot (save at pca_models root)
        event_conditions = [r['event_condition'] for r in performance_results]
        total_var_explained_all = [r['total_variance_explained'] for r in performance_results]
        
        plt.figure(figsize=(18, 8))
        bars = plt.bar(event_conditions, [v*100 for v in total_var_explained_all])
        plt.ylabel("Total Variance Explained (%)")
        plt.xlabel("Event-Condition Combination")
        plt.title("PCA Model Performance Comparison - All Event-Conditions")
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, total_var_explained_all):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(pca_models_dir, "pca_model_comparison_all.png"), dpi=300)
        plt.close()
        
        # Save CSV summaries at pca_models root
        # Save detailed comparison CSV (grouped by event)
        comparison_data = []
        for event, conditions_data in events_data.items():
            for cd in conditions_data:
                result = cd['result']
                comparison_data.append({
                    'Event': event,
                    'Condition': cd['condition'],
                    'Event_Condition': cd['event_condition'],
                    'N_Components': result['n_components'],
                    'Total_Variance_Explained': f"{result['total_variance_explained']:.4f}",
                    'Total_Variance_Explained_Pct': f"{result['total_variance_explained']*100:.2f}%",
                    'N_Samples': result['n_samples'],
                    'N_Features': result['n_features'],
                    'PC1_Variance': f"{result['explained_variance_ratio'][0]:.4f}" if len(result['explained_variance_ratio']) > 0 else "N/A",
                    'PC2_Variance': f"{result['explained_variance_ratio'][1]:.4f}" if len(result['explained_variance_ratio']) > 1 else "N/A"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(os.path.join(pca_models_dir, "pca_model_comparison.csv"), index=False)
        
        # Create summary table by event
        summary_data = []
        for event, conditions_data in events_data.items():
            conditions_dict = {cd['condition']: cd['total_variance_explained'] for cd in conditions_data}
            summary_data.append({
                'Event': event,
                'Normal_Variance_Pct': f"{conditions_dict.get('Normal', 0)*100:.2f}%" if 'Normal' in conditions_dict else "N/A",
                'AR_Variance_Pct': f"{conditions_dict.get('AR', 0)*100:.2f}%" if 'AR' in conditions_dict else "N/A", 
                'VR_Variance_Pct': f"{conditions_dict.get('VR', 0)*100:.2f}%" if 'VR' in conditions_dict else "N/A",
                'Best_Condition': max(conditions_data, key=lambda x: x['total_variance_explained'])['condition'],
                'Worst_Condition': min(conditions_data, key=lambda x: x['total_variance_explained'])['condition']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(pca_models_dir, "pca_event_summary.csv"), index=False)
        
        print(f"Model comparison plots and summaries saved to {pca_models_dir}")
        print(f"Created {len(events_data)} individual event comparison plots")
        

if __name__ == "__main__":
    # Set up paths
    out_root = "Z:/Upper Body/Results/10 Participants/features"
    results_dir = "Z:/Upper Body/Results/10 Participants/models"
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Parameters
    datatype = "IMU"
    n_components = None  # Automatically set to n_participants - 1
    top_k_features = 10
    
    # Get all available event-condition combinations
    available_data = NormativePCAModel.get_available_event_conditions(out_root, datatype)
    print("Available event-condition combinations:")
    for event_condition, participants in available_data.items():
        print(f"  {event_condition}: {len(participants)} participants")
    
    # TEST MODE: Limit to first 2 event-conditions for initial test
    test_mode = True
    if test_mode:
        available_data = dict(list(available_data.items())[:2])
        print("TEST MODE: Processing only first 2 event-conditions")
    
    # Store performance results for comparison
    performance_results = []
    
    # Process all event-condition combinations
    for event_condition in available_data.keys():
        participants = available_data[event_condition]
        print(f"\n" + "="*50)
        print(f"Processing event-condition: {event_condition}")
        print(f"Available participants: {participants}")
        
        try:
            # Create normative PCA model
            perf_result = NormativePCAModel.create_normative_pca_model(
                out_root, datatype, event_condition, results_dir, 
                n_components=n_components, top_k_features=top_k_features
            )
            
            performance_results.append(perf_result)
            print(f"Successfully processed {event_condition}")
            
        except Exception as e:
            print(f"Error processing {event_condition}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare model performances
    NormativePCAModel.compare_models_performance(performance_results, results_dir)
    
    print(f"\n" + "="*50)
    print("PCA model creation completed for all event-condition combinations.")
    print(f"Performance comparison saved to {results_dir}")
