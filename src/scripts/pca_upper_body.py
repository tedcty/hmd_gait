import os
import json
import pandas as pd
import numpy as np, csv, json
import matplotlib.pyplot as plt
from gias3.learning.PCA import PCA
from upper_body_classifier import UpperBodyClassifier


class NormativePCAModel:
    
    @staticmethod
    def load_top_features(results_dir, datatype, event, top_k=10):
        """
        Load the top k features from the prevalence-based feature selection results.
        """
        # Fix filename format to handle spaces in event names
        event_filename = event.replace(' ', '_')
        feature_file = os.path.join(
            results_dir, 
            f"{datatype}_{event_filename}_prevalence_top100_features.json"
        )
        
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Feature selection file not found: {feature_file}")
        
        with open(feature_file, 'r') as f:
            features_dict = json.load(f)
        
        # Sort features by importance (descending) and take top k
        sorted_features = sorted(features_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature_name for feature_name, _ in sorted_features[:top_k]]
        
        print(f"Loaded top {len(top_features)} features for {datatype} - {event}")
        return top_features
    
    @staticmethod
    def load_event_condition_data(out_root, datatype, event, participants, top_features, 
                                  filter_event_only=True):
        """
        Load and filter IMU feature data for a specific event-condition combination.
        """
        print(f"Loading data for {datatype} - {event}")
        print(f"Participants: {participants}")
        print(f"Number of features to load: {len(top_features)}")
        
        # Load all features for the event using the existing function
        X_all, y_all, groups = UpperBodyClassifier.load_features_for_participants(
            out_root, datatype, event, participants, return_groups=True
        )
        
        # Filter to only include the top features
        # Check which features are actually available
        available_features = [f for f in top_features if f in X_all.columns]
        missing_features = [f for f in top_features if f not in X_all.columns]
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features not found in data:")
            for feature in missing_features[:5]:  # Show first 5 missing features
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
    def get_available_events_and_participants(out_root, datatype):
        """
        Get all available events and participants for a given datatype.
        """
        base_dir = os.path.join(out_root, datatype)
        
        if not os.path.exists(base_dir):
            return {}
        
        events_participants = {}
        
        for event_dir in os.listdir(base_dir):
            event_path = os.path.join(base_dir, event_dir)
            if os.path.isdir(event_path):
                # Convert directory name back to event name (replace _ with space)
                event_name = event_dir.replace("_", " ")
                
                participants = []
                for participant_dir in os.listdir(event_path):
                    participant_path = os.path.join(event_path, participant_dir)
                    if os.path.isdir(participant_path):
                        participants.append(participant_dir)
                
                if participants:
                    events_participants[event_name] = sorted(participants)
        
        return events_participants

    @staticmethod
    def create_normative_pca_model(out_root, datatype, event, results_dir, n_components=None, 
                                   top_k_features=10, exclude_participants=None):
        """
        Create a normative PCA model using participant data.
        """
        print(f"Creating normative PCA model for {datatype} - {event}")
        
        # Create output filename and directory
        outname = f"{datatype}_{event.replace(' ', '_')}_top{top_k_features}_pca"
        pc_model_dir = os.path.join(results_dir, "pc_model")
        os.makedirs(pc_model_dir, exist_ok=True)
        
        # Load top features from prevalence-based feature selection
        top_features = NormativePCAModel.load_top_features(results_dir, datatype, event, top_k_features)
        
        # Get available participants
        available_data = NormativePCAModel.get_available_events_and_participants(out_root, datatype)
        if event not in available_data:
            raise ValueError(f"Event '{event}' not found in available data")
        
        all_participants = available_data[event]
        
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
            out_root, datatype, event, participants, top_features, filter_event_only=True
        )
        
        if len(X_filtered) == 0:
            raise ValueError("No event windows found for the specified participants")
        
        # PCA Model
        pca = PCA()
        pca.setData(X_filtered.T)  # transposes (samples, features) to (features, samples)
        pca.inc_svd_decompose(n_components)
        pc = pca.PC

        # Explained variance
        if hasattr(pc, 'explained_variance_ratio_') and pc.explained_variance_ratio_ is not None:
            ratio = np.array(pc.explained_variance_ratio_, dtype=float)
            expl = np.array(pc.explained_variance_, dtype=float)
        else:
            ratio = np.array(pc.getNormSpectrum(), dtype=float)
            expl = ratio * ratio.sum()  # if absolute variance not available
        
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
            scores = pc.projectedWeights.T  # (samples, components)
            scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
            scores_df.to_csv(os.path.join(pc_model_dir, f"{outname}_scores.csv"), index=False)

        # PCA plots
        k_plot = len(ratio)
        
        # Bar plot of explained variance %
        plt.figure(figsize=(10, 6))
        plt.bar([f"PC{i}" for i in range(1, k_plot + 1)], ratio * 100)
        plt.ylabel("Explained Variance (%)")
        plt.xlabel("Principal Component")
        plt.title(f"Explained Variance - {event} (top{top_k_features})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(pc_model_dir, f"{outname}_explained_variance.png"), dpi=300)
        plt.close()
        
        # Cumulative variance plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, k_plot + 1), cum * 100, marker='o', linewidth=2, markersize=6)
        plt.ylabel("Cumulative Explained Variance (%)")
        plt.xlabel("Number of Principal Components")
        plt.title(f"Cumulative Explained Variance - {event} (top{top_k_features})")
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
            plt.title(f"PC1 vs PC2 Scores - {event} (top{top_k_features})")
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
            'event': event,
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
        Compare PCA models across events by variance explained.
        """
        if not performance_results:
            return
        
        # Create comparison plots
        events = [r['event'] for r in performance_results]
        total_var_explained = [r['total_variance_explained'] for r in performance_results]
        
        # Bar plot comparing total variance explained
        plt.figure(figsize=(12, 6))
        bars = plt.bar(events, [v*100 for v in total_var_explained])
        plt.ylabel("Total Variance Explained (%)")
        plt.xlabel("Event")
        plt.title("PCA Model Performance Comparison")
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, total_var_explained):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val*100:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "pca_model_comparison.png"), dpi=300)
        plt.close()
        
        # Save comparison summary
        comparison_df = pd.DataFrame([
            {
                'Event': r['event'],
                'N_Components': r['n_components'],
                'Total_Variance_Explained': f"{r['total_variance_explained']:.4f}",
                'Total_Variance_Explained_Pct': f"{r['total_variance_explained']*100:.2f}%",
                'N_Samples': r['n_samples'],
                'N_Features': r['n_features'],
                'PC1_Variance': f"{r['explained_variance_ratio'][0]:.4f}" if len(r['explained_variance_ratio']) > 0 else "N/A",
                'PC2_Variance': f"{r['explained_variance_ratio'][1]:.4f}" if len(r['explained_variance_ratio']) > 1 else "N/A"
            }
            for r in performance_results
        ])
        
        comparison_df.to_csv(os.path.join(results_dir, "pca_model_comparison.csv"), index=False)
        print(f"Model comparison saved to {results_dir}")


if __name__ == "__main__":
    # Set up paths
    out_root = "Z:/Upper Body/Results/10 Participants/features"
    results_dir = "Z:/Upper Body/Results/10 Participants/models"
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Parameters
    datatype = "IMU"
    n_components = None  # Will automatically set to n_participants - 1
    top_k_features = 10
    
    # Get all available events and participants
    available_data = NormativePCAModel.get_available_events_and_participants(out_root, datatype)
    print("Available events and participants:")
    for event_name, participants in available_data.items():
        print(f"  {event_name}: {len(participants)} participants")
    
    # Store performance results for comparison
    performance_results = []
    
    # Process ALL events
    for event in available_data.keys():
        participants = available_data[event]
        print(f"\n")
        print(f"Processing event: {event}")
        print(f"Available participants: {participants}")
        
        try:
            # Create normative PCA model
            perf_result = NormativePCAModel.create_normative_pca_model(
                out_root, datatype, event, results_dir, 
                n_components=n_components, top_k_features=top_k_features
            )
            
            performance_results.append(perf_result)
            print(f"Successfully processed {event}")
            
        except Exception as e:
            print(f"Error processing {event}: {e}")
            import traceback
            traceback.print_exc()
            continue  # Continue with next event
    
    # Compare model performances
    NormativePCAModel.compare_models_performance(performance_results, results_dir)
    
    print(f"\n")
    print("PCA model creation completed for all events.")
    print(f"Performance comparison saved to {results_dir}")
