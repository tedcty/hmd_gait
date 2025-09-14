import os
import json
import pandas as pd
import numpy as np
from gias3.learning.PCA import PCA
from upper_body_classifier import UpperBodyClassifier


class NormativePCAModel:
    
    @staticmethod
    def load_top_features(results_dir, datatype, event, top_k=10):
        """
        Load the top k features from the prevalence-based feature selection results.
        """
        feature_file = os.path.join(
            results_dir, 
            f"{datatype}_{event}_prevalence_top100_features.json"
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
        
        return X_filtered, y_filtered, groups_filtered
    
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
        
        print(f"Using {len(participants)} participants for normative model: {participants}")
        
        # Load and filter data (only event windows)
        X_filtered, y_filtered, groups_filtered = NormativePCAModel.load_event_condition_data(
            out_root, datatype, event, participants, top_features, filter_event_only=True
        )
        
        if len(X_filtered) == 0:
            raise ValueError("No event windows found for the specified participants")
        
        # PCA Model
        pca = PCA()
        pca.setData(X_filtered.T)  # transposes (samples, features) to (features, samples)
        pca.inc_svd_decompose(n_components)  # NOTE: should I have n_components or None?
        pc = pca.PC

        # Create output filename and directory
        outname = f"{datatype}_{event.replace(' ', '_')}_top{top_k_features}_pca"
        pc_model_dir = os.path.join(results_dir, "pc_model")
        
        if not os.path.exists(pc_model_dir):
            os.makedirs(pc_model_dir)
        
        # Save PCA model files
        pc.save(os.path.join(pc_model_dir, outname))
        pc.savemat(os.path.join(pc_model_dir, f"{outname}.mat"))
        print(f"Done {outname}")
        
        return model_data


if __name__ == "__main__":
    # Set up paths
    out_root = "Z:/Upper Body/Results/10 Participants/features"
    results_dir = "Z:/Upper Body/Results/10 Participants/models"
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Parameters
    datatype = "IMU"
    n_components = 5
    top_k_features = 10
    
    # Get all available events and participants
    available_data = NormativePCAModel.get_available_events_and_participants(out_root, datatype)
    print("Available events and participants:")
    for event_name, participants in available_data.items():
        print(f"  {event_name}: {len(participants)} participants")
    
    # Process ALL events
    for event in available_data.keys():
        participants = available_data[event]
        print(f"\n")
        print(f"Processing event: {event}")
        print(f"Available participants: {participants}")
        
        try:
            # Create event-specific results directory
            event_results_dir = os.path.join(results_dir, datatype, event.replace(" ", "_"))
            os.makedirs(event_results_dir, exist_ok=True)
            
            # Create normative PCA model
            model_data = NormativePCAModel.create_normative_pca_model(
                out_root, datatype, event, event_results_dir, 
                n_components=n_components, top_k_features=top_k_features
            )
            
            print(f"Successfully processed {event}")
            
        except Exception as e:
            print(f"Error processing {event}: {e}")
            import traceback
            traceback.print_exc()
            continue  # Continue with next event
    
    print(f"\n")
    print("PCA model creation completed for all events.")
