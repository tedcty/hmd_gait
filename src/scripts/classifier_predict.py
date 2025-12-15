import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from sklearn.metrics import confusion_matrix


class ClassifierDeviationAnalysis:
    """
    Perform event deviation analysis using trained Random Forest classifiers.
    Tests specificity: how well classifiers correctly reject a different event as non-event.
    """

    @staticmethod
    def load_classifier(results_dir, event, datatype="IMU", minimal_imu_set=False, minimal_results_dir=None):
        """
        Load a trained Random Forest classifier for a specific event.
        """
        event_filename = event.replace(' ', '_')
        
        # Determine model directory based on minimal IMU set flag
        if minimal_imu_set:
            if minimal_results_dir is None:
                raise ValueError("minimal_results_dir must be provided when minimal_imu_set=True")
            model_file = os.path.join(
                minimal_results_dir,
                event_filename,
                "product_usecase_right",
                "top_100",
                f"{datatype}_{event_filename}_final_product_usecase_right_top100_rf_model.pkl"
            )
        else:
            model_file = os.path.join(
                results_dir,
                datatype,
                event_filename,
                "top_100",
                f"{datatype}_{event_filename}_final_top100_rf_model.pkl"
            )
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Classifier model not found: {model_file}")
        
        print(f"Loading classifier: {model_file}")
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        return model

    @staticmethod
    def load_top_features(results_dir, event, datatype="IMU", minimal_results_dir=None):
        """
        Load the top 100 features used for training the classifier.
        """
        event_filename = event.replace(' ', '_')

        if minimal_results_dir:
            feature_file = os.path.join(
                minimal_results_dir,
                event_filename,
                "product_usecase_right",
                "top_100",
                f"{datatype}_{event}_prevalence_top100_features.json"
            )
        else:
            feature_file = os.path.join(
                results_dir,
                datatype,
                event_filename,
                "top_100",
                f"{datatype}_{event}_prevalence_top100_features.json"
            )

        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"Feature file not found: {feature_file}")

        with open(feature_file, 'r') as f:
            features_dict = json.load(f)

        return list(features_dict.keys())

    @staticmethod
    def load_test_data(out_root, datatype, event, condition, participants, top_features, minimal_imu_set=True):
        """
        Load test data for specified participants, event, and condition.
        Filters to event windows only, then relabels as non-event (0) for specificity testing.
        """
        event_dir = event.replace(" ", "_")
        event_path = os.path.join(out_root, datatype, event_dir)

        minimal_imus = {'Head_imu', 'RightForeArm_imu'} if minimal_imu_set else None

        all_features = []
        all_groups = []

        for participant in participants:
            participant_path = os.path.join(event_path, participant)

            if not os.path.exists(participant_path):
                print(f"Warning: Participant directory not found: {participant_path}")
                continue

            for imu_location in os.listdir(participant_path):
                if minimal_imus is not None and imu_location not in minimal_imus:
                    continue

                imu_path = os.path.join(participant_path, imu_location)

                if not os.path.isdir(imu_path):
                    continue

                for file in os.listdir(imu_path):
                    if file.endswith('_X.csv') and f"_{condition}_" in file:
                        feature_file = os.path.join(imu_path, file)
                        label_file = os.path.join(imu_path, file.replace('_X.csv', '_y.csv'))

                        if os.path.exists(feature_file) and os.path.exists(label_file):
                            try:
                                features_df = pd.read_csv(feature_file)
                                labels_df = pd.read_csv(label_file)

                                # Filter to top features
                                available_features = [f for f in top_features if f in features_df.columns]
                                if len(available_features) == 0:
                                    continue
                                features_df = features_df[available_features]

                                if 'label' in labels_df.columns:
                                    labels = labels_df['label'].values
                                else:
                                    labels = labels_df.iloc[:, -1].values

                                # Ensure same length
                                if len(features_df) != len(labels):
                                    min_len = min(len(features_df), len(labels))
                                    features_df = features_df.iloc[:min_len]
                                    labels = labels[:min_len]

                                # Filter to only event windows (label = 1)
                                event_mask = (labels == 1)
                                features_df = features_df[event_mask]

                                all_features.append(features_df)
                                all_groups.extend([participant] * len(features_df))

                            except Exception as e:
                                print(f"Error loading {feature_file}: {e}")
                                continue

        if not all_features:
            print(f"Warning: No feature files found for {event} - {condition}")
            return None, None, None

        X = pd.concat(all_features, ignore_index=True)
        groups = np.array(all_groups)
        
        # All samples are relabeled as non-event (0) since they're from a different event
        y = np.zeros(len(X), dtype=int)

        # Handle NaN values
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            X = X.fillna(0)
            print(f"Filled {nan_count} NaN values with zeros")

        print(f"Loaded {len(X)} samples for {event} - {condition} (relabeled as non-event)")
        print(f"Participants: {np.unique(groups)}")

        return X, y, groups

    @staticmethod
    def normative_gait_specificity_analysis(out_root, results_dir, test_events, conditions,
                                           test_participants, minimal_results_dir=None,
                                           datatype="IMU", minimal_imu_set=True):
        """
        Test all non-walking events against Straight walk classifier.
        Measures how well walking classifier rejects various activities.
        """
        print("="*80)
        print("NORMATIVE GAIT SPECIFICITY ANALYSIS")
        print(f"Testing events against Straight walk classifier")
        print(f"Conditions: {conditions}")
        print("="*80)

        target_event = "Straight walk"
        results_dict = {}

        for source_event in test_events:
            print(f"\n{'='*60}")
            print(f"Testing: {source_event} data → Straight walk classifier")
            print(f"{'='*60}")

            event_results = {}

            for condition in conditions:
                print(f"  Condition: {condition}")

                # Load Straight walk classifier for this condition
                try:
                    clf_target = ClassifierDeviationAnalysis.load_classifier(
                        results_dir, target_event, datatype, minimal_imu_set, minimal_results_dir
                    )
                    print(f"  Loaded Straight walk {condition} classifier")
                except Exception as e:
                    print(f"  Error loading Straight walk {condition} classifier: {e}")
                    continue

                # Load top features for Straight walk
                try:
                    top_features = ClassifierDeviationAnalysis.load_top_features(
                        results_dir, target_event, datatype, minimal_results_dir
                    )
                    print(f"  Loaded {len(top_features)} top features from Straight walk")
                except Exception as e:
                    print(f"  Error loading top features for Straight walk: {e}")
                    continue

                # Load source event data for this condition
                X_test, y_test, groups_test = ClassifierDeviationAnalysis.load_test_data(
                    out_root, datatype, source_event, condition,
                    test_participants, top_features, minimal_imu_set
                )

                if X_test is None or len(X_test) == 0:
                    print(f"  No test data available for {source_event} - {condition}")
                    continue

                # Convert to numpy array to avoid feature name checking
                X_test = X_test.values

                # Predict using Straight walk classifier
                y_pred = clf_target.predict(X_test)

                # Compute confusion matrix
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                
                # Calculate specificity: TN / (TN + FP)
                tn, fp = cm[0, 0], cm[0, 1]
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                # Per-participant specificity
                participant_specificities = []
                for participant in np.unique(groups_test):
                    mask = groups_test == participant
                    p_cm = confusion_matrix(y_test[mask], y_pred[mask], labels=[0, 1])
                    p_tn, p_fp = p_cm[0, 0], p_cm[0, 1]
                    p_specificity = p_tn / (p_tn + p_fp) if (p_tn + p_fp) > 0 else 0
                    participant_specificities.append(p_specificity)
                
                participant_specificities = np.array(participant_specificities)

                # Store results
                condition_result = {
                    'predictions': y_pred,
                    'true_labels': y_test,
                    'participants': groups_test,
                    'n_samples': len(X_test),
                    'confusion_matrix': cm,
                    'specificity': specificity,
                    'participant_specificities': participant_specificities,
                    'mean_participant_specificity': participant_specificities.mean(),
                    'std_participant_specificity': participant_specificities.std()
                }

                event_results[condition] = condition_result

                print(f"  Overall Specificity: {specificity:.4f}")
                print(f"  Mean Participant Specificity: {participant_specificities.mean():.4f} ± {participant_specificities.std():.4f}")
                print(f"  True Negatives: {tn}, False Positives: {fp}")

            results_dict[source_event] = event_results

        return results_dict

    @staticmethod
    def classifier_event_deviation_analysis(out_root, results_dir, event_pairs, conditions,
                                           test_participants, minimal_results_dir=None,
                                           datatype="IMU", minimal_imu_set=True):
        """
        Analyse event-based deviation using classifiers.
        Measures specificity: how well target event's classifier rejects source event data.
        """
        print("="*80)
        print("CLASSIFIER-BASED EVENT SPECIFICITY ANALYSIS")
        print(f"Conditions: {conditions}")
        print("="*80)

        results_dict = {}

        for source_event, target_event in event_pairs:
            print(f"\n{'='*60}")
            print(f"Using {target_event} classifier to test specificity on {source_event} data")
            print(f"{'='*60}")

            pair_key = f"{source_event}→{target_event}"
            pair_results = {}

            for condition in conditions:
                print(f"  Condition: {condition}")

                # Load target event's classifier for this condition
                try:
                    clf_target = ClassifierDeviationAnalysis.load_classifier(
                        results_dir, target_event, datatype, minimal_imu_set, minimal_results_dir
                    )
                    print(f"  Loaded {target_event} {condition} classifier")
                except Exception as e:
                    print(f"  Error loading {target_event} {condition} classifier: {e}")
                    continue

                # Load top features for target event
                try:
                    top_features = ClassifierDeviationAnalysis.load_top_features(
                        results_dir, target_event, datatype, minimal_results_dir
                    )
                    print(f"  Loaded {len(top_features)} top features from {target_event}")
                except Exception as e:
                    print(f"  Error loading top features for {target_event}: {e}")
                    continue

                # Load source event data for this condition
                X_test, y_test, groups_test = ClassifierDeviationAnalysis.load_test_data(
                    out_root, datatype, source_event, condition,
                    test_participants, top_features, minimal_imu_set
                )

                if X_test is None or len(X_test) == 0:
                    print(f"  No test data available for {source_event} - {condition}")
                    continue

                # Convert to numpy array to avoid feature name checking
                X_test = X_test.values

                # Predict using target event's classifier
                y_pred = clf_target.predict(X_test)

                # Compute confusion matrix
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                
                # Calculate specificity: TN / (TN + FP)
                tn, fp = cm[0, 0], cm[0, 1]
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                # Per-participant specificity
                participant_specificities = []
                for participant in np.unique(groups_test):
                    mask = groups_test == participant
                    p_cm = confusion_matrix(y_test[mask], y_pred[mask], labels=[0, 1])
                    p_tn, p_fp = p_cm[0, 0], p_cm[0, 1]
                    p_specificity = p_tn / (p_tn + p_fp) if (p_tn + p_fp) > 0 else 0
                    participant_specificities.append(p_specificity)
                
                participant_specificities = np.array(participant_specificities)

                # Store results
                condition_result = {
                    'predictions': y_pred,
                    'true_labels': y_test,
                    'participants': groups_test,
                    'n_samples': len(X_test),
                    'confusion_matrix': cm,
                    'specificity': specificity,
                    'participant_specificities': participant_specificities,
                    'mean_participant_specificity': participant_specificities.mean(),
                    'std_participant_specificity': participant_specificities.std()
                }

                pair_results[condition] = condition_result

                print(f"  Overall Specificity: {specificity:.4f}")
                print(f"  Mean Participant Specificity: {participant_specificities.mean():.4f} ± {participant_specificities.std():.4f}")
                print(f"  True Negatives: {tn}, False Positives: {fp}")

            results_dict[pair_key] = pair_results

        return results_dict

    @staticmethod
    def save_classifier_results(results_dict, output_dir, analysis_type):
        """
        Save classifier specificity analysis results to CSV files.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create summary DataFrame
        summary_data = []

        for key, conditions in results_dict.items():
            for condition, result in conditions.items():
                summary_data.append({
                    'Event_or_Pair': key,
                    'Condition': condition,
                    'N_Samples': result['n_samples'],
                    'Specificity': result['specificity'],
                    'Mean_Participant_Specificity': result['mean_participant_specificity'],
                    'Std_Participant_Specificity': result['std_participant_specificity']
                })

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, f"{analysis_type}_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")

        # Save confusion matrices
        for key, conditions in results_dict.items():
            for condition, result in conditions.items():
                safe_key = key.replace('→', '_to_').replace(' ', '_')
                cm_path = os.path.join(output_dir, f"{safe_key}_{condition}_confusion_matrix.csv")
                cm_df = pd.DataFrame(result['confusion_matrix'], 
                                    index=['True_NonEvent', 'True_Event'],
                                    columns=['Pred_NonEvent', 'Pred_Event'])
                cm_df.to_csv(cm_path)

        return summary_df

    @staticmethod
    def visualise_normative_gait_specificity(results_dict, output_dir, title):
        """
        Create visualization for normative gait specificity (all events vs Straight walk).
        Shows per-participant specificity distributions with conditions side-by-side.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data for plotting
        plot_data = []
        for event, conditions in results_dict.items():
            for condition, result in conditions.items():
                for spec in result['participant_specificities']:
                    plot_data.append({
                        'Event': event,
                        'Condition': condition,
                        'Specificity': spec * 100  # Convert to percentage
                    })

        df_plot = pd.DataFrame(plot_data)

        # Create box plot with conditions side-by-side
        fig, ax = plt.subplots(figsize=(14, 6))

        # Define colours for conditions (matching PCA plots)
        condition_colours = {'Normal': '#3182BD', 'AR': '#2ECC71', 'VR': '#E74C3C'}

        sns.boxplot(data=df_plot, x='Event', y='Specificity', hue='Condition',
                    palette=condition_colours, ax=ax, showfliers=False)

        ax.set_xlabel('Event (tested against Straight walk classifier)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Specificity (%)', fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        filename = title.replace(' ', '_').replace(':', '').lower()
        plot_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved normative gait specificity plot: {plot_path}")

    @staticmethod
    def visualise_classifier_specificity(results_dict, output_dir, title):
        """
        Create visualisation for classifier-based specificity analysis.
        Shows per-participant specificity distributions with conditions side-by-side.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data for plotting
        plot_data = []
        for pair_key, conditions in results_dict.items():
            for condition, result in conditions.items():
                for spec in result['participant_specificities']:
                    plot_data.append({
                        'Event_Pair': pair_key,
                        'Condition': condition,
                        'Specificity': spec * 100  # Convert to percentage
                    })

        df_plot = pd.DataFrame(plot_data)

        # Create box plot with conditions side-by-side
        fig, ax = plt.subplots(figsize=(14, 6))

        # Define colours for conditions (matching PCA plots)
        condition_colours = {'Normal': '#3182BD', 'AR': '#2ECC71', 'VR': '#E74C3C'}

        sns.boxplot(data=df_plot, x='Event_Pair', y='Specificity', hue='Condition',
                    palette=condition_colours, ax=ax, showfliers=False)

        ax.set_xlabel('Event Pair (Source→Target Classifier)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Specificity (%)', fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        filename = title.replace(' ', '_').replace(':', '').lower()
        plot_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved classifier specificity plot: {plot_path}")


if __name__ == "__main__":
    # Configuration
    out_root = "Z:/Upper Body/Results/30 Participants/features"
    results_dir = "Z:/Upper Body/Results/30 Participants/models"
    minimal_results_dir = "Z:/Upper Body/Results/30 Participants/minimal_imu_models"
    output_dir = "Z:/Upper Body/Results/30 Participants/classifier_deviation_analysis"

    datatype = "IMU"
    minimal_imu_set = True  # Use only Head_imu and RightForeArm_imu

    # Define test participants
    test_participants = [
        "P001", "P003", "P004", "P005", "P006", 
        "P007", "P008", "P010", "P011", "P012",
        "P014", "P016", "P017", "P018", "P019",
        "P020", "P021", "P022", "P023", "P024",
        "P025", "P026", "P027", "P028", "P030",
        "P031", "P032", "P033", "P035", "P043"
    ]

    print("="*80)
    print("CLASSIFIER-BASED EVENT SPECIFICITY ANALYSIS")
    print("Using Minimal IMU Set: Head_imu and RightForeArm_imu")
    print(f"Test participants: {test_participants}")
    print("="*80)

    # ANALYSIS 1: Normative Gait Specificity (all non-walking events → Straight walk classifier)
    print("\n\n")
    print("#"*80)
    print("# ANALYSIS 1: NORMATIVE GAIT SPECIFICITY")
    print("#"*80)
    
    # All events except Straight walk
    all_events = [
        "Dribbling basketball",
        "Pick up basketball",
        "Put down basketball",
        "Step over cone",
        "Stair down",
        "Stair up",
        "Place ping pong ball in cup"
    ]

    normative_results = ClassifierDeviationAnalysis.normative_gait_specificity_analysis(
        out_root=out_root,
        results_dir=results_dir,
        test_events=all_events,
        conditions=["Normal", "AR", "VR"],
        test_participants=test_participants,
        minimal_results_dir=minimal_results_dir,
        datatype=datatype,
        minimal_imu_set=minimal_imu_set
    )

    # Save results for Analysis 1
    normative_output_dir = os.path.join(output_dir, "normative_gait")
    normative_summary = ClassifierDeviationAnalysis.save_classifier_results(
        normative_results, normative_output_dir, "normative_gait_specificity"
    )

    # Visualise results for Analysis 1
    ClassifierDeviationAnalysis.visualise_normative_gait_specificity(
        normative_results, normative_output_dir,
        "Normative Gait Specificity (Straight Walk Classifier)"
    )

    # ANALYSIS 2: Biomechanically Related Events (event pairs)
    print("\n\n")
    print("#"*80)
    print("# ANALYSIS 2: BIOMECHANICALLY RELATED EVENTS")
    print("#"*80)

    # Define biomechanically related event pairs
    event_pairs = [
        ("Pick up basketball", "Put down basketball"),
        ("Put down basketball", "Pick up basketball"),
        ("Stair down", "Stair up"),
        ("Stair up", "Stair down")
    ]

    # Run classifier-based specificity analysis for all conditions
    classifier_results = ClassifierDeviationAnalysis.classifier_event_deviation_analysis(
        out_root=out_root,
        results_dir=results_dir,
        event_pairs=event_pairs,
        conditions=["Normal", "AR", "VR"],
        test_participants=test_participants,
        minimal_results_dir=minimal_results_dir,
        datatype=datatype,
        minimal_imu_set=minimal_imu_set
    )

    # Save results for Analysis 2
    classifier_output_dir = os.path.join(output_dir, "related_events")
    classifier_summary = ClassifierDeviationAnalysis.save_classifier_results(
        classifier_results, classifier_output_dir, "classifier_specificity"
    )

    # Visualise results for Analysis 2
    ClassifierDeviationAnalysis.visualise_classifier_specificity(
        classifier_results, classifier_output_dir,
        "Classifier Specificity on Related Events"
    )

    print("\n\n")
    print("="*80)
    print("CLASSIFIER SPECIFICITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Normative gait specificity: {len(normative_results)} events")
    print(f"  - Related events specificity: {len(classifier_results)} event pairs")
    print("\nAnalysis complete!")