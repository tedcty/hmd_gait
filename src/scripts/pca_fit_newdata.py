import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gias3.learning.PCA import loadPrincipalComponents
import json
from pathlib import Path


class DeviationAnalysis:
    """
    Perform deviation analysis by projecting unseen test trials into trained PCA models.
    """

    @staticmethod
    def load_pca_model(model_dir, event, condition, datatype="IMU"):
        """
        Load a trained PCA model.
        """
        event_filename = event.replace(' ', '_')
        pc_model_path = os.path.join(
            model_dir, "pca_models", event_filename, condition, "pc_model_minimal",
            f"{datatype}_{event_filename}_{condition}.pc"
        )

        if not os.path.exists(pc_model_path):
            raise FileNotFoundError(f"PCA model not found: {pc_model_path}")

        print(f"Loading PCA model: {pc_model_path}")
        pc = loadPrincipalComponents(pc_model_path)
        return pc

    @staticmethod
    def load_top_features(results_dir, event, datatype="IMU", minimal_results_dir=None):
        """
        Load the top 100 features used for training the PCA model.
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

        Args:
            out_root: Root directory containing feature data
            datatype: Data type (e.g., "IMU")
            event: Event name
            condition: Condition (e.g., "Normal", "AR", "VR")
            participants: List of participant IDs
            top_features: List of feature names to load
            minimal_imu_set: Whether to use minimal IMU set (Head_imu, RightForeArm_imu)

        Returns:
            X: Feature data
            y: Labels
            groups: Participant IDs for each sample
        """
        event_dir = event.replace(" ", "_")
        event_path = os.path.join(out_root, datatype, event_dir)

        minimal_imus = {'Head_imu', 'RightForeArm_imu'} if minimal_imu_set else None

        all_features = []
        all_labels = []
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
                                labels = labels[event_mask]

                                all_features.append(features_df)
                                all_labels.extend(labels)
                                all_groups.extend([participant] * len(features_df))

                            except Exception as e:
                                print(f"Error loading {feature_file}: {e}")
                                continue

        if not all_features:
            print(f"Warning: No feature files found for {event} - {condition}")
            return None, None, None

        X = pd.concat(all_features, ignore_index=True)
        y = np.array(all_labels)
        groups = np.array(all_groups)

        # Handle NaN values
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            X = X.fillna(0)
            print(f"Filled {nan_count} NaN values with zeros")

        print(f"Loaded {len(X)} samples for {event} - {condition}")
        print(f"Participants: {np.unique(groups)}")

        return X, y, groups

    @staticmethod
    def compute_reconstruction_error(pc, X_test):
        """
        Compute reconstruction error for test data projected into PCA space.
        """
        # Project test data into PCA space
        # GIAS3 PCA expects data as (features, samples)
        test_scores = pc.project(X_test.values.T).T  # Result is (samples, components)

        # Reconstruct from PCA space: X_reconstructed = mean + (scores @ loadings.T)
        X_reconstructed = pc.mean.reshape(-1, 1) + (pc.modes @ test_scores.T)
        X_reconstructed = X_reconstructed.T  # Back to (samples, features)

        # Compute reconstruction error per sample (L2 norm)
        reconstruction_errors = np.sqrt(((X_test.values - X_reconstructed) ** 2).sum(axis=1))

        # Compute original data magnitude
        original_magnitudes = np.sqrt((X_test.values ** 2).sum(axis=1))

        # Compute percentage error
        percentage_errors = (reconstruction_errors / (original_magnitudes + 1e-10)) * 100

        return reconstruction_errors, percentage_errors, X_reconstructed, test_scores

    @staticmethod
    def condition_based_deviation_analysis(out_root, results_dir, events, test_conditions,
                                            test_participants, minimal_results_dir=None,
                                            datatype="IMU", minimal_imu_set=True):
        """
        Analyse how AR and VR conditions deviate from Normal condition for each event.
        Projects AR and VR trials into the Normal PCA model.
        """
        print("="*80)
        print("CONDITION-BASED DEVIATION ANALYSIS")
        print("Projecting AR and VR trials into Normal PCA models")
        print("="*80)

        results_dict = {}

        for event in events:
            print(f"\n{'='*60}")
            print(f"Processing Event: {event}")
            print(f"{'='*60}")

            event_results = {}

            # Load Normal condition PCA model
            try:
                pc_normal = DeviationAnalysis.load_pca_model(
                    results_dir, event, "Normal", datatype
                )
                print(f"Loaded Normal PCA model for {event}")
                print(f"  Number of components: {pc_normal.modes.shape[1]}")
                print(f"  Number of features: {pc_normal.modes.shape[0]}")
            except Exception as e:
                print(f"Error loading Normal PCA model for {event}: {e}")
                continue

            # Load top features for this event
            try:
                top_features = DeviationAnalysis.load_top_features(
                    results_dir, event, datatype, minimal_results_dir
                )
                print(f"Loaded {len(top_features)} top features")
            except Exception as e:
                print(f"Error loading top features for {event}: {e}")
                continue

            # Analyse each test condition
            for test_condition in test_conditions:
                print(f"\n  Testing condition: {test_condition}")

                # Load test data
                X_test, y_test, groups_test = DeviationAnalysis.load_test_data(
                    out_root, datatype, event, test_condition,
                    test_participants, top_features, minimal_imu_set
                )

                if X_test is None or len(X_test) == 0:
                    print(f"  No test data available for {test_condition}")
                    continue

                # Compute reconstruction error
                recon_errors, percent_errors, X_recon, scores = DeviationAnalysis.compute_reconstruction_error(
                    pc_normal, X_test)

                # Store results
                condition_result = {
                    'reconstruction_errors': recon_errors,
                    'percentage_errors': percent_errors,
                    'participants': groups_test,
                    'n_samples': len(X_test),
                    'mean_error': recon_errors.mean(),
                    'std_error': recon_errors.std(),
                    'mean_percent_error': percent_errors.mean(),
                    'std_percent_error': percent_errors.std(),
                    'median_percent_error': np.median(percent_errors),
                    'pca_scores': scores
                }

                event_results[test_condition] = condition_result

                print(f"  Mean reconstruction error: {recon_errors.mean():.4f} ± {recon_errors.std():.4f}")
                print(f"  Mean percentage error: {percent_errors.mean():.2f}% ± {percent_errors.std():.2f}%")
                print(f"  Median percentage error: {np.median(percent_errors):.2f}%")

            results_dict[event] = event_results

        return results_dict

    @staticmethod
    def event_based_deviation_analysis(out_root, results_dir, event_pairs, condition,
                                        test_participants, minimal_results_dir=None,
                                        datatype="IMU", minimal_imu_set=True):
        """
        Analyse event-based deviation by projecting one event's data into another event's PCA model.
        """
        print("="*80)
        print("EVENT-BASED DEVIATION ANALYSIS")
        print(f"Condition: {condition}")
        print("="*80)

        results_dict = {}

        for source_event, target_event in event_pairs:
            print(f"\n{'='*60}")
            print(f"Projecting {source_event} into {target_event} PCA model")
            print(f"{'='*60}")

            pair_key = f"{source_event}→{target_event}"

            # Load target event's PCA model
            try:
                pc_target = DeviationAnalysis.load_pca_model(
                    results_dir, target_event, condition, datatype
                )
                print(f"Loaded {target_event} PCA model")
            except Exception as e:
                print(f"Error loading {target_event} PCA model: {e}")
                continue

            # Load top features for target event (PCA was trained on these)
            try:
                top_features = DeviationAnalysis.load_top_features(
                    results_dir, target_event, datatype, minimal_results_dir
                )
                print(f"Loaded {len(top_features)} top features from {target_event}")
            except Exception as e:
                print(f"Error loading top features for {target_event}: {e}")
                continue

            # Load source event data
            X_test, y_test, groups_test = DeviationAnalysis.load_test_data(
                out_root, datatype, source_event, condition,
                test_participants, top_features, minimal_imu_set
            )

            if X_test is None or len(X_test) == 0:
                print(f"No test data available for {source_event}")
                continue

            # Compute reconstruction error
            recon_errors, percent_errors, X_recon, scores = \
                DeviationAnalysis.compute_reconstruction_error(pc_target, X_test)

            # Store results
            pair_result = {
                'reconstruction_errors': recon_errors,
                'percentage_errors': percent_errors,
                'participants': groups_test,
                'n_samples': len(X_test),
                'mean_error': recon_errors.mean(),
                'std_error': recon_errors.std(),
                'mean_percent_error': percent_errors.mean(),
                'std_percent_error': percent_errors.std(),
                'median_percent_error': np.median(percent_errors),
                'pca_scores': scores
            }

            results_dict[pair_key] = pair_result

            print(f"Mean reconstruction error: {recon_errors.mean():.4f} ± {recon_errors.std():.4f}")
            print(f"Mean percentage error: {percent_errors.mean():.2f}% ± {percent_errors.std():.2f}%")
            print(f"Median percentage error: {np.median(percent_errors):.2f}%")

        return results_dict

    @staticmethod
    def deviation_from_normative_gait(out_root, results_dir, all_events, condition,
                                        test_participants, minimal_results_dir=None,
                                        datatype="IMU", minimal_imu_set=True):
        """
        Project all events into Straight walk PCA model to assess deviation from normative gait.
        """
        print("="*80)
        print("DEVIATION FROM NORMATIVE GAIT")
        print(f"Projecting all events into Straight walk PCA model")
        print(f"Condition: {condition}")
        print("="*80)

        normative_event = "Straight walk"

        # Load Straight walk PCA model
        try:
            pc_normative = DeviationAnalysis.load_pca_model(
                results_dir, normative_event, condition, datatype
            )
            print(f"Loaded Straight walk PCA model")
            print(f"  Number of components: {pc_normative.modes.shape[1]}")
        except Exception as e:
            print(f"Error loading Straight walk PCA model: {e}")
            return {}

        # Load top features for Straight walk
        try:
            top_features = DeviationAnalysis.load_top_features(
                results_dir, normative_event, datatype, minimal_results_dir
            )
            print(f"Loaded {len(top_features)} top features")
        except Exception as e:
            print(f"Error loading top features: {e}")
            return {}

        results_dict = {}

        for event in all_events:
            print(f"\n{'='*60}")
            print(f"Testing event: {event}")
            print(f"{'='*60}")

            # Load event data
            X_test, y_test, groups_test = DeviationAnalysis.load_test_data(
                out_root, datatype, event, condition,
                test_participants, top_features, minimal_imu_set
            )

            if X_test is None or len(X_test) == 0:
                print(f"No test data available for {event}")
                continue

            # Compute reconstruction error
            recon_errors, percent_errors, X_recon, scores = \
                DeviationAnalysis.compute_reconstruction_error(pc_normative, X_test)

            # Store results
            event_result = {
                'reconstruction_errors': recon_errors,
                'percentage_errors': percent_errors,
                'participants': groups_test,
                'n_samples': len(X_test),
                'mean_error': recon_errors.mean(),
                'std_error': recon_errors.std(),
                'mean_percent_error': percent_errors.mean(),
                'std_percent_error': percent_errors.std(),
                'median_percent_error': np.median(percent_errors),
                'pca_scores': scores
            }

            results_dict[event] = event_result

            print(f"Mean reconstruction error: {recon_errors.mean():.4f} ± {recon_errors.std():.4f}")
            print(f"Mean percentage error: {percent_errors.mean():.2f}% ± {percent_errors.std():.2f}%")
            print(f"Median percentage error: {np.median(percent_errors):.2f}%")

        return results_dict

    @staticmethod
    def save_deviation_results(results_dict, output_dir, analysis_type):
        """
        Save deviation analysis results to CSV files.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create summary DataFrame
        summary_data = []

        for key, result in results_dict.items():
            if isinstance(result, dict) and 'mean_error' in result:
                # Single result (event-based or normative)
                summary_data.append({
                    'Analysis': key,
                    'N_Samples': result['n_samples'],
                    'Mean_Error': result['mean_error'],
                    'Std_Error': result['std_error'],
                    'Mean_Percent_Error': result['mean_percent_error'],
                    'Std_Percent_Error': result['std_percent_error'],
                    'Median_Percent_Error': result['median_percent_error']
                })
            else:
                # Nested results (condition-based)
                for sub_key, sub_result in result.items():
                    if isinstance(sub_result, dict) and 'mean_error' in sub_result:
                        summary_data.append({
                            'Event': key,
                            'Condition': sub_key,
                            'N_Samples': sub_result['n_samples'],
                            'Mean_Error': sub_result['mean_error'],
                            'Std_Error': sub_result['std_error'],
                            'Mean_Percent_Error': sub_result['mean_percent_error'],
                            'Std_Percent_Error': sub_result['std_percent_error'],
                            'Median_Percent_Error': sub_result['median_percent_error']
                        })

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, f"{analysis_type}_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")

        return summary_df

    @staticmethod
    def visualise_condition_deviation(results_dict, output_dir):
        """
        Create visualisation for condition-based deviation analysis.
        Single plot with all events on x-axis, AR and VR side-by-side.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data for plotting
        plot_data = []
        for event, conditions in results_dict.items():
            for condition, result in conditions.items():
                per_participant = pd.DataFrame({
                    'Participant': result['participants'],
                    'Percentage_Error': result['percentage_errors'],
                    'Event': event,
                    'Condition': condition
                }).groupby('Participant')['Percentage_Error'].mean().reset_index()

                for _, row in per_participant.iterrows():
                    plot_data.append({
                        'Event': event,
                        'Condition': condition,
                        'Percentage_Error': row['Percentage_Error']
                    })

        df_plot = pd.DataFrame(plot_data)

        # Create single box plot with all events
        fig, ax = plt.subplots(figsize=(14, 6))

        sns.boxplot(data=df_plot, x='Event', y='Percentage_Error', hue='Condition',
                    palette={'AR': '#E74C3C', 'VR': '#3498DB'},
                    ax=ax, showfliers=False)

        ax.set_xlabel('Event', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reconstruction Error (%)', fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title='Condition', loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "condition_deviation_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved condition deviation plot: {plot_path}")

    @staticmethod
    def visualise_event_deviation(results_dict, output_dir, title, color='#3182BD'):
        """
        Create visualisation for event-based deviation analysis using box plots.
        Shows per-participant distribution of reconstruction errors.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data for plotting - aggregate per participant
        plot_data = []
        for key, result in results_dict.items():
            per_participant = pd.DataFrame({
                'Participant': result['participants'],
                'Percentage_Error': result['percentage_errors']
            }).groupby('Participant')['Percentage_Error'].mean().reset_index()

            for _, row in per_participant.iterrows():
                plot_data.append({
                    'Event': key,
                    'Percentage_Error': row['Percentage_Error']
                })

        df_plot = pd.DataFrame(plot_data)

        # Create box plot
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.boxplot(data=df_plot, x='Event', y='Percentage_Error',
                    color=color, ax=ax, showfliers=False)

        ax.set_xlabel('Event Projection', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reconstruction Error (%)', fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        filename = title.replace(' ', '_').replace(':', '').lower()
        plot_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved event deviation plot: {plot_path}")


if __name__ == "__main__":
    # Configuration
    out_root = "Z:/Upper Body/Results/30 Participants/features"
    results_dir = "Z:/Upper Body/Results/30 Participants/models"
    minimal_results_dir = "Z:/Upper Body/Results/30 Participants/minimal_imu_models"
    output_dir = "Z:/Upper Body/Results/30 Participants/deviation_analysis"

    datatype = "IMU"
    minimal_imu_set = True  # Use only Head_imu and RightForeArm_imu

    # Define events
    all_events = [
        "Dribbling basketball",
        "Pick up basketball",
        "Put down basketball",
        "Step over cone",
        "Stair down",
        "Stair up",
        "Place ping pong ball in cup",
        "Straight walk"
    ]

    # Define test participants (hold-out set for testing)
    # Using all 30 participants
    test_participants = [
        "P001", "P003", "P004", "P005", "P006", 
        "P007", "P008", "P010", "P011", "P012",
        "P014", "P016", "P017", "P018", "P019",
        "P020", "P021", "P022", "P023", "P024",
        "P025", "P026", "P027", "P028", "P030",
        "P031", "P032", "P033", "P035", "P043"
    ]

    print("="*80)
    print("PCA DEVIATION ANALYSIS")
    print("Using Minimal IMU Set: Head_imu and RightForeArm_imu")
    print(f"Test participants: {test_participants}")
    print("="*80)

    # 1. CONDITION-BASED DEVIATION ANALYSIS
    print("\n\n")
    condition_results = DeviationAnalysis.condition_based_deviation_analysis(
        out_root=out_root,
        results_dir=results_dir,
        events=all_events,
        test_conditions=["AR", "VR"],
        test_participants=test_participants,
        minimal_results_dir=minimal_results_dir,
        datatype=datatype,
        minimal_imu_set=minimal_imu_set
    )

    # Save condition-based results
    condition_output_dir = os.path.join(output_dir, "condition_based")
    condition_summary = DeviationAnalysis.save_deviation_results(
        condition_results, condition_output_dir, "condition_based"
    )

    # Visualise condition-based results
    DeviationAnalysis.visualise_condition_deviation(
        condition_results, condition_output_dir
    )

    # 2. EVENT-BASED DEVIATION ANALYSIS
    print("\n\n")

    # a. Deviation from Normative Gait (other events data to Straight walk model)
    # Exclude Straight walk itself to only measure deviation of OTHER events
    events_to_test = [e for e in all_events if e != "Straight walk"]

    normative_results = DeviationAnalysis.deviation_from_normative_gait(
        out_root=out_root,
        results_dir=results_dir,
        all_events=events_to_test,
        condition="Normal",
        test_participants=test_participants,
        minimal_results_dir=minimal_results_dir,
        datatype=datatype,
        minimal_imu_set=minimal_imu_set
    )

    # Save normative gait results
    normative_output_dir = os.path.join(output_dir, "normative_gait")
    normative_summary = DeviationAnalysis.save_deviation_results(
        normative_results, normative_output_dir, "normative_gait"
    )

    # Visualise normative gait results
    DeviationAnalysis.visualise_event_deviation(
        normative_results, normative_output_dir,
        "Deviation from Normative Gait (Straight Walk)",
        color='#27AE60'
    )

    # b. Biomechanically Related Event Pairs
    print("\n\n")

    event_pairs = [
        ("Pick up basketball", "Put down basketball"),
        ("Put down basketball", "Pick up basketball"),
        ("Stair down", "Stair up"),
        ("Stair up", "Stair down")
    ]

    related_events_results = DeviationAnalysis.event_based_deviation_analysis(
        out_root=out_root,
        results_dir=results_dir,
        event_pairs=event_pairs,
        condition="Normal",
        test_participants=test_participants,
        minimal_results_dir=minimal_results_dir,
        datatype=datatype,
        minimal_imu_set=minimal_imu_set
    )

    # Save related events results
    related_output_dir = os.path.join(output_dir, "related_events")
    related_summary = DeviationAnalysis.save_deviation_results(
        related_events_results, related_output_dir, "related_events"
    )

    # Visualise related events results
    DeviationAnalysis.visualise_event_deviation(
        related_events_results, related_output_dir,
        "Deviation Between Biomechanically Related Events",
        color='#8E44AD'
    )

    # 3. COMBINED SUMMARY
    print("\n\n")
    print("="*80)
    print("DEVIATION ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nSummary:")
    print(f"  - Condition-based analysis: {len(condition_results)} events")
    print(f"  - Normative gait analysis: {len(normative_results)} events")
    print(f"  - Related events analysis: {len(related_events_results)} event pairs")

    print("\nAnalysis complete!")
