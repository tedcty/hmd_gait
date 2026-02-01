import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gias3.learning.PCA import loadPrincipalComponents
from scipy.optimize import least_squares
from matplotlib.patches import Patch
import argparse


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
    def load_top_features(results_dir, event, condition, datatype="IMU"):
        """
        Load feature order directly from trained PCA model loadings.
        This guarantees exact feature order match with the trained model.
        """
        event_filename = event.replace(' ', '_')
        loadings_file = os.path.join(
            results_dir, "pca_models", event_filename, condition, "pc_model_minimal",
            f"{datatype}_{event_filename}_{condition}_loadings.csv"
        )
        
        if not os.path.exists(loadings_file):
            raise FileNotFoundError(f"Loadings file not found: {loadings_file}")
        
        loadings_df = pd.read_csv(loadings_file, index_col=0)
        return loadings_df.index.tolist()

    @staticmethod
    def load_test_data(out_root, datatype, event, condition, participants, top_features, minimal_imu_set=True):
        """
        Load test data for specified participants, event, and condition.
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

        # Trained PCA feature space (same order)
        X = X.reindex(columns=top_features).fillna(0)

        print(f"Loaded {len(X)} samples for {event} - {condition}")
        print(f"Participants: {np.unique(groups)}")

        return X, y, groups

    @staticmethod
    def load_feature_means_stds(results_dir, event, condition, datatype="IMU"):
        """
        Load feature means and stds saved with the PCA model.
        """
        event_filename = event.replace(' ', '_')
        means_path = os.path.join(
            results_dir, "pca_models", event_filename, condition, "pc_model_minimal",
            f"{datatype}_{event_filename}_{condition}_feature_means.csv"
        )
        stds_path = os.path.join(
            results_dir, "pca_models", event_filename, condition, "pc_model_minimal",
            f"{datatype}_{event_filename}_{condition}_feature_stds.csv"
        )
        if not os.path.exists(means_path) or not os.path.exists(stds_path):
            raise FileNotFoundError(f"Feature means or stds not found for {event} {condition}")
        means_df = pd.read_csv(means_path, index_col=0)
        stds_df = pd.read_csv(stds_path, index_col=0)

        means = means_df.iloc[:, 0]
        stds = stds_df.iloc[:, 0]
        return means, stds

    @staticmethod
    def standardise_X(X, means, stds):
        """
        Standardise X using provided means and stds (z-score).
        """
        # Align stats to the current feature order to avoid silent misalignment
        means_aligned = means.reindex(X.columns)
        stds_aligned = stds.reindex(X.columns)

        # Any missing stats (e.g., because a feature was absent) get safe defaults
        means_aligned = means_aligned.fillna(0.0)
        stds_aligned = stds_aligned.fillna(1.0)
        stds_aligned = stds_aligned.copy()
        stds_aligned[stds_aligned == 0] = 1.0

        return (X - means_aligned) / stds_aligned

    @staticmethod
    def fit_to_pca_model(X_test, pc, modes, m_weight=1.0, verbose=False):
        """
        Fit test data to PCA model by optimizing PCA scores to minimize reconstruction error.
        
        Based on fitSSMTo3DPoints from gias3.learning.PCA_fitting, adapted for feature data.
        This optimization-based approach finds the best PCA weights that reconstruct the test data.
        """
        n_samples = X_test.shape[0]
        n_modes = len(modes)
        
        # Get PCA components for selected modes
        mean = pc.mean
        modes_matrix = pc.modes[:, modes]  # (features, n_modes)
        
        # Store optimized scores
        scores_opt = np.zeros((n_samples, n_modes))
        X_reconstructed = np.zeros_like(X_test)
        
        # Precompute penalty scaling
        use_penalty = (m_weight is not None) and (m_weight > 0)
        penalty_scale = np.sqrt(m_weight) if use_penalty else 0.0
        
        # Optimize for each sample
        for i in range(n_samples):
            target = X_test[i, :]
            
            def objective(weights_sd):
                """
                Objective function: reconstruction error + Mahalanobis penalty
                
                Minimizes: ||X - X_recon||^2 + m_weight * ||weights_sd||^2
                """
                # Convert SD weights to actual weights using GIAS3 method
                weights = pc.getWeightsBySD(modes, weights_sd)
                
                # Reconstruct: X_recon = mean + (modes @ weights)
                X_recon = mean + modes_matrix @ weights
                
                # Reconstruction error per feature
                recon_error = target - X_recon
                
                # Mahalanobis penalty as vector (one per mode)
                if use_penalty:
                    penalty_vector = penalty_scale * weights_sd
                    return np.append(recon_error, penalty_vector)
                else:
                    return recon_error
            
            # Initial guess: zero weights (mean shape)
            x0 = np.zeros(n_modes)
            
            # Optimize using least squares
            res = least_squares(objective, x0, method='lm', ftol=1e-8, xtol=1e-8, gtol=1e-8, verbose=2 if verbose else 0)
            xopt = res.x
            
            # Store optimized scores (in SD units)
            scores_opt[i, :] = xopt
            
            # Reconstruct with optimized weights
            weights_opt = pc.getWeightsBySD(modes, xopt)
            X_reconstructed[i, :] = mean + modes_matrix @ weights_opt
            
            if verbose and (i + 1) % 100 == 0:
                print(f"    Fitted {i + 1}/{n_samples} samples")
        
        # Compute reconstruction errors
        reconstruction_errors = np.sqrt(((X_test - X_reconstructed) ** 2).sum(axis=1))
        
        # Percentage error in the same standardised feature space
        eps = 1e-12
        original_magnitudes = np.sqrt((X_test ** 2).sum(axis=1)) + eps
        percentage_errors = (reconstruction_errors / original_magnitudes) * 100
        
        if verbose:
            print(f"    Mean reconstruction error: {reconstruction_errors.mean():.4f}")
            print(f"    Mean percentage error: {percentage_errors.mean():.2f}%")
        
        return scores_opt, X_reconstructed, reconstruction_errors, percentage_errors

    @staticmethod
    def compute_reconstruction_error(pc, X_test, n_modes=None, m_weight=0.01):
        """
        Compute reconstruction error for test data using optimization-based PCA fitting.
        Uses least squares optimization to find best-fit PCA scores with Mahalanobis regularization.

        Uses all available PCs in the trained model by default (n_modes=None).
        Optionally you can cap to first n_modes PCs if you want.
        """
        max_available = pc.modes.shape[1]  # How many PCs exist in the trained model
        n_modes_to_use = max_available if n_modes is None else min(int(n_modes), max_available)
        modes = np.arange(n_modes_to_use, dtype=int)
        
        print(f"    Fitting {len(X_test)} samples to PCA model using optimization...")
        print(f"    Using {n_modes_to_use}/{max_available} PCA modes")

        X_np = X_test.to_numpy() if hasattr(X_test, "to_numpy") else np.asarray(X_test)
        test_scores, X_reconstructed, reconstruction_errors, percentage_errors = \
            DeviationAnalysis.fit_to_pca_model(
                X_np, pc, modes, m_weight=m_weight, verbose=False
            )

        return reconstruction_errors, percentage_errors, X_reconstructed, test_scores
    
    @staticmethod
    def per_participant_mean_errors(groups, percent_errors):
        df = pd.DataFrame({
            'Participant': groups,
            'Percentage_Error': percent_errors
        })
        return df.groupby('Participant')['Percentage_Error'].mean()
    
    @staticmethod
    def baseline_stats_from_participants(groups, percent_errors):
        """
        Baseline mean/std computed over per-participant mean percentage errors.
        This avoids bias from participants with more samples.
        """
        pp = DeviationAnalysis.per_participant_mean_errors(groups, percent_errors)
        mu = float(pp.mean())
        sd = float(pp.std(ddof=1))
        if sd == 0 or np.isnan(sd):
            sd = 1.0 # Safe default to avoid zero std
        return mu, sd, pp  # Return per-participant series too

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
                    results_dir, event, "Normal", datatype
                )
                print(f"Loaded {len(top_features)} top features")
            except Exception as e:
                print(f"Error loading top features for {event}: {e}")
                continue

            # Analyse each condition
            all_conditions = ["Normal"] + [c for c in test_conditions if c != "Normal"]
            baseline_mu = None
            baseline_sd = None

            for test_condition in all_conditions:
                print(f"\n  Testing condition: {test_condition}")

                # Load test data
                X_test, y_test, groups_test = DeviationAnalysis.load_test_data(
                    out_root, datatype, event, test_condition,
                    test_participants, top_features, minimal_imu_set
                )

                if X_test is None or len(X_test) == 0:
                    print(f"  No test data available for {test_condition}")
                    continue

                # Standardise features
                means, stds = DeviationAnalysis.load_feature_means_stds(results_dir, event, "Normal", datatype)
                X_test = DeviationAnalysis.standardise_X(X_test, means, stds)

                # Compute reconstruction error
                recon_errors, percent_errors, X_recon, scores = DeviationAnalysis.compute_reconstruction_error(
                    pc_normal, X_test, n_modes=None, m_weight=0.01)
                
                # If this is the baseline (Normal), compute baseline stats
                if test_condition == "Normal":
                    baseline_mu, baseline_sd, baseline_pp = DeviationAnalysis.baseline_stats_from_participants(groups_test, percent_errors)
                    print(f"  Baseline (Normal->Normal) per-participant mean percentage error: {baseline_mu:.2f}% ± {baseline_sd:.2f}%")

                # z-score each sample relative to baseline
                if baseline_mu is not None and baseline_sd is not None:
                    z_scores = (percent_errors - baseline_mu) / baseline_sd
                else:
                    z_scores = None

                pp_z = DeviationAnalysis.per_participant_mean_errors(groups_test, z_scores) if z_scores is not None else None
                pp_percent = DeviationAnalysis.per_participant_mean_errors(groups_test, percent_errors)

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
                    'pca_scores': scores,
                    'baseline_mean_percent_error': baseline_mu,
                    'baseline_sd_percent_error': baseline_sd,
                    'z_scores': z_scores,
                    'median_z': np.median(z_scores) if z_scores is not None else None,
                    'mean_z': float(np.mean(z_scores)) if z_scores is not None else None,
                    'std_z': float(np.std(z_scores, ddof=1)) if z_scores is not None else None,
                    'per_participant_mean_percent_error': pp_percent,
                    'per_participant_mean_z': pp_z
                }

                event_results[test_condition] = condition_result

                print(f"  Mean reconstruction error: {recon_errors.mean():.4f} ± {recon_errors.std():.4f}")
                print(f"  Mean percentage error: {percent_errors.mean():.2f}% ± {percent_errors.std():.2f}%")
                print(f"  Median percentage error: {np.median(percent_errors):.2f}%")

            results_dict[event] = event_results

        return results_dict

    @staticmethod
    def event_based_deviation_analysis(out_root, results_dir, event_pairs, conditions,
                                        test_participants, minimal_results_dir=None,
                                        datatype="IMU", minimal_imu_set=True):
        """
        Analyse event-based deviation by projecting one event's data into another event's PCA model.
        Each condition is projected into its respective target event PCA model.
        """
        print("="*80)
        print("EVENT-BASED DEVIATION ANALYSIS")
        print(f"Conditions: {conditions}")
        print("="*80)

        results_dict = {}
        baseline_cache = {}

        for source_event, target_event in event_pairs:
            print(f"\n{'='*60}")
            print(f"Projecting {source_event} into {target_event} PCA models")
            print(f"{'='*60}")

            pair_key = f"{source_event}→{target_event}"

            pair_results = {}

            for condition in conditions:
                print(f"  Condition: {condition}")

                # Load target event's PCA model for THIS condition
                try:
                    pc_target = DeviationAnalysis.load_pca_model(
                        results_dir, target_event, condition, datatype
                    )
                    print(f"  Loaded {target_event} {condition} PCA model")
                except Exception as e:
                    print(f"  Error loading {target_event} {condition} PCA model: {e}")
                    continue

                # Load top features for target event (PCA was trained on these)
                try:
                    top_features = DeviationAnalysis.load_top_features(
                        results_dir, target_event, condition, datatype
                    )
                    print(f"  Loaded {len(top_features)} top features from {target_event}")
                except Exception as e:
                    print(f"  Error loading top features for {target_event}: {e}")
                    continue

                cache_key = (target_event, condition)
                if cache_key in baseline_cache:
                    base_mu, base_sd, _ = baseline_cache[cache_key]
                else:
                    # Baseline: target_event -> target_event model (same condition)
                    X_base, y_base, groups_base = DeviationAnalysis.load_test_data(
                        out_root, datatype, target_event, condition,
                        test_participants, top_features, minimal_imu_set
                    )

                    if X_base is None or len(X_base) == 0:
                        print(f"  No baseline data available for {target_event} - {condition}")
                        continue

                    means, stds = DeviationAnalysis.load_feature_means_stds(results_dir, target_event, condition, datatype)
                    X_base = DeviationAnalysis.standardise_X(X_base, means, stds)

                    _, base_percent_errors, _, _ = DeviationAnalysis.compute_reconstruction_error(
                        pc_target, X_base, n_modes=None, m_weight=0.01)
                    
                    base_mu, base_sd, base_pp = DeviationAnalysis.baseline_stats_from_participants(groups_base, base_percent_errors)

                    print(f"  Baseline ({target_event}->{target_event}) per-participant mean percentage error: {base_mu:.2f}% ± {base_sd:.2f}%")

                    baseline_cache[cache_key] = (base_mu, base_sd, base_pp)

                # Load source event data for this condition
                X_test, y_test, groups_test = DeviationAnalysis.load_test_data(
                    out_root, datatype, source_event, condition,
                    test_participants, top_features, minimal_imu_set
                )

                if X_test is None or len(X_test) == 0:
                    print(f"  No test data available for {source_event} - {condition}")
                    continue

                # Standardise features
                means, stds = DeviationAnalysis.load_feature_means_stds(results_dir, target_event, condition, datatype)
                X_test = DeviationAnalysis.standardise_X(X_test, means, stds)

                # Compute reconstruction error
                recon_errors, percent_errors, X_recon, scores = \
                    DeviationAnalysis.compute_reconstruction_error(pc_target, X_test, n_modes=None, m_weight=0.01)
                
                z_scores = (percent_errors - base_mu) / base_sd if base_mu is not None and base_sd is not None else None
                
                pp_z = DeviationAnalysis.per_participant_mean_errors(groups_test, z_scores) if z_scores is not None else None
                pp_percent = DeviationAnalysis.per_participant_mean_errors(groups_test, percent_errors)

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
                    'pca_scores': scores,
                    'source_event': source_event,
                    'target_event': target_event,
                    'condition': condition,
                    'baseline_mean_percent_error': float(base_mu) if base_mu is not None else None,
                    'baseline_sd_percent_error': float(base_sd) if base_sd is not None else None,
                    'z_scores': z_scores,
                    'mean_z': float(np.mean(z_scores)) if z_scores is not None else None,
                    'std_z': float(np.std(z_scores, ddof=1)) if z_scores is not None else None,
                    'median_z': np.median(z_scores) if z_scores is not None else None,
                    'per_participant_mean_percent_error': pp_percent,
                    'per_participant_mean_z': pp_z,
                    'mean_pp_z': float(pp_z.mean()) if pp_z is not None else None,
                    'median_pp_z': float(pp_z.median()) if pp_z is not None else None
                }

                pair_results[condition] = condition_result

                print(f"  Mean reconstruction error: {recon_errors.mean():.4f} ± {recon_errors.std():.4f}")
                print(f"  Mean percentage error: {percent_errors.mean():.2f}% ± {percent_errors.std():.2f}%")

            results_dict[pair_key] = pair_results

        return results_dict

    @staticmethod
    def deviation_from_normative_gait(out_root, results_dir, all_events, conditions,
                                        test_participants, minimal_results_dir=None,
                                        datatype="IMU", minimal_imu_set=True):
        """
        Project all events into Straight walk PCA model to assess deviation from normative gait.
        Each condition is projected into its respective Straight walk PCA model.
        """
        print("="*80)
        print("DEVIATION FROM NORMATIVE GAIT")
        print(f"Projecting events into their respective Straight walk PCA models")
        print(f"Conditions: {conditions}")
        print("="*80)

        normative_event = "Straight walk"

        results_dict = {}

        for event in all_events:
            print(f"\n{'='*60}")
            print(f"Testing event: {event}")
            print(f"{'='*60}")

            event_results = {}

            for condition in conditions:
                print(f"  Condition: {condition}")

                # Load Straight walk PCA model for THIS condition
                try:
                    pc_normative = DeviationAnalysis.load_pca_model(
                        results_dir, normative_event, condition, datatype
                    )
                    print(f"  Loaded Straight walk {condition} PCA model")
                    print(f"    Number of components: {pc_normative.modes.shape[1]}")
                except Exception as e:
                    print(f"  Error loading Straight walk {condition} PCA model: {e}")
                    continue

                # Load top features for Straight walk
                try:
                    top_features = DeviationAnalysis.load_top_features(
                        results_dir, normative_event, condition, datatype
                    )
                    print(f"  Loaded {len(top_features)} top features")
                except Exception as e:
                    print(f"  Error loading top features: {e}")
                    
                # Baseline: Straight walk -> Straight walk model (same condition)
                X_base, y_base, groups_base = DeviationAnalysis.load_test_data(
                    out_root, datatype, normative_event, condition,
                    test_participants, top_features, minimal_imu_set
                )
                
                if X_base is None or len(X_base) == 0:
                    print(f"  No baseline data available for Straight walk - {condition}")
                    continue

                means, stds = DeviationAnalysis.load_feature_means_stds(results_dir, normative_event, condition, datatype)
                X_base = DeviationAnalysis.standardise_X(X_base, means, stds)

                _, base_percent_errors, _, _ = DeviationAnalysis.compute_reconstruction_error(
                    pc_normative, X_base, n_modes=None, m_weight=0.01)
                
                base_mu, base_sd, base_pp = DeviationAnalysis.baseline_stats_from_participants(groups_base, base_percent_errors)

                print(f"  Baseline (Straight walk->Straight walk) per-participant mean percentage error: {base_mu:.2f}% ± {base_sd:.2f}%")

                # Load event data for this condition
                X_test, y_test, groups_test = DeviationAnalysis.load_test_data(
                    out_root, datatype, event, condition,
                    test_participants, top_features, minimal_imu_set
                )

                if X_test is None or len(X_test) == 0:
                    print(f"  No test data available for {event} - {condition}")
                    continue

                # Standardise features
                means, stds = DeviationAnalysis.load_feature_means_stds(results_dir, normative_event, condition, datatype)
                X_test = DeviationAnalysis.standardise_X(X_test, means, stds)

                # Compute reconstruction error
                recon_errors, percent_errors, X_recon, scores = \
                    DeviationAnalysis.compute_reconstruction_error(pc_normative, X_test, n_modes=None, m_weight=0.01)
                
                # Compute z-scores relative to baseline
                z_scores = (percent_errors - base_mu) / base_sd

                pp_z = DeviationAnalysis.per_participant_mean_errors(groups_test, z_scores) if z_scores is not None else None
                pp_percent = DeviationAnalysis.per_participant_mean_errors(groups_test, percent_errors)

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
                    'pca_scores': scores,
                    'baseline_mean_percent_error': float(base_mu) if base_mu is not None else None,
                    'baseline_sd_percent_error': float(base_sd) if base_sd is not None else None,
                    'z_scores': z_scores,
                    'mean_z': float(np.mean(z_scores)) if z_scores is not None else None,
                    'std_z': float(np.std(z_scores, ddof=1)) if z_scores is not None else None,
                    'median_z': np.median(z_scores) if z_scores is not None else None,
                    'per_participant_mean_percent_error': pp_percent,
                    'per_participant_mean_z': pp_z
                }

                event_results[condition] = condition_result

                print(f"  Mean reconstruction error: {recon_errors.mean():.4f} ± {recon_errors.std():.4f}")
                print(f"  Mean percentage error: {percent_errors.mean():.2f}% ± {percent_errors.std():.2f}%")

            results_dict[event] = event_results

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
    def save_per_participant_errors(results_dict, output_dir, analysis_type):
        """
        Save per-participant mean percentage errors for each event/condition.
        """
        os.makedirs(output_dir, exist_ok=True)
        rows = []
        for event_key, event_val in results_dict.items():
            # event_val is dict of conditions or a single dict
            if isinstance(event_val, dict) and 'participants' in event_val:
                # Single result (shouldn't happen for your current analyses)
                df = pd.DataFrame({
                    'Participant': event_val['participants'],
                    'Percentage_Error': event_val['percentage_errors'],
                    'Event': event_key,
                    'Condition': 'Normal'
                })
                per_participant = df.groupby(['Event', 'Condition', 'Participant'])['Percentage_Error'].mean().reset_index()
                rows.append(per_participant)
            else:
                for cond_key, cond_val in event_val.items():
                    df = pd.DataFrame({
                        'Participant': cond_val['participants'],
                        'Percentage_Error': cond_val['percentage_errors'],
                        'Event': event_key,
                        'Condition': cond_key
                    })
                    per_participant = df.groupby(['Event', 'Condition', 'Participant'])['Percentage_Error'].mean().reset_index()
                    rows.append(per_participant)
        if rows:
            all_df = pd.concat(rows, ignore_index=True)
            out_path = os.path.join(output_dir, f"{analysis_type}_per_participant_errors.csv")
            all_df.to_csv(out_path, index=False)
            print(f"Saved per-participant errors to: {out_path}")
        else:
            print("No per-participant errors to save.")

@staticmethod
def save_per_participant_zscores(results_dict, output_dir, analysis_type):
    """
    Save per-participant mean z-scores for each event/condition (if z_scores exist).
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for event_key, event_val in results_dict.items():
        # event_val should be dict of conditions
        if isinstance(event_val, dict) and 'participants' in event_val:
            # single-result case
            if event_val.get('z_scores') is None:
                continue
            df = pd.DataFrame({
                'Participant': event_val['participants'],
                'Z_Score': event_val['z_scores'],
                'Event': event_key,
                'Condition': 'Normal'
            })
            per_participant = df.groupby(['Event', 'Condition', 'Participant'])['Z_Score'].mean().reset_index()
            rows.append(per_participant)
        else:
            for cond_key, cond_val in event_val.items():
                if cond_val.get('z_scores') is None:
                    continue
                df = pd.DataFrame({
                    'Participant': cond_val['participants'],
                    'Z_Score': cond_val['z_scores'],
                    'Event': event_key,
                    'Condition': cond_key
                })
                per_participant = df.groupby(['Event', 'Condition', 'Participant'])['Z_Score'].mean().reset_index()
                rows.append(per_participant)

    if rows:
        all_df = pd.concat(rows, ignore_index=True)
        out_path = os.path.join(output_dir, f"{analysis_type}_per_participant_zscores.csv")
        all_df.to_csv(out_path, index=False)
        print(f"Saved per-participant z-scores to: {out_path}")
    else:
        print("No per-participant z-scores to save (z_scores missing or None).")

    @staticmethod
    def visualise_condition_deviation(results_dict, output_dir):
        """
        Create a single grouped boxplot for condition-based deviation analysis.
        Each event is a group, with AR and VR side-by-side, using consistent color coding.
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

        # Create grouped boxplots (single axes)
        fig, ax = plt.subplots(figsize=(14, 8))

        # Use the same color code as event-based deviation
        condition_colours = {'AR': '#2ECC71', 'VR': '#E74C3C'}
        conditions_order = [c for c in ['AR', 'VR'] if c in df_plot['Condition'].unique()]
        events_order = list(df_plot['Event'].unique())

        # Build data arrays in event-condition order
        data = []
        positions = []
        box_colors = []
        width = 0.22
        gap = 0.15

        for e_idx, event_name in enumerate(events_order):
            base = e_idx * (len(conditions_order) * width + gap)
            for c_idx, cond in enumerate(conditions_order):
                vals = df_plot[(df_plot['Event'] == event_name) & (df_plot['Condition'] == cond)]['Percentage_Error'].values
                if len(vals) == 0:
                    continue
                data.append(vals)
                positions.append(base + c_idx * width)
                box_colors.append(condition_colours.get(cond, '#999999'))

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="black", linewidth=1.2),
            capprops=dict(color="black", linewidth=1.2),
            boxprops=dict(linewidth=1.2)
        )

        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)
            patch.set_edgecolor('black')

        # X ticks: centre under each event group
        group_centers = [
            e_idx * (len(conditions_order) * width + gap) + (len(conditions_order) - 1) * width / 2.0
            for e_idx in range(len(events_order))
        ]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(events_order, rotation=45, ha='right', fontsize=16)

        ax.set_xlabel('Event', fontsize=18, fontweight='bold')
        ax.set_ylabel('Reconstruction Error (%)', fontsize=18, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Legend
        legend_handles = [Patch(facecolor=condition_colours[c], edgecolor='black', label=c) for c in conditions_order]
        ax.legend(handles=legend_handles, title='Condition', loc='best', fontsize=16, title_fontsize=16)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "condition_deviation_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved condition deviation plot: {plot_path}")

    @staticmethod
    def visualise_event_deviation(results_dict, output_dir, title):
        """
        Create visualisation for event-based deviation analysis using box plots.
        Shows per-participant distribution of reconstruction errors with conditions side-by-side.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data for plotting - aggregate per participant
        plot_data = []
        for key, result in results_dict.items():
            if isinstance(result, dict) and 'percentage_errors' in result:
                # Old format (single condition) - shouldn't happen with new code
                per_participant = pd.DataFrame({
                    'Participant': result['participants'],
                    'Percentage_Error': result['percentage_errors']
                }).groupby('Participant')['Percentage_Error'].mean().reset_index()

                for _, row in per_participant.iterrows():
                    plot_data.append({
                        'Event': key,
                        'Condition': 'Normal',
                        'Percentage_Error': row['Percentage_Error']
                    })
            else:
                # New format (multiple conditions)
                for condition, cond_result in result.items():
                    per_participant = pd.DataFrame({
                        'Participant': cond_result['participants'],
                        'Percentage_Error': cond_result['percentage_errors']
                    }).groupby('Participant')['Percentage_Error'].mean().reset_index()

                    for _, row in per_participant.iterrows():
                        plot_data.append({
                            'Event': key,
                            'Condition': condition,
                            'Percentage_Error': row['Percentage_Error']
                        })

        df_plot = pd.DataFrame(plot_data)

        # Create grouped boxplots
        fig, ax = plt.subplots(figsize=(14, 8))

        # Define colours for conditions
        condition_colours = {'Normal': '#3182BD', 'AR': '#2ECC71', 'VR': '#E74C3C'}
        conditions_order = [c for c in ['Normal', 'AR', 'VR'] if c in df_plot['Condition'].unique()]
        events_order = list(df_plot['Event'].unique())

        # Build data arrays in event-condition order
        data = []
        positions = []
        box_colors = []
        width = 0.22
        gap = 0.15

        for e_idx, event_name in enumerate(events_order):
            base = e_idx * (len(conditions_order) * width + gap)
            for c_idx, cond in enumerate(conditions_order):
                vals = df_plot[(df_plot['Event'] == event_name) & (df_plot['Condition'] == cond)]['Percentage_Error'].values
                if len(vals) == 0:
                    continue
                data.append(vals)
                positions.append(base + c_idx * width)
                box_colors.append(condition_colours.get(cond, '#999999'))

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="black", linewidth=1.2),
            capprops=dict(color="black", linewidth=1.2),
            boxprops=dict(linewidth=1.2)
        )

        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)
            patch.set_edgecolor('black')

        # X ticks: centre under each event group
        group_centers = [
            e_idx * (len(conditions_order) * width + gap) + (len(conditions_order) - 1) * width / 2.0
            for e_idx in range(len(events_order))
        ]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(events_order, rotation=45, ha='right', fontsize=16)

        ax.set_xlabel('Event Projection', fontsize=18, fontweight='bold')
        ax.set_ylabel('Reconstruction Error (%)', fontsize=18, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Legend
        legend_handles = [Patch(facecolor=condition_colours[c], edgecolor='black', label=c) for c in conditions_order]
        ax.legend(handles=legend_handles, title='Condition', loc='best', fontsize=16, title_fontsize=16)

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
    test_participants = [
        "P001", "P003", "P004", "P005", "P006", 
        "P007", "P008", "P010", "P011", "P012",
        "P014", "P016", "P017", "P018", "P019",
        "P020", "P021", "P022", "P023", "P024",
        "P025", "P026", "P027", "P028", "P030",
        "P031", "P032", "P033", "P035", "P043"
    ]

    parser = argparse.ArgumentParser(description="PCA Deviation Analysis")
    parser.add_argument('--condition', action='store_true', help='Run condition-based deviation analysis')
    parser.add_argument('--normative', action='store_true', help='Run deviation from normative gait analysis')
    parser.add_argument('--related', action='store_true', help='Run deviation between related events analysis')
    parser.add_argument('--self_fit', action='store_true', help='Fit Normal data on Normal PCA models')

    args = parser.parse_args()

    print("="*80)
    print("PCA DEVIATION ANALYSIS")
    print("Using Minimal IMU Set: Head_imu and RightForeArm_imu")
    print(f"Test participants: {test_participants}")
    print("="*80)

    if args.condition:
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

        # Save per-participant errors (for later combined boxplot)
        DeviationAnalysis.save_per_participant_errors(
            condition_results, condition_output_dir, "condition_based"
        )

        DeviationAnalysis.save_per_participant_zscores(
            condition_results, condition_output_dir, "condition_based"
        )

        # Do NOT visualise condition-based results here
        # DeviationAnalysis.visualise_condition_deviation(
        #     condition_results, condition_output_dir
        # )

    if args.normative:
        print("\n\n")
        events_to_test = [e for e in all_events if e != "Straight walk"]

        normative_results = DeviationAnalysis.deviation_from_normative_gait(
            out_root=out_root,
            results_dir=results_dir,
            all_events=events_to_test,
            conditions=["Normal", "AR", "VR"],
            test_participants=test_participants,
            minimal_results_dir=minimal_results_dir,
            datatype=datatype,
            minimal_imu_set=minimal_imu_set
        )

        normative_output_dir = os.path.join(output_dir, "normative_gait")
        normative_summary = DeviationAnalysis.save_deviation_results(
            normative_results, normative_output_dir, "normative_gait"
        )

        # Save per-participant errors
        DeviationAnalysis.save_per_participant_errors(
            normative_results, normative_output_dir, "normative_gait"
        )

        DeviationAnalysis.save_per_participant_zscores(
            normative_results, normative_output_dir, "normative_gait"
        )

        # Visualise normative gait results
        DeviationAnalysis.visualise_event_deviation(
            normative_results, normative_output_dir,
            "Deviation from Normative Gait (Straight Walk)"
        )

    if args.related:
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
            conditions=["Normal", "AR", "VR"],
            test_participants=test_participants,
            minimal_results_dir=minimal_results_dir,
            datatype=datatype,
            minimal_imu_set=minimal_imu_set
        )

        related_output_dir = os.path.join(output_dir, "related_events")
        related_summary = DeviationAnalysis.save_deviation_results(
            related_events_results, related_output_dir, "related_events"
        )

        # Save per-participant errors
        DeviationAnalysis.save_per_participant_errors(
            related_events_results, related_output_dir, "related_events"
        )

        DeviationAnalysis.save_per_participant_zscores(
            related_events_results, related_output_dir, "related_events"
        )

        # Visualise related events results
        DeviationAnalysis.visualise_event_deviation(
            related_events_results, related_output_dir,
            "Deviation Between Biomechanically Related Events"
        )

    if args.self_fit:
        print("\n\n")
        self_fit_results = DeviationAnalysis.condition_based_deviation_analysis(
            out_root=out_root,
            results_dir=results_dir,
            events=all_events,
            test_conditions=["Normal"],
            test_participants=test_participants,
            minimal_results_dir=minimal_results_dir,
            datatype=datatype,
            minimal_imu_set=minimal_imu_set
        )

        self_fit_output_dir = os.path.join(output_dir, "self_fit_normal")
        self_fit_summary = DeviationAnalysis.save_deviation_results(
            self_fit_results, self_fit_output_dir, "self_fit_normal"
        )

        self_fit_per_participant_errors = DeviationAnalysis.save_per_participant_errors(
            self_fit_results, self_fit_output_dir, "self_fit_normal"
        )

        DeviationAnalysis.save_per_participant_zscores(
            self_fit_results, self_fit_output_dir, "self_fit_normal"
        )

        # Visualise
        DeviationAnalysis.visualise_condition_deviation(
            self_fit_results, self_fit_output_dir
        )

    if not (args.condition or args.normative or args.related or args.self_fit):
        print("No analysis selected. Use --condition, --normative, or --related to run a specific analysis.")

    print("\n\n")
    print("="*80)
    print("DEVIATION ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nAnalysis complete!")
