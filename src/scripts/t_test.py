
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

# Load data
input_file = "Z:/Upper Body/Results/30 Participants/event_fold_metrics.xlsx"
output_file = "Z:/Upper Body/Results/30 Participants/imu_subset_ttests.xlsx"

df = pd.read_excel(input_file)

print("Columns:", list(df.columns))
print("Events:", df['Event'].unique())
print("IMU configs:", df['IMU_config'].unique())

metrics = ["F1", "AUC"]

events = df['Event'].unique()

# Identify baseline (full) and minimal subsets
baseline = "Full_11"

all_configs = sorted(df['IMU_config'].unique())

# Minimal subsets are all IMU configs except baseline
minimal_configs = [c for c in all_configs if c != baseline]

comparisons = []

# Compare baseline  vs every minimal subset
for cfg in minimal_configs:
    comparisons.append((baseline, cfg))

# Compare minimal subsets against each other
for i in range(len(minimal_configs)):
    for j in range(i + 1, len(minimal_configs)):
        comparisons.append((minimal_configs[i], minimal_configs[j]))

print("\nConfigured comparisons")
for c in comparisons:
    print("  ", c)

# Helper function to compute Cohen's d for paired samples
def cohens_d_paired(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    diff = x - y
    if diff.std(ddof=1) == 0:
        return np.nan
    return diff.mean() / diff.std(ddof=1)

# Run paired t-tests
results = []

for event in events:
    event_df = df[df['Event'] == event]
    for metric in metrics:
        for cfg1, cfg2 in comparisons:

            sub1 = (
                event_df[event_df['IMU_config'] == cfg1]
                .loc[:, ["Fold", metric]]
                .sort_values(by="Fold")
            )
            sub2 = (
                event_df[event_df['IMU_config'] == cfg2]
                .loc[:, ["Fold", metric]]
                .sort_values(by="Fold")
            )

            # Align by fold using an inner join
            merged = pd.merge(sub1, sub2, on="Fold", suffixes=("_1", "_2"))

            if merged.empty:
                print(f"Warning: No overlapping folds for {event}, {metric}, {cfg1} vs {cfg2}")
                continue

            x = merged[f"{metric}_1"].values
            y = merged[f"{metric}_2"].values

            if len(x) != len(y) or len(x) == 0:
                print(f"Warning: Mismatched or empty data for {event}, {metric}, {cfg1} vs {cfg2}")
                continue

            # Paired t-test
            t_stat, p_val = ttest_rel(x, y)

            # Effect size (Cohen's d)
            d = cohens_d_paired(x, y)

            results.append({
                "Event": event,
                "Metric": metric,
                "Config_1": cfg1,
                "Config_2": cfg2,
                "n_folds": len(x),
                "Mean_1": x.mean(),
                "Mean_2": y.mean(),
                "Mean_Diff": (x - y).mean(),
                "t_stat": t_stat,
                "p_value": p_val,
                "Cohen_d": d
            })

# Save results
results_df = pd.DataFrame(results)

# Sort for readability
results_df = results_df.sort_values(by=["Event", "Metric", "Config_1", "Config_2"])

print("\nFirst few rows of results:")
print(results_df.head())

# Save to Excel
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    results_df.to_excel(writer, index=False, sheet_name='ttest_results')

print(f"\nT-test results saved to {output_file}")