import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# [debug] Listing files to confirm location
# import os
# print(os.listdir('../'))

# 1. Load the dataset
file_path = '../step2_crosswalk_matrix.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}")
except FileNotFoundError:
    # Fallback if running in same directory
    df = pd.read_csv('step2_crosswalk_matrix.csv')
    print("Successfully loaded step2_crosswalk_matrix.csv (local)")

# 2. Identify architecture control columns
# Based on metadata, first 6 columns are metadata: 
# req_id, source, function, requirement, bundle, competency_statement
# The rest are architecture controls.
metadata_cols = ['req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement']
control_cols = [c for c in df.columns if c not in metadata_cols]

print(f"Identified {len(control_cols)} control columns.")

# 3. Calculate sum of mappings for each control
# The matrix contains 'X' or similar for mappings, or NaN/empty for no mapping.
# We count non-null and non-empty string values.
control_counts = {}

for col in control_cols:
    # Count non-NA and non-empty strings
    count = df[col].notna().sum()
    # If the format uses empty strings instead of NaN, handle that:
    # But read_csv usually handles empty as NaN. Let's verify by checking specific values if needed.
    # Assuming standard CSV 'X' or NaN.
    control_counts[col] = count

# Convert to Series for easier handling
counts_series = pd.Series(control_counts).sort_values(ascending=False)

# 4. Sort and analyze distribution
total_mappings = counts_series.sum()
print(f"\nTotal control mappings observed: {total_mappings}")

# Top 20% of controls (18 controls * 0.2 = 3.6 -> Top 4 controls)
top_n = int(np.ceil(len(control_cols) * 0.2))
top_n_controls = counts_series.head(top_n)
top_n_sum = top_n_controls.sum()
top_n_perc = (top_n_sum / total_mappings) * 100

print(f"\nTop {top_n} controls (Top 20%):")
print(top_n_controls)
print(f"Cumulative coverage of Top 20% controls: {top_n_perc:.2f}%")

# 5. Chi-Square Goodness of Fit Test
# Null Hypothesis: Mappings are uniformly distributed across all 18 controls.
# Expected frequency for each control = Total Mappings / 18
expected_freq = total_mappings / len(control_cols)
expected_counts = [expected_freq] * len(control_cols)
observed_counts = counts_series.values

chi2_stat, p_val = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

print(f"\nChi-Square Goodness of Fit Test:")
print(f"Chi2 Statistic: {chi2_stat:.4f}")
print(f"P-value: {p_val:.4e}")

if p_val < 0.05:
    print("Result: Significant deviation from uniform distribution (Reject H0).")
else:
    print("Result: No significant deviation from uniform distribution (Fail to reject H0).")

# 6. Visualization
plt.figure(figsize=(12, 8))
bars = plt.barh(counts_series.index, counts_series.values, color='skyblue')
plt.xlabel('Number of Mappings')
plt.title('Frequency of Architecture Control Mappings (Pareto Check)')
plt.gca().invert_yaxis()  # Highest frequency at top

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
             f'{int(width)}', va='center')

plt.tight_layout()
plt.show()

# Print Gini Coefficient for inequality measure (optional but useful for Pareto)
def gini(array):
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

gini_score = gini(observed_counts)
print(f"\nGini Coefficient of Control Applicability: {gini_score:.4f}")
