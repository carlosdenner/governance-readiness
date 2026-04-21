import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import sys

# Load the dataset
file_path = '../step2_crosswalk_matrix.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}")
except FileNotFoundError:
    # Fallback for local testing or if file is in current dir
    file_path = 'step2_crosswalk_matrix.csv'
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}")

# Identify metadata columns and control columns
# The first 6 columns are metadata: req_id, source, function, requirement, bundle, competency_statement
metadata_cols = ['req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement']
control_cols = [c for c in df.columns if c not in metadata_cols]

print(f"Identified {len(control_cols)} architecture control columns.")

# Preprocess data: Convert 'X' to 1, NaN to 0
df_controls = df.copy()
df_controls[control_cols] = df_controls[control_cols].notna().astype(int)

# Check for empty vectors (rows with all zeros)
row_sums = df_controls[control_cols].sum(axis=1)
zeros_count = (row_sums == 0).sum()
if zeros_count > 0:
    print(f"Warning: {zeros_count} rows map to 0 controls. These will result in NaN Jaccard similarities if compared with other zero vectors.")

# Split data by bundle
trust_df = df_controls[df_controls['bundle'] == 'Trust Readiness']
integ_df = df_controls[df_controls['bundle'] == 'Integration Readiness']

trust_vectors = trust_df[control_cols].values
integ_vectors = integ_df[control_cols].values

print(f"Trust Readiness samples: {len(trust_vectors)}")
print(f"Integration Readiness samples: {len(integ_vectors)}")

# Calculate pairwise Jaccard similarities
# pdist returns distances, so Similarity = 1 - Distance
# Jaccard distance is undefined if union is 0 (both vectors all zeros). 
# However, typically governance requirements map to at least one control. 

def get_jaccard_similarities(vectors):
    if len(vectors) < 2:
        return np.array([])
    # pdist computes pairwise distances
    dists = pdist(vectors, metric='jaccard')
    # Handle potential NaNs if any (0/0 case)
    dists = np.nan_to_num(dists, nan=1.0) # If both 0, union 0, dist often NaN. If we treat 0-0 as identical, dist=0. 
    # But Jaccard excludes 0-0 matches. Usually undefined. 
    # Let's filter NaNs from the similarity result instead.
    sims = 1 - dists
    return sims

trust_sims = get_jaccard_similarities(trust_vectors)
integ_sims = get_jaccard_similarities(integ_vectors)

# Filter out any NaNs if they persist (though pdist usually handles boolean vectors gracefully unless empty)
trust_sims = trust_sims[~np.isnan(trust_sims)]
integ_sims = integ_sims[~np.isnan(integ_sims)]

# Descriptive Statistics
print("\n--- Trust Readiness Internal Consistency ---")
print(f"Mean Jaccard Similarity: {np.mean(trust_sims):.4f}")
print(f"Median Jaccard Similarity: {np.median(trust_sims):.4f}")
print(f"Std Dev: {np.std(trust_sims):.4f}")
print(f"Count of pairs: {len(trust_sims)}")

print("\n--- Integration Readiness Internal Consistency ---")
print(f"Mean Jaccard Similarity: {np.mean(integ_sims):.4f}")
print(f"Median Jaccard Similarity: {np.median(integ_sims):.4f}")
print(f"Std Dev: {np.std(integ_sims):.4f}")
print(f"Count of pairs: {len(integ_sims)}")

# Hypothesis Testing: Mann-Whitney U Test
# H0: The distributions are equal.
# H1: Integration Readiness has higher similarity (stochastic dominance).
stat, p_value = mannwhitneyu(integ_sims, trust_sims, alternative='greater')

print("\n--- Mann-Whitney U Test ---")
print(f"U-statistic: {stat}")
print(f"P-value (one-sided, Integration > Trust): {p_value:.4e}")

alpha = 0.05
if p_value < alpha:
    print("Result: Statistically Significant. Integration Readiness controls are more internally cohesive.")
else:
    print("Result: Not Statistically Significant.")

# Visualization
plt.figure(figsize=(10, 6))
plt.boxplot([trust_sims, integ_sims], labels=['Trust Readiness', 'Integration Readiness'])
plt.title('Distribution of Pairwise Jaccard Similarities by Bundle')
plt.ylabel('Jaccard Similarity Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
