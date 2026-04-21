# Experiment 32: node_3_14

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_14` |
| **ID in Run** | 32 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:30:24.450927+00:00 |
| **Runtime** | 175.3s |
| **Parent** | `node_2_4` |
| **Children** | `node_4_8` |
| **Creation Index** | 33 |

---

## Hypothesis

> The 'Integration Readiness' bundle exhibits significantly higher internal
cohesion (similarity between control mappings) than the 'Trust Readiness'
bundle, implying that engineering requirements are more standardized while
governance requirements are more heterogeneous.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7321 (Likely True) |
| **Posterior** | 0.1737 (Likely False) |
| **Surprise** | -0.6481 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 24.0 |
| Uncertain | 2.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 90.0 |

---

## Experiment Plan

**Objective:** Quantify and compare the internal consistency of control mappings within each bundle.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Extract the binary vectors for the 18 architecture controls for all rows.
- 3. Split the dataset into two subsets based on 'bundle'.
- 4. For each subset, calculate the pairwise Jaccard similarity between all row vectors.
- 5. Flatten the similarity matrices to obtain a distribution of similarity scores for each bundle.
- 6. Perform a Mann-Whitney U test to determine if the median similarity in 'Integration Readiness' is significantly higher than in 'Trust Readiness'.

### Deliverables
- Boxplots of Jaccard similarity scores for both bundles and statistical test results.

---

## Analysis

The experiment successfully calculated and compared the internal consistency of
the two competency bundles. Contrary to the hypothesis that 'Integration
Readiness' would exhibit higher cohesion, the results indicate the opposite.
'Trust Readiness' vectors showed higher internal similarity (Mean: 0.2516,
Median: 0.2000) compared to 'Integration Readiness' (Mean: 0.1866, Median:
0.0000). The Mann-Whitney U test (p=0.9948 for Integration > Trust) confirms
that Integration Readiness is not significantly more cohesive; in fact, the
distribution suggests Governance/Trust requirements tend to map to overlapping
sets of broad controls, whereas Technical/Integration requirements map to more
distinct, orthogonal controls (resulting in many zero-similarity pairs). The
visualization reinforces this, showing a collapsed distribution near zero for
Integration Readiness.

---

## Review

The experiment successfully tested the hypothesis regarding the internal
cohesion of competency bundles. Contrary to the expectation that 'Integration
Readiness' (engineering) would be more standardized and cohesive, the results
indicate that 'Trust Readiness' (governance) exhibits higher internal
consistency. The mean pairwise Jaccard similarity for Trust Readiness was 0.2516
(Median: 0.20) compared to 0.1866 (Median: 0.00) for Integration Readiness. The
Mann-Whitney U test for the hypothesis 'Integration > Trust' yielded a p-value
of 0.995, leading to the rejection of the hypothesis. This implies that
governance requirements tend to map to broad, overlapping sets of architecture
controls (high redundancy), whereas technical requirements map to specific,
orthogonal controls (low overlap).

---

## Code

```python
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

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Successfully loaded step2_crosswalk_matrix.csv
Identified 18 architecture control columns.
Trust Readiness samples: 19
Integration Readiness samples: 23

--- Trust Readiness Internal Consistency ---
Mean Jaccard Similarity: 0.2516
Median Jaccard Similarity: 0.2000
Std Dev: 0.3024
Count of pairs: 171

--- Integration Readiness Internal Consistency ---
Mean Jaccard Similarity: 0.1866
Median Jaccard Similarity: 0.0000
Std Dev: 0.2785
Count of pairs: 253

--- Mann-Whitney U Test ---
U-statistic: 18745.0
P-value (one-sided, Integration > Trust): 9.9478e-01
Result: Not Statistically Significant.

STDERR:
<ipython-input-1-20dae8ca516a>:103: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot([trust_sims, integ_sims], labels=['Trust Readiness', 'Integration Readiness'])


=== Plot Analysis (figure 1) ===
Based on the image provided, here is a detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Box Plot (also known as a Box-and-Whisker Plot).
*   **Purpose:** This plot visualizes the distribution of numerical data (Jaccard Similarity Scores) across distinct categorical groups ("Bundles"). It displays statistical summaries such as the minimum, first quartile (Q1), median, third quartile (Q3), maximum (excluding outliers), and outliers.

### 2. Axes
*   **X-Axis:**
    *   **Label:** Represents the categories of "Bundles".
    *   **Categories:** "Trust Readiness" and "Integration Readiness".
*   **Y-Axis:**
    *   **Label:** "Jaccard Similarity Score".
    *   **Range:** 0.0 to 1.0 (representing a normalized similarity ratio).
    *   **Units:** The score is a dimensionless ratio/coefficient between 0 and 1.

### 3. Data Trends
*   **Trust Readiness (Left Box):**
    *   **Median:** The median line (orange) is located at **0.2**.
    *   **Spread:** The interquartile range (the box) spans from **0.0 (Q1)** to **0.5 (Q3)**.
    *   **Range:** The whiskers extend from **0.0** all the way to **1.0**, indicating that the data covers the full spectrum of possible similarity scores.
*   **Integration Readiness (Right Box):**
    *   **Median:** The median line is at **0.0**, indicating that more than half of the pairwise comparisons in this bundle have zero similarity.
    *   **Spread:** The interquartile range spans from **0.0** to approximately **0.33**.
    *   **Range:** The top whisker extends to approximately **0.65**.
    *   **Outliers:** There is a distinct outlier (represented by a circle) at **1.0**, indicating at least one pair (or a cluster of pairs) with perfect similarity, despite the generally low scores of the group.

### 4. Annotations and Legends
*   **Title:** "Distribution of Pairwise Jaccard Similarities by Bundle" clearly defines the context of the data.
*   **Grid Lines:** Horizontal dashed grid lines appear at intervals of 0.2 (0.0, 0.2, 0.4, etc.) to assist in estimating the values of the box plot elements.
*   **Color Coding:** The boxes are white with black borders, and the median is marked in **orange**.

### 5. Statistical Insights
*   **Higher Overall Similarity in Trust Readiness:** The "Trust Readiness" bundle shows a higher tendency for similarity compared to "Integration Readiness." Its median is higher (0.2 vs 0.0), and its upper quartile reaches 0.5 compared to roughly 0.33 for the other group.
*   **Zero-Inflated Data:** Both groups have a first quartile (Q1) of 0.0, and "Integration Readiness" even has a median of 0.0. This suggests that for a significant portion of pairs in both bundles, there is no overlap (Jaccard similarity of 0). This is particularly dominant in the "Integration Readiness" bundle.
*   **Variability:** "Trust Readiness" exhibits higher variability. Its distribution stretches consistently from 0 to 1. In contrast, "Integration Readiness" is more compressed toward the lower end (0 to ~0.65), except for the specific outliers at 1.0.
*   **Outlier Behavior:** While "Integration Readiness" has generally lower similarity scores, the presence of outliers at 1.0 suggests that there are specific, isolated cases within that bundle that are identical, whereas the rest are largely dissimilar.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
