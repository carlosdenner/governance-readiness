# Experiment 40: node_5_0

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_0` |
| **ID in Run** | 40 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T02:55:12.628737+00:00 |
| **Runtime** | 209.0s |
| **Parent** | `node_4_4` |
| **Children** | `node_6_1` |
| **Creation Index** | 41 |

---

## Hypothesis

> Governance 'Rust': AI systems deployed prior to the 2020 AI Executive Order
(legacy systems) demonstrate significantly lower 'Governance Readiness' scores
than systems deployed post-2020, indicating a lack of retroactive compliance
efforts.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.2742 (Likely False) |
| **Posterior** | 0.0989 (Definitely False) |
| **Surprise** | -0.2104 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 3.0 |
| Uncertain | 1.0 |
| Maybe False | 21.0 |
| Definitely False | 5.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 2.0 |
| Definitely False | 58.0 |

---

## Experiment Plan

**Objective:** Investigate the relationship between system age and governance maturity.

### Steps
- 1. Load `eo13960_scored`.
- 2. Parse `20_date_implemented` to extract the year. Drop invalid dates.
- 3. Create a binary bin: 'Legacy' (<2021) and 'Modern' (>=2021).
- 4. Compare the mean `gov_readiness_pct` (or `total_gov_score`) between the two groups using an Independent Samples T-test.
- 5. Visualize the trend with a boxplot of scores by year.

### Deliverables
- T-test results, mean scores per group, and a boxplot visualization.

---

## Analysis

The experiment was successfully executed and provided clear data to evaluate the
'Governance Rust' hypothesis.

1.  **Hypothesis Rejection**: The hypothesis that pre-2021 'Legacy' systems
would demonstrate significantly lower governance readiness than 'Modern' systems
was **rejected**. The data actually showed a slight trend in the opposite
direction, with Legacy systems having a higher mean score (12.30%) compared to
Modern systems (9.92%).

2.  **Statistical Significance**: The Independent Samples T-test yielded a
p-value of **0.0714** (t=1.81). As this is above the standard alpha of 0.05, the
difference is not statistically significant. This indicates that the deployment
era (pre- or post-2020 Executive Order) has had no measurable impact on the
depth of governance controls recorded.

3.  **Systemic Low Maturity**: The analysis highlights a broader issue of low
governance maturity across the board. With means hovering around 10-12%, both
legacy and modern systems fail to demonstrate robust compliance with the
8-control composite index. This suggests that the 'Governance Gap' is a
universal attribute of the dataset, rather than a function of system age.

The code successfully generated all deliverables, including the t-test
statistics and a boxplot visualizing the comparable distributions.

---

## Review

The experiment was successfully executed and provided the necessary data to
evaluate the 'Governance Rust' hypothesis regarding the relationship between
system age and governance maturity.

1. **Hypothesis Rejection**: The hypothesis that 'Legacy' systems (pre-2021)
would demonstrate significantly lower governance readiness than 'Modern' systems
(post-2021) was rejected. Contrary to the prediction, 'Legacy' systems actually
showed a slightly higher mean governance score (12.30%) compared to 'Modern'
systems (9.92%).

2. **Statistical Significance**: The Independent Samples T-test resulted in a
p-value of 0.0714. This is above the standard significance threshold (p < 0.05),
indicating that the difference between the two cohorts is not statistically
significant. Consequently, the null hypothesis cannot be rejected; the era of
deployment (pre- or post-EO 13960) has no measurable impact on the depth of
governance controls recorded in this dataset.

3. **Key Finding - Universal Low Maturity**: The analysis reveals a 'floor
effect' across the entire dataset. With mean scores for both groups hovering
around 10-12%, there is a systemic lack of documented governance controls
regardless of whether the system is a legacy deployment or a modern one. This
suggests that the 'Governance Gap' is a fundamental characteristic of the
ecosystem, rather than a function of technical debt or system age.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import re

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if running locally in a different structure
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# --- 1. Date Parsing ---
# Function to extract year from various formats
def extract_year(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val)
    # Look for 4 consecutive digits
    match = re.search(r'(\d{4})', val_str)
    if match:
        year = int(match.group(1))
        # Basic sanity check for year range (e.g., 1980 to 2030)
        if 1980 <= year <= 2030:
            return year
    return np.nan

eo_df['impl_year'] = eo_df['20_date_implemented'].apply(extract_year)

# Filter out rows with no valid year
valid_date_df = eo_df.dropna(subset=['impl_year']).copy()

# --- 2. Construct Governance Score ---
# Select key binary/control columns representing governance maturity
gov_cols = [
    '28_iqa_compliance',        # Data Quality
    '40_has_ato',               # Security/Auth
    '52_impact_assessment',     # Impact Assessment
    '55_independent_eval',      # Independent Eval
    '56_monitor_postdeploy',    # Monitoring
    '61_adverse_impact',        # Adverse Impact check
    '62_disparity_mitigation',  # Bias Mitigation
    '65_appeal_process'         # Human Recourse
]

# Normalize to 0/1. strict check for 'Yes' case-insensitive
# Note: Some fields are verbose. We assume containing 'yes' implies presence of control.
for col in gov_cols:
    if col in valid_date_df.columns:
        valid_date_df[f'score_{col}'] = valid_date_df[col].astype(str).str.contains('yes', case=False, na=False).astype(int)
    else:
        valid_date_df[f'score_{col}'] = 0

score_cols = [f'score_{c}' for c in gov_cols]
valid_date_df['gov_score_raw'] = valid_date_df[score_cols].sum(axis=1)
valid_date_df['gov_score_pct'] = (valid_date_df['gov_score_raw'] / len(gov_cols)) * 100

# --- 3. Binning ---
# Legacy: < 2021 (Pre-2021)
# Modern: >= 2021 (2021 and later)
valid_date_df['cohort'] = valid_date_df['impl_year'].apply(lambda x: 'Modern (>=2021)' if x >= 2021 else 'Legacy (<2021)')

legacy_scores = valid_date_df[valid_date_df['cohort'] == 'Legacy (<2021)']['gov_score_pct']
modern_scores = valid_date_df[valid_date_df['cohort'] == 'Modern (>=2021)']['gov_score_pct']

# --- 4. Statistical Test ---
t_stat, p_val = stats.ttest_ind(legacy_scores, modern_scores, equal_var=False, nan_policy='omit')

# --- 5. Output Results ---
print("Analysis of Governance Scores by Implementation Era")
print("---------------------------------------------------")
print(f"Total records with valid dates: {len(valid_date_df)}")
print(f"Legacy Cohort Size (<2021):   {len(legacy_scores)}")
print(f"Modern Cohort Size (>=2021):  {len(modern_scores)}")
print("\nMean Governance Readiness Score (0-100%):")
print(f"  Legacy: {legacy_scores.mean():.2f}%")
print(f"  Modern: {modern_scores.mean():.2f}%")
print(f"\nDifference: {modern_scores.mean() - legacy_scores.mean():.2f}%")
print(f"T-test results: Statistic={t_stat:.4f}, p-value={p_val:.4e}")

if p_val < 0.05:
    print("Result: Statistically Significant Difference.")
else:
    print("Result: No Statistically Significant Difference.")

# --- 6. Visualization ---
plt.figure(figsize=(10, 6))
data_to_plot = [legacy_scores, modern_scores]
labels = [f'Legacy (<2021)\nn={len(legacy_scores)}', f'Modern (>=2021)\nn={len(modern_scores)}']

plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))
plt.title('Governance Readiness Scores: Pre- vs. Post-2021 Deployment')
plt.ylabel('Governance Score (%)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate with p-value
top_val = max(valid_date_df['gov_score_pct'].max(), 10) + 2
plt.text(1.5, top_val, f'p = {p_val:.4e}', ha='center', va='bottom', fontsize=12, color='red')

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Analysis of Governance Scores by Implementation Era
---------------------------------------------------
Total records with valid dates: 612
Legacy Cohort Size (<2021):   123
Modern Cohort Size (>=2021):  489

Mean Governance Readiness Score (0-100%):
  Legacy: 12.30%
  Modern: 9.92%

Difference: -2.38%
T-test results: Statistic=1.8147, p-value=7.1402e-02
Result: No Statistically Significant Difference.

STDERR:
<ipython-input-1-4fc7af1c841d>:99: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data_to_plot, labels=labels, patch_artist=True,


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (also known as a Box-and-Whisker plot).
*   **Purpose:** The plot compares the distribution of "Governance Readiness Scores" between two distinct groups: Legacy systems (deployed before 2021) and Modern systems (deployed in or after 2021). It visualizes the median, quartiles, variability, and outliers for each group.

### 2. Axes
*   **X-Axis:**
    *   **Label/Categories:** Two categorical groups: "Legacy (<2021)" and "Modern (>=2021)".
    *   **Context:** Each label includes the sample size ($n$), indicating the number of data points in each category ($n=123$ for Legacy, $n=489$ for Modern).
*   **Y-Axis:**
    *   **Label:** "Governance Score (%)".
    *   **Range:** The axis markings range from 0 to 50, with grid lines every 10 units. The data spans from a minimum of 0 to a maximum outlier of 50.

### 3. Data Trends
*   **Distribution Similarity:** The most striking trend is that the two box plots appear visually identical. Both groups exhibit the same interquartile range (IQR), median, whiskers, and outlier positions.
*   **Low Scores Dominance:** The "boxes" (representing the middle 50% of the data) are situated low on the graph, spanning roughly from 0% to 12.5%. This indicates that the majority of deployments in both groups have low governance readiness scores.
*   **Medians (Red Lines):** The red line (median) appears at the top of the blue box (approximately 12.5%) for both groups. This suggests a skew where the median and the 75th percentile (Q3) are very close or identical.
*   **Outliers:** Both groups display identical outlier patterns. There are distinct data points (represented by circles) at approximately 37.5% and 50%.
*   **Spread:** The whiskers extend from the top of the box to roughly 25%, indicating that the non-outlier maximum is fairly low. The minimum value for the IQR (bottom of the box) is 0%.

### 4. Annotations and Legends
*   **P-Value:** Located at the top center in red text: `p = 7.1407e-02`.
    *   Converted from scientific notation, this is roughly **0.071**.
*   **Sample Size ($n$):** Annotated under the x-axis labels.
    *   Legacy: $n=123$
    *   Modern: $n=489$
    *   This highlights an unbalanced dataset, where the Modern group is nearly four times larger than the Legacy group.

### 5. Statistical Insights
*   **Lack of Statistical Significance:** The p-value is **0.071**, which is greater than the standard significance threshold of 0.05 (5%). This indicates that there is **no statistically significant difference** in Governance Readiness Scores between the Legacy (<2021) and Modern (>=2021) deployments.
*   **Consistency Across Time:** Despite the difference in deployment eras and the significantly larger sample size for the Modern group, the governance readiness profile remains unchanged. The organization or system has not seen a measurable improvement (or decline) in these scores post-2021.
*   **Low Maturity Overall:** The data suggests that regardless of deployment time, governance readiness is generally low, with the 75th percentile of systems scoring only around 12.5%, and only a few outlier systems achieving scores up to 50%.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
