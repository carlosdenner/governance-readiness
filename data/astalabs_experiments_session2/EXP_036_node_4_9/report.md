# Experiment 36: node_4_9

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_9` |
| **ID in Run** | 36 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T02:43:27.408243+00:00 |
| **Runtime** | 351.5s |
| **Parent** | `node_3_8` |
| **Children** | `node_5_1`, `node_5_93` |
| **Creation Index** | 37 |

---

## Hypothesis

> Commercial AI 'Black Boxes': AI systems procured commercially (COTS) exhibit
significantly lower Tier 2 Governance scores (deep safeguards like bias
mitigation and independent evaluation) compared to custom-developed systems, due
to lack of vendor transparency.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2527 (Likely False) |
| **Surprise** | -0.5870 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 30.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Compare deep governance compliance (Tier 2 scores) between Commercial and Custom AI systems using '22_dev_method' as the grouping variable and text-mining the governance columns.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `eo13960_scored`.
- 2. Create a grouping variable `acquisition_type` based on `22_dev_method`: map 'Developed with contracting resources.' to 'Commercial' and 'Developed in-house.' to 'Custom'. Filter out other values to ensure a clean comparison.
- 3. Implement a text-scoring function to convert descriptive text in Tier 2 columns into binary scores (1=Compliant, 0=Non-compliant):
    - `52_impact_assessment`: Score 1 if it strictly contains 'Yes'.
    - `62_disparity_mitigation`: Score 1 if text length > 10 chars AND contains keywords ('test', 'eval', 'monitor', 'bias', 'fairness') AND does NOT contain ('None', 'N/A', 'waived').
    - `55_independent_eval`: Score 1 if text contains 'Yes'.
    - `53_real_world_testing`: Score 1 if text contains 'operational environment'.
    - `56_monitor_postdeploy`: Score 1 if text contains 'monitoring' OR 'Automated' OR 'Established Process', provided it does NOT contain 'No monitoring'.
- 4. Sum these binary indicators to create a `tier2_score` (0-5) for each system.
- 5. Calculate summary statistics (mean, std dev) for Commercial vs. Custom groups.
- 6. Perform a Mann-Whitney U test to determine if the difference in distributions is statistically significant.
- 7. Generate a boxplot visualizing the score distributions.

### Deliverables
- Boxplot of Tier 2 Scores by Acquisition Type; Statistical test results (Mann-Whitney U) and summary statistics.

---

## Analysis

The experiment was successfully executed using '22_dev_method' as the grouping
variable, resolving the previous data error. The analysis processed 920 records
(481 Commercial, 439 Custom). The text-mining approach successfully converted
descriptive governance fields into quantitative scores.

The results **reject the hypothesis**. There is no statistically significant
difference in Tier 2 Governance scores between Commercial (Mean=0.31) and Custom
(Mean=0.34) systems (Mann-Whitney U p=0.18). The boxplot and summary statistics
reveal a strong 'floor effect,' where the median score for both groups is 0 out
of 5. This indicates that deep governance safeguards (like independent
evaluation and bias mitigation) are largely absent in *both* procurement models,
rather than being a specific deficit of commercial 'black boxes'.

---

## Review

The experiment was successfully executed and the hypothesis was tested
efficiently. By using '22_dev_method' as a proxy for acquisition type, the
analysis compared 481 Commercial systems against 439 Custom systems. The text-
mining approach successfully converted unstructured governance descriptions into
quantitative scores (0-5 scale). The results lead to the **rejection of the
hypothesis**: there is no statistically significant difference in deep
governance compliance between Commercial (Mean=0.31) and Custom (Mean=0.34)
systems (Mann-Whitney U p=0.18). Instead, the data reveals a systemic 'floor
effect' where the median score for both groups is 0, indicating that deep
safeguards like bias mitigation and independent evaluation are largely absent
across the entire federal AI portfolio, regardless of whether the system is
bought or built.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

# [debug]
print("Starting experiment: Commercial vs Custom AI Governance Scores (Retry)")

# 1. Load the dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# 2. Filter for 'eo13960_scored' subset
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded EO 13960 data with {len(eo_data)} rows.")

# 3. Create Grouping Variable based on '22_dev_method'
# Mapping dictionary
dev_method_map = {
    'Developed with contracting resources.': 'Commercial (COTS)',
    'Developed in-house.': 'Custom (GOTS)'
}

# Clean column and map
eo_data['acquisition_type'] = eo_data['22_dev_method'].str.strip().map(dev_method_map)

# Filter for only the two groups of interest
eo_subset = eo_data.dropna(subset=['acquisition_type']).copy()
print(f"Data filtered for Commercial vs Custom comparison: {len(eo_subset)} rows")
print(eo_subset['acquisition_type'].value_counts())

# 4. Implement Text-Scoring Functions for Tier 2 columns

def score_impact_assessment(text):
    # 52_impact_assessment: Score 1 if strictly 'Yes' (ignoring case/whitespace)
    if pd.isna(text): return 0
    t = str(text).lower().strip()
    return 1 if t == 'yes' else 0

def score_disparity_mitigation(text):
    # 62_disparity_mitigation: Length > 10, keywords, not 'None/N/A'
    if pd.isna(text): return 0
    t = str(text).lower().strip()
    if len(t) < 10: return 0
    
    # Keywords indicating real content
    keywords = ['test', 'eval', 'monitor', 'bias', 'fairness', 'mitigat']
    has_keyword = any(k in t for k in keywords)
    
    # Exclusion keywords
    exclusions = ['none', 'n/a', 'waived', 'not applicable']
    is_excluded = any(e in t for e in exclusions)
    
    return 1 if (has_keyword and not is_excluded) else 0

def score_independent_eval(text):
    # 55_independent_eval: Contains 'Yes'
    if pd.isna(text): return 0
    t = str(text).lower().strip()
    return 1 if 'yes' in t else 0

def score_real_world_testing(text):
    # 53_real_world_testing: Contains 'operational environment'
    if pd.isna(text): return 0
    t = str(text).lower().strip()
    return 1 if 'operational environment' in t else 0

def score_postdeploy_monitoring(text):
    # 56_monitor_postdeploy: Monitor/Automated/Established AND NOT 'No monitoring'
    if pd.isna(text): return 0
    t = str(text).lower().strip()
    
    positive_signals = ['monitor', 'automated', 'established process']
    has_signal = any(s in t for s in positive_signals)
    
    negative_signals = ['no monitoring', 'not available']
    has_negative = any(n in t for n in negative_signals)
    
    return 1 if (has_signal and not has_negative) else 0

# Apply scoring
print("Applying text scoring to Tier 2 columns...")
eo_subset['s_52'] = eo_subset['52_impact_assessment'].apply(score_impact_assessment)
eo_subset['s_62'] = eo_subset['62_disparity_mitigation'].apply(score_disparity_mitigation)
eo_subset['s_55'] = eo_subset['55_independent_eval'].apply(score_independent_eval)
eo_subset['s_53'] = eo_subset['53_real_world_testing'].apply(score_real_world_testing)
eo_subset['s_56'] = eo_subset['56_monitor_postdeploy'].apply(score_postdeploy_monitoring)

# Sum scores
score_cols = ['s_52', 's_62', 's_55', 's_53', 's_56']
eo_subset['tier2_score'] = eo_subset[score_cols].sum(axis=1)

# 5. Statistical Analysis
commercial_scores = eo_subset[eo_subset['acquisition_type'] == 'Commercial (COTS)']['tier2_score']
custom_scores = eo_subset[eo_subset['acquisition_type'] == 'Custom (GOTS)']['tier2_score']

mean_comm = commercial_scores.mean()
std_comm = commercial_scores.std()
mean_cust = custom_scores.mean()
std_cust = custom_scores.std()

print(f"\n--- Summary Statistics (Tier 2 Score: 0-{len(score_cols)}) ---")
print(f"Commercial AI (n={len(commercial_scores)}): Mean={mean_comm:.2f}, Std={std_comm:.2f}")
print(f"Custom AI     (n={len(custom_scores)}):     Mean={mean_cust:.2f}, Std={std_cust:.2f}")

# Mann-Whitney U Test
u_stat, p_val = stats.mannwhitneyu(commercial_scores, custom_scores, alternative='two-sided')
print(f"\nMann-Whitney U Test Results:\nU-statistic: {u_stat}\nP-value: {p_val:.4e}")

alpha = 0.05
if p_val < alpha:
    print("Result: Significant difference detected.")
    if mean_comm < mean_cust:
        print("Direction: Commercial AI has LOWER governance scores (supporting hypothesis).")
    else:
        print("Direction: Commercial AI has HIGHER governance scores (contradicting hypothesis).")
else:
    print("Result: No significant difference detected.")

# 6. Visualization
plt.figure(figsize=(10, 6))
data_to_plot = [commercial_scores, custom_scores]
labels = [f'Commercial\n(n={len(commercial_scores)})', f'Custom\n(n={len(custom_scores)})']

plt.boxplot(data_to_plot, labels=labels, showmeans=True)
plt.title('Deep Governance (Tier 2) Compliance by Acquisition Type')
plt.ylabel('Governance Score (0-5)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment: Commercial vs Custom AI Governance Scores (Retry)
Loaded EO 13960 data with 1757 rows.
Data filtered for Commercial vs Custom comparison: 920 rows
acquisition_type
Commercial (COTS)    481
Custom (GOTS)        439
Name: count, dtype: int64
Applying text scoring to Tier 2 columns...

--- Summary Statistics (Tier 2 Score: 0-5) ---
Commercial AI (n=481): Mean=0.31, Std=1.02
Custom AI     (n=439):     Mean=0.34, Std=0.90

Mann-Whitney U Test Results:
U-statistic: 102447.0
P-value: 1.8205e-01
Result: No significant difference detected.

STDERR:
<ipython-input-1-1770d2d19a88>:135: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data_to_plot, labels=labels, showmeans=True)


=== Plot Analysis (figure 1) ===
Based on the provided image, here is a detailed analysis of the plot:

### 1. Plot Type
*   **Type:** This is a **Box Plot** (also known as a box-and-whisker plot).
*   **Purpose:** It compares the distribution of "Deep Governance (Tier 2) Compliance" scores between two different categorical groups: Commercial and Custom acquisition types. It is designed to visualize central tendency (median/mean), dispersion, and outliers.

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Governance Score (0-5)".
    *   **Range:** The axis spans from roughly -0.2 to 5.2, covering the discrete integer scoring range of 0 to 5.
    *   **Units:** The score is a discrete numerical value (likely an index or count) on a scale of 0 to 5.
*   **X-Axis:**
    *   **Label:** This axis categorizes the data by "Acquisition Type."
    *   **Categories:** The two labels are "Commercial" and "Custom."
    *   **Sample Size:** The axis labels include the sample count (n) for each group: Commercial (n=481) and Custom (n=439).

### 3. Data Trends
*   **Collapsed Distribution (The "Floor Effect"):** For both "Commercial" and "Custom" categories, the "box" portion of the plot (which typically represents the Interquartile Range, or the middle 50% of the data) is collapsed into a single orange line at the value **0**. This indicates that the 25th percentile, the median (50th percentile), and the 75th percentile are all 0.
*   **Medians:** The orange horizontal line represents the median, which is clearly situated at **0** for both groups.
*   **Means:** The small green triangles represent the mean (average). The mean is positioned slightly above 0 (estimated around 0.3 to 0.4), indicating the data is positively skewed.
*   **Outliers:** There are distinct open circles plotted at integers 1, 2, 3, 4, and 5 for both categories. In the context of a box plot where the box is flattened at 0, these points represent the minority of cases that achieved any score higher than 0.

### 4. Annotations and Legends
*   **Title:** "Deep Governance (Tier 2) Compliance by Acquisition Type."
*   **Gridlines:** Horizontal dashed gridlines are present at integer intervals (0, 1, 2, 3, 4, 5) to aid in reading the y-values of the outliers.
*   **Sample Size (n):** Annotations on the x-axis clarify the robustness of the dataset, showing a fairly balanced comparison (481 vs 439).

### 5. Statistical Insights
*   **Low Compliance Overall:** The most significant insight is that the vast majority of acquisitions—regardless of type—have a Deep Governance Score of **0**. The data is heavily right-skewed.
*   **Lack of Differentiation:** There is virtually no discernible difference in performance between "Commercial" and "Custom" acquisition types regarding this metric. Both display the same median (0), similar means (near 0), and a similar spread of outliers.
*   **Outlier Significance:** The presence of scores 1 through 5 as outliers suggests that achieving a high governance score is the exception rather than the rule. These "outliers" are the only positive performers in the dataset.
*   **Conclusion:** The acquisition type (Commercial vs. Custom) does not appear to be a predictor for Tier 2 Governance compliance, as both methods demonstrate a systematic failure to achieve scores above 0 for the majority of the population.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
