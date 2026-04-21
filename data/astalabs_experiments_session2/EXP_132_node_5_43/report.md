# Experiment 132: node_5_43

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_43` |
| **ID in Run** | 132 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:13:07.743944+00:00 |
| **Runtime** | 368.2s |
| **Parent** | `node_4_35` |
| **Children** | `node_6_67` |
| **Creation Index** | 133 |

---

## Hypothesis

> The 'Deployment-Gate' Failure: There is no statistically significant difference
in the rate of 'Real World Testing' between AI systems in the 'Development'
stage and those in the 'Production' stage, implying that testing is not
effectively acting as a deployment gate.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.5806 (Maybe True) |
| **Posterior** | 0.1978 (Likely False) |
| **Surprise** | -0.4594 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 21.0 |
| Uncertain | 0.0 |
| Maybe False | 7.0 |
| Definitely False | 2.0 |

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

**Objective:** Evaluate if real-world testing frequency increases significantly as systems move to production.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'eo13960_scored'.
- 2. Group '16_dev_stage' into 'Pre-Production' (Planned, Development) and 'Post-Production' (Implemented, Operational).
- 3. Create binary 'Testing_Status' from '53_real_world_testing'.
- 4. Perform a Chi-square test (or Z-test for two proportions) to compare testing rates between the two lifecycle groups.

### Deliverables
- Comparison of testing rates (percentages); Statistical test results confirming or rejecting the difference.

---

## Analysis

The experiment successfully corrected the data mapping logic and executed the
analysis on 140 relevant AI systems from the EO 13960 dataset.

**Findings:**
1.  **Distribution:** The analysis compared 46 Pre-Production systems
(Development/Planned) against 94 Post-Production systems
(Operational/Maintenance).
2.  **Testing Rates:** A stark contrast was observed. Only **8.7%** of Pre-
Production systems had undergone 'Real World Testing' (typically relying on
benchmarks instead), whereas **87.2%** of Post-Production systems had completed
such testing.
3.  **Statistical Significance:** The Chi-Square test yielded a p-value of
**0.0000**, overwhelmingly rejecting the null hypothesis that testing rates are
similar across stages.

**Conclusion:**
The results refute the 'Deployment-Gate Failure' hypothesis. The data indicates
that Real World Testing is a strong differentiator between development and
production phases, suggesting that it effectively functions as a deployment gate
for the vast majority (87.2%) of systems. However, a porousness of ~13% remains
where systems are in production without documented operational testing.

---

## Review

The experiment successfully tested the 'Deployment-Gate Failure' hypothesis
after correcting for data mapping issues in the 'Real World Testing' column. The
final analysis correctly segmented 140 AI systems into 'Pre-Production' (n=46)
and 'Post-Production' (n=94) stages.

**Findings:**
1.  **Hypothesis Rejection:** The results strongly refute the hypothesis that
testing rates are stagnant across the lifecycle. There is a statistically
significant difference (Chi-Square p < 0.001) between stages.
2.  **Gate Effectiveness:** 'Real World Testing' acts as a clear deployment
gate. Only 8.7% of Pre-Production systems had completed operational testing
(relying mostly on benchmarks), whereas 87.2% of Post-Production systems had
validated performance in an operational environment.
3.  **Operational Gap:** While the gate functions for the majority, the analysis
identified a ~13% 'porousness' where systems are in production without
documented operational testing.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# [debug]
print("Starting corrected analysis...")

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Inspect columns
stage_col = '16_dev_stage'
test_col = '53_real_world_testing'

# --- REFINED MAPPING LOGIC ---

def map_stage(val):
    if pd.isna(val):
        return np.nan
    val_lower = str(val).lower()
    
    # Post-Production keywords
    if any(x in val_lower for x in ['operation', 'maintenance', 'in production', 'in mission', 'implementation', 'deployed']):
        return 'Post-Production'
    # Pre-Production keywords
    elif any(x in val_lower for x in ['acquisition', 'development', 'initiated', 'planned', 'design']):
        return 'Pre-Production'
    else:
        return np.nan

def map_testing_strict(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    
    # Strict prefix/keyword matching based on known unique values
    if val_str.startswith('Performance evaluation') or val_str.startswith('Impact evaluation') or val_str.lower() == 'yes':
        return 'Yes'
    elif val_str.startswith('No testing') or val_str.startswith('Benchmark evaluation') or val_str.startswith('Agency CAIO'):
        return 'No'
    else:
        # Fallback for unexpected strings, treat as NaN to be safe
        return np.nan

# Apply mappings
eo_data['stage_group'] = eo_data[stage_col].apply(map_stage)
eo_data['testing_binary'] = eo_data[test_col].apply(map_testing_strict)

# Filter for analysis (drop NaNs in relevant cols)
analysis_df = eo_data.dropna(subset=['stage_group', 'testing_binary']).copy()

# Debug print to verify fix
print(f"Data points for analysis: {len(analysis_df)}")
print("\nDistribution of Testing Status (Corrected):")
print(analysis_df['testing_binary'].value_counts())
print("\nDistribution of Stage Group:")
print(analysis_df['stage_group'].value_counts())

# Create Contingency Table
contingency_table = pd.crosstab(analysis_df['stage_group'], analysis_df['testing_binary'])
print("\nContingency Table (Count):")
print(contingency_table)

# Calculate Percentages
contingency_pct = pd.crosstab(analysis_df['stage_group'], analysis_df['testing_binary'], normalize='index') * 100
print("\nContingency Table (Percentage within Stage):")
print(contingency_pct)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")

# Visualization
plt.figure(figsize=(8, 6))

# Plot 'Yes' rates if available
if 'Yes' in contingency_pct.columns:
    ax = contingency_pct['Yes'].plot(kind='bar', color=['skyblue', 'salmon'], edgecolor='black')
    plt.title('Rate of Real-World Operational Testing by Stage')
    plt.ylabel('Percentage of Systems Tested (%)')
    plt.xlabel('Lifecycle Stage')
    plt.ylim(0, 100)
    plt.xticks(rotation=0)

    # Annotate
    for p_rect in ax.patches:
        h = p_rect.get_height()
        ax.annotate(f"{h:.1f}%", (p_rect.get_x() + p_rect.get_width() / 2., h),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
else:
    print("No 'Yes' data to plot.")

plt.tight_layout()
plt.show()

# Conclusion generator
alpha = 0.05
print("\n--- Conclusion ---")
if p < alpha:
    print("Result: Statistically Significant Difference.")
    pre_rate = contingency_pct.loc['Pre-Production', 'Yes']
    post_rate = contingency_pct.loc['Post-Production', 'Yes']
    print(f"Pre-Production Testing Rate: {pre_rate:.1f}%")
    print(f"Post-Production Testing Rate: {post_rate:.1f}%")
    if post_rate > pre_rate:
        print("Observation: Testing increases significantly as systems move to production (Gate functioning).")
    else:
        print("Observation: Testing decreases significantly in production (Potential issue).")
else:
    print("Result: No Statistically Significant Difference.")
    print("Observation: The rate of real-world testing does not statistically differ between development and production stages.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting corrected analysis...
Data points for analysis: 140

Distribution of Testing Status (Corrected):
testing_binary
Yes    86
No     54
Name: count, dtype: int64

Distribution of Stage Group:
stage_group
Post-Production    94
Pre-Production     46
Name: count, dtype: int64

Contingency Table (Count):
testing_binary   No  Yes
stage_group             
Post-Production  12   82
Pre-Production   42    4

Contingency Table (Percentage within Stage):
testing_binary          No        Yes
stage_group                          
Post-Production  12.765957  87.234043
Pre-Production   91.304348   8.695652

Chi-Square Test Results:
Chi2 Statistic: 77.1249
p-value: 0.0000

--- Conclusion ---
Result: Statistically Significant Difference.
Pre-Production Testing Rate: 8.7%
Post-Production Testing Rate: 87.2%
Observation: Testing increases significantly as systems move to production (Gate functioning).


=== Plot Analysis (figure 1) ===
Based on the analysis of the provided plot, here is the detailed description:

### 1. Plot Type
*   **Type:** Vertical Bar Plot (Bar Chart).
*   **Purpose:** The plot is designed to compare categorical data, specifically the prevalence (percentage) of real-world operational testing occurring at two different stages of a system's lifecycle.

### 2. Axes
*   **Y-Axis:**
    *   **Title/Label:** "Percentage of Systems Tested (%)"
    *   **Range:** The axis runs from 0 to 100, with tick marks at intervals of 20 units (0, 20, 40, 60, 80, 100).
*   **X-Axis:**
    *   **Title/Label:** "Lifecycle Stage"
    *   **Categories:** Two distinct categories are displayed: "Post-Production" and "Pre-Production".

### 3. Data Trends
*   **Tallest Bar:** The "Post-Production" bar (colored sky blue) is the dominant category, reaching a substantial height relative to the scale.
*   **Shortest Bar:** The "Pre-Production" bar (colored salmon/light red) is significantly shorter.
*   **Pattern/Comparison:** There is a stark disparity between the two stages. Testing in the Post-Production phase is nearly 10 times more prevalent than testing in the Pre-Production phase based on the visual height differences.

### 4. Annotations and Legends
*   **Value Labels:** Exact percentage values are annotated directly above each bar for precision:
    *   Post-Production: **87.2%**
    *   Pre-Production: **8.7%**
*   **Title:** The chart is titled "Rate of Real-World Operational Testing by Stage".
*   **Color Coding:** Distinct colors are used to differentiate the stages visually, with blue for Post-Production and red/salmon for Pre-Production.

### 5. Statistical Insights
*   **Dominance of Post-Production Testing:** The data suggests that the overwhelming majority of real-world operational testing (87.2%) occurs after the system has been produced. This indicates a reliance on validating systems in the field or after deployment rather than during development.
*   **Lack of Pre-Production Validation:** Only a small fraction (8.7%) of systems undergo real-world operational testing before production. This could imply that pre-production testing is either conducted in simulated/lab environments rather than "real-world" conditions, or that it is frequently skipped entirely in favor of fixing issues post-release.
*   **Completeness of Data:** The sum of the two values is 95.9%. This suggests that approximately 4.1% of systems are either not tested in these specific real-world operational contexts, fall into a different undefined stage, or that the data set has a small margin of unclassified results.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
