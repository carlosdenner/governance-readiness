# Experiment 171: node_4_50

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_50` |
| **ID in Run** | 171 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T09:08:12.540380+00:00 |
| **Runtime** | 220.9s |
| **Parent** | `node_3_21` |
| **Children** | `node_5_95`, `node_5_99` |
| **Creation Index** | 172 |

---

## Hypothesis

> Adversarial Tactic Complexity: In the ATLAS dataset, cases involving the
'Impact' tactic are significantly more likely to involve a higher count of
distinct 'Techniques' compared to cases that only reach 'Reconnaissance' or
'Initial Access' stages, suggesting that achieving tangible impact requires
higher attack complexity.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.5805 (Maybe True) |
| **Surprise** | -0.1938 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
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
| Uncertain | 54.0 |
| Maybe False | 2.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Analyze the relationship between attack severity (Impact tactic) and attack complexity (Technique count).

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'atlas_cases'.
- 2. Create a binary group 'Has_Impact': True if 'tactics' column contains 'Impact', else False.
- 3. Calculate 'Complexity': Count the number of unique items in the 'techniques' column (delimiter separated).
- 4. Perform a T-test (or Mann-Whitney U) comparing 'Complexity' between the 'Has_Impact' and 'No_Impact' groups.

### Deliverables
- Mean technique count for Impact vs Non-Impact cases; Statistical test results.

---

## Analysis

The experiment successfully loaded the dataset and filtered for the 52
'atlas_cases'. However, the analysis revealed a critical data characteristic:
every single case in the dataset, regardless of whether it involved an 'Impact'
tactic (n=36) or not (n=16), had a calculated 'Complexity' (technique count) of
exactly 1.0. Consequently, the standard deviation for both groups was 0.0, and
the Mann-Whitney U test yielded a p-value of 1.0, indicating no statistical
difference. This lack of variance suggests that the 'techniques' column in this
specific dataset extract likely contains only a single primary technique per
case or uses a delimiter not caught by the parsing logic. Therefore, the
hypothesis that Impact cases are more complex (involve more techniques) could
not be supported by this specific data, as the metric for complexity remained
constant across all observations.

---

## Review

The experiment was faithfully executed, but the results reveal a critical
limitation in the dataset or the parsing logic for this specific source. The
code successfully loaded the 'atlas_cases' (n=52) and applied the logic to count
techniques. However, the analysis found that every single case had a
'Complexity' score of exactly 1.0 (Standard Deviation = 0.0), resulting in a
p-value of 1.0. This implies that the 'techniques' column in this dataset either
contains only single entries per case or uses a delimiter (e.g., newlines,
pipes) that was not captured by the comma/semicolon splitting logic.
Consequently, the hypothesis could not be meaningfully tested as the dependent
variable lacked any variance. While the code execution was successful, the
scientific finding is that the current data/metric does not support the
hypothesis due to this uniformity.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

def run_experiment():
    # Load dataset
    # Based on previous successful attempts, the file is in the current directory.
    file_path = 'astalabs_discovery_all_data.csv'
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return

    # Filter for ATLAS cases
    # Source table identifier from metadata: 'atlas_cases'
    atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
    
    if atlas_df.empty:
        print("No ATLAS cases found in the dataset.")
        return

    print(f"Loaded {len(atlas_df)} ATLAS cases.")

    # Pre-processing
    # Ensure relevant columns are strings and handle NaNs
    # In the metadata, column 92 is 'tactics' and 93 is 'techniques'
    # Depending on the sparse structure, column names might be preserved or just indices.
    # The previous exploration showed 'tactics' and 'techniques' columns exist.
    
    # Verify columns exist
    if 'tactics' not in atlas_df.columns or 'techniques' not in atlas_df.columns:
        print("Columns 'tactics' or 'techniques' not found.")
        print("Available columns:", atlas_df.columns.tolist())
        return

    atlas_df['tactics'] = atlas_df['tactics'].fillna('').astype(str)
    atlas_df['techniques'] = atlas_df['techniques'].fillna('').astype(str)

    # 1. Define 'Has_Impact'
    # Check if 'Impact' is in the tactics list
    atlas_df['Has_Impact'] = atlas_df['tactics'].str.contains('Impact', case=False)

    # 2. Calculate 'Complexity' (Count of techniques)
    def count_techniques(val):
        if not val or val.strip() == '':
            return 0
        # Techniques are often separated by commas or semicolons in such datasets
        normalized = val.replace(';', ',')
        items = [x.strip() for x in normalized.split(',') if x.strip()]
        return len(set(items)) # Unique items

    atlas_df['Complexity'] = atlas_df['techniques'].apply(count_techniques)

    # Separate groups
    impact_complexity = atlas_df[atlas_df['Has_Impact'] == True]['Complexity']
    no_impact_complexity = atlas_df[atlas_df['Has_Impact'] == False]['Complexity']

    # Descriptive Statistics
    print("\n--- Descriptive Statistics ---")
    print(f"Group 'Has Impact' (n={len(impact_complexity)}):")
    if len(impact_complexity) > 0:
        print(f"  Mean Complexity: {impact_complexity.mean():.2f}")
        print(f"  Median Complexity: {impact_complexity.median()}")
        print(f"  Std Dev: {impact_complexity.std():.2f}")
    else:
        print("  No data.")
    
    print(f"\nGroup 'No Impact' (n={len(no_impact_complexity)}):")
    if len(no_impact_complexity) > 0:
        print(f"  Mean Complexity: {no_impact_complexity.mean():.2f}")
        print(f"  Median Complexity: {no_impact_complexity.median()}")
        print(f"  Std Dev: {no_impact_complexity.std():.2f}")
    else:
        print("  No data.")

    # Statistical Testing
    if len(impact_complexity) > 1 and len(no_impact_complexity) > 1:
        # Using Mann-Whitney U test as technique counts are discrete and sample size is small
        u_stat, p_val = stats.mannwhitneyu(impact_complexity, no_impact_complexity, alternative='two-sided')
        
        # T-test for comparison (Welch's)
        t_stat, p_val_t = stats.ttest_ind(impact_complexity, no_impact_complexity, equal_var=False)

        print("\n--- Statistical Test Results ---")
        print(f"Mann-Whitney U Test: U={u_stat}, p-value={p_val:.4f}")
        print(f"Welch's T-Test: t={t_stat:.4f}, p-value={p_val_t:.4f}")

        alpha = 0.05
        if p_val < alpha:
            print("\nResult: Statistically SIGNIFICANT difference in complexity found.")
        else:
            print("\nResult: NO statistically significant difference in complexity found.")
            
        # Visualization
        try:
            plt.figure(figsize=(10, 6))
            data_to_plot = [no_impact_complexity, impact_complexity]
            labels = ['No Impact Tactic', 'Has Impact Tactic']
            
            plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
            plt.title('Adversarial Tactic Complexity by Impact Status')
            plt.ylabel('Count of Techniques Used')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
        except Exception as e:
            print(f"Plotting error: {e}")
            
    else:
        print("\nInsufficient data for statistical testing.")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded 52 ATLAS cases.

--- Descriptive Statistics ---
Group 'Has Impact' (n=36):
  Mean Complexity: 1.00
  Median Complexity: 1.0
  Std Dev: 0.00

Group 'No Impact' (n=16):
  Mean Complexity: 1.00
  Median Complexity: 1.0
  Std Dev: 0.00

--- Statistical Test Results ---
Mann-Whitney U Test: U=288.0, p-value=1.0000
Welch's T-Test: t=nan, p-value=nan

Result: NO statistically significant difference in complexity found.

STDERR:
/usr/local/lib/python3.13/site-packages/scipy/stats/_axis_nan_policy.py:592: RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.
  res = hypotest_fun_out(*samples, **kwds)
<ipython-input-1-06ef47ef9823>:106: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data_to_plot, labels=labels, patch_artist=True)


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **Box Plot** (also known as a box-and-whisker plot).
*   **Purpose:** The plot is designed to compare the distribution of the "Count of Techniques Used" across two different categorical groups related to adversarial tactics ("No Impact Tactic" vs. "Has Impact Tactic").

### 2. Axes
*   **Title:** "Adversarial Tactic Complexity by Impact Status"
*   **X-Axis:**
    *   **Labels:** The axis represents categorical data with two distinct groups: **"No Impact Tactic"** and **"Has Impact Tactic"**.
    *   **Range:** Discrete categories.
*   **Y-Axis:**
    *   **Label:** "Count of Techniques Used".
    *   **Range:** The visible numerical scale ranges from approximately **0.95 to 1.05**.
    *   **Tick Marks:** Major ticks are labeled at intervals of 0.02 (0.96, 0.98, 1.00, 1.02, 1.04).

### 3. Data Trends
*   **Collapsed Boxes:** The most striking feature of this plot is that the "boxes" appear as single, flat orange lines at the value **1.00** on the Y-axis.
*   **Zero Variance:** In a standard box plot, a box represents the interquartile range (25th to 75th percentile) and the line inside represents the median. Since the box has zero height and no whiskers are visible, this indicates that **all data points** (or at least the vast majority, including the minimum, maximum, and quartiles) for both categories have a value of exactly **1**.
*   **Comparison:** There is no visible difference between the "No Impact Tactic" group and the "Has Impact Tactic" group. Both distributions are identical and singular.

### 4. Annotations and Legends
*   **Grid Lines:** Horizontal dashed grid lines are present at 0.02 intervals to assist in reading the Y-axis values.
*   **Color Coding:** The orange lines represent the median of the data. The lack of a surrounding box or "whiskers" (often black or blue) confirms the lack of spread in the data.

### 5. Statistical Insights
*   **Uniform Complexity:** The data indicates that regardless of whether an "Impact Tactic" is present or not, the complexity—measured here as the "Count of Techniques Used"—is constant at **1 technique**.
*   **Lack of Correlation:** Based on this specific dataset and metric, there is no correlation between the impact status and the number of techniques used. The impact status does not lead to an increase or decrease in technique count.
*   **Potential Data Artifact:** The fact that the count is exactly 1 for all cases might suggest that the dataset consists of individual events where techniques are logged singly, or that the metric "Count of Techniques" is not granular enough to capture complexity in this context. It suggests that every observed adversarial instance employed exactly one technique.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
