# Experiment 33: node_3_15

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_15` |
| **ID in Run** | 33 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:34:32.969262+00:00 |
| **Runtime** | 181.2s |
| **Parent** | `node_2_4` |
| **Children** | None |
| **Creation Index** | 34 |

---

## Hypothesis

> Attacks that successfully trigger a 'Response Failure' are significantly more
complex (higher technique count) than those resulting in 'Prevention Failure',
suggesting that complex attacks are more likely to bypass initial defenses and
require reactive mitigation.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.7206 (Likely True) |
| **Surprise** | -0.0248 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

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
| Uncertain | 3.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Assess the relationship between failure modes and attack complexity.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Group the data by 'failure_mode' (Prevention Failure vs. Response Failure vs. Detection Failure).
- 3. Extract 'technique_count' for each group.
- 4. Perform a Kruskal-Wallis test to compare distributions across failure modes.
- 5. Calculate summary statistics (mean/median technique count) for each failure mode.

### Deliverables
- Comparative statistics and statistical test results indicating if failure mode is a function of attack complexity.

---

## Analysis

The experiment successfully loaded the dataset but revealed a critical data
distribution issue that prevented testing the specific hypothesis regarding
'Response Failure'. The 'step3_incident_coding.csv' dataset contains 52
incidents, but the 'failure_mode' column is heavily skewed, containing only
'prevention_failure' (n=51) and 'detection_failure' (n=1). There were zero
instances of 'response_failure', making it impossible to assess the complexity
of attacks triggering that specific failure mode.

Comparing the available groups, 'prevention_failure' incidents had a mean
technique count of 7.67 (Median=7), whereas the single 'detection_failure' had a
technique count of 1. A Kruskal-Wallis test between these two unequal groups
yielded a p-value of 0.0869, indicating no statistically significant difference
at the 95% confidence level, likely due to the single-sample size of the
detection group. Consequently, the hypothesis cannot be validated due to the
absence of the target class ('Response Failure') in the dataset.

---

## Review

The experiment was faithfully implemented, but the hypothesis could not be
tested due to data limitations identified during execution.

1. **Execution**: The code successfully loaded 'step3_incident_coding.csv',
grouped data by 'failure_mode', and attempted a Kruskal-Wallis test.
2. **Findings**: The analysis revealed a critical class imbalance. The dataset
contains 51 'prevention_failure' incidents (Mean complexity: 7.67 techniques)
and 1 'detection_failure' incident (Complexity: 1 technique). Crucially, there
are **zero** 'response_failure' incidents.
3. **Conclusion**: Because the 'response_failure' class is empty, the hypothesis
comparing its complexity to 'prevention_failure' cannot be statistically
evaluated. The Kruskal-Wallis test performed on the available groups (Prevention
vs. Detection) yielded a p-value of 0.0869, which is not significant, but this
result is largely an artifact of the single-sample 'detection' group.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# [debug] Check for file existence in parent directory
filename = 'step3_incident_coding.csv'
parent_path = os.path.join('..', filename)
local_path = filename

if os.path.exists(parent_path):
    filepath = parent_path
elif os.path.exists(local_path):
    filepath = local_path
else:
    # Fallback to creating a dummy dataset if file is missing (for robust execution in unknown envs)
    # However, per instructions, I should assume dataset is available.
    print(f"File {filename} not found in . or ..")
    filepath = None

if filepath:
    df = pd.read_csv(filepath)
    print(f"Successfully loaded {filepath}")
    print(f"Dataset shape: {df.shape}")

    # 2. Group by 'failure_mode' and analyze 'technique_count'
    if 'failure_mode' in df.columns and 'technique_count' in df.columns:
        
        # Descriptive Statistics
        print("\n=== Technique Count Statistics by Failure Mode ===")
        stats_summary = df.groupby('failure_mode')['technique_count'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        print(stats_summary)

        # Prepare data for Statistical Test
        groups = []
        group_names = []
        for name, group in df.groupby('failure_mode'):
            groups.append(group['technique_count'].values)
            group_names.append(name)

        # 3. Perform Kruskal-Wallis Test
        # We use Kruskal-Wallis because sample sizes are likely very unequal and distributions may not be normal.
        if len(groups) > 1:
            h_stat, p_val = stats.kruskal(*groups)
            print("\n=== Kruskal-Wallis H-test ===")
            print(f"H-statistic: {h_stat:.4f}")
            print(f"P-value: {p_val:.4f}")
            
            alpha = 0.05
            if p_val < alpha:
                print("Conclusion: Significant difference in attack complexity between failure modes.")
            else:
                print("Conclusion: No significant difference in attack complexity between failure modes.")
        else:
            print("\nNot enough failure mode groups to perform statistical testing.")

        # 4. Visualization
        plt.figure(figsize=(10, 6))
        plt.boxplot(groups, labels=group_names)
        plt.title('Distribution of Attack Complexity (Technique Count) by Failure Mode')
        plt.ylabel('Technique Count (Complexity)')
        plt.xlabel('Failure Mode')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        
    else:
        print("Required columns 'failure_mode' or 'technique_count' not found.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Successfully loaded step3_incident_coding.csv
Dataset shape: (52, 22)

=== Technique Count Statistics by Failure Mode ===
                    count      mean  median      std  min  max
failure_mode                                                  
detection_failure       1  1.000000     1.0      NaN    1    1
prevention_failure     51  7.666667     7.0  2.81188    3   16

=== Kruskal-Wallis H-test ===
H-statistic: 2.9316
P-value: 0.0869
Conclusion: No significant difference in attack complexity between failure modes.

STDERR:
<ipython-input-1-09d2363696c9>:62: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(groups, labels=group_names)


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (also known as a Box-and-Whisker Plot).
*   **Purpose:** This plot compares the distribution of a quantitative variable ("Technique Count") across two different categorical groups ("detection_failure" and "prevention_failure"). It visualizes the median, quartiles, range, and outliers for each group.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Failure Mode"
    *   **Categories:** The axis displays two discrete categories: `detection_failure` and `prevention_failure`.
*   **Y-Axis:**
    *   **Label:** "Technique Count (Complexity)"
    *   **Range:** The visual scale extends from roughly 0 to 17.
    *   **Units:** Integer counts representing the number of techniques involved in an attack.
    *   **Ticks:** Major grid lines mark every 2 units (2, 4, 6, ..., 16).

### 3. Data Trends
*   **"detection_failure" Category:**
    *   **Pattern:** The plot for this category appears as a single flattened line (the median line).
    *   **Value:** It sits at the value of **1**.
    *   **Spread:** There is effectively zero variance shown here. The minimum, maximum, and median are all identical. This indicates that all recorded "detection_failure" events involved an attack complexity of exactly 1 technique.
*   **"prevention_failure" Category:**
    *   **Spread:** This category shows a significantly wider distribution of data.
    *   **Median:** The central orange line indicates a median technique count of approximately **7**.
    *   **Interquartile Range (IQR):** The box (representing the middle 50% of the data) ranges from a count of roughly **6 (Q1)** to **9 (Q3)**.
    *   **Whiskers:** The bottom whisker extends down to **3**, and the top whisker extends up to **12**.
    *   **Outliers:** There are three distinct outliers plotted as circles above the top whisker, located at values of **14, 15, and 16**.

### 4. Annotations and Legends
*   **Title:** "Distribution of Attack Complexity (Technique Count) by Failure Mode" appears at the top, summarizing the chart's content.
*   **Grid Lines:** Horizontal dashed grey lines are included to assist in reading the Y-axis values across the plot area.
*   **Colors:** The boxes are outlined in black with orange horizontal lines representing the medians. Outliers are represented by empty circles.

### 5. Statistical Insights
*   **Complexity Correlation:** There is a stark statistical difference between the two failure modes. **Detection failures** are associated with extremely low-complexity attacks (single technique), whereas **prevention failures** occur against attacks with much higher complexity (median of 7 techniques).
*   **Variability:** The "detection_failure" mode is highly predictable regarding complexity (no variance), while "prevention_failure" is highly variable, suggesting that prevention mechanisms fail under a wide variety of stress loads, ranging from 3 techniques to 16.
*   **Extreme Cases:** The presence of outliers only in the "prevention_failure" group (at 14, 15, and 16) suggests that as attack complexity becomes extreme, the likelihood of a prevention failure persists, whereas detection failures do not appear to be recorded at these high complexities in this dataset.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
