# Experiment 92: node_5_17

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_17` |
| **ID in Run** | 92 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:25:30.208862+00:00 |
| **Runtime** | 363.7s |
| **Parent** | `node_4_12` |
| **Children** | `node_6_49` |
| **Creation Index** | 93 |

---

## Hypothesis

> The 'Paperwork vs. Protection' Gap: Governance requirements categorized under
the NIST AI RMF 'MAP' function map to significantly fewer technical architecture
controls than those under the 'MANAGE' function.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.3563 (Maybe False) |
| **Surprise** | -0.4627 |
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
| Maybe False | 32.0 |
| Definitely False | 24.0 |

---

## Experiment Plan

**Objective:** Determine if 'Map' requirements are less technically operationalizable than 'Manage' requirements.

### Steps
- 1. Load 'context_crosswalk_evidence.json' (or the 'step2_crosswalk_matrix' subset).
- 2. Group records by the 'function' field (e.g., MAP vs MANAGE).
- 3. For each record, count the number of unique items in the 'applicable_controls' (or 'arch_controls') list.
- 4. Perform a One-Way ANOVA or T-test comparing the control counts between the two functions.

### Deliverables
- Box plot of Control Counts by NIST Function; Statistical comparison results.

---

## Analysis

The experiment was successfully executed. The programmer correctly resolved
previous data issues by merging the CSV dataset (containing the 'function'
classification) with the JSON dataset (containing the detailed
'applicable_controls' list) using the 'req_id' key.

### Hypothesis Verification:
The hypothesis that **'MAP' requirements map to significantly fewer technical
controls than 'MANAGE' requirements** was **rejected**.

*   **Observed Trend**: Contrary to the hypothesis, 'MAP' requirements actually
mapped to a higher average number of controls (Mean = 2.33) compared to 'MANAGE'
requirements (Mean = 1.80).
*   **Statistical Significance**: The difference was not statistically
significant. A T-test yielded a p-value of 0.0998 (t=1.8353), which exceeds the
standard alpha of 0.05.
*   **Data Insight**: The 'MANAGE' function showed high rigidity, with most
requirements mapping to exactly 2 controls, whereas 'MAP' showed slightly more
variability (2 to 3 controls).

The analysis successfully integrated the split datasets to provide a valid
statistical test, concluding that there is no evidence to support the claim that
'MAP' requirements are less technically operationalizable in this framework.

---

## Review

The experiment was successfully executed. The programmer correctly resolved
previous data issues by merging the CSV dataset (containing the 'function'
classification) with the JSON dataset (containing the detailed
'applicable_controls' list) using the 'req_id' key.

### Hypothesis Verification:
The hypothesis that **'MAP' requirements map to significantly fewer technical
controls than 'MANAGE' requirements** was **rejected**.

*   **Observed Trend**: Contrary to the hypothesis, 'MAP' requirements actually
mapped to a higher average number of controls (Mean = 2.33) compared to 'MANAGE'
requirements (Mean = 1.80).
*   **Statistical Significance**: The difference was not statistically
significant. A T-test yielded a p-value of 0.0998 (t=1.8353), which exceeds the
standard alpha of 0.05.
*   **Data Insight**: The 'MANAGE' function showed high rigidity, with most
requirements mapping to exactly 2 controls, whereas 'MAP' showed slightly more
variability (2 to 3 controls).

The analysis successfully integrated the split datasets to provide a valid
statistical test, concluding that there is no evidence to support the claim that
'MAP' requirements are less technically operationalizable in this framework.

---

## Code

```python
import pandas as pd
import json
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# 1. Load CSV for 'function' and 'req_id'
csv_path = 'astalabs_discovery_all_data.csv'
if not os.path.exists(csv_path):
    csv_path = f"../{csv_path}"

df_csv = pd.read_csv(csv_path, low_memory=False)

# Filter for crosswalk matrix
csv_subset = df_csv[df_csv['source_table'] == 'step2_crosswalk_matrix'].copy()

# Keep relevant columns
if 'function' not in csv_subset.columns or 'req_id' not in csv_subset.columns:
    print("Error: Missing columns in CSV")
    exit(1)

# Normalize req_id in CSV (strip whitespace)
csv_subset['req_id'] = csv_subset['req_id'].astype(str).str.strip()
csv_data = csv_subset[['req_id', 'function']]

print(f"CSV Matrix Rows: {len(csv_data)}")
print(f"CSV Head req_id: {csv_data['req_id'].head().tolist()}")

# 2. Load JSON for 'req_id' and 'applicable_controls'
json_path = 'context_crosswalk_evidence.json'
if not os.path.exists(json_path):
    json_path = f"../{json_path}"

with open(json_path, 'r') as f:
    json_list = json.load(f)

df_json = pd.DataFrame(json_list)
# Normalize req_id in JSON
df_json['req_id'] = df_json['req_id'].astype(str).str.strip()

print(f"JSON Rows: {len(df_json)}")
print(f"JSON Head req_id: {df_json['req_id'].head().tolist()}")

# 3. Merge
# Try inner join
merged = pd.merge(csv_data, df_json, on='req_id', how='inner')
print(f"Merged Rows: {len(merged)}")

# If merge failed (0 rows) and counts match (42), try positional merge as fallback
if len(merged) == 0 and len(csv_data) == 42 and len(df_json) == 42:
    print("Warning: Merge on req_id failed. Attempting positional merge based on row order.")
    csv_data = csv_data.reset_index(drop=True)
    df_json = df_json.reset_index(drop=True)
    merged = pd.concat([csv_data, df_json.drop(columns=['req_id'])], axis=1)
    print(f"Positional Merge Rows: {len(merged)}")

# 4. Calculate Control Counts
def calc_len(x):
    if isinstance(x, list):
        return len(x)
    return 0

merged['control_count'] = merged['applicable_controls'].apply(calc_len)

# 5. Analyze MAP vs MANAGE
target_functions = ['MAP', 'MANAGE']
analysis_set = merged[merged['function'].isin(target_functions)].copy()

print(f"Analysis Set Size: {len(analysis_set)}")
print(analysis_set.groupby('function')['control_count'].describe())

map_data = analysis_set[analysis_set['function'] == 'MAP']['control_count']
manage_data = analysis_set[analysis_set['function'] == 'MANAGE']['control_count']

if len(map_data) > 0 and len(manage_data) > 0:
    t_stat, p_val = stats.ttest_ind(map_data, manage_data, equal_var=False)
    print(f"\nT-test Results (MAP vs MANAGE):\nT-statistic: {t_stat:.4f}\nP-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Result: Significant difference.")
    else:
        print("Result: No significant difference.")

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.boxplot([map_data, manage_data], tick_labels=['MAP', 'MANAGE'])
    plt.title('Technical Controls: MAP vs MANAGE')
    plt.ylabel('Count of Controls')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("Insufficient data for analysis.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: CSV Matrix Rows: 42
CSV Head req_id: ['NIST-GV-1', 'NIST-GV-2', 'NIST-GV-3', 'NIST-GV-4', 'NIST-GV-5']
JSON Rows: 42
JSON Head req_id: ['NIST-GV-1', 'NIST-GV-2', 'NIST-GV-3', 'NIST-GV-4', 'NIST-GV-5']
Merged Rows: 42
Analysis Set Size: 11
          count      mean       std  min  25%  50%   75%  max
function                                                     
MANAGE      5.0  1.800000  0.447214  1.0  2.0  2.0  2.00  2.0
MAP         6.0  2.333333  0.516398  2.0  2.0  2.0  2.75  3.0

T-test Results (MAP vs MANAGE):
T-statistic: 1.8353
P-value: 0.0998
Result: No significant difference.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (also known as a Box-and-Whisker Plot).
*   **Purpose:** This plot compares the statistical distribution of a numerical variable ("Count of Controls") between two distinct categorical groups ("MAP" and "MANAGE"). It visualizes the median, quartiles, and potential outliers for each group.

### 2. Axes
*   **X-Axis:**
    *   **Label:** Represents the categories. The distinct groups are **"MAP"** and **"MANAGE"**.
*   **Y-Axis:**
    *   **Title:** **"Count of Controls"**.
    *   **Range:** The axis is marked from **1.00 to 3.00** with increments of 0.25.
    *   **Units:** Integer counts (implied by the variable name and integer data points).

### 3. Data Trends
*   **MAP Group (Left):**
    *   **Median:** The orange line indicates the median value is **2.0**.
    *   **Spread:** The data ranges from a minimum of **2.0** to a maximum of **3.0**.
    *   **Quartiles:** The box extends from 2.0 (Q1/lower quartile) to 2.75 (Q3/upper quartile). This indicates that the upper 50% of the data shows some variance between 2 and 3, while the lower 50% is clustered strictly at 2.
    *   **Shape:** The distribution is skewed upwards, as there is no bottom whisker (the minimum equals the median).

*   **MANAGE Group (Right):**
    *   **Median:** The median is also **2.0**.
    *   **Spread/Consistency:** This group is highly consistent. The box is collapsed into a flat line at 2.0, meaning the 25th percentile, median, and 75th percentile are all likely the same value (2).
    *   **Outliers:** There is a distinct outlier represented by a hollow circle at **1.0**.

### 4. Annotations and Legends
*   **Chart Title:** "Technical Controls: MAP vs MANAGE" at the top center.
*   **Grid Lines:** Horizontal dashed grey lines are provided at 0.25 intervals to facilitate reading exact Y-axis values.
*   **Outlier Marker:** A small circle is used to denote data points that fall statistically far from the rest of the dataset (visible in the MANAGE column at y=1).

### 5. Statistical Insights
*   **Central Tendency:** Both the MAP and MANAGE processes generally utilize the same number of technical controls on average, with a median of **2 controls** for both.
*   **Variability:**
    *   **MAP** is more variable on the higher end. While its baseline is 2 controls, a significant portion of the data points extend up to 3 controls.
    *   **MANAGE** is extremely rigid. Almost all data points result in exactly 2 controls, showing very little variance.
*   **Anomalies:** The MANAGE process has a rare instance (outlier) where only 1 control was used, whereas the MAP process never drops below 2.
*   **Conclusion:** If consistency is the goal, MANAGE is the more consistent process (mostly 2s). If a higher count of controls is preferred/safer, MAP appears "safer" as it ranges between 2 and 3, whereas MANAGE has a risk of dropping to 1.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
