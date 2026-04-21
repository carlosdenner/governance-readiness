# Experiment 231: node_6_46

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_46` |
| **ID in Run** | 231 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T11:51:02.730920+00:00 |
| **Runtime** | 385.0s |
| **Parent** | `node_5_37` |
| **Children** | None |
| **Creation Index** | 232 |

---

## Hypothesis

> The 'Evasion-Mitigation' Void: In adversarial cases (ATLAS), attacks classified
as 'Evasion' involve a significantly higher number of 'Competency Gaps' (missing
controls) than 'Poisoning' attacks, suggesting evasion is harder to
architecturally prevent.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.4176 (Maybe False) |
| **Surprise** | -0.3892 |
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
| Maybe False | 60.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Analyze if certain attack tactics expose more governance gaps than others.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `step3_incident_coding` (or join `atlas_cases` with coding).
- 2. Categorize cases based on `tactics_used` (contains 'Evasion' vs 'Poisoning').
- 3. Count the number of items in the `competency_gaps` column for each case (parse the list/string).
- 4. Perform a T-test comparing the average gap count between Evasion and Poisoning cases.

### Deliverables
- 1. Average Gap Count by Tactic Type.
- 2. Statistical test results.

---

## Analysis

The experiment tested the 'Evasion-Mitigation Void' hypothesis, which posits
that adversarial attacks using 'Evasion' tactics result in a higher number of
competency gaps (missing controls) compared to 'Poisoning' attacks.

**Methodology:**
- The programmer successfully mapped ATLAS tactic IDs to human-readable names
using the dataset's internal metadata, identifying 'AML.TA0007' as the primary
ID for Evasion in this specific dataset version.
- Cases were categorized into 'Evasion' (n=30) and 'Poisoning' (n=2) based on
these mappings.

**Findings:**
1. **Gap Counts:** The data showed extremely low variance.
   - **Poisoning:** All cases (100%) had exactly 1.0 competency gap.
   - **Evasion:** The mean gap count was 0.97, with the vast majority having 1.0
gap and a single outlier having 0 gaps.
2. **Statistical Comparison:** A Welch's T-test yielded a p-value of 0.3256,
indicating no statistically significant difference between the two groups.

**Conclusion:**
The hypothesis is **not supported**. The data indicates that 'Evasion' and
'Poisoning' attacks are associated with a nearly identical number of missing
controls (typically one primary gap per incident) in this dataset. The analysis
is limited by the very small sample size for Poisoning (n=2) and the lack of
variance in the 'Competency Gaps' metric.

---

## Review

The experiment was successfully executed. The programmer correctly resolved the
schema issue from previous attempts by mapping ATLAS tactic IDs (e.g.,
'AML.TA0007') to human-readable names, allowing for the categorization of
'Evasion' (n=30) and 'Poisoning' (n=2) cases. Although the sample size for
Poisoning was extremely small, the analysis was faithfully implemented.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
import sys
import os

# [debug] 
# print("# [debug] Starting mapping and analysis script...")

# Load dataset
filenames = ['astalabs_discovery_all_data.csv', '../astalabs_discovery_all_data.csv']
file_path = None
for fn in filenames:
    if os.path.exists(fn):
        file_path = fn
        break

if file_path is None:
    print("Error: Dataset file not found.")
    sys.exit(1)

df = pd.read_csv(file_path, low_memory=False)

# 1. Build a Mapping Dictionary for Tactic IDs to Names
# We look for rows where tactic_id and tactic_name are present
map_df = df.dropna(subset=['tactic_id', 'tactic_name'])[['tactic_id', 'tactic_name']].drop_duplicates()
id_to_name = dict(zip(map_df['tactic_id'], map_df['tactic_name']))

# Fallback/Augment: manual typical ATLAS mappings if not found in data, 
# but we prioritize data. 'Evasion' usually maps to 'Defense Evasion' (AML.TA0006)
# 'Poisoning' is tricky as it's often a technique, but let's see if the data defines it as a tactic or if we need to search techniques.

# Check if we have 'Poisoning' in our map
poisoning_ids = [k for k, v in id_to_name.items() if 'poisoning' in str(v).lower()]
evasion_ids = [k for k, v in id_to_name.items() if 'evasion' in str(v).lower()]

print("Found Tactic Mappings:")
print(f"  Poisoning IDs: {poisoning_ids}")
print(f"  Evasion IDs: {evasion_ids}")

# If we didn't find 'Poisoning' in tactics, we might need to look at techniques.
# Let's inspect 'technique_id' and 'technique_name' if they exist, or just search for the strings in 'techniques' column of atlas_cases
# The dataframe has 'step3_incident_coding' which has 'tactics_used' (IDs) and 'techniques_used' (IDs).

# Let's try to find technique mappings as well
tech_map = {}
if 'technique_id' in df.columns and 'technique_name' in df.columns:
    t_map_df = df.dropna(subset=['technique_id', 'technique_name'])[['technique_id', 'technique_name']].drop_duplicates()
    tech_map = dict(zip(t_map_df['technique_id'], t_map_df['technique_name']))

poisoning_tech_ids = [k for k, v in tech_map.items() if 'poisoning' in str(v).lower()]
evasion_tech_ids = [k for k, v in tech_map.items() if 'evasion' in str(v).lower()]

# 2. Prepare Analysis Data
target_table = 'step3_incident_coding'
df_coding = df[df['source_table'] == target_table].copy()

# Gap Counting Function
def count_gaps(gap_str):
    if pd.isna(gap_str) or gap_str == '':
        return 0
    cleaned = str(gap_str).replace('[', '').replace(']', '').replace("'", "").replace('"', '')
    if not cleaned.strip():
        return 0
    return len([x.strip() for x in cleaned.split(',') if x.strip()])

df_coding['gap_count'] = df_coding['competency_gaps'].apply(count_gaps)

# Categorization Function
def categorize_case(row):
    # Get lists of IDs
    tactics = str(row.get('tactics_used', '')).split(';')
    techniques = str(row.get('techniques_used', '')).split(';')
    
    tactics = [t.strip() for t in tactics]
    techniques = [t.strip() for t in techniques]
    
    is_evasion = False
    is_poisoning = False
    
    # Check Tactics (using the map we built)
    for tid in tactics:
        name = id_to_name.get(tid, '').lower()
        if 'evasion' in name: is_evasion = True
        if 'poisoning' in name: is_poisoning = True
        # Also check hardcoded known IDs just in case map is incomplete
        if tid == 'AML.TA0006': is_evasion = True # Defense Evasion

    # Check Techniques (using the map)
    for teid in techniques:
        name = tech_map.get(teid, '').lower()
        if 'evasion' in name: is_evasion = True
        if 'poisoning' in name: is_poisoning = True
        # Check known technique IDs if map failed
        if teid == 'AML.T0015': is_evasion = True # Evasion
        if teid == 'AML.T0020': is_poisoning = True # Data Poisoning
        if teid == 'AML.T0021': is_poisoning = True # Model Poisoning

    if is_evasion and not is_poisoning:
        return 'Evasion'
    elif is_poisoning and not is_evasion:
        return 'Poisoning'
    elif is_evasion and is_poisoning:
        return 'Mixed'
    else:
        return 'Other'

df_coding['category'] = df_coding.apply(categorize_case, axis=1)

print("\n--- Analysis Categories ---")
print(df_coding['category'].value_counts())

# 3. Statistical Analysis
evasion_scores = df_coding[df_coding['category'] == 'Evasion']['gap_count']
poisoning_scores = df_coding[df_coding['category'] == 'Poisoning']['gap_count']

print(f"\nEvasion (n={len(evasion_scores)}): Mean={evasion_scores.mean():.2f}")
print(f"Poisoning (n={len(poisoning_scores)}): Mean={poisoning_scores.mean():.2f}")

if len(evasion_scores) > 1 and len(poisoning_scores) > 1:
    t_stat, p_val = ttest_ind(evasion_scores, poisoning_scores, equal_var=False)
    print(f"\nWelch's T-Test: t={t_stat:.4f}, p-value={p_val:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.boxplot([evasion_scores, poisoning_scores], tick_labels=['Evasion', 'Poisoning'])
    plt.title('Competency Gaps: Evasion vs. Poisoning')
    plt.ylabel('Number of Missing Controls')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()
else:
    print("\nInsufficient data for statistical comparison.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Found Tactic Mappings:
  Poisoning IDs: []
  Evasion IDs: ['AML.TA0007']

--- Analysis Categories ---
category
Evasion      30
Other        14
Mixed         6
Poisoning     2
Name: count, dtype: int64

Evasion (n=30): Mean=0.97
Poisoning (n=2): Mean=1.00

Welch's T-Test: t=-1.0000, p-value=0.3256

STDERR:
/usr/local/lib/python3.13/site-packages/scipy/stats/_axis_nan_policy.py:592: RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.
  res = hypotest_fun_out(*samples, **kwds)


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (also known as a Box-and-Whisker Plot).
*   **Purpose:** This plot compares the distribution of the "Number of Missing Controls" between two different categories: "Evasion" and "Poisoning". It is designed to visualize the central tendency (median), spread, and potential outliers in the data for each group.

### 2. Axes
*   **X-Axis:**
    *   **Labels:** The axis represents categorical data with two groups: **"Evasion"** and **"Poisoning"**.
*   **Y-Axis:**
    *   **Title:** **"Number of Missing Controls"**.
    *   **Range:** The visual scale ranges from **0.0 to 1.0**, with tick marks at intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).
    *   **Units:** While not explicitly specified as a percentage or raw count, the discrete values (0 and 1) suggest this is likely a count of controls, or a binary indicator (0 = present, 1 = missing).

### 3. Data Trends
*   **Collapsed Boxes:** For both "Evasion" and "Poisoning", the "box" portion of the plot is collapsed into a single orange line at the value **1.0**. This indicates that the Median, 25th percentile (Q1), and 75th percentile (Q3) are all identical at 1.0.
*   **Poisoning:** The data for the "Poisoning" category appears to be constant. There is an orange line at **1.0**, with no visible whiskers or outliers. This suggests that every data point for "Poisoning" in this dataset has a value of 1.
*   **Evasion:** Similar to Poisoning, the median and interquartile range are compressed at **1.0**. However, there is a distinct **outlier** represented by a hollow circle at **0.0**.

### 4. Annotations and Legends
*   **Title:** The chart is titled **"Competency Gaps: Evasion vs. Poisoning"**, indicating the context is likely cybersecurity or machine learning security (Adversarial ML), analyzing where defenses or competencies are lacking.
*   **Grid Lines:** Horizontal dashed grid lines are present at every 0.2 interval on the y-axis to aid in reading the values.
*   **Colors:** The central tendency markers (medians) are colored **orange**, while the outlier is a black circle.

### 5. Statistical Insights
*   **High Uniformity:** The data is extremely uniform. For both attack types (Evasion and Poisoning), the "Number of Missing Controls" is almost universally **1**. This suggests a consistent gap in competency or defense where exactly one control is missing across the board.
*   **The Evasion Exception:** The key statistical difference is the outlier in the "Evasion" category at **0.0**. This indicates that there was at least one instance (or a small number of instances) regarding Evasion where there were **zero** missing controls (i.e., the controls were fully present/competent).
*   **Comparison:** While both categories show a significant gap (consistently sitting at 1.0), the "Poisoning" category is strictly worse in terms of consistency, as it has no recorded instances of 0 missing controls, whereas "Evasion" has at least one successful case of 0 missing controls.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
