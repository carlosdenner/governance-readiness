# Experiment 21: node_4_4

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_4` |
| **ID in Run** | 21 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:08:58.646822+00:00 |
| **Runtime** | 213.1s |
| **Parent** | `node_3_2` |
| **Children** | `node_5_5` |
| **Creation Index** | 22 |

---

## Hypothesis

> The 'Audit Logging & Telemetry' architecture control is mapped to a
significantly higher number of governance requirements than the average of all
other controls, indicating its role as a foundational 'super-control'.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7742 (Likely True) |
| **Posterior** | 0.1983 (Likely False) |
| **Surprise** | -0.6683 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 4.0 |
| Maybe True | 26.0 |
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
| Definitely False | 90.0 |

---

## Experiment Plan

**Objective:** Quantify the centrality of 'Audit Logging & Telemetry' compared to other controls.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Sum the number of non-empty values ('X') for each of the 18 architecture control columns to get the total requirement count per control.
- 3. Isolate the count for 'Audit Logging & Telemetry'.
- 4. Calculate the mean and standard deviation of the counts for the *other* 17 controls.
- 5. Calculate a Z-score for 'Audit Logging & Telemetry' to determine if it is a statistical outlier (e.g., Z > 1.96).

### Deliverables
- Control counts, Z-score for Audit Logging, and interpretation.

---

## Analysis

The experiment successfully loaded 'step2_crosswalk_matrix.csv' and quantified
the mapping frequency for all 18 architecture controls. Contrary to the
hypothesis that 'Audit Logging & Telemetry' acts as a 'super-control', the data
revealed it is one of the least mapped controls, with only 1 associated
requirement. In contrast, the top controls ('AI Risk Policy & Accountability
Structures' and 'Evaluation & Monitoring Infrastructure') mapped to 13
requirements each. The statistical analysis yielded a Z-score of -0.8920 for
'Audit Logging & Telemetry', placing it approximately 0.9 standard deviations
*below* the mean of the other controls (4.47). Consequently, the hypothesis is
strongly rejected; 'Audit Logging & Telemetry' is not a central node in this
governance framework.

---

## Review

The experiment was successfully executed and the code faithfully implemented the
analysis plan. The dataset 'step2_crosswalk_matrix.csv' was correctly loaded,
and the frequency of governance mappings for 'Audit Logging & Telemetry' was
compared against the other 17 architecture controls. The results conclusively
disprove the hypothesis: rather than being a 'super-control', 'Audit Logging &
Telemetry' mapped to only 1 requirement, whereas the top controls mapped to 13.
With a Z-score of -0.89, it falls below the average (4.47) rather than
significantly above it. The analysis is complete and valid.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

print("=== Starting Experiment: Audit Logging & Telemetry Centrality ===")

# 1. Load Data with robust path checking
filename = 'step2_crosswalk_matrix.csv'
filepath = None

if os.path.exists(filename):
    filepath = filename
elif os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    print(f"Error: {filename} not found in current ({os.getcwd()}) or parent directory.")
    sys.exit(1)

print(f"Loading dataset from: {filepath}")
df = pd.read_csv(filepath)

# 2. Identify Control Columns
# Metadata columns as per dataset description
metadata_cols = ['req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement']
control_cols = [c for c in df.columns if c not in metadata_cols]

print(f"Number of control columns identified: {len(control_cols)}")

target_control = 'Audit Logging & Telemetry'
if target_control not in control_cols:
    print(f"Error: '{target_control}' column not found.")
    print("Available columns:", control_cols)
    sys.exit(1)

# 3. Calculate Counts
# The dataset has 'X' for mappings, empty otherwise.
control_counts = {}

for col in control_cols:
    # Fill NA with empty string, convert to string, strip whitespace, uppercase, compare to 'X'
    count = (df[col].fillna('').astype(str).str.strip().str.upper() == 'X').sum()
    control_counts[col] = count

# Convert to Series for easier handling
counts_series = pd.Series(control_counts).sort_values(ascending=False)

# 4. Isolate Target and Others
target_count = counts_series[target_control]
other_counts = counts_series.drop(target_control)

# 5. Calculate Statistics
mean_others = other_counts.mean()
std_others = other_counts.std()

# 6. Calculate Z-Score
# Z = (x - mean) / std
if std_others == 0:
    z_score = 0
    print("Warning: Standard deviation of other controls is 0.")
else:
    z_score = (target_count - mean_others) / std_others

# 7. Output Results
print("\n--- Control Frequency Analysis ---")
print(f"Target Control: '{target_control}'")
print(f"Target Count: {target_count}")
print(f"Mean of Other Controls: {mean_others:.4f}")
print(f"Std Dev of Other Controls: {std_others:.4f}")
print(f"Z-Score: {z_score:.4f}")

if abs(z_score) > 1.96:
    print("Interpretation: The control is a statistical outlier (p < 0.05).")
else:
    print("Interpretation: The control is NOT a statistical outlier.")

print("\nTop 5 Controls:")
print(counts_series.head(5))

# 8. Visualization
plt.figure(figsize=(12, 8))
colors = ['red' if x == target_control else 'skyblue' for x in counts_series.index]
counts_series.plot(kind='bar', color=colors)
plt.title(f'Requirement Mapping Frequency per Architecture Control (Z={z_score:.2f})')
plt.xlabel('Architecture Control')
plt.ylabel('Number of Mapped Requirements')
plt.axhline(y=mean_others, color='green', linestyle='--', label=f'Mean (Others): {mean_others:.1f}')
plt.legend()
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: === Starting Experiment: Audit Logging & Telemetry Centrality ===
Loading dataset from: step2_crosswalk_matrix.csv
Number of control columns identified: 18

--- Control Frequency Analysis ---
Target Control: 'Audit Logging & Telemetry'
Target Count: 1
Mean of Other Controls: 4.4706
Std Dev of Other Controls: 3.8910
Z-Score: -0.8920
Interpretation: The control is NOT a statistical outlier.

Top 5 Controls:
AI Risk Policy & Accountability Structures     13
Evaluation & Monitoring Infrastructure         13
GenAIOps / MLOps Lifecycle Governance          10
Nondeterminism Controls & Output Validation     6
Data Governance & Access Controls               5
dtype: int64


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot compares the frequency of "Mapped Requirements" across various categories of "Architecture Control." It is designed to show which controls are most heavily utilized or regulated versus those that are least involved, ranking them from highest frequency to lowest.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Architecture Control"
    *   **Content:** Categorical labels representing specific technical and governance controls (e.g., "AI Risk Policy & Accountability Structures," "GenAIOps / MLOps Lifecycle Governance").
*   **Y-Axis:**
    *   **Label:** "Number of Mapped Requirements"
    *   **Value Range:** 0 to approximately 13 (based on the height of the tallest bars).
    *   **Units:** Integer count (frequency).

### 3. Data Trends
*   **Highest Values:** The top two controls are tied for the highest frequency with **13** mapped requirements each:
    *   "AI Risk Policy & Accountability Structures"
    *   "Evaluation & Monitoring Infrastructure"
*   **Lowest Values:** The bottom four controls are tied with only **1** mapped requirement each:
    *   "Multi-Agent Orchestration Pattern"
    *   "Single-Agent Orchestration Pattern"
    *   "Scalable Modular Architecture (Archetypes)"
    *   "Audit Logging & Telemetry" (highlighted in red)
*   **Overall Pattern:** The data follows a descending distribution (Pareto-like). There is a steep drop-off after the top three categories, followed by a gradual decline, eventually plateauing at the lowest value of 1 for the final four categories.

### 4. Annotations and Legends
*   **Green Dashed Line:** Indicates the "Mean (Others): 4.5". This horizontal line represents the average number of mapped requirements for the group (likely excluding the specific item of interest or representing the population mean).
*   **Red Bar Highlight:** The bar for **"Audit Logging & Telemetry"** is colored red, drawing specific attention to it as the subject of analysis for this specific view.
*   **Title Annotation (Z=-0.89):** The title includes a Z-score. A Z-score of -0.89 indicates that the red highlighted bar ("Audit Logging & Telemetry") is approximately 0.89 standard deviations below the mean of the dataset.

### 5. Statistical Insights
*   **Below Average Performance:** The highlighted control, "Audit Logging & Telemetry," is significantly below the average (1 vs. the mean of 4.5). Being in the lowest tier suggests this area is under-represented in the current requirements mapping compared to other architectural controls.
*   **Focus on Governance:** The chart indicates a heavy skew toward high-level governance and monitoring. The top three categories (Risk Policy, Evaluation Infrastructure, and Lifecycle Governance) occupy the vast majority of the requirements, suggesting the project or system is currently prioritizing policy and oversight over specific technical implementation patterns.
*   **The "Long Tail":** There is a long tail of 12 categories that fall below the mean line of 4.5. This suggests that while a few controls are critical, the majority of architecture controls have relatively few specific requirements mapped to them.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
