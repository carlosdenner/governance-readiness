# Experiment 42: node_3_17

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_17` |
| **ID in Run** | 42 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:50:00.989530+00:00 |
| **Runtime** | 195.9s |
| **Parent** | `node_2_1` |
| **Children** | `node_4_18`, `node_4_22`, `node_4_23` |
| **Creation Index** | 43 |

---

## Hypothesis

> Governance requirements classified as 'Trust Readiness' map to a significantly
higher number of architecture controls than 'Integration Readiness'
requirements, indicating that governance interventions are structurally broader
and more cross-cutting.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.1901 (Likely False) |
| **Surprise** | -0.6405 |
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
| Definitely False | 90.0 |

---

## Experiment Plan

**Objective:** Compare the architectural breadth (control density) of Trust vs. Integration requirements.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Identify architecture control columns (columns 6 to the end).
- 3. Create a new variable 'control_count' by summing the non-empty/marked cells in these columns for each row.
- 4. Group the data by 'bundle' ('Trust Readiness' vs 'Integration Readiness').
- 5. Perform an independent samples t-test to compare the mean 'control_count' between the two bundles.

### Deliverables
- Group means for control counts, t-test statistic, and p-value.

---

## Analysis

The experiment successfully loaded 'step2_crosswalk_matrix.csv' and quantified
the architectural breadth of governance requirements by summing the mapped
controls for each row.

**Key Findings:**
- **Descriptive Statistics:** Contrary to the hypothesis, 'Integration
Readiness' requirements mapped to a higher average number of architecture
controls (Mean = 2.00, SD = 0.52) compared to 'Trust Readiness' requirements
(Mean = 1.63, SD = 0.76).
- **Statistical Significance:** The independent samples t-test yielded a
t-statistic of -1.7907 and a p-value of 0.0831. Since the p-value is greater
than 0.05, the difference in control density between the two bundles is not
statistically significant.
- **Distribution:** The boxplot analysis reveals that 'Integration Readiness' is
highly consistent (median = 2.0 with low variance), whereas 'Trust Readiness' is
more variable at the lower end (median = 1.0).

**Conclusion:** The hypothesis that 'Trust Readiness' requirements map to a
significantly higher number of controls is **rejected**. The data suggests the
opposite trend (Integration > Trust), but the difference does not reach
statistical significance. This implies that governance interventions in this
framework are not structurally 'broader' than engineering requirements; instead,
engineering (Integration) requirements appear to have a slightly more consistent
architectural footprint.

---

## Review

The experiment was faithfully implemented according to the plan. The code
successfully loaded 'step2_crosswalk_matrix.csv', calculated the control density
for each requirement, and performed the specified independent samples t-test.

**Hypothesis:** The hypothesis that 'Trust Readiness' requirements map to a
significantly higher number of architecture controls than 'Integration
Readiness' requirements was **rejected**.

**Findings:**
- **Directionality:** Contrary to the hypothesis, 'Integration Readiness'
requirements actually had a higher mean control count (Mean = 2.00, SD = 0.52)
compared to 'Trust Readiness' (Mean = 1.63, SD = 0.76).
- **Significance:** The t-test yielded a p-value of 0.0831, which is above the
standard alpha of 0.05. Therefore, the difference is not statistically
significant, though it approaches significance in the opposite direction of the
hypothesis.
- **Distribution:** The boxplot analysis highlighted that 'Integration
Readiness' requirements are highly consistent (mostly mapping to exactly 2
controls), whereas 'Trust Readiness' requirements show more variability at the
lower end (often mapping to only 1 control).

**Conclusion:** Governance (Trust) interventions in this framework do not have a
broader architectural footprint than Engineering (Integration) interventions; in
fact, the data suggests engineering requirements have a slightly denser and more
consistent mapping to architecture controls.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# Define file path based on instructions
file_name = 'step2_crosswalk_matrix.csv'
# Try loading from parent directory first as per instruction, fall back to current if not found
if os.path.exists(f'../{file_name}'):
    file_path = f'../{file_name}'
else:
    file_path = file_name

print(f"Loading dataset from: {file_path}")

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File {file_name} not found in ../ or current directory.")
    exit(1)

# Validate structure
# Columns 0-5 are metadata, Columns 6+ are architecture controls
metadata_cols = df.columns[:6]
control_cols = df.columns[6:]

print(f"Metadata columns ({len(metadata_cols)}): {list(metadata_cols)}")
print(f"Control columns ({len(control_cols)}): {list(control_cols)}")

# Calculate 'control_count': sum of non-null/non-empty cells in control columns
# We check for not null and not empty string just in case
df['control_count'] = df[control_cols].apply(lambda x: x.notna() & (x != '')).sum(axis=1)

# Group by Bundle
bundles = df['bundle'].unique()
print(f"\nBundles found: {bundles}")

trust_data = df[df['bundle'] == 'Trust Readiness']['control_count']
integration_data = df[df['bundle'] == 'Integration Readiness']['control_count']

# Descriptive Statistics
print("\n--- Descriptive Statistics for Control Counts ---")
print(f"Trust Readiness (n={len(trust_data)}): Mean={trust_data.mean():.2f}, Std={trust_data.std():.2f}")
print(f"Integration Readiness (n={len(integration_data)}): Mean={integration_data.mean():.2f}, Std={integration_data.std():.2f}")

# Independent Samples T-Test
# We use Welch's t-test (equal_var=False) as sample sizes and variances might differ
t_stat, p_val = stats.ttest_ind(trust_data, integration_data, equal_var=False)

print("\n--- T-Test Results ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("Result: Statistically significant difference.")
else:
    print("Result: No statistically significant difference.")

# Visualization
plt.figure(figsize=(10, 6))
data_to_plot = [trust_data, integration_data]
plt.boxplot(data_to_plot, labels=['Trust Readiness', 'Integration Readiness'], patch_artist=True)
plt.title('Distribution of Architecture Control Counts per Requirement')
plt.ylabel('Number of Mapped Controls')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step2_crosswalk_matrix.csv
Metadata columns (6): ['req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement']
Control columns (18): ['Single-Agent Orchestration Pattern', 'Multi-Agent Orchestration Pattern', 'Tool-Use Boundaries & Least-Privilege Access', 'Human-in-the-Loop Approval Gates', 'Nondeterminism Controls & Output Validation', 'RAG Architecture & Data Grounding', 'GenAIOps / MLOps Lifecycle Governance', 'Evaluation & Monitoring Infrastructure', 'Prompt Management & Secret Handling', 'Scalable Modular Architecture (Archetypes)', 'AI Risk Policy & Accountability Structures', 'Threat Modeling & Red-Teaming', 'Incident Response & Recovery Playbooks', 'Audit Logging & Telemetry', 'Regulatory Compliance Documentation', 'Supply Chain & Vendor Risk Controls', 'Data Governance & Access Controls', 'Human Override & Control Mechanisms']

Bundles found: <StringArray>
['Trust Readiness', 'Integration Readiness']
Length: 2, dtype: str

--- Descriptive Statistics for Control Counts ---
Trust Readiness (n=19): Mean=1.63, Std=0.76
Integration Readiness (n=23): Mean=2.00, Std=0.52

--- T-Test Results ---
T-statistic: -1.7907
P-value: 0.0831
Result: No statistically significant difference.

STDERR:
<ipython-input-1-16217de67cc9>:67: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data_to_plot, labels=['Trust Readiness', 'Integration Readiness'], patch_artist=True)


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Box Plot (also known as a Box-and-Whisker Plot).
*   **Purpose:** This plot visualizes the distribution, central tendency, and variability of numerical data ("Number of Mapped Controls") across distinct categorical groups ("Trust Readiness" and "Integration Readiness"). It is designed to show the median, quartiles, and outliers.

### 2. Axes
*   **Y-Axis:**
    *   **Title:** "Number of Mapped Controls".
    *   **Range:** The visual axis spans from approximately 0.9 to 3.1, with tick marks at intervals of 0.25. The actual data points represented are discrete integers (1.0, 2.0, 3.0).
*   **X-Axis:**
    *   **Title:** There is no explicit label for the axis itself, but it represents "Requirement Categories."
    *   **Labels:** Two distinct categories: "Trust Readiness" and "Integration Readiness".

### 3. Data Trends
*   **Trust Readiness (Left):**
    *   **Distribution:** The distribution is skewed toward the lower end. The median line (orange) sits at the bottom of the box (value 1.0), indicating that at least 50% of the requirements in this category have only 1 mapped control.
    *   **Spread:** The Interquartile Range (IQR)—represented by the blue box—spans from 1.0 to 2.0.
    *   **Range:** The upper whisker extends to 3.0, showing that while most values are 1 or 2, some requirements have up to 3 controls.
*   **Integration Readiness (Right):**
    *   **Concentration:** The "box" has collapsed into a single orange line at the value 2.0. This indicates that the 25th percentile, Median, and 75th percentile are all equal to 2.0. The vast majority of data points in this category are exactly 2.
    *   **Outliers:** There are distinct circular markers at 1.0 and 3.0. These represent outliers, indicating that while most requirements have 2 controls, there are rare exceptions with 1 or 3 controls.

### 4. Annotations and Legends
*   **Title:** "Distribution of Architecture Control Counts per Requirement" clearly defines the scope of the analysis.
*   **Grid Lines:** Horizontal dashed grey lines are provided at intervals of 0.25 to assist in reading the Y-axis values.
*   **Color Coding:**
    *   **Blue Box:** Represents the Interquartile Range (IQR) representing the middle 50% of the data.
    *   **Orange Line:** Represents the median value.
    *   **Circles:** Represent outlier data points.

### 5. Statistical Insights
*   **Consistency vs. Variability:** "Integration Readiness" is highly consistent; nearly all requirements in this category map to exactly 2 architecture controls. "Trust Readiness" shows more variability in the lower range, fluctuating commonly between 1 and 2 controls.
*   **Central Tendency:** The median count for "Integration Readiness" (2.0) is higher than that of "Trust Readiness" (1.0). This suggests that, on average, Integration Readiness requirements tend to demand a slightly higher number of architecture controls than Trust Readiness requirements.
*   **Maximum Complexity:** Both categories have a maximum mapped control count of 3, though reaching this count is within the normal range for "Trust Readiness" but is considered a statistical outlier for "Integration Readiness."
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
