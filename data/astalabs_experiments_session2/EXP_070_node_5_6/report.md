# Experiment 70: node_5_6

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_6` |
| **ID in Run** | 70 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T04:19:46.472434+00:00 |
| **Runtime** | 346.1s |
| **Parent** | `node_4_18` |
| **Children** | `node_6_34` |
| **Creation Index** | 71 |

---

## Hypothesis

> Documentation as a Trailing Indicator: Projects in the 'Operation' stage are
significantly more likely to possess 'Data Documentation' than projects in
'Development', indicating that documentation is primarily a post-hoc compliance
activity.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.8956 (Likely True) |
| **Surprise** | +0.1844 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
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
| Definitely True | 54.0 |
| Maybe True | 6.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Analyze the timing of documentation practices relative to the development lifecycle.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` (EO13960 subset).
- 2. Group `16_dev_stage` into 'Development' and 'Operation'.
- 3. Convert `34_data_docs` to binary.
- 4. Perform a T-test or Chi-Square test to compare documentation rates between stages.

### Deliverables
- Comparison plot of Data Documentation rates by Development Stage; Statistical test summary.

---

## Analysis

The experiment successfully tested the hypothesis that data documentation acts
as a trailing indicator in the AI lifecycle. By categorizing 1,757 projects from
the EO 13960 dataset into 'Development' and 'Operation' stages, the analysis
revealed a statistically significant difference in documentation practices.

Key findings include:
1.  **Higher Compliance in Operation**: Projects in the 'Operation' stage
demonstrated a 65.0% documentation rate, compared to 59.5% for projects in
'Development'.
2.  **Statistical Significance**: The Chi-Square test produced a p-value of
0.0208, confirming that this 5.5% difference is statistically significant (p <
0.05).
3.  **Support for Hypothesis**: The results support the hypothesis that
documentation is often treated as a late-stage compliance activity rather than a
prerequisite for development.

However, the data also highlights a broader issue: even among operational
systems, 35% lack data documentation, indicating a persistent governance gap.

---

## Review

The experiment was executed successfully and faithfully followed the plan. The
programmer effectively resolved previous data type errors by implementing robust
text-parsing functions, ensuring accurate categorization of the 'Development'
and 'Operation' stages.

The results provide statistically significant support for the hypothesis that
data documentation acts as a 'trailing indicator' in the AI lifecycle. Key
findings include:
1.  **Prevalence**: Projects in the 'Operation' stage have a documentation rate
of 65.0%, compared to 59.5% for projects in 'Development'.
2.  **Significance**: The Chi-Square test yielded a p-value of 0.0208,
confirming the difference is statistically significant (p < 0.05).
3.  **Implication**: While the higher rate in operational systems supports the
idea that documentation is often prioritized later in the lifecycle, the data
reveals a critical insight: over one-third (35%) of operational AI systems in
the federal inventory still lack basic data documentation, highlighting a
persistent gap in transparency and governance.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    # Reading with low_memory=False to avoid mixed type warnings
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if file is in current directory
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Robust Processing Functions ---

def categorize_stage(val):
    # Robustly convert to string and lower case to handle NaNs (floats)
    s = str(val).lower()
    if 'operation' in s or 'maintenance' in s:
        return 'Operation'
    if 'retired' in s:
        # Retired systems are post-operational, grouping with Operation for 'Legacy' context
        return 'Operation'
    return 'Development'

def parse_docs(val):
    # Robustly convert to string and normalize
    s = str(val).lower().strip()
    # Define values that map to False/0 (missing documentation)
    if s in ['nan', 'no', 'none', 'n/a', '0', 'false', '']:
        return 0
    return 1

# Apply categorization directly to source columns
eo_data['stage_group'] = eo_data['16_dev_stage'].apply(categorize_stage)
eo_data['has_docs'] = eo_data['34_data_docs'].apply(parse_docs)

# --- Analysis ---

# Calculate rates
summary = eo_data.groupby('stage_group')['has_docs'].agg(['count', 'sum', 'mean'])
summary.rename(columns={'count': 'Total', 'sum': 'With Docs', 'mean': 'Rate'}, inplace=True)

print("--- Summary Statistics: Data Documentation by Stage ---")
print(summary)
print("\n")

# Contingency Table
contingency_table = pd.crosstab(eo_data['stage_group'], eo_data['has_docs'])
print("--- Contingency Table ---")
print(contingency_table)
print("\n")

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"--- Chi-Square Test Results ---")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Interpretation
alpha = 0.05
if p < alpha:
    print("Result: Statistically Significant Difference found.")
    if summary.loc['Operation', 'Rate'] > summary.loc['Development', 'Rate']:
        print("Hypothesis Supported: Operation stage has significantly higher documentation rates.")
    else:
        print("Hypothesis Contradicted: Operation stage does not have higher documentation rates.")
else:
    print("Result: No Statistically Significant Difference found.")

# --- Visualization ---
plt.figure(figsize=(8, 6))

# Define colors
bar_colors = ['#1f77b4' if idx == 'Development' else '#ff7f0e' for idx in summary.index]

bars = plt.bar(summary.index, summary['Rate'], color=bar_colors, edgecolor='black', alpha=0.8)

plt.title('Data Documentation Availability by Development Stage')
plt.ylabel('Proportion of Projects with Documentation')
plt.xlabel('Project Stage')
plt.ylim(0, 1.15)

# Add labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Summary Statistics: Data Documentation by Stage ---
             Total  With Docs      Rate
stage_group                            
Development    997        593  0.594784
Operation      760        494  0.650000


--- Contingency Table ---
has_docs       0    1
stage_group          
Development  404  593
Operation    266  494


--- Chi-Square Test Results ---
Chi2 Statistic: 5.3416
P-value: 2.0823e-02
Result: Statistically Significant Difference found.
Hypothesis Supported: Operation stage has significantly higher documentation rates.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot is designed to compare the prevalence of data documentation across two distinct phases of a project's lifecycle: "Development" and "Operation."

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Label:** "Project Stage"
    *   **Categories:** The axis displays two categorical variables: "Development" and "Operation."
*   **Y-Axis (Vertical):**
    *   **Label:** "Proportion of Projects with Documentation"
    *   **Range:** The scale ranges from **0.0 to approximately 1.15**, with grid lines marking intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).
    *   **Units:** The axis represents a decimal proportion, corresponding to percentages (0% to 100%).

### 3. Data Trends
*   **Comparison:** The bar representing the **Operation** stage is taller than the bar representing the **Development** stage.
*   **Tallest Bar:** The "Operation" category has the highest value at a proportion of **0.650**.
*   **Shortest Bar:** The "Development" category has the lowest value at a proportion of **0.595**.
*   **Pattern:** There is an upward trend in documentation availability as projects move from the development phase to the operation phase.

### 4. Annotations and Legends
*   **Title:** The chart is titled "**Data Documentation Availability by Development Stage**."
*   **Data Labels:** Specific percentage values are annotated in bold directly above each bar to provide precise readings:
    *   Above Development: **59.5%**
    *   Above Operation: **65.0%**
*   **Grid Lines:** Horizontal dashed grey lines are included to assist in estimating the bar heights against the Y-axis.
*   **Color Coding:**
    *   Development is represented by a **blue** bar.
    *   Operation is represented by an **orange** bar.

### 5. Statistical Insights
*   **Documentation Gap:** While documentation becomes more common as projects mature into the operation phase, it is not universal. Even in the Operation stage, **35%** of projects still lack data documentation.
*   **Lifecycle Improvement:** There is a **5.5 percentage point increase** (65.0% - 59.5%) in documentation availability between the Development and Operation stages. This suggests that teams may prioritize creating or finalizing documentation as a project prepares for or enters the maintenance/live phase.
*   **Overall Availability:** In both stages, more than half of the projects possess data documentation, but neither stage reaches a high saturation point (e.g., above 80%), indicating a potential area for improvement in standard engineering practices.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
