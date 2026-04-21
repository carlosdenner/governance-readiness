# Experiment 34: node_4_8

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_8` |
| **ID in Run** | 34 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:34:32.971239+00:00 |
| **Runtime** | 151.0s |
| **Parent** | `node_3_14` |
| **Children** | None |
| **Creation Index** | 35 |

---

## Hypothesis

> The mapping of governance requirements to architecture controls follows a
Pareto-like distribution, where a small minority of 'Keystone' controls accounts
for a disproportionately large share of regulatory mappings.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8871 (Likely True) |
| **Posterior** | 0.9711 (Definitely True) |
| **Surprise** | +0.0975 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 18.0 |
| Maybe True | 12.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 90.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Assess the distribution of control applicability to identify potential bottlenecks or central hubs.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Identify the 18 architecture control columns (excluding metadata columns).
- 3. Calculate the sum of mappings (non-empty cells) for each control column.
- 4. Sort the controls by frequency in descending order.
- 5. Calculate the cumulative percentage of mappings covered by the top 20% of controls.
- 6. Perform a Chi-Square Goodness of Fit test comparing the observed distribution of mappings against a Uniform distribution (null hypothesis: all controls are equally applicable).

### Deliverables
- Bar chart of control frequencies, Cumulative percentage of top 20%, and Chi-Square test results.

---

## Analysis

The experiment successfully tested the hypothesis that architecture control
mappings follow a Pareto-like distribution. Analysis of
'step2_crosswalk_matrix.csv' revealed a total of 77 mappings across 18
architecture controls. The distribution is highly non-uniform (Chi-Square
p=1.38e-06), supporting the existence of 'Keystone' controls. The top 20% of
controls (n=4) account for 54.55% of all regulatory mappings. Specifically, 'AI
Risk Policy & Accountability Structures' (n=13), 'Evaluation & Monitoring
Infrastructure' (n=13), and 'GenAIOps / MLOps Lifecycle Governance' (n=10) serve
as the primary hubs, collectively representing 47% of the total connectivity. In
contrast, tactical controls like 'Audit Logging' and 'Orchestration Patterns'
appear centrally less frequently (n=1). A Gini coefficient of 0.438 further
quantifies this inequality, confirming that a small subset of governance and
infrastructure controls bear the majority of the regulatory compliance burden.

---

## Review

The experiment successfully tested the hypothesis that architecture control
mappings follow a Pareto-like distribution. Analysis of
'step2_crosswalk_matrix.csv' revealed a total of 77 mappings across 18
architecture controls. The distribution is highly non-uniform (Chi-Square
p=1.38e-06), supporting the existence of 'Keystone' controls. The top 20% of
controls (n=4) account for 54.55% of all regulatory mappings. Specifically, 'AI
Risk Policy & Accountability Structures' (n=13), 'Evaluation & Monitoring
Infrastructure' (n=13), and 'GenAIOps / MLOps Lifecycle Governance' (n=10) serve
as the primary hubs, collectively representing 47% of the total connectivity. In
contrast, tactical controls like 'Audit Logging' and 'Orchestration Patterns'
appear centrally less frequently (n=1). A Gini coefficient of 0.438 further
quantifies this inequality, confirming that a small subset of governance and
infrastructure controls bear the majority of the regulatory compliance burden.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# [debug] Listing files to confirm location
# import os
# print(os.listdir('../'))

# 1. Load the dataset
file_path = '../step2_crosswalk_matrix.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}")
except FileNotFoundError:
    # Fallback if running in same directory
    df = pd.read_csv('step2_crosswalk_matrix.csv')
    print("Successfully loaded step2_crosswalk_matrix.csv (local)")

# 2. Identify architecture control columns
# Based on metadata, first 6 columns are metadata: 
# req_id, source, function, requirement, bundle, competency_statement
# The rest are architecture controls.
metadata_cols = ['req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement']
control_cols = [c for c in df.columns if c not in metadata_cols]

print(f"Identified {len(control_cols)} control columns.")

# 3. Calculate sum of mappings for each control
# The matrix contains 'X' or similar for mappings, or NaN/empty for no mapping.
# We count non-null and non-empty string values.
control_counts = {}

for col in control_cols:
    # Count non-NA and non-empty strings
    count = df[col].notna().sum()
    # If the format uses empty strings instead of NaN, handle that:
    # But read_csv usually handles empty as NaN. Let's verify by checking specific values if needed.
    # Assuming standard CSV 'X' or NaN.
    control_counts[col] = count

# Convert to Series for easier handling
counts_series = pd.Series(control_counts).sort_values(ascending=False)

# 4. Sort and analyze distribution
total_mappings = counts_series.sum()
print(f"\nTotal control mappings observed: {total_mappings}")

# Top 20% of controls (18 controls * 0.2 = 3.6 -> Top 4 controls)
top_n = int(np.ceil(len(control_cols) * 0.2))
top_n_controls = counts_series.head(top_n)
top_n_sum = top_n_controls.sum()
top_n_perc = (top_n_sum / total_mappings) * 100

print(f"\nTop {top_n} controls (Top 20%):")
print(top_n_controls)
print(f"Cumulative coverage of Top 20% controls: {top_n_perc:.2f}%")

# 5. Chi-Square Goodness of Fit Test
# Null Hypothesis: Mappings are uniformly distributed across all 18 controls.
# Expected frequency for each control = Total Mappings / 18
expected_freq = total_mappings / len(control_cols)
expected_counts = [expected_freq] * len(control_cols)
observed_counts = counts_series.values

chi2_stat, p_val = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

print(f"\nChi-Square Goodness of Fit Test:")
print(f"Chi2 Statistic: {chi2_stat:.4f}")
print(f"P-value: {p_val:.4e}")

if p_val < 0.05:
    print("Result: Significant deviation from uniform distribution (Reject H0).")
else:
    print("Result: No significant deviation from uniform distribution (Fail to reject H0).")

# 6. Visualization
plt.figure(figsize=(12, 8))
bars = plt.barh(counts_series.index, counts_series.values, color='skyblue')
plt.xlabel('Number of Mappings')
plt.title('Frequency of Architecture Control Mappings (Pareto Check)')
plt.gca().invert_yaxis()  # Highest frequency at top

# Add value labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
             f'{int(width)}', va='center')

plt.tight_layout()
plt.show()

# Print Gini Coefficient for inequality measure (optional but useful for Pareto)
def gini(array):
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

gini_score = gini(observed_counts)
print(f"\nGini Coefficient of Control Applicability: {gini_score:.4f}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Successfully loaded step2_crosswalk_matrix.csv (local)
Identified 18 control columns.

Total control mappings observed: 77

Top 4 controls (Top 20%):
AI Risk Policy & Accountability Structures     13
Evaluation & Monitoring Infrastructure         13
GenAIOps / MLOps Lifecycle Governance          10
Nondeterminism Controls & Output Validation     6
dtype: int64
Cumulative coverage of Top 20% controls: 54.55%

Chi-Square Goodness of Fit Test:
Chi2 Statistic: 59.2857
P-value: 1.3779e-06
Result: Significant deviation from uniform distribution (Reject H0).

Gini Coefficient of Control Applicability: 0.4380


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Horizontal Bar Plot.
*   **Purpose:** The plot visualizes the frequency distribution of various "Architecture Control Mappings." By sorting the bars in descending order, it acts as a **Pareto analysis**, aiming to identify the "vital few" controls that appear most frequently compared to the "trivial many."

### 2. Axes
*   **Y-Axis (Vertical):**
    *   **Label:** Represents specific categories of Architecture Controls (e.g., "AI Risk Policy...", "Data Governance...").
    *   **Range:** Contains 18 distinct categories.
*   **X-Axis (Horizontal):**
    *   **Title:** "Number of Mappings".
    *   **Units:** Integer counts (Frequency).
    *   **Range:** The axis ticks range from 0 to 12, though the data extends slightly beyond to a maximum value of 13.

### 3. Data Trends
*   **Sorting:** The data is sorted in descending order, creating a funnel shape from top to bottom.
*   **Top Contributors (High Frequency):**
    *   The most frequent mappings are **"AI Risk Policy & Accountability Structures"** and **"Evaluation & Monitoring Infrastructure,"** both with a count of **13**.
    *   **"GenAIOps / MLOps Lifecycle Governance"** follows closely with a count of **10**.
*   **Mid-Range:**
    *   There is a significant drop-off after the top three. The next cluster ranges from **6 to 4** mappings (e.g., Nondeterminism Controls, Data Governance).
*   **Low Frequency (The "Long Tail"):**
    *   The bottom of the chart shows a plateau of items with very low frequency.
    *   Four categories share the lowest count of **1**: "Multi-Agent Orchestration Pattern," "Single-Agent Orchestration Pattern," "Scalable Modular Architecture (Archetypes)," and "Audit Logging & Telemetry."

### 4. Annotations and Legends
*   **Title:** "Frequency of Architecture Control Mappings (Pareto Check)" clearly defines the subject and the analytical intent (checking for Pareto distribution).
*   **Value Labels:** Each bar is annotated with its exact numerical value at the end (e.g., 13, 10, 6, etc.), allowing for precise reading without estimating against the axis grid.
*   **Color:** All bars use a uniform light blue color, indicating that no secondary variable or grouping is being differentiated; the focus is solely on the count per category.

### 5. Statistical Insights
*   **Concentration of Focus (Pareto Principle):** The distribution suggests that the architecture is heavily weighted toward high-level Governance and Policy rather than low-level technical implementation details.
    *   The top 3 categories alone ($13 + 13 + 10 = 36$) account for nearly **47%** of the total count (assuming a sum of roughly 77 mappings).
    *   The top 5 categories account for approximately **61%** of all mappings.
*   **Prioritization:** If this data represents gaps or required controls, an organization should prioritize "AI Risk Policy" and "Evaluation Infrastructure" as they constitute the bulk of the mappings.
*   **Operational vs. Strategic:** The lowest frequency items (Orchestration Patterns, Logging) are tactical/technical execution details, whereas the highest frequency items are strategic/governance focused. This implies the dataset or system being analyzed is currently more focused on establishing rules and monitoring rather than specific architectural patterns.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
