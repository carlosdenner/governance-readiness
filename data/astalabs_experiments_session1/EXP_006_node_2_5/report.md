# Experiment 6: node_2_5

| Property | Value |
|---|---|
| **Experiment ID** | `node_2_5` |
| **ID in Run** | 6 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T21:38:01.381337+00:00 |
| **Runtime** | 181.2s |
| **Parent** | `node_1_0` |
| **Children** | `node_3_0`, `node_3_4` |
| **Creation Index** | 7 |

---

## Hypothesis

> Competency statements classified as 'Trust Readiness' require a significantly
higher number of distinct architecture controls to satisfy than 'Integration
Readiness' competencies, reflecting the diffuse and multi-faceted nature of
governance implementation.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7417 (Likely True) |
| **Posterior** | 0.1854 (Likely False) |
| **Surprise** | -0.6456 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 29.0 |
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

**Objective:** Analyze the 'control density' of governance requirements vs. engineering requirements.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Identify the 18 architecture control columns (columns 7 to 24).
- 3. Calculate a new variable 'control_count' for each row by summing the binary values ('X' converted to 1) in the control columns.
- 4. Group the data by the 'bundle' column.
- 5. Perform a t-test to compare the mean 'control_count' between 'Trust Readiness' and 'Integration Readiness' requirements.

### Deliverables
- Descriptive statistics of control counts by bundle, t-test results, and a histogram overlay of control counts.

---

## Analysis

The experiment successfully analyzed the 'control density' of governance
requirements by processing `step2_crosswalk_matrix.csv`. The results contradict
the initial hypothesis that 'Trust Readiness' competencies require a higher
number of architecture controls.

Descriptive statistics reveal that 'Integration Readiness' competencies actually
map to a higher average number of controls (Mean = 2.00, n=23) compared to
'Trust Readiness' competencies (Mean = 1.63, n=19). The visualization highlights
distinct structural patterns: Integration requirements strongly cluster around 2
controls (mode=2), whereas Trust requirements most frequently map to a single
control (mode=1).

Statistical testing (Welch's t-test) yielded a t-statistic of -1.7907 and a
p-value of 0.0831. While this indicates a trend where Integration is more
architecturally dense than Trust, the result is not statistically significant at
the standard alpha level of 0.05. Therefore, the hypothesis is rejected on two
counts: the direction of the difference is opposite to the prediction, and the
difference itself is marginally non-significant.

---

## Review

The experiment was faithfully implemented according to the plan. The code
correctly loaded the `step2_crosswalk_matrix.csv` dataset, calculated the
'control density' (number of architecture controls per requirement), and
performed the required statistical comparison (Welch's t-test) and
visualization.

**Hypothesis status:** Rejected.

**Key Findings:**
1.  **Directionality:** The results contradicted the hypothesis. Integration
Readiness competencies had a higher mean control count (2.00) compared to Trust
Readiness competencies (1.63), whereas the hypothesis predicted Trust would be
higher.
2.  **Significance:** The difference was not statistically significant (p =
0.0831 > 0.05).
3.  **Distribution:** The visualization revealed distinct structural patterns:
'Integration' requirements are highly standardized (mostly mapping to exactly 2
controls), while 'Trust' requirements are often satisfied by a single control
(mode=1). This suggests that governance requirements are often more specific or
siloed in their architectural implementation compared to integration
requirements.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
# Loading from one level above as per instructions
try:
    df = pd.read_csv('../step2_crosswalk_matrix.csv')
except FileNotFoundError:
    df = pd.read_csv('step2_crosswalk_matrix.csv')

# Identify control columns (excluding metadata)
# Metadata columns are the first 6 columns
metadata_cols = ['req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement']
control_cols = [c for c in df.columns if c not in metadata_cols]

# Calculate control count per row (robust against case/whitespace)
# We count cells containing 'X'
def count_controls(row):
    count = 0
    for val in row:
        if isinstance(val, str) and val.strip().upper() == 'X':
            count += 1
    return count

df['control_count'] = df[control_cols].apply(count_controls, axis=1)

# Group by bundle
trust_scores = df[df['bundle'] == 'Trust Readiness']['control_count']
integration_scores = df[df['bundle'] == 'Integration Readiness']['control_count']

# Statistics
print("=== Descriptive Statistics: Control Density by Bundle ===")
desc = df.groupby('bundle')['control_count'].describe()
print(desc)

# T-Test (Welch's for unequal variances)
t_stat, p_val = stats.ttest_ind(trust_scores, integration_scores, equal_var=False)

print("\n=== T-Test Results ===")
print(f"Trust Readiness Mean: {trust_scores.mean():.2f} (n={len(trust_scores)})")
print(f"Integration Readiness Mean: {integration_scores.mean():.2f} (n={len(integration_scores)})")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

if p_val < 0.05:
    print("Result: Significant difference detected (p < 0.05).")
else:
    print("Result: No significant difference detected (p >= 0.05).")

# Histogram
plt.figure(figsize=(10, 6))
# Calculate common bins for clean alignment
min_val = df['control_count'].min()
max_val = df['control_count'].max()
bins = np.arange(min_val - 0.5, max_val + 1.5, 1)

plt.hist(trust_scores, bins=bins, alpha=0.6, label='Trust Readiness', density=True, color='skyblue', edgecolor='black')
plt.hist(integration_scores, bins=bins, alpha=0.6, label='Integration Readiness', density=True, color='orange', edgecolor='black')

plt.title('Distribution of Architecture Control Density by Readiness Bundle')
plt.xlabel('Number of Controls Mapped')
plt.ylabel('Density')
plt.legend()
plt.xticks(np.arange(min_val, max_val + 1, 1))
plt.grid(axis='y', alpha=0.3)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: === Descriptive Statistics: Control Density by Bundle ===
                       count      mean       std  min  25%  50%  75%  max
bundle                                                                   
Integration Readiness   23.0  2.000000  0.522233  1.0  2.0  2.0  2.0  3.0
Trust Readiness         19.0  1.631579  0.760886  1.0  1.0  1.0  2.0  3.0

=== T-Test Results ===
Trust Readiness Mean: 1.63 (n=19)
Integration Readiness Mean: 2.00 (n=23)
T-statistic: -1.7907
P-value: 0.0831
Result: No significant difference detected (p >= 0.05).


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Overlaid Density Histogram (or Bar Chart visualizing density distributions).
*   **Purpose:** The plot compares the probability distribution of the "Number of Controls Mapped" for two different categories: "Trust Readiness" and "Integration Readiness." It shows the relative frequency (density) of items having 1, 2, or 3 controls mapped within each bundle.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Number of Controls Mapped"
    *   **Range:** The axis displays discrete integer values representing counts: **1, 2, and 3**.
*   **Y-Axis:**
    *   **Label:** "Density"
    *   **Range:** The values range from **0.0 to roughly 0.75**.
    *   **Description:** This axis represents the proportion or relative frequency of the data falling into each bin. Since it is a density plot, the sum of the bar heights for each category likely sums to 1.0 (100%).

### 3. Data Trends
The plot reveals distinct distribution patterns for the two readiness bundles:

*   **Trust Readiness (Light Blue Bars):**
    *   **Trend:** This distribution is right-skewed or decreasing.
    *   **Dominant Value:** The highest density is at **1 Control Mapped** (approximately 0.53 or 53%).
    *   **Pattern:** As the number of controls increases, the density decreases. There is a moderate amount at 2 controls (~0.31) and the lowest amount at 3 controls (~0.16).
    *   **Interpretation:** Items in the "Trust Readiness" bundle are most likely to map to a single control.

*   **Integration Readiness (Orange Bars):**
    *   **Trend:** This distribution is centered and peaked (bell-shaped).
    *   **Dominant Value:** There is a very strong peak at **2 Controls Mapped** (approximately 0.74 or 74%).
    *   **Pattern:** The densities for 1 control and 3 controls are nearly symmetrical and low (both appear to be around 0.13).
    *   **Interpretation:** Items in the "Integration Readiness" bundle predominantly map to exactly two controls.

### 4. Annotations and Legends
*   **Title:** "Distribution of Architecture Control Density by Readiness Bundle" located at the top center.
*   **Legend:** Located in the top-right corner.
    *   **Light Blue Square:** Represents the "Trust Readiness" dataset.
    *   **Orange Square:** Represents the "Integration Readiness" dataset.
*   **Grid:** Horizontal grid lines appear at 0.1 intervals to assist in reading density values.

### 5. Statistical Insights
*   **Divergent Behavior:** The two bundles exhibit contrasting structural complexity. "Trust Readiness" tends to be simpler, often involving a single control mapping, whereas "Integration Readiness" is more complex but highly standardized, converging strongly on pairs of controls (2 mappings).
*   **Predictability:** "Integration Readiness" is more predictable regarding control density, as nearly three-quarters of its data falls into a single category (2 controls). "Trust Readiness" has more variance, spreading its density more significantly across 1 and 2 controls.
*   **Mode Comparison:** The mode for Trust Readiness is 1, while the mode for Integration Readiness is 2.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
