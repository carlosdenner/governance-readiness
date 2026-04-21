# Experiment 46: node_5_5

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_5` |
| **ID in Run** | 46 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:59:14.704752+00:00 |
| **Runtime** | 221.5s |
| **Parent** | `node_4_4` |
| **Children** | `node_6_5` |
| **Creation Index** | 47 |

---

## Hypothesis

> The 'Human-in-the-Loop Approval Gates' architecture control is mapped
significantly more frequently to 'Trust Readiness' requirements than to
'Integration Readiness' requirements, statistically confirming its role as a
governance-enforcing control rather than an engineering utility.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7903 (Likely True) |
| **Posterior** | 0.2025 (Likely False) |
| **Surprise** | -0.6822 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 6.0 |
| Maybe True | 24.0 |
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

**Objective:** Determine the bundle affinity of the 'Human-in-the-Loop Approval Gates' control.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Identify the column 'Human-in-the-Loop Approval Gates'.
- 3. Create a binary variable (1 if 'X', 0 otherwise) for this column.
- 4. Group by 'bundle' and calculate the mean usage rate (proportion of requirements using this control).
- 5. Perform a Chi-square test or Z-test for proportions to check if the usage rate differs significantly between Trust and Integration bundles.

### Deliverables
- Usage proportions per bundle and statistical test results.

---

## Analysis

The experiment successfully loaded 'step2_crosswalk_matrix.csv' and analyzed the
bundle affinity of the 'Human-in-the-Loop Approval Gates' control. The
hypothesis predicted that this control would be mapped significantly more
frequently to 'Trust Readiness'. The results contradict this prediction in both
direction and significance. The control was mapped to 0% of 'Trust Readiness'
requirements (0/19) and 13.04% of 'Integration Readiness' requirements (3/23).
Statistical testing (Fisher's Exact Test p=0.2387) confirmed that this
difference is not statistically significant, likely due to the low overall
frequency of the control (only 3 total occurrences). Therefore, the hypothesis
is rejected; the control appears exclusively (though sparsely) in the
Integration bundle rather than the Trust bundle.

---

## Review

The experiment was successfully executed and the analysis is methodologically
sound. The code correctly identified the usage frequency of 'Human-in-the-Loop
Approval Gates' across the two bundles. The hypothesis, which predicted a higher
affinity for 'Trust Readiness', was conclusively rejected. The data showed the
opposite trend: the control was mapped to 0% of 'Trust Readiness' requirements
(0/19) versus ~13% of 'Integration Readiness' requirements (3/23). Fisher's
Exact Test (p=0.2387) confirmed that while there is an observed difference, it
is not statistically significant due to the scarcity of the control (only 3
total occurrences). Thus, the control is not a dominant feature of either bundle
in this dataset, and certainly not a 'Trust' enforcer as hypothesized.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import sys

# Robust file loading
filename = 'step2_crosswalk_matrix.csv'
file_path = None

# Check current directory and parent directory
possible_paths = [filename, f'../{filename}']

for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    print(f"Error: {filename} not found in current or parent directory.")
    sys.exit(1)

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# Target column and bundle column
control_col = 'Human-in-the-Loop Approval Gates'
bundle_col = 'bundle'

# Verify columns exist
if control_col not in df.columns or bundle_col not in df.columns:
    print(f"Error: Required columns not found. Available columns: {df.columns.tolist()}")
    sys.exit(1)

# Preprocess: Convert control column to binary (1 if 'X', 0 otherwise)
# Fill NA with empty string, convert to string, strip whitespace, check for 'X'
df['is_mapped'] = df[control_col].fillna('').astype(str).str.strip().apply(lambda x: 1 if x.upper() == 'X' else 0)

# Group by bundle to see raw counts and proportions
summary = df.groupby(bundle_col)['is_mapped'].agg(['count', 'sum', 'mean'])
summary.rename(columns={'count': 'Total Requirements', 'sum': 'Control Mapped Count', 'mean': 'Usage Proportion'}, inplace=True)
print("\n=== Descriptive Statistics ===")
print(summary)

# Create Contingency Table for Statistical Test
# Rows: Bundle (Trust vs Integration)
# Cols: Control Mapped (0 vs 1)
contingency_table = pd.crosstab(df[bundle_col], df['is_mapped'])
print("\n=== Contingency Table ===")
print(contingency_table)

# Check if we have enough data for a valid test (at least 2x2)
if contingency_table.shape == (2, 2):
    # Perform Chi-square test of independence
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    print("\n=== Statistical Test Results (Chi-square) ===")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"p-value: {p:.4f}")

    # Fisher's Exact Test (often better for small sample sizes)
    odds_ratio, fisher_p = stats.fisher_exact(contingency_table)
    print(f"Fisher's Exact Test p-value: {fisher_p:.4f}")
    print(f"Odds Ratio: {odds_ratio:.4f}")
else:
    print("\nWarning: Contingency table is not 2x2. One bundle might have 0 mappings.")
    # If one bundle has 0 mappings, we can still do a Fisher exact test if we construct the 2x2 manually or handle it
    # But let's see the output first. 

# Visualization
plt.figure(figsize=(8, 6))
# We want to plot the proportion of 'is_mapped' == 1 for each bundle
proportions = summary['Usage Proportion']
ax = proportions.plot(kind='bar', color=['skyblue', 'salmon'], alpha=0.8)
plt.title(f"Proportion of Requirements Mapping to\n'{control_col}' by Bundle")
plt.ylabel("Proportion (Usage Rate)")
plt.xlabel("Competency Bundle")
plt.ylim(0, 1.0)

# Add value labels
for i, v in enumerate(proportions):
    ax.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step2_crosswalk_matrix.csv

=== Descriptive Statistics ===
                       Total Requirements  ...  Usage Proportion
bundle                                     ...                  
Integration Readiness                  23  ...          0.130435
Trust Readiness                        19  ...          0.000000

[2 rows x 3 columns]

=== Contingency Table ===
is_mapped               0  1
bundle                      
Integration Readiness  20  3
Trust Readiness        19  0

=== Statistical Test Results (Chi-square) ===
Chi2 Statistic: 1.0646
p-value: 0.3022
Fisher's Exact Test p-value: 0.2387
Odds Ratio: 0.0000


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot (Vertical Column Chart).
*   **Purpose:** The plot compares categorical data ("Competency Bundles") against a numerical variable ("Proportion/Usage Rate"). Specifically, it visualizes how frequently requirements map to "Human-in-the-Loop Approval Gates" for two different bundles.

### 2. Axes
*   **X-axis:**
    *   **Title:** "Competency Bundle"
    *   **Labels:** Two categorical labels are presented vertically: "Integration Readiness" and "Trust Readiness".
*   **Y-axis:**
    *   **Title:** "Proportion (Usage Rate)"
    *   **Range:** The axis ranges from **0.0 to 1.0** (representing 0% to 100%).
    *   **Intervals:** The scale is marked in increments of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
*   **Highest Value:** The "Integration Readiness" bundle represents the highest (and only non-zero) value in this comparison.
*   **Lowest Value:** The "Trust Readiness" bundle represents the lowest value at exactly zero.
*   **Pattern:** There is a significant disparity between the two categories. While Integration Readiness has a measurable association with the metric, Trust Readiness has absolutely none.

### 4. Annotations and Legends
*   **Annotations:** There are explicit numerical annotations placed above each category location to indicate exact values:
    *   Above **Integration Readiness**: "13.04%"
    *   Above **Trust Readiness**: "0.00%"
*   **Legend:** There is no separate legend box; the categories are identified directly by the x-axis labels.

### 5. Statistical Insights
*   **Exclusivity of Requirements:** The specific requirement of "Human-in-the-Loop Approval Gates" appears to be exclusive to the **Integration Readiness** bundle (13.04%) in this comparison, as the **Trust Readiness** bundle shows a 0.00% mapping rate.
*   **Low Overall Prevalence:** Even within the Integration Readiness bundle, the usage rate is relatively low (roughly 1 in 8 requirements, or ~13%). This suggests that "Human-in-the-Loop" gates are a minority requirement even where they are present.
*   **Implication for Automation:** The 0.00% score for Trust Readiness implies that "Trust" requirements in this specific framework either rely entirely on automated validation or do not utilize human approval gates as a control mechanism.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
