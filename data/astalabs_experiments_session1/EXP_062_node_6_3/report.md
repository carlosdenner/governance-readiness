# Experiment 62: node_6_3

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_3` |
| **ID in Run** | 62 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:27:28.397302+00:00 |
| **Runtime** | 259.7s |
| **Parent** | `node_5_7` |
| **Children** | None |
| **Creation Index** | 63 |

---

## Hypothesis

> The architecture controls 'GenAIOps / MLOps Lifecycle Governance' and
'Evaluation & Monitoring Infrastructure' co-occur in governance requirements
significantly more often than expected by chance, indicating a coupled
engineering capability.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8065 (Likely True) |
| **Posterior** | 0.2128 (Likely False) |
| **Surprise** | -0.6890 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 8.0 |
| Maybe True | 22.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 3.0 |
| Definitely False | 87.0 |

---

## Experiment Plan

**Objective:** Identify structural clustering between operational governance and evaluation infrastructure controls.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Extract columns for 'GenAIOps / MLOps Lifecycle Governance' and 'Evaluation & Monitoring Infrastructure'.
- 3. Convert 'X' markers to binary (1/0).
- 4. Create a contingency table (00, 01, 10, 11) and perform a Chi-square test of independence or calculate the Phi coefficient.

### Deliverables
- Heatmap of co-occurrence, contingency table, and statistical test results.

---

## Analysis

The experiment successfully analyzed the relationship between 'GenAIOps / MLOps
Lifecycle Governance' and 'Evaluation & Monitoring Infrastructure' using the
'step2_crosswalk_matrix.csv' dataset. The analysis reveals that these two
architecture controls do not significantly co-occur. The majority of
requirements (57%, 24/42) map to neither control. Among the requirements that do
map to 'GenAIOps' (10 total), exactly half (5) also map to 'Evaluation
Infrastructure', while the other half do not. The Fisher's Exact Test yielded a
p-value of 0.2383, which is well above the significance threshold of 0.05, and
the Phi coefficient of 0.2303 indicates a weak correlation. Consequently, the
hypothesis that these controls are structurally coupled in governance
requirements is rejected; they appear to be applied independently or
conditionally rather than as a standard pair.

---

## Review

The experiment successfully tested the hypothesis that 'GenAIOps / MLOps
Lifecycle Governance' and 'Evaluation & Monitoring Infrastructure' co-occur
significantly in governance requirements. Using the `step2_crosswalk_matrix.csv`
dataset, the analysis produced a contingency table showing that the majority of
requirements (24/42) map to neither control. Of the 10 requirements mapping to
GenAIOps, only 5 also mapped to Evaluation Infrastructure. The Fisher's Exact
Test yielded a p-value of 0.2383, and the Phi coefficient was 0.2303, indicating
a weak and statistically insignificant correlation. Therefore, the hypothesis is
rejected; the data suggests these architecture controls are not structurally
coupled in the current governance framework.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
import os

# Load the dataset
filename = 'step2_crosswalk_matrix.csv'

# Fallback to check parent directory if not found locally, just in case
if not os.path.exists(filename):
    filename = '../' + filename

df = pd.read_csv(filename)

# Define the columns of interest
col_ops = 'GenAIOps / MLOps Lifecycle Governance'
col_eval = 'Evaluation & Monitoring Infrastructure'

# Preprocess: Convert 'X' to 1, others to 0
df['Ops_Binary'] = df[col_ops].apply(lambda x: 1 if str(x).strip() == 'X' else 0)
df['Eval_Binary'] = df[col_eval].apply(lambda x: 1 if str(x).strip() == 'X' else 0)

# Create Contingency Table
contingency_table = pd.crosstab(df['Ops_Binary'], df['Eval_Binary'])

# Rename index/columns for clarity (0=No, 1=Yes)
contingency_table.index = ['No Ops Gov', 'Has Ops Gov']
contingency_table.columns = ['No Eval Infra', 'Has Eval Infra']

# Calculate Statistics
# 1. Chi-square test (correction=False for raw chi2)
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table, correction=False)

# 2. Fisher's Exact Test
odds_ratio, p_value_fisher = stats.fisher_exact(contingency_table)

# 3. Phi Coefficient
phi_coeff = matthews_corrcoef(df['Ops_Binary'], df['Eval_Binary'])

print("=== Co-occurrence Analysis ===")
print(f"File Loaded: {filename}")
print(f"Dataset shape: {df.shape}")
print(f"\nContingency Table:\n{contingency_table}")
print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value (Chi-square): {p_value:.4f}")
print(f"P-value (Fisher's Exact): {p_value_fisher:.4f}")
print(f"Phi Coefficient: {phi_coeff:.4f}")

# Interpretation
alpha = 0.05
if p_value_fisher < alpha:
    print("\nResult: Statistically significant co-occurrence detected.")
else:
    print("\nResult: No statistically significant co-occurrence detected (Null hypothesis not rejected).")

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Co-occurrence of GenAIOps and Eval Infrastructure')
plt.xlabel('Evaluation & Monitoring Infrastructure')
plt.ylabel('GenAIOps / MLOps Lifecycle Governance')
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: === Co-occurrence Analysis ===
File Loaded: step2_crosswalk_matrix.csv
Dataset shape: (42, 26)

Contingency Table:
             No Eval Infra  Has Eval Infra
No Ops Gov              24               8
Has Ops Gov              5               5

Chi-square Statistic: 2.2281
P-value (Chi-square): 0.1355
P-value (Fisher's Exact): 0.2383
Phi Coefficient: 0.2303

Result: No statistically significant co-occurrence detected (Null hypothesis not rejected).


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **Heatmap** representing a **Contingency Table** (or Cross-tabulation).
*   **Purpose:** It visualizes the co-occurrence frequency between two categorical variables: the presence of GenAIOps/MLOps Lifecycle Governance and the presence of Evaluation & Monitoring Infrastructure. The color intensity represents the magnitude of the count in each category intersection.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Evaluation & Monitoring Infrastructure"
    *   **Categories:** "No Eval Infra" (Left) and "Has Eval Infra" (Right).
*   **Y-Axis:**
    *   **Label:** "GenAIOps / MLOps Lifecycle Governance"
    *   **Categories:** "No Ops Gov" (Top) and "Has Ops Gov" (Bottom).
*   **Value Ranges:** The axes represent binary categories (No vs. Has). The numerical values within the cells range from **5 to 24**.

### 3. Data Trends
*   **Highest Value (Darkest Blue):** The intersection of **"No Ops Gov"** and **"No Eval Infra"** contains the highest count (**24**). This indicates that the majority of the sample lacks both governance and evaluation infrastructure.
*   **Lowest Value (Lightest Color):** Two intersections share the lowest count of **5**:
    *   "Has Ops Gov" and "No Eval Infra"
    *   "Has Ops Gov" and "Has Eval Infra"
*   **Secondary Cluster:** The intersection of "No Ops Gov" and "Has Eval Infra" has a count of **8**, showing that some entities possess evaluation infrastructure despite lacking lifecycle governance.

### 4. Annotations and Legends
*   **Title:** "Co-occurrence of GenAIOps and Eval Infrastructure" clearly defines the scope of the comparison.
*   **Cell Annotations:** Each cell contains a numerical annotation (24, 8, 5, 5) indicating the exact count of observations for that specific overlap.
*   **Color Coding:** A sequential blue color palette is used, where dark navy blue represents the highest frequency and white/pale blue represents the lowest frequency.

### 5. Statistical Insights
*   **Low Maturity Dominance:** A significant majority of the dataset (**57%**, calculated as 24/42) falls into the "No Ops Gov / No Eval Infra" category, suggesting that most systems or organizations surveyed are at a low level of maturity regarding both GenAI operations and evaluation.
*   **Governance Scarcity:** Only **10 out of 42** respondents (approx. 24%) have "GenAIOps / MLOps Lifecycle Governance" (combining the bottom row: 5 + 5).
*   **Evaluation vs. Governance:**
    *   It appears slightly more common to have Evaluation Infrastructure without Governance (8 counts) than to have Governance without Evaluation Infrastructure (5 counts).
    *   Only **12%** (5/42) of the sample possesses **both** capabilities ("Has Ops Gov" and "Has Eval Infra"), indicating that a fully mature setup is currently rare in this dataset.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
