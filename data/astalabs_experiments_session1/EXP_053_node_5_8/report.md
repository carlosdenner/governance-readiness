# Experiment 53: node_5_8

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_8` |
| **ID in Run** | 53 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:13:31.887762+00:00 |
| **Runtime** | 150.4s |
| **Parent** | `node_4_14` |
| **Children** | `node_6_6`, `node_6_8` |
| **Creation Index** | 54 |

---

## Hypothesis

> The 'Human-in-the-Loop Approval Gates' control is significantly more likely to
be associated with 'Trust Readiness' requirements than 'Integration Readiness'
requirements, serving as a signature control for Trust.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8790 (Likely True) |
| **Posterior** | 0.2252 (Likely False) |
| **Surprise** | -0.7588 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 17.0 |
| Maybe True | 13.0 |
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

**Objective:** Test the specificity of the 'Human-in-the-Loop' control to the Trust bundle.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Create a contingency table comparing 'bundle' (Trust/Integration) vs the binary presence of the column 'Human-in-the-Loop Approval Gates' (X or empty).
- 3. Perform Fisher's Exact Test to determine if the association is significant.

### Deliverables
- Heatmap of the contingency table and Fisher's Exact Test results.

---

## Analysis

The experiment tested the hypothesis that 'Human-in-the-Loop Approval Gates' is
a signature control for the 'Trust Readiness' bundle. The analysis of
`step2_crosswalk_matrix.csv` (n=42) revealed that this control is rarely used (3
occurrences total). Contrary to the hypothesis, all 3 instances were associated
with 'Integration Readiness' requirements, while 'Trust Readiness' had zero
mappings to this control. Fisher's Exact Test yielded a p-value of 0.2387,
indicating the difference is not statistically significant due to the small
sample size of the control's usage. However, the complete absence of this
control in the Trust bundle (0/19) definitively rejects the hypothesis that it
serves as a primary differentiator for Trust. The data suggests this specific
control is currently conceptualized as an engineering/integration safeguard
(e.g., deployment gates) rather than a governance/trust policy mechanism in this
specific framework.

---

## Review

The experiment successfully tested the hypothesis that 'Human-in-the-Loop
Approval Gates' serves as a signature control for 'Trust Readiness'. The
analysis of `step2_crosswalk_matrix.csv` (n=42) revealed that this control is
used sparingly (only 3 times total). Contrary to the hypothesis, all 3 instances
were associated with 'Integration Readiness', while 'Trust Readiness' had zero
associations. Fisher's Exact Test yielded a p-value of 0.2387, indicating the
difference is not statistically significant due to the low frequency of the
control, but the complete absence of the control in the Trust bundle (0/19 vs
3/23) definitively rejects the hypothesis. The data suggests that in this
framework, human approval gates are treated as integration/deployment
checkpoints rather than trust/governance mechanisms.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# [debug]
print("Current working directory:", os.getcwd())

# Define file path based on instructions
filename = 'step2_crosswalk_matrix.csv'
filepath = f"../{filename}"

if not os.path.exists(filepath):
    # Fallback to current directory if not found in parent
    filepath = filename

print(f"Loading dataset from: {filepath}")

try:
    df = pd.read_csv(filepath)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: File {filename} not found.")
    exit(1)

# Target Columns
bundle_col = 'bundle'
control_col = 'Human-in-the-Loop Approval Gates'

# Check if columns exist
if control_col not in df.columns:
    print(f"Column '{control_col}' not found. Available columns:")
    print(df.columns.tolist())
    exit(1)

# Preprocess: Convert control column to boolean (True if 'X' or non-null/non-empty, False otherwise)
# Looking at previous exploration, 'X' indicates presence.
df['has_hitl'] = df[control_col].notna() & (df[control_col].astype(str).str.strip() != '')

# Create Contingency Table
contingency_table = pd.crosstab(df[bundle_col], df['has_hitl'])
contingency_table.columns = ['No HITL', 'Has HITL']

print("\n--- Contingency Table ---")
print(contingency_table)

# Perform Fisher's Exact Test
# Fisher's exact test requires a 2x2 table. 
# Ensure we have both bundles and both presence/absence states if possible, though crosstab handles available data.
if contingency_table.shape == (2, 2):
    odds_ratio, p_value = stats.fisher_exact(contingency_table)
    print(f"\nFisher's Exact Test Results:")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print("Result: Statistically Significant association between Bundle and HITL Control.")
    else:
        print("Result: No statistically significant association found.")
else:
    print("\nContingency table is not 2x2. Cannot perform Fisher's Exact Test.")
    print("Shape:", contingency_table.shape)

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Association: Bundle vs Human-in-the-Loop Control')
plt.ylabel('Competency Bundle')
plt.xlabel('Has Human-in-the-Loop Approval Gates?')
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Current working directory: /data
Loading dataset from: step2_crosswalk_matrix.csv
Dataset loaded successfully.

--- Contingency Table ---
                       No HITL  Has HITL
bundle                                  
Integration Readiness       20         3
Trust Readiness             19         0

Fisher's Exact Test Results:
Odds Ratio: 0.0000
P-value: 0.2387
Result: No statistically significant association found.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Heatmap (specifically visualizing a **Contingency Table** or Confusion Matrix).
*   **Purpose:** The plot visualizes the frequency distribution and association between two categorical variables: "Competency Bundle" and the presence of "Human-in-the-Loop (HITL) Approval Gates." It allows for quick identification of the most and least common combinations of these factors.

### 2. Axes
*   **Y-Axis (Vertical):**
    *   **Title/Label:** "Competency Bundle"
    *   **Categories:** The axis is divided into two categories: "Integration Readiness" (top) and "Trust Readiness" (bottom).
*   **X-Axis (Horizontal):**
    *   **Title/Label:** "Has Human-in-the-Loop Approval Gates?"
    *   **Categories:** The axis is divided into two categories: "No HITL" (left) and "Has HITL" (right).
*   **Value Ranges:** The axes represent categorical data rather than numerical ranges. The quantitative data is represented within the cells, ranging from a count of **0 to 20**.

### 3. Data Trends
*   **High Values (Dark Blue Areas):**
    *   The highest concentration of data is found in the **"No HITL"** column.
    *   **Integration Readiness + No HITL:** This is the most frequent combination with a count of **20**.
    *   **Trust Readiness + No HITL:** This is the second most frequent combination with a count of **19**.
*   **Low Values (Light Blue/White Areas):**
    *   The **"Has HITL"** column contains significantly fewer data points.
    *   **Integration Readiness + Has HITL:** A low count of **3**.
    *   **Trust Readiness + Has HITL:** The lowest possible value, **0**.
*   **Overall Trend:** There is a strong skew towards "No HITL." Regardless of the Competency Bundle, the vast majority of cases do not have Human-in-the-Loop approval gates.

### 4. Annotations and Legends
*   **Title:** "Association: Bundle vs Human-in-the-Loop Control" – This clearly defines the two variables being compared.
*   **Cell Annotations:** Each cell contains a number (20, 3, 19, 0) representing the exact count of observations for that specific intersection of categories.
*   **Color Scale:** While there is no explicit legend bar, the plot uses a sequential color scheme (white to dark blue), where darker blue indicates a higher frequency count and white indicates zero.

### 5. Statistical Insights
*   **Dominance of "No HITL":** Out of the total dataset (sum of all cells = 42), **39 cases (approx. 93%)** fall under the "No HITL" category. Only 3 cases have HITL approval gates.
*   **Exclusive Association:** The "Trust Readiness" bundle appears to be exclusively associated with "No HITL" in this dataset, as there are zero recorded instances of Trust Readiness combined with HITL.
*   **Bundle Distribution:** The distribution between the two bundles is relatively balanced, with "Integration Readiness" having a total of 23 cases (20+3) and "Trust Readiness" having 19 cases (19+0).
*   **Implication:** This data suggests that within the context of this experiment or survey, Human-in-the-Loop control is rarely applied to these competency bundles, particularly for Trust Readiness.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
