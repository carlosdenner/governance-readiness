# Experiment 49: node_5_7

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_7` |
| **ID in Run** | 49 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:07:04.766382+00:00 |
| **Runtime** | 212.0s |
| **Parent** | `node_4_6` |
| **Children** | `node_6_3` |
| **Creation Index** | 50 |

---

## Hypothesis

> The 'Audit Logging & Telemetry' architecture control is significantly more
prevalent in 'Trust Readiness' requirements than in 'Integration Readiness'
requirements, serving as a primary mechanism for governance verification.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7984 (Likely True) |
| **Posterior** | 0.2045 (Likely False) |
| **Surprise** | -0.6892 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 7.0 |
| Maybe True | 23.0 |
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

**Objective:** Verify if observability controls are structurally biased towards governance bundles by comparing the frequency of the 'Audit Logging & Telemetry' control in 'Trust Readiness' versus 'Integration Readiness' requirements.

### Steps
- 1. Load the dataset 'step2_crosswalk_matrix.csv'.
- 2. Verify the existence of the columns 'Audit Logging & Telemetry' and 'bundle'.
- 3. Preprocess the 'Audit Logging & Telemetry' column to create a binary variable (1 if value is 'X', 0 otherwise).
- 4. Create a contingency table showing the count of requirements with and without this control for each 'bundle' category ('Trust Readiness' vs. 'Integration Readiness').
- 5. Calculate the percentage prevalence of the control in each bundle.
- 6. Perform a Fisher's Exact Test on the contingency table to determine if the difference in prevalence is statistically significant (p < 0.05).

### Deliverables
- 1. A contingency table of control presence by bundle.
- 2. Fisher's Exact Test results (p-value, odds ratio).
- 3. A bar chart comparing the percentage of requirements containing 'Audit Logging & Telemetry' in Trust vs. Integration bundles.

---

## Analysis

The experiment successfully analyzed the prevalence of 'Audit Logging &
Telemetry' in 'step2_crosswalk_matrix.csv' after correcting the file path. The
data reveals that this architecture control is extremely rare across the entire
dataset. It appears in only **4.3%** (1/23) of 'Integration Readiness'
requirements and is completely absent (**0.0%**, 0/19) from 'Trust Readiness'
requirements.

The Fisher's Exact Test yielded a p-value of **1.0**, confirming that the
observed difference is not statistically significant. Consequently, the
hypothesis that 'Audit Logging & Telemetry' is significantly more prevalent in
'Trust Readiness' is **rejected**. The results indicate that this specific
control is not currently a primary differentiator for governance-focused
competencies in this framework, and is in fact slightly (though not
significantly) more associated with the engineering/integration bundle.

---

## Review

The experiment was successfully re-run with the correct file path. The analysis
faithfully implements the experiment plan, generating the required contingency
table, statistical test (Fisher's Exact Test), and visualization. The results
are clear: the 'Audit Logging & Telemetry' control is nearly absent in this
dataset (found in only 1 Integration requirement and 0 Trust requirements),
leading to a p-value of 1.0. The hypothesis is correctly rejected based on these
findings.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Try to locate the file in the current directory or one level up
filename = 'step2_crosswalk_matrix.csv'
if os.path.exists(filename):
    file_path = filename
elif os.path.exists(f'../{filename}'):
    file_path = f'../{filename}'
else:
    raise FileNotFoundError(f"Could not find {filename} in current or parent directory.")

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# Target columns
control_col = 'Audit Logging & Telemetry'
bundle_col = 'bundle'

# Clean and prepare data
# Treat 'X' as Present (1) and NaN/Empty as Absent (0)
df['has_control'] = df[control_col].fillna('').apply(lambda x: 1 if str(x).strip().upper() == 'X' else 0)

# Create Contingency Table
contingency_table = pd.crosstab(df[bundle_col], df['has_control'])

# Ensure both 0 (Absent) and 1 (Present) columns exist
if 0 not in contingency_table.columns:
    contingency_table[0] = 0
if 1 not in contingency_table.columns:
    contingency_table[1] = 0

# Reorder columns to Absent (0), Present (1)
contingency_table = contingency_table[[0, 1]]
contingency_table.columns = ['Absent', 'Present']

print("\n--- Contingency Table (Count) ---")
print(contingency_table)

# Calculate Percentages
contingency_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
print("\n--- Contingency Table (Percentage) ---")
print(contingency_pct)

# Perform Fisher's Exact Test
try:
    # Extract counts for the test
    # Row 1: Integration Readiness
    # Row 2: Trust Readiness
    # Columns: Present, Absent (swapped from table for Odds Ratio interpretation: Present/Absent)
    
    ir_present = contingency_table.loc['Integration Readiness', 'Present']
    ir_absent = contingency_table.loc['Integration Readiness', 'Absent']
    tr_present = contingency_table.loc['Trust Readiness', 'Present']
    tr_absent = contingency_table.loc['Trust Readiness', 'Absent']
    
    # Matrix for Fisher's: [[Trust_Present, Trust_Absent], [Integration_Present, Integration_Absent]]
    # Testing if Trust has higher prevalence than Integration
    fisher_matrix = [[tr_present, tr_absent], [ir_present, ir_absent]]
    
    odds_ratio, p_value = stats.fisher_exact(fisher_matrix, alternative='two-sided')

    print("\n--- Statistical Test Results ---")
    print(f"Fisher's Exact Test P-value: {p_value:.4f}")
    print(f"Odds Ratio (Trust/Integration): {odds_ratio:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print("Result: Statistically Significant (Reject Null Hypothesis)")
    else:
        print("Result: Not Statistically Significant (Fail to Reject Null Hypothesis)")

except KeyError as e:
    print(f"Error processing bundles for stats: {e}")

# Visualization
plt.figure(figsize=(8, 6))
bundles = contingency_pct.index
percentages = contingency_pct['Present']

# Color mapping
colors = ['salmon' if 'Trust' in b else 'skyblue' for b in bundles]

bars = plt.bar(bundles, percentages, color=colors)

plt.title(f"Prevalence of '{control_col}' by Bundle")
plt.ylabel('Percentage of Requirements with Control (%)')
plt.xlabel('Competency Bundle')
plt.ylim(0, 100)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step2_crosswalk_matrix.csv

--- Contingency Table (Count) ---
                       Absent  Present
bundle                                
Integration Readiness      22        1
Trust Readiness            19        0

--- Contingency Table (Percentage) ---
                           Absent   Present
bundle                                     
Integration Readiness   95.652174  4.347826
Trust Readiness        100.000000  0.000000

--- Statistical Test Results ---
Fisher's Exact Test P-value: 1.0000
Odds Ratio (Trust/Integration): 0.0000
Result: Not Statistically Significant (Fail to Reject Null Hypothesis)


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot (Column Chart).
*   **Purpose:** To compare the prevalence (percentage) of a specific control feature, "Audit Logging & Telemetry," across distinct categorical groups known as "Competency Bundles."

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Competency Bundle"
    *   **Categories:** The axis displays two discrete categories: "Integration Readiness" and "Trust Readiness."
*   **Y-Axis:**
    *   **Label:** "Percentage of Requirements with Control (%)"
    *   **Range:** The axis ranges from 0 to 100.
    *   **Scale:** The scale is linear with major tick marks every 20 units (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Tallest Bar:** The "Integration Readiness" category has the highest value, represented by a light blue bar.
*   **Shortest Bar:** The "Trust Readiness" category has the lowest value, represented by the absence of a visible bar (height is zero).
*   **Pattern:** Both categories show extremely low prevalence values. The data is heavily skewed toward the bottom of the 0-100% scale, indicating that this specific control is rare in both bundles.

### 4. Annotations and Legends
*   **Title:** "Prevalence of 'Audit Logging & Telemetry' by Bundle" appears at the top center.
*   **Data Labels:**
    *   Above the "Integration Readiness" bar, there is a specific numerical annotation: **"4.3%"**.
    *   Above the "Trust Readiness" position, there is a specific numerical annotation: **"0.0%"**.
*   **Legend:** There is no separate legend box; the categories are identified directly on the x-axis.

### 5. Statistical Insights
*   **Absence in Trust Readiness:** The "Trust Readiness" bundle has zero coverage (0.0%) for "Audit Logging & Telemetry," suggesting this specific control is currently completely missing from that bundle's requirements.
*   **Low Prevalence in Integration Readiness:** While "Integration Readiness" performs better than "Trust Readiness," the prevalence is still very low at only 4.3%. This indicates that less than 1 in 20 requirements in this bundle include this control.
*   **Overall Gap:** The data highlights a significant gap in "Audit Logging & Telemetry" controls across both competency bundles analyzed. Neither bundle comes close to substantial coverage.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
