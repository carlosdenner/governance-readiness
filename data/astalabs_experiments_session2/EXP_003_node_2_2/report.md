# Experiment 3: node_2_2

| Property | Value |
|---|---|
| **Experiment ID** | `node_2_2` |
| **ID in Run** | 3 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T01:11:59.809669+00:00 |
| **Runtime** | 315.8s |
| **Parent** | `node_1_0` |
| **Children** | `node_3_1`, `node_3_10`, `node_3_18` |
| **Creation Index** | 4 |

---

## Hypothesis

> Commercial 'Black-Box' Opacity: AI systems procured from commercial vendors
(COTS) exhibit significantly lower rates of deep governance compliance
(specifically Independent Evaluation and Impact Assessments) compared to custom-
developed government systems, due to vendor proprietary constraints.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.2328 (Likely False) |
| **Surprise** | -0.6207 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 25.0 |
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
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Compare Tier 2 governance readiness scores between commercial and custom AI systems in the federal inventory.

### Steps
- 1. Load the `astalabs_discovery_all_data.csv` dataset and filter for rows where `source_table` is 'eo13960_scored'.
- 2. Create a binary variable `is_commercial` based on the column `10_commercial_ai` (Yes vs No/Custom).
- 3. Create binary compliance flags for `52_impact_assessment` and `55_independent_eval` (convert 'Yes'/'No' text to 1/0).
- 4. Perform a Chi-square test of independence to compare the rates of compliance for both controls between the `is_commercial` groups.
- 5. Calculate the Odds Ratio to quantify the likelihood of compliance for custom systems vs commercial systems.

### Deliverables
- Contingency tables, Chi-square test statistics (p-values), and Odds Ratios for Impact Assessment and Independent Evaluation compliance.

---

## Analysis

The experiment successfully loaded the EO 13960 dataset and compared governance
compliance rates between Custom (n=1357) and Commercial (n=353) AI systems.
Contrary to the hypothesis that commercial 'Black-Box' systems would show
significantly lower compliance, the analysis revealed no statistically
significant difference between the two groups for either control.

For Impact Assessments (Control 52), the compliance rate was 3.46% for Custom vs
3.40% for Commercial systems (Chi-Square: 0.00, p=1.00, OR=0.98). For
Independent Evaluation (Control 55), the compliance rate was 2.51% for Custom vs
2.27% for Commercial systems (Chi-Square: 0.004, p=0.95, OR=0.90).

These results indicate that deep governance compliance is uniformly low across
the federal inventory, regardless of whether the system is custom-developed or
commercially procured. The hypothesis is rejected based on this dataset.

---

## Review

The experiment was successfully executed according to the plan. The code
correctly loaded the EO 13960 dataset, segmented the systems into Commercial vs.
Custom groups using the specified '10_commercial_ai' column, and performed the
required Chi-square tests and Odds Ratio calculations.

**Hypothesis Test Results:**
1.  **Impact Assessment (Control 52):**
    *   Compliance Rates: Custom (3.46%) vs. Commercial (3.40%).
    *   Stats: Chi-Square = 0.00, p-value = 1.00, Odds Ratio = 0.98.
    *   Result: No statistically significant difference.
2.  **Independent Evaluation (Control 55):**
    *   Compliance Rates: Custom (2.51%) vs. Commercial (2.27%).
    *   Stats: Chi-Square = 0.0043, p-value = 0.95, Odds Ratio = 0.90.
    *   Result: No statistically significant difference.

**Conclusion:**
The hypothesis that commercial 'Black-Box' systems exhibit significantly lower
rates of deep governance compliance is **rejected**. The analysis reveals that
compliance with deep governance controls (Impact Assessments and Independent
Evaluations) is uniformly low (<4%) across the federal inventory, regardless of
whether the system is identified as a specific commercial use case or a
custom/other implementation.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import numpy as np
import os

# Define file path
file_path = 'astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = '../astalabs_discovery_all_data.csv'

# Load dataset
print(f"Loading dataset from {file_path}...")
df = pd.read_csv(file_path, low_memory=False)

# Filter for EO 13960 Scored
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Subset shape: {subset.shape}")

# --- Step 1: Analyze and Create 'is_commercial' ---
print("\n--- Value Counts for '10_commercial_ai' (Top 5) ---")
print(subset['10_commercial_ai'].value_counts(dropna=False).head(5))

# Logic: 'None of the above.' = Custom (0), Any other text = Commercial (1)
def map_commercial(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    if val_str == "None of the above.":
        return 0 # Custom
    else:
        return 1 # Commercial

subset['is_commercial'] = subset['10_commercial_ai'].apply(map_commercial)
subset = subset.dropna(subset=['is_commercial'])

print("\n--- Commercial vs Custom Counts ---")
print(subset['is_commercial'].value_counts())

# --- Step 2: Analyze and Create Compliance Flags ---
# Inspect values to determine binarization logic
print("\n--- Value Counts for '52_impact_assessment' ---")
print(subset['52_impact_assessment'].value_counts(dropna=False).head())
print("\n--- Value Counts for '55_independent_eval' ---")
print(subset['55_independent_eval'].value_counts(dropna=False).head())

def binarize_compliance(val):
    if pd.isna(val):
        return 0 # Treat missing as No
    val_str = str(val).lower().strip()
    if val_str.startswith('yes'):
        return 1
    return 0

subset['has_impact_assess'] = subset['52_impact_assessment'].apply(binarize_compliance)
subset['has_indep_eval'] = subset['55_independent_eval'].apply(binarize_compliance)

# --- Step 3: Statistical Analysis ---
def run_stats(df, target_col, label):
    print(f"\n=== Analysis for: {label} ===")
    
    # Contingency Table
    # Rows: 0=Custom, 1=Commercial
    # Cols: 0=No, 1=Yes
    ct = pd.crosstab(df['is_commercial'], df[target_col])
    
    # Ensure 2x2
    ct = ct.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    
    print("Contingency Table (Rows: 0=Custom, 1=Comm; Cols: 0=No, 1=Yes):")
    print(ct)
    
    # Check if we have enough data
    if (ct.values == 0).any():
        print("Warning: Zero values in contingency table. Adding correction (0.5) for Odds Ratio.")
        ct_adj = ct + 0.5
    else:
        ct_adj = ct
        
    # Chi-Square
    # Use the original table for Chi2, unless margins are 0
    try:
        chi2, p, dof, expected = stats.chi2_contingency(ct)
    except ValueError as e:
        print(f"Chi-Square failed: {e}")
        chi2, p = 0, 1

    # Odds Ratio Calculation
    # OR = (Odds Commercial) / (Odds Custom)
    # Odds = Yes / No
    # OR = (Comm_Yes/Comm_No) / (Cust_Yes/Cust_No)
    
    comm_yes = ct_adj.loc[1, 1]
    comm_no = ct_adj.loc[1, 0]
    cust_yes = ct_adj.loc[0, 1]
    cust_no = ct_adj.loc[0, 0]
    
    odds_comm = comm_yes / comm_no
    odds_cust = cust_yes / cust_no
    or_val = odds_comm / odds_cust
    
    # Output
    print(f"Compliance Rate (Custom):     {cust_yes/(cust_yes+cust_no):.2%}")
    print(f"Compliance Rate (Commercial): {comm_yes/(comm_yes+comm_no):.2%}")
    print(f"Chi-Square Statistic:         {chi2:.4f}")
    print(f"P-Value:                      {p:.5f}")
    print(f"Odds Ratio (Comm vs Cust):    {or_val:.4f}")
    
    if p < 0.05:
        print("Result: Significant difference detected.")
    else:
        print("Result: No significant difference.")

run_stats(subset, 'has_impact_assess', 'Impact Assessment (Control 52)')
run_stats(subset, 'has_indep_eval', 'Independent Evaluation (Control 55)')

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
Subset shape: (1757, 196)

--- Value Counts for '10_commercial_ai' (Top 5) ---
10_commercial_ai
None of the above.                                                                      1357
Searching for information using AI.                                                      109
NaN                                                                                       47
Creating visual representations of data sets for reports and presentations using AI.      38
Summarizing the key points of a lengthy report using AI.                                  37
Name: count, dtype: int64

--- Commercial vs Custom Counts ---
is_commercial
0.0    1357
1.0     353
Name: count, dtype: int64

--- Value Counts for '52_impact_assessment' ---
52_impact_assessment
NaN                        1557
No                           63
Yes                          58
Planned or in-progress.      31
YES                           1
Name: count, dtype: int64

--- Value Counts for '55_independent_eval' ---
55_independent_eval
NaN                                                                                                      1595
Planned or in-progress                                                                                     59
Yes – by the CAIO                                                                                          22
Yes – by another appropriate agency office that was not directly involved in the system’s development      18
TRUE                                                                                                       10
Name: count, dtype: int64

=== Analysis for: Impact Assessment (Control 52) ===
Contingency Table (Rows: 0=Custom, 1=Comm; Cols: 0=No, 1=Yes):
has_impact_assess     0   1
is_commercial              
0                  1310  47
1                   341  12
Compliance Rate (Custom):     3.46%
Compliance Rate (Commercial): 3.40%
Chi-Square Statistic:         0.0000
P-Value:                      1.00000
Odds Ratio (Comm vs Cust):    0.9808
Result: No significant difference.

=== Analysis for: Independent Evaluation (Control 55) ===
Contingency Table (Rows: 0=Custom, 1=Comm; Cols: 0=No, 1=Yes):
has_indep_eval     0   1
is_commercial           
0               1323  34
1                345   8
Compliance Rate (Custom):     2.51%
Compliance Rate (Commercial): 2.27%
Chi-Square Statistic:         0.0043
P-Value:                      0.94763
Odds Ratio (Comm vs Cust):    0.9023
Result: No significant difference.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
