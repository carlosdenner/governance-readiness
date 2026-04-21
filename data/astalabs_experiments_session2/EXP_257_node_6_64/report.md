# Experiment 257: node_6_64

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_64` |
| **ID in Run** | 257 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:13:04.804343+00:00 |
| **Runtime** | 406.0s |
| **Parent** | `node_5_14` |
| **Children** | None |
| **Creation Index** | 258 |

---

## Hypothesis

> The Vendor 'Black Box' Paradox: Government use cases relying on 'Commercial' AI
are statistically less likely to possess 'Code Access' or perform 'Impact
Assessments' than 'Custom/In-house' systems, confirming that proprietary
procurement hinders critical governance visibility.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.7967 (Likely True) |
| **Surprise** | -0.2246 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 30.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 48.0 |
| Uncertain | 12.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Assess the impact of commercial procurement on governance transparency.

### Steps
- 1. Load the 'eo13960_scored' subset.
- 2. Clean '10_commercial_ai' to binary (Commercial vs. Custom/Other).
- 3. Clean '38_code_access' to binary (Yes vs. No/Other).
- 4. Clean '52_impact_assessment' to binary (Yes vs. No/Other).
- 5. Perform two Chi-Square tests: (Commercial vs. Code Access) and (Commercial vs. Impact Assessment).
- 6. Compute Odds Ratios to quantify the strength of the negative association.

### Deliverables
- Two contingency tables; Chi-Square results for both tests; Odds Ratios indicating the likelihood of transparency gaps in commercial systems.

---

## Analysis

The experiment was successfully executed using `37_custom_code` as a robust
proxy for the Commercial vs. Custom distinction (yielding 967 analyzable cases).

**Findings:**

1.  **Code Access (Strongly Supported):** The 'Black Box' effect is undeniable
regarding technical transparency. Commercial systems are approximately **33
times less likely** (Odds Ratio: 0.03) to provide source code access compared to
custom/in-house systems. This difference is statistically significant (Chi-
Square: 367.6, p < 1e-80).

2.  **Impact Assessments (Not Supported):** The hypothesis that commercial
procurement hinders impact assessment compliance was **not** supported. While
commercial systems had a slightly lower assessment rate (4.5%) compared to
custom systems (6.6%), the difference was **not statistically significant** (p =
0.29).

**Conclusion:** The 'Vendor Black Box' paradox effectively blocks technical
auditability (Code Access) but does not appear to be the primary driver for the
lack of Impact Assessments. Instead, the data reveals a systemic failure across
*both* procurement types, with over 93% of all systems lacking completed impact
assessments regardless of their origin.

---

## Review

The experiment was successfully executed after a necessary pivot in the data
cleaning strategy. The initial attempt to use '10_commercial_ai' was correctly
identified as flawed due to non-binary values. The programmer successfully
identified '37_custom_code' as a robust proxy for the Commercial (COTS) vs.
Custom (In-house) distinction, allowing for a valid test of the 'Black Box'
hypothesis. The analysis was rigorous, using Chi-Square tests and Odds Ratios to
quantify the findings.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import sys
import os

# --- Step 1: Load the dataset ---
file_name = 'astalabs_discovery_all_data.csv'
file_path = file_name
if not os.path.exists(file_path):
    file_path = f'../{file_name}'
    if not os.path.exists(file_path):
        print(f"Error: File {file_name} not found.")
        sys.exit(1)

print(f"Loading dataset from {file_path}...")
df = pd.read_csv(file_path, low_memory=False)

# Filter for the relevant source table
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded {len(df_eo)} rows from EO 13960 subset.")

# --- Step 2: Data Cleaning ---

# Independent Variable: Commercial vs Custom (Proxy: 37_custom_code)
# 'No' -> No custom code -> Commercial/COTS
# 'Yes' -> Custom code -> Custom/In-house
def clean_commercial_proxy(val):
    s = str(val).strip().lower()
    if s == 'no':
        return 'Commercial'
    elif s == 'yes':
        return 'Custom/In-house'
    return None

# Dependent Variable 1: Code Access (38_code_access)
# 'Yes...' -> Yes, 'No...' -> No
def clean_code_access(val):
    s = str(val).strip().lower()
    if s.startswith('yes'):
        return 'Yes'
    # Treat No, nan, blank, or other descriptions as 'No'
    return 'No'

# Dependent Variable 2: Impact Assessment (52_impact_assessment)
# 'Yes' -> Yes, 'No' or 'Planned' -> No
def clean_impact_assessment(val):
    s = str(val).strip().lower()
    if s == 'yes':
        return 'Yes'
    # 'planned or in-progress' counts as No for "completed" assessment
    return 'No'

# Apply cleaning
df_eo['Commercial_Status'] = df_eo['37_custom_code'].apply(clean_commercial_proxy)
df_eo['Has_Code_Access'] = df_eo['38_code_access'].apply(clean_code_access)
df_eo['Has_Impact_Assessment'] = df_eo['52_impact_assessment'].apply(clean_impact_assessment)

# Filter to valid groups
df_clean = df_eo.dropna(subset=['Commercial_Status'])

print(f"Analyzable use cases after cleaning: {len(df_clean)}")
print("Group Distribution:")
print(df_clean['Commercial_Status'].value_counts())

# --- Step 3: Analysis - Commercial vs Code Access ---
print("\n=======================================================")
print("TEST 1: Commercial Status vs. Code Access")
print("=======================================================")

ct_code = pd.crosstab(df_clean['Commercial_Status'], df_clean['Has_Code_Access'])
print("Contingency Table (Code Access):")
print(ct_code)

chi2_code, p_code, dof_code, ex_code = chi2_contingency(ct_code)
print(f"\nChi-Square Statistic: {chi2_code:.4f}")
print(f"P-Value: {p_code:.4e}")

# Calculate Odds Ratio (Odds of Commercial having Access / Odds of Custom having Access)
try:
    # Add small smoothing to avoid div by zero if necessary
    smoothing = 0.5 if (ct_code == 0).any().any() else 0
    
    n_comm_yes = ct_code.loc['Commercial', 'Yes'] + smoothing
    n_comm_no = ct_code.loc['Commercial', 'No'] + smoothing
    n_cust_yes = ct_code.loc['Custom/In-house', 'Yes'] + smoothing
    n_cust_no = ct_code.loc['Custom/In-house', 'No'] + smoothing
    
    odds_comm = n_comm_yes / n_comm_no
    odds_cust = n_cust_yes / n_cust_no
    
    or_code = odds_comm / odds_cust
    print(f"Odds Ratio (Commercial vs Custom): {or_code:.4f}")
    if or_code < 1:
        print(f"Interpretation: Commercial systems are {1/or_code:.2f}x LESS likely to have Code Access.")
except Exception as e:
    print(f"Could not calculate Odds Ratio: {e}")

# --- Step 4: Analysis - Commercial vs Impact Assessment ---
print("\n=======================================================")
print("TEST 2: Commercial Status vs. Impact Assessment")
print("=======================================================")

ct_impact = pd.crosstab(df_clean['Commercial_Status'], df_clean['Has_Impact_Assessment'])
print("Contingency Table (Impact Assessment):")
print(ct_impact)

chi2_impact, p_impact, dof_impact, ex_impact = chi2_contingency(ct_impact)
print(f"\nChi-Square Statistic: {chi2_impact:.4f}")
print(f"P-Value: {p_impact:.4e}")

# Calculate Odds Ratio
try:
    smoothing = 0.5 if (ct_impact == 0).any().any() else 0
    
    n_comm_yes = ct_impact.loc['Commercial', 'Yes'] + smoothing
    n_comm_no = ct_impact.loc['Commercial', 'No'] + smoothing
    n_cust_yes = ct_impact.loc['Custom/In-house', 'Yes'] + smoothing
    n_cust_no = ct_impact.loc['Custom/In-house', 'No'] + smoothing
    
    odds_comm = n_comm_yes / n_comm_no
    odds_cust = n_cust_yes / n_cust_no
    
    or_impact = odds_comm / odds_cust
    print(f"Odds Ratio (Commercial vs Custom): {or_impact:.4f}")
    if or_impact < 1:
        print(f"Interpretation: Commercial systems are {1/or_impact:.2f}x LESS likely to have Impact Assessments.")
except Exception as e:
    print(f"Could not calculate Odds Ratio: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
Loaded 1757 rows from EO 13960 subset.
Analyzable use cases after cleaning: 967
Group Distribution:
Commercial_Status
Custom/In-house    722
Commercial         245
Name: count, dtype: int64

=======================================================
TEST 1: Commercial Status vs. Code Access
=======================================================
Contingency Table (Code Access):
Has_Code_Access     No  Yes
Commercial_Status          
Commercial         219   26
Custom/In-house    147  575

Chi-Square Statistic: 367.5988
P-Value: 6.2376e-82
Odds Ratio (Commercial vs Custom): 0.0304
Interpretation: Commercial systems are 32.95x LESS likely to have Code Access.

=======================================================
TEST 2: Commercial Status vs. Impact Assessment
=======================================================
Contingency Table (Impact Assessment):
Has_Impact_Assessment   No  Yes
Commercial_Status              
Commercial             234   11
Custom/In-house        674   48

Chi-Square Statistic: 1.1346
P-Value: 2.8679e-01
Odds Ratio (Commercial vs Custom): 0.6601
Interpretation: Commercial systems are 1.51x LESS likely to have Impact Assessments.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
