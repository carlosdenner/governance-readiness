# Experiment 289: node_6_81

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_81` |
| **ID in Run** | 289 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T14:53:33.344007+00:00 |
| **Runtime** | 278.9s |
| **Parent** | `node_5_95` |
| **Children** | None |
| **Creation Index** | 290 |

---

## Hypothesis

> Government AI systems involving 'Biometric' or 'Facial Recognition' technologies
are significantly more likely to report active 'Disparity Mitigation' measures
(e.g., testing, monitoring, human review) compared to other AI systems,
reflecting higher scrutiny.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9758 (Definitely True) |
| **Posterior** | 0.9918 (Definitely True) |
| **Surprise** | +0.0191 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 29.0 |
| Maybe True | 1.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 60.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Assess whether biometric AI systems have a higher rate of documented disparity mitigation compared to non-biometric systems by parsing free-text governance responses.

### Steps
- 1. Load the 'astalabs_discovery_all_data.csv' dataset and filter for rows where 'source_table' is 'eo13960_scored'.
- 2. Identify the 'Disparity Mitigation' column (look for 'disparity_mitigation' in column names, likely index 62 or similar).
- 3. Create a boolean column 'Is_Biometric' by checking if '2_use_case_name' or '11_purpose_benefits' contains any of the keywords: ['face', 'facial', 'biometric', 'recognition', 'surveillance', 'gait', 'iris'] (case-insensitive).
- 4. Create a boolean column 'Has_Mitigation' by parsing the mitigation column. Convert text to lowercase. Set to True if the text contains action-oriented keywords: ['test', 'eval', 'monitor', 'assess', 'audit', 'mitigat', 'review', 'human', 'valid', 'bias', 'fair', 'check']. Set to False if the text is missing, 'nan', or fails to match these keywords. (Note: The column contains free text descriptions, not just Yes/No).
- 5. Generate a Contingency Table (crosstab) of 'Is_Biometric' vs 'Has_Mitigation'.
- 6. Calculate the percentage of 'Has_Mitigation' for both the Biometric and Non-Biometric groups.
- 7. Perform a Chi-square test of independence to determine if the association is statistically significant.
- 8. Print the contingency table, the calculated percentages, and the Chi-square test results (statistic and p-value).

### Deliverables
- 1. Contingency table of Biometric status vs. Mitigation status.
- 2. Calculated mitigation rates for both groups.
- 3. Chi-square test statistic and p-value.

---

## Analysis

The experiment successfully analyzed 1,757 AI use cases from the EO 13960
inventory.

1. **Classification**:
   - **Biometric Systems**: 184 systems (10.5%) were identified as biometric
based on keywords (e.g., 'face', 'recognition').
   - **Mitigation Controls**: 44 systems (2.5%) were identified as having
documented disparity mitigation measures based on positive action keywords
(e.g., 'test', 'monitor', 'audit') in the 'disparity_mitigation' field. Note
that this field is often sparse or populated with 'N/A' in the raw data.

2. **Findings**:
   - **Biometric Systems**: 10.87% (20/184) showed evidence of disparity
mitigation.
   - **Non-Biometric Systems**: Only 1.53% (24/1,573) showed evidence of
disparity mitigation.

3. **Statistical Significance**: The Chi-square test yielded a statistic of
55.14 and a p-value of 1.12e-13 (p < 0.05).

**Conclusion**: The null hypothesis is rejected. There is a strong,
statistically significant association where government AI systems involving
biometrics are far more likely (approx. 7x) to report specific disparity
mitigation efforts compared to the baseline of other AI applications.

---

## Review

The experiment was successfully executed. The code improved upon the previous
attempt by using keyword-based heuristics to parse the free-text
'disparity_mitigation' column rather than looking for exact matches. This
allowed for the identification of 44 systems with documented mitigation
strategies. The statistical analysis (Chi-square) was appropriate for the
categorical data, and the findings (p < 0.05) support the hypothesis.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy import stats
import sys
import os

# Define dataset path
file_path = '../astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = 'astalabs_discovery_all_data.csv'

print(f"Loading dataset from {file_path}...")

try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print("Error: Dataset not found.")
    sys.exit(1)

# Filter for EO 13960 Scored dataset
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored subset shape: {subset.shape}")

# Identify columns dynamically
use_case_col = next((c for c in subset.columns if 'use_case_name' in c), None)
purpose_col = next((c for c in subset.columns if 'purpose_benefits' in c), None)
mitigation_col = next((c for c in subset.columns if 'disparity_mitigation' in c), None)

print(f"Columns identified: Use Case='{use_case_col}', Purpose='{purpose_col}', Mitigation='{mitigation_col}'")

if not (use_case_col and purpose_col and mitigation_col):
    print("Critical columns missing. Aborting.")
    sys.exit(1)

# --- 1. Define Biometric Systems ---
# Keywords for biometrics
bio_keywords = ['face', 'facial', 'biometric', 'recognition', 'surveillance', 'gait', 'iris']

# Combine text for searching
subset['text_search'] = subset[use_case_col].fillna('').astype(str).str.lower() + " " + subset[purpose_col].fillna('').astype(str).str.lower()

# Apply boolean mask
subset['Is_Biometric'] = subset['text_search'].apply(lambda x: any(k in x for k in bio_keywords))

print(f"Biometric systems found: {subset['Is_Biometric'].sum()} out of {len(subset)}")

# --- 2. Define Mitigation Presence ---
# Positive keywords indicating some form of check/control exists
mitigation_keywords = ['test', 'eval', 'monitor', 'assess', 'audit', 'mitigat', 'review', 'human', 'valid', 'bias', 'fair', 'check', 'control', 'feedback']

# Function to check mitigation text
def check_mitigation(text):
    if pd.isna(text):
        return False
    text = str(text).lower()
    # Check for positive keywords
    has_positive = any(k in text for k in mitigation_keywords)
    return has_positive

subset['Has_Mitigation'] = subset[mitigation_col].apply(check_mitigation)

print(f"Systems with Mitigation found: {subset['Has_Mitigation'].sum()} out of {len(subset)}")

# --- 3. Statistical Analysis ---
contingency = pd.crosstab(subset['Is_Biometric'], subset['Has_Mitigation'])
print("\nContingency Table (Rows=Is_Biometric, Cols=Has_Mitigation):")
print(contingency)

# Calculate rates safely
try:
    # Biometric stats
    bio_total = contingency.loc[True].sum() if True in contingency.index else 0
    bio_mitigated = contingency.loc[True, True] if (True in contingency.index and True in contingency.columns) else 0
    bio_rate = bio_mitigated / bio_total if bio_total > 0 else 0.0

    # Non-Biometric stats
    non_bio_total = contingency.loc[False].sum() if False in contingency.index else 0
    non_bio_mitigated = contingency.loc[False, True] if (False in contingency.index and True in contingency.columns) else 0
    non_bio_rate = non_bio_mitigated / non_bio_total if non_bio_total > 0 else 0.0

    print(f"\nBiometric Mitigation Rate: {bio_rate:.2%}")
    print(f"Non-Biometric Mitigation Rate: {non_bio_rate:.2%}")

except Exception as e:
    print(f"Error calculating rates: {e}")

# Perform Chi-Square Test
if contingency.size >= 4:
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Result:")
    print(f"Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    if p < 0.05:
        print("Result: Statistically Significant (Reject Null Hypothesis)")
    else:
        print("Result: Not Statistically Significant (Fail to Reject Null Hypothesis)")
else:
    print("\nContingency table is too small for Chi-Square test (needs 2x2).")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
EO 13960 Scored subset shape: (1757, 196)
Columns identified: Use Case='2_use_case_name', Purpose='11_purpose_benefits', Mitigation='62_disparity_mitigation'
Biometric systems found: 184 out of 1757
Systems with Mitigation found: 44 out of 1757

Contingency Table (Rows=Is_Biometric, Cols=Has_Mitigation):
Has_Mitigation  False  True 
Is_Biometric                
False            1549     24
True              164     20

Biometric Mitigation Rate: 10.87%
Non-Biometric Mitigation Rate: 1.53%

Chi-Square Test Result:
Statistic: 55.1408
P-value: 1.1220e-13
Result: Statistically Significant (Reject Null Hypothesis)

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
