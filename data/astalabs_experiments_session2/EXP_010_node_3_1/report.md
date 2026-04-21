# Experiment 10: node_3_1

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_1` |
| **ID in Run** | 10 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T01:24:23.552287+00:00 |
| **Runtime** | 325.2s |
| **Parent** | `node_2_2` |
| **Children** | `node_4_2`, `node_4_33` |
| **Creation Index** | 11 |

---

## Hypothesis

> Vendor Transparency Gap: AI systems identified as specific 'Commercial'
applications in the EO 13960 inventory are significantly less likely to grant
'Code Access' for governance review compared to systems classified as 'None of
the above' (proxy for Custom/In-House), creating a 'Black Box' governance
barrier.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7823 (Likely True) |
| **Posterior** | 0.9258 (Definitely True) |
| **Surprise** | +0.1723 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 5.0 |
| Maybe True | 25.0 |
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

**Objective:** Compare the rate of source code accessibility between Commercial and Custom AI systems by correctly parsing verbose survey responses.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for the subset where 'source_table' is 'eo13960_scored'.
- 2. Create a derived variable 'is_commercial': Mark rows as 0 (Custom) if the column '10_commercial_ai' contains the string 'None of the above' (or is null); mark as 1 (Commercial) if it contains any other specific text description.
- 3. Create a derived variable 'has_code_access': Parse column '38_code_access'. Convert text to lowercase and mark as 1 if the string starts with 'yes' (capturing verbose responses like 'Yes – agency has access...'), otherwise 0.
- 4. Generate a contingency table comparing 'is_commercial' (rows) vs 'has_code_access' (columns).
- 5. Perform a Chi-square test of independence on the contingency table.
- 6. Calculate the Odds Ratio to quantify the likelihood of code access for Custom vs Commercial systems.

### Deliverables
- 1. Contingency table showing counts of Code Access (Yes/No) for Commercial vs Custom systems.
- 2. Chi-square test statistic and p-value.
- 3. Odds Ratio (OR) with an interpretation of the transparency gap.

---

## Analysis

The experiment successfully tested the 'Vendor Transparency Gap' hypothesis
using the EO 13960 dataset. By correctly parsing verbose survey responses, the
analysis classified 1,404 systems as Custom/Other and 353 as Commercial. The
results reveal a striking disparity in governance transparency: only 10.2% of
Commercial systems (36/353) grant code access, compared to 40.3% of Custom
systems (566/1,404).

Statistical analysis confirms this difference is highly significant (Chi-Square
= 112.25, p < 0.001). The Odds Ratio of 5.95 indicates that Custom systems are
nearly six times more likely to provide code access than their Commercial
counterparts. This strongly supports the hypothesis that commercial AI
procurement in the federal government introduces a significant 'Black Box'
barrier, where proprietary constraints severely limit the code-level visibility
required for deep governance and independent auditing.

---

## Review

The experiment was successfully executed. The code correctly loaded the EO 13960
dataset, implemented the logic to parse verbose survey responses for both
'Commercial' status and 'Code Access', and generated the required contingency
table and statistics. The results are statistically significant and directly
address the hypothesis.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
filename = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(filename, low_memory=False)
    print(f"Successfully loaded {filename}")
except FileNotFoundError:
    # Fallback if file is in parent directory
    df = pd.read_csv(f'../{filename}', low_memory=False)
    print(f"Successfully loaded ../{filename}")

# Filter for EO 13960 scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored Data Shape: {eo_data.shape}")

# --- 1. Define Commercial vs Custom ---
# Logic: 0 (Custom) if '10_commercial_ai' is 'None of the above.' or null.
#        1 (Commercial) otherwise.

def classify_commercial(val):
    s = str(val).strip()
    if pd.isna(val) or s.lower() == 'nan' or s == '':
        return 0
    if 'none of the above' in s.lower():
        return 0
    return 1

eo_data['is_commercial'] = eo_data['10_commercial_ai'].apply(classify_commercial)

print("\n--- Commercial Classification ---")
print(eo_data['is_commercial'].value_counts())
# Print a check to ensure we caught the specific commercial cases
print("Sample of Commercial descriptions:")
print(eo_data[eo_data['is_commercial'] == 1]['10_commercial_ai'].head(3).tolist())

# --- 2. Define Code Access (Target) ---
# Logic: 1 (Yes) if '38_code_access' starts with 'yes' (case insensitive).
#        0 (No) otherwise.

def classify_code_access(val):
    s = str(val).strip().lower()
    if s.startswith('yes'):
        return 1
    return 0

eo_data['has_code_access'] = eo_data['38_code_access'].apply(classify_code_access)

print("\n--- Code Access Classification ---")
print(eo_data['has_code_access'].value_counts())

# --- 3. Statistical Analysis ---
# Create contingency table
#              No Access (0)   Has Access (1)
# Custom (0)      n00             n01
# Commercial (1)  n10             n11
contingency_table = pd.crosstab(eo_data['is_commercial'], eo_data['has_code_access'])
contingency_table.index = ['Custom', 'Commercial']
contingency_table.columns = ['No Access', 'Has Access']

print("\n--- Contingency Table ---")
print(contingency_table)

# Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Odds Ratio Calculation
# OR = (Odds of Access for Custom) / (Odds of Access for Commercial)
#    = (n01 / n00) / (n11 / n10)

try:
    n00 = contingency_table.loc['Custom', 'No Access']
    n01 = contingency_table.loc['Custom', 'Has Access']
    n10 = contingency_table.loc['Commercial', 'No Access']
    n11 = contingency_table.loc['Commercial', 'Has Access']
    
    odds_custom = n01 / n00 if n00 != 0 else np.inf
    odds_commercial = n11 / n10 if n10 != 0 else np.inf
    
    print(f"\nOdds of Access (Custom): {odds_custom:.4f}")
    print(f"Odds of Access (Commercial): {odds_commercial:.4f}")

    if odds_commercial == 0:
        print("Odds Ratio undefined (Commercial odds is 0).")
    else:
        or_val = odds_custom / odds_commercial
        print(f"Odds Ratio (Custom vs Commercial): {or_val:.4f}")
        
        if p < 0.05:
            print("\nResult is statistically significant.")
            if or_val > 1:
                print(f"Custom systems are {or_val:.2f} times more likely to grant code access than Commercial systems.")
            else:
                print(f"Commercial systems are {1/or_val:.2f} times more likely to grant code access than Custom systems.")
        else:
            print("\nResult is not statistically significant.")

except Exception as e:
    print(f"Error calculating stats: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Successfully loaded astalabs_discovery_all_data.csv
EO 13960 Scored Data Shape: (1757, 196)

--- Commercial Classification ---
is_commercial
0    1404
1     353
Name: count, dtype: int64
Sample of Commercial descriptions:
['Improving the quality of written communications using AI tools.', 'Creating visually appealing presentations using AI-driven design suggestions.', 'Improving the quality of written communications using AI tools.']

--- Code Access Classification ---
has_code_access
0    1155
1     602
Name: count, dtype: int64

--- Contingency Table ---
            No Access  Has Access
Custom            838         566
Commercial        317          36

Chi-square Statistic: 112.2473
P-value: 3.1541e-26

Odds of Access (Custom): 0.6754
Odds of Access (Commercial): 0.1136
Odds Ratio (Custom vs Commercial): 5.9474

Result is statistically significant.
Custom systems are 5.95 times more likely to grant code access than Commercial systems.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
