# Experiment 264: node_5_95

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_95` |
| **ID in Run** | 264 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:32:30.881334+00:00 |
| **Runtime** | 246.9s |
| **Parent** | `node_4_50` |
| **Children** | `node_6_81` |
| **Creation Index** | 265 |

---

## Hypothesis

> Autonomy-Intent Inversion: Incidents involving 'Intentional' harm (malicious
use) are strongly associated with lower levels of AI autonomy, whereas
'Unintentional' accidents are associated with higher autonomy.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7823 (Likely True) |
| **Posterior** | 0.4203 (Maybe False) |
| **Surprise** | -0.4343 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

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
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 56.0 |
| Definitely False | 4.0 |

---

## Experiment Plan

**Objective:** Investigate the relationship between system autonomy and the intentionality of harm.

### Steps
- 1. Load 'aiid_incidents'.
- 2. Clean 'Autonomy Level' (Low vs High/Medium) and 'Intentional Harm' (Yes/True vs No/False).
- 3. Create a contingency table.
- 4. Perform a Chi-square test to see if Low Autonomy correlates with Intentional Harm.

### Deliverables
- Contingency table; Chi-square statistics.

---

## Analysis

The experiment successfully analyzed 177 incidents from the AIID dataset where
both 'Autonomy Level' and 'Intentional Harm' data were available. The data
cleaning logic correctly mapped 'Autonomy1' to 'Low' (n=103) and
'Autonomy2'/'Autonomy3' to 'High/Medium' (n=74).

The contingency table revealed that 'Intentional' harm is rare across both
groups:
- **High/Medium Autonomy**: 3 Intentional (4.1%), 71 Unintentional.
- **Low Autonomy**: 5 Intentional (4.9%), 98 Unintentional.

A Chi-square test yielded a statistic of 0.0000 and a p-value of 1.0000 (likely
due to Yates' correction on the small sample size of intentional incidents).
Consequently, the null hypothesis could not be rejected. The analysis finds **no
significant association** between the level of AI autonomy and the
intentionality of harm; incidents in this dataset are overwhelmingly classified
as unintentional regardless of autonomy level.

---

## Review

Hypothesis: The 'Autonomy-Intent Inversion' hypothesis posited that incidents
involving 'Intentional' harm (malicious use) are associated with lower levels of
AI autonomy, while 'Unintentional' accidents are associated with higher
autonomy.

Experiment Results: The experiment successfully analyzed 177 fully labeled
incidents from the AIID dataset. The data cleaning process mapped 'Autonomy1' to
'Low' and 'Autonomy2/3' to 'High/Medium', and classified intent based on
descriptive fields. The resulting contingency table showed that intentional harm
is rare in the dataset (n=8 total) and distributed nearly evenly across autonomy
levels: 4.1% of High/Medium autonomy cases (3/74) and 4.9% of Low autonomy cases
(5/103) involved intentional harm.

Findings: A Chi-square test yielded a p-value of 1.0000, indicating absolutely
no statistically significant association between autonomy level and the
intentionality of harm in this dataset. Consequently, the hypothesis is not
supported; the data suggests that in the recorded incidents, the likelihood of
harm being intentional is independent of the system's autonomy level.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid)} rows")

# Define target columns
col_autonomy = 'Autonomy Level'
col_intent = 'Intentional Harm'

# Check for missing data in target columns
subset = aiid[[col_autonomy, col_intent]].dropna()
print(f"Rows with complete data: {len(subset)}")

# Mapping functions based on observed values
def clean_autonomy(val):
    s = str(val).strip()
    if s == 'Autonomy1':
        return 'Low'
    elif s in ['Autonomy2', 'Autonomy3']:
        return 'High/Medium'
    return None  # Exclude 'unclear' or others

def clean_intent(val):
    s = str(val).strip()
    if s.startswith('Yes'):
        return 'Intentional'
    elif s.startswith('No'):
        return 'Unintentional'
    return None  # Exclude 'unclear' or others

# Apply mapping
subset['Autonomy_Bin'] = subset[col_autonomy].apply(clean_autonomy)
subset['Intent_Bin'] = subset[col_intent].apply(clean_intent)

# Drop unmapped rows
final_df = subset.dropna(subset=['Autonomy_Bin', 'Intent_Bin'])
print(f"Final analysis set: {len(final_df)} rows")

# Generate Contingency Table
# Hypothesis: Intentional -> Low Autonomy; Unintentional -> High Autonomy
contingency_table = pd.crosstab(final_df['Autonomy_Bin'], final_df['Intent_Bin'])

print("\n--- Contingency Table ---")
print(contingency_table)

# Perform Chi-square test
if contingency_table.size > 0 and contingency_table.sum().sum() > 0:
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print("\n--- Chi-square Test Results ---")
    print(f"Chi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    print(f"Degrees of Freedom: {dof}")
    
    print("\n--- Expected Frequencies ---")
    print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))

    # Interpretation
    alpha = 0.05
    if p < alpha:
        print("\nResult: Significant association found (reject H0).")
        # Check directionality by comparing observed vs expected
        # Specifically checking if Intentional is higher in Low Autonomy than expected
        obs_low_intent = contingency_table.loc['Low', 'Intentional'] if 'Low' in contingency_table.index and 'Intentional' in contingency_table.columns else 0
        exp_low_intent = expected[contingency_table.index.get_loc('Low'), contingency_table.columns.get_loc('Intentional')] if 'Low' in contingency_table.index and 'Intentional' in contingency_table.columns else 0
        
        if obs_low_intent > exp_low_intent:
            print("Direction: Low Autonomy is associated with Intentional Harm.")
        else:
            print("Direction: Association exists but direction differs from hypothesis.")
            
    else:
        print("\nResult: No significant association found (fail to reject H0).")
else:
    print("Contingency table is empty or invalid. Cannot perform Chi-square test.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset...
AIID Incidents loaded: 1362 rows
Rows with complete data: 199
Final analysis set: 177 rows

--- Contingency Table ---
Intent_Bin    Intentional  Unintentional
Autonomy_Bin                            
High/Medium             3             71
Low                     5             98

--- Chi-square Test Results ---
Chi-square Statistic: 0.0000
P-value: 1.0000e+00
Degrees of Freedom: 1

--- Expected Frequencies ---
Intent_Bin    Intentional  Unintentional
Autonomy_Bin                            
High/Medium      3.344633      70.655367
Low              4.655367      98.344633

Result: No significant association found (fail to reject H0).

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
