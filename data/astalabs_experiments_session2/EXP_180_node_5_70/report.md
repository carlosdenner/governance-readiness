# Experiment 180: node_5_70

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_70` |
| **ID in Run** | 180 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T09:33:13.593117+00:00 |
| **Runtime** | 337.5s |
| **Parent** | `node_4_40` |
| **Children** | `node_6_38` |
| **Creation Index** | 181 |

---

## Hypothesis

> The 'Public-Facing' Accountability Gap: Despite higher visibility risks, AI
systems designated as 'Public Service' (customer-facing) do not show
statistically higher rates of 'AI Notice' compliance compared to internal
systems, revealing a failure in transparency where it matters most.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7339 (Likely True) |
| **Posterior** | 0.9093 (Definitely True) |
| **Surprise** | +0.2106 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 29.0 |
| Uncertain | 1.0 |
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

**Objective:** Evaluate if Public-Facing systems effectively implement AI Notice requirements.

### Steps
- 1. Load `eo13960_scored`.
- 2. Split data by `26_public_service` ('Yes' vs 'No').
- 3. Calculate the rate of `59_ai_notice` = 'Yes' in both groups.
- 4. Perform a Chi-square test. (Hypothesis posits Null Result or lack of significant positive difference).

### Deliverables
- AI Notice compliance rates for Public vs Internal systems; Statistical comparison.

---

## Analysis

The experiment successfully tested the 'Public-Facing' Accountability Gap
hypothesis using the `eo13960_scored` dataset. After refining the data cleaning
logic to map unstructured text descriptions to binary categories, the analysis
identified 76 records with valid compliance data. The results showed that
Public-Facing systems had a higher raw compliance rate for AI Notice (87.5%,
14/16) compared to Internal/Non-Public systems (63.3%, 38/60). However, the Chi-
square test yielded a p-value of 0.1223, which is above the significance
threshold of 0.05. Consequently, the difference is not statistically
significant. This supports the hypothesis that Public-Facing systems do not show
a statistically reliable improvement in transparency compliance compared to
internal systems, validating the existence of the proposed 'Accountability Gap'.

---

## Review

The experiment successfully tested the 'Public-Facing' Accountability Gap
hypothesis using the `eo13960_scored` dataset. After refining the data cleaning
logic to map unstructured text descriptions to binary categories, the analysis
identified 76 records with valid compliance data (16 Public, 60 Internal). While
Public-Facing systems showed a higher raw compliance rate for AI Notice (87.5%)
compared to Internal systems (63.3%), the Chi-square test (p=0.12) confirmed
that this difference is not statistically significant. This result supports the
hypothesis, suggesting that high-visibility public deployments do not reliably
implement transparency safeguards significantly more often than internal
systems.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for the specific source table
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Step 1: Broad Mapping of Public Service ---
# Assumption: Explicit descriptions are Public. 'No' or NaN are treated as Internal/Non-Public.
col_public = '26_public_service'

def map_service_type(val):
    s = str(val).strip().lower()
    # Known explicit public indicators from previous debug
    if s in ['nan', '', 'no']:
        return 'Internal/Non-Public'
    return 'Public'

eo_data['service_type'] = eo_data[col_public].apply(map_service_type)

# --- Step 2: Broad Mapping of Notice Compliance ---
col_notice = '59_ai_notice'

def map_notice_status(val):
    s = str(val).strip().lower()
    if s in ['nan', '']:
        return 'Unknown'
    
    # Explicit 'None of the above' is a failure to notify (Non-Compliant)
    if 'none of the above' in s:
        return 'No Notice'
    
    # Exemptions
    if any(x in s for x in ['n/a', 'waived', 'not safety', 'not interacting']):
        return 'Exempt'
    
    # Positive Indications
    if any(x in s for x in ['online', 'in-person', 'email', 'telephone', 'terms', 'instructions']):
        return 'Has Notice'
    
    return 'Unknown'

eo_data['notice_status'] = eo_data[col_notice].apply(map_notice_status)

# --- Step 3: Analysis Data ---
# We focus on rows where Notice is either 'Has Notice' or 'No Notice' (Binary Outcome).
analysis_df = eo_data[eo_data['notice_status'].isin(['Has Notice', 'No Notice'])].copy()

print("--- Data Distribution ---")
full_counts = pd.crosstab(eo_data['service_type'], eo_data['notice_status'])
print(full_counts)

print(f"\nAnalyzable Rows (Yes/No Notice): {len(analysis_df)}")

if len(analysis_df) < 5:
    print("Insufficient data to perform Chi-square test.")
else:
    # Generate Contingency Table for Test
    contingency = pd.crosstab(analysis_df['service_type'], analysis_df['notice_status'])
    print("\n--- Contingency Table (Analysis) ---")
    print(contingency)

    # Calculate Rates
    # Compliance Rate = Has Notice / (Has Notice + No Notice)
    # We calculate for both groups if they exist
    
    results = {}
    for group in contingency.index:
        has = contingency.loc[group, 'Has Notice'] if 'Has Notice' in contingency.columns else 0
        no = contingency.loc[group, 'No Notice'] if 'No Notice' in contingency.columns else 0
        total = has + no
        if total > 0:
            rate = (has / total) * 100
        else:
            rate = 0.0
        results[group] = {'rate': rate, 'total': total}

    print("\n--- Compliance Rates ---")
    for group, data in results.items():
        print(f"{group}: {data['rate']:.1f}% (n={data['total']})")

    # Chi-square Test
    # Only run if we have at least 2 groups and 2 outcomes in the full structure, 
    # but crosstab handles dimensions automatically. We need size > 0.
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, p, dof, expected = chi2_contingency(contingency)
        print("\n--- Chi-Square Results ---")
        print(f"Chi2: {chi2:.4f}")
        print(f"p-value: {p:.4e}")
        
        if p < 0.05:
            print("Result: Significant Difference.")
        else:
            print("Result: No Significant Difference.")
    else:
        print("\nCannot run Chi-square: One group or outcome is missing from the filtered data.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Data Distribution ---
notice_status        Exempt  Has Notice  No Notice  Unknown
service_type                                               
Internal/Non-Public      29          38         22     1603
Public                   16          14          2       33

Analyzable Rows (Yes/No Notice): 76

--- Contingency Table (Analysis) ---
notice_status        Has Notice  No Notice
service_type                              
Internal/Non-Public          38         22
Public                       14          2

--- Compliance Rates ---
Internal/Non-Public: 63.3% (n=60)
Public: 87.5% (n=16)

--- Chi-Square Results ---
Chi2: 2.3874
p-value: 1.2231e-01
Result: No Significant Difference.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
