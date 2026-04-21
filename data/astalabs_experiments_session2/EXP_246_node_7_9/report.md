# Experiment 246: node_7_9

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_9` |
| **ID in Run** | 246 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T12:38:00.225022+00:00 |
| **Runtime** | 280.5s |
| **Parent** | `node_6_3` |
| **Children** | None |
| **Creation Index** | 247 |

---

## Hypothesis

> The Commercial Compliance Gap: Federal AI systems procured from Commercial
sources (COTS) are significantly less likely to have completed 'Impact
Assessments' than systems developed Internally, reflecting a 'black box'
governance oversight where agencies assume vendors handle compliance.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7903 (Likely True) |
| **Posterior** | 0.3901 (Maybe False) |
| **Surprise** | -0.4803 |
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
| Maybe False | 44.0 |
| Definitely False | 16.0 |

---

## Experiment Plan

**Objective:** Determine if the source of AI acquisition (Commercial vs. Custom) impacts the likelihood of performing Impact Assessments.

### Steps
- 1. Load 'eo13960_scored'.
- 2. Clean '10_commercial_ai' to create a binary variable: 'Commercial' vs 'Internal/Custom'.
- 3. Clean '52_impact_assessment' to create a binary variable: 'Yes' (Completed) vs 'No' (Not Completed/NA).
- 4. Create a contingency table (Source vs Assessment).
- 5. Perform a Chi-square test (or Fisher's Exact Test if counts are low) to check for independence.

### Deliverables
- Contingency table of Source vs Impact Assessment; Statistical test results (Chi-square/Fisher p-value).

---

## Analysis

The experiment successfully tested the 'Commercial Compliance Gap' hypothesis
using the EO 13960 dataset.

**Methodology:**
The analysis categorized 920 AI systems based on their development method:
- **External/Commercial**: 481 systems ('Developed with contracting resources').
- **Internal/Government**: 439 systems ('Developed in-house').

**Findings:**
- **Internal Compliance**: 6.61% (29/439) had completed Impact Assessments.
- **External Compliance**: 4.37% (21/481) had completed Impact Assessments.

**Statistical Conclusion:**
While the Internal group had a slightly higher compliance rate, the Chi-square
test ($p \approx 0.177$) indicated that this difference is **not statistically
significant**.

The hypothesis is **not supported**. The data suggests that low compliance with
Impact Assessment requirements is a systemic issue across federal agencies,
regardless of whether the system is built in-house or procured from commercial
contractors. The 'Black Box' effect of commercial vendors does not appear to
significantly worsen the already low baseline of governance documentation.

---

## Review

The experiment successfully tested the 'Commercial Compliance Gap' hypothesis
using the EO 13960 dataset.

**Methodology:**
The analysis categorized 920 AI systems based on their development method
('22_dev_method'):
- **External/Commercial**: 481 systems identified as 'Developed with contracting
resources'.
- **Internal/Government**: 439 systems identified as 'Developed in-house'.

**Results:**
- **Internal Compliance**: 6.61% (29/439) had completed Impact Assessments.
- **External Compliance**: 4.37% (21/481) had completed Impact Assessments.

**Statistical Conclusion:**
The Chi-square test yielded a p-value of approximately 0.177, which is above the
standard alpha of 0.05. Consequently, the difference is **not statistically
significant**.

**Findings:**
The hypothesis that commercial/COTS systems represent a 'black box' of lower
governance compliance was **not supported**. Instead, the data reveals a
systemic lack of Impact Assessment documentation across the federal landscape
(overall compliance < 6%), regardless of whether the system is developed
internally or by external contractors.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Data Cleaning & Mapping ---

# Mapping '22_dev_method' to 'External' vs 'Internal'
def map_dev_method(val):
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).lower().strip()
    
    if 'contracting resources' in val_str and 'both' not in val_str:
        return 'External/Commercial'
    elif 'in-house' in val_str and 'both' not in val_str:
        return 'Internal/Government'
    else:
        return 'Other/Mixed'

eo_df['source_group'] = eo_df['22_dev_method'].apply(map_dev_method)

# Map '52_impact_assessment' to Binary (Yes vs No)
def map_impact(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    if 'yes' in val_str:
        return 1
    return 0

eo_df['has_impact_assessment'] = eo_df['52_impact_assessment'].apply(map_impact)

# Filter for analysis: Compare External vs Internal
analysis_df = eo_df[eo_df['source_group'].isin(['External/Commercial', 'Internal/Government'])].copy()

print(f"\nRows for analysis (External vs Internal): {len(analysis_df)}")
print("Group Counts:")
print(analysis_df['source_group'].value_counts())

if len(analysis_df) > 0:
    # Contingency Table
    contingency = pd.crosstab(analysis_df['source_group'], analysis_df['has_impact_assessment'])
    # Ensure columns are ordered 0 (No), 1 (Yes) if both exist, or handle missing
    if 0 not in contingency.columns: contingency[0] = 0
    if 1 not in contingency.columns: contingency[1] = 0
    contingency = contingency[[0, 1]]
    contingency.columns = ['No Assessment', 'Has Assessment']
    
    print("\nContingency Table:")
    print(contingency)
    
    # Rates
    rates = pd.crosstab(analysis_df['source_group'], analysis_df['has_impact_assessment'], normalize='index') * 100
    print("\nCompliance Rates (%):")
    print(rates)
    
    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)
    print("\n--- Statistical Test Results ---")
    print(f"Chi2: {chi2:.4f}, p-value: {p:.5f}")
    
    # Interpretation
    ext_rate = rates.loc['External/Commercial', 1] if 1 in rates.columns else 0
    int_rate = rates.loc['Internal/Government', 1] if 1 in rates.columns else 0
    
    print(f"\nCompare: External ({ext_rate:.2f}%) vs Internal ({int_rate:.2f}%)")
    
    if p < 0.05:
        print("Result: Significant Difference.")
        if ext_rate < int_rate:
            print("Direction: External < Internal (Supports Hypothesis: Commercial Gap)")
        else:
            print("Direction: External > Internal (Contradicts Hypothesis)")
    else:
        print("Result: No Significant Difference (Null Hypothesis retained).")
else:
    print("Insufficient data for comparison.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: 
Rows for analysis (External vs Internal): 920
Group Counts:
source_group
External/Commercial    481
Internal/Government    439
Name: count, dtype: int64

Contingency Table:
                     No Assessment  Has Assessment
source_group                                      
External/Commercial            460              21
Internal/Government            410              29

Compliance Rates (%):
has_impact_assessment          0         1
source_group                              
External/Commercial    95.634096  4.365904
Internal/Government    93.394077  6.605923

--- Statistical Test Results ---
Chi2: 1.8262, p-value: 0.17658

Compare: External (4.37%) vs Internal (6.61%)
Result: No Significant Difference (Null Hypothesis retained).

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
