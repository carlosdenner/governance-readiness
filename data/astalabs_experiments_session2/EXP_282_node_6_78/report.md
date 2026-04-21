# Experiment 282: node_6_78

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_78` |
| **ID in Run** | 282 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T14:30:45.780505+00:00 |
| **Runtime** | 184.7s |
| **Parent** | `node_5_46` |
| **Children** | None |
| **Creation Index** | 283 |

---

## Hypothesis

> The 'Risk-Governance' Decoupling: 'High-Stakes' Use Cases (Law Enforcement,
National Security) do not have significantly higher rates of documented 'Impact
Assessments' compared to 'Low-Stakes' (Administrative, Operations) cases,
indicating a failure to scale governance with risk.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7742 (Likely True) |
| **Posterior** | 0.9176 (Definitely True) |
| **Surprise** | +0.1721 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 4.0 |
| Maybe True | 26.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 58.0 |
| Maybe True | 2.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Assess if governance documentation scales with the sensitivity of the AI use case domain.

### Steps
- 1. Load 'eo13960_scored'.
- 2. Create 'Risk_Level': Search `8_topic_area` or `2_use_case_name` for ['Enforcement', 'Security', 'Surveillance', 'Justice', 'Health'] -> 'High Stakes'; ['Admin', 'Operations', 'Management', 'Logistics', 'Finance'] -> 'Low Stakes'.
- 3. Create 'Has_Impact_Assess': Map `52_impact_assessment` to binary (Yes=1, No=0).
- 4. Compare proportions via Chi-square test.

### Deliverables
- Proportion of Impact Assessments in High vs Low Stakes groups; Statistical test results.

---

## Analysis

The experiment successfully tested the 'Risk-Governance Decoupling' hypothesis
using the EO 13960 dataset. By keyword-categorizing use cases, the analysis
identified 367 'High Stakes' systems (e.g., Enforcement, Health) and 72 'Low
Stakes' systems (e.g., Admin, Logistics).

The results showed a generally low rate of documented governance across the
board. High-stakes systems had a 7.08% rate of documented Impact Assessments
(26/367), while Low-stakes systems had a 1.39% rate (1/72). Although the rate
for high-stakes systems was nominally higher, the Chi-square test returned a
p-value of 0.1162, which is above the standard significance threshold of 0.05.

Consequently, the difference is not statistically significant. This supports the
hypothesis that governance documentation is 'decoupled' from risk, as there is
no statistical evidence that high-risk federal AI systems are subject to
significantly more rigorous impact assessment documentation than low-risk
administrative systems.

---

## Review

The experiment was successfully executed and faithfully implemented the analysis
plan. The code correctly loaded the dataset, applied the keyword-based
categorization to segment 'High Stakes' (n=367) and 'Low Stakes' (n=72) use
cases, and performed the Chi-square test.

The analysis supports the 'Risk-Governance Decoupling' hypothesis. While 'High
Stakes' systems (e.g., Law Enforcement, Health) had a higher nominal rate of
documented Impact Assessments (7.1%) compared to 'Low Stakes' systems (1.4%),
the difference was not statistically significant (p = 0.116). This suggests that
widely deployed federal AI systems currently lack a consistent scaling of
governance documentation relative to their risk profile.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset (trying current directory based on previous success)
filename = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(filename, low_memory=False)
except FileNotFoundError:
    # Fallback to parent directory if current fails, though previous error suggests parent was wrong
    df = pd.read_csv('../' + filename, low_memory=False)

# Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded {len(df_eo)} rows from EO 13960 source.")

# Construct Search Text
# Concatenate Topic Area and Use Case Name
df_eo['search_text'] = (df_eo['8_topic_area'].fillna('') + ' ' + df_eo['2_use_case_name'].fillna('')).str.lower()

# Define Keywords
high_stakes_kw = ['enforcement', 'security', 'surveillance', 'justice', 'health']
low_stakes_kw = ['admin', 'operations', 'management', 'logistics', 'finance']

# Vectorized categorization
def get_risk_level(text):
    if any(kw in text for kw in high_stakes_kw):
        return 'High Stakes'
    elif any(kw in text for kw in low_stakes_kw):
        return 'Low Stakes'
    return 'Other'

df_eo['risk_level'] = df_eo['search_text'].apply(get_risk_level)

# Categorize Impact Assessment
# Looking for explicit 'Yes'
df_eo['has_impact_assess'] = df_eo['52_impact_assessment'].fillna('No').astype(str).str.strip().str.lower() == 'yes'
df_eo['has_impact_assess_int'] = df_eo['has_impact_assess'].astype(int)

# Filter for analysis
df_analysis = df_eo[df_eo['risk_level'] != 'Other'].copy()

# Generate Contingency Table
contingency = pd.crosstab(df_analysis['risk_level'], df_analysis['has_impact_assess'])

print("\n--- Contingency Table (Risk Level vs Impact Assessment) ---")
print(contingency)

# Calculate Proportions
summary = df_analysis.groupby('risk_level')['has_impact_assess_int'].agg(['count', 'sum', 'mean'])
summary.columns = ['Total Cases', 'With Assessment', 'Proportion']
print("\n--- Proportions ---")
print(summary)

# Statistical Test
if contingency.size > 0:
    chi2, p, dof, expected = chi2_contingency(contingency)
    print("\n--- Chi-Square Test Results ---")
    print(f"Chi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    if p < 0.05:
        print("Conclusion: Statistically significant difference exists.")
    else:
        print("Conclusion: No statistically significant difference found (Governance Decoupling supported).")
else:
    print("Insufficient data for Chi-square test.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded 1757 rows from EO 13960 source.

--- Contingency Table (Risk Level vs Impact Assessment) ---
has_impact_assess  False  True 
risk_level                     
High Stakes          341     26
Low Stakes            71      1

--- Proportions ---
             Total Cases  With Assessment  Proportion
risk_level                                           
High Stakes          367               26    0.070845
Low Stakes            72                1    0.013889

--- Chi-Square Test Results ---
Chi-square Statistic: 2.4680
P-value: 1.1618e-01
Conclusion: No statistically significant difference found (Governance Decoupling supported).

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
