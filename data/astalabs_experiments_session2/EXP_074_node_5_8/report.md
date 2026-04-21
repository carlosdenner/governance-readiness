# Experiment 74: node_5_8

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_8` |
| **ID in Run** | 74 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T04:30:05.426192+00:00 |
| **Runtime** | 224.0s |
| **Parent** | `node_4_2` |
| **Children** | `node_6_61` |
| **Creation Index** | 75 |

---

## Hypothesis

> The 'Operational Rigor' Hypothesis: Operational AI systems ('Implemented') in
the federal inventory exhibit significantly higher rates of Impact Assessment
compliance compared to pre-operational systems ('Planned' or 'Pilot'),
reflecting a 'gatekeeper' governance model where controls are enforced primarily
at deployment.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6613 (Maybe True) |
| **Posterior** | 0.8846 (Likely True) |
| **Surprise** | +0.2680 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 25.0 |
| Uncertain | 0.0 |
| Maybe False | 5.0 |
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

**Objective:** Determine if governance controls like Impact Assessments are gated by development stage.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'eo13960_scored'.
- 2. Create a binary variable 'is_operational': Set to 1 if '16_dev_stage' contains 'Implemented' or 'Operational', else 0 (for 'Planned', 'Pilot', 'R&D').
- 3. Create a binary variable 'has_impact_assessment': Parse '52_impact_assessment' for affirmative responses (starts with 'Yes').
- 4. Generate a contingency table: Operational Status (Rows) vs Impact Assessment (Columns).
- 5. Perform a Chi-square test and calculate the Odds Ratio.

### Deliverables
- Contingency table, Chi-square statistic, p-value, and Odds Ratio comparing compliance rates of operational vs. pre-operational systems.

---

## Analysis

The experiment successfully tested the 'Operational Rigor' hypothesis after
correcting the data parsing logic. By identifying 920 'Operational' systems
(including those in 'Operation and Maintenance', 'In production', 'In mission',
and 'Implementation') and 837 'Pre-Operational' systems, the analysis revealed a
stark contrast in governance documentation. Only 2 pre-operational systems
(0.24%) possessed an Impact Assessment, compared to 59 operational systems
(6.4%). The statistical analysis confirms this disparity is highly significant
(Chi-Square = 48.03, p < 0.001). The Odds Ratio of 28.61 indicates that
operational systems are nearly 29 times more likely to have a documented impact
assessment than those in earlier development stages. This strongly supports the
hypothesis that impact assessments function as a 'gatekeeper' control, primarily
enforced or documented only when systems reach the implementation or operational
phase.

---

## Review

The experiment was successfully executed. The code corrected the previous
parsing error by accounting for the specific vocabulary used in the
'16_dev_stage' column (e.g., 'Operation and Maintenance', 'In production'),
resulting in a valid segmentation of 920 operational and 837 pre-operational
systems. The analysis confirmed the 'Operational Rigor' hypothesis with high
statistical significance: while overall compliance is low, operational systems
(6.4% compliance) are nearly 29 times more likely (Odds Ratio = 28.61) to have a
documented Impact Assessment than pre-operational systems (0.24% compliance).
The Chi-square statistic (48.03, p < 0.001) reinforces that impact assessments
function primarily as a deployment-phase 'gatekeeper' control rather than an
early-stage planning tool in this dataset.

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

# Filter for EO 13960 data
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Subset shape: {subset.shape}")

# --- Process '16_dev_stage' for Operational Status ---
# Based on previous exploration, the values are:
# 'Operation and Maintenance', 'Acquisition and/or Development', 'Initiated',
# 'Implementation and Assessment', 'Retired', 'Planned', 'In production', 'In mission'

def check_operational(val):
    if pd.isna(val):
        return 0
    s = str(val).lower()
    # Keywords to capture 'Operation', 'In production', 'In mission', 'Implementation'
    # The prompt asked for 'Implemented' or 'Operational', but the data uses variations.
    if any(keyword in s for keyword in ['operation', 'implement', 'production', 'mission']):
        return 1
    return 0

subset['is_operational'] = subset['16_dev_stage'].apply(check_operational)

# --- Process '52_impact_assessment' for Compliance ---
def check_impact(val):
    if pd.isna(val):
        return 0
    s = str(val).strip().lower()
    if s.startswith('yes'):
        return 1
    return 0

subset['has_impact_assessment'] = subset['52_impact_assessment'].apply(check_impact)

# Print counts for verification
print("\nOperational Status Distribution:")
print(subset['is_operational'].value_counts())
print("\nImpact Assessment Distribution:")
print(subset['has_impact_assessment'].value_counts())

# --- Contingency Table ---
# Ensure we have a 2x2 table even if some categories are missing using reindex
contingency = pd.crosstab(subset['is_operational'], subset['has_impact_assessment'])
# Reindex to ensure all possibilities [0, 1] are present
contingency = contingency.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

contingency.index = ['Pre-Operational', 'Operational']
contingency.columns = ['No Impact Assessment', 'Has Impact Assessment']

print("\n--- Contingency Table ---")
print(contingency)

# --- Statistical Analysis ---
chi2, p, dof, ex = chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Odds Ratio Calculation
a = contingency.loc['Pre-Operational', 'No Impact Assessment']
b = contingency.loc['Pre-Operational', 'Has Impact Assessment']
c = contingency.loc['Operational', 'No Impact Assessment']
d = contingency.loc['Operational', 'Has Impact Assessment']

if b * c > 0:
    odds_ratio = (d * a) / (c * b)
    print(f"Odds Ratio: {odds_ratio:.4f}")
else:
    print("Odds Ratio: Undefined (division by zero)")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Subset shape: (1757, 196)

Operational Status Distribution:
is_operational
1    920
0    837
Name: count, dtype: int64

Impact Assessment Distribution:
has_impact_assessment
0    1696
1      61
Name: count, dtype: int64

--- Contingency Table ---
                 No Impact Assessment  Has Impact Assessment
Pre-Operational                   835                      2
Operational                       861                     59

Chi-Square Statistic: 48.0260
P-value: 4.2061e-12
Odds Ratio: 28.6092

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
