# Experiment 253: node_6_60

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_60` |
| **ID in Run** | 253 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:00:06.283676+00:00 |
| **Runtime** | 208.4s |
| **Parent** | `node_5_28` |
| **Children** | None |
| **Creation Index** | 254 |

---

## Hypothesis

> The 'Legacy Governance Debt': AI systems in the 'Operation and Maintenance'
stage in the EO 13960 inventory are significantly less likely to have a
documented 'Impact Assessment' compared to systems in the
'Development/Acquisition' stage, suggesting that new governance requirements are
not being retroactively applied to older deployments.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.2555 (Likely False) |
| **Surprise** | -0.5934 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 29.0 |
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

**Objective:** Assess whether governance compliance (Impact Assessment) correlates with the lifecycle stage of the AI system.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'source_table' == 'eo13960_scored'.
- 2. Create a variable 'lifecycle_stage': Categorize '16_dev_stage' into 'Development' (containing 'Development', 'Acquisition', 'Pilot') vs 'Operation' (containing 'Operation', 'Maintenance'). Discard ambiguous rows.
- 3. Create a binary variable 'has_impact_assessment': Parse '52_impact_assessment' for affirmative responses (starts with 'Yes').
- 4. Create a contingency table of Lifecycle Stage vs Impact Assessment.
- 5. Perform a Chi-square test and calculate the Odds Ratio.

### Deliverables
- Contingency table, Chi-square statistics, and Odds Ratio comparing compliance in Operation vs Development stages.

---

## Analysis

The experiment successfully tested the 'Legacy Governance Debt' hypothesis using
the EO 13960 dataset (n=978). The analysis categorized systems into 'Operation'
(n=627) and 'Development' (n=351) stages. Contrary to the hypothesis that legacy
(operational) systems would lack governance documentation, the results show that
Operational systems are significantly **more** likely to have an Impact
Assessment (8.93%) compared to systems in Development (0.57%). The Chi-square
test (X2=26.72, p<0.001) and Odds Ratio (17.11) indicate a strong positive
correlation between deployment status and impact assessment compliance. This
suggests that Impact Assessments function as a 'Deployment Gate'—a requirement
fulfilled before a system goes live—rather than a new standard that older
systems have bypassed. The original hypothesis is rejected.

---

## Review

The experiment was successfully executed according to the plan. The code
correctly loaded the EO 13960 dataset, categorized the lifecycle stages, and
parsed the compliance data. The analysis (n=978) revealed a statistically
significant relationship (Chi2=26.72, p<0.001) that **contradicts** the original
'Legacy Governance Debt' hypothesis. Instead of legacy (operational) systems
lacking governance, they are significantly **more** likely (8.93%) to have an
Impact Assessment compared to systems in development (0.57%). The Odds Ratio of
17.11 suggests that Impact Assessments likely function as a 'deployment gate'—a
requirement that must be satisfied before a system moves from development to
operation—rather than a new standard that older systems have bypassed.

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

# Filter for EO 13960 scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Categorize Lifecycle Stage
def categorize_stage(stage):
    if pd.isna(stage):
        return None
    stage_lower = str(stage).lower()
    # Operation keywords
    if any(x in stage_lower for x in ['operation', 'maintenance', 'use', 'implemented']):
        return 'Operation'
    # Development keywords
    elif any(x in stage_lower for x in ['development', 'acquisition', 'pilot', 'planning', 'research']):
        return 'Development'
    return None

eo_data['lifecycle_group'] = eo_data['16_dev_stage'].apply(categorize_stage)
analysis_df = eo_data.dropna(subset=['lifecycle_group']).copy()

# Binarize Impact Assessment
def parse_impact_assessment(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip().lower()
    if val_str.startswith('yes'):
        return 1
    return 0

analysis_df['has_ia'] = analysis_df['52_impact_assessment'].apply(parse_impact_assessment)

# Contingency Table
contingency_table = pd.crosstab(analysis_df['lifecycle_group'], analysis_df['has_ia'])
contingency_table.columns = ['No Impact Assessment', 'Has Impact Assessment']

print("--- Contingency Table (Counts) ---")
print(contingency_table)

# Percentages
props = pd.crosstab(analysis_df['lifecycle_group'], analysis_df['has_ia'], normalize='index') * 100
props.columns = ['No IA (%)', 'Has IA (%)']
print("\n--- Contingency Table (Percentages) ---")
print(props)

# Statistical Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Odds Ratio Calculation
try:
    dev_yes = contingency_table.loc['Development', 'Has Impact Assessment']
    dev_no = contingency_table.loc['Development', 'No Impact Assessment']
    op_yes = contingency_table.loc['Operation', 'Has Impact Assessment']
    op_no = contingency_table.loc['Operation', 'No Impact Assessment']
    
    # Handle zeros with pseudocount if necessary
    if dev_yes == 0 or dev_no == 0 or op_yes == 0 or op_no == 0:
        print("\n(Note: Zero counts detected, adding 0.5 correction for Odds Ratio)")
        dev_yes += 0.5
        dev_no += 0.5
        op_yes += 0.5
        op_no += 0.5

    odds_dev = dev_yes / dev_no
    odds_op = op_yes / op_no
    
    # Calculate OR comparing Operation relative to Development
    odds_ratio_op_vs_dev = odds_op / odds_dev
    
    print(f"\nOdds (Development): {odds_dev:.4f}")
    print(f"Odds (Operation): {odds_op:.4f}")
    print(f"Odds Ratio (Operation / Development): {odds_ratio_op_vs_dev:.4f}")
    
    print("\nInterpretation:")
    if p < 0.05:
        if odds_ratio_op_vs_dev > 1:
            print(f"Significant Result: Systems in Operation are {odds_ratio_op_vs_dev:.2f} times MORE likely to have an Impact Assessment than those in Development.")
        else:
            print(f"Significant Result: Systems in Operation are {1/odds_ratio_op_vs_dev:.2f} times LESS likely to have an Impact Assessment than those in Development.")
    else:
        print("No statistically significant difference found between lifecycle stages.")

except Exception as e:
    print(f"Error calculating odds ratio: {e}")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Contingency Table (Counts) ---
                 No Impact Assessment  Has Impact Assessment
lifecycle_group                                             
Development                       349                      2
Operation                         571                     56

--- Contingency Table (Percentages) ---
                 No IA (%)  Has IA (%)
lifecycle_group                       
Development      99.430199    0.569801
Operation        91.068581    8.931419

Chi-square statistic: 26.7230
P-value: 2.3481e-07

Odds (Development): 0.0057
Odds (Operation): 0.0981
Odds Ratio (Operation / Development): 17.1138

Interpretation:
Significant Result: Systems in Operation are 17.11 times MORE likely to have an Impact Assessment than those in Development.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
