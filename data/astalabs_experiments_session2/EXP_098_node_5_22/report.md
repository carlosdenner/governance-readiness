# Experiment 98: node_5_22

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_22` |
| **ID in Run** | 98 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:45:34.517147+00:00 |
| **Runtime** | 231.8s |
| **Parent** | `node_4_27` |
| **Children** | `node_6_8`, `node_6_51` |
| **Creation Index** | 99 |

---

## Hypothesis

> Legacy Governance Debt: Operational AI systems (those in 'Operation and
Maintenance') are significantly less likely to report compliance with newer
'Disparity Mitigation' requirements compared to systems currently in
'Development' or 'Acquisition' phases, indicating a governance gap in legacy
systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2527 (Likely False) |
| **Surprise** | -0.5870 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 30.0 |
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

**Objective:** Assess whether development stage correlates with bias mitigation compliance.

### Steps
- 1. Filter dataset for 'eo13960_scored'.
- 2. Group '16_dev_stage' into 'Operational' (e.g., 'Use', 'Operation', 'Maintenance') and 'Pre-Operational' (e.g., 'Development', 'Acquisition', 'Planning').
- 3. Convert '62_disparity_mitigation' to binary (Yes=1, No=0).
- 4. Create a contingency table comparing Development Stage vs. Disparity Mitigation.
- 5. Run a Chi-square test and calculate the compliance percentage for each group.

### Deliverables
- Contingency table, p-value, and compliance percentages per stage.

---

## Analysis

The experiment successfully executed and addressed the previous parsing error by
implementing heuristic text analysis to identify substantive disparity
mitigation descriptions. Out of 1620 analyzed AI systems, 62 were identified as
having affirmative mitigation measures.

Contrary to the hypothesis that legacy (Operational) systems would lag in
governance, the analysis revealed that Operational systems were significantly
*more* likely to report disparity mitigation (6.05%) compared to Pre-Operational
systems (2.36%). The Chi-square test (statistic=13.36, p<0.001) confirms this
difference is statistically significant. The Odds Ratio of 2.66 indicates that
an operational system has over twice the odds of having reported mitigation
measures compared to one in development. This suggests that systems in active
use may be subject to stricter scrutiny or have had more time to implement
controls than those still in early planning/acquisition phases.

---

## Review

The experiment was successfully executed and the previous data parsing issues
were resolved. The code correctly filtered the 'eo13960_scored' dataset,
categorized development stages, and applied heuristic text analysis to identify
affirmative disparity mitigation efforts (overcoming the lack of standardized
'Yes' values). The findings refute the 'Legacy Governance Debt' hypothesis:
Operational systems were significantly more likely to report disparity
mitigation compliance (6.05%) compared to Pre-Operational systems (2.36%) (Chi-
square p < 0.001, OR = 2.66). This suggests that active deployment status
correlates with higher documented governance controls, likely due to operational
requirements, whereas pre-operational systems (often in early planning) may not
yet have reached the mitigation implementation stage.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import numpy as np
import sys

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except:
        print("Error: Dataset not found.")
        sys.exit(1)

# Filter for relevant source table
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Subset size: {len(subset)}")

# Define columns
stage_col = '16_dev_stage'
mitigation_col = '62_disparity_mitigation'

# -- Data Cleaning & Categorization --

# 1. Categorize Development Stage
def categorize_stage(val):
    if pd.isna(val):
        return np.nan
    val = str(val).lower()
    # Operational keywords
    if any(x in val for x in ['operation', 'maintenance', 'use', 'production', 'deployed', 'mission']):
        return 'Operational'
    # Pre-Operational keywords
    elif any(x in val for x in ['development', 'acquisition', 'plan', 'design', 'pilot', 'test', 'initiat', 'implement']):
        return 'Pre-Operational'
    return 'Other'

subset['stage_category'] = subset[stage_col].apply(categorize_stage)

# 2. Categorize Disparity Mitigation
# Logic: Treat explicit "N/A", "No", "None", or Null as 0. Treat substantive descriptions as 1.
def categorize_mitigation(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip().lower()
    
    # Check for negatives
    if val_str.startswith(('n/a', 'no ', 'none', 'not applicable', 'not safety')):
        return 0
    if val_str == 'no':
        return 0
    
    # Check for positives (heuristics based on text analysis)
    # If it's not N/A and has some length, it's likely a description of a control or process
    if len(val_str) > 3:
        return 1
        
    return 0

subset['mitigation_binary'] = subset[mitigation_col].apply(categorize_mitigation)

# Filter analysis set
analysis_df = subset[subset['stage_category'].isin(['Operational', 'Pre-Operational'])].copy()
print(f"Analysis set size: {len(analysis_df)}")
print("Stage distribution:\n", analysis_df['stage_category'].value_counts())
print("Mitigation distribution:\n", analysis_df['mitigation_binary'].value_counts())

# -- Analysis --

# Contingency Table
contingency = pd.crosstab(analysis_df['stage_category'], analysis_df['mitigation_binary'])
print("\nContingency Table (Count):")
print(contingency)

# Check if we have both 0 and 1 columns
if 1 not in contingency.columns:
    print("\nError: No positive mitigation cases found in the filtered dataset. Cannot perform Chi-square test.")
    # Debug print to see what text triggered this if any
    print("Sample of mitigation text that mapped to 0:")
    print(subset[subset['mitigation_binary']==0][mitigation_col].head(5))
    sys.exit(0)

if 0 not in contingency.columns:
    # Unlikely given the data, but possible
    contingency[0] = 0

# Reorder columns for consistency [0, 1]
contingency = contingency[[0, 1]]
contingency.columns = ['No Mitigation', 'Has Mitigation']

# Calculate percentages
props = contingency.div(contingency.sum(axis=1), axis=0) * 100
print("\nContingency Table (Percentages):")
print(props)

# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Odds Ratio
try:
    # OR = (Op_Yes / Op_No) / (Pre_Yes / Pre_No)
    op_yes = contingency.loc['Operational', 'Has Mitigation']
    op_no = contingency.loc['Operational', 'No Mitigation']
    pre_yes = contingency.loc['Pre-Operational', 'Has Mitigation']
    pre_no = contingency.loc['Pre-Operational', 'No Mitigation']
    
    # Add small epsilon if 0 to avoid division by zero error
    if op_no == 0: op_no = 0.5
    if pre_no == 0: pre_no = 0.5
    if pre_yes == 0: pre_yes = 0.5
    
    odds_op = op_yes / op_no
    odds_pre = pre_yes / pre_no
    odds_ratio = odds_op / odds_pre
    
    print(f"\nOdds (Operational): {odds_op:.4f}")
    print(f"Odds (Pre-Operational): {odds_pre:.4f}")
    print(f"Odds Ratio (Operational / Pre-Operational): {odds_ratio:.4f}")
except Exception as e:
    print(f"\nError calculating OR: {e}")

# Interpret
if p < 0.05:
    print("\nConclusion: Statistically significant difference.")
    if props.loc['Operational', 'Has Mitigation'] < props.loc['Pre-Operational', 'Has Mitigation']:
        print("Direction: Operational systems are LESS likely to report mitigation (Supporting Hypothesis).")
    else:
        print("Direction: Operational systems are MORE likely to report mitigation (Refuting Hypothesis).")
else:
    print("\nConclusion: No statistically significant difference.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Subset size: 1757
Analysis set size: 1620
Stage distribution:
 stage_category
Pre-Operational    975
Operational        645
Name: count, dtype: int64
Mitigation distribution:
 mitigation_binary
0    1558
1      62
Name: count, dtype: int64

Contingency Table (Count):
mitigation_binary    0   1
stage_category            
Operational        606  39
Pre-Operational    952  23

Contingency Table (Percentages):
                 No Mitigation  Has Mitigation
stage_category                                
Operational          93.953488        6.046512
Pre-Operational      97.641026        2.358974

Chi-square Statistic: 13.3571
P-value: 2.5745e-04

Odds (Operational): 0.0644
Odds (Pre-Operational): 0.0242
Odds Ratio (Operational / Pre-Operational): 2.6638

Conclusion: Statistically significant difference.
Direction: Operational systems are MORE likely to report mitigation (Refuting Hypothesis).

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
