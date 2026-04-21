# Experiment 135: node_6_8

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_8` |
| **ID in Run** | 135 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:25:08.930405+00:00 |
| **Runtime** | 274.4s |
| **Parent** | `node_5_22` |
| **Children** | None |
| **Creation Index** | 136 |

---

## Hypothesis

> Commercial Opacity: In the EO13960 dataset, AI systems procured from
'Commercial' sources are statistically less likely to report having undergone
'Independent Evaluation' compared to 'Government-Developed' (custom) systems,
due to vendor proprietary barriers.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.6868 (Maybe True) |
| **Surprise** | -0.0661 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | No |

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
| Maybe True | 38.0 |
| Uncertain | 22.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Compare independent evaluation rates between commercial and non-commercial AI systems.

### Steps
- 1. Filter for `source_table` = 'eo13960_scored'.
- 2. Categorize `10_commercial_ai` into 'Commercial' (e.g., 'Yes', 'Commercial', 'COTS') and 'Non-Commercial' (e.g., 'No', 'GOTS', 'Custom').
- 3. Convert `55_independent_eval` to binary (1 for affirmative evidence, 0 for negative/missing).
- 4. Create a contingency table.
- 5. Perform a Chi-square test and calculate the Odds Ratio.

### Deliverables
- Contingency table, Chi-square statistic, p-value, and Odds Ratio.

---

## Analysis

The experiment successfully executed using '37_custom_code' as a proxy for
commercial status after previous debugging revealed '10_commercial_ai' contained
unstructured text. The analysis compared 245 'Commercial/COTS' systems (No
custom code) against 722 'Custom/GOTS' systems (Yes custom code).

The results show a directional trend supporting the 'Commercial Opacity'
hypothesis: Commercial systems had a lower rate of reported Independent
Evaluation (3.67%) compared to Custom systems (5.96%). The Odds Ratio of 0.60
suggests Commercial systems are roughly 40% less likely to have independent
evaluations. However, the Chi-square test yielded a p-value of 0.2284, which is
above the standard 0.05 significance threshold. Therefore, while the data
suggests a potential gap in transparency for commercial systems, the difference
is not statistically significant in this sample size.

---

## Review

The experiment was successfully executed and faithfully implemented the analysis
plan after correcting for data quality issues identified in the initial
attempts. The programmer correctly pivoted to using '37_custom_code' as a
reliable proxy for the Commercial (COTS) vs. Custom (GOTS) distinction and
applied appropriate text parsing to identify affirmative independent
evaluations.

Summary of Findings:
1. **Hypothesis**: The hypothesis that commercial AI systems are less likely to
undergo independent evaluation ('Commercial Opacity') was tested.
2. **Results**: The analysis found a directional trend supporting the
hypothesis. Systems identified as Commercial/COTS (via lack of custom code) had
a lower independent evaluation rate (3.67%, n=245) compared to Custom/GOTS
systems (5.96%, n=722).
3. **Statistical Significance**: The observed difference yielded a Chi-square
statistic of 1.45 and a p-value of 0.228. Consequently, the result is not
statistically significant at the p<0.05 level.
4. **Conclusion**: While the Odds Ratio (0.60) suggests commercial systems are
40% less likely to report independent evaluation, the null hypothesis cannot be
rejected based on this sample. The 'Commercial Opacity' effect is observed
directionally but lacks statistical robustness in the current dataset.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import os

# Define file path
filename = 'astalabs_discovery_all_data.csv'
possible_paths = [filename, f'../{filename}']
filepath = next((p for p in possible_paths if os.path.exists(p)), None)

if not filepath:
    print(f"Error: {filename} not found in current or parent directory.")
    exit(1)

print(f"Loading dataset from {filepath}...")
df = pd.read_csv(filepath, low_memory=False)

# Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Total EO 13960 records: {len(df_eo)}")

# NOTE: Previous exploration showed '10_commercial_ai' contains use-case descriptions (e.g., 'Searching for information...'),
# not a binary Commercial/Custom flag. 
# Column '37_custom_code' contains 'Yes'/'No' values which is a better proxy for Commercial Opacity.
# 'No' Custom Code implies Commercial/COTS (Opaque).
# 'Yes' Custom Code implies Government/Custom (Transparent).

# 1. Define Commercial vs Custom based on '37_custom_code'
def categorize_source(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    if s == 'no':
        return 'Commercial (COTS)'  # No custom code -> Commercial
    elif s == 'yes':
        return 'Custom (GOTS)'      # Custom code -> Government/Custom
    return np.nan

df_eo['source_category'] = df_eo['37_custom_code'].apply(categorize_source)

# Filter to valid rows
df_valid = df_eo.dropna(subset=['source_category']).copy()
print(f"Records with valid source category (Commercial/Custom): {len(df_valid)}")
print(df_valid['source_category'].value_counts())

# 2. Define Independent Evaluation Status
# '55_independent_eval' contains 'Yes...', 'TRUE', 'Planned', 'NaN', etc.
# Hypothesis requires 'having undergone', so 'Planned' is treated as 0.
def check_evaluation(val):
    if pd.isna(val):
        return 0
    s = str(val).lower().strip()
    # affirmative keywords
    if 'yes' in s or 'true' in s:
        return 1
    return 0

df_valid['has_eval'] = df_valid['55_independent_eval'].apply(check_evaluation)

# 3. Create Contingency Table
contingency = pd.crosstab(df_valid['source_category'], df_valid['has_eval'])
# Ensure columns are 0 and 1
if 0 not in contingency.columns:
    contingency[0] = 0
if 1 not in contingency.columns:
    contingency[1] = 0
contingency = contingency[[0, 1]]
contingency.columns = ['No Eval', 'Has Eval']

print("\nContingency Table (Custom Code Status vs. Independent Eval):")
print(contingency)

# 4. Statistical Test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# 5. Odds Ratio
# Commercial (No Eval) = a, Commercial (Has Eval) = b
# Custom (No Eval) = c, Custom (Has Eval) = d
try:
    comm_row = contingency.loc['Commercial (COTS)']
    cust_row = contingency.loc['Custom (GOTS)']
    
    a = comm_row['No Eval']
    b = comm_row['Has Eval']
    c = cust_row['No Eval']
    d = cust_row['Has Eval']
    
    # Add smoothing if zeros exist
    if a*d == 0 or b*c == 0:
        print("\n(Using Haldane-Anscombe correction for zero cells)")
        odds_ratio = ((b + 0.5) * (c + 0.5)) / ((a + 0.5) * (d + 0.5))
    else:
        odds_ratio = (b * c) / (a * d)
        
    print(f"Odds Ratio (Commercial likelihood of Eval vs Custom): {odds_ratio:.4f}")
    
    # Interpret
    comm_rate = b / (a + b) * 100
    cust_rate = d / (c + d) * 100
    print(f"\nEvaluation Rate [Commercial/COTS]: {comm_rate:.2f}%")
    print(f"Evaluation Rate [Custom/GOTS]:     {cust_rate:.2f}%")
    
except KeyError as e:
    print(f"Error calculating odds ratio: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
Total EO 13960 records: 1757
Records with valid source category (Commercial/Custom): 967
source_category
Custom (GOTS)        722
Commercial (COTS)    245
Name: count, dtype: int64

Contingency Table (Custom Code Status vs. Independent Eval):
                   No Eval  Has Eval
source_category                     
Commercial (COTS)      236         9
Custom (GOTS)          679        43

Chi-square statistic: 1.4508
P-value: 2.2840e-01
Odds Ratio (Commercial likelihood of Eval vs Custom): 0.6022

Evaluation Rate [Commercial/COTS]: 3.67%
Evaluation Rate [Custom/GOTS]:     5.96%

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
