# Experiment 123: node_5_39

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_39` |
| **ID in Run** | 123 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:53:52.427338+00:00 |
| **Runtime** | 234.4s |
| **Parent** | `node_4_22` |
| **Children** | None |
| **Creation Index** | 124 |

---

## Hypothesis

> Transparency Begets Accountability: AI systems that provide a public 'AI Notice'
are significantly more likely to also offer a formal 'Appeal Process' for
affected individuals, suggesting a cluster of transparency-oriented controls.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.3462 (Maybe False) |
| **Surprise** | -0.4749 |
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
| Maybe False | 34.0 |
| Definitely False | 26.0 |

---

## Experiment Plan

**Objective:** Test the correlation between two distinct transparency/recourse mechanisms: AI Notice and Appeal Process, using robust string matching to handle descriptive survey responses.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for the `eo13960_scored` source table.
- 2. Extract columns `59_ai_notice` and `65_appeal_process`.
- 3. Clean `59_ai_notice` into a binary variable `has_notice` using keyword matching: Map values containing 'Online', 'In-person', or 'Yes' to 'Yes'; map values containing 'No', 'N/A', or 'None' to 'No'.
- 4. Clean `65_appeal_process` into a binary variable `has_appeal` using keyword matching: Map values starting with 'Yes' to 'Yes'; map values starting with 'No' or 'N/A' to 'No'.
- 5. Create a contingency table of `has_notice` vs. `has_appeal`.
- 6. Calculate the percentage of systems with an appeal process given the presence/absence of an AI notice.
- 7. Perform a Chi-Square test to assess statistical significance.

### Deliverables
- Contingency table of AI Notice vs. Appeal Process; Conditional probabilities (e.g., P(Appeal | Notice)); Chi-Square test results (statistic and p-value).

---

## Analysis

The experiment successfully analyzed 102 valid records from the EO 13960 dataset
where both 'AI Notice' and 'Appeal Process' fields contained usable data. The
analysis revealed that systems providing a public AI Notice had a 55.0% rate of
offering an Appeal Process (22/40), while systems without such a Notice had a
50.0% rate (31/62). The Chi-Square test resulted in a p-value of 0.77, which is
far above the threshold for statistical significance (p < 0.05). Consequently,
the hypothesis is rejected: there is no evidence in this dataset that
transparency (AI Notice) acts as a predictor for accountability (Appeal
Process). The high rate of data attrition (only 102 usable records out of 1,757)
suggests significant missingness or inconsistent reporting for these specific
governance controls.

---

## Review

The experiment was executed faithfully. The code implemented the robust string
matching logic requested in the plan to handle the descriptive text in the 'EO
13960' dataset. Although the strict filtering resulted in a high attrition rate
(leaving 102 valid records out of 1,757), the analysis correctly interpreted the
resulting contingency table. The statistical test (p=0.77) and conditional
probabilities (55% vs 50%) support the conclusion to reject the hypothesis.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    # Fallback for local testing if needed, though strictly we use the provided filename
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# Define columns
col_notice = '59_ai_notice'
col_appeal = '65_appeal_process'

print(f"Processing {len(df_eo)} records from EO 13960 subset.")

# 1. Clean '59_ai_notice'
# Logic: 'Online', 'In-person', 'Yes' -> Yes. 'N/A', 'No' -> No.
def clean_notice(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    
    # Positive indicators
    if any(x in s for x in ['online', 'in-person', 'yes', 'public']):
        return 'Yes'
    # Negative indicators
    if any(x in s for x in ['n/a', 'no', 'none', 'not applicable']):
        return 'No'
    return np.nan

# 2. Clean '65_appeal_process'
# Logic: Starts with 'Yes' -> Yes. Starts with 'No', 'N/A' -> No.
def clean_appeal(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    
    if s.startswith('yes'):
        return 'Yes'
    if s.startswith('no') or s.startswith('n/a') or 'not applicable' in s:
        return 'No'
    return np.nan

# Apply cleaning
df_eo['Notice_Clean'] = df_eo[col_notice].apply(clean_notice)
df_eo['Appeal_Clean'] = df_eo[col_appeal].apply(clean_appeal)

# Drop rows where either value is NaN to ensure valid comparison
df_analysis = df_eo.dropna(subset=['Notice_Clean', 'Appeal_Clean'])

print(f"Valid records after cleaning: {len(df_analysis)}")

if len(df_analysis) > 0:
    # Create Contingency Table
    contingency = pd.crosstab(df_analysis['Notice_Clean'], df_analysis['Appeal_Clean'])
    
    # Reindex to ensure consistent order (Yes/No)
    desired_index = [x for x in ['Yes', 'No'] if x in contingency.index]
    desired_cols = [x for x in ['Yes', 'No'] if x in contingency.columns]
    contingency = contingency.loc[desired_index, desired_cols]
    
    print("\nContingency Table (Rows: Notice, Cols: Appeal):")
    print(contingency)
    
    # Chi-Square Test
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Conditional Probabilities
    # P(Appeal=Yes | Notice=Yes)
    if 'Yes' in contingency.index and 'Yes' in contingency.columns:
        n_notice_yes = contingency.loc['Yes'].sum()
        n_both_yes = contingency.loc['Yes', 'Yes']
        if n_notice_yes > 0:
            print(f"Rate of Appeal Process when Notice is provided: {n_both_yes/n_notice_yes:.1%} ({n_both_yes}/{n_notice_yes})")
    
    # P(Appeal=Yes | Notice=No)
    if 'No' in contingency.index and 'Yes' in contingency.columns:
        n_notice_no = contingency.loc['No'].sum()
        n_appeal_yes_notice_no = contingency.loc['No', 'Yes']
        if n_notice_no > 0:
            print(f"Rate of Appeal Process when Notice is NOT provided: {n_appeal_yes_notice_no/n_notice_no:.1%} ({n_appeal_yes_notice_no}/{n_notice_no})")
            
else:
    print("No valid records found after cleaning. Check value patterns.")
    print("Sample Notice values:", df_eo[col_notice].unique()[:5])
    print("Sample Appeal values:", df_eo[col_appeal].unique()[:5])
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Processing 1757 records from EO 13960 subset.
Valid records after cleaning: 102

Contingency Table (Rows: Notice, Cols: Appeal):
Appeal_Clean  Yes  No
Notice_Clean         
Yes            22  18
No             31  31

Chi-Square Statistic: 0.0844
P-value: 7.7143e-01
Rate of Appeal Process when Notice is provided: 55.0% (22/40)
Rate of Appeal Process when Notice is NOT provided: 50.0% (31/62)

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
