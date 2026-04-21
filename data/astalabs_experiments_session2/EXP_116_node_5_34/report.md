# Experiment 116: node_5_34

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_34` |
| **ID in Run** | 116 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:33:11.557181+00:00 |
| **Runtime** | 242.0s |
| **Parent** | `node_4_32` |
| **Children** | `node_6_13`, `node_6_65` |
| **Creation Index** | 117 |

---

## Hypothesis

> The 'High-Stakes Transparency' Gap: AI systems classified as 'Rights-Impacting'
or 'Safety-Impacting' do not exhibit statistically higher rates of 'Public
Notice' compared to low-impact administrative systems.

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

**Objective:** Quantify the transparency gap by correctly parsing categorical notice data in the EO 13960 inventory.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for rows where 'source_table' is 'eo13960_scored'.
- 2. Create a 'High_Impact' binary variable based on column '17_impact_type': assign 1 if the text contains 'rights' or 'safety' (case-insensitive), else 0.
- 3. Create a 'Has_Notice' binary variable based on column '59_ai_notice'. Parse the categorical text: Map to 0 if the value is NaN, contains 'None of the above', or contains 'N/A' (indicating no interaction). Map to 1 if the value indicates a notice method (e.g., 'Online', 'Email', 'In-person', 'Other').
- 4. Calculate the percentage of 'Has_Notice' for High Impact vs. Low Impact systems.
- 5. Perform a Chi-square test of independence to determine if the difference is statistically significant.

### Deliverables
- 1. Contingency table of Impact Status vs. Public Notice.
- 2. Calculated Notice Rates for High Impact and Low Impact groups.
- 3. Chi-square statistic and p-value.

---

## Analysis

The experiment successfully re-evaluated the 'High-Stakes Transparency' Gap
using corrected parsing logic for the '59_ai_notice' column. By properly
categorizing descriptive responses (e.g., 'Email', 'In-person', 'Online') as
valid notice indicators versus explicit negatives (e.g., 'None of the above',
'N/A'), the analysis revealed a distinct trend. Contrary to the hypothesis that
high-stakes systems would not show higher transparency, the data indicates that
**High Impact systems are significantly more likely** to provide public notice
(16.88%) compared to Low Impact systems (3.63%). The Chi-square test (p < 0.001)
confirms this difference is statistically significant. While this refutes the
hypothesis of 'no difference,' the finding is sobering: even among high-stakes
'Rights' and 'Safety' impacting systems, over 83% fail to provide public notice.

---

## Review

The experiment successfully re-evaluated the 'High-Stakes Transparency' Gap
hypothesis with corrected data parsing logic.

**Hypothesis Status:** Refuted.

**Execution Summary:**
1. **Data Parsing:** The code correctly identified 'High Impact' systems (n=77)
vs. 'Low Impact' systems (n=1,680). Crucially, it fixed the previous error by
correctly mapping descriptive notice methods (e.g., 'Email', 'Online', 'In-
person') to a positive 'Has Notice' status, while treating 'None of the above',
'N/A', and missing values as negative.

**Findings:**
- **Rates:** High Impact systems demonstrated a **16.88%** notice rate (13/77),
compared to a **3.63%** rate (61/1,680) for Low Impact systems.
- **Significance:** The Chi-square test yielded a p-value of 7.82e-08,
confirming the difference is statistically significant.

**Interpretation:**
Contrary to the hypothesis that high-stakes systems would fail to outperform
administrative ones, the data shows they are significantly *more* likely to
provide public notice. However, the finding is nuanced: while the relative
performance is better, the absolute compliance is critically low, with over 83%
of Rights- and Safety-impacting systems still lacking a recorded public notice
mechanism.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os
import sys

# Load data
file_paths = ['../astalabs_discovery_all_data.csv', 'astalabs_discovery_all_data.csv']
df = None
for fp in file_paths:
    if os.path.exists(fp):
        print(f"Loading dataset from {fp}...")
        df = pd.read_csv(fp, low_memory=False)
        break

if df is None:
    print("Error: Dataset not found.")
    sys.exit(1)

# Filter for EO13960
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Total EO 13960 records: {len(eo_df)}")

# 1. Define High Impact
# Column: 17_impact_type
# Logic: contains 'rights' or 'safety' (case insensitive)
# Convert to string to handle NaNs safely
eo_df['17_impact_type_str'] = eo_df['17_impact_type'].fillna('').astype(str).str.lower()
eo_df['is_high_impact'] = eo_df['17_impact_type_str'].apply(lambda x: 1 if ('rights' in x or 'safety' in x) else 0)

# 2. Define Has Notice
# Column: 59_ai_notice
# Logic from prompt: Map to 0 if NaN, 'None of the above', or 'N/A'. Map to 1 otherwise.
def parse_notice(val):
    if pd.isna(val) or val == '':
        return 0
    s = str(val).lower().strip()
    if s == 'nan':
        return 0
    if 'none' in s:
        return 0
    if 'n/a' in s:
        return 0
    # If it's not None, not N/A, and not missing, we assume it describes a notice method
    return 1

eo_df['has_notice'] = eo_df['59_ai_notice'].apply(parse_notice)

# Debugging: show what values mapped to what
print("\nMapping check (Sample of values -> Result):")
sample_mapping = eo_df[['59_ai_notice', 'has_notice']].drop_duplicates().head(10)
print(sample_mapping)

# 3. Contingency Table
contingency = pd.crosstab(eo_df['is_high_impact'], eo_df['has_notice'])
contingency.index = ['Low Impact', 'High Impact']
contingency.columns = ['No Notice', 'Has Notice']

print("\nContingency Table:")
print(contingency)

# 4. Rates
low_stats = contingency.loc['Low Impact']
high_stats = contingency.loc['High Impact']

low_n = low_stats.sum()
high_n = high_stats.sum()

low_rate = low_stats['Has Notice'] / low_n if low_n > 0 else 0
high_rate = high_stats['Has Notice'] / high_n if high_n > 0 else 0

print(f"\nLow Impact Notice Rate:  {low_rate:.2%} ({low_stats['Has Notice']}/{low_n})")
print(f"High Impact Notice Rate: {high_rate:.2%} ({high_stats['Has Notice']}/{high_n})")

# 5. Chi-square Test
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

if p < 0.05:
    print("Conclusion: Significant difference in notice rates.")
else:
    print("Conclusion: No significant difference in notice rates.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
Total EO 13960 records: 1757

Mapping check (Sample of values -> Result):
                                          59_ai_notice  has_notice
0                                                  NaN           0
49   Online - in the terms or instructions for the ...           1
53                                     In-person,Other           1
58   Online - in the terms or instructions for the ...           1
107  N/A - individuals are not interacting with the...           0
108                                  None of the above           0
113                                          In-person           1
213                                             Other            1
222                                              Email           1
243                                             Email            1

Contingency Table:
             No Notice  Has Notice
Low Impact        1619          61
High Impact         64          13

Low Impact Notice Rate:  3.63% (61/1680)
High Impact Notice Rate: 16.88% (13/77)

Chi-Square Statistic: 28.8494
P-value: 7.8229e-08
Conclusion: Significant difference in notice rates.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
