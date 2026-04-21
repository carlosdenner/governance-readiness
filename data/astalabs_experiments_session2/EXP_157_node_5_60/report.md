# Experiment 157: node_5_60

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_60` |
| **ID in Run** | 157 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:29:03.638557+00:00 |
| **Runtime** | 217.4s |
| **Parent** | `node_4_21` |
| **Children** | `node_6_42`, `node_6_58` |
| **Creation Index** | 158 |

---

## Hypothesis

> Sector-Specific Opacity: AI systems deployed in 'Law Enforcement' and 'Defense'
topic areas are significantly less likely to provide 'Code Access' or 'Data
Documentation' compared to systems in 'Science' or 'Health', indicating a
transparency gap in high-stakes coercive domains.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9758 (Definitely True) |
| **Posterior** | 0.3324 (Maybe False) |
| **Surprise** | -0.7721 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 29.0 |
| Maybe True | 1.0 |
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

**Objective:** Assess transparency availability (Code/Data access) across different government sectors.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `eo13960_scored`.
- 2. Categorize `8_topic_area` into 'Coercive' (Law Enforcement, Defense, Security, Justice) and 'Civil/Scientific' (Science, Health, Environment, Education).
- 3. Create a composite binary variable `is_transparent`: 1 if `38_code_access` != 'No' OR `27_data_docs` != 'No' (adjust based on actual negative values like 'None', 'Not Applicable'), else 0.
- 4. Perform a Chi-square test comparing `is_transparent` rates between 'Coercive' and 'Civil/Scientific' sectors.
- 5. Compute the transparency percentage for each group.

### Deliverables
- Contingency table, transparency rates per sector group, and Chi-square test statistics.

---

## Analysis

The experiment successfully tested the 'Sector-Specific Opacity' hypothesis
using the EO 13960 dataset. After classifying 475 federal AI systems into
'Civil/Scientific' (n=389) and 'Coercive' (n=86) sectors, the analysis compared
their transparency rates (defined as having either Code Access or Data
Documentation).

The results showed relatively high transparency rates for both groups:
- Civil/Scientific: 72.5% transparent.
- Coercive (Law Enforcement/Defense): 68.6% transparent.

A Chi-square test yielded a p-value of 0.5533, which is well above the 0.05
threshold. Consequently, the hypothesis is **rejected**. There is no
statistically significant evidence in this dataset that high-stakes coercive
sectors are more opaque than scientific or civil sectors regarding code and data
availability; both appear to adhere to similar disclosure standards.

---

## Review

The experiment was successfully executed and the code faithfully implemented the
analysis plan. After correcting for data type issues in the first attempt
(handling mixed types/NaNs in the 'topic_area' column), the analysis
successfully categorized 475 federal AI systems into 'Civil/Scientific' (n=389)
and 'Coercive' (n=86) sectors.

Hypothesis: Sector-Specific Opacity (Coercive sectors are less transparent than
Civil/Scientific sectors).

Findings:
1. Transparency Rates (defined as having accessible Code OR Data Documentation):
   - Civil/Scientific: 72.49% (282/389)
   - Coercive (Law Enforcement/Defense): 68.60% (59/86)

2. Statistical Test:
   - Chi-Square Statistic: 0.3514
   - p-value: 0.5533

Conclusion:
The p-value (> 0.05) indicates no statistically significant difference in
transparency rates between the two sector groups. The hypothesis is
**rejected**. The data suggests that federal AI systems in high-stakes coercive
domains (Defense, Law Enforcement) adhere to transparency standards (regarding
code and data availability) at rates comparable to scientific and civil sectors.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import sys
import numpy as np

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# Filter for EO 13960 source table
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# Identify columns dynamically
cols = df_eo.columns
topic_col = next((c for c in cols if 'topic_area' in c.lower()), None)
code_col = next((c for c in cols if 'code_access' in c.lower()), None)
data_col = next((c for c in cols if 'data_docs' in c.lower()), None)

if not all([topic_col, code_col, data_col]):
    print(f"Error: Critical columns missing. Found: Topic={topic_col}, Code={code_col}, Data={data_col}")
    sys.exit(1)

print(f"Processing with columns: '{topic_col}', '{code_col}', '{data_col}'")

# --- Step 1: Sector Categorization ---
# Ensure the column is string and handle NaNs
df_eo['topic_clean'] = df_eo[topic_col].fillna('').astype(str).str.lower()

# Define keywords
coercive_kw = ['law enforcement', 'defense', 'security', 'justice', 'border', 'police', 'military']
civil_kw = ['health', 'science', 'environment', 'education']

def classify_sector(val):
    if not isinstance(val, str):
        return 'Other'
    if any(k in val for k in coercive_kw):
        return 'Coercive'
    elif any(k in val for k in civil_kw):
        return 'Civil/Scientific'
    else:
        return 'Other'

df_eo['sector_group'] = df_eo['topic_clean'].apply(classify_sector)

# Filter dataset for analysis
df_analysis = df_eo[df_eo['sector_group'].isin(['Coercive', 'Civil/Scientific'])].copy()
print(f"Filtered for analysis: {len(df_analysis)} rows (Coercive + Civil/Scientific)")

# --- Step 2: Define Transparency ---
negative_values = ['no', 'none', 'not applicable', 'n/a', 'nan', 'false', '0', 'closed', 'restricted', '']

def is_transparent_feature(val):
    s = str(val).lower().strip()
    if s in negative_values or s == 'nan':
        return 0
    return 1

df_analysis['code_open'] = df_analysis[code_col].apply(is_transparent_feature)
df_analysis['data_open'] = df_analysis[data_col].apply(is_transparent_feature)

# Composite Variable: Transparent if EITHER code OR data docs are available
df_analysis['is_transparent'] = ((df_analysis['code_open'] == 1) | (df_analysis['data_open'] == 1)).astype(int)

# --- Step 3: Statistical Test ---
contingency = pd.crosstab(df_analysis['sector_group'], df_analysis['is_transparent'])

# Ensure both 0 and 1 columns exist
if 0 not in contingency.columns:
    contingency[0] = 0
if 1 not in contingency.columns:
    contingency[1] = 0
contingency = contingency[[0, 1]]
contingency.columns = ['Opaque (0)', 'Transparent (1)']

print("\n--- Contingency Table (Sector vs Transparency) ---")
print(contingency)

# Calculate Transparency Rates
rates = df_analysis.groupby('sector_group')['is_transparent'].mean() * 100
print("\n--- Transparency Rates (% with Code or Data Access) ---")
print(rates)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print("\n--- Chi-Square Test Results ---")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p:.6f}")

if p < 0.05:
    print("Result: Statistically significant difference found.")
else:
    print("Result: No statistically significant difference found.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Processing with columns: '8_topic_area', '38_code_access', '34_data_docs'
Filtered for analysis: 475 rows (Coercive + Civil/Scientific)

--- Contingency Table (Sector vs Transparency) ---
                  Opaque (0)  Transparent (1)
sector_group                                 
Civil/Scientific         107              282
Coercive                  27               59

--- Transparency Rates (% with Code or Data Access) ---
sector_group
Civil/Scientific    72.493573
Coercive            68.604651
Name: is_transparent, dtype: float64

--- Chi-Square Test Results ---
Chi-square statistic: 0.3514
p-value: 0.553294
Result: No statistically significant difference found.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
