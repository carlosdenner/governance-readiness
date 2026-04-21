# Experiment 240: node_6_52

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_52` |
| **ID in Run** | 240 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T12:15:40.072801+00:00 |
| **Runtime** | 488.7s |
| **Parent** | `node_5_44` |
| **Children** | `node_7_20` |
| **Creation Index** | 241 |

---

## Hypothesis

> Sector-Specific Harm Domains: AI incidents in the 'Government' sector are
disproportionately associated with 'Civil Rights' or 'Liberty' harms compared to
the 'Healthcare' sector, which is dominated by 'Physical Safety' harms.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9274 (Definitely True) |
| **Posterior** | 0.7246 (Likely True) |
| **Surprise** | -0.2434 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 23.0 |
| Maybe True | 7.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 28.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Compare the distribution of harm domains across Government and Healthcare sectors.

### Steps
- 1. Filter for 'aiid_incidents'.
- 2. Create a 'Sector Group' variable: Map '78_Sector of Deployment' entries containing 'Government', 'Police', 'Military' to 'Government', and 'Healthcare', 'Medical' to 'Healthcare'.
- 3. Create a 'Harm Group' variable from '73_Harm Domain': Group into 'Rights/Liberties' vs 'Physical/Safety'.
- 4. Create a contingency table for the subset of data matching these groups.
- 5. Perform a Chi-square test.

### Deliverables
- Sector vs. Harm contingency table, Chi-square stats.

---

## Analysis

The experiment was executed successfully, utilizing a text-mining approach
('title' + 'description') after previous attempts revealed that the structured
columns 'Harm Domain' and 'Tangible Harm' contained non-taxonomic boolean/status
data (e.g., 'yes', 'maybe', 'near-miss') rather than the expected categories.
The text analysis inferred sector and harm categories for a small subset of the
data (N=8 incidents). The resulting contingency table was too sparse to yield a
statistically significant result (p=1.0). Consequently, the hypothesis that the
Government sector is disproportionately associated with Rights/Liberties harms
while Healthcare is associated with Physical/Safety harms cannot be validated
with this dataset due to the lack of explicit taxonomic coding in the available
columns. The null result is primarily a reflection of data limitations.

---

## Review

The experiment was successfully executed, though the results were limited by
data quality issues identified during the process. The initial hypothesis relied
on the assumption that the 'Harm Domain' or 'Tangible Harm' columns contained
specific taxonomic categories (e.g., 'Civil Rights', 'Physical Safety').
However, the investigation revealed that these columns actually contained status
indicators (e.g., 'near-miss', 'definitively occurred') or boolean flags ('yes',
'no').

The programmer correctly adapted by attempting to infer these categories from
the unstructured 'title' and 'description' text fields using keyword matching.
This fallback approach yielded a very small sample size (N=8) suitable for
analysis. The resulting Chi-square test (p=1.0) showed no statistically
significant difference between the Government and Healthcare sectors regarding
harm types in this limited sample. Consequently, the hypothesis cannot be
supported, primarily due to the lack of explicit harm-type coding in the
available structured data.

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

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# Prepare text fields for analysis (fill NaNs)
aiid_df['text_content'] = aiid_df['title'].fillna('') + " " + aiid_df['description'].fillna('')
aiid_df['text_content'] = aiid_df['text_content'].str.lower()

# --- MAPPING FUNCTIONS ---

def map_sector(s):
    s = str(s).lower()
    # Government/Public Sector
    if any(x in s for x in ['public administration', 'defense', 'law enforcement', 'justice', 'police', 'government', 'military']):
        return 'Government'
    # Healthcare
    if any(x in s for x in ['human health', 'medical', 'hospital', 'medicine', 'healthcare', 'clinical']):
        return 'Healthcare'
    return None

def map_harm_from_text(text):
    # Keywords for Rights/Liberties
    rights_keywords = ['discrimination', 'bias', 'privacy', 'surveillance', 'civil rights', 'liberty', 'racist', 'sexist', 'wrongful arrest', 'false arrest', 'denied', 'unfair']
    # Keywords for Physical Safety
    physical_keywords = ['death', 'killed', 'injury', 'injured', 'accident', 'crash', 'collision', 'physical harm', 'safety', 'died', 'fatal']
    
    has_rights = any(k in text for k in rights_keywords)
    has_physical = any(k in text for k in physical_keywords)
    
    if has_rights and not has_physical:
        return 'Rights/Liberties'
    if has_physical and not has_rights:
        return 'Physical/Safety'
    if has_rights and has_physical:
        return 'Mixed/Both' # Exclude to be clean
    return None

# Apply mappings
aiid_df['Sector_Group'] = aiid_df['Sector of Deployment'].apply(map_sector)
aiid_df['Harm_Group'] = aiid_df['text_content'].apply(map_harm_from_text)

# Filter for analysis (exclude Mixed or None)
final_df = aiid_df.dropna(subset=['Sector_Group', 'Harm_Group'])
final_df = final_df[final_df['Harm_Group'].isin(['Rights/Liberties', 'Physical/Safety'])]

print(f"Total AIID Incidents: {len(aiid_df)}")
print(f"Incidents with Sector/Harm data (inferred from text): {len(final_df)}")

if len(final_df) > 0:
    # Contingency Table
    contingency_table = pd.crosstab(final_df['Sector_Group'], final_df['Harm_Group'])
    print("\nContingency Table (Sector vs Harm):")
    print(contingency_table)

    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")

    # Row percentages
    row_pcts = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    print("\nRow Percentages:")
    print(row_pcts.round(2))
    
    # Check hypothesis
    if p < 0.05:
        gov_rights = row_pcts.loc['Government', 'Rights/Liberties'] if 'Rights/Liberties' in row_pcts.columns else 0
        health_phys = row_pcts.loc['Healthcare', 'Physical/Safety'] if 'Physical/Safety' in row_pcts.columns else 0
        
        print(f"\nGov Rights %: {gov_rights:.1f}%")
        print(f"Health Safety %: {health_phys:.1f}%")
        
        if gov_rights > 50 and health_phys > 50:
            print("\nRESULT: Hypothesis Supported. Government sector is dominated by Rights/Liberties harms, Healthcare by Physical/Safety harms.")
        else:
            print("\nRESULT: Significant difference found, but proportions may not fully align with hypothesis direction.")
    else:
        print("\nRESULT: No statistically significant association found.")
else:
    print("\nNo data available after text analysis mapping.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total AIID Incidents: 1362
Incidents with Sector/Harm data (inferred from text): 8

Contingency Table (Sector vs Harm):
Harm_Group    Physical/Safety  Rights/Liberties
Sector_Group                                   
Government                  0                 4
Healthcare                  1                 3

Chi-square Statistic: 0.0000
P-value: 1.0000e+00

Row Percentages:
Harm_Group    Physical/Safety  Rights/Liberties
Sector_Group                                   
Government                0.0             100.0
Healthcare               25.0              75.0

RESULT: No statistically significant association found.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
