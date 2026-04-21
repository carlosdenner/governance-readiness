# Experiment 149: node_6_13

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_13` |
| **ID in Run** | 149 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:05:36.436791+00:00 |
| **Runtime** | 172.6s |
| **Parent** | `node_5_34` |
| **Children** | `node_7_10` |
| **Creation Index** | 150 |

---

## Hypothesis

> The 'Opaque Shield' of Defense: Defense and Security agencies exhibit
significantly lower 'Public Transparency' rates (Public Notice) than Civilian
Social Service agencies, despite the high-stakes nature of security AI.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9597 (Definitely True) |
| **Posterior** | 0.3269 (Maybe False) |
| **Surprise** | -0.7593 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 27.0 |
| Maybe True | 3.0 |
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

**Objective:** Assess if security agencies utilize a 'secrecy shield' that bypasses standard transparency mandates.

### Steps
- 1. Filter for 'eo13960_scored'.
- 2. Categorize `3_agency` into 'Defense/Security' (e.g., DOD, DHS, DOJ, CIA, VA - check context) vs. 'Civilian/Service' (e.g., HHS, ED, HUD, DOT, USDA).
- 3. Analyze `59_ai_notice`: Create a binary 'Has_Notice' variable (parse 'Yes'/'Online'/'In-person' as 1, 'No'/'None' as 0).
- 4. Compare 'Has_Notice' rates between the two agency groups using a Chi-square test.

### Deliverables
- Transparency rates by Agency Type; Statistical significance of the difference.

---

## Analysis

The experiment successfully tested the 'Opaque Shield' hypothesis by comparing
public transparency rates between Defense/Security agencies and Civilian/Service
agencies.

**Hypothesis Status:** Not Supported (No Significant Difference).

**Execution Summary:**
1.  **Categorization:** The code successfully segmented the EO 13960 inventory
into 'Civilian/Service' (n=1,067) and 'Defense/Security' (n=463) groups based on
agency abbreviations and names.
2.  **Metric:** A binary 'Has Notice' variable was derived from the
`59_ai_notice` column, treating descriptive methods (e.g., 'Online', 'In-
person') as positive and explicit negatives/nones as negative.

**Findings:**
-   **Transparency Rates:** Both sectors exhibited very low public transparency
rates. Civilian agencies had a rate of **4.31%** (46/1067), while Defense
agencies had a rate of **3.89%** (18/463).
-   **Significance:** The Chi-square test yielded a p-value of **0.8095**,
indicating no statistically significant difference between the two groups.

**Interpretation:**
The data does not support the existence of a unique 'Opaque Shield' for defense
agencies regarding public notice. Instead, the results suggest a **universal
lack of transparency** across the federal government, with >95% of AI systems in
both civilian and defense sectors failing to provide documented public notice.

---

## Review

The experiment successfully tested the 'Opaque Shield' hypothesis by comparing
public transparency rates between Defense/Security agencies and Civilian/Service
agencies.

**Hypothesis Status:** Not Supported (No Significant Difference).

**Execution Summary:**
1.  **Categorization:** The code successfully segmented the EO 13960 inventory
into 'Civilian/Service' (n=1,067) and 'Defense/Security' (n=463) groups based on
agency abbreviations and names.
2.  **Metric:** A binary 'Has Notice' variable was derived from the
`59_ai_notice` column, treating descriptive methods (e.g., 'Online', 'In-
person') as positive and explicit negatives/nones as negative.

**Findings:**
-   **Transparency Rates:** Both sectors exhibited very low public transparency
rates. Civilian agencies had a rate of **4.31%** (46/1067), while Defense
agencies had a rate of **3.89%** (18/463).
-   **Significance:** The Chi-square test yielded a p-value of **0.8095**,
indicating no statistically significant difference between the two groups.

**Interpretation:**
The data does not support the existence of a unique 'Opaque Shield' for defense
agencies regarding public notice. Instead, the results suggest a **universal
lack of transparency** across the federal government, with >95% of AI systems in
both civilian and defense sectors failing to provide documented public notice.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# [debug] import matplotlib.pyplot as plt

print("Starting 'Opaque Shield' Experiment...\n")

# 1. Load Data
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for EO 13960
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Dataset Loaded. Rows: {len(eo_df)}")

# 3. Categorize Agencies
# Defense/Security List per prompt: DOD, DHS, DOJ, VA, STATE (implied context of security/diplomacy/defense bundle)
# Civilian/Service List: HHS, ED, HUD, DOT, USDA, DOL, DOE, DOC, DOI, TREAS, SSA, EPA, GSA, NASA, NSF, NRC, OPM, SBA, USAID

defense_security_abbr = ['DOD', 'DHS', 'DOJ', 'VA', 'STATE']
civilian_service_abbr = ['HHS', 'ED', 'HUD', 'DOT', 'USDA', 'DOL', 'DOE', 'DOC', 'DOI', 'TREAS', 'SSA', 'EPA', 'GSA', 'NASA', 'NSF', 'NRC', 'OPM', 'SBA', 'USAID']

def categorize_agency(row):
    abr = str(row['3_abr']).upper().strip()
    agency = str(row['3_agency']).upper().strip()
    
    # Check exact abbreviation match first
    if abr in defense_security_abbr:
        return 'Defense/Security'
    if abr in civilian_service_abbr:
        return 'Civilian/Service'
        
    # Fallback to name keywords
    if any(x in agency for x in ['DEFENSE', 'HOMELAND', 'JUSTICE', 'VETERAN', 'STATE DEPARTMENT']):
        return 'Defense/Security'
    if any(x in agency for x in ['HEALTH', 'EDUCATION', 'HOUSING', 'TRANSPORTATION', 'AGRICULTURE', 'LABOR', 'ENERGY', 'COMMERCE', 'INTERIOR', 'TREASURY', 'SOCIAL SECURITY', 'ENVIRONMENTAL', 'AERONAUTICS', 'SCIENCE FOUNDATION']):
        return 'Civilian/Service'
    
    return 'Other/Unknown'

eo_df['agency_category'] = eo_df.apply(categorize_agency, axis=1)

# Filter for analysis
analysis_df = eo_df[eo_df['agency_category'] != 'Other/Unknown'].copy()

print("\nAgency Categorization Counts:")
print(analysis_df['agency_category'].value_counts())

# 4. Define Has_Notice
# Target: '59_ai_notice'. Parse categorical text.
# Positive: 'Online', 'Email', 'In-person', 'Yes', 'Physical', 'Mailed', 'Other'
# Negative: 'None', 'N/A', 'No', nan

def parse_notice(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    
    # Explicit negatives
    if 'none' in val_str or 'n/a' in val_str or val_str == 'no':
        return 0
        
    # Explicit positives or descriptive text indicating method
    positive_keywords = ['online', 'email', 'person', 'yes', 'physical', 'mail', 'other']
    if any(kw in val_str for kw in positive_keywords):
        return 1
        
    return 0

analysis_df['has_notice'] = analysis_df['59_ai_notice'].apply(parse_notice)

# 5. Analysis
group_stats = analysis_df.groupby('agency_category')['has_notice'].agg(['count', 'sum', 'mean'])
group_stats['percent'] = group_stats['mean'] * 100

contingency_table = pd.crosstab(analysis_df['agency_category'], analysis_df['has_notice'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

# 6. Deliverables
print("\n--- Analysis of Public Transparency (Notice) by Agency Type ---")
print(group_stats)
print("\nContingency Table (0=No Notice, 1=Has Notice):")
print(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Hypothesis Check
def_rate = group_stats.loc['Defense/Security', 'mean']
civ_rate = group_stats.loc['Civilian/Service', 'mean']

print("\n--- Conclusion ---")
if p < 0.05:
    if def_rate < civ_rate:
        print("Hypothesis SUPPORTED: Defense/Security agencies have significantly LOWER transparency rates.")
    else:
        print("Hypothesis REFUTED: Defense/Security agencies have significantly HIGHER transparency rates.")
else:
    print("Hypothesis NOT SUPPORTED: No significant difference found.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting 'Opaque Shield' Experiment...

EO 13960 Dataset Loaded. Rows: 1757

Agency Categorization Counts:
agency_category
Civilian/Service    1067
Defense/Security     463
Name: count, dtype: int64

--- Analysis of Public Transparency (Notice) by Agency Type ---
                  count  sum      mean   percent
agency_category                                 
Civilian/Service   1067   46  0.043112  4.311153
Defense/Security    463   18  0.038877  3.887689

Contingency Table (0=No Notice, 1=Has Notice):
has_notice           0   1
agency_category           
Civilian/Service  1021  46
Defense/Security   445  18

Chi-Square Statistic: 0.0581
P-value: 8.0948e-01

--- Conclusion ---
Hypothesis NOT SUPPORTED: No significant difference found.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
