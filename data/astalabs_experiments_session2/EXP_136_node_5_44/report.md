# Experiment 136: node_5_44

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_44` |
| **ID in Run** | 136 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:25:08.931248+00:00 |
| **Runtime** | 288.8s |
| **Parent** | `node_4_27` |
| **Children** | `node_6_52`, `node_6_83` |
| **Creation Index** | 137 |

---

## Hypothesis

> Temporal Severity Escalation: The proportion of AI incidents classified as
'Severe' or 'Critical' in the AIID dataset has increased significantly in the
post-2020 era (2021-Present) compared to the pre-2021 era, reflecting the
deployment of more capable and risky systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6048 (Maybe True) |
| **Posterior** | 0.2170 (Likely False) |
| **Surprise** | -0.4654 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 15.0 |
| Uncertain | 13.0 |
| Maybe False | 2.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 4.0 |
| Definitely False | 56.0 |

---

## Experiment Plan

**Objective:** Evaluate if incident severity has increased over time.

### Steps
- 1. Filter dataset for 'aiid_incidents'.
- 2. Convert 'date' to datetime objects and extract the Year.
- 3. Create a 'Period' variable: 'Pre-2021' (<= 2020) and 'Post-2020' (>= 2021).
- 4. Bin '75_AI Harm Level' into 'High_Severity' (e.g., 'Critical', 'Severe', 'Death') and 'Low_Severity' (e.g., 'Minor', 'Moderate', 'Near miss').
- 5. Generate a contingency table and run a Chi-square test.

### Deliverables
- Yearly distribution plot (optional), Contingency table, Chi-square stats.

---

## Analysis

The experiment was successfully executed following the insights from the
previous debugging step. The programmer correctly mapped the specific dataset
values ('AI tangible harm event' as High Severity; 'AI tangible harm near-miss',
'issue', and 'none' as Low Severity) to the analysis categories.

The analysis of 191 qualified incidents revealed:
- **Pre-2021 (N=140):** 25.71% High Severity (36 incidents).
- **Post-2020 (N=51):** 15.69% High Severity (8 incidents).

Contrary to the hypothesis, the proportion of high-severity incidents actually
decreased in the post-2020 period in this sample, although the Chi-square test
(p=0.2070) indicates that this difference is not statistically significant.
Therefore, the hypothesis that AI incident severity has significantly escalated
in the post-2020 era is rejected based on this dataset.

---

## Review

The experiment was successfully executed. Initial assumptions about the data
schema (expecting keywords like 'Critical' or 'Severe') were incorrect, but the
programmer successfully debugged the dataset, identified the actual taxonomy
('AI tangible harm event', 'AI tangible harm near-miss', etc.), and adapted the
mapping logic accordingly. The analysis of 191 dated incidents revealed that the
proportion of high-severity incidents actually decreased from 25.7% in the
Pre-2021 era to 15.7% in the Post-2020 era, though this difference was not
statistically significant (p=0.207). Consequently, the hypothesis that incident
severity has escalated over time is rejected based on this sample.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

def run_experiment():
    try:
        print("Starting Temporal Severity Escalation Experiment...")
        
        # 1. Load Dataset
        filename = 'astalabs_discovery_all_data.csv'
        if not os.path.exists(filename):
            if os.path.exists(f'../{filename}'):
                filename = f'../{filename}'
            else:
                print("Error: Dataset not found.")
                return

        df = pd.read_csv(filename, low_memory=False)
        
        # 2. Filter for AIID incidents
        aiid = df[df['source_table'] == 'aiid_incidents'].copy()
        print(f"AIID Incidents loaded: {len(aiid)}")

        # 3. Process Dates
        aiid['date'] = pd.to_datetime(aiid['date'], errors='coerce')
        aiid = aiid.dropna(subset=['date'])
        aiid['year'] = aiid['date'].dt.year
        
        # Define Periods
        # Pre-2021 (<= 2020) and Post-2020 (>= 2021)
        aiid['Period'] = aiid['year'].apply(lambda x: 'Post-2020' if x >= 2021 else 'Pre-2021')
        
        # 4. Process Severity (Using schema found in debug)
        # High: 'AI tangible harm event'
        # Low: 'AI tangible harm near-miss', 'AI tangible harm issue', 'none'
        sev_col = 'AI Harm Level'
        
        def map_severity(val):
            if pd.isna(val):
                return None
            val = str(val).strip()
            if val == 'AI tangible harm event':
                return 'High_Severity'
            elif val in ['AI tangible harm near-miss', 'AI tangible harm issue', 'none']:
                return 'Low_Severity'
            return None

        aiid['Severity_Class'] = aiid[sev_col].apply(map_severity)
        
        # Filter for valid analysis rows
        analysis_df = aiid.dropna(subset=['Severity_Class']).copy()
        
        print(f"Records included in analysis: {len(analysis_df)}")
        print("Distribution of Severity Classes:")
        print(analysis_df['Severity_Class'].value_counts())

        # 5. Contingency Table
        ct = pd.crosstab(analysis_df['Period'], analysis_df['Severity_Class'])
        # Ensure consistent order
        ct = ct.reindex(index=['Pre-2021', 'Post-2020'], columns=['Low_Severity', 'High_Severity'], fill_value=0)
        
        print("\n--- Contingency Table ---")
        print(ct)

        # 6. Chi-Square Test
        chi2, p, dof, expected = chi2_contingency(ct)
        print(f"\nChi-Square Statistic: {chi2:.4f}")
        print(f"P-Value: {p:.4f}")
        
        # Interpretation
        if p < 0.05:
            print("Result: Statistically significant difference found.")
        else:
            print("Result: No statistically significant difference found.")

        # 7. Proportions
        props = pd.crosstab(analysis_df['Period'], analysis_df['Severity_Class'], normalize='index') * 100
        print("\n--- Proportions (%) ---")
        print(props.round(2))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting Temporal Severity Escalation Experiment...
AIID Incidents loaded: 1362
Records included in analysis: 191
Distribution of Severity Classes:
Severity_Class
Low_Severity     147
High_Severity     44
Name: count, dtype: int64

--- Contingency Table ---
Severity_Class  Low_Severity  High_Severity
Period                                     
Pre-2021                 104             36
Post-2020                 43              8

Chi-Square Statistic: 1.5924
P-Value: 0.2070
Result: No statistically significant difference found.

--- Proportions (%) ---
Severity_Class  High_Severity  Low_Severity
Period                                     
Post-2020               15.69         84.31
Pre-2021                25.71         74.29

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
