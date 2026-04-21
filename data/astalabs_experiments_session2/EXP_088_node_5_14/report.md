# Experiment 88: node_5_14

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_14` |
| **ID in Run** | 88 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:13:49.351228+00:00 |
| **Runtime** | 300.4s |
| **Parent** | `node_4_23` |
| **Children** | `node_6_5`, `node_6_64` |
| **Creation Index** | 89 |

---

## Hypothesis

> Agency Risk Appetite: Security-focused agencies (DOD, DHS, DOJ) deploy a
significantly higher proportion of 'Rights-Impacting' AI systems compared to
Social-service agencies (HHS, Education, VA), yet show no higher rates of
Independent Evaluation.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6774 (Maybe True) |
| **Posterior** | 0.3791 (Maybe False) |
| **Surprise** | -0.3580 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 26.0 |
| Uncertain | 0.0 |
| Maybe False | 4.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 54.0 |
| Definitely False | 6.0 |

---

## Experiment Plan

**Objective:** Compare the risk profile and governance rigor of security vs. civilian agencies.

### Steps
- 1. Load `eo13960_scored`.
- 2. Create an `Agency_Category` variable: Map 'Department of Defense', 'Homeland Security', 'Justice' to 'Security'; map 'Health', 'Education', 'Veterans' to 'Civilian'.
- 3. Calculate the ratio of 'Rights-Impacting' (`17_impact_type`) systems for each group.
- 4. For the Rights-Impacting subset, compare rates of `55_independent_eval` (Yes/No).
- 5. Perform Z-tests for proportions for both metrics.

### Deliverables
- Comparison table of Rights-Impacting ratios; Comparison table of Independent Evaluation rates; Z-test results.

---

## Analysis

The experiment successfully categorized 683 AI use cases into 'Security' (183)
and 'Social' (500) agency clusters.

1. **Risk Profile**: The analysis confirms the first part of the hypothesis.
Security agencies deploy a significantly higher proportion of 'Rights-Impacting'
systems (14.2%, n=26) compared to Social agencies (2.0%, n=10), with a Z-score
of 6.32 (p < 0.001).

2. **Governance Rigor**: The second part of the hypothesis—that Security
agencies show "no higher rates of Independent Evaluation"—was **rejected**.
Within the subset of Rights-Impacting systems, Security agencies demonstrated a
significantly higher compliance rate for Independent Evaluation (57.7%, 15/26)
compared to Social agencies, who reported zero independent evaluations for their
10 rights-impacting systems (0.0%). The Z-test (z=3.14, p=0.0017) confirms this
difference is statistically significant.

This suggests that while Security agencies take on more high-risk AI
applications, they are also applying more rigorous governance controls
(Independent Evaluation) compared to their civilian counterparts in this
dataset.

---

## Review

The experiment was successfully executed and robustly tested the 'Agency Risk
Appetite' hypothesis.

**Findings:**
1. **Risk Profile (Supported):** The analysis confirmed that Security-focused
agencies (DOD, DHS, DOJ) deploy a significantly higher proportion of 'Rights-
Impacting' AI systems (14.2%, n=26/183) compared to Social-service agencies
(HHS, Education, VA) (2.0%, n=10/500). The Z-test (z=6.32, p<0.001) provides
strong evidence for this difference.

2. **Governance Rigor (Rejected):** Contrary to the hypothesis which predicted
'no higher rates' of evaluation, Security agencies demonstrated significantly
superior governance rigor. Within the subset of rights-impacting systems,
Security agencies reported a 57.7% rate of Independent Evaluation (15/26),
whereas Social agencies reported 0.0% (0/10). The Z-test (z=3.14, p=0.0017)
confirms this is a statistically significant difference.

**Conclusion:** The hypothesis is partially supported (regarding risk profile)
but falsified regarding governance. Security agencies are not loosely governing
high-risk systems; rather, they appear to be the primary drivers of both high-
risk AI adoption and the associated rigorous evaluation practices in this
dataset, while Social agencies lag in both high-risk adoption and the
implementation of independent evaluations for the few sensitive systems they do
possess.

---

## Code

```python
import pandas as pd
import numpy as np
import os
import math

# --- Helper Functions ---

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def z_test_proportions(count1, nobs1, count2, nobs2):
    # Calculates two-sided Z-test for proportions
    if nobs1 == 0 or nobs2 == 0:
        return 0.0, 1.0
    
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    p_pool = (count1 + count2) / (nobs1 + nobs2)
    
    if p_pool == 0 or p_pool == 1:
        return 0.0, 1.0
        
    se = np.sqrt(p_pool * (1 - p_pool) * (1/nobs1 + 1/nobs2))
    
    if se == 0:
        return 0.0, 1.0
        
    z = (p1 - p2) / se
    p_value = 2 * (1 - norm_cdf(abs(z)))
    return z, p_value

# --- Experiment Code ---

def run_experiment():
    print("Starting Agency Risk Appetite Experiment...\n")
    
    # 1. Load Dataset
    filename = 'astalabs_discovery_all_data.csv'
    filepath = filename
    if not os.path.exists(filepath):
        filepath = '../' + filename
        
    if not os.path.exists(filepath):
        print(f"Error: Dataset {filename} not found.")
        return

    print(f"Loading dataset from {filepath}...")
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Filter for EO13960 data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"Loaded EO13960 subset: {len(df_eo)} rows")

    # 2. Categorize Agencies
    security_keywords = ['Department of Defense', 'Homeland Security', 'Department of Justice']
    social_keywords = ['Health and Human Services', 'Department of Education', 'Veterans Affairs']

    def get_category(agency):
        if pd.isna(agency):
            return None
        agency_str = str(agency)
        if any(k in agency_str for k in security_keywords):
            return 'Security'
        if any(k in agency_str for k in social_keywords):
            return 'Social'
        return 'Other'

    df_eo['Agency_Category'] = df_eo['3_agency'].apply(get_category)
    
    # Filter for analysis groups
    df_analysis = df_eo[df_eo['Agency_Category'].isin(['Security', 'Social'])].copy()
    
    print("\nAgency Category Distribution:")
    print(df_analysis['Agency_Category'].value_counts())

    if df_analysis.empty:
        print("No matching agencies found. Exiting.")
        return

    # 3. Analyze Rights-Impacting Systems
    # Robust check using vectorized string operation
    df_analysis['is_rights'] = df_analysis['17_impact_type'].astype(str).str.contains('Rights', case=False, na=False)

    print("\n--- Analysis 1: Rights-Impacting Systems ---")
    # Groupby to get counts
    rights_counts = df_analysis.groupby('Agency_Category')['is_rights'].sum()
    total_counts = df_analysis.groupby('Agency_Category')['is_rights'].count()
    
    rights_stats = pd.DataFrame({'Rights_Count': rights_counts, 'Total_Systems': total_counts})
    rights_stats['Proportion'] = rights_stats['Rights_Count'] / rights_stats['Total_Systems']
    print(rights_stats)

    # Z-test for Rights-Impacting
    if 'Security' in rights_stats.index and 'Social' in rights_stats.index:
        sec_cnt = rights_stats.loc['Security', 'Rights_Count']
        sec_tot = rights_stats.loc['Security', 'Total_Systems']
        soc_cnt = rights_stats.loc['Social', 'Rights_Count']
        soc_tot = rights_stats.loc['Social', 'Total_Systems']
        
        z1, p1 = z_test_proportions(sec_cnt, sec_tot, soc_cnt, soc_tot)
        print(f"\nZ-Test (Rights-Impacting - Security vs Social): z = {z1:.4f}, p = {p1:.4e}")
        if p1 < 0.05:
            print("Result: Significant difference.")
        else:
            print("Result: No significant difference.")
    else:
        print("Cannot perform Z-test: Missing Security or Social group.")

    # 4. Analyze Independent Evaluation (Subset: Rights-Impacting)
    df_rights_subset = df_analysis[df_analysis['is_rights']].copy()
    
    # Robust check for 'Yes' in '55_independent_eval'
    df_rights_subset['has_eval'] = df_rights_subset['55_independent_eval'].astype(str).str.contains('Yes', case=False, na=False)

    print("\n--- Analysis 2: Independent Evaluation (Rights-Impacting Subset) ---")
    if len(df_rights_subset) == 0:
        print("No Rights-Impacting systems found to analyze.")
    else:
        eval_counts = df_rights_subset.groupby('Agency_Category')['has_eval'].sum()
        eval_total = df_rights_subset.groupby('Agency_Category')['has_eval'].count()
        
        eval_stats = pd.DataFrame({'Eval_Yes_Count': eval_counts, 'Total_Rights_Systems': eval_total})
        eval_stats['Proportion'] = eval_stats['Eval_Yes_Count'] / eval_stats['Total_Rights_Systems']
        print(eval_stats)

        # Z-test for Independent Eval
        if 'Security' in eval_stats.index and 'Social' in eval_stats.index:
            sec_e_cnt = eval_stats.loc['Security', 'Eval_Yes_Count']
            sec_e_tot = eval_stats.loc['Security', 'Total_Rights_Systems']
            soc_e_cnt = eval_stats.loc['Social', 'Eval_Yes_Count']
            soc_e_tot = eval_stats.loc['Social', 'Total_Rights_Systems']
            
            if sec_e_tot > 0 and soc_e_tot > 0:
                z2, p2 = z_test_proportions(sec_e_cnt, sec_e_tot, soc_e_cnt, soc_e_tot)
                print(f"\nZ-Test (Independent Eval - Security vs Social): z = {z2:.4f}, p = {p2:.4e}")
                if p2 < 0.05:
                    print("Result: Significant difference.")
                else:
                    print("Result: No significant difference.")
            else:
                 print("Cannot perform Z-test: One group has 0 systems.")
        else:
            print("Insufficient categories for comparison in subset.")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting Agency Risk Appetite Experiment...

Loading dataset from astalabs_discovery_all_data.csv...
Loaded EO13960 subset: 1757 rows

Agency Category Distribution:
Agency_Category
Social      500
Security    183
Name: count, dtype: int64

--- Analysis 1: Rights-Impacting Systems ---
                 Rights_Count  Total_Systems  Proportion
Agency_Category                                         
Security                   26            183    0.142077
Social                     10            500    0.020000

Z-Test (Rights-Impacting - Security vs Social): z = 6.3234, p = 2.5590e-10
Result: Significant difference.

--- Analysis 2: Independent Evaluation (Rights-Impacting Subset) ---
                 Eval_Yes_Count  Total_Rights_Systems  Proportion
Agency_Category                                                  
Security                     15                    26    0.576923
Social                        0                    10    0.000000

Z-Test (Independent Eval - Security vs Social): z = 3.1449, p = 1.6617e-03
Result: Significant difference.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
