# Experiment 274: node_7_13

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_13` |
| **ID in Run** | 274 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T14:00:42.120644+00:00 |
| **Runtime** | 233.3s |
| **Parent** | `node_6_22` |
| **Children** | None |
| **Creation Index** | 275 |

---

## Hypothesis

> Surveillance Secrecy: AI use cases categorized under 'Law Enforcement' or
'National Security' are significantly less likely to provide 'Public Notice'
compared to 'Service Delivery' use cases.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.3736 (Maybe False) |
| **Surprise** | -0.7323 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 30.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 14.0 |
| Definitely False | 46.0 |

---

## Experiment Plan

**Objective:** Investigate if sensitive topic areas correlate with reduced public transparency.

### Steps
- 1. Load 'eo13960_scored'.
- 2. Categorize '8_topic_area' or '11_purpose_benefits' into 'Surveillance/Security' vs. 'Service/Benefits' using keyword matching.
- 3. Clean '59_ai_notice' (Yes/No).
- 4. perform a Chi-Square test comparing the two topic groups against public notice compliance.

### Deliverables
- Categorization logic summary; Contingency table; Statistical test results.

---

## Analysis

The experiment was successfully executed. The script correctly rectified the
previous logic error by implementing a robust mapping function for the
'59_ai_notice' column, identifying affirmative notice methods (e.g., 'Online',
'In-person') versus negative indicators (e.g., 'N/A', 'None').

The analysis isolated 62 relevant use cases (45 Service/Benefits, 17
Surveillance/Security). Contrary to the hypothesis that surveillance use cases
would be less transparent, the data showed a higher (though not statistically
significant) rate of public notice for Surveillance/Security (47.1%) compared to
Service/Benefits (33.3%).

The Chi-Square test yielded a p-value of 0.4818, which is well above the 0.05
threshold. Consequently, we **reject** the hypothesis of Surveillance Secrecy.
The data suggests that sensitive use cases in this inventory are not
systematically hiding public notice more than standard service delivery
applications; in fact, the directionality suggests a potential (non-significant)
tendency towards higher compliance in the security sector.

---

## Review

The experiment was successfully executed. The code corrected the previous logic
error regarding the '59_ai_notice' column by implementing a robust mapping of
descriptive values (e.g., 'Online', 'In-person') to binary outcomes. This
allowed for the isolation of 62 valid data points (45 Service, 17 Surveillance).
The analysis revealed that Surveillance/Security use cases actually had a higher
rate of public notice (47.1%) compared to Service/Benefits (33.3%), though the
difference was not statistically significant (Chi-Square p=0.4818).
Consequently, we **reject** the hypothesis of Surveillance Secrecy. The data
suggests that within this federal inventory, sensitive security applications are
not systematically less transparent regarding public notice than standard
service delivery applications.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

def run_experiment():
    # 1. Load Data
    fname = 'astalabs_discovery_all_data.csv'
    fpath = f'../{fname}' if os.path.exists(f'../{fname}') else fname
    
    print("Loading data...")
    try:
        df = pd.read_csv(fpath, low_memory=False)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return
    
    # Filter EO13960
    if 'source_table' not in df.columns:
        print("Column 'source_table' missing.")
        return

    eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO13960 rows: {len(eo)}")
    
    # 2. Inspect Unique Values for Mapping
    topic_col = '8_topic_area'
    notice_col = '59_ai_notice'
    
    print(f"\nUnique values in '{topic_col}':")
    print(eo[topic_col].dropna().unique())
    
    print(f"\nUnique values in '{notice_col}':")
    unique_notices = eo[notice_col].dropna().unique()
    for val in unique_notices:
        print(f"  - {val}")

    # 3. Categorize Topic (IV)
    def categorize_topic(val):
        val_str = str(val).lower()
        # Surveillance / Security / Enforcement
        if any(x in val_str for x in ['law', 'justice', 'security', 'defense', 'intelligence', 'border', 'police', 'enforcement']):
            return 'Surveillance/Security'
        # Service / Benefits
        if any(x in val_str for x in ['health', 'service', 'benefit', 'education', 'transportation', 'housing', 'agriculture', 'energy', 'labor', 'veteran', 'environment', 'commerce']):
            return 'Service/Benefits'
        return 'Other'

    eo['Group'] = eo[topic_col].apply(categorize_topic)
    
    # 4. Clean Notice (DV)
    # Logic: If the field indicates a method of notice (Online, Email, In-person), it's Yes.
    # If it says 'None', 'N/A', or is empty, it's No.
    def clean_notice(val):
        if pd.isna(val):
            return np.nan # Treat missing as missing, or could be No. Let's exclude for now.
        val_str = str(val).lower().strip()
        
        # Negative indicators
        if any(x in val_str for x in ['none', 'n/a', 'not applicable', 'no notice']):
            return 'No'
            
        # Positive indicators (explicit methods)
        if any(x in val_str for x in ['online', 'email', 'in-person', 'person', 'web', 'mail', 'media', 'press', 'notification', 'posted']):
            return 'Yes'
            
        # Ambiguous 'Other' - check if it has content, but usually 'Other' implies some notice method was used but not listed.
        # However, let's be conservative. If it just says 'other', we might treat as Yes if we assume they selected 'Other' to describe a method.
        # Let's inspect 'Other' cases if they dominate, but generally 'Other' in a checkbox list implies 'Yes, via another method'.
        if 'other' in val_str:
            return 'Yes'
            
        return 'No' # Default fall-through

    eo['Notice_Provided'] = eo[notice_col].apply(clean_notice)
    
    # 5. Analysis
    # Filter for the two groups of interest
    valid = eo[
        (eo['Group'].isin(['Surveillance/Security', 'Service/Benefits'])) & 
        (eo['Notice_Provided'].notna())
    ].copy()
    
    print(f"\nValid rows for analysis: {len(valid)}")
    print("Group distribution:\n", valid['Group'].value_counts())
    print("Notice distribution:\n", valid['Notice_Provided'].value_counts())
    
    # Contingency Table
    ct = pd.crosstab(valid['Group'], valid['Notice_Provided'])
    print("\n--- Contingency Table (Counts) ---")
    print(ct)
    
    # Check for empty intersection
    if ct.size == 0 or 'Yes' not in ct.columns or 'No' not in ct.columns:
        print("\nError: Contingency table is missing columns (Yes/No) or is empty. Cannot run Chi-Square.")
        return

    # Percentages
    ct_pct = pd.crosstab(valid['Group'], valid['Notice_Provided'], normalize='index') * 100
    print("\n--- Contingency Table (Percentages) ---")
    print(ct_pct.round(2))
    
    # Chi-Square Test
    chi2, p, dof, exp = chi2_contingency(ct)
    print(f"\n--- Statistical Test Results ---")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Interpretation
    surv_rate = ct_pct.loc['Surveillance/Security', 'Yes']
    serv_rate = ct_pct.loc['Service/Benefits', 'Yes']
    
    print("\n--- Interpretation ---")
    print(f"Surveillance Notice Rate: {surv_rate:.1f}%")
    print(f"Service Notice Rate:      {serv_rate:.1f}%")
    
    if p < 0.05:
        if surv_rate < serv_rate:
            print("Significant: Surveillance provides LESS notice (Hypothesis Supported).")
        else:
            print("Significant: Surveillance provides MORE notice (Hypothesis Rejected).")
    else:
        print("Not Significant: No statistical difference found.")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading data...
EO13960 rows: 1757

Unique values in '8_topic_area':
<StringArray>
[                                                'Law & Justice',
                                              'Mission-Enabling',
  'Government Services (includes Benefits and Service Delivery)',
                                              'Health & Medical',
                                             'Diplomacy & Trade',
                                         'Education & Workforce',
                                          'Emergency Management',
                                                'Transportation',
                                                         'Other',
                    'Mission-Enabling (internal agency support)',
                                               'Science & Space',
                   'Mission-Enabling (internal agency support) ',
                                      'Energy & the Environment',
                                   'Natural Language Processing',
                                                 'Deep Learning',
                                           'Statistical Methods',
                                                'Classification',
                                     'AIML Platform/Environment',
                                                           'NLP',
                                            'Mission-Enabeling ',
             'Other; Mission-Enabling (internal agency support)',
 'Diplomacy & Trade; Mission-Enabling (internal agency support)']
Length: 22, dtype: str

Unique values in '59_ai_notice':
  - Online - in the terms or instructions for the service.,In-person,Other
  - In-person,Other
  - Online - in the terms or instructions for the service.
  - N/A - individuals are not interacting with the AI for this use case
  - None of the above
  - In-person
  - Other 
  - Email
  - Email 
  - Other, if other please explain in next question.
  - Online – in the terms or instruction for the service
  - In person 
  - Email , Other, if other please explain in next question.
  - Agency CAIO has waived this minimum practice and reported such waiver to OMB.
  - AI is not safety or rights-impacting.
  - Other
  - Telephone

Valid rows for analysis: 62
Group distribution:
 Group
Service/Benefits         45
Surveillance/Security    17
Name: count, dtype: int64
Notice distribution:
 Notice_Provided
No     39
Yes    23
Name: count, dtype: int64

--- Contingency Table (Counts) ---
Notice_Provided        No  Yes
Group                         
Service/Benefits       30   15
Surveillance/Security   9    8

--- Contingency Table (Percentages) ---
Notice_Provided           No    Yes
Group                              
Service/Benefits       66.67  33.33
Surveillance/Security  52.94  47.06

--- Statistical Test Results ---
Chi-Square Statistic: 0.4948
P-value: 4.8181e-01

--- Interpretation ---
Surveillance Notice Rate: 47.1%
Service Notice Rate:      33.3%
Not Significant: No statistical difference found.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
