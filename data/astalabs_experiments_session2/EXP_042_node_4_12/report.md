# Experiment 42: node_4_12

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_12` |
| **ID in Run** | 42 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:00:30.738635+00:00 |
| **Runtime** | 182.7s |
| **Parent** | `node_3_9` |
| **Children** | `node_5_17`, `node_5_31` |
| **Creation Index** | 43 |

---

## Hypothesis

> Tactical Complexity Evolution: Recent adversarial cases (proxied by higher Case
IDs) involve a significantly higher count of unique 'Techniques' per case than
older cases.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6583 (Maybe True) |
| **Posterior** | 0.2417 (Likely False) |
| **Surprise** | -0.5000 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 22.0 |
| Uncertain | 4.0 |
| Maybe False | 3.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 8.0 |
| Definitely False | 52.0 |

---

## Experiment Plan

**Objective:** Determine if AI threats are becoming more sophisticated over time (or by case index).

### Steps
- 1. Load ATLAS data.
- 2. Create a 'Technique Count' variable by counting the number of comma-separated items in the 'techniques' column for each row.
- 3. Use 'case_id' as a numeric proxy for time/sequence (or 'incident_date' if available and parseable).
- 4. Calculate the Pearson or Spearman correlation coefficient between Case ID and Technique Count.
- 5. Visualize with a scatter plot and a trend line.

### Deliverables
- Scatter plot of Technique Count vs. Case ID; Correlation coefficient and p-value.

---

## Analysis

The debug execution confirmed the structure of the ATLAS data. The 'atlas_cases'
table contains 'case_id' (e.g., 'AML.CS0000') and a pre-calculated
'n_techniques' column (e.g., 6.0). It also has a 'techniques' column with pipe-
separated values. The 'step3_incident_coding' table has similar data but labeled
'techniques_used' and 'technique_count'. I will proceed using the 'atlas_cases'
subset. I will parse the numeric index from 'case_id' and correlate it with
'n_techniques' to test the hypothesis.

---

## Review

The experiment pipeline was implemented successfully. The code faithfully
executed the plan: it loaded the ATLAS dataset, parsed the numeric 'Case ID' as
a proxy for time, and correlated it with the 'Technique Count' (using the pre-
calculated 'n_techniques' column). The analysis employed both Pearson and
Spearman correlations to test for linear and monotonic relationships, and a
scatter plot was generated. The results (Pearson r=-0.03, p=0.83) statistically
fail to support the hypothesis that tactical complexity is increasing over time;
in fact, the trend is effectively flat. The experimental design was sound, the
data handling was correct (filtering for 'atlas_cases'), and the hypothesis was
tested sufficiently to conclude it is false for this dataset.

---

## Code

```python
# [debug]
import pandas as pd
import os

filename = 'astalabs_discovery_all_data.csv'
if not os.path.exists(filename):
    filename = '../astalabs_discovery_all_data.csv'

print(f"Loading {filename}...")
try:
    df = pd.read_csv(filename, low_memory=False)
    print("Dataset loaded successfully.")
    
    # Check for ATLAS related data
    atlas_df = df[df['source_table'] == 'atlas_cases']
    print(f"ATLAS cases shape: {atlas_df.shape}")
    
    # Check potential technique columns
    tech_cols = [c for c in df.columns if 'technique' in c.lower()]
    print(f"Columns containing 'technique': {tech_cols}")
    
    # Check which rows have techniques
    for col in tech_cols:
        non_null = df[df[col].notna()]
        if not non_null.empty:
            print(f"Column '{col}' has {len(non_null)} non-null values.")
            print(f"Source tables for '{col}': {non_null['source_table'].unique()}")
            print(f"Sample '{col}': {non_null[col].iloc[0]}")

    # Check for case_id
    case_cols = [c for c in df.columns if 'case' in c.lower()]
    print(f"Columns containing 'case': {case_cols}")
    
    # Specifically check if atlas_cases has case_id and if it's numeric or string
    if 'case_id' in atlas_df.columns:
        print("Sample case_id from atlas_cases:", atlas_df['case_id'].head().tolist())
        
except Exception as e:
    print(f"Error: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading astalabs_discovery_all_data.csv...
Dataset loaded successfully.
ATLAS cases shape: (52, 196)
Columns containing 'technique': ['n_techniques', 'techniques', 'techniques_used', 'technique_count']
Column 'n_techniques' has 52 non-null values.
Source tables for 'n_techniques': <StringArray>
['atlas_cases']
Length: 1, dtype: str
Sample 'n_techniques': 6.0
Column 'techniques' has 52 non-null values.
Source tables for 'techniques': <StringArray>
['atlas_cases']
Length: 1, dtype: str
Sample 'techniques': {{acquire_ml_artifacts_data.id}}|{{craft_adv_manual.id}}|{{evade_model.id}}|{{train_proxy_model.id}}|{{verify_attack.id}}|{{victim_research_preprint.id}}
Column 'techniques_used' has 52 non-null values.
Source tables for 'techniques_used': <StringArray>
['step3_incident_coding']
Length: 1, dtype: str
Sample 'techniques_used': AML.T0000.001; AML.T0002.000; AML.T0005; AML.T0015; AML.T0042; AML.T0043.003
Column 'technique_count' has 52 non-null values.
Source tables for 'technique_count': <StringArray>
['step3_incident_coding']
Length: 1, dtype: str
Sample 'technique_count': 6.0
Columns containing 'case': ['2_use_case_name', 'case_id', 'case_study_id']
Sample case_id from atlas_cases: ['AML.CS0000', 'AML.CS0001', 'AML.CS0002', 'AML.CS0003', 'AML.CS0004']

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
