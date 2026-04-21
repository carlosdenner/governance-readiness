# Experiment 100: node_5_23

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_23` |
| **ID in Run** | 100 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:50:11.909375+00:00 |
| **Runtime** | 228.3s |
| **Parent** | `node_4_7` |
| **Children** | None |
| **Creation Index** | 101 |

---

## Hypothesis

> The Adversarial Blind Spot: In ATLAS case studies, 'Exfiltration' tactics
(stealing model/data) are associated with significantly fewer documented
'Mitigations' than 'Evasion' tactics (fooling the model), indicating a focus on
model performance over asset security.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.7027 (Likely True) |
| **Surprise** | -0.0471 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
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
| Maybe True | 0.0 |
| Uncertain | 6.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Analyze the ATLAS dataset to find gaps in defensive coverage.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `step3_incident_coding` (which maps tactics to mitigations) or `atlas_cases` joined with mitigation data.
- 2. Group records by `tactics` (e.g., 'Evasion', 'Exfiltration', 'Inference').
- 3. Count the number of unique `mitigation_id` or `mitigation_name` entries associated with each tactic group.
- 4. Normalize by the number of cases per tactic if necessary.
- 5. Compare the mean mitigations per tactic type using a bar chart and variance analysis.

### Deliverables
- 1. Summary table of Mitigation Counts by Tactic.
- 2. Analysis of the ratio of Defenses-to-Attacks per category.
- 3. Bar chart showing the defensive coverage gap.

---

## Analysis

The code will load the `atlas_cases` data, which contains pre-calculated counts
of mitigations per case (`n_mitigations`) and a list of tactics involved
(`tactics`). By parsing the `tactics` column for keywords 'exfiltration' and
'evasion', we can categorize the cases. We will then compare the average number
of mitigations documented for cases involving Exfiltration versus those
involving Evasion to test the hypothesis that Exfiltration is an 'Adversarial
Blind Spot' with fewer defenses.

---

## Review

The previous steps successfully identified that the `atlas_cases` subset
contains the necessary columns: `tactics` (containing the tactic identifiers)
and `n_mitigations` (containing the count of defenses). The next logical step is
to perform the statistical comparison defined in the hypothesis: comparing the
average number of mitigations for cases involving 'Exfiltration' vs. 'Evasion'.

---

## Code

```python
# [debug]
import pandas as pd

ds_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(ds_path, low_memory=False)
    
    # Filter for atlas_cases
    atlas_df = df[df['source_table'] == 'atlas_cases']
    
    # Find non-null columns in atlas_df
    non_null_cols = atlas_df.dropna(axis=1, how='all').columns.tolist()
    print(f"Active columns in atlas_cases: {non_null_cols}")
    
    # Check if there is a column related to mitigations in the active columns
    mitigation_cols = [c for c in non_null_cols if 'mitigation' in c.lower() or 'defense' in c.lower()]
    print(f"Potential mitigation columns: {mitigation_cols}")
    
    # Also check step3_mitigation_gaps to see what it links to
    gaps_df = df[df['source_table'] == 'step3_mitigation_gaps']
    print(f"\n--- step3_mitigation_gaps columns: {gaps_df.dropna(axis=1, how='all').columns.tolist()}")
    print(gaps_df.head(3))

except Exception as e:
    print(f"Error: {e}")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Active columns in atlas_cases: ['source_table', 'source_row_num', 'case_id', 'name', 'summary', 'n_tactics', 'n_techniques', 'n_mitigations', 'tactics', 'techniques']
Potential mitigation columns: ['n_mitigations']

--- step3_mitigation_gaps columns: ['source_table', 'source_row_num', 'mitigation_id', 'mitigation_name', 'category', 'incident_count', 'competency_gap']
               source_table  source_row_num  ... evidence recommendation
6620  step3_mitigation_gaps               1  ...      NaN            NaN
6621  step3_mitigation_gaps               2  ...      NaN            NaN
6622  step3_mitigation_gaps               3  ...      NaN            NaN

[3 rows x 196 columns]

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
