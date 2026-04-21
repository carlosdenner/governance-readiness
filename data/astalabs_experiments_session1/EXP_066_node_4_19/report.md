# Experiment 66: node_4_19

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_19` |
| **ID in Run** | 66 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:37:51.377724+00:00 |
| **Runtime** | 278.9s |
| **Parent** | `node_3_8` |
| **Children** | `node_5_17` |
| **Creation Index** | 67 |

---

## Hypothesis

> The architecture controls 'Audit Logging & Telemetry' and 'Incident Response &
Recovery Playbooks' exhibit a strong positive correlation in their co-occurrence
within governance requirements, suggesting they are treated as a coupled
'Reactive Defense' construct by standard bodies.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9677 (Definitely True) |
| **Posterior** | 0.2479 (Likely False) |
| **Surprise** | -0.8354 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 28.0 |
| Maybe True | 2.0 |
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
| Definitely False | 90.0 |

---

## Experiment Plan

**Objective:** Assess the statistical coupling between 'Audit Logging & Telemetry' and 'Incident Response & Recovery Playbooks' using robust methods for sparse data.

### Steps
- 1. Load the dataset 'step2_crosswalk_matrix.csv' from the current directory.
- 2. Extract the columns corresponding to 'Audit Logging & Telemetry' and 'Incident Response & Recovery Playbooks'.
- 3. Preprocess the data by converting 'X' (or non-null values) to 1 and empty/null values to 0.
- 4. Create a 2x2 contingency table of their co-occurrence.
- 5. Perform Fisher's Exact Test (two-tailed) to test for independence, as Chi-Square is inappropriate for the observed low frequencies (counts < 5).
- 6. Calculate the Jaccard Index (Intersection over Union) to quantify the magnitude of overlap.

### Deliverables
- Contingency table, Fisher's Exact Test results (p-value, Odds Ratio), and Jaccard Index.

---

## Analysis

The experiment successfully loaded the 'step2_crosswalk_matrix.csv' dataset and
tested the coupling between 'Audit Logging & Telemetry' and 'Incident Response &
Recovery Playbooks'. The results contradict the hypothesis of a strong positive
correlation. The contingency table showed zero co-occurrences (Intersection=0),
with 'Incident Response' appearing twice and 'Audit Logging' appearing once in
isolation. The Jaccard Index was 0.0000, and the Fisher's Exact Test yielded a
p-value of 1.0000, indicating no statistical association between the two
controls. This suggests that within the governance frameworks analyzed, these
reactive defense controls are mapped to distinct, non-overlapping requirements
rather than being treated as a unified construct.

---

## Review

The experiment pipeline was faithfully implemented and the hypothesis was tested
sufficiently. All steps, from data loading to statistical analysis (Fisher's
Exact Test and Jaccard Index), were executed without errors. The results provide
clear evidence to reject the hypothesis.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import sys

def run_experiment():
    # 1. Load the dataset
    file_path = 'step2_crosswalk_matrix.csv'
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    # 2. Extract and Preprocess Columns
    col_audit = 'Audit Logging & Telemetry'
    col_ir = 'Incident Response & Recovery Playbooks'

    if col_audit not in df.columns or col_ir not in df.columns:
        print(f"Error: Columns '{col_audit}' or '{col_ir}' not found.")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Binarize: 'X' -> 1, others -> 0
    # Using fillna('') to handle NaNs safely before string manipulation
    audit_binary = df[col_audit].fillna('').astype(str).str.strip().str.upper().apply(lambda x: 1 if x == 'X' else 0)
    ir_binary = df[col_ir].fillna('').astype(str).str.strip().str.upper().apply(lambda x: 1 if x == 'X' else 0)

    # 3. Create Contingency Table
    # Format: [[Both Absent (0,0), IR Present (0,1)], [Audit Present (1,0), Both Present (1,1)]]
    # However, crosstab default is index=row_var, columns=col_var
    contingency_table = pd.crosstab(audit_binary, ir_binary)
    
    # Ensure 2x2 shape even if some combinations are missing (e.g. if no 1s exist)
    contingency_filled = contingency_table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    
    print("\nContingency Table:")
    print("Rows: Audit Logging & Telemetry (0, 1)")
    print("Cols: Incident Response & Recovery Playbooks (0, 1)")
    print(contingency_filled)

    # Extract values for clarity
    # n00 = neither
    # n01 = IR only
    # n10 = Audit only
    # n11 = Both
    n00 = contingency_filled.loc[0, 0]
    n01 = contingency_filled.loc[0, 1]
    n10 = contingency_filled.loc[1, 0]
    n11 = contingency_filled.loc[1, 1]

    print(f"\nCounts: Neither={n00}, IR_only={n01}, Audit_only={n10}, Both={n11}")

    # 4. Fisher's Exact Test
    # We use the table [[n00, n01], [n10, n11]]
    # Note: Structure affects Odds Ratio direction, but p-value remains same for independence test.
    odds_ratio, p_value = stats.fisher_exact(contingency_filled)

    print("\nFisher's Exact Test Results:")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")

    # 5. Jaccard Index
    # J = (Intersection) / (Union) = n11 / (n10 + n01 + n11)
    union_count = n10 + n01 + n11
    if union_count == 0:
        jaccard = 0.0
    else:
        jaccard = n11 / union_count

    print(f"\nJaccard Index (Intersection over Union): {jaccard:.4f}")

    # Conclusion
    print("\nConclusion:")
    if p_value < 0.05:
        print("Reject Null: There is a statistically significant association.")
    else:
        print("Fail to Reject Null: No statistically significant association found.")

    if jaccard > 0.5:
        print("Overlap: High")
    elif jaccard > 0.2:
        print("Overlap: Moderate")
    else:
        print("Overlap: Low/Negligible")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Successfully loaded step2_crosswalk_matrix.csv. Shape: (42, 24)

Contingency Table:
Rows: Audit Logging & Telemetry (0, 1)
Cols: Incident Response & Recovery Playbooks (0, 1)
Incident Response & Recovery Playbooks   0  1
Audit Logging & Telemetry                    
0                                       39  2
1                                        1  0

Counts: Neither=39, IR_only=2, Audit_only=1, Both=0

Fisher's Exact Test Results:
Odds Ratio: 0.0000
P-value: 1.0000

Jaccard Index (Intersection over Union): 0.0000

Conclusion:
Fail to Reject Null: No statistically significant association found.
Overlap: Low/Negligible

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
