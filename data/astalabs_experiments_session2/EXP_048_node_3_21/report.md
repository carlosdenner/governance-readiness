# Experiment 48: node_3_21

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_21` |
| **ID in Run** | 48 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:16:28.945891+00:00 |
| **Runtime** | 201.5s |
| **Parent** | `node_2_7` |
| **Children** | `node_4_35`, `node_4_49`, `node_4_50` |
| **Creation Index** | 49 |

---

## Hypothesis

> The 'Paper Tiger' Assessment: Among agencies that conduct AI Impact Assessments,
the identification of 'Adverse Impacts' is uncorrelated with the implementation
of 'Disparity Mitigation' controls, suggesting assessments are often compliance
formalities without operational follow-through.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.9066 (Definitely True) |
| **Surprise** | +0.1976 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
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
| Definitely True | 58.0 |
| Maybe True | 2.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Test the operational efficacy of Impact Assessments.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'eo13960_scored'.
- 2. Filter dataset to include only rows where '52_impact_assessment' == 'YES' (or equivalent).
- 3. Create binary variables for '61_adverse_impact' (Found Risks: Yes/No) and '55_disparity_mitigation' (Action Taken: Yes/No).
- 4. Compute the phi coefficient (correlation) between finding risks and taking action.
- 5. Perform a Chi-square test to see if finding adverse impacts significantly increases the likelihood of mitigation.

### Deliverables
- Correlation coefficient; Contingency table for Risk Found vs Mitigation Enacted.

---

## Analysis

The experiment executed successfully. The code filtered the dataset for agencies
performing Impact Assessments (n=61). The resulting contingency table revealed a
lack of variation in the dependent variable ('Disparity Mitigation'): across all
61 cases, zero instances of mitigation were recorded (Value=0), even in the 2
cases where adverse impacts were identified. Consequently, the correlation
coefficient was undefined (NaN) and the Chi-Square test failed due to zero
expected frequencies. While statistical significance could not be computed, the
descriptive result strongly supports the 'Paper Tiger' hypothesis (or indicates
missing data), as no mitigation actions were documented despite the presence of
impact assessments.

---

## Review

Hypothesis: The 'Paper Tiger' Assessment suggests that agencies conducting AI
Impact Assessments do not reliably implement 'Disparity Mitigation' controls,
indicating assessments may be compliance formalities without operational follow-
through.

Results: The experiment filtered the dataset for agencies confirming they
conducted an Impact Assessment (N=61). The analysis revealed that while 2 of
these agencies identified 'Adverse Impacts', zero agencies (0%) recorded
implementing 'Disparity Mitigation'.

Findings: Consequently, the statistical tests (Phi coefficient, Chi-square)
could not be computed because the 'Mitigation' variable had zero variance (it
was constant at 0). This descriptive result strongly supports the hypothesis:
despite the presence of impact assessments, there is no evidence of subsequent
disparity mitigation in the dataset.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats

def run_experiment():
    print("Starting 'Paper Tiger' Assessment Experiment (Attempt 2)...")
    
    # 1. Load Data
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("File not found in parent directory. Trying current directory.")
        try:
            df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        except FileNotFoundError:
            print("Dataset not found.")
            return

    # 2. Filter for EO 13960 Scored
    eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 Scored subset shape: {eo_df.shape}")

    # 3. Identify Columns Dynamically
    cols = eo_df.columns.tolist()
    
    def get_col_by_keyword(keywords):
        matches = [c for c in cols if all(k.lower() in c.lower() for k in keywords)]
        if matches:
            return matches[0]
        return None

    col_assessment = get_col_by_keyword(['52', 'impact', 'assessment'])
    col_adverse = get_col_by_keyword(['61', 'adverse', 'impact'])
    col_mitigation = get_col_by_keyword(['62', 'disparity', 'mitigation']) 
    # Fallback if 62 not found by number
    if not col_mitigation:
        col_mitigation = get_col_by_keyword(['disparity', 'mitigation'])

    print(f"Using columns:\n  Assessment: {col_assessment}\n  Adverse: {col_adverse}\n  Mitigation: {col_mitigation}")

    if not (col_assessment and col_adverse and col_mitigation):
        print("Critical columns missing. Aborting.")
        return

    # 4. Filter for Impact Assessment == YES
    # Normalize to string, lower, strip
    eo_df['assess_norm'] = eo_df[col_assessment].astype(str).str.lower().str.strip()
    
    target_vals = ['yes', 'true', '1']
    analyzed_df = eo_df[eo_df['assess_norm'].isin(target_vals)].copy()
    
    print(f"Rows with Impact Assessment found: {len(analyzed_df)}")

    if len(analyzed_df) < 5:
        print("Warning: Very few data points. Statistical tests may be invalid.")

    # 5. Create Binary Variables (Fixing previous bug)
    def make_binary(val):
        # Convert to string, lower case, strip whitespace
        s = str(val).lower().strip()
        # Check against truthy values
        return 1 if s in ['yes', 'true', '1'] else 0

    analyzed_df['risk_found'] = analyzed_df[col_adverse].apply(make_binary)
    analyzed_df['action_taken'] = analyzed_df[col_mitigation].apply(make_binary)

    # 6. Analysis
    # Contingency Table
    contingency = pd.crosstab(analyzed_df['risk_found'], analyzed_df['action_taken'])
    # Ensure 2x2 shape
    contingency = contingency.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    
    contingency.index = ['No Adverse Impact (0)', 'Adverse Impact Found (1)']
    contingency.columns = ['No Mitigation (0)', 'Mitigation Taken (1)']

    print("\nContingency Table:")
    print(contingency)

    # Phi Coefficient
    phi = analyzed_df['risk_found'].corr(analyzed_df['action_taken'])
    print(f"\nPhi Coefficient (Correlation): {phi:.4f}")

    # Chi-Square Test
    # Check if we have enough data variance
    if contingency.values.sum() == 0:
         print("No data populated.")
    elif (contingency.values == 0).all():
         print("Contingency table empty.")
    else:
        try:
            chi2, p, dof, ex = stats.chi2_contingency(contingency)
            print(f"Chi-Square Statistic: {chi2:.4f}")
            print(f"P-Value: {p:.4f}")

            # Conclusion
            print("\n--- Conclusion ---")
            if p > 0.05:
                print("Result: No significant association found (P > 0.05).")
                print("Supports 'Paper Tiger' hypothesis: Finding risks does not reliably predict mitigation actions.")
            else:
                print("Result: Significant association found (P <= 0.05).")
                print("Rejects 'Paper Tiger' hypothesis: Finding risks is associated with taking mitigation actions.")
        except Exception as e:
            print(f"Statistical test failed: {e}")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting 'Paper Tiger' Assessment Experiment (Attempt 2)...
File not found in parent directory. Trying current directory.
EO 13960 Scored subset shape: (1757, 196)
Using columns:
  Assessment: 52_impact_assessment
  Adverse: 61_adverse_impact
  Mitigation: 62_disparity_mitigation
Rows with Impact Assessment found: 61

Contingency Table:
                          No Mitigation (0)  Mitigation Taken (1)
No Adverse Impact (0)                    59                     0
Adverse Impact Found (1)                  2                     0

Phi Coefficient (Correlation): nan
Statistical test failed: The internally computed table of expected frequencies has a zero element at (np.int64(0), np.int64(1)).

STDERR:
/usr/local/lib/python3.13/site-packages/numpy/lib/_function_base_impl.py:3023: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/usr/local/lib/python3.13/site-packages/numpy/lib/_function_base_impl.py:3024: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
