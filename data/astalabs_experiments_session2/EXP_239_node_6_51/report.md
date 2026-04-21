# Experiment 239: node_6_51

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_51` |
| **ID in Run** | 239 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T12:15:40.071156+00:00 |
| **Runtime** | 224.8s |
| **Parent** | `node_5_22` |
| **Children** | None |
| **Creation Index** | 240 |

---

## Hypothesis

> Assessment-Monitoring Decoupling: In federal AI deployments (EO13960), there is
no significant positive correlation between performing an 'Impact Assessment'
and conducting 'Post-Deployment Monitoring', suggesting that governance checks
are often one-off compliance events rather than continuous lifecycle management.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.2672 (Likely False) |
| **Surprise** | -0.5793 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 29.0 |
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
| Definitely False | 56.0 |

---

## Experiment Plan

**Objective:** Test the correlation between pre-deployment impact assessment and post-deployment monitoring.

### Steps
- 1. Filter for `source_table` = 'eo13960_scored'.
- 2. Clean `52_impact_assessment` and `56_monitor_postdeploy` into binary variables (1=Yes/Evidence, 0=No/Missing) using heuristic text analysis (length > 3 or keywords).
- 3. Create a contingency table (Assessment vs. Monitoring).
- 4. Perform a Chi-square test and calculate Pearson's Phi coefficient (or Cramer's V) for effect size.

### Deliverables
- Contingency table, Chi-square stats, and Phi coefficient.

---

## Analysis

The experiment successfully analyzed the relationship between 'Impact
Assessment' and 'Post-Deployment Monitoring' within the EO 13960 dataset
(N=1757). The code correctly parsed the sparse text fields into binary variables
using heuristic text analysis. The results show that while overall compliance is
low (5.2% for assessments, 6.3% for monitoring), there is a very strong,
statistically significant positive correlation between the two activities (Phi
Coefficient = 0.72, p < 1e-196). The contingency table reveals that of the 92
systems with impact assessments, 74 (80%) also had post-deployment monitoring,
whereas only 36 (2.2%) of the 1,665 systems without assessments had monitoring.
These findings strongly refute the 'Assessment-Monitoring Decoupling'
hypothesis, suggesting instead that governance controls are highly clustered:
agencies either apply a comprehensive governance suite or (more commonly) apply
none at all.

---

## Review

The experiment was successfully executed and faithfully implemented the analysis
plan. The code correctly handled the data loading (resolving the initial path
error) and applied appropriate heuristic text analysis to parse the sparse
binary variables from the EO 13960 dataset.

**Hypothesis:** The experiment tested the 'Assessment-Monitoring Decoupling'
hypothesis, which posited that pre-deployment 'Impact Assessments' and 'Post-
Deployment Monitoring' are not significantly correlated in federal AI
deployments.

**Results:** The analysis of 1,757 AI systems revealed a very strong,
statistically significant positive correlation (Phi = 0.72, p < 1e-196). The
contingency table showed that 80% (74/92) of systems with an impact assessment
also had post-deployment monitoring, whereas only 2.2% (36/1665) of systems
without an assessment had monitoring.

**Findings:** The hypothesis is definitively rejected. The results indicate that
governance controls are highly 'clustered' rather than decoupled. Agencies tend
to either implement a comprehensive governance suite (both assessment and
monitoring) or apply minimal controls. The presence of a pre-deployment
assessment is a strong predictor of post-deployment monitoring.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import os

def clean_binary_text(val):
    if pd.isna(val):
        return 0
    text = str(val).strip().lower()
    if not text:
        return 0
    
    # List of terms indicating absence of the control
    negatives = [
        'no', 'none', 'n/a', 'not applicable', 'tbd', 'unknown', 
        'false', '0', 'nan', 'not established', 'not currently'
    ]
    if text in negatives:
        return 0
    
    # explicit negative phrases
    if text.startswith('no ') or text.startswith('not '):
        return 0
        
    # Default to 1 (presence of evidence/description)
    return 1

def main():
    filename = 'astalabs_discovery_all_data.csv'
    
    # specific check for file existence
    if not os.path.exists(filename):
        # fallback to parent if current not found, though previous error suggests parent is wrong
        if os.path.exists(f'../{filename}'):
            filename = f'../{filename}'
        else:
            print(f"Error: {filename} not found in current or parent directory.")
            return
            
    print(f"Loading dataset from {filename}...")
    try:
        df = pd.read_csv(filename, low_memory=False)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # Filter for EO 13960 Scored
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"Filtered EO 13960 data: {len(df_eo)} rows")

    # Columns
    col_impact = '52_impact_assessment'
    col_monitor = '56_monitor_postdeploy'
    
    # Apply binary classification
    df_eo['has_impact_assessment'] = df_eo[col_impact].apply(clean_binary_text)
    df_eo['has_monitoring'] = df_eo[col_monitor].apply(clean_binary_text)
    
    # Descriptive stats
    print(f"\nImpact Assessment (1=Yes, 0=No):\n{df_eo['has_impact_assessment'].value_counts()}")
    print(f"\nPost-Deployment Monitoring (1=Yes, 0=No):\n{df_eo['has_monitoring'].value_counts()}")
    
    # Contingency Table
    contingency = pd.crosstab(df_eo['has_impact_assessment'], df_eo['has_monitoring'])
    print("\nContingency Table (Rows: Assessment, Cols: Monitoring):")
    print(contingency)
    
    # Check if we have a valid 2x2
    if contingency.shape != (2, 2):
        print("\nWarning: Contingency table is not 2x2 (likely missing 0s or 1s in one dimension). Stats may be degenerate.")

    # Statistics
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    n = contingency.sum().sum()
    
    # Pearson correlation (equivalent to Phi for binary variables, preserving sign)
    correlation = df_eo['has_impact_assessment'].corr(df_eo['has_monitoring'])
    
    print(f"\n--- Statistics ---")
    print(f"Chi-Square: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    print(f"Phi Coefficient (Pearson r): {correlation:.4f}")
    
    # Interpretation
    print("\n--- Conclusion ---")
    if p < 0.05:
        if correlation > 0.1:
            print("Result: Significant POSITIVE correlation. (Hypothesis Rejected)")
            print("Interpretation: Agencies performing impact assessments ARE likely to perform monitoring.")
        elif correlation < -0.1:
            print("Result: Significant NEGATIVE correlation. (Hypothesis Supported)")
            print("Interpretation: Inverse relationship found.")
        else:
            print("Result: Statistically significant but negligible effect size.")
    else:
        print("Result: No significant correlation found. (Hypothesis Supported)")
        print("Interpretation: Assessment and Monitoring appear to be decoupled.")

if __name__ == "__main__":
    main()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
Filtered EO 13960 data: 1757 rows

Impact Assessment (1=Yes, 0=No):
has_impact_assessment
0    1665
1      92
Name: count, dtype: int64

Post-Deployment Monitoring (1=Yes, 0=No):
has_monitoring
0    1647
1     110
Name: count, dtype: int64

Contingency Table (Rows: Assessment, Cols: Monitoring):
has_monitoring            0   1
has_impact_assessment          
0                      1629  36
1                        18  74

--- Statistics ---
Chi-Square: 896.8497
P-value: 4.7497e-197
Phi Coefficient (Pearson r): 0.7197

--- Conclusion ---
Result: Significant POSITIVE correlation. (Hypothesis Rejected)
Interpretation: Agencies performing impact assessments ARE likely to perform monitoring.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
