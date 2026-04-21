# Experiment 59: node_4_24

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_24` |
| **ID in Run** | 59 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:52:08.097996+00:00 |
| **Runtime** | 239.0s |
| **Parent** | `node_3_20` |
| **Children** | `node_5_11`, `node_5_46` |
| **Creation Index** | 60 |

---

## Hypothesis

> The 'Testing-Deployment' Decay: Operational AI systems (Stage: 'Operation and
Maintenance') have significantly lower rates of documented 'Real World Testing'
compared to systems in the 'Development' phase, suggesting that testing evidence
is not maintained post-deployment.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.2610 (Likely False) |
| **Surprise** | -0.5868 |
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
| Maybe False | 2.0 |
| Definitely False | 58.0 |

---

## Experiment Plan

**Objective:** Determine if testing documentation is sustained throughout the AI lifecycle by comparing testing rates between developmental and operational systems.

### Steps
- 1. Load the dataset 'astalabs_discovery_all_data.csv' (ensure correct path in current directory).
- 2. Filter the dataframe to include only rows where `source_table` is 'eo13960_scored'.
- 3. Create a new categorization column 'lifecycle_phase' based on `16_dev_stage`: map values containing 'Operation' or 'Maintenance' to 'Operational', and values containing 'Development', 'Implementation', or 'Planning' to 'Development'. Exclude other stages if necessary.
- 4. Create a binary column 'has_testing_evidence' based on `53_real_world_testing`: set to 1 if the value is explicitly 'Yes' (case-insensitive), else 0.
- 5. Calculate the percentage of systems with testing evidence for both 'Operational' and 'Development' groups.
- 6. Perform a Chi-square test of independence on the contingency table of `lifecycle_phase` vs `has_testing_evidence`.
- 7. Print the contingency table, the rates per group, and the Chi-square test results (statistic and p-value).

### Deliverables
- 1. Percentage of systems with documented 'Real World Testing' for Operational vs. Development phases.
- 2. Chi-square test results indicating if the difference is statistically significant.

---

## Analysis

The experiment successfully tested the 'Testing-Deployment Decay' hypothesis
using the EO 13960 dataset. The code correctly loaded the data, segmented the
lifecycle stages into 'Development' (n=646) and 'Operational' (n=627), and
applied the strict criteria for testing evidence (looking for the explicit
string 'Yes').

The analysis revealed that contrary to the hypothesis, 'Operational' systems
were significantly *more* likely to have explicit 'Yes' documentation for real-
world testing (3.51%, 22 systems) compared to 'Development' systems, which had
zero instances (0.00%) of the explicit 'Yes' value. The Chi-square test (p <
0.001) confirmed this difference was statistically significant.

It is worth noting from the raw value counts that many systems use descriptive
labels (e.g., 'Performance evaluation in operational environment') instead of a
simple 'Yes'. However, under the strict boolean criteria defined in the
experiment plan, the hypothesis that operational systems have *lower* rates of
documented testing was rejected.

---

## Review

The experiment was faithfully implemented according to the plan. The code
correctly loaded the dataset, handled the lifecycle categorization, and applied
the strict criteria for identifying testing evidence. The statistical analysis
was appropriate for the categorical data.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

# Attempt to locate the file
filename = 'astalabs_discovery_all_data.csv'
possible_paths = [filename, '../' + filename, '/content/' + filename]
data_path = None

for path in possible_paths:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    # If not found, try to list directories to debug, but for now assuming it's in current based on Exp 1
    data_path = filename

print(f"Loading dataset from: {data_path}")

try:
    df = pd.read_csv(data_path, low_memory=False)
    
    # Filter for EO13960 scored data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO13960 subset shape: {df_eo.shape}")

    # Check columns
    stage_col = '16_dev_stage'
    test_col = '53_real_world_testing'
    
    if stage_col not in df_eo.columns or test_col not in df_eo.columns:
        print(f"Columns missing. Available: {df_eo.columns.tolist()[:10]}...")
    else:
        # Clean and Normalize Stage
        # Inspect unique values first
        print("\n--- Raw Lifecycle Stages ---")
        print(df_eo[stage_col].value_counts(dropna=False))
        
        def map_lifecycle(stage):
            s = str(stage).lower()
            if 'operation' in s or 'maintenance' in s:
                return 'Operational'
            elif 'development' in s or 'implementation' in s or 'plan' in s:
                return 'Development'
            else:
                return 'Other'
        
        df_eo['lifecycle_group'] = df_eo[stage_col].apply(map_lifecycle)
        
        # Filter for only Operational and Development
        df_analysis = df_eo[df_eo['lifecycle_group'].isin(['Operational', 'Development'])].copy()
        
        print("\n--- Analysis Groups ---")
        print(df_analysis['lifecycle_group'].value_counts())
        
        # Clean and Normalize Testing Evidence
        print("\n--- Raw Testing Values ---")
        print(df_analysis[test_col].value_counts(dropna=False).head(10))
        
        # Strict 'Yes' criteria for evidence. Anything else (No, N/A, blank) is treated as lack of positive evidence.
        df_analysis['has_evidence'] = df_analysis[test_col].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
        
        # Calculate Rates
        results = df_analysis.groupby('lifecycle_group')['has_evidence'].agg(['count', 'sum', 'mean'])
        results['pct'] = results['mean'] * 100
        
        print("\n--- Testing Documentation Rates ---")
        print(results)
        
        # Statistical Test (Chi-Square)
        # Contingency Table
        contingency = pd.crosstab(df_analysis['lifecycle_group'], df_analysis['has_evidence'])
        print("\n--- Contingency Table (0=No Evidence, 1=Yes Evidence) ---")
        print(contingency)
        
        chi2, p, dof, expected = chi2_contingency(contingency)
        
        print(f"\nChi-Square Statistic: {chi2:.4f}")
        print(f"P-Value: {p:.4e}")
        
        # Hypothesis Check
        op_rate = results.loc['Operational', 'mean']
        dev_rate = results.loc['Development', 'mean']
        
        print(f"\nOperational Rate: {op_rate:.2%}")
        print(f"Development Rate: {dev_rate:.2%}")
        
        if p < 0.05:
            print("Result: Statistically significant difference.")
            if op_rate < dev_rate:
                print("Hypothesis SUPPORTED: Operational systems have significantly lower documentation rates.")
            else:
                print("Hypothesis REJECTED: Operational systems do not have lower rates (direction inverted).")
        else:
            print("Result: No statistically significant difference.")
            print("Hypothesis REJECTED.")
            
except Exception as e:
    print(f"Execution failed: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: astalabs_discovery_all_data.csv
EO13960 subset shape: (1757, 196)

--- Raw Lifecycle Stages ---
16_dev_stage
Operation and Maintenance         627
Acquisition and/or Development    351
Initiated                         329
Implementation and Assessment     275
Retired                           133
Planned                            20
In production                      14
In mission                          4
NaN                                 4
Name: count, dtype: int64

--- Analysis Groups ---
lifecycle_group
Development    646
Operational    627
Name: count, dtype: int64

--- Raw Testing Values ---
53_real_world_testing
NaN                                                                                                                                                                                                                                                                                                                                                                                                              1147
Performance evaluation in operational environment: The AI use case has been tested in an operational environment before being fully implemented as a solution, or has been tested in simulated or controlled environment using operational or synthetic data.                                                                                                                                                      49
Benchmark evaluation: Testing of the AI model has involved the use of benchmarks (either publicly available or internally created) to estimate performance in real-world settings, but has not been tested in an operational environment.                                                                                                                                                                          24
Yes                                                                                                                                                                                                                                                                                                                                                                                                                22
Impact evaluation in operational environment: The AI use case has been tested in an operational environment before being fully implemented as a solution and has utilized randomized experiments with a control group or other counterfactual, social systems analysis, or other rigorous research methodologies to evaluate impact and identify potential harm to users as well as broader groups of people.      15
No testing: No testing of the model to simulate performance in an operational environment has been conducted thus far.                                                                                                                                                                                                                                                                                             15
Agency CAIO has waived this minimum practice and reported such waiver to OMB.                                                                                                                                                                                                                                                                                                                                       1
Name: count, dtype: int64

--- Testing Documentation Rates ---
                 count  sum      mean       pct
lifecycle_group                                
Development        646    0  0.000000  0.000000
Operational        627   22  0.035088  3.508772

--- Contingency Table (0=No Evidence, 1=Yes Evidence) ---
has_evidence       0   1
lifecycle_group         
Development      646   0
Operational      605  22

Chi-Square Statistic: 21.0455
P-Value: 4.4850e-06

Operational Rate: 3.51%
Development Rate: 0.00%
Result: Statistically significant difference.
Hypothesis REJECTED: Operational systems do not have lower rates (direction inverted).

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
