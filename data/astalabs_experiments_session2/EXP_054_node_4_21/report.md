# Experiment 54: node_4_21

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_21` |
| **ID in Run** | 54 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:34:53.180098+00:00 |
| **Runtime** | 228.6s |
| **Parent** | `node_3_10` |
| **Children** | `node_5_12`, `node_5_60`, `node_5_87` |
| **Creation Index** | 55 |

---

## Hypothesis

> Operational Complacency: AI systems currently in the 'Operation' stage are
statistically less likely to have documented 'Real World Testing' compared to
systems still in the 'Development' stage, reflecting a 'deploy-and-forget'
governance gap.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2527 (Likely False) |
| **Surprise** | -0.5870 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
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
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Compare the rates of Real World Testing compliance between Operational and Developmental AI systems in the federal inventory.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `source_table` = 'eo13960_scored'.
- 2. Create a binary variable `is_operational`: map `16_dev_stage` values containing 'Operation' or 'Use' to 1, and 'Development'/'Testing' to 0 (exclude Retired/Other).
- 3. Create a binary variable `has_testing`: map `53_real_world_testing` ('Yes' -> 1, Others -> 0).
- 4. Perform a Chi-square test of independence between `is_operational` and `has_testing`.
- 5. Calculate the compliance percentage for both groups.

### Deliverables
- Contingency table, Chi-square p-value, and comparative compliance percentages.

---

## Analysis

The experiment successfully tested the 'Operational Complacency' hypothesis
using the EO 13960 dataset. After robustly cleaning the data to handle verbose
text descriptions in the '53_real_world_testing' column and classifying 1,616
systems into 'Operational' (n=916) or 'Development' (n=700) stages, the analysis
found a statistically significant difference in governance compliance (Chi-
Square=53.66, p<0.001). Contrary to the hypothesis that operational systems
would exhibit 'deploy-and-forget' negligence, they actually demonstrated a
significantly higher rate of documented Real World Testing (8.95%) compared to
systems in development (0.57%). This result refutes the hypothesis, suggesting
that while overall compliance is low, the specific control of 'Real World
Testing' is primarily applied or documented only after systems have matured
beyond the initial development phases.

---

## Review

The experiment was successfully executed and the code faithfully implemented the
analysis plan. After correcting for data quality issues (handling verbose text
descriptions and mixed types), the analysis successfully classified 1,616
federal AI systems into 'Operational' (n=916) and 'Development' (n=700) stages.

**Hypothesis Test Results:**
1.  **Compliance Rates:**
    -   **Operational Systems:** 8.95% (82/916) documented Real World Testing.
    -   **Development Systems:** 0.57% (4/700) documented Real World Testing.
2.  **Statistical Test:**
    -   Chi-Square Statistic: 53.66
    -   p-value: < 0.0001 (Highly Significant)

**Conclusion:**
The hypothesis that operational systems suffer from 'deploy-and-forget'
complacency (lower testing rates) is **rejected**. The data reveals a strong,
statistically significant relationship in the **opposite direction**: systems in
the operational/implementation phase are far more likely to have documented
real-world testing than those in early development. This suggests that 'Real
World Testing' is a control typically applied or documented only as systems
mature into production, rather than being neglected after deployment.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import sys

# [debug] Print library versions if needed
# print(f"Pandas version: {pd.__version__}")

try:
    # Load dataset
    file_path = 'astalabs_discovery_all_data.csv'
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)

    # Filter for EO 13960 data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 records loaded: {len(df_eo)}")

    # Columns
    col_stage = '16_dev_stage'
    col_testing = '53_real_world_testing'

    # --- CLEANING STAGE ---
    # Ensure string type and handle NaNs
    df_eo[col_stage] = df_eo[col_stage].fillna('nan').astype(str).str.lower().str.strip()

    # Logic for is_operational (Independent Variable)
    def map_stage(val):
        # Exclude retired or missing
        if 'retire' in val or 'nan' == val or val == '':
            return None
        
        # Operational keywords
        # 'operation', 'production', 'implementation', 'use', 'maintain'
        if any(x in val for x in ['oper', 'prod', 'impl', 'use', 'maintain', 'sustain']):
            return 'Operational'
        
        # Development keywords
        # 'development', 'initiated', 'planned', 'acquisition'
        elif any(x in val for x in ['dev', 'init', 'plan', 'acq', 'test', 'pilot', 'experiment']):
            return 'Development'
        
        return None # Unclassified

    df_eo['stage_group'] = df_eo[col_stage].apply(map_stage)
    
    # Drop unclassified rows
    df_clean = df_eo.dropna(subset=['stage_group']).copy()
    print(f"\nRecords after stage classification: {len(df_clean)}")
    print("Stage Distribution:")
    print(df_clean['stage_group'].value_counts())

    # --- CLEANING TESTING ---
    # Ensure string type
    df_clean[col_testing] = df_clean[col_testing].fillna('no').astype(str).str.lower().str.strip()

    # Logic for has_testing (Dependent Variable)
    # We look for explicit mentions of 'operational environment' or 'yes'.
    # 'benchmark' explicitly states 'has not been tested in an operational environment', so it is 0.
    def map_testing(val):
        if 'performance evaluation' in val: return 1
        if 'impact evaluation' in val: return 1
        if val == 'yes': return 1
        return 0

    df_clean['has_testing'] = df_clean[col_testing].apply(map_testing)
    
    print("\nTesting Logic Check (sample of raw vs mapped):")
    print(df_clean[[col_testing, 'has_testing']].drop_duplicates().head(10))

    # --- ANALYSIS ---

    # Contingency Table
    contingency_table = pd.crosstab(df_clean['stage_group'], df_clean['has_testing'])
    # Check if we have both 0 and 1 columns. If not, reindex to ensure shape.
    if 0 not in contingency_table.columns: contingency_table[0] = 0
    if 1 not in contingency_table.columns: contingency_table[1] = 0
    contingency_table = contingency_table[[0, 1]]
    contingency_table.columns = ['No Real-World Testing', 'Has Real-World Testing']
    
    print("\nContingency Table (Real World Testing by Stage):")
    print(contingency_table)

    # Calculate Percentages
    summary = df_clean.groupby('stage_group')['has_testing'].agg(['count', 'sum', 'mean'])
    summary['pct_compliant'] = summary['mean'] * 100
    print("\nCompliance Rates (% with Real World Testing):")
    print(summary[['count', 'sum', 'pct_compliant']])

    # Statistical Test
    # Chi-square test of independence
    if contingency_table.sum().sum() > 0:
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\nChi-Square Test Results:")
        print(f"Chi2 Statistic: {chi2:.4f}")
        print(f"p-value: {p:.4f}")
        
        alpha = 0.05
        if p < alpha:
            print("Result: Statistically Significant (Reject Null Hypothesis)")
            op_rate = summary.loc['Operational', 'pct_compliant']
            dev_rate = summary.loc['Development', 'pct_compliant']
            print(f"Operational Rate: {op_rate:.2f}%")
            print(f"Development Rate: {dev_rate:.2f}%")
            
            if op_rate < dev_rate:
                print("Conclusion: Operational systems have LOWER testing compliance. (SUPPORTS Hypothesis)")
            else:
                print("Conclusion: Operational systems have HIGHER testing compliance. (REFUTES Hypothesis)")
        else:
            print("Result: Not Statistically Significant (Fail to Reject Null Hypothesis)")
    else:
        print("\nWarning: Insufficient data for Chi-square test.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
EO 13960 records loaded: 1757

Records after stage classification: 1616
Stage Distribution:
stage_group
Operational    916
Development    700
Name: count, dtype: int64

Testing Logic Check (sample of raw vs mapped):
                                 53_real_world_testing  has_testing
0                                                   no            0
49   performance evaluation in operational environm...            1
108  benchmark evaluation: testing of the ai model ...            0
132  impact evaluation in operational environment: ...            1
184  no testing: no testing of the model to simulat...            0
520                                                yes            1
886  agency caio has waived this minimum practice a...            0

Contingency Table (Real World Testing by Stage):
             No Real-World Testing  Has Real-World Testing
stage_group                                               
Development                    696                       4
Operational                    834                      82

Compliance Rates (% with Real World Testing):
             count  sum  pct_compliant
stage_group                           
Development    700    4       0.571429
Operational    916   82       8.951965

Chi-Square Test Results:
Chi2 Statistic: 53.6573
p-value: 0.0000
Result: Statistically Significant (Reject Null Hypothesis)
Operational Rate: 8.95%
Development Rate: 0.57%
Conclusion: Operational systems have HIGHER testing compliance. (REFUTES Hypothesis)

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
