# Experiment 121: node_6_3

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_3` |
| **ID in Run** | 121 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:48:15.431570+00:00 |
| **Runtime** | 334.1s |
| **Parent** | `node_5_25` |
| **Children** | `node_7_8`, `node_7_9` |
| **Creation Index** | 122 |

---

## Hypothesis

> Sectoral Autonomy Divergence: The 'Transportation' sector suffers significantly
more incidents involving 'High Autonomy' systems compared to the 'Government'
sector, which is dominated by 'Low Autonomy' failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8710 (Likely True) |
| **Posterior** | 0.2967 (Likely False) |
| **Surprise** | -0.6891 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 16.0 |
| Maybe True | 14.0 |
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

**Objective:** Map the risk profile of different sectors regarding AI autonomy levels.

### Steps
- 1. Filter 'aiid_incidents'.
- 2. Filter for rows where '78_Sector of Deployment' is 'Transportation' or 'Government'.
- 3. Map '81_Autonomy Level' to 'High' vs. 'Low'.
- 4. Create a contingency table (Sector vs. Autonomy).
- 5. Perform a Chi-square test to see if Autonomy distributions differ significantly by Sector.

### Deliverables
- Autonomy distribution comparison between Transportation and Government sectors; Test results.

---

## Analysis

The experiment successfully tested the 'Sectoral Autonomy Divergence' hypothesis
using the AIID dataset. After resolving previous column identification and value
mapping errors, the code successfully categorized 39 relevant incidents.

The results statistically **contradicted** the hypothesis. The hypothesis
predicted that the 'Transportation' sector would have a higher prevalence of
'High Autonomy' incidents. However, the Chi-square test ($p \approx 0.009$)
revealed a significant difference in the opposite direction. The 'Government'
sector exhibited a much higher rate of High Autonomy incidents (50.0%, 6/12)
compared to the 'Transportation' sector, where incidents were overwhelmingly Low
Autonomy (92.6%, 25/27). This suggests that while transportation failures in
this dataset are frequent, they typically involve lower-level automation (e.g.,
driver assistance), whereas government failures often involve more autonomous
decision-making systems.

---

## Review

The experiment successfully tested the 'Sectoral Autonomy Divergence' hypothesis
using the AIID dataset after resolving initial data parsing issues
(specifically, handling the 'Autonomy1-5' taxonomy and correctly identifying
column names).

The results statistically **contradicted** the hypothesis. The hypothesis
proposed that the Transportation sector would be more prone to 'High Autonomy'
incidents compared to the Government sector. However, the Chi-square test ($p
\approx 0.009$) indicated a significant difference in the opposite direction.

**Findings:**
- **Government Sector:** Excluded a much higher relative proportion of 'High
Autonomy' failures (50%, 6/12 incidents).
- **Transportation Sector:** Failures were overwhelmingly 'Low Autonomy' (92.6%,
25/27 incidents).

This suggests that while transportation incidents are more frequent in raw
numbers, they typically involve lower-level automation (e.g., driver assistance
or mechanical control systems), whereas government AI failures in this dataset
are significantly more likely to involve fully autonomous decision-making
systems.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
import os

def run_experiment():
    # Load dataset
    file_path = 'astalabs_discovery_all_data.csv'
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        file_path = '../astalabs_discovery_all_data.csv'
        df = pd.read_csv(file_path, low_memory=False)

    # Filter for AIID incidents
    df_incidents = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"AIID Incidents loaded: {len(df_incidents)} rows")

    # Dynamic Column Identification
    cols = df_incidents.columns
    sector_col = next((c for c in cols if 'Sector' in c and 'Deployment' in c), None)
    autonomy_col = next((c for c in cols if 'Autonomy' in c and 'Level' in c), None)

    print(f"Using Sector Column: {sector_col}")
    print(f"Using Autonomy Column: {autonomy_col}")

    if not sector_col or not autonomy_col:
        print("Critical columns missing. Aborting.")
        return

    # 1. Map Sector
    def map_sector(val):
        if pd.isna(val): return None
        val_str = str(val).lower().strip()
        if 'transportation' in val_str:
            return 'Transportation'
        if any(x in val_str for x in ['government', 'public', 'administration']):
            return 'Government'
        return None

    df_incidents['mapped_sector'] = df_incidents[sector_col].apply(map_sector)
    
    # Filter to relevant sectors
    df_filtered = df_incidents[df_incidents['mapped_sector'].notna()].copy()
    print(f"Rows after sector filtering: {len(df_filtered)}")
    print(f"Sector counts:\n{df_filtered['mapped_sector'].value_counts()}")

    # 2. Map Autonomy
    # Mapping based on observed values: ['Autonomy1', 'Autonomy2', 'Autonomy3', 'unclear']
    # Assumption: 1-2 = Low, 3+ = High
    def map_autonomy(val):
        if pd.isna(val): return None
        val_str = str(val).lower().strip()
        
        # Specific dataset tags
        if 'autonomy1' in val_str or 'autonomy2' in val_str:
            return 'Low'
        if 'autonomy3' in val_str or 'autonomy4' in val_str or 'autonomy5' in val_str:
            return 'High'
        
        # Generic text fallback
        if 'high' in val_str: return 'High'
        if 'low' in val_str: return 'Low'
        
        return None

    df_filtered['mapped_autonomy'] = df_filtered[autonomy_col].apply(map_autonomy)
    
    # Filter valid autonomy rows
    df_final = df_filtered[df_filtered['mapped_autonomy'].notna()].copy()
    print(f"Rows after autonomy filtering: {len(df_final)}")
    
    if len(df_final) == 0:
        print("No data available for analysis.")
        return

    # 3. Contingency Table
    contingency = pd.crosstab(df_final['mapped_sector'], df_final['mapped_autonomy'])
    print("\n--- Contingency Table (Sector vs Autonomy) ---")
    print(contingency)

    # 4. Chi-square Test
    # Ensure we have data in the table to run the test
    if contingency.size >= 4:
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print("\n--- Chi-square Test Results ---")
        print(f"Chi-square Statistic: {chi2:.4f}")
        print(f"P-value: {p:.6e}")
        
        # Interpretation
        alpha = 0.05
        if p < alpha:
            print("Result: REJECT Null Hypothesis. Significant difference found.")
            
            # Calculate High Autonomy percentages
            row_props = pd.crosstab(df_final['mapped_sector'], df_final['mapped_autonomy'], normalize='index') * 100
            print("\nProportions (%):")
            print(row_props.round(2))
            
            try:
                trans_high = row_props.loc['Transportation', 'High']
            except KeyError:
                trans_high = 0
            
            try:
                gov_high = row_props.loc['Government', 'High']
            except KeyError:
                gov_high = 0
            
            print(f"\nAnalysis: Transportation High Autonomy Rate: {trans_high:.1f}%")
            print(f"Analysis: Government High Autonomy Rate: {gov_high:.1f}%")
            
            if trans_high > gov_high:
                print("Conclusion: Hypothesis SUPPORTED. Transportation has a higher rate of high-autonomy incidents.")
            else:
                print("Conclusion: Hypothesis CONTRADICTED. Government has a higher rate (or equal) of high-autonomy incidents.")
        else:
            print("Result: FAIL TO REJECT Null Hypothesis. No significant difference in autonomy levels between sectors.")
    else:
        print("Contingency table too small for Chi-square test.")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: AIID Incidents loaded: 1362 rows
Using Sector Column: Sector of Deployment
Using Autonomy Column: Autonomy Level
Rows after sector filtering: 41
Sector counts:
mapped_sector
Transportation    28
Government        13
Name: count, dtype: int64
Rows after autonomy filtering: 39

--- Contingency Table (Sector vs Autonomy) ---
mapped_autonomy  High  Low
mapped_sector             
Government          6    6
Transportation      2   25

--- Chi-square Test Results ---
Chi-square Statistic: 6.8156
P-value: 9.036452e-03
Result: REJECT Null Hypothesis. Significant difference found.

Proportions (%):
mapped_autonomy   High    Low
mapped_sector                
Government       50.00  50.00
Transportation    7.41  92.59

Analysis: Transportation High Autonomy Rate: 7.4%
Analysis: Government High Autonomy Rate: 50.0%
Conclusion: Hypothesis CONTRADICTED. Government has a higher rate (or equal) of high-autonomy incidents.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
