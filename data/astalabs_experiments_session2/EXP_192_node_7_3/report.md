# Experiment 192: node_7_3

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_3` |
| **ID in Run** | 192 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:03:57.947035+00:00 |
| **Runtime** | 384.5s |
| **Parent** | `node_6_2` |
| **Children** | None |
| **Creation Index** | 193 |

---

## Hypothesis

> Sector-Specific Bias Risk: The 'Finance' and 'Public Administration' sectors
exhibit a significantly higher ratio of 'Discrimination/Bias' incidents in the
AIID compared to 'Healthcare' and 'Transportation' sectors.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.4176 (Maybe False) |
| **Surprise** | -0.3892 |
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
| Maybe False | 60.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Validate if Bias/Discrimination harms are concentrated in specific sectors.

### Steps
- 1. Filter `aiid_incidents`.
- 2. Map `78_Sector of Deployment` to broader groups: 'Admin/Finance' (Public Admin, Finance) vs 'Health/Transport' (Healthcare, Transportation).
- 3. Create a binary `Is_Bias` variable by searching `74_Tangible Harm` and `73_Harm Domain` for terms like 'Bias', 'Discrimination', 'Civil Rights'.
- 4. Compare the rate of Bias incidents between the two sector groups using a Chi-square test.

### Deliverables
- Sector-wise incident counts, Bias rates, and statistical test results.

---

## Analysis

The experiment successfully tested the 'Sector-Specific Bias Risk' hypothesis
using the AIID incidents dataset. The code correctly mapped 47 incidents to the
target sectors: 'Admin/Finance' (n=19) and 'Health/Transport' (n=28).

**Findings:**
1.  **Zero Incidence:** The keyword search for 'bias', 'discrimination', or
'civil rights' in the harm columns returned **zero** matches for both sector
groups. Consequently, the bias incident rate was 0.0% for both.
2.  **Statistical Result:** The Chi-square test yielded a p-value of 1.0,
confirming no statistical difference.

**Hypothesis Status:** Not Supported.

**Interpretation:** The hypothesis that Finance/Admin sectors exhibit higher
bias rates was not supported, as no bias incidents were identified in either
group within this dataset subset. This null result suggests that either such
incidents are absent in this specific data slice, or the keywords used for
detection were not present in the 'Harm Domain'/'Tangible Harm' columns for
these cases.

---

## Review

The experiment was successfully executed and faithfully followed the plan. The
code correctly adapted to the dataset schema by identifying the valid column
names ('Sector of Deployment', 'Harm Domain', 'Tangible Harm') and properly
mapping the sectors. The analysis of the 47 relevant incidents (19
Admin/Finance, 28 Health/Transport) found zero matches for bias-related keywords
in the harm columns, resulting in a 0% bias rate for both groups and a p-value
of 1.0. Consequently, the hypothesis was not supported by this specific data
subset.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def run_experiment():
    # Load dataset
    file_name = 'astalabs_discovery_all_data.csv'
    try:
        df = pd.read_csv(file_name, low_memory=False)
    except FileNotFoundError:
        try:
            df = pd.read_csv('../' + file_name, low_memory=False)
        except FileNotFoundError:
            print("Error: Dataset not found.")
            return

    # Filter for AIID incidents
    aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"Loaded {len(aiid_df)} AIID incidents.")

    # Correct Column Names based on previous exploration
    col_sector = 'Sector of Deployment'
    col_harm_domain = 'Harm Domain'
    col_tangible_harm = 'Tangible Harm'

    # 1. Map Sectors to Groups
    # Group A: 'Admin/Finance' (Public Admin, Finance)
    # Group B: 'Health/Transport' (Healthcare, Transportation)
    
    def map_sector(val):
        if pd.isna(val):
            return None
        val_lower = str(val).lower()
        if 'public administration' in val_lower or 'finance' in val_lower or 'financial' in val_lower:
            return 'Admin/Finance'
        elif 'healthcare' in val_lower or 'transportation' in val_lower:
            return 'Health/Transport'
        return None

    aiid_df['Sector_Group'] = aiid_df[col_sector].apply(map_sector)
    
    # Filter for only relevant sectors
    analysis_df = aiid_df.dropna(subset=['Sector_Group']).copy()
    print(f"\nAnalysis Subset: {len(analysis_df)} incidents in target sectors.")
    print(analysis_df['Sector_Group'].value_counts())

    if len(analysis_df) == 0:
        print("No data found for the target sectors.")
        return

    # 2. Create Binary Bias Variable
    # Keywords: 'bias', 'discrimination', 'civil rights'
    keywords = ['bias', 'discrimination', 'civil rights']

    def is_bias_incident(row):
        # Combine text fields for search
        text_content = f"{str(row[col_harm_domain])} {str(row[col_tangible_harm])}".lower()
        return 1 if any(k in text_content for k in keywords) else 0

    analysis_df['Is_Bias'] = analysis_df.apply(is_bias_incident, axis=1)

    # 3. Statistical Analysis
    # Contingency Table
    contingency = pd.crosstab(analysis_df['Sector_Group'], analysis_df['Is_Bias'])
    print("\nContingency Table (0=No Bias, 1=Bias):")
    print(contingency)

    # Calculate Rates
    stats = analysis_df.groupby('Sector_Group')['Is_Bias'].agg(['count', 'sum', 'mean'])
    stats.columns = ['Total', 'Bias_Count', 'Bias_Rate']
    print("\nDescriptive Statistics:")
    print(stats)

    # Chi-square Test
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-square Test Results:")
    print(f"Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    if p < 0.05:
        print("Result: Significant difference found (Reject Null).")
    else:
        print("Result: No significant difference found (Fail to Reject Null).")

    # 4. Visualization
    plt.figure(figsize=(10, 6))
    
    # Colors: Admin/Finance (Blue), Health/Transport (Orange)
    colors = ['#1f77b4' if 'Admin' in idx else '#ff7f0e' for idx in stats.index]
    
    bars = plt.bar(stats.index, stats['Bias_Rate'], color=colors, alpha=0.8)
    
    plt.title('Bias/Discrimination Incident Rate by Sector Group')
    plt.ylabel('Proportion of Incidents Involving Bias')
    plt.ylim(0, max(stats['Bias_Rate'].max() * 1.2, 0.1))

    # Annotate bars
    for bar, total, bias_c in zip(bars, stats['Total'], stats['Bias_Count']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.1%} (n={bias_c}/{total})',
                 ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded 1362 AIID incidents.

Analysis Subset: 47 incidents in target sectors.
Sector_Group
Health/Transport    28
Admin/Finance       19
Name: count, dtype: int64

Contingency Table (0=No Bias, 1=Bias):
Is_Bias            0
Sector_Group        
Admin/Finance     19
Health/Transport  28

Descriptive Statistics:
                  Total  Bias_Count  Bias_Rate
Sector_Group                                  
Admin/Finance        19           0        0.0
Health/Transport     28           0        0.0

Chi-square Test Results:
Statistic: 0.0000
P-value: 1.0000
Result: No significant difference found (Fail to Reject Null).


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Bar Plot.
*   **Purpose:** The plot is designed to compare the rate or proportion of bias/discrimination incidents across different professional sector groups (Admin/Finance vs. Health/Transport).

**2. Axes**
*   **X-axis:**
    *   **Labels:** Categorical labels for sector groups: "Admin/Finance" and "Health/Transport".
    *   **Range:** Two discrete categories.
*   **Y-axis:**
    *   **Title:** "Proportion of Incidents Involving Bias".
    *   **Value Range:** 0.00 to 0.10.
    *   **Units:** Proportions (decimal format, representing percentages from 0% to 10%).

**3. Data Trends**
*   **Pattern:** The most distinct trend is the complete absence of visible bars. Both categories show a value of 0.
*   **Highs/Lows:** There are no "tallest" bars; both categories are tied at the lowest possible value (0.0).
*   **Comparison:** There is no variation between the two sector groups regarding the proportion of incidents involving bias; both are identical at zero.

**4. Annotations and Legends**
*   **Annotations:** There are text annotations placed directly above the x-axis labels for each category providing precise data:
    *   **Admin/Finance:** "0.0% (n=0/19)" – This indicates that out of a sample size of 19, there were 0 incidents.
    *   **Health/Transport:** "0.0% (n=0/28)" – This indicates that out of a sample size of 28, there were 0 incidents.
*   **Legend:** There is no separate legend required as the categories are labeled directly on the x-axis.

**5. Statistical Insights**
*   **Absence of Incidents:** The data reveals that no bias or discrimination incidents occurred or were reported in either the "Admin/Finance" or "Health/Transport" sector groups within the timeframe or dataset analyzed.
*   **Sample Size Context:** While the incident rate is 0% for both, the sample sizes differ slightly. The Admin/Finance group had a sample size ($n$) of 19, and the Health/Transport group had a larger sample size ($n$) of 28.
*   **Conclusion:** Based on this specific dataset, neither sector shows a higher prevalence of bias incidents, as the phenomenon was essentially non-existent in the observed samples.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
