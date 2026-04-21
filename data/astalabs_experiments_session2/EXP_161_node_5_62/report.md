# Experiment 161: node_5_62

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_62` |
| **ID in Run** | 161 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:40:18.363196+00:00 |
| **Runtime** | 165.2s |
| **Parent** | `node_4_25` |
| **Children** | `node_6_85` |
| **Creation Index** | 162 |

---

## Hypothesis

> The Maturity-Privacy Paradox: AI systems in the 'Operational' stage are
significantly more likely to process PII (`29_contains_pii`) compared to systems
in the 'Development' stage, suggesting privacy risks escalate rather than being
mitigated as systems mature.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.4023 (Maybe False) |
| **Surprise** | -0.4076 |
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
| Maybe False | 48.0 |
| Definitely False | 8.0 |

---

## Experiment Plan

**Objective:** Investigate if privacy risk exposure increases with system lifecycle maturity.

### Steps
- 1. Filter `eo13960_scored` for systems in 'Operational' vs 'Development' stages using `16_dev_stage`.
- 2. Analyze the `29_contains_pii` column (Yes/No).
- 3. Compare the proportion of PII-processing systems between the two stages using a Z-test for proportions.

### Deliverables
- Proportions of PII usage in Operational vs Development stages; Z-test statistic and p-value.

---

## Analysis

The experiment successfully tested the 'Maturity-Privacy Paradox' hypothesis
using the EO 13960 dataset. After mapping the `16_dev_stage` column into
'Operational' (e.g., Operation and Maintenance) and 'Development' (e.g.,
Acquisition, Planned) categories and filtering for valid PII indicators, 652
systems were analyzed.

The results showed that Operational systems had a slightly higher prevalence of
PII (22.27%, n=503) compared to Development systems (18.12%, n=149). However,
the Chi-square test yielded a p-value of 0.3314, which is well above the
standard significance threshold of 0.05.

Therefore, the hypothesis is rejected. While there is a nominal increase in PII
usage in operational systems, the difference is not statistically significant.
The data suggests that privacy risks regarding PII presence do not significantly
escalate between the development and operational stages in this specific federal
inventory.

---

## Review

The experiment successfully tested the 'Maturity-Privacy Paradox' hypothesis
using the EO 13960 dataset. After mapping the `16_dev_stage` column into
'Operational' (e.g., Operation and Maintenance) and 'Development' (e.g.,
Acquisition, Planned) categories and filtering for valid PII indicators, 652
systems were analyzed. The results showed that Operational systems had a
slightly higher prevalence of PII (22.27%, n=503) compared to Development
systems (18.12%, n=149). However, the Chi-square test yielded a p-value of
0.3314, which is well above the standard significance threshold of 0.05.
Therefore, the hypothesis is rejected. While there is a nominal increase in PII
usage in operational systems, the difference is not statistically significant.
The data suggests that privacy risks regarding PII presence do not significantly
escalate between the development and operational stages in this specific federal
inventory.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print("--- Data Loading ---")
print(f"Total EO 13960 records: {len(eo_data)}")

# Inspect columns for mapping
print("\nUnique values in '16_dev_stage':")
print(eo_data['16_dev_stage'].unique())
print("\nUnique values in '29_contains_pii':")
print(eo_data['29_contains_pii'].unique())

# 1. Map Development Stage
# Hypothesized mapping based on federal IT standards:
# Operational: 'Operation and maintenance', 'Implemented'
# Development: 'Development and acquisition', 'Planned', 'Research and development'

def map_stage(val):
    if pd.isna(val):
        return None
    val_lower = str(val).lower()
    if 'operation' in val_lower or 'implemented' in val_lower or 'use' in val_lower:
        return 'Operational'
    elif 'development' in val_lower or 'planned' in val_lower or 'acquisition' in val_lower:
        return 'Development'
    else:
        return None # Exclude other categories (e.g. retired) if ambiguous

eo_data['stage_group'] = eo_data['16_dev_stage'].apply(map_stage)

# 2. Map PII
def map_pii(val):
    if pd.isna(val):
        return None
    val_lower = str(val).lower()
    if 'yes' in val_lower:
        return True
    elif 'no' in val_lower:
        return False
    return None

eo_data['has_pii'] = eo_data['29_contains_pii'].apply(map_pii)

# Filter for valid data
valid_data = eo_data.dropna(subset=['stage_group', 'has_pii'])

print(f"\nRecords after filtering for valid Stage and PII info: {len(valid_data)}")

# 3. Analysis
# Contingency Table
contingency_table = pd.crosstab(valid_data['stage_group'], valid_data['has_pii'])
print("\nContingency Table (Stage vs Has PII):")
print(contingency_table)

# Calculate Proportions
results = valid_data.groupby('stage_group')['has_pii'].agg(['count', 'sum', 'mean'])
results.columns = ['Total', 'With_PII', 'Proportion']
print("\nProportions by Stage:")
print(results)

# 4. Statistical Test (Chi-Square Test of Independence)
# We use Chi-Square as it's equivalent to Z-test for proportions with 2 groups but handles the contingency table directly
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\n--- Statistical Test Results (Chi-Square) ---")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Interpretation
alpha = 0.05
if p < alpha:
    print("Result: Significant difference found. Hypothesis supported (if direction matches) or rejected (if opposite).")
else:
    print("Result: No significant difference found.")

# Check directionality
if 'Operational' in results.index and 'Development' in results.index:
    op_prop = results.loc['Operational', 'Proportion']
    dev_prop = results.loc['Development', 'Proportion']
    print(f"Operational PII Rate: {op_prop:.2%}")
    print(f"Development PII Rate: {dev_prop:.2%}")
    if op_prop > dev_prop:
        print("Direction: Operational systems have a HIGHER rate of PII usage.")
    else:
        print("Direction: Operational systems have a LOWER or EQUAL rate of PII usage.")

# 5. Visualization
plt.figure(figsize=(8, 6))
sns.barplot(x=results.index, y='Proportion', data=results.reset_index(), palette='viridis')
plt.ylabel('Proportion of Systems Containing PII')
plt.title('PII Usage by Lifecycle Stage')
plt.ylim(0, 1.0)
for index, row in results.reset_index().iterrows():
    plt.text(index, row['Proportion'] + 0.02, f"{row['Proportion']:.1%}", ha='center')
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Data Loading ---
Total EO 13960 records: 1757

Unique values in '16_dev_stage':
<StringArray>
[ 'Implementation and Assessment', 'Acquisition and/or Development',
                      'Initiated',                        'Retired',
      'Operation and Maintenance',                  'In production',
                     'In mission',                        'Planned',
                              nan]
Length: 9, dtype: str

Unique values in '29_contains_pii':
<StringArray>
['No', 'Yes', nan, ' ']
Length: 4, dtype: str

Records after filtering for valid Stage and PII info: 652

Contingency Table (Stage vs Has PII):
has_pii      False  True 
stage_group              
Development    122     27
Operational    391    112

Proportions by Stage:
             Total With_PII Proportion
stage_group                           
Development    149       27   0.181208
Operational    503      112   0.222664

--- Statistical Test Results (Chi-Square) ---
Chi2 Statistic: 0.9435
P-value: 3.3137e-01
Result: No significant difference found.
Operational PII Rate: 22.27%
Development PII Rate: 18.12%
Direction: Operational systems have a HIGHER rate of PII usage.

STDERR:
<ipython-input-1-12caf8b617ca>:104: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x=results.index, y='Proportion', data=results.reset_index(), palette='viridis')


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** The plot is designed to compare the prevalence of Personally Identifiable Information (PII) across two distinct groups of systems based on their lifecycle stage ("Development" vs. "Operational").

### 2. Axes
*   **X-Axis:**
    *   **Label:** `stage_group`
    *   **Categories:** Two categorical values are presented: "Development" and "Operational".
*   **Y-Axis:**
    *   **Label:** "Proportion of Systems Containing PII"
    *   **Range:** The axis is scaled from 0.0 to 1.0 (representing 0% to 100%).
    *   **Ticks:** Major ticks mark every 0.2 increment (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
*   **Tallest Bar:** The "Operational" group has the taller bar, indicating a higher proportion of systems containing PII.
*   **Shortest Bar:** The "Development" group has the shorter bar.
*   **Pattern:** There is an upward trend in PII usage when moving from the Development stage to the Operational stage. Despite the difference, both bars remain in the lower quartile of the total possible range (below 0.25).

### 4. Annotations and Legends
*   **Bar Annotations:** Specific percentage values are annotated directly above each bar to provide precise data points:
    *   **Development:** 18.1%
    *   **Operational:** 22.3%
*   **Color Coding:** Although there is no separate legend box, the bars are distinct in color (dark blue for Development and green for Operational) to visually distinguish the categories.

### 5. Statistical Insights
*   **Prevalence in Production:** Operational systems are more likely to contain PII compared to development systems. Specifically, 22.3% of operational systems contain PII versus 18.1% of development systems.
*   **Gap Analysis:** There is a **4.2 percentage point difference** between the two stages. This represents a relative increase of approximately 23% in PII prevalence when shifting from development to operation.
*   **Risk Assessment:** While it is expected that Operational systems handle real user data, the fact that nearly 1 in 5 (18.1%) Development systems contain PII is a notable finding. In many security frameworks, development environments are expected to use synthetic or anonymized data to reduce data breach risks.
*   **Overall Density:** The majority of systems in both lifecycle stages do *not* contain PII (81.9% of Development and 77.7% of Operational systems are PII-free).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
