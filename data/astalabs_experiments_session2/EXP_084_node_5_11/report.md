# Experiment 84: node_5_11

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_11` |
| **ID in Run** | 84 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:04:46.765066+00:00 |
| **Runtime** | 287.1s |
| **Parent** | `node_4_24` |
| **Children** | `node_6_10` |
| **Creation Index** | 85 |

---

## Hypothesis

> The 'Vendor Reliance' Trap: Commercial (COTS) AI systems in federal agencies
have significantly lower rates of documented 'Impact Assessments' compared to
Government-developed systems, suggesting an over-reliance on vendor assurances.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7742 (Likely True) |
| **Posterior** | 0.4121 (Maybe False) |
| **Surprise** | -0.4345 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 4.0 |
| Maybe True | 26.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 54.0 |
| Definitely False | 6.0 |

---

## Experiment Plan

**Objective:** Compare 'Impact Assessment' compliance rates between Commercial and Government-developed AI systems.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'source_table' = 'eo13960_scored'.
- 2. Create a 'Development_Source' variable using '10_commercial_ai': 'Commercial' (Yes) vs 'Government' (No/Custom).
- 3. Binarize '52_impact_assessment' (Yes=1, Others=0).
- 4. Generate a contingency table and perform a Chi-square test.
- 5. Calculate and compare the compliance percentages.

### Deliverables
- Impact Assessment rates for Commercial vs Government systems; Chi-square test results.

---

## Analysis

The experiment successfully tested the 'Vendor Reliance Trap' hypothesis using
the EO 13960 dataset. After correcting the variable selection logic (switching
from the descriptive '10_commercial_ai' to the more robust '22_dev_method'), the
code successfully classified 920 systems into 'Commercial' (n=481) and
'Government' (n=439) categories.

The analysis revealed that while Commercial systems did have a lower rate of
documented Impact Assessments (4.4%) compared to Government-developed systems
(6.6%), the difference was not statistically significant (Chi-square = 1.83, p =
0.177). Consequently, the hypothesis that Commercial systems have
*significantly* lower compliance cannot be supported. The data instead suggests
a systemic lack of documentation across both development models, with compliance
rates below 7% for both groups.

---

## Review

The experiment was faithfully executed after correcting the variable selection
logic. The programmer correctly identified that '10_commercial_ai' contained
descriptive text rather than binary indicators and switched to the more robust
'22_dev_method' to distinguish between 'Commercial' (Contractor-developed) and
'Government' (In-house) systems. The statistical analysis was appropriate for
the data types.

**Findings:**
1. **Data Yield:** The analysis successfully categorized 920 systems from the EO
13960 dataset, providing a solid sample size (Commercial n=481, Government
n=439).
2. **Compliance Rates:** Both groups showed very low compliance with documented
Impact Assessments. Government-developed systems had a slightly higher rate
(6.6%) compared to Commercial systems (4.4%).
3. **Hypothesis Verification:** The Chi-square test (p = 0.177) indicated that
this difference is **not statistically significant**. Therefore, the hypothesis
that Commercial systems have *significantly* lower compliance rates is not
supported by the data. Instead, the results point to a systemic lack of
documentation across federal AI inventories regardless of the development
source.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
from scipy.stats import chi2_contingency

# Load dataset
file_path = 'astalabs_discovery_all_data.csv'
print(f"Loading dataset from {file_path}...")

try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if file is in parent directory
    file_path = '../astalabs_discovery_all_data.csv'
    df = pd.read_csv(file_path, low_memory=False)

# Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Filtered EO 13960 records: {len(df_eo)}")

# --- Variable Creation ---

# Note: '10_commercial_ai' was found to contain use-case descriptions rather than 
# binary source indicators. Using '22_dev_method' as the robust proxy for 
# 'Development Source' (Government vs Vendor).

col_method = '22_dev_method'
col_impact = '52_impact_assessment'

# Define categorization logic for Development Source
def classify_source(val):
    if pd.isna(val):
        return None
    s = str(val).lower().strip()
    if 'in-house' in s and 'contracting' not in s:
        return 'Government'
    elif 'contracting' in s and 'in-house' not in s:
        return 'Commercial'
    # Exclude 'Both' or other ambiguous cases to ensure clean comparison
    return None

df_eo['Development_Source'] = df_eo[col_method].apply(classify_source)

# Define categorization logic for Impact Assessment
# Strict criteria: Only explicit 'Yes' counts as evidence.
def classify_impact(val):
    if pd.isna(val):
        return 0
    s = str(val).lower().strip()
    if s == 'yes':
        return 1
    return 0

df_eo['Has_Impact_Assessment'] = df_eo[col_impact].apply(classify_impact)

# Filter for valid analysis rows
df_analysis = df_eo.dropna(subset=['Development_Source'])
print(f"Records with valid Development Source: {len(df_analysis)}")
print(f"Distribution of Development Source:\n{df_analysis['Development_Source'].value_counts()}")

# --- Statistical Analysis ---

# Contingency Table
contingency = pd.crosstab(df_analysis['Development_Source'], df_analysis['Has_Impact_Assessment'])
contingency.columns = ['No Assessment', 'Has Assessment']
print("\nContingency Table (Source vs Impact Assessment):")
print(contingency)

# Chi-square Test
chi2, p, dof, expected = chi2_contingency(contingency)

# Compliance Rates
rates = df_analysis.groupby('Development_Source')['Has_Impact_Assessment'].mean() * 100
print("\nCompliance Rates (%):")
print(rates)

print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.5f}")

# Interpret
if p < 0.05:
    print("Result: Statistically Significant.")
    if rates['Commercial'] < rates['Government']:
        print("Hypothesis Supported: Commercial systems have significantly LOWER impact assessment rates.")
    else:
        print("Hypothesis Rejected: Commercial systems do NOT have lower rates.")
else:
    print("Result: Not Statistically Significant.")

# --- Visualization ---
plt.figure(figsize=(8, 6))
colors = ['#1f77b4', '#ff7f0e']
ax = rates.plot(kind='bar', color=colors, alpha=0.8)
plt.title('Impact Assessment Compliance: Commercial vs. Government')
plt.ylabel('Compliance Rate (%)')
plt.xlabel('Development Source')
plt.xticks(rotation=0)
plt.ylim(0, max(rates.max() + 5, 10))

# Annotate bars
for i, v in enumerate(rates):
    ax.text(i, v + 0.2, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
Filtered EO 13960 records: 1757
Records with valid Development Source: 920
Distribution of Development Source:
Development_Source
Commercial    481
Government    439
Name: count, dtype: int64

Contingency Table (Source vs Impact Assessment):
                    No Assessment  Has Assessment
Development_Source                               
Commercial                    460              21
Government                    410              29

Compliance Rates (%):
Development_Source
Commercial    4.365904
Government    6.605923
Name: Has_Impact_Assessment, dtype: float64

Chi-square Statistic: 1.8262
P-value: 0.17658
Result: Not Statistically Significant.


=== Plot Analysis (figure 1) ===
Based on the analysis of the provided plot, here are the detailed findings:

**1. Plot Type**
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot compares the "Impact Assessment Compliance" rates between two distinct categories of development sources: Commercial and Government.

**2. Axes**
*   **X-Axis:**
    *   **Title:** "Development Source"
    *   **Labels:** Two categorical labels: "Commercial" and "Government".
*   **Y-Axis:**
    *   **Title:** "Compliance Rate (%)"
    *   **Range:** The axis is marked in increments of 2, starting from 0 and ending at 10. The visual space extends slightly beyond 10, likely up to 12.
    *   **Units:** Percentage (%).

**3. Data Trends**
*   **Tallest Bar:** The "Government" category represents the highest value.
*   **Shortest Bar:** The "Commercial" category represents the lowest value.
*   **Pattern:** There is a clear trend indicating that government-developed sources have a higher compliance rate compared to commercial sources within the context of this dataset.

**4. Annotations and Legends**
*   **Value Annotations:** Specific data values are annotated directly above each bar in bold text for clarity:
    *   Commercial: **4.4%**
    *   Government: **6.6%**
*   **Chart Title:** "Impact Assessment Compliance: Commercial vs. Government" is displayed at the top.
*   **Color Coding:** The Commercial bar is colored blue, and the Government bar is colored orange, serving to visually distinguish the two categories (though no separate legend box is required as the x-axis is labeled).

**5. Statistical Insights**
*   **Relative Performance:** The Government sector demonstrates a compliance rate that is **2.2 percentage points higher** than the Commercial sector ($6.6\% - 4.4\%$).
*   **Proportional Difference:** In relative terms, the compliance rate for Government sources is **50% higher** than that of Commercial sources ($6.6 / 4.4 = 1.5$).
*   **Overall Magnitude:** despite the relative difference, the absolute compliance rates for both sectors are notably low, with neither category reaching even a 7% compliance rate. This suggests a systemic issue with impact assessment compliance regardless of the development source.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
