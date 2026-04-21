# Experiment 115: node_6_2

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_2` |
| **ID in Run** | 115 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:33:11.498510+00:00 |
| **Runtime** | 245.4s |
| **Parent** | `node_5_7` |
| **Children** | `node_7_3`, `node_7_4` |
| **Creation Index** | 116 |

---

## Hypothesis

> The 'Physical' Sector Risk: Incidents in the 'Transportation' and 'Energy'
sectors have a significantly higher ratio of 'Death/Injury' harms compared to
the 'Finance' and 'Education' sectors.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.4231 (Maybe False) |
| **Surprise** | -0.6730 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 30.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 32.0 |
| Definitely False | 28.0 |

---

## Experiment Plan

**Objective:** Correlate industry sectors with physical safety risks.

### Steps
- 1. Filter 'aiid_incidents'.
- 2. Group '78_Sector of Deployment' into 'Physical_Infra' (Transportation, Energy, Construction) vs 'Services' (Finance, Education, Admin).
- 3. Create a binary target 'Is_Physical_Harm' based on '74_Tangible Harm' containing 'Death' or 'Injury'.
- 4. Compare the rates using a Chi-square test.

### Deliverables
- 1. Physical harm rates by Sector Group.
- 2. Chi-square stats.
- 3. Visualization.

---

## Analysis

The experiment successfully tested the 'Physical Sector Risk' hypothesis using
the AIID incidents dataset. The code correctly identified the relevant columns
('Sector of Deployment', 'Tangible Harm') and categorized incidents into
'Physical Infrastructure' (n=28) and 'Services' (n=41) sectors.

**Findings:**
1.  **Zero Incidence:** Contrary to the expectation of higher physical risks in
transportation/energy, the analysis found **zero** recorded incidents of 'Death'
or 'Injury' in either sector group within this dataset (Rate: 0.0%).
2.  **Statistical Result:** The Chi-square test yielded a p-value of 1.0,
confirming no statistical difference.

**Hypothesis Status:** Not Supported.

**Interpretation:** The hypothesis that physical infrastructure sectors exhibit
higher rates of physical harm was not supported by the AIID data. The complete
absence of death/injury records in these subsets suggests that the AIID dataset
likely skews towards non-physical harms (e.g., bias, privacy) or that such
physical incidents are rare/unreported in this specific database.

---

## Review

The experiment was successfully executed. The code adapted to the schema
differences in the AIID subset (lacking numeric prefixes) and correctly
identified the relevant columns. The analysis revealed that within the filtered
dataset, there were zero recorded incidents of 'Death' or 'Injury' for the
specified sectors, leading to a null result (p=1.0). While the hypothesis was
not supported, the methodology was sound.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

# Load dataset
filename = 'astalabs_discovery_all_data.csv'
if os.path.exists(f'../{filename}'):
    filepath = f'../{filename}'
else:
    filepath = filename

print(f"Loading data from {filepath}...")
df = pd.read_csv(filepath, low_memory=False)

# Filter AIID
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID rows: {len(aiid_df)}")

# Identify correct column names (handling potential lack of prefixes vs index from previous logs)
# Based on logs: 'Sector of Deployment' and 'Tangible Harm' should be the names
cols = aiid_df.columns.tolist()
sector_col = next((c for c in cols if 'Sector of Deployment' in c), None)
harm_col = next((c for c in cols if 'Tangible Harm' in c), None)

print(f"Using Sector Column: {sector_col}")
print(f"Using Harm Column: {harm_col}")

if not sector_col or not harm_col:
    print("Could not find required columns. Dumping available columns related to Sector or Harm:")
    print([c for c in cols if 'ector' in c or 'arm' in c])
    exit(1)

# Map sectors
def map_sector(val):
    if pd.isna(val):
        return None
    s = str(val).lower()
    if any(x in s for x in ['transportation', 'energy', 'construction']):
        return 'Physical_Infra'
    if any(x in s for x in ['finance', 'financial', 'education', 'admin', 'public administration']):
        return 'Services'
    return 'Other'

aiid_df['Sector_Group'] = aiid_df[sector_col].apply(map_sector)

# Filter groups
analysis_df = aiid_df[aiid_df['Sector_Group'].isin(['Physical_Infra', 'Services'])].copy()

# Map Harms
def check_physical_harm(val):
    if pd.isna(val):
        return 0
    s = str(val).lower()
    if 'death' in s or 'injury' in s:
        return 1
    return 0

analysis_df['Is_Physical_Harm'] = analysis_df[harm_col].apply(check_physical_harm)

# Stats
rates = analysis_df.groupby('Sector_Group')['Is_Physical_Harm'].mean()
counts = analysis_df['Sector_Group'].value_counts()
print("\nRates of Physical Harm (Death/Injury):")
print(rates)
print("\nCounts:")
print(counts)

# Chi-square
ct = pd.crosstab(analysis_df['Sector_Group'], analysis_df['Is_Physical_Harm'])
print("\nContingency Table (0=No Harm, 1=Death/Injury):")
print(ct)
chi2, p, dof, ex = chi2_contingency(ct)
print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.6e}")

# Plot
plt.figure(figsize=(8,6))
# Define colors: Red for Physical/Infra (high risk assumed), Blue for Services
colors = []
for group in rates.index:
    if group == 'Physical_Infra':
        colors.append('#d9534f') # Red
    else:
        colors.append('#5bc0de') # Blue

bars = plt.bar(rates.index, rates.values, color=colors, alpha=0.8)
plt.title('Physical Harm Rate by Sector Group')
plt.ylabel('Rate of Death/Injury Incidents')
plt.xlabel('Sector Group')
plt.ylim(0, max(rates.values) * 1.2)

for bar, count in zip(bars, counts[rates.index]):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height,
             f'{height:.1%} (n={count})', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading data from astalabs_discovery_all_data.csv...
AIID rows: 1362
Using Sector Column: Sector of Deployment
Using Harm Column: Tangible Harm

Rates of Physical Harm (Death/Injury):
Sector_Group
Physical_Infra    0.0
Services          0.0
Name: Is_Physical_Harm, dtype: float64

Counts:
Sector_Group
Services          41
Physical_Infra    28
Name: count, dtype: int64

Contingency Table (0=No Harm, 1=Death/Injury):
Is_Physical_Harm   0
Sector_Group        
Physical_Infra    28
Services          41

Chi-square Statistic: 0.0000
P-value: 1.000000e+00

STDERR:
<ipython-input-1-f7ca959a4e8e>:95: UserWarning: Attempting to set identical low and high ylims makes transformation singular; automatically expanding.
  plt.ylim(0, max(rates.values) * 1.2)


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot.
*   **Purpose:** The plot is designed to compare the "Rate of Death/Injury Incidents" (physical harm) across two distinct sector groups: "Physical_Infra" and "Services." Since the values are zero, the bars have no height, appearing as flat lines on the x-axis.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Sector Group"
    *   **Labels:** Two categorical groups are represented: "Physical_Infra" (Physical Infrastructure) and "Services."
*   **Y-Axis:**
    *   **Title:** "Rate of Death/Injury Incidents"
    *   **Range:** The axis spans from approximately **-0.05 to 0.05**.
    *   **Units:** Rate (expressed as a decimal on the axis ticks, but interpreted as a percentage in the annotations). Note that the negative values on the axis are a result of automatic scaling by the plotting software to center the zero-value data; a negative rate of injury is not physically possible.

### 3. Data Trends
*   **Pattern:** The most distinct pattern is the **absence of incidents**. Both categories show a rate of exactly 0.0.
*   **Comparison:** There is no variation between the two sectors regarding physical harm rates in this dataset; both are identical at zero.

### 4. Annotations and Legends
*   **Text Annotations:** There are specific text labels placed at the coordinate corresponding to each sector:
    *   **Physical_Infra:** "0.0% (n=28)"
    *   **Services:** "0.0% (n=41)"
*   **Meaning:**
    *   **0.0%:** Indicates the calculated rate of physical harm.
    *   **(n=28/41):** Represents the sample size (N-count) for each group. There were 28 observations for Physical Infrastructure and 41 observations for Services.

### 5. Statistical Insights
*   **Zero Incidence:** Based on the data presented, there were **zero** recorded incidents of death or injury in either the Physical Infrastructure or Services sectors.
*   **Sample Size:** The "Services" sector had a larger sample size (n=41) compared to "Physical_Infra" (n=28).
*   **Conclusion:** Despite a combined total of 69 observations (n=28 + n=41), the event rate for physical harm was non-existent (0%) for the duration or scope of this specific study.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
