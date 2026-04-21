# Experiment 72: node_5_7

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_7` |
| **ID in Run** | 72 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T04:25:34.867280+00:00 |
| **Runtime** | 268.2s |
| **Parent** | `node_4_5` |
| **Children** | `node_6_2`, `node_6_29` |
| **Creation Index** | 73 |

---

## Hypothesis

> The 'Defense' Transparency Gap: Defense and Security agencies (DOD, DHS, DOJ)
are significantly less likely to provide 'Public Notice' of their AI use cases
compared to Civilian/Social agencies, reflecting a culture of opacity.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9032 (Definitely True) |
| **Posterior** | 0.3462 (Maybe False) |
| **Surprise** | -0.6685 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 20.0 |
| Maybe True | 10.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 14.0 |
| Definitely False | 46.0 |

---

## Experiment Plan

**Objective:** Quantify the difference in public transparency between defense and civilian agencies by correctly classifying descriptive notice values.

### Steps
- 1. Load the `astalabs_discovery_all_data.csv` dataset and filter for rows where `source_table` is 'eo13960_scored'.
- 2. Create a `Agency_Type` column: classify rows with `3_abr` values of 'DOD', 'DHS', or 'DOJ' as 'Defense/Security', and all others as 'Civilian'.
- 3. Clean the `59_ai_notice` column. Inspect unique values and create a binary `Has_Notice` variable:
    - Map affirmative values (e.g., containing 'Online', 'In-person', 'Email', 'Telephone', 'Other') to 1.
    - Map negative/exclusionary values (e.g., 'None of the above', 'N/A', 'Waived', 'not safety', or missing) to 0.
- 4. Calculate the proportion of systems with `Has_Notice` = 1 for both Agency Types.
- 5. Perform a Chi-square test of independence to determine if the difference in notice rates is statistically significant.
- 6. Generate a bar chart comparing the Notice Rates with error bars or annotations for sample size.

### Deliverables
- 1. Contingency table of Agency Type vs. Public Notice status.
- 2. Chi-square test results (statistic and p-value).
- 3. Bar chart visualization of Transparency Rates by Agency Type.

---

## Analysis

The experiment successfully tested the 'Defense Transparency Gap' hypothesis
using the EO 13960 dataset.

**Data Processing:**
- The script corrected the previous logic error by properly parsing the
`59_ai_notice` column. It distinguished between affirmative notice methods
(e.g., 'Online', 'In-person', 'Email') and non-compliance/exclusions (e.g.,
'None of the above', 'N/A', 'Waived').
- A total of 1,757 systems were analyzed: 1,574 Civilian and 183
Defense/Security.

**Results:**
- **Civilian Agencies:** 3.81% (60/1574) provided public notice.
- **Defense/Security Agencies:** 6.01% (11/183) provided public notice.
- **Statistical Test:** The Chi-square test yielded a p-value of 0.2181
(Statistic=1.52), indicating no statistically significant difference between the
two groups.

**Hypothesis Evaluation:**
The hypothesis that Defense agencies are "significantly less likely" to provide
public notice is **not supported**.
1.  **Directionality:** The observed trend actually runs opposite to the
hypothesis; Defense agencies had a nominally *higher* rate of public notice
(6.0% vs 3.8%) than Civilian agencies.
2.  **Significance:** The difference is not statistically significant (p >
0.05).
3.  **Overall Insight:** The most significant finding is the universally low
compliance rate. Regardless of agency type, over 93% of federal AI systems in
this inventory lack a recorded method of public notice.

---

## Review

The experiment successfully tested the 'Defense Transparency Gap' hypothesis
using the EO 13960 dataset. The implementation correctly parsed the descriptive
'Public Notice' field, distinguishing between affirmative notice methods (e.g.,
Online, In-person) and non-compliance. The analysis revealed that contrary to
the hypothesis, Defense/Security agencies (6.01%) actually have a slightly
higher rate of public notice than Civilian agencies (3.81%), though the
difference is not statistically significant (p = 0.2181). The hypothesis is
therefore refuted. The most critical finding is the universally low compliance
rate across the entire federal inventory (>93% missing public notice).

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# Normalize Agency Abbreviation
eo_df['3_abr'] = eo_df['3_abr'].astype(str).str.upper().str.strip()

# Define Agency Types
defense_codes = ['DOD', 'DHS', 'DOJ']
eo_df['Agency_Type'] = eo_df['3_abr'].apply(lambda x: 'Defense/Security' if x in defense_codes else 'Civilian')

# Define logic to parse '59_ai_notice'
def parse_notice(val):
    s = str(val).lower().strip()
    # Negative indicators
    if s == 'nan' or s == '': return 0
    if 'none of the above' in s: return 0
    if 'n/a' in s: return 0
    if 'waived' in s: return 0
    if 'not safety' in s: return 0
    
    # Affirmative indicators (if not caught by above)
    # The previous output showed values like 'Online', 'In-person', 'Email', 'Telephone', 'Other'
    # Since we filtered out negatives, we assume the rest are affirmative forms of notice.
    return 1

# Apply parsing
eo_df['Has_Notice'] = eo_df['59_ai_notice'].apply(parse_notice)

# Calculate Rates
rates = eo_df.groupby('Agency_Type')['Has_Notice'].agg(['count', 'sum', 'mean'])
rates.columns = ['Total Systems', 'Systems with Notice', 'Notice Rate']

print("--- Transparency Rates by Agency Type ---")
print(rates)
print("\n")

# Contingency Table for Chi-Square
contingency = pd.crosstab(eo_df['Agency_Type'], eo_df['Has_Notice'])
print("--- Contingency Table (0=No Notice, 1=Notice) ---")
print(contingency)
print("\n")

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency)
print("--- Chi-square Test Results ---")
print(f"Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Interpretation
alpha = 0.05
is_significant = p < alpha
print(f"\nStatistically Significant: {is_significant}")
if is_significant:
    def_rate = rates.loc['Defense/Security', 'Notice Rate']
    civ_rate = rates.loc['Civilian', 'Notice Rate']
    if def_rate < civ_rate:
        print("Direction: Defense agencies have significantly LOWER transparency.")
    else:
        print("Direction: Defense agencies have significantly HIGHER transparency.")
else:
    print("No significant difference in transparency rates.")

# Visualization
plt.figure(figsize=(8, 6))
colors = ['#1f77b4', '#d62728'] # Blue for Civilian, Red for Defense
ax = rates['Notice Rate'].plot(kind='bar', color=colors, rot=0)
plt.title('Public AI Notice Compliance: Civilian vs Defense')
plt.ylabel('Proportion of Systems with Public Notice')
plt.ylim(0, 1.0)

# Add value labels
for i, v in enumerate(rates['Notice Rate']):
    ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Transparency Rates by Agency Type ---
                  Total Systems  Systems with Notice  Notice Rate
Agency_Type                                                      
Civilian                   1574                   60     0.038119
Defense/Security            183                   11     0.060109


--- Contingency Table (0=No Notice, 1=Notice) ---
Has_Notice           0   1
Agency_Type               
Civilian          1514  60
Defense/Security   172  11


--- Chi-square Test Results ---
Statistic: 1.5166
P-value: 2.1814e-01

Statistically Significant: False
No significant difference in transparency rates.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot (Vertical).
*   **Purpose:** The plot compares a single metric (proportion of compliance) across two distinct categorical groups: "Civilian" agencies and "Defense/Security" agencies.

### 2. Axes
*   **X-axis:**
    *   **Title:** `Agency_Type`
    *   **Labels:** Two distinct categories are represented: "Civilian" and "Defense/Security".
*   **Y-axis:**
    *   **Title:** "Proportion of Systems with Public Notice"
    *   **Range:** The axis is scaled from 0.0 to 1.0, representing a proportion from 0% to 100%.
    *   **Units:** The axis uses decimal proportions (0.0, 0.2, 0.4, etc.), while the data annotations convert these to percentages.

### 3. Data Trends
*   **Tallest Bar:** The "Defense/Security" category represents the higher value.
*   **Shortest Bar:** The "Civilian" category represents the lower value.
*   **Pattern:** Both bars are extremely short relative to the total y-axis scale of 1.0. This indicates that the measured values are very small fractions of the whole. There is a slight upward trend moving from Civilian to Defense/Security.

### 4. Annotations and Legends
*   **Title:** "Public AI Notice Compliance: Civilian vs Defense" – clear indication of the subject matter.
*   **Data Labels:** There are explicit percentage annotations placed above each bar to provide exact values:
    *   Civilian: **3.8%**
    *   Defense/Security: **6.0%**
*   **Color Coding:** The bars are colored differently (Blue for Civilian, Red for Defense/Security) to visually distinguish the categories, though a separate legend is not strictly necessary due to the x-axis labels.

### 5. Statistical Insights
*   **Low Overall Compliance:** The most significant insight is the remarkably low compliance rate across *both* sectors. With the y-axis scaled to 1.0 (100%), the vast amount of empty white space highlights that over 90% of systems in both categories do not have public notices.
*   **Sector Comparison:** Defense/Security agencies show a slightly higher compliance rate (6.0%) compared to Civilian agencies (3.8%). While the Defense sector's compliance is roughly 1.5 times that of the Civilian sector in relative terms, the absolute difference is small (2.2 percentage points).
*   **Conclusion:** While Defense/Security agencies are technically performing better on this metric, neither sector has achieved significant penetration regarding public AI notices based on this data.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
