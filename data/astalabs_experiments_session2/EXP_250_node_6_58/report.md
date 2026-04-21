# Experiment 250: node_6_58

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_58` |
| **ID in Run** | 250 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T12:47:35.835413+00:00 |
| **Runtime** | 276.7s |
| **Parent** | `node_5_60` |
| **Children** | None |
| **Creation Index** | 251 |

---

## Hypothesis

> The Autonomy-Physicality Link: High-autonomy AI systems (e.g., autonomous
vehicles/robots) are statistically more likely to cause 'Physical' harm, whereas
lower-autonomy systems (decision support) are associated with 'Intangible' harm
(bias/economic loss).

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8790 (Likely True) |
| **Posterior** | 0.2995 (Likely False) |
| **Surprise** | -0.6955 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 17.0 |
| Maybe True | 13.0 |
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

**Objective:** Investigate the correlation between system autonomy levels and the domain of harm caused in incidents.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` (AIID subset).
- 2. Normalize `Autonomy Level`: Group into 'High' (High, Full) and 'Low' (Low, Human-in-the-loop, No).
- 3. Normalize `Harm Domain`: Group into 'Physical' vs. 'Intangible' (Societal, Economic, etc.).
- 4. Filter out 'Unknown' values.
- 5. Run a Chi-square test of independence.
- 6. Compare the proportion of Physical harm in High vs Low autonomy groups.

### Deliverables
- Contingency table, p-value, and comparative bar chart data (proportions).

---

## Analysis

The experiment successfully tested the 'Autonomy-Physicality Link' hypothesis
using the AIID dataset. After mapping 135 valid incidents into 'High'
(Autonomy3) and 'Low' (Autonomy1/2) autonomy levels and categorizing harms as
'Physical' or 'Intangible', the analysis found virtually no difference in the
nature of harm between the two groups.

**Findings:**
1.  **Physical Harm Rates:**
    -   **High Autonomy:** 48.7% (19/39 incidents involved physical harm).
    -   **Low Autonomy:** 47.9% (46/96 incidents involved physical harm).
2.  **Statistical Test:**
    -   Chi-Square Statistic: 0.0000
    -   p-value: 1.0000

**Conclusion:**
The hypothesis that high-autonomy systems are more likely to cause physical harm
is **rejected**. The data reveals that incidents involving high-autonomy systems
(e.g., autonomous vehicles) and low-autonomy systems (e.g., decision support)
share an almost identical distribution of physical vs. intangible harm types
(~48% physical) within this dataset. The level of autonomy appears uncorrelated
with the domain of harm (physical vs. intangible).

---

## Review

The experiment was successfully executed and the code faithfully implemented the
analysis plan. After correcting for data coding issues in the previous attempt
(handling specific values like 'Autonomy3' and verbose harm descriptions), the
analysis successfully classified 135 AI incidents into 'High' (n=39) and 'Low'
(n=96) autonomy levels.

**Hypothesis Test Results:**
1.  **Physical Harm Rates:**
    -   **High Autonomy:** 48.72% (19/39) involved physical harm.
    -   **Low Autonomy:** 47.92% (46/96) involved physical harm.
2.  **Statistical Test:**
    -   Chi-Square Statistic: 0.0000
    -   p-value: 1.0000

**Conclusion:**
The hypothesis that high-autonomy systems are disproportionately linked to
physical harm is **rejected**. The data shows a remarkable parity: incidents
involving high-autonomy systems (e.g., autonomous vehicles/robots) and low-
autonomy systems (e.g., human-in-the-loop decision support) exhibit nearly
identical rates of physical vs. intangible harm (~48% physical) within this
dataset. This suggests that the severity or domain of harm is not strictly a
function of the system's autonomy level.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except:
        print("Dataset not found.")
        sys.exit(1)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# 1. Map Autonomy
# Autonomy3 = High
# Autonomy1, Autonomy2 = Low
def map_autonomy(val):
    if pd.isna(val): return np.nan
    if val == 'Autonomy3': return 'High'
    if val in ['Autonomy1', 'Autonomy2']: return 'Low'
    return np.nan

aiid_df['autonomy_group'] = aiid_df['Autonomy Level'].apply(map_autonomy)

# 2. Map Harm
# Physical: 'tangible harm definitively occurred'
# Intangible: 'yes' in Special Interest Intangible Harm AND NOT Physical
def map_harm(row):
    tangible = str(row['Tangible Harm'])
    intangible = str(row['Special Interest Intangible Harm'])
    
    is_physical = 'tangible harm definitively occurred' in tangible
    is_intangible = 'yes' == intangible.lower()
    
    if is_physical:
        return 'Physical'
    elif is_intangible:
        return 'Intangible'
    else:
        return np.nan

aiid_df['harm_group'] = aiid_df.apply(map_harm, axis=1)

# 3. Filter for valid rows
analysis_df = aiid_df.dropna(subset=['autonomy_group', 'harm_group'])

print(f"Valid incidents for analysis: {len(analysis_df)}")
print("Distribution:")
print(analysis_df.groupby(['autonomy_group', 'harm_group']).size())

if len(analysis_df) < 5:
    print("Insufficient data for statistical testing.")
    sys.exit(0)

# 4. Chi-Square Test
contingency = pd.crosstab(analysis_df['autonomy_group'], analysis_df['harm_group'])
print("\nContingency Table:")
print(contingency)

chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")

# 5. Plotting
# Calculate proportion of Physical harm
props = contingency.div(contingency.sum(axis=1), axis=0)
physical_props = props['Physical'] if 'Physical' in props.columns else pd.Series([0,0], index=['High', 'Low'])

plt.figure(figsize=(8, 6))
bars = plt.bar(physical_props.index, physical_props.values, color=['#d62728', '#1f77b4'])
plt.ylabel('Proportion of Incidents Involving Physical Harm')
plt.title('Physical Harm Rates by Autonomy Level')
plt.ylim(0, 1.0)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, 
             f'{height:.1%}', ha='center', va='bottom')

plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Valid incidents for analysis: 135
Distribution:
autonomy_group  harm_group
High            Intangible    20
                Physical      19
Low             Intangible    50
                Physical      46
dtype: int64

Contingency Table:
harm_group      Intangible  Physical
autonomy_group                      
High                    20        19
Low                     50        46

Chi-Square Statistic: 0.0000
P-value: 1.0000


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot is designed to compare the proportion of incidents involving physical harm between two distinct categories of autonomy: "High" and "Low". It allows for a direct visual comparison of rates across these two groups.

### 2. Axes
*   **Y-Axis:**
    *   **Title/Label:** "Proportion of Incidents Involving Physical Harm".
    *   **Units:** The axis is scaled as a proportion (decimal), ranging from **0.0 to 1.0** (equivalent to 0% to 100%).
    *   **Ticks:** The axis is marked at intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).
*   **X-Axis:**
    *   **Categories:** The axis displays two discrete categories representing the Autonomy Level: **"High"** and **"Low"**.

### 3. Data Trends
*   **Tallest Bar:** The bar representing **"High"** autonomy is the tallest, indicating a slightly higher rate of physical harm incidents.
*   **Shortest Bar:** The bar representing **"Low"** autonomy is the shortest.
*   **Pattern:** The most notable trend is the similarity between the two values. There is very little variance between the groups; the heights of the red and blue bars are nearly identical visually.

### 4. Annotations and Legends
*   **Value Annotations:** Exact percentage values are annotated on top of each bar to provide precision beyond the y-axis scale:
    *   The **"High"** autonomy bar is labeled **48.7%**.
    *   The **"Low"** autonomy bar is labeled **47.9%**.
*   **Color Coding:**
    *   The "High" category is represented by a **red** bar.
    *   The "Low" category is represented by a **blue** bar.

### 5. Statistical Insights
*   **Marginal Difference:** The difference in physical harm rates between "High" and "Low" autonomy levels is extremely small, at only **0.8%** ($48.7\% - 47.9\%$).
*   **Parity of Risk:** The data suggests that, within the context of this specific dataset, the level of autonomy does not significantly correlate with a drastic change in the proportion of incidents that result in physical harm. Both levels result in physical harm in roughly half of the recorded incidents.
*   **Implication:** If the goal of high autonomy systems is to drastically reduce the rate of physical harm in incidents compared to low autonomy systems, this data (in isolation) suggests that goal has not yet been achieved, or that the severity profile of incidents is similar across both autonomy levels. Conversely, it also shows that high autonomy does not introduce a massive *new* risk of physical harm compared to lower levels.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
