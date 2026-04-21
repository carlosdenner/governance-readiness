# Experiment 56: node_4_22

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_22` |
| **ID in Run** | 56 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:40:23.408760+00:00 |
| **Runtime** | 242.5s |
| **Parent** | `node_3_8` |
| **Children** | `node_5_39`, `node_5_59` |
| **Creation Index** | 57 |

---

## Hypothesis

> Autonomy Escalates Harm: In the AIID dataset, incidents involving 'High
Autonomy' systems are significantly more likely to result in 'Physical' harm
compared to 'Low Autonomy' systems, which primarily result in 'Financial' or
'Intangible' harm.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7177 (Likely True) |
| **Posterior** | 0.2610 (Likely False) |
| **Surprise** | -0.5481 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 28.0 |
| Uncertain | 1.0 |
| Maybe False | 1.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 6.0 |
| Definitely False | 54.0 |

---

## Experiment Plan

**Objective:** Verify if higher AI autonomy levels are associated with physical safety risks.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `aiid_incidents`.
- 2. Map `Autonomy Level` (or `autonomy`) to 'High' (e.g., High, Full) vs 'Low' (e.g., Low, Medium, Human-in-loop).
- 3. Map `Harm Domain` (or derive from `description`/`Tangible Harm`) to 'Physical' vs 'Non-Physical'.
- 4. Create a contingency table of Autonomy vs Harm Type.
- 5. Perform a Chi-Square test or Fisher's Exact Test.

### Deliverables
- Stacked Bar Chart of Harm Types by Autonomy Level; Statistical test results.

---

## Analysis

The experiment successfully tested the 'Autonomy Escalates Harm' hypothesis by
mapping 'Autonomy Level' values (mapping 'Autonomy3' to High and 'Autonomy1/2'
to Low) and text-mining incident descriptions to categorize harm as 'Physical'
or 'Financial/Intangible'.

Analysis of the 68 intersecting records reveals that the hypothesis is
**rejected**. Contrary to the expectation that high autonomy correlates with
physical risks, the data showed no statistically significant difference in the
composition of harm types between the two groups (Chi-Square p=1.0). High
Autonomy systems resulted in Physical harm 41.2% of the time, while Low Autonomy
systems did so 45.1% of the time. The results suggest that in this dataset, the
nature of the harm (Physical vs. Intangible) is independent of the system's
autonomy level.

---

## Review

The experiment was successfully executed. The code adapted to the specific data
coding issues identified in the previous attempt (mapping 'Autonomy1/2/3'
correctly and using text mining for harm classification since the structured
field was boolean). The statistical analysis was performed on the intersection
of valid records (n=68). The results (p=1.00, with Physical Harm rates of 41.2%
for High Autonomy vs 45.1% for Low Autonomy) provide clear evidence to reject
the hypothesis.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
ds_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(ds_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Loaded AIID incidents: {len(aiid_df)}")

# --- 1. Map Autonomy ---
# Observed values: 'Autonomy1', 'Autonomy2', 'Autonomy3'
# Assumption: 1=Low, 2=Medium, 3=High. Hypothesis compares High vs Low/Medium.
autonomy_col = 'Autonomy Level'
if autonomy_col not in aiid_df.columns:
    # Fallback search
    cols = [c for c in aiid_df.columns if 'autonomy' in c.lower()]
    if cols:
        autonomy_col = cols[0]

print(f"Using Autonomy column: {autonomy_col}")

def map_autonomy_level(val):
    s = str(val).lower()
    if 'autonomy3' in s or 'high' in s or 'full' in s:
        return 'High'
    if 'autonomy1' in s or 'autonomy2' in s or 'low' in s or 'medium' in s:
        return 'Low'
    return np.nan

aiid_df['Autonomy_Bin'] = aiid_df[autonomy_col].apply(map_autonomy_level)
print("Autonomy distribution:\n", aiid_df['Autonomy_Bin'].value_counts())

# --- 2. Map Harm Type ---
# 'Harm Domain' is boolean (yes/no), so we must extract type from text description.

# Identify the best text column
text_candidates = ['description', 'summary', 'title', 'incident_description']
text_col = None

# 1. Try explicit names
for c in text_candidates:
    if c in aiid_df.columns:
        text_col = c
        break

# 2. If not found, try searching column names
if not text_col:
    for c in aiid_df.columns:
        if 'description' in c.lower() or 'summary' in c.lower():
            text_col = c
            break

# 3. If still not found, find the object column with highest average length
if not text_col:
    object_cols = aiid_df.select_dtypes(include=['object']).columns
    best_col = None
    max_len = 0
    for c in object_cols:
        # Sample first 100 non-nulls
        sample = aiid_df[c].dropna().head(100).astype(str)
        if len(sample) > 0:
            avg_len = sample.str.len().mean()
            if avg_len > max_len:
                max_len = avg_len
                best_col = c
    if max_len > 30: # Threshold to ensure it's not just a long ID
        text_col = best_col

print(f"Using Text column for Harm classification: {text_col}")

def map_harm_type(text):
    if pd.isna(text):
        return np.nan
    t = str(text).lower()
    
    # Keywords
    physical = ['death', 'dead', 'kill', 'injur', 'hurt', 'fatal', 'accident', 'crash', 'collision', 'safety', 'physical', 'bodily', 'life', 'medical']
    financial_intangible = ['financ', 'money', 'dollar', 'cost', 'credit', 'bank', 'fraud', 'scam', 'discriminat', 'bias', 'racis', 'sexis', 'reputation', 'privacy', 'surveillance', 'rights', 'civil']
    
    # Priority: Physical (since hypothesis asks if autonomy escalates to physical)
    if any(k in t for k in physical):
        return 'Physical'
    if any(k in t for k in financial_intangible):
        return 'Financial/Intangible'
    
    return 'Other/Unclear'

if text_col:
    aiid_df['Harm_Bin'] = aiid_df[text_col].apply(map_harm_type)
else:
    print("No suitable text column found for harm classification.")
    aiid_df['Harm_Bin'] = np.nan

print("Harm distribution:\n", aiid_df['Harm_Bin'].value_counts())

# --- 3. Analysis ---
df_clean = aiid_df.dropna(subset=['Autonomy_Bin', 'Harm_Bin'])
df_clean = df_clean[df_clean['Harm_Bin'] != 'Other/Unclear']

print(f"Records available for analysis: {len(df_clean)}")

if len(df_clean) > 0:
    ct = pd.crosstab(df_clean['Autonomy_Bin'], df_clean['Harm_Bin'])
    print("\nContingency Table:")
    print(ct)
    
    # Plot
    ct.plot(kind='bar', stacked=True, figsize=(8, 6), color=['skyblue', 'salmon'])
    plt.title('Harm Type by Autonomy Level')
    plt.xlabel('Autonomy Level')
    plt.ylabel('Count of Incidents')
    plt.xticks(rotation=0)
    plt.legend(title='Harm Type')
    plt.tight_layout()
    plt.show()
    
    # Stats
    chi2, p, dof, ex = chi2_contingency(ct)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Calculate proportions
    props = pd.crosstab(df_clean['Autonomy_Bin'], df_clean['Harm_Bin'], normalize='index')
    print("\nProportions (Row-wise):")
    print(props)
    
    # Hypothesis Check
    high_phys = props.loc['High', 'Physical'] if 'High' in props.index and 'Physical' in props.columns else 0
    low_phys = props.loc['Low', 'Physical'] if 'Low' in props.index and 'Physical' in props.columns else 0
    
    print(f"\nPhysical Harm Rate - High Autonomy: {high_phys:.1%}")
    print(f"Physical Harm Rate - Low Autonomy: {low_phys:.1%}")
    
    if p < 0.05:
        print("Result: Significant difference detected.")
    else:
        print("Result: No significant difference detected.")
else:
    print("Insufficient data for statistical testing.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded AIID incidents: 1362
Using Autonomy column: Autonomy Level
Autonomy distribution:
 Autonomy_Bin
Low     132
High     53
Name: count, dtype: int64
Using Text column for Harm classification: description
Harm distribution:
 Harm_Bin
Other/Unclear           794
Financial/Intangible    366
Physical                202
Name: count, dtype: int64
Records available for analysis: 68

Contingency Table:
Harm_Bin      Financial/Intangible  Physical
Autonomy_Bin                                
High                            10         7
Low                             28        23

Chi-Square Statistic: 0.0000
P-value: 1.0000e+00

Proportions (Row-wise):
Harm_Bin      Financial/Intangible  Physical
Autonomy_Bin                                
High                      0.588235  0.411765
Low                       0.549020  0.450980

Physical Harm Rate - High Autonomy: 41.2%
Physical Harm Rate - Low Autonomy: 45.1%
Result: No significant difference detected.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Chart.
*   **Purpose:** This chart compares the frequency ("Count of Incidents") of two different categories of harm ("Financial/Intangible" and "Physical") across two levels of autonomy ("High" and "Low"). The stacking allows the viewer to see the total number of incidents for each autonomy level while visualizing the proportion contributed by each harm type.

### 2. Axes
*   **X-axis:**
    *   **Title:** "Autonomy Level"
    *   **Labels:** Categorical variables "High" and "Low".
*   **Y-axis:**
    *   **Title:** "Count of Incidents"
    *   **Range:** The axis starts at 0 and extends slightly past 50, with major tick marks at intervals of 10 (0, 10, 20, 30, 40, 50).
    *   **Units:** Count (integer values representing the number of incidents).

### 3. Data Trends
*   **Bar Heights (Total Incidents):**
    *   The **"Low"** autonomy bar is the tallest, indicating a significantly higher total number of incidents (approximately 51 total incidents).
    *   The **"High"** autonomy bar is the shortest, showing a much lower total frequency of incidents (approximately 17 total incidents).
*   **Composition Patterns:**
    *   **High Autonomy:** The "Financial/Intangible" segment (blue) appears to represent roughly 10 incidents, while the "Physical" segment (red) represents roughly 7 incidents. Financial/Intangible harm is slightly more prevalent here.
    *   **Low Autonomy:** The "Financial/Intangible" segment (blue) is substantial, reaching approximately 28 on the y-axis. The "Physical" segment (red) sits on top, adding roughly another 23 incidents to reach a total just above 50.
*   **General Trend:** Across both autonomy levels, "Financial/Intangible" harm accounts for a slightly larger portion of the incidents than "Physical" harm, though the distribution is relatively balanced within the "Low" category.

### 4. Annotations and Legends
*   **Chart Title:** "Harm Type by Autonomy Level" is displayed at the top center.
*   **Legend:** A box in the upper-left corner titled "Harm Type" distinguishes the data series:
    *   **Light Blue Square:** Represents "Financial/Intangible" harm.
    *   **Salmon/Light Red Square:** Represents "Physical" harm.

### 5. Statistical Insights
*   **Volume Disparity:** There is a stark contrast in the volume of reported incidents based on autonomy level. Incidents involving "Low" autonomy systems are approximately **3 times more frequent** in this dataset than those involving "High" autonomy systems.
*   **Harm Prevalence:** In absolute numbers, both Financial/Intangible and Physical harms occur much more frequently in Low autonomy scenarios.
    *   Physical harm incidents rise from ~7 in the High autonomy group to ~23 in the Low autonomy group.
    *   Financial/Intangible incidents rise from 10 in the High autonomy group to ~28 in the Low autonomy group.
*   **Risk Profile:** While "Low" autonomy has higher raw counts, the ratio of Physical to Financial harm remains somewhat consistent across both levels (Physical harm is roughly 40-45% of the total in both cases), suggesting that the *type* of harm caused may not strongly depend on whether the autonomy level is High or Low, even though the *frequency* differs greatly.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
