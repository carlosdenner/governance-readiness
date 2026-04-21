# Experiment 238: node_5_93

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_93` |
| **ID in Run** | 238 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T12:10:52.204307+00:00 |
| **Runtime** | 285.2s |
| **Parent** | `node_4_9` |
| **Children** | None |
| **Creation Index** | 239 |

---

## Hypothesis

> The 'Autonomous' Escalation: High-autonomy AI systems (e.g., those that act
independently) are significantly more likely to cause 'Collective/Societal' harm
rather than 'Individual' harm, compared to Low-autonomy (assistive) systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.7418 (Likely True) |
| **Surprise** | -0.0002 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

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
| Maybe True | 58.0 |
| Uncertain | 2.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Assess if the level of AI autonomy correlates with the scale of harm distribution.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `aiid_incidents`.
- 2. Bin `81: Autonomy Level` into 'High' (High, Full, System-Level) and 'Low' (Low, Assistive, Human-in-the-loop).
- 3. Bin `76: Harm Distribution Basis` into 'Collective' (Group, Society, Community) and 'Individual'.
- 4. Create a contingency table and perform a Chi-Square test.
- 5. Visualize the ratio of Individual vs Collective harm for each autonomy level.

### Deliverables
- Chi-Square test results; Stacked bar chart of Harm Distribution by Autonomy Level.

---

## Analysis

The experiment was successfully executed and the code correctly mapped the
ordinal autonomy codes and unstructured harm distribution text into binary
categories. The analysis of 180 incidents revealed a visible trend aligning with
the hypothesis: High Autonomy systems were associated with a higher rate of
Collective/Societal harm (48%) compared to Low Autonomy systems (33%). However,
the Chi-Square test resulted in a p-value of 0.0801, which fails to meet the
standard significance threshold of 0.05. Consequently, the hypothesis is
technically rejected; while the data suggests a potential link between higher
autonomy and broader societal risks, the association is not statistically
significant within this specific sample size.

---

## Review

The experiment was faithfully executed. The programmer successfully adapted to
data quality issues by identifying that 'Autonomy Level' used ordinal codes
(mapping 'Autonomy3' to High and 'Autonomy1/2' to Low) and that 'Harm
Distribution Basis' required mapping specific demographic traits to 'Collective'
harm. The analysis of 180 incidents revealed a distinct trend: High Autonomy
systems were associated with a higher rate of Collective/Societal harm (48%)
compared to Low Autonomy systems (33%). However, the Chi-Square test yielded a
p-value of 0.0801, which fails to meet the standard significance threshold of
0.05. Consequently, the hypothesis is formally rejected based on this sample,
although the data suggests a potential underlying relationship that might reach
significance with a larger dataset.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# Mapping functions based on debug findings
def map_autonomy(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    
    if val_str == 'Autonomy3':
        return 'High' # High Autonomy
    elif val_str in ['Autonomy1', 'Autonomy2']:
        return 'Low' # Low/Assistive Autonomy
    return np.nan

def map_harm_dist(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().strip()
    
    if val_str == 'none':
        # If harm is not distributed based on a group trait, we categorize it as Individual/General
        return 'Individual'
    elif val_str == 'unclear':
        return np.nan
    else:
        # Any presence of demographic traits (race, sex, etc.) implies Collective/Group-based harm
        return 'Collective'

# Apply mappings
aiid_df['autonomy_bin'] = aiid_df['Autonomy Level'].apply(map_autonomy)
aiid_df['harm_bin'] = aiid_df['Harm Distribution Basis'].apply(map_harm_dist)

# Filter out unmapped data
analysis_df = aiid_df.dropna(subset=['autonomy_bin', 'harm_bin'])

print(f"Data points for analysis: {len(analysis_df)}")
print("Autonomy counts:")
print(analysis_df['autonomy_bin'].value_counts())
print("Harm Scale counts:")
print(analysis_df['harm_bin'].value_counts())

if len(analysis_df) == 0:
    print("No valid data points found after mapping. Exiting.")
else:
    # Create Contingency Table
    contingency_table = pd.crosstab(analysis_df['autonomy_bin'], analysis_df['harm_bin'])
    print("\nContingency Table (Count):")
    print(contingency_table)

    # Calculate Proportions (Row-wise) for plotting
    props = pd.crosstab(analysis_df['autonomy_bin'], analysis_df['harm_bin'], normalize='index')
    print("\nProportions (Row-wise):")
    print(props)

    # Perform Chi-Square Test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    # Interpret results
    alpha = 0.05
    if p < alpha:
        print("\nConclusion: Reject Null Hypothesis. There is a significant association between Autonomy Level and Harm Scale.")
    else:
        print("\nConclusion: Fail to Reject Null Hypothesis. No significant association found.")

    # Visualization
    # Colors: Collective (Red/Orange), Individual (Blue/Green)
    # Check column order for colors. Usually alphabetical: Collective, Individual.
    colors = ['#d62728', '#1f77b4'] 
    
    ax = props.plot(kind='bar', stacked=True, color=colors, figsize=(8, 6))
    plt.title('Harm Distribution Scale by AI Autonomy Level')
    plt.xlabel('Autonomy Level')
    plt.ylabel('Proportion of Incidents')
    plt.legend(title='Harm Scale', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Add labels
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', label_type='center', color='white')

    plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Data points for analysis: 180
Autonomy counts:
autonomy_bin
Low     128
High     52
Name: count, dtype: int64
Harm Scale counts:
harm_bin
Individual    113
Collective     67
Name: count, dtype: int64

Contingency Table (Count):
harm_bin      Collective  Individual
autonomy_bin                        
High                  25          27
Low                   42          86

Proportions (Row-wise):
harm_bin      Collective  Individual
autonomy_bin                        
High            0.480769    0.519231
Low             0.328125    0.671875

Chi-Square Statistic: 3.0629
P-value: 0.0801

Conclusion: Fail to Reject Null Hypothesis. No significant association found.


=== Plot Analysis (figure 1) ===
Based on the analysis of the provided plot, here are the detailed findings:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** The chart is designed to compare the relative proportions of two categories of harm ("Individual" and "Collective") across two distinct levels of AI autonomy ("High" and "Low"). By stacking the bars to a total height of 1.0, it facilitates a direct comparison of the composition of incidents rather than absolute counts.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Autonomy Level"
    *   **Categories:** The axis displays categorical data with two specific groups: "High" and "Low".
*   **Y-Axis:**
    *   **Label:** "Proportion of Incidents"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100%).
    *   **Ticks:** The axis is marked at intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
*   **High Autonomy Level:**
    *   The distribution is nearly balanced but leans slightly towards Individual harm.
    *   **Individual Harm (Blue):** Comprises **0.52** (52%) of incidents.
    *   **Collective Harm (Red):** Comprises **0.48** (48%) of incidents.
*   **Low Autonomy Level:**
    *   There is a clear dominance of Individual harm over Collective harm.
    *   **Individual Harm (Blue):** Comprises **0.67** (67%) of incidents.
    *   **Collective Harm (Red):** Comprises **0.33** (33%) of incidents.
*   **Comparison:** As the autonomy level shifts from Low to High, the proportion of "Collective" harm increases significantly (from 0.33 to 0.48), while "Individual" harm decreases proportionally.

### 4. Annotations and Legends
*   **Title:** "Harm Distribution Scale by AI Autonomy Level" describes the chart's content.
*   **Legend:** Located at the top right, titled "Harm Scale."
    *   **Red:** Represents "Collective" harm.
    *   **Blue:** Represents "Individual" harm.
*   **Data Labels:** White numerical annotations are placed centrally within each bar segment, indicating the exact proportion for that specific category (e.g., "0.52", "0.48").

### 5. Statistical Insights
*   **Prevalence of Individual Harm:** Across both autonomy levels, "Individual" harm is the majority category (exceeding 0.50 in both cases). This suggests that regardless of autonomy level, AI incidents are more likely to impact individuals rather than collectives.
*   **Autonomy and Collective Risk:** There is a positive correlation between higher AI autonomy and the scale of collective harm. When AI systems operate with **High Autonomy**, the risk of collective harm is nearly equal to that of individual harm (48% vs 52%). In contrast, **Low Autonomy** systems are twice as likely to cause individual harm (67%) as they are to cause collective harm (33%).
*   **Implication:** This suggests that as AI systems become more autonomous, the nature of the risks they pose shifts from isolated incidents affecting specific users to broader issues affecting groups or society at large.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
