# Experiment 267: node_7_12

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_12` |
| **ID in Run** | 267 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:43:38.650268+00:00 |
| **Runtime** | 290.8s |
| **Parent** | `node_6_26` |
| **Children** | `node_8_4` |
| **Creation Index** | 268 |

---

## Hypothesis

> The 'Autonomy-Harm' Escalation: AI systems with 'High' autonomy levels are
significantly more likely to be associated with 'Tangible Harm' (Physical
injury/death) in incident reports compared to 'Low' autonomy systems, which
primarily cause 'Intangible Harm'.

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

**Objective:** Determine if higher autonomy correlates with physical danger in reported incidents.

### Steps
- 1. Load `aiid_incidents`.
- 2. Parse `Autonomy Level` into 'High' (e.g., Autonomous, High) and 'Low' (e.g., Low, Human-in-the-loop).
- 3. Categorize `Tangible Harm` into 'Physical' (Death, Injury) and 'Non-Physical' (Financial, Reputation, None).
- 4. Perform a Chi-square test or Z-test comparing the proportion of Physical Harm in High vs. Low autonomy groups.

### Deliverables
- Proportion comparison, Z-test/Chi-square results, and an interpretation of the physical risk of autonomy.

---

## Analysis

The experiment was successfully executed and provided conclusive evidence to
reject the 'Autonomy-Harm Escalation' hypothesis. By mapping 'Autonomy3' to High
Autonomy (n=53) and 'Autonomy1/2' to Low Autonomy (n=129), the analysis compared
the rates of definitive tangible (physical) harm.

1. **Identical Risk Profiles**: Contrary to the hypothesis that high autonomy
increases physical danger, the rates of tangible harm were statistically
identical between groups: 35.8% for High Autonomy (19/53) versus 35.7% for Low
Autonomy (46/129).

2. **Statistical Significance**: Fisher's Exact Test yielded a p-value of 1.00
and an Odds Ratio of 1.01, indicating zero statistical association between the
level of autonomy and the likelihood of physical harm in this dataset.

3. **Interpretation**: The data suggests that the 'physicality' of a system
(whether it can cause physical harm) is likely a function of its domain (e.g.,
robotics, autonomous vehicles) rather than its level of decision-making
independence. A human-in-the-loop system is just as likely to be involved in a
physical incident as a fully autonomous one within the reported cases.

---

## Review

The experiment was successfully executed and rigorously tested the 'Autonomy-
Harm Escalation' hypothesis. The implementation correctly handled the specific
taxonomy of the AIID dataset ('Autonomy1-3' and 'Tangible Harm' status) after
initial debugging. The statistical analysis using Fisher's Exact Test was
appropriate for the sample sizes involved.

Findings:
1.  **Hypothesis Rejection**: The results definitively reject the hypothesis
that higher autonomy levels correlate with higher rates of tangible (physical)
harm.
2.  **Identical Rates**: The analysis revealed a striking parity in risk
profiles: High Autonomy systems had a physical harm rate of 35.8% (19/53), while
Low Autonomy systems had a rate of 35.7% (46/129).
3.  **Statistical Inference**: With a p-value of 1.00 and an Odds Ratio of 1.01,
the data indicates that within the reported incidents, the level of autonomy is
not a discriminator for physical safety risk.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    # Fallback if file is in parent directory (standard for some envs)
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# Clean and Prepare Data
# We only analyze rows where both Autonomy and Harm are coded
analysis_df = aiid_df.dropna(subset=['Autonomy Level', 'Tangible Harm'])

# Exclude 'unclear' values
analysis_df = analysis_df[
    (analysis_df['Autonomy Level'] != 'unclear') & 
    (analysis_df['Tangible Harm'] != 'unclear')
]

# Mapping Functions
def map_autonomy(val):
    # Autonomy3 is typically 'System is autonomous' (High)
    if val == 'Autonomy3':
        return 'High'
    # Autonomy1 ('System is human-in-the-loop') and Autonomy2 ('System is human-supervised') (Low)
    elif val in ['Autonomy1', 'Autonomy2']:
        return 'Low'
    return None

def map_harm(val):
    # Hypothesis focuses on 'Tangible Harm' (Physical injury/death)
    # We define 'Tangible' as cases where it definitively occurred.
    if val == 'tangible harm definitively occurred':
        return 'Tangible (Physical)'
    else:
        # Includes near-misses, issues, no tangible harm
        return 'Intangible/None'

# Apply mappings
analysis_df['Autonomy_Group'] = analysis_df['Autonomy Level'].apply(map_autonomy)
analysis_df['Harm_Group'] = analysis_df['Tangible Harm'].apply(map_harm)

# Drop any rows that failed mapping (though previous filter should catch them)
analysis_df = analysis_df.dropna(subset=['Autonomy_Group', 'Harm_Group'])

# Generate Summary Stats
print(f"Total Incidents Analyzed: {len(analysis_df)}")
print(analysis_df['Autonomy_Group'].value_counts())

# Create Contingency Table
contingency = pd.crosstab(analysis_df['Autonomy_Group'], analysis_df['Harm_Group'])
print("\n--- Contingency Table ---")
print(contingency)

# Calculate Rates
high_auto_total = contingency.loc['High'].sum() if 'High' in contingency.index else 0
high_auto_tangible = contingency.loc['High', 'Tangible (Physical)'] if 'High' in contingency.index and 'Tangible (Physical)' in contingency.columns else 0
rate_high = high_auto_tangible / high_auto_total if high_auto_total > 0 else 0

low_auto_total = contingency.loc['Low'].sum() if 'Low' in contingency.index else 0
low_auto_tangible = contingency.loc['Low', 'Tangible (Physical)'] if 'Low' in contingency.index and 'Tangible (Physical)' in contingency.columns else 0
rate_low = low_auto_tangible / low_auto_total if low_auto_total > 0 else 0

print(f"\nHigh Autonomy Tangible Harm Rate: {rate_high:.1%} ({high_auto_tangible}/{high_auto_total})")
print(f"Low Autonomy Tangible Harm Rate:  {rate_low:.1%} ({low_auto_tangible}/{low_auto_total})")

# Statistical Test
# Fisher's Exact Test is appropriate given the relatively small sample sizes in some cells
if contingency.size == 4:
    odds_ratio, p_value = stats.fisher_exact(contingency.loc[['High', 'Low'], ['Tangible (Physical)', 'Intangible/None']])
    print(f"\nFisher's Exact Test P-value: {p_value:.5f}")
    print(f"Odds Ratio: {odds_ratio:.2f}")
    
    if p_value < 0.05:
        print("Result: Significant correlation found.")
    else:
        print("Result: No significant correlation found.")
else:
    print("\nInsufficient data structure for 2x2 statistical test.")

# Visualization
plt.figure(figsize=(8, 5))
plt.bar(['High Autonomy', 'Low Autonomy'], [rate_high, rate_low], color=['#d62728', '#1f77b4'])
plt.ylabel('Rate of Tangible Harm')
plt.title('Tangible Harm Rates by Autonomy Level')
plt.ylim(0, 1.0)
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total Incidents Analyzed: 182
Autonomy_Group
Low     129
High     53
Name: count, dtype: int64

--- Contingency Table ---
Harm_Group      Intangible/None  Tangible (Physical)
Autonomy_Group                                      
High                         34                   19
Low                          83                   46

High Autonomy Tangible Harm Rate: 35.8% (19/53)
Low Autonomy Tangible Harm Rate:  35.7% (46/129)

Fisher's Exact Test P-value: 1.00000
Odds Ratio: 1.01
Result: No significant correlation found.


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot (or Bar Chart).
*   **Purpose:** This plot is designed to compare a quantitative variable ("Rate of Tangible Harm") across two distinct categorical groups ("High Autonomy" and "Low Autonomy").

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Rate of Tangible Harm"
    *   **Range:** The axis spans from **0.0 to 1.0**, representing a probability or percentage (0% to 100%).
    *   **Ticks:** The axis is marked at intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).
*   **X-Axis:**
    *   **Label:** The axis does not have a collective label (e.g., "Autonomy Level"), but the categories are clearly labeled as **"High Autonomy"** and **"Low Autonomy"**.

### 3. Data Trends
*   **Values:**
    *   **High Autonomy (Red Bar):** The height of the bar indicates a rate of tangible harm of approximately **0.36 to 0.37**.
    *   **Low Autonomy (Blue Bar):** The height of the bar indicates a rate of tangible harm of approximately **0.35 to 0.36**.
*   **Comparison:** The two bars are nearly identical in height. The "High Autonomy" group shows a negligibly higher rate of tangible harm compared to the "Low Autonomy" group. The difference appears visually minimal (likely less than 0.02 or 2%).

### 4. Annotations and Legends
*   **Title:** The chart is titled **"Tangible Harm Rates by Autonomy Level,"** clearly defining the subject of the comparison.
*   **Color Coding:** Distinct colors are used to differentiate the categories visually:
    *   **Red:** High Autonomy
    *   **Blue:** Low Autonomy
*   There is no separate legend box, as the x-axis labels serve to identify the data points directly.

### 5. Statistical Insights
*   **Equivalence of Risk:** Based strictly on this visualization, the level of autonomy (High vs. Low) appears to have little to no impact on the rate of tangible harm. Both scenarios result in a harm rate hovering around 36%.
*   **Magnitude:** The absolute rate of harm is substantial but stays below the midpoint (0.5), residing in the lower-middle quartile of the scale.
*   **Research Implication:** If the goal of the underlying study was to prove that High Autonomy systems are safer (or more dangerous) than Low Autonomy systems, this data does not support a significant difference. It suggests that factors other than the autonomy level itself may be driving the rate of tangible harm.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
