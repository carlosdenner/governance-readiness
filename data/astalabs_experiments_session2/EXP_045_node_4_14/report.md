# Experiment 45: node_4_14

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_14` |
| **ID in Run** | 45 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:11:56.554668+00:00 |
| **Runtime** | 257.7s |
| **Parent** | `node_3_16` |
| **Children** | `node_5_9`, `node_5_74` |
| **Creation Index** | 46 |

---

## Hypothesis

> Autonomy-Harm Escalation: In the AIID dataset, higher levels of system autonomy
(e.g., 'Autonomy3') are statistically associated with a higher degree of harm
realization (e.g., 'Harm Event' > 'Near Miss') compared to lower autonomy
levels.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6210 (Maybe True) |
| **Posterior** | 0.8709 (Likely True) |
| **Surprise** | +0.2999 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 20.0 |
| Uncertain | 5.0 |
| Maybe False | 5.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 60.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Correctly map the specific labels found in the AIID dataset ('Autonomy1-3', 'Harm Event/Near-miss') to ordinal scales and re-test the correlation between autonomy and harm severity.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for rows where `source_table` is 'aiid_incidents'.
- 2. Print the unique values of `Autonomy Level` and `AI Harm Level` to verify the labels.
- 3. Create an ordinal mapping for `Autonomy Level`: {'Autonomy1': 1, 'Autonomy2': 2, 'Autonomy3': 3}. Convert non-matches to NaN.
- 4. Create an ordinal mapping for `AI Harm Level` based on the realization of harm: {'none': 0, 'AI tangible harm near-miss': 1, 'AI tangible harm issue': 2, 'AI tangible harm event': 3}. Convert non-matches to NaN.
- 5. Drop rows with NaN in these two new columns.
- 6. Perform a Spearman Rank Correlation test between the two ordinal variables.
- 7. Generate a boxplot of the mapped Harm Level grouped by the mapped Autonomy Level.

### Deliverables
- Printed unique values for verification; Spearman correlation coefficient and p-value; Boxplot of Harm Realization by Autonomy Level.

---

## Analysis

The experiment successfully tested the 'Autonomy-Harm Escalation' hypothesis
using the correct data mappings for the AIID dataset.

1.  **Data Recovery**: By mapping specific labels (e.g., 'Autonomy1', 'AI
tangible harm event') to ordinal scales, the analysis recovered 177 valid data
points for correlation testing.

2.  **Statistical Results**: The Spearman Rank Correlation yielded a coefficient
of **0.2331** with a p-value of **0.0018**. Since the p-value is well below the
0.05 threshold, the result is **statistically significant**.

3.  **Hypothesis Evaluation**: The hypothesis is **supported**. There is a
positive, statistically significant correlation between system autonomy and the
severity of realized harm. Higher autonomy levels are associated with worse
outcomes.

4.  **Detailed Insight**: The contingency table reveals the nature of this
escalation:
    *   **Level 1 (Assist)**: The safest category. 80% (78/98) of incidents
resulted in 'None' (no harm).
    *   **Level 2 (Select)**: Appears highly volatile. Only 37% (10/27) resulted
in 'None', with the median outcome shifting up to 'Issue' (2).
    *   **Level 3 (Act)**: Shows a mixed profile. While a majority (31/52)
resulted in 'None', a significant portion (29%) resulted in 'Events' (Critical
Harm), compared to only 15% for Level 1.

5.  **Conclusion**: Moving from assistive to autonomous systems correlates with
an increased risk of tangible harm events, with 'Level 2' (System Selects/Human-
in-the-loop) potentially representing a particularly risky transitional state in
this dataset.

---

## Review

The experiment was successfully executed. The code correctly mapped the specific
string labels from the AIID dataset ('Autonomy1'...'Autonomy3',
'none'...'event') to ordinal scales, overcoming the data sparsity issue
encountered in the previous attempt. With 177 valid data points, the Spearman
correlation test (rho=0.2331, p=0.0018) provided statistically significant
evidence to support the hypothesis.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

print(f"Loaded {len(aiid_df)} rows from AIID incidents.")

# Define columns
autonomy_col = 'Autonomy Level'
harm_col = 'AI Harm Level'

# Print unique values for verification
print("Unique Autonomy values:", aiid_df[autonomy_col].unique())
print("Unique Harm values:", aiid_df[harm_col].unique())

# Define mappings based on observed values
autonomy_mapping = {
    'Autonomy1': 1,  # System provides information/assists
    'Autonomy2': 2,  # System selects action/human in loop
    'Autonomy3': 3   # System acts/autonomous
}

# Mapping logic: None < Near-miss < Issue < Event
harm_mapping = {
    'none': 0,
    'AI tangible harm near-miss': 1,
    'AI tangible harm issue': 2,
    'AI tangible harm event': 3
}

# Apply mappings
aiid_df['Autonomy_Ordinal'] = aiid_df[autonomy_col].map(autonomy_mapping)
aiid_df['Harm_Ordinal'] = aiid_df[harm_col].map(harm_mapping)

# Drop NaNs (including 'unclear' or other unmapped values)
analysis_df = aiid_df.dropna(subset=['Autonomy_Ordinal', 'Harm_Ordinal'])

print(f"\nData points available for analysis after cleaning: {len(analysis_df)}")

if len(analysis_df) > 5:
    # Spearman Correlation
    corr, p_value = spearmanr(analysis_df['Autonomy_Ordinal'], analysis_df['Harm_Ordinal'])
    print(f"\nSpearman Correlation: {corr:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    if p_value < 0.05:
        print("Result: Statistically significant correlation.")
    else:
        print("Result: Not statistically significant.")

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Autonomy_Ordinal', y='Harm_Ordinal', data=analysis_df)
    plt.title('Harm Realization by Autonomy Level')
    plt.xlabel('Autonomy Level (1=Assist, 2=Select, 3=Act)')
    plt.ylabel('Harm Level (0=None, 1=Near-miss, 2=Issue, 3=Event)')
    
    # Set x-tick labels manually for clarity
    plt.xticks(ticks=[0, 1, 2], labels=['1: Assist', '2: Select', '3: Act'])
    plt.yticks(ticks=[0, 1, 2, 3], labels=['None', 'Near-miss', 'Issue', 'Event'])
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    # Crosstab for detailed counts
    print("\nContingency Table:")
    print(pd.crosstab(analysis_df['Autonomy_Ordinal'], analysis_df['Harm_Ordinal']))
else:
    print("Insufficient data points to perform correlation analysis.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded 1362 rows from AIID incidents.
Unique Autonomy values: <StringArray>
['Autonomy1', 'Autonomy2', 'Autonomy3', 'unclear', nan]
Length: 5, dtype: str
Unique Harm values: <StringArray>
[                      'none',     'AI tangible harm event',
                    'unclear', 'AI tangible harm near-miss',
                          nan,     'AI tangible harm issue']
Length: 6, dtype: str

Data points available for analysis after cleaning: 177

Spearman Correlation: 0.2331
P-value: 1.7966e-03
Result: Statistically significant correlation.

Contingency Table:
Harm_Ordinal      0.0  1.0  2.0  3.0
Autonomy_Ordinal                    
1.0                78    3    2   15
2.0                10    3    3   11
3.0                31    3    3   15


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (also known as a box-and-whisker plot).
*   **Purpose:** To display the distribution of data based on a five-number summary: minimum, first quartile (Q1), median, third quartile (Q3), and maximum. It is used here to compare the distribution of "Harm Level" across three different "Autonomy Levels."

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Title:** "Autonomy Level (1=Assist, 2=Select, 3=Act)"
    *   **Labels/Categories:** "1: Assist", "2: Select", "3: Act".
    *   **Range:** Categorical, representing three distinct stages of system autonomy.
*   **Y-Axis (Vertical):**
    *   **Title:** "Harm Level (0=None, 1=Near-miss, 2=Issue, 3=Event)"
    *   **Labels:** "None", "Near-miss", "Issue", "Event".
    *   **Range:** Ordinal/Categorical mapped to numerical values 0 through 3.

### 3. Data Trends
*   **1: Assist:**
    *   The box plot is "collapsed" at the bottom line ("None"). This indicates that the median, 25th percentile, and 75th percentile are all 0 (None).
    *   There is almost no variance in the main distribution; however, there are distinct **outliers** (represented by circles) at the "Near-miss," "Issue," and "Event" levels. This suggests that while harm is rare at this level, it does occur sporadically.
*   **2: Select:**
    *   The distribution shifts significantly upwards. The box (representing the Interquartile Range) spans from "None" to "Event."
    *   The **median line** is visible at the "Issue" level. This indicates that for the "Select" autonomy level, the typical outcome is an "Issue," representing a much higher risk profile than the "Assist" level.
*   **3: Act:**
    *   The box is very large, covering the entire range from "None" to "Event."
    *   Unlike the "Select" category, the median line is not distinct in the middle, suggesting it may overlap with the upper or lower quartiles (likely the upper, given the trend, or indicating a very wide, uniform spread). The variability here is maximum, meaning outcomes are highly unpredictable, ranging frequently across the entire spectrum of harm.

### 4. Annotations and Legends
*   **Outliers:** Represented by open circles (`o`). These are prominent in the "1: Assist" category, showing data points that fall statistically far from the rest of the cluster (which is at zero).
*   **Gridlines:** Horizontal dashed gray lines help visually align the box heights with the categorical harm levels on the Y-axis.
*   **Color:** The boxes are shaded in a uniform blue, indicating they belong to the same dataset grouping, differentiated only by the X-axis category.

### 5. Statistical Insights
*   **Positive Correlation with Risk:** There is a clear positive correlation between the level of autonomy and the realization of harm. As the system moves from "Assist" to "Select" to "Act," the severity and frequency of harmful events increase.
*   **Stability vs. Instability:**
    *   **Level 1 (Assist)** is the most stable and safest system, with the vast majority of operations resulting in "None" (no harm).
    *   **Level 2 (Select)** introduces significant instability, raising the median outcome to "Issue."
    *   **Level 3 (Act)** shows high volatility. The large Interquartile Range (IQR) indicates that the system's performance is inconsistent; it is just as likely to result in a severe "Event" as it is to have "None," making it the riskiest configuration.
*   **Conclusion:** Increasing autonomy in this specific context appears to degrade safety performance, moving the system from a "fail-safe" state (Level 1) to a state where issues and events become common (Levels 2 and 3).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
