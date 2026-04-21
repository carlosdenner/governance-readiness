# Experiment 79: node_5_9

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_9` |
| **ID in Run** | 79 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T04:42:48.257366+00:00 |
| **Runtime** | 182.7s |
| **Parent** | `node_4_14` |
| **Children** | `node_6_4` |
| **Creation Index** | 80 |

---

## Hypothesis

> The 'Production Gap': There is no statistically significant difference in the
completion rate of AI Impact Assessments between systems in 'Use' versus those
in 'Development', suggesting a failure to gate deployment based on risk
evaluation.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6290 (Maybe True) |
| **Posterior** | 0.2143 (Likely False) |
| **Surprise** | -0.4977 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 24.0 |
| Uncertain | 0.0 |
| Maybe False | 4.0 |
| Definitely False | 2.0 |

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

**Objective:** Assess if operational status drives the completion of impact assessments.

### Steps
- 1. Filter dataset for 'eo13960_scored'.
- 2. Group '16_dev_stage' into 'Operational' (e.g., Use, Maintenance) and 'Pre-Operational' (e.g., Development, Acquisition).
- 3. Convert '52_impact_assessment' into a binary 'Yes/No' variable.
- 4. Create a contingency table (Stage vs. Assessment).
- 5. Run a Chi-square test of independence.

### Deliverables
- Contingency table, Chi-square statistic, p-value, and bar chart of assessment rates by development stage.

---

## Analysis

The experiment successfully tested the 'Production Gap' hypothesis using the EO
13960 dataset.

1. **Data Processing**: The code successfully mapped 1,012 systems into
'Operational' (e.g., Use, Maintenance, Production) and 'Pre-Operational' (e.g.,
Development, Acquisition, Planned) categories using keyword matching.

2. **Statistical Results**: The Chi-square test yielded a statistic of 27.73
with a p-value of < 0.0001, indicating a highly statistically significant
difference between the two groups.

3. **Hypothesis Evaluation**: The null hypothesis (that development stage does
not affect assessment completion) was **rejected**. Operational systems are
significantly more likely to have completed an AI Impact Assessment (8.7%)
compared to systems in development (0.5%).

4. **Key Insight**: The data reveals a 'late-stage' governance model. Impact
assessments are almost non-existent during the design/development phase (2 out
of 371 cases), suggesting they function as final compliance gates rather than
tools for 'security by design.' Furthermore, the overall completion rate is
alarmingly low (<10%) even for deployed systems.

---

## Review

The experiment successfully tested the 'Production Gap' hypothesis using the EO
13960 dataset.

1. **Data Processing**: The code successfully mapped 1,012 systems into
'Operational' (e.g., Use, Maintenance, Production) and 'Pre-Operational' (e.g.,
Development, Acquisition, Planned) categories using keyword matching.

2. **Statistical Results**: The Chi-square test yielded a statistic of 27.73
with a p-value of < 0.0001, indicating a highly statistically significant
difference between the two groups.

3. **Hypothesis Evaluation**: The null hypothesis (that development stage does
not affect assessment completion) was **rejected**. Operational systems are
significantly more likely to have completed an AI Impact Assessment (8.7%)
compared to systems in development (0.5%).

4. **Key Insight**: The data reveals a 'late-stage' governance model. Impact
assessments are almost non-existent during the design/development phase (2 out
of 371 cases), suggesting they function as final compliance gates rather than
tools for 'security by design.' Furthermore, the overall completion rate is
alarmingly low (<10%) even for deployed systems.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print("--- Unique Values Analysis ---")
print("16_dev_stage:", eo_data['16_dev_stage'].unique())
print("52_impact_assessment:", eo_data['52_impact_assessment'].unique())

# Mapping Logic
# Operational: Use, Maintenance
# Pre-Operational: Development, Acquisition, Pilot
# Note: Adjusting based on actual values found in standard EO13960 datasets

def map_stage(stage):
    if pd.isna(stage):
        return np.nan
    stage = str(stage).lower()
    if 'use' in stage or 'maintain' in stage or 'maintenance' in stage or 'production' in stage:
        return 'Operational'
    elif 'dev' in stage or 'acq' in stage or 'pilot' in stage or 'plan' in stage:
        return 'Pre-Operational'
    else:
        return 'Other'

def map_assessment(val):
    if pd.isna(val):
        return 0
    val = str(val).lower()
    # Assuming 'yes' indicates completion. strict mapping.
    if val.strip() == 'yes':
        return 1
    return 0

# Apply mappings
eo_data['Status'] = eo_data['16_dev_stage'].apply(map_stage)
eo_data['Has_Assessment'] = eo_data['52_impact_assessment'].apply(map_assessment)

# Filter out 'Other' status if necessary, or just focus on Op vs Pre-Op
analysis_df = eo_data[eo_data['Status'].isin(['Operational', 'Pre-Operational'])].copy()

# Contingency Table
contingency_table = pd.crosstab(analysis_df['Status'], analysis_df['Has_Assessment'])
contingency_table.columns = ['No Assessment', 'Has Assessment']

print("\n--- Contingency Table ---")
print(contingency_table)

# Calculate percentages
rates = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
print("\n--- Assessment Rates (% of Stage) ---")
print(rates)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-Value: {p:.4f}")

# Interpretation
alpha = 0.05
if p < alpha:
    print("Result: Statistically Significant Difference (Reject Null Hypothesis)")
else:
    print("Result: No Statistically Significant Difference (Fail to Reject Null Hypothesis)")

# Visualization
plt.figure(figsize=(8, 6))
rates['Has Assessment'].plot(kind='bar', color=['skyblue', 'orange'])
plt.title('AI Impact Assessment Completion Rate by Development Stage')
plt.ylabel('Completion Rate (%)')
plt.xlabel('Development Stage')
plt.ylim(0, 100)
plt.xticks(rotation=0)
for i, v in enumerate(rates['Has Assessment']):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Unique Values Analysis ---
16_dev_stage: <StringArray>
[ 'Implementation and Assessment', 'Acquisition and/or Development',
                      'Initiated',                        'Retired',
      'Operation and Maintenance',                  'In production',
                     'In mission',                        'Planned',
                              nan]
Length: 9, dtype: str
52_impact_assessment: <StringArray>
[nan, 'Planned or in-progress.', 'Yes', 'No', 'YES']
Length: 5, dtype: str

--- Contingency Table ---
                 No Assessment  Has Assessment
Status                                        
Operational                585              56
Pre-Operational            369               2

--- Assessment Rates (% of Stage) ---
                 No Assessment  Has Assessment
Status                                        
Operational          91.263651        8.736349
Pre-Operational      99.460916        0.539084

Chi-Square Statistic: 27.7288
P-Value: 0.0000
Result: Statistically Significant Difference (Reject Null Hypothesis)


=== Plot Analysis (figure 1) ===
Based on the analysis of the provided plot, here are the detailed observations:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** The plot is designed to compare the "AI Impact Assessment Completion Rate" across two distinct categories of development stages: "Operational" and "Pre-Operational."

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Title:** "Development Stage"
    *   **Categories:** The axis displays two categorical variables: "Operational" and "Pre-Operational."
*   **Y-Axis (Vertical):**
    *   **Title:** "Completion Rate (%)"
    *   **Range:** The scale ranges from 0 to 100, representing percentage points.
    *   **Ticks:** The axis is marked in intervals of 20 (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Tallest Bar:** The "Operational" category (represented by the light blue bar) has the highest value at 8.7%.
*   **Shortest Bar:** The "Pre-Operational" category (represented by the orange bar) has the lowest value at 0.5%.
*   **Pattern:** There is a significant disparity between the two stages. The completion rate for AI systems that are already operational is substantially higher than for those in the pre-operational phase, though both values are relatively low on the 0-100% scale.

### 4. Annotations and Legends
*   **Value Labels:** Specific data values are annotated directly above each bar for clarity:
    *   "8.7%" is placed above the Operational bar.
    *   "0.5%" is placed above the Pre-Operational bar.
*   **Color Coding:** While there is no separate legend box, the bars are distinct colors—light blue for Operational and orange for Pre-Operational—to differentiate the categories visually.

### 5. Statistical Insights
*   **Low Overall Adherence:** The most striking insight is that the completion rates for AI Impact Assessments are extremely low across the board. Even the highest category (Operational) has not even reached a 10% completion rate.
*   **Stage Discrepancy:** There is a massive relative difference between the stages. Operational projects are roughly **17.4 times** more likely to have a completed impact assessment than Pre-Operational projects ($8.7 / 0.5 = 17.4$).
*   **Implication:** This suggests that impact assessments are possibly being treated as a final "gatekeeping" step prior to deployment or a compliance requirement for active systems, rather than an integral part of the early development (pre-operational) lifecycle. Alternatively, it highlights a potential compliance gap where the vast majority of AI projects, regardless of stage, lack these assessments.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
