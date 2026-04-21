# Experiment 44: node_4_13

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_13` |
| **ID in Run** | 44 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:05:09.434732+00:00 |
| **Runtime** | 251.0s |
| **Parent** | `node_3_13` |
| **Children** | `node_5_25`, `node_5_38` |
| **Creation Index** | 45 |

---

## Hypothesis

> The 'Paperwork-Operations' Gap: Federal agencies are significantly more likely
to complete bureaucratic governance steps (e.g., 'Impact Assessments') than
operational assurance steps (e.g., 'Post-Deployment Monitoring'), indicating a
compliance-first rather than safety-first approach.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7984 (Likely True) |
| **Posterior** | 0.2720 (Likely False) |
| **Surprise** | -0.6317 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 7.0 |
| Maybe True | 23.0 |
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

**Objective:** Quantify the disparity between pre-deployment paperwork and post-deployment monitoring.

### Steps
- 1. Filter 'eo13960_scored' for valid entries.
- 2. Extract paired columns: '52_impact_assessment' and '56_monitor_postdeploy'.
- 3. Map values to binary (Yes vs No/NA).
- 4. Create a contingency table for the paired data (Assessment=Yes/Monitor=No vs Assessment=No/Monitor=Yes).
- 5. Perform McNemar's test for paired nominal data to test if the discordance is directional.

### Deliverables
- Contingency table of Assessment vs Monitoring; McNemar's test statistic and p-value.

---

## Analysis

The experiment successfully tested the 'Paperwork-Operations Gap' hypothesis
using the EO 13960 dataset. Unlike the previous attempt, the programmer
correctly implemented text parsing logic for the '56_monitor_postdeploy' column,
identifying positive monitoring indicators (e.g., 'automated', 'established')
instead of relying on a strict 'Yes' match.

The results definitively rejected the hypothesis. Contrary to the expectation
that agencies prioritize paperwork over operations, the data showed the opposite
trend: 51 systems had operational monitoring without a corresponding impact
assessment ('Ops Only'), while only 5 systems had an impact assessment without
monitoring ('Paperwork Only'). McNemar's test (p < 0.001) confirmed this
difference is statistically significant. The findings suggest an 'Operational
Shadow' where legacy or operational systems exist with monitoring protocols but
lack the formal governance documentation (Impact Assessments) required by newer
mandates.

---

## Review

The experiment successfully tested the 'Paperwork-Operations Gap' hypothesis
using the EO 13960 dataset. The programmer effectively resolved the previous
data parsing issue by implementing a keyword-based mapping strategy for the
'56_monitor_postdeploy' column, correctly categorizing descriptive entries
(e.g., 'Established Process', 'Automated') as valid monitoring.

The results definitively rejected the hypothesis. Statistical analysis using
McNemar's test (p < 0.001) revealed a significant 'Reverse Gap': agencies are
far more likely to have operational monitoring without a formal impact
assessment (51 cases) than to have an impact assessment without monitoring (5
cases). This finding suggests that federal AI governance suffers less from
'paperwork without action' and more from 'action without documentation'—a state
where legacy or engineering-led monitoring exists but lacks the formal
governance artifacts required by modern compliance mandates.

---

## Code

```python
import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Columns
col_assess = '52_impact_assessment'
col_monitor = '56_monitor_postdeploy'

# Define Mapping Logic
def map_assessment(val):
    s = str(val).lower().strip()
    if s in ['yes', 'yes.']:
        return 1
    # 'Planned or in-progress', 'No', 'nan' -> 0
    return 0

def map_monitoring(val):
    s = str(val).lower().strip()
    if pd.isna(val) or s == 'nan':
        return 0
    
    # Positive indicators based on previous exploration
    if any(x in s for x in ['intermittent', 'automated', 'established process']):
        return 1
    
    # Negative indicators
    if any(x in s for x in ['no monitoring', 'not safety', 'under development']):
        return 0
        
    return 0

# Apply Mapping
eo_data['assess_bin'] = eo_data[col_assess].apply(map_assessment)
eo_data['monitor_bin'] = eo_data[col_monitor].apply(map_monitoring)

# Create Contingency Table
ct = pd.crosstab(eo_data['assess_bin'], eo_data['monitor_bin'])
ct = ct.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

print("\nContingency Table (Assessment vs Monitoring):")
print(ct)
print("(Row=Assessment, Col=Monitoring; 0=No, 1=Yes)")

# Cells
no_assess_no_mon = ct.loc[0, 0]
no_assess_yes_mon = ct.loc[0, 1]
yes_assess_no_mon = ct.loc[1, 0]
yes_assess_yes_mon = ct.loc[1, 1]

# McNemar's Test
result = mcnemar(ct, exact=False, correction=True)

print(f"\n--- McNemar's Test Results ---")
print(f"Statistic (chi-squared): {result.statistic:.4f}")
print(f"P-value: {result.pvalue:.4e}")

# Analysis
total = len(eo_data)
print(f"\n--- Detailed Analysis ---")
print(f"Total Systems: {total}")
print(f"Assessment Completed: {eo_data['assess_bin'].sum()} ({(eo_data['assess_bin'].sum()/total)*100:.1f}%)")
print(f"Monitoring Established: {eo_data['monitor_bin'].sum()} ({(eo_data['monitor_bin'].sum()/total)*100:.1f}%)")

print(f"\nDiscordant Pairs:")
print(f"Paperwork Only (Assess=Yes, Mon=No): {yes_assess_no_mon}")
print(f"Ops Only (Assess=No, Mon=Yes): {no_assess_yes_mon}")

if yes_assess_no_mon > no_assess_yes_mon:
    ratio = yes_assess_no_mon / (no_assess_yes_mon if no_assess_yes_mon > 0 else 1)
    print(f"Result: The 'Paperwork-Operations' Gap is confirmed. Agencies are {ratio:.2f}x more likely to have Assessment without Monitoring than vice versa.")
else:
    print("Result: No significant Paperwork-Operations Gap detected in the expected direction.")

# Visualization
labels = ['Assess & Mon', 'Assess Only', 'Mon Only', 'Neither']
counts = [yes_assess_yes_mon, yes_assess_no_mon, no_assess_yes_mon, no_assess_no_mon]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, counts, color=['#2ca02c', '#ff7f0e', '#1f77b4', '#7f7f7f'])
plt.title('The Paperwork-Operations Gap (EO 13960)')
plt.ylabel('Number of AI Systems')
plt.bar_label(bars)
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: 
Contingency Table (Assessment vs Monitoring):
monitor_bin     0   1
assess_bin           
0            1645  51
1               5  56
(Row=Assessment, Col=Monitoring; 0=No, 1=Yes)

--- McNemar's Test Results ---
Statistic (chi-squared): 36.1607
P-value: 1.8170e-09

--- Detailed Analysis ---
Total Systems: 1757
Assessment Completed: 61 (3.5%)
Monitoring Established: 107 (6.1%)

Discordant Pairs:
Paperwork Only (Assess=Yes, Mon=No): 5
Ops Only (Assess=No, Mon=Yes): 51
Result: No significant Paperwork-Operations Gap detected in the expected direction.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is a detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** This chart compares the count of AI systems across four distinct categories regarding their status of assessment and monitoring, likely in the context of compliance with Executive Order (EO) 13960.

### 2. Axes
*   **X-Axis:**
    *   **Labels:** The axis categorizes the systems into four groups: "Assess & Mon" (Assessed and Monitored), "Assess Only", "Mon Only" (Monitored Only), and "Neither".
    *   **Nature:** Categorical/Nominal data.
*   **Y-Axis:**
    *   **Title:** "Number of AI Systems".
    *   **Range:** The scale begins at 0 and extends to 1600, with tick marks every 200 units. The effective data range goes up to the maximum value of 1645.

### 3. Data Trends
*   **Dominant Category (Tallest Bar):** The "Neither" category is overwhelmingly the largest, with a count of **1645**. This indicates that the vast majority of AI systems in this dataset have undergone neither assessment nor monitoring.
*   **Minority Categories:**
    *   **"Assess Only"** is the rarest category with only **5** systems.
    *   **"Assess & Mon"** and **"Mon Only"** are comparable in size but very low in absolute numbers, with **56** and **51** systems respectively.
*   **Pattern:** There is a drastic disparity between the number of systems that have some form of oversight (assessment or monitoring) and those that have none. The visual weight of the grey "Neither" bar highlights a significant lack of coverage or compliance.

### 4. Annotations and Legends
*   **Chart Title:** "The Paperwork-Operations Gap (EO 13960)" suggests the chart is illustrating a failure or gap in the implementation of the Executive Order.
*   **Data Labels:** Exact values are annotated on top of each bar (56, 5, 51, 1645), allowing for precise reading of the data without estimating from the y-axis.
*   **Color Coding:** The bars are distinct colors to differentiate the categories:
    *   **Green:** Assess & Mon
    *   **Orange:** Assess Only
    *   **Blue:** Mon Only
    *   **Grey:** Neither

### 5. Statistical Insights
*   **Total Population:** Summing the values ($56 + 5 + 51 + 1645$) gives a total of **1,757** AI systems represented in this dataset.
*   **The "Gap":**
    *   **93.6%** of the AI systems ($1645/1757$) fall into the "Neither" category, meaning they are completely outside the assessment and monitoring framework depicted here.
    *   Only **3.2%** of systems ($56/1757$) are fully covered ("Assess & Mon").
*   **Operational Insight:** The title "Paperwork-Operations Gap" is statistically substantiated by the data. If the goal of EO 13960 is to assess and monitor AI systems, the current operational reality is missing that target for over 90% of the inventory.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
