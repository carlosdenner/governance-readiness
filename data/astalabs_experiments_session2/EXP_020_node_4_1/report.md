# Experiment 20: node_4_1

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_1` |
| **ID in Run** | 20 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T01:56:47.688401+00:00 |
| **Runtime** | 248.8s |
| **Parent** | `node_3_2` |
| **Children** | `node_5_24`, `node_5_86` |
| **Creation Index** | 21 |

---

## Hypothesis

> Crisis Governance Regression: AI systems implemented during the COVID-19
emergency (2020-2021) exhibit statistically lower rates of 'Impact Assessment'
compliance compared to systems implemented in the post-crisis period
(2022-2023).

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

**Objective:** Determine if the urgency of the pandemic era correlated with reduced procedural governance (Impact Assessments).

### Steps
- 1. Load `astalabs_discovery_all_data.csv` (EO13960 subset).
- 2. Parse `20_date_implemented` or `18_date_initiated` to extract the Year.
- 3. Define two groups: Crisis (2020-2021) and Post-Crisis (2022-2023).
- 4. Compare the proportion of `52_impact_assessment` = 'Yes' between groups using a Chi-Square test or T-test.

### Deliverables
- Time-series plot of Impact Assessment rates; Statistical comparison between time periods.

---

## Analysis

The experiment successfully analyzed 311 valid EO13960 records from 2020 to 2023
to test the 'Crisis Governance Regression' hypothesis. The results **do not
support** the hypothesis that the crisis period (2020-2021) had statistically
lower Impact Assessment compliance than the post-crisis period (2022-2023).

1. **Statistical Independence**: The Chi-Square test yielded a p-value of 1.000,
indicating absolutely no statistical difference between the two periods. The
compliance rates were effectively identical: 8.70% for the Crisis period and
8.68% for the Post-Crisis period.

2. **Overall Low Compliance**: The analysis reveals a consistently low standard
of governance, with less than 9% of AI systems in either period having a
completed Impact Assessment.

3. **Yearly Nuance**: The time-series visualization provides critical context
masked by the period grouping. While the 'Crisis' average was 8.7%, 2020 itself
had the lowest recorded compliance (2.7%), which surged to a peak in 2021
(12.7%). The 'Post-Crisis' period shows a downward trend from that 2021 peak
(11.7% in 2022 to 7.0% in 2023).

---

## Review

The experiment was executed faithfully according to the plan. The programmer
correctly filtered the EO13960 dataset, parsed the implementation dates to
define the specific time periods (Crisis: 2020-2021, Post-Crisis: 2022-2023),
and applied appropriate data cleaning to the 'Impact Assessment' field to
distinguish completed assessments from other states. The statistical analysis
(Chi-Square) and visualizations (Bar Chart and Time Series) were correctly
implemented and interpreted.

Key Findings:
1.  **Hypothesis Rejection:** The results do not support the hypothesis. The
compliance rates for Impact Assessments were statistically identical (p=1.00)
between the Crisis period (8.70%) and the Post-Crisis period (8.68%).
2.  **Granular Temporal Dynamics:** The yearly time-series plot reveals that the
aggregation masked significant volatility. 2020 had the lowest compliance
(2.7%), but 2021 saw a sharp peak (12.7%), averaging out to 8.7%. Similarly, the
post-crisis period saw a decline from 11.7% in 2022 to 7.0% in 2023.
3.  **Low Baseline:** Regardless of the period, the overall governance maturity
is low, with fewer than 13% of systems having a completed impact assessment in
any given year.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Total EO13960 records: {len(eo_data)}")

# Target columns
date_col = '20_date_implemented'
impact_col = '52_impact_assessment'

# Check if columns exist
if date_col not in eo_data.columns or impact_col not in eo_data.columns:
    print(f"Error: Missing required columns. Available columns: {eo_data.columns.tolist()}")
    exit()

# Parse dates
# The column might contain various formats, force coercion
eo_data['year'] = pd.to_datetime(eo_data[date_col], errors='coerce').dt.year

# Filter for valid years in the range of interest (2020-2023)
eo_data = eo_data[eo_data['year'].isin([2020, 2021, 2022, 2023])]
print(f"Records in 2020-2023 with valid dates: {len(eo_data)}")

# Define periods
def get_period(year):
    if year in [2020, 2021]:
        return 'Crisis (2020-2021)'
    elif year in [2022, 2023]:
        return 'Post-Crisis (2022-2023)'
    return None

eo_data['period'] = eo_data['year'].apply(get_period)

# Clean Impact Assessment column
# Normalize to boolean: True if 'Yes', False otherwise
# Let's inspect unique values first to be safe
print(f"Unique values in {impact_col}: {eo_data[impact_col].unique()}")

# normalizing
valid_yes = ['yes', 'true', '1', 'y']
eo_data['has_impact_assessment'] = eo_data[impact_col].astype(str).str.lower().isin(valid_yes)

# Group analysis
summary = eo_data.groupby('period')['has_impact_assessment'].agg(['count', 'sum', 'mean']).rename(columns={'count': 'Total', 'sum': 'Compliant', 'mean': 'Rate'})
print("\nSummary Statistics by Period:")
print(summary)

# Chi-Square Test
# Contingency table: [[Crisis_Compliant, Crisis_NonCompliant], [Post_Compliant, Post_NonCompliant]]
crisis_stats = summary.loc['Crisis (2020-2021)']
post_stats = summary.loc['Post-Crisis (2022-2023)']

contingency_table = [
    [int(crisis_stats['Compliant']), int(crisis_stats['Total'] - crisis_stats['Compliant'])],
    [int(post_stats['Compliant']), int(post_stats['Total'] - post_stats['Compliant'])]
]

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\n--- Statistical Test Results ---")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-Value: {p:.4f}")

if p < 0.05:
    print("Result: Significant difference found.")
else:
    print("Result: No significant difference found.")

# Visualization
plt.figure(figsize=(10, 6))

# 1. Bar chart for Period comparison
periods = summary.index
rates = summary['Rate'] * 100

colors = ['#ff9999', '#66b3ff']
bars = plt.bar(periods, rates, color=colors, edgecolor='black', alpha=0.7)

plt.title('Impact Assessment Compliance Rate: Crisis vs Post-Crisis', fontsize=14)
plt.ylabel('Compliance Rate (%)', fontsize=12)
plt.ylim(0, 100)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# 2. Time Series Plot by Year
yearly_summary = eo_data.groupby('year')['has_impact_assessment'].mean() * 100
plt.figure(figsize=(10, 6))
plt.plot(yearly_summary.index, yearly_summary.values, marker='o', linestyle='-', color='purple', linewidth=2)
plt.title('Impact Assessment Compliance Rate over Time (2020-2023)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Compliance Rate (%)', fontsize=12)
plt.ylim(0, 100)
plt.xticks([2020, 2021, 2022, 2023])
plt.grid(True, linestyle='--', alpha=0.5)
for x, y in zip(yearly_summary.index, yearly_summary.values):
    plt.text(x, y + 2, f'{y:.1f}%', ha='center', fontweight='bold')
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total EO13960 records: 1757
Records in 2020-2023 with valid dates: 311
Unique values in 52_impact_assessment: <StringArray>
[nan, 'Planned or in-progress.', 'Yes', 'No']
Length: 4, dtype: str

Summary Statistics by Period:
                         Total  Compliant      Rate
period                                             
Crisis (2020-2021)          92          8  0.086957
Post-Crisis (2022-2023)    219         19  0.086758

--- Statistical Test Results ---
Chi-Square Statistic: 0.0000
P-Value: 1.0000
Result: No significant difference found.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Bar Chart (Vertical Bar Plot).
*   **Purpose:** The plot is designed to compare a specific metric ("Compliance Rate") across two distinct categorical time periods ("Crisis" and "Post-Crisis").

### 2. Axes
*   **Y-Axis (Vertical):**
    *   **Title:** "Compliance Rate (%)"
    *   **Value Range:** 0 to 100.
    *   **Intervals:** The axis is marked in increments of 20 (0, 20, 40, 60, 80, 100).
*   **X-Axis (Horizontal):**
    *   **Labels:** The axis represents two time categories: "Crisis (2020-2021)" and "Post-Crisis (2022-2023)".

### 3. Data Trends
*   **Pattern:** The most notable trend is the lack of change. Both bars are identical in height.
*   **Values:**
    *   **Crisis (2020-2021):** The bar reaches a height of 8.7%.
    *   **Post-Crisis (2022-2023):** The bar also reaches a height of 8.7%.
*   **Relative Magnitude:** Both bars are visually very short compared to the total scale (100%), sitting well below the first major gridline of 20%.

### 4. Annotations and Legends
*   **Main Title:** "Impact Assessment Compliance Rate: Crisis vs Post-Crisis".
*   **Data Labels:** There are explicit numerical annotations placed directly on top of each bar ("8.7%"), providing the exact value for the viewer so estimation is not required.
*   **Color Coding:**
    *   The "Crisis" period is represented by a light pink/red bar.
    *   The "Post-Crisis" period is represented by a light blue bar.
*   **Gridlines:** Horizontal dashed gridlines are included at 20%, 40%, 60%, and 80% to assist with visual scaling.

### 5. Statistical Insights
*   **Stagnation:** There has been absolutely no change in the Impact Assessment Compliance Rate between the crisis period (2020-2021) and the post-crisis period (2022-2023). The transition out of the crisis did not lead to an improvement or a decline in compliance.
*   **Low Compliance:** The overall compliance rate is extremely low (8.7%). Assuming a target of 100% (or even 50%), a rate below 10% indicates that impact assessments are rarely being completed or complying with standards in both periods.
*   **Conclusion:** Whatever factors were limiting compliance during the crisis (e.g., resource diversion, emergency protocols) appear to persist in the post-crisis environment, or new factors have arisen to keep the rate at the exact same low level.
==================================================

=== Plot Analysis (figure 2) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Line Plot (Time Series).
*   **Purpose:** The plot is designed to visualize the trend of "Impact Assessment Compliance Rate" over a four-year period, allowing the viewer to observe fluctuations, peaks, and declines in compliance percentages year over year.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Year"
    *   **Range:** Discrete yearly data points from **2020 to 2023**.
*   **Y-Axis:**
    *   **Label:** "Compliance Rate (%)"
    *   **Range:** Continuous numerical scale from **0 to 100**.
    *   **Scale:** The axis is marked in increments of 20 (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Overall Trend:** The data shows a significant spike in compliance followed by a gradual decline. The values remain in the bottom quartile of the graph (below 25%) throughout the entire timeline.
*   **Initial Growth (2020-2021):** There was a sharp increase in compliance between 2020 and 2021, jumping from the lowest recorded value to the highest.
*   **Peak:** The maximum compliance rate occurred in **2021**.
*   **Decline (2021-2023):** Following the 2021 peak, the rate slightly decreased in 2022 and continued to drop further in 2023.

### 4. Annotations and Legends
*   **Data Point Labels:** Each data point is explicitly annotated with its exact percentage value in bold text, positioned slightly above the marker. This allows for precise reading of data that might otherwise be difficult to discern due to the wide Y-axis scale.
    *   2020: **2.7%**
    *   2021: **12.7%**
    *   2022: **11.7%**
    *   2023: **7.0%**
*   **Visual Elements:** The trend line is colored purple with solid circular markers indicating the specific data points. A light gray, dashed grid is used to assist in aligning points with the axes.

### 5. Statistical Insights
*   **Low Overall Compliance:** Despite the Y-axis extending to 100%, the compliance rate never exceeds 12.7%. This indicates that throughout the observed period, the vast majority of impact assessments (over 87% even at the peak) were non-compliant.
*   **Volatility:**
    *   **2020 to 2021:** The rate increased by **10 percentage points** (an approximate 370% relative increase from the base of 2.7%).
    *   **2021 to 2023:** The rate has since dropped by **5.7 percentage points** from the peak.
*   **Current Trajectory:** The trend is currently negative. After stabilizing slightly in 2022 (a drop of only 1%), the decline accelerated in 2023 (a drop of 4.7%).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
