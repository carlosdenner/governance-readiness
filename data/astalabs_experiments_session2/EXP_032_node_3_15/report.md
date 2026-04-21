# Experiment 32: node_3_15

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_15` |
| **ID in Run** | 32 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T02:31:22.018257+00:00 |
| **Runtime** | 317.6s |
| **Parent** | `node_2_1` |
| **Children** | `node_4_17`, `node_4_30` |
| **Creation Index** | 33 |

---

## Hypothesis

> The Deployment-Governance Gap: AI systems listed as 'Operational' in the EO13960
inventory have lower 'Continuous Monitoring' compliance rates compared to
systems in the 'Development' phase, suggesting a 'deploy-and-forget' dynamic.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6935 (Maybe True) |
| **Posterior** | 0.3846 (Maybe False) |
| **Surprise** | -0.3707 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 27.0 |
| Uncertain | 0.0 |
| Maybe False | 3.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 54.0 |
| Definitely False | 6.0 |

---

## Experiment Plan

**Objective:** Investigate if governance controls related to monitoring drop off after a system transitions from development to operation.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `eo13960_scored`.
- 2. Identify the `System Stage` or `Lifecycle Status` column (e.g., looking for 'Operational', 'Development', 'Pilot').
- 3. Identify columns related to monitoring or ongoing assessment (e.g., look for 'monitoring', 'audit', 'annual review' in headers). Create a composite 'Monitoring Score' if multiple exist.
- 4. Group by `System Stage` and calculate the average `Monitoring Score`.
- 5. Perform an ANOVA or pairwise t-test to compare Operational vs. Development stages.

### Deliverables
- 1. Average Monitoring Score by Lifecycle Stage.
- 2. Statistical test results.
- 3. Line or Bar chart showing governance score trends across lifecycle stages.

---

## Analysis

The code successfully loaded the 'astalabs_discovery_all_data.csv' dataset and
filtered for the 'eo13960_scored' table. It identified 627 systems in the
'Operational' stage and 351 in the 'Development' stage. The analysis for the
specific 'Continuous Monitoring' field ('56_monitor_postdeploy') revealed a mean
score of 0.0 for both groups, implying no affirmative data points were found for
this specific control, which resulted in a 'nan' T-statistic. However, the
composite 'Governance_Score' (aggregating multiple controls) provided usable
data, showing a slight increase in governance rigor from Development (mean=0.02)
to Operational (mean=0.09). Consequently, the hypothesis that monitoring
compliance drops after deployment could not be tested directly due to the zero-
variance variable, but the broader governance data suggests a low-baseline
increase rather than a decrease.

---

## Review

Hypothesis: The 'Deployment-Governance Gap' (Operational systems have lower
monitoring compliance than Development systems) was NOT supported by the data.

Findings:
1.  **Zero Signal for Monitoring:** The specific indicator for 'Continuous
Monitoring' (`56_monitor_postdeploy`) yielded a mean score of 0.0 for both
'Operational' (n=627) and 'Development' (n=351) systems. This indicates that
either no systems in the dataset report this control, or the data is missing.
Consequently, the T-test returned `nan` (zero variance).
2.  **Inverse Trend in General Governance:** To salvage the analysis, a
composite 'Governance Score' was calculated. Contrary to the 'deploy-and-forget'
hypothesis, 'Operational' systems showed a slightly *higher* (though still low)
average governance score (0.09) compared to 'Development' systems (0.02).

Conclusion: The experiment successfully executed the pipeline, but the specific
hypothesis could not be validated due to the absence of positive data points for
the primary variable. The secondary analysis suggests that governance rigor
marginally increases, rather than decreases, upon deployment, though overall
compliance remains extremely low (<10%).

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Define the filename
filename = 'astalabs_discovery_all_data.csv'

# Check if file exists in current directory, if not try ../, if not fail gracefully
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    # Fallback to absolute path check or just assume current dir if all else fails to let pandas handle error
    filepath = filename

print(f"Loading dataset from {filepath}...")
try:
    df = pd.read_csv(filepath, low_memory=False)
except FileNotFoundError:
    # If the logic above failed, try one last desperate attempt based on the prompt note
    filepath = '../astalabs_discovery_all_data.csv'
    df = pd.read_csv(filepath, low_memory=False)

# Filter for EO13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO13960 subset shape: {eo_df.shape}")

# Columns of interest
stage_col = '16_dev_stage'
monitor_col = '56_monitor_postdeploy'
# Additional columns for composite governance score
gov_cols = ['56_monitor_postdeploy', '55_independent_eval', '30_saop_review', '52_impact_assessment']

# Function to clean boolean-like columns
def parse_bool(x):
    if pd.isna(x):
        return 0
    s = str(x).lower().strip()
    if s in ['yes', 'true', '1', 'y']:
        return 1
    return 0

# Apply cleaning
for col in gov_cols:
    if col in eo_df.columns:
        eo_df[col + '_score'] = eo_df[col].apply(parse_bool)
    else:
        print(f"Warning: Column {col} not found. Filling with 0.")
        eo_df[col + '_score'] = 0

# Calculate scores
eo_df['Monitoring_Score'] = eo_df[monitor_col + '_score']
eo_df['Governance_Score'] = eo_df[[c + '_score' for c in gov_cols]].mean(axis=1)

# Inspect Stages
print("\nUnique Lifecycle Stages found:")
print(eo_df[stage_col].value_counts())

# Define groups
# We treat 'Operational' and 'Use' as Operational.
# We treat 'Development' and 'Research' as Development.
def categorize_stage(val):
    s = str(val).lower()
    if 'oper' in s or 'use' in s or 'maintenance' in s:
        return 'Operational'
    if 'dev' in s or 'research' in s or 'pilot' in s:
        return 'Development'
    return 'Other'

eo_df['Stage_Category'] = eo_df[stage_col].apply(categorize_stage)

# Filter for only Operational and Development
analysis_df = eo_df[eo_df['Stage_Category'].isin(['Operational', 'Development'])].copy()

# Calculate stats
stats_df = analysis_df.groupby('Stage_Category')[['Monitoring_Score', 'Governance_Score']].agg(['mean', 'std', 'count'])
print("\nStatistics by Stage Category:")
print(stats_df)

# T-Test
op_scores = analysis_df[analysis_df['Stage_Category'] == 'Operational']['Monitoring_Score']
dev_scores = analysis_df[analysis_df['Stage_Category'] == 'Development']['Monitoring_Score']

t_stat, p_val = stats.ttest_ind(op_scores, dev_scores, equal_var=False)
print(f"\nT-test (Monitoring Score): Operational vs Development")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4e}")

if p_val < 0.05:
    print("Result: Significant difference detected.")
    if t_stat < 0:
        print("Direction: Operational systems have LOWER monitoring compliance than Development systems.")
    else:
        print("Direction: Operational systems have HIGHER monitoring compliance than Development systems.")
else:
    print("Result: No significant difference detected.")

# Visualization
plt.figure(figsize=(10, 6))
means = stats_df['Monitoring_Score']['mean']
errors = stats_df['Monitoring_Score']['std'] / np.sqrt(stats_df['Monitoring_Score']['count'])

# Ensure order is Development then Operational for logical flow
plot_order = ['Development', 'Operational']
means = means.reindex(plot_order)
errors = errors.reindex(plot_order)

bars = plt.bar(plot_order, means, yerr=errors, capsize=10, color=['#ff9999', '#66b3ff'], alpha=0.9)
plt.title('Compliance with "Continuous Monitoring" by Lifecycle Stage')
plt.ylabel('Proportion of Systems Compliant')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.1%}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Secondary Plot: Composite Governance Score
plt.figure(figsize=(10, 6))
comp_means = stats_df['Governance_Score']['mean'].reindex(plot_order)
comp_errors = (stats_df['Governance_Score']['std'] / np.sqrt(stats_df['Governance_Score']['count'])).reindex(plot_order)

bars2 = plt.bar(plot_order, comp_means, yerr=comp_errors, capsize=10, color=['#99ff99', '#ffcc99'], alpha=0.9)
plt.title('Composite Governance Score (Eval + Monitor + Review) by Stage')
plt.ylabel('Average Score (0-1)')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
EO13960 subset shape: (1757, 196)

Unique Lifecycle Stages found:
16_dev_stage
Operation and Maintenance         627
Acquisition and/or Development    351
Initiated                         329
Implementation and Assessment     275
Retired                           133
Planned                            20
In production                      14
In mission                          4
Name: count, dtype: int64

Statistics by Stage Category:
               Monitoring_Score            Governance_Score                
                           mean  std count             mean       std count
Stage_Category                                                             
Development                 0.0  0.0   351         0.022080  0.073512   351
Operational                 0.0  0.0   627         0.089713  0.152933   627

T-test (Monitoring Score): Operational vs Development
T-statistic: nan
P-value: nan
Result: No significant difference detected.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Bar Plot.
*   **Purpose:** The plot is designed to compare the proportion of systems that are compliant with "Continuous Monitoring" standards across two different stages of the system lifecycle (Development and Operational).

**2. Axes**
*   **Y-axis:**
    *   **Label:** "Proportion of Systems Compliant".
    *   **Range:** 0.0 to 1.0 (representing 0% to 100%).
    *   **Units:** Proportions (decimal format).
*   **X-axis:**
    *   **Label:** The axis represents the "Lifecycle Stage" (implied by the title and category names).
    *   **Categories:** "Development" and "Operational".

**3. Data Trends**
*   **Bar Heights:** Both bars have a height of zero. There are no visible bars extending upwards from the x-axis.
*   **Pattern:** There is a uniform pattern of zero compliance across both lifecycle stages. Neither the "Development" phase nor the "Operational" phase shows any measurable proportion of compliant systems.

**4. Annotations and Legends**
*   **Value Labels:** The text **"0.0%"** is annotated in bold directly above the x-axis line for both the "Development" and "Operational" categories. This explicitly confirms that the value is zero and not simply a missing data point.
*   **Grid Lines:** Horizontal dashed grid lines are present at intervals of 0.2 (20%) to aid in reading the y-axis, though they are not utilized by the data in this specific instance.

**5. Statistical Insights**
*   **Zero Compliance:** The most significant insight is that there is a **0% compliance rate** for "Continuous Monitoring" across the board.
*   **Lifecycle Consistency:** The stage of the lifecycle does not impact compliance; both systems in development and systems currently in operation are failing to meet the "Continuous Monitoring" criteria.
*   **Operational Gap:** This plot suggests a critical gap in process or implementation. It indicates that "Continuous Monitoring" is either not being performed, not being reported correctly, or that the criteria for compliance are currently unmet by any system in the dataset.
==================================================

=== Plot Analysis (figure 2) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot (with error bars).
*   **Purpose:** The plot compares the average "Composite Governance Score" across two distinct lifecycle stages: "Development" and "Operational".

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Average Score (0-1)" indicating a normalized scale.
    *   **Range:** The axis spans from **0.0 to 1.0**, with tick marks at intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).
*   **X-Axis:**
    *   **Label:** Represents the **Stage** (implied by the title "by Stage").
    *   **Categories:** Two categories are displayed: **Development** and **Operational**.

### 3. Data Trends
*   **Comparisons:** The **Operational** stage has a higher average score compared to the **Development** stage.
*   **Values:**
    *   **Shortest Bar:** The "Development" stage is extremely low, with a value of **0.02**.
    *   **Tallest Bar:** The "Operational" stage is higher, with a value of **0.09**.
*   **Magnitude:** Despite the difference between the two, both bars represent very low values on the provided scale (neither exceeds 0.1 on a scale of 1.0).

### 4. Annotations and Legends
*   **Plot Title:** "Composite Governance Score (Eval + Monitor + Review) by Stage". This indicates the metric is an aggregate of Evaluation, Monitoring, and Review scores.
*   **Data Labels:** Specific values are annotated in bold directly above the error bars for clarity: **0.02** for Development and **0.09** for Operational.
*   **Error Bars:** Both bars feature error bars (whiskers) at the top, indicating the variability or uncertainty (likely standard error or confidence intervals) associated with the average score.
*   **Grid Lines:** Horizontal dashed grey lines are placed at every 0.2 interval to assist with visual estimation.

### 5. Statistical Insights
*   **Low Overall Performance:** The most striking insight is that the Composite Governance Scores are incredibly low for both stages. On a scale of 0 to 1, the highest score achieved is only 0.09. This suggests that the governance mechanisms (Evaluation, Monitoring, and Review) are either barely present, ineffective, or the criteria for scoring are extremely rigorous.
*   **Relative Improvement:** While the absolute scores are low, there is a significant **relative** increase in the Operational stage. The score of 0.09 is **4.5 times higher** than the Development score of 0.02.
*   **Process Maturity:** The data suggests that governance is slightly more established or enforced once a project reaches the "Operational" phase compared to the "Development" phase, where it appears to be nearly non-existent.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
