# Experiment 191: node_6_29

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_29` |
| **ID in Run** | 191 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:03:57.942113+00:00 |
| **Runtime** | 236.4s |
| **Parent** | `node_5_7` |
| **Children** | `node_7_5` |
| **Creation Index** | 192 |

---

## Hypothesis

> The 'Pilot' Sustainability Gap: AI systems in the 'Pilot' stage are
significantly less likely to have confirmed 'Timely Resources' for maintenance
than systems in 'Operation', predicting a failure to scale.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7581 (Likely True) |
| **Posterior** | 0.2582 (Likely False) |
| **Surprise** | -0.5998 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 2.0 |
| Maybe True | 28.0 |
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

**Objective:** Assess resource planning maturity across lifecycle stages.

### Steps
- 1. Filter 'eo13960_scored'.
- 2. Group '16_dev_stage' into 'Pilot/Planned' vs 'Operation/Use'.
- 3. Binarize '47_timely_resources' (Yes vs No/Unknown).
- 4. Compare the proportion of resourced systems between stages using a Chi-square test.

### Deliverables
- 1. Resource confirmation rates by Stage.
- 2. Statistical test results.
- 3. Bar chart.

---

## Analysis

The experiment successfully tested the 'Pilot' Sustainability Gap hypothesis
using the EO 13960 dataset.

**Hypothesis Status:** Refuted.

**Findings:**
1. **Data Processing:** A total of 1,131 systems were analyzed, grouped into
'Operation/Use' (n=760) and 'Pilot/Planned' (n=371).
2. **Resource Confirmation Rates:**
   - **Pilot/Planned Stage:** 57.95% (215/371) have confirmed timely resources.
   - **Operation/Use Stage:** 43.29% (329/760) have confirmed timely resources.
3. **Statistical Significance:** The Chi-square test yielded a p-value of
**4.88e-06** (Statistic=20.89), indicating a highly significant difference.

**Interpretation:**
The results directly contradict the hypothesis. Instead of pilots being
resource-starved, they are significantly *more* likely (approx. 1.34x) to have
confirmed 'Timely Resources' than systems in the operational phase. This
suggests a "Sustainability Cliff": agencies are successful at securing resources
for the initial development and pilot phases (likely via innovation funds), but
struggle to confirm long-term maintenance and operational resources for deployed
systems. The hypothesis that pilots fail due to lack of resources is incorrect;
rather, the resource gap appears to open up *after* deployment.

---

## Review

The experiment was faithfully implemented. The code correctly filtered the EO
13960 dataset, grouped the lifecycle stages, and binarized the resource
confirmation field. The Chi-square test and visualization were successfully
generated after correcting the indexing error. The results provide strong
evidence to evaluate the hypothesis.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# Filter for EO 13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Clean and Group Lifecycle Stage (16_dev_stage)
def map_stage(stage):
    if pd.isna(stage):
        return None
    stage_lower = str(stage).lower()
    if any(x in stage_lower for x in ['use', 'operation', 'maintenance', 'retired']):
        return 'Operation/Use'
    elif any(x in stage_lower for x in ['plan', 'develop', 'pilot', 'acquisition']):
        return 'Pilot/Planned'
    else:
        return 'Other/Unknown'

eo_data['Stage_Group'] = eo_data['16_dev_stage'].apply(map_stage)

# Filter out Unknown/None stages for the analysis
analysis_df = eo_data[eo_data['Stage_Group'].isin(['Operation/Use', 'Pilot/Planned'])].copy()

# Clean and Binarize 'Timely Resources' (47_timely_resources)
def map_resources(val):
    if pd.isna(val):
        return 0
    val_lower = str(val).lower()
    if 'yes' in val_lower:
        return 1
    return 0

analysis_df['Has_Resources'] = analysis_df['47_timely_resources'].apply(map_resources)

# Generate Contingency Table
contingency_table = pd.crosstab(analysis_df['Stage_Group'], analysis_df['Has_Resources'])
contingency_table.columns = ['No/Unknown', 'Yes (Confirmed)']
print("\nContingency Table (Stage vs Timely Resources):")
print(contingency_table)

# Calculate Rates
rates = analysis_df.groupby('Stage_Group')['Has_Resources'].mean()
counts = analysis_df.groupby('Stage_Group')['Has_Resources'].count()
success_counts = analysis_df.groupby('Stage_Group')['Has_Resources'].sum()

print("\nResource Confirmation Rates:")
print(rates)

# Perform Chi-square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square Test Results:")
print(f"Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Visualization
plt.figure(figsize=(8, 6))
bar_colors = ['#ff9999', '#66b3ff']
ax = rates.plot(kind='bar', color=bar_colors, alpha=0.8, edgecolor='black')

plt.title("Proportion of AI Systems with Confirmed 'Timely Resources' by Stage")
plt.ylabel('Proportion (Yes / Total)')
plt.xlabel('Lifecycle Stage Group')
plt.ylim(0, 1.1)

# Add value labels using .iloc for positional indexing
for i, v in enumerate(rates):
    # Fix: use .iloc to access values by integer position since the Series index is string-based
    n_success = success_counts.iloc[i]
    n_total = counts.iloc[i]
    ax.text(i, v + 0.02, f"{v:.1%} (n={n_success}/{n_total})", 
            ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: 
Contingency Table (Stage vs Timely Resources):
               No/Unknown  Yes (Confirmed)
Stage_Group                               
Operation/Use         431              329
Pilot/Planned         156              215

Resource Confirmation Rates:
Stage_Group
Operation/Use    0.432895
Pilot/Planned    0.579515
Name: Has_Resources, dtype: float64

Chi-square Test Results:
Statistic: 20.8851
P-value: 4.8767e-06


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot (Bar Chart).
*   **Purpose:** The plot compares the proportion of AI systems that have confirmed "Timely Resources" across two distinct groups based on their lifecycle stage. It allows for a direct visual comparison of resource availability between systems in operation and those in planning/pilot phases.

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Title:** "Lifecycle Stage Group".
    *   **Labels:** The axis features categorical labels representing two groups: "Operation/Use" and "Pilot/Planned". The text is rotated vertically for readability.
*   **Y-Axis (Vertical):**
    *   **Title:** "Proportion (Yes / Total)".
    *   **Units:** The values represent a ratio or probability, ranging from 0 to 1 (equivalent to 0% to 100%).
    *   **Range:** The axis ticks range from 0.0 to 1.0, with an extended upper bound around 1.1 to accommodate labels. Major grid lines appear every 0.2 units.

### 3. Data Trends
*   **Comparison of Groups:** There is a notable disparity between the two lifecycle stages.
    *   **Tallest Bar:** The **"Pilot/Planned"** group (represented in light blue) shows a higher proportion, reaching approximately **0.58**.
    *   **Shortest Bar:** The **"Operation/Use"** group (represented in light red/pink) shows a lower proportion, reaching approximately **0.43**.
*   **Pattern:** AI systems that are still in the preliminary stages (Pilot or Planned) are reported to have "Timely Resources" at a significantly higher rate than systems that are already in the "Operation/Use" stage.

### 4. Annotations and Legends
*   **Plot Title:** "Proportion of AI Systems with Confirmed 'Timely Resources' by Stage" – clearly defines the metric and grouping variable.
*   **Bar Annotations:** Each bar is topped with precise statistical labels containing two key pieces of information:
    *   **Percentage:** The proportion expressed as a percentage (e.g., **43.3%** and **58.0%**).
    *   **Sample Size (n):** The raw counts formatted as `(n = successes / total sample)`.
        *   For "Operation/Use": **(n=329/760)**.
        *   For "Pilot/Planned": **(n=215/371)**.
*   **Grid Lines:** Horizontal dashed grey lines serve as a visual aid to estimate bar height against the Y-axis.

### 5. Statistical Insights
*   **Resource Gap:** There is a **14.7 percentage point gap** between the two groups. Systems in the "Pilot/Planned" stage are 1.34 times more likely to report having timely resources confirmed compared to those in "Operation/Use".
*   **Sample Size Context:** While the "Operation/Use" group has a lower success rate (43.3%), it represents a much larger sample size (760 total systems) compared to the "Pilot/Planned" group (371 total systems). This suggests the data for operational systems is more robust, though the trend of lower resource confirmation is significant.
*   **Operational Reality vs. Planning Optimism:** The data suggests a potential "optimism bias" or resource bottleneck. It appears easier to confirm or allocate timely resources during the planning and pilot phases. However, once systems scale to full operation ("Operation/Use"), maintaining or confirming those timely resources becomes more difficult, dropping from nearly 6 in 10 systems to roughly 4 in 10.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
