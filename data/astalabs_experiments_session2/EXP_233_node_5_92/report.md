# Experiment 233: node_5_92

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_92` |
| **ID in Run** | 233 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T11:57:30.699556+00:00 |
| **Runtime** | 217.1s |
| **Parent** | `node_4_26` |
| **Children** | None |
| **Creation Index** | 234 |

---

## Hypothesis

> Lifecycle Governance Decay: AI systems in the 'Operation' phase have
significantly lower aggregated governance scores compared to systems in the
'Development' phase, suggesting that governance is treated as a pre-deployment
gate rather than a continuous operational monitor.

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

**Objective:** Compare aggregate governance scores across lifecycle stages.

### Steps
- 1. Filter for `eo13960_scored`.
- 2. Group `16_dev_stage` into 'Development' (Planning, Development) and 'Operation' (Operation, Maintenance, Use).
- 3. Create a 'Governance Score' by summing binary compliance columns: `40_has_ato`, `52_impact_assessment`, `55_independent_eval`, `59_ai_notice`, `62_disparity_mitigation` (ensure column names match dataset).
- 4. Compare the distribution of Governance Scores between the two groups using a T-test or Mann-Whitney U test.
- 5. Visualize with box plots.

### Deliverables
- Box plot of Governance Scores by Lifecycle Stage; Statistical test results.

---

## Analysis

The experiment successfully tested the 'Lifecycle Governance Decay' hypothesis
using the EO 13960 dataset.

1. **Data Processing**: The code successfully categorized 1,287 AI systems into
'Development' (n=371) and 'Operation' (n=916) stages. A Governance Score (0-5)
was calculated by summing five binary controls: ATO, Impact Assessment,
Independent Evaluation, AI Notice, and Disparity Mitigation.

2. **Statistical Results**:
   - **Development Stage**: Mean Score = 0.38, Median = 0.0.
   - **Operation Stage**: Mean Score = 0.57, Median = 1.0.
   - **Significance**: The Mann-Whitney U test yielded a p-value of 0.0000,
indicating a statistically significant difference between the groups.

3. **Hypothesis Evaluation**: The hypothesis predicted that governance scores
would be lower in the 'Operation' phase ('Decay'). The results show the
**opposite trend**: systems in the 'Operation' phase have significantly *higher*
governance scores than those in 'Development'.

4. **Conclusion**: The hypothesis is **not supported**. Rather than decaying,
formal governance controls appear to be more prevalent in deployed systems,
suggesting that compliance gates (like ATOs) are enforced prior to or during
operational deployment rather than being front-loaded in the early development
phase.

---

## Review

The experiment was faithfully implemented and the hypothesis was successfully
tested using the EO 13960 dataset.

1. **Data Processing**: The code successfully identified and mapped the five
governance columns (ATO, Impact Assessment, Independent Evaluation, AI Notice,
Bias Mitigation) and categorized 1,287 AI systems into 'Development' (n=371) and
'Operation' (n=916) stages based on keyword matching.

2. **Statistical Results**:
   - **Development Stage**: Mean Governance Score = 0.38, Median = 0.0.
   - **Operation Stage**: Mean Governance Score = 0.57, Median = 1.0.
   - **Significance**: The Mann-Whitney U test yielded a p-value of 0.0000,
confirming the difference is statistically significant.

3. **Hypothesis Evaluation**: The hypothesis ('Lifecycle Governance Decay')
predicted that governance scores would be lower in the Operation phase. The
results demonstrate the **opposite**: systems in the Operation phase have
significantly *higher* governance scores than those in Development.

4. **Conclusion**: The hypothesis is **not supported**. The data suggests that
governance acts as a 'gate' or accumulation process where controls (like ATOs)
are more likely to be finalized and documented for deployed systems than for
those still in early development.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# [debug]
print("Starting experiment: Lifecycle Governance Decay")

# Load dataset
try:
    # Dataset files are one level above
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded EO 13960 data: {len(eo_df)} records")

# Check available columns to identify correct names
all_cols = eo_df.columns.tolist()
target_keywords = {
    'ato': 'has_ato',
    'impact': 'impact_assessment',
    'eval': 'independent_eval',
    'notice': 'ai_notice',
    'bias': 'disparity_mitigation'
}

found_cols = []
for key, keyword in target_keywords.items():
    matches = [c for c in all_cols if keyword in str(c).lower()]
    if matches:
        # Prefer the one that looks like the standard 'XX_name' format
        # Usually the shortest or the one appearing first is fine, but let's just take the first match
        found_cols.append(matches[0])
        print(f"Mapped '{key}' to column: {matches[0]}")
    else:
        print(f"Warning: Could not find column for '{key}'")

if not found_cols:
    print("Error: No governance columns found. Exiting.")
    exit(1)

# 2. Group 16_dev_stage
stage_col_matches = [c for c in all_cols if 'dev_stage' in str(c).lower()]
stage_col = stage_col_matches[0] if stage_col_matches else '16_dev_stage'
print(f"Using stage column: {stage_col}")

# Check unique values to define mapping
print("Unique development stages:", eo_df[stage_col].unique())

def classify_stage(val):
    s = str(val).lower()
    if any(x in s for x in ['plan', 'dev', 'design', 'acqui', 'research', 'pilot', 'test']):
        return 'Development'
    if any(x in s for x in ['oper', 'use', 'maint', 'deploy', 'implement', 'product']): 
        return 'Operation'
    return 'Other'

eo_df['lifecycle_group'] = eo_df[stage_col].apply(classify_stage)

# Filter out 'Other' or undefined
analysis_df = eo_df[eo_df['lifecycle_group'].isin(['Development', 'Operation'])].copy()
print(f"Records after stage filtering: {len(analysis_df)}")
print(analysis_df['lifecycle_group'].value_counts())

# 3. Create Governance Score
# Normalize binary values to 0/1
def normalize_binary(val):
    s = str(val).lower()
    if s in ['yes', 'true', '1', '1.0', 'y']:
        return 1
    return 0

for col in found_cols:
    analysis_df[col] = analysis_df[col].apply(normalize_binary)

analysis_df['governance_score'] = analysis_df[found_cols].sum(axis=1)

# 4. Statistical Analysis
dev_scores = analysis_df[analysis_df['lifecycle_group'] == 'Development']['governance_score']
ops_scores = analysis_df[analysis_df['lifecycle_group'] == 'Operation']['governance_score']

print(f"\n--- Results ---")
print(f"Development: Mean Score = {dev_scores.mean():.2f}, Median = {dev_scores.median()}, n = {len(dev_scores)}")
print(f"Operation:   Mean Score = {ops_scores.mean():.2f}, Median = {ops_scores.median()}, n = {len(ops_scores)}")

u_stat, p_val = mannwhitneyu(dev_scores, ops_scores, alternative='two-sided')
print(f"Mann-Whitney U Test: U={u_stat}, p-value={p_val:.4f}")

if p_val < 0.05:
    print("Result: Statistically significant difference found.")
else:
    print("Result: No statistically significant difference found.")

# 5. Visualization
plt.figure(figsize=(8, 6))
boxplot_data = [dev_scores.values, ops_scores.values]
plt.boxplot(boxplot_data, labels=['Development', 'Operation'])
plt.title('Governance Score by Lifecycle Stage')
plt.ylabel('Governance Score (0-5)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment: Lifecycle Governance Decay
Loaded EO 13960 data: 1757 records
Mapped 'ato' to column: 40_has_ato
Mapped 'impact' to column: 52_impact_assessment
Mapped 'eval' to column: 55_independent_eval
Mapped 'notice' to column: 59_ai_notice
Mapped 'bias' to column: 62_disparity_mitigation
Using stage column: 16_dev_stage
Unique development stages: <StringArray>
[ 'Implementation and Assessment', 'Acquisition and/or Development',
                      'Initiated',                        'Retired',
      'Operation and Maintenance',                  'In production',
                     'In mission',                        'Planned',
                              nan]
Length: 9, dtype: str
Records after stage filtering: 1287
lifecycle_group
Operation      916
Development    371
Name: count, dtype: int64

--- Results ---
Development: Mean Score = 0.38, Median = 0.0, n = 371
Operation:   Mean Score = 0.57, Median = 1.0, n = 916
Mann-Whitney U Test: U=144590.5, p-value=0.0000
Result: Statistically significant difference found.

STDERR:
<ipython-input-1-d8ef9eb33965>:103: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(boxplot_data, labels=['Development', 'Operation'])


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (also known as a Box-and-Whisker Plot).
*   **Purpose:** This plot is used to display the distribution of numerical data (Governance Score) across different categorical groups (Lifecycle Stages). It visualizes the five-number summary: minimum, first quartile (Q1), median, third quartile (Q3), and maximum, along with potential outliers.

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Labels:** Represents the lifecycle stages: **"Development"** and **"Operation"**.
    *   **Type:** Categorical variable.
*   **Y-Axis (Vertical):**
    *   **Title:** "Governance Score (0-5)".
    *   **Units/Range:** The axis is numerical. While the label indicates a theoretical scale of 0 to 5, the visible axis ticks range from **0.0 to 3.0** in increments of 0.5.

### 3. Data Trends
*   **"Development" Stage:**
    *   **Median:** The median line (orange) is at **0.0**, indicating that at least half of the projects in the development stage have a governance score of 0.
    *   **Interquartile Range (IQR):** The box extends from 0.0 to 1.0.
    *   **Range:** The whiskers extend from 0.0 up to **2.0**, which is the maximum non-outlier value.
    *   **Outliers:** There are no visible outliers for this group.
*   **"Operation" Stage:**
    *   **Median:** The median line (orange) appears at **1.0** (located at the top of the box), suggesting a higher central tendency than the Development stage.
    *   **Interquartile Range (IQR):** Similar to Development, the box spans from 0.0 to 1.0.
    *   **Range:** The whiskers extend from 0.0 to **2.0**.
    *   **Outliers:** There is a distinct outlier plotted as a circle at the value **3.0**.

### 4. Annotations and Legends
*   **Title:** The plot is titled **"Governance Score by Lifecycle Stage"**.
*   **Grid Lines:** Horizontal dashed grid lines are present at 0.5 intervals to aid in reading the y-axis values.
*   **Outlier Marker:** A small circle above the "Operation" whisker indicates a data point that is statistically distant from the rest of the data.

### 5. Statistical Insights
*   **Low Overall Scores:** despite the axis label suggesting a scale of 0-5, the actual data is concentrated at the lower end. No data point exceeds a score of 3, and the vast majority of data falls between 0 and 2. This suggests that governance scores are generally low across both lifecycle stages.
*   **Comparison of Stages:** The **Operation** stage performs slightly better than the **Development** stage. The median score for Operations is 1.0, whereas for Development it is 0.0.
*   **Distribution Skew:** Both distributions are positively skewed (skewed right). The data is clustered near zero, with whiskers extending upwards.
*   **Variability:** The **Operation** stage shows higher variability in the upper range, evidenced by the presence of an outlier at 3.0, indicating that while rare, some operation-stage projects achieve higher governance scores than any observed development-stage projects.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
