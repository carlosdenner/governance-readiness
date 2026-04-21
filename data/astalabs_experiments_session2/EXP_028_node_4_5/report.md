# Experiment 28: node_4_5

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_5` |
| **ID in Run** | 28 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T02:21:01.760903+00:00 |
| **Runtime** | 335.8s |
| **Parent** | `node_3_6` |
| **Children** | `node_5_7` |
| **Creation Index** | 29 |

---

## Hypothesis

> The 'Operational Governance' Gate: AI systems in the 'Operation' stage are
significantly more likely to have undergone an 'Independent Evaluation' than
systems in 'Development' or 'Pilot' stages, suggesting that rigorous governance
is treated as a late-stage gatekeeper rather than a continuous lifecycle
process.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.9121 (Definitely True) |
| **Surprise** | +0.2042 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
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
| Definitely True | 60.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Determine if Independent Evaluation is correlated with the development lifecycle stage.

### Steps
- 1. Filter `eo13960_scored` data.
- 2. Clean `16_dev_stage` to group into 'Operation' (Use, Operation, Maintenance) vs 'Pre-Operation' (Development, Pilot, Planning).
- 3. Binarize `55_independent_eval` (Yes vs No/Other).
- 4. Perform a Chi-square test to compare the rate of Independent Evaluation between the two groups.

### Deliverables
- Contingency table of Stage vs Evaluation; Chi-square test statistics; Visualization of evaluation rates by stage.

---

## Analysis

The experiment successfully tested the 'Operational Governance' Gate hypothesis
using the EO 13960 dataset.

**Hypothesis Status:** Supported.

**Findings:**
1. **Data Processing:** 1,620 systems were successfully categorized into
'Operation' (n=920) or 'Pre-Operation' (n=700) stages.
2. **Evaluation Rates:**
   - **Operation Stage:** 4.24% (39/920) of systems have undergone independent
evaluation.
   - **Pre-Operation Stage:** 1.86% (13/700) of systems have undergone
independent evaluation.
3. **Statistical Significance:** The Chi-square test yielded a p-value of
**0.0107** (Statistic=6.51), which is statistically significant (p < 0.05).

**Interpretation:**
The analysis confirms a statistically significant correlation between lifecycle
stage and governance rigor. AI systems in the 'Operation' stage are more than
**twice as likely** to have undergone independent evaluation compared to those
in development/pilot stages. This supports the hypothesis that rigorous
governance (specifically independent evaluation) functions more as a late-stage
or post-deployment check rather than a continuous process integrated into early
development. However, the critical insight is the remarkably low absolute rate
of evaluation across *both* groups (<5%), suggesting that while the "gate"
exists, it is rarely utilized.

---

## Review

The experiment successfully tested the 'Operational Governance' Gate hypothesis
using the EO 13960 dataset. The implementation robustly handled data quality
issues regarding lifecycle stage categorization and the varied text descriptions
for independent evaluation status.

**Hypothesis Status:** Supported.

**Findings:**
1. **Data Processing:** A total of 1,620 AI systems were analyzed, categorized
into 'Operation' (n=920) and 'Pre-Operation' (n=700) stages.
2. **Evaluation Rates:**
   - **Operation Stage:** 4.24% (39/920) of systems have undergone independent
evaluation.
   - **Pre-Operation Stage:** 1.86% (13/700) of systems have undergone
independent evaluation.
3. **Statistical Significance:** The Chi-square test yielded a p-value of
**0.0107** (Statistic=6.51), confirming the difference is statistically
significant (p < 0.05).

**Interpretation:**
The results support the hypothesis that independent evaluation is treated more
as a late-stage gatekeeper than a continuous lifecycle process, as operational
systems are more than twice as likely to have undergone evaluation compared to
those in development. However, the most striking finding is the critically low
absolute rate of independent evaluation across the entire inventory (<5%),
indicating that while the 'gate' is statistically visible, it is functionally
open or ignored for the vast majority of federal AI systems.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
subset = df[df['source_table'] == 'eo13960_scored'].copy()

print("--- Data Loading ---")
print(f"EO 13960 Records: {len(subset)}")

# Inspect Independent Evaluation Column
print("\n--- Unique Values in '55_independent_eval' ---")
unique_evals = subset['55_independent_eval'].unique()
print(unique_evals)

# Define categorization function for Lifecycle Stage
def categorize_stage(val):
    s = str(val).lower()
    if 'retired' in s:
        return 'Retired'
    # Operation keywords
    op_keywords = ['operation', 'maintenance', 'production', 'mission', 'implementation', 'deployed', 'use']
    if any(k in s for k in op_keywords):
        return 'Operation'
    # Pre-Operation keywords
    pre_op_keywords = ['acquisition', 'development', 'initiated', 'planned', 'pilot', 'research', 'design', 'testing']
    if any(k in s for k in pre_op_keywords):
        return 'Pre-Operation'
    return 'Other'

subset['stage_group'] = subset['16_dev_stage'].apply(categorize_stage)

# Filter for relevant groups
analysis_df = subset[subset['stage_group'].isin(['Operation', 'Pre-Operation'])].copy()

# Binarize Independent Evaluation
# Robust check for affirmative values
def is_affirmative(val):
    s = str(val).lower().strip()
    # Check for Yes, True, or 1
    if s in ['yes', 'true', '1', '1.0']:
        return 1
    # Check for "completed" or similar if inspection reveals it
    if 'yes' in s:
        return 1
    return 0

analysis_df['has_eval'] = analysis_df['55_independent_eval'].apply(is_affirmative)

# Calculate Rates
group_stats = analysis_df.groupby('stage_group')['has_eval'].agg(['count', 'sum', 'mean'])
group_stats['mean_pct'] = group_stats['mean'] * 100
print("\n--- Independent Evaluation Statistics ---")
print(group_stats)

# Extract counts for Chi-Square
# Table format: [[Op_Yes, Op_No], [Pre_Yes, Pre_No]]
op_yes = group_stats.loc['Operation', 'sum']
op_total = group_stats.loc['Operation', 'count']
op_no = op_total - op_yes

pre_yes = group_stats.loc['Pre-Operation', 'sum']
pre_total = group_stats.loc['Pre-Operation', 'count']
pre_no = pre_total - pre_yes

contingency_table = [[op_yes, op_no], [pre_yes, pre_no]]

print("\n--- Contingency Table (Yes, No) ---")
print(f"Operation:     {op_yes}, {op_no}")
print(f"Pre-Operation: {pre_yes}, {pre_no}")

# Check for validity of Chi-Square
total_yes = op_yes + pre_yes
if total_yes == 0:
    print("\nRESULT: No independent evaluations found in the dataset (0 positive cases).")
    print("Cannot perform Chi-Square test.")
else:
    # Perform Chi-Square Test
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-Value: {p:.4e}")

    if p < 0.05:
        print("Result: Statistically significant difference between stages.")
    else:
        print("Result: No statistically significant difference.")

    # Visualization
    plt.figure(figsize=(8, 6))
    bars = plt.bar(group_stats.index, group_stats['mean_pct'], color=['#2ca02c', '#1f77b4'])
    plt.title('Independent Evaluation Rate by Lifecycle Stage')
    plt.ylabel('Percentage with Independent Eval (%)')
    plt.xlabel('Lifecycle Stage')
    # Dynamic ylim
    ymax = group_stats['mean_pct'].max()
    if ymax == 0:
        ymax = 10
    plt.ylim(0, ymax * 1.2)

    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (ymax*0.02),
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Data Loading ---
EO 13960 Records: 1757

--- Unique Values in '55_independent_eval' ---
<StringArray>
[                                                                                                    nan,
                                                                                     'Yes – by the CAIO',
                                                                                'Planned or in-progress',
 'Yes – by another appropriate agency office that was not directly involved in the system’s development',
               'Yes – by an agency AI oversight board not directly involved in the system’s development',
                                       'Does not apply, use case is neither safety or rights impacting.',
                         'Agency CAIO has waived this minimum practice and reported such waiver to OMB.',
                                                                 'AI is not safety or rights-impacting.',
                                                                                                  'TRUE']
Length: 9, dtype: str

--- Independent Evaluation Statistics ---
               count  sum      mean  mean_pct
stage_group                                  
Operation        920   39  0.042391  4.239130
Pre-Operation    700   13  0.018571  1.857143

--- Contingency Table (Yes, No) ---
Operation:     39, 881
Pre-Operation: 13, 687

Chi-Square Statistic: 6.5134
P-Value: 1.0706e-02
Result: Statistically significant difference between stages.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the plot:

**1. Plot Type**
*   **Type:** Vertical Bar Chart.
*   **Purpose:** This chart compares the rate (percentage) of independent evaluations across two distinct categorical stages of a lifecycle ("Operation" and "Pre-Operation").

**2. Axes**
*   **X-axis:**
    *   **Title:** "Lifecycle Stage"
    *   **Categories:** Two categories are displayed: "Operation" and "Pre-Operation".
*   **Y-axis:**
    *   **Title:** "Percentage with Independent Eval (%)"
    *   **Units:** Percentage points.
    *   **Range:** The scale ranges from 0 to 5, with increments of 1 unit.

**3. Data Trends**
*   **Tallest Bar:** The "Operation" stage (green bar) is the tallest, indicating a higher frequency of independent evaluations.
*   **Shortest Bar:** The "Pre-Operation" stage (blue bar) is the shortest, indicating a lower frequency of independent evaluations.
*   **Pattern:** There is a significant drop-off in the evaluation rate when moving from the Operation stage to the Pre-Operation stage.

**4. Annotations and Legends**
*   **Data Labels:** Both bars are annotated with specific percentage values in bold text positioned directly above the bars:
    *   Operation: **4.2%**
    *   Pre-Operation: **1.9%**
*   **Color Coding:** Although there is no separate legend box, the bars are distinct by color: Green for Operation and Blue for Pre-Operation.

**5. Statistical Insights**
*   **Significant Disparity:** The rate of independent evaluation is significantly higher during the "Operation" stage compared to the "Pre-Operation" stage.
*   **Relative Comparison:** The evaluation rate for "Operation" (4.2%) is more than double the rate for "Pre-Operation" (1.9%). Specifically, it is approximately **2.2 times higher**.
*   **Absolute Difference:** There is a **2.3 percentage point difference** between the two stages (4.2% - 1.9%). This suggests that once a project or entity moves into the operational phase, the likelihood or requirement for independent evaluation increases substantially.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
