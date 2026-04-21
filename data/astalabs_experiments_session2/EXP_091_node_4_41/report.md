# Experiment 91: node_4_41

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_41` |
| **ID in Run** | 91 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:25:30.173428+00:00 |
| **Runtime** | 257.9s |
| **Parent** | `node_3_9` |
| **Children** | `node_5_32`, `node_5_72` |
| **Creation Index** | 92 |

---

## Hypothesis

> Rights-Impacting Compliance Paradox: AI systems explicitly classified as
'Rights-Impacting' do not show significantly higher rates of 'Independent
Evaluation' compared to non-rights-impacting systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.2555 (Likely False) |
| **Surprise** | -0.5934 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 29.0 |
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

**Objective:** Verify if high-risk 'Rights-Impacting' systems are actually receiving the enhanced scrutiny (Independent Eval) they require, correcting the previous file path error.

### Steps
- 1. Load the dataset `astalabs_discovery_all_data.csv` from the current directory.
- 2. Filter the dataframe to include only rows where `source_table` is 'eo13960_scored'.
- 3. Inspect unique values in column `17_impact_type` to identify labels corresponding to rights-impacting systems (e.g., look for 'Rights', 'Civil', 'Both'). Create a binary column `is_rights_impacting` (True if the system is rights-impacting, False otherwise).
- 4. Inspect unique values in column `55_independent_eval` to identify affirmative responses (e.g., 'Yes', 'Completed'). Create a binary column `is_evaluated` (True if evaluated, False otherwise).
- 5. Generate a contingency table (crosstab) of `is_rights_impacting` vs `is_evaluated`.
- 6. Calculate the percentage of systems evaluated within each group (Rights-Impacting vs. Others).
- 7. Perform a Chi-Square test of independence to determine if there is a statistically significant difference in evaluation rates between the two groups.
- 8. If the p-value is > 0.05, or if the evaluation rate for rights-impacting systems is not significantly higher, the hypothesis (the paradox) is supported.

### Deliverables
- 1. Contingency table of Risk Status vs. Evaluation Status.
- 2. Calculated evaluation rates (%) for both groups.
- 3. Chi-Square test statistics (Chi2 value and p-value).
- 4. A bar chart comparing the evaluation rates.

---

## Analysis

The experiment successfully analyzed 1,757 AI use cases from the EO13960 dataset
to test the 'Rights-Impacting Compliance Paradox'. By filtering for 'Rights-
Impacting' or 'Both' impact types, the code identified 209 high-risk systems and
1,548 low-risk/other systems.

The statistical analysis **refuted** the hypothesis that rights-impacting
systems do not receive significantly higher scrutiny. The Chi-Square test
yielded a p-value of 0.0000 (Statistic: 68.75), confirming a highly significant
difference between the groups. Rights-Impacting systems had an independent
evaluation rate of **11.0%** (23/209), which is nearly 8.5 times higher than the
**1.3%** (20/1548) observed for other systems.

While the hypothesis of 'no difference' was disproven, the results highlight a
more concerning absolute finding: despite receiving significantly *more*
scrutiny than low-risk tools, **89% of Rights-Impacting AI systems still lack
independent evaluation**, suggesting that while prioritization exists, overall
compliance levels remain critically low.

---

## Review

The experiment was successfully executed and the code faithfully implemented the
plan. The programmer corrected the file path issue, successfully loaded the
dataset, and performed the required data transformation and statistical
analysis. The logic used to classify 'Rights-Impacting' systems and 'Evaluated'
status was appropriate for the textual data in the EO13960 inventory.

The Chi-Square test results (p < 0.001) provide strong evidence to **reject**
the hypothesis that there is no significant difference in evaluation rates.
Contrary to the 'Paradox' hypothesis, Rights-Impacting systems are evaluated at
a significantly higher rate (11.0%) compared to other systems (1.3%). However,
the analysis notably reveals that while the difference is statistically
significant, the *absolute* level of compliance is concerningly low, with ~89%
of high-risk systems lacking independent evaluation.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

# Load dataset from current directory
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    print("Error: Dataset not found in current directory.")
    sys.exit(1)

# Filter for EO13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO13960 Subset Shape: {eo_df.shape}")

# Define columns
col_impact = '17_impact_type'
col_eval = '55_independent_eval'

# Inspect unique values to ensure correct mapping
print(f"\nUnique values in '{col_impact}':\n{eo_df[col_impact].dropna().unique()}")
print(f"\nUnique values in '{col_eval}':\n{eo_df[col_eval].dropna().unique()}")

# 1. Create 'High Risk' (Rights-Impacting) flag
# We classify 'Rights-Impacting' and 'Both' (Rights + Safety) as High Risk.
def classify_risk(val):
    s = str(val).lower()
    if 'rights' in s or 'both' in s:
        return 'Rights-Impacting'
    return 'Other'

eo_df['risk_group'] = eo_df[col_impact].apply(classify_risk)

# 2. Create 'Evaluated' flag
# We look for affirmative 'Yes'
def classify_eval(val):
    s = str(val).lower()
    if s.startswith('yes'):
        return 'Evaluated'
    return 'Not Evaluated'

eo_df['eval_status'] = eo_df[col_eval].apply(classify_eval)

# 3. Generate Contingency Table
contingency = pd.crosstab(eo_df['risk_group'], eo_df['eval_status'])
print("\nContingency Table (Counts):")
print(contingency)

# Check if we have enough data for statistical testing
if contingency.shape[0] < 2 or contingency.shape[1] < 2:
    print("\nInsufficient data variation for statistical testing.")
else:
    # 4. Perform Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:")
    print(f"Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    # Interpret
    alpha = 0.05
    if p < alpha:
        print("Result: Significant difference in evaluation rates found (Null Hypothesis Rejected).")
    else:
        print("Result: No significant difference in evaluation rates found (Supporting the 'Compliance Paradox').")

    # 5. Calculate Proportions for Plotting
    # Normalize by index (row) to get % evaluated within each risk group
    props = pd.crosstab(eo_df['risk_group'], eo_df['eval_status'], normalize='index') * 100
    print("\nEvaluation Rates (%):")
    print(props)

    # Plot
    if 'Evaluated' in props.columns:
        plt.figure(figsize=(8, 6))
        ax = props['Evaluated'].plot(kind='bar', color=['#1f77b4', '#d62728'], rot=0)
        plt.title('Percentage of AI Systems with Independent Evaluation by Risk Group')
        plt.ylabel('% Independently Evaluated')
        plt.xlabel('Risk Classification')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for p_rect in ax.patches:
            height = p_rect.get_height()
            ax.annotate(f'{height:.1f}%', 
                        (p_rect.get_x() + p_rect.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=10, weight='bold')
        
        plt.tight_layout()
        plt.show()
    else:
        print("No 'Evaluated' systems found to plot.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: EO13960 Subset Shape: (1757, 196)

Unique values in '17_impact_type':
<StringArray>
['Neither', 'Rights-Impacting\n', 'Safety-Impacting', 'Both',
 'Safety-impacting']
Length: 5, dtype: str

Unique values in '55_independent_eval':
<StringArray>
[                                                                                    'Yes – by the CAIO',
                                                                                'Planned or in-progress',
 'Yes – by another appropriate agency office that was not directly involved in the system’s development',
               'Yes – by an agency AI oversight board not directly involved in the system’s development',
                                       'Does not apply, use case is neither safety or rights impacting.',
                         'Agency CAIO has waived this minimum practice and reported such waiver to OMB.',
                                                                 'AI is not safety or rights-impacting.',
                                                                                                  'TRUE']
Length: 8, dtype: str

Contingency Table (Counts):
eval_status       Evaluated  Not Evaluated
risk_group                                
Other                    20           1528
Rights-Impacting         23            186

Chi-Square Test Results:
Statistic: 68.7495
P-value: 0.0000
Result: Significant difference in evaluation rates found (Null Hypothesis Rejected).

Evaluation Rates (%):
eval_status       Evaluated  Not Evaluated
risk_group                                
Other              1.291990      98.708010
Rights-Impacting  11.004785      88.995215


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot is designed to compare a quantitative variable (percentage of independent evaluation) across two distinct categorical groups ("Other" vs. "Rights-Impacting" AI systems).

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Risk Classification"
    *   **Categories:** The axis displays two distinct categories: "Other" and "Rights-Impacting".
*   **Y-Axis:**
    *   **Label:** "% Independently Evaluated"
    *   **Range:** The scale runs from 0 to 100, representing percentage points.
    *   **Increments:** Major gridlines are placed at intervals of 20 (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Tallest Bar:** The "Rights-Impacting" category corresponds to the tallest bar (red), reaching a value of 11.0%.
*   **Shortest Bar:** The "Other" category corresponds to the shortest bar (blue), reaching a value of only 1.3%.
*   **Pattern:** There is a notable disparity between the two categories. While the overall percentage of independent evaluation is low for both groups, the rate for "Rights-Impacting" systems is significantly higher relative to the "Other" category.

### 4. Annotations and Legends
*   **Data Labels:** Specific percentage values are annotated in bold directly on top of each bar (**1.3%** and **11.0%**), allowing for precise reading of the data without relying solely on the y-axis grid.
*   **Title:** The chart is titled "Percentage of AI Systems with Independent Evaluation by Risk Group," clearly defining the scope of the data.
*   **Color Coding:** The bars are color-coded (Blue for "Other" and Red for "Rights-Impacting"), likely to visually distinguish the risk levels, with red often semantically associated with higher risk or importance.
*   **Gridlines:** Horizontal dashed gridlines are included to aid in visual estimation of bar heights relative to the y-axis.

### 5. Statistical Insights
*   **Low Overall Adherence:** The most immediate insight is that independent evaluation is extremely rare across the board. Even for the higher-priority category, nearly 89% of systems do *not* undergo independent evaluation.
*   **Relative Risk Prioritization:** Despite the low overall numbers, there is a clear correlation between risk classification and evaluation frequency. AI systems classified as "Rights-Impacting" are approximately **8.5 times more likely** (11.0% vs 1.3%) to undergo independent evaluation than those classified as "Other."
*   **Gap in Governance:** The data suggests a significant gap in AI governance. While rights-impacting systems are receiving more scrutiny than lower-risk systems, the vast majority of these high-stakes systems still lack independent evaluation.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
