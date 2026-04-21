# Experiment 43: node_5_3

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_3` |
| **ID in Run** | 43 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:53:18.289795+00:00 |
| **Runtime** | 354.3s |
| **Parent** | `node_4_7` |
| **Children** | `node_6_10` |
| **Creation Index** | 44 |

---

## Hypothesis

> Incidents causing 'Security' harms are significantly more likely to involve
simultaneous 'Both' (Trust & Integration) gaps compared to 'Reliability'
incidents, which are more likely to be isolated to a single domain.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.1963 (Likely False) |
| **Surprise** | -0.6333 |
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
| Maybe False | 3.0 |
| Definitely False | 87.0 |

---

## Experiment Plan

**Objective:** To explore if security incidents are more systemic (involving both governance and engineering failures) compared to reliability incidents.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Create a binary variable 'is_both_gap' where 'trust_integration_split' contains 'Both'.
- 3. Filter the dataset to include only 'security' and 'reliability' harm types.
- 4. Create a contingency table of Harm Type vs. 'is_both_gap'.
- 5. Perform Fisher's Exact Test to analyze the association.
- 6. Visualize with a heatmap or stacked bar chart.

### Deliverables
- 1. Contingency table.
- 2. Fisher's Exact Test results.
- 3. Visualization.

---

## Analysis

The experiment successfully analyzed the relationship between 'harm_type' and
the scope of competency gaps ('gap_scope') using the 'step3_incident_coding.csv'
dataset.

**Descriptive Statistics:**
- **Security Incidents (n=36):** 31 incidents (86.1%) involved simultaneous gaps
in 'Both' Trust and Integration domains, while 5 (13.9%) involved a 'Single'
domain gap.
- **Reliability Incidents (n=4):** 4 incidents (100%) involved gaps in 'Both'
domains; none were isolated to a single domain.

**Statistical Testing:**
- **Method:** Fisher's Exact Test (due to small sample size in the Reliability
category).
- **Result:** P-value = 1.0000; Odds Ratio = inf (due to the zero count in the
Reliability/Single cell).
- **Conclusion:** There is no statistically significant difference between the
two groups.

**Hypothesis Evaluation:**
The hypothesis that "Security incidents are significantly more likely to involve
simultaneous 'Both' gaps compared to Reliability incidents" is **REJECTED**. In
fact, the data shows the opposite trend numerically (though not statistically
significant): 100% of Reliability incidents involved 'Both' gaps compared to 86%
of Security incidents. This suggests that reliability failures in this dataset
are just as, if not more, systemic than security failures.

---

## Review

Hypothesis: Incidents causing 'Security' harms are significantly more likely to
involve simultaneous 'Both' (Trust & Integration) gaps compared to 'Reliability'
incidents.

Experiment Results: The experiment was faithfully implemented using the
'step3_incident_coding.csv' dataset. After initial debugging to ensure small
classes were correctly handled, the analysis compared 36 'Security' incidents
against 4 'Reliability' incidents.

Findings:
1. Hypothesis Rejection: The hypothesis is rejected. Contrary to the expectation
that Reliability incidents would be more isolated (single domain), the data
showed that 100% (4/4) of Reliability incidents involved gaps in 'Both' domains,
compared to 86.1% (31/36) of Security incidents.
2. Statistical Evidence: Fisher's Exact Test yielded a p-value of 1.000,
indicating no statistically significant difference between the groups. The Odds
Ratio was infinite due to the zero count of single-domain Reliability incidents.
3. Conclusion: The data suggests that both Security and Reliability failures in
this dataset are predominantly systemic, involving simultaneous deficiencies in
governance (Trust) and engineering (Integration). Reliability incidents appear
even more likely to be systemic than Security incidents, though the small sample
size (n=4) for Reliability limits the generalization of this specific
comparison.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load the dataset
file_path = 'step3_incident_coding.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    try:
        # Fallback for different directory structure
        df = pd.read_csv('../step3_incident_coding.csv')
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# 2. Preprocess Data
# Handle missing values and normalize strings
df['harm_type'] = df['harm_type'].fillna('').astype(str).str.strip().str.lower()
df['trust_integration_split'] = df['trust_integration_split'].fillna('').astype(str).str.strip().str.lower()

# Filter for specific harm types
target_harms = ['security', 'reliability']
df_subset = df[df['harm_type'].isin(target_harms)].copy()

print(f"Total incidents in analysis: {len(df_subset)}")
print(df_subset['harm_type'].value_counts())

# 3. Create Categorical Variables
# Define 'Both' vs 'Single' (Single includes 'trust-dominant', 'integration-dominant', or missing/other)
df_subset['gap_scope'] = df_subset['trust_integration_split'].apply(lambda x: 'Both Domains' if x == 'both' else 'Single Domain')

# 4. Generate Contingency Table
# explicit crosstab with reindexing to ensure all categories appear
contingency_table = pd.crosstab(df_subset['harm_type'], df_subset['gap_scope'])

# Ensure columns verify 'Single Domain' and 'Both Domains' exist
expected_cols = ['Single Domain', 'Both Domains']
for col in expected_cols:
    if col not in contingency_table.columns:
        contingency_table[col] = 0

# Ensure rows verify 'security' and 'reliability' exist
contingency_table = contingency_table.reindex(target_harms).fillna(0).astype(int)
# Reorder columns for consistency
contingency_table = contingency_table[expected_cols]

print("\nContingency Table (Harm Type vs. Gap Scope):")
print(contingency_table)

# 5. Fisher's Exact Test
# The table is 2x2: [[Security_Single, Security_Both], [Reliability_Single, Reliability_Both]]
odds_ratio, p_value = stats.fisher_exact(contingency_table)

print("\nFisher's Exact Test Results:")
print(f"Odds Ratio: {odds_ratio:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Conclusion: Reject Null Hypothesis. Significant association exists.")
else:
    print("Conclusion: Fail to Reject Null Hypothesis. No significant association detected.")

# 6. Visualization
plt.figure(figsize=(8, 6))

# Calculate proportions
prop_table = contingency_table.div(contingency_table.sum(axis=1), axis=0)

# Plot Stacked Bar Chart
ax = prop_table.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], edgecolor='black', width=0.6)

plt.title('Proportion of Gap Scope (Single vs Both) by Harm Type')
plt.xlabel('Harm Type')
plt.ylabel('Proportion of Incidents')
plt.legend(title='Gap Scope', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)

# Annotate bars with values
for c in ax.containers:
    # Filter labels to avoid cluttering if segment is 0
    labels = [f'{v.get_height():.2f}' if v.get_height() > 0 else '' for v in c]
    ax.bar_label(c, labels=labels, label_type='center')

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total incidents in analysis: 40
harm_type
security       36
reliability     4
Name: count, dtype: int64

Contingency Table (Harm Type vs. Gap Scope):
gap_scope    Single Domain  Both Domains
harm_type                               
security                 5            31
reliability              0             4

Fisher's Exact Test Results:
Odds Ratio: inf
P-value: 1.0000
Conclusion: Fail to Reject Null Hypothesis. No significant association detected.


=== Plot Analysis (figure 2) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Chart.
*   **Purpose:** This chart is designed to compare the relative proportions of two categories of "Gap Scope" (Single Domain vs. Both Domains) across different "Harm Types" (security vs. reliability). By normalizing the height of the bars to 1.0, it focuses on the composition ratio rather than absolute counts.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Harm Type"
    *   **Categories:** Two discrete categories are plotted: "security" and "reliability".
*   **Y-Axis:**
    *   **Label:** "Proportion of Incidents"
    *   **Range:** The scale runs from **0.0 to 1.0**, representing 0% to 100% of the incidents.
    *   **Ticks:** Intervals are marked at every 0.2 units (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
*   **Security Category:**
    *   The bar is split into two sections.
    *   The smaller portion (bottom, pink) represents **0.14** (or 14%).
    *   The larger portion (top, blue) represents **0.86** (or 86%).
    *   **Pattern:** The vast majority of security incidents involve "Both Domains".
*   **Reliability Category:**
    *   The bar is entirely a single color (blue).
    *   The value is **1.00** (or 100%).
    *   **Pattern:** Reliability incidents in this dataset appear exclusively in the "Both Domains" scope; there are no "Single Domain" instances recorded.
*   **Overall Comparison:** The "Both Domains" scope is the dominant category for both harm types, but "reliability" shows a homogenous distribution compared to the slight variation seen in "security."

### 4. Annotations and Legends
*   **Title:** "Proportion of Gap Scope (Single vs Both) by Harm Type" appearing at the top.
*   **Legend:** Located in the upper right corner titled "**Gap Scope**":
    *   **Pink/Light Red:** Represents "Single Domain".
    *   **Blue/Light Blue:** Represents "Both Domains".
*   **Data Labels:** Numerical values are annotated directly inside the bar segments for clarity:
    *   Inside the security bar: **0.14** (bottom) and **0.86** (top).
    *   Inside the reliability bar: **1.00** (centered).

### 5. Statistical Insights
*   **Prevalence of Multi-Domain Scope:** Across both harm types presented, the incidents predominantly span "Both Domains." This suggests that when these harms occur, they rarely affect only a single domain in isolation.
*   **Reliability vs. Security Nature:** There is a distinct difference in the nature of the harms. Reliability issues are shown to be strictly multi-domain (100% Both Domains), implying that reliability failures in this context are systemic or cross-cutting.
*   **Security Variability:** While security issues are also largely cross-cutting (86%), there is a non-negligible subset (14%) that remains contained within a "Single Domain." This indicates slightly more variability in the scope of security incidents compared to reliability incidents.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
