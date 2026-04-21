# Experiment 89: node_5_21

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_21` |
| **ID in Run** | 89 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:26:41.311079+00:00 |
| **Runtime** | 192.6s |
| **Parent** | `node_4_24` |
| **Children** | None |
| **Creation Index** | 90 |

---

## Hypothesis

> Requirements sourced from the 'OWASP Top 10 LLM' are significantly more likely
to map to 'Integration Readiness' competencies, whereas 'NIST AI RMF'
requirements map to 'Trust Readiness', verifying a distinct 'Security vs. Risk'
framework bias.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.9339 (Definitely True) |
| **Surprise** | +0.2228 |
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
| Definitely True | 90.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Test for framework bias in competency classification.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Filter data to include only rows where source is 'OWASP Top 10 LLM' or 'NIST AI RMF 1.0'.
- 3. Create a contingency table of 'Source' (OWASP vs NIST) against 'Bundle' (Integration vs Trust).
- 4. Perform a Chi-Square test of independence.

### Deliverables
- Contingency table, Chi-Square statistic, and p-value.

---

## Analysis

The experiment successfully tested the hypothesis that governance frameworks
exhibit distinct biases, specifically that 'OWASP Top 10 LLM' targets
'Integration Readiness' (engineering) while 'NIST AI RMF' targets 'Trust
Readiness' (policy).

**Data & Results:**
- **OWASP Top 10 LLM (n=10):** 90% (9/10) mapped to Integration Readiness, 10%
(1/10) to Trust Readiness.
- **NIST AI RMF 1.0 (n=19):** 42% (8/19) mapped to Integration Readiness, 58%
(11/19) to Trust Readiness.

**Statistical Significance:**
A Chi-Square test of independence yielded a statistic of 4.3786 and a **p-value
of 0.0364**. This is below the significance threshold of 0.05, allowing us to
reject the null hypothesis.

**Conclusion:**
The analysis confirms a statistically significant difference in focus. OWASP is
overwhelmingly focused on engineering controls (Integration), while NIST
provides a more balanced approach with a slight majority focus on
governance/policy (Trust). The hypothesis is supported.

---

## Review

The experiment successfully tested and supported the hypothesis that governance
frameworks exhibit distinct biases in competency mapping.

**Methodology:**
Using the `step2_crosswalk_matrix.csv` dataset, a Chi-Square test of
independence was performed to analyze the relationship between framework source
('OWASP Top 10 LLM' vs. 'NIST AI RMF 1.0') and competency bundle ('Integration
Readiness' vs. 'Trust Readiness').

**Results:**
- **OWASP Top 10 LLM (n=10):** Heavily skewed towards engineering, with 90% (9)
of requirements mapping to 'Integration Readiness' and only 10% (1) to 'Trust
Readiness'.
- **NIST AI RMF 1.0 (n=19):** Showed a more balanced but policy-leaning
distribution, with 58% (11) mapping to 'Trust Readiness' and 42% (8) to
'Integration Readiness'.
- **Statistical Significance:** The Chi-Square test yielded a statistic of
4.3786 and a **p-value of 0.0364**.

**Conclusion:**
With p < 0.05, the null hypothesis is rejected. The analysis confirms a
statistically significant difference: OWASP Top 10 LLM is predominantly focused
on 'Integration Readiness' (technical/engineering controls), while NIST AI RMF
prioritizes 'Trust Readiness' (governance/policy controls).

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def load_dataset(filename):
    # Check current directory
    if os.path.exists(filename):
        return pd.read_csv(filename)
    # Check parent directory
    elif os.path.exists(os.path.join('..', filename)):
        return pd.read_csv(os.path.join('..', filename))
    else:
        raise FileNotFoundError(f"Could not find {filename} in current or parent directory.")

# 1. Load dataset
filename = 'step2_crosswalk_matrix.csv'
try:
    df = load_dataset(filename)
    print(f"Successfully loaded {filename}")
except FileNotFoundError as e:
    print(e)
    sys.exit(1)

# 2. Filter for relevant sources
# We want to compare OWASP Top 10 LLM (Security focus) vs NIST AI RMF 1.0 (Risk focus)
target_sources = ['OWASP Top 10 LLM', 'NIST AI RMF 1.0']

# Filter and create a copy to avoid SettingWithCopyWarning
filtered_df = df[df['source'].isin(target_sources)].copy()

print(f"\nFiltered Data Shape: {filtered_df.shape}")
print("Source Counts:\n", filtered_df['source'].value_counts())
print("Bundle Counts:\n", filtered_df['bundle'].value_counts())

# 3. Create Contingency Table
# Rows: Source, Columns: Bundle
contingency_table = pd.crosstab(filtered_df['source'], filtered_df['bundle'])
print("\nContingency Table (Observed):\n", contingency_table)

# 4. Perform Chi-Square Test of Independence
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\n=== Chi-Square Test Results ===")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")

print("\nExpected Frequencies:\n", pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))

# Interpretation
alpha = 0.05
print("\n=== Interpretation ===")
if p < alpha:
    print(f"P-value ({p:.4f}) < {alpha}: Reject Null Hypothesis.")
    print("There is a statistically significant association between the Framework Source and the Competency Bundle.")
    print("This supports the hypothesis of framework bias (Security vs Risk).")
else:
    print(f"P-value ({p:.4f}) >= {alpha}: Fail to Reject Null Hypothesis.")
    print("There is no statistically significant association found.")

# 5. Visualization
plt.figure(figsize=(10, 6))

# Calculate proportions for stacked bar chart
contingency_props = contingency_table.div(contingency_table.sum(1), axis=0)

# Plot
ax = contingency_props.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e'], alpha=0.9)

plt.title('Competency Bundle Distribution by Framework Source')
plt.xlabel('Framework Source')
plt.ylabel('Proportion of Requirements')
plt.legend(title='Competency Bundle', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Annotate bars with counts and percentages
for i, (idx, row) in enumerate(contingency_props.iterrows()):
    cum_height = 0
    for col in contingency_props.columns:
        height = row[col]
        count = contingency_table.loc[idx, col]
        if height > 0:
            plt.text(i, cum_height + height/2, f"{count}\n({height:.1%})", 
                     ha='center', va='center', color='white', fontweight='bold')
            cum_height += height

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Successfully loaded step2_crosswalk_matrix.csv

Filtered Data Shape: (29, 24)
Source Counts:
 source
NIST AI RMF 1.0     19
OWASP Top 10 LLM    10
Name: count, dtype: int64
Bundle Counts:
 bundle
Integration Readiness    17
Trust Readiness          12
Name: count, dtype: int64

Contingency Table (Observed):
 bundle            Integration Readiness  Trust Readiness
source                                                  
NIST AI RMF 1.0                       8               11
OWASP Top 10 LLM                      9                1

=== Chi-Square Test Results ===
Chi-Square Statistic: 4.3786
P-value: 0.0364
Degrees of Freedom: 1

Expected Frequencies:
 bundle            Integration Readiness  Trust Readiness
source                                                  
NIST AI RMF 1.0               11.137931         7.862069
OWASP Top 10 LLM               5.862069         4.137931

=== Interpretation ===
P-value (0.0364) < 0.05: Reject Null Hypothesis.
There is a statistically significant association between the Framework Source and the Competency Bundle.
This supports the hypothesis of framework bias (Security vs Risk).

STDERR:
<ipython-input-1-28ab67acfa4f>:71: Pandas4Warning: Starting with pandas version 4.0 all arguments of sum will be keyword-only.
  contingency_props = contingency_table.div(contingency_table.sum(1), axis=0)


=== Plot Analysis (figure 2) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** This chart visualizes the relative distribution (proportions) of two specific categories ("Competency Bundles") within two different groups ("Framework Sources"). By normalizing the height of the bars to 1.0 (100%), it allows for a direct comparison of the composition of requirements between the frameworks, regardless of the total number of requirements in each.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Framework Source"
    *   **Labels:** Two categorical variables: "NIST AI RMF 1.0" and "OWASP Top 10 LLM".
*   **Y-Axis:**
    *   **Title:** "Proportion of Requirements"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100%).
    *   **Intervals:** Marked at 0.2 increments (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
*   **NIST AI RMF 1.0:**
    *   Shows a more balanced distribution between the two bundles compared to the OWASP framework.
    *   The majority share is held by **Trust Readiness** (Orange) at roughly 58%, while **Integration Readiness** (Blue) comprises roughly 42%.
*   **OWASP Top 10 LLM:**
    *   Displays a highly skewed distribution.
    *   It is overwhelmingly dominated by **Integration Readiness** (Blue), which occupies 90% of the bar.
    *   **Trust Readiness** (Orange) represents a very small minority at only 10%.

### 4. Annotations and Legends
*   **Legend:** located on the top right, titled "Competency Bundle."
    *   **Blue:** Represents "Integration Readiness".
    *   **Orange:** Represents "Trust Readiness".
*   **Annotations:** White text overlaying the bar segments provides exact values:
    *   **NIST AI RMF 1.0:**
        *   Integration Readiness: Count of **8** (42.1%).
        *   Trust Readiness: Count of **11** (57.9%).
    *   **OWASP Top 10 LLM:**
        *   Integration Readiness: Count of **9** (90.0%).
        *   Trust Readiness: Count of **1** (10.0%).

### 5. Statistical Insights
*   **Divergent Focus:** The two frameworks prioritize these competency bundles very differently. The **OWASP Top 10 LLM** framework appears to be heavily focused on the technical or operational aspects of implementation (Integration Readiness), whereas the **NIST AI RMF 1.0** places a higher emphasis on governance, reliability, or ethical considerations (Trust Readiness).
*   **Sample Size:** While the chart is normalized to show proportions, the annotations reveal the sample sizes are relatively small. The NIST column represents a total of 19 requirements (8 + 11), while the OWASP column represents 10 requirements (9 + 1).
*   **Inverse Relationship:** There is an inverse relationship visible between the frameworks regarding these bundles; where NIST leans toward Trust, OWASP leans heavily toward Integration.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
