# Experiment 51: node_4_14

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_14` |
| **ID in Run** | 51 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:10:38.992771+00:00 |
| **Runtime** | 171.1s |
| **Parent** | `node_3_3` |
| **Children** | `node_5_8`, `node_5_13` |
| **Creation Index** | 52 |

---

## Hypothesis

> Competency statements rated with 'High' confidence are supported by a
significantly larger volume of bibliographic evidence (citations) than those
rated 'Medium' or 'Low'.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.1901 (Likely False) |
| **Surprise** | -0.6405 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 3.0 |
| Maybe True | 25.0 |
| Uncertain | 1.0 |
| Maybe False | 1.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 90.0 |

---

## Experiment Plan

**Objective:** Validate if confidence scores correlate with evidence density.

### Steps
- 1. Load 'step2_competency_statements.csv'.
- 2. Create a 'citation_count' column by using regex to count occurrences of citation markers (e.g., '[#XX]') in the 'evidence_summary' column.
- 3. Convert 'confidence' to an ordinal scale or keep as categorical (High/Medium/Low).
- 4. Perform a One-Way ANOVA or Kruskal-Wallis test to compare 'citation_count' across confidence levels.
- 5. If significant, perform post-hoc tests (e.g., Tukey's HSD) to identify which groups differ.

### Deliverables
- Boxplot of citation counts by confidence level and ANOVA/Kruskal-Wallis results.

---

## Analysis

The experiment tested the hypothesis that higher confidence scores in competency
statements are driven by a higher volume of bibliographic evidence. The analysis
of 'step2_competency_statements.csv' (n=42) revealed that 'High' confidence
statements (n=36) have a mean citation count of 2.19, while 'Medium' confidence
statements (n=6) have a mean of 2.17. The Kruskal-Wallis H-test yielded a
p-value of 0.8651, indicating no statistically significant difference between
the groups. Consequently, the hypothesis is rejected. The data suggests that
confidence ratings in this dataset are determined by qualitative factors (e.g.,
source authority, directness of the requirement mapping) rather than the sheer
quantity of citations, as the volume of evidence is uniform across confidence
levels (median = 2 citations).

---

## Review

The experiment pipeline was faithfully implemented. The final experiment
rejected the hypothesis that 'High' confidence competency statements have more
citations than 'Medium' ones (Kruskal-Wallis p=0.8651); evidence volume is
uniform (median=2) across groups.

Overall, the analysis of the 5-step pipeline reveals:
1. **Holistic Failures**: AI incidents do not separate cleanly into 'Trust' or
'Integration' gaps but involve both simultaneously (88% of cases).
2. **Prevention Dominance**: 98% of analyzed failures are 'Prevention Failures',
making specific attack techniques (like Prompt Injection) poor predictors of
failure mode.
3. **Qualitative Confidence**: Competency confidence scores are likely driven by
source authority rather than citation quantity.

The data suggests the research artifacts are internally consistent but reflect a
specific worldview where AI security requires unified, preventative governance
and engineering controls.

---

## Code

```python
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy import stats

# Load dataset
try:
    df = pd.read_csv('../step2_competency_statements.csv')
except FileNotFoundError:
    # Fallback if running in a different environment structure, though instruction says one level above
    try:
        df = pd.read_csv('step2_competency_statements.csv')
    except FileNotFoundError:
        print("Error: Could not find step2_competency_statements.csv")
        exit(1)

# Feature Engineering: Count citations in 'evidence_summary'
# Pattern looks for strings like [#21], [#1], etc.
def count_citations(text):
    if pd.isna(text):
        return 0
    return len(re.findall(r'\[#\d+\]', str(text)))

df['citation_count'] = df['evidence_summary'].apply(count_citations)

# Descriptive Statistics
print("=== Citation Counts by Confidence Level ===")
group_stats = df.groupby('confidence')['citation_count'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
print(group_stats)

# Prepare data for statistical test
# We define a logical order for the groups
ordered_levels = ['High', 'Medium', 'Low']
groups = []
labels = []

for level in ordered_levels:
    if level in df['confidence'].unique():
        data = df[df['confidence'] == level]['citation_count']
        groups.append(data)
        labels.append(level)

# Check for any other labels not in High/Medium/Low
other_levels = [x for x in df['confidence'].unique() if x not in ordered_levels]
for level in other_levels:
    data = df[df['confidence'] == level]['citation_count']
    groups.append(data)
    labels.append(level)

# Statistical Test: Kruskal-Wallis H-test
# Used instead of ANOVA due to potentially small sample sizes and count data (non-normality)
print("\n=== Statistical Test Results ===")
if len(groups) > 1:
    stat, p_value = stats.kruskal(*groups)
    print(f"Test: Kruskal-Wallis H-test")
    print(f"Statistic: {stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Result: Significant difference found (p < 0.05).")
    else:
        print("Result: No significant difference found (p >= 0.05).")
else:
    print("Insufficient groups for statistical testing.")

# Visualization
plt.figure(figsize=(8, 6))
plt.boxplot(groups, labels=labels)
plt.title('Evidence Density: Citation Counts vs. Confidence Level')
plt.ylabel('Number of Citations per Statement')
plt.xlabel('Confidence Level')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: === Citation Counts by Confidence Level ===
            count      mean  median       std  min  max
confidence                                             
high           36  2.194444     2.0  0.467177    1    3
medium          6  2.166667     2.0  0.408248    2    3

=== Statistical Test Results ===
Test: Kruskal-Wallis H-test
Statistic: 0.0289
P-value: 0.8651
Result: No significant difference found (p >= 0.05).

STDERR:
<ipython-input-1-ed92a45699ca>:71: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(groups, labels=labels)


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (also known as a Box-and-Whisker Plot).
*   **Purpose:** This plot visualizes the distribution of "Number of Citations per Statement" across two categorical groups ("high" and "medium" confidence levels). It displays the central tendency (median), dispersion (spread), and outliers for each category.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Confidence Level"
    *   **Labels:** Two categorical variables: **"high"** and **"medium"**.
*   **Y-Axis:**
    *   **Title:** "Number of Citations per Statement"
    *   **Value Range:** The axis ticks range from **1.00 to 3.00**, with intervals of 0.25. The data appears to be discrete integer values (1, 2, 3).

### 3. Data Trends
*   **Central Tendency (Medians):** For both the "high" and "medium" confidence levels, the median line (represented by the orange bar) is located exactly at **2.0**.
*   **Distribution Shape (The Boxes):** The "boxes" (which typically represent the Interquartile Range, from the 25th to 75th percentile) appear as flat lines. This indicates extremely low variance; specifically, it suggests that the 25th percentile, the median, and the 75th percentile are all the same value (2 citations). The vast majority of statements in this dataset have exactly 2 citations.
*   **Outliers:**
    *   **High Confidence:** There are visible outliers (represented by open circles) at **3.0** and **1.0**. This indicates that while most high-confidence statements have 2 citations, a few have 1 or 3.
    *   **Medium Confidence:** There is a visible outlier at **3.0**. Unlike the "high" category, there is no visible outlier at 1.0, suggesting the minimum citation count for the "medium" group in this specific dataset is likely 2.

### 4. Annotations and Legends
*   **Title:** "Evidence Density: Citation Counts vs. Confidence Level" appears at the top, summarizing the plot's intent.
*   **Grid:** Dashed horizontal grid lines are present at every 0.25 interval to assist in reading the Y-axis values.
*   **Markers:**
    *   **Orange Lines:** Represent the median of the data.
    *   **Black Circles:** Represent outlier data points that fall outside the typical distribution range.

### 5. Statistical Insights
*   **Uniformity of Evidence:** There is a remarkable consistency in "Evidence Density" regardless of confidence level. Whether a statement is marked as "high" or "medium" confidence, the most probable number of supporting citations is **2**.
*   **Lack of Correlation:** The plot suggests that a higher confidence level is **not** driven by a higher quantity of citations. If quantity were the driving factor, we would expect the "high" box to be shifted higher up the Y-axis compared to the "medium" box. Instead, they are identical in central tendency.
*   **Discrete Data:** The data is discrete and tightly clustered. The lack of visible whiskers and the flattened boxes suggest the data consists almost entirely of the integer 2, with rare deviations to 1 or 3.
*   **Variance:** The "high" confidence group actually shows slightly *more* variance at the lower end (presence of statements with only 1 citation) compared to the "medium" group shown here.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
