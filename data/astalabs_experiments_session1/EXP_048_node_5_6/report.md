# Experiment 48: node_5_6

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_6` |
| **ID in Run** | 48 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:02:58.375397+00:00 |
| **Runtime** | 158.1s |
| **Parent** | `node_4_12` |
| **Children** | None |
| **Creation Index** | 49 |

---

## Hypothesis

> Incidents involving 'Both' Trust and Integration failures exhibit higher attack
sophistication (measured by technique count) than those involving single-domain
failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2087 (Likely False) |
| **Surprise** | -0.6189 |
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
| Maybe False | 9.0 |
| Definitely False | 81.0 |

---

## Experiment Plan

**Objective:** Investigate if multi-domain failures correlate with more complex attacks.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Group the data by 'trust_integration_split' into two categories: 'Both' vs. 'Distinct' (grouping Trust-dominant and Integration-dominant together due to low N).
- 3. Compare the mean 'technique_count' between these two groups using an independent samples t-test.
- 4. Visualize the distributions with a boxplot.

### Deliverables
- Mean technique count for Both vs. Distinct groups and t-test results.

---

## Analysis

The experiment successfully tested the hypothesis that incidents involving
multi-domain failures ('Both' Trust and Integration gaps) exhibit higher attack
sophistication than single-domain failures. Using 'technique_count' as a proxy
for sophistication, the analysis compared the 'Both' group (n=46) against the
'Distinct' group (n=6).

Contrary to the hypothesis, there was no statistically significant difference
between the groups (Welch's t-test: t=0.252, p=0.807). The mean technique count
was nearly identical (Both: 7.57 vs. Distinct: 7.33), and both groups shared a
median of 7 techniques. While the 'Both' group displayed a wider range (min=1,
max=16) and contained outliers, likely due to the larger sample size, the
central tendency suggests that the complexity of an adversarial attack is
independent of whether the resulting failure manifests across one or both
competency domains.

---

## Review

The experiment was successfully executed and the hypothesis was tested
methodically. The code correctly loaded the incident dataset, grouped the 52
incidents into 'Both' (n=46) and 'Distinct' (n=6) categories based on competency
gaps, and performed a Welch's t-test to compare attack sophistication (technique
count).

**Findings:**
1.  **Descriptive Stats:** The mean technique count was remarkably similar
between the two groups (Both: 7.57 vs. Distinct: 7.33), with identical medians
(7.0).
2.  **Statistical Test:** The t-test yielded a p-value of 0.8070, indicating no
statistically significant difference in attack sophistication between multi-
domain and single-domain failures.
3.  **Conclusion:** The hypothesis is rejected. The data suggests that the
complexity of an adversarial attack (number of techniques used) is not
correlated with whether the resulting failure manifests across one or both
competency domains. The 'Both' group showed higher variance (range 1-16)
compared to the 'Distinct' group (range 5-10), but this is likely attributable
to the significant difference in sample size.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Define file path based on instruction
file_path = '../step3_incident_coding.csv'

# Check if file exists there, otherwise try current directory as fallback
if not os.path.exists(file_path):
    file_path = 'step3_incident_coding.csv'

print(f"Loading dataset from: {file_path}")

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit(1)

# Normalize the split column to handle case sensitivity and whitespace
df['split_clean'] = df['trust_integration_split'].astype(str).str.lower().str.strip()

# Create the binary grouping: 'Both' vs 'Distinct'
df['group'] = df['split_clean'].apply(lambda x: 'Both' if x == 'both' else 'Distinct')

# Extract technique counts for each group
both_counts = df[df['group'] == 'Both']['technique_count']
distinct_counts = df[df['group'] == 'Distinct']['technique_count']

# Print group sizes
print("\nGroup Sample Sizes:")
print(f"Both: {len(both_counts)}")
print(f"Distinct: {len(distinct_counts)}")

# Descriptive Statistics
print("\nDescriptive Statistics for Technique Count:")
print(df.groupby('group')['technique_count'].describe())

# T-test (Using Welch's t-test due to likely unequal sample sizes and variance)
t_stat, p_val = stats.ttest_ind(both_counts, distinct_counts, equal_var=False)

print("\nIndependent Samples T-Test (Welch's):")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4f}")

# Interpretation
alpha = 0.05
if p_val < alpha:
    print("\nResult: Statistically significant difference found.")
else:
    print("\nResult: No statistically significant difference found.")

# Visualization
plt.figure(figsize=(8, 6))
plt.boxplot([both_counts, distinct_counts], labels=['Both', 'Distinct'])
plt.title('Attack Sophistication (Technique Count) by Competency Gap Type')
plt.ylabel('Number of Techniques Used')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_incident_coding.csv

Group Sample Sizes:
Both: 46
Distinct: 6

Descriptive Statistics for Technique Count:
          count      mean       std  min  25%  50%   75%   max
group                                                         
Both       46.0  7.565217  3.052519  1.0  6.0  7.0  9.00  16.0
Distinct    6.0  7.333333  1.966384  5.0  6.0  7.0  8.75  10.0

Independent Samples T-Test (Welch's):
t-statistic: 0.2520
p-value: 0.8070

Result: No statistically significant difference found.

STDERR:
<ipython-input-1-a65908e9540a>:59: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot([both_counts, distinct_counts], labels=['Both', 'Distinct'])


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (or Box-and-Whisker Plot).
*   **Purpose:** This plot is used to visualize and compare the distribution of a quantitative variable ("Number of Techniques Used") across two different categorical groups ("Both" and "Distinct"). It displays the central tendency (median), dispersion (interquartile range), and outliers for each group.

### 2. Axes
*   **X-Axis:**
    *   **Label/Categories:** Represents the "Competency Gap Type" with two distinct categories: **"Both"** and **"Distinct"**.
*   **Y-Axis:**
    *   **Label:** "Number of Techniques Used".
    *   **Units:** Count (integer values representing the number of techniques).
    *   **Range:** The axis spans from **0 to roughly 17**, with tick marks labeled every 2 units (2, 4, 6, ..., 16).

### 3. Data Trends
*   **Category: "Both"**
    *   **Spread:** This category exhibits a wide spread of data. The main body of data (the box) ranges from a lower quartile (Q1) of 6 to an upper quartile (Q3) of 9.
    *   **Whiskers:** The range of non-outlier data extends from a minimum of 3 to a maximum of 12.
    *   **Outliers:** There are significant outliers present. Specifically, there are high-value outliers at approximately 14, 15, and 16, and a low-value outlier at 1.
    *   **Median:** The median line (orange) is positioned at 7.

*   **Category: "Distinct"**
    *   **Spread:** This distribution is much tighter and more consistent than the "Both" category. The box ranges from a lower quartile (Q1) of 6 to an upper quartile (Q3) of roughly 8.5 or 9.
    *   **Whiskers:** The whiskers indicate a total range of 5 to 10.
    *   **Outliers:** There are no visible outliers in this category.
    *   **Median:** The median line is also positioned at 7.

### 4. Annotations and Legends
*   **Title:** "Attack Sophistication (Technique Count) by Competency Gap Type".
*   **Grid:** Horizontal dashed grid lines are included to assist in reading the Y-axis values.
*   **Colors:** The box plot uses standard formatting with black outlines for the boxes and whiskers, and an orange horizontal line to denote the median.

### 5. Statistical Insights
*   **Central Tendency Similarity:** Interestingly, both the "Both" and "Distinct" competency gap types share the same median value (7 techniques). This indicates that the "typical" or central level of attack sophistication is identical between the two groups.
*   **Variability Difference:** The key difference lies in consistency. The "Distinct" group represents a very predictable range of sophistication (most attacks use between 5 and 10 techniques). In contrast, the "Both" group is highly volatile; while the middle 50% is similar to the "Distinct" group, the "Both" group includes both the simplest attacks (1 technique) and the most sophisticated attacks (up to 16 techniques).
*   **Risk Profile:** From a security perspective, the "Both" category presents a more complex risk profile due to the presence of extreme outliers (high sophistication), whereas the "Distinct" category represents a bounded, moderate threat level.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
