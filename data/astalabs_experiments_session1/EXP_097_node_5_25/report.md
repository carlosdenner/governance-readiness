# Experiment 97: node_5_25

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_25` |
| **ID in Run** | 97 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:43:15.806745+00:00 |
| **Runtime** | 232.6s |
| **Parent** | `node_4_21` |
| **Children** | None |
| **Creation Index** | 98 |

---

## Hypothesis

> Sub-competencies within the 'Integration Readiness' bundle map to a
significantly higher mean number of unique adversarial tactics than those in the
'Trust Readiness' bundle, indicating a broader technical attack surface.

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
| Definitely False | 90.0 |

---

## Experiment Plan

**Objective:** Compare the tactical breadth of Trust vs Integration competencies.

### Steps
- 1. Load 'step3_tactic_frequency.csv'.
- 2. Group the data by 'sub_competency_id' and 'bundle'.
- 3. Count the number of unique 'tactic_name' entries for each sub-competency.
- 4. Perform a T-test comparing the mean unique tactic counts of Trust Readiness sub-competencies against Integration Readiness sub-competencies.
- 5. Generate a Violin Plot comparing the distributions.

### Deliverables
- T-test results and Violin Plot.

---

## Analysis

The experiment successfully loaded the 'step3_tactic_frequency.csv' dataset and
compared the tactical breadth of the two competency bundles. The analysis
included 9 sub-competencies (5 Trust, 4 Integration). The results show nearly
identical means for unique adversarial tactics: Trust Readiness (Mean=1.80,
SD=0.84) and Integration Readiness (Mean=1.75, SD=1.50). The independent samples
T-test yielded a p-value of 0.955, indicating no statistically significant
difference between the groups. Therefore, the hypothesis that Integration
Readiness competencies map to a significantly broader attack surface is
rejected. However, the standard deviations and the generated Violin Plot reveal
that Integration Readiness exhibits higher variance, containing both the sub-
competency with the highest tactic count (IR-2, n=4) and several with the lowest
(n=1), whereas Trust Readiness is more consistent.

---

## Review

The experiment successfully loaded the 'step3_tactic_frequency.csv' dataset and
tested the hypothesis regarding tactical breadth. The analysis, covering 9 sub-
competencies (5 Trust, 4 Integration), revealed that the mean number of unique
adversarial tactics is nearly identical between the two bundles (Trust: 1.80,
Integration: 1.75). The independent samples T-test yielded a p-value of 0.955,
indicating no statistically significant difference, which leads to the rejection
of the hypothesis that Integration Readiness competencies map to a significantly
broader attack surface. However, the Violin Plot and standard deviation metrics
(Trust: 0.84 vs. Integration: 1.50) uncover a nuanced finding: Integration
Readiness exhibits significantly higher variance, encompassing sub-competencies
with both the highest complexity (4 tactics) and the lowest (1 tactic), whereas
Trust Readiness is operationally more consistent.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Function to load dataset
def load_dataset(filename):
    paths = [filename, os.path.join('..', filename)]
    for path in paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f"Could not find {filename} in {paths}")

# 1. Load Data
df = load_dataset('step3_tactic_frequency.csv')

# 2. Data Processing
# Group by sub_competency_id and bundle, then count unique tactic_ids
tactic_counts = df.groupby(['sub_competency_id', 'bundle'])['tactic_id'].nunique().reset_index()
tactic_counts.rename(columns={'tactic_id': 'unique_tactic_count'}, inplace=True)

print("Summary of Tactic Counts per Sub-Competency:")
print(tactic_counts)

# 3. Statistical Test
trust_group = tactic_counts[tactic_counts['bundle'] == 'Trust Readiness']['unique_tactic_count']
integration_group = tactic_counts[tactic_counts['bundle'] == 'Integration Readiness']['unique_tactic_count']

print(f"\nTrust Readiness (n={len(trust_group)}): Mean = {trust_group.mean():.2f}, Std = {trust_group.std():.2f}")
print(f"Integration Readiness (n={len(integration_group)}): Mean = {integration_group.mean():.2f}, Std = {integration_group.std():.2f}")

# T-test (independent samples)
# Using Welch's t-test (equal_var=False) due to small sample sizes and potential variance differences
t_stat, p_val = stats.ttest_ind(trust_group, integration_group, equal_var=False)
print(f"\nT-test results: Statistic = {t_stat:.4f}, p-value = {p_val:.4f}")

# 4. Visualization
plt.figure(figsize=(10, 6))
# Fix for seaborn warning: assign x to hue and set legend=False
sns.violinplot(x='bundle', y='unique_tactic_count', hue='bundle', data=tactic_counts, inner='stick', palette='muted', legend=False)
plt.title('Distribution of Unique Adversarial Tactics per Competency Bundle')
plt.ylabel('Count of Unique Tactics')
plt.xlabel('Competency Bundle')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Summary of Tactic Counts per Sub-Competency:
  sub_competency_id                 bundle  unique_tactic_count
0              IR-1  Integration Readiness                    1
1              IR-2  Integration Readiness                    4
2              IR-6  Integration Readiness                    1
3              IR-8  Integration Readiness                    1
4              TR-1        Trust Readiness                    1
5              TR-2        Trust Readiness                    2
6              TR-3        Trust Readiness                    2
7              TR-4        Trust Readiness                    3
8              TR-6        Trust Readiness                    1

Trust Readiness (n=5): Mean = 1.80, Std = 0.84
Integration Readiness (n=4): Mean = 1.75, Std = 1.50

T-test results: Statistic = 0.0597, p-value = 0.9550


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Violin Plot.
*   **Purpose:** This plot visualizes the distribution of numeric data ("Count of Unique Tactics") across two different categories ("Competency Bundles"). It combines a box plot with a kernel density plot, showing not just summary statistics (like medians and ranges) but also the probability density of the data at different values. The width of the violin at any given y-value represents the frequency or density of data points at that value.

### 2. Axes
*   **X-axis:**
    *   **Title:** "Competency Bundle"
    *   **Labels:** Two categories are displayed: "Integration Readiness" and "Trust Readiness".
*   **Y-axis:**
    *   **Title:** "Count of Unique Tactics"
    *   **Units:** Integer counts (number of tactics).
    *   **Value Range:** The axis markings range from -1 to 6. Note: While the plot extends into negative numbers due to the smoothing function of the visualization (Kernel Density Estimation), the actual data likely starts at 0, as a count of tactics cannot be negative.

### 3. Data Trends
*   **Integration Readiness (Blue Violin):**
    *   **Shape:** This distribution is multi-modal or skewed. It has a very wide base around a count of 1, indicating a high concentration of data points with a low tactic count. However, there is a significant secondary bulge (expansion in width) around the 3 to 5 range.
    *   **Range:** It has a wider spread than the Trust Readiness group, extending from 0 up to roughly 6 unique tactics. This suggests higher variability in this category.
*   **Trust Readiness (Orange Violin):**
    *   **Shape:** This distribution is more compact and diamond-shaped. The widest point is centered around the count of 2.
    *   **Range:** The range is narrower, roughly falling between 0 and 4. The distribution tapers off symmetrically, suggesting that cases with high numbers of unique tactics (above 4) are rare or non-existent in this group.

### 4. Annotations and Legends
*   **Gridlines:** Horizontal dashed gridlines are placed at every integer interval (0, 1, 2, etc.) to aid in reading the specific counts.
*   **Inner Lines (Quartiles):** Inside each violin, there are three thin horizontal lines representing the quartiles:
    *   **Integration Readiness:** The median (middle line) appears to be around 1. The upper quartile (top line) is at 4, and the lower quartile (bottom line) is near 0 or 1.
    *   **Trust Readiness:** The median is clearly at 2. The upper quartile is at 3, and the lower quartile is at 1.
*   **Colors:** Blue is used for "Integration Readiness" and Orange for "Trust Readiness".

### 5. Statistical Insights
*   **Variability vs. Consistency:** "Trust Readiness" is a more consistent bundle; you can reliably expect between 1 and 3 unique adversarial tactics. "Integration Readiness" is highly variable; while the most common outcome is a single tactic (count of 1), there is a significant risk of encountering a much higher complexity (4 to 6 tactics).
*   **Central Tendency:** The median count for "Trust Readiness" (2) is higher than the median for "Integration Readiness" (1). However, "Integration Readiness" has a much higher maximum capacity for unique tactics.
*   **High-Risk Identification:** If the "Count of Unique Tactics" serves as a proxy for difficulty or vulnerability, the "Integration Readiness" bundle presents the "worst-case" scenarios, containing the outliers with the highest number of unique tactics (the peak reaching up to 6).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
