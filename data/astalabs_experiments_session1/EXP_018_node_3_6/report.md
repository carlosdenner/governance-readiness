# Experiment 18: node_3_6

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_6` |
| **ID in Run** | 18 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:01:23.901368+00:00 |
| **Runtime** | 171.6s |
| **Parent** | `node_2_0` |
| **Children** | `node_4_26`, `node_4_29` |
| **Creation Index** | 19 |

---

## Hypothesis

> Integration Readiness competencies necessitate a significantly more complex
architectural footprint (measured by the count of applicable controls) compared
to Trust Readiness competencies.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.4000 (Maybe False) |
| **Posterior** | 0.2875 (Likely False) |
| **Surprise** | -0.1306 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 10.0 |
| Uncertain | 0.0 |
| Maybe False | 16.0 |
| Definitely False | 3.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 90.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Determine if Integration competencies differ structurally from Trust competencies by requiring a higher density of architecture controls.

### Steps
- 1. Load 'step2_competency_statements.csv'.
- 2. Create a new column 'control_count' by splitting the 'applicable_controls' string by the delimiter (semicolon) and counting the elements.
- 3. Group the data by the 'bundle' column ('Trust Readiness' vs. 'Integration Readiness').
- 4. Calculate descriptive statistics (mean, median, std) for 'control_count' for each bundle.
- 5. Perform a statistical test (independent t-test or Mann-Whitney U test depending on normality) to check for a significant difference in control counts between the two bundles.

### Deliverables
- Boxplot of control counts by bundle and statistical test results (t-statistic/U-statistic and p-value).

---

## Analysis

The experiment successfully tested the hypothesis that Integration Readiness
competencies require a significantly more complex architectural footprint than
Trust Readiness competencies. The code loaded the competency statements and
calculated the number of applicable architecture controls for each.

Descriptive statistics revealed that 'Integration Readiness' has a slightly
higher mean control count (2.00) compared to 'Trust Readiness' (1.68), though
both have a median of 2.0. 'Integration Readiness' showed significantly lower
variance (std=0.52) compared to 'Trust Readiness' (std=0.75), suggesting a more
standardized set of controls.

However, the Mann-Whitney U test yielded a p-value of 0.0902, which exceeds the
significance threshold of 0.05. Therefore, the difference is not statistically
significant. The null hypothesis cannot be rejected, and the experimental
hypothesis is unsupported: there is no significant difference in architectural
complexity (as measured by control count) between the two capability bundles.

---

## Review

The experiment successfully tested the hypothesis that 'Integration Readiness'
competencies require a significantly more complex architectural footprint (more
controls) than 'Trust Readiness' competencies.

**Methodology:**
- The dataset `step2_competency_statements.csv` was loaded.
- A `control_count` variable was derived by counting semicolon-separated entries
in the `applicable_controls` field.
- Descriptive statistics were calculated, and normality was assessed using the
Shapiro-Wilk test (p < 0.05 for both groups, indicating non-normality).
- A Mann-Whitney U test was performed to compare the distributions.

**Results:**
- **Integration Readiness:** Mean = 2.00, Median = 2.0, Std = 0.52 (n=23).
- **Trust Readiness:** Mean = 1.68, Median = 2.0, Std = 0.75 (n=19).
- **Statistical Test:** Mann-Whitney U statistic = 158.5, p-value = 0.0902.

**Findings:**
- The analysis found no statistically significant difference in architectural
complexity between the two bundles (p = 0.0902 > 0.05).
- Although 'Integration Readiness' had a slightly higher mean control count, the
difference was not strong enough to reject the null hypothesis given the sample
size and distribution.
- The hypothesis is therefore **unsupported**. The data suggests that both trust
and integration competencies generally map to a similar number of architectural
controls (typically 1-3).

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Define file path (one level above as per instructions)
file_path = '../step2_competency_statements.csv'

# Load dataset
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    # Fallback to current directory for robustness if the file isn't found at ../
    try:
        df = pd.read_csv('step2_competency_statements.csv')
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

print(f"Dataset loaded. Shape: {df.shape}")

# Calculate control counts
# Assuming semicolon separation based on dataset description
df['control_count'] = df['applicable_controls'].apply(lambda x: len([c for c in str(x).split(';') if c.strip()]) if pd.notna(x) else 0)

# Group by bundle
bundle_groups = df.groupby('bundle')

# Descriptive Statistics
summary_stats = bundle_groups['control_count'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
print("\n=== Descriptive Statistics for Control Counts ===")
print(summary_stats)

# Extract series for testing
trust_data = df[df['bundle'] == 'Trust Readiness']['control_count']
integration_data = df[df['bundle'] == 'Integration Readiness']['control_count']

# Normality Check (Shapiro-Wilk)
_, p_trust = stats.shapiro(trust_data)
_, p_int = stats.shapiro(integration_data)

print("\n=== Normality Check (Shapiro-Wilk) ===")
print(f"Trust Readiness: p={p_trust:.4f}")
print(f"Integration Readiness: p={p_int:.4f}")

# Statistical Test Selection
# Use Mann-Whitney U test if data is not normal or sample size is small, otherwise t-test
# Given n is roughly 20 per group, and count data often isn't normal, Mann-Whitney is safer.
use_parametric = (p_trust > 0.05) and (p_int > 0.05)

if use_parametric:
    test_stat, p_val = stats.ttest_ind(trust_data, integration_data, equal_var=False)
    test_name = "Welch's t-test"
else:
    test_stat, p_val = stats.mannwhitneyu(trust_data, integration_data, alternative='two-sided')
    test_name = "Mann-Whitney U test"

print(f"\n=== Hypothesis Test ({test_name}) ===")
print(f"Statistic: {test_stat:.4f}")
print(f"P-value:   {p_val:.4f}")

if p_val < 0.05:
    print("Conclusion: Significant difference in control complexity between bundles (Reject H0).")
else:
    print("Conclusion: No significant difference in control complexity (Fail to reject H0).")

# Visualization
plt.figure(figsize=(10, 6))
data_to_plot = [trust_data, integration_data]
labels = ['Trust Readiness', 'Integration Readiness']

# Boxplot
bplot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True, medianprops=dict(color='black'))

# Colors
colors = ['lightblue', 'lightgreen']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

# Jitter plot overlay
for i, data in enumerate(data_to_plot):
    y = data
    x = np.random.normal(1 + i, 0.04, size=len(y))
    plt.scatter(x, y, alpha=0.6, color='darkblue', s=20)

plt.title('Complexity Comparison: Count of Architecture Controls per Competency')
plt.ylabel('Number of Applicable Controls')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Dataset loaded. Shape: (42, 8)

=== Descriptive Statistics for Control Counts ===
                       count      mean  median       std  min  max
bundle                                                            
Integration Readiness     23  2.000000     2.0  0.522233    1    3
Trust Readiness           19  1.684211     2.0  0.749269    1    3

=== Normality Check (Shapiro-Wilk) ===
Trust Readiness: p=0.0005
Integration Readiness: p=0.0000

=== Hypothesis Test (Mann-Whitney U test) ===
Statistic: 158.5000
P-value:   0.0902
Conclusion: No significant difference in control complexity (Fail to reject H0).

STDERR:
<ipython-input-1-f17b16091f25>:77: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  bplot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True, medianprops=dict(color='black'))


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the plot:

### 1. Plot Type
*   **Type:** This is a **Box Plot (Box-and-Whisker Plot)** overlaid with a **Strip Plot** (jittered scatter points).
*   **Purpose:** The plot compares the distribution of a numerical variable (count of controls) across two categorical groups ("Trust Readiness" and "Integration Readiness"). The box plot visualizes summary statistics (medians, quartiles, and outliers), while the overlaid scatter points show the individual data density, preventing information loss due to over-summarization.

### 2. Axes
*   **X-Axis:**
    *   **Label:** Represents the different competencies.
    *   **Categories:** "Trust Readiness" and "Integration Readiness".
*   **Y-Axis:**
    *   **Label:** "Number of Applicable Controls".
    *   **Units:** Count (integer values).
    *   **Range:** The visual axis ranges from approximately 0.9 to 3.1, with tick marks every 0.25 units. The actual data points fall on integer values: 1.0, 2.0, and 3.0.

### 3. Data Trends
*   **Trust Readiness (Left):**
    *   **Distribution:** Shows a broader spread than Integration Readiness. The "box" component spans from a lower quartile (Q1) of 1.0 to an upper quartile (Q3) of 2.0.
    *   **Range:** The whiskers extend up to 3.0, indicating the data spans the full range of 1 to 3 controls without any values being statistically classified as outliers.
    *   **Cluster:** There is a heavy concentration of individual data points (purple dots) at both 1.0 and 2.0.

*   **Integration Readiness (Right):**
    *   **Distribution:** Highly concentrated. The box plot appears as a single horizontal line at 2.0. This indicates that the 25th percentile, Median, and 75th percentile are all identical (value = 2.0).
    *   **Outliers:** There are open black circles at 1.0 and 3.0. In box plot syntax, these represent statistical outliers, suggesting that while counts of 1 and 3 exist, they are rare compared to the overwhelming frequency of the count 2.
    *   **Cluster:** The purple scatter points are densely packed at the value 2.0.

### 4. Annotations and Legends
*   **Title:** "Complexity Comparison: Count of Architecture Controls per Competency".
*   **Gridlines:** Horizontal dashed gridlines facilitate easy reading of the Y-axis values.
*   **Markers:**
    *   **Blue/Purple Dots:** Represent individual data points (jittered horizontally to show density).
    *   **Black Open Circles:** Represent outlier values determined by the box plot's interquartile range calculation (visible on the right side).
    *   **Light Blue Box:** Represents the Interquartile Range (IQR), covering the middle 50% of the data.

### 5. Statistical Insights
*   **Variability:** "Trust Readiness" is more variable in terms of complexity. A project or component in this category is reasonably likely to have 1, 2, or 3 controls.
*   **Predictability:** "Integration Readiness" is highly predictable. The vast majority of items in this category require exactly 2 architecture controls. Deviations (1 or 3 controls) are statistically rare exceptions.
*   **Median Values:** While exact calculation requires raw data, the visual evidence suggests the median for "Integration Readiness" is strictly 2.0. For "Trust Readiness," the median is contained within the 1.0 to 2.0 range.
*   **Overall Complexity:** Both competencies top out at 3 controls, but "Trust Readiness" frequently allows for simpler implementations (1 control), whereas "Integration Readiness" standardizes heavily on 2 controls.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
