# Experiment 22: node_3_8

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_8` |
| **ID in Run** | 22 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:08:58.648356+00:00 |
| **Runtime** | 193.1s |
| **Parent** | `node_2_2` |
| **Children** | `node_4_19`, `node_4_27` |
| **Creation Index** | 23 |

---

## Hypothesis

> Competencies classified as 'Integration Readiness' require a significantly
higher number of distinct architecture controls to implement than those
classified as 'Trust Readiness', reflecting a higher technical complexity.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.5333 (Uncertain) |
| **Posterior** | 0.3208 (Maybe False) |
| **Surprise** | -0.2466 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 16.0 |
| Uncertain | 1.0 |
| Maybe False | 12.0 |
| Definitely False | 0.0 |

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

**Objective:** Compare the architectural 'density' (number of required controls) between the two competency bundles.

### Steps
- 1. Load 'step2_competency_statements.csv'.
- 2. Create a new column 'control_count' by splitting the 'applicable_controls' string by the delimiter (semicolon) and counting the elements.
- 3. Group the data by 'bundle' ('Trust Readiness' vs. 'Integration Readiness') and calculate descriptive statistics for 'control_count'.
- 4. Perform an independent sample t-test (or Mann-Whitney U test) to compare the means of 'control_count' between the two bundles.

### Deliverables
- Descriptive statistics (mean, std) per bundle and T-test/Mann-Whitney results.

---

## Analysis

The experiment was successfully executed. The 'step2_competency_statements.csv'
dataset was loaded, and the 'control_count' feature was correctly derived by
counting semicolon-separated entries in the 'applicable_controls' column.
Descriptive statistics revealed that 'Integration Readiness' competencies have a
slightly higher mean control count (2.00) compared to 'Trust Readiness' (1.68),
but 'Trust Readiness' exhibits higher variance (std=0.75 vs 0.52). The Mann-
Whitney U test yielded a p-value of 0.0902, and the T-test yielded a p-value of
0.1308. Both are above the standard alpha of 0.05, leading to the conclusion
that there is no statistically significant difference in architectural density
between the two bundles. The hypothesis that Integration Readiness requires
significantly more controls is therefore not supported by this metric.

---

## Review

The experiment was successfully executed and faithfully implemented the analysis
plan. The dataset 'step2_competency_statements.csv' was correctly loaded (after
correcting the file path), and the 'control_count' metric was accurately derived
by parsing the semicolon-separated 'applicable_controls' field. The statistical
analysis compared the architectural density of 'Integration Readiness'
(Mean=2.00, SD=0.52) versus 'Trust Readiness' (Mean=1.68, SD=0.75). Although
'Integration Readiness' competencies showed a slightly higher average number of
controls, the Mann-Whitney U test (p=0.0902) and Welch's T-test (p=0.1308)
indicated that this difference is not statistically significant at the 0.05
level. Therefore, the hypothesis that Integration Readiness requires a
significantly higher number of architecture controls is not supported by the
data.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# Check file existence before loading
file_name = 'step2_competency_statements.csv'
file_path = file_name if os.path.exists(file_name) else '../' + file_name

print(f"Loading dataset from {file_path}...")
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: Could not find {file_name} in current or parent directory.")
    exit(1)

# Verify columns exist
if 'applicable_controls' not in df.columns or 'bundle' not in df.columns:
    print("Error: Required columns not found.")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

# Calculate architectural density (control_count)
def count_controls(val):
    if pd.isna(val) or str(val).strip() == '':
        return 0
    # Split by semicolon and strip whitespace to count valid entries
    controls = [c.strip() for c in str(val).split(';') if c.strip()]
    return len(controls)

df['control_count'] = df['applicable_controls'].apply(count_controls)

# Group by bundle
group_integration = df[df['bundle'] == 'Integration Readiness']['control_count']
group_trust = df[df['bundle'] == 'Trust Readiness']['control_count']

# Calculate descriptive statistics
stats_df = df.groupby('bundle')['control_count'].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
print("\n=== Descriptive Statistics (Control Count) ===")
print(stats_df)

# Perform Mann-Whitney U Test (Non-parametric test for difference in distributions)
# We use Mann-Whitney because control counts are discrete and sample sizes are small
u_stat, p_val_mw = stats.mannwhitneyu(group_integration, group_trust, alternative='two-sided')

# Perform Independent T-test (for comparison)
t_stat, p_val_t = stats.ttest_ind(group_integration, group_trust, equal_var=False)

print("\n=== Statistical Test Results ===")
print(f"Mann-Whitney U Test: U={u_stat:.1f}, p-value={p_val_mw:.4f}")
print(f"Welch's T-test:      t={t_stat:.4f}, p-value={p_val_t:.4f}")

interpretation = ""
if p_val_mw < 0.05:
    interpretation = "Result: Statistically significant difference detected."
else:
    interpretation = "Result: No statistically significant difference detected."
print(f"\n{interpretation}")

# Visualization
plt.figure(figsize=(10, 6))
data_to_plot = [group_integration, group_trust]
labels = ['Integration Readiness', 'Trust Readiness']

plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))

plt.title('Distribution of Architecture Control Density by Bundle')
plt.ylabel('Number of Applicable Controls')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add scatter points
for i, data in enumerate(data_to_plot):
    y = data
    x = np.random.normal(i + 1, 0.04, size=len(y))
    plt.plot(x, y, 'r.', alpha=0.5)

plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from step2_competency_statements.csv...

=== Descriptive Statistics (Control Count) ===
                       count      mean       std  median  min  max
bundle                                                            
Integration Readiness     23  2.000000  0.522233     2.0    1    3
Trust Readiness           19  1.684211  0.749269     2.0    1    3

=== Statistical Test Results ===
Mann-Whitney U Test: U=278.5, p-value=0.0902
Welch's T-test:      t=1.5519, p-value=0.1308

Result: No statistically significant difference detected.

STDERR:
<ipython-input-1-f662c3a0d09c>:69: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data_to_plot, labels=labels, patch_artist=True,


=== Plot Analysis (figure 1) ===
Based on the provided image, here is a detailed analysis of the plot:

### 1. Plot Type
**Type:** Box Plot with an overlay of a Jitter/Strip Plot.
**Purpose:** This plot is designed to visualize and compare the distribution of a numerical variable ("Number of Applicable Controls") across two categorical groups ("Integration Readiness" and "Trust Readiness").
*   The **box plot** component displays statistical summaries (median, quartiles, and outliers).
*   The **jitter plot** (red dots) displays the individual data points, revealing the sample size and specific density of data at discrete values.

### 2. Axes
*   **Title:** "Distribution of Architecture Control Density by Bundle"
*   **Y-Axis:**
    *   **Label:** "Number of Applicable Controls"
    *   **Range:** The visual axis spans from **1.00 to 3.00**.
    *   **Scale:** Linear, with major ticks marked every 0.25 units. The data appears to be discrete integers (1, 2, 3).
*   **X-Axis:**
    *   **Label:** The axis represents the "Bundle" category (though not explicitly labeled "Bundle").
    *   **Categories:** Two specific bundles are compared:
        1.  **Integration Readiness**
        2.  **Trust Readiness**

### 3. Data Trends
**Integration Readiness (Left Group):**
*   **Clustering/Variance:** The data is extremely concentrated. The vast majority of data points (red dots) are clustered tightly at the value of **2.0**.
*   **Box Plot Shape:** The box plot appears "collapsed" into a single line at y=2. This indicates that the 25th percentile (Q1), Median, and 75th percentile (Q3) are all equal to 2.
*   **Outliers:** There are distinct outlier points visible at **1.0** and **3.0** (marked by black circles and red dots), but these represent a small minority of the cases.

**Trust Readiness (Right Group):**
*   **Clustering/Variance:** This group shows significantly higher variance compared to the Integration group. The data points are distributed across values 1, 2, and 3.
*   **Box Plot Shape:** There is a clearly visible light blue box.
    *   The bottom of the box is at **1.0** (25th percentile).
    *   The top of the box is at **2.0** (75th percentile).
    *   The median line (red horizontal bar inside the box) is at **2.0**.
*   **Whiskers:** There is an upper whisker extending to **3.0**, indicating data points exist in the upper range within 1.5 times the Interquartile Range (IQR).

### 4. Annotations and Legends
*   **Red Dots:** Represent individual observations/data points. They are slightly transparent to show density where points overlap.
*   **Red Horizontal Line:** Represents the median value within the box plots.
*   **Light Blue Box:** Represents the Interquartile Range (IQR), covering the middle 50% of the data.
*   **Black Circles:** Represent outliers calculated by the box plot algorithm (visible specifically in the Integration Readiness column).
*   **Grid Lines:** Horizontal dashed grey lines facilitate easier reading of the y-axis values.

### 5. Statistical Insights
*   **Central Tendency:** Both bundles have a median "Number of Applicable Controls" of **2**.
*   **Consistency:** The **Integration Readiness** bundle is highly consistent. It is almost strictly standardized to 2 controls, with very few exceptions.
*   **Variability:** The **Trust Readiness** bundle is much more variable. While its median is also 2, a significant portion of its data falls at 1 (indicated by the box extending down to 1), and there is a notable spread up to 3.
*   **Distribution Skew:** The Trust Readiness data appears slightly skewed or uniformly distributed across the lower half (1 to 2) compared to the upper half, whereas Integration Readiness is heavily centered (kurtotic) around the mean.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
