# Experiment 9: node_3_0

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_0` |
| **ID in Run** | 9 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T21:49:01.962808+00:00 |
| **Runtime** | 188.3s |
| **Parent** | `node_2_5` |
| **Children** | `node_4_0`, `node_4_7`, `node_4_10` |
| **Creation Index** | 10 |

---

## Hypothesis

> Competency statements with 'High' evidence confidence map to a significantly
higher number of architecture controls than those with 'Medium' or 'Low'
confidence, reflecting a maturity bias in the framework.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6875 (Maybe True) |
| **Posterior** | 0.1447 (Likely False) |
| **Surprise** | -0.6299 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 18.0 |
| Uncertain | 5.0 |
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

**Objective:** Explore the relationship between the strength of literature evidence and the architectural complexity of the resulting competency.

### Steps
- 1. Load 'step2_competency_statements.csv'.
- 2. Create a 'control_count' variable by counting the number of semicolon-separated items in 'applicable_controls'.
- 3. Group the data by 'confidence' level ('High' vs. 'Medium'/'Low').
- 4. Perform a one-way ANOVA or t-test to compare the mean 'control_count' across confidence levels.

### Deliverables
- Mean control counts by confidence level, ANOVA/T-test results, and a violin plot showing the distribution of control counts for each confidence level.

---

## Analysis

The experiment successfully analyzed the relationship between evidence
confidence and architectural complexity.

**Descriptive Statistics:**
- **High Confidence:** n=36, Mean=1.86 controls, Std Dev=0.64.
- **Medium/Low Confidence:** n=6, Mean=1.83 controls, Std Dev=0.75.

**Statistical Testing:**
- **Method:** Welch's t-test (unequal variances assumed).
- **Result:** t-statistic = 0.0854, p-value = 0.9346.
- **Conclusion:** The difference in means (0.028) is negligible and not
statistically significant (p > 0.05).

**Visualization:**
The violin plot confirms that both groups have a mode of roughly 2 controls. The
'High' confidence group shows a tighter distribution (values 1-3), while the
'Medium/Low' group has a slightly wider spread but a very small sample size,
making the kernel density estimation less reliable.

**Hypothesis Evaluation:**
The hypothesis that 'High' confidence statements map to significantly more
controls is **rejected**. The data suggests that the complexity of the
architecture controls is independent of the confidence level of the evidence
supporting the competency.

---

## Review

The experiment was faithfully executed and the hypothesis was tested
sufficiently. The analysis of 'step2_competency_statements.csv' revealed that
'High' confidence competencies (n=36) map to an average of 1.86 controls, while
'Medium/Low' confidence competencies (n=6) map to an average of 1.83 controls.
The Welch's t-test resulted in a p-value of 0.9346, indicating no statistically
significant difference between the groups. Consequently, the hypothesis that
high-confidence statements require significantly more architecture controls is
rejected; the data suggests that architectural complexity is independent of the
evidence confidence level.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import sys
import os

# 1. Load the dataset
# Check current directory first, then parent directory to be robust
file_name = 'step2_competency_statements.csv'
if os.path.exists(file_name):
    file_path = file_name
elif os.path.exists(f'../{file_name}'):
    file_path = f'../{file_name}'
else:
    print(f"Error: {file_name} not found.")
    sys.exit(1)

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}. Shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

# 2. Feature Engineering: Calculate control_count
# applicable_controls is semicolon-separated. Handle NaNs if any.
df['applicable_controls'] = df['applicable_controls'].fillna('')
df['control_count'] = df['applicable_controls'].apply(lambda x: len([c.strip() for c in x.split(';') if c.strip()]))

# Check unique confidence levels
print("\nUnique confidence levels:")
print(df['confidence'].value_counts())

# 3. Grouping: High vs Not High (Medium/Low)
# The hypothesis specifically contrasts High vs others.
df['confidence_group'] = df['confidence'].apply(lambda x: 'High' if str(x).lower() == 'high' else 'Medium/Low')

high_group = df[df['confidence_group'] == 'High']['control_count']
other_group = df[df['confidence_group'] == 'Medium/Low']['control_count']

# Descriptive Statistics
print("\n=== Descriptive Statistics by Confidence Group ===")
group_stats = df.groupby('confidence_group')['control_count'].describe()
print(group_stats)

# 4. Statistical Test (T-test)
# We assume unequal variances (Welch's t-test)
t_stat, p_val = stats.ttest_ind(high_group, other_group, equal_var=False)

print("\n=== Statistical Test Results (Welch's t-test) ===")
print(f"Comparison: High Confidence (n={len(high_group)}) vs Medium/Low Confidence (n={len(other_group)})")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("Result: Statistically Significant (Reject Null Hypothesis)")
else:
    print("Result: Not Statistically Significant (Fail to Reject Null Hypothesis)")

# 5. Visualization
plt.figure(figsize=(10, 6))
sns.violinplot(x='confidence_group', y='control_count', data=df, inner='box', palette='muted')
plt.title('Distribution of Architecture Control Counts by Evidence Confidence')
plt.xlabel('Evidence Confidence')
plt.ylabel('Number of Applicable Controls')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Overlay individual points
sns.stripplot(x='confidence_group', y='control_count', data=df, color='black', alpha=0.5, jitter=True)

plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Successfully loaded step2_competency_statements.csv. Shape: (42, 8)

Unique confidence levels:
confidence
high      36
medium     6
Name: count, dtype: int64

=== Descriptive Statistics by Confidence Group ===
                  count      mean       std  min   25%  50%  75%  max
confidence_group                                                     
High               36.0  1.861111  0.639320  1.0  1.00  2.0  2.0  3.0
Medium/Low          6.0  1.833333  0.752773  1.0  1.25  2.0  2.0  3.0

=== Statistical Test Results (Welch's t-test) ===
Comparison: High Confidence (n=36) vs Medium/Low Confidence (n=6)
T-statistic: 0.0854
P-value: 0.9346
Result: Not Statistically Significant (Fail to Reject Null Hypothesis)

STDERR:
<ipython-input-1-7f693195a10f>:67: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.violinplot(x='confidence_group', y='control_count', data=df, inner='box', palette='muted')


=== Plot Analysis (figure 1) ===
Based on the provided image, here is a detailed analysis of the plot:

### 1. Plot Type
*   **Type:** This is a **Violin Plot** overlaid with a **Box Plot** and a **Strip Plot** (scatter points).
*   **Purpose:** The violin plot visualizes the probability density of the data at different values, showing the distribution shape. The inner box plot summarizes statistical quartiles (median, interquartile range), and the individual scatter points show the raw data distribution, revealing sample size and clustering.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Evidence Confidence"
    *   **Labels:** Two categorical groups: "High" and "Medium/Low".
*   **Y-Axis:**
    *   **Title:** "Number of Applicable Controls"
    *   **Range:** The axis is labeled from integers 0 to 4.
    *   **Units:** Counts (discrete integer values).

### 3. Data Trends
*   **High Evidence Confidence (Left, Blue):**
    *   **Shape:** The distribution is multi-modal with distinct "bulges" at integer values 1, 2, and 3.
    *   **Density:** The widest section (highest density) is at **2 applicable controls**. There is also significant density at 1 and 3.
    *   **Range:** The data is tightly clustered between approximately 0.5 and 3.5, with no visible points at 0 or 4.
*   **Medium/Low Evidence Confidence (Right, Orange):**
    *   **Shape:** The shape is more elongated compared to the "High" category.
    *   **Density:** Similar to the High category, the highest density is around **2 applicable controls**.
    *   **Range:** The spread is wider, extending clearly from 0 up to 4. This indicates a higher variance in this category.
    *   **Points:** The individual data points are sparser than in the "High" category, suggesting a smaller sample size for this group.

### 4. Annotations and Legends
*   **Title:** "Distribution of Architecture Control Counts by Evidence Confidence" is displayed at the top.
*   **Grid:** Horizontal dashed grid lines are placed at integer intervals (0, 1, 2, 3, 4) to aid in reading the specific count values.
*   **Internal Markers:** Inside each violin:
    *   A thick grey vertical bar represents the Interquartile Range (IQR).
    *   Black dots represent individual data observations.

### 5. Statistical Insights
*   **Central Tendency:** For both "High" and "Medium/Low" confidence levels, the most common number of applicable controls is **2**.
*   **Variance/Spread:** The "Medium/Low" confidence group exhibits higher variance. While the "High" confidence group is strictly bound between 1 and 3 controls, the "Medium/Low" group includes outliers or edge cases reaching up to 4 controls.
*   **Discrete vs. Continuous Representation:** Although violin plots are typically used for continuous data (hence the smooth curves), the data here is clearly discrete (counts of 1, 2, 3, etc.). The "High" plot reflects this discreteness well with its "bumpy" shape, whereas the "Medium/Low" plot is smoothed out, likely due to the kernel density estimation smoothing over a smaller dataset spread over a wider range.
*   **Correlation:** There appears to be a relationship where higher evidence confidence correlates with a more predictable number of controls (tighter distribution), whereas lower confidence is associated with a wider range of potential control counts.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
