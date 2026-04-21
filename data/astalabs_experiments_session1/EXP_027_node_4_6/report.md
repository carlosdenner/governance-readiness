# Experiment 27: node_4_6

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_6` |
| **ID in Run** | 27 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:21:29.454615+00:00 |
| **Runtime** | 195.3s |
| **Parent** | `node_3_4` |
| **Children** | `node_5_1`, `node_5_7` |
| **Creation Index** | 28 |

---

## Hypothesis

> There is a significant positive correlation between the number of architecture
controls a requirement maps to (control density) and the evidence confidence of
the resulting competency statement.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6452 (Maybe True) |
| **Posterior** | 0.1653 (Likely False) |
| **Surprise** | -0.5569 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 23.0 |
| Uncertain | 2.0 |
| Maybe False | 5.0 |
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

**Objective:** Determine if highly specific, well-evidenced competencies drive more complex architectural requirements.

### Steps
- 1. Load 'step2_competency_statements.csv'.
- 2. Derive 'control_density' by counting the number of semicolon-separated items in the 'applicable_controls' column.
- 3. Map 'confidence' to a numeric scale (High=3, Medium=2, Low=1).
- 4. Calculate the Spearman rank correlation coefficient between 'control_density' and 'confidence'.

### Deliverables
- Scatter plot with jitter, Spearman correlation coefficient, and p-value.

---

## Analysis

The experiment successfully analyzed the relationship between architecture
control density and evidence confidence. The analysis of 42 competency
statements yielded a Spearman correlation coefficient of 0.0190 with a p-value
of 0.9051, indicating no statistically significant correlation between the
number of architecture controls and the confidence level of the competency
statement. The hypothesis is therefore rejected. The visualization and
contingency table reveal a 'ceiling effect' where the vast majority of
statements (36/42) have 'High' confidence regardless of whether they map to 1,
2, or 3 architecture controls. This suggests that the strength of evidence for a
governance requirement is independent of its architectural complexity.

---

## Review

The experiment was successfully executed after correcting the file path issue.
The analysis methodology (Spearman rank correlation) was appropriate for the
ordinal nature of the 'confidence' variable. The deliverables—including the
correlation coefficient, p-value, contingency table, and visualization—provide a
clear basis for evaluating the hypothesis.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np
import os

# Robust file loading strategy
filename = 'step2_competency_statements.csv'
file_path = filename # Default to current directory

if not os.path.exists(file_path):
    # Check parent directory if not found in current
    if os.path.exists(f'../{filename}'):
        file_path = f'../{filename}'
    else:
        # Fallback to current which will raise error, or print warning
        print(f"Warning: {filename} not found in current or parent directory. Attempting current.")

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# 1. Derive control_density
# The applicable_controls column contains semicolon-separated strings.
def count_controls(val):
    if pd.isna(val) or str(val).strip() == '':
        return 0
    return len([x for x in str(val).split(';') if x.strip()])

df['control_density'] = df['applicable_controls'].apply(count_controls)

# 2. Map confidence to numeric
# Normalize text to title case to handle potential inconsistencies (e.g. 'high', 'High')
confidence_map = {'High': 3, 'Medium': 2, 'Low': 1}
df['confidence_norm'] = df['confidence'].astype(str).str.strip().str.title()
df['confidence_numeric'] = df['confidence_norm'].map(confidence_map)

# Filter out any rows where confidence couldn't be mapped (if any)
df_clean = df.dropna(subset=['confidence_numeric'])

print(f"Analyzable records: {len(df_clean)}")

# 3. Calculate Spearman Correlation
# We use Spearman because 'confidence' is ordinal
corr, p_value = spearmanr(df_clean['control_density'], df_clean['confidence_numeric'])

print(f"\nSpearman Correlation Coefficient: {corr:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret results
alpha = 0.05
if p_value < alpha:
    print("Result: Statistically significant correlation.")
else:
    print("Result: No statistically significant correlation.")

# 4. Visualization
plt.figure(figsize=(10, 6))

# Add jitter for visualization purposes since data is discrete
jitter_x = np.random.uniform(-0.15, 0.15, size=len(df_clean))
jitter_y = np.random.uniform(-0.15, 0.15, size=len(df_clean))

plt.scatter(
    df_clean['control_density'] + jitter_x, 
    df_clean['confidence_numeric'] + jitter_y, 
    alpha=0.6, 
    c='teal', 
    s=100, 
    edgecolors='black'
)

plt.title(f'Control Density vs. Evidence Confidence\n(Spearman r={corr:.2f}, p={p_value:.3f})')
plt.xlabel('Control Density (Number of Architecture Controls)')
plt.ylabel('Evidence Confidence')
plt.yticks([1, 2, 3], ['Low (1)', 'Medium (2)', 'High (3)'])
plt.grid(True, linestyle='--', alpha=0.5)

# Add a trend line (linear regression fit) just for visual aid, even if correlation is non-parametric
sns.regplot(
    x=df_clean['control_density'], 
    y=df_clean['confidence_numeric'], 
    scatter=False, 
    color='darkred', 
    line_kws={'linestyle': ':'}
)

plt.tight_layout()
plt.show()

# Print detailed frequency table to help explain the result
print("\nContingency Table (Counts):")
crosstab = pd.crosstab(df_clean['confidence_norm'], df_clean['control_density'])
print(crosstab)
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step2_competency_statements.csv
Analyzable records: 42

Spearman Correlation Coefficient: 0.0190
P-value: 0.9051
Result: No statistically significant correlation.

Contingency Table (Counts):
control_density   1   2  3
confidence_norm           
High             10  21  5
Medium            2   3  1


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Scatter plot with a linear regression fit line and a confidence interval band.
*   **Purpose:** The plot is designed to examine the relationship (correlation) between two variables: "control_density" (independent variable) and "confidence_numeric" (dependent variable).

### 2. Axes
*   **X-Axis:**
    *   **Label:** `control_density`
    *   **Range:** The axis ticks are marked from **1.0 to 3.0**, though data points extend slightly beyond these markers (approximately 0.9 to 3.2).
    *   **Structure:** The data appears to be grouped around discrete values (integers 1, 2, and 3), suggesting the control density was likely measured in integer increments.
*   **Y-Axis:**
    *   **Label:** `confidence_numeric`
    *   **Range:** The axis is labeled with qualitative markers mapped to numeric values: **"Medium (2)"** and **"High (3)"**.
    *   **Scale:** The numeric scale runs vertically, with points mostly clustered between 2 and 3.5.

### 3. Data Trends
*   **Clustering:** The data points (teal circles) are distinctly clustered vertically around three x-values: roughly 1.0, 2.0, and 3.0.
*   **Distribution:**
    *   The vast majority of the data points are located at the top of the graph, clustered around the "High (3)" confidence level.
    *   There is a sparse scattering of lower values near or below "Medium (2)" at all three density levels (1, 2, and 3).
    *   There is one notable outlier at x ≈ 1.9 with a value significantly lower than "Medium (2)".
*   **Trend Line:** A dark red dotted trend line runs horizontally across the plot. It is effectively flat, indicating that as control density increases, the confidence numeric does not systematically increase or decrease.

### 4. Annotations and Legends
*   **Title:** "Control Density vs. Evidence Confidence"
*   **Subtitle (Statistical Annotation):** "(Spearman r=0.02, p=0.905)" – This provides the statistical summary of the correlation.
*   **Regression Elements:**
    *   **Dotted Red Line:** Represents the linear fit (regression line).
    *   **Pink Shaded Region:** Represents the confidence interval for the regression line. The band is wide, indicating uncertainty in the prediction, particularly because the data is not tightly grouped along a line.
*   **Data Points:** Represented as teal circles with dark outlines.

### 5. Statistical Insights
*   **No Correlation:** The Spearman correlation coefficient (**r=0.02**) is extremely close to zero. This indicates that there is **virtually no monotonic relationship** between control density and evidence confidence.
*   **Non-Significance:** The p-value (**p=0.905**) is significantly higher than the standard alpha level of 0.05. This means the result is **not statistically significant**; any slight trend observed is likely due to random chance.
*   **Visual Interpretation:** The flat regression line confirms the statistical metrics. Changing the control density (x-axis) has no predictive power over the confidence score (y-axis).
*   **Ceiling Effect:** The data shows a strong "ceiling effect," where most participants or samples reported "High" confidence regardless of the control density, which limits the variability necessary to find a strong correlation.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
