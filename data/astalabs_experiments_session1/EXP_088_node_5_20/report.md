# Experiment 88: node_5_20

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_20` |
| **ID in Run** | 88 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:22:12.189604+00:00 |
| **Runtime** | 266.7s |
| **Parent** | `node_4_0` |
| **Children** | None |
| **Creation Index** | 89 |

---

## Hypothesis

> Sub-competencies within the 'Trust Readiness' bundle cover a significantly
broader range of distinct harm types per competency than 'Integration Readiness'
sub-competencies, which tend to address specific, narrower risk vectors.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7407 (Likely True) |
| **Posterior** | 0.2286 (Likely False) |
| **Surprise** | -0.5944 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 26.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 27.0 |
| Definitely False | 63.0 |

---

## Experiment Plan

**Objective:** Analyze the 'risk entropy' or breadth of scope for sub-competencies in each bundle by comparing the count of distinct harm types addressed.

### Steps
- 1. Load the dataset 'step3_coverage_map.csv'.
- 2. Create a new variable 'distinct_harm_count' by parsing the 'primary_harm_types' column. This column contains semicolon-separated values (e.g., 'security(16); privacy(2)'). Count the number of distinct items separated by semicolons for each row. If the field is empty or NaN, the count is 0.
- 3. Group the data by the 'bundle' column ('Trust Readiness' vs. 'Integration Readiness').
- 4. Calculate descriptive statistics (mean, standard deviation, count) for 'distinct_harm_count' for each bundle.
- 5. Perform a Welch's t-test (independent samples, unequal variance) to compare the mean distinct harm counts between the two bundles.
- 6. Generate a strip plot or box plot to visualize the distribution of harm counts for each bundle.

### Deliverables
- 1. Descriptive statistics table showing mean harm counts and standard deviations by bundle.
- 2. T-test results (t-statistic and p-value).
- 3. A visualization (strip plot or box plot) comparing the breadth of harm types.

---

## Analysis

The experiment successfully analyzed the 'risk entropy' of sub-competencies
using 'step3_coverage_map.csv'.

**Descriptive Statistics:**
- **Trust Readiness (n=8):** Mean = 1.50 distinct harm types (Std Dev = 1.20).
- **Integration Readiness (n=8):** Mean = 1.88 distinct harm types (Std Dev =
1.36).

**Statistical Inference:**
- **Method:** Welch's t-test.
- **Result:** t-statistic = -0.5867, p-value = 0.5669.
- **Conclusion:** The hypothesis is **rejected**. Contrary to the prediction
that Trust competencies would cover a broader range of harms, the data shows
that 'Integration Readiness' sub-competencies actually have a slightly higher
(though statistically insignificant) mean count of distinct harm types. The high
p-value indicates that there is no significant difference in the breadth of risk
scope between the two bundles; both are similarly varied in the number of harm
types they address.

**Visualization:**
The strip plot confirms substantial overlap between the two groups, with the
'Integration' mean line slightly above the 'Trust' mean line, visually
reinforcing the rejection of the hypothesis.

---

## Review

The experiment was faithfully executed. The code successfully loaded the
dataset, parsed the harm types, and performed the required statistical
comparison. The results contradict the hypothesis.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Define the file name
filename = 'step3_coverage_map.csv'

# Try to locate the file
file_path = None
possible_paths = [filename, os.path.join('..', filename)]

for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    print(f"Error: Could not find {filename} in current or parent directory.")
    print(f"Current working directory: {os.getcwd()}")
    sys.exit(1)

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# Function to count distinct harms
def count_harms(val):
    if pd.isna(val) or str(val).strip() == '':
        return 0
    # Split by semicolon
    parts = str(val).split(';')
    # Filter empty strings
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)

# Apply counting
df['distinct_harm_count'] = df['primary_harm_types'].apply(count_harms)

# Filter for relevant bundles
valid_bundles = ['Trust Readiness', 'Integration Readiness']
df_filtered = df[df['bundle'].isin(valid_bundles)].copy()

# Group data
trust_data = df_filtered[df_filtered['bundle'] == 'Trust Readiness']['distinct_harm_count']
integration_data = df_filtered[df_filtered['bundle'] == 'Integration Readiness']['distinct_harm_count']

# Descriptive Statistics
print("\n=== Descriptive Statistics: Distinct Harm Count by Bundle ===")
print(f"Trust Readiness (n={len(trust_data)}): Mean={trust_data.mean():.2f}, Std={trust_data.std():.2f}")
print(f"Integration Readiness (n={len(integration_data)}): Mean={integration_data.mean():.2f}, Std={integration_data.std():.2f}")

# T-Test (Welch's)
t_stat, p_val = stats.ttest_ind(trust_data, integration_data, equal_var=False)
print("\n=== Welch's T-Test Results ===")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

# Visualization
plt.figure(figsize=(8, 6))

# Jitter for strip plot
jitter_trust = np.random.normal(0, 0.05, size=len(trust_data))
jitter_integ = np.random.normal(0, 0.05, size=len(integration_data))

plt.scatter(np.zeros_like(trust_data) + jitter_trust, trust_data, 
            label='Trust Readiness', alpha=0.7, s=100, color='skyblue', edgecolors='black')
plt.scatter(np.ones_like(integration_data) + jitter_integ, integration_data, 
            label='Integration Readiness', alpha=0.7, s=100, color='salmon', edgecolors='black')

# Plot Means
plt.plot([-0.2, 0.2], [trust_data.mean(), trust_data.mean()], color='blue', lw=3, linestyle='--', label='Mean Trust')
plt.plot([0.8, 1.2], [integration_data.mean(), integration_data.mean()], color='red', lw=3, linestyle='--', label='Mean Integration')

plt.xticks([0, 1], ['Trust Readiness', 'Integration Readiness'])
plt.ylabel('Count of Distinct Harm Types')
plt.title('Risk Scope: Distinct Harms per Sub-Competency')
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.6)

plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_coverage_map.csv

=== Descriptive Statistics: Distinct Harm Count by Bundle ===
Trust Readiness (n=8): Mean=1.50, Std=1.20
Integration Readiness (n=8): Mean=1.88, Std=1.36

=== Welch's T-Test Results ===
T-statistic: -0.5867
P-value: 0.5669


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Categorical Scatter Plot (also known as a Strip Plot or Jitter Plot) with overlaying mean lines.
*   **Purpose:** The plot compares the distribution of a count variable ("Distinct Harm Types") across two different categories ("Trust Readiness" and "Integration Readiness"). The jittering (horizontal displacement of dots) prevents data points with the same value from overlapping completely, allowing the viewer to estimate the density of data at each value.

### 2. Axes
*   **X-Axis:**
    *   **Title/Labels:** Categorical labels representing two groups: **"Trust Readiness"** and **"Integration Readiness"**.
    *   **Range:** Two discrete categories.
*   **Y-Axis:**
    *   **Title:** **"Count of Distinct Harm Types"**.
    *   **Units:** Integer counts (distinct types of harm).
    *   **Range:** The axis is marked from **0.0 to 3.0**, with grid lines at 0.5 intervals. The data points fall strictly on integer values (0, 1, 2, 3).

### 3. Data Trends
*   **Trust Readiness (Blue Circles):**
    *   **Distribution:** The data points appear very evenly distributed across the entire range. There are pairs of data points visible at each integer level: 0, 1, 2, and 3.
    *   **Pattern:** This suggests a uniform distribution where low, medium, and high counts of distinct harms are equally likely within this sample.
*   **Integration Readiness (Salmon Circles):**
    *   **Distribution:** The data appears less uniform than the Trust group. There is a cluster of points at the maximum value (3) and a cluster at the minimum value (0), with fewer points in the middle (values 1 and 2).
    *   **Pattern:** This suggests a somewhat bimodal distribution where "Integration Readiness" scenarios tend to result in either a high number of distinct harms or none at all, rather than a moderate amount.

### 4. Annotations and Legends
*   **Plot Title:** "Risk Scope: Distinct Harms per Sub-Competency".
*   **Legend:** Located on the right side, defining four elements:
    *   **Light Blue Circle:** Represents individual data points for "Trust Readiness".
    *   **Salmon Circle:** Represents individual data points for "Integration Readiness".
    *   **Blue Dashed Line:** Represents the "Mean Trust" score.
    *   **Red Dashed Line:** Represents the "Mean Integration" score.
*   **Grid:** Horizontal dotted lines aid in reading the specific Y-values of the points.

### 5. Statistical Insights
*   **Comparison of Means:**
    *   The **Mean Trust** (blue dashed line) sits exactly at **1.5**. This is consistent with the visual observation of an evenly distributed dataset across 0, 1, 2, and 3.
    *   The **Mean Integration** (red dashed line) is higher, sitting just below **2.0** (approximately 1.9).
*   **Risk Assessment:**
    *   **Higher Average Risk:** "Integration Readiness" has a higher average count of distinct harm types compared to "Trust Readiness," indicating it may be a riskier area on average regarding the scope of harms.
    *   **Consistency vs. Volatility:** "Trust Readiness" shows consistent variability (uniform spread). "Integration Readiness" shows a tendency toward extremes—mostly clustering at the highest harm count (3), despite having several zero-harm instances pulling the average down.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
