# Experiment 85: node_5_18

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_18` |
| **ID in Run** | 85 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:17:34.791507+00:00 |
| **Runtime** | 275.2s |
| **Parent** | `node_4_22` |
| **Children** | None |
| **Creation Index** | 86 |

---

## Hypothesis

> Integration Readiness sub-competencies exhibit significantly higher variance in
incident coverage than Trust Readiness sub-competencies, indicating that
engineering risks cluster around specific 'hotspots' rather than being uniformly
distributed.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6944 (Maybe True) |
| **Posterior** | 0.0631 (Definitely False) |
| **Surprise** | -0.7327 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 7.0 |
| Uncertain | 1.0 |
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

**Objective:** Compare the variance of incident counts between 'Trust Readiness' and 'Integration Readiness' bundles to assess risk concentration.

### Steps
- 1. Load the dataset 'step3_coverage_map.csv' from the current directory.
- 2. Filter the data into two groups based on the 'bundle' column: 'Trust Readiness' and 'Integration Readiness'.
- 3. Extract the 'incident_count' variable for each group.
- 4. Calculate the variance and standard deviation for both groups.
- 5. Perform Levene's Test (using center='median') to test the null hypothesis that the variances are equal.
- 6. Print the group variances, Levene's test statistic, and the p-value.

### Deliverables
- Calculated variances for both bundles, Levene's test statistic, and the resulting p-value.

---

## Analysis

The experiment successfully loaded the 'step3_coverage_map.csv' dataset and
compared the variance of incident counts between 'Trust Readiness' and
'Integration Readiness' sub-competencies.

Key Findings:
- **Descriptive Statistics:** 'Trust Readiness' (n=8) exhibited a variance of
42.84 (SD=6.55), while 'Integration Readiness' (n=8) showed a slightly lower
variance of 37.71 (SD=6.14). The means were nearly identical (5.62 vs 5.50).
- **Hypothesis Test:** Levene's Test for equality of variances (using the robust
median center) yielded a statistic of 0.0037 and a p-value of 0.9524.
- **Conclusion:** The p-value (0.9524) is far above the significance level of
0.05, meaning we fail to reject the null hypothesis. There is no statistically
significant difference in the variance of incident coverage between the two
bundles. This rejects the hypothesis that engineering risks (Integration)
cluster around specific 'hotspots' more than governance risks (Trust); both
domains demonstrate similar variability in how incidents map to their sub-
competencies.

---

## Review

The experiment was faithfully executed according to the plan. The code
successfully loaded the dataset, calculated the variances for both bundles, and
performed Levene's test to check for equality of variances. The results
(p=0.9524) clearly indicate that there is no significant difference in variance
between the two groups, leading to the rejection of the hypothesis.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
import os

# [debug] Check current directory and file existence
print(f"Current working directory: {os.getcwd()}")
file_name = 'step3_coverage_map.csv'
if os.path.exists(file_name):
    file_path = file_name
elif os.path.exists(f'../{file_name}'):
    file_path = f'../{file_name}'
else:
    print(f"Error: {file_name} not found in current or parent directory.")
    sys.exit(1)

# 1. Load the dataset
try:
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully from {file_path}.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# 2. Group by 'bundle' and extract 'incident_count'
trust_counts = df[df['bundle'] == 'Trust Readiness']['incident_count']
integration_counts = df[df['bundle'] == 'Integration Readiness']['incident_count']

# 3. Calculate variances and descriptive stats
var_trust = trust_counts.var(ddof=1)
var_integration = integration_counts.var(ddof=1)
mean_trust = trust_counts.mean()
mean_integration = integration_counts.mean()
std_trust = trust_counts.std(ddof=1)
std_integration = integration_counts.std(ddof=1)

print("\n--- Descriptive Statistics ---")
print(f"Trust Readiness:       n={len(trust_counts)}, Mean={mean_trust:.2f}, Variance={var_trust:.2f}, Std Dev={std_trust:.2f}")
print(f"Integration Readiness: n={len(integration_counts)}, Mean={mean_integration:.2f}, Variance={var_integration:.2f}, Std Dev={std_integration:.2f}")

# 4. Perform Levene's Test for equality of variances
# center='median' is robust (Brown-Forsythe)
stat, p_value = stats.levene(trust_counts, integration_counts, center='median')

print("\n--- Levene's Test for Equality of Variances (center='median') ---")
print(f"Statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Result: Variances are significantly different (Reject H0).")
else:
    print("Result: Variances are not significantly different (Fail to reject H0).")

# 5. Visualization
plt.figure(figsize=(8, 6))
data_to_plot = [trust_counts, integration_counts]
labels = ['Trust Readiness', 'Integration Readiness']

# Create boxplot
plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', alpha=0.5), 
            medianprops=dict(color='black'))

# Add individual points (jittered x-coordinates for visibility)
import numpy as np
np.random.seed(42)
jitter_trust = np.random.normal(1, 0.04, size=len(trust_counts))
jitter_integration = np.random.normal(2, 0.04, size=len(integration_counts))

plt.scatter(jitter_trust, trust_counts, color='blue', alpha=0.6, label='Trust Data')
plt.scatter(jitter_integration, integration_counts, color='red', alpha=0.6, label='Integration Data')

plt.title('Distribution of Incident Counts by Bundle')
plt.ylabel('Incident Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Current working directory: /data
Dataset loaded successfully from step3_coverage_map.csv.

--- Descriptive Statistics ---
Trust Readiness:       n=8, Mean=5.62, Variance=42.84, Std Dev=6.55
Integration Readiness: n=8, Mean=5.50, Variance=37.71, Std Dev=6.14

--- Levene's Test for Equality of Variances (center='median') ---
Statistic: 0.0037
P-value: 0.9524
Result: Variances are not significantly different (Fail to reject H0).

STDERR:
<ipython-input-1-e73028e043a2>:65: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data_to_plot, labels=labels, patch_artist=True,


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **Box Plot with an overlaid Scatter Plot** (often referred to as a box-and-whisker plot with jitter or strip plot overlay).
*   **Purpose:** It is used to visualize the distribution of "Incident Counts" across two different categories ("Trust Readiness" and "Integration Readiness"). The box plot component summarizes the distribution (median, quartiles, range), while the overlaid points show the specific values of individual data points, allowing the viewer to see the sample size and density of the data.

### 2. Axes
*   **X-Axis:**
    *   **Label/Title:** Represents categorical groups: **"Trust Readiness"** and **"Integration Readiness"**.
    *   **Value Range:** Two distinct categories.
*   **Y-Axis:**
    *   **Label/Title:** **"Incident Count"**.
    *   **Units:** Count (numerical frequency).
    *   **Value Range:** The axis ticks range from **0 to 17.5**, with grid lines every 2.5 units. The actual data extends slightly higher, to approximately 19.

### 3. Data Trends
*   **Trust Readiness (Left Group):**
    *   **Distribution:** The data is right-skewed.
    *   **Clusters:** There is a cluster of data points at the very bottom (0 to 2 incidents), indicating many bundles have few or no incidents.
    *   **Box Dimensions:** The Interquartile Range (IQR)—represented by the light blue box—spans roughly from 1 to 9. The median (black line inside the box) is approximately 3.5.
    *   **Outliers:** There is a significant outlier or maximum value near **19**.
*   **Integration Readiness (Right Group):**
    *   **Distribution:** Also right-skewed but with a slightly different clustering pattern than the Trust group.
    *   **Clusters:** Data points cluster near 0-1 and another cluster appears around 7-8.
    *   **Box Dimensions:** The IQR spans roughly from 1 to 8. The median is slightly higher than the Trust group, sitting at approximately **4.5**.
    *   **Outliers:** There is a significant maximum value near **18**.

### 4. Annotations and Legends
*   **Legend:** Located at the bottom center of the chart.
    *   **Blue Circle:** Represents **"Trust Data"** (corresponding to the Trust Readiness column).
    *   **Red Circle:** Represents **"Integration Data"** (corresponding to the Integration Readiness column).
*   **Grid Lines:** Horizontal dashed grey lines facilitate reading specific values on the Y-axis.
*   **Whiskers:** The black vertical lines extending from the boxes indicate the full range of the data (excluding potential statistical outliers, though here they appear to reach the maximum values).

### 5. Statistical Insights
*   **High Variability:** Both bundles show a high degree of variability. While the median incident count is low (under 5 for both), the "worst-case" scenarios result in incident counts nearly four times the median (18-19).
*   **Similarity in Performance:** The two categories perform relatively similarly. Their ranges are almost identical (0-19 vs 0-18), and their interquartile ranges (the middle 50% of the data) overlap significantly.
*   **Skewed Risk:** The distribution suggests that for both Trust and Integration bundles, the majority of cases run smoothly (low incident counts), but there is a persistent risk of high-incident events, as evidenced by the long upper whiskers and the high data points at the top of the graph.
*   **Median Comparison:** "Integration Readiness" has a slightly higher median incident count (~4.5) compared to "Trust Readiness" (~3.5), suggesting that, on average, the Integration bundles might experience slightly more incidents, though the sample size appears small.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
