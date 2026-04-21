# Experiment 14: node_4_0

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_0` |
| **ID in Run** | 14 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T21:55:35.560153+00:00 |
| **Runtime** | 167.5s |
| **Parent** | `node_3_0` |
| **Children** | `node_5_2`, `node_5_20` |
| **Creation Index** | 15 |

---

## Hypothesis

> The specific adversarial tactics associated with 'Integration Readiness' gaps
occur with significantly higher frequency across the incident corpus than
tactics associated with 'Trust Readiness' gaps.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.5652 (Maybe True) |
| **Posterior** | 0.1283 (Likely False) |
| **Surprise** | -0.5071 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 11.0 |
| Uncertain | 7.0 |
| Maybe False | 3.0 |
| Definitely False | 1.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 6.0 |
| Definitely False | 84.0 |

---

## Experiment Plan

**Objective:** Compare the prevalence of adversarial tactics between the two bundles.

### Steps
- 1. Load 'step3_tactic_frequency.csv'.
- 2. Group the data by 'bundle'.
- 3. Perform a t-test to compare the mean 'incident_count' of tactics belonging to 'Trust Readiness' vs. 'Integration Readiness'.

### Deliverables
- Mean frequency per tactic by bundle, T-test results, and a boxplot of tactic frequencies.

---

## Analysis

The experiment successfully analyzed `step3_tactic_frequency.csv` to compare the
frequency of adversarial tactics associated with 'Trust Readiness' versus
'Integration Readiness'.

**Descriptive Statistics:**
- **Integration Readiness:** Mean = 29.14, Std Dev = 26.59 (n=7). The high
standard deviation and the boxplot indicate extreme variability, with incident
counts ranging from ~2 to 77. This suggests that while some technical
integration gaps are heavily exploited (e.g., specific execution controls),
others are rarely targeted.
- **Trust Readiness:** Mean = 21.89, Std Dev = 11.82 (n=9). The distribution is
much more compact, indicating a consistent, moderate frequency of exploitation
for governance-related gaps.

**Statistical Inference:**
The Welch's t-test yielded a t-statistic of 0.6720 and a p-value of 0.5209.
Since p > 0.05, the observed difference in means (29.14 vs 21.89) is not
statistically significant.

**Conclusion:**
The hypothesis is **rejected**. Although the mean frequency for Integration
tactics is numerically higher, the difference is not statistically significant
due to the extreme variance within the Integration group and the small sample
size. The data characterizes Integration risks as 'high-variance' (boom or bust)
compared to the 'consistent' risk profile of Trust gaps.

---

## Review

The experiment successfully analyzed the 'step3_tactic_frequency.csv' dataset to
compare the prevalence of adversarial tactics between 'Trust Readiness' and
'Integration Readiness' bundles.

**Key Findings:**
1.  **Descriptive Statistics:** 'Integration Readiness' tactics (n=7) showed a
higher mean frequency (29.14) compared to 'Trust Readiness' tactics (n=9,
Mean=21.89). However, the Integration group exhibited extreme variance (Standard
Deviation = 26.59 vs 11.82 for Trust), with incident counts ranging from ~2 to
77.
2.  **Statistical Significance:** The Welch's t-test yielded a t-statistic of
0.6720 and a p-value of 0.5209. Since the p-value is greater than 0.05, the
observed difference in means is not statistically significant.
3.  **Visualization:** The boxplot confirms that 'Integration' risks are high-
variance (boom-or-bust), while 'Trust' risks are more consistent in their
frequency of exploitation.

**Conclusion:** The hypothesis that Integration tactics occur with
*significantly* higher frequency is **rejected**. While the raw mean is higher,
the statistical evidence does not support a significant difference due to the
high variability and small sample size.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Locate the dataset
filename = 'step3_tactic_frequency.csv'
paths_to_check = [filename, f"../{filename}"]
file_path = None

for path in paths_to_check:
    if os.path.exists(path):
        file_path = path
        break

if not file_path:
    raise FileNotFoundError(f"Could not find {filename} in current or parent directory.")

# 1. Load the dataset
df = pd.read_csv(file_path)
print(f"Successfully loaded {filename}")
print(df.head())

# 2. Group data by 'bundle'
trust_counts = df[df['bundle'] == 'Trust Readiness']['incident_count']
integration_counts = df[df['bundle'] == 'Integration Readiness']['incident_count']

# 3. Descriptive Statistics
mean_trust = trust_counts.mean()
mean_integration = integration_counts.mean()
std_trust = trust_counts.std()
std_integration = integration_counts.std()

print("\n=== Descriptive Statistics ===")
print(f"Trust Readiness:       n={len(trust_counts)}, Mean={mean_trust:.2f}, Std Dev={std_trust:.2f}")
print(f"Integration Readiness: n={len(integration_counts)}, Mean={mean_integration:.2f}, Std Dev={std_integration:.2f}")

# 4. Perform T-test (Welch's t-test for unequal variances)
t_stat, p_val = stats.ttest_ind(integration_counts, trust_counts, equal_var=False)

print("\n=== Statistical Test Results (Welch's t-test) ===")
print(f"Hypothesis: Integration > Trust")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("Result: Statistically Significant Difference")
else:
    print("Result: No Statistically Significant Difference")

# 5. Visualization
plt.figure(figsize=(8, 6))
plt.boxplot([trust_counts, integration_counts], labels=['Trust Readiness', 'Integration Readiness'])
plt.title('Adversarial Tactic Frequency by Competency Bundle')
plt.ylabel('Incident Frequency (Count)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Successfully loaded step3_tactic_frequency.csv
    tactic_id  ...                          competency_domain
0  AML.TA0003  ...           Resource & Supply Chain Controls
1  AML.TA0011  ...              Impact Containment & Recovery
2  AML.TA0004  ...  Access Boundary & Initial Access Controls
3  AML.TA0005  ...            Execution Controls & Sandboxing
4  AML.TA0007  ...                  Defense Evasion Detection

[5 rows x 6 columns]

=== Descriptive Statistics ===
Trust Readiness:       n=9, Mean=21.89, Std Dev=11.82
Integration Readiness: n=7, Mean=29.14, Std Dev=26.59

=== Statistical Test Results (Welch's t-test) ===
Hypothesis: Integration > Trust
T-statistic: 0.6720
P-value: 0.5209
Result: No Statistically Significant Difference

STDERR:
<ipython-input-1-776ab0f4a8aa>:57: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot([trust_counts, integration_counts], labels=['Trust Readiness', 'Integration Readiness'])


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (also known as a Box-and-Whisker Plot).
*   **Purpose:** This plot visualizes the distribution of numerical data ("Incident Frequency") across two different categorical groups ("Trust Readiness" and "Integration Readiness"). It displays the median, quartiles, variability (spread), and potential outliers for each group.

### 2. Axes
*   **X-Axis:**
    *   **Label/Categories:** Represents "Competency Bundles" with two specific categories: **"Trust Readiness"** and **"Integration Readiness"**.
*   **Y-Axis:**
    *   **Title:** **"Incident Frequency (Count)"**.
    *   **Units:** Count (integer values representing the number of incidents).
    *   **Range:** The axis is marked from **0 to 80** with increments of 10. The visible data spans from approximately 2 to 77.

### 3. Data Trends
*   **Trust Readiness (Left Box):**
    *   **Distribution:** The data is relatively compact. The interquartile range (the box itself) spans from approximately **15 to 24**.
    *   **Median:** The median line (orange) is situated around **17 or 18**.
    *   **Range:** The whiskers extend from roughly **11 to 34**.
    *   **Outliers:** There is a distinct, single outlier depicted as a circle at a frequency of approximately **47**.

*   **Integration Readiness (Right Box):**
    *   **Distribution:** The data is much more spread out compared to Trust Readiness. The interquartile range is tall, spanning from approximately **7 to 39**.
    *   **Median:** The median is significantly higher than the other group, sitting at approximately **33**.
    *   **Range:** The whiskers cover a very wide range, extending from a minimum of roughly **2** to a maximum of roughly **77**.
    *   **Outliers:** No distinct outliers are marked for this category; the variability is inherently high.

### 4. Annotations and Legends
*   **Title:** "Adversarial Tactic Frequency by Competency Bundle" clearly defines the context of the data.
*   **Grid Lines:** Horizontal dashed grid lines are provided at intervals of 10 to assist in estimating the Y-axis values.
*   **Box Components:**
    *   **Orange Line:** Represents the median value.
    *   **Box Edges:** Represent the 25th percentile (bottom) and 75th percentile (top).
    *   **Whiskers:** Indicate the range of the data outside the middle 50% (typically 1.5x IQR).
    *   **Circle:** Represents an outlier point.

### 5. Statistical Insights
*   **Higher Frequency in Integration Readiness:** On average (median), adversarial tactics appear more frequently in the "Integration Readiness" bundle (median ~33) compared to "Trust Readiness" (median ~17).
*   **Variability Differences:** "Integration Readiness" shows significantly higher volatility. The incident counts for this category can be very low (near 0) or extremely high (near 80). In contrast, "Trust Readiness" is much more consistent and predictable, with most counts clustering between 15 and 25.
*   **Maximum Impact:** The worst-case scenarios (highest counts) are found in the "Integration Readiness" group, where frequencies reach up to ~77. The "Trust Readiness" group tops out around 34 (excluding the outlier).
*   **Anomalies:** While "Trust Readiness" is generally consistent, the outlier at 47 suggests that there are rare events where the tactic frequency spikes significantly above the norm for that category.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
