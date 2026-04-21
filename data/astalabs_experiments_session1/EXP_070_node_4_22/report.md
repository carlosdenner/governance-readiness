# Experiment 70: node_4_22

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_22` |
| **ID in Run** | 70 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:45:35.526487+00:00 |
| **Runtime** | 176.2s |
| **Parent** | `node_3_17` |
| **Children** | `node_5_14`, `node_5_18` |
| **Creation Index** | 71 |

---

## Hypothesis

> 'Trust Readiness' sub-competencies have a significantly higher mean incident
coverage than 'Integration Readiness' sub-competencies, challenging the
assumption that AI security is primarily a technical engineering problem.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7661 (Likely True) |
| **Posterior** | 0.1963 (Likely False) |
| **Surprise** | -0.6614 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 3.0 |
| Maybe True | 27.0 |
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

**Objective:** Compare the mean incident counts of sub-competencies between the two bundles.

### Steps
- 1. Load 'step3_coverage_map.csv'.
- 2. Group by 'bundle' ('Trust Readiness' vs 'Integration Readiness').
- 3. Extract 'incident_count' for each sub-competency in the groups.
- 4. Perform an Independent Samples T-test (or Mann-Whitney if non-normal) to compare the mean incident counts.

### Deliverables
- Group means, Test statistic, p-value.

---

## Analysis

The experiment successfully tested the hypothesis that 'Trust Readiness' sub-
competencies have higher incident coverage than 'Integration Readiness' sub-
competencies. The code loaded 'step3_coverage_map.csv', grouped the 16 sub-
competencies by bundle (n=8 per group), and performed descriptive and
inferential statistics.

Key Findings:
1. Descriptive Statistics: The mean incident count for 'Trust Readiness' (Mean =
5.62, SD = 6.55) was virtually identical to 'Integration Readiness' (Mean =
5.50, SD = 6.14).
2. Normality & Test Selection: Both groups passed the Shapiro-Wilk normality
test (p > 0.05), leading to the use of Welch's Independent Samples T-test.
3. Hypothesis Test Results: The t-test yielded a statistic of 0.0394 and a
p-value of 0.9691. This p-value is far above the alpha threshold of 0.05.

Conclusion: The hypothesis is rejected. There is no statistically significant
difference in incident coverage between the two bundles. The data indicates that
real-world AI incidents map evenly to both governance (Trust) and engineering
(Integration) competencies, suggesting that AI security failures are just as
likely to stem from architectural gaps as from policy/governance gaps.

---

## Review

The experiment was successfully executed and the code faithfully implemented the
analysis plan. The dataset 'step3_coverage_map.csv' was loaded, and the 16 sub-
competencies were correctly grouped by bundle (n=8 per group).

**Findings:**
1.  **Descriptive Statistics:** The mean incident count for 'Trust Readiness'
sub-competencies (Mean = 5.62, SD = 6.55) was virtually identical to that of
'Integration Readiness' sub-competencies (Mean = 5.50, SD = 6.14).
2.  **Hypothesis Test:** A Welch's Independent Samples T-test was performed
after verifying normality (Shapiro-Wilk p > 0.05). The test yielded a p-value of
0.9691, which is far above the significance threshold (alpha = 0.05).
3.  **Conclusion:** The hypothesis that 'Trust Readiness' has significantly
higher incident coverage is **rejected**. The data suggests that real-world AI
incidents map evenly across both governance (Trust) and engineering
(Integration) domains, indicating that AI security is a dual-domain problem
rather than purely a governance or engineering one.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# Define file path (try parent directory first, then current)
filename = 'step3_coverage_map.csv'
file_path = f"../{filename}"
if not os.path.exists(file_path):
    file_path = filename

print(f"Loading dataset from: {file_path}")

try:
    df = pd.read_csv(file_path)
    
    # filter for specific bundles
    trust_data = df[df['bundle'] == 'Trust Readiness']['incident_count']
    integration_data = df[df['bundle'] == 'Integration Readiness']['incident_count']
    
    print("\n--- Descriptive Statistics ---")
    print(f"Trust Readiness: n={len(trust_data)}, Mean={trust_data.mean():.2f}, Std={trust_data.std():.2f}")
    print(f"Integration Readiness: n={len(integration_data)}, Mean={integration_data.mean():.2f}, Std={integration_data.std():.2f}")
    
    # Check for normality to decide on test
    print("\n--- Normality Check (Shapiro-Wilk) ---")
    shapiro_trust = stats.shapiro(trust_data)
    shapiro_integration = stats.shapiro(integration_data)
    print(f"Trust: W={shapiro_trust.statistic:.4f}, p={shapiro_trust.pvalue:.4f}")
    print(f"Integration: W={shapiro_integration.statistic:.4f}, p={shapiro_integration.pvalue:.4f}")
    
    alpha = 0.05
    if shapiro_trust.pvalue > alpha and shapiro_integration.pvalue > alpha:
        print("\nData appears normal. Performing Independent Samples T-test (Welch's).")
        stat, p_value = stats.ttest_ind(trust_data, integration_data, equal_var=False)
        test_type = "Welch's T-test"
    else:
        print("\nData deviates from normality. Performing Mann-Whitney U test.")
        stat, p_value = stats.mannwhitneyu(trust_data, integration_data, alternative='two-sided')
        test_type = "Mann-Whitney U Test"
        
    print(f"\n--- Hypothesis Test Results ({test_type}) ---")
    print(f"Statistic: {stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Result: Statistically Significant Difference")
    else:
        print("Result: No Statistically Significant Difference")
        
    # Visualization
    plt.figure(figsize=(10, 6))
    # Create a list of data to plot
    data_to_plot = [trust_data, integration_data]
    
    # Create the boxplot
    box = plt.boxplot(data_to_plot, patch_artist=True, labels=['Trust Readiness', 'Integration Readiness'])
    
    # Colors
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        
    # Add individual points (jittered)
    for i, data in enumerate(data_to_plot):
        y = data
        x = np.random.normal(1 + i, 0.04, size=len(y))
        plt.plot(x, y, 'r.', alpha=0.5)

    plt.title('Incident Coverage Distribution by Competency Bundle')
    plt.ylabel('Incident Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_coverage_map.csv

--- Descriptive Statistics ---
Trust Readiness: n=8, Mean=5.62, Std=6.55
Integration Readiness: n=8, Mean=5.50, Std=6.14

--- Normality Check (Shapiro-Wilk) ---
Trust: W=0.8445, p=0.0837
Integration: W=0.8443, p=0.0834

Data appears normal. Performing Independent Samples T-test (Welch's).

--- Hypothesis Test Results (Welch's T-test) ---
Statistic: 0.0394
P-value: 0.9691
Result: No Statistically Significant Difference

STDERR:
<ipython-input-1-0d5c43733226>:61: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  box = plt.boxplot(data_to_plot, patch_artist=True, labels=['Trust Readiness', 'Integration Readiness'])


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Box Plot with Overlaid Strip Plot (also known as a Box-and-Whisker plot with jittered data points).
*   **Purpose:** The plot compares the distribution of "Incident Counts" across two different categories ("Trust Readiness" and "Integration Readiness"). The box plot summarizes statistical distribution (median, quartiles, range), while the red dots (strip plot) reveal the individual data points and sample size.

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Incident Count".
    *   **Units:** Count (numerical integer values).
    *   **Range:** The axis ticks range from 0 to 17.5, but the visual data extends from 0 to approximately 19.
*   **X-Axis:**
    *   **Label:** Not explicitly labeled, but represents "Competency Bundle" categories.
    *   **Categories:** Two specific groups are displayed:
        1.  **Trust Readiness**
        2.  **Integration Readiness**

### 3. Data Trends
*   **Trust Readiness (Left, Blue Box):**
    *   **Median:** The median line (orange) is positioned roughly at 3.5.
    *   **Spread (IQR):** The main concentration of data (the box) ranges from approximately 1 to 9.
    *   **Range:** The whiskers extend from 0 (minimum) to 19 (maximum).
    *   **Cluster/Distribution:** The red data points show a cluster of low values (between 0 and 2) and a sparse spread for higher values, with a significant maximum value at 19.
*   **Integration Readiness (Right, Green Box):**
    *   **Median:** The median line is slightly higher than the Trust group, positioned roughly at 4.5.
    *   **Spread (IQR):** The interquartile range is slightly tighter than the Trust group, roughly from 1 to 8.
    *   **Range:** The whiskers extend from 0 (minimum) to 18 (maximum).
    *   **Cluster/Distribution:** Similar to the Trust group, there is a cluster of values near zero, a small cluster around 7-8, and a single high extreme value at 18.

### 4. Annotations and Legends
*   **Title:** "Incident Coverage Distribution by Competency Bundle" displayed at the top.
*   **Data Points:** Small red dots represent individual observations, overlaid to show the actual density of data which is sparse (approx. 6-8 data points per category).
*   **Grid Lines:** Horizontal dashed grey lines are provided at intervals of 2.5 to assist in reading the Y-axis values.
*   **Colors:** Light blue is used for "Trust Readiness" and light green for "Integration Readiness" to distinguish the categories visually.

### 5. Statistical Insights
*   **Skewed Distribution:** Both categories exhibit a strong positive skew (right-skewed). The majority of the data points are clustered at the lower end of the scale (below 10), but the mean is pulled upward by single high-value incidents (18 and 19) in both groups.
*   **Similarity of Bundles:** There is no statistically significant visual difference between the two bundles. Their ranges (0–19 vs. 0–18), medians (~3.5 vs. ~4.5), and lower quartiles are nearly identical.
*   **Variability:** Both groups show high variability relative to their medians. For example, while the typical (median) incident count is low (under 5), it is possible to see counts nearly four times that amount in both competency bundles.
*   **Small Sample Size:** The overlaid red dots reveal that the sample size for this dataset is quite small (likely less than 10 samples per category). This suggests that while the box plot provides a summary, the statistical robustness of the comparison might be limited by the low n-count.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
