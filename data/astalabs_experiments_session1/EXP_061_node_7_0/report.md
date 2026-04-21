# Experiment 61: node_7_0

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_0` |
| **ID in Run** | 61 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:27:28.395663+00:00 |
| **Runtime** | 204.3s |
| **Parent** | `node_6_1` |
| **Children** | None |
| **Creation Index** | 62 |

---

## Hypothesis

> AI incidents resulting in 'Security' harms involve a significantly higher number
of adversarial techniques (technique_count) than incidents resulting in
'Privacy' or 'Reliability' harms, indicating that security breaches require more
complex kill chains.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.5887 (Maybe True) |
| **Posterior** | 0.1570 (Likely False) |
| **Surprise** | -0.5010 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 20.0 |
| Uncertain | 1.0 |
| Maybe False | 9.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 3.0 |
| Definitely False | 87.0 |

---

## Experiment Plan

**Objective:** Analyze the relationship between harm type and attack complexity.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Create a binary category: 'Security' (where harm_type == 'security') vs. 'Other' (all other harm types).
- 3. Extract the 'technique_count' for both groups.
- 4. Perform an Independent Samples T-test (or Mann-Whitney U test) to compare the mean technique counts.
- 5. Visualize the distributions.

### Deliverables
- Violin plot or boxplot of technique counts for Security vs. Other harms, T-test/Mann-Whitney results.

---

## Analysis

The experiment tested the hypothesis that AI security incidents involve more
complex kill chains (higher technique counts) than other harm types (privacy,
reliability, etc.). The analysis of 52 case studies revealed that 'Security'
incidents (n=36) had a mean technique count of 7.50 (Median=7.0), while 'Other'
incidents (n=16) had a slightly higher mean of 7.62 (Median=8.0). A Mann-Whitney
U test (one-sided) yielded a p-value of 0.745, failing to reject the null
hypothesis; thus, the data does not support the claim that security incidents
are significantly more complex. The visualization highlights that while the
Security group exhibits greater variance (range 1-16) with distinct high-
complexity outliers, the central tendency for attack complexity is remarkably
similar across all harm types.

---

## Review

The experiment successfully tested the hypothesis regarding attack complexity
(technique count) across different harm types. The analysis of 52 MITRE ATLAS
case studies revealed that 'Security' incidents (n=36) had a mean technique
count of 7.50 (Median=7.00), while 'Other' incidents (n=16) had a slightly
higher mean of 7.62 (Median=8.00). The Mann-Whitney U test yielded a p-value of
0.745 (one-sided), failing to reject the null hypothesis. Consequently, the data
does not support the claim that security breaches require more complex kill
chains than other harm types (e.g., privacy, reliability). The visualization
confirms that while security incidents show a wider range (1-16) and contain the
most complex outliers, the central tendency of attack complexity is uniform
across harm categories.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import sys
import numpy as np

# Robust file loading
filename = 'step3_incident_coding.csv'
file_path = None

if os.path.exists(filename):
    file_path = filename
elif os.path.exists(os.path.join('..', filename)):
    file_path = os.path.join('..', filename)
else:
    print(f"Error: {filename} not found in current or parent directory.")
    sys.exit(1)

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# Data Preparation
# Group: Security (harm_type == 'security') vs Other
security_counts = df[df['harm_type'] == 'security']['technique_count']
other_counts = df[df['harm_type'] != 'security']['technique_count']

# Descriptive Statistics
print("\n=== Descriptive Statistics (Technique Count) ===")
print(f"Security Incidents (n={len(security_counts)}):")
print(f"  Mean: {security_counts.mean():.2f}")
print(f"  Median: {security_counts.median():.2f}")
print(f"  Std Dev: {security_counts.std():.2f}")

print(f"\nOther Incidents (n={len(other_counts)}):")
print(f"  Mean: {other_counts.mean():.2f}")
print(f"  Median: {other_counts.median():.2f}")
print(f"  Std Dev: {other_counts.std():.2f}")

# Statistical Analysis: Mann-Whitney U Test
# Testing if Security > Other
u_stat, p_val = stats.mannwhitneyu(security_counts, other_counts, alternative='greater')

print("\n=== Mann-Whitney U Test Results ===")
print(f"U-statistic: {u_stat}")
print(f"P-value (one-sided, Security > Other): {p_val:.5f}")

if p_val < 0.05:
    print("Result: Statistically significant. Security incidents involve significantly more techniques.")
else:
    print("Result: Not statistically significant.")

# Visualization: Boxplot with Strip Plot overlay
plt.figure(figsize=(10, 6))
data = [security_counts, other_counts]
labels = [f'Security (n={len(security_counts)})', f'Other (n={len(other_counts)})']

# Boxplot
box = plt.boxplot(data, patch_artist=True, labels=labels, widths=0.6)

# Styling
colors = ['#ff9999', '#99ccff']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add individual data points (jittered)
for i, d in enumerate(data):
    y = d
    x = np.random.normal(1 + i, 0.04, size=len(y))
    plt.scatter(x, y, alpha=0.6, color='black', s=20, zorder=3)

plt.title('Complexity of Attack Chains: Security vs. Other Harm Types')
plt.ylabel('Technique Count (Kill Chain Length)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_incident_coding.csv

=== Descriptive Statistics (Technique Count) ===
Security Incidents (n=36):
  Mean: 7.50
  Median: 7.00
  Std Dev: 3.17

Other Incidents (n=16):
  Mean: 7.62
  Median: 8.00
  Std Dev: 2.42

=== Mann-Whitney U Test Results ===
U-statistic: 255.5
P-value (one-sided, Security > Other): 0.74516
Result: Not statistically significant.

STDERR:
<ipython-input-1-66ae2b9dbfe7>:62: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  box = plt.boxplot(data, patch_artist=True, labels=labels, widths=0.6)


=== Plot Analysis (figure 1) ===
Based on the provided image, here is a detailed analysis:

### 1. Plot Type
*   **Type:** This is a **Box Plot (Box-and-Whisker Plot)** overlaid with a **Strip Plot** (also known as a jitter or swarm plot).
*   **Purpose:** The plot is designed to compare the distributions of "Technique Count" (a measure of complexity) between two distinct categories ("Security" and "Other"). By overlaying the individual data points (the dots) on top of the box plots, it provides a comprehensive view of the data's central tendency (median), spread (interquartile range), and individual observations, preventing the concealment of underlying data structures that simple box plots might hide.

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Technique Count (Kill Chain Length)". This represents the number of steps or techniques involved in an attack chain.
    *   **Range:** The axis is marked from 2 to 16 with an interval of 2. The visible data spans from 1 (lowest point in Security) to 16 (highest point in Security).
    *   **Units:** Count (integer values).
*   **X-Axis:**
    *   **Labels:** Two categorical groups:
        1.  "Security (n=36)"
        2.  "Other (n=16)"
    *   **Meaning:** These labels denote the type of harm being analyzed, with "n=" indicating the sample size for each group.

### 3. Data Trends
*   **Security Group (Pink Box):**
    *   **Median:** The orange line is at **7**.
    *   **Interquartile Range (IQR):** The middle 50% of the data falls between **6 and 9**.
    *   **Range & Outliers:** The data has a wide spread. While the main whiskers extend from 3 to 12, there are distinct **outliers** at the high end (values of 14, 15, and 16) and one outlier at the low end (value of 1).
    *   **Cluster:** There is a dense cluster of data points around the median and lower quartile (values 6 and 7).

*   **Other Group (Blue Box):**
    *   **Median:** The orange line is at **8**, slightly higher than the Security group.
    *   **Interquartile Range (IQR):** The middle 50% appears to range from approximately **6.5 to 9**.
    *   **Range & Outliers:** The whiskers extend from 3 to 12. Unlike the Security group, there are **no visible outliers** beyond the whiskers. The data is more contained.

### 4. Annotations and Legends
*   **Title:** "Complexity of Attack Chains: Security vs. Other Harm Types". This sets the context of the analysis as a cybersecurity risk assessment.
*   **Sample Size (n):** The X-axis labels include the sample size (`n=36` for Security, `n=16` for Other), which is crucial for interpreting the statistical weight of the data.
*   **Gridlines:** Horizontal dashed gridlines appear every 2 units, aiding in the precise reading of Y-axis values.
*   **Color Coding:**
    *   **Pink/Red:** Represents "Security" harm types.
    *   **Blue:** Represents "Other" harm types.

### 5. Statistical Insights
*   **Similar Central Tendency:** Both groups have similar medians (7 vs. 8) and upper quartiles (9). This suggests that the "typical" complexity of an attack is comparable whether it is a security incident or another type of harm.
*   **Higher Peak Complexity in Security:** The key differentiator is the "long tail" of the Security group. The presence of outliers at technique counts of 14, 15, and 16 indicates that **extreme complexity is a characteristic specific to Security incidents** in this dataset. The "Other" category caps out at a complexity of 12.
*   **Variability:** The Security group shows greater variability (standard deviation), ranging from 1 to 16, whereas the "Other" group is more consistent, ranging only from 3 to 12.
*   **Sample Size Consideration:** The insights for the "Other" category are drawn from a smaller sample size (16) compared to "Security" (36), which means the distribution for "Other" might be less robustly defined.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
