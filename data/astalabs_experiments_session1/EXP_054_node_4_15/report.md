# Experiment 54: node_4_15

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_15` |
| **ID in Run** | 54 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:13:31.915620+00:00 |
| **Runtime** | 208.0s |
| **Parent** | `node_3_18` |
| **Children** | `node_5_9`, `node_5_16` |
| **Creation Index** | 55 |

---

## Hypothesis

> Incidents categorized as 'Security' harms involve a significantly higher number
of distinct adversarial techniques (complexity) compared to 'Non-Security' harms
like Reliability or Privacy.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7097 (Likely True) |
| **Posterior** | 0.1880 (Likely False) |
| **Surprise** | -0.6054 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 28.0 |
| Uncertain | 0.0 |
| Maybe False | 2.0 |
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

**Objective:** Assess whether security-related failures are technically more complex (in terms of MITRE ATLAS techniques used) than other failure types.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Categorize 'harm_type' into 'Security' (security, supply_chain) and 'Non-Security' (reliability, privacy, bias_discrimination, etc.).
- 3. Extract the 'technique_count' for each incident.
- 4. Perform an independent samples t-test (or Mann-Whitney U test) to compare the mean technique count between the two groups.
- 5. Visualize the distributions using boxplots.

### Deliverables
- Boxplot of Technique Counts by Harm Category and statistical test results (t-statistic, p-value).

---

## Analysis

The experiment tested the hypothesis that security-related incidents involve a
significantly higher level of technical complexity (measured by the count of
distinct MITRE ATLAS techniques) compared to non-security incidents. The
analysis of 52 case studies (40 Security, 12 Non-Security) refutes this
hypothesis.

Descriptive statistics show a remarkable similarity between the two groups: the
'Security' group had a mean technique count of 7.53 (Median=7.0), while the
'Non-Security' group had a mean of 7.58 (Median=7.5). The Mann-Whitney U test
yielded a p-value of 0.7844, confirming that there is no statistically
significant difference in the distribution of technique counts.

While the 'Security' group displayed a wider range of values (1 to 16
techniques) compared to the 'Non-Security' group (4 to 12 techniques), this
variance is likely attributable to the larger sample size (n=40 vs n=12)
capturing more outlier events. The core finding is that the complexity of an AI
incident, in terms of the number of adversarial steps or failures involved, is
consistent regardless of whether the outcome is a security breach or a
safety/reliability failure.

---

## Review

The experiment tested the hypothesis that security-related incidents involve a
significantly higher level of technical complexity (measured by the count of
distinct MITRE ATLAS techniques) compared to non-security incidents. The
analysis of 52 case studies (40 Security, 12 Non-Security) refutes this
hypothesis.

Descriptive statistics show a remarkable similarity between the two groups: the
'Security' group had a mean technique count of 7.53 (Median=7.0), while the
'Non-Security' group had a mean of 7.58 (Median=7.5). The Mann-Whitney U test
yielded a p-value of 0.7844, confirming that there is no statistically
significant difference in the distribution of technique counts.

While the 'Security' group displayed a wider range of values (1 to 16
techniques) compared to the 'Non-Security' group (4 to 12 techniques), this
variance is likely attributable to the larger sample size (n=40 vs n=12)
capturing more outlier events. The core finding is that the complexity of an AI
incident, in terms of the number of adversarial steps or failures involved, is
consistent regardless of whether the outcome is a security breach or a
safety/reliability failure.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind
import os

# [debug] Check file existence
file_name = 'step3_incident_coding.csv'
path = f"../{file_name}"
if not os.path.exists(path):
    path = file_name # Fallback to current directory

print(f"Loading dataset from: {path}")

try:
    df = pd.read_csv(path)
except FileNotFoundError:
    print(f"Error: Could not find {file_name} in ../ or current directory.")
    exit(1)

# Define categorization logic
security_harms = ['security', 'supply_chain']
# Any harm type not in security_harms is considered Non-Security for this hypothesis

def categorize_harm(harm_type):
    if harm_type in security_harms:
        return 'Security'
    else:
        return 'Non-Security'

# Apply categorization
if 'harm_type' in df.columns:
    df['Harm_Category'] = df['harm_type'].apply(categorize_harm)
else:
    print("Column 'harm_type' not found.")
    exit(1)

# Ensure technique_count exists
if 'technique_count' not in df.columns:
    # Attempt to calculate if missing, though metadata says it exists
    if 'techniques_used' in df.columns:
        df['technique_count'] = df['techniques_used'].astype(str).apply(lambda x: len(x.split(';')) if x.lower() != 'nan' else 0)
    else:
        print("Column 'technique_count' or 'techniques_used' not found.")
        exit(1)

# Separate groups
security_group = df[df['Harm_Category'] == 'Security']['technique_count']
non_security_group = df[df['Harm_Category'] == 'Non-Security']['technique_count']

# Descriptive Statistics
print("\n=== Descriptive Statistics (Technique Count) ===")
print(f"Security Group (n={len(security_group)}):")
print(security_group.describe())
print(f"\nNon-Security Group (n={len(non_security_group)}):")
print(non_security_group.describe())

# Statistical Test (Mann-Whitney U is preferred due to likely non-normal distribution and unequal sample sizes)
stat, p_value = mannwhitneyu(security_group, non_security_group, alternative='two-sided')

print("\n=== Statistical Test Results (Mann-Whitney U) ===")
print(f"U-statistic: {stat}")
print(f"P-value: {p_value:.5f}")

alpha = 0.05
if p_value < alpha:
    print("Result: Statistically significant difference found.")
else:
    print("Result: No statistically significant difference found.")

# Visualization
plt.figure(figsize=(8, 6))
data_to_plot = [security_group, non_security_group]
labels = [f'Security\n(n={len(security_group)})', f'Non-Security\n(n={len(non_security_group)})']

plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))
plt.title('Distribution of Technique Counts by Harm Category')
plt.ylabel('Number of Techniques Used')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_incident_coding.csv

=== Descriptive Statistics (Technique Count) ===
Security Group (n=40):
count    40.000000
mean      7.525000
std       3.129574
min       1.000000
25%       6.000000
50%       7.000000
75%       9.000000
max      16.000000
Name: technique_count, dtype: float64

Non-Security Group (n=12):
count    12.000000
mean      7.583333
std       2.274696
min       4.000000
25%       6.500000
50%       7.500000
75%       9.000000
max      12.000000
Name: technique_count, dtype: float64

=== Statistical Test Results (Mann-Whitney U) ===
U-statistic: 227.0
P-value: 0.78441
Result: No statistically significant difference found.

STDERR:
<ipython-input-1-8ee261e7786f>:78: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data_to_plot, labels=labels, patch_artist=True,


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Boxplot (or Box-and-Whisker Plot).
*   **Purpose:** The plot displays the distribution of a numerical variable ("Number of Techniques Used") across two distinct categorical groups ("Security" and "Non-Security"). It allows for a comparison of central tendency (median), dispersion (interquartile range), ranges, and outliers between the two categories.

### 2. Axes
*   **X-Axis:**
    *   **Label/Title:** Represents the **Harm Category**.
    *   **Categories:** Two categories are displayed: "Security" and "Non-Security".
    *   **Metadata:** The axis labels include the sample size ($n$) for each group: Security ($n=40$) and Non-Security ($n=12$).
*   **Y-Axis:**
    *   **Label/Title:** **Number of Techniques Used**.
    *   **Range:** The axis ticks range from 2 to 16, though the data points extend slightly below 2 (to 1).
    *   **Units:** Count (integer values representing the number of techniques).

### 3. Data Trends
*   **Security Category ($n=40$):**
    *   **Median:** The red line sits at approximately **7**.
    *   **Interquartile Range (IQR):** The box spans from approximately **6** (25th percentile) to **9** (75th percentile).
    *   **Range:** The whiskers extend from **3** to **12**.
    *   **Outliers:** This category shows significant outliers. There is a low outlier at **1**, and high outliers at **14, 15, and 16**.
*   **Non-Security Category ($n=12$):**
    *   **Median:** The red line is slightly higher than the Security category, positioned around **7.5**.
    *   **Interquartile Range (IQR):** The box spans from roughly **6.5** to **9**.
    *   **Range:** The whiskers extend from **4** to **12**.
    *   **Outliers:** There are **no visible outliers** for this category.

### 4. Annotations and Legends
*   **Title:** "Distribution of Technique Counts by Harm Category" clearly defines the scope of the analysis.
*   **Sample Size ($n$):** Annotated directly on the x-axis labels, indicating an unbalanced dataset with significantly more data points for the "Security" category than "Non-Security".
*   **Grid Lines:** Horizontal dashed grid lines are included to assist in reading specific values on the Y-axis.

### 5. Statistical Insights
*   **Central Tendency Similarity:** Despite the difference in categories, the median number of techniques used is quite similar, hovering between 7 and 7.5 for both groups. This suggests the typical complexity (in terms of technique count) is comparable regardless of the harm category.
*   **Variability and Extremes:** The "Security" category exhibits much higher variability than the "Non-Security" category. While the core spread (the box) is similar, the "Security" group contains extreme cases (outliers) reaching up to 16 techniques, whereas the "Non-Security" group is capped at 12.
*   **Impact of Sample Size:** The presence of outliers in the "Security" group and the lack thereof in the "Non-Security" group may be influenced by the sample size. With nearly 3.5 times more data points in the "Security" group ($n=40$ vs $n=12$), there is a higher statistical probability of capturing edge cases and anomalies.
*   **Conclusion:** Most incidents, regardless of harm category, utilize between 6 and 9 techniques. However, "Security" incidents have a potential long tail of high-complexity events (14+ techniques) that are not observed in the current "Non-Security" dataset.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
