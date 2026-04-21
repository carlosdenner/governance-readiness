# Experiment 15: node_3_5

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_5` |
| **ID in Run** | 15 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T21:58:24.793981+00:00 |
| **Runtime** | 177.6s |
| **Parent** | `node_2_1` |
| **Children** | `node_4_11`, `node_4_17`, `node_4_24` |
| **Creation Index** | 16 |

---

## Hypothesis

> Incidents that expose 'Integration Readiness' gaps involve a significantly
higher number of adversarial techniques (higher complexity) than those exposing
'Trust Readiness' gaps.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7177 (Likely True) |
| **Posterior** | 0.3698 (Maybe False) |
| **Surprise** | -0.4038 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 28.0 |
| Uncertain | 1.0 |
| Maybe False | 1.0 |
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

**Objective:** Determine if technical engineering failures correlate with more complex multi-step attacks compared to policy failures.

### Steps
- 1. Load 'step3_enrichments.json'.
- 2. Parse the 'sub_competency_ids' field (list of IDs like 'TR-1', 'IR-5').
- 3. Calculate an 'integration_ratio' for each incident: (Count of IR-xx IDs) / (Total Count of IDs).
- 4. Correlate 'integration_ratio' with 'technique_count' using Pearson or Spearman correlation.
- 5. Alternatively, split incidents into 'High Integration' (>0.5) and 'High Trust' (<=0.5) and compare mean 'technique_count' using a t-test.

### Deliverables
- Correlation coefficient or t-test results comparing attack complexity against competency domain.

---

## Analysis

The experiment successfully tested the hypothesis that incidents involving
'Integration Readiness' gaps (engineering failures) are associated with higher
attack complexity (technique count) than 'Trust Readiness' gaps
(policy/governance failures). Processing 52 MITRE ATLAS case studies, the
analysis found a weak negative correlation (Pearson r = -0.1973, p = 0.1609)
between the integration competency ratio and technique count. A Welch's t-test
comparing 'High Integration' incidents (n=18, mean techniques=7.06) against
'High Trust' incidents (n=34, mean techniques=7.79) yielded a p-value of 0.3884,
indicating no statistically significant difference. The hypothesis is therefore
not supported; in fact, the trend—though insignificant—suggests that policy-
focused failures may involve slightly more complex attack chains than
engineering failures in this dataset.

---

## Review

The experiment successfully tested the hypothesis that incidents involving
'Integration Readiness' gaps (engineering failures) are associated with higher
attack complexity (technique count) than 'Trust Readiness' gaps
(policy/governance failures). Processing 52 MITRE ATLAS case studies, the
analysis found a weak negative correlation (Pearson r = -0.1973, p = 0.1609)
between the integration competency ratio and technique count. A Welch's t-test
comparing 'High Integration' incidents (n=18, mean techniques=7.06) against
'High Trust' incidents (n=34, mean techniques=7.79) yielded a p-value of 0.3884,
indicating no statistically significant difference. The hypothesis is therefore
not supported; in fact, the trend—though insignificant—suggests that policy-
focused failures may involve slightly more complex attack chains than
engineering failures in this dataset.

---

## Code

```python
import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# Define file path
file_name = 'step3_enrichments.json'
file_path = f'../{file_name}' if os.path.exists(f'../{file_name}') else file_name

# Load dataset
print(f"Loading dataset from: {file_path}")
try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: File {file_path} not found.")
    exit(1)

# Process data
records = []
for entry in data:
    sub_competencies = entry.get('sub_competency_ids', [])
    technique_count = entry.get('technique_count', 0)
    
    # Handle cases where sub_competency_ids might be a string (CSV parsing artifact) or list
    if isinstance(sub_competencies, str):
        sub_competencies = [x.strip() for x in sub_competencies.split(';') if x.strip()]
    
    if not isinstance(sub_competencies, list):
        sub_competencies = []
        
    ir_count = sum(1 for cid in sub_competencies if cid.startswith('IR-'))
    total_count = len(sub_competencies)
    
    if total_count > 0:
        integration_ratio = ir_count / total_count
        records.append({
            'case_study_id': entry.get('case_study_id'),
            'technique_count': technique_count,
            'integration_ratio': integration_ratio,
            'sub_competency_count': total_count
        })

df = pd.DataFrame(records)

print(f"Processed {len(df)} incidents with valid competency mappings.")

# Analysis 1: Correlation
pearson_corr, p_value_corr = stats.pearsonr(df['integration_ratio'], df['technique_count'])
print(f"\nCorrelation (Integration Ratio vs Technique Count):")
print(f"  Pearson r: {pearson_corr:.4f}")
print(f"  P-value: {p_value_corr:.4f}")

# Analysis 2: Group Comparison
# Split into High Integration (> 0.5) and High Trust (<= 0.5)
df['group'] = df['integration_ratio'].apply(lambda x: 'High Integration' if x > 0.5 else 'High Trust')

group_counts = df['group'].value_counts()
print(f"\nGroup Sizes:\n{group_counts}")

high_integration_scores = df[df['group'] == 'High Integration']['technique_count']
high_trust_scores = df[df['group'] == 'High Trust']['technique_count']

# T-test
t_stat, p_value_ttest = stats.ttest_ind(high_integration_scores, high_trust_scores, equal_var=False)

print(f"\nGroup Statistics (Technique Count):")
print(f"  High Integration Mean: {high_integration_scores.mean():.2f} (std: {high_integration_scores.std():.2f})")
print(f"  High Trust Mean:       {high_trust_scores.mean():.2f} (std: {high_trust_scores.std():.2f})")
print(f"\nT-test results (Welch's):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value_ttest:.4f}")

# Visualization
plt.figure(figsize=(10, 6))

# Boxplot
boxplot_data = [high_trust_scores, high_integration_scores]
plt.boxplot(boxplot_data, labels=['High Trust (Ratio <= 0.5)', 'High Integration (Ratio > 0.5)'])
plt.title('Attack Complexity (Technique Count) by Competency Domain')
plt.ylabel('Technique Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add scatter points for visibility
for i, d in enumerate(boxplot_data, start=1):
    y = d
    x = np.random.normal(i, 0.04, size=len(y))
    plt.plot(x, y, 'r.', alpha=0.5)

plt.show()

# Interpretation
alpha = 0.05
print("\n=== Conclusion ===")
if p_value_ttest < alpha:
    if t_stat > 0:
        print("Result: Statistically significant. Integration-focused incidents involve MORE techniques.")
    else:
        print("Result: Statistically significant. Integration-focused incidents involve FEWER techniques.")
else:
    print("Result: No statistically significant difference in technique count between groups.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_enrichments.json
Processed 52 incidents with valid competency mappings.

Correlation (Integration Ratio vs Technique Count):
  Pearson r: -0.1973
  P-value: 0.1609

Group Sizes:
group
High Trust          34
High Integration    18
Name: count, dtype: int64

Group Statistics (Technique Count):
  High Integration Mean: 7.06 (std: 2.86)
  High Trust Mean:       7.79 (std: 2.98)

T-test results (Welch's):
  t-statistic: -0.8730
  p-value: 0.3884

=== Conclusion ===
Result: No statistically significant difference in technique count between groups.

STDERR:
<ipython-input-1-a208eb6eac5c>:84: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(boxplot_data, labels=['High Trust (Ratio <= 0.5)', 'High Integration (Ratio > 0.5)'])


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **Box Plot (Box-and-Whisker Plot)** overlaid with a **Strip Plot** (jittered scatter plot).
*   **Purpose:** The plot compares the distribution of "Attack Complexity" (measured by technique count) across two different categories ("High Trust" vs. "High Integration").
    *   The **box plot** summarizes the distribution statistics: the median (orange line), the interquartile range (the box itself representing the middle 50% of data), and the range of non-outlier data (whiskers).
    *   The **overlaid red dots** show the individual data points, allowing the viewer to see the sample size and density of the data that might otherwise be hidden by the box summary.

### 2. Axes
*   **Title:** "Attack Complexity (Technique Count) by Competency Domain"
*   **Y-Axis:**
    *   **Label:** "Technique Count"
    *   **Range:** The axis marks range from 2 to 16, though data points exist as low as 1 and as high as 16.
    *   **Units:** Count (integer number of techniques).
*   **X-Axis:**
    *   **Label/Categories:** This axis is categorical, split into two specific domains based on a ratio:
        1.  **High Trust (Ratio <= 0.5)**
        2.  **High Integration (Ratio > 0.5)**

### 3. Data Trends
*   **High Trust (Ratio <= 0.5):**
    *   **Median:** The median technique count is **8**.
    *   **Spread:** The middle 50% of the data (the box) ranges from **6 to 9**.
    *   **Range:** The whiskers extend from **3 to 12**.
    *   **Outliers:** There are two distinct high-value outliers at **14** and **16**.
*   **High Integration (Ratio > 0.5):**
    *   **Median:** The median technique count is slightly lower, at **7**.
    *   **Spread:** The middle 50% of the data is tighter, ranging from **6 to 8**.
    *   **Range:** The whiskers extend from **4 to 10**.
    *   **Outliers:** There is one high-value outlier at **15** and one low-value outlier at **1**.

### 4. Annotations and Legends
*   **Box Components:**
    *   **Orange Line:** Represents the **median** of the dataset.
    *   **Black Box:** Represents the **Interquartile Range (IQR)** (25th to 75th percentile).
    *   **Whiskers:** Indicate the range of the data excluding outliers (typically 1.5x the IQR).
    *   **Open Black Circles:** Represent **statistical outliers**.
*   **Red Dots:** Represent individual data observations. They are slightly jittered horizontally to prevent them from overlapping perfectly, making it easier to see the density of data points at specific values (e.g., multiple points at value 6 or 8).
*   **Grid Lines:** Horizontal dashed grey lines are provided at intervals of 2 units to assist with reading the Y-axis values.

### 5. Statistical Insights
*   **Higher Complexity in "High Trust":** The "High Trust" domain generally exhibits a slightly higher attack complexity. Its median (8) is higher than that of the "High Integration" domain (7), and its upper whisker extends to 12 compared to 10.
*   **Variance differences:** The "High Trust" domain shows greater variability. Its box is taller (larger IQR), and its whiskers cover a wider range (3-12) compared to the "High Integration" group (4-10). This suggests that attacks in the "High Trust" domain vary more significantly in complexity than those in the "High Integration" domain.
*   **Outlier Behavior:** Both domains are subject to extreme outliers on the high end (14-16 techniques), indicating that while the average attack uses 7-8 techniques, highly complex attacks are possible in both domains. However, only the "High Integration" domain shows a distinct low-complexity outlier (1 technique).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
