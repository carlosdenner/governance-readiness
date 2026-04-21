# Experiment 79: node_4_26

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_26` |
| **ID in Run** | 79 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:05:47.946328+00:00 |
| **Runtime** | 172.3s |
| **Parent** | `node_3_6` |
| **Children** | `node_5_23` |
| **Creation Index** | 80 |

---

## Hypothesis

> Competency statements classified as 'Trust Readiness' are supported by
significantly higher evidence confidence levels in the literature compared to
'Integration Readiness' competencies.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.5565 (Maybe True) |
| **Posterior** | 0.1426 (Likely False) |
| **Surprise** | -0.4804 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 18.0 |
| Uncertain | 2.0 |
| Maybe False | 9.0 |
| Definitely False | 1.0 |

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

**Objective:** Assess if governance/policy competencies are more firmly established in literature than emerging technical/integration competencies.

### Steps
- 1. Load 'step2_competency_statements.csv'.
- 2. Map the 'confidence' column to numeric values (High=3, Medium=2, Low=1).
- 3. Group the data by 'bundle' ('Trust Readiness' vs. 'Integration Readiness').
- 4. Compare the numeric confidence scores between the two groups using a Mann-Whitney U test.
- 5. Calculate the mean confidence score for each bundle.

### Deliverables
- Bar chart of mean confidence scores and Mann-Whitney U test results.

---

## Analysis

The experiment tested the hypothesis that 'Trust Readiness' competencies are
supported by significantly higher evidence confidence levels than 'Integration
Readiness' competencies. The analysis of 42 competency statements mapped
confidence ratings to a numeric scale (High=3, Medium=2, Low=1).

Descriptive statistics showed nearly identical high-confidence levels for both
groups: 'Trust Readiness' had a mean score of 2.84 (n=19), while 'Integration
Readiness' had a slightly higher mean of 2.87 (n=23). The Mann-Whitney U test
(U=212.5, p=0.8187) confirmed that this difference is not statistically
significant.

Therefore, the hypothesis is unsupported. The results indicate that the evidence
base for the framework is uniformly strong across both governance and
architectural domains, with the vast majority of competencies in both bundles
rated as 'High' confidence.

---

## Review

The experiment successfully tested the hypothesis that 'Trust Readiness'
competencies are supported by significantly higher evidence confidence levels
than 'Integration Readiness' competencies. The analysis of 42 competency
statements mapped confidence ratings to a numeric scale (High=3, Medium=2,
Low=1). Descriptive statistics showed nearly identical high-confidence levels
for both groups: 'Trust Readiness' had a mean score of 2.84 (n=19), while
'Integration Readiness' had a slightly higher mean of 2.87 (n=23). The Mann-
Whitney U test (U=212.5, p=0.8187) confirmed that this difference is not
statistically significant. Therefore, the hypothesis is unsupported. The results
indicate that the evidence base for the framework is uniformly strong across
both governance and architectural domains, with the vast majority of
competencies in both bundles rated as 'High' confidence.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Load Data
try:
    df = pd.read_csv('../step2_competency_statements.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset not found at '../step2_competency_statements.csv'. Checking current directory...")
    try:
        df = pd.read_csv('step2_competency_statements.csv')
        print("Dataset loaded from current directory.")
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# 2. Preprocessing
# Map confidence to numeric values
# Assuming values might be 'High', 'Medium', 'Low' (case-insensitive)
confidence_map = {'high': 3, 'medium': 2, 'low': 1}
df['confidence_cleaned'] = df['confidence'].astype(str).str.strip().str.lower()
df['confidence_score'] = df['confidence_cleaned'].map(confidence_map)

# Debug: Check for unmapped values
unmapped = df[df['confidence_score'].isna()]
if not unmapped.empty:
    print(f"Warning: {len(unmapped)} records have unmapped confidence values:")
    print(unmapped['confidence'].unique())
    # Drop them for analysis
    df = df.dropna(subset=['confidence_score'])

# 3. Grouping
trust_scores = df[df['bundle'] == 'Trust Readiness']['confidence_score']
integration_scores = df[df['bundle'] == 'Integration Readiness']['confidence_score']

# 4. Descriptive Statistics
mean_trust = trust_scores.mean()
std_trust = trust_scores.std(ddof=1)
median_trust = trust_scores.median()
n_trust = len(trust_scores)

mean_integration = integration_scores.mean()
std_integration = integration_scores.std(ddof=1)
median_integration = integration_scores.median()
n_integration = len(integration_scores)

print("\n--- Descriptive Statistics ---")
print(f"Trust Readiness (n={n_trust}): Mean={mean_trust:.2f}, Median={median_trust}, Std={std_trust:.2f}")
print(f"Integration Readiness (n={n_integration}): Mean={mean_integration:.2f}, Median={median_integration}, Std={std_integration:.2f}")

# Cross-tabulation for detailed view
print("\n--- Confidence Level Distribution ---")
ct = pd.crosstab(df['bundle'], df['confidence_cleaned'])
# Reorder columns if they exist
order = [col for col in ['low', 'medium', 'high'] if col in ct.columns]
print(ct[order])

# 5. Statistical Test (Mann-Whitney U)
# Using Mann-Whitney U because data is ordinal and likely not normally distributed
u_stat, p_val = stats.mannwhitneyu(trust_scores, integration_scores, alternative='two-sided')

print("\n--- Mann-Whitney U Test Results ---")
print(f"U-statistic: {u_stat}")
print(f"P-value: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("Conclusion: Reject the null hypothesis. There is a statistically significant difference in confidence levels.")
else:
    print("Conclusion: Fail to reject the null hypothesis. No statistically significant difference in confidence levels detected.")

# 6. Visualization
labels = ['Trust Readiness', 'Integration Readiness']
means = [mean_trust, mean_integration]
stds = [std_trust, std_integration]

plt.figure(figsize=(8, 6))
# Use standard error for error bars (std / sqrt(n))
se_trust = std_trust / np.sqrt(n_trust)
se_integration = std_integration / np.sqrt(n_integration)
errors = [se_trust, se_integration]

bars = plt.bar(labels, means, yerr=errors, capsize=10, color=['#4c72b0', '#55a868'], alpha=0.8)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{height:.2f}',
             ha='center', va='bottom')

plt.title('Mean Evidence Confidence Score by Competency Bundle')
plt.ylabel('Confidence Score (1=Low, 2=Med, 3=High)')
plt.ylim(0, 3.5)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Error: Dataset not found at '../step2_competency_statements.csv'. Checking current directory...
Dataset loaded from current directory.

--- Descriptive Statistics ---
Trust Readiness (n=19): Mean=2.84, Median=3.0, Std=0.37
Integration Readiness (n=23): Mean=2.87, Median=3.0, Std=0.34

--- Confidence Level Distribution ---
confidence_cleaned     medium  high
bundle                             
Integration Readiness       3    20
Trust Readiness             3    16

--- Mann-Whitney U Test Results ---
U-statistic: 212.5
P-value: 0.8187
Conclusion: Fail to reject the null hypothesis. No statistically significant difference in confidence levels detected.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot (with error bars).
*   **Purpose:** The plot compares the mean "Evidence Confidence Score" between two distinct categories (Competency Bundles). It utilizes error bars to visualize the variability or uncertainty associated with the mean values.

### 2. Axes
*   **X-axis:**
    *   **Label/Title:** The specific axis title is implied by the main chart title "Competency Bundle."
    *   **Categories:** The axis displays two distinct categories: **"Trust Readiness"** and **"Integration Readiness."**
*   **Y-axis:**
    *   **Label:** "Confidence Score (1=Low, 2=Med, 3=High)". This label provides both the metric name and a key for interpreting the numerical scale.
    *   **Range:** The axis spans from **0.0 to 3.5**, with tick marks at intervals of 0.5.

### 3. Data Trends
*   **Tallest Bar:** The **"Integration Readiness"** bar (colored green) is the tallest, with a mean value of **2.87**.
*   **Shortest Bar:** The **"Trust Readiness"** bar (colored blue) is slightly shorter, with a mean value of **2.84**.
*   **Comparison:** The difference between the two bundles is minimal (only 0.03 points). Both bars are nearing the top of the defined scale (3.0), indicating that the values are clustered at the high end of the spectrum.
*   **Error Bars:** Both bars feature error bars at the top. The overlap in the vertical range of these error bars relative to the means suggests that the difference between the two groups may not be statistically significant.

### 4. Annotations and Legends
*   **Value Labels:** The exact mean values (**2.84** and **2.87**) are annotated directly above the error bars for precise reading.
*   **Gridlines:** Horizontal dashed gridlines are included at every 0.5 interval to assist with visual estimation of bar height.
*   **Color Coding:** The bars are distinct colors (blue for Trust Readiness, green for Integration Readiness) to visually differentiate the categories, though no separate legend box is provided or necessary given the x-axis labels.

### 5. Statistical Insights
*   **High Confidence Levels:** Based on the y-axis definition where 3 represents "High," both competency bundles exhibit very high confidence scores (2.84 and 2.87). This indicates that, on average, the evidence provided for both Trust and Integration readiness is considered very reliable or robust.
*   **Uniformity:** There is a lack of significant disparity between "Trust Readiness" and "Integration Readiness." This suggests that the confidence in evidence is consistent across these different competency areas; one area is not significantly lagging behind or outperforming the other in terms of evidence quality.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
