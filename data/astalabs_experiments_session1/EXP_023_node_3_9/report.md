# Experiment 23: node_3_9

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_9` |
| **ID in Run** | 23 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:12:33.460679+00:00 |
| **Runtime** | 180.1s |
| **Parent** | `node_2_3` |
| **Children** | `node_4_5`, `node_4_28` |
| **Creation Index** | 24 |

---

## Hypothesis

> Trust Readiness competency statements map to a significantly higher number of
unique architecture controls per statement than Integration Readiness
statements, indicating that governance competencies are architecturally more
diffuse/cross-cutting.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7328 (Likely True) |
| **Posterior** | 0.1912 (Likely False) |
| **Surprise** | -0.6286 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 27.0 |
| Uncertain | 1.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

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

**Objective:** Analyze the architectural density of different competency bundles.

### Steps
- 1. Load 'step2_competency_statements.csv'.
- 2. For each row, count the number of items in the 'applicable_controls' column (split by semicolon).
- 3. Group the data by 'bundle' (Trust vs Integration).
- 4. Calculate descriptive statistics (mean, median) for control counts in each bundle.
- 5. Perform a T-test or Mann-Whitney U test to determine if the difference in control density is statistically significant.

### Deliverables
- Bar chart of average control count by bundle, Statistical test summary.

---

## Analysis

The experiment successfully analyzed the architectural density of competency
bundles using 'step2_competency_statements.csv'. The hypothesis that 'Trust
Readiness' statements map to a significantly higher number of architecture
controls was **rejected**.

**Findings:**
1.  **Directionality:** Contrary to the hypothesis, 'Integration Readiness'
statements actually had a higher mean control density (2.00 controls/statement)
compared to 'Trust Readiness' (1.68 controls/statement).
2.  **Statistical Significance:** The difference was not statistically
significant. The Mann-Whitney U test yielded a p-value of 0.0902, and Welch's
T-test yielded p=0.1308, both exceeding the standard alpha of 0.05.

**Conclusion:** There is no statistical evidence to support the claim that
governance (Trust) competencies are more architecturally diffuse than
engineering (Integration) competencies. In fact, the data suggests a slight
(though non-significant) trend where engineering competencies map to a denser
set of specific controls.

---

## Review

The experiment successfully tested the hypothesis regarding architectural
control density. The analysis of 42 competency statements revealed that,
contrary to the hypothesis, 'Integration Readiness' statements actually mapped
to a higher average number of architecture controls (Mean=2.00) compared to
'Trust Readiness' statements (Mean=1.68). The statistical analysis (Mann-Whitney
U test: U=158.5, p=0.0902; Welch's T-test: p=0.1308) indicated that this
difference is not statistically significant at the p<0.05 level. Therefore, the
hypothesis that governance competencies are more architecturally diffuse
(mapping to more controls) is rejected; the data suggests a non-significant
trend in the opposite direction, where engineering competencies are slightly
denser in their control mapping.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# [debug] Check current directory and parent directory contents if needed, but per instructions assume ../ exists
# Load dataset
filename = 'step2_competency_statements.csv'
filepath = f'../{filename}'

# Fallback if running in an environment where file is in current dir
if not os.path.exists(filepath):
    if os.path.exists(filename):
        filepath = filename
    else:
        # Start debug print
        print(f"File not found at {filepath} or {filename}")
        # End debug print

print(f"Loading {filepath}...")
try:
    df = pd.read_csv(filepath)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Parse 'applicable_controls' to count number of controls per statement
# Format is semicolon separated string
def count_controls(val):
    if pd.isna(val) or str(val).strip() == '':
        return 0
    # Split by semicolon and filter empty strings
    return len([x for x in str(val).split(';') if x.strip()])

df['control_count'] = df['applicable_controls'].apply(count_controls)

# Group data by bundle
trust_data = df[df['bundle'] == 'Trust Readiness']['control_count']
integration_data = df[df['bundle'] == 'Integration Readiness']['control_count']

# Calculate statistics
stats_summary = df.groupby('bundle')['control_count'].agg(['mean', 'median', 'std', 'count', 'sem'])
print("\n=== Architectural Density Statistics (Controls per Statement) ===")
print(stats_summary)

# Perform Statistical Test
# Shapiro-Wilk test for normality
_, p_trust_norm = stats.shapiro(trust_data)
_, p_int_norm = stats.shapiro(integration_data)

print(f"\nNormality Check (Shapiro-Wilk): Trust (p={p_trust_norm:.4f}), Integration (p={p_int_norm:.4f})")

# Use Mann-Whitney U Test (Non-parametric) as count data is often non-normal
u_stat, p_val_mw = stats.mannwhitneyu(trust_data, integration_data, alternative='two-sided')

# Use Welch's T-test (for comparison, robust to unequal variances)
t_stat, p_val_t = stats.ttest_ind(trust_data, integration_data, equal_var=False)

print("\n=== Hypothesis Test Results ===")
print(f"Mann-Whitney U Test: U={u_stat}, p={p_val_mw:.4f}")
print(f"Welch's T-Test: t={t_stat:.4f}, p={p_val_t:.4f}")

# Interpretation
if p_val_mw < 0.05:
    print("Result: Statistically significant difference in control density found.")
else:
    print("Result: No statistically significant difference in control density found.")

# Visualization: Bar Chart with Error Bars (SEM)
plt.figure(figsize=(8, 6))

bundles = stats_summary.index
means = stats_summary['mean']
sems = stats_summary['sem']

colors = ['#1f77b4', '#ff7f0e'] # Blue for Integration, Orange for Trust usually, but let's just map them
bar_colors = ['skyblue' if 'Integration' in b else 'salmon' for b in bundles]

bars = plt.bar(bundles, means, yerr=sems, capsize=10, color=bar_colors, alpha=0.8)

plt.title('Average Architecture Control Density by Competency Bundle')
plt.ylabel('Avg. Number of Controls per Statement')
plt.xlabel('Competency Bundle')
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.2f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading step2_competency_statements.csv...

=== Architectural Density Statistics (Controls per Statement) ===
                           mean  median       std  count       sem
bundle                                                            
Integration Readiness  2.000000     2.0  0.522233     23  0.108893
Trust Readiness        1.684211     2.0  0.749269     19  0.171894

Normality Check (Shapiro-Wilk): Trust (p=0.0005), Integration (p=0.0000)

=== Hypothesis Test Results ===
Mann-Whitney U Test: U=158.5, p=0.0902
Welch's T-Test: t=-1.5519, p=0.1308
Result: No statistically significant difference in control density found.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot with Error Bars.
*   **Purpose:** The plot compares the mean values of "Architecture Control Density" across two distinct categories ("Competency Bundles"). The inclusion of error bars indicates the variability or uncertainty associated with these mean measurements.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Competency Bundle"
    *   **Labels:** Two categorical groups: "Integration Readiness" and "Trust Readiness".
*   **Y-Axis:**
    *   **Title:** "Avg. Number of Controls per Statement"
    *   **Units:** Count/Ratio (Controls per Statement).
    *   **Value Range:** The axis is scaled from **0.00 to roughly 2.20**, with tick marks at intervals of **0.25** (0.00, 0.25, 0.50, ..., 2.00).

### 3. Data Trends
*   **Tallest Bar:** The **"Integration Readiness"** bundle (colored light blue) is the tallest, reaching an average value of **2.00**.
*   **Shortest Bar:** The **"Trust Readiness"** bundle (colored salmon/light red) is shorter, with an average value of **1.68**.
*   **Comparison:** There is a clear visible difference between the two categories. The "Integration Readiness" bundle has a higher average control density compared to "Trust Readiness."
*   **Variability:** Both bars feature error bars (whiskers). The error bars appear roughly similar in absolute length, suggesting comparable variance or standard error between the two datasets, though the "Trust Readiness" error bar extends noticeably further relative to its mean.

### 4. Annotations and Legends
*   **Value Labels:** The exact mean values are annotated directly above the error bars for clarity:
    *   **2.00** for Integration Readiness.
    *   **1.68** for Trust Readiness.
*   **Grid Lines:** Horizontal dashed grid lines are included to assist with visual estimation of bar heights against the Y-axis.
*   **Legend:** There is no separate legend box; the bars are distinguished by color and labeled directly on the X-axis.

### 5. Statistical Insights
*   **Control Density Disparity:** The data suggests that **Integration Readiness** requires or possesses a higher density of architecture controls per statement compared to **Trust Readiness**. Specifically, Integration Readiness has approximately **19% more controls per statement** on average than Trust Readiness ($2.00$ vs $1.68$).
*   **Significance:** While the exact statistical metric for the error bars (e.g., Standard Deviation, Standard Error, or 95% Confidence Interval) is not specified, the visual separation between the top of the "Trust Readiness" error bar and the mean of the "Integration Readiness" bar suggests a likely meaningful difference between the two groups. However, the error bars do overlap slightly in vertical range (the bottom of the blue error bar and the top of the red error bar appear to occupy similar Y-values around 1.85–1.90), implying that while the means are different, the population distributions may have some overlap.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
