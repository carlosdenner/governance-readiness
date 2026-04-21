# Experiment 100: node_6_10

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_10` |
| **ID in Run** | 100 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:47:10.671816+00:00 |
| **Runtime** | 203.1s |
| **Parent** | `node_5_3` |
| **Children** | None |
| **Creation Index** | 101 |

---

## Hypothesis

> Incidents resulting in 'Security' harms exhibit a significantly higher mean
'Technique Count' compared to incidents resulting in non-security harms (e.g.,
Privacy, Reliability, Bias), indicating greater adversarial complexity.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.1901 (Likely False) |
| **Surprise** | -0.6405 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 30.0 |
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

**Objective:** Validate whether security incidents are inherently more complex (in terms of ATLAS techniques) than other harm types.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Create a binary classification variable: 'Harm_Category' ('Security' vs. 'Other').
- 3. Group the data by 'Harm_Category'.
- 4. Perform a statistical test (T-test or Mann-Whitney U) to compare 'technique_count' between Security and Other incidents.
- 5. Create a bar chart with error bars representing the standard error of the mean for each category.

### Deliverables
- Statistical comparison results, Bar chart with error bars.

---

## Analysis

The experiment successfully tested the hypothesis that security incidents
involve a higher number of adversarial techniques (indicating greater
complexity) compared to other harm types. Using the 'step3_incident_coding.csv'
dataset (N=52), the analysis compared 'Security' incidents (n=36) against
'Other' incidents (n=16).

Findings:
1. **Descriptive Statistics**: The mean technique count for 'Security' incidents
was 7.50 (SD=3.17), while 'Other' incidents had a mean of 7.625 (SD=2.42).
Contrary to the hypothesis, non-security incidents had a nominally higher
average complexity.
2. **Statistical Testing**: Both the Mann-Whitney U test (p=0.52) and Welch's
T-test (p=0.88) failed to find any statistically significant difference between
the two groups.
3. **Hypothesis Evaluation**: The hypothesis is **REJECTED**. The data indicates
that adversarial complexity (as measured by the number of ATLAS techniques) is
not a distinguishing feature of security incidents in this dataset; privacy,
reliability, and supply chain incidents exhibit comparable levels of technical
complexity.

---

## Review

The experiment successfully tested the hypothesis that security incidents
involve a higher number of adversarial techniques (indicating greater
complexity) compared to other harm types. Using the 'step3_incident_coding.csv'
dataset (N=52), the analysis compared 'Security' incidents (n=36) against
'Other' incidents (n=16).

Findings:
1. **Descriptive Statistics**: The mean technique count for 'Security' incidents
was 7.50 (SD=3.17), while 'Other' incidents had a mean of 7.625 (SD=2.42).
Contrary to the hypothesis, non-security incidents had a nominally higher
average complexity.
2. **Statistical Testing**: Both the Mann-Whitney U test (p=0.52) and Welch's
T-test (p=0.88) failed to find any statistically significant difference between
the two groups.
3. **Hypothesis Evaluation**: The hypothesis is **REJECTED**. The data indicates
that adversarial complexity (as measured by the number of ATLAS techniques) is
not a distinguishing feature of security incidents in this dataset; privacy,
reliability, and supply chain incidents exhibit comparable levels of technical
complexity.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Define file path based on instructions (one level above)
file_path = '../step3_incident_coding.csv'

# Fallback to current directory if not found in parent (robustness)
if not os.path.exists(file_path) and os.path.exists('step3_incident_coding.csv'):
    file_path = 'step3_incident_coding.csv'

print(f"Loading dataset from: {file_path}")

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    # Attempting to list directories to aid debugging if this fails
    print(f"Contents of ..: {os.listdir('..')}")
    raise

# Preprocessing
# Ensure technique_count is numeric
df['technique_count'] = pd.to_numeric(df['technique_count'], errors='coerce')

# Create binary category
# 'security' vs everything else
df['Harm_Category'] = df['harm_type'].apply(lambda x: 'Security' if str(x).lower() == 'security' else 'Other')

# Descriptive Statistics
group_stats = df.groupby('Harm_Category')['technique_count'].agg(['count', 'mean', 'std', 'sem'])
print("\n=== Descriptive Statistics (Technique Count) ===")
print(group_stats)

# Prepare data for testing
sec_data = df[df['Harm_Category'] == 'Security']['technique_count'].dropna()
other_data = df[df['Harm_Category'] == 'Other']['technique_count'].dropna()

# Statistical Testing
# 1. Mann-Whitney U Test (Non-parametric, robust to outliers/non-normality)
u_stat, p_val_mw = stats.mannwhitneyu(sec_data, other_data, alternative='two-sided')

# 2. Welch's T-test (Parametric, assumes normality but robust to unequal variances)
t_stat, p_val_t = stats.ttest_ind(sec_data, other_data, equal_var=False)

print("\n=== Statistical Test Results ===")
print(f"Mann-Whitney U Test: Statistic={u_stat}, p-value={p_val_mw:.5f}")
print(f"Welch's T-Test:      Statistic={t_stat:.4f}, p-value={p_val_t:.5f}")

# Interpretation helper
if p_val_mw < 0.05:
    print("Result: Statistically significant difference detected (p < 0.05).")
else:
    print("Result: No statistically significant difference detected (p >= 0.05).")

# Visualization
plt.figure(figsize=(8, 6))

# Reorder for consistent plotting
plot_order = ['Security', 'Other']
plot_means = [group_stats.loc[cat, 'mean'] for cat in plot_order]
plot_sems = [group_stats.loc[cat, 'sem'] for cat in plot_order]
plot_colors = ['salmon', 'lightgray']

bars = plt.bar(plot_order, plot_means, yerr=plot_sems, capsize=10, color=plot_colors, edgecolor='black', alpha=0.8)

plt.title('Mean ATLAS Technique Count by Harm Category')
plt.ylabel('Mean Technique Count (+/- SEM)')
plt.xlabel('Harm Category')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{height:.2f}',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_incident_coding.csv

=== Descriptive Statistics (Technique Count) ===
               count   mean       std       sem
Harm_Category                                  
Other             16  7.625  2.418677  0.604669
Security          36  7.500  3.166792  0.527799

=== Statistical Test Results ===
Mann-Whitney U Test: Statistic=255.5, p-value=0.52260
Welch's T-Test:      Statistic=-0.1557, p-value=0.87708
Result: No statistically significant difference detected (p >= 0.05).


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot (with error bars).
*   **Purpose:** The plot compares the mean values of a numerical variable (ATLAS Technique Count) across two distinct categorical groups (Harm Categories: Security vs. Other). The inclusion of error bars indicates it is visualizing central tendency (mean) along with a measure of variability or precision (Standard Error of the Mean).

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Harm Category"
    *   **Labels:** Two categories are represented: "Security" and "Other".
*   **Y-Axis:**
    *   **Title:** "Mean Technique Count (+/- SEM)"
    *   **Range:** The axis spans from 0 to slightly above 8, with grid lines marking integer intervals (0, 1, 2... 8).
    *   **Units:** Count (numerical frequency).

### 3. Data Trends
*   **Bar Heights:**
    *   The **"Other"** category represents the tallest bar with a value of **7.62**.
    *   The **"Security"** category is slightly shorter with a value of **7.50**.
*   **Pattern:** The two categories are remarkably similar in height. There is very little visual difference between the mean technique count for security-related harm compared to other categories.

### 4. Annotations and Legends
*   **Title:** "Mean ATLAS Technique Count by Harm Category" appears at the top center.
*   **Value Labels:** Specific mean values are annotated in bold text above the bars:
    *   Security: **7.50**
    *   Other: **7.62**
*   **Error Bars:** Black "whiskers" extend above and below the top of the bars. The Y-axis label specifies these represent the Standard Error of the Mean (SEM). The error bars appear to overlap when projected horizontally.
*   **Color Coding:** The "Security" bar is colored salmon/light red, while the "Other" bar is colored light gray.

### 5. Statistical Insights
*   **Minimal Difference:** The difference between the mean technique count for "Security" (7.50) and "Other" (7.62) is numerically very small (0.12).
*   **Variability and Overlap:** The error bars (SEM) for both categories overlap significantly. In statistical terms, overlapping error bars (specifically SEM) often suggest that the difference between the two groups is likely not statistically significant.
*   **Conclusion:** Based on this chart, the category of harm (Security vs. Other) does not appear to be a strong predictor for the number of ATLAS techniques used, as the average counts are nearly identical.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
