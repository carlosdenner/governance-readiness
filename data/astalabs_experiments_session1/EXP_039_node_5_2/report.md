# Experiment 39: node_5_2

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_2` |
| **ID in Run** | 39 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:45:56.534109+00:00 |
| **Runtime** | 208.9s |
| **Parent** | `node_4_0` |
| **Children** | `node_6_7` |
| **Creation Index** | 40 |

---

## Hypothesis

> Competency statements classified as 'Integration Readiness' are supported by a
significantly higher density of literature evidence (citations) than 'Trust
Readiness' statements, reflecting a larger volume of technical guidance compared
to governance guidance.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7339 (Likely True) |
| **Posterior** | 0.1880 (Likely False) |
| **Surprise** | -0.6335 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 29.0 |
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

**Objective:** Compare the empirical grounding strength between the two competency bundles.

### Steps
- 1. Load 'step2_competency_statements.csv'.
- 2. Create a new variable 'citation_count' by applying a regex pattern (e.g., '\[#\d+\]') to the 'evidence_summary' column and counting matches per row.
- 3. Group data by 'bundle'.
- 4. Perform a Welch's t-test to compare the mean 'citation_count' between 'Trust Readiness' and 'Integration Readiness'.

### Deliverables
- Descriptive statistics of citation counts, t-test results, and a boxplot of citation counts by bundle.

---

## Analysis

The experiment successfully analyzed the density of literature evidence
supporting the two competency bundles.

**Descriptive Statistics:**
- **Trust Readiness (n=19):** Mean = 2.21 citations, Std Dev = 0.54.
- **Integration Readiness (n=23):** Mean = 2.17 citations, Std Dev = 0.39.

**Statistical Inference:**
- **Method:** Welch's t-test (unequal variances).
- **Result:** t-statistic = -0.2491, p-value = 0.8049.
- **Conclusion:** The p-value is significantly greater than 0.05, indicating no
statistically significant difference between the means.

**Hypothesis Evaluation:**
The hypothesis that 'Integration Readiness' statements are supported by a
significantly higher density of evidence is **rejected**. The data reveals a
highly uniform evidence depth across the framework, with both bundles averaging
approximately 2.2 citations per statement. This suggests that the framework's
construction followed a consistent rigorous standard for evidence inclusion,
regardless of whether the domain was governance (Trust) or technical engineering
(Integration).

---

## Review

The experiment was successfully executed. The code correctly loaded the dataset,
extracted citation counts using regex, and performed the statistical comparison.
The results are valid and the conclusion matches the data.

---

## Code

```python
import pandas as pd
import re
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Define the filename
filename = 'step2_competency_statements.csv'

# robustly find the file
search_paths = [filename, f'../{filename}', f'../../{filename}']
file_path = None

for path in search_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    # Print debug info if file not found
    print(f"Could not find {filename} in searched paths: {search_paths}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Files in CWD: {os.listdir('.')}")
    try:
        print(f"Files in Parent: {os.listdir('..')}")
    except Exception as e:
        print(f"Could not list parent directory: {e}")
    raise FileNotFoundError(f"{filename} not found")

print(f"Loading dataset from {file_path}...")
df = pd.read_csv(file_path)

# Feature Engineering: Count citations in 'evidence_summary'
def count_citations(text):
    if pd.isna(text):
        return 0
    # Matches patterns like [#1], [#12], etc.
    return len(re.findall(r'\[#\d+\]', str(text)))

df['citation_count'] = df['evidence_summary'].apply(count_citations)

# Grouping by Bundle
trust_group = df[df['bundle'] == 'Trust Readiness']['citation_count']
integration_group = df[df['bundle'] == 'Integration Readiness']['citation_count']

# Descriptive Statistics
print("\n--- Descriptive Statistics for Citation Counts ---")
print(f"Trust Readiness (n={len(trust_group)}): Mean={trust_group.mean():.2f}, Std={trust_group.std():.2f}")
print(f"Integration Readiness (n={len(integration_group)}): Mean={integration_group.mean():.2f}, Std={integration_group.std():.2f}")

# Statistical Test (Welch's t-test)
t_stat, p_val = stats.ttest_ind(integration_group, trust_group, equal_var=False)
print("\n--- Welch's T-Test Results ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("Result: Statistically significant difference found.")
else:
    print("Result: No statistically significant difference found.")

# Visualization
plt.figure(figsize=(10, 6))
# Create boxplot data
data_to_plot = [trust_group, integration_group]
labels = ['Trust Readiness', 'Integration Readiness']

plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))

plt.title('Distribution of Literature Citations per Competency Bundle')
plt.ylabel('Citation Count per Statement')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from step2_competency_statements.csv...

--- Descriptive Statistics for Citation Counts ---
Trust Readiness (n=19): Mean=2.21, Std=0.54
Integration Readiness (n=23): Mean=2.17, Std=0.39

--- Welch's T-Test Results ---
T-statistic: -0.2491
P-value: 0.8049
Result: No statistically significant difference found.

STDERR:
<ipython-input-1-1acc433ea062>:72: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data_to_plot, labels=labels, patch_artist=True,


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is a detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (or Box-and-Whisker Plot).
*   **Purpose:** This plot visualizes the distribution, variability, and central tendency of citation counts for two distinct categories ("Competency Bundles"). It allows for a quick statistical comparison between the two groups regarding how heavily their constituent statements are cited in the literature.

### 2. Axes
*   **X-Axis:**
    *   **Label/Title:** Represents the categories of "Competency Bundles".
    *   **Categories:** "Trust Readiness" and "Integration Readiness".
*   **Y-Axis:**
    *   **Label/Title:** "Citation Count per Statement".
    *   **Units:** Count (discrete number of citations).
    *   **Range:** The axis ticks range from 1.00 to 3.00, with grid lines marking intervals of 0.25.

### 3. Data Trends
*   **Trust Readiness (Left Box):**
    *   **Median:** The red line indicates a median citation count of **2.0**.
    *   **Spread (IQR):** The box extends from 2.0 (the 25th percentile/Q1) to 2.5 (the 75th percentile/Q3). The fact that the median line overlaps with the bottom of the box suggests that at least 50% of the data points are concentrated at the value of 2.0.
    *   **Range:** The upper whisker extends to **3.0**, indicating the maximum non-outlier value.
    *   **Outliers:** There is a distinct outlier represented by a circle at **1.0**.
*   **Integration Readiness (Right Box):**
    *   **Median:** The red line indicates a median citation count of **2.0**.
    *   **Spread:** The "box" appears as a single flat line. This indicates that the 25th percentile, the median, and the 75th percentile are all identical (value of 2.0). There is almost no variance in the main body of the data.
    *   **Outliers:** There is a single outlier circle at **3.0**.

### 4. Annotations and Legends
*   **Title:** "Distribution of Literature Citations per Competency Bundle" is centered at the top.
*   **Grid Lines:** Horizontal dashed grey lines are provided to assist in reading the Y-axis values accurately.
*   **Color Coding:**
    *   **Blue Box:** Represents the Interquartile Range (IQR).
    *   **Red Line:** Represents the median value.
    *   **Black Circles:** Represent statistical outliers.

### 5. Statistical Insights
*   **Central Tendency:** Both competency bundles have the same median citation count of **2.0** per statement. This suggests that, on average, statements in both categories typically have 2 supporting citations.
*   **Variability Comparison:**
    *   **Trust Readiness** shows greater variability. While many statements have 2 citations, a significant portion (the upper quartile) ranges between 2 and 2.5, extending up to 3. There is also a case of lower evidence support (1 citation).
    *   **Integration Readiness** is highly consistent. The collapsed box suggests that nearly every statement in this bundle has exactly 2 citations, with the sole exception of the outlier(s) at 3.0.
*   **Skewness:** The "Trust Readiness" distribution is skewed upward (positive skew), as the variability extends towards higher values (up to 3.0) from the median, whereas "Integration Readiness" shows almost zero distribution width.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
