# Experiment 57: node_5_10

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_10` |
| **ID in Run** | 57 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:20:44.229844+00:00 |
| **Runtime** | 195.9s |
| **Parent** | `node_4_10` |
| **Children** | None |
| **Creation Index** | 58 |

---

## Hypothesis

> Requirements derived from the 'EU AI Act' mandate a significantly higher number
of architecture controls per requirement than those derived from 'NIST AI RMF',
reflecting the regulatory intensity of the EU framework.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.1963 (Likely False) |
| **Surprise** | -0.6333 |
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
| Maybe False | 3.0 |
| Definitely False | 87.0 |

---

## Experiment Plan

**Objective:** Compare the architectural density (number of controls) of requirements sourced from the EU AI Act versus NIST AI RMF.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Filter rows where 'source' contains 'EU AI Act' or 'NIST AI RMF'.
- 3. Calculate a 'control_count' for each row by summing the binary indicators in the 18 architecture control columns.
- 4. Perform a t-test to compare the mean 'control_count' between the EU and NIST groups.

### Deliverables
- 1. Descriptive statistics of control counts by source.
- 2. T-test results (t-statistic, p-value).
- 3. Box plot comparing the distributions.

---

## Analysis

The experiment successfully tested the hypothesis regarding the architectural
density of EU AI Act versus NIST AI RMF requirements using the
'step2_crosswalk_matrix.csv' dataset.

**Descriptive Statistics:**
- **EU AI Act:** n=9 requirements, Mean = 1.67 controls per requirement, Std Dev
= 0.71.
- **NIST AI RMF:** n=23 requirements (combining NIST AI RMF 1.0 and GenAI
Profile), Mean = 1.87 controls per requirement, Std Dev = 0.69.

**Statistical Testing:**
- **Method:** Welch's t-test (unequal variances assumed).
- **Result:** t-statistic = -0.7335, p-value = 0.4750.
- **Conclusion:** The difference in means (-0.20) is not statistically
significant (p > 0.05). In fact, the NIST framework showed a slightly higher
mean density than the EU framework, contrary to the hypothesis direction.

**Visualization:**
- The box plot reveals highly similar distributions for both groups, with
medians centered at 2 controls and interquartile ranges spanning 1-2 controls.
This confirms that despite the regulatory differences, the architectural
implications of the requirements are structurally similar in terms of control
mapping density.

**Hypothesis Evaluation:**
- The hypothesis that EU AI Act requirements mandate a significantly higher
number of architecture controls is **rejected**.

---

## Review

The experiment was faithfully implemented and the analysis is methodologically
sound. The grouping of NIST variants ('NIST AI RMF 1.0' and 'NIST GenAI
Profile') was appropriate, and the counting logic correctly excluded metadata
columns. The statistical test (Welch's t-test) was suitable for the unequal
sample sizes (n=9 vs n=23). The findings clearly indicate that the hypothesis is
rejected.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def run_experiment():
    # 1. Load the dataset
    # Checking parent directory first as per instructions
    filename = 'step2_crosswalk_matrix.csv'
    if os.path.exists(f'../{filename}'):
        filepath = f'../{filename}'
    elif os.path.exists(filename):
        filepath = filename
    else:
        print(f"Error: {filename} not found in current or parent directory.")
        return

    print(f"Loading dataset from: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return
    
    # 2. Identify Control Columns
    # Known metadata columns: 'req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement'
    # All subsequent columns are architecture controls
    metadata_cols = {'req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement'}
    control_cols = [c for c in df.columns if c not in metadata_cols]
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Number of control columns identified: {len(control_cols)}")

    # 3. Calculate Control Count
    # Cells contain 'X' (or text) if mapped, empty/NaN otherwise.
    def get_control_count(row):
        count = 0
        for col in control_cols:
            val = row[col]
            if pd.notna(val) and str(val).strip() != '':
                count += 1
        return count

    df['control_count'] = df.apply(get_control_count, axis=1)

    # 4. Filter and Group by Source
    # We define two groups: 'EU AI Act' and 'NIST AI RMF' (grouping all NIST variants)
    print("\nOriginal Source Counts:")
    print(df['source'].value_counts())

    def classify_source(s):
        s_str = str(s).upper()
        if 'EU AI ACT' in s_str:
            return 'EU AI Act'
        elif 'NIST' in s_str:
            return 'NIST AI RMF'
        return None

    df['framework_group'] = df['source'].apply(classify_source)
    
    # Remove rows that don't match these two groups (e.g., OWASP)
    df_filtered = df.dropna(subset=['framework_group'])
    
    print("\nGrouped Framework Counts:")
    print(df_filtered['framework_group'].value_counts())

    # 5. Statistical Analysis
    group_eu = df_filtered[df_filtered['framework_group'] == 'EU AI Act']['control_count']
    group_nist = df_filtered[df_filtered['framework_group'] == 'NIST AI RMF']['control_count']

    # Descriptive Statistics
    print("\n--- Descriptive Statistics (Control Count) ---")
    print(df_filtered.groupby('framework_group')['control_count'].describe())

    # Check if we have data in both groups
    if len(group_eu) < 2 or len(group_nist) < 2:
        print("\nInsufficient data for t-test.")
    else:
        # Welch's t-test (equal_var=False)
        t_stat, p_val = stats.ttest_ind(group_eu, group_nist, equal_var=False)
        
        print("\n--- Welch's T-Test Results ---")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_val:.4f}")
        
        alpha = 0.05
        if p_val < alpha:
            print("Result: Statistically significant difference found.")
        else:
            print("Result: No statistically significant difference found.")

    # 6. Visualization
    if len(group_eu) > 0 and len(group_nist) > 0:
        plt.figure(figsize=(8, 6))
        data = [group_eu, group_nist]
        labels = ['EU AI Act', 'NIST AI RMF']
        
        plt.boxplot(data, labels=labels, patch_artist=True, 
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red'))
        
        plt.title('Architectural Control Density: EU AI Act vs NIST AI RMF')
        plt.ylabel('Number of Mapped Controls')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.show()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step2_crosswalk_matrix.csv
Dataset Shape: (42, 24)
Number of control columns identified: 18

Original Source Counts:
source
NIST AI RMF 1.0          19
OWASP Top 10 LLM         10
EU AI Act (2024/1689)     9
NIST GenAI Profile        4
Name: count, dtype: int64

Grouped Framework Counts:
framework_group
NIST AI RMF    23
EU AI Act       9
Name: count, dtype: int64

--- Descriptive Statistics (Control Count) ---
                 count      mean       std  min  25%  50%  75%  max
framework_group                                                    
EU AI Act          9.0  1.666667  0.707107  1.0  1.0  2.0  2.0  3.0
NIST AI RMF       23.0  1.869565  0.694416  1.0  1.0  2.0  2.0  3.0

--- Welch's T-Test Results ---
T-statistic: -0.7335
P-value: 0.4750
Result: No statistically significant difference found.

STDERR:
<ipython-input-1-1bf5b6a153c4>:102: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data, labels=labels, patch_artist=True,


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Boxplot (also known as a box-and-whisker plot).
*   **Purpose:** The plot compares the distribution of "Architectural Control Density" (specifically the number of mapped controls) between two different AI governance frameworks: the **EU AI Act** and the **NIST AI RMF**. It visualizes the median, quartiles, and range of the data for each group.

### 2. Axes
*   **X-Axis:**
    *   **Label:** Represents the two categories being compared: **"EU AI Act"** and **"NIST AI RMF"**.
    *   **Range:** Categorical (2 items).
*   **Y-Axis:**
    *   **Title:** **"Number of Mapped Controls"**.
    *   **Units:** Count (numerical value representing the quantity of controls).
    *   **Range:** The visible scale ranges from **1.00 to 3.00**, with grid lines marked at intervals of 0.25.

### 3. Data Trends
*   **Identical Distributions:** The boxplots for both the EU AI Act and NIST AI RMF appear visually identical.
*   **Medians (Red Lines):** For both categories, the red median line is situated at **2.00**. This indicates that the central tendency for the number of mapped controls in both frameworks is 2.
*   **Interquartile Range (The Box):**
    *   **Q3 (75th Percentile):** The top of the blue box is at **2.00**. Because the median line overlaps with the top of the box, it suggests that at least 25% of the data points (from the 50th to 75th percentile) are exactly 2.
    *   **Q1 (25th Percentile):** The bottom of the blue box is at **1.00**.
    *   This means the middle 50% of the data falls between 1 and 2 controls.
*   **Whiskers (Range):**
    *   **Upper Whisker:** Extends to **3.00**, indicating the maximum number of mapped controls in the dataset (excluding potential outliers, though none are plotted as dots).
    *   **Lower Whisker:** There is no visible lower whisker extending below the box. This implies that the minimum value is equal to the first quartile (Q1), which is **1.00**.

### 4. Annotations and Legends
*   **Title:** "Architectural Control Density: EU AI Act vs NIST AI RMF" is displayed at the top, defining the context of the comparison.
*   **Grid Lines:** Horizontal dashed gray lines are provided to facilitate accurate reading of the Y-axis values.
*   **Color Coding:** The boxes are filled with light blue and outlined in darker blue. The median is highlighted in red. No separate legend key is required as the X-axis labels identify the groups.

### 5. Statistical Insights
*   **Uniformity of Density:** The analysis suggests a strong similarity in the "density" of controls between the two frameworks regarding the specific architectural mapping being analyzed. Neither framework shows a higher median or wider spread than the other in this context.
*   **Low Variability:** The range of mapped controls is very tight (from 1 to 3). This indicates a highly standardized or constrained mapping where architectural elements are rarely associated with more than 3 controls, and never fewer than 1.
*   **Skew/Concentration:** The data appears to be concentrated at the lower end of the scale (1 and 2). With the median and the 75th percentile both at 2, and the minimum and 25th percentile both at 1, the vast majority of the data likely consists only of the integers 1 and 2, with fewer instances reaching 3.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
