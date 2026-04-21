# Experiment 30: node_3_12

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_12` |
| **ID in Run** | 30 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:24:50.673605+00:00 |
| **Runtime** | 221.3s |
| **Parent** | `node_2_6` |
| **Children** | `node_4_25` |
| **Creation Index** | 31 |

---

## Hypothesis

> Governance requirements derived from the 'NIST AI RMF' family map to a
significantly higher number of architecture controls per requirement than those
from the 'EU AI Act', suggesting a higher density of technical prescription in
the NIST framework.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.2903 (Likely False) |
| **Posterior** | 0.2603 (Likely False) |
| **Surprise** | -0.0348 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 2.0 |
| Uncertain | 0.0 |
| Maybe False | 28.0 |
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

**Objective:** Compare the 'Control Density' (number of mapped controls) between NIST-sourced and EU-sourced requirements.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Calculate 'Control Density' for each row by summing the 'X' markers in the architecture control columns (columns 7 through 24).
- 3. Create a grouping variable based on the 'source' column (Group 1: Contains 'NIST', Group 2: Contains 'EU AI Act').
- 4. Perform an independent samples t-test on Control Density between the two groups.
- 5. Visualize with a bar chart of means with error bars.

### Deliverables
- T-test results, group means for control density, and visualization.

---

## Analysis

The experiment successfully loaded the 'step2_crosswalk_matrix.csv' dataset and
calculated the 'Control Density' for each governance requirement. The data was
split into two groups: NIST-sourced requirements (n=23) and EU AI Act-sourced
requirements (n=9).

Descriptive statistics show a slight directional difference supporting the
hypothesis: the NIST group had a higher mean control density (1.87 controls/req)
compared to the EU group (1.67 controls/req). However, the independent samples
t-test yielded a p-value of 0.4750 (t=0.7335), which is well above the alpha
level of 0.05. Consequently, the difference is not statistically significant,
and we fail to reject the null hypothesis.

The visualization confirms this finding, showing error bars that suggest
overlapping confidence intervals. While NIST requirements appear slightly more
prescriptive on average, the variance within both groups and the small sample
size (particularly for the EU group) mean we cannot conclusively state that NIST
requirements are denser in their technical mapping than the EU AI Act based on
this dataset.

---

## Review

The experiment was successfully executed and faithfully followed the proposed
plan. The code correctly loaded the 'step2_crosswalk_matrix.csv' dataset,
calculated the 'Control Density' metric for each governance requirement, and
grouped them by source (NIST vs. EU AI Act). The statistical analysis (Welch's
t-test) revealed a slight directional difference, with NIST-sourced requirements
mapping to a higher mean number of controls (1.87) compared to EU-sourced
requirements (1.67). However, the p-value of 0.4750 indicates this difference is
not statistically significant. The generated visualization accurately reflects
these findings, showing overlapping error bars and comparable means. Thus, while
the hypothesis was tested sufficiently, the data does not support the claim that
NIST requirements are significantly more technically prescriptive than those of
the EU AI Act in this specific mapping context.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Define the file path (using current directory based on previous error)
file_path = 'step2_crosswalk_matrix.csv'

try:
    # 1. Load the dataset
    df = pd.read_csv(file_path)

    # 2. Identify Metadata and Control Columns
    # Metadata columns based on dataset description
    metadata_cols = ['req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement']
    
    # All other columns are architecture controls
    control_cols = [c for c in df.columns if c not in metadata_cols]
    
    # 3. Calculate 'Control Density'
    # Count 'X' (case-insensitive) in control columns for each row
    def calculate_density(row):
        count = 0
        for col in control_cols:
            val = str(row[col]).strip().upper()
            if val == 'X':
                count += 1
        return count

    df['control_density'] = df.apply(calculate_density, axis=1)

    # 4. Create Grouping Variables
    # Group 1: NIST (NIST AI RMF 1.0, NIST GenAI Profile)
    nist_df = df[df['source'].str.contains('NIST', case=False, na=False)]
    
    # Group 2: EU AI Act
    eu_df = df[df['source'].str.contains('EU AI Act', case=False, na=False)]

    # Extract density series
    nist_density = nist_df['control_density']
    eu_density = eu_df['control_density']

    # 5. Statistical Analysis
    nist_mean = nist_density.mean()
    nist_std = nist_density.std()
    nist_n = len(nist_density)

    eu_mean = eu_density.mean()
    eu_std = eu_density.std()
    eu_n = len(eu_density)

    # Independent samples t-test (Welch's t-test for unequal variances)
    t_stat, p_val = stats.ttest_ind(nist_density, eu_density, equal_var=False)

    # Print Statistical Results
    print("=== Control Density Analysis: NIST vs EU AI Act ===")
    print(f"NIST Group (n={nist_n}): Mean = {nist_mean:.4f}, Std = {nist_std:.4f}")
    print(f"EU Group   (n={eu_n}): Mean = {eu_mean:.4f}, Std = {eu_std:.4f}")
    print(f"Difference in Means: {nist_mean - eu_mean:.4f}")
    print(f"T-Test Results: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
    
    alpha = 0.05
    if p_val < alpha:
        print("Result: Statistically Significant Difference (Reject Null Hypothesis)")
    else:
        print("Result: No Statistically Significant Difference (Fail to Reject Null Hypothesis)")

    # 6. Visualization
    # Prepare data for plotting
    groups = ['NIST Family', 'EU AI Act']
    means = [nist_mean, eu_mean]
    # Calculate Standard Error of the Mean (SEM) for error bars
    sems = [stats.sem(nist_density), stats.sem(eu_density)]

    plt.figure(figsize=(8, 6))
    # Create bar chart with error bars
    bars = plt.bar(groups, means, yerr=sems, capsize=10, color=['#4C72B0', '#55A868'], alpha=0.9, width=0.6)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.1, 
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Mean Control Density (Mapped Controls per Req)')
    plt.title('Comparison of Technical Prescription: NIST vs EU AI Act')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add significance annotation line if significant
    if p_val < 0.05:
        x1, x2 = 0, 1
        y_max = max(means) + max(sems) + 0.5
        h = 0.1
        plt.plot([x1, x1, x2, x2], [y_max, y_max+h, y_max+h, y_max], lw=1.5, c='k')
        plt.text((x1+x2)*.5, y_max+h, f"p={p_val:.3f}", ha='center', va='bottom', color='k')

    plt.ylim(0, max(means) + max(sems) + 1.5)  # Adjust y-axis limit for labels
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Please check the directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: === Control Density Analysis: NIST vs EU AI Act ===
NIST Group (n=23): Mean = 1.8696, Std = 0.6944
EU Group   (n=9): Mean = 1.6667, Std = 0.7071
Difference in Means: 0.2029
T-Test Results: t-statistic = 0.7335, p-value = 0.4750
Result: No Statistically Significant Difference (Fail to Reject Null Hypothesis)


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart with Error Bars.
*   **Purpose:** The plot compares the mean values of "Control Density" between two distinct regulatory/framework categories ("NIST Family" and "EU AI Act"). The inclusion of error bars indicates the variability or uncertainty associated with these mean measurements.

### 2. Axes
*   **X-Axis:**
    *   **Labels:** Categorical variables representing two frameworks: **"NIST Family"** and **"EU AI Act"**.
    *   **Nature:** Nominal/Categorical.
*   **Y-Axis:**
    *   **Label:** **"Mean Control Density (Mapped Controls per Req)"**.
    *   **Units:** The unit is a count ratio (Mapped Controls per Requirement).
    *   **Value Range:** The axis scale ranges from **0.0 to 3.5**, with grid lines marking increments of 0.5.

### 3. Data Trends
*   **Tallest Bar:** The **"NIST Family"** (blue) bar is the tallest, indicating a higher average control density.
*   **Shortest Bar:** The **"EU AI Act"** (green) bar is shorter.
*   **Specific Values:**
    *   **NIST Family:** Approximately **1.87**.
    *   **EU AI Act:** Approximately **1.67**.
*   **Comparison:** The NIST Family framework has a higher mean control density compared to the EU AI Act. Specifically, the NIST Family averages 0.20 more mapped controls per requirement than the EU AI Act.

### 4. Annotations and Legends
*   **Title:** "Comparison of Technical Prescription: NIST vs EU AI Act" – sets the context of the analysis as a comparison of technical specificity or "prescription."
*   **Data Labels:** The exact mean values are annotated in bold text above the error bars (**1.87** for NIST and **1.67** for EU AI Act).
*   **Error Bars:** Both bars feature black error bars (whiskers) extending above and below the top of the bar. These represent the variability of the data (likely Standard Deviation or Standard Error). The overlap between the error bars suggests that while the means are different, the distributions of the underlying data points are close.
*   **Grid:** Dashed horizontal lines serve as a visual aid to estimate values against the Y-axis.

### 5. Statistical Insights
*   **Technical Prescription:** The metric "Mapped Controls per Req" serves as a proxy for "Technical Prescription." A higher value indicates that for every high-level requirement, there are more specific technical controls defined.
*   **Granularity:** The **NIST Family** appears to be more technically prescriptive or granular, requiring nearly 2 technical controls per requirement on average (1.87). In contrast, the **EU AI Act** is slightly less prescriptive with a ratio of 1.67.
*   **Implication:** This suggests that the NIST framework might offer more specific technical guidance or stricter implementation details for each requirement compared to the EU AI Act, which may be higher-level or principal-based in its drafting.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
