# Experiment 74: node_4_23

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_23` |
| **ID in Run** | 74 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:54:24.711921+00:00 |
| **Runtime** | 186.9s |
| **Parent** | `node_3_17` |
| **Children** | `node_5_19` |
| **Creation Index** | 75 |

---

## Hypothesis

> Incidents categorized as 'Security' harms implicate a significantly higher
number of missing sub-competencies per case than 'Non-Security' (e.g., Privacy,
Reliability) incidents, due to the multi-stage nature of adversarial attacks.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7097 (Likely True) |
| **Posterior** | 0.2128 (Likely False) |
| **Surprise** | -0.5767 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 27.0 |
| Uncertain | 2.0 |
| Maybe False | 1.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 15.0 |
| Definitely False | 75.0 |

---

## Experiment Plan

**Objective:** Compare the mean count of mapped sub-competencies between Security and Non-Security incidents.

### Steps
- 1. Load 'step3_enrichments.json'.
- 2. Create a 'sub_competency_count' variable by counting elements in 'sub_competency_ids' for each incident.
- 3. Create a binary group: 'Security' (harm_type == 'security') vs 'Other'.
- 4. Compute descriptive statistics (mean, std) for 'sub_competency_count' for both groups.
- 5. Perform a Mann-Whitney U test to assess if the 'Security' group has a statistically higher median count.

### Deliverables
- Group means, Mann-Whitney U statistic, p-value.

---

## Analysis

The experiment successfully tested the hypothesis that 'Security' incidents
implicate a significantly higher number of missing sub-competencies than 'Non-
Security' incidents.

**Key Findings:**
- **Descriptive Statistics:** The two groups showed remarkably similar central
tendencies. Security incidents (n=36) had a mean of 4.89 sub-competencies
(SD=1.53) and a median of 5.0. Non-Security incidents (n=16) had a mean of 4.63
(SD=1.09) and a median of 5.0.
- **Variance:** Security incidents exhibited higher variance, with counts
ranging from 1 to 9, whereas Non-Security incidents were more tightly clustered
(range 2-6).
- **Statistical Significance:** The Mann-Whitney U test yielded a p-value of
0.287, which is well above the significance threshold of 0.05.

**Conclusion:** The hypothesis is **rejected**. There is no statistically
significant difference in the complexity (number of competency gaps) between
Security and Non-Security incidents. Both types of failures tend to implicate
approximately 5 sub-competencies on average, though security incidents show
greater variability.

---

## Review

The experiment was successfully executed and the analysis is sound. The
hypothesis that 'Security' incidents implicate a significantly higher number of
missing sub-competencies than 'Non-Security' incidents was tested using a Mann-
Whitney U test.

**Key Findings:**
- **Descriptive Statistics:** The two groups showed remarkably similar central
tendencies. Security incidents (n=36) had a mean of 4.89 sub-competencies
(SD=1.53) and a median of 5.0. Non-Security incidents (n=16) had a mean of 4.63
(SD=1.09) and a median of 5.0.
- **Variance:** Security incidents exhibited higher variance, with counts
ranging from 1 to 9, whereas Non-Security incidents were more tightly clustered
(range 2-6).
- **Statistical Significance:** The Mann-Whitney U test yielded a p-value of
0.287, which is well above the significance threshold of 0.05.

**Conclusion:** The hypothesis is **rejected**. There is no statistically
significant difference in the complexity (number of competency gaps) between
Security and Non-Security incidents. Both types of failures tend to implicate
approximately 5 sub-competencies on average, though security incidents show
greater variability.

---

## Code

```python
import json
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import os

# [debug] Print current directory and list files to locate dataset
# print(f"Current working directory: {os.getcwd()}")
# print(f"Files in current directory: {os.listdir('.')}")
# if os.path.exists('../'):
#     print(f"Files in parent directory: {os.listdir('../')}")

# Define file path (prioritizing parent directory as per instructions)
filename = 'step3_enrichments.json'
file_path = f"../{filename}"
if not os.path.exists(file_path):
    file_path = filename  # Fallback to current directory

if not os.path.exists(file_path):
    print(f"Error: {filename} not found in ../ or .")
else:
    print(f"Loading {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Process data
    records = []
    for entry in data:
        # Extract relevant fields
        case_id = entry.get('case_study_id')
        harm = entry.get('harm_type')
        
        # Determine sub-competency count
        # Checking both 'sub_competency_ids' (primary) and handling potential formats
        ids = entry.get('sub_competency_ids', [])
        
        count = 0
        if isinstance(ids, list):
            count = len(ids)
        elif isinstance(ids, str):
            # If semicolon separated string
            if ids.strip():
                count = len([x for x in ids.split(';') if x.strip()])
        
        records.append({
            'case_study_id': case_id,
            'harm_type': harm,
            'sub_competency_count': count
        })
    
    df = pd.DataFrame(records)
    
    # Define groups
    df['group'] = df['harm_type'].apply(lambda x: 'Security' if x == 'security' else 'Non-Security')
    
    # Calculate Descriptive Statistics
    group_stats = df.groupby('group')['sub_competency_count'].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
    print("\n=== Descriptive Statistics by Group ===")
    print(group_stats)
    
    # Prepare samples for statistical test
    sec_counts = df[df['group'] == 'Security']['sub_competency_count']
    non_sec_counts = df[df['group'] == 'Non-Security']['sub_competency_count']
    
    # Perform Mann-Whitney U Test
    # Hypothesis: Security > Non-Security
    stat, p_val = mannwhitneyu(sec_counts, non_sec_counts, alternative='greater')
    
    print("\n=== Mann-Whitney U Test Results ===")
    print(f"Hypothesis: 'Security' incidents have more mapped sub-competencies than 'Non-Security'.")
    print(f"U-statistic: {stat}")
    print(f"P-value: {p_val:.5f}")
    
    alpha = 0.05
    if p_val < alpha:
        print("Result: Statistically Significant (Reject Null Hypothesis)")
    else:
        print("Result: Not Statistically Significant (Fail to Reject Null Hypothesis)")

    # Visualization
    plt.figure(figsize=(10, 6))
    # Create boxplot data
    data_to_plot = [sec_counts, non_sec_counts]
    labels = [f'Security (n={len(sec_counts)})', f'Non-Security (n={len(non_sec_counts)})']
    
    plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red'))
    
    plt.title('Distribution of Missing Sub-Competencies per Incident')
    plt.ylabel('Count of Sub-Competencies')
    plt.xlabel('Incident Harm Type')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate means
    means = [sec_counts.mean(), non_sec_counts.mean()]
    for i, mean in enumerate(means, 1):
        plt.text(i, mean + 0.1, f'Mean: {mean:.2f}', 
                 horizontalalignment='center', color='darkblue', fontweight='bold')

    plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading step3_enrichments.json...

=== Descriptive Statistics by Group ===
              count      mean       std  median  min  max
group                                                    
Non-Security     16  4.625000  1.087811     5.0    2    6
Security         36  4.888889  1.526486     5.0    1    9

=== Mann-Whitney U Test Results ===
Hypothesis: 'Security' incidents have more mapped sub-competencies than 'Non-Security'.
U-statistic: 315.5
P-value: 0.28675
Result: Not Statistically Significant (Fail to Reject Null Hypothesis)

STDERR:
<ipython-input-1-4de916f5d19c>:90: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data_to_plot, labels=labels, patch_artist=True,


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the plot:

### 1. Plot Type
*   **Type:** **Box Plot** (also known as a Box-and-Whisker plot).
*   **Purpose:** The plot compares the distribution of a numerical variable ("Count of Sub-Competencies") across two categorical groups ("Security" and "Non-Security"). It visualizes the median, quartiles, range, and outliers for each group.

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Count of Sub-Competencies"
    *   **Range:** The axis ticks range from **1 to 9**.
    *   **Units:** Integer count (representing the number of missing sub-competencies).
*   **X-Axis:**
    *   **Label:** "Incident Harm Type"
    *   **Categories:** Two categories are displayed:
        1.  **Security** (Sample size $n=36$)
        2.  **Non-Security** (Sample size $n=16$)

### 3. Data Trends
*   **Security Group ($n=36$):**
    *   **Spread:** This group shows a wide spread of data. The whiskers extend from a minimum of **1** to a maximum of **9**.
    *   **Interquartile Range (IQR):** The box (representing the middle 50% of the data) ranges from **4 to 6**.
    *   **Median:** The red median line is located at **5**.
    *   **Distribution:** The distribution appears relatively symmetrical around the median, covering a broad range of values without visible outliers.

*   **Non-Security Group ($n=16$):**
    *   **Spread:** The data is much more concentrated than the Security group.
    *   **Box:** The box is extremely compressed (appearing almost as a single line), suggesting that the 25th percentile, Median, and 75th percentile are all located at or very near the value of **5**.
    *   **Outliers:** There are distinct outliers (represented by open circles) visible at values **2, 3, 4, and 6**. This indicates that while the vast majority of cases have 5 missing sub-competencies, there are a few exceptions significantly lower or slightly higher than the norm.

### 4. Annotations and Legends
*   **Mean Values:** The plot includes text annotations explicitly stating the mean for each group:
    *   **Security:** "Mean: 4.89" (printed in dark blue).
    *   **Non-Security:** "Mean: 4.62" (printed in dark blue).
*   **Sample Size ($n$):** The x-axis labels include the sample count for each category ($n=36$ for Security, $n=16$ for Non-Security).
*   **Grid Lines:** Horizontal dashed grid lines are present at integer intervals to assist in reading the Y-values.

### 5. Statistical Insights
*   **Central Tendency Similarity:** Both groups have a median of **5**, and their means are quite close (4.89 vs 4.62). This suggests that the "typical" number of missing sub-competencies is similar regardless of the incident harm type.
*   **Variance Disparity:** The primary difference between the two groups is **variability**.
    *   **Security** incidents are highly variable; an incident might have anywhere from 1 to 9 missing sub-competencies.
    *   **Non-Security** incidents are highly consistent; they cluster tightly around the count of 5.
*   **Sample Size Impact:** The Security group has more than double the data points of the Non-Security group ($36$ vs $16$), which may contribute to the wider range observed (as larger samples are more likely to capture extreme values), though the extreme compression of the Non-Security IQR suggests a genuine difference in consistency between the types.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
