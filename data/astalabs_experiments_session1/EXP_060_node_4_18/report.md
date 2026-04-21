# Experiment 60: node_4_18

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_18` |
| **ID in Run** | 60 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:24:18.204291+00:00 |
| **Runtime** | 188.4s |
| **Parent** | `node_3_17` |
| **Children** | `node_5_12` |
| **Creation Index** | 61 |

---

## Hypothesis

> Adversarial tactics mapped to the 'Integration Readiness' bundle appear with
significantly higher frequency in real-world incidents than those mapped to
'Trust Readiness', indicating that technical exploits are more common than
governance exploits.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6583 (Maybe True) |
| **Posterior** | 0.3396 (Maybe False) |
| **Surprise** | -0.3699 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 23.0 |
| Uncertain | 2.0 |
| Maybe False | 4.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 84.0 |
| Definitely False | 6.0 |

---

## Experiment Plan

**Objective:** Compare the incident frequency of tactics across the two competency bundles.

### Steps
- 1. Load 'step3_tactic_frequency.csv'.
- 2. Group data by 'bundle' ('Integration Readiness' vs 'Trust Readiness').
- 3. Extract the 'incident_count' for each tactic in both groups.
- 4. Perform a Mann-Whitney U test to compare the distributions of incident counts between the two bundles.

### Deliverables
- Group means/medians, Mann-Whitney U statistic, p-value.

---

## Analysis

The experiment successfully loaded 'step3_tactic_frequency.csv' to compare the
incident frequency of adversarial tactics across competency bundles.

**Descriptive Statistics:**
- **Integration Readiness (n=7):** Mean = 29.14, Median = 33.00, SD = 26.59.
- **Trust Readiness (n=9):** Mean = 21.89, Median = 17.00, SD = 11.82.

While the 'Integration Readiness' bundle shows a higher median and mean incident
count compared to 'Trust Readiness', it also exhibits significantly higher
variance (SD=26.59 vs 11.82).

**Statistical Testing:**
The Mann-Whitney U test yielded a U-statistic of 32.0 and a p-value of 1.00000.
The U-statistic is almost exactly equal to the expected mean rank sum (31.5) for
these sample sizes, indicating that the distributions of ranks are statistically
indistinguishable.

**Conclusion:**
The hypothesis that 'Integration Readiness' tactics appear with significantly
higher frequency is **not supported**. Despite the visual trend of a higher
median for Integration tactics, the small sample size and high variance result
in no statistically significant difference in incident frequency between
technical (Integration) and governance (Trust) exploits in this dataset.

---

## Review

The experiment was faithfully implemented. The code successfully loaded
'step3_tactic_frequency.csv', calculated the required descriptive statistics,
and performed the Mann-Whitney U test as planned.

**Findings:**
- **Descriptive Statistics:** 'Integration Readiness' tactics (n=7) exhibited a
higher mean (29.14 vs 21.89) and median (33.00 vs 17.00) incident count compared
to 'Trust Readiness' tactics (n=9). However, the Integration bundle also showed
significantly higher variance (SD=26.59 vs 11.82).
- **Statistical Significance:** The Mann-Whitney U test resulted in a
U-statistic of 32.0 and a p-value of 1.0. With the U-statistic being almost
exactly equal to the expected value for identical distributions (31.5), the test
indicates that the rank sums are statistically indistinguishable.
- **Conclusion:** The hypothesis that technical 'Integration Readiness' tactics
appear with significantly higher frequency is **not supported**. Despite the
visual trend of a higher median for Integration tactics, the statistical
evidence confirms that there is no significant difference in the frequency
distribution between the two bundles.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# [debug]
print("Current Working Directory:", os.getcwd())
print("Files in parent directory:", os.listdir('..'))

# Define file path based on instruction
file_path = '../step3_tactic_frequency.csv'

# Robustness check in case environment differs from instruction
if not os.path.exists(file_path):
    file_path = 'step3_tactic_frequency.csv'

try:
    # Load dataset
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}")
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Filter data by bundle
    integration_data = df[df['bundle'] == 'Integration Readiness']['incident_count']
    trust_data = df[df['bundle'] == 'Trust Readiness']['incident_count']

    # Calculate Descriptive Statistics
    int_mean = integration_data.mean()
    int_median = integration_data.median()
    int_std = integration_data.std()
    int_n = len(integration_data)

    trust_mean = trust_data.mean()
    trust_median = trust_data.median()
    trust_std = trust_data.std()
    trust_n = len(trust_data)

    print("\n=== Descriptive Statistics ===")
    print(f"Integration Readiness (n={int_n}): Mean={int_mean:.2f}, Median={int_median:.2f}, Std={int_std:.2f}")
    print(f"Trust Readiness       (n={trust_n}): Mean={trust_mean:.2f}, Median={trust_median:.2f}, Std={trust_std:.2f}")

    # Perform Mann-Whitney U Test
    # Using 'two-sided' to detect any difference, though hypothesis implies Integration > Trust
    u_stat, p_val = stats.mannwhitneyu(integration_data, trust_data, alternative='two-sided')

    print("\n=== Mann-Whitney U Test Results ===")
    print(f"U-statistic: {u_stat}")
    print(f"P-value: {p_val:.5f}")
    
    if p_val < 0.05:
        print("Result: Statistically Significant Difference")
    else:
        print("Result: No Statistically Significant Difference")

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.boxplot([integration_data, trust_data], labels=['Integration Readiness', 'Trust Readiness'], patch_artist=True)
    plt.title('Incident Count Frequency by Bundle')
    plt.ylabel('Incident Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Current Working Directory: /data
Files in parent directory: ['bin', '.uv', 'run', 'lib', 'tmp', '__modal', 'srv', 'etc', 'dev', 'opt', 'lib64', 'sys', 'sbin', 'root', 'var', 'home', 'boot', 'media', 'mnt', 'usr', 'proc', 'data']
Successfully loaded step3_tactic_frequency.csv
Dataset Shape: (16, 6)
Columns: ['tactic_id', 'tactic_name', 'incident_count', 'bundle', 'sub_competency_id', 'competency_domain']

=== Descriptive Statistics ===
Integration Readiness (n=7): Mean=29.14, Median=33.00, Std=26.59
Trust Readiness       (n=9): Mean=21.89, Median=17.00, Std=11.82

=== Mann-Whitney U Test Results ===
U-statistic: 32.0
P-value: 1.00000
Result: No Statistically Significant Difference

STDERR:
<ipython-input-1-721bddaa5e02>:61: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot([integration_data, trust_data], labels=['Integration Readiness', 'Trust Readiness'], patch_artist=True)


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (or Box-and-Whisker Plot).
*   **Purpose:** This plot visualizes the distribution of numerical data ("Incident Count") across distinct categorical groups ("Bundles"). It displays key statistical summaries such as the median, quartiles, and range, allowing for a comparison of central tendency and variability between the groups.

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Incident Count"
    *   **Units:** Integer count (frequency of incidents).
    *   **Range:** The axis is scaled from 0 to 80, with major tick marks every 10 units.
*   **X-Axis:**
    *   **Labels:** Represents the specific bundles being analyzed: "Integration Readiness" and "Trust Readiness".
    *   **Range:** Categorical (two distinct categories).

### 3. Data Trends
*   **Integration Readiness (Left Box):**
    *   **Spread:** Exhibits a high degree of variability. The data ranges from a minimum near 2 to a maximum near 77.
    *   **Interquartile Range (IQR):** The box (representing the middle 50% of data) is tall, stretching from approximately 7 to 39.
    *   **Median:** The central orange line indicates a median value of approximately 33.
*   **Trust Readiness (Right Box):**
    *   **Spread:** Shows much lower variability compared to the Integration Readiness bundle. The main data range (excluding the outlier) is roughly 11 to 34.
    *   **Interquartile Range (IQR):** The box is compact, ranging from approximately 15 to 24.
    *   **Median:** The median value is significantly lower, sitting at approximately 17.
    *   **Outliers:** There is a single distinct outlier (represented by the open circle) located at approximately 47.

### 4. Annotations and Legends
*   **Title:** "Incident Count Frequency by Bundle" clearly defines the scope of the analysis.
*   **Grid Lines:** Horizontal dashed grid lines are included at intervals of 10 to assist in estimating the y-values of the box plot elements.
*   **Colors:** The boxes are shaded blue, and the median is highlighted in orange for contrast. No separate legend is required as the x-axis labels identify the categories.

### 5. Statistical Insights
*   **Higher Volume in Integration Readiness:** The "Integration Readiness" bundle generally experiences a higher frequency of incidents. Its median (~33) is nearly double that of the "Trust Readiness" bundle (~17).
*   **Volatility:** "Integration Readiness" is significantly more volatile and unpredictable. The large range (0–77) suggests that while some instances have very few incidents, others have extremely high counts. In contrast, "Trust Readiness" is more consistent and predictable.
*   **Skewness:** The "Integration Readiness" distribution appears right-skewed, indicated by the long upper whisker and the concentration of the lower quartile near the bottom of the scale.
*   **Anomalies:** While "Trust Readiness" is generally stable, the presence of the outlier at 47 indicates a rare event where incident counts spiked significantly above the norm for that category.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
