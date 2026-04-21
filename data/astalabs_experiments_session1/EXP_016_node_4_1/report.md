# Experiment 16: node_4_1

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_1` |
| **ID in Run** | 16 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T21:58:24.795716+00:00 |
| **Runtime** | 169.5s |
| **Parent** | `node_3_1` |
| **Children** | None |
| **Creation Index** | 17 |

---

## Hypothesis

> Adversarial tactics associated with the 'Integration' domain have a
significantly higher frequency of occurrence in incidents than tactics
associated with the 'Trust' domain.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.2984 (Likely False) |
| **Posterior** | 0.2624 (Likely False) |
| **Surprise** | -0.0418 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 3.0 |
| Uncertain | 2.0 |
| Maybe False | 22.0 |
| Definitely False | 3.0 |

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

**Objective:** Evaluate the empirical frequency of tactics based on their mapped competency domain.

### Steps
- 1. Load 'step3_tactic_frequency.csv'.
- 2. Group the data by 'competency_domain' (or 'bundle').
- 3. Compare the 'incident_count' distributions between the groups using a Mann-Whitney U test.
- 4. Calculate the mean incident count for Trust tactics vs. Integration tactics.

### Deliverables
- Mean incident counts by domain; Mann-Whitney U test results.

---

## Analysis

The experiment successfully loaded the 'step3_tactic_frequency.csv' dataset and
analyzed the frequency of 16 adversarial tactics categorized by competency
domain.

**Findings:**
- **Integration Readiness (n=7):** Exhibited a higher central tendency (Mean =
29.14, Median = 33.0) but extreme variability (Std = 26.59), with incident
counts ranging from ~2 to 77.
- **Trust Readiness (n=9):** Showed a lower central tendency (Mean = 21.89,
Median = 17.0) but higher consistency (Std = 11.82).

**Statistical Result:**
Despite the visual difference in medians, the Mann-Whitney U test returned a
**p-value of 0.500** (U=32.0). This indicates that the distributions are not
statistically distinct; the wide variance in Integration tactics means they do
not consistently rank higher than Trust tactics.

**Conclusion:** The hypothesis is **not supported**. While Integration tactics
include the highest-frequency outliers, they also include low-frequency items,
resulting in no significant statistical difference in overall occurrence rates
compared to Trust tactics.

---

## Review

The experiment successfully loaded the 'step3_tactic_frequency.csv' dataset and
analyzed the frequency of 16 adversarial tactics categorized by competency
domain.

**Findings:**
- **Integration Readiness (n=7):** Exhibited a higher central tendency (Mean =
29.14, Median = 33.0) but extreme variability (Std = 26.59), with incident
counts ranging from ~2 to 77.
- **Trust Readiness (n=9):** Showed a lower central tendency (Mean = 21.89,
Median = 17.0) but higher consistency (Std = 11.82).

**Statistical Result:**
Despite the visual difference in medians, the Mann-Whitney U test returned a
**p-value of 0.500** (U=32.0). This indicates that the distributions are not
statistically distinct; the wide variance in Integration tactics means they do
not consistently rank higher than Trust tactics.

**Conclusion:** The hypothesis is **not supported**. While Integration tactics
include the highest-frequency outliers, they also include low-frequency items,
resulting in no significant statistical difference in overall occurrence rates
compared to Trust tactics.

---

## Code

```python
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import numpy as np

# Function to locate file either in current directory or one level up
def get_file_path(filename):
    if os.path.exists(filename):
        return filename
    elif os.path.exists(os.path.join("..", filename)):
        return os.path.join("..", filename)
    else:
        return filename  # Return original to let read_csv fail with clear error if not found

# 1. Load the dataset
file_name = 'step3_tactic_frequency.csv'
file_path = get_file_path(file_name)

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_name}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
except FileNotFoundError:
    print(f"Error: Could not find {file_name}")
    exit(1)

# 2. Group the data by 'bundle' (Competency Domain)
# Verify the column name for grouping
if 'bundle' not in df.columns:
    # Fallback based on metadata description if 'bundle' isn't there but 'competency_domain' is
    if 'competency_domain' in df.columns:
        group_col = 'competency_domain'
    else:
        print("Error: Neither 'bundle' nor 'competency_domain' column found.")
        exit(1)
else:
    group_col = 'bundle'

print(f"Grouping by column: {group_col}")
print(f"Unique groups: {df[group_col].unique()}")

# Filter groups
trust_data = df[df[group_col].str.contains('Trust', case=False, na=False)]
integration_data = df[df[group_col].str.contains('Integration', case=False, na=False)]

trust_counts = trust_data['incident_count']
integration_counts = integration_data['incident_count']

# 3. Calculate Descriptive Statistics
t_n = len(trust_counts)
t_mean = trust_counts.mean()
t_median = trust_counts.median()
t_std = trust_counts.std(ddof=1) if t_n > 1 else 0

i_n = len(integration_counts)
i_mean = integration_counts.mean()
i_median = integration_counts.median()
i_std = integration_counts.std(ddof=1) if i_n > 1 else 0

print("\n=== Descriptive Statistics ===")
print(f"Trust Readiness (n={t_n}): Mean={t_mean:.2f}, Median={t_median}, Std={t_std:.2f}")
print(f"Integration Readiness (n={i_n}): Mean={i_mean:.2f}, Median={i_median}, Std={i_std:.2f}")

# 4. Statistical Test (Mann-Whitney U)
# Hypothesis: Integration frequency > Trust frequency
u_stat, p_val = mannwhitneyu(integration_counts, trust_counts, alternative='greater')

print("\n=== Mann-Whitney U Test Results ===")
print(f"Hypothesis: Integration > Trust (frequency)")
print(f"U-statistic: {u_stat}")
print(f"p-value: {p_val:.5f}")

if p_val < 0.05:
    print("Result: Statistically Significant (Reject Null)")
else:
    print("Result: Not Statistically Significant (Fail to Reject Null)")

# 5. Visualization
plt.figure(figsize=(10, 6))
# Create boxplot data
data_to_plot = [trust_counts, integration_counts]
labels = [f'Trust Readiness\n(n={t_n})', f'Integration Readiness\n(n={i_n})']

plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
            boxprops=dict(facecolor="lightblue"), 
            medianprops=dict(color="red"))

plt.title('Incident Frequency Distribution: Trust vs Integration Tactics')
plt.ylabel('Incident Count per Tactic')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Successfully loaded step3_tactic_frequency.csv
Dataset shape: (16, 6)
Columns: ['tactic_id', 'tactic_name', 'incident_count', 'bundle', 'sub_competency_id', 'competency_domain']
Grouping by column: bundle
Unique groups: <StringArray>
['Integration Readiness', 'Trust Readiness']
Length: 2, dtype: str

=== Descriptive Statistics ===
Trust Readiness (n=9): Mean=21.89, Median=17.0, Std=11.82
Integration Readiness (n=7): Mean=29.14, Median=33.0, Std=26.59

=== Mann-Whitney U Test Results ===
Hypothesis: Integration > Trust (frequency)
U-statistic: 32.0
p-value: 0.50000
Result: Not Statistically Significant (Fail to Reject Null)

STDERR:
<ipython-input-1-209a34a6f203>:89: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data_to_plot, labels=labels, patch_artist=True,


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box and Whisker Plot (Boxplot).
*   **Purpose:** This plot visualizes the distribution of quantitative data ("Incident Count") across two categorical groups ("Trust Readiness" vs. "Integration Readiness"). It is designed to facilitate the comparison of central tendency (median), spread (variability), and the presence of outliers between these two groups.

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Incident Count per Tactic"
    *   **Range:** The axis is marked from 0 to 80 in increments of 10.
    *   **Units:** Count (integer values representing the number of incidents).
*   **X-Axis:**
    *   **Label:** Represents the two categories of tactics being analyzed.
    *   **Categories:**
        1.  "Trust Readiness"
        2.  "Integration Readiness"

### 3. Data Trends
*   **Trust Readiness (Left Box):**
    *   **Distribution:** This group shows a relatively compact distribution. The Interquartile Range (IQR)—represented by the blue box height—is small, roughly spanning from 15 to 24.
    *   **Spread:** The overall range (excluding outliers) is narrower, extending from approximately 11 to 34.
    *   **Outlier:** There is a distinct outlier (represented by the open circle) at approximately 47. This indicates that one specific tactic within this group resulted in a significantly higher incident count than the rest of the cohort.
*   **Integration Readiness (Right Box):**
    *   **Distribution:** This group exhibits a very wide distribution. The box is tall, with an IQR roughly spanning from 7 to 39.
    *   **Spread:** The whiskers cover a vast range, from a minimum of roughly 2 to a maximum of roughly 77. This indicates high variability in the results; some tactics had very few incidents, while others had the highest in the dataset.
    *   **Outliers:** No outliers are visible for this group; the spread is continuous.

### 4. Annotations and Legends
*   **Title:** "Incident Frequency Distribution: Trust vs Integration Tactics" — Defines the context of the comparison.
*   **Sample Size (n):** The X-axis labels include annotations for sample size:
    *   Trust Readiness: `n=9`
    *   Integration Readiness: `n=7`
*   **Median Line:** The red horizontal line inside each blue box indicates the median value for that group.
*   **Grid Lines:** Horizontal dashed grey lines serve as a visual aid to estimate Y-axis values.

### 5. Statistical Insights
*   **Central Tendency:** The **Integration Readiness** group has a noticeably higher median incident count (approx. 33) compared to the **Trust Readiness** group (approx. 17). This suggests that, typically, Integration Readiness tactics are associated with a higher frequency of incidents.
*   **Variability (Risk):** The **Integration Readiness** group shows significantly higher volatility. The large spread suggests unpredictability; while the median is higher, the group contains both the lowest and highest incident counts in the entire dataset. Conversely, **Trust Readiness** is more consistent and predictable, with most data points clustered tightly, barring the single outlier.
*   **Impact of Sample Size:** The sample sizes are small (`n=9` and `n=7`). While the visual difference in distribution is strong, the small `n` suggests that the outlier in the Trust group significantly impacts the mean, though the median remains robust. The extreme range in the Integration group is notable given there are only 7 data points.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
