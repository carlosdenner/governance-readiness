# Experiment 11: node_3_2

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_2` |
| **ID in Run** | 11 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T21:52:11.671291+00:00 |
| **Runtime** | 154.5s |
| **Parent** | `node_2_2` |
| **Children** | `node_4_4` |
| **Creation Index** | 12 |

---

## Hypothesis

> Mitigation gaps related to 'Data' categories appear in significantly more
incidents on average than mitigation gaps related to 'Model' or 'System'
categories, suggesting data issues are the most pervasive root cause.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.2823 (Likely False) |
| **Posterior** | 0.0723 (Definitely False) |
| **Surprise** | -0.2437 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 2.0 |
| Uncertain | 0.0 |
| Maybe False | 27.0 |
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

**Objective:** Compare the frequency of varying mitigation gap categories in real-world incidents.

### Steps
- 1. Load 'step3_mitigation_gaps.csv'.
- 2. Identify the unique values in the 'category' column.
- 3. Group the data by 'category' and calculate the mean 'incident_count' for each.
- 4. Perform a one-way ANOVA test to determine if there are statistically significant differences in incident counts across the categories.

### Deliverables
- Mean incident counts per category and ANOVA F-statistic with p-value.

---

## Analysis

The experiment successfully loaded 'step3_mitigation_gaps.csv' and identified
three actual mitigation categories: 'Policy' (n=7), 'Technical - Cyber' (n=12),
and 'Technical - ML' (n=16). While the hypothesis anticipated 'Data', 'Model',
and 'System' categories, the dataset utilizes a broader taxonomy. The
statistical analysis showed that 'Technical - ML' had the highest average
incident count (15.38), followed closely by 'Technical - Cyber' (13.50) and
'Policy' (13.29). However, the one-way ANOVA test yielded an F-statistic of
0.3422 and a p-value of 0.7128, which is well above the 0.05 significance
threshold. This result indicates that there is no statistically significant
difference in the frequency of incidents across these mitigation categories.
Therefore, the hypothesis that certain mitigation gaps are significantly more
pervasive than others is not supported by this dataset.

---

## Review

The experiment was successfully executed. The code loaded
'step3_mitigation_gaps.csv' and identified the actual categories present in the
dataset: 'Policy', 'Technical - Cyber', and 'Technical - ML' (not the 'Data',
'Model', 'System' taxonomy anticipated by the hypothesis). The statistical
analysis (One-Way ANOVA) revealed that while 'Technical - ML' mitigation gaps
had the highest average incident count (15.38), followed by 'Technical - Cyber'
(13.50) and 'Policy' (13.29), the differences were not statistically significant
(F=0.34, p=0.71). Consequently, the hypothesis is rejected: there is no evidence
that one category of mitigation gaps is significantly more pervasive in
incidents than others.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the dataset
# The user specified that datasets are one level above the current working directory
file_path = '../step3_mitigation_gaps.csv'

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}")
except FileNotFoundError:
    # Fallback to current directory if the ../ path fails (just in case the environment differs)
    df = pd.read_csv('step3_mitigation_gaps.csv')
    print("Loaded step3_mitigation_gaps.csv from current directory")

# 2. Identify unique values in 'category'
categories = df['category'].unique()
print(f"\nUnique Categories Identified: {categories}")

# 3. Group by 'category' and calculate mean 'incident_count'
# We'll calculate count, mean, and std to understand the distribution
grouped_stats = df.groupby('category')['incident_count'].agg(['count', 'mean', 'std', 'min', 'max'])
print("\nIncident Count Statistics by Category:")
print(grouped_stats)

# 4. Perform One-Way ANOVA
# Prepare the data for ANOVA: list of arrays, one for each category
groups = [df[df['category'] == cat]['incident_count'] for cat in categories]

# We need at least two groups to perform ANOVA
if len(groups) > 1:
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\nOne-way ANOVA Results:")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_val:.4f}")
    
    alpha = 0.05
    if p_val < alpha:
        print("Result: Statistically significant difference found between categories (Reject H0).")
    else:
        print("Result: No statistically significant difference found between categories (Fail to reject H0).")
else:
    print("\nInsufficient categories for ANOVA.")

# Visualization: Boxplot to show the distribution variance
plt.figure(figsize=(10, 6))
# Create a boxplot grouping by category
# Using pandas plotting directly for convenience
df.boxplot(column='incident_count', by='category', grid=True)
plt.title('Distribution of Incident Counts by Mitigation Category')
plt.suptitle('')  # Removes the default pandas suptitle
plt.ylabel('Incident Count')
plt.xlabel('Category')
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded step3_mitigation_gaps.csv from current directory

Unique Categories Identified: <StringArray>
['Technical - Cyber', 'Technical - ML', 'Policy']
Length: 3, dtype: str

Incident Count Statistics by Category:
                   count       mean       std  min  max
category                                               
Policy                 7  13.285714  4.151879    8   20
Technical - Cyber     12  13.500000  7.090326    6   33
Technical - ML        16  15.375000  7.762087    2   31

One-way ANOVA Results:
F-statistic: 0.3422
P-value: 0.7128
Result: No statistically significant difference found between categories (Fail to reject H0).


=== Plot Analysis (figure 2) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (also known as a box-and-whisker plot).
*   **Purpose:** This plot is designed to visualize and compare the distribution of numerical data ("Incident Count") across distinct categorical groups ("Mitigation Category"). It effectively displays the median, quartiles, range, and potential outliers for each category.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Category"
    *   **Categories:** Three distinct groups representing mitigation types: "Policy", "Technical - Cyber", and "Technical - ML".
*   **Y-Axis:**
    *   **Label:** "Incident Count"
    *   **Range:** The visual scale displays tick marks from **5 to 30**. The actual data extends slightly beyond these bounds, ranging from approximately **2 to 33**.

### 3. Data Trends
*   **Policy:**
    *   Shows a relatively compact distribution.
    *   The median appears to be around **13**.
    *   The data ranges from approximately **8 to 20**, with no visible outliers.
*   **Technical - Cyber:**
    *   This group has the most compact Interquartile Range (IQR), indicating consistency in the majority of the data.
    *   The median is the lowest of the three groups, sitting at roughly **12.5**.
    *   **Outlier:** There is a significant outlier visible as a circle at the top of the plot, representing a value of approximately **33**. This indicates a single instance (or rare cluster) with a much higher incident count than typical for this category.
*   **Technical - ML:**
    *   This category exhibits the **highest variability** (spread).
    *   The box (IQR) is the tallest, and the whiskers extend the furthest, ranging from a low of roughly **2** to a high of roughly **31**.
    *   The median is the highest among the groups, at approximately **14**.

### 4. Annotations and Legends
*   **Title:** "Distribution of Incident Counts by Mitigation Category" clearly states the subject of the analysis.
*   **Color Coding:**
    *   **Blue Boxes:** Represent the Interquartile Range (IQR), encompassing the middle 50% of the data (25th to 75th percentile).
    *   **Green Lines:** Indicate the median value for each category.
    *   **Black Whiskers:** Extend to the minimum and maximum values within 1.5 times the IQR.
    *   **Circle:** Represents an outlier point falling outside the whiskers.
*   **Grid:** A grid overlay is provided to help estimate values on the Y-axis.

### 5. Statistical Insights
*   **Central Tendency:** The median incident counts are fairly similar across all three categories, hovering between 12.5 and 14. This suggests that the "typical" number of incidents is comparable regardless of the mitigation category.
*   **Volatility in ML:** The "Technical - ML" category is the least predictable. It has incidents ranging from very low (near 0) to very high (over 30). This broad spread suggests that technical mitigations involving Machine Learning may behave inconsistently or apply to a wider variety of scenarios than the other categories.
*   **Consistency in Cyber:** Apart from the single extreme outlier, the "Technical - Cyber" category shows the most consistent performance (smallest box), suggesting that typical cyber mitigation incidents fall within a narrower expected range than policy or ML-based mitigations.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
