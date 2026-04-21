# Experiment 40: node_4_10

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_10` |
| **ID in Run** | 40 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:45:56.535188+00:00 |
| **Runtime** | 242.6s |
| **Parent** | `node_3_0` |
| **Children** | `node_5_4`, `node_5_10` |
| **Creation Index** | 41 |

---

## Hypothesis

> Incidents classified as involving 'Both' Trust and Integration failures (multi-
domain) exhibit a significantly higher 'technique_count' than incidents
classified as single-domain failures, supporting the theory that multi-domain
failures represent more sophisticated threats.

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

**Objective:** Assess whether the complexity of an attack (technique count) is associated with the breadth of the competency gap (single vs. dual bundle) using the correct file path.

### Steps
- 1. Load the dataset 'step3_incident_coding.csv' from the current directory.
- 2. Create a new column 'domain_complexity' derived from 'trust_integration_split': label as 'Multi-Domain' if the value contains 'both' (case-insensitive), otherwise 'Single-Domain'.
- 3. Calculate summary statistics (count, mean, std) for 'technique_count' grouped by 'domain_complexity'.
- 4. Perform a Welch's t-test to compare the mean 'technique_count' between 'Multi-Domain' and 'Single-Domain' groups.
- 5. Generate a box plot or violin plot comparing the distributions.

### Deliverables
- 1. Descriptive statistics table of technique counts by domain complexity.
- 2. T-test results (t-statistic and p-value).
- 3. Visualization of technique count distributions.

---

## Analysis

The experiment successfully analyzed the relationship between incident
complexity (number of ATLAS techniques) and competency gap breadth (Multi-Domain
vs. Single-Domain).

**Descriptive Statistics:**
- **Multi-Domain (Both):** n=46, Mean=7.57 techniques, Median=7.0, Std Dev=3.05.
Range: 1-16.
- **Single-Domain:** n=6, Mean=7.33 techniques, Median=7.0, Std Dev=1.97. Range:
5-10.

**Statistical Results:**
- **Method:** Welch's t-test (unequal variances assumed).
- **Result:** t-statistic = 0.2520, p-value = 0.8070.
- **Conclusion:** The difference in means (0.24) is negligible and not
statistically significant (p > 0.05).

**Key Observations:**
1. **Hypothesis Rejection:** The data does not support the hypothesis that
multi-domain failures involve significantly more attack techniques. The average
complexity is nearly identical between the two groups.
2. **Variance Differences:** As shown in the box plot, Multi-Domain incidents
exhibit much higher variance, covering the full spectrum of complexity (1 to 16
techniques), whereas Single-Domain incidents are tightly clustered (5 to 10
techniques). This suggests that while multi-domain failures *can* be more
complex (as evidenced by high outliers), they are not *inherently* more complex
on average.
3. **Sample Imbalance:** The analysis is limited by the small sample size of
Single-Domain incidents (n=6), which reflects the dataset's domination by multi-
domain failures.

---

## Review

The experiment was faithfully implemented and the hypothesis was tested
utilizing the correct dataset and statistical methods. The analysis of
'step3_incident_coding.csv' compared the complexity (technique count) of multi-
domain failures (n=46) against single-domain failures (n=6).

**Hypothesis Status:** Rejected.

**Key Findings:**
1.  **No Significant Difference:** The Welch's t-test yielded a p-value of
0.8070, far exceeding the 0.05 threshold. The mean technique count for multi-
domain incidents (7.57) was statistically indistinguishable from single-domain
incidents (7.33).
2.  **Structural Variance:** While the medians were identical (7.0), the
visualization revealed that multi-domain failures exhibit significantly higher
variance (range 1-16) compared to the tightly clustered single-domain failures
(range 5-10). This indicates that while multi-domain failures do account for the
most complex outliers, they are not inherently more complex on average.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
import os

# Define the filename
filename = 'step3_incident_coding.csv'

# Attempt to load the dataset, handling potential path issues
if os.path.exists(filename):
    df = pd.read_csv(filename)
elif os.path.exists(f'../{filename}'):
    df = pd.read_csv(f'../{filename}')
else:
    print(f"Error: {filename} not found in current or parent directory.")
    sys.exit(1)

# Preprocess the data
# Normalize the grouping column
df['split_normalized'] = df['trust_integration_split'].astype(str).str.lower().str.strip()

# Create the Complexity Group variable
# 'Both' implies Multi-Domain; anything else (Trust-Dominant, Integration-Dominant) is Single-Domain
df['domain_complexity'] = df['split_normalized'].apply(
    lambda x: 'Multi-Domain' if 'both' in x else 'Single-Domain'
)

# Extract the technique counts for each group
group_multi = df[df['domain_complexity'] == 'Multi-Domain']['technique_count'].dropna()
group_single = df[df['domain_complexity'] == 'Single-Domain']['technique_count'].dropna()

# 1. Descriptive Statistics
print("=== Descriptive Statistics: Technique Count by Domain Complexity ===")
stats_df = df.groupby('domain_complexity')['technique_count'].describe()
print(stats_df)
print("\n")

# 2. Statistical Test (Welch's t-test)
# We use equal_var=False because sample sizes are likely unequal (Metadata suggests 46 vs 6 split)
t_stat, p_val = stats.ttest_ind(group_multi, group_single, equal_var=False)

print("=== Welch's T-Test Results ===")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value:     {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("Conclusion: The difference is statistically significant (Reject H0).")
else:
    print("Conclusion: The difference is NOT statistically significant (Fail to Reject H0).")

# 3. Visualization
plt.figure(figsize=(8, 6))
# Using a boxplot for clear comparison of medians and spread
plt.boxplot([group_multi, group_single], labels=['Multi-Domain', 'Single-Domain'], patch_artist=True,
            boxprops=dict(facecolor="lightblue"))
plt.title('Distribution of Attack Technique Counts by Competency Gap Complexity')
plt.ylabel('Count of ATLAS Techniques Used')
plt.xlabel('Competency Domain Gap')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: === Descriptive Statistics: Technique Count by Domain Complexity ===
                   count      mean       std  min  25%  50%   75%   max
domain_complexity                                                      
Multi-Domain        46.0  7.565217  3.052519  1.0  6.0  7.0  9.00  16.0
Single-Domain        6.0  7.333333  1.966384  5.0  6.0  7.0  8.75  10.0


=== Welch's T-Test Results ===
T-statistic: 0.2520
P-value:     0.8070
Conclusion: The difference is NOT statistically significant (Fail to Reject H0).

STDERR:
<ipython-input-1-c0d461a861aa>:59: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot([group_multi, group_single], labels=['Multi-Domain', 'Single-Domain'], patch_artist=True,


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (also known as a Box-and-Whisker Plot).
*   **Purpose:** This plot is designed to display and compare the distribution of numerical data ("Count of ATLAS Techniques Used") across different categorical groups ("Multi-Domain" vs. "Single-Domain"). It effectively visualizes the central tendency (median), data spread (interquartile range), and the presence of outliers.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Competency Domain Gap"
    *   **Labels:** Categorical variables representing two groups: "Multi-Domain" and "Single-Domain".
*   **Y-Axis:**
    *   **Title:** "Count of ATLAS Techniques Used"
    *   **Units:** Count (integer values representing the number of techniques).
    *   **Value Range:** The axis ticks range from 2 to 16, though the data points extend from approximately 1 to 16.

### 3. Data Trends
*   **Multi-Domain (Left Box):**
    *   **Median:** The orange line sits at **7**, indicating the median number of techniques used is 7.
    *   **Spread:** The interquartile range (IQR)—represented by the blue box—spans from **6** (25th percentile) to **9** (75th percentile).
    *   **Range:** The whiskers extend from a lower bound of **3** to an upper bound of **12**.
    *   **Outliers:** There is significant outlier activity in this category. There is a low outlier at **1** and several high outliers at **14, 15, and 16**. This suggests high variability in this group.

*   **Single-Domain (Right Box):**
    *   **Median:** The median is also located at **7**, matching the Multi-Domain group.
    *   **Spread:** The IQR is slightly tighter, ranging from **6** to roughly **8.5 or 9**.
    *   **Range:** The whiskers indicate a much more contained overall range, spanning from a minimum of **5** to a maximum of **10**.
    *   **Outliers:** There are **no outliers** visible for this group.

### 4. Annotations and Legends
*   **Title:** "Distribution of Attack Technique Counts by Competency Gap Complexity" sits at the top, clearly defining the scope of the analysis.
*   **Grid Lines:** Horizontal dashed grid lines appear at intervals of 2 (2, 4, 6... 16) to aid in estimating Y-axis values.
*   **Color Coding:** The boxes are filled with light blue, and the median is marked with an orange line. Outliers are depicted as hollow circles.

### 5. Statistical Insights
*   **Similar Central Tendency:** Both the "Multi-Domain" and "Single-Domain" groups share the same median count of techniques (7). This implies that in a typical scenario, the complexity in terms of the number of techniques used is similar regardless of the domain gap type.
*   **Difference in Variance:** The primary difference between the groups lies in predictability and variance. The "Single-Domain" group is highly consistent; technique counts never drop below 5 or exceed 10.
*   **Complexity "Long Tail":** The "Multi-Domain" group exhibits a "long tail" of complexity. While the median is the same, the existence of outliers up to 16 indicates that multi-domain gaps have the potential to spiral into significantly more complex scenarios requiring a much higher number of ATLAS techniques than single-domain gaps ever do. Conversely, they also occasionally require very few techniques (outlier at 1), suggesting a wider spectrum of possible scenarios.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
