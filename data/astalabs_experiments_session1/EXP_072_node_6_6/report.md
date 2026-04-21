# Experiment 72: node_6_6

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_6` |
| **ID in Run** | 72 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:50:03.062535+00:00 |
| **Runtime** | 228.6s |
| **Parent** | `node_5_8` |
| **Children** | None |
| **Creation Index** | 73 |

---

## Hypothesis

> 'Integration Readiness' competency statements require more verbose definitions
(higher word count) to capture technical nuance compared to 'Trust Readiness'
statements.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.4839 (Uncertain) |
| **Posterior** | 0.1240 (Likely False) |
| **Surprise** | -0.4177 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 12.0 |
| Uncertain | 4.0 |
| Maybe False | 14.0 |
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

**Objective:** Analyze the linguistic complexity/verbosity of competency statements across bundles.

### Steps
- 1. Load 'step2_competency_statements.csv'.
- 2. Calculate the word count for each 'competency_statement'.
- 3. Group by 'bundle'.
- 4. Perform a t-test to compare the mean word counts.
- 5. Visualize the word count distributions.

### Deliverables
- Boxplot of Word Counts by Bundle, T-Test Results.

---

## Analysis

The experiment tested the hypothesis that 'Integration Readiness' competency
statements are more verbose than 'Trust Readiness' statements due to technical
nuance. The analysis of `step2_competency_statements.csv` (n=42) measured the
word count of each statement. The results showed nearly identical means:
'Integration Readiness' statements averaged 26.65 words (SD=4.14), while 'Trust
Readiness' statements averaged 26.63 words (SD=3.83). An independent t-test
yielded a p-value of 0.9868, indicating no statistically significant difference.
The boxplot visualization confirms similar distributions, though 'Integration
Readiness' had one outlier (39 words). Consequently, the hypothesis is rejected.
The uniformity in sentence length suggests the agentic pipeline utilized a
standardized drafting template or style guide when generating competency
statements, regardless of the domain (governance vs. engineering).

---

## Review

The experiment tested the hypothesis that 'Integration Readiness' competency
statements require more verbose definitions than 'Trust Readiness' statements
due to higher technical nuance. The analysis of 42 competency statements from
'step2_competency_statements.csv' revealed that the word counts for both bundles
are nearly identical. 'Integration Readiness' statements had a mean length of
26.65 words (SD=4.14), while 'Trust Readiness' statements had a mean length of
26.63 words (SD=3.83). An independent samples t-test yielded a p-value of
0.9868, indicating no statistically significant difference between the groups.
Consequently, the hypothesis is rejected. The boxplot visualization further
confirms the similarity in distributions, with a single outlier in the
Integration group. The findings suggest that the agentic pipeline employed a
standardized stylistic template or length constraint when generating competency
statements, regardless of the underlying domain (governance vs. engineering).

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load the dataset
file_path = '../step2_competency_statements.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}")
except FileNotFoundError:
    # Fallback if running in a different environment structure, though instruction said one level up
    try:
        df = pd.read_csv('step2_competency_statements.csv')
        print("Loaded from current directory")
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# Calculate word count for each competency statement
# Using simple whitespace splitting for word count
df['word_count'] = df['competency_statement'].fillna('').astype(str).apply(lambda x: len(x.split()))

# Group data by bundle
integration_group = df[df['bundle'] == 'Integration Readiness']['word_count']
trust_group = df[df['bundle'] == 'Trust Readiness']['word_count']

# Calculate descriptive statistics
stats_summary = df.groupby('bundle')['word_count'].describe()
print("\n=== Descriptive Statistics (Word Count) ===")
print(stats_summary)

# Perform Independent T-test (Welch's t-test, assuming unequal variances)
t_stat, p_val = stats.ttest_ind(integration_group, trust_group, equal_var=False)

print("\n=== Statistical Test Results ===")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")
if p_val < 0.05:
    print("Result: Significant difference in word counts between bundles.")
else:
    print("Result: No significant difference in word counts between bundles.")

# Visualization
plt.figure(figsize=(10, 6))
plt.boxplot([integration_group, trust_group], labels=['Integration Readiness', 'Trust Readiness'])
plt.title('Distribution of Competency Statement Word Counts by Bundle')
plt.ylabel('Word Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded from current directory

=== Descriptive Statistics (Word Count) ===
                       count       mean       std   min   25%   50%   75%   max
bundle                                                                         
Integration Readiness   23.0  26.652174  4.140802  22.0  23.0  27.0  29.0  39.0
Trust Readiness         19.0  26.631579  3.832761  21.0  24.0  25.0  28.5  34.0

=== Statistical Test Results ===
T-statistic: 0.0167
P-value: 0.9868
Result: No significant difference in word counts between bundles.

STDERR:
<ipython-input-1-914408a2830f>:48: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot([integration_group, trust_group], labels=['Integration Readiness', 'Trust Readiness'])


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is a detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (also known as a Box-and-Whisker Plot).
*   **Purpose:** The plot visualizes the distribution of quantitative data ("Word Count") across categorical groups ("Bundles"). It displays statistical summaries such as the median, quartiles (interquartile range), minimum/maximum values, and outliers, allowing for a comparison of central tendency and variability between the two groups.

### 2. Axes
*   **X-Axis:**
    *   **Label:** Represents the Categories of Bundles.
    *   **Values:** "Integration Readiness" and "Trust Readiness".
*   **Y-Axis:**
    *   **Label:** "Word Count" (Number of words).
    *   **Range:** The axis ticks range from roughly **21 to 39** (visual scale shows grid lines from 22.5 to 37.5).
    *   **Units:** Count (integer values representing the number of words).

### 3. Data Trends
*   **Integration Readiness (Left Box):**
    *   **Median:** The orange line is situated at approximately **27 words**.
    *   **Interquartile Range (IQR):** The middle 50% of the data spans from approximately **23 to 29 words**.
    *   **Whiskers:** The lower whisker extends to **22**, and the upper whisker extends to **32**.
    *   **Outliers:** There is a distinct outlier plotted as a circle at approximately **39 words**, indicating one specific statement was significantly longer than the rest of the group.
*   **Trust Readiness (Right Box):**
    *   **Median:** The orange line is lower than the other group, sitting at exactly **25 words**.
    *   **Interquartile Range (IQR):** The middle 50% spans from approximately **24 to 28.5 words**.
    *   **Whiskers:** This group shows a wider total range (excluding outliers) with the bottom whisker at roughly **21** and the top whisker extending up to **34**.
    *   **Outliers:** No outliers are visible for this category.

### 4. Annotations and Legends
*   **Title:** "Distribution of Competency Statement Word Counts by Bundle" clearly defines the dataset context.
*   **Grid Lines:** Horizontal dashed grid lines appear every 2.5 units (22.5, 25.0, 27.5, etc.) to assist in estimating specific Y-values.
*   **Box Components:**
    *   **Orange horizontal line:** Represents the median.
    *   **Box boundaries:** Represent the 1st Quartile (25th percentile) and 3rd Quartile (75th percentile).
    *   **Whiskers (T-bars):** Represent the minimum and maximum values (excluding outliers).
    *   **Circle:** Represents an outlier (a data point more than 1.5 times the IQR past the box).

### 5. Statistical Insights
*   **Central Tendency:** "Integration Readiness" statements are, on average (median), slightly longer (27 words) than "Trust Readiness" statements (25 words).
*   **Variability:** While "Trust Readiness" has a broader range within its whiskers (spanning ~13 words from min to max), the "Integration Readiness" has a slightly larger Interquartile Range (height of the box), suggesting that the "middle 50%" of Integration statements are more variable in length than the middle 50% of Trust statements.
*   **Skewness:**
    *   The "Trust Readiness" box shows a positive skew (the upper whisker is much longer than the lower whisker, and the median is closer to the bottom of the box). This means there is a tendency toward longer word counts in the upper range of this group.
    *   The "Integration Readiness" distribution is heavily influenced by the single extreme outlier at 39 words. Without the outlier, the data appears reasonably symmetric but slightly clustered toward the lower end of its box (median 27 vs Q1 23).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
