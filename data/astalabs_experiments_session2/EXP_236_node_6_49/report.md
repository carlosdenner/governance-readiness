# Experiment 236: node_6_49

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_49` |
| **ID in Run** | 236 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T12:04:47.048182+00:00 |
| **Runtime** | 263.0s |
| **Parent** | `node_5_17` |
| **Children** | None |
| **Creation Index** | 237 |

---

## Hypothesis

> Adversarial attacks in the ATLAS dataset that achieve the 'Impact' tactic
utilize a significantly higher average number of unique 'Techniques' than
attacks limited to 'Initial Access' or 'Discovery', implying greater complexity
is required for impact.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.4176 (Maybe False) |
| **Surprise** | -0.3892 |
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
| Maybe False | 60.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Quantify the tactical complexity (technique count) required to reach the 'Impact' stage in adversarial AI attacks.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` (ATLAS subset).
- 2. Parse the `tactics` column to identify cases containing 'Impact'.
- 3. Calculate the number of unique items in the `techniques` column for each case.
- 4. Compare the mean technique count for 'Impact' cases vs 'Non-Impact' cases using a T-test (or Mann-Whitney U test if non-normal).

### Deliverables
- Box plot of Technique Counts for Impact vs Non-Impact cases; T-test results.

---

## Analysis

The experiment was successfully executed using the corrected parsing logic for
the 'tactics' column (matching 'impact.id').

### Hypothesis Verification
The hypothesis that **'Impact' attacks utilize a significantly higher number of
techniques** was **not statistically supported** (p > 0.05).

*   **Observed Trend**: 'Impact' cases (n=36) did show a higher average
complexity (Mean = 7.97, Median = 7.5) compared to 'Non-Impact' cases (n=16,
Mean = 6.56, Median = 6.0). The box plot confirms a higher interquartile range
and upper whiskers for the Impact group.
*   **Statistical Significance**: Despite the visual trend, the difference was
not statistically significant.
    *   **Mann-Whitney U Test**: p = 0.1475
    *   **Welch's T-Test**: p = 0.0972
*   **Conclusion**: While adversarial attacks resulting in 'Impact' tend to be
more complex on average, the current sample size (N=52) is insufficient to rule
out random chance as the cause of this difference. The null hypothesis stands.

---

## Review

The experiment was successfully executed. The code correctly loaded the ATLAS
dataset and, after a necessary correction to the filtering logic (targeting
'impact.id' instead of the string 'Impact'), successfully segmented the cases
into 'Impact' (n=36) and 'Non-Impact' (n=16) groups.

### Hypothesis Verification:
The hypothesis that **attacks achieving the 'Impact' tactic utilize a
significantly higher number of techniques** was **not supported** by the
statistical analysis.

*   **Observed Trend**: There is a visible trend where 'Impact' cases exhibit
higher complexity (Mean = 7.97, Median = 7.5) compared to 'Non-Impact' cases
(Mean = 6.56, Median = 6.0). The box plot illustrates a wider interquartile
range and higher upper bound for the Impact group.
*   **Statistical Significance**: The difference failed to reach statistical
significance at the alpha = 0.05 level. The Mann-Whitney U test yielded a
p-value of **0.1475**, and Welch's T-test yielded a p-value of **0.0972**.
*   **Conclusion**: While the data suggests 'Impact' cases may be more complex,
the evidence is insufficient to reject the null hypothesis. The observed
difference could be due to random sampling variability within the small dataset
(N=52).

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if running in a different environment structure
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for ATLAS cases
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()

# Helper function to parse list-like strings
def parse_list_col(val):
    if pd.isna(val):
        return []
    if '|' in str(val):
        return [x.strip() for x in str(val).split('|') if x.strip()]
    return [str(val).strip()]

# Calculate technique counts
# The 'techniques' column follows the same pipe-separated format
atlas_df['technique_count'] = atlas_df['techniques'].apply(lambda x: len(set(parse_list_col(x))))

# Identify 'Impact' cases
# Debugging showed format is like {{impact.id}}
atlas_df['has_impact'] = atlas_df['tactics'].apply(lambda x: 'impact.id' in str(x).lower() if pd.notna(x) else False)

# Split groups
impact_counts = atlas_df[atlas_df['has_impact'] == True]['technique_count']
no_impact_counts = atlas_df[atlas_df['has_impact'] == False]['technique_count']

print(f"Group 'Impact': n={len(impact_counts)}, Mean={impact_counts.mean():.2f}, Median={impact_counts.median()}")
print(f"Group 'Non-Impact': n={len(no_impact_counts)}, Mean={no_impact_counts.mean():.2f}, Median={no_impact_counts.median()}")

# Statistical Test
# Mann-Whitney U is safer for small sample sizes and non-normal distributions
stat, p_val = stats.mannwhitneyu(impact_counts, no_impact_counts, alternative='two-sided')
print(f"\nMann-Whitney U Test: U={stat}, p-value={p_val:.4f}")

t_stat, t_p_val = stats.ttest_ind(impact_counts, no_impact_counts, equal_var=False)
print(f"Welch's T-Test: t={t_stat:.4f}, p-value={t_p_val:.4f}")

if p_val < 0.05:
    print("\nResult: Statistically significant difference found.")
else:
    print("\nResult: No statistically significant difference found.")

# Visualization
plt.figure(figsize=(8, 6))
data_to_plot = [no_impact_counts, impact_counts]
plt.boxplot(data_to_plot, tick_labels=['Non-Impact', 'Impact'])
plt.title('Technique Complexity: Impact vs. Non-Impact Cases')
plt.ylabel('Number of Unique Techniques')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Group 'Impact': n=36, Mean=7.97, Median=7.5
Group 'Non-Impact': n=16, Mean=6.56, Median=6.0

Mann-Whitney U Test: U=361.0, p-value=0.1475
Welch's T-Test: t=1.7077, p-value=0.0972

Result: No statistically significant difference found.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is a detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Box Plot (also known as a Box-and-Whisker Plot).
*   **Purpose:** This plot is used to visually compare the distribution of numerical data—specifically the "Number of Unique Techniques"—across two distinct categorical groups: "Non-Impact" and "Impact." It displays the data's central tendency (median), dispersion (interquartile range), and outliers.

### 2. Axes
*   **X-Axis:**
    *   **Labels:** The axis represents two categories: **"Non-Impact"** and **"Impact"**.
    *   **Type:** Categorical (Nominal).
*   **Y-Axis:**
    *   **Title:** "Number of Unique Techniques".
    *   **Units:** Count (integer values).
    *   **Range:** The axis ticks range from **2 to 16**, though the data points extend from **1 to 16**.

### 3. Data Trends
*   **Non-Impact Cases:**
    *   **Median:** The median (orange line) is located at approximately **6**.
    *   **Interquartile Range (IQR):** The box (representing the middle 50% of data) spans from roughly **6 to 8**.
    *   **Whiskers:** The lower whisker extends to **3**, and the upper whisker extends to **9**.
    *   **Outliers:** There are distinct outliers visible as circles at values **1** and **12**.
*   **Impact Cases:**
    *   **Median:** The median is higher than the Non-Impact group, situated at approximately **7.5**.
    *   **Interquartile Range (IQR):** The box is taller, indicating more variation in the middle 50%, spanning from roughly **6 to 9.2**.
    *   **Whiskers:** The lower whisker extends to **3**, but the upper whisker extends significantly higher to **14**.
    *   **Outliers:** There are high-value outliers at **15** and **16**.

### 4. Annotations and Legends
*   **Title:** "Technique Complexity: Impact vs. Non-Impact Cases" appears at the top.
*   **Grid Lines:** Horizontal dashed grey lines are provided at intervals of 2 to assist in reading the Y-axis values.
*   **Visual Indicators:**
    *   **Orange Lines:** Represent the median of each dataset.
    *   **Circles:** Represent outliers (data points that fall beyond 1.5 times the IQR from the quartiles).

### 5. Statistical Insights
*   **Higher Complexity in Impact Cases:** The "Impact" group demonstrates a higher median complexity compared to the "Non-Impact" group (approx. 7.5 vs. 6). This suggests that cases resulting in an impact generally involve a higher number of unique techniques.
*   **Greater Variability in Impact Cases:** The "Impact" category shows a much wider spread of data. The distance between the whiskers (3 to 14) and the presence of extreme outliers (up to 16) indicates that while some impact cases are simple, there is a significant potential for high complexity.
*   **Skewness:** Both distributions appear positively skewed (skewed towards higher numbers), but the skew is more pronounced in the "Impact" cases, as evidenced by the long upper whisker and high outliers.
*   **Baseline Similarity:** Interestingly, the lower quartiles (bottom of the boxes) and lower whiskers are relatively similar (flooring around 3 to 6). This implies that low-complexity events can result in either Impact or Non-Impact outcomes; however, as complexity increases, the likelihood or association with "Impact" outcomes appears to rise.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
