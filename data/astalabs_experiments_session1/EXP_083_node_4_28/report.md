# Experiment 83: node_4_28

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_28` |
| **ID in Run** | 83 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:12:08.020205+00:00 |
| **Runtime** | 301.3s |
| **Parent** | `node_3_9` |
| **Children** | None |
| **Creation Index** | 84 |

---

## Hypothesis

> Incidents dominated by 'Integration Readiness' gaps involve a significantly
higher number of distinct attack techniques than incidents dominated by 'Trust
Readiness' gaps.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6855 (Maybe True) |
| **Posterior** | 0.6281 (Maybe True) |
| **Surprise** | -0.0666 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 25.0 |
| Uncertain | 3.0 |
| Maybe False | 2.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 39.0 |
| Uncertain | 51.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Assess whether engineering failures involve more complex attack chains than governance failures.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Filter for rows where 'trust_integration_split' is 'Integration-Dominant' or 'Trust-Dominant'.
- 3. Compare the 'technique_count' variable between these two groups using a T-test.

### Deliverables
- Boxplot of Technique Count by Competency Split; T-test results.

---

## Analysis

The experiment successfully analyzed the complexity of attack chains (measured
by 'technique_count') across different competency gap types.

**Findings:**
1.  **Sample Size Limitations:** The majority of incidents in the dataset are
classified as 'Both'. The 'pure' categories are small: 'Integration-Dominant'
(n=4) and 'Trust-Dominant' (n=2).
2.  **Descriptive Statistics:** The hypothesis appears directionally correct.
Incidents dominated by engineering gaps ('Integration-Dominant') involved a
higher average number of attack techniques (Mean = 8.25, SD = 1.71) compared to
governance gaps ('Trust-Dominant') (Mean = 5.50, SD = 0.71).
3.  **Statistical Significance:** The results are mixed due to the small sample
size.
    -   **Welch's T-test:** Yielded a p-value of **0.0499**, just barely
crossing the threshold for statistical significance (< 0.05).
    -   **Mann-Whitney U Test:** Yielded a p-value of **0.1588**, failing to
find significance. Given the extremely small sample sizes (N=4, N=2), the non-
parametric U-test is generally more robust, suggesting the T-test result may be
a false positive driven by the magnitude of difference rather than statistical
reliability.

**Conclusion:** While there is a strong qualitative signal that engineering-
related failures involve more complex attack chains (nearly 3 additional
techniques on average), the statistical evidence is inconclusive due to the
scarcity of incidents that are purely one type or the other.

---

## Review

The experiment successfully analyzed the complexity of attack chains (measured
by 'technique_count') across different competency gap types.

**Findings:**
1.  **Sample Size Limitations:** The majority of incidents in the dataset are
classified as 'Both'. The 'pure' categories are small: 'Integration-Dominant'
(n=4) and 'Trust-Dominant' (n=2).
2.  **Descriptive Statistics:** The hypothesis appears directionally correct.
Incidents dominated by engineering gaps ('Integration-Dominant') involved a
higher average number of attack techniques (Mean = 8.25, SD = 1.71) compared to
governance gaps ('Trust-Dominant') (Mean = 5.50, SD = 0.71).
3.  **Statistical Significance:** The results are mixed due to the small sample
size.
    -   **Welch's T-test:** Yielded a p-value of **0.0499**, just barely
crossing the threshold for statistical significance (< 0.05).
    -   **Mann-Whitney U Test:** Yielded a p-value of **0.1588**, failing to
find significance. Given the extremely small sample sizes (N=4, N=2), the non-
parametric U-test is generally more robust, suggesting the T-test result may be
a false positive driven by the magnitude of difference rather than statistical
reliability.

**Conclusion:** While there is a strong qualitative signal that engineering-
related failures involve more complex attack chains (nearly 3 additional
techniques on average), the statistical evidence is inconclusive due to the
scarcity of incidents that are purely one type or the other.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import sys

# Load the dataset
file_name = 'step3_incident_coding.csv'
try:
    df = pd.read_csv(file_name)
    print(f"Successfully loaded {file_name} with shape {df.shape}")
except FileNotFoundError:
    print(f"Error: {file_name} not found.")
    sys.exit(1)

# Normalize the column to handle potential casing issues
if 'trust_integration_split' in df.columns:
    # Print original unique values for debugging
    print("Original unique values in 'trust_integration_split':")
    print(df['trust_integration_split'].unique())
    
    # Normalize to lowercase and strip whitespace
    df['trust_integration_split_norm'] = df['trust_integration_split'].astype(str).str.lower().str.strip()
else:
    print("Column 'trust_integration_split' not found.")
    sys.exit(1)

# Define target groups (normalized)
target_groups = ['integration-dominant', 'trust-dominant']
subset = df[df['trust_integration_split_norm'].isin(target_groups)].copy()

# Display counts to verify sample size
print("\n--- Sample Counts (Normalized) ---")
counts = subset['trust_integration_split_norm'].value_counts()
print(counts)

# Extract vectors for statistical testing
integration_scores = subset[subset['trust_integration_split_norm'] == 'integration-dominant']['technique_count']
trust_scores = subset[subset['trust_integration_split_norm'] == 'trust-dominant']['technique_count']

# Descriptive Statistics
print("\n--- Descriptive Statistics for Technique Count ---")
if not subset.empty:
    desc_stats = subset.groupby('trust_integration_split_norm')['technique_count'].describe()
    print(desc_stats)
else:
    print("No data found for the specified groups after normalization.")

# Statistical Testing
print("\n--- Statistical Test Results ---")
if len(integration_scores) > 1 and len(trust_scores) > 1:
    # Welch's T-test (does not assume equal variance)
    t_stat, p_val = stats.ttest_ind(integration_scores, trust_scores, equal_var=False)
    print(f"Welch's T-test: Statistic={t_stat:.4f}, p-value={p_val:.4f}")
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_p_val = stats.mannwhitneyu(integration_scores, trust_scores)
    print(f"Mann-Whitney U test: Statistic={u_stat:.4f}, p-value={u_p_val:.4f}")
    
    alpha = 0.05
    if p_val < alpha:
        print("Result: Statistically significant difference found (p < 0.05).")
    else:
        print("Result: No statistically significant difference found (p >= 0.05).")
else:
    print("Insufficient sample size for statistical testing (need at least 2 per group).")
    print(f"Integration-Dominant count: {len(integration_scores)}")
    print(f"Trust-Dominant count: {len(trust_scores)}")

# Visualization
if not subset.empty:
    plt.figure(figsize=(8, 6))
    # Map normalized names back to Title Case for display
    subset['Display Label'] = subset['trust_integration_split_norm'].map({
        'integration-dominant': 'Integration-Dominant',
        'trust-dominant': 'Trust-Dominant'
    })
    
    # Boxplot to show distribution
    sns.boxplot(x='Display Label', y='technique_count', data=subset, palette="Set2", order=['Trust-Dominant', 'Integration-Dominant'])
    # Swarmplot to show individual data points
    sns.swarmplot(x='Display Label', y='technique_count', data=subset, color=".25", size=8, order=['Trust-Dominant', 'Integration-Dominant'])
    
    plt.title('Attack Technique Count by Competency Gap Type')
    plt.ylabel('Number of Distinct Techniques')
    plt.xlabel('Competency Gap Dominance')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Successfully loaded step3_incident_coding.csv with shape (52, 22)
Original unique values in 'trust_integration_split':
<StringArray>
['both', 'trust-dominant', 'integration-dominant']
Length: 3, dtype: str

--- Sample Counts (Normalized) ---
trust_integration_split_norm
integration-dominant    4
trust-dominant          2
Name: count, dtype: int64

--- Descriptive Statistics for Technique Count ---
                              count  mean       std  min   25%  50%   75%   max
trust_integration_split_norm                                                   
integration-dominant            4.0  8.25  1.707825  6.0  7.50  8.5  9.25  10.0
trust-dominant                  2.0  5.50  0.707107  5.0  5.25  5.5  5.75   6.0

--- Statistical Test Results ---
Welch's T-test: Statistic=2.7791, p-value=0.0499
Mann-Whitney U test: Statistic=7.5000, p-value=0.1588
Result: Statistically significant difference found (p < 0.05).

STDERR:
<ipython-input-1-8d673a31e4cf>:83: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(x='Display Label', y='technique_count', data=subset, palette="Set2", order=['Trust-Dominant', 'Integration-Dominant'])


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the analysis:

### 1. Plot Type
*   **Type:** This is a **box plot** (also known as a box-and-whisker plot) with overlaid data points.
*   **Purpose:** It is used to compare the distribution of a quantitative variable ("Number of Distinct Techniques") across two categorical groups ("Trust-Dominant" and "Integration-Dominant"). It visualizes the median, interquartile range (IQR), and the overall spread (min/max) of the data.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Competency Gap Dominance"
    *   **Categories:** Two categories are displayed: "Trust-Dominant" and "Integration-Dominant".
*   **Y-Axis:**
    *   **Label:** "Number of Distinct Techniques"
    *   **Range:** The visible axis ticks range from 5 to 10, representing integer counts.
    *   **Grid:** Horizontal dashed grid lines are placed at each integer value (5, 6, 7, etc.) to assist with readability.

### 3. Data Trends
*   **Trust-Dominant (Left Group):**
    *   **Range:** The data is tightly clustered between 5 and 6.
    *   **Distribution:** This group shows very low variability. The box (representing the middle 50% of data) is narrow, situated roughly between 5.25 and 5.75.
    *   **Median:** The median line is centered at roughly 5.5.
*   **Integration-Dominant (Right Group):**
    *   **Range:** The data is spread much wider, ranging from a minimum of 6 to a maximum of 10.
    *   **Distribution:** This group shows high variability. The box spans from approximately 7.5 (25th percentile) to roughly 9.25 (75th percentile).
    *   **Median:** The median is significantly higher, located at approximately 8.5.

### 4. Annotations and Legends
*   **Title:** "Attack Technique Count by Competency Gap Type" appears at the top.
*   **Color Coding:**
    *   **Teal/Green Box:** Represents the "Trust-Dominant" category.
    *   **Orange/Salmon Box:** Represents the "Integration-Dominant" category.
*   **Data Points:** Dark circular markers are superimposed on the whiskers and boxes (e.g., at 5, 6, 8, 9, 10). These likely represent the actual observed data points or specific outliers, indicating that the underlying data consists of discrete integer counts.

### 5. Statistical Insights
*   **Significant Difference in Magnitude:** There is a clear separation between the two groups. The "Integration-Dominant" gaps are associated with a significantly higher number of distinct attack techniques compared to "Trust-Dominant" gaps. The lowest value for Integration-Dominant (6) is equal to the highest value for Trust-Dominant.
*   **Variance Discrepancy:** The "Trust-Dominant" group is highly consistent (low variance), suggesting predictable behavior where the number of techniques is almost always 5 or 6. In contrast, the "Integration-Dominant" group is much more volatile, with attack technique counts varying widely between 6 and 10.
*   **Conclusion:** The plot suggests that security vulnerabilities or gaps classified as "Integration-Dominant" are more complex or severe in terms of the variety of attack techniques they attract or facilitate compared to "Trust-Dominant" gaps.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
