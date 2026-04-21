# Experiment 208: node_6_36

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_36` |
| **ID in Run** | 208 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:44:53.577775+00:00 |
| **Runtime** | 181.1s |
| **Parent** | `node_5_52` |
| **Children** | None |
| **Creation Index** | 209 |

---

## Hypothesis

> Adversarial Complexity: In ATLAS cases, adversarial attacks that achieve the
'Impact' tactic (successful damage/disruption) utilize a significantly higher
number of unique 'Tactics' (chain length) compared to attacks that only reach
'Discovery' or 'Evasion' stages.

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

**Objective:** Analyze if successful impact requires a more complex kill chain.

### Steps
- 1. Filter 'atlas_cases'.
- 2. Parse the 'tactics' column to count the number of pipe-delimited items per row (Chain Length).
- 3. Create a binary variable 'Achieved_Impact': True if 'Impact' is present in the 'tactics' string, False otherwise.
- 4. Compare the mean Chain Length between the Impact and Non-Impact groups using a Mann-Whitney U test.

### Deliverables
- Mean chain lengths, Mann-Whitney U test results, and a boxplot of chain length by Impact status.

---

## Analysis

The experiment successfully analyzed 52 adversarial cases from the MITRE ATLAS
dataset to determine if successful attacks (achieving 'Impact') require more
complex kill chains than unsuccessful ones. The analysis parsed the 'tactics'
column to calculate chain length and categorized cases based on the presence of
the 'Impact' tactic.

**Findings:**
1.  **Descriptive Statistics:** Attacks achieving 'Impact' (n=36) exhibited a
higher mean chain length of 5.97 tactics (Median: 6.0) compared to those that
halted at Discovery or Evasion (n=16), which had a mean of 5.12 tactics (Median:
5.5).
2.  **Distribution:** The boxplot visualization reveals that successful attacks
have a wider range of complexity (up to 11 tactics) compared to unsuccessful
ones (max 8 tactics). It also suggests a potential minimum threshold, as no
attacks with a chain length of 1 achieved impact.
3.  **Statistical Significance:** Despite the observed trend where higher
complexity correlates with success, the Mann-Whitney U test returned a p-value
of 0.1276. This exceeds the significance threshold of 0.05.

**Conclusion:** The hypothesis is **not supported**. While the data suggests a
trend that successful attacks utilize slightly more complex kill chains, the
difference is not statistically significant given the current sample size.

---

## Review

The experiment was faithfully implemented according to the plan. The code
correctly loaded the dataset, filtered for ATLAS cases, parsed the 'tactics'
column to compute chain length, and categorized cases based on the presence of
the 'Impact' tactic. A Mann-Whitney U test was performed as requested, and a
boxplot was generated to visualize the distributions. The analysis correctly
interpreted the statistical results (p-value > 0.05) to conclude that the
hypothesis was not supported, despite a visible trend in the means.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import os

def analyze_adversarial_complexity():
    # Load dataset
    file_path = '../astalabs_discovery_all_data.csv'
    if not os.path.exists(file_path):
        file_path = 'astalabs_discovery_all_data.csv' # Fallback for local testing if needed

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}")
        return

    # Filter for ATLAS cases
    atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
    
    if atlas_df.empty:
        print("No ATLAS cases found in the dataset.")
        return

    print(f"Loaded {len(atlas_df)} ATLAS cases.")

    # Determine delimiter by inspecting a sample
    sample_tactics = atlas_df['tactics'].dropna().iloc[0] if not atlas_df['tactics'].dropna().empty else ""
    delimiter = '|' if '|' in str(sample_tactics) else ','
    print(f"Detected delimiter for tactics: '{delimiter}'")

    # Function to parse tactics and count length
    def parse_and_count(tactic_str):
        if pd.isna(tactic_str):
            return 0, False
        
        # Normalize and split
        t_str = str(tactic_str)
        items = [x.strip() for x in t_str.split(delimiter) if x.strip()]
        
        # Count unique tactics
        unique_items = list(set(items))
        chain_length = len(unique_items)
        
        # Check for 'Impact' tactic (case-insensitive)
        has_impact = any('impact' in item.lower() for item in unique_items)
        
        return chain_length, has_impact

    # Apply processing
    # Result is a DataFrame with two columns, which we assign back
    results = atlas_df['tactics'].apply(lambda x: parse_and_count(x))
    atlas_df['chain_length'] = results.apply(lambda x: x[0])
    atlas_df['achieved_impact'] = results.apply(lambda x: x[1])

    # Separate groups
    impact_group = atlas_df[atlas_df['achieved_impact'] == True]['chain_length']
    no_impact_group = atlas_df[atlas_df['achieved_impact'] == False]['chain_length']

    # Statistics
    n_impact = len(impact_group)
    n_no_impact = len(no_impact_group)
    
    print(f"Cases achieving Impact: {n_impact}")
    print(f"Cases NOT achieving Impact: {n_no_impact}")
    
    if n_impact == 0 or n_no_impact == 0:
        print("Cannot perform statistical test: One of the groups is empty.")
        return

    mean_impact = impact_group.mean()
    mean_no_impact = no_impact_group.mean()
    median_impact = impact_group.median()
    median_no_impact = no_impact_group.median()
    
    print(f"\nMean Chain Length (Impact): {mean_impact:.2f} (Median: {median_impact})")
    print(f"Mean Chain Length (No Impact): {mean_no_impact:.2f} (Median: {median_no_impact})")

    # Mann-Whitney U Test (Impact > No Impact)
    stat, p_value = mannwhitneyu(impact_group, no_impact_group, alternative='greater')
    print(f"\nMann-Whitney U Test results:")
    print(f"Statistic: {stat}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Conclusion: Statistically significant difference (p < 0.05). Hypothesis SUPPORTED.")
    else:
        print("Conclusion: No statistically significant difference (p >= 0.05). Hypothesis NOT SUPPORTED.")

    # Visualization
    plt.figure(figsize=(10, 6))
    # Create boxplot
    bp = plt.boxplot([no_impact_group, impact_group], 
                     labels=['No Impact', 'Impact Achieved'],
                     patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Adversarial Kill Chain Complexity: Impact vs. No Impact')
    plt.ylabel('Number of Unique Tactics (Chain Length)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add swarm plot or jitter for individual points if N is small
    y_no = no_impact_group
    x_no = np.random.normal(1, 0.04, size=len(y_no))
    plt.scatter(x_no, y_no, alpha=0.6, color='blue', s=20)

    y_yes = impact_group
    x_yes = np.random.normal(2, 0.04, size=len(y_yes))
    plt.scatter(x_yes, y_yes, alpha=0.6, color='green', s=20)

    plt.show()

if __name__ == "__main__":
    analyze_adversarial_complexity()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded 52 ATLAS cases.
Detected delimiter for tactics: '|'
Cases achieving Impact: 36
Cases NOT achieving Impact: 16

Mean Chain Length (Impact): 5.97 (Median: 6.0)
Mean Chain Length (No Impact): 5.12 (Median: 5.5)

Mann-Whitney U Test results:
Statistic: 345.0
P-value: 0.1276
Conclusion: No statistically significant difference (p >= 0.05). Hypothesis NOT SUPPORTED.

STDERR:
<ipython-input-1-0f5787c57a19>:97: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  bp = plt.boxplot([no_impact_group, impact_group],


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **Box Plot (Box-and-Whisker Plot) with an overlaid Jitter/Strip Plot**.
*   **Purpose:** The plot compares the distribution of a quantitative variable (chain length) across two categorical groups ("No Impact" vs. "Impact Achieved").
    *   The **box plot** component visualizes the five-number summary: minimum, first quartile (Q1), median, third quartile (Q3), and maximum.
    *   The **overlaid dots** show the individual data points, allowing the viewer to see the sample size and specific distribution density, which helps identify if the data is clustered or sparse.

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Number of Unique Tactics (Chain Length)".
    *   **Units:** Count (integer values representing the number of tactics).
    *   **Range:** The visual axis spans from roughly **0 to 11**. Grid lines are marked at intervals of 2 (2, 4, 6, 8, 10).
*   **X-Axis:**
    *   **Label:** Represents the Outcome of the attack.
    *   **Categories:** Two distinct categories: **"No Impact"** (left) and **"Impact Achieved"** (right).

### 3. Data Trends
*   **No Impact (Left/Blue Group):**
    *   **Range:** The data spans from a minimum of **1** to a maximum of **8**.
    *   **Concentration:** The bulk of the data (the box) is concentrated between **4 and 6.25**.
    *   **Clustering:** There is a noticeable cluster of data points around the median (approx. 5.5).
*   **Impact Achieved (Right/Green Group):**
    *   **Range:** The data has a wider span, ranging from a minimum of **2** to a maximum of **11**.
    *   **Concentration:** The interquartile range (the box) is positioned higher, roughly between **5 and 7**.
    *   **Outliers:** There is a clear outlier indicated by a hollow circle at the very top (value **11**), along with a data point at the same level.
*   **Comparison:** The "Impact Achieved" group generally exhibits higher values and a larger spread (variance) than the "No Impact" group.

### 4. Annotations and Legends
*   **Title:** "Adversarial Kill Chain Complexity: Impact vs. No Impact". This sets the context that the data relates to cybersecurity attack patterns.
*   **Color Coding:**
    *   **Blue/Light Blue:** Represents the "No Impact" category.
    *   **Green/Light Green:** Represents the "Impact Achieved" category.
*   **Grid Lines:** Horizontal dashed grey lines are provided at intervals of 2 to assist in estimating the Y-axis values.

### 5. Statistical Insights
*   **Correlation between Complexity and Success:** There appears to be a positive correlation between the complexity of the kill chain (number of tactics) and the likelihood of achieving an impact. The median for "Impact Achieved" (approx. 6) is higher than the median for "No Impact" (approx. 5.5).
*   **Variability:** Successful attacks ("Impact Achieved") show greater variability. While impact can be achieved with as few as 2 tactics, the group also contains the most complex chains (up to 11 tactics). In contrast, the "No Impact" group is more tightly clustered and caps out at 8 tactics.
*   **Minimum Threshold:** The data suggests that a chain length of 1 is insufficient for impact (as the lowest value for "Impact Achieved" is 2, while "No Impact" has a point at 1).
*   **Upper Extremes:** High complexity (chains of length 9, 10, or 11) is almost exclusively associated with successful impacts, suggesting that highly complex attacks are rarely stopped without some impact occurring (or that complexity is required to bypass defenses).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
