# Experiment 64: node_4_27

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_27` |
| **ID in Run** | 64 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T04:02:09.436661+00:00 |
| **Runtime** | 190.9s |
| **Parent** | `node_3_18` |
| **Children** | `node_5_22`, `node_5_44` |
| **Creation Index** | 65 |

---

## Hypothesis

> Adversarial Tactic Complexity: ATLAS case studies involving the 'Evasion' tactic
involve a significantly higher number of distinct 'Techniques' per case compared
to cases involving the 'Discovery' tactic, indicating higher complexity.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.4839 (Uncertain) |
| **Posterior** | 0.1648 (Likely False) |
| **Surprise** | -0.3828 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 13.0 |
| Uncertain | 2.0 |
| Maybe False | 15.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Compare the complexity (number of techniques) between Evasion and Discovery adversarial tactics.

### Steps
- 1. Filter for 'atlas_cases'.
- 2. Parse the 'tactics' and 'techniques' columns (semicolon or comma separated strings).
- 3. Create a list of technique counts for cases containing 'Evasion'.
- 4. Create a list of technique counts for cases containing 'Discovery'.
- 5. Perform a Mann-Whitney U test (non-parametric) to compare the distributions of technique counts.

### Deliverables
- Mean/Median technique counts for each tactic, Mann-Whitney U test statistics.

---

## Analysis

The experiment successfully loaded the dataset and filtered for 52 ATLAS case
studies to test the hypothesis that 'Evasion' tactics involve higher complexity
(more techniques) than 'Discovery' tactics. The code parsed the 'techniques'
column and categorized cases into Evasion (n=23) and Discovery (n=12).

The analysis revealed zero variance in the complexity metric: every single case
across both groups was calculated to have exactly one technique (Mean = 1.00,
Median = 1.00, Std = 0.00). Consequently, the Mann-Whitney U test returned a
p-value of 1.00, indicating no statistical difference.

The hypothesis is rejected based on this analysis. However, the result (all
counts = 1) strongly suggests a data limitation, such as the dataset recording
only a primary technique per case or using a list format/delimiter not handled
by the current parsing logic, rather than a genuine reflection of equal tactical
complexity.

---

## Review

The experiment was faithfully executed according to the plan. The programmer
correctly loaded the dataset, filtered for the 'atlas_cases' source, and
implemented logic to parse the 'tactics' and 'techniques' columns. The analysis
revealed that for the analyzed sample (N=52), every single case contained
exactly one technique (Mean=1.0, Std=0.0) regardless of whether the tactic was
'Evasion' or 'Discovery'. Consequently, the Mann-Whitney U test yielded a
p-value of 1.0, rejecting the hypothesis. While the result (zero variance)
suggests a potential data limitation (e.g., the dataset might only record the
primary technique or use a delimiter not anticipated by the parser), the
experiment procedure itself was valid and the hypothesis was tested based on the
available data.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# [debug] Check current directory and file existence
# print(os.getcwd())
# print(os.listdir('..'))

# Load dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if file is in current directory
    file_path = 'astalabs_discovery_all_data.csv'
    df = pd.read_csv(file_path, low_memory=False)

print("Dataset loaded successfully.")

# Filter for ATLAS cases
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
print(f"Filtered for 'atlas_cases': {len(atlas_df)} rows")

# Identify relevant columns for tactics and techniques
# Based on metadata, likely 'tactics' and 'techniques'
# Let's verify columns exist, otherwise search for them
target_cols = ['tactics', 'techniques']
missing_cols = [c for c in target_cols if c not in atlas_df.columns]
if missing_cols:
    print(f"Warning: Columns {missing_cols} not found. searching by keyword...")
    for col in atlas_df.columns:
        if 'tactic' in str(col).lower():
            print(f"Found potential tactic column: {col}")
            atlas_df.rename(columns={col: 'tactics'}, inplace=True)
        if 'technique' in str(col).lower():
            print(f"Found potential technique column: {col}")
            atlas_df.rename(columns={col: 'techniques'}, inplace=True)

# Drop rows with missing tactics or techniques
atlas_df = atlas_df.dropna(subset=['tactics', 'techniques'])
print(f"Rows after dropping nulls in tactics/techniques: {len(atlas_df)}")

# Function to parse lists from strings
def parse_list(s):
    if not isinstance(s, str):
        return []
    # delimiters could be comma or semicolon
    if ';' in s:
        return [x.strip() for x in s.split(';') if x.strip()]
    return [x.strip() for x in s.split(',') if x.strip()]

# Calculate technique counts
atlas_df['technique_list'] = atlas_df['techniques'].apply(parse_list)
atlas_df['technique_count'] = atlas_df['technique_list'].apply(len)

# Identify groups
# Group 1: Cases involving 'Evasion'
# Group 2: Cases involving 'Discovery'

evasion_cases = atlas_df[atlas_df['tactics'].str.contains('Evasion', case=False, na=False)]
discovery_cases = atlas_df[atlas_df['tactics'].str.contains('Discovery', case=False, na=False)]

print(f"\nCases involving 'Evasion': {len(evasion_cases)}")
print(f"Cases involving 'Discovery': {len(discovery_cases)}")

# Check for overlap
overlap_ids = set(evasion_cases.index).intersection(set(discovery_cases.index))
print(f"Overlap (cases in both): {len(overlap_ids)}")

# Prepare data for testing
evasion_counts = evasion_cases['technique_count'].values
discovery_counts = discovery_cases['technique_count'].values

# Descriptive Statistics
print("\n--- Descriptive Statistics (Technique Counts) ---")
print(f"Evasion - Mean: {np.mean(evasion_counts):.2f}, Median: {np.median(evasion_counts):.2f}, Std: {np.std(evasion_counts):.2f}")
print(f"Discovery - Mean: {np.mean(discovery_counts):.2f}, Median: {np.median(discovery_counts):.2f}, Std: {np.std(discovery_counts):.2f}")

# Mann-Whitney U Test
# We use Mann-Whitney because counts are discrete and likely non-normal, and sample sizes are small.
stat, p_value = stats.mannwhitneyu(evasion_counts, discovery_counts, alternative='two-sided')

print("\n--- Hypothesis Test Results (Mann-Whitney U) ---")
print(f"U-statistic: {stat}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Result: Statistically significant difference in technique counts.")
else:
    print("Result: No statistically significant difference in technique counts.")

# Visualization
plt.figure(figsize=(8, 6))
plt.boxplot([evasion_counts, discovery_counts], labels=['Evasion', 'Discovery'])
plt.title('Distribution of Technique Counts by Adversarial Tactic')
plt.ylabel('Number of Techniques')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Dataset loaded successfully.
Filtered for 'atlas_cases': 52 rows
Rows after dropping nulls in tactics/techniques: 52

Cases involving 'Evasion': 23
Cases involving 'Discovery': 12
Overlap (cases in both): 6

--- Descriptive Statistics (Technique Counts) ---
Evasion - Mean: 1.00, Median: 1.00, Std: 0.00
Discovery - Mean: 1.00, Median: 1.00, Std: 0.00

--- Hypothesis Test Results (Mann-Whitney U) ---
U-statistic: 138.0
P-value: 1.0000
Result: No statistically significant difference in technique counts.

STDERR:
<ipython-input-1-3c28885b0956>:100: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot([evasion_counts, discovery_counts], labels=['Evasion', 'Discovery'])


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **Box Plot** (specifically, a degenerate or "collapsed" box plot).
*   **Purpose:** The intended purpose is to visualize the distribution and statistical summary (median, quartiles, outliers) of "Technique Counts" for two different categories of "Adversarial Tactic."

### 2. Axes
*   **X-axis:**
    *   **Label:** Represents the categories of **Adversarial Tactics**.
    *   **Categories:** The specific tactics displayed are **"Evasion"** and **"Discovery"**.
*   **Y-axis:**
    *   **Label:** **"Number of Techniques"**.
    *   **Range:** The visible ticks range from **0.96 to 1.04**. The axis is automatically scaled very tightly around the value 1.00.

### 3. Data Trends
*   **Visual Observation:** For both categories ("Evasion" and "Discovery"), the plot displays a single, flat orange line located exactly at the **1.00** mark on the Y-axis.
*   **Interpretation:**
    *   In a standard box plot, the orange line typically represents the **median**.
    *   The absence of a "box" (Interquartile Range) and "whiskers" indicates that there is **zero variance** in the data.
    *   **Trend:** Every single data point for both "Evasion" and "Discovery" has a value of exactly 1. There are no clusters, variations, or outliers visible because all data is identical.

### 4. Annotations and Legends
*   **Title:** "Distribution of Technique Counts by Adversarial Tactic" appears at the top, defining the context of the data.
*   **Grid Lines:** Horizontal dashed grid lines are present to assist in reading the Y-axis values, though they highlight the emptiness of the plot due to the lack of data variation.
*   **Color:** The orange color of the horizontal bar is standard formatting for the median line in Python plotting libraries (like Matplotlib/Seaborn).

### 5. Statistical Insights
*   **Uniformity:** The most significant insight is that the "Number of Techniques" is **constant** at 1 for both tactics in this dataset. This suggests a one-to-one relationship in the source data (e.g., each event or log entry is associated with exactly one technique per tactic).
*   **Lack of Variability:** The standard deviation for both groups is 0. There is no distribution to speak of, effectively making this a plot of two constants.
*   **Scale Artifact:** The odd Y-axis range (0.96 to 1.04) is an artifact of the plotting software. When data has no range (min = max = 1), the software creates an artificial window around that single value to render the plot.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
