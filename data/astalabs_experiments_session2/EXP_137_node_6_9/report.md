# Experiment 137: node_6_9

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_9` |
| **ID in Run** | 137 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:30:00.083569+00:00 |
| **Runtime** | 215.5s |
| **Parent** | `node_5_33` |
| **Children** | `node_7_14` |
| **Creation Index** | 138 |

---

## Hypothesis

> The Contractor Transparency Friction: AI systems developed via 'Contracting
Resources' are significantly less likely to provide public 'AI Notice' compared
to systems developed 'In-house', suggesting that outsourcing hinders
transparency.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6810 (Maybe True) |
| **Posterior** | 0.2444 (Likely False) |
| **Surprise** | -0.5240 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 22.0 |
| Uncertain | 5.0 |
| Maybe False | 1.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 8.0 |
| Definitely False | 52.0 |

---

## Experiment Plan

**Objective:** Determine if development method (In-house vs Contracted) affects the likelihood of providing AI Notice.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'eo13960_scored'.
- 2. Clean '22_dev_method' to categorize into 'In-house' vs 'Contracted'.
- 3. Clean '59_ai_notice' into a binary 'Yes'/'No' flag.
- 4. Calculate the proportion of systems providing notice in each group.
- 5. Perform a Chi-Square test of independence.

### Deliverables
- Contingency table of Development Method vs. AI Notice; Bar chart; Statistical test results.

---

## Analysis

The experiment successfully tested the 'Contractor Transparency Friction'
hypothesis using the EO 13960 dataset. After filtering for records with
definitive 'In-house' vs. 'Contracted' development methods and valid 'AI Notice'
entries, the study analyzed a subset of 50 systems (25 In-house, 25 Contracted).
The results revealed a universal lack of transparency: 100% of systems in both
groups failed to provide public AI Notice (0% compliance). A Chi-Square test
yielded a p-value of 1.0, confirming no statistical difference between the
groups. Consequently, the hypothesis that contracting resources specifically
hinder transparency is not supported; instead, the data suggests a systemic
failure to provide AI Notice across the board for this subset of federal AI
systems.

---

## Review

The experiment successfully tested the hypothesis using the available data,
though the sample size (n=50) was significantly reduced due to missing values in
the '22_dev_method' and '59_ai_notice' columns. The analysis correctly
identified that among the records where development method and notice status
were both populated, there was zero compliance with AI Notice requirements
regardless of the development method.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Dynamic file path resolution
filename = 'astalabs_discovery_all_data.csv'
filepath = filename
if not os.path.exists(filepath):
    if os.path.exists('../' + filename):
        filepath = '../' + filename
    else:
        print(f"Warning: {filename} not found in current or parent directory. Attempting current directory default.")

print(f"Loading dataset from {filepath}...")
try:
    df = pd.read_csv(filepath, low_memory=False)
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit(1)

# Filter for EO 13960 scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 records loaded: {len(eo_df)}")

# Target Columns
col_dev = '22_dev_method'
col_notice = '59_ai_notice'

# Normalize and clean '22_dev_method'
# We want to distinguish 'In-house' vs 'Contracted'
def clean_dev_method(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower()
    # Check for both/hybrid first if necessary, but hypothesis compares the two distinct groups.
    # Common logic: if it mentions contract, it involves contractors.
    # If it mentions in-house ONLY, it's in-house.
    if 'contract' in s:
        # specific check for hybrid could go here, but let's classify as 'Contracted' for now or 'Hybrid'
        if 'in-house' in s or 'government' in s:
             return 'Hybrid/Both'
        return 'Contracted'
    if 'in-house' in s or 'government' in s:
        return 'In-house'
    return 'Other'

eo_df['dev_category'] = eo_df[col_dev].apply(clean_dev_method)

# Normalize and clean '59_ai_notice'
def clean_notice(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower()
    if 'yes' in s:
        return 'Yes'
    if 'no' in s:
        return 'No'
    return np.nan

eo_df['notice_flag'] = eo_df[col_notice].apply(clean_notice)

# Filter data for the hypothesis test (In-house vs Contracted)
# Excluding Hybrid/Other to test the specific friction between purely in-house vs contracted.
analysis_df = eo_df[eo_df['dev_category'].isin(['In-house', 'Contracted'])].copy()
analysis_df = analysis_df.dropna(subset=['notice_flag'])

print(f"\nRecords for analysis (In-house vs Contracted, valid notice): {len(analysis_df)}")
print("Distribution by Development Method:")
print(analysis_df['dev_category'].value_counts())

# Contingency Table
contingency = pd.crosstab(analysis_df['dev_category'], analysis_df['notice_flag'])
print("\nContingency Table (Dev Method vs AI Notice):")
print(contingency)

# Calculate percentages
props = pd.crosstab(analysis_df['dev_category'], analysis_df['notice_flag'], normalize='index') * 100
print("\nProportions (%):")
print(props)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square Test Results:")
print(f"Chi2: {chi2:.4f}, p-value: {p:.4e}")

# Visualization
if not props.empty:
    # Sorting to ensure consistent order if needed, or just relying on index
    ax = props.plot(kind='bar', stacked=True, figsize=(8, 6), color=['#d62728', '#2ca02c'], alpha=0.8)
    plt.title('AI Notice Compliance: In-house vs Contracted')
    plt.xlabel('Development Method')
    plt.ylabel('Percentage')
    plt.legend(title='Notice Provided', loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.xticks(rotation=0)
    
    # Annotate
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white', weight='bold')
    
    plt.tight_layout()
    plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
EO 13960 records loaded: 1757

Records for analysis (In-house vs Contracted, valid notice): 50
Distribution by Development Method:
dev_category
In-house      25
Contracted    25
Name: count, dtype: int64

Contingency Table (Dev Method vs AI Notice):
notice_flag   No
dev_category    
Contracted    25
In-house      25

Proportions (%):
notice_flag      No
dev_category       
Contracted    100.0
In-house      100.0

Chi-Square Test Results:
Chi2: 0.0000, p-value: 1.0000e+00


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot (or Column Chart).
*   **Purpose:** This plot compares a categorical variable (Development Method) against a quantitative variable (Percentage) to illustrate the distribution of AI notice compliance status between two distinct groups.

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Title:** "Development Method"
    *   **Labels:** Two distinct categories: "Contracted" and "In-house".
*   **Y-Axis (Vertical):**
    *   **Title:** "Percentage"
    *   **Range:** 0 to 100 (scale ticks appear at intervals of 20: 0, 20, 40, 60, 80, 100).
    *   **Units:** Percentage (%).

### 3. Data Trends
*   **Pattern:** The data shows perfect uniformity. There is no variation between the two categories.
*   **Values:**
    *   The **Contracted** bar reaches exactly 100%.
    *   The **In-house** bar reaches exactly 100%.
*   **Highlight:** Both bars represent the maximum possible value on the scale shown for the specific category being plotted ("No").

### 4. Annotations and Legends
*   **Chart Title:** "AI Notice Compliance: In-house vs Contracted".
*   **Legend:** Located in the top right corner with the title "Notice Provided". It displays a single category:
    *   **Color:** Red (Salmon/light red).
    *   **Label:** "No".
*   **Bar Annotations:** Each bar contains white text centered within the bar reading "100.0%", explicitly stating the exact value of the data point.

### 5. Statistical Insights
*   **Total Non-Compliance/Absence:** The plot indicates that for the metric "Notice Provided," the answer was "No" for 100% of the cases in this dataset.
*   **No Correlation with Development Method:** There is no statistical difference between AI tools developed via "Contracted" methods versus "In-house" methods regarding this metric. Both methods resulted in a 0% rate of providing notice (or a 100% rate of *not* providing notice).
*   **Conclusion:** Based on this figure, neither development strategy has successfully implemented AI Notice provision in the sampled data; notice is consistently absent regardless of whether the development was outsourced or done internally.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
