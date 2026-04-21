# Experiment 96: node_5_20

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_20` |
| **ID in Run** | 96 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:39:35.595296+00:00 |
| **Runtime** | 356.6s |
| **Parent** | `node_4_36` |
| **Children** | `node_6_21` |
| **Creation Index** | 97 |

---

## Hypothesis

> The Vendor-Bias Blindspot: Commercial AI systems are significantly less likely
to have documented 'Disparity Mitigation' evidence compared to internally
developed systems, reflecting a lack of insight into vendor training data.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7581 (Likely True) |
| **Posterior** | 0.2582 (Likely False) |
| **Surprise** | -0.5998 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 2.0 |
| Maybe True | 28.0 |
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
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Analyze if commercial AI procurement correlates with lower visibility into bias mitigation.

### Steps
- 1. Filter for `eo13960_scored`.
- 2. Use `10_commercial_ai` as the independent variable and `55_disparity_mitigation` as the dependent variable.
- 3. Map values to binary logic (e.g., 'Yes' vs 'No/Not Applicable').
- 4. Compare mitigation rates between Commercial and Non-Commercial systems.
- 5. Perform a Fisher's Exact Test or Chi-square test.

### Deliverables
- Comparative bar chart of Disparity Mitigation rates; Statistical test results.

---

## Analysis

The experiment successfully tested the 'Vendor-Bias Blindspot' hypothesis using
the EO 13960 dataset. After correcting the initial column identification errors
(using `22_dev_method` for development source and `62_disparity_mitigation` for
mitigation evidence), the analysis compared 481 'Commercial/Vendor' systems
against 439 'Internal/Gov' systems.

The results **rejected the hypothesis** that commercial systems are less likely
to have documented disparity mitigation. The analysis found no statistically
significant difference between the two groups (Chi-Square p-value = 0.8443).

Instead, the data revealed a systemic lack of transparency across *both*
sectors: only **5.4%** of Commercial systems and **5.9%** of Internal government
systems had any documented evidence of disparity mitigation. The vast majority
(>94%) of systems in both categories reported 'No', 'N/A', or provided no data
regarding how they mitigate bias. This suggests that the 'blindspot' is not
specific to vendors but is a pervasive issue in the current federal AI
landscape.

---

## Review

The experiment successfully tested the 'Vendor-Bias Blindspot' hypothesis using
the EO 13960 dataset. After correcting the initial column identification errors
(using `22_dev_method` for development source and `62_disparity_mitigation` for
mitigation evidence), the analysis compared 481 'Commercial/Vendor' systems
against 439 'Internal/Gov' systems.

The results **rejected the hypothesis** that commercial systems are less likely
to have documented disparity mitigation. The analysis found no statistically
significant difference between the two groups (Chi-Square p-value = 0.8443).

Instead, the data revealed a systemic lack of transparency across *both*
sectors: only **5.4%** of Commercial systems and **5.9%** of Internal government
systems had any documented evidence of disparity mitigation. The vast majority
(>94%) of systems in both categories reported 'No', 'N/A', or provided no data
regarding how they mitigate bias. This suggests that the 'blindspot' is not
specific to vendors but is a pervasive issue in the current federal AI
landscape.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# Define columns
col_dev = '22_dev_method'
col_mitig = '62_disparity_mitigation'

# 1. Map Development Method (Independent Variable)
def map_dev_method(val):
    s = str(val).lower().strip()
    if 'contracting' in s and 'both' not in s:
        return 'Commercial/Vendor'
    elif 'in-house' in s and 'both' not in s:
        return 'Internal/Gov'
    return 'Other/Mixed'

df_eo['dev_source'] = df_eo[col_dev].apply(map_dev_method)

# Filter for only the two distinct groups
df_analysis = df_eo[df_eo['dev_source'].isin(['Commercial/Vendor', 'Internal/Gov'])].copy()

# 2. Map Disparity Mitigation (Dependent Variable)
def map_mitigation(val):
    if pd.isna(val):
        return 'No/Unknown'
    
    s = str(val).lower().strip()
    
    # List of phrases indicating NO mitigation or Not Applicable
    negative_indicators = [
        'nan',
        'n/a',
        'not applicable',
        'none',
        'no demographic',
        'not safety',
        'not rights',
        'tbd',
        'unknown'
    ]
    
    # Check if the text starts with negative phrases or matches exactly
    if any(s.startswith(x) for x in negative_indicators):
        return 'No/Unknown'
    
    # If the text is just "no" or very short negative
    if len(s) < 5 and 'no' in s:
        return 'No/Unknown'
        
    # Otherwise, assume the presence of descriptive text implies evidence
    return 'Documented'

df_analysis['mitigation_status'] = df_analysis[col_mitig].apply(map_mitigation)

# 3. Generate Statistics
print(f"\n--- Analysis Dataset (n={len(df_analysis)}) ---")
contingency_table = pd.crosstab(df_analysis['dev_source'], df_analysis['mitigation_status'])
print("\n--- Contingency Table ---")
print(contingency_table)

# Calculate percentages
props = pd.crosstab(df_analysis['dev_source'], df_analysis['mitigation_status'], normalize='index')
print("\n--- Proportions ---")
print(props)

# 4. Statistical Test (Chi-Square)
if contingency_table.size > 0:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-Square Test Results:\nChi2 Statistic: {chi2:.4f}\nP-value: {p:.4e}\nDegrees of Freedom: {dof}")
    
    alpha = 0.05
    if p < alpha:
        print("Result: Significant difference found. Hypothesis supported or rejected based on direction.")
    else:
        print("Result: No significant difference found.")
else:
    print("Insufficient data for statistical test.")

# 5. Visualization
plt.figure(figsize=(10, 6))
ax = props.plot(kind='bar', stacked=True, color=['#2ca02c', '#d62728'], figsize=(8, 6))

plt.title('Disparity Mitigation Documentation by Dev Source')
plt.xlabel('Development Source')
plt.ylabel('Proportion of Systems')
plt.legend(title='Mitigation Evidence', loc='upper right', bbox_to_anchor=(1.25, 1))
plt.xticks(rotation=0)

# Annotate bars
for c in ax.containers:
    ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white', weight='bold')

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: 
--- Analysis Dataset (n=920) ---

--- Contingency Table ---
mitigation_status  Documented  No/Unknown
dev_source                               
Commercial/Vendor          26         455
Internal/Gov               26         413

--- Proportions ---
mitigation_status  Documented  No/Unknown
dev_source                               
Commercial/Vendor    0.054054    0.945946
Internal/Gov         0.059226    0.940774

Chi-Square Test Results:
Chi2 Statistic: 0.0386
P-value: 8.4433e-01
Degrees of Freedom: 1
Result: No significant difference found.


=== Plot Analysis (figure 2) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Plot (specifically, a 100% stacked bar chart).
*   **Purpose:** The plot compares the distribution of disparity mitigation documentation (Documented vs. No/Unknown) across two different sources of system development (Commercial/Vendor vs. Internal/Gov). It visualizes the proportion of systems within each category that have evidence of mitigation strategies.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Development Source"
    *   **Categories:** "Commercial/Vendor" and "Internal/Gov".
*   **Y-Axis:**
    *   **Label:** "Proportion of Systems"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100%).
    *   **Ticks:** Intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
*   **Dominant Category:** In both development sources, the vast majority of systems fall into the "No/Unknown" category (represented by the red bars). This section takes up nearly the entire height of the plot.
*   **Minority Category:** The "Documented" category (green bars) represents a very small fraction of the systems in both cases.
*   **Consistency:** The distribution is remarkably similar between the two groups. Both "Commercial/Vendor" and "Internal/Gov" sources show almost identical proportions of documented versus undocumented systems.
*   **Visual vs. Label Discrepancy:** The text annotations label the red sections as "0.9%" and the green sections as "0.1%". However, the Y-axis is a proportion from 0 to 1. Given that the bars fill the entire height (summing to 1.0), it is highly likely the labels represent the proportions **0.9 (90%)** and **0.1 (10%)**, and the percentage sign `%` is a typographical error. If read literally as 0.9%, the bars would not fill the chart. Visually, the green bar sits slightly below the 0.1 tick mark, suggesting the documentation rate is roughly 5-10%.

### 4. Annotations and Legends
*   **Title:** "Disparity Mitigation Documentation by Dev Source"
*   **Legend:** Located at the top right, labeled "Mitigation Evidence".
    *   **Green:** "Documented"
    *   **Red:** "No/Unknown"
*   **Bar Annotations:**
    *   White text inside the red bars reads **"0.9%"**.
    *   White text inside the green bars reads **"0.1%"**.

### 5. Statistical Insights
*   **Lack of Documentation:** The primary insight is a systemic lack of documentation regarding disparity mitigation. Regardless of whether a system is developed commercially or internally by the government, roughly 90% (assuming 0.9 proportion) of systems lack evidence of mitigation.
*   **No Source Advantage:** There is no discernible advantage in documentation practices between commercial vendors and internal government developers; both perform equally poorly according to this metric.
*   **Transparency Gap:** The data suggests a significant transparency gap in the field, with "Documented" systems being the extreme exception rather than the norm.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
