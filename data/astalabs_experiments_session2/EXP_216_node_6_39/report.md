# Experiment 216: node_6_39

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_39` |
| **ID in Run** | 216 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T11:05:29.419104+00:00 |
| **Runtime** | 291.5s |
| **Parent** | `node_5_58` |
| **Children** | None |
| **Creation Index** | 217 |

---

## Hypothesis

> The 'Commercial Opacity' Effect: Government AI systems acquired from commercial
vendors are significantly less likely to have accessible technical documentation
compared to government-developed systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8145 (Likely True) |
| **Posterior** | 0.9368 (Definitely True) |
| **Surprise** | +0.1468 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 9.0 |
| Maybe True | 21.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 60.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Investigate if relying on commercial AI vendors in the federal government creates a transparency gap regarding data and code documentation.

### Steps
- 1. Load `eo13960_scored`.
- 2. Create two groups based on `10_commercial_ai`: 'Commercial' (Yes) and 'Custom/Internal' (No).
- 3. For each group, calculate the percentage of systems that have affirmative values for `34_data_docs` (Data Documentation) or `31_code_access` (Code Access). Note that values might be text like 'Yes', 'No', 'Available', so normalize them to binary.
- 4. Perform a Chi-Square test of independence to compare the documentation rates between the two groups.

### Deliverables
- Bar chart comparing Documentation Availability rates for Commercial vs. Internal AI; Chi-Square test results.

---

## Analysis

The experiment successfully tested the 'Commercial Opacity' hypothesis using the
EO 13960 dataset. After correcting for column identification in the debug phase,
920 AI use cases were analyzed, split into 'Commercial' (n=481) and 'Internal'
(n=439) groups based on the '22_dev_method' field. The analysis defined
'Technical Documentation Availability' as having either valid data documentation
('34_data_docs') or source code access ('38_code_access'). The results reveal a
striking disparity: 94.3% of Internal/In-House systems possessed accessible
technical documentation, compared to only 67.2% of Commercial systems. A Chi-
Square test of independence confirmed this difference is statistically
significant (Chi2=104.5, p=1.57e-24). Consequently, the hypothesis is strongly
supported; government AI systems acquired from commercial vendors are
significantly less likely to have accessible technical documentation than those
developed in-house, validating the existence of a 'Commercial Opacity' effect in
the current federal AI landscape.

---

## Review

The experiment was successfully executed and faithfully tested the 'Commercial
Opacity' hypothesis. Although the original plan specified using column
'10_commercial_ai', the programmer correctly identified during the debug phase
that this column contained unstructured text descriptions rather than a
classification flag. The programmer appropriately adapted the strategy by using
'22_dev_method' to accurately distinguish between 'Commercial' (Developed with
contracting resources, n=481) and 'Internal' (Developed in-house, n=439)
systems.

The analysis of technical documentation availability (defined as having valid
data documentation or source code access) revealed a statistically significant
disparity. Internal government systems demonstrated a high documentation rate of
94.3%, whereas Commercial systems lagged significantly at 67.2%. The Chi-Square
test yielded a p-value of 1.57e-24, strongly rejecting the null hypothesis.
These findings validate the hypothesis that relying on commercial vendors
creates a transparency gap regarding technical documentation in federal AI
deployments.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load dataset
print("Loading dataset...")
try:
    df_all = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    # Fallback for different environment structures
    df_all = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
df_eo = df_all[df_all['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored subset shape: {df_eo.shape}")

# Define Columns
col_dev_method = '22_dev_method'
col_data_docs = '34_data_docs'
col_code_access = '38_code_access'  # Identified in debug

# 1. Define Groups (Commercial vs Internal)
# We focus on the two distinct categories identified in debug
def define_group(val):
    s = str(val).strip()
    if s == 'Developed with contracting resources.':
        return 'Commercial'
    elif s == 'Developed in-house.':
        return 'Internal'
    else:
        return None # Exclude 'Both', 'NaN', etc.

df_eo['group'] = df_eo[col_dev_method].apply(define_group)

# Filter dataset to only these two groups
df_analysis = df_eo.dropna(subset=['group']).copy()
print(f"\nAnalysis Subset Shape (Commercial + Internal only): {df_analysis.shape}")
print("Group Distribution:")
print(df_analysis['group'].value_counts())

# 2. Define Documentation Metric
def check_data_docs(val):
    if pd.isna(val):
        return False
    s = str(val).lower().strip()
    # Negative indicators
    if 'missing' in s or 'not available' in s or 'not reported' in s or s == 'no':
        return False
    # Positive indicators (completeness, partial, existing)
    if 'complete' in s or 'available' in s or 'yes' in s or 'partial' in s:
        return True
    return False

def check_code_access(val):
    if pd.isna(val):
        return False
    s = str(val).lower().strip()
    # Look for explicit Yes
    if s.startswith('yes'):
        return True
    return False

# Apply logic
df_analysis['has_data_docs'] = df_analysis[col_data_docs].apply(check_data_docs)
df_analysis['has_code_access'] = df_analysis[col_code_access].apply(check_code_access)

# Combined Metric: Has EITHER valid data docs OR code access
df_analysis['has_tech_docs'] = df_analysis['has_data_docs'] | df_analysis['has_code_access']

# 3. Statistical Analysis
# Calculate rates
group_stats = df_analysis.groupby('group')['has_tech_docs'].agg(['count', 'sum', 'mean'])
group_stats.columns = ['Total', 'With_Docs', 'Rate']
print("\nDocumentation Statistics by Group:")
print(group_stats)

# Chi-Square Test
# Contingency Table
#              Has Docs | No Docs
# Commercial      A     |    B
# Internal        C     |    D
contingency = pd.crosstab(df_analysis['group'], df_analysis['has_tech_docs'])
print("\nContingency Table:")
print(contingency)

chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square Results:\n  Statistic: {chi2:.4f}\n  P-value: {p:.4e}")

# Interpretation
alpha = 0.05
if p < alpha:
    print("\nResult: Statistically Significant Difference found.")
else:
    print("\nResult: No Statistically Significant Difference found.")

# 4. Visualization
plt.figure(figsize=(8, 6))

# Order: Commercial, Internal
groups = ['Commercial', 'Internal']
rates = [group_stats.loc['Commercial', 'Rate'], group_stats.loc['Internal', 'Rate']]
colors = ['#ff9999', '#66b3ff'] # Redish for commercial, Blueish for internal

bars = plt.bar(groups, rates, color=colors, edgecolor='black', alpha=0.8)

plt.ylabel('Proportion with Accessible Tech Docs')
plt.title(f"'Commercial Opacity' Effect: Technical Documentation Availability\n(Commercial vs. In-House Government AI)\np-value = {p:.4e}")
plt.ylim(0, 1.05)

# Add labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset...
EO 13960 Scored subset shape: (1757, 196)

Analysis Subset Shape (Commercial + Internal only): (920, 197)
Group Distribution:
group
Commercial    481
Internal      439
Name: count, dtype: int64

Documentation Statistics by Group:
            Total  With_Docs      Rate
group                                 
Commercial    481        323  0.671518
Internal      439        414  0.943052

Contingency Table:
has_tech_docs  False  True 
group                      
Commercial       158    323
Internal          25    414

Chi-Square Results:
  Statistic: 104.5040
  P-value: 1.5687e-24

Result: Statistically Significant Difference found.


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot (Vertical Bar Chart).
*   **Purpose:** This chart is designed to compare a categorical variable (source of AI: Commercial vs. Internal) against a quantitative metric (Proportion with Accessible Technical Documentation). It specifically illustrates the difference in transparency or documentation availability between the two groups.

### 2. Axes
*   **X-Axis:**
    *   **Label:** Represents the category of the AI origin.
    *   **Categories:** "Commercial" (left) and "Internal" (right).
*   **Y-Axis:**
    *   **Label:** "Proportion with Accessible Tech Docs".
    *   **Units:** Proportions (0 to 1 scale), which correspond to percentages.
    *   **Range:** The axis spans from **0.0 to 1.0**, with grid lines marked at intervals of 0.2.

### 3. Data Trends
*   **Comparison:** There is a distinct disparity between the two categories.
    *   **Tallest Bar:** The **"Internal"** bar (light blue) is the tallest, indicating a much higher rate of documentation availability.
    *   **Shortest Bar:** The **"Commercial"** bar (light pink) is significantly shorter.
*   **Values:**
    *   **Internal AI:** Approximately **94.3%** of Internal/In-House Government AI projects have accessible technical documentation.
    *   **Commercial AI:** Only **67.2%** of Commercial AI projects have accessible technical documentation.
*   **Pattern:** The data suggests a trend where in-house government projects are considerably more transparent regarding technical documentation than their commercial counterparts.

### 4. Annotations and Legends
*   **Main Title:** "'Commercial Opacity' Effect: Technical Documentation Availability (Commercial vs. In-House Government AI)". This establishes the context of the study, framing the lower score of commercial AI as "Commercial Opacity."
*   **Statistical Annotation:** A subtitle displays **"p-value = 1.5687e-24"**.
*   **Bar Labels:** Specific percentage values (**67.2%** and **94.3%**) are annotated directly above the corresponding bars for precise reading.
*   **Grid:** Horizontal dashed gray lines are included to assist in visual estimation of the bar heights relative to the Y-axis.

### 5. Statistical Insights
*   **The "Opacity Effect":** The plot visualizes a significant gap of **27.1 percentage points** between Internal and Commercial AI. This supports the hypothesis suggested by the title—that commercial AI acts with greater "opacity" (less transparency regarding documentation) compared to internal government projects.
*   **Statistical Significance:** The provided p-value is **1.5687e-24** ($1.5687 \times 10^{-24}$). This number is infinitesimally small (practically zero).
    *   **Interpretation:** In scientific research, a p-value this low indicates that the difference observed between the two groups is **statistically significant** to a very high degree. We can reject the null hypothesis; it is extremely unlikely that this difference in documentation availability occurred by random chance.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
