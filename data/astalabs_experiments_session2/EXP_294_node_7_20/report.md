# Experiment 294: node_7_20

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_20` |
| **ID in Run** | 294 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T15:04:26.789297+00:00 |
| **Runtime** | 388.5s |
| **Parent** | `node_6_52` |
| **Children** | None |
| **Creation Index** | 295 |

---

## Hypothesis

> Agencies with high 'Basic Control' (Tier 1) scores also demonstrate greater
transparency by having significantly higher rates of public code access.

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
| Maybe True | 14.0 |
| Uncertain | 1.0 |
| Maybe False | 14.0 |
| Definitely False | 1.0 |

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

**Objective:** Test if internal governance discipline correlates with external transparency.

### Steps
- 1. Filter 'eo13960_scored'.
- 2. Calculate the median 'tier1_score'.
- 3. Split agencies into 'High Governance' (score > median) and 'Low Governance'.
- 4. Compare the proportion of 'Yes' values in '38_code_access' between the two groups using a Chi-square test.

### Deliverables
- Comparison of code availability rates between high and low governance agencies.

---

## Analysis

The experiment was successfully executed following the correction of the data
parsing logic. The programmer implemented custom text-scoring functions to
correctly quantify the verbose entries in the 'Basic Control' columns
(Documentation, Authorization, Notice), resulting in a valid segmentation of the
dataset into 'High Governance' (N=525) and 'Low Governance' (N=1,232) groups
based on a median split.

The analysis revealed that public code availability is extremely rare across the
entire federal dataset, with only ~2.7% of projects making code publicly
available. The Chi-square test yielded a statistic of 0.0000 and a p-value of
1.0, indicating absolutely no statistically significant difference in
transparency between agencies with high vs. low internal governance scores.
Consequently, the hypothesis that internal governance discipline correlates with
external transparency is rejected; the two attributes appear to be independent
in this dataset.

---

## Review

The experiment was successfully executed following the correction of the data
parsing logic. The programmer implemented custom text-scoring functions to
correctly quantify the verbose entries in the 'Basic Control' columns
(Documentation, Authorization, Notice), resulting in a valid segmentation of the
dataset into 'High Governance' (N=525) and 'Low Governance' (N=1,232) groups
based on a median split.

The analysis revealed that public code availability is extremely rare across the
entire federal dataset, with only ~2.7% of projects making code publicly
available. The Chi-square test yielded a statistic of 0.0000 and a p-value of
1.0, indicating absolutely no statistically significant difference in
transparency between agencies with high vs. low internal governance scores.
Consequently, the hypothesis that internal governance discipline correlates with
external transparency is rejected; the two attributes appear to be independent
in this dataset.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Data Loaded. Shape: {eo_df.shape}")

# --- Step 2: Calculate Tier 1 Score ---
# Using inspected columns: 
# 34_data_docs (Documentation)
# 40_has_ato (Authorization)
# 59_ai_notice (Transparency/Notice)

def score_docs(val):
    if pd.isna(val): return 0
    s = str(val).lower()
    # If it explicitly says missing or not available, score 0
    if 'missing' in s or 'not available' in s:
        return 0
    # If it says complete, available, existing, partially -> 1
    if 'complete' in s or 'available' in s or 'exist' in s or 'partially' in s:
        return 1
    return 0

def score_ato(val):
    if pd.isna(val): return 0
    s = str(val).lower()
    if s.startswith('yes'):
        return 1
    return 0

def score_notice(val):
    if pd.isna(val): return 0
    s = str(val).lower()
    # If none or n/a -> 0
    if 'none' in s or 'n/a' in s:
        return 0
    # If online, terms, physical -> 1
    if 'online' in s or 'physical' in s or 'terms' in s or 'instruction' in s:
        return 1
    return 0

# Apply scoring
eo_df['score_docs'] = eo_df['34_data_docs'].apply(score_docs)
eo_df['score_ato'] = eo_df['40_has_ato'].apply(score_ato)
eo_df['score_notice'] = eo_df['59_ai_notice'].apply(score_notice)

eo_df['tier1_score'] = eo_df['score_docs'] + eo_df['score_ato'] + eo_df['score_notice']

# --- Step 3: Split into Groups ---
median_score = eo_df['tier1_score'].median()
print(f"\nTier 1 Score Distribution:\n{eo_df['tier1_score'].value_counts().sort_index()}")
print(f"Median Score: {median_score}")

# High > Median, Low <= Median
# If median is high (e.g. 2 or 3), this split might be unbalanced, but we stick to the plan.
eo_df['governance_group'] = np.where(eo_df['tier1_score'] > median_score, 'High Governance', 'Low Governance')
print(f"\nGroup Counts:\n{eo_df['governance_group'].value_counts()}")

# --- Step 4: Analyze Code Access ---
# Target: 38_code_access
# We want 'publicly available'.
def score_public_code(val):
    if pd.isna(val): return 0
    s = str(val).lower()
    if 'publicly available' in s:
        return 1
    return 0

eo_df['is_public_code'] = eo_df['38_code_access'].apply(score_public_code)

# Contingency Table
contingency = pd.crosstab(eo_df['governance_group'], eo_df['is_public_code'])
contingency.columns = ['Not Public', 'Public']

print("\nContingency Table (Public Code Access):")
print(contingency)

# Chi-square
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Plot
props = contingency.div(contingency.sum(axis=1), axis=0)
ax = props.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'])
plt.title(f'Public Code Access by Governance Tier (Median={median_score})')
plt.ylabel('Proportion')
plt.xlabel('Governance Group')
plt.legend(title='Code Status', loc='upper right')
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: EO 13960 Data Loaded. Shape: (1757, 196)

Tier 1 Score Distribution:
tier1_score
0    831
1    401
2    498
3     27
Name: count, dtype: int64
Median Score: 1.0

Group Counts:
governance_group
Low Governance     1232
High Governance     525
Name: count, dtype: int64

Contingency Table (Public Code Access):
                  Not Public  Public
governance_group                    
High Governance          511      14
Low Governance          1198      34

Chi-square Statistic: 0.0000
P-value: 1.0000e+00


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Plot (or Stacked Column Chart).
*   **Purpose:** This plot is designed to compare the proportional distribution of a categorical variable (Code Status: Public vs. Not Public) across two different groups (High Governance vs. Low Governance). It normalizes the data to a total proportion of 1.0 (or 100%) to facilitate a direct comparison of the *composition* of each group.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Governance Group"
    *   **Categories:** Two discrete groups labeled "High Governance" and "Low Governance". The text labels are rotated 90 degrees vertically.
*   **Y-Axis:**
    *   **Label:** "Proportion"
    *   **Range:** The axis runs from **0.0 to 1.0**, with tick marks every 0.2 units.
    *   **Units:** The values represent a ratio or probability, where 1.0 equates to 100% of the sample in that group.

### 3. Data Trends
*   **Dominant Category:** In both the "High Governance" and "Low Governance" groups, the **"Not Public"** category (represented by the salmon/light red color) makes up the overwhelming majority of the proportion. Visually, it appears to cover approximately 95% to 98% of each bar.
*   **Minor Category:** The **"Public"** category (represented by the light blue color) is a very thin sliver at the top of each bar, indicating a very low proportion (likely <5%).
*   **Similarity:** There is virtually no discernible difference between the two bars. The ratio of Public to Not Public code appears nearly identical regardless of whether the governance tier is High or Low.

### 4. Annotations and Legends
*   **Title:** "Public Code Access by Governance Tier (Median=1.0)". This indicates the chart is analyzing code accessibility based on governance levels. The "(Median=1.0)" likely refers to a statistic regarding the governance scoring or the dataset split point.
*   **Legend:** Located in the top right corner, titled **"Code Status"**. It defines the color coding:
    *   **Salmon/Pink:** "Not Public"
    *   **Blue:** "Public"

### 5. Statistical Insights
*   **Low Availability:** The primary insight is that public code access is extremely rare in this dataset. The vast majority of projects or entities analyzed do not make their code public.
*   **No Governance Effect:** The plot suggests that the level of governance ("High" vs. "Low") has **no impact** on the likelihood of code being public. Since the proportions are nearly identical across both groups, governance tier does not appear to be a differentiating factor for code accessibility.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
