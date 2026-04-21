# Experiment 270: node_6_71

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_71` |
| **ID in Run** | 270 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:48:47.436307+00:00 |
| **Runtime** | 307.9s |
| **Parent** | `node_5_56` |
| **Children** | None |
| **Creation Index** | 271 |

---

## Hypothesis

> The Fairness Blindspot: Systems classified strictly as 'Safety-Impacting' are
significantly less likely to undergo 'Disparity/Bias Mitigation' compared to
'Rights-Impacting' systems, despite the potential for safety failures to
disproportionately affect specific demographics.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7417 (Likely True) |
| **Posterior** | 0.9139 (Definitely True) |
| **Surprise** | +0.2067 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 29.0 |
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

**Objective:** Determine if safety-critical systems are overlooked for bias testing compared to rights-critical systems.

### Steps
- 1. Load 'eo13960_scored'.
- 2. Filter for rows where '17_impact_type' contains 'Safety' or 'Rights'.
- 3. Create a category column: 'Safety-Only' vs. 'Rights-Related'.
- 4. Parse '62_disparity_mitigation' into binary (Yes/No).
- 5. Compare the mitigation rates using a Z-test for two proportions.

### Deliverables
- Comparison plot of Bias Mitigation rates; Z-test results.

---

## Analysis

The experiment successfully tested the 'Fairness Blindspot' hypothesis by
employing heuristic text analysis to parse the verbose 'Disparity Mitigation'
field in the EO 13960 dataset.

The analysis identified 77 relevant systems, split into 'Rights-Related' (n=59)
and 'Safety-Only' (n=18) cohorts.

The results **strongly support the hypothesis**:
1. **Observed Rates**: 'Rights-Related' systems showed a 23.7% rate of
documented disparity mitigation (14/59), whereas 'Safety-Only' systems showed a
0.0% rate (0/18).
2. **Statistical Significance**: The Z-test yielded a p-value of 0.0223,
confirming that the difference is statistically significant (p < 0.05).
3. **Implication**: The complete absence of bias mitigation evidence in the
'Safety-Only' cohort suggests a governance gap where safety-critical systems are
not being evaluated for demographic disparities, confirming the existence of a
'Fairness Blindspot'.

---

## Review

The experiment successfully tested the 'Fairness Blindspot' hypothesis. Initial
failures due to unstructured text data were resolved by implementing a heuristic
keyword analysis to parse the 'Disparity Mitigation' field. The analysis
correctly categorized 77 systems into 'Rights-Related' (n=59) and 'Safety-Only'
(n=18) cohorts. The Z-test and visualization were successfully generated,
confirming a statistically significant difference (p=0.0223) between the groups.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
import re

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Target columns
impact_col = '17_impact_type'
mitigation_col = '62_disparity_mitigation'

# Drop rows where impact type is missing
eo_data = eo_data.dropna(subset=[impact_col])

# ---------------------------------------------------------
# 1. Categorize Systems (Safety-Only vs Rights-Related)
# ---------------------------------------------------------
# Normalize text
eo_data[impact_col] = eo_data[impact_col].astype(str)

# Define masks
is_rights = eo_data[impact_col].str.contains('Rights', case=False, na=False)
is_safety = eo_data[impact_col].str.contains('Safety', case=False, na=False)

# Assign groups
# Priority: If it has Rights (even if mixed with Safety), it goes to 'Rights-Related'.
# If it has Safety but NO Rights, it goes to 'Safety-Only'.
conditions = [
    is_rights,
    (is_safety & ~is_rights)
]
choices = ['Rights-Related', 'Safety-Only']
eo_data['impact_group'] = np.select(conditions, choices, default='Other')

# Filter for only the groups of interest
analysis_df = eo_data[eo_data['impact_group'].isin(['Rights-Related', 'Safety-Only'])].copy()

print(f"Data filtered. Rows: {len(analysis_df)}")
print(f"Counts by group:\n{analysis_df['impact_group'].value_counts()}")

# ---------------------------------------------------------
# 2. Parse Outcome Variable (Disparity Mitigation) using Heuristics
# ---------------------------------------------------------

def classify_mitigation(text):
    if pd.isna(text):
        return 0 # Treat missing as no evidence of mitigation
    
    text = str(text).lower().strip()
    
    # Negative indicators (Strong override)
    # "no analysis", "none", "n/a", "not applicable", "no specific", "does not"
    negatives = [
        r'no\s+analysis', r'no\s+specific', r'^none', r'n/a', r'not\s+applicable', 
        r'does\s+not', r'no\s+impact', r'no\s+mitigation'
    ]
    for neg in negatives:
        if re.search(neg, text):
            return 0
            
    # Positive indicators
    # "test", "eval", "monitor", "review", "audit", "assess", "mitigat", "bias", "fair", "check"
    positives = [
        'test', 'eval', 'monitor', 'review', 'audit', 'assess', 
        'mitigat', 'bias', 'fair', 'check', 'ensure', 'validat', 'analy'
    ]
    for pos in positives:
        if pos in text:
            return 1
            
    # Default to 0 if no positive evidence found
    return 0

analysis_df['has_mitigation'] = analysis_df[mitigation_col].apply(classify_mitigation)

# ---------------------------------------------------------
# 3. Statistical Analysis
# ---------------------------------------------------------
summary = analysis_df.groupby('impact_group')['has_mitigation'].agg(['count', 'sum', 'mean'])
summary.columns = ['Total', 'Mitigation_Yes', 'Rate']

print("\nSummary Statistics (Heuristic Parsing):")
print(summary)

# Prepare for Z-test
if 'Rights-Related' in summary.index and 'Safety-Only' in summary.index:
    count_success = np.array([
        summary.loc['Safety-Only', 'Mitigation_Yes'], 
        summary.loc['Rights-Related', 'Mitigation_Yes']
    ])
    nobs = np.array([
        summary.loc['Safety-Only', 'Total'], 
        summary.loc['Rights-Related', 'Total']
    ])
    
    # proportions_ztest
    stat, pval = proportions_ztest(count_success, nobs)
    
    print(f"\nZ-test Results (Safety-Only vs Rights-Related):")
    print(f"Z-statistic: {stat:.4f}")
    print(f"P-value: {pval:.4e}")
    
    # ---------------------------------------------------------
    # 4. Visualization
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    bars = plt.bar(summary.index, summary['Rate'], color=['#FF9999', '#66B2FF'])
    plt.ylabel('Proportion with Disparity Mitigation')
    plt.title('Disparity Mitigation Evidence: Safety-Only vs Rights-Related')
    plt.ylim(0, 1.1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
else:
    print("\nInsufficient data for comparison (one or both groups missing).")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Data filtered. Rows: 77
Counts by group:
impact_group
Rights-Related    59
Safety-Only       18
Name: count, dtype: int64

Summary Statistics (Heuristic Parsing):
                Total  Mitigation_Yes      Rate
impact_group                                   
Rights-Related     59              14  0.237288
Safety-Only        18               0  0.000000

Z-test Results (Safety-Only vs Rights-Related):
Z-statistic: -2.2848
P-value: 2.2324e-02


=== Plot Analysis (figure 1) ===
Based on the analysis of the provided plot image, here are the details:

**1. Plot Type**
*   **Type:** Vertical Bar Plot (or Column Chart).
*   **Purpose:** The plot is designed to compare the proportion of instances showing evidence of "Disparity Mitigation" across two distinct categories: "Rights-Related" and "Safety-Only".

**2. Axes**
*   **X-Axis:**
    *   **Labels:** Categorical labels representing two groups: "Rights-Related" and "Safety-Only".
    *   **Range:** Two discrete categories.
*   **Y-Axis:**
    *   **Title:** "Proportion with Disparity Mitigation".
    *   **Units:** The axis uses decimal notation representing proportions (0.0 to 1.0), which corresponds to percentages (0% to 100%).
    *   **Range:** The scale runs from 0.0 to 1.0, with grid marks every 0.2 units. The visual axis extends slightly beyond 1.0 to approximately 1.1.

**3. Data Trends**
*   **Tallest Bar:** The "Rights-Related" category is the tallest (and only visible) bar, reaching a value of roughly 0.24 on the y-axis.
*   **Shortest Bar:** The "Safety-Only" category effectively has no height, representing a value of 0.
*   **Pattern:** There is a stark contrast between the two categories. While nearly a quarter of "Rights-Related" cases show evidence of disparity mitigation, there is absolutely zero evidence found in the "Safety-Only" cases.

**4. Annotations and Legends**
*   **Annotations:**
    *   **"23.7%"**: Located directly above the "Rights-Related" bar, indicating the exact percentage value for that category.
    *   **"0.0%"**: Located just above the baseline for the "Safety-Only" category, indicating a null value.
*   **Title:** "Disparity Mitigation Evidence: Safety-Only vs Rights-Related" clearly defines the scope of the comparison.

**5. Statistical Insights**
*   **Exclusive Association:** The plot suggests a strong conditional relationship where evidence of disparity mitigation is exclusively found in the "Rights-Related" category (23.7%) within this dataset.
*   **Complete Absence in Safety-Only:** The 0.0% finding for "Safety-Only" implies that when an issue is categorized purely as a safety concern, mitigation regarding disparity (unequal outcomes among groups) is either not performed or not documented at all.
*   **Conclusion:** This data indicates that "Rights-Related" issues are significantly more likely to trigger or be associated with efforts to mitigate disparity than issues viewed solely through a "Safety" lens.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
