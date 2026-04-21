# Experiment 230: node_5_90

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_90` |
| **ID in Run** | 230 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T11:47:01.134873+00:00 |
| **Runtime** | 239.2s |
| **Parent** | `node_4_4` |
| **Children** | `node_6_53` |
| **Creation Index** | 231 |

---

## Hypothesis

> The 'High Stakes' Compliance Gap: Systems self-identified as 'High Impact' in
the federal inventory do not show a statistically significant higher rate of
'Independent Evaluation' than 'Low Impact' systems, suggesting a failure of
risk-based tiering.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7339 (Likely True) |
| **Posterior** | 0.2500 (Likely False) |
| **Surprise** | -0.5806 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 29.0 |
| Uncertain | 1.0 |
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

**Objective:** Assess if risk tiering drives independent auditing.

### Steps
- 1. Load `eo13960_scored`.
- 2. Parse `17_impact_type` to categorize into 'High' (or Significant) vs 'Low' (or Moderate/Minor).
- 3. Parse `55_independent_eval` into binary.
- 4. Perform a Z-test for proportions comparing Independent Evaluation rates between High and Low impact groups.

### Deliverables
- Proportions, Z-test results, and a conclusion on the efficacy of risk-based governance.

---

## Analysis

The experiment successfully tested the 'High Stakes Compliance Gap' hypothesis
by analyzing 1,718 AI systems from the EO 13960 inventory.

1. **Hypothesis Rejection**: The hypothesis posited that there would be *no*
significant difference in independent evaluation rates between 'High Impact' and
'Low Impact' systems. The data **strongly rejects** this. 'High Impact' systems
showed a significantly higher rate of independent evaluation (14.5%) compared to
'Low Impact' systems (1.3%).

2. **Statistical Evidence**: The Z-test resulted in a Z-score of 10.71 and a
p-value of < 0.0001, confirming that the observed difference is highly
statistically significant. This suggests that risk-based tiering *is*
functioning to some extent, as agencies are differentiating their governance
burden based on impact classification.

3. **Governance Reality**: While the relative difference is massive (High Impact
systems are ~11x more likely to be evaluated), the absolute compliance rate
remains low. Only ~14.5% of systems identified as impacting rights or safety
have undergone independent evaluation, indicating that while risk-tiering
exists, the overall maturity of external oversight remains nascent across the
federal portfolio.

---

## Review

The experiment successfully tested the 'High Stakes Compliance Gap' hypothesis
by analyzing 1,718 AI systems from the EO 13960 inventory.

1. **Hypothesis Rejection**: The hypothesis posited that there would be *no*
significant difference in independent evaluation rates between 'High Impact' and
'Low Impact' systems. The data **strongly rejects** this. 'High Impact' systems
showed a significantly higher rate of independent evaluation (14.5%) compared to
'Low Impact' systems (1.3%).

2. **Statistical Evidence**: The Z-test resulted in a Z-score of 10.71 and a
p-value of < 0.0001, confirming that the observed difference is highly
statistically significant. This suggests that risk-based tiering *is*
functioning to some extent, as agencies are differentiating their governance
burden based on impact classification.

3. **Governance Reality**: While the relative difference is massive (High Impact
systems are ~11x more likely to be evaluated), the absolute compliance rate
remains low. Only ~14.5% of systems identified as impacting rights or safety
have undergone independent evaluation, indicating that while risk-tiering
exists, the overall maturity of external oversight remains nascent across the
federal portfolio.

---

## Code

```python
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import sys

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# Filter for EO 13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# refined mapping logic based on previous output
def map_impact_level(val):
    if pd.isna(val):
        return None
    val_lower = str(val).lower()
    
    # High Impact: Rights, Safety, Both, High, Significant
    if any(x in val_lower for x in ['rights', 'safety', 'both', 'high', 'significant', 'critical']):
        return 'High Impact'
    # Low Impact: Neither, Low, Moderate, Minimal
    # Note: 'Neither' in EO 13960 context means neither safety nor rights impacting, effectively 'Low' for this binary.
    elif any(x in val_lower for x in ['neither', 'low', 'moderate', 'minimal', 'minor']):
        return 'Low Impact'
    
    return 'Uncategorized'

def map_evaluation_evidence(val):
    if pd.isna(val):
        return 0
    val_lower = str(val).lower()
    
    # Affirmative keywords
    affirmative = ['yes', 'conducted', 'completed', 'performed', 'ongoing', '3rd party', 'third party', 'independent', 'true']
    
    if any(x in val_lower for x in affirmative):
        # Exclude negations if necessary. 
        # Specific check for 'not conducted' or 'does not apply' if 'yes' is absent, 
        # but sometimes 'yes' appears in a sentence saying 'yes we did it'.
        # However, 'yes - by the caio' is a positive.
        # 'planned' is not done yet, so usually 0, but let's be strict: has it been done?
        if 'planned' in val_lower and 'completed' not in val_lower and 'ongoing' not in val_lower:
             return 0
        return 1
    return 0

# Apply Mappings
eo_df['impact_group'] = eo_df['17_impact_type'].apply(map_impact_level)
eo_df['has_eval'] = eo_df['55_independent_eval'].apply(map_evaluation_evidence)

# Check distribution
print("Impact Group Distribution:")
print(eo_df['impact_group'].value_counts())

# Filter for analysis groups
analysis_df = eo_df[eo_df['impact_group'].isin(['High Impact', 'Low Impact'])].copy()

# Calculate Statistics
group_stats = analysis_df.groupby('impact_group')['has_eval'].agg(['count', 'sum', 'mean'])
group_stats['pct'] = group_stats['mean'] * 100

print("\n--- Comparative Statistics ---")
print(group_stats)

# Ensure we have both groups
if 'High Impact' not in group_stats.index or 'Low Impact' not in group_stats.index:
    print("\nError: Missing one of the comparison groups (High Impact or Low Impact). Aborting Z-test.")
else:
    # Statistical Test (Z-test)
    high_grp = group_stats.loc['High Impact']
    low_grp = group_stats.loc['Low Impact']

    count_arr = np.array([high_grp['sum'], low_grp['sum']])
    nobs_arr = np.array([high_grp['count'], low_grp['count']])

    z_score, p_value = proportions_ztest(count_arr, nobs_arr, alternative='two-sided')

    print(f"\nZ-Score: {z_score:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    significance = "Significant" if p_value < 0.05 else "Not Significant"
    print(f"Result: {significance}")

    # Visualization
    plt.figure(figsize=(10, 6))
    bar_colors = ['#d62728', '#1f77b4'] # Red for High, Blue for Low (alphabetical order H, L)
    
    # Create bar plot
    # Note: groupby sorts alphabetically by index: High Impact, Low Impact. 
    # High Impact (H) comes before Low Impact (L)? No, H comes before L.
    bars = plt.bar(group_stats.index, group_stats['pct'], color=bar_colors, alpha=0.8)

    # Error bars
    se = np.sqrt(group_stats['mean'] * (1 - group_stats['mean']) / group_stats['count']) * 100
    plt.errorbar(group_stats.index, group_stats['pct'], yerr=se, fmt='none', ecolor='black', capsize=5)

    plt.title('Independent Evaluation Rates: High vs Low Impact AI')
    plt.ylabel('Percentage with Independent Evaluation (%)')
    plt.xlabel('Impact Classification')
    plt.ylim(0, max(group_stats['pct']) * 1.5 if len(group_stats) > 0 else 10)

    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontweight='bold')

    stats_text = (f"Z-test: z={z_score:.2f}, p={p_value:.3f}\n"
                  f"n(High)={int(high_grp['count'])}, n(Low)={int(low_grp['count'])}")
    plt.text(0.5, 0.9, stats_text, transform=plt.gca().transAxes, 
             ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Impact Group Distribution:
impact_group
Low Impact     1491
High Impact     227
Name: count, dtype: int64

--- Comparative Statistics ---
              count  sum      mean        pct
impact_group                                 
High Impact     227   33  0.145374  14.537445
Low Impact     1491   20  0.013414   1.341382

Z-Score: 10.7118
P-Value: 0.0000
Result: Significant


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **vertical bar chart** with error bars.
*   **Purpose:** The plot compares the percentage of AI systems that undergo independent evaluation across two distinct categories: "High Impact" and "Low Impact." It is designed to visualize the disparity in evaluation rates between these two classifications.

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Label:** "Impact Classification"
    *   **Categories:** The axis displays two categorical groups: "High Impact" and "Low Impact."
*   **Y-Axis (Vertical):**
    *   **Label:** "Percentage with Independent Evaluation (%)"
    *   **Range:** The axis is scaled from **0.0 to 20.0**, with major tick marks every 2.5 units (0.0, 2.5, 5.0, ..., 20.0). The maximum visual range extends slightly above 20%.
    *   **Units:** Percentage (%).

### 3. Data Trends
*   **Bar Comparison:**
    *   **High Impact (Red Bar):** This is the tallest bar, indicating a significantly higher rate of independent evaluation. The value is **14.5%**.
    *   **Low Impact (Blue Bar):** This is the shortest bar, indicating a very low rate of independent evaluation. The value is **1.3%**.
*   **Pattern:** There is a drastic difference in evaluation rates. High-impact AI systems are over 11 times more likely to have independent evaluations compared to low-impact systems.
*   **Error Bars:** Both bars include black error bars indicating uncertainty (likely a 95% confidence interval). The error bar for the "High Impact" group is noticeably wider/taller than the "Low Impact" group, reflecting higher uncertainty, likely due to the smaller sample size ($n=227$ vs. $n=1491$).

### 4. Annotations and Legends
*   **Data Labels:** The exact percentage values (**14.5%** and **1.3%**) are annotated in bold text directly above their respective bars.
*   **Statistical Annotation Box:** A box located in the upper-center of the plot provides statistical context:
    *   **Z-test:** Displays a Z-score of **10.71**, indicating a very large difference between the means relative to the variance.
    *   **p-value:** Displays **p=0.000**, indicating the difference is statistically significant (typically interpreted as $p < 0.001$).
    *   **Sample Sizes:** $n(\text{High})=227$ and $n(\text{Low})=1491$, showing that the dataset contains far more low-impact models than high-impact ones.

### 5. Statistical Insights
*   **Statistical Significance:** The difference between the two groups is highly statistically significant ($p=0.000$). We can confidently reject the null hypothesis that evaluation rates are the same for both groups.
*   **Disparity in Scrutiny:** The data suggests that AI systems classified as "High Impact" are subjected to far more rigorous external scrutiny (independent evaluation) than those classified as "Low Impact."
*   **Prevalence:** Despite "Low Impact" AI being much more common in this dataset ($n=1491$ vs $n=227$), a negligible fraction of them (1.3%) undergo independent evaluation.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
