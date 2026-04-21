# Experiment 163: node_5_63

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_63` |
| **ID in Run** | 163 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:44:12.393157+00:00 |
| **Runtime** | 288.8s |
| **Parent** | `node_4_44` |
| **Children** | None |
| **Creation Index** | 164 |

---

## Hypothesis

> The 'High-Stakes' Compliance Variance: In the EO13960 inventory, systems in the
'Health' topic area are significantly more likely to report having 'Disparity
Mitigation' measures than systems in 'Government Services' or 'Law Enforcement',
reflecting domain-specific regulatory pressure.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.4032 (Maybe False) |
| **Posterior** | 0.1374 (Likely False) |
| **Surprise** | -0.3190 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 9.0 |
| Uncertain | 0.0 |
| Maybe False | 21.0 |
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

**Objective:** Compare Disparity Mitigation rates across key Topic Areas.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' (EO13960 subset).
- 2. Filter for top topic areas: 'Health', 'Law Enforcement', 'Government Services'.
- 3. Clean '62_disparity_mitigation' into Binary (Yes/No).
- 4. Perform a Chi-Square Test (or ANOVA if comparing multiple groups via rates).
- 5. Use post-hoc analysis to identify which sector differs most.

### Deliverables
- Bar chart of Mitigation Rates by Sector; Statistical test results.

---

## Analysis

The experiment successfully analyzed 478 records from the EO13960 dataset to
evaluate the 'High-Stakes Compliance Variance' hypothesis. The results
**contradict** the specific hypothesis that 'Health' systems are the most likely
to report disparity mitigation.

1. **Statistical Significance**: The Chi-Square test yielded a p-value of
6.22e-05, confirming a highly statistically significant difference in reporting
rates between the sectors.
2. **Directionality Reversal**: Contrary to the expectation that Health would
lead, **'Law Enforcement'** systems demonstrated the highest rate of disparity
mitigation reporting (14.0%). This is more than triple the rate of 'Health'
systems (4.3%) and ten times that of 'Government Services' (1.3%).
3. **Governance Insight**: While overall reporting is low across all sectors,
the data suggests that scrutiny regarding bias and fairness is currently more
operationalized in federal Law Enforcement AI deployments than in Health or
general administrative services.

---

## Review

The experiment successfully analyzed 478 records from the EO13960 dataset to
evaluate the 'High-Stakes Compliance Variance' hypothesis. The results
**reject** the hypothesis that 'Health' systems are the most likely to report
disparity mitigation measures.

**Key Findings:**
1.  **Hypothesis Rejection:** Contrary to the expectation that Health systems
would lead in fairness controls, **'Law Enforcement'** systems demonstrated the
highest rate of disparity mitigation reporting (14.0%).
2.  **Statistical Significance:** The difference is highly significant (Chi-
Square p < 0.0001). Law Enforcement reporting rates are more than triple that of
'Health' (4.3%) and ten times higher than 'Government Services' (1.3%).
3.  **Governance Insight:** The data indicates that regulatory pressure or
internal controls regarding bias are currently more operationalized in federal
Law Enforcement AI deployments—likely due to their coercive nature—than in
Health or general administrative services, where reporting rates are minimal
(<5%).

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO13960 subset
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# Map Topics to Experiment Categories
# Based on previous output, we map the actual dataset values to the target labels
topic_mapping = {
    'Health & Medical': 'Health',
    'Law & Justice': 'Law Enforcement',
    'Government Services (includes Benefits and Service Delivery)': 'Government Services'
}

eo_df['topic_mapped'] = eo_df['8_topic_area'].map(topic_mapping)

# Filter for only the mapped topics
subset = eo_df.dropna(subset=['topic_mapped']).copy()

print(f"Filtered subset size: {len(subset)}")
print("Counts per topic in subset:")
print(subset['topic_mapped'].value_counts())

# Clean Mitigation Column (62_disparity_mitigation)
def clean_mitigation(val):
    if pd.isna(val):
        return 0
    
    val_str = str(val).lower().strip()
    
    # explicit negative indicators
    negative_starts = ['n/a', 'no ', 'none', 'not ', '0', 'false']
    if any(val_str.startswith(x) for x in negative_starts):
        return 0
    
    # specific negative phrases found in previous inspection
    if 'does not take into account' in val_str:
        return 0
    if 'not safety or rights-impacting' in val_str:
        return 0
        
    # If it contains content that isn't negative, assume it describes a mitigation
    return 1

subset['has_mitigation'] = subset['62_disparity_mitigation'].apply(clean_mitigation)

# Verify cleaning
print("\nMitigation distribution (Binary) by Topic:")
print(subset.groupby('topic_mapped')['has_mitigation'].value_counts().unstack())

# Contingency Table
contingency = pd.crosstab(subset['topic_mapped'], subset['has_mitigation'])
print("\nContingency Table:")
print(contingency)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square Test Results:\nChi2 Statistic: {chi2:.4f}, p-value: {p:.4e}")

# Calculate Rates
rates = subset.groupby('topic_mapped')['has_mitigation'].mean()
print("\nMitigation Rates by Sector:")
print(rates)

# Visualization
plt.figure(figsize=(10, 6))
ax = rates.plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Disparity Mitigation Reporting Rates by Topic Area')
plt.ylabel('Proportion Reporting Mitigation')
plt.xlabel('Topic Area')
plt.ylim(0, 1.0)
plt.axhline(y=rates.mean(), color='r', linestyle='--', label=f'Mean Rate ({rates.mean():.2f})')

# Add value labels
for i, v in enumerate(rates):
    ax.text(i, v + 0.02, f"{v:.1%}", ha='center')

plt.legend()
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Filtered subset size: 478
Counts per topic in subset:
topic_mapped
Health                 233
Government Services    159
Law Enforcement         86
Name: count, dtype: int64

Mitigation distribution (Binary) by Topic:
has_mitigation         0   1
topic_mapped                
Government Services  157   2
Health               223  10
Law Enforcement       74  12

Contingency Table:
has_mitigation         0   1
topic_mapped                
Government Services  157   2
Health               223  10
Law Enforcement       74  12

Chi-Square Test Results:
Chi2 Statistic: 19.3704, p-value: 6.2198e-05

Mitigation Rates by Sector:
topic_mapped
Government Services    0.012579
Health                 0.042918
Law Enforcement        0.139535
Name: has_mitigation, dtype: float64


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is a detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The chart is designed to compare categorical data, specifically the proportion of "Disparity Mitigation Reporting" across three different topic areas.

### 2. Axes
*   **X-axis:**
    *   **Title:** "Topic Area"
    *   **Labels:** The axis displays three distinct categories: "Government Services", "Health", and "Law Enforcement".
*   **Y-axis:**
    *   **Title:** "Proportion Reporting Mitigation"
    *   **Range:** The scale ranges from **0.0 to 1.0**, representing a probability or proportion (0% to 100%).
    *   **Tick Marks:** The axis is marked at intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
*   **Tallest Bar:** "Law Enforcement" exhibits the highest reporting rate among the three categories.
*   **Shortest Bar:** "Government Services" shows the lowest reporting rate.
*   **Pattern:** There is a visible disparity between the categories. While Government Services and Health have relatively minimal reporting rates (visually small bars), Law Enforcement has a noticeably higher rate, exceeding the other two combined.

### 4. Annotations and Legends
*   **Legend:** Located in the top right corner.
    *   **Red dashed line:** Represented as "Mean Rate (0.07)", indicating the average reporting rate across all data points is 0.07 (or 7%).
    *   **Blue bar icon:** Labeled "has_mitigation", indicating the bars represent the presence of mitigation reporting.
*   **Bar Annotations:** Specific percentage values are placed directly above each bar for precision:
    *   Government Services: **1.3%**
    *   Health: **4.3%**
    *   Law Enforcement: **14.0%**
*   **Reference Line:** A horizontal red dashed line spans the width of the graph at the y-position corresponding to the mean rate of 0.07.

### 5. Statistical Insights
*   **Law Enforcement is an Outlier:** The Law Enforcement category (14.0%) is the only topic area performing above the overall mean rate of 7%. In fact, its rate is exactly double the mean.
*   **Below Average Performance:** Both Government Services (1.3%) and Health (4.3%) fall significantly below the mean rate. Government Services reports mitigation efforts at a rate more than 5 times lower than the average.
*   **Overall Low Reporting:** Despite the Y-axis extending to 1.0 (100%), the highest value on the chart is only 0.14 (14%). This suggests that disparity mitigation reporting is generally a low-frequency event across all observed topic areas.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
