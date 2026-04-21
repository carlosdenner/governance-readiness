# Experiment 57: node_4_23

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_23` |
| **ID in Run** | 57 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:45:45.966450+00:00 |
| **Runtime** | 248.3s |
| **Parent** | `node_3_12` |
| **Children** | `node_5_14`, `node_5_41` |
| **Creation Index** | 58 |

---

## Hypothesis

> Risk-Governance Decoupling: There is no statistically significant association
between self-reported 'Impact Type' (High vs Low) and the completion of 'Impact
Assessments' in the federal inventory, suggesting a failure to scale oversight
with risk.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7903 (Likely True) |
| **Posterior** | 0.2692 (Likely False) |
| **Surprise** | -0.6253 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 6.0 |
| Maybe True | 24.0 |
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

**Objective:** Test if higher-risk AI use cases strictly correlate with higher compliance in completing Impact Assessments.

### Steps
- 1. Load 'eo13960_scored' subset.
- 2. Filter rows with valid '17_impact_type' (High vs Low/Non-High) and '52_impact_assessment' (Yes/No).
- 3. Create a cross-tabulation of Impact Type vs. Assessment Completion.
- 4. Run a Chi-Square Test of Independence.
- 5. Calculate the compliance rate for each impact tier.

### Deliverables
- Cross-tabulation table; Chi-Square statistic and p-value; Comparative bar plot of compliance rates.

---

## Analysis

The experiment was successfully executed and robustly handled the data cleaning
requirements identified in previous iterations. By correctly mapping 'Neither'
to Low Impact and 'Both' to High Impact, the script successfully isolated 157
valid use cases for analysis (up from 25 in the failed debug attempt).

The Chi-Square test of independence yielded a statistic of 26.36 with a p-value
of 2.83e-07, providing strong statistical evidence to **reject the null
hypothesis** of Risk-Governance Decoupling. The results demonstrate a
significant positive association between risk level and oversight: High Impact
use cases have a 68.6% impact assessment compliance rate, compared to just 24.5%
for Low Impact cases. While this suggests the governance system is functioning
as intended (scaling oversight with risk), a critical gap remains: 31.4% of
self-identified High Impact use cases still lack the required impact
assessments.

---

## Review

The experiment was successfully executed and robustly handled the data cleaning
requirements identified in previous iterations. By correctly mapping 'Neither'
to Low Impact and 'Both' to High Impact, the script successfully isolated 157
valid use cases for analysis. The Chi-Square test (p < 0.001) provided strong
evidence to **reject the hypothesis** of Risk-Governance Decoupling. Instead of
decoupling, the data shows a significant positive association between risk level
and oversight: High Impact use cases have a 68.6% impact assessment compliance
rate, compared to 24.5% for Low Impact cases. While this indicates that
governance mechanisms are scaling with risk as intended, a critical gap remains,
as nearly one-third (31.4%) of High Impact use cases still lack the required
assessments.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Load dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback for different execution environments if needed, though instruction says use given dataset
    df = pd.read_csv(f'../{file_path}', low_memory=False)

# Filter for EO 13960 subset
subset = df[df['source_table'] == 'eo13960_scored'].copy()

# Clean and Map Impact Type
def map_impact(val):
    val_str = str(val).strip()
    if val_str in ['Both', 'Rights-Impacting', 'Safety-Impacting', 'Safety-impacting', 'Rights-Impacting\n']:
        return 'High Impact'
    elif val_str in ['Neither', 'Low', 'None']:
        return 'Low Impact'
    return 'Unknown'

# Clean and Map Assessment Status
def map_assessment(val):
    val_str = str(val).strip().upper()
    if 'YES' in val_str:
        return 'Yes'
    elif 'NO' in val_str or 'PLANNED' in val_str:
        return 'No'
    return 'Unknown'

subset['impact_tier'] = subset['17_impact_type'].apply(map_impact)
subset['assessment_done'] = subset['52_impact_assessment'].apply(map_assessment)

# Filter for valid analysis rows
analysis_df = subset[
    (subset['impact_tier'] != 'Unknown') & 
    (subset['assessment_done'] != 'Unknown')
].copy()

print(f"Analysis Subset Shape: {analysis_df.shape}")
print("\nDistribution of Impact Tiers:")
print(analysis_df['impact_tier'].value_counts())

# Create Contingency Table
contingency_table = pd.crosstab(analysis_df['impact_tier'], analysis_df['assessment_done'])
print("\nContingency Table (Impact Tier vs Assessment Completion):")
print(contingency_table)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Test Results:\nStatistic: {chi2:.4f}\np-value: {p:.4e}\nDoF: {dof}")

# Calculate Compliance Rates
rates = analysis_df.groupby('impact_tier')['assessment_done'].value_counts(normalize=True).unstack().fillna(0)
compliance_rates = rates['Yes'] * 100
print("\nCompliance Rates (% Assessment Completed):")
print(compliance_rates)

# Visualization
plt.figure(figsize=(10, 6))
# Calculate percentages for the plot
prop_df = (analysis_df.groupby(['impact_tier'])['assessment_done']
           .value_counts(normalize=True)
           .rename('percentage')
           .reset_index())
prop_df['percentage'] *= 100

sns.barplot(x='impact_tier', y='percentage', hue='assessment_done', data=prop_df, palette='viridis')
plt.title('Impact Assessment Compliance by Risk Tier')
plt.ylabel('Percentage of Use Cases (%)')
plt.xlabel('Impact Tier')
plt.ylim(0, 100)
plt.legend(title='Assessment Completed')

# Annotate bars
for p in plt.gca().patches:
    txt = f"{p.get_height():.1f}%"
    plt.gca().text(p.get_x() + p.get_width()/2, p.get_height() + 1, txt, ha='center')

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Analysis Subset Shape: (157, 198)

Distribution of Impact Tiers:
impact_tier
Low Impact     106
High Impact     51
Name: count, dtype: int64

Contingency Table (Impact Tier vs Assessment Completion):
assessment_done  No  Yes
impact_tier             
High Impact      16   35
Low Impact       80   26

Chi-Square Test Results:
Statistic: 26.3604
p-value: 2.8328e-07
DoF: 1

Compliance Rates (% Assessment Completed):
impact_tier
High Impact    68.627451
Low Impact     24.528302
Name: Yes, dtype: float64


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Grouped Bar Chart (or Clustered Bar Chart).
*   **Purpose:** The chart is designed to compare the completion status of impact assessments ("Yes" vs. "No") across two distinct categories of risk ("High Impact" vs. "Low Impact"). It allows viewers to visualize the relationship between the severity of the risk and the likelihood of a compliance assessment being completed.

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Title:** "Impact Tier"
    *   **Labels:** "High Impact" and "Low Impact". These represent categorical variables defining the risk level of the use cases.
*   **Y-Axis (Vertical):**
    *   **Title:** "Percentage of Use Cases (%)"
    *   **Value Range:** 0 to 100.
    *   **Increments:** The axis is marked in increments of 20 (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **High Impact Tier:**
    *   **Trend:** There is a higher rate of compliance in this tier.
    *   **Tallest Bar:** The "Yes" bar (Assessment Completed) is the dominant bar at **68.6%**.
    *   **Shortest Bar:** The "No" bar is significantly lower at **31.4%**.
*   **Low Impact Tier:**
    *   **Trend:** There is a strong inverse trend compared to the High Impact tier; non-compliance is the norm here.
    *   **Tallest Bar:** The "No" bar (Assessment Not Completed) is the dominant bar at **75.5%**.
    *   **Shortest Bar:** The "Yes" bar is the shortest in this group at **24.5%**.
*   **Overall Pattern:** There is a clear inversion of behavior between the two tiers. High Impact cases are mostly assessed, whereas Low Impact cases are mostly unassessed.

### 4. Annotations and Legends
*   **Legend:** Located in the top-right corner titled "Assessment Completed".
    *   **Dark Blue/Grey:** Represents "Yes" (Assessment was completed).
    *   **Green:** Represents "No" (Assessment was not completed).
*   **Annotations:**
    *   Each bar is annotated with a specific percentage value on top (e.g., 68.6%, 31.4%, 24.5%, 75.5%).
    *   There is a small, potentially anomalous label of "0.0%" visible at the base of the High Impact group, likely an artifact of the plotting software or representing a negligible data point not visible as a bar.

### 5. Statistical Insights
*   **Prioritization of Risk:** The data indicates a strong prioritization strategy where resources for impact assessments are focused on "High Impact" use cases. Compliance is nearly 3x higher for High Impact cases (68.6%) compared to Low Impact cases (24.5%).
*   **Gap in High Impact Compliance:** Despite the focus on High Impact, nearly one-third (**31.4%**) of High Impact use cases still lack a completed assessment. Depending on the regulatory environment, this could represent a significant liability.
*   **Low Impact Neglect:** Three-quarters (**75.5%**) of Low Impact use cases have not undergone an impact assessment, suggesting that these are systematically deprioritized or considered low risk enough to bypass formal assessment processes.
*   **Summation:** The percentages within each tier sum to exactly 100% (68.6 + 31.4 = 100; 24.5 + 75.5 = 100), confirming that the data represents the full distribution of use cases within each specific tier.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
