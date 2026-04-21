# Experiment 196: node_5_76

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_76` |
| **ID in Run** | 196 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:14:04.671679+00:00 |
| **Runtime** | 347.8s |
| **Parent** | `node_4_46` |
| **Children** | None |
| **Creation Index** | 197 |

---

## Hypothesis

> The 'Siloed' Risk Management: Systems classified as 'Safety-Impacted' are
significantly less likely to implement 'Disparity Mitigation' controls compared
to 'Rights-Impacted' systems, indicating a fragmented approach where safety-
critical systems overlook fairness risks.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.2555 (Likely False) |
| **Surprise** | -0.5934 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 29.0 |
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

**Objective:** Compare the implementation rate of bias/disparity mitigation controls between Safety-Impacted and Rights-Impacted AI deployments.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'eo13960_scored'.
- 2. Clean '17_impact_type' to isolate rows strictly labeled as 'Safety-Impacted' or 'Rights-Impacted' (exclude others).
- 3. Convert '55_disparity_mitigation' into a binary variable (Yes vs No/Other).
- 4. Calculate the compliance rate for both groups.
- 5. Perform a Chi-Square test of independence.

### Deliverables
- Contingency table of Impact Type vs Disparity Mitigation; Bar chart of mitigation rates; Chi-Square test statistics.

---

## Analysis

The experiment successfully tested the 'Siloed Risk Management' hypothesis by
analyzing 77 distinctively classified AI systems (59 Rights-Impacted and 18
Safety-Impacted) from the EO 13960 dataset. Using the corrected column
'62_disparity_mitigation', the results revealed a universal absence of
documented bias mitigation controls: 0.0% of Rights-Impacted systems (n=59) and
0.0% of Safety-Impacted systems (n=18) reported having these measures. Due to
the complete lack of positive cases in either group, a Chi-Square test could not
be performed. These findings do not support the hypothesis that Safety-Impacted
systems are *uniquely* or *significantly less* likely to have these controls
than Rights-Impacted ones; rather, they indicate a systemic gap where *neither*
high-impact archetype currently evidences disparity mitigation in the available
data.

---

## Review

The experiment was successfully executed and faithfully tested the 'Siloed Risk
Management' hypothesis. After resolving initial column naming issues, the
analysis correctly identified the relevant data subset (59 Rights-Impacted and
18 Safety-Impacted systems) and the correct target variable
('62_disparity_mitigation'). The results revealed a universal absence of
disparity mitigation controls across the studied federal AI inventory, with a 0%
implementation rate for both 'Rights-Impacted' and 'Safety-Impacted' systems.
While the lack of variance (0 vs 0) prevented the planned Chi-Square test, the
code handled this edge case correctly. The findings do not support the specific
hypothesis of a 'divide' between safety and rights systems but rather indicate a
systemic gap in governance across both high-impact categories.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the dataset
filepath = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(filepath, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored records
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Validated column name from previous debug
mitigation_col = '62_disparity_mitigation'

# Clean Impact Type
def categorize_impact(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower().strip()
    
    # Strict separation: Exclude 'Both'
    if 'both' in val_str:
        return None
    
    is_safety = 'safety' in val_str
    is_rights = 'rights' in val_str
    
    if is_safety and not is_rights:
        return 'Safety-Impacted'
    elif is_rights and not is_safety:
        return 'Rights-Impacted'
    return None

eo_data['impact_category'] = eo_data['17_impact_type'].apply(categorize_impact)

# Filter for analysis
analysis_df = eo_data[eo_data['impact_category'].notna()].copy()

# Binarize Mitigation
def check_mitigation(val):
    if pd.isna(val):
        return 0
    # Looking for affirmative 'Yes'
    if str(val).lower().strip().startswith('yes'):
        return 1
    return 0

analysis_df['has_mitigation'] = analysis_df[mitigation_col].apply(check_mitigation)

# Create Contingency Table
contingency = pd.crosstab(analysis_df['impact_category'], analysis_df['has_mitigation'])
# Ensure we have both columns [0, 1]
contingency = contingency.reindex(columns=[0, 1], fill_value=0)
contingency.columns = ['No', 'Yes']

print("\nContingency Table (Impact Type vs. Disparity Mitigation):")
print(contingency)

# Calculate Rates
rates = analysis_df.groupby('impact_category')['has_mitigation'].mean()
print("\nMitigation Rates:")
print(rates)

# Statistical Test Check
total_positives = contingency['Yes'].sum()

if total_positives == 0:
    print("\nCannot perform Chi-Square test: No positive cases ('Yes') observed in either group.")
    print("Both groups have 0% compliance rate.")
    p_value = 1.0  # Technically undefined, but effectively no difference
else:
    # Only run test if we have data
    try:
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"\nChi-Square Statistic: {chi2:.4f}")
        print(f"P-value: {p:.6f}")
    except ValueError as e:
        print(f"\nStatistical test failed: {e}")

# Visualization
plt.figure(figsize=(8, 6))
colors = ['#e74c3c' if 'Safety' in idx else '#3498db' for idx in rates.index]
bars = plt.bar(rates.index, rates.values * 100, color=colors)

plt.title('Disparity Mitigation Controls: Safety vs. Rights Impacted Systems')
plt.ylabel('Percentage with Mitigation (%)')
plt.ylim(0, 100)

# Add annotations
for bar, idx in zip(bars, rates.index):
    height = bar.get_height()
    n_yes = contingency.loc[idx, 'Yes']
    n_total = contingency.loc[idx].sum()
    # If height is 0, place text slightly above 0
    y_pos = height + 1 if height > 0 else 2
    plt.text(bar.get_x() + bar.get_width()/2, y_pos, 
             f"{height:.1f}%\n(n={n_yes}/{n_total})", 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: 
Contingency Table (Impact Type vs. Disparity Mitigation):
                 No  Yes
impact_category         
Rights-Impacted  59    0
Safety-Impacted  18    0

Mitigation Rates:
impact_category
Rights-Impacted    0.0
Safety-Impacted    0.0
Name: has_mitigation, dtype: float64

Cannot perform Chi-Square test: No positive cases ('Yes') observed in either group.
Both groups have 0% compliance rate.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot.
*   **Purpose:** The plot is designed to compare the percentage of systems that have implemented "Disparity Mitigation Controls" across two distinct categories of systems: "Rights-Impacted" and "Safety-Impacted."

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Percentage with Mitigation (%)"
    *   **Range:** 0 to 100.
    *   **Scale:** Linear, with tick marks every 20 units (0, 20, 40, 60, 80, 100).
*   **X-Axis:**
    *   **Label:** The axis represents categorical system types.
    *   **Categories:** Two distinct groups are labeled: "Rights-Impacted" and "Safety-Impacted."

### 3. Data Trends
*   **Bar Height/Pattern:** There are no visible bars extending upwards from the x-axis.
*   **Values:** Both categories exhibit a value of exactly 0%.
*   **Comparison:** There is no variation between the groups; both show a complete absence of the metric being measured (mitigation controls).

### 4. Annotations and Legends
*   **Annotations:**
    *   **Above "Rights-Impacted":** The text reads **"0.0% (n=0/59)"**. This indicates that out of a sample size of 59 systems in this category, 0 had mitigation controls.
    *   **Above "Safety-Impacted":** The text reads **"0.0% (n=0/18)"**. This indicates that out of a sample size of 18 systems in this category, 0 had mitigation controls.
*   **Legend:** There is no legend provided or required, as the categories are explicitly labeled on the x-axis.

### 5. Statistical Insights
*   **Universal Lack of Mitigation:** The most significant insight is that **none** of the systems surveyed in this study have implemented disparity mitigation controls. This applies to both systems that impact human rights and systems that impact safety.
*   **Sample Size Context:** The total sample size analyzed is 77 systems (59 Rights-Impacted + 18 Safety-Impacted). The fact that the percentage is 0.0% across a sample of this size suggests a systemic gap in the deployment or documentation of fairness/disparity controls within the specific dataset or domain being analyzed.
*   **Risk Implication:** Given the high-stakes nature of the categories (Rights and Safety), the complete absence of disparity mitigation controls highlights a potential area of significant ethical or regulatory risk.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
