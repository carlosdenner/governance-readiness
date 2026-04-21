# Experiment 206: node_5_79

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_79` |
| **ID in Run** | 206 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:37:30.406133+00:00 |
| **Runtime** | 283.0s |
| **Parent** | `node_4_11` |
| **Children** | `node_6_86` |
| **Creation Index** | 207 |

---

## Hypothesis

> Governance Competency Clustering: Organizations that implement 'Impact
Assessments' are statistically highly likely to also implement 'Stakeholder
Consultation', suggesting these controls form a coupled 'Accountability' bundle.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8952 (Likely True) |
| **Posterior** | 0.9643 (Definitely True) |
| **Surprise** | +0.0829 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 19.0 |
| Maybe True | 11.0 |
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

**Objective:** Identify co-occurrence patterns in governance controls to validate competency bundles.

### Steps
- 1. Load 'eo13960_scored'.
- 2. Extract binary vectors for '52_impact_assessment' and '63_stakeholder_consult'.
- 3. Construct a 2x2 contingency table (Impact Assessment Yes/No vs Stakeholder Consult Yes/No).
- 4. Calculate the Phi coefficient (correlation for binary vars) and the Odds Ratio.
- 5. If Phi > 0.3 and p < 0.05, confirm the clustering hypothesis.

### Deliverables
- Contingency table; Phi coefficient; Odds Ratio; Heatmap of the 2x2 matrix.

---

## Analysis

The experiment successfully validated the 'Governance Competency Clustering'
hypothesis using the EO 13960 dataset. By implementing text-analysis mapping to
extract binary indicators from the 'Impact Assessment' and 'Stakeholder
Consultation' columns, the analysis revealed a statistically significant
relationship between the two governance controls.

Key Findings:
1. **Statistical Significance**: The Chi-Square test resulted in a p-value of
3.99e-56, effectively zero, confirming that the occurrence of these two controls
is not independent.
2. **Strength of Association**: The Phi coefficient was 0.385, satisfying the
success criterion (> 0.3) and indicating a moderate positive correlation.
3. **Clustering Effect**: The Odds Ratio of 31.14 is extremely high, suggesting
that organizations implementing Impact Assessments are over 30 times more likely
to also perform Stakeholder Consultation than those that do not.
4. **Prevalence**: While the specific bundle (both controls present) is rare (25
out of 1,757 cases), the high correlation confirms that they function as a
coupled 'Accountability' bundle when they do appear.

The hypothesis is confirmed.

---

## Review

The experiment successfully validated the 'Governance Competency Clustering'
hypothesis using the EO 13960 dataset. By implementing text-analysis mapping to
extract binary indicators from the 'Impact Assessment' and 'Stakeholder
Consultation' columns, the analysis revealed a statistically significant
relationship between the two governance controls.

Key Findings:
1. **Statistical Significance**: The Chi-Square test resulted in a p-value of
3.99e-56, effectively zero, confirming that the occurrence of these two controls
is not independent.
2. **Strength of Association**: The Phi coefficient was 0.385, satisfying the
success criterion (> 0.3) and indicating a moderate positive correlation.
3. **Clustering Effect**: The Odds Ratio of 31.14 is extremely high, suggesting
that organizations implementing Impact Assessments are over 30 times more likely
to also perform Stakeholder Consultation than those that do not.
4. **Prevalence**: While the specific bundle (both controls present) is rare (25
out of 1,757 cases), the high correlation confirms that they function as a
coupled 'Accountability' bundle when they do appear.

The hypothesis is confirmed.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess

# function to install packages if needed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

# Try importing seaborn, install if missing
try:
    import seaborn as sns
except ImportError:
    install('seaborn')
    import seaborn as sns

# Try importing scipy, install if missing
try:
    from scipy.stats import chi2_contingency, fisher_exact
except ImportError:
    install('scipy')
    from scipy.stats import chi2_contingency, fisher_exact

# 1. Load dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for 'eo13960_scored'
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# 3. Data Cleaning & Mapping

# Mapping for Impact Assessment (Strict implementation)
def map_impact(val):
    s = str(val).strip().lower()
    if s == 'yes':
        return 1
    return 0

# Mapping for Stakeholder Consultation (Text analysis)
def map_consult(val):
    s = str(val).strip().lower()
    # Handle missing/nan
    if s == 'nan' or s == '':
        return 0
    # explicit negatives
    if 'none' in s:
        return 0
    if 'n/a' in s:
        return 0
    if 'waived' in s:
        return 0
    # default to positive if it contains content that isn't negative
    return 1

col_impact = '52_impact_assessment'
col_consult = '63_stakeholder_consult'

eo_df['impact_bin'] = eo_df[col_impact].apply(map_impact)
eo_df['consult_bin'] = eo_df[col_consult].apply(map_consult)

# 4. Construct Contingency Table
contingency = pd.crosstab(eo_df['impact_bin'], eo_df['consult_bin'])
contingency.index.name = "Impact Assessment"
contingency.columns.name = "Stakeholder Consult"

# Ensure 2x2
contingency = contingency.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

print("\nContingency Table:")
print(contingency)

# Check for zero variance
if (contingency.sum(axis=1) == 0).any() or (contingency.sum(axis=0) == 0).any():
    print("\nWarning: One of the variables has zero variance. Cannot compute meaningful statistics.")
    # Create a dummy result for the sake of flow, or exit
    phi = 0
    odds_ratio = 0
    p = 1.0
else:
    # 5. Statistical Tests
    # Use Chi2
    chi2, p, dof, ex = chi2_contingency(contingency)
    
    # Calculate Phi Coefficient
    a = contingency.loc[0, 0]
    b = contingency.loc[0, 1]
    c = contingency.loc[1, 0]
    d = contingency.loc[1, 1]
    
    phi_denom = np.sqrt((a+b)*(c+d)*(a+c)*(b+d))
    phi = (a*d - b*c) / phi_denom if phi_denom > 0 else 0
    
    # Odds Ratio
    if b*c == 0:
        odds_ratio = np.inf
    else:
        odds_ratio = (d * a) / (b * c)

    print(f"\nStatistical Results:")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-Value: {p:.4e}")
    print(f"Phi Coefficient: {phi:.4f}")
    print(f"Odds Ratio: {odds_ratio:.4f}")

    # Interpretation
    if p < 0.05 and phi > 0.3:
        print("\nResult: Hypothesis CONFIRMED. Significant positive clustering observed.")
    elif p < 0.05 and phi > 0:
        print("\nResult: Hypothesis WEAK. Significant but low correlation.")
    else:
        print("\nResult: Hypothesis REJECTED.")

# 6. Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No/Other', 'Yes'], yticklabels=['No/Other', 'Yes'])
plt.title(f'Competency Bundle: Impact Assessment vs Stakeholder Consult\n(Phi={phi:.2f}, OR={odds_ratio:.2f})')
plt.xlabel('Stakeholder Consultation')
plt.ylabel('Impact Assessment')
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: 
Contingency Table:
Stakeholder Consult     0   1
Impact Assessment            
0                    1659  37
1                      36  25

Statistical Results:
Chi-Square Statistic: 249.1465
P-Value: 3.9857e-56
Phi Coefficient: 0.3850
Odds Ratio: 31.1374

Result: Hypothesis CONFIRMED. Significant positive clustering observed.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Heatmap visualization of a Contingency Table (or Confusion Matrix).
*   **Purpose:** The plot displays the frequency distribution and co-occurrence of two categorical variables: **Impact Assessment** and **Stakeholder Consultation**. It helps identify how often these two competencies appear together versus independently.

### 2. Axes
*   **Y-Axis:**
    *   **Title:** Impact Assessment
    *   **Labels:** "No/Other" (Top row), "Yes" (Bottom row).
*   **X-Axis:**
    *   **Title:** Stakeholder Consultation
    *   **Labels:** "No/Other" (Left column), "Yes" (Right column).
*   **Value Ranges:** The axes represent categorical choices (Binary: Yes vs. No/Other) rather than numerical ranges. The values inside the grid represent counts.

### 3. Data Trends
*   **Highest Value (Dominant Category):** The top-left cell (**No/Other, No/Other**) contains the vast majority of the data points (**1659**). This indicates that in most observed cases, neither Impact Assessment nor Stakeholder Consultation was present. This cell is colored dark blue, indicating the highest density.
*   **Low Values:** The remaining three quadrants contain significantly lower counts:
    *   **No/Other, Yes:** 37
    *   **Yes, No/Other:** 36
    *   **Yes, Yes:** 25
*   **Visual Pattern:** The heatmap is extremely sparse/imbalanced. The color gradient clearly highlights the massive imbalance between the null cases (No/No) and the active cases (Yes).

### 4. Annotations and Legends
*   **Title:** "Competency Bundle: Impact Assessment vs Stakeholder Consult".
*   **Statistical Metrics (in Title):**
    *   **Phi=0.38:** Represents the Phi coefficient, a measure of association for two binary variables.
    *   **OR=31.14:** Represents the Odds Ratio.
*   **Cell Annotations:** Each cell contains the exact count of observations for that specific intersection of categories (1659, 37, 36, and 25).

### 5. Statistical Insights
*   **High Odds Ratio (OR = 31.14):** This is a significant finding. It suggests a very strong association between the two variables. Specifically, the odds of having an "Impact Assessment" are over 31 times higher if "Stakeholder Consultation" is also present, compared to when it is not.
*   **Moderate Correlation (Phi = 0.38):** The Phi coefficient indicates a moderate positive relationship between the two variables. While they are correlated, they are not perfectly predictive of one another.
*   **Co-occurrence vs. Isolation:**
    *   When "Impact Assessment" occurs (Total Yes = 36 + 25 = 61), it co-occurs with "Stakeholder Consultation" about 41% of the time (25/61).
    *   Similarly, when "Stakeholder Consultation" occurs (Total Yes = 37 + 25 = 62), it co-occurs with "Impact Assessment" about 40% of the time (25/62).
*   **Rarity of Competencies:** The "Yes" condition is rare for both variables within this dataset (occurring in less than 4% of the total 1,757 cases). The vast majority of the bundle lacks both specific competencies.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
