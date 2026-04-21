# Experiment 34: node_3_16

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_16` |
| **ID in Run** | 34 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T02:37:19.603831+00:00 |
| **Runtime** | 314.1s |
| **Parent** | `node_2_5` |
| **Children** | `node_4_14`, `node_4_38` |
| **Creation Index** | 35 |

---

## Hypothesis

> PII Privacy-Fairness Disconnect: The presence of PII (`29_contains_pii`) in a
system is not a statistically significant predictor of the implementation of
'Bias Mitigation' (`55_disparity_mitigation`), suggesting privacy controls are
decoupled from fairness controls.

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

**Objective:** Assess if privacy sensitivity drives broader ethical governance (fairness).

### Steps
- 1. Filter for `eo13960_scored`.
- 2. Clean `29_contains_pii` and `55_disparity_mitigation` into binary (1/0) variables.
- 3. Create a contingency table (PII Yes/No vs Bias Mitigation Yes/No).
- 4. Run a Chi-square test of independence.
- 5. Compare the conditional probability P(Bias Mitigated | PII) vs P(Bias Mitigated | No PII).

### Deliverables
- Contingency table, Chi-square statistics, conditional probabilities, and interpretation of the association.

---

## Analysis

The experiment successfully tested the 'Privacy-Fairness Disconnect' hypothesis
using the 'eo13960_scored' dataset. By implementing a text-parsing logic for the
'62_disparity_mitigation' column, the analysis recovered 981 valid data points.

1. **Statistical Results**: The Chi-square test yielded a p-value of 0.0052,
which is below the alpha of 0.05. This indicates a **statistically significant
association** between the presence of PII and the implementation of bias
mitigation.

2. **Hypothesis Evaluation**: The original hypothesis stating that PII is *not*
a predictor of fairness controls was **rejected**. Systems containing PII are
more likely to implement bias mitigation (9.94%) compared to those without PII
(4.32%).

3. **Practical Insight**: While the association is statistically significant, a
practical 'governance gap' persists. Even among systems processing PII, 90%
(154/171) lack explicit bias mitigation controls. The 'disconnect' is not total,
but privacy sensitivity is clearly not a strong enough driver to ensure
universal fairness governance.

---

## Review

The experiment successfully tested the 'Privacy-Fairness Disconnect' hypothesis
using the 'eo13960_scored' dataset. By implementing a text-parsing logic for the
'62_disparity_mitigation' column, the analysis recovered 981 valid data points.

1. **Statistical Results**: The Chi-square test yielded a p-value of 0.0052,
which is below the alpha of 0.05. This indicates a **statistically significant
association** between the presence of PII and the implementation of bias
mitigation.

2. **Hypothesis Evaluation**: The original hypothesis stating that PII is *not*
a predictor of fairness controls was **rejected**. Systems containing PII are
more likely to implement bias mitigation (9.94%) compared to those without PII
(4.32%).

3. **Practical Insight**: While the association is statistically significant, a
practical 'governance gap' persists. Even among systems processing PII, 90%
(154/171) lack explicit bias mitigation controls. The 'disconnect' is not total,
but privacy sensitivity is clearly not a strong enough driver to ensure
universal fairness governance.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

print("Dataset loaded. Processing 'eo13960_scored'...")

eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# Columns
col_pii = '29_contains_pii'
col_bias = '62_disparity_mitigation'

# --- Step 1: Clean PII Variable ---
# We only keep rows where PII status is explicitly 'Yes' or 'No' to ensure validity.
def clean_pii(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ['yes', 'y', 'true', '1']:
        return 1
    elif s in ['no', 'n', 'false', '0']:
        return 0
    return np.nan

eo_df['pii_binary'] = eo_df[col_pii].apply(clean_pii)

# Drop rows where PII is unknown (NaN)
analysis_df = eo_df.dropna(subset=['pii_binary']).copy()
print(f"Rows after filtering for valid PII (Yes/No): {len(analysis_df)}")
print(f"PII Distribution:\n{analysis_df['pii_binary'].value_counts()}")

# --- Step 2: Clean Bias Mitigation Variable ---
# Logic: NaN is treated as 0 (No mitigation listed).
# Text is analyzed: 'N/A', 'None', 'Not applicable' -> 0.
# Substantive text -> 1.

def clean_bias_mitigation(val):
    if pd.isna(val):
        return 0  # Treat missing as no mitigation
    
    text = str(val).strip().lower()
    
    # Check for empty or essentially empty strings
    if not text or text == 'nan':
        return 0
        
    # Keywords indicating LACK of mitigation or irrelevance
    negatives = [
        r'^n/a',
        r'^none',
        r'not applicable',
        r'no demographic',
        r'does not use',
        r'not safety',
        r'waived',
        r'not using pii'
    ]
    
    for pattern in negatives:
        if re.search(pattern, text):
            return 0
            
    # If text is present and not a negative keyword, assume it describes a mitigation
    return 1

analysis_df['bias_binary'] = analysis_df[col_bias].apply(clean_bias_mitigation)

print(f"Bias Mitigation Distribution:\n{analysis_df['bias_binary'].value_counts()}")

# --- Step 3: Analysis ---
if len(analysis_df) == 0:
    print("Error: No data remaining.")
else:
    # Contingency Table
    contingency = pd.crosstab(analysis_df['pii_binary'], analysis_df['bias_binary'])
    
    # Ensure 2x2 shape
    for i in [0, 1]:
        if i not in contingency.index:
            contingency.loc[i] = [0, 0]
        if i not in contingency.columns:
            contingency[i] = 0
            
    contingency = contingency.sort_index().sort_index(axis=1)
    contingency.index = ['No PII', 'Has PII']
    contingency.columns = ['No Mitigation', 'Has Mitigation']
    
    print("\n--- Contingency Table ---")
    print(contingency)
    
    # Chi-square
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square: {chi2:.4f}, p-value: {p:.4e}")
    
    # Probabilities
    # P(Mitigation | PII)
    p_mit_pii = contingency.loc['Has PII', 'Has Mitigation'] / contingency.loc['Has PII'].sum()
    # P(Mitigation | No PII)
    p_mit_no_pii = contingency.loc['No PII', 'Has Mitigation'] / contingency.loc['No PII'].sum()
    
    print(f"P(Mitigation | PII)    = {p_mit_pii:.2%}")
    print(f"P(Mitigation | No PII) = {p_mit_no_pii:.2%}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Greens')
    plt.title('PII Presence vs. Fairness/Bias Mitigation')
    plt.show()
    
    # Conclusion
    print("\n--- Conclusion ---")
    if p < 0.05:
        print("Statistically Significant Association detected.")
    else:
        print("No Statistically Significant Association (Supports 'Disconnect' Hypothesis).")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Dataset loaded. Processing 'eo13960_scored'...
Rows after filtering for valid PII (Yes/No): 981
PII Distribution:
pii_binary
0.0    810
1.0    171
Name: count, dtype: int64
Bias Mitigation Distribution:
bias_binary
0    929
1     52
Name: count, dtype: int64

--- Contingency Table ---
         No Mitigation  Has Mitigation
No PII             775              35
Has PII            154              17

Chi-Square: 7.8012, p-value: 5.2212e-03
P(Mitigation | PII)    = 9.94%
P(Mitigation | No PII) = 4.32%

--- Conclusion ---
Statistically Significant Association detected.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Heatmap / Confusion Matrix-style Contingency Table.
*   **Purpose:** This plot visualizes the frequency distribution between two categorical variables: the presence of Personally Identifiable Information (PII) and the status of Fairness/Bias Mitigation. It allows for a quick comparison of counts across these intersecting categories using color intensity.

### 2. Axes
*   **X-Axis:**
    *   **Label:** Represents the mitigation status.
    *   **Categories:** "No Mitigation" and "Has Mitigation".
*   **Y-Axis:**
    *   **Label:** Represents the PII status.
    *   **Categories:** "No PII" and "Has PII".
*   **Color Scale (Z-Axis):**
    *   **Legend:** A vertical color bar on the right.
    *   **Range:** The scale ranges from approximately 0 (lightest green/white) to over 700 (darkest green).
    *   **Units:** Raw counts of occurrences.

### 3. Data Trends
*   **High Areas:**
    *   The overwhelming majority of the data falls in the **top-left quadrant** ("No PII" and "No Mitigation"), with a count of **775**. This cell is colored the darkest green, indicating the highest density.
*   **Low Areas:**
    *   The **bottom-right quadrant** ("Has PII" and "Has Mitigation") contains the lowest count at **17**, indicated by the lightest color.
    *   The **top-right quadrant** ("No PII" and "Has Mitigation") is also very low with a count of **35**.
*   **Secondary Cluster:**
    *   The **bottom-left quadrant** ("Has PII" and "No Mitigation") has a moderate count of **154**, represented by a pale green color.

### 4. Annotations and Legends
*   **Title:** "PII Presence vs. Fairness/Bias Mitigation" clearly defines the two variables being compared.
*   **Cell Annotations:** Each of the four grid cells contains a numerical annotation (775, 35, 154, 17) providing the exact count for that intersection.
*   **Color Bar:** The gradient bar on the right provides a visual reference for the magnitude of the counts, moving from light green (low) to dark green (high).

### 5. Statistical Insights
*   **Prevalence of "No Mitigation":** The vast majority of the dataset (929 out of 981 total entries, or roughly **94.7%**) lacks Fairness/Bias Mitigation, regardless of PII presence.
*   **Prevalence of "No PII":** Most entries (810 out of 981, or roughly **82.6%**) do not contain PII.
*   **Intersection of PII and Mitigation:** Even when PII is present (171 cases total), mitigation is rarely applied. Only 17 out of 171 PII-positive cases (approx. **9.9%**) have mitigation.
*   **Conclusion:** The standard operating procedure for this dataset appears to be omitting mitigation. While mitigation is slightly more common proportionally when PII is present (9.9%) compared to when it is not (4.3%), the overall adoption of Fairness/Bias mitigation strategies is extremely low.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
