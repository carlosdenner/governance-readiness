# Experiment 167: node_6_19

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_19` |
| **ID in Run** | 167 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:56:57.058307+00:00 |
| **Runtime** | 284.0s |
| **Parent** | `node_5_4` |
| **Children** | `node_7_17` |
| **Creation Index** | 168 |

---

## Hypothesis

> Compliance Clustering: There is a strong positive dependency between 'Impact
Assessment' (Control 52) and 'AI Notice' (Control 59) in federal agencies,
suggesting that governance is often 'all-or-nothing' rather than piecemeal.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.9121 (Definitely True) |
| **Surprise** | +0.2042 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 30.0 |
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

**Objective:** Test if governance controls co-occur effectively.

### Steps
- 1. Filter `eo13960_scored` data.
- 2. Create binary variables for `52_impact_assessment` and `59_ai_notice`.
- 3. Create a contingency table.
- 4. Calculate the Chi-square statistic and the Phi coefficient (correlation for binary variables) to measure the strength of association.

### Deliverables
- Contingency table, Chi-square p-value, and Phi coefficient.

---

## Analysis

The experiment successfully tested the 'Compliance Clustering' hypothesis using
the EO 13960 dataset.

**Findings:**
1.  **Low Baseline Compliance:** Both controls are rare. Only 3.4% (61/1,757) of
systems had an Impact Assessment, and 4.0% (71/1,757) provided AI Notice.
2.  **Strong Conditional Probability:** The contingency table reveals a striking
dependency. Among systems with *No* Impact Assessment, only **2.5%** (43/1,696)
provided AI Notice. However, among systems *With* an Impact Assessment,
**45.9%** (28/61) provided AI Notice. This represents an ~18x increase in the
likelihood of transparency when impact assessments are present.
3.  **Statistical Significance:** The Chi-square test yielded a statistic of
274.5 and a p-value of 1.19e-61, decisively rejecting the null hypothesis of
independence.
4.  **Strength of Association:** The Phi coefficient of 0.40 indicates a
moderate-to-strong positive association, confirming that these governance
controls are highly correlated.

**Conclusion:**
The hypothesis is **supported**. Governance controls in federal AI projects
appear to be implemented in 'clusters' rather than incrementally. Agencies that
invest in deep governance (Impact Assessments) are significantly more likely to
also implement basic transparency (AI Notice), while the vast majority of
systems lack both.

---

## Review

The experiment successfully tested the 'Compliance Clustering' hypothesis using
the EO 13960 dataset.

**Findings:**
1.  **Low Baseline Compliance:** Both controls are rare. Only 3.4% (61/1,757) of
systems had an Impact Assessment, and 4.0% (71/1,757) provided AI Notice.
2.  **Strong Conditional Probability:** The contingency table reveals a striking
dependency. Among systems with *No* Impact Assessment, only **2.5%** (43/1,696)
provided AI Notice. However, among systems *With* an Impact Assessment,
**45.9%** (28/61) provided AI Notice. This represents an ~18x increase in the
likelihood of transparency when impact assessments are present.
3.  **Statistical Significance:** The Chi-square test yielded a statistic of
274.5 and a p-value of 1.19e-61, decisively rejecting the null hypothesis of
independence.
4.  **Strength of Association:** The Phi coefficient of 0.40 indicates a
moderate-to-strong positive association, confirming that these governance
controls are highly correlated.

**Conclusion:**
The hypothesis is **supported**. Governance controls in federal AI projects
appear to be implemented in 'clusters' rather than incrementally. Agencies that
invest in deep governance (Impact Assessments) are significantly more likely to
also implement basic transparency (AI Notice), while the vast majority of
systems lack both.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Robust Data Loading ---
filename = 'astalabs_discovery_all_data.csv'
possible_paths = [filename, f'../{filename}']
ds_path = None

for path in possible_paths:
    if os.path.exists(path):
        ds_path = path
        break

if ds_path is None:
    raise FileNotFoundError(f"{filename} not found in current or parent directory.")

print(f"Loading dataset from {ds_path}...")
df = pd.read_csv(ds_path, low_memory=False)

# --- 2. Filter Data ---
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 subset shape: {eo_data.shape}")

# --- 3. Binarization ---
col_impact = '52_impact_assessment'
col_notice = '59_ai_notice'

# Logic for Impact Assessment (Control 52)
# We treat 'Yes' as 1. 'Planned' is treated as 0 (not yet compliant).
def binarize_impact(val):
    if pd.isna(val):
        return 0
    s_val = str(val).lower().strip()
    if s_val in ['yes', 'true', '1', '1.0']:
        return 1
    return 0

# Logic for AI Notice (Control 59)
# Based on observed values, we filter out explicit non-compliance/exclusions.
def binarize_notice(val):
    if pd.isna(val):
        return 0
    s_val = str(val).strip()
    
    # explicit negatives
    negatives = [
        'None of the above',
        'N/A - individuals are not interacting with the AI for this use case',
        'Agency CAIO has waived this minimum practice and reported such waiver to OMB.',
        'AI is not safety or rights-impacting.'
    ]
    if s_val in negatives:
        return 0
    
    # If it's not a negative and is a non-empty string, it's a form of notice (Online, Email, etc.)
    if len(s_val) > 0:
        return 1
    return 0

eo_data['impact_binary'] = eo_data[col_impact].apply(binarize_impact)
eo_data['notice_binary'] = eo_data[col_notice].apply(binarize_notice)

print(f"\nBinary counts for Impact Assessment (1=Yes):\n{eo_data['impact_binary'].value_counts()}")
print(f"Binary counts for AI Notice (1=Provided):\n{eo_data['notice_binary'].value_counts()}")

# --- 4. Create Contingency Table ---
contingency_table = pd.crosstab(eo_data['impact_binary'], eo_data['notice_binary'])
# Ensure 2x2
contingency_table = contingency_table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

# Display Table
display_table = contingency_table.copy()
display_table.index = ['No Impact Assessment', 'Has Impact Assessment']
display_table.columns = ['No AI Notice', 'Has AI Notice']

print("\n--- Contingency Table ---")
print(display_table)

# --- 5. Statistical Analysis ---
# Check for degeneracy (if any row/col sums to 0)
row_sums = contingency_table.sum(axis=1)
col_sums = contingency_table.sum(axis=0)

if (row_sums == 0).any() or (col_sums == 0).any():
    print("\nWarning: Degenerate contingency table (one or more categories have 0 observations).")
    print("Cannot perform Chi-square test.")
else:
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Phi Coefficient
    n = contingency_table.sum().sum()
    phi = np.sqrt(chi2 / n)

    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    print(f"Phi Coefficient: {phi:.4f}")

    # --- 6. Visualization ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(display_table, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Co-occurrence: Impact Assessment vs. AI Notice')
    plt.ylabel('Impact Assessment (Control 52)')
    plt.xlabel('AI Notice (Control 59)')
    plt.tight_layout()
    plt.show()

    # --- 7. Interpretation ---
    alpha = 0.05
    print("\n--- Conclusion ---")
    if p < alpha:
        print("Result: Statistically Significant Dependency.")
        if phi > 0.5:
            strength = "Strong"
        elif phi > 0.3:
            strength = "Moderate"
        elif phi > 0.1:
            strength = "Weak"
        else:
            strength = "Negligible"
        print(f"There is a {strength} positive association (Phi={phi:.2f}) between the controls.")
        print("Hypothesis Supported: Agencies that perform Impact Assessments are significantly more likely to provide AI Notice.")
    else:
        print("Result: No Statistically Significant Dependency.")
        print("Hypothesis Rejected: Governance controls do not appear to cluster significantly in this dataset.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
EO 13960 subset shape: (1757, 196)

Binary counts for Impact Assessment (1=Yes):
impact_binary
0    1696
1      61
Name: count, dtype: int64
Binary counts for AI Notice (1=Provided):
notice_binary
0    1686
1      71
Name: count, dtype: int64

--- Contingency Table ---
                       No AI Notice  Has AI Notice
No Impact Assessment           1653             43
Has Impact Assessment            33             28

Chi-square Statistic: 274.4979
P-value: 1.1875e-61
Phi Coefficient: 0.3953

--- Conclusion ---
Result: Statistically Significant Dependency.
There is a Moderate positive association (Phi=0.40) between the controls.
Hypothesis Supported: Agencies that perform Impact Assessments are significantly more likely to provide AI Notice.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **Heatmap representing a Co-occurrence Matrix** (also known as a Contingency Table or Confusion Matrix).
*   **Purpose:** It visualizes the frequency distribution and relationship between two categorical variables: "Impact Assessment" and "AI Notice." The intensity of the color corresponds to the count (frequency) in each intersection.

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Impact Assessment (Control 52)"
    *   **Categories:** Two discrete categories: "No Impact Assessment" (top row) and "Has Impact Assessment" (bottom row).
*   **X-Axis:**
    *   **Label:** "AI Notice (Control 59)"
    *   **Categories:** Two discrete categories: "No AI Notice" (left column) and "Has AI Notice" (right column).
*   **Value Ranges:** The axes represent binary categories (No/Has), not numerical ranges. The values within the matrix range from a minimum of **28** to a maximum of **1653**.

### 3. Data Trends
*   **Dominant Group (High Value):** The vast majority of the data points fall into the top-left quadrant (No Impact Assessment / No AI Notice). This cell has a count of **1653** and is colored deep blue, indicating it is the most frequent occurrence by a significant margin.
*   **Minority Groups (Low Values):** The other three quadrants represent a very small fraction of the total data. They are colored very pale blue/white, indicating low frequency:
    *   No Impact Assessment / Has AI Notice: **43**
    *   Has Impact Assessment / No AI Notice: **33**
    *   Has Impact Assessment / Has AI Notice: **28**

### 4. Annotations and Legends
*   **Title:** "Co-occurrence: Impact Assessment vs. AI Notice" clearly defines the scope of the comparison.
*   **Cell Annotations:** Each cell contains the exact integer count of instances falling into that specific intersection (1653, 43, 33, 28).
*   **Color Coding:** While there is no explicit legend bar, the visual encoding uses a single-hue sequential color scale (likely Blues), where darker blue represents higher counts and white represents lower counts.

### 5. Statistical Insights
*   **Rarity of Controls:** Both "Impact Assessment" and "AI Notice" are rare attributes in this dataset. The "default" state for the vast majority of cases (approx. 94%) is the absence of both controls.
*   **Association:** Despite the rarity, there appears to be a conditional relationship between the two controls.
    *   If a case **Has Impact Assessment** (Total = $33 + 28 = 61$), there is a roughly **46%** chance ($28/61$) it also has an AI Notice.
    *   In contrast, if a case has **No Impact Assessment**, there is only a **2.5%** chance ($43/1696$) it has an AI Notice.
*   **Conclusion:** The presence of an "Impact Assessment" is strongly associated with a higher likelihood of having an "AI Notice," suggesting that these compliance or governance controls likely occur together within organizational processes.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
