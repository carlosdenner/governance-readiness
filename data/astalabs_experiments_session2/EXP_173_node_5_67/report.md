# Experiment 173: node_5_67

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_67` |
| **ID in Run** | 173 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T09:12:48.989402+00:00 |
| **Runtime** | 491.7s |
| **Parent** | `node_4_45` |
| **Children** | `node_6_68` |
| **Creation Index** | 174 |

---

## Hypothesis

> The Fairness-Safety Divergence: 'Fairness/Bias' technical failures are
significantly more likely to result in 'Intangible' harms (e.g., Civil Rights,
Reputation) while 'Safety/Robustness' failures are more likely to result in
'Tangible' harms (e.g., Physical, Financial).

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9274 (Definitely True) |
| **Posterior** | 0.9753 (Definitely True) |
| **Surprise** | +0.0574 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 23.0 |
| Maybe True | 7.0 |
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

**Objective:** Map technical failure modes to real-world harm domains and perform a Chi-square test to verify the association.

### Steps
- 1. Load the dataset 'astalabs_discovery_all_data.csv' (handle potential file path variations).
- 2. Filter for rows where `source_table` is 'aiid_incidents'.
- 3. Define a robust function to categorize `Known AI Technical Failure` into 'Bias/Fairness' (keywords: 'bias', 'fair', 'discrimination', 'equity') and 'Safety/Robustness' (keywords: 'mistake', 'error', 'crash', 'safety', 'robust', 'reliab'). Ensure the function handles non-string (NaN/float) inputs gracefully.
- 4. Define a robust function to categorize Harm columns (concatenate `Harm Domain`, `Tangible Harm`, `Special Interest Intangible Harm`) into 'Intangible' (keywords: 'civil rights', 'reputation', 'psychological', 'societal', 'privacy') and 'Tangible' (keywords: 'physical', 'financial', 'property', 'economic', 'life', 'health').
- 5. Apply these functions to create `Failure_Category` and `Harm_Category` columns.
- 6. Drop rows where either category is missing.
- 7. Create a contingency table (crosstab) of Failure vs. Harm categories.
- 8. Perform a Chi-square test of independence using `scipy.stats.chi2_contingency`.
- 9. Generate a heatmap of the contingency table.

### Deliverables
- 1. Contingency table printed to stdout.
- 2. Chi-square statistic and p-value printed.
- 3. Heatmap visualization saved or displayed.

---

## Analysis

The experiment successfully tested the 'Fairness-Safety Divergence' hypothesis
using 62 coded AI incidents. The Chi-square test (Statistic=12.97, p=0.0003)
confirmed a statistically significant association between Technical Failure
Modes and Harm Domains.

Key findings:
1. **Bias/Fairness** failures (n=20) were exclusively associated with
**Intangible** harms (Civil Rights, Reputation), with 0 incidents resulting in
Tangible harm.
2. **Safety/Robustness** failures (n=42) were evenly split between Intangible
(21) and Tangible (21) harms.

The hypothesis is **supported**: Bias failures are strongly predictive of
Intangible harms, while Safety failures are the sole driver of Tangible harms in
this sample, although Safety failures also frequently cause Intangible harms.

---

## Review

The experiment was faithfully implemented and the hypothesis was successfully
tested. The previous data schema issues were resolved by correctly mapping the
specific value indicators (e.g., 'tangible harm definitively occurred', 'yes')
rather than searching for generic keywords in boolean-like columns. The
statistical analysis on 62 categorized incidents yielded a Chi-square statistic
of 12.97 and a p-value of 0.0003, confirming a significant association. The
results strongly support the hypothesis: 'Bias/Fairness' failures were
exclusively associated with 'Intangible' harms (20 incidents vs. 0 Tangible),
while 'Safety/Robustness' failures were the sole driver of 'Tangible' harms in
this sample (21 incidents), although they also frequently caused 'Intangible'
harms (21 incidents).

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import sys
import os

# 1. Load Data
filename = 'astalabs_discovery_all_data.csv'
possible_paths = [filename, '../' + filename]
filepath = None
for p in possible_paths:
    if os.path.exists(p):
        filepath = p
        break

if not filepath:
    print("Dataset not found.")
    sys.exit(1)

print(f"Loading dataset from {filepath}...")
df = pd.read_csv(filepath, low_memory=False)

# 2. Filter for AIID Incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID Incidents: {len(aiid)}")

# 3. Map Failure Category
def map_failure(val):
    if pd.isna(val):
        return None
    text = str(val).lower()
    
    # Bias / Fairness Keywords
    if any(k in text for k in ['bias', 'fair', 'discriminat', 'equit', 'civil rights', 'demographic']):
        return 'Bias/Fairness'
    
    # Safety / Robustness Keywords
    if any(k in text for k in ['mistake', 'error', 'crash', 'safety', 'robust', 'generalization', 'context', 'hazard', 'unsafe', 'performance', 'failure']):
        return 'Safety/Robustness'
    
    return None

aiid['Failure_Category'] = aiid['Known AI Technical Failure'].apply(map_failure)

# 4. Map Harm Category
# Logic: Tangible = 'definitively occurred', Intangible = 'yes' in Special Interest
def map_harm(row):
    tangible_val = str(row.get('Tangible Harm', '')).lower()
    is_tangible = 'definitively occurred' in tangible_val
    
    intangible_val = str(row.get('Special Interest Intangible Harm', '')).lower()
    is_intangible = 'yes' in intangible_val
    
    # Classify
    if is_tangible and not is_intangible:
        return 'Tangible'
    elif is_intangible and not is_tangible:
        return 'Intangible'
    elif is_tangible and is_intangible:
        return 'Both' # Excluded from Chi-square to ensure mutual exclusivity
    else:
        return None

aiid['Harm_Category'] = aiid.apply(map_harm, axis=1)

# 5. Filter for Analysis
# We exclude 'Both' to strictly test the divergence between Tangible and Intangible outcomes
analysis_df = aiid.dropna(subset=['Failure_Category', 'Harm_Category'])
analysis_df = analysis_df[analysis_df['Harm_Category'] != 'Both']

print(f"Rows for analysis (Exclusive categories): {len(analysis_df)}")
print("\nCounts by Failure Category:")
print(analysis_df['Failure_Category'].value_counts())
print("\nCounts by Harm Category:")
print(analysis_df['Harm_Category'].value_counts())

# 6. Statistical Test
if len(analysis_df) < 5:
    print("\nInsufficient data for Chi-square test.")
else:
    contingency = pd.crosstab(analysis_df['Failure_Category'], analysis_df['Harm_Category'])
    print("\nContingency Table:")
    print(contingency)
    
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    if p < 0.05:
        print("Result: Significant association found (Reject Null Hypothesis).")
    else:
        print("Result: No significant association found (Fail to reject Null Hypothesis).")
    
    # 7. Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues')
    plt.title('Technical Failure Mode vs Harm Domain')
    plt.ylabel('Technical Failure')
    plt.xlabel('Harm Domain (Exclusive)')
    plt.tight_layout()
    plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
Total AIID Incidents: 1362
Rows for analysis (Exclusive categories): 62

Counts by Failure Category:
Failure_Category
Safety/Robustness    42
Bias/Fairness        20
Name: count, dtype: int64

Counts by Harm Category:
Harm_Category
Intangible    41
Tangible      21
Name: count, dtype: int64

Contingency Table:
Harm_Category      Intangible  Tangible
Failure_Category                       
Bias/Fairness              20         0
Safety/Robustness          21        21

Chi-Square Statistic: 12.9720
P-value: 3.1618e-04
Result: Significant association found (Reject Null Hypothesis).


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is a detailed analysis:

### 1. Plot Type
*   **Type:** Heatmap (specifically representing a 2x2 contingency table).
*   **Purpose:** The plot visualizes the frequency of intersections between two categorical variables: "Technical Failure" modes and "Harm Domain" categories. The color intensity corresponds to the count of occurrences in each category intersection.

### 2. Axes
*   **Y-Axis (Vertical):**
    *   **Label:** "Technical Failure"
    *   **Categories:** "Bias/Fairness" (top) and "Safety/Robustness" (bottom).
*   **X-Axis (Horizontal):**
    *   **Label:** "Harm Domain (Exclusive)"
    *   **Categories:** "Intangible" (left) and "Tangible" (right).
*   **Color Scale (Legend):**
    *   A vertical bar on the right indicates the magnitude of the values.
    *   **Range:** 0 to approximately 21.
    *   **Gradient:** Light blue/white represents low values (0), darkening to deep blue for high values (20+).

### 3. Data Trends
*   **High Values (Dark Blue):**
    *   The intersection of **Safety/Robustness** and **Intangible** has a count of **21**.
    *   The intersection of **Safety/Robustness** and **Tangible** also has a count of **21**.
    *   The intersection of **Bias/Fairness** and **Intangible** is high as well, with a count of **20**.
*   **Low Values (White):**
    *   The intersection of **Bias/Fairness** and **Tangible** is **0**, indicating no occurrences in this category.
*   **Row Patterns:** The "Safety/Robustness" row shows an even distribution across both harm domains (21 vs. 21). The "Bias/Fairness" row is heavily skewed entirely toward "Intangible" harm.

### 4. Annotations and Legends
*   **Title:** "Technical Failure Mode vs Harm Domain" prominently displayed at the top.
*   **Cell Annotations:** The exact integer counts (20, 0, 21, 21) are written inside each cell for precise reading.
*   **Color Bar:** Located to the right of the heatmap to provide a visual reference for the data magnitude relative to the color intensity.

### 5. Statistical Insights
*   **Exclusivity of Bias/Fairness:** The most significant insight is that in this dataset, **Bias/Fairness** failures result exclusively in **Intangible** harm (20 counts). There are zero recorded instances of Bias/Fairness leading to Tangible harm.
*   **Even Split for Safety/Robustness:** **Safety/Robustness** failures are equally likely to result in Intangible or Tangible harm (21 counts each).
*   **Dominant Failure Mode:** **Safety/Robustness** is the more frequent technical failure mode overall, accounting for 42 total incidents ($21 + 21$), compared to 20 total incidents for Bias/Fairness.
*   **Dominant Harm Domain:** **Intangible** harm is significantly more common than Tangible harm. Intangible harm accounts for approximately 66% of the data points ($20 + 21 = 41$), while Tangible harm accounts for roughly 34% ($0 + 21 = 21$).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
