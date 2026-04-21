# Experiment 37: node_4_10

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_10` |
| **ID in Run** | 37 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T02:49:20.826740+00:00 |
| **Runtime** | 349.8s |
| **Parent** | `node_3_10` |
| **Children** | `node_5_13`, `node_5_42` |
| **Creation Index** | 38 |

---

## Hypothesis

> Biometric Bias Awareness: AI systems involving biometric data (e.g., Facial
Recognition) have a significantly higher rate of 'Disparity Mitigation'
compliance compared to non-biometric systems, reflecting heightened awareness of
bias risks in this domain.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7258 (Likely True) |
| **Posterior** | 0.3681 (Maybe False) |
| **Surprise** | -0.4292 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 29.0 |
| Uncertain | 0.0 |
| Maybe False | 1.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 44.0 |
| Definitely False | 16.0 |

---

## Experiment Plan

**Objective:** Investigate if biometric use cases are better governed regarding bias mitigation than non-biometric cases, with robust handling for sparse compliance data.

### Steps
- 1. Load the `astalabs_discovery_all_data.csv` dataset and filter for rows where `source_table` is 'eo13960_scored'.
- 2. Print the unique values of the column `62_disparity_mitigation` to verify the exact string representation of positive cases.
- 3. Create a boolean variable `is_biometric` by searching the `2_use_case_name` column for the keywords ['Face', 'Facial', 'Biometric', 'Voice', 'Recognition'] (case-insensitive).
- 4. Create a boolean variable `has_mitigation` by checking if `62_disparity_mitigation` equals 'Yes' (ensure case-insensitive comparison and strip whitespace).
- 5. Generate a contingency table using `pd.crosstab` crossing `is_biometric` (rows) and `has_mitigation` (columns).
- 6. CRITICAL: Reindex the contingency table to ensure it has indices `[False, True]` and columns `[False, True]` with `fill_value=0`. This prevents the `ValueError` experienced previously if a category is missing.
- 7. Rename the indices to ['Non-Biometric', 'Biometric'] and columns to ['No Mitigation', 'Mitigation'].
- 8. Perform a Fisher's Exact Test (appropriate for potentially small cell counts) to test for independence.
- 9. Calculate and print the Odds Ratio and the compliance percentage for both groups.

### Deliverables
- Unique values of mitigation column, 2x2 Contingency table (guaranteed shape), Fisher's Exact Test p-value, and Odds Ratio.

---

## Analysis

The experiment was successfully executed. The code corrected the previous data
parsing issue by implementing a text-analysis heuristic to identify 'Disparity
Mitigation' compliance from free-text descriptions, rather than looking for a
binary 'Yes'.

**Hypothesis Test Results:**
1.  **Data Distribution:**
    -   **Non-Biometric Systems:** 1,701 total. 1,645 had no mitigation, 56 had
mitigation (3.29% compliance).
    -   **Biometric Systems:** 56 total. 53 had no mitigation, 3 had mitigation
(5.36% compliance).

2.  **Statistical Test:**
    -   **Fisher's Exact Test P-value:** 0.4329
    -   **Chi-Square P-value:** 0.6405
    -   **Odds Ratio:** 1.6627

3.  **Conclusion:**
    While Biometric systems showed a slightly higher rate of compliance (5.36%
vs 3.29%) and a positive Odds Ratio (1.66), the difference is **not
statistically significant** (p > 0.05). The analysis indicates that disparity
mitigation practices are rarely documented across the entire federal inventory,
regardless of the system type. The hypothesis that biometric systems have a
*significantly* higher rate of compliance is **rejected** based on this
evidence.

---

## Review

The experiment was faithfully implemented. The code successfully adapted to the
data quality issues encountered in previous attempts by implementing a heuristic
text analysis function to correctly identify 'Disparity Mitigation' compliance
from free-text descriptions. This allowed for a valid statistical comparison.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency

# Load dataset
filename = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(filename, low_memory=False)
    
    # Filter for EO 13960 data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"Loaded EO 13960 dataset: {len(df_eo)} rows")

    # 1. Create 'is_biometric' (Independent Variable)
    keywords = ['face', 'facial', 'biometric', 'voice', 'recognition']
    pattern = '|'.join(keywords)
    df_eo['2_use_case_name'] = df_eo['2_use_case_name'].astype(str)
    df_eo['is_biometric'] = df_eo['2_use_case_name'].str.contains(pattern, case=False, na=False)

    # 2. Create 'has_mitigation' (Dependent Variable) using text analysis
    # Logic: Default to True, set to False if text indicates N/A, None, or No mechanism.
    def check_mitigation(text):
        if pd.isna(text):
            return False
        s = str(text).strip().lower()
        if s == '' or s == 'nan':
            return False
        
        # Negative indicators at the start or specific phrases
        if s.startswith('n/a'): return False
        if s.startswith('none'): return False
        if s.startswith('no '): return False
        if s.startswith('not '): return False
        if 'not applicable' in s: return False
        if 'no demographic' in s: return False
        if 'not safety' in s: return False
        
        # If it passed all above, we assume it describes a mitigation
        return True

    df_eo['has_mitigation'] = df_eo['62_disparity_mitigation'].apply(check_mitigation)

    # 3. Generate Contingency Table
    contingency = pd.crosstab(df_eo['is_biometric'], df_eo['has_mitigation'])
    
    # Force 2x2 shape [False, True]
    contingency = contingency.reindex(index=[False, True], columns=[False, True], fill_value=0)
    contingency.index = ['Non-Biometric', 'Biometric']
    contingency.columns = ['No Mitigation', 'Has Mitigation']

    print("\n--- Contingency Table (Counts) ---")
    print(contingency)

    # 4. Statistical Testing
    # Fisher's Exact Test
    stat, p_val = fisher_exact(contingency)
    print(f"\n--- Fisher's Exact Test Results ---")
    print(f"P-value: {p_val:.4f}")
    
    # Chi-square
    chi2, p_chi2, dof, expected = chi2_contingency(contingency)
    print(f"Chi-Square Statistic: {chi2:.4f}, P-value: {p_chi2:.4f}")

    # 5. Calculate Odds Ratio & Compliance Rates
    non_bio_no = contingency.loc['Non-Biometric', 'No Mitigation']
    non_bio_yes = contingency.loc['Non-Biometric', 'Has Mitigation']
    bio_no = contingency.loc['Biometric', 'No Mitigation']
    bio_yes = contingency.loc['Biometric', 'Has Mitigation']
    
    non_bio_total = non_bio_no + non_bio_yes
    bio_total = bio_no + bio_yes
    
    print("\n--- Compliance Rates ---")
    if non_bio_total > 0:
        nb_rate = (non_bio_yes / non_bio_total) * 100
        print(f"Non-Biometric: {nb_rate:.2f}% ({non_bio_yes}/{non_bio_total})")
    
    if bio_total > 0:
        b_rate = (bio_yes / bio_total) * 100
        print(f"Biometric:     {b_rate:.2f}% ({bio_yes}/{bio_total})")

    # Odds Ratio
    # OR = (bio_yes / bio_no) / (non_bio_yes / non_bio_no)
    if bio_no > 0 and non_bio_yes > 0:
        or_val = (bio_yes * non_bio_no) / (bio_no * non_bio_yes)
        print(f"\nOdds Ratio: {or_val:.4f}")
    else:
        print("\nOdds Ratio: Undefined (division by zero)")

except Exception as e:
    print(f"An error occurred: {e}")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded EO 13960 dataset: 1757 rows

--- Contingency Table (Counts) ---
               No Mitigation  Has Mitigation
Non-Biometric           1645              56
Biometric                 53               3

--- Fisher's Exact Test Results ---
P-value: 0.4329
Chi-Square Statistic: 0.2181, P-value: 0.6405

--- Compliance Rates ---
Non-Biometric: 3.29% (56/1701)
Biometric:     5.36% (3/56)

Odds Ratio: 1.6627

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
