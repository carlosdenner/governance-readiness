# Experiment 188: node_5_73

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_73` |
| **ID in Run** | 188 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T09:52:30.709973+00:00 |
| **Runtime** | 307.2s |
| **Parent** | `node_4_33` |
| **Children** | None |
| **Creation Index** | 189 |

---

## Hypothesis

> The Privacy Shield Paradox: AI systems processing Personally Identifiable
Information (PII) are significantly more likely to have undergone a general
Impact Assessment than non-PII systems, suggesting that established privacy
regulations (e.g., Privacy Act) act as the primary driver/gateway for broader AI
governance compliance in the federal government.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9194 (Definitely True) |
| **Posterior** | 0.9725 (Definitely True) |
| **Surprise** | +0.0638 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 22.0 |
| Maybe True | 8.0 |
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

**Objective:** Determine if the presence of PII is a predictor for the completion of Impact Assessments.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `source_table` = 'eo13960_scored'.
- 2. Create a binary variable `has_pii` from column `29_contains_pii` (1 if 'Yes', 0 otherwise).
- 3. Create a binary variable `has_impact_assessment` from column `52_impact_assessment` (1 if 'Yes', 0 otherwise).
- 4. Generate a contingency table for `has_pii` vs `has_impact_assessment`.
- 5. Perform a Chi-square test of independence and calculate the Odds Ratio.
- 6. Interpret if PII systems are statistically more compliant with impact assessment requirements.

### Deliverables
- Contingency table, Chi-square statistic, p-value, and Odds Ratio comparing impact assessment rates between PII and non-PII systems.

---

## Analysis

The experiment successfully tested the 'Privacy Shield Paradox' hypothesis using
the EO 13960 dataset (n=1,757). The analysis examined whether the presence of
Personally Identifiable Information (PII) acts as a predictor for the completion
of Impact Assessments.

**Findings:**
1.  **Low Baseline Compliance:** Impact Assessments are rare across the entire
inventory, present in only 3.5% (61/1,757) of all systems.
2.  **Significant Disparity:** Despite the low baseline, a strong distinction
exists based on data sensitivity:
    *   **PII Systems:** 10.5% (18/171) have completed an impact assessment.
    *   **Non-PII Systems:** Only 2.7% (43/1,586) have completed one.
3.  **Statistical Strength:** The Chi-square test confirms this relationship is
highly significant ($X^2$ = 25.85, p < 0.001).
4.  **Magnitude of Effect:** The Odds Ratio of 4.22 indicates that systems
handling PII are more than four times as likely to undergo an impact assessment
compared to those that do not.

**Conclusion:**
The hypothesis is **supported**. The data suggests that existing privacy
regulations (likely the Privacy Act of 1974 and E-Government Act of 2002) act as
a 'shield' or primary gateway for AI governance. Systems that trigger these
established legal frameworks receive significantly more scrutiny, while non-PII
systems—regardless of their potential for other types of harm—are largely left
unassessed.

---

## Review

The experiment was successfully executed and faithfully implemented the analysis
plan. After resolving initial file path and data type errors (specifically
handling NaN/float values in text columns), the code correctly filtered the EO
13960 dataset and generated the required binary variables. The statistical
analysis (Chi-square test and Odds Ratio) was performed correctly on the 2x2
contingency table. The results are statistically significant and directly
support the hypothesis, providing clear quantitative evidence for the 'Privacy
Shield Paradox'.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

def run_experiment():
    print("Starting Privacy Shield Paradox Experiment...\n")
    
    # 1. Load dataset
    filename = 'astalabs_discovery_all_data.csv'
    if os.path.exists(filename):
        filepath = filename
    elif os.path.exists(f'../{filename}'):
        filepath = f'../{filename}'
    else:
        print(f"Error: {filename} not found.")
        return

    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # 2. Filter for 'eo13960_scored'
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 Subset Shape: {df_eo.shape}")

    if df_eo.empty:
        print("No data in EO 13960 subset.")
        return

    # Helper function for robust string conversion
    def robust_contains_yes(val):
        s = str(val).lower().strip()
        # Check if it starts with yes or is just yes (handling potential verbose answers)
        # The prompt says: 1 if 'Yes', 0 otherwise. 
        # Usually these fields contain "Yes", "No", "Not Applicable" or verbose explanations starting with Yes.
        # We will check if "yes" is in the string to be inclusive, or starts with it.
        if 'yes' in s:
            return 1
        return 0

    # 3. Create binary variable 'has_pii'
    if '29_contains_pii' not in df_eo.columns:
        print("Error: Column '29_contains_pii' not found.")
        return
    
    df_eo['has_pii'] = df_eo['29_contains_pii'].apply(robust_contains_yes)

    # 4. Create binary variable 'has_impact_assessment'
    if '52_impact_assessment' not in df_eo.columns:
        print("Error: Column '52_impact_assessment' not found.")
        return

    df_eo['has_impact_assessment'] = df_eo['52_impact_assessment'].apply(robust_contains_yes)

    # 5. Generate Contingency Table
    # Rows: Has PII (0, 1)
    # Cols: Has Impact Assessment (0, 1)
    contingency_table = pd.crosstab(
        df_eo['has_pii'], 
        df_eo['has_impact_assessment'], 
        rownames=['Has PII'], 
        colnames=['Has Impact Assessment']
    )

    print("\n--- Contingency Table ---")
    print(contingency_table)
    
    # Ensure table is 2x2 for consistent OR calc
    # Reindex to ensure all keys exist
    contingency_table = contingency_table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

    # 6. Perform Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Calculate Odds Ratio
    # OR = (a*d) / (b*c)
    # a = PII=0, IA=0
    # b = PII=0, IA=1
    # c = PII=1, IA=0
    # d = PII=1, IA=1
    
    a = contingency_table.loc[0, 0]
    b = contingency_table.loc[0, 1]
    c = contingency_table.loc[1, 0]
    d = contingency_table.loc[1, 1]
    
    if b * c > 0:
        odds_ratio = (a * d) / (b * c)
    else:
        odds_ratio = float('inf')

    # 7. Print Results
    print("\n--- Statistical Results ---")
    print(f"Chi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    
    # Interpretation
    print("\n--- Interpretation ---")
    if p < 0.05:
        print("Result: Statistically Significant")
        if odds_ratio > 1:
            print("Conclusion: Systems handling PII are MORE likely to have an Impact Assessment.")
        else:
            print("Conclusion: Systems handling PII are LESS likely to have an Impact Assessment.")
    else:
        print("Result: Not Statistically Significant")
        print("Conclusion: No significant association between PII and Impact Assessments.")

    # Descriptive Stats
    total_n = len(df_eo)
    pii_n = df_eo['has_pii'].sum()
    ia_n = df_eo['has_impact_assessment'].sum()
    
    print(f"\nTotal Analyzed: {total_n}")
    print(f"Systems with PII: {pii_n} ({pii_n/total_n:.1%})")
    print(f"Systems with Impact Assessment: {ia_n} ({ia_n/total_n:.1%})")
    
    if pii_n > 0:
        ia_rate_pii = d / pii_n
        print(f"Impact Assessment Rate (Given PII): {ia_rate_pii:.1%}")
    if (total_n - pii_n) > 0:
        ia_rate_no_pii = b / (total_n - pii_n)
        print(f"Impact Assessment Rate (Given No PII): {ia_rate_no_pii:.1%}")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting Privacy Shield Paradox Experiment...

EO 13960 Subset Shape: (1757, 196)

--- Contingency Table ---
Has Impact Assessment     0   1
Has PII                        
0                      1543  43
1                       153  18

--- Statistical Results ---
Chi-square Statistic: 25.8473
P-value: 3.6953e-07
Odds Ratio: 4.2216

--- Interpretation ---
Result: Statistically Significant
Conclusion: Systems handling PII are MORE likely to have an Impact Assessment.

Total Analyzed: 1757
Systems with PII: 171 (9.7%)
Systems with Impact Assessment: 61 (3.5%)
Impact Assessment Rate (Given PII): 10.5%
Impact Assessment Rate (Given No PII): 2.7%

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
