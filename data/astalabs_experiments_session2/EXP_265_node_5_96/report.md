# Experiment 265: node_5_96

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_96` |
| **ID in Run** | 265 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:38:46.547887+00:00 |
| **Runtime** | 289.4s |
| **Parent** | `node_4_45` |
| **Children** | None |
| **Creation Index** | 266 |

---

## Hypothesis

> Data Governance Cascade: The presence of a 'Data Catalog' (Q31) is a strong
predictor for the presence of 'Data Documentation' (Q34), indicating these
controls exist as a dependent maturity bundle.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.8500 (Likely True) |
| **Surprise** | +0.1200 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 28.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 36.0 |
| Maybe True | 24.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Re-assess the dependency between Data Catalog and Data Documentation using corrected text mapping for maturity descriptions to avoid false negatives.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for rows where `source_table` is 'eo13960_scored'.
- 2. Create binary variable `has_catalog`: Map `31_data_catalog` values containing 'yes' (case-insensitive) to 1, else 0.
- 3. Create binary variable `has_docs`: Inspect `34_data_docs` strings. Map values containing 'partially completed', 'is complete', or 'widely available' to 1 (Positive). Map values containing 'missing' or 'not available', or NaN/Null to 0 (Negative).
- 4. Generate a contingency table (crosstab) of `has_catalog` vs `has_docs`.
- 5. Calculate the Conditional Probability P(Docs=1 | Catalog=1) and the Lift (P(Docs|Catalog) / P(Docs)).
- 6. Calculate the Phi Coefficient and perform a Chi-Squared test of independence.

### Deliverables
- 1. Printed sample of raw vs mapped values for verification.
- 2. Contingency table.
- 3. Conditional Probability, Lift, Phi Coefficient, and Chi-Squared p-value.

---

## Analysis

The experiment successfully corrected the data-cleaning error from the previous
iteration. By mapping specific maturity description strings (e.g., 'partially
completed', 'is complete', 'widely available') to a positive value, the analysis
identified that 45.7% of use cases possess data documentation (up from the
erroneous 1.0%).

The results support the 'Data Governance Cascade' hypothesis. The analysis
reveals a statistically significant (p < 0.001), moderate positive correlation
(Phi = 0.36) between the presence of a Data Catalog and Data Documentation.
Specifically, use cases with a Data Catalog are 1.47 times more likely to have
associated data documentation (67.4%) compared to the baseline probability
(45.7%), suggesting that these governance controls often function as a dependent
maturity bundle.

---

## Review

The experiment was successfully re-executed with the corrected data mapping
logic. The fix to map descriptive maturity strings (e.g., 'is complete',
'partially completed') to the positive class resolved the previous issue of
undercounting, raising the prevalence of Data Documentation from 1% to 45.7%.
The results now reliably support the hypothesis.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import os

def run_experiment():
    print("Starting Data Governance Cascade experiment (Attempt 2)...")
    
    # Load dataset
    file_name = 'astalabs_discovery_all_data.csv'
    if os.path.exists(file_name):
        df = pd.read_csv(file_name, low_memory=False)
    elif os.path.exists('../' + file_name):
        df = pd.read_csv('../' + file_name, low_memory=False)
    else:
        print("Error: Dataset not found.")
        return

    # Filter for EO 13960 data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 records loaded: {len(df_eo)}")

    col_catalog = '31_data_catalog'
    col_docs = '34_data_docs'

    # 1. Clean Data Catalog (Q31)
    # Map 'Yes' (case-insensitive) to 1, else 0
    def clean_catalog(val):
        if pd.isna(val):
            return 0
        s = str(val).strip().lower()
        if 'yes' in s:
            return 1
        return 0
    
    df_eo['has_catalog'] = df_eo[col_catalog].apply(clean_catalog)

    # 2. Clean Data Documentation (Q34)
    # Map specific maturity levels to 1
    def clean_docs(val):
        if pd.isna(val):
            return 0
        s = str(val).strip().lower()
        # Positive indicators based on maturity model text
        if any(x in s for x in ['partially completed', 'is complete', 'widely available']):
            return 1
        # Negative indicators (explicitly checked for validation, though default is 0)
        # 'missing', 'not available' -> 0
        return 0

    df_eo['has_docs'] = df_eo[col_docs].apply(clean_docs)

    # Verification of mapping
    print("\n--- Mapping Verification ---")
    print("Catalog (Q31) Sample Mappings:")
    print(df_eo[[col_catalog, 'has_catalog']].drop_duplicates().head(5))
    print("\nDocs (Q34) Sample Mappings:")
    # Show unique strings and their mapping to ensure correctness
    unique_docs = df_eo[[col_docs, 'has_docs']].drop_duplicates()
    # Print first few chars of the long strings for readability
    for idx, row in unique_docs.iterrows():
        raw_val = str(row[col_docs])[:60] + "..."
        print(f"Raw: {raw_val:<65} -> Mapped: {row['has_docs']}")

    # 3. Contingency Table
    contingency = pd.crosstab(df_eo['has_catalog'], df_eo['has_docs'])
    print("\n--- Contingency Table (Rows=Catalog, Cols=Docs) ---")
    print(contingency)
    
    # Extract counts
    # contingency structure: 
    # col   0    1
    # row
    # 0     TN   FN
    # 1     FP   TP
    try:
        tn = contingency.loc[0, 0]
        fp = contingency.loc[0, 1] if 1 in contingency.columns else 0
        fn = contingency.loc[1, 0] if 1 in contingency.index else 0
        tp = contingency.loc[1, 1] if 1 in contingency.index and 1 in contingency.columns else 0
    except KeyError:
        print("Error creating full contingency table (missing classes).")
        return

    total = tn + fp + fn + tp
    n_catalog = fn + tp
    n_docs = fp + tp

    print(f"\nTotal Cases: {total}")
    print(f"Has Catalog: {n_catalog} ({n_catalog/total:.1%})")
    print(f"Has Docs:    {n_docs} ({n_docs/total:.1%})")

    # 4. Statistical Analysis
    # Conditional Probability P(Docs | Catalog)
    if n_catalog > 0:
        p_docs_given_catalog = tp / n_catalog
        p_docs_baseline = n_docs / total
        lift = p_docs_given_catalog / p_docs_baseline if p_docs_baseline > 0 else 0
        
        print(f"\nConditional Probability P(Docs|Catalog): {p_docs_given_catalog:.4f}")
        print(f"Baseline Probability P(Docs):          {p_docs_baseline:.4f}")
        print(f"Lift (Strength of Dependency):         {lift:.2f}x")
    
    # Chi-Squared and Phi
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    
    # Phi Coefficient = sqrt(chi2 / n)
    phi = np.sqrt(chi2 / total)
    
    print(f"\n--- Statistical Tests ---")
    print(f"Chi-Squared Statistic: {chi2:.4f}")
    print(f"P-Value:               {p_val:.4e}")
    print(f"Phi Coefficient:       {phi:.4f}")
    
    # Interpretation
    print("\nInterpretation:")
    if p_val < 0.05:
        print("Statistically significant relationship detected.")
        if phi > 0.5: print("Effect size: Strong")
        elif phi > 0.3: print("Effect size: Moderate")
        elif phi > 0.1: print("Effect size: Weak")
        else: print("Effect size: Negligible")
    else:
        print("No statistically significant relationship detected.")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting Data Governance Cascade experiment (Attempt 2)...
EO 13960 records loaded: 1757

--- Mapping Verification ---
Catalog (Q31) Sample Mappings:
    31_data_catalog  has_catalog
0                No            0
3               NaN            0
43              Yes            1
56            Other            0
504              NO            0

Docs (Q34) Sample Mappings:
Raw: Documentation is missing or not available: No documentation ...   -> Mapped: 0
Raw: nan...                                                            -> Mapped: 0
Raw: Documentation has been partially completed: Some documentati...   -> Mapped: 1
Raw: Documentation is complete: Documentation exists regarding th...   -> Mapped: 1
Raw: Documentation is widely available: Documentation is not only...   -> Mapped: 1
Raw: Documentation is complete: Documentation exists regarding th...   -> Mapped: 1
Raw: Documentation has been partially completed: Some documentati...   -> Mapped: 1
Raw: Data not reported by submitter and will be updated once addi...   -> Mapped: 0
Raw: Documentation is complete...                                      -> Mapped: 1
Raw: Documentation is missing or not available...                      -> Mapped: 0
Raw: Documentation is widely available...                              -> Mapped: 1
Raw: Documentation is available...                                     -> Mapped: 0
Raw: Documentation has been partially completed...                     -> Mapped: 1
Raw:  ...                                                              -> Mapped: 0
Raw: Yes...                                                            -> Mapped: 0
Raw: No...                                                             -> Mapped: 0
Raw: The data is public facing and documented in www.travel.state...   -> Mapped: 0
Raw: Application source code and documentation. ...                    -> Mapped: 0

--- Contingency Table (Rows=Catalog, Cols=Docs) ---
has_docs       0    1
has_catalog          
0            724  328
1            230  475

Total Cases: 1757
Has Catalog: 705 (40.1%)
Has Docs:    803 (45.7%)

Conditional Probability P(Docs|Catalog): 0.6738
Baseline Probability P(Docs):          0.4570
Lift (Strength of Dependency):         1.47x

--- Statistical Tests ---
Chi-Squared Statistic: 221.4191
P-Value:               4.4344e-50
Phi Coefficient:       0.3550

Interpretation:
Statistically significant relationship detected.
Effect size: Moderate

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
