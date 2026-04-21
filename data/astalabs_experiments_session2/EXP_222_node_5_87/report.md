# Experiment 222: node_5_87

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_87` |
| **ID in Run** | 222 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T11:23:03.919684+00:00 |
| **Runtime** | 171.9s |
| **Parent** | `node_4_21` |
| **Children** | `node_6_57`, `node_6_66` |
| **Creation Index** | 223 |

---

## Hypothesis

> Adversarial Kill Chain: In the ATLAS dataset, case studies that involve
'Reconnaissance' tactics are statistically more likely to also involve 'Resource
Development' tactics compared to cases without Reconnaissance, reflecting a
structured sequential attack pattern.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.2610 (Likely False) |
| **Surprise** | -0.5868 |
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
| Maybe False | 2.0 |
| Definitely False | 58.0 |

---

## Experiment Plan

**Objective:** Test the co-occurrence of specific adversarial tactics (Reconnaissance and Resource Development) in ATLAS case studies.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `atlas_cases`.
- 2. Parse the `tactics` column (string of delimited values). Create binary flags: `has_reconnaissance` and `has_resource_dev`.
- 3. Generate a 2x2 contingency table for these two flags.
- 4. Perform Fisher's Exact Test (due to likely small sample size of ATLAS cases).
- 5. Calculate the Phi coefficient or Odds Ratio to measure the strength of the association.

### Deliverables
- Contingency table, Fisher's Exact Test p-value, and association metric (Phi/OR).

---

## Analysis

The experiment successfully analyzed the 52 ATLAS case studies present in the
dataset to test for a co-occurrence pattern between 'Reconnaissance' and
'Resource Development' tactics. The code correctly identified the 'tactics'
column and parsed the values. The analysis revealed that while 'Reconnaissance'
was documented in 20 of the 52 cases, 'Resource Development' was not found in
any of the cases (0 occurrences).

Due to the complete absence of the 'Resource Development' tactic in this subset
of the data, the contingency table contained a column of zeros, resulting in a
Fisher's Exact Test p-value of 1.0 and an undefined (NaN) Odds Ratio.
Consequently, the hypothesis regarding a sequential attack pattern involving
these two specific tactics could not be validated or statistically tested with
this specific dataset, as the necessary variance was missing.

---

## Review

The experiment was successfully executed and faithfully implemented the analysis
plan. The code correctly identified 52 ATLAS case studies and parsed the
'tactics' column.

**Hypothesis:** Adversarial Kill Chain (Cases with 'Reconnaissance' are more
likely to have 'Resource Development').

**Findings:**
1.  **Data Distribution:**
    - 'Reconnaissance' was present in 20 of 52 cases (38.5%).
    - 'Resource Development' was present in 0 of 52 cases (0%).
2.  **Statistical Test:** Due to the complete absence of the 'Resource
Development' tactic in this dataset, the contingency table contained a column of
zeros (`[[32, 0], [20, 0]]`).
    - Fisher's Exact Test p-value: 1.0.
    - Odds Ratio: Undefined (NaN).

**Conclusion:**
The hypothesis could not be validated. The analysis revealed that the specific
tactic 'Resource Development' is not coded in any of the available 52 ATLAS case
studies in this dataset, rendering a correlation test with 'Reconnaissance'
impossible (degenerate). The null hypothesis stands.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import sys
import os

# Define path handling for dataset
file_name = 'astalabs_discovery_all_data.csv'
possible_paths = [f'../{file_name}', file_name]
dataset_path = next((p for p in possible_paths if os.path.exists(p)), None)

if not dataset_path:
    print(f"Error: Dataset {file_name} not found.")
    sys.exit(1)

print(f"Loading dataset from {dataset_path}...")
try:
    df = pd.read_csv(dataset_path, low_memory=False)
except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

# Filter for ATLAS cases
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
print(f"ATLAS cases found: {len(atlas_df)}")

# Identify the 'tactics' column
# Metadata indicates column 92 is 'tactics', but sparse CSVs might have shifted names or indices.
tactics_col = 'tactics'
if tactics_col not in atlas_df.columns:
    # search for it
    candidates = [c for c in atlas_df.columns if 'tactics' in str(c).lower()]
    if candidates:
        tactics_col = candidates[0]
        print(f"Found tactics column: {tactics_col}")
    else:
        print("Error: 'tactics' column not found in dataset.")
        print("Available columns:", atlas_df.columns.tolist())
        sys.exit(1)

# Process tactics
# Tactics are likely comma-separated strings. We check for substring presence.
atlas_df[tactics_col] = atlas_df[tactics_col].fillna('').astype(str)

# Create binary flags
# Using case-insensitive match just in case, though metadata suggests capitalized Proper Nouns
atlas_df['has_recon'] = atlas_df[tactics_col].str.contains('Reconnaissance', case=False, regex=False)
atlas_df['has_res_dev'] = atlas_df[tactics_col].str.contains('Resource Development', case=False, regex=False)

# Generate Contingency Table
contingency_table = pd.crosstab(
    atlas_df['has_recon'], 
    atlas_df['has_res_dev'], 
    rownames=['Has Reconnaissance'], 
    colnames=['Has Resource Dev']
)

print("\n--- Contingency Table ---")
print(contingency_table)

# Ensure 2x2 shape for valid output interpretation, fill missing if necessary
# (crosstab might define fewer rows/cols if all are True or all are False)
# We manually construct the 2x2 array for the test to ensure alignment
n_recon_no_res = len(atlas_df[(atlas_df['has_recon'] == True) & (atlas_df['has_res_dev'] == False)])
n_recon_yes_res = len(atlas_df[(atlas_df['has_recon'] == True) & (atlas_df['has_res_dev'] == True)])
n_no_recon_no_res = len(atlas_df[(atlas_df['has_recon'] == False) & (atlas_df['has_res_dev'] == False)])
n_no_recon_yes_res = len(atlas_df[(atlas_df['has_recon'] == False) & (atlas_df['has_res_dev'] == True)])

obs = [[n_no_recon_no_res, n_no_recon_yes_res], [n_recon_no_res, n_recon_yes_res]]
print(f"\nFormatted for Fisher's Test ([[NoRecon/NoRes, NoRecon/YesRes], [Recon/NoRes, Recon/YesRes]]):\n{obs}")

# Perform Fisher's Exact Test
# H0: The presence of Reconnaissance is independent of the presence of Resource Development
odds_ratio, p_value = stats.fisher_exact(obs)

print("\n--- Statistical Test Results ---")
print(f"Fisher's Exact Test p-value: {p_value:.4f}")
print(f"Odds Ratio: {odds_ratio:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Result: Statistically Significant (Reject H0)")
    print("Interpretation: There is a significant association between Reconnaissance and Resource Development tactics.")
else:
    print("Result: Not Statistically Significant (Fail to Reject H0)")
    print("Interpretation: No significant association found between these tactics in this dataset.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
ATLAS cases found: 52

--- Contingency Table ---
Has Resource Dev    False
Has Reconnaissance       
False                  32
True                   20

Formatted for Fisher's Test ([[NoRecon/NoRes, NoRecon/YesRes], [Recon/NoRes, Recon/YesRes]]):
[[32, 0], [20, 0]]

--- Statistical Test Results ---
Fisher's Exact Test p-value: 1.0000
Odds Ratio: nan
Result: Not Statistically Significant (Fail to Reject H0)
Interpretation: No significant association found between these tactics in this dataset.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
