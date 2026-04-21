# Experiment 29: node_4_6

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_6` |
| **ID in Run** | 29 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T02:26:39.601221+00:00 |
| **Runtime** | 217.2s |
| **Parent** | `node_3_4` |
| **Children** | `node_5_18` |
| **Creation Index** | 30 |

---

## Hypothesis

> Commercial Opacity: Government AI systems procured from commercial vendors are
significantly less likely to have accessible code compared to systems developed
in-house, indicating a transparency trade-off in procurement.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.9945 (Definitely True) |
| **Surprise** | +0.0128 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 30.0 |
| Maybe True | 0.0 |
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

**Objective:** Assess the transparency gap between commercial and custom government AI systems.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `eo13960_scored`.
- 2. Clean `10_commercial_ai` to binary (Yes/No) and `38_code_access` to binary (Yes/No).
- 3. Create a contingency table of Commercial Status vs. Code Access.
- 4. Perform a Chi-square test (or Fisher's Exact Test if counts are low) to check for independence.
- 5. Calculate the odds ratio to quantify the likelihood of code opacity in commercial systems.

### Deliverables
- Contingency table, p-value, and odds ratio.

---

## Analysis

The experiment successfully tested the 'Commercial Opacity' hypothesis using the
EO 13960 dataset (n=950). By using `37_custom_code` as a proxy for procurement
type (where 'No' indicates COTS/Commercial and 'Yes' indicates Custom/In-house),
the analysis revealed a stark contrast in transparency.

1. **Descriptive Stats**: Only 2.5% (6/237) of Commercial systems had accessible
source code, compared to 12.6% (90/713) of Custom/In-house systems.
2. **Statistical Test**: The Chi-square test yielded a highly significant
p-value (1.42e-05), rejecting the null hypothesis of independence.
3. **Effect Size**: The Odds Ratio for opacity (lack of code access) was
approximately 5.56. This indicates that government AI systems procured
commercially are over 5 times more likely to lack source code access compared to
those developed with custom code.

The hypothesis is strongly supported: there is a significant transparency trade-
off in commercial AI procurement.

---

## Review

The experiment was successfully executed and the analysis is sound. The initial
issue with column selection was correctly identified and resolved by using
`37_custom_code` as a robust proxy for commercial vs. in-house development. The
statistical methods (Chi-square and Odds Ratio) were appropriate for the binary
categorical data, and the sample size (n=950) was sufficient to draw significant
conclusions.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
subset = df[df['source_table'] == 'eo13960_scored'].copy()

# Define column names
col_custom = '37_custom_code'
col_access = '38_code_access'

# Debug: Print unique values to define mapping logic
print(f"Unique values in '{col_custom}': {subset[col_custom].unique()}")
print(f"Unique values in '{col_access}': {subset[col_access].unique()}")

# Cleaning functions
def clean_commercial_proxy(val):
    # If 'custom code' is NO, we treat it as Commercial/COTS (Is Commercial = 1)
    # If 'custom code' is YES, we treat it as In-House/Custom (Is Commercial = 0)
    s = str(val).strip().lower()
    if 'no' in s:  # No custom code -> Commercial
        return 1
    elif 'yes' in s: # Yes custom code -> Non-Commercial
        return 0
    return np.nan

def clean_access(val):
    s = str(val).strip().lower()
    if 'no' in s: 
        return 0
    elif 'yes' in s:
        return 1
    return np.nan

# Apply cleaning
subset['is_commercial'] = subset[col_custom].apply(clean_commercial_proxy)
subset['has_code_access'] = subset[col_access].apply(clean_access)

# Drop NaNs for analysis
analysis_df = subset.dropna(subset=['is_commercial', 'has_code_access'])

print(f"\nData points after cleaning: {len(analysis_df)}")

if len(analysis_df) > 0:
    # Create Contingency Table
    # Rows: Commercial Status (0=Custom, 1=Commercial)
    # Cols: Code Access (0=No, 1=Yes)
    ct = pd.crosstab(analysis_df['is_commercial'], analysis_df['has_code_access'])
    print("\nContingency Table (Rows: Commercial [0=Custom, 1=Comm], Cols: Access [0=No, 1=Yes]):")
    print(ct)
    
    # Statistics
    chi2, p, dof, ex = stats.chi2_contingency(ct)
    
    # Calculate Odds Ratio manually for 2x2: (a*d)/(b*c)
    # Table layout:
    #           Access=0   Access=1
    # Comm=0 (a)         (b)
    # Comm=1 (c)         (d)
    # OR of Access given Commercial? 
    # Let's calculate odds of *No Access* (Opacity) for Commercial vs Custom.
    # Opacity = Access 0.
    # Odds(Opacity | Commercial) = Count(Comm=1, Acc=0) / Count(Comm=1, Acc=1)
    # Odds(Opacity | Custom)     = Count(Comm=0, Acc=0) / Count(Comm=0, Acc=1)
    # OR = Odds(Comm) / Odds(Custom)
    
    try:
        # c = Comm=1, Acc=0; d = Comm=1, Acc=1
        # a = Comm=0, Acc=0; b = Comm=0, Acc=1
        c = ct.loc[1, 0] if 0 in ct.columns and 1 in ct.index else 0
        d = ct.loc[1, 1] if 1 in ct.columns and 1 in ct.index else 0
        a = ct.loc[0, 0] if 0 in ct.columns and 0 in ct.index else 0
        b = ct.loc[0, 1] if 1 in ct.columns and 0 in ct.index else 0
        
        odds_commercial_opacity = c / d if d > 0 else np.inf
        odds_custom_opacity = a / b if b > 0 else np.inf
        or_opacity = odds_commercial_opacity / odds_custom_opacity
        
        print(f"\nChi-Square p-value: {p:.4e}")
        print(f"Odds Ratio (Likelihood of Opacity for Commercial vs Custom): {or_opacity:.4f}")
        
        if p < 0.05:
            print("Significant result.")
            if or_opacity > 1:
                print("Commercial systems are significantly more opaque (less code access).")
            else:
                print("Commercial systems are significantly less opaque.")
        else:
            print("No significant difference found.")
            
    except Exception as e:
        print(f"Error calculating odds ratio: {e}")
        
else:
    print("No valid data points.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Unique values in '37_custom_code': <StringArray>
['No', 'Yes', nan, ' ']
Length: 4, dtype: str
Unique values in '38_code_access': <StringArray>
[             'No – agency does not have access to source code.',
                                                             nan,
 'Yes – agency has access to source code, but it is not public.',
                      'Yes – source code is publicly available.',
                                                           'Yes',
                                                           'YES',
                                                             ' ']
Length: 7, dtype: str

Data points after cleaning: 950

Contingency Table (Rows: Commercial [0=Custom, 1=Comm], Cols: Access [0=No, 1=Yes]):
has_code_access  0.0  1.0
is_commercial            
0.0              623   90
1.0              231    6

Chi-Square p-value: 1.4187e-05
Odds Ratio (Likelihood of Opacity for Commercial vs Custom): 5.5618
Significant result.
Commercial systems are significantly more opaque (less code access).

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
