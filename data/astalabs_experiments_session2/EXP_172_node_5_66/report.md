# Experiment 172: node_5_66

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_66` |
| **ID in Run** | 172 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T09:08:12.541823+00:00 |
| **Runtime** | 274.0s |
| **Parent** | `node_4_49` |
| **Children** | None |
| **Creation Index** | 173 |

---

## Hypothesis

> The 'Risk-Based' Oversight Model: Systems classified as 'High Impact' (or having
significant benefits/risks) are significantly more likely to undergo 'SAOP
Review' (Senior Agency Official for Privacy) than lower-impact systems,
indicating a functional risk-based governance approach.

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

**Objective:** Verify if higher risk/impact leads to higher administrative oversight.

### Steps
- 1. Filter 'eo13960_scored'.
- 2. Clean '17_impact_type' (or derived impact variable) into High vs Low/Medium.
- 3. Clean '30_saop_review' into Yes/No.
- 4. Create a cross-tabulation.
- 5. Perform a Chi-square test to verify if High Impact implies higher probability of Review.

### Deliverables
- Contingency table; Chi-square statistic; Probability of Review given High Impact vs Low Impact.

---

## Analysis

The experiment successfully validated the 'Risk-Based' Oversight hypothesis.
After correcting the data mapping logic to correctly identify 'Rights-
Impacting', 'Safety-Impacting', and 'Both' as 'High Impact', the analysis
proceeded with valid sample sizes (High Impact n=227, Low Impact n=1491).

Key Findings:
1. **Probabilities**: High-impact systems were more than twice as likely to
undergo Senior Agency Official for Privacy (SAOP) review (25.11%) compared to
low-impact systems (11.94%).
2. **Statistical Significance**: The Chi-square test yielded a statistic of
27.84 with a p-value of 1.32e-07 (p < 0.001), strongly rejecting the null
hypothesis.
3. **Effect Size**: The Phi coefficient of 0.1273 indicates a weak but positive
association.

While the hypothesis is supported statistically—riskier systems do attract more
oversight—the absolute compliance rate is notable: nearly 75% of identified
high-impact systems did *not* undergo SAOP review, suggesting that while the
risk-based mechanism exists, its enforcement coverage is limited.

---

## Review

The experiment was faithfully executed. After initial data exploration revealed
that 'High Impact' was not an explicit label but rather a category composed of
'Rights-Impacting', 'Safety-Impacting', and 'Both', the code was correctly
adjusted to map these values. The statistical analysis was rigorous, using a
Chi-square test and Phi coefficient to validate the association.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO13960 scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print("--- Analysis of Impact vs SAOP Review ---")

# 1. Clean Independent Variable: Impact
# Mapping based on previous inspection:
# High Impact = 'Rights-Impacting', 'Safety-Impacting', 'Both'
# Low Impact = 'Neither'
def categorize_impact(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if val_str in ['Rights-Impacting', 'Safety-Impacting', 'Safety-impacting', 'Both']:
        return 'High Impact'
    elif val_str == 'Neither':
        return 'Low Impact'
    else:
        return np.nan # Exclude NaN or unknown categories

eo_data['impact_binary'] = eo_data['17_impact_type'].apply(categorize_impact)

# 2. Clean Dependent Variable: SAOP Review
# Mapping: Yes/YES -> Yes, everything else (No, NaN, blank) -> No
def categorize_review(val):
    if pd.isna(val):
        return 'No'
    val_str = str(val).strip().upper()
    if val_str == 'YES':
        return 'Yes'
    return 'No'

eo_data['review_binary'] = eo_data['30_saop_review'].apply(categorize_review)

# Filter for valid impact data
valid_data = eo_data.dropna(subset=['impact_binary'])

# 3. Create Contingency Table
contingency_table = pd.crosstab(valid_data['impact_binary'], valid_data['review_binary'])

print("\nContingency Table (Impact [Rows] vs SAOP Review [Cols]):")
print(contingency_table)

# 4. Statistical Analysis
if 'Yes' in contingency_table.columns and 'High Impact' in contingency_table.index and 'Low Impact' in contingency_table.index:
    # Probability calculations
    high_row = contingency_table.loc['High Impact']
    low_row = contingency_table.loc['Low Impact']
    
    n_high = high_row.sum()
    k_high_yes = high_row['Yes'] if 'Yes' in high_row else 0
    prob_high = k_high_yes / n_high if n_high > 0 else 0
    
    n_low = low_row.sum()
    k_low_yes = low_row['Yes'] if 'Yes' in low_row else 0
    prob_low = k_low_yes / n_low if n_low > 0 else 0
    
    print(f"\nProbability of Review | High Impact: {prob_high:.2%} ({k_high_yes}/{n_high})")
    print(f"Probability of Review | Low Impact:  {prob_low:.2%} ({k_low_yes}/{n_low})")

    # Chi-square Test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Phi Coefficient
    n_total = contingency_table.sum().sum()
    phi = np.sqrt(chi2 / n_total)
    
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    print(f"Phi Coefficient: {phi:.4f}")
    
    if p < 0.05:
        print("Result: Statistically Significant.")
        if prob_high > prob_low:
            print("Direction: Supports hypothesis (Higher impact leads to higher review probability).")
        else:
            print("Direction: Contradicts hypothesis.")
    else:
        print("Result: Not Statistically Significant.")
else:
    print("\nData structure insufficient for test (missing categories).")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Analysis of Impact vs SAOP Review ---

Contingency Table (Impact [Rows] vs SAOP Review [Cols]):
review_binary    No  Yes
impact_binary           
High Impact     170   57
Low Impact     1313  178

Probability of Review | High Impact: 25.11% (57/227)
Probability of Review | Low Impact:  11.94% (178/1491)

Chi-square Statistic: 27.8427
P-value: 1.3159e-07
Phi Coefficient: 0.1273
Result: Statistically Significant.
Direction: Supports hypothesis (Higher impact leads to higher review probability).

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
