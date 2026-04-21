# Experiment 90: node_5_22

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_22` |
| **ID in Run** | 90 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:26:41.312673+00:00 |
| **Runtime** | 140.8s |
| **Parent** | `node_4_17` |
| **Children** | None |
| **Creation Index** | 91 |

---

## Hypothesis

> Normative governance frameworks (e.g., NIST, EU AI Act) are significantly more
likely to map to 'Trust Readiness' competencies, whereas technical guidelines
(e.g., OWASP) are significantly more likely to map to 'Integration Readiness'
competencies.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9355 (Definitely True) |
| **Posterior** | 0.9835 (Definitely True) |
| **Surprise** | +0.0557 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 24.0 |
| Maybe True | 6.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 90.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Determine if there is a statistically significant association between the source of the requirement (Governance vs. Technical) and the resulting competency bundle.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Create a 'source_type' variable: classify 'OWASP Top 10 LLM' as 'Technical' and all others (NIST AI RMF, NIST GenAI, EU AI Act) as 'Normative'.
- 3. Create a contingency table of 'source_type' vs. 'bundle'.
- 4. Perform a Chi-square test of independence (or Fisher's Exact Test if counts are low).
- 5. Calculate Cramer's V to estimate effect size.

### Deliverables
- Contingency table, Chi-square statistic, p-value, Cramer's V.

---

## Analysis

The experiment successfully tested the hypothesis that the source of a
governance requirement dictates its competency bundle. Processing the
'step2_crosswalk_matrix.csv' dataset (n=42), requirements were classified as
'Normative' (NIST AI RMF, NIST GenAI, EU AI Act; n=32) or 'Technical' (OWASP Top
10 LLM; n=10). The analysis revealed a statistically significant association
between source type and bundle classification (Fisher's Exact p=0.0128; Chi-
square p=0.0277). Specifically, 'Technical' sources overwhelmingly mapped to
'Integration Readiness' (90%, 9/10), whereas 'Normative' sources were more
balanced but leaned toward 'Trust Readiness' (56%, 18/32). With a moderate
effect size (Cramer's V = 0.34), the hypothesis is supported: technical
guidelines drive engineering controls, while normative frameworks drive
governance and policy controls.

---

## Review

The experiment successfully tested the hypothesis that the source of a
governance requirement dictates its competency bundle. Processing the
'step2_crosswalk_matrix.csv' dataset (n=42), requirements were classified as
'Normative' (NIST AI RMF, NIST GenAI, EU AI Act; n=32) or 'Technical' (OWASP Top
10 LLM; n=10). The analysis revealed a statistically significant association
between source type and bundle classification (Fisher's Exact p=0.0128; Chi-
square p=0.0277). Specifically, 'Technical' sources overwhelmingly mapped to
'Integration Readiness' (90%, 9/10), whereas 'Normative' sources were more
balanced but leaned toward 'Trust Readiness' (56%, 18/32). With a moderate
effect size (Cramer's V = 0.34), the hypothesis is supported: technical
guidelines drive engineering controls, while normative frameworks drive
governance and policy controls.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import os

# [debug] Check if file exists in current or parent directory
file_name = 'step2_crosswalk_matrix.csv'
file_path = file_name
if not os.path.exists(file_path):
    if os.path.exists(f'../{file_name}'):
        file_path = f'../{file_name}'
    else:
        print(f"Error: {file_name} not found.")

print(f"Loading dataset from: {file_path}")

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Failed to load CSV: {e}")
    exit(1)

# Map sources to types
# Technical: OWASP Top 10 LLM
# Normative: NIST AI RMF 1.0, NIST GenAI Profile, EU AI Act

def map_source(source_name):
    if 'OWASP' in str(source_name):
        return 'Technical'
    else:
        return 'Normative'

df['source_type'] = df['source'].apply(map_source)

# Create Contingency Table
contingency_table = pd.crosstab(df['source_type'], df['bundle'])

print("\n=== Contingency Table (Source Type vs Bundle) ===")
print(contingency_table)

# Calculate percentages for better context
contingency_pct = pd.crosstab(df['source_type'], df['bundle'], normalize='index') * 100
print("\n=== Contingency Table (Percentages) ===")
print(contingency_pct.round(1))

# Statistical Testing
# Using Fisher's Exact Test if 2x2, otherwise Chi-Square
chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

print("\n=== Chi-Square Test of Independence ===")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p_val:.4f}")

# Calculate Cramer's V (Effect Size)
n = contingency_table.sum().sum()
min_dim = min(contingency_table.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim))

print(f"Cramer's V: {cramers_v:.4f}")

# Fisher's Exact Test (specifically for 2x2 small samples, which this likely is)
if contingency_table.shape == (2, 2):
    # Note: scipy returns odds ratio and p-value
    odds_ratio, fisher_p = stats.fisher_exact(contingency_table)
    print("\n=== Fisher's Exact Test (2x2) ===")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {fisher_p:.4f}")

# Interpretation
alpha = 0.05
print("\n=== Interpretation ===")
if p_val < alpha:
    print("Result: Statistically Significant Association.")
    print("There is a significant relationship between the source type (Normative vs Technical) and the competency bundle.")
else:
    print("Result: No Statistically Significant Association.")
    print("The source type does not appear to dictate the competency bundle significantly in this dataset.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step2_crosswalk_matrix.csv

=== Contingency Table (Source Type vs Bundle) ===
bundle       Integration Readiness  Trust Readiness
source_type                                        
Normative                       14               18
Technical                        9                1

=== Contingency Table (Percentages) ===
bundle       Integration Readiness  Trust Readiness
source_type                                        
Normative                     43.8             56.2
Technical                     90.0             10.0

=== Chi-Square Test of Independence ===
Chi2 Statistic: 4.8442
P-value: 0.0277
Cramer's V: 0.3396

=== Fisher's Exact Test (2x2) ===
Odds Ratio: 0.0864
P-value: 0.0128

=== Interpretation ===
Result: Statistically Significant Association.
There is a significant relationship between the source type (Normative vs Technical) and the competency bundle.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
