# Experiment 47: node_4_13

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_13` |
| **ID in Run** | 47 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:02:58.374044+00:00 |
| **Runtime** | 244.4s |
| **Parent** | `node_3_13` |
| **Children** | None |
| **Creation Index** | 48 |

---

## Hypothesis

> There is a significant structural alignment between source frameworks and
competency bundles: regulatory frameworks (e.g., EU AI Act) disproportionately
drive 'Trust Readiness', while technical frameworks (e.g., OWASP Top 10 LLM)
drive 'Integration Readiness'.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9355 (Definitely True) |
| **Posterior** | 0.7975 (Likely True) |
| **Surprise** | -0.1601 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

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
| Definitely True | 0.0 |
| Maybe True | 90.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Test the independence between the source framework and the resulting competency bundle.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Filter the dataset to include rows where 'source' is 'EU AI Act' or 'OWASP Top 10 LLM' (or include all sources if sample size permits).
- 3. Create a contingency table of 'source' vs. 'bundle'.
- 4. Perform a Chi-square test of independence to determine if the source framework significantly predicts the bundle classification.

### Deliverables
- Contingency table and Chi-square test statistics.

---

## Analysis

The experiment tested the hypothesis that regulatory frameworks (EU AI Act)
align with 'Trust Readiness' while technical frameworks (OWASP Top 10 LLM) align
with 'Integration Readiness'. The analysis of 'step2_crosswalk_matrix.csv'
produced a 2x2 contingency table: EU AI Act mapped to 4 Integration/5 Trust
competencies, while OWASP mapped to 9 Integration/1 Trust. Although the trend
strongly supports the hypothesis (OWASP is 90% Integration-focused vs. EU AI
Act's 56% Trust-focus), the Fisher's Exact Test yielded a p-value of 0.0573,
which is marginally above the standard significance threshold of 0.05.
Consequently, while the structural alignment is visually apparent, it is not
statistically significant due to the small sample size (n=19).

---

## Review

The experiment was successfully executed and faithfully implemented the analysis
plan.

**Hypothesis**: Regulatory frameworks (EU AI Act) align with 'Trust Readiness'
while technical frameworks (OWASP Top 10 LLM) align with 'Integration
Readiness'.

**Method**: A subset of the `step2_crosswalk_matrix.csv` dataset was filtered to
isolate 'EU AI Act (2024/1689)' and 'OWASP Top 10 LLM'. A 2x2 contingency table
was constructed, and Fisher's Exact Test was applied due to small cell counts
(one cell < 5).

**Results**:
- **OWASP Top 10 LLM**: Heavily skewed toward Integration (9 Integration vs. 1
Trust).
- **EU AI Act**: More balanced, slightly leaning toward Trust (4 Integration vs.
5 Trust).
- **Statistical Test**: Fisher's Exact Test yielded a p-value of **0.0573**
(Odds Ratio: 0.0889).

**Conclusion**: While there is a strong directional trend supporting the
hypothesis (technical frameworks driving integration controls), the result is
marginally non-significant at the standard alpha=0.05 level, likely due to the
small sample size (N=19). We strictly fail to reject the null hypothesis, but
the low p-value suggests a potential relationship that might reach significance
with a larger dataset.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import os

# Define filename
filename = 'step2_crosswalk_matrix.csv'

# Try loading from parent directory first, then current directory
filepath = os.path.join('..', filename)
if not os.path.exists(filepath):
    filepath = filename

print(f"Loading dataset from: {filepath}")

try:
    df = pd.read_csv(filepath)
    print(f"Successfully loaded. Shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# Define the specific sources to compare based on previous output
regulatory_source = 'EU AI Act (2024/1689)'
technical_source = 'OWASP Top 10 LLM'

print(f"\n=== Hypothesis Testing: Regulatory ({regulatory_source}) vs Technical ({technical_source}) ===")

# Filter dataset
subset_df = df[df['source'].isin([regulatory_source, technical_source])]

# Create Contingency Table
contingency_table = pd.crosstab(subset_df['source'], subset_df['bundle'])
print("\nContingency Table:")
print(contingency_table)

# Check if table is 2x2
if contingency_table.shape == (2, 2):
    # Perform Fisher's Exact Test (appropriate for small sample sizes where cell counts < 5)
    # The table has a cell with value 1, so Fisher's is more robust than Chi-square here.
    odds_ratio, p_value_fisher = stats.fisher_exact(contingency_table)
    
    print(f"\nFisher's Exact Test:")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value_fisher:.4f}")
    
    # Also performing Chi-Square for completeness as requested
    chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table, correction=True)
    print(f"\nChi-square Test (with Yates correction):")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p_value_chi2:.4f}")

    alpha = 0.05
    if p_value_fisher < alpha:
        print("\nResult: Significant difference found. The source framework strongly predicts the competency bundle.")
    else:
        print("\nResult: No statistically significant difference found (p >= 0.05).")
else:
    print("\nError: Contingency table is not 2x2. Check source names.")
    print("Found sources:", subset_df['source'].unique())
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step2_crosswalk_matrix.csv
Successfully loaded. Shape: (42, 24)

=== Hypothesis Testing: Regulatory (EU AI Act (2024/1689)) vs Technical (OWASP Top 10 LLM) ===

Contingency Table:
bundle                 Integration Readiness  Trust Readiness
source                                                       
EU AI Act (2024/1689)                      4                5
OWASP Top 10 LLM                           9                1

Fisher's Exact Test:
Odds Ratio: 0.0889
P-value: 0.0573

Chi-square Test (with Yates correction):
Chi2 Statistic: 2.6856
P-value: 0.1013

Result: No statistically significant difference found (p >= 0.05).

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
