# Experiment 65: node_6_5

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_5` |
| **ID in Run** | 65 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:37:51.375872+00:00 |
| **Runtime** | 198.4s |
| **Parent** | `node_5_5` |
| **Children** | None |
| **Creation Index** | 66 |

---

## Hypothesis

> Governance requirements sourced from the 'OWASP Top 10 LLM' are significantly
more likely to be classified as 'Integration Readiness' compared to requirements
from the 'EU AI Act', reflecting the distinct technical vs. policy focus of
these normative frameworks.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.7479 (Likely True) |
| **Surprise** | +0.0070 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

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
| Definitely True | 0.0 |
| Maybe True | 90.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Determine if there is a statistically significant association between the source of a governance requirement and its capability bundle classification.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Filter the dataset to include only rows where 'source' is 'OWASP Top 10 LLM' or 'EU AI Act'.
- 3. Create a contingency table of 'source' vs. 'bundle'.
- 4. Perform a Chi-square test of independence (or Fisher's Exact Test if counts are low) to assess the relationship.

### Deliverables
- Contingency table, Chi-square statistic, and p-value.

---

## Analysis

The experiment successfully loaded 'step2_crosswalk_matrix.csv' and analyzed the
relationship between the source of a governance requirement and its bundle
classification. To address previous string matching errors, the analysis
normalized the 'source' column and identified 'EU AI Act (2024/1689)' and 'OWASP
Top 10 LLM' as the target groups.

The resulting contingency table showed a clear directional trend:
- **OWASP Top 10 LLM**: 90% 'Integration Readiness' (9/10), 10% 'Trust
Readiness' (1/10).
- **EU AI Act**: ~44% 'Integration Readiness' (4/9), ~56% 'Trust Readiness'
(5/9).

Fisher's Exact Test yielded a p-value of **0.0573**. While this result is
marginally above the standard significance threshold of 0.05, it indicates a
strong trend suggesting that OWASP requirements are indeed more
technical/integration-focused, whereas the EU AI Act is more balanced or policy-
oriented. Strictly speaking, however, the null hypothesis cannot be rejected at
the 95% confidence level.

---

## Review

The experiment was successfully executed and the analysis is methodologically
sound. The third attempt correctly handled the string matching issues for the
'EU AI Act' source, allowing for a valid comparison. The hypothesis posited a
significant difference in bundle classification between 'OWASP Top 10 LLM' and
'EU AI Act'. The data revealed a strong directional trend: OWASP requirements
were 90% 'Integration Readiness' (9/10), whereas EU AI Act requirements were
only 44% 'Integration Readiness' (4/9). However, Fisher's Exact Test yielded a
p-value of 0.0573, which is marginally above the standard significance threshold
of 0.05. Consequently, while the distinct technical focus of OWASP is observable
in the proportions, the hypothesis of a *statistically significant* difference
is strictly rejected at the 95% confidence level, likely due to the small sample
size (n=19).

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import numpy as np

# Load the dataset
filename = 'step2_crosswalk_matrix.csv'
df = pd.read_csv(filename)

# Inspect and clean the 'source' column
# We normalize by stripping whitespace to handle potential formatting issues
df['source'] = df['source'].astype(str).str.strip()

# Define the target groups using robust string matching
# This handles cases where the string might be 'EU AI Act ' or similar variants
def classify_source(s):
    if 'OWASP' in s:
        return 'OWASP Top 10 LLM'
    elif 'EU AI' in s:
        return 'EU AI Act'
    else:
        return None

# Apply classification
df['target_source'] = df['source'].apply(classify_source)

# Filter for only the relevant rows
sub_df = df[df['target_source'].notna()].copy()

print("--- Data Inspection ---")
print(f"Unique sources found in raw data: {df['source'].unique()}")
print(f"Filtered dataset shape: {sub_df.shape}")
print("Counts per target source:")
print(sub_df['target_source'].value_counts())

# Create Contingency Table
contingency_table = pd.crosstab(sub_df['target_source'], sub_df['bundle'])
print("\n--- Contingency Table ---")
print(contingency_table)

# Perform Statistical Test
# We use Fisher's Exact Test if the table is 2x2, as sample sizes are likely small
if contingency_table.shape == (2, 2):
    test_name = "Fisher's Exact Test"
    # fisher_exact returns (odds_ratio, p_value)
    statistic, p_value = stats.fisher_exact(contingency_table)
    stat_label = "Odds Ratio"
else:
    test_name = "Chi-Square Test"
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    statistic = chi2
    stat_label = "Chi-Square Statistic"

print(f"\n--- Statistical Analysis ({test_name}) ---")
print(f"{stat_label}: {statistic:.4f}")
print(f"P-Value: {p_value:.4f}")

# Interpretation
alpha = 0.05
print("\n--- Interpretation ---")
if p_value < alpha:
    print(f"Result: Significant (p < {alpha}). The source framework predicts the readiness bundle.")
else:
    print(f"Result: Not Significant (p >= {alpha}). No statistical evidence that source predicts bundle.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Data Inspection ---
Unique sources found in raw data: <StringArray>
[      'NIST AI RMF 1.0',    'NIST GenAI Profile', 'EU AI Act (2024/1689)',
      'OWASP Top 10 LLM']
Length: 4, dtype: str
Filtered dataset shape: (19, 25)
Counts per target source:
target_source
OWASP Top 10 LLM    10
EU AI Act            9
Name: count, dtype: int64

--- Contingency Table ---
bundle            Integration Readiness  Trust Readiness
target_source                                           
EU AI Act                             4                5
OWASP Top 10 LLM                      9                1

--- Statistical Analysis (Fisher's Exact Test) ---
Odds Ratio: 0.0889
P-Value: 0.0573

--- Interpretation ---
Result: Not Significant (p >= 0.05). No statistical evidence that source predicts bundle.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
