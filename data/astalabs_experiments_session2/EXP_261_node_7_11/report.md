# Experiment 261: node_7_11

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_11` |
| **ID in Run** | 261 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:29:24.673917+00:00 |
| **Runtime** | 183.1s |
| **Parent** | `node_6_57` |
| **Children** | None |
| **Creation Index** | 262 |

---

## Hypothesis

> Control Clustering: The governance controls 'Real World Testing' and
'Independent Evaluation' are positively correlated, suggesting they form a
coherent 'Verification & Validation' maturity bundle rather than being applied
randomly.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.9121 (Definitely True) |
| **Surprise** | +0.2042 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | Yes |

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
| Definitely True | 60.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Assess the co-occurrence of Testing and Independent Evaluation controls.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `source_table` = 'eo13960_scored'.
- 2. Create binary variables for `53_real_world_testing` and `55_independent_eval`.
- 3. Generate a 2x2 contingency table.
- 4. Calculate the Phi Coefficient (correlation for binary variables) and perform a Chi-square test.
- 5. Report the conditional probability: P(Evaluation | Testing).

### Deliverables
- Contingency table, Phi coefficient, and conditional probability analysis.

---

## Analysis

The experiment successfully validated the 'Control Clustering' hypothesis using
the EO 13960 dataset. After refining the text parsing logic to correctly
identify verbose descriptions of 'Real World Testing' (e.g., 'operational
environment') and 'Independent Evaluation', the analysis found a statistically
significant positive correlation between the two controls.

**Findings:**
1. **Contingency:**
   - **No Testing:** 1608 cases had neither; only 1 case had Evaluation without
Testing.
   - **Has Testing:** Of 148 cases with Testing, 52 (35.1%) also had Independent
Evaluation.
2. **Statistics:**
   - **Chi-Square:** 557.96 (p < 1e-120), indicating the relationship is not
random.
   - **Phi Coefficient:** 0.5635, representing a strong positive association.
   - **Conditional Probability:** P(Independent Eval | Real World Testing) =
35.14%, compared to <0.1% for cases without testing.

**Conclusion:**
The hypothesis is **supported**. The data suggests a strict maturity hierarchy:
'Independent Evaluation' is virtually non-existent (1/1609) unless 'Real World
Testing' is arguably already present, confirming they function as a coherent
governance bundle where Testing is likely a prerequisite for Evaluation.

---

## Review

The experiment was successfully executed and the code faithfully implemented the
analysis plan. The refined text parsing logic correctly handled the verbose
descriptions in the EO 13960 dataset (e.g., mapping 'Performance evaluation in
operational environment' to 'Real World Testing').

**Hypothesis:** Control Clustering (Governance controls 'Real World Testing' and
'Independent Evaluation' are positively correlated).

**Findings:**
1.  **Data Distribution:**
    -   **No Testing:** 1609 cases lacked real-world testing. Of these, only
**1** case (0.06%) had independent evaluation, indicating that independent
evaluation virtually never happens without prior testing.
    -   **Has Testing:** 148 cases had real-world testing. Of these, **52**
cases (35.1%) also had independent evaluation.
2.  **Statistical Test:**
    -   **Chi-Square Statistic:** 557.96 (p < 0.001), indicating a highly
significant relationship.
    -   **Phi Coefficient:** 0.5635, showing a strong positive association.
    -   **Conditional Probability:** P(Independent Eval | Real World Testing) =
35.14%.

**Conclusion:**
The hypothesis is **supported**. The data reveals a clear maturity hierarchy or
dependency: 'Real World Testing' acts as a prerequisite gate for 'Independent
Evaluation'. The near-zero probability of having evaluation without testing
suggests these controls are not applied randomly but follow a structured
implementation sequence.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# Define column names based on previous exploration
col_test = '53_real_world_testing'
col_eval = '55_independent_eval'

# strict mapping functions based on unique values
def map_testing(val):
    if pd.isna(val):
        return 0
    s = str(val).lower()
    # 'operational environment' covers both 'Performance evaluation...' and 'Impact evaluation...'
    if 'operational environment' in s:
        return 1
    # Legacy/Simple boolean values
    if s in ['yes', 'true', '1']:
        return 1
    # 'Benchmark evaluation' explicitly states 'not been tested in an operational environment'
    return 0

def map_eval(val):
    if pd.isna(val):
        return 0
    s = str(val).lower()
    # Check for explicit 'yes' or 'true'
    if s.startswith('yes') or s == 'true' or s == '1':
        return 1
    return 0

# Apply mapping
eo_df['has_testing'] = eo_df[col_test].apply(map_testing)
eo_df['has_eval'] = eo_df[col_eval].apply(map_eval)

# Generate Contingency Table
contingency = pd.crosstab(eo_df['has_testing'], eo_df['has_eval'])
print("Contingency Table (Rows: Real World Testing, Cols: Independent Eval):")
print(contingency)

# Stats
chi2, p, dof, ex = chi2_contingency(contingency)

# Phi Coefficient
n = contingency.sum().sum()
phi = np.sqrt(chi2 / n)

# Conditional Probability P(Eval | Testing)
# Testing=1 is the second row (index 1)
try:
    testing_yes_total = contingency.loc[1].sum()
    testing_yes_eval_yes = contingency.loc[1, 1]
    prob_eval_given_testing = testing_yes_eval_yes / testing_yes_total if testing_yes_total > 0 else 0
except KeyError:
    # Handle case where index 1 might not exist if no testing found
    prob_eval_given_testing = 0
    testing_yes_total = 0

print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Phi Coefficient: {phi:.4f}")
print(f"Conditional Probability P(Independent Eval | Real World Testing): {prob_eval_given_testing:.2%}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Contingency Table (Rows: Real World Testing, Cols: Independent Eval):
has_eval        0   1
has_testing          
0            1608   1
1              96  52

Chi-Square Statistic: 557.9624
P-value: 2.3328e-123
Phi Coefficient: 0.5635
Conditional Probability P(Independent Eval | Real World Testing): 35.14%

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
