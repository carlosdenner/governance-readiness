# Experiment 82: node_4_27

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_27` |
| **ID in Run** | 82 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:09:19.696910+00:00 |
| **Runtime** | 164.5s |
| **Parent** | `node_3_8` |
| **Children** | None |
| **Creation Index** | 83 |

---

## Hypothesis

> 'Prevention Failures' are disproportionately represented in 'Integration-
dominant' incidents, while 'Detection' and 'Response' failures are
disproportionately represented in 'Trust-dominant' incidents, suggesting that
technical controls are the primary gatekeepers while governance is responsible
for oversight and reaction.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7339 (Likely True) |
| **Posterior** | 0.5771 (Maybe True) |
| **Surprise** | -0.1819 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 29.0 |
| Uncertain | 1.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 63.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Investigate the relationship between the mode of failure and the type of competency gap.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Filter for rows where 'trust_integration_split' is 'trust-dominant' or 'integration-dominant'.
- 3. Create a simplified failure mode variable: 'Prevention' vs 'Post-Prevention' (grouping Detection and Response).
- 4. Create a contingency table of Failure Mode vs. Split.
- 5. Perform a Fisher's Exact Test.

### Deliverables
- Contingency table and Fisher's Exact Test results.

---

## Analysis

The experiment was successfully executed but revealed that the dataset lacks
sufficient variance to support the hypothesis. Out of 52 total incidents, only 6
were classified as purely 'trust-dominant' (n=2) or 'integration-dominant'
(n=4), as the vast majority (n=46) were classified as 'both'. Furthermore,
within this small subset of 6 incidents, 100% were classified as 'Prevention'
failures. Due to the absence of 'Post-Prevention' failures in the filtered group
(a column of zeros in the contingency table), the Fisher's Exact Test yielded a
p-value of 1.0 and an undefined odds ratio. Consequently, no statistical
relationship could be established between the dominant competency bundle and the
failure mode.

---

## Review

Hypothesis: 'Prevention Failures' are disproportionately represented in
'Integration-dominant' incidents, while 'Detection' and 'Response' failures are
disproportionately represented in 'Trust-dominant' incidents.

Results: The experiment was successfully executed on the
'step3_incident_coding.csv' dataset. However, the data revealed extreme skewness
that prevented meaningful statistical testing. Out of 52 total incidents, 46
were classified as 'both', leaving only 6 incidents for the specific comparison
(4 'integration-dominant', 2 'trust-dominant'). Furthermore, within this small
subset, 100% of the incidents were 'Prevention' failures; there were zero
'Detection' or 'Response' failures.

Findings: Fisher's Exact Test resulted in a p-value of 1.0 and an undefined odds
ratio due to the empty 'Post-Prevention' column in the contingency table.
Consequently, the hypothesis could not be supported. The analysis highlights two
major characteristics of the dataset: 1) Failure modes are almost exclusively
prevention-based, and 2) The majority of incidents require both Trust and
Integration competencies, making the 'dominant' classification a rare edge case.

---

## Code

```python
import pandas as pd
from scipy.stats import fisher_exact
import os

# 1. Load 'step3_incident_coding.csv'
file_path = '../step3_incident_coding.csv'
if not os.path.exists(file_path):
    # Fallback for local testing if needed
    file_path = 'step3_incident_coding.csv'

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit(1)

# 2. Filter for rows where 'trust_integration_split' is 'trust-dominant' or 'integration-dominant'
target_splits = ['trust-dominant', 'integration-dominant']
filtered_df = df[df['trust_integration_split'].isin(target_splits)].copy()

print(f"\nTotal incidents: {len(df)}")
print(f"Filtered incidents (excluding 'both'): {len(filtered_df)}")
print(f"Counts by split in filtered data:\n{filtered_df['trust_integration_split'].value_counts()}")

# 3. Create simplified failure mode variable: 'Prevention' vs 'Post-Prevention'
def simplify_failure(mode):
    if pd.isna(mode):
        return "Unknown"
    mode = str(mode).lower()
    if 'prevention' in mode:
        return 'Prevention'
    elif 'detection' in mode or 'response' in mode:
        return 'Post-Prevention'
    else:
        return 'Other'

filtered_df['failure_category'] = filtered_df['failure_mode'].apply(simplify_failure)

# 4. Create a contingency table of Failure Mode vs. Split
# We ensure the table structure is fixed for the test (2x2)
# Rows: integration-dominant, trust-dominant
# Cols: Prevention, Post-Prevention

expected_rows = ['integration-dominant', 'trust-dominant']
expected_cols = ['Prevention', 'Post-Prevention']

# Initialize matrix with zeros
test_matrix = pd.DataFrame(0, index=expected_rows, columns=expected_cols)

# Fill with actual counts
contingency = pd.crosstab(filtered_df['trust_integration_split'], filtered_df['failure_category'])

for r in expected_rows:
    for c in expected_cols:
        if r in contingency.index and c in contingency.columns:
            test_matrix.loc[r, c] = contingency.loc[r, c]

print("\nContingency Table (for Fisher's Exact Test):")
print(test_matrix)

# 5. Perform Fisher's Exact Test
# Note: If the sample size is very small or one category is empty, this handles it.
total_observations = test_matrix.sum().sum()
if total_observations < 2:
    print("\nInsufficient data to perform Fisher's Exact Test.")
else:
    odds_ratio, p_value = fisher_exact(test_matrix.to_numpy())
    print("\nFisher's Exact Test Results:")
    print(f"Odds Ratio: {odds_ratio}")
    print(f"P-value: {p_value}")
    
    if p_value < 0.05:
        print("Result: Statistically significant association.")
    else:
        print("Result: No statistically significant association found (null hypothesis retained).")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Dataset loaded successfully.

Total incidents: 52
Filtered incidents (excluding 'both'): 6
Counts by split in filtered data:
trust_integration_split
integration-dominant    4
trust-dominant          2
Name: count, dtype: int64

Contingency Table (for Fisher's Exact Test):
                      Prevention  Post-Prevention
integration-dominant           4                0
trust-dominant                 2                0

Fisher's Exact Test Results:
Odds Ratio: nan
P-value: 1.0
Result: No statistically significant association found (null hypothesis retained).

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
