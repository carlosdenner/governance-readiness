# Experiment 81: node_5_17

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_17` |
| **ID in Run** | 81 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:09:19.695873+00:00 |
| **Runtime** | 165.1s |
| **Parent** | `node_4_19` |
| **Children** | None |
| **Creation Index** | 82 |

---

## Hypothesis

> Incidents classified as 'Security' harm types are significantly more likely to
manifest as 'Prevention Failures' compared to 'Reliability' incidents, which are
more likely to manifest as Detection or Response failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7581 (Likely True) |
| **Posterior** | 0.3740 (Maybe False) |
| **Surprise** | -0.4458 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 2.0 |
| Maybe True | 28.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 87.0 |
| Definitely False | 3.0 |

---

## Experiment Plan

**Objective:** Analyze the relationship between harm type and failure mode to understand if security issues are predominantly preventative gaps.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Filter the dataset to include only rows where 'harm_type' is 'security' or 'reliability'.
- 3. Create a binary column 'is_prevention_failure' (1 if failure_mode == 'prevention_failure', 0 otherwise).
- 4. Create a contingency table comparing Harm Type (Security vs. Reliability) against Failure Mode (Prevention vs. Other).
- 5. Perform Fisher's Exact Test to assess statistical significance.

### Deliverables
- Contingency table and Fisher's Exact Test p-value.

---

## Analysis

The experiment successfully loaded the 'step3_incident_coding.csv' dataset and
filtered for 'Security' (n=36) and 'Reliability' (n=4) incidents. The analysis
revealed a near-total dominance of 'Prevention Failure' as the recorded failure
mode (39 out of 40 incidents). Specifically, Security incidents showed 35
prevention failures and 1 detection failure, while Reliability incidents showed
4 prevention failures and 0 others. Consequently, Fisher's Exact Test yielded a
p-value of 1.0, indicating no statistically significant difference between the
groups. The hypothesis could not be validated due to the lack of variance in the
'failure_mode' variable, as both harm types were overwhelmingly characterized by
prevention failures.

---

## Review

The experiment was successfully executed and the code faithfully followed the
analysis plan. The dataset 'step3_incident_coding.csv' was correctly loaded and
filtered for 'Security' (n=36) and 'Reliability' (n=4) incidents. The analysis
revealed a near-total dominance of 'Prevention Failure' across the board,
accounting for 39 out of the 40 analyzed incidents. Specifically, Security
incidents had 35 prevention failures and 1 detection failure, while Reliability
incidents had 4 prevention failures and 0 others. Consequently, Fisher's Exact
Test yielded a p-value of 1.0, indicating no statistically significant
difference between the groups. The hypothesis that Security incidents are *more*
likely to be prevention failures than Reliability incidents cannot be supported,
as both harm types in this dataset predominantly manifest as prevention
failures.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import os

# Define file path based on instructions
file_path = '../step3_incident_coding.csv'

# Check if file exists at the expected location, else try current directory
if not os.path.exists(file_path):
    if os.path.exists('step3_incident_coding.csv'):
        file_path = 'step3_incident_coding.csv'
    else:
        print(f"Error: File not found at {file_path} or current directory.")
        exit(1)

print(f"Loading dataset from: {file_path}")

# Load the dataset
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Filter for 'security' and 'reliability' harm types
subset = df[df['harm_type'].isin(['security', 'reliability'])].copy()

print(f"Filtered subset size: {len(subset)} rows")
print(f"Harm Type counts:\n{subset['harm_type'].value_counts()}")
print(f"Failure Mode counts:\n{subset['failure_mode'].value_counts()}")

# Create binary classification: Prevention vs Other
# Note: 'Other' includes detection_failure, response_failure, etc.
subset['is_prevention'] = subset['failure_mode'].apply(lambda x: 1 if x == 'prevention_failure' else 0)

# Construct Contingency Table
# Rows: Security, Reliability
# Columns: Prevention (1), Other (0)

# Counts
sec_prev = len(subset[(subset['harm_type'] == 'security') & (subset['is_prevention'] == 1)])
sec_other = len(subset[(subset['harm_type'] == 'security') & (subset['is_prevention'] == 0)])

rel_prev = len(subset[(subset['harm_type'] == 'reliability') & (subset['is_prevention'] == 1)])
rel_other = len(subset[(subset['harm_type'] == 'reliability') & (subset['is_prevention'] == 0)])

# Table structure: [[Security_Prev, Security_Other], [Reliability_Prev, Reliability_Other]]
contingency_table = [[sec_prev, sec_other], [rel_prev, rel_other]]

print("\nContingency Table:")
print("              | Prevention | Other")
print(f"Security      | {sec_prev:<10} | {sec_other}")
print(f"Reliability   | {rel_prev:<10} | {rel_other}")

# Perform Fisher's Exact Test
# We use two-sided to test for any significant difference in distribution
odds_ratio, p_value = stats.fisher_exact(contingency_table, alternative='two-sided')

print(f"\nFisher's Exact Test Results:")
print(f"Odds Ratio: {odds_ratio}")
print(f"P-value: {p_value}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("\nResult: Statistically significant difference found.")
else:
    print("\nResult: No statistically significant difference found.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_incident_coding.csv
Filtered subset size: 40 rows
Harm Type counts:
harm_type
security       36
reliability     4
Name: count, dtype: int64
Failure Mode counts:
failure_mode
prevention_failure    39
detection_failure      1
Name: count, dtype: int64

Contingency Table:
              | Prevention | Other
Security      | 35         | 1
Reliability   | 4          | 0

Fisher's Exact Test Results:
Odds Ratio: 0.0
P-value: 1.0

Result: No statistically significant difference found.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
