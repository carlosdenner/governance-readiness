# Experiment 75: node_5_15

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_15` |
| **ID in Run** | 75 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:57:47.219108+00:00 |
| **Runtime** | 241.8s |
| **Parent** | `node_4_17` |
| **Children** | None |
| **Creation Index** | 76 |

---

## Hypothesis

> AI incidents classified as 'Security' harms are significantly more likely to
manifest as 'Prevention Failures', whereas non-security incidents (Reliability,
Privacy, etc.) are significantly more likely to manifest as 'Detection' or
'Response' failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.5645 (Maybe True) |
| **Posterior** | 0.1446 (Likely False) |
| **Surprise** | -0.4873 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 19.0 |
| Uncertain | 0.0 |
| Maybe False | 11.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 90.0 |

---

## Experiment Plan

**Objective:** Assess whether failure modes differ significantly by harm type (Security vs. Other) using the incident coding dataset.

### Steps
- 1. Load the dataset 'step3_incident_coding.csv' from the current directory.
- 2. Create a new column 'Harm_Category' derived from 'harm_type': set to 'Security' if the value is 'security', otherwise set to 'Other'.
- 3. Create a new column 'Failure_Category' derived from 'failure_mode': set to 'Prevention' if the text contains 'prevention', otherwise set to 'Detection/Response'.
- 4. Generate a 2x2 contingency table (crosstab) of Harm_Category (rows) vs. Failure_Category (columns).
- 5. Perform a Fisher's Exact Test on the contingency table to determine if there is a statistically significant association.
- 6. Output the contingency table, the Odds Ratio, and the P-value.

### Deliverables
- 1. Contingency table of Harm Category vs. Failure Category.
- 2. Fisher's Exact Test results (Odds Ratio and P-value).

---

## Analysis

The experiment successfully tested the hypothesis that failure modes differ by
harm type using Fisher's Exact Test on 52 incidents from
'step3_incident_coding.csv'. The data revealed an extreme imbalance: 51 of 52
incidents (98%) were classified as 'Prevention' failures. Contrary to the
hypothesis that non-security incidents (Reliability, Privacy, etc.) would be
more likely to manifest as Detection/Response failures, the 'Other' category
consisted exclusively of Prevention failures (16/16). The single recorded
'Detection/Response' failure (AML.CS0042) occurred within the 'Security'
category (35 Prevention, 1 Detection). The Fisher's Exact Test resulted in a
p-value of 1.0, indicating no statistically significant difference in failure
mode distribution between Security and non-Security harms. The hypothesis is
therefore rejected.

---

## Review

The experiment was successfully executed and faithfully implemented the analysis
plan. The hypothesis that failure modes differ by harm type (specifically that
non-security harms manifest as detection/response failures) was tested using
Fisher's Exact Test. The results definitively rejected the hypothesis (p=1.0).
The analysis highlighted a critical characteristic of the dataset: an
overwhelming prevalence of 'prevention_failure' modes (51 out of 52 incidents),
with the solitary 'detection_failure' occurring within the Security category
rather than the 'Other' category. The findings confirm that, within the MITRE
ATLAS corpus, incidents are consistently characterized as failures of preventive
controls regardless of whether the harm is security-related or safety/privacy-
related.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import os

# Define file path
file_name = 'step3_incident_coding.csv'
file_path = file_name

# Robust path checking based on feedback
if not os.path.exists(file_path):
    # Check parent directory just in case
    if os.path.exists(f"../{file_name}"):
        file_path = f"../{file_name}"
    else:
        print(f"Error: {file_name} not found in current or parent directory.")
        # List current dir for debugging purposes if file not found
        print("Current directory contents:", os.listdir('.'))
        exit(1)

print(f"Loading {file_path}...")
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Failed to load CSV: {e}")
    exit(1)

# Data Processing
# 1. Harm Category: Security vs Other
df['harm_normalized'] = df['harm_type'].astype(str).str.strip().str.lower()
df['Harm_Category'] = df['harm_normalized'].apply(lambda x: 'Security' if x == 'security' else 'Other')

# 2. Failure Category: Prevention vs Detection/Response
df['failure_normalized'] = df['failure_mode'].astype(str).str.strip().str.lower()

def classify_failure(val):
    if 'prevention' in val:
        return 'Prevention'
    elif 'detection' in val or 'response' in val:
        return 'Detection/Response'
    else:
        return 'Other/Unknown'

df['Failure_Category'] = df['failure_normalized'].apply(classify_failure)

# Filter out Unknown if any
df_clean = df[df['Failure_Category'] != 'Other/Unknown'].copy()

# Generate Contingency Table
# Rows: Harm (Security, Other)
# Cols: Failure (Prevention, Detection/Response)
contingency = pd.crosstab(df_clean['Harm_Category'], df_clean['Failure_Category'])

# Ensure all columns/rows exist for the test
expected_cols = ['Prevention', 'Detection/Response']
for col in expected_cols:
    if col not in contingency.columns:
        contingency[col] = 0

# Reorder columns
contingency = contingency[expected_cols]

# Ensure all rows exist
expected_rows = ['Security', 'Other']
for row in expected_rows:
    if row not in contingency.index:
        contingency.loc[row] = [0, 0]

# Reorder rows
contingency = contingency.reindex(expected_rows)

print("\n=== Contingency Table ===")
print(contingency)

# Fisher's Exact Test
odds_ratio, p_value = stats.fisher_exact(contingency)

print("\n=== Fisher's Exact Test Results ===")
print(f"Odds Ratio: {odds_ratio}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("Result: Significant association (p < 0.05)")
else:
    print("Result: No significant association (p >= 0.05)")

# Insight into the rare class
print("\n=== Detailed Breakdown of Non-Prevention Failures ===")
non_prev = df[df['Failure_Category'] == 'Detection/Response']
if not non_prev.empty:
    print(non_prev[['case_study_id', 'harm_type', 'failure_mode']])
else:
    print("No Detection/Response failures found.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading step3_incident_coding.csv...

=== Contingency Table ===
Failure_Category  Prevention  Detection/Response
Harm_Category                                   
Security                  35                   1
Other                     16                   0

=== Fisher's Exact Test Results ===
Odds Ratio: 0.0
P-value: 1.0
Result: No significant association (p >= 0.05)

=== Detailed Breakdown of Non-Prevention Failures ===
   case_study_id harm_type       failure_mode
42    AML.CS0042  security  detection_failure

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
