# Experiment 31: node_3_13

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_13` |
| **ID in Run** | 31 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:30:24.449015+00:00 |
| **Runtime** | 247.0s |
| **Parent** | `node_2_2` |
| **Children** | `node_4_12`, `node_4_13` |
| **Creation Index** | 32 |

---

## Hypothesis

> Incidents with distinct 'Trust' or 'Integration' dominance show different
failure mode distributions (Prevention vs. Detection/Response), indicating a
structural dependency between the nature of the competency gap and the incident
lifecycle stage.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.4876 (Uncertain) |
| **Surprise** | -0.2952 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
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
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 54.0 |
| Maybe False | 36.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Investigate if the nature of the competency gap (Trust-dominant vs. Integration-dominant) predicts how the failure manifests (Prevention vs. Detection/Response) by analyzing the subset of incidents with distinct dominances.

### Steps
- 1. Load the dataset 'step3_incident_coding.csv'.
- 2. Filter the DataFrame to exclude rows where 'trust_integration_split' is 'Both' (case-insensitive).
- 3. Create a new binary column 'failure_category' where 'failure_mode' containing 'prevention' maps to 'Prevention' and all others map to 'Post-Incident'.
- 4. Generate a contingency table of 'trust_integration_split' vs. 'failure_category'.
- 5. If the contingency table has at least 2 rows and 2 columns, perform a Fisher's Exact Test (or Chi-square if appropriate) to test for independence.
- 6. If the table is smaller (e.g., due to zero variance in failure modes), print the table and explicitly state that statistical testing is not possible due to insufficient variance.

### Deliverables
- 1. Value counts of 'trust_integration_split' and 'failure_mode' for the full dataset.
- 2. The contingency table for the filtered subset.
- 3. Fisher's Exact Test p-value (if applicable) or a statement explaining why the test could not be run.

---

## Analysis

The experiment successfully loaded the 'step3_incident_coding.csv' dataset
(n=52) and filtered for incidents with distinct competency dominances (excluding
'both'). The analysis revealed an extreme lack of variance in the data:

1.  **Dominance Split:** 88% of incidents (46/52) were classified as 'both',
leaving only 6 incidents with distinct dominances (4 'integration-dominant', 2
'trust-dominant').
2.  **Failure Modes:** 98% of all incidents (51/52) were 'prevention_failure',
with only a single 'detection_failure' in the entire dataset (which occurred in
the 'both' category).
3.  **Subset Analysis:** Within the distinct dominance subset (n=6), 100% of
incidents were prevention failures.

Consequently, the contingency table was 2x1 (Integration: 4, Trust: 2; both
Prevention), rendering statistical testing (Fisher's Exact / Chi-Square)
impossible due to zero variance in the dependent variable. The hypothesis could
not be tested, but the data strongly suggests that current adversarial AI case
studies (MITRE ATLAS) are overwhelmingly characterized by prevention failures
where both trust and integration competencies are implicated simultaneously.

---

## Review

The experiment was faithfully implemented. The code successfully loaded the
'step3_incident_coding.csv' dataset, filtered for distinct dominances, and
attempted to generate a contingency table as planned.

**Findings:**
1.  **Data Skew:** The dataset exhibits extreme skew, with 88% (46/52) of
incidents involving 'Both' trust and integration gaps, and 98% (51/52)
classified as 'Prevention' failures.
2.  **Subset Analysis:** The subset of incidents with distinct dominances (n=6)
consisted entirely of 'Prevention' failures (4 Integration-dominant, 2 Trust-
dominant).
3.  **Hypothesis Test:** Due to zero variance in the dependent variable (Failure
Mode) within the subset, the resulting 2x1 contingency table precluded
statistical testing (Fisher's Exact/Chi-Square).

**Conclusion:** The hypothesis could not be statistically validated. The data
suggests that current adversarial AI case studies (MITRE ATLAS) predominantly
represent multifaceted prevention failures where trust and integration issues
co-occur, rather than distinct failure modes driven by isolated competency gaps.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import os

# --- robust file loading ---
filename = 'step3_incident_coding.csv'
file_paths = [filename, f'../{filename}']

df = None
for path in file_paths:
    if os.path.exists(path):
        print(f"Found dataset at: {path}")
        df = pd.read_csv(path)
        break

if df is None:
    print(f"Error: Could not find {filename} in current or parent directory.")
    # List current directory contents for debugging if needed, but per instructions, we exit.
    exit(1)

# --- Data Preparation ---
print("\n=== Dataset Statistics ===")
print(f"Total rows: {len(df)}")
print("\nDistribution of 'trust_integration_split':")
print(df['trust_integration_split'].value_counts())
print("\nDistribution of 'failure_mode':")
print(df['failure_mode'].value_counts())

# Filter out 'Both' to focus on distinct dominance
# We normalize to lowercase to be safe, though metadata suggests 'Both' is capitalized
subset = df[~df['trust_integration_split'].str.lower().eq('both')].copy()
print(f"\nSubset size (excluding 'Both'): {len(subset)}")

if len(subset) == 0:
    print("No records found with distinct Trust or Integration dominance.")
    exit(0)

# Map failure modes
# Prevention vs Post-Incident (Detection/Response)
subset['failure_category'] = subset['failure_mode'].apply(
    lambda x: 'Prevention' if 'prevention' in str(x).lower() else 'Post-Incident'
)

print("\nSubset Failure Category Distribution:")
print(subset['failure_category'].value_counts())

# --- Contingency Table ---
contingency_table = pd.crosstab(subset['trust_integration_split'], subset['failure_category'])
print("\n=== Contingency Table (Split vs Failure Category) ===")
print(contingency_table)

# --- Statistical Test ---
# We need at least 2 rows and 2 columns to test independence between variables.
# If all rows fall into one column (e.g., all Prevention), we cannot run Chi-Square/Fisher.

rows, cols = contingency_table.shape
if rows >= 2 and cols >= 2:
    if rows == 2 and cols == 2:
        odds_ratio, p_value = stats.fisher_exact(contingency_table)
        print(f"\nFisher's Exact Test Results:")
        print(f"Odds Ratio: {odds_ratio}")
        print(f"P-value: {p_value}")
    else:
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\nChi-Square Test Results:")
        print(f"Chi2 Stat: {chi2}")
        print(f"P-value: {p}")
else:
    print("\nSkipping statistical test: Contingency table dimensions are insufficient (need at least 2x2).")
    print("Observed dimensions:", (rows, cols))
    print("This indicates a lack of variance in one or both variables within the subset.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Found dataset at: step3_incident_coding.csv

=== Dataset Statistics ===
Total rows: 52

Distribution of 'trust_integration_split':
trust_integration_split
both                    46
integration-dominant     4
trust-dominant           2
Name: count, dtype: int64

Distribution of 'failure_mode':
failure_mode
prevention_failure    51
detection_failure      1
Name: count, dtype: int64

Subset size (excluding 'Both'): 6

Subset Failure Category Distribution:
failure_category
Prevention    6
Name: count, dtype: int64

=== Contingency Table (Split vs Failure Category) ===
failure_category         Prevention
trust_integration_split            
integration-dominant              4
trust-dominant                    2

Skipping statistical test: Contingency table dimensions are insufficient (need at least 2x2).
Observed dimensions: (2, 1)
This indicates a lack of variance in one or both variables within the subset.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
