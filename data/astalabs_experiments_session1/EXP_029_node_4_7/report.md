# Experiment 29: node_4_7

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_7` |
| **ID in Run** | 29 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:24:50.671116+00:00 |
| **Runtime** | 331.9s |
| **Parent** | `node_3_0` |
| **Children** | `node_5_3`, `node_5_26` |
| **Creation Index** | 30 |

---

## Hypothesis

> Incidents linked to 'Trust Readiness' deficiencies are significantly more likely
to manifest as 'Prevention Failures', whereas 'Integration Readiness'
deficiencies are more likely to manifest as 'Detection' or 'Response' failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.3698 (Maybe False) |
| **Surprise** | -0.4319 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
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
| Uncertain | 0.0 |
| Maybe False | 87.0 |
| Definitely False | 3.0 |

---

## Experiment Plan

**Objective:** To statistically validate whether governance-related (Trust) gaps correlate with prevention failures while engineering (Integration) gaps correlate with detection/response failures.

### Steps
- 1. Load the dataset 'step3_incident_coding.csv' from the current directory.
- 2. Filter the dataset to include only incidents where 'trust_integration_split' is either 'Trust-dominant' or 'Integration-dominant'.
- 3. Create a binary failure mode variable: map 'Prevention' to 'Prevention' and map 'Detection'/'Response' to 'Detection/Response'.
- 4. Generate a contingency table of 'trust_integration_split' (rows) vs. the binary failure mode (columns).
- 5. Perform Fisher's Exact Test on the contingency table to calculate the p-value and odds ratio.
- 6. Visualize the relationship using a stacked bar chart showing the proportion of failure modes for each dominance type.

### Deliverables
- 1. Contingency table of Readiness Split vs. Failure Mode.
- 2. Fisher's Exact Test results (p-value, odds ratio).
- 3. Stacked bar chart visualization.

---

## Analysis

The experiment successfully analyzed the relationship between readiness
deficiencies and failure modes using the 6 incidents where a dominant gap was
identified (4 'Integration-dominant', 2 'Trust-dominant').

**Findings:**
- **Integration-dominant:** 4 incidents, 100% classified as 'Prevention'
failures.
- **Trust-dominant:** 2 incidents, 100% classified as 'Prevention' failures.

**Hypothesis Evaluation:**
The hypothesis is **REJECTED**. It predicted that 'Integration Readiness'
deficiencies would manifest as 'Detection' or 'Response' failures. However, the
data shows that Integration gaps in this subset exclusively resulted in
'Prevention' failures (4/4).

**Statistical Significance:**
Fisher's Exact Test yielded a p-value of 1.0. This is because there was zero
variance in the dependent variable (Failure Mode)—every single incident in the
subset was a Prevention failure.

**Conclusion:**
Both governance (Trust) and engineering (Integration) gaps appear to manifest
primarily as prevention failures in this specific subset of data, contradicting
the expectation that engineering gaps would correlate with detection/response
issues.

---

## Review

Hypothesis: Incidents linked to 'Trust Readiness' deficiencies are significantly
more likely to manifest as 'Prevention Failures', whereas 'Integration
Readiness' deficiencies are more likely to manifest as 'Detection' or 'Response'
failures.

Experiment Results: The experiment was faithfully implemented using the
'step3_incident_coding.csv' dataset. The analysis filtered for incidents with a
dominant readiness gap, resulting in a subset of 6 incidents (2 Trust-dominant,
4 Integration-dominant). The failure modes were mapped to a binary
classification (Prevention vs. Detection/Response) and analyzed using a
contingency table and Fisher's Exact Test.

Findings:
1. Hypothesis Rejection: The hypothesis is rejected. While Trust-dominant
incidents manifested as Prevention Failures (consistent with the hypothesis),
Integration-dominant incidents also exclusively manifested as Prevention
Failures (4/4), contradicting the prediction that they would lead to
Detection/Response failures.
2. Statistical Evidence: Fisher's Exact Test yielded a p-value of 1.0,
reflecting identical distributions (100% Prevention) in both groups.
3. Limitations: The validity of the statistical test is constrained by the small
sample size (n=6), as 88% of the dataset (46/52 incidents) possessed 'Both' gap
types and were excluded to test the specific dominance hypothesis.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
import os

# Handle file loading robustly
filename = 'step3_incident_coding.csv'
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    print(f"File {filename} not found.")
    exit(1)

df = pd.read_csv(filepath)
print(f"Loaded dataset with {len(df)} records.")

# 1. Clean and Inspect Columns
print("\n--- unique values in trust_integration_split ---")
print(df['trust_integration_split'].unique())

# Normalize column for filtering
df['split_norm'] = df['trust_integration_split'].astype(str).str.strip().str.lower()

# 2. Filter out 'both' to isolate dominant factors
# We look for rows that do NOT contain 'both' but ARE valid
# Based on metadata, we expect roughly 6 records here
subset = df[~df['split_norm'].str.contains('both')].copy()

# 3. Categorize into Trust vs Integration
def classify_split(val):
    if 'trust' in val:
        return 'Trust'
    elif 'integration' in val:
        return 'Integration'
    return None

subset['group'] = subset['split_norm'].apply(classify_split)
subset = subset.dropna(subset=['group'])

print(f"\nFiltered subset size: {len(subset)}")
print(subset['group'].value_counts())

# 4. Categorize Failure Mode
def classify_failure(val):
    val_str = str(val).lower()
    if 'prevention' in val_str:
        return 'Prevention'
    else:
        return 'Detection/Response'

subset['failure_type'] = subset['failure_mode'].apply(classify_failure)

# 5. Contingency Table
ct = pd.crosstab(subset['group'], subset['failure_type'])
print("\n--- Contingency Table ---")
print(ct)

# Ensure 2x2 shape for Fisher Test
expected_rows = ['Integration', 'Trust']
expected_cols = ['Detection/Response', 'Prevention']
ct_full = ct.reindex(index=expected_rows, columns=expected_cols, fill_value=0)
print("\n--- Full Table for Stats ---")
print(ct_full)

# 6. Statistical Test (Fisher's Exact)
if ct_full.values.sum() > 0:
    odds_ratio, p_value = fisher_exact(ct_full)
    print(f"\nFisher's Exact Test Results:")
    print(f"Odds Ratio: {odds_ratio}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Conclusion: Statistically significant association found.")
    else:
        print("Conclusion: No statistically significant association (likely due to small sample size).")
else:
    print("Not enough data for statistical testing.")

# 7. Visualization
if not ct_full.empty and ct_full.values.sum() > 0:
    plt.figure(figsize=(8, 6))
    ct_full.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], ax=plt.gca())
    plt.title('Failure Mode by Readiness Dominance')
    plt.xlabel('Dominant Readiness Gap')
    plt.ylabel('Incident Count')
    plt.xticks(rotation=0)
    plt.legend(title='Failure Mode')
    plt.tight_layout()
    plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded dataset with 52 records.

--- unique values in trust_integration_split ---
<StringArray>
['both', 'trust-dominant', 'integration-dominant']
Length: 3, dtype: str

Filtered subset size: 6
group
Integration    4
Trust          2
Name: count, dtype: int64

--- Contingency Table ---
failure_type  Prevention
group                   
Integration            4
Trust                  2

--- Full Table for Stats ---
failure_type  Detection/Response  Prevention
group                                       
Integration                    0           4
Trust                          0           2

Fisher's Exact Test Results:
Odds Ratio: nan
P-value: 1.0000
Conclusion: No statistically significant association (likely due to small sample size).


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Vertical Bar Plot (specifically designed as a grouped bar chart, though only one group has visible data).
*   **Purpose:** The plot compares the count of incidents categorized by "Failure Mode" (Prevention vs. Detection/Response) across different "Dominant Readiness Gaps" (Integration and Trust).

**2. Axes**
*   **X-axis:**
    *   **Title:** "Dominant Readiness Gap"
    *   **Labels:** The axis features categorical variables: "Integration" and "Trust".
*   **Y-axis:**
    *   **Title:** "Incident Count"
    *   **Range:** The values range from 0.0 to a maximum tick of 4.0, with intervals of 0.5.

**3. Data Trends**
*   **Dominant Category:** The "Integration" category represents the highest number of incidents.
*   **Comparison:**
    *   **Integration:** Shows a count of **4.0** incidents related to "Prevention".
    *   **Trust:** Shows a count of **2.0** incidents related to "Prevention".
*   **Missing/Zero Values:** There are no visible bars for the "Detection/Response" failure mode (Pink/Red) for either category, indicating a count of zero for this specific failure mode in this dataset.
*   **Pattern:** The chart indicates that "Prevention" failures are twice as common in scenarios with an "Integration" readiness gap compared to those with a "Trust" readiness gap.

**4. Annotations and Legends**
*   **Chart Title:** "Failure Mode by Readiness Dominance" is displayed at the top center.
*   **Legend:** Located in the top right corner with the title "Failure Mode".
    *   **Pink/Light Red:** Represents "Detection/Response".
    *   **Blue/Light Blue:** Represents "Prevention".

**5. Statistical Insights**
*   **Exclusive Failure Mode:** Within the scope of this data, 100% of the recorded incidents are classified as "Prevention" failures. There were no "Detection/Response" incidents recorded for these specific readiness gaps.
*   **Risk Area:** The "Integration" readiness gap appears to be a more significant driver of Prevention failures (4 incidents) than the "Trust" gap (2 incidents). This suggests that efforts to improve system readiness should prioritize addressing Integration gaps to reduce the overall incident count.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
