# Experiment 19: node_4_3

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_3` |
| **ID in Run** | 19 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:04:16.927960+00:00 |
| **Runtime** | 227.0s |
| **Parent** | `node_3_4` |
| **Children** | `node_5_0`, `node_5_11` |
| **Creation Index** | 20 |

---

## Hypothesis

> Incidents attributed to 'Trust Readiness' competency gaps are significantly more
likely to result in 'Privacy' harms compared to 'Integration Readiness' gaps,
which are predominantly associated with 'Security' harms.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7339 (Likely True) |
| **Posterior** | 0.3430 (Maybe False) |
| **Surprise** | -0.4537 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
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
| Uncertain | 0.0 |
| Maybe False | 75.0 |
| Definitely False | 15.0 |

---

## Experiment Plan

**Objective:** Verify if the type of competency gap (Trust vs Integration) predicts the specific category of harm using statistical association testing.

### Steps
- 1. Load the dataset 'step3_incident_coding.csv'.
- 2. Normalize the text in columns 'trust_integration_split' and 'harm_type' to lowercase.
- 3. Filter the dataframe to include only rows where 'trust_integration_split' is either 'trust-dominant' or 'integration-dominant' AND 'harm_type' is either 'security' or 'privacy'.
- 4. Create a contingency table (crosstab) with 'trust_integration_split' as rows and 'harm_type' as columns.
- 5. Perform a Fisher's Exact Test on the contingency table to calculate the Odds Ratio and P-value.
- 6. Generate a heatmap visualization of the contingency table to display the counts.

### Deliverables
- Contingency table of Harm Type by Competency Gap, Fisher's Exact Test results (Odds Ratio, P-value), and a heatmap plot.

---

## Analysis

The experiment successfully tested the association between competency gap types
(Trust vs. Integration) and harm types (Privacy vs. Security). Due to the high
prevalence of incidents classified as 'both' in the dataset (as noted in
metadata), the strict filter for 'trust-dominant' or 'integration-dominant'
incidents resulted in a small sample size of n=5.

The contingency table revealed that 100% of the analyzed incidents resulted in
'Security' harms (4 for Integration-dominant, 1 for Trust-dominant). There were
zero 'Privacy' incidents in this specific subset. Consequently, the Fisher's
Exact Test yielded a p-value of 1.0000 and an undefined (NaN) Odds Ratio,
indicating no statistical association. The hypothesis is rejected, as the data
provides no evidence that Trust Readiness gaps are more likely to result in
Privacy harms; rather, the limited data suggests Security harms are the
predominant outcome regardless of the competency gap type.

---

## Review

The experiment was faithfully implemented. The code correctly loaded the
dataset, applied the necessary filters to isolate 'trust-dominant' vs
'integration-dominant' incidents and 'security' vs 'privacy' harms, and
performed the Fisher's Exact Test. The low sample size (n=5) was an expected
artifact of the dataset's distribution (where 46/52 incidents are classified as
'both'), and the code handled this gracefully.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# [debug] Check file existence to handle path variability
filename = 'step3_incident_coding.csv'
possible_paths = [filename, f'../{filename}']
file_path = None
for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if not file_path:
    print(f"Error: Could not find {filename} in current or parent directory.")
else:
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    # Normalize text to lowercase to ensure matching
    df['trust_integration_split'] = df['trust_integration_split'].astype(str).str.lower().str.strip()
    df['harm_type'] = df['harm_type'].astype(str).str.lower().str.strip()

    print("\nUnique splits found:", df['trust_integration_split'].unique())
    print("Unique harms found:", df['harm_type'].unique())

    # Define the categories of interest
    target_splits = ['trust-dominant', 'integration-dominant']
    target_harms = ['security', 'privacy']

    # Filter the dataframe
    filtered_df = df[
        df['trust_integration_split'].isin(target_splits) &
        df['harm_type'].isin(target_harms)
    ].copy()

    print(f"\nFiltered dataset shape: {filtered_df.shape}")
    
    if filtered_df.empty:
        print("No records match the filter criteria (Trust/Integration dominant AND Security/Privacy).")
    else:
        # Create contingency table
        contingency_table = pd.crosstab(filtered_df['trust_integration_split'], filtered_df['harm_type'])
        
        # Ensure all expected columns/indexes are present for the test, filling with 0 if missing
        # We want rows: integration-dominant, trust-dominant
        # We want cols: privacy, security
        # (Order matters for odds ratio interpretation, though p-value is invariant)
        contingency_table = contingency_table.reindex(index=target_splits, columns=target_harms, fill_value=0)

        print("\nContingency Table:")
        print(contingency_table)

        # Perform Fisher's Exact Test
        # null hypothesis: the true odds ratio of the populations underlying the observations is one (no association)
        odds_ratio, p_value = stats.fisher_exact(contingency_table)

        print(f"\nFisher's Exact Test Results:")
        print(f"Odds Ratio: {odds_ratio:.4f}")
        print(f"P-value: {p_value:.4f}")

        # Visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('Harm Type by Competency Split')
        plt.ylabel('Competency Split')
        plt.xlabel('Harm Type')
        plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_incident_coding.csv

Unique splits found: <StringArray>
['both', 'trust-dominant', 'integration-dominant']
Length: 3, dtype: str
Unique harms found: <StringArray>
[             'security', 'intellectual_property',   'bias_discrimination',
           'reliability',          'supply_chain',               'privacy',
       'autonomy_misuse']
Length: 7, dtype: str

Filtered dataset shape: (5, 22)

Contingency Table:
harm_type                security  privacy
trust_integration_split                   
trust-dominant                  1        0
integration-dominant            4        0

Fisher's Exact Test Results:
Odds Ratio: nan
P-value: 1.0000


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Heatmap (specifically visualizing a contingency table or confusion matrix).
*   **Purpose:** The plot visualizes the frequency or count of occurrences at the intersection of two categorical variables: "Competency Split" and "Harm Type." It uses color intensity to represent the magnitude of the values, making it easy to spot patterns, dominance, or scarcity of data points.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Harm Type"
    *   **Labels:** The axis represents categorical data with two specific categories: **"security"** and **"privacy"**.
*   **Y-Axis:**
    *   **Title:** "Competency Split"
    *   **Labels:** The axis represents categorical data with two specific categories: **"trust-dominant"** and **"integration-dominant"**.
*   **Color Scale (Z-Axis):**
    *   **Range:** The numerical values range from **0.0 to 4.0**.
    *   **Gradient:** The color gradient shifts from a pale cream/yellow (representing 0) through teal/green (representing lower numbers like 1-2) to a deep dark blue (representing the maximum value of 4).

### 3. Data Trends
*   **High Values:** The highest value in the dataset is located at the intersection of **"integration-dominant"** and **"security"**, with a count of **4**. This is visualized by the darkest blue square.
*   **Low Values:** The lowest values are found in the **"privacy"** column. Both "trust-dominant" and "integration-dominant" splits show a count of **0** for privacy harms, indicated by the palest yellow squares.
*   **Intermediate Values:** The intersection of "trust-dominant" and "security" shows a low count of **1**, represented by a light green color.
*   **Overall Pattern:** The data indicates that "security" harms are the only type of harm recorded in this dataset, while "privacy" harms are non-existent. Furthermore, these security harms are heavily skewed toward the "integration-dominant" category.

### 4. Annotations and Legends
*   **Title:** The chart is titled **"Harm Type by Competency Split"**, clearly defining the two variables being compared.
*   **Cell Annotations:** Each cell in the 2x2 grid contains a numerical annotation (1, 0, 4, 0) indicating the exact count for that specific intersection.
*   **Color Bar Legend:** A vertical color bar on the right side provides a reference for interpreting the cell colors, with ticks marking intervals at 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, and 4.0.

### 5. Statistical Insights
*   **Dominance of Security Harms:** 100% of the recorded data points (5 total) fall under the "security" harm type. There were zero incidents of "privacy" harm.
*   **Competency Split Distribution:**
    *   **Integration-dominant:** Accounts for **80%** (4 out of 5) of the total observed harms.
    *   **Trust-dominant:** Accounts for **20%** (1 out of 5) of the total observed harms.
*   **Conclusion:** Based on this sample, there is a strong correlation between "integration-dominant" competency splits and "security" issues. Conversely, "privacy" does not appear to be a factor for either competency split in this specific dataset.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
