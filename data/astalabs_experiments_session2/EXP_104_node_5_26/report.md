# Experiment 104: node_5_26

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_26` |
| **ID in Run** | 104 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:59:48.819380+00:00 |
| **Runtime** | 342.2s |
| **Parent** | `node_4_25` |
| **Children** | `node_6_16` |
| **Creation Index** | 105 |

---

## Hypothesis

> The Vendor-Opacity Hypothesis (Revised): AI systems developed by external
'Contractors' are significantly less likely to have accessible 'Data
Documentation' compared to systems developed 'In-house' by government agencies.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.9148 (Definitely True) |
| **Surprise** | +0.1978 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 29.0 |
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

**Objective:** Re-evaluate the correlation between AI development method (Contractor vs. In-house) and data documentation availability using refined text-parsing logic that correctly identifies negative assertions (e.g., 'missing', 'not available').

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for rows where 'source_table' is 'eo13960_scored'.
- 2. Create a 'Developer_Type' column by mapping '22_dev_method': classify values containing 'contract', 'vendor', 'commercial', 'third-party' as 'Contractor'; classify 'government', 'agency', 'in-house', 'federal', 'staff' as 'In-house'. Filter the dataset to include only these two groups.
- 3. Create a 'Has_Docs' column by parsing '34_data_docs' with strict order-of-operations: 
    a. First, map values containing negative phrases ('missing', 'not available', 'no documentation', 'not reported', 'none') to 'No'.
    b. Second, map remaining values containing positive phrases ('complete', 'available', 'yes', 'link', 'http', 'partially') to 'Yes'.
    c. Map remaining or null values to 'No'.
- 4. Print a sample of 10 distinct values from '34_data_docs' alongside their mapped 'Has_Docs' status to verify the logic.
- 5. Generate a contingency table (crosstab) for 'Developer_Type' vs. 'Has_Docs'.
- 6. Calculate the percentage of systems with documentation for each group.
- 7. Perform a Chi-square test of independence and output the p-value.

### Deliverables
- 1. Data mapping validation samples.
- 2. Contingency table of Developer Type vs. Documentation Status.
- 3. Documentation rates per group.
- 4. Chi-square test results.

---

## Analysis

The experiment successfully re-tested the 'Vendor-Opacity Hypothesis' using the
corrected text-parsing logic. The data validation step confirmed that the new
mapping function correctly identifies negative assertions (e.g., 'Documentation
is missing or not available') as 'No', resolving the previous error.

Analysis of the EO 13960 dataset revealed a significant disparity in
transparency practices:
- **In-house** teams (n=439) provided data documentation in **82.7%** of cases.
- **Contractors** (n=481) provided data documentation in only **62.2%** of
cases.

The Chi-square test yielded a p-value of **7.44e-12**, overwhelmingly rejecting
the null hypothesis. These results strongly support the hypothesis that AI
systems developed by external contractors are significantly less likely to have
accessible data documentation compared to those developed internally by
government agencies.

---

## Review

The experiment was successfully executed and addressed the previous logical
flaw. The refined text-parsing logic correctly prioritized negative assertions
(e.g., mapping 'Documentation is missing or not available' to 'No'), ensuring
accurate classification. The analysis of 920 systems revealed a statistically
significant disparity: In-house teams provided data documentation in 82.7% of
cases, whereas Contractors did so in only 62.2% of cases. The Chi-square test (p
< 0.001) confirms that external development is associated with reduced
transparency in data documentation, supporting the Vendor-Opacity Hypothesis.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    print("Dataset not found. Please ensure 'astalabs_discovery_all_data.csv' is in the working directory.")
    exit(1)

# Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Mapping Logic ---

def map_developer(val):
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).lower()
    
    # Handle mixed/ambiguous cases first
    if 'both' in val_str:
        return 'Other'
        
    # Contractor keywords
    contractor_keys = ['contract', 'vendor', 'commercial', 'cots', 'third-party', 'external', 'purchase', 'industry']
    if any(k in val_str for k in contractor_keys):
        return 'Contractor'
    
    # In-house keywords
    inhouse_keys = ['government', 'agency', 'in-house', 'federal', 'staff', 'internal', 'developed by agency']
    if any(k in val_str for k in inhouse_keys):
        return 'In-house'
    
    return 'Other'

def map_documentation(val):
    if pd.isna(val):
        return 'No'
    val_str = str(val).lower()
    
    # Negative assertions (Priority 1)
    negative_keys = ['missing', 'not available', 'no documentation', 'not reported', 'none', 'unknown']
    if any(k in val_str for k in negative_keys):
        return 'No'
        
    # Positive assertions (Priority 2)
    positive_keys = ['complete', 'available', 'yes', 'link', 'http', 'partial', 'attached', 'pdf', 'doc']
    if any(k in val_str for k in positive_keys):
        return 'Yes'
        
    return 'No'

# Apply mappings
df_eo['Developer_Type'] = df_eo['22_dev_method'].apply(map_developer)
df_eo['Has_Docs'] = df_eo['34_data_docs'].apply(map_documentation)

# --- Validation ---
print("--- Data Mapping Validation (Sample) ---")
sample_docs = df_eo[['34_data_docs', 'Has_Docs']].drop_duplicates().dropna().head(10)
pd.set_option('display.max_colwidth', 100)
print(sample_docs)
print("\n")

# --- Analysis ---

# Filter for valid developer types
analysis_df = df_eo[df_eo['Developer_Type'].isin(['Contractor', 'In-house'])].copy()

# Contingency Table
contingency_table = pd.crosstab(analysis_df['Developer_Type'], analysis_df['Has_Docs'])

print("--- Contingency Table (Developer vs Documentation) ---")
print(contingency_table)

# Rates
rates = pd.crosstab(analysis_df['Developer_Type'], analysis_df['Has_Docs'], normalize='index') * 100
print("\n--- Documentation Rates (%) ---")
print(rates)

# Statistical Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\n--- Chi-square Test Results ---")
print(f"Chi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

alpha = 0.05
if p < alpha:
    print("Result: Statistically Significant. The null hypothesis is rejected.")
else:
    print("Result: Not Statistically Significant. Failed to reject the null hypothesis.")

# Visualization
plt.figure(figsize=(10, 6))
# Re-calculate rates for plotting to ensure order
plot_data = rates[['Yes', 'No']] if 'Yes' in rates.columns and 'No' in rates.columns else rates
plot_data.plot(kind='bar', stacked=True, color=['#66b3ff', '#ff9999'], ax=plt.gca())

plt.title('Data Documentation Availability by Developer Type (Refined)')
plt.xlabel('Developer Type')
plt.ylabel('Percentage')
plt.legend(title='Has Documentation', loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Data Mapping Validation (Sample) ---
                                                                                            34_data_docs Has_Docs
0    Documentation is missing or not available: No documentation exists regarding maintenance, compos...       No
43   Documentation has been partially completed: Some documentation exists (detailing the composition...      Yes
44   Documentation is complete: Documentation exists regarding the maintenance, composition, quality,...      Yes
120  Documentation is widely available: Documentation is not only complete, but is widely accessible ...      Yes
196  Documentation is complete: Documentation exists regarding the maintenance, composition, quality,...      Yes
240  Documentation has been partially completed: Some documentation exists (detailing the composition...      Yes
325          Data not reported by submitter and will be updated once additional information is collected       No
501                                                                            Documentation is complete      Yes
506                                                            Documentation is missing or not available       No
512                                                                    Documentation is widely available      Yes


--- Contingency Table (Developer vs Documentation) ---
Has_Docs         No  Yes
Developer_Type          
Contractor      182  299
In-house         76  363

--- Documentation Rates (%) ---
Has_Docs               No        Yes
Developer_Type                      
Contractor      37.837838  62.162162
In-house        17.312073  82.687927

--- Chi-square Test Results ---
Chi-square Statistic: 46.9084
P-value: 7.4383e-12
Result: Statistically Significant. The null hypothesis is rejected.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** This chart is used to compare the relative proportions of a binary categorical variable (whether documentation exists: "Yes" or "No") across two distinct groups ("Contractor" and "In-house"). By normalizing the height of the bars to 100%, it facilitates a direct comparison of percentages rather than raw quantities.

### 2. Axes
*   **X-axis:**
    *   **Title:** "Developer Type"
    *   **Labels:** Two categorical groups: "Contractor" and "In-house".
*   **Y-axis:**
    *   **Title:** "Percentage"
    *   **Range:** 0 to 100 (representing 0% to 100%).
    *   **Intervals:** Marked in increments of 20 (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Contractor:**
    *   **"Yes" (Blue):** The blue segment reaches slightly above the 60 mark, indicating approximately **62%** of contractor projects have documentation.
    *   **"No" (Red):** The red segment occupies the remaining space, representing approximately **38%** of projects lacking documentation.
*   **In-house:**
    *   **"Yes" (Blue):** The blue segment is significantly taller, reaching just above the 80 mark, indicating approximately **82-83%** of in-house projects have documentation.
    *   **"No" (Red):** The red segment is much smaller, representing approximately **17-18%** of projects lacking documentation.

### 4. Annotations and Legends
*   **Chart Title:** "Data Documentation Availability by Developer Type (Refined)".
*   **Legend:** Located on the right side of the chart, titled "Has Documentation".
    *   **Blue Square:** Represents "Yes" (documentation is available).
    *   **Light Red/Pink Square:** Represents "No" (documentation is unavailable).

### 5. Statistical Insights
*   **Prevalence of Documentation:** "In-house" developers demonstrate a significantly higher rate of data documentation availability compared to "Contractors."
*   **The Gap:** There is an approximate **20 percentage point gap** in documentation compliance between the two groups (~82% for In-house vs. ~62% for Contractors).
*   **Risk Assessment:** Nearly **40%** of the data associated with Contractors lacks documentation, whereas less than **20%** of In-house data is undocumented. This suggests that relying on external contractors may carry a higher risk of "knowledge loss" or undocumented code/data unless specific documentation requirements are enforced.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
