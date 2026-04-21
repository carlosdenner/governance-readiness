# Experiment 266: node_6_68

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_68` |
| **ID in Run** | 266 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:38:46.549748+00:00 |
| **Runtime** | 204.6s |
| **Parent** | `node_5_67` |
| **Children** | None |
| **Creation Index** | 267 |

---

## Hypothesis

> Defense vs Civilian Transparency: Defense and Security agencies are
significantly less likely to report having an 'Appeal Process' (Q65) for their
AI systems compared to Civilian agencies.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7258 (Likely True) |
| **Posterior** | 0.2527 (Likely False) |
| **Surprise** | -0.5677 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 29.0 |
| Uncertain | 0.0 |
| Maybe False | 1.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 2.0 |
| Definitely False | 58.0 |

---

## Experiment Plan

**Objective:** Compare the availability of redress/appeal mechanisms between Defense/Security agencies and Civilian agencies.

### Steps
- 1. Filter `eo13960_scored`.
- 2. Categorize `3_agency` into 'Defense/Security' (DOD, DHS, DOJ, State) vs 'Civilian' (HHS, DOE, VA, USDA, etc.).
- 3. Create binary `Has_Appeal_Process` from `65_appeal_process` (Yes vs No).
- 4. Perform a Chi-square test comparing the two agency groups.

### Deliverables
- Grouped bar chart of Appeal Process availability and statistical test results.

---

## Analysis

The experiment successfully tested the 'Defense vs Civilian Transparency'
hypothesis but found no statistical support for it.

1. **Data Classification**: The code successfully categorized 1,757 federal AI
use cases into 'Civilian' (1,372 cases) and 'Defense/Security' (385 cases)
sectors.

2. **Results**:
   - **Civilian Agencies**: Only **4.2%** (58/1372) of systems reported having
an appeal process.
   - **Defense/Security Agencies**: **4.7%** (18/385) of systems reported having
an appeal process.

3. **Statistical Significance**: The Chi-square test yielded a statistic of
0.0576 and a p-value of **0.8103**. This high p-value indicates there is no
statistically significant difference between the two groups.

4. **Conclusion**: The hypothesis that Defense agencies are *significantly less
likely* to provide appeal processes is **not supported**. Instead, the data
reveals a systemic issue across the entire federal government: over 95% of AI
systems in this dataset—regardless of agency mission—do not report having an
appeal process for affected individuals.

---

## Review

The experiment was successfully executed and the hypothesis was tested
thoroughly. The categorization logic for agencies and the binary classification
for the appeal process were implemented correctly based on the dataset's unique
values. The statistical analysis was appropriate for the categorical data.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

# [debug]
print("Starting experiment: Defense vs Civilian Transparency...")

# 1. Load the dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        print("Dataset loaded from current directory.")
    except FileNotFoundError:
        print("Error: Dataset not found in ../ or ./")
        exit(1)

# 2. Filter for eo13960_scored
# source_table column identifies the subset
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Subset 'eo13960_scored' size: {len(subset)}")

# 3. Categorize Agencies
# Inspect unique agencies to ensure robust mapping
# unique_agencies = subset['3_agency'].unique()
# print("Unique Agencies:", unique_agencies)

def categorize_agency(agency_name):
    if pd.isna(agency_name):
        return 'Unknown'
    agency_lower = str(agency_name).lower()
    defense_keywords = ['defense', 'homeland', 'justice', 'state', 'intelligence', 'army', 'navy', 'air force', 'dod', 'dhs', 'doj']
    if any(keyword in agency_lower for keyword in defense_keywords):
        return 'Defense/Security'
    else:
        return 'Civilian'

subset['Agency_Type'] = subset['3_agency'].apply(categorize_agency)
subset = subset[subset['Agency_Type'] != 'Unknown']

# 4. Create binary Has_Appeal_Process
# Inspect column 65_appeal_process
print("Unique values in '65_appeal_process':", subset['65_appeal_process'].unique())

def check_appeal(val):
    if pd.isna(val):
        return False
    # Check for affirmative 'Yes' or similar variants
    val_str = str(val).lower().strip()
    return val_str == 'yes' or val_str == 'true'

subset['Has_Appeal'] = subset['65_appeal_process'].apply(check_appeal)

# 5. Create Contingency Table
contingency_table = pd.crosstab(subset['Agency_Type'], subset['Has_Appeal'])
contingency_table.columns = ['No Appeal Process', 'Has Appeal Process']
print("\nContingency Table:")
print(contingency_table)

# 6. Statistical Test (Chi-Square)
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Calculate percentages for clearer interpretation
summary = contingency_table.copy()
summary['Total'] = summary.sum(axis=1)
summary['% with Appeal'] = (summary['Has Appeal Process'] / summary['Total']) * 100
print("\nSummary Statistics:")
print(summary)

# 7. Visualization
# Plotting % with Appeal Process
plt.figure(figsize=(8, 6))
agency_types = summary.index
percentages = summary['% with Appeal']

bars = plt.bar(agency_types, percentages, color=['skyblue', 'salmon'])
plt.title('Percentage of AI Systems with Appeal Processes by Agency Type')
plt.xlabel('Agency Category')
plt.ylabel('Percentage Reporting Appeal Process (%)')
plt.ylim(0, max(percentages) * 1.2 if max(percentages) > 0 else 10)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, 
             f'{height:.1f}%', 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Interpretation
alpha = 0.05
if p < alpha:
    print("\nResult: The difference in appeal process availability is statistically significant.")
    if summary.loc['Defense/Security', '% with Appeal'] < summary.loc['Civilian', '% with Appeal']:
        print("Hypothesis Supported: Defense/Security agencies report significantly fewer appeal processes.")
    else:
        print("Hypothesis Refuted: Defense/Security agencies report significantly MORE appeal processes.")
else:
    print("\nResult: No statistically significant difference found between agency types.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment: Defense vs Civilian Transparency...
Dataset loaded from current directory.
Subset 'eo13960_scored' size: 1757
Unique values in '65_appeal_process': <StringArray>
[                                                                                                                  nan,
                                                                                                                 'Yes',
                                                               'No – it is not operationally practical to offer this.',
                         'N/A; COTS tool used for code conversion, no individual's information is input into a model.',
 'No – Law, operational limitations, or governmentwide guidance precludes an opportunity for an individual to appeal.',
                                       'Agency CAIO has waived this minimum practice and reported such waiver to OMB.']
Length: 6, dtype: str

Contingency Table:
                  No Appeal Process  Has Appeal Process
Agency_Type                                            
Civilian                       1314                  58
Defense/Security                367                  18

Chi-square Statistic: 0.0576
P-value: 8.1032e-01

Summary Statistics:
                  No Appeal Process  Has Appeal Process  Total  % with Appeal
Agency_Type                                                                  
Civilian                       1314                  58   1372       4.227405
Defense/Security                367                  18    385       4.675325

Result: No statistically significant difference found between agency types.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** This plot compares a specific metric (percentage of AI systems with appeal processes) across two distinct categorical groups (Civilian agencies vs. Defense/Security agencies).

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Label:** "Agency Category"
    *   **Categories:** The axis displays two distinct categories: "Civilian" and "Defense/Security".
*   **Y-Axis (Vertical):**
    *   **Label:** "Percentage Reporting Appeal Process (%)"
    *   **Units:** Percentage (%).
    *   **Range:** The scale runs from 0 to 5, with increments of 1 unit. The visual range extends slightly beyond 5 to accommodate the bar height.

### 3. Data Trends
*   **Civilian Agencies:** Represented by the light blue bar on the left, this category shows a lower value of **4.2%**.
*   **Defense/Security Agencies:** Represented by the salmon-colored bar on the right, this category shows the tallest bar with a value of **4.7%**.
*   **Overall Trend:** The Defense/Security sector reports a slightly higher percentage of AI systems with appeal processes compared to the Civilian sector, though the difference is marginal (0.5%). Both categories show values below 5%.

### 4. Annotations and Legends
*   **Chart Title:** "Percentage of AI Systems with Appeal Processes by Agency Type" — Clearly defines the scope of the data.
*   **Value Labels:** Specific percentage values ("4.2%" and "4.7%") are annotated directly above each bar, providing precise data points that might be difficult to infer strictly from the Y-axis scale.
*   **Color Coding:** The plot uses distinct colors (Blue for Civilian, Salmon for Defense/Security) to visually separate the two categories, though no separate legend box is provided or necessary given the clear X-axis labels.

### 5. Statistical Insights
*   **Extremely Low Adoption Rates:** The most significant insight is the overall scarcity of appeal processes. In both sectors, less than 5% of AI systems have a reported appeal process. This implies that over 95% of AI systems in these government agencies lack a mechanism for appealing decisions made by the AI.
*   **Minimal Disparity:** While Defense/Security agencies are slightly ahead of Civilian agencies (4.7% vs 4.2%), the difference is relatively negligible. This suggests that the lack of appeal processes is a systemic issue across both types of government agencies rather than being specific to one sector.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
