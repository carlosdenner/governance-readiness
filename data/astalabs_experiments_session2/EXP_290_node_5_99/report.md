# Experiment 290: node_5_99

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_99` |
| **ID in Run** | 290 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T14:53:33.345551+00:00 |
| **Runtime** | 239.5s |
| **Parent** | `node_4_50` |
| **Children** | None |
| **Creation Index** | 291 |

---

## Hypothesis

> The Safety-Governance Gap: AI systems deployed in safety-critical agencies
(Health, Energy, Transportation, Homeland Security) do not demonstrate a
statistically higher implementation rate of 'Independent Evaluations' compared
to systems in administrative agencies, indicating a failure to scale governance
with risk.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7581 (Likely True) |
| **Posterior** | 0.9176 (Definitely True) |
| **Surprise** | +0.1914 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
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
| Definitely True | 60.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Test if safety-critical sectors apply stricter governance controls.

### Steps
- 1. Load 'eo13960_scored'.
- 2. Define 'Safety_Critical' group based on '3_agency' or '8_topic_area' (Health, Energy, Transportation, Homeland Security).
- 3. Define 'Administrative' group (Education, Commerce, Treasury, etc.).
- 4. Convert '55_independent_eval' to binary (Yes/No).
- 5. Compare the proportion of Independent Evaluations between the two groups using a Z-test for proportions or Chi-square.

### Deliverables
- Proportions of evaluations for both groups; Statistical test results.

---

## Analysis

The experiment successfully tested the 'Safety-Governance Gap' hypothesis by
analyzing the implementation of independent evaluations across different agency
types. After correcting the data parsing logic to properly identify affirmative
responses in the '55_independent_eval' column, the study classified agencies
into Administrative (N=111) and Safety-Critical (N=539) groups. The results
revealed a counter-intuitive trend: Safety-Critical agencies, despite their
high-stakes domain, demonstrated a lower raw rate of independent evaluation
(4.82%) compared to Administrative agencies (6.31%). A Chi-square test yielded a
p-value of 0.6814, indicating that this difference is not statistically
significant. This finding strongly supports the hypothesis that governance
controls are not scaling with risk, as safety-critical deployments do not show a
statistically higher rate of independent oversight than standard administrative
functions, and overall adoption remains critically low (<7%) across the federal
government.

---

## Review

The experiment was successfully executed and robustly tested the 'Safety-
Governance Gap' hypothesis. The code correctly loaded the dataset, filtered for
the relevant source ('eo13960_scored'), and implemented the classification logic
to distinguish between 'Safety-Critical' (n=539) and 'Administrative' (n=111)
agencies. Crucially, the programmer corrected the initial parsing error for the
'Independent Evaluation' variable, allowing for a valid statistical comparison.
The analysis revealed that Safety-Critical agencies actually had a *lower* rate
of independent evaluation (4.82%) compared to Administrative agencies (6.31%),
though the difference was not statistically significant (p=0.68). This finding
strongly supports the hypothesis that governance mechanisms are not scaling with
risk, as high-stakes agencies do not demonstrate the expected higher tier of
oversight.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# [debug]
print("Starting experiment: Safety-Governance Gap (Attempt 2)")

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        print("Dataset loaded successfully (local).")
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# Filter for relevant source table
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Filtered for eo13960_scored: {len(df_eo)} rows")

# Normalize column names for easier access if needed (using existing names)
agency_col = '3_agency'
eval_col = '55_independent_eval'

# Define Groups based on keywords in agency name
def classify_agency(agency_name):
    if pd.isna(agency_name):
        return None
    agency_upper = str(agency_name).upper()
    
    safety_keywords = ['HEALTH', 'ENERGY', 'TRANSPORTATION', 'HOMELAND']
    admin_keywords = ['EDUCATION', 'COMMERCE', 'TREASURY']
    
    for kw in safety_keywords:
        if kw in agency_upper:
            return 'Safety-Critical'
    
    for kw in admin_keywords:
        if kw in agency_upper:
            return 'Administrative'
            
    return 'Other'

df_eo['group'] = df_eo[agency_col].apply(classify_agency)

# Filter only for the two groups of interest
df_analysis = df_eo[df_eo['group'].isin(['Safety-Critical', 'Administrative'])].copy()
print(f"Rows after group filtering: {len(df_analysis)}")

# Process the target variable '55_independent_eval'
# Updated logic: Check if string starts with 'Yes'
def parse_eval(val):
    if pd.isna(val):
        return 0
    # Normalize string
    s = str(val).strip().upper()
    if s.startswith('YES'):
        return 1
    return 0

df_analysis['has_eval'] = df_analysis[eval_col].apply(parse_eval)

# Verify parsing
print("\nValue counts for has_eval:")
print(df_analysis['has_eval'].value_counts())

# Contingency Table
contingency_table = pd.crosstab(df_analysis['group'], df_analysis['has_eval'])
print("\nContingency Table (Raw):")
print(contingency_table)

# Handle column naming safely based on what is present
# has_eval can be 0, 1, or both.
mapping = {0: 'No Eval', 1: 'Has Eval'}
contingency_table.columns = [mapping.get(c, c) for c in contingency_table.columns]

print("\nContingency Table (Labeled):")
print(contingency_table)

# Calculate proportions
props = df_analysis.groupby('group')['has_eval'].agg(['mean', 'count', 'sum'])
props.columns = ['Proportion', 'Total Count', 'Eval Count']
print("\nProportions:")
print(props)

# Statistical Test (Chi-Square)
# Only run if we have data in both groups and variance in the dependent variable
if contingency_table.shape == (2, 2):
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"p-value: {p:.4f}")
    
    alpha = 0.05
    if p < alpha:
        print("Result: Statistically significant difference found.")
    else:
        print("Result: No statistically significant difference found.")
        
    # Visualization
    plt.figure(figsize=(8, 6))
    groups = props.index
    means = props['Proportion']
    
    # Plot
    bars = plt.bar(groups, means, color=['skyblue', 'salmon'])
    plt.title('Proportion of AI Systems with Independent Evaluations')
    plt.ylabel('Proportion')
    plt.ylim(0, max(means.max() * 1.2, 0.1)) # Scale y-axis
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.2%}',
                 ha='center', va='bottom')
                 
    plt.show()
else:
    print("Insufficient data structure for Chi-Square test (expected 2x2 table).")
    print(f"Current shape: {contingency_table.shape}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment: Safety-Governance Gap (Attempt 2)
Dataset loaded successfully (local).
Filtered for eo13960_scored: 1757 rows
Rows after group filtering: 650

Value counts for has_eval:
has_eval
0    617
1     33
Name: count, dtype: int64

Contingency Table (Raw):
has_eval           0   1
group                   
Administrative   104   7
Safety-Critical  513  26

Contingency Table (Labeled):
                 No Eval  Has Eval
group                             
Administrative       104         7
Safety-Critical      513        26

Proportions:
                 Proportion  Total Count  Eval Count
group                                               
Administrative     0.063063          111           7
Safety-Critical    0.048237          539          26

Chi-Square Test Results:
Chi2 Statistic: 0.1685
p-value: 0.6814
Result: No statistically significant difference found.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot (or Bar Chart).
*   **Purpose:** The plot is designed to compare a numerical variable (the proportion of AI systems undergoing independent evaluations) across two distinct categorical groups ("Administrative" and "Safety-Critical").

### 2. Axes
*   **X-Axis:**
    *   **Labels:** The axis displays two categories: **"Administrative"** and **"Safety-Critical"**.
    *   **Value Range:** N/A (Categorical data).
*   **Y-Axis:**
    *   **Title:** **"Proportion"**.
    *   **Units:** The scale is in decimals representing percentages.
    *   **Value Range:** The axis ranges from **0.00 to 0.10** (representing 0% to 10%).

### 3. Data Trends
*   **Tallest Bar:** The **"Administrative"** bar (colored light blue) is the tallest, indicating a higher proportion of independent evaluations.
*   **Shortest Bar:** The **"Safety-Critical"** bar (colored salmon/red) is the shortest.
*   **Pattern:** There is a visible disparity where administrative AI systems are more frequently subject to independent evaluations compared to safety-critical systems.

### 4. Annotations and Legends
*   **Annotations:** Both bars have specific percentage values written directly above them to provide precise data points:
    *   Administrative: **6.31%**
    *   Safety-Critical: **4.82%**
*   **Title:** The chart is titled **"Proportion of AI Systems with Independent Evaluations"**, clearly stating the subject matter.
*   **Color Coding:** While there is no separate legend box, the bars are color-coded (blue for Administrative, red for Safety-Critical) to visually distinguish the categories.

### 5. Statistical Insights
*   **Low Overall Adoption:** The most striking insight is that independent evaluation appears to be extremely rare across the board. In both categories, the proportion is well below 10% (specifically under 7%), suggesting that the vast majority of AI systems in this dataset do not undergo independent evaluation.
*   **Counter-Intuitive Finding:** One might expect "Safety-Critical" systems (which likely involve higher risks to health, safety, or infrastructure) to undergo rigorous independent testing more often than "Administrative" systems. However, the data shows the opposite: Administrative systems are evaluated independently at a rate approximately **1.49 percentage points higher** than Safety-Critical systems (6.31% vs 4.82%).
*   **Relative Difference:** The proportion of independent evaluations for Administrative systems is roughly **31% higher** relative to the proportion for Safety-Critical systems.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
