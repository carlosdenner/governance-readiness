# Experiment 9: node_3_0

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_0` |
| **ID in Run** | 9 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T01:24:23.550936+00:00 |
| **Runtime** | 347.3s |
| **Parent** | `node_2_4` |
| **Children** | `node_4_0`, `node_4_36` |
| **Creation Index** | 10 |

---

## Hypothesis

> The Autonomy-Harm Escalation: In real-world AI incidents, systems with 'High'
autonomy levels are associated with a significantly higher severity of harm
compared to systems with 'Low' or 'No' autonomy.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7097 (Likely True) |
| **Posterior** | 0.4066 (Maybe False) |
| **Surprise** | -0.3637 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 28.0 |
| Uncertain | 0.0 |
| Maybe False | 2.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 60.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Investigate the relationship between system autonomy and harm severity in the AIID dataset.

### Steps
- 1. Filter the dataset for `aiid_incidents`.
- 2. Clean and map `81: Autonomy Level` into ordinal categories (e.g., Low, Medium, High).
- 3. Clean and map `75: AI Harm Level` into an ordinal scale or binary 'Severe' vs 'Minor' classification.
- 4. Perform a Kruskal-Wallis H-test or Chi-Square test to determine if harm severity distribution differs by autonomy level.

### Deliverables
- Box plot or Stacked Bar chart of Harm Levels by Autonomy; Statistical test results.

---

## Analysis

The experiment successfully tested the 'Autonomy-Harm Escalation' hypothesis
using the AIID dataset. After applying the precise string mappings identified in
the debugging phase ('Autonomy1'='Low', etc.), the code successfully extracted
58 fully coded incidents where both Autonomy Level and Harm Severity were known.

The statistical analysis **rejected the hypothesis** that higher autonomy levels
are associated with higher harm severity. The Chi-Square test yielded a p-value
of 0.962 (Chi2=0.6093), indicating no significant relationship between the
variables.

Instead, the data shows that **Severe Harm** ('AI tangible harm event') is the
predominant outcome across all autonomy categories, comprising 75% of Low
autonomy incidents, 64.7% of Medium, and 71.4% of High. Interestingly, 'Medium'
autonomy systems displayed the highest relative proportion of 'Minor' issues
(17.6%), contradicting the expectation of a linear escalation of harm.

A critical limitation noted is the small sample size (n=58) relative to the
total dataset (n=1,362), as most incidents were excluded due to 'Unclear',
'None', or missing values in the specific metadata fields required for this
test.

---

## Review

The experiment successfully tested the 'Autonomy-Harm Escalation' hypothesis
using the AIID dataset. By applying the precise string mappings identified in
the debugging phase ('Autonomy1'='Low', etc.), the code successfully extracted
58 fully coded incidents where both Autonomy Level and Harm Severity were known.
The statistical analysis **rejected the hypothesis** that higher autonomy levels
are associated with higher harm severity. The Chi-Square test yielded a p-value
of 0.962 (Chi2=0.6093), indicating no significant relationship between the
variables. Instead, the data shows that **Severe Harm** ('AI tangible harm
event') is the predominant outcome across all autonomy categories, comprising
75% of Low autonomy incidents, 64.7% of Medium, and 71.4% of High.
Interestingly, 'Medium' autonomy systems displayed the highest relative
proportion of 'Minor' issues (17.6%), contradicting the expectation of a linear
escalation of harm. A critical limitation noted is the small sample size (n=58)
relative to the total dataset (n=1,362), as most incidents were excluded due to
'Unclear', 'None', or missing values in the specific metadata fields required
for this test.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        exit(1)

# Filter for AIID incidents
df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Loaded AIID Incidents: {len(df_aiid)} rows")

# Define exact mappings based on debug findings
autonomy_map = {
    'Autonomy1': 'Low',
    'Autonomy2': 'Medium',
    'Autonomy3': 'High'
}

harm_map = {
    'AI tangible harm issue': 'Minor',
    'AI tangible harm near-miss': 'Moderate',
    'AI tangible harm event': 'Severe'
}

# Apply mappings
# We use the column names identified in previous steps directly or dynamically if needed, 
# but 'Autonomy Level' and 'AI Harm Level' were confirmed.
autonomy_col = 'Autonomy Level'
harm_col = 'AI Harm Level'

if autonomy_col not in df_aiid.columns or harm_col not in df_aiid.columns:
    print("Column names mismatch. Checking columns...")
    # Fallback to dynamic search just in case
    autonomy_col = next((c for c in df_aiid.columns if 'autonomy' in c.lower() and 'level' in c.lower()), autonomy_col)
    harm_col = next((c for c in df_aiid.columns if 'harm' in c.lower() and 'level' in c.lower()), harm_col)

# Create clean columns
df_aiid['Autonomy_Clean'] = df_aiid[autonomy_col].map(autonomy_map)
df_aiid['Harm_Clean'] = df_aiid[harm_col].map(harm_map)

# Drop rows that didn't map (unclear, none, nan)
df_clean = df_aiid.dropna(subset=['Autonomy_Clean', 'Harm_Clean']).copy()

print(f"Data points after cleaning: {len(df_clean)}")
print("Autonomy Distribution:\n", df_clean['Autonomy_Clean'].value_counts())
print("Harm Distribution:\n", df_clean['Harm_Clean'].value_counts())

if len(df_clean) > 0:
    # Define Order
    autonomy_order = ['Low', 'Medium', 'High']
    harm_order = ['Minor', 'Moderate', 'Severe']
    
    # Contingency Table
    ct = pd.crosstab(df_clean['Autonomy_Clean'], df_clean['Harm_Clean'])
    ct = ct.reindex(index=autonomy_order, columns=harm_order).fillna(0)
    
    print("\nContingency Table:\n", ct)
    
    # Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    print(f"\nChi-Square Test Results:\nChi2 Statistic: {chi2:.4f}, p-value: {p:.4e}")
    
    if p < 0.05:
        print("Result: Statistically significant relationship found between Autonomy and Harm Severity.")
    else:
        print("Result: No statistically significant relationship found.")
        
    # Calculate Row Percentages for easier interpretation
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    print("\nRow Percentages (%):\n", ct_pct.round(1))
    
    # Visualization: Stacked Bar Chart
    plt.figure(figsize=(10, 6))
    # Normalize to 1.0 for stacked bar
    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    
    # Plot
    ct_norm.plot(kind='bar', stacked=True, colormap='Reds', edgecolor='black', ax=plt.gca())
    
    plt.title('Harm Severity Distribution by AI Autonomy Level')
    plt.xlabel('Autonomy Level')
    plt.ylabel('Proportion of Incidents')
    plt.legend(title='Harm Severity', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
else:
    print("No valid data available for analysis.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded AIID Incidents: 1362 rows
Data points after cleaning: 58
Autonomy Distribution:
 Autonomy_Clean
High      21
Low       20
Medium    17
Name: count, dtype: int64
Harm Distribution:
 Harm_Clean
Severe      41
Moderate     9
Minor        8
Name: count, dtype: int64

Contingency Table:
 Harm_Clean      Minor  Moderate  Severe
Autonomy_Clean                         
Low                 2         3      15
Medium              3         3      11
High                3         3      15

Chi-Square Test Results:
Chi2 Statistic: 0.6093, p-value: 9.6202e-01
Result: No statistically significant relationship found.

Row Percentages (%):
 Harm_Clean      Minor  Moderate  Severe
Autonomy_Clean                         
Low              10.0      15.0    75.0
Medium           17.6      17.6    64.7
High             14.3      14.3    71.4


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Plot.
*   **Purpose:** This plot is designed to compare the relative percentage distribution of incident severity categories (Minor, Moderate, Severe) across three different levels of AI autonomy (Low, Medium, High). It emphasizes the proportion of each severity type within each group rather than the total count of incidents.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Autonomy Level"
    *   **Categories:** Three distinct categorical groups labeled "Low," "Medium," and "High."
*   **Y-Axis:**
    *   **Title:** "Proportion of Incidents"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100%).
    *   **Increments:** Ticks are marked every 0.2 units (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
*   **Severe Incidents (Dark Red):** This category dominates all three autonomy levels. In every column, the "Severe" segment occupies the largest portion of the bar, consistently representing over 60% of the incidents.
    *   **Low Autonomy:** Visually appears to have the highest proportion of severe incidents (starting at approx. 0.25 on the y-axis, extending to 1.0).
    *   **Medium Autonomy:** Appears to have the lowest proportion of severe incidents relative to the other groups (starting higher up the axis, around 0.35).
*   **Minor Incidents (White/Cream):** This is the smallest category across all groups.
    *   **Medium Autonomy:** Displays the largest proportion of minor incidents (approx. 18%).
    *   **Low Autonomy:** Displays the smallest proportion of minor incidents (approx. 10%).
*   **Moderate Incidents (Orange):** This segment remains relatively consistent but is slightly larger in the "Medium" category compared to "Low" or "High."

### 4. Annotations and Legends
*   **Chart Title:** "Harm Severity Distribution by AI Autonomy Level" positioned at the top center.
*   **Legend:** A box located to the right of the chart titled "Harm Severity," defining the color codes:
    *   **White:** Minor
    *   **Orange:** Moderate
    *   **Dark Red:** Severe

### 5. Statistical Insights
*   **High Severity Prevalence:** The most significant insight is that regardless of the AI's autonomy level (Low, Medium, or High), the majority of recorded incidents result in "Severe" harm. This suggests that when failures occur in this dataset, they tend to be catastrophic rather than minor.
*   **Non-Linear Progression:** There is no linear correlation shown between Autonomy Level and Severity. One might expect severity to increase strictly as autonomy increases, but the data shows that "Medium" autonomy actually has a safer profile (larger proportion of Minor/Moderate incidents) than both "Low" and "High."
*   **Risk Profile:** "Low" autonomy appears to have the most dangerous proportional profile, with the smallest percentage of Minor incidents and the largest percentage of Severe incidents (roughly 75% of the bar).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
