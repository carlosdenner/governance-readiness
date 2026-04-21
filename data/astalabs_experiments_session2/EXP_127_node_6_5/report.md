# Experiment 127: node_6_5

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_5` |
| **ID in Run** | 127 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:03:45.283800+00:00 |
| **Runtime** | 172.3s |
| **Parent** | `node_5_14` |
| **Children** | `node_7_0` |
| **Creation Index** | 128 |

---

## Hypothesis

> The 'Legacy Debt' of Governance: AI Systems initiated prior to the issuance of
EO 13960 (Year < 2021) exhibit significantly lower compliance with critical
safeguards (e.g., Impact Assessments) compared to post-2021 systems, indicating
a systemic failure to retrofit legacy infrastructure with modern governance
controls.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7581 (Likely True) |
| **Posterior** | 0.2582 (Likely False) |
| **Surprise** | -0.5998 |
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
| Maybe False | 0.0 |
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Determine if system age predicts governance compliance.

### Steps
- 1. Load the 'eo13960_scored' subset from the dataset.
- 2. Parse the '18_date_initiated' column to extract the 4-digit Year. Handle date parsing errors by coercing to NaT and dropping invalid rows.
- 3. Create a binary variable 'Era': 'Legacy' (Year < 2021) vs. 'Modern' (Year >= 2021).
- 4. Create a binary target variable from '52_impact_assessment' (Map 'Yes' to 1, others to 0).
- 5. Perform a Chi-Square test of independence between 'Era' and 'Impact Assessment Compliance'.
- 6. Calculate and compare the compliance proportions for both groups.

### Deliverables
- Contingency table; Chi-Square statistics (statistic, p-value); Bar chart comparing compliance rates by Era.

---

## Analysis

The experiment was successfully executed and yielded statistically significant
results that directly **contradict the hypothesis**.

1.  **Data Processing**: The script successfully parsed dates for 1,191 use
cases, categorizing them into 304 Legacy (<2021) and 887 Modern (2021+) systems.
2.  **Statistical Findings**: The Chi-Square test resulted in a statistic of
21.29 with a p-value of 3.95e-06, indicating a highly significant relationship
between system age and governance compliance.
3.  **Hypothesis Evaluation (Falsified)**: The hypothesis predicted that Legacy
systems would have *lower* compliance due to 'technical debt'. The data shows
the exact opposite: **Legacy systems exhibited a 10.2% compliance rate, which is
more than triple the 3.3% compliance rate of Modern systems.**

**Implications**: While compliance is critically low across the board, the
'Legacy Debt' theory is rejected. Instead, the data suggests a 'New Entrant'
problem, where systems initiated *after* the issuance of EO 13960 are failing to
implement impact assessments at significantly higher rates than older, pre-
existing systems.

---

## Review

The experiment was successfully executed and robustly tested the 'Legacy Debt'
hypothesis. The code correctly parsed 1,191 valid dates, split them into
'Legacy' (n=304) and 'Modern' (n=887) cohorts, and applied a rigorous Chi-Square
test.

**Findings:**
1.  **Hypothesis Rejection:** The results statistically significantly
**contradict** the hypothesis that legacy systems suffer from lower compliance.
The Chi-Square test yielded a statistic of 21.29 (p < 0.001), confirming a
strong relationship between system age and compliance, but in the opposite
direction predicted.
2.  **Inverse Relationship:** 'Legacy' systems (pre-2021) demonstrated a
compliance rate of **10.2%**, which is more than **triple** the rate of 'Modern'
systems (post-2021), which sits at a dismal **3.3%**.
3.  **Implication:** Rather than a 'Legacy Debt' where old systems fail to
modernize, the data suggests a 'New Entrant Failure,' where the surge of new AI
systems initiated after the Executive Order are failing to implement basic
impact assessments at significantly higher rates than older, established
systems.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

# Define file path logic based on instruction
filename = 'astalabs_discovery_all_data.csv'
filepath = f'../{filename}' if os.path.exists(f'../{filename}') else filename

print(f"Loading dataset from: {filepath}")

try:
    df = pd.read_csv(filepath, low_memory=False)
except FileNotFoundError:
    print(f"Error: {filename} not found in current or parent directory.")
    exit(1)

# Filter for EO 13960 Scored subset
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored subset shape: {df_eo.shape}")

# Columns of interest
date_col = '18_date_initiated'
impact_col = '52_impact_assessment'

# Check if columns exist
if date_col not in df_eo.columns or impact_col not in df_eo.columns:
    print(f"Error: Missing columns. Available columns: {list(df_eo.columns)}")
    exit(1)

# Preview raw date data for debugging potential parsing issues
print("\nRaw date sample (first 5 non-null):")
print(df_eo[date_col].dropna().head(5).values)

# 1. Parse Dates
# Coerce errors to NaT (Not a Time) to handle malformed strings
df_eo['parsed_date'] = pd.to_datetime(df_eo[date_col], errors='coerce')

# Drop rows with missing or unparseable dates
df_clean = df_eo.dropna(subset=['parsed_date']).copy()
print(f"\nRows with valid dates: {len(df_clean)} (dropped {len(df_eo) - len(df_clean)} rows)")

# 2. Create Era Variable
# Extract year
df_clean['year'] = df_clean['parsed_date'].dt.year

# Define Era: Legacy (< 2021) vs Modern (>= 2021)
df_clean['Era'] = df_clean['year'].apply(lambda y: 'Legacy (<2021)' if y < 2021 else 'Modern (2021+)')

print("\nDistribution by Era:")
print(df_clean['Era'].value_counts())

# 3. Process Impact Assessment (Target)
# Normalize text: 'Yes' -> 1, Anything else -> 0
# Inspect unique values first
print(f"\nUnique values in '{impact_col}': {df_clean[impact_col].unique()}")

df_clean['Compliance'] = df_clean[impact_col].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

# 4. Statistical Analysis
# Contingency Table
contingency = pd.crosstab(df_clean['Era'], df_clean['Compliance'])
contingency.columns = ['Non-Compliant', 'Compliant']
print("\nContingency Table (Era vs. Compliance):")
print(contingency)

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nChi-Square Test Results:")
print(f"Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Calculate Compliance Rates
rates = df_clean.groupby('Era')['Compliance'].mean()
print("\nCompliance Rates (Proportion 'Yes'):")
print(rates)

# 5. Visualization
plt.figure(figsize=(8, 6))
ax = rates.plot(kind='bar', color=['#d95f02', '#1b9e77'], alpha=0.8, edgecolor='black')
plt.title('Impact Assessment Compliance: Legacy vs. Modern Systems')
plt.ylabel('Compliance Rate')
plt.xlabel('System Era')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.xticks(rotation=0)

# Add value labels
for i, v in enumerate(rates):
    ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: astalabs_discovery_all_data.csv
EO 13960 Scored subset shape: (1757, 196)

Raw date sample (first 5 non-null):
<StringArray>
['7/1/2023', '8/29/2023', '8/29/2023', '9/15/2023', '1/1/2022']
Length: 5, dtype: str

Rows with valid dates: 1191 (dropped 566 rows)

Distribution by Era:
Era
Modern (2021+)    887
Legacy (<2021)    304
Name: count, dtype: int64

Unique values in '52_impact_assessment': <StringArray>
[nan, 'Planned or in-progress.', 'Yes', 'No', 'YES']
Length: 5, dtype: str

Contingency Table (Era vs. Compliance):
                Non-Compliant  Compliant
Era                                     
Legacy (<2021)            273         31
Modern (2021+)            858         29

Chi-Square Test Results:
Statistic: 21.2893
P-value: 3.9493e-06

Compliance Rates (Proportion 'Yes'):
Era
Legacy (<2021)    0.101974
Modern (2021+)    0.032694
Name: Compliance, dtype: float64


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Bar Chart (Vertical).
*   **Purpose:** The plot is designed to compare the "Compliance Rate" of Impact Assessments between two distinct categories of systems classified by their era: "Legacy" (older than 2021) and "Modern" (2021 and newer).

**2. Axes**
*   **X-axis (Horizontal):**
    *   **Title:** "System Era".
    *   **Categories:** Two categorical groups: "Legacy (<2021)" and "Modern (2021+)".
*   **Y-axis (Vertical):**
    *   **Title:** "Compliance Rate".
    *   **Value Range:** The axis ranges from **0.0 to 1.0**, representing a probability or proportion (where 1.0 equals 100%).
    *   **Increments:** Ticks are placed at intervals of 0.2.

**3. Data Trends**
*   **Legacy Systems (<2021):** Represented by an orange bar. This group shows a significantly higher compliance rate compared to the modern systems.
*   **Modern Systems (2021+):** Represented by a teal/green bar. This group shows a very low compliance rate.
*   **Trend:** There is a sharp decline in compliance rates when moving from Legacy systems to Modern systems. The compliance rate for Modern systems is less than a third of that for Legacy systems.

**4. Annotations and Legends**
*   **Bar Labels:** Specific percentage values are annotated directly above each bar for clarity:
    *   Legacy: **10.2%**
    *   Modern: **3.3%**
*   **Title:** The chart is titled "Impact Assessment Compliance: Legacy vs. Modern Systems".
*   **Grid Lines:** Horizontal dashed grid lines are included at 0.2 intervals to assist in visual estimation of the bar heights against the Y-axis.

**5. Statistical Insights**
*   **Significant Disparity:** There is a substantial gap in performance between the two eras. Legacy systems are approximately **3.1 times more compliant** than Modern systems (10.2% vs 3.3%).
*   **Overall Low Compliance:** Despite the relative difference, both groups exhibit objectively low compliance rates. Even the better-performing Legacy group only achieves a ~10% compliance rate, leaving nearly 90% of systems non-compliant. The Modern group is failing to comply in over 96% of cases.
*   **Regression in Modern Systems:** The data suggests that newer systems (implemented in or after 2021) are struggling significantly more with Impact Assessment protocols than their predecessors. This could indicate stricter recent regulations, increased system complexity, or a lack of maturity in compliance processes for newer technologies.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
