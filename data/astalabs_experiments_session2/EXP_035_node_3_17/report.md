# Experiment 35: node_3_17

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_17` |
| **ID in Run** | 35 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T02:43:27.407146+00:00 |
| **Runtime** | 328.0s |
| **Parent** | `node_2_4` |
| **Children** | `node_4_37`, `node_4_46` |
| **Creation Index** | 36 |

---

## Hypothesis

> The Vendor Transparency Deficit: Government AI systems procured from commercial
vendors are significantly less likely to provide code access compared to custom-
developed systems, creating a transparency barrier.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.9945 (Definitely True) |
| **Surprise** | +0.0128 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 30.0 |
| Maybe True | 0.0 |
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

**Objective:** Compare the rate of code access availability between commercial and non-commercial AI procurements in the EO 13960 dataset.

### Steps
- 1. Filter the dataset for `eo13960_scored` records.
- 2. Create a binary group based on `10_commercial_ai` ('Yes' vs. 'No'/'Custom').
- 3. Create a binary outcome from `38_code_access` ('Yes' vs. others).
- 4. Calculate the proportion of systems with code access in each group.
- 5. Perform a Chi-Square test of independence to assess statistical significance.

### Deliverables
- Contingency table of Vendor Type vs. Code Access; Chi-Square statistic and p-value; Bar chart comparing transparency rates.

---

## Analysis

The experiment successfully tested the 'Vendor Transparency Deficit' hypothesis
by analyzing 956 valid records from the EO 13960 dataset. The code classified
systems into 'Commercial/COTS' (n=81) and 'Custom/General' (n=875) based on the
`10_commercial_ai` field, and evaluated code accessibility using the
`38_code_access` field.

The results provide strong statistical support for the hypothesis:
- **Commercial/COTS systems** had a code access rate of **44.44%**.
- **Custom/General systems** had a significantly higher code access rate of
**64.34%**.

A Chi-Square test of independence yielded a statistic of 11.71 and a **p-value
of 0.00062**, confirming that the difference is statistically significant (p <
0.05). This finding suggests that reliance on commercial AI vendors introduces a
tangible transparency barrier, with government-developed or general systems
being roughly 1.45 times more likely to allow code inspection than their
commercial counterparts.

---

## Review

The experiment successfully validated the 'Vendor Transparency Deficit'
hypothesis. Analyzing 956 valid records from the EO 13960 dataset, the study
found a statistically significant difference in code accessibility between
commercial and custom/general AI systems (Chi-Square p=0.0006). Commercial/COTS
systems (n=81) demonstrated a code access rate of only 44.4%, compared to 64.3%
for Custom/General systems (n=875). This confirms that procuring commercial AI
introduces a measurable transparency barrier, with custom-developed systems
being approximately 1.45 times more likely to allow code inspection.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 records
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"EO 13960 Records: {len(df_eo)}")

# --- Variable Construction ---

# 1. Independent Variable: System Type (Commercial vs Custom)
# Logic: '10_commercial_ai' containing 'None of the above' is treated as Custom/General.
# Any specific commercial category is treated as Commercial.

def classify_type(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().strip()
    if 'none of the above' in val_str:
        return 'Custom/General'
    else:
        return 'Commercial/COTS'

df_eo['system_type'] = df_eo['10_commercial_ai'].apply(classify_type)

# 2. Dependent Variable: Code Access
# Logic: Explicit 'Yes' vs 'No'. NaNs are excluded from the test.

def classify_access(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().strip()
    if 'yes' in val_str:
        return 'Yes'
    elif 'no' in val_str:
        return 'No'
    return np.nan

df_eo['code_access'] = df_eo['38_code_access'].apply(classify_access)

# Filter for valid records (excluding NaNs in either variable)
valid_df = df_eo.dropna(subset=['system_type', 'code_access']).copy()

print(f"\nValid Records for Analysis: {len(valid_df)}")
print("\n--- System Type Distribution (Valid Records) ---")
print(valid_df['system_type'].value_counts())

print("\n--- Code Access Distribution (Valid Records) ---")
print(valid_df['code_access'].value_counts())

# --- Statistical Analysis ---

# Contingency Table
contingency = pd.crosstab(valid_df['system_type'], valid_df['code_access'])
# Ensure columns are ordered No, Yes for consistency
if 'No' not in contingency.columns: contingency['No'] = 0
if 'Yes' not in contingency.columns: contingency['Yes'] = 0
contingency = contingency[['No', 'Yes']]

print("\n--- Contingency Table ---")
print(contingency)

# Calculate Rates
comm_row = contingency.loc['Commercial/COTS']
custom_row = contingency.loc['Custom/General']

comm_total = comm_row.sum()
custom_total = custom_row.sum()

comm_rate = comm_row['Yes'] / comm_total if comm_total > 0 else 0
custom_rate = custom_row['Yes'] / custom_total if custom_total > 0 else 0

print(f"\nCommercial Code Access Rate: {comm_rate:.2%} ({comm_row['Yes']}/{comm_total})")
print(f"Custom/General Code Access Rate: {custom_rate:.2%} ({custom_row['Yes']}/{custom_total})")

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Interpret Result
alpha = 0.05
if p < alpha:
    print("Result: Statistically Significant difference detected.")
else:
    print("Result: No statistically significant difference detected.")

# --- Visualization ---
plt.figure(figsize=(8, 6))
bar_labels = ['Commercial/COTS', 'Custom/General']
bar_values = [comm_rate, custom_rate]
colors = ['#d62728', '#1f77b4']

bars = plt.bar(bar_labels, bar_values, color=colors, alpha=0.8)

plt.ylabel('Proportion with Code Access')
plt.title('Code Access: Commercial vs Custom AI (EO 13960)')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: EO 13960 Records: 1757

Valid Records for Analysis: 956

--- System Type Distribution (Valid Records) ---
system_type
Custom/General     875
Commercial/COTS     81
Name: count, dtype: int64

--- Code Access Distribution (Valid Records) ---
code_access
Yes    599
No     357
Name: count, dtype: int64

--- Contingency Table ---
code_access       No  Yes
system_type              
Commercial/COTS   45   36
Custom/General   312  563

Commercial Code Access Rate: 44.44% (36/81)
Custom/General Code Access Rate: 64.34% (563/875)

Chi-Square Statistic: 11.7096
P-value: 6.2178e-04
Result: Statistically Significant difference detected.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** The plot compares the proportion of code access available between two specific categories of Artificial Intelligence software: "Commercial/COTS" (Commercial Off-The-Shelf) and "Custom/General."

### 2. Axes
*   **X-axis:**
    *   **Labels:** The axis features categorical labels for the two groups being compared: "Commercial/COTS" and "Custom/General."
*   **Y-axis:**
    *   **Title:** "Proportion with Code Access."
    *   **Units:** The axis uses decimal proportions (0.0 to 1.0), representing percentages from 0% to 100%.
    *   **Range:** The visible scale ranges from **0.0 to 1.1** (with the top tick mark at 1.0), encompassing the full range of possible probabilities.

### 3. Data Trends
*   **Tallest Bar:** The "Custom/General" category (blue bar) is the tallest, indicating a higher frequency of code access.
*   **Shortest Bar:** The "Commercial/COTS" category (red bar) is the shortest, indicating lower code availability.
*   **Pattern:** There is a clear disparity between the two categories, with Custom/General AI solutions providing significantly higher code accessibility compared to commercial off-the-shelf solutions.

### 4. Annotations and Legends
*   **Title:** "Code Access: Commercial vs Custom AI (EO 13960)" indicates the data relates to compliance or metrics associated with Executive Order 13960 (likely regarding AI use in government).
*   **Value Labels:**
    *   The Commercial/COTS bar is annotated with **44.4%**.
    *   The Custom/General bar is annotated with **64.3%**.
*   **Grid:** Horizontal dashed grid lines are present at intervals of 0.2 to assist in reading the bar heights against the y-axis.

### 5. Statistical Insights
*   **Magnitude of Difference:** There is a **19.9 percentage point difference** between the two categories (64.3% - 44.4%).
*   **Interpretation:** Custom or General AI software is roughly **1.45 times more likely** to have code access available compared to Commercial Off-The-Shelf (COTS) software.
*   **Contextual Insight:** This suggests that proprietary restrictions inherent to commercial software significantly limit code transparency compared to custom-built or general-purpose AI solutions within the context of the dataset (presumably federal or regulated AI inventories).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
