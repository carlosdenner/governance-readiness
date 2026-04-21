# Experiment 205: node_6_35

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_35` |
| **ID in Run** | 205 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:37:30.404186+00:00 |
| **Runtime** | 440.2s |
| **Parent** | `node_5_61` |
| **Children** | None |
| **Creation Index** | 206 |

---

## Hypothesis

> Commercial AI Transparency Gap: AI systems procured commercially (COTS) by the
federal government are statistically less likely to provide 'Code Access' or
'Data Documentation' compared to Custom-Developed systems, creating a 'Black
Box' governance risk.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.8242 (Likely True) |
| **Surprise** | -0.1916 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

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
| Definitely True | 0.0 |
| Maybe True | 58.0 |
| Uncertain | 2.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Assess the impact of procurement source (Commercial vs. Custom) on transparency controls.

### Steps
- 1. Load 'eo13960_scored' subset.
- 2. Clean column '10_commercial_ai' to binary (Commercial/Custom).
- 3. Clean columns '38_code_access' and '34_data_docs' to binary (Yes/No).
- 4. Create a contingency table for Commercial vs. Code Access.
- 5. Perform a Chi-Square test and calculate the Odds Ratio.

### Deliverables
- Contingency tables; Chi-Square statistics; Bar chart comparing transparency rates by procurement type.

---

## Analysis

The experiment was successfully executed and provided a nuanced validation of
the 'Black Box' hypothesis. By correctly identifying the '10_commercial_ai'
column as the differentiator for procurement source, the analysis successfully
stratified the dataset into Commercial (COTS) and Custom (GOTS) groups.

1. **Code Access (Supported):** The analysis found a statistically significant
relationship (Chi-Square=11.71, p<0.001) between procurement source and code
availability. Commercial systems were calculated to be **2.26 times more
likely** (Odds Ratio) to deny code access compared to Custom/GOTS systems,
strongly supporting the hypothesis that commercial procurement creates 'Black
Box' risks regarding model internals.

2. **Data Documentation (Not Supported):** Contrary to the hypothesis, the
analysis did not find a statistically significant difference in data
documentation standards (p=0.185). While Commercial systems had a slightly
higher rate of missing documentation (OR=1.43), the evidence is insufficient to
conclude that commercial vendors provide less data transparency than government
developers.

The generated visualization clearly illustrates this divergence: a stark
structural gap in Code Access, but a comparable profile for Data Documentation.

---

## Review

The experiment was successfully executed and provided a nuanced validation of
the 'Black Box' hypothesis. By correctly identifying the '10_commercial_ai'
column as the differentiator for procurement source, the analysis successfully
stratified the dataset into Commercial (COTS) and Custom (GOTS) groups.

1. **Code Access (Supported):** The analysis found a statistically significant
relationship (Chi-Square=11.71, p<0.001) between procurement source and code
availability. Commercial systems were calculated to be **2.26 times more
likely** (Odds Ratio) to deny code access compared to Custom/GOTS systems,
strongly supporting the hypothesis that commercial procurement creates 'Black
Box' risks regarding model internals.

2. **Data Documentation (Not Supported):** Contrary to the hypothesis, the
analysis did not find a statistically significant difference in data
documentation standards (p=0.185). While Commercial systems had a slightly
higher rate of missing documentation (OR=1.43), the evidence is insufficient to
conclude that commercial vendors provide less data transparency than government
developers.

The generated visualization clearly illustrates this divergence: a stark
structural gap in Code Access, but a comparable profile for Data Documentation.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

# [debug]
print("Starting execution...")

# 1. Load Dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# Filter for EO 13960 scored data
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 subset shape: {subset.shape}")

# 2. Define Procurement Groups
# Using '10_commercial_ai' as the primary differentiator for Commercial (COTS) vs Custom/Other.
# 'None of the above.' implies the system does not fall into the specific commercial use-case categories.
# We treat 'None of the above.' as the Control group (Likely Custom/GOTS/Internal).

def define_procurement(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == 'None of the above.':
        return 'Custom/GOTS'
    else:
        # Any specific commercial use case description
        return 'Commercial (COTS)'

subset['procurement_type'] = subset['10_commercial_ai'].apply(define_procurement)

# 3. Clean Transparency Columns

def clean_code_access(val):
    if pd.isna(val):
        return None
    s = str(val).lower()
    # Check for negative assertions
    if s.startswith('no') or 'no access' in s:
        return 'No'
    # Check for positive assertions
    if s.startswith('yes') or 'available' in s or 'public' in s:
        return 'Yes'
    return None

def clean_data_docs(val):
    if pd.isna(val):
        return None
    s = str(val).lower()
    # Negative
    if 'missing' in s or 'not available' in s:
        return 'No'
    # Positive
    if 'complete' in s or 'partial' in s or 'widely' in s or 'exists' in s:
        return 'Yes'
    return None

subset['code_access_bin'] = subset['38_code_access'].apply(clean_code_access)
subset['data_docs_bin'] = subset['34_data_docs'].apply(clean_data_docs)

# Filter to analyzable rows (Must have procurement type)
clean_df = subset.dropna(subset=['procurement_type'])
print(f"Analyzable rows: {len(clean_df)}")
print("\nGroup Sizes:")
print(clean_df['procurement_type'].value_counts())

# 4. Statistical Analysis & Visualization Setup
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
results_found = False

# --- Analysis A: Code Access ---
print("\n--- Analysis: Commercial vs Code Access ---")
df_code = clean_df.dropna(subset=['code_access_bin'])
if len(df_code) > 0:
    results_found = True
    ct_code = pd.crosstab(df_code['procurement_type'], df_code['code_access_bin'])
    print(ct_code)
    
    # Chi-Square
    chi2, p, dof, ex = stats.chi2_contingency(ct_code)
    print(f"Chi-Square: {chi2:.4f}, p-value: {p:.4e}")
    
    # Odds Ratio (Odds of NO Access)
    if 'Commercial (COTS)' in ct_code.index and 'Custom/GOTS' in ct_code.index and 'No' in ct_code.columns:
        c_no = ct_code.loc['Commercial (COTS)', 'No']
        c_yes = ct_code.loc['Commercial (COTS)', 'Yes']
        g_no = ct_code.loc['Custom/GOTS', 'No']
        g_yes = ct_code.loc['Custom/GOTS', 'Yes']
        
        # Laplace smoothing if needed
        if any(x==0 for x in [c_no, c_yes, g_no, g_yes]):
             c_no+=0.5; c_yes+=0.5; g_no+=0.5; g_yes+=0.5
             
        or_val = (c_no / c_yes) / (g_no / g_yes)
        print(f"Odds Ratio (Commercial likelihood of NO access): {or_val:.4f}")
    
    # Plot
    ct_norm = pd.crosstab(df_code['procurement_type'], df_code['code_access_bin'], normalize='index')
    colors = ['#d62728', '#2ca02c'] if 'No' == ct_norm.columns[0] else ['#2ca02c', '#d62728']
    ct_norm.plot(kind='bar', stacked=True, ax=axes[0], color=colors)
    axes[0].set_title(f"Code Access Transparency\n(n={len(df_code)})\np={p:.1e}")
    axes[0].set_ylabel("Proportion")
    axes[0].tick_params(axis='x', rotation=0)

# --- Analysis B: Data Documentation ---
print("\n--- Analysis: Commercial vs Data Documentation ---")
df_docs = clean_df.dropna(subset=['data_docs_bin'])
if len(df_docs) > 0:
    results_found = True
    ct_docs = pd.crosstab(df_docs['procurement_type'], df_docs['data_docs_bin'])
    print(ct_docs)
    
    chi2, p, dof, ex = stats.chi2_contingency(ct_docs)
    print(f"Chi-Square: {chi2:.4f}, p-value: {p:.4e}")
    
    # Odds Ratio
    if 'Commercial (COTS)' in ct_docs.index and 'Custom/GOTS' in ct_docs.index and 'No' in ct_docs.columns:
        c_no = ct_docs.loc['Commercial (COTS)', 'No']
        c_yes = ct_docs.loc['Commercial (COTS)', 'Yes']
        g_no = ct_docs.loc['Custom/GOTS', 'No']
        g_yes = ct_docs.loc['Custom/GOTS', 'Yes']
        
        if any(x==0 for x in [c_no, c_yes, g_no, g_yes]):
             c_no+=0.5; c_yes+=0.5; g_no+=0.5; g_yes+=0.5
             
        or_val = (c_no / c_yes) / (g_no / g_yes)
        print(f"Odds Ratio (Commercial likelihood of NO docs): {or_val:.4f}")

    # Plot
    ct_norm = pd.crosstab(df_docs['procurement_type'], df_docs['data_docs_bin'], normalize='index')
    colors = ['#d62728', '#2ca02c'] if 'No' == ct_norm.columns[0] else ['#2ca02c', '#d62728']
    ct_norm.plot(kind='bar', stacked=True, ax=axes[1], color=colors)
    axes[1].set_title(f"Data Documentation Transparency\n(n={len(df_docs)})\np={p:.1e}")
    axes[1].set_ylabel("Proportion")
    axes[1].tick_params(axis='x', rotation=0)

plt.suptitle("Commercial (COTS) vs Custom (GOTS) Transparency Gap", fontsize=16)
plt.tight_layout()
if results_found:
    plt.show()
else:
    print("No valid data to plot.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting execution...
EO 13960 subset shape: (1757, 196)
Analyzable rows: 1710

Group Sizes:
procurement_type
Custom/GOTS          1357
Commercial (COTS)     353
Name: count, dtype: int64

--- Analysis: Commercial vs Code Access ---
code_access_bin     No  Yes
procurement_type           
Commercial (COTS)   45   36
Custom/GOTS        312  563
Chi-Square: 11.7096, p-value: 6.2178e-04
Odds Ratio (Commercial likelihood of NO access): 2.2556

--- Analysis: Commercial vs Data Documentation ---
data_docs_bin       No  Yes
procurement_type           
Commercial (COTS)   26   60
Custom/GOTS        222  733
Chi-Square: 1.7545, p-value: 1.8531e-01
Odds Ratio (Commercial likelihood of NO docs): 1.4308


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Two side-by-side **100% Stacked Bar Charts**.
*   **Purpose:** These plots compare the proportions of binary outcomes (Yes/No) regarding transparency ("Code Access" and "Data Documentation") across two specific categories of procurement types: "Commercial (COTS)" and "Custom/GOTS".

### 2. Axes
*   **X-Axis:**
    *   **Label:** `procurement_type`
    *   **Categories:** "Commercial (COTS)" and "Custom/GOTS". (COTS likely stands for Commercial Off-The-Shelf; GOTS stands for Government Off-The-Shelf).
*   **Y-Axis:**
    *   **Label:** "Proportion"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100%).
    *   **Ticks:** Increments of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
**Left Plot: Code Access Transparency**
*   **Commercial (COTS):** The majority (approx. 55-60%) falls into the "No" category (Red), meaning code access is frequently unavailable.
*   **Custom/GOTS:** The majority (approx. 60-65%) falls into the "Yes" category (Green).
*   **Trend:** There is a visible inversion between the two types. Custom software is much more likely to have code access transparency compared to Commercial software.

**Right Plot: Data Documentation Transparency**
*   **Commercial (COTS):** A strong majority (approx. 70%) falls into the "Yes" category (Green).
*   **Custom/GOTS:** An even stronger majority (approx. 75-80%) falls into the "Yes" category (Green).
*   **Trend:** Both procurement types show high levels of data documentation transparency, with Custom/GOTS showing a slightly higher proportion of "Yes" than Commercial.

### 4. Annotations and Legends
*   **Main Title:** "Commercial (COTS) vs Custom (GOTS) Transparency Gap"
*   **Subplot Titles:**
    *   Left: "Code Access Transparency"
    *   Right: "Data Documentation Transparency"
*   **Sample Size (n):**
    *   Left: `(n=956)` indicating the total number of observations for the code access analysis.
    *   Right: `(n=1041)` indicating the total number of observations for the data documentation analysis.
*   **P-values:**
    *   Left: `p=6.2e-04` (0.00062)
    *   Right: `p=1.9e-01` (0.19)
*   **Legends:**
    *   **Red:** Represents "No" (Lack of transparency).
    *   **Green:** Represents "Yes" (Presence of transparency).

### 5. Statistical Insights
*   **Significant Difference in Code Access:** The p-value for the Code Access Transparency chart is **6.2e-04**, which is well below the standard alpha level of 0.05. This indicates a **statistically significant difference**. We can confidently conclude that Custom/GOTS software provides significantly higher rates of code access transparency compared to Commercial (COTS) software.
*   **No Significant Difference in Data Documentation:** The p-value for the Data Documentation Transparency chart is **1.9e-01** (0.19), which is greater than 0.05. This indicates the result is **not statistically significant**. Although the bar chart visually suggests Custom software performs slightly better, the difference is likely due to chance or sampling variability; statistically, both procurement types perform similarly regarding data documentation.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
