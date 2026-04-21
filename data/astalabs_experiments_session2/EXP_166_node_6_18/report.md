# Experiment 166: node_6_18

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_18` |
| **ID in Run** | 166 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:52:05.276795+00:00 |
| **Runtime** | 288.9s |
| **Parent** | `node_5_40` |
| **Children** | `node_7_7` |
| **Creation Index** | 167 |

---

## Hypothesis

> Defense and Intelligence agencies (DOD, DOJ, DHS) have a significantly lower
compliance rate for 'AI Public Notice' compared to Civilian Service agencies
(HHS, VA, Education), validating the tension between transparency and secrecy.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7742 (Likely True) |
| **Posterior** | 0.2637 (Likely False) |
| **Surprise** | -0.6125 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 4.0 |
| Maybe True | 26.0 |
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

**Objective:** Quantify the transparency trade-off in national security vs. public service agencies.

### Steps
- 1. Filter 'eo13960_scored'.
- 2. Categorize '3_agency' into 'Defense/Intel' (DOD, DOJ, DHS, State) and 'Civilian/Service' (HHS, VA, DOT, DOE, HUD, Education).
- 3. Create binary variable from '59_ai_notice' (Yes=1, No=0).
- 4. Perform a Chi-square test comparing Public Notice rates between the two agency categories.

### Deliverables
- Contingency table and Chi-square results showing the disparity in public transparency.

---

## Analysis

The experiment successfully tested the 'Transparency Trade-off' hypothesis on
the 'eo13960_scored' dataset.

1. **Data Processing**: After addressing the data sparsity issues identified in
the debug phase, the final analysis included 57 valid use cases (Civilian: 43,
Defense: 14). The majority of the 1,757 original rows were excluded because the
'59_ai_notice' field was either missing (NaN) or marked as 'N/A' (not
applicable), indicating that public notice is only relevant or reported for a
small fraction of the inventory.

2. **Descriptive Statistics**: Contrary to the hypothesis, Defense/Intel
agencies demonstrated a high compliance rate of **92.9%** (13/14), which is
effectively identical to the Civilian/Service agencies' rate of **93.0%**
(40/43).

3. **Statistical Testing**: Given the small sample size, a Fisher's Exact Test
was performed. The resulting **p-value was 1.0**, indicating absolutely no
statistical difference between the two groups.

4. **Conclusion**: We **reject the hypothesis**. The data does not support the
claim that Defense agencies are less transparent regarding AI Public Notice than
Civilian agencies. Among the use cases where notice was applicable and recorded,
compliance was near-universal across both sectors.

---

## Review

The experiment successfully tested the 'Transparency Trade-off' hypothesis on
the 'eo13960_scored' dataset.

1. **Data Processing**: After addressing the data sparsity issues identified in
the debug phase, the final analysis included 57 valid use cases (Civilian: 43,
Defense: 14). The majority of the 1,757 original rows were excluded because the
'59_ai_notice' field was either missing (NaN) or marked as 'N/A' (not
applicable), indicating that public notice is only relevant or reported for a
small fraction of the inventory.

2. **Descriptive Statistics**: Contrary to the hypothesis, Defense/Intel
agencies demonstrated a high compliance rate of **92.9%** (13/14), which is
effectively identical to the Civilian/Service agencies' rate of **93.0%**
(40/43).

3. **Statistical Testing**: Given the small sample size, a Fisher's Exact Test
was performed. The resulting **p-value was 1.0**, indicating absolutely no
statistical difference between the two groups.

4. **Conclusion**: We **reject the hypothesis**. The data does not support the
claim that Defense agencies are less transparent regarding AI Public Notice than
Civilian agencies. Among the use cases where notice was applicable and recorded,
compliance was near-universal across both sectors.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.pyplot as plt

# Load the dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# Filter for EO 13960 Scored data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# --- 1. Agency Categorization ---
def categorize_agency(row):
    agency = str(row['3_agency']).upper()
    abr = str(row['3_abr']).upper()
    
    # Defense / Intel / Security
    # Target: DOD, DOJ, DHS, State
    # Known abbreviations/substrings from debug: DHS, STATE, HOMELAND SECURITY, DEPARTMENT OF STATE
    defense_ids = ['DHS', 'DOD', 'DOJ', 'DOS', 'STATE', 'CIA', 'NSA', 'ODNI']
    defense_names = ['DEFENSE', 'JUSTICE', 'HOMELAND', 'STATE', 'INTELLIGENCE', 'SECURITY']
    
    if any(k == abr for k in defense_ids) or any(k in agency for k in defense_names):
         return 'Defense/Intel'

    # Civilian / Service
    # Target: HHS, VA, Education, DOE, DOT, HUD
    # Known abbreviations/substrings from debug: HHS, VA, DOE, DOT, HUD, ED, USDA, SSA
    civilian_ids = ['HHS', 'VA', 'DOT', 'DOE', 'HUD', 'ED', 'USDA', 'SSA', 'DOC', 'DOL', 'TREAS']
    civilian_names = ['HEALTH', 'VETERANS', 'TRANSPORTATION', 'ENERGY', 'HOUSING', 'EDUCATION', 'AGRICULTURE', 'SOCIAL SECURITY', 'COMMERCE', 'LABOR', 'TREASURY']

    if any(k == abr for k in civilian_ids) or any(k in agency for k in civilian_names):
        return 'Civilian/Service'
        
    return 'Other'

df_eo['Agency_Category'] = df_eo.apply(categorize_agency, axis=1)

# --- 2. Notice Compliance Cleaning ---
# Based on debug output:
# Compliant: 'Online', 'In-person', 'Email', 'Telephone', 'Other'
# Non-Compliant: 'None of the above', 'waived'
# Exclude: 'NaN', 'N/A - individuals are not interacting...', 'AI is not safety...'

def clean_notice(val):
    s = str(val).lower().strip()
    
    # Check for exclusions first
    if s == 'nan' or 'n/a' in s or 'not safety' in s:
        return np.nan
        
    # Check for Non-Compliance
    if 'none of the above' in s or 'waived' in s:
        return 0
        
    # Check for Compliance (Positive indicators)
    if any(x in s for x in ['online', 'in-person', 'email', 'telephone', 'other', 'terms']): 
        return 1
        
    return np.nan

df_eo['Notice_Compliance'] = df_eo['59_ai_notice'].apply(clean_notice)

# --- 3. Analysis ---

# Filter for valid rows (Known Agency Category AND Known Notice Status)
df_analysis = df_eo[
    (df_eo['Agency_Category'].isin(['Defense/Intel', 'Civilian/Service'])) &
    (df_eo['Notice_Compliance'].notna())
].copy()

print(f"Valid analysis rows: {len(df_analysis)}")

# Generate Stats
group_stats = df_analysis.groupby('Agency_Category')['Notice_Compliance'].agg(['count', 'mean', 'sum'])
group_stats['compliance_pct'] = group_stats['mean'] * 100

print("\n--- Compliance Statistics ---")
print(group_stats)

# Contingency Table
contingency_table = pd.crosstab(df_analysis['Agency_Category'], df_analysis['Notice_Compliance'])
print("\n--- Contingency Table (0=No Notice, 1=Notice) ---")
print(contingency_table)

# Check if we have data
if len(df_analysis) == 0:
    print("No valid data found for analysis after filtering.")
    exit(0)

# Statistical Test
# Use Fisher's Exact if sample size is small, otherwise Chi-square
if contingency_table.size == 4 and contingency_table.min().min() < 5:
    print("\nUsing Fisher's Exact Test (small sample size)...")
    odds_ratio, p_value = fisher_exact(contingency_table)
    test_name = "Fisher's Exact"
else:
    print("\nUsing Chi-Square Test...")
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    test_name = "Chi-Square"

print(f"\n--- {test_name} Results ---")
print(f"p-value: {p_value:.4e}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Result: Statistically Significant Difference found.")
else:
    print("Result: No Statistically Significant Difference.")

# Plot
plt.figure(figsize=(8, 5))
colors = ['#1f77b4', '#ff7f0e']
ax = group_stats['mean'].plot(kind='bar', color=colors, alpha=0.8, yerr=1.96 * np.sqrt(group_stats['mean']*(1-group_stats['mean'])/group_stats['count']), capsize=5)
plt.title('AI Public Notice Compliance: Defense vs Civilian')
plt.ylabel('Compliance Rate')
plt.xlabel('Agency Category')
plt.xticks(rotation=0)
plt.ylim(0, 1.1)

# Add labels
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Valid analysis rows: 57

--- Compliance Statistics ---
                  count      mean   sum  compliance_pct
Agency_Category                                        
Civilian/Service     43  0.930233  40.0       93.023256
Defense/Intel        14  0.928571  13.0       92.857143

--- Contingency Table (0=No Notice, 1=Notice) ---
Notice_Compliance  0.0  1.0
Agency_Category            
Civilian/Service     3   40
Defense/Intel        1   13

Using Fisher's Exact Test (small sample size)...

--- Fisher's Exact Results ---
p-value: 1.0000e+00
Result: No Statistically Significant Difference.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is a detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot with Error Bars.
*   **Purpose:** The plot compares the mean "Compliance Rate" between two distinct categorical groups: "Civilian/Service" agencies and "Defense/Intel" agencies. The error bars are included to indicate the variability, uncertainty, or confidence intervals associated with the data.

### 2. Axes
*   **X-axis:**
    *   **Label:** "Agency Category"
    *   **Categories:** The axis displays two discrete categories: "Civilian/Service" and "Defense/Intel".
*   **Y-axis:**
    *   **Label:** "Compliance Rate"
    *   **Range:** The visible scale ranges from **0.0 to 1.0** (representing 0% to 100%), with tick marks every 0.2 units. The graph space extends slightly above 1.0 to accommodate the error bars.
    *   **Units:** The axis uses decimal notation (0.0–1.0) to represent proportions/rates, which correspond to the percentage values annotated on the bars.

### 3. Data Trends
*   **Bar Heights:**
    *   The **Civilian/Service** bar (blue) shows a compliance rate of **93.0%**.
    *   The **Defense/Intel** bar (orange) shows a compliance rate of **92.9%**.
*   **Comparison:** The heights of the two bars are nearly identical, indicating an extremely minimal difference between the two agency categories.
*   **Variability:** Both bars feature error bars. The error bar for the "Defense/Intel" category appears to have a slightly wider range (extending higher and lower) compared to the "Civilian/Service" category, suggesting slightly higher variability or a wider confidence interval for the Defense/Intel data.

### 4. Annotations and Legends
*   **Title:** "AI Public Notice Compliance: Defense vs Civilian" – This sets the context that the data relates to adherence to public notice requirements for Artificial Intelligence systems.
*   **Data Labels:** Specific percentage values are annotated directly above the bars:
    *   **"93.0%"** above the Civilian/Service bar.
    *   **"92.9%"** above the Defense/Intel bar.
*   **Error Bars:** The black "I" shaped lines represent the margin of error or standard deviation/error, indicating the precision of the mean estimates.

### 5. Statistical Insights
*   **Negligible Difference:** There is virtually no difference in compliance rates between the two groups. The difference is only **0.1%** (93.0% vs. 92.9%).
*   **High Compliance:** Both sectors demonstrate a high level of compliance, exceeding 90%.
*   **Statistical Significance:** Given the significant overlap of the error bars and the nearly identical means, it is highly likely that the difference between Civilian and Defense agencies is **statistically insignificant**. This suggests that agency type is not a predictor for AI public notice compliance; both perform equally well.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
