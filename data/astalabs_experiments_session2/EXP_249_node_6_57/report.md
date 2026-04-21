# Experiment 249: node_6_57

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_57` |
| **ID in Run** | 249 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T12:47:35.834395+00:00 |
| **Runtime** | 321.1s |
| **Parent** | `node_5_87` |
| **Children** | `node_7_11` |
| **Creation Index** | 250 |

---

## Hypothesis

> Diagnosability of Tangible Harm: AI incidents resulting in 'Tangible' harm
(e.g., physical, financial) are significantly more likely to have a specific
'Known AI Technical Failure' identified than incidents with 'Intangible' harm,
reflecting the difficulty in diagnosing socio-technical failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2582 (Likely False) |
| **Surprise** | -0.5804 |
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
| Maybe False | 2.0 |
| Definitely False | 58.0 |

---

## Experiment Plan

**Objective:** Test if Tangible Harm incidents have higher rates of identified technical failures compared to Intangible Harm incidents.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `source_table` = 'aiid_incidents'.
- 2. Define `harm_type`: Use `74_tangible_harm` to categorize as 'Tangible' (e.g., 'Yes', 'Physical') vs 'Intangible' (e.g., 'No', 'Reputation').
- 3. Define `has_tech_diagnosis`: Check if `85_known_ai_technical_failure` is not null/empty and not 'Unknown'.
- 4. Generate a contingency table (Harm Type vs. Diagnosis Present).
- 5. Perform a Chi-square test.
- 6. Calculate the diagnosis rate for Tangible vs. Intangible harms.

### Deliverables
- Contingency table, Chi-square p-value, and comparative diagnosis rates.

---

## Analysis

The experiment successfully tested the 'Diagnosability of Tangible Harm'
hypothesis using the AIID incidents dataset. After mapping the verbose
descriptions in the 'Tangible Harm' column, 166 incidents were successfully
categorized into 'Tangible' (n=72) or 'Intangible' (n=94) harm.

The analysis revealed that technical failures were identified in **65.3%** of
Tangible harm cases (47/72) compared to **64.9%** of Intangible harm cases
(61/94). A Chi-square test yielded a statistic of 0.00 and a p-value of 1.0,
indicating absolutely no statistically significant difference between the two
groups.

Consequently, the hypothesis is **rejected**. The data suggests that the nature
of the harm (tangible vs. intangible) does not influence the likelihood of a
technical root cause being diagnosed or documented; both types of incidents have
a ~65% rate of identified technical failures.

---

## Review

The experiment was successfully executed. After refining the data mapping logic
to handle verbose text descriptions in the 'Tangible Harm' column (e.g.,
'tangible harm definitively occurred'), the analysis successfully categorized
166 incidents into 'Tangible' (n=72) and 'Intangible' (n=94) groups.

**Hypothesis Test Results:**
1. **Diagnosis Rates:**
   - **Tangible Harm:** 65.3% (47/72) had identified technical failures.
   - **Intangible Harm:** 64.9% (61/94) had identified technical failures.
2. **Statistical Test:**
   - Chi-Square Statistic: 0.0000
   - p-value: 1.0000

**Conclusion:**
The hypothesis that tangible harms are more easily diagnosed is **rejected**.
The data shows almost identical rates of technical failure identification
between the two groups (~65%), indicating that the tangibility of the harm does
not significantly affect the likelihood of a technical root cause being
identified in the AIID dataset.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# 1. Load dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback for different directory structure if needed
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for 'aiid_incidents'
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid_df)}")

# 3. Identify columns
# Normalize column names to find matches easily
aiid_df.columns = [c.strip() for c in aiid_df.columns]

tangible_col = next((c for c in aiid_df.columns if 'Tangible Harm' in c), None)
tech_fail_col = next((c for c in aiid_df.columns if 'Known AI Technical Failure' in c), None)

if not tangible_col or not tech_fail_col:
    print(f"Columns not found. Tangible: {tangible_col}, Tech: {tech_fail_col}")
    exit()

print(f"Using columns: '{tangible_col}' and '{tech_fail_col}'")

# 4. Define Mapping Logic based on observed values
# Observed values: 
# - 'tangible harm definitively occurred'
# - 'no tangible harm, near-miss, or issue'
# - 'non-imminent risk...'
# - 'imminent risk...'
# - NaN

def categorize_harm(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower().strip()
    
    # Exact string matching based on previous debug output
    if 'tangible harm definitively occurred' in val_str:
        return 'Tangible'
    elif 'no tangible harm' in val_str:
        return 'Intangible'
    else:
        return None # Exclude risks/near-misses/unclear to be strict about 'Harm Incidents'

aiid_df['harm_category'] = aiid_df[tangible_col].apply(categorize_harm)

# Filter out nulls (which includes NaNs and Risks)
analysis_df = aiid_df[aiid_df['harm_category'].notna()].copy()

print(f"\nRows available for analysis after filtering: {len(analysis_df)}")
print(analysis_df['harm_category'].value_counts())

# 5. Define Diagnosis Logic
# Check if 'Known AI Technical Failure' is populated with something meaningful
def categorize_diagnosis(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    # If it says 'unknown' explicitly, or is empty, it's not diagnosed.
    if val_str in ['', 'unknown', 'unspecified', 'nan']:
        return 0
    return 1

analysis_df['has_diagnosis'] = analysis_df[tech_fail_col].apply(categorize_diagnosis)

# 6. Statistical Analysis
if analysis_df.empty or len(analysis_df['harm_category'].unique()) < 2:
    print("Not enough data for comparison.")
else:
    # Contingency Table
    ct = pd.crosstab(analysis_df['harm_category'], analysis_df['has_diagnosis'])
    # Ensure both columns 0/1 exist for display
    if 0 not in ct.columns: ct[0] = 0
    if 1 not in ct.columns: ct[1] = 0
    ct = ct[[0, 1]]
    ct.columns = ['No Diagnosis', 'Diagnosis Present']
    
    print("\n--- Contingency Table ---")
    print(ct)

    # Chi-square Test
    chi2, p, dof, expected = chi2_contingency(ct)
    
    print(f"\nChi-square statistic: {chi2:.4f}")
    print(f"p-value: {p:.4e}")

    # Rates
    rates = analysis_df.groupby('harm_category')['has_diagnosis'].mean()
    print("\n--- Diagnosis Rates ---")
    print(rates)

    # Visualization
    plt.figure(figsize=(8, 6))
    colors = ['#ff9999', '#66b3ff'] # Red for Intangible, Blue for Tangible
    ax = rates.plot(kind='bar', color=colors, edgecolor='black', rot=0)
    
    plt.title('Technical Failure Diagnosis Rate: Tangible vs Intangible Harm')
    plt.ylabel('Proportion with Identified Technical Failure')
    plt.xlabel('Harm Type')
    plt.ylim(0, 1.0)
    
    # Annotate bars
    for i, v in enumerate(rates):
        ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: AIID Incidents loaded: 1362
Using columns: 'Tangible Harm' and 'Known AI Technical Failure'

Rows available for analysis after filtering: 166
harm_category
Intangible    94
Tangible      72
Name: count, dtype: int64

--- Contingency Table ---
               No Diagnosis  Diagnosis Present
harm_category                                 
Intangible               33                 61
Tangible                 25                 47

Chi-square statistic: 0.0000
p-value: 1.0000e+00

--- Diagnosis Rates ---
harm_category
Intangible    0.648936
Tangible      0.652778
Name: has_diagnosis, dtype: float64


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot is designed to compare the proportion of cases where a technical failure was identified across two distinct categories of harm: "Intangible" and "Tangible."

### 2. Axes
*   **X-axis (Horizontal):**
    *   **Title:** "Harm Type"
    *   **Labels:** Two categorical labels: "Intangible" and "Tangible".
*   **Y-axis (Vertical):**
    *   **Title:** "Proportion with Identified Technical Failure"
    *   **Range:** The axis scales from **0.0 to 1.0**.
    *   **Grid:** Horizontal dashed grid lines appear at intervals of 0.2 (0.2, 0.4, 0.6, 0.8) to assist in reading the bar heights.

### 3. Data Trends
*   **Comparison of Heights:** The two bars are nearly identical in height, indicating a very similar diagnosis rate between the two categories.
*   **Tallest Bar:** The **Tangible** harm category (blue bar) is the tallest, reaching a value of **65.3%** (or 0.653).
*   **Shortest Bar:** The **Intangible** harm category (salmon/light red bar) is slightly shorter, reaching a value of **64.9%** (or 0.649).
*   **Visual Pattern:** Visually, the difference between the two categories is negligible. Both bars extend slightly past the 0.6 grid line, settling around the mid-60% range.

### 4. Annotations and Legends
*   **Bar Labels:** Exact percentage values are annotated in bold text directly above each bar:
    *   Above the Intangible bar: **64.9%**
    *   Above the Tangible bar: **65.3%**
*   **Colors:** The plot uses distinct colors to differentiate the categories visually:
    *   **Intangible:** Represented by a light red or salmon color.
    *   **Tangible:** Represented by a light blue color.
*   **Title:** The chart is titled "**Technical Failure Diagnosis Rate: Tangible vs Intangible Harm**".

### 5. Statistical Insights
*   **Parity in Diagnosis Rates:** The most significant insight from this plot is the lack of a substantial difference between the two groups. The difference in the rate of identified technical failures between Tangible (65.3%) and Intangible (64.9%) harm is only **0.4%**.
*   **Implication:** This suggests that the nature of the harm (whether it is tangible, like physical damage, or intangible, like reputational damage) has little to no correlation with the likelihood of a technical failure being diagnosed in this dataset. The diagnosis process appears to be consistent regardless of the harm classification.
*   **Overall Prevalence:** In approximately two-thirds (roughly 65%) of cases for both harm types, a technical failure is identified.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
