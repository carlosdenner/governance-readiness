# Experiment 252: node_6_59

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_59` |
| **ID in Run** | 252 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T12:52:59.831855+00:00 |
| **Runtime** | 423.4s |
| **Parent** | `node_5_74` |
| **Children** | None |
| **Creation Index** | 253 |

---

## Hypothesis

> Sector-Specific Harm Profiles: The 'Healthcare' sector is associated with a
significantly higher proportion of 'Physical/Safety' harm domains compared to
the 'Financial' sector, which correlates more strongly with
'Economic/Reputational' harm.

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

**Objective:** Resolve previous data mapping failures by inspecting specific column values in the AIID dataset and re-testing the association between Sector (Healthcare vs. Financial) and Harm Type (Physical vs. Economic).

### Steps
- 1. Load the 'astalabs_discovery_all_data.csv' dataset and filter for rows where `source_table` is 'aiid_incidents'.
- 2. Print the top 50 unique values and their counts for the columns `Sector of Deployment`, `Harm Domain`, and `Tangible Harm` to establish ground truth for mapping.
- 3. Create a derived column `Sector_Group` using case-insensitive string matching: map terms like 'health', 'medic', 'hospital' to 'Healthcare'; map 'financ', 'bank', 'insur', 'trading' to 'Financial'. Filter the dataset to keep only rows belonging to these two groups.
- 4. Create a derived column `Harm_Category` based primarily on the `Tangible Harm` column (falling back to `Harm Domain` if null). Map terms like 'physical', 'safety', 'death', 'injury' to 'Physical/Safety'. Map terms like 'financial', 'economic', 'monetary', 'property' to 'Economic'. Ignore or label others as 'Other'.
- 5. Filter the dataset to include only rows where `Harm_Category` is either 'Physical/Safety' or 'Economic' to strictly test the tradeoff proposed in the hypothesis.
- 6. Generate a contingency table (cross-tabulation) of `Sector_Group` vs. `Harm_Category`.
- 7. Perform a Chi-square test of independence on this contingency table.
- 8. Generate a heatmap of the contingency table to visualize the relationship.

### Deliverables
- List of unique values for Sector/Harm columns; Contingency table of Sector vs. Harm Category; Chi-square test statistic and p-value; Heatmap visualization.

---

## Analysis

The experiment successfully tested the 'Sector-Specific Harm Profiles'
hypothesis using a text-mining approach to overcome the metadata sparsity
encountered in previous attempts. By deriving `Sector` and `Harm Type` from
incident descriptions, the analysis recovered 187 valid data points.

1. **Statistical Evidence**: The Chi-square test yielded a statistic of 30.37
with a p-value of 3.57e-08, which is well below the 0.05 threshold. This
confirms a statistically significant dependence between the sector of deployment
and the nature of the harm.

2. **Hypothesis Confirmation**: The row probabilities support the specific
claims of the hypothesis:
   - **Financial Sector**: 82% of incidents were associated with
'Economic/Reputational' harm.
   - **Healthcare Sector**: 70% of incidents were associated with
'Physical/Safety' harm.

3. **Conclusion**: The hypothesis is **supported**. There is strong evidence
that AI incidents in the healthcare sector are disproportionately likely to
involve physical safety risks, whereas financial AI incidents are predominantly
characterized by economic or reputational damage.

---

## Review

The experiment successfully tested the hypothesis using a text-mining approach
to overcome the metadata limitations identified in previous iterations. By
deriving 'Sector' and 'Harm Type' from incident descriptions, the analysis
recovered 187 valid data points. The Chi-square test (p < 0.001) and row
probabilities (Financial: 82% Economic; Healthcare: 70% Physical) provide
strong, statistically significant evidence supporting the hypothesis that AI
risks are sector-dependent.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID incidents loaded: {len(aiid)}")

# Combine title and description for text mining
aiid['text_content'] = aiid['title'].fillna('') + " " + aiid['description'].fillna('')
aiid['text_content'] = aiid['text_content'].str.lower()

# ---------------------------------------------------------
# Text Mining Functions
# ---------------------------------------------------------

def get_sector(text):
    health_kw = ['health', 'hospital', 'medic', 'patient', 'doctor', 'nurse', 'diagnos', 'clinic', 'cancer', 'surgery']
    finance_kw = ['bank', 'financ', 'loan', 'credit', 'trading', 'stock', 'market', 'money', 'invest', 'crypto', 'currency']
    
    is_health = any(k in text for k in health_kw)
    is_finance = any(k in text for k in finance_kw)
    
    if is_health and not is_finance:
        return 'Healthcare'
    if is_finance and not is_health:
        return 'Financial'
    return None  # Ambiguous or neither

def get_harm(text):
    # Physical/Safety keywords
    phys_kw = ['death', 'kill', 'inju', 'physic', 'safety', 'accident', 'crash', 'violen', 'attack', 'collision', 'burn', 'murder', 'died']
    # Economic/Reputational keywords (avoiding 'credit' here if it's too overlapping, but 'fraud' is good)
    econ_kw = ['fraud', 'scam', 'theft', 'monetary', 'bankrupt', 'loss', 'fine', 'penalty', 'reputation', 'steal', 'stolen', 'cost']
    
    is_phys = any(k in text for k in phys_kw)
    is_econ = any(k in text for k in econ_kw)
    
    if is_phys and not is_econ:
        return 'Physical/Safety'
    if is_econ and not is_phys:
        return 'Economic/Reputational'
    # If both, prioritize Physical/Safety as it is the more severe category often distinguishing these sectors
    if is_phys and is_econ:
        return 'Physical/Safety'
    return None

# Apply mappings
aiid['Derived_Sector'] = aiid['text_content'].apply(get_sector)
aiid['Derived_Harm'] = aiid['text_content'].apply(get_harm)

# Filter for valid rows
analysis_df = aiid[aiid['Derived_Sector'].notna() & aiid['Derived_Harm'].notna()].copy()

print(f"\nDerived Data Subset Size: {len(analysis_df)}")
print("Counts by Sector:\n", analysis_df['Derived_Sector'].value_counts())
print("Counts by Harm:\n", analysis_df['Derived_Harm'].value_counts())

# ---------------------------------------------------------
# Statistical Test
# ---------------------------------------------------------
if len(analysis_df) < 5:
    print("Insufficient data for statistical testing.")
else:
    contingency = pd.crosstab(analysis_df['Derived_Sector'], analysis_df['Derived_Harm'])
    print("\nContingency Table:")
    print(contingency)
    
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"p-value: {p:.4e}")
    
    # Calculate Row Percentages for clarity
    row_probs = contingency.div(contingency.sum(axis=1), axis=0)
    print("\nRow Percentages:")
    print(row_probs)

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    # Using a heatmap of counts
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues')
    plt.title('Heatmap: Sector vs. Inferred Harm Type')
    plt.xlabel('Harm Type')
    plt.ylabel('Sector')
    plt.tight_layout()
    plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: AIID incidents loaded: 1362

Derived Data Subset Size: 187
Counts by Sector:
 Derived_Sector
Financial     160
Healthcare     27
Name: count, dtype: int64
Counts by Harm:
 Derived_Harm
Economic/Reputational    139
Physical/Safety           48
Name: count, dtype: int64

Contingency Table:
Derived_Harm    Economic/Reputational  Physical/Safety
Derived_Sector                                        
Financial                         131               29
Healthcare                          8               19

Chi-square Statistic: 30.3680
p-value: 3.5738e-08

Row Percentages:
Derived_Harm    Economic/Reputational  Physical/Safety
Derived_Sector                                        
Financial                    0.818750         0.181250
Healthcare                   0.296296         0.703704


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Heatmap (Confusion Matrix style).
*   **Purpose:** This plot visualizes the frequency distribution between two categorical variables: the **Sector** (Financial vs. Healthcare) and the **Inferred Harm Type** (Economic/Reputational vs. Physical/Safety). The intensity of the color represents the magnitude of the count in each category intersection.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Harm Type"
    *   **Labels:** "Economic/Reputational", "Physical/Safety"
    *   **Units:** Categorical categories.
*   **Y-Axis:**
    *   **Title:** "Sector"
    *   **Labels:** "Financial", "Healthcare"
    *   **Units:** Categorical categories.
*   **Value Ranges:**
    *   The axes do not have numerical ranges as they represent discrete categories.
    *   The **Color Scale (Right Sidebar)** represents the count values, ranging visually from a light blue (low count) to a dark blue (high count). The numerical ticks on the color bar range from **20 to 120**, though the actual data values span from **8 to 131**.

### 3. Data Trends
*   **High Values (Hotspots):** The highest concentration of data is found in the intersection of the **Financial** sector and **Economic/Reputational** harm, with a count of **131**. This cell is colored the darkest blue, indicating it is the dominant category in this dataset.
*   **Low Values:** The lowest count is observed in the intersection of the **Healthcare** sector and **Economic/Reputational** harm, with a value of **8**, indicated by the lightest blue color.
*   **Secondary Trends:**
    *   The **Financial** sector has a moderate association with **Physical/Safety** harm (count: 29).
    *   The **Healthcare** sector shows a higher count for **Physical/Safety** harm (count: 19) compared to Economic/Reputational harm (count: 8).

### 4. Annotations and Legends
*   **Cell Annotations:** Each of the four cells contains a numerical value representing the exact count of observations for that specific combination of Sector and Harm Type (131, 29, 8, 19).
*   **Color Bar:** Located on the right side of the plot, this legend maps the shade of blue to the numerical count, providing a visual reference for density.
*   **Title:** The chart is titled "**Heatmap: Sector vs. Inferred Harm Type**", clearly defining the variables under comparison.

### 5. Statistical Insights
*   **Sector Skew:** The dataset is heavily skewed toward the Financial sector.
    *   Total Financial observations: 160 ($131 + 29$)
    *   Total Healthcare observations: 27 ($8 + 19$)
    *   This suggests the dataset primarily consists of financial incidents or data points.
*   **Correlation of Harm Type by Sector:**
    *   **Financial Sector:** There is a strong correlation between the Financial sector and Economic/Reputational harm. Approximately **82%** ($131/160$) of financial incidents fall into this category.
    *   **Healthcare Sector:** Conversely, the Healthcare sector shows a preference for Physical/Safety harm. Approximately **70%** ($19/27$) of healthcare incidents are categorized as Physical/Safety.
*   **Conclusion:** The plot demonstrates a logical alignment between sector function and harm type. Financial entities are prone to monetary and reputation loss, while healthcare entities are more prone to safety and physical risks. The visualization effectively highlights that while the Financial sector dominates the raw volume of this dataset, the nature of the harm is distinct for each sector.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
