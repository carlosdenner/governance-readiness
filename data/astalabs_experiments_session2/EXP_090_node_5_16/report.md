# Experiment 90: node_5_16

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_16` |
| **ID in Run** | 90 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:18:52.210420+00:00 |
| **Runtime** | 395.6s |
| **Parent** | `node_4_8` |
| **Children** | `node_6_11`, `node_6_76` |
| **Creation Index** | 91 |

---

## Hypothesis

> The 'Physicality' of Autonomy: Higher levels of system autonomy are strongly
associated with 'Physical' harms, whereas lower autonomy systems are dominated
by 'Intangible' harms (e.g., economic, bias).

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.4435 (Maybe False) |
| **Posterior** | 0.1951 (Likely False) |
| **Surprise** | -0.2982 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 11.0 |
| Uncertain | 1.0 |
| Maybe False | 18.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 16.0 |
| Definitely False | 44.0 |

---

## Experiment Plan

**Objective:** Map the relationship between the degree of AI autonomy and the nature of the resulting harm.

### Steps
- 1. Filter for `aiid_incidents`.
- 2. Clean `Autonomy Level`: Bin into 'High' (Autonomous, High) vs. 'Low/Medium' (Assisted, Low).
- 3. Clean `Tangible Harm`: Group into 'Physical' vs. 'Intangible' (Financial, Reputational, Civil Rights, etc.).
- 4. Create a contingency table and run a Chi-square test.
- 5. Calculate Cramer's V to determine effect size.

### Deliverables
- Contingency table, Chi-square results, Cramer's V, and a Heatmap of Autonomy vs. Harm Type.

---

## Analysis

The experiment successfully tested the hypothesis linking AI autonomy levels to
harm types using the AIID dataset (N=154). The analysis mapped 'Autonomy Level'
to High vs. Low/Medium and classified harms into 'Physical/Tangible' vs.
'Intangible'.

**Findings:**
The hypothesis was **rejected**. The Chi-square test yielded a p-value of
**0.6587** (Statistic=0.1951), indicating **no statistically significant
association** between the level of autonomy and the type of harm.

**Key Observations:**
1.  **Uniform Distribution:** Physical/Tangible harm was the dominant category
for *both* groups. It accounted for **63.6%** of incidents in High Autonomy
systems and **58.2%** in Low/Medium Autonomy systems.
2.  **Negligible Effect:** The Cramer's V score of **0.0356** confirms an
extremely weak, negligible relationship.

The data suggests that, within this specific subset of incidents, the nature of
the harm (physical vs. intangible) is independent of the system's autonomy
level.

---

## Review

The experiment was successfully executed after initial data mapping challenges
were resolved. The programmer correctly identified the necessary columns
('Autonomy Level', 'Tangible Harm', 'Special Interest Intangible Harm') and
applied appropriate cleaning logic to categorize incidents into 'High' vs.
'Low/Medium' autonomy and 'Physical' vs. 'Intangible' harm.

**Hypothesis Evaluation:**
The hypothesis that higher autonomy is strongly associated with physical harms
was **rejected**. The statistical analysis (Chi-square test) yielded a p-value
of **0.6587** and a Cramer's V of **0.0356**, indicating no statistically
significant relationship between the level of autonomy and the nature of the
harm in the analyzed sample (N=154).

**Findings:**
1.  **Uniformity:** Physical/Tangible harm was the predominant outcome for both
High Autonomy (63.6%) and Low/Medium Autonomy (58.2%) systems.
2.  **Independence:** The nature of the harm appears independent of the system's
autonomy level in this dataset, contradicting the expectation that lower
autonomy systems would disproportionately cause intangible harms.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid_df)}")

# 3. Define Mapping Functions

def map_autonomy(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower()
    if 'autonomy3' in val_str:
        return 'High Autonomy'
    elif 'autonomy1' in val_str or 'autonomy2' in val_str:
        return 'Low/Medium Autonomy'
    return np.nan

def map_harm_composite(row):
    # Extract values
    tangible = str(row['Tangible Harm']).lower()
    intangible = str(row['Special Interest Intangible Harm']).lower()
    
    # Check for Physical/Tangible Harm indicators
    # We include definitive occurrences and imminent risks (near misses) as 'Physical/Tangible'
    is_tangible = False
    if 'tangible harm definitively occurred' in tangible:
        is_tangible = True
    elif 'imminent risk' in tangible:
        is_tangible = True
        
    # Check for Intangible Harm
    is_intangible = False
    if 'yes' in intangible:
        is_intangible = True
        
    # Classification Logic
    if is_tangible:
        return 'Physical/Tangible Harm'
    elif is_intangible:
        return 'Intangible Harm'
    else:
        return np.nan

# 4. Apply Mappings
aiid_df['Autonomy_Clean'] = aiid_df['Autonomy Level'].apply(map_autonomy)
aiid_df['Harm_Clean'] = aiid_df.apply(map_harm_composite, axis=1)

# 5. Filter for Analysis
analysis_df = aiid_df.dropna(subset=['Autonomy_Clean', 'Harm_Clean'])

print(f"\nRecords ready for analysis: {len(analysis_df)}")
print("Distribution by Autonomy:\n", analysis_df['Autonomy_Clean'].value_counts())
print("Distribution by Harm:\n", analysis_df['Harm_Clean'].value_counts())

if len(analysis_df) < 5:
    print("Insufficient data for statistical testing.")
else:
    # 6. Statistical Analysis
    contingency_table = pd.crosstab(analysis_df['Autonomy_Clean'], analysis_df['Harm_Clean'])
    print("\nContingency Table:")
    print(contingency_table)

    # Chi-square
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"p-value: {p:.4e}")

    # Cramer's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    print(f"Cramer's V: {cramers_v:.4f}")

    # 7. Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Oranges')
    plt.title('Autonomy Level vs. Harm Type')
    plt.ylabel('Autonomy Level')
    plt.xlabel('Harm Type')
    plt.tight_layout()
    plt.show()

    # Row Percentages
    row_props = pd.crosstab(analysis_df['Autonomy_Clean'], analysis_df['Harm_Clean'], normalize='index') * 100
    print("\nRow Proportions (%):")
    print(row_props)

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset...
AIID Incidents loaded: 1362

Records ready for analysis: 154
Distribution by Autonomy:
 Autonomy_Clean
Low/Medium Autonomy    110
High Autonomy           44
Name: count, dtype: int64
Distribution by Harm:
 Harm_Clean
Physical/Tangible Harm    92
Intangible Harm           62
Name: count, dtype: int64

Contingency Table:
Harm_Clean           Intangible Harm  Physical/Tangible Harm
Autonomy_Clean                                              
High Autonomy                     16                      28
Low/Medium Autonomy               46                      64

Chi-square Statistic: 0.1951
p-value: 6.5873e-01
Cramer's V: 0.0356

Row Proportions (%):
Harm_Clean           Intangible Harm  Physical/Tangible Harm
Autonomy_Clean                                              
High Autonomy              36.363636               63.636364
Low/Medium Autonomy        41.818182               58.181818


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Heatmap (specifically a 2x2 contingency table visualization).
*   **Purpose:** To visualize the frequency distribution or relationship between two categorical variables: "Autonomy Level" and "Harm Type." The color intensity represents the magnitude of the count in each intersection.

### 2. Axes
*   **X-Axis:**
    *   **Title:** Harm Type
    *   **Labels:** "Intangible Harm" and "Physical/Tangible Harm".
*   **Y-Axis:**
    *   **Title:** Autonomy Level
    *   **Labels:** "High Autonomy" and "Low/Medium Autonomy".
*   **Color Bar (Z-Axis equivalent):**
    *   **Range:** The scale on the right indicates values ranging roughly from just below 20 to just above 60.
    *   **Gradient:** Light beige indicates lower values, while dark burnt orange indicates higher values.

### 3. Data Trends
*   **Highest Value:** The intersection of **Low/Medium Autonomy** and **Physical/Tangible Harm** (bottom-right quadrant) contains the highest count at **64**. This is represented by the darkest brown color.
*   **Lowest Value:** The intersection of **High Autonomy** and **Intangible Harm** (top-left quadrant) contains the lowest count at **16**, represented by the lightest beige color.
*   **Row Comparison:** The "Low/Medium Autonomy" row generally contains significantly higher values (46 and 64) compared to the "High Autonomy" row (16 and 28).
*   **Column Comparison:** The "Physical/Tangible Harm" column generally contains higher values (28 and 64) compared to the "Intangible Harm" column (16 and 46).

### 4. Annotations and Legends
*   **Title:** "Autonomy Level vs. Harm Type" is centered at the top.
*   **Cell Values:** Each of the four quadrants is annotated with the exact count:
    *   Top-Left: 16
    *   Top-Right: 28
    *   Bottom-Left: 46
    *   Bottom-Right: 64
*   **Color Scale Legend:** A vertical bar on the right side provides a reference for the color coding, with tick marks at 20, 30, 40, 50, and 60.

### 5. Statistical Insights
*   **Total Sample Size:** The total count represented in the plot is **154** ($16 + 28 + 46 + 64$).
*   **Prevalence of Physical Harm:** Physical/Tangible harm ($28 + 64 = 92$) is more prevalent in this dataset than Intangible harm ($16 + 46 = 62$), regardless of the autonomy level.
*   **Prevalence of Lower Autonomy:** Incidents involving Low/Medium Autonomy systems ($46 + 64 = 110$) are much more frequent than those involving High Autonomy systems ($16 + 28 = 44$). This suggests that the dataset may be skewed toward lower autonomy systems or that lower autonomy systems are more prone to recorded incidents.
*   **Dominant Scenario:** The most common scenario is a Low/Medium Autonomy system causing Physical/Tangible Harm (approx. 41.5% of the total).
*   **Rarest Scenario:** The least common scenario is a High Autonomy system causing Intangible Harm (approx. 10.4% of the total).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
