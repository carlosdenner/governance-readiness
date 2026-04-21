# Experiment 212: node_5_83

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_83` |
| **ID in Run** | 212 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:53:02.737291+00:00 |
| **Runtime** | 390.1s |
| **Parent** | `node_4_51` |
| **Children** | `node_6_72` |
| **Creation Index** | 213 |

---

## Hypothesis

> The Autonomy-Physical Risk Link: AI systems operating at 'High' or 'Full'
autonomy levels are statistically more likely to result in 'Physical' harm,
whereas 'Augmented' or 'Human-in-the-loop' systems predominantly result in
'Intangible' (Economic/Psychological) harm.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6694 (Maybe True) |
| **Posterior** | 0.2445 (Likely False) |
| **Surprise** | -0.5098 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 22.0 |
| Uncertain | 7.0 |
| Maybe False | 1.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 6.0 |
| Definitely False | 54.0 |

---

## Experiment Plan

**Objective:** Test the correlation between Autonomy Level and Harm Type in the AIID dataset.

### Steps
- 1. Filter AIID data.
- 2. Clean and categorize 'Autonomy Level' into 'High' (Autonomous) vs 'Low' (Assistive).
- 3. Categorize 'Tangible Harm' into 'Physical' vs 'Intangible/Economic'.
- 4. Create a cross-tabulation of Autonomy vs. Harm Type.
- 5. Calculate Cramér's V or perform a Chi-square test to measure the strength of association.

### Deliverables
- Crosstab of Autonomy vs. Harm; Statistical test results; Stacked bar chart of Harm types by Autonomy level.

---

## Analysis

The experiment successfully tested the 'Autonomy-Physical Risk Link' hypothesis
using 166 fully coded incidents from the AIID dataset. The analysis refutes the
hypothesis.

Findings:
1.  **No Statistical Significance**: The Chi-Square test yielded a p-value of
0.7572, well above the 0.05 threshold, indicating no significant relationship
between Autonomy Level and Harm Type.
2.  **Harm Distribution**: Contrary to the expectation that High Autonomy would
skew heavily toward Physical harm and Low Autonomy toward Intangible harm, the
distributions were remarkably similar. Physical harm was the dominant outcome
for both High Autonomy (58.3%) and Low/Augmented Autonomy (54.2%) systems.
3.  **Intangible Harm**: While Intangible harm was slightly more common in Low
Autonomy systems (45.8% vs 41.7%), the difference (Cramer's V = 0.024) is
negligible.

Conclusion: The level of autonomy (High vs. Low/Augmented) is not a predictor of
harm type (Physical vs. Intangible) in this dataset. Physical risks remain the
primary outcome across all autonomy levels analyzed.

---

## Review

The experiment was executed successfully after initial data mapping issues were
resolved. The programmer correctly debugged the non-standard schema (e.g.,
'Autonomy1', 'Autonomy3') and adjusted the categorization logic to allow for a
valid statistical test on 166 incidents. The analysis faithfully tested the
hypothesis using Chi-square and Cramer's V tests.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Load Data
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    print("Error: Dataset not found.")
    exit(1)

# 2. Filter for AIID Incidents
df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()

# 3. Categorize Autonomy Level
# Mapping based on debug findings: Autonomy1/2 -> Low, Autonomy3 -> High
def classify_autonomy(val):
    s = str(val).lower()
    if 'autonomy3' in s:
        return 'High (Autonomous)'
    elif 'autonomy1' in s or 'autonomy2' in s:
        return 'Low (Augmented)'
    return 'Unknown'

df_aiid['Autonomy_Cat'] = df_aiid['Autonomy Level'].apply(classify_autonomy)

# 4. Categorize Harm Type
# Use 'Tangible Harm' for Physical and 'Special Interest Intangible Harm' for Intangible
def classify_harm(row):
    # Check Tangible Harm (Physical)
    tangible = str(row.get('Tangible Harm', '')).lower()
    is_physical = 'definitively' in tangible or 'imminent' in tangible
    
    # Check Intangible Harm
    intangible_col = str(row.get('Special Interest Intangible Harm', '')).lower()
    is_intangible = 'yes' in intangible_col
    
    if is_physical:
        return 'Physical'
    elif is_intangible:
        return 'Intangible'
    else:
        # Fallback: if 'no tangible harm' is explicitly stated and 'Harm Domain' is yes
        harm_domain = str(row.get('Harm Domain', '')).lower()
        if 'no tangible' in tangible and 'yes' in harm_domain:
            return 'Intangible'
            
    return 'Unknown'

df_aiid['Harm_Cat'] = df_aiid.apply(classify_harm, axis=1)

# 5. Filter for Analysis
analysis_df = df_aiid[
    (df_aiid['Autonomy_Cat'] != 'Unknown') &
    (df_aiid['Harm_Cat'] != 'Unknown')
].copy()

print(f"Data points for analysis: {len(analysis_df)}")
print("Distribution:")
print(analysis_df.groupby(['Autonomy_Cat', 'Harm_Cat']).size())

# 6. Statistical Analysis
ct = pd.crosstab(analysis_df['Autonomy_Cat'], analysis_df['Harm_Cat'])

print("\n--- Contingency Table ---")
print(ct)

if not ct.empty and ct.shape == (2, 2):
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * 1)) # min_dim - 1 is 1
    
    print("\n--- Statistical Test Results ---")
    print(f"Chi-Square: {chi2:.4f}")
    print(f"P-Value: {p:.4e}")
    print(f"Cramer's V: {cramers_v:.4f}")
    
    alpha = 0.05
    if p < alpha:
        print("\nResult: Reject Null Hypothesis. Significant association found.")
    else:
        print("\nResult: Fail to reject Null Hypothesis. No significant association.")
        
    # 7. Visualization
    # Normalize rows to percentages
    ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
    ax = ct_norm.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], figsize=(8, 6))
    
    plt.title('Harm Type Distribution by Autonomy Level')
    plt.xlabel('Autonomy Level')
    plt.ylabel('Percentage')
    plt.legend(title='Harm Type', bbox_to_anchor=(1.05, 1))
    
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center')
        
    plt.tight_layout()
    plt.show()

elif not ct.empty:
    print("\nNot enough categories for 2x2 Chi-Square. Printing raw counts only.")
    print(ct)
else:
    print("\nNo valid data found after filtering.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Data points for analysis: 166
Distribution:
Autonomy_Cat       Harm_Cat  
High (Autonomous)  Intangible    20
                   Physical      28
Low (Augmented)    Intangible    54
                   Physical      64
dtype: int64

--- Contingency Table ---
Harm_Cat           Intangible  Physical
Autonomy_Cat                           
High (Autonomous)          20        28
Low (Augmented)            54        64

--- Statistical Test Results ---
Chi-Square: 0.0956
P-Value: 7.5721e-01
Cramer's V: 0.0240

Result: Fail to reject Null Hypothesis. No significant association.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** This chart is used to compare the relative percentage distribution of two categorical variables ("Intangible" vs. "Physical" harm) across distinct groups ("High" vs. "Low" autonomy levels). It visualizes how the composition of harm types changes depending on the level of autonomy.

### 2. Axes
*   **X-axis:**
    *   **Title:** "Autonomy Level"
    *   **Labels:** Two categorical groups are presented: "High (Autonomous)" and "Low (Augmented)". The labels are oriented vertically (90 degrees).
*   **Y-axis:**
    *   **Title:** "Percentage"
    *   **Range:** The scale runs from 0 to 100.
    *   **Units:** Percent (%).

### 3. Data Trends
*   **General Pattern:** In both autonomy levels, "Physical" harm constitutes the majority (over 50%) of the distribution, while "Intangible" harm constitutes the minority.
*   **High (Autonomous) Level:**
    *   **Physical Harm (Blue):** Comprises the larger portion at **58.3%**.
    *   **Intangible Harm (Pink):** Comprises the smaller portion at **41.7%**.
*   **Low (Augmented) Level:**
    *   **Physical Harm (Blue):** Comprises **54.2%**.
    *   **Intangible Harm (Pink):** Comprises **45.8%**.
*   **Comparison:** As the autonomy level shifts from High (Autonomous) to Low (Augmented), the proportion of Physical harm decreases slightly, while the proportion of Intangible harm increases.

### 4. Annotations and Legends
*   **Chart Title:** "Harm Type Distribution by Autonomy Level" appears at the top center.
*   **Legend:** Located on the right side, titled "Harm Type". It indicates:
    *   **Pink/Light Red:** Represents "Intangible" harm.
    *   **Light Blue:** Represents "Physical" harm.
*   **Data Labels:** Specific percentage values are annotated directly inside each bar segment (e.g., "58.3%", "41.7%"), providing precise data points without needing to estimate from the Y-axis.

### 5. Statistical Insights
*   **dominance of Physical Harm:** Regardless of whether the system is fully autonomous or augmented (low autonomy), physical harm is the primary concern, consistently accounting for more than half of the recorded harm types.
*   **Shift in Harm Composition:** There is a notable, albeit small, shift in the distribution based on autonomy.
    *   **High Autonomy** is associated with a higher likelihood of Physical harm (58.3%) compared to Low Autonomy (54.2%).
    *   **Low Autonomy (Augmented)** shows a higher relative incidence of Intangible harm (45.8%) compared to High Autonomy (41.7%).
*   **Conclusion:** While physical risks remain paramount in both scenarios, the data suggests that as human involvement increases (shifting from Autonomous to Augmented), the relative share of intangible harms (potentially psychological, financial, or societal) rises by approximately 4.1 percentage points.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
