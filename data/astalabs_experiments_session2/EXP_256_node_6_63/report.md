# Experiment 256: node_6_63

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_63` |
| **ID in Run** | 256 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:04:53.524956+00:00 |
| **Runtime** | 271.4s |
| **Parent** | `node_5_51` |
| **Children** | None |
| **Creation Index** | 257 |

---

## Hypothesis

> Risk-Control Misalignment: In federal AI deployments, systems self-classified as
'High Impact' are not statistically more likely to possess 'Independent
Evaluation' controls than 'Low Impact' systems, indicating a gap between risk
identification and verification rigor.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2527 (Likely False) |
| **Surprise** | -0.5870 |
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
| Maybe False | 0.0 |
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Assess if self-reported risk levels correlate with the presence of independent evaluations.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for the `eo13960_scored` subset.
- 2. Clean the `17_impact_type` column to group values into 'High' (High/Moderate-High) and 'Low/Moderate' (Low/Moderate).
- 3. Binarize the `55_independent_eval` column (Yes vs No/Not Applicable).
- 4. Create a contingency table of Impact Level vs. Independent Evaluation.
- 5. Perform a Chi-square test of independence to determine if High Impact systems are significantly more likely to have evaluations.

### Deliverables
- 1. Contingency table of Impact vs. Evaluation.
- 2. Chi-square test statistics and p-value.
- 3. Bar plot showing evaluation rates by impact level.

---

## Analysis

The experiment was successfully executed with corrected data mapping logic.

1. **Data Processing**:
   - The 'High Impact' category was correctly aggregated from 'Rights-
Impacting', 'Safety-Impacting', and 'Both', yielding 227 systems.
   - The 'Low/Moderate Impact' category ('Neither') yielded 1,491 systems.
   - Independent Evaluation controls were binary-coded based on 'Yes'/'TRUE'
values.

2. **Statistical Results**:
   - **Contingency Table**: High Impact systems showed 33 evaluations (14.5%),
while Low/Moderate systems showed only 20 (1.3%).
   - **Chi-Square Test**: The test yielded a statistic of 110.37 with a p-value
of 8.12e-26, indicating an extremely statistically significant relationship.

3. **Hypothesis Evaluation**:
   - The original hypothesis posited that High Impact systems are *not*
statistically more likely to possess Independent Evaluation controls.
   - **Result**: The hypothesis is **rejected**. High Impact systems are
significantly more likely (over 10x higher rate) to have independent evaluations
than Low Impact systems.
   - **Nuance**: While the statistical difference is strong, the *absolute* rate
of compliance for High Impact systems is still alarmingly low (14.5%),
confirming the broader concern about governance gaps, even if the relative
distribution behaves as expected.

---

## Review

The experiment was successfully executed. The code correctly mapped the sparse
and inconsistent values in the '17_impact_type' and '55_independent_eval'
columns to binary categories, enabling a valid statistical test.

**Findings:**
1.  **Hypothesis Rejection:** The hypothesis that High Impact systems are *not*
statistically more likely to possess Independent Evaluation controls was
**rejected**. The analysis shows a highly significant dependency (Chi-square =
110.37, p < 0.001).
2.  **Risk-Control Alignment:** Contrary to the hypothesis, High Impact systems
(Rights/Safety-impacting) are over **10 times more likely** to undergo
independent evaluation (14.5%) compared to Low/Moderate impact systems (1.3%).
3.  **Governance Gap:** While the relative difference contradicts the specific
'misalignment' hypothesis, the absolute figures reveal a major governance gap:
**85.5% of self-identified High Impact AI systems** in the federal inventory
still lack independent evaluation, despite it being a critical safeguard.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load dataset
print("Loading dataset...")
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 subset
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 subset shape: {df_eo.shape}")

# 1. Map Impact Type
# Logic: 'Neither' -> Low/Moderate; 'Both', 'Rights-Impacting', 'Safety-Impacting' -> High Impact
def map_impact(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    if val_str == 'Neither':
        return 'Low/Moderate Impact'
    # Check for keywords indicating high impact
    if val_str in ['Both', 'Rights-Impacting', 'Safety-Impacting', 'Safety-impacting']:
        return 'High Impact'
    return None

df_eo['impact_group'] = df_eo['17_impact_type'].apply(map_impact)

# Drop rows where impact is undefined
df_analysis = df_eo.dropna(subset=['impact_group']).copy()

# 2. Map Independent Eval
# Logic: Starts with 'Yes' or is 'TRUE' -> Yes; else -> No
def map_eval(val):
    if pd.isna(val):
        return 'No'
    val_str = str(val).strip().lower()
    if val_str.startswith('yes') or val_str == 'true':
        return 'Yes'
    return 'No'

df_analysis['has_indep_eval'] = df_analysis['55_independent_eval'].apply(map_eval)

# 3. Generate Contingency Table
contingency = pd.crosstab(df_analysis['impact_group'], df_analysis['has_indep_eval'])
print("\nContingency Table (Impact vs Independent Eval):")
print(contingency)

# Calculate percentages for display
print("\nPercentages:")
print(pd.crosstab(df_analysis['impact_group'], df_analysis['has_indep_eval'], normalize='index') * 100)

# 4. Statistical Test
if contingency.size >= 4:
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # 5. Visualization
    # Calculate positive rates
    rates = df_analysis.groupby('impact_group')['has_indep_eval'].apply(lambda x: (x=='Yes').mean())
    
    plt.figure(figsize=(8, 6))
    # Ensure consistent order
    order = ['High Impact', 'Low/Moderate Impact']
    try:
        rates = rates.reindex(order)
    except:
        pass
        
    ax = rates.plot(kind='bar', color=['#d62728', '#1f77b4'], rot=0)
    
    plt.title('Independent Evaluation Rate by AI System Impact Level')
    plt.ylabel('Proportion with Independent Evaluation')
    plt.xlabel('Impact Level')
    plt.ylim(0, max(rates.max() * 1.2, 0.1))  # Dynamic ylim
    
    # Add labels
    for p_rect in ax.patches:
        width = p_rect.get_width()
        height = p_rect.get_height()
        x, y = p_rect.get_xy() 
        ax.text(x + width/2, 
                y + height + 0.005, 
                f'{height:.1%}', 
                ha='center', 
                va='bottom',
                fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("Insufficient data dimensions for Chi-square test.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset...
EO 13960 subset shape: (1757, 196)

Contingency Table (Impact vs Independent Eval):
has_indep_eval         No  Yes
impact_group                  
High Impact           194   33
Low/Moderate Impact  1471   20

Percentages:
has_indep_eval              No        Yes
impact_group                             
High Impact          85.462555  14.537445
Low/Moderate Impact  98.658618   1.341382

Chi-square Statistic: 110.3715
P-value: 8.1243e-26


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Vertical Bar Plot.
*   **Purpose:** This plot compares a numerical variable (the proportion of independent evaluations) across two distinct categorical groups (High Impact vs. Low/Moderate Impact AI systems).

**2. Axes**
*   **X-axis:**
    *   **Title:** "Impact Level"
    *   **Categories:** Two distinct categories are displayed: "High Impact" and "Low/Moderate Impact".
*   **Y-axis:**
    *   **Title:** "Proportion with Independent Evaluation"
    *   **Range:** The axis ranges from **0.00 to 0.17** (approximate visual cap), with grid lines marked every **0.02 units** (0.00, 0.02, ... 0.16).
    *   **Units:** The axis uses decimal proportions (0.14 = 14%), though specific data points are annotated as percentages.

**3. Data Trends**
*   **Tallest Bar:** The "High Impact" category is the tallest, colored in red, reaching a value of **0.145** (or 14.5%).
*   **Shortest Bar:** The "Low/Moderate Impact" category is the shortest, colored in blue, reaching a value of **0.013** (or 1.3%).
*   **Pattern:** There is a drastic disparity between the two categories. The independent evaluation rate drops significantly as the impact level of the AI system decreases.

**4. Annotations and Legends**
*   **Value Labels:** Specific percentage values are annotated in bold text directly above each bar:
    *   **14.5%** above the High Impact bar.
    *   **1.3%** above the Low/Moderate Impact bar.
*   **Grid Lines:** Horizontal dashed grid lines are included to assist in estimating the height of the bars relative to the Y-axis.
*   **Color Coding:** The bars are distinct in color (Red for High Impact, Blue for Low/Moderate), emphasizing the contrast between the two groups.

**5. Statistical Insights**
*   **Significant Disparity:** High Impact AI systems are more than **11 times** more likely ($14.5 / 1.3 \approx 11.15$) to undergo independent evaluation compared to Low/Moderate Impact systems.
*   **Low Absolute Rates:** Despite the "High Impact" label, the absolute rate of independent evaluation is still quite low. Fewer than **1 in 6** (14.5%) high-impact systems are independently evaluated.
*   **Minimal Oversight for Lower Impact:** For Low/Moderate impact systems, independent evaluation is virtually non-existent at just 1.3%.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
