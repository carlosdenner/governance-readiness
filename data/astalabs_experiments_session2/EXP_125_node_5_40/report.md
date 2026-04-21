# Experiment 125: node_5_40

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_40` |
| **ID in Run** | 125 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:57:49.125089+00:00 |
| **Runtime** | 353.7s |
| **Parent** | `node_4_38` |
| **Children** | `node_6_18` |
| **Creation Index** | 126 |

---

## Hypothesis

> Autonomy-Harm Escalation: High-autonomy AI systems (e.g., 'System Selects and
Acts') are disproportionately associated with 'Tangible Harms' (Physical/Safety)
compared to Low-autonomy systems (e.g., 'Human-in-the-loop'), which are
associated with 'Intangible' or 'Civil Rights' harms.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.4176 (Maybe False) |
| **Surprise** | -0.3892 |
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
| Maybe False | 60.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Map autonomy levels to harm severity to justify stricter controls for autonomous agents.

### Steps
- 1. Filter for 'aiid_incidents'.
- 2. Clean '81_Autonomy Level' into ordinal bins (Low: Human-in-loop, High: Autonomous).
- 3. Clean '74_Tangible Harm' or '73_Harm Domain' into binary categories (Tangible/Physical vs Intangible/Societal).
- 4. Create a contingency table and run a Chi-square test or Fisher's Exact test.

### Deliverables
- Heatmap of Autonomy vs Harm Type and statistical test outputs.

---

## Analysis

The experiment successfully analyzed 159 incidents from the AIID dataset to test
the 'Autonomy-Harm Escalation' hypothesis.

1. **Data Distribution**: The dataset was skewed toward 'Low Autonomy' systems
(115 incidents) compared to 'High Autonomy' (44 incidents). Tangible harms were
more prevalent overall (58%).

2. **Statistical Results**:
   - **Directionality**: The data directionally supported the hypothesis. High
Autonomy systems had a higher probability of causing Tangible/Safety harms
(63.64%) compared to Low Autonomy systems (55.65%).
   - **Significance**: However, the Chi-square test yielded a **p-value of
0.4638**, well above the significant threshold of 0.05. The Fisher's Exact test
confirmed this (p=0.3768).

3. **Conclusion**: We **fail to reject the null hypothesis**. While there is a
slight increase in the rate of tangible harms for autonomous systems in this
sample, the difference is not statistically significant. The observed
association could be due to random chance. The 'Low Autonomy - Tangible Harm'
category remains the most frequent scenario (n=64), suggesting that human-in-
the-loop systems are still heavily implicated in physical safety incidents.

---

## Review

The experiment successfully tested the 'Autonomy-Harm Escalation' hypothesis
using the 'aiid_incidents' dataset after resolving significant data mapping
challenges (specifically regarding column names and the interpretation of
categorical codes for Autonomy and Harm). The final execution analyzed 159
incidents.

1. **Results**: The analysis found that 63.64% of High Autonomy incidents
(28/44) resulted in Tangible/Safety harms, compared to 55.65% of Low Autonomy
incidents (64/115).

2. **Statistical Significance**: The Chi-square test yielded a p-value of 0.4638
(Fisher's Exact p=0.3768), which is well above the significance threshold of
0.05.

3. **Conclusion**: We fail to reject the null hypothesis. There is no
statistically significant evidence in this sample to suggest that higher
autonomy levels are disproportionately associated with tangible or physical
harms compared to lower autonomy systems. The experiment was faithfully
implemented and the adaptive data cleaning strategy was rigorous.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

print("--- Data Inspection ---")
# Columns identified from previous steps
autonomy_col = 'Autonomy Level'
tangible_col = 'Tangible Harm'
intangible_col1 = 'Harm Distribution Basis'
intangible_col2 = 'Special Interest Intangible Harm'

# --- Step 1: Categorize Autonomy ---
# Autonomy3 -> High
# Autonomy1, Autonomy2 -> Low
# Unclear -> Drop
def map_autonomy(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower()
    if 'autonomy3' in val_str:
        return 'High Autonomy'
    elif 'autonomy1' in val_str or 'autonomy2' in val_str:
        return 'Low Autonomy'
    return None

aiid_df['Autonomy_Bin'] = aiid_df[autonomy_col].apply(map_autonomy)

# --- Step 2: Categorize Harm ---
# Tangible: 'tangible harm definitively occurred' OR 'imminent risk'
# Intangible: 'no tangible harm' AND (Intangible cols are populated/Yes)

def map_harm(row):
    t_val = str(row[tangible_col]).lower() if pd.notna(row[tangible_col]) else ''
    
    # Check Tangible
    if 'definitively occurred' in t_val or 'imminent risk' in t_val:
        return 'Tangible/Safety'
    
    # Check Intangible
    # If tangible is explicitly 'no' or 'issue' (non-imminent), check for intangible markers
    # We check if the intangible columns have meaningful content (not nan, not 'no')
    i1_val = str(row[intangible_col1]).lower() if pd.notna(row[intangible_col1]) else ''
    i2_val = str(row[intangible_col2]).lower() if pd.notna(row[intangible_col2]) else ''
    
    has_intangible = False
    if i1_val and i1_val not in ['nan', 'no', 'none', 'unclear']:
        has_intangible = True
    if i2_val and i2_val not in ['nan', 'no', 'none', 'unclear']:
        has_intangible = True
        
    if has_intangible:
        return 'Intangible/Societal'
        
    return None

aiid_df['Harm_Bin'] = aiid_df.apply(map_harm, axis=1)

# Drop unmapped rows
analysis_df = aiid_df.dropna(subset=['Autonomy_Bin', 'Harm_Bin'])

print(f"\nRows valid for analysis: {len(analysis_df)}")
print("Autonomy counts:\n", analysis_df['Autonomy_Bin'].value_counts())
print("Harm counts:\n", analysis_df['Harm_Bin'].value_counts())

if len(analysis_df) < 5:
    print("Insufficient data for statistical testing.")
else:
    # --- Step 3: Contingency Table & Stats ---
    contingency = pd.crosstab(analysis_df['Autonomy_Bin'], analysis_df['Harm_Bin'])
    print("\n--- Contingency Table ---")
    print(contingency)
    
    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    # Fisher's Exact (if 2x2)
    if contingency.shape == (2, 2):
        odds, fisher_p = fisher_exact(contingency)
        print(f"Fisher's Exact P-value: {fisher_p:.4f}")
        print(f"Odds Ratio: {odds:.4f}")
    
    # Calculate Conditional Probabilities
    print("\nConditional Probabilities:")
    for autonomy in contingency.index:
        total = contingency.loc[autonomy].sum()
        if 'Tangible/Safety' in contingency.columns:
            tangible_count = contingency.loc[autonomy, 'Tangible/Safety']
            prob = tangible_count / total
            print(f"P(Tangible Harm | {autonomy}) = {tangible_count}/{total} ({prob:.2%})")
        else:
             print(f"P(Tangible Harm | {autonomy}) = 0/{total} (0.00%)")

    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues')
    plt.title('Autonomy Level vs. Harm Type')
    plt.xlabel('Harm Category')
    plt.ylabel('Autonomy Level')
    plt.tight_layout()
    plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Data Inspection ---

Rows valid for analysis: 159
Autonomy counts:
 Autonomy_Bin
Low Autonomy     115
High Autonomy     44
Name: count, dtype: int64
Harm counts:
 Harm_Bin
Tangible/Safety        92
Intangible/Societal    67
Name: count, dtype: int64

--- Contingency Table ---
Harm_Bin       Intangible/Societal  Tangible/Safety
Autonomy_Bin                                       
High Autonomy                   16               28
Low Autonomy                    51               64

Chi-Square Statistic: 0.5368
P-value: 0.4638
Fisher's Exact P-value: 0.3768
Odds Ratio: 0.7171

Conditional Probabilities:
P(Tangible Harm | High Autonomy) = 28/44 (63.64%)
P(Tangible Harm | Low Autonomy) = 64/115 (55.65%)


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Heatmap (specifically a Confusion Matrix or Contingency Table visualization).
*   **Purpose:** The plot visualizes the frequency distribution or relationship between two categorical variables: "Autonomy Level" and "Harm Category." The color intensity represents the count of occurrences for each combination of categories.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Harm Category"
    *   **Labels:** The axis contains two categorical labels: "Intangible/Societal" and "Tangible/Safety".
*   **Y-Axis:**
    *   **Title:** "Autonomy Level"
    *   **Labels:** The axis contains two categorical labels: "High Autonomy" and "Low Autonomy".
*   **Color Scale (Z-Axis equivalent):**
    *   **Range:** The color bar on the right indicates a numerical scale ranging from approximately 15 (lightest blue) to 65 (darkest blue). The values represent counts.

### 3. Data Trends
*   **Highest Value (Darkest Blue):** The intersection of **"Low Autonomy"** and **"Tangible/Safety"** contains the highest count at **64**. This indicates that incidents or data points involving low autonomy systems resulting in tangible safety harms are the most frequent in this dataset.
*   **Lowest Value (Lightest Blue):** The intersection of **"High Autonomy"** and **"Intangible/Societal"** contains the lowest count at **16**.
*   **General Pattern:**
    *   **Autonomy:** There is a significantly higher number of counts associated with "Low Autonomy" compared to "High Autonomy" across both harm categories.
    *   **Harm Type:** "Tangible/Safety" harms are consistently more frequent than "Intangible/Societal" harms, regardless of the autonomy level.

### 4. Annotations and Legends
*   **Title:** "Autonomy Level vs. Harm Type" appears at the top, clearly defining the subject of the chart.
*   **Cell Annotations:** Each of the four cells contains a numerical label (16, 28, 51, 64) providing the exact count for that specific intersection, eliminating the need to estimate from the color bar.
*   **Color Bar:** Located on the right side, it provides a visual legend mapping the shade of blue to the numerical count, with ticks at 20, 30, 40, 50, and 60.

### 5. Statistical Insights
*   **Dominance of Low Autonomy:** "Low Autonomy" accounts for the majority of the data points ($51 + 64 = 115$), representing approximately **72%** of the total counts ($115/159$).
*   **Prevalence of Tangible Harms:** "Tangible/Safety" harms ($28 + 64 = 92$) are more common than "Intangible/Societal" harms ($16 + 51 = 67$), comprising roughly **58%** of the total.
*   **Interaction:** The dataset suggests that tangible safety issues in low-autonomy systems are the primary driver of the data, being four times more frequent ($64$) than intangible societal issues in high-autonomy systems ($16$).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
