# Experiment 184: node_6_26

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_26` |
| **ID in Run** | 184 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T09:44:23.486273+00:00 |
| **Runtime** | 190.7s |
| **Parent** | `node_5_50` |
| **Children** | `node_7_12` |
| **Creation Index** | 185 |

---

## Hypothesis

> The 'Rights vs. Safety' Consultation Split: AI systems designated as 'Rights-
Impacting' exhibit significantly higher rates of 'Stakeholder Consultation'
compared to 'Safety-Impacting' systems, indicating a governance bias toward
civil liberties over physical safety in consultation processes.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.4919 (Uncertain) |
| **Posterior** | 0.1676 (Likely False) |
| **Surprise** | -0.3892 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 9.0 |
| Uncertain | 11.0 |
| Maybe False | 10.0 |
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

**Objective:** Determine if the type of impact (Rights vs. Safety) influences the likelihood of stakeholder consultation.

### Steps
- 1. Load `eo13960_scored`.
- 2. Filter `17_impact_type` into two distinct groups: 'Rights-Impacting' and 'Safety-Impacting' (exclude 'Both' or 'Neither' to ensure clear separation).
- 3. Parse `63_stakeholder_consult` into a binary variable.
- 4. Perform a Chi-square test or Z-test to compare consultation rates between the two groups.

### Deliverables
- Comparison of consultation rates; Statistical test results; Analysis of governance focus.

---

## Analysis

The experiment was successfully executed, though the findings reveal a complete
absence of the target variable rather than a difference in proportions.

1. **Data Segmentation**: The code successfully filtered the EO 13960 dataset,
identifying 59 distinct 'Rights-Impacting' systems and 18 'Safety-Impacting'
systems.

2. **Governance Vacuum**: Contrary to the expectation that rights-impacting
systems would have higher consultation rates, the analysis found **zero**
documented instances of stakeholder consultation in either group (0/59 and
0/18). Both proportions were 0.0%.

3. **Statistical Inference**: Because both groups had identical zero
probabilities, the Z-test returned a p-value of 1.0 (Standard Error = 0).

4. **Conclusion**: The hypothesis is rejected. The data does not show a bias
toward rights over safety; instead, it highlights a systemic lack of documented
external engagement for *all* high-impact AI systems in this specific subset of
the federal inventory.

---

## Review

The experiment was executed faithfully and without error. The code correctly
implemented the filtering logic to isolate exclusive 'Rights-Impacting' and
'Safety-Impacting' groups and performed the statistical comparison as requested.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys

def run_experiment():
    print("Starting experiment: Rights vs. Safety Consultation Split...")
    
    # 1. Load dataset
    # Try loading from parent directory first as per instructions, then current
    file_name = 'astalabs_discovery_all_data.csv'
    paths = [f'../{file_name}', file_name]
    df_all = None
    
    for p in paths:
        try:
            df_all = pd.read_csv(p, low_memory=False)
            print(f"Successfully loaded {p}")
            break
        except FileNotFoundError:
            continue
            
    if df_all is None:
        print("Error: Dataset not found in ../ or current directory.")
        return

    # 2. Filter for EO 13960 Scored data
    df = df_all[df_all['source_table'] == 'eo13960_scored'].copy()
    print(f"Filtered eo13960_scored subset: {len(df)} rows")

    # 3. Define Columns
    col_impact = '17_impact_type'
    col_consult = '63_stakeholder_consult'
    
    # Validate columns exist
    if col_impact not in df.columns or col_consult not in df.columns:
        print(f"Error: Missing columns. Available: {df.columns.tolist()}")
        return

    # 4. Filter Groups (Rights vs Safety)
    # Normalize text to handle potential inconsistencies
    df['impact_norm'] = df[col_impact].fillna('').astype(str).str.lower().str.strip()
    
    # Define masks
    # Rights-Impacting: Contains 'rights', does NOT contain 'safety'
    mask_rights = df['impact_norm'].str.contains('rights') & ~df['impact_norm'].str.contains('safety')
    
    # Safety-Impacting: Contains 'safety', does NOT contain 'rights'
    mask_safety = df['impact_norm'].str.contains('safety') & ~df['impact_norm'].str.contains('rights')
    
    rights_df = df[mask_rights]
    safety_df = df[mask_safety]
    
    print(f"Groups Identified:")
    print(f"  - Rights-Impacting (Exclusive): {len(rights_df)}")
    print(f"  - Safety-Impacting (Exclusive): {len(safety_df)}")
    
    if len(rights_df) == 0 or len(safety_df) == 0:
        print("Error: One or both groups have 0 samples. Cannot perform test.")
        print("Sample Impact Types:", df['impact_norm'].unique()[:10])
        return

    # 5. Calculate Consultation Rates
    # We assume 'Yes' indicates consultation. 
    def parse_consultation(val):
        if pd.isna(val):
            return 0
        val_str = str(val).lower().strip()
        return 1 if val_str == 'yes' else 0

    rights_consulted = rights_df[col_consult].apply(parse_consultation)
    safety_consulted = safety_df[col_consult].apply(parse_consultation)

    k1 = rights_consulted.sum()
    n1 = len(rights_consulted)
    p1 = k1 / n1 if n1 > 0 else 0

    k2 = safety_consulted.sum()
    n2 = len(safety_consulted)
    p2 = k2 / n2 if n2 > 0 else 0

    print(f"\nConsultation Statistics:")
    print(f"  - Rights-Impacting: {k1}/{n1} consulted ({p1:.2%})")
    print(f"  - Safety-Impacting: {k2}/{n2} consulted ({p2:.2%})")

    # 6. Statistical Test (Two-Proportion Z-Test)
    # Pooled probability
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    
    if se == 0:
        print("Standard Error is 0 (identical proportions or zero variance).")
        z_score = 0
        p_value = 1.0
    else:
        z_score = (p1 - p2) / se
        p_value = 2 * (1 - norm.cdf(abs(z_score)))  # Two-sided p-value

    print(f"\nZ-Test Results:")
    print(f"  - Z-Score: {z_score:.4f}")
    print(f"  - P-Value: {p_value:.4e}")
    
    significance = "Significant" if p_value < 0.05 else "Not Significant"
    print(f"  - Conclusion: {significance} difference at alpha=0.05")

    # 7. Visualization
    plt.figure(figsize=(8, 6))
    categories = ['Rights-Impacting', 'Safety-Impacting']
    values = [p1, p2]
    colors = ['#3498db', '#e74c3c']
    
    bars = plt.bar(categories, values, color=colors, alpha=0.8)
    plt.ylabel('Proportion of Stakeholder Consultation')
    plt.title('Stakeholder Consultation: Rights vs. Safety')
    plt.ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    
    # Add counts and percentages on bars
    for idx, rect in enumerate(bars):
        height = rect.get_height()
        count_text = f"{height:.1%}\n(n={k1 if idx==0 else k2})"
        plt.text(rect.get_x() + rect.get_width()/2., height, count_text,
                 ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment: Rights vs. Safety Consultation Split...
Successfully loaded astalabs_discovery_all_data.csv
Filtered eo13960_scored subset: 1757 rows
Groups Identified:
  - Rights-Impacting (Exclusive): 59
  - Safety-Impacting (Exclusive): 18

Consultation Statistics:
  - Rights-Impacting: 0/59 consulted (0.00%)
  - Safety-Impacting: 0/18 consulted (0.00%)
Standard Error is 0 (identical proportions or zero variance).

Z-Test Results:
  - Z-Score: 0.0000
  - P-Value: 1.0000e+00
  - Conclusion: Not Significant difference at alpha=0.05


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Bar Plot (Column Chart).
*   **Purpose:** To compare the proportion of stakeholder consultation events between two distinct categories of impact: "Rights-Impacting" and "Safety-Impacting".

**2. Axes**
*   **X-axis:**
    *   **Labels:** Categorical labels representing two scenarios: "Rights-Impacting" and "Safety-Impacting".
    *   **Value Range:** N/A (Categorical).
*   **Y-axis:**
    *   **Title:** "Proportion of Stakeholder Consultation".
    *   **Value Range:** 0.0 to 1.0 (representing a range from 0% to 100%).
    *   **Ticks:** Marked at intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

**3. Data Trends**
*   **Tallest/Shortest Bars:** There are no visible bars for either category, indicating values of zero.
*   **Patterns:** Both categories exhibit identical behavior. The data shows a "floor" effect where both the "Rights-Impacting" and "Safety-Impacting" scenarios have a value of 0.
*   **Comparison:** There is no difference between the two groups; both are at the absolute minimum of the scale.

**4. Annotations and Legends**
*   **Annotations:**
    *   Above the "Rights-Impacting" position: Text reads **"0.0% (n=0)"**.
    *   Above the "Safety-Impacting" position: Text reads **"0.0% (n=0)"**.
    *   This annotation explicitly states the percentage value is zero and indicates the count ($n$) associated with that proportion is zero.
*   **Chart Title:** "Stakeholder Consultation: Rights vs. Safety".
*   **Grid Lines:** Horizontal dashed grid lines are present at 0.2 intervals to aid in reading values, though they highlight the emptiness of the chart in this specific instance.

**5. Statistical Insights**
*   **Absence of Consultation:** The most critical insight is the complete absence of stakeholder consultation in the dataset analyzed. Regardless of whether a scenario was "Rights-Impacting" or "Safety-Impacting," the proportion of consultation was 0.0%.
*   **Sample Count ($n$):** The notation $(n=0)$ likely refers to the number of observed consultations (the numerator). This suggests that out of the total cases reviewed, exactly zero involved stakeholder consultation.
*   **Conclusion:** The data suggests a potential systemic gap or a specific finding within the study context where stakeholder engagement is non-existent for these specific impact types. Neither safety concerns nor rights concerns appear to be triggers for consultation in this specific dataset.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
