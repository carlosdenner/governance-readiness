# Experiment 223: node_6_43

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_43` |
| **ID in Run** | 223 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T11:27:10.950281+00:00 |
| **Runtime** | 428.3s |
| **Parent** | `node_5_64` |
| **Children** | `node_7_21` |
| **Creation Index** | 224 |

---

## Hypothesis

> Autonomy-Safety Escalation: Incidents involving 'High Autonomy' systems are
significantly more likely to result in 'Physical Safety' harms compared to 'Low
Autonomy' systems, which skew towards 'Intangible' harms.

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

**Objective:** Test if higher autonomy levels correlate with physical risks.

### Steps
- 1. Filter 'aiid_incidents'.
- 2. Group `81_Autonomy Level` into 'High' (Autonomous/Automation) vs 'Low' (Augmentation/Assistance).
- 3. Group `73_Harm Domain` into 'Physical' vs 'Intangible'.
- 4. Perform Chi-square analysis.

### Deliverables
- Autonomy-Harm contingency table and statistical significance of the association.

---

## Analysis

The experiment successfully tested the 'Autonomy-Safety Escalation' hypothesis
on the 'aiid_incidents' dataset.

1. **Data Processing**: By implementing a keyword-based classification strategy
for 'Harm Type' (scanning titles/summaries) and mapping 'Autonomy Level'
metadata, the analysis successfully identified 78 usable records (21 High
Autonomy, 57 Low Autonomy).

2. **Statistical Results**: The Chi-square test yielded a p-value of **0.0772**
(Chi2 = 3.12). This exceeds the standard alpha of 0.05, indicating the results
are not statistically significant.

3. **Hypothesis Evaluation**: The hypothesis that High Autonomy predicts
Physical Harm was **rejected**. The observed data actually suggested an opposite
trend: High Autonomy systems were overwhelmingly associated with 'Intangible'
harms (86%, 18/21), whereas 'Low Autonomy' systems had a higher relative
proportion of 'Physical' harms (38%, 22/57).

4. **Conclusion**: The evidence does not support the claim that higher autonomy
escalates physical safety risks. Instead, high-autonomy failures in this dataset
are predominantly informational or sociotechnical (bias/privacy), while physical
accidents appear more frequently in lower-autonomy, human-in-the-loop contexts.

---

## Review

The experiment was successfully executed. The programmer correctly adapted to
data quality issues (sparse metadata in 'Autonomy Level' and 'Harm Domain') by
implementing a robust keyword-based classification strategy on incident
descriptions. This allowed for a valid statistical test (N=78) where previous
attempts failed. The analysis faithfully followed the plan to group autonomy
levels and harm types and perform a Chi-square test.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def run_experiment():
    try:
        # Load dataset
        filename = 'astalabs_discovery_all_data.csv'
        if os.path.exists(filename):
            file_path = filename
        elif os.path.exists(f'../{filename}'):
            file_path = f'../{filename}'
        else:
            print(f"Error: {filename} not found.")
            return

        print(f"Loading dataset from {file_path}...")
        df = pd.read_csv(file_path, low_memory=False)
        
        # Filter for AIID incidents
        aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
        print(f"AIID Incidents loaded: {len(aiid_df)} rows")
        
        # Identify useful columns
        # Autonomy
        autonomy_cols = [c for c in aiid_df.columns if 'autonomy' in str(c).lower() and 'level' in str(c).lower()]
        aut_col = autonomy_cols[0] if autonomy_cols else None
        
        # Text for keyword search (Description/Summary/Title)
        text_cols = [c for c in aiid_df.columns if c.lower() in ['description', 'summary', 'title', 'text', 'incident_description']]
        # Also include 'Tangible Harm' and 'Harm Distribution Basis' for context
        context_cols = [c for c in aiid_df.columns if 'harm' in str(c).lower()]
        
        search_cols = text_cols + context_cols
        print(f"Using Autonomy Column: {aut_col}")
        print(f"Using Text/Context Columns for Harm classification: {search_cols}")
        
        if not aut_col:
            print("Autonomy column not found.")
            return

        # --- MAPPING FUNCTIONS ---
        
        def get_autonomy(row):
            val = str(row[aut_col]).lower()
            if 'autonomy3' in val or 'autonomous' in val:
                return 'High'
            if 'autonomy1' in val or 'autonomy2' in val or 'assist' in val or 'augment' in val:
                return 'Low'
            return None

        def get_harm_type(row):
            # Combine text from all relevant columns
            text_content = " "
            for c in search_cols:
                if c in row and pd.notna(row[c]):
                    text_content += str(row[c]).lower() + " "
            
            # Keywords
            physical_keys = ['death', 'dead', 'kill', 'inju', 'hurt', 'crash', 'accident', 'collision', 'physical safety', 'burned', 'broke', 'fracture']
            intangible_keys = ['bias', 'discriminat', 'racis', 'sexi', 'gender', 'fairness', 'civil right', 'privacy', 'surveillance', 'reputation', 'economic', 'financial', 'credit', 'loan', 'arrest']
            
            # Classification Logic
            has_physical = any(k in text_content for k in physical_keys)
            has_intangible = any(k in text_content for k in intangible_keys)
            
            if has_physical:
                return 'Physical'
            elif has_intangible:
                return 'Intangible'
            else:
                return None

        # Apply mappings
        aiid_df['Autonomy_Bin'] = aiid_df.apply(get_autonomy, axis=1)
        aiid_df['Harm_Bin'] = aiid_df.apply(get_harm_type, axis=1)
        
        # Filter valid rows
        analysis_df = aiid_df.dropna(subset=['Autonomy_Bin', 'Harm_Bin'])
        print(f"\nRows mapped and ready for analysis: {len(analysis_df)}")
        print("Sample of mapped data:")
        print(analysis_df[['Autonomy_Bin', 'Harm_Bin']].head())
        
        if len(analysis_df) < 5:
            print("Insufficient data.")
            return

        # --- STATISTICAL ANALYSIS ---
        
        contingency = pd.crosstab(analysis_df['Autonomy_Bin'], analysis_df['Harm_Bin'])
        print("\n--- Contingency Table ---")
        print(contingency)
        
        # Chi-square
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"\nChi-Square Statistic: {chi2:.4f}")
        print(f"P-value: {p:.4f}")
        
        if p < 0.05:
            print("Result: Statistically SIGNIFICANT association (p < 0.05).")
        else:
            print("Result: NOT statistically significant (p >= 0.05).")
            
        # Visualization
        contingency.plot(kind='bar', stacked=False, figsize=(8, 6))
        plt.title('Harm Type by Autonomy Level')
        plt.xlabel('Autonomy Level')
        plt.ylabel('Incident Count')
        plt.legend(title='Harm Type')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
AIID Incidents loaded: 1362 rows
Using Autonomy Column: Autonomy Level
Using Text/Context Columns for Harm classification: ['description', 'title', 'summary', 'Alleged harmed or nearly harmed parties', 'Harm Domain', 'Tangible Harm', 'AI Harm Level', 'Harm Distribution Basis', 'Special Interest Intangible Harm', 'Intentional Harm', 'harm_type', 'primary_harm_types']

Rows mapped and ready for analysis: 78
Sample of mapped data:
     Autonomy_Bin    Harm_Bin
1758          Low    Physical
1759          Low    Physical
1760          Low  Intangible
1761         High    Physical
1762          Low  Intangible

--- Contingency Table ---
Harm_Bin      Intangible  Physical
Autonomy_Bin                      
High                  18         3
Low                   35        22

Chi-Square Statistic: 3.1231
P-value: 0.0772
Result: NOT statistically significant (p >= 0.05).


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is a detailed analysis:

### 1. Plot Type
*   **Type:** Grouped Bar Plot (or Clustered Bar Chart).
*   **Purpose:** This plot compares the frequency of "Incident Counts" across two categorical variables: "Autonomy Level" (High vs. Low) and "Harm Type" (Intangible vs. Physical). Grouping the bars allows for direct comparison of harm types within each autonomy level, as well as comparing across levels.

### 2. Axes
*   **X-axis:**
    *   **Title:** "Autonomy Level"
    *   **Categories:** Two discrete categories labeled "High" and "Low". The labels are oriented vertically.
*   **Y-axis:**
    *   **Title:** "Incident Count"
    *   **Range:** The axis ranges from 0 to 35, with major grid/tick marks at intervals of 5.
    *   **Units:** Integer count (number of incidents).

### 3. Data Trends
*   **Tallest and Shortest Bars:**
    *   The **tallest bar** represents "Intangible" harm in the "Low" autonomy category, with a count of approximately **35**.
    *   The **shortest bar** represents "Physical" harm in the "High" autonomy category, with a count of approximately **3**.
*   **Intangible vs. Physical Harm:**
    *   Across both autonomy levels, "Intangible" harm (blue bars) is consistently higher than "Physical" harm (orange bars).
*   **Autonomy Level Comparison:**
    *   **Low Autonomy:** Shows a significantly higher volume of total incidents compared to High Autonomy. The Intangible count is at the maximum of the scale (35), and Physical harm is substantial (approx. 22).
    *   **High Autonomy:** Shows fewer total incidents. While Intangible harm is moderate (approx. 18), Physical harm is nearly negligible (approx. 3).

### 4. Annotations and Legends
*   **Chart Title:** "Harm Type by Autonomy Level" centered at the top.
*   **Legend:** Located in the top right corner.
    *   **Title:** "Harm Type"
    *   **Keys:**
        *   **Blue:** Represents "Intangible" harm.
        *   **Orange:** Represents "Physical" harm.

### 5. Statistical Insights
*   **Prevalence of Intangible Harm:** Regardless of the system's autonomy level, incidents resulting in intangible harm are the dominant failure mode in this dataset.
*   **Safety Correlation with Autonomy:** There appears to be an inverse relationship between autonomy level and incident frequency. The "High" autonomy group is associated with fewer incidents overall (roughly 21 total) compared to the "Low" autonomy group (roughly 57 total). This could suggest that higher autonomy systems in this context are safer or perform better.
*   **Physical Safety Disparity:** The reduction in physical harm is particularly drastic when moving from Low to High autonomy. Physical incidents drop from ~22 in the Low setting to just ~3 in the High setting. This suggests high autonomy systems are particularly effective at mitigating physical risks compared to their lower autonomy counterparts.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
