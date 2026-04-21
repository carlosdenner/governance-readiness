# Experiment 27: node_3_13

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_13` |
| **ID in Run** | 27 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T02:21:01.709588+00:00 |
| **Runtime** | 308.4s |
| **Parent** | `node_2_6` |
| **Children** | `node_4_13`, `node_4_25` |
| **Creation Index** | 28 |

---

## Hypothesis

> The 'Autonomy-Severity' Escalation: Incidents involving systems with 'High'
autonomy levels are associated with a statistically significant increase in the
occurrence of 'Physical' harm compared to 'Low' autonomy systems, which
primarily cause 'Intangible' harm.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6210 (Maybe True) |
| **Posterior** | 0.2115 (Likely False) |
| **Surprise** | -0.4913 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 22.0 |
| Uncertain | 1.0 |
| Maybe False | 7.0 |
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

**Objective:** Test if higher AI autonomy correlates with more severe, physical harms.

### Steps
- 1. Filter 'aiid_incidents'.
- 2. Map '81_Autonomy Level' to two buckets: 'High' (e.g., 'High automation', 'Full') vs 'Low/Medium' (e.g., 'No automation', 'System assistance').
- 3. Map '74_Tangible Harm' to 'Physical' vs 'Non-Physical' (Economic, Intangible).
- 4. Perform a Chi-square test on the Autonomy vs. Harm Type matrix.

### Deliverables
- Proportion of Physical Harm events in High vs Low autonomy groups; Statistical test results.

---

## Analysis

The experiment successfully tested the 'Autonomy-Severity' Escalation hypothesis
using the AIID incidents dataset. After correcting the mapping logic to
interpret 'Autonomy3' as 'High' and 'Autonomy1/2' as 'Low', and filtering for
definitive harm outcomes, 182 valid incidents were analyzed. The results
revealed a striking lack of correlation between autonomy level and harm
severity. Incidents involving High Autonomy systems resulted in Physical harm
35.85% of the time, while Low Autonomy systems resulted in Physical harm 35.66%
of the time. The Chi-square test yielded a p-value of 1.0, statistically
confirming no significant association. Consequently, the hypothesis is rejected;
within this dataset, higher autonomy levels do not correlate with an increased
likelihood of physical harm compared to lower autonomy systems.

---

## Review

The experiment successfully tested the 'Autonomy-Severity' Escalation
hypothesis. After refining the data mapping to correctly interpret specific
dataset values ('Autonomy3' as High vs. 'Autonomy1/2' as Low; 'definitively
occurred' tangible harm as Physical), the analysis revealed 182 valid incidents.
Contrary to the hypothesis, there was no significant difference in the
proportion of Physical harm between High Autonomy systems (35.85%) and Low
Autonomy systems (35.66%). The Chi-square test resulted in a p-value of 1.0000,
definitively rejecting the hypothesis. The results suggest that within the AIID
dataset, the level of system autonomy is not a predictor for the physical
severity of harm.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def run_experiment():
    print("Starting Autonomy-Severity analysis (Attempt 2)...")
    
    # 1. Load the dataset
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

    # 2. Filter for AIID incidents
    aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"AIID Incidents loaded: {len(aiid)}")

    # 3. Identify correct columns
    # Based on previous exploration, we look for 'Autonomy Level' and 'Tangible Harm'
    # The dataset metadata suggests column names might be close to these.
    cols = aiid.columns
    autonomy_col = next((c for c in cols if 'Autonomy Level' in c), 'Autonomy Level')
    harm_col = next((c for c in cols if 'Tangible Harm' in c), 'Tangible Harm')
    
    print(f"Using columns: '{autonomy_col}' and '{harm_col}'")

    # 4. Map Autonomy Level
    # Previous findings: ['Autonomy1', 'Autonomy2', 'Autonomy3', 'unclear']
    def map_autonomy(val):
        s = str(val).strip()
        if s == 'Autonomy3':
            return 'High'
        elif s in ['Autonomy1', 'Autonomy2']:
            return 'Low'
        else:
            return np.nan  # Exclude 'unclear' or nan

    aiid['Autonomy_Bucket'] = aiid[autonomy_col].apply(map_autonomy)
    
    # 5. Map Tangible Harm
    # Previous findings: ['tangible harm definitively occurred', 'no tangible harm...', ...]
    # Hypothesis: Physical (Tangible) vs Intangible (Non-Physical)
    def map_harm(val):
        s = str(val).lower()
        if 'definitively occurred' in s:
            return 'Physical'  # Tangible harm happened
        elif 'no tangible harm' in s or 'issue' in s or 'risk' in s:
            return 'Intangible' # No tangible harm (economic, reputation, or near-miss)
        else:
            return np.nan # Exclude 'unclear'

    aiid['Harm_Bucket'] = aiid[harm_col].apply(map_harm)

    # 6. Drop NaNs in buckets
    valid_data = aiid.dropna(subset=['Autonomy_Bucket', 'Harm_Bucket'])
    print(f"Valid rows after mapping: {len(valid_data)}")
    
    if len(valid_data) == 0:
        print("No valid data found after mapping. Check values again.")
        print("Autonomy values:", aiid[autonomy_col].unique())
        print("Harm values:", aiid[harm_col].unique())
        return

    # 7. Generate Contingency Table
    ct = pd.crosstab(valid_data['Autonomy_Bucket'], valid_data['Harm_Bucket'])
    print("\nContingency Table (Count):")
    print(ct)

    # 8. Calculate Proportions
    # We want to see % of Physical harm in High vs Low autonomy
    props = pd.crosstab(valid_data['Autonomy_Bucket'], valid_data['Harm_Bucket'], normalize='index') * 100
    print("\nProportions (%):")
    print(props.round(2))

    # 9. Statistical Test (Chi-Square)
    if ct.shape == (2, 2):
        chi2, p, dof, ex = stats.chi2_contingency(ct)
        print(f"\nChi-square Statistic: {chi2:.4f}")
        print(f"P-value: {p:.4e}")
        
        alpha = 0.05
        if p < alpha:
            print("Result: Significant association found (p < 0.05).")
        else:
            print("Result: No significant association found (p >= 0.05).")
    else:
        print("\nContingency table is not 2x2, skipping Chi-square.")

    # 10. Visualization
    # Stacked bar chart
    ax = props.plot(kind='bar', stacked=True, color=['lightgray', 'firebrick'], figsize=(8, 6))
    plt.title('Proportion of Physical vs Intangible Harm by Autonomy Level')
    plt.ylabel('Percentage')
    plt.xlabel('Autonomy Level')
    plt.xticks(rotation=0)
    
    # Annotate bars
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center')
        
    plt.legend(title='Harm Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting Autonomy-Severity analysis (Attempt 2)...
AIID Incidents loaded: 1362
Using columns: 'Autonomy Level' and 'Tangible Harm'
Valid rows after mapping: 182

Contingency Table (Count):
Harm_Bucket      Intangible  Physical
Autonomy_Bucket                      
High                     34        19
Low                      83        46

Proportions (%):
Harm_Bucket      Intangible  Physical
Autonomy_Bucket                      
High                  64.15     35.85
Low                   64.34     35.66

Chi-square Statistic: 0.0000
P-value: 1.0000e+00
Result: No significant association found (p >= 0.05).


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** This chart illustrates the relative composition of two variables ("Physical" vs. "Intangible" harm) across distinct categories ("High" and "Low" Autonomy Level). It allows for an easy comparison of proportions rather than absolute counts.

### 2. Axes
*   **X-axis:**
    *   **Label:** "Autonomy Level"
    *   **Categories:** Two categorical values: "High" and "Low".
*   **Y-axis:**
    *   **Label:** "Percentage"
    *   **Range:** 0 to 100 (representing 0% to 100%).
    *   **Ticks:** Intervals of 20 (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Dominant Category:** In both autonomy scenarios, "Intangible" harm (represented by the gray bottom section) is the dominant category, comprising nearly two-thirds of the total.
*   **Minor Category:** "Physical" harm (represented by the red top section) consistently makes up slightly more than one-third of the total.
*   **Consistency:** The distribution is remarkably consistent between the two bars. The variation between the "High" and "Low" autonomy columns is only 0.1%, which is negligible.

### 4. Annotations and Legends
*   **Chart Title:** "Proportion of Physical vs Intangible Harm by Autonomy Level"
*   **Legend:** Located on the right side, titled "Harm Type":
    *   **Gray Square:** Represents "Intangible" harm.
    *   **Red Square:** Represents "Physical" harm.
*   **Value Labels:**
    *   **High Autonomy:** Intangible (64.2%), Physical (35.8%).
    *   **Low Autonomy:** Intangible (64.3%), Physical (35.7%).

### 5. Statistical Insights
*   **Prevalence of Intangible Harm:** Regardless of the autonomy level, intangible harm is significantly more common than physical harm in this dataset, occurring at a rate of approximately 64% compared to roughly 36% for physical harm.
*   **Independence of Variables:** The plot suggests a lack of correlation between "Autonomy Level" and the type of harm ("Harm Type"). The proportions are nearly identical (64.2% vs. 64.3%), indicating that increasing or decreasing the autonomy level does not shift the balance between physical and intangible harm risks.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
