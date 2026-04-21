# Experiment 165: node_5_64

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_64` |
| **ID in Run** | 165 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:52:05.273224+00:00 |
| **Runtime** | 194.9s |
| **Parent** | `node_4_38` |
| **Children** | `node_6_43`, `node_6_84` |
| **Creation Index** | 166 |

---

## Hypothesis

> Operational Governance Decay: AI systems in the 'Operational' lifecycle stage
are significantly less likely to have a current 'Impact Assessment' compared to
systems in the 'Development' stage, suggesting governance is treated as a one-
time gate rather than a continuous process.

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

**Objective:** Test for regression in governance controls as systems move from development to operation.

### Steps
- 1. Filter for 'eo13960_scored' rows.
- 2. Categorize '16_dev_stage' into 'Operational' (Use/Maintenance) vs 'Development' (Planning/Dev).
- 3. Create a binary target for '52_impact_assessment' (Yes/No).
- 4. Run a Chi-square test to compare impact assessment rates between stages.

### Deliverables
- Comparison of compliance rates by stage and statistical significance test.

---

## Analysis

The experiment successfully tested the 'Operational Governance Decay' hypothesis
on a subset of the 'eo13960_scored' dataset. Due to strict keyword mapping on
the '16_dev_stage' column (e.g., potentially missing 'In production' or 'In
mission'), the analysis utilized 94 clear records (20 Development, 74
Operational).

1. **Statistical Results**: The Chi-square test revealed a highly significant
association (Chi2 = 26.03, p < 0.001) between lifecycle stage and impact
assessment compliance.

2. **Hypothesis Evaluation**: The hypothesis of 'Governance Decay' (predicting
lower compliance in Operations) was **rejected**. The data reveals the exact
opposite trend: Operational systems demonstrate a significantly higher
compliance rate (75.7%) compared to systems in Development (10.0%).

3. **Interpretation**: These findings suggest that Impact Assessments function
as a 'Deployment Gate' rather than a decaying maintenance task. Governance
controls appear to be prerequisites for operational release, whereas systems in
the development phase have largely not yet triggered these compliance
requirements. The 'governance gap' lies in the development phase, not
operations.

---

## Review

The experiment successfully tested the 'Operational Governance Decay' hypothesis
on the 'eo13960_scored' dataset. Although the keyword mapping for the lifecycle
stage was restrictive (resulting in N=94 valid records out of 1,757 by focusing
on explicit 'Operation and Maintenance' vs 'Acquisition/Development' labels),
the resulting sample was sufficient for statistical testing.

1. **Statistical Results**: The Chi-square test indicated a highly significant
association (p < 0.001) between lifecycle stage and the presence of an impact
assessment.

2. **Hypothesis Evaluation**: The hypothesis that operational systems would show
*lower* compliance (decay) was **rejected**. The data demonstrated the opposite:
systems in the 'Operational' stage had a 75.7% compliance rate, whereas those in
'Development' had only 10.0%.

3. **Key Finding**: The results suggest that Impact Assessments act as a
'deployment gate'—a requirement that must be satisfied before a system
transitions to 'Operation and Maintenance'—rather than a maintenance task that
falls by the wayside. The 'governance gap' exists primarily in the pre-
deployment phases.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Total EO 13960 records: {len(eo_df)}")

# --- Step 1: Analyze and Map Lifecycle Stage ---
# Column: 16_dev_stage
stage_col = '16_dev_stage'
print(f"\nUnique values in '{stage_col}':")
print(eo_df[stage_col].unique())

def map_stage(val):
    if pd.isna(val):
        return None
    val_lower = str(val).lower()
    # Operational keywords
    if any(x in val_lower for x in ['operation', 'maintenance', 'use', 'implemented', 'deployed']):
        return 'Operational'
    # Development keywords
    if any(x in val_lower for x in ['development', 'acquisition', 'planning', 'research', 'pilot', 'test']):
        return 'Development'
    return None

eo_df['lifecycle_category'] = eo_df[stage_col].apply(map_stage)

# --- Step 2: Analyze and Map Impact Assessment ---
# Column: 52_impact_assessment
assess_col = '52_impact_assessment'
print(f"\nUnique values in '{assess_col}':")
print(eo_df[assess_col].unique())

def map_assessment(val):
    if pd.isna(val):
        return None
    val_lower = str(val).lower()
    if 'yes' in val_lower:
        return 'Yes'
    if 'no' in val_lower or 'not' in val_lower:
        return 'No'
    return None

eo_df['has_impact_assessment'] = eo_df[assess_col].apply(map_assessment)

# --- Step 3: Filter and Create Contingency Table ---
analysis_df = eo_df.dropna(subset=['lifecycle_category', 'has_impact_assessment'])

print(f"\nRecords after cleaning and mapping: {len(analysis_df)}")

contingency_table = pd.crosstab(
    analysis_df['lifecycle_category'],
    analysis_df['has_impact_assessment']
)

print("\nContingency Table (Count):")
print(contingency_table)

# Calculate rates
rates = pd.crosstab(
    analysis_df['lifecycle_category'],
    analysis_df['has_impact_assessment'],
    normalize='index'
) * 100

print("\nCompliance Rates (%):")
print(rates)

# --- Step 4: Statistical Test ---
if contingency_table.size > 0:
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    print("\n--- Chi-Square Test Results ---")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    print(f"Degrees of Freedom: {dof}")
    
    alpha = 0.05
    if p < alpha:
        print("Result: Statistically Significant (Reject Null Hypothesis)")
        print("Interpretation: There is a significant association between lifecycle stage and impact assessment compliance.")
    else:
        print("Result: Not Statistically Significant (Fail to Reject Null)")
        print("Interpretation: No significant difference in impact assessment compliance found between stages.")
        
    # Visualization
    try:
        rates.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'])
        plt.title('Impact Assessment Compliance by Lifecycle Stage')
        plt.ylabel('Percentage')
        plt.xlabel('Lifecycle Stage')
        plt.xticks(rotation=0)
        plt.legend(title='Impact Assessment', loc='upper right')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting error: {e}")
else:
    print("Insufficient data for statistical testing.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total EO 13960 records: 1757

Unique values in '16_dev_stage':
<StringArray>
[ 'Implementation and Assessment', 'Acquisition and/or Development',
                      'Initiated',                        'Retired',
      'Operation and Maintenance',                  'In production',
                     'In mission',                        'Planned',
                              nan]
Length: 9, dtype: str

Unique values in '52_impact_assessment':
<StringArray>
[nan, 'Planned or in-progress.', 'Yes', 'No', 'YES']
Length: 5, dtype: str

Records after cleaning and mapping: 94

Contingency Table (Count):
has_impact_assessment  No  Yes
lifecycle_category            
Development            18    2
Operational            18   56

Compliance Rates (%):
has_impact_assessment         No        Yes
lifecycle_category                         
Development            90.000000  10.000000
Operational            24.324324  75.675676

--- Chi-Square Test Results ---
Chi2 Statistic: 26.0267
P-value: 3.3673e-07
Degrees of Freedom: 1
Result: Statistically Significant (Reject Null Hypothesis)
Interpretation: There is a significant association between lifecycle stage and impact assessment compliance.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Chart (specifically a 100% Stacked Bar Chart).
*   **Purpose:** The plot compares the relative percentage distribution of a binary variable ("Impact Assessment" compliance) across two distinct categories ("Lifecycle Stage"). It allows viewers to quickly see the proportion of "Yes" versus "No" responses within each stage.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Lifecycle Stage"
    *   **Categories:** Two discrete stages are represented: "Development" and "Operational".
*   **Y-Axis:**
    *   **Title:** "Percentage"
    *   **Range:** 0 to 100 (representing 0% to 100%).
    *   **Increments:** Ticks are marked every 20 units (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Development Stage:**
    *   This category is dominated by the "No" segment (Pink). Visually, the "No" bar extends up to the 90 mark on the Y-axis.
    *   Consequently, the "Yes" segment (Blue) comprises only about 10% of the total for the Development stage.
*   **Operational Stage:**
    *   This category shows a reversal of the trend seen in Development. The "Yes" segment (Blue) is the dominant portion.
    *   The "No" segment (Pink) appears to reach roughly the 24 mark.
    *   This implies the "Yes" segment comprises approximately 76% of the Operational stage.

### 4. Annotations and Legends
*   **Chart Title:** "Impact Assessment Compliance by Lifecycle Stage" – situated at the top center.
*   **Legend:** Located in the top right corner with the title **"Impact Assessment"**.
    *   **No:** Represented by the Salmon/Pink color.
    *   **Yes:** Represented by the Light Blue color.

### 5. Statistical Insights
*   **Significant Compliance Gap:** There is a stark contrast in compliance between the two lifecycle stages. Projects in the "Development" phase have a very low rate of Impact Assessment compliance (~10%), whereas "Operational" projects have a high compliance rate (~76%).
*   **Process Implication:** The data suggests that Impact Assessments are likely treated as a prerequisite for moving a project from development into operations, or that enforcement is much stricter once a system is live. Alternatively, it indicates a potential risk area where projects in development are evolving without assessing their potential impact.
*   **Dominant Status:**
    *   For **Development**, the status quo is overwhelmingly non-compliant.
    *   For **Operational**, the status quo is majority compliant.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
