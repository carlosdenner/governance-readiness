# Experiment 103: node_5_25

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_25` |
| **ID in Run** | 103 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:59:48.817857+00:00 |
| **Runtime** | 181.3s |
| **Parent** | `node_4_13` |
| **Children** | `node_6_3` |
| **Creation Index** | 104 |

---

## Hypothesis

> The 'Legacy-Governance' Gap: AI systems currently in the 'Operation' phase are
significantly less likely to have a completed Impact Assessment than systems in
the 'Development' phase, reflecting a regulatory lag where older systems are
grandfathered out of compliance.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2706 (Likely False) |
| **Surprise** | -0.5656 |
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
| Definitely False | 54.0 |

---

## Experiment Plan

**Objective:** Determine if the lifecycle stage of an AI system predicts its compliance with Impact Assessment requirements.

### Steps
- 1. Load 'eo13960_scored'.
- 2. categorize '16_dev_stage' into two groups: 'Legacy/Ops' (Operation, Maintenance) vs. 'New/Dev' (Planning, Development).
- 3. Map '52_impact_assessment' to binary (Yes=1, No/Other=0).
- 4. Compute the percentage of Impact Assessments for both groups.
- 5. Use a Chi-square test to determine if the 'New/Dev' group has a statistically higher assessment rate.

### Deliverables
- Comparison of compliance rates between Lifecycle stages; Statistical test results.

---

## Analysis

The experiment successfully tested the 'Legacy-Governance Gap' hypothesis using
the EO 13960 dataset. The analysis categorized 992 AI systems into 'Legacy/Ops'
(641 systems) and 'New/Dev' (351 systems) based on their lifecycle stage.

The results statistically contradicted the hypothesis. The hypothesis predicted
that new systems would have higher compliance rates due to modern regulations,
while legacy systems would lag. The data revealed the opposite: 'Legacy/Ops'
systems had a compliance rate of 8.7% (56/641), whereas 'New/Dev' systems had a
near-zero compliance rate of 0.6% (2/351). The Chi-square test (p < 0.001)
confirmed this difference is statistically significant.

These findings indicate that Impact Assessments are not being performed during
the design/development phase (Privacy by Design), but rather—if performed at
all—are likely treated as a pre-deployment gate or retroactive paperwork for
operational systems. The 'Legacy Gap' does not exist; instead, there is a
'Development Void' where governance artifacts are almost entirely absent.

---

## Review

The experiment successfully tested the 'Legacy-Governance Gap' hypothesis using
the EO 13960 dataset. The analysis categorized 992 AI systems into 'Legacy/Ops'
(641 systems) and 'New/Dev' (351 systems) based on their lifecycle stage.

The results statistically contradicted the hypothesis. The hypothesis predicted
that new systems would have higher compliance rates due to modern regulations,
while legacy systems would lag. The data revealed the opposite: 'Legacy/Ops'
systems had a compliance rate of 8.7% (56/641), whereas 'New/Dev' systems had a
near-zero compliance rate of 0.6% (2/351). The Chi-square test (p < 0.001)
confirmed this difference is statistically significant.

These findings indicate that Impact Assessments are not being performed during
the design/development phase (Privacy by Design), but rather—if performed at
all—are likely treated as a pre-deployment gate or retroactive paperwork for
operational systems. The 'Legacy Gap' does not exist; instead, there is a
'Development Void' where governance artifacts are almost entirely absent.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import os

# Load the dataset
# Try potential paths
paths = ['../astalabs_discovery_all_data.csv', 'astalabs_discovery_all_data.csv']
file_path = None
for p in paths:
    if os.path.exists(p):
        file_path = p
        break

if file_path is None:
    print("Error: Dataset file not found.")
else:
    try:
        df = pd.read_csv(file_path, low_memory=False)
        
        # Filter for EO 13960 scored data
        eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
        print(f"Loaded {len(eo_df)} rows from EO 13960 dataset.")

        # --- Step 1: Categorize Lifecycle Stage (16_dev_stage) ---
        # Inspect unique values to ensure correct mapping
        unique_stages = eo_df['16_dev_stage'].unique()
        print(f"\nUnique values in '16_dev_stage' (first 10): {unique_stages[:10]}")

        def map_lifecycle(stage):
            if pd.isna(stage):
                return None
            stage_str = str(stage).lower()
            
            ops_keywords = ['operation', 'maintenance', 'use', 'implemented', 'production']
            dev_keywords = ['development', 'planning', 'acquisition', 'initiation', 'design', 'pilot']
            
            # Check for Ops keywords
            if any(k in stage_str for k in ops_keywords):
                return 'Legacy/Ops'
            # Check for Dev keywords
            elif any(k in stage_str for k in dev_keywords):
                return 'New/Dev'
            else:
                return 'Other'

        eo_df['lifecycle_group'] = eo_df['16_dev_stage'].apply(map_lifecycle)

        # Filter out 'Other' or None
        valid_lifecycle_df = eo_df[eo_df['lifecycle_group'].isin(['Legacy/Ops', 'New/Dev'])].copy()

        print(f"\nRows after filtering for valid lifecycle stages: {len(valid_lifecycle_df)}")
        print(valid_lifecycle_df['lifecycle_group'].value_counts())

        # --- Step 2: Categorize Impact Assessment (52_impact_assessment) ---
        # Inspect unique values
        unique_assessments = valid_lifecycle_df['52_impact_assessment'].unique()
        print(f"\nUnique values in '52_impact_assessment': {unique_assessments}")

        def map_assessment(val):
            if pd.isna(val):
                return 'No'
            val_str = str(val).lower()
            # Strict 'yes' check or explicit positive indicator
            if val_str == 'yes' or 'completed' in val_str:
                return 'Yes'
            return 'No'

        valid_lifecycle_df['has_assessment'] = valid_lifecycle_df['52_impact_assessment'].apply(map_assessment)

        # --- Step 3: Analysis ---

        # Contingency Table
        contingency = pd.crosstab(valid_lifecycle_df['lifecycle_group'], valid_lifecycle_df['has_assessment'])
        # Ensure columns exist even if one category is empty
        if 'Yes' not in contingency.columns:
            contingency['Yes'] = 0
        if 'No' not in contingency.columns:
            contingency['No'] = 0
            
        contingency = contingency[['No', 'Yes']] # Reorder
        
        print("\nContingency Table (Count):")
        print(contingency)

        # Calculate percentages
        results = contingency.copy()
        results['Total'] = results['No'] + results['Yes']
        results['Compliance Rate (%)'] = (results['Yes'] / results['Total']) * 100

        print("\nCompliance Rates by Lifecycle Stage:")
        print(results[['Total', 'Compliance Rate (%)']])

        # --- Step 4: Statistical Test (Chi-Square) ---
        chi2, p, dof, expected = chi2_contingency(contingency)

        print(f"\nChi-Square Test Results:")
        print(f"Chi2 Statistic: {chi2:.4f}")
        print(f"P-value: {p:.6f}")

        alpha = 0.05
        if p < alpha:
            print("Result: Statistically Significant.")
            rate_dev = results.loc['New/Dev', 'Compliance Rate (%)']
            rate_ops = results.loc['Legacy/Ops', 'Compliance Rate (%)']
            if rate_dev > rate_ops:
                print(f"Hypothesis Supported: New/Dev systems ({rate_dev:.1f}%) have higher compliance than Legacy/Ops ({rate_ops:.1f}%).")
            else:
                print(f"Hypothesis Contradicted: Legacy/Ops systems ({rate_ops:.1f}%) have higher compliance than New/Dev ({rate_dev:.1f}%).")
        else:
            print("Result: Not Statistically Significant. No evidence of difference in compliance.")

        # Visualization
        if not results.empty:
            ax = results['Compliance Rate (%)'].plot(kind='bar', color=['skyblue', 'salmon'], figsize=(8, 6))
            plt.title('Impact Assessment Compliance by Lifecycle Stage')
            plt.ylabel('Compliance Rate (%)')
            plt.xlabel('Lifecycle Stage')
            plt.ylim(0, 100)
            
            for i, v in enumerate(results['Compliance Rate (%)']):
                ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"An error occurred during execution: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded 1757 rows from EO 13960 dataset.

Unique values in '16_dev_stage' (first 10): <StringArray>
[ 'Implementation and Assessment', 'Acquisition and/or Development',
                      'Initiated',                        'Retired',
      'Operation and Maintenance',                  'In production',
                     'In mission',                        'Planned',
                              nan]
Length: 9, dtype: str

Rows after filtering for valid lifecycle stages: 992
lifecycle_group
Legacy/Ops    641
New/Dev       351
Name: count, dtype: int64

Unique values in '52_impact_assessment': <StringArray>
[nan, 'Planned or in-progress.', 'Yes', 'No', 'YES']
Length: 5, dtype: str

Contingency Table (Count):
has_assessment    No  Yes
lifecycle_group          
Legacy/Ops       585   56
New/Dev          349    2

Compliance Rates by Lifecycle Stage:
has_assessment   Total  Compliance Rate (%)
lifecycle_group                            
Legacy/Ops         641             8.736349
New/Dev            351             0.569801

Chi-Square Test Results:
Chi2 Statistic: 26.0141
P-value: 0.000000
Result: Statistically Significant.
Hypothesis Contradicted: Legacy/Ops systems (8.7%) have higher compliance than New/Dev (0.6%).


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** The plot is designed to compare the "Compliance Rate" percentages across two distinct categorical groups representing different stages of a project or product lifecycle ("Legacy/Ops" and "New/Dev").

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Compliance Rate (%)"
    *   **Range:** 0 to 100.
    *   **Scale:** Linear, with tick marks at intervals of 20 (0, 20, 40, 60, 80, 100).
*   **X-Axis:**
    *   **Label:** "Lifecycle Stage"
    *   **Categories:** Two distinct categories are displayed: "Legacy/Ops" and "New/Dev". The labels are oriented vertically (90 degrees) for readability.

### 3. Data Trends
*   **Tallest Bar:** The "Legacy/Ops" category (colored light blue) represents the highest value on the chart at **8.7%**.
*   **Shortest Bar:** The "New/Dev" category (colored salmon/light red) is significantly lower, barely visible on the scale, at **0.6%**.
*   **Pattern:** There is a drastic drop-off in compliance between the operational legacy stage and the new development stage. While the "Legacy/Ops" bar is visibly larger, both values occupy the very bottom fraction of the overall 0-100% scale.

### 4. Annotations and Legends
*   **Data Labels:** Both bars feature bold numerical annotations directly above them indicating the exact percentage values:
    *   Legacy/Ops: **8.7%**
    *   New/Dev: **0.6%**
*   **Color Coding:** The bars use distinct colors—light blue for "Legacy/Ops" and red/salmon for "New/Dev"—to visually distinguish the categories. No separate legend box is provided or necessary as the X-axis labels define the groups.

### 5. Statistical Insights
*   **Extremely Low Overall Compliance:** The most critical insight is that impact assessment compliance is exceptionally low across the board. Even the better-performing category (Legacy/Ops) has not reached 10% compliance, leaving a gap of over 90% to reach full compliance.
*   **Process Failure in Development:** The "New/Dev" stage has a near-zero compliance rate (0.6%). This suggests that impact assessments are being almost completely bypassed, ignored, or are not yet implemented during the new development phases.
*   **Relative Comparison:** Although both figures are low, the "Legacy/Ops" stage is performing significantly better in relative terms—it is roughly **14.5 times higher** than the "New/Dev" stage ($8.7 \div 0.6 \approx 14.5$). This suggests that governance or compliance checks are likely more established in older, operational systems than in new builds.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
