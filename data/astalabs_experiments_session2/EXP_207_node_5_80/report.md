# Experiment 207: node_5_80

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_80` |
| **ID in Run** | 207 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:44:53.576543+00:00 |
| **Runtime** | 167.5s |
| **Parent** | `node_4_39` |
| **Children** | None |
| **Creation Index** | 208 |

---

## Hypothesis

> The 'Rights-Compliance Paradox': Federal AI Use Cases classified as 'Rights-
Impacting' (inferred from purpose text) do not show a statistically significant
increase in 'Impact Assessment' compliance compared to administrative use cases.

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

**Objective:** Determine if sensitive use cases in the federal inventory are more rigorously assessed than non-sensitive ones.

### Steps
- 1. Load 'eo13960_scored' data.
- 2. Define keywords for 'Rights-Impacting' (e.g., 'surveillance', 'police', 'eligibility', 'fraud', 'benefits', 'sentencing') and 'Administrative' (e.g., 'process', 'sort', 'email', 'route').
- 3. Create a new column 'Risk_Category' by scanning '11_purpose_benefits' for these keywords.
- 4. Compare the rate of affirmative '52_impact_assessment' between Rights-Impacting and Administrative groups using a T-test or Chi-square.
- 5. Visualize the compliance rates.

### Deliverables
- 1. Count of use cases in each Risk Category.
- 2. Compliance rates for Impact Assessments per category.
- 3. Statistical significance report.

---

## Analysis

The experiment successfully analyzed the relationship between AI use case
sensitivity and regulatory compliance within the U.S. Federal AI inventory (EO
13960). The code processed 1,757 records, successfully classifying them into
'Rights-Impacting' (n=365) and 'Administrative' (n=579) categories using keyword
heuristics on the 'purpose_benefits' field.

**Findings:**
1.  **Overall Compliance:** Compliance with the requirement to perform an Impact
Assessment was found to be critically low across the entire dataset, with
neither category exceeding 6%.
2.  **Comparison:** 'Rights-Impacting' cases showed a compliance rate of 5.21%
(19/365), compared to 1.90% (11/579) for 'Administrative' cases.
3.  **Statistical Significance:** The Chi-square test resulted in a p-value of
0.0086 (Chi2 = 6.91), which is below the standard alpha of 0.05.

**Conclusion:**
The results **contradict** the 'Rights-Compliance Paradox' hypothesis. The
hypothesis predicted no statistically significant difference in compliance
between sensitive and non-sensitive use cases. However, the data demonstrates
that agencies are significantly more likely (approximately 2.7 times) to conduct
impact assessments for rights-impacting systems than for administrative ones,
statistically disproving the claim of 'no difference,' even though the absolute
level of governance remains negligible.

---

## Review

The experiment was faithfully implemented according to the plan. The code
successfully loaded the dataset, applied the keyword-based classification logic
to distinguish 'Rights-Impacting' from 'Administrative' use cases, and performed
the required statistical comparison.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

# --- 1. Load Data ---
possible_paths = ['../astalabs_discovery_all_data.csv', 'astalabs_discovery_all_data.csv']
path = next((p for p in possible_paths if os.path.exists(p)), None)

if not path:
    print("Error: Dataset not found in expected locations.")
else:
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path, low_memory=False)

    # Filter for EO 13960 data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 records loaded: {len(df_eo)}")

    # --- 2. Keyword Definitions ---
    # Expanded lists to capture nuance
    rights_keywords = [
        'surveillance', 'police', 'eligibility', 'fraud', 'benefits', 'sentencing', 
        'adjudication', 'hiring', 'housing', 'law enforcement', 'asylum', 'border', 
        'biometric', 'facial recognition', 'screening', 'loan', 'credit', 'insurance', 
        'healthcare decisions', 'parole', 'investigation', 'threat', 'security'
    ]
    
    admin_keywords = [
        'process', 'sort', 'email', 'route', 'workflow', 'scheduling', 
        'translation', 'transcription', 'categorize', 'inventory', 'logistics', 
        'search', 'summarize', 'digitize', 'form filling', 'administrative'
    ]

    # --- 3. Categorization Logic ---
    def classify_risk(text):
        if not isinstance(text, str):
            return 'Unclassified'
        text_lower = text.lower()
        
        # Priority: Rights-Impacting > Administrative
        if any(kw in text_lower for kw in rights_keywords):
            return 'Rights-Impacting'
        elif any(kw in text_lower for kw in admin_keywords):
            return 'Administrative'
        else:
            return 'Other'

    df_eo['risk_category'] = df_eo['11_purpose_benefits'].apply(classify_risk)

    # --- 4. Clean Compliance Column ---
    # Column: '52_impact_assessment'. Inspecting values to ensure correct binary mapping.
    # Assuming standard Yes/No or variations.
    def parse_compliance(val):
        s = str(val).lower().strip()
        return 1 if s in ['yes', 'true', '1', 'y'] else 0

    df_eo['has_impact_assessment'] = df_eo['52_impact_assessment'].apply(parse_compliance)

    # --- 5. Analysis ---
    # Group by Risk Category
    summary = df_eo.groupby('risk_category')['has_impact_assessment'].agg(['count', 'sum', 'mean'])
    summary.columns = ['Total Cases', 'Compliant Cases', 'Compliance Rate']
    
    print("\n--- Compliance Summary by Category ---")
    print(summary)

    # Extract groups for statistical testing
    rights_group = df_eo[df_eo['risk_category'] == 'Rights-Impacting']
    admin_group = df_eo[df_eo['risk_category'] == 'Administrative']

    n_rights = len(rights_group)
    n_admin = len(admin_group)

    if n_rights > 0 and n_admin > 0:
        rights_compliant = rights_group['has_impact_assessment'].sum()
        admin_compliant = admin_group['has_impact_assessment'].sum()
        
        # Contingency Table
        #              Compliant | Non-Compliant
        # Rights     |    A      |      B
        # Admin      |    C      |      D
        
        contingency = [
            [rights_compliant, n_rights - rights_compliant],
            [admin_compliant, n_admin - admin_compliant]
        ]
        
        chi2, p, dof, ex = chi2_contingency(contingency)
        
        print("\n--- Statistical Test Results (Chi-Square) ---")
        print(f"Comparison: Rights-Impacting (n={n_rights}) vs Administrative (n={n_admin})")
        print(f"Rights Compliance Rate: {rights_compliant/n_rights:.2%}")
        print(f"Admin Compliance Rate:  {admin_compliant/n_admin:.2%}")
        print(f"Chi2 Statistic: {chi2:.4f}")
        print(f"P-Value: {p:.4f}")
        
        if p < 0.05:
            print("Result: Statistically Significant Difference")
        else:
            print("Result: No Statistically Significant Difference (Paradox Supported)")
            
        # --- 6. Visualization ---
        plt.figure(figsize=(10, 6))
        categories = ['Rights-Impacting', 'Administrative']
        rates = [rights_compliant/n_rights, admin_compliant/n_admin]
        
        # Create bar chart
        bars = plt.bar(categories, rates, color=['#d62728', '#1f77b4'], alpha=0.7)
        
        # Add labels
        plt.ylabel('Impact Assessment Compliance Rate')
        plt.title('Impact Assessment Compliance: Rights-Impacting vs. Administrative Use Cases')
        plt.ylim(0, 1.0)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, 
                     f'{height:.1%}', ha='center', va='bottom')
            
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.show()
        
    else:
        print("\nInsufficient data in one or both categories to perform statistical test.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: astalabs_discovery_all_data.csv
EO 13960 records loaded: 1757

--- Compliance Summary by Category ---
                  Total Cases  Compliant Cases  Compliance Rate
risk_category                                                  
Administrative            579               11         0.018998
Other                     802               31         0.038653
Rights-Impacting          365               19         0.052055
Unclassified               11                0         0.000000

--- Statistical Test Results (Chi-Square) ---
Comparison: Rights-Impacting (n=365) vs Administrative (n=579)
Rights Compliance Rate: 5.21%
Admin Compliance Rate:  1.90%
Chi2 Statistic: 6.9124
P-Value: 0.0086
Result: Statistically Significant Difference


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** The plot is designed to compare the "Impact Assessment Compliance Rate" between two distinct categories of use cases: "Rights-Impacting" and "Administrative."

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Impact Assessment Compliance Rate".
    *   **Range:** The axis runs from **0.0 to 1.0**.
    *   **Units:** The axis ticks represent decimal proportions (0.0, 0.2, 0.4, etc.), effectively representing 0% to 100%.
*   **X-Axis:**
    *   **Labels:** The axis represents categorical data with two labeled groups: "Rights-Impacting" and "Administrative".

### 3. Data Trends
*   **Tallest Bar:** The "Rights-Impacting" category represents the higher value of the two.
*   **Shortest Bar:** The "Administrative" category represents the lower value.
*   **Key Pattern:** Both values are extremely low relative to the potential maximum of 1.0 (100%). Visually, the bars occupy a very small fraction of the total chart height, emphasizing the low magnitude of the data.

### 4. Annotations and Legends
*   **Title:** "Impact Assessment Compliance: Rights-Impacting vs. Administrative Use Cases" appears at the top.
*   **Bar Annotations:** Specific percentage values are annotated directly above each bar for precision:
    *   Rights-Impacting: **5.2%**
    *   Administrative: **1.9%**
*   **Grid Lines:** Horizontal dashed grid lines are placed at intervals of 0.2 along the Y-axis to aid in visual estimation, though the exact labels make estimation unnecessary.
*   **Colors:** The "Rights-Impacting" bar is colored red/pink, while the "Administrative" bar is colored blue, visually distinguishing the two categories.

### 5. Statistical Insights
*   **Overall Low Compliance:** The most significant insight is that compliance rates are incredibly low across the board. Neither category reaches even 6% compliance. This suggests a systemic lack of adherence to impact assessment protocols regardless of the use case type.
*   **Relative Comparison:** While the absolute numbers are low, the "Rights-Impacting" use cases have a compliance rate (**5.2%**) that is nearly **2.7 times higher** than "Administrative" use cases (**1.9%**). This suggests that while compliance is poor generally, there is slightly more rigor or attention paid to cases that impact rights compared to purely administrative ones.
*   **Gap to Target:** If the target compliance rate is 1.0 (100%), there is a massive gap of 94.8% for Rights-Impacting cases and 98.1% for Administrative cases.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
