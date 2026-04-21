# Experiment 53: node_4_20

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_20` |
| **ID in Run** | 53 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:34:53.178696+00:00 |
| **Runtime** | 328.1s |
| **Parent** | `node_3_18` |
| **Children** | `node_5_4`, `node_5_21` |
| **Creation Index** | 54 |

---

## Hypothesis

> Development Transparency Gap: Federal AI systems in the 'Implemented' or
'Operational' stage are significantly more likely to provide public 'AI Notice'
(Control 59) compared to systems in 'Development' or 'Pilot' stages.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7823 (Likely True) |
| **Posterior** | 0.2665 (Likely False) |
| **Surprise** | -0.6189 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 5.0 |
| Maybe True | 25.0 |
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

**Objective:** Assess if operational maturity correlates with higher transparency (AI Notice compliance).

### Steps
- 1. Filter for 'eo13960_scored'.
- 2. Group '16_dev_stage' into 'Operational' (Implemented, Retired, Operation) and 'Pre-Operational' (Development, Planning, Pilot).
- 3. Convert '59_ai_notice' to binary.
- 4. Compare proportions using a Chi-square test.

### Deliverables
- Contingency table, p-value, visualization of compliance rates by stage.

---

## Analysis

The experiment successfully analyzed 1,753 federal AI systems from the EO 13960
inventory to test if operational maturity correlates with higher transparency
(AI Notice).

**Findings:**
- **Operational Systems:** 4.56% compliance (48 out of 1,053 systems provided
notice).
- **Pre-Operational Systems:** 3.29% compliance (23 out of 700 systems provided
notice).

**Statistical Test:**
- The Chi-square test yielded a statistic of 1.44 and a **p-value of 0.23**.
- The Odds Ratio was 1.41, suggesting a slight directional trend where
operational systems are more likely to provide notice, but this result is **not
statistically significant** (p > 0.05).

**Conclusion:**
The hypothesis that operational systems are *significantly* more likely to
provide AI Notice is **rejected**. While a small positive correlation exists,
the difference is not statistically robust. The most notable finding is the
uniformly low transparency across the federal inventory, with over 95% of
systems in both development and operational stages failing to provide public AI
notices.

---

## Review

The experiment was successfully executed and the code faithfully implemented the
analysis plan after correcting the mapping logic for the 'AI Notice' variable.
The results provide a clear test of the hypothesis.

**Hypothesis:** Development Transparency Gap: Federal AI systems in the
'Implemented' or 'Operational' stage are significantly more likely to provide
public 'AI Notice' compared to systems in 'Development' or 'Pilot' stages.

**Findings:**
1.  **Compliance Rates:** Public transparency is extremely low across the board.
Operational systems had a compliance rate of 4.56% (48/1053), while Pre-
Operational systems had a rate of 3.29% (23/700).
2.  **Statistical Test:** The Chi-square test yielded a p-value of 0.2301 (Chi2
= 1.44), which is greater than the standard significance level of 0.05.
3.  **Odds Ratio:** The Odds Ratio of 1.41 suggests a slight positive
association between operational maturity and transparency, but the lack of
statistical significance indicates this could be due to chance.

**Conclusion:** The hypothesis is **rejected**. There is no statistically
significant difference in transparency compliance between operational and pre-
operational AI systems in the federal inventory. The primary finding is a
systemic lack of public notice (95%+ non-compliance) regardless of development
stage.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import os

# Define file path
file_path = '../astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = 'astalabs_discovery_all_data.csv'

print(f"Loading dataset from: {file_path}")

try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print("Error: Dataset file not found.")
    exit(1)

# Filter for EO 13960 Scored dataset
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored rows loaded: {len(eo_data)}")

# --- Map Development Stage ---
def robust_map_dev_stage(val):
    if pd.isna(val):
        return None
    
    val_str = str(val).lower().strip()
    
    # Operational keywords
    op_keywords = [
        'implemented', 'implementation', 'operation', 'operational', 
        'retired', 'production', 'mission', 'use', 'deployed'
    ]
    if any(k in val_str for k in op_keywords):
        return 'Operational'
    
    # Pre-Operational keywords
    pre_keywords = [
        'development', 'planning', 'planned', 'pilot', 
        'research', 'design', 'initiated', 'acquisition'
    ]
    if any(k in val_str for k in pre_keywords):
        return 'Pre-Operational'
        
    return 'Other'

eo_data['stage_group'] = eo_data['16_dev_stage'].apply(robust_map_dev_stage)

# Filter for relevant groups
analysis_df = eo_data[eo_data['stage_group'].isin(['Operational', 'Pre-Operational'])].copy()
print(f"Rows retained for analysis: {len(analysis_df)}")

# --- Map AI Notice (Corrected Logic) ---
def map_notice_corrected(val):
    if pd.isna(val):
        return 0
    
    val_str = str(val).strip().lower()
    
    # Negative indicators (No notice provided)
    negative_indicators = [
        'none of the above', 
        'n/a', 
        'waived',
        'not safety',
        'nan'
    ]
    
    if any(neg in val_str for neg in negative_indicators):
        return 0
    
    # If it's not negative, it implies some form of notice (Online, Email, In-person, Other)
    return 1

analysis_df['notice_binary'] = analysis_df['59_ai_notice'].apply(map_notice_corrected)

# --- Generate Contingency Table ---
contingency_table = pd.crosstab(analysis_df['stage_group'], analysis_df['notice_binary'])

# Ensure columns 0 and 1 exist
contingency_table = contingency_table.reindex(columns=[0, 1], fill_value=0)
contingency_table.columns = ['No Notice', 'Has Notice']

print("\nContingency Table:")
print(contingency_table)

# --- Statistical Analysis ---
total_positives = contingency_table['Has Notice'].sum()

if total_positives == 0:
    print("\n[Analysis Outcome] No positive 'AI Notice' cases found even with corrected mapping.")
else:
    # Chi-Square Test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")

    # Compliance Rates
    compliance_rates = analysis_df.groupby('stage_group')['notice_binary'].mean()
    print("\nCompliance Rates (Proportion with AI Notice):")
    print(compliance_rates)

    # Odds Ratio
    try:
        op_yes = contingency_table.loc['Operational', 'Has Notice']
        op_no = contingency_table.loc['Operational', 'No Notice']
        pre_yes = contingency_table.loc['Pre-Operational', 'Has Notice']
        pre_no = contingency_table.loc['Pre-Operational', 'No Notice']
        
        if op_no == 0 or pre_no == 0:
            print("Warning: Zero count in denominator (No Notice). Odds Ratio undefined.")
        elif pre_yes == 0:
             print("Warning: Zero count in Pre-Operational Yes. Odds Ratio undefined.")
        else:
            odds_op = op_yes / op_no
            odds_pre = pre_yes / pre_no
            odds_ratio = odds_op / odds_pre
            print(f"Odds Ratio (Operational vs Pre-Operational): {odds_ratio:.4f}")
    except Exception as e:
        print(f"Could not calculate Odds Ratio: {e}")

    # --- Visualization ---
    plt.figure(figsize=(8, 6))
    stages = ['Operational', 'Pre-Operational']
    rates = [compliance_rates.get(s, 0) for s in stages]

    bars = plt.bar(stages, rates, color=['#4CAF50','#FFC107'])
    plt.title('AI Notice Compliance by Development Stage')
    plt.ylabel('Compliance Rate')
    plt.xlabel('Development Stage')
    plt.ylim(0, max(rates) * 1.2 if max(rates) > 0 else 0.1)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1%}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: astalabs_discovery_all_data.csv
EO 13960 Scored rows loaded: 1757
Rows retained for analysis: 1753

Contingency Table:
                 No Notice  Has Notice
stage_group                           
Operational           1005          48
Pre-Operational        677          23

Chi-Square Test Results:
Chi2 Statistic: 1.4404
P-value: 2.3008e-01

Compliance Rates (Proportion with AI Notice):
stage_group
Operational        0.045584
Pre-Operational    0.032857
Name: notice_binary, dtype: float64
Odds Ratio (Operational vs Pre-Operational): 1.4058


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot is designed to compare a quantitative variable ("Compliance Rate") across two distinct categorical groups ("Development Stage").

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Title:** "Development Stage"
    *   **Categories:** The axis displays two categories: "Operational" and "Pre-Operational".
*   **Y-Axis (Vertical):**
    *   **Title:** "Compliance Rate"
    *   **Range:** The scale ranges from 0.00 to 0.05.
    *   **Units:** The axis uses decimal notation (e.g., 0.04 corresponds to 4%), with tick marks at 0.01 intervals.

### 3. Data Trends
*   **Tallest Bar:** The green bar representing the **"Operational"** stage is the tallest, indicating the highest value in the dataset.
*   **Shortest Bar:** The yellow/gold bar representing the **"Pre-Operational"** stage is the shortest.
*   **Pattern:** There is a visible disparity between the two stages, with systems in the operational phase demonstrating higher compliance than those in the pre-operational phase.

### 4. Annotations and Legends
*   **Data Labels:** The specific values are annotated directly on top of each bar for clarity:
    *   **Operational:** 4.6%
    *   **Pre-Operational:** 3.3%
*   **Color Coding:** The bars are color-coded (Green for Operational, Yellow for Pre-Operational) to visually distinguish the categories, though no separate legend box is provided or necessary given the clear x-axis labels.

### 5. Statistical Insights
*   **Compliance Gap:** There is a **1.3 percentage point difference** between the two stages. Operational AI systems have a compliance rate of 4.6%, whereas pre-operational systems sit at 3.3%. This suggests that compliance efforts are likely prioritized or enforced more strictly once a system moves from development into active operation.
*   **Low Overall Compliance:** Despite the operational stage performing better relative to the pre-operational stage, the absolute compliance rates for both are extremely low (below 5%). This indicates that the vast majority of AI systems analyzed, regardless of their development stage, are failing to meet the "AI Notice Compliance" standards referenced in the study.
*   **Relative Comparison:** The compliance rate for Operational systems is approximately **39% higher** relative to the Pre-Operational baseline ($1.3 / 3.3 \approx 0.39$), which is a statistically meaningful difference despite the low base rates.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
