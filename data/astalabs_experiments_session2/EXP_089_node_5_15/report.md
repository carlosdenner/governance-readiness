# Experiment 89: node_5_15

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_15` |
| **ID in Run** | 89 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:18:52.208124+00:00 |
| **Runtime** | 227.0s |
| **Parent** | `node_4_15` |
| **Children** | `node_6_31` |
| **Creation Index** | 90 |

---

## Hypothesis

> The 'Maturity Clustering' Effect: Governance capabilities are not isolated;
federal AI systems that have achieved a security 'Authority to Operate' (ATO)
are significantly more likely to also possess a documented 'Impact Assessment'
compared to systems without an ATO.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7823 (Likely True) |
| **Posterior** | 0.9258 (Definitely True) |
| **Surprise** | +0.1723 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

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
| Definitely True | 60.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Determine if security compliance (ATO) is a predictor of broader governance compliance (Impact Assessments) in federal AI.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'eo13960_scored'.
- 2. Clean '40_has_ato' into a binary variable (Yes vs No/Pending). Treat 'Yes' variants as 1, others as 0.
- 3. Clean '52_impact_assessment' into a binary variable (Documented vs Not Documented) using text analysis (exclude 'N/A', 'None').
- 4. Create a contingency table.
- 5. Perform a Chi-square test of independence.
- 6. Calculate the Odds Ratio to quantify the strength of the association.

### Deliverables
- Contingency table, Chi-square statistics, Odds Ratio, and a clustered bar chart.

---

## Analysis

The experiment successfully tested the 'Maturity Clustering' hypothesis. The
code loaded the dataset, correctly cleaned the 'Authority to Operate' (ATO) and
'Impact Assessment' columns based on the values identified in the debug phase,
and performed a Chi-square test of independence.

**Findings:**
1.  **Strong Association:** The data reveals a highly significant association
between security compliance (ATO) and governance compliance (Impact Assessment).
The Chi-square statistic is 63.26 with a p-value of 1.81e-15, overwhelmingly
rejecting the null hypothesis of independence.
2.  **Odds Ratio:** Systems with an ATO are approximately **10.9 times** more
likely to have a documented impact assessment than those without one.
3.  **Compliance Rates:**
    - **No ATO:** Only 0.80% (9/1119) have an impact assessment.
    - **Has ATO:** 8.15% (52/638) have an impact assessment.

The visualization confirms this disparity, showing a massive relative difference
despite the overall low absolute compliance rates.

---

## Review

The experiment was successfully executed and the hypothesis was rigorously
tested. The programmer correctly handled the file path issues identified in the
debug step and implemented robust data cleaning for the 'Authority to Operate'
(ATO) and 'Impact Assessment' columns. The use of a Chi-square test and Odds
Ratio provided a comprehensive statistical evaluation of the relationship
between security compliance and governance documentation.

**Hypothesis Evaluation**:
The 'Maturity Clustering' hypothesis is **strongly supported**.

**Key Findings**:
1.  **Statistical Significance**: There is an extremely strong, statistically
significant association between achieving an ATO and possessing a documented
Impact Assessment (Chi-square = 63.26, p < 0.001).
2.  **Magnitude of Effect**: The Odds Ratio of 10.94 indicates that systems with
an ATO are nearly **11 times more likely** to have an Impact Assessment than
those without one.
3.  **Absolute vs. Relative**: While the relative difference is massive, the
absolute compliance rates remain low. Only 8.15% of systems *with* an ATO have a
documented Impact Assessment, compared to a negligible 0.80% for those without.
This suggests that while ATO is a strong predictor of broader governance, the
overall maturity of federal AI systems regarding impact assessments is still
nascent.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Load the dataset
filename = 'astalabs_discovery_all_data.csv'
df = pd.read_csv(filename, low_memory=False)

# Filter for EO 13960 scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# 2. Clean '40_has_ato' (Authority to Operate)
# Positive values indicating an ATO or equivalent approved status
ato_positive = ['Yes', 'Operated in an approved enclave']

# Function to clean ATO column
def clean_ato(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip()
    if val_str in ato_positive:
        return 1
    return 0

eo_data['has_ato_binary'] = eo_data['40_has_ato'].apply(clean_ato)

# 3. Clean '52_impact_assessment'
# Positive values indicating a documented assessment
impact_positive = ['Yes', 'YES']

# Function to clean Impact Assessment column
def clean_impact(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip()
    if val_str in impact_positive:
        return 1
    return 0

eo_data['has_impact_binary'] = eo_data['52_impact_assessment'].apply(clean_impact)

# 4. Create Contingency Table
contingency_table = pd.crosstab(eo_data['has_ato_binary'], eo_data['has_impact_binary'])
contingency_table.index = ['No ATO', 'Has ATO']
contingency_table.columns = ['No Impact Assessment', 'Has Impact Assessment']

print("--- Contingency Table ---")
print(contingency_table)
print("\n")

# 5. Perform Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# 6. Calculate Odds Ratio
# OR = (a*d) / (b*c) where a=HasATO_HasImpact, b=HasATO_NoImpact, c=NoATO_HasImpact, d=NoATO_NoImpact
# But crosstab order is [0,0], [0,1], [1,0], [1,1]
# NoATO_NoImpact (0,0), NoATO_HasImpact (0,1)
# HasATO_NoImpact (1,0), HasATO_HasImpact (1,1)

n00 = contingency_table.iloc[0, 0] # No ATO, No Impact
n01 = contingency_table.iloc[0, 1] # No ATO, Has Impact
n10 = contingency_table.iloc[1, 0] # Has ATO, No Impact
n11 = contingency_table.iloc[1, 1] # Has ATO, Has Impact

# Use Haldane-Anscombe correction if any cell is 0 (add 0.5), though unlikely here with n=1757
if (n00==0 or n01==0 or n10==0 or n11==0):
    odds_ratio = ((n11 + 0.5) * (n00 + 0.5)) / ((n10 + 0.5) * (n01 + 0.5))
else:
    odds_ratio = (n11 * n00) / (n10 * n01)

print(f"--- Chi-Square Results ---")
print(f"Chi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Odds Ratio: {odds_ratio:.4f}")

# Calculate percentages for plotting
ato_compliance_rate = n11 / (n11 + n10) * 100
no_ato_compliance_rate = n01 / (n01 + n00) * 100

print(f"\nCompliance Rate (Has ATO): {ato_compliance_rate:.2f}%")
print(f"Compliance Rate (No ATO): {no_ato_compliance_rate:.2f}%")

# 7. Visualization
labels = ['No ATO', 'Has ATO']
rates = [no_ato_compliance_rate, ato_compliance_rate]

plt.figure(figsize=(8, 6))
plt.bar(labels, rates, color=['#e74c3c', '#2ecc71'], alpha=0.8)
plt.ylabel('Percentage with Documented Impact Assessment')
plt.title('Impact Assessment Compliance by ATO Status')
plt.ylim(0, max(rates) * 1.2)

for i, v in enumerate(rates):
    plt.text(i, v + 0.2, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Contingency Table ---
         No Impact Assessment  Has Impact Assessment
No ATO                   1110                      9
Has ATO                   586                     52


--- Chi-Square Results ---
Chi-square Statistic: 63.2583
P-value: 1.8130e-15
Odds Ratio: 10.9443

Compliance Rate (Has ATO): 8.15%
Compliance Rate (No ATO): 0.80%


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot (or Bar Chart).
*   **Purpose:** The plot compares the percentage of documented impact assessments between two distinct categorical groups: those with "No ATO" and those with "Has ATO". It is designed to visualize the disparity in compliance rates based on ATO status.

### 2. Axes
*   **X-Axis:**
    *   **Label/Categories:** The axis represents categorical data with two groups: **"No ATO"** and **"Has ATO"**.
    *   **Range:** Categorical.
*   **Y-Axis:**
    *   **Label:** "Percentage with Documented Impact Assessment".
    *   **Units:** Percentage (%).
    *   **Value Range:** The axis displays tick marks from **0 to 8**, with the vertical space extending slightly higher (likely to a maximum of 10) to accommodate the tallest bar.

### 3. Data Trends
*   **Tallest Bar:** The **"Has ATO"** category is represented by the tallest bar (green), indicating a higher value.
*   **Shortest Bar:** The **"No ATO"** category is represented by the shortest bar (red), indicating a very low value.
*   **Pattern:** There is a significant positive correlation between having an ATO and having a documented impact assessment. The visual difference suggests that entities with an ATO are much more likely to be compliant than those without.

### 4. Annotations and Legends
*   **Bar Labels:**
    *   Above the "No ATO" bar, the specific value **0.8%** is annotated in bold.
    *   Above the "Has ATO" bar, the specific value **8.2%** is annotated in bold.
*   **Color Coding:**
    *   **Red ("No ATO"):** Likely used to signify a "warning," "poor performance," or negative status regarding compliance.
    *   **Green ("Has ATO"):** Likely used to signify "good," "safe," or positive status relative to the other category.

### 5. Statistical Insights
*   **Relative Performance:** Entities that have an ATO ("Authority to Operate," typically) are more than **10 times (10.25x)** more likely to have a documented impact assessment compared to those without an ATO ($8.2\% / 0.8\% \approx 10.25$).
*   **Absolute Performance:** despite the "Has ATO" group performing significantly better relatively, the absolute compliance rate is still quite low. An **8.2%** compliance rate indicates that even among those with an ATO, **91.8%** of the population *lacks* a documented impact assessment.
*   **Conclusion:** While obtaining an ATO status is strongly associated with better documentation practices, the overall adherence to documenting impact assessments is critically low across both groups.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
