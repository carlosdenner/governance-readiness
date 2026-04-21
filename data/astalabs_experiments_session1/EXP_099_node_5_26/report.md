# Experiment 99: node_5_26

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_26` |
| **ID in Run** | 99 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:47:10.648974+00:00 |
| **Runtime** | 181.4s |
| **Parent** | `node_4_7` |
| **Children** | None |
| **Creation Index** | 100 |

---

## Hypothesis

> The 'GOVERN' function requirements are significantly more likely to map to
'Trust Readiness' competencies, while 'MAP', 'MEASURE', and 'MANAGE' functions
are significantly more likely to map to 'Integration Readiness', confirming the
strategic vs. operational split of the framework.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7742 (Likely True) |
| **Posterior** | 0.9421 (Definitely True) |
| **Surprise** | +0.1949 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 4.0 |
| Maybe True | 26.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 90.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** To validate the alignment between standard AI RMF functions and the derived Trust/Integration bundles.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Create a standardized function variable: group 'GOVERN' and 'GOVERNANCE' as 'Strategic', and 'MAP', 'MEASURE', 'MANAGE' as 'Operational'.
- 3. Create a contingency table of Standardized Function vs. Bundle.
- 4. Perform a Chi-square test of independence.
- 5. Visualize the relationship with a stacked bar chart.

### Deliverables
- 1. Contingency table.
- 2. Chi-square test results.
- 3. Stacked bar chart.

---

## Analysis

The experiment successfully validated the hypothesis that 'GOVERN' functions
align with 'Trust Readiness' while operational functions ('MAP', 'MEASURE',
'MANAGE') align with 'Integration Readiness'.

**Statistical Results:**
- **Dataset:** `step2_crosswalk_matrix.csv` (n=27 filtered records).
- **Chi-Square Test:** $\chi^2 = 5.65$, $p = 0.0175$.
- **Conclusion:** The relationship is statistically significant ($p < 0.05$),
rejecting the null hypothesis of independence.

**Distribution:**
- **Strategic Functions (GOVERN):** 91% (10/11) mapped to **Trust Readiness**,
strongly confirming the strategic nature of trust governance.
- **Operational Functions:** 62.5% (10/16) mapped to **Integration Readiness**,
confirming the engineering focus of mapping and measuring, though a significant
minority (37.5%) still involves trust competencies.

**Visualization:**
The stacked bar chart visually confirms this split, with the 'Strategic'
category overwhelmingly dominated by Trust Readiness, while the 'Operational'
category shows a clear but less exclusive preference for Integration Readiness.
This confirms the framework's structural logic: high-level governance requires
policy/trust competencies, while implementation requires integration/engineering
competencies.

---

## Review

The experiment successfully validated the hypothesis that 'GOVERN' functions
align significantly with 'Trust Readiness' while operational functions ('MAP',
'MEASURE', 'MANAGE') align with 'Integration Readiness'. The implementation
faithfully followed the plan by classifying the functions into 'Strategic' and
'Operational' categories and performing a Chi-square test on the filtered
dataset (n=27). The statistical results (Chi-square = 5.6485, p = 0.0175)
indicate a significant dependency between the framework function and the
readiness bundle, rejecting the null hypothesis. Specifically, Strategic
functions were overwhelmingly associated with Trust Readiness (91%, 10/11),
while Operational functions showed a preference for Integration Readiness
(62.5%, 10/16). The visualization and analysis confirm the strategic vs.
operational split of the framework.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Handle file loading with path check
filename = 'step2_crosswalk_matrix.csv'
file_path = f"../{filename}" if os.path.exists(f"../{filename}") else filename

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# 2. Create standardized function variable
# 'GOVERN' and 'GOVERNANCE' -> 'Strategic'
# 'MAP', 'MEASURE', 'MANAGE' -> 'Operational'
def classify_function(func):
    if pd.isna(func):
        return None
    func = str(func).upper().strip()
    if func in ['GOVERN', 'GOVERNANCE']:
        return 'Strategic'
    elif func in ['MAP', 'MEASURE', 'MANAGE']:
        return 'Operational'
    else:
        return 'Other'

df['function_category'] = df['function'].apply(classify_function)

# Filter for only Strategic and Operational to test the specific hypothesis
analysis_df = df[df['function_category'].isin(['Strategic', 'Operational'])].copy()

print(f"\nData filtered for analysis (n={len(analysis_df)}):")
print(analysis_df['function_category'].value_counts())

# 3. Create Contingency Table
contingency_table = pd.crosstab(analysis_df['function_category'], analysis_df['bundle'])

print("\nContingency Table (Observed):")
print(contingency_table)

# 4. Perform Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\n=== Chi-Square Test Results ===")
print(f"Chi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(expected)

# Interpretation
alpha = 0.05
if p < alpha:
    print("\nResult: Statistically Significant. The variables are dependent.")
else:
    print("\nResult: Not Statistically Significant. The variables are independent.")

# 5. Visualization
plt.figure(figsize=(10, 6))
# Calculate proportions for stacked bar
cross_tab_prop = pd.crosstab(index=analysis_df['function_category'],
                             columns=analysis_df['bundle'],
                             normalize="index")

cross_tab_prop.plot(kind='bar', stacked=True, color=['#4c72b0', '#55a868'], figsize=(10, 6))

plt.title('Proportion of Readiness Bundles by Framework Function Category')
plt.xlabel('Function Category')
plt.ylabel('Proportion')
plt.legend(title='Competency Bundle', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step2_crosswalk_matrix.csv

Data filtered for analysis (n=27):
function_category
Operational    16
Strategic      11
Name: count, dtype: int64

Contingency Table (Observed):
bundle             Integration Readiness  Trust Readiness
function_category                                        
Operational                           10                6
Strategic                              1               10

=== Chi-Square Test Results ===
Chi-square Statistic: 5.6485
P-value: 0.0175
Degrees of Freedom: 1
Expected Frequencies:
[[6.51851852 9.48148148]
 [4.48148148 6.51851852]]

Result: Statistically Significant. The variables are dependent.


=== Plot Analysis (figure 2) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Chart (specifically a 100% Stacked Bar Chart).
*   **Purpose:** The plot visualizes the relative composition of "Readiness Bundles" within two distinct "Function Categories." It allows for a direct comparison of the proportion of each bundle type across the different categories.

### 2. Axes
*   **X-axis:**
    *   **Label:** "Function Category"
    *   **Values:** Two categorical variables: "Operational" and "Strategic".
*   **Y-axis:**
    *   **Label:** "Proportion"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100%).
    *   **Units:** The values represent a fractional share of the whole (dimensionless ratio).

### 3. Data Trends
*   **Operational Category:**
    *   This category is dominated by the blue segment (**Integration Readiness**), which makes up approximately **62%** (0.62) of the bar.
    *   The green segment (**Trust Readiness**) comprises the remaining portion, approximately **38%**.
*   **Strategic Category:**
    *   This category shows a striking inversion compared to the Operational category. It is overwhelmingly dominated by the green segment (**Trust Readiness**).
    *   The green segment appears to take up over **90%** of the bar.
    *   The blue segment (**Integration Readiness**) is very minor, representing less than **10%** (approx. 0.09) of the total.

### 4. Annotations and Legends
*   **Title:** "Proportion of Readiness Bundles by Framework Function Category" appears at the top, clearly defining the chart's subject.
*   **Legend:** Located to the right of the plot, titled "**Competency Bundle**."
    *   **Blue Box:** Represents "Integration Readiness."
    *   **Green Box:** Represents "Trust Readiness."

### 5. Statistical Insights
*   **Inverse Relationship:** There is a clear inverse relationship between the function category and the type of readiness required.
*   **Operational Focus:** Operational functions rely heavily on **Integration Readiness**. This suggests that day-to-day operations are more dependent on the technical ability to connect and integrate systems or processes.
*   **Strategic Focus:** Strategic functions are almost entirely composed of **Trust Readiness**. This implies that high-level strategy and long-term planning are fundamentally driven by trust-based competencies rather than technical integration capabilities.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
