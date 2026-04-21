# Experiment 133: node_7_0

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_0` |
| **ID in Run** | 133 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:19:18.315991+00:00 |
| **Runtime** | 347.6s |
| **Parent** | `node_6_5` |
| **Children** | `node_8_3` |
| **Creation Index** | 134 |

---

## Hypothesis

> The 'Vendor Opacity' Effect: Commercial-off-the-shelf (COTS) AI systems acquired
by federal agencies are significantly less likely to undergo 'Independent
Evaluation' compared to Custom-developed systems, creating a governance blind
spot for purchased technology.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8306 (Likely True) |
| **Posterior** | 0.2830 (Likely False) |
| **Surprise** | -0.6572 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 11.0 |
| Maybe True | 19.0 |
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

**Objective:** Compare independent evaluation rates between Commercial and Custom AI systems.

### Steps
- 1. Load the 'eo13960_scored' dataset.
- 2. Filter for rows where '10_commercial_ai' and '55_independent_eval' are not null.
- 3. Categorize '10_commercial_ai' into 'Commercial' (Yes) and 'Custom' (No).
- 4. Convert '55_independent_eval' to binary (Yes=1, No=0).
- 5. Perform a Chi-Square test of independence.
- 6. Visualize the evaluation rates with a bar chart.

### Deliverables
- Contingency table; Chi-Square statistic and p-value; Bar chart of Independent Evaluation rates by procurement type.

---

## Analysis

The experiment was successfully executed using the '22_dev_method' column to
distinguish between Commercial (Vendor-developed) and Custom (In-house) systems,
identifying 92 valid cases with non-null evaluation data.

1. **Findings**: The analysis revealed nearly identical rates of Independent
Evaluation between the two groups. Commercial systems had a compliance rate of
**46.0%** (23/50), while Custom systems had a rate of **50.0%** (21/42).

2. **Statistical Significance**: The Chi-Square test yielded a statistic of
0.0300 and a **p-value of 0.8626**, which is well above the 0.05 threshold.

3. **Conclusion (Hypothesis Rejected)**: The results **falsify** the 'Vendor
Opacity' hypothesis. There is no statistically significant evidence that
commercial/COTS systems are less likely to undergo independent evaluation than
custom in-house systems. Instead, the data suggests that governance rigor (or
lack thereof) is uniform across procurement types, with both groups hovering
around a 50% evaluation rate for the subset of reported cases.

---

## Review

The experiment was successfully executed and the code correctly adapted to data
limitations by selecting a more robust proxy variable ('22_dev_method') for the
cohort analysis when the original variable ('10_commercial_ai') proved
insufficient. The statistical analysis was rigorous, and the interpretation of
the results is sound.

**Hypothesis:** The 'Vendor Opacity' hypothesis—that Commercial/COTS systems are
less likely to undergo independent evaluation than Custom systems—was
**rejected**.

**Experiment Results:**
1. **Cohorts:** Using 'Development Method' as the differentiator, the analysis
identified 92 cases with reported evaluation status, split between
Commercial/Contractor-developed (n=50) and Custom/In-House (n=42).
2. **Evaluation Rates:** The rates of independent evaluation were statistically
indistinguishable: 46.0% for Commercial systems (23/50) versus 50.0% for Custom
systems (21/42).
3. **Significance:** The Chi-Square test yielded a p-value of 0.8626, indicating
no statistically significant difference.

**Findings:** Contrary to the expectation that vendor-acquired technology might
bypass governance scrutiny ('opacity'), the data suggests that federal agencies
apply independent evaluation protocols (or fail to apply them) at uniform rates
regardless of whether the system is built in-house or procured from vendors.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import sys

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# Define Columns based on debug findings
col_method = '22_dev_method'  # Proxy for Commercial vs Custom
col_eval = '55_independent_eval'

# Filter for rows where both relevant columns are not null (per instructions)
df_analysis = df_eo.dropna(subset=[col_method, col_eval]).copy()

# Define Categories for Procurement Method
# Mapping 'Developed with contracting resources' -> Commercial
# Mapping 'Developed in-house' -> Custom
# Excluding 'Both' to ensure distinct groups for the hypothesis
target_methods = ['Developed with contracting resources.', 'Developed in-house.']
df_analysis = df_analysis[df_analysis[col_method].isin(target_methods)].copy()

df_analysis['Procurement_Type'] = df_analysis[col_method].map({
    'Developed with contracting resources.': 'Commercial',
    'Developed in-house.': 'Custom'
})

# Define Binary Target for Evaluation
# 'Yes...' and 'TRUE' -> 1
# 'Planned', 'Waived', 'Does not apply', etc. -> 0
def is_evaluated(val):
    s = str(val).strip()
    if s.upper() == 'TRUE':
        return 1
    if s.startswith('Yes'):
        return 1
    return 0

df_analysis['Has_Eval'] = df_analysis[col_eval].apply(is_evaluated)

# Generate Contingency Table
contingency = pd.crosstab(df_analysis['Procurement_Type'], df_analysis['Has_Eval'])

# Check if data remains
if contingency.empty:
    print("No valid data intersection found between Development Method and Independent Evaluation.")
    sys.exit(0)

# Ensure columns exist (0 and 1)
if 0 not in contingency.columns: contingency[0] = 0
if 1 not in contingency.columns: contingency[1] = 0
contingency = contingency[[0, 1]]
contingency.columns = ['No Eval', 'Has Eval']

# Statistics
rates = df_analysis.groupby('Procurement_Type')['Has_Eval'].mean()
counts = df_analysis['Procurement_Type'].value_counts()
chi2, p, dof, expected = chi2_contingency(contingency)

# Output
print("--- EXPERIMENT RESULTS: Vendor Opacity Effect ---")
print(f"Data Source: EO 13960 Scored (Filtered for non-null responses)")
print(f"Total valid cases analyzed: {len(df_analysis)}")

print("\nCounts by Procurement Type:")
print(counts)

print("\nContingency Table (Eval Status):")
print(contingency)

print("\nEvaluation Rates:")
for pt in ['Custom', 'Commercial']:
    if pt in rates:
        print(f"  {pt}: {rates[pt]:.2%} ({contingency.loc[pt, 'Has Eval']}/{counts[pt]}) evaluated")

print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-Value: {p:.4e}")

result_text = "SIGNIFICANT" if p < 0.05 else "NOT SIGNIFICANT"
print(f"\nResult: {result_text}")

# Plot
plt.figure(figsize=(8, 6))
bars = plt.bar(rates.index, rates.values, color=['skyblue', 'orange'])
plt.title(f'Independent Evaluation Rates: Commercial (Vendor) vs Custom (In-House)\n(p={p:.4e})')
plt.ylabel('Proportion with Independent Evaluation')
plt.ylim(0, 1.0)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, 
             f'{height:.1%}', 
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- EXPERIMENT RESULTS: Vendor Opacity Effect ---
Data Source: EO 13960 Scored (Filtered for non-null responses)
Total valid cases analyzed: 92

Counts by Procurement Type:
Procurement_Type
Commercial    50
Custom        42
Name: count, dtype: int64

Contingency Table (Eval Status):
                  No Eval  Has Eval
Procurement_Type                   
Commercial             27        23
Custom                 21        21

Evaluation Rates:
  Custom: 50.00% (21/42) evaluated
  Commercial: 46.00% (23/50) evaluated

Chi-Square Statistic: 0.0300
P-Value: 8.6260e-01

Result: NOT SIGNIFICANT


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot (or Bar Chart).
*   **Purpose:** The plot compares categorical data (types of software or systems: Commercial vs. Custom) against a numerical variable (Proportion with Independent Evaluation). It is designed to visualize the difference in evaluation rates between the two distinct groups.

### 2. Axes
*   **X-Axis:**
    *   **Label/Categories:** Represents the two groups being compared: **"Commercial"** (left) and **"Custom"** (right).
*   **Y-Axis:**
    *   **Title:** "Proportion with Independent Evaluation".
    *   **Range:** The axis spans from **0.0 to 1.0**.
    *   **Units:** The values represent proportions (decimals), which are equivalent to percentages (0.0 = 0%, 1.0 = 100%).

### 3. Data Trends
*   **Comparison:**
    *   **Commercial (Vendor):** This group is represented by the light blue bar. It has a slightly lower value, annotated at **46.0%**.
    *   **Custom (In-House):** This group is represented by the orange bar. It is the taller of the two bars, annotated at **50.0%**.
*   **Pattern:** The values are relatively close to each other, with both groups hovering near the 50% mark. The "Custom" group has a slightly higher rate of independent evaluation compared to the "Commercial" group (a difference of 4 percentage points).

### 4. Annotations and Legends
*   **Main Title:** "Independent Evaluation Rates: Commercial (Vendor) vs Custom (In-House)". This clearly defines the subject of the comparison.
*   **Statistical Annotation (in Title):** **"(p=8.6260e-01)"**. This indicates the p-value resulting from a statistical test (likely a Chi-square test or T-test) comparing the two proportions.
*   **Bar Labels:**
    *   Above the blue bar: **"46.0%"**
    *   Above the orange bar: **"50.0%"**
*   **Colors:** Light blue is used for Commercial, and Orange is used for Custom.

### 5. Statistical Insights
*   **No Statistically Significant Difference:** The most critical insight comes from the p-value in the title ($p = 8.6260 \times 10^{-1}$ or **0.8626**).
    *   In scientific research, a p-value less than 0.05 is typically required to claim statistical significance.
    *   Since 0.8626 is much greater than 0.05, the observed difference between 46.0% and 50.0% is **not statistically significant**.
*   **Conclusion:** While the "Custom" bar is visually taller, statistically speaking, there is no evidence to suggest that Custom (In-House) solutions are independently evaluated at a different rate than Commercial (Vendor) solutions. The observed difference is likely due to random sampling variability.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
