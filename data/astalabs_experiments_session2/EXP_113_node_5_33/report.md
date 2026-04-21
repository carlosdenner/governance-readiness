# Experiment 113: node_5_33

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_33` |
| **ID in Run** | 113 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:28:03.633974+00:00 |
| **Runtime** | 305.8s |
| **Parent** | `node_4_37` |
| **Children** | `node_6_9`, `node_6_74` |
| **Creation Index** | 114 |

---

## Hypothesis

> Commercial COTS AI systems in the federal inventory are significantly less
likely to have undergone Independent Evaluations compared to custom-developed
systems, indicating a reliance on vendor assurances over objective testing.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8952 (Likely True) |
| **Posterior** | 0.3599 (Maybe False) |
| **Surprise** | -0.6423 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 19.0 |
| Maybe True | 11.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 20.0 |
| Definitely False | 40.0 |

---

## Experiment Plan

**Objective:** Determine if reliance on Commercial/COTS AI correlates with a lack of Independent Evaluation controls.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `eo13960_scored`.
- 2. Clean the `10_commercial_ai` column to categorize systems into 'Commercial/COTS' vs 'Custom/GOTS' (Government Off-The-Shelf).
- 3. Clean the `55_independent_eval` column into a binary compliance flag (Yes vs No).
- 4. Create a contingency table of Commercial Status vs Independent Evaluation.
- 5. Perform a Chi-Square test of independence to assess the relationship.

### Deliverables
- Contingency table; Bar chart comparing evaluation rates by vendor type; Statistical test results.

---

## Analysis

The experiment successfully tested the hypothesis by adapting the methodology to
use '37_custom_code' as a reliable proxy for vendor type (No='Commercial/COTS',
Yes='Custom/GOTS'), given the data quality issues with '10_commercial_ai'.
Analyzing 967 valid records, the study found that Independent Evaluation rates
are universally low: only 3.67% for Commercial/COTS systems (9/245) and 5.96%
for Custom/GOTS systems (43/722). A Chi-Square test yielded a p-value of 0.2284,
indicating that the difference is not statistically significant. Consequently,
the hypothesis that commercial systems are *significantly* less likely to
undergo independent evaluation is not supported. Instead, the data reveals a
systemic governance gap where over 94% of federal AI systems—regardless of
whether they are commercial or custom-built—lack reported independent
evaluation.

---

## Review

The experiment successfully tested the hypothesis by adapting the methodology to
use '37_custom_code' as a reliable proxy for vendor type (No='Commercial/COTS',
Yes='Custom/GOTS'), given the data quality issues with '10_commercial_ai'.
Analyzing 967 valid records, the study found that Independent Evaluation rates
are universally low: only 3.67% for Commercial/COTS systems (9/245) and 5.96%
for Custom/GOTS systems (43/722). A Chi-Square test yielded a p-value of 0.2284,
indicating that the difference is not statistically significant. Consequently,
the hypothesis that commercial systems are *significantly* less likely to
undergo independent evaluation is not supported. Instead, the data reveals a
systemic governance gap where over 94% of federal AI systems—regardless of
whether they are commercial or custom-built—lack reported independent
evaluation.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Determine Commercial vs Custom using '37_custom_code'
# Rationale: '10_commercial_ai' contains use-case descriptions, not vendor status.
# '37_custom_code' (Yes/No) is a reliable proxy: 'Yes' = Custom/GOTS, 'No' = Commercial/COTS.
valid_code_status = eo_data.dropna(subset=['37_custom_code']).copy()

def map_vendor_type(val):
    val_str = str(val).strip().lower()
    if val_str == 'yes':
        return 'Custom/GOTS'
    elif val_str == 'no':
        return 'Commercial/COTS'
    return None

valid_code_status['Vendor_Type'] = valid_code_status['37_custom_code'].apply(map_vendor_type)
valid_code_status = valid_code_status.dropna(subset=['Vendor_Type'])

# Determine Independent Evaluation Status using '55_independent_eval'
def map_eval_status(val):
    if pd.isna(val):
        return 'No Evaluation'
    val_str = str(val).strip().lower()
    # Strict criteria: Must start with yes or be 'true'
    if val_str.startswith('yes') or val_str == 'true':
        return 'Independent Eval'
    # 'Planned', 'Not applicable', etc. count as No for "have undergone"
    return 'No Evaluation'

valid_code_status['Eval_Status'] = valid_code_status['55_independent_eval'].apply(map_eval_status)

# Generate Contingency Table
contingency = pd.crosstab(valid_code_status['Vendor_Type'], valid_code_status['Eval_Status'])
print("\n--- Contingency Table (Counts) ---")
print(contingency)

# Calculate Proportions
props = pd.crosstab(valid_code_status['Vendor_Type'], valid_code_status['Eval_Status'], normalize='index')
print("\n--- Contingency Table (Proportions) ---")
print(props)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Interpret results
alpha = 0.05
if p < alpha:
    print("\nResult: Statistically significant relationship found.")
    commercial_eval_rate = props.loc['Commercial/COTS', 'Independent Eval'] if 'Independent Eval' in props.columns else 0
    custom_eval_rate = props.loc['Custom/GOTS', 'Independent Eval'] if 'Independent Eval' in props.columns else 0
    print(f"Commercial Eval Rate: {commercial_eval_rate:.2%}")
    print(f"Custom Eval Rate: {custom_eval_rate:.2%}")
    if commercial_eval_rate < custom_eval_rate:
        print("Hypothesis Supported: Commercial systems are less likely to have independent evaluation.")
    else:
        print("Hypothesis Refuted: Commercial systems are NOT less likely to have independent evaluation.")
else:
    print("\nResult: No statistically significant difference found.")

# Visualization
plt.figure(figsize=(10, 6))
ax = props.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], rot=0)
plt.title('Independent Evaluation Rates: Commercial (No Custom Code) vs Custom (Custom Code)')
plt.ylabel('Proportion')
plt.xlabel('System Type')
plt.legend(title='Evaluation Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Add labels
for c in ax.containers:
    ax.bar_label(c, fmt='%.2f', label_type='center')

plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: 
--- Contingency Table (Counts) ---
Eval_Status      Independent Eval  No Evaluation
Vendor_Type                                     
Commercial/COTS                 9            236
Custom/GOTS                    43            679

--- Contingency Table (Proportions) ---
Eval_Status      Independent Eval  No Evaluation
Vendor_Type                                     
Commercial/COTS          0.036735       0.963265
Custom/GOTS              0.059557       0.940443

Chi-Square Statistic: 1.4508
P-value: 2.2840e-01

Result: No statistically significant difference found.


=== Plot Analysis (figure 2) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Chart.
*   **Purpose:** The plot is designed to compare the relative proportions of "Evaluation Status" (Independent Evaluation vs. No Evaluation) across two different "System Types" (Commercial/COTS vs. Custom/GOTS). The stacking allows the viewer to see how the total (1.0 or 100%) is divided between the two statuses for each category.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "System Type"
    *   **Labels:** "Commercial/COTS" (Commercial Off-The-Shelf) and "Custom/GOTS" (Custom/Government Off-The-Shelf). The main title clarifies these correspond to "No Custom Code" and "Custom Code" respectively.
*   **Y-Axis:**
    *   **Title:** "Proportion"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100%).
    *   **Scale:** Linear, marked at intervals of 0.2.

### 3. Data Trends
*   **Dominant Pattern:** In both categories, the blue section ("No Evaluation") makes up the vast majority of the bar, indicating that most systems do not undergo independent evaluation.
*   **Commercial/COTS:**
    *   **Shortest Segment:** Independent Eval (Pink) at a proportion of **0.04**.
    *   **Tallest Segment:** No Evaluation (Blue) at a proportion of **0.96**.
*   **Custom/GOTS:**
    *   **Shortest Segment:** Independent Eval (Pink) at a proportion of **0.06**.
    *   **Tallest Segment:** No Evaluation (Blue) at a proportion of **0.94**.

### 4. Annotations and Legends
*   **Legend:** Located on the right side, titled "Evaluation Status."
    *   **Red/Pink:** Represents "Independent Eval".
    *   **Blue:** Represents "No Evaluation".
*   **In-Graph Annotations:** The specific proportion values are written directly on the corresponding bar segments:
    *   For Commercial/COTS: **0.96** (blue area) and **0.04** (pink area).
    *   For Custom/GOTS: **0.94** (blue area) and **0.06** (pink area).

### 5. Statistical Insights
*   **Low Evaluation Rates:** Independent evaluation is extremely rare for both system types. Only 4% of Commercial systems and 6% of Custom systems receive independent evaluation.
*   **Marginal Difference:** While Custom/GOTS systems have a slightly higher rate of independent evaluation compared to Commercial/COTS systems (6% vs. 4%), the difference is minimal (2 percentage points).
*   **Conclusion:** The presence of custom code (in Custom/GOTS systems) does not significantly correlate with a higher likelihood of independent evaluation; the overwhelming trend for both groups is a lack of independent evaluation.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
