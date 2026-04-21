# Experiment 13: node_3_4

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_4` |
| **ID in Run** | 13 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T21:55:35.554554+00:00 |
| **Runtime** | 149.9s |
| **Parent** | `node_2_5` |
| **Children** | `node_4_3`, `node_4_6` |
| **Creation Index** | 14 |

---

## Hypothesis

> The 'Human-in-the-Loop Approval Gates' architecture control is significantly
more prevalent in 'Trust Readiness' governance requirements than in 'Integration
Readiness' requirements.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8226 (Likely True) |
| **Posterior** | 0.2107 (Likely False) |
| **Surprise** | -0.7101 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 10.0 |
| Maybe True | 20.0 |
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
| Definitely False | 90.0 |

---

## Experiment Plan

**Objective:** Verify if human oversight mechanisms are structurally bound to Trust competencies rather than Integration competencies.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Identify the column 'Human-in-the-Loop Approval Gates' and the 'bundle' column.
- 3. Create a contingency table counting the presence ('X') vs. absence of this control for each bundle ('Trust Readiness' vs. 'Integration Readiness').
- 4. Perform a Fisher's Exact Test or Chi-square test to evaluate the association.

### Deliverables
- Contingency table, statistical test results, and a bar chart showing the percentage of requirements in each bundle that map to Human-in-the-Loop controls.

---

## Analysis

The experiment successfully tested the hypothesis regarding the prevalence of
'Human-in-the-Loop Approval Gates'. The analysis revealed that this architecture
control is present in 13.0% (3/23) of 'Integration Readiness' requirements but
is completely absent (0/19) in 'Trust Readiness' requirements. This finding
directly contradicts the hypothesis, which predicted a higher prevalence in
Trust competencies. Furthermore, the Fisher's Exact Test yielded a p-value of
0.2387, indicating that the observed difference is not statistically
significant. Therefore, the hypothesis is rejected on both directional and
statistical grounds.

---

## Review

The experiment successfully tested the hypothesis regarding the prevalence of
'Human-in-the-Loop Approval Gates'. The analysis revealed that this architecture
control is present in 13.0% (3/23) of 'Integration Readiness' requirements but
is completely absent (0/19) in 'Trust Readiness' requirements. This finding
directly contradicts the hypothesis, which predicted a higher prevalence in
Trust competencies. Furthermore, the Fisher's Exact Test yielded a p-value of
0.2387, indicating that the observed difference is not statistically
significant. Therefore, the hypothesis is rejected on both directional and
statistical grounds.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

# Load dataset
try:
    df = pd.read_csv('../step2_crosswalk_matrix.csv')
except FileNotFoundError:
    try:
        df = pd.read_csv('step2_crosswalk_matrix.csv')
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# Define target column and bundle column
control_col = 'Human-in-the-Loop Approval Gates'
bundle_col = 'bundle'

# Preprocess: Convert 'X' to 1, others to 0
df['has_control'] = df[control_col].apply(lambda x: 1 if str(x).strip().upper() == 'X' else 0)

# Generate Contingency Table
# We ensure all possible values (0 and 1) are present for both bundles to ensure a 2x2 matrix
contingency = pd.crosstab(df[bundle_col], df['has_control'])

# Ensure columns 0 and 1 exist (in case one is missing entirely from the dataset)
if 0 not in contingency.columns:
    contingency[0] = 0
if 1 not in contingency.columns:
    contingency[1] = 0
    
# Sort columns to be [0, 1] for consistent interpretation: 0=Absent, 1=Present
contingency = contingency[[0, 1]]

print("=== Contingency Table (Rows=Bundle, Cols=Control Presence) ===")
print(contingency)
print("\n")

# Calculate percentages for reporting
bundle_stats = df.groupby(bundle_col)['has_control'].agg(['count', 'sum', 'mean'])
bundle_stats['percentage'] = bundle_stats['mean'] * 100
print("=== Descriptive Statistics ===")
print(bundle_stats)
print("\n")

# Perform Fisher's Exact Test
# Fisher's is chosen over Chi-square due to small sample size (N=42)
# The contingency table index usually sorts alphabetically: Integration, Trust.
# We need to pass the 2x2 matrix.
try:
    odds_ratio, p_value = stats.fisher_exact(contingency)
    print("=== Fisher's Exact Test Results ===")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: Statistically Significant (p < {alpha})")
    else:
        print(f"Result: Not Statistically Significant (p >= {alpha})")
except Exception as e:
    print(f"Error performing statistical test: {e}")

# Visualization
plt.figure(figsize=(8, 6))
colors = ['skyblue' if b == 'Integration Readiness' else 'salmon' for b in bundle_stats.index]
bars = plt.bar(bundle_stats.index, bundle_stats['percentage'], color=colors, edgecolor='black')

plt.title(f"Prevalence of '{control_col}' by Bundle")
plt.ylabel('Percentage of Requirements with Control (%)')
plt.xlabel('Competency Bundle')
plt.ylim(0, 100)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: === Contingency Table (Rows=Bundle, Cols=Control Presence) ===
has_control             0  1
bundle                      
Integration Readiness  20  3
Trust Readiness        19  0


=== Descriptive Statistics ===
                       count  sum      mean  percentage
bundle                                                 
Integration Readiness     23    3  0.130435   13.043478
Trust Readiness           19    0  0.000000    0.000000


=== Fisher's Exact Test Results ===
Odds Ratio: 0.0000
P-value: 0.2387
Result: Not Statistically Significant (p >= 0.05)


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Vertical Bar Plot.
*   **Purpose:** To compare the percentage prevalence of a specific control mechanism ("Human-in-the-Loop Approval Gates") across distinct categories ("Competency Bundles").

**2. Axes**
*   **X-axis:**
    *   **Title:** "Competency Bundle"
    *   **Categories:** Two distinct categories are displayed: "Integration Readiness" and "Trust Readiness."
*   **Y-axis:**
    *   **Title:** "Percentage of Requirements with Control (%)"
    *   **Range:** The scale ranges from 0 to 100, marked in increments of 20 (0, 20, 40, 60, 80, 100).

**3. Data Trends**
*   **Tallest Bar:** The "Integration Readiness" bundle represents the highest value.
*   **Shortest Bar:** The "Trust Readiness" bundle represents the lowest value.
*   **Pattern:** There is a significant disparity between the two categories. While one bundle shows a measurable presence of the control, the other shows a complete absence.

**4. Annotations and Legends**
*   **Chart Title:** "Prevalence of 'Human-in-the-Loop Approval Gates' by Bundle" appears at the top.
*   **Data Labels:** Exact percentage values are annotated directly above each bar to provide precise readings:
    *   Integration Readiness: **13.0%**
    *   Trust Readiness: **0.0%**
*   **Styling:** The bars are filled with a light blue color and outlined in black.

**5. Statistical Insights**
*   **Prevalence in Integration:** 13.0% of the requirements within the "Integration Readiness" bundle include 'Human-in-the-Loop Approval Gates.' This suggests that human oversight is a minority but notable component of integration procedures.
*   **Absence in Trust:** 0.0% of the requirements within the "Trust Readiness" bundle utilize these approval gates. This indicates that human manual approval is currently not a factor in the requirements defined for Trust Readiness in this dataset.
*   **Comparative Insight:** Use of human approval gates is exclusively associated with Integration Readiness in this comparison, implying distinct operational or compliance standards between the two competency bundles.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
