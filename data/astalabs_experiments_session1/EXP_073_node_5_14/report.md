# Experiment 73: node_5_14

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_14` |
| **ID in Run** | 73 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:54:24.710954+00:00 |
| **Runtime** | 200.4s |
| **Parent** | `node_4_22` |
| **Children** | None |
| **Creation Index** | 74 |

---

## Hypothesis

> Competencies supported by 'High' confidence literature evidence map to a
significantly higher number of architecture controls than those with 'Medium' or
'Low' confidence, implying that well-understood requirements are more
technically prescriptive.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6613 (Maybe True) |
| **Posterior** | 0.1694 (Likely False) |
| **Surprise** | -0.5709 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 22.0 |
| Uncertain | 6.0 |
| Maybe False | 2.0 |
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

**Objective:** Correlate the strength of literature evidence with the density of architectural mappings by comparing control counts between high and low/medium confidence groups.

### Steps
- 1. Load the dataset 'step2_competency_statements.csv'.
- 2. Create a new variable 'control_count' for each row by counting the number of semicolon-separated items in the 'applicable_controls' column.
- 3. Create two groups based on the 'confidence' column: Group A ('High') and Group B ('Medium' or 'Low').
- 4. Calculate the mean and standard deviation of 'control_count' for both groups.
- 5. Perform an independent samples t-test (Welch's t-test) to compare the means of Group A and Group B.
- 6. Print the group statistics, t-statistic, and p-value.

### Deliverables
- 1. Descriptive statistics (mean, std) for High vs. Medium/Low confidence groups.
- 2. T-test results (t-statistic, p-value).
- 3. Conclusion on statistical significance.

---

## Analysis

The experiment successfully loaded 'step2_competency_statements.csv' and tested
the hypothesis that evidence confidence correlates with architectural control
density.

**Key Findings:**
- **Descriptive Statistics:** The 'High' confidence group (n=36) had a mean
control count of 1.86 (SD=0.64), while the 'Medium/Low' confidence group (n=6)
had a nearly identical mean of 1.83 (SD=0.75). The dataset is heavily skewed
toward high-confidence evidence.
- **Hypothesis Test:** The Welch's t-test yielded a t-statistic of 0.0854 and a
p-value of 0.9346.
- **Conclusion:** With a p-value far exceeding 0.05, the hypothesis is
**rejected**. The data demonstrates that the density of technical controls
mapped to a requirement is independent of the strength of the supporting
literature evidence. Both well-supported and emerging requirements receive a
similar level of architectural prescription.

---

## Review

The experiment was faithfully implemented and the hypothesis was tested
sufficiently. The code successfully loaded 'step2_competency_statements.csv',
correctly quantified the architectural control density by parsing the
'applicable_controls' column, and performed a robust Welch's Independent Samples
T-test to account for the unequal sample sizes (High n=36 vs. Med/Low n=6).

**Hypothesis:** The hypothesis that 'High' confidence competencies map to a
significantly higher number of architecture controls was **rejected**.

**Findings:**
1. **Descriptive Statistics:** The mean control count for 'High' confidence
competencies (Mean = 1.86, SD = 0.64) was virtually identical to that of
'Medium/Low' confidence competencies (Mean = 1.83, SD = 0.75).
2. **Significance:** The t-test yielded a t-statistic of 0.0854 and a p-value of
0.9346. This p-value is far above the significance threshold of 0.05, indicating
no statistical difference.
3. **Implication:** The density of technical prescription (number of controls)
is independent of the strength of the supporting literature evidence. Emerging
or less-referenced requirements are mapped to architecture controls with the
same granularity as well-established requirements.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import numpy as np

# Attempt to load the dataset using the current directory
filename = 'step2_competency_statements.csv'
try:
    df = pd.read_csv(filename)
    print(f"Successfully loaded {filename}")
except FileNotFoundError:
    # Fallback to checking one level up if current dir fails, though previous error suggests current dir is correct context
    try:
        df = pd.read_csv('../' + filename)
        print(f"Successfully loaded ../{filename}")
    except FileNotFoundError:
        print(f"Error: Could not find {filename} in current or parent directory.")
        raise

# Helper function to count controls in the 'applicable_controls' column
# The column contains semicolon-separated values
def count_controls(val):
    if pd.isna(val) or str(val).strip() == '':
        return 0
    # Split by semicolon, strip whitespace, and filter out empty strings
    items = [x.strip() for x in str(val).split(';') if x.strip()]
    return len(items)

# Apply the counting function
df['control_count'] = df['applicable_controls'].apply(count_controls)

# Normalize the 'confidence' column to handle potential case inconsistencies
df['confidence_norm'] = df['confidence'].astype(str).str.lower().str.strip()

# Create the two groups: High vs Medium/Low
group_high = df[df['confidence_norm'] == 'high']['control_count']
group_others = df[df['confidence_norm'].isin(['medium', 'low'])]['control_count']

# Calculate Descriptive Statistics
mean_high = group_high.mean()
std_high = group_high.std()
n_high = len(group_high)

mean_others = group_others.mean()
std_others = group_others.std()
n_others = len(group_others)

print("\n--- Descriptive Statistics ---")
print(f"High Confidence (N={n_high}): Mean = {mean_high:.4f}, Std Dev = {std_high:.4f}")
print(f"Medium/Low Confidence (N={n_others}): Mean = {mean_others:.4f}, Std Dev = {std_others:.4f}")

# Perform Welch's Independent Samples T-test (does not assume equal variance)
t_stat, p_val = stats.ttest_ind(group_high, group_others, equal_var=False)

print("\n--- Hypothesis Test Results ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

# Interpretation
alpha = 0.05
if p_val < alpha:
    print("Conclusion: The difference is statistically significant (Reject H0).")
else:
    print("Conclusion: The difference is NOT statistically significant (Fail to reject H0).")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Successfully loaded step2_competency_statements.csv

--- Descriptive Statistics ---
High Confidence (N=36): Mean = 1.8611, Std Dev = 0.6393
Medium/Low Confidence (N=6): Mean = 1.8333, Std Dev = 0.7528

--- Hypothesis Test Results ---
T-statistic: 0.0854
P-value: 0.9346
Conclusion: The difference is NOT statistically significant (Fail to reject H0).

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
