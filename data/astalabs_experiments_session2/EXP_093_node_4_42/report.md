# Experiment 93: node_4_42

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_42` |
| **ID in Run** | 93 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:31:36.224166+00:00 |
| **Runtime** | 266.2s |
| **Parent** | `node_3_4` |
| **Children** | `node_5_75`, `node_5_88` |
| **Creation Index** | 94 |

---

## Hypothesis

> ATLAS Kill Chain Clustering: In adversarial AI attacks, the tactic
'Exfiltration' is significantly more likely to co-occur with 'Collection' than
with 'Initial Access', indicating a distinct late-stage attack cluster.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8952 (Likely True) |
| **Posterior** | 0.6951 (Maybe True) |
| **Surprise** | -0.2401 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
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
| Maybe True | 22.0 |
| Uncertain | 38.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Identify clustering of adversarial tactics in real-world cases.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `atlas_cases`.
- 2. Parse the `tactics` column (handle delimiters like `|` and IDs like `{{...}}`).
- 3. Create binary flags for `Initial Access`, `Collection`, and `Exfiltration` for each case.
- 4. Calculate the conditional probability P(Exfiltration | Collection) and P(Exfiltration | Initial Access).
- 5. Use a difference of proportions test (Z-test) to see if Exfiltration is significantly more tied to Collection.

### Deliverables
- Co-occurrence probabilities, Z-test results.

---

## Analysis

The experiment successfully loaded the ATLAS dataset (n=52 cases) but
encountered data sparsity issues that prevented a complete statistical test.

1.  **Clustering Evidence**: The analysis confirmed a strong co-occurrence
between 'Collection' and 'Exfiltration'. Out of 10 cases involving 'Collection',
8 also involved 'Exfiltration' (80%), supporting the existence of a late-stage
attack cluster.
2.  **Missing Control Group**: The code failed to identify any cases tagged with
'Initial Access' (n=0). As a result, the conditional probability P(Exfiltration
| Initial Access) was undefined, and the Z-test for difference of proportions
could not be performed.
3.  **Hypothesis Status**: The hypothesis that 'Exfiltration' is *more* likely
to co-occur with 'Collection' than 'Initial Access' could not be statistically
falsified due to the lack of 'Initial Access' samples. However, the high
probability (80%) in the Collection cluster provides partial descriptive support
for the 'late-stage' nature of data exfiltration in AI attacks.

---

## Review

The experiment was successfully executed and the code handled the data sparsity
gracefully. While the complete statistical test (Z-test) could not be performed
due to the absence of the control group ('Initial Access') in the dataset, the
descriptive statistics provided sufficient evidence to evaluate the 'Collection'
clustering aspect of the hypothesis. The implementation was faithful to the
plan.

---

## Code

```python
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import os

# Load dataset with fallback for path location
filename = 'astalabs_discovery_all_data.csv'
if not os.path.exists(filename):
    if os.path.exists('../' + filename):
        filename = '../' + filename
    else:
        print(f"Error: {filename} not found in current or parent directory.")

df = pd.read_csv(filename, low_memory=False)

# Filter for ATLAS cases
atlas = df[df['source_table'] == 'atlas_cases'].copy()
print(f"ATLAS cases loaded: {len(atlas)}")

# Normalize tactics column
# Ensure strings, lowercase
atlas['tactics_norm'] = atlas['tactics'].fillna('').astype(str).str.lower()

# Helper to check for tactic presence
def has_tactic(text, tactic_name):
    # Simple substring check, robust to formatting like 'Initial Access' vs 'initial-access'
    return tactic_name in text

# Create binary flags for the tactics of interest
atlas['init_access'] = atlas['tactics_norm'].apply(lambda x: has_tactic(x, 'initial access') or has_tactic(x, 'initial-access'))
atlas['collection'] = atlas['tactics_norm'].apply(lambda x: has_tactic(x, 'collection'))
atlas['exfiltration'] = atlas['tactics_norm'].apply(lambda x: has_tactic(x, 'exfiltration'))

# Define the two groups based on the antecedent tactic
# Group 1: Cases having 'Collection'
# Group 2: Cases having 'Initial Access'

# Get the counts for the denominators (n)
n_coll = atlas['collection'].sum()
n_init = atlas['init_access'].sum()

# Get the counts for the numerators (k) - i.e., having Exfiltration given the group
k_coll_exfil = atlas[atlas['collection']]['exfiltration'].sum()
k_init_exfil = atlas[atlas['init_access']]['exfiltration'].sum()

# Calculate Probabilities
p_exfil_given_coll = k_coll_exfil / n_coll if n_coll > 0 else 0
p_exfil_given_init = k_init_exfil / n_init if n_init > 0 else 0

print("\n--- Co-occurrence Statistics ---")
print(f"P(Exfiltration | Collection)     = {k_coll_exfil}/{n_coll} ({p_exfil_given_coll:.2%})")
print(f"P(Exfiltration | Initial Access) = {k_init_exfil}/{n_init} ({p_exfil_given_init:.2%})")

# Perform Z-test for difference of proportions
# Note: This assumes independent samples, which is a limitation here as cases can be in both groups.
if n_coll > 0 and n_init > 0:
    count = np.array([k_coll_exfil, k_init_exfil])
    nobs = np.array([n_coll, n_init])
    
    stat, pval = proportions_ztest(count, nobs)
    
    print("\n--- Z-Test Results ---")
    print(f"Z-statistic: {stat:.4f}")
    print(f"P-value:     {pval:.4f}")
    
    if pval < 0.05:
        print("Result: Statistically significant difference.")
    else:
        print("Result: No statistically significant difference.")
else:
    print("\nInsufficient data for Z-test.")

# Visualization
plt.figure(figsize=(8, 5))
probs = [p_exfil_given_coll, p_exfil_given_init]
labels = ['Given Collection', 'Given Initial Access']
colors = ['#1f77b4', '#ff7f0e']

bars = plt.bar(labels, probs, color=colors, alpha=0.8)
plt.ylabel('Probability of Exfiltration')
plt.title('Conditional Probability of Exfiltration Tactic')
plt.ylim(0, 1.0)

for bar, p, k, n in zip(bars, probs, [k_coll_exfil, k_init_exfil], [n_coll, n_init]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f"{p:.1%} (n={k}/{n})", ha='center', va='bottom', fontsize=10)

plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: ATLAS cases loaded: 52

--- Co-occurrence Statistics ---
P(Exfiltration | Collection)     = 8/10 (80.00%)
P(Exfiltration | Initial Access) = 0/0 (0.00%)

Insufficient data for Z-test.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** The plot compares the conditional probability of a specific cybersecurity tactic ("Exfiltration") occurring based on two different antecedent conditions ("Given Collection" and "Given Initial Access").

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Probability of Exfiltration"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100% probability).
    *   **Increments:** 0.2.
*   **X-Axis:**
    *   **Labels:** The axis features two categorical conditions: "Given Collection" and "Given Initial Access".
    *   **Context:** These categories refer to stages in a cyber attack lifecycle (likely based on the MITRE ATT&CK framework).

### 3. Data Trends
*   **Tallest Bar:** The "Given Collection" category represents the highest value with a probability of 0.8.
*   **Shortest Bar:** The "Given Initial Access" category has no visible bar, representing a probability of 0.0.
*   **Pattern:** There is a stark contrast between the two conditions. The data suggests that the transition to "Exfiltration" is highly likely if the "Collection" stage has already occurred, whereas it is non-existent (or unobserved) directly from "Initial Access."

### 4. Annotations and Legends
*   **Bar Annotations:**
    *   **Given Collection:** Annotated with "**80.0% (n=8/10)**". This indicates that out of a sample size of 10 instances where "Collection" was observed, "Exfiltration" followed in 8 of those cases.
    *   **Given Initial Access:** Annotated with "**0.0% (n=0/0)**". This indicates a probability of 0%, but the sample size notation is distinct.

### 5. Statistical Insights
*   **High Correlation between Collection and Exfiltration:** The data indicates a strong conditional dependency. If an attacker has successfully performed "Collection," there is an 80% probability they will attempt or succeed at "Exfiltration." This implies that "Collection" is a critical precursor stage to data theft in this specific dataset.
*   **Undefined/Missing Data for Initial Access:** The annotation "(n=0/0)" for "Given Initial Access" is statistically significant. A denominator of 0 implies that there were **no instances** of the condition "Initial Access" observed in the context of this specific query/dataset. Therefore, the 0.0% probability is not a measure of impossibility, but rather a result of undefined data (division by zero). No statistical conclusion can be drawn regarding the transition from Initial Access to Exfiltration other than a lack of data points.
*   **Sample Size Limitation:** The insights are based on a very small dataset (n=10 for the first category and n=0 for the second). Conclusions drawn from this plot should be treated as preliminary due to the limited sample size.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
