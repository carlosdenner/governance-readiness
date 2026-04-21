# Experiment 160: node_6_15

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_15` |
| **ID in Run** | 160 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:35:13.465217+00:00 |
| **Runtime** | 223.0s |
| **Parent** | `node_5_2` |
| **Children** | None |
| **Creation Index** | 161 |

---

## Hypothesis

> Public Accountability Gap: Federal AI systems classified as 'Public-Facing' are
significantly more likely to provide an 'Appeal Process' than internal
administrative systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7097 (Likely True) |
| **Posterior** | 0.9011 (Definitely True) |
| **Surprise** | +0.2297 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 3.0 |
| Maybe True | 24.0 |
| Uncertain | 0.0 |
| Maybe False | 2.0 |
| Definitely False | 1.0 |

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

**Objective:** Assess whether public-facing systems include mechanism for recourse/redress.

### Steps
- 1. Load 'eo13960_scored'.
- 2. Create a 'Public-Facing' flag based on '26_public_service' (Yes vs No).
- 3. Map '65_appeal_process' to a binary variable (Yes vs No/NaN).
- 4. Compare the proportion of systems with appeal processes in Public vs Internal groups.
- 5. Perform a Fisher's Exact Test or Chi-Square test.
- 6. Plot the comparison.

### Deliverables
- Contingency table; Statistical test results; Comparison plot.

---

## Analysis

The experiment was successfully executed. After correcting the data mapping
logic to properly classify 'Public-Facing' systems based on descriptive text in
the '26_public_service' column (n=65) versus 'Internal/Admin' systems (n=1,692),
the analysis found strong statistical evidence supporting the hypothesis.
Public-facing AI systems are significantly more likely to provide an appeal
process (15.4%) compared to internal administrative systems (3.9%). The Chi-
Square test yielded a p-value of 3.24e-05 (p < 0.001) and an Odds Ratio of 4.48,
confirming that public deployment is a strong predictor for the presence of
recourse mechanisms. However, the absolute low rate of compliance (only ~15%
even in public systems) highlights a broader accountability gap across the
federal inventory.

---

## Review

The experiment was successfully executed and the code faithfully implemented the
analysis plan after correcting for data quality issues. The programmer correctly
identified that the '26_public_service' column contained descriptive text rather
than a simple binary, adjusting the mapping logic in the second attempt. The
statistical analysis (Chi-Square test) and visualization were appropriate for
the dataset. The results robustly support the hypothesis, showing a
statistically significant difference (p < 0.001) where public-facing systems are
over 4 times more likely (Odds Ratio ~4.48) to offer an appeal process compared
to internal systems, although the absolute prevalence remains low (15.4% vs
3.9%).

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

print("Starting Public Accountability Gap analysis (Attempt 2)...")

# 1. Load the dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for Federal AI Inventory data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded EO 13960 data: {len(eo_data)} records")

# 3. Preprocessing with Corrected Logic
col_public = '26_public_service'
col_appeal = '65_appeal_process'

# Define helper function for Public Service classification
def is_public_service(val):
    s = str(val).strip().lower()
    # If it is 'no', empty, or 'nan', it is NOT public facing
    if s in ['no', 'nan', '', 'null']:
        return False
    # Any other descriptive text implies it is a public use case
    return True

# Define helper function for Appeal Process classification
def has_appeal_process(val):
    s = str(val).strip().lower()
    # Only explicit 'yes' counts as having a process
    return s == 'yes'

# Apply mappings
eo_data['is_public'] = eo_data[col_public].apply(is_public_service)
eo_data['has_appeal'] = eo_data[col_appeal].apply(has_appeal_process)

# Print check to ensure we have data in both groups
print("\nDistribution of 'is_public':")
print(eo_data['is_public'].value_counts())
print("\nDistribution of 'has_appeal':")
print(eo_data['has_appeal'].value_counts())

# 4. Analysis
# Contingency Table
contingency = pd.crosstab(eo_data['is_public'], eo_data['has_appeal'])

# Check shape before assigning index/columns to avoid errors if a category is missing
if contingency.shape != (2, 2):
    print("\nWarning: Contingency table is not 2x2. One category may be missing.")
    print(contingency)
else:
    contingency.index = ['Internal/Admin', 'Public-Facing']
    contingency.columns = ['No Appeal Process', 'Has Appeal Process']
    print("\n--- Contingency Table ---")
    print(contingency)

# Calculate Proportions & Stats
results = eo_data.groupby('is_public')['has_appeal'].agg(['count', 'sum', 'mean'])
results.index = ['Internal/Admin', 'Public-Facing'] if len(results) == 2 else results.index
results.columns = ['Total Systems', 'With Appeal', 'Proportion']
print("\n--- Compliance Rates ---")
print(results)

# Statistical Test (Fisher's Exact if sample small, Chi-Square otherwise)
# Given N=1757, Chi-Square is appropriate
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Calculate Odds Ratio
# (Public_Has / Public_Not) / (Internal_Has / Internal_Not)
if contingency.shape == (2, 2):
    a = contingency.iloc[1, 1] # Public, Has Appeal
    b = contingency.iloc[1, 0] # Public, No Appeal
    c = contingency.iloc[0, 1] # Internal, Has Appeal
    d = contingency.iloc[0, 0] # Internal, No Appeal
    
    odds_ratio = (a / b) / (c / d) if (b * c) > 0 else np.nan
    print(f"Odds Ratio: {odds_ratio:.4f}")

# 5. Visualization
plt.figure(figsize=(10, 6))

categories = results.index
proportions = results['Proportion'].values
counts = results['With Appeal'].values
totals = results['Total Systems'].values

colors = ['#6c757d', '#007bff'] if len(categories) == 2 else ['#6c757d']
bars = plt.bar(categories, proportions, color=colors, alpha=0.8)

# Add labels
for bar, count, total, prop in zip(bars, counts, totals, proportions):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, 
             f'{prop:.1%} (n={count}/{total})',
             ha='center', va='bottom')

plt.ylabel('Proportion with Appeal Process')
plt.title('Public Accountability Gap: Availability of Appeal Processes\n(Public-Facing vs. Internal Federal AI Systems)')
plt.ylim(0, max(proportions) * 1.3 if len(proportions) > 0 and max(proportions) > 0 else 0.1)

# Add stats annotation
significance = "Significant" if p < 0.05 else "Not Significant"
plt.annotate(f'p-value: {p:.4e}\n({significance})', 
             xy=(0.5, 0.85), xycoords='axes fraction', 
             ha='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting Public Accountability Gap analysis (Attempt 2)...
Loaded EO 13960 data: 1757 records

Distribution of 'is_public':
is_public
False    1692
True       65
Name: count, dtype: int64

Distribution of 'has_appeal':
has_appeal
False    1681
True       76
Name: count, dtype: int64

--- Contingency Table ---
                No Appeal Process  Has Appeal Process
Internal/Admin               1626                  66
Public-Facing                  55                  10

--- Compliance Rates ---
                Total Systems  With Appeal  Proportion
Internal/Admin           1692           66    0.039007
Public-Facing              65           10    0.153846

Chi-Square Statistic: 17.2688
P-value: 3.2446e-05
Odds Ratio: 4.4793


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot.
*   **Purpose:** The plot compares the proportion of two distinct categories of Federal AI systems ("Internal/Admin" versus "Public-Facing") that feature an availability of appeal processes. It aims to highlight the disparity, termed the "Public Accountability Gap," between these two system types.

### 2. Axes
*   **X-Axis:**
    *   **Labels:** Categorical variables representing the type of AI system: **"Internal/Admin"** and **"Public-Facing"**.
    *   **Range:** N/A (Categorical).
*   **Y-Axis:**
    *   **Label:** **"Proportion with Appeal Process"**.
    *   **Range:** The axis spans from **0.000 to 0.200** (representing 0% to 20%).

### 3. Data Trends
*   **Comparison of Heights:** There is a substantial difference in bar height. The "Public-Facing" bar is approximately four times taller than the "Internal/Admin" bar.
*   **Tallest Bar:** The **"Public-Facing"** category has the highest proportion of appeal processes at **15.4%**.
*   **Shortest Bar:** The **"Internal/Admin"** category has the lowest proportion at **3.9%**.
*   **Pattern:** The data indicates that AI systems designed for public interaction are significantly more likely to have an appeal process in place compared to those used for internal or administrative purposes.

### 4. Annotations and Legends
*   **Bar Annotations (Data Labels):**
    *   **Internal/Admin:** Labeled with "3.9% (n=66/1692)". This indicates that out of a sample size of 1,692 internal systems, only 66 had appeal processes.
    *   **Public-Facing:** Labeled with "15.4% (n=10/65)". This indicates that out of a much smaller sample size of 65 public-facing systems, 10 had appeal processes.
*   **Statistical Annotation:** A box centered at the top of the chart displays **"p-value: 3.2446e-05 (Significant)"**. This indicates the result of a hypothesis test comparing the two proportions.
*   **Title:** The title "Public Accountability Gap: Availability of Appeal Processes (Public-Facing vs. Internal Federal AI Systems)" sets the context for the analysis.

### 5. Statistical Insights
*   **Statistical Significance:** The reported p-value ($3.2446 \times 10^{-5}$) is extremely small (well below standard thresholds like 0.05 or 0.01). This confirms that the observed difference in appeal process availability between internal and public-facing systems is **statistically significant** and highly unlikely to be due to random chance.
*   **Sample Size Disparity:** There is a massive disparity in sample sizes ($n=1692$ vs. $n=65$). Despite the Public-Facing group being a much smaller cohort, the proportion difference was strong enough to yield high statistical significance.
*   **The "Gap":** The plot effectively visualizes the "Accountability Gap" mentioned in the title. While the vast majority of AI systems in this dataset are Internal/Admin (1692 total), they are severely lacking in appeal mechanisms (3.9%) compared to the minority of Public-Facing systems (15.4%). However, it is worth noting that even in the "better" category (Public-Facing), an 84.6% lack of appeal processes suggests low accountability across the board.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
