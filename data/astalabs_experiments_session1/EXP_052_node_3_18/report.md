# Experiment 52: node_3_18

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_18` |
| **ID in Run** | 52 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:10:38.996729+00:00 |
| **Runtime** | 169.8s |
| **Parent** | `node_2_0` |
| **Children** | `node_4_15`, `node_4_16` |
| **Creation Index** | 53 |

---

## Hypothesis

> The EU AI Act acts as a stronger driver for 'Trust Readiness' requirements,
whereas NIST frameworks (AI RMF, GenAI Profile) are biased towards generating
'Integration Readiness' requirements.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.1901 (Likely False) |
| **Surprise** | -0.6405 |
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
| Definitely False | 90.0 |

---

## Experiment Plan

**Objective:** Analyze the normative frameworks to see if they predispose organizations toward different competency bundles.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Create a 'Framework Family' column: categorize 'EU AI Act' as 'EU' and 'NIST AI RMF 1.0'/'NIST GenAI Profile' as 'NIST'. Ignore others if low count.
- 3. Create a contingency table of 'Framework Family' vs. 'bundle'.
- 4. Perform a Chi-Square Test of Independence or Fisher's Exact Test.
- 5. Calculate the odds ratio of a requirement being 'Trust Readiness' if it comes from the EU vs. NIST.

### Deliverables
- Bar chart of Bundle proportions by Framework Family and statistical test results.

---

## Analysis

The experiment tested the hypothesis that the EU AI Act prioritizes 'Trust
Readiness' while NIST frameworks favor 'Integration Readiness'. The results
strongly refute this hypothesis. The analysis of 32 mapped requirements (9 EU,
23 NIST) reveals a nearly identical distribution of competency bundles across
both framework families: EU requirements mapped 56% to Trust (5/9) and 44% to
Integration (4/9), while NIST requirements mapped 57% to Trust (13/23) and 43%
to Integration (10/23). Statistical testing yielded a Fisher's Exact p-value of
1.0000 and an odds ratio of 0.96, indicating zero statistically significant
difference in the orientation of these frameworks. Both families drive Trust and
Integration competencies in equal proportions, suggesting a high degree of
alignment in their underlying architectural demands despite their different
regulatory natures.

---

## Review

The experiment successfully tested the hypothesis regarding regulatory framework
bias. The analysis of 32 mapped requirements (9 from EU AI Act, 23 from NIST
frameworks) revealed a nearly identical distribution of competency bundles
across both families, refuting the hypothesis that the EU AI Act
disproportionately drives 'Trust Readiness' or that NIST drives 'Integration
Readiness'. Specifically, EU requirements mapped 56% to Trust (5/9) and 44% to
Integration (4/9), while NIST requirements mapped 57% to Trust (13/23) and 43%
to Integration (10/23). Statistical testing (Fisher's Exact Test p=1.00, Odds
Ratio=0.96) confirmed there is no significant association between the framework
source and the resulting competency bundle. The findings suggest that despite
differences in legal standing (regulation vs. standard), both frameworks imply a
highly similar balance of governance (Trust) and engineering (Integration)
controls.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import os

# Define file path handling
filename = 'step2_crosswalk_matrix.csv'
# Try checking parent directory first as per instructions
filepath = os.path.join('..', filename)
if not os.path.exists(filepath):
    filepath = filename  # Fallback to current dir

print(f"Loading dataset from: {filepath}")

try:
    df = pd.read_csv(filepath)
except FileNotFoundError:
    print(f"Error: File {filename} not found in current or parent directory.")
    sys.exit(1)

# --- Data Preprocessing ---

# Function to categorize frameworks
def categorize_framework(source_text):
    if pd.isna(source_text):
        return None
    source_upper = source_text.upper()
    if 'EU AI ACT' in source_upper:
        return 'EU'
    elif 'NIST' in source_upper:
        return 'NIST'
    else:
        return 'Other'

# Apply categorization
df['Framework_Family'] = df['source'].apply(categorize_framework)

# Filter for analysis (Exclude 'Other')
df_analysis = df[df['Framework_Family'] != 'Other'].copy()

print("\n=== Data Summary ===")
print(f"Total records loaded: {len(df)}")
print(f"Records in analysis (EU vs NIST): {len(df_analysis)}")
print(df_analysis['Framework_Family'].value_counts())

# --- Contingency Table ---
# We want to see: Framework vs Bundle
# Orient columns for Fisher's Exact Test logic: 
# Target comparison: Trust Readiness vs Integration Readiness
contingency = pd.crosstab(df_analysis['Framework_Family'], df_analysis['bundle'])

# Reorder specifically for Odds Ratio calculation:
# Rows: EU, NIST
# Cols: Trust Readiness, Integration Readiness
desired_index = ['EU', 'NIST']
desired_columns = ['Trust Readiness', 'Integration Readiness']

# Ensure all keys exist, fill with 0 if missing
contingency_ordered = contingency.reindex(index=desired_index, columns=desired_columns, fill_value=0)

print("\n=== Contingency Table (Observed Counts) ===")
print(contingency_ordered)

# --- Statistical Analysis ---

# 1. Chi-Square Test of Independence
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_ordered)

print("\n=== Statistical Test Results ===")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"Chi-Square P-value:   {p_chi2:.4f}")

# 2. Fisher's Exact Test (More appropriate for small sample sizes)
odds_ratio, p_fisher = stats.fisher_exact(contingency_ordered)

print(f"Fisher's Exact P-value: {p_fisher:.4f}")
print(f"Odds Ratio:             {odds_ratio:.4f}")

# Interpretation of Odds Ratio
# OR = (Odds of Trust given EU) / (Odds of Trust given NIST)
print("\n--- Interpretation ---")
if odds_ratio > 1:
    print(f"The odds of a requirement mapping to 'Trust Readiness' are {odds_ratio:.2f} times higher for EU AI Act than for NIST.")
elif odds_ratio < 1:
    print(f"The odds of a requirement mapping to 'Trust Readiness' are {1/odds_ratio:.2f} times higher for NIST than for EU AI Act.")
else:
    print("No difference in odds between frameworks.")

# --- Visualization ---

# Calculate proportions for plotting
props = contingency_ordered.div(contingency_ordered.sum(axis=1), axis=0)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Plot stacked bar chart
# Use specific colors for clarity: Trust (Greenish), Integration (Blueish)
props.plot(kind='bar', stacked=True, ax=ax, color=['#2ca02c', '#1f77b4'], alpha=0.8)

plt.title('Proportion of Competency Bundles by Framework Family')
plt.xlabel('Framework Family')
plt.ylabel('Proportion')
plt.legend(title='Competency Bundle', loc='upper right', bbox_to_anchor=(1.25, 1))
plt.xticks(rotation=0)

# Annotate bars with counts
for n, x in enumerate([*contingency_ordered.index.values]):
    for (cn, y) in enumerate(props.loc[x]):
        if y > 0:
            # Calculate cumulative height for position
            y_pos = props.loc[x].iloc[:cn].sum() + y/2
            # Get raw count
            raw_count = contingency_ordered.loc[x].iloc[cn]
            plt.text(n, y_pos, f"{raw_count}\n({y:.0%})", 
                     ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step2_crosswalk_matrix.csv

=== Data Summary ===
Total records loaded: 42
Records in analysis (EU vs NIST): 32
Framework_Family
NIST    23
EU       9
Name: count, dtype: int64

=== Contingency Table (Observed Counts) ===
bundle            Trust Readiness  Integration Readiness
Framework_Family                                        
EU                              5                      4
NIST                           13                     10

=== Statistical Test Results ===
Chi-Square Statistic: 0.0000
Chi-Square P-value:   1.0000
Fisher's Exact P-value: 1.0000
Odds Ratio:             0.9615

--- Interpretation ---
The odds of a requirement mapping to 'Trust Readiness' are 1.04 times higher for NIST than for EU AI Act.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** To visualize and compare the relative proportions of different subgroups ("Competency Bundles") within categorical groups ("Framework Families"). It allows for an easy comparison of the composition of each category rather than the total values.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Framework Family"
    *   **Labels:** Two categorical variables: "EU" and "NIST".
*   **Y-Axis:**
    *   **Title:** "Proportion"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100%).
    *   **Ticks:** Increments of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
*   **EU Framework:**
    *   **Trust Readiness (Green):** This is the larger segment, representing 5 items or **56%** of the total.
    *   **Integration Readiness (Blue):** This is the smaller segment, representing 4 items or **44%** of the total.
*   **NIST Framework:**
    *   **Trust Readiness (Green):** This is the larger segment, representing 13 items or **57%** of the total.
    *   **Integration Readiness (Blue):** This is the smaller segment, representing 10 items or **43%** of the total.
*   **Overall Pattern:** Both frameworks show a very similar distribution pattern where "Trust Readiness" constitutes the majority (over 50%) compared to "Integration Readiness."

### 4. Annotations and Legends
*   **Legend:** Located in the upper right corner, titled "Competency Bundle."
    *   **Green:** Represents "Trust Readiness."
    *   **Blue:** Represents "Integration Readiness."
*   **Bar Annotations:** Inside each bar segment, white text provides the specific data values:
    *   The top number indicates the raw **count** (frequency).
    *   The bottom number in parentheses indicates the **percentage** relative to that specific column.

### 5. Statistical Insights
*   **Consistency Across Frameworks:** The most striking insight is the near-identical distribution of competency bundles across both framework families. The difference in "Trust Readiness" is only 1 percentage point (56% for EU vs. 57% for NIST), suggesting that both frameworks prioritize or structure these competencies in a highly similar ratio.
*   **Sample Size Disparity:** While the proportions are similar, the absolute volume of data differs significantly. The NIST framework contains more than double the total number of competency bundles (Total $n=23$) compared to the EU framework (Total $n=9$).
*   **Dominance of Trust Readiness:** In both contexts, "Trust Readiness" is the dominant competency bundle, consistently outweighing "Integration Readiness."
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
