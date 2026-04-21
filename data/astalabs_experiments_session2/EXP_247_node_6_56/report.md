# Experiment 247: node_6_56

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_56` |
| **ID in Run** | 247 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T12:42:54.925764+00:00 |
| **Runtime** | 278.1s |
| **Parent** | `node_5_54` |
| **Children** | None |
| **Creation Index** | 248 |

---

## Hypothesis

> Vendor Opacity: AI systems acquired from commercial vendors are significantly
less likely to provide 'Code Access' compared to custom-developed systems,
creating a 'black box' governance risk.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.9945 (Definitely True) |
| **Surprise** | +0.0128 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 30.0 |
| Maybe True | 0.0 |
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

**Objective:** Compare code access availability between commercial and custom AI systems in the federal inventory.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` (EO13960 subset).
- 2. Create `source_type`: Map `10_commercial_ai`='Yes' to 'Commercial', and `37_custom_code`='Yes' (or `22_dev_method`='Agency') to 'Custom'. Ensure mutually exclusive groups if possible, or prioritize 'Commercial' if both are present.
- 3. Create binary `has_code_access` from `38_code_access` (Map 'Yes' to 1, others to 0).
- 4. Perform a Chi-Square test of independence.
- 5. Calculate percentage of code access for both groups.

### Deliverables
- Contingency table; Chi-Square statistics; Bar chart of Code Access rates by Source Type.

---

## Analysis

The experiment successfully tested the 'Vendor Opacity' hypothesis after
correcting the data filtering logic from the previous attempt. By mapping
'37_custom_code' ('Yes' as Custom, 'No' as Commercial) and parsing
'38_code_access', the analysis successfully categorized 950 systems (713 Custom,
237 Commercial).

**Findings:**
1.  **Strong Support for Hypothesis:** The results strongly support the
hypothesis that commercial AI systems are less likely to provide code access.
Only **11.0%** of Commercial systems provided code access, compared to **80.6%**
of Custom-developed systems.
2.  **Statistical Significance:** The Chi-Square test yielded a statistic of
368.55 and a p-value of **3.87e-82**, indicating the difference is highly
statistically significant.
3.  **Governance Implication:** The data confirms a 'black box' risk for
commercial AI in the federal inventory, where nearly 90% of vendor-acquired
systems lack the transparency (code access) available in internally developed
solutions.

---

## Review

The experiment successfully tested the 'Vendor Opacity' hypothesis. The
programmer correctly adapted the data processing strategy after discovering in
the debug phase that the '10_commercial_ai' column contained descriptive text
rather than binary flags. By utilizing '37_custom_code' as the primary
discriminator ('Yes'=Custom, 'No'=Commercial), the analysis isolated 237
commercial systems and 713 custom systems.

**Findings:**
1.  **Hypothesis Supported:** The results strongly support the hypothesis that
commercial AI systems are significantly less likely to provide code access.
Commercial systems showed an access rate of only **11.0%**, compared to
**80.6%** for custom-developed systems.
2.  **Statistical Significance:** The difference is highly statistically
significant (Chi-Square p < 1e-81), confirming that the disparity is not due to
chance.
3.  **Implications:** This validates the concern that federal agencies face a
'black box' governance challenge with vendor-acquired AI, as they lack the
transparency (code access) typically available in internal developments.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# 1. Define Source Type using '37_custom_code'
# 'Yes' -> Custom, 'No' -> Commercial
def classify_source(val):
    s = str(val).strip().lower()
    if s == 'yes':
        return 'Custom'
    elif s == 'no':
        return 'Commercial'
    return None

eo_data['source_type'] = eo_data['37_custom_code'].apply(classify_source)

# 2. Define Code Access using '38_code_access'
# Map values starting with 'Yes' to 1, 'No' to 0
def classify_access(val):
    s = str(val).strip().lower()
    if s.startswith('yes'):
        return 1
    elif s.startswith('no'):
        return 0
    return None

eo_data['has_code_access'] = eo_data['38_code_access'].apply(classify_access)

# Drop rows with missing values in the relevant columns
analysis_df = eo_data.dropna(subset=['source_type', 'has_code_access']).copy()

print(f"Rows for analysis: {len(analysis_df)}")
print("Source distribution:\n", analysis_df['source_type'].value_counts())
print("Code Access distribution:\n", analysis_df['has_code_access'].value_counts())

# 3. Contingency Table
contingency_table = pd.crosstab(analysis_df['source_type'], analysis_df['has_code_access'])
contingency_table.columns = ['No Access', 'Has Access']
print("\nContingency Table (Source Type x Code Access):\n", contingency_table)

# 4. Statistical Test
# Using Chi-Square Test of Independence
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p_value:.5e}")

# 5. Calculate Percentages for Plotting
summary = analysis_df.groupby('source_type')['has_code_access'].agg(['count', 'mean'])
summary['percent_access'] = summary['mean'] * 100
print("\nAccess Rates:\n", summary[['count', 'percent_access']])

# 6. Visualization
plt.figure(figsize=(8, 6))
colors = ['#d62728', '#1f77b4'] # Red for Commercial, Blue for Custom
bars = plt.bar(summary.index, summary['percent_access'], color=colors)
plt.title(f'Code Access Availability: Commercial vs. Custom AI\n(p={p_value:.2e})')
plt.xlabel('Source Type')
plt.ylabel('Percentage with Code Access (%)')
plt.ylim(0, 100)

# Add labels
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
Code output: Rows for analysis: 950
Source distribution:
 source_type
Custom        713
Commercial    237
Name: count, dtype: int64
Code Access distribution:
 has_code_access
1.0    601
0.0    349
Name: count, dtype: int64

Contingency Table (Source Type x Code Access):
              No Access  Has Access
source_type                       
Commercial         211          26
Custom             138         575

Chi-Square Statistic: 368.5534
P-value: 3.86514e-82

Access Rates:
              count  percent_access
source_type                       
Commercial     237       10.970464
Custom         713       80.645161


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** The plot compares categorical data ("Source Type") against a quantitative variable ("Percentage with Code Access"). It is designed to visualize the disparity in code availability between two specific types of Artificial Intelligence sources.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Source Type"
    *   **Labels:** The axis represents two distinct categories: "Commercial" and "Custom".
*   **Y-Axis:**
    *   **Title:** "Percentage with Code Access (%)"
    *   **Range:** The axis spans from 0 to 100, representing percentage points.
    *   **Intervals:** The scale is marked in increments of 20 (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Tallest Bar:** The "Custom" category (blue bar) is significantly taller, reaching a value of **80.6%**.
*   **Shortest Bar:** The "Commercial" category (red bar) is much shorter, reaching a value of **11.0%**.
*   **Pattern:** There is a stark contrast between the two categories. Custom AI sources are overwhelmingly more likely to provide code access compared to Commercial AI sources. The "Custom" category is nearly 7.3 times higher than the "Commercial" category.

### 4. Annotations and Legends
*   **Bar Labels:** Exact percentage values are annotated directly above each bar ("11.0%" for Commercial and "80.6%" for Custom) to provide precise data points.
*   **Title Annotation (P-value):** The plot title includes a statistical significance notation: **"(p=3.87e-82)"**.
*   **Color Coding:** The bars are distinct in color (Red for Commercial, Blue for Custom) to visually differentiate the categories, though a separate legend is not needed as the X-axis labels explain the grouping.

### 5. Statistical Insights
*   **Statistical Significance:** The p-value included in the title ($3.87 \times 10^{-82}$) is infinitesimally small (far below standard thresholds like 0.05 or 0.01). This indicates that the difference in code availability between Commercial and Custom AI is **highly statistically significant** and is not due to random chance.
*   **Availability Gap:** The data suggests a fundamental difference in transparency or distribution models. "Custom" AI models appear to be largely open-source or accessible (over 4/5ths availability), whereas "Commercial" AI models act largely as "black boxes" or proprietary software, with only roughly 1 in 10 offering code access.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
