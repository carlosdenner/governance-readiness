# Experiment 15: node_3_6

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_6` |
| **ID in Run** | 15 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T01:43:30.473677+00:00 |
| **Runtime** | 268.6s |
| **Parent** | `node_2_6` |
| **Children** | `node_4_5`, `node_4_32` |
| **Creation Index** | 16 |

---

## Hypothesis

> The 'Legacy Governance' Gap: Federal AI systems initiated before 2020 are
significantly less likely to have a recorded 'Impact Assessment' than systems
initiated post-2020, indicating a lack of retroactive governance application.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8387 (Likely True) |
| **Posterior** | 0.2857 (Likely False) |
| **Surprise** | -0.6636 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 12.0 |
| Maybe True | 18.0 |
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

**Objective:** Quantify the governance gap between legacy and modern AI systems.

### Steps
- 1. Filter 'eo13960_scored'.
- 2. Parse '18_date_initiated' to extract the year. Handle missing or malformed dates.
- 3. Create a binary variable 'Is_Legacy' (Year < 2020).
- 4. Compare the proportion of 'Yes' in '52_impact_assessment' between Legacy and Non-Legacy systems using a Chi-square test.

### Deliverables
- Impact Assessment rates for Pre-2020 vs Post-2020 systems; Statistical significance of the gap.

---

## Analysis

The experiment successfully tested the 'Legacy Governance' Gap hypothesis using
the `eo13960_scored` dataset. Out of 1,757 records, 1,191 (67.8%) had parseable
initiation dates. The analysis compared 'Legacy' systems (initiated < 2020,
n=228) against 'Modern' systems (initiated 2020+, n=963).

Contrary to the hypothesis that legacy systems would lack retroactive
governance, the results show that **Legacy systems are significantly more
likely** to have a recorded Impact Assessment (10.53%) compared to Modern
systems (3.74%). The Chi-square test yielded a p-value of 5.22e-05, confirming
the difference is statistically significant. This finding refutes the initial
hypothesis and suggests that while overall compliance is low (93-96% missing
assessments), newer systems are entering the inventory with even lower
documented governance readiness than the older, established systems.

---

## Review

The experiment successfully tested the 'Legacy Governance' Gap hypothesis. The
implementation correctly parsed initiation dates for 1,191 systems from the EO
13960 dataset and compared Impact Assessment rates between Legacy (<2020) and
Modern (2020+) cohorts.

**Hypothesis Status:** Refuted (Significant Reverse Trend).

**Findings:** Contrary to the hypothesis that legacy systems would lack
retroactive governance compared to newer systems, the data reveals that Legacy
systems are significantly *more* likely to have a recorded Impact Assessment
(10.53%) compared to Modern systems (3.74%). The Chi-square test (p < 0.001)
confirms this difference is statistically significant.

**Interpretation:** This unexpected finding suggests that 'modern' AI systems in
the federal inventory—potentially comprising rapid pilots or experimental
tools—are entering with lower documented governance rigor than older,
established systems. However, the absolute rates indicate that the vast majority
of systems in both cohorts (>89%) lack impact assessments.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'astalabs_discovery_all_data.csv'
# Use low_memory=False to avoid DtypeWarning, or specify dtype if known. 
# Given the sparse nature, low_memory=False is safer for a quick script.
df = pd.read_csv(file_path, low_memory=False)

# Filter for EO 13960 scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Total EO 13960 records: {len(eo_df)}")

# --- Step 1: Parse Dates ---
# Column: '18_date_initiated'
# Attempt to convert to datetime. This handles various formats.
eo_df['date_parsed'] = pd.to_datetime(eo_df['18_date_initiated'], errors='coerce')

# Extract year
eo_df['initiation_year'] = eo_df['date_parsed'].dt.year

# Filter out rows where date could not be parsed
valid_date_df = eo_df.dropna(subset=['initiation_year']).copy()
print(f"Records with valid dates: {len(valid_date_df)} ({(len(valid_date_df)/len(eo_df))*100:.1f}%)")

# Define Legacy vs Modern
# Legacy: < 2020
# Modern: >= 2020
valid_date_df['is_legacy'] = valid_date_df['initiation_year'] < 2020
valid_date_df['cohort'] = valid_date_df['is_legacy'].map({True: 'Legacy (<2020)', False: 'Modern (2020+)'})

# --- Step 2: Analyze Impact Assessment ---
# Column: '52_impact_assessment'
# Check unique values to determine binary mapping
print("\nUnique values in '52_impact_assessment':")
print(valid_date_df['52_impact_assessment'].value_counts(dropna=False))

# Binarize: 'Yes' vs Others
# We define 'Has Assessment' as explicitly 'Yes'. 
# 'No', 'N/A', and specific reasons for No are treated as 'No Assessment'.
valid_date_df['has_assessment'] = valid_date_df['52_impact_assessment'].astype(str).str.strip().str.lower() == 'yes'

# --- Step 3: Statistical Analysis ---

# Group by Cohort
summary = valid_date_df.groupby('cohort')['has_assessment'].agg(['count', 'sum', 'mean'])
summary.columns = ['Total Systems', 'With Assessment', 'Assessment Rate']
summary['Assessment Rate %'] = (summary['Assessment Rate'] * 100).round(2)

print("\n--- Governance Gap Analysis: Impact Assessments ---")
print(summary[['Total Systems', 'With Assessment', 'Assessment Rate %']])

# Contingency Table for Chi-Square
contingency_table = pd.crosstab(valid_date_df['cohort'], valid_date_df['has_assessment'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

alpha = 0.05
if p < alpha:
    print("Result: Statistically Significant difference in governance rates.")
else:
    print("Result: No statistically significant difference found.")

# --- Step 4: Visualization ---
plt.figure(figsize=(8, 6))
colors = ['#ff9999', '#66b3ff']
ax = summary['Assessment Rate %'].plot(kind='bar', color=colors, edgecolor='black')
plt.title('Impact Assessment Rate: Legacy vs. Modern AI Systems')
plt.ylabel('Percentage with Impact Assessment (%)')
plt.xlabel('Initiation Cohort')
plt.ylim(0, 100)

# Add value labels
for i, v in enumerate(summary['Assessment Rate %']):
    ax.text(i, v + 2, f"{v}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total EO 13960 records: 1757
Records with valid dates: 1191 (67.8%)

Unique values in '52_impact_assessment':
52_impact_assessment
NaN                        1053
Yes                          59
No                           49
Planned or in-progress.      29
YES                           1
Name: count, dtype: int64

--- Governance Gap Analysis: Impact Assessments ---
                Total Systems  With Assessment  Assessment Rate %
cohort                                                           
Legacy (<2020)            228               24              10.53
Modern (2020+)            963               36               3.74

Chi-Square Statistic: 16.3653
P-value: 5.2232e-05
Result: Statistically Significant difference in governance rates.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot compares the frequency (percentage) of impact assessments between two distinct categories of AI systems based on their initiation time period ("Legacy" vs. "Modern").

### 2. Axes
*   **X-axis (Horizontal):**
    *   **Title:** "Initiation Cohort"
    *   **Labels:** The axis features two categorical labels rotated 90 degrees for readability: "Legacy (<2020)" and "Modern (2020+)".
*   **Y-axis (Vertical):**
    *   **Title:** "Percentage with Impact Assessment (%)"
    *   **Range:** The scale runs from 0 to 100.
    *   **Increments:** Major tick marks are placed every 20 units (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Tallest Bar:** The "Legacy (<2020)" cohort represents the highest value.
*   **Shortest Bar:** The "Modern (2020+)" cohort represents the lowest value.
*   **Pattern:** There is a notable downward trend between the two groups. The rate of impact assessments has decreased significantly in the modern cohort compared to the legacy cohort. Both bars are relatively small compared to the full y-axis scale, indicating generally low rates across the board.

### 4. Annotations and Legends
*   **Data Labels:** Specific percentage values are annotated directly above each bar in bold text for precision:
    *   Legacy (<2020): **10.53%**
    *   Modern (2020+): **3.74%**
*   **Color Coding:** The bars are distinct in color (Light Red/Pink for Legacy and Light Blue for Modern) to visually separate the two cohorts, though the x-axis labels provide the primary identification.

### 5. Statistical Insights
*   **Significant Decline:** Systems initiated after 2020 ("Modern") are significantly less likely to have an impact assessment than those initiated before 2020 ("Legacy"). The rate dropped by roughly 6.8 percentage points.
*   **Relative Drop:** The transition from 10.53% to 3.74% represents a relative decrease of approximately **64.5%**. This suggests that as AI systems have become "Modern," the practice of conducting impact assessments has become less than half as common as it was previously.
*   **Low Overall Adoption:** Despite the difference between the two groups, the absolute values for both are low. Even in the higher-performing "Legacy" group, roughly 90% of systems did not have an impact assessment. In the "Modern" group, over 96% lack one. The vast empty space in the plot (from 10% up to 100%) visually emphasizes the scarcity of these assessments in the dataset.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
