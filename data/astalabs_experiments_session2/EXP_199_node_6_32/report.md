# Experiment 199: node_6_32

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_32` |
| **ID in Run** | 199 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:23:59.403890+00:00 |
| **Runtime** | 292.6s |
| **Parent** | `node_5_69` |
| **Children** | None |
| **Creation Index** | 200 |

---

## Hypothesis

> The 'Transparency-Accountability' Decoupling: At the agency level, there is no
significant positive correlation between 'Transparency' (rate of AI Notice) and
'Accountability' (rate of Independent Evaluation).

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2527 (Likely False) |
| **Surprise** | -0.5870 |
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
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Determine if agencies that are transparent (notify public) are also accountable (evaluate systems), or if these practices are uncorrelated.

### Steps
- 1. Load `eo13960_scored`.
- 2. Group data by `3_agency`.
- 3. For each agency, calculate: Rate of `59_ai_notice` = Yes, and Rate of `55_independent_eval` = Yes.
- 4. Filter for agencies with >10 systems to ensure stability.
- 5. Calculate Pearson and Spearman correlation coefficients between the two rates.
- 6. Scatter plot the Agency Transparency Rate vs. Accountability Rate.

### Deliverables
- 1. Correlation coefficients.
- 2. Scatter plot with regression line.

---

## Analysis

The experiment successfully tested the 'Transparency-Accountability Decoupling'
hypothesis using the EO13960 dataset.

1. **Data Processing**: After applying corrected text-matching logic to handle
verbose field values, the code identified 71 transparent systems and 53
accountable systems across 1,757 records. It aggregated these metrics for 21
agencies that had at least 10 AI systems.

2. **Statistical Findings**:
   - **Pearson Correlation**: 0.6921 (p = 0.0005)
   - **Spearman Correlation**: 0.8465 (p < 0.0001)
   - **Result**: These results indicate a **strong, statistically significant
positive correlation** between an agency's rate of providing AI Notice
(Transparency) and its rate of conducting Independent Evaluations
(Accountability).

3. **Visual Analysis**: The scatter plot reveals a clear trend where agencies
with higher transparency rates also tend to have higher accountability rates,
although the slope (0.26) suggests that accountability mechanisms lag behind
transparency efforts in absolute adoption terms. A significant cluster of
agencies sits at (0,0), failing on both metrics.

4. **Conclusion**: The hypothesis that these two governance dimensions are
'decoupled' (uncorrelated) is **refuted**. Instead, the data suggests they are
**strongly coupled**: agencies that invest in public transparency are also the
ones investing in independent evaluation, while laggards tend to neglect both.

---

## Review

The experiment successfully tested the 'Transparency-Accountability Decoupling'
hypothesis using the EO13960 dataset.

1. **Data Processing**: After applying corrected text-matching logic to handle
verbose field values (e.g., mapping 'Online - in the terms...' to Transparent),
the code identified 71 transparent systems and 53 accountable systems across
1,757 records. It aggregated these metrics for 21 agencies that had at least 10
AI systems.

2. **Statistical Findings**:
   - **Pearson Correlation**: 0.6921 (p = 0.0005)
   - **Spearman Correlation**: 0.8465 (p < 0.0001)
   - **Result**: These results indicate a **strong, statistically significant
positive correlation** between an agency's rate of providing AI Notice
(Transparency) and its rate of conducting Independent Evaluations
(Accountability).

3. **Visual Analysis**: The scatter plot reveals a clear trend where agencies
with higher transparency rates also tend to have higher accountability rates,
although the slope (0.26) suggests that accountability mechanisms lag behind
transparency efforts in absolute adoption terms. A significant cluster of
agencies sits at (0,0), failing on both metrics.

4. **Conclusion**: The hypothesis that these two governance dimensions are
'decoupled' (uncorrelated) is **refuted**. Instead, the data suggests they are
**strongly coupled**: agencies that invest in public transparency are also the
ones investing in independent evaluation, while laggards tend to neglect both.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import sys

# Set the dataset path
dataset_path = "astalabs_discovery_all_data.csv"

# Load the dataset
print("Loading dataset...")
try:
    df = pd.read_csv(dataset_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: Dataset not found at {dataset_path}")
    sys.exit(1)

# Filter for EO13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO13960 Scored records: {len(eo_data)}")

# Define target columns
agency_col = '3_agency'
transparency_col = '59_ai_notice'
accountability_col = '55_independent_eval'

# Updated Logic based on Debug Output
def check_transparency(val):
    if pd.isna(val):
        return 0
    s = str(val).strip().lower()
    # Explicit negative cases
    if 'none of the above' in s: return 0
    if 'n/a' in s: return 0
    if 'waived' in s: return 0
    if 'not safety' in s: return 0
    
    # Positive indicators found in the dataset
    if 'online' in s: return 1
    if 'in-person' in s: return 1
    if 'in person' in s: return 1
    if 'email' in s: return 1
    if 'telephone' in s: return 1
    if 'other' in s: return 1
    
    return 0

def check_accountability(val):
    if pd.isna(val):
        return 0
    s = str(val).strip().lower()
    # Positive indicators
    if s.startswith('yes'): return 1
    if s == 'true': return 1
    return 0

# Apply mapping
eo_data['is_transparent'] = eo_data[transparency_col].apply(check_transparency)
eo_data['is_accountable'] = eo_data[accountability_col].apply(check_accountability)

print(f"Total Transparent Systems: {eo_data['is_transparent'].sum()}")
print(f"Total Accountable Systems: {eo_data['is_accountable'].sum()}")

# Group by Agency
agency_stats = eo_data.groupby(agency_col).agg(
    system_count=('source_row_num', 'count'),
    transparency_rate=('is_transparent', 'mean'),
    accountability_rate=('is_accountable', 'mean')
).reset_index()

# Filter for agencies with > 10 systems
min_systems = 10
filtered_agencies = agency_stats[agency_stats['system_count'] > min_systems].copy()

print(f"\nAgencies with > {min_systems} systems: {len(filtered_agencies)}")
print(filtered_agencies[[agency_col, 'system_count', 'transparency_rate', 'accountability_rate']])

# Check for variance
x = filtered_agencies['transparency_rate']
y = filtered_agencies['accountability_rate']

if len(filtered_agencies) < 2:
    print("\nNot enough agencies to calculate correlation.")
elif np.std(x) == 0 or np.std(y) == 0:
    print("\nVariance is zero for one of the variables. Cannot calculate correlation.")
    print(f"Std Dev Transparency: {np.std(x)}")
    print(f"Std Dev Accountability: {np.std(y)}")
else:
    # Calculate Correlations
    pearson_corr, pearson_p = stats.pearsonr(x, y)
    spearman_corr, spearman_p = stats.spearmanr(x, y)

    print("\n--- Correlation Results ---")
    print(f"Pearson Correlation:  {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    print(f"Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")

    if pearson_p < 0.05:
        print("Result: Statistically significant correlation found.")
    else:
        print("Result: No statistically significant correlation found.")

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7, edgecolors='b', s=filtered_agencies['system_count']*2)
    
    # Add regression line
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', linestyle='--', label=f'Fit: y={m:.2f}x + {b:.2f}')

    # Label points (top 5 by count to avoid clutter)
    top_agencies = filtered_agencies.nlargest(5, 'system_count')
    for i, row in top_agencies.iterrows():
        plt.text(row['transparency_rate'], row['accountability_rate'], row[agency_col][:15]+'...', fontsize=8, alpha=0.9)

    plt.title('Agency Transparency (AI Notice) vs. Accountability (Indep. Eval)')
    plt.xlabel('Transparency Rate (Proportion of Systems with AI Notice)')
    plt.ylabel('Accountability Rate (Proportion of Systems with Indep. Eval)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset...
EO13960 Scored records: 1757
Total Transparent Systems: 71
Total Accountable Systems: 53

Agencies with > 10 systems: 21
                                             3_agency  ...  accountability_rate
0    Board of Governors of the Federal Reserve System  ...             0.000000
3                           Department of Agriculture  ...             0.000000
4                              Department of Commerce  ...             0.122807
5                                Department of Energy  ...             0.101266
6             Department of Health and Human Services  ...             0.000000
7                     Department of Homeland Security  ...             0.098361
9                                 Department of Labor  ...             0.014286
10                                Department of State  ...             0.000000
11                     Department of Veterans Affairs  ...             0.044053
13                         Department of the Interior  ...             0.000000
14                         Department of the Treasury  ...             0.000000
16                    Environmental Protection Agency  ...             0.000000
18              Federal Deposit Insurance Corporation  ...             0.000000
20                     Federal Housing Finance Agency  ...             0.000000
22                    General Services Administration  ...             0.041667
23      National Aeronautics and Space Administration  ...             0.000000
25                        National Science Foundation  ...             0.000000
31                 Securities and Exchange Commission  ...             0.000000
32                     Social Security Administration  ...             0.000000
33                         Tennessee Valley Authority  ...             0.000000
35  United States Agency for International Develop...  ...             0.007299

[21 rows x 4 columns]

--- Correlation Results ---
Pearson Correlation:  0.6921 (p-value: 0.0005)
Spearman Correlation: 0.8465 (p-value: 0.0000)
Result: Statistically significant correlation found.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is a detailed analysis of the plot:

### 1. Plot Type
*   **Type:** This is a **bubble plot** (a variation of a scatter plot) overlaid with a linear regression trend line.
*   **Purpose:** The plot aims to visualize the correlation between two metrics of agency performance regarding AI systems: Transparency (providing notice) and Accountability (conducting independent evaluations). The varying size of the bubbles suggests a third dimension of data, likely representing the total number of AI systems or the size of the agency, although this specific metric is not explicitly labeled in the legend.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Transparency Rate (Proportion of Systems with AI Notice)"
    *   **Range:** 0.00 to 0.35.
    *   **Meaning:** Represents the percentage (expressed as a decimal) of an agency's systems for which they provide AI notices.
*   **Y-Axis:**
    *   **Title:** "Accountability Rate (Proportion of Systems with Indep. Eval)"
    *   **Range:** 0.00 to roughly 0.125.
    *   **Meaning:** Represents the percentage (expressed as a decimal) of an agency's systems that undergo independent evaluation.

### 3. Data Trends
*   **Clustering at Origin:** There is a significant cluster of data points (bubbles) at the bottom-left corner (0.00, 0.00). This indicates that a large number of agencies have both zero transparency and zero accountability rates regarding their AI systems.
*   **Positive Correlation:** There is a general upward trend, indicated by the red line. As the Transparency Rate increases, the Accountability Rate tends to increase as well.
*   **Outliers/High Performers:**
    *   There is one notable outlier with the highest **Accountability Rate** (approx. 0.12) and a moderate Transparency Rate (approx. 0.16).
    *   The point with the highest **Transparency Rate** (approx. 0.33) also has a high Accountability Rate (approx. 0.10).
    *   Several agencies (labeled "Department of H...", "Department of V...") sit in the middle ground, showing some progress in both metrics but remaining below 0.10 for accountability.

### 4. Annotations and Legends
*   **Trend Line Legend:** A legend in the top right corner identifies the red dashed line as the "Fit: y=0.26x + 0.01".
*   **Agency Labels:** Several bubbles have text labels, though some are truncated. Visible labels include:
    *   "Department of H..."
    *   "Department of V..."
    *   "United States A..."
    *   "Department of T..."
*   **Grid:** A light dotted grid is present to help align the data points with the axis values.

### 5. Statistical Insights
*   **Linear Relationship (The Equation):** The regression line equation $y = 0.26x + 0.01$ provides specific insights:
    *   **Slope (0.26):** This suggests that for every 1 unit increase (or 100% increase) in the Transparency Rate, the Accountability Rate generally increases by only 0.26 (or 26%). This implies that agencies are generally much faster to implement transparency measures (notices) than they are to implement accountability measures (independent evaluations).
    *   **Intercept (0.01):** The Y-intercept is very close to zero. This confirms that agencies with zero transparency typically have near-zero accountability.
*   **Disparity in Adoption:** The scales of the axes highlight a disparity. While transparency rates reach up to ~33%, accountability rates top out much lower at ~12%. This suggests that "Accountability" (independent evaluation) is a harder metric for these agencies to achieve than "Transparency" (AI notices).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
