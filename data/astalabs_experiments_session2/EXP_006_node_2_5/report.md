# Experiment 6: node_2_5

| Property | Value |
|---|---|
| **Experiment ID** | `node_2_5` |
| **ID in Run** | 6 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T01:17:16.936678+00:00 |
| **Runtime** | 193.6s |
| **Parent** | `node_1_0` |
| **Children** | `node_3_3`, `node_3_11`, `node_3_16` |
| **Creation Index** | 7 |

---

## Hypothesis

> Sector-Specific Technical Failures: The 'Financial' sector is disproportionately
plagued by 'Fairness/Bias' technical failures, whereas the 'Transportation'
sector is dominated by 'Robustness' and 'Safety' failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.6648 (Maybe True) |
| **Surprise** | -0.3828 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

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
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 60.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Map technical failure types to deployment sectors to identify sector-specific risks.

### Steps
- 1. Filter for `aiid_incidents`.
- 2. Select top sectors (e.g., Finance, Transportation, Healthcare, Government) based on count.
- 3. Create a contingency table of `Sector of Deployment` vs `Known AI Technical Failure` (grouping failures into broad categories if sparse, e.g., Bias vs Robustness vs Privacy).
- 4. Use Cramer's V to measure the strength of association.
- 5. Identify the failure type with the highest standardized residual for each sector.

### Deliverables
- Heatmap of Sector vs Failure Type; Cramer's V statistic; List of dominant failure modes per sector.

---

## Analysis

The code successfully loaded the dataset and performed the contingency analysis
using the `aiid_incidents` subset. However, the analysis reveals significant
data sparsity issues. After filtering for the top 8 sectors and failure types
and removing missing values, only 28 data points remained from the original
1,366 incidents.

1. **Statistical Significance**: Due to the small sample size (N=28), the Chi-
square test yielded a p-value of 0.125, indicating the results are not
statistically significant at the 0.05 level, despite a relatively high Cramer's
V of 0.56.

2. **Hypothesis Evaluation**:
   - **Transportation**: The results offer partial support. The `transportation
and storage` sector was dominated by `Latency Issues` (Residual: 2.27), which is
a subset of Robustness/Safety failures.
   - **Financial**: The Financial sector did not appear in the top 8 sectors
analyzed, preventing evaluation of the 'Fairness/Bias' hypothesis for this
domain.

3. **Other Findings**: The analysis identified `Hardware Failure` as a dominant
risk for `human health and social work activities` (Residual: 3.47) and
`Distributional Bias` for `information and communication` (Residual: 1.87).
Future iterations should check if the sparse columns are due to mapping issues
or inherent missing data.

---

## Review

The experiment was implemented faithfully, but the results are limited by severe
data sparsity. The analysis successfully generated the contingency table and
heatmaps, but the aggressive filtering (Top 8 sectors x Top 8 failures) combined
with `dropna()` reduced the dataset from 1,366 incidents to only 28 data points.
Consequently, the 'Financial' sector—a key part of the hypothesis—was excluded
from the top 8 sectors, preventing its evaluation. The 'Transportation'
hypothesis was tested and directionally supported (Latency Issues are a
Robustness failure), but the Chi-square test (p=0.125) was not statistically
significant. Future iterations should explicitly map/group specific failure
labels into broader categories (e.g., 'Bias', 'Robustness', 'Safety') before
filtering to retain more data.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
# Note: Dataset files are one level above the current working directory
file_path = '../astalabs_discovery_all_data.csv'

try:
    # Using low_memory=False to avoid DtypeWarning mixing types
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if running in a different environment structure
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for aiid_incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# 3. Select relevant columns and clean data
sector_col = 'Sector of Deployment'
failure_col = 'Known AI Technical Failure'

# Check if columns exist (using exact names from metadata or previous output)
# Metadata listed: 'Sector of Deployment', 'Known AI Technical Failure'
if sector_col not in aiid_df.columns or failure_col not in aiid_df.columns:
    print(f"Columns '{sector_col}' or '{failure_col}' not found. Available columns:")
    print(aiid_df.columns.tolist())
    exit(1)

# Drop NaN values in relevant columns for this analysis
subset = aiid_df[[sector_col, failure_col]].dropna()

# Clean strings
subset[sector_col] = subset[sector_col].astype(str).str.strip()
subset[failure_col] = subset[failure_col].astype(str).str.strip()

# 4. Filter for Top Sectors and Failures to ensure statistical relevance
# Select top N sectors by count
top_sectors_count = 8
top_sectors = subset[sector_col].value_counts().nlargest(top_sectors_count).index.tolist()

# Select top M failure types by count
# (AIID failure types can be sparse, so we focus on the most common ones)
top_failures_count = 8
top_failures = subset[failure_col].value_counts().nlargest(top_failures_count).index.tolist()

# Filter the dataframe
filtered_df = subset[
    (subset[sector_col].isin(top_sectors)) & 
    (subset[failure_col].isin(top_failures))
]

print(f"Analyzing top {top_sectors_count} sectors and top {top_failures_count} failure types.")
print(f"Data points after filtering: {len(filtered_df)}")

# 5. Create Contingency Table
contingency_table = pd.crosstab(filtered_df[sector_col], filtered_df[failure_col])

# 6. Chi-Square Test and Cramer's V
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# Cramer's V calculation
n = contingency_table.sum().sum()
min_dim = min(contingency_table.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0

print(f"\nCramer's V: {cramers_v:.4f}")
print(f"P-value: {p:.4e}")

# 7. Calculate Standardized Residuals
# Residual = (Observed - Expected) / sqrt(Expected)
# Adjusted Residual (Standardized) considers row/col totals, but Pearson residual is often sufficient for heatmaps.
# Here we calculate Pearson residuals for the heatmap visualization
pearson_residuals = (contingency_table - expected) / np.sqrt(expected)

# Identify dominant failure mode per sector (max residual)
dominant_failures = {}
for sector in contingency_table.index:
    sector_residuals = pearson_residuals.loc[sector]
    max_res_failure = sector_residuals.idxmax()
    max_res_val = sector_residuals.max()
    dominant_failures[sector] = (max_res_failure, max_res_val)

print("\nDominant Failure Mode per Sector (highest positive residual):")
for sector, (fail_type, res_val) in dominant_failures.items():
    print(f"  - {sector}: {fail_type} (Residual: {res_val:.2f})")

# 8. Visualization
plt.figure(figsize=(12, 8))

# Plotting Counts
plt.subplot(2, 1, 1)
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
plt.title(f'Contingency Table: Sector vs Technical Failure (Top {top_sectors_count}x{top_failures_count})')
plt.ylabel('Sector')
plt.xlabel('Technical Failure Type')

# Plotting Residuals (Associations)
plt.subplot(2, 1, 2)
# Use a diverging colormap to show positive (over-represented) and negative (under-represented) associations
sns.heatmap(pearson_residuals, annot=True, fmt='.2f', cmap='vlag', center=0)
plt.title('Pearson Residuals (Association Strength)')
plt.ylabel('Sector')
plt.xlabel('Technical Failure Type')

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Analyzing top 8 sectors and top 8 failure types.
Data points after filtering: 28

Cramer's V: 0.5599
P-value: 1.2529e-01

Dominant Failure Mode per Sector (highest positive residual):
  - Arts, entertainment and recreation, information and communication: Generalization Failure, Context Misidentification (Residual: 1.34)
  - human health and social work activities: Hardware Failure (Residual: 3.47)
  - information and communication: Distributional Bias (Residual: 1.87)
  - information and communication, Arts, entertainment and recreation: Misinformation Generation Hazard, Unsafe Exposure or Access (Residual: 2.27)
  - law enforcement: Harmful Application (Residual: 1.70)
  - transportation and storage: Latency Issues (Residual: 2.27)
  - wholesale and retail trade: Harmful Application (Residual: 1.70)

STDERR:
<ipython-input-1-39009af4353a>:111: UserWarning: Tight layout not applied. tight_layout cannot make Axes height small enough to accommodate all Axes decorations.
  plt.tight_layout()


=== Plot Analysis (figure 1) ===
Based on the provided image, here is a detailed analysis of the plots:

### 1. Plot Type
*   **Composite Visualization:** The image contains two vertically stacked **Heatmaps**.
*   **Top Plot:** A **Contingency Table Heatmap** displaying raw counts (frequencies) of occurrences. Its purpose is to show the absolute volume of intersections between two categorical variables.
*   **Bottom Plot:** A **Pearson Residuals Heatmap**. Its purpose is to visualize the strength and direction of the association between the variables, highlighting which combinations occur more or less frequently than would be expected by chance.

### 2. Axes
*   **Y-Axis (Both Plots):**
    *   **Label:** "Sector"
    *   **Categories:** The axis lists various industry sectors, including "law enforcement," "transportation and storage," "wholesale and retail trade," "human health and social work activities," and distinct categories involving "information and communication" (sometimes combined with "Arts, entertainment...").
*   **X-Axis (Both Plots):**
    *   **Label:** "Technical Failure Type"
    *   **Categories:** The axis lists specific failure types, such as "Context Misidentification," "Distributional Bias," "Generalization Failure," "Hardware Failure," "Harmful Application," and "Latency Issues."
    *   *Note:* The labels on the top plot are partially cut off at the bottom but correspond directly to the legible labels in the bottom plot.
*   **Value Ranges:**
    *   **Top Plot (Color Bar):** Ranges from **0 to 8** (Counts).
    *   **Bottom Plot (Color Bar):** Ranges approximately from **-1 to +3.5** (Pearson Residual scores).

### 3. Data Trends
**Top Plot (Contingency Table):**
*   **Sparse Matrix:** The majority of the cells contain values of **0** or **1** (light yellow), indicating that most specific sector/failure combinations are rare or unobserved in this dataset.
*   **High Frequency Cluster:** There is a notable cluster in the **"information and communication"** sector row.
    *   **Highest Value:** The intersection of "information and communication" and **"Distributional Bias"** has the highest count of **8** (dark blue).
    *   **Secondary High Value:** The intersection with **"Generalization Failure"** has a count of **3** (teal).

**Bottom Plot (Pearson Residuals):**
*   **Strong Positive Associations (Red):** These cells indicate relationships that are stronger than random chance.
    *   **Highest Association:** "human health and social work activities" vs. "Hardware Failure" has the highest residual of **3.47** (dark red).
    *   **Other Strong Associations:**
        *   "transportation and storage" vs. "Latency Issues" (**2.27**)
        *   "information and communication, Arts..." vs. "Misinformation Generation..." (**2.27**)
        *   "information and communication" vs. "Distributional Bias" (**1.87**)
        *   "law enforcement" vs. "Harmful Application" (**1.70**)
*   **Negative Associations (Blue/Grey):** These indicate combinations that happen less often than expected.
    *   "information and communication" vs. "Harmful Application" (**-1.18**)
    *   "transportation and storage" vs. "Distributional Bias" (**-1.13**)

### 4. Annotations and Legends
*   **Titles:**
    *   Top: "Contingency Table: Sector vs Technical Failure (Top 8x8)" indicating this is a subset of the most relevant categories.
    *   Bottom: "Pearson Residuals (Association Strength)".
*   **Color Legends:**
    *   **Top:** A sequential color map (Yellow $\to$ Blue) representing magnitude of counts.
    *   **Bottom:** A diverging color map (Blue $\to$ White $\to$ Red). Blue indicates negative residuals (under-represented), White is neutral, and Red indicates positive residuals (over-represented).
*   **Cell Annotations:** Every cell is annotated with its exact value (integer for the top plot, float with two decimal places for the bottom).

### 5. Statistical Insights
*   **Volume vs. Significance:** While the "information and communication" sector has the highest raw volume of failures (specifically "Distributional Bias" with 8 counts), the **Pearson Residuals** plot reveals that other sectors have statistically stronger associations with specific failures despite lower counts.
*   **Specific Risk Profiles:**
    *   **Healthcare:** Even though failures may be rare in raw numbers, the sector has a very specific vulnerability to **Hardware Failures** (Residual 3.47).
    *   **Transportation:** This sector is disproportionately associated with **Latency Issues** (Residual 2.27) and **Generalization Failures** (Count 2, Residual 0.80).
    *   **Retail & Law Enforcement:** Both show a higher-than-expected association with **Harmful Applications**.
*   **Bias Concentration:** "Distributional Bias" is heavily concentrated in the "information and communication" sector, both in raw numbers and statistical strength, while being under-represented in "transportation and storage."
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
