# Experiment 96: node_7_1

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_1` |
| **ID in Run** | 96 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:39:29.519410+00:00 |
| **Runtime** | 211.7s |
| **Parent** | `node_6_9` |
| **Children** | None |
| **Creation Index** | 97 |

---

## Hypothesis

> There is a significant positive correlation between the number of 'Trust'
deficiencies and 'Integration' deficiencies in single incidents, supporting the
'Joint Failure' theory.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.1921 (Likely False) |
| **Surprise** | -0.6474 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 2.0 |
| Maybe True | 27.0 |
| Uncertain | 1.0 |
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

**Objective:** Test if failures in Trust sub-competencies co-occur with failures in Integration sub-competencies.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Parse the 'llm_sub_competencies' column (list of IDs like TR-01, IR-05).
- 3. For each row, count the number of IDs starting with 'TR' (Trust Count) and 'IR' (Integration Count).
- 4. Perform a Pearson (or Spearman) correlation analysis between Trust Count and Integration Count.
- 5. Fit a simple linear regression to visualize the relationship.

### Deliverables
- Scatter Plot of Trust vs. Integration Deficiencies with Regression Line and Correlation Coefficient (r, p-value).

---

## Analysis

The experiment successfully loaded the 'step3_incident_coding.csv' dataset
(n=52) and analyzed the relationship between Trust Readiness (TR) and
Integration Readiness (IR) competency gaps.

**Data Processing:**
- The 'llm_sub_competencies' column was parsed to count distinct Trust (TR-xx)
and Integration (IR-xx) deficiencies for each incident.
- Descriptive statistics show an average of 0.96 Trust gaps and 1.00 Integration
gaps per incident.

**Statistical Results:**
- **Pearson Correlation:** r = -0.7573, p < 0.001
- **Spearman Correlation:** rho = -0.7552, p < 0.001

**Conclusion:**
The hypothesis that there is a significant *positive* correlation (supporting a
'Joint Failure' theory) is **rejected**. The analysis reveals a strong,
statistically significant **negative** correlation (r = -0.76). This indicates a
distinct trade-off in the dataset: incidents characterized by multiple Trust
failures tend to have fewer Integration failures, and vice versa. This suggests
that the coding framework or the nature of the incidents tends to isolate the
primary root cause into one domain (Trust vs. Integration) rather than
attributing failures to a collapse of both bundles simultaneously.

---

## Review

The experiment was successfully executed and the code faithfully implemented the
analysis plan. The dataset was correctly loaded, and the 'llm_sub_competencies'
field was parsed to quantify Trust (TR) and Integration (IR) gaps per incident.
The statistical analysis (Pearson and Spearman correlations) was appropriate for
the objective.

**Findings:**
1.  **Hypothesis Rejection:** The hypothesis predicting a significant positive
correlation (supporting 'Joint Failure' theory) was rejected. Instead, the
analysis found a strong, statistically significant **negative correlation**
(Pearson r = -0.7573, p < 0.0001).
2.  **Interpretation:** The negative correlation implies a trade-off or
separation of concerns in the current dataset: incidents are typically
characterized by either Trust deficiencies or Integration deficiencies, but
rarely a high volume of both simultaneously. This may reflect the distinct
nature of the failure modes or an artifact of the upstream coding process (e.g.,
if the LLM was constrained to select a small, fixed number of root causes,
picking a Trust cause would inherently reduce the slot available for an
Integration cause).
3.  **Visualization:** The scatter plot with jitter effectively visualized this
inverse relationship and the clustering of data points around low integer
values.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import re

# Define the file path based on the instruction that datasets are one level above
file_path = '../step3_incident_coding.csv'

# Fallback to current directory if the file is not found in the parent directory
if not os.path.exists(file_path):
    if os.path.exists('step3_incident_coding.csv'):
        file_path = 'step3_incident_coding.csv'
    else:
        print("Warning: step3_incident_coding.csv not found in ../ or ./")

# Load the dataset
print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# Function to extract and count unique Trust (TR) and Integration (IR) competency gaps
def count_competencies(val):
    if pd.isna(val):
        return 0, 0
    val_str = str(val)
    # Regex to find codes like TR-1, TR-01, IR-5, IR-08, etc.
    # Using set() to ensure we count unique competencies per incident
    tr_matches = set(re.findall(r'TR-\d+', val_str, re.IGNORECASE))
    ir_matches = set(re.findall(r'IR-\d+', val_str, re.IGNORECASE))
    return len(tr_matches), len(ir_matches)

# Apply the counting function
df[['Trust_Count', 'Integration_Count']] = df['llm_sub_competencies'].apply(
    lambda x: pd.Series(count_competencies(x))
)

# Perform Correlation Analysis
pearson_r, pearson_p = stats.pearsonr(df['Trust_Count'], df['Integration_Count'])
spearman_r, spearman_p = stats.spearmanr(df['Trust_Count'], df['Integration_Count'])

# Print Summary Statistics
print("=== Summary Statistics ===")
print(df[['Trust_Count', 'Integration_Count']].describe())
print("\n=== Correlation Analysis ===")
print(f"Pearson Correlation (r): {pearson_r:.4f} (p-value: {pearson_p:.4f})")
print(f"Spearman Correlation (rho): {spearman_r:.4f} (p-value: {spearman_p:.4f})")

# Visualization: Scatter Plot with Regression Line
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Use jitter to prevent overplotting of integer values
ax = sns.regplot(
    data=df,
    x='Trust_Count',
    y='Integration_Count',
    x_jitter=0.2,
    y_jitter=0.2,
    scatter_kws={'alpha': 0.6, 's': 60, 'edgecolor': 'white'},
    line_kws={'color': 'red', 'label': f'Linear Fit (r={pearson_r:.2f})'}
)

plt.title('Correlation: Trust vs. Integration Competency Deficiencies', fontsize=14)
plt.xlabel('Number of Trust Readiness Gaps (TR)', fontsize=12)
plt.ylabel('Number of Integration Readiness Gaps (IR)', fontsize=12)
plt.legend()

# Annotate with statistical results
stats_text = f'Pearson r = {pearson_r:.2f} (p={pearson_p:.3f})\nSpearman rho = {spearman_r:.2f} (p={spearman_p:.3f})'
plt.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_incident_coding.csv
=== Summary Statistics ===
       Trust_Count  Integration_Count
count    52.000000          52.000000
mean      0.961538           1.000000
std       0.277350           0.280056
min       0.000000           0.000000
25%       1.000000           1.000000
50%       1.000000           1.000000
75%       1.000000           1.000000
max       2.000000           2.000000

=== Correlation Analysis ===
Pearson Correlation (r): -0.7573 (p-value: 0.0000)
Spearman Correlation (rho): -0.7552 (p-value: 0.0000)


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Scatter plot with a linear regression overlay.
*   **Purpose:** The plot is designed to investigate the relationship and correlation between two variables: "Trust Readiness Gaps" and "Integration Readiness Gaps." The regression line helps visualize the general direction and strength of this relationship.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Number of Trust Readiness Gaps (TR)"
    *   **Range:** The axis ticks range from **0.0 to 2.0**, though the plot display area extends slightly from approximately -0.2 to 2.2.
*   **Y-Axis:**
    *   **Label:** "Number of Integration Readiness Gaps (IR)"
    *   **Range:** The axis ticks range from **0.0 to 2.0**, with the display area extending similarly from roughly -0.2 to 2.2.
*   **Units:** While explicit units (e.g., "count," "score") aren't listed, the decimal values (1.0, 1.5) suggest these represent normalized scores or averages rather than raw integer counts of gaps.

### 3. Data Trends
*   **Clustering:** The vast majority of the data points are tightly clustered in the center of the plot, specifically where both TR and IR values are between roughly **0.8 and 1.2**. This indicates that most subjects/samples have moderate deficiencies in both areas.
*   **Outliers/Leverage Points:** There are significant outliers that appear to drive the regression slope:
    *   **Top-Left:** A data point showing very low Trust Gaps (< 0) but very high Integration Gaps (~2.0).
    *   **Bottom-Right:** A data point showing very high Trust Gaps (> 2.0) but low Integration Gaps (~0.2).
    *   **Bottom-Center:** A singular point with moderate Trust Gaps (~1.1) but very low Integration Gaps (< 0.1).
*   **Directionality:** The red regression line slopes downward from left to right, indicating a **negative correlation**. As the number of Trust Readiness Gaps increases, the number of Integration Readiness Gaps tends to decrease.

### 4. Annotations and Legends
*   **Regression Line (Legend):** The legend at the top right indicates a "Linear Fit" with an r-value of **-0.76**. The red line represents the best-fit linear model.
*   **Confidence Interval:** A light red/pink shaded area surrounds the regression line. This represents the confidence interval (likely 95%). It is narrowest in the center where data is dense and fans out significantly at the ends where data is sparse, indicating higher uncertainty in the prediction at extreme values.
*   **Statistical Text (Top Left):**
    *   **Pearson r = -0.76 (p=0.000):** Indicates a strong negative linear correlation.
    *   **Spearman rho = -0.76 (p=0.000):** Indicates a strong negative monotonic relationship (rank-based correlation). The p-values of 0.000 indicate these results are statistically significant.

### 5. Statistical Insights
*   **Strong Negative Correlation:** Both Pearson and Spearman coefficients are **-0.76**, suggesting a strong inverse relationship between Trust and Integration deficiencies. Statistically, as one type of gap increases, the other decreases.
*   **Outlier Influence:** While the statistics suggest a strong correlation, visual inspection suggests this relationship is heavily influenced by the extreme outliers in the top-left and bottom-right corners. The central cluster (where most data resides) appears relatively amorphous with no obvious slope. If the outliers were removed, the correlation would likely be much weaker.
*   **Trade-off Interpretation:** The data suggests a potential trade-off or distinct categorization of problems. Subjects tend to either have high Trust issues *or* high Integration issues, but rarely extreme levels of both simultaneously (as the top-right quadrant is empty). However, the majority of subjects simply sit in the middle with average scores for both.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
