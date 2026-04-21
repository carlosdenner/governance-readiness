# Experiment 37: node_3_16

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_16` |
| **ID in Run** | 37 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:41:27.663869+00:00 |
| **Runtime** | 267.2s |
| **Parent** | `node_2_7` |
| **Children** | `node_4_21`, `node_4_30` |
| **Creation Index** | 38 |

---

## Hypothesis

> Sub-competencies defined with greater semantic breadth (higher word count in
definition) have significantly higher empirical incident coverage, suggesting
that vague competencies artificially capture more failure cases.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.3760 (Maybe False) |
| **Surprise** | -0.4247 |
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
| Maybe False | 90.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Assess if definition granularity biases incident mapping coverage.

### Steps
- 1. Load 'step1_sub_competencies.csv' and 'step3_coverage_map.csv'.
- 2. Calculate the word count of the 'definition' field for each sub-competency in Step 1.
- 3. Merge with 'incident_count' from Step 3 using sub-competency IDs.
- 4. Perform a Pearson or Spearman correlation test between Definition Word Count and Incident Count.
- 5. Generate a Scatter Plot with regression line.

### Deliverables
- Correlation metrics and Scatter Plot.

---

## Analysis

The experiment successfully tested the hypothesis linking definition semantic
breadth (word count) to incident coverage. By merging
'step1_sub_competencies.csv' and 'step3_coverage_map.csv', 8 common sub-
competencies were identified for analysis. The statistical results yielded a
Pearson correlation of r=0.274 (p=0.512) and a Spearman correlation of r=0.189
(p=0.654). These metrics indicate a weak positive relationship that is not
statistically significant. The scatter plot reinforces this finding, displaying
high variance where competencies with similar definition lengths (e.g., TR-1 vs
TR-4) have vastly different incident counts (19 vs 2). Consequently, the
hypothesis that longer, potentially vaguer definitions artificially capture more
incident data is not supported. The limited sample size (n=8) resulting from the
merge implies that the Step 1 dataset contains fewer defined constructs than the
16 tracked in the Step 3 coverage map.

---

## Review

The experiment successfully tested the hypothesis linking definition semantic
breadth to incident coverage. By merging 'step1_sub_competencies.csv' and
'step3_coverage_map.csv', 8 common sub-competencies were identified for
analysis. The statistical results yielded a Pearson correlation of r=0.27
(p=0.51) and a Spearman correlation of r=0.19 (p=0.65). These metrics indicate a
weak positive relationship that is not statistically significant. The scatter
plot reinforces this finding, displaying high variance where competencies with
identical definition lengths (e.g., TR-1 vs TR-4, both ~21 words) have vastly
different incident counts (19 vs 2). Consequently, the hypothesis that longer,
potentially vaguer definitions artificially capture more incident data is not
supported. The analysis suggests that incident frequency is driven by the nature
of the competency itself (e.g., high-level policy failures are more common)
rather than the verbosity of its definition.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import numpy as np

# Define file names
file_step1 = 'step1_sub_competencies.csv'
file_step3 = 'step3_coverage_map.csv'

# Resolve paths (check current and parent directory)
def get_path(filename):
    if os.path.exists(filename):
        return filename
    elif os.path.exists(os.path.join('..', filename)):
        return os.path.join('..', filename)
    return filename

path_step1 = get_path(file_step1)
path_step3 = get_path(file_step3)

# Load datasets
try:
    df_definitions = pd.read_csv(path_step1)
    df_coverage = pd.read_csv(path_step3)
    print(f"Loaded {file_step1} with shape {df_definitions.shape}")
    print(f"Loaded {file_step3} with shape {df_coverage.shape}")
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

# Calculate semantic breadth (word count) for definitions
# Ensure definition column is string
df_definitions['definition'] = df_definitions['definition'].astype(str)
df_definitions['word_count'] = df_definitions['definition'].apply(lambda x: len(x.split()))

# Merge datasets on ID
# step1 uses 'id', step3 uses 'sub_competency_id'
merged_df = pd.merge(
    df_definitions[['id', 'name', 'word_count']],
    df_coverage[['sub_competency_id', 'incident_count', 'coverage_status']],
    left_on='id',
    right_on='sub_competency_id',
    how='inner'
)

print(f"\nMerged dataset shape: {merged_df.shape}")
if merged_df.empty:
    print("No overlapping IDs found between datasets.")
    exit(1)

print("\n--- Sample Data ---")
print(merged_df[['id', 'word_count', 'incident_count']].head())

# Statistical Correlation
x = merged_df['word_count']
y = merged_df['incident_count']

pearson_r, pearson_p = stats.pearsonr(x, y)
spearman_r, spearman_p = stats.spearmanr(x, y)

print("\n--- Statistical Results ---")
print(f"Pearson Correlation: r={pearson_r:.3f}, p={pearson_p:.3f}")
print(f"Spearman Correlation: r={spearman_r:.3f}, p={spearman_p:.3f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Sub-competencies')

# Annotate points
for i, txt in enumerate(merged_df['id']):
    plt.annotate(txt, (x.iloc[i], y.iloc[i]), xytext=(5, 5), textcoords='offset points')

# Regression line
if len(merged_df) > 1:
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red', linestyle='--', label=f'Fit: y={m:.2f}x + {b:.2f}')

plt.title('Definition Semantic Breadth vs. Incident Coverage')
plt.xlabel('Definition Word Count')
plt.ylabel('Incident Count')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded step1_sub_competencies.csv with shape (11, 10)
Loaded step3_coverage_map.csv with shape (16, 7)

Merged dataset shape: (8, 6)

--- Sample Data ---
     id  word_count  incident_count
0  TR-1          21              19
1  TR-2          23               1
2  TR-3          17               9
3  TR-4          21               2
4  TR-5          17               0

--- Statistical Results ---
Pearson Correlation: r=0.274, p=0.512
Spearman Correlation: r=0.189, p=0.654


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Scatter plot with a linear regression trend line.
*   **Purpose:** The plot aims to visualize the relationship between the length of a definition (semantic breadth) and the number of incidents associated with it (incident coverage). It assesses whether longer, potentially more specific definitions correlate with higher or lower incident counts.

### 2. Axes
*   **X-Axis:**
    *   **Title:** Definition Word Count
    *   **Range:** The axis displays values from roughly **16.5 to 25.5**, with major tick marks labeled from **17 to 25**.
    *   **Units:** Count (number of words).
*   **Y-Axis:**
    *   **Title:** Incident Count
    *   **Range:** The axis displays values from roughly **-1 to 19.5**, with major tick marks labeled from **0 to 17.5** in increments of 2.5.
    *   **Units:** Count (number of incidents).

### 3. Data Trends
*   **General Trend:** The red trend line suggests a positive correlation; as the definition word count increases, the incident count tends to increase slightly.
*   **Data Points & Scatter:** The data is highly scattered, indicating a high variance and likely a weak correlation.
    *   **Highest Values:** **TR-1** (approx. 21 words, ~19 incidents) and **IR-3** (approx. 24 words, ~18 incidents) represent the highest incident counts.
    *   **Lowest Values:** **TR-5** (17 words, 0 incidents) and **TR-2** (23 words, 1 incident) represent the lowest incident counts.
    *   **Clusters:** There is a grouping of points around the 24–25 word count mark (IR-1, IR-3, IR-2), though their incident counts vary significantly (ranging from roughly 8 to 18).
    *   **Outliers/Deviations:** **TR-1** is a significant positive outlier relative to the trend line, having a much higher incident count than predicted by its word count. Conversely, **TR-2** is well below the trend line.

### 4. Annotations and Legends
*   **Title:** "Definition Semantic Breadth vs. Incident Coverage"
*   **Legend:**
    *   **Blue Circle:** Represents "Sub-competencies" (the individual data points).
    *   **Red Dashed Line:** Represents the linear fit model.
*   **Data Labels:** Each data point is annotated with a specific identifier (e.g., TR-1, TR-2, TR-3, TR-4, TR-5, IR-1, IR-2, IR-3), likely referring to specific technical requirements or incident reports.
*   **Equation:** The legend includes the regression equation: **Fit: y=0.64x + -5.63**.

### 5. Statistical Insights
*   **Slope (0.64):** The positive slope indicates that, on average, for every additional word added to the definition, the incident count increases by approximately 0.64.
*   **Predictive Power:** Despite the fit line, the predictive power of "Definition Word Count" for "Incident Count" appears weak. The data points are not tightly clustered around the line. For example, data points at x=17 range from y=0 to y=9, and at x=24 range from y=8 to y=18.
*   **Interpretation:** The analysis implies that while there is a slight tendency for wordier definitions to be associated with more incidents, the length of the definition is not the sole or primary driver of incident count. Other factors (represented by the widely spread labels like TR vs. IR) likely play a significant role.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
