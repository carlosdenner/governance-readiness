# Experiment 51: node_4_18

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_18` |
| **ID in Run** | 51 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:27:14.398791+00:00 |
| **Runtime** | 191.5s |
| **Parent** | `node_3_19` |
| **Children** | `node_5_6`, `node_5_55` |
| **Creation Index** | 52 |

---

## Hypothesis

> Architectural Complexity of NIST Functions: Governance requirements mapped to
the NIST 'PROTECT' function require a significantly higher density of
architectural controls than those mapped to 'GOVERN' or 'MAP', reflecting a
'Technical Implementability Gap'.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7016 (Likely True) |
| **Posterior** | 0.2665 (Likely False) |
| **Surprise** | -0.5222 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 28.0 |
| Uncertain | 0.0 |
| Maybe False | 1.0 |
| Definitely False | 1.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 10.0 |
| Definitely False | 50.0 |

---

## Experiment Plan

**Objective:** Compare the number of architectural controls per NIST Function.

### Steps
- 1. Load `step2_crosswalk_matrix` subset from CSV (contains `function` and `req_id`) and `context_crosswalk_evidence.json` (contains `req_id` and list of `applicable_controls`).
- 2. Merge datasets on `req_id`.
- 3. Calculate the count of unique controls in `applicable_controls` for each requirement.
- 4. Group by `function` (GOVERN, MAP, MEASURE, MANAGE/PROTECT).
- 5. Perform Kruskal-Wallis test (non-parametric ANOVA) on control counts.

### Deliverables
- Boxplot of Control Counts by NIST Function; Kruskal-Wallis test results.

---

## Analysis

The experiment successfully loaded the dataset and merged the crosswalk matrix
with the JSON evidence to analyze architectural control density. The analysis
identified 9 functional categories, reflecting a mix of NIST AI RMF functions
(GOVERN, MAP, MEASURE, MANAGE) and other principles (e.g., SECURITY,
TRANSPARENCY).

The results **do not support** the hypothesis that 'PROTECT' (or 'SECURITY')
functions require a significantly higher density of architectural controls. The
Kruskal-Wallis test yielded a p-value of 0.4792, indicating no statistically
significant difference in control counts across functions.

Descriptive statistics revealed that the 'MAP' function actually had the highest
mean control count (2.33), while 'MANAGE' (1.80) and 'SECURITY' (1.90) were
comparable to 'GOVERN' (1.57) and 'MEASURE' (1.80). The boxplot visualization
confirms this uniformity, with most categories showing a median of 2 mapped
controls. This suggests that the 'Technical Implementability Gap' does not
manifest as a disparity in control density across these high-level functional
domains.

---

## Review

The experiment was executed successfully and faithfully followed the proposed
plan. The programmer effectively handled the file path issues encountered in the
first attempt and correctly merged the CSV and JSON datasets to link governance
requirements with architectural controls.

The analysis revealed a mix of functional categories beyond the standard NIST AI
RMF functions (e.g., 'SECURITY', 'TRANSPARENCY'), but the statistical approach
(Kruskal-Wallis test) remained valid for comparing these groups.

The results **reject** the hypothesis that 'PROTECT' (mapped here as 'SECURITY'
and 'MANAGE') functions require a significantly higher density of architectural
controls than 'GOVERN' or 'MAP'.

Key findings include:
1.  **No Statistical Significance**: The Kruskal-Wallis test resulted in a
p-value of 0.4792, indicating no significant difference in control density
across the functional domains.
2.  **Contradictory Trends**: Contrary to the hypothesis, the 'MAP' function
exhibited the highest mean control count (2.33), while 'SECURITY' (1.90) and
'MANAGE' (1.80) were statistically indistinguishable from 'GOVERN' (1.57).
3.  **Uniformity**: The boxplot visualization confirms that most requirements,
regardless of function, map to approximately 2 architectural controls,
suggesting a standardized level of implementation complexity across the
framework.

---

## Code

```python
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import os

# 1. Load Data
print("Loading datasets...")

# Try current directory first, then parent directory if not found
csv_filename = 'astalabs_discovery_all_data.csv'
json_filename = 'context_crosswalk_evidence.json'

if os.path.exists(csv_filename):
    csv_path = csv_filename
elif os.path.exists(f'../{csv_filename}'):
    csv_path = f'../{csv_filename}'
else:
    print(f"Error: {csv_filename} not found.")
    csv_path = None

if os.path.exists(json_filename):
    json_path = json_filename
elif os.path.exists(f'../{json_filename}'):
    json_path = f'../{json_filename}'
else:
    print(f"Error: {json_filename} not found.")
    json_path = None

if csv_path and json_path:
    # Load CSV and filter for step2_crosswalk_matrix
    try:
        df_csv = pd.read_csv(csv_path, low_memory=False)
        # Filter for the specific source table relevant to the hypothesis
        df_matrix = df_csv[df_csv['source_table'] == 'step2_crosswalk_matrix'].copy()
        
        # Select only relevant columns. 'function' might be in a column named 'function' or similar index
        # Based on previous exploration, 'function' is a column name.
        # 'req_id' is also a column.
        if 'function' in df_matrix.columns and 'req_id' in df_matrix.columns:
            df_matrix = df_matrix[['req_id', 'function']]
        else:
            print(f"Columns 'req_id' or 'function' missing in CSV. Available columns: {df_matrix.columns}")
            sys.exit(1)
            
        print(f"Loaded {len(df_matrix)} rows from CSV step2_crosswalk_matrix.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    # Load JSON
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        df_json = pd.DataFrame(json_data)
        print(f"Loaded {len(df_json)} rows from JSON context_crosswalk_evidence.")
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)

    # 2. Preprocessing & Merging
    # Normalize req_id
    df_matrix['req_id'] = df_matrix['req_id'].astype(str).str.strip()
    df_json['req_id'] = df_json['req_id'].astype(str).str.strip()

    # Merge
    merged_df = pd.merge(df_matrix, df_json, on='req_id', how='inner')
    print(f"Merged dataset shape: {merged_df.shape}")

    if merged_df.empty:
        print("Merged dataframe is empty. Check req_id matching.")
        sys.exit(0)

    # 3. Calculate Control Counts
    def count_controls(controls):
        if isinstance(controls, list):
            return len(controls)
        return 0

    merged_df['control_count'] = merged_df['applicable_controls'].apply(count_controls)

    # Clean Function column
    merged_df['function'] = merged_df['function'].astype(str).str.upper().str.strip()
    
    # Filter to NIST functions if possible, or just print unique values found
    print("Found functions:", merged_df['function'].unique())

    # 4. Statistical Test (Kruskal-Wallis)
    # We compare control counts across the different functions
    functions = merged_df['function'].unique()
    groups = [merged_df[merged_df['function'] == f]['control_count'] for f in functions]

    print("\nDescriptive Statistics:")
    print(merged_df.groupby('function')['control_count'].describe())

    if len(groups) > 1:
        stat, p_value = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis Test Results:")
        print(f"Statistic: {stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("Result: Significant difference found.")
        else:
            print("Result: No significant difference found.")
    else:
        print("Not enough groups for statistical testing.")

    # 5. Visualization
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='function', y='control_count', data=merged_df)
    plt.title('Architectural Control Density by NIST Function')
    plt.xlabel('NIST Function')
    plt.ylabel('Number of Mapped Controls')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("Could not verify file paths.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading datasets...
Loaded 42 rows from CSV step2_crosswalk_matrix.
Loaded 42 rows from JSON context_crosswalk_evidence.
Merged dataset shape: (42, 8)
Found functions: <StringArray>
[               'GOVERN',                   'MAP',               'MEASURE',
                'MANAGE',            'GOVERNANCE',          'TRANSPARENCY',
       'HUMAN OVERSIGHT', 'ACCURACY & ROBUSTNESS',              'SECURITY']
Length: 9, dtype: str

Descriptive Statistics:
                       count      mean       std  min   25%  50%   75%  max
function                                                                   
ACCURACY & ROBUSTNESS    1.0  2.000000       NaN  2.0  2.00  2.0  2.00  2.0
GOVERN                   7.0  1.571429  0.975900  1.0  1.00  1.0  2.00  3.0
GOVERNANCE               4.0  2.000000  0.816497  1.0  1.75  2.0  2.25  3.0
HUMAN OVERSIGHT          1.0  2.000000       NaN  2.0  2.00  2.0  2.00  2.0
MANAGE                   5.0  1.800000  0.447214  1.0  2.00  2.0  2.00  2.0
MAP                      6.0  2.333333  0.516398  2.0  2.00  2.0  2.75  3.0
MEASURE                  5.0  1.800000  0.447214  1.0  2.00  2.0  2.00  2.0
SECURITY                10.0  1.900000  0.567646  1.0  2.00  2.0  2.00  3.0
TRANSPARENCY             3.0  1.333333  0.577350  1.0  1.00  1.0  1.50  2.0

Kruskal-Wallis Test Results:
Statistic: 7.5440
P-value: 0.4792
Result: No significant difference found.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Box Plot (also known as a Box-and-Whisker Plot).
*   **Purpose:** This plot visualizes the distribution, central tendency, and variability of the "Number of Mapped Controls" across different "NIST Function" categories. It allows for a quick comparison of how control density varies between different functional areas of the framework.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "NIST Function"
    *   **Labels:** Categorical variables representing specific functions: GOVERN, MAP, MEASURE, MANAGE, GOVERNANCE, TRANSPARENCY, HUMAN OVERSIGHT, ACCURACY & ROBUSTNESS, SECURITY.
*   **Y-Axis:**
    *   **Title:** "Number of Mapped Controls"
    *   **Range:** The axis spans from 1.00 to 3.00, formatted with decimal places.
    *   **Units:** Count (integer values representing the number of controls).

### 3. Data Trends
*   **High Variability:**
    *   **GOVERNANCE:** Shows the largest interquartile range (the height of the blue box) and whiskers extending from 1.0 to 3.0, indicating the most variation in the number of controls mapped to this function.
    *   **GOVERN:** Also shows significant spread, covering the full range of the y-axis (1 to 3).
    *   **MAP:** The distribution is skewed higher compared to others, with the box extending towards the upper limit (approx. 2.75).
*   **Low Variability / Consistency:**
    *   **HUMAN OVERSIGHT** and **ACCURACY & ROBUSTNESS:** These categories appear as flat lines at the 2.0 mark. This indicates zero variance; every data point in these categories has exactly 2 mapped controls.
    *   **MEASURE** and **MANAGE:** These are highly concentrated at 2.0, appearing almost as flat lines, but with single outliers at 1.0.
*   **Lower Values:**
    *   **TRANSPARENCY:** The distribution is skewed lower, with the median or lower quartile pushing down towards 1.0 and 1.5.

### 4. Annotations and Legends
*   **Title:** "Architectural Control Density by NIST Function" is clearly displayed at the top.
*   **Outliers:** Small circles represent outliers (data points that fall significantly outside the typical range). These are visible for:
    *   **MEASURE:** Outlier at 1.0.
    *   **MANAGE:** Outlier at 1.0.
    *   **SECURITY:** Outliers at both 1.0 and 3.0.
*   **Grid Lines:** Horizontal dashed grid lines are included to assist in estimating the y-axis values for the boxes and whiskers.

### 5. Statistical Insights
*   **Central Tendency is ~2:** Across almost all categories, the median value (often represented by the line inside the box) hovers around 2 mapped controls. This suggests a standardized approach where most functional areas are typically addressed by two architectural controls.
*   **Governance Complexity:** The presence of both "GOVERN" and "GOVERNANCE" as categories is notable (potentially indicating data from slightly different frameworks or versions). Both show high variability compared to technical categories like "ACCURACY," suggesting that governance functions are applied less inconsistently or depend heavily on the specific context.
*   **Standardization in Technical Functions:** Technical functions like "HUMAN OVERSIGHT" and "ACCURACY & ROBUSTNESS" are extremely rigid (flat lines), suggesting a strict "one-size-fits-all" mapping (likely 2 controls) for these areas in the architecture being analyzed.
*   **Security Distribution:** While the "SECURITY" function is centered at 2, the presence of outliers at both the minimum (1) and maximum (3) suggests edge cases where security is either lighter or heavier than the standard configuration.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
