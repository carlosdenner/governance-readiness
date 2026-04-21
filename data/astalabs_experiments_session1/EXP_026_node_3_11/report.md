# Experiment 26: node_3_11

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_11` |
| **ID in Run** | 26 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:15:55.782118+00:00 |
| **Runtime** | 332.1s |
| **Parent** | `node_2_3` |
| **Children** | None |
| **Creation Index** | 27 |

---

## Hypothesis

> There is a positive correlation between the operational complexity of a sub-
competency (measured by the number of its defined observable practices) and its
frequency of implication in real-world AI incidents.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7097 (Likely True) |
| **Posterior** | 0.5739 (Maybe True) |
| **Surprise** | -0.1576 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 28.0 |
| Uncertain | 0.0 |
| Maybe False | 2.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 57.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** To determine if sub-competencies that are more operationally complex (having more defined practices) are more prone to failure or gaps in real-world scenarios.

### Steps
- 1. Load the datasets 'step1_sub_competencies.csv' and 'step3_coverage_map.csv'.
- 2. Perform an inner join between the two datasets, matching 'id' from Step 1 with 'sub_competency_id' from Step 3.
- 3. Create a new variable 'practice_count' by processing the 'observable_practices' column in the Step 1 dataset. Calculate this by counting the number of semicolon separators (';') in the string plus one (or using a similar delimiter-based counting method appropriate for the data format).
- 4. Extract the 'incident_count' variable from the Step 3 dataset.
- 5. Compute the Spearman Rank Correlation Coefficient (and p-value) between 'practice_count' and 'incident_count'.
- 6. Generate a scatter plot with 'practice_count' on the x-axis and 'incident_count' on the y-axis, including a regression line or trend indicator if possible.

### Deliverables
- 1. Merged dataframe validation (row count).
- 2. Spearman correlation coefficient and p-value.
- 3. Scatter plot visualizing Operational Complexity (Practice Count) vs. Incident Frequency.

---

## Analysis

The experiment was successfully executed, loading the datasets and merging them
to identify 8 common sub-competencies. However, the calculated independent
variable, 'Operational Complexity' (derived from the count of observable
practices), exhibited zero variance across the dataset—every single sub-
competency had exactly 3 defined practices. As a result, the Spearman Rank
Correlation Coefficient could not be calculated (NaN), and the scatter plot
displayed a single vertical line of data points. Consequently, the hypothesis
that operational complexity correlates with incident frequency could not be
tested, as the chosen metric offers no discriminatory power within this specific
dataset framework.

---

## Review

The experiment was successfully executed and the code correctly implemented the
planned steps. The analysis properly identified that the independent variable
('practice_count') had zero variance across the matched dataset (all values were
3), rendering the correlation calculation impossible (NaN). The findings
accurately reflect this data limitation.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

def load_dataset(filename):
    parent_path = os.path.join("..", filename)
    if os.path.exists(parent_path):
        return pd.read_csv(parent_path)
    elif os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        raise FileNotFoundError(f"Dataset {filename} not found.")

try:
    # 1. Load Datasets
    df_competencies = load_dataset('step1_sub_competencies.csv')
    df_coverage = load_dataset('step3_coverage_map.csv')

    # Clean column names
    df_competencies.columns = df_competencies.columns.str.strip()
    df_coverage.columns = df_coverage.columns.str.strip()

    # 2. Merge Datasets
    merged_df = pd.merge(
        df_competencies[['id', 'observable_practices', 'name']],
        df_coverage[['sub_competency_id', 'incident_count', 'coverage_status']],
        left_on='id',
        right_on='sub_competency_id',
        how='inner'
    )

    print(f"Merged dataset shape: {merged_df.shape}")

    # 3. Calculate 'practice_count' (Operational Complexity)
    # Assuming semicolon separation based on dataset metadata descriptions for similar fields
    def count_practices(text):
        if pd.isna(text):
            return 0
        # Split by semicolon, strip whitespace, and filter out empty strings
        items = [item for item in str(text).split(';') if item.strip()]
        return len(items)

    merged_df['practice_count'] = merged_df['observable_practices'].apply(count_practices)

    print("\nData Preview (ID, Practice Count, Incident Count):")
    print(merged_df[['id', 'practice_count', 'incident_count']])

    # 4. Calculate Spearman's Rank Correlation
    if len(merged_df) > 1 and merged_df['practice_count'].std() > 0 and merged_df['incident_count'].std() > 0:
        corr, p_value = spearmanr(merged_df['practice_count'], merged_df['incident_count'])
        print("\n=== Statistical Analysis ===")
        print(f"Spearman's Rank Correlation Coefficient: {corr:.4f}")
        print(f"P-value: {p_value:.4f}")
    else:
        corr, p_value = float('nan'), float('nan')
        print("\nCannot calculate correlation: insufficient data or zero variance.")

    # 5. Generate Scatter Plot
    plt.figure(figsize=(10, 6))
    
    # Using regplot to show the scatter and a linear regression fit to visualize the trend
    sns.regplot(
        x='practice_count',
        y='incident_count',
        data=merged_df,
        color='teal',
        ci=None, # Disable confidence interval shading for cleaner look with few points
        scatter_kws={'s': 100, 'alpha': 0.7}
    )

    # Annotate points with IDs
    for i, row in merged_df.iterrows():
        plt.text(
            row['practice_count'] + 0.1, 
            row['incident_count'] + 0.1, 
            row['id'], 
            fontsize=9
        )

    plt.title(f'Operational Complexity vs. Real-World Incident Frequency\n(Spearman r={corr:.2f}, p={p_value:.3f})')
    plt.xlabel('Operational Complexity (Count of Observable Practices)')
    plt.ylabel('Incident Frequency (Count of MITRE ATLAS Cases)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Force integer ticks for count data
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Merged dataset shape: (8, 6)

Data Preview (ID, Practice Count, Incident Count):
     id  practice_count  incident_count
0  TR-1               3              19
1  TR-2               3               1
2  TR-3               3               9
3  TR-4               3               2
4  TR-5               3               0
5  IR-1               3               8
6  IR-2               3               8
7  IR-3               3              18

Cannot calculate correlation: insufficient data or zero variance.


=== Plot Analysis (figure 1) ===
Based on the analysis of the provided plot image, here is the detailed breakdown:

### 1. Plot Type
*   **Type:** Scatter Plot.
*   **Purpose:** The plot aims to visualize the relationship (correlation) between two variables: "Operational Complexity" and "Incident Frequency." However, due to the nature of the data presented, it effectively functions as a one-dimensional distribution plot along the vertical axis.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Operational Complexity (Count of Observable Practices)"
    *   **Value Range:** The visual axis spans from roughly **2.84 to 3.16**. However, all data points are situated exactly at the value **3.00**.
    *   **Units:** Count (integer values representing practices).
*   **Y-Axis:**
    *   **Title:** "Incident Frequency (Count of MITRE ATLAS Cases)"
    *   **Value Range:** The axis is marked from **0 to 18** (with intervals of 3), covering a data range of approximately 0 to 19.
    *   **Units:** Count (integer values representing cases).

### 3. Data Trends
*   **Vertical Alignment:** The most distinct trend is that **every single data point** has an x-coordinate of **3.00**. There is absolutely no variation in the "Operational Complexity" variable across the dataset.
*   **Clusters:** While the X value is constant, the Y values (Incident Frequency) are clustered in three distinct groups:
    *   **High Frequency:** Two points located at approximately Y=18 and Y=19.
    *   **Medium Frequency:** Two points located at approximately Y=8 and Y=9.
    *   **Low/Zero Frequency:** Three points located at approximately Y=0, Y=1, and Y=2.
*   **Outliers:** Given the lack of a trendline, it is difficult to classify outliers, but the grouping suggests a high variance in incident frequency despite identical complexity scores.

### 4. Annotations and Legends
*   **Chart Title:** "Operational Complexity vs. Real-World Incident Frequency (Spearman r=nan, p=nan)".
*   **Data Point Labels:** Specific identifiers are placed next to the data points to label the entities being measured:
    *   Top Group: **TR-1**, **IR-3**
    *   Middle Group: **TR-3**, **IR-2**
    *   Bottom Group: **TR-4**, **TR-2**, **TR-5**
*   **Grid Lines:** A dashed grid is used to help visually quantify the position of the points.

### 5. Statistical Insights
*   **Correlation (r=nan):** The title states a Spearman correlation of `nan` (Not a Number). This occurs because the standard deviation of the X-variable (Operational Complexity) is zero. Since all X values are identical (3.00), there is no variation to correlate against the Y values. Mathematically, the correlation calculation involves division by zero variance.
*   **Interpretation:** The plot indicates that **Operational Complexity** (as measured by the count of observable practices) **is not a predictor** for Incident Frequency in this specific dataset. Whether an entity has high or low incident frequency, they all share the exact same complexity score of 3.
*   **Conclusion:** To find a cause for the difference in incident frequency (why TR-1 is 19 while TR-5 is 0), one would need to investigate a different independent variable, as "Operational Complexity" provides no discriminatory power here.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
