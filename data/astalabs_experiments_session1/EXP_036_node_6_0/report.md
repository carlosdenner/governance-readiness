# Experiment 36: node_6_0

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_0` |
| **ID in Run** | 36 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:37:35.715523+00:00 |
| **Runtime** | 229.4s |
| **Parent** | `node_5_0` |
| **Children** | None |
| **Creation Index** | 37 |

---

## Hypothesis

> The distribution of failure modes (Prevention vs. Detection/Response) differs
significantly between Trust-dominant and Integration-dominant incidents.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7581 (Likely True) |
| **Posterior** | 0.3802 (Maybe False) |
| **Surprise** | -0.4386 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 2.0 |
| Maybe True | 28.0 |
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

**Objective:** Assess if the nature of the competency gap (Trust vs Integration) influences whether the failure is a prevention failure or a detection/response failure.

### Steps
- 1. Load the dataset 'step3_incident_coding.csv' (files are in the current directory).
- 2. Create a 'failure_category' column by mapping values containing 'prevention' to 'Prevention' and those containing 'detection' or 'response' to 'Detection/Response'.
- 3. Filter the dataframe to keep only rows where 'trust_integration_split' is either 'trust-dominant' or 'integration-dominant'.
- 4. Generate a contingency table (crosstab) with 'trust_integration_split' as rows and 'failure_category' as columns.
- 5. Check the dimensions of the contingency table. If it is 2x2, perform a Fisher's Exact Test to calculate the P-value. If it is not 2x2 (e.g., if one failure category has 0 counts), output the counts and a message stating that statistical testing cannot be performed due to lack of variance.
- 6. Generate a stacked bar chart visualizing the proportion of 'Prevention' vs 'Detection/Response' for each competency split.

### Deliverables
- Contingency table of Failure Mode by Competency Split, Fisher's Exact Test results (or variance warning), and a stacked bar chart visualization.

---

## Analysis

The experiment successfully loaded the 'step3_incident_coding.csv' dataset and
attempted to test the dependency between Incident Failure Mode (Prevention vs.
Detection/Response) and Competency Gap Bundle (Trust vs. Integration). After
filtering the 52 incidents for those explicitly classified as 'trust-dominant'
or 'integration-dominant' (excluding the majority classified as 'both'), only 6
incidents remained.

The analysis of this subset revealed that 100% of the incidents resulted in
'Prevention' failures (4 Integration-dominant, 2 Trust-dominant). Because the
'Detection/Response' failure category had zero counts, the resulting contingency
table was degenerate (2x1 dimensions: [4, 0] vs [2, 0]), rendering a standard
Fisher's Exact Test for independence impossible due to a complete lack of
variance in the dependent variable.

Consequently, the hypothesis that 'Integration gaps lead to detection/response
failures' is not supported by this specific data subset. Instead, the available
evidence suggests that Prevention failures are the dominant mode for both trust
and integration competency gaps in these specific case studies.

---

## Review

The experiment successfully loaded the 'step3_incident_coding.csv' dataset and
attempted to test the dependency between Incident Failure Mode (Prevention vs.
Detection/Response) and Competency Gap Bundle (Trust vs. Integration). After
filtering the 52 incidents for those explicitly classified as 'trust-dominant'
or 'integration-dominant' (excluding the majority classified as 'both'), only 6
incidents remained.

The analysis of this subset revealed that 100% of the incidents resulted in
'Prevention' failures (4 Integration-dominant, 2 Trust-dominant). Because the
'Detection/Response' failure category had zero counts, the resulting contingency
table was degenerate (2x1 dimensions), rendering a standard Fisher's Exact Test
for independence impossible due to a complete lack of variance in the dependent
variable.

Consequently, the hypothesis that 'Integration gaps lead to detection/response
failures' is not supported by this specific data subset. Instead, the available
evidence suggests that Prevention failures are the dominant mode for both trust
and integration competency gaps in these specific case studies.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def run_experiment():
    filename = 'step3_incident_coding.csv'
    # Try current directory first, then parent if not found (robustness)
    if not os.path.exists(filename):
        if os.path.exists(f'../{filename}'):
            filename = f'../{filename}'
        else:
            print(f"Error: {filename} not found in . or ..")
            return

    try:
        # 1. Load the dataset
        df = pd.read_csv(filename)
        print(f"Dataset loaded from {filename}. Rows: {len(df)}")
        
        # 2. Preprocess / Map categories
        # Map failure_mode
        def map_failure(mode):
            if pd.isna(mode):
                return 'Other'
            mode = str(mode).lower().strip()
            if 'prevention' in mode:
                return 'Prevention'
            elif 'detection' in mode or 'response' in mode:
                return 'Detection/Response'
            else:
                return 'Other'

        df['failure_category'] = df['failure_mode'].apply(map_failure)
        
        # Normalize split
        df['trust_integration_split'] = df['trust_integration_split'].astype(str).str.lower().str.strip()
        
        # 3. Filter for specific splits
        target_splits = ['trust-dominant', 'integration-dominant']
        df_filtered = df[df['trust_integration_split'].isin(target_splits)].copy()
        
        print(f"Filtered dataset size: {len(df_filtered)}")
        print("Split distribution in filtered set:")
        print(df_filtered['trust_integration_split'].value_counts())
        print("Failure category distribution in filtered set:")
        print(df_filtered['failure_category'].value_counts())

        if len(df_filtered) == 0:
            print("No data matching filter criteria. Cannot perform test.")
            return

        # 4. Create Contingency Table
        contingency_table = pd.crosstab(
            df_filtered['trust_integration_split'], 
            df_filtered['failure_category']
        )
        
        print("\nContingency Table:")
        print(contingency_table)

        # 5. Statistical Test
        # Check shape. If we have both rows and both columns, do Fisher.
        # If we are missing columns (e.g. only Prevention), we can't do Fisher test of independence easily 2x2.
        
        row_count, col_count = contingency_table.shape
        
        if row_count == 2 and col_count == 2:
            odds_ratio, p_value = stats.fisher_exact(contingency_table)
            print(f"\nFisher's Exact Test Results:")
            print(f"Odds Ratio: {odds_ratio}")
            print(f"P-value: {p_value}")
        else:
            print("\nContingency table is not 2x2 (likely due to zero counts in one category). Cannot perform standard 2x2 Fisher's Exact Test.")
            print(f"Shape is {contingency_table.shape}")

        # 6. Visualization
        if not contingency_table.empty:
            # Align columns for consistent coloring
            # We want 'Prevention' and 'Detection/Response' if they exist
            cols_to_plot = []
            if 'Prevention' in contingency_table.columns:
                cols_to_plot.append('Prevention')
            if 'Detection/Response' in contingency_table.columns:
                cols_to_plot.append('Detection/Response')
            
            if cols_to_plot:
                ax = contingency_table[cols_to_plot].plot(kind='bar', stacked=True, figsize=(8, 6), color=['skyblue', 'salmon'])
                plt.title('Failure Mode Distribution by Competency Split')
                plt.xlabel('Competency Split')
                plt.ylabel('Count')
                plt.xticks(rotation=0)
                plt.legend(title='Failure Category')
                plt.tight_layout()
                plt.show()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Dataset loaded from step3_incident_coding.csv. Rows: 52
Filtered dataset size: 6
Split distribution in filtered set:
trust_integration_split
integration-dominant    4
trust-dominant          2
Name: count, dtype: int64
Failure category distribution in filtered set:
failure_category
Prevention    6
Name: count, dtype: int64

Contingency Table:
failure_category         Prevention
trust_integration_split            
integration-dominant              4
trust-dominant                    2

Contingency table is not 2x2 (likely due to zero counts in one category). Cannot perform standard 2x2 Fisher's Exact Test.
Shape is (2, 1)


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** The plot is designed to compare the frequency (count) of a specific failure category across two different distinct groups or "competency splits."

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Competency Split"
    *   **Labels:** The axis represents categorical data with two specific groups: "integration-dominant" and "trust-dominant".
*   **Y-Axis:**
    *   **Title:** "Count"
    *   **Range:** The axis spans from 0.0 to 4.2 (approximately), with tick marks at intervals of 0.5 (0.0, 0.5, 1.0, ..., 4.0).
    *   **Units:** Frequency/Number of occurrences.

### 3. Data Trends
*   **Tallest Bar:** The "integration-dominant" category corresponds to the tallest bar, reaching a count of exactly **4.0**.
*   **Shortest Bar:** The "trust-dominant" category corresponds to the shorter bar, reaching a count of exactly **2.0**.
*   **Pattern:** There is a clear descending trend from left to right, showing that the "integration-dominant" split has a significantly higher count compared to the "trust-dominant" split.

### 4. Annotations and Legends
*   **Chart Title:** "Failure Mode Distribution by Competency Split" – This clearly scopes the chart as an analysis of failures relative to competency configurations.
*   **Legend:** Located in the top-right corner, titled "Failure Category." It identifies the data represented by the sky-blue bars as **"Prevention"**.

### 5. Statistical Insights
*   **Relative Frequency:** The "Prevention" failure mode occurs twice as often in the "integration-dominant" setting compared to the "trust-dominant" setting (a 2:1 ratio).
*   **Distribution:**
    *   **Integration-dominant:** Represents approximately 67% (4 out of 6) of the total recorded failures shown in this view.
    *   **Trust-dominant:** Represents approximately 33% (2 out of 6) of the total recorded failures.
*   **Implication:** This suggests that "Prevention" failures are more strongly associated with systems or scenarios characterized as "integration-dominant" rather than those that are "trust-dominant."
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
