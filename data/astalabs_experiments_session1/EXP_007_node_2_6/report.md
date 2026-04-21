# Experiment 7: node_2_6

| Property | Value |
|---|---|
| **Experiment ID** | `node_2_6` |
| **ID in Run** | 7 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T21:41:03.697781+00:00 |
| **Runtime** | 161.4s |
| **Parent** | `node_1_0` |
| **Children** | `node_3_12`, `node_3_19` |
| **Creation Index** | 8 |

---

## Hypothesis

> Incidents characterized by 'Integration Readiness' failures exhibit
significantly higher attack complexity (measured by technique count) than 'Trust
Readiness' failures, suggesting that engineering gaps are exploited via complex
technical chains while governance gaps are exploited via simpler mechanisms.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6774 (Maybe True) |
| **Posterior** | 0.6260 (Maybe True) |
| **Surprise** | -0.0596 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 22.0 |
| Uncertain | 8.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 39.0 |
| Uncertain | 51.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Compare the mean technique count of incidents across competency split categories.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Filter for incidents where 'trust_integration_split' is either 'Trust-Dominant' or 'Integration-Dominant' (exclude 'Both' to ensure separation).
- 3. Extract the 'technique_count' for both groups.
- 4. Visualize distributions using box plots.
- 5. Perform an independent samples t-test (or Mann-Whitney U test if non-normal) to compare the means of 'technique_count' between the two groups.

### Deliverables
- Box plot visualization, mean/median technique counts per group, and hypothesis test results.

---

## Analysis

The experiment successfully loaded the 'step3_incident_coding.csv' dataset and
performed the requested comparative analysis. Filtering for 'Trust-Dominant' and
'Integration-Dominant' splits resulted in small sample sizes (n=2 and n=4,
respectively), as the majority of incidents (46/52) were classified as 'Both'
and thus excluded.

Descriptive statistics reveal a notable difference in attack complexity:
Integration-Dominant incidents had a higher mean technique count (8.25) compared
to Trust-Dominant incidents (5.50), directionally supporting the hypothesis that
engineering/architecture failures are associated with more complex attack
chains. However, due to the extremely small sample size, the Mann-Whitney U test
yielded a p-value of 0.1588, indicating the difference is not statistically
significant. The generated box plot visually corroborates the trend, showing a
higher median and wider interquartile range for the Integration group, but the
results should be interpreted with caution given the low statistical power.

---

## Review

The experiment was successfully executed and faithfully followed the plan. The
code correctly loaded the dataset, filtered for the specific
'trust_integration_split' categories, and performed the requested statistical
and visual analysis. While the sample sizes for the specific groups were very
small (n=2 and n=4) due to the dominance of the 'Both' category, the analysis
correctly identified this limitation and produced valid descriptive statistics
and hypothesis test results based on the available data.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys

# [debug] Print python version and current working directory to understand environment
# import os
# print(sys.version)
# print(os.getcwd())

def run_experiment():
    try:
        # Load dataset
        # Note: User specified datasets are one level above
        file_path = '../step3_incident_coding.csv'
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            # Fallback if running in a different environment structure
            df = pd.read_csv('step3_incident_coding.csv')
            
        print(f"Loaded dataset with {len(df)} records.")
        
        # Check unique values in split column to ensure correct filtering
        print(f"Unique values in 'trust_integration_split': {df['trust_integration_split'].unique()}")
        
        # Standardize strings just in case
        df['split_norm'] = df['trust_integration_split'].astype(str).str.lower().str.strip()
        
        # Filter groups
        # Target groups: 'trust-dominant' and 'integration-dominant'
        trust_group = df[df['split_norm'] == 'trust-dominant']['technique_count']
        integration_group = df[df['split_norm'] == 'integration-dominant']['technique_count']
        
        print(f"\nSample sizes:\nTrust-Dominant: {len(trust_group)}\nIntegration-Dominant: {len(integration_group)}")
        
        # Check if we have enough data
        if len(trust_group) < 2 or len(integration_group) < 2:
            print("\nWARNING: Sample sizes are too small for reliable statistical testing.")
            print("Proceeding with available data for demonstration purposes.")

        # Calculate Summary Statistics
        stats_summary = pd.DataFrame({
            'Group': ['Trust-Dominant', 'Integration-Dominant'],
            'Count': [len(trust_group), len(integration_group)],
            'Mean': [trust_group.mean(), integration_group.mean()],
            'Median': [trust_group.median(), integration_group.median()],
            'Std': [trust_group.std(), integration_group.std()]
        })
        print("\n=== Summary Statistics ===")
        print(stats_summary.to_string(index=False))
        
        # Statistical Test
        # Using Mann-Whitney U test as sample sizes are small and normality is not guaranteed
        if len(trust_group) > 0 and len(integration_group) > 0:
            u_stat, p_val = stats.mannwhitneyu(trust_group, integration_group, alternative='two-sided')
            print("\n=== Statistical Test Results (Mann-Whitney U) ===")
            print(f"U-statistic: {u_stat}")
            print(f"P-value: {p_val:.4f}")
            alpha = 0.05
            if p_val < alpha:
                print("Result: Statistically significant difference (p < 0.05)")
            else:
                print("Result: No statistically significant difference (p >= 0.05)")
        else:
            print("\nCannot perform statistical test due to empty groups.")

        # Visualization
        if len(trust_group) > 0 or len(integration_group) > 0:
            plt.figure(figsize=(10, 6))
            # Prepare data for boxplot
            data_to_plot = []
            labels = []
            
            if len(trust_group) > 0: 
                data_to_plot.append(trust_group)
                labels.append(f'Trust-Dominant\n(n={len(trust_group)})')
            if len(integration_group) > 0:
                data_to_plot.append(integration_group)
                labels.append(f'Integration-Dominant\n(n={len(integration_group)})')
            
            plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
            plt.title('Distribution of Technique Counts by Competency Split')
            plt.ylabel('Technique Count')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded dataset with 52 records.
Unique values in 'trust_integration_split': <StringArray>
['both', 'trust-dominant', 'integration-dominant']
Length: 3, dtype: str

Sample sizes:
Trust-Dominant: 2
Integration-Dominant: 4

=== Summary Statistics ===
               Group  Count  Mean  Median      Std
      Trust-Dominant      2  5.50     5.5 0.707107
Integration-Dominant      4  8.25     8.5 1.707825

=== Statistical Test Results (Mann-Whitney U) ===
U-statistic: 0.5
P-value: 0.1588
Result: No statistically significant difference (p >= 0.05)

STDERR:
<ipython-input-1-6c8b60d0efdb>:85: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data_to_plot, labels=labels, patch_artist=True)


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (also known as a Box-and-Whisker Plot).
*   **Purpose:** This plot visualizes the distribution, central tendency (median), and variability of numerical data ("Technique Count") across different categorical groups ("Competency Split"). It allows for easy comparison between the groups.

### 2. Axes
*   **Title:**
    *   **Main Title:** "Distribution of Technique Counts by Competency Split"
*   **Y-Axis:**
    *   **Label:** "Technique Count"
    *   **Range:** The axis ticks range from **5 to 10**.
*   **X-Axis:**
    *   **Label:** None explicitly named, but the categories represent the "Competency Split".
    *   **Categories:** "Trust-Dominant" and "Integration-Dominant".

### 3. Data Trends
*   **Trust-Dominant Group:**
    *   **Median:** The median line (orange) is at **5.5**.
    *   **Spread:** The distribution is very tight and low. The entire range (whiskers) spans only from **5 to 6**.
    *   **Interquartile Range (IQR):** The box (representing the middle 50% of data) spans roughly from **5.25 to 5.75**.
*   **Integration-Dominant Group:**
    *   **Median:** The median is significantly higher, at **8.5**.
    *   **Spread:** The distribution is much wider. The range (whiskers) spans from **6 to 10**.
    *   **Interquartile Range (IQR):** The box spans from **7.5 to roughly 9.25**, indicating a larger variance in technique counts compared to the first group.

### 4. Annotations and Legends
*   **Sample Size (n):** The x-axis labels include critical sample size annotations:
    *   Trust-Dominant: **(n=2)**
    *   Integration-Dominant: **(n=4)**
*   **Grid Lines:** Horizontal dashed grid lines are present at integer intervals (5, 6, 7, etc.) to aid in reading the specific values of the box plot elements.

### 5. Statistical Insights
*   **Significant Difference in Volume:** The "Integration-Dominant" group generally utilizes a higher number of techniques than the "Trust-Dominant" group. The lowest value in the Integration group (6) is equal to the highest value in the Trust group.
*   **Variability Discrepancy:** The Integration-Dominant group shows much higher variability (a range of 4 units) compared to the Trust-Dominant group (a range of 1 unit). This suggests that integration-focused competencies might allow for or require a more flexible/varied number of techniques compared to trust-focused ones.
*   **Sample Size Limitation:** It is crucial to note the extremely small sample sizes (n=2 and n=4). While the visual trend suggests a clear difference, statistical power is very low. The "box" for the n=2 group is essentially constructed from just two data points, making it an unreliable representation of a broader population distribution. Conclusions drawn from this data should be treated as preliminary or illustrative rather than definitive.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
