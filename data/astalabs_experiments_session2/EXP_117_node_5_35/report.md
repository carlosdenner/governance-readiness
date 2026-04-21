# Experiment 117: node_5_35

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_35` |
| **ID in Run** | 117 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:37:19.433228+00:00 |
| **Runtime** | 342.9s |
| **Parent** | `node_4_34` |
| **Children** | `node_6_70` |
| **Creation Index** | 118 |

---

## Hypothesis

> Sectoral Bias Blindspots: Despite high stakes, 'Law Enforcement' AI deployments
report significantly lower rates of 'Disparity Mitigation' compared to 'Health'
sector deployments.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.2555 (Likely False) |
| **Surprise** | -0.5934 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 29.0 |
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

**Objective:** Compare bias mitigation efforts across sensitive sectors using improved text analysis logic to capture narrative evidence.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for the 'eo13960_scored' source table.
- 2. Create a 'Sector' variable by categorizing '8_topic_area': Map terms like 'Law Enforcement', 'Justice', 'Security' to 'Law Enforcement' and 'Health' to 'Health'. Filter the dataset to these two sectors.
- 3. Implement an improved classification function for '62_disparity_mitigation' to flag Affirmative evidence (1) vs. None/Unclear (0). 
   - Treat as Affirmative (1) if the text contains keywords indicating active measures: 'test', 'eval', 'monitor', 'review', 'analy', 'assess', 'mitigat', 'audit', 'check', 'ensur', 'bias', 'fair', 'equit'.
   - Treat as Negative (0) if the text is null, empty, or explicitly indicates absence (e.g., starts with 'No', 'None', 'N/A', or contains 'not applicable' without subsequent qualification).
- 4. Calculate the proportion of affirmative mitigation evidence for both sectors.
- 5. Perform a Chi-square test (or Fisher's Exact test if counts are low) to assess the statistical significance of the difference.
- 6. Generate a bar chart comparing the mitigation rates.

### Deliverables
- Bar chart of Disparity Mitigation rates by Sector (with improved text parsing); Statistical test results.

---

## Analysis

The experiment successfully tested the 'Sectoral Bias Blindspots' hypothesis
using the EO 13960 dataset with improved text analysis logic.

The results **strongly contradict the hypothesis**:
1. **Observed Rates**: Contrary to the expectation that Law Enforcement would
have lower mitigation rates, 'Law Enforcement' deployments showed a
significantly *higher* rate of affirmative mitigation evidence (15.1%) compared
to 'Health' deployments (3.4%).
2. **Statistical Significance**: The Chi-Square test yielded a statistic of
12.11 and a p-value of 0.0005, confirming that this difference is statistically
significant.
3. **Insights**: The improved keyword search (capturing terms like 'test',
'monitor', 'eval') revealed that while overall documentation is low, Law
Enforcement agencies are nearly 4.5 times more likely to document bias testing
than Health agencies in this dataset. This may be driven by standardized vendor
language for biometric tools (e.g., 'tested extensively... face types') compared
to more custom or less documented clinical algorithms.

---

## Review

null

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def run_experiment():
    print("Starting Experiment: Sectoral Bias Blindspots (Improved Text Analysis)...")
    
    # 1. Load dataset
    filename = 'astalabs_discovery_all_data.csv'
    if not os.path.exists(filename):
        filename = '../astalabs_discovery_all_data.csv'
    
    try:
        df = pd.read_csv(filename, low_memory=False)
        print(f"Dataset loaded from: {filename}")
    except FileNotFoundError:
        print(f"Error: Dataset {filename} not found.")
        return

    # Filter for EO13960 source table
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO13960 scored records: {len(df_eo)}")

    # 2. Segment Data by Topic Area
    def get_sector(s):
        if pd.isna(s): return None
        s_str = str(s).lower()
        if any(x in s_str for x in ['law enforcement', 'justice', 'security']):
            return 'Law Enforcement'
        if 'health' in s_str:
            return 'Health'
        return None

    df_eo['analysis_sector'] = df_eo['8_topic_area'].apply(get_sector)
    
    # Filter for target sectors
    df_subset = df_eo[df_eo['analysis_sector'].isin(['Law Enforcement', 'Health'])].copy()
    
    print("\n--- Data Segmentation ---")
    print(df_subset['analysis_sector'].value_counts())

    if df_subset.empty:
        print("No data found for target sectors.")
        return

    # 3. Improved Classification Logic
    def classify_mitigation(val):
        if pd.isna(val):
            return 0
        text = str(val).strip().lower()
        if not text or text == 'nan':
            return 0
            
        # Keywords indicating affirmative action
        # Added 'human' based on previous output "human can evaluate"
        positive_keywords = [
            'test', 'eval', 'monitor', 'review', 'analy', 'assess', 
            'mitigat', 'audit', 'check', 'ensur', 'bias', 'fair', 
            'equit', 'control', 'human', 'scan', 'detect'
        ]
        
        # Negative starts
        negative_starts = ('no ', 'none', 'n/a', 'not ')
        is_negative_start = text.startswith(negative_starts) or text in ['no', 'none', 'n/a']
        
        has_positive = any(kw in text for kw in positive_keywords)
        
        if is_negative_start:
            # Heuristic: If it starts with negative but is long or contains contrast, it might be a "Soft Negative" (Qualified)
            # E.g. "None, however we..." or "None for X, but Y..."
            if len(text) > 60 and has_positive:
                return 1
            return 0
        else:
            # If it's not explicitly negative at start, check for positive content
            if has_positive:
                return 1
            return 0

    df_subset['mitigation_flag'] = df_subset['62_disparity_mitigation'].apply(classify_mitigation)

    # Validation of classification
    print("\n--- Classification Validation ---")
    print("Sample Positive (1):")
    print(df_subset[df_subset['mitigation_flag']==1]['62_disparity_mitigation'].head(3).tolist())
    print("\nSample Negative (0):")
    print(df_subset[df_subset['mitigation_flag']==0]['62_disparity_mitigation'].head(3).tolist())

    # 4. Comparative Analysis
    summary = df_subset.groupby('analysis_sector')['mitigation_flag'].agg(['count', 'sum', 'mean'])
    summary.columns = ['Total', 'Mitigated', 'Rate']
    
    print("\n--- Summary Statistics ---")
    print(summary)

    # 5. Statistical Test (Chi-Square)
    contingency = pd.crosstab(df_subset['analysis_sector'], df_subset['mitigation_flag'])
    print("\n--- Contingency Table ---")
    print(contingency)

    # Ensure valid shape
    if 0 not in contingency.columns: contingency[0] = 0
    if 1 not in contingency.columns: contingency[1] = 0
    contingency = contingency[[0, 1]] # Ensure order

    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")

    if p < 0.05:
        print("Result: Statistically significant difference detected.")
    else:
        print("Result: No statistically significant difference detected.")

    # 6. Visualization
    plt.figure(figsize=(10, 6))
    # Color: Health=Blue, LE=Red
    colors = ['#1f77b4', '#d62728']
    
    ax = summary['Rate'].plot(kind='bar', color=colors, alpha=0.8, edgecolor='black', rot=0)
    
    plt.title('Disparity Mitigation Rates: Health vs. Law Enforcement\n(Broad Keyword Search)')
    plt.ylabel('Proportion of Affirmative Mitigation Evidence')
    plt.xlabel('Sector')
    plt.ylim(0, 1.0) # Keep 0-1 scale for context, or zoom in if values are small but visible
    if summary['Rate'].max() < 0.2:
        plt.ylim(0, 0.25) # Zoom if rates are still low
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for i, v in enumerate(summary['Rate']):
        ax.text(i, v + (plt.ylim()[1]*0.02), f"{v:.1%}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting Experiment: Sectoral Bias Blindspots (Improved Text Analysis)...
Dataset loaded from: astalabs_discovery_all_data.csv
EO13960 scored records: 1757

--- Data Segmentation ---
analysis_sector
Health             233
Law Enforcement     86
Name: count, dtype: int64

--- Classification Validation ---
Sample Positive (1):
['The threshold for the biometric matching was tested extensively with a variety of face types for several months to establish a match threshold for the identification.', 'The threshold for the biometric matching was tested extensively with a variety of face types for several months to establish a match threshold for the identification.', 'The threshold for the biometric matching was tested extensively with a variety of face types for several months to establish a match threshold for the identification.']

Sample Negative (0):
[nan, nan, nan]

--- Summary Statistics ---
                 Total  Mitigated      Rate
analysis_sector                            
Health             233          8  0.034335
Law Enforcement     86         13  0.151163

--- Contingency Table ---
mitigation_flag    0   1
analysis_sector         
Health           225   8
Law Enforcement   73  13

Chi-Square Statistic: 12.1063
P-value: 0.0005
Result: Statistically significant difference detected.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Vertical Bar Plot (or Bar Chart).
*   **Purpose:** The plot is designed to compare the rates of "Affirmative Mitigation Evidence" between two distinct sectors: Health and Law Enforcement.

**2. Axes**
*   **X-Axis:**
    *   **Label:** "Sector"
    *   **Categories:** The axis displays two categorical variables: "Health" and "Law Enforcement".
*   **Y-Axis:**
    *   **Label:** "Proportion of Affirmative Mitigation Evidence"
    *   **Units:** The axis is scaled in decimals representing proportions (0.00 to 0.25), which corresponds to percentages (0% to 25%).
    *   **Range:** The visible range is from 0.00 to 0.25, with horizontal grid lines marked at intervals of 0.05.

**3. Data Trends**
*   **Tallest Bar:** The "Law Enforcement" sector is represented by the tallest bar (colored red).
*   **Shortest Bar:** The "Health" sector is represented by the shortest bar (colored blue).
*   **Comparison:** There is a significant disparity in values between the two sectors. The proportion of affirmative mitigation evidence in the Law Enforcement sector is noticeably higher than that in the Health sector.

**4. Annotations and Legends**
*   **Title:** The main title is "Disparity Mitigation Rates: Health vs. Law Enforcement," with a subtitle indicating the data source method is a "(Broad Keyword Search)."
*   **Value Labels:** Specific percentage values are annotated directly above each bar in bold text:
    *   Health: **3.4%**
    *   Law Enforcement: **15.1%**
*   **Gridlines:** Horizontal dashed grey lines serve as visual guides to estimate the height of the bars against the Y-axis.

**5. Statistical Insights**
*   **Magnitude of Difference:** The Law Enforcement sector (15.1%) has a mitigation rate that is approximately **4.4 times higher** than the Health sector (3.4%).
*   **Absolute Difference:** There is an **11.7 percentage point gap** between the two sectors.
*   **Conclusion:** Based on the broad keyword search methodology mentioned in the title, the data suggests that evidence of efforts to mitigate disparities is significantly more prevalent or more easily detectable in the Law Enforcement sector compared to the Health sector. The Health sector shows a very low proportion of evidence (less than 1 in 20 instances), whereas Law Enforcement shows a moderate proportion (roughly 1 in 7 instances).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
