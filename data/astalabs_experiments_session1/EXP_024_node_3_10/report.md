# Experiment 24: node_3_10

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_10` |
| **ID in Run** | 24 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:12:33.464050+00:00 |
| **Runtime** | 200.8s |
| **Parent** | `node_2_7` |
| **Children** | None |
| **Creation Index** | 25 |

---

## Hypothesis

> Incidents classified as 'Detection Failures' or 'Response Failures' are
significantly more likely to be mapped to 'Integration Readiness' gaps (lack of
monitoring infrastructure) compared to 'Prevention Failures', which associate
with 'Trust Readiness'.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7742 (Likely True) |
| **Posterior** | 0.5904 (Maybe True) |
| **Surprise** | -0.2133 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 4.0 |
| Maybe True | 26.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 63.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Link the ATLAS failure mode lifecycle (Prevent/Detect/Respond) to competency bundles.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Group 'failure_mode' into 'Prevention' vs 'Post-Breach' (Detection + Response).
- 3. Cross-tabulate with 'trust_integration_split' (Trust vs Integration).
- 4. Run a Fisher's Exact Test or Chi-Square test.
- 5. Generate a Heatmap of Failure Mode vs. Bundle.

### Deliverables
- Statistical test results and Heatmap visualization.

---

## Analysis

The experiment successfully processed the incident dataset to test the
relationship between failure modes (Prevention vs. Post-Breach) and competency
bundles. However, the analysis reveals that the dataset is heavily imbalanced,
preventing robust statistical validation of the hypothesis.

Key Findings:
1. **Extreme Data Skew:** 98% of the analyzed incidents (51/52) are classified
as 'prevention_failure'. Only a single incident represents a 'Post-Breach'
scenario (detection/response failure).
2. **Hypothesis Evaluation:** The hypothesis that post-breach failures map to
'Integration Readiness' could not be supported. The single post-breach incident
mapped to the 'Both' category.
3. **Statistical Significance:** The Chi-Square test yielded a p-value of 0.94,
indicating no statistically significant association. This result is driven by
the lack of variance in the data rather than a true independence of variables.
4. **Conclusion:** The current dataset predominantly describes theoretical or
preventative failure modes. To validate lifecycle-specific competency gaps, the
dataset would need to be enriched with more real-world operational incident
reports involving successful breaches and subsequent detection/response
workflows.

---

## Review

The experiment was faithfully implemented according to the plan. The code
successfully loaded the incident dataset, grouped the failure modes into
'Prevention' vs 'Post-Breach' categories, and performed the required statistical
tests (Chi-Square) and visualization (Heatmap).

**Summary of Findings:**
1.  **Hypothesis:** The hypothesis that 'Detection/Response' failures correlate
strongly with 'Integration Readiness' gaps could not be statistically supported
due to extreme data skew.
2.  **Results:** The analysis revealed that 98% of the dataset (51/52 incidents)
consists of 'Prevention Failures'. Only one incident was classified as a
'Detection Failure' (Post-Breach), and it mapped to the 'Both' bundle rather
than specifically to 'Integration Readiness'.
3.  **Conclusion:** The Chi-Square test yielded a non-significant p-value of
0.94, confirming that the current dataset is insufficient for analyzing post-
breach competency gaps. The experiment successfully identified that the
underlying data focuses almost exclusively on preventative measures, consistent
with the metadata warnings.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency, fisher_exact

# Helper to find file
def find_file(filename):
    paths = [filename, os.path.join('..', filename)]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

file_path = find_file('step3_incident_coding.csv')

if not file_path:
    print("Error: step3_incident_coding.csv not found.")
else:
    try:
        # 1. Load Data
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with shape: {df.shape}")

        # 2. Group 'failure_mode' into 'Prevention' vs 'Post-Breach'
        # Check unique values first
        print("\nUnique Failure Modes:", df['failure_mode'].unique())
        
        def group_failure(mode):
            m = str(mode).lower()
            if 'prevention' in m:
                return 'Prevention'
            elif 'detection' in m or 'response' in m:
                return 'Post-Breach'
            else:
                return 'Other'
        
        df['Failure_Phase'] = df['failure_mode'].apply(group_failure)
        
        # 3. Cross-tabulate with 'trust_integration_split'
        # Check unique values
        print("Unique Bundles:", df['trust_integration_split'].unique())
        
        # Create Contingency Table
        contingency = pd.crosstab(df['Failure_Phase'], df['trust_integration_split'])
        print("\n=== Contingency Table (Failure Phase vs Bundle) ===")
        print(contingency)
        
        # 4. Statistical Test
        # Determine which test to run based on table size and content
        # We are looking for association. Chi-Square is standard, Fisher for small samples.
        # Given the metadata warnings about skew, we likely have small cell counts.
        
        chi2, p, dof, expected = chi2_contingency(contingency)
        print(f"\nChi-Square Test Results:\nChi2 Statistic: {chi2:.4f}\nP-value: {p:.4f}\nDegrees of Freedom: {dof}")
        print("Expected Frequencies:\n", expected)
        
        # If 2x2, run Fisher's Exact as well for robustness
        if contingency.shape == (2, 2):
            odds_ratio, p_fisher = fisher_exact(contingency)
            print(f"\nFisher's Exact Test P-value: {p_fisher:.4f}")

        # 5. Generate Heatmap
        plt.figure(figsize=(8, 5))
        # Use matplotlib directly to avoid seaborn dependency issues
        plt.imshow(contingency, cmap='Blues', aspect='auto')
        plt.colorbar(label='Incident Count')
        
        # Add labels
        cols = contingency.columns.tolist()
        rows = contingency.index.tolist()
        
        plt.xticks(range(len(cols)), cols, rotation=45)
        plt.yticks(range(len(rows)), rows)
        plt.title('Heatmap of Failure Phase vs. Competency Bundle')
        plt.xlabel('Competency Bundle')
        plt.ylabel('Failure Phase')
        
        # Add text annotations
        for i in range(len(rows)):
            for j in range(len(cols)):
                plt.text(j, i, contingency.iloc[i, j], ha='center', va='center', color='black')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"An error occurred during processing: {e}")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded dataset with shape: (52, 22)

Unique Failure Modes: <StringArray>
['prevention_failure', 'detection_failure']
Length: 2, dtype: str
Unique Bundles: <StringArray>
['both', 'trust-dominant', 'integration-dominant']
Length: 3, dtype: str

=== Contingency Table (Failure Phase vs Bundle) ===
trust_integration_split  both  integration-dominant  trust-dominant
Failure_Phase                                                      
Post-Breach                 1                     0               0
Prevention                 45                     4               2

Chi-Square Test Results:
Chi2 Statistic: 0.1330
P-value: 0.9357
Degrees of Freedom: 2
Expected Frequencies:
 [[8.84615385e-01 7.69230769e-02 3.84615385e-02]
 [4.51153846e+01 3.92307692e+00 1.96153846e+00]]


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Annotated Heatmap (or contingency table visualization).
*   **Purpose:** The plot visualizes the frequency distribution of incidents across two categorical variables: "Failure Phase" and "Competency Bundle." It is designed to reveal correlations and concentrations of data points within specific intersections of these categories.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Competency Bundle"
    *   **Labels:** Three categorical labels: "both", "integration-dominant", and "trust-dominant". The labels are rotated 45 degrees for readability.
*   **Y-Axis:**
    *   **Title:** "Failure Phase"
    *   **Labels:** Two categorical labels: "Post-Breach" (top) and "Prevention" (bottom).
*   **Color Scale (Legend):**
    *   **Title:** "Incident Count"
    *   **Range:** The scale runs from 0 (white/lightest blue) to 45 (dark navy blue).

### 3. Data Trends
*   **Areas of High Concentration:**
    *   The most significant trend is heavily skewed toward the intersection of the **"Prevention"** phase and the **"both"** competency bundle. This cell contains a count of **45**, which is overwhelmingly higher than any other cell in the matrix.
*   **Areas of Low Concentration:**
    *   The **"Post-Breach"** row is nearly devoid of incidents, containing only a single incident in the "both" column and 0 in the others.
    *   The "integration-dominant" and "trust-dominant" columns show very low activity, with counts of 4 and 2 respectively (both within the Prevention phase).
*   **Overall Pattern:** The data indicates a strong clustering effect. The vast majority of observations fall into a single category combination (Prevention + both).

### 4. Annotations and Legends
*   **Cell Annotations:** Each cell in the grid is annotated with the exact numerical count of incidents (e.g., 45, 4, 2, 1, 0, 0), allowing for precise reading of the data without relying solely on color perception.
*   **Color Bar:** Located on the right side, the color bar indicates that darker shades of blue represent higher incident counts, while lighter shades represent lower counts.

### 5. Statistical Insights
*   **Dominance of Prevention Phase:** Out of the total incidents displayed ($N=52$), **98% (51/52)** occurred during the "Prevention" phase. Only **2% (1/52)** occurred during the "Post-Breach" phase.
*   **Dominance of 'both' Competency Bundle:** The "both" category accounts for **88.5% (46/52)** of all incidents.
*   **Conclusion:** This dataset suggests that failures are predominantly identified or occur during the prevention stage rather than after a breach has occurred. Furthermore, the "both" competency bundle is the primary context for these incidents, while "integration-dominant" and "trust-dominant" bundles are significantly less represented in failure reports.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
