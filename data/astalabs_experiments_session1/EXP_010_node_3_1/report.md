# Experiment 10: node_3_1

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_1` |
| **ID in Run** | 10 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T21:49:01.963708+00:00 |
| **Runtime** | 172.9s |
| **Parent** | `node_2_1` |
| **Children** | `node_4_1` |
| **Creation Index** | 11 |

---

## Hypothesis

> 'Prevention Failures' are predominantly associated with governance (Trust) gaps,
whereas 'Detection' and 'Response' failures are significantly more associated
with engineering (Integration) gaps.

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

**Objective:** Map failure modes to competency domains.

### Steps
- 1. Load 'step3_enrichments.json'.
- 2. Categorize 'failure_mode' into two groups: 'Prevention' vs. 'Post-Prevention' (Detection/Response).
- 3. Calculate 'integration_ratio' for each incident based on 'sub_competency_ids'.
- 4. Compare the mean 'integration_ratio' between the two failure mode groups using a t-test or Mann-Whitney U test.

### Deliverables
- Mean integration ratios by failure mode group, statistical test results.

---

## Analysis

The experiment successfully loaded the dataset and categorized the 52 incidents
by failure mode. However, the data revealed an extreme class imbalance: 51
incidents were classified as 'prevention_failure' and only 1 as
'detection_failure' (with 0 response failures).

Descriptive statistics show:
- **Prevention Failures (n=51):** Mean Integration Ratio of 0.46 (SD=0.14),
indicating a balanced mix of Trust and Integration competency gaps, rather than
being predominantly Trust-based as hypothesized.
- **Post-Prevention Failures (n=1):** Mean Integration Ratio of 1.0, indicating
a purely engineering/integration gap for the single detection failure observed.

Due to the insufficient sample size of the second group (n=1), the planned Mann-
Whitney U test could not be performed. Consequently, while the single data point
for detection failure aligns with the hypothesis (pure Integration), the
hypothesis regarding prevention failures being predominantly Trust-based is not
strongly supported (actual split is ~54% Trust / 46% Integration). The
hypothesis cannot be statistically validated given the current dataset
limitations.

---

## Review

The experiment was faithfully implemented, but the pre-existing data skew (51
prevention failures vs. 1 detection failure) rendered the planned statistical
significance test (Mann-Whitney U) impossible to perform.

**Findings:**
1. **Prevention Failures (n=51):** The mean 'Integration Ratio' was 0.46
(SD=0.14). This indicates that prevention failures are caused by a nearly
balanced mix of governance (54%) and engineering (46%) gaps, rather than being
'predominantly' Trust/Governance gaps as hypothesized.
2. **Detection Failures (n=1):** The single incident had an Integration Ratio of
1.0, aligning with the hypothesis that detection failures are engineering gaps,
but n=1 is statistically insignificant.

**Conclusion:** The hypothesis is **not supported** regarding prevention
failures (which are mixed/balanced) and **inconclusive** regarding detection
failures due to insufficient data.

---

## Code

```python
import json
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind
import matplotlib.pyplot as plt
import os

# Define file path (one level up as per instructions)
file_path = '../step3_enrichments.json'

# Check if file exists, otherwise try current directory (fallback)
if not os.path.exists(file_path):
    file_path = 'step3_enrichments.json'

print(f"Loading dataset from: {file_path}")

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Successfully loaded {len(data)} records.")
    
    # Process data into a list of dictionaries for DataFrame creation
    processed_rows = []
    
    for entry in data:
        # Extract failure mode
        f_mode = entry.get('failure_mode', 'unknown')
        
        # Extract sub-competencies
        # Metadata mentions 'sub_competency_ids' or 'llm_sub_competencies'. 
        # We check which one is a list or parseable string.
        sc_ids = entry.get('sub_competency_ids', [])
        if not sc_ids:
            sc_ids = entry.get('llm_sub_competencies', [])
            
        # Ensure sc_ids is a list
        if isinstance(sc_ids, str):
            # formatting might be "['TR-1', 'IR-2']" or "TR-1;IR-2"
            if sc_ids.startswith('['):
                try:
                    sc_ids = eval(sc_ids)
                except:
                    sc_ids = []
            elif ';' in sc_ids:
                sc_ids = sc_ids.split(';')
            else:
                sc_ids = [sc_ids]
        
        if not isinstance(sc_ids, list):
            sc_ids = []
            
        # Calculate Integration Ratio
        # Heuristic: TR-xx is Trust, IR-xx is Integration
        ir_count = sum(1 for x in sc_ids if 'IR-' in str(x))
        tr_count = sum(1 for x in sc_ids if 'TR-' in str(x))
        total_count = ir_count + tr_count
        
        if total_count > 0:
            integration_ratio = ir_count / total_count
        else:
            integration_ratio = np.nan # No competencies mapped
            
        processed_rows.append({
            'case_study_id': entry.get('case_study_id'),
            'failure_mode': f_mode,
            'integration_ratio': integration_ratio,
            'total_competencies': total_count
        })
    
    df = pd.DataFrame(processed_rows)
    
    # Drop rows with no competencies mapped if any (though we expect them to have mappings)
    df_clean = df.dropna(subset=['integration_ratio'])
    
    # Categorize Failure Modes
    # Group 1: Prevention
    # Group 2: Post-Prevention (Detection, Response)
    
    group_prevention = df_clean[df_clean['failure_mode'] == 'prevention_failure']
    group_post_prevention = df_clean[df_clean['failure_mode'].isin(['detection_failure', 'response_failure'])]
    
    # Summary Stats
    print("\n--- Descriptive Statistics ---")
    print(f"Total Incidents with Mapped Competencies: {len(df_clean)}")
    
    print(f"\nGroup 1: Prevention Failure (n={len(group_prevention)})")
    if not group_prevention.empty:
        print(f"Mean Integration Ratio: {group_prevention['integration_ratio'].mean():.4f}")
        print(f"Std Dev: {group_prevention['integration_ratio'].std():.4f}")
    
    print(f"\nGroup 2: Post-Prevention (Detection/Response) (n={len(group_post_prevention)})")
    if not group_post_prevention.empty:
        print(f"Mean Integration Ratio: {group_post_prevention['integration_ratio'].mean():.4f}")
        print(f"Std Dev: {group_post_prevention['integration_ratio'].std():.4f}")
        
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.boxplot([group_prevention['integration_ratio'], group_post_prevention['integration_ratio']], 
                labels=['Prevention', 'Post-Prevention (Detect/Respond)'])
    plt.title('Integration Competency Ratio by Failure Mode')
    plt.ylabel('Integration Ratio (0=All Trust, 1=All Integration)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Statistical Test
    print("\n--- Statistical Test Results ---")
    if len(group_prevention) > 1 and len(group_post_prevention) > 1:
        u_stat, p_val = mannwhitneyu(group_prevention['integration_ratio'], 
                                     group_post_prevention['integration_ratio'], 
                                     alternative='two-sided')
        print(f"Mann-Whitney U Statistic: {u_stat}")
        print(f"P-Value: {p_val:.5f}")
        
        if p_val < 0.05:
            print("Result: Statistically Significant Difference.")
        else:
            print("Result: No Statistically Significant Difference.")
    else:
        print("Insufficient sample size in one or both groups to perform Mann-Whitney U test.")
        print("Note: Previous exploration indicated a severe skew (51/52 prevention failures).")

except Exception as e:
    print(f"An error occurred: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_enrichments.json
Successfully loaded 52 records.

--- Descriptive Statistics ---
Total Incidents with Mapped Competencies: 52

Group 1: Prevention Failure (n=51)
Mean Integration Ratio: 0.4617
Std Dev: 0.1434

Group 2: Post-Prevention (Detection/Response) (n=1)
Mean Integration Ratio: 1.0000
Std Dev: nan

--- Statistical Test Results ---
Insufficient sample size in one or both groups to perform Mann-Whitney U test.
Note: Previous exploration indicated a severe skew (51/52 prevention failures).

STDERR:
<ipython-input-1-e430d38043e6>:102: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot([group_prevention['integration_ratio'], group_post_prevention['integration_ratio']],


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (also known as a box-and-whisker plot).
*   **Purpose:** To display the distribution of numerical data (Integration Competency Ratio) across different categorical groups (Failure Modes), highlighting the median, quartiles, and variability of the data.

### 2. Axes
*   **X-Axis:**
    *   **Labels:** The axis categorizes the data into two groups: **"Prevention"** and **"Post-Prevention (Detect/Respond)"**.
    *   **Value Range:** N/A (Categorical).
*   **Y-Axis:**
    *   **Title/Label:** "Integration Ratio (0=All Trust, 1=All Integration)". This indicates a normalized scale where lower values imply reliance on trust and higher values imply reliance on integration.
    *   **Value Range:** The axis scale runs from **0.2 to 1.0**, with grid lines marked every 0.1 units.

### 3. Data Trends
*   **Prevention Group:**
    *   **High Variability:** This group shows a significant spread in data. The box (representing the Interquartile Range or IQR) spans from approximately **0.33 to 0.60**.
    *   **Median:** The median value (indicated by the orange line inside the box) is centered at **0.5**.
    *   **Range:** The whiskers (representing the minimum and maximum non-outlier values) extend from **0.2 to 0.8**.
    *   **Pattern:** This indicates that for "Prevention" scenarios, there is a diverse mix of strategies ranging from high trust to high integration.

*   **Post-Prevention (Detect/Respond) Group:**
    *   **Zero Variability:** The plot for this category is compressed into a single flat line at the top of the chart.
    *   **Value:** The median, quartiles, and whiskers all seemingly converge at **1.0**.
    *   **Pattern:** This suggests that nearly every data point in the "Post-Prevention" category has an Integration Ratio of 1.0 (All Integration).

### 4. Annotations and Legends
*   **Title:** "Integration Competency Ratio by Failure Mode" appears at the top, summarizing the chart's content.
*   **Grid Lines:** Horizontal and vertical dashed grid lines are present to assist in reading specific values.
*   **Color Coding:** Standard box plot coloring is used—black outlines for the box and whiskers, and an orange line to denote the median.

### 5. Statistical Insights
*   **Distinct Operational Modes:** There is a stark contrast between how the system handles "Prevention" versus "Post-Prevention."
    *   **Prevention** is nuanced: It employs a balanced approach, with a median split evenly between trust and integration (0.5), but varies widely depending on the specific case (ranging from 0.2 to 0.8).
    *   **Post-Prevention** is absolute: Once a failure reaches the "Detect/Respond" stage, the system shifts entirely to "All Integration" (Ratio = 1.0). There is no reliance on "Trust" (value 0) in this phase.
*   **Consistency:** The "Post-Prevention" phase is highly consistent/deterministic compared to the "Prevention" phase, which is highly variable/probabilistic.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
