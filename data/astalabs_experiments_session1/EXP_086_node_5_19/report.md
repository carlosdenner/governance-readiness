# Experiment 86: node_5_19

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_19` |
| **ID in Run** | 86 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:17:34.797754+00:00 |
| **Runtime** | 246.2s |
| **Parent** | `node_4_23` |
| **Children** | None |
| **Creation Index** | 87 |

---

## Hypothesis

> Incidents categorized as 'Supply Chain' harms are associated with a
significantly higher number of missing sub-competencies than 'Reliability'
harms, due to the structural/systemic nature of supply chain compromises.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.5484 (Uncertain) |
| **Posterior** | 0.2459 (Likely False) |
| **Surprise** | -0.3511 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 16.0 |
| Uncertain | 4.0 |
| Maybe False | 10.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 51.0 |
| Definitely False | 39.0 |

---

## Experiment Plan

**Objective:** Compare the competency gap breadth between Supply Chain and Reliability incidents.

### Steps
- 1. Load 'step3_enrichments.json'.
- 2. Filter dataset to include only incidents where 'harm_type' is 'supply_chain' or 'reliability'.
- 3. Calculate 'sub_competency_count' for each incident.
- 4. Group by 'harm_type'.
- 5. Perform a Mann-Whitney U test to compare the distributions of competency gaps between the two harm types.

### Deliverables
- Group means, Mann-Whitney U statistic, and p-value.

---

## Analysis

The experiment successfully loaded 'step3_enrichments.json' and compared the
competency gap breadth between 'Supply Chain' and 'Reliability' incidents.

**Key Findings:**
- **Sample Size:** The dataset contains a small number of incidents for these
specific harm types (Supply Chain: n=4, Reliability: n=4).
- **Descriptive Statistics:** Contrary to the hypothesis, 'Supply Chain'
incidents had a slightly *lower* mean number of missing sub-competencies (Mean =
4.25, SD = 1.50) compared to 'Reliability' incidents (Mean = 4.50, SD = 1.00).
- **Statistical Significance:** The Mann-Whitney U test yielded a U-statistic of
7.5 and a p-value of 1.0000. This indicates that the distributions are
statistically indistinguishable.
- **Visualization:** The boxplot confirms a 'ceiling effect' where both groups
are heavily clustered around 4-5 missing sub-competencies. Reliability incidents
are more consistently high (tight box 4.5-5.0), while Supply Chain incidents
show slightly more variance (ranging down to 2).

**Conclusion:** The hypothesis that 'Supply Chain' harms are associated with a
significantly higher number of missing sub-competencies is **rejected**. In this
dataset, both harm types implicate a similar, high number of competency gaps
(approx. 4-5), with Reliability incidents appearing slightly more consistent in
their complexity.

---

## Review

The experiment was successfully executed. The code correctly identified the file
location (after an initial path error), loaded the dataset, and performed the
specified filtering and statistical analysis. The use of a Mann-Whitney U test
was appropriate for the small sample size and non-normal distribution, and the
generation of the boxplot provided visual confirmation of the statistical
results.

---

## Code

```python
import json
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# [debug]
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))

# Try loading from current directory first, as previous experiments succeeded there
file_path = 'step3_enrichments.json'
if not os.path.exists(file_path):
    # If not found, try the parent directory as per the note, though it failed last time
    file_path = '../step3_enrichments.json'

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Filter for relevant harm types
    target_harms = ['supply_chain', 'reliability']
    df_filtered = df[df['harm_type'].isin(target_harms)].copy()
    
    # Calculate sub-competency count (handle potential missing or null lists)
    # The metadata indicates 'sub_competency_ids' is the field
    def get_count(x):
        if isinstance(x, list):
            return len(x)
        elif isinstance(x, str):
            # Handle case where it might be a string representation of a list or semicolon separated
            if x.startswith('['):
                try:
                    return len(eval(x))
                except:
                    return 0
            return len(x.split(';'))
        return 0

    df_filtered['gap_count'] = df_filtered['sub_competency_ids'].apply(get_count)
    
    # Separate groups
    supply_chain_group = df_filtered[df_filtered['harm_type'] == 'supply_chain']['gap_count']
    reliability_group = df_filtered[df_filtered['harm_type'] == 'reliability']['gap_count']
    
    # Descriptive Statistics
    print(f"\n=== Descriptive Statistics (N={len(df_filtered)}) ===")
    print(f"Supply Chain (n={len(supply_chain_group)}): Mean={supply_chain_group.mean():.2f}, Std={supply_chain_group.std():.2f}")
    print(f"Reliability (n={len(reliability_group)}): Mean={reliability_group.mean():.2f}, Std={reliability_group.std():.2f}")
    
    # Mann-Whitney U Test
    # Alternative 'two-sided' is standard for generic difference testing
    u_stat, p_val = stats.mannwhitneyu(supply_chain_group, reliability_group, alternative='two-sided')
    
    print("\n=== Mann-Whitney U Test Results ===")
    print(f"U-statistic: {u_stat}")
    print(f"P-value: {p_val:.4f}")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.boxplot([supply_chain_group, reliability_group], labels=['Supply Chain', 'Reliability'])
    plt.title('Competency Gap Breadth: Supply Chain vs Reliability')
    plt.ylabel('Number of Missing Sub-Competencies')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Jitter plot for individual points
    import numpy as np
    x_supply = np.random.normal(1, 0.04, size=len(supply_chain_group))
    x_rel = np.random.normal(2, 0.04, size=len(reliability_group))
    plt.scatter(x_supply, supply_chain_group, alpha=0.6, label='Supply Chain Cases')
    plt.scatter(x_rel, reliability_group, alpha=0.6, label='Reliability Cases')
    plt.legend()
    
    plt.show()
    
except Exception as e:
    print(f"An error occurred: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Current working directory: /data
Files in current directory: ['.placeholder', 'step1_construct_definitions.md', 'step1_evidence.json', 'step1_sub_competencies.csv', 'step2_competency_statements.csv', 'step2_crosswalk_evidence.json', 'step2_crosswalk_matrix.csv', 'step3_coverage_map.csv', 'step3_coverage_map.md', 'step3_enrichments.json', 'step3_incident_coding.csv', 'step3_mitigation_gaps.csv', 'step3_tactic_frequency.csv', 'step4_propositions.csv', 'step4_propositions.json', 'step4_propositions.md', 'step5_validation.json', 'step5_validation_issues.csv', 'step5_validation_report.md']

=== Descriptive Statistics (N=8) ===
Supply Chain (n=4): Mean=4.25, Std=1.50
Reliability (n=4): Mean=4.50, Std=1.00

=== Mann-Whitney U Test Results ===
U-statistic: 7.5
P-value: 1.0000

STDERR:
<ipython-input-1-bd95367f1bf3>:67: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot([supply_chain_group, reliability_group], labels=['Supply Chain', 'Reliability'])


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
**Type:** This is a **Box Plot (Box-and-Whisker Plot)** overlaid with a **Strip Plot (Jittered Scatter Plot)**.
**Purpose:** The plot compares the distribution of "Missing Sub-Competencies" between two distinct groups: "Supply Chain" and "Reliability."
*   The **box plot** component summarizes the distribution statistically (showing the median, quartiles, and potential statistical outliers).
*   The **strip plot** component (the colored dots) displays the individual data points, allowing the viewer to see the exact sample size and density, which helps visualize overlapping data that a box plot alone might hide.

### 2. Axes
*   **X-Axis:**
    *   **Label/Title:** Represents the categories. The labels are **"Supply Chain"** and **"Reliability"**.
*   **Y-Axis:**
    *   **Title:** "Number of Missing Sub-Competencies".
    *   **Range:** The visual axis ranges from **2.0 to 5.0** (likely representing integer counts). Grid lines appear at intervals of 0.5.

### 3. Data Trends
*   **Supply Chain Group (Left):**
    *   **Concentration:** There is a heavy concentration of data points at the value of **5**.
    *   **Distribution:** The box indicates that the upper quartile (75th percentile) and likely the median are capped at 5. The lower quartile (25th percentile) appears to be around 4.25.
    *   **Outliers:** There is a notable outlier visible at the value of **2**.
*   **Reliability Group (Right):**
    *   **Concentration:** Similar to Supply Chain, the vast majority of data points are clustered at the value of **5**.
    *   **Distribution:** The box is tighter than the Supply Chain group. The top of the box and median are at 5, while the bottom of the box (25th percentile) is at 4.5.
    *   **Outliers:** There is a single outlier visible at the value of **3**.

### 4. Annotations and Legends
*   **Title:** "Competency Gap Breadth: Supply Chain vs Reliability".
*   **Legend:** Located in the bottom right corner:
    *   **Blue Dots:** Represent "Supply Chain Cases".
    *   **Orange Dots:** Represent "Reliability Cases".
*   **Grid Lines:** Horizontal dashed grey lines facilitate reading the Y-axis values.
*   **Box Plot Markings:** The black open circles (distinct from the colored dots) denote statistical outliers identified by the box plot calculation (typically 1.5 * IQR).

### 5. Statistical Insights
*   **Ceiling Effect:** Both groups exhibit a strong "ceiling effect," where the median value is the maximum possible value shown (5). This suggests that for the majority of cases in both Supply Chain and Reliability, the "Competency Gap Breadth" is quite high (5 missing sub-competencies).
*   **Variance:** The "Supply Chain" group shows slightly more variance (spread) than the "Reliability" group. While both are centered at 5, Supply Chain cases drop as low as 2, whereas Reliability cases only drop to 3.
*   **Comparative Consistency:** The Reliability group is more consistently clustered at the high end. The box for Reliability is shorter (ranging from 4.5 to 5) compared to Supply Chain (ranging from ~4.25 to 5), indicating that the middle 50% of the Reliability data is more condensed.
*   **Conclusion:** In the context of this data, both domains suffer from significant competency gaps, but the gaps are slightly more uniform and severe in the Reliability domain, whereas the Supply Chain domain has a few exceptional cases with fewer gaps.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
