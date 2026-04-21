# Experiment 76: node_4_24

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_24` |
| **ID in Run** | 76 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:57:47.220939+00:00 |
| **Runtime** | 219.4s |
| **Parent** | `node_3_5` |
| **Children** | `node_5_21` |
| **Creation Index** | 77 |

---

## Hypothesis

> Incidents resulting in 'Security' harms are significantly more strongly
associated with 'Integration Readiness' gaps compared to incidents resulting in
Non-Security harms (Privacy, Reliability, etc.), which skew towards 'Trust
Readiness'.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2459 (Likely False) |
| **Surprise** | -0.5757 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 28.0 |
| Uncertain | 1.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 27.0 |
| Definitely False | 63.0 |

---

## Experiment Plan

**Objective:** Determine if specific harm types correlate with the type of competency gap (Engineering vs. Governance).

### Steps
- 1. Load 'step3_enrichments.json'.
- 2. Create a binary category for 'harm_type': 'Security' vs. 'Non-Security' (grouping Privacy, Reliability, Bias, etc.).
- 3. Parse 'sub_competency_ids' to calculate an 'Integration Gap Ratio' (Count of IR codes / Total codes) for each incident.
- 4. Perform a T-test or Mann-Whitney U test comparing the 'Integration Gap Ratio' between Security and Non-Security incidents.

### Deliverables
- Mean Integration Gap Ratio for Security vs. Non-Security groups, statistical test results, and a boxplot visualization plan.

---

## Analysis

The experiment successfully tested the hypothesis that 'Security' harm incidents
are more strongly associated with 'Integration Readiness' gaps (engineering
failures) compared to 'Non-Security' harms (e.g., privacy, reliability, bias).
analyzing 52 case studies from 'step3_enrichments.json'.

The results do not support the hypothesis.

**Descriptive Statistics:**
- **Security Incidents (n=36):** Mean Integration Gap Ratio = 0.46 (Median =
0.47). This indicates that security incidents are caused by a roughly even mix
of Trust (governance) and Integration (engineering) failures.
- **Non-Security Incidents (n=16):** Mean Integration Gap Ratio = 0.49 (Median =
0.50). This group also shows a balanced split, slightly higher in integration
gaps than the security group, contrary to the hypothesis.

**Statistical Significance:**
- **Mann-Whitney U Test:** U=235.0, p=0.292
- **Welch's T-Test:** t=-0.66, p=0.513

Both tests yielded p-values well above the 0.05 threshold, confirming there is
no statistically significant difference in the proportion of engineering vs.
governance failures between security and non-security harm types. The boxplot
visualization further illustrates that while the 'Security' category has a wider
range of variance (including some incidents that are 100% integration failures),
the central tendencies of both groups are nearly identical around the 0.5 mark.

---

## Review

The experiment successfully tested the hypothesis that incidents resulting in
'Security' harms are more strongly associated with 'Integration Readiness'
(engineering) gaps compared to 'Non-Security' harms.

**Methodology:**
The analysis processed 52 MITRE ATLAS case studies from
'step3_enrichments.json', categorizing them into 'Security' (n=36) and 'Non-
Security' (n=16) groups. An 'Integration Gap Ratio' was calculated for each
incident based on the proportion of missing controls classified as Integration
Readiness.

**Results:**
- **Security Incidents:** Mean Integration Ratio = 0.46 (SD=0.17)
- **Non-Security Incidents:** Mean Integration Ratio = 0.49 (SD=0.14)
- **Statistical Significance:** A Mann-Whitney U test (p=0.292) and Welch's
T-test (p=0.513) both failed to find any statistically significant difference
between the groups.

**Conclusion:**
The hypothesis is **not supported**. The data indicates that both security and
non-security failures (such as privacy or reliability issues) stem from a
balanced mix of engineering (Integration) and governance (Trust) competency
gaps. There is no evidence in this dataset to suggest that security incidents
are predominantly engineering failures while other harms are predominantly
governance failures.

---

## Code

```python
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import sys

# Ensure scipy is installed for statistical tests
try:
    import scipy.stats as stats
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "scipy"])
    import scipy.stats as stats

def run_experiment():
    # Attempt to locate the file based on the instruction
    file_path = '../step3_enrichments.json'
    if not os.path.exists(file_path):
        # Fallback to current directory if the note was context-dependent
        file_path = 'step3_enrichments.json'
    
    if not os.path.exists(file_path):
        print(f"Error: Dataset not found at {file_path}")
        return

    print(f"Loading dataset from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)

    records = []
    
    for entry in data:
        # Extract harm type
        harm_type = entry.get('harm_type', 'unknown')
        if not harm_type:
            harm_type = 'unknown'
        
        harm_norm = harm_type.strip().lower()
        is_security = (harm_norm == 'security')
        
        # Extract sub_competency_ids (handle list or string)
        sub_ids = entry.get('sub_competency_ids', [])
        if isinstance(sub_ids, str):
            # Clean and split if string (e.g. "IR-1; TR-2")
            sub_ids_list = [x.strip() for x in sub_ids.replace(';', ',').split(',') if x.strip()]
        elif isinstance(sub_ids, list):
            sub_ids_list = sub_ids
        else:
            sub_ids_list = []
            
        # Calculate counts for Integration (IR) and Trust (TR)
        ir_count = 0
        tr_count = 0
        
        for pid in sub_ids_list:
            pid_upper = pid.upper().strip()
            if pid_upper.startswith('IR'):
                ir_count += 1
            elif pid_upper.startswith('TR'):
                tr_count += 1
        
        total = ir_count + tr_count
        
        # Only include records that have mappable competencies
        if total > 0:
            ratio = ir_count / total
            records.append({
                'id': entry.get('case_study_id', 'unknown'),
                'harm_type': harm_norm,
                'category': 'Security' if is_security else 'Non-Security',
                'integration_ratio': ratio,
                'total_competencies': total
            })
            
    df = pd.DataFrame(records)
    
    # Summary Statistics
    print("=== Descriptive Statistics by Category ===")
    summary = df.groupby('category')['integration_ratio'].describe()
    print(summary)
    print("\n")
    
    # Prepare groups for statistical testing
    sec_data = df[df['category'] == 'Security']['integration_ratio']
    non_sec_data = df[df['category'] == 'Non-Security']['integration_ratio']
    
    # Mann-Whitney U Test (Non-parametric)
    u_stat, p_val_mw = stats.mannwhitneyu(sec_data, non_sec_data, alternative='two-sided')
    
    # Welch's T-test (Parametric, unequal variance)
    t_stat, p_val_ttest = stats.ttest_ind(sec_data, non_sec_data, equal_var=False)
    
    print("=== Statistical Test Results ===")
    print(f"Mann-Whitney U Statistic: {u_stat}")
    print(f"Mann-Whitney P-value: {p_val_mw:.5f}")
    print(f"Welch's T-Test Statistic: {t_stat:.4f}")
    print(f"Welch's T-Test P-value: {p_val_ttest:.5f}")
    
    if p_val_mw < 0.05:
        print("Result: Statistically significant difference found.")
    else:
        print("Result: No statistically significant difference found.")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    # Combine data for plotting
    plot_data = [sec_data, non_sec_data]
    labels = [f'Security (n={len(sec_data)})', f'Non-Security (n={len(non_sec_data)})']
    
    plt.boxplot(plot_data, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor="lightblue"), 
                medianprops=dict(color="red", linewidth=1.5))
                
    plt.title('Integration Readiness Gap Ratio by Harm Category')
    plt.ylabel('Integration Ratio\n(IR Gaps / Total Gaps)')
    plt.xlabel('Harm Category')
    plt.ylim(-0.05, 1.05)  # Ratios are 0-1
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_enrichments.json
=== Descriptive Statistics by Category ===
              count      mean       std  min       25%       50%  75%       max
category                                                                       
Non-Security   16.0  0.492708  0.140761  0.2  0.400000  0.500000  0.6  0.666667
Security       36.0  0.462875  0.169418  0.2  0.333333  0.472222  0.6  1.000000


=== Statistical Test Results ===
Mann-Whitney U Statistic: 235.0
Mann-Whitney P-value: 0.29164
Welch's T-Test Statistic: -0.6612
Welch's T-Test P-value: 0.51287
Result: No statistically significant difference found.

STDERR:
<ipython-input-1-11fce066470d>:114: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(plot_data, labels=labels, patch_artist=True,


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box and Whisker Plot (Boxplot).
*   **Purpose:** The plot is designed to compare the distribution of the "Integration Readiness Gap Ratio" between two distinct categories: "Security" and "Non-Security." It visualizes the median, quartiles (interquartile range), and the full range (min/max) of the data.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Harm Category"
    *   **Categories:** Two distinct groups are plotted:
        1.  **Security** (Sample size $n=36$)
        2.  **Non-Security** (Sample size $n=16$)
*   **Y-Axis:**
    *   **Label:** "Integration Ratio (IR Gaps / Total Gaps)"
    *   **Range:** The visual axis spans from slightly below 0.0 to just above 1.0.
    *   **Ticks:** Marked at intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).
    *   **Units:** The values represent a ratio, ranging from 0 to 1 (conceptually 0% to 100%).

### 3. Data Trends
*   **Security Category (Left):**
    *   **Range:** Displays a very wide spread. The whiskers extend from a minimum of **0.2** to a maximum of **1.0**.
    *   **Interquartile Range (IQR):** The box (representing the middle 50% of data) spans roughly from **0.33 to 0.60**.
    *   **Median:** The red line indicates a median value slightly below 0.5, approximately **0.47**.
*   **Non-Security Category (Right):**
    *   **Range:** Displays a narrower spread compared to the Security group. The whiskers extend from a minimum of **0.2** to a maximum of approximately **0.67**.
    *   **Interquartile Range (IQR):** The box spans from **0.40 to 0.60**.
    *   **Median:** The red line sits exactly at **0.5**.

### 4. Annotations and Legends
*   **Title:** "Integration Readiness Gap Ratio by Harm Category" appears at the top.
*   **Sample Size ($n$):** The x-axis labels explicitly annotate the sample size for each group ($n=36$ for Security, $n=16$ for Non-Security).
*   **Red Lines:** These indicate the **median** value of each dataset.
*   **Grid Lines:** Horizontal dashed grid lines are included at 0.2 intervals to assist in estimating Y-axis values.

### 5. Statistical Insights
*   **Comparison of Medians:** The median integration ratios for both groups are quite similar (around 0.47–0.50). This suggests that the central tendency for the proportion of gaps related to integration readiness is roughly equal regardless of whether the harm category is Security or Non-Security.
*   **Variability:** The **Security** category has significantly higher variability than the **Non-Security** category. While Non-Security ratios are concentrated mostly between 0.2 and 0.67, the Security ratios span the entire upper range up to 1.0. This means some Security cases have a 100% Integration Readiness Gap ratio, which is not observed in the Non-Security group.
*   **Distribution Shape:** The Security distribution appears right-skewed (the upper whisker is much longer than the lower whisker), indicating a subset of security issues has very high integration gap ratios. The Non-Security distribution is more compact and appears somewhat symmetrical within its smaller range.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
