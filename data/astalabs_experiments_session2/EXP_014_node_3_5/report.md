# Experiment 14: node_3_5

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_5` |
| **ID in Run** | 14 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T01:37:18.325249+00:00 |
| **Runtime** | 370.5s |
| **Parent** | `node_2_0` |
| **Children** | `node_4_4`, `node_4_28` |
| **Creation Index** | 15 |

---

## Hypothesis

> Development Stage Governance Decay: AI systems in the 'Operational' phase show
lower evidence of active risk monitoring compared to systems in the
'Development/Testing' phase, suggesting governance is treated as a one-time
gate.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2527 (Likely False) |
| **Surprise** | -0.5870 |
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
| Maybe False | 0.0 |
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Investigate if governance controls are sustained post-deployment.

### Steps
- 1. Load `eo13960_scored` subset.
- 2. Filter for relevant stages in `16_dev_stage` ('Operational' vs 'Development/Testing').
- 3. Select time-dependent control columns: `56_monitor_postdeploy`, `28_iqa_compliance`, `54_key_risks`.
- 4. Calculate the percentage of 'Yes' responses for these controls in each stage.
- 5. Use a Z-test for proportions to compare the two groups.

### Deliverables
- Proportion comparison table; Z-test results; Interpretation of 'Governance Decay'.

---

## Analysis

The experiment was successfully executed and provided clear evidence to evaluate
the 'Governance Decay' hypothesis. By implementing semantic mapping to interpret
verbose text fields, the analysis overcame previous parsing errors and yielded
the following insights:

1.  **Hypothesis Rejection (Maturation vs. Decay)**: The data significantly
contradicts the hypothesis that governance fades after deployment. Specifically,
'Post-Deployment Monitoring' was significantly higher (p=0.009) in the
Operational stage (10.7%) compared to the Development/Testing stage (5.7%). This
indicates a 'Governance Maturation' effect where monitoring controls are more
rigorously applied—or at least more frequently documented—once systems go live.

2.  **Statistical Significance**:
    -   **Monitoring**: z=2.63, p=0.009 (Significant; Ops > Dev).
    -   **IQA Compliance**: z=-0.80, p=0.424 (Not Significant).
    -   **Risk Assessment**: z=1.47, p=0.140 (Not Significant).

3.  **Low Baseline Compliance**: A secondary but critical finding is the low
overall rate of active governance evidence. Across all categories and stages,
affirmative evidence did not exceed 14%, suggesting that while the 'Decay'
hypothesis is false, a broader 'Governance Gap' exists across the entire
lifecycle.

The code produced all required deliverables, including the proportion
comparison, Z-test statistics, and a visualization clearly annotated with
significance levels.

---

## Review

The experiment was successfully executed and provided clear evidence to evaluate
the 'Governance Decay' hypothesis. By implementing semantic mapping to interpret
verbose text fields, the analysis overcame previous parsing errors and yielded
the following insights:

1.  **Hypothesis Rejection (Maturation vs. Decay)**: The data significantly
contradicts the hypothesis that governance fades after deployment. Specifically,
'Post-Deployment Monitoring' was significantly higher (p=0.009) in the
Operational stage (10.7%) compared to the Development/Testing stage (5.7%). This
indicates a 'Governance Maturation' effect where monitoring controls are more
rigorously applied—or at least more frequently documented—once systems go live.

2.  **Statistical Significance**:
    -   **Monitoring**: z=2.63, p=0.009 (Significant; Ops > Dev).
    -   **IQA Compliance**: z=-0.80, p=0.424 (Not Significant).
    -   **Risk Assessment**: z=1.47, p=0.140 (Not Significant).

3.  **Low Baseline Compliance**: A secondary but critical finding is the low
overall rate of active governance evidence. Across all categories and stages,
affirmative evidence did not exceed 14%, suggesting that while the 'Decay'
hypothesis is false, a broader 'Governance Gap' exists across the entire
lifecycle.

The code produced all required deliverables, including the proportion
comparison, Z-test statistics, and a visualization clearly annotated with
significance levels.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess

# Install statsmodels if not present
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

try:
    from statsmodels.stats.proportion import proportions_ztest
except ImportError:
    install("statsmodels")
    from statsmodels.stats.proportion import proportions_ztest

# 1. Load Dataset
file_path = '../astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = 'astalabs_discovery_all_data.csv'

print(f"Loading dataset from {file_path}...")
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    sys.exit(1)

# 2. Filter for EO 13960 Scored data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 subset loaded. Shape: {df_eo.shape}")

# 3. Define and Apply Stage Grouping
stage_col = '16_dev_stage'
def categorize_stage(val):
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).lower().strip()
    if any(x in val_str for x in ['operation', 'use', 'deployed']):
        return 'Operational'
    if any(x in val_str for x in ['develop', 'test', 'pilot']):
        return 'Development/Testing'
    return 'Other'

df_eo['stage_group'] = df_eo[stage_col].apply(categorize_stage)

# Create groups
group_ops = df_eo[df_eo['stage_group'] == 'Operational'].copy()
group_dev = df_eo[df_eo['stage_group'] == 'Development/Testing'].copy()

print(f"\nOperational Group Size: {len(group_ops)}")
print(f"Development/Testing Group Size: {len(group_dev)}")

if len(group_ops) == 0 or len(group_dev) == 0:
    print("Error: Insufficient data in one or both groups to perform analysis.")
    sys.exit(0)

# 4. Text Analysis Mapping Functions
def map_monitoring(val):
    if pd.isna(val):
        return 0
    v = str(val).lower()
    # Positive indicators based on debug output
    if any(x in v for x in ['intermittent', 'automated', 'established process', 'manually updated']):
        return 1
    # Negative indicators: 'no monitoring', 'not safety', 'under development'
    return 0

def map_iqa(val):
    if pd.isna(val):
        return 0
    v = str(val).lower()
    # Negative indicators first
    if any(x in v for x in ['not applicable', 'n/a', 'non-public', 'proof of concept', 'no iqa', 'not in production']):
        return 0
    # Positive indicators: check for presence of meaningful text
    # If it's not explicitly N/A and has significant length, assume some compliance is cited
    if len(v) > 10:
        return 1
    return 0

def map_risks(val):
    if pd.isna(val):
        return 0
    v = str(val).lower()
    # Negative indicators
    if any(x in v for x in ['forthcoming', 'not applicable', 'no key risks', 'n/a', 'not safety']):
        return 0
    # Positive indicators
    if len(v) > 10:
        return 1
    return 0

# Apply mappings
metrics = {
    '56_monitor_postdeploy': {'label': 'Post-Deploy Monitoring', 'func': map_monitoring},
    '28_iqa_compliance': {'label': 'IQA Compliance', 'func': map_iqa},
    '54_key_risks': {'label': 'Risk Assessment', 'func': map_risks}
}

results = []
print("\n--- Governance Control Analysis (Mapped) ---")

for col, meta in metrics.items():
    label = meta['label']
    func = meta['func']
    
    if col not in df_eo.columns:
        print(f"Skipping {label}: Column {col} not found.")
        continue
        
    # Calculate counts
    ops_success = group_ops[col].apply(func).sum()
    ops_total = len(group_ops)
    
    dev_success = group_dev[col].apply(func).sum()
    dev_total = len(group_dev)
    
    ops_prop = ops_success / ops_total if ops_total > 0 else 0
    dev_prop = dev_success / dev_total if dev_total > 0 else 0
    
    print(f"\nAnalyzing '{label}'")
    print(f"  Operational: {ops_success}/{ops_total} ({ops_prop:.2%})")
    print(f"  Dev/Testing: {dev_success}/{dev_total} ({dev_prop:.2%})")
    
    # Z-test
    stat, pval = 0.0, 1.0
    significant = False
    if (ops_success > 0 or dev_success > 0) and (ops_success < ops_total or dev_success < dev_total):
        stat, pval = proportions_ztest([ops_success, dev_success], [ops_total, dev_total])
        significant = pval < 0.05
        print(f"  Z-test: z={stat:.4f}, p={pval:.4e}")
    else:
        print("  Z-test: Skipped (no variance)")

    # Interpretation
    if significant:
        if ops_prop < dev_prop:
            interp = "DECAY DETECTED (Ops < Dev)"
        else:
            interp = "MATURATION DETECTED (Ops > Dev)"
    else:
        interp = "No Significant Difference"
    print(f"  Result: {interp}")

    results.append({
        'Metric': label,
        'Ops_Prop': ops_prop,
        'Dev_Prop': dev_prop,
        'P_Value': pval,
        'Significant': significant
    })

# 5. Visualization
if results:
    res_df = pd.DataFrame(results)
    
    x = np.arange(len(res_df))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, res_df['Ops_Prop'], width, label='Operational', color='#4c72b0')
    rects2 = ax.bar(x + width/2, res_df['Dev_Prop'], width, label='Dev/Testing', color='#dd8452')
    
    ax.set_ylabel('Active Governance (Proportion)')
    ax.set_title('Governance Lifecycle Analysis: Operational vs Development')
    ax.set_xticks(x)
    ax.set_xticklabels(res_df['Metric'])
    ax.legend()
    ax.set_ylim(0, 1.15)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    
    # Add p-values
    for i, row in res_df.iterrows():
        h = max(row['Ops_Prop'], row['Dev_Prop'])
        txt = f"p={row['P_Value']:.3f}"
        if row['Significant']: txt += " *"
        ax.text(i, h + 0.05, txt, ha='center', fontweight='bold' if row['Significant'] else 'normal')

    plt.tight_layout()
    plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
EO 13960 subset loaded. Shape: (1757, 196)

Operational Group Size: 627
Development/Testing Group Size: 351

--- Governance Control Analysis (Mapped) ---

Analyzing 'Post-Deploy Monitoring'
  Operational: 67/627 (10.69%)
  Dev/Testing: 20/351 (5.70%)
  Z-test: z=2.6283, p=8.5824e-03
  Result: MATURATION DETECTED (Ops > Dev)

Analyzing 'IQA Compliance'
  Operational: 73/627 (11.64%)
  Dev/Testing: 47/351 (13.39%)
  Z-test: z=-0.7990, p=4.2428e-01
  Result: No Significant Difference

Analyzing 'Risk Assessment'
  Operational: 56/627 (8.93%)
  Dev/Testing: 22/351 (6.27%)
  Z-test: z=1.4749, p=1.4024e-01
  Result: No Significant Difference


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Grouped Bar Chart (Clustered Bar Chart).
*   **Purpose:** To compare the proportion of "Active Governance" between two distinct environments ("Operational" vs. "Dev/Testing") across three different stages of the governance lifecycle.

### 2. Axes
*   **X-Axis:**
    *   **Title/Labels:** The axis represents categorical stages of the lifecycle: **"Post-Deploy Monitoring"**, **"IQA Compliance"**, and **"Risk Assessment"**.
    *   **Range:** Categorical (Nominal).
*   **Y-Axis:**
    *   **Title:** "Active Governance (Proportion)".
    *   **Units:** The axis uses decimal proportions ranging from **0.0 to roughly 1.1** (representing 0% to 110%).
    *   **Note:** While the axis is scaled to 1.0 (100%), the actual data points are quite low, clustering below 0.2 (20%).

### 3. Data Trends
*   **Post-Deploy Monitoring:**
    *   **Operational (Blue):** Higher at **10.7%**.
    *   **Dev/Testing (Orange):** Lower at **5.7%**.
    *   *Trend:* Operational governance is nearly double that of Dev/Testing in this stage.
*   **IQA Compliance:**
    *   **Operational (Blue):** **11.6%**.
    *   **Dev/Testing (Orange):** Slightly higher at **13.4%**.
    *   *Trend:* This is the only category where Dev/Testing shows a higher proportion than Operational, though the difference is small.
*   **Risk Assessment:**
    *   **Operational (Blue):** **8.9%**.
    *   **Dev/Testing (Orange):** **6.3%**.
    *   *Trend:* Operational is slightly higher.
*   **Overall Pattern:** Active governance proportions are generally low across all categories, with no single metric exceeding 14%.

### 4. Annotations and Legends
*   **Legend:** Located in the top right corner.
    *   Blue: **Operational**
    *   Orange: **Dev/Testing**
*   **Bar Labels:** Exact percentage values are annotated directly above each bar (e.g., "10.7%", "13.4%").
*   **Statistical Annotations (P-values):** Floating above each pair of bars is a p-value indicating the statistical significance of the difference between the two groups.
    *   Post-Deploy Monitoring: **p=0.009 \*** (includes an asterisk).
    *   IQA Compliance: **p=0.424**.
    *   Risk Assessment: **p=0.140**.

### 5. Statistical Insights
*   **Significant Difference:** The difference in **Post-Deploy Monitoring** is statistically significant (**p=0.009**), indicated by the p-value being less than the standard threshold of 0.05 and the presence of an asterisk. This suggests that "Active Governance" is reliably higher in the Operational environment compared to Dev/Testing for this specific stage.
*   **Insignificant Differences:**
    *   For **IQA Compliance** (p=0.424) and **Risk Assessment** (p=0.140), the differences between Operational and Dev/Testing are likely not statistically significant. The observed differences could be due to random chance.
*   **General Observation:** The vast amount of empty space in the chart (y-axis goes to 1.0, but data peaks at ~0.13) highlights that "Active Governance" is relatively rare in both environments across all three lifecycle stages analyzed.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
