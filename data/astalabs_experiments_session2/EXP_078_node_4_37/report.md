# Experiment 78: node_4_37

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_37` |
| **ID in Run** | 78 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T04:39:15.282693+00:00 |
| **Runtime** | 194.8s |
| **Parent** | `node_3_17` |
| **Children** | `node_5_33` |
| **Creation Index** | 79 |

---

## Hypothesis

> The 'Shadow AI' Monitoring Gap: Federal AI systems that lack a formal Authority
to Operate (ATO) are significantly less likely to have Post-Deployment
Monitoring mechanisms in place compared to authorized systems, identifying a
risk management void in experimental deployments.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9677 (Definitely True) |
| **Posterior** | 0.9890 (Definitely True) |
| **Surprise** | +0.0255 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 28.0 |
| Maybe True | 2.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 60.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Investigate if systems without formal authorization (ATO) lack critical post-deployment monitoring controls.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'eo13960_scored'.
- 2. Group data by '40_has_ato' (Yes vs No/Pending/Unknown).
- 3. Analyze '56_monitor_postdeploy' to create a binary compliance flag.
- 4. Compare monitoring rates across ATO status groups.
- 5. Perform a Chi-Square test of independence.

### Deliverables
- Contingency table of ATO Status vs. Post-Deployment Monitoring; Bar chart showing monitoring gaps; Statistical significance metrics.

---

## Analysis

The experiment successfully validated the 'Shadow AI' Monitoring Gap hypothesis.
Analyzing 1,757 records from the EO 13960 dataset, the study found a strong
statistical link between formal authorization (ATO) and post-deployment
monitoring. Systems with an Authority to Operate (n=638) had a monitoring rate
of 14.4%, whereas unauthorized or pending systems (n=1,119) had a significantly
lower rate of 4.8%. A Chi-Square test yielded a p-value of 4.62e-12, confirming
the difference is highly significant. While the hypothesis is
supported—authorized systems are ~3 times more likely to be monitored—the data
reveals a broader governance failure, as over 85% of even the authorized systems
lack reported monitoring mechanisms.

---

## Review

The experiment faithfully implemented the plan and successfully validated the
'Shadow AI' Monitoring Gap hypothesis. The code appropriately handled the
unstructured text in the '56_monitor_postdeploy' column by implementing keyword-
based categorization after an initial failure with strict equality checks. The
analysis of 1,757 records from the EO 13960 dataset revealed a statistically
significant disparity (Chi-Square p=4.62e-12): systems with formal Authority to
Operate (ATO) are nearly three times as likely to have post-deployment
monitoring (14.4%) compared to unauthorized or pending systems (4.8%). While the
hypothesis is confirmed, the results also highlight a critical systemic issue
where over 85% of even the authorized federal AI systems lack reported
monitoring mechanisms.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Cleaning Functions ---

def clean_ato(val):
    s = str(val).strip().lower()
    if s == 'yes':
        return 'Has ATO'
    if 'approved enclave' in s:
        return 'Has ATO'
    return 'No ATO'

def clean_monitoring(val):
    s = str(val).strip().lower()
    # Positive indicators
    if 'automated' in s or 'established' in s or 'intermittent' in s:
        return 'Monitored'
    # Negative indicators (explicit 'no', 'under development', or irrelevant justification)
    # 'nan' will fall through to here as it doesn't contain positive keywords
    return 'Not Monitored'

# Apply cleaning
eo_data['ato_status'] = eo_data['40_has_ato'].apply(clean_ato)
eo_data['monitoring_status'] = eo_data['56_monitor_postdeploy'].apply(clean_monitoring)

# Debug: Check distribution
print("ATO Status Distribution:")
print(eo_data['ato_status'].value_counts())
print("\nMonitoring Status Distribution:")
print(eo_data['monitoring_status'].value_counts())

# Create Contingency Table
contingency_table = pd.crosstab(eo_data['ato_status'], eo_data['monitoring_status'])
print("\nContingency Table:")
print(contingency_table)

# Calculate percentages for visualization
# We want P(Monitored | ATO Status)
ato_summary = pd.crosstab(eo_data['ato_status'], eo_data['monitoring_status'], normalize='index') * 100
print("\nMonitoring Percentages by ATO Status:")
print(ato_summary)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Visualization
if 'Monitored' in ato_summary.columns:
    plt.figure(figsize=(10, 6))
    # Plotting the 'Monitored' percentage
    # Reorder index if needed to have Has ATO vs No ATO
    desired_order = ['Has ATO', 'No ATO']
    # Filter to only existing indices
    existing_order = [x for x in desired_order if x in ato_summary.index]
    
    plot_data = ato_summary.loc[existing_order, 'Monitored']
    
    bars = plt.bar(plot_data.index, plot_data.values, color=['#4CAF50', '#F44336'])
    
    plt.title('Impact of ATO Authorization on Post-Deployment Monitoring')
    plt.xlabel('Authorization Status')
    plt.ylabel('Percentage of Systems with Monitoring (%)')
    plt.ylim(0, 100)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("No 'Monitored' category found to plot.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: ATO Status Distribution:
ato_status
No ATO     1119
Has ATO     638
Name: count, dtype: int64

Monitoring Status Distribution:
monitoring_status
Not Monitored    1611
Monitored         146
Name: count, dtype: int64

Contingency Table:
monitoring_status  Monitored  Not Monitored
ato_status                                 
Has ATO                   92            546
No ATO                    54           1065

Monitoring Percentages by ATO Status:
monitoring_status  Monitored  Not Monitored
ato_status                                 
Has ATO            14.420063      85.579937
No ATO              4.825737      95.174263

Chi-Square Test Results:
Chi2 Statistic: 47.8400
P-value: 4.6247e-12


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart (or Column Chart).
*   **Purpose:** The plot is designed to compare a specific metric ("Percentage of Systems with Monitoring") across two distinct categorical groups based on their authorization status ("Has ATO" vs. "No ATO").

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Label:** "Authorization Status"
    *   **Categories:** Two discrete categories: "Has ATO" and "No ATO".
*   **Y-Axis (Vertical):**
    *   **Label:** "Percentage of Systems with Monitoring (%)"
    *   **Units:** Percentage points.
    *   **Range:** The axis scale runs from **0 to 100**, with major tick marks at intervals of 20 (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Tallest Bar:** The "Has ATO" group is the tallest, represented by a green bar reaching a value of **14.4%**.
*   **Shortest Bar:** The "No ATO" group is the shortest, represented by a red bar reaching a value of **4.8%**.
*   **Pattern:** There is a positive correlation between having an ATO (Authority to Operate) and the presence of post-deployment monitoring. The bar representing authorized systems is noticeably larger than the unauthorized group.

### 4. Annotations and Legends
*   **Chart Title:** "Impact of ATO Authorization on Post-Deployment Monitoring" appears at the top, defining the context of the analysis.
*   **Data Labels:** Specific percentage values are annotated directly above each bar ("14.4%" and "4.8%"), eliminating the need to estimate values based on the grid lines.
*   **Grid Lines:** Horizontal dashed grey lines are provided at every 20% increment to assist with visual scaling.
*   **Color Coding:** The bars use distinct colors to reinforce the categories:
    *   **Green:** Used for "Has ATO" (generally associated with positive/compliant status).
    *   **Red:** Used for "No ATO" (generally associated with negative/non-compliant status).

### 5. Statistical Insights
*   **Relative Impact:** Systems with an ATO are **3 times more likely** (14.4% vs 4.8%) to have post-deployment monitoring compared to systems without an ATO. This suggests that the authorization process effectively encourages or enforces monitoring standards.
*   **Low Overall Compliance:** Despite the "Has ATO" group performing better, the absolute values are quite low. Even among authorized systems, only **14.4%** have monitoring, leaving **85.6%** of authorized systems without it (based on this metric). This indicates that while ATO improves the odds of monitoring, it does not guarantee it.
*   **Conclusion:** The authorization process (ATO) acts as a significant differentiator for system visibility, but there is substantial room for improvement in monitoring coverage across all systems, regardless of authorization status.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
