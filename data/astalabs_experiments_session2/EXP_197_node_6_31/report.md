# Experiment 197: node_6_31

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_31` |
| **ID in Run** | 197 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:19:56.454372+00:00 |
| **Runtime** | 213.1s |
| **Parent** | `node_5_15` |
| **Children** | None |
| **Creation Index** | 198 |

---

## Hypothesis

> The 'Monitoring Maturity' Gap: AI systems in the 'Operational' phase are
significantly more likely to have documented 'Post-Deployment Monitoring' than
systems in 'Development' (planned monitoring), validating that monitoring is a
late-stage implementation rather than a design-phase requirement.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7903 (Likely True) |
| **Posterior** | 0.9286 (Definitely True) |
| **Surprise** | +0.1659 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 6.0 |
| Maybe True | 24.0 |
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

**Objective:** Check if monitoring is designed-in or added-on by comparing development stages.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'eo13960_scored'.
- 2. Bin `16_dev_stage` into 'Operational' and 'Development' (similar to the Legacy Gap experiment).
- 3. Clean `56_monitor_postdeploy` into binary (Yes/Documented vs. No/NA).
- 4. Perform a Chi-square test to see if Operational systems have higher monitoring documentation rates.
- 5. Calculate the compliance gap percentage.

### Deliverables
- Contingency table, Chi-square statistics, and a comparative bar chart of monitoring readiness.

---

## Analysis

The experiment successfully tested the 'Monitoring Maturity' Gap hypothesis
using the EO 13960 dataset. The code classified 1,327 AI systems into
'Operational' (N=627) and 'Development' (N=700) stages and analyzed the presence
of documented post-deployment monitoring.

**Findings:**
1.  **Hypothesis Supported:** The data confirms that monitoring is treated as a
late-stage implementation. Operational systems are significantly more likely to
have documented monitoring strategies compared to those in development.
2.  **Statistical Significance:** The Chi-square test (Statistic=13.51,
p=0.00024) reveals a statistically significant difference between the groups.
3.  **Magnitude of Effect:** Systems in the 'Operational' phase are
approximately **2 times more likely** (Odds Ratio = 2.07) to have documented
monitoring than those in the 'Development' phase.
4.  **Compliance Rates:**
    - **Operational:** 12.4% (78/627) have documented monitoring.
    - **Development:** 6.4% (45/700) have documented monitoring.

**Conclusion:** The hypothesis is **supported**. The significant gap suggests
that monitoring frameworks are often not 'designed-in' from the start but are
added closer to or after deployment. However, the low absolute rate (only 12.4%
for operational systems) indicates a broader systemic maturity gap, where the
vast majority of federal AI systems—even those currently in use—lack documented
performance or safety monitoring.

---

## Review

The experiment successfully tested the 'Monitoring Maturity' Gap hypothesis
using the EO 13960 dataset. The code classified 1,327 AI systems into
'Operational' (N=627) and 'Development' (N=700) stages and analyzed the presence
of documented post-deployment monitoring.

**Findings:**
1.  **Hypothesis Supported:** The data confirms that monitoring is treated as a
late-stage implementation. Operational systems are significantly more likely to
have documented monitoring strategies compared to those in development.
2.  **Statistical Significance:** The Chi-square test (Statistic=13.51,
p=0.00024) reveals a statistically significant difference between the groups.
3.  **Magnitude of Effect:** Systems in the 'Operational' phase are
approximately **2 times more likely** (Odds Ratio = 2.07) to have documented
monitoring than those in the 'Development' phase.
4.  **Compliance Rates:**
    - **Operational:** 12.4% (78/627) have documented monitoring.
    - **Development:** 6.4% (45/700) have documented monitoring.

**Conclusion:** The hypothesis is **supported**. The significant gap suggests
that monitoring frameworks are often not 'designed-in' from the start but are
added closer to or after deployment. However, the low absolute rate (only 12.4%
for operational systems) indicates a broader systemic maturity gap, where the
vast majority of federal AI systems—even those currently in use—lack documented
performance or safety monitoring.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

def run_experiment():
    # Load dataset
    file_path = '../astalabs_discovery_all_data.csv'
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        try:
            df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        except FileNotFoundError:
            print("Dataset not found.")
            return

    # Filter for EO 13960 scored data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    
    # 1. Process '16_dev_stage'
    # Define mapping
    # Operational: Operation, Maintenance, Use, Deployed
    # Development: Development, Acquisition, Planning, Initiation, Pilot
    def bin_stage(val):
        if pd.isna(val):
            return None
        s = str(val).lower()
        if any(x in s for x in ['oper', 'use', 'maint', 'deploy']):
            return 'Operational'
        elif any(x in s for x in ['dev', 'plan', 'acq', 'init', 'pilot', 'test']):
            return 'Development'
        return None # Exclude retired/unknown

    df_eo['stage_bin'] = df_eo['16_dev_stage'].apply(bin_stage)
    
    # Filter relevant rows
    df_analysis = df_eo.dropna(subset=['stage_bin']).copy()
    
    # 2. Process '56_monitor_postdeploy' (Monitoring Documentation)
    # Heuristic: treat typical negative responses as 0, substantive text as 1
    negatives = ['no', 'none', 'n/a', 'na', 'not applicable', '0', '-', 'false', 'unknown', 'tbd']
    
    def bin_monitoring(val):
        if pd.isna(val):
            return 0
        s = str(val).strip().lower()
        if s in negatives:
            return 0
        # Check for "No ..." sentences that are actually negations
        if s.startswith('no ') and len(s) < 20:
            return 0
        return 1

    df_analysis['monitor_bin'] = df_analysis['56_monitor_postdeploy'].apply(bin_monitoring)
    
    # 3. Contingency Table
    contingency = pd.crosstab(df_analysis['stage_bin'], df_analysis['monitor_bin'])
    # Ensure columns exist (handle case where one might be missing)
    if 0 not in contingency.columns: contingency[0] = 0
    if 1 not in contingency.columns: contingency[1] = 0
    contingency = contingency[[0, 1]]
    contingency.columns = ['No Documentation', 'Has Documentation']
    
    print("--- Contingency Table (Stage vs Monitoring) ---")
    print(contingency)
    print("\n")

    # 4. Summary Statistics
    summary = contingency.copy()
    summary['Total'] = summary['No Documentation'] + summary['Has Documentation']
    summary['% Documented'] = (summary['Has Documentation'] / summary['Total']) * 100
    
    print("--- Summary Statistics ---")
    print(summary[['Total', '% Documented']])
    print("\n")

    # 5. Chi-square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print("--- Chi-Square Test Results ---")
    print(f"Chi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    if p < 0.05:
        print("Result: Statistically Significant")
    else:
        print("Result: Not Statistically Significant")

    # 6. Odds Ratio
    # (Operational_Yes * Development_No) / (Operational_No * Development_Yes)
    try:
        op_yes = contingency.loc['Operational', 'Has Documentation']
        op_no = contingency.loc['Operational', 'No Documentation']
        dev_yes = contingency.loc['Development', 'Has Documentation']
        dev_no = contingency.loc['Development', 'No Documentation']
        
        # Handle division by zero
        if op_no == 0 or dev_yes == 0:
             print("Odds Ratio: Undefined (Zero count in denominator)")
        else:
             or_val = (op_yes * dev_no) / (op_no * dev_yes)
             print(f"Odds Ratio (Operational vs Development): {or_val:.4f}")
    except KeyError:
        print("Error calculating odds ratio: Missing category keys.")

    # 7. Visualization
    plt.figure(figsize=(8, 6))
    bars = plt.bar(summary.index, summary['% Documented'], color=['#ff9999', '#66b3ff'])
    plt.title('Documented Monitoring by Lifecycle Stage')
    plt.ylabel('% with Documented Monitoring')
    plt.xlabel('Lifecycle Stage')
    plt.ylim(0, max(summary['% Documented']) * 1.2)
    
    # Add value labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Contingency Table (Stage vs Monitoring) ---
             No Documentation  Has Documentation
stage_bin                                       
Development               655                 45
Operational               549                 78


--- Summary Statistics ---
             Total  % Documented
stage_bin                       
Development    700      6.428571
Operational    627     12.440191


--- Chi-Square Test Results ---
Chi-square Statistic: 13.5072
P-value: 2.3765e-04
Result: Statistically Significant
Odds Ratio (Operational vs Development): 2.0680


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the analysis:

**1. Plot Type**
*   **Type:** Bar Plot (Vertical Column Chart).
*   **Purpose:** This chart is designed to compare a quantitative metric (percentage of documented monitoring) across two distinct categories (lifecycle stages).

**2. Axes**
*   **X-Axis (Horizontal):**
    *   **Title:** "Lifecycle Stage"
    *   **Labels:** Two categorical variables: "Development" and "Operational".
*   **Y-Axis (Vertical):**
    *   **Title:** "% with Documented Monitoring"
    *   **Units:** Percentage (%).
    *   **Range:** The axis spans from 0 to 15 (visual limit), with numbered tick marks appearing at intervals of 2, ranging specifically from 0 to 14.

**3. Data Trends**
*   **Tallest Bar:** The "Operational" category represents the highest value, visually dominating the chart.
*   **Shortest Bar:** The "Development" category is the lowest value, appearing roughly half the height of the Operational bar.
*   **Pattern:** There is a clear upward trend in the frequency of documented monitoring as a project or system moves from the Development stage to the Operational stage.

**4. Annotations and Legends**
*   **Value Labels:** Specific percentage values are annotated above each bar to provide exact data points:
    *   Above Development: **6.4%**
    *   Above Operational: **12.4%**
*   **Legend:** There is no separate legend key; the categories are identified directly via the X-axis labels.

**5. Statistical Insights**
*   **Comparison:** Systems in the "Operational" stage are nearly twice as likely to have documented monitoring compared to those in the "Development" stage ($12.4\% \div 6.4\% \approx 1.94$).
*   **Absolute Difference:** There is a 6.0 percentage point gap between the two stages.
*   **Inference:** This distribution suggests that while monitoring documentation occurs during development, it becomes a much higher priority or requirement once a system goes live (Operational). This likely aligns with industry standards where operational stability requires more rigorous documentation and oversight than the experimental development phase.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
