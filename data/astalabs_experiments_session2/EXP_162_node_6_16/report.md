# Experiment 162: node_6_16

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_16` |
| **ID in Run** | 162 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:40:18.365099+00:00 |
| **Runtime** | 231.2s |
| **Parent** | `node_5_26` |
| **Children** | None |
| **Creation Index** | 163 |

---

## Hypothesis

> The 'Stage-Gate' Monitoring Hypothesis: AI systems in the
'Operation/Maintenance' phase are significantly more likely to have 'Post-
Deployment Monitoring' in place compared to systems in the
'Development/Acquisition' phase.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.9945 (Definitely True) |
| **Surprise** | +0.0128 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 30.0 |
| Maybe True | 0.0 |
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

**Objective:** Verify if monitoring controls are correctly phased, or if operational systems lack required oversight (Governance Gap).

### Steps
- 1. Load 'eo13960_scored'.
- 2. Map '16_dev_stage' to 'Operational' (Use, Maintenance) vs 'Pre-Operational' (Development, Acquisition, Planning).
- 3. Map '56_monitor_postdeploy' to Binary (Yes/No).
- 4. Perform a Chi-square test.
- 5. Specifically analyze the 'Operational' group: what % lacks monitoring? This is a key insight.

### Deliverables
- Monitoring compliance rate for operational systems; Chi-square test comparing stages.

---

## Analysis

The experiment successfully verified the 'Stage-Gate Monitoring Hypothesis'
using the EO 13960 dataset. The code correctly mapped 992 AI systems into
'Operational' (n=641) and 'Pre-Operational' (n=351) phases and evaluated their
post-deployment monitoring status.

Key Findings:
1. **Statistical Significance**: The Chi-square test (p=0.0158) indicates a
statistically significant difference between the groups, supporting the
hypothesis that operational systems are more likely to be monitored than those
in development.
2. **Compliance Rates**: Operational systems showed a 10.5% monitoring rate,
compared to 5.7% for pre-operational systems.
3. **Critical Governance Gap**: Despite the relative difference, the absolute
numbers reveal a severe lack of oversight. A staggering **89.5% of operational
AI systems** (574 out of 641) lack reported post-deployment monitoring. The code
correctly flagged this as a 'Major Governance Gap'.

---

## Review

The experiment successfully tested the 'Stage-Gate Monitoring Hypothesis' on the
EO 13960 dataset. The implementation correctly categorized 992 AI systems into
lifecycle stages ('Operational' vs. 'Pre-Operational') and parsed the 'Post-
Deployment Monitoring' text field into binary values.

**Key Findings:**
1.  **Statistical Support**: The Chi-square test (p = 0.016) confirms a
statistically significant difference. Operational systems are indeed more likely
to have monitoring protocols in place compared to pre-operational ones.
2.  **Magnitude of Difference**: Operational systems showed a 10.5% monitoring
rate versus 5.7% for pre-operational systems.
3.  **Critical Governance Gap**: The most significant insight is the absolute
lack of oversight. A staggering **89.5%** (574/641) of AI systems currently in
the 'Operation/Maintenance' phase lack reported post-deployment monitoring. This
highlights a severe gap between the theoretical requirement for lifecycle
governance and actual agency practice.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import os

def run_experiment():
    # Attempt to locate the dataset in current or parent directory
    filename = 'astalabs_discovery_all_data.csv'
    paths = [filename, f'../{filename}']
    df = None
    for p in paths:
        if os.path.exists(p):
            print(f"Dataset found at: {p}")
            df = pd.read_csv(p, low_memory=False)
            break
            
    if df is None:
        print("Error: Dataset not found.")
        return

    # Filter for EO 13960 source
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"Total EO 13960 records: {len(df_eo)}")

    # Map '16_dev_stage' to Lifecycle Phases
    def map_stage(val):
        s = str(val).lower()
        # Operational keywords
        if any(x in s for x in ['operation', 'maintenance', 'use', 'production', 'sustainment']):
            return 'Operational'
        # Pre-Operational keywords
        if any(x in s for x in ['development', 'acquisition', 'planning', 'design', 'pilot', 'test']):
            return 'Pre-Operational'
        return None

    df_eo['Phase'] = df_eo['16_dev_stage'].apply(map_stage)
    df_analysis = df_eo.dropna(subset=['Phase']).copy()
    print(f"Records with valid phase: {len(df_analysis)}")

    # Map '56_monitor_postdeploy' to Binary Compliance (Yes/No)
    def map_monitoring(val):
        s = str(val).lower()
        # Strict negative filter first
        if any(x in s for x in ['no', 'none', 'not ', 'never', 'n/a', 'false', '0']):
            return 'No'
        # Positive keywords
        if any(x in s for x in ['yes', 'monitor', 'review', 'audit', 'check', 'ongoing', 'continuous', 'annual']):
            return 'Yes'
        # Default fallback (conservative)
        return 'No'

    df_analysis['Monitored'] = df_analysis['56_monitor_postdeploy'].apply(map_monitoring)

    # Generate Contingency Table
    ct = pd.crosstab(df_analysis['Phase'], df_analysis['Monitored'])
    print("\nContingency Table:")
    print(ct)

    # Calculate Rates
    rates = pd.crosstab(df_analysis['Phase'], df_analysis['Monitored'], normalize='index') * 100
    print("\nMonitoring Compliance Rates (%):")
    print(rates)

    # Statistical Test
    chi2, p, dof, ex = stats.chi2_contingency(ct)
    print(f"\nChi-Square Test Result:")
    print(f"Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")

    # Deliverables & Insights
    try:
        op_rate = rates.loc['Operational', 'Yes']
        pre_rate = rates.loc['Pre-Operational', 'Yes']
        print(f"\nInsight: {op_rate:.1f}% of Operational systems have monitoring vs {pre_rate:.1f}% of Pre-Operational systems.")
        
        if p < 0.05:
            print("Conclusion: Significant difference detected between stages.")
        else:
            print("Conclusion: No significant difference detected.")
            
        # Check for Governance Gap
        if op_rate < 50:
            print("ALERT: Major Governance Gap. Less than 50% of operational systems are monitored.")
    except KeyError:
        print("Insufficient data to calculate specific rates.")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Dataset found at: astalabs_discovery_all_data.csv
Total EO 13960 records: 1757
Records with valid phase: 992

Contingency Table:
Monitored         No  Yes
Phase                    
Operational      574   67
Pre-Operational  331   20

Monitoring Compliance Rates (%):
Monitored               No        Yes
Phase                                
Operational      89.547582  10.452418
Pre-Operational  94.301994   5.698006

Chi-Square Test Result:
Statistic: 5.8273
P-value: 1.5780e-02

Insight: 10.5% of Operational systems have monitoring vs 5.7% of Pre-Operational systems.
Conclusion: Significant difference detected between stages.
ALERT: Major Governance Gap. Less than 50% of operational systems are monitored.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
