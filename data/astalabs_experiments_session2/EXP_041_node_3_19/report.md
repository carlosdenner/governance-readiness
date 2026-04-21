# Experiment 41: node_3_19

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_19` |
| **ID in Run** | 41 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:00:30.734150+00:00 |
| **Runtime** | 277.0s |
| **Parent** | `node_2_3` |
| **Children** | `node_4_18`, `node_4_31` |
| **Creation Index** | 42 |

---

## Hypothesis

> Operational AI systems (in 'O&M' stage) demonstrate significantly lower
adherence to 'Monitoring' and 'Real-World Testing' requirements compared to
systems currently in 'Development', indicating a 'compliance debt' in legacy
deployments.

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

**Objective:** Assess whether legacy (O&M) systems lag behind developing systems in active governance controls.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` (EO13960 subset).
- 2. Group `16_dev_stage` into 'Legacy' (Operation and Maintenance) and 'New' (Planning/Development).
- 3. Convert `56_monitor_postdeploy` and `53_real_world_testing` to binary integers.
- 4. Perform a Chi-Square test of independence for each control against the development stage groups.

### Deliverables
- Bar chart of compliance rates by Development Stage; Chi-Square test statistics.

---

## Analysis

The experiment successfully tested the hypothesis regarding governance
compliance across development stages. The programmer correctly implemented text-
parsing logic to convert verbose descriptions in the EO 13960 dataset into
binary indicators for 'Monitoring' and 'Real-World Testing'.

The results **contradicted** the hypothesis that Legacy (O&M) systems lag
behind. Instead, Legacy systems demonstrated significantly **higher** compliance
rates than systems in Development:
- **Monitoring Post-Deployment**: Legacy (85.9%) vs. New (62.9%), p=0.003.
- **Real-World Testing**: Legacy (91.0%) vs. New (24.2%), p<0.001.

These findings indicate that rather than suffering from 'compliance debt',
operational systems have likely already cleared governance gates required for
deployment, while systems in development are still maturing. The Chi-Square
tests confirmed these differences were statistically significant.

---

## Review

The experiment was executed faithfully and successfully adapted to the data
quality issues identified in the debugging phase. The programmer correctly
implemented text-parsing logic to convert verbose descriptions in the EO 13960
dataset into binary indicators for 'Monitoring' and 'Real-World Testing'.

The results **contradicted** the hypothesis that Legacy (O&M) systems lag
behind. Instead, Legacy systems demonstrated significantly **higher** compliance
rates than systems in Development:
- **Monitoring Post-Deployment**: Legacy (85.9%) vs. New (62.9%), p=0.003.
- **Real-World Testing**: Legacy (91.0%) vs. New (24.2%), p<0.001.

These findings indicate that rather than suffering from 'compliance debt',
operational systems have likely already cleared governance gates required for
deployment, while systems in development are still maturing. The Chi-Square
tests confirmed these differences were statistically significant.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import sys

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# Filter for EO13960 data subset
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# --- 1. Map Development Stage ---
def map_stage(stage):
    if pd.isna(stage):
        return np.nan
    s = str(stage).lower().strip()
    
    # Legacy / O&M
    if any(x in s for x in ['operation', 'production', 'mission']):
        return 'Legacy (O&M)'
    
    # New / Development
    if any(x in s for x in ['acquisition', 'development', 'initiated', 'implementation', 'planned']):
        return 'New (Dev/Plan)'
        
    return np.nan

eo_df['Stage_Group'] = eo_df['16_dev_stage'].apply(map_stage)

# --- 2. Map Controls to Binary ---

def map_monitoring(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    
    # Negatives
    if 'no monitoring' in s or 'not safety' in s:
        return 0
    
    # Positives (Intermittent, Automated, Established, Under development)
    if any(x in s for x in ['intermittent', 'automated', 'established', 'under development']):
        return 1
        
    return np.nan

def map_testing(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    
    # Positives: Real-world / Operational environment
    if 'performance evaluation' in s or 'impact evaluation' in s or s == 'yes':
        return 1
    
    # Negatives: No testing, Benchmark only (explicitly not operational), Waived
    if 'no testing' in s or 'benchmark' in s or 'waived' in s or 'not safety' in s:
        return 0
        
    return np.nan

eo_df['Monitor_Bin'] = eo_df['56_monitor_postdeploy'].apply(map_monitoring)
eo_df['Test_Bin'] = eo_df['53_real_world_testing'].apply(map_testing)

# --- 3. Analysis Loop ---
controls = {
    'Monitor_Bin': 'Monitoring Post-Deployment',
    'Test_Bin': 'Real-World Testing'
}

results = []

for col_var, col_label in controls.items():
    # Filter for rows that have both a Stage and a valid Answer (0 or 1)
    subset = eo_df.dropna(subset=['Stage_Group', col_var]).copy()
    
    if len(subset) == 0:
        print(f"No valid data for {col_label}")
        continue
        
    contingency = pd.crosstab(subset['Stage_Group'], subset[col_var])
    
    # Check if we have enough data dimensions
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        print(f"Insufficient variance for {col_label}. Shape: {contingency.shape}")
        print(contingency)
        continue
        
    chi2, p, dof, ex = chi2_contingency(contingency)
    
    # Calculate Compliance Rates
    rates = subset.groupby('Stage_Group')[col_var].mean()
    counts = subset.groupby('Stage_Group')[col_var].count()
    
    legacy_rate = rates.get('Legacy (O&M)', 0)
    new_rate = rates.get('New (Dev/Plan)', 0)
    
    results.append({
        'Control': col_label,
        'Legacy_Rate': legacy_rate,
        'New_Rate': new_rate,
        'p_value': p,
        'Legacy_N': counts.get('Legacy (O&M)', 0),
        'New_N': counts.get('New (Dev/Plan)', 0)
    })
    
    print(f"\n--- Analysis: {col_label} ---")
    print(contingency)
    print(f"Legacy Compliance: {legacy_rate:.2%} (n={counts.get('Legacy (O&M)', 0)})")
    print(f"New    Compliance: {new_rate:.2%} (n={counts.get('New (Dev/Plan)', 0)})")
    print(f"Chi-Square p-value: {p:.5f}")

# --- 4. Plotting ---
if results:
    res_df = pd.DataFrame(results)
    
    x = np.arange(len(res_df))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, res_df['Legacy_Rate'], width, label='Legacy (O&M)', color='indianred')
    rects2 = ax.bar(x + width/2, res_df['New_Rate'], width, label='New (Dev/Plan)', color='steelblue')
    
    ax.set_ylabel('Compliance Rate (1=Yes, 0=No/Partial)')
    ax.set_title('Governance Compliance by Development Stage (EO 13960)')
    ax.set_xticks(x)
    ax.set_xticklabels(res_df['Control'])
    ax.legend()
    ax.set_ylim(0, 1.15)
    
    # Annotate p-values
    for i, row in res_df.iterrows():
        max_h = max(row['Legacy_Rate'], row['New_Rate'])
        sig_text = "*" if row['p_value'] < 0.05 else "ns"
        ax.text(i, max_h + 0.05, f"p={row['p_value']:.3f}\n({sig_text})", ha='center')
        
    plt.tight_layout()
    plt.show()
    
    # Interpret results
    print("\n--- Conclusion ---")
    for i, row in res_df.iterrows():
        if row['p_value'] < 0.05:
            direction = "Lower" if row['Legacy_Rate'] < row['New_Rate'] else "Higher"
            print(f"{row['Control']}: Significant difference. Legacy systems show {direction} compliance.")
        else:
            print(f"{row['Control']}: No significant difference detected.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: 
--- Analysis: Monitoring Post-Deployment ---
Monitor_Bin     0.0  1.0
Stage_Group             
Legacy (O&M)     11   67
New (Dev/Plan)   23   39
Legacy Compliance: 85.90% (n=78)
New    Compliance: 62.90% (n=62)
Chi-Square p-value: 0.00314

--- Analysis: Real-World Testing ---
Test_Bin        0.0  1.0
Stage_Group             
Legacy (O&M)      7   71
New (Dev/Plan)   47   15
Legacy Compliance: 91.03% (n=78)
New    Compliance: 24.19% (n=62)
Chi-Square p-value: 0.00000

--- Conclusion ---
Monitoring Post-Deployment: Significant difference. Legacy systems show Higher compliance.
Real-World Testing: Significant difference. Legacy systems show Higher compliance.


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Grouped Bar Chart (also known as a Clustered Bar Chart).
*   **Purpose:** The plot is designed to compare quantitative values (Compliance Rate) across two categorical variables: the stage of development (x-axis) and the status/age of the system (Legacy vs. New).

### 2. Axes
*   **X-Axis:**
    *   **Label:** Development Stages.
    *   **Categories:** "Monitoring Post-Deployment" and "Real-World Testing."
*   **Y-Axis:**
    *   **Label:** "Compliance Rate (1=Yes, 0=No/Partial)."
    *   **Units:** The unit is a ratio or probability ranging from 0 to 1.
    *   **Range:** The visual axis spans from 0.0 to approximately 1.15, with tick marks at 0.0, 0.2, 0.4, 0.6, 0.8, and 1.0.

### 3. Data Trends
*   **Legacy (O&M) Dominance:** In both development stages presented, the "Legacy (O&M)" group (represented by red bars) demonstrates a significantly higher compliance rate compared to the "New (Dev/Plan)" group (represented by blue bars).
*   **Monitoring Post-Deployment:**
    *   **Legacy:** Shows a high compliance rate, approximately **0.85**.
    *   **New:** Shows a moderate compliance rate, approximately **0.62**.
*   **Real-World Testing:**
    *   **Legacy:** Shows the highest compliance rate on the chart, appearing to be slightly above **0.90**.
    *   **New:** Shows the lowest compliance rate on the chart, appearing to be roughly **0.25**.
*   **Largest Disparity:** The gap between Legacy and New systems is most drastic in the "Real-World Testing" category.

### 4. Annotations and Legends
*   **Legend:** Located in the top right corner.
    *   **Red/Deep Pink:** Represents "Legacy (O&M)" (Operations & Maintenance).
    *   **Blue:** Represents "New (Dev/Plan)" (Development/Planning).
*   **Annotations (Statistical Significance):**
    *   Above the **Monitoring Post-Deployment** group, there is an annotation reading **"p=0.003 (*)"**.
    *   Above the **Real-World Testing** group, there is an annotation reading **"p=0.000 (*)"**.
    *   The **(*)** symbol is a standard convention used to denote that the p-value has met a specific threshold for statistical significance (usually p < 0.05).

### 5. Statistical Insights
*   **Significant Difference in Compliance:** The presence of p-values (p=0.003 and p=0.000) indicates that the difference in compliance rates between "Legacy" and "New" systems is statistically significant in both categories. The difference is not due to random chance.
*   **Legacy Systems Outperform New Systems:** Contrary to what might be expected in some technological contexts (where newer usually equals better), "Legacy" systems currently in the Operations & Maintenance phase are complying with Governance EO 13960 at a much higher rate than "New" systems currently in the planning or development phase.
*   **Critical Gap in Testing:** The extremely low p-value (0.000) and the visual disparity in the "Real-World Testing" category suggest a systemic issue or bottleneck where new projects in the development pipeline are failing to meet governance standards during the testing phase, whereas established systems have largely satisfied these requirements.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
