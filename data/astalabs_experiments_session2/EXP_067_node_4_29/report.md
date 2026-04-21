# Experiment 67: node_4_29

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_29` |
| **ID in Run** | 67 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T04:12:40.649120+00:00 |
| **Runtime** | 244.2s |
| **Parent** | `node_3_22` |
| **Children** | `node_5_10`, `node_5_51` |
| **Creation Index** | 68 |

---

## Hypothesis

> Commercial AI (COTS) deployments in federal agencies are significantly less
likely to possess Transparency controls (Code Access, Data Documentation)
compared to Custom/In-house developed AI, despite having similar rates of
Security controls (ATO), indicating a 'Black Box' trade-off.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8629 (Likely True) |
| **Posterior** | 0.4588 (Uncertain) |
| **Surprise** | -0.4849 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 15.0 |
| Maybe True | 15.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 60.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Compare transparency vs. security control compliance rates between Commercial and Custom AI deployments.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `source_table` == 'eo13960_scored'.
- 2. Group data by `10_commercial_ai` (Commercial vs. Government/Custom).
- 3. Define binary flags for Transparency (`38_code_access`, `34_data_docs`) and Security (`40_has_ato`).
- 4. Calculate the percentage of 'Yes' responses for each group.
- 5. Perform Chi-square tests of independence for each control to verify if the difference is statistically significant.

### Deliverables
- 1. Contingency tables for Code Access, Data Docs, and ATO by Source Type.
- 2. Chi-square test statistics and p-values.
- 3. Bar chart comparing compliance rates.

---

## Analysis

The experiment successfully tested the hypothesis using the `37_custom_code`
column as a proxy for Commercial (No) vs. Custom (Yes) development. The analysis
of 967 systems (245 Commercial, 722 Custom) yielded mixed support for the
hypothesis:

1. **Transparency (Supported):** The 'Black Box' nature of Commercial AI was
strongly confirmed. Commercial systems had negligible transparency compliance,
with 0.8% for Code Access and 0.0% for Data Documentation. Custom systems scored
significantly higher (6.4% and 2.2% respectively, p<0.05), though overall
compliance was poor for both groups.

2. **Security (Not Supported):** The hypothesis that Security controls (ATO)
would be similar was rejected. Custom systems showed a significantly higher ATO
compliance rate (59.8%) compared to Commercial systems (42.0%) with a p-value <
0.001.

**Conclusion:** While the 'Black Box' opacity of commercial tools is confirmed,
the data suggests they do not match the security governance rigor of custom-
developed government systems either. Commercial AI in this dataset appears less
compliant across both transparency and security dimensions.

---

## Review

The experiment was successfully executed and provided a rigorous test of the
hypothesis using '37_custom_code' as a reliable proxy for Commercial vs. Custom
development.

**Findings:**
1. **Transparency Gap (Confirmed):** The data strongly supports the hypothesis
that Commercial (COTS) AI lacks transparency. Commercial systems demonstrated
negligible compliance for 'Code Access' (0.8%) and 'Data Documentation' (0.0%),
significantly lower than Custom (Gov) systems (6.4% and 2.2% respectively,
p<0.05).

2. **Security Parity (Rejected):** The hypothesis that security controls (ATO)
would be similar was effectively falsified. Custom systems exhibited a
significantly higher Authority to Operate (ATO) compliance rate (59.8%) compared
to Commercial systems (42.0%), with a p-value < 0.001.

**Conclusion:** The results indicate that Commercial AI deployments in this
dataset do not merely represent a 'Black Box' trade-off (high security, low
transparency) but rather lag behind custom government-developed systems in
*both* security governance and transparency controls.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import sys

# --- 1. Load Dataset ---
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# Filter for EO 13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Subset Shape: {eo_df.shape}")

# --- 2. Define Groups using '37_custom_code' ---
# Hypothesis: 'No' Custom Code -> Commercial (COTS), 'Yes' Custom Code -> Custom (Gov)
group_col = '37_custom_code'
valid_groups = ['Yes', 'No']

# Filter data to only include valid 'Yes'/'No' responses
analysis_df = eo_df[eo_df[group_col].isin(valid_groups)].copy()

# Map to readable labels
analysis_df['group'] = analysis_df[group_col].map({'No': 'Commercial (COTS)', 'Yes': 'Custom (Gov)'})
print(f"Analysis Subset Shape: {analysis_df.shape}")
print(analysis_df['group'].value_counts())

# --- 3. Statistical Analysis ---
targets = {
    'Transparency: Code Access': '38_code_access',
    'Transparency: Data Docs': '34_data_docs',
    'Security: ATO': '40_has_ato'
}

def normalize_response(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    # Treat definitive affirmative answers as 1
    if val_str in ['yes', 'y', 'true', '1']:
        return 1
    return 0

stats_results = []
plot_data = {}

print("\n--- Chi-Square Test Results ---")

for label, col in targets.items():
    if col not in analysis_df.columns:
        print(f"Warning: Column {col} not found. Skipping.")
        continue
        
    # Normalize
    analysis_df[f'clean_{col}'] = analysis_df[col].apply(normalize_response)
    
    # Create Contingency Table
    # Rows: Groups, Cols: Compliance (0, 1)
    contingency = pd.crosstab(analysis_df['group'], analysis_df[f'clean_{col}'])
    
    # Ensure 0 and 1 columns exist
    for c in [0, 1]:
        if c not in contingency.columns:
            contingency[c] = 0
            
    # Calculate percentages
    totals = contingency.sum(axis=1)
    # contingency[1] is the count of 'Yes'
    compliance_rates = (contingency[1] / totals * 100)
    
    # Chi-square test
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    
    stats_results.append({
        'Control': label,
        'Comm_Rate': compliance_rates.get('Commercial (COTS)', 0),
        'Custom_Rate': compliance_rates.get('Custom (Gov)', 0),
        'p_value': p
    })
    
    plot_data[label] = compliance_rates
    
    print(f"\nControl: {label}")
    print(contingency)
    print(f"  Commercial (COTS) Compliance: {compliance_rates.get('Commercial (COTS)', 0):.1f}%")
    print(f"  Custom (Gov) Compliance:      {compliance_rates.get('Custom (Gov)', 0):.1f}%")
    print(f"  p-value: {p:.4e}")
    if p < 0.05:
        print("  -> Statistically Significant")
    else:
        print("  -> Not Significant")

# --- 4. Visualization ---
results_df = pd.DataFrame(stats_results)
if not results_df.empty:
    labels = [r['Control'] for r in stats_results]
    comm_vals = [r['Comm_Rate'] for r in stats_results]
    cust_vals = [r['Custom_Rate'] for r in stats_results]
    p_vals = [r['p_value'] for r in stats_results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, comm_vals, width, label='Commercial (COTS)', color='#ff7f0e')
    rects2 = ax.bar(x + width/2, cust_vals, width, label='Custom (Gov)', color='#1f77b4')

    ax.set_ylabel('Compliance Rate (%)')
    ax.set_title('Transparency vs Security: Commercial vs Custom AI (EO 13960)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add p-values
    for i, p in enumerate(p_vals):
        height = max(comm_vals[i], cust_vals[i]) + 3
        p_text = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.text(i, height, p_text, ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()
else:
    print("No results to plot.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: EO 13960 Subset Shape: (1757, 196)
Analysis Subset Shape: (967, 197)
group
Custom (Gov)         722
Commercial (COTS)    245
Name: count, dtype: int64

--- Chi-Square Test Results ---

Control: Transparency: Code Access
clean_38_code_access    0   1
group                        
Commercial (COTS)     243   2
Custom (Gov)          676  46
  Commercial (COTS) Compliance: 0.8%
  Custom (Gov) Compliance:      6.4%
  p-value: 1.0059e-03
  -> Statistically Significant

Control: Transparency: Data Docs
clean_34_data_docs    0   1
group                      
Commercial (COTS)   245   0
Custom (Gov)        706  16
  Commercial (COTS) Compliance: 0.0%
  Custom (Gov) Compliance:      2.2%
  p-value: 3.9416e-02
  -> Statistically Significant

Control: Security: ATO
clean_40_has_ato     0    1
group                      
Commercial (COTS)  142  103
Custom (Gov)       290  432
  Commercial (COTS) Compliance: 42.0%
  Custom (Gov) Compliance:      59.8%
  p-value: 1.8774e-06
  -> Statistically Significant


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is a detailed analysis:

### 1. Plot Type
This is a **grouped bar chart**. Its purpose is to compare the "Compliance Rate" between two distinct groups of AI systems ("Commercial" vs. "Custom") across three different performance or regulatory categories.

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Compliance Rate (%)"
    *   **Range:** 0 to 100, with increments of 20 marked by dashed grid lines.
*   **X-Axis:**
    *   **Labels:** The axis represents three distinct categories:
        1.  "Transparency: Code Access"
        2.  "Transparency: Data Docs"
        3.  "Security: ATO" (Authority to Operate)
    *   **Format:** The labels are slightly rotated to accommodate length and improve readability.

### 3. Data Trends
*   **Overall Pattern:** Across all three categories, **Custom (Gov)** systems (represented by the blue bars) demonstrate higher compliance rates compared to **Commercial (COTS)** systems (represented by the orange bars).
*   **Tallest Bars:** The "Security: ATO" category shows the highest compliance rates for both groups.
    *   Custom (Gov) reaches exactly **60%**.
    *   Commercial (COTS) is slightly above **40%**.
*   **Shortest Bars:** The Transparency categories show extremely low compliance rates overall.
    *   "Transparency: Data Docs" is the lowest, with Commercial systems showing near **0%** and Custom systems showing a very minimal rate (likely **<5%**).
    *   "Transparency: Code Access" is similarly low, with Commercial systems near **1-2%** and Custom systems roughly around **6-7%**.

### 4. Annotations and Legends
*   **Legend:** Located in the top right corner:
    *   **Orange:** Commercial (COTS) - Commercial Off-The-Shelf
    *   **Blue:** Custom (Gov) - Custom Government developed
*   **Statistical Annotations:** p-values are placed above each group of bars to indicate statistical significance:
    *   Above "Transparency: Code Access": **p=0.001**
    *   Above "Transparency: Data Docs": **p=0.039**
    *   Above "Security: ATO": **p<0.001**
*   **Title:** "Transparency vs Security: Commercial vs Custom AI (EO 13960)" indicates the data is related to Executive Order 13960 compliance.

### 5. Statistical Insights
*   **Significance:** All three comparisons show statistically significant differences between Commercial and Custom systems (assuming a standard significance threshold of $\alpha = 0.05$), as all p-values are below 0.05.
*   **Security Disparity:** The difference in Security (ATO) compliance is the most statistically significant (**p<0.001**), suggesting a very strong confidence that Custom (Gov) systems perform better in this metric than Commercial systems.
*   **Transparency Struggle:** While there is a statistically significant difference favoring Custom systems in the Transparency metrics (Code Access and Data Docs), the **practical significance** is that both Commercial and Custom systems are failing almost entirely in these areas, with compliance rates remaining in the single digits.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
