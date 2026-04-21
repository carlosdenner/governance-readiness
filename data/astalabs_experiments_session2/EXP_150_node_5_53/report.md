# Experiment 150: node_5_53

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_53` |
| **ID in Run** | 150 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:05:36.450081+00:00 |
| **Runtime** | 387.4s |
| **Parent** | `node_4_32` |
| **Children** | None |
| **Creation Index** | 151 |

---

## Hypothesis

> The 'Agile-Monitoring' Advantage: Federal AI systems developed using 'Agile' or
'DevOps' methodologies are significantly more likely to report 'Post-Deployment
Monitoring' than those using 'Waterfall' methods.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.7500 (Likely True) |
| **Surprise** | +0.0000 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 29.0 |
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
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Assess if modern development methodologies correlate with better continuous monitoring.

### Steps
- 1. Filter 'eo13960_scored'.
- 2. Parse '22_dev_method' to categorize entries into 'Agile' (Agile, DevOps, Iterative) and 'Waterfall' (Waterfall, SDLC, Traditional).
- 3. Compare the binary rate of '56_monitor_postdeploy' (Yes) between the two groups using a Chi-square test.

### Deliverables
- Monitoring rates for Agile vs. Waterfall systems; Statistical significance.

---

## Analysis

The experiment successfully pivoted to investigate the 'Contractor Oversight
Gap' by re-interpreting the '22_dev_method' column as a sourcing variable rather
than a development methodology.

**Findings:**
1.  **Significant Disparity:** Systems developed **In-House** demonstrated a
significantly higher rate of documented post-deployment monitoring (85.7%, n=63)
compared to systems developed by **Contractors** (60.4%, n=53).
2.  **Statistical Validation:** The Chi-square test yielded a p-value of
**0.0038**, confirming that this difference is statistically significant.
3.  **Data Limitations:** While the result is significant, the analysis was
constrained by sparse data; over 90% of the inventory lacked entries for the
monitoring field, reducing the effective sample size to 116 records out of
1,757.

**Conclusion:** The results suggest that federal agencies exercise more rigorous
ongoing governance over internally developed AI tools compared to outsourced
solutions, highlighting a potential risk area in vendor management.

---

## Review

The experiment successfully adapted to the data discovery that '22_dev_method'
contained sourcing information rather than development methodologies. The
pivoted analysis on the 'Contractor Oversight Gap' was executed correctly.

**Hypothesis (Pivoted):** Systems developed In-House are more likely to have
documented 'Post-Deployment Monitoring' than those developed by Contractors.

**Findings:**
1. **Significant Disparity:** In-House systems showed a significantly higher
monitoring rate (**85.7%**, n=63) compared to Contractor-developed systems
(**60.4%**, n=53).
2. **Statistical Significance:** The Chi-square test resulted in a p-value of
**0.0038**, confirming the difference is statistically significant.
3. **Implication:** The results highlight a potential governance gap in the
federal AI supply chain, where outsourced systems are subject to less rigorous
ongoing monitoring than internal tools. Note that the analysis was limited to
the ~6.6% of the inventory (116/1757) that contained valid data for both fields.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Total EO 13960 records: {len(eo_data)}")

# --- PIVOT: 22_dev_method contains Sourcing info, not Agile/Waterfall ---
# New Objective: Compare In-House vs. Contractor Sourcing on Monitoring Rates

# 1. Categorize Sourcing (formerly '22_dev_method')
sourcing_col = '22_dev_method'

def categorize_sourcing(val):
    text = str(val).lower().strip()
    if 'in-house' in text and 'contracting' not in text:
        return 'In-House'
    elif 'contracting' in text and 'in-house' not in text:
        return 'Contractor'
    elif 'both' in text or ('in-house' in text and 'contracting' in text):
        return 'Hybrid'
    else:
        return 'Unknown'

eo_data['sourcing_category'] = eo_data[sourcing_col].apply(categorize_sourcing)
print("\nSourcing Category Distribution:")
print(eo_data['sourcing_category'].value_counts())

# 2. Categorize Monitoring (56_monitor_postdeploy)
monitor_col = '56_monitor_postdeploy'

def categorize_monitoring(val):
    if pd.isna(val):
        return 'Unknown'
    text = str(val).lower().strip()
    
    # Positive indicators (Active monitoring)
    if any(x in text for x in ['intermittent', 'automated', 'established process', 'regularly scheduled']):
        return 'Monitored'
    # Negative indicators (No monitoring)
    elif 'no monitoring' in text:
        return 'Not Monitored'
    else:
        return 'Unknown'

eo_data['monitoring_status'] = eo_data[monitor_col].apply(categorize_monitoring)
print("\nMonitoring Status Distribution:")
print(eo_data['monitoring_status'].value_counts())

# 3. Create Analysis Set (Filter out Unknowns)
analysis_set = eo_data[
    (eo_data['sourcing_category'].isin(['In-House', 'Contractor'])) &
    (eo_data['monitoring_status'].isin(['Monitored', 'Not Monitored']))
].copy()

print(f"\nRecords with valid Sourcing AND Monitoring data: {len(analysis_set)}")

# 4. Statistical Analysis
if len(analysis_set) > 0:
    # Convert to binary for mean calculation
    analysis_set['is_monitored_bool'] = (analysis_set['monitoring_status'] == 'Monitored').astype(int)
    
    # Group stats
    in_house = analysis_set[analysis_set['sourcing_category'] == 'In-House']
    contractor = analysis_set[analysis_set['sourcing_category'] == 'Contractor']
    
    rate_in_house = in_house['is_monitored_bool'].mean()
    rate_contractor = contractor['is_monitored_bool'].mean()
    
    print(f"\nIn-House Monitoring Rate: {rate_in_house:.2%} (n={len(in_house)})")
    print(f"Contractor Monitoring Rate: {rate_contractor:.2%} (n={len(contractor)})")
    
    # Contingency Table
    contingency = pd.crosstab(analysis_set['sourcing_category'], analysis_set['monitoring_status'])
    print("\nContingency Table:")
    print(contingency)
    
    # Chi-Square Test
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test: p-value = {p:.5f}")
    
    if p < 0.05:
        print("Result: Statistically Significant Difference")
    else:
        print("Result: No Statistically Significant Difference")
        
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.bar(['In-House', 'Contractor'], [rate_in_house, rate_contractor], color=['#4CAF50', '#FF9800'])
    plt.title('AI Monitoring Rates by Sourcing Method')
    plt.ylabel('Proportion Monitored')
    plt.ylim(0, 1.1)
    
    for i, v in enumerate([rate_in_house, rate_contractor]):
        plt.text(i, v + 0.02, f"{v:.1%}", ha='center')
        
    plt.show()
else:
    print("\nInsufficient overlapping data to perform statistical test.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total EO 13960 records: 1757

Sourcing Category Distribution:
sourcing_category
Unknown       650
Contractor    481
In-House      439
Hybrid        187
Name: count, dtype: int64

Monitoring Status Distribution:
monitoring_status
Unknown          1611
Monitored         107
Not Monitored      39
Name: count, dtype: int64

Records with valid Sourcing AND Monitoring data: 116

In-House Monitoring Rate: 85.71% (n=63)
Contractor Monitoring Rate: 60.38% (n=53)

Contingency Table:
monitoring_status  Monitored  Not Monitored
sourcing_category                          
Contractor                32             21
In-House                  54              9

Chi-Square Test: p-value = 0.00383
Result: Statistically Significant Difference


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** This plot is designed to compare a quantitative variable ("Proportion Monitored") across two distinct categorical groups ("In-House" and "Contractor"). It facilitates a direct comparison of AI monitoring rates based on the sourcing method used.

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Labels:** The axis represents categorical data with two specific groups: "In-House" and "Contractor".
    *   **Title:** While there is no explicit label (e.g., "Sourcing Type"), the category names clearly define the variable.
*   **Y-Axis (Vertical):**
    *   **Title:** "Proportion Monitored".
    *   **Range:** The axis ticks range from **0.0 to 1.0**, representing a standard proportion scale (equivalent to 0% to 100%). The plot view extends slightly to 1.1 to accommodate the bar annotations.
    *   **Units:** Decimal proportion (though the annotations convert this to percentages).

### 3. Data Trends
*   **Tallest Bar:** The **In-House** category (represented by the green bar) is the highest, indicating it has the leading rate of monitoring.
*   **Shortest Bar:** The **Contractor** category (represented by the orange bar) is significantly shorter.
*   **Pattern:** There is a clear descending trend from internal sourcing to external contracting. The visual disparity suggests that AI projects sourced internally are subject to significantly higher oversight rates than those outsourced to contractors.

### 4. Annotations and Legends
*   **Title:** The chart is titled "AI Monitoring Rates by Sourcing Method".
*   **Bar Annotations:** Specific percentage values are placed directly above each bar to provide exact data points:
    *   **In-House:** Annotated with **85.7%**.
    *   **Contractor:** Annotated with **60.4%**.
*   **Colors:** The plot uses distinct colors to differentiate the categories: Green for "In-House" and Orange for "Contractor". There is no separate legend, as the x-axis labels suffice.

### 5. Statistical Insights
*   **Significant Discrepancy:** There is a substantial gap of **25.3 percentage points** between the monitoring rates of in-house AI and contractor-sourced AI.
*   **High Internal Oversight:** In-house teams have a very high monitoring compliance rate (over 85%), suggesting robust internal governance protocols.
*   **External Governance Gap:** The monitoring rate drops to roughly 60% for contractors. This suggests a potential risk area where external vendors or outsourced solutions may not be subject to the same rigorous ongoing evaluation or observability standards as internal projects.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
