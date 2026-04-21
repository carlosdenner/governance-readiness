# Experiment 102: node_4_44

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_44` |
| **ID in Run** | 102 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:54:02.500065+00:00 |
| **Runtime** | 270.9s |
| **Parent** | `node_3_2` |
| **Children** | `node_5_49`, `node_5_63` |
| **Creation Index** | 103 |

---

## Hypothesis

> Vendor Opacity: Federally deployed 'Commercial' AI systems are significantly
less likely to have reported 'Post-Deployment Monitoring' compared to 'Custom-
Developed' systems, indicating a governance gap in vendor-procured solutions.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7903 (Likely True) |
| **Posterior** | 0.2692 (Likely False) |
| **Surprise** | -0.6253 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

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
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Compare post-deployment monitoring rates between Commercial and Custom AI systems in the federal inventory.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `source_table == 'eo13960_scored'`.
- 2. Clean `10_commercial_ai` to categorize entries into 'Commercial' (COTS) vs 'Custom/Federal'.
- 3. Clean `56_monitor_postdeploy` into binary (Yes/No).
- 4. Create a contingency table.
- 5. Perform a Chi-Square test of independence.

### Deliverables
- Contingency table of Source vs Monitoring; Chi-Square test results; Bar chart comparing monitoring rates.

---

## Analysis

The experiment successfully analyzed 1,710 records from the EO13960 dataset to
evaluate the 'Vendor Opacity' hypothesis. The results **reject** the hypothesis
that Commercial AI systems are less likely to report post-deployment monitoring
than Custom/Federal systems.

1.  **Statistical Significance**: The Chi-Square test yielded a p-value of
0.0083, indicating a statistically significant difference between the two
groups.
2.  **Directionality**: Contrary to the hypothesis, 'Commercial' systems showed
a **higher** rate of reported monitoring (11.9%) compared to 'Custom/Federal'
systems (7.4%).
3.  **Governance Insight**: Rather than 'Vendor Opacity', the data suggests a
'Federal Maturity Gap', where custom government-developed systems are
significantly less likely to have established or reported post-deployment
monitoring protocols compared to their commercial counterparts.

---

## Review

The experiment successfully verified the 'Vendor Opacity' hypothesis using the
EO13960 dataset. The implementation correctly adapted to the specific data
values found during the debugging phase, categorizing 'None of the above' as
'Custom/Federal' and specific commercial use-cases as 'Commercial'.

**Findings:**
1.  **Hypothesis Rejection:** The results statistically **reject** the
hypothesis that Commercial AI systems are less likely to report post-deployment
monitoring than Custom/Federal systems.
2.  **Inverse Relationship:** The data reveals a significant inverse trend (Chi-
Square p=0.0083). Commercial systems reported a higher monitoring rate (11.9%)
compared to Custom/Federal systems (7.4%).
3.  **Implication:** Instead of 'Vendor Opacity', the findings point to a
potential 'Federal Maturity Gap', where custom government-developed tools lack
the standardized post-deployment monitoring protocols present in commercial off-
the-shelf solutions.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Step 1: Clean '10_commercial_ai' (System Source) ---
# Logic: 'None of the above.' implies the system is not one of the standard COTS tools listed, 
# hence categorized as 'Custom/Federal' for this comparison. 
# Specific string values imply 'Commercial' use cases.

def classify_source(val):
    if pd.isna(val):
        return np.nan
    s_val = str(val).strip()
    if s_val == 'None of the above.':
        return 'Custom/Federal'
    else:
        return 'Commercial'

df_eo['system_source'] = df_eo['10_commercial_ai'].apply(classify_source)

# Drop rows where source is unknown (NaN in 10_commercial_ai)
df_analysis = df_eo.dropna(subset=['system_source']).copy()

# --- Step 2: Clean '56_monitor_postdeploy' (Monitoring Status) ---
# Logic: Identify positive assertions of monitoring infrastructure.
# NaNs are treated as 'No/Not Reported' in this context of survey compliance.

def classify_monitoring(val):
    if pd.isna(val):
        return 'No'
    
    s_val = str(val).lower()
    
    # Positive keywords based on unique values analysis
    # "Intermittent and Manually Updated..."
    # "Automated and Regularly Scheduled..."
    # "Established Process..."
    if any(keyword in s_val for keyword in ['intermittent', 'automated', 'established', 'manually updated']):
        return 'Yes'
    
    # Negative keywords: "No monitoring protocols", "not safety impacting"
    # "under development" is treated as No (not currently active)
    return 'No'

df_analysis['has_monitoring'] = df_analysis['56_monitor_postdeploy'].apply(classify_monitoring)

# --- Step 3: Analysis ---
print(f"Records for analysis: {len(df_analysis)}")
print("System Source Distribution:\n", df_analysis['system_source'].value_counts())
print("Monitoring Status Distribution:\n", df_analysis['has_monitoring'].value_counts())

# Contingency Table
contingency_table = pd.crosstab(df_analysis['system_source'], df_analysis['has_monitoring'])
print("\nContingency Table (Source vs Monitoring):")
print(contingency_table)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Calculate Odds Ratio
# OR = (a*d) / (b*c)
# Table structure usually: 
#              No   Yes
# Commercial   a    b
# Custom       c    d
# But let's check the columns of crosstab first

if 'Yes' in contingency_table.columns and 'No' in contingency_table.columns:
    # Calculate percentage of 'Yes' for each group
    monitoring_rates = pd.crosstab(df_analysis['system_source'], df_analysis['has_monitoring'], normalize='index') * 100
    print("\nMonitoring Rates (%):")
    print(monitoring_rates)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    
    # Ensure order is Commercial then Custom for comparison
    sources = ['Commercial', 'Custom/Federal']
    # Handle case if one source is missing
    present_sources = [s for s in sources if s in monitoring_rates.index]
    
    yes_rates = monitoring_rates.loc[present_sources, 'Yes']
    
    bars = plt.bar(present_sources, yes_rates, color=['#ff7f0e', '#1f77b4'])
    
    plt.title('Post-Deployment Monitoring Rates: Commercial vs Custom AI')
    plt.xlabel('System Source')
    plt.ylabel('Reported Monitoring (%)')
    plt.ylim(0, max(yes_rates.max() * 1.2, 10))
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data to generate 'Yes' column for monitoring.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Records for analysis: 1710
System Source Distribution:
 system_source
Custom/Federal    1357
Commercial         353
Name: count, dtype: int64
Monitoring Status Distribution:
 has_monitoring
No     1568
Yes     142
Name: count, dtype: int64

Contingency Table (Source vs Monitoring):
has_monitoring    No  Yes
system_source            
Commercial       311   42
Custom/Federal  1257  100

Chi-Square Statistic: 6.9624
P-value: 8.3239e-03

Monitoring Rates (%):
has_monitoring         No        Yes
system_source                       
Commercial      88.101983  11.898017
Custom/Federal  92.630803   7.369197


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** The plot compares the percentages of reported post-deployment monitoring for two distinct categories of AI system sources: "Commercial" and "Custom/Federal."

### 2. Axes
*   **X-axis:**
    *   **Title:** "System Source"
    *   **Labels:** The axis represents categorical data with two groups: "Commercial" and "Custom/Federal."
*   **Y-axis:**
    *   **Title:** "Reported Monitoring (%)"
    *   **Units:** Percentage (%).
    *   **Range:** The axis scale ranges from **0 to 14**, with major tick marks at intervals of 2.

### 3. Data Trends
*   **Tallest Bar:** The orange bar representing **Commercial** systems is the tallest, indicating a higher rate of reported monitoring.
*   **Shortest Bar:** The blue bar representing **Custom/Federal** systems is the shortest.
*   **Pattern:** There is a clear disparity between the two sources. AI systems sourced commercially are reported to be monitored post-deployment at a higher frequency than those developed via custom or federal channels.

### 4. Annotations and Legends
*   **Title:** "Post-Deployment Monitoring Rates: Commercial vs Custom AI" provides the main context for the visualization.
*   **Value Labels:** Exact percentage values are annotated above each bar for precision:
    *   Commercial: **11.9%**
    *   Custom/Federal: **7.4%**
*   **Gridlines:** Horizontal dashed gridlines are included to assist in visual estimation of bar heights relative to the Y-axis.
*   **Colors:** The bars are color-coded (Orange for Commercial, Blue for Custom/Federal) to visually distinguish the categories.

### 5. Statistical Insights
*   **Absolute Difference:** There is a **4.5 percentage point gap** between the two categories ($11.9\% - 7.4\% = 4.5\%$).
*   **Relative Comparison:** Commercial AI systems have a reported monitoring rate approximately **1.6 times higher** than Custom/Federal systems.
*   **Implication:** The data suggests that commercial vendors may have more established protocols or requirements for post-deployment monitoring compared to custom or federally developed solutions. Alternatively, this could indicate a gap in reporting standards where federal/custom projects are less likely to disclose monitoring activities even if they occur.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
