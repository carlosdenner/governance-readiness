# Experiment 82: node_5_10

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_10` |
| **ID in Run** | 82 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T04:56:30.664575+00:00 |
| **Runtime** | 330.8s |
| **Parent** | `node_4_29` |
| **Children** | `node_6_62` |
| **Creation Index** | 83 |

---

## Hypothesis

> Ownership-Monitoring Correlation: Systems developed 'In-house' are significantly
more likely to have 'Continuous Monitoring' protocols than systems developed by
'Contractors', reflecting the challenge of long-term maintenance in outsourced
projects.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.9121 (Definitely True) |
| **Surprise** | +0.2042 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
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
| Definitely True | 60.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Assess the impact of development ownership on long-term AI lifecycle maintenance.

### Steps
- 1. Load EO13960 data.
- 2. Use `22_dev_method` to classify 'In-house' vs 'Contracting resources'.
- 3. Analyze `56_monitor_postdeploy` (convert to binary).
- 4. Compare the percentage of monitored systems in both groups.
- 5. Run a Z-test for proportions or Chi-square test.

### Deliverables
- 1. Monitoring rates for In-house vs Contractor systems.
- 2. Statistical test results.
- 3. Visualization of the monitoring gap.

---

## Analysis

The experiment successfully tested the 'Ownership-Monitoring Correlation'
hypothesis by correcting the data mapping for development methods and monitoring
status.

**Methodology:**
Using the EO13960 dataset (n=920 qualifying systems), the analysis compared 'In-
house' development (n=439) against 'Contractor' development (n=481). Monitoring
status was derived from free-text descriptions, classifying 'Automated',
'Intermittent', or 'Established' processes as positive indicators.

**Findings:**
1.  **Monitoring Gap (Confirmed):** The data supports the hypothesis. Systems
developed In-house demonstrated a monitoring rate of **12.3%** (54/439), nearly
double the **6.7%** (32/481) rate observed for Contractor-developed systems.
2.  **Statistical Significance:** A Chi-Square test confirmed this difference is
statistically significant (Chi2=7.99, p=0.0047, p<0.05).
3.  **Overall Maturity:** Despite the relative advantage of in-house ownership,
the absolute rates of continuous monitoring are alarmingly low across both
groups (<13%), suggesting a systemic lack of MLOps maturity in federal AI
deployments.

**Conclusion:**
The hypothesis is **supported**. In-house ownership is significantly correlated
with better post-deployment monitoring, likely due to the incentives for long-
term maintenance that are often absent or descoped in limited-term external
contracts.

---

## Review

The experiment successfully validated the 'Ownership-Monitoring Correlation'
hypothesis using the EO13960 dataset. After an initial debugging phase to
identify correct string mappings, the pipeline correctly classified 920 systems
into 'In-house' (n=439) and 'Contractor' (n=481) groups and parsed the free-text
'monitor_postdeploy' field to identify active monitoring protocols.

Findings:
1. **Hypothesis Supported**: The analysis revealed a statistically significant
difference in monitoring rates (Chi-square p=0.0047). Systems developed in-house
were nearly twice as likely (12.3%) to have continuous monitoring compared to
those developed by contractors (6.7%).
2. **Low Maturity Baseline**: Despite the relative advantage of in-house
ownership, the absolute rates of monitoring are remarkably low (<13% for both
groups), highlighting a critical gap in MLOps and long-term maintenance across
the federal AI inventory.

The implementation was faithful to the plan, and the statistical methods (Chi-
square) were appropriate for the categorical data.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# Filter for EO13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO13960 Data Shape: {eo_data.shape}")

# --- 1. Define Groups (Development Method) ---
# Specific strings found in debug step
IN_HOUSE_STR = "Developed in-house."
CONTRACTOR_STR = "Developed with contracting resources."

def classify_dev(val):
    if pd.isna(val):
        return None
    if val == IN_HOUSE_STR:
        return "In-house"
    if val == CONTRACTOR_STR:
        return "Contractor"
    return None

eo_data['dev_group'] = eo_data['22_dev_method'].apply(classify_dev)

# Filter only for the two groups of interest
analysis_df = eo_data[eo_data['dev_group'].notna()].copy()

# --- 2. Define Outcome (Continuous Monitoring) ---
# Mapping based on debugged values
# Positive indicators: 'Intermittent', 'Automated', 'Established'
# Negative/Null indicators: 'No monitoring', 'AI is not safety', NaN

def classify_monitor(val):
    if pd.isna(val):
        return 0 # Treating missing as 'No Monitoring'
    val_str = str(val)
    if "Intermittent" in val_str or "Automated" in val_str or "Established" in val_str:
        return 1
    return 0

analysis_df['is_monitored'] = analysis_df['56_monitor_postdeploy'].apply(classify_monitor)

# --- 3. Analysis ---
stats = analysis_df.groupby('dev_group')['is_monitored'].agg(['count', 'sum', 'mean'])
stats.columns = ['Total Systems', 'Monitored Systems', 'Monitoring Rate']

print("\n--- Monitoring Compliance by Development Ownership ---")
print(stats)

# Contingency Table
contingency = pd.crosstab(analysis_df['dev_group'], analysis_df['is_monitored'])
print("\n--- Contingency Table (0=No, 1=Yes) ---")
print(contingency)

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

if p < 0.05:
    print("Result: Statistically significant difference detected.")
else:
    print("Result: No statistically significant difference detected.")

# --- 4. Visualization ---
plt.figure(figsize=(8, 6))

# Prepare data for plotting
groups = stats.index.tolist()
rates = stats['Monitoring Rate'].tolist()
colors = ['#e74c3c', '#3498db'] # Red for Contractor, Blue for In-house (usually)
if groups[0] == 'In-house':
    colors = ['#3498db', '#e74c3c']

bars = plt.bar(groups, rates, color=colors, alpha=0.8, edgecolor='black')

plt.title('AI System Continuous Monitoring Rates: In-house vs. Contractor')
plt.ylabel('Proportion of Systems with Monitoring')
plt.xlabel('Development Ownership')
plt.ylim(0, max(rates) * 1.2 if max(rates) > 0 else 0.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Annotate bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + (max(rates)*0.02 if max(rates)>0 else 0.005),
             f'{height:.1%}',
             ha='center', va='bottom', fontweight='bold')

plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: EO13960 Data Shape: (1757, 196)

--- Monitoring Compliance by Development Ownership ---
            Total Systems  Monitored Systems  Monitoring Rate
dev_group                                                    
Contractor            481                 32         0.066528
In-house              439                 54         0.123007

--- Contingency Table (0=No, 1=Yes) ---
is_monitored    0   1
dev_group            
Contractor    449  32
In-house      385  54

Chi-Square Statistic: 7.9862
P-value: 4.7136e-03
Result: Statistically significant difference detected.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart (Column Chart).
*   **Purpose:** This chart compares the prevalence of continuous monitoring in AI systems across two distinct categories of development ownership: "Contractor" and "In-house."

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Title:** "Development Ownership"
    *   **Labels:** Categorical variables representing the two groups: "Contractor" and "In-house."
*   **Y-Axis (Vertical):**
    *   **Title:** "Proportion of Systems with Monitoring"
    *   **Range:** The axis is scaled from 0.00 to 0.14 (with grid lines indicating it could extend slightly to 0.15).
    *   **Units:** The axis ticks are formatted as decimals (proportions), ranging from 0.00 to 0.14 in increments of 0.02.

### 3. Data Trends
*   **Tallest Bar:** The "In-house" category (blue bar) represents the higher value.
*   **Shortest Bar:** The "Contractor" category (salmon/red bar) represents the lower value.
*   **Pattern:** There is a distinct disparity between the two groups. The proportion of in-house developed systems that utilize continuous monitoring is significantly higher—nearly double—than that of systems developed by contractors.

### 4. Annotations and Legends
*   **Bar Labels:** Specific percentage values are annotated directly above each bar for clarity:
    *   Contractor: **6.7%**
    *   In-house: **12.3%**
*   **Color Coding:** The bars are distinctively colored (Salmon for Contractor, Blue for In-house) to visually separate the categories, though no separate legend box is provided or necessary given the clear x-axis labels.
*   **Grid Lines:** Horizontal dashed grid lines are included to assist in estimating values relative to the Y-axis.

### 5. Statistical Insights
*   **Significant Gap in Standards:** Systems developed in-house are approximately **1.8 times** more likely (12.3% vs. 6.7%) to have continuous monitoring implemented compared to those developed by contractors.
*   **Low Overall Adoption:** Despite the difference between the groups, the absolute adoption rates for continuous monitoring are low across the board. Even in the higher-performing "In-house" group, only about 1 in 8 systems (12.3%) are continuously monitored.
*   **Implication:** This data suggests a potential governance gap in outsourced AI development. It implies that external contracts may lack stringent requirements for post-deployment monitoring or that in-house teams have better infrastructure and incentives for long-term system maintenance and observability.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
