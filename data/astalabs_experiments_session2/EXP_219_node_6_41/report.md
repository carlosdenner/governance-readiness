# Experiment 219: node_6_41

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_41` |
| **ID in Run** | 219 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T11:17:47.345697+00:00 |
| **Runtime** | 274.8s |
| **Parent** | `node_5_24` |
| **Children** | None |
| **Creation Index** | 220 |

---

## Hypothesis

> The Legacy Gap: Systems in the 'Operations and Maintenance' stage are
statistically less likely to have a completed 'Impact Assessment' than systems
in the 'Development' or 'Implementation' stages, reflecting that older systems
were deployed before rigorous governance mandates.

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

**Objective:** Identify if legacy operational systems bypass current governance documentation requirements.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` (EO13960 subset).
- 2. Group `16_dev_stage` into 'Operations' vs 'Development/Implementation'.
- 3. Compare the binary rate of `52_impact_assessment` (Yes/No).
- 4. Perform a Chi-Square Test.

### Deliverables
- Comparison chart of Impact Assessment compliance by lifecycle stage; Statistical analysis.

---

## Analysis

The experiment successfully analyzed 1,253 EO13960 records to test the 'Legacy
Gap' hypothesis. The results are statistically significant (p < 0.0001) but
**strongly contradict the directional prediction** of the hypothesis.

1.  **Hypothesis Inversion**: The hypothesis predicted that 'Operations
(Legacy)' systems would have lower compliance than 'Dev/Implementation' systems
due to being deployed before mandates. The data shows the exact opposite.
Systems in the **Operations** stage had an Impact Assessment compliance rate of
**8.93%**, whereas systems in the **Development/Implementation** stage had a
near-zero compliance rate of **0.80%**.

2.  **Statistical Significance**: The Chi-Square test (p=0.0000) confirms that
this disparity is not due to chance. Operational systems are over 11 times more
likely to have a completed impact assessment than those in development.

3.  **Governance Interpretation**: This suggests that Impact Assessments
function effectively as a 'gate' or retrospective audit tool for live systems,
rather than a planning tool used during development. The 'Legacy Gap' does not
exist in the way hypothesized; instead, there is a 'Development Gap' where
governance documentation is almost entirely absent until the system reaches
maturity.

---

## Review

The experiment successfully analyzed 1,253 EO13960 records to test the 'Legacy
Gap' hypothesis. The results are statistically significant (p < 0.0001) but
**strongly contradict the directional prediction** of the hypothesis.

1.  **Hypothesis Inversion**: The hypothesis predicted that 'Operations
(Legacy)' systems would have lower compliance than 'Dev/Implementation' systems.
The data shows the exact opposite. Systems in the **Operations** stage had an
Impact Assessment compliance rate of **8.93%**, whereas systems in the
**Development/Implementation** stage had a near-zero compliance rate of
**0.80%**.

2.  **Statistical Significance**: The Chi-Square test (p=0.0000) confirms that
this disparity is not due to chance. Operational systems are over 11 times more
likely to have a completed impact assessment than those in development.

3.  **Governance Interpretation**: This suggests that Impact Assessments
function effectively as a 'gate' or retrospective audit tool for live systems,
rather than a planning tool used during development. The 'Legacy Gap' does not
exist in the way hypothesized; instead, there is a 'Development Gap' where
governance documentation is almost entirely absent until the system reaches
maturity.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import os

# Define the filename
filename = 'astalabs_discovery_all_data.csv'

# Check if file exists in current directory, if not try parent
if os.path.exists(filename):
    file_path = filename
elif os.path.exists(f'../{filename}'):
    file_path = f'../{filename}'
else:
    print(f"Error: {filename} not found in current or parent directory.")
    # List current dir for debugging purposes in case of failure
    print(f"Current dir content: {os.listdir('.')}")
    sys.exit(1)

print(f"Loading dataset from: {file_path}")

# Load the dataset
try:
    df = pd.read_csv(file_path, low_memory=False)
except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

# Filter for EO13960 data
sub_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO13960 records loaded: {len(sub_df)}")

# ---------------------------------------------------------
# Data Preparation: Lifecycle Stage
# ---------------------------------------------------------
# Inspect unique values to ensure correct mapping
print("\nDistribution of '16_dev_stage':")
stage_counts = sub_df['16_dev_stage'].value_counts(dropna=False)
print(stage_counts)

def map_stage(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower()
    # Operation & Maintenance stages
    if 'operation' in val_str or 'maintenance' in val_str or 'use' in val_str:
        return 'Operations (Legacy)'
    # Development & Implementation stages
    elif 'development' in val_str or 'implementation' in val_str or 'acquisition' in val_str or 'planning' in val_str:
        return 'Dev/Implementation (New)'
    else:
        return None # Exclude 'Retired' or other unclear stages

sub_df['Lifecycle_Group'] = sub_df['16_dev_stage'].apply(map_stage)

# Drop rows where Lifecycle Group is undefined
sub_df = sub_df.dropna(subset=['Lifecycle_Group'])
print(f"\nRecords after filtering for relevant lifecycle stages: {len(sub_df)}")
print(sub_df['Lifecycle_Group'].value_counts())

# ---------------------------------------------------------
# Data Preparation: Impact Assessment
# ---------------------------------------------------------
# Map to Binary: Yes vs Not Yes (No, N/A, NaN)
def map_assessment(val):
    if pd.isna(val):
        return 'No'
    val_str = str(val).strip().lower()
    if val_str == 'yes':
        return 'Yes'
    return 'No'

sub_df['Has_Assessment'] = sub_df['52_impact_assessment'].apply(map_assessment)
print("\nImpact Assessment Distribution:")
print(sub_df['Has_Assessment'].value_counts())

# ---------------------------------------------------------
# Statistical Analysis
# ---------------------------------------------------------
# Create Contingency Table
contingency_table = pd.crosstab(sub_df['Lifecycle_Group'], sub_df['Has_Assessment'])
print("\nContingency Table (Count):")
print(contingency_table)

# Calculate Percentages
contingency_pct = pd.crosstab(sub_df['Lifecycle_Group'], sub_df['Has_Assessment'], normalize='index') * 100
print("\nContingency Table (Percentage):")
print(contingency_pct)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print("\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")

# Interpretation
alpha = 0.05
if p < alpha:
    print("\nResult: Statistically Significant Association found (Reject H0).")
else:
    print("\nResult: No Statistically Significant Association found (Fail to reject H0).")

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
# Plotting the percentage of 'Yes' for Impact Assessment by Group
yes_rates = contingency_pct['Yes'] if 'Yes' in contingency_pct.columns else pd.Series([0,0], index=contingency_pct.index)

plt.figure(figsize=(10, 6))
colors = ['#d62728', '#1f77b4'] # Red for Dev, Blue for Ops (or vice versa depending on sort)
ax = yes_rates.plot(kind='bar', color=colors, alpha=0.8)

plt.title('Impact Assessment Compliance by Lifecycle Stage')
plt.ylabel('Percentage with Completed Impact Assessment (%)')
plt.xlabel('Lifecycle Stage')
plt.ylim(0, max(yes_rates.max() * 1.2, 10)) # Add some headroom
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for i, v in enumerate(yes_rates):
    ax.text(i, v + 0.2, f"{v:.1f}%", ha='center', fontweight='bold')

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: astalabs_discovery_all_data.csv
EO13960 records loaded: 1757

Distribution of '16_dev_stage':
16_dev_stage
Operation and Maintenance         627
Acquisition and/or Development    351
Initiated                         329
Implementation and Assessment     275
Retired                           133
Planned                            20
In production                      14
In mission                          4
NaN                                 4
Name: count, dtype: int64

Records after filtering for relevant lifecycle stages: 1253
Lifecycle_Group
Operations (Legacy)         627
Dev/Implementation (New)    626
Name: count, dtype: int64

Impact Assessment Distribution:
Has_Assessment
No     1192
Yes      61
Name: count, dtype: int64

Contingency Table (Count):
Has_Assessment             No  Yes
Lifecycle_Group                   
Dev/Implementation (New)  621    5
Operations (Legacy)       571   56

Contingency Table (Percentage):
Has_Assessment                   No       Yes
Lifecycle_Group                              
Dev/Implementation (New)  99.201278  0.798722
Operations (Legacy)       91.068581  8.931419

Chi-Square Test Results:
Chi2 Statistic: 42.9971
p-value: 0.0000

Result: Statistically Significant Association found (Reject H0).


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart (Column Chart).
*   **Purpose:** The plot compares the percentage of completed impact assessments across two distinct stages of a project's lifecycle: "Dev/Implementation (New)" and "Operations (Legacy)." It is designed to highlight the disparity in compliance between new developments and legacy operations.

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Title:** "Lifecycle Stage"
    *   **Categories:** Two categorical variables are presented: "Dev/Implementation (New)" on the left and "Operations (Legacy)" on the right.
*   **Y-Axis (Vertical):**
    *   **Title/Label:** "Percentage with Completed Impact Assessment (%)"
    *   **Units:** Percentage (%).
    *   **Range:** The axis is marked from 0 to 10 with intervals of 2 (0, 2, 4, 6, 8, 10). The visible range extends slightly above 10, accommodating the data comfortably.

### 3. Data Trends
*   **Tallest Bar:** The "Operations (Legacy)" category (blue bar) is the tallest, reaching a value of 8.9%.
*   **Shortest Bar:** The "Dev/Implementation (New)" category (red bar) is the shortest, reaching a value of only 0.8%.
*   **Pattern:** There is a substantial disparity between the two categories. The compliance rate for legacy operations is significantly higher than that for new development/implementation phases. The visual difference indicates over an 11-fold difference in magnitude between the two groups.

### 4. Annotations and Legends
*   **Data Labels:** Specific numerical percentages are annotated directly above each bar ("0.8%" and "8.9%"). These annotations provide precise values, removing the need to estimate based on the grid lines.
*   **Grid Lines:** Horizontal dashed grid lines are present at intervals of 2% to assist with visual alignment and scale reading.
*   **Color Coding:**
    *   **Red:** Used for "Dev/Implementation (New)," potentially highlighting the critically low nature of this statistic.
    *   **Blue:** Used for "Operations (Legacy)."

### 5. Statistical Insights
*   **Significant Compliance Gap:** The data reveals a massive gap in compliance processes depending on the lifecycle stage. Operations (Legacy) projects are over **11 times (11.1x)** more likely to have a completed impact assessment than projects in the Dev/Implementation (New) stage.
*   **Overall Low Compliance:** Despite "Operations" performing better relatively, the absolute values are objectively low. Even the highest performing category has not reached a 10% compliance rate (8.9%). This suggests a systemic issue where impact assessments are not being completed for the vast majority of projects, regardless of their lifecycle stage.
*   **Process Bottleneck:** The extremely low rate (0.8%) for new implementations suggests a possible process failure, lack of enforcement, or lack of resources dedicated to impact assessments during the early stages of development.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
