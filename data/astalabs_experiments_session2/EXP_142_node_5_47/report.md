# Experiment 142: node_5_47

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_47` |
| **ID in Run** | 142 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:44:35.205684+00:00 |
| **Runtime** | 238.0s |
| **Parent** | `node_4_8` |
| **Children** | `node_6_77` |
| **Creation Index** | 143 |

---

## Hypothesis

> Outsourced Accountability Gap: AI systems developed primarily by contractors are
significantly less likely to have a documented 'Appeal Process' compared to
systems developed in-house, due to the distance between developers and end-
users.

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

**Objective:** Investigate if development provenance (Contractor vs. In-house) influences the presence of accountability mechanisms.

### Steps
- 1. Filter for `eo13960_scored`.
- 2. Map `22_dev_method` to 'Contractor' (Contracting) vs. 'In-house' (Government/In-house). Exclude hybrid/unknowns if necessary.
- 3. Map `65_appeal_process` to Binary (Yes vs. No).
- 4. Perform a Chi-square test of independence.
- 5. Compare the proportion of systems with appeal processes in both groups.

### Deliverables
- Statistical test results and a comparative bar chart of Appeal Process presence by Development Method.

---

## Analysis

The experiment was successfully executed with the corrected filtering logic,
resolving the previous error where 'In-house' records were dropped.

**Data Processing:**
The script successfully categorized 920 systems from the EO 13960 dataset:
- **Contractor:** 481 systems
- **In-house:** 439 systems

**Results:**
- **In-house Development:** 8.0% (35/439) of systems have a documented appeal
process.
- **Contractor Development:** 4.6% (22/481) of systems have a documented appeal
process.

**Statistical Significance:**
The Chi-square test yielded a statistic of 3.9962 and a **p-value of 0.0456**.

**Conclusion:**
The hypothesis is **supported**. The results indicate a statistically
significant difference (p < 0.05), confirming that AI systems developed by
contractors are significantly less likely to have a documented appeal process
compared to those developed in-house. While compliance is low across both
groups, in-house teams are nearly twice as likely to implement this specific
accountability mechanism, supporting the 'Outsourced Accountability Gap' theory.

---

## Review

The experiment was successfully executed with the corrected filtering logic,
resolving the previous error where 'In-house' records were dropped.

**Data Processing:**
The script successfully categorized 920 systems from the EO 13960 dataset:
- **Contractor:** 481 systems
- **In-house:** 439 systems

**Results:**
- **In-house Development:** 8.0% (35/439) of systems have a documented appeal
process.
- **Contractor Development:** 4.6% (22/481) of systems have a documented appeal
process.

**Statistical Significance:**
The Chi-square test yielded a statistic of 3.9962 and a **p-value of 0.0456**.

**Conclusion:**
The hypothesis is **supported**. The results indicate a statistically
significant difference (p < 0.05), confirming that AI systems developed by
contractors are significantly less likely to have a documented appeal process
compared to those developed in-house. While compliance is low across both
groups, in-house teams are nearly twice as likely to implement this specific
accountability mechanism, supporting the 'Outsourced Accountability Gap' theory.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    print("File not found at '../astalabs_discovery_all_data.csv', trying local directory...")
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# Filter for EO 13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print("Dataset filtered. EO 13960 records:", len(eo_data))

# Inspect columns of interest
dev_col = '22_dev_method'
appeal_col = '65_appeal_process'

# Data Cleaning & Mapping

# Corrected Mapping Function based on previous output unique values
def map_dev_method(val):
    if pd.isna(val):
        return None
    val = str(val).lower().strip()
    
    # Check for hybrid first to exclude it
    if 'both' in val:
        return None
        
    if 'contract' in val:
        return 'Contractor'
    elif 'in-house' in val or 'government' in val:
        return 'In-house'
    
    return None

eo_data['dev_category'] = eo_data[dev_col].apply(map_dev_method)

# Map Appeal Process
def map_appeal(val):
    if pd.isna(val):
        return 'No'
    val = str(val).lower().strip()
    if val == 'yes':
        return 'Yes'
    return 'No'

eo_data['has_appeal'] = eo_data[appeal_col].apply(map_appeal)

# Filter for valid development categories
analysis_df = eo_data.dropna(subset=['dev_category'])

print(f"\nRecords after filtering for valid development method (Contractor vs In-house): {len(analysis_df)}")
print("Distribution by Category:")
print(analysis_df['dev_category'].value_counts())

# Contingency Table
contingency_table = pd.crosstab(analysis_df['dev_category'], analysis_df['has_appeal'])
print("\nContingency Table (Appeal Process vs Dev Method):")
print(contingency_table)

# Chi-square Test
if contingency_table.shape[0] == 2:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
else:
    print("\nError: Contingency table does not have 2 rows. Cannot perform Chi-square test properly.")

# Calculate percentages
props = pd.crosstab(analysis_df['dev_category'], analysis_df['has_appeal'], normalize='index') * 100
print("\nProportions (%):")
print(props)

# Visualization
if not props.empty:
    try:
        ax = props.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#d62728', '#2ca02c'])
        plt.title('Presence of Appeal Process by Development Method')
        plt.xlabel('Development Method')
        plt.ylabel('Percentage')
        plt.legend(title='Has Appeal Process', loc='upper right', labels=['No', 'Yes'])
        plt.xticks(rotation=0)
        
        # Add value labels
        for c in ax.containers:
            ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white')
            
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting error: {e}")
else:
    print("Insufficient data for plotting.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: File not found at '../astalabs_discovery_all_data.csv', trying local directory...
Dataset filtered. EO 13960 records: 1757

Records after filtering for valid development method (Contractor vs In-house): 920
Distribution by Category:
dev_category
Contractor    481
In-house      439
Name: count, dtype: int64

Contingency Table (Appeal Process vs Dev Method):
has_appeal     No  Yes
dev_category          
Contractor    459   22
In-house      404   35

Chi-square Statistic: 3.9962
P-value: 4.5604e-02

Proportions (%):
has_appeal           No       Yes
dev_category                     
Contractor    95.426195  4.573805
In-house      92.027335  7.972665


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** The plot compares the distribution of a binary categorical variable ("Has Appeal Process") across two different groups ("Development Method"). It visualizes the relative proportion of systems that do or do not have an appeal process within each development category.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Development Method".
    *   **Categories:** Two discrete categories are displayed: "Contractor" and "In-house".
*   **Y-Axis:**
    *   **Title:** "Percentage".
    *   **Range:** 0 to 100 (representing 0% to 100%).
    *   **Ticks:** Increments of 20 (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Dominant Trend (No Appeal Process):** Across both development methods, the vast majority of cases do **not** have an appeal process. This is represented by the large red segments taking up most of the bar height.
    *   For **Contractor** developed projects, **95.4%** do not have an appeal process.
    *   For **In-house** developed projects, **92.0%** do not have an appeal process.
*   **Minor Trend (Yes Appeal Process):** A very small minority of projects include an appeal process, represented by the green segments at the top of the bars.
    *   **Contractor:** 4.6%
    *   **In-house:** 8.0%

### 4. Annotations and Legends
*   **Title:** "Presence of Appeal Process by Development Method" (Centered at the top).
*   **Legend:** Located in the top right corner with the title "Has Appeal Process".
    *   **Red:** Corresponds to "No".
    *   **Green:** Corresponds to "Yes".
*   **Data Labels:** White percentage text is overlaid directly onto the bars to provide precise values (e.g., "95.4%", "4.6%", "92.0%", "8.0%").

### 5. Statistical Insights
*   **Overwhelming Absence of Appeals:** There is a significant lack of appeal processes in the systems analyzed. Regardless of who developed the system, over 90% lack this feature. This suggests that implementing an appeal mechanism is generally not a standard requirement or priority in this domain.
*   **Comparison of Methods:** While both numbers are low, systems developed **In-house** are nearly twice as likely to have an appeal process (8.0%) compared to those developed by **Contractors** (4.6%). This could indicate that internal teams may have slightly more awareness of or incentive to include user recourse mechanisms, or that contractor specifications rarely include this requirement.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
