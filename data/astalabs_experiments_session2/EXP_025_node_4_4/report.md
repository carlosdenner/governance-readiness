# Experiment 25: node_4_4

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_4` |
| **ID in Run** | 25 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T02:15:04.133203+00:00 |
| **Runtime** | 250.2s |
| **Parent** | `node_3_5` |
| **Children** | `node_5_0`, `node_5_5`, `node_5_90` |
| **Creation Index** | 26 |

---

## Hypothesis

> Sector-Specific Justice: 'Enforcement' agencies (e.g., DHS, DOJ) have
significantly lower 'Appeal Process' availability for their AI systems compared
to 'Benefits' agencies (e.g., HHS, VA, Education).

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7823 (Likely True) |
| **Posterior** | 0.2665 (Likely False) |
| **Surprise** | -0.6189 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 7.0 |
| Maybe True | 22.0 |
| Uncertain | 0.0 |
| Maybe False | 1.0 |
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

**Objective:** Compare procedural justice mechanisms (appeals) across agency sectors.

### Steps
- 1. Load `eo13960_scored`.
- 2. Map `3_agency` to sectors: 'Enforcement' (DHS, DOJ, State, Defense) vs. 'Benefits' (HHS, VA, Education, HUD, SSA). Filter out others.
- 3. Parse `65_appeal_process` into binary (1=Process exists, 0=No process/Silent).
- 4. Compare proportions of Appeal Availability between Enforcement and Benefits sectors using a Z-test.

### Deliverables
- Sector-wise compliance table, Z-test results, and interpretation of the 'Justice Gap'.

---

## Analysis

The experiment was successfully executed and provided a definitive, albeit
counter-intuitive, answer to the hypothesis.

1. **Hypothesis Rejection**: The hypothesis stated that 'Enforcement' agencies
would have lower appeal process availability than 'Benefits' agencies. The data
**strongly refutes** this. In fact, the relationship is inverted: 'Enforcement'
agencies demonstrated a significantly higher rate of defined appeal processes
(6.4%) compared to 'Benefits' agencies (0.2%).

2. **Statistical Significance**: The Z-test yielded a Z-statistic of -5.53 and a
p-value of < 0.0001, confirming that the difference is statistically significant
and not due to chance.

3. **The 'Justice Gap' Reality**: The analysis reveals a concerning 'Justice
Gap' in the **Benefits sector**, not Enforcement. Out of 529 AI systems in the
Benefits sector (HHS, VA, Education, etc.), only **1** system had a confirmed
appeal process (`Yes`), compared to 15 out of 234 in Enforcement. This suggests
that while procedural justice mechanisms are scarce across the federal
government (overall <4%), they are virtually non-existent in the sector most
directly responsible for distributing social welfare and entitlements.

---

## Review

The experiment was successfully executed and provided a definitive, albeit
counter-intuitive, answer to the hypothesis.

1. **Hypothesis Rejection**: The hypothesis stated that 'Enforcement' agencies
would have lower appeal process availability than 'Benefits' agencies. The data
**strongly refutes** this. In fact, the relationship is inverted: 'Enforcement'
agencies demonstrated a significantly higher rate of defined appeal processes
(6.4%) compared to 'Benefits' agencies (0.2%).

2. **Statistical Significance**: The Z-test yielded a Z-statistic of -5.53 and a
p-value of < 0.0001, confirming that the difference is statistically significant
and not due to chance.

3. **The 'Justice Gap' Reality**: The analysis reveals a concerning 'Justice
Gap' in the **Benefits sector**, not Enforcement. Out of 529 AI systems in the
Benefits sector (HHS, VA, Education, etc.), only **1** system had a confirmed
appeal process (`Yes`), compared to 15 out of 234 in Enforcement. This suggests
that while procedural justice mechanisms are scarce across the federal
government (overall <4%), they are virtually non-existent in the sector most
directly responsible for distributing social welfare and entitlements.

---

## Code

```python
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting experiment: Sector-Specific Justice - Appeal Process Analysis")

# 1. Load Data
# Using the correct path verified in debug step
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored subset
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded EO 13960 subset: {len(eo_data)} rows")

# 2. Map Agencies to Sectors
# Define sector mappings based on standard agency names
enforcement_agencies = [
    'Department of Homeland Security',
    'Department of Justice',
    'Department of State',
    'Department of Defense'
]

benefits_agencies = [
    'Department of Health and Human Services',
    'Department of Veterans Affairs',
    'Department of Education',
    'Department of Housing and Urban Development',
    'Social Security Administration'
]

def classify_sector(agency_name):
    if pd.isna(agency_name):
        return None
    # Normalize for matching
    name = str(agency_name).strip()
    if name in enforcement_agencies:
        return 'Enforcement'
    if name in benefits_agencies:
        return 'Benefits'
    return 'Other'

# Apply classification
eo_data['sector'] = eo_data['3_agency'].apply(classify_sector)

# Filter for only relevant sectors
analysis_df = eo_data[eo_data['sector'].isin(['Enforcement', 'Benefits'])].copy()
print(f"Filtered for target sectors: {len(analysis_df)} rows")
print(analysis_df['sector'].value_counts())

# 3. Parse '65_appeal_process'
# Inspect unique values to ensure robust parsing
unique_vals = analysis_df['65_appeal_process'].dropna().unique()
print(f"\nSample of raw '65_appeal_process' values (first 5): {unique_vals[:5]}")

def parse_appeal(val):
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    # Heuristic: Check for explicit 'yes' or specific phrases indicating existence.
    if val_str.startswith('yes'):
        return 1
    return 0

analysis_df['has_appeal'] = analysis_df['65_appeal_process'].apply(parse_appeal)

# 4. Statistical Analysis
# Group by Sector
summary = analysis_df.groupby('sector')['has_appeal'].agg(['count', 'sum', 'mean'])
summary.columns = ['Total Systems', 'With Appeal Process', 'Proportion']
print("\nSector Analysis Summary:")
print(summary)

# Prepare for Z-test
# Identify groups
sector_names = summary.index.tolist()
if len(sector_names) == 2:
    group1 = sector_names[0]
    group2 = sector_names[1]
    
    count1 = summary.loc[group1, 'With Appeal Process']
    nobs1 = summary.loc[group1, 'Total Systems']
    count2 = summary.loc[group2, 'With Appeal Process']
    nobs2 = summary.loc[group2, 'Total Systems']
    
    print(f"\nComparing: {group1} (n={nobs1}) vs {group2} (n={nobs2})")
    
    counts = np.array([count1, count2])
    nobs = np.array([nobs1, nobs2])
    
    stat, pval = proportions_ztest(counts, nobs)
    print(f"\nZ-Test Results:")
    print(f"Z-statistic: {stat:.4f}")
    print(f"P-value: {pval:.4f}")
    
    alpha = 0.05
    if pval < alpha:
        print("Conclusion: Significant difference in appeal process availability between sectors.")
    else:
        print("Conclusion: No significant difference found.")
else:
    print("Error: Insufficient groups for comparison.")

# 5. Visualization
plt.figure(figsize=(8, 6))
sns.barplot(x=summary.index, y='Proportion', data=summary.reset_index(), hue='sector', palette=['#1f77b4', '#ff7f0e'])
plt.title('Proportion of AI Systems with Defined Appeal Processes by Sector')
plt.ylabel('Proportion (0-1)')
plt.xlabel('Agency Sector')
plt.ylim(0, 1.0)

# Add labels
for i, row in enumerate(summary.itertuples()):
    # itertuples yields (Index, Total, With Appeal, Proportion)
    plt.text(i, row.Proportion + 0.02, f"{row.Proportion:.1%}\n(n={row._1})", 
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment: Sector-Specific Justice - Appeal Process Analysis
Loaded EO 13960 subset: 1757 rows
Filtered for target sectors: 763 rows
sector
Benefits       529
Enforcement    234
Name: count, dtype: int64

Sample of raw '65_appeal_process' values (first 5): <StringArray>
[                                                                          'Yes',
                         'No – it is not operationally practical to offer this.',
 'Agency CAIO has waived this minimum practice and reported such waiver to OMB.']
Length: 3, dtype: str

Sector Analysis Summary:
             Total Systems  With Appeal Process  Proportion
sector                                                     
Benefits               529                    1    0.001890
Enforcement            234                   15    0.064103

Comparing: Benefits (n=529) vs Enforcement (n=234)

Z-Test Results:
Z-statistic: -5.5304
P-value: 0.0000
Conclusion: Significant difference in appeal process availability between sectors.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Bar Plot.
*   **Purpose:** The plot compares categorical data across two distinct groups ("Benefits" vs. "Enforcement") to visualize the prevalence of defined appeal processes within AI systems used in those sectors.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Agency Sector"
    *   **Labels:** Two categories are represented: **Benefits** and **Enforcement**.
*   **Y-Axis:**
    *   **Title:** "Proportion (0-1)"
    *   **Range:** The axis spans from **0.0 to 1.0**.
    *   **Ticks:** The axis is marked at intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
*   **Comparison of Heights:** The bar representing the "Enforcement" sector is significantly taller than the bar for the "Benefits" sector, which is barely visible on the scale.
*   **Magnitude:** Both values are extremely low relative to the maximum possible value of 1.0 (or 100%). The vast majority of the plot area is empty, indicating low proportions for both categories.
*   **Pattern:** There is a clear disparity where the enforcement sector has a notably higher rate of defined appeal processes compared to the benefits sector, though both rates are objectively low.

### 4. Annotations and Legends
*   **Bar Annotations:** Specific values are annotated directly above each bar, providing exact percentages and sample sizes ($n$):
    *   **Benefits:** **0.2%** with a sample size of **(n=529)**.
    *   **Enforcement:** **6.4%** with a sample size of **(n=234)**.
*   **Color Coding:**
    *   **Blue:** Represents the Benefits sector.
    *   **Orange:** Represents the Enforcement sector.

### 5. Statistical Insights
*   **Extremely Low Adoption:** Overall, the data indicates that AI systems with defined appeal processes are rare in both sectors. Even the higher of the two (Enforcement) only reaches 6.4%.
*   **Sector Disparity:** Despite the low overall numbers, there is a massive relative difference. An AI system in the Enforcement sector is roughly **32 times more likely** (6.4% vs 0.2%) to have a defined appeal process than one in the Benefits sector.
*   **Sample Size Context:** The "Benefits" sector has a much larger sample size ($n=529$) compared to "Enforcement" ($n=234$), yet the absolute number of systems with appeal processes is likely much lower in Benefits (approx. 1 system) compared to Enforcement (approx. 15 systems). This highlights a potential systemic lack of recourse mechanisms in benefits-related AI systems compared to enforcement-related ones.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
