# Experiment 47: node_4_16

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_16` |
| **ID in Run** | 47 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:16:28.944883+00:00 |
| **Runtime** | 250.6s |
| **Parent** | `node_3_14` |
| **Children** | `node_5_82` |
| **Creation Index** | 48 |

---

## Hypothesis

> Vendor Transparency Gap: AI systems procured from commercial vendors are
significantly less likely to grant agencies 'Code Access' compared to systems
developed in-house, creating a 'black box' governance risk that correlates with
lower explainability.

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

**Objective:** Quantify the transparency trade-off inherent in commercial AI procurement within the federal government.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'eo13960_scored'.
- 2. Clean column '10_commercial_ai' to binary (Commercial vs. In-House) and '38_code_access' to binary (Yes vs. No/Missing).
- 3. Create a contingency table of Procurement Source vs. Code Access.
- 4. Perform a Chi-square test of independence.
- 5. Calculate the Odds Ratio to quantify the likelihood of losing code access when choosing commercial solutions.

### Deliverables
- Contingency table; Chi-square statistic and p-value; Odds Ratio; Bar chart comparing Code Access rates by procurement type.

---

## Analysis

The experiment successfully tested the 'Vendor Transparency Gap' hypothesis
using the EO 13960 dataset.

1. **Data Processing**:
   - The '22_dev_method' column was used to categorize systems into
**Commercial** ('Developed with contracting resources', n=481) and **In-House**
('Developed in-house', n=439).
   - The '38_code_access' column was mapped to a binary indicator, where valid
'Yes' responses (including restricted access) were treated as having
transparency.

2. **Descriptive Statistics**:
   - **In-House Systems**: 80.6% (354/439) granted code access.
   - **Commercial Systems**: Only 27.4% (132/481) granted code access.

3. **Inferential Statistics**:
   - **Chi-Square Test**: The difference was highly significant (Chi-square =
258.49, p < 0.001), indicating that procurement source and code transparency are
strongly dependent.
   - **Odds Ratio**: The calculation focused on the *lack* of access. Commercial
systems were found to be **11.01 times more likely** to deny code access
compared to in-house systems (Odds Ratio for 'No Access').

**Conclusion**: The results overwhelmingly support the hypothesis. Relying on
commercial vendors introduces a severe 'black box' risk, with the vast majority
of such systems failing to provide the code transparency found in government-
developed solutions.

---

## Review

The experiment successfully tested the 'Vendor Transparency Gap' hypothesis
using the EO 13960 dataset. After correcting the column mapping for procurement
source (using '22_dev_method' instead of the descriptive '10_commercial_ai'),
the analysis proceeded with 920 relevant cases (481 Commercial, 439 In-House).

Key Findings:
1. **Descriptive Statistics**: There is a stark contrast in transparency. Only
27.4% of Commercial AI systems grant the agency code access, compared to 80.6%
of In-House systems.
2. **Statistical Significance**: The Chi-square test of independence yielded a
statistic of 258.49 (p < 0.001), confirming that code accessibility is strongly
dependent on the procurement method.
3. **Effect Size**: The Odds Ratio for 'No Access' is 11.01, indicating that
Commercial AI systems are roughly 11 times more likely to deny code access
compared to In-House solutions.

Conclusion: The results overwhelmingly support the hypothesis. The reliance on
commercial vendors introduces a significant 'black box' risk, with the vast
majority of such systems failing to provide the transparency levels standard in
government-developed software.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Load Data
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for 'eo13960_scored'
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored Subset: {len(eo_df)} rows")

# 3. Clean '22_dev_method' (Procurement Source)
# Map raw values to 'Commercial' or 'In-House'
# We ignore 'Developed with both...' and NaNs for the strict comparison groups
def map_procurement(val):
    s = str(val).strip()
    if s == 'Developed with contracting resources.':
        return 'Commercial'
    elif s == 'Developed in-house.':
        return 'In-House'
    return None

eo_df['procurement_type'] = eo_df['22_dev_method'].apply(map_procurement)

# Filter to only keep the two groups of interest
analysis_df = eo_df.dropna(subset=['procurement_type']).copy()
print(f"Analysis Subset (Commercial vs In-House): {len(analysis_df)} rows")
print(analysis_df['procurement_type'].value_counts())

# 4. Clean '38_code_access'
# Map to Binary: Yes (1) vs No/Missing (0)
# 'Yes' variants include: 'Yes – agency has access...', 'Yes – source code is publicly...', 'Yes', 'YES'
def map_access(val):
    s = str(val).lower()
    if 'yes' in s:
        return 1
    else:
        return 0

analysis_df['has_code_access'] = analysis_df['38_code_access'].apply(map_access)
print("Code Access Distribution:\n", analysis_df['has_code_access'].value_counts())

# 5. Contingency Table
# Rows: Procurement (Commercial, In-House)
# Cols: Code Access (0=No, 1=Yes)
contingency = pd.crosstab(analysis_df['procurement_type'], analysis_df['has_code_access'])
contingency.columns = ['No Access', 'Access']
print("\nContingency Table:\n", contingency)

# 6. Chi-square Test
chi2, p, dof, ex = stats.chi2_contingency(contingency)
print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# 7. Odds Ratio Calculation
# We want the odds of *losing* code access (No Access) for Commercial vs In-House.
# OR = (Odds No Access | Commercial) / (Odds No Access | In-House)
# Odds No Access = (Count No Access) / (Count Access)

# Extract counts
# Row 'Commercial'
comm_no = contingency.loc['Commercial', 'No Access']
comm_yes = contingency.loc['Commercial', 'Access']

# Row 'In-House'
house_no = contingency.loc['In-House', 'No Access']
house_yes = contingency.loc['In-House', 'Access']

# Add small epsilon if any cell is zero to avoid div by zero (though unlikely here)
if comm_yes == 0 or house_no == 0:
    print("Warning: Zero count detected, adding epsilon.")
    comm_yes += 0.5
    house_no += 0.5
    comm_no += 0.5
    house_yes += 0.5

odds_commercial = comm_no / comm_yes
odds_inhouse = house_no / house_yes

odds_ratio = odds_commercial / odds_inhouse

print(f"Odds of No Access (Commercial): {odds_commercial:.4f}")
print(f"Odds of No Access (In-House): {odds_inhouse:.4f}")
print(f"Odds Ratio (Commercial vs In-House for No Access): {odds_ratio:.4f}")

# 8. Visualization
# Calculate percentages for the plot
rates = analysis_df.groupby('procurement_type')['has_code_access'].mean()
# rates gives % with access. We might want to plot % with access to show the gap.
# If commercial has lower access, its bar will be lower.

plt.figure(figsize=(8, 6))
colors = ['#d9534f', '#5bc0de'] # Red for Commercial, Blue for In-House usually, but let's see order
# rates.index is alphabetical: Commercial, In-House
plt.bar(rates.index, rates.values, color=colors)
plt.ylabel('Proportion with Code Access')
plt.title('Vendor Transparency Gap: Code Access by Procurement Source')
plt.ylim(0, 1.1)

for i, v in enumerate(rates.values):
    plt.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')

plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: EO 13960 Scored Subset: 1757 rows
Analysis Subset (Commercial vs In-House): 920 rows
procurement_type
Commercial    481
In-House      439
Name: count, dtype: int64
Code Access Distribution:
 has_code_access
1    486
0    434
Name: count, dtype: int64

Contingency Table:
                   No Access  Access
procurement_type                   
Commercial              349     132
In-House                 85     354

Chi-square Statistic: 258.4928
P-value: 3.6564e-58
Odds of No Access (Commercial): 2.6439
Odds of No Access (In-House): 0.2401
Odds Ratio (Commercial vs In-House for No Access): 11.0112


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot compares a specific metric ("Proportion with Code Access") across two distinct categorical groups ("Commercial" and "In-House" procurement sources). It is designed to visualize the magnitude of difference—or "gap"—between these two groups.

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Label:** Implicitly represents "Procurement Source" based on the chart title and category names.
    *   **Categories:** The axis features two discrete categories: **"Commercial"** and **"In-House"**.
*   **Y-Axis (Vertical):**
    *   **Label:** "Proportion with Code Access".
    *   **Units:** The axis is scaled in decimals representing proportions (0.0 to 1.0), which corresponds to 0% to 100%.
    *   **Range:** The tick marks range from **0.0 to 1.0**, with an interval of 0.2. The visible plot area extends slightly to roughly 1.1 to accommodate the bar height and labels.

### 3. Data Trends
*   **Tallest Bar:** The **"In-House"** bar (colored light blue) is the tallest, indicating a high prevalence of code access.
*   **Shortest Bar:** The **"Commercial"** bar (colored red) is significantly shorter, indicating a low prevalence of code access.
*   **Pattern:** There is a stark contrast between the two categories. The visual disparity suggests that code accessibility is heavily dependent on the source of the software, with In-House solutions offering vastly superior transparency compared to Commercial ones.

### 4. Annotations and Legends
*   **Title:** "Vendor Transparency Gap: Code Access by Procurement Source" – This title frames the data as an issue of transparency, specifically highlighting a "gap."
*   **Data Labels:**
    *   **Commercial:** Annotated with **"27.4%"** sitting directly atop the red bar.
    *   **In-House:** Annotated with **"80.6%"** sitting directly atop the blue bar.
    *   *Note:* While the Y-axis uses decimal notation (e.g., 0.8), the annotations translate these figures into percentages for easier reading.
*   **Color Coding:** The use of **Red** for Commercial (often associated with "stop" or "warning") and **Blue** for In-House helps reinforce the negative transparency implication of the Commercial bar versus the positive implication of the In-House bar.

### 5. Statistical Insights
*   **Significant Disparity:** There is a massive "transparency gap" of **53.2 percentage points** between the two groups ($80.6\% - 27.4\%$).
*   **Relative Likelihood:** Code access is nearly **3 times more likely** ($\approx 2.94x$) to be available for In-House procurement compared to Commercial vendors.
*   **Conclusion:** The data strongly supports the hypothesis that commercial vendors are often "black boxes," restricting code access for the majority of their products (nearly 3/4ths), whereas in-house development projects maintain code transparency in the vast majority of cases (over 4/5ths).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
