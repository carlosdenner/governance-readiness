# Experiment 46: node_4_15

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_15` |
| **ID in Run** | 46 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:11:56.559794+00:00 |
| **Runtime** | 270.5s |
| **Parent** | `node_3_11` |
| **Children** | `node_5_15`, `node_5_77` |
| **Creation Index** | 47 |

---

## Hypothesis

> The 'Black Box' Procurement Risk: Commercially procured AI systems in government
are significantly less likely to have documented 'Disparity Mitigation'
strategies compared to Custom-developed systems, due to proprietary vendor
limitations.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7581 (Likely True) |
| **Posterior** | 0.4176 (Maybe False) |
| **Surprise** | -0.4086 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 2.0 |
| Maybe True | 28.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 58.0 |
| Definitely False | 2.0 |

---

## Experiment Plan

**Objective:** Compare the presence of bias/disparity mitigation documentation between Commercial and Custom AI systems in the EO 13960 inventory.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for `source_table` == 'eo13960_scored'.
- 2. Clean `10_commercial_ai` to binary (Commercial vs. Custom).
- 3. Clean `62_disparity_mitigation` to binary (Yes/Present vs. No/None/Null). Note: Treat 'Not Applicable' as No or exclude, but preferably treat as 'No Mitigation'.
- 4. Create a contingency table: Commercial Status vs. Disparity Mitigation.
- 5. Compute the percentage of systems with mitigation for each group.
- 6. Perform a Chi-square test of independence.

### Deliverables
- Contingency table, bar chart of mitigation rates by source type, and Chi-square test results.

---

## Analysis

The experiment successfully tested the 'Black Box Procurement Risk' hypothesis
using the EO 13960 dataset. After correcting the column mapping (using
`22_dev_method` for procurement source), the code analyzed 920 AI systems, split
into Commercial/Vendor (N=481) and Custom/In-House (N=439) categories.

**Findings:**
1.  **Low Overall Compliance:** Documented disparity mitigation strategies are
rare across the board. Only 8.1% of Commercial systems and 10.3% of Custom
systems had valid documentation.
2.  **Hypothesis Evaluation:** While the results directionally support the
hypothesis (Custom systems had a slightly higher compliance rate: 10.3% vs
8.1%), the difference is **not statistically significant** (Chi-Square = 1.02,
p-value = 0.311).

**Conclusion:** The hypothesis is **not supported**. The data suggests that the
lack of bias mitigation documentation is a systemic issue across federal AI
deployments, regardless of whether the system is procured from a vendor or
developed in-house. The 'Black Box' nature of commercial vendors does not appear
to be a statistically significant driver of this transparency gap compared to
government-developed systems.

---

## Review

The experiment was faithfully implemented and successfully overcame previous
data mapping issues. By correctly using the `22_dev_method` column to classify
procurement types and implementing robust text cleaning for the
`62_disparity_mitigation` field, the analysis provided a valid test of the
'Black Box Procurement Risk' hypothesis.

**Results:**
- **Sample Size:** 920 AI systems analyzed (481 Commercial, 439 Custom).
- **Documentation Rates:** Commercial systems had a documented disparity
mitigation rate of 8.1% (39/481), while Custom/In-House systems had a rate of
10.3% (45/439).
- **Statistical Significance:** The Chi-square test yielded a p-value of 0.311
(Statistic=1.02), indicating the difference is not statistically significant.

**Findings:**
The hypothesis is **not supported**. While Custom systems showed a marginally
higher rate of documentation, the difference is statistically negligible. The
primary finding is a systemic lack of transparency: approximately 90% of all
federal AI systems lack documented disparity mitigation strategies, regardless
of whether they are commercially procured or developed in-house. The 'black box'
nature of vendors does not appear to be the primary driver of this documentation
gap.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Define file path
filename = 'astalabs_discovery_all_data.csv'
if os.path.exists(f'../{filename}'):
    filepath = f'../{filename}'
elif os.path.exists(filename):
    filepath = filename
else:
    filepath = filename 
    print("File not found in expected locations, attempting current dir...")

# Load dataset
df = pd.read_csv(filepath, low_memory=False)

# Filter for EO 13960
target_source = 'eo13960_scored'
eo_df = df[df['source_table'] == target_source].copy()
print(f"Filtered for {target_source}: {len(eo_df)} records")

# Columns
col_method = '22_dev_method'
col_mitig = '62_disparity_mitigation'

# Map Procurement Method (Commercial vs Custom)
def map_procurement(val):
    s = str(val).strip().lower()
    if 'contracting' in s and 'in-house' not in s:
        return 'Commercial (Vendor)'
    elif 'in-house' in s and 'contracting' not in s:
        return 'Custom (In-House)'
    else:
        return np.nan # Exclude Mixed or Unknown for clean comparison

eo_df['procurement_type'] = eo_df[col_method].apply(map_procurement)

# Map Mitigation Status (Documented vs Not)
def map_mitigation(val):
    if pd.isna(val):
        return 'Not Documented'
    s = str(val).strip().lower()
    # Check for non-substantive answers
    if s in ['nan', 'none', 'n/a', 'no', 'not applicable', 'none.', 'na']:
        return 'Not Documented'
    if len(s) < 5: # arbitrarily short strings likely meaningless
        return 'Not Documented'
    return 'Documented'

eo_df['mitigation_status'] = eo_df[col_mitig].apply(map_mitigation)

# Filter for analysis
analysis_df = eo_df.dropna(subset=['procurement_type'])

print(f"Records for analysis (Commercial vs Custom): {len(analysis_df)}")
print("Breakdown by Procurement Type:")
print(analysis_df['procurement_type'].value_counts())

# Contingency Table
ct = pd.crosstab(analysis_df['procurement_type'], analysis_df['mitigation_status'])
print("\nContingency Table (Count):")
print(ct)

# Percentages
ct_pct = pd.crosstab(analysis_df['procurement_type'], analysis_df['mitigation_status'], normalize='index') * 100
print("\nContingency Table (Percentage):")
print(ct_pct)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(ct)
print(f"\nChi-Square Test Results:")
print(f"Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Plot
if 'Documented' in ct_pct.columns:
    rates = ct_pct['Documented']
else:
    rates = pd.Series([0, 0], index=ct.index)

plt.figure(figsize=(8, 6))
colors = ['#ff9999' if 'Commercial' in x else '#66b3ff' for x in rates.index]
bars = plt.bar(rates.index, rates.values, color=colors, edgecolor='black')
plt.title('Disparity Mitigation Documentation Rate: Commercial vs Custom AI')
plt.ylabel('Percent with Documented Mitigation (%)')
plt.xlabel('Procurement Method')
plt.ylim(0, max(rates.values)*1.2 if len(rates)>0 and max(rates.values)>0 else 100)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Filtered for eo13960_scored: 1757 records
Records for analysis (Commercial vs Custom): 920
Breakdown by Procurement Type:
procurement_type
Commercial (Vendor)    481
Custom (In-House)      439
Name: count, dtype: int64

Contingency Table (Count):
mitigation_status    Documented  Not Documented
procurement_type                               
Commercial (Vendor)          39             442
Custom (In-House)            45             394

Contingency Table (Percentage):
mitigation_status    Documented  Not Documented
procurement_type                               
Commercial (Vendor)    8.108108       91.891892
Custom (In-House)     10.250569       89.749431

Chi-Square Test Results:
Statistic: 1.0247
P-value: 3.1140e-01


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** The plot serves to compare categorical data, specifically contrasting the percentage of AI projects that document disparity mitigation efforts based on how the AI was procured (Commercial Vendor vs. Custom In-House development).

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Procurement Method"
    *   **Categories:** The axis displays two distinct categories: "Commercial (Vendor)" and "Custom (In-House)."
*   **Y-Axis:**
    *   **Title:** "Percent with Documented Mitigation (%)"
    *   **Units:** Percentage points.
    *   **Value Range:** The axis scale ranges from 0 to 12, with major tick marks at intervals of 2 (0, 2, 4, 6, 8, 10, 12).

### 3. Data Trends
*   **Highest Value:** The tallest bar represents "Custom (In-House)" AI solutions, reaching a value of **10.3%**.
*   **Lowest Value:** The shortest bar represents "Commercial (Vendor)" AI solutions, reaching a value of **8.1%**.
*   **Comparison:** There is a visible trend where in-house (custom) solutions show a higher rate of documentation for disparity mitigation compared to commercial (vendor) solutions.

### 4. Annotations and Legends
*   **Chart Title:** "Disparity Mitigation Documentation Rate: Commercial vs Custom AI" provides the overall context for the data.
*   **Data Labels:** Each bar is annotated with its exact numerical value directly above it ("8.1%" for Commercial and "10.3%" for Custom), allowing for precise reading without estimating from the y-axis.
*   **Color Coding:** The bars are distinct in color (salmon/light red for Commercial and light blue for Custom) to visually differentiate the two categories, though no separate legend box is provided or necessary given the clear x-axis labels.

### 5. Statistical Insights
*   **Performance Gap:** Custom (In-House) AI procurement is associated with a **2.2 percentage point higher** rate of documented disparity mitigation compared to Commercial (Vendor) AI.
*   **Relative Difference:** Custom solutions are approximately **27% more likely** (2.2 / 8.1) to have documented mitigation than commercial solutions.
*   **Overall Low Compliance:** Despite the difference between the two methods, the most significant insight is that both rates are quite low. Approximately **90%** of AI projects (89.7% of Custom and 91.9% of Commercial) do **not** have documented disparity mitigation, indicating a widespread lack of documentation regarding fairness and bias mitigation across the industry, regardless of the procurement source.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
