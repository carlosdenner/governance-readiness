# Experiment 272: node_5_97

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_97` |
| **ID in Run** | 272 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:54:35.842474+00:00 |
| **Runtime** | 165.9s |
| **Parent** | `node_4_51` |
| **Children** | None |
| **Creation Index** | 273 |

---

## Hypothesis

> The Justice Sector Transparency Deficit: Agencies in the 'Law Enforcement' and
'Justice' topic areas are statistically less likely to implement 'Public Notice'
or 'Appeal Processes' for their AI systems compared to 'Health' or 'Social
Services' agencies, despite the higher civil liberties risks.

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

**Objective:** Investigate sector-specific transparency gaps in federal AI inventories.

### Steps
- 1. Filter EO13960 data.
- 2. Categorize '3_agency' or '8_topic_area' into 'Justice/Security' vs. 'Health/Services'.
- 3. Compare the boolean rates of '59_ai_notice' and '65_appeal_process' between these two groups.
- 4. Use a Z-test for proportions to check for significance.

### Deliverables
- Comparison of transparency rates by agency type; Statistical significance metrics.

---

## Analysis

The experiment successfully tested the 'Justice Sector Transparency Deficit'
hypothesis using the EO 13960 dataset.

**Findings:**
1.  **Public Notice:** The adoption rate was **0.0%** for both sectors. No
agencies in either the 'Justice/Security' (n=385) or 'Health/Services' (n=886)
groups reported implementing AI public notices in this dataset. Consequently, no
statistical difference could be calculated.
2.  **Appeal Process:** The results **refute** the hypothesis. Contrary to the
prediction that the Justice sector would have lower transparency, the
'Justice/Security' sector showed a significantly **higher** adoption rate
(4.68%) compared to the 'Health/Services' sector (0.34%).
3.  **Statistical Significance:** A Z-test confirmed this difference is
statistically significant (Z=5.57, p < 0.0001).

**Conclusion:** The hypothesis is rejected. While overall maturity is extremely
low across the federal government (below 5%), the Justice sector is
statistically *more* likely to implement appeal processes than the Health
sector, possibly due to pre-existing legal frameworks regarding due process.

---

## Review

The experiment successfully tested the 'Justice Sector Transparency Deficit'
hypothesis using the EO 13960 dataset.

**Findings:**
1.  **Public Notice:** The adoption rate was **0.0%** for both sectors. No
agencies in either the 'Justice/Security' (n=385) or 'Health/Services' (n=886)
groups reported implementing AI public notices in this dataset. Consequently, no
statistical difference could be calculated.
2.  **Appeal Process:** The results **refute** the hypothesis. Contrary to the
prediction that the Justice sector would have lower transparency, the
'Justice/Security' sector showed a significantly **higher** adoption rate
(4.68%) compared to the 'Health/Services' sector (0.34%).
3.  **Statistical Significance:** A Z-test confirmed this difference is
statistically significant (Z=5.57, p < 0.0001).

**Conclusion:** The hypothesis is rejected. While overall maturity is extremely
low across the federal government (below 5%), the Justice sector is
statistically *more* likely to implement appeal processes than the Health
sector, possibly due to pre-existing legal frameworks regarding due process.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest

# [debug]
print("Starting experiment: Justice Sector Transparency Deficit")

# 1. Load Data
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Dataset not found in parent directory. Trying current directory.")
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 records found: {len(eo_data)}")

# 2. Categorize Agencies
# Define keyword lists for categorization based on Agency Name (3_agency)
justice_security_keywords = [
    'Justice', 'Homeland Security', 'Defense', 'State', 'Intelligence', 
    'Investigation', 'Prisons', 'Police'
]
health_services_keywords = [
    'Health', 'Human Services', 'Education', 'Social Security', 'Veterans', 
    'Labor', 'Housing', 'Agriculture', 'Interior'
]

def categorize_agency(agency_name):
    if pd.isna(agency_name):
        return 'Other'
    agency_str = str(agency_name).lower()
    
    # Check Justice/Security
    for kw in justice_security_keywords:
        if kw.lower() in agency_str:
            return 'Justice/Security'
            
    # Check Health/Services
    for kw in health_services_keywords:
        if kw.lower() in agency_str:
            return 'Health/Services'
            
    return 'Other'

eo_data['Sector_Group'] = eo_data['3_agency'].apply(categorize_agency)

# Filter out 'Other'
analysis_df = eo_data[eo_data['Sector_Group'].isin(['Justice/Security', 'Health/Services'])].copy()
print("\nGroup Sizes:")
print(analysis_df['Sector_Group'].value_counts())

# 3. Clean Target Columns
# Convert to boolean/numeric (1 for Yes, 0 for No/Other)
def clean_boolean_col(val):
    if pd.isna(val):
        return 0
    if str(val).strip().lower() == 'yes':
        return 1
    return 0

target_cols = {
    '59_ai_notice': 'Public Notice',
    '65_appeal_process': 'Appeal Process'
}

results = {}

for col, label in target_cols.items():
    analysis_df[col + '_clean'] = analysis_df[col].apply(clean_boolean_col)
    
    # Group statistics
    stats = analysis_df.groupby('Sector_Group')[col + '_clean'].agg(['sum', 'count', 'mean'])
    stats['percentage'] = stats['mean'] * 100
    results[label] = stats

    print(f"\n--- {label} Statistics ---")
    print(stats[['count', 'sum', 'percentage']])

# 4. Statistical Tests (Z-test)
print("\n--- Statistical Significance (Z-Test) ---")

sig_results = []

for col, label in target_cols.items():
    col_clean = col + '_clean'
    
    # Extract counts and nobs for the two groups
    group_stats = analysis_df.groupby('Sector_Group')[col_clean].agg(['sum', 'count'])
    
    # Ensure we have both groups
    if len(group_stats) != 2:
        print(f"Skipping {label}: Insufficient groups.")
        continue
        
    count = np.array([group_stats.loc['Justice/Security', 'sum'], group_stats.loc['Health/Services', 'sum']])
    nobs = np.array([group_stats.loc['Justice/Security', 'count'], group_stats.loc['Health/Services', 'count']])
    
    stat, pval = proportions_ztest(count, nobs, alternative='two-sided')
    
    print(f"{label}:")
    print(f"  Justice/Security Rate: {group_stats.loc['Justice/Security', 'sum']/group_stats.loc['Justice/Security', 'count']:.2%}")
    print(f"  Health/Services Rate:  {group_stats.loc['Health/Services', 'sum']/group_stats.loc['Health/Services', 'count']:.2%}")
    print(f"  Z-score: {stat:.4f}")
    print(f"  P-value: {pval:.4e}")
    sig_results.append({'Metric': label, 'P-value': pval, 'Significant': pval < 0.05})

# 5. Visualization
labels = list(target_cols.values())
justice_means = [analysis_df[analysis_df['Sector_Group']=='Justice/Security'][col + '_clean'].mean() for col in target_cols.keys()]
health_means = [analysis_df[analysis_df['Sector_Group']=='Health/Services'][col + '_clean'].mean() for col in target_cols.keys()]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, justice_means, width, label='Justice/Security', color='#d62728')
rects2 = ax.bar(x + width/2, health_means, width, label='Health/Services', color='#1f77b4')

ax.set_ylabel('Adoption Rate')
ax.set_title('Transparency Controls by Sector (EO 13960)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1%}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment: Justice Sector Transparency Deficit
Dataset not found in parent directory. Trying current directory.
EO 13960 records found: 1757

Group Sizes:
Sector_Group
Health/Services     886
Justice/Security    385
Name: count, dtype: int64

--- Public Notice Statistics ---
                  count  sum  percentage
Sector_Group                            
Health/Services     886    0         0.0
Justice/Security    385    0         0.0

--- Appeal Process Statistics ---
                  count  sum  percentage
Sector_Group                            
Health/Services     886    3    0.338600
Justice/Security    385   18    4.675325

--- Statistical Significance (Z-Test) ---
Public Notice:
  Justice/Security Rate: 0.00%
  Health/Services Rate:  0.00%
  Z-score: nan
  P-value: nan
Appeal Process:
  Justice/Security Rate: 4.68%
  Health/Services Rate:  0.34%
  Z-score: 5.5734
  P-value: 2.4987e-08

STDERR:
/usr/local/lib/python3.13/site-packages/statsmodels/stats/weightstats.py:792: RuntimeWarning: invalid value encountered in scalar divide
  zstat = value / std


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Grouped Bar Chart (or Clustered Bar Plot).
*   **Purpose:** The plot compares the adoption rates of two specific transparency controls ("Public Notice" and "Appeal Process") across two different government sectors ("Justice/Security" and "Health/Services").

### 2. Axes
*   **X-axis:**
    *   **Label/Title:** Categorical labels representing the type of transparency control: "Public Notice" and "Appeal Process".
*   **Y-axis:**
    *   **Label/Title:** "Adoption Rate".
    *   **Range:** From 0.0 to 1.0 (representing 0% to 100%).
    *   **Units:** The axis uses decimal notation (0.2, 0.4, etc.), but the data annotations are presented as percentages.

### 3. Data Trends
*   **Overall Pattern:** The most striking trend is the extremely low adoption rate across all categories. The vast majority of the chart area is empty, indicating that these transparency controls are rarely implemented in the contexts measured.
*   **Public Notice:**
    *   Both sectors ("Justice/Security" and "Health/Services") show an adoption rate of **0.0%**. There are no visible bars for this category.
*   **Appeal Process:**
    *   **Tallest Bar:** The "Justice/Security" sector (Red) has the highest value on the chart, though it is still quite low at **4.7%**.
    *   **Shortest Bar (Non-zero):** The "Health/Services" sector (Blue) has a barely visible bar representing **0.3%**.

### 4. Annotations and Legends
*   **Title:** "Transparency Controls by Sector (EO 13960)".
*   **Legend:** Located in the top right corner:
    *   **Red:** Represents the "Justice/Security" sector.
    *   **Blue:** Represents the "Health/Services" sector.
*   **Data Labels:** Percentage values are annotated directly above the corresponding bar locations to provide precise values:
    *   "0.0%" for both Public Notice categories.
    *   "4.7%" for Justice/Security under Appeal Process.
    *   "0.3%" for Health/Services under Appeal Process.

### 5. Statistical Insights
*   **Zero Compliance for Public Notice:** Neither sector has adopted "Public Notice" controls according to this dataset (0.0% rate).
*   **Sector Disparity:** There is a notable disparity in the "Appeal Process" category. The Justice/Security sector is more than 15 times more likely to have an appeal process than the Health/Services sector (4.7% vs 0.3%), although both rates are objectively very low.
*   **Low Maturity:** The fact that the highest adoption rate on the entire chart is less than 5% suggests that the implementation of these specific transparency controls (mandated or suggested by EO 13960) is in a very early or non-existent stage for these sectors.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
