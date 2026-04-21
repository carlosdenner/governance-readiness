# Experiment 50: node_4_17

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_17` |
| **ID in Run** | 50 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:20:41.563561+00:00 |
| **Runtime** | 390.9s |
| **Parent** | `node_3_15` |
| **Children** | `node_5_36`, `node_5_71` |
| **Creation Index** | 51 |

---

## Hypothesis

> The 'Public-Allocative Bias': Incidents attributed to Public Sector deployers
are significantly more likely to cause 'Allocative Harms' (e.g., discrimination,
denial of benefits) compared to Private Sector incidents, which skew towards
'Quality of Service' or 'Physical' harms.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7903 (Likely True) |
| **Posterior** | 0.4341 (Maybe False) |
| **Surprise** | -0.4275 |
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
| Maybe False | 60.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Analyze the relationship between deployer sector and harm type.

### Steps
- 1. Filter AIID incidents.
- 2. Categorize 'Alleged deployer of AI system' into 'Public' (Gov, Police, Agency) and 'Private' (Corp, Company, Ltd) using keyword matching.
- 3. Categorize 'Harm Type' or 'Harm Domain' into 'Allocative' (Civil Rights, Economic Opportunity) vs. 'Physical/Performance'.
- 4. Create a contingency table.
- 5. Perform a Chi-Square Test of Independence.

### Deliverables
- 1. Contingency table of Deployer Sector vs. Harm Type.
- 2. Chi-Square test results.
- 3. Stacked bar chart of harm proportions by sector.

---

## Analysis

The experiment tested the 'Public-Allocative Bias' hypothesis, which posited
that Public sector AI incidents are disproportionately associated with
Allocative harms (discrimination, rights violations), while Private sector
incidents lean towards Physical/Safety harms.

**Methodology:**
Data from the AI Incident Database (AIID) was filtered to n=86 incidents where
both sector and harm type could be definitively categorized. 'Allocative' harm
was identified where the 'Harm Distribution Basis' was explicitly recorded
(e.g., race, sex), and 'Physical' harm was identified where 'Tangible Harm' was
recorded as 'definitively occurred' or 'imminent risk'.

**Results:**
- **Directional Support:** The data showed a trend consistent with the
hypothesis. Public sector incidents were 52% Allocative (vs. 48% Physical),
whereas Private sector incidents were only 42% Allocative (vs. 58% Physical).
- **Statistical Significance:** The Chi-Square test yielded a p-value of 0.5578
(Chi2=0.34), which is well above the 0.05 threshold.

**Conclusion:**
The hypothesis is **not supported**. While the observed proportions aligned with
the expected direction, the difference was not statistically significant. The
analysis was constrained by the sparsity of the metadata (only 86 of 1,362
incidents had sufficient data for this specific cross-tabulation), suggesting
that a larger coded sample might be necessary to detect a genuine effect.

---

## Review

The experiment successfully tested the 'Public-Allocative Bias' hypothesis using
the AI Incident Database (AIID). After resolving initial data parsing issues
with the 'Alleged deployer' (JSON strings) and 'Harm Domain' columns, the
analysis proceeded with a subset of 86 fully coded incidents.

**Hypothesis**: Public Sector incidents are disproportionately associated with
Allocative Harms (discrimination, rights violations), whereas Private Sector
incidents skew towards Physical/Safety harms.

**Results**:
- **Descriptive Trends**: The data showed a directional trend consistent with
the hypothesis. Public Sector incidents were 52% Allocative / 48% Physical,
while Private Sector incidents were 42% Allocative / 58% Physical.
- **Statistical Significance**: The Chi-Square test yielded a p-value of 0.5578
(Chi2=0.34), indicating that the observed difference is not statistically
significant.

**Conclusion**: The hypothesis is **not supported**. While the proportions
leaned in the predicted direction, the sample size (N=86) was insufficient to
prove a significant association. The analysis highlights a data quality
limitation: most incidents in the database lack structured metadata for 'Harm
Distribution Basis' or definitive 'Tangible Harm' assessments, restricting the
power of the statistical test.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import re
import sys

# 1. Load Dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID Incidents: {len(aiid_df)}")

# 2. Define Columns
deployer_col = 'Alleged deployer of AI system'
harm_dist_col = 'Harm Distribution Basis'
tangible_col = 'Tangible Harm'

# 3. Clean and Categorize Sector
def categorize_sector(val):
    if pd.isna(val):
        return 'Unknown'
    # Clean JSON artifacts
    val_clean = re.sub(r'[\[\]"\']', '', str(val)).lower()
    
    # Explicit Private Entities (Top frequency check)
    private_entities = [
        'tesla', 'google', 'openai', 'facebook', 'amazon', 'meta', 'microsoft', 
        'cruise', 'waymo', 'uber', 'xai', 'tiktok', 'youtube', 'apple', 'twitter',
        'snapchat', 'instagram', 'whatsapp', 'linkedin', 'salesforce', 'ibm', 'intel',
        'adobe', 'oracle', 'nvidia', 'palantir', 'deepmind', 'stability ai', 'midjourney'
    ]
    
    # General Private Keywords
    private_keywords = [
        'inc', 'corp', 'llc', 'ltd', 'company', 'technologies', 'systems', 
        'solutions', 'group', 'motors', 'airlines', 'bank', 'entertainment'
    ]
    
    # Public Keywords
    public_keywords = [
        'police', 'government', 'dept', 'department', 'ministry', 'agency', 
        'commission', 'authority', 'council', 'state', 'city', 'county', 
        'federal', 'national', 'bureau', 'sheriff', 'nhs', 'army', 'navy', 
        'air force', 'dhs', 'fbi', 'cia', 'school', 'university', 'college', 
        'court', 'judge', 'municipality', 'parliament', 'congress'
    ]
    
    # Classification Logic
    if any(e == val_clean or e in val_clean.split('-') for e in private_entities):
        return 'Private'
    if any(k in val_clean for k in private_keywords):
        return 'Private'
    if any(k in val_clean for k in public_keywords):
        return 'Public'
        
    return 'Other'

# 4. Categorize Harm Type
def categorize_harm(row):
    dist_basis = str(row.get(harm_dist_col, '')).lower()
    tangible = str(row.get(tangible_col, '')).lower()
    
    # Allocative Signal: Valid entry in 'Harm Distribution Basis'
    # Exclude: 'none', 'unclear', 'nan', empty strings
    if dist_basis not in ['none', 'unclear', 'nan', '']:
        return 'Allocative'
    
    # Physical Signal: 'Tangible Harm' explicitly indicates occurrence
    if 'definitively occurred' in tangible or 'imminent risk' in tangible:
        return 'Physical'
        
    return 'Other'

# Apply Categorization
aiid_df['Sector_Category'] = aiid_df[deployer_col].apply(categorize_sector)
aiid_df['Harm_Category'] = aiid_df.apply(categorize_harm, axis=1)

# 5. Filter for Analysis
analysis_df = aiid_df[
    (aiid_df['Sector_Category'].isin(['Public', 'Private'])) & 
    (aiid_df['Harm_Category'].isin(['Allocative', 'Physical']))
].copy()

print(f"Records for Analysis: {len(analysis_df)}")
print("\n--- Counts by Sector ---")
print(analysis_df['Sector_Category'].value_counts())
print("\n--- Counts by Harm Type ---")
print(analysis_df['Harm_Category'].value_counts())

if len(analysis_df) < 5:
    print("Insufficient data for statistical testing.")
    sys.exit(0)

# 6. Statistical Test (Chi-Square)
contingency_table = pd.crosstab(analysis_df['Sector_Category'], analysis_df['Harm_Category'])
print("\n--- Contingency Table ---")
print(contingency_table)

chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\n--- Chi-Square Results ---")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

if p < 0.05:
    print("Conclusion: Significant relationship found between Sector and Harm Type.")
else:
    print("Conclusion: No significant relationship found.")

# 7. Visualization
# Normalize to get proportions
props = contingency_table.div(contingency_table.sum(axis=1), axis=0)

plt.figure(figsize=(10, 6))
# Plot stacked bar
ax = props.plot(kind='bar', stacked=True, color=['#d62728', '#1f77b4'], ax=plt.gca())

plt.title('Proportion of Allocative vs. Physical Harms by Sector')
plt.ylabel('Proportion')
plt.xlabel('Deployer Sector')
plt.legend(title='Harm Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()

# Annotate bars
for c in ax.containers:
    # Only label non-zero segments
    labels = [f'{v.get_height():.2f}' if v.get_height() > 0.01 else '' for v in c]
    ax.bar_label(c, labels=labels, label_type='center', color='white', weight='bold')

plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total AIID Incidents: 1362
Records for Analysis: 86

--- Counts by Sector ---
Sector_Category
Private    59
Public     27
Name: count, dtype: int64

--- Counts by Harm Type ---
Harm_Category
Physical      47
Allocative    39
Name: count, dtype: int64

--- Contingency Table ---
Harm_Category    Allocative  Physical
Sector_Category                      
Private                  25        34
Public                   14        13

--- Chi-Square Results ---
Chi2 Statistic: 0.3435
P-value: 5.5780e-01
Conclusion: No significant relationship found.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Plot.
*   **Purpose:** The plot is designed to compare the relative proportions of two distinct categories of harm ("Allocative" and "Physical") across two different sectors ("Private" and "Public"). By stacking the bars to a total height of 1.0, it emphasizes the composition or percentage distribution within each sector rather than absolute counts.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Deployer Sector"
    *   **Categories:** Two categorical variables: "Private" and "Public".
*   **Y-Axis:**
    *   **Label:** "Proportion"
    *   **Range:** The axis ranges from 0.0 to 1.0 (representing 0% to 100%). The tick marks are spaced at intervals of 0.2.

### 3. Data Trends
*   **Private Sector:**
    *   **Physical Harms (Blue):** This segment is the larger portion of the bar, representing a proportion of **0.58** (58%).
    *   **Allocative Harms (Red):** This segment is the smaller portion, representing a proportion of **0.42** (42%).
    *   **Trend:** In the Private sector, physical harms are more prevalent than allocative harms.
*   **Public Sector:**
    *   **Allocative Harms (Red):** This segment is the larger portion of the bar, representing a proportion of **0.52** (52%).
    *   **Physical Harms (Blue):** This segment is the smaller portion, representing a proportion of **0.48** (48%).
    *   **Trend:** In the Public sector, allocative harms are slightly more prevalent than physical harms.

### 4. Annotations and Legends
*   **Title:** "Proportion of Allocative vs. Physical Harms by Sector" describes the overall subject of the chart.
*   **Legend:** Located on the right side titled "Harm Type". It indicates:
    *   **Red:** Allocative Harms.
    *   **Blue:** Physical Harms.
*   **Value Annotations:** White numeric labels are centered within each bar segment, explicitly stating the exact proportion values (e.g., "0.58", "0.42", "0.52", "0.48"). This aids in precise reading of the data without relying solely on axis estimation.

### 5. Statistical Insights
*   **Sectoral Inversion:** There is an inversion in the dominant type of harm between the two sectors. The Private sector leans towards **Physical harms** (58%), whereas the Public sector leans towards **Allocative harms** (52%).
*   **Comparison of Harm Types:**
    *   Allocative harms are **10 percentage points higher** in the Public sector (52%) compared to the Private sector (42%).
    *   Conversely, Physical harms are **10 percentage points higher** in the Private sector (58%) compared to the Public sector (48%).
*   **Overall Balance:** While there is a clear distinction between the sectors, the split is relatively balanced in both cases. Neither sector is exclusively dominated by one type of harm; the distributions are close to a 50/50 split, with only mild deviations favoring one type over the other.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
