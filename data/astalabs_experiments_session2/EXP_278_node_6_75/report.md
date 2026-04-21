# Experiment 278: node_6_75

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_75` |
| **ID in Run** | 278 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T14:11:36.917686+00:00 |
| **Runtime** | 302.1s |
| **Parent** | `node_5_48` |
| **Children** | None |
| **Creation Index** | 279 |

---

## Hypothesis

> Sector-Blind Security: The distribution of victim sectors in real-world AI
incidents (AIID) differs significantly from the target sectors in adversarial
case studies (ATLAS), indicating a misalignment between security research focus
and actual public risk.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9758 (Definitely True) |
| **Posterior** | 0.7783 (Likely True) |
| **Surprise** | -0.2370 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 29.0 |
| Maybe True | 1.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 22.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Compare the sector distributions of theoretical threats vs. actual incidents.

### Steps
- 1. Load `aiid_incidents` and `atlas_cases`.
- 2. Normalize sector columns in both datasets to a common taxonomy (e.g., 'Defense', 'Healthcare', 'Consumer', 'Finance').
- 3. Calculate the percentage distribution for each source.
- 4. Perform a Chi-Square Goodness of Fit test (using ATLAS as expected, AIID as observed) or compare two independent proportions.

### Deliverables
- Side-by-side bar chart of Sector Distributions; Statistical comparison results.

---

## Analysis

The experiment successfully executed the code to compare sector distributions,
but the findings are inconclusive due to significant data limitations.

1. **Data Quality Issue**: The normalization logic categorized 100% of ATLAS
cases and 91.6% of AIID incidents as 'Other/Unknown'. This strongly suggests
that the structured columns used (`sector` for ATLAS and `Sector of Deployment`
for AIID) are either largely empty in this sparse dataset or contain values that
did not match the keyword lists.

2. **Statistical Artifact**: The Chi-Square test yielded a p-value of 0.5783,
indicating 'no significant difference'. However, this result is misleading
because it primarily compares the overwhelming number of 'Unknown' records in
both datasets.

3. **Hypothesis validation**: The hypothesis cannot be validated or rejected
based on this result. The experiment highlights that sector information is
likely unstructured (hidden in text summaries) rather than structured, similar
to the 'Harm' categories in the previous experiment. Future iterations should
use NLP or keyword extraction on the `summary` or `description` fields to
retrieve sector data.

---

## Review

The experiment was faithfully executed, but the results highlight a critical
data quality issue rather than a confirmation or rejection of the hypothesis.

**Hypothesis**: 'Sector-Blind Security' (that victim sectors in real-world
incidents differ from target sectors in research).

**Results**:
1. **Data Limitations**: The keyword-based normalization strategy failed to
retrieve sector data for the vast majority of records. 91.6% of AIID incidents
and 100% of ATLAS cases were categorized as 'Other/Unknown', indicating that the
structured 'sector' columns are likely empty or do not contain the expected
keywords.
2. **Statistical Invalidity**: While a Chi-Square test was performed (p=0.5783),
the result is statistical noise driven by the overwhelming number of 'Unknown'
labels in both groups. It does not reflect a true comparison of sector
distributions.

**Findings**: The experiment successfully demonstrated that sector information
in these datasets is not readily available in the target structured columns. As
noted in the analysis, future iterations must employ Natural Language Processing
(NLP) or keyword extraction on the unstructured 'summary' or 'description'
fields to recover this metadata.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

# [debug]
print("Starting experiment...")

# 1. Load the dataset
# Try loading from parent directory first, then current directory
file_name = 'astalabs_discovery_all_data.csv'
paths = [f'../{file_name}', file_name]
ds_path = None

for p in paths:
    if os.path.exists(p):
        ds_path = p
        break

if ds_path is None:
    print(f"Error: {file_name} not found in {paths}.")
    exit(1)

try:
    df = pd.read_csv(ds_path, low_memory=False)
    print(f"Dataset loaded successfully from {ds_path}.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# 2. Filter subsets
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()

print(f"AIID Incidents count: {len(aiid_df)}")
print(f"ATLAS Cases count: {len(atlas_df)}")

# 3. Normalization Logic
def normalize_sector(val):
    if pd.isna(val):
        return 'Other/Unknown'
    val_str = str(val).lower()
    
    # Check keywords mapping to standardized sectors
    if any(x in val_str for x in ['defense', 'military', 'government', 'public sector', 'security', 'police', 'surveillance']):
        return 'Defense/Govt'
    elif any(x in val_str for x in ['health', 'medical', 'hospital', 'biotech']):
        return 'Healthcare'
    elif any(x in val_str for x in ['finance', 'financial', 'bank', 'insurance', 'trading']):
        return 'Finance'
    elif any(x in val_str for x in ['consumer', 'retail', 'entertainment', 'media', 'social media', 'technology', 'internet', 'software', 'app']):
        return 'Consumer/Tech'
    elif any(x in val_str for x in ['transport', 'automotive', 'vehicle', 'aviation', 'driving', 'autonomous']):
        return 'Transportation'
    elif any(x in val_str for x in ['education', 'academic', 'school', 'university']):
        return 'Education'
    else:
        return 'Other/Unknown'

# Apply normalization
# AIID uses 'Sector of Deployment', ATLAS uses 'sector'
aiid_df['norm_sector'] = aiid_df['Sector of Deployment'].apply(normalize_sector)
atlas_df['norm_sector'] = atlas_df['sector'].apply(normalize_sector)

# 4. Calculate Distributions
aiid_counts = aiid_df['norm_sector'].value_counts()
atlas_counts = atlas_df['norm_sector'].value_counts()

# Align indices for comparison
all_sectors = sorted(list(set(aiid_counts.index) | set(atlas_counts.index)))

# Create a DataFrame for the contingency table (Counts)
comparison_df = pd.DataFrame(index=all_sectors)
comparison_df['AIID'] = comparison_df.index.map(aiid_counts).fillna(0).astype(int)
comparison_df['ATLAS'] = comparison_df.index.map(atlas_counts).fillna(0).astype(int)

print("\nSector Distribution (Counts):")
print(comparison_df)

# Calculate Percentages for plotting
plot_df = comparison_df.copy()
plot_df['AIID_pct'] = plot_df['AIID'] / plot_df['AIID'].sum() * 100
plot_df['ATLAS_pct'] = plot_df['ATLAS'] / plot_df['ATLAS'].sum() * 100

print("\nSector Distribution (Percentages):")
print(plot_df[['AIID_pct', 'ATLAS_pct']])

# 5. Statistical Test: Chi-Square Test of Homogeneity
# We check if the distribution of sectors depends on the dataset source.
# Transpose so rows are [AIID, ATLAS] and columns are sectors
chi2, p, dof, expected = chi2_contingency(comparison_df[['AIID', 'ATLAS']].T)

print(f"\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"p-value: {p:.4e}")
print(f"Degrees of Freedom: {dof}")

interpretation = "Significantly different" if p < 0.05 else "Not significantly different"
print(f"Conclusion: The sector distributions are {interpretation}.")

# 6. Visualization
fig, ax = plt.subplots(figsize=(12, 6))

width = 0.35
x_indices = np.arange(len(all_sectors))

# Plot bars
rects1 = ax.bar(x_indices - width/2, plot_df['AIID_pct'], width, label='AIID (Real Incidents)', alpha=0.8)
rects2 = ax.bar(x_indices + width/2, plot_df['ATLAS_pct'], width, label='ATLAS (Adversarial Research)', alpha=0.8)

ax.set_ylabel('Percentage of Cases')
ax.set_title('Sector Distribution: Real-World Incidents vs. Adversarial Research')
ax.set_xticks(x_indices)
ax.set_xticklabels(all_sectors, rotation=45, ha='right')
ax.legend()

# Add text labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment...
Dataset loaded successfully from astalabs_discovery_all_data.csv.
AIID Incidents count: 1362
ATLAS Cases count: 52

Sector Distribution (Counts):
                AIID  ATLAS
Consumer/Tech     54      0
Defense/Govt       2      0
Education         13      0
Finance            6      0
Healthcare        15      0
Other/Unknown   1248     52
Transportation    24      0

Sector Distribution (Percentages):
                 AIID_pct  ATLAS_pct
Consumer/Tech    3.964758        0.0
Defense/Govt     0.146843        0.0
Education        0.954479        0.0
Finance          0.440529        0.0
Healthcare       1.101322        0.0
Other/Unknown   91.629956      100.0
Transportation   1.762115        0.0

Chi-Square Test Results:
Chi2 Statistic: 4.7341
p-value: 5.7834e-01
Degrees of Freedom: 6
Conclusion: The sector distributions are Not significantly different.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Grouped Bar Chart (Clustered Bar Chart).
*   **Purpose:** The plot compares the distribution of cases across different industry sectors for two distinct datasets: "AIID (Real Incidents)" and "ATLAS (Adversarial Research)." It aims to highlight the differences in sector attribution between real-world AI incidents and theoretical adversarial research.

### 2. Axes
*   **X-Axis:**
    *   **Label:** The axis represents different **Sectors**.
    *   **Categories:** Consumer/Tech, Defense/Govt, Education, Finance, Healthcare, Other/Unknown, Transportation.
*   **Y-Axis:**
    *   **Label:** **Percentage of Cases**.
    *   **Value Range:** 0 to 100 (representing 0% to 100%).
    *   **Units:** Percentage (%).

### 3. Data Trends
*   **Dominant Category:** The "Other/Unknown" category dominates the chart significantly for both datasets.
    *   **ATLAS (Orange):** The bar reaches the maximum possible value of **100.0%**.
    *   **AIID (Blue):** The bar is also extremely high, at **91.6%**.
*   **Zero Values for ATLAS:** For every specific sector category (Consumer/Tech, Defense/Govt, Education, Finance, Healthcare, Transportation), the ATLAS dataset (Adversarial Research) shows **0.0%**. This indicates that none of the cases in the ATLAS dataset were attributed to these specific sectors.
*   **Minor Distribution for AIID:** While the vast majority of AIID cases are "Other/Unknown," there is a small distribution across specific sectors:
    *   **Tallest Specific Sector:** Consumer/Tech at **4.0%**.
    *   **Followed by:** Transportation (1.8%), Healthcare (1.1%), Education (1.0%), Finance (0.4%), and Defense/Govt (0.1%).

### 4. Annotations and Legends
*   **Legend:** Located in the top-left corner.
    *   **Blue Square:** Represents "AIID (Real Incidents)".
    *   **Orange Square:** Represents "ATLAS (Adversarial Research)".
*   **Annotations:**
    *   Numerical percentage labels are placed directly above each bar (e.g., "91.6%", "100.0%", "4.0%"). This provides precise data values, removing ambiguity for the viewer.
*   **Title:** "Sector Distribution: Real-World Incidents vs. Adversarial Research" clearly defines the scope of the comparison.

### 5. Statistical Insights
*   **Data Specificity Gap:** There is a stark contrast in data labeling between the two groups. The ATLAS dataset classifies **100%** of its cases as "Other/Unknown," suggesting that adversarial research scenarios are either generic/theoretical and not tied to specific industries, or the dataset lacks sector-specific metadata entirely.
*   **Real-World Variety:** While real-world incidents (AIID) are also heavily skewed toward "Other/Unknown" (91.6%), they demonstrate that incidents *do* occur and are recorded in specific verticals like Consumer Technology and Transportation.
*   **Sector Vulnerability:** Based on the AIID data, the **Consumer/Tech** sector appears to be the most impacted specific industry (4.0%) relative to others like Finance or Defense, though the high percentage of "Unknown" cases suggests that sector attribution remains a major challenge in classifying AI incidents overall.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
