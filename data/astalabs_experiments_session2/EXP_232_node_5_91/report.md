# Experiment 232: node_5_91

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_91` |
| **ID in Run** | 232 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T11:51:02.736628+00:00 |
| **Runtime** | 173.7s |
| **Parent** | `node_4_30` |
| **Children** | None |
| **Creation Index** | 233 |

---

## Hypothesis

> The 'Failure Mode Determinism': 'Specification Failures' (misaligned goals) are
significantly more prevalent in 'Social Media' and 'Advertising' sectors,
whereas 'Robustness Failures' (errors/attacks) dominate 'Security' and
'Industrial' sectors.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8145 (Likely True) |
| **Posterior** | 0.6300 (Maybe True) |
| **Surprise** | -0.2214 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 9.0 |
| Maybe True | 21.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 44.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Investigate if specific industries are prone to specific types of technical AI failures.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `aiid_incidents`.
- 2. Clean `Known AI Technical Failure` column and group into 'Specification' (gaming, objective misalignment) vs 'Robustness' (adversarial, distribution shift, error).
- 3. Group `Sector of Deployment` into relevant clusters.
- 4. Perform a Chi-square test between Failure Type and Sector.
- 5. Analyze residuals to identify which specific sector-failure pairs are overrepresented.

### Deliverables
- 1. Stacked bar chart of Failure Types by Sector.
- 2. Chi-square test results.
- 3. List of sector-failure combinations with significant adjusted residuals.

---

## Analysis

The experiment successfully executed the classification and statistical testing
pipeline, but the results were heavily constrained by data sparsity. Out of
1,362 AIID incidents, only 9 data points (<1%) contained both a mappable 'Known
AI Technical Failure' and a relevant 'Sector'. The 'Known AI Technical Failure'
column contained 1,084 null values.

Findings from the limited sample (n=9):
1. **Social/Media/Ad (n=2):** Both recorded failures were 'Robustness' issues.
This contradicts the hypothesis that 'Specification' failures (misalignment)
dominate this sector, although the sample size is negligible.
2. **Transportation (n=1):** Recorded 100% 'Robustness' failures.
3. **Other (n=6):** This category contained the only instances of
'Specification' failures (66% of the group).

Statistical Outcome:
The Chi-Square test yielded a p-value of 0.1653, failing to reject the null
hypothesis. The standardized residuals did not exceed significance thresholds
(+/- 1.96). Consequently, the hypothesis that specific sectors are
deterministically linked to specific failure modes is **not supported** by the
current dataset due to the overwhelming absence of technical failure labels.

---

## Review

The experiment successfully executed the classification and statistical testing
pipeline, but the results were heavily constrained by data sparsity. Out of
1,362 AIID incidents, only 9 data points (<1%) contained both a mappable 'Known
AI Technical Failure' and a relevant 'Sector'. The 'Known AI Technical Failure'
column contained 1,084 null values.

Findings from the limited sample (n=9):
1. **Social/Media/Ad (n=2):** Both recorded failures were 'Robustness' issues.
This contradicts the hypothesis that 'Specification' failures (misalignment)
dominate this sector, although the sample size is negligible.
2. **Transportation (n=1):** Recorded 100% 'Robustness' failures.
3. **Other (n=6):** This category contained the only instances of
'Specification' failures (66% of the group).

Statistical Outcome:
The Chi-Square test yielded a p-value of 0.1653, failing to reject the null
hypothesis. The standardized residuals did not exceed significance thresholds
(+/- 1.96). Consequently, the hypothesis that specific sectors are
deterministically linked to specific failure modes is **not supported** by the
current dataset due to the overwhelming absence of technical failure labels.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import os

# [debug] Print current directory and list files to ensure path is correct
# print(f"Current working directory: {os.getcwd()}")
# print(f"Files in parent directory: {os.listdir('..')}")

# Load dataset
file_path = '../astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = 'astalabs_discovery_all_data.csv'  # Fallback for local testing

try:
    df = pd.read_csv(file_path, low_memory=False)
except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID incidents loaded: {len(aiid)}")

# Define column names based on metadata
col_failure = 'Known AI Technical Failure'
col_sector = 'Sector of Deployment'

# Check for column existence
missing_cols = [c for c in [col_failure, col_sector] if c not in aiid.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}. Available columns sample: {aiid.columns[:10]}")
    sys.exit(1)

# --- Step 1: Clean and Map Failure Types ---
# Inspect unique values to guide mapping (printing top 20)
print("\n--- Top 20 Raw Failure Types ---")
print(aiid[col_failure].value_counts(dropna=False).head(20))

def map_failure_type(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().strip()
    
    # Mapping based on hypothesis definitions
    if 'specification' in val_str:
        return 'Specification'
    elif 'robustness' in val_str or 'adversarial' in val_str:
        return 'Robustness'
    # Some definitions map 'reliability' or 'error' to robustness in broad terms, 
    # but we stick to strict keywords first.
    return 'Other'

aiid['Failure_Class'] = aiid[col_failure].apply(map_failure_type)

# --- Step 2: Clean and Map Sectors ---
# Inspect unique values
print("\n--- Top 20 Raw Sectors ---")
print(aiid[col_sector].value_counts(dropna=False).head(20))

def map_sector_group(val):
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).lower().strip()
    
    # Hypothesis Group 1: Social Media & Advertising
    if any(k in val_str for k in ['social media', 'advertising', 'entertainment', 'news', 'media']):
        return 'Social/Media/Ad'
    
    # Hypothesis Group 2: Security & Industrial
    if any(k in val_str for k in ['security', 'defense', 'industrial', 'manufacturing', 'robotics', 'military', 'surveillance']):
        return 'Security/Industrial'
    
    # Other distinct groups for context
    if any(k in val_str for k in ['transportation', 'automotive', 'vehicle']):
        return 'Transportation'
    if any(k in val_str for k in ['healthcare', 'medicine', 'hospital']):
        return 'Healthcare'
    if any(k in val_str for k in ['financial', 'finance', 'banking']):
        return 'Finance'
        
    return 'Other'

aiid['Sector_Class'] = aiid[col_sector].apply(map_sector_group)

# --- Step 3: Filter Data for Analysis ---
# We focus on rows that have a valid mapped Failure Class (Specification or Robustness)
df_analysis = aiid[aiid['Failure_Class'].isin(['Specification', 'Robustness'])].copy()

# We remove 'Unknown' sectors to clean up the plot, but keep 'Other' for comparison
df_analysis = df_analysis[df_analysis['Sector_Class'] != 'Unknown']

print(f"\nData points remaining for analysis (Specification vs Robustness): {len(df_analysis)}")

if len(df_analysis) < 5:
    print("Insufficient data for statistical analysis.")
    sys.exit(0)

# --- Step 4: Statistical Analysis (Chi-Square) ---
contingency_table = pd.crosstab(df_analysis['Sector_Class'], df_analysis['Failure_Class'])
print("\n--- Contingency Table (Sector vs Failure Type) ---")
print(contingency_table)

chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Degrees of Freedom: {dof}")

# Calculate Standardized Residuals to identify drivers of significance
# Residual = (Observed - Expected) / sqrt(Expected)
std_residuals = (contingency_table - expected) / np.sqrt(expected)
print("\n--- Standardized Residuals (Values > 1.96 or < -1.96 differ significantly) ---")
print(std_residuals)

# --- Step 5: Visualization ---
# Plot proportions
props = contingency_table.div(contingency_table.sum(axis=1), axis=0)

ax = props.plot(kind='barh', stacked=True, figsize=(10, 6), colormap='viridis')
plt.title('Proportion of Failure Types by Sector')
plt.xlabel('Proportion')
plt.ylabel('Sector Group')
plt.legend(title='Failure Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total AIID incidents loaded: 1362

--- Top 20 Raw Failure Types ---
Known AI Technical Failure
NaN                                                            1084
Generalization Failure                                           24
Misinformation Generation Hazard, Unsafe Exposure or Access      17
Distributional Bias                                              13
Context Misidentification                                         5
Lack of Transparency                                              5
Generalization Failure, Context Misidentification                 5
Harmful Application                                               4
Unsafe Exposure or Access                                         4
Unsafe Exposure or Access, Misinformation Generation Hazard       4
Misinformation Generation Hazard                                  4
Latency Issues                                                    3
Algorithmic Bias                                                  3
Context Misidentification, Generalization Failure                 3
Human Error                                                       2
Hardware Failure                                                  2
Generalization Failure, Lack of Safety Protocols                  2
Distributional Bias, Limited Dataset                              2
Algorithmic Bias, Problematic Features                            2
Unauthorized Data                                                 2
Name: count, dtype: int64

--- Top 20 Raw Sectors ---
Sector of Deployment
NaN                                                                  1161
information and communication                                          44
transportation and storage                                             21
Arts, entertainment and recreation, information and communication      14
wholesale and retail trade                                             11
human health and social work activities                                10
Arts, entertainment and recreation                                      9
law enforcement                                                         9
information and communication, Arts, entertainment and recreation       9
Education                                                               7
administrative and support service activities                           5
financial and insurance activities                                      4
public administration                                                   4
professional, scientific and technical activities                       3
accommodation and food service activities                               3
law enforcement, public administration                                  3
manufacturing                                                           3
wholesale and retail trade, information and communication               3
wholesale and retail trade, transportation and storage                  3
transportation and storage, information and communication               2
Name: count, dtype: int64

Data points remaining for analysis (Specification vs Robustness): 9

--- Contingency Table (Sector vs Failure Type) ---
Failure_Class    Robustness  Specification
Sector_Class                              
Other                     2              4
Social/Media/Ad           2              0
Transportation            1              0

Chi-Square Test Results:
Chi2 Statistic: 3.6000
P-value: 1.6530e-01
Degrees of Freedom: 2

--- Standardized Residuals (Values > 1.96 or < -1.96 differ significantly) ---
Failure_Class    Robustness  Specification
Sector_Class                              
Other             -0.730297       0.816497
Social/Media/Ad    0.843274      -0.942809
Transportation     0.596285      -0.666667


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Horizontal Stacked Bar Chart (specifically a 100% stacked bar chart).
*   **Purpose:** To visualize the relative proportions of different "Failure Types" within specific "Sector Groups," allowing for easy comparison of the composition of failures across sectors.

### 2. Axes
*   **Y-axis (Vertical):**
    *   **Label:** "Sector Group"
    *   **Categories:** The axis lists three distinct categorical groups: "Transportation", "Social/Media/Ad", and "Other".
*   **X-axis (Horizontal):**
    *   **Label:** "Proportion"
    *   **Range:** The axis ranges from **0.0 to 1.0**, representing a probability or percentage (0% to 100%).
    *   **Scale:** Linear scale marked at intervals of 0.2.

### 3. Data Trends
*   **Transportation Sector:** The bar for this sector is entirely colored dark purple. This indicates that **100%** of the failures recorded in this sector are attributed to "Robustness." There is no visible yellow segment.
*   **Social/Media/Ad Sector:** Similar to Transportation, this bar is entirely dark purple, indicating that **100%** of the failures are "Robustness" types.
*   **Other Sector:** This is the only category showing variation.
    *   The bar is split between two colors.
    *   Approximately **33-35%** (from 0.0 to roughly 0.33) constitutes "Robustness" failures (dark purple).
    *   The remaining **65-67%** constitutes "Specification" failures (yellow). This is the only sector where "Specification" failures appear in this dataset.

### 4. Annotations and Legends
*   **Chart Title:** "Proportion of Failure Types by Sector" located at the top center.
*   **Legend:** A box located on the top right, titled "Failure Type," decodes the color scheme:
    *   **Dark Purple:** Represents "Robustness".
    *   **Yellow:** Represents "Specification".

### 5. Statistical Insights
*   **dominance of Robustness Failures:** "Robustness" is the most prevalent failure type overall. It is the exclusive cause of failure in the "Transportation" and "Social/Media/Ad" sectors.
*   **Uniqueness of "Other" Category:** The "Other" category is statistically distinct from the specific named sectors. It is the only group affected by "Specification" failures, which actually make up the majority (roughly two-thirds) of its failures.
*   **Sector Specificity:** The data suggests a strong correlation between the specific named sectors (Transportation, Social Media) and robustness issues, whereas the "Other" category likely contains a mix of sectors where specification compliance is a more significant challenge than robustness.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
