# Experiment 111: node_5_31

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_31` |
| **ID in Run** | 111 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:18:58.083192+00:00 |
| **Runtime** | 543.2s |
| **Parent** | `node_4_12` |
| **Children** | `node_6_48` |
| **Creation Index** | 112 |

---

## Hypothesis

> Sector-Based Harm Specificity: The 'Healthcare' sector is significantly more
likely to experience 'Physical' harms, whereas the 'Financial' sector is
significantly more likely to experience 'Economic' harms.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9758 (Definitely True) |
| **Posterior** | 0.8379 (Likely True) |
| **Surprise** | -0.1655 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | No |

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
| Definitely True | 4.0 |
| Maybe True | 56.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Analyze the relationship between deployment sector and harm type using keyword-based classification on unstructured text fields, addressing previous data quality issues.

### Steps
- 1. Load the `astalabs_discovery_all_data.csv` dataset and filter for rows where `source_table` is 'aiid_incidents'.
- 2. Create a `text_content` column for each row by concatenating relevant string columns available (e.g., `title`, `description`, `summary`, `Alleged harmed or nearly harmed parties`) to ensure sufficient context for classification.
- 3. Define and apply a keyword-based classification function to derive a `Calculated Harm Type`: 
    - 'Physical' if terms like 'death', 'injury', 'kill', 'physical', 'medical', 'hospital', 'patient' are present and dominant.
    - 'Economic' if terms like 'financial', 'money', 'fraud', 'bank', 'loss', 'dollar', 'scam', 'credit' are present and dominant.
    - Exclude cases that are ambiguous (both or neither).
- 4. Clean and categorize the `Sector of Deployment` column into 'Healthcare' and 'Financial' using keyword matching (e.g., 'health', 'medic' vs. 'financ', 'bank', 'insurance').
- 5. Filter the dataset to include only rows with valid assigned Sector and Harm Type.
- 6. Generate a contingency table (Sector vs. Harm Type) and perform a Chi-Square Test of Independence.
- 7. Visualize the results with a stacked bar chart showing the proportion of Physical vs. Economic harms per sector.

### Deliverables
- 1. Contingency table of Sector vs. Calculated Harm Type.
- 2. Chi-Square test results (statistic and p-value).
- 3. Stacked bar chart visualization of harm proportions by sector.

---

## Analysis

The experiment successfully pivoted to a keyword-based text analysis strategy to
overcome previous data quality issues, identifying 15 incidents that matched the
specific Sector and Harm criteria. The analysis found a statistically
significant relationship (Chi-Square p=0.0174) between the sector of deployment
and the realized harm type. Specifically, 100% of the analyzed Financial sector
incidents (n=5) resulted in Economic harm, while 80% of the Healthcare sector
incidents (n=10) resulted in Physical harm. These findings strongly support the
hypothesis that the Healthcare sector is significantly more prone to Physical
harms and the Financial sector to Economic harms, although the relatively small
sample size (n=15) suggests these results should be interpreted as a strong
signal rather than a definitive population-wide conclusion.

---

## Review

The experiment was faithfully implemented and successfully overcame previous
data quality issues by leveraging unstructured text analysis. The code correctly
loaded the dataset, performed robust text processing to derive 'Harm Type' and
'Sector' variables, and executed the planned statistical test. The results
(p=0.0174) support the hypothesis.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np

# [1] Load Data
filename = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(filename, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# [2] Filter Source
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID incidents: {len(aiid)}")

# [3] Robust Text Concatenation
# Identify columns
potential_text_cols = ['title', 'description', 'summary', 'Alleged harmed or nearly harmed parties']
found_text_cols = [c for c in potential_text_cols if c in aiid.columns]
print(f"Using columns for text analysis: {found_text_cols}")

# Fill NaNs with empty string and ensure string type for each column individually
for col in found_text_cols:
    aiid[col] = aiid[col].fillna('').astype(str)

# Concatenate
aiid['full_text'] = aiid[found_text_cols].agg(' '.join, axis=1).str.lower()

# [4] Classification Logic

# Sector Classification
def get_sector(val):
    if not isinstance(val, str):
        return None
    val = val.lower()
    if 'health' in val or 'medic' in val or 'hospital' in val or 'doctor' in val:
        return 'Healthcare'
    if 'financ' in val or 'bank' in val or 'insurance' in val or 'invest' in val or 'trading' in val:
        return 'Financial'
    return None

# Harm Classification (Keyword-based)
def get_harm_type(text):
    if not isinstance(text, str):
        return None
    
    # Keywords for Physical Harm
    physical_keys = [
        'death', 'kill', 'dead', 'injur', 'hurt', 'physical', 'fatal', 
        'accident', 'crash', 'collision', 'burn', 'medical', 'patient', 
        'hospital', 'surgery', 'pain', 'assault', 'hit', 'struck'
    ]
    
    # Keywords for Economic Harm
    economic_keys = [
        'money', 'financ', 'dollar', 'cost', 'fund', 'bank', 'credit', 
        'fraud', 'scam', 'loss', 'market', 'trade', 'economic', 'price', 
        'fee', 'charge', 'wealth', 'asset', 'theft', 'steal', 'embezzle'
    ]
    
    # Count occurrences
    p_count = sum(1 for k in physical_keys if k in text)
    e_count = sum(1 for k in economic_keys if k in text)
    
    if p_count > 0 and e_count == 0:
        return 'Physical'
    if e_count > 0 and p_count == 0:
        return 'Economic'
    if p_count > e_count:
        return 'Physical'
    if e_count > p_count:
        return 'Economic'
    
    # If tied or neither, we can't definitively classify for this binary test
    return None

# Apply Classification
aiid['analyzed_sector'] = aiid['Sector of Deployment'].fillna('').apply(get_sector)
aiid['analyzed_harm'] = aiid['full_text'].apply(get_harm_type)

# [5] Filter Data
analysis_df = aiid.dropna(subset=['analyzed_sector', 'analyzed_harm']).copy()

print(f"\nIncidents after filtering for (Healthcare/Financial) and (Physical/Economic): {len(analysis_df)}")
print("Distribution:")
print(analysis_df.groupby(['analyzed_sector', 'analyzed_harm']).size())

# [6] Statistical Test and Visualization
if len(analysis_df) > 5:
    # Contingency Table
    contingency_table = pd.crosstab(analysis_df['analyzed_sector'], analysis_df['analyzed_harm'])
    print("\nContingency Table:")
    print(contingency_table)
    
    # Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    props = contingency_table.div(contingency_table.sum(axis=1), axis=0)
    ax = props.plot(kind='bar', stacked=True, color=['#FF9999', '#66B2FF'], edgecolor='black')
    
    plt.title('Proportion of Harm Types by Sector (Healthcare vs. Financial)')
    plt.xlabel('Sector')
    plt.ylabel('Proportion')
    plt.legend(title='Harm Type (Derived)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', label_type='center')
        
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data for valid analysis.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total AIID incidents: 1362
Using columns for text analysis: ['title', 'description', 'summary', 'Alleged harmed or nearly harmed parties']

Incidents after filtering for (Healthcare/Financial) and (Physical/Economic): 15
Distribution:
analyzed_sector  analyzed_harm
Financial        Economic         5
Healthcare       Economic         2
                 Physical         8
dtype: int64

Contingency Table:
analyzed_harm    Economic  Physical
analyzed_sector                    
Financial               5         0
Healthcare              2         8

Chi-Square Test Results:
Chi2 Statistic: 5.6585
P-value: 1.7371e-02


=== Plot Analysis (figure 2) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Chart (specifically a 100% stacked bar chart).
*   **Purpose:** The plot is designed to compare the relative proportions of different categories (Harm Types) within specific groups (Sectors). It allows the viewer to see the composition of "Harm" for the Financial sector versus the Healthcare sector.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Sector"
    *   **Categories:** Two distinct categories are displayed: "Financial" and "Healthcare".
*   **Y-Axis:**
    *   **Label:** "Proportion"
    *   **Range:** The axis ranges from **0.0 to 1.0**, representing a probability or percentage (0% to 100%).
    *   **Units:** The values are unitless ratios relative to the whole (1.0).

### 3. Data Trends
*   **Financial Sector:**
    *   This bar is entirely dominated by one category. The "Economic" portion takes up the entire bar height.
    *   **Economic Harm:** 1.00 (or 100%).
    *   **Physical Harm:** 0.00 (or 0%).
*   **Healthcare Sector:**
    *   This bar shows a split distribution between the two harm types, though one is clearly dominant.
    *   **Physical Harm:** This is the dominant category, comprising the upper, larger portion of the bar with a value of **0.80** (80%).
    *   **Economic Harm:** This comprises the smaller, lower portion of the bar with a value of **0.20** (20%).

### 4. Annotations and Legends
*   **Chart Title:** "Proportion of Harm Types by Sector (Healthcare vs. Financial)"
*   **Legend:** Located in the upper right corner, titled "Harm Type (Derived)".
    *   **Pink/Salmon:** Represents "Economic" harm.
    *   **Blue:** Represents "Physical" harm.
*   **Bar Labels:** The specific numerical proportions are annotated directly onto the bars:
    *   **1.00** and **0.00** on the Financial bar.
    *   **0.80** and **0.20** on the Healthcare bar.

### 5. Statistical Insights
*   **Sector-Specific Nature of Harm:** The data suggests a strong correlation between the sector and the type of harm experienced. The Financial sector is depicted as exclusively susceptible to "Economic" harm in this dataset, with zero incidence of "Physical" harm.
*   **Healthcare Complexity:** While the Financial sector is homogenous in its harm type, the Healthcare sector is heterogeneous. While it is primarily associated with "Physical" harm (likely relating to patient health and safety), a significant minority (20%) of the harm is classified as "Economic."
*   **Inverse Relationship:** There is an inverse relationship visible between the sectors; where Economic harm is at its maximum (Financial), Physical harm is non-existent, and where Physical harm is dominant (Healthcare), Economic harm is reduced.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
