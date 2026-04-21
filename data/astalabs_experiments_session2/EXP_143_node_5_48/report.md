# Experiment 143: node_5_48

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_48` |
| **ID in Run** | 143 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:48:42.815686+00:00 |
| **Runtime** | 438.1s |
| **Parent** | `node_4_19` |
| **Children** | `node_6_75` |
| **Creation Index** | 144 |

---

## Hypothesis

> Sector-Specific Failure Modes: The 'Healthcare' sector is statistically
overrepresented in 'Physical' harm incidents, while the 'Financial' sector is
overrepresented in 'Economic' harm incidents, indicating distinct sector-based
risk profiles.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.5000 (Uncertain) |
| **Surprise** | -0.5806 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

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
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 60.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Map AI failure modes to specific deployment sectors to confirm distinct risk profiles.

### Steps
- 1. Load `aiid_incidents`.
- 2. Filter data to include only 'Healthcare' and 'Financial' sectors (normalize text from `Sector of Deployment`).
- 3. Categorize `Tangible Harm` into 'Physical', 'Economic', and 'Other'.
- 4. Create a contingency table (Sector vs. Harm Category).
- 5. Perform a Chi-Square Test of Independence.

### Deliverables
- Contingency table; Chi-Square results; Stacked bar chart of Harm Categories by Sector.

---

## Analysis

The experiment successfully analyzed the AIID dataset to evaluate sector-
specific failure modes, specifically comparing 'Healthcare' and 'Financial'
sectors against 'Physical' and 'Economic' harm categories. Due to the sparse and
boolean nature of the structured 'Harm Domain' and 'Tangible Harm' columns found
in previous steps, a text-analysis fallback strategy was implemented to classify
harm based on incident titles and descriptions.

1. **Data Processing**: The text-based classification logic successfully
categorized 21 relevant incidents (6 Financial, 15 Healthcare).
2. **Hypothesis Testing**: The Chi-Square Test of Independence yielded a p-value
of 0.3710, which is greater than the alpha of 0.05. Therefore, the null
hypothesis could not be rejected; there is no statistically significant
association between sector and harm category in this limited sample.
3. **Descriptive Trends**:
   - **Economic Harm**: The data directionally supported the hypothesis for the
Financial sector, where 50.0% of incidents involved Economic harm, compared to
only 26.7% in Healthcare.
   - **Physical Harm**: The data **contradicted** the hypothesis for the
Healthcare sector. Physical harm was identified in only 6.7% of Healthcare
incidents, whereas it appeared in 16.7% of Financial incidents (though the
counts are very low: 1 vs 1).
   - **Other Harm**: Healthcare incidents were dominated by 'Other' harm types
(66.7%), potentially indicating privacy violations or service denials rather
than direct physical injury.

The experiment execution is complete, with the caveat that the small sample size
(n=21) limits the statistical power of the findings.

---

## Review

The experiment successfully analyzed the relationship between deployment sectors
and AI failure modes, despite significant data limitations that required a text-
mining fallback strategy.

**Hypothesis Test Results:**
The analysis **failed to support** the hypothesis that the Healthcare sector is
statistically overrepresented in 'Physical' harm incidents compared to the
Financial sector.

1.  **Statistical Significance:** The Chi-Square Test of Independence yielded a
p-value of **0.3710**, which is well above the significance level of 0.05.
Consequently, we fail to reject the null hypothesis; the observed differences in
harm distribution are not statistically significant in this sample.

2.  **Sample Limitations:** The rigorous filtering for specific sectors resulted
in a small sample size (n=21: 15 Healthcare, 6 Financial), which severely
limited the statistical power of the test.

3.  **Directional Findings:**
    *   **Financial Sector:** The data directionally supported the hypothesis
for this sector, with **50.0%** of incidents involving Economic harm (compared
to 26.7% in Healthcare).
    *   **Healthcare Sector:** The data **contradicted** the hypothesis.
'Physical' harm was identified in only **6.7%** of Healthcare incidents.
Instead, the sector was dominated by 'Other' harm types (**66.7%**), suggesting
that reported AI failures in healthcare may predominantly involve data privacy,
service denial, or intangible harms rather than direct physical injury.

**Methodological Note:**
Initial attempts to use structured columns (`Tangible Harm`, `Harm Domain`)
revealed they contained boolean/status indicators rather than harm categories.
The successful implementation of a keyword-based text analysis on incident
descriptions was a necessary and effective adaptation to classify the harm
types.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

# Define dataset path
dataset_path = 'astalabs_discovery_all_data.csv'

print(f"Loading dataset from {dataset_path}...")
try:
    df = pd.read_csv(dataset_path, low_memory=False)
except Exception as e:
    print(f"Error loading CSV: {e}")
    sys.exit(1)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID incidents loaded: {len(aiid_df)}")

# 1. Normalize and Filter Sectors
sector_col = 'Sector of Deployment'

def map_sector(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower()
    if any(x in val_str for x in ['health', 'medical', 'hospital', 'clinic', 'doctor', 'patient']):
        return 'Healthcare'
    elif any(x in val_str for x in ['financ', 'bank', 'trading', 'insurance', 'credit', 'loan', 'money']):
        return 'Financial'
    else:
        return 'Other'

aiid_df['normalized_sector'] = aiid_df[sector_col].apply(map_sector)

# 2. Categorize Harm using Text Analysis (Fallback strategy since structured columns were uninformative)
# We combine title and description for a richer context
aiid_df['text_content'] = aiid_df['title'].fillna('') + ' ' + aiid_df['description'].fillna('')

def map_harm_text(text):
    text = str(text).lower()
    
    # Keywords for Physical Harm (Safety, Life, Health)
    physical_keywords = ['death', 'dead', 'kill', 'injur', 'hurt', 'physical', 'safety', 'bodily', 
                         'crash', 'accident', 'burn', 'poison', 'patient harm']
    
    # Keywords for Economic Harm (Financial loss, Fraud, etc.)
    economic_keywords = ['financial', 'money', 'dollar', 'economic', 'loss', 'credit', 'bank', 
                         'fraud', 'scam', 'theft', 'fund', 'wallet', 'crypto', 'payment', 'charge']
    
    has_physical = any(k in text for k in physical_keywords)
    has_economic = any(k in text for k in economic_keywords)
    
    if has_physical and not has_economic:
        return 'Physical'
    elif has_economic and not has_physical:
        return 'Economic'
    elif has_physical and has_economic:
        # Conflict resolution: usually physical takes precedence in severity, 
        # but for this study let's call it 'Mixed/Physical'
        return 'Physical'
    else:
        return 'Other'

aiid_df['harm_category'] = aiid_df['text_content'].apply(map_harm_text)

# Filter only for the target sectors
study_df = aiid_df[aiid_df['normalized_sector'].isin(['Healthcare', 'Financial'])].copy()
print(f"Incidents in target sectors (Healthcare/Financial): {len(study_df)}")

# 3. Create Contingency Table
contingency_table = pd.crosstab(study_df['normalized_sector'], study_df['harm_category'])

# Ensure columns exist
for col in ['Physical', 'Economic', 'Other']:
    if col not in contingency_table.columns:
        contingency_table[col] = 0

# Reorder
contingency_table = contingency_table[['Physical', 'Economic', 'Other']]

print("\n--- Contingency Table (Sector vs Harm) ---")
print(contingency_table)

# 4. Perform Chi-Square Test
# We check if we have enough data
if len(study_df) < 5:
    print("\nWarning: Sample size too small for reliable Chi-Square test.")
else:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    print(f"Degrees of Freedom: {dof}")

    alpha = 0.05
    if p < alpha:
        print("Result: Statistically Significant (Reject Null Hypothesis)")
    else:
        print("Result: Not Statistically Significant (Fail to Reject Null Hypothesis)")

# 5. Calculate Row Percentages
row_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
print("\n--- Row Percentages ---")
print(row_pct.round(2))

# 6. Visualize
try:
    plt.figure(figsize=(10, 6))
    ax = row_pct.plot(kind='bar', stacked=True, colormap='RdYlBu', figsize=(10, 6))
    plt.title('Distribution of Harm Types by Sector (Text-Inferred)')
    plt.xlabel('Sector')
    plt.ylabel('Percentage of Incidents')
    plt.legend(title='Harm Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Add labels if possible
    for c in ax.containers:
        # Only label if segment is big enough
        labels = [f'{v.get_height():.1f}%' if v.get_height() > 5 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center')
        
    plt.show()
except Exception as e:
    print(f"Plotting error: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
Total AIID incidents loaded: 1362
Incidents in target sectors (Healthcare/Financial): 21

--- Contingency Table (Sector vs Harm) ---
harm_category      Physical  Economic  Other
normalized_sector                           
Financial                 1         3      2
Healthcare                1         4     10

Chi-Square Statistic: 1.9833
P-value: 3.7096e-01
Degrees of Freedom: 2
Result: Not Statistically Significant (Fail to Reject Null Hypothesis)

--- Row Percentages ---
harm_category      Physical  Economic  Other
normalized_sector                           
Financial             16.67     50.00  33.33
Healthcare             6.67     26.67  66.67


=== Plot Analysis (figure 2) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** This chart is used to compare the relative distribution (percentage composition) of different "Harm Categories" within two distinct "Sectors" (Financial and Healthcare). By normalizing the height of the bars to 100%, it focuses on the proportion of each harm type rather than the absolute number of incidents.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Sector"
    *   **Categories:** The axis represents categorical data with two groups: "Financial" and "Healthcare".
*   **Y-Axis:**
    *   **Label:** "Percentage of Incidents"
    *   **Value Range:** 0 to 100.
    *   **Units:** Percent (%).

### 3. Data Trends
*   **Financial Sector:**
    *   **Dominant Category:** "Economic" harm is the most prevalent type, accounting for half of the incidents (**50.0%**).
    *   **Secondary Category:** "Other" harm types account for one-third of incidents (**33.3%**).
    *   **Minor Category:** "Physical" harm is the least common, representing **16.7%** of incidents.
*   **Healthcare Sector:**
    *   **Dominant Category:** "Other" harm types are overwhelmingly dominant, constituting two-thirds of the incidents (**66.7%**).
    *   **Secondary Category:** "Economic" harm accounts for **26.7%**.
    *   **Minor Category:** "Physical" harm is very low, at only **6.7%**.

### 4. Annotations and Legends
*   **Chart Title:** "Distribution of Harm Types by Sector (Text-Inferred)" indicates the dataset is likely derived from natural language processing or text analysis of incident reports.
*   **Legend:** Located on the right side, titled "Harm Category," mapping colors to specific categories:
    *   **Dark Red/Burgundy:** Physical
    *   **Pale Yellow:** Economic
    *   **Dark Blue:** Other
*   **Data Labels:** Percentage values are explicitly annotated within each segment of the bars, providing exact figures for easy reading (e.g., 50.0%, 33.3%).

### 5. Statistical Insights
*   **Sector-Specific Risks:** There is a distinct difference in the profile of harm between the two sectors. The **Financial sector** is characterized heavily by **Economic harm** (50.0%), which aligns with the nature of the industry. Conversely, the **Healthcare sector** is characterized predominantly by **"Other" types of harm** (66.7%), suggesting the risks there fall outside traditional physical or economic definitions (possibly data privacy, psychological, or service denial).
*   **Physical Harm Prevalence:** While "Physical" harm is the minority category in both sectors, it is notably **more than twice as prevalent** in the Financial sector (16.7%) compared to the Healthcare sector (6.7%) according to this specific dataset.
*   **Inverse Relationship:** There appears to be an inverse relationship between the "Economic" and "Other" categories across the two sectors; as the share of Economic harm decreases (from Financial to Healthcare), the share of "Other" harm increases significantly.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
