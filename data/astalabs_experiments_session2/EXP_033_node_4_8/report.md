# Experiment 33: node_4_8

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_8` |
| **ID in Run** | 33 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T02:37:19.602514+00:00 |
| **Runtime** | 365.9s |
| **Parent** | `node_3_11` |
| **Children** | `node_5_16`, `node_5_47` |
| **Creation Index** | 34 |

---

## Hypothesis

> Agency AI Culture (Build vs. Buy): Defense and Security agencies (e.g., DOD,
DHS) have a significantly higher reliance on 'Custom' AI compared to Civilian
agencies, which rely more on 'Commercial' solutions.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8710 (Likely True) |
| **Posterior** | 0.4615 (Uncertain) |
| **Surprise** | -0.4913 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 18.0 |
| Maybe True | 11.0 |
| Uncertain | 0.0 |
| Maybe False | 1.0 |
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

**Objective:** Analyze procurement patterns (Build vs. Buy) across government sectors using corrected agency classification logic to avoid previous misclassification errors.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for the subset where `source_table` is 'eo13960_scored'.
- 2. Create a specific mapping function for `3_agency` / `3_abr` to 'Defense/Security' vs 'Civilian'. 
   - STRICT RULE: 'Defense/Security' must ONLY include 'Department of Defense' (DOD), 'Department of Homeland Security' (DHS), 'Department of Justice' (DOJ), and 'Department of State' (DOS/STATE).
   - CORRECTION: Ensure 'United States Agency for International Development' (USAID) and other agencies with 'United States' in their name are classified as 'Civilian', not Defense. Do not use broad substring matching for 'STATE'.
- 3. Use the column `37_custom_code` to determine procurement type: Map 'Yes' to 'Custom (Build)' and 'No' to 'Commercial (Buy)'. Filter out rows with null or other values.
- 4. Create a contingency table of Agency Category vs. Procurement Type.
- 5. Perform a Chi-square test of independence.
- 6. Generate a 100% stacked bar chart showing the proportion of Custom vs. Commercial AI for each sector.
- 7. Print the list of unique agencies in the 'Defense/Security' bucket to verify the exclusion of USAID.

### Deliverables
- Contingency table, Chi-square test results, 100% Stacked Bar Chart, and a verification list of Defense agencies.

---

## Analysis

The experiment was successfully re-run with strict agency mapping, effectively
removing the 'United States Agency for International Development' (USAID)
contamination. The 'Defense/Security' category is now comprised of the
Department of Homeland Security and the Department of State (N=77), as the
Department of Defense and Justice appear absent from this specific public
inventory file.

With the corrected data:
1.  **Procurement Patterns:** Both sectors show a strong preference for 'Custom
(Build)' solutions (Defense: 79.2%, Civilian: 74.3%).
2.  **Hypothesis Evaluation:** While the Defense sector shows a slightly higher
reliance on custom solutions (+4.9%), the Chi-square test yields a p-value of
0.4112, indicating the difference is **not statistically significant**.

The hypothesis that Defense agencies rely *significantly* more on custom AI is
not supported by this dataset. The result suggests that the 'Build first'
culture is prevalent across the entire U.S. government, not unique to the
defense sector.

---

## Review

The experiment was successfully executed with the corrected, strict agency
mapping. The exclusion of 'United States Agency for International Development'
from the Defense category resulted in a clean comparison groups (Defense N=77 vs
Civilian N=890). The analysis faithfully tested the hypothesis using the
`37_custom_code` proxy for procurement.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"EO 13960 records loaded: {len(eo_df)}")

# Step 1: Strict Agency Mapping
# Define explicit sets for Defense/Security
defense_abrs = {'DOD', 'DHS', 'DOJ', 'DOS', 'STATE'}
defense_names = {
    'Department of Defense',
    'Department of Homeland Security',
    'Department of Justice',
    'Department of State'
}

def categorize_agency_strict(row):
    abr = str(row.get('3_abr', '')).upper().strip()
    agency = str(row.get('3_agency', '')).strip()
    
    # Check exact abbreviations
    if abr in defense_abrs:
        return 'Defense/Security'
    
    # Check specific agency names (using startswith to catch sub-agencies if formatted like "Department of Defense - Army")
    # But based on previous output, names seem consistent. 
    # We strictly want to avoid "United States..." matching "State"
    for d_name in defense_names:
        if agency == d_name or agency.startswith(d_name + " "):
            return 'Defense/Security'
            
    return 'Civilian'

eo_df['agency_category'] = eo_df.apply(categorize_agency_strict, axis=1)

# Verify Classification
print("\n--- Verification of Defense/Security Agencies ---")
defense_agencies_found = eo_df[eo_df['agency_category'] == 'Defense/Security']['3_agency'].unique()
for ag in defense_agencies_found:
    print(f"  - {ag}")

# Step 2: Determine Procurement Type using '37_custom_code'
def categorize_procurement(val):
    val_str = str(val).lower().strip()
    if val_str == 'yes':
        return 'Custom (Build)'
    elif val_str == 'no':
        return 'Commercial (Buy)'
    else:
        return 'Unknown'

eo_df['procurement_type'] = eo_df['37_custom_code'].apply(categorize_procurement)

# Filter out Unknown procurement types
analysis_df = eo_df[eo_df['procurement_type'] != 'Unknown'].copy()

print(f"\nRecords after filtering for known procurement type: {len(analysis_df)}")
print(analysis_df['agency_category'].value_counts())

# Step 3: Contingency Table
contingency_table = pd.crosstab(analysis_df['agency_category'], analysis_df['procurement_type'])

print("\nContingency Table (Agency Category vs Procurement Type):")
print(contingency_table)

# Step 4: Chi-square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Calculate percentages
props = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
print("\nProportions (%):")
print(props)

# Step 5: Visualization
plt.figure(figsize=(10, 6))
ax = props.plot(kind='bar', stacked=True, color=['#ff7f0e', '#1f77b4'], figsize=(10, 6))

plt.title('AI Procurement Strategy: Defense/Security vs Civilian Agencies (Strict Mapping)')
plt.xlabel('Agency Sector')
plt.ylabel('Percentage of Systems')
plt.legend(title='Procurement Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)

# Add percentage labels
for c in ax.containers:
    ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: EO 13960 records loaded: 1757

--- Verification of Defense/Security Agencies ---
  - Department of Homeland Security
  - Department of State

Records after filtering for known procurement type: 967
agency_category
Civilian            890
Defense/Security     77
Name: count, dtype: int64

Contingency Table (Agency Category vs Procurement Type):
procurement_type  Commercial (Buy)  Custom (Build)
agency_category                                   
Civilian                       229             661
Defense/Security                16              61

Chi-square Statistic: 0.6753
P-value: 4.1122e-01

Proportions (%):
procurement_type  Commercial (Buy)  Custom (Build)
agency_category                                   
Civilian                 25.730337       74.269663
Defense/Security         20.779221       79.220779


=== Plot Analysis (figure 2) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** To compare the proportional composition of AI procurement strategies ("Commercial" vs. "Custom") across two distinct agency sectors ("Civilian" and "Defense/Security"). This chart visualizes the relative percentage share of each procurement type within the total number of systems for each sector.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Agency Sector"
    *   **Labels:** Two distinct categories: "Civilian" and "Defense/Security".
*   **Y-Axis:**
    *   **Title:** "Percentage of Systems"
    *   **Range:** 0 to 100 (representing 0% to 100%).
    *   **Increments:** Ticks are marked every 20 units (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Dominance of Custom Solutions:** In both the Civilian and Defense/Security sectors, the "Custom (Build)" strategy (represented by blue) overwhelmingly dominates the "Commercial (Buy)" strategy.
*   **Comparison of Sectors:**
    *   **Civilian Agencies:** Approximately three-quarters (74.3%) of systems are custom-built, while roughly one-quarter (25.7%) are commercial off-the-shelf purchases.
    *   **Defense/Security Agencies:** Reliance on custom-built systems is even higher here, accounting for nearly 80% (79.2%) of systems, with only about one-fifth (20.8%) being commercial.
*   **Overall Pattern:** There is a consistent trend across government sectors favoring custom development over purchasing commercial AI solutions, though the trend is slightly more pronounced in the defense sector.

### 4. Annotations and Legends
*   **Chart Title:** "AI Procurement Strategy: Defense/Security vs Civilian Agencies (Strict Mapping)".
*   **Legend:** Located at the top right, titled "Procurement Type".
    *   **Orange:** Represents "Commercial (Buy)".
    *   **Blue:** Represents "Custom (Build)".
*   **Data Labels:** White text annotations are placed directly inside the bar segments to indicate specific values:
    *   **Civilian:** 74.3% (Custom) and 25.7% (Commercial).
    *   **Defense/Security:** 79.2% (Custom) and 20.8% (Commercial).

### 5. Statistical Insights
*   **Preference for Bespoke Solutions:** The data indicates a strong governmental preference for developing bespoke AI solutions rather than adopting existing commercial products. This suggests that government requirements for AI are likely specialized, unique, or sensitive enough that off-the-shelf commercial products (COTS) are insufficient for the majority of use cases.
*   **Security Implications:** The Defense/Security sector shows a higher dependency on "Custom (Build)" solutions (79.2% vs 74.3% for Civilian). This 4.9% difference likely reflects the stricter security requirements, classified environments, and highly specialized mission needs inherent to defense operations, which commercial vendors may not accommodate.
*   **Market Opportunity:** While "Commercial (Buy)" is the minority strategy, it still accounts for 20-25% of the systems. This represents a significant, though secondary, market share for commercial AI vendors selling to the government.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
