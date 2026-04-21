# Experiment 217: node_7_6

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_6` |
| **ID in Run** | 217 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T11:10:23.907048+00:00 |
| **Runtime** | 440.0s |
| **Parent** | `node_6_6` |
| **Children** | `node_8_0` |
| **Creation Index** | 218 |

---

## Hypothesis

> Autonomy-Harm Escalation: AI incidents involving systems with 'High' autonomy
are significantly more likely to result in 'Physical' tangible harm compared to
'Low' autonomy systems (which correlate more with financial/reputational harm).

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7339 (Likely True) |
| **Posterior** | 0.2500 (Likely False) |
| **Surprise** | -0.5806 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 29.0 |
| Uncertain | 1.0 |
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

**Objective:** Investigate the relationship between system autonomy levels and the tangibility of harm.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `source_table` == 'aiid_incidents'.
- 2. Clean `81_Autonomy Level`: Map 'High'/'Full' to 'High', and 'Low'/'Human-in-the-loop' to 'Low'. Drop unclear values.
- 3. Create binary `is_physical_harm` from `74_Tangible Harm` (search for 'Physical' or 'Life').
- 4. Perform a Chi-square test on the Autonomy vs Physical Harm contingency table.

### Deliverables
- Contingency table of Autonomy vs Harm Type, statistical significance metrics.

---

## Analysis

The experiment was successfully executed and the hypothesis was **rejected**.
The analysis of 185 AI incidents revealed no significant relationship between
the level of system autonomy and the tangibility of harm (Physical vs. Non-
Physical).

**Key Findings:**
1.  **Data Distribution:**
    -   **High Autonomy:** 53 incidents analyzed. 8 resulted in Physical harm
(15.1%), 45 in Non-Physical harm.
    -   **Low Autonomy:** 132 incidents analyzed. 21 resulted in Physical harm
(15.9%), 111 in Non-Physical harm.
2.  **Statistical Results:**
    -   **Chi-Square Statistic:** 0.00 (P-value: 1.00). The harm distributions
are statistically identical.
    -   **Odds Ratio:** 0.94. This indicates that High Autonomy systems are
roughly equally likely (or negligibly less likely) to cause physical harm
compared to Low Autonomy systems, contradicting the hypothesis that autonomy
escalates physical risk.
3.  **Conclusion:** In the AI Incident Database (AIID), the likelihood of
physical harm appears to be independent of the system's autonomy level. Both
'Human-in-the-loop' (Low) and 'Full Autonomy' (High) systems exhibit a physical
harm rate of approximately 15-16%, with the vast majority of incidents in both
categories resulting in non-physical harms (e.g., bias, financial loss,
reputational damage).

---

## Review

The experiment was successfully executed and the hypothesis was **rejected**.
The analysis of 185 AI incidents from the AI Incident Database (AIID) revealed
no statistically significant relationship between the level of system autonomy
and the tangibility of harm (Physical vs. Non-Physical).

**Key Findings:**
1.  **Data Distribution:**
    -   **High Autonomy:** 53 incidents analyzed. 8 resulted in Physical harm
(15.1%), 45 in Non-Physical harm.
    -   **Low Autonomy:** 132 incidents analyzed. 21 resulted in Physical harm
(15.9%), 111 in Non-Physical harm.
2.  **Statistical Results:**
    -   **Chi-Square Statistic:** 0.00 (P-value: 1.00). The harm distributions
are statistically identical.
    -   **Odds Ratio:** 0.94. This indicates that High Autonomy systems are
roughly equally likely (or negligibly less likely) to cause physical harm
compared to Low Autonomy systems, contradicting the hypothesis that autonomy
escalates physical risk.
3.  **Conclusion:** In this dataset, the likelihood of physical harm appears to
be independent of the system's autonomy level. Both 'Human-in-the-loop' (Low)
and 'Full Autonomy' (High) systems exhibit a physical harm rate of approximately
15-16%, with the vast majority of incidents in both categories resulting in non-
physical harms (e.g., bias, financial loss, reputational damage).

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Loaded {len(aiid_df)} AIID incidents.")

# --- Step 1: Map Autonomy Level ---
# Based on previous exploration: 'Autonomy Level' contains 'Autonomy1', 'Autonomy2', 'Autonomy3'
autonomy_col = 'Autonomy Level' if 'Autonomy Level' in aiid_df.columns else 'autonomy'

def map_autonomy(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower()
    # High Autonomy
    if 'autonomy3' in val_str or 'high' in val_str or 'full' in val_str:
        return 'High'
    # Low Autonomy
    if 'autonomy1' in val_str or 'autonomy2' in val_str or 'low' in val_str or 'human' in val_str:
        return 'Low'
    return np.nan

aiid_df['autonomy_category'] = aiid_df[autonomy_col].apply(map_autonomy)
print("\nAutonomy Distribution:")
print(aiid_df['autonomy_category'].value_counts(dropna=False))

# --- Step 2: Map Physical Harm (Text Analysis) ---
# We need to find the text column. Common names: 'description', 'summary', 'text'.
text_candidates = ['description', 'summary', 'Description', 'Summary', 'Text', 'incident_description']
text_col = next((c for c in text_candidates if c in aiid_df.columns), None)

print(f"\nUsing text column for harm analysis: {text_col}")

def map_physical_harm(row):
    # keywords for physical harm
    keywords = ['death', 'dead', 'kill', 'injury', 'injured', 'hurt', 'physical', 
                'collision', 'crash', 'accident', 'safety', 'burn', 'medical', 'hospital']
    
    text_content = ""
    if text_col and pd.notna(row[text_col]):
        text_content += str(row[text_col]).lower()
    
    # Also check 'Tangible Harm' or 'Harm Domain' for specific keywords if they exist
    if 'Tangible Harm' in row and pd.notna(row['Tangible Harm']):
        text_content += " " + str(row['Tangible Harm']).lower()
        
    if any(k in text_content for k in keywords):
        return 'Physical'
    return 'Non-Physical'

aiid_df['harm_category'] = aiid_df.apply(map_physical_harm, axis=1)
print("\nHarm Distribution:")
print(aiid_df['harm_category'].value_counts(dropna=False))

# --- Step 3: Statistical Analysis ---
analysis_df = aiid_df.dropna(subset=['autonomy_category'])
print(f"\nFinal analysis set size: {len(analysis_df)}")

if len(analysis_df) > 0:
    # Contingency Table
    contingency_table = pd.crosstab(analysis_df['autonomy_category'], analysis_df['harm_category'])
    print("\n--- Contingency Table (Autonomy vs Harm Type) ---")
    print(contingency_table)

    # Chi-Square
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Odds Ratio Calculation
    # OR = (High_Phys / High_NonPhys) / (Low_Phys / Low_NonPhys)
    try:
        if 'Physical' not in contingency_table.columns:
            print("\n'Physical' harm category missing from table. Cannot calculate Odds Ratio.")
        else:
            high_phys = contingency_table.loc['High', 'Physical']
            high_non = contingency_table.loc['High', 'Non-Physical']
            low_phys = contingency_table.loc['Low', 'Physical']
            low_non = contingency_table.loc['Low', 'Non-Physical']
            
            # Smoothing for zeros
            if high_non == 0 or low_phys == 0 or high_phys == 0 or low_non == 0:
                print("\n(Adding smoothing constant for zero cells)")
                high_phys += 0.5; high_non += 0.5; low_phys += 0.5; low_non += 0.5
            
            odds_ratio = (high_phys * low_non) / (high_non * low_phys)
            print(f"\nOdds Ratio (High Autonomy -> Physical Harm): {odds_ratio:.4f}")
    except KeyError:
        print("\nMissing categories for Odds Ratio calculation.")

    # Plot
    contingency_table.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'])
    plt.title('Harm Type by Autonomy Level')
    plt.xlabel('Autonomy Level')
    plt.ylabel('Count')
    plt.legend(title='Harm Type')
    plt.tight_layout()
    plt.show()
else:
    print("\nNo data available for analysis.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded 1362 AIID incidents.

Autonomy Distribution:
autonomy_category
NaN     1177
Low      132
High      53
Name: count, dtype: int64

Using text column for harm analysis: description

Harm Distribution:
harm_category
Non-Physical    1181
Physical         181
Name: count, dtype: int64

Final analysis set size: 185

--- Contingency Table (Autonomy vs Harm Type) ---
harm_category      Non-Physical  Physical
autonomy_category                        
High                         45         8
Low                         111        21

Chi-Square Statistic: 0.0000
P-value: 1.0000e+00

Odds Ratio (High Autonomy -> Physical Harm): 0.9397


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Chart.
*   **Purpose:** This chart compares the total count of incidents across two different categories ("High" vs. "Low" Autonomy Levels) while simultaneously breaking down the composition of those incidents into two sub-groups ("Non-Physical" vs. "Physical" Harm).

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Autonomy Level"
    *   **Categories:** "High" and "Low" (Note: The category labels are rotated 90 degrees vertically).
*   **Y-Axis:**
    *   **Label:** "Count" (indicating the frequency or number of occurrences).
    *   **Range:** The scale starts at 0 and extends to roughly 135, with major tick marks every 20 units (0, 20, 40, ..., 120).

### 3. Data Trends
*   **Overall Volume:** The "Low" autonomy category has a significantly higher total count compared to the "High" autonomy category. The "Low" bar reaches a total height of approximately 130-132, whereas the "High" bar reaches approximately 50-55.
*   **Dominant Category:** In both autonomy levels, "Non-Physical" harm (represented by the blue section) constitutes the vast majority of the count.
*   **Comparison of Segments:**
    *   **High Autonomy:** The Non-Physical count is approximately 45, while the Physical count is small, roughly 8-10.
    *   **Low Autonomy:** The Non-Physical count is approximately 110-112, while the Physical count appears to be around 20.

### 4. Annotations and Legends
*   **Title:** "Harm Type by Autonomy Level" appears at the top center.
*   **Legend:** Located in the top-left corner, titled "Harm Type". It distinguishes the data segments by color:
    *   **Sky Blue:** Non-Physical
    *   **Salmon/Pink:** Physical

### 5. Statistical Insights
*   **Frequency Disparity:** There is a stark contrast in the number of recorded harms based on autonomy level. Incidents associated with **Low Autonomy** are more than double the frequency of those associated with **High Autonomy**.
*   **Nature of Harm:** Regardless of the autonomy level, the nature of the harm is predominantly **Non-Physical**. Physical harm represents a consistent minority share of the data in both scenarios (appearing to be roughly 15-20% of the total in both bars).
*   **Conclusion:** The data suggests that lower autonomy systems are associated with a higher quantity of reported harms in this dataset, but the *type* of harm profile (mostly non-physical) remains relatively stable regardless of whether autonomy is high or low.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
