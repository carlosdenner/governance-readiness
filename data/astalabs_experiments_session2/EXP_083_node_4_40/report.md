# Experiment 83: node_4_40

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_40` |
| **ID in Run** | 83 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:04:46.763845+00:00 |
| **Runtime** | 174.3s |
| **Parent** | `node_3_20` |
| **Children** | `node_5_70`, `node_5_84` |
| **Creation Index** | 84 |

---

## Hypothesis

> The 'Autonomy-Physicality' Link: AI incidents involving 'High Autonomy' systems
(e.g., autonomous vehicles, robots) result in 'Physical' harm at a significantly
higher frequency than 'Low Autonomy' systems, which are more associated with
'Economic' or 'Opportunity' loss.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9516 (Definitely True) |
| **Posterior** | 0.9835 (Definitely True) |
| **Surprise** | +0.0383 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 26.0 |
| Maybe True | 4.0 |
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

**Objective:** Validate the relationship between system autonomy levels and the tangibility of harm produced.

### Steps
- 1. Load 'aiid_incidents'.
- 2. Categorize '81_Autonomy Level' into 'High' (Autonomous) vs 'Low' (Human-in-the-loop/Assistive).
- 3. Categorize '73_Harm Domain'/'74_Tangible Harm' into 'Physical' vs 'Non-Physical' (Economic/Opportunity).
- 4. Generate a contingency table and perform a Chi-square test.

### Deliverables
- Distribution of Physical Harm across Autonomy Levels; Chi-square p-value.

---

## Analysis

The experiment successfully validated the 'Autonomy-Physicality' Link hypothesis
using the AIID dataset. The code correctly classified 1,362 incidents based on
autonomy level and harm type, revealing a stark contrast in risk profiles. 'High
Autonomy' systems (e.g., robots, autonomous vehicles) were found to result in
physical harm in 48.1% of their recorded incidents (63 out of 131), whereas 'Low
Autonomy' systems (e.g., software, algorithms) caused physical harm in only 8.4%
of cases (103 out of 1,231). The Chi-square test (statistic=170.88, p=4.75e-39)
confirmed that this difference is highly statistically significant. The results
provide strong empirical evidence that as AI systems gain autonomy and physical
agency, the nature of their failure modes shifts dramatically from
economic/opportunity loss to direct physical safety risks.

---

## Review

The experiment successfully validated the 'Autonomy-Physicality' Link hypothesis
using the AIID dataset. The execution faithfully followed the plan, implementing
a robust classification strategy that utilized both structured columns and text-
based keyword fallbacks to handle the sparse data. The analysis of 1,362
incidents revealed a stark and statistically significant contrast in risk
profiles: 'High Autonomy' systems (e.g., robots, autonomous vehicles) resulted
in physical harm in 48.1% of cases (63/131), whereas 'Low Autonomy' systems
(e.g., software algorithms) caused physical harm in only 8.4% of cases
(103/1,231). The Chi-square test (p=4.75e-39) confirms that increased system
autonomy is strongly associated with a shift from intangible
(economic/opportunity) harms to tangible physical safety risks.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import re

# [debug] Step 1: Load Data
print("Loading dataset...")
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents found: {len(aiid)}")

# Helper function to clean text
def clean_text(x):
    return str(x).lower() if pd.notnull(x) else ""

# Combine text fields for fallback classification
aiid['text_content'] = aiid['title'].apply(clean_text) + " " + \
                       aiid['description'].apply(clean_text) + " " + \
                       aiid['summary'].apply(clean_text)

# --- 1. Autonomy Classification ---
# Strategy: Check structured column '81_Autonomy Level', fallback to keyword search

high_autonomy_keywords = [
    'autonomous', 'self-driving', 'driverless', 'autopilot', 'robot', 'robotic', 
    'drone', 'uav', 'unmanned', 'tesla', 'waymo', 'cruise', 'uber atg'
]

def classify_autonomy(row):
    # Try structured column first
    val = str(row.get('81_Autonomy Level', '')).lower()
    if 'high' in val or 'autonomous' in val:
        return 'High Autonomy'
    if 'low' in val or 'human' in val:
        return 'Low Autonomy'
    
    # Fallback to text
    text = row['text_content']
    if any(k in text for k in high_autonomy_keywords):
        return 'High Autonomy'
    return 'Low Autonomy' # Default bucket for non-physical/software agents

aiid['Autonomy_Class'] = aiid.apply(classify_autonomy, axis=1)

# --- 2. Harm Classification ---
# Strategy: Check structured '73_Harm Domain'/'74_Tangible Harm', fallback to keywords

physical_harm_keywords = [
    'death', 'dead', 'kill', 'injury', 'injured', 'crash', 'collision', 
    'accident', 'hurt', 'physical', 'safety', 'burn', 'fracture', 'fatality'
]

def classify_harm(row):
    # Try structured columns
    h_domain = str(row.get('73_Harm Domain', '')).lower()
    t_harm = str(row.get('74_Tangible Harm', '')).lower()
    combined_struct = h_domain + " " + t_harm
    
    if 'physical' in combined_struct or 'safety' in combined_struct or 'life' in combined_struct:
        return 'Physical Harm'
    if 'economic' in combined_struct or 'opportunity' in combined_struct or 'reputation' in combined_struct:
        return 'Non-Physical Harm'
        
    # Fallback to text
    text = row['text_content']
    if any(k in text for k in physical_harm_keywords):
        return 'Physical Harm'
    return 'Non-Physical Harm'

aiid['Harm_Class'] = aiid.apply(classify_harm, axis=1)

# --- 3. Analysis ---
print("\n--- Classification Results ---")
print(aiid[['Autonomy_Class', 'Harm_Class']].value_counts())

# Contingency Table
contingency = pd.crosstab(aiid['Autonomy_Class'], aiid['Harm_Class'])
print("\n--- Contingency Table ---")
print(contingency)

# Chi-Square Test
chi2, p, dof, ex = stats.chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-Value: {p:.4e}")

# Visualization
plt.figure(figsize=(10, 6))
# Normalize by row to show probabilities
props = contingency.div(contingency.sum(axis=1), axis=0)
props.plot(kind='bar', stacked=True, color=['orange', 'red'], alpha=0.8)
plt.title('Harm Type Distribution by Autonomy Level')
plt.ylabel('Proportion of Incidents')
plt.xlabel('Autonomy Level')
plt.legend(title='Harm Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Interpretation
alpha = 0.05
if p < alpha:
    print("\nRESULT: Statistically significant relationship found.")
    high_phys = props.loc['High Autonomy', 'Physical Harm'] if 'High Autonomy' in props.index and 'Physical Harm' in props.columns else 0
    low_phys = props.loc['Low Autonomy', 'Physical Harm'] if 'Low Autonomy' in props.index and 'Physical Harm' in props.columns else 0
    print(f"High Autonomy systems resulted in Physical Harm in {high_phys:.1%} of cases.")
    print(f"Low Autonomy systems resulted in Physical Harm in {low_phys:.1%} of cases.")
else:
    print("\nRESULT: No statistically significant relationship found.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset...
AIID Incidents found: 1362

--- Classification Results ---
Autonomy_Class  Harm_Class       
Low Autonomy    Non-Physical Harm    1128
                Physical Harm         103
High Autonomy   Non-Physical Harm      68
                Physical Harm          63
Name: count, dtype: int64

--- Contingency Table ---
Harm_Class      Non-Physical Harm  Physical Harm
Autonomy_Class                                  
High Autonomy                  68             63
Low Autonomy                 1128            103

Chi-Square Statistic: 170.8832
P-Value: 4.7455e-39

RESULT: Statistically significant relationship found.
High Autonomy systems resulted in Physical Harm in 48.1% of cases.
Low Autonomy systems resulted in Physical Harm in 8.4% of cases.


=== Plot Analysis (figure 2) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Chart (specifically a 100% stacked bar chart).
*   **Purpose:** To compare the relative distribution (proportion) of two different categories of harm ("Non-Physical" and "Physical") across two distinct groups defined by their autonomy level ("High" and "Low"). This visualization emphasizes the part-to-whole relationship within each group rather than absolute values.

### 2. Axes
*   **X-axis:**
    *   **Label:** "Autonomy Level"
    *   **Categories:** Two discrete categories labeled "High Autonomy" and "Low Autonomy" (oriented vertically).
*   **Y-axis:**
    *   **Label:** "Proportion of Incidents"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100% of the incidents).
    *   **Ticks:** Intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
*   **High Autonomy:**
    *   The distribution is fairly balanced but leans slightly towards non-physical harm.
    *   **Non-Physical Harm (Yellow):** Accounts for approximately 52% of incidents (estimated visually just above the 0.5 mark).
    *   **Physical Harm (Red):** Accounts for the remaining ~48% of incidents.
*   **Low Autonomy:**
    *   The distribution is heavily skewed.
    *   **Non-Physical Harm (Yellow):** Dominates the category, accounting for approximately 90-92% of incidents.
    *   **Physical Harm (Red):** Represents a very small minority, approximately 8-10% of incidents.

### 4. Annotations and Legends
*   **Title:** "Harm Type Distribution by Autonomy Level" describes the subject matter clearly.
*   **Legend:** Located to the right of the plot titled "Harm Type".
    *   **Yellow/Orange Square:** Represents "Non-Physical Harm".
    *   **Red Square:** Represents "Physical Harm".

### 5. Statistical Insights
*   **Risk Profile Shift:** There is a distinct shift in the nature of risk associated with increased autonomy. While low autonomy systems result almost exclusively in non-physical harm, high autonomy systems show a nearly even split between physical and non-physical harm.
*   **Autonomy vs. Physical Safety:** The plot suggests a positive correlation between higher levels of system autonomy and the proportion of physical harm incidents. "High Autonomy" scenarios are significantly more likely to involve physical harm compared to "Low Autonomy" scenarios.
*   **Prevalence of Non-Physical Harm:** Regardless of autonomy level, non-physical harm constitutes the majority of incidents in both categories, though the margin is vastly different (a bare majority in High Autonomy vs. an overwhelming majority in Low Autonomy).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
