# Experiment 263: node_6_67

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_67` |
| **ID in Run** | 263 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:32:30.880089+00:00 |
| **Runtime** | 372.6s |
| **Parent** | `node_5_43` |
| **Children** | None |
| **Creation Index** | 264 |

---

## Hypothesis

> The 'Missing Mitigation' Cluster: In the incident coding dataset, 'Evasion'
tactics are associated with a significantly higher count of 'Competency Gaps'
per incident compared to 'Exfiltration' tactics, suggesting current
architectures are less mature against adversarial inputs.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6810 (Maybe True) |
| **Posterior** | 0.2219 (Likely False) |
| **Surprise** | -0.5509 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 21.0 |
| Uncertain | 7.0 |
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

**Objective:** Identify which adversarial tactics expose the most governance gaps.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'step3_incident_coding'.
- 2. Group data by 'type' (Tactic category, e.g., Evasion, Exfiltration).
- 3. Extract the count of gaps from 'competency_gaps' (count semicolon-delimited items or use a count column if available, otherwise count rows per tactic if expanded).
- 4. Compare average gap counts per tactic type using a bar chart or ANOVA if N is sufficient.
- 5. Identify the tactic with the highest mean gaps.

### Deliverables
- Bar chart of Average Competency Gaps by Tactic; Identification of the most 'ungovernable' tactic.

---

## Analysis

The experiment successfully tested the 'Missing Mitigation Cluster' hypothesis.
By correctly mapping Tactic IDs (e.g., AML.TA0000) to names using the
'cross_taxonomy_map' source and analyzing the 'step3_incident_coding' dataset,
the programmer calculated the average number of competency gaps per adversarial
tactic.

**Findings:**
1.  **Hypothesis Rejection:** The hypothesis that 'Evasion' tactics lead to
significantly more gaps than 'Exfiltration' was **refuted**. The analysis showed
that 'Exfiltration' incidents had a mean gap count of **1.0** (standard
deviation 0.0), implying a consistent deficiency in governance for every
recorded incident. In contrast, 'Evasion' had a slightly lower mean of **0.96**.
2.  **Most 'Ungovernable' Tactics:** The analysis identified a cluster of
tactics with a perfect **1.0** average gap count, indicating they are always
associated with a governance failure in this dataset. These include **AI Attack
Staging**, **AI Model Access**, **Collection**, and **Exfiltration**.
3.  **Data Insight:** The low variance and means capping at 1.0 suggest that
incidents in this dataset are typically coded with a single primary competency
gap, rather than a list of multiple gaps, making the 'count' effectively a
binary indicator of 'presence of gap'.

---

## Review

The experiment successfully tested the 'Missing Mitigation Cluster' hypothesis.
By correctly mapping Tactic IDs (e.g., AML.TA0000) to names using the
'cross_taxonomy_map' source and analyzing the 'step3_incident_coding' dataset,
the programmer calculated the average number of competency gaps per adversarial
tactic.

**Findings:**
1.  **Hypothesis Rejection:** The hypothesis that 'Evasion' tactics lead to
significantly more gaps than 'Exfiltration' was **refuted**. The analysis showed
that 'Exfiltration' incidents had a mean gap count of **1.0** (standard
deviation 0.0), implying a consistent deficiency in governance for every
recorded incident. In contrast, 'Evasion' had a slightly lower mean of **0.96**.
2.  **Most 'Ungovernable' Tactics:** The analysis identified a cluster of
tactics with a perfect **1.0** average gap count, indicating they are always
associated with a governance failure in this dataset. These include **AI Attack
Staging**, **AI Model Access**, **Collection**, and **Exfiltration**.
3.  **Data Insight:** The low variance and means capping at 1.0 suggest that
incidents in this dataset are typically coded with a single primary competency
gap, rather than a list of multiple gaps, making the 'count' effectively a
binary indicator of 'presence of gap'.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Build Tactic ID -> Name Mapping
# Try to find mapping in cross_taxonomy_map
map_df = df[df['source_table'] == 'cross_taxonomy_map']

# Extract potential mappings from source_id -> source_label
# Assuming ATLAS IDs are in source_id (e.g., AML.TAxxxx)
tactic_map = {}

# Check source columns
for idx, row in map_df.iterrows():
    if isinstance(row['source_id'], str) and 'AML.TA' in row['source_id'] and isinstance(row['source_label'], str):
        tactic_map[row['source_id']] = row['source_label']
    # Check target columns just in case
    if isinstance(row['target_id'], str) and 'AML.TA' in row['target_id'] and isinstance(row['target_label'], str):
        tactic_map[row['target_id']] = row['target_label']

# If map is empty or small, try to extract from other columns if they exist (fallback)
if not tactic_map:
    # Try to find any column combination in the whole df that links id to name
    # This is a bit expensive so we rely on the map_df first.
    pass

# Hardcoded fallback if mapping is missing (Based on MITRE ATLAS standard if necessary, but prefer data)
# Just in case the data is missing the labels.
if 'AML.TA0005' not in tactic_map:
    # partial fallback for key items mentioned in prompt if not found
    tactic_map.update({
        'AML.TA0000': 'Reconnaissance',
        'AML.TA0001': 'Resource Development',
        'AML.TA0002': 'Initial Access',
        'AML.TA0003': 'ML Model Access',
        'AML.TA0004': 'ML Attack Staging',
        'AML.TA0005': 'Defense Evasion',  # Often just Evasion
        'AML.TA0006': 'Discovery',
        'AML.TA0007': 'Persistence',
        'AML.TA0008': 'Privilege Escalation',
        'AML.TA0009': 'Lateral Movement',
        'AML.TA0010': 'Exfiltration', # Key for hypothesis
        'AML.TA0011': 'Impact',
        'AML.TA0043': 'Reconnaissance', # 2024 updates sometimes change IDs, but stick to data
    })

print(f"Tactic Mapping (First 5): {dict(list(tactic_map.items())[:5])}")

# 3. Process Incident Coding
incidents = df[df['source_table'] == 'step3_incident_coding'].copy()

# Calculate Competency Gaps Count
# Assuming semicolon delimited. Empty string/NaN is 0.
def count_gaps(val):
    if pd.isna(val) or val == '':
        return 0
    return str(val).count(';') + 1

incidents['gap_count'] = incidents['competency_gaps'].apply(count_gaps)

# Process Tactics
# Split tactics_used by '; ' or ';'
incidents['tactics_list'] = incidents['tactics_used'].astype(str).apply(lambda x: [t.strip() for t in x.replace('; ', ';').split(';') if 'AML.TA' in t])

# Explode
exploded = incidents.explode('tactics_list')

# Map Tactic Names
exploded['tactic_name'] = exploded['tactics_list'].map(tactic_map)

# Handle unmapped tactics (use ID if Name not found)
exploded['tactic_name'] = exploded['tactic_name'].fillna(exploded['tactics_list'])

# Consolidate Names (e.g. 'Defense Evasion' -> 'Evasion' to match prompt terminology if needed)
exploded['tactic_group'] = exploded['tactic_name'].apply(lambda x: 'Evasion' if 'Evasion' in str(x) else x)
exploded['tactic_group'] = exploded['tactic_group'].apply(lambda x: 'Exfiltration' if 'Exfiltration' in str(x) else x)

# 4. Analyze
# Group by Tactic Group and calculate mean gap_count
grouped = exploded.groupby('tactic_group')['gap_count'].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False)

print("\nAverage Competency Gaps by Tactic:")
print(grouped)

# Identify 'Evasion' vs 'Exfiltration'
target_tactics = ['Evasion', 'Exfiltration']
subset = grouped[grouped.index.isin(target_tactics)]
print("\nComparison for Hypothesis:")
print(subset)

# 5. Plot
plt.figure(figsize=(10, 6))
# Filter for meaningful groups (N > 2)
plot_data = grouped[grouped['count'] > 2]

plt.bar(plot_data.index, plot_data['mean'], yerr=plot_data['std'].fillna(0), capsize=5)
plt.title('Average Governance Gaps per Adversarial Tactic')
plt.ylabel('Avg. Competency Gaps per Incident')
plt.xlabel('Tactic')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 6. Conclusion
highest_tactic = grouped.index[0]
print(f"\nThe tactic with the highest mean gaps is: {highest_tactic}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Tactic Mapping (First 5): {'AML.TA0000': 'AI Model Access', 'AML.TA0001': 'AI Attack Staging', 'AML.TA0004': 'Initial Access', 'AML.TA0005': 'Execution', 'AML.TA0006': 'Persistence'}

Average Competency Gaps by Tactic:
                          mean  count       std
tactic_group                                   
AI Attack Staging     1.000000     20  0.000000
AI Model Access       1.000000     22  0.000000
Collection            1.000000     10  0.000000
Exfiltration          1.000000     15  0.000000
Persistence           1.000000     11  0.000000
Reconnaissance        1.000000     20  0.000000
Lateral Movement      1.000000      2  0.000000
Initial Access        0.973684     38  0.162221
Impact                0.972222     36  0.166667
Execution             0.961538     26  0.196116
Evasion               0.956522     23  0.208514
Resource Development  0.947368     38  0.226294
Discovery             0.916667     12  0.288675
Credential Access     0.818182     11  0.404520
Privilege Escalation  0.777778      9  0.440959
Command and Control   0.750000      4  0.500000

Comparison for Hypothesis:
                  mean  count       std
tactic_group                           
Exfiltration  1.000000     15  0.000000
Evasion       0.956522     23  0.208514

The tactic with the highest mean gaps is: AI Attack Staging


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is a detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot with Error Bars.
*   **Purpose:** The plot compares the average number of "competency gaps" (deficiencies in governance or defense) across various adversarial tactics. It visualizes both the central tendency (mean) and the variability (likely standard deviation or confidence interval) for each tactic.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Tactic"
    *   **Labels:** Categorical labels representing different stages of an adversarial attack (e.g., "AI Attack Staging," "Initial Access," "Command and Control"). The labels are rotated 45 degrees for readability.
*   **Y-Axis:**
    *   **Title:** "Avg. Competency Gaps per Incident"
    *   **Range:** The axis spans from **0.0 to approximately 1.3**, with tick marks at 0.2 intervals.

### 3. Data Trends
*   **The "Plateau" (Tallest Bars):** The first six tactics on the left—**AI Attack Staging, AI Model Access, Collection, Exfiltration, Persistence,** and **Reconnaissance**—all exhibit the highest possible average value (appearing to be exactly **1.0**).
*   **The Decline:** Moving from left to right, starting with "Initial Access," the average height of the bars begins to decrease slightly.
*   **Shortest Bars:** The tactics with the lowest average gaps are on the far right: **Privilege Escalation** and **Command and Control**, which hover around **0.75 to 0.8**.
*   **Variability Pattern:** There is a distinct trend in the error bars:
    *   For the first six high-value tactics, the error bars are negligible (almost non-existent), indicating very little variation in the data—these gaps are consistently present.
    *   As the average value decreases (starting around "Initial Access"), the error bars become significantly larger. **Command and Control** has the largest spread, ranging roughly from 0.25 to 1.25.

### 4. Annotations and Legends
*   **Title:** "Average Governance Gaps per Adversarial Tactic" is displayed at the top.
*   **Error Bars:** The black lines capping each bar indicate the uncertainty or variability of the data.
*   **No Legend:** There is no separate legend box, as the data is encoded directly into the labeled x-axis categories.

### 5. Statistical Insights
*   **AI and Data Vulnerability:** The tactics specific to AI ("AI Attack Staging," "AI Model Access") and data handling ("Collection," "Exfiltration") show an average of 1.0 with almost zero variance. This statistically suggests a systemic failure or lack of governance in these areas; essentially, *every* recorded incident in this dataset had a competency gap regarding these tactics.
*   **Inconsistency in Traditional Tactics:** Traditional infrastructure tactics like "Command and Control" and "Privilege Escalation" have lower averages but much higher variance. This suggests that defenses against these tactics are inconsistent—some organizations or incidents show strong governance (no gaps), while others show significant gaps.
*   **Maturity Gap:** The data implies that governance frameworks are likely more mature for traditional cyber threats (hence the lower averages and higher variance on the right), but are consistently immature or non-existent for AI-specific threats and data exfiltration (the consistent 1.0s on the left).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
