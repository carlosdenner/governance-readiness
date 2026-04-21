# Experiment 52: node_4_19

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_19` |
| **ID in Run** | 52 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:27:14.424309+00:00 |
| **Runtime** | 456.8s |
| **Parent** | `node_3_2` |
| **Children** | `node_5_48`, `node_5_98` |
| **Creation Index** | 53 |

---

## Hypothesis

> The 'Kill Chain' Gateway: In adversarial AI attacks, the presence of 'Evasion'
tactics is significantly associated with the successful execution of
'Exfiltration' tactics, acting as a technical prerequisite.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.4355 (Maybe False) |
| **Posterior** | 0.3132 (Maybe False) |
| **Surprise** | -0.1468 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 10.0 |
| Uncertain | 2.0 |
| Maybe False | 18.0 |
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

**Objective:** Identify correlations between specific adversarial tactics in the ATLAS dataset.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` (ATLAS subset).
- 2. Parse the `tactics` column (list of strings).
- 3. Create binary columns for 'Has_Evasion' and 'Has_Exfiltration'.
- 4. Perform a Fisher's Exact Test (due to small sample size) to test the association.

### Deliverables
- Heatmap of Tactic Co-occurrence; Fisher's Exact Test p-value and Odds Ratio.

---

## Analysis

The experiment successfully analyzed the ATLAS dataset (n=52) after correcting
the data parsing logic to handle pipe-delimited tactic strings.

1.  **Hypothesis Test**: The analysis **failed to support** the hypothesis that
'Defense Evasion' is significantly associated with 'Exfiltration'. The Fisher's
Exact Test yielded a p-value of 0.3688 and an Odds Ratio of 0.53, indicating no
statistically significant correlation between these two tactics in the current
dataset.

2.  **Tactic Co-occurrence Heatmap**: The generated heatmap provides valuable
insights into the broader threat landscape:
    *   **Dominant Kill Chain**: 'Initial Access', 'Resource Development', and
'Impact' appear as the most frequent and strongly co-occurring tactics,
suggesting a common attack pattern where adversaries gain access and immediately
pivot to impact or resource usage.
    *   **ML-Specific Patterns**: A distinct cluster exists involving 'ML Model
Access', 'ML Attack Staging', and 'Defense Evasion', highlighting specialized
behaviors in AI-targeted attacks.
    *   **Rare Tactics**: 'Lateral Movement' and 'Command and Control' are
underrepresented, implying that recorded AI incidents often bypass traditional
network pivots.

The code execution was successful and the deliverables (statistical test and
visualization) are complete.

---

## Review

The experiment was successfully executed and the data parsing issues from
previous attempts were resolved. The robust parser correctly handled the pipe-
delimited tactic strings, allowing for a valid analysis of the ATLAS dataset
(n=52).

**Hypothesis Test Results:**
The analysis **failed to support** the hypothesis that 'Defense Evasion' is a
prerequisite for or significantly associated with 'Exfiltration'.
- **Statistical Test:** Fisher's Exact Test yielded a p-value of **0.3688** (>
0.05) and an Odds Ratio of **0.53**.
- **Interpretation:** There is no statistically significant correlation between
the presence of evasion tactics and exfiltration tactics in this dataset. In
fact, Exfiltration occurred in 10 cases *without* Evasion, compared to 5 cases
*with* Evasion.

**Additional Findings (from Heatmap):**
- **Dominant Attack Pattern:** The heatmap reveals a strong cluster involving
'Initial Access', 'Resource Development', and 'Impact', suggesting that many AI
incidents in the database involve adversaries gaining access and immediately
leveraging resources or causing impact, often skipping intermediate lateral
movement.
- **AI-Specific Cluster:** A distinct sub-pattern connects 'ML Model Access',
'ML Attack Staging', and 'Defense Evasion', indicating specialized tradecraft
for attacking model components.
- **Outliers:** Traditional network tactics like 'Lateral Movement' and 'Command
and Control' are rare, highlighting the unique nature of the AI threat landscape
compared to enterprise IT attacks.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import scipy.stats as stats
import re
import os
import numpy as np

# [debug]
print("Starting experiment: ATLAS Kill Chain Analysis (Attempt 4 - Fixed Parser)")

# Load dataset
filename = 'astalabs_discovery_all_data.csv'
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    raise FileNotFoundError(f"Could not find {filename}")

df = pd.read_csv(filepath, low_memory=False)

# Filter for ATLAS cases
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
print(f"ATLAS cases loaded: {len(atlas_df)}")

# robust parser for pipe-delimited or list-like strings
def parse_and_clean_tactics(tactic_str):
    if pd.isna(tactic_str):
        return []
    
    tactic_str = str(tactic_str).strip()
    
    # List of raw tactic tokens
    tokens = []
    
    # Check for pipe delimiter first (as seen in debug output)
    if '|' in tactic_str:
        tokens = tactic_str.split('|')
    # Check for list literal format
    elif tactic_str.startswith('[') and tactic_str.endswith(']'):
        try:
            parsed = ast.literal_eval(tactic_str)
            if isinstance(parsed, list):
                tokens = [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            # Fallback for malformed lists
            tokens = tactic_str.replace('[', '').replace(']', '').replace("'", '').replace('"', '').split(',')
    else:
        # Assume comma separated
        tokens = tactic_str.split(',')
        
    # Clean tokens
    cleaned_tokens = []
    for token in tokens:
        token = token.strip()
        if not token: continue
        
        # Regex to extract name from {{name.id}}
        match = re.search(r'\{\{([a-zA-Z0-9_]+)\.id\}\}', token)
        if match:
            name = match.group(1)
            # Normalize: underscores to spaces, title case
            # Handle specific mapping if needed, but Title Case is usually sufficient
            name = name.replace('_', ' ').title()
            cleaned_tokens.append(name)
        else:
            # If no curly braces, just clean the string
            # Remove potential .id suffix if it exists plainly
            name = token.replace('.id', '').replace('_', ' ').title()
            cleaned_tokens.append(name)
            
    return sorted(list(set(cleaned_tokens))) # Return unique tactics for this row

# Apply parsing
atlas_df['tactics_clean'] = atlas_df['tactics'].apply(parse_and_clean_tactics)

# Verify parsing
all_clean_tactics = set()
for tactics in atlas_df['tactics_clean']:
    all_clean_tactics.update(tactics)
all_clean_tactics = sorted(list(all_clean_tactics))

print(f"Unique tactics identified: {len(all_clean_tactics)}")
print(f"Tactics list: {all_clean_tactics}")

# Define target variables
# Hypothesis: 'Defense Evasion' (Evasion) associated with 'Exfiltration'
atlas_df['has_evasion'] = atlas_df['tactics_clean'].apply(lambda x: 'Defense Evasion' in x)
atlas_df['has_exfiltration'] = atlas_df['tactics_clean'].apply(lambda x: 'Exfiltration' in x)

# Contingency Table
contingency_table = pd.crosstab(atlas_df['has_evasion'], atlas_df['has_exfiltration'])
print("\nContingency Table (Defense Evasion vs Exfiltration):")
print(contingency_table)

# Ensure 2x2 for Fisher's test (handling missing columns/rows)
# Expected columns: False, True. Expected index: False, True.
full_contingency = pd.DataFrame(0, index=[False, True], columns=[False, True])
for i in [False, True]:
    for c in [False, True]:
        if i in contingency_table.index and c in contingency_table.columns:
            full_contingency.loc[i, c] = contingency_table.loc[i, c]

print("\nFull Contingency Table:")
print(full_contingency)

# Fisher's Exact Test
odds_ratio, p_value = stats.fisher_exact(full_contingency)
print(f"\nFisher's Exact Test Results:")
print(f"Odds Ratio: {odds_ratio:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Conclusion: Significant association found (Reject H0).")
else:
    print("Conclusion: No significant association found (Fail to reject H0).")

# Co-occurrence Heatmap
if len(all_clean_tactics) > 0:
    co_occurrence = pd.DataFrame(0, index=all_clean_tactics, columns=all_clean_tactics)

    for tactics in atlas_df['tactics_clean']:
        # Permutation: count co-occurrence for every pair in the list
        for t1 in tactics:
            for t2 in tactics:
                co_occurrence.loc[t1, t2] += 1

    plt.figure(figsize=(12, 10))
    # Mask diagonal to emphasize co-occurrences? No, standard heatmap.
    sns.heatmap(co_occurrence, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Adversarial Tactic Co-occurrence Heatmap (ATLAS) - Final')
    plt.tight_layout()
    plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment: ATLAS Kill Chain Analysis (Attempt 4 - Fixed Parser)
ATLAS cases loaded: 52
Unique tactics identified: 16
Tactics list: ['Collection', 'Command And Control', 'Credential Access', 'Defense Evasion', 'Discovery', 'Execution', 'Exfiltration', 'Impact', 'Initial Access', 'Lateral Movement', 'Ml Attack Staging', 'Ml Model Access', 'Persistence', 'Privilege Escalation', 'Reconnaissance', 'Resource Development']

Contingency Table (Defense Evasion vs Exfiltration):
has_exfiltration  False  True 
has_evasion                   
False                19     10
True                 18      5

Full Contingency Table:
       False  True 
False     19     10
True      18      5

Fisher's Exact Test Results:
Odds Ratio: 0.5278
P-value: 0.3688
Conclusion: No significant association found (Fail to reject H0).


=== Plot Analysis (figure 1) ===
Based on the image provided, here is the detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Co-occurrence Heatmap (Symmetric Matrix).
*   **Purpose:** This plot visualizes the frequency with which different adversarial tactics (from the MITRE ATLAS framework) occur together. The grid structure allows for a pairwise comparison of all tactics, showing relationship strength through color intensity and numerical counts.

### 2. Axes
*   **Labels:** Both the X-axis (bottom) and Y-axis (left) list the same categorical variables representing **Adversarial Tactics**:
    *   Collection, Command And Control, Credential Access, Defense Evasion, Discovery, Execution, Exfiltration, Impact, Initial Access, Lateral Movement, MI Attack Staging, MI Model Access, Persistence, Privilege Escalation, Reconnaissance, Resource Development.
*   **Units:** The axes represent categories, not numerical units. The values within the cells represent **counts (frequency)** of co-occurrence.
*   **Ranges:** N/A (Categorical).

### 3. Data Trends
*   **High Activity Areas (Dark Blue):**
    *   **The Diagonal:** The highest values generally appear on the diagonal line from top-left to bottom-right (e.g., **Initial Access at 38**, **Resource Development at 38**, **Impact at 36**). This usually represents the total frequency of that specific tactic appearing in the dataset.
    *   **Key Clusters:** There is a distinct cluster of high co-occurrence involving **Initial Access**, **Impact**, **Execution**, and **Resource Development**.
        *   *Initial Access* & *Impact*: **30** occurrences.
        *   *Initial Access* & *Resource Development*: **27** occurrences.
        *   *Execution* & *Initial Access*: **23** occurrences.
        *   *Resource Development* & *Execution*: **21** occurrences.
*   **Low Activity Areas (Light Yellow):**
    *   **Lateral Movement:** This row/column is predominantly light yellow, with values mostly between 0 and 2. This suggests that in this dataset, Lateral Movement is rarely observed co-occurring with other tactics.
    *   **Command And Control:** similarly shows very low engagement with other tactics (mostly single digits).
*   **Asymmetry/Symmetry:** The matrix appears to be **symmetric**. For example, the intersection of "Impact" (Row) and "Initial Access" (Column) is 30, and "Initial Access" (Row) and "Impact" (Column) is also 30. This indicates a non-directional relationship (Tactic A happened with Tactic B).

### 4. Annotations and Legends
*   **Color Bar (Right):** A gradient legend indicates the scale of values.
    *   **Light Yellow (0):** Represents no or very low co-occurrence.
    *   **Teal/Green (~15-20):** Represents moderate co-occurrence.
    *   **Dark Blue (>35):** Represents high frequency/co-occurrence.
*   **Cell Annotations:** Each cell contains an integer specifying the exact number of times the two intersecting tactics occurred together.
*   **Title:** "Adversarial Tactic Co-occurrence Heatmap (ATLAS) - Final" identifies the dataset context (likely related to Machine Learning security, given the "MI" in tactics like "MI Model Access").

### 5. Statistical Insights
*   **Dominant Tactic:** **Initial Access** appears to be the most "connected" or prevalent tactic. It has the highest self-occurrence (38) and strongly co-occurs with almost every other major category (Impact, Execution, Defense Evasion, Resource Development). This suggests that most attacks in this dataset rely heavily on gaining initial access.
*   **The "Impact" Correlation:** The tactic **Impact** is strongly correlated with **Initial Access (30)** and **Resource Development (25)**. This might suggest a pattern where attacks move quickly from entry to impact, or that these tactics are frequently logged together in incident reports.
*   **Specialized Tactics:** Tactics specific to Machine Learning (MI Attack Staging, MI Model Access) show moderate clustering. Specifically, **MI Model Access** co-occurs significantly with **MI Attack Staging (20)** and **Defense Evasion (22)**, indicating a sub-pattern of behavior specific to ML-based threats.
*   **Outliers:** **Lateral Movement** and **Command and Control** are statistical outliers in that they are underrepresented. This could imply that the attacks analyzed in this dataset (ATLAS) are perhaps more focused on immediate impact or data theft rather than long-term movement through a network.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
