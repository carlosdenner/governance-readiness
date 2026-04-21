# Experiment 220: node_5_86

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_86` |
| **ID in Run** | 220 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T11:17:47.355447+00:00 |
| **Runtime** | 314.2s |
| **Parent** | `node_4_1` |
| **Children** | None |
| **Creation Index** | 221 |

---

## Hypothesis

> Adversarial Attack Chains: In AI security incidents, the tactic 'Exfiltration'
(TA0010) is statistically dependent on the presence of 'Collection' (TA0009),
confirming a sequential attack chain pattern in AI threat landscapes.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8226 (Likely True) |
| **Posterior** | 0.9396 (Definitely True) |
| **Surprise** | +0.1404 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 10.0 |
| Maybe True | 20.0 |
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

**Objective:** Validate the co-occurrence of specific adversarial tactics.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `atlas_cases` or `step3_incident_coding`.
- 2. Parse the `92_tactics` column (handling semi-colon delimiters) to create binary flags for 'Exfiltration' and 'Collection'.
- 3. Perform a Fisher's Exact Test (due to likely small sample size) to test the association.
- 4. Calculate the Jaccard Similarity Coefficient for the two tactics.

### Deliverables
- Co-occurrence Matrix; Fisher's Exact Test p-value; Jaccard score.

---

## Analysis

The experiment successfully validated the hypothesis regarding the sequential
dependency between 'Collection' (TA0009) and 'Exfiltration' (TA0010) in AI
adversarial incidents. Unlike the previous attempt, this execution correctly
identified the text-based `tactics` column, allowing for accurate string
matching of tactic names and IDs.

1. **Statistical Significance**: The Fisher's Exact Test yielded a p-value of
practically 0.0000 and an Odds Ratio of 49.71, overwhelmingly rejecting the null
hypothesis. This confirms a very strong statistical dependence between the two
tactics.
2. **Attack Chain Confirmation**: The contingency table reveals that
'Collection' rarely occurs without 'Exfiltration' (only 2 instances), while they
co-occur in 8 instances. This supports the 'kill chain' logic where data is
collected specifically to be exfiltrated. Interestingly, there were 7 cases of
Exfiltration without explicit Collection, potentially indicating immediate theft
or data access without a distinct staging phase.
3. **Jaccard Similarity**: A coefficient of 0.47 indicates a moderate to strong
overlap relative to the rarity of these specific tactics in the broader dataset
(where 87 cases involved neither).

---

## Review

The experiment successfully validated the hypothesis regarding the sequential
dependency between 'Collection' (TA0009) and 'Exfiltration' (TA0010) in AI
adversarial incidents. Unlike the previous attempt, this execution correctly
identified the text-based `tactics` column, allowing for accurate string
matching of tactic names and IDs.

1. **Statistical Significance**: The Fisher's Exact Test yielded a p-value of
practically 0.0000 and an Odds Ratio of 49.71, overwhelmingly rejecting the null
hypothesis. This confirms a very strong statistical dependence between the two
tactics.
2. **Attack Chain Confirmation**: The contingency table reveals that
'Collection' rarely occurs without 'Exfiltration' (only 2 instances), while they
co-occur in 8 instances. This supports the 'kill chain' logic where data is
collected specifically to be exfiltrated. Interestingly, there were 7 cases of
Exfiltration without explicit Collection, potentially indicating immediate theft
or data access without a distinct staging phase.
3. **Jaccard Similarity**: A coefficient of 0.47 indicates a moderate to strong
overlap relative to the rarity of these specific tactics in the broader dataset
(where 87 cases involved neither).

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact
import os
import sys

# 1. Load Data
filename = 'astalabs_discovery_all_data.csv'
file_path = filename if os.path.exists(filename) else os.path.join('..', filename)

print(f"Loading dataset from {file_path}...")
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: {filename} not found.")
    sys.exit(1)

# 2. Filter for ATLAS cases / Incident Coding
target_sources = ['atlas_cases', 'step3_incident_coding']
atlas_df = df[df['source_table'].isin(target_sources)].copy()
print(f"Filtered {len(atlas_df)} records from sources: {target_sources}")

# 3. Identify Tactics Column
# Priority: 'tactics', then 'tactics_used', avoiding 'n_tactics'
tactics_col = None
if 'tactics' in atlas_df.columns:
    tactics_col = 'tactics'
elif 'tactics_used' in atlas_df.columns:
    tactics_col = 'tactics_used'
else:
    # Fallback search
    for col in atlas_df.columns:
        if 'tactics' in str(col).lower() and 'n_' not in str(col).lower() and 'question' not in str(col).lower():
            tactics_col = col
            break

if not tactics_col:
    print("Error: Could not identify 'tactics' text column. Available columns with 'tactics':")
    print([c for c in atlas_df.columns if 'tactics' in str(c).lower()])
    sys.exit(1)

print(f"Using column '{tactics_col}' for tactics analysis.")

# Inspect data to ensure it contains text
print("Sample values from tactics column:")
print(atlas_df[tactics_col].dropna().head(5).values)

# Normalize tactics column
atlas_df[tactics_col] = atlas_df[tactics_col].fillna('').astype(str).str.lower()

# 4. Create Binary Flags
# 'Collection' (TA0009) and 'Exfiltration' (TA0010)
def has_tactic(text, names, ids):
    text = text.lower()
    for name in names:
        if name.lower() in text:
            return True
    for tid in ids:
        if tid.lower() in text:
            return True
    return False

atlas_df['has_collection'] = atlas_df[tactics_col].apply(
    lambda x: has_tactic(x, ['collection'], ['ta0009'])
)

atlas_df['has_exfiltration'] = atlas_df[tactics_col].apply(
    lambda x: has_tactic(x, ['exfiltration'], ['ta0010'])
)

# 5. Contingency Table
contingency_table = pd.crosstab(
    atlas_df['has_collection'], 
    atlas_df['has_exfiltration'], 
    rownames=['Has Collection (TA0009)'], 
    colnames=['Has Exfiltration (TA0010)']
)

# Ensure 2x2
contingency_table = contingency_table.reindex(
    index=[False, True], columns=[False, True], fill_value=0
)

print("\nContingency Table:")
print(contingency_table)

# 6. Statistical Test (Fisher's Exact Test)
odds_ratio, p_value = fisher_exact(contingency_table, alternative='greater')

print(f"\nFisher's Exact Test Results:")
print(f"Odds Ratio: {odds_ratio:.4f}")
print(f"P-value: {p_value:.4f}")

# 7. Jaccard Similarity
n_collection = atlas_df['has_collection'].sum()
n_exfiltration = atlas_df['has_exfiltration'].sum()
n_both = contingency_table.loc[True, True]
n_union = n_collection + n_exfiltration - n_both

jaccard = n_both / n_union if n_union > 0 else 0.0
print(f"\nJaccard Similarity Coefficient: {jaccard:.4f}")

# 8. Visualization
plt.figure(figsize=(6, 5))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Adversarial Chains: Collection (TA0009) vs Exfiltration (TA0010)')
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
Filtered 104 records from sources: ['atlas_cases', 'step3_incident_coding']
Using column 'tactics' for tactics analysis.
Sample values from tactics column:
<StringArray>
[                                        '{{defense_evasion.id}}|{{ml_attack_staging.id}}|{{reconnaissance.id}}|{{resource_development.id}}',
                                         '{{defense_evasion.id}}|{{ml_attack_staging.id}}|{{reconnaissance.id}}|{{resource_development.id}}',
                                             '{{initial_access.id}}|{{ml_attack_staging.id}}|{{persistence.id}}|{{resource_development.id}}',
 '{{defense_evasion.id}}|{{discovery.id}}|{{ml_attack_staging.id}}|{{ml_model_access.id}}|{{reconnaissance.id}}|{{resource_development.id}}',
                              '{{impact.id}}|{{initial_access.id}}|{{ml_model_access.id}}|{{reconnaissance.id}}|{{resource_development.id}}']
Length: 5, dtype: str

Contingency Table:
Has Exfiltration (TA0010)  False  True 
Has Collection (TA0009)                
False                         87      7
True                           2      8

Fisher's Exact Test Results:
Odds Ratio: 49.7143
P-value: 0.0000

Jaccard Similarity Coefficient: 0.4706


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **Contingency Table** visualized as a **Heatmap** (or confusion matrix).
*   **Purpose:** Its purpose is to display the frequency distribution of two categorical variables simultaneously. Specifically, it visualizes the co-occurrence of "Collection" and "Exfiltration" tactics within adversarial chains.

### 2. Axes
*   **Vertical Axis (Y-axis):**
    *   **Label:** "Has Collection (TA0009)"
    *   **Categories:** "False" (top row) and "True" (bottom row).
    *   **Context:** Refers to the MITRE ATT&CK tactic TA0009 (Collection).
*   **Horizontal Axis (X-axis):**
    *   **Label:** "Has Exfiltration (TA0010)"
    *   **Categories:** "False" (left column) and "True" (right column).
    *   **Context:** Refers to the MITRE ATT&CK tactic TA0010 (Exfiltration).
*   **Value Ranges:** The axes represent binary categories (Boolean True/False), not continuous numerical ranges.

### 3. Data Trends
*   **Highest Value (Darkest Area):** The top-left cell (False/False) contains the value **87**. This indicates that the vast majority of the observed adversarial chains involved neither Collection nor Exfiltration.
*   **Lowest Values:** The bottom-left cell (True/False) contains the lowest value, **2**. This indicates it is very rare to have "Collection" without "Exfiltration" in this dataset.
*   **Patterns:**
    *   There is a distinct disparity in sample sizes. The "False/False" category dominates the dataset.
    *   The second most common category is the "True/True" case (bottom-right) with **8** occurrences, followed closely by "False/True" (top-right) with **7** occurrences.

### 4. Annotations and Legends
*   **Title:** "Adversarial Chains: Collection (TA0009) vs Exfiltration (TA0010)". This establishes the subject matter as cybersecurity threat analysis based on the MITRE ATT&CK framework.
*   **Cell Annotations:** Each quadrant contains a numerical count:
    *   **87:** No Collection, No Exfiltration.
    *   **7:** No Collection, Has Exfiltration.
    *   **2:** Has Collection, No Exfiltration.
    *   **8:** Has Collection, Has Exfiltration.
*   **Color Scale:** The plot uses a sequential blue color palette where darker blue indicates a higher count/frequency and lighter blue/white indicates a lower count.

### 5. Statistical Insights
*   **Strong Correlation in Presence:** When "Collection" is present (True), "Exfiltration" is highly likely to also be present.
    *   Out of 10 cases where Collection is True, 8 involved Exfiltration (80%).
*   **Low Baseline Rate:** When "Collection" is absent (False), "Exfiltration" is unlikely.
    *   Out of 94 cases where Collection is False, only 7 involved Exfiltration (~7.4%).
*   **Dataset Composition:** The total sample size is 104. Approximately **83.6%** (87/104) of the data points represent negative samples for both tactics.
*   **Cybersecurity Context:** The data supports a logical progression in an attack chain: attackers often perform Collection (gathering data) specifically to perform Exfiltration (stealing data). Therefore, seeing Collection without Exfiltration is rare (only 2 cases), likely representing failed attacks or interdicted chains. The 7 cases of Exfiltration without Collection might represent scenarios where data was accessed and stolen immediately without a distinct staging/collection phase.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
