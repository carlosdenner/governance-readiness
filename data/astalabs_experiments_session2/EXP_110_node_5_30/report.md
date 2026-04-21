# Experiment 110: node_5_30

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_30` |
| **ID in Run** | 110 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:12:51.160459+00:00 |
| **Runtime** | 364.5s |
| **Parent** | `node_4_28` |
| **Children** | `node_6_12`, `node_6_25` |
| **Creation Index** | 111 |

---

## Hypothesis

> The 'Malicious Sector' Profile: Incidents characterized as 'Intentional Harm'
(Malicious Use) are disproportionately concentrated in the 'Public Sector' and
'Media' sectors, whereas 'Unintentional' failures cluster in 'Transportation'
and 'Healthcare'.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2527 (Likely False) |
| **Surprise** | -0.5870 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 30.0 |
| Uncertain | 0.0 |
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

**Objective:** Map the intent of AI failures (Malicious vs Accidental) to specific deployment sectors, with robust error handling for sparse or missing intentionality data.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for rows where 'source_table' is 'aiid_incidents'.
- 2. Inspect and print the unique values of the 'Intentional Harm' column to identify the correct labels (e.g., 'True'/'False', 'Malicious'/'Accidental', or '1'/'0').
- 3. Create a binary 'is_intentional' column based on the observed values (map affirmative values to 1, others to 0).
- 4. Filter the dataset to include only the top 5 sectors by incident volume.
- 5. Create a contingency table of Sector vs. is_intentional.
- 6. Check if the 'is_intentional' column in the contingency table contains any non-zero values.
- 7. If non-zero values exist, perform a Chi-square test of independence and calculate standardized residuals to identify the sector with the strongest association with intentional harm.
- 8. If the column sum is zero (no intentional incidents), skip the test to prevent execution errors and report the finding.
- 9. Generate a stacked bar chart of the proportions.

### Deliverables
- 1. List of unique values found in the 'Intentional Harm' column.
- 2. Contingency table of Sector vs. Intentionality.
- 3. Chi-square test statistics (if applicable).
- 4. Identification of the sector most associated with malicious use (or a statement that no such incidents were found in the top sectors).

---

## Analysis

The experiment was successfully executed with the corrected data mapping logic.
The results **contradict** the 'Malicious Sector' hypothesis.

1. **Hypothesis Rejection**: The hypothesis predicted 'Transportation' and
'Healthcare' would cluster with 'Unintentional' failures. However, the data
shows **100% intentionality** for 'Transportation and storage' (21/21 incidents)
and 'Human health and social work activities' (10/10 incidents).
2. **Uniform Malice**: Across the top 5 sectors, the rate of intentional harm
was uniformly high (85% - 100%), rather than being concentrated solely in
'Public Sector' or 'Media'.
3. **Statistical Significance**: The Chi-square test (p = 0.23) indicates no
statistically significant difference in the maliciousness profile across these
sectors; they are all similarly prone to intentional misuse in this dataset.
4. **Operational Insight**: The semantic mapping update (`startswith('yes')`)
correctly parsed the verbose labels, revealing that the dataset's top sectors
are dominated by malicious use cases rather than accidental robustness failures.

---

## Review

The experiment was successfully recovered and executed. The final iteration
correctly handled the verbose data values in the 'Intentional Harm' column,
allowing for a valid statistical test. The analysis confirms that the hypothesis
is unsupported by the data.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# --- 1. Load Dataset ---
candidates = ['astalabs_discovery_all_data.csv', '../astalabs_discovery_all_data.csv']
file_path = None
for c in candidates:
    if os.path.exists(c):
        file_path = c
        break

if file_path is None:
    file_path = 'astalabs_discovery_all_data.csv'

print(f"Loading dataset from: {file_path}")
df_all = pd.read_csv(file_path, low_memory=False)

# --- 2. Data Preparation ---
df_aiid = df_all[df_all['source_table'] == 'aiid_incidents'].copy()

intent_col = 'Intentional Harm'
sector_col = 'Sector of Deployment'

# --- 3. Improved Mapping Logic ---
# Based on previous output, values start with 'Yes.' or 'No.'
def is_intentional(val):
    s = str(val).lower().strip()
    # Check for explicit affirmative start or keyword
    if s.startswith('yes') or 'intentionally designed' in s:
        return 1
    return 0

df_aiid['is_intentional'] = df_aiid[intent_col].apply(is_intentional)

# Clean Sector and remove 'Unknown' for better analysis
df_aiid['sector_clean'] = df_aiid[sector_col].fillna('Unknown').astype(str)
df_analyzable = df_aiid[df_aiid['sector_clean'].str.lower() != 'unknown'].copy()

# --- 4. Filter Top 5 Known Sectors ---
top_sectors = df_analyzable['sector_clean'].value_counts().nlargest(5).index.tolist()
df_top = df_analyzable[df_analyzable['sector_clean'].isin(top_sectors)].copy()

print(f"\nTop 5 Known Sectors: {top_sectors}")

# --- 5. Contingency Table ---
contingency = pd.crosstab(df_top['sector_clean'], df_top['is_intentional'])

# Ensure columns 0 and 1 exist
for c in [0, 1]:
    if c not in contingency.columns:
        contingency[c] = 0
contingency = contingency[[0, 1]]
contingency.columns = ['Unintentional', 'Intentional']

print("\nContingency Table (Observed):")
print(contingency)

# --- 6. Statistical Test ---
total_intentional = contingency['Intentional'].sum()

if total_intentional == 0:
    print("\n[!] Still no intentional incidents found in top sectors after logic update.")
else:
    # Perform Chi-Square Test
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    
    # Calculate Standardized Residuals
    observed_intent = contingency['Intentional']
    expected_intent = pd.Series(expected[:, 1], index=contingency.index)
    
    # Avoid division by zero in residuals (add small epsilon if expected is 0, though chi2 usually handles this)
    safe_expected = expected_intent.replace(0, 1e-9)
    std_residuals = (observed_intent - expected_intent) / np.sqrt(safe_expected)

    results_df = pd.DataFrame({
        'Total': contingency.sum(axis=1),
        'Intentional': observed_intent,
        'Rate': observed_intent / contingency.sum(axis=1),
        'Exp_Intent': expected_intent,
        'Residual': std_residuals
    }).sort_values('Residual', ascending=False)

    print("\n--- Chi-Square Results ---")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-Value:        {p_val:.4e}")
    
    print("\n--- Malicious Sector Profile (Ranked by Residual) ---")
    # Pretty print
    print_df = results_df.copy()
    print_df['Rate'] = print_df['Rate'].map('{:.1%}'.format)
    print_df['Residual'] = print_df['Residual'].map('{:.2f}'.format)
    print_df['Exp_Intent'] = print_df['Exp_Intent'].map('{:.1f}'.format)
    print(print_df[['Total', 'Intentional', 'Rate', 'Exp_Intent', 'Residual']].to_string())

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    # Normalize rows to 100%
    props = contingency.div(contingency.sum(axis=1), axis=0)
    # Sort by Intentional proportion
    props = props.sort_values('Intentional', ascending=True)
    
    ax = props.plot(kind='barh', stacked=True, color=['#A6CEE3', '#E31A1C'], figsize=(10, 6))
    plt.title('Proportion of Intentional vs Unintentional Harm by Top 5 Sectors')
    plt.xlabel('Proportion')
    plt.ylabel('Sector')
    plt.legend(['Unintentional', 'Intentional'], loc='lower right')
    plt.tight_layout()
    plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: astalabs_discovery_all_data.csv

Top 5 Known Sectors: ['information and communication', 'transportation and storage', 'Arts, entertainment and recreation, information and communication', 'wholesale and retail trade', 'human health and social work activities']

Contingency Table (Observed):
                                                    Unintentional  Intentional
sector_clean                                                                  
Arts, entertainment and recreation, information...              2           12
human health and social work activities                         0           10
information and communication                                   2           42
transportation and storage                                      0           21
wholesale and retail trade                                      0           11

--- Chi-Square Results ---
Chi2 Statistic: 5.6412
P-Value:        2.2759e-01

--- Malicious Sector Profile (Ranked by Residual) ---
                                                                   Total  Intentional    Rate Exp_Intent Residual
sector_clean                                                                                                     
transportation and storage                                            21           21  100.0%       20.2     0.19
wholesale and retail trade                                            11           11  100.0%       10.6     0.14
human health and social work activities                               10           10  100.0%        9.6     0.13
information and communication                                         44           42   95.5%       42.2    -0.04
Arts, entertainment and recreation, information and communication     14           12   85.7%       13.4    -0.39


=== Plot Analysis (figure 2) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Horizontal 100% Stacked Bar Chart.
*   **Purpose:** The chart is designed to compare the relative proportions of two specific categories ("Intentional" vs. "Unintentional" harm) across five different categorical groups (sectors). It allows for an easy visual comparison of the composition of harm within each sector.

### 2. Axes
*   **Y-Axis (Vertical):**
    *   **Title:** "Sector"
    *   **Labels:** The axis lists five specific sectors:
        1.  wholesale and retail trade
        2.  transportation and storage
        3.  human health and social work activities
        4.  information and communication
        5.  Arts, entertainment and recreation, information and communication
*   **X-Axis (Horizontal):**
    *   **Title:** "Proportion"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100%).
    *   **Scale:** Linear, with major grid/tick marks every 0.2 units.

### 3. Data Trends
*   **Dominant Trend:** Across all five sectors, **Intentional harm** (represented by the red bars) is the overwhelmingly dominant category. In every sector, it accounts for the vast majority of the proportion (over 80%).
*   **Exclusively Intentional Sectors:** Three sectors appear to have **100% Intentional harm** with no visible "Unintentional" component:
    *   wholesale and retail trade
    *   transportation and storage
    *   human health and social work activities
*   **Presence of Unintentional Harm:** Only the bottom two sectors display any "Unintentional" harm (light blue bars):
    *   "Arts, entertainment and recreation, information and communication" has the largest proportion of unintentional harm, appearing to be approximately **0.15 (15%)**.
    *   "information and communication" has a very small proportion of unintentional harm, appearing to be roughly **0.05 (5%)**.

### 4. Annotations and Legends
*   **Chart Title:** "Proportion of Intentional vs Unintentional Harm by Top 5 Sectors" — sets the context for the data.
*   **Legend:** Located at the bottom right corner.
    *   **Light Blue:** Represents "Unintentional".
    *   **Red:** Represents "Intentional".

### 5. Statistical Insights
*   **High Risk of Malice:** The data suggests that for these top sectors, harm is rarely accidental. The incidents recorded are primarily driven by malicious or deliberate actions rather than negligence or accidents.
*   **Sector Vulnerability:** "Wholesale and retail," "Transportation," and "Health/Social work" show a distinct pattern where unintentional harm is negligible or non-existent in this dataset.
*   **Operational Context:** The sector labeled "Arts, entertainment and recreation, information and communication" shows the highest variability in the type of harm, indicating that while malicious actors are the primary threat, accidental or unintentional issues play a more significant role here than in the other listed industries.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
