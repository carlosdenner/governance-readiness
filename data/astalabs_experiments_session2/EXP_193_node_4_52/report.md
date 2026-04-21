# Experiment 193: node_4_52

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_52` |
| **ID in Run** | 193 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:10:25.602985+00:00 |
| **Runtime** | 181.6s |
| **Parent** | `node_3_4` |
| **Children** | `node_5_89` |
| **Creation Index** | 194 |

---

## Hypothesis

> Autonomy-Intent Paradox: Incidents involving 'Low Autonomy' systems are
significantly more likely to be triggered by 'Intentional' malicious acts
compared to 'High Autonomy' systems, which are more prone to 'Unintentional'
failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.4121 (Maybe False) |
| **Surprise** | -0.3958 |
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
| Maybe False | 58.0 |
| Definitely False | 2.0 |

---

## Experiment Plan

**Objective:** Explore the relationship between system autonomy and the intentionality of the incident.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `aiid_incidents`.
- 2. Map `Autonomy Level` to binary: 'Low' (Assistive/Partial) vs 'High' (Autonomous).
- 3. Use `Intentional Harm` column (Boolean/Binary).
- 4. Perform a Fisher's Exact Test or Chi-square test on Autonomy vs. Intentionality.

### Deliverables
- Odds ratio indicating the association between low autonomy and intentional misuse.

---

## Analysis

The experiment successfully tested the Autonomy-Intent Paradox hypothesis using
the AIID dataset. After cleaning the 'Autonomy Level' (mapped to Low/High) and
'Intentional Harm' (mapped to Intentional/Unintentional) columns, 177 valid
incidents remained for analysis.

**Results**:
- **Contingency Counts**:
  - Low Autonomy: 5 Intentional (4.9%), 98 Unintentional.
  - High Autonomy: 3 Intentional (4.1%), 71 Unintentional.
- **Fisher's Exact Test**:
  - Odds Ratio: 1.2075
  - P-value: 0.5524

**Conclusion**: The analysis found no statistically significant difference in
the likelihood of intentional harm between Low and High autonomy systems. The
hypothesis is **not supported**; the proportion of intentional incidents is
nearly identical across autonomy levels in this dataset.

---

## Review

The experiment was faithfully implemented. The programmer correctly identified
the need to adjust the data mapping logic for 'Autonomy Level' (mapping
'Autonomy1' to Low and 'Autonomy2/3' to High) and 'Intentional Harm' (parsing
long-form text responses) after an initial failed attempt. The statistical test
(Fisher's Exact Test) was appropriate for the sample size (n=177) and the binary
nature of the variables.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import sys

# Load the dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback for alternative directory structures if needed, though instruction says same level
    # Trying relative path based on previous prompt hint
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

print(f"Total AIID Incidents: {len(aiid_df)}")

# --- Corrected Preprocessing ---

# 1. Clean Autonomy Level
# Mapping based on typical definitions for Autonomy1/2/3 in this dataset context:
# Autonomy1 = System recommends action, human decides (Low)
# Autonomy2 = System initiates action, human can override (High/Medium)
# Autonomy3 = System initiates action, human out of loop (High)

def map_autonomy_corrected(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    
    if val_str == 'Autonomy1':
        return 'Low'
    elif val_str in ['Autonomy2', 'Autonomy3']:
        return 'High'
    return None

# 2. Clean Intentional Harm
# Values start with 'Yes' or 'No'

def map_intent_corrected(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip().lower()
    
    if val_str.startswith('yes'):
        return 'Intentional'
    elif val_str.startswith('no'):
        return 'Unintentional'
    return None

aiid_df['autonomy_bin'] = aiid_df['Autonomy Level'].apply(map_autonomy_corrected)
aiid_df['intent_bin'] = aiid_df['Intentional Harm'].apply(map_intent_corrected)

# Drop records where we couldn't classify
analysis_df = aiid_df.dropna(subset=['autonomy_bin', 'intent_bin'])

print(f"Records available for analysis: {len(analysis_df)}")

# --- Analysis ---

# Create Contingency Table
contingency_table = pd.crosstab(analysis_df['autonomy_bin'], analysis_df['intent_bin'])
print("\nContingency Table (Autonomy vs Intentionality):")
print(contingency_table)

# Check if we have data for all cells to be safe
if contingency_table.empty:
    print("\nNo valid data found for contingency table.")
else:
    # Extract values for Fisher's Exact Test
    # Structure: [[Low_Intentional, Low_Unintentional], [High_Intentional, High_Unintentional]]
    
    # Helper to safely get value
    def get_val(r, c):
        try:
            return contingency_table.loc[r, c]
        except KeyError:
            return 0

    low_intent = get_val('Low', 'Intentional')
    low_unintent = get_val('Low', 'Unintentional')
    high_intent = get_val('High', 'Intentional')
    high_unintent = get_val('High', 'Unintentional')
    
    table_for_stats = [[low_intent, low_unintent], [high_intent, high_unintent]]
    
    print(f"\nMatrix for Fisher's Test:\n {table_for_stats}")
    
    if (low_intent + low_unintent == 0) or (high_intent + high_unintent == 0):
         print("Cannot run test: One row is empty.")
    else:
        # We expect Low Autonomy to be MORE associated with Intentional Harm
        # So Odds Ratio > 1
        odds_ratio, p_value = stats.fisher_exact(table_for_stats, alternative='greater')
        
        print(f"\n--- Statistical Results ---")
        print(f"Odds Ratio: {odds_ratio:.4f}")
        print(f"P-Value (one-sided 'greater'): {p_value:.4f}")
        
        # Calculate percentages for clearer interpretation
        pct_low_intent = (low_intent / (low_intent + low_unintent)) * 100
        pct_high_intent = (high_intent / (high_intent + high_unintent)) * 100
        
        print(f"\nInterpretation:")
        print(f"% of Low Autonomy incidents that were Intentional: {pct_low_intent:.1f}%")
        print(f"% of High Autonomy incidents that were Intentional: {pct_high_intent:.1f}%")
        
        if p_value < 0.05:
            print("\nResult: Statistically Significant. The hypothesis is supported.")
        else:
            print("\nResult: Not Statistically Significant.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total AIID Incidents: 1362
Records available for analysis: 177

Contingency Table (Autonomy vs Intentionality):
intent_bin    Intentional  Unintentional
autonomy_bin                            
High                    3             71
Low                     5             98

Matrix for Fisher's Test:
 [[np.int64(5), np.int64(98)], [np.int64(3), np.int64(71)]]

--- Statistical Results ---
Odds Ratio: 1.2075
P-Value (one-sided 'greater'): 0.5524

Interpretation:
% of Low Autonomy incidents that were Intentional: 4.9%
% of High Autonomy incidents that were Intentional: 4.1%

Result: Not Statistically Significant.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
