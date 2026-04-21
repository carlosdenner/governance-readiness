# Experiment 218: node_6_40

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_40` |
| **ID in Run** | 218 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T11:10:23.913230+00:00 |
| **Runtime** | 267.0s |
| **Parent** | `node_5_42` |
| **Children** | None |
| **Creation Index** | 219 |

---

## Hypothesis

> The Autonomy-Harm Hypothesis: Fully autonomous systems are significantly more
associated with 'Tangible Harm' than systems with 'Human-in-the-loop' (Assisted)
autonomy.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.5242 (Uncertain) |
| **Posterior** | 0.1841 (Likely False) |
| **Surprise** | -0.4082 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 14.0 |
| Uncertain | 5.0 |
| Maybe False | 11.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 2.0 |
| Definitely False | 58.0 |

---

## Experiment Plan

**Objective:** Test if higher levels of autonomy correlate with physical consequences.

### Steps
- 1. Load 'aiid_incidents' data.
- 2. Create a variable `autonomy_group` grouping `Autonomy Level` into 'High' (e.g., Full, High) and 'Low' (e.g., Assisted, Human-in-the-loop).
- 3. Define `is_tangible` as before (strict string match).
- 4. Generate a contingency table of Autonomy Group vs. Tangibility.
- 5. Perform a statistical test (Chi-square or Fisher's).
- 6. Analyze if removing the human from the loop statistically increases the likelihood of tangible failure.

### Deliverables
- Grouped autonomy counts, contingency table, and statistical significance test.

---

## Analysis

The experiment successfully tested the Autonomy-Harm Hypothesis using the AIID
dataset, incorporating the correct data mappings identified in the previous
debug step.

**Findings:**
1.  **Data Distribution:**
    -   **High Autonomy (Level 3):** 53 incidents. 35.85% (19) resulted in
tangible harm.
    -   **Low Autonomy (Levels 1 & 2):** 129 incidents. 35.66% (46) resulted in
tangible harm.

2.  **Statistical Test:**
    -   **Fisher's Exact Test P-value:** 0.5553
    -   **Odds Ratio:** 1.0083

**Conclusion:**
The results show almost identical rates of tangible harm between fully
autonomous systems and human-in-the-loop systems (approx. 36% for both). With an
odds ratio near 1.0 and a high p-value, the difference is **not statistically
significant**. The hypothesis that higher autonomy correlates with higher
tangible harm is **strongly rejected** by this dataset; the presence of a human
in the loop does not appear to statistically alter the likelihood of physical or
financial consequences in the recorded incidents.

---

## Review

The experiment was successfully executed and faithfully implemented the analysis
plan using the specific data mappings identified in the debugging phase.

**Hypothesis:** The Autonomy-Harm Hypothesis (High autonomy systems are
significantly more likely to cause tangible harm than low autonomy/human-in-the-
loop systems).

**Methodology:**
- **Dataset:** AIID Incidents (n=1,362).
- **High Autonomy:** Mapped from 'Autonomy3' (System-level/Autonomous).
- **Low Autonomy:** Mapped from 'Autonomy1' and 'Autonomy2' (Assisted/Human-in-
the-loop).
- **Tangible Harm:** Mapped specifically from incidents where 'tangible harm
definitively occurred'.

**Findings:**
1. **Data Distribution:**
   - **High Autonomy (n=53):** 19 incidents (35.85%) resulted in tangible harm.
   - **Low Autonomy (n=129):** 46 incidents (35.66%) resulted in tangible harm.

2. **Statistical Analysis:**
   - **Fisher's Exact Test P-value:** 0.5553 (> 0.05).
   - **Odds Ratio:** 1.0083.

**Conclusion:**
The hypothesis is **rejected**. The analysis reveals a striking similarity in
tangible harm rates between fully autonomous systems and those with human
oversight (approx. 36% for both). The Odds Ratio of ~1.0 indicates no
association between the level of autonomy and the tangibility of the harm in
this dataset. This suggests that simply having a human in the loop (at least as
classified in AIID) does not statistically differentiate the likelihood of
physical or financial consequences compared to fully autonomous systems.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

# Load the dataset
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
df_incidents = df[df['source_table'] == 'aiid_incidents'].copy()

print("--- Data Loading ---")
print(f"Total Incidents Loaded: {len(df_incidents)}")

# --- Data Cleaning & Mapping ---

# Map Autonomy Level based on debug findings
# Autonomy3 -> High (Autonomous)
# Autonomy1, Autonomy2 -> Low (Assisted/Human-in-the-loop)
def map_autonomy(val):
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).strip()
    if val_str == 'Autonomy3':
        return 'High'
    elif val_str in ['Autonomy1', 'Autonomy2']:
        return 'Low'
    return 'Unknown'

# Map Tangible Harm based on debug findings
# 'tangible harm definitively occurred' -> Tangible
# All other valid categories (near-misses, issues, no harm) -> Intangible (i.e., did not result in tangible harm)
def map_harm(val):
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).strip()
    
    if val_str == 'tangible harm definitively occurred':
        return 'Tangible'
    elif val_str in ['no tangible harm, near-miss, or issue', 
                     'non-imminent risk of tangible harm (an issue) occurred',
                     'imminent risk of tangible harm (near miss) did occur']:
        return 'Intangible'
    return 'Unknown'

# Apply mappings
df_incidents['Autonomy_Group'] = df_incidents['Autonomy Level'].apply(map_autonomy)
df_incidents['Harm_Type'] = df_incidents['Tangible Harm'].apply(map_harm)

# Filter out Unknowns for the analysis
analysis_df = df_incidents[
    (df_incidents['Autonomy_Group'] != 'Unknown') & 
    (df_incidents['Harm_Type'] != 'Unknown')
].copy()

print("\n--- Mapped Data Distribution (Analysis Subset) ---")
print(pd.crosstab(analysis_df['Autonomy_Group'], analysis_df['Harm_Type']))

# --- Statistical Test ---
# Contingency Table structure: 
#           Tangible | Intangible
# High      a          b
# Low       c          d

contingency_table = pd.crosstab(
    analysis_df['Autonomy_Group'], 
    analysis_df['Harm_Type']
).reindex(index=['High', 'Low'], columns=['Tangible', 'Intangible'], fill_value=0)

print("\n--- Contingency Table for Statistical Test ---")
print(contingency_table)

if contingency_table.values.sum() == 0:
    print("\nInsufficient data for statistical test.")
else:
    # Fisher's Exact Test
    # Alternative='greater' tests if High Autonomy is MORE associated with Tangible Harm than Low Autonomy is.
    odds_ratio, p_value = fisher_exact(contingency_table, alternative='greater')

    print("\n--- Fisher's Exact Test Results ---")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Calculate percentages for context
    high_sum = contingency_table.loc['High'].sum()
    low_sum = contingency_table.loc['Low'].sum()
    
    high_tangible_rate = (contingency_table.loc['High', 'Tangible'] / high_sum) if high_sum > 0 else 0
    low_tangible_rate = (contingency_table.loc['Low', 'Tangible'] / low_sum) if low_sum > 0 else 0

    print(f"\nHigh Autonomy Tangible Rate: {high_tangible_rate:.2%} ({contingency_table.loc['High', 'Tangible']}/{high_sum})")
    print(f"Low Autonomy Tangible Rate:  {low_tangible_rate:.2%} ({contingency_table.loc['Low', 'Tangible']}/{low_sum})")

    if p_value < 0.05:
        print("\nResult: Statistically Significant. The Autonomy-Harm Hypothesis is supported.")
    else:
        print("\nResult: Not Statistically Significant. The Autonomy-Harm Hypothesis is NOT supported.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Data Loading ---
Total Incidents Loaded: 1362

--- Mapped Data Distribution (Analysis Subset) ---
Harm_Type       Intangible  Tangible
Autonomy_Group                      
High                    34        19
Low                     83        46

--- Contingency Table for Statistical Test ---
Harm_Type       Tangible  Intangible
Autonomy_Group                      
High                  19          34
Low                   46          83

--- Fisher's Exact Test Results ---
Odds Ratio: 1.0083
P-value: 0.5553

High Autonomy Tangible Rate: 35.85% (19/53)
Low Autonomy Tangible Rate:  35.66% (46/129)

Result: Not Statistically Significant. The Autonomy-Harm Hypothesis is NOT supported.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
