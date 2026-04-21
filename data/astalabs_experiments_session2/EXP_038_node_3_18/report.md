# Experiment 38: node_3_18

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_18` |
| **ID in Run** | 38 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T02:49:20.828493+00:00 |
| **Runtime** | 180.8s |
| **Parent** | `node_2_2` |
| **Children** | `node_4_20`, `node_4_27` |
| **Creation Index** | 39 |

---

## Hypothesis

> Autonomy-Harm Escalation: In the AIID dataset, AI incidents involving 'High'
autonomy systems (e.g., systems that act without human intervention) are
statistically more likely to result in 'Severe' or 'Critical' harm levels
compared to 'Low' autonomy systems (e.g., recommender systems).

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.4176 (Maybe False) |
| **Surprise** | -0.3892 |
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
| Maybe False | 60.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Assess the relationship between system autonomy levels and the severity of reported harms.

### Steps
- 1. Filter for `source_table` = 'aiid_incidents'.
- 2. Clean and categorize `Autonomy Level` into 'High' (e.g., 'Autonomous', 'High') and 'Low' (e.g., 'Human in the loop', 'Support').
- 3. Clean and categorize `AI Harm Level` into 'Severe' (e.g., 'Critical', 'Severe', 'Death') and 'Not Severe' (e.g., 'Minor', 'Moderate').
- 4. Create a contingency table and perform a Chi-square test.
- 5. Calculate the Odds Ratio of Severe Harm for High vs. Low Autonomy.

### Deliverables
- Contingency table, Chi-square statistics, and Odds Ratio.

---

## Analysis

The experiment successfully analyzed 177 AIID incidents where both autonomy and
harm data were available. The contingency table revealed 15 severe harm cases
out of 52 high-autonomy incidents (28.8%) compared to 26 severe harm cases out
of 125 low-autonomy incidents (20.8%). While the Odds Ratio of 1.54 suggests
that high-autonomy systems had higher odds of resulting in severe harm, the Chi-
square test (p=0.337) indicates this difference is not statistically
significant. Therefore, the hypothesis that high autonomy significantly
escalates harm severity cannot be supported by this specific dataset slice.

---

## Review

The experiment was successfully executed following the specified plan. The
programmer correctly adapted to the specific value schemas found in the dataset
(mapping 'Autonomy3' to High and 'AI tangible harm event' to Severe). The
analysis of 177 incidents revealed that while High Autonomy systems had a higher
rate of severe harm (28.8%) compared to Low Autonomy systems (20.8%), the
difference was not statistically significant (Chi-Square=0.92, p=0.34). Thus,
the hypothesis that high autonomy significantly escalates harm severity is
rejected based on this sample.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import numpy as np

try:
    # Load dataset
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    
    # Filter for AIID incidents
    aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    
    # --- Preprocessing Autonomy Level ---
    def map_autonomy(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        if val == 'Autonomy3':
            return 'High'
        elif val in ['Autonomy1', 'Autonomy2']:
            return 'Low'
        return np.nan

    aiid['autonomy_bin'] = aiid['Autonomy Level'].apply(map_autonomy)
    
    # --- Preprocessing Harm Severity ---
    def map_severity(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        if val == 'AI tangible harm event':
            return 'Severe'
        elif val in ['AI tangible harm near-miss', 'AI tangible harm issue', 'none']:
            return 'Not Severe'
        return np.nan

    aiid['severity_bin'] = aiid['AI Harm Level'].apply(map_severity)
    
    # Drop rows where either variable is NaN
    analysis_df = aiid.dropna(subset=['autonomy_bin', 'severity_bin'])
    
    print(f"Data points for analysis: {len(analysis_df)}")
    
    # Contingency Table
    contingency = pd.crosstab(analysis_df['autonomy_bin'], analysis_df['severity_bin'])
    print("\n--- Contingency Table (Autonomy vs Severity) ---")
    print(contingency)
    
    if contingency.shape == (2, 2):
        # Chi-square test
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        
        print(f"\nChi-square Statistic: {chi2:.4f}")
        print(f"P-value: {p:.4f}")
        
        # Odds Ratio Calculation
        # Standard OR formula: (a*d) / (b*c)
        # Where a = High+Severe, b = High+NotSevere, c = Low+Severe, d = Low+NotSevere
        # However, crosstab order depends on sorting. Let's extract explicitly.
        
        try:
            n_high_severe = contingency.loc['High', 'Severe']
            n_high_not = contingency.loc['High', 'Not Severe']
            n_low_severe = contingency.loc['Low', 'Severe']
            n_low_not = contingency.loc['Low', 'Not Severe']
            
            if n_high_not * n_low_severe == 0:
                odds_ratio = np.inf
            else:
                odds_ratio = (n_high_severe * n_low_not) / (n_high_not * n_low_severe)
                
            print(f"Odds Ratio (High Autonomy -> Severe Harm): {odds_ratio:.4f}")
            
        except KeyError as e:
            print(f"Could not calculate OR due to missing keys in contingency table: {e}")
    else:
        print("\nContingency table is not 2x2. Cannot perform standard binary Odds Ratio calculation.")

except Exception as e:
    print(f"An error occurred: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Data points for analysis: 177

--- Contingency Table (Autonomy vs Severity) ---
severity_bin  Not Severe  Severe
autonomy_bin                    
High                  37      15
Low                   99      26

Chi-square Statistic: 0.9220
P-value: 0.3370
Odds Ratio (High Autonomy -> Severe Harm): 1.5437

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
