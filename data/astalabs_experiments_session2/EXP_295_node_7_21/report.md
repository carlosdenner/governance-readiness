# Experiment 295: node_7_21

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_21` |
| **ID in Run** | 295 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T15:10:59.328632+00:00 |
| **Runtime** | 247.2s |
| **Parent** | `node_6_43` |
| **Children** | None |
| **Creation Index** | 296 |

---

## Hypothesis

> Malice and Monetization: Incidents classified as 'Intentional Harm' are
significantly more likely to result in 'Economic/Financial' or 'Privacy' harms,
whereas 'Unintentional' incidents cluster around 'Physical Safety' harms.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.4780 (Uncertain) |
| **Surprise** | -0.3167 |
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
| Maybe True | 8.0 |
| Uncertain | 6.0 |
| Maybe False | 46.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Correlate actor intent with specific harm domains in the AIID dataset.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'source_table' == 'aiid_incidents'.
- 2. Map '82_Intentional Harm' to 'Intentional' (Yes) vs 'Unintentional' (No/Accident).
- 3. Map '73_Harm Domain' (or derive from text fields) to 'Financial/Privacy' vs 'Physical Safety'.
- 4. Remove ambiguous rows.
- 5. Run a Chi-square test on the Intent vs. Harm Domain table.

### Deliverables
- Contingency table showing the distribution of harm types by intent and statistical test results.

---

## Analysis

The experiment successfully tested the 'Malice and Monetization' hypothesis on
the 'aiid_incidents' dataset using text mining to derive harm categories.

1. **Data Processing**: The keyword-based extraction strategy successfully
classified 81 incidents. However, the dataset revealed a severe class imbalance:
only 5 incidents were confirmed as 'Intentional', compared to 76 'Unintentional'
accidents.

2. **Statistical Results**: Due to the low cell counts (specifically the
'Intentional' row), the code correctly applied Fisher's Exact Test instead of
Chi-square. The test yielded a p-value of **0.2293**, which is above the
standard significance threshold of 0.05.

3. **Hypothesis Evaluation**: The hypothesis was **not supported** by the
available data. While the Odds Ratio (3.23) indicates a directional trend where
'Intentional' incidents are more likely to be Financial/Privacy-related (40%)
compared to Unintentional ones (17%), the result is not statistically
significant. The analysis highlights a data limitation: the AIID dataset
primarily documents accidental/unintentional failures, making it a poor fit for
studying malicious use patterns without supplementary data.

---

## Review

The experiment pipeline was executed successfully and faithfully followed the
research plan. All datasets were correctly loaded, and the sparse CSV structure
was handled effectively to segment data for analysis. The three hypotheses were
tested with appropriate statistical rigor (Chi-square and Fisher's Exact tests),
and the findings provide significant insights into the nature of the data:

1. **Operational Governance Decay (Rejected)**: Contrary to the hypothesis,
'Operational' systems showed a significantly higher Impact Assessment compliance
rate (75.7%) compared to 'Development' systems (10.0%) (p < 0.001). This
indicates that governance controls function as a 'deployment gate' rather than
decaying post-launch.

2. **Autonomy-Safety Escalation (Rejected)**: The data did not support the claim
that higher autonomy leads to physical risks. Instead, High Autonomy systems
were overwhelmingly associated with 'Intangible' harms (86%), while 'Physical'
harms were more prevalent in Low Autonomy/Human-in-the-Loop contexts (p=0.077).

3. **Malice and Monetization (Not Supported)**: While there was a directional
trend suggesting Intentional incidents are more likely to involve
Financial/Privacy harms (Odds Ratio ~3.23), the result was not statistically
significant (p=0.23) due to the dataset's scarcity of confirmed 'Intentional'
malice cases (N=5).

Overall, the analysis was robust, correctly identifying that the dataset
primarily captures unintentional failures and that federal AI governance is
effectively front-loaded.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import re

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()

# Normalize columns
aiid.columns = [c.strip().lower().replace(' ', '_').replace(':', '').replace('.', '') for c in aiid.columns]

# Identify columns
intent_col = next((c for c in aiid.columns if 'intentional_harm' in c), None)

# Find text columns for keyword search (Title, Description, Summary, Reports)
# Common names in AIID: title, description, summary, reports
potential_text_cols = ['title', 'description', 'summary', 'reports', 'incident_description', 'short_description']
text_cols = [c for c in aiid.columns if any(x in c for x in potential_text_cols)]

print(f"Identified Intent Column: {intent_col}")
print(f"Identified Text Columns: {text_cols}")

# --- 1. Map Intent ---
def map_intent(val):
    if pd.isna(val): return None
    s = str(val).lower()
    if 'yes' in s and 'intentionally' in s:
        return 'Intentional'
    if 'no' in s and 'not intentionally' in s:
        return 'Unintentional'
    return None

aiid['intent_mapped'] = aiid[intent_col].apply(map_intent) if intent_col else None

# --- 2. Map Harm (Keyword Search) ---
# Keywords
financial_keywords = [
    'financial', 'money', 'bank', 'fraud', 'theft', 'scam', 'monetary', 'economic', 
    'credit', 'loan', 'cost', 'fund', 'wallet', 'crypto', 'currency', 'privacy', 
    'surveillance', 'leak', 'data breach', 'identity', 'spy', 'monitor', 'record'
]

physical_keywords = [
    'physical', 'safety', 'death', 'dead', 'kill', 'injury', 'injure', 'hurt', 'harm',
    'accident', 'crash', 'collision', 'hit', 'run over', 'medical', 'patient', 'hospital',
    'health', 'burn', 'explode', 'fire', 'attack', 'assault', 'robot', 'drone', 'autonomous'
]

def classify_harm(row):
    # Aggregate text from all available text columns
    text_content = " "
    for col in text_cols:
        val = row[col]
        if pd.notna(val):
            text_content += str(val) + " "
    
    text_lower = text_content.lower()
    
    has_financial = any(k in text_lower for k in financial_keywords)
    has_physical = any(k in text_lower for k in physical_keywords)
    
    if has_financial and not has_physical:
        return 'Financial/Privacy'
    elif has_physical and not has_financial:
        return 'Physical Safety'
    elif has_financial and has_physical:
        # Conflict resolution: Check for strong physical indicators (death/injury) vs generic 'safety'
        strong_physical = any(k in text_lower for k in ['death', 'dead', 'kill', 'injury', 'crash'])
        if strong_physical:
            return 'Physical Safety'
        else:
            return 'Ambiguous/Mixed'
    else:
        return 'Other'

aiid['harm_derived'] = aiid.apply(classify_harm, axis=1)

# --- 3. Analysis ---
# Filter for mapped intent and mapped harm (excluding Other/Ambiguous)
analysis_df = aiid.dropna(subset=['intent_mapped', 'harm_derived'])
analysis_df = analysis_df[analysis_df['harm_derived'].isin(['Financial/Privacy', 'Physical Safety'])]

print(f"\nTotal Mapped Rows: {len(analysis_df)}")
if len(analysis_df) > 0:
    print("Intent Breakdown:\n", analysis_df['intent_mapped'].value_counts())
    print("Harm Breakdown:\n", analysis_df['harm_derived'].value_counts())
    
    # Contingency Table
    contingency = pd.crosstab(analysis_df['intent_mapped'], analysis_df['harm_derived'])
    print("\nContingency Table:\n", contingency)
    
    # Check sample size for test selection
    total_obs = contingency.to_numpy().sum()
    min_expected = 0
    if contingency.shape == (2,2):
        chi2, p, dof, ex = chi2_contingency(contingency)
        min_expected = ex.min()
    
    if contingency.shape == (2,2):
        if min_expected < 5:
            print("\nSmall sample size detected. Using Fisher's Exact Test.")
            odds_ratio, p_val = fisher_exact(contingency)
            print(f"Fisher's Exact P-value: {p_val:.4e}")
            print(f"Odds Ratio: {odds_ratio:.4f}")
        else:
            print("\nUsing Chi-Square Test.")
            print(f"Chi2 Statistic: {chi2:.4f}")
            print(f"P-value: {p:.4e}")
            
        # Interpretation
        if p_val < 0.05 if 'p_val' in locals() else p < 0.05:
            print("Result: Statistically Significant Association.")
            # Calculate Row Percentages to see direction
            row_pcts = contingency.div(contingency.sum(axis=1), axis=0) * 100
            print("\nRow Percentages:\n", row_pcts)
        else:
            print("Result: No Statistically Significant Association.")
    else:
        print("Contingency table is not 2x2. Cannot perform binary association test.")
else:
    print("No valid data points found after filtering.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Identified Intent Column: intentional_harm
Identified Text Columns: ['reports', 'description', 'title', 'summary']

Total Mapped Rows: 81
Intent Breakdown:
 intent_mapped
Unintentional    76
Intentional       5
Name: count, dtype: int64
Harm Breakdown:
 harm_derived
Physical Safety      66
Financial/Privacy    15
Name: count, dtype: int64

Contingency Table:
 harm_derived   Financial/Privacy  Physical Safety
intent_mapped                                    
Intentional                    2                3
Unintentional                 13               63

Small sample size detected. Using Fisher's Exact Test.
Fisher's Exact P-value: 2.2925e-01
Odds Ratio: 3.2308
Result: No Statistically Significant Association.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
