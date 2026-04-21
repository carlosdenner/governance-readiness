# Experiment 259: node_6_65

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_65` |
| **ID in Run** | 259 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:19:53.734948+00:00 |
| **Runtime** | 265.4s |
| **Parent** | `node_5_34` |
| **Children** | None |
| **Creation Index** | 260 |

---

## Hypothesis

> The 'Automation-Overconfidence' Paradox: Systems flagged as fully 'Autonomous'
in the federal inventory are significantly less likely to have 'Key Risks'
identified compared to 'Human-Assisted' systems, suggesting an automation bias
where removing the human is mistakenly equated with removing risk.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7258 (Likely True) |
| **Posterior** | 0.3846 (Maybe False) |
| **Surprise** | -0.4094 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 28.0 |
| Uncertain | 2.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 50.0 |
| Definitely False | 10.0 |

---

## Experiment Plan

**Objective:** Investigate if autonomy correlates with a decrease in reported risk awareness.

### Steps
- 1. Filter for 'eo13960_scored'.
- 2. Parse `57_autonomous_impact` to categorize systems into 'Autonomous' (values indicating full automation) vs. 'Human-Assisted' (values indicating human-in-the-loop/audit/tool).
- 3. Parse `54_key_risks` to binary (1 if risks identified/text present, 0 if 'No', 'None', or NaN).
- 4. Compare risk identification rates using a t-test or Chi-square test.

### Deliverables
- Risk Identification rates by Autonomy Level; Statistical comparison.

---

## Analysis

The experiment successfully tested the 'Automation-Overconfidence' Paradox using
Fisher's Exact Test to account for the small sample size of fully autonomous
systems (n=7) versus human-assisted ones (n=143).

**Hypothesis Status:** Unsupported.

**Findings:**
1. **Direction:** Contrary to the hypothesis that autonomous systems would have
lower risk identification rates (implying overconfidence), they exhibited a
higher rate of documented risks (85.7%, 6/7) compared to human-assisted systems
(61.5%, 88/143).
2. **Significance:** The difference was not statistically significant (Fisher's
Exact Test p-value = 0.2578).

**Interpretation:** There is no evidence of an 'automation-overconfidence'
effect where risk documentation is neglected for autonomous systems. The trend
suggests the opposite—that fully autonomous systems may be subject to higher
scrutiny—but the extremely low prevalence of fully autonomous systems in the
federal inventory limits the statistical power of this conclusion.

---

## Review

The experiment successfully tested the 'Automation-Overconfidence' Paradox
hypothesis. Using Fisher's Exact Test to account for the small sample size of
fully autonomous systems (n=7), the analysis found no evidence to support the
claim that autonomous systems are less likely to have identified risks. In fact,
the observed trend contradicted the hypothesis, with autonomous systems showing
a higher rate of risk identification (85.7%) compared to human-assisted systems
(61.5%), though this difference was not statistically significant (p = 0.26).
The hypothesis is therefore not supported.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
import sys

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# Filter for EO 13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

col_autonomy = '57_autonomous_impact'
col_risks = '54_key_risks'

# --- Classification Logic ---

def classify_autonomy(val):
    s = str(val).lower().strip()
    if s == 'nan' or s == 'missing':
        return 'Unknown'
    
    # Autonomous: "Yes - All individual..." or "Other - Immediate human intervention is not practicable..."
    # Both imply the system acts without direct human control/intervention in the loop.
    if s.startswith('yes - all individual') or 'immediate human intervention is not practicable' in s:
        return 'Autonomous'
    
    # Human-Assisted: "No - Some individual decisions..."
    if s.startswith('no - some individual'):
        return 'Human-Assisted'
    
    return 'Unknown'

def classify_risk(val):
    # Treat NaN as 0 (No risk identified/documented)
    if pd.isna(val) or val == 'nan':
        return 0
    
    s = str(val).lower().strip()
    negative_terms = ['no', 'none', 'n/a', 'not applicable', 'none identified', '0', 'missing']
    
    # exact match check
    if s in negative_terms:
        return 0
    
    # distinct phrases check
    if 'no key risks identified' in s:
        return 0
    if s.startswith('n/a'):
        return 0
        
    # specific check for 'none.'
    if s == 'none.':
        return 0

    # If text exists and isn't a negative, assume risks are described/identified
    return 1

# Apply classification
eo_df['Autonomy_Class'] = eo_df[col_autonomy].apply(classify_autonomy)
eo_df['Risk_Flag'] = eo_df[col_risks].apply(classify_risk)

# Filter for analysis groups
analysis_df = eo_df[eo_df['Autonomy_Class'].isin(['Autonomous', 'Human-Assisted'])]

# --- Generate Statistics ---
print("--- Analysis of Automation-Overconfidence Paradox ---")

# Group counts
group_counts = analysis_df['Autonomy_Class'].value_counts()
print(f"\nSample Sizes:\n{group_counts}")

# Risk Identification Rates
risk_stats = analysis_df.groupby('Autonomy_Class')['Risk_Flag'].agg(['count', 'sum', 'mean'])
risk_stats.columns = ['Total', 'Risks_Identified', 'Rate']
print("\nRisk Identification Stats:")
print(risk_stats)

# Construct Contingency Table for Fisher's Exact Test
# Rows: Autonomous, Human-Assisted
# Cols: Risk Identified (1), Risk Not Identified (0)

auto_identified = risk_stats.loc['Autonomous', 'Risks_Identified']
auto_total = risk_stats.loc['Autonomous', 'Total']
auto_not = auto_total - auto_identified

human_identified = risk_stats.loc['Human-Assisted', 'Risks_Identified']
human_total = risk_stats.loc['Human-Assisted', 'Total']
human_not = human_total - human_identified

contingency_table = [[auto_identified, auto_not], [human_identified, human_not]]

print("\nContingency Table (Rows: Auto, Human; Cols: Identified, Not Identified):")
print(contingency_table)

# Fisher's Exact Test (Two-sided)
odds_ratio, p_value = fisher_exact(contingency_table, alternative='two-sided')

print(f"\nFisher's Exact Test Results:")
print(f"Odds Ratio: {odds_ratio:.4f}")
print(f"P-value: {p_value:.4e}")

alpha = 0.05
if p_value < alpha:
    print("Conclusion: Statistically significant difference found.")
else:
    print("Conclusion: No statistically significant difference found.")

# Interpretation helper
if risk_stats.loc['Autonomous', 'Rate'] < risk_stats.loc['Human-Assisted', 'Rate']:
    print("Direction: Autonomous systems have a LOWER risk identification rate.")
else:
    print("Direction: Autonomous systems have a HIGHER (or equal) risk identification rate.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Analysis of Automation-Overconfidence Paradox ---

Sample Sizes:
Autonomy_Class
Human-Assisted    143
Autonomous          7
Name: count, dtype: int64

Risk Identification Stats:
                Total  Risks_Identified      Rate
Autonomy_Class                                   
Autonomous          7                 6  0.857143
Human-Assisted    143                88  0.615385

Contingency Table (Rows: Auto, Human; Cols: Identified, Not Identified):
[[np.int64(6), np.int64(1)], [np.int64(88), np.int64(55)]]

Fisher's Exact Test Results:
Odds Ratio: 3.7500
P-value: 2.5777e-01
Conclusion: No statistically significant difference found.
Direction: Autonomous systems have a HIGHER (or equal) risk identification rate.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
