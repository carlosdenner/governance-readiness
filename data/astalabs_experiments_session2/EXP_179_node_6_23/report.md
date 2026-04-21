# Experiment 179: node_6_23

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_23` |
| **ID in Run** | 179 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T09:33:13.575663+00:00 |
| **Runtime** | 174.5s |
| **Parent** | `node_5_46` |
| **Children** | `node_7_15` |
| **Creation Index** | 180 |

---

## Hypothesis

> The 'Deployment-Privacy' Tradeoff: Operational AI systems are significantly less
likely to document 'Opt-Out' mechanisms compared to systems in the 'Development'
phase, suggesting privacy controls are de-prioritized during deployment.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.5565 (Maybe True) |
| **Posterior** | 0.1896 (Likely False) |
| **Surprise** | -0.4403 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 19.0 |
| Uncertain | 0.0 |
| Maybe False | 10.0 |
| Definitely False | 1.0 |

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

**Objective:** Investigate if privacy controls like Opt-Out mechanisms are maintained as systems move to production.

### Steps
- 1. Load 'eo13960_scored'.
- 2. Define 'Stage': 'Operation/Maintenance' vs 'Development/Planning' based on `16_dev_stage`.
- 3. Define 'Has_Opt_Out': `67_opt_out` (or `61_opt_out` depending on column index/name) == 'Yes' -> 1, else 0.
- 4. Calculate Opt-Out rates per Stage.
- 5. Run Chi-square test.

### Deliverables
- Opt-Out rates by Lifecycle Stage; Chi-square statistics.

---

## Analysis

The experiment successfully tested the 'Deployment-Privacy Tradeoff' hypothesis
using the EO 13960 dataset. The code correctly segmented the data into
'Development' (n=626) and 'Operational' (n=627) lifecycle phases and analyzed
the documentation of 'Opt-Out' mechanisms.

The results contradict the hypothesis: Operational systems were significantly
*more* likely to have documented 'Opt-Out' mechanisms (8.61%, 54 systems)
compared to Development systems (3.35%, 21 systems). The Chi-square test
(statistic=14.47, p<0.001) confirms this difference is statistically
significant. Instead of privacy controls being de-prioritized during deployment,
the data suggests these controls are more likely to be finalized and documented
in operational systems than in those still under development. The hypothesis is
therefore rejected.

---

## Review

The experiment was faithfully implemented and successfully tested the
'Deployment-Privacy Tradeoff' hypothesis. The code correctly loaded the EO 13960
dataset, segmented systems into 'Development' (n=626) and 'Operational' (n=627)
phases, and analyzed the presence of documented 'Opt-Out' mechanisms.

Contrary to the hypothesis that privacy controls decay in deployment, the
results showed that 'Operational' systems were significantly *more* likely to
have documented 'Opt-Out' mechanisms (8.61%) compared to systems in
'Development' (3.35%). The Chi-square test (p < 0.001) confirmed this difference
was statistically significant. The hypothesis is therefore rejected; the data
suggests that privacy controls like opt-out mechanisms are more frequently
finalized or documented in deployed systems rather than during the development
phase.

---

## Code

```python
import pandas as pd
from scipy.stats import chi2_contingency
import os

# Define file path based on instructions
file_name = 'astalabs_discovery_all_data.csv'
file_path = f'../{file_name}'

if not os.path.exists(file_path):
    # Fallback to current directory if not found in parent
    if os.path.exists(file_name):
        file_path = file_name
    else:
        print(f"Error: {file_name} not found in current or parent directory.")
        exit(1)

print(f"Loading dataset from {file_path}...")
try:
    df = pd.read_csv(file_path, low_memory=False)
except Exception as e:
    print(f"Failed to load CSV: {e}")
    exit(1)

# Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 subset shape: {df_eo.shape}")

# Lifecycle Categorization
def categorize_stage(stage):
    if pd.isna(stage):
        return None
    stage_str = str(stage).lower()
    if 'operation' in stage_str or 'maintenance' in stage_str:
        return 'Operational'
    if 'development' in stage_str or 'implementation' in stage_str or 'planning' in stage_str:
        return 'Development'
    return None

df_eo['lifecycle_phase'] = df_eo['16_dev_stage'].apply(categorize_stage)

# Drop rows where phase could not be determined
df_eo = df_eo.dropna(subset=['lifecycle_phase'])

# Opt-Out Categorization
# Column '67_opt_out' is the target. Identify correct column name if slight variation exists.
target_col = '67_opt_out'
if target_col not in df_eo.columns:
    # Try to find it
    possible_cols = [c for c in df_eo.columns if 'opt_out' in c.lower()]
    if possible_cols:
        target_col = possible_cols[0]
        print(f"Warning: Exact column '{target_col}' found and used instead of '67_opt_out'.")
    else:
        print("Error: '67_opt_out' column not found.")
        exit(1)

# Create binary variable: 1 if 'Yes', 0 otherwise
df_eo['has_opt_out'] = df_eo[target_col].astype(str).str.strip().str.lower().apply(lambda x: 1 if x == 'yes' else 0)

# Generate Summary Statistics
summary = df_eo.groupby('lifecycle_phase')['has_opt_out'].agg(['count', 'sum', 'mean'])
summary['percent'] = summary['mean'] * 100

print("\n--- Opt-Out Documentation Rates by Lifecycle Phase ---")
print(summary)

# Contingency Table for Chi-Square
contingency_table = pd.crosstab(df_eo['lifecycle_phase'], df_eo['has_opt_out'])

# Ensure table has columns for both 0 and 1
for c in [0, 1]:
    if c not in contingency_table.columns:
        contingency_table[c] = 0
contingency_table = contingency_table[[0, 1]]

print("\n--- Contingency Table (0=No, 1=Yes) ---")
print(contingency_table)

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\n--- Chi-Square Test Results ---")
print(f"Chi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

if p < 0.05:
    print("Conclusion: Statistically significant difference detected.")
else:
    print("Conclusion: No statistically significant difference detected.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
EO 13960 subset shape: (1757, 196)

--- Opt-Out Documentation Rates by Lifecycle Phase ---
                 count  sum      mean   percent
lifecycle_phase                                
Development        626   21  0.033546  3.354633
Operational        627   54  0.086124  8.612440

--- Contingency Table (0=No, 1=Yes) ---
has_opt_out        0   1
lifecycle_phase         
Development      605  21
Operational      573  54

--- Chi-Square Test Results ---
Chi-square Statistic: 14.4683
P-value: 1.4254e-04
Conclusion: Statistically significant difference detected.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
