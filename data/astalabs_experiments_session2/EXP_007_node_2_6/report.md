# Experiment 7: node_2_6

| Property | Value |
|---|---|
| **Experiment ID** | `node_2_6` |
| **ID in Run** | 7 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T01:20:31.779028+00:00 |
| **Runtime** | 186.0s |
| **Parent** | `node_1_0` |
| **Children** | `node_3_6`, `node_3_13`, `node_3_20` |
| **Creation Index** | 8 |

---

## Hypothesis

> The Privacy-Control Gap: Among federal AI systems that explicitly process PII
(Personally Identifiable Information), a significant majority fail to provide an
'Opt-Out' mechanism, indicating a disconnect between privacy risk
acknowledgement and privacy control implementation.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9758 (Definitely True) |
| **Posterior** | 0.9918 (Definitely True) |
| **Surprise** | +0.0191 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 29.0 |
| Maybe True | 1.0 |
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

**Objective:** Quantify the gap between PII presence and Opt-Out control availability.

### Steps
- 1. Filter `eo13960_scored` for systems where `29_contains_pii` is 'Yes'.
- 2. Within this subset, analyze the distribution of `67_opt_out` (Yes/No).
- 3. Calculate the 'Privacy Control Gap' as the percentage of PII-processing systems with `67_opt_out` = 'No'.
- 4. Compare this gap across different agencies (`3_abr`) to see if specific agencies (e.g., DHS vs HHS) have better implementation rates using a Chi-square test.

### Deliverables
- Percentage of PII systems missing Opt-Out controls; Agency-level breakdown of the Privacy Control Gap.

---

## Analysis

The experiment successfully quantified the Privacy-Control Gap within the EO
13960 dataset. Out of 1,757 systems, the code identified 171 systems that
explicitly process PII. The analysis revealed a massive disconnect between
privacy risk and control implementation: only 13 of these 171 systems (7.6%)
provide an explicit 'Opt-Out' mechanism, resulting in a Privacy-Control Gap of
92.40%.

Agency-level breakdown for those with significant PII system counts showed
distinct variations (Chi-square p=0.021). HHS (0/14) and SSA (0/19) had a 100%
gap, offering no opt-out mechanisms for their PII-processing AI. The VA had a
94.7% gap, while DHS performed best among the large holders, though still with a
high gap of 81.4%. The analysis treated specific justifications for lacking opt-
out (e.g., law enforcement exceptions, waivers) as part of the gap, accurately
reflecting the unavailability of the control to the end-user.

---

## Review

The experiment successfully tested the hypothesis regarding the Privacy-Control
Gap. The implementation correctly handled the messy data in the `67_opt_out`
column by treating various 'No' justifications (e.g., waivers, law enforcement
exceptions) and 'N/A' responses as a lack of affirmative opt-out controls, which
is methodologically sound for quantifying a 'gap'. The analysis revealed that
among 171 federal AI systems processing PII, 92.4% fail to provide an opt-out
mechanism. The Chi-square test (p=0.02) confirmed that this gap is not uniform,
with agencies like HHS and SSA showing a 100% gap, while DHS performed
comparatively better (81% gap).

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import numpy as np

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    # Try one level up if not found in current directory as per hint, though instructions say "use dataset given"
    # Assuming standard path first based on previous context.
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

print("Dataset loaded successfully.")

# Filter for EO 13960 data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored subset shape: {df_eo.shape}")

# Normalize column names just in case (though previous output showed them clearly)
# Columns of interest: '29_contains_pii', '67_opt_out', '3_abr'

# Check unique values to ensure correct filtering
print("\nUnique values in '29_contains_pii':", df_eo['29_contains_pii'].unique())
print("Unique values in '67_opt_out':", df_eo['67_opt_out'].unique())

# Standardize values to boolean-like logic for analysis
# Assuming 'Yes' indicates presence. Adjusting for potential case sensitivity.
df_eo['has_pii'] = df_eo['29_contains_pii'].astype(str).str.strip().str.lower() == 'yes'
df_eo['has_opt_out'] = df_eo['67_opt_out'].astype(str).str.strip().str.lower() == 'yes'

# Filter for systems containing PII
df_pii = df_eo[df_eo['has_pii']].copy()
pii_count = len(df_pii)
print(f"\nSystems processing PII: {pii_count} (out of {len(df_eo)} total EO systems)")

if pii_count == 0:
    print("No PII systems found. Exiting analysis.")
else:
    # Calculate Privacy Control Gap (Percentage of PII systems missing Opt-Out)
    missing_opt_out_count = len(df_pii[~df_pii['has_opt_out']])
    privacy_control_gap = (missing_opt_out_count / pii_count) * 100

    print(f"\n--- Privacy-Control Gap Analysis ---")
    print(f"PII Systems with Opt-Out: {len(df_pii) - missing_opt_out_count}")
    print(f"PII Systems MISSING Opt-Out: {missing_opt_out_count}")
    print(f"Overall Privacy Control Gap: {privacy_control_gap:.2f}%")

    # Agency-level breakdown
    # Group by Agency ('3_abr')
    agency_col = '3_abr'
    
    # Create a crosstab of Agency vs Has_Opt_Out for PII systems only
    agency_stats = pd.crosstab(df_pii[agency_col], df_pii['has_opt_out'])
    agency_stats.columns = ['No_Opt_Out', 'Has_Opt_Out']  # False is No, True is Yes
    
    # Note: If all are True or all are False, crosstab might have 1 column. Handle this.
    if 'No_Opt_Out' not in agency_stats.columns:
        agency_stats['No_Opt_Out'] = 0
    if 'Has_Opt_Out' not in agency_stats.columns:
        agency_stats['Has_Opt_Out'] = 0
        
    agency_stats['Total_PII_Systems'] = agency_stats['No_Opt_Out'] + agency_stats['Has_Opt_Out']
    agency_stats['Gap_Percentage'] = (agency_stats['No_Opt_Out'] / agency_stats['Total_PII_Systems']) * 100
    
    # Filter for agencies with a meaningful number of PII systems (e.g., > 10) to reduce noise
    relevant_agencies = agency_stats[agency_stats['Total_PII_Systems'] >= 10].sort_values('Gap_Percentage', ascending=False)
    
    print("\n--- Agency-Level Breakdown (Agencies with >= 10 PII Systems) ---")
    print(relevant_agencies[['Total_PII_Systems', 'No_Opt_Out', 'Gap_Percentage']].round(2))

    # Chi-square test
    # We test if the distribution of Opt-Out (Yes/No) is independent of the Agency
    # Using the relevant agencies subset to ensure statistical validity
    if len(relevant_agencies) > 1:
        contingency_table = relevant_agencies[['No_Opt_Out', 'Has_Opt_Out']]
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        print(f"\n--- Chi-Square Test for Independence (Agency vs Opt-Out Availability) ---")
        print(f"Chi2 Statistic: {chi2:.4f}")
        print(f"P-value: {p:.4e}")
        if p < 0.05:
            print("Result: Statistically significant. Opt-out availability depends on the agency.")
        else:
            print("Result: Not statistically significant. Opt-out availability appears independent of agency.")
    else:
        print("\nInsufficient data for valid Chi-square test across agencies.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Dataset loaded successfully.
EO 13960 Scored subset shape: (1757, 196)

Unique values in '29_contains_pii': <StringArray>
['No', 'Yes', nan, ' ']
Length: 4, dtype: str
Unique values in '67_opt_out': <StringArray>
[                                                                                                                                                                                                                                            nan,
                                                                                                                                                                                                                                           'Yes',
 'No – This AI use case is not subject to the opt-out requirement because the AI functionality is solely used for the prevention, detection, and investigation of fraud or cybersecurity incidents, or the conduct of a criminal investigation.',
                                                                                                                                                                                                                                         'Other',
                                                                                                                                                        'No – There is law or governmentwide guidance that restricts opt-out for this context. ',
                                                                                                                                                   'N/A; COTS tool used for code conversion, no individual's information is input into a model.',
                                                                                                                                                                 'Agency CAIO has waived this minimum practice and reported such waiver to OMB.']
Length: 7, dtype: str

Systems processing PII: 171 (out of 1757 total EO systems)

--- Privacy-Control Gap Analysis ---
PII Systems with Opt-Out: 13
PII Systems MISSING Opt-Out: 158
Overall Privacy Control Gap: 92.40%

--- Agency-Level Breakdown (Agencies with >= 10 PII Systems) ---
       Total_PII_Systems  No_Opt_Out  Gap_Percentage
3_abr                                               
HHS                   14          14          100.00
SSA                   19          19          100.00
VA                    57          54           94.74
DHS                   43          35           81.40

--- Chi-Square Test for Independence (Agency vs Opt-Out Availability) ---
Chi2 Statistic: 9.7078
P-value: 2.1221e-02
Result: Statistically significant. Opt-out availability depends on the agency.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
