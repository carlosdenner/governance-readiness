# Experiment 276: node_7_14

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_14` |
| **ID in Run** | 276 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T14:06:46.525369+00:00 |
| **Runtime** | 219.9s |
| **Parent** | `node_6_9` |
| **Children** | None |
| **Creation Index** | 277 |

---

## Hypothesis

> Operational AI systems (in 'O&M' stage) are significantly less likely to comply
with 'AI Notice' requirements compared to systems in 'Implementation' or
'Development' stages, suggesting 'grandfathering' of legacy systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.9121 (Definitely True) |
| **Surprise** | +0.2042 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
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
| Definitely True | 60.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Determine if development stage impacts transparency compliance.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'eo13960_scored'.
- 2. Group '16_dev_stage' into 'Operational' (Operation and Maintenance) vs 'New/Dev' (Development, Implementation, Planning).
- 3. Clean '59_ai_notice' into a binary Yes/No.
- 4. Compare AI Notice compliance rates across stages using a Chi-Square test.

### Deliverables
- Contingency table; Bar chart of Notice compliance by Stage; Statistical test results.

---

## Analysis

The experiment successfully tested the hypothesis that operational AI systems
are less likely to comply with AI Notice requirements than those in development.
After cleaning the data to exclude waivers and non-applicable cases, 94 relevant
records remained. The analysis revealed a statistically significant difference
(Chi-Square p=0.0013). Systems in the 'New/Dev' stage demonstrated a high
compliance rate of 94.6% (35/37), whereas 'Operational' systems showed a
significantly lower compliance rate of 63.2% (36/57). This 31.4% gap supports
the hypothesis, suggesting that while new systems are being designed with
transparency in mind, legacy or active systems ('grandfathered' in) lag behind
in retroactive compliance.

---

## Review

The experiment successfully tested the hypothesis regarding the 'grandfathering'
of legacy AI systems w.r.t. transparency compliance.

**Methodology Assessment:**
The implementation correctly handled the data quality issues identified in the
first attempt. By grouping specific text values into 'Operational' vs 'New/Dev'
and filtering the 'AI Notice' field to exclude valid exemptions (e.g., 'N/A',
'Waived'), the study isolated the relevant population (n=94) where compliance is
expected.

**Findings:**
- **Statistical Significance:** The Chi-Square test yielded a p-value of 0.0013,
indicating a highly significant relationship between development stage and
compliance.
- **Effect Size:** There is a substantial gap in compliance: New/Dev systems
achieved 94.6% compliance, while Operational systems lagged significantly at
63.2%.

**Conclusion:**
The hypothesis is supported. The data suggests that while new federal AI systems
are largely adhering to transparency mandates during development, existing
operational systems face a 'compliance debt,' struggling to retroactively
implement AI Notice protocols.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback to parent directory as per instructions
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO13960 scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Total EO13960 records: {len(eo_df)}")

# 1. Map Development Stage
operational_list = ['Operation and Maintenance', 'In production', 'In mission']
new_dev_list = ['Implementation and Assessment', 'Acquisition and/or Development', 'Initiated', 'Planned']

def map_stage(x):
    if pd.isna(x):
        return None
    val = str(x).strip()
    if val in operational_list:
        return 'Operational'
    elif val in new_dev_list:
        return 'New/Dev'
    return None  # Excludes 'Retired' and others

eo_df['stage_group'] = eo_df['16_dev_stage'].apply(map_stage)

# 2. Map AI Notice
# We define Compliance (Yes) vs Non-Compliance (No).
# We exclude cases where the requirement is N/A or Waived.

# Based on previous output, specific exclusion strings:
exclusions = [
    'N/A - individuals are not interacting with the AI for this use case',
    'AI is not safety or rights-impacting.',
    'Agency CAIO has waived this minimum practice and reported such waiver to OMB.'
]

def map_notice(x):
    if pd.isna(x):
        return None
    val = str(x).strip()
    
    # Check for exclusions
    if val in exclusions:
        return None
    
    # Check for Non-Compliance
    if 'None of the above' in val:
        return 'No'
    
    # If it's not N/A and not 'None of the above', it implies some form of notice was selected
    # (e.g. 'Online', 'Email', 'In-person', 'Other')
    return 'Yes'

eo_df['notice_compliance'] = eo_df['59_ai_notice'].apply(map_notice)

# Create analysis dataframe
analysis_df = eo_df.dropna(subset=['stage_group', 'notice_compliance']).copy()

print(f"Records available for analysis after cleaning: {len(analysis_df)}")

if len(analysis_df) < 5:
    print("Insufficient data for analysis.")
else:
    # Contingency Table
    contingency = pd.crosstab(analysis_df['stage_group'], analysis_df['notice_compliance'])
    print("\nContingency Table (Stage vs Notice Compliance):")
    print(contingency)
    
    # Rates
    rates = pd.crosstab(analysis_df['stage_group'], analysis_df['notice_compliance'], normalize='index') * 100
    print("\nCompliance Rates (%):")
    print(rates)

    # Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    # Visualize
    plt.figure(figsize=(8, 6))
    
    # Extract 'Yes' rates for plotting
    if 'Yes' in rates.columns:
        yes_rates = rates['Yes']
    else:
        yes_rates = pd.Series([0, 0], index=['New/Dev', 'Operational'])
        
    # Ensure both categories exist in index for plotting consistency
    for cat in ['New/Dev', 'Operational']:
        if cat not in yes_rates.index:
            yes_rates[cat] = 0
            
    # Sort for consistent order
    yes_rates = yes_rates.sort_index()
    
    bars = plt.bar(yes_rates.index, yes_rates.values, color=['skyblue', 'salmon'])
    plt.title('AI Notice Compliance by Development Stage')
    plt.ylabel('Compliance Rate (%)')
    plt.ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total EO13960 records: 1757
Records available for analysis after cleaning: 94

Contingency Table (Stage vs Notice Compliance):
notice_compliance  No  Yes
stage_group               
New/Dev             2   35
Operational        21   36

Compliance Rates (%):
notice_compliance         No        Yes
stage_group                            
New/Dev             5.405405  94.594595
Operational        36.842105  63.157895

Chi-Square Test Results:
Chi2 Statistic: 10.3568
P-value: 0.0013


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Vertical Bar Chart (or Column Chart).
*   **Purpose:** The plot compares the percentage of compliance regarding "AI Notice" across two distinct categories of AI development stages ("New/Dev" and "Operational").

**2. Axes**
*   **Y-Axis (Vertical):**
    *   **Label:** "Compliance Rate (%)".
    *   **Units:** Percentage points.
    *   **Range:** The scale runs from 0 to 100, with major tick marks at intervals of 20 (0, 20, 40, 60, 80, 100).
*   **X-Axis (Horizontal):**
    *   **Label:** Although there is no collective axis title, the categories represent the "Development Stage."
    *   **Categories:** "New/Dev" and "Operational".

**3. Data Trends**
*   **Tallest Bar:** The "New/Dev" category (represented by the sky-blue bar) has the highest compliance rate.
*   **Shortest Bar:** The "Operational" category (represented by the salmon/light red bar) has a significantly lower compliance rate.
*   **Pattern:** There is a notable downward trend in compliance when moving from the development phase to the operational phase.

**4. Annotations and Legends**
*   **Chart Title:** "AI Notice Compliance by Development Stage" appears at the top center.
*   **Data Labels:** The exact percentage values are annotated directly above each bar:
    *   New/Dev: **94.6%**
    *   Operational: **63.2%**
*   **Color Coding:** The bars use distinct colors to differentiate the categories visually (Blue for New/Dev, Red/Salmon for Operational), though a separate legend is not required as the X-axis labels define the categories.

**5. Statistical Insights**
*   **High Compliance in Development:** Systems in the "New/Dev" stage show a very high compliance rate of **94.6%**, suggesting that new projects are likely being built with current compliance standards in mind (compliance by design).
*   **Compliance Gap:** There is a significant discrepancy of **31.4 percentage points** between the two stages.
*   **Operational Risk:** The "Operational" stage has a much lower compliance rate of **63.2%**. This suggests that legacy systems or AI models currently in production may not have been updated to meet new notice requirements, or that enforcement/checks are less rigorous after the initial development phase.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
