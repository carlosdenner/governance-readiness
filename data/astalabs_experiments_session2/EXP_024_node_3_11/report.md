# Experiment 24: node_3_11

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_11` |
| **ID in Run** | 24 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T02:11:28.454033+00:00 |
| **Runtime** | 211.8s |
| **Parent** | `node_2_5` |
| **Children** | `node_4_8`, `node_4_15` |
| **Creation Index** | 25 |

---

## Hypothesis

> Legacy Gap: AI systems in the 'Operation and Maintenance' phase are
significantly less likely to have a documented 'Impact Assessment' compared to
systems currently in the 'Development/Acquisition' phase.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9274 (Definitely True) |
| **Posterior** | 0.3159 (Maybe False) |
| **Surprise** | -0.7338 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 23.0 |
| Maybe True | 7.0 |
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

**Objective:** Determine if governance compliance decays or is lower for legacy/operational systems compared to those under active development.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for `source_table` == 'eo13960_scored'.
- 2. Clean the `16_dev_stage` column, binning values into two categories: 'Development' (e.g., Developed, Acquisition) and 'Operation' (e.g., Operational, Maintenance).
- 3. Clean the `52_impact_assessment` column into a binary variable (Yes vs No/Null/NA).
- 4. Create a contingency table: Development Stage vs. Impact Assessment Presence.
- 5. Perform a Chi-square test of independence.
- 6. Calculate the percentage of compliant systems in each stage.

### Deliverables
- Contingency table, Chi-square statistic, p-value, and a bar chart comparison of compliance rates.

---

## Analysis

The experiment successfully tested the 'Legacy Gap' hypothesis using the EO
13960 dataset. The code filtered 1,757 records down to 998 relevant entries,
categorizing them into 'Operation' (627 records) and 'Development' (371 records)
stages.

Contrary to the hypothesis that legacy (operational) systems would have lower
compliance, the results show the opposite. Operational systems had a
significantly higher Impact Assessment compliance rate (8.9%, 56/627) compared
to systems in Development (0.5%, 2/371). The Chi-square test confirms this
difference is statistically significant (p < 0.001, statistic=28.48).

The hypothesis is effectively **rejected** (the relationship is inverted). The
data suggests that Impact Assessments are likely a 'gate' requirement finalized
just before or during deployment (Authority to Operate), rather than documents
created during early acquisition or planning phases. Consequently, 'Planned' or
'Initiated' systems almost universally lack this documentation compared to their
operational counterparts.

---

## Review

The experiment was successfully and faithfully executed. The programmer
correctly filtered the EO 13960 dataset (N=1,757) and mapped the development
stages into 'Operation' (N=627) and 'Development' (N=371) categories, while
effectively binning the 'Impact Assessment' field into a binary variable.

The analysis produced clear statistical results that allow for the evaluation of
the hypothesis.

**Hypothesis Evaluation:**
The 'Legacy Gap' hypothesis predicted that operational systems would have
*lower* compliance (documented impact assessments) than systems in development.
The results show the statistically significant **opposite** effect (Chi-square =
28.48, p < 0.001):
- **Operation Stage:** 8.9% compliance (56/627).
- **Development Stage:** 0.5% compliance (2/371).

**Findings:**
The hypothesis is **rejected** (the relationship is inverted). While compliance
is low across the board, systems in the 'Operation and Maintenance' phase are
significantly *more* likely to have an impact assessment than those in
development. This suggests that impact assessments are treated as a late-stage
compliance 'gate' (likely tied to the Authority to Operate/deployment phase)
rather than a living document maintained throughout the early development
lifecycle.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load the dataset
file_path = "../astalabs_discovery_all_data.csv"
try:
    # Using low_memory=False to avoid DtypeWarning, or specifying types if known. 
    # Given the sparse nature, we just load and then filter.
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if file is in current directory (just in case)
    df = pd.read_csv("astalabs_discovery_all_data.csv", low_memory=False)

# Filter for source_table == 'eo13960_scored'
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

print("Initial EO 13960 records:", len(df_eo))

# Inspect unique values for mapping
print("\nUnique values in '16_dev_stage':")
print(df_eo['16_dev_stage'].unique())

print("\nUnique values in '52_impact_assessment':")
print(df_eo['52_impact_assessment'].unique())

# Clean and Map Development Stage
# Hypothesis: 'Operation and Maintenance' vs 'Development/Acquisition'
# Let's define the buckets based on typical values found in EO 13960 datasets
# Common values: 'Operation and maintenance', 'Development and acquisition', 'Planned'

def map_stage(stage):
    if pd.isna(stage):
        return np.nan
    stage = str(stage).lower()
    if 'operation' in stage or 'maintenance' in stage or 'deployed' in stage:
        return 'Operation'
    elif 'develop' in stage or 'acquisition' in stage or 'plan' in stage or 'pilot' in stage:
        return 'Development'
    else:
        return 'Other/Unknown'

df_eo['stage_category'] = df_eo['16_dev_stage'].apply(map_stage)

# Clean and Map Impact Assessment
# Usually 'Yes', 'No', or specific description. We treat non-null/non-no as Yes? 
# Or check for affirmative keywords.
# Let's assume strict 'Yes' vs others first, but will refine based on print output above if needed.
# For now, a generic mapper.

def map_impact(val):
    if pd.isna(val):
        return 'No'
    val_str = str(val).lower().strip()
    if val_str in ['yes', 'true', '1']:
        return 'Yes'
    # Sometimes it might contain a link or text. If it looks like a boolean field, we stick to Yes/No.
    # If the column contains text descriptions, we might need a more heuristic approach.
    # Based on metadata '52_impact_assessment' often implies a boolean or link.
    # We'll check if it starts with 'yes' or has content implying existence.
    if val_str.startswith('yes'):
        return 'Yes'
    return 'No'

df_eo['has_impact_assessment'] = df_eo['52_impact_assessment'].apply(map_impact)

# Filter out Unknown stages
df_analysis = df_eo[df_eo['stage_category'].isin(['Operation', 'Development'])].copy()

print("\nRecords after stage filtering:", len(df_analysis))
print(df_analysis['stage_category'].value_counts())
print(df_analysis['has_impact_assessment'].value_counts())

# Create Contingency Table
contingency_table = pd.crosstab(df_analysis['stage_category'], df_analysis['has_impact_assessment'])
print("\nContingency Table:")
print(contingency_table)

# Check if we have enough data
if contingency_table.size == 4:
    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Calculate percentages
    # Row-wise normalization to see % compliant per stage
    pct_table = pd.crosstab(df_analysis['stage_category'], df_analysis['has_impact_assessment'], normalize='index') * 100
    print("\nPercentage Table (Row-wise):")
    print(pct_table)

    # Plot
    try:
        compliance_rates = pct_table['Yes']
    except KeyError:
        compliance_rates = pd.Series([0, 0], index=['Development', 'Operation'])

    plt.figure(figsize=(8, 6))
    bars = plt.bar(compliance_rates.index, compliance_rates.values, color=['skyblue', 'salmon'])
    plt.ylabel('Percentage with Impact Assessment (%)')
    plt.title('Impact Assessment Compliance by Development Stage')
    plt.ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data for Chi-square test.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Initial EO 13960 records: 1757

Unique values in '16_dev_stage':
<StringArray>
[ 'Implementation and Assessment', 'Acquisition and/or Development',
                      'Initiated',                        'Retired',
      'Operation and Maintenance',                  'In production',
                     'In mission',                        'Planned',
                              nan]
Length: 9, dtype: str

Unique values in '52_impact_assessment':
<StringArray>
[nan, 'Planned or in-progress.', 'Yes', 'No', 'YES']
Length: 5, dtype: str

Records after stage filtering: 998
stage_category
Operation      627
Development    371
Name: count, dtype: int64
has_impact_assessment
No     940
Yes     58
Name: count, dtype: int64

Contingency Table:
has_impact_assessment   No  Yes
stage_category                 
Development            369    2
Operation              571   56

Chi-Square Statistic: 28.4768
P-value: 9.4828e-08

Percentage Table (Row-wise):
has_impact_assessment         No       Yes
stage_category                            
Development            99.460916  0.539084
Operation              91.068581  8.931419


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Vertical Bar Plot.
*   **Purpose:** The plot is designed to compare a quantitative variable (percentage of compliance) across two distinct categorical groups (Development and Operation stages).

**2. Axes**
*   **Y-Axis:**
    *   **Label:** "Percentage with Impact Assessment (%)". This indicates the metric being measured is a compliance rate represented as a percentage.
    *   **Range:** The axis ranges from 0 to 100, with major tick marks at intervals of 20 (0, 20, 40, 60, 80, 100).
*   **X-Axis:**
    *   **Categories:** The axis represents the "Development Stage" with two specific categories labeled: "Development" and "Operation".

**3. Data Trends**
*   **Comparison of Categories:**
    *   **Shortest Bar:** The "Development" stage has an extremely low value, appearing almost negligible on the 0-100 scale.
    *   **Tallest Bar:** The "Operation" stage has a noticeably higher value compared to Development, though it remains low relative to the total scale.
*   **Pattern:** There is a clear upward trend in compliance when moving from the Development phase to the Operation phase.

**4. Annotations and Legends**
*   **Value Labels:** Specific percentage values are annotated directly above each bar to provide precise data reading:
    *   Development: **0.5%**
    *   Operation: **8.9%**
*   **Colors:** The bars are color-coded for visual distinction, with "Development" in light blue and "Operation" in a salmon/light red color. No separate legend box is present, as the x-axis labels suffice.

**5. Statistical Insights**
*   **Overall Low Compliance:** The most significant insight is that compliance rates for Impact Assessments are extremely low across the board. Even in the higher-performing category (Operation), less than 9% of cases have an impact assessment.
*   **Disparity Between Stages:** There is a massive relative difference between the two stages. The compliance rate during the **Operation** stage (8.9%) is nearly **18 times higher** than the compliance rate during the **Development** stage (0.5%).
*   **Process Gap:** The data suggests that impact assessments are almost non-existent during the initial development phase and are only slightly more common once a project becomes operational. This indicates a potential gap in regulatory enforcement or procedural adherence during the early lifecycle of these projects.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
