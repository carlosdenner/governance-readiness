# Experiment 26: node_3_12

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_12` |
| **ID in Run** | 26 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T02:15:04.138691+00:00 |
| **Runtime** | 355.9s |
| **Parent** | `node_2_0` |
| **Children** | `node_4_11`, `node_4_23` |
| **Creation Index** | 27 |

---

## Hypothesis

> Autonomy-Failure Alignment: Incidents involving 'High Autonomy' systems are
statistically correlated with 'Specification/Objective' failures, whereas 'Low
Autonomy' systems correlate with 'Human-Computer Interaction' failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.4203 (Maybe False) |
| **Surprise** | -0.3956 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 29.0 |
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

**Objective:** Map the relationship between AI autonomy levels and technical failure modes.

### Steps
- 1. Load `aiid_incidents` subset.
- 2. Encode `Autonomy Level` into ordinal or binary (Low/High) categories.
- 3. Clean `Known AI Technical Failure` into high-level categories (e.g., 'Robustness', 'Specification', 'Human-Interaction').
- 4. Generate a Cross-tabulation of Autonomy vs. Failure Mode.
- 5. Perform a Chi-Square test and visualize with a stacked bar chart.

### Deliverables
- Cross-tabulation table; Chi-Square statistics; Stacked bar chart of failure modes by autonomy level.

---

## Analysis

The experiment successfully mapped autonomy levels and failure modes for 91 AIID
incidents, utilizing specific schema mappings (e.g., 'Autonomy3' to High,
'Autonomy1' to Low) identified in previous steps. The Chi-Square test (p=0.517)
indicated no statistically significant relationship between Autonomy Level and
Failure Mode.

The hypothesis was only partially consistent with the data directionally: High
Autonomy systems did exhibit a higher proportion of 'Specification' failures
(32.0%) compared to Low Autonomy systems (21.2%), suggesting that
objective/alignment definition becomes more critical as autonomy increases.
However, the second part of the hypothesis failed: Low Autonomy systems did not
show a higher prevalence of 'Human-Interaction' failures (10.6%) compared to
High Autonomy (12.0%). Instead, 'Robustness' failures were the dominant mode
across both categories (56% for High, 68% for Low), indicating that fundamental
capability and reliability issues currently outweigh autonomy-specific failure
dynamics in this dataset.

---

## Review

The experiment was successfully executed. The code correctly implemented the
refined mapping logic to handle the specific schema of the 'aiid_incidents'
dataset (e.g., mapping 'Autonomy3' to High Autonomy), resulting in a usable
subset of 91 incidents for analysis. The statistical analysis (Chi-Square
p=0.517) and cross-tabulation revealed that the hypothesis was largely
unsupported. While High Autonomy systems did exhibit a higher proportion of
'Specification' failures (32.0%) compared to Low Autonomy (21.2%)—directionally
aligning with the first part of the hypothesis—the second part regarding Low
Autonomy and 'Human-Interaction' failures failed to materialize (10.6% vs
12.0%). Instead, 'Robustness' failures proved to be the dominant failure mode
across all autonomy levels (>50%), suggesting that fundamental reliability
issues currently overshadow autonomy-specific failure dynamics in this dataset.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

# Load the dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    print("Dataset not found at astalabs_discovery_all_data.csv")
    sys.exit(1)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID Incidents Loaded: {len(aiid)}")

# --- Define Mapping Functions based on previous debug output ---

def map_autonomy(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    
    # Based on observed values: Autonomy1, Autonomy2, Autonomy3
    # Mapping Assumption: Autonomy1 = Low/Assisted, Autonomy3 = High/Autonomous
    if s in ['Autonomy3', 'Autonomy4', 'Autonomy5', 'High', 'Full Autonomy']:
        return 'High Autonomy'
    if s in ['Autonomy0', 'Autonomy1', 'Low', 'Assisted', 'No Autonomy']:
        return 'Low Autonomy'
        
    # Fallback for string matching if exact match fails
    s_lower = s.lower()
    if 'autonomy3' in s_lower or 'high' in s_lower:
        return 'High Autonomy'
    if 'autonomy1' in s_lower or 'low' in s_lower:
        return 'Low Autonomy'
        
    return None

def map_failure(val):
    if pd.isna(val):
        return None
    s = str(val).lower().strip()
    
    # Robustness: Failures of capability/reliability/security
    if any(x in s for x in ['robustness', 'generalization', 'context', 'gaming', 'attack', 'adversarial', 'distributional', 'reliability', 'hardware']):
        return 'Robustness'
        
    # Specification: Alignment issues, wrong objective, harmful generation (hallucination/misinfo often fit here in CSET taxonomy)
    if any(x in s for x in ['specification', 'objective', 'goal', 'alignment', 'misinformation', 'harmful', 'unsafe', 'bias']):
        # Note: Bias is tricky, but often categorized as spec/alignment in broader governance contexts if not explicitly 'fairness'
        return 'Specification'
        
    # Human-Interaction: Operator error, transparency, use error
    if any(x in s for x in ['human', 'operator', 'interaction', 'user', 'mistake', 'transparency']):
        return 'Human-Interaction'
        
    return 'Other'

# --- Apply Mappings ---
aiid['Autonomy_Category'] = aiid['Autonomy Level'].apply(map_autonomy)
aiid['Failure_Category'] = aiid['Known AI Technical Failure'].apply(map_failure)

# --- Filter for Analysis ---
analysis_df = aiid[
    (aiid['Autonomy_Category'].notna()) &
    (aiid['Failure_Category'].isin(['Robustness', 'Specification', 'Human-Interaction']))
].copy()

print(f"\nData points after filtering: {len(analysis_df)}")
print("Autonomy distribution:")
print(analysis_df['Autonomy_Category'].value_counts())
print("Failure distribution:")
print(analysis_df['Failure_Category'].value_counts())

if len(analysis_df) < 5:
    print("Not enough data to perform Chi-Square test.")
else:
    # --- Cross-Tabulation ---
    ct = pd.crosstab(analysis_df['Autonomy_Category'], analysis_df['Failure_Category'])
    print("\n--- Cross-Tabulation (Counts) ---")
    print(ct)

    # --- Chi-Square Test ---
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")

    # --- Visualization ---
    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    print("\n--- Row Proportions (Normalized) ---")
    print(ct_norm)

    # Plot
    ax = ct_norm.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Distribution of Failure Modes by Autonomy Level', fontsize=14)
    plt.xlabel('Autonomy Level', fontsize=12)
    plt.ylabel('Proportion of Incidents', fontsize=12)
    plt.legend(title='Failure Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # --- Hypothesis Check ---
    # H1: High Autonomy -> Specification
    # H2: Low Autonomy -> Human-Interaction
    
    try:
        high_spec = ct_norm.loc['High Autonomy', 'Specification'] if 'Specification' in ct_norm.columns else 0
        low_hci = ct_norm.loc['Low Autonomy', 'Human-Interaction'] if 'Human-Interaction' in ct_norm.columns else 0
        
        print(f"\nHigh Autonomy -> Specification: {high_spec:.1%}")
        print(f"Low Autonomy -> Human-Interaction: {low_hci:.1%}")
        
        if high_spec > 0.3 and low_hci > 0.3:
             print("Result: Hypothesis Supported (Trends match expectations)")
        else:
             print("Result: Hypothesis Not Supported (Trends do not match expectations)")
    except Exception as e:
        print(f"Could not validate hypothesis completely: {e}")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total AIID Incidents Loaded: 1362

Data points after filtering: 91
Autonomy distribution:
Autonomy_Category
Low Autonomy     66
High Autonomy    25
Name: count, dtype: int64
Failure distribution:
Failure_Category
Robustness           59
Specification        22
Human-Interaction    10
Name: count, dtype: int64

--- Cross-Tabulation (Counts) ---
Failure_Category   Human-Interaction  Robustness  Specification
Autonomy_Category                                              
High Autonomy                      3          14              8
Low Autonomy                       7          45             14

Chi-Square Statistic: 1.3199
P-value: 5.1688e-01

--- Row Proportions (Normalized) ---
Failure_Category   Human-Interaction  Robustness  Specification
Autonomy_Category                                              
High Autonomy               0.120000    0.560000       0.320000
Low Autonomy                0.106061    0.681818       0.212121

High Autonomy -> Specification: 32.0%
Low Autonomy -> Human-Interaction: 10.6%
Result: Hypothesis Not Supported (Trends do not match expectations)


=== Plot Analysis (figure 1) ===
Based on the provided image, here is a detailed analysis of the plot:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** This plot is designed to compare the relative distribution (proportion) of three distinct categories of failure modes across two different groups (High Autonomy vs. Low Autonomy). The stacked nature allows the viewer to see the part-to-whole relationship for each group, where the total height always sums to 1.0 (100%).

### 2. Axes
*   **X-axis:**
    *   **Label:** "Autonomy Level"
    *   **Categories:** The axis represents categorical data with two groups: "High Autonomy" and "Low Autonomy".
*   **Y-axis:**
    *   **Label:** "Proportion of Incidents"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100% of the incidents).
    *   **Ticks:** The axis is marked at intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
*   **Dominant Category:** For both autonomy levels, the **Robustness** failure mode (orange) constitutes the largest proportion of incidents, taking up more than 50% of the bar in both cases.
*   **High Autonomy:**
    *   **Specification (Green):** Occupies a significant portion of the top segment (visually estimated at ~32%, from y=0.68 to 1.0).
    *   **Robustness (Orange):** Occupies the middle segment (visually estimated at ~56%, from y=0.12 to 0.68).
    *   **Human-Interaction (Blue):** The smallest segment at the bottom (visually estimated at ~12%).
*   **Low Autonomy:**
    *   **Specification (Green):** This segment is visibly smaller than in the High Autonomy bar (visually estimated at ~21%, from y=0.79 to 1.0).
    *   **Robustness (Orange):** This segment appears larger here than in the High Autonomy bar (visually estimated at ~68%, from y=0.11 to 0.79).
    *   **Human-Interaction (Blue):** Remains roughly consistent with the High Autonomy level, representing a small fraction (visually estimated at ~11%).

### 4. Annotations and Legends
*   **Title:** "Distribution of Failure Modes by Autonomy Level" is displayed at the top center.
*   **Legend:** Located at the top right, titled "Failure Mode," categorizing the colors:
    *   **Blue:** Human-Interaction
    *   **Orange:** Robustness
    *   **Green:** Specification

### 5. Statistical Insights
*   **Robustness is critical regardless of autonomy:** Robustness issues are the primary driver of failure incidents in both high and low autonomy systems.
*   **Trade-off between Specification and Robustness:** There is an inverse relationship visible between Specification and Robustness failures. As autonomy decreases (High to Low), the proportion of Specification failures decreases, while the proportion of Robustness failures increases. This suggests that "High Autonomy" systems are more prone to errors in how tasks are specified/defined, whereas "Low Autonomy" systems are more susceptible to robustness issues (likely environmental or operational stability).
*   **Consistency of Human Interaction:** The proportion of failures attributed to Human-Interaction remains relatively stable and low across both levels. This implies that the level of autonomy (High vs. Low) does not significantly alter the *proportion* of errors caused by human interaction relative to other failure types.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
