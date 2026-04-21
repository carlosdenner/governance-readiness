# Experiment 152: node_5_55

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_55` |
| **ID in Run** | 152 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:12:06.320403+00:00 |
| **Runtime** | 250.7s |
| **Parent** | `node_4_18` |
| **Children** | `node_6_33` |
| **Creation Index** | 153 |

---

## Hypothesis

> Autonomy-Failure Divergence: AI systems with 'High' autonomy levels are
significantly more correlated with 'Robustness' failures, whereas 'Low' autonomy
systems correlate with 'Operator Error', reflecting distinct failure modes.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7411 (Likely True) |
| **Posterior** | 0.2358 (Likely False) |
| **Surprise** | -0.6063 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 27.0 |
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

**Objective:** Link Autonomy Levels to specific Technical Failure types in incident data.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` (AIID subset).
- 2. Clean `Autonomy Level` (High vs Low/Human-in-the-loop) and `Known AI Technical Failure`.
- 3. Categorize failures into 'Robustness' (e.g., adversarial, distribution shift) and 'Operator' (e.g., use error, configuration) using keyword mapping.
- 4. Perform a Chi-Square test of independence.

### Deliverables
- Stacked bar chart of Failure Types by Autonomy Level; Chi-Square test results.

---

## Analysis

The experiment successfully mapped the specific AIID autonomy codes ('Autonomy3'
to High, 'Autonomy1/2' to Low) and categorized incidents into 'Robustness' vs.
'Operator' failures. The analysis revealed that 'Robustness Failures'
overwhelmingly dominate the dataset regardless of autonomy level. High Autonomy
systems showed 100% Robustness failures (17 cases), while Low Autonomy systems
showed 96.9% Robustness failures (63 cases) and only 3.1% Operator errors (2
cases).

The Chi-Square test yielded a p-value of 1.0, indicating no statistically
significant difference in the distribution of failure types between the two
groups. The hypothesis that Low Autonomy systems would correlate significantly
with Operator Error is **not supported**; the data suggests that technical
robustness issues are the primary failure mode for AI systems across all levels
of autonomy in this dataset.

---

## Review

The experiment was executed successfully and faithfully followed the plan. The
programmer effectively utilized the insights from the previous failed attempt to
correctly map the specific autonomy codes ('Autonomy1'/'Autonomy2' to Low,
'Autonomy3' to High) and implemented a robust keyword matching strategy for
technical failures.

The analysis **rejects** the hypothesis that 'Low' autonomy systems correlate
significantly with 'Operator Error'.

Key Findings:
1.  **Dominance of Robustness Failures**: Regardless of autonomy level,
'Robustness Failures' (e.g., generalization errors, adversarial attacks)
accounted for the overwhelming majority of incidents (100% in High Autonomy,
96.9% in Low Autonomy).
2.  **Scarcity of Operator Error**: Only 2 incidents were classified as
'Operator Error' within the 'Low Autonomy' group, compared to 63 'Robustness
Failures'.
3.  **No Statistical Difference**: The Chi-Square test yielded a p-value of
1.000, confirming that the distribution of failure types is statistically
identical across autonomy levels in this dataset.

This suggests that the AIID dataset is heavily skewed towards reporting
technical system failures rather than human usage errors, or that 'Operator
Error' is not a distinct failure mode but rather a rare antecedent to technical
failures in this taxonomy.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load the dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback for local testing or different directory structure if needed, though instruction is explicit
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# --- 1. Map Autonomy Level ---
# Based on AIID taxonomy: Autonomy1 = Human-in-the-loop, Autonomy2 = Human-on-the-loop, Autonomy3 = Human-out-of-the-loop (High)
# Hypothesis groups "Low/Human-in-the-loop" vs "High"
def map_autonomy_code(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    
    if val_str == 'Autonomy3':
        return 'High Autonomy'
    elif val_str in ['Autonomy1', 'Autonomy2']:
        return 'Low Autonomy'
    else:
        return np.nan

aiid_df['Autonomy_Class'] = aiid_df['Autonomy Level'].apply(map_autonomy_code)

# --- 2. Categorize Technical Failures ---
def map_failure_type_refined(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower()
    
    # Extended Robustness Keywords
    robustness_keys = [
        'robustness', 'adversarial', 'distribution', 'drift', 'generalization', 
        'model error', 'algorithm', 'prediction', 'classification', 
        'precision', 'recall', 'accuracy', 'reliability', 'bias',
        'context', 'misidentification', 'hallucination', 'generation hazard', 
        'unsafe exposure' # Often relates to model outputting unsafe content (robustness/alignment)
    ]
    
    # Operator/Human Error Keywords
    operator_keys = [
        'operator', 'human', 'user', 'configuration', 'setup', 'mistake', 
        'accidental', 'process', 'procedure'
    ]
    
    is_robustness = any(k in val_str for k in robustness_keys)
    is_operator = any(k in val_str for k in operator_keys)
    
    if is_robustness and not is_operator:
        return 'Robustness Failure'
    elif is_operator and not is_robustness:
        return 'Operator Error'
    elif is_robustness and is_operator:
        return 'Mixed/Ambiguous'
    else:
        return 'Other'

aiid_df['Failure_Category'] = aiid_df['Known AI Technical Failure'].apply(map_failure_type_refined)

# --- 3. Prepare Analysis Dataframe ---
# Filter for defined Autonomy and relevant Failure Categories
analysis_df = aiid_df.dropna(subset=['Autonomy_Class', 'Failure_Category'])

print("\n--- Failure Category Distribution (Before Filtering for Hypothesis) ---")
print(analysis_df['Failure_Category'].value_counts())

# Filter for specific hypothesis categories
final_df = analysis_df[analysis_df['Failure_Category'].isin(['Robustness Failure', 'Operator Error'])]

print("\n--- Final Analysis Dataset Summary ---")
print(final_df.groupby(['Autonomy_Class', 'Failure_Category']).size())

# --- 4. Statistical Test (Chi-Square) ---
contingency_table = pd.crosstab(final_df['Autonomy_Class'], final_df['Failure_Category'])

if contingency_table.empty or contingency_table.shape != (2, 2):
    print("\nInsufficient data for 2x2 Chi-Square test after filtering.")
    print("Contingency Table:\n", contingency_table)
else:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print("\n--- Chi-Square Test Results ---")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Visualization
    contingency_pct = contingency_table.div(contingency_table.sum(1), axis=0) * 100
    ax = contingency_pct.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title('Distribution of Failure Types by Autonomy Level')
    plt.xlabel('Autonomy Level')
    plt.ylabel('Percentage of Incidents (%)')
    plt.legend(title='Failure Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center')
    plt.tight_layout()
    plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: 
--- Failure Category Distribution (Before Filtering for Hypothesis) ---
Failure_Category
Robustness Failure    80
Other                 41
Mixed/Ambiguous        3
Operator Error         2
Name: count, dtype: int64

--- Final Analysis Dataset Summary ---
Autonomy_Class  Failure_Category  
High Autonomy   Robustness Failure    17
Low Autonomy    Operator Error         2
                Robustness Failure    63
dtype: int64

--- Chi-Square Test Results ---
Chi-Square Statistic: 0.0000
P-value: 1.0000e+00

STDERR:
<ipython-input-1-24126a99e51e>:98: Pandas4Warning: Starting with pandas version 4.0 all arguments of sum will be keyword-only.
  contingency_pct = contingency_table.div(contingency_table.sum(1), axis=0) * 100


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Stacked Bar Chart.
*   **Purpose:** This plot compares the proportional distribution of two specific failure types ("Operator Error" and "Robustness Failure") across two distinct categories of system autonomy ("High Autonomy" and "Low Autonomy"). The stacking allows the viewer to see the part-to-whole relationship, where the total height represents 100% of incidents for that category.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Autonomy Level"
    *   **Labels:** Two categorical labels are presented vertically: "High Autonomy" and "Low Autonomy".
*   **Y-Axis:**
    *   **Title:** "Percentage of Incidents (%)"
    *   **Range:** The axis spans from 0 to slightly over 100, with tick marks at intervals of 20 (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Dominant Trend:** "Robustness Failure" (represented in yellow) is the overwhelmingly dominant failure mode in both autonomy settings.
*   **High Autonomy:**
    *   **Robustness Failure:** Comprises the entirety of the incidents (tallest segment).
    *   **Operator Error:** Is non-existent in this category (shortest segment/non-visible).
*   **Low Autonomy:**
    *   **Robustness Failure:** Still comprises the vast majority of incidents.
    *   **Operator Error:** Represents a very small fraction of the total incidents.

### 4. Annotations and Legends
*   **Legend:** Located on the top right, titled "Failure Type".
    *   **Purple Square:** Represents "Operator Error".
    *   **Yellow Square:** Represents "Robustness Failure".
*   **Annotations:**
    *   **High Autonomy Bar:** Labeled with "100.0%" in the yellow section and "0.0%" at the baseline.
    *   **Low Autonomy Bar:** Labeled with "96.9%" in the yellow section and "3.1%" in the purple section.
*   **Title:** The chart is titled "Distribution of Failure Types by Autonomy Level".

### 5. Statistical Insights
*   **Elimination of Operator Error:** The most significant insight is that moving from "Low Autonomy" to "High Autonomy" completely eliminates "Operator Error" in this dataset, dropping from 3.1% to 0.0%.
*   **Persistence of System Fragility:** Regardless of the autonomy level, the system's own robustness is the primary bottleneck. Robustness failures account for nearly all incidents (96.9% to 100%), suggesting that while automation removes human error, it does not necessarily solve intrinsic system reliability issues.
*   **Impact of Autonomy:** The transition to high autonomy effectively shifts the failure profile entirely onto the system itself, removing the human operator as a variable in failure causation.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
