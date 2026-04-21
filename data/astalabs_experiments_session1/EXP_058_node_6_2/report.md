# Experiment 58: node_6_2

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_2` |
| **ID in Run** | 58 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:20:44.232631+00:00 |
| **Runtime** | 211.6s |
| **Parent** | `node_5_4` |
| **Children** | None |
| **Creation Index** | 59 |

---

## Hypothesis

> The 'Human-in-the-Loop Approval Gates' architecture control is significantly
more likely to be required by 'Trust Readiness' competencies, whereas
'Nondeterminism Controls' are significantly more associated with 'Integration
Readiness'.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.3760 (Maybe False) |
| **Surprise** | -0.4247 |
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
| Maybe False | 90.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Validate the structural alignment of specific controls (Human-centric vs. Technical) to their respective bundles.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Create two binary flags: 'Requires_HITL' (if 'Human-in-the-Loop Approval Gates' == 'X') and 'Requires_Nondeterminism' (if 'Nondeterminism Controls & Output Validation' == 'X').
- 3. Create a contingency table for 'bundle' vs. 'Requires_HITL' and another for 'bundle' vs. 'Requires_Nondeterminism'.
- 4. Perform Chi-square tests for both associations.

### Deliverables
- Contingency tables, Chi-square test results for both controls, and a grouped bar chart showing the percentage of requirements in each bundle triggering these controls.

---

## Analysis

The experiment successfully analyzed the relationship between specific
architecture controls and competency bundles using 'step2_crosswalk_matrix.csv'.

**Statistical Results:**
1. **Human-in-the-Loop (HITL) Approval Gates:**
   - **Distribution:** Integration Readiness (3/23, 13.0%) vs. Trust Readiness
(0/19, 0%).
   - **Test:** Chi-square p-value = 0.3022.
   - **Result:** The difference is not statistically significant due to the low
sample size, but the direction contradicts the hypothesis (HITL appeared
exclusively in Integration, not Trust).

2. **Nondeterminism Controls:**
   - **Distribution:** Integration Readiness (6/23, 26.1%) vs. Trust Readiness
(0/19, 0%).
   - **Test:** Chi-square p-value = 0.0498.
   - **Result:** The difference is statistically significant (p < 0.05).
Nondeterminism controls are exclusively and significantly associated with
Integration Readiness.

**Visual Analysis:**
The bar chart visually confirms the findings, displaying bars only for the
'Integration Readiness' category. The 'Trust Readiness' category is empty for
both controls, indicating these specific technical implementations are mapped
entirely to the engineering/architecture side of the framework, rather than the
governance/policy side.

---

## Review

The experiment pipeline was faithfully implemented, successfully loading and
analyzing the complete set of 18 datasets. The final step specifically tested
the structural alignment of 'Human-in-the-Loop' (HITL) and 'Nondeterminism'
controls.

**Results & Findings:**
1.  **Hypothesis Partial Rejection (HITL):** The hypothesis that HITL controls
align with 'Trust Readiness' was rejected. The data showed HITL controls are
exclusively mapped to 'Integration Readiness' (3 occurrences) rather than 'Trust
Readiness' (0 occurrences), though the difference was not statistically
significant (p=0.30). This suggests the framework treats HITL as a technical
engineering implementation rather than a pure governance policy.
2.  **Hypothesis Confirmation (Nondeterminism):** The hypothesis that
'Nondeterminism Controls' align with 'Integration Readiness' was supported.
These controls appeared exclusively in the Integration bundle (6 occurrences,
26%) with statistical significance (p=0.05).
3.  **Overall Pipeline Synthesis:** The holistic review reveals a clear
dichotomy in the 'Strategic AI Orientation' framework. 'Trust Readiness' appears
restricted to high-level governance/policy (as evidenced by the lack of
technical control mapping), while 'Integration Readiness' encompasses the actual
architectural enforcement of those policies (including HITL and Nondeterminism).
Furthermore, the MITRE ATLAS validation data (Step 3) demonstrated that real-
world failures are overwhelmingly 'Security' focused and 'Prevention' failures,
lacking the nuance to distinguish between these theoretical bundles in practice.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

# Define file name
file_name = 'step2_crosswalk_matrix.csv'

# Try to locate the file (current dir or parent dir)
if os.path.exists(file_name):
    file_path = file_name
elif os.path.exists(os.path.join('..', file_name)):
    file_path = os.path.join('..', file_name)
else:
    # Fallback to absolute path check or list dir for debugging if needed, 
    # but for now assume it's in the current dir based on previous success.
    file_path = file_name

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}")
    
    # Target columns
    col_hitl = 'Human-in-the-Loop Approval Gates'
    col_nondeter = 'Nondeterminism Controls & Output Validation'
    
    # Clean and create binary flags (assuming 'X' indicates presence)
    # We treat NaN as 0 (False) and 'X' (or any non-empty string) as 1 (True)
    df['Requires_HITL'] = df[col_hitl].fillna('').apply(lambda x: 1 if str(x).strip().upper() == 'X' else 0)
    df['Requires_Nondeterminism'] = df[col_nondeter].fillna('').apply(lambda x: 1 if str(x).strip().upper() == 'X' else 0)
    
    # --- Analysis 1: Human-in-the-Loop Approval Gates ---
    print(f"\n=== Analysis: {col_hitl} ===")
    contingency_hitl = pd.crosstab(df['bundle'], df['Requires_HITL'])
    print("Contingency Table (Bundle vs HITL):")
    print(contingency_hitl)
    
    chi2_hitl, p_hitl, dof_hitl, ex_hitl = stats.chi2_contingency(contingency_hitl)
    print(f"Chi-square statistic: {chi2_hitl:.4f}, p-value: {p_hitl:.4f}")
    
    # --- Analysis 2: Nondeterminism Controls ---
    print(f"\n=== Analysis: {col_nondeter} ===")
    contingency_nd = pd.crosstab(df['bundle'], df['Requires_Nondeterminism'])
    print("Contingency Table (Bundle vs Nondeterminism):")
    print(contingency_nd)
    
    chi2_nd, p_nd, dof_nd, ex_nd = stats.chi2_contingency(contingency_nd)
    print(f"Chi-square statistic: {chi2_nd:.4f}, p-value: {p_nd:.4f}")
    
    # --- Visualization ---
    # Calculate percentage of requirements in each bundle that have the control
    # Group by bundle, calculate mean of binary flag, multiply by 100
    summary = df.groupby('bundle')[['Requires_HITL', 'Requires_Nondeterminism']].mean() * 100
    
    print("\nPercentage of Requirements triggering control per Bundle:")
    print(summary)
    
    ax = summary.plot(kind='bar', figsize=(10, 6), rot=0, color=['skyblue', 'lightgreen'])
    plt.title('Association of Controls with Competency Bundles')
    plt.ylabel('Percentage of Requirements (%)')
    plt.xlabel('Competency Bundle')
    plt.legend(['Human-in-the-Loop (HITL)', 'Nondeterminism Controls'])
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        if height > 0:
            ax.annotate(f'{height:.1f}%', 
                        (x + width/2, y + height + 1), 
                        ha='center')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Successfully loaded step2_crosswalk_matrix.csv

=== Analysis: Human-in-the-Loop Approval Gates ===
Contingency Table (Bundle vs HITL):
Requires_HITL           0  1
bundle                      
Integration Readiness  20  3
Trust Readiness        19  0
Chi-square statistic: 1.0646, p-value: 0.3022

=== Analysis: Nondeterminism Controls & Output Validation ===
Contingency Table (Bundle vs Nondeterminism):
Requires_Nondeterminism   0  1
bundle                        
Integration Readiness    17  6
Trust Readiness          19  0
Chi-square statistic: 3.8484, p-value: 0.0498

Percentage of Requirements triggering control per Bundle:
                       Requires_HITL  Requires_Nondeterminism
bundle                                                       
Integration Readiness      13.043478                26.086957
Trust Readiness             0.000000                 0.000000


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Grouped Bar Chart (or Clustered Bar Plot).
*   **Purpose:** The plot compares the percentage of requirements associated with two specific types of controls (Human-in-the-Loop and Nondeterminism Controls) across different categories of "Competency Bundles."

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Competency Bundle"
    *   **Labels:** "Integration Readiness" and "Trust Readiness".
*   **Y-Axis:**
    *   **Title:** "Percentage of Requirements (%)"
    *   **Range:** The axis runs from 0 to 100.
    *   **Ticks:** Major tick marks are placed at intervals of 20 (0, 20, 40, 60, 80, 100). Horizontal grid lines correspond to these ticks.

### 3. Data Trends
*   **Integration Readiness:**
    *   This category contains all the visible data in the plot.
    *   **Tallest Bar:** The "Nondeterminism Controls" (green) bar is the tallest, reaching a value of 26.1%.
    *   **Shortest Bar:** The "Human-in-the-Loop (HITL)" (blue) bar is shorter, reaching a value of 13.0%.
*   **Trust Readiness:**
    *   There are no visible bars for this category. This indicates that the values are either 0% or the data is missing for both control types regarding "Trust Readiness."

### 4. Annotations and Legends
*   **Legend:** Located in the top-right corner.
    *   **Blue Square:** Represents "Human-in-the-Loop (HITL)".
    *   **Green Square:** Represents "Nondeterminism Controls".
*   **Annotations:**
    *   Specific percentage values are written directly on top of the bars for clarity:
        *   **13.0%** above the HITL bar in the Integration Readiness group.
        *   **26.1%** above the Nondeterminism Controls bar in the Integration Readiness group.

### 5. Statistical Insights
*   **Dominance of Nondeterminism Controls:** Within the scope of "Integration Readiness," requirements related to Nondeterminism Controls (26.1%) are significantly more prevalent—approximately double—than those related to Human-in-the-Loop controls (13.0%).
*   **Lack of Association with Trust Readiness:** The complete absence of bars for "Trust Readiness" suggests that neither HITL nor Nondeterminism controls are associated with requirements in the Trust Readiness competency bundle in this dataset.
*   **Overall Focus:** The plot indicates that these specific control mechanisms are primarily relevant to the technical integration phase (Integration Readiness) rather than the trust-building phase (Trust Readiness).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
