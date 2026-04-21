# Experiment 159: node_5_61

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_61` |
| **ID in Run** | 159 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:35:13.463655+00:00 |
| **Runtime** | 302.5s |
| **Parent** | `node_4_11` |
| **Children** | `node_6_35`, `node_6_87` |
| **Creation Index** | 160 |

---

## Hypothesis

> Public-Facing Transparency Correlation: AI systems marked as 'Publicly
Accessible' are statistically more likely to implement 'Opt-Out' mechanisms than
internal-only systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2527 (Likely False) |
| **Surprise** | -0.5870 |
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
| Maybe False | 0.0 |
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Verify if public visibility drives the implementation of user agency controls.

### Steps
- 1. Load 'eo13960_scored'.
- 2. Define 'Public Facing' using column '26_public_service' (Yes/No) or '27_public_info'.
- 3. Define 'Has Opt-Out' using column '67_opt_out' (Yes/No).
- 4. Run a Chi-Square test to see if Public Facing systems have a higher rate of Opt-Out compliance.
- 5. Calculate the relative risk or likelihood ratio.

### Deliverables
- Contingency table; Chi-Square p-value; Bar chart showing Opt-Out rates for Public vs Internal systems.

---

## Analysis

The experiment was successfully executed and produced statistically significant
results that refute the initial hypothesis. By implementing robust string
parsing to handle the descriptive text fields in the 'eo13960_scored' dataset,
the code successfully extracted a valid sample of 121 AI systems. The Chi-Square
test yielded a p-value of 2.41e-06, indicating a strong correlation between
system visibility and opt-out availability. However, the direction of this
relationship is the inverse of the hypothesis: Internal-facing systems
demonstrated a significantly higher opt-out rate (81.9%) compared to Public-
facing systems (36.8%). This suggests that while internal users (likely federal
employees) retain agency, public-facing systems often lack opt-out mechanisms,
possibly due to their mandatory nature (e.g., law enforcement, border security,
benefit adjudication) or statutory limitations identified in the data.

---

## Review

The experiment was successfully executed. The programmer correctly implemented
the feedback to use robust string parsing, allowing for the extraction of 121
valid data points from the descriptive text fields in the 'eo13960_scored'
dataset. The analysis was rigorous, employing a Chi-Square test (p < 0.001) and
calculating relative risk. The results refute the original hypothesis: contrary
to the expectation that public visibility drives user agency, Internal systems
demonstrated a significantly higher Opt-Out rate (81.9%) compared to Public
systems (36.8%). The generated visualization clearly communicates this inverse
relationship.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

def run_experiment():
    # Load dataset
    file_path = '../astalabs_discovery_all_data.csv'
    if not os.path.exists(file_path):
        file_path = 'astalabs_discovery_all_data.csv'
    
    print(f"Loading dataset from: {file_path}")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        return

    # Filter for eo13960_scored
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"Subset 'eo13960_scored' loaded. Rows: {len(df_eo)}")

    # Columns
    col_svc = '26_public_service'
    col_info = '27_public_info'
    col_opt = '67_opt_out'

    # Robust Cleaning Functions
    def is_public_indicator(val):
        # Logic: If it contains descriptive text (longer than 'No'), it's likely a public use case description.
        # If it is 'No', 'N/A', or empty, it is not.
        if pd.isna(val):
            return False
        s = str(val).strip()
        if not s:
            return False
        s_lower = s.lower()
        if s_lower in ['no', 'n/a', 'none', 'false', '0']:
            return False
        # If it's a long string description (e.g., 'Enabling trusted travelers...'), it implies Yes.
        return True

    def parse_opt_out(val):
        if pd.isna(val):
            return None
        s = str(val).strip().lower()
        if s.startswith('yes'):
            return 'Yes'
        if s.startswith('no') or s.startswith('n/a') or 'waived' in s:
            return 'No'
        return None # 'Other' or ambiguous

    # Apply cleaning
    df_eo['is_public_svc'] = df_eo[col_svc].apply(is_public_indicator)
    df_eo['is_public_info'] = df_eo[col_info].apply(is_public_indicator)
    
    # Define Visibility
    # Public if either service description or info description is present
    df_eo['visibility'] = np.where(df_eo['is_public_svc'] | df_eo['is_public_info'], 'Public', 'Internal')

    # Parse Opt-Out
    df_eo['has_opt_out'] = df_eo[col_opt].apply(parse_opt_out)

    # Filter valid data
    df_clean = df_eo.dropna(subset=['has_opt_out'])
    print(f"Rows with valid Opt-Out status: {len(df_clean)}")
    
    # Contingency Table
    ct = pd.crosstab(df_clean['visibility'], df_clean['has_opt_out'])
    print("\nContingency Table:")
    print(ct)

    if ct.empty or ct.shape != (2, 2):
        print("Warning: Contingency table is not 2x2. Check data distribution.")
        # If 2x2 is not formed, fill missing cols/rows with 0 for robust plotting if possible
        for v in ['Public', 'Internal']:
            if v not in ct.index: ct.loc[v] = [0, 0]
        for c in ['No', 'Yes']:
            if c not in ct.columns: ct[c] = 0
        ct = ct.loc[['Internal', 'Public'], ['No', 'Yes']]
        print("Adjusted Table:")
        print(ct)

    # Statistics
    chi2, p, dof, ex = stats.chi2_contingency(ct)
    print(f"\nChi-Square: {chi2:.4f}, p-value: {p:.4e}")

    # Calculate Opt-Out Rates
    try:
        rate_public = ct.loc['Public', 'Yes'] / ct.loc['Public'].sum()
        rate_internal = ct.loc['Internal', 'Yes'] / ct.loc['Internal'].sum()
        print(f"Opt-Out Rate (Public): {rate_public:.2%}")
        print(f"Opt-Out Rate (Internal): {rate_internal:.2%}")
        
        rr = rate_public / rate_internal if rate_internal > 0 else float('inf')
        print(f"Relative Risk (Public/Internal): {rr:.2f}x")
    except Exception as e:
        print(f"Could not calculate rates: {e}")

    # Plot
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ax = ct_pct.plot(kind='barh', stacked=True, color=['#d9534f', '#5cb85c'], figsize=(10, 6))
    
    plt.title('Opt-Out Implementation by System Visibility')
    plt.xlabel('Percentage')
    plt.ylabel('Visibility')
    plt.axvline(50, color='gray', linestyle='--', alpha=0.5)
    
    # Annotate
    for c in ax.containers:
        # format labels, skip if 0
        labels = [f'{v:.1f}%' if v > 0 else '' for v in c.datavalues]
        ax.bar_label(c, labels=labels, label_type='center', color='white', weight='bold')
    
    plt.legend(title='Has Opt-Out?', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: astalabs_discovery_all_data.csv
Subset 'eo13960_scored' loaded. Rows: 1757
Rows with valid Opt-Out status: 121

Contingency Table:
has_opt_out  No  Yes
visibility          
Internal     15   68
Public       24   14

Chi-Square: 22.2372, p-value: 2.4096e-06
Opt-Out Rate (Public): 36.84%
Opt-Out Rate (Internal): 81.93%
Relative Risk (Public/Internal): 0.45x


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Horizontal 100% Stacked Bar Chart.
*   **Purpose:** This chart compares the proportional distribution of a binary variable ("Has Opt-Out?") across two different categories of system visibility ("Internal" and "Public"). It is designed to visualize how the presence of an opt-out mechanism varies depending on whether a system is internal or public.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Percentage"
    *   **Range:** 0 to 100 (representing 0% to 100%).
    *   **Increments:** Ticks are marked every 20 units (0, 20, 40, 60, 80, 100).
*   **Y-Axis:**
    *   **Label:** "Visibility"
    *   **Categories:** The axis displays two categorical groups: "Internal" (bottom) and "Public" (top).
*   **Title:** "Opt-Out Implementation by System Visibility"

### 3. Data Trends
*   **Internal Systems:**
    *   This category shows a high adoption rate for opt-out mechanisms. The **"Yes"** segment (Green) is the dominant portion, accounting for **81.9%**.
    *   The **"No"** segment (Red) is the minority, representing only **18.1%**.
*   **Public Systems:**
    *   This category shows the inverse trend. The **"No"** segment (Red) is the dominant portion at **63.2%**.
    *   The **"Yes"** segment (Green) represents the minority at **36.8%**.
*   **Overall Pattern:** There is a stark contrast between the two categories. Internal systems are overwhelmingly likely to have an opt-out implementation, whereas public systems are more likely *not* to have one.

### 4. Annotations and Legends
*   **Legend:** Located in the top right corner with the title "Has Opt-Out?".
    *   **Red:** Represents "No".
    *   **Green:** Represents "Yes".
*   **Data Labels:** Each bar segment contains white text indicating the exact percentage value (e.g., "18.1%", "81.9%").
*   **Reference Line:** A grey, vertical dashed line runs through the x-axis at the **50% mark**. This serves as a visual anchor to easily distinguish which segment constitutes the majority for each bar.

### 5. Statistical Insights
*   **Visibility Correlation:** System visibility is a strong predictor of opt-out implementation. There is a significant discrepancy between the two groups; the "Yes" rate drops by **45.1 percentage points** (from 81.9% to 36.8%) when moving from Internal to Public systems.
*   **Majority Inversion:** The 50% threshold line highlights a complete inversion of the majority. For Internal systems, the majority (approx. 4 out of 5) allow opting out. For Public systems, the majority (approx. 2 out of 3) do *not* allow opting out.
*   **Implication:** This suggests that privacy controls or user agency (via opt-outs) are prioritized or technically easier to enforce within internal-facing systems compared to public-facing ones.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
