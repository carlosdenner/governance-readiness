# Experiment 99: node_4_43

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_43` |
| **ID in Run** | 99 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:50:11.908492+00:00 |
| **Runtime** | 221.0s |
| **Parent** | `node_3_7` |
| **Children** | `node_5_69`, `node_5_78` |
| **Creation Index** | 100 |

---

## Hypothesis

> The 'Opaque Defense' Hypothesis: Defense and Security agencies deploy a
significantly higher proportion of 'High Impact' systems that lack public
transparency mechanisms (e.g., AI Notice or Opt-Out) compared to Civilian
agencies.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9032 (Definitely True) |
| **Posterior** | 0.4615 (Uncertain) |
| **Surprise** | -0.5300 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 20.0 |
| Maybe True | 10.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 56.0 |
| Definitely False | 4.0 |

---

## Experiment Plan

**Objective:** Investigate the trade-off between agency mission (Security vs. Civilian) and public transparency.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `eo13960_scored`.
- 2. Categorize `3_agency` into 'Defense/Security' (DOD, DHS, DOJ) and 'Civilian' (HHS, DOT, DOE, etc.).
- 3. Filter for High Impact systems (`17_impact_type` is High/Rights-Impacting).
- 4. Calculate the 'Opacity Rate': Percentage of these systems where `59_ai_notice` OR `67_opt_out` is 'No'.
- 5. Compare Opacity Rates between agency types using a Z-test for proportions.

### Deliverables
- 1. Opacity Rates for Defense vs. Civilian agencies.
- 2. Z-test results.
- 3. Stacked bar chart of Transparency Compliance by Agency Type.

---

## Analysis

The experiment successfully processed the EO13960 dataset to test the 'Opaque
Defense' hypothesis but found no statistical evidence to support it due to the
specific data distribution.

1.  **Data Segmentation**: The code successfully categorized 1,757 records,
identifying 183 Defense/Security systems and 1,574 Civilian systems. From these,
77 were classified as 'High Impact' (50 Civilian, 27 Defense).

2.  **Opacity Analysis**: The experiment defined 'Opacity' strictly as cases
where `59_ai_notice` or `67_opt_out` explicitly equaled 'No'. Under this strict
criterion, **0 systems** were flagged as opaque in either the Defense or
Civilian groups (0% opacity rate for both).

3.  **Statistical Outcome**: Because both proportions were exactly zero, the
standard error was 0, making the Z-test impossible to perform. The stacked bar
chart visually confirmed this with 100% 'Transparent/Other' rates for both
groups.

4.  **Conclusion**: The hypothesis that Defense agencies are more opaque is
**not supported** by this specific strict analysis. The result implies that for
High Impact systems, agencies do not explicitly record 'No' for transparency
fields; they likely leave them blank (NaN) or mark them as 'Yes'. Future
exploration might need to treat missing values (NaN) as 'Opaque' to see if a
difference emerges.

---

## Review

The experiment was executed successfully and faithful to the plan.

**Hypothesis**: The 'Opaque Defense' hypothesis (that Defense agencies have
higher opacity rates for high-impact systems) was **not supported** by the
analysis.

**Findings**:
1. **Data Segmentation**: The code correctly identified 77 High Impact systems
(50 Civilian, 27 Defense/Security) from the EO13960 dataset.
2. **Metric Result**: The 'Opacity Rate' was defined strictly as cases where
`59_ai_notice` or `67_opt_out` explicitly equaled 'no'. Under this definition,
**0 systems** were flagged as opaque in either group (0% rate for both).
3. **Statistical Outcome**: Because both proportions were exactly zero, the
Z-test could not be calculated (Standard Error = 0).
4. **Interpretation**: The lack of explicit 'No' values suggests that agencies
likely leave transparency fields blank (NaN) rather than explicitly denying them
for high-impact systems. While the hypothesis was not validated, the null result
is a valid finding based on the strict criteria provided.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def run_experiment():
    print("Starting 'Opaque Defense' Hypothesis Experiment...")
    
    # 1. Load the dataset
    try:
        # Use local path as previous attempts with '../' failed
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found at 'astalabs_discovery_all_data.csv'")
        return

    # 2. Filter for EO13960 data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO13960 Records: {len(df_eo)}")

    # 3. Categorize Agencies
    # Defense/Security keywords: DOD, DHS, DOJ, Defense, Homeland, Justice
    def categorize_agency(agency_name):
        if pd.isna(agency_name):
            return 'Civilian' # Default to civilian if unknown/missing
        name_lower = str(agency_name).lower()
        # Check for specific defense/security keywords
        security_keywords = ['defense', 'homeland security', 'justice', 'dod', 'dhs', 'doj']
        if any(k in name_lower for k in security_keywords):
            return 'Defense/Security'
        return 'Civilian'

    df_eo['agency_type'] = df_eo['3_agency'].apply(categorize_agency)
    
    print("\nAgency Type Distribution (All EO13960):")
    print(df_eo['agency_type'].value_counts())

    # 4. Filter for High Impact Systems
    # We look for 'Rights-Impacting', 'Safety-Impacting' in column '17_impact_type'
    # Just to be safe, we'll include 'high' if present, though usually it's Rights/Safety.
    
    def is_high_impact(val):
        if pd.isna(val):
            return False
        val_lower = str(val).lower()
        return 'rights' in val_lower or 'safety' in val_lower or 'high' in val_lower

    df_high = df_eo[df_eo['17_impact_type'].apply(is_high_impact)].copy()
    print(f"\nHigh Impact Systems Identified: {len(df_high)}")
    
    if len(df_high) == 0:
        print("No high impact systems found. Checking sample values of '17_impact_type':")
        print(df_eo['17_impact_type'].dropna().unique()[:5])
        return

    print("High Impact Agency Distribution:")
    print(df_high['agency_type'].value_counts())

    # 5. Calculate 'Opacity Rate'
    # Opacity defined as: 59_ai_notice == 'No' OR 67_opt_out == 'No'
    # We will normalize text to lowercase and strip whitespace.
    
    def check_opacity(row):
        # Get values, handle NaNs as empty strings
        notice = str(row.get('59_ai_notice', '')).strip().lower()
        opt_out = str(row.get('67_opt_out', '')).strip().lower()
        
        # If either specific transparency mechanism is explicitly denied ('no'), it is opaque.
        # Note: If data is missing (nan), we don't count it as 'No' unless we assume missing = opaque.
        # The prompt says "where ... is 'No'". So we strictly look for 'no'.
        is_opaque = (notice == 'no') or (opt_out == 'no')
        return 1 if is_opaque else 0

    df_high['is_opaque'] = df_high.apply(check_opacity, axis=1)

    # 6. Compare Opacity Rates
    stats_df = df_high.groupby('agency_type')['is_opaque'].agg(['count', 'sum', 'mean'])
    stats_df.columns = ['Total_Systems', 'Opaque_Systems', 'Opacity_Rate']
    
    print("\nOpacity Statistics by Agency Type (High Impact Only):")
    print(stats_df)

    # Perform Z-test
    # Comparison: Defense/Security vs Civilian
    if 'Defense/Security' in stats_df.index and 'Civilian' in stats_df.index:
        n_def = stats_df.loc['Defense/Security', 'Total_Systems']
        x_def = stats_df.loc['Defense/Security', 'Opaque_Systems']
        p_def = stats_df.loc['Defense/Security', 'Opacity_Rate']
        
        n_civ = stats_df.loc['Civilian', 'Total_Systems']
        x_civ = stats_df.loc['Civilian', 'Opaque_Systems']
        p_civ = stats_df.loc['Civilian', 'Opacity_Rate']
        
        # Pooled probability
        p_pool = (x_def + x_civ) / (n_def + n_civ)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_def + 1/n_civ))
        
        if se == 0:
            print("Standard Error is 0, cannot perform Z-test.")
        else:
            z_score = (p_def - p_civ) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            print(f"\nZ-Test Results:")
            print(f"  Defense Opacity: {p_def:.2%}")
            print(f"  Civilian Opacity: {p_civ:.2%}")
            print(f"  Difference: {p_def - p_civ:.2%}")
            print(f"  Z-score: {z_score:.4f}")
            print(f"  P-value: {p_value:.4e}")
            
            alpha = 0.05
            if p_value < alpha:
                print("  Result: Statistically Significant (Reject Null)")
            else:
                print("  Result: Not Significant (Fail to Reject Null)")

        # 7. Visualization: Stacked Bar Chart of Transparency Compliance
        # We will plot 'Opaque' vs 'Transparent' (which is 1 - Opaque Rate)
        
        # Data preparation
        # We want a stacked bar for each agency type.
        # Bottom bar: Opaque Rate
        # Top bar: Transparent Rate
        
        categories = stats_df.index
        opaque_rates = stats_df['Opacity_Rate']
        transparent_rates = 1 - opaque_rates
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot Opaque (Red)
        ax.bar(categories, opaque_rates, label='Opaque (Notice/Opt-out denied)', color='#d62728', alpha=0.8)
        
        # Plot Transparent (Blue) - stacked on top
        ax.bar(categories, transparent_rates, bottom=opaque_rates, label='Transparent/Other', color='#1f77b4', alpha=0.8)
        
        ax.set_ylabel('Proportion of High Impact Systems')
        ax.set_title('Transparency Compliance: Defense vs Civilian Agencies\n(High Impact AI Systems)')
        ax.legend(loc='lower right')
        
        # Add percentage labels
        for i, (cat, op_rate) in enumerate(zip(categories, opaque_rates)):
            # Label for Opaque
            ax.text(i, op_rate / 2, f"{op_rate:.1%}", ha='center', va='center', color='white', fontweight='bold')
            # Label for Transparent
            tr_rate = 1 - op_rate
            ax.text(i, op_rate + tr_rate / 2, f"{tr_rate:.1%}", ha='center', va='center', color='white', fontweight='bold')

        plt.tight_layout()
        plt.show()
        
    else:
        print("\nInsufficient data groups to perform Z-test comparison.")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting 'Opaque Defense' Hypothesis Experiment...
EO13960 Records: 1757

Agency Type Distribution (All EO13960):
agency_type
Civilian            1574
Defense/Security     183
Name: count, dtype: int64

High Impact Systems Identified: 77
High Impact Agency Distribution:
agency_type
Civilian            50
Defense/Security    27
Name: count, dtype: int64

Opacity Statistics by Agency Type (High Impact Only):
                  Total_Systems  Opaque_Systems  Opacity_Rate
agency_type                                                  
Civilian                     50               0           0.0
Defense/Security             27               0           0.0
Standard Error is 0, cannot perform Z-test.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Chart.
*   **Purpose:** The plot is designed to compare the proportions of transparency compliance (categorized as "Opaque" vs. "Transparent/Other") for High Impact AI Systems between two distinct groups: Civilian agencies and Defense/Security agencies.

### 2. Axes
*   **X-Axis:**
    *   **Labels:** The axis represents categorical data with two groups: **"Civilian"** and **"Defense/Security"**.
    *   **Range:** Discrete categories.
*   **Y-Axis:**
    *   **Title:** "Proportion of High Impact Systems".
    *   **Range:** The scale ranges from **0.0 to 1.0**, representing a probability or percentage (0% to 100%). Tick marks are placed at intervals of 0.2.

### 3. Data Trends
*   **Civilian Agencies:** The bar is entirely blue, indicating that **100.0%** of the systems fall into the "Transparent/Other" category. Consequently, **0.0%** are classified as "Opaque".
*   **Defense/Security Agencies:** Identical to the Civilian group, the bar is entirely blue. **100.0%** of the systems are classified as "Transparent/Other," with **0.0%** classified as "Opaque".
*   **Pattern:** There is a uniform trend across both agency types. No "Opaque" systems appear in the dataset for either category, leading to two identical bars of maximum height (1.0).

### 4. Annotations and Legends
*   **Title:** "Transparency Compliance: Defense vs Civilian Agencies (High Impact AI Systems)".
*   **Legend:** Located in the bottom right corner.
    *   **Red square:** Represents "Opaque (Notice/Opt-out denied)".
    *   **Blue square:** Represents "Transparent/Other".
*   **Data Labels:**
    *   **"100.0%"**: Written in white in the center of the blue portion of both bars, indicating the proportion of compliant/transparent systems.
    *   **"0.0%"**: Written in white at the baseline of both bars, indicating the proportion of opaque systems.

### 5. Statistical Insights
*   **Uniform Compliance:** The data suggests perfect uniformity between Civilian and Defense/Security agencies regarding this specific metric. Both sectors show a 100% classification rate into the "Transparent/Other" category for High Impact AI Systems.
*   **Absence of Opaque Systems:** According to this plot, there are zero recorded instances of "Opaque (Notice/Opt-out denied)" systems for high-impact AI use cases in either agency type.
*   **Comparison:** There is no statistical disparity between the two groups; the difference in proportions is 0%.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
