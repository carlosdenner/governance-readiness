# Experiment 153: node_5_56

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_56` |
| **ID in Run** | 153 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:16:19.657723+00:00 |
| **Runtime** | 260.9s |
| **Parent** | `node_4_34` |
| **Children** | `node_6_71` |
| **Creation Index** | 154 |

---

## Hypothesis

> The Automation Paradox: Systems classified as 'Autonomous' are significantly
more likely to have 'Post-Deployment Monitoring' in place compared to 'Human-in-
the-Loop' systems, suggesting over-reliance on human operators for non-
autonomous tools.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.4176 (Maybe False) |
| **Surprise** | -0.3892 |
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
| Maybe False | 60.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Determine if human-in-the-loop presence correlates with reduced automated monitoring.

### Steps
- 1. Load 'eo13960_scored'.
- 2. Parse '57_autonomous_impact'. Treat 'Yes' as Autonomous and 'No' as Human-in-the-Loop.
- 3. Parse '56_monitor_postdeploy' (Yes/No).
- 4. Create a 2x2 contingency table.
- 5. Perform a Chi-square test comparing monitoring rates.

### Deliverables
- Monitoring rates for Autonomous vs HITL systems; Chi-square statistic.

---

## Analysis

The experiment successfully tested the 'Automation Paradox' hypothesis after
resolving data mapping issues with verbose column values. The analysis
identified 145 analyzable systems: 139 'Human-in-the-Loop' (HITL) and only 6
'Autonomous'.

The results **fail to support the hypothesis** that autonomous systems are more
likely to be monitored:
1. **Observed Rates**: HITL systems showed a higher rate of post-deployment
monitoring (74.1%, 103/139) compared to Autonomous systems (50.0%, 3/6).
2. **Statistical Significance**: The Chi-Square test yielded a p-value of 0.405,
indicating the difference is not statistically significant, largely due to the
extremely small sample size of fully autonomous systems.
3. **implications**: The scarcity of 'Autonomous' classifications (n=6) suggests
that federal agencies rarely classify systems as fully autonomous or that such
systems are outliers. While the directional trend contradicts the hypothesis
(suggesting HITL systems actually receive *more* scrutiny), the lack of
statistical power prevents a definitive conclusion regarding the 'Automation
Paradox'.

---

## Review

The experiment successfully tested the 'Automation Paradox' hypothesis after
resolving data mapping issues. The initial attempt failed due to verbose column
values, but the second attempt correctly mapped these to binary categories. The
analysis identified 145 analyzable systems (139 Human-in-the-Loop, 6
Autonomous). The results **fail to support the hypothesis** that autonomous
systems are more likely to be monitored. In fact, the observed trend was the
opposite: Human-in-the-Loop systems had a higher monitoring rate (74.1%)
compared to Autonomous systems (50.0%), though the difference was not
statistically significant (p=0.405) due to the extremely small sample size of
fully autonomous deployments. The low count of autonomous systems (n=6) is a
significant finding in itself, suggesting federal agencies rarely classify
systems as fully autonomous.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import os

def map_autonomy(val):
    val = str(val).strip().lower()
    if val.startswith('yes'):
        return 'Autonomous'
    if val.startswith('other'):
        # "immediate human intervention is not practicable" implies autonomy
        return 'Autonomous'
    if val.startswith('no -'):
        return 'Human-in-the-Loop'
    return None

def map_monitoring(val):
    val = str(val).strip().lower()
    if 'no monitoring protocols' in val:
        return 'No'
    if 'under development' in val:
        return 'No'
    if 'intermittent' in val or 'automated' in val or 'established process' in val:
        return 'Yes'
    return None

def run_experiment():
    # Attempt to load the dataset
    file_path = 'astalabs_discovery_all_data.csv'
    if not os.path.exists(file_path):
        file_path = '../astalabs_discovery_all_data.csv'
    
    if not os.path.exists(file_path):
        print(f"Error: Dataset not found at {file_path}")
        return

    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)

    # Filter for 'eo13960_scored'
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO13960 Scored subset shape: {df_eo.shape}")

    # Apply Mappings
    df_eo['autonomy_category'] = df_eo['57_autonomous_impact'].apply(map_autonomy)
    df_eo['monitoring_category'] = df_eo['56_monitor_postdeploy'].apply(map_monitoring)

    # Drop NaNs in relevant columns
    df_analyzable = df_eo.dropna(subset=['autonomy_category', 'monitoring_category']).copy()

    print(f"Analyzable records after mapping: {len(df_analyzable)}")
    
    if len(df_analyzable) == 0:
        print("No valid data found for analysis after mapping.")
        return

    # Create Contingency Table
    # Rows: Autonomy (Autonomous, HITL)
    # Cols: Monitoring (No, Yes)
    crosstab = pd.crosstab(df_analyzable['autonomy_category'], df_analyzable['monitoring_category'])
    
    # Reorder for consistency if possible
    try:
        crosstab = crosstab.reindex(index=['Human-in-the-Loop', 'Autonomous'], columns=['No', 'Yes'])
    except:
        pass # Keep as is if keys missing
        
    print("\nContingency Table (Counts):")
    print(crosstab)

    # Descriptive Stats
    try:
        n_hitl = crosstab.loc['Human-in-the-Loop'].sum() if 'Human-in-the-Loop' in crosstab.index else 0
        k_hitl = crosstab.loc['Human-in-the-Loop', 'Yes'] if 'Human-in-the-Loop' in crosstab.index and 'Yes' in crosstab.columns else 0
        p_hitl = k_hitl / n_hitl if n_hitl > 0 else 0
        
        n_auto = crosstab.loc['Autonomous'].sum() if 'Autonomous' in crosstab.index else 0
        k_auto = crosstab.loc['Autonomous', 'Yes'] if 'Autonomous' in crosstab.index and 'Yes' in crosstab.columns else 0
        p_auto = k_auto / n_auto if n_auto > 0 else 0
        
        print("\n--- Descriptive Statistics ---")
        print(f"Human-in-the-Loop (n={n_hitl}): {k_hitl} monitored ({p_hitl:.2%})")
        print(f"Autonomous        (n={n_auto}): {k_auto} monitored ({p_auto:.2%})")
    except Exception as e:
        print(f"Error calculating stats: {e}")
        return

    # Chi-Square Test
    # We need a valid 2x2 table for this to be meaningful
    # Check for NaNs in the crosstab (reindex might introduce them)
    crosstab_filled = crosstab.fillna(0)
    
    if n_hitl > 0 and n_auto > 0:
        chi2, p_val, dof, expected = stats.chi2_contingency(crosstab_filled)
        print("\n--- Chi-Square Test Results ---")
        print(f"Chi2 Statistic: {chi2:.4f}")
        print(f"p-value: {p_val:.4e}")
        
        significant = p_val < 0.05
        print("\nConclusion:")
        if significant:
            print("Reject Null Hypothesis: Significant association detected.")
            if p_auto > p_hitl:
                print("Direction: Autonomous systems are MORE likely to be monitored.")
            else:
                print("Direction: Autonomous systems are LESS likely to be monitored.")
        else:
            print("Fail to Reject Null Hypothesis: No significant difference.")
            
        # Visualization
        plt.figure(figsize=(10, 6))
        categories = ['Human-in-the-Loop', 'Autonomous']
        proportions = [p_hitl, p_auto]
        
        bars = plt.bar(categories, proportions, color=['#1f77b4', '#d62728'], alpha=0.8)
        
        plt.ylabel('Proportion with Post-Deployment Monitoring')
        plt.title('The Automation Paradox: Monitoring Rates by Autonomy Level')
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        for bar, count, total in zip(bars, [k_hitl, k_auto], [n_hitl, n_auto]):
            height = bar.get_height()
            if total > 0:
                plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                         f"{height:.1%}\n(n={count}/{total})", 
                         ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.show()
        
    else:
        print("\nInsufficient data for comparison (missing one of the groups).")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: astalabs_discovery_all_data.csv
EO13960 Scored subset shape: (1757, 196)
Analyzable records after mapping: 145

Contingency Table (Counts):
monitoring_category  No  Yes
autonomy_category           
Human-in-the-Loop    36  103
Autonomous            3    3

--- Descriptive Statistics ---
Human-in-the-Loop (n=139): 103 monitored (74.10%)
Autonomous        (n=6): 3 monitored (50.00%)

--- Chi-Square Test Results ---
Chi2 Statistic: 0.6944
p-value: 4.0466e-01

Conclusion:
Fail to Reject Null Hypothesis: No significant difference.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Bar Chart.
*   **Purpose:** The plot compares the frequency (proportion) of post-deployment monitoring between two distinct categories of system autonomy: "Human-in-the-Loop" and "Autonomous".

**2. Axes**
*   **X-axis:**
    *   **Labels:** Categorical labels representing the level of autonomy: "Human-in-the-Loop" and "Autonomous".
*   **Y-axis:**
    *   **Title:** "Proportion with Post-Deployment Monitoring".
    *   **Value Range:** The axis ranges from 0.0 to roughly 1.05, with grid lines marking intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

**3. Data Trends**
*   **Tallest Bar:** The "Human-in-the-Loop" category (blue bar) is the tallest, indicating a higher rate of monitoring.
*   **Shortest Bar:** The "Autonomous" category (red bar) is shorter, indicating a lower rate of monitoring.
*   **Pattern:** There is a visible downward trend in monitoring rates as the level of autonomy increases (from human-in-the-loop to fully autonomous).

**4. Annotations and Legends**
*   **Data Labels:**
    *   Above the **Human-in-the-Loop** bar: Annotated with **"74.1% (n=103/139)"**. This indicates that 74.1% of the systems in this category are monitored, corresponding to 103 out of a total sample size of 139.
    *   Above the **Autonomous** bar: Annotated with **"50.0% (n=3/6)"**. This indicates that 50% of the systems in this category are monitored, corresponding to 3 out of a total sample size of 6.
*   **Grid:** Horizontal dashed grid lines are included to assist in estimating the bar heights against the Y-axis.
*   **Title:** The chart is titled "The Automation Paradox: Monitoring Rates by Autonomy Level," suggesting the data is meant to illustrate a counter-intuitive finding regarding automation and supervision.

**5. Statistical Insights**
*   **The "Paradox":** The title and data suggest an "Automation Paradox" where fully autonomous systems—which arguably might require rigorous safety checks—are monitored *less* frequently (50.0%) than systems that already have a human in the loop (74.1%).
*   **Sample Size Disparity:** A critical statistical insight is the vast difference in sample sizes ($n$). The "Human-in-the-Loop" group has a robust sample size of 139, whereas the "Autonomous" group has an extremely small sample size of only 6.
*   **Statistical Significance Warning:** Due to the very small sample size for the Autonomous group ($n=6$), the 50% figure is highly volatile; a change in the status of just one system would shift the percentage significantly (e.g., to 33% or 66%). Therefore, while the trend looks stark, the comparison lacks statistical power and should be interpreted with caution. The data also implies that fully autonomous deployments (in this dataset) are much rarer than human-in-the-loop deployments.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
