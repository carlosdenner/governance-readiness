# Experiment 241: node_6_53

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_53` |
| **ID in Run** | 241 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T12:23:51.749964+00:00 |
| **Runtime** | 280.4s |
| **Parent** | `node_5_90` |
| **Children** | None |
| **Creation Index** | 242 |

---

## Hypothesis

> The 'Legacy Governance' Lag: AI systems implemented prior to the 2021 Executive
Order show significantly lower compliance with 'Bias Mitigation' controls
compared to systems implemented post-2021, indicating a lack of retroactive
governance.

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

**Objective:** Assess the temporal adoption of bias mitigation controls.

### Steps
- 1. Load `eo13960_scored`.
- 2. Parse `20_date_implemented` to extract the year.
- 3. Filter out invalid dates.
- 4. Split data into 'Pre-2021' and 'Post-2021' groups.
- 5. Parse `62_disparity_mitigation` into binary (1=Yes/Evidence, 0=No/None).
- 6. Compare rates using a Z-test for proportions.

### Deliverables
- Compliance rates over time (line chart or bar chart), Z-test results for the Pre/Post split.

---

## Analysis

The experiment successfully tested the 'Legacy Governance Lag' hypothesis using
606 AI systems with valid implementation dates from the EO 13960 inventory.

1. **Hypothesis Rejection**: The hypothesis posited that pre-2021 'Legacy'
systems would show significantly lower bias mitigation compliance than post-2021
'Modern' systems. The data **refutes** this. In fact, Pre-2021 systems showed a
slightly higher (though not statistically significant) compliance rate (13.01%)
compared to Post-2021 systems (11.39%).

2. **Statistical Evidence**: The Z-test yielded a p-value of 0.6178, confirming
that there is no statistically significant difference in bias mitigation rates
between the two eras.

3. **Governance Reality**: The analysis highlights a phenomenon of 'Governance
Stagnation' rather than a legacy lag. Despite the 2020/2021 regulatory focus,
modern systems are not outperforming older systems in documenting bias controls.
The temporal plot reveals a dip in 2020 followed by a recovery, but overall
compliance remains low (~11-13%) across both cohorts.

---

## Review

The experiment successfully tested the 'Legacy Governance Lag' hypothesis using
606 AI systems from the EO 13960 inventory with valid implementation dates.

1. **Hypothesis Rejection**: The hypothesis posited that pre-2021 'Legacy'
systems would show significantly lower bias mitigation compliance than post-2021
'Modern' systems. The data **refutes** this. Counter to expectation, Pre-2021
systems showed a slightly higher compliance rate (13.01%) compared to Post-2021
systems (11.39%).

2. **Statistical Evidence**: The Z-test yielded a p-value of 0.6178, confirming
that there is no statistically significant difference in bias mitigation rates
between the two eras. The null hypothesis cannot be rejected.

3. **Governance Reality**: The analysis highlights a phenomenon of 'Governance
Stagnation' rather than a legacy lag. Despite the 2020/2021 regulatory focus (EO
13960), modern systems are not outperforming older systems in documenting bias
controls. The temporal plot reveals a significant dip in 2020 (~3%) followed by
a recovery, but overall compliance remains low (~11-13%) across both cohorts.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
import os

# Ensure statsmodels is installed
try:
    from statsmodels.stats.proportion import proportions_ztest
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "statsmodels"])
    from statsmodels.stats.proportion import proportions_ztest

def run_experiment():
    try:
        # Attempt to find the dataset
        filename = 'astalabs_discovery_all_data.csv'
        if os.path.exists(filename):
            filepath = filename
        elif os.path.exists(f'../{filename}'):
            filepath = f'../{filename}'
        else:
            # If not found, list current directory to help debug, though we must fail eventually
            print("Dataset not found in current or parent directory.")
            return

        print(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)
        
        # Filter for the relevant source table
        df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
        print(f"Total EO 13960 records: {len(df_eo)}")

        # 1. Parse '20_date_implemented' to extract year
        # Standardize date format handling
        df_eo['impl_date'] = pd.to_datetime(df_eo['20_date_implemented'], errors='coerce')
        
        # Drop rows where date is unknown
        df_clean = df_eo.dropna(subset=['impl_date']).copy()
        df_clean['impl_year'] = df_clean['impl_date'].dt.year
        
        # Filter for valid years (e.g., 1990-2025) to remove data entry errors
        df_clean = df_clean[(df_clean['impl_year'] >= 1990) & (df_clean['impl_year'] <= 2025)]
        print(f"Records with valid implementation dates (1990-2025): {len(df_clean)}")

        # 2. Parse '62_disparity_mitigation' into binary
        # Metadata: '62_disparity_mitigation' contains text describing mitigation or 'No', 'N/A', etc.
        def parse_mitigation(val):
            if pd.isna(val):
                return 0
            val_str = str(val).lower().strip()
            if not val_str:
                return 0
            # List of values indicating absence of control
            negatives = ['no', 'n/a', 'none', 'not applicable', '0', 'false', 'unknown', 'tbd', 'not evaluated', 'NaN']
            if val_str in negatives:
                return 0
            # If it contains text not in negatives, assume it describes a mitigation
            return 1

        df_clean['has_mitigation'] = df_clean['62_disparity_mitigation'].apply(parse_mitigation)

        # 3. Split into Pre-2021 and Post-2021 (EO 13960 was late 2020, usually taking effect 2021 for this analysis)
        # The hypothesis specifies "Post-2021", usually meaning >= 2021.
        pre_2021 = df_clean[df_clean['impl_year'] < 2021]
        post_2021 = df_clean[df_clean['impl_year'] >= 2021]

        n_pre = len(pre_2021)
        count_pre = pre_2021['has_mitigation'].sum()
        prop_pre = count_pre / n_pre if n_pre > 0 else 0

        n_post = len(post_2021)
        count_post = post_2021['has_mitigation'].sum()
        prop_post = count_post / n_post if n_post > 0 else 0

        print("\n--- Comparative Analysis (Cutoff: 2021) ---")
        print(f"Pre-2021 Systems (Legacy): n={n_pre}")
        print(f"  Bias Mitigation Compliance: {count_pre} ({prop_pre:.2%})")
        print(f"Post-2021 Systems (Modern): n={n_post}")
        print(f"  Bias Mitigation Compliance: {count_post} ({prop_post:.2%})")

        # 4. Perform Z-test
        if n_pre > 0 and n_post > 0:
            count = np.array([count_pre, count_post])
            nobs = np.array([n_pre, n_post])
            stat, pval = proportions_ztest(count, nobs, alternative='two-sided')
            print(f"\nZ-Test Results:")
            print(f"  Z-statistic: {stat:.4f}")
            print(f"  P-value: {pval:.4e}")
            
            if pval < 0.05:
                print("  Conclusion: Statistically Significant Difference.")
            else:
                print("  Conclusion: No Statistically Significant Difference.")
        else:
            print("\nInsufficient data for Z-test.")

        # 5. Visualization: Compliance Rate over Time
        # Group by year
        yearly_stats = df_clean.groupby('impl_year')['has_mitigation'].agg(['mean', 'count']).reset_index()
        
        # Filter to years with at least a few systems to avoid noisy spikes
        yearly_stats_plot = yearly_stats[yearly_stats['count'] >= 5]

        plt.figure(figsize=(10, 6))
        plt.plot(yearly_stats_plot['impl_year'], yearly_stats_plot['mean'], marker='o', linestyle='-', linewidth=2, label='Mitigation Rate')
        plt.axvline(x=2021, color='r', linestyle='--', label='EO 13960 Era (2021+)')
        
        plt.title('Temporal Adoption of Bias Mitigation Controls')
        plt.xlabel('Year Implemented')
        plt.ylabel('Proportion of Systems with Controls')
        plt.ylim(-0.05, 0.40) # Adjusted Y-limit based on likely low compliance rates to make chart readable
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
Total EO 13960 records: 1757
Records with valid implementation dates (1990-2025): 606

--- Comparative Analysis (Cutoff: 2021) ---
Pre-2021 Systems (Legacy): n=123
  Bias Mitigation Compliance: 16 (13.01%)
Post-2021 Systems (Modern): n=483
  Bias Mitigation Compliance: 55 (11.39%)

Z-Test Results:
  Z-statistic: 0.4990
  P-value: 6.1776e-01
  Conclusion: No Statistically Significant Difference.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Line plot with markers (specifically a time series plot).
*   **Purpose:** The plot visualizes the change in the adoption rate of bias mitigation controls in systems over a period of approximately 9 years. It allows for the observation of trends, volatility, and the potential impact of specific events over time.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Year Implemented"
    *   **Range:** The axis ticks are labeled from 2016 to 2024, but data points appear to span from **2015 to 2024**.
    *   **Units:** Years.
*   **Y-Axis:**
    *   **Title:** "Proportion of Systems with Controls"
    *   **Range:** The visible scale ranges from **-0.05 to 0.40**.
    *   **Units:** Proportion (decimal value representing a percentage, e.g., 0.20 = 20%).

### 3. Data Trends
*   **Initial Volatility and Peak (2015–2017):** The trend begins at a proportion of **0.20 (20%)** in 2015. The line then rises sharply, exceeding the upper limit of the graph (0.40) somewhere around 2017, indicating a period of very high adoption or a potential outlier year where the proportion was significantly higher than the rest of the timeline.
*   **Decline (2018–2020):** By 2018, the proportion drops back down to approximately **0.14**. It continues a downward trend, reaching its global minimum in **2020**, where the proportion falls to roughly **0.03 (3%)**.
*   **Recovery and Stabilization (2021–2024):** Following the 2020 low, there is a sharp rebound in 2021 to approximately **0.15**. The trend then stabilizes somewhat, hovering between **0.10 and 0.15** for the years 2021 through 2024.

### 4. Annotations and Legends
*   **Legend:** Located in the top right corner, identifying:
    *   **"Mitigation Rate" (Blue line with circle markers):** Represents the primary data series.
    *   **"EO 13960 Era (2021+)" (Red dashed line):** Indicates a specific event or time marker.
*   **Vertical Marker:** A red dashed vertical line is placed at the year **2021**. This corresponds to the "EO 13960 Era" mentioned in the legend, likely marking the implementation or effect of Executive Order 13960.
*   **Grid:** A light grid is present to facilitate reading specific x and y values.

### 5. Statistical Insights
*   **Impact of Executive Order 13960:** There is a strong temporal correlation between the start of the "EO 13960 Era" (2021) and a significant recovery in bias mitigation adoption. The rate jumped from a historical low of ~3% in 2020 to ~15% in 2021, suggesting the Executive Order may have driven compliance or awareness.
*   **Inconsistency in Adoption:** The data shows high instability. Early adoption (pre-2018) was erratic and reached highs not seen since. The period leading up to 2020 saw a collapse in mitigation controls, suggesting a lapse in industry standards or focus during that specific window.
*   **Current State:** While the rate has recovered from the 2020 trough, the proportion of systems with controls in 2024 (~11%) remains lower than the initial levels recorded in 2015 (~20%) and significantly lower than the implied peak of 2017. This suggests that despite regulatory intervention, adoption rates have not fully returned to their historical highs.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
