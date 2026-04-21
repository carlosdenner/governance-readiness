# Experiment 288: node_7_18

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_18` |
| **ID in Run** | 288 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T14:48:37.373324+00:00 |
| **Runtime** | 292.8s |
| **Parent** | `node_6_27` |
| **Children** | None |
| **Creation Index** | 289 |

---

## Hypothesis

> Retirement Risk: Retired or Decommissioned AI systems are significantly more
likely to have been 'Rights-Impacting' compared to currently Active systems,
implying that high-risk systems are more prone to cancellation or failure.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.3077 (Maybe False) |
| **Surprise** | -0.5211 |
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
| Maybe False | 20.0 |
| Definitely False | 40.0 |

---

## Experiment Plan

**Objective:** Analyze survivorship bias by comparing active vs. retired systems.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `eo13960_scored`.
- 2. Use `16_dev_stage` or `21_date_retired` to classify systems as 'Retired' (if stage says Retired/Decommissioned or date is present) vs 'Active'.
- 3. Use `17_impact_type` to classify as 'Rights-Impacting' vs 'Other'.
- 4. Generate a contingency table.
- 5. Perform a Chi-Square test or Fisher's Exact Test (if retired counts are low).

### Deliverables
- Contingency table, Rights-Impacting rates by Status (Retired vs Active), and statistical results.

---

## Analysis

The experiment successfully tested the 'Retirement Risk' hypothesis using the EO
13960 dataset. Analyzing 1,757 AI systems, the code classified them into Active
(n=1,613) and Retired (n=144) statuses. Contrary to the hypothesis that high-
risk systems are more prone to cancellation, the data showed that 'Rights-
Impacting' systems constitute a slightly *higher* proportion of the Active
inventory (3.47%, n=56) compared to the Retired inventory (2.08%, n=3).

Statistical testing yielded a Chi-Square p-value of 0.519 and a Fisher's Exact
Test Odds Ratio of 0.59 (p=0.476), indicating that the observed difference is
not statistically significant. The Odds Ratio below 1.0 further underscores that
retired systems were actually less likely to be rights-impacting in this sample,
though the result could simply be due to chance given the low counts.
Consequently, the hypothesis that retired systems are significantly more likely
to have been rights-impacting is **not supported**.

---

## Review

The experiment was successfully executed according to the plan. The code
correctly loaded the EO 13960 dataset, handled the classification of system
status (Active vs. Retired) and impact type (Rights-Impacting vs. Other), and
performed the appropriate statistical tests (Chi-Square and Fisher's Exact
Test). The results were clearly interpreted, and the hypothesis was tested
against the evidence.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load dataset
# Using current directory based on previous context, ignoring the relative path instruction if it failed previously
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    # Fallback if the file is indeed in the parent directory as hinted in the prompt
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 subset shape: {eo_df.shape}")

# Helper function to determine status
def get_status(row):
    stage = str(row.get('16_dev_stage', '')).lower()
    date_ret = row.get('21_date_retired', np.nan)
    
    # Check explicit stage labels
    if 'retired' in stage or 'decommissioned' in stage:
        return 'Retired'
    
    # Check if a retirement date exists
    # If it's a string not equal to 'nan', or a valid number
    if pd.notna(date_ret) and str(date_ret).lower() != 'nan' and str(date_ret).strip() != '':
        return 'Retired'
    
    return 'Active'

# Helper function to determine impact type
def get_impact(row):
    impact = str(row.get('17_impact_type', '')).lower()
    if 'rights' in impact:
        return 'Rights-Impacting'
    return 'Other'

# Apply classifications
eo_df['status'] = eo_df.apply(get_status, axis=1)
eo_df['impact_category'] = eo_df.apply(get_impact, axis=1)

# Generate Contingency Table
contingency = pd.crosstab(eo_df['status'], eo_df['impact_category'])
print("\nContingency Table (Count):")
print(contingency)

# Calculate percentages
rates = pd.crosstab(eo_df['status'], eo_df['impact_category'], normalize='index') * 100
print("\nDistribution Rates (%):")
print(rates)

# Statistical Tests
if contingency.shape == (2, 2):
    # Chi-Square Test
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test: Statistic={chi2:.4f}, p-value={p:.6e}")
    
    # Fisher's Exact Test for Odds Ratio
    # We want to check if Retired are more likely to be Rights-Impacting
    # Table structure usually: [[A, B], [C, D]]
    # Let's align it explicitly:
    #              Rights-Impacting   Other
    # Retired      a                  b
    # Active       c                  d
    
    try:
        a = contingency.loc['Retired', 'Rights-Impacting']
        b = contingency.loc['Retired', 'Other']
        c = contingency.loc['Active', 'Rights-Impacting']
        d = contingency.loc['Active', 'Other']
        
        table_ordered = [[a, b], [c, d]]
        odds_ratio, p_fisher = stats.fisher_exact(table_ordered)
        print(f"Fisher's Exact Test: Odds Ratio={odds_ratio:.4f}, p-value={p_fisher:.6e}")
        print(f"Interpretation: Retired systems are {odds_ratio:.2f}x as likely to be Rights-Impacting as Active systems (in terms of odds).")
    except KeyError:
        print("KeyError: Could not construct 2x2 table properly (missing categories).")
else:
    print("Contingency table is not 2x2. Skipping statistical tests.")

# Visualization
plt.figure(figsize=(10, 6))
# Plotting
ax = rates.plot(kind='bar', stacked=True, color=['#d62728', '#1f77b4'], alpha=0.8)
plt.title('Proportion of Rights-Impacting Systems: Active vs. Retired')
plt.xlabel('System Status')
plt.ylabel('Percentage')
plt.legend(title='Impact Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: EO 13960 subset shape: (1757, 196)

Contingency Table (Count):
impact_category  Other  Rights-Impacting
status                                  
Active            1557                56
Retired            141                 3

Distribution Rates (%):
impact_category      Other  Rights-Impacting
status                                      
Active           96.528208          3.471792
Retired          97.916667          2.083333

Chi-Square Test: Statistic=0.4157, p-value=5.190682e-01
Fisher's Exact Test: Odds Ratio=0.5916, p-value=4.763768e-01
Interpretation: Retired systems are 0.59x as likely to be Rights-Impacting as Active systems (in terms of odds).


=== Plot Analysis (figure 2) ===
Based on the provided plot image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Stacked Bar Chart (specifically a 100% stacked bar chart).
*   **Purpose:** This plot compares the proportional composition of system types ("Other" vs. "Rights-Impacting") across two different statuses ("Active" and "Retired"). It allows the viewer to see the relative percentage of rights-impacting systems within each status category.

**2. Axes**
*   **X-axis:**
    *   **Label:** "System Status".
    *   **Categories:** Two categorical variables: "Active" and "Retired".
*   **Y-axis:**
    *   **Label:** "Percentage".
    *   **Range:** 0 to 100, with tick marks at intervals of 20 (0, 20, 40, 60, 80, 100).
    *   **Units:** Percent (%).

**3. Data Trends**
*   **Dominant Category:** In both the "Active" and "Retired" columns, the "Other" category (represented by the red/coral color) makes up the vast majority of the systems. It appears to occupy approximately 95% to 98% of each bar.
*   **Minority Category:** The "Rights-Impacting" category (represented by the blue color) is a very small minority in both statuses.
*   **Comparison:** Visually, the proportion of "Rights-Impacting" systems appears slightly larger in the "Active" bar compared to the "Retired" bar, though the difference is minimal. The blue sliver is slightly thicker for "Active" systems.

**4. Annotations and Legends**
*   **Title:** "Proportion of Rights-Impacting Systems: Active vs. Retired".
*   **Legend:** Located on the right side with the title "Impact Type".
    *   **Red Square:** Corresponds to "Other".
    *   **Blue Square:** Corresponds to "Rights-Impacting".

**5. Statistical Insights**
*   **Low Prevalence of Rights-Impacting Systems:** The most significant insight is that systems classified as "Rights-Impacting" are rare relative to "Other" systems. They constitute likely less than 5% of the total in both active and retired pools.
*   **Stability Across Lifecycle:** The proportion of rights-impacting systems does not change drastically between active and retired phases. However, the slightly thinner blue section in the "Retired" bar suggests that rights-impacting systems might be retired at a slightly lower rate or make up a smaller portion of historical systems compared to current active ones.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
