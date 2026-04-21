# Experiment 146: node_5_50

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_50` |
| **ID in Run** | 146 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:56:03.309278+00:00 |
| **Runtime** | 194.8s |
| **Parent** | `node_4_28` |
| **Children** | `node_6_26`, `node_6_69` |
| **Creation Index** | 147 |

---

## Hypothesis

> The 'Risk-Control' Mismatch: AI systems self-designated as 'Critical' or 'High
Impact' do NOT show statistically higher rates of 'Independent Evaluation' than
'Low/Moderate' impact systems, indicating a failure to scale governance with
risk.

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

**Objective:** Determine if governance rigor scales appropriately with self-reported risk levels.

### Steps
- 1. Load `eo13960_scored`.
- 2. Clean `17_impact_type`. Group into 'High Risk' (Critical/High) and 'Standard Risk' (Moderate/Low).
- 3. Analyze `55_independent_eval` (Yes/No) rates for both groups.
- 4. Perform a One-sided Z-test to see if High Risk > Standard Risk.
- 5. If p-value > 0.05, the hypothesis (Mismatch/Failure) is supported.

### Deliverables
- Compliance rates by risk tier, Z-test results, Interpretation of risk-proportionate governance.

---

## Analysis

The experiment was successfully executed and provided clear statistical evidence
regarding the 'Risk-Control Mismatch' hypothesis. By correcting the
categorization logic for the `17_impact_type` column, the analysis successfully
segmented the data into 'High Risk' (n=227) and 'Standard Risk' (n=1491) groups.

1.  **Hypothesis Rejection**: The results strongly contradict the hypothesis
that governance does not scale with risk. 'High Risk' systems (impacting rights
or safety) demonstrated a significantly higher rate of 'Independent Evaluation'
(14.5%) compared to 'Standard Risk' systems (1.3%).

2.  **Statistical Significance**: The one-sided Z-test yielded a Z-statistic of
**10.71** and a p-value of **4.48e-27**, conclusively proving that the
difference in governance rigor is statistically significant. This supports a
model of 'Risk-Proportionate Governance' rather than a 'Mismatch'.

3.  **Operational Insight**: While the experiment confirms that agencies are
correctly prioritizing high-risk systems (a 10x higher evaluation rate), the
absolute compliance rate remains low. Even among critical systems, only ~14.5%
have documented independent evaluations, suggesting a resource bottleneck rather
than a lack of prioritization.

---

## Review

The experiment was successfully executed and provided clear statistical evidence
regarding the 'Risk-Control Mismatch' hypothesis. By correcting the
categorization logic for the `17_impact_type` column, the analysis successfully
segmented the data into 'High Risk' (n=227) and 'Standard Risk' (n=1491) groups.

1.  **Hypothesis Rejection**: The results strongly contradict the hypothesis
that governance does not scale with risk. 'High Risk' systems (impacting rights
or safety) demonstrated a significantly higher rate of 'Independent Evaluation'
(14.5%) compared to 'Standard Risk' systems (1.3%).

2.  **Statistical Significance**: The one-sided Z-test yielded a Z-statistic of
**10.71** and a p-value of **4.48e-27**, conclusively proving that the
difference in governance rigor is statistically significant. This supports a
model of 'Risk-Proportionate Governance' rather than a 'Mismatch'.

3.  **Operational Insight**: While the experiment confirms that agencies are
correctly prioritizing high-risk systems (a 10x higher evaluation rate), the
absolute compliance rate remains low. Even among critical systems, only ~14.5%
have documented independent evaluations, suggesting a resource bottleneck rather
than a lack of prioritization.

---

## Code

```python
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt

# Load dataset
# Using the provided relative path based on previous successful executions
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored subset
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Step 1: Inspect and Clean Risk Levels (17_impact_type) ---
# Actual values observed: 'Neither', 'Both', 'Rights-Impacting', 'Safety-impacting', 'Safety-Impacting'

def categorize_risk(val):
    if pd.isna(val):
        return np.nan
    val_clean = str(val).strip()
    # High Risk categories: Rights, Safety, or Both
    if val_clean in ['Both', 'Rights-Impacting', 'Safety-impacting', 'Safety-Impacting']:
        return 'High Risk'
    # Standard Risk categories: Neither
    elif val_clean == 'Neither':
        return 'Standard Risk'
    else:
        return np.nan

eo_df['risk_tier'] = eo_df['17_impact_type'].apply(categorize_risk)

# --- Step 2: Inspect and Clean Control Evidence (55_independent_eval) ---
# Actual values include: 'Yes – by the CAIO', 'Planned or in-progress', 'TRUE', etc.

def parse_eval(val):
    if pd.isna(val):
        return 0
    val_lower = str(val).lower()
    # Strict 'Yes' criteria: must start with 'yes' or be explicitly 'true'
    # 'Planned or in-progress' is typically considered NOT yet fully compliant in strict audits,
    # but we will check strict 'Yes' first. 
    if val_lower.startswith('yes') or val_lower == 'true':
        return 1
    return 0

eo_df['has_eval'] = eo_df['55_independent_eval'].apply(parse_eval)

# Drop rows where risk tier is undefined
analysis_df = eo_df.dropna(subset=['risk_tier'])

# --- Step 3: Calculate Statistics ---
groups = analysis_df.groupby('risk_tier')['has_eval'].agg(['sum', 'count', 'mean'])
groups.columns = ['eval_count', 'total', 'proportion']

print("\n--- Descriptive Statistics ---")
print(groups)

# --- Step 4: Statistical Testing ---
if 'High Risk' in groups.index and 'Standard Risk' in groups.index:
    # Counts of successes (evaluations)
    count = np.array([groups.loc['High Risk', 'eval_count'], groups.loc['Standard Risk', 'eval_count']])
    # Total observations
    nobs = np.array([groups.loc['High Risk', 'total'], groups.loc['Standard Risk', 'total']])
    
    # H0: p_high <= p_standard
    # H1: p_high > p_standard (expecting High Risk to have higher eval rates)
    stat, pval = proportions_ztest(count, nobs, alternative='larger')
    
    print(f"\nZ-test Statistic: {stat:.4f}")
    print(f"P-value (one-sided): {pval:.4e}")
    
    alpha = 0.05
    if pval < alpha:
        print("Result: REJECT Null Hypothesis. High Risk systems DO have statistically higher evaluation rates.")
        print("Interpretation: Governance is scaling with risk (Risk-Proportionate).")
    else:
        print("Result: FAIL TO REJECT Null Hypothesis. High Risk systems do NOT have higher evaluation rates.")
        print("Interpretation: Governance is NOT scaling with risk (Risk-Control Mismatch supported).")

    # --- Step 5: Visualization ---
    plt.figure(figsize=(10, 6))
    bar_colors = ['#d9534f' if idx == 'High Risk' else '#5bc0de' for idx in groups.index]
    bars = plt.bar(groups.index, groups['proportion'], color=bar_colors, alpha=0.8)
    
    plt.title('Independent Evaluation Rates by Impact Type (Risk Tier)')
    plt.ylabel('Proportion with Independent Evaluation')
    plt.ylim(0, max(groups['proportion']) * 1.3)  # Add headroom for text
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, 
                 f'{height:.1%}\n(n={int(groups.loc[groups.index[list(bars).index(bar)], "total"])})', 
                 ha='center', va='bottom')
    
    # Add significance annotation
    sig_text = "Significant" if pval < 0.05 else "Not Significant"
    plt.text(0.5, 0.9, f'p-value: {pval:.4e}\n({sig_text})', 
             transform=plt.gca().transAxes, ha='center', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
             
    plt.tight_layout()
    plt.show()

else:
    print("\nError: Could not identify both High Risk and Standard Risk groups in the data.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: 
--- Descriptive Statistics ---
               eval_count  total  proportion
risk_tier                                   
High Risk              33    227    0.145374
Standard Risk          20   1491    0.013414

Z-test Statistic: 10.7118
P-value (one-sided): 4.4799e-27
Result: REJECT Null Hypothesis. High Risk systems DO have statistically higher evaluation rates.
Interpretation: Governance is scaling with risk (Risk-Proportionate).


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** The plot compares the frequency (proportion) of independent evaluations occurring between two distinct categories of impact types: "High Risk" and "Standard Risk."

### 2. Axes
*   **X-Axis:**
    *   **Labels:** Categorical labels representing the risk tiers: **"High Risk"** and **"Standard Risk"**.
    *   **Value Range:** N/A (Categorical).
*   **Y-Axis:**
    *   **Title:** "Proportion with Independent Evaluation".
    *   **Units:** The values represent a ratio/decimal (0.00 to 1.00), effectively representing percentages.
    *   **Value Range:** The displayed axis ticks range from **0.000 to 0.175** (0% to 17.5%).

### 3. Data Trends
*   **Tallest Bar:** The **"High Risk"** category (colored salmon/red) has a significantly higher proportion of independent evaluations. The bar reaches a height of approximately 0.145.
*   **Shortest Bar:** The **"Standard Risk"** category (colored sky blue) has a very low proportion of independent evaluations, reaching a height of approximately 0.013.
*   **Comparison:** There is a stark contrast between the two groups. The "High Risk" group is subject to independent evaluation at a rate more than 10 times higher than the "Standard Risk" group.

### 4. Annotations and Legends
*   **Bar Annotations:**
    *   **High Risk:** Annotated with **"14.5%"** and a sample size of **"(n=227)"**.
    *   **Standard Risk:** Annotated with **"1.3%"** and a sample size of **"(n=1491)"**.
*   **Statistical Annotation:** A box at the top center displays the result of a statistical hypothesis test:
    *   **"p-value: 4.4799e-27"**
    *   **"(Significant)"**

### 5. Statistical Insights
*   **Statistical Significance:** The p-value ($4.48 \times 10^{-27}$) is extremely close to zero, far below standard alpha levels (e.g., 0.05 or 0.01). This indicates that the difference in evaluation rates between High Risk and Standard Risk groups is **statistically significant** and not due to random chance.
*   **Effect Size:** There is a substantial effect size. High Risk items have a **14.5%** probability of undergoing independent evaluation, compared to only **1.3%** for Standard Risk items.
*   **Sample Size Context:** It is notable that the "Standard Risk" group has a much larger sample size ($n=1491$) compared to the "High Risk" group ($n=227$). Despite the smaller sample size for the High Risk group, the signal is strong enough to yield a highly significant p-value.
*   **Conclusion:** The data strongly suggests that the risk tier is a primary determinant for whether an independent evaluation is conducted, with high-risk items being prioritized for scrutiny.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
